// brae's interFoam on an ADAPTIVE mesh against REAL OpenFOAM's -- unit 8e, the end of the
// dynamicRefineFvMesh port: the whole chain in one run, from the cells the case selects to the fields
// the solver leaves behind.
//
// THE ORACLE is OpenFOAM's own written state after exactly N identical fixed steps of
// laminar/damBreakWithObstacle: the REFINED polyMesh in each time directory, alpha.water, p_rgh, p, U,
// phi, Uf, rAU and cellLevel -- plus its log, solve by solve. A written time directory of a refining
// case carries its own mesh, so the comparison is on the mesh OpenFOAM produced, which is what makes a
// cell-by-cell comparison of the FIELDS mean anything: brae's own refinement is held to OpenFOAM's
// numbering by tests/hex_ref8_vs_openfoam.sh and tests/refine_update_vs_openfoam.sh, and the mesh arms
// here re-assert it so a field difference cannot be a numbering difference in disguise.
//
// WHAT ONLY THIS GATE CAN SEE. refine_update_vs_openfoam holds the driver on an ANALYTIC field, so
// nothing it compares depends on the solver; and the units before it compare one stage at a time. What
// is left is everything the solver does with a changed mesh: the flux the mapper wrote being rebuilt from
// Sf & Uf and corrected by a pcorr solve, the old-time levels that every ddt term reads, the recomputed
// gh/ghf/mixture/curvature, Uf existing at all on a mesh that never moves, and the sizes of the members
// the change invalidates.
//
// tests/interfoam_amr_vs_openfoam.sh says what the arms are, what they measured, and which of them
// damBreakWithObstacle cannot witness.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "patch_entry_lookup.cuh"
#include "inter_solve_log.cuh"
#include "inter_amr_cpp.cuh"
#include "time_instances.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

struct Diff
{
    scalar linf = 0;
    scalar refMax = 0;
    scalar rel() const
    {
        return linf/std::fmax(refMax, scalar(1e-300));
    }
};

Diff compare(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.linf = std::fmax(d.linf, std::fabs(a[i] - b[i]));
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    return d;
}

Diff compare(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.linf = std::fmax(d.linf, mag(a[i] - b[i]));
        d.refMax = std::fmax(d.refMax, mag(b[i]));
    }
    return d;
}

template <class T>
std::vector<T> cellValues(
    const FieldData<T>& fd,
    label nC)
{
    if (fd.internalUniform)
    {
        return std::vector<T>(static_cast<std::size_t>(nC), fd.internalUniformValue);
    }
    return fd.internalField;
}

// OpenFOAM's "Refined from A to B cells." lines, in order -- the count each change ended at. A gate that
// only compared fields could pass on a mesh that refined at a different step.
std::vector<label> readOfRefinedTo(const std::string& logPath)
{
    std::vector<label> out;
    std::ifstream in(logPath);
    std::string line;
    const std::string kTo = " to ";
    while (std::getline(in, line))
    {
        // ...and "Unrefined from N to M cells." (dynamicRefineFvMesh.C:587), because the last change
        // of a run can be an unrefinement and the mesh then ends on ITS count
        if (line.rfind("Refined from ", 0) != 0 && line.rfind("Unrefined from ", 0) != 0) continue;
        const std::size_t a = line.find(kTo);
        if (a == std::string::npos) continue;
        out.push_back(static_cast<label>(std::atol(line.c_str() + a + kTo.size())));
    }
    return out;
}

// OpenFOAM's `- selected N cell(s) with volume V` lines, in order: cellSetOption::setVol prints one
// whenever the selection has been re-run (cellSetOption.C:167-168, forced by V_ = -GREAT at every
// topology change). The LAST one is the selection the run ends with.
struct Selected
{
    label  n = -1;
    scalar V = 0;
};

std::vector<Selected> readOfSelections(const std::string& logPath)
{
    std::vector<Selected> out;
    std::ifstream in(logPath);
    std::string line;
    const std::string kSel = "- selected ", kVol = " cell(s) with volume ";
    while (std::getline(in, line))
    {
        const std::size_t a = line.find(kSel);
        const std::size_t b = line.find(kVol);
        if (a == std::string::npos || b == std::string::npos || b < a) continue;
        Selected s;
        s.n = static_cast<label>(std::atol(line.c_str() + a + kSel.size()));
        s.V = static_cast<scalar>(std::atof(line.c_str() + b + kVol.size()));
        out.push_back(s);
    }
    return out;
}

// OPENFOAM'S OWN RENUMBERED cellZone, which is the sharpest oracle in this whole gate and exists only
// because a topoChanging mesh writes its polyMesh into every time directory -- cellZones with it. So the
// zone a change produced is not inferred from a printed count, as the fvOptions profile has to do: it is
// read back cell by cell from the file OpenFOAM wrote. `name` empty takes the FIRST zone.
std::vector<label> readOfCellZone(const std::string& polyMeshDir, const std::string& name)
{
    std::vector<label> out;
    std::ifstream in(polyMeshDir + "/cellZones");
    if (!in) return out;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // past the FoamFile block, then the named sub-dictionary, then its cellLabels list
    std::size_t i = text.find('}');
    i = (i == std::string::npos) ? 0 : i + 1;
    if (!name.empty())
    {
        i = text.find(name, i);
        if (i == std::string::npos) return out;
    }
    const std::size_t k = text.find("cellLabels", i);
    if (k == std::string::npos) return out;
    const std::size_t lp = text.find('(', k);
    const std::size_t rp = text.find(')', lp);
    if (lp == std::string::npos || rp == std::string::npos) return out;
    std::istringstream body(text.substr(lp + 1, rp - lp - 1));
    label c = 0;
    while (body >> c) out.push_back(c);
    return out;
}

// ...and its continuityErrs.H lines: dt*|div(phi)| weighted-averaged over the cell volumes, which is
// what says the flux the step ENDS with is divergence-free on the new mesh. The last one of the run is
// the one this gate reads, because that is the flux `fieldsOut` hands back.
std::vector<scalar> readOfSumLocalContErr(const std::string& logPath)
{
    std::vector<scalar> out;
    std::ifstream in(logPath);
    std::string line;
    const std::string kS = "sum local = ";
    while (std::getline(in, line))
    {
        if (line.find("time step continuity errors") == std::string::npos) continue;
        const std::size_t a = line.find(kS);
        if (a == std::string::npos) continue;
        out.push_back(static_cast<scalar>(std::atof(line.c_str() + a + kS.size())));
    }
    return out;
}

// continuityErrs.H on a flux of brae's own: sum local = dt*sum(|div(phi)|*V)/sum(V), the signed global
// one beside it. fvc::div(phi) is the face sum over the cell volume, so the V in the weighted average
// cancels it -- the same arithmetic OpenFOAM prints, in the same order of operations.
struct ContErr
{
    scalar sumLocal = 0;
    scalar global = 0;
};

ContErr continuityErrs(
    const SurfaceScalarField& phi,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar deltaT)
{
    std::vector<scalar> div(static_cast<std::size_t>(m.nCells()), scalar(0));
    for (label f = 0; f < m.nInternalFaces(); ++f)
    {
        const std::size_t fi = static_cast<std::size_t>(f);
        div[static_cast<std::size_t>(m.owner()[fi])] += phi.internal[fi];
        div[static_cast<std::size_t>(m.neighbour()[fi])] -= phi.internal[fi];
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        for (label i = 0; i < q.size; ++i)
        {
            div[static_cast<std::size_t>(q.faceCells[static_cast<std::size_t>(i)])] +=
                phi.boundary[pi][static_cast<std::size_t>(i)];
        }
    }
    ContErr e;
    scalar sumV = 0;
    for (label c = 0; c < m.nCells(); ++c)
    {
        const std::size_t ci = static_cast<std::size_t>(c);
        sumV += g.V()[ci];
        e.sumLocal += std::fabs(div[ci]);
        e.global += div[ci];
    }
    e.sumLocal = deltaT*e.sumLocal/sumV;
    e.global = deltaT*e.global/sumV;
    return e;
}

// A polyMesh point list -- `points` or points0MotionSolver's `points0`, which share the format -- as
// OpenFOAM wrote it, at full precision
std::vector<vector> readPointsFile(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    static const std::regex head(R"(\n\s*([0-9]+)\s*\n?\()");
    std::smatch mh;
    std::vector<vector> out;
    if (!std::regex_search(text, mh, head)) return out;
    const std::size_t n = static_cast<std::size_t>(std::atol(mh[1].str().c_str()));
    const char* p = text.c_str() + mh.position(0) + mh.length(0);
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        while (*p && *p != '(')
        {
            ++p;
        }
        if (!*p) break;
        ++p;
        char* end = nullptr;
        vector v;
        v.x = std::strtod(p, &end);
        p = end;
        v.y = std::strtod(p, &end);
        p = end;
        v.z = std::strtod(p, &end);
        p = end;
        out.push_back(v);
    }
    return out;
}

// the largest |a - b| over two point lists, or -1 when their lengths differ
scalar worstPointGap(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    if (a.size() != b.size()) return scalar(-1);
    scalar w = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        w = std::fmax(w, std::fmax(std::fabs(a[i].x - b[i].x),
                                   std::fmax(std::fabs(a[i].y - b[i].y), std::fabs(a[i].z - b[i].z))));
    }
    return w;
}

// One arm: the whole run on a mesh of its own, so a control does not inherit the mesh the last arm
// refined. The caller sets the environment before calling and clears it afterwards.
struct Arm
{
    PrimitiveMesh        m;
    FvGeometry           g;
    std::vector<FvPatch> patches;
    InterFields          f;
    RunReport            r;
};

void runArm(
    Arm&               a,
    const std::string& caseDir,
    const std::string& startDir,
    label              nSteps,
    bool               onDevice = false)
{
    // THE SAME INSTANCE RESOLUTION THE SHIPPED SOLVER DOES, not constant/polyMesh: on every profile but
    // the restart-instance one the two are the same directory, and on that one they are not. A harness that
    // resolved differently from braeInterFoam.cu would be testing a mesh the solver never reads.
    const cpu::timePaths::MeshInstances mi = cpu::timePaths::meshInstancesForStartDir(caseDir, startDir);
    a.m.read(mi.pointsDir(caseDir), mi.facesDir(caseDir), mi.boundaryDir(caseDir));
    a.g.build(a.m);
    a.patches = buildPatches(a.m, a.g);
    MutableMesh mm;
    mm.m = &a.m;
    mm.g = &a.g;
    mm.patches = &a.patches;
    if (onDevice)
    {
        a.r = runInterFoamDevice(caseDir, startDir, a.m, a.g, a.patches, nSteps, /*verbose=*/false,
                                 &a.f, scalar(1.0e300), nullptr, &mm);
        return;
    }
    a.r = runInterFoam(caseDir, startDir, a.m, a.g, a.patches, nSteps, /*verbose=*/false, &a.f,
                       scalar(1.0e300), nullptr, &mm);
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: an ADAPTIVE mesh ==\n");
    if (argc < 7)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <profile>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string profile = argv[6];
    // OPTIONAL: the time directory of a SECOND OpenFOAM run of the same case with ONE VALUE of the initial
    // state perturbed by ONE ULP. It is not a comparison arm -- it is how a bound gets a justification on
    // a case that AMPLIFIES, and this one does: see the mrf block below.
    const std::string ulpDir = (argc > 7) ? argv[7] : std::string();
    std::printf("  profile: %s\n", profile.c_str());

    // THE BOUNDS ARE PER PROFILE AND ARE THE MEASURED AGREEMENT, each a little above what the run reads.
    // They are floating-point floors, not tolerances: this solver reproduces OpenFOAM's own arithmetic on
    // this case, and the closed profile's floor is a little higher than the open one's because a closed
    // domain adds adjustPhi and a whole-field reference shift on top of the same solves.
    //
    //             alpha        p_rgh      U        p        rAU      phi      Uf       contErr
    //   open    1.8848e-15   3.7e-15   2.4e-13  2.3e-14  1.5e-12  2.6e-13  2.5e-13  3.0e-11
    //   closed  1.6209e-14   3.3e-15   2.8e-13  2.0e-14  1.7e-12  1.6e-12  1.9e-13  6.3e-11
    // and the device arm, against OpenFOAM and against the host arm:
    //   open    2.2204e-15 / 2.1176e-15 alpha, 6.1e-15 / 5.2e-15 p_rgh, 3.2e-13 U, 3.3e-13 / 2.6e-13 phi
    //   closed  1.6209e-14 / 3.9968e-15 alpha, 7.2e-15 / 5.8e-15 p_rgh, 3.2e-13 U, 2.6e-13 / 1.4e-12 phi
    struct Bounds
    {
        scalar alpha, pRgh, u, p, rAU, phi, Uf, contErr, devAlpha, devPhi, hostDevAlpha, hostDevPhi;
        // the device arm's own floor on alpha and p_rgh, which is a little above the host's under
        // CrankNicolson: the two arms run the same scheme on the same mapped state and differ only by the
        // order their reductions sum in.
        scalar devPRgh, hostDevPRgh;
        // the PATCH U floor, relative, which is per profile because the relative measure divides by the
        // worst patch's OWN largest value and that is not the field's: on the porosity profile the worst
        // patch's U reaches 1.78e-04, so an absolute 4.0145e-16 -- a smaller absolute than the interior
        // 2.3205e-13 the same run passes on -- reads 2.2560e-12 relative. On the open profile the same
        // patch reaches 1.7e-03 and the same class of round-off reads 8.3203e-14.
        scalar patchU;
        // THE CONTINUITY ERROR AS AN ABSOLUTE FLOOR, where 0 keeps the relative comparison. A relative
        // comparison of two numbers that are both at the CANCELLATION FLOOR measures nothing: with every
        // solve pinned to 1e-14 the mrf profile's pcorr drives sum-local continuity to 5.736483e-18 in
        // brae and 5.687700e-18 in OpenFOAM, whose relative difference is 8.6e-03 and says only that two
        // numbers eighteen orders below the field disagree in their last digits. The statement worth
        // asserting there is the absolute one -- both codes end divergence-free -- so the profile sets
        // this and the relative is printed beside it.
        scalar contErrAbs;
        // the DEVICE arm's U floor, relative, against OpenFOAM and against the host arm. These were a
        // hardcoded 1e-12, which is right for a case at the floating-point floor and wrong for one that
        // amplifies: the mrf profile reads 4.0190e-09 and 5.7751e-09, both INSIDE the envelope OpenFOAM's
        // own one-ulp twin draws (8.6395e-09), and a hardcoded bound cannot say that.
        scalar devU, hostDevU;
        // THE TURBULENCE FIELDS, on the ras profile: k, the second scalar and nut, each relative, and the
        // device's own. Zero on every other profile, where there is no closure to compare.
        scalar k, second, nut, devK, devSecond, devNut;
        // the PATCH p_rgh floor, relative, per profile for the same reason patchU is: it was a hardcoded
        // 1e-14 and the ras profile reads 1.0349e-14 at an ABSOLUTE 2.9104e-11 on a field reaching 2.8e+03,
        // which is round-off rather than a defect.
        scalar patchP;
        // HOW CLOSE TO THE ONE-ULP TWIN brae must be, as a multiple. 1 on the mrf profile, where brae is
        // measurably INSIDE the envelope (0.48x on p_rgh) and `<=` is a true statement worth asserting. 2 on
        // the levels profile, where brae sits marginally ABOVE it -- 1.35x on alpha, 1.37x on p_rgh, 1.29x
        // on U against a twin at 7.7953e-10 / 3.3041e-09 / 2.7944e-08. Both are the same ORDER, and that is
        // the discriminating statement: this gate's own controls sit 1e6 further out, so a defect here is
        // orders and not factors. 0 = the arm is not asserted, for a profile handed no twin.
        scalar ulpFactor;
    };
    //   cn      7.7716e-15 alpha, 7.3e-15 p_rgh, 4.5e-13 U, 2.7e-14 p, 6.9e-12 rAU, 1.2e-13 phi
    //   and the cn profile's DEVICE arm: alpha 2.1982e-14 from OpenFOAM and 2.2714e-14 from the host,
    //   p_rgh 2.0474e-14 and 1.9853e-14, U 8.8e-13, phi 4.3e-13
    const bool closed = (profile == "closed");
    const bool cn = (profile == "cn");
    const bool porosity = (profile == "porosity");
    const bool mrf = (profile == "mrf");
    // the two turbulent profiles of tests/interfoam_amr_ras_vs_openfoam.sh: `ke` as shipped and `sst` the
    // same case made kOmegaSST. They share every bound; what differs is that only kOmegaSST builds a CELL
    // wall distance, which is the half of that unit `ke` cannot witness.
    const bool ras = (profile == "ke" || profile == "sst" || profile == "motorBike");
    // ...AND THE DEVICE LOOP'S kOmegaSST FLOOR, which is not this unit's and is not a loosening. The static
    // RAS gate names it once and holds every SST profile to it (tests/test_inter_ras_dambreak_vs_openfoam.cu
    // :146-151): "nut carries the device loop's U difference through k, omega and F2, so an SST profile sits
    // three orders above a kEpsilon one", with its own `sst` profile reading omega 3.09e-11. This one reads
    // 1.1299e-10 through two topology changes, the same order. The HOST arm is at 1e-14 on both profiles,
    // which is what says the floor is the device closure's and not the carry's.
    const bool sstProfile = (profile == "sst");
    // `motorBike` is RAS/motorBike itself: a snappyHexMesh mesh at levels 0 to 3, its cellLevel and
    // pointLevel BINARY, no refinementHistory (Allrun.pre removes it), 61 patches and kOmegaSST. It is what
    // the levels, instance, RAS and autoMap units were built for, and it needed no further port -- only two
    // narrowings the case itself found: an empty pointZone was refused, and the split-cell count printed the
    // length of a list. Its mesh is produced ONCE by real OpenFOAM's own serial snappyHexMesh, the path its
    // Allrun.pre ships commented out; snappy is not ported and is not going to be.
    const bool motorBikeProfile = (profile == "motorBike");
    // `levels`: the mesh STARTS ALREADY REFINED, from a seed real OpenFOAM produced, so this is the only
    // profile whose FIRST change maps a developed state. Its sharpest arm is not a field at all -- it is
    // `every cell's refinement level is OpenFOAM's`, because the levels are the state the NEXT change reads.
    // `levelsBinary` is the SAME fixture re-encoded by OpenFOAM's own foamFormatConvert, with the
    // refinementHistory removed as motorBike's Allrun.pre removes it. It shares every bound: the decoding is
    // what differs, so any difference between the two profiles IS the decoding.
    // `restart` is the same fixture with the WRITTEN STATE KEPT -- Uf, phi and alphaPhi0 all present -- which
    // the two levels profiles deliberately omit. It shares their bounds and adds the Uf read's own control.
    const bool restartProfile = (profile == "restart");
    // `restartInstance` is a GENUINE restart: nothing is moved, so the refined mesh is in the time directory
    // and constant/polyMesh still holds the one blockMesh made. It shares the levels family's bounds and
    // arms; what it adds is that the mesh, its levels and its zones are found by RESOLVING the instance
    // (cpu::timePaths, mirroring polyMesh.C:175-295) rather than by being put where brae used to look.
    const bool instanceProfile = (profile == "restartInstance");
    const bool levelsProfile = (profile == "levels" || profile == "levelsBinary" || restartProfile
                             || instanceProfile);
    const bool levelsBinary = (profile == "levelsBinary");
    // `box` is laminar/oscillatingBox AS SHIPPED at fixed steps: a mesh that REFINES AND MOVES in every
    // step -- dynamicRefineFvMesh on alpha.water plus a solidBody oscillatingLinearMotion of the whole mesh
    // (tests/interfoam_amr_motion_vs_openfoam.sh). What it adds is the motion's own state through each
    // change: points0 mapped by points0MotionSolver::updateMesh, V0 the change's, the mesh flux recreated
    // and then swept by the move that follows the change.
    // `boxUnrefine` is the same staging run to sixty steps, where OpenFOAM UNREFINES (9540 -> 9400 cells at
    // step 57): points0 through point REMOVAL, the merged cells' V0, and the mapping of a boundary-only
    // carry through merged faces -- which segfaulted until this unit (mapSurfaceField's interpolative branch).
    const bool boxProfile = (profile == "box" || profile == "boxUnrefine");
    //   porosity 1.5190e-14 alpha, 1.2e-14 p_rgh, 3.1e-13 U, 7.5e-15 p, 1.5e-11 rAU, 2.2e-13 phi,
    //   2.3e-13 Uf, 2.5e-11 contErr -- three steps with an explicitPorositySource over a cellZone, whose
    //   re-selection is what this profile exists to measure; and its DEVICE arm: alpha 1.6986e-14 from
    //   OpenFOAM and 2.2773e-14 from the host, p_rgh 1.3737e-14 and 1.8231e-14, U 3.5e-13, phi 2.9e-13
    const Bounds B = closed
        ? Bounds{5e-14, 1e-14, 1e-12, 1e-13, 1e-11, 5e-12, 1e-12, 1e-9, 5e-14, 1e-12, 1e-14, 5e-12,
                 1e-14, 1e-14, 1e-12, 0, 1e-12, 1e-12, 0, 0, 0, 0, 0, 0, 1e-14, 0}
        : cn
        ? Bounds{1e-14, 1e-14, 1e-12, 1e-13, 1e-11, 1e-12, 1e-12, 1e-9, 5e-14, 1e-12, 5e-14, 1e-12,
                 5e-14, 5e-14, 1e-12, 0, 1e-12, 1e-12, 0, 0, 0, 0, 0, 0, 1e-14, 0}
        : porosity
        ? Bounds{5e-14, 5e-14, 1e-12, 1e-13, 5e-11, 1e-12, 1e-12, 1e-9, 5e-14, 1e-12, 5e-14, 1e-12,
                 5e-14, 5e-14, 5e-12, 0, 1e-12, 1e-12, 0, 0, 0, 0, 0, 0, 1e-14, 0}
        // restartInstance HAS ITS OWN FLOOR, and it is the fixture's conditioning rather than a relaxation:
        // the continuation runs at maxRefinement 3 and ends on 18,921 cells, where the levels fixture stays
        // at 4,998 and level 2. MEASURED over its two steps -- alpha 4.3982e-11, p_rgh 8.6504e-08,
        // U 2.3084e-07, p 2.1872e-07, rAU 7.2071e-11, phi 1.9542e-07, Uf 2.9374e-07, patch U 1.4499e-10 and
        // patch p_rgh 3.7771e-08; the DEVICE arm alpha 4.2145e-11, p_rgh 8.6506e-08, U 2.3084e-07,
        // phi 1.9541e-07, and against the host arm alpha 3.1286e-11, p_rgh 9.0766e-11, U 1.4583e-09,
        // phi 2.1037e-10 -- and every one of the 6 p_rgh and 3 pcorr solves takes OpenFOAM's iteration count.
        // WHAT JUSTIFIES IT is the interface one-ulp twin: OpenFOAM against itself moves p_rgh 1.7991e-05 and
        // U 7.6190e-05 here, so brae sits at 0.005x and 0.003x of its own oracle's round-off, and ulpFactor
        // is 1 rather than the levels profile's 2.
        : instanceProfile
        ? Bounds{5e-10, 5e-07, 5e-07, 5e-07, 5e-10, 5e-07, 5e-07, 1e-9, 5e-10, 5e-07, 5e-10, 5e-09,
                 5e-07, 5e-09, 5e-09, 1e-13, 5e-07, 5e-08, 0, 0, 0, 0, 0, 0, 5e-07, 1}
        : levelsProfile
        ? Bounds{5e-09, 5e-08, 5e-07, 5e-08, 5e-07, 5e-07, 5e-07, 1e-9, 5e-09, 5e-07, 5e-09, 5e-07,
                 5e-08, 5e-08, 5e-07, 1e-13, 5e-07, 5e-07, 0, 0, 0, 0, 0, 0, 5e-08, 2}
        // MEASURED over two fixed steps on the serial snappy mesh (8655 cells, refining to 11,693 and then
        // 21,297 -- OpenFOAM's own counts to the cell): alpha 3.0414e-15, p_rgh 9.2875e-15, U 2.2091e-13,
        // p 2.0284e-13, rAU 2.5635e-12, phi 9.1492e-14, Uf 2.7228e-13, patch U 1.0487e-13, patch p_rgh
        // 6.5477e-15, k 8.2588e-15, omega 5.7939e-14, nut 1.3239e-12; both continuity errors 2.58e-15; and
        // all 12 p_rgh and 6 pcorr solves take OpenFOAM's iteration count from OpenFOAM's initial residuals.
        // These are the FLOATING-POINT FLOOR and not an amplified envelope -- Co is 0.026 over two steps --
        // which is why this profile needs no one-ulp twin to justify them and has ulpFactor 0.
        //
        // THE DEVICE ARM'S TURBULENCE IS THE ONE LOOSE SLOT, and it is recorded rather than hidden: k
        // 1.0896e-10, omega 7.4032e-09, nut 6.6811e-08 relative, where the `sst` profile's device closure
        // reads 7.4632e-11 / 1.1299e-10 / 1.0470e-10 on its own fixture -- 66x and 640x on omega and nut.
        // The HOST arm is at 5.7939e-14 and 1.3239e-12 here, so it is the DEVICE closure and not the carry
        // through the change; and it does not reach the solution at two steps, where the device's alpha is
        // 2.8727e-15 and its U 2.5973e-12. Whether that is this case's y+ spread over 60 wall patches or a
        // device gap the other fixture cannot see is its own question, named in PORT.md and not settled here.
        : motorBikeProfile
        ? Bounds{1e-14, 5e-14, 1e-12, 1e-12, 1e-11, 5e-13, 1e-12, 1e-9, 1e-14, 1e-11, 1e-14, 1e-11,
                 5e-14, 5e-14, 5e-13, 1e-13, 1e-11, 1e-11,
                 5e-14, 5e-13, 1e-11, 5e-10, 5e-08, 5e-07, 5e-14, 0}
        : sstProfile
        ? Bounds{5e-14, 1e-13, 1e-12, 1e-13, 1e-12, 5e-12, 1e-12, 1e-9, 5e-14, 5e-10, 5e-14, 5e-10,
                 1e-13, 1e-13, 1e-13, 1e-13, 5e-10, 5e-10,
                 // host k 1.9062e-14, omega 5.1909e-15, nut 1.8203e-14; DEVICE k 7.4632e-11,
                 // omega 1.1299e-10, nut 1.0470e-10, with alpha 5.1070e-15 and p_rgh 6.7935e-14 -- the
                 // device SST floor above, reached through the closure and not through the mapping.
                 1e-13, 1e-13, 1e-13, 5e-10, 5e-10, 5e-10, 1e-13, 0}
        : ras
        ? Bounds{5e-14, 1e-13, 1e-12, 1e-13, 1e-12, 5e-12, 1e-12, 1e-9, 5e-14, 5e-12, 5e-14, 5e-12,
                 1e-13, 1e-13, 1e-13, 1e-13, 5e-12, 5e-12,
                 // k, the second scalar and nut, host then device. MEASURED with every solve pinned:
                 //   host    k 2.2516e-14   epsilon 2.0314e-14   nut 3.7415e-14
                 //   device  k 2.1519e-13   epsilon 6.1862e-13   nut 9.1907e-14, and alpha 2.7756e-14,
                 //           p_rgh 2.2903e-14, U 1.7083e-12, phi 2.7135e-12 from OpenFOAM
                 // The device's floor is an order above the host's on the two transported scalars, which is
                 // the device closure's own distance on this case and not the change's: the static RAS gate
                 // reads the same shape (tests/interfoam_ras_dambreak_vs_openfoam.sh).
                 // The continuity floor is ABSOLUTE (1e-13): both codes end at 2.44e-15, where the relative
                 // difference is 9.5e-05 and measures their last digits.
                 1e-13, 1e-13, 1e-13, 1e-12, 1e-12, 1e-12, 1e-13, 0}
        // MEASURED over twenty fixed steps with every solve pinned (1000 -> 8700 cells, the mesh moving at
        // every step): alpha 6.9944e-15, p_rgh 1.2620e-14, U 4.7340e-14, p 1.4906e-14, rAU 8.4167e-13,
        // phi 5.6936e-15, Uf 3.2490e-14, patch U 6.4190e-13, patch p_rgh 1.2620e-14; points, points0 and
        // the mesh flux exactly OpenFOAM's; all 60 p_rgh and 21 pcorr solves at OpenFOAM's iteration counts
        // from its initial residuals. The continuity floor is ABSOLUTE: both codes end at the cancellation
        // floor (brae 1.10e-15, OpenFOAM 8.64e-16). The case does not amplify -- one ulp on one alpha cell
        // moves OpenFOAM's own alpha 5.7e-15 by step 20 -- so these are round-off floors and need no twin.
        // The device slots are the host's until the device arm runs the case (it refuses it by name).
        : boxProfile
        ? Bounds{1e-14, 5e-14, 1e-13, 5e-14, 5e-12, 5e-14, 1e-13, 1e-9, 1e-14, 5e-14, 1e-14, 5e-14,
                 5e-14, 5e-14, 5e-12, 1e-14, 1e-13, 1e-13, 0, 0, 0, 0, 0, 0, 5e-14, 0}
        : mrf
        ? Bounds{1e-10, 5e-09, 1e-08, 5e-09, 5e-10, 1e-09, 1e-08, 0, 1e-10, 1e-09, 1e-10, 1e-09,
                 1e-08, 1e-08, 1e-12, 1e-15, 1e-08, 1e-08, 0, 0, 0, 0, 0, 0, 1e-14, 1}
        : Bounds{5e-15, 1e-14, 1e-12, 1e-13, 1e-11, 1e-12, 1e-12, 1e-9, 5e-15, 1e-12, 5e-15, 1e-12,
                 1e-14, 1e-14, 1e-12, 0, 1e-12, 1e-12, 0, 0, 0, 0, 0, 0, 1e-14, 0};

    Arm A;
    runArm(A, caseDir, startDir, nSteps);
    const label nC = A.m.nCells();
    const label nIF = A.m.nInternalFaces();
    std::printf("  brae ran %ld steps to t = %.10g on %ld cells (%ld internal faces)\n",
                (long)A.r.steps, (double)A.r.time, (long)nC, (long)nIF);
    check("the run took every step the oracle did", A.r.steps == nSteps);
    check("the case was recognised as adaptive", A.f.amr && A.f.amr->active);
    // createUfIfPresent.H gates Uf on mesh.dynamic(), which a REFINING mesh is: OpenFOAM's own run of
    // this case writes a Uf beside every time directory, and brae built none until this unit.
    check("the mesh is dynamic, so Uf exists", A.f.meshIsDynamic && !A.f.Uf.internal.empty());

    // ---- THE MESH, first: a field comparison is only a field comparison if the cells are the same
    //
    // THE ORACLE'S MESH IS RESOLVED, NOT ASSUMED, and that is not a convenience -- OpenFOAM does not write a
    // full polyMesh into every time directory. On the restartInstance fixture the continuation changes no
    // topology, so 0.003 and 0.004 carry cellLevel, pointLevel, level0Edge and refinementHistory and NO
    // faces or points; the mesh they are on is 0.002's. Reading `<ofDir>/polyMesh` blind threw
    // "cannot open .../0.004/polyMesh/points" there, which is the same instance resolution this unit is
    // about, applied to the oracle side.
    PrimitiveMesh ofM;
    {
        std::string ofCase = ofDir;
        const std::size_t slash = ofCase.find_last_of('/');
        if (slash != std::string::npos) ofCase = ofCase.substr(0, slash);
        const cpu::timePaths::MeshInstances om =
            cpu::timePaths::meshInstancesForStartDir(ofCase, ofDir);
        ofM.read(om.pointsDir(ofCase), om.facesDir(ofCase), om.boundaryDir(ofCase));
    }
    std::printf("  OpenFOAM's %s: %ld cells, %ld faces (%ld internal), %ld points\n", ofDir.c_str(),
                (long)ofM.nCells(), (long)ofM.nFaces(), (long)ofM.nInternalFaces(), (long)ofM.nPoints());
    check("the same cell count", ofM.nCells() == nC);
    check("the same face count", ofM.nFaces() == A.m.nFaces());
    check("the same internal face count", ofM.nInternalFaces() == nIF);
    check("the same point count", ofM.nPoints() == A.m.nPoints());
    if (ofM.nFaces() == A.m.nFaces() && ofM.nCells() == nC)
    {
        bool own = true, nei = true;
        for (label f = 0; f < A.m.nFaces(); ++f)
        {
            own = own && (ofM.owner()[static_cast<std::size_t>(f)] == A.m.owner()[static_cast<std::size_t>(f)]);
        }
        for (label f = 0; f < nIF; ++f)
        {
            nei = nei
               && (ofM.neighbour()[static_cast<std::size_t>(f)] == A.m.neighbour()[static_cast<std::size_t>(f)]);
        }
        check("every face's owner is OpenFOAM's", own);
        check("every internal face's neighbour is OpenFOAM's", nei);
    }
    // the refinement LEVEL, which is the state the next change reads and the one field on disk that is
    // the refiner's own rather than the solver's
    {
        const FieldData<scalar> lv = readField<scalar>(ofDir + "/cellLevel");
        const std::vector<scalar> ofLevel = cellValues(lv, nC);
        bool same = ofLevel.size() == static_cast<std::size_t>(nC)
                 && A.f.amr->state.levels.cellLevel.size() == static_cast<std::size_t>(nC);
        for (std::size_t c = 0; same && c < ofLevel.size(); ++c)
        {
            same = (static_cast<label>(ofLevel[c] + scalar(0.5)) == A.f.amr->state.levels.cellLevel[c]);
        }
        check("every cell's refinement level is OpenFOAM's", same);
    }
    // ...and that the changes happened at the same steps, on the same counts
    {
        const std::vector<label> ofTo = readOfRefinedTo(logPath);
        std::printf("  OpenFOAM changed the mesh %zu times, last to %ld cells\n", ofTo.size(),
                    ofTo.empty() ? 0L : (long)ofTo.back());
        check("OpenFOAM's log shows a refinement at all (the fixture can witness one)", !ofTo.empty());
        check("brae's last change ended on OpenFOAM's cell count",
              !ofTo.empty() && ofTo.back() == nC);
    }

    // ---- THE FIELDS
    const FieldData<scalar> alFd = readField<scalar>(ofDir + "/" + A.f.alphaName);
    const FieldData<scalar> prFd = readField<scalar>(ofDir + "/p_rgh");
    const FieldData<vector> uFd = readField<vector>(ofDir + "/U");
    const std::vector<scalar> ofAlpha = cellValues(alFd, nC);
    const std::vector<scalar> ofPrgh = cellValues(prFd, nC);
    const std::vector<vector> ofU = cellValues(uFd, nC);
    const std::vector<scalar> ofP = cellValues(readField<scalar>(ofDir + "/p"), nC);
    const std::vector<scalar> ofRAU = cellValues(readField<scalar>(ofDir + "/rAU"), nC);
    const FieldData<scalar> phiFd = readField<scalar>(ofDir + "/phi");
    const FieldData<vector> ufFd = readField<vector>(ofDir + "/Uf");
    const std::vector<scalar> ofPhi = cellValues(phiFd, nIF);
    const std::vector<vector> ofUf = cellValues(ufFd, nIF);

    const Diff dAlpha = compare(A.f.alpha1.internal, ofAlpha);
    const Diff dPrgh = compare(A.f.p_rgh.internal, ofPrgh);
    const Diff dU = compare(A.f.U.internal, ofU);
    const Diff dP = compare(A.f.p, ofP);
    const Diff dRAU = compare(A.f.rAU, ofRAU);
    const Diff dPhi = compare(A.f.phi.internal, ofPhi);
    const Diff dUf = compare(A.f.Uf.internal, ofUf);
    std::printf("  alpha  %.4e (rel %.4e)   p_rgh %.4e (rel %.4e)   U %.4e (rel %.4e)\n",
                (double)dAlpha.linf, (double)dAlpha.rel(), (double)dPrgh.linf, (double)dPrgh.rel(),
                (double)dU.linf, (double)dU.rel());
    std::printf("  p      %.4e (rel %.4e)   rAU   %.4e (rel %.4e)   phi %.4e (rel %.4e)   Uf %.4e (rel %.4e)\n",
                (double)dP.linf, (double)dP.rel(), (double)dRAU.linf, (double)dRAU.rel(),
                (double)dPhi.linf, (double)dPhi.rel(), (double)dUf.linf, (double)dUf.rel());

    // ---- THE MOTION THROUGH THE CHANGES, on the box profile: the points, points0 and the mesh flux as
    // OpenFOAM WROTE them into the last time directory (points0 is AUTO_WRITE at the time of the last
    // change, points0MotionSolver.C:213-217; meshPhi by fvMesh::writeObject). points and points0 are held
    // BITWISE: points = transform(points0) is exact arithmetic on the same inputs, and the added points'
    // points0 differ from the unmoved lattice by at most an ulp (measured on OpenFOAM's own files: 4122 of
    // 10255 at step two, by up to 4.4e-16), so only a bitwise arm can see that formula at all.
    if (boxProfile)
    {
        const DynamicMotionSolverFvMesh* dm = A.f.dynamicMesh.get();
        check("the refining mesh also MOVES: its motion was built as a list member",
              dm != nullptr && dm->listForm());
        if (dm)
        {
            const std::vector<vector> ofPts = readPointsFile(ofDir + "/polyMesh/points");
            const std::vector<vector> ofPts0 = readPointsFile(ofDir + "/polyMesh/points0");
            const scalar gPts = worstPointGap(A.m.points(), ofPts);
            const scalar gPts0 = worstPointGap(dm->points0(), ofPts0);
            const std::vector<scalar> ofMeshPhi = cellValues(readField<scalar>(ofDir + "/meshPhi"), nIF);
            const Diff dMeshPhi = compare(dm->meshPhi().internal, ofMeshPhi);
            std::printf("  motion: points %zu (OpenFOAM %zu) worst %.4e   points0 %zu (OpenFOAM %zu) worst %.4e   "
                        "meshPhi %.4e (rel %.4e)\n",
                        A.m.points().size(), ofPts.size(), (double)gPts, dm->points0().size(), ofPts0.size(),
                        (double)gPts0, (double)dMeshPhi.linf, (double)dMeshPhi.rel());
            check("the moved points are OpenFOAM's, to the bit", gPts == scalar(0));
            check("...and so is points0, carried through every change", gPts0 == scalar(0));
            check("the mesh flux is OpenFOAM's", dMeshPhi.rel() < B.phi);
        }
    }

    // THE BOUNDS ARE THE MEASURED AGREEMENT, each a little above what the run actually reads, and
    // interfoam_amr_vs_openfoam.sh records those numbers. They are floating-point floors and not
    // tolerances: this solver reproduces OpenFOAM's own arithmetic on this case, so alpha lands at
    // 1.9e-15 on 82,264 cells and every one of the nine solves matches OpenFOAM's residual to the bit.
    // THE restart PROFILE'S FIELDS ARE NOW ASSERTED, on the levels profile's own bounds and with nothing
    // widened, and what changed is the CONTROL rather than the code.
    //
    // It used to read 20x to 32x OpenFOAM's own one-ulp twin -- alpha 5.6049e-10, p_rgh 2.7323e-09,
    // U 4.2657e-08 against a twin at about 2.8e-11 / 8.5e-11 / 1.7e-09 -- and the difference was recorded
    // here as not localised. IT WAS THE TWIN. That twin perturbed the first cell whose alpha is exactly 1,
    // which on a developed damBreak is deep in the water: every neighbour is 1 as well, and the curvature is
    // a function of alpha's GRADIENT, so the perturbation never reaches it. Measured with
    // tools/dumpInterFoam on both runs: it moves 12 of 12,487 faces of the surface tension force, by
    // 3.5e-21, and its p_rgh after three steps moves 8.5e-11 -- under the 1e-10 the amplification arm below
    // asks for. tests/interfoam_amr_ulp_cell.py now perturbs a cell AT THE INTERFACE -- the owner of the
    // first internal face whose alpha differs either side, cell 2268 on this fixture -- and against that
    // twin the gate reads, with nothing else changed:
    //     levels        0.98x alpha   0.85x p_rgh   1.03x U   (twin 1.0750e-09 / 5.3029e-09 / 3.5029e-08)
    //     restart       0.86x alpha   0.95x p_rgh   0.78x U   (twin 6.5006e-10 / 2.8660e-09 / 5.4635e-08)
    // The 20x was the control's and not the port's.
    //
    // THE STAGE TRACE IS WHAT SETTLED IT, at the first change and the first corrector, brae against
    // OpenFOAM's own dumps (BRAE_STAGE_DUMP_DIR against tools/dumpInterFoam's BRAE_DUMP_ITER):
    //     Uf as mapped, phi = Sf & Uf, alpha at calculateK, and the post-refinement Sf, C, V, weights and
    //     deltaCoeffs are ALL BIT-IDENTICAL; pcorr's phi 1.1199e-15; UEqn's source 2.7943e-16 and its
    //     diagonal 3.7501e-16; rAU 1.9391e-13; HbyA 2.3941e-15; phiHbyA 1.9645e-15.
    // The three operations this unit set out to separate -- the Sf & Uf rebuild, the pcorr solve and
    // correctUf -- are therefore not where the difference is born; the interface chain (nHatf 1.42x the
    // interface twin, sigmaK 0.07x, stf 2.88x) is where it lives, and it is round-off there.
    {
        check("alpha is at this profile's floor from OpenFOAM's", dAlpha.linf < B.alpha);
        check("p_rgh is at this profile's floor, relative", dPrgh.rel() < B.pRgh);
        check("U is at this profile's floor, relative", dU.rel() < B.u);
        check("p is at this profile's floor, relative", dP.rel() < B.p);
        check("rAU is at this profile's floor, relative", dRAU.rel() < B.rAU);
        check("phi is at this profile's floor, relative", dPhi.rel() < B.phi);
        check("Uf is at this profile's floor, relative", dUf.rel() < B.Uf);
    }

    // HOW DIFFUSE the Uf difference is, for the levels/restart profiles. It reports the worst face and how
    // many faces are within a hundredth of it, and it does NOT claim to say whether those faces are ones the
    // change ADDED: a face index would say that for CELLS, where hexRef8 modifies the parent in place and
    // appends the seven children, but polyTopoChange renumbers FACES wholesale, so a low face index means
    // only "early in the new numbering". That distinction was worth writing down because the first reading
    // of this line drew the wrong conclusion from it. What it does establish is the SHAPE: on the restart
    // profile 1,611 of 12,487 faces sit within a hundredth of the worst, so the difference is spread across
    // the mesh rather than sitting on one site -- which is what rules out a single mis-mapped face and
    // points at the surface-field mapping as a whole. On a first change from REST every surface field is zero
    // and the hull average is exactly 0 in both codes, which is why no other profile can see this at all.
    if (levelsProfile)
    {
        const auto worstAt = [](const std::vector<vector>& mine, const std::vector<vector>& of)
        {
            std::size_t at = 0;
            scalar worst = 0;
            for (std::size_t i = 0; i < mine.size() && i < of.size(); ++i)
            {
                const scalar d = std::fmax(std::fmax(std::fabs(mine[i].x - of[i].x),
                                                     std::fabs(mine[i].y - of[i].y)),
                                           std::fabs(mine[i].z - of[i].z));
                if (d > worst) { worst = d; at = i; }
            }
            return std::pair<std::size_t, scalar>{at, worst};
        };
        const auto wUf = worstAt(A.f.Uf.internal, ofUf);
        std::size_t above = 0;
        for (std::size_t i = 0; i < A.f.Uf.internal.size() && i < ofUf.size(); ++i)
        {
            const scalar d = std::fmax(std::fmax(std::fabs(A.f.Uf.internal[i].x - ofUf[i].x),
                                                 std::fabs(A.f.Uf.internal[i].y - ofUf[i].y)),
                                       std::fabs(A.f.Uf.internal[i].z - ofUf[i].z));
            if (d > scalar(0.01)*wUf.second) ++above;
        }
        std::printf("  Uf's difference is spread over %zu of %ld internal faces (within a hundredth of the "
                    "worst, which is face %zu -- an index in the NEW numbering, not a claim about which "
                    "faces the change added)\n", above, (long)nIF, wUf.first);
    }

    // ...and the PATCH values, which are the patch fields' own autoMap: a mapped patch field whose
    // unmapped faces were left at zero reads exactly here and nowhere else.
    {
        Diff worstA, worstU, worstP;
        for (std::size_t pi = 0; pi < A.patches.size(); ++pi)
        {
            if (A.patches[pi].type == "empty" || A.patches[pi].size == 0) continue;
            const PatchFieldData<scalar>* ba = findPatchEntry(alFd, A.patches[pi]);
            const PatchFieldData<vector>* bu = findPatchEntry(uFd, A.patches[pi]);
            const PatchFieldData<scalar>* bp = findPatchEntry(prFd, A.patches[pi]);
            const std::size_t n = static_cast<std::size_t>(A.patches[pi].size);
            if (ba && !ba->valueUniform && ba->values.size() == n)
            {
                const Diff d = compare(A.f.alpha1.boundary[pi]->value(), ba->values);
                if (d.linf > worstA.linf) worstA = d;
            }
            if (bu && !bu->valueUniform && bu->values.size() == n)
            {
                const Diff d = compare(A.f.U.boundary[pi]->value(), bu->values);
                if (d.linf > worstU.linf) worstU = d;
            }
            if (bp && !bp->valueUniform && bp->values.size() == n)
            {
                const Diff d = compare(A.f.p_rgh.boundary[pi]->value(), bp->values);
                if (d.linf > worstP.linf) worstP = d;
            }
        }
        std::printf("  patch values: alpha %.4e   U %.4e (rel %.4e)   p_rgh %.4e (rel %.4e)\n",
                    (double)worstA.linf, (double)worstU.linf, (double)worstU.rel(),
                    (double)worstP.linf, (double)worstP.rel());
        check("every patch's alpha is at this profile's floor", worstA.linf < B.alpha);
        check("every patch's U is at this profile's floor, relative", worstU.rel() < B.patchU);
        check("every patch's p_rgh is at this profile's floor, relative", worstP.rel() < B.patchP);
    }

    // ---- THE SOLVES, which is the arm that says the two codes solved the same systems. A pcorr solve
    // exists only because the mesh changed, and its iteration count is the sharpest statement this log
    // can make about the flux it was handed.
    const std::vector<LinearSolveRecord> ofP_rgh = gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofPcorr = gatecheck::readOfSolves(logPath, "pcorr");
    std::printf("  OpenFOAM: %zu p_rgh solves, %zu pcorr solves; brae %zu and %zu\n",
                ofP_rgh.size(), ofPcorr.size(), A.r.pSolves.size(), A.r.pcorrSolves.size());
    if (levelsProfile)
    {
        // ON THIS PROFILE THE COUNT IS MEASURED AND NOT ASSERTED EQUAL, for a reason established by
        // measurement rather than by tolerance. The gate reads 8 of 9 equal, the odd one 114/115, and
        // OpenFOAM's solve 3 ends at 9.267e-15 against a pinned tolerance of 1e-14 -- SEVEN PER CENT under
        // it. A solve stopping that close to its tolerance is a threshold crossing, and a difference of
        // either sign flips it. What says this is not brae's is the ONE-ULP TWIN: OpenFOAM against itself at
        // one ulp produces 121 118 114 118 113 110 117 108 104, IDENTICAL to the oracle's -- so the
        // threshold is not flipped by one ulp, and brae's own distance (1.37x that twin, see the ulp arm
        // below) is what flips it. compareSolves' `tolerance` parameter is not used because its rule wants
        // the shorter run's residual within a THOUSANDTH of the tolerance and this is within a fourteenth;
        // widening that rule would weaken it for every other caller.
        //
        // WHAT IS STILL ASSERTED: the solve COUNT (how many solves ran), the initial and final residuals,
        // and that at most one iteration count differs and by at most one. A second differing count, or one
        // differing by two, is a different solve and fails here.
        const int before = failures;
        (void)before;
        gatecheck::compareSolves("p_rgh", A.r.pSolves, ofP_rgh, nSteps, "p_rgh", scalar(1e-10), scalar(1e-5),
                                 scalar(-1), nullptr, /*assertArms=*/false);
        check("it ran as many p_rgh solves as OpenFOAM logged", A.r.pSolves.size() == ofP_rgh.size());
        if (A.r.pSolves.size() == ofP_rgh.size())
        {
            int differing = 0, worstGap = 0;
            for (std::size_t i = 0; i < ofP_rgh.size(); ++i)
            {
                const int gap = std::abs(static_cast<int>(A.r.pSolves[i].nIterations)
                                         - static_cast<int>(ofP_rgh[i].nIterations));
                if (gap) { ++differing; worstGap = (gap > worstGap) ? gap : worstGap; }
            }
            std::printf("  p_rgh iteration counts: %d of %zu differ, by at most %d\n",
                        differing, ofP_rgh.size(), worstGap);
            check("at most ONE p_rgh count differs from OpenFOAM's, and by at most one iteration -- a solve "
                  "stopping 7% under its tolerance, not a different solve",
                  differing <= 1 && worstGap <= 1);
        }
    }
    else
    {
        failures += gatecheck::compareSolves("p_rgh", A.r.pSolves, ofP_rgh, nSteps, "p_rgh");
    }
    {
        // THREE pcorr solves, and the first is initCorrectPhi.H's before the time loop -- OpenFOAM prints
        // it too (0 iterations on this case, its start flux already being divergence-free). The
        // alignment is from the END so that a log with fewer is still compared against the solves it
        // does carry rather than against another step's.
        std::vector<LinearSolveRecord> mine = A.r.pcorrSolves;
        while (mine.size() > ofPcorr.size()) mine.erase(mine.begin());
        check("brae solved pcorr as often as OpenFOAM did", A.r.pcorrSolves.size() == ofPcorr.size());
        failures += gatecheck::compareSolves("pcorr", mine, ofPcorr, nSteps, "pcorr");
    }

    // ---- THE CONTINUITY ERROR of the flux the run ends with, against OpenFOAM's own last line
    {
        const std::vector<scalar> ofCe = readOfSumLocalContErr(logPath);
        const ContErr ce = continuityErrs(A.f.phi, A.m, A.g, A.patches, A.r.deltaT);
        std::printf("  continuity: brae sum local %.6e (global %.6e), OpenFOAM's last %.6e\n",
                    (double)ce.sumLocal, (double)ce.global, ofCe.empty() ? 0.0 : (double)ofCe.back());
        check("OpenFOAM's log carries a continuity error to compare against", !ofCe.empty());
        if (!ofCe.empty())
        {
            const scalar rel = std::fabs(ce.sumLocal - ofCe.back())/std::fmax(ofCe.back(), scalar(1e-300));
            std::printf("  ...relative difference %.4e\n", (double)rel);
            if (B.contErrAbs > scalar(0))
            {
                check("BOTH codes end divergence-free to this profile's absolute floor, which is the only "
                      "statement a relative comparison of two cancellation-floor numbers could not make",
                      std::fabs(ce.sumLocal) < B.contErrAbs && std::fabs(ofCe.back()) < B.contErrAbs);
            }
            else
            {
                check("the final flux carries OpenFOAM's own continuity error to 1e-9 relative",
                      rel < B.contErr);
            }
        }
    }

    // ---- THE CONTROLS. Each makes one half of this unit wrong on purpose and must be caught by the
    // bounds above; a gate whose control passes is measuring nothing.
    {
        setenv("BRAE_CONTROL_AMR_NO_CORRECTPHI", "1", 1);
        Arm B;
        runArm(B, caseDir, startDir, nSteps);
        unsetenv("BRAE_CONTROL_AMR_NO_CORRECTPHI");
        const Diff cA = compare(B.f.alpha1.internal, ofAlpha);
        const Diff cU = compare(B.f.U.internal, ofU);
        std::printf("  CONTROL (the mapped flux left alone, no Sf & Uf rebuild and no pcorr): alpha %.4e, "
                    "U rel %.4e, max(alpha) %.10g against OpenFOAM's 1\n",
                    (double)cA.linf, (double)cU.rel(), (double)B.r.alphaMax);
        check("...the control ran every step", B.r.steps == nSteps);
        check("...and is caught: its alpha is more than a million times further out than the gate's",
              cA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
        check("...and it overshoots alpha = 1 where the ported path does not",
              B.r.alphaMax > scalar(1) + scalar(1e-4) && A.r.alphaMax < scalar(1) + scalar(1e-6));
    }
    {
        setenv("BRAE_CONTROL_AMR_RESIZE_NOT_MAP", "1", 1);
        Arm C;
        // A DELIBERATELY WRONG RUN THAT CANNOT FINISH IS CAUGHT, not a harness crash: on boxUnrefine the
        // resized state reaches an asymmetric p_rgh matrix by the time the mesh unrefines, which the GAMG
        // port refuses by name
        std::string controlThrew;
        try
        {
            runArm(C, caseDir, startDir, nSteps);
        }
        catch (const std::exception& e)
        {
            controlThrew = e.what();
        }
        unsetenv("BRAE_CONTROL_AMR_RESIZE_NOT_MAP");
        if (!controlThrew.empty())
        {
            std::printf("  CONTROL (the cell fields resized, not mapped): stopped -- %s\n",
                        controlThrew.substr(0, 100).c_str());
            check("...and is caught: the deliberately wrong run does not reach the end", true);
        }
        else
        {
            const Diff cA = compare(C.f.alpha1.internal, ofAlpha);
            const Diff cU = compare(C.f.U.internal, ofU);
            std::printf("  CONTROL (the cell fields resized, not mapped): alpha %.4e, U rel %.4e\n",
                        (double)cA.linf, (double)cU.rel());
            check("...the control ran every step", C.r.steps == nSteps);
            if (motorBikeProfile)
            {
                // ON THIS FIXTURE ALPHA CANNOT WITNESS IT, which is measured and not assumed: the cells this
                // case refines are deep in the air, where alpha is 0, so RESIZING a cell field zero-fills them
                // with the value mapping would have given -- the control reads alpha 2.9724e-11 against the
                // gate's 3.0414e-15, under the million. U is where it shows, at 3.2311e-02 against 2.2091e-13,
                // eleven orders. The arm asserts on the field that can see it rather than on the field the
                // other profiles use.
                check("...and is caught on U: 3.2e-02 against the gate's 2.2e-13, where alpha cannot see it "
                      "because this case refines into air",
                      cU.rel() > scalar(1e6)*std::fmax(dU.rel(), scalar(1e-300)));
            }
            else
            {
                check("...and is caught: its alpha is more than a million times further out than the gate's",
                      cA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
            }
        }
    }
    // ...AND THE ONE THAT CANNOT WITNESS, measured rather than assumed. At the top of a step every
    // old-time level is a copy of its own field -- OpenFOAM's storeOldTimes runs on the step's first
    // access and brae's loop assigns them at the end of the previous step -- and the change happens
    // before anything has solved. So mapping alpha.oldTime() and re-capturing it from the mapped alpha
    // are THE SAME NUMBERS, and this arm asserts they are the same to the BIT rather than quietly
    // standing in for a test of the old-time mapping. A case with `moveMeshOuterCorrectors yes`, where
    // the change happens after a corrector has moved U, is what would part them.
    {
        setenv("BRAE_CONTROL_AMR_NO_OLDTIME_MAP", "1", 1);
        Arm D;
        runArm(D, caseDir, startDir, nSteps);
        unsetenv("BRAE_CONTROL_AMR_NO_OLDTIME_MAP");
        const Diff sameA = compare(D.f.alpha1.internal, A.f.alpha1.internal);
        const Diff sameU = compare(D.f.U.internal, A.f.U.internal);
        std::printf("  the old-time levels re-captured instead of mapped: alpha %.4e and U %.4e from the "
                    "gate's own arm\n", (double)sameA.linf, (double)sameU.linf);
        if (cn)
        {
            // UNDER CrankNicolson IT IS A CONTROL. The scheme reads the old-OLD levels -- U.oldTime()
            // .oldTime() and phi's -- and those are NOT copies of the current fields at the top of a step:
            // they are two steps back. So the same switch that changes nothing under Euler moves the
            // answer here, which is the sharpest thing this profile says about the old-time mapping.
            check("...and under CrankNicolson it is CAUGHT: the old-old levels are not copies of the "
                  "current fields", sameA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
        }
        else
        {
            // ...AND UNDER EULER IT CANNOT WITNESS, measured: at the top of a step every old-time level is
            // a copy of its own field (OpenFOAM's storeOldTimes runs on the step's first access; brae's
            // loop assigns them at the end of the previous step) and the change happens before anything
            // has solved. The arm is kept because the identity is the statement.
            check("...re-capturing the old-time levels changes NOTHING under Euler, so this profile does "
                  "not test their mapping", sameA.linf == scalar(0) && sameU.linf == scalar(0));
        }
    }

    // ---- THE PRESSURE REFERENCE, on the closed profile. OpenFOAM locates it ONCE in createFields
    // (createFields.H:104-113) and NEVER renumbers or re-locates it; pRefCell is a plain label that
    // every pEqn indexes. So after a refinement OpenFOAM pins the SAME INDEX -- a different cell on the
    // refined mesh -- and brae must do the same. The oracle is exact: pEqn.H adds
    // (pRefValue - p[pRefCell]) to the whole field, so OpenFOAM's own written p at that index IS
    // pRefValue, and finding the same pin means finding the same cell.
    if (profile == "closed")
    {
        check("the closed case needs a pressure reference", A.f.pRef.needReference);
        check("...and it was located, on the mesh the run STARTED on", A.f.pRef.pRefCell >= 0);
        const std::size_t rc = static_cast<std::size_t>(A.f.pRef.pRefCell);
        if (A.f.pRef.needReference && rc < ofP.size() && rc < A.f.p.size())
        {
            std::printf("  the reference is cell %ld of %ld; p there: OpenFOAM %.6e, brae %.6e "
                        "(pRefValue %.6e)\n", (long)A.f.pRef.pRefCell, (long)nC, (double)ofP[rc],
                        (double)A.f.p[rc], (double)A.f.pRef.pRefValue);
            const scalar pScale = std::fmax(std::fabs(dP.refMax), scalar(1e-300));
            check("OpenFOAM's own p is pinned at pRefValue at that index after the refinement",
                  std::fabs(ofP[rc] - A.f.pRef.pRefValue) < scalar(1e-9)*pScale);
            check("...and brae's is, at the SAME index",
                  std::fabs(A.f.p[rc] - A.f.pRef.pRefValue) < scalar(1e-9)*pScale);
        }
        // THE CONTROL THAT DISCRIMINATES: pin a cell the change ADDED. pEqn adds
        // (pRefValue - p[pRefCell]) to the whole field, so the choice of cell is a whole-field offset,
        // and this proves the gate sees it. It is not a plausible implementation -- it is the proof that
        // the arm above can fail, which the two vacuous arms below cannot give.
        {
            setenv("BRAE_CONTROL_PREF_ANOTHER_CELL", "1", 1);
            Arm R;
            runArm(R, caseDir, startDir, nSteps);
            unsetenv("BRAE_CONTROL_PREF_ANOTHER_CELL");
            const Diff cP = compare(R.f.p, ofP);
            std::printf("  CONTROL (a cell the change added pinned): p rel %.4e against the gate's %.4e\n",
                        (double)cP.rel(), (double)dP.rel());
            check("...the control ran every step", R.r.steps == nSteps);
            check("...and is caught: pinning another cell moves p a thousand times further than the "
                  "gate's own distance", cP.rel() > scalar(1e3)*std::fmax(dP.rel(), scalar(1e-300)));
        }
        // ...AND THE ONE THAT CANNOT WITNESS, measured rather than assumed: renumbering the reference
        // through the change's own reverseCellMap. hexRef8 MODIFIES the parent in place and adds the
        // other seven children, so every retained cell keeps its index and the map is the identity --
        // 16400 -> 16400 here. It is kept because that identity is the reason this unit is a range check
        // and not a renumbering, and because an UNREFINING fixture would part the two.
        {
            setenv("BRAE_CONTROL_PREF_RENUMBER", "1", 1);
            Arm R;
            runArm(R, caseDir, startDir, nSteps);
            unsetenv("BRAE_CONTROL_PREF_RENUMBER");
            const Diff same = compare(R.f.p, A.f.p);
            std::printf("  VACUOUS BY MEASUREMENT (the reference renumbered through the map): p %.4e from "
                        "the gate's own arm\n", (double)same.linf);
            check("...renumbering through the map changes NOTHING under pure refinement, so no arm here "
                  "tests the choice of that rule", same.linf == scalar(0));
        }
    }

    // ---- THE CrankNicolson LEVELS, on the cn2d profile. The control drops the mapped levels and lets the
    // scheme re-create each at the new size: every field is then the right SIZE, nothing throws, the run
    // completes, and the scheme silently restarts as Euler at each change.
    if (cn)
    {
        setenv("BRAE_CONTROL_AMR_NO_CN_MAP", "1", 1);
        Arm K;
        runArm(K, caseDir, startDir, nSteps);
        unsetenv("BRAE_CONTROL_AMR_NO_CN_MAP");
        const Diff cA = compare(K.f.alpha1.internal, ofAlpha);
        const Diff cU = compare(K.f.U.internal, ofU);
        std::printf("  CONTROL (the CrankNicolson state re-created, not mapped): alpha %.4e, U rel %.4e\n",
                    (double)cA.linf, (double)cU.rel());
        check("...the control ran every step", K.r.steps == nSteps);
        // IT DISCRIMINATES, and that had to be measured rather than argued: a ddt0 LEVEL created at step k
        // is zero for the whole of step k, so the level ddt0(rho,U) itself is still zero at the only change
        // that maps it here -- but the state this control drops is the whole of InterAmrCn, and Uf's
        // old-old level, phi's, and the alpha flux's two blend levels are NOT zero there. Measured: alpha
        // 1.07e-02 and U 3.1e-01 from the gate's own arm, twelve orders above its distance from OpenFOAM.
        check("...and is caught: its alpha is more than a million times further out than the gate's",
              cA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
    }

    // ---- THE fvOPTIONS SELECTION, on the porosity profile. OpenFOAM re-runs the selection at every
    // topology change and prints what it got; brae must have the same cells, and the volume is the
    // sharper half of the comparison because it is INVARIANT -- the count grows eightfold on the refined
    // part of the zone while the volume does not move a digit.
    if (profile == "porosity")
    {
        const std::vector<Selected> ofSel = readOfSelections(logPath);
        check("OpenFOAM's log carries a selection line to compare against", !ofSel.empty());
        check("the case has an fvOption to re-select", !A.f.fvOptions.empty());
        if (!ofSel.empty() && !A.f.fvOptions.empty())
        {
            const Selected& want = ofSel.back();
            const std::vector<label>& cells = A.f.fvOptions.options.front().cells;
            scalar V = 0;
            for (const label c : cells)
            {
                if (c >= 0 && c < nC) V += A.g.V()[static_cast<std::size_t>(c)];
            }
            std::printf("  the selection: OpenFOAM %ld cells of volume %.12g, brae %ld of %.12g "
                        "(it started at %ld)\n", (long)want.n, (double)want.V, (long)cells.size(),
                        (double)V, (long)ofSel.front().n);
            check("the zone GREW through the change, so this profile can witness the re-selection",
                  want.n > ofSel.front().n);
            check("brae re-selected OpenFOAM's own cell count", static_cast<label>(cells.size()) == want.n);
            check("...and its volume, which the refinement does not change",
                  std::fabs(V - want.V) < scalar(1e-10)*std::fmax(want.V, scalar(1e-300)));
            // THE CONTROL: keep the labels instead of resolving the selection again.
            setenv("BRAE_CONTROL_AMR_NO_RESELECT", "1", 1);
            Arm K;
            runArm(K, caseDir, startDir, nSteps);
            unsetenv("BRAE_CONTROL_AMR_NO_RESELECT");
            const std::vector<label>& kept = K.f.fvOptions.options.front().cells;
            scalar kV = 0;
            for (const label c : kept)
            {
                if (c >= 0 && c < static_cast<label>(K.g.V().size())) kV += K.g.V()[static_cast<std::size_t>(c)];
            }
            const Diff cU = compare(K.f.U.internal, ofU);
            std::printf("  CONTROL (the selections kept, not resolved again): %ld cells of volume %.12g, "
                        "U rel %.4e\n", (long)kept.size(), (double)kV, (double)cU.rel());
            check("...the control ran every step", K.r.steps == nSteps);
            check("...and is caught on the selection: fewer cells and a smaller volume than OpenFOAM's",
                  static_cast<label>(kept.size()) < want.n && kV < scalar(0.99)*want.V);
            check("...and on the answer: U further from OpenFOAM than the gate's own distance",
                  cU.rel() > scalar(1e3)*std::fmax(dU.rel(), scalar(1e-300)));
        }
    }

    // ---- THE REFINEMENT STATE OFF DISK, on the levels profile. The arms above already hold the cell count
    // and every cell's LEVEL to OpenFOAM's, which is this unit's claim; what is left is to show that they
    // would not hold without the read.
    if (levelsProfile)
    {
        check("the mesh brae started from carries a refinement state at all",
              A.f.amr && !A.f.amr->state.levels.cellLevel.empty());
        label mx = 0;
        std::size_t split = 0;
        if (A.f.amr)
        {
            for (const label l : A.f.amr->state.levels.cellLevel) mx = (l > mx) ? l : mx;
            split = A.f.amr->state.history.parent.size();
        }
        std::printf("  brae read the state: cellLevel up to %ld, %zu split cell(s) in the history\n",
                    (long)mx, split);
        // THE FIXTURE MUST BE ABLE TO WITNESS: a UNIFORM level makes the whole unit invisible, because
        // consistentRefinement's 2:1 constraint only differs where levels differ and the >8-anchor protected
        // set only fires at a transition. The script asserts the seed's spread from OpenFOAM's own file;
        // this asserts that brae ended up holding a non-trivial one.
        check("...and it is not all zero, so reading it can be told from not reading it", mx > 0);
        if (levelsBinary)
        {
            // THE HISTORY IS GONE ON PURPOSE on this profile, so a fresh one is correct: with no file
            // OpenFOAM builds an ACTIVE history with every cell visible and NO parents (hexRef8.C:1953-1966),
            // which is what makes snappy's own refinement permanent and is why motorBike deletes the file.
            // `split` is then the mesh's cell count, not the seed's 3,198 split entries.
            check("...and with the history file removed, a FRESH one is built rather than none",
                  A.f.amr && A.f.amr->state.history.active);
        }
        else
        {
            check("...and the history carries split cells, so an unrefinement has parents to walk", split > 0);
        }

        // ...AND ON THE restart PROFILE, THE Uf READ, which is what that profile exists for.
        if (restartProfile)
        {
            check("the start directory carries a Uf, so this profile can witness the read at all",
                  A.f.UfWasRead);
            // THE CONTROL: interpolate Uf from U instead of reading it, which is what this port did.
            // createUfIfPresent.H is IOobject::READ_IF_PRESENT with fvc::interpolate(U) only as the FALLBACK,
            // and a refining mesh is dynamic -- so it is AUTO_WRITE and every time directory it writes
            // carries a Uf. The old comment said "a `Uf` file in the start directory is a restart's, and a
            // restart of a moving mesh is refused", which was true of a MOVING mesh and false of this one.
            // MEASURED: alpha 8.8625e-02 against the port's 5.6049e-10, and rAU 8.0261e-01 against
            // 1.2132e-08 -- because `phi = mesh.Sf() & Uf()` (interFoam.C:131) consumes it at the change.
            setenv("BRAE_CONTROL_NO_UF_READ", "1", 1);
            Arm Q;
            runArm(Q, caseDir, startDir, nSteps);
            unsetenv("BRAE_CONTROL_NO_UF_READ");
            const Diff qA = compare(Q.f.alpha1.internal, ofAlpha);
            const Diff qU = compare(Q.f.U.internal, ofU);
            std::printf("  CONTROL (Uf interpolated, not read): alpha %.4e, U rel %.4e\n",
                        (double)qA.linf, (double)qU.rel());
            check("...the control ran every step", Q.r.steps == nSteps);
            check("...and did NOT read the file, which is what makes it a control", !Q.f.UfWasRead);
            check("...and is caught: interpolating Uf where OpenFOAM reads it is a million times further "
                  "out than the gate", qA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
        }

        // THE CONTROL: do not read them. This restores exactly what brae SHIPPED before this unit -- level 0
        // everywhere, whatever the files say -- which is also what OpenFOAM does when the files are ABSENT.
        // So the control is a port that existed rather than one invented for the gate.
        //
        // It is caught STRUCTURALLY and not by a field bound, which is the strongest form available here:
        // `cellLevel[celli] < maxRefinement` (dynamicRefineFvMesh.C:861) is true for every cell when the
        // levels read zero, so the control refines cells that are already at the cap. MEASURED: 18,466 cells
        // against OpenFOAM's 4,998 -- nearly four times the mesh -- with alpha 8.5760e-02 and U 4.2137e-01.
        setenv("BRAE_CONTROL_AMR_NO_LEVELS", "1", 1);
        Arm L;
        runArm(L, caseDir, startDir, nSteps);
        unsetenv("BRAE_CONTROL_AMR_NO_LEVELS");
        const Diff lA = compare(L.f.alpha1.internal, ofAlpha);
        std::printf("  CONTROL (the refinement state NOT read): %ld cells against OpenFOAM's %ld, alpha "
                    "%.4e\n", (long)L.m.nCells(), (long)nC, (double)lA.linf);
        check("...the control ran every step", L.r.steps == nSteps);
        check("...and is caught on the MESH: taking every cell as level 0 refines cells already at "
              "maxRefinement, so it does not even end on OpenFOAM's cell count",
              L.m.nCells() != nC);
        check("...and on the answer, a million times further out than the gate's",
              lA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
    }

    // ---- THE TURBULENCE FIELDS, on the ras profile, which is the whole point of that unit: k, the
    // second transported scalar and nut are registered AUTO_WRITE fields that MapGeometricFields autoMaps
    // in OpenFOAM, so brae maps them through the same cell and patch mappers as alpha1 -- and OpenFOAM
    // writes all three beside every time directory, so the oracle is its own fields rather than a residual.
    if (ras)
    {
        const bool sst = (A.f.turbulence.model == cpu::interFoam::InterRasModel::KOmegaSST);
        const std::string secondName = sst ? "omega" : "epsilon";
        check("the case was recognised as turbulent", A.f.turbulence.on);
        const std::vector<scalar> ofK = cellValues(readField<scalar>(ofDir + "/k"), nC);
        const std::vector<scalar> ofS = cellValues(readField<scalar>(ofDir + "/" + secondName), nC);
        const std::vector<scalar> ofNut = cellValues(readField<scalar>(ofDir + "/nut"), nC);
        const Diff dK = compare(A.f.turbulence.k.internal, ofK);
        const Diff dS = compare(sst ? A.f.turbulence.omega.internal : A.f.turbulence.epsilon.internal, ofS);
        const Diff dNut = compare(A.f.turbulence.nut.internal, ofNut);
        std::printf("  k %.4e (rel %.4e)   %s %.4e (rel %.4e)   nut %.4e (rel %.4e)\n",
                    (double)dK.linf, (double)dK.rel(), secondName.c_str(),
                    (double)dS.linf, (double)dS.rel(), (double)dNut.linf, (double)dNut.rel());
        check("k is at this profile's floor, relative", dK.rel() < B.k);
        check("the second transported scalar is at this profile's floor, relative", dS.rel() < B.second);
        check("nut is at this profile's floor, relative", dNut.rel() < B.nut);

        // THE FIXTURE MUST BE ABLE TO WITNESS, and for a MAPPER that means the fields must not be uniform
        // at the moment of the change: a uniform field is mapped correctly by a broken mapper. This case
        // ships k and epsilon uniform 0.1 and nut uniform 0, and they develop in ONE step -- measured from
        // OpenFOAM's own written fields, epsilon spans 0.0998 to 8.449 at t = 0.001, so the SECOND change
        // maps a developed field. The spread is asserted rather than trusted.
        const auto spread = [](const std::vector<scalar>& v)
        {
            scalar lo = v.empty() ? scalar(0) : v[0], hi = lo;
            for (const scalar x : v) { lo = std::fmin(lo, x); hi = std::fmax(hi, x); }
            return hi - lo;
        };
        std::printf("  the mapped fields' spread: k %.4g, %s %.4g, nut %.4g (a UNIFORM field would be "
                    "mapped correctly by a broken mapper)\n", (double)spread(ofK), secondName.c_str(),
                    (double)spread(ofS), (double)spread(ofNut));
        check("k varies across the mesh, so its mapping can be witnessed", spread(ofK) > scalar(1e-6));
        check("...and the second scalar's", spread(ofS) > scalar(1e-6));
        check("...and nut's", spread(ofNut) > scalar(1e-9));

        // ...AND THE WALL FUNCTIONS' OWN FACES, which is the other half. A wall function's y and its face
        // cell change when a wall face is SPLIT, so a refinement that never touches a wall patch would
        // leave that half unwitnessed. MEASURED on this fixture from OpenFOAM's own written meshes:
        // leftWall 50 -> 56 -> 68 faces and lowerWall 62 -> 68 -> 80, while rightWall stays at 50 because
        // the water has not reached it -- so the split wall faces are real and one wall is a control.
        // The per-patch counts need no comparison of their own: the arms above already hold every face's
        // owner and every internal face's neighbour to OpenFOAM's, which pins the boundary layout with them.
        label wallFaces = 0, wallPatches = 0;
        for (const FvPatch& q : A.patches)
        {
            if (q.type != "wall") continue;
            wallFaces += q.size;
            ++wallPatches;
        }
        std::printf("  wall-function faces: %ld over %ld wall patch(es)\n",
                    (long)wallFaces, (long)wallPatches);
        check("the mesh has wall-function faces at all, so the wall treatment is exercised", wallFaces > 0);

        // THE CONTROL: do not recompute the wall distance and the filter width. It is a TEST on the sst
        // profile and a STATEMENT ABOUT THE FIXTURE on the ke one, and the gate asserts which rather than
        // running the same switch twice and reporting one number -- the same shape as the old-time control
        // above, which is vacuous under Euler and live under CrankNicolson.
        //
        // kEpsilon BUILDS NO CELL WALL DISTANCE (only kOmegaSST's F1 and F2 blend on one), and its filter
        // width is LES's, so on `ke` there is nothing for this switch to leave stale. MEASURED: identical
        // to the last digit -- alpha 1.1425e-14 and k 2.2516e-14 either way. On `sst` it is caught by
        // twelve orders: alpha 2.0796e-03 against 2.8866e-15, p_rgh 2.3079e-02, U 1.1012e-01.
        setenv("BRAE_CONTROL_AMR_NO_TURB_UPDATE", "1", 1);
        Arm T;
        runArm(T, caseDir, startDir, nSteps);
        unsetenv("BRAE_CONTROL_AMR_NO_TURB_UPDATE");
        const Diff tA = compare(T.f.alpha1.internal, ofAlpha);
        const Diff tU = compare(T.f.U.internal, ofU);
        const Diff tSelfA = compare(T.f.alpha1.internal, A.f.alpha1.internal);
        std::printf("  CONTROL (the wall distance and filter width NOT recomputed): alpha %.4e, U rel "
                    "%.4e, and %.4e from the gate's own arm\n",
                    (double)tA.linf, (double)tU.rel(), (double)tSelfA.linf);
        check("...the control ran every step", T.r.steps == nSteps);
        if (motorBikeProfile)
        {
            // AND HERE IT IS ALPHA THAT IS BLIND, again measured: the control leaves alpha BIT-IDENTICAL to
            // the gate's own arm (0.0000e+00 between them) because two steps at Co 0.026 do not carry a
            // nut difference into the phase fraction. It moves U to 4.6894e-03 against the gate's
            // 2.2091e-13, ten orders, so that is what this profile asserts.
            check("...and is caught on U: kOmegaSST blends F1 and F2 on the cell wall distance, and leaving "
                  "it at the old mesh's puts U ten orders out where alpha does not move at all",
                  tU.rel() > scalar(1e6)*std::fmax(dU.rel(), scalar(1e-300)));
        }
        else if (sstProfile)
        {
            check("...and is caught: kOmegaSST blends F1 and F2 on the cell wall distance, so leaving it "
                  "at the old mesh's is a million times further out than the gate",
                  tA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
        }
        else
        {
            check("...and it changes NOTHING on this profile, because kEpsilon builds no cell wall "
                  "distance -- so this profile does not test that recompute and says so",
                  tSelfA.linf == scalar(0));
        }

        // THE TURBULENCE SOLVES, against OpenFOAM's own log. A field at 1e-14 could still be a solver that
        // stopped somewhere else; an iteration count cannot.
        const std::vector<LinearSolveRecord> ofKs = gatecheck::readOfSolves(logPath, "k");
        const std::vector<LinearSolveRecord> ofSs = gatecheck::readOfSolves(logPath, secondName.c_str());
        check("OpenFOAM's log carries k solves to compare against", !ofKs.empty());
        if (!ofKs.empty())
        {
            failures += gatecheck::compareSolves("k", A.r.kSolves, ofKs, nSteps, "k");
        }
        if (!ofSs.empty())
        {
            failures += gatecheck::compareSolves(
                secondName.c_str(), sst ? A.r.omegaSolves : A.r.epsilonSolves, ofSs, nSteps,
                secondName.c_str());
        }
    }

    // ---- MRF THROUGH A CHANGE, on the mrf profile. TWO ORACLES, and the first is the sharper of the
    // two because it does not go through a solve at all: a topoChanging mesh writes its polyMesh into
    // every time directory, cellZones among them, so OpenFOAM's OWN RENUMBERED ZONE is on disk and the
    // comparison is cell by cell. The fvOptions profile had to read a printed count off a log line;
    // MRFZone prints nothing at all (it has no Info<< of its own), and this is better than one anyway.
    if (mrf)
    {
        const std::vector<label> ofZone = readOfCellZone(ofDir + "/polyMesh", "");
        const std::vector<label> ofZone0 = readOfCellZone(caseDir + "/constant/polyMesh", "");
        check("OpenFOAM wrote a cellZone into its time directory to compare against", !ofZone.empty());
        check("the case carries a cellZone at construction", !ofZone0.empty());
        check("brae kept a zone of that name", !A.f.cellZones.empty());
        if (!ofZone.empty() && !A.f.cellZones.empty())
        {
            const std::vector<label>& mine = A.f.cellZones.begin()->second;
            std::printf("  the zone `%s`: OpenFOAM %ld cells at t = %.10g, brae %ld (it started at %ld "
                        "on %ld cells, and the mesh is %ld now)\n",
                        A.f.cellZones.begin()->first.c_str(), (long)ofZone.size(), (double)A.r.time,
                        (long)mine.size(), (long)ofZone0.size(),
                        (long)readOfCellZone(caseDir + "/constant/polyMesh", "").size(), (long)nC);
            // THE FIXTURE MUST BE ABLE TO WITNESS, and for this unit that is not "the mesh refined" but
            // "the MRF ZONE ITSELF refined". A zone away from the refinement band would be carried by
            // doing nothing, and the control below would pass.
            check("the MRF zone GREW through the change, so this profile can witness the carry",
                  ofZone.size() > ofZone0.size());
            check("brae's zone has OpenFOAM's own cell count", mine.size() == ofZone.size());
            if (mine.size() == ofZone.size())
            {
                std::size_t wrong = 0;
                label firstWrong = -1;
                for (std::size_t i = 0; i < mine.size(); ++i)
                {
                    if (mine[i] == ofZone[i]) continue;
                    if (!wrong) firstWrong = static_cast<label>(i);
                    ++wrong;
                }
                if (wrong)
                {
                    std::printf("  ...%ld of %ld labels differ, first at %ld (brae %ld, OpenFOAM %ld)\n",
                                (long)wrong, (long)mine.size(), (long)firstWrong,
                                (long)mine[static_cast<std::size_t>(firstWrong)],
                                (long)ofZone[static_cast<std::size_t>(firstWrong)]);
                }
                check("...and EVERY ONE of its labels, cell by cell, is OpenFOAM's", wrong == 0);
            }
            // resetZones walks the per-cell zone id ASCENDING (polyTopoChange.C:1900-1925), so the list
            // OpenFOAM writes is sorted -- and a port that appended each parent's children after the
            // parent would match the COUNT and not the order.
            check("OpenFOAM's own carried zone is ascending, which is what makes the order comparable",
                  std::is_sorted(ofZone.begin(), ofZone.end()));
            check("...and brae's is too", std::is_sorted(mine.begin(), mine.end()));
        }
        // THE MRF FACE LISTS the zone rebuild exists for, printed so the arm says what it covers: a zone
        // carried with the right cells but stale face lists is exactly what the control below is.
        if (!A.f.mrfZones.empty())
        {
            const cpu::MRF::Zone& z = A.f.mrfZones.front();
            std::size_t inc = 0, exc = 0;
            for (const std::vector<label>& v : z.includedFaces) inc += v.size();
            for (const std::vector<label>& v : z.excludedFaces) exc += v.size();
            std::printf("  the rebuilt face lists: %ld internal, %ld included, %ld excluded on %ld cells "
                        "(the mesh has %ld internal faces)\n", (long)z.internalFaces.size(), (long)inc,
                        (long)exc, (long)z.cells.size(), (long)nIF);
            check("the zone's internal-face list is inside the NEW mesh's face range",
                  z.internalFaces.empty() || z.internalFaces.back() < nIF);
            check("the zone moves faces with the frame, so makeRelative and zeroFilter have work to do",
                  !z.internalFaces.empty() && inc > 0);
        }
        // THE CONTROL: keep the face lists the zone was built with, which is the port that does not throw.
        setenv("BRAE_CONTROL_AMR_NO_MRF_UPDATE", "1", 1);
        Arm K;
        runArm(K, caseDir, startDir, nSteps);
        unsetenv("BRAE_CONTROL_AMR_NO_MRF_UPDATE");
        const Diff cA = compare(K.f.alpha1.internal, ofAlpha);
        const Diff cU = compare(K.f.U.internal, ofU);
        const Diff cP = compare(K.f.p_rgh.internal, ofPrgh);
        std::printf("  CONTROL (the MRF face lists kept, not rebuilt): alpha %.4e, U rel %.4e, "
                    "p_rgh rel %.4e\n", (double)cA.linf, (double)cU.rel(), (double)cP.rel());
        check("...the control ran every step", K.r.steps == nSteps);
        check("...and is caught: its alpha is a million times further out than the gate's",
              cA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
        check("...and its U, which is what the frame drives", cU.rel() > scalar(1e6)*std::fmax(dU.rel(), scalar(1e-300)));
        // ...AND THE SAME CONTROL ON THE DEVICE ARM, which is a separate rebuild and needs its own
        // fail-proof. The host control alone would leave the device half covered by nothing but a
        // device-vs-host bound -- and a device that never rebuilt its zones would still track a host that
        // never rebuilt its own, so that bound cannot witness the thing this unit ported. The device zones
        // are built from the host zones, so this control reaches both, and the arm says which one it is.
        setenv("BRAE_CONTROL_AMR_NO_MRF_UPDATE", "1", 1);
        Arm KD;
        runArm(KD, caseDir, startDir, nSteps, /*onDevice=*/true);
        unsetenv("BRAE_CONTROL_AMR_NO_MRF_UPDATE");
        const Diff cdA = compare(KD.f.alpha1.internal, ofAlpha);
        const Diff cdU = compare(KD.f.U.internal, ofU);
        std::printf("  CONTROL on the DEVICE arm: alpha %.4e, U rel %.4e\n",
                    (double)cdA.linf, (double)cdU.rel());
        check("...the device control ran every step", KD.r.steps == nSteps);
        check("...and the DEVICE arm is caught on it too, so its rebuild is proven and not inferred",
              cdA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
    }

    // ---- WHAT THE BOUNDS ABOVE ARE, on a case that AMPLIFIES. The mrf profile's bounds are 1e-09, not
    // 1e-14, and a bound that loose has to be justified by something other than the code that has to meet
    // it. So the gate is handed a SECOND OpenFOAM run of the same case with ONE cell of the initial
    // alpha.water perturbed by ONE ULP, and compares OpenFOAM to ITSELF.
    //
    // MEASURED on the mrf profile, OpenFOAM against itself at one ulp (1.11e-16 on one cell):
    //     t        alpha       p_rgh rel    U rel        phi rel      Uf rel
    //     1 step   1.11e-16    0.00e+00     0.00e+00     0.00e+00     0.00e+00
    //     2 steps  1.11e-16    2.13e-14     8.80e-15     1.29e-15     5.73e-15
    //     3 steps  1.11e-15    6.10e-10     2.32e-10     2.80e-11     2.50e-10
    //     4 steps  1.85e-12    3.96e-09     2.29e-09     2.99e-10     2.66e-09
    // -- four orders between step 2 and step 3, which is the step whose change first gives pcorr real
    // work (183 iterations at initial residual 1, where the first two solve nothing). brae at four steps
    // reads p_rgh 2.58e-09 against that 3.96e-09: CLOSER TO OPENFOAM THAN OPENFOAM'S OWN ONE-ULP TWIN.
    // ...AND IT IS MANDATORY ON THE mrf PROFILE, because that profile's bounds are five orders looser than
    // every other one's ON THE STRENGTH OF THIS ARM. With the argument optional, six arguments asserted
    // 1e-09 with nothing behind it and the whole gate went green.
    check("the mrf and levels profiles were handed their one-ulp twin, which is the only thing justifying "
          "their bounds",
          (!mrf && !levelsProfile) || !ulpDir.empty());
    if (!ulpDir.empty())
    {
        const std::vector<scalar> uA = cellValues(readField<scalar>(ulpDir + "/" + A.f.alphaName), nC);
        const std::vector<scalar> uP = cellValues(readField<scalar>(ulpDir + "/p_rgh"), nC);
        const std::vector<vector> uU = cellValues(readField<vector>(ulpDir + "/U"), nC);
        const bool sized = uA.size() == ofAlpha.size() && uP.size() == ofPrgh.size()
                        && uU.size() == ofU.size();
        check("the one-ulp twin is on the same mesh as the oracle", sized);
        if (sized)
        {
            const Diff eA = compare(uA, ofAlpha);
            const Diff eP = compare(uP, ofPrgh);
            const Diff eU = compare(uU, ofU);
            std::printf("  OpenFOAM against ITSELF at one ulp: alpha %.4e   p_rgh rel %.4e   U rel %.4e\n",
                        (double)eA.linf, (double)eP.rel(), (double)eU.rel());
            std::printf("  ...brae is %.2fx that on alpha, %.2fx on p_rgh, %.2fx on U\n",
                        (double)(dAlpha.linf/std::fmax(eA.linf, scalar(1e-300))),
                        (double)(dPrgh.rel()/std::fmax(eP.rel(), scalar(1e-300))),
                        (double)(dU.rel()/std::fmax(eU.rel(), scalar(1e-300))));
            // THE FIXTURE MUST BE ABLE TO WITNESS ITS OWN BOUND. If one ulp did NOT amplify here, these
            // bounds would be slack rather than the case's conditioning, and this arm says which.
            check("the case AMPLIFIES: one ulp of OpenFOAM's own input moves its p_rgh by more than 1e-10, "
                  "so this profile's bounds are the case's conditioning and not slack",
                  eP.rel() > scalar(1e-10));
            // ...and the statement the bound is worth: brae is INSIDE the envelope OpenFOAM's own
            // round-off draws, asserted at 1x rather than at a factor, because a factor would be a
            // tolerance and this is a comparison.
            //
            // IT IS ASSERTED ON p_rgh AND U AND NOT ON alpha, and that is measured rather than chosen. The
            // two curves CROSS: brae's distance starts at its own 160-iteration PCG floor and the twin's
            // starts at one ulp in one cell, so the twin is BELOW brae early and overtakes it as the case
            // amplifies. Measured on this fixture, brae as a multiple of the twin:
            //     steps   alpha   p_rgh    U
            //     1       0.00x   18.15x   242.06x
            //     2       4.34x   0.82x    0.86x
            //     3       0.58x   0.68x    0.83x
            //     4       0.69x   0.48x    0.64x
            // The amplification assertion above pins the regime -- at one step the twin's p_rgh is 1.78e-14
            // and that check fails, which is the correct answer, because there the 1e-09 bounds WOULD be
            // slack. Inside the regime p_rgh and U are below 1x at every step; alpha is NOT (4.34x at two
            // steps, and non-monotone because brae's own alpha is exactly 0 at one step). So alpha is held
            // by its ABSOLUTE bound instead (1e-10, measured 7.87e-11) and its ratio is printed, not
            // asserted -- an assertion that happens to hold at the step count someone picked is not one.
            check("brae is within this profile's measured multiple of OpenFOAM's own one-ulp twin, on p_rgh",
                  B.ulpFactor > scalar(0) && dPrgh.rel() <= B.ulpFactor*eP.rel());
            check("...and on U",
                  B.ulpFactor > scalar(0) && dU.rel() <= B.ulpFactor*eU.rel());
        }
    }

    // ---- THE DEVICE ARM. The mesh change is HOST work on either loop -- topology surgery and six
    // integer maps -- so what this measures is the round trip: every mesh-sized buffer down to the
    // host, the change, and every one of them back up on a DeviceMesh rebuilt from scratch. A buffer
    // left at the old size, or a schedule cache replaying the old addressing, lands here.
    // ...EXCEPT on the box profile, where the device loop REFUSES by name: its topology and motion
    // branches are not yet composed in OpenFOAM's order, and a refusal nobody runs stops firing.
    if (boxProfile)
    {
        std::string msg;
        try
        {
            Arm D;
            runArm(D, caseDir, startDir, nSteps, /*onDevice=*/true);
        }
        catch (const std::exception& e)
        {
            msg = e.what();
        }
        std::printf("  device: %s\n", msg.empty() ? "RAN" : msg.substr(0, 120).c_str());
        check("the device refuses a mesh that refines and moves, by name",
              msg.find("refines AND a motion solver") != std::string::npos);
    }
    else
    {
        Arm D;
        runArm(D, caseDir, startDir, nSteps, /*onDevice=*/true);
        std::printf("  device: %ld steps to t = %.10g on %ld cells\n",
                    (long)D.r.steps, (double)D.r.time, (long)D.m.nCells());
        check("the device arm took every step", D.r.steps == nSteps);
        check("the device arm refined onto OpenFOAM's cell count", D.m.nCells() == nC);
        if (D.m.nCells() == nC)
        {
            const Diff dvA = compare(D.f.alpha1.internal, ofAlpha);
            const Diff dvP = compare(D.f.p_rgh.internal, ofPrgh);
            const Diff dvU = compare(D.f.U.internal, ofU);
            const Diff dvPhi = compare(D.f.phi.internal, ofPhi);
            std::printf("  device vs OpenFOAM: alpha %.4e   p_rgh rel %.4e   U rel %.4e   phi rel %.4e\n",
                        (double)dvA.linf, (double)dvP.rel(), (double)dvU.rel(), (double)dvPhi.rel());
            const Diff hvA = compare(D.f.alpha1.internal, A.f.alpha1.internal);
            const Diff hvP = compare(D.f.p_rgh.internal, A.f.p_rgh.internal);
            const Diff hvU = compare(D.f.U.internal, A.f.U.internal);
            const Diff hvPhi = compare(D.f.phi.internal, A.f.phi.internal);
            std::printf("  device vs the host arm: alpha %.4e   p_rgh rel %.4e   U rel %.4e   phi rel %.4e\n",
                        (double)hvA.linf, (double)hvP.rel(), (double)hvU.rel(), (double)hvPhi.rel());
            // The bounds are the two arms' own floor on this case, measured; the host arm is held to
            // OpenFOAM above, so a device number is the device's own distance.
            check("the device's alpha is at this profile's floor from OpenFOAM's", dvA.linf < B.devAlpha);
            check("the device's p_rgh is at this profile's floor, relative", dvP.rel() < B.devPRgh);
            check("the device's U is at this profile's floor, relative", dvU.rel() < B.devU);
            check("the device's phi is at this profile's floor, relative", dvPhi.rel() < B.devPhi);
            // ...and against the HOST arm, which is the sharper of the two: both loops run the same
            // host mapper and the same host pcorr, so what is left between them is the device's own
            // arithmetic on the mapped fields.
            check("the device is at this profile's floor from the host arm's alpha", hvA.linf < B.hostDevAlpha);
            check("the device is at this profile's floor from the host arm's p_rgh",
                  hvP.rel() < B.hostDevPRgh);
            check("the device is at this profile's floor from the host arm's U", hvU.rel() < B.hostDevU);
            check("the device is at this profile's floor from the host arm's phi", hvPhi.rel() < B.hostDevPhi);
            // ...AND THE CLOSURE'S OWN FIELDS on the device arm, which is what the ras unit ported there.
            // They are compared SEPARATELY from U and p_rgh because the two halves fail differently: nut
            // reaches the momentum equation through nuEff, so a wrong nut shows up as U and p_rgh while
            // alpha stays near the floor -- which is exactly how this unit's first device attempt read.
            if (ras && D.f.turbulence.on)
            {
                const bool dsst = (D.f.turbulence.model == cpu::interFoam::InterRasModel::KOmegaSST);
                const std::string dsn = dsst ? "omega" : "epsilon";
                const std::vector<scalar> ofKd = cellValues(readField<scalar>(ofDir + "/k"), nC);
                const std::vector<scalar> ofSd = cellValues(readField<scalar>(ofDir + "/" + dsn), nC);
                const std::vector<scalar> ofNd = cellValues(readField<scalar>(ofDir + "/nut"), nC);
                const Diff dvK = compare(D.f.turbulence.k.internal, ofKd);
                const Diff dvS = compare(dsst ? D.f.turbulence.omega.internal
                                              : D.f.turbulence.epsilon.internal, ofSd);
                const Diff dvNut = compare(D.f.turbulence.nut.internal, ofNd);
                std::printf("  device vs OpenFOAM: k %.4e   %s %.4e   nut %.4e (relative)\n",
                            (double)dvK.rel(), dsn.c_str(), (double)dvS.rel(), (double)dvNut.rel());
                check("the device's k is at this profile's floor, relative", dvK.rel() < B.devK);
                check("...its second transported scalar", dvS.rel() < B.devSecond);
                check("...and its nut, which is what reaches the momentum equation", dvNut.rel() < B.devNut);
            }
        }
        // ...AND THE SAME ARM TWICE IN ONE PROCESS, which is the detector for a cache keyed on a
        // recycled pointer: the device pool hands the second run the first run's blocks, so a schedule
        // cached against an address rather than against DeviceLduView::addressingId replays the FIRST
        // mesh's walk on the second run. It has cost this project a 3.0e-02 on U before.
        Arm D2;
        runArm(D2, caseDir, startDir, nSteps, /*onDevice=*/true);
        const Diff twice = compare(D2.f.alpha1.internal, D.f.alpha1.internal);
        const Diff twiceU = compare(D2.f.U.internal, D.f.U.internal);
        std::printf("  the device arm run twice in one process: alpha %.4e, U %.4e apart\n",
                    (double)twice.linf, (double)twiceU.linf);
        check("the second device run in the same process is bit-identical to the first",
              twice.linf == scalar(0) && twiceU.linf == scalar(0));
    }

    std::printf("test_inter_amr_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
