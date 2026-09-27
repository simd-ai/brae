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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
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
        if (line.rfind("Refined from ", 0) != 0) continue;
        const std::size_t a = line.find(kTo);
        if (a == std::string::npos) continue;
        out.push_back(static_cast<label>(std::atol(line.c_str() + a + kTo.size())));
    }
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
    a.m.read(caseDir + "/constant/polyMesh");
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
    };
    //   cn      7.7716e-15 alpha, 7.3e-15 p_rgh, 4.5e-13 U, 2.7e-14 p, 6.9e-12 rAU, 1.2e-13 phi
    const bool closed = (profile == "closed");
    const bool cn = (profile == "cn");
    const Bounds B = closed
        ? Bounds{5e-14, 1e-14, 1e-12, 1e-13, 1e-11, 5e-12, 1e-12, 1e-9, 5e-14, 1e-12, 1e-14, 5e-12}
        : cn
        ? Bounds{1e-14, 1e-14, 1e-12, 1e-13, 1e-11, 1e-12, 1e-12, 1e-9, 1e-14, 1e-12, 1e-14, 1e-12}
        : Bounds{5e-15, 1e-14, 1e-12, 1e-13, 1e-11, 1e-12, 1e-12, 1e-9, 5e-15, 1e-12, 5e-15, 1e-12};

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
    PrimitiveMesh ofM;
    ofM.read(ofDir + "/polyMesh");
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
        std::printf("  OpenFOAM refined %zu times, last to %ld cells\n", ofTo.size(),
                    ofTo.empty() ? 0L : (long)ofTo.back());
        check("OpenFOAM's log shows a refinement at all (the fixture can witness one)", !ofTo.empty());
        check("brae's last refinement ended on OpenFOAM's cell count",
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

    // THE BOUNDS ARE THE MEASURED AGREEMENT, each a little above what the run actually reads, and
    // interfoam_amr_vs_openfoam.sh records those numbers. They are floating-point floors and not
    // tolerances: this solver reproduces OpenFOAM's own arithmetic on this case, so alpha lands at
    // 1.9e-15 on 82,264 cells and every one of the nine solves matches OpenFOAM's residual to the bit.
    check("alpha is at this profile's floor from OpenFOAM's", dAlpha.linf < B.alpha);
    check("p_rgh is at this profile's floor, relative", dPrgh.rel() < B.pRgh);
    check("U is at this profile's floor, relative", dU.rel() < B.u);
    check("p is at this profile's floor, relative", dP.rel() < B.p);
    check("rAU is at this profile's floor, relative", dRAU.rel() < B.rAU);
    check("phi is at this profile's floor, relative", dPhi.rel() < B.phi);
    check("Uf is at this profile's floor, relative", dUf.rel() < B.Uf);

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
        std::printf("  patch values: alpha %.4e   U %.4e   p_rgh %.4e\n",
                    (double)worstA.linf, (double)worstU.linf, (double)worstP.linf);
        check("every patch's alpha is at this profile's floor", worstA.linf < B.alpha);
        check("every patch's U is within 1e-12 relative", worstU.rel() < scalar(1e-12));
        check("every patch's p_rgh is within 1e-14 relative", worstP.rel() < scalar(1e-14));
    }

    // ---- THE SOLVES, which is the arm that says the two codes solved the same systems. A pcorr solve
    // exists only because the mesh changed, and its iteration count is the sharpest statement this log
    // can make about the flux it was handed.
    const std::vector<LinearSolveRecord> ofP_rgh = gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofPcorr = gatecheck::readOfSolves(logPath, "pcorr");
    std::printf("  OpenFOAM: %zu p_rgh solves, %zu pcorr solves; brae %zu and %zu\n",
                ofP_rgh.size(), ofPcorr.size(), A.r.pSolves.size(), A.r.pcorrSolves.size());
    failures += gatecheck::compareSolves("p_rgh", A.r.pSolves, ofP_rgh, nSteps, "p_rgh");
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
            check("the final flux carries OpenFOAM's own continuity error to 1e-9 relative",
                  rel < B.contErr);
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
        runArm(C, caseDir, startDir, nSteps);
        unsetenv("BRAE_CONTROL_AMR_RESIZE_NOT_MAP");
        const Diff cA = compare(C.f.alpha1.internal, ofAlpha);
        const Diff cU = compare(C.f.U.internal, ofU);
        std::printf("  CONTROL (the cell fields resized, not mapped): alpha %.4e, U rel %.4e\n",
                    (double)cA.linf, (double)cU.rel());
        check("...the control ran every step", C.r.steps == nSteps);
        check("...and is caught: its alpha is more than a million times further out than the gate's",
              cA.linf > scalar(1e6)*std::fmax(dAlpha.linf, scalar(1e-300)));
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

    // ---- THE DEVICE ARM. The mesh change is HOST work on either loop -- topology surgery and six
    // integer maps -- so what this measures is the round trip: every mesh-sized buffer down to the
    // host, the change, and every one of them back up on a DeviceMesh rebuilt from scratch. A buffer
    // left at the old size, or a schedule cache replaying the old addressing, lands here.
    if (cn)
    {
        // CrankNicolson beside refinement is REFUSED on the device arm: its ddt0 levels are device
        // buffers, and mapping them is a unit of its own. A refusal that stops firing is how a port
        // surfaces, so the gate asserts it here rather than leaving the arm out.
        bool refused = false;
        std::string what;
        try
        {
            Arm D;
            runArm(D, caseDir, startDir, nSteps, /*onDevice=*/true);
        }
        catch (const std::exception& e)
        {
            refused = true;
            what = e.what();
        }
        check("the device arm refuses CrankNicolson beside refinement", refused);
        check("...and names the scheme", what.find("CrankNicolson") != std::string::npos);
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
            check("the device's p_rgh is within 1e-14 relative", dvP.rel() < scalar(1e-14));
            check("the device's U is within 1e-12 relative", dvU.rel() < scalar(1e-12));
            check("the device's phi is at this profile's floor, relative", dvPhi.rel() < B.devPhi);
            // ...and against the HOST arm, which is the sharper of the two: both loops run the same
            // host mapper and the same host pcorr, so what is left between them is the device's own
            // arithmetic on the mapped fields.
            check("the device is at this profile's floor from the host arm's alpha", hvA.linf < B.hostDevAlpha);
            check("the device is within 1e-14 of the host arm's p_rgh", hvP.rel() < scalar(1e-14));
            check("the device is within 1e-12 of the host arm's U", hvU.rel() < scalar(1e-12));
            check("the device is at this profile's floor from the host arm's phi", hvPhi.rel() < B.hostDevPhi);
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
