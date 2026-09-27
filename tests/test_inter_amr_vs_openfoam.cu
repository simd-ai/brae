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
    label              nSteps)
{
    a.m.read(caseDir + "/constant/polyMesh");
    a.g.build(a.m);
    a.patches = buildPatches(a.m, a.g);
    MutableMesh mm;
    mm.m = &a.m;
    mm.g = &a.g;
    mm.patches = &a.patches;
    a.r = runInterFoam(caseDir, startDir, a.m, a.g, a.patches, nSteps, /*verbose=*/false, &a.f,
                       scalar(1.0e300), nullptr, &mm);
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: an ADAPTIVE mesh ==\n");
    if (argc < 6)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];

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
    check("alpha is within 5e-15 of OpenFOAM's", dAlpha.linf < scalar(5e-15));
    check("p_rgh is within 1e-14 relative", dPrgh.rel() < scalar(1e-14));
    check("U is within 1e-12 relative", dU.rel() < scalar(1e-12));
    check("p is within 1e-13 relative", dP.rel() < scalar(1e-13));
    check("rAU is within 1e-11 relative", dRAU.rel() < scalar(1e-11));
    check("phi is within 1e-12 relative", dPhi.rel() < scalar(1e-12));
    check("Uf is within 1e-12 relative", dUf.rel() < scalar(1e-12));

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
        check("every patch's alpha is within 5e-15", worstA.linf < scalar(5e-15));
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
                  rel < scalar(1e-9));
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
        std::printf("  VACUOUS BY MEASUREMENT (the old-time levels re-captured instead of mapped): "
                    "alpha %.4e and U %.4e from the gate's own arm\n",
                    (double)sameA.linf, (double)sameU.linf);
        check("...re-capturing the old-time levels changes NOTHING on this fixture, so no arm here "
              "tests their mapping", sameA.linf == scalar(0) && sameU.linf == scalar(0));
    }

    std::printf("test_inter_amr_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
