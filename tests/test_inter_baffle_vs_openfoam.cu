// brae's interFoam across a CYCLIC BAFFLE against REAL OpenFOAM's, on RAS/damBreakPorousBaffle.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_baffle_vs_openfoam.sh stages it and carries the measurements. TWO PROFILES:
//   cyclic   the baffle pair with p_rgh a plain `cyclic`: the coupling alone, through every operator,
//            matrix and linear solver of the host loop
//   porous   the case as createBaffles leaves it: p_rgh a porousBafflePressure, a cyclic with a JUMP
//            that updateCoeffs rebuilds from the flux through the baffle
// THE CONTROL of `cyclic` is OpenFOAM's own answer with the baffle two WALLS -- what the shared factory's
// zeroGradient placeholder silently ran; the control of `porous` is OpenFOAM's `cyclic` answer.
//
// THE DEVICE LOOP MUST REFUSE the case, naming the patch.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include "inter_solve_log.cuh"
#include "patch_entry_lookup.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

// MEASURED, per profile; each bound is about 30x its measurement. `jump` is the owner's jump against the
// one OpenFOAM writes, relative to the largest jump on the patch.
//   cyclic     20 steps: alpha 6.9e-15, p_rgh 6.6e-15, U 5.0e-14, k 1.8e-14, epsilon 4.6e-14, nut 1.1e-14
//   porous     20 steps: alpha 7.8e-15, p_rgh 8.0e-15, U 9.9e-13, k 7.1e-13, epsilon 6.0e-13, nut 4.5e-14, jump 1.6e-12
//   cyclicWet  60 steps: alpha 2.4e-13, p_rgh 5.1e-13, U 3.2e-12, k 1.4e-12, epsilon 1.9e-13, nut 1.3e-12
//   porousWet  60 steps: alpha 6.1e-14, p_rgh 3.0e-13, U 1.1e-12, k 7.5e-13, epsilon 1.1e-13, nut 5.6e-13, jump 5.4e-12
//   cyclicWetExplicit  60 steps, MULESCorr off: alpha 3.5e-14, p_rgh 2.4e-13, U 4.0e-13, k 3.3e-13, epsilon 1.9e-13, nut 2.4e-13
struct Bounds
{
    scalar alpha;
    scalar prgh;
    scalar U;
    scalar k;
    scalar epsilon;
    scalar nut;
    scalar jump;
};
const Bounds CYCLIC{2e-13, 2e-13, 1.5e-12, 6e-13, 1.5e-12, 4e-13, 0};
const Bounds POROUS{3e-13, 3e-13, 3e-11, 2e-11, 2e-11, 1.5e-12, 5e-11};
const Bounds CYCLIC_WET{8e-12, 1.5e-11, 1e-10, 4e-11, 6e-12, 4e-11, 0};
const Bounds POROUS_WET{2e-12, 1e-11, 3e-11, 2.5e-11, 4e-12, 2e-11, 2e-10};
const Bounds CYCLIC_WET_EXPLICIT{1e-12, 7e-12, 1.2e-11, 1e-11, 6e-12, 7e-12, 0};

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
    scalar rel() const { return linf/std::fmax(refMax, scalar(1e-300)); }
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
        const vector e{a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z};
        d.linf = std::fmax(d.linf, mag(e));
        d.refMax = std::fmax(d.refMax, mag(b[i]));
    }
    return d;
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/damBreakPorousBaffle (cyclic baffle) ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <controlOfTimeDir> <cyclic|porous>\n",
                    argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string controlDir = argv[6];
    const std::string profile = argv[7];
    const bool porous = (profile == "porous" || profile == "porousWet");
    const bool explicitMules = (profile == "cyclicWetExplicit");
    const bool wet = (profile == "cyclicWet" || profile == "porousWet" || explicitMules);
    const Bounds& B = explicitMules ? CYCLIC_WET_EXPLICIT
                    : wet ? (porous ? POROUS_WET : CYCLIC_WET) : (porous ? POROUS : CYCLIC);
    std::printf("  profile: %s -- %s, %s\n", profile.c_str(),
                porous ? "p_rgh a porousBafflePressure, the cyclic with a jump" : "p_rgh a plain cyclic, the coupling alone",
                wet ? "the water column staged across the baffle" : "the column as shipped, air alone at the baffle");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> patches = buildPatches(m, g);
    const std::vector<FvPatch> uncoupled = patches;      // for the device loop, which must refuse
    attachCyclicCoupling(patches, m, g);
    const label nC = m.nCells();

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);
    check(explicitMules ? "brae ran the EXPLICIT limiter: MULESCorr is off" : "brae ran the semi-implicit limiter: MULESCorr is on",
          fin.alphaCtl.MULESCorr != explicitMules);
    check("brae ran the case turbulent, under kEpsilon",
          fin.turbulence.on && fin.turbulence.model == InterRasModel::KEpsilon);
    // THE PATH: the pair is coupled, in every field, and carries a jump exactly where the profile says
    std::size_t nCoupled = 0;
    std::size_t nCoupledFaces = 0;
    std::size_t nFieldsCoupled = 0;
    std::size_t nJump = 0;
    scalar jumpMax = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!patches[pi].coupled) continue;
        ++nCoupled;
        nCoupledFaces += static_cast<std::size_t>(patches[pi].size);
        nFieldsCoupled += fin.alpha1.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.U.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.p_rgh.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.turbulence.k.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.turbulence.epsilon.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.turbulence.nut.boundary[pi]->coupled() ? 1 : 0;
        if (const std::vector<scalar>* j = fin.p_rgh.boundary[pi]->coupledJump())
        {
            ++nJump;
            for (scalar v : *j) jumpMax = std::fmax(jumpMax, std::fabs(v));
        }
    }
    std::printf("  coupled patches: %zu, %zu faces; p_rgh patches with a jump: %zu, |jump| up to %.4e\n",
                nCoupled, nCoupledFaces, nJump, (double)jumpMax);
    // THE JUMP ITSELF against the `jump` OpenFOAM writes on the owner side: the last pressure assembly's
    scalar dJump = -1;
    scalar ofJump = 0;
    if (porous)
    {
        const FieldData<scalar> ofP = readField<scalar>(ofDir + "/p_rgh");
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].coupled || !patches[pi].owner) continue;
            const PatchFieldData<scalar>* bp = findPatchEntry(ofP.boundary, patches[pi]);
            const std::vector<scalar>* j = fin.p_rgh.boundary[pi]->coupledJump();
            if (!bp || !bp->hasJump || !j) continue;
            dJump = 0;
            std::size_t worst = 0;
            for (std::size_t i = 0; i < j->size(); ++i)
            {
                const scalar o = bp->jumpIsUniform ? bp->jumpUniform : bp->jumpValues[i];
                ofJump = std::fmax(ofJump, std::fabs(o));
                if (std::fabs((*j)[i] - o) >= dJump)
                {
                    dJump = std::fabs((*j)[i] - o);
                    worst = i;
                }
            }
            const scalar ow = bp->jumpIsUniform ? bp->jumpUniform : bp->jumpValues[worst];
            std::printf("  the owner's jump, %s in OpenFOAM's file: worst face %zu brae %.15e, OpenFOAM %.15e; "
                        "Linf %.4e of |jump| up to %.4e\n", bp->jumpIsUniform ? "uniform" : "NONUNIFORM", worst,
                        (double)(*j)[worst], (double)ow, (double)dJump, (double)ofJump);
        }
    }
    check("the baffle is ONE coupled pair", nCoupled == 2 && nCoupledFaces > 0);
    check("...coupled in all six fields, on both sides", nFieldsCoupled == 12);
    // WHAT CROSSES THE BAFFLE: the phase fraction on the pair's faces, at the start and at the end
    {
        const FieldData<scalar> a0 = readField<scalar>(startDir + "/alpha.water");
        std::size_t wetStart = 0;
        std::size_t wetEnd = 0;
        std::size_t partEnd = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].coupled) continue;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const scalar aStart = a0.internalUniform ? a0.internalUniformValue : a0.internalField[c];
                const scalar aEnd = fin.alpha1.internal[c];
                wetStart += (aStart > scalar(0.5)) ? 1 : 0;
                wetEnd += (aEnd > scalar(0.5)) ? 1 : 0;
                partEnd += (aEnd > scalar(1e-6) && aEnd < scalar(1) - scalar(1e-6)) ? 1 : 0;
            }
        }
        std::printf("  cells beside the baffle: %zu wet at the start, %zu at the end, %zu holding the interface at the end\n",
                    wetStart, wetEnd, partEnd);
        if (wet)
        {
            check("water stands on BOTH sides of the baffle and the free surface crosses it",
                  wetStart > 0 && wetStart < nCoupledFaces && partEnd > 0);
        }
        else
        {
            check("as shipped, nothing but air reaches the baffle in the gated run", wetStart == 0 && wetEnd == 0);
        }
    }
    if (porous)
    {
        check("the owner's jump is the one OpenFOAM wrote, face by face",
              dJump >= scalar(0) && ofJump > scalar(0) && dJump <= B.jump*ofJump);
        check("p_rgh carries a jump on both sides, and it is not zero", nJump == 2 && jumpMax > scalar(0));
    }
    else
    {
        check("no patch carries a jump", nJump == 0);
    }

    auto readCells = [&](const std::string& path)
    {
        const FieldData<scalar> fd = readField<scalar>(path);
        std::vector<scalar> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };
    auto readVectorCells = [&](const std::string& path)
    {
        const FieldData<vector> fd = readField<vector>(path);
        std::vector<vector> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };

    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofA = brae::gatecheck::readOfSolves(logPath, fin.alphaName);
    const std::vector<LinearSolveRecord> ofE = brae::gatecheck::readOfSolves(logPath, "epsilon");
    const std::vector<LinearSolveRecord> ofK = brae::gatecheck::readOfSolves(logPath, "k");
    // the FAIL-PROOF: nothing in compareSolves can pass on an empty parse
    check("OpenFOAM's log gave one epsilon and one k solve per step",
          ofE.size() == static_cast<std::size_t>(nSteps) && ofK.size() == ofE.size());
    if (!wet)
    {
        failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(2e-6), scalar(2e-6));
    }
    else
    {
        // THE WET PROFILES' PRESSURE SOLVES ARE PINNED AT 1e-13 (the script says why), and a PCG residual
        // that close to round-off lands on either side of the tolerance: MEASURED, 15 of 180 solves on
        // cyclicWet and 11 of 180 on porousWet one iteration apart, in both directions, the shorter run
        // ending between 8.88e-14 and 9.99e-14. So the counts are held to ONE, and a differing pair to a
        // shorter run that stopped within 15% of the tolerance -- two apart, or one apart from anywhere
        // else, fails. (The shared edge-stop rule in compareSolves asks for a thousandth, which holds at
        // the waves gate's residual level and not at this one's: the last iterations here reduce the
        // residual by a tenth each, on a number that is itself round-off of an O(1e3) pressure.)
        brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(2e-6), scalar(2e-6),
                                       scalar(-1), nullptr, false);
        const scalar tolP = fin.pSolveFinal.tol;
        std::size_t nSame = 0;
        std::size_t nEdge = 0;
        std::size_t nOther = 0;
        scalar worstInit = 0;
        for (std::size_t q = 0; q < ofP.size() && q < r.pSolves.size(); ++q)
        {
            const int d = r.pSolves[q].nIterations - ofP[q].nIterations;
            const LinearSolveRecord& shorter = d > 0 ? ofP[q] : r.pSolves[q];
            if (d == 0)
            {
                ++nSame;
            }
            else if (std::abs(d) == 1 && shorter.finalResidual <= tolP && shorter.finalResidual >= scalar(0.85)*tolP)
            {
                ++nEdge;
            }
            else
            {
                ++nOther;
            }
            worstInit = std::fmax(worstInit, std::fabs(r.pSolves[q].initialResidual - ofP[q].initialResidual)
                                                 /std::fmax(ofP[q].initialResidual, scalar(1e-300)));
        }
        std::printf("  p_rgh at tolerance %.1e: %zu of %zu counts equal, %zu one apart at an edge stop, %zu otherwise; "
                    "initial residual worst %.3e\n", (double)tolP, nSame, ofP.size(), nEdge, nOther, (double)worstInit);
        check("it ran as many p_rgh solves as OpenFOAM logged, three a step",
              r.pSolves.size() == ofP.size() && ofP.size() == static_cast<std::size_t>(3*nSteps));
        check("the staged tolerance was read", tolP > scalar(0) && tolP < scalar(1e-12));
        check("every p_rgh count is OpenFOAM's, or one apart at an edge stop", nOther == 0 && nSame > nEdge);
        check("...from OpenFOAM's initial residuals", worstInit < scalar(2e-8));
    }
    if (explicitMules)
    {
        // no MULESCorr, no implicit alpha equation: neither code solves one
        check("neither code solved an alpha equation", r.alphaSolves.empty() && ofA.empty());
    }
    else
    {
        failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                                   scalar(1e-10), scalar(1e-9));
    }
    failures += brae::gatecheck::compareSolves("host", r.epsilonSolves, ofE, nSteps, "epsilon",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));
    failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));

    // BEFORE any fmax: std::fmax drops a NaN -- see tests/device_gate_finite.cuh
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
    failures += brae::gatecheck::nonFinite("brae epsilon", fin.turbulence.epsilon.internal);
    failures += brae::gatecheck::nonFinite("brae nut", fin.turbulence.nut.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const std::vector<scalar> ofKf = readCells(ofDir + "/k");
    const std::vector<scalar> ofEf = readCells(ofDir + "/epsilon");
    const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
    check("OpenFOAM's fields have one value per cell",
          ofAlpha.size() == static_cast<std::size_t>(nC) && ofKf.size() == ofAlpha.size()
       && ofEf.size() == ofAlpha.size() && ofNut.size() == ofAlpha.size());

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    const Diff dK = compare(fin.turbulence.k.internal, ofKf);
    const Diff dE = compare(fin.turbulence.epsilon.internal, ofEf);
    const Diff dN = compare(fin.turbulence.nut.internal, ofNut);
    // WHERE the worst cell is, for each field: a gap that starts at one patch names its cause
    auto worstAt = [&](const char* name, auto diffOf)
    {
        scalar worst = 0;
        label at = -1;
        for (label c = 0; c < nC; ++c)
        {
            const scalar d = diffOf(c);
            if (d > worst)
            {
                worst = d;
                at = c;
            }
        }
        if (at >= 0)
        {
            std::printf("  worst %-7s cell %d at (%.4g %.4g %.4g): %.4e\n", name, (int)at, (double)g.C()[at].x,
                        (double)g.C()[at].y, (double)g.C()[at].z, (double)worst);
        }
    };
    worstAt("alpha", [&](label c) { return std::fabs(fin.alpha1.internal[c] - ofAlpha[c]); });
    worstAt("p_rgh", [&](label c) { return std::fabs(fin.p_rgh.internal[c] - ofPrgh[c]); });
    worstAt("U", [&](label c)
    {
        const vector e{fin.U.internal[c].x - ofU[c].x, fin.U.internal[c].y - ofU[c].y, fin.U.internal[c].z - ofU[c].z};
        return mag(e);
    });
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
    std::printf("  epsilon: relative %.4e   (epsilon up to %.4e)\n", (double)dE.rel(), (double)dE.refMax);
    std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dN.rel(), (double)dN.refMax);

    // the wall nut, which the cell comparison cannot see: nutkWallFunction's own values
    const FieldData<scalar> ofNutFd = readField<scalar>(ofDir + "/nut");
    scalar dWallNut = 0;
    scalar wallNutScale = 0;
    std::size_t nWallFaces = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        const PatchFieldData<scalar>* b = findPatchEntry(ofNutFd.boundary, patches[pi]);
        if (!b || b->valueUniform) continue;
        const Diff d = compare(fin.turbulence.nut.boundary[pi]->value(), b->values);
        dWallNut = std::fmax(dWallNut, d.linf);
        wallNutScale = std::fmax(wallNutScale, d.refMax);
        nWallFaces += b->values.size();
    }
    std::printf("  wall nut: Linf %.4e on %zu wall faces (nut_w up to %.4e)\n", (double)dWallNut,
                nWallFaces, (double)wallNutScale);

    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B.alpha);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B.prgh);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B.U);
    check("k agrees with OpenFOAM's relatively", dK.rel() < B.k);
    check("epsilon agrees with OpenFOAM's relatively", dE.rel() < B.epsilon);
    check("nut agrees with OpenFOAM's relatively", dN.rel() < B.nut);
    check("the wall's nut is OpenFOAM's nutkWallFunction value",
          nWallFaces > 0 && dWallNut <= B.nut*std::fmax(wallNutScale, scalar(1e-300)));

    // THE BAFFLE ITSELF: OpenFOAM writes a cyclic's patch values, so brae's are held to them
    const FieldData<scalar> ofPFd = readField<scalar>(ofDir + "/p_rgh");
    scalar dBaffleP = 0;
    scalar bafflePScale = 0;
    std::size_t nCompared = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!patches[pi].coupled) continue;
        const PatchFieldData<scalar>* bp = findPatchEntry(ofPFd.boundary, patches[pi]);
        if (!bp || !bp->hasValue) continue;
        const std::size_t nf = static_cast<std::size_t>(patches[pi].size);
        const std::vector<scalar> ofPb = bp->valueUniform ? std::vector<scalar>(nf, bp->uniformValue) : bp->values;
        const Diff dp = compare(fin.p_rgh.boundary[pi]->value(), ofPb);
        dBaffleP = std::fmax(dBaffleP, dp.linf);
        bafflePScale = std::fmax(bafflePScale, dp.refMax);
        nCompared += nf;
    }
    std::printf("  on the baffle: p_rgh_b Linf %.4e (|p_rgh_b| up to %.4e), %zu faces\n",
                (double)dBaffleP, (double)bafflePScale, nCompared);
    if (porous)
    {
        check("the baffle's p_rgh is OpenFOAM's written value, face by face, on both sides",
              nCompared == nCoupledFaces && dBaffleP <= B.prgh*std::fmax(dP.refMax, scalar(1e-300)));
    }
    else
    {
        // OpenFOAM writes a plain cyclic as its type alone, so the pair's own values have no oracle
        check("OpenFOAM writes no value on a plain cyclic, so none is compared", nCompared == 0);
    }

    // THE CONTROL, on the oracle
    const Diff dCtlU = compare(readVectorCells(controlDir + "/U"), ofU);
    const Diff dCtlP = compare(readCells(controlDir + "/p_rgh"), ofPrgh);
    std::printf("  CONTROL: OpenFOAM's %s answer against this profile's, U relative %.4e, p_rgh %.4e\n",
                porous ? "plain-cyclic" : "two-walls", (double)dCtlU.rel(), (double)dCtlP.rel());
    check("what the profile adds moves OpenFOAM's own U far more than brae is from it",
          dCtlU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dCtlU.rel() > scalar(1e-6));

    // THE DEVICE LOOP REFUSES, by name
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device refusal is not exercised)\n");
    }
    else
    {
        bool named = false;
        try
        {
            InterFields dev;
            runInterFoamDevice(caseDir, startDir, m, g, uncoupled, nSteps, false, &dev);
        }
        catch (const std::exception& e)
        {
            named = std::string(e.what()).find("porous_half0") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses the case and names the patch", named);

        // NO DEVICE ARM ON THIS CASE, and two separate reasons: the tutorial sets
        // `nOuterCorrectors 3`, which the device loop does not run, and its p_rgh carries a JUMP,
        // which the device carries but is not yet exact in. Both are refused by name. The periodic
        // fixture's `jump` profile (tests/interfoam_cyclic_vs_openfoam.sh) is where the jump path is
        // measured, on a case the device loop can otherwise run end to end.
    }

    std::printf("test_inter_baffle_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
