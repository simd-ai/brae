// brae's interFoam across a CODED cyclicACMI BAFFLE against REAL OpenFOAM's, on RAS/damBreakLeakage.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps, plus its solver log --
// tests/interfoam_leakage_vs_openfoam.sh stages it and carries the measurements. The baffle is a
// createBaffles cyclicACMI pair whose non-overlap patches are symmetry planes, and whose `scale` is a
// coded PatchFunction1: zero everywhere until t > 0.5, then one on the two faces whose centres lie in
// 0.07 < y < 0.1. The water column stands behind the closed baffle and leaks through the opening.
// TWO CONTROLS, both OpenFOAM's own answer at the same instant: the baffle never opened (the scale's
// time threshold moved past the run) and the baffle opened on EVERY face (the face selection dropped).
// The script runs this twice: from t = 0, and RESTARTED from OpenFOAM's own written state at t = 0.49 --
// the start time is the start directory's, which is what the coded scale's this->time() reads.
//
// THE DEVICE LOOP MUST REFUSE the case, naming the patch.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "cyclic_acmi_cpp.cuh"
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

// MEASURED, 520 fixed steps of 1e-3 from t = 0, the baffle opening at step 500 (bounds about 30x):
//   alpha 3.4e-13, p_rgh 1.5e-13, U 4.9e-12, k 9.2e-13, epsilon 2.2e-13, nut 3.5e-13; the flux through
//   the four baffle patches 3.2e-14 of its largest; p_rgh initial residuals 2.8e-14 absolute while shut
//   and 2.3e-10 relative once open.
const scalar B_ALPHA = 1e-11;
const scalar B_PRGH = 5e-12;
const scalar B_U = 1.5e-10;
const scalar B_K = 3e-11;
const scalar B_EPSILON = 7e-12;
const scalar B_NUT = 1e-11;
const scalar B_PHI = 1e-12;
// THE DEVICE ARM'S OWN BOUNDS -- not the host's, and the reason is measured, not assumed. The device is
// DETERMINISTIC on this case (a second run from fresh state differs by exactly 0) and every stage of the
// pair agrees with the host loop to 1e-13 on the opening step (phiHbyA, phig, the flux correction and
// phi on the pair's own faces). What separates the arms enters one step later, in ONE near-dry cell
// under the jet: alpha 2.483113e-09 on the device against 2.483522e-09 on the host, 4e-13 apart -- the
// device's MULES gathers its limiter budgets in another order (device_mules.cuh says what cannot be
// bit-identical) -- and the density ratio of 1000 turns that into 4e-10 of rho in a cell of air, which
// is the momentum diagonal there and so U. It is not the pressure solve's stopping point (the gap is
// the same to four digits with p_rgh at 1e-11) and not rest noise carried through the opening (host
// against host under that perturbation grows 18x across it; this grows 2900x). MEASURED against
// OpenFOAM: alpha 3.6e-11, p_rgh 2.0e-11, U 1.3e-09, k 1.0e-10, epsilon 3.9e-11, nut 5.6e-11, the flux
// through the baffle 9.2e-12 of itself -- the same on the run from 0 and on the restart. The bounds are
// five times that, EXCEPT k and the baffle's flux, which are 2.5 and 2.2 times: those two are what
// witnesses the step's rescale POINT on this arm (phic formed after the rescale reads k 5.8e-10 and the
// flux 2.9e-11, with U at 1.6e-09 -- inside this arm's own floor, where the host arm sees it in U).
// WHAT THE BOUNDS CANNOT SEE is said in the script's header, fail-proof by fail-proof.
const scalar D_ALPHA = 2e-10;
const scalar D_PRGH = 1e-10;
const scalar D_U = 7e-9;
const scalar D_K = 2.5e-10;
const scalar D_EPSILON = 2e-10;
const scalar D_NUT = 3e-10;
const scalar D_PHI = 2e-11;
const scalar B_PRES_CLOSED = 8e-13;
const scalar B_PRES_OPEN = 7e-9;

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/damBreakLeakage (coded cyclicACMI baffle) ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <closedOfTimeDir> "
                    "<allOpenOfTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string closedDir = argv[6];
    const std::string allOpenDir = argv[7];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> patches = buildPatches(m, g, /*mirrorACMI=*/true);
    // for the device loop, which must refuse: the geometry and patches before any coupling
    const FvGeometry gRaw = g;
    const std::vector<FvPatch> uncoupled = patches;
    // the start time is the start directory's, as the driver's clock
    const scalar t0 = static_cast<scalar>(std::stod(std::filesystem::path(startDir).filename().string()));
    cpu::cyclicACMI::Interfaces acmi = cpu::cyclicACMI::setup(m, g, patches, t0);
    attachCyclicCoupling(patches, m, g);
    const label nC = m.nCells();

    MutableMesh mm;
    mm.m = &m;
    mm.g = &g;
    mm.patches = &patches;
    mm.acmi = &acmi;
    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/false, &fin,
                                     /*endTime=*/scalar(1e30), /*pressureTaps=*/nullptr, &mm);
    check("brae ran the same number of steps", r.steps == nSteps);
    std::printf("  brae's time after %d steps: %.17g\n", (int)r.steps, (double)r.time);
    check("brae ran the semi-implicit limiter: MULESCorr is on", fin.alphaCtl.MULESCorr);
    check("brae ran the case turbulent, under kEpsilon",
          fin.turbulence.on && fin.turbulence.model == InterRasModel::KEpsilon);

    // THE PATH: one coded pair, coupled in every field, its masks where the code puts them
    check("the mesh has ONE cyclicACMI pair, scaled by a coded PatchFunction1",
          acmi.sides().size() == 2 && acmi.scaled() && acmi.sides()[0].coded && acmi.sides()[1].coded);
    std::size_t nFieldsCoupled = 0;
    std::size_t nOpen = 0;
    std::size_t nClosed = 0;
    std::size_t nOther = 0;
    std::size_t nOpenInBand = 0;
    for (const cpu::cyclicACMI::Side& s : acmi.sides())
    {
        const std::size_t pi = static_cast<std::size_t>(s.patch);
        nFieldsCoupled += fin.alpha1.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.U.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.p_rgh.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.turbulence.k.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.turbulence.epsilon.boundary[pi]->coupled() ? 1 : 0;
        nFieldsCoupled += fin.turbulence.nut.boundary[pi]->coupled() ? 1 : 0;
        for (std::size_t i = 0; i < s.scaledMask.size(); ++i)
        {
            const scalar mk = s.scaledMask[i];
            const scalar y = patches[pi].Cf[i].y;
            if (mk == scalar(1) - cpu::cyclicACMI::tolerance)
            {
                ++nOpen;
                nOpenInBand += (y > scalar(0.07) && y < scalar(0.1)) ? 1 : 0;
            }
            else if (mk == cpu::cyclicACMI::tolerance)
            {
                ++nClosed;
            }
            else
            {
                ++nOther;
            }
        }
    }
    std::printf("  ACMI masks at the end: %zu faces open, %zu closed, %zu between (both sides)\n",
                nOpen, nClosed, nOther);
    check("...coupled in all six fields, on both sides", nFieldsCoupled == 12);
    check("...and at the end exactly the faces the code names are open, on both sides",
          nOpen == 4 && nOpenInBand == 4 && nOther == 0 && nClosed + nOpen == 26);

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
    // THE PRESSURE SOLVES ARE PINNED at 1e-13, relTol 0 (the script says why), and a PCG residual that
    // close to round-off lands on either side of the tolerance: MEASURED, 41 of 1560 counts one apart,
    // both directions. So a count is OpenFOAM's or one apart where the shorter run stopped within 15% of
    // the tolerance (the wet baffle gate's rule). The INITIAL residuals are held in two phases. While the
    // baffle is shut the column stands at rest and its residuals are round-off, 1e-10 to 1e-12 of the
    // first solve's: MEASURED 3.5e-03 apart relative, 2.8e-14 absolute, so they are held absolutely.
    // Once it opens they are real, 1e-4 and up, and are held relatively: MEASURED 2.3e-10.
    brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(2e-6), scalar(2e-6),
                                   scalar(-1), nullptr, false);
    {
        // the first step whose time is past the coded scale's 0.5 -- accumulated as both clocks do
        label openStep = -1;
        scalar t = t0;
        for (label k = 1; k <= nSteps; ++k)
        {
            t += fin.deltaT;
            if (openStep < 0 && t > scalar(0.5))
            {
                openStep = k;
            }
        }
        const scalar tolP = fin.pSolveFinal.tol;
        const std::size_t perStep = ofP.size()/static_cast<std::size_t>(std::max(nSteps, label(1)));
        std::size_t nSame = 0;
        std::size_t nEdge = 0;
        std::size_t nOther = 0;
        scalar closedAbs = 0;
        scalar openRel = 0;
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
            const label step = static_cast<label>(q/std::max(perStep, std::size_t(1))) + 1;
            const scalar dr = std::fabs(r.pSolves[q].initialResidual - ofP[q].initialResidual);
            if (step < openStep)
            {
                closedAbs = std::fmax(closedAbs, dr);
            }
            else
            {
                openRel = std::fmax(openRel, dr/std::fmax(ofP[q].initialResidual, scalar(1e-300)));
            }
        }
        std::printf("  p_rgh at tolerance %.1e: %zu of %zu counts equal, %zu one apart at an edge stop, %zu otherwise; "
                    "the baffle opens at step %d; initial residuals %.3e apart absolutely while shut, %.3e "
                    "relatively once open\n", (double)tolP, nSame, ofP.size(), nEdge, nOther, (int)openStep,
                    (double)closedAbs, (double)openRel);
        check("it ran as many p_rgh solves as OpenFOAM logged, three a step",
              r.pSolves.size() == ofP.size() && ofP.size() == static_cast<std::size_t>(3*nSteps));
        check("the staged tolerance was read", tolP > scalar(0) && tolP < scalar(1e-12));
        check("the baffle opened inside the run", openStep > 0 && openStep < nSteps);
        check("every p_rgh count is OpenFOAM's, or one apart at an edge stop", nOther == 0 && nSame > nEdge);
        check("...from OpenFOAM's round-off residuals while the baffle is shut", closedAbs < B_PRES_CLOSED);
        check("...and from its initial residuals once it is open", openRel < B_PRES_OPEN);
    }
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               scalar(1e-10), scalar(1e-9));
    failures += brae::gatecheck::compareSolves("host", r.epsilonSolves, ofE, nSteps, "epsilon",
                                               scalar(1e-10), scalar(1e-10), scalar(1e-5));
    // k's FINAL residual is not held: pinned at 1e-13, it ends near 5.7e-14, where round-off sets the
    // last digits (MEASURED 1.1e-02 apart relatively, 6e-16 absolutely). What that arm exists for --
    // the same smoother, not a Krylov solver that also takes one sweep -- epsilon's holds exactly, under
    // the same `(U|k|epsilon).*` entry.
    failures += brae::gatecheck::compareSolves("host", r.kSolves, ofK, nSteps, "k",
                                               scalar(1e-10), scalar(1e-10));

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

    // THE FLUX THROUGH THE BAFFLE, face by face, against the phi OpenFOAM writes: on the coupled patch it
    // is what leaks, on the symmetry planes what the mask left them -- the quantity the area split sets
    const FieldData<scalar> ofPhi = readField<scalar>(ofDir + "/phi");
    scalar dPhi = 0;
    scalar phiScale = 0;
    std::size_t nPhiFaces = 0;
    for (const cpu::cyclicACMI::Side& s : acmi.sides())
    {
        for (const label pi : {s.patch, s.nonOverlap})
        {
            const FvPatch& q = patches[static_cast<std::size_t>(pi)];
            const PatchFieldData<scalar>* b = findPatchEntry(ofPhi.boundary, q);
            if (!b)
            {
                continue;
            }
            const std::size_t nf = static_cast<std::size_t>(q.size);
            const std::vector<scalar> ofb = b->valueUniform ? std::vector<scalar>(nf, b->uniformValue) : b->values;
            const Diff d = compare(fin.phi.boundary[static_cast<std::size_t>(pi)], ofb);
            std::printf("  phi on %-14s Linf %.4e (|phi| up to %.4e)\n", q.name.c_str(), (double)d.linf,
                        (double)d.refMax);
            dPhi = std::fmax(dPhi, d.linf);
            phiScale = std::fmax(phiScale, d.refMax);
            nPhiFaces += ofb.size();
        }
    }
    check("OpenFOAM wrote phi on the four baffle patches", nPhiFaces == 52);
    check("water leaks through the opening: phi on the coupled faces is not zero", phiScale > scalar(1e-8));

    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B_ALPHA);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B_PRGH);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B_U);
    check("k agrees with OpenFOAM's relatively", dK.rel() < B_K);
    check("epsilon agrees with OpenFOAM's relatively", dE.rel() < B_EPSILON);
    check("nut agrees with OpenFOAM's relatively", dN.rel() < B_NUT);
    check("the flux through the baffle is OpenFOAM's, face by face", dPhi <= B_PHI*phiScale);

    // THE CONTROLS, on the oracle
    const Diff dClosed = compare(readVectorCells(closedDir + "/U"), ofU);
    const Diff dAll = compare(readVectorCells(allOpenDir + "/U"), ofU);
    std::printf("  CONTROL: OpenFOAM with the baffle never opened, U relative %.4e\n", (double)dClosed.rel());
    std::printf("  CONTROL: OpenFOAM with the baffle opened on every face, U relative %.4e\n", (double)dAll.rel());
    check("opening the baffle moves OpenFOAM's own U far more than brae is from it",
          dClosed.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dClosed.rel() > scalar(1e-6));
    check("...and so does WHICH faces it opens",
          dAll.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)) && dAll.rel() > scalar(1e-6));

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
            runInterFoamDevice(caseDir, startDir, m, gRaw, uncoupled, nSteps, false, &dev);
        }
        catch (const std::exception& e)
        {
            named = std::string(e.what()).find("coupled_half0") != std::string::npos;
            std::printf("  device: %s\n", e.what());
        }
        check("the device loop refuses the pair handed to it UNCOUPLED, and names the patch", named);

        // THE DEVICE ARM, on the pair COUPLED. The host run above has moved g, patches and acmi to
        // the END state -- the rescale is in place -- so this arm starts from its own: the geometry
        // and patches as they stood before any coupling, set up again at the start time.
        FvGeometry gD = gRaw;
        std::vector<FvPatch> patchesD = uncoupled;
        cpu::cyclicACMI::Interfaces acmiD = cpu::cyclicACMI::setup(m, gD, patchesD, t0);
        attachCyclicCoupling(patchesD, m, gD);
        MutableMesh mmD;
        mmD.m = &m;
        mmD.g = &gD;
        mmD.patches = &patchesD;
        mmD.acmi = &acmiD;
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, gD, patchesD, nSteps, false, &dev,
                                                scalar(1e30), nullptr, &mmD);
        check("the device driver ran the same number of steps", rd.steps == nSteps);
        std::printf("  the device's time after %d steps: %.17g (host %.17g)\n", (int)rd.steps,
                    (double)rd.time, (double)r.time);
        check("...to the host loop's clock, bit for bit", rd.time == r.time);
        // both arms opened the same faces: the masks after the run
        bool sameMasks = acmiD.sides().size() == acmi.sides().size();
        for (std::size_t si = 0; sameMasks && si < acmi.sides().size(); ++si)
        {
            sameMasks = acmiD.sides()[si].scaledMask == acmi.sides()[si].scaledMask;
        }
        check("...and opened the faces the host loop opened", sameMasks);
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        const Diff eK = compare(dev.turbulence.k.internal, ofKf);
        const Diff eE = compare(dev.turbulence.epsilon.internal, ofEf);
        const Diff eN = compare(dev.turbulence.nut.internal, ofNut);
        std::printf("  DEVICE:  alpha %.4e, p_rgh %.4e, U %.4e, k %.4e, epsilon %.4e, nut %.4e\n",
                    (double)eA.linf, (double)eP.rel(), (double)eU.rel(), (double)eK.rel(),
                    (double)eE.rel(), (double)eN.rel());
        std::printf("  host:    alpha %.4e, p_rgh %.4e, U %.4e, k %.4e, epsilon %.4e, nut %.4e\n",
                    (double)dA.linf, (double)dP.rel(), (double)dU.rel(), (double)dK.rel(),
                    (double)dE.rel(), (double)dN.rel());
        // the flux through the baffle, on the device's own patches
        scalar ePhi = 0;
        for (const cpu::cyclicACMI::Side& sd : acmiD.sides())
        {
            for (const label pi : {sd.patch, sd.nonOverlap})
            {
                const FvPatch& q = patchesD[static_cast<std::size_t>(pi)];
                const PatchFieldData<scalar>* b = findPatchEntry(ofPhi.boundary, q);
                if (!b)
                {
                    continue;
                }
                const std::size_t nf = static_cast<std::size_t>(q.size);
                const std::vector<scalar> ofb = b->valueUniform ? std::vector<scalar>(nf, b->uniformValue)
                                                                : b->values;
                ePhi = std::fmax(ePhi, compare(dev.phi.boundary[static_cast<std::size_t>(pi)], ofb).linf);
            }
        }
        std::printf("  DEVICE phi through the baffle: Linf %.4e of %.4e (host %.4e)\n", (double)ePhi,
                    (double)phiScale, (double)dPhi);
        // held to the device arm's OWN bounds (D_*, at the top of this file, with the measurement)
        check("the device's alpha agrees with OpenFOAM's absolutely", eA.linf < D_ALPHA);
        check("...its p_rgh", eP.rel() < D_PRGH);
        check("...its U", eU.rel() < D_U);
        check("...its k", eK.rel() < D_K);
        check("...its epsilon", eE.rel() < D_EPSILON);
        check("...its nut", eN.rel() < D_NUT);
        check("...and its flux through the baffle, face by face", ePhi <= D_PHI*phiScale);
        check("opening the baffle moves OpenFOAM's own U far more than the device is from it",
              dClosed.rel() > scalar(1000)*std::fmax(eU.rel(), scalar(1e-14)));
    }

    std::printf("test_inter_leakage_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
