// brae interFoam on a FULLY PERIODIC mesh against REAL OpenFOAM's, on validation/interFoamCyclic.
//
// WHAT THIS CASE EXERCISES that the baffle gate (tests/interfoam_baffle_vs_openfoam.sh) does not: the
// pair is the DOMAIN's x boundary, not an internal baffle, so every field advects across it for the
// whole run and there is no wall behind it to damp what the coupling gets wrong. The water surface is
// staged as a STEP at that boundary -- 0.3 on one side, 0.15 on the other -- so alpha, its gradient,
// nHatf and the pressure all carry a jump through the pair from the first step.
//
// THE ORACLE is OpenFOAM's own fields at the same instant, ten FIXED steps of 2e-3 in, written at 17
// digits. THE CONTROL is OpenFOAM's own answer with the pair replaced by two WALLS: the same mesh,
// the same cells, the coupling gone.
//
// 800 cells on purpose. This fixture exists so the CYCLIC DEVICE PATH can be run end to end cheaply,
// and the device arm's state is asserted here too.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "cyclic_interface.cuh"
#include "inter_driver_cpp.cuh"
#include "inter_solve_log.cuh"
#include "device_gate_finite.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

// MEASURED, ten steps of 2e-3 (the bounds are about 30x):
//   alpha 1.9e-13, p_rgh 9.8e-12 relative, U 7.5e-13 relative, all 30 p_rgh iteration counts OpenFOAM's
const scalar B_ALPHA = 6e-12;
const scalar B_PRGH = 3e-10;
const scalar B_U = 2e-11;
// ...and the DEVICE arm's (about 30x again). MEASURED against OpenFOAM: alpha 4.2e-11, p_rgh 1.98e-11
// relative, U 5.7e-11 relative; against brae's own host arm 4.2e-11, 1.02e-11, 5.8e-11. The alpha
// figure is the device alpha solver's own stopping point -- with every solve pinned at 1e-16 the two
// arms' alpha agrees to 6.1e-14 -- and not a discretisation difference.
const scalar B_ALPHA_DEV = 1.5e-09;
const scalar B_PRGH_DEV = 6e-10;
const scalar B_U_DEV = 2e-09;
const scalar B_ALPHA_DEV_HOST = 1.5e-09;
const scalar B_PRGH_DEV_HOST = 6e-10;
const scalar B_U_DEV_HOST = 2e-09;
// THE CLOSURE'S OWN FIELDS on the `sst` and `les` profiles. MEASURED, ten steps: `sst` k 4.8e-13,
// omega 8.8e-14, nut 2.4e-12; `les` k 1.8e-13, nut 1.8e-13. The bound is 4x the worst of those and
// not the usual 30x, BECAUSE IT HAS TO WITNESS THE DEFECT IT WAS WRITTEN FOR: rebuilding DkEff on the
// pair's faces from the patch's own nut instead of keeping fvc::interpolate's value reads k 1.3e-11,
// omega 4.7e-11 and nut 7.7e-11, every one of which passes at 1e-10. Both arms are host runs of a
// fixed mesh from a fixed start, so there is no run-to-run spread to leave room for.
const scalar B_TURB = 1e-11;
// ...and the DEVICE arm's. MEASURED, ten steps: `sst` k 2.4e-11, omega 7.0e-12, nut 5.0e-10; `les`
// k and nut both 1.3e-11. The bound is 10x the worst, and it WITNESSES every way the pair has been
// dropped inside a closure so far: k 4.8e-05 / omega 6.7e-04 / nut 3.6e-03 with CDkOmega's two
// gradients built without the interface, and k 4.3e-01 / nut 2.8e-01 (and U 8.7e-03) with the LES
// closure handed no pair at all.
const scalar B_TURB_DEV = 5e-09;
// THE `outer` PROFILE'S ARE LOOSER, and the reason is the stopping point and not the loop. Three
// PIMPLE outer correctors run the p_rgh solve three times as often, each starting from where the last
// one stopped, so the case's own `tolerance 1e-12` is reached three times per step and its slack
// compounds: MEASURED, device against host at the case's tolerance alpha 1.9e-09, p_rgh 2.2e-09,
// U 2.2e-07 -- and with every solve pinned at 1e-16 the SAME comparison reads alpha 3.8e-13,
// p_rgh 1.8e-11 relative, U 9.2e-13. The host arm is OpenFOAM's to 3.2e-14 / 6.7e-13 / 1.3e-13
// either way, which is what says the loop is right.
// ...and the two CRANKNICOLSON profiles', which sit one order above the Euler ones for the reason
// `cnFull` does in interfoam_cn_vs_openfoam: the scheme's two-step recurrence carries round-off
// undamped, so the two arms' stopping points compound rather than cancel. MEASURED, ten steps:
// sstCN alpha 7.9e-12, p_rgh 5.7e-10, U 1.2e-10 (against the host arm 7.6e-12 / 6.3e-10 / 1.2e-10);
// lesCN 4.5e-12 / 4.4e-10 / 9.8e-11. The bound is ~5x, and it witnesses what it was written for by
// seven orders: with the pair's ddtCorr left at the EULER form these read alpha 2.7612e-03,
// p_rgh 7.8874e-03, U 8.2265e-02.
const scalar B_ALPHA_DEV_CN = 5e-11;
const scalar B_PRGH_DEV_CN = 3e-09;
const scalar B_U_DEV_CN = 1e-09;
const scalar B_ALPHA_DEV_OUTER = 6e-08;
const scalar B_PRGH_DEV_OUTER = 7e-08;
const scalar B_U_DEV_OUTER = 7e-06;

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
    std::printf("== brae interFoam vs OpenFOAM interFoam: a fully periodic two-phase channel ==\n");
    if (argc < 7)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <wallsOfTimeDir>"
                    " [<profile>]\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string wallsDir = argv[6];
    const std::string profile = (argc > 7) ? argv[7] : "MULESCorr";
    const bool jumpProfile = (profile == "jump");
    const bool outerProfile = (profile == "outer");
    // A CLOSURE ACROSS THE PAIR. `sst` is kOmegaSST with the wall-function family on the walls,
    // `les` is kEqn with the filter width the fixture's uniform cells give. The closure is what is
    // under test in those two, so its OWN fields are compared and not only the three the other
    // profiles share: a defect confined to k reaches U through nuEff alone and arrives divided by the
    // Reynolds number.
    const bool sstProfile = (profile == "sst" || profile == "sstCN" || profile == "sstLim"
                          || profile == "sstLimU" || profile == "sstLimDiv" || profile == "sstLsq");
    // `sstLimDiv` names `Gauss limitedLinear 1` for div(phi,k) and div(phi,omega) on a mesh with a
    // PAIR -- a combination brae could not run at all: fvm::div threw the moment it was handed a
    // scheme's weights and a coupled patch, and the device assembler refused it by name. OpenFOAM
    // gives the coupled patch the scheme's own weights there (gaussConvectionScheme.C:105-108).
    // `sstLimU` adds `cellLimited grad(U)` to the two the closure limits, ON BOTH ARMS. It read
    // U 8.9149e-03, k 3.3383e-02, nut 4.1315e-01 on the device before the defect it found was named:
    // KOmegaSSTInput carries the grad(U) limiter TWICE and the interFoam site filled only the coeffs
    // half, so the device closure's production ran on an unlimited gradient.
    // `sstLimDiv` is HOST ONLY. The device closure computes the pair's limited weights from inputs
    // measured identical to the host's (flux 4.1e-14, CD weights exactly, both cells' field 4.4e-16,
    // delta exactly, gradient 7.1e-14) and still lands the other way on 4 of 40 pair faces for omega
    // and 6 for k -- every one of them a face where gradf is exactly zero (NVDTVD's 1000x branch, the
    // scheme decided by the sign of a round-off gradcf) or one ulp. That is worth nut 7.1e-04 here, so
    // these fields cannot tell the tie from a defect; the arm is refused by name until its assembled
    // system is held against OpenFOAM's own, as RAS/waterChannel's is.
    // `sstLimDiv` runs on BOTH arms now, but its device arm is not asserted on FIELDS: at t = 0 this
    // fixture's k and omega are uniform, so NVDTVD's gradf is a cancellation on every face and the
    // limiter is decided by the last bit -- the two arms differ on 4 of 40 pair faces by up to 0.5,
    // worth nut 7.1e-04 after ten steps. What holds the device here is the ASSEMBLED SYSTEM from a
    // spun-up field (the `assembly` arm of tests/interfoam_cyclic_vs_openfoam.sh): the pair's own
    // off-diagonal 1.7e-10 against the host's, with the device's upwind run as the control.
    const bool hostOnlyProfile = (profile == "sstLimDiv");
    const bool lesProfile = (profile == "les" || profile == "lesCN");
    const bool turbProfile = sstProfile || lesProfile;
    // ...and the two CRANKNICOLSON profiles, whose control is the SAME case under Euler rather than a
    // walled pair: what they measure is that the scheme reaches the closure's own ddt and not only
    // the loop's.
    const bool cnProfile = (profile == "sstCN" || profile == "lesCN");

    std::printf("  profile: %s\n", profile.c_str());
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> patches = buildPatches(m, g);
    // the pair, which brae couples exactly as braeInterFoam.cu's own driver couples it
    attachCyclicCoupling(patches, m, g);
    const label nC = m.nCells();

    std::size_t nCoupled = 0;
    for (const FvPatch& q : patches)
    {
        if (q.coupled)
        {
            nCoupled += static_cast<std::size_t>(q.size);
        }
    }
    std::printf("  mesh: %d cells, %zu coupled faces\n", (int)nC, nCoupled);
    check("the mesh carries a periodic pair at all", nCoupled > 0);
    if (!nCoupled)
    {
        std::printf("test_inter_cyclic_vs_openfoam [%s]: %d failures\n", profile.c_str(), failures);
        return 1;
    }

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);

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
    // THE INITIAL-RESIDUAL BOUND IS THE SECOND CORRECTOR'S OWN SCALE. In step one OpenFOAM's three
    // solves start at 1, 9.916e-08 and 7.359e-12: the correctors after the first begin from a residual
    // that has already collapsed, and a residual is a difference of nearly equal O(1) quantities
    // normalised back to O(1), so round-off on it is ~1e-15 ABSOLUTE however small it gets. On 9.9e-08
    // that is 1.6e-08 relative, which is what this run measures -- the two codes' fields agree to
    // 1.9e-13. The ITERATION COUNTS are what discriminates here and they are asserted exactly: walling
    // the pair moves OpenFOAM's own step one from 70/32/4 to 64/50/1.
    // ...and with THREE outer correctors the same collapse happens three times per step, so the worst
    // over the run is 1.88e-06 where one corrector reads 1.64e-08. The ITERATION COUNTS are exact in
    // both -- 90 of 90 here -- which is what discriminates.
    // ...and the two CLOSURE profiles go one order deeper again, for the same reason and no other.
    // A closure raises nuEff, so the momentum matrix is better conditioned and the third corrector
    // starts from further down: OpenFOAM's step-one solves are 1 / 1.066e-06 / 2.071e-09 under
    // kOmegaSST and 1 / 7.785e-07 / 1.075e-09 under kEqn, where the laminar case reads
    // 1 / 9.916e-08 / 7.359e-12. The worst difference on those is 8.155e-07 and 2.096e-06 RELATIVE,
    // which is 1.69e-15 and 2.25e-15 ABSOLUTE -- one or two ulp of the O(1) the residual is
    // normalised to, on both arms of both profiles. The ITERATION COUNTS are what discriminates and
    // they are exact: 30 of 30 on each.
    const scalar stepOneBound = turbProfile ? scalar(5e-6) : scalar(5e-7);
    failures += brae::gatecheck::compareSolves(
        "host", r.pSolves, ofP, nSteps, "p_rgh", stepOneBound,
        outerProfile ? scalar(6e-5) : stepOneBound);

    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/" + fin.alphaName);
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B_ALPHA);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B_PRGH);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B_U);

    // ...AND THE CLOSURE'S OWN FIELDS, where the profile has one. k, omega and nut are read from
    // OpenFOAM's write and from brae's turbulence block, which is where the host closure leaves them.
    if (turbProfile)
    {
        check("brae built the closure this profile asks for", fin.turbulence.on);
        failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
        const std::vector<scalar> ofK = readCells(ofDir + "/k");
        const Diff dK = compare(fin.turbulence.k.internal, ofK);
        std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
        check("k agrees with OpenFOAM's relatively", dK.rel() < B_TURB);
        const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
        const Diff dN = compare(fin.turbulence.nut.internal, ofNut);
        std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dN.rel(), (double)dN.refMax);
        check("nut agrees with OpenFOAM's relatively", dN.rel() < B_TURB);
        if (sstProfile)
        {
            const std::vector<scalar> ofW = readCells(ofDir + "/omega");
            const Diff dW = compare(fin.turbulence.omega.internal, ofW);
            std::printf("  omega:   relative %.4e   (omega up to %.4e)\n",
                        (double)dW.rel(), (double)dW.refMax);
            check("omega agrees with OpenFOAM's relatively", dW.rel() < B_TURB);
        }
        // ...and the closure is not inert here: OpenFOAM's own nut has to be doing something to the
        // momentum, or every number above would pass with the closure's convection dropped.
        scalar nutMax = 0;
        for (const scalar v : ofNut) nutMax = std::fmax(nutMax, std::fabs(v));
        std::printf("  OpenFOAM's own nut reaches %.4e against a laminar nu of 1e-06\n", (double)nutMax);
        check("the closure is live on this fixture", nutMax > scalar(1e-7));
    }

    // THE FIXTURE CARRIES SOMETHING ACROSS THE PAIR, in OpenFOAM's own numbers: without a flux there
    // the coupling could be dropped entirely and every comparison above would still pass.
    {
        scalar worstU = 0;
        for (const FvPatch& q : patches)
        {
            if (!q.coupled)
            {
                continue;
            }
            for (label i = 0; i < q.size; ++i)
            {
                worstU = std::fmax(worstU, mag(ofU[static_cast<std::size_t>(q.faceCells[i])]));
            }
        }
        std::printf("  OpenFOAM's |U| in the pair's own cells reaches %.4e\n", (double)worstU);
        check("OpenFOAM moves fluid through the periodic pair", worstU > scalar(1e-3));
    }

    // THE CONTROL, on the oracle. For the plain profiles it is the same mesh with the pair replaced by
    // two WALLS; for `jump` it is OpenFOAM's own PLAIN-CYCLIC answer, which is the pair coupled and
    // nothing else -- so what the comparison shows there is the jump itself and not the coupling.
    const std::vector<scalar> wAlpha = readCells(wallsDir + "/alpha.water");
    const std::vector<vector> wU = readVectorCells(wallsDir + "/U");
    const Diff cA = compare(wAlpha, ofAlpha);
    const Diff cU = compare(wU, ofU);
    std::printf("  CONTROL: OpenFOAM with %s: alpha %.4e, U relative %.4e\n",
                outerProfile ? "nOuterCorrectors 1 (this case has 3)"
                : (cnProfile ? "the Euler ddt (this case runs CrankNicolson 0.9)"
                : (jumpProfile ? "the pair a PLAIN CYCLIC (no jump)" : "the pair two WALLS")),
                (double)cA.linf, (double)cU.rel());
    check(outerProfile ? "the OUTER CORRECTORS move OpenFOAM's own alpha far more than brae is from it"
          : (cnProfile ? "CRANKNICOLSON moves OpenFOAM's own alpha far more than brae is from it"
          : (jumpProfile ? "the JUMP moves OpenFOAM's own alpha far more than brae is from it"
                         : "walling the pair moves OpenFOAM's own alpha far more than brae is from it")),
          cA.linf > scalar(1000)*std::fmax(dA.linf, scalar(1e-16)));
    check("...and its U", cU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-16)));

    // THE DEVICE ARM, against the SAME OpenFOAM fields. It is not a refusal any more: the pressure
    // corrector, the alpha step and the momentum source all carry the pair. What it took, in the order
    // this fixture found it -- the host<->device boundary-array layout, the momentum matrix's own
    // interface coefficient in fvMatrix::H(), the Gauss-Seidel smoother applying the interface to its
    // right-hand side every sweep (OpenFOAM's GaussSeidelSmoother.C:117-143), fvc::ddtCorr on the pair,
    // and divDevReff's grad(U) and stress flux there. EVERY ONE of them is identically zero at step one,
    // where U and phi are zero at rest, which is why this gate has to run more than one step.
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    // ...AND ON THE TWO CLOSURE PROFILES, which is what makes them worth running twice: kOmegaSST and
    // LES kEqn refused a pair on the device until the five sites the kEpsilon closure carries were
    // transcribed into them.
    // ...AND ON THE TWO CRANKNICOLSON PROFILES, which took the scheme's two flux terms on the pair's
    // own array: phiCN = cnCoeff*phi + (1 - cnCoeff)*phi.oldTime() on the coupled faces
    // (alphaEqn.H:91-97), and the end-of-step un-blend of alphaPhi10 against ITS old level there
    // (:253-262), with the pair's rhoPhi taken beside the raw phi as the branch requires. The host
    // arm gets both for free: it walks phi.boundary and alphaPhi10.boundary whole, coupled patch
    // included, and the device keeps those faces in a third array.
    if (nDev > 0 && hostOnlyProfile)
    {
        std::printf("  (this profile is HOST ONLY -- see the note at hostOnlyProfile above: the device\n"
                    "   closure's limited weights across the pair differ from the host's only where\n"
                    "   OpenFOAM's own limiter is decided by the last bit, and this fixture's fields\n"
                    "   cannot tell that from a defect. Refused by name.)\n");
    }
    else if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device arm is not exercised)\n");
    }
    else
    {
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        check("the device arm ran the same number of steps", rd.steps == nSteps);
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        std::printf("  device alpha:   Linf %.4e\n", (double)eA.linf);
        std::printf("  device p_rgh:   relative %.4e\n", (double)eP.rel());
        std::printf("  device U:       relative %.4e\n", (double)eU.rel());
        const scalar bA = outerProfile ? B_ALPHA_DEV_OUTER : cnProfile ? B_ALPHA_DEV_CN : B_ALPHA_DEV;
        const scalar bP = outerProfile ? B_PRGH_DEV_OUTER : cnProfile ? B_PRGH_DEV_CN : B_PRGH_DEV;
        const scalar bU = outerProfile ? B_U_DEV_OUTER : cnProfile ? B_U_DEV_CN : B_U_DEV;
        check("the device's alpha agrees with OpenFOAM's absolutely", eA.linf < bA);
        check("the device's p_rgh agrees with OpenFOAM's relatively", eP.rel() < bP);
        check("the device's U agrees with OpenFOAM's relatively", eU.rel() < bU);
        // ...and THE CLOSURE'S OWN FIELDS, which is where a pair dropped inside the closure shows
        // first: k and omega reach U only through nuEff, so a gap of 1e-05 in U is a much larger one
        // in k. Compared against OpenFOAM and printed beside the host arm's, so a reader can tell a
        // closure defect from a loop defect without another run.
        if (turbProfile)
        {
            const std::vector<scalar> ofK = readCells(ofDir + "/k");
            const std::vector<scalar> ofNut = readCells(ofDir + "/nut");
            const Diff dvK = compare(dev.turbulence.k.internal, ofK);
            const Diff dvN = compare(dev.turbulence.nut.internal, ofNut);
            std::printf("  device k:       relative %.4e\n", (double)dvK.rel());
            std::printf("  device nut:     relative %.4e\n", (double)dvN.rel());
            check("the device's k agrees with OpenFOAM's relatively", dvK.rel() < B_TURB_DEV);
            check("...and its nut", dvN.rel() < B_TURB_DEV);
            if (sstProfile)
            {
                const std::vector<scalar> ofW = readCells(ofDir + "/omega");
                const Diff dvW = compare(dev.turbulence.omega.internal, ofW);
                std::printf("  device omega:   relative %.4e\n", (double)dvW.rel());
                check("...and its omega", dvW.rel() < B_TURB_DEV);
            }
        }

        // ...and against the HOST arm, which is the tighter question: the two run the same
        // discretisation, so what separates them is the port and not the scheme.
        const Diff hA = compare(dev.alpha1.internal, fin.alpha1.internal);
        const Diff hP = compare(dev.p_rgh.internal, fin.p_rgh.internal);
        const Diff hU = compare(dev.U.internal, fin.U.internal);
        std::printf("  device vs HOST: alpha %.4e, p_rgh %.4e relative, U %.4e relative\n",
                    (double)hA.linf, (double)hP.rel(), (double)hU.rel());
        check("...and with brae's own host arm, which runs the same discretisation",
              hA.linf < bA && hP.rel() < bP && hU.rel() < bU);
    }

    std::printf("test_inter_cyclic_vs_openfoam [%s]: %d failures\n", profile.c_str(), failures);
    return failures == 0 ? 0 : 1;
}
