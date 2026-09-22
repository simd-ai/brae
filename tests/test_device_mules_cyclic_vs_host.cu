// MULES's limiter ACROSS A CYCLIC PAIR: the device's lambda against the host's, at identical inputs.
//
// A cyclic patch is not in the device mesh's boundary gather at all (device_mesh.cuh:41-44), so before
// this the device limiter saw a periodic mesh as one with a wall there: the pair's faces contributed no
// extrema, no flux budget and got no limiter. The host does three things with them, and each is
// transcribed into the device kernels rather than reasoned about:
//   the EXTREMA take the neighbour CELL's psi (mules_cpp.cu:181-185), not a stored patch value;
//   the BUDGETS take phiBD and phiCorr with the owner-face sign, as any patch face does (:199-203);
//   the LIMITER is this side's per-cell one (:311-318) and then the PAIR IS SYNCED to the smaller of
//   the two (:321-345) -- syncTools::syncFaceList with minEqOp, which is NOT a no-op in serial and is
//   what makes a periodic face behave as the one internal face it is.
//
// The two layouts differ and the mapping is the fiddly part: the host's boundary arrays hold every
// patch, coupled included; the device's hold only the uncoupled ones and keep the pair in a separate
// interface array in buildDeviceCyclic's order. The gate builds one set of inputs and hands each arm
// its own view of them.
//
// THE BOUND IS NOT BITWISE. The device gathers per cell in a different order from the host's face loops
// (device_mules.cuh says so for the internal faces already), so the budgets differ in the last bits and
// a limiter that is a ratio of them can differ by more. What must hold exactly is the STRUCTURE: the
// pair's two sides carry the SAME limiter after the sync, and a face OpenFOAM would leave at 1 is not
// pulled down.
//
// MEASURED on validation/cyclicChannel: the device's lambda is the host's to 2.8e-16 on the internal
// faces and 1.1e-16 on the pair, the limiter reaches 0 so it is biting, and both sides of all 20
// periodic faces carry the same limiter to the last bit. BROKEN ONCE EACH, and each leaves its own
// signature: the interface extrema and budgets dropped -- internal 3.9e-01, interface 2.5e-01; the
// interface limiter dropped -- interface lambda stays at 1.0 and the internal faces go 9.1e-01; the
// SYNC dropped -- the two sides of a face differ by 7.8e-01, which only the pair arm above sees.
//
// THE DONOR FLUX on the pair is gated here too, bit for bit: an uncoupled patch has phiBD overwritten
// by phiPsi, and a coupled face keeps upwind's own flux from the cell the flux leaves
// (MULESTemplates.C:605). BROKEN once, upwind taken from the wrong side: 2.9e-02 of a 3.9e-02 flux.
// That number is the fixture's doing -- psi is TILTED so the pair's two sides hold different values;
// with the interface running in y alone the same break moved the flux by 6.9e-18 and only the bitwise
// arm could see it.
//
// AND THE ALPHA FLUX ITSELF on the pair, all three schemes, bit for bit: a coupled face takes the
// scheme's own weight as an internal face does (LimitedScheme's calcLimiter, bLim[patchi].coupled()),
// with the neighbour CELL across the pair. What that arm found is a caller obligation, not a kernel
// defect: the vanLeer limiter reads fvc::grad(alpha), and a device gradient without the pair's own
// contribution is the gradient of a mesh with a wall there -- linear and upwind stayed exact while
// vanLeer went 1.6e-02 of a 3.9e-02 flux. deviceCyclicAddGrad after deviceGaussGrad is the fix, and
// device_alpha_flux.cuh now says so where the caller will read it.
//
// AND THE EXPLICIT SOLVE, which is where those fluxes become an alpha update: MULES::explicitSolve
// divides fvc::div(phiPsi), and fvc::div sums a coupled patch's flux into its face cell like any
// patch's (fvc.cu:548-550). Device against host, both meshes: 4.4e-16 and 1.1e-16 of a psi reaching
// 1.65. BROKEN once, the pair's flux left out of the divergence: 4.4e-01 -- alpha simply does not cross
// the pair, which is a mesh with a wall there and not a slower answer.
//
// CMULES's limiter is here too, and it is NOT the explicit one with a flag (notes A to D): no donor
// flux, the budget measured against psi as it stands, and a COUPLED face limited whichever way its flux
// goes -- the inlet test an uncoupled face gets does not apply to it (CMULESTemplates.C:516). Device
// against host: internal 5.6e-17 and 2.8e-17, interface 1.1e-16 and 0.0, with the limiter reaching
// 0.002 on the pair so it is biting. BROKEN once, the pair left out of its budgets: 5.6e-01 on both.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_field.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"
#include "mules_cpp.cuh"
#include "device_mules.cuh"
#include "device_alpha_flux.cuh"
#include "alpha_eqn_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
}   // namespace

int main(int argc, char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicChannel";
    std::printf("== MULES's limiter across a cyclic pair: the device against the host ==\n");

    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    attachCyclicCoupling(fvp, m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);

    const label nC = m.nCells(), nIf = m.nInternalFaces();
    std::size_t nCoupled = 0;
    for (const FvPatch& q : fvp) if (q.coupled) nCoupled += static_cast<std::size_t>(q.size);
    std::printf("  mesh: %d cells, %d internal faces, %zu coupled faces\n", (int)nC, (int)nIf, nCoupled);
    check("the fixture has a coupled pair", nCoupled > 0);
    if (!nCoupled) { std::printf("test_device_mules_cyclic_vs_host: %d failures\n", failures); return 1; }

    // psi in [0, 1] with an interface running through the mesh, as alpha is, and an old value that
    // differs from it -- MULES's budget is the difference.
    std::vector<scalar> psiCell(static_cast<std::size_t>(nC)), psiOld(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        // TILTED, on purpose: the pair joins x = min to x = max, so an interface that ran in y alone
        // would put nearly the same psi on both sides and the upwind branch would be unmeasurable in
        // value terms. MEASURED with it running in y only: taking upwind from the wrong side moved the
        // donor flux by 6.9e-18 and only the bitwise arm saw it.
        const scalar a = scalar(0.5)
                       + scalar(0.5)*std::tanh(scalar(6)*(scalar(0.55) - C.y + scalar(0.35)*C.x));
        psiCell[static_cast<std::size_t>(c)] = std::fmin(std::fmax(a, scalar(0)), scalar(1));
        psiOld[static_cast<std::size_t>(c)] =
            std::fmin(std::fmax(a + scalar(0.02)*std::sin(scalar(3)*C.x), scalar(0)), scalar(1));
    }
    // psi's patch fields, as interFoam's own buildInterFields makes them: a coupled cyclic patch gets a
    // CoupledCyclicPatchField, which reads its neighbour THROUGH the attached FvPatch. The halo-based
    // CyclicFvPatchField that buildCyclicField hands out overrides only the no-argument
    // patchNeighbourField, and the limiter's gradient asks for the live one.
    GeometricField<scalar> psi;
    psi.internal = psiCell;
    for (const FvPatch& q : fvp)
    {
        if (q.coupled)      psi.boundary.push_back(std::make_unique<CoupledCyclicPatchField<scalar>>(q));
        else if (q.type == "empty") psi.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else                psi.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    }
    psi.evaluateBoundary();
    // psi's patch values in the DEVICE's boundary order (coupled patches skipped), for the gradient the
    // vanLeer limiter takes
    std::vector<scalar> bndPsiForGrad;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;
        const std::vector<scalar>& pv = psi.boundary[pi]->value();
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            bndPsiForGrad.push_back(static_cast<std::size_t>(i) < pv.size()
                                    ? pv[static_cast<std::size_t>(i)] : scalar(0));
        }
    }

    // phiBD and phiCorr: a donor flux and an antidiffusive correction big enough that the limiter
    // actually bites somewhere -- a gate on a lambda that is 1 everywhere measures nothing.
    SurfaceScalarField phiBD, phiCorr;
    phiBD.internal.resize(static_cast<std::size_t>(nIf));
    phiCorr.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        phiBD.internal[static_cast<std::size_t>(f)] = scalar(0.02)*std::sin(scalar(0.9)*scalar(f));
        phiCorr.internal[static_cast<std::size_t>(f)] = scalar(0.05)*std::cos(scalar(1.3)*scalar(f));
    }
    phiBD.boundary.resize(fvp.size());
    phiCorr.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phiBD.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        phiCorr.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        if (!fvp[pi].coupled) continue;   // an uncoupled patch's phiCorr is zero by construction
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(i);
            phiBD.boundary[pi][k] = scalar(0.02)*std::sin(scalar(1.1)*scalar(i) + scalar(pi));
            phiCorr.boundary[pi][k] = scalar(0.06)*std::cos(scalar(0.8)*scalar(i) + scalar(pi));
        }
    }

    // ---- THE DONOR FLUX on the pair ---------------------------------------------------------------
    // Before the limiter there is the flux it limits. On an uncoupled patch phiBD is overwritten by
    // phiPsi, which is what makes phiCorr zero there; a COUPLED face keeps upwind's own flux from the
    // cell the flux leaves (MULESTemplates.C:605, mules_cpp.cu:88-94), so its correction is real and the
    // limiter has work to do on it.
    {
        SurfaceScalarField phiField, phiPsi;
        phiField.internal.assign(static_cast<std::size_t>(nIf), scalar(0));
        phiField.boundary.resize(fvp.size());
        phiPsi.internal.assign(static_cast<std::size_t>(nIf), scalar(0));
        phiPsi.boundary.resize(fvp.size());
        for (label f = 0; f < nIf; ++f)
        {
            phiField.internal[static_cast<std::size_t>(f)] = scalar(0.03)*std::sin(scalar(0.7)*scalar(f));
        }
        std::size_t nPos = 0, nNeg = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            phiField.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
            phiPsi.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
            if (!fvp[pi].coupled) continue;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const scalar v = scalar(0.04)*std::sin(scalar(1.7)*scalar(i) + scalar(0.5)*scalar(pi));
                phiField.boundary[pi][static_cast<std::size_t>(i)] = v;
                if (v >= 0) ++nPos; else ++nNeg;
            }
        }
        check("the interface flux changes sign, so both sides of the upwind branch are taken",
              nPos > 0 && nNeg > 0);

        SurfaceScalarField hostBD;
        cpu::MULES::boundedDonorFlux(phiField, psi, phiPsi, m, fvp, hostBD);

        DeviceCyclic cycD = buildDeviceCyclic(cyclics, g, fvp);
        {
            std::vector<scalar> flat;
            for (const CyclicInterface& c : cyclics)
            {
                for (std::size_t i = 0; i < c.faceCells.size(); ++i)
                {
                    flat.push_back(phiField.boundary[static_cast<std::size_t>(c.patch)][i]);
                }
            }
            cycD.phi.copyFrom(flat);
        }
        DeviceBuffer<scalar> dPsiCell(psiCell), devBD;
        deviceMulesDonorFluxCyclic(cycD, dPsiCell, devBD);
        std::vector<scalar> dbd;
        devBD.copyTo(dbd);

        std::vector<scalar> hbd;
        for (const CyclicInterface& c : cyclics)
        {
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                hbd.push_back(hostBD.boundary[static_cast<std::size_t>(c.patch)][i]);
            }
        }
        scalar worst = 0, scale = 0;
        for (std::size_t j = 0; j < dbd.size() && j < hbd.size(); ++j)
        {
            worst = std::fmax(worst, std::fabs(dbd[j] - hbd[j]));
            scale = std::fmax(scale, std::fabs(hbd[j]));
        }
        std::printf("  donor flux on the pair: worst |device - host| %.4e (fluxes up to %.4e)\n",
                    (double)worst, (double)scale);
        check("the device's donor flux on a coupled face IS the host's, bit for bit",
              dbd.size() == hbd.size() && worst == scalar(0));
        check("...and it is not identically zero, which an overwritten patch would be", scale > scalar(0));
    }

    const DeviceMesh dmEarly = buildDeviceMesh(m, g, fvp);

    // ---- THE ANTIDIFFUSIVE FLUX's higher-order half on the pair -------------------------------------
    // fvc::flux(phi, alpha, vanLeer) on a coupled face takes the SCHEME's own weight, as on an internal
    // face: LimitedScheme's calcLimiter has a bLim[patchi].coupled() branch, and the host arm follows it
    // (alpha_eqn_cpp.cu:263-303) with the neighbour CELL across the pair, the patch's delta and its
    // central weight. phiCorr is this flux less the donor one, so an interface that took the patch value
    // instead -- which is what an UNCOUPLED patch does -- would make the correction wrong, not absent.
    for (int sch = 0; sch < 3; ++sch)
    {
        const cpu::interFoam::AlphaFluxScheme hostScheme =
            (sch == 0) ? cpu::interFoam::AlphaFluxScheme::linear
                       : (sch == 1) ? cpu::interFoam::AlphaFluxScheme::upwind
                                    : cpu::interFoam::AlphaFluxScheme::vanLeer;
        const char* schemeName = (sch == 0) ? "linear" : (sch == 1) ? "upwind" : "vanLeer";

        SurfaceScalarField phiF;
        phiF.internal.assign(static_cast<std::size_t>(nIf), scalar(0));
        phiF.boundary.resize(fvp.size());
        for (label f = 0; f < nIf; ++f)
        {
            phiF.internal[static_cast<std::size_t>(f)] = scalar(0.03)*std::sin(scalar(0.7)*scalar(f));
        }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            phiF.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
            if (!fvp[pi].coupled) continue;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                phiF.boundary[pi][static_cast<std::size_t>(i)] =
                    scalar(0.04)*std::sin(scalar(1.7)*scalar(i) + scalar(0.5)*scalar(pi));
            }
        }

        SurfaceScalarField hostOut;
        cpu::interFoam::fluxWithScheme(phiF, psi, hostScheme, m, g, fvp, hostOut);

        DeviceCyclic cycF = buildDeviceCyclic(cyclics, g, fvp);
        {
            std::vector<scalar> flat;
            for (const CyclicInterface& c : cyclics)
                for (std::size_t i = 0; i < c.faceCells.size(); ++i)
                    flat.push_back(phiF.boundary[static_cast<std::size_t>(c.patch)][i]);
            cycF.phi.copyFrom(flat);
        }
        // the limiter's gradient, the same fvc::grad(alpha) the host's fluxWithScheme takes
        DeviceBuffer<scalar> dPsiC(psiCell), dPsiB(bndPsiForGrad), gx, gy, gz;
        deviceGaussGrad(dmEarly, dPsiC, dPsiB, gx, gy, gz);
        // ...AND the pair's own contribution to it. The device mesh keeps a cyclic patch out of the
        // boundary gather, so deviceGaussGrad alone is the gradient of a mesh with a wall there, and the
        // vanLeer limiter reads that gradient in the cells next to the pair. MEASURED without this:
        // linear and upwind exact, vanLeer 1.6e-02 of a 3.9e-02 flux.
        DeviceBuffer<scalar> dVcell(g.V());
        deviceCyclicAddGrad(cycF, dPsiC, dVcell, gx, gy, gz);
        DeviceBuffer<scalar> devOut;
        deviceAlphaCyclicFlux(cycF, sch, dPsiC, gx, gy, gz, devOut);

        std::vector<scalar> dv, hv;
        devOut.copyTo(dv);
        for (const CyclicInterface& c : cyclics)
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
                hv.push_back(hostOut.boundary[static_cast<std::size_t>(c.patch)][i]);
        scalar worst = 0, scale = 0;
        for (std::size_t j = 0; j < dv.size() && j < hv.size(); ++j)
        {
            worst = std::fmax(worst, std::fabs(dv[j] - hv[j]));
            scale = std::fmax(scale, std::fabs(hv[j]));
        }
        std::printf("  alpha flux on the pair (%s): worst |device - host| %.4e (up to %.4e)\n",
                    schemeName, (double)worst, (double)scale);
        check("the device's alpha flux on a coupled face IS the host's",
              dv.size() == hv.size() && worst <= scalar(1e-15)*std::fmax(scale, scalar(1e-300)));
    }

    cpu::MULES::Fields hf;           // interFoam's: rho, Sp, Su, psiMax, psiMin all the constants
    cpu::MULES::Controls hc;
    hc.nLimiterIter = 3;
    const scalar rDeltaT = scalar(100);

    cpu::MULES::Limiter hostLambda;
    cpu::MULES::limiter(hostLambda, rDeltaT, psi, psiOld, phiBD, phiCorr, hf, hc, m, g, fvp);

    // ---- the device's view of the same inputs ---------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);

    std::vector<scalar> bndPsi, bndPhiBD, bndPhiCorr;
    std::vector<int> bndFixes, bndFlag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;    // the device mesh skips these
        const std::vector<scalar>& pv = psi.boundary[pi]->value();
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(i);
            bndPsi.push_back(k < pv.size() ? pv[k] : scalar(0));
            bndPhiBD.push_back(phiBD.boundary[pi][k]);
            bndPhiCorr.push_back(phiCorr.boundary[pi][k]);
            bndFixes.push_back(psi.boundary[pi]->fixesValue() ? 1 : 0);
            bndFlag.push_back(fvp[pi].type == "empty" ? 1 : (fvp[pi].type == "wedge" ? 2 : 0));
        }
    }
    std::vector<scalar> ifPhiBD, ifPhiCorr;
    for (const CyclicInterface& c : cyclics)
    {
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            ifPhiBD.push_back(phiBD.boundary[static_cast<std::size_t>(c.patch)][i]);
            ifPhiCorr.push_back(phiCorr.boundary[static_cast<std::size_t>(c.patch)][i]);
        }
    }

    DeviceBuffer<scalar> dPsi(psiCell), dPsiOld(psiOld), dPsiBnd(bndPsi);
    DeviceBuffer<int> dFixes(bndFixes), dFlag(bndFlag);
    DeviceBuffer<scalar> dPhiBDInt(phiBD.internal), dPhiBDBnd(bndPhiBD);
    DeviceBuffer<scalar> dPhiCorrInt(phiCorr.internal), dPhiCorrBnd(bndPhiCorr);
    DeviceBuffer<scalar> dPhiBDIf(ifPhiBD), dPhiCorrIf(ifPhiCorr);
    DeviceMulesFields df;
    DeviceMulesControls dc;
    dc.nLimiterIter = hc.nLimiterIter;
    DeviceBuffer<scalar> lamInt, lamBnd, lamIf;
    deviceMulesLimiter(dm, (int)nIf, (int)bndPhiBD.size(), rDeltaT, dPsi, dPsiOld, dPsiBnd,
                       dFixes, dFlag, dPhiBDInt, dPhiBDBnd, dPhiCorrInt, dPhiCorrBnd, df, dc,
                       lamInt, lamBnd, &cyc, &dPhiBDIf, &dPhiCorrIf, &lamIf);

    std::vector<scalar> devInt, devIf;
    lamInt.copyTo(devInt);
    lamIf.copyTo(devIf);

    // ---- does the limiter BITE? ------------------------------------------------------------------
    scalar minInt = 1, minIf = 1;
    for (const scalar v : devInt) minInt = std::fmin(minInt, v);
    for (const scalar v : devIf)  minIf  = std::fmin(minIf, v);
    std::printf("  lambda reaches %.4f on the internal faces and %.4f on the pair\n",
                (double)minInt, (double)minIf);
    check("the limiter actually bites somewhere, so this gate is not comparing two fields of ones",
          minInt < scalar(0.999) || minIf < scalar(0.999));

    // ---- the pair's two sides agree, which is what the sync is for --------------------------------
    std::vector<label> twin;
    cyc.twin.copyTo(twin);
    scalar worstPair = 0;
    std::size_t nPaired = 0;
    for (std::size_t j = 0; j < devIf.size(); ++j)
    {
        if (twin[j] < 0) continue;
        ++nPaired;
        worstPair = std::fmax(worstPair, std::fabs(devIf[j] - devIf[static_cast<std::size_t>(twin[j])]));
    }
    std::printf("  the pair: %zu of %zu faces have a twin, worst |lambda - lambda_twin| %.4e\n",
                nPaired, devIf.size(), (double)worstPair);
    check("both sides of every periodic face carry the SAME limiter", nPaired == devIf.size());
    check("...to the last bit, which is what syncFaceList's minEqOp leaves", worstPair == scalar(0));

    // ---- against the host -------------------------------------------------------------------------
    scalar worstInt = 0;
    for (std::size_t f = 0; f < devInt.size() && f < hostLambda.internal.size(); ++f)
    {
        worstInt = std::fmax(worstInt, std::fabs(devInt[f] - hostLambda.internal[f]));
    }
    std::vector<scalar> hostIf;
    for (const CyclicInterface& c : cyclics)
    {
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            hostIf.push_back(hostLambda.boundary[static_cast<std::size_t>(c.patch)][i]);
        }
    }
    scalar worstIf = 0;
    for (std::size_t j = 0; j < devIf.size() && j < hostIf.size(); ++j)
    {
        worstIf = std::fmax(worstIf, std::fabs(devIf[j] - hostIf[j]));
    }
    std::printf("  vs the host: internal %.4e, interface %.4e\n", (double)worstInt, (double)worstIf);
    check("the device's internal-face limiter is the host's", worstInt <= scalar(1e-12));
    check("...and its interface limiter is the host's",
          devIf.size() == hostIf.size() && worstIf <= scalar(1e-12));

    // ---- CMULES's LIMITER on a periodic mesh -------------------------------------------------------
    // The corrected path is not the explicit one with a flag (device_mules.cuh, notes A to D): there is
    // no donor flux, the budget is measured against psi as it stands, and a COUPLED face is limited
    // whichever way its flux goes -- the inlet test an uncoupled face gets does not apply to it
    // (CMULESTemplates.C:516). Everything else about the pair is the same: the neighbour cell's psi in
    // the extrema, phiCorr in the budgets, this side's limiter on the face, then the sync.
    {
        SurfaceScalarField phiC, phiCorrC;
        phiC.internal.assign(static_cast<std::size_t>(nIf), scalar(0));
        phiCorrC.internal.assign(static_cast<std::size_t>(nIf), scalar(0));
        phiC.boundary.resize(fvp.size());
        phiCorrC.boundary.resize(fvp.size());
        for (label f = 0; f < nIf; ++f)
        {
            phiC.internal[static_cast<std::size_t>(f)] = scalar(0.02)*std::sin(scalar(0.9)*scalar(f));
            phiCorrC.internal[static_cast<std::size_t>(f)] = scalar(0.05)*std::cos(scalar(1.3)*scalar(f));
        }
        std::vector<scalar> ifCorr;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            phiC.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
            phiCorrC.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        }
        for (const CyclicInterface& c : cyclics)
        {
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                const scalar v = scalar(0.06)*std::cos(scalar(0.8)*scalar(i) + scalar(c.patch));
                phiCorrC.boundary[static_cast<std::size_t>(c.patch)][i] = v;
                phiC.boundary[static_cast<std::size_t>(c.patch)][i] =
                    scalar(0.02)*std::sin(scalar(1.1)*scalar(i) + scalar(c.patch));
                ifCorr.push_back(v);
            }
        }

        cpu::MULES::Limiter hostLam;
        cpu::MULES::limiterCorr(hostLam, rDeltaT, psi, phiC, phiCorrC, hf, hc, m, g, fvp);

        DeviceCyclic cycC = buildDeviceCyclic(cyclics, g, fvp);
        std::vector<scalar> bndCorr;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                bndCorr.push_back(phiCorrC.boundary[pi][static_cast<std::size_t>(i)]);
            }
        }
        DeviceBuffer<scalar> dPsiC2(psiCell), dPsiB2(bndPsiForGrad);
        DeviceBuffer<scalar> dCorrInt(phiCorrC.internal), dCorrBnd(bndCorr), dCorrIf(ifCorr);
        DeviceBuffer<int> dFix2(bndFixes), dFlag2(bndFlag);
        DeviceMulesFields dfC;
        DeviceMulesControls dcC;
        dcC.nLimiterIter = hc.nLimiterIter;
        DeviceBuffer<scalar> lamI, lamB, lamIf2;
        // the boundary VOLUMETRIC flux the outlet test reads (note C)
        std::vector<scalar> bndPhi;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                bndPhi.push_back(phiC.boundary[pi][static_cast<std::size_t>(i)]);
            }
        }
        DeviceBuffer<scalar> dPhiBnd2(bndPhi);
        deviceMulesLimiterCorr(dm, (int)nIf, (int)bndCorr.size(), rDeltaT, dPsiC2, dPsiB2, dFix2, dFlag2,
                               dPhiBnd2, dCorrInt, dCorrBnd, dfC, dcC, lamI, lamB,
                               &cycC, &dCorrIf, &lamIf2);
        std::vector<scalar> dLamI, dLamIf;
        lamI.copyTo(dLamI);
        lamIf2.copyTo(dLamIf);

        std::vector<scalar> hLamIf;
        for (const CyclicInterface& c : cyclics)
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
                hLamIf.push_back(hostLam.boundary[static_cast<std::size_t>(c.patch)][i]);
        scalar wI = 0, wIf = 0, minLam = 1;
        for (std::size_t f = 0; f < dLamI.size() && f < hostLam.internal.size(); ++f)
            wI = std::fmax(wI, std::fabs(dLamI[f] - hostLam.internal[f]));
        for (std::size_t j = 0; j < dLamIf.size() && j < hLamIf.size(); ++j)
        {
            wIf = std::fmax(wIf, std::fabs(dLamIf[j] - hLamIf[j]));
            minLam = std::fmin(minLam, dLamIf[j]);
        }
        std::printf("  CMULES: vs the host internal %.4e, interface %.4e (lambda reaches %.4f on the "
                    "pair)\n", (double)wI, (double)wIf, (double)minLam);
        check("the device's CMULES limiter is the host's on the internal faces", wI <= scalar(1e-12));
        check("...and on the pair", dLamIf.size() == hLamIf.size() && wIf <= scalar(1e-12));
        check("...and it bites there, so the arm is not comparing two fields of ones",
              minLam < scalar(0.999));
    }

    // ---- THE EXPLICIT SOLVE on a periodic mesh ---------------------------------------------------
    // MULES::explicitSolve divides fvc::div(phiPsi), and fvc::div sums a COUPLED patch's flux into its
    // face cell exactly as it sums any patch's (fvc.cu:548-550). The device mesh's boundary arrays hold
    // no cyclic face, so the pair's flux has to be added separately or the cells along it advance as if
    // the mesh had a wall there -- alpha simply does not cross.
    {
        SurfaceScalarField phiPsi;
        phiPsi.internal.assign(static_cast<std::size_t>(nIf), scalar(0));
        phiPsi.boundary.resize(fvp.size());
        for (label f = 0; f < nIf; ++f)
        {
            phiPsi.internal[static_cast<std::size_t>(f)] = scalar(0.02)*std::sin(scalar(1.3)*scalar(f));
        }
        std::vector<scalar> ifFlux;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            phiPsi.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        }
        for (const CyclicInterface& c : cyclics)
        {
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                const scalar v = scalar(0.05)*std::sin(scalar(2.1)*scalar(i) + scalar(c.patch));
                phiPsi.boundary[static_cast<std::size_t>(c.patch)][i] = v;
                ifFlux.push_back(v);
            }
        }

        std::vector<scalar> hostPsi;
        cpu::MULES::explicitSolve(rDeltaT, hostPsi, psiOld, phiPsi, hf, m, g, fvp);

        DeviceCyclic cycE = buildDeviceCyclic(cyclics, g, fvp);
        std::vector<scalar> bndFlux;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            for (label i = 0; i < fvp[pi].size; ++i) bndFlux.push_back(scalar(0));
        }
        DeviceBuffer<scalar> dPhiPsiInt(phiPsi.internal), dPhiPsiBnd(bndFlux), dIfFlux(ifFlux);
        DeviceBuffer<scalar> dPsiOldE(psiOld), devPsi;
        DeviceMulesFields dfE;
        deviceMulesExplicitSolve(dmEarly, rDeltaT, dPsiOldE, dPhiPsiInt, dPhiPsiBnd, dfE, devPsi,
                                 &cycE, &dIfFlux);
        std::vector<scalar> dp;
        devPsi.copyTo(dp);

        scalar worst = 0, scale = 0;
        for (label c = 0; c < nC; ++c)
        {
            const std::size_t k = static_cast<std::size_t>(c);
            worst = std::fmax(worst, std::fabs(dp[k] - hostPsi[k]));
            scale = std::fmax(scale, std::fabs(hostPsi[k]));
        }
        std::printf("  explicit solve: worst |device - host| %.4e (psi up to %.4e)\n",
                    (double)worst, (double)scale);
        check("the device's explicit solve IS the host's on a periodic mesh",
              worst <= scalar(1e-15)*std::fmax(scale, scalar(1e-300)));
    }

    std::printf("test_device_mules_cyclic_vs_host: %d failures\n", failures);
    return failures ? 1 : 0;
}
