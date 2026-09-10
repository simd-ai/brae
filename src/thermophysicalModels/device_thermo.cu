// device_thermo.cu -- kernels behind deviceThermoCorrect / deviceThermoRho.

#include "device_thermo.cuh"
#include "nsrds_functions.cuh"   // liquid property correlations (properties liquid)
#include "equation_of_state.cuh"
#include "thermo_model.cuh"
#include "transport_model.cuh"
#include "device_buffer.cuh"
#include "device_blas.cuh"   // deviceCopy/deviceHadamard, for thermo.rho()
#include "device_boundary.cuh"
#include "pcuda_compat.cuh"
#include <stdexcept>
#include <cstdio>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdlib>

namespace brae {

namespace {

constexpr int TPB = 256;

inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

}   // namespace

// One pass over the cells: invert he to get T, then evaluate every T-dependent property from it.
// Fused rather than split per property because each one is a handful of flops on a value already in a
// register -- splitting would re-read T from global memory four times for no benefit.
__device__
void thermoUpdateK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ p,
    const scalar* __restrict__ he,
    scalar* __restrict__ T,
    scalar* __restrict__ rho,
    scalar* __restrict__ psi,
    scalar* __restrict__ mu,
    scalar* __restrict__ alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar Ti = hConstHeToT(he[i], c);
    T[i] = Ti;

    // Bound p before it reaches the EOS. An unbounded iterate can go negative mid-solve, and
    // rho = p/(R T) would then hand a negative density to the next momentum predictor.
    const scalar pi = fmax(p[i], c.pMin);

    const scalar psii = perfectGasPsi(Ti, c);
    psi[i] = psii;
    // null rho -> this call is OF's thermo.correct(), which leaves the solver's rho field alone.
    if (rho) rho[i] = fmin(fmax(psii * pi, c.rhoMin), c.rhoMax);

    const scalar mui = transportMu(Ti, c);
    mu[i] = mui;
    alpha[i] = transportAlpha(mui, c);
}

// he from T, used once at startup: cases ship a T field, the energy equation wants he.
__device__
void heFromTK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ T,
    scalar* __restrict__ he)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    he[i] = hConstTToHe(T[i], c);
}

// hePsiThermo::calculate -- T, psi, mu, alpha. There is no rho parameter in OF's signature, so the
// kernel is handed a null rho and leaves the solver's density alone.
void deviceHePsiThermoCalculate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    {
        const int n = th.n; const scalar* pd = p.data(); const scalar* hed = th.he.data();
        scalar* Td = th.T.data(); scalar* psid = th.psi.data(); scalar* mud = th.mu.data(); scalar* alphad = th.alpha.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            thermoUpdateK(n, c, pd, hed, Td, nullptr, psid, mud, alphad); });
    }
    cudaCheck(cudaGetLastError(), "hePsiThermoCalculate");
}

// heRhoThermo::calculate -- T, psi, RHO, mu, alpha (heRhoThermo.C: rhoCells[celli] = mixture_.rho(p,T)).
void deviceHeRhoThermoCalculate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    {
        const int n = th.n; const scalar* pd = p.data(); const scalar* hed = th.he.data();
        scalar* Td = th.T.data(); scalar* rhod = th.rhoThermo.data();
        scalar* psid = th.psi.data(); scalar* mud = th.mu.data(); scalar* alphad = th.alpha.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            thermoUpdateK(n, c, pd, hed, Td, rhod, psid, mud, alphad); });
    }
    cudaCheck(cudaGetLastError(), "heRhoThermoCalculate");
}

// Liquid calculate(): invert the transported energy for T, then evaluate every property from THAT T.
//
// ORDER IS THE WHOLE POINT. T must be updated before the properties are, or the correlations are
// evaluated at the previous iteration's temperature and the whole update lags by one outer iteration --
// silently, because the fields would still look smooth and plausible.
//
// The previous T is Newton's initial guess, per cell, which is the physically natural continuation from
// the last SIMPLE iteration and is why the inversion takes 3-5 steps rather than restarting cold.
void deviceThermoLiquidCalculate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    const EnergyForm form = c.internalEnergy ? EnergyForm::sensibleInternalEnergy
                                             : EnergyForm::sensibleEnthalpy;
    DeviceBuffer<scalar> Tnew, residual;
    DeviceBuffer<label>  ok;
    // th.T is BOTH the guess and, after the copy below, the result. Written through a separate buffer so
    // a cell that fails to converge does not leave a half-updated temperature field behind.
    deviceH2OEnergyToT(form, th.he, &p, th.T, Tnew, ok, residual);

    // Failure must surface here, not as CFD instability three iterations later. One label copy per
    // correction; the detailed scan only runs when something actually failed.
    const std::vector<label> hOk = ok.host();
    int nBad = 0;
    for (label v : hOk) if (!v) ++nBad;
    if (nBad)
    {
        const std::vector<scalar> hT0 = th.T.host(), hT = Tnew.host();
        const std::vector<scalar> hHe = th.he.host(), hP = p.host(), hR = residual.host();
        int first = 0;
        while (first < static_cast<int>(hOk.size()) && hOk[first]) ++first;
        // BRAE_DUMP_EEQN=<dir>: WHICH cells failed, not just how many. Whether the failures form a
        // recognisable region (one layer against a wall, a plume off the inlet, or scattered) separates
        // a boundary-coefficient defect from a source/convection one far faster than any global norm.
        if (const char* dd = std::getenv("BRAE_DUMP_EEQN"))
        {
            std::error_code dec;
            std::filesystem::create_directories(dd, dec);
            std::ofstream o(std::string(dd) + "/bad_cells");
            o.precision(17);
            o << "# cell he p Tguess residual\n";
            for (std::size_t i = 0; i < hOk.size(); ++i)
                if (!hOk[i])
                    o << i << ' ' << hHe[i] << ' ' << hP[i] << ' ' << hT0[i] << ' ' << hR[i] << '\n';
        }
        char msg[512];
        std::snprintf(msg, sizeof msg,
            "brae: the liquid energy inversion failed to converge in %d of %d cells. First failure at "
            "cell %d: p = %.9g Pa, %s = %.9g J/kg, initial T = %.9g K, final T = %.9g K, "
            "relative residual = %.3e. The transported energy is not attainable by this substance "
            "anywhere in [%.2f, %.2f] K, which usually means the pressure or energy field diverged "
            "upstream rather than that the inversion is at fault.",
            nBad, static_cast<int>(hOk.size()), first, (double)hP[first],
            c.internalEnergy ? "e" : "h", (double)hHe[first],
            (double)hT0[first], (double)hT[first], (double)hR[first],
            (double)H2OLiquid::Tt, (double)H2OLiquid::Tc);
        throw std::runtime_error(msg);
    }

    deviceCopy(th.T, Tnew);
    deviceThermoLiquidProperties(th, c);   // Cp, mu, kappa, rhoThermo, alpha -- all from the NEW T
}

// basicThermo::correct() -> the mixture's calculate(). The virtual dispatch OF does at run time.
void deviceThermoCorrect(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c)
{
    // The gas path is not routed through the generalised inverter: it has a closed-form he->T and no
    // reason to run Newton, and leaving it literally untouched is what keeps the nine compressible
    // gates protected by construction rather than by testing.
    if (c.model == ThermoModel::liquidH2O) { deviceThermoLiquidCalculate(th, p, c); return; }
    if (c.rhoThermoType) deviceHeRhoThermoCalculate(th, p, c);
    else                 deviceHePsiThermoCalculate(th, p, c);
}

// thermo.rho(): psiThermo returns p_*psi_, rhoThermo returns the stored rho_.
void deviceThermoRho(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& rhoOut)
{
    if (th.n == 0) return;
    if (c.rhoThermoType) deviceCopy(rhoOut, th.rhoThermo);    // rhoThermo::rho() -> the stored rho_
    else                 deviceHadamard(rhoOut, p, th.psi);   // psiThermo::rho() -> p_*psi_
}

// rho = rhoPrev + a*(rho - rhoPrev), then rhoPrev = rho. Bounds are re-applied because relaxing toward a
// previous value cannot violate them, but relaxing AWAY from a clamped value can.
__device__
void rhoRelaxK(
    int n,
    scalar a,
    scalar rhoMin,
    scalar rhoMax,
    scalar* __restrict__ rho,
    scalar* __restrict__ rhoPrev)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar blended = rhoPrev[i] + a * (rho[i] - rhoPrev[i]);
    const scalar bounded = fmin(fmax(blended, rhoMin), rhoMax);
    rho[i] = bounded;
    rhoPrev[i] = bounded;
}

__device__
void rhoSeedPrevK(
    int n,
    const scalar* __restrict__ rho,
    scalar* __restrict__ rhoPrev)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rhoPrev[i] = rho[i];
}

// The same relaxation applied to an arbitrary rho buffer, for the BOUNDARY half of the solver's rho
// field. OF has one `rho` volScalarField and `rho.relax()` relaxes its internal AND boundary values
// together; brae stores the two separately, so the boundary one needs the identical treatment or the
// pressure equation's phiHbyA weighting sees an unrelaxed density. Same kernel, deliberately: two
// copies of this arithmetic is two places for them to diverge.
void deviceRhoRelaxBuffer(
    DeviceBuffer<scalar>& rho,
    DeviceBuffer<scalar>& rhoPrev,
    const ThermoCoeffs& c)
{
    const int n = static_cast<int>(rho.size());
    if (n == 0 || rhoPrev.size() != rho.size()) return;
    {
        const scalar a = c.relaxRho; const scalar rMin = c.rhoMin; const scalar rMax = c.rhoMax;
        scalar* rhod = rho.data(); scalar* rhoPrevd = rhoPrev.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            rhoRelaxK(n, a, rMin, rMax, rhod, rhoPrevd); });
    }
    cudaCheck(cudaGetLastError(), "rhoRelaxBuffer");
}

void deviceRhoRelax(
    DeviceThermo& th,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    {
        const int n = th.n; const scalar a = c.relaxRho; const scalar rMin = c.rhoMin; const scalar rMax = c.rhoMax;
        scalar* rhod = th.rho.data(); scalar* rhoPrevd = th.rhoPrev.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            rhoRelaxK(n, a, rMin, rMax, rhod, rhoPrevd); });
    }
    cudaCheck(cudaGetLastError(), "rhoRelax");
}

void deviceRhoSeedPrev(DeviceThermo& th)
{
    if (th.n == 0) return;
    {
        const int n = th.n; const scalar* rhod = th.rho.data(); scalar* rhoPrevd = th.rhoPrev.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            rhoSeedPrevK(n, rhod, rhoPrevd); });
    }
    cudaCheck(cudaGetLastError(), "rhoSeedPrev");
}

// alphat = rho*nut/Prt. Kept in its own kernel rather than folded into thermoUpdateK because nut is
// owned by the turbulence model and is only valid after correctTurbulence(), which runs at the END of the
// outer iteration -- whereas thermoUpdateK runs in the middle of it.
__device__
void alphatK(
    int n,
    scalar Prt,
    const scalar* __restrict__ rho,
    const scalar* __restrict__ nut,
    scalar* __restrict__ alphat)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    alphat[i] = rho[i] * nut[i] / Prt;
}

void deviceAlphat(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& nut,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    if (static_cast<int>(nut.size()) != th.n) return;   // laminar: no nut field, alphat stays zero
    {
        const int n = th.n; const scalar prt = c.Prt;
        const scalar* rhod = th.rho.data(); const scalar* nutd = nut.data(); scalar* alphatd = th.alphat.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            alphatK(n, prt, rhod, nutd, alphatd); });
    }
    cudaCheck(cudaGetLastError(), "alphat");
}

// ---------------------------------------------------------------------------------------------------
// BOUNDARY TEMPERATURE -- the one place he_b becomes T_b, and the argument every boundary property is
// then evaluated from.
//
// THIS MIRRORS OF's STRUCTURE, WHICH BRAE HAD LOST. heRhoThermo::calculate walks the patches ONCE and,
// per face, establishes T_b and then evaluates psi/rho/mu/alphah from (p_b, T_b). brae had split that
// into four independent kernels which each re-derived T_b from he_b by the perfect-gas formula. On the
// gas path the four agreed, so it merely cost three redundant divisions; on the liquid path every one
// of them was wrong, and identically wrong, which is the worst kind: consistent enough to look right.
//
// Measured, once the he boundary was corrected to the liquid value: hConstHeToT(-1.5641742e7) returned
// T_b = -21369.6 K, from which transportMu took sqrt of a negative and produced NaN in all 112000 cells.
// The gas inversion is exact and closed-form; the liquid one is Newton, and OF runs Newton at the
// boundary too (mixture_.THE(phe, pp, pT) is the same thermo::T as the cell path).
__device__
void TBoundaryGasK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ heBnd,
    scalar* __restrict__ TBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    TBnd[i] = hConstHeToT(heBnd[i], c);
}

// The liquid inversion, one Newton per face. The GUESS is the owner cell's temperature -- OF starts from
// the patch's own previous T_b, and the adjacent cell is the same quality of warm start while keeping
// this function stateless. It only sets the iteration count in any case: acceptance is on the energy
// residual (see nsrds_functions.cuh), so the answer does not depend on where the iteration started, and
// tests/test_hetot.cu proves recovery from the opposite end of the valid range.
__device__
void TBoundaryLiquidK(
    int n,
    EnergyForm form,
    const scalar* __restrict__ pBnd,     // null -> 0 Pa; only the internal-energy form reads it
    const scalar* __restrict__ heBnd,
    const label*  __restrict__ faceCell,
    const scalar* __restrict__ Tcell,
    const label*  __restrict__ fixMask,  // 1 where the T patch fixes a value (OF fvPatchField::fixesValue)
    const scalar* __restrict__ TFix,     // the prescribed T on those faces
    scalar* __restrict__ TBnd,
    scalar* __restrict__ heRef)          // dbHe.refValue, REWRITTEN on the fixed faces
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar pb = pBnd ? pBnd[i] : scalar(0);

    // OF heRhoThermo::calculate branches on pT.fixesValue(), and this is that branch. Where T is
    // prescribed, T IS THE INPUT: OF never inverts he_b there, it recomputes
    //     phe[facei] = mixture_.HE(pp[facei], pT[facei])
    // from the CURRENT boundary pressure, every calculate(). brae converted T -> he once at setup and
    // then inverted that frozen he_b at the new pressure, so a fixed-temperature wall drifted as p moved.
    //
    // Measured on squareBendLiq: inlet p ran 100 kPa -> 150 kPa during iteration 1, and since
    // e = h(T) - p/rho(T), a stale e_b is wrong by dp/rho = 50.25 J/kg, i.e. dT = dp/(rho*Cp) =
    // +0.012013 K. The inlet came out at 300.012019 K against OF's exact 300. Predicted and measured
    // agree to 6e-6 K, which is what identifies this as the cause rather than a tolerance.
    if (fixMask && fixMask[i])
    {
        const scalar Tb = TFix[i];
        TBnd[i] = Tb;                                  // exactly the prescribed value, never round-tripped
        if (heRef) heRef[i] = h2oEnergy(form, pb, Tb); // re-derived at the CURRENT p_b
        return;
    }
    const scalar T0 = (faceCell && Tcell) ? Tcell[faceCell[i]] : scalar(300);
    TBnd[i] = h2oEnergyToT(form, heBnd[i], pb, T0).T;
}

void deviceThermoTBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    const DeviceBuffer<scalar>* Tcell,
    const DeviceBuffer<label>*  TFixMask,
    const DeviceBuffer<scalar>* TFix,
    DeviceBuffer<scalar>& TBnd)
{
    if (dbHe.n == 0) return;
    DeviceBuffer<scalar> heBnd;
    deviceBCValue(dbHe, he, heBnd);
    TBnd.resize(dbHe.n);
    if (c.model == ThermoModel::liquidH2O)
    {
        DeviceBuffer<scalar> pBnd;
        const bool needP = c.internalEnergy && dbP.n == dbHe.n;
        if (needP) deviceBCValue(dbP, p, pBnd);
        const bool haveGuess = Tcell && Tcell->size() && dbHe.faceCell.size();
        const bool haveFix =
            TFixMask && TFix
            && TFixMask->size() == static_cast<std::size_t>(dbHe.n)
            && TFix->size()     == static_cast<std::size_t>(dbHe.n);
        // refValue is what a fixedValue he face contributes to the matrix, so refreshing it here is
        // what actually propagates the corrected energy into the next energy solve.
        const bool haveRef = dbHe.refValue.size() == static_cast<std::size_t>(dbHe.n);
        const int n = dbHe.n;
        const EnergyForm form = c.internalEnergy ? EnergyForm::sensibleInternalEnergy : EnergyForm::sensibleEnthalpy;
        const scalar* pBndD = needP ? pBnd.data() : nullptr;
        const scalar* heBndD = heBnd.data();
        const label* faceCellD = haveGuess ? dbHe.faceCell.data() : nullptr;
        const scalar* TcellD = haveGuess ? Tcell->data() : nullptr;
        const label* fixMaskD = haveFix ? TFixMask->data() : nullptr;
        const scalar* TFixD = haveFix ? TFix->data() : nullptr;
        scalar* TBndD = TBnd.data();
        scalar* heRefD = (haveFix && haveRef) ? dbHe.refValue.data() : nullptr;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            TBoundaryLiquidK(n, form, pBndD, heBndD, faceCellD, TcellD, fixMaskD, TFixD, TBndD, heRefD); });
        cudaCheck(cudaGetLastError(), "TBoundaryLiquid");
        return;
    }
    {
        const int n = dbHe.n; const scalar* heBndD = heBnd.data(); scalar* TBndD = TBnd.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            TBoundaryGasK(n, c, heBndD, TBndD); });
    }
    cudaCheck(cudaGetLastError(), "TBoundaryGas");
}

// rho_b, face by face -- OF mixture_.rho(p_b, T_b). Perfect gas: p/(R T), clamped as the cell update
// clamps. Liquid: the NSRDS rho(T) correlation, which has no pressure dependence at all and is NOT
// clamped, because rhoMin/rhoMax exist to keep a compressible EOS away from its singularity at T -> 0
// and a liquid correlation has none.
__device__
void rhoBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ pBnd,
    const scalar* __restrict__ TBnd,
    scalar* __restrict__ rhoBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (c.model == ThermoModel::liquidH2O)
    {
        rhoBnd[i] = H2OLiquid::rho(fmin(fmax(TBnd[i], H2OLiquid::Tt), H2OLiquid::Tc));
        return;
    }
    const scalar pb = fmax(pBnd[i], c.pMin);
    rhoBnd[i] = fmin(fmax(pb / (c.R * TBnd[i]), c.rhoMin), c.rhoMax);
}

void deviceThermoRhoBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    const DeviceBuffer<scalar>& TBnd,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& rhoBnd)
{
    if (dbP.n == 0) return;
    DeviceBuffer<scalar> pBnd;
    deviceBCValue(dbP, p, pBnd);
    rhoBnd.resize(dbP.n);
    {
        const int n = dbP.n; const scalar* pBndD = pBnd.data(); const scalar* TBndD = TBnd.data(); scalar* rhoBndD = rhoBnd.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            rhoBoundaryK(n, c, pBndD, TBndD, rhoBndD); });
    }
    cudaCheck(cudaGetLastError(), "rhoBoundary");
}

// mu_b = Sutherland(T_b), face by face -- OF's transport_.mu(patchi).
//
// Needed because the compressible momentum boundary wants muEff_b = mu_b + rho_b*nut_b, and the wall
// functions want the KINEMATIC nu_b = mu_b/rho_b. Both are per-face: a single scalar viscosity is only
// right for a constant-property incompressible case.
__device__
void muBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ TBnd,
    scalar* __restrict__ muBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Sutherland is sqrt(T)-based, so a liquid reaching it with the gas-inverted T_b took the square
    // root of a negative number: this branch is what turned the whole field into NaN.
    muBnd[i] = (c.model == ThermoModel::liquidH2O)
                 ? H2OLiquid::mu(fmin(fmax(TBnd[i], H2OLiquid::Tt), H2OLiquid::Tc))
                 : transportMu(TBnd[i], c);
}

void deviceThermoMuBoundary(
    const DeviceBuffer<scalar>& TBnd,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& muBnd)
{
    const int n = static_cast<int>(TBnd.size());
    if (n == 0) return;
    muBnd.resize(n);
    {
        const scalar* TBndD = TBnd.data(); scalar* muBndD = muBnd.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            muBoundaryK(n, c, TBndD, muBndD); });
    }
    cudaCheck(cudaGetLastError(), "muBoundary");
}

// nu_b = mu_b/rho_b, face by face -- OF turbulenceModel::nu(patchi), which is literally
// transport_.mu(patchi)/rho_.boundaryField()[patchi]. Every OF wall function reads this, so brae's wall
// functions need it too: in compressible flow it is a field, not the single number a constant-property
// incompressible case can get away with.
__device__
void nuBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ pBnd,
    const scalar* __restrict__ TBnd,
    scalar* __restrict__ nuBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (c.model == ThermoModel::liquidH2O)
    {
        const scalar Tb = fmin(fmax(TBnd[i], H2OLiquid::Tt), H2OLiquid::Tc);
        nuBnd[i] = H2OLiquid::mu(Tb) / H2OLiquid::rho(Tb);
        return;
    }
    const scalar pb = fmax(pBnd[i], c.pMin);
    const scalar rb = fmin(fmax(pb / (c.R * TBnd[i]), c.rhoMin), c.rhoMax);
    nuBnd[i] = transportMu(TBnd[i], c) / rb;
}

void deviceThermoNuBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    const DeviceBuffer<scalar>& TBnd,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& nuBnd)
{
    if (dbP.n == 0) return;
    DeviceBuffer<scalar> pBnd;
    deviceBCValue(dbP, p, pBnd);
    nuBnd.resize(dbP.n);
    {
        const int n = dbP.n; const scalar* pBndD = pBnd.data(); const scalar* TBndD = TBnd.data(); scalar* nuBndD = nuBnd.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            nuBoundaryK(n, c, pBndD, TBndD, nuBndD); });
    }
    cudaCheck(cudaGetLastError(), "nuBoundary");
}

// alphaEff AT BOUNDARY FACES = alpha_b + alphat_b, which is what OF's laplacian(alphaEff, he) uses on a
// patch (heThermo::alphaEff(patchi); CpByCpv = 1 for sensibleEnthalpy).
//
//   alphat_b = rho_b * nut_b / Prt_b       -- OF alphatWallFunction::updateCoeffs is operator==(rhow*tnutw/Prt_)
//   alpha_b  = mu_b / Pr
//
// Prt_b is PER FACE because the two Prt in a compressible case are different numbers: the wall function's
// own (default 0.85) on its patches, and the turbulence model's (default 1.0) everywhere else. nut_b is
// the wall-function nut on a wall face and the extrapolated cell nut elsewhere, matching what the momentum
// boundary already uses -- so momentum and energy see one consistent wall eddy viscosity.
__device__
void alphaEffBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ muBnd,
    const scalar* __restrict__ rhoBnd,
    const scalar* __restrict__ nutBnd,
    const scalar* __restrict__ prtBnd,
    const scalar* __restrict__ TBnd,
    scalar* __restrict__ alphaEffBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // transportAlpha, not mu/Pr: under sutherland OF uses the Eucken kappa/Cp and there is no Pr at all.
    // A LIQUID TAKES NEITHER. OF's alphah for these substances is kappa(T)/Cp(T) from the two NSRDS
    // correlations (liquidPropertiesI.H); the Eucken relation is a dilute-gas result and c.Pr is a gas
    // scalar the liquid path never sets, so both gas routes are unreachable here by construction.
    scalar alphaLam;
    if (c.model == ThermoModel::liquidH2O)
    {
        const scalar Tb = fmin(fmax(TBnd[i], H2OLiquid::Tt), H2OLiquid::Tc);
        alphaLam = H2OLiquid::kappa(Tb) / H2OLiquid::Cp(Tb);
    }
    else alphaLam = transportAlpha(muBnd[i], c);
    const scalar alphaTur = (nutBnd && prtBnd) ? rhoBnd[i] * nutBnd[i] / prtBnd[i] : scalar(0);
    // CpByCpv, as on the cell path (heThermo::alphaEff) -- gamma for sensibleInternalEnergy, 1 for h.
    alphaEffBnd[i] = thermoCpByCpv(c) * (alphaLam + alphaTur);
}

void deviceAlphaEffBoundary(
    const DeviceBuffer<scalar>& muBnd,
    const DeviceBuffer<scalar>& rhoBnd,
    const DeviceBuffer<scalar>* nutBnd,
    const DeviceBuffer<scalar>* prtBnd,
    const DeviceBuffer<scalar>& TBnd,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& alphaEffBnd)
{
    const int n = static_cast<int>(muBnd.size());
    if (n == 0) return;
    if (c.model == ThermoModel::liquidH2O && static_cast<int>(TBnd.size()) != n)
    {
        throw std::runtime_error(
            "brae: the liquid alphaEff boundary needs a boundary temperature field sized for the same "
            "faces as mu -- alphah is kappa(T)/Cp(T) for a liquid and cannot be recovered from mu.");
    }
    alphaEffBnd.resize(n);
    {
        const scalar* muBndD = muBnd.data(); const scalar* rhoBndD = rhoBnd.data();
        const scalar* nutBndD = (nutBnd && nutBnd->size() == muBnd.size()) ? nutBnd->data() : nullptr;
        const scalar* prtBndD = (prtBnd && prtBnd->size() == muBnd.size()) ? prtBnd->data() : nullptr;
        const scalar* TBndD = TBnd.data(); scalar* alphaEffBndD = alphaEffBnd.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            alphaEffBoundaryK(n, c, muBndD, rhoBndD, nutBndD, prtBndD, TBndD, alphaEffBndD); });
    }
    cudaCheck(cudaGetLastError(), "alphaEffBoundary");
}

// OF pressureControl::limit(p): clamp p into [pMin, pMax] after the pressure solve, before rho is
// recomputed from it (rhoSimpleFoam/pEqn.H:90).
//
// Not a cosmetic safety net. On the stock aerofoilNACA0012 tutorial (Mach 0.72) OF's own log reports
//   pressureControl: p max 2332401     (23x ambient)
//   pressureControl: p min -241053.81  (NEGATIVE)
// during start-up. A negative p gives a negative rho through the perfect-gas EOS, and the next momentum
// solve is NaN -- which is exactly what brae did on that case before this existed.
__device__
void clampPressureK(
    int n,
    scalar pMin,
    scalar pMax,
    scalar* __restrict__ p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = fmin(fmax(p[i], pMin), pMax);
}

void deviceLimitPressure(
    DeviceBuffer<scalar>& p,
    scalar pMin,
    scalar pMax)
{
    const int n = static_cast<int>(p.size());
    if (n == 0) return;
    {
        scalar* pd = p.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { clampPressureK(n, pMin, pMax, pd); });
    }
    cudaCheck(cudaGetLastError(), "limitPressure");
}

// Liquid he-from-T: e = h(T) - p/rho(T) for sensibleInternalEnergy, h(T) for sensibleEnthalpy.
// PRESSURE IS REQUIRED here, which the gas form never needed -- hConstTToHe is a pure function of T.
__device__
void liquidHeFromTK(
    int n,
    EnergyForm form,
    const scalar* __restrict__ p,
    const scalar* __restrict__ T,
    scalar* __restrict__ he)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    he[i] = h2oEnergy(form, p ? p[i] : scalar(0), T[i]);
}

void deviceThermoHeFromT(
    DeviceThermo& th,
    const ThermoCoeffs& c,
    const DeviceBuffer<scalar>* p)
{
    if (th.n == 0) return;
    // The startup seed of he from the case's T field. It had no liquid branch, so a liquid case seeded
    // he with the PERFECT GAS hConstTToHe using the unset scalar Cp/R defaults: measured on
    // squareBendLiq, e came out -84258 J/kg where OF's Es(1e5, 300 K) is -15850730 -- 1005*(300-298.15)
    // - 287*300 exactly. Every cell then failed the energy inversion on the first correct(), which is
    // how it was found: the inversion refused rather than quietly returning a clamped temperature.
    if (c.model == ThermoModel::liquidH2O)
    {
        const EnergyForm form = c.internalEnergy ? EnergyForm::sensibleInternalEnergy
                                                 : EnergyForm::sensibleEnthalpy;
        if (c.internalEnergy && !p)
            throw std::runtime_error(
                "brae: seeding he from T for a liquid with sensibleInternalEnergy needs the pressure "
                "field (e = h(T) - p/rho(T)); the caller passed none.");
        const int n = th.n; const scalar* pd = p ? p->data() : nullptr;
        const scalar* Td = th.T.data(); scalar* hed = th.he.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            liquidHeFromTK(n, form, pd, Td, hed); });
        cudaCheck(cudaGetLastError(), "liquidHeFromT");
        return;
    }
    {
        const int n = th.n; const scalar* Td = th.T.data(); scalar* hed = th.he.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () { heFromTK(n, c, Td, hed); });
    }
    cudaCheck(cudaGetLastError(), "heFromT");
}

// ---------------------------------------------------------------------------------------------------
// LIQUID PROPERTIES. One evaluation point for the whole NSRDS family, so there is exactly one place
// where T maps to (Cp, mu, kappa, rho) and the internal and boundary paths cannot drift apart.
//
// alpha for a liquid is kappa/Cp, NOT the gas path's mu/Pr: the liquid correlations give the thermal
// conductivity directly, so deriving it from a Prandtl number would discard kappa(T) and substitute a
// constant-Pr assumption that is not what the case asked for. (OF: thermophysicalProperties' alpha is
// kappa/Cp for these mixtures; the Pr route belongs to the const/sutherland transport models.)
__device__
void liquidPropsK(
    int n,
    const scalar* __restrict__ T,
    scalar* __restrict__ Cp,
    scalar* __restrict__ mu,
    scalar* __restrict__ kappa,
    scalar* __restrict__ rho,
    scalar* __restrict__ alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar t  = T[i];
    const scalar cp = H2OLiquid::Cp(t);
    const scalar k  = H2OLiquid::kappa(t);
    Cp[i]    = cp;
    mu[i]    = H2OLiquid::mu(t);
    kappa[i] = k;
    rho[i]   = H2OLiquid::rho(t);
    if (alpha) alpha[i] = k/cp;      // OF alpha = kappa/Cp  [kg/(m s)]
}

void deviceThermoLiquidProperties(
    DeviceThermo& th,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    if (c.model != ThermoModel::liquidH2O)
        return;   // the gas path owns its own scalars; silently doing nothing here would hide a miswire
    if (th.CpField.size() != static_cast<std::size_t>(th.n)) th.allocateLiquid();
    {
        const int n = th.n; const scalar* Td = th.T.data();
        scalar* Cpd = th.CpField.data(); scalar* mud = th.mu.data(); scalar* kappad = th.kappa.data();
        scalar* rhod = th.rhoThermo.data();
        scalar* alphad = th.alpha.size() == static_cast<std::size_t>(th.n) ? th.alpha.data() : nullptr;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            liquidPropsK(n, Td, Cpd, mud, kappad, rhod, alphad); });
    }
    cudaCheck(cudaGetLastError(), "liquidProperties");
}

// The same correlations at boundary FACES, from a boundary temperature field. Takes T_b directly rather
// than deriving it, so this stays independent of the he->T inversion (which is a later step) and can be
// tested on a temperature field alone.
void deviceThermoLiquidBoundary(
    const DeviceBuffer<scalar>& Tb,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& CpB,
    DeviceBuffer<scalar>& muB,
    DeviceBuffer<scalar>& kappaB,
    DeviceBuffer<scalar>& rhoB)
{
    const int n = static_cast<int>(Tb.size());
    if (n == 0 || c.model != ThermoModel::liquidH2O) return;
    CpB.resize(n);
    muB.resize(n);
    kappaB.resize(n);
    rhoB.resize(n);
    {
        const scalar* Tbd = Tb.data();
        scalar* CpBd = CpB.data(); scalar* muBd = muB.data(); scalar* kappaBd = kappaB.data(); scalar* rhoBd = rhoB.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            liquidPropsK(n, Tbd, CpBd, muBd, kappaBd, rhoBd, nullptr); });
    }
    cudaCheck(cudaGetLastError(), "liquidBoundary");
}

// h -> T for a whole field. Each cell inverts independently from its OWN initial guess, which is what
// lets thermo.correct() later pass the previous temperature per cell rather than one global guess.
//
// The per-cell convergence flag is returned rather than aborting on the device: a single cell failing to
// invert is a diagnosable condition (a bad enthalpy from a diverging pressure iterate, say), and the
// caller is better placed than the kernel to decide whether that is fatal. OF aborts because it is on
// the host and has one cell in hand; here the whole field is in flight at once.
__device__
void h2oEnergyToTK(
    int n,
    EnergyForm form,
    scalar tol,
    int maxIter,
    const scalar* __restrict__ target,
    const scalar* __restrict__ pf,      // null => enthalpy form, where p is unused
    const scalar* __restrict__ Tguess,
    scalar* __restrict__ T,
    label* __restrict__ ok,
    scalar* __restrict__ residual)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const HeToTResult r = h2oEnergyToT(form, target[i], pf ? pf[i] : scalar(0), Tguess[i], tol, maxIter);
    T[i] = r.T;
    if (ok)       ok[i]       = r.converged ? 1 : 0;
    if (residual) residual[i] = r.residual;
}

void deviceH2OEnergyToT(
    EnergyForm form,
    const DeviceBuffer<scalar>& target,
    const DeviceBuffer<scalar>* p,
    const DeviceBuffer<scalar>& Tguess,
    DeviceBuffer<scalar>& T,
    DeviceBuffer<label>& ok,
    DeviceBuffer<scalar>& residual,
    scalar tol,
    int maxIter)
{
    const int n = static_cast<int>(target.size());
    if (n == 0) return;
    T.resize(n);
    ok.resize(n);
    residual.resize(n);
    {
        const scalar* targetD = target.data();
        const scalar* pD = (p && p->size() == target.size()) ? p->data() : nullptr;
        const scalar* TguessD = Tguess.data();
        scalar* Td = T.data(); label* okd = ok.data(); scalar* residualD = residual.data();
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            h2oEnergyToTK(n, form, tol, maxIter, targetD, pD, TguessD, Td, okd, residualD); });
    }
    cudaCheck(cudaGetLastError(), "h2oEnergyToT");
}

// Enthalpy form, so the step-3 call sites are untouched by the generalisation.
void deviceH2OHToT(
    const DeviceBuffer<scalar>& hTarget,
    const DeviceBuffer<scalar>& Tguess,
    DeviceBuffer<scalar>& T,
    DeviceBuffer<label>& ok,
    DeviceBuffer<scalar>& residual,
    scalar tol,
    int maxIter)
{
    deviceH2OEnergyToT(EnergyForm::sensibleEnthalpy, hTarget, nullptr, Tguess, T, ok, residual, tol, maxIter);
}

} // namespace brae
