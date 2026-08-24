// device_energy.cu -- alphaEff assembly and the he solve.

#include "device_energy.cuh"
#include "nsrds_functions.cuh"
#include "stage_dump.cuh"
#include "device_scalar_transport.cuh"
#include "thermo_model.cuh"
#include "pcuda_compat.cuh"
#include <stdexcept>

namespace brae {

// Boundary he from boundary T. Linear for hConst, so bcType is untouched and only the value moves.
//
// The GRADIENT transforms too, and by a different rule than the value: OF's gradientEnergy sets
// gradient() = Cpv(p, Tw)*Tw.snGrad() (gradientEnergyFvPatchScalarField.C:111). For hConst,
// d(he)/dT is exactly Cpv -- Cv when the energy variable is sensibleInternalEnergy, Cp otherwise --
// so the chain rule reproduces OF's expression without needing Tw at all. Note Hf drops out: it
// shifts he by a constant and a constant has zero gradient. A fixedGradient T wall whose gradient
// was copied across UNSCALED would be wrong by a factor of Cpv, i.e. ~1005 for air -- so this is
// the difference between a heat-flux wall and an essentially adiabatic one.
__device__
void heBndFromTK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ refT,
    const scalar* __restrict__ gradT,
    scalar* __restrict__ refHe,
    scalar* __restrict__ gradHe)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    refHe[i] = hConstTToHe(refT[i], c);
    if (gradT) gradHe[i] = (c.internalEnergy ? thermoCv(c) : c.Cp) * gradT[i];
}

// The SAME conversion for a liquid, where both the value and the slope come from the correlations
// rather than from the perfect-gas scalars.
//
// THIS WAS THE ITERATION-1 EXPLOSION. A `fixedValue T` patch had its energy value computed by
// hConstTToHe, i.e. Cp*(T - Tref) - R*T on ThermoCoeffs' GAS defaults, which the liquid path never
// populates. squareBendLiq's walls (350 K) therefore carried he = -48361 J/kg where the correct
// Es(1e5, 350) is -15641742 J/kg -- an error of +1.56e7 J/kg imposed on every temperature boundary,
// which then diffused into the interior and drove 406 cells to an energy no liquid can attain.
//
// The gradient coefficient is d(he)/dT = Cpv, evaluated AT THE FACE TEMPERATURE. For this liquid OF
// sets CpMCv = 0, so Cpv is Cp(T_b) for both energy forms -- not the gas Cp - R (717.9 vs 4183, a
// factor of 5.8 on any fixedGradient/heat-flux temperature patch).
__device__
void heBndFromTLiquidK(
    int n,
    EnergyForm form,
    const scalar* __restrict__ pBnd,     // null -> 0 Pa; only the internal-energy form reads it
    const scalar* __restrict__ refT,
    const scalar* __restrict__ gradT,
    scalar* __restrict__ refHe,
    scalar* __restrict__ gradHe)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // PROJECT into the correlation's valid range before evaluating. refValue is only meaningful on
    // patches that FIX a value; on zeroGradient/extrapolated patches it is unset, and H2OLiquid::rho
    // computes pow(1 - T/Tc, 0.081), which is NaN for any T > Tc. Reading an unset refValue therefore
    // turned the whole he boundary into NaN -- measured: all 112000 cells NaN on the first correct().
    // The gas formula was a polynomial and silently absorbed the same garbage, which is why this only
    // appeared on the liquid path.
    const scalar Tb = fmin(fmax(refT[i], H2OLiquid::Tt), H2OLiquid::Tc);
    const scalar pb = pBnd ? pBnd[i] : scalar(0);
    refHe[i] = h2oEnergy(form, pb, Tb);
    if (gradT) gradHe[i] = h2oCpv(form, pb, Tb) * gradT[i];
}

void deviceEnergyBoundaryFromT(
    const DeviceBoundary& dbT,
    const ThermoCoeffs& c,
    DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>* pBnd)
{
    if (dbT.n == 0) return;
    if (dbHe.n != dbT.n)
    {
        throw std::runtime_error(
            "brae: deviceEnergyBoundaryFromT needs the he boundary to share the T boundary's structure.");
    }
    dbHe.refValue.resize(dbT.n);
    const bool hasGrad = dbT.refGrad.size() == static_cast<std::size_t>(dbT.n);
    if (hasGrad) dbHe.refGrad.resize(dbT.n);
    if (c.model == ThermoModel::liquidH2O)
    {
        const EnergyForm form = c.internalEnergy ? EnergyForm::sensibleInternalEnergy
                                                 : EnergyForm::sensibleEnthalpy;
        if (c.internalEnergy && (!pBnd || pBnd->size() != static_cast<std::size_t>(dbT.n)))
            throw std::runtime_error(
                "brae: converting a temperature boundary to energy for a liquid with "
                "sensibleInternalEnergy needs the boundary pressure (e = h(T) - p/rho(T)); the caller "
                "passed none, or one sized for a different patch set.");
        const int n = dbT.n;
        const scalar* pBndD = (c.internalEnergy && pBnd) ? pBnd->data() : nullptr;
        const scalar* refTD = dbT.refValue.data();
        const scalar* gradTD = hasGrad ? dbT.refGrad.data() : nullptr;
        scalar* refHeD = dbHe.refValue.data();
        scalar* gradHeD = hasGrad ? dbHe.refGrad.data() : nullptr;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            heBndFromTLiquidK(n, form, pBndD, refTD, gradTD, refHeD, gradHeD); });
        cudaCheck(cudaGetLastError(), "heBndFromTLiquid");
        return;
    }
    {
        const int n = dbT.n;
        const scalar* refTD = dbT.refValue.data();
        const scalar* gradTD = hasGrad ? dbT.refGrad.data() : nullptr;
        scalar* refHeD = dbHe.refValue.data();
        scalar* gradHeD = hasGrad ? dbHe.refGrad.data() : nullptr;
        pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
            heBndFromTK(n, c, refTD, gradTD, refHeD, gradHeD); });
    }
    cudaCheck(cudaGetLastError(), "heBndFromT");
}

// alphaEff = alpha + alphat. Separate kernel rather than folded into thermoUpdateK because alphat is
// owned by the turbulence model, which runs after the thermo update, not before it.
__device__
void alphaEffK(
    int n,
    scalar CpByCpv,
    const scalar* __restrict__ alpha,
    const scalar* __restrict__ alphat,
    scalar* __restrict__ alphaEff)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // OF heThermo::alphaEff = CpByCpv*(alpha + alphat). CpByCpv = 1 for sensibleEnthalpy, gamma = Cp/Cv
    // for sensibleInternalEnergy -- so an "e" case diffuses ~1.4x faster than an "h" case with the same
    // alpha, and dropping the factor still converges.
    alphaEff[i] = CpByCpv * (alpha[i] + alphat[i]);
}

void deviceAlphaEff(
    const DeviceThermo& th,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& alphaEff)
{
    if (th.n == 0) return;
    alphaEff.resize(th.n);
    const int n = th.n; const scalar cpByCpv = thermoCpByCpv(c);
    const scalar* alphaD = th.alpha.data(); const scalar* alphatD = th.alphat.data();
    scalar* alphaEffD = alphaEff.data();
    pcudaParallelFor(nBlocks(n), TPB, [=] __device__ () {
        alphaEffK(n, cpByCpv, alphaD, alphatD, alphaEffD); });
    cudaCheck(cudaGetLastError(), "alphaEff");
}

// The quantity OF's EEqn convects alongside he:
//
//   sensibleEnthalpy       ->  K   = 0.5|U|^2
//   sensibleInternalEnergy ->  Ekp = 0.5|U|^2 + p/rho
//
// (OF EEqn.H: he.name() == "e" ? fvc::div(phi, Ekp) : fvc::div(phi, K).) The p/rho term is the flow work
// that enthalpy already carries internally and internal energy does not, so leaving it out of an "e" case
// loses the pressure work entirely -- and still converges.
__device__
void kineticEnergyK(
    int n,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ p,     // null -> K (sensibleEnthalpy); else Ekp adds p/rho
    const scalar* __restrict__ rho,
    scalar* __restrict__ K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const scalar k = 0.5 * (Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    K[c] = p ? k + p[c] / rho[c] : k;
}

// Face value of K/Ekp times the mass flux: ffc_f = phi_f * K_f.
//
// The face value follows the case's div(phi,K|Ekp) scheme, exactly as the implicit div(phi,he) does.
// Upwind is W = pos0(phi); limitedLinear is OF's NVD/TVD blend W = limiter*cdw + (1-limiter)*pos0 with
// limiter = clamp(twoByk*r, 0, 1) -- the SAME formula divLimitedFaceKernel uses for the implicit term, so
// the explicit and implicit halves of the energy equation cannot end up on different schemes.
// Hardcoding upwind here is worth ~6.6e-4 in T on a limitedLinear case whose implicit term is exact.
__device__
void kineticFaceFluxK(
    int nF,
    const label* __restrict__ owner,
    const label* __restrict__ neighbour,
    const scalar* __restrict__ phiInt,
    const scalar* __restrict__ K,
    const scalar* __restrict__ cdw,     // null -> upwind|linearUpwind; else limitedLinear (gradient below)
    int linUpwind,                      // 1 -> Gauss linearUpwind: K_f = K_up + (Cf - C_up).grad(K)_up
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    scalar twoByk,
    scalar* __restrict__ ffc)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    const int P = owner[f], N = neighbour[f];
    const scalar ph = phiInt[f];
    const scalar pos0 = (ph >= scalar(0)) ? scalar(1) : scalar(0);
    if (linUpwind)
    {
        // OF's EEqn carries this term as fvc::div (fully EXPLICIT), so linearUpwind is simply a different
        // face value -- no deferred-correction split is needed, unlike the implicit div(phi,he).
        const int Uc = (ph > scalar(0)) ? P : N;
        const scalar dxu = (Uc == P) ? dOwnX[f] : dNeiX[f];
        const scalar dyu = (Uc == P) ? dOwnY[f] : dNeiY[f];
        const scalar dzu = (Uc == P) ? dOwnZ[f] : dNeiZ[f];
        ffc[f] = ph * (K[Uc] + dxu*gx[Uc] + dyu*gy[Uc] + dzu*gz[Uc]);
        return;
    }
    scalar W = pos0;
    if (cdw)
    {
        const scalar dx = dOwnX[f] - dNeiX[f], dy = dOwnY[f] - dNeiY[f], dz = dOwnZ[f] - dNeiZ[f];
        const int U = (ph > scalar(0)) ? P : N;
        const scalar gradcf = dx*gx[U] + dy*gy[U] + dz*gz[U];
        const scalar gradf  = K[N] - K[P];
        scalar r;
        if (fabs(gradcf) >= scalar(1000) * fabs(gradf))
            r = scalar(2)*scalar(1000)*((gradcf >= 0) ? scalar(1) : scalar(-1))*((gradf >= 0) ? scalar(1) : scalar(-1)) - scalar(1);
        else
            r = scalar(2)*(gradcf/gradf) - scalar(1);
        scalar limiter = twoByk * r;
        limiter = (limiter < scalar(0)) ? scalar(0) : (limiter > scalar(1) ? scalar(1) : limiter);
        W = limiter * cdw[f] + (scalar(1) - limiter) * pos0;
    }
    ffc[f] = ph * (W * K[P] + (scalar(1) - W) * K[N]);
}

// Boundary half: += phi_b*K_b, and the "bounded" correction -K_c*(sum_f phi_f) that OF's
// boundedConvectionScheme subtracts. Both scattered with atomics onto the adjacent cell.
__device__
void kineticBoundaryK(
    int nB,
    const label* __restrict__ faceCell,
    const scalar* __restrict__ phiBnd,
    const scalar* __restrict__ Kb,
    scalar* __restrict__ src,
    scalar* __restrict__ sumPhi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    const int c = faceCell[i];
    atomicAdd(&src[c], phiBnd[i] * Kb[i]);
    atomicAdd(&sumPhi[c], phiBnd[i]);
}

__device__
void kineticBoundedK(
    int n,
    const scalar* __restrict__ K,
    const scalar* __restrict__ sumPhi,
    scalar* __restrict__ src)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    src[c] -= K[c] * sumPhi[c];   // OF: boundedConvectionScheme subtracts fvc::surfaceIntegrate(phi)*vf
}

void deviceEnergyKineticSource(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    DeviceBuffer<scalar>& src,
    const DeviceBuffer<scalar>* p,        // sensibleInternalEnergy: cell p and rho -> Ekp = K + p/rho
    const DeviceBuffer<scalar>* rho,
    const DeviceBuffer<scalar>* pBndIn,   // and the same at boundary faces
    const DeviceBuffer<scalar>* rhoBndIn,
    const DeviceBoundary* dbHe,           // for grad(K) when the scheme is limitedLinear or linearUpwind
    bool limited,
    scalar twoByk,
    bool linearUpwind,
    scalar gradLimitK,
    bool bounded)
{
    const bool ekp = (p && rho && p->size() && rho->size());
    const int nC = dm.nCells;
    const int nF = dm.nInternalFaces;
    DeviceBuffer<scalar> K;
    K.resize(nC);
    {
        const scalar* Uxd = Ux.data(); const scalar* Uyd = Uy.data(); const scalar* Uzd = Uz.data();
        const scalar* pd = ekp ? p->data() : nullptr; const scalar* rhod = ekp ? rho->data() : nullptr;
        scalar* Kd = K.data();
        pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
            kineticEnergyK(nC, Uxd, Uyd, Uzd, pd, rhod, Kd); });
    }
    cudaCheck(cudaGetLastError(), "kineticEnergy");
    if (stageDumpActive() && stageDumpFirstOnly("Ekp")) stageDump("stage_Ekp", K);

    // K at boundary faces, built from the BOUNDARY velocity and (for Ekp) the boundary p and rho -- the
    // same expression as the cell value, which is exactly what OF's constructed volScalarField("Ekp", ...)
    // carries on its calculated patches. Hoisted above the gradient because grad(K) needs it.
    //
    // It previously used deviceBCValue(*dbHe, K, ...), i.e. it applied the ENERGY field's BC descriptor to
    // the K array: at a fixedValue he patch that returns he's refValue, not K's boundary value at all. The
    // error was masked for limitedLinear, whose gradient only feeds a limiter clamped to [0,1], but it
    // goes straight into the face value for linearUpwind -- enabling linearUpwind on top of the wrong
    // gradient made T four times WORSE (9.3e-02 vs 2.0e-02), which is how this was found.
    const int nB = dbU.comp[0].n;
    DeviceBuffer<scalar> Kb, ubx, uby, ubz;
    const bool ekpB = ekp && pBndIn && rhoBndIn && nB > 0
                   && pBndIn->size() == static_cast<std::size_t>(nB)
                   && rhoBndIn->size() == static_cast<std::size_t>(nB);
    if (nB > 0)
    {
        deviceBCValue(dbU.comp[0], Ux, ubx);
        deviceBCValue(dbU.comp[1], Uy, uby);
        deviceBCValue(dbU.comp[2], Uz, ubz);
        Kb.resize(nB);
        {
            const scalar* ubxd = ubx.data(); const scalar* ubyd = uby.data(); const scalar* ubzd = ubz.data();
            const scalar* pd = ekpB ? pBndIn->data() : nullptr; const scalar* rhod = ekpB ? rhoBndIn->data() : nullptr;
            scalar* Kbd = Kb.data();
            pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () {
                kineticEnergyK(nB, ubxd, ubyd, ubzd, pd, rhod, Kbd); });
        }
        cudaCheck(cudaGetLastError(), "kineticEnergyBnd");
    }

    DeviceBuffer<scalar> gx, gy, gz;
    if (limited || linearUpwind)
    {
        deviceGaussGrad(dm, K, Kb, gx, gy, gz);
        // `linearUpwind <gradName>` names a CELL-LIMITED gradient in every stock aerofoil case. Running
        // this correction unlimited put the whole first-iteration energy error of NACA0012 into the two
        // cells either side of the first face off the wall (measured: -1.5e3/+1.5e3 on cells 2400/2401).
        if (gradLimitK > 0.0) deviceCellLimitGrad(dm, K, Kb, gx, gy, gz, gradLimitK);
    }
    DeviceBuffer<scalar> ffc;
    ffc.resize(nF);
    {
        const label* own = dm.owner.data(); const label* nei = dm.nei.data();
        const scalar* phiIntD = phiInt.data(); const scalar* Kd = K.data();
        const scalar* cdwD = limited ? dm.w.data() : nullptr;
        const int linUp = linearUpwind ? 1 : 0;
        const scalar* gxd = (limited || linearUpwind) ? gx.data() : nullptr;
        const scalar* gyd = (limited || linearUpwind) ? gy.data() : nullptr;
        const scalar* gzd = (limited || linearUpwind) ? gz.data() : nullptr;
        const scalar* dOwnX = dm.dOwnX.data(); const scalar* dOwnY = dm.dOwnY.data(); const scalar* dOwnZ = dm.dOwnZ.data();
        const scalar* dNeiX = dm.dNeiX.data(); const scalar* dNeiY = dm.dNeiY.data(); const scalar* dNeiZ = dm.dNeiZ.data();
        scalar* ffcd = ffc.data();
        pcudaParallelFor(nBlocks(nF), TPB, [=] __device__ () {
            kineticFaceFluxK(nF, own, nei, phiIntD, Kd, cdwD, linUp, gxd, gyd, gzd,
                              dOwnX, dOwnY, dOwnZ, dNeiX, dNeiY, dNeiZ, twoByk, ffcd); });
    }
    cudaCheck(cudaGetLastError(), "kineticFaceFlux");
    // deviceFaceDivSource returns MINUS V*div (it exists for the laplacian non-orth correction, which
    // wants that sign). Negate so src is +V*div(phi,K), matching the boundary half added below and the
    // sign OF's fvc::div has. Getting this wrong is invisible in an "h" case, where K is ~0.003% of he,
    // and dominant in an "e" case, where Ekp carries p/rho ~ 86 kJ/kg against e ~ 215 kJ/kg.
    deviceFaceDivSource(dm, ffc, src);
    deviceScale(src, -1.0);

    // K at boundary faces, from the boundary velocity (noSlip walls give K_b = 0, which is the point:
    // the flow gives up its kinetic energy there and OF puts that into the enthalpy equation).
    if (nB > 0)
    {
        DeviceBuffer<scalar> sumPhi;
        sumPhi.resize(nC);
        cudaCheck(cudaMemsetAsync(sumPhi.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "sumPhi zero");
        {
            const label* faceCell = dbU.comp[0].faceCell.data(); const scalar* phiBndD = phiBnd.data();
            const scalar* Kbd = Kb.data(); scalar* srcd = src.data(); scalar* sumPhid = sumPhi.data();
            pcudaParallelFor(nBlocks(nB), TPB, [=] __device__ () {
                kineticBoundaryK(nB, faceCell, phiBndD, Kbd, srcd, sumPhid); });
        }
        cudaCheck(cudaGetLastError(), "kineticBoundary");
        // THE bounded CORRECTION IS PART OF THE SCHEME, NOT OF fvc::div. OF writes
        //     fvc::div(phi, Ekp)
        // and looks div(phi,Ekp) up in fvSchemes; only if that entry says `bounded` does
        // boundedConvectionScheme::fvcDiv subtract fvc::surfaceIntegrate(phi)*vf. Applying it
        // unconditionally adds a spurious source of -Ekp*div(phi) wherever the flux is not yet
        // conservative -- which is precisely the inlet region during the transient, and Ekp there is
        // dominated by p/rho ~ 86 kJ/kg. Measured on gasMixing/injectorPipe (div(phi,Ekp) NOT bounded):
        // the energy source in the affected cells ran 8600x the rest of the field at iteration 1 and the
        // temperature bled from 300 K to 178 K over 1200 iterations while OpenFOAM held 300 K. squareBend
        // is `bounded Gauss upwind`, so it saw the correct term and matched OF to 1e-6 -- which is why
        // this survived: the flag was parsed into ctl.boundedKin and then never read.
        if (bounded)
        {
            // internal faces contribute to sum_f phi_f too -- same gather, same negation as above so the
            // internal and boundary halves of sumPhi carry the SAME sign.
            DeviceBuffer<scalar> divPhiInt;
            deviceFaceDivSource(dm, phiInt, divPhiInt);
            deviceAxpy(-1.0, divPhiInt, sumPhi);
            {
                const scalar* Kd = K.data(); const scalar* sumPhid = sumPhi.data(); scalar* srcd = src.data();
                pcudaParallelFor(nBlocks(nC), TPB, [=] __device__ () {
                    kineticBoundedK(nC, Kd, sumPhid, srcd); });
            }
            cudaCheck(cudaGetLastError(), "kineticBounded");
        }
    }
}

void deviceSolveEnergy(
    const DeviceMesh& dm,
    const DeviceBoundary& dbHe,
    DeviceThermo& th,
    const ThermoCoeffs& c,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& divU,
    bool bounded,
    bool limited,
    bool linearUpwind,
    bool nonOrth,
    scalar twoByk,
    scalar relax,
    scalar tol,
    scalar relTol,
    int checkEvery,
    bool useGS,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    const DeviceBuffer<scalar>* kineticSrc,
    const DeviceBuffer<scalar>* alphaEffBnd,
    scalar gradLimitK,
    const DeviceWallData* fixT,
    const DeviceBuffer<scalar>* fixTHe)
{
    if (th.n == 0) return;

    DeviceBuffer<scalar> alphaEff;
    deviceAlphaEff(th, c, alphaEff);

    // bounded = false: he is not a positive-definite turbulence quantity, so the -Sp(div(phi),he)
    // correction the k/epsilon models use does not belong here.
    //
    // The only source is OF's kinetic-energy transport: EEqn carries + fvc::div(phi, K) on the LHS with
    // K = 0.5|U|^2, so it enters the RHS with a minus. Negligible in a slow laminar duct, which is why
    // phase 1 left it out; at 50 m/s against a noSlip wall it is not, and it shifts T by percent.
    deviceSolveScalarTransport(
        dm,
        dbHe,
        th.he,
        "he",
        alphaEff,
        phiInt,
        phiBnd,
        divU,
        bounded,
        limited,
        linearUpwind,
        nonOrth,
        twoByk,
        relax,
        tol,
        relTol,
        checkEvery,
        useGS,
        [&](DeviceBuffer<scalar>&, DeviceBuffer<scalar>& source)
        {
            if (kineticSrc && kineticSrc->size()) deviceAxpy(-1.0, *kineticSrc, source);
        },
        fixT,        // fixedTemperatureConstraint -> setValues, as the eps wall constraint does
        fixTHe,
        ami,
        cyc,
        ScalarDdt{},
        alphaEffBnd,
        gradLimitK,   // grad(he) cellLimited coeff, from the gradient `linearUpwind` names in fvSchemes
        false);       // no bound(he) -- OF bounds only positive-definite quantities, and he is not one
}

} // namespace brae
