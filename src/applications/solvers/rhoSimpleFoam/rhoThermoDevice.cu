// Device-resident implementations of the driver's three thermo hooks. See rhoThermoDevice.cuh for why
// they exist and what they refuse.
#include "rhoThermoDevice.cuh"
#include "thermo_model.cuh"          // hConstHeToT, thermoCpByCpv
#include "equation_of_state.cuh"     // perfectGasPsi, perfectGasRho
#include "transport_model.cuh"       // transportMu, transportAlpha
#include "device_blas.cuh"
#include <stdexcept>

namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// The scope guard. It lives in one place rather than at the top of each kernel launch because a partial
// refusal -- correcting the temperature on the gas path and then evaluating a liquid's transport -- is
// the failure this project keeps naming.
void requirePerfectGas(const ThermoCoeffs& c, const char* what)
{
    if (c.model != ThermoModel::perfectGas)
    {
        throw std::runtime_error(
            std::string("rhoSimpleFoam(cuda): ") + what + " is implemented for perfectGas + hConst + "
            "(const | sutherland) only, and this case selects a liquid. The liquid path replaces Cp, mu, "
            "kappa and rho with per-cell NSRDS correlations and inverts he -> T by Newton rather than in "
            "closed form; device_thermo.cu carries that path for the legacy solver, but no compressible "
            "liquid fixture gates it through this driver. Refusing rather than running a gas equation of "
            "state against a liquid's coefficients.");
    }
}

// he -> T -> psi, and the thermo's own rho from the CURRENT p and T. One kernel over cells, one over
// boundary faces, because the boundary temperature is T's own boundary condition rather than an
// inversion of the boundary enthalpy.
__global__ void thermoCorrectCellKernel(
    int                   n,
    const scalar* __restrict__ he,
    const scalar* __restrict__ p,
    ThermoCoeffs          c,
    scalar* __restrict__  T,
    scalar* __restrict__  psi,
    scalar* __restrict__  rhoThermo)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar t = hConstHeToT(he[i], c);
    T[i]         = t;
    psi[i]       = perfectGasPsi(t, c);
    rhoThermo[i] = perfectGasRho(p[i], t, c);
}

// heRhoThermo::calculate()'s patch loop (heRhoThermo.C:102-142), per face. A face whose T fixesValue()
// -- fixedValue (bcType 1) and every mixed-derived one (inletOutlet, outletInlet, mixed: their masks;
// mixedFvPatchField::fixesValue() is true whatever the flux sign, mixedFvPatchField.H:197) -- KEEPS its
// T and gets he_b = HE(p_b, T_b); every other face inverts T_b = THE(he_b, p_b). psi_b and rho_b follow.
// The evaluate this replaced (deviceBCValue on dbT, from the just-corrected cells) is what OpenFOAM does
// NOT do here: a mixed T patch is evaluated only inside the energy conditions' updateCoeffs, at the
// energy assembly -- see the step's deviceBCValue(dbT) before assembleEEqn (item 26).
__global__ void thermoCorrectBndKernel(
    int                   n,
    const label* __restrict__ bcType,
    const label* __restrict__ ioMask,
    const label* __restrict__ oioMask,
    const label* __restrict__ mixedMask,
    const scalar* __restrict__ pBnd,
    ThermoCoeffs          c,
    scalar* __restrict__  heBnd,
    scalar* __restrict__  TBnd,
    scalar* __restrict__  psiBnd,
    scalar* __restrict__  rhoThermoBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bool fixes = bcType[i] == 1
                    || (ioMask    && ioMask[i])
                    || (oioMask   && oioMask[i])
                    || (mixedMask && mixedMask[i]);
    if (fixes) heBnd[i] = hConstTToHe(TBnd[i], c);
    else       TBnd[i]  = hConstHeToT(heBnd[i], c);
    psiBnd[i]       = perfectGasPsi(TBnd[i], c);
    rhoThermoBnd[i] = perfectGasRho(pBnd[i], TBnd[i], c);
}

// rho = p/(R T), used only on the hePsiThermo branch -- the heRhoThermo branch is a copy of the stored
// field and needs no arithmetic at all.
__global__ void rhoFromPTKernel(
    int                   n,
    const scalar* __restrict__ p,
    const scalar* __restrict__ T,
    ThermoCoeffs          c,
    scalar* __restrict__  rho)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rho[i] = perfectGasRho(p[i], T[i], c);
}

// muEff and alphaEff over an arbitrary run of entries. Cells and boundary faces take the SAME kernel
// because the arithmetic is identical -- what differs is only which nut and which alphat are passed, and
// making that the caller's choice is what keeps the boundary from quietly inheriting the cell's.
__global__ void effectiveTransportKernel(
    int                   n,
    const scalar* __restrict__ T,
    const scalar* __restrict__ rho,
    const scalar* __restrict__ nut,      // null on a laminar case
    const scalar* __restrict__ alphat,   // null when the case ships none
    ThermoCoeffs          c,
    scalar                cpByCpv,
    scalar* __restrict__  muEff,
    scalar* __restrict__  alphaEff)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar muLam = transportMu(T[i], c);
    // mut = rho*nut, and alphat is READ from the field the turbulence model maintains rather than
    // rebuilt from nut/Prt here, so the two cannot drift apart.
    const scalar mut    = nut    ? rho[i] * nut[i] : scalar(0);
    const scalar alphaT = alphat ? alphat[i]       : scalar(0);
    muEff[i]    = muLam + mut;
    alphaEff[i] = cpByCpv * (transportAlpha(muLam, c) + alphaT);
}

} // namespace


void thermoCorrect(
    RhoSolverFields&      f,
    const DeviceBoundary& dbT,
    const ThermoCoeffs&   c)
{
    requirePerfectGas(c, "thermo.correct()");
    const int nC = static_cast<int>(f.he.size());
    if (nC == 0) return;

    f.T.resize(nC);
    f.psi.resize(nC);
    f.rhoThermo.resize(nC);
    thermoCorrectCellKernel<<<nBlocks(nC), TPB>>>(
        nC, f.he.data(), f.p.data(), c, f.T.data(), f.psi.data(), f.rhoThermo.data());
    cudaCheck(cudaGetLastError(), "rhoThermoCorrectCell");

    // The boundary half: calculate()'s patch loop, NOT an evaluate of T's own conditions -- see the
    // kernel. T's boundary was evaluated at the energy assembly and stands; he_b is written on the
    // fixesValue faces, T_b on the others.
    const int nB = static_cast<int>(f.TBnd.size());
    if (nB == 0) return;
    if (f.heBnd.size() != static_cast<std::size_t>(nB) || dbT.n != nB)
    {
        throw std::runtime_error("rhoThermoDevice: thermo.correct() needs he's and T's boundary values on "
                                 "every boundary face -- the energy solve's evaluate has not run.");
    }
    f.psiBnd.resize(nB);
    f.rhoThermoBnd.resize(nB);
    thermoCorrectBndKernel<<<nBlocks(nB), TPB>>>(
        nB, dbT.bcType.data(),
        dbT.ioMask.size()    ? dbT.ioMask.data()    : nullptr,
        dbT.oioMask.size()   ? dbT.oioMask.data()   : nullptr,
        dbT.mixedMask.size() ? dbT.mixedMask.data() : nullptr,
        f.pBnd.data(), c, f.heBnd.data(), f.TBnd.data(), f.psiBnd.data(), f.rhoThermoBnd.data());
    cudaCheck(cudaGetLastError(), "rhoThermoCorrectBnd");
}


void updateRho(
    RhoSolverFields&    f,
    const ThermoCoeffs& c)
{
    requirePerfectGas(c, "thermo.rho()");
    const int nC = static_cast<int>(f.p.size());
    if (nC == 0) return;

    if (c.rhoThermoType)
    {
        // rhoThermo::rho() returns the STORED rho_, which calculate() last filled inside thermo.correct()
        // -- from the pressure BEFORE the pressure equation ran. Recomputing it live here would be a
        // different density on any case where p moves appreciably in one iteration.
        deviceCopy(f.rho, f.rhoThermo);
        deviceCopy(f.rhoBnd, f.rhoThermoBnd);
        return;
    }

    f.rho.resize(nC);
    rhoFromPTKernel<<<nBlocks(nC), TPB>>>(nC, f.p.data(), f.T.data(), c, f.rho.data());
    cudaCheck(cudaGetLastError(), "rhoFromPT");

    const int nB = static_cast<int>(f.pBnd.size());
    if (nB == 0) return;
    f.rhoBnd.resize(nB);
    rhoFromPTKernel<<<nBlocks(nB), TPB>>>(nB, f.pBnd.data(), f.TBnd.data(), c, f.rhoBnd.data());
    cudaCheck(cudaGetLastError(), "rhoFromPTBnd");
}


void effectiveTransport(
    const RhoSolverFields& f,
    const ThermoCoeffs&    c,
    bool                   turbulent,
    DeviceBuffer<scalar>&  muEff,
    DeviceBuffer<scalar>&  muEffBnd,
    DeviceBuffer<scalar>&  alphaEff,
    DeviceBuffer<scalar>&  alphaEffBnd)
{
    requirePerfectGas(c, "the effective transport");
    const scalar cpByCpv = thermoCpByCpv(c);
    // The reference's own predicate: turbulent AND a nut field that actually exists. A case declared
    // turbulent whose closure has not been read is laminar as far as the transport is concerned.
    const bool turb = turbulent && f.nut.size() > 0;

    const int nC = static_cast<int>(f.T.size());
    if (nC > 0)
    {
        muEff.resize(nC);
        alphaEff.resize(nC);
        effectiveTransportKernel<<<nBlocks(nC), TPB>>>(
            nC, f.T.data(), f.rho.data(),
            turb ? f.nut.data() : nullptr,
            (turb && f.alphat.size() == static_cast<std::size_t>(nC)) ? f.alphat.data() : nullptr,
            c, cpByCpv, muEff.data(), alphaEff.data());
        cudaCheck(cudaGetLastError(), "rhoEffTransportCell");
    }

    const int nB = static_cast<int>(f.TBnd.size());
    if (nB > 0)
    {
        muEffBnd.resize(nB);
        alphaEffBnd.resize(nB);
        // THE PATCH's nut and alphat, not the owner cell's. On a wall carrying a nut wall function the
        // two differ by the whole of the turbulent viscosity, and on one carrying
        // compressible::alphatWallFunction by the whole of the turbulent diffusivity.
        effectiveTransportKernel<<<nBlocks(nB), TPB>>>(
            nB, f.TBnd.data(), f.rhoBnd.data(),
            (turb && f.nutBnd.size() == static_cast<std::size_t>(nB)) ? f.nutBnd.data() : nullptr,
            (turb && f.alphatBnd.size() == static_cast<std::size_t>(nB)) ? f.alphatBnd.data() : nullptr,
            c, cpByCpv, muEffBnd.data(), alphaEffBnd.data());
        cudaCheck(cudaGetLastError(), "rhoEffTransportBnd");
    }
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
