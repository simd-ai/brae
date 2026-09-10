// Device-resident implementations of the driver's three thermo hooks. See rhoThermoDevice.cuh for why
// they exist and what they refuse.
#include "rhoThermoDevice.cuh"
#include "thermo_model.cuh"          // thermoCpByCpv
#include "liquid_thermo.cuh"         // thermo*Of / thermoHeToT: the host step's accessors, BRAE_HD
#include "device_blas.cuh"
#include <climits>
#include <cstdio>
#include <stdexcept>

namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// EVERY PROPERTY THROUGH liquid_thermo.cuh, as on the host step (stage H3.6). These kernels called the
// perfect-gas closed forms directly -- hConstHeToT, perfectGasPsi, perfectGasRho, transportMu,
// transportAlpha -- behind a requirePerfectGas guard, which is what kept a liquid off this arm. The
// accessors branch once on the model and are BRAE_HD, so the kernels below and the host step call the
// same function for every property; on the gas path each accessor is the closed form it replaced,
// operation for operation, which is why no gas gate can move.
//
// The he -> T inversion is the one arithmetic that can FAIL. OpenFOAM raises a FatalError on its
// iteration cap (species::thermo<>::T, thermoI.H:80-87); a kernel cannot throw, so a face or cell that
// does not converge leaves its fields untouched and records its index, and the host throws with the
// inputs read back -- the same refusal and the same message shape as the host step's heToTFailure.
__device__ __forceinline__ void recordFailure(int* failIdx, int i)
{
    atomicMin(failIdx, i);
}

// he -> T -> psi, and the thermo's own rho from the CURRENT p and T. One kernel over cells, one over
// boundary faces, because the boundary temperature is T's own boundary condition rather than an
// inversion of the boundary enthalpy.
//
// T[i] on entry is the SEED: OpenFOAM passes TCells[celli] to THE (heRhoThermo.C:76-82), and its
// inversion's tolerance is built once from that seed, so the answer depends on it.
__global__ void thermoCorrectCellKernel(
    int                   n,
    const scalar* __restrict__ he,
    const scalar* __restrict__ p,
    ThermoCoeffs          c,
    scalar* __restrict__  T,
    scalar* __restrict__  psi,
    scalar* __restrict__  rhoThermo,
    int* __restrict__     failIdx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const HeToTResult inv = thermoHeToT(he[i], p[i], T[i], c);
    if (!inv.converged)
    {
        recordFailure(failIdx, i);
        return;
    }
    T[i]         = inv.T;
    psi[i]       = thermoPsiOf(p[i], inv.T, c);
    rhoThermo[i] = thermoRhoOf(p[i], inv.T, c);
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
    scalar* __restrict__  rhoThermoBnd,
    int* __restrict__     failIdx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bool fixes = bcType[i] == 1
                    || (ioMask    && ioMask[i])
                    || (oioMask   && oioMask[i])
                    || (mixedMask && mixedMask[i]);
    if (fixes)
    {
        heBnd[i] = thermoHeOf(pBnd[i], TBnd[i], c);
    }
    else
    {
        // Seeded with the face's own current T, as heRhoThermo.C:133 seeds it.
        const HeToTResult inv = thermoHeToT(heBnd[i], pBnd[i], TBnd[i], c);
        if (!inv.converged)
        {
            recordFailure(failIdx, i);
            return;
        }
        TBnd[i] = inv.T;
    }
    psiBnd[i]       = thermoPsiOf(pBnd[i], TBnd[i], c);
    rhoThermoBnd[i] = thermoRhoOf(pBnd[i], TBnd[i], c);
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
    rho[i] = thermoRhoOf(p[i], T[i], c);
}

// muEff and alphaEff over an arbitrary run of entries. Cells and boundary faces take the SAME kernel
// because the arithmetic is identical -- what differs is only which nut and which alphat are passed, and
// making that the caller's choice is what keeps the boundary from quietly inheriting the cell's.
__global__ void effectiveTransportKernel(
    int                   n,
    const scalar* __restrict__ p,
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
    const scalar muLam = thermoMuOf(p[i], T[i], c);
    // mut = rho*nut, and alphat is READ from the field the turbulence model maintains rather than
    // rebuilt from nut/Prt here, so the two cannot drift apart.
    const scalar mut    = nut    ? rho[i] * nut[i] : scalar(0);
    const scalar alphaT = alphat ? alphat[i]       : scalar(0);
    muEff[i]    = muLam + mut;
    alphaEff[i] = cpByCpv * (thermoAlphaOf(p[i], T[i], c) + alphaT);
}

// One face of the energy boundary update -- see updateEnergyBoundaryCoeffs in the header.
__global__ void energyBoundaryKernel(
    int                        n,
    const label* __restrict__  kind,
    const scalar* __restrict__ pBnd,
    const scalar* __restrict__ TBnd,
    const scalar* __restrict__ TrefValue,
    const scalar* __restrict__ TrefGrad,       // null when T carries no gradient on any face
    const scalar* __restrict__ Tvf,            // null when T carries no plain mixed face
    ThermoCoeffs               c,
    scalar* __restrict__       heRefValue,
    scalar* __restrict__       heRefGrad,      // null when he carries no gradient slot
    scalar* __restrict__       heVf,           // null when he carries no plain mixed face
    scalar* __restrict__       heBnd)          // he's stored patch values
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (kind[i])
    {
        case 1:
            heRefValue[i] = thermoHeOf(pBnd[i], TBnd[i], c);
            heBnd[i]      = heRefValue[i];
            break;
        case 2:
            if (heRefGrad) heRefGrad[i] = TrefGrad ? thermoCpvOf(pBnd[i], TBnd[i], c) * TrefGrad[i] : scalar(0);
            break;
        case 3:
            if (heVf && Tvf) heVf[i] = Tvf[i];
            heRefValue[i] = thermoHeOf(pBnd[i], TrefValue[i], c);
            if (heRefGrad) heRefGrad[i] = TrefGrad ? thermoCpvOf(pBnd[i], TBnd[i], c) * TrefGrad[i] : scalar(0);
            break;
        case 4:
            heRefValue[i] = thermoHeOf(pBnd[i], TrefValue[i], c);
            break;
        default:
            break;
    }
}

} // namespace


void updateEnergyBoundaryCoeffs(
    DeviceBoundary&             dbHe,
    const DeviceBoundary&       dbT,
    const DeviceBuffer<scalar>& pBnd,
    const DeviceBuffer<scalar>& TBnd,
    const DeviceBuffer<label>&  kind,
    const ThermoCoeffs&         c,
    DeviceBuffer<scalar>&       heBnd)
{
    const int n = dbHe.n;
    if (n == 0) return;
    const std::size_t sn = static_cast<std::size_t>(n);
    if (dbT.n != n || kind.size() != sn || pBnd.size() != sn || TBnd.size() != sn
        || dbHe.refValue.size() != sn || dbT.refValue.size() != sn || heBnd.size() != sn)
    {
        throw std::runtime_error("rhoThermoDevice: the energy boundary update needs he's and T's boundary "
                                 "descriptors, p_b and T_b on every boundary face -- a size disagrees.");
    }
    energyBoundaryKernel<<<nBlocks(n), TPB>>>(
        n, kind.data(), pBnd.data(), TBnd.data(), dbT.refValue.data(),
        dbT.refGrad.size() == sn ? dbT.refGrad.data() : nullptr,
        dbT.valueFraction.size() == sn ? dbT.valueFraction.data() : nullptr,
        c, dbHe.refValue.data(),
        dbHe.refGrad.size() == sn ? dbHe.refGrad.data() : nullptr,
        dbHe.valueFraction.size() == sn ? dbHe.valueFraction.data() : nullptr,
        heBnd.data());
    cudaCheck(cudaGetLastError(), "rhoEnergyBoundary");
}


// The host half of the failure path: one int on the device, INT_MAX meaning "every entry converged".
namespace
{
struct FailFlag
{
    DeviceBuffer<int> idx;
    FailFlag() { idx.copyFrom(std::vector<int>(1, INT_MAX)); }
    int read() const { return idx.host()[0]; }
};

[[noreturn]] void throwHeToTFailure(
    const char*                  where,
    int                          i,
    const DeviceBuffer<scalar>&  he,
    const DeviceBuffer<scalar>&  p,
    const DeviceBuffer<scalar>&  T)
{
    const double h = he.host()[static_cast<std::size_t>(i)];
    const double pp = p.host()[static_cast<std::size_t>(i)];
    const double t0 = T.host()[static_cast<std::size_t>(i)];
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "brae: rhoSimpleFoam(cuda) thermo.correct() could not invert he -> T at %s %d. "
                  "he = %.10g J/kg, p = %.10g Pa, seed T = %.10g K. OpenFOAM raises a FatalError here "
                  "(species::thermo<>::T, thermoI.H:80-87); brae refuses rather than carrying an "
                  "unconverged temperature into psi, rho and the pressure equation.",
                  where, i, h, pp, t0);
    throw std::runtime_error(buf);
}
} // namespace

void thermoCorrect(
    RhoSolverFields&      f,
    const DeviceBoundary& dbT,
    const ThermoCoeffs&   c)
{
    const int nC = static_cast<int>(f.he.size());
    if (nC == 0) return;
    // T is the inversion's SEED, so it has to exist already -- createDeviceFields projects it.
    if (f.T.size() != static_cast<std::size_t>(nC))
        throw std::runtime_error("rhoThermoDevice: thermo.correct() needs the current T as the he -> T "
                                 "seed, and the device T is not sized to the mesh.");

    f.psi.resize(nC);
    f.rhoThermo.resize(nC);
    {
        FailFlag fail;
        thermoCorrectCellKernel<<<nBlocks(nC), TPB>>>(
            nC, f.he.data(), f.p.data(), c, f.T.data(), f.psi.data(), f.rhoThermo.data(),
            fail.idx.data());
        cudaCheck(cudaGetLastError(), "rhoThermoCorrectCell");
        const int bad = fail.read();
        if (bad != INT_MAX) throwHeToTFailure("cell", bad, f.he, f.p, f.T);
    }

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
    FailFlag fail;
    thermoCorrectBndKernel<<<nBlocks(nB), TPB>>>(
        nB, dbT.bcType.data(),
        dbT.ioMask.size()    ? dbT.ioMask.data()    : nullptr,
        dbT.oioMask.size()   ? dbT.oioMask.data()   : nullptr,
        dbT.mixedMask.size() ? dbT.mixedMask.data() : nullptr,
        f.pBnd.data(), c, f.heBnd.data(), f.TBnd.data(), f.psiBnd.data(), f.rhoThermoBnd.data(),
        fail.idx.data());
    cudaCheck(cudaGetLastError(), "rhoThermoCorrectBnd");
    const int bad = fail.read();
    if (bad != INT_MAX) throwHeToTFailure("boundary face", bad, f.heBnd, f.pBnd, f.TBnd);
}


void updateRho(
    RhoSolverFields&    f,
    const ThermoCoeffs& c)
{
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
            nC, f.p.data(), f.T.data(), f.rho.data(),
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
            nB, f.pBnd.data(), f.TBnd.data(), f.rhoBnd.data(),
            (turb && f.nutBnd.size() == static_cast<std::size_t>(nB)) ? f.nutBnd.data() : nullptr,
            (turb && f.alphatBnd.size() == static_cast<std::size_t>(nB)) ? f.alphatBnd.data() : nullptr,
            c, cpByCpv, muEffBnd.data(), alphaEffBnd.data());
        cudaCheck(cudaGetLastError(), "rhoEffTransportBnd");
    }
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
