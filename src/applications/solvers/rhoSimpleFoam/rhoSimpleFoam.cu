// CUDA driver for rhoSimpleFoam. See rhoSimpleFoam.cuh for the provenance, the order and the contract.
#include "rhoSimpleFoam.cuh"
#include "device_fvoptions.cuh"
#include <string>   // deviceSetValues: fvOptions.constrain(EEqn)
#include "pEqn.cuh"              // correctVelocity, relaxField -- the stages that ARE shared
#include "device_pcg.cuh"
#include "device_blas.cuh"
#include "device_amg.cuh"
#include "device_simple.cuh"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <chrono>
namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

DeviceLduView foldedView(const DeviceMesh& dm, const PressureMatrix& P, const DeviceBuffer<scalar>& diagC)
{
    DeviceLduView A{};
    A.nCells = dm.nCells;
    A.nInternalFaces = dm.nInternalFaces;
    A.diag = diagC.data();
    A.upper = P.upper.data();
    A.lower = P.lower.data();
    A.owner = dm.owner.data();
    A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data();
    A.losortStart = dm.losortStart.data();
    return A;
}

DeviceLduView foldedViewM(const DeviceMesh& dm, const MomentumMatrix& M, const DeviceBuffer<scalar>& diagC)
{
    DeviceLduView A{};
    A.nCells = dm.nCells;
    A.nInternalFaces = dm.nInternalFaces;
    A.diag = diagC.data();
    A.upper = M.upper.data();
    A.lower = M.lower.data();
    A.owner = dm.owner.data();
    A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data();
    A.losortStart = dm.losortStart.data();
    return A;
}


// phi = phiHbyA + pEqn.flux(). PLUS, and gpu::correctFlux cannot be reused for it.
//
// rhoSimpleFoam writes the pressure equation as `fvc::div(phiHbyA) - fvm::laplacian(...) == 0` and the
// reference negates the ENTIRE assembled matrix -- diag, off-diagonals, source, both boundary coefficient
// arrays and the face-flux correction -- to match. The incompressible solver writes
// `fvm::laplacian(...) == fvc::div(phiHbyA)` and subtracts. Same physics, opposite sign, and the sign is
// what makes phi discretely conservative rather than merely plausible: a wrong one leaves div(phi) != 0
// while the pressure equation still solves happily.
void correctFluxCompressible(
    DeviceBuffer<scalar>&       phiInt,
    DeviceBuffer<scalar>&       phiBnd,
    const DeviceBuffer<scalar>& phiHbyAInt,
    const DeviceBuffer<scalar>& phiHbyABnd,
    const PressureMatrix&       P,
    const DeviceMesh&           dm,
    const DeviceBoundary&       dbP,
    const DeviceBuffer<scalar>& pSolved)
{
    DeviceBuffer<scalar> fInt, fBnd;
    deviceMatrixFluxInternal(P.view(dm), pSolved, fInt);
    deviceMatrixFluxBoundary(dbP, P.iC, P.bC, pSolved, fBnd);
    // fvMatrix.C:1688 -- `if (faceFluxCorrectionPtr_) fieldFlux += *faceFluxCorrectionPtr_;`
    if (P.faceFluxCorr.size() > 0) deviceAxpy(1.0, P.faceFluxCorr, fInt);

    deviceCopy(phiInt, phiHbyAInt);
    deviceAxpy(1.0, fInt, phiInt);
    deviceCopy(phiBnd, phiHbyABnd);
    deviceAxpy(1.0, fBnd, phiBnd);
}


// fvOptions.correct(he) for limitTemperature: clamp he between he(p,Tmin) and he(p,Tmax). A CORRECTION,
// so nothing in the assembly changes -- it acts on the solved field and then thermo.correct() turns it
// into a temperature. The bounds arrive already in energy; see the note in RhoStepInput.
__global__ void limitEnergyKernel(
    int    nC,
    scalar heMin,
    scalar heMax,
    scalar* __restrict__ he)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    he[c] = fmin(fmax(he[c], heMin), heMax);
}


// pressureControl::limit -- a clamp, applied in place. OpenFOAM returns true on `limitMaxP || limitMinP`
// rather than on whether any value actually moved, and the caller re-evaluates p's boundary on that
// return, so the boundary refresh below is keyed the same way.
__global__ void limitPressureKernel(
    int    nC,
    int    doMax,
    int    doMin,
    scalar pMax,
    scalar pMin,
    scalar* __restrict__ p)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    if (doMax) p[c] = fmin(p[c], pMax);
    if (doMin) p[c] = fmax(p[c], pMin);
}


// The closed-volume correction: p += (initialMass - domainIntegrate(psi*p))/domainIntegrate(psi).
// Two reductions and a scalar add; done on the host because it is two numbers, and the alternative is a
// device reduction whose result has to come back anyway.
void closedVolumeCorrection(
    DeviceBuffer<scalar>&       p,
    const DeviceBuffer<scalar>& psi,
    const DeviceMesh&           dm,
    double                      initialMass)
{
    const std::vector<scalar> hp = p.host(), hpsi = psi.host(), V = dm.V.host();
    double num = 0.0, den = 0.0;
    for (int c = 0; c < dm.nCells; ++c)
    {
        num += (double)hpsi[c] * (double)hp[c] * (double)V[c];
        den += (double)hpsi[c] * (double)V[c];
    }
    if (!(den > 0.0)) return;
    const scalar dp = (scalar)((initialMass - num) / den);
    std::vector<scalar> out(hp);
    for (int c = 0; c < dm.nCells; ++c) out[c] += dp;
    p.copyFrom(out);
}

// p.relax() on the totalPressure faces. GeometricField::relax assigns BOTH halves through operator==
// (GeometricField.C:1094, :1420), so the patch VALUE OpenFOAM carries out of p.relax() is the blend
// prevIter_b + alpha*(p_b - prevIter_b). On the device that value lives in refValue -- deviceBCValue
// reproduces a fixedValue face from it -- so the blended f.pBnd is written back there on the faces
// tpMask marks. Every other fixedValue face is unchanged by the blend, and every zeroGradient face is
// re-derived from the relaxed cell: the same function on the same operands, bit for bit.
__global__
void storeTotalPressureValueKernel(
    int    n,
    const label*  __restrict__ tpMask,
    const scalar* __restrict__ pBnd,
    scalar*       __restrict__ refValue)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !tpMask[i]) return;
    refValue[i] = pBnd[i];
}

} // namespace


// Instrument: BRAE_STAGE_DUMP_DIR=<dir> (+ BRAE_STAGE_DUMP_ITER=n, default 1) writes this step's stages
// at ONE iteration as plain columns, the device twin of the host step's StageDump (rhoSimpleFoam_cpp.cu):
// same names, same layout (vectors as three columns; a surface field as the internal faces plus one
// `_b` file holding every boundary face in patch order), so the two arms can be held against each
// other stage by stage from the SAME in-memory trajectory. Costs nothing unless the variable is set.
namespace
{
struct DeviceStageDump
{
    std::string dir;
    bool        on = false;

    void scalars(const char* name, const DeviceBuffer<scalar>& v) const
    {
        if (!on) return;
        const std::vector<scalar> h = v.host();
        std::FILE* fp = std::fopen((dir + "/" + name).c_str(), "w");
        if (!fp) return;
        for (scalar x : h) std::fprintf(fp, "%.17g\n", (double)x);
        std::fclose(fp);
    }
    void vectors(const char* name, const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& z) const
    {
        if (!on) return;
        const std::vector<scalar> hx = x.host(), hy = y.host(), hz = z.host();
        std::FILE* fp = std::fopen((dir + "/" + name).c_str(), "w");
        if (!fp) return;
        for (std::size_t i = 0; i < hx.size(); ++i)
            std::fprintf(fp, "%.17g %.17g %.17g\n", (double)hx[i], (double)hy[i], (double)hz[i]);
        std::fclose(fp);
    }
    void surface(const char* name, const DeviceBuffer<scalar>& internal, const DeviceBuffer<scalar>& bnd) const
    {
        if (!on) return;
        scalars(name, internal);
        scalars((std::string(name) + "_b").c_str(), bnd);
    }
};

DeviceStageDump deviceStageDump()
{
    DeviceStageDump d;
    const char* dd = std::getenv("BRAE_STAGE_DUMP_DIR");
    if (!dd) return d;
    static int calls = 0;
    const char* it = std::getenv("BRAE_STAGE_DUMP_ITER");
    d.dir = dd;
    d.on  = (++calls == (it ? std::atoi(it) : 1));
    return d;
}
} // namespace

// flowRateInletVelocity's updateCoeffs: avgU = -mdot/gSum(rho*magSf) against rho's PATCH value as it
// stands (flowRateInletVelocityFvPatchVectorField.C:210-220, lookupPatchField(rhoName_)). Called where
// OpenFOAM reaches that updateCoeffs: the momentum assembly, and the velocity correction's
// correctBoundaryConditions, where rho's patch value has moved since the assembly (the host step carries
// the sbMatched measurement, queue item 26).
static void updateFlowRateInlets(
    RhoSolverFields&       f,
    const RhoStepInput&    in,
    DeviceVectorBoundary&  dbU)
{
    if (!(in.frMagSf && in.frMdot && in.frNx && in.frNy && in.frNz)) return;
    for (std::size_t k = 0; k < in.frMagSf->size() && k < in.frMdot->size(); ++k)
    {
        const scalar sumRhoA = deviceDot(f.rhoBnd, (*in.frMagSf)[k]);
        if (sumRhoA <= scalar(0)) continue;
        deviceUpdateFlowRateInlet(dbU, (*in.frMagSf)[k], -(*in.frMdot)[k] / sumRhoA,
                                  *in.frNx, *in.frNy, *in.frNz);
    }
}

// updateCoeffs() for the boundary conditions whose coefficients are a function of the SOLUTION.
//
// A named function rather than a run of statements inside the step, because it has a contract of its own
// that is worth testing on its own: given a flux and a boundary density, it must produce the patch
// coefficients OpenFOAM's updateCoeffs() would. The driver's gate exercises it directly -- doubling the
// boundary density must halve a flowRateInletVelocity's velocity, and reversing the flux must flip an
// inletOutlet face between fixedValue and zeroGradient -- and neither of those is visible from a
// whole-iteration comparison on a fixture whose patches have no coefficients that move.
void updateBoundaryCoeffs(
    RhoSolverFields&      f,
    DeviceVectorBoundary& dbU,
    DeviceBoundary&       dbP,
    DeviceBoundary&       dbHe,
    DeviceBoundary&       dbT,
    const RhoStepInput&   in)
{
    // OpenFOAM runs this inside the fvMatrix constructor, so it has happened before any coefficient is
    // read. Here the device boundary objects are a snapshot and the driver has to do it by hand; the
    // order is the reference driver's, which is OpenFOAM's.
    //
    // 1. The FLUX SWITCH. inletOutlet/outletInlet pick fixedValue or zeroGradient per face from the sign
    //    of phi, and OpenFOAM lags it: the flux used is the one this iteration STARTS with. U, he and T
    //    are the fields that carry one on a compressible case.
    //
    //    dbT is refreshed even though nothing in THIS function reads it. T's boundary is consumed by
    //    thermo.correct(), which is the caller's hook; a host thermo evaluates T's patches on its own
    //    host field and will not notice, but a device-resident one reads dbT and would otherwise get a
    //    flux switch frozen at its seeded state. Refreshing it here keeps the two thermo implementations
    //    interchangeable, which is the whole point of the hook being a hook.
    deviceUpdateInletOutlet(dbU, f.phiBnd);
    deviceUpdateInletOutlet(dbHe, f.phiBnd);
    deviceUpdateInletOutlet(dbT, f.phiBnd);

    // 2. The FREESTREAM BLEND, a different rule from the switch above: valueFraction is rebuilt from the
    //    current flow ANGLE, 0.5 - 0.5*(Up & nf)/mag(Up), and freestreamPressure follows the velocity
    //    patch. Left alone, every far-field face keeps the half-and-half blend it was seeded with.
    //    U only here: freestreamPressure's valueFraction is rebuilt inside the pressure fvMatrix constructor
    //    (below, before the assembly) and under the limiter, and p's boundary is read AS IT STANDS by the
    //    momentum gradient -- the relaxed blend, or the limiter's re-evaluation (queue items 23 and 25).
    if (in.hasMixed)
    {
        deviceUpdateMixedFreestream(dbU, dbP, f.phiBnd, f.Ux, f.Uy, f.Uz, &f.rhoBnd, /*which=*/1);
    }

    // 2b. pressureInletOutletVelocity, whose updateCoeffs OpenFOAM reaches inside the momentum fvMatrix
    //     constructor (fvMatrix.C:396). It is not a coefficient update alone: it sets valueFraction =
    //     neg(phi)*(I - nn) from the flux THIS iteration starts with and then calls directionMixed::evaluate
    //     itself (pressureInletOutletVelocityFvPatchVectorField.C:180-183), so the patch VALUE at the
    //     assembly is n(n & U_cell) on inflow faces from the cell velocity as it stands here -- the number
    //     the previous iteration's post-correction evaluate left, and at iteration 1 what OpenFOAM computes
    //     too, so the 0/U seed never reaches a momentum assembly.
    //
    //     totalPressure is NOT updated here any more. OpenFOAM reaches p's updateCoeffs only inside the
    //     PRESSURE equation's constructor (once the momentum solve has moved U's patch value), so what
    //     -fvc::grad(p) reads at the momentum assembly is the value the previous tail left: the relaxed
    //     blend from p.relax(), or the recompute pEqn.H:100-103 runs after the limiter. Updating it here
    //     from the pre-solve velocity put a one-solve-old dynamic head into the pressure equation and a
    //     fresh value where OpenFOAM carries the blend: rhoTP at t=1 read U 2.3e-01 relL2 against
    //     OpenFOAM on this arm with the written inlet still at the (5 0 0) seed.
    {
        deviceUpdatePressureInletOutletVelocity(dbU, f.phiBnd, f.Ux, f.Uy, f.Uz, /*directionMixed=*/true);
        // symmetry/slip and wedge, against THIS iteration's cell velocity -- the header's caller
        // contract has listed both since it was written (rhoUEqn.cuh, clause 3), and the incompressible
        // driver has always run them (device_simple_foam.cu:955-956); this driver did not, which no
        // axis-aligned fixture could see: with n along one axis the per-component vf=|n_k| decouples
        // and the stale snapshot equals the fresh one. A TILTED symmetry plane (rhoBoxSym) couples the
        // components and is where the missing calls measured.
        deviceUpdateSymmetry(dbU, f.Ux, f.Uy, f.Uz);
        deviceUpdateWedge(dbU, f.Ux, f.Uy, f.Uz);
    }

    // 3. flowRateInletVelocity, and WHICH rho matters: avgU = -mdot/gSum(rho*magSf) is held against the
    //    boundary density the flux is actually carrying -- the solver's relaxed rho, which is what
    //    f.rhoBnd holds here -- not thermo.rho(). Feeding it the other one is the angledDuct defect,
    //    where the inlet quietly lost the prescribed mass flow. Last, because it reads that rho.
    updateFlowRateInlets(f, in, dbU);
}


// ---------------------------------------------------------------------------------------------------
// BRAE_PHASE_TIME (item 67): where the mirror's outer iteration goes. Off unless the variable is set --
// attributing device work needs a synchronise at each boundary, and that is itself a cost, so the
// instrument must not be part of a measured run. The V2 driver has carried a hook split since item 54;
// the mirror had none, so its per-iteration cost could not be attributed at all. The marks are the
// step's own section boundaries, so what falls between pEqn and turbulence->correct() (the closing
// stage dumps) is charged to the pressure phase.
namespace
{
double g_tU = 0.0, g_tE = 0.0, g_tP = 0.0, g_tTurb = 0.0;
std::chrono::steady_clock::time_point g_phaseMark;
bool phaseTimeOn()
{
    static const bool on = std::getenv("BRAE_PHASE_TIME") != nullptr;
    return on;
}
// charge the time since the last mark to `slot` (null = start the clock), then restart it
void phaseMark(double* slot)
{
    if (!phaseTimeOn()) return;
    cudaDeviceSynchronize();
    const auto now = std::chrono::steady_clock::now();
    if (slot) *slot += std::chrono::duration<double>(now - g_phaseMark).count();
    g_phaseMark = now;
}
}   // namespace

void rhoPhaseTimeReport(int iterations)
{
    if (!phaseTimeOn() || iterations <= 0) return;
    const double tot = g_tU + g_tE + g_tP + g_tTurb;
    std::printf("  [phase] over %d iterations: UEqn %.3f s (%.1f ms/it), EEqn %.3f s (%.1f ms/it), "
                "pEqn %.3f s (%.1f ms/it), turbulence %.3f s (%.1f ms/it); the four total %.3f s (%.1f ms/it)\n",
                iterations,
                g_tU,    1e3 * g_tU    / iterations,
                g_tE,    1e3 * g_tE    / iterations,
                g_tP,    1e3 * g_tP    / iterations,
                g_tTurb, 1e3 * g_tTurb / iterations,
                tot,     1e3 * tot     / iterations);
}

Residuals rhoSimpleStep(
    RhoSolverFields&            f,
    RhoSolverWorkspace&         w,
    const DeviceMesh&           dm,
    // NON-const: the updateCoeffs() block at the top of the step rewrites refValue and valueFraction on
    // the patches that switch on the flux, blend on the flow angle, or carry a prescribed mass flow.
    DeviceVectorBoundary&       dbU,
    DeviceBoundary&             dbP,
    DeviceBoundary&             dbHe,
    DeviceBoundary&             dbT,
    const RhoStepInput&         in)
{
    Residuals res;
    const int nC = dm.nCells;

    // rho.prevIter(), stored where OpenFOAM stores it. simpleControl::loop() calls storePrevIterFields()
    // at the START of the iteration (simpleControl.C:157) and rho.relax() at the tail is
    // prevIter + alpha*(rho - prevIter) (GeometricField.C:1089-1095); pcEqn.H:1's `rho = thermo.rho()`
    // does not touch prevIter. This capture used to sit at the tail, one line before the tail's own
    // updateRho -- exact on the pEqn branch, where rho does not move in between, and wrong on the
    // SIMPLEC branch, whose pcEqn.H opens with rho = thermo.rho(): the relaxation then blended towards
    // that mid-iteration density instead of the one the iteration started with. No fixture could see it
    // (every consistent+subsonic one relaxes rho at 1.0); the gate is rhoBox with `consistent yes` and
    // `rho 0.5`, both arms against OpenFOAM at a matched iteration count.
    DeviceBuffer<scalar> rhoPrevIter, rhoBndPrevIter;
    deviceCopy(rhoPrevIter, f.rho);
    deviceCopy(rhoBndPrevIter, f.rhoBnd);

    if (!in.muEffCell || !in.muEffBndFace || !in.alphaEffCell || !in.alphaEffBndFace)
    {
        throw std::runtime_error(
            "rhoSimpleFoam(cuda): muEff and alphaEff are required on cells AND boundary faces. They are "
            "the ONLY place the closure enters the momentum and energy equations, and the boundary value "
            "is the patch's, not the owner cell's -- on a wall with an alphat wall function the two "
            "differ by the whole of alphat.");
    }
    if (!in.thermoCorrect || !in.updateRho)
    {
        throw std::runtime_error(
            "rhoSimpleFoam(cuda): thermoCorrect and updateRho are required hooks. EEqn.H ends in "
            "thermo.correct(), which moves T and therefore psi, and every consumer below that point "
            "reads the result; pcEqn.H opens with rho = thermo.rho(). Running without them would solve "
            "the whole iteration against the state it started with.");
    }

    if (w.ones.size() != static_cast<std::size_t>(nC))
    {
        w.ones.copyFrom(std::vector<scalar>(nC, scalar(1.0)));
    }

    // storePrevIter(): OpenFOAM banks prevIter at the TOP of the iteration, so p.relax() below relaxes
    // against the value p had before this iteration touched it -- not against the value it had at the
    // start of the pressure solve.
    DeviceBuffer<scalar> pPrev, pBndPrev;
    deviceCopy(pPrev, f.p);
    // BOTH halves: p.relax() assigns the boundary too (GeometricField.C:1094, :1420), and the
    // totalPressure inlet is a patch whose value moves between here and the tail -- it is recomputed
    // before the pressure assembly -- so the blend has to be against the value the iteration started with.
    deviceCopy(pBndPrev, f.pBnd);

    updateBoundaryCoeffs(f, dbU, dbP, dbHe, dbT, in);

    // ---- UEqn.H ------------------------------------------------------------------------------
    phaseMark(nullptr);
    RhoMomentumInput uin;
    uin.phiInt = &f.phiInt;          uin.phiBnd = &f.phiBnd;
    uin.rhoCell = &f.rho;            uin.rhoBndFace = &f.rhoBnd;
    // THE DYNAMIC SLOT, NOT THE KINEMATIC ONE. RhoMomentumInput carries both: muEffCell/muEffBndFace are
    // used verbatim, while nuEffCell/nuEffBndFace are KINEMATIC and the module forms rho*nuEff from them
    // (linearViscousStress.C:107-117). Feeding the dynamic muEff into the kinematic slot multiplies it by
    // rho a second time -- measured on rhoBox as a constant 1.161 on the whole diffusion term, which is
    // exactly p/(R*T) = 100000/(287.1*300) there, and it reached the converged velocity as a drift of
    // 5.4e-04 at iteration 1 growing to 3.9e-03 by iteration 8 while p, T and he all stayed at ~1e-7.
    uin.muEffCell = in.muEffCell;    uin.muEffBndFace = in.muEffBndFace;
    uin.UxBndFace = &f.UxBnd;        uin.UyBndFace = &f.UyBnd;        uin.UzBndFace = &f.UzBnd;
    uin.relaxU = in.relaxU;
    uin.relaxEquationU = in.relaxEquationU;
    uin.bounded = in.boundedU;
    uin.scheme = in.schemeU;
    uin.schemeCoeff = in.schemeCoeffU;
    uin.gradULimitK = in.gradULimitK;
    uin.correctedLaplacian = in.correctedLaplacian;
    uin.snGradLimitCoeff = in.snGradLimitCoeff;
    // The porosity the momentum module has always been able to apply, and which the driver never passed.
    uin.porosity = in.porosity;
    uin.hasMRF = in.hasMRF;
    uin.hasFvOptions = in.hasFvOptions;
    uin.hasCoupledPatches = in.hasCoupledPatches;
    uin.fvOptionUnsupported = in.fvOptionUnsupported;

    const DeviceStageDump sd = deviceStageDump();
    sd.vectors("Uass", f.Ux, f.Uy, f.Uz);
    sd.scalars("UassBx", f.UxBnd);
    sd.scalars("UassBy", f.UyBnd);
    sd.scalars("UassBz", f.UzBnd);
    sd.scalars("rhoU", f.rho);
    sd.surface("phiU", f.phiInt, f.phiBnd);
    sd.scalars("nutU", f.nut);
    MomentumMatrix UEqn;
    assembleUEqn(UEqn, dm, dbU, f.Ux, f.Uy, f.Uz, uin);
    sd.scalars("UDiag", UEqn.diag);
    sd.scalars("UUpper", UEqn.upper);
    sd.scalars("USrcX", UEqn.source[0]);
    sd.scalars("USrcY", UEqn.source[1]);
    sd.scalars("USrcZ", UEqn.source[2]);
    sd.scalars("muEffAss", *in.muEffCell);

    {
        // solve(UEqn == -fvc::grad(p)) on a COPY. The pressure equation needs the ORIGINAL for A(), H()
        // and H1(); adding grad(p) here would leave the source carrying it and move rAU and HbyA with it.
        MomentumMatrix Mp;
        deviceCopy(Mp.diag, UEqn.diag);
        deviceCopy(Mp.upper, UEqn.upper);
        deviceCopy(Mp.lower, UEqn.lower);
        deviceCopy(Mp.relaxedDiag, UEqn.relaxedDiag);
        Mp.relaxed = UEqn.relaxed;
        for (int k = 0; k < 3; ++k)
        {
            deviceCopy(Mp.source[k], UEqn.source[k]);
            deviceCopy(Mp.iC[k], UEqn.iC[k]);
            deviceCopy(Mp.bC[k], UEqn.bC[k]);
        }

        DeviceBuffer<scalar> gpx, gpy, gpz;
        // f.pBnd as it stands: the relaxed blend, or the limiter's re-evaluation, never re-derived here.
        deviceGaussGrad(dm, f.p, f.pBnd, gpx, gpy, gpz);
        addPressureGradient(Mp, dm, gpx, gpy, gpz);

        DeviceBuffer<scalar>* U[3] = {&f.Ux, &f.Uy, &f.Uz};
        // U's residual is cmptMax over the components OpenFOAM SOLVES: fvMatrix<vector>::solveSegregated
        // `continue`s on every component polyMesh::solutionD() knocks out (fvMatrixSolve.C:157-164), that
        // component's SolverPerformance stays Zero (SolverPerformance.H:117-121), and residualControl
        // compares cmptMax over the stored vector (solutionControl.C:232, simpleControl.C:67-71).
        //
        // This loop reported component 0 -- wrong whenever Uy's initial residual exceeds Ux's, which
        // on rhoBox is iterations 2, 3 and 8 (OpenFOAM at iteration 2: Ux 3.224e-01, Uy 6.042e-01; this
        // arm printed 3.224e-01) -- and the reason it did not take a max over three is that it solved
        // all three unconditionally: the empty direction's system has a ~0 right-hand side and a zero
        // field, so its normFactor-scaled residual reads 1 on every iteration (measured on rhoBox: Uz
        // 1.000e+00 at iterations 1..3), which would block convergence on every 2D case. The mask
        // in.solutionD is derived from the empty patches exactly as calcDirections does
        // (solution_directions.cuh) and the knocked-out component is not solved, as in OpenFOAM.
        scalar uInitialResidual = 0.0;
        // Every solved component's system first (its own folded diagonal and source, the shared
        // upper/lower, its normFactor -- none reads another component's psi), then the solves: the
        // Gauss-Seidel walks FUSED into one level walk per sweep (item 60a, byte-identical to one walk
        // per component, tests/gs_fused_identity; BRAE_GS_FUSED=0 restores those), BiCGStab per
        // component as before.
        DeviceBuffer<scalar> diagC[3], b[3], dnf[3];
        DeviceLduView A[3];
        int solved[3];
        int nSolved = 0;
        for (int k = 0; k < 3; ++k)
        {
            if (in.solutionD[k] < 0) continue;
            deviceFold(dm, Mp.relaxed ? Mp.relaxedDiag : Mp.diag, Mp.source[k], Mp.iC[k], Mp.bC[k], diagC[k], b[k]);
            A[k] = foldedViewM(dm, Mp, diagC[k]);
            deviceNormFactorInto(A[k], *U[k], b[k], w.ones, dnf[k]);   // stays on the device (item 66)
            solved[nSolved++] = k;
        }
        DeviceSolverPerf perfs[3];
        if (in.uSymGaussSeidel)
        {
            GSFusedComponent comps[3];
            DeviceSolverPerf fp[3];
            for (int i = 0; i < nSolved; ++i)
            {
                const int k = solved[i];
                comps[i] = {&A[k], &b[k], U[k], 1.0, dnf[k].data()};
            }
            deviceSymGaussSeidelFused(nSolved, comps, in.tolU, in.relTolU, in.maxIterU, in.minIterU, in.nSweepsU,
                                      in.uGaussSeidelSymmetric, fp);
            for (int i = 0; i < nSolved; ++i) perfs[solved[i]] = fp[i];
        }
        else
        {
            for (int i = 0; i < nSolved; ++i)
            {
                const int k = solved[i];
                perfs[k] = deviceJacobiBiCGStab(A[k], b[k], *U[k], dnf[k].data(), in.tolU, in.relTolU, in.maxIterU, /*checkEvery=*/1,
                                                in.minIterU, in.preconU);
            }
        }
        for (int i = 0; i < nSolved; ++i) uInitialResidual = std::max(uInitialResidual, perfs[solved[i]].initialResidual);
        res["U"] = uInitialResidual;
    }
    sd.vectors("Upred", f.Ux, f.Uy, f.Uz);

    // ---- EEqn.H ------------------------------------------------------------------------------
    phaseMark(&g_tU);
    {
        RhoEnergyInput ein;
        ein.phiInt = &f.phiInt;           ein.phiBnd = &f.phiBnd;
        ein.alphaEffCell = in.alphaEffCell;
        ein.alphaEffBndFace = in.alphaEffBndFace;
        ein.Ux = &f.Ux; ein.Uy = &f.Uy; ein.Uz = &f.Uz;
        ein.pCell = &f.p; ein.rhoCell = &f.rho;
        // Refreshed here, from the just-solved U: EEqn.H's kinetic-energy source is evaluated on
        // boundary faces as well as cells, and the momentum solve above has moved every one of them.
        //
        // AND THE SYMMETRY/WEDGE refValue WITH THEM. deviceBCValue reproduces
        // correctBoundaryConditions() only for a patch whose value is a function of (refValue, refGrad,
        // internal). symmetry/slip and wedge are not those: OpenFOAM gives them no updateCoeffs at all
        // -- symmetryPlaneFvPatchField::evaluate() is (iF + transform(I - 2*sqr(nHat), iF))/2 and
        // wedgeFvPatchField::evaluate() is transform(cellT(), iF), both read at the moment of
        // evaluation -- while brae carries them as a mixed refValue that some earlier kernel had to
        // build FROM the internal field. Left unrefreshed, this call blends the new U_c towards a ref
        // built from the U the iteration started with.
        //
        // AND THE piov refValue: fvMatrixSolve.C:242 ends the solve with psi.correctBoundaryConditions(),
        // which on that patch is updateCoeffs -> evaluate again -- the NEW cell velocity projected with
        // the flux mask the iteration started with (f.phiBnd is still that flux here).
        deviceUpdatePressureInletOutletVelocity(dbU, f.phiBnd, f.Ux, f.Uy, f.Uz, /*directionMixed=*/true);
        deviceUpdateSymmetry(dbU, f.Ux, f.Uy, f.Uz);
        deviceUpdateWedge(dbU, f.Ux, f.Uy, f.Uz);
        deviceBCValue(dbU.comp[0], f.Ux, f.UxBnd);
        deviceBCValue(dbU.comp[1], f.Uy, f.UyBnd);
        deviceBCValue(dbU.comp[2], f.Uz, f.UzBnd);
        ein.UxBnd = &f.UxBnd; ein.UyBnd = &f.UyBnd; ein.UzBnd = &f.UzBnd;
        ein.pBnd = &f.pBnd; ein.rhoBnd = &f.rhoBnd;
        ein.isE = in.isE;
        ein.relaxHe = in.relaxHe;
        ein.relaxEquationHe = in.relaxEquationHe;
        ein.boundedHe = in.boundedHe;
        ein.boundedKE = in.boundedKE;
        ein.schemeHe = in.schemeHe;
        ein.schemeKE = in.schemeKE;
        ein.gradHeLimitK = in.gradHeLimitK;
        ein.gradKELimitK = in.gradKELimitK;
        ein.correctedLaplacian = in.correctedLaplacian;
        ein.snGradLimitCoeff = in.snGradLimitCoeff;
        ein.hasMRF = in.hasMRF;
        ein.hasFvOptions = in.hasFvOptions;
        ein.hasCoupledPatches = in.hasCoupledPatches;

        PressureMatrix E;
        // Tw.evaluate() -- every energy condition's updateCoeffs evaluates T's patch from the cells as
        // they stand at the energy assembly (fixedEnergy .C:108, gradientEnergy .C:109, mixedEnergy .C:97).
        // The ONLY evaluate T's boundary gets in an iteration: thermo.correct() keeps it on fixesValue
        // faces and inverts he_b on the rest (rhoThermoDevice.cu), so the outlet T_b -- and the rho_b,
        // mu_b, alphaEff_b built from it -- lag the cells by one iteration exactly as OpenFOAM's do.
        deviceBCValue(dbT, f.T, f.TBnd);
        assembleEEqn(E, dm, dbHe, f.he, ein);

        // fvOptions.constrain(EEqn) -- EEqn.H:20, on the ASSEMBLED matrix and before the solve, which
        // is where OpenFOAM applies it. fixedTemperatureConstraint is what lands here, and OpenFOAM
        // pins he(p, Tuniform), not the temperature: setValues on the energy equation takes an ENERGY.
        // The driver converts, because only it knows the thermo. Applied here rather than inside
        // assembleEEqn because setValues writes psi as well as the matrix, and the assembly takes he
        // by const reference -- the solve owns it.
        if (in.fvoHeMask && in.fvoHeVal
            && in.fvoHeMask->size() == static_cast<std::size_t>(dm.nCells))
            deviceSetValues(dm, *in.fvoHeMask, *in.fvoHeVal, E.diag, E.upper, E.lower, E.source,
                            E.iC, E.bC, f.he);

        DeviceBuffer<scalar> diagC, b;
        deviceFold(dm, E.diag, E.source, E.iC, E.bC, diagC, b);
        const DeviceLduView A = foldedView(dm, E, diagC);
        DeviceBuffer<scalar> dnf;
        deviceNormFactorInto(A, f.he, b, w.ones, dnf);                // stays on the device (item 66)
        // The solver the case asked for (item 58): `smoothSolver` + a GaussSeidel-family smoother runs
        // OpenFOAM's own sweep, level-scheduled, under its stopping rule and its nSweeps; anything else
        // keeps BiCGStab and is announced. squareBend and angledDuct both name a smoothSolver here.
        DeviceSolverPerf perf;
        if (in.heSymGaussSeidel)
            deviceSymGaussSeidel(A, b, f.he, dnf.data(), in.tolHe, in.relTolHe, in.maxIterHe, &perf, in.minIterHe,
                                 in.nSweepsHe, in.heGaussSeidelSymmetric);
        else
            perf = deviceJacobiBiCGStab(A, b, f.he, dnf.data(), in.tolHe, in.relTolHe, in.maxIterHe, /*checkEvery=*/1, in.minIterHe,
                                        in.preconHe);
        res[in.isE ? "e" : "h"] = perf.initialResidual;

        // fvOptions.correct(he), EEqn.H:27 -- AFTER the solve and BEFORE thermo.correct(), which is what
        // makes it reach T at all. Applying it later would clamp an energy the thermo had already turned
        // into a temperature, and applying it earlier would clamp the field the solve is about to
        // overwrite.
        if (in.limitHe)
        {
            limitEnergyKernel<<<(nC + 255) / 256, 256>>>(nC, in.heMin, in.heMax, f.he.data());
            cudaCheck(cudaGetLastError(), "rhoSimpleFoam limitEnergy");
        }
        deviceBCValue(dbHe, f.he, f.heBnd);
    }

    // EEqn.H ends with thermo.correct(): T, and therefore psi, move HERE and everything below sees them.
    in.thermoCorrect();
    sd.scalars("he", f.he);
    sd.scalars("T", f.T);
    sd.scalars("psi", f.psi);

    // ---- pEqn.H or pcEqn.H -------------------------------------------------------------------
    phaseMark(&g_tE);
    RhoPressureInput pin;
    pin.rhoCell = &f.rho;            pin.rhoBndFace = &f.rhoBnd;
    pin.psiCell = &f.psi;            pin.psiBndFace = &f.psiBnd;
    pin.transonic = in.transonic;
    pin.relaxP = in.relaxPEqn;
    pin.relaxPSpecified = in.relaxPEqnSpecified;
    pin.pRefCell = in.pRefCell;      pin.pRefValue = in.pRefValue;
    pin.correctedLaplacian = in.correctedLaplacian;
    pin.snGradLimitCoeff = in.snGradLimitCoeff;
    pin.takeUAtBoundary = in.takeUAtBoundary;
    pin.adjustable = in.adjustable;
    pin.hasMRF = in.hasMRF;
    pin.hasFvOptions = in.hasFvOptions;
    pin.hasCoupledPatches = in.hasCoupledPatches;
    pin.fvOptionUnsupported = in.fvOptionUnsupported;

    RhoPressureStages        st;
    ConsistentPressureStages cst;
    bool closedVolume = false;

    if (in.consistent)
    {
        // pcEqn.H OPENS with `rho = thermo.rho()`. pEqn.H does not -- so the SIMPLEC pressure equation is
        // built from a density that already reflects the just-solved T and the plain SIMPLE one is not.
        in.updateRho();
        consistentPressurePredictor(cst, dm, dbU, dbP, UEqn, f.Ux, f.Uy, f.Uz, f.p, pin);
        sd.scalars("rAU", cst.rAU);
        sd.scalars("rAtU", cst.rAtU);
        sd.scalars("rhorAtU", cst.rhorAtU);
        sd.vectors("HbyA", cst.HbyA0[0], cst.HbyA0[1], cst.HbyA0[2]);
        sd.vectors("HbyAc", cst.HbyA[0], cst.HbyA[1], cst.HbyA[2]);
        sd.surface("phiHbyA0", cst.phiHbyA0Int, cst.phiHbyA0Bnd);
        sd.surface("phiHbyAc", cst.phiHbyAInt, cst.phiHbyABnd);
        sd.surface("phid", cst.phidInt, cst.phidBnd);
        sd.scalars("rhoP", f.rho);
        closedVolume = cst.closedVolume;
    }
    else
    {
        pressurePredictor(st, dm, dbU, dbP, UEqn, f.Ux, f.Uy, f.Uz, f.p, pin);
        closedVolume = st.closedVolume;
    }

    // totalPressure's updateCoeffs, where OpenFOAM runs it: inside the pressure fvMatrix's constructor
    // (fvMatrix.C:396; totalPressureFvPatchScalarField.C:152-225, the psiName_ == "none" branch), from
    // U's PATCH value as the momentum solve left it (f.UxBnd, refreshed above), the flux the iteration
    // started with, and rho's patch value as it stands -- pcEqn.H:1's on the SIMPLEC branch, the previous
    // tail's on the other.
    // ...and freestreamPressure's valueFraction, from U's patch value as the momentum solve left it.
    if (in.hasMixed) deviceUpdateMixedFreestream(dbU, dbP, f.phiBnd, f.Ux, f.Uy, f.Uz, &f.rhoBnd, /*which=*/2);
    deviceUpdateTotalPressure(dbP, f.phiBnd, f.UxBnd, f.UyBnd, f.UzBnd, &f.rhoBnd);
    deviceBCValue(dbP, f.p, f.pBnd);

    // The non-orthogonal corrector loop. solutionControlI.H:78-95 runs it nNonOrth+1 times, and only the
    // FINAL pass writes phi (simple.finalNonOrthogonalIter()).
    const label nCorr = in.nNonOrthogonalCorrectors + 1;
    for (label corr = 1; corr <= nCorr; ++corr)
    {
        PressureMatrix& P = w.P;                          // persistent -- see RhoSolverWorkspace
        if (in.consistent) assemblePcEqn(P, cst, dm, dbP, f.p, pin);
        else               assemblePEqn(P, st, dm, dbP, f.p, pin);

        DeviceBuffer<scalar>& diagC = w.diagC;
        DeviceBuffer<scalar>& b     = w.b;
        deviceFold(dm, P.diag, P.source, P.iC, P.bC, diagC, b);
        const DeviceLduView A = foldedView(dm, P, diagC);
        DeviceBuffer<scalar> dnf;
        deviceNormFactorInto(A, f.p, b, w.ones, dnf);                 // stays on the device (item 66)

        DeviceSolverPerf perf;
        if (in.transonic)
        {
            // fvm::div(phid, p) makes lower = -w*phi and upper = lower + phi, so upper != lower at every
            // face with flow through it. A symmetric solver on that matrix is not slow, it is wrong: CG
            // burned the full 3000-iteration cap and the case stalled before printing iteration 1.
            perf = deviceJacobiBiCGStab(A, b, f.p, dnf.data(), in.tolP, in.relTolP, in.maxIterP, in.pcgCheckEvery, in.minIterP);
        }
        else
        {
            if (!w.amgBuilt)
            {
                // Face weights SLICED TO INTERNAL FACES. DeviceMesh::magSf is |Sf| over ALL faces, laid
                // out [internal | non-cyclic boundary], while owner/nei are internal-only -- handing the
                // whole array to buildAMG pairs an internal-face addressing with a weight array that
                // runs on into the boundary, and every decision the agglomeration makes (which faces are
                // strong, hence which cells merge) is downstream of that pairing.
                const std::vector<label>  own = dm.owner.host(), nei = dm.nei.host();
                const std::vector<scalar> magSfAll = dm.magSf.host();
                const std::size_t nIf = static_cast<std::size_t>(dm.nInternalFaces);
                const std::vector<label>  ownInt(own.begin(), own.begin() + std::min(nIf, own.size()));
                const std::vector<label>  neiInt(nei.begin(), nei.begin() + std::min(nIf, nei.size()));
                const std::vector<scalar> fw(magSfAll.begin(),
                                             magSfAll.begin() + std::min(nIf, magSfAll.size()));
                // The hierarchy is a function of the MESH: only the STRUCTURE is serialised, and
                // cDiag/cUpper/cLower are Galerkin-rebuilt every step. So a cache a simpleFoam run wrote
                // for this mesh is valid here, and until this line existed the compressible path started
                // cold on every run even when one was sitting next to constant/polyMesh.
                w.amg = in.amgCacheDir.empty()
                      ? buildAMG(ownInt, neiInt, fw, dm.nCells)
                      : buildOrLoadAMG(ownInt, neiInt, fw, dm.nCells, in.amgCacheDir, true);
                w.amgBuilt = true;
            }
            amgGalerkin(w.amg, diagC, P.upper, P.lower);
            perf = deviceAMGPCG(A, w.amg, b, f.p, dnf.data(), in.tolP, in.relTolP, in.maxIterP,
                                in.captureVcycle, in.pcgCheckEvery, /*corrScaling=*/false, in.minIterP);
        }
        // solutionControl.C:230-233 takes sp.first() -- the FIRST solve of the iteration, not the last.
        if (corr == 1) res["p"] = perf.initialResidual;

        if (corr == nCorr)
        {
            correctFluxCompressible(f.phiInt, f.phiBnd,
                                    in.consistent ? cst.phiHbyAInt : st.phiHbyAInt,
                                    in.consistent ? cst.phiHbyABnd : st.phiHbyABnd,
                                    P, dm, dbP, f.p);
        }
    }

    // p.relax() -- the FIELD factor, not the equation one. Both are spelled `p` in fvSolution and they
    // live in different sub-dictionaries; using the equation factor here relaxes the wrong thing.
    // AFTER the flux correction and BEFORE the velocity correction, so phi is built from the unrelaxed
    // pressure and U from the relaxed one.
    //
    // BOTH HALVES. GeometricField::relax is operator==(prevIter() + alpha*(*this - prevIter()))
    // (GeometricField.C:1094) and operator== assigns the boundary too (:1420), after the solve's own
    // correctBoundaryConditions (fvMatrixSolve.C:309) has put the solved cell value on every zeroGradient
    // face -- hence the boundary evaluation BEFORE the cells are relaxed. The blend is what
    // U = HbyA - rAU*grad(p) reads next and, without a pressure limiter, what the next momentum assembly
    // reads. On the totalPressure inlet it is a different number from both the fresh p0 - 0.5*rho*|U_b|^2
    // and the previous one (rhoTP at t=1 blends the 100200 seed towards 100095 at 0.3); everywhere else
    // it is the same number the old evaluate-after-relax produced.
    deviceBCValue(dbP, f.p, f.pBnd);
    relaxField(f.p, pPrev, in.relaxP);
    relaxField(f.pBnd, pBndPrev, in.relaxP);
    sd.scalars("pRel", f.p);
    sd.surface("phi", f.phiInt, f.phiBnd);
    if (dbP.n > 0)
    {
        storeTotalPressureValueKernel<<<(dbP.n + 255) / 256, 256>>>(
            dbP.n,
            dbP.tpMask.data(),
            f.pBnd.data(),
            dbP.refValue.data());
        cudaCheck(cudaGetLastError(), "rhoSimpleFoam storeTotalPressureValue");
    }

    // U = HbyA - rAtU*fvc::grad(p), with rAtU on the SIMPLEC path and rAU otherwise.
    {
        DeviceBuffer<scalar> gpx, gpy, gpz;
        deviceGaussGrad(dm, f.p, f.pBnd, gpx, gpy, gpz);
        PressureStages shim;
        if (in.consistent)
        {
            for (int k = 0; k < 3; ++k) deviceCopy(shim.HbyA[k], cst.HbyA[k]);
            deviceCopy(shim.rAtU, cst.rAtU);
        }
        else
        {
            for (int k = 0; k < 3; ++k) deviceCopy(shim.HbyA[k], st.HbyA[k]);
            deviceCopy(shim.rAtU, st.rAU);
        }
        correctVelocity(f.Ux, f.Uy, f.Uz, shim, gpx, gpy, gpz);

        // U.correctBoundaryConditions() -- pEqn.H:87 and pcEqn.H:100, on the line IMMEDIATELY after
        // `U = HbyA - rAU*fvc::grad(p)`. The driver refreshed U's boundary once, before the energy
        // equation, and never again: from the velocity correction onward f.UxBnd/UyBnd/UzBnd held the
        // values U had BEFORE the pressure correction, for the rest of the iteration and into the next.
        //
        // Found by comparing every field against the host reference rather than the handful the driver
        // gate reports. At the end of iteration 1 on sbMatched every reported field agreed to ~1e-12
        // while UxBnd was out by 6.81e-01 and UyBnd/UzBnd by 1.00e+00 -- entirely different values, not
        // a drift. It fed forward through everything that reads U's patch values: fvc::div(phi, Ekp)
        // evaluates Ekp on boundary faces, the closure's production and its turbulentIntensity inlet
        // both read U_b, and the next iteration's flux switch reads the boundary flux built from it.
        //
        // The symmetry/wedge refValue is rebuilt first, for the reason given at the energy equation's
        // refresh above: their value is a function of the CURRENT internal field, and the velocity
        // correction has just moved it.
        //
        // And every flux-switched patch evaluates through its updateCoeffs again here (the updated flag
        // was cleared by the solve's evaluate), with phi the NEW flux by now (pEqn.H:73): inletOutlet's
        // valueFraction and the piov mask are rebuilt from it, the piov value from the corrected cells --
        // what gets written, what the closure reads, and what the next momentum assembly finds.
        deviceUpdateInletOutlet(dbU, f.phiBnd);
        deviceUpdatePressureInletOutletVelocity(dbU, f.phiBnd, f.Ux, f.Uy, f.Uz, /*directionMixed=*/true);
        // flowRateInletVelocity recomputed from rho's patch value as it stands here (see updateFlowRateInlets).
        updateFlowRateInlets(f, in, dbU);
        // freestreamVelocity's valueFraction rebuilt from the patch's current value, as OpenFOAM's evaluate
        // does here (freestreamVelocityFvPatchVectorField.C:106; the host step carries the measurement).
        if (in.hasMixed) deviceUpdateMixedFreestream(dbU, dbP, f.phiBnd, f.Ux, f.Uy, f.Uz, &f.rhoBnd, /*which=*/1);
        deviceUpdateSymmetry(dbU, f.Ux, f.Uy, f.Uz);
        deviceUpdateWedge(dbU, f.Ux, f.Uy, f.Uz);
        deviceBCValue(dbU.comp[0], f.Ux, f.UxBnd);
        deviceBCValue(dbU.comp[1], f.Uy, f.UyBnd);
        deviceBCValue(dbU.comp[2], f.Uz, f.UzBnd);
    }

    sd.vectors("Upost", f.Ux, f.Uy, f.Uz);

    // pressureControl.limit(p), HERE and not earlier: pEqn.H applies it after the velocity correction, so
    // U is built from the unclipped pressure and only p carries the clip.
    const bool pLimited = in.limitMaxP || in.limitMinP;
    if (pLimited)
    {
        limitPressureKernel<<<(nC + 255) / 256, 256>>>(nC, in.limitMaxP ? 1 : 0, in.limitMinP ? 1 : 0,
                                                       in.pMaxLimit, in.pMinLimit, f.p.data());
        cudaCheck(cudaGetLastError(), "rhoSimpleFoam limitPressure");
    }

    // The closed-volume mass correction. `closedVolume` is set by the predictor, on the same condition
    // adjustPhi and pRefCell are: no patch fixes a pressure value, so the level is undetermined and the
    // total mass is what pins it.
    if (closedVolume)
    {
        closedVolumeCorrection(f.p, f.psi, dm, f.initialMass);
    }

    // ONE refresh for both, keyed as OpenFOAM keys it: `if (pLimited || closedVolume)` (pEqn.H:100-103).
    // On the totalPressure inlet that correctBoundaryConditions is an updateCoeffs (the flag was cleared
    // by the solve's evaluate): a recompute from the NEW flux, the corrected U's patch value and rho's
    // patch value as it stands -- BEFORE the tail's rho = thermo.rho() below. It is the value written to
    // disk and the one the next momentum assembly reads through grad(p).
    if (pLimited || closedVolume)
    {
        if (in.hasMixed) deviceUpdateMixedFreestream(dbU, dbP, f.phiBnd, f.Ux, f.Uy, f.Uz, &f.rhoBnd, /*which=*/2);
        deviceUpdateTotalPressure(dbP, f.phiBnd, f.UxBnd, f.UyBnd, f.UzBnd, &f.rhoBnd);
        deviceBCValue(dbP, f.p, f.pBnd);
    }

    // rho = thermo.rho(), then rho.relax() -- only when NOT transonic.
    //
    // THE BOUNDARY IS RELAXED TOO. GeometricField::relax is operator==(prevIter + alpha*(*this -
    // prevIter)) and operator== assigns BOTH halves. Relaxing only the internal field leaves rho's patch
    // values at the unrelaxed thermo value, which is a different field from the one OpenFOAM carries --
    // and it matters directly, because flowRateInletVelocity holds the prescribed mass flow against
    // rho's PATCH values, so an unrelaxed boundary sets a different inlet velocity every iteration.
    {
        // Against rhoPrevIter, captured at the top of the step -- see the note there.
        in.updateRho();
        if (!in.transonic)
        {
            relaxField(f.rho, rhoPrevIter, in.relaxRho);
            relaxField(f.rhoBnd, rhoBndPrevIter, in.relaxRho);
        }
    }

    sd.scalars("p", f.p);
    sd.scalars("rhoTail", f.rho);
    sd.scalars("kIn", f.k);
    sd.scalars("epsIn", f.epsilon);
    sd.scalars("nutIn", f.nut);

    // turbulence->correct() -- LAST, so the NEXT iteration's momentum equation uses this iteration's
    // closure. OpenFOAM's lagged coupling.
    phaseMark(&g_tP);
    if (in.correct) in.correct();
    phaseMark(&g_tTurb);
    sd.scalars("kOut", f.k);
    sd.scalars("epsOut", f.epsilon);
    sd.scalars("nutOut", f.nut);

    return res;
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae


