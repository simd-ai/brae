#pragma once
// device_scalar_transport.cuh -- the GENERIC scalar-transport solve scaffold, templated on the model Reaction
// functor. Assembles div(phi,f) [+ -Sp(div(phi),f) if bounded] + laplacian(D,f) with the deferred limited/
// linearUpwind/non-orth corrections, adds the model reaction (diag+source), relaxes, applies the eps near-wall
// setValues, couples cyclic/AMI interfaces, solves (BiCGStab or symGaussSeidel), and bounds. Shared by k/epsilon/
// omega/nuTilda today and by energy/species/compressible transport later. Extracted verbatim from device_kepsilon.cu.
#include "device_kepsilon.cuh"    // deviceBoundField + DeviceMesh/DeviceBoundary/DeviceWallData
#include "device_ldu.cuh"
#include "device_dilu.cuh"      // OF DILU preconditioner (null -> Jacobi)
#include "device_pcg.cuh"         // deviceJacobiBiCGStab
#include "device_simple.cuh"      // deviceFold/deviceRelaxDiag/deviceDiv*Coeffs/deviceLinearUpwindCorr/...
#include "device_blas.cuh"
#include "device_ami.cuh"
#include "device_cyclic.cuh"
#include "device_interface.cuh"   // interfaceAssembleMomentum/OffDiagSum/ZeroWallIfCoeff
#include "device_amg.cuh"         // deviceSymGaussSeidel
#include "stage_dump.cuh"      // Phase 0 stage harness
#include "device_ddt.cuh"         // ScalarDdt + deviceFvmDdtDiag/Source (transient turbulence)
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>

namespace brae {

// --- scaffold-local helpers (moved from device_kepsilon.cu so the template above sees them) ---
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }
// OF-style turbulence residual report store; clearTurbulenceReport/turbulenceReport (device_kepsilon.cu) wrap this.
inline std::vector<ScalarSolveEntry>& turbStore() { static std::vector<ScalarSolveEntry> s; return s; }

// The DILU preconditioner every turbulence solve uses, or null for Jacobi. Held here, in the same
// file-scope idiom as turbStore above, rather than threaded through deviceKEpsilonCorrect,
// deviceKOmegaSSTCorrect, deviceSpalartCorrect and deviceEnergyCorrect: it is one object shared by all of
// them (the level schedule depends only on the mesh, and diluUpdate recomputes rD from whichever matrix
// is being solved), so a parameter on four signatures would carry the same pointer four times.
//
// The solver sets it once at construction and it stays put; nothing else writes it.
inline const DeviceDilu*& turbPrecon() { static const DeviceDilu* p = nullptr; return p; }

namespace {
// setValues (eps wall constraint): zero wall-cell off-diagonals + move the known eps0 to the neighbour RHS.
//
// DETERMINISM. This was one kernel, one thread per internal FACE, moving the contribution with
//     if (ow) atomicAdd(&source[n], -lower[f]*eps0[o]);
//     if (nw) atomicAdd(&source[o], -upper[f]*eps0[n]);
// A cell receives one such contribution per constrained face it touches, so the summation order followed
// face scheduling. It is rare -- it only bites a cell in the near-wall band with more than one constrained
// face -- which is exactly what made it hard to see: pitzDaily/kEpsilon came out bit-identical at 1, 5, 8,
// 10 and 15 iterations and differed at 12.
//
// Split into a GATHER over cells (fixed order, no atomics) and a separate zeroing pass over faces. The
// split is required, not cosmetic: the gather must read the ORIGINAL upper/lower, and a single kernel
// cannot order "everyone reads" before "everyone zeroes". Two launches give that ordering for free.
//
// Per cell c the contributions are, in this order:
//   faces where c is OWNER      f in [ownerStart[c], ownerStart[c+1])   ->  -upper[f]*eps0[nei[f]]
//   faces where c is NEIGHBOUR  f = losort[k], k in [losortStart[c], losortStart[c+1])
//                                                                      ->  -lower[f]*eps0[own[f]]
// which is the same set of terms the scatter produced, just accumulated in a fixed sequence.
__global__
void svGatherKernel(
    int nC,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ isW,
    const scalar* __restrict__ eps0,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ lower,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = 0.0;
    for (label f = ownerStart[c]; f < ownerStart[c+1]; ++f)
    {
        const int n = nei[f];
        if (isW[n]) s -= upper[f] * eps0[n];
    }
    for (label k = losortStart[c]; k < losortStart[c+1]; ++k)
    {
        const label f = losort[k];
        const int o = own[f];
        if (isW[o]) s -= lower[f] * eps0[o];
    }
    if (s != 0.0) source[c] += s;
}
// Zeroing pass: independent per face, no accumulation, so it needs no ordering guarantee of its own.
__global__
void svZeroFaceKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const label* __restrict__ isW,
    scalar* __restrict__ upper,
    scalar* __restrict__ lower)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    if (isW[own[f]] || isW[nei[f]]) { upper[f] = 0.0; lower[f] = 0.0; }
}


__global__
void svBndKernel(
    int nB,
    const label* __restrict__ faceCell,
    const label* __restrict__ isW,
    scalar* __restrict__ iC,
    scalar* __restrict__ bC)
{
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi < nB && isW[faceCell[bi]]) { iC[bi] = 0.0; bC[bi] = 0.0; }
}


__global__
void svCellKernel(
    int nC,
    const label* __restrict__ isW,
    const scalar* __restrict__ relaxedDiag,
    const scalar* __restrict__ eps0,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC && isW[c]) source[c] = relaxedDiag[c] * eps0[c];
}
// shared turbulence-common kernels (effective diffusivity D + OF bound override); used by k/eps + k-omega.
static __global__
void depsKernel(int nC, const scalar* __restrict__ nut, scalar sigma, scalar nu, scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) D[c] = nut[c] / sigma + nu;
}


static __global__
// OF epsilonWallFunctionFvPatchScalarField::updateCoeffs(weights), epsilonWallFunction.C:586 --
//     G[celli]       = (1 - w)*G[celli]       + w*G0[celli];
//     epsilon[celli] = (1 - w)*epsilon[celli] + w*epsilon0[celli];
// applied only where w > tolerance_. w is the non-overlap patch's weight, (1 - ACMI mask), and is 1 on
// every ordinary wall -- so this reduces to the plain override everywhere except a partially covered
// cyclicACMI face, which is the only place a wall is a fraction of a wall. `wallW` may be null for a
// caller with no such patches, in which case isW alone decides, exactly as before.
void overrideKernel(
    int nC,
    const label* __restrict__ isW,
    const scalar* __restrict__ G0,
    scalar* __restrict__ eps0,
    scalar* __restrict__ G,
    scalar* __restrict__ eps,
    const scalar* __restrict__ wallW = nullptr)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC || !isW[c]) return;
    const scalar w = wallW ? wallW[c] : scalar(1);
    const scalar e = (scalar(1) - w)*eps[c] + w*eps0[c];
    G[c]   = (scalar(1) - w)*G[c]   + w*G0[c];
    eps[c] = e;
    // ...AND THE MATRIX CONSTRAINT TAKES THE BLENDED VALUE, not the raw near-wall one.
    // epsilonWallFunctionFvPatchScalarField::manipulateMatrix appends `epsilon[celli]` -- the field
    // updateCoeffs(weights) has just blended -- and hands that to matrix.setValues. brae blended the
    // field but then constrained the matrix to eps0, the UNBLENDED wall value. Identical on an ordinary
    // wall, where w = 1 and the blend is the identity; different on exactly the cells a cyclicACMI
    // partially covers. Measured on oscillatingInletACMI2D: epsilon on those cells was 0.72x OpenFOAM's
    // while the median ACMI cell was already right to 1.7e-07.
    eps0[c] = e;
}
} // anon (scaffold setValues kernels)


// One scalar-transport sub-step of a two-equation RAS correct(): assemble div(phi,f) - laplacian(D,f)
// [- Sp(div(phi),f) if bounded], add the model reaction (diag+source via `reaction`), relax, optionally
// apply the eps near-wall setValues constraint, fold, BiCGStab (loose relTol), bound. Reused by k-omega SST.
template <class Reaction>
void deviceSolveScalarTransport(
    const DeviceMesh& dm,
    const DeviceBoundary& db,
    DeviceBuffer<scalar>& field,
    const char* fieldName,
    const DeviceBuffer<scalar>& D,
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
    scalar relTolKE,
    int keCheckEvery,
    bool useGS,
    Reaction&& reaction,
    const DeviceWallData* wall = nullptr,
    const DeviceBuffer<scalar>* eps0 = nullptr,
    DeviceAMI* ami = nullptr,
    DeviceCyclic* cyc = nullptr,
    const ScalarDdt& ddt = ScalarDdt{},   // transient fvm::ddt(f); default (steady) -> no-op
    const DeviceBuffer<scalar>* DBnd = nullptr,   // per-FACE boundary diffusivity; null -> adjacent-cell value
    // OF `grad(<field>) cellLimited Gauss linear <k>`. 0 = unlimited. The gradient feeds the limitedLinear
    // weight, the linearUpwind deferred correction and the non-orthogonal laplacian correction, so an
    // unlimited gradient on a field the case asked to limit is a different discretisation, not a detail.
    // Only grad(U) was ever honoured; grad(k)/grad(omega)/grad(e) lines were parsed by nothing.
    scalar gradLimitK = 0.0,
    // OF's bound(f, fMin) exists for POSITIVE-DEFINITE quantities: k, epsilon, omega, nuTilda. Every
    // turbulence caller here is one of those, so the default is true. The ENERGY is not: EEqn.H never
    // bounds he, and OF's sensible energy is legitimately NEGATIVE below the reference temperature
    // (air at 298 K has he = -8.59e4 J/kg). Bounding it there replaces the whole field with ~0 and
    // pins T at Cp*Tref/Cv = 417.7 K, which is exactly what NACA0012 did once he was given OF's
    // reference point. Harmless while brae used he = Cv*T > 0 -- which is why it survived this long.
    bool boundPositive = true,
    // fvOptions scalarFixedValueConstraint: OF FixedValueConstraint::constrain -> eqn.setValues(cells, value).
    // Applied BEFORE the wall block, because kEpsilon.C runs fvOptions.constrain(epsEqn) at line 266 and
    // boundaryManipulate (the epsilon wall function's own setValues) at 267 -- so on a cell claimed by both,
    // the WALL FUNCTION wins. Same three kernels; only the mask and the values differ.
    const DeviceBuffer<label>* fvoSetMask = nullptr,
    const DeviceBuffer<scalar>* fvoSetVal = nullptr,
    // OF limitFuncs: a limited scheme on a NON-SCALAR field builds its limiter from one derived scalar
    // (magSqr of the tensor), so every component shares a single per-face limiter -- it is not six
    // independent limiters. Pass that scalar and its gradient here; null means "build it from `field`
    // itself", which is what every scalar caller wants and leaves them byte-for-byte unchanged.
    const DeviceBuffer<scalar>* limField = nullptr,
    const DeviceBuffer<scalar>* limGradX = nullptr,
    const DeviceBuffer<scalar>* limGradY = nullptr,
    const DeviceBuffer<scalar>* limGradZ = nullptr,
    // OF DILU preconditioner for this field's BiCGStab; null keeps Jacobi. LAST in the list so every
    // existing positional call is untouched. The level schedule depends only on the mesh, so ONE
    // instance serves every field -- diluUpdate recomputes rD from the current matrix per solve.
    const DeviceDilu* precon = nullptr,
    // fvSolution solvers/<field>/nSweeps (smoothSolver.C:78, default 1): smoothing sweeps between
    // residual EVALUATIONS, so the stop test is consulted only on a multiple of it and the solve
    // overshoots its relTol by whatever the extra sweeps buy. Trailing, like `precon`, so every existing
    // positional call keeps its behaviour exactly. Meaningful only on the smoothSolver path -- the
    // BiCGStab branch below has no such control, which is OpenFOAM's rule too (nSweeps lives on
    // smoothSolver alone).
    int nSweeps = 1,
    // WHICH GaussSeidel smoother the case named. true = symGaussSeidel (ascending then descending);
    // false = GaussSeidel, whose sweep loop in GaussSeidelSmoother.C is the ascending walk ONLY. They
    // are different smoothers, so the same relTol stops in a different place. Trailing, like nSweeps.
    bool gsSymmetric = true)
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> Df;
    deviceInterpolate(dm, D, Df);
    DeviceBuffer<scalar> aD, aU, aL, lD, lU, lL, luCorr, lapCorr;   // luCorr/lapCorr = linearUpwind / non-orth deferred sources (empty otherwise)
    DeviceBuffer<scalar> gx, gy, gz;         // gradSchemes `default`: limitedLinear's limiter + non-orth
    DeviceBuffer<scalar> lgx, lgy, lgz;      // the gradient linearUpwind NAMES (limited when that name is cellLimited)
    // ONE GRADIENT CANNOT SERVE ALL THREE. Only linearUpwind takes a named gradient
    // (linearUpwind.C: mesh.gradScheme(gradSchemeName_)); limitedLinear's limiter and the non-orthogonal
    // laplacian correction both resolve gradSchemes `default`, which is plain Gauss linear here. Limiting
    // the shared buffer in place therefore over-limited the other two -- measured on squareBendLiq as a
    // transient nut excursion to 1.5e+01 against OF's 1.4e-03.
    const bool limitLU = (gradLimitK > 0.0) && linearUpwind;
    if (limited || linearUpwind || nonOrth)
    {
        DeviceBuffer<scalar> bv;
        deviceBCValue(db, field, bv);
        deviceGaussGrad(dm, field, bv, gx, gy, gz);
        // THE COUPLED FACES, which this gradient never had. deviceGaussGrad sums over the internal and
        // non-coupled boundary faces; a cyclic/AMI/ACMI face is in neither list, so the gradient of k,
        // epsilon, omega, nuTilda and he simply stopped at the interface -- while the MOMENTUM predictor
        // has added them all along. All three consumers below read this buffer: limitedLinear's weight,
        // linearUpwind's deferred correction and the non-orthogonal laplacian correction.
        if (cyc && cyc->n > 0) interfaceAddGrad(*cyc, field, dm.V, gx, gy, gz);
        if (ami && ami->n > 0) interfaceAddGrad(*ami, field, dm.V, gx, gy, gz);
        if (limitLU)
        {
            deviceCopy(lgx, gx); deviceCopy(lgy, gy); deviceCopy(lgz, gz);
            // ...and the limiter has to see them too: OF's cellLimitedGrad folds a coupled patch's
            // patchNeighbourField into its range and clips the extrapolation to that face. A SCALAR is
            // never rotated across the interface, so the neighbour value is the raw (AMI-interpolated)
            // cell value. See CellLimitInterface and test_cell_limit_interface.
            CellLimitInterface ifs[2];
            int nIfs = 0;
            DeviceBuffer<scalar> cycNbr, amiNbr, empty;
            if (cyc && cyc->n > 0)
            {
                deviceCyclicNbrValue(*cyc, field, empty, empty, empty, 0, cycNbr);
                ifs[nIfs++] = { cyc->n, cyc->ownCell.data(), cycNbr.data(),
                                cyc->dOwnX.data(), cyc->dOwnY.data(), cyc->dOwnZ.data() };
            }
            if (ami && ami->n > 0)
            {
                deviceAmiInterpolate(*ami, field, amiNbr);
                ifs[nIfs++] = { ami->n, ami->ownCell.data(), amiNbr.data(),
                                ami->dOwnX.data(), ami->dOwnY.data(), ami->dOwnZ.data() };
            }
            deviceCellLimitGrad(dm, field, bv, lgx, lgy, lgz, gradLimitK, ifs, nIfs);
        }
    }
    const bool sharedLim = limField && limGradX && limGradY && limGradZ;
    if (limited)
    {
        deviceDivLimitedCoeffs(dm, phiInt,
                               sharedLim ? *limField  : field,
                               sharedLim ? *limGradX  : gx,
                               sharedLim ? *limGradY  : gy,
                               sharedLim ? *limGradZ  : gz,
                               twoByk, aD, aU, aL);   // Gauss limitedLinear/vanAlbada: implicit limited weight
    }
    else
    {
        deviceDivUpwindCoeffs(dm, phiInt, aD, aU, aL);
        if (linearUpwind)
        {
            deviceLinearUpwindCorr(dm, phiInt, limitLU ? lgx : gx, limitLU ? lgy : gy, limitLU ? lgz : gz, luCorr);   // Gauss linearUpwind: upwind matrix + deferred corr
            // ...and at the interface, which OF's linearUpwind::correction() reaches through its
            // `if (pSfCorr.coupled())` boundary loop. The momentum predictor has done this since the
            // interface work; the scalar path never did. rotate=false: a scalar is not transformed.
            const DeviceBuffer<scalar>& lx = limitLU ? lgx : gx;
            const DeviceBuffer<scalar>& ly = limitLU ? lgy : gy;
            const DeviceBuffer<scalar>& lz = limitLU ? lgz : gz;
            if (cyc && cyc->n > 0) interfaceAddLinUpwindCorr(*cyc, lx, ly, lz, luCorr);
            if (ami && ami->n > 0) interfaceAddLinUpwindCorr(*ami, lx, ly, lz, luCorr);
        }
    }
    // laplacian "corrected": nonOrthDeltaCoeffs implicit (in deviceLaplacianCoeffs) + corrVec.grad(field) explicit (deviceLaplacianCorr).
    deviceLaplacianCoeffs(dm, Df, lD, lU, lL, nonOrth);
    deviceAxpy(-1.0, lD, aD); deviceAxpy(-1.0, lU, aU); deviceAxpy(-1.0, lL, aL);
    if (nonOrth)
    {
        deviceLaplacianCorr(dm, Df, gx, gy, gz, lapCorr);
        // The interface's own non-orthogonal correction. An AMI face is non-conformal by construction, so
        // its corrVec is NOT small even on a mesh that is orthogonal everywhere else -- the one place a
        // "corrected" laplacian actually has work to do on this fixture.
        if (cyc && cyc->n > 0) interfaceAddLapCorr(*cyc, D, gx, gy, gz, lapCorr);
        if (ami && ami->n > 0) interfaceAddLapCorr(*ami, D, gx, gy, gz, lapCorr);
    }
    if (bounded) { DeviceBuffer<scalar> bt; deviceHadamard(bt, divU, dm.V); deviceAxpy(-1.0, bt, aD); }   // -Sp(div(phi),f)
    DeviceBuffer<scalar> src(static_cast<std::size_t>(nC));
    cudaCheck(cudaMemsetAsync(src.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "src zero");
    // BRAE_DUMP_TERMS=<dir>: every contribution to this field's equation, separately, per cell, per call.
    // A global norm cannot say WHICH term drives a localised excursion -- the deferred linearUpwind source
    // and the reaction production are added to the same RHS and are indistinguishable afterwards. Captured
    // here, before they are folded together.
    DeviceBuffer<scalar> dgConv, dgReact, srcReact;
    const char* termDir = std::getenv("BRAE_DUMP_TERMS");
    if (termDir) { deviceCopy(dgConv, aD); }
    reaction(aD, src);                                            // model reaction: adds to diag + source
    if (termDir)
    {
        deviceCopy(dgReact, aD); deviceCopy(srcReact, src);
        static int callNo = 0;
        const int myCall = callNo++;
        std::error_code tec;
        std::filesystem::create_directories(termDir, tec);
        char fn[512];
        std::snprintf(fn, sizeof fn, "%s/%s_%04d", termDir, fieldName, myCall);
        std::ofstream o(fn);
        o.precision(10);
        const std::vector<scalar> hF = field.host(), hDc = dgConv.host(), hDr = dgReact.host(), hSr = srcReact.host();
        const std::vector<scalar> hLU = luCorr.size()  ? luCorr.host()  : std::vector<scalar>(nC, 0.0);
        const std::vector<scalar> hLC = lapCorr.size() ? lapCorr.host() : std::vector<scalar>(nC, 0.0);
        o << "# cell field diagConvLap diagAfterReact srcReact luCorr lapCorr\n";
        for (int c = 0; c < nC; ++c)
            o << c << ' ' << hF[c] << ' ' << hDc[c] << ' ' << hDr[c] << ' '
              << hSr[c] << ' ' << hLU[c] << ' ' << hLC[c] << '\n';
    }
    if (luCorr.size())  deviceAxpy(-1.0, luCorr, src);           // linearUpwind deferred correction (explicit RHS)
    if (lapCorr.size()) deviceAxpy(-1.0, lapCorr, src);          // non-orth laplacian correction (explicit RHS, mirrors momentum)
    DeviceBuffer<scalar> aIC, aBC, lIC, lBC; deviceBCDivCoeffs(db, phiBnd, aIC, aBC);
    // OF evaluates the laplacian coefficient with the PATCH diffusivity (gamma.boundaryField()), not the
    // adjacent cell's. They coincide for a zeroGradient wall (no flux anyway) but not at a fixedValue one,
    // which is where a wall function puts its whole effect -- so the energy equation supplies DBnd.
    if (DBnd && DBnd->size()) deviceBCLaplacianCoeffsFace(db, *DBnd, lIC, lBC);
    else                      deviceBCLaplacianCoeffs(db, D, lIC, lBC);
    deviceAxpy(-1.0, lIC, aIC); deviceAxpy(-1.0, lBC, aBC);
    // interface (cyclic/cyclicAMI) coupling: fold div(phi,f) - laplacian(D,f) at the interface into the diagonal and
    // set the off-diagonal ifCoeff. A scalar is invariant under the cyclic transform (no rotation of the value), so the
    // translational momentum assembly + a plain weighted off-diagonal apply even for a ROTATIONAL interface.
    //
    // ...WITH THE SCHEME'S OWN FACE WEIGHT, not upwind's. `limited` means the case named a TVD scheme for
    // div(phi,f) -- the SST and pipeCyclic tutorials all say `bounded Gauss limitedLinear 1` on k,
    // epsilon, omega and nuTilda -- and OpenFOAM limits a coupled face exactly as it limits an internal
    // one. Assembling the interface upwind regardless is the same defect the momentum predictor carried,
    // reaching every turbulence scalar through this one call. Null for upwind and linearUpwind, whose
    // matrix IS upwind's.
    DeviceBuffer<scalar> ifWsch;
    if (limited)
    {
        const DeviceBuffer<scalar>& lf = sharedLim ? *limField : field;
        const DeviceBuffer<scalar>& lx = sharedLim ? *limGradX : gx;
        const DeviceBuffer<scalar>& ly = sharedLim ? *limGradY : gy;
        const DeviceBuffer<scalar>& lz = sharedLim ? *limGradZ : gz;
        if (ami && ami->n)      deviceAmiLimitedWeights(*ami, lf, lx, ly, lz, twoByk, ifWsch);
        else if (cyc && cyc->n) deviceCyclicLimitedWeights(*cyc, lf, lx, ly, lz, twoByk, ifWsch);
    }
    const DeviceBuffer<scalar>* ifW = ifWsch.size() ? &ifWsch : nullptr;
    DeviceBuffer<scalar> ifSumOff;
    if (ami && ami->n) { interfaceAssembleMomentum(*ami, D, aD, ifW);
        ifSumOff.copyFrom(std::vector<scalar>(nC, 0.0)); interfaceOffDiagSum(*ami, ifSumOff); }
    else if (cyc && cyc->n) { interfaceAssembleMomentum(*cyc, D, aD, ifW);
        ifSumOff.copyFrom(std::vector<scalar>(nC, 0.0)); interfaceOffDiagSum(*cyc, ifSumOff); }
    // implicit fvm::ddt(f) (URANS transient turbulence): the diagonal into the assembled aD (BEFORE relax = OF assembles
    // ddt into the eqn then relaxes), the source (old-time) into src. steady (ddt.c.active==false) -> exact no-op, so this
    // stays byte-for-byte the steady scalar transport. rho=1 (incompressible). Matches the momentum ddt wiring.
    deviceFvmDdtDiag(dm.V, ddt.c, 1.0, aD);
    if (ddt.old) { DeviceBuffer<scalar> e2; deviceFvmDdtSource(dm.V, ddt.c, 1.0, *ddt.old, ddt.old2 ? *ddt.old2 : e2, src, ddt.ddt0); }
    DeviceBuffer<scalar> aRD, aDelta; deviceRelaxDiag(deviceLduView(dm, aD, aU, aL), dm, aIC, relax, aRD, aDelta,
                                                      ifSumOff.size() ? ifSumOff.data() : nullptr);
    { DeviceBuffer<scalar> t; deviceHadamard(t, aDelta, field); deviceAxpy(1.0, t, src); }
    if (fvoSetMask && fvoSetVal)   // fvOptions scalarFixedValueConstraint (OF: before boundaryManipulate)
    {
        svGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(),
                                            dm.losort.data(), dm.losortStart.data(), fvoSetMask->data(), fvoSetVal->data(),
                                            aU.data(), aL.data(), src.data());
        svZeroFaceKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(), fvoSetMask->data(), aU.data(), aL.data());
        svBndKernel<<<nBlocks(dm.nBndFaces), TPB>>>(dm.nBndFaces, dm.bndCell.data(), fvoSetMask->data(), aIC.data(), aBC.data());
        svCellKernel<<<nBlocks(nC), TPB>>>(nC, fvoSetMask->data(), aRD.data(), fvoSetVal->data(), src.data());
        cudaCheck(cudaGetLastError(), "fvOptionsSetValues");
    }
        if (wall && eps0)   // eps near-wall setValues constraint (k has none)
    {
        svGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.owner.data(), dm.nei.data(), dm.ownerStart.data(),
                                            dm.losort.data(), dm.losortStart.data(), wall->isWallCell.data(), eps0->data(),
                                            aU.data(), aL.data(), src.data());
        svZeroFaceKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(), wall->isWallCell.data(), aU.data(), aL.data());
        svBndKernel<<<nBlocks(dm.nBndFaces), TPB>>>(dm.nBndFaces, dm.bndCell.data(), wall->isWallCell.data(), aIC.data(), aBC.data());
        svCellKernel<<<nBlocks(nC), TPB>>>(nC, wall->isWallCell.data(), aRD.data(), eps0->data(), src.data());
        if (ami && ami->n) interfaceZeroWallIfCoeff(*ami, wall->isWallCell);   // wall/interface cells: don't perturb the fixed eps
        if (cyc && cyc->n) interfaceZeroWallIfCoeff(*cyc, wall->isWallCell);
        cudaCheck(cudaGetLastError(), "setValues");
    }
    DeviceBuffer<scalar> diagC, B; deviceFold(dm, aRD, src, aIC, aBC, diagC, B);
    // Stage harness: the assembled system for this transported scalar, at its first assembly only.
    if (stageDumpActive() && stageDumpFirstOnly((std::string("xport-") + fieldName).c_str()))
    {
        stageDump(std::string("stage_") + fieldName + "D",   diagC);
        stageDump(std::string("stage_") + fieldName + "Src", B);
        stageDump(std::string("stage_") + fieldName + "Diff", D);
        stageDump(std::string("stage_") + fieldName + "Psi",  field);
    }
    const DeviceLduView sv = (ami && ami->n)
        ? deviceLduViewAmi(dm, diagC, aU, aL, ami->n, ami->ownCell.data(), ami->off.data(), ami->nbrCell.data(), ami->weight.data(), ami->ifCoeff.data())
        : (cyc && cyc->n)
          ? deviceLduViewCyclic(dm, diagC, aU, aL, cyc->n, cyc->ownCell.data(), cyc->nbrCell.data(), cyc->ifCoeff.data())
          : deviceLduView(dm, diagC, aU, aL);
    // Linear solver SELECTED FROM fvSolution like OF (solvers.<field>: solver smoothSolver; smoother symGaussSeidel
    // -> useGS, set by the caller from the dict). cf's multicolor GS does NOT carry the interface (cyclic/AMI)
    // off-diagonal, so interface LDUs fall back to BiCGStab (which folds the interface into Amul), a documented GPU
    // limitation, NOT a heuristic. BRAE_SCALAR_GS overrides for debugging (=0 force off, !=0 force on).
    bool wantGS = useGS;
    if (const char* e = std::getenv("BRAE_SCALAR_GS")) wantGS = (std::atoi(e) != 0);
    const bool gs = wantGS && !(ami && ami->n) && !(cyc && cyc->n);
    // OF lduMatrix::solver::normFactor, NOT sum|b|.
    //
    //     sumA   = row sums of A            (matrix_.sumA(...))
    //     xRef   = average(psi)             (gAverage(psi))
    //     normF  = sum(|A.psi - sumA*xRef| + |b - sumA*xRef|) + small
    //
    // This scales EVERY residual the solver reports and tests, so getting it wrong changes what the
    // absolute `tolerance` means (relTol is a ratio and cancels it, but `tol` does not) and makes brae's
    // "Solving for epsilon" lines incomparable with OF's. brae already had the correct formula in
    // pcg.cu, gamg.cu and parallel_pbicgstab.cuh -- this path alone used sum|b|, which for the epsilon
    // equation is dominated by the production source and the near-wall setValues rows and so runs far
    // larger than OF's, making the normalised residual look converged sooner than it is.
    //
    // sumA comes from A applied to a field of ones, which is the row sum by definition, so the interface
    // off-diagonals are included exactly as deviceAmul accounts for them -- no second traversal to keep
    // in step with the LDU layout.
    // Item 66: the same three reductions and the same operations (the divide by n, the sumA*xRef scale,
    // the two sums, (n1 + n2) + 1e-20) stay on the device in deviceNormFactorInto, and the solvers
    // divide by the result there; the host never reads it. BRAE_NORMFACTOR_HOST=1 keeps the previous
    // host arithmetic below, the identity gate's other arm.
    DeviceBuffer<scalar> dnf;
    scalar normF = scalar(1);
    if (normFactorOnHost())
    {
        normF = [&]{
            const int n = static_cast<int>(field.size());
            if (n == 0) return scalar(1);
            DeviceBuffer<scalar> sumA, Apsi, t, w;
            const DeviceBuffer<scalar>& ones = deviceOnes(n);
            deviceAmul(sv, ones, sumA);                     // sumA = A * 1 = row sums
            deviceAmul(sv, field, Apsi);                    // A.psi
            const scalar xRef = deviceDot(field, ones) / static_cast<scalar>(n);
            deviceCopy(t, sumA);
            deviceScale(t, xRef);                           // t = sumA*xRef
            deviceCopy(w, Apsi); deviceAxpy(-1.0, t, w);    // w = A.psi - t
            scalar nf = deviceSumMag(w);
            deviceCopy(w, B);    deviceAxpy(-1.0, t, w);    // w = b - t
            nf += deviceSumMag(w);                          // sum|..| + sum|..| == sum(|..|+|..|)
            return nf + scalar(1e-20);                      // solverPerformance::small_
        }();
        dnf.resize(1);
        cudaMemcpyAsync(dnf.data(), &normF, sizeof(scalar), cudaMemcpyHostToDevice, cudaStreamPerThread);
    }
    else
    {
        deviceNormFactorInto(sv, field, B, deviceOnes(static_cast<int>(field.size())), dnf);
    }
    DeviceSolverPerf perf;                                        // OF-style report: init/final/nIter for this scalar
    // The PER-CELL residual of this transport equation, r = B - A*psi, before the solve moves anything.
    // Same instrument as the momentum one: a global initial residual says a scalar equation disagrees
    // with OpenFOAM's converged state but never WHERE, and every defect found in this port was found by
    // splitting a residual by region. Named by field so k, epsilon, omega, nuTilda, ReThetat and
    // gammaInt each get their own.
    if (stageDumpActive() && stageDumpFirstOnly((std::string("resid_") + fieldName).c_str()))
    {
        DeviceBuffer<scalar> Apsi, r;
        deviceAmul(sv, field, Apsi);
        deviceCopy(r, B);
        deviceAxpy(-1.0, Apsi, r);
        stageDump(std::string("stage_resid_") + fieldName, r);
        stageDump(std::string("stage_resid_") + fieldName + "_normFactor",
                  std::vector<scalar>(1, normFactorOnHost() ? normF : deviceReadScalar(dnf.data())));
    }
    // The smoothSolver the case asked for, algorithm included: OpenFOAM's own sweep, level-scheduled
    // (device_sym_gauss_seidel.cuh), symmetric or ascending-only as the `smoother` entry names.
    if (gs) deviceSymGaussSeidel(sv, B, field, dnf.data(), tol, relTolKE, 3000, &perf, /*minIter*/0, nSweeps,
                                 gsSymmetric);
    // DILU when the case asks for it, Jacobi otherwise. NOT a cost choice: both reach the requested
    // relTol, but they stop in different places -- on turbulentFlatPlate:kEpsilon OpenFOAM's DILU solve
    // lands at a median 0.0064 of the initial residual against brae's Jacobi 0.0726, and that gap leaves
    // k and epsilon mutually inconsistent enough to diverge. See DeviceSimpleControls::diluKE.
    else
    {
        const DeviceDilu* pc = precon ? precon : turbPrecon();
        perf = deviceJacobiBiCGStab(sv, B, field, dnf.data(), tol, relTolKE, 3000, keCheckEvery, 0,
                                    (pc && pc->valid) ? pc : nullptr);
    }
    turbStore().push_back({fieldName, perf});                    // record for the "Solving for <field>" line
    if (boundPositive) deviceBoundField(dm, field, 1e-15);        // OF bound(field): neg -> local avg, not floor
}

} // namespace brae
