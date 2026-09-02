// CUDA implementation of the compressible standard k-epsilon closure. See kEpsilon.cuh for the
// provenance, the contract and the refusal list.
//
// Transcribed from kEpsilon_cpp.cu IN THE ORDER THAT FILE PRODUCES ITS STAGES, which is the order
// kEpsilon.C produces them. Where a legacy kernel in src/cuda already computes a stage with the same
// arithmetic it is reused; where the legacy kernel groups the arithmetic differently, or encodes a
// substitution this module refuses, a new kernel is written and the reason is recorded above it.
#include "kEpsilon.cuh"
#include "device_fvoptions.cuh"   // deviceSetValues: fvMatrix::setValues, shared with the energy equation
#include "device_pcg.cuh"
#include "device_blas.cuh"      // deviceAxpy / deviceCopy / deviceHadamard
#include "device_simple.cuh"    // deviceRelaxDiag -- fvMatrix::relax, already gated
#include "nut_wall_function.cuh"    // nutkWallFunctionValue / yPlusWall are BRAE_HD -- one definition
#include <stdexcept>
#include <vector>
#include <cmath>

namespace brae {
namespace gpu {
namespace kEpsilonRAS {

namespace {

constexpr int TPB = 256;
inline int nBlk(int n) { return (n + TPB - 1) / TPB; }

// DeviceBuffer::resize takes from DevicePool::take, whose contract is explicit: "Returned memory is NOT
// zeroed." Every buffer this module ACCUMULATES into has to be memset first, and the trap is that a
// single-call test cannot see the omission -- the FIRST call is correct because the pool has no
// same-size block to hand back yet. It is the SECOND correct() that reads another stage's leavings.
void zeroed(DeviceBuffer<scalar>& b, int n)
{
    b.resize(n);
    cudaMemsetAsync(b.data(), 0, std::size_t(n) * sizeof(scalar), cudaStreamPerThread);
}

void zeroedL(DeviceBuffer<label>& b, int n)
{
    b.resize(n);
    cudaMemsetAsync(b.data(), 0, std::size_t(n) * sizeof(label), cudaStreamPerThread);
}


// OpenFOAM's wall overwrite, epsilonWallFunctionFvPatchScalarField: G[celli] = G0[celli] and
// epsilon[celli] = epsilon0[celli] on every cell with at least one wall-function face.
//
// The legacy overrideKernel in device_scalar_transport.cuh also applies the cyclicACMI `wallW` blend,
// which this module refuses along with every other coupled patch, and it lives in an anonymous namespace
// inside a header, so it cannot be linked to from another translation unit in any case.
__global__ void wallOverrideKernel(
    int           nC,
    const label*  isWallCell,
    const scalar* G0,
    const scalar* eps0,
    scalar*       G,
    scalar*       eps)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    if (!isWallCell[c]) return;
    G[c]   = G0[c];
    eps[c] = eps0[c];
}


// DkEff()/DepsilonEff() on CELLS: (nut/sigma + nu)*rho. The legacy depsKernel is `nut/sigma + nu` with a
// SCALAR nu and no rho -- the compressible lineage has neither, which is why the reference is handed
// /*nu=*/0.0 and reads a field instead.
//
// DEff is written out WITHOUT the rho, because that is OpenFOAM's own DepsilonEff() and it is what the
// gate compares against; D carries the rho the compressible form multiplies in.
__global__ void effDiffCellKernel(
    int           nC,
    const scalar* nut,
    const scalar* nu,
    const scalar* rho,
    scalar        sigma,
    scalar*       DEff,
    scalar*       D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar d = nut[c] / sigma + nu[c];
    DEff[c] = d;
    D[c]    = d * rho[c];
}


// ...and on BOUNDARY faces, from nut's OWN boundary value rather than the owner cell's. DkEff is a
// volScalarField, so fvm::laplacian takes its patch value; at an inlet OF's nut_b is Cmu*k_b^2/eps_b,
// which is nowhere near the adjacent cell's.
__global__ void effDiffBndKernel(
    int           n,
    const scalar* nutB,
    const scalar* nuB,
    const scalar* rhoB,
    scalar        sigma,
    scalar*       DB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    DB[i] = (nutB[i] / sigma + nuB[i]) * rhoB[i];
}


// The epsilon equation's reactions, grouped EXACTLY as kEpsilon_cpp.cu writes them -- the rho inside the
// fmax/fmin argument, not factored out of it. The legacy deviceEpsReaction computes the same terms with
// rho factored outside, which is mathematically identical and one ulp apart; that difference is enough
// to turn a 1e-16 stage comparison into a 1e-14 one and stop the gate pinning the arithmetic.
__global__ void epsReactionKernel(
    int           nC,
    const scalar* V,
    const scalar* rho,
    const scalar* gByNu,
    const scalar* k,
    const scalar* eps,
    const scalar* divU,
    const scalar* divPhi,
    scalar        C1,
    scalar        C2,
    scalar        C3,
    scalar        Cmu,
    int           bounded,
    scalar*       diag,
    scalar*       source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar v = V[c];
    const scalar r = rho[c];

    // == C1*alpha*rho*GbyNu*Cmu*k. NOT C1*G/k: the two agree only where nut == Cmu k^2/eps, which is
    // exactly what a wall cell has just had overwritten.
    source[c] += C1 * r * gByNu[c] * Cmu * k[c] * v;

    // - fvm::SuSp(((2/3)*C1 - C3)*alpha*rho*divU, epsilon)
    const scalar sp = ((scalar(2.0) / scalar(3.0)) * C1 - C3) * r * divU[c];
    diag[c]   += v * fmax(sp, scalar(0.0));
    source[c] -= v * fmin(sp, scalar(0.0)) * eps[c];

    // - fvm::Sp(C2*alpha*rho*epsilon/k, epsilon)
    diag[c] += C2 * r * eps[c] / k[c] * v;

    // `bounded`: - fvm::Sp(fvc::div(phi), epsilon), against the EQUATION's mass flux. It vanishes where
    // phi is conservative, so it cannot move a converged state -- which is why it needs its own
    // measurement rather than being assumed harmless.
    if (bounded) diag[c] -= divPhi[c] * v;
}


// The k equation's reactions. Same grouping argument as above.
__global__ void kReactionKernel(
    int           nC,
    const scalar* V,
    const scalar* rho,
    const scalar* G,
    const scalar* k,
    const scalar* eps,
    const scalar* divU,
    const scalar* divPhi,
    int           bounded,
    scalar*       diag,
    scalar*       source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar v = V[c];
    const scalar r = rho[c];

    // == alpha*rho*G, with G already carrying the wall override at wall cells.
    source[c] += r * G[c] * v;

    // - fvm::SuSp((2/3)*alpha*rho*divU, k)
    const scalar sp = (scalar(2.0) / scalar(3.0)) * r * divU[c];
    diag[c]   += v * fmax(sp, scalar(0.0));
    source[c] -= v * fmin(sp, scalar(0.0)) * k[c];

    // - fvm::Sp(alpha*rho*epsilon/k, k). The rho is NOT optional, and its absence is invisible in the
    // incompressible lineage where it is 1: at rho ~ 0.38 leaving it out made k's destruction 2.6x too
    // strong across the whole field while epsilon, solved first and correctly weighted, looked fine.
    diag[c] += r * eps[c] / k[c] * v;

    if (bounded) diag[c] -= divPhi[c] * v;
}


// fvMatrix::setValues, in four launches. The decomposition is forced by the read-then-write hazard: the
// gather has to read the ORIGINAL upper/lower, and one kernel cannot order "everyone reads" before
// "everyone zeroes".
//
// WHY GATHERING EVERYTHING BEFORE ZEROING ANYTHING IS STILL THE HOST'S ANSWER. The host zeroes each face
// inside its per-cell loop, so on a face shared by two constrained cells only the first cell processed
// transfers. This version transfers both. The difference lands ONLY in the source of a constrained cell,
// and the fourth kernel overwrites exactly those with value*diag -- so the two agree everywhere the
// result survives. Faces between a constrained and a free cell transfer once either way.
// The four setValues kernels moved to device_fvoptions.cu: the energy equation applies the same
// constraint, and one matrix manipulation this delicate should have one implementation.


// Foam::bound, area-weighted. OF's fvc::average is surfaceSum(magSf*ssf)/surfaceSum(magSf)
// (fvcAverage.C), and the host reference is the same with the PATCH value on a boundary face.
//
// deviceBoundField is NOT reused: its gather takes an unweighted face-count mean and uses the CELL value
// on boundary faces. That difference lands in every bounded cell, and bound() fires exactly where the
// solve went negative -- which is where an error compounds rather than washes out.
__global__ void boundGatherKernel(
    int           nIf,
    const label*  own,
    const label*  nei,
    const scalar* w,
    const scalar* magSf,
    const scalar* x,
    scalar        floorV,
    scalar*       num,
    scalar*       den)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar co = fmax(x[own[f]], floorV);
    const scalar cn = fmax(x[nei[f]], floorV);
    const scalar vf = w[f] * co + (scalar(1.0) - w[f]) * cn;
    const scalar a  = magSf[f];
    atomicAdd(&num[own[f]], a * vf);
    atomicAdd(&den[own[f]], a);
    atomicAdd(&num[nei[f]], a * vf);
    atomicAdd(&den[nei[f]], a);
}


__global__ void boundGatherBndKernel(
    int           nB,
    const label*  bndCell,
    const scalar* magSf,
    const scalar* bval,
    scalar        floorV,
    scalar*       num,
    scalar*       den)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    const label  c = bndCell[i];
    const scalar a = magSf[i];
    atomicAdd(&num[c], a * fmax(bval[i], floorV));
    atomicAdd(&den[c], a);
}


__global__ void boundApplyKernel(
    int           nC,
    const scalar* num,
    const scalar* den,
    scalar        floorV,
    scalar*       x)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar avg = (den[c] > scalar(0.0)) ? num[c] / den[c] : floorV;
    // pos0(-vsf) selects the neighbour average ONLY where the solve went non-positive; elsewhere the
    // floor alone applies. A hard clamp everywhere is a different operator and it is not what OF does.
    const scalar cand = (x[c] <= scalar(0.0)) ? avg : scalar(0.0);
    x[c] = fmax(fmax(x[c], cand), floorV);
}


__global__ void boundBndKernel(int nB, scalar floorV, scalar* bval)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    bval[i] = fmax(bval[i], floorV);
}


// correctNut's BOUNDARY half. A turbulence-wall-function face takes nutkWallFunction; every other face
// takes Cmu*k_b^2/eps_b -- NOT the owner cell's nut. deviceBoundaryNut is close but keys its wall
// predicate on the CELL rather than the face, and falls back to the cell nut, which is exactly what a
// `.*`-matched `calculated` nut patch got wrong: OF had 0.6 to 2.5 there and brae held the initial 0.
__global__ void nutBoundaryKernel(
    int           nB,
    const label*  wfMask,
    const label*  bndCell,
    const scalar* y,
    const scalar* kCell,
    const scalar* kBnd,
    const scalar* epsBnd,
    const scalar* nuFace,
    scalar        Cmu,
    scalar        Cmu25,
    scalar        kappa,
    scalar        E,
    scalar        yplLam,
    scalar*       nutBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    if (wfMask[i])
    {
        // The wall function reads the NEAR-WALL CELL's k, and nu AT THE FACE.
        const scalar kc = kCell[bndCell[i]];
        const scalar yp = yPlusWall(Cmu25, y[i], kc, nuFace[i]);
        nutBnd[i] = nutkWallFunctionValue(yp, nuFace[i], yplLam, kappa, E);
        return;
    }
    nutBnd[i] = Cmu * kBnd[i] * kBnd[i] / epsBnd[i];
}


// EddyDiffusivity::correctNut -- alphat = rho*nut/Prt, unconditional and whole-field. A deviceHadamard
// plus a scale helper would need a scratch buffer and a scale entry point that does not exist.
__global__ void alphatKernel(
    int           nC,
    const scalar* rho,
    const scalar* nut,
    scalar        Prt,
    scalar*       alphat)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    alphat[c] = rho[c] * nut[c] / Prt;
}


// The wall-cell mask, from the per-wall-face cell list DeviceWallData already carries. isWallCell in
// that struct answers "is this cell's epsilon fixed by a wall function", which is the same question --
// it is copied rather than recomputed so the two cannot drift.
__global__ void copyMaskKernel(int nC, const label* src, label* dst)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    dst[c] = src[c] ? 1 : 0;
}


// The refusals, in one place so the module's scope is stated once.
void refuseUnported(const KEpsilonInput& in)
{
    if (in.co.realizable || in.co.rng)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): the case selects realizableKE or RNGkEpsilon, which this module does not "
            "implement. Refusing rather than running the standard model: by the time the coefficients "
            "reach here turbulence_setup has already substituted that model's constants, so the result "
            "would be standard-kEpsilon arithmetic on another model's coefficients -- a closure that "
            "exists in no source and that a brae-vs-brae gate could not detect, because both sides "
            "would run it.");
    }
    if (in.hasCoupledPatches)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): the mesh has cyclic/AMI/processor patches. buildDeviceMesh keeps those "
            "faces out of the LDU, so they would contribute nothing to the convection or the diffusion "
            "of k and epsilon -- silently. correctNut's boundary assignment is also wrong on a coupled "
            "patch, where OpenFOAM's correctBoundaryConditions() overwrites it with the interpolated "
            "value and Cmu*k_b^2/eps_b != interpolate(Cmu*k^2/eps).");
    }
    if (in.hasUnportedFvOption)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): the case declares an fvOption this path does not implement"
            + (in.fvOptionUnsupported.empty() ? std::string()
                                              : std::string(" (") + in.fvOptionUnsupported + ")")
            + ". kEpsilon.C applies fvOptions to both the k and the epsilon equation.");
    }
    if (in.hasNonUpwindDivScheme)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): div(phi,k)/div(phi,epsilon) asks for a scheme this module does not "
            "assemble"
            + (in.divSchemeUnsupported.empty() ? std::string()
                                               : std::string(" (") + in.divSchemeUnsupported + ")")
            + ". Only Gauss upwind, with or without `bounded`, is ported -- which is what the host "
              "reference assembles. Running upwind where the case said otherwise is the substitution "
              "this project keeps finding.");
    }
    if (in.hasNonWallTurbWallFunc)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): a turbulence wall function sits on a patch that is not of type `wall`. "
            "nearWallDist only fills y on `wall` patches, so the wall treatment would divide by an "
            "unset distance.");
    }
    if (in.boundedK != in.boundedEps)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): the case bounds one of div(phi,k) / div(phi,epsilon) and not the other. "
            "The host reference carries ONE `bounded` flag for both, so honouring the split here would "
            "compare against a reference that cannot express it.");
    }
    if (!in.phiInt || !in.phiBnd || !in.phiByRhoInt || !in.phiByRhoBnd)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): both fluxes are required, on internal AND boundary faces. The MASS flux "
            "convects; the VOLUMETRIC flux is the dilatation divU. They differ by rho and using one for "
            "the other is invisible in the incompressible lineage.");
    }
    if (!in.rhoCell || !in.rhoBndFace || !in.nuCell || !in.nuBndFace || !in.nuWallFace)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): rho and nu are required on cells, boundary faces AND wall faces. There is "
            "no case-constant nu on the compressible path -- the reference is handed 0.0 for exactly "
            "that reason -- and the wall functions are written in terms of nu at the WALL FACE.");
    }
    if (!in.nutBndFace)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): nut's boundary field is required. DkEff()/DepsilonEff() are volScalarFields "
            "and fvm::laplacian takes their PATCH values, which are not the owner cells'.");
    }
    if (!in.Ux || !in.Uy || !in.Uz)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): the velocity is required -- the production term is built from fvc::grad(U).");
    }
    if (!in.wfBndMask || !in.wallYBndFace)
    {
        throw std::runtime_error(
            "kEpsilon(cuda): the wall-function FACE mask and the near-wall distance per boundary face are "
            "required. correctNut rewrites nut on EVERY boundary face -- `nut_ = Cmu*sqr(k_)/epsilon_` is "
            "a GeometricField assignment -- and without the mask this module could not tell a wall face "
            "from an inlet one, so it would leave nut's boundary at the previous iteration's values with "
            "nothing to report it.");
    }
}

} // namespace


void production(
    KEpsilonStages&             st,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& nut,
    const KEpsilonInput&        in)
{
    refuseUnported(in);
    const int nC = dm.nCells;

    // fvc::grad(U), then GbyNu = gradU && devTwoSymm(gradU). Both reused: the legacy kernels compute
    // exactly these and are already gated.
    deviceGradU(dm, dbU, *in.Ux, *in.Uy, *in.Uz, st.gradU);
    deviceGByNuFromGradU(st.gradU, nC, st.gByNu);

    // divU is the DILATATION and comes from the VOLUMETRIC flux; divPhi is the EQUATION's own mass-flux
    // divergence and is only read by `bounded`. In the incompressible lineage these are one field.
    deviceDiv(dm, *in.phiByRhoInt, *in.phiByRhoBnd, st.divU);
    deviceDiv(dm, *in.phiInt, *in.phiBnd, st.divPhi);

    // G = nut*GbyNu, captured BEFORE the wall replacement -- which is where OpenFOAM writes it too.
    deviceHadamard(st.G, nut, st.gByNu);
}


void wallTreatment(
    KEpsilonStages&             st,
    DeviceBuffer<scalar>&       epsilon,
    const DeviceMesh&           dm,
    const DeviceWallData&       wall,
    const DeviceBuffer<scalar>& k,
    const KEpsilonInput&        in)
{
    const int nC = dm.nCells;

    // epsilonWallFunction's near-wall values. Reused: deviceWallEpsG0 already carries the STEPWISE
    // log/viscous branch and the lowRe production guard -- and the guard is on the G0 ACCUMULATION, not
    // on the replacement, so a wall-adjacent cell whose only wall face is viscous-resolved has its
    // volume G overwritten with 0 rather than left alone. That asymmetry is OpenFOAM's.
    zeroed(st.eps0, nC);
    zeroed(st.G0, nC);
    deviceWallEpsG0(wall, k, *in.Ux, *in.Uy, *in.Uz, /*nu=*/scalar(0.0), st.eps0, st.G0, in.co,
                    /*nutWall=*/0, /*atmZ0=*/scalar(0.0), /*atmBoundNut=*/true, in.nuWallFace);

    zeroedL(st.isWallCell, nC);
    copyMaskKernel<<<nBlk(nC), TPB>>>(nC, wall.isWallCell.data(), st.isWallCell.data());
    cudaCheck(cudaGetLastError(), "kEpsilon wall mask");

    wallOverrideKernel<<<nBlk(nC), TPB>>>(nC, st.isWallCell.data(), st.G0.data(), st.eps0.data(),
                                          st.G.data(), epsilon.data());
    cudaCheck(cudaGetLastError(), "kEpsilon wall override");
    st.wallCells = wall.nWC;
}


namespace {

// The diffusion + convection half, shared by the two equations because they differ only in sigma and in
// which field they transport. Writing it once is what keeps the two from drifting.
void assembleTransport(
    PressureMatrix&             M,
    DeviceBuffer<scalar>&       DEffOut,
    DeviceBuffer<scalar>&       gammaFace,
    DeviceBuffer<scalar>&       gammaBnd,
    const DeviceMesh&           dm,
    const DeviceBoundary&       db,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& nut,
    scalar                      sigma,
    const KEpsilonInput&        in)
{
    const int nC = dm.nCells;
    const int nB = db.n;

    // DEff on cells, then interpolated to internal faces; the BOUNDARY coefficient is built from nut's
    // own patch value instead.
    DeviceBuffer<scalar> Dcell;
    DEffOut.resize(nC);
    Dcell.resize(nC);
    effDiffCellKernel<<<nBlk(nC), TPB>>>(nC, nut.data(), in.nuCell->data(), in.rhoCell->data(), sigma,
                                         DEffOut.data(), Dcell.data());
    cudaCheck(cudaGetLastError(), "kEpsilon DEff cells");
    deviceInterpolate(dm, Dcell, gammaFace);

    gammaBnd.resize(nB);
    effDiffBndKernel<<<nBlk(nB), TPB>>>(nB, in.nutBndFace->data(), in.nuBndFace->data(),
                                        in.rhoBndFace->data(), sigma, gammaBnd.data());
    cudaCheck(cudaGetLastError(), "kEpsilon DEff boundary");

    // fvm::div(phi, field), upwind. The boundary half carries the flux-conditional switch the caller has
    // already applied to db.
    deviceDivUpwindCoeffs(dm, *in.phiInt, M.diag, M.upper, M.lower);
    zeroed(M.source, nC);
    deviceBCDivCoeffs(db, *in.phiBnd, M.iC, M.bC);

    // - fvm::laplacian(gamma, field). `corrected` is TWO changes and this module makes both: the
    // implicit coefficient takes nonOrthDeltaCoeffs, and the non-orthogonal part enters as an explicit
    // source. Implementing only the implicit half moves the SOURCE while leaving the DIAGONAL exact,
    // which no gate comparing D() can see.
    {
        DeviceBuffer<scalar> lDiag, lUp, lLo, lIC, lBC;
        deviceLaplacianCoeffs(dm, gammaFace, lDiag, lUp, lLo, in.correctedLaplacian);
        deviceBCLaplacianCoeffsFace(db, gammaBnd, lIC, lBC);
        deviceAxpy(-1.0, lDiag, M.diag);
        deviceAxpy(-1.0, lUp, M.upper);
        deviceAxpy(-1.0, lLo, M.lower);
        deviceAxpy(-1.0, lIC, M.iC);
        deviceAxpy(-1.0, lBC, M.bC);

        if (in.correctedLaplacian)
        {
            DeviceBuffer<scalar> bval, gx, gy, gz, ffc, corr;
            deviceBCValue(db, field, bval);
            deviceGaussGrad(dm, field, bval, gx, gy, gz);
            if (in.snGradLimitCoeff > scalar(0.0))
            {
                deviceLaplacianCorrFluxLimited(dm, gammaFace, field, gx, gy, gz, in.snGradLimitCoeff, ffc);
                deviceFaceDivSource(dm, ffc, corr);
            }
            else
            {
                deviceLaplacianCorr(dm, gammaFace, gx, gy, gz, corr);
            }
            // deviceLaplacianCorr returns -V*div(faceFluxCorr) -- already negated -- and the laplacian
            // itself enters this equation with -1, so its explicit source does too. The two signs
            // compose to the reference's `L.source -= corr` followed by `M -= L`.
            deviceAxpy(-1.0, corr, M.source);
        }
    }
}

} // namespace


void assembleEpsEqn(
    PressureMatrix&             E,
    KEpsilonStages&             st,
    const DeviceMesh&           dm,
    DeviceBoundary&             dbEps,
    const DeviceBoundary&       dbK,
    const DeviceBuffer<scalar>& epsilon,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& nut,
    const KEpsilonInput&        in)
{
    const int nC = dm.nCells;

    // epsilon_.boundaryFieldRef().updateCoeffs(). turbulentMixingLengthDissipationRateInlet recomputes
    // its refValue from k's CURRENT patch values here, and the flux switch resolves inletOutlet -- both
    // BEFORE the matrix is built, because fvMatrix's constructor is where OpenFOAM does it.
    // ORDER: the turbulent inlet recomputes refValue FIRST, then the flux switch resolves which faces
    // are fixedValue at all. Reversed, the switch would act on the previous iteration's refValue.
    if (in.turbInletEpsMask && in.turbInletEpsLen)
    {
        deviceUpdateTurbulentInletSecond(dbK, *in.turbInletEpsMask, *in.turbInletEpsLen,
                                         in.co.Cmu, dbEps);
    }
    deviceUpdateInletOutlet(dbEps, *in.phiBnd);

    assembleTransport(E, st.DepsilonEff, st.gammaEpsFace, st.gammaEpsBnd, dm, dbEps, epsilon, nut,
                      in.co.sigmaEps, in);

    epsReactionKernel<<<nBlk(nC), TPB>>>(nC, dm.V.data(), in.rhoCell->data(), st.gByNu.data(), k.data(),
                                         epsilon.data(), st.divU.data(), st.divPhi.data(),
                                         in.co.C1, in.co.C2, in.co.C3, in.co.Cmu,
                                         in.boundedEps ? 1 : 0, E.diag.data(), E.source.data());
    cudaCheck(cudaGetLastError(), "kEpsilon eps reaction");
}


void assembleKEqn(
    PressureMatrix&             K,
    KEpsilonStages&             st,
    const DeviceMesh&           dm,
    DeviceBoundary&             dbK,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& epsilon,
    const DeviceBuffer<scalar>& nut,
    const KEpsilonInput&        in)
{
    const int nC = dm.nCells;

    // k_'s own boundary refresh: turbulentIntensityKineticEnergyInlet reads U's CURRENT patch values.
    if (in.turbInletKMask && in.turbInletKInt)
    {
        deviceUpdateTurbulentInletK(dbU, *in.turbInletKMask, *in.turbInletKInt, dbK);
    }
    deviceUpdateInletOutlet(dbK, *in.phiBnd);

    assembleTransport(K, st.DkEff, st.gammaKFace, st.gammaKBnd, dm, dbK, k, nut, in.co.sigmaK, in);

    kReactionKernel<<<nBlk(nC), TPB>>>(nC, dm.V.data(), in.rhoCell->data(), st.G.data(), k.data(),
                                       epsilon.data(), st.divU.data(), st.divPhi.data(),
                                       in.boundedK ? 1 : 0, K.diag.data(), K.source.data());
    cudaCheck(cudaGetLastError(), "kEpsilon k reaction");
}


void boundField(
    DeviceBuffer<scalar>& x,
    const DeviceMesh&     dm,
    const DeviceBoundary& db,
    scalar                floorV)
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> num, den;
    zeroed(num, nC);
    zeroed(den, nC);
    boundGatherKernel<<<nBlk(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(),
                                                        dm.w.data(), dm.magSf.data(), x.data(), floorV,
                                                        num.data(), den.data());
    cudaCheck(cudaGetLastError(), "kEpsilon bound gather");

    // The BOUNDARY faces are part of the same average, and they contribute the PATCH value rather than
    // the cell's. Omitting them is not a small error: on a wall-adjacent cell the wall face is a large
    // part of the total area, and bound() fires exactly there.
    if (db.n)
    {
        DeviceBuffer<scalar> bval;
        deviceBCValue(db, x, bval);
        boundGatherBndKernel<<<nBlk(db.n), TPB>>>(db.n, dm.bndCell.data(), db.magSf.data(), bval.data(),
                                                  floorV, num.data(), den.data());
        cudaCheck(cudaGetLastError(), "kEpsilon bound gather boundary");
    }

    boundApplyKernel<<<nBlk(nC), TPB>>>(nC, num.data(), den.data(), floorV, x.data());
    cudaCheck(cudaGetLastError(), "kEpsilon bound apply");
}


// alphat_.correctBoundaryConditions() for a compressible::alphatWallFunction face:
// operator==(rhow*tnutw/Prt_) (alphatWallFunctionFvPatchScalarField.C:125), with the PATCH's own Prt_.
// Only masked faces are written; every other face keeps whatever its own condition left there.
__global__ void alphatBndKernel(
    int           nB,
    const label*  __restrict__ mask,
    const scalar* __restrict__ rhoB,
    const scalar* __restrict__ nutB,
    const scalar* __restrict__ prt,
    scalar*       __restrict__ alphatB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB || !mask[i]) return;
    const scalar p = prt[i] > scalar(0) ? prt[i] : scalar(0.85);
    alphatB[i] = rhoB[i] * nutB[i] / p;
}


void correctNut(
    DeviceBuffer<scalar>&       nut,
    DeviceBuffer<scalar>&       nutBnd,
    DeviceBuffer<scalar>*       alphat,
    DeviceBuffer<scalar>*       alphatBnd,
    const DeviceMesh&           dm,
    const DeviceBoundary&       dbK,
    const DeviceBoundary&       dbEps,
    const DeviceWallData&       wall,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& epsilon,
    const KEpsilonInput&        in)
{
    const int nC = dm.nCells;

    // nut_ = Cmu*sqr(k_)/epsilon_ on cells.
    deviceNut(k, epsilon, nut, in.co);

    // ...and on the boundary, which OpenFOAM's GeometricField assignment also writes. Reached through
    // the caller's wall-function FACE mask, because the predicate is per face and not per cell.
    {
        const int nB = dbEps.n;
        DeviceBuffer<scalar> kB, eB;
        deviceBCValue(dbK, k, kB);
        deviceBCValue(dbEps, epsilon, eB);
        nutBnd.resize(nB);
        // The WALL FUNCTION's Cmu (nutkWallFunction's y+), not the model's -- the model's Cmu below is for
        // the calculated patches' Cmu*k^2/epsilon, which IS the model's. See KEpsilonCoeffs::CmuWall.
        const scalar Cmu25  = std::pow(in.co.CmuWall, scalar(0.25));
        const scalar yplLam = brae::yPlusLam(in.co.kappa, in.co.E);
        nutBoundaryKernel<<<nBlk(nB), TPB>>>(nB, in.wfBndMask->data(), dm.bndCell.data(),
                                             in.wallYBndFace ? in.wallYBndFace->data() : nullptr,
                                             k.data(), kB.data(), eB.data(), in.nuBndFace->data(),
                                             in.co.Cmu, Cmu25, in.co.kappa, in.co.E, yplLam,
                                             nutBnd.data());
        cudaCheck(cudaGetLastError(), "kEpsilon nut boundary");
    }

    // EddyDiffusivity::correctNut. Unconditional and whole-field -- NOT inside the wall-patch loop,
    // which is where the host reference used to have it, so that a case with no wall function left the
    // energy equation on a stale alphat with nothing to catch it.
    if (alphat)
    {
        alphat->resize(nC);
        alphatKernel<<<nBlk(nC), TPB>>>(nC, in.rhoCell->data(), nut.data(), in.Prt, alphat->data());
        cudaCheck(cudaGetLastError(), "kEpsilon alphat");
    }

    // ...and the BOUNDARY, which OF writes in the same call (EddyDiffusivity.C:38). Left out until now,
    // so a device-resident alphaEff read whatever 0/alphat shipped -- `value uniform 0` at the walls on
    // both compressible fixtures -- for the whole run, and the wall lost its entire turbulent
    // diffusivity. The HOST path was fixed first (rhoSimpleFoam_cpp.cu, measured 1.0 -> 2.25e-04 against
    // OpenFOAM's own written field); this is its device counterpart.
    if (alphatBnd && in.alphatWallMask && in.alphatPrtFace && in.rhoBndFace)
    {
        const int nB = dbEps.n;
        alphatBnd->resize(nB);
        alphatBndKernel<<<nBlk(nB), TPB>>>(nB, in.alphatWallMask->data(), in.rhoBndFace->data(),
                                           nutBnd.data(), in.alphatPrtFace->data(), alphatBnd->data());
        cudaCheck(cudaGetLastError(), "kEpsilon alphat boundary");
    }
    (void)wall;
}


namespace {

// relax() -> fvOptions.constrain() -> setValues(wall), in OpenFOAM's order (kEpsilon.C:265-267). The k
// equation passes a null wall mask, because kEpsilon.C:286-288 has no boundaryManipulate for it --
// kqRWallFunction is zeroGradient, and constraining k in wall cells the way epsilon is constrained is a
// different equation.
void finishAndSolve(
    PressureMatrix&             M,
    DeviceBuffer<scalar>&       field,
    const DeviceMesh&           dm,
    bool                        relaxEquation,
    scalar                      alpha,
    const DeviceBuffer<label>*  fvoMask,
    const DeviceBuffer<scalar>* fvoVal,
    const DeviceBuffer<label>*  wallMask,
    const DeviceBuffer<scalar>* wallVal,
    const KEpsilonInput&        in,
    scalar&                     residualOut)
{
    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nB  = dm.nBndFaces;

    // The guard is "the case NAMES a factor", not "the factor is below 1": fvMatrix::relax early-returns
    // only on alpha <= 0, so relax(1.0) still applies the dominance clamp and adds (D - D0)*psi.
    if (relaxEquation && alpha > scalar(0.0))
    {
        DeviceBuffer<scalar> relaxedDiag, delta, t;
        deviceRelaxDiag(M.view(dm), dm, M.iC, alpha, relaxedDiag, delta);
        deviceCopy(M.diag, relaxedDiag);
        deviceHadamard(t, delta, field);
        deviceAxpy(1.0, t, M.source);
    }

    // Both constraints go through the SAME four kernels, because in OpenFOAM they are the same call --
    // FixedValueConstraint::constrain and epsilonWallFunction::manipulateMatrix both end in
    // fvMatrix::setValues. Order matters and is OpenFOAM's: whichever runs first is the one whose value
    // reaches the neighbours, since setValues zeroes the coefficient it just transferred through.
    auto applySetValues = [&](const DeviceBuffer<label>* mask, const DeviceBuffer<scalar>* val)
    {
        if (!mask || !val) return;
        deviceSetValues(dm, *mask, *val, M.diag, M.upper, M.lower, M.source, M.iC, M.bC, field);
    };
    applySetValues(fvoMask, fvoVal);
    applySetValues(wallMask, wallVal);

    // Fold the boundary coefficients in exactly as fvMatrix::solve does, then solve. BiCGStab, not PCG:
    // upwind convection makes upper != lower, so the matrix is asymmetric and a symmetric solver would
    // be solving a different system.
    DeviceBuffer<scalar> diagC, b, ones;
    deviceFold(dm, M.diag, M.source, M.iC, M.bC, diagC, b);

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

    ones.copyFrom(std::vector<scalar>(nC, scalar(1.0)));
    const scalar normF = deviceNormFactor(A, field, b, ones);
    const DeviceSolverPerf perf =
        deviceJacobiBiCGStab(A, b, field, normF, in.tol, in.relTol, in.maxIter, /*checkEvery=*/1, in.minIter);
    residualOut = perf.initialResidual;
}

} // namespace


void correct(
    DeviceBuffer<scalar>&       k,
    DeviceBuffer<scalar>&       epsilon,
    DeviceBuffer<scalar>&       nut,
    DeviceBuffer<scalar>&       nutBnd,
    DeviceBuffer<scalar>*       alphat,
    DeviceBuffer<scalar>*       alphatBnd,
    KEpsilonStages&             st,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    DeviceBoundary&             dbK,
    DeviceBoundary&             dbEps,
    const DeviceWallData&       wall,
    const KEpsilonInput&        in)
{
    production(st, dm, dbU, nut, in);
    wallTreatment(st, epsilon, dm, wall, k, in);

    // ---- the epsilon equation ----------------------------------------------------------------
    // Solved FIRST, and the k equation below then reads the epsilon this solve produced. That lag is
    // OpenFOAM's and reversing it is a different algorithm that still converges to something plausible.
    {
        PressureMatrix E;
        assembleEpsEqn(E, st, dm, dbEps, dbK, epsilon, k, nut, in);

        // The wall constraint's VALUE IS THE CURRENT FIELD, not the eps0 array -- OpenFOAM's
        // epsilonWallFunction::manipulateMatrix is
        // `matrix.setValues(patch().faceCells(), patchInternalField())`
        // (epsilonWallFunctionFvPatchScalarField.C:616), and patchInternalField() is epsilon as it
        // stands when the wall pass runs. That matters exactly where an fvOptions constraint has just
        // pinned the same cell: fvMatrix::setValues writes psi as well as the matrix, so OpenFOAM's
        // wall pass READS BACK the constrained value and re-pins the cell to it. Passing the frozen
        // eps0 instead overwrote the constraint -- measured on OpenFOAM's own angledDuct tutorial,
        // whose porous zone pins epsilon to 150 on 8000 cells: OpenFOAM ends with 8000 cells at 150
        // and brae with 6480, the 1520 wall-adjacent ones sitting at the wall value 320.618 instead.
        //
        // wallTreatment has already written eps0 into epsilon for every wall cell, so on a case with no
        // constraint this reads exactly what it read before.
        finishAndSolve(E, epsilon, dm, in.relaxEquationEps, in.relaxEps,
                       in.fvoEpsMask, in.fvoEpsVal,
                       &st.isWallCell, &epsilon, in, st.epsResidual);

        boundField(epsilon, dm, dbEps, scalar(1e-15));
    }

    // ---- the k equation ----------------------------------------------------------------------
    {
        PressureMatrix K;
        assembleKEqn(K, st, dm, dbK, dbU, k, epsilon, nut, in);

        // No wall mask: see finishAndSolve.
        finishAndSolve(K, k, dm, in.relaxEquationK, in.relaxK,
                       in.fvoKMask, in.fvoKVal,
                       nullptr, nullptr, in, st.kResidual);

        boundField(k, dm, dbK, scalar(1e-15));
    }

    correctNut(nut, nutBnd, alphat, alphatBnd, dm, dbK, dbEps, wall, k, epsilon, in);
}

} // namespace kEpsilonRAS
} // namespace gpu
} // namespace brae
