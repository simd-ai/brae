// The OF-mirror kOmegaSST on the device. See kOmegaSST.cuh for why this exists alongside the legacy
// deviceKOmegaSSTCorrect, and kOmegaSST_cpp.cu for the host twin this must agree with.
#include "kOmegaSST.cuh"
#include "sst_stage_dump.cuh"
#include "turbulence_transport.cuh"   // assembleScalarTransport / solveScalarEqn -- shared with kEpsilon
#include "kEpsilon.cuh"               // boundField: Foam::bound, the mirror's area-weighted form
#include "device_blas.cuh"
#include "device_fvoptions.cuh"
#include "nut_wall_function.cuh"
#include <fstream>
#include <filesystem>
#include <system_error>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <cstdlib>

namespace brae {
namespace gpu {
namespace kOmegaSSTRAS {

namespace {

int nBlk(int n) { return (n + 255) / 256; }
constexpr int TPB = 256;

void zeroed(DeviceBuffer<scalar>& b, int n)
{
    b.resize(static_cast<std::size_t>(n));
    cudaCheck(cudaMemsetAsync(b.data(), 0, static_cast<std::size_t>(n) * sizeof(scalar), cudaStreamPerThread),
              "kOmegaSST zero");
}

// Every refusal the host twin carries, so an arm that cannot run a case says so instead of running a
// different one. Same set and same wording shape as the kEpsilon closure's.
void refuseUnsupported(const KOmegaSSTInput& in)
{
    if (in.hasCoupledPatches)
        throw std::runtime_error(
            "kOmegaSST(cuda): the mesh has cyclic/AMI/processor patches. buildDeviceMesh keeps those "
            "faces out of the LDU, so they would contribute nothing to the convection or the diffusion "
            "of k and omega -- silently.");
    if (in.hasUnportedFvOption)
        throw std::runtime_error(
            "kOmegaSST(cuda): the case declares an fvOption this path does not implement"
            + (in.fvOptionUnsupported.empty() ? std::string()
                                              : std::string(" (") + in.fvOptionUnsupported + ")")
            + ". kOmegaSSTBase.C applies fvOptions to both the k and the omega equation.");
    if (in.hasNonUpwindDivScheme)
        throw std::runtime_error(
            "kOmegaSST(cuda): div(phi,k)/div(phi,omega) asks for a scheme this module does not assemble"
            + (in.divSchemeUnsupported.empty() ? std::string()
                                               : std::string(" (") + in.divSchemeUnsupported + ")")
            + ". Gauss upwind and Gauss limitedLinear <k> are ported, with or without `bounded`.");
    if (in.co.F3)
        throw std::runtime_error(
            "kOmegaSST(cuda): the F3 near-wall switch is set. kOmegaSSTBase multiplies F23 by F3 "
            "(kOmegaSSTBase.C:F23), which changes both the eddy-viscosity limiter and the production "
            "limiter. Not implemented; refusing rather than silently running with F3 off.");
    if (in.hasNonWallTurbWallFunc)
        throw std::runtime_error(
            "kOmegaSST(cuda): a turbulence wall function sits on a patch that is not of type `wall`. "
            "nearWallDist only fills y on `wall` patches, so the wall treatment would divide by an "
            "unset distance.");
    if (in.boundedK != in.boundedOmega)
        throw std::runtime_error(
            "kOmegaSST(cuda): `bounded` is set on one of div(phi,k)/div(phi,omega) and not the other. "
            "This closure carries one flag for both; refusing rather than bounding an equation the case "
            "did not ask to bound.");
    if (!in.nuCell || !in.nuBndFace)
        throw std::runtime_error(
            "kOmegaSST(cuda): the compressible closure needs nu = mu(T)/rho per cell AND per boundary "
            "face. A case-constant nu is not the same field on a wall with a temperature gradient.");
}

// The scheme block every transported scalar here shares -- one place, so k and omega cannot drift.
turbulence::TransportScheme schemeOf(const KOmegaSSTInput& in)
{
    turbulence::TransportScheme sc;
    sc.phiInt             = in.phiInt;
    sc.phiBnd             = in.phiBnd;
    sc.limitedLinear      = in.limitedLinear;
    sc.limiterCoeff       = in.limiterCoeff;
    sc.limGradK           = in.limGradK;
    sc.limGradLeastSq     = in.limGradLeastSq;
    sc.linearUpwind       = in.linearUpwind;
    sc.luGradK            = in.luGradK;
    sc.correctedLaplacian = in.correctedLaplacian;
    sc.gradFieldLimitK    = in.co.gradKLimitK;
    sc.snGradLimitCoeff   = in.snGradLimitCoeff;
    return sc;
}

turbulence::SolveControls solveOf(const KOmegaSSTInput& in)
{
    turbulence::SolveControls sv;
    sv.tol         = in.tol;
    sv.relTol      = in.relTol;
    sv.maxIter     = in.maxIter;
    sv.minIter     = in.minIter;
    sv.nSweeps     = in.nSweepsKE;
    sv.gsSymmetric = in.gsSymmetric;
    sv.precon      = in.precon;
    sv.polyDeg     = in.polyDeg;
    return sv;
}

// The wall cells' omega and production are the WALL FUNCTION's, not the transport's. OpenFOAM sets
// both inside omegaWallFunction::updateCoeffs before the equations are formed
// (omegaWallFunctionFvPatchScalarField.C), which is why this runs first.
__global__
void overrideWallKernel(
    int nC,
    const label*  __restrict__ isWallCell,
    const scalar* __restrict__ G0,
    const scalar* __restrict__ omega0,
    scalar*       __restrict__ G,
    scalar*       __restrict__ omega)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC || !isWallCell[c]) return;
    G[c]     = G0[c];
    omega[c] = omega0[c];
}

// F1 at a boundary face, taken from the ADJACENT CELL. Not the field's boundary condition: F1 is a
// calculated field and applying omega's conditions to it returns omega's refValue where F1 must lie in
// [0,1]. Measured on the legacy path: that made omega 100x worse (3.1e-05 -> 3.2e-03).
__global__
void gatherCellToFaceKernel(
    int nB,
    const label*  __restrict__ faceCell,
    const scalar* __restrict__ cell,
    scalar*       __restrict__ face)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    face[i] = cell[faceCell[i]];
}


// F1 ON the boundary faces. F1 is a volScalarField built by field algebra from k_, omega_, y_ and
// CDkOmega (kOmegaSSTBase.C:47-70), so its patch value is that expression evaluated with each
// operand's PATCH value -- not the owner cell's F1 interpolated out, and not omega's BCs applied to
// F1. The two differ the moment k_b or omega_b stops equalling the cell, i.e. from the second
// iteration on a uniform start; measured k 3.4e-07 vs OpenFOAM in the inlet cells at iteration 2,
// decaying downstream, with every wall row at 1e-12 -- because a zeroGradient k patch and a pinned
// omega wall row both multiply this diffusivity by zero, so the inlet is the only patch it reaches.
// CDkOmega's own patch value needs the gradients' patch values, and gaussGrad::correctBoundaryConditions
// (gaussGrad.C:96-115) replaces their normal component with the patch snGrad: gb = gc + n*(snGrad - n&gc).
// y_b is the wall-distance field's patch value: zeroGradient off a wall, so the owner cell's.
__global__
void f1BoundaryKernel(
    int nB,
    const label*  __restrict__ faceCell,
    const label*  __restrict__ gFace,
    const label*  __restrict__ f1One,       // 1 on a wall or empty patch: F1 = 1 there (y_b = 0)
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ deltaCoeffs,
    const scalar* __restrict__ kCell,
    const scalar* __restrict__ omCell,
    const scalar* __restrict__ kB,
    const scalar* __restrict__ omB,
    const scalar* __restrict__ kgx,
    const scalar* __restrict__ kgy,
    const scalar* __restrict__ kgz,
    const scalar* __restrict__ ogx,
    const scalar* __restrict__ ogy,
    const scalar* __restrict__ ogz,
    const scalar* __restrict__ yCell,
    const scalar* __restrict__ nuB,
    scalar betaStar,
    scalar alphaOmega2,
    scalar* __restrict__ F1b)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    if (f1One && f1One[i]) { F1b[i] = 1.0; return; }

    const int    c  = faceCell[i];
    const scalar ob = omB[i];
    const scalar yb = yCell[c];              // y is zeroGradient off a wall patch
    if (!(ob > 0.0) || !(yb > 0.0)) { F1b[i] = 1.0; return; }

    const int    f    = gFace[i];
    const scalar iMag = 1.0 / magSf[f];
    const scalar nx = Sfx[f]*iMag, ny = Sfy[f]*iMag, nz = Sfz[f]*iMag;

    const scalar snK = (kB[i]  - kCell[c])  * deltaCoeffs[i];
    const scalar snO = (ob     - omCell[c]) * deltaCoeffs[i];
    const scalar nK  = nx*kgx[c] + ny*kgy[c] + nz*kgz[c];
    const scalar nO  = nx*ogx[c] + ny*ogy[c] + nz*ogz[c];
    const scalar gKx = kgx[c] + nx*(snK - nK), gKy = kgy[c] + ny*(snK - nK), gKz = kgz[c] + nz*(snK - nK);
    const scalar gOx = ogx[c] + nx*(snO - nO), gOy = ogy[c] + ny*(snO - nO), gOz = ogz[c] + nz*(snO - nO);

    const scalar CDb = (2.0*alphaOmega2) * (gKx*gOx + gKy*gOy + gKz*gOz) / ob;
    const scalar CDp = fmax(CDb, scalar(1.0e-10));
    const scalar a   = fmax((1.0/betaStar)*sqrt(fmax(kB[i], scalar(0.0)))/(ob*yb),
                            500.0*nuB[i]/(yb*yb*ob));
    const scalar b   = (4.0*alphaOmega2)*kB[i]/(CDp*yb*yb);
    const scalar arg1 = fmin(fmin(a, b), scalar(10.0));
    const scalar a4   = arg1*arg1*arg1*arg1;
    F1b[i] = tanh(a4);
}

// `bounded Gauss ...` -- boundedConvectionScheme subtracts fvm::Sp(fvc::div(phi), psi), i.e. divPhi*V
// off the diagonal, with the EQUATION's own (mass) flux. The kEpsilon closure folds this into its
// reaction kernels; the SST's reactions are shared with the legacy path and take no such flag, so it
// is applied here instead. Same term either way.
__global__
void boundedSpKernel(
    int nC,
    const scalar* __restrict__ divPhi,
    const scalar* __restrict__ V,
    scalar*       __restrict__ diag)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    diag[c] -= divPhi[c] * V[c];
}

// The nut wall-function FAMILY on the wall-function faces, applied over the field assignment
// deviceSSTNutBoundary just made. A nut wall function is a property of nut's OWN patch, not of the
// turbulence model, so this is the same dispatch and the same BRAE_HD formulas the kEpsilon closure
// makes -- OpenFOAM reaches both through the one virtual calcNut()
// (nutWallFunctionFvPatchScalarField.C:182).
__global__
void wallNutDispatchKernel(
    int nB,
    const label*  __restrict__ wfMask,
    const label*  __restrict__ bndCell,
    const scalar* __restrict__ y,
    const scalar* __restrict__ kCell,
    const scalar* __restrict__ nuFace,
    const label*  __restrict__ wfKind,
    const scalar* __restrict__ wfCmu25,
    const scalar* __restrict__ wfKappa,
    const scalar* __restrict__ wfE,
    const scalar* __restrict__ wfYplLam,
    const scalar* __restrict__ Ucx, const scalar* __restrict__ Ucy, const scalar* __restrict__ Ucz,
    const scalar* __restrict__ Ubx, const scalar* __restrict__ Uby, const scalar* __restrict__ Ubz,
    scalar*       __restrict__ nutBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB || !wfMask[i]) return;
    const scalar kappa  = wfKappa  ? wfKappa[i]  : scalar(0.41);
    const scalar E      = wfE      ? wfE[i]      : scalar(9.8);
    const scalar Cmu25  = wfCmu25  ? wfCmu25[i]  : scalar(0.5477225575051661);
    const scalar yplLam = wfYplLam ? wfYplLam[i] : scalar(11.53);
    const int kind = wfKind ? static_cast<int>(wfKind[i]) : static_cast<int>(NutWall::Nutk);
    if (kind == static_cast<int>(NutWall::LowRe)) { nutBnd[i] = scalar(0); return; }
    if (kind == static_cast<int>(NutWall::NutU) && Ucx && Ubx)
    {
        const int c = bndCell[i];
        const scalar dx = Ucx[c] - Ubx[i], dy = Ucy[c] - Uby[i], dz = Ucz[c] - Ubz[i];
        nutBnd[i] = nutUWallValue(sqrt(dx*dx + dy*dy + dz*dz), y[i], nuFace[i], kappa, E, yplLam);
        return;
    }
    const scalar yp = yPlusWall(Cmu25, y[i], kCell[bndCell[i]], nuFace[i]);
    nutBnd[i] = nutkWallFunctionValue(yp, nuFace[i], yplLam, kappa, E);
}

// BRAE_DUMP_TERMS=<dir>: every contribution to this field's equation, separately, per cell, per call --
// the SAME columns and the same capture points as the legacy path's dump in device_scalar_transport.cuh,
// so a file from each closure diffs line for line. A global norm cannot say WHICH term is wrong; this
// can. The reaction is passed as a callable so the diagonal is captured either side of it.
template <typename Reaction>
void dumpTerms(const char* fieldName, int nC, const DeviceBuffer<scalar>& field,
               PressureMatrix& M, Reaction&& reaction)
{
    const char* termDir = std::getenv("BRAE_DUMP_TERMS");
    if (!termDir) { reaction(); return; }
    DeviceBuffer<scalar> dgConv;
    deviceCopy(dgConv, M.diag);
    reaction();
    static int callNo = 0;
    const int myCall = callNo++;
    std::error_code tec;
    std::filesystem::create_directories(termDir, tec);
    char fn[512];
    std::snprintf(fn, sizeof fn, "%s/%s_%04d", termDir, fieldName, myCall);
    std::ofstream o(fn);
    o.precision(10);
    const std::vector<scalar> hF = field.host(), hDc = dgConv.host(),
                              hDr = M.diag.host(), hSr = M.source.host();
    o << "# cell field diagConvLap diagAfterReact srcReact\n";
    for (int c = 0; c < nC; ++c)
        o << c << ' ' << hF[c] << ' ' << hDc[c] << ' ' << hDr[c] << ' ' << hSr[c] << '\n';
    // The OFF-DIAGONALS and the BOUNDARY coefficients. The per-cell columns above cover the diagonal
    // and the source only, and internalCoeffs never enters M.diag -- it is folded in at solve time --
    // so a wrong boundary diffusivity is invisible there. That is exactly where the boundary DEff
    // (DomB/DkB) lands, and where a near-wall error would come from.
    auto l2 = [](const DeviceBuffer<scalar>& b)
    {
        if (!b.size()) return double(0);
        const std::vector<scalar> h = b.host();
        double s2 = 0; for (scalar v : h) s2 += double(v) * double(v);
        return std::sqrt(s2);
    };
    std::printf("  [sst] %-5s |upper| %.10g  |lower| %.10g  |iC| %.10g  |bC| %.10g\n",
                fieldName, l2(M.upper), l2(M.lower), l2(M.iC), l2(M.bC));
}

} // namespace


void correct(
    DeviceBuffer<scalar>&       k,
    DeviceBuffer<scalar>&       omega,
    DeviceBuffer<scalar>&       nut,
    DeviceBuffer<scalar>&       nutBnd,
    DeviceBuffer<scalar>*       alphat,
    DeviceBuffer<scalar>*       alphatBnd,
    KOmegaSSTResiduals&         res,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    DeviceBoundary&             dbK,
    DeviceBoundary&             dbOmega,
    const DeviceWallData&       wall,
    const KOmegaSSTInput&       in)
{
    refuseUnsupported(in);
    const int nC = dm.nCells;
    const int nB = dbK.n;

    // omega's PATCH VALUES as OpenFOAM reads them for CDkOmega, F1's patch values and the omega
    // assembly: the LAST evaluate's (the previous solve's correctBoundaryConditions), reassigned only on
    // the omegaWallFunction faces by calculateTurbulenceFields (`opf == scalarField(omega0, faceCells)`,
    // omegaWallFunctionFvPatchScalarField.C:167-174). Reconstructed here, before this call refreshes the
    // turbulent inlet or the flux switch or overrides a wall cell -- see the kEpsilon twin for why that
    // is the last evaluate, and for the one thing it cannot reproduce (Foam::bound's boundary pass).
    // Measured under kOmegaSST on squareBendLiq's geometry: the host with every omega patch
    // re-evaluated read omega 3.2e-06 off OpenFOAM at iteration 2, and 1.6e-12 with only the wall
    // patches reassigned.
    DeviceBuffer<scalar> omegaBndLast;
    if (dbOmega.n) deviceBCValue(dbOmega, omega, omegaBndLast);
    // k's, as the HOST REFERENCE reads them: kOmegaSST_cpp.cu refreshes the k patches with
    // updateTurbulentInlet/updateFromFlux (:657-658, coefficients only) and its k gradients read
    // k.boundary[pi]->value() (:670, :708) -- the value the previous solve's k.evaluateBoundary() left
    // (:726). Rebuilt here for the same reason as the kEpsilon twin (kBndLast).
    DeviceBuffer<scalar> kBndLast;
    if (dbK.n) deviceBCValue(dbK, k, kBndLast);

    // The closure instrument -- see sst_stage_dump.cuh. Same names as the host twin's, so the two arms
    // diff stage by stage.
    const brae::turbulence::SstStageDump sd = brae::turbulence::sstStageDump("cuda");
    if (sd.on)
    {
        if (in.yCell) sd.scalars("y", in.yCell->host());
        sd.scalars("kIn", k.host());
        sd.scalars("omegaIn", omega.host());
        sd.scalars("nutIn", nut.host());
        sd.scalars("Ux", in.Ux->host());
        sd.scalars("Uy", in.Uy->host());
        sd.scalars("Uz", in.Uz->host());
        // The patch values the device gradient sums over, in the same patch order as the host's. This
        // arm stores none -- it evaluates them from the conditions and the cells -- so the same
        // reconstruction the gradient uses is what gets written.
        DeviceBuffer<scalar> ubx, uby, ubz;
        deviceBCValue(dbU.comp[0], *in.Ux, ubx);
        deviceBCValue(dbU.comp[1], *in.Uy, uby);
        deviceBCValue(dbU.comp[2], *in.Uz, ubz);
        sd.scalars("UbX", ubx.host());
        sd.scalars("UbY", uby.host());
        sd.scalars("UbZ", ubz.host());
    }

    // ---- production, from the CURRENT nut (the previous outer iteration's correctNut) ----------
    DeviceBuffer<scalar> gradU, S2, GbyNu0, G;
    deviceGradU(dm, dbU, *in.Ux, *in.Uy, *in.Uz, gradU);
    if (in.gradULimitK > scalar(0))
        deviceCellLimitGradU(dm, dbU, *in.Ux, *in.Uy, *in.Uz, gradU, in.gradULimitK);
    deviceS2(gradU, nC, S2);
    deviceGByNuFromGradU(gradU, nC, GbyNu0);
    G.resize(nC);
    deviceHadamard(G, nut, GbyNu0);
    if (sd.on)
    {
        // The device holds gradU COMPONENT-major (all xx, then all xy, ...); the host holds one tensor
        // per cell. Remapped here so both arms write the same nine columns per cell and a diff of the
        // two files is a diff of the gradient, not of two layouts.
        const std::vector<scalar> gu = gradU.host();
        std::vector<scalar> flat(gu.size());
        for (int comp = 0; comp < 9; ++comp)
            for (int c = 0; c < nC; ++c)
                flat[static_cast<std::size_t>(c) * 9 + comp] = gu[static_cast<std::size_t>(comp) * nC + c];
        sd.components("gradU", flat, 9);
        sd.scalars("S2", S2.host());
        sd.scalars("GbyNu0", GbyNu0.host());
    }

    // TWO divergences, because OpenFOAM uses two different fluxes and they coincide only at constant
    // density: `bounded` subtracts fvm::Sp(fvc::div(phi), psi) with the equation's OWN (mass) flux,
    // while the reactions' (2/3)divU takes the VOLUMETRIC one. Feeding the mass-flux divergence to the
    // reactions makes that term wrong by a factor of rho.
    DeviceBuffer<scalar> divPhi, divU;
    deviceDiv(dm, *in.phiInt, *in.phiBnd, divPhi);
    if (in.phiByRhoInt && in.phiByRhoBnd) deviceDiv(dm, *in.phiByRhoInt, *in.phiByRhoBnd, divU);
    else                                  deviceCopy(divU, divPhi);

    // ---- omegaWallFunction FIRST: OpenFOAM's order --------------------------------------------
    // kOmegaSSTBase::correct() calls omega_.boundaryFieldRef().updateCoeffs() (kOmegaSSTBase.C:541)
    // before it forms anything, and that is where the near-wall omega and the G override are set.
    // omega = sqrt(omegaVis^2 + omegaLog^2) -- OpenFOAM's DEFAULT blender here is binomial with n = 2,
    // not the stepwise the epsilon wall function takes.
    DeviceBuffer<scalar> omega0, G0;
    // omega_.boundaryFieldRef().updateCoeffs() (kOmegaSSTBase.C:541), in full: the turbulent inlet
    // recomputes its refValue from k's CURRENT patch values FIRST, then the flux switch resolves which
    // faces are fixedValue at all -- reversed, the switch would act on the previous iteration's value.
    // The Cmu is turbulentMixingLengthFrequencyInletFvPatchScalarField.C:137-138's
    // `turbModel.coeffDict().getOrDefault("Cmu", 0.09)`, which for kOmegaSST is 0.09 unless the case
    // wrote a Cmu into kOmegaSSTCoeffs -- betaStar is a different key the inlet never reads.
    if (in.turbInletOmegaMask && in.turbInletOmegaLen)
        deviceUpdateTurbulentInletSecond(dbK, *in.turbInletOmegaMask, *in.turbInletOmegaLen,
                                         in.co.Cmu, dbOmega);
    if (in.phiBnd) deviceUpdateInletOutlet(dbOmega, *in.phiBnd);

    // nutWallFace is the STORED nut boundary, which is what omegaWallFunction's G0 reads; with it the
    // /*nutWall=*/0 literal below is inert for G0, since the stored value already carries the family.
    deviceWallOmegaG0(wall, k, *in.Ux, *in.Uy, *in.Uz, scalar(0), omega0, G0, in.co,
                      /*nutWall=*/0, /*atmZ0=*/0.0, /*atmBoundNut=*/true, in.nuWallFace,
                      in.nutWallFace);
    // The wall cells' production is the wall function's, not the strain's.
    if (wall.nWF > 0)
    {
        overrideWallKernel<<<nBlk(nC), TPB>>>(nC, wall.isWallCell.data(), G0.data(), omega0.data(),
                                              G.data(), omega.data());
        cudaCheck(cudaGetLastError(), "kOmegaSST wall override");
    }
    if (dbOmega.n && in.wfBndMask && in.wfBndMask->size() == static_cast<std::size_t>(dbOmega.n))
    {
        turbulence::wallFacesTakeCell(dm, *in.wfBndMask, omega, omegaBndLast);
    }

    // ---- CDkOmega, F1, F2 ---------------------------------------------------------------------
    // CDkOmega takes fvc::grad(k_) & fvc::grad(omega_) (kOmegaSSTBase.C:555-558), and fvc::grad(vf)
    // resolves its scheme by the FIELD's name -- "grad(k)", "grad(omega)" -- through gradSchemes
    // (fvcGrad.C:149). A case naming those `cellLimited Gauss linear 1` therefore limits both, which the
    // host reference does (kOmegaSST_cpp.cu:458-461) and this arm did not: it took the plain Gauss
    // gradient whatever the case said, the same defect grad(U) already carried a limiter for above.
    // Invisible on every fixture whose grad(k)/grad(omega) is plain `Gauss linear`. Measured on
    // OpenFOAM's own aerofoilNACA0012 tutorial, which names cellLimited for U, k and omega: CUDA arm
    // k 7.0e-09 off OpenFOAM at iteration 1 in the 554 cells around the farfield, growing to omega
    // 3.7e-03 and U 1.1e-08 by iteration 3, where the host arm holds 2.6e-12 throughout.
    DeviceBuffer<scalar> kbv, obv, kgx, kgy, kgz, ogx, ogy, ogz, CD, F1, F2;
    deviceBCValue(dbK, k, kbv);
    deviceGaussGrad(dm, k, kbv, kgx, kgy, kgz);
    if (in.co.gradKLimitK > scalar(0))
        deviceCellLimitGrad(dm, k, kbv, kgx, kgy, kgz, in.co.gradKLimitK);
    if (omegaBndLast.size()) deviceCopy(obv, omegaBndLast);
    else                     deviceBCValue(dbOmega, omega, obv);
    deviceGaussGrad(dm, omega, obv, ogx, ogy, ogz);
    if (in.co.gradKLimitK > scalar(0))
        deviceCellLimitGrad(dm, omega, obv, ogx, ogy, ogz, in.co.gradKLimitK);
    deviceCDkOmega(kgx, kgy, kgz, ogx, ogy, ogz, omega, in.co.alphaOmega2, CD);
    // F1/F2 blend on the KINEMATIC laminar viscosity, per cell -- the compressible lineage has no
    // case-constant nu, and arg1/arg2 are written in nu, not mu.
    deviceF1(k, omega, *in.yCell, CD, scalar(0), in.co, F1, /*lm=*/false, in.nuCell);
    deviceF2(k, omega, *in.yCell, scalar(0), in.co, F2, in.nuCell);
    if (sd.on)
    {
        sd.scalars("G", G.host());
        sd.scalars("CD", CD.host());
        sd.scalars("F1", F1.host());
        sd.scalars("F23", F2.host());
    }

    // ---- the production limiter: omega uses the LIMITED GbyNu, k uses the raw G ---------------
    // kOmegaSSTBase.C reassigns GbyNu0 = GbyNu(GbyNu0, F23, S2) AFTER G was taken from the raw value.
    // Using one for both is the easy mistake here; they are different quantities.
    DeviceBuffer<scalar> gamma, beta, GbyNu0lim;
    deviceBlend(F1, in.co.gamma1, in.co.gamma2, gamma);
    deviceBlend(F1, in.co.beta1,  in.co.beta2,  beta);
    deviceGbyNuLimit(GbyNu0, omega, F2, S2, in.co, GbyNu0lim);
    if (sd.on) sd.scalars("GbyNuLim", GbyNu0lim.host());

    // The boundary diffusivities take F1 EVALUATED ON the patch faces -- see f1BoundaryKernel. Not
    // deviceBCValue on omega's boundary (that applies OMEGA's conditions to F1, so a fixedValue omega
    // inlet returns omega's refValue where F1 must lie in [0,1]), and not the owner cell's F1 either.
    DeviceBuffer<scalar> F1b, DomB, DkB;
    if (in.nutBndFace && in.nutBndFace->size())
    {
        F1b.resize(static_cast<std::size_t>(nB));
        f1BoundaryKernel<<<nBlk(nB), TPB>>>(
            nB, dbOmega.faceCell.data(), dm.bndGFace.data(),
            in.f1OneMask ? in.f1OneMask->data() : nullptr,
            dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), dm.magSf.data(), dbOmega.deltaCoeffs.data(),
            k.data(), omega.data(), kbv.data(), obv.data(),
            kgx.data(), kgy.data(), kgz.data(), ogx.data(), ogy.data(), ogz.data(),
            in.yCell->data(), in.nuBndFace->data(),
            in.co.betaStar, in.co.alphaOmega2, F1b.data());
        cudaCheck(cudaGetLastError(), "kOmegaSST F1 boundary");
        deviceDEff(F1b, *in.nutBndFace, in.co.alphaOmega1, in.co.alphaOmega2, scalar(0), DomB);
        deviceDEff(F1b, *in.nutBndFace, in.co.alphaK1,     in.co.alphaK2,     scalar(0), DkB);
        // rho*D + mu on the boundary, the dynamic form the laplacian's patch coefficient wants. mu_b is
        // formed here from nu_b*rho_b rather than carried as its own buffer: both are already required
        // inputs, and a third field that must agree with them is a third field that can disagree.
        if (in.rhoBndFace)
        {
            DeviceBuffer<scalar> muB;
            muB.resize(static_cast<std::size_t>(nB));
            deviceHadamard(muB, *in.nuBndFace, *in.rhoBndFace);
            deviceHadamard(DomB, DomB, *in.rhoBndFace); deviceAxpy(1.0, muB, DomB);
            deviceHadamard(DkB,  DkB,  *in.rhoBndFace); deviceAxpy(1.0, muB, DkB);
        }
    }

    // BRAE_SST_DEBUG=1 prints the extremes of every intermediate this closure forms, in the order it
    // forms them. The omega equation is the one that does not yet match OpenFOAM, and the terms it is
    // built from are the things to compare -- a whole-field rel-L2 cannot say which of them is wrong.
    if (std::getenv("BRAE_SST_DEBUG"))
    {
        auto rng = [&](const char* nm, const DeviceBuffer<scalar>& b)
        {
            if (!b.size()) { std::printf("  [sst] %-10s EMPTY\n", nm); return; }
            const std::vector<scalar> h = b.host();
            scalar lo = h[0], hi = h[0];
            for (scalar v : h) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); }
            std::printf("  [sst] %-10s min %.6g  max %.6g\n", nm, (double)lo, (double)hi);
        };
        rng("S2", S2); rng("GbyNu0", GbyNu0); rng("G", G);
        rng("divPhi", divPhi); rng("divU", divU);
        rng("omega0", omega0); rng("G0", G0);
        rng("CD", CD); rng("F1", F1); rng("F2", F2);
        rng("gamma", gamma); rng("beta", beta); rng("GbyNu0lim", GbyNu0lim);
        rng("omega", omega); rng("k", k); rng("nut", nut);
        std::printf("  [sst] sizes: nC %d  nB %d  dm.nBndFaces %d  DomB %zu  DkB %zu  nutBnd %zu\n",
                    nC, nB, dm.nBndFaces, DomB.size(), DkB.size(),
                    in.nutBndFace ? in.nutBndFace->size() : std::size_t(0));
    }

    const turbulence::TransportScheme sc = schemeOf(in);
    const turbulence::SolveControls   sv = solveOf(in);

    // ---- the omega equation -------------------------------------------------------------------
    // Solved FIRST, and the k equation below reads the omega this solve produced. That lag is
    // OpenFOAM's; reversing it is a different algorithm that still converges to something plausible.
    {
        DeviceBuffer<scalar> DomegaEff, gammaFace;
        deviceDEff(F1, nut, in.co.alphaOmega1, in.co.alphaOmega2, scalar(0), DomegaEff);
        if (in.rhoCell)
        {
            deviceHadamard(DomegaEff, DomegaEff, *in.rhoCell);
            DeviceBuffer<scalar> muCell; muCell.resize(nC);
            deviceHadamard(muCell, *in.nuCell, *in.rhoCell);
            deviceAxpy(1.0, muCell, DomegaEff);
        }
        deviceInterpolate(dm, DomegaEff, gammaFace);

        PressureMatrix M;
        turbulence::TransportScheme scOmega = sc;
        scOmega.bndValues = omegaBndLast.size() ? &omegaBndLast : nullptr;
        turbulence::assembleScalarTransport(M, dm, dbOmega, omega, gammaFace,
                                            DomB.size() ? DomB : gammaFace, scOmega);
        if (in.boundedOmega)
        {
            boundedSpKernel<<<nBlk(nC), TPB>>>(nC, divPhi.data(), dm.V.data(), M.diag.data());
            cudaCheck(cudaGetLastError(), "kOmegaSST bounded omega");
        }
        dumpTerms("omega", nC, omega, M, [&]{
            deviceOmegaReaction(dm.V, gamma, beta, GbyNu0lim, F1, CD, omega, divU,
                                M.diag, M.source, in.rhoCell);
        });

        turbulence::solveScalarEqn(M, omega, dm, in.relaxEquationOmega, in.relaxOmega,
                                   in.fvoOmegaMask, in.fvoOmegaVal,
                                   // The per-CELL mask, not the wall-cell LIST: fvMatrix::setValues
                                   // takes cells and values indexed the same way, and wfCell is a
                                   // compacted list of indices. The VALUE is omega0, the wall
                                   // function's own near-wall omega -- unlike epsilon, whose
                                   // constraint value is the current field.
                                   wall.nWF > 0 ? &wall.isWallCell : nullptr,
                                   wall.nWF > 0 ? &omega0 : nullptr,
                                   sv, res.omega, std::string(), in.gsOmega);
        // Foam::bound(omega_, omegaMin_) -- the mirror's area-weighted form, not a clamp.
        kEpsilonRAS::boundField(omega, dm, dbOmega, in.co.omegaMin, "omega");
        if (std::getenv("BRAE_SST_DEBUG"))
        {
            std::printf("  [sst] omega solve: initialResidual %.6g  (tol %.3g relTol %.3g maxIter %d gs %d)\n",
                        (double)res.omega, (double)in.tol, (double)in.relTol, in.maxIter, (int)in.gsOmega);
            std::printf("  [sst] relaxOmega %.6g (named %d)  relaxK %.6g (named %d)\n",
                        (double)in.relaxOmega, (int)in.relaxEquationOmega,
                        (double)in.relaxK, (int)in.relaxEquationK);
            std::printf("  [sst] precon %s, valid %d, polyDeg %d\n",
                        in.precon ? "SET" : "NULL", in.precon ? (int)in.precon->valid : -1, in.polyDeg);
            // Is every wall cell still pinned to omega0? fvMatrix::setValues writes psi as well as the
            // matrix, so after the solve a wall cell must hold exactly the wall function's value. If it
            // does not, the constraint is not reaching the solve.
            const std::vector<scalar> ho = omega.host(), h0 = omega0.host();
            const std::vector<label>  hw = wall.isWallCell.host();
            int nw = 0, ndrift = 0; scalar worst = 0;
            for (int c = 0; c < nC; ++c)
            {
                if (!hw[c]) continue;
                ++nw;
                const scalar d = std::fabs(ho[c] - h0[c]) / std::fmax(std::fabs(h0[c]), scalar(1e-30));
                if (d > scalar(1e-10)) { ++ndrift; worst = std::fmax(worst, d); }
            }
            std::printf("  [sst] wall cells %d, drifted off omega0 after the solve: %d (worst rel %.3g)\n",
                        nw, ndrift, (double)worst);
        }
    }

    // ---- the k equation -----------------------------------------------------------------------
    {
        DeviceBuffer<scalar> DkEff, gammaFace;
        deviceDEff(F1, nut, in.co.alphaK1, in.co.alphaK2, scalar(0), DkEff);
        if (in.rhoCell)
        {
            deviceHadamard(DkEff, DkEff, *in.rhoCell);
            DeviceBuffer<scalar> muCell; muCell.resize(nC);
            deviceHadamard(muCell, *in.nuCell, *in.rhoCell);
            deviceAxpy(1.0, muCell, DkEff);
        }
        deviceInterpolate(dm, DkEff, gammaFace);

        // k_'s own boundary refresh, at OpenFOAM's point for it: the fvMatrix constructor at
        // kOmegaSSTBase.C:600, i.e. AFTER the gradients and F1 were taken from the previous iteration's
        // k boundary. turbulentIntensityKineticEnergyInlet reads U's CURRENT patch values.
        if (in.turbInletKMask && in.turbInletKInt)
            deviceUpdateTurbulentInletK(dbU, *in.turbInletKMask, *in.turbInletKInt, dbK);
        if (in.phiBnd) deviceUpdateInletOutlet(dbK, *in.phiBnd);

        PressureMatrix M;
        turbulence::TransportScheme scK = sc;
        scK.bndValues = kBndLast.size() ? &kBndLast : nullptr;
        turbulence::assembleScalarTransport(M, dm, dbK, k, gammaFace,
                                            DkB.size() ? DkB : gammaFace, scK);
        if (in.boundedK)
        {
            boundedSpKernel<<<nBlk(nC), TPB>>>(nC, divPhi.data(), dm.V.data(), M.diag.data());
            cudaCheck(cudaGetLastError(), "kOmegaSST bounded k");
        }
        dumpTerms("k", nC, k, M, [&]{
            deviceKReactionSST(dm.V, k, omega, G, divU, in.co, M.diag, M.source,
                               /*gammaIntEff=*/nullptr, /*FDES=*/nullptr, in.rhoCell);
        });

        // No wall mask: kOmegaSSTBase.C has no boundaryManipulate for k -- kqRWallFunction is
        // zeroGradient, and constraining k in wall cells the way omega is constrained is a different
        // equation.
        turbulence::solveScalarEqn(M, k, dm, in.relaxEquationK, in.relaxK,
                                   in.fvoKMask, in.fvoKVal, nullptr, nullptr,
                                   sv, res.k, std::string(), in.gsK);
        kEpsilonRAS::boundField(k, dm, dbK, in.co.kMin, "k");
        if (std::getenv("BRAE_SST_DEBUG"))
            std::printf("  [sst] k     solve: initialResidual %.6g\n", (double)res.k);
    }

    // ---- correctNut(S2) -----------------------------------------------------------------------
    // F2 is RECOMPUTED from the just-solved, just-bounded k and omega: OpenFOAM's correctNut reads the
    // members, and they have moved since the F2 above was formed for the production limiter.
    {
        DeviceBuffer<scalar> F2n;
        deviceF2(k, omega, *in.yCell, scalar(0), in.co, F2n, in.nuCell);
        deviceNutSST(k, omega, F2n, S2, in.co, nut);

        if (nB > 0)
        {
            DeviceBuffer<scalar> kB, oB;
            deviceBCValue(dbK, k, kB);
            deviceBCValue(dbOmega, omega, oB);
            nutBnd.resize(nB);
            deviceSSTNutBoundary(dbU, kB, oB, *in.yCell, in.nuBndFace, scalar(0), gradU, nC,
                                 *in.Ux, *in.Uy, *in.Uz,
                                 in.nutCalcMask ? *in.nutCalcMask : dbK.bcType,
                                 in.co, nut, nutBnd);
            // The nut wall-function FAMILY, per face, on top of the field assignment -- the same
            // dispatch the kEpsilon closure makes, from the same per-face codes.
            if (in.wfBndMask && in.wallYBndFace)
            {
                DeviceBuffer<scalar> uBx, uBy, uBz;
                deviceBCValue(dbU.comp[0], *in.Ux, uBx);
                deviceBCValue(dbU.comp[1], *in.Uy, uBy);
                deviceBCValue(dbU.comp[2], *in.Uz, uBz);
                wallNutDispatchKernel<<<nBlk(nB), TPB>>>(
                    nB, in.wfBndMask->data(), dm.bndCell.data(), in.wallYBndFace->data(),
                    k.data(), in.nuBndFace->data(),
                    in.nutWfKindBnd   ? in.nutWfKindBnd->data()   : nullptr,
                    in.nutWfCmu25Bnd  ? in.nutWfCmu25Bnd->data()  : nullptr,
                    in.nutWfKappaBnd  ? in.nutWfKappaBnd->data()  : nullptr,
                    in.nutWfEBnd      ? in.nutWfEBnd->data()      : nullptr,
                    in.nutWfYplLamBnd ? in.nutWfYplLamBnd->data() : nullptr,
                    in.Ux->data(), in.Uy->data(), in.Uz->data(),
                    uBx.data(), uBy.data(), uBz.data(), nutBnd.data());
                cudaCheck(cudaGetLastError(), "kOmegaSST wall nut");
            }
        }
    }

    // EddyDiffusivity::correctNut -- alphat = rho*nut/Prt, unconditional and whole-field.
    if (alphat && in.rhoCell)
    {
        alphat->resize(nC);
        deviceHadamard(*alphat, *in.rhoCell, nut);
        deviceScale(*alphat, scalar(1) / in.Prt);
        // The BOUNDARY half goes through the same kernel the kEpsilon closure uses, on the same mask and
        // the same per-face Prt -- see KOmegaSSTInput::alphatWallMask. Doing it whole-field here wrote
        // over the fixedValue faces and used the model's Prt on the wall-function ones.
        if (alphatBnd && in.alphatWallMask && in.alphatPrtFace && in.rhoBndFace && nB > 0)
            kEpsilonRAS::alphatBoundary(*alphatBnd, nB, *in.alphatWallMask, *in.rhoBndFace,
                                        nutBnd, *in.alphatPrtFace);
    }
}

} // namespace kOmegaSSTRAS
} // namespace gpu
} // namespace brae
