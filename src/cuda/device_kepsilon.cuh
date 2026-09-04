#pragma once
// cf GPU offload (#3): k-epsilon turbulence physics on device. The production term GbyNu = gradU &&
// devTwoSymm(gradU) and the eddy viscosity nut = Cmu k^2/eps. The transport equations (eps/k) reuse the
// existing device assembly (div + laplacian, G4) + reaction terms + the scalar BiCGStab (G6); the wall
// functions + setValues constraint are the remaining wiring.
#include "cf_types.cuh"
#include "kepsilon_coeffs.cuh"
#include "device_buffer.cuh"
#include "device_ddt.cuh"          // ScalarDdt (transient turbulence ddt bundle)
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "near_wall_dist.cuh"
#include "komega_sst_coeffs.cuh"
#include "spalart_coeffs.cuh"
#include "device_pcg.cuh"     // DeviceSolverPerf (turbulence solve report)
#include <vector>
#include <string>

namespace brae {

struct DeviceAMI;     // cyclicAMI interface (scalar-transport coupling, optional)
struct DeviceCyclic;  // cyclic interface     (scalar-transport coupling, optional)

// OF-style turbulence residual reporting. Every turbulence scalar solve (k / epsilon / omega / nuTilda / ReThetat /
// gammaInt) appends its field name + solve perf here, in solve order, so the SIMPLE driver can print an
// "smoothSolver:  Solving for <field>, Initial residual = ..." block exactly like OpenFOAM. Cleared once per SIMPLE step.
struct ScalarSolveEntry
{
    std::string field;
    DeviceSolverPerf perf;
};
void clearTurbulenceReport();
const std::vector<ScalarSolveEntry>& turbulenceReport();

// k-epsilon model coefficients (shared CPU/device definition).
// GbyNu = gradU && devTwoSymm(gradU), devTwoSymm(g) = g + g^T - (2/3)tr(g) I.  G = nut*GbyNu.
void deviceGbyNu(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                 const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                 DeviceBuffer<scalar>& gByNu, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr);

// gradU tensor (9*nC, OF convention column i = gaussGrad(U_i)) + GbyNu from a prebuilt gradU. Shared by k-eps
// (GbyNu) and kOmegaSST (which also needs gradU for S2 = 2 magSqr(symm(gradU))).
void deviceGradU(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                 const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                 DeviceBuffer<scalar>& gradU, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr);
void deviceGByNuFromGradU(const DeviceBuffer<scalar>& gradU, int nC, DeviceBuffer<scalar>& gByNu);
// OF grad(U) cellLimited Gauss linear <k> applied to a grad(U) TENSOR (per U-component minmod limiter). For the
// turbulence strain S2 + production, which OF computes from fvc::grad(U) = the (cellLimited) grad(U) scheme.
void deviceCellLimitGradU(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                          const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                          DeviceBuffer<scalar>& gradU, scalar kc,
                          const DeviceCyclic* cyc = nullptr, const DeviceAMI* ami = nullptr);

// nut = Cmu k^2 / eps.
void deviceNut(const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& eps, DeviceBuffer<scalar>& nut,
               const KEpsilonCoeffs& co = {});

// OF Foam::bound(vsf, lowerBound): where a cell falls below the floor, replace it with the bounded
// neighbour-average (fvc::average(max(vsf,floor))*pos0(-vsf)), NOT a hard clamp; floor elsewhere.
void deviceBoundField(const DeviceMesh& dm, DeviceBuffer<scalar>& x, scalar floor);

// Static wall geometry for the wall functions (nearWallDist y, wall-face cell/deltaCoeffs/velocity, 1/nWallFaces).
struct DeviceWallData
{
    int nWF = 0;
    // isWallCell: "this cell's epsilon/omega is FIXED by a wall function". Not the same question as
    // "does this cell touch a wall patch", and the difference is a cyclicACMI. Its non-overlap patch is
    // a wall of area (1-mask)*A, so a fully OPEN one is a wall face with no wall behind it, and OF does
    // not constrain its cell -- cyclicACMIFvPatchField::manipulateMatrix redirects to the non-overlap
    // patch with weights (1-mask), and epsilonWallFunction only constrains where weight > 1e-5.
    // See buildDeviceWallData for what counting them anyway cost.
    DeviceBuffer<label>  wfCell, isWallCell;
    // Wall-face gather grouped by cell -- see the note in buildWallData. nWC wall cells; wcCell[i] is the
    // mesh cell, and wcFace[wcStart[i] .. wcStart[i+1]) are its wall faces in ascending face index.
    int                  nWC = 0;
    DeviceBuffer<label>  wcCell, wcStart, wcFace;
    // ...and the weight itself, for the partially blocked faces in between: OF blends rather than
    // switches, G[c] = (1-w)*G[c] + w*G0[c] and the same for epsilon (epsilonWallFunction.C:592).
    DeviceBuffer<scalar> wallW;
    DeviceBuffer<scalar> wfY, wfDc, wfUwx, wfUwy, wfUwz, invNw;
};
// The predicate the wall set is built on, in one place so the DeviceWallData faces and the wall-face ->
// boundary-face map below cannot drift apart.
inline bool isTurbWallPatch(const std::vector<FvPatch>& fvp, std::size_t pi, const std::vector<char>& wfPatch)
{
    if (fvp[pi].type != "wall") return false;
    return wfPatch.empty() || (pi < wfPatch.size() && wfPatch[pi]);
}

// The wall velocity comes in per patch rather than from a GeometricField, because on a MOVING mesh the
// two can disagree: `movingWallVelocity` is assigned into the solver's device boundary after the move
// (setPatchVelocity), and the host field is not what the solver imposes. See refreshWallData.
// `wfPatch`, when non-empty, says which patches carry the turbulence wall function -- see
// DeviceSimpleControls::turbWallPatch and wallFunctionPatchMask below. A patch has to be BOTH a `wall`
// and named by its epsilon/omega BC, because that is the set OpenFOAM overrides: the wall function is a
// BC object on that field, so a `wall`-typed patch whose epsilon BC is plain zeroGradient gets nothing.
// Empty = fall back to the patch type alone, which is what the SA and LES paths want.
inline DeviceWallData buildDeviceWallData(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::vector<std::vector<vector>>& wallU,
    const std::vector<char>& wfPatch = {})
{
    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
    std::vector<label> wfCell;
    std::vector<scalar> wfY, wfDc, wux, wuy, wuz;
    std::vector<label> nw(m.nCells(), 0);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        if (isTurbWallPatch(fvp, pi, wfPatch))
        {
            const std::vector<vector>& uv = wallU[pi];
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label c = fvp[pi].faceCells[i];
                wfCell.push_back(c);
                wfY.push_back(yW[pi][i]);
                wfDc.push_back(fvp[pi].deltaCoeffs[i]);
                wux.push_back(uv[i].x);
                wuy.push_back(uv[i].y);
                wuz.push_back(uv[i].z);
                ++nw[c];
            }
        }
    std::vector<scalar> invNw(m.nCells(), 0.0);
    std::vector<label> isW(m.nCells(), 0);
    for (label c = 0; c < m.nCells(); ++c)
        if (nw[c] > 0)
        {
            invNw[c] = 1.0 / nw[c];
            isW[c] = 1;
        }

    // WALL-CELL GATHER ADDRESSING (determinism).
    //
    // The wall kernels used to run one thread per wall FACE and atomicAdd the near-wall production into
    // G0[cell] and eps0/omega0[cell]. `invNw` is 1/(wall faces on this cell), so cells with more than one
    // wall face demonstrably exist -- and for those the summation order was whatever order the faces
    // happened to be scheduled in. It is a rare race (most wall cells have exactly one wall face), which
    // made it worse rather than better: two identical 1-iteration pitzDaily runs came out bit-identical
    // twice and 1 ULP apart on the third, on epsilon and nut only. The SIMPLE loop then amplified that
    // single ULP to 1.3e-3 by iteration 20.
    //
    // Grouping the wall faces by cell lets one thread own a cell and sum its faces in ascending face
    // index -- a fixed order -- then write once.
    std::vector<label> wcCell;
    for (label c = 0; c < m.nCells(); ++c) if (nw[c] > 0) wcCell.push_back(c);
    std::vector<label> cellSlot(m.nCells(), -1);
    for (std::size_t i = 0; i < wcCell.size(); ++i) cellSlot[wcCell[i]] = static_cast<label>(i);

    std::vector<label> wcStart(wcCell.size() + 1, 0);
    for (std::size_t f = 0; f < wfCell.size(); ++f) ++wcStart[cellSlot[wfCell[f]] + 1];
    for (std::size_t i = 0; i < wcCell.size(); ++i) wcStart[i+1] += wcStart[i];
    std::vector<label> wcFace(wfCell.size());
    {
        std::vector<label> at(wcStart.begin(), wcStart.end() - 1);
        for (std::size_t f = 0; f < wfCell.size(); ++f)
            wcFace[at[cellSlot[wfCell[f]]]++] = static_cast<label>(f);
    }
    // THE ACMI NON-OVERLAP WALL IS ONLY A WALL WHERE IT IS CLOSED.
    //
    // A cyclicACMI carries a coincident wall (its nonOverlapPatch) whose area is (1-mask)*A, so on the
    // covered part of the interface it is a `wall` patch with essentially zero area -- and brae counted
    // every one of its faces as a wall face. That put the near-wall epsilon on cells that are not near a
    // wall at all, and worse, deviceSolveScalarTransport zeroes the AMI off-diagonal for wall cells
    // (their value is fixed), so epsilon lost its interface coupling entirely.
    //
    // OF gates it: cyclicACMIFvPatchField::manipulateMatrix hands the non-overlap patch field a weight
    // of (1-mask) and epsilonWallFunction acts only where that exceeds tolerance_ = 1e-5, blending
    // rather than switching in between. The weight is exactly the patch's areaFraction, which brae
    // already has as scaled/raw |Sf| -- 1 on every ordinary wall, since rawMagSf falls back to magSf.
    //
    // Measured on pimpleFoam/RAS/oscillatingInletACMI2D, one step from OpenFOAM's own t=0.01: epsilon on
    // the channel's interface column was 8.20x OF's and the duct's covered band 4.09x, with the fully
    // blocked cells already correct to 3.9e-06 -- the giveaway that this was about which faces count as
    // wall, not about the wall function itself.
    constexpr scalar ACMI_WALL_TOL = 1e-5;   // OF epsilonWallFunctionFvPatchScalarField::tolerance_
    std::vector<scalar> wallW(m.nCells(), 0.0);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        if (fvp[pi].type == "wall")
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label f = fvp[pi].start + i;
                const scalar raw = g.rawMagSf(f);
                const scalar frac = raw > scalar(0) ? g.magSf()[f]/raw : scalar(1);   // OF areaFraction()
                const label c = fvp[pi].faceCells[i];
                if (frac > wallW[c]) wallW[c] = frac;
            }
    for (label c = 0; c < m.nCells(); ++c)
        if (wallW[c] <= ACMI_WALL_TOL) isW[c] = 0;   // a wall face with no wall behind it
    DeviceWallData w;
    w.nWF = static_cast<int>(wfCell.size());
    w.nWC = static_cast<int>(wcCell.size());
    w.wcCell.copyFrom(wcCell);
    w.wcStart.copyFrom(wcStart);
    w.wcFace.copyFrom(wcFace);
    w.wfCell.copyFrom(wfCell);
    w.wfY.copyFrom(wfY);
    w.wfDc.copyFrom(wfDc);
    w.wfUwx.copyFrom(wux);
    w.wfUwy.copyFrom(wuy);
    w.wfUwz.copyFrom(wuz);
    w.invNw.copyFrom(invNw);
    w.isWallCell.copyFrom(isW);
    w.wallW.copyFrom(wallW);
    return w;
}

// Convenience overload: take the wall velocities off a U field's boundary. This is the construction-time
// path, where the host field and the device boundary still agree.
inline DeviceWallData buildDeviceWallData(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const GeometricField<vector>& U,
    const std::vector<char>& wfPatch = {})
{
    std::vector<std::vector<vector>> wallU(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        if (fvp[pi].type == "wall") wallU[pi] = U.boundary[pi]->value();
    return buildDeviceWallData(m, g, fvp, wallU, wfPatch);
}

// epsilonWallFunction near-wall values: eps0 = (1/nWall) Cmu^.75 k^1.5/(kappa y); G0 = (1/nWall)(nutw+nu)*
// |snGrad U|*Cmu^.25 sqrt(k)/(kappa y); nutw = nutkWallFunction. (eps0/G0 zeroed then scattered with atomics.)
void deviceWallEpsG0(const DeviceWallData& w, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& Ux,
                     const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz, scalar nu,
                     DeviceBuffer<scalar>& eps0, DeviceBuffer<scalar>& G0, const KEpsilonCoeffs& co = {}, int nutWall = 0,
                     scalar atmZ0 = 0.0, bool atmBoundNut = true,   // z0>0 -> atmNutkWallFunction (rough) for the G0 wall nut
                     const DeviceBuffer<scalar>* nuFace = nullptr,       // compressible: nu = mu_b/rho_b per WALL face
                     // The STORED wall nut in WALL-face order (the nut boundary as it entered correct()).
                     // OpenFOAM's epsilonWallFunction reads nutw[facei] from the nut patch field -- the
                     // value the previous correctNut() or validate() left there -- not a fresh
                     // nutkWallFunction of the current k and nu_w. Null keeps the recomputation.
                     const DeviceBuffer<scalar>* nutwStored = nullptr);
// (epsilonWallFunction's `lowReCorrection` rides on KEpsilonCoeffs::epsLowRe, so it reaches the kernel
//  without threading a flag through every caller.)

// add the eps / k reaction (Sp/Su) + SuSp(divU) terms to a matrix's diag/source (in place).
void deviceEpsReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& eps, const DeviceBuffer<scalar>& k,
                       const DeviceBuffer<scalar>& gByNu, const DeviceBuffer<scalar>& divU,
                       DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source, const KEpsilonCoeffs& co = {},
                       const DeviceBuffer<scalar>* rho = nullptr);   // compressible: alpha*rho on every RHS term
void deviceKReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& eps,
                     const DeviceBuffer<scalar>& G, const DeviceBuffer<scalar>& divU,
                     DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source,
                     const DeviceBuffer<scalar>* rho = nullptr);   // compressible: alpha*rho on every RHS term
// realizableKE (OF RAS/realizableKE): variable Cmu (rCmu) + strain magnitude magS from the gradU tensor; nut =
// rCmu*k^2/eps; eps reaction = strain production C1*magS*eps - destruction C2*eps^2/(k+sqrt(nu*eps)). Cell-local
// (no halo), so the distributed kEps correct reuses them on its already-halo-consistent gradU tensor.
void deviceRealizableStrain(const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& k,
                            const DeviceBuffer<scalar>& eps, scalar A0, int nC,
                            DeviceBuffer<scalar>& rCmu, DeviceBuffer<scalar>& magS);
void deviceRealizableNut(const DeviceBuffer<scalar>& rCmu, const DeviceBuffer<scalar>& k,
                         const DeviceBuffer<scalar>& eps, DeviceBuffer<scalar>& nut);
void deviceEpsReactionRealizable(const DeviceMesh& dm, const DeviceBuffer<scalar>& eps, const DeviceBuffer<scalar>& k,
                                 const DeviceBuffer<scalar>& magS, scalar nu, scalar C2,
                                 DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source);
// kOmegaSSTLM (Langtry-Menter gamma-ReThetat transition) source-prep, exported for the distributed LM correct.
// Cell-local (gradU is the only halo-coupled input, provided by the caller). ReThetat/gammaInt are the two extra
// transport fields; deviceLMReDiff = sigmaThetat*(nut+nu); Prep fills the semi-implicit sp/su + Fth; GammaEff the
// effective intermittency (feeds the SST k-production modulation); AddReaction folds sp/su into a transport diag/src.
void deviceLMReDiff(const DeviceBuffer<scalar>& nut, scalar nu, DeviceBuffer<scalar>& D);
void deviceLMReThetatPrep(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, scalar nu,
    DeviceBuffer<scalar>& Fth, DeviceBuffer<scalar>& spR, DeviceBuffer<scalar>& suR);
void deviceLMGammaPrep(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, scalar nu,
    DeviceBuffer<scalar>& spG, DeviceBuffer<scalar>& suG);
void deviceLMGammaEff(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, const DeviceBuffer<scalar>& Fth,
    scalar nu, DeviceBuffer<scalar>& gammaIntEff);
void deviceLMAddReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& sp, const DeviceBuffer<scalar>& su,
    DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source);

// Boundary nut per face: wall faces -> nutkWallFunction(k[cell], y, nu); other faces -> nut[cell]. Gives the
// true wall eddy viscosity for the momentum boundary nuEff (vs the cell-value approximation).
void deviceBoundaryNut(const DeviceBoundary& db, const DeviceBuffer<label>& isWall, const DeviceBuffer<scalar>& y,
                       const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& nut, scalar nu, DeviceBuffer<scalar>& nutBnd,
                       const KEpsilonCoeffs& co = {}, scalar atmZ0 = 0.0, bool atmBoundNut = true,   // z0>0 -> atmNutkWallFunction
                       const DeviceBuffer<scalar>* nuFace = nullptr,   // compressible: per-face nu = mu_b/rho_b
                       // kEpsilon: on a 'calculated' nut patch OF carries Cmu*k_b^2/eps_b, not the cell value.
                       const DeviceBuffer<label>*  calcMask = nullptr,
                       const DeviceBuffer<scalar>* kBnd = nullptr,
                       const DeviceBuffer<scalar>* epsBnd = nullptr);

// The full device kEpsilon::correct(): production -> wall functions/override -> eps eqn (with the wall
// setValues constraint) -> k eqn -> correctNut. Updates k/eps/nut in place. Mirrors the CPU correct().
// div(phi,k)/div(phi,epsilon) scheme: upwind by default; limitedLinear when limitedK/limitedEps set (twoByk* =
// 2/max(k_,SMALL) from the scheme coefficient). limitedLinear reduces to upwind at limiter=0 (bit-identical off).
void deviceKEpsilonCorrect(const DeviceMesh& dm, const DeviceWallData& wall, const DeviceBoundary& dbEps,
                           const DeviceBoundary& dbK, const DeviceVectorBoundary& dbU,
                           const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                           DeviceBuffer<scalar>& k, DeviceBuffer<scalar>& eps, DeviceBuffer<scalar>& nut,
                           const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
                           scalar nu, scalar relaxEps, scalar relaxK, scalar tol, bool bounded = false,
                           bool boundedEps = false,   // div(phi,epsilon) `bounded` (bounded = div(phi,k)'s)
                           bool limitedK = false, bool limitedEps = false, scalar twoBykK = 2.0, scalar twoBykEps = 2.0,
                           const KEpsilonCoeffs& co = {}, scalar relTolKE = 0.0, int keCheckEvery = 1,
                           bool linearUpwindK = false, bool linearUpwindEps = false, bool nonOrth = false,
                           bool gsK = false, bool gsEps = false, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr,
                           int nutWall = 0,   // 0=nutk 1=nutUSpalding 2=nutUBlended: near-wall G0 uses the BC-chosen nutw
                           scalar atmZ0 = 0.0, bool atmBoundNut = true,   // z0>0 -> atmNutkWallFunction (rough wall)
                           const ScalarDdt& kDdt = {}, const ScalarDdt& eDdt = {},
                            const DeviceBuffer<scalar>* rho = nullptr,          // compressible: alpha*rho weighting
                            const DeviceBuffer<scalar>* muLam = nullptr,        // compressible: laminar dynamic mu
                            const DeviceBuffer<scalar>* rhoBnd = nullptr,       // compressible: rho at boundary faces
                            const DeviceBuffer<scalar>* nuWallFace = nullptr,  // compressible: nu = mu_b/rho_b per wall face
                            const DeviceBuffer<scalar>* nutBnd = nullptr,      // nut at boundary faces -> patch diffusivity
                            const DeviceBuffer<scalar>* muBnd = nullptr,       // compressible: mu at boundary faces
                            // OF `grad(k)`/`grad(epsilon)` cellLimited coefficient (0 = unlimited); see the
                            // kOmegaSST declaration below for why this is distinct from gradULimitK.
                            scalar gradScalarLimitK = 0.0,
                            // OF `grad(U)` cellLimited coefficient. kEpsilon::correct() builds its
                            // production from fvc::grad(U), which resolves that entry; leaving it
                            // unlimited makes the production term too large wherever the limiter bites.
                            scalar gradULimitK = 0.0,
                            // fvOptions scalarFixedValueConstraint on k / epsilon (OF eqn.setValues).
                            // Applied BEFORE the eps wall function, per kEpsilon.C:266-267.
                            const DeviceBuffer<label>*  fvoKMask = nullptr,
                            const DeviceBuffer<scalar>* fvoKVal  = nullptr,
                            const DeviceBuffer<label>*  fvoEMask = nullptr,
                            const DeviceBuffer<scalar>* fvoEVal  = nullptr);

// Closed device kOmegaSST::correct(): production (raw GbyNu0 + omega-wall G0 override) -> F1/F2/CDkOmega/S2 ->
// omega eqn (loose solve, omega-wall setValues) -> bound -> k eqn (loose solve) -> bound -> correctNut (Bradshaw
// limiter). y = precomputed cell wall distance (cellWallDist). All blocks reuse the C1-6 validated kernels +
// the shared deviceSolveScalarTransport scaffold. Updates k/omega/nut in place. Mirrors kOmegaSSTBase::correct().
void deviceKOmegaSSTCorrect(const DeviceMesh& dm, const DeviceWallData& wall, const DeviceBoundary& dbOmega,
                            const DeviceBoundary& dbK, const DeviceVectorBoundary& dbU,
                            const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                            DeviceBuffer<scalar>& k, DeviceBuffer<scalar>& omega, DeviceBuffer<scalar>& nut,
                            const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
                            scalar nu, scalar relaxOmega, scalar relaxK, scalar tol, bool bounded = false,
                            bool boundedEps = false,   // div(phi,omega) `bounded` (bounded = div(phi,k)'s)
                            bool limitedK = false, bool limitedOmega = false, scalar twoBykK = 2.0, scalar twoBykOmega = 2.0,
                            const KOmegaSSTCoeffs& co = {}, scalar relTolKE = 0.0, int keCheckEvery = 1,
                            bool linearUpwindK = false, bool linearUpwindOmega = false, bool nonOrth = false, scalar gradULimitK = 0.0,
                            bool gsK = false, bool gsEps = false, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr,
                            const scalar* gammaIntEff = nullptr,   // kOmegaSSTLM transition: scales k Pk/epsilonByk
                            int nutWall = 0,   // 0=nutk 1=nutUSpalding 2=nutUBlended: near-wall G0 uses the BC-chosen nutw
                            scalar atmZ0 = 0.0, bool atmBoundNut = true,   // z0>0 -> atmNutkWallFunction (rough wall)
                            const ScalarDdt& kDdt = {}, const ScalarDdt& sDdt = {},   // transient fvm::ddt(k)/ddt(omega)
                            bool des = false,   // kOmegaSSTDDES: DES limiter on the k destruction
                            bool iddes = false,   // kOmegaSSTIDDES: the improved (WMLES) length scale (needs hmax+hwn)
                            const DeviceBuffer<scalar>* hmax = nullptr,   // per-cell maxDeltaxyz (IDDES delta,
                                                                          // null -> DDES cubeRootVol
                            const DeviceBuffer<scalar>* hwn = nullptr,   // per-cell wall-normal spacing (IDDES delta 3rd term)
                            const DeviceBuffer<scalar>* rho = nullptr,   // compressible: rho-weight reactions + diffusivity
                            const DeviceBuffer<scalar>* muLam = nullptr,   // compressible: laminar DYNAMIC viscosity mu [Pa s]
                            const DeviceBuffer<scalar>* nuWallFace = nullptr,   // compressible: nu = mu_b/rho_b per WALL face
                            const DeviceBuffer<scalar>* rhoBnd = nullptr,   // compressible: rho at boundary faces (volumetric flux for divU)
                            const DeviceBuffer<scalar>* nutBnd = nullptr,   // nut at boundary faces -> patch diffusivity
                            const DeviceBuffer<scalar>* muBnd = nullptr,   // compressible: mu at boundary faces
                            // OF `grad(k)`/`grad(omega)` cellLimited coefficient (0 = unlimited). Distinct from
                            // gradULimitK above, which limits grad(U) for the production term; this one limits the
                            // TRANSPORTED scalar's own gradient, which feeds its limitedLinear weight, its
                            // linearUpwind correction and the non-orth laplacian correction.
                            scalar gradScalarLimitK = 0.0,
                            // fvOptions scalarFixedValueConstraint on k / epsilon (OF eqn.setValues).
                            // Applied BEFORE the eps wall function, per kEpsilon.C:266-267.
                            const DeviceBuffer<label>*  fvoKMask = nullptr,
                            const DeviceBuffer<scalar>* fvoKVal  = nullptr,
                            const DeviceBuffer<label>*  fvoEMask = nullptr,
                            const DeviceBuffer<scalar>* fvoEVal  = nullptr,
                            // The case's LES filter width (`delta maxDeltaxyz`); null keeps OF's
                            // cubeRootVol. Must match what the convection scheme uses.
                            const DeviceBuffer<scalar>* lesDelta = nullptr);

// nuWall[i] = nuBnd[wfBndIdx[i]] -- OF nu(patchi) re-indexed from boundary-face into wall-face ordering.
void deviceGatherWallNu(const DeviceBuffer<label>& wfBndIdx, const DeviceBuffer<scalar>& nuBnd,
                        DeviceBuffer<scalar>& nuWall);

// kOmegaSSTLM (Langtry-Menter gamma-ReThetat transition) extra step: after the SST k/omega solve, transport ReThetat
// and gammaInt and update gammaIntEff (fed back into the SST k equation next iteration). See PORTING_KOMEGASSTLM.md.
void deviceKOmegaSSTLMCorrect(const DeviceMesh& dm, const DeviceVectorBoundary& dbU, const DeviceBoundary& dbReThetat,
                              const DeviceBoundary& dbGammaInt, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                              const DeviceBuffer<scalar>& Uz, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
                              const DeviceBuffer<scalar>& nut, const DeviceBuffer<scalar>& y,
                              DeviceBuffer<scalar>& ReThetat, DeviceBuffer<scalar>& gammaInt, DeviceBuffer<scalar>& gammaIntEff,
                              const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd, scalar nu,
                              scalar relax, scalar tol, scalar relTolKE, int keCheckEvery, bool bounded, bool nonOrth,
                              bool gsEps = false, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr,
                              const ScalarDdt& reDdt = {}, const ScalarDdt& giDdt = {},   // transient fvm::ddt(ReThetat)/ddt(gammaInt)
                              // div(phi,ReThetat) / div(phi,gammaInt) scheme, from the case's fvSchemes.
                              bool limitedLinear = false, bool linearUpwind = false);

// Spalart-Allmaras (one-equation): solve the nuTilda transport (div - laplacian(DnuTildaEff) + Sp(destruction) ==
// production + Cb2 grad^2) via the shared scaffold, then nut = nuTilda*fv1. Mirrors SpalartAllmarasBase::correct()
// (incompressible, steady, ft2 off). dbNuTilda = the nuTilda boundary (freestream + fixedValue-0 wall). y = cell
// wall distance. nuTilda/nut updated in place.
void deviceSpalartAllmarasCorrect(const DeviceMesh& dm, const DeviceVectorBoundary& dbU, const DeviceBoundary& dbNuTilda,
                                  const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                                  DeviceBuffer<scalar>& nuTilda, DeviceBuffer<scalar>& nut, const DeviceBuffer<scalar>& y,
                                  const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
                                  scalar nu, scalar relax, scalar tol, bool bounded = false, bool limited = false,
                                  scalar twoByk = 2.0, const SpalartAllmarasCoeffs& co = {}, scalar relTol = 0.0, int checkEvery = 1,
                                  bool linearUpwind = false, bool nonOrth = false,
                                  bool gsK = false, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr,
                                  const ScalarDdt& ntDdt = {},   // transient fvm::ddt(nuTilda)
                                  bool des = false,   // SpalartAllmarasDDES: DES length-scale limiter on the SA destruction
                                  bool iddes = false,   // SpalartAllmarasIDDES: use the improved (WMLES) length scale (needs hmax+hwn)
                                  const DeviceBuffer<scalar>* hmax = nullptr,   // per-cell maxDeltaxyz (IDDES delta); null -> DDES cubeRootVol
                                  const DeviceBuffer<scalar>* hwn = nullptr,   // per-cell wall-normal spacing (IDDES delta 3rd term)
                                  // The LES filter width the case selected (`delta maxDeltaxyz`); nullptr
                                  // keeps OF's default cubeRootVol. Must match what the momentum scheme uses.
                                  const DeviceBuffer<scalar>* lesDelta = nullptr,
                                  // The OUTWARD unit normal of the nearest wall face, packed 3 x nC (OF
                                  // wallDist::n()). Only ZDES2020 shielding reads it; nullptr (or the
                                  // wrong size) leaves the standard DDES fd in place.
                                  const DeviceBuffer<scalar>* wallN = nullptr,
                                  // OF `grad(U)` cellLimited coefficient. SpalartAllmarasBase::correct
                                  // builds Omega and Stilda from fvc::grad(U) (SpalartAllmarasBase.C:461,
                                  // Omega = sqrt(2)*mag(skew(gradU)) at :103), which resolves the case's
                                  // gradSchemes entry. There was no such parameter, so SA's Omega ran
                                  // unlimited on every driver: measured on windAroundBuildingsBox at
                                  // t=400, Omega peaks at 7.14e-01 unlimited against 3.24e-02 limited.
                                  scalar gradULimitK = 0.0);

// Standalone SA correctNut (nut = nuTilda*fv1(nuTilda)) for the solver startup validate().
void deviceNutSA(const DeviceBuffer<scalar>& nuTilda, scalar nu, scalar Cv1, DeviceBuffer<scalar>& nut);
// The same relation on BOUNDARY faces, which is what a `calculated` nut patch carries: OF's
// SpalartAllmaras::correctNut does `nut_ = nuTilda_*fv1`, a GeometricField assignment, so the patch value
// comes from nuTilda's patch value -- not from the file and not from the adjacent cell. nuFace supplies a
// per-face nu (mu/rho) on a compressible mesh; nullptr uses the uniform nu.
void deviceNutSABoundary(const DeviceBuffer<scalar>& nuTildaB, const DeviceBuffer<scalar>* nuFace,
                         scalar nu, scalar Cv1, DeviceBuffer<scalar>& nutB);
// ZDES2020 shielding fd from its eight input fields (unit-test/DES hook; the solver path builds the two
// gradients itself inside deviceSpalartAllmarasCorrect).
void deviceSAZdesFd(int nC, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& gradU,
                    const DeviceBuffer<scalar>& nuTilda, scalar nu,
                    const DeviceBuffer<scalar>& wnx, const DeviceBuffer<scalar>& wny, const DeviceBuffer<scalar>& wnz,
                    const DeviceBuffer<scalar>& gnx, const DeviceBuffer<scalar>& gny, const DeviceBuffer<scalar>& gnz,
                    const DeviceBuffer<scalar>& gox, const DeviceBuffer<scalar>& goy, const DeviceBuffer<scalar>& goz,
                    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& fd);
// SA-DDES length scale dTilda = y - fd*max(0, y - CDES*cubeRootVol(V)); fd = 1 - tanh((8 rd)^3). (Unit-test/DES hook.)
void deviceSADDESdTilda(int nC, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nuTilda, scalar nu,
    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& dTilda);
// SA-IDDES length scale (Shur et al. 2008): dTilda = fdTilde*(1+fe)*y + (1-fdTilde)*CDES*Delta, Delta from hmax+hwn. (Unit-test/DES hook.)
void deviceSAIDDESdTilda(int nC, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& hmax,
    const DeviceBuffer<scalar>& hwn, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nuTilda, scalar nu,
    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& dTilda);
// Spalart-Allmaras source-prep, exported for the distributed SA correct. Cell-local (gradU + grad(nuTilda) supplied
// by the caller). Stilda/Fw/DEff/Reaction reuse the anon kernels; deviceSAReaction ACCUMULATES (+=) -> zero first.
void deviceSAStilda(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nuTilda,
    const DeviceBuffer<scalar>& y, scalar nu, const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& Stilda);
void deviceSAFw(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, const DeviceBuffer<scalar>& Stilda,
    const DeviceBuffer<scalar>& y, const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& fw);
void deviceSAMagSqr(const DeviceMesh& dm, const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& out);
void deviceSADEff(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, scalar nu, scalar sigmaNut, DeviceBuffer<scalar>& D);
void deviceSAReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, const DeviceBuffer<scalar>& Stilda,
    const DeviceBuffer<scalar>& fw, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& gradNt2,
    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src);

// nutUSpaldingWallFunction boundary nut: wall faces -> Newton uTau from Spalding; other faces -> adjacent cell nut.
// y / isWall are per boundary face (nearWallDist y, wall mask); nutBnd is in/out (warm-started seed). nu+nutBnd at
// walls gives the SA wall shear (deep wall-function meshes, y+ >> 1).
// nutUWallFunction (STEPWISE blender, OF's default): nut from the log-law yPlus fixed-point iteration.
void deviceBoundaryNutU(const DeviceVectorBoundary& dbU, const DeviceBuffer<label>& isWall,
                        const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& Ux,
                        const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                        const DeviceBuffer<scalar>& nutCell, scalar nu, scalar kappa, scalar E,
                        DeviceBuffer<scalar>& nutBnd,
                        const DeviceBuffer<scalar>* nuFace = nullptr,
    const DeviceBuffer<scalar>* nutFile = nullptr);

void deviceBoundaryNutSpalding(const DeviceVectorBoundary& dbU, const DeviceBuffer<label>& isWall,
                               const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& Ux,
                               const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                               const DeviceBuffer<scalar>& nutCell, scalar nu, const SpalartAllmarasCoeffs& co,
                               DeviceBuffer<scalar>& nutBnd,
                               const DeviceBuffer<scalar>* nuFace = nullptr,
    const DeviceBuffer<scalar>* nutFile = nullptr);   // compressible: per-face nu = mu_b/rho_b

// nutUBlendedWallFunction wall nut (velocity-based binomial n=4 blend); kappa/E explicit (any RAS model).
void deviceBoundaryNutBlended(const DeviceVectorBoundary& dbU, const DeviceBuffer<label>& isWall,
                              const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& Ux,
                              const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                              const DeviceBuffer<scalar>& nutCell, scalar nu, scalar kappa, scalar E,
                              DeviceBuffer<scalar>& nutBnd,
                              const DeviceBuffer<scalar>* nuFace = nullptr,
    const DeviceBuffer<scalar>* nutFile = nullptr);   // compressible: per-face nu = mu_b/rho_b

} // namespace brae
