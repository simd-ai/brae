// CUDA implementation -- see rhoUEqn.cuh for the provenance and the contract with the _cpp reference.
//
// The stage order here is NOT the reference's stage order, and the reason is one word: ASSIGN.
// deviceDivDevReff's tensorDivKernel ends with `dX[c] = d[0]` -- it OVERWRITES source[0..2] rather than
// accumulating into them (device_divdevreff.cu). rhoUEqn_cpp.cu applies linearUpwind's correction first,
// as OpenFOAM does inside fvm::div, and addDivDevReff last; transcribing that order into CUDA gives a
// matrix that is exact in diag/upper/lower and silently missing every deferred source term, with nothing
// thrown. So every source contribution is placed AFTER the divDevRhoReff call. Their order among
// themselves is free -- they are all additions into the same array -- but being after the assignment and
// before relax(), which reads the source, is not.
#include "rhoUEqn.cuh"
#include "device_blas.cuh"
#include "device_divdevreff.cuh"
#include "device_simple.cuh"
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace brae {
namespace gpu {
namespace rhoSimple {

namespace {

// resize() does NOT zero: DevicePool::take hands back a RECYCLED block, the same contract as cudaMalloc
// (device_buffer.cuh:25 -- "Returned memory is NOT zeroed ... cf always writes a buffer before reading
// it"). Any buffer this file ACCUMULATES into rather than assigns must therefore be memset first. The
// incompressible twin's limitedLinear branch axpy's magSqr(U) into an un-zeroed buffer and so reads
// garbage from the second assembly onward, once a same-size block has been through the pool.
void zeroBuffer(
    DeviceBuffer<scalar>& b,
    int n)
{
    b.resize(static_cast<std::size_t>(n));
    if (n <= 0) return;
    cudaCheck(
        cudaMemsetAsync(
            b.data(),
            0,
            static_cast<std::size_t>(n) * sizeof(scalar),
            cudaStreamPerThread),
        "rhoUEqn zeroBuffer");
}


void refuseUnsupported(
    const DeviceMesh&       dm,
    const RhoMomentumInput& in)
{
    // Same triggers, same citations and same wording as rhoUEqn_cpp.cu's refuseUnsupported. The gates
    // assert the refusals as well as the numbers, and a component that is out of scope has to say so:
    // brae has shipped a solver that read MRFProperties, ignored it, converged, and reported nothing.
    //
    // UEqn.H reaches MRF TWICE and only ONE of the two is this module's. MRF.correctBoundaryVelocity(U)
    // (UEqn.H:3) writes Omega x (Cf - origin) onto U's boundary values on the included patches; it mutates
    // the FIELD, not the matrix, and it has to run before buildDeviceVectorBoundary snapshots refValue --
    // which is where simpleFoamV2.cu:652-674 puts it, on the host, via cpu::MRF::correctBoundaryVelocity.
    // DeviceMRFZone carries Omega and the included-face mask but neither the origin nor Cf, so it could not
    // be applied here even if it belonged here. It is a CALLER precondition, stated in rhoUEqn.cuh's
    // pre-assembly sequence; what this assembly applies is MRF.DDt(rho, U) (UEqn.H:8).
    if (in.hasMRF && !in.mrf)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): the case declares MRF. UEqn.H reaches it twice: "
            "MRF.correctBoundaryVelocity(U) (UEqn.H:3), which the CALLER must have applied to U before the "
            "device boundary was built, and MRF.DDt(rho, U) (UEqn.H:8), which this assembly applies. No "
            "zones were supplied to this assembly; refusing rather than silently solving a different "
            "equation.");
    }
    if (in.hasFvOptions)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): the case declares an fvOption this port does not implement"
            + (in.fvOptionUnsupported.empty() ? std::string()
                                              : std::string(" -- '") + in.fvOptionUnsupported + "'")
            + ". UEqn.H applies fvOptions(rho, U), fvOptions.constrain(UEqn) and fvOptions.correct(U) "
              "(rhoSimpleFoam/UEqn.H:11,17,21). explicitPorositySource (fixedCoeff) IS implemented; "
              "refusing rather than silently solving a different equation.");
    }

    // phi, with the same shape of check the viscosity already gets and for the same reason. Both default
    // to nullptr, and forming `*in.phiInt` on a null pointer is UB before any kernel runs. The SHORT case
    // is worse than the null one: divFaceKernel launches nBlocks(dm.nInternalFaces) threads and reads
    // phi[f] over that whole range whatever phiInt.size() is (device_fvm.cu:538), bcDivKernel launches
    // nBlocks(db.n) and reads phiB[i] the same way (device_boundary_assembly.cu:214), and deviceAxpy takes
    // its trip count from the SOURCE buffer without resizing the destination (blas1.cu:164-169), so a
    // length mismatch propagates as an out-of-bounds WRITE. What comes out is a matrix whose coefficients
    // look plausible. The mass flux is the one field this solver's convection operator is entirely made of.
    if (!in.phiInt || !in.phiBnd)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): fvm::div(phi, U) needs the MASS flux at internal AND boundary faces "
            "(compressibleCreatePhi.H -- phi is rho*(U & Sf), kg/s, not the volumetric flux). phiInt or "
            "phiBnd was not supplied.");
    }
    if (static_cast<int>(in.phiInt->size()) != dm.nInternalFaces)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): phiInt has the wrong internal-face count. fvm::div reads one flux "
            "per internal face; a short array is read past its end rather than truncated.");
    }
    if (static_cast<int>(in.phiBnd->size()) != dm.nBndFaces)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): phiBnd has the wrong boundary-face count. The boundary div "
            "coefficients read one flux per non-coupled boundary face, in the DeviceBoundary's own order.");
    }

    // Coupled patches -- cyclic, cyclicAMI, cyclicACMI, processor. buildDeviceMesh SKIPS them in the
    // boundary gather (device_mesh.cuh:110-123, isCoupledInterfaceType) and deliberately keeps them OUT of
    // the internal-face LDU (device_mesh.cuh:34-38), because the periodic coupling is a SEPARATE interface.
    // This assembly supplies no interface to any stage, so on such a mesh it loses the same faces four
    // times over: from gradU and from the tensor divergence inside deviceDivDevReff (its cyc/ami hooks
    // exist for exactly that -- device_divdevreff.cuh:35-37, "else x-invariance drifts on periodic
    // meshes"), from the convection/laplacian off-diagonals, from the corrected laplacian's deferred source
    // (device_cyclic.cuh has the counterpart), and from deviceRelaxDiag's diagonal-dominance sum, which
    // then under-counts sumOff in every interface cell and carries the error into rAU and HbyA.
    //
    // DeviceMesh cannot report this -- it does not carry the interface list -- so the DRIVER says it, the
    // same way it already says hasMRF and hasFvOptions. A comment is not a refusal.
    if (in.hasCoupledPatches)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): the mesh has a coupled patch (cyclic / cyclicAMI / cyclicACMI / "
            "processor). Those faces are not in the DeviceMesh LDU or the boundary gather, and this "
            "assembly passes no interface to divDevRhoReff, to the corrected-laplacian source or to the "
            "relaxation's diagonal-dominance sum, so all four would silently drop them. Refusing rather "
            "than assembling a momentum matrix that is missing the interface.");
    }

    const bool haveMu  = in.muEffCell && in.muEffBndFace;
    const bool haveRho = in.rhoCell && in.rhoBndFace && in.nuEffCell && in.nuEffBndFace;
    if (!haveMu && !haveRho)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): divDevRhoReff needs the DYNAMIC viscosity rho*nuEff "
            "(linearViscousStress.C:107-117). Supply either muEffCell/muEffBndFace directly, or "
            "rhoCell/rhoBndFace with nuEffCell/nuEffBndFace so it can be formed. Neither was given.");
    }

    // explicitPorositySource, DarcyForchheimer branch, on a FORCE-dimensioned equation. OpenFOAM's
    // DarcyForchheimerTemplates.C:52-53 builds Cd = mu[celli]*D + (rho[celli]*mag(U))*F from the per-cell
    // LAMINAR dynamic viscosity and the per-cell density. deviceFvoPorosityDiag/Source take ONE scalar
    // viscosity and no rho field, and RhoMomentumInput carries neither mu_lam nor a kinematic nu, so the
    // correct resistance cannot be formed here at all.
    //
    // The host reference passes nu = 0.0 into the shared addSup (rhoUEqn_cpp.cu:223), which on this
    // branch drops the whole viscous Darcy term AND the rho on the Forchheimer term. That is a known
    // defect, unreached by angledDuct (fixedCoeff) and therefore covered by no gate. Reproducing it in
    // CUDA would make two implementations of a wrong number instead of one; refusing names it.
    if (in.porosity && in.porosity->active && !in.porosity->fixed)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): explicitPorositySource with DarcyForchheimer. On a force-dimensioned "
            "momentum equation OpenFOAM builds Cd = mu*D + (rho*mag(U))*F from the per-cell LAMINAR dynamic "
            "viscosity and the per-cell density (DarcyForchheimerTemplates.C:52-53); this assembly is given "
            "neither, so the resistance cannot be formed. fixedCoeff IS implemented -- it takes rhoRef "
            "from the dictionary, which is the branch UEqn.dimensions() == dimForce selects "
            "(fixedCoeff.C:202-207) -- and is the model the compressible tutorials use. Refusing rather "
            "than silently solving the kinematic form with nu = 0.");
    }
}


// mu_eff = rho*nu_eff at cells and at boundary faces -- the device twin of the reference's
// dynamicViscosity/dynamicViscosityBoundary, and THE compressible difference. The incompressible lineage
// carries the kinematic nu_eff here; assembling this equation with it reads 6.2e-01 against OpenFOAM's own
// matrix where the dynamic form reads 6.1e-15 (the host gate's measurement, rhoUEqn.cuh:32).
//
// muEffCell/muEffBndFace, when supplied, are used verbatim and rho/nuEff are ignored: that is the gate's
// injection path for OpenFOAM's own turbulence->muEff(), so the ASSEMBLY can be measured without a ported
// compressible closure in the way.
void resolveDynamicViscosity(
    const DeviceMesh&           dm,
    const RhoMomentumInput&     in,
    DeviceBuffer<scalar>&       muCellOwned,
    DeviceBuffer<scalar>&       muBndOwned,
    const DeviceBuffer<scalar>*& muCell,
    const DeviceBuffer<scalar>*& muBnd)
{
    if (in.muEffCell && in.muEffBndFace)
    {
        muCell = in.muEffCell;
        muBnd = in.muEffBndFace;
    }
    else
    {
        if (in.rhoCell->size() != in.nuEffCell->size())
        {
            throw std::runtime_error(
                "rhoSimpleFoam UEqn(cuda): rhoCell and nuEffCell differ in length.");
        }
        if (in.rhoBndFace->size() != in.nuEffBndFace->size())
        {
            throw std::runtime_error(
                "rhoSimpleFoam UEqn(cuda): rhoBndFace and nuEffBndFace differ in length.");
        }
        deviceHadamard(muCellOwned, *in.rhoCell, *in.nuEffCell);
        deviceHadamard(muBndOwned, *in.rhoBndFace, *in.nuEffBndFace);
        muCell = &muCellOwned;
        muBnd = &muBndOwned;
    }

    // The host's effectiveFaceViscosity SKIPS its boundary overwrite when a patch is missing or its length
    // disagrees, leaving fvc::interpolate's default -- the OWNER CELL value -- with no warning, which is
    // exactly the wall-shear defect the boundary_mu_eff gate exists to catch. The dev2 term's own fallback
    // is different again (0.0, not the cell value). Neither silent fallback is reproduced: a wrong-length
    // array is a refusal here.
    if (static_cast<int>(muCell->size()) != dm.nCells)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): the dynamic viscosity has the wrong cell count. divDevRhoReff needs "
            "mu_eff at every cell (linearViscousStress.C:107-117).");
    }
    if (static_cast<int>(muBnd->size()) != dm.nBndFaces)
    {
        throw std::runtime_error(
            "rhoSimpleFoam UEqn(cuda): the dynamic viscosity has the wrong boundary-face count. The face "
            "viscosity is the PATCH value, not the owner cell's -- on a wall carrying a nut wall function "
            "the two differ by the whole of nut_wall, and taking the cell value under-predicts wall shear "
            "silently.");
    }
}

} // namespace


void assembleUEqn(
    MomentumMatrix&             M,
    const DeviceMesh&           dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const RhoMomentumInput&     in)
{
    refuseUnsupported(dm, in);

    // mu_eff, before anything reads it.
    DeviceBuffer<scalar> muCellOwned, muBndOwned;
    const DeviceBuffer<scalar>* muCell = nullptr;
    const DeviceBuffer<scalar>* muBnd = nullptr;
    resolveDynamicViscosity(dm, in, muCellOwned, muBndOwned, muCell, muBnd);

    // The INTERNAL-face viscosity, formed by interpolating the PRODUCT. Interpolating rho and nu_eff
    // separately and multiplying is a different number -- interp(rho)*interp(nu) != interp(rho*nu) by the
    // interpolation error of the product -- and OpenFOAM interpolates the finished volScalarField
    // (laplacianScheme.C:87-94 hands fvm::laplacian the interpolated gamma). The BOUNDARY faces are not
    // interpolated at all: they take muBnd verbatim, which is effectiveFaceViscosity's overwrite.
    DeviceBuffer<scalar> muFace;
    deviceInterpolate(dm, *muCell, muFace);

    // fvm::div(phi, U), with phi the MASS flux (kg/s). The operator is the incompressible one unchanged:
    // what carries the density is the field it is handed, not the discretisation. This ASSIGNS
    // diag/upper/lower, so it must come first -- deviceAxpy does not resize its destination, and every
    // accumulation below would otherwise write out of bounds.
    switch (in.scheme)
    {
        case cpu::rhoSimple::DivScheme::limitedLinearV:
        {
            // The kernel indexes U[0..2], so it wants CONTIGUOUS 3-arrays rather than an array of
            // pointers.
            //
            // THE LIMITER'S GRADIENT IS THE CASE'S grad(U) ENTRY, not a plain Gauss gradient.
            // LimitedScheme::calcLimiter is `tlPhi = LimitFunc<Type>()(phi); tgradc(fvc::grad(lPhi))`
            // (LimitedScheme.C:51-55), and limitedLinearV is instantiated through
            // makeLimitedVSurfaceInterpolationScheme as NVDVTVDV + limitFuncs::null
            // (LimitedScheme.H:202-203, limitedLinear.C:38-41). limitFuncs::null returns phi ITSELF
            // (LimitFuncs.H, `return phi;`), so lPhi.name() is still "U" and fvc::grad(lPhi) resolves
            // "grad(" + vf.name() + ')' == grad(U) in gradSchemes (fvcGrad.C:149). A case writing
            // `grad(U) cellLimited Gauss linear 1` therefore builds these weights from the CELL-LIMITED
            // gradient. Running the unlimited one under that name is the same class of defect as
            // `Gauss linear orthogonal` where the case said `corrected`: a different discretisation, and
            // it does not vanish at convergence because it changes the IMPLICIT face weights.
            //
            // This is applied exactly as the two linearUpwind branches below apply it, per component.
            // The sibling limitedLinear branch is genuinely different and stays unlimited: limitFuncs::
            // magSqr RENAMES the field to magSqr(U), which no case names, so it falls back to gradSchemes
            // `default`.
            //
            // DIVERGENCE FROM THE HOST REFERENCE, deliberate and recorded in PORT.md's open findings:
            // rhoUEqn_cpp.cu:34-37 still builds this limiter from fvc::gaussGrad, i.e. unlimited. Until
            // that is corrected the two disagree on a case that is both limitedLinearV and cellLimited;
            // OpenFOAM's source is the authority for which of them is right.
            const DeviceBuffer<scalar>* Usrc[3] = {&Ux, &Uy, &Uz};
            DeviceBuffer<scalar> Uarr[3], gx[3], gy[3], gz[3], ub;
            for (int k = 0; k < 3; ++k)
            {
                deviceCopy(Uarr[k], *Usrc[k]);
                deviceBCValue(dbU.comp[k], *Usrc[k], ub);
                deviceGaussGrad(dm, *Usrc[k], ub, gx[k], gy[k], gz[k]);
                if (in.gradULimitK > 0.0)
                {
                    deviceCellLimitGrad(dm, *Usrc[k], ub, gx[k], gy[k], gz[k], in.gradULimitK);
                }
            }
            deviceDivLimitedVCoeffs(
                dm,
                *in.phiInt,
                Uarr,
                gx,
                gy,
                gz,
                2.0 / std::fmax(in.schemeCoeff, 1e-15),
                M.diag,
                M.upper,
                M.lower);
            break;
        }
        case cpu::rhoSimple::DivScheme::limitedLinear:
        {
            // limitedLinear on a VECTOR limits on the SCALAR magSqr(U): LimitedScheme.H instantiates it as
            // NVDTVD + limitFuncs::magSqr, so it is neither per-component nor the V form. Reusing the
            // limitedLinearV kernel here would be a different scheme.
            DeviceBuffer<scalar> mag2, m2b, t, ub, gx, gy, gz;
            const DeviceBuffer<scalar>* U3[3] = {&Ux, &Uy, &Uz};
            zeroBuffer(mag2, dm.nCells);
            for (int k = 0; k < 3; ++k)
            {
                deviceHadamard(t, *U3[k], *U3[k]);
                deviceAxpy(1.0, t, mag2);
            }
            zeroBuffer(m2b, dm.nBndFaces);
            for (int k = 0; k < 3; ++k)
            {
                deviceBCValue(dbU.comp[k], *U3[k], ub);
                deviceHadamard(t, ub, ub);
                deviceAxpy(1.0, t, m2b);
            }
            deviceGaussGrad(dm, mag2, m2b, gx, gy, gz);
            deviceDivLimitedCoeffs(
                dm,
                *in.phiInt,
                mag2,
                gx,
                gy,
                gz,
                2.0 / std::fmax(in.schemeCoeff, 1e-15),
                M.diag,
                M.upper,
                M.lower);
            break;
        }
        case cpu::rhoSimple::DivScheme::LUST:
        {
            // weights = 0.75*linear + 0.25*upwind. The coefficients are LINEAR in the weights
            // (lower = -w*phi, upper = lower + phi, then negSumDiag), so blending the two coefficient sets
            // is exact rather than an approximation of a blended-weight kernel. LUST is TWO overrides: this
            // one, and 0.25 of linearUpwind's deferred correction below.
            DeviceBuffer<scalar> cD, cU, cL, uD, uU, uL;
            deviceDivCentralCoeffs(dm, *in.phiInt, cD, cU, cL);
            deviceDivUpwindCoeffs(dm, *in.phiInt, uD, uU, uL);
            deviceCopy(M.diag, cD);
            deviceScale(M.diag, 0.75);
            deviceAxpy(0.25, uD, M.diag);
            deviceCopy(M.upper, cU);
            deviceScale(M.upper, 0.75);
            deviceAxpy(0.25, uU, M.upper);
            deviceCopy(M.lower, cL);
            deviceScale(M.lower, 0.75);
            deviceAxpy(0.25, uL, M.lower);
            break;
        }
        default:
            // upwind, linearUpwind and linearUpwindV share the upwind weights: linearUpwind DERIVES from
            // upwind, its whole effect being the deferred correction.
            deviceDivUpwindCoeffs(dm, *in.phiInt, M.diag, M.upper, M.lower);
            break;
    }

    // - fvm::laplacian(muEff, U), the implicit half of divDevRhoReff. OpenFOAM writes
    // `- fvm::laplacian(alpha*rho*nuEff, U)` inside divDevRhoReff and UEqn.H ADDS divDevRhoReff, so the
    // laplacian enters with coefficient -1 -- and the boundary coefficients below are negated too, because
    // fvMatrix::negate() negates internalCoeffs_ and boundaryCoeffs_ as well as the LDU (fvMatrix.C:1594).
    // Negating only diag/upper/lower leaves the interior exact and every patch-adjacent cell wrong, which
    // reads as a boundary-condition bug rather than a sign bug.
    //
    // correctedLaplacian selects nonOrthDeltaCoeffs for the implicit coefficient itself, not just the
    // deferred source. Doing only one of the two halves is the "Gauss linear orthogonal where the case said
    // corrected" defect this port has already paid for on the energy and pressure equations (PORT.md:394).
    {
        DeviceBuffer<scalar> lD, lU, lL;
        deviceLaplacianCoeffs(dm, muFace, lD, lU, lL, in.correctedLaplacian);
        deviceAxpy(-1.0, lD, M.diag);
        deviceAxpy(-1.0, lU, M.upper);
        deviceAxpy(-1.0, lL, M.lower);
    }

    // Boundary coefficients, per component. A vector boundary is three scalar boundaries: the div and
    // laplacian internalCoeffs are isotropic and only the refValue-dependent boundaryCoeffs differ by
    // component -- except on slip/symmetry, where symMask makes the valueFraction per-component (|n_k|),
    // which is exactly why relax() below needs cmptMaxMag and cmptMin over the three.
    // The laplacian half takes muBnd, the PATCH face viscosity, not the owner cell's.
    for (int k = 0; k < 3; ++k)
    {
        deviceBCDivCoeffs(dbU.comp[k], *in.phiBnd, M.iC[k], M.bC[k]);
        DeviceBuffer<scalar> lIC, lBC;
        deviceBCLaplacianCoeffsFace(dbU.comp[k], *muBnd, lIC, lBC);
        deviceAxpy(-1.0, lIC, M.iC[k]);
        deviceAxpy(-1.0, lBC, M.bC[k]);
    }

    // `bounded Gauss <scheme>`: - fvm::Sp(fvc::div(phi), U), a diagonal-only term
    // (boundedConvectionScheme.C:84-87). BEFORE relax and BEFORE the porosity, as OpenFOAM has it:
    // anything reaching M.diag after deviceRelaxDiag misses the diagonal-dominance clamp and misses rAU,
    // because MomentumMatrix::view switches to relaxedDiag once `relaxed` is set.
    // deviceDiv returns the PER-VOLUME divergence, so the extensive contribution carries V explicitly. On
    // this solver div(phi) is the MASS imbalance, so the term vanishes as the continuity error does.
    if (in.bounded)
    {
        DeviceBuffer<scalar> divPhi, t;
        deviceDiv(dm, *in.phiInt, *in.phiBnd, divPhi);
        deviceHadamard(t, divPhi, dm.V);
        deviceAxpy(-1.0, t, M.diag);
    }

    // The explicit half of divDevRhoReff: -fvc::div(muEff*dev2(T(grad U))).
    //
    // NO SIGN AND NO VOLUME AT THIS CALL SITE. The kernel returns the EXTENSIVE V*fvc::div(sigma) with
    // sigma = muEff*dev2(T(gradU)), and that IS the momentum source. The host reference reaches the same
    // number through two visible flips that cancel: divDevReffExplicit returns -fvc::div(...) per volume,
    // then `source -= expl*V`. Transcribing the host's two flips on top of an already-signed kernel output
    // inverts the stress term.
    //
    // gradULimitK IS forwarded. OF's fvc::grad(U) inside divDevRhoReff resolves the NAMED gradSchemes
    // entry, so `grad(U) cellLimited Gauss linear 1` limits this gradient too. The incompressible twin
    // leaves it defaulted and is measurably worse for it: on oscillatingInletACMI2D the difference from
    // OpenFOAM is 4.7e-04 unlimited against 6.7e-08 limited at nu = 1e-3, a factor of 7000
    // (device_divdevreff.cuh:44-52). The whole term carries a factor of the viscosity, so on this lineage
    // -- where it is mu_eff = rho*nu_eff -- the error is larger still.
    //
    // cyc/ami/proc are nullptr: this path is single-GPU and single-block-mesh. A cyclic, cyclicAMI or
    // processor patch would contribute nothing to gradU and nothing to the tensor divergence, silently.
    // Nothing here can detect one -- DeviceMesh does not carry the interface list -- so it is stated
    // rather than guarded.
    deviceDivDevReff(
        dm,
        dbU,
        Ux,
        Uy,
        Uz,
        *muCell,
        *muBnd,
        M.source[0],
        M.source[1],
        M.source[2],
        nullptr,
        nullptr,
        nullptr,
        in.gradULimitK);

    // The explicit half of `corrected`: the non-orthogonal deferred source. AFTER divDevRhoReff, which
    // ASSIGNS the source -- added before, it compiles, runs, and is discarded.
    //
    // SIGN: deviceLaplacianCorr returns the LAPLACIAN's own source correction, -V*fvc::div(faceFluxCorr).
    // divDevRhoReff carries MINUS the laplacian (fvMatrix::negate() negates source_ too), so what reaches
    // the momentum source is +V*fvc::div(...) and the axpy factor is -1. Getting this backwards made the
    // corrected momentum WORSE than the uncorrected one while the pressure improved -- U against real
    // OpenFOAM on shearedChannel, 1.69e-01 corrected vs 8.47e-02 uncorrected. That asymmetry is what
    // identified it, on the host side, by measurement.
    //
    // The gradient is a plain UNLIMITED Gauss gradient, deliberately different from the dev2 term's above,
    // even when gradULimitK > 0. That is OpenFOAM's own behaviour: correctedSnGrad<Type>::correction calls
    // fullGradCorrection per COMPONENT, which resolves `grad(U.component(0))` in gradSchemes; no case names
    // a component entry, so it falls back to `default` -- plain Gauss linear on aerofoilNACA0012 while
    // grad(U) itself is `cellLimited Gauss linear 1`. Making the two agree is tidier and is a different
    // discretisation (linearViscousStress_cpp.cu:103-110).
    if (in.correctedLaplacian)
    {
        const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
        DeviceBuffer<scalar> gxc[3], gyc[3], gzc[3];
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> ub;
            deviceBCValue(dbU.comp[k], *U[k], ub);
            deviceGaussGrad(dm, *U[k], ub, gxc[k], gyc[k], gzc[k]);
        }
        if (in.snGradLimitCoeff > 0.0)
        {
            // `limited <k> corrected`. OF's limitedSnGrad takes mag() of the WHOLE snGrad and of the WHOLE
            // correction, so all three components share ONE per-face limiter -- which is why this cannot
            // live inside the per-component loop above. Limiting each component on its own is a different
            // scheme, measured at 0.6% on airFoil2D.
            DeviceBuffer<scalar> ffc[3];
            deviceLaplacianCorrFluxLimitedVec(
                dm,
                muFace,
                *U[0],
                *U[1],
                *U[2],
                gxc,
                gyc,
                gzc,
                in.snGradLimitCoeff,
                ffc);
            for (int k = 0; k < 3; ++k)
            {
                DeviceBuffer<scalar> lc;
                deviceFaceDivSource(dm, ffc[k], lc);
                deviceAxpy(-1.0, lc, M.source[k]);
            }
        }
        else
        {
            // snGradLimitCoeff == 0 means UNCAPPED, which is what `corrected` means. OF's k = 0 would be
            // `uncorrected`, and that is expressed by correctedLaplacian == false instead.
            for (int k = 0; k < 3; ++k)
            {
                DeviceBuffer<scalar> lc;
                deviceLaplacianCorr(dm, muFace, gxc[k], gyc[k], gzc[k], lc);
                deviceAxpy(-1.0, lc, M.source[k]);
            }
        }
    }

    // linearUpwindV's deferred correction: a DIFFERENT correction from linearUpwind's, not a scaled one
    // (its limiter couples the three components per face), so it branches before the factor below.
    // SUBTRACTED, because OpenFOAM applies it as `fvm += fvc::surfaceIntegrate(...)` inside fvm::div and
    // fvMatrix::operator+=(DimensionedField) is `source() -= V*su` (fvMatrix.C:1855-1862); the correction
    // helper already returns the extensive per-cell face sum, so no volume factor appears here.
    if (in.scheme == cpu::rhoSimple::DivScheme::linearUpwindV)
    {
        const DeviceBuffer<scalar>* Usrc[3] = {&Ux, &Uy, &Uz};
        DeviceBuffer<scalar> gx[3], gy[3], gz[3], ub, cx, cy, cz;
        for (int k = 0; k < 3; ++k)
        {
            deviceBCValue(dbU.comp[k], *Usrc[k], ub);
            deviceGaussGrad(dm, *Usrc[k], ub, gx[k], gy[k], gz[k]);
            if (in.gradULimitK > 0.0)
            {
                deviceCellLimitGrad(dm, *Usrc[k], ub, gx[k], gy[k], gz[k], in.gradULimitK);
            }
        }
        deviceLinearUpwindVCorr(dm, *in.phiInt, gx, gy, gz, Ux, Uy, Uz, cx, cy, cz);
        const DeviceBuffer<scalar>* cc[3] = {&cx, &cy, &cz};
        for (int k = 0; k < 3; ++k)
        {
            deviceAxpy(-1.0, *cc[k], M.source[k]);
        }
    }

    // How much of linearUpwind's deferred correction this scheme carries: 1 for linearUpwind, 0.25 for
    // LUST (LUST.H overrides correction() as 0.25*linearUpwind::correction), 0 otherwise. A port that took
    // LUST's weights without this factor, or this factor without the weights, is running a third scheme.
    const scalar corrFac =
        (in.scheme == cpu::rhoSimple::DivScheme::linearUpwind || in.linearUpwind) ? 1.0
      : (in.scheme == cpu::rhoSimple::DivScheme::LUST)                            ? 0.25
      :                                                                             0.0;
    if (corrFac != 0.0)
    {
        const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> ub, gx, gy, gz, lu;
            deviceBCValue(dbU.comp[k], *U[k], ub);
            deviceGaussGrad(dm, *U[k], ub, gx, gy, gz);
            // `linearUpwind <name>`, where <name> resolves to `cellLimited Gauss linear <k>`. This
            // correction does NOT vanish at convergence, so an unlimited gradient under a limited name is a
            // different equation, not a transient difference.
            if (in.gradULimitK > 0.0)
            {
                deviceCellLimitGrad(dm, *U[k], ub, gx, gy, gz, in.gradULimitK);
            }
            deviceLinearUpwindCorr(dm, *in.phiInt, gx, gy, gz, lu);
            deviceAxpy(-corrFac, lu, M.source[k]);
        }
    }

    // + MRF.DDt(rho, U), UEqn.H:8. MRFZoneList::DDt(rho,U) IS rho*DDt(U) (MRFZoneList.C:210-217): the
    // Coriolis acceleration is formed first and then rho-weighted per cell.
    //
    // THIS IS THE SECOND OF UEqn.H'S TWO MRF CALLS. The first, MRF.correctBoundaryVelocity(U) at UEqn.H:3,
    // is NOT applied here and must not be: it overwrites U's boundary values on the included patches with
    // the frame velocity Omega x (Cf - origin) (MRFZone.C:499-526), which has to be in the field the
    // DeviceVectorBoundary is BUILT FROM -- once dbU exists the snapshot is taken. brae puts it in the
    // driver for that reason (cpu::MRF::correctBoundaryVelocity, simpleFoamV2.cu:652-674), and
    // DeviceMRFZone carries neither origin nor Cf, so it could not be applied from here in any case. It is
    // stated as a caller precondition in rhoUEqn.cuh's pre-assembly sequence; assembling against a stale
    // rotating-wall value is a wrong equation, so the precondition is load-bearing, not advisory.
    //
    // THE rho IS THE COMPRESSIBLE DIFFERENCE HERE TOO. deviceMrfCoriolisZone hardcodes
    // `src[c] -= V[c]*(Omega x U)_k` with no density, so the incompressible call cannot be reused verbatim
    // -- it would lose exactly a factor of rho. The kernel IS reused, into a zeroed accumulator that is
    // then multiplied by rho: acc already carries the volume and the minus sign, and rho is a cell field,
    // so rho*(a*V) == (rho*a)*V and weighting afterwards is a rearrangement rather than a different term.
    if (in.mrf && !in.mrf->empty())
    {
        if (!in.rhoCell)
        {
            throw std::runtime_error(
                "rhoSimpleFoam UEqn(cuda): MRF.DDt(rho, U) needs rho itself (MRFZoneList.C -- DDt(rho,U) is "
                "rho*DDt(U)); only a pre-formed muEff was supplied, from which rho cannot be recovered.");
        }
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> acc, t;
            zeroBuffer(acc, dm.nCells);
            deviceMrfCoriolisZone(*in.mrf, dm.V, Ux, Uy, Uz, k, acc);
            deviceHadamard(t, acc, *in.rhoCell);
            deviceAxpy(1.0, t, M.source[k]);
        }
    }

    // == fvOptions(rho, U): explicitPorositySource. BEFORE relax, as UEqn.H has it -- the diagonal takes
    // the isotropic part implicitly and the source the anisotropic remainder, which is what keeps a large
    // resistance stable, and it has to be inside the dominance clamp and inside rAU.
    //
    // The equation is FORCE-dimensioned, which is what selects fixedCoeff's rhoRef branch over the
    // kinematic one (fixedCoeff.C:202-207). DevicePorosity carries that rhoRef and porFixedDiagKernel /
    // porFixedSrcKernel apply it, so the caller must have populated it from the dictionary; with rhoRef
    // left at its 1.0 default the resistance is under by exactly that factor. The `nu` argument is dead on
    // the fixed branch and the other branch is refused above, so 0.0 is passed rather than a value that
    // would look meaningful.
    //
    // SIGN: two negations cancel. explicitPorositySource does `eqn -= porosityEqn` and UEqn.H writes
    // `UEqn == fvOptions(...)`, i.e. UEqn - optionsEqn, so the net on the momentum matrix is fixedCoeff's
    // own `Udiag += V*isoCd; Usource -= V*((Cd - I*isoCd) & U)` -- which is what these two helpers apply.
    // Applying one negation and not the other turns the resistance into a source.
    if (in.porosity && in.porosity->active)
    {
        deviceFvoPorosityDiag(*in.porosity, 0.0, dm.V, Ux, Uy, Uz, M.diag);
        for (int k = 0; k < 3; ++k)
        {
            deviceFvoPorositySource(*in.porosity, k, 0.0, dm.V, Ux, Uy, Uz, M.source[k]);
        }
    }

    // UEqn.relax(). LAST, after every matrix and source contribution and before -fvc::grad(p), which the
    // driver adds to a COPY: `solve(UEqn == -fvc::grad(p))` builds a new equation from the ALREADY-RELAXED
    // matrix, and pEqn.H needs A() and H() of the relaxed, pressure-free one.
    //
    // OpenFOAM's fvMatrix::relax is ASYMMETRIC: it ADDS cmptMax(cmptMag(internalCoeffs)) to the diagonal
    // and later REMOVES cmptMin(internalCoeffs) -- max(|x|,|y|,|z|) against the SIGNED min(x,y,z). They are
    // different numbers for a vector and agree only when the three components do, which is false on every
    // slip/symmetry patch. Both are built here rather than approximated by |iC[0]|; folding them into one
    // quantity turns relax into a pure 1/alpha scaling and is wrong in every boundary cell.
    //
    // The guard is part of the semantics, not an optimisation, and what it turns on is whether the case
    // NAMES a factor -- not what the factor is. fvMatrix::relax() is
    // `if (psi_.mesh().relaxEquation(name, relaxCoeff)) relax(relaxCoeff);` (fvMatrix.C:1250-1263), and
    // solution::relaxEquation is true whenever the name or "default" is in eqnRelaxDict_
    // (solution.C:330-334). relax(alpha) early-returns ONLY at alpha <= 0 (fvMatrix.C:1102-1107), so
    // alpha == 1 executes the whole body: the diagonal-dominance clamp D = max(|D|, sumOff), the
    // asymmetric boundary add/remove, and S += (D - D0)*psi. relax at alpha == 1 is not the identity, and
    // the conclusion drawn from that here used to be the wrong one -- it conflated "the case names NO
    // factor" (no relax) with "the case names 1" (relax runs). `equations { U 1; }` is an ordinary SIMPLEC
    // setting. Measured on the reference: assembling at relaxU = 1 with the old guard, against
    // relaxU = 1-1e-13 so that relax runs at alpha ~ 1, differs by 2.554e-01 on the diagonal and 2.834e+02
    // relative on source x for validation/pitzDailyTurb, and 1.061e+00 / 1.317e+03 for
    // validation/windAroundBuildingsBox. That diagonal is what rAU and H() are built from, so it reaches
    // pEqn.
    //
    // relaxEquationU is therefore the sentinel and relaxU carries only the value. alpha <= 0 with the
    // entry present needs no separate case: OpenFOAM's own relax(alpha) returns immediately for it, which
    // is indistinguishable from the entry being absent.
    //
    // psi is U, not rho*U: rhoSimpleFoam's momentum equation solves for U with a mass flux, so no density
    // enters this step.
    M.relaxed = false;
    if (in.relaxEquationU && in.relaxU > 0.0)
    {
        DeviceBuffer<scalar> iCmaxMag, iCmin;
        deviceCmptMaxMag3(M.iC[0], M.iC[1], M.iC[2], iCmaxMag);
        deviceCmptMin3(M.iC[0], M.iC[1], M.iC[2], iCmin);

        // Taken while M.relaxed is still false, so A.diag is the RAW diagonal; relaxedDiag is a separate
        // buffer, so there is no aliasing and the raw diagonal survives the call.
        const DeviceLduView A = M.view(dm);
        deviceRelaxDiag(
            A,
            dm,
            M.iC[0],
            in.relaxU,
            M.relaxedDiag,
            M.delta,
            nullptr,
            iCmaxMag.data(),
            iCmin.data());
        M.relaxed = true;

        // source += (relaxedDiag - rawDiag)*psi, per component -- the reference's
        //     M.source[c] += (M.diag[c] - D0[c]) * psi.internal[c]
        const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> t;
            deviceHadamard(t, M.delta, *U[k]);
            deviceAxpy(1.0, t, M.source[k]);
        }
    }
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
