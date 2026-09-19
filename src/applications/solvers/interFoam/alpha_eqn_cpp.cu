// interFoam's alpha equation, flux assembly -- see alpha_eqn_cpp.cuh for the provenance and for the
// four things in it that are not arithmetic.
#include "alpha_eqn_cpp.cuh"
#include "limitedSchemes_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "pbicgstab.cuh"
#include "smooth_solver_cpp.cuh"
#include <memory>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

// OpenFOAM's Switch, with the one behaviour that matters here: an UNRECOGNISED value is refused, not
// treated as off. `MULESCorr` selects the semi-implicit path, and a case that misspells it into silence
// runs the explicit limiter while its fvSolution says otherwise -- a difference in the solution, not in
// a diagnostic. Switch::find FatalErrors on a bad token too.
bool switchOr(const FoamDict& d, const std::string& key, bool def)
{
    const std::string v = d.wordOr(key, "");
    if (v.empty())                                                   return def;
    if (v == "yes" || v == "true"  || v == "on"  || v == "y" || v == "t" || v == "1") return true;
    if (v == "no"  || v == "false" || v == "off" || v == "n" || v == "f" || v == "0") return false;
    throw std::runtime_error(
        "brae interFoam alphaEqn: `" + key + " " + v + ";` is not a Switch. OpenFOAM accepts "
        "yes/no, true/false, on/off, y/n, t/f, 1/0 and FatalErrors on anything else; defaulting an "
        "unrecognised value to off would change which alpha solver runs without saying so.");
}

}   // namespace

AlphaControls readAlphaControls(const FoamDict& fvSolution, const std::string& alphaFieldName)
{
    const FoamDict* solvers = fvSolution.subDict("solvers");
    const FoamDict* sd = solvers ? solvers->subDict(alphaFieldName) : nullptr;
    if (!sd)
        throw std::runtime_error(
            "brae interFoam alphaEqn: fvSolution has no `solvers/" + alphaFieldName + "` entry. "
            "alphaControls.H reads every control of the alpha solve from there.");

    AlphaControls c;
    // nAlphaCorr and nAlphaSubCycles are get<label> in alphaControls.H -- no default, FatalError if
    // absent. A default here would pick a corrector count nobody chose, and nAlphaSubCycles in
    // particular changes the time step the alpha equation actually advances by.
    const scalar nCorr = sd->scalarOr("nAlphaCorr", scalar(-1));
    if (nCorr < 0)
        throw std::runtime_error(
            "brae interFoam alphaEqn: `nAlphaCorr` is missing from solvers/" + alphaFieldName
            + ". OpenFOAM has no default for it (get<label>, alphaControls.H) and FatalErrors.");
    const scalar nSub = sd->scalarOr("nAlphaSubCycles", scalar(-1));
    if (nSub < 0)
        throw std::runtime_error(
            "brae interFoam alphaEqn: `nAlphaSubCycles` is missing from solvers/" + alphaFieldName
            + ". OpenFOAM has no default for it (get<label>, alphaControls.H) and FatalErrors; it sets "
              "how many sub-steps the alpha field is advanced by inside one PIMPLE time step.");
    c.nAlphaCorr      = static_cast<label>(nCorr);
    c.nAlphaSubCycles = static_cast<label>(nSub);
    if (c.nAlphaCorr < 1 || c.nAlphaSubCycles < 1)
        throw std::runtime_error(
            "brae interFoam alphaEqn: nAlphaCorr and nAlphaSubCycles must be at least 1.");

    c.MULESCorr          = switchOr(*sd, "MULESCorr", false);
    c.alphaApplyPrevCorr = switchOr(*sd, "alphaApplyPrevCorr", false);
    c.icAlpha            = sd->scalarOr("icAlpha", scalar(0));
    c.scAlpha            = sd->scalarOr("scAlpha", scalar(0));
    return c;
}


scalar offCentringCoeff(AlphaDdt scheme,
                        label    nAlphaSubCycles,
                        scalar   schemeOcCoeff,
                        bool     warmedUp)
{
    switch (scheme)
    {
        case AlphaDdt::Euler:
        case AlphaDdt::localEuler:
            return scalar(0);                                   // alphaEqn.H:20-25

        case AlphaDdt::CrankNicolson:
            if (nAlphaSubCycles > 1)
                throw std::runtime_error(
                    "brae interFoam alphaEqn: `CrankNicolson` ddt(alpha) with nAlphaSubCycles > 1. "
                    "OpenFOAM FatalErrors on exactly this (alphaEqn.H:28-34): the off-centred flux is "
                    "built from the whole time step, and a sub-cycle advances by a fraction of it.");
            // alphaEqn.H:36-45: a cold start's FIRST step has no old-time ddt to off-centre against, so
            // the scheme runs Euler for one step and only then picks up its own coefficient.
            return warmedUp ? schemeOcCoeff : scalar(0);

        case AlphaDdt::other:
        default:
            throw std::runtime_error(
                "brae interFoam alphaEqn: ddtSchemes `ddt(alpha)` names a scheme that is neither Euler, "
                "localEuler nor CrankNicolson. OpenFOAM refuses it too -- \"Only Euler and CrankNicolson "
                "ddt schemes are supported\" (alphaEqn.H:49-51) -- so `backward`, which the rest of brae "
                "accepts, is not a gap in this port.");
    }
}


void compressionFlux(scalar                       cAlpha,
                     const SurfaceScalarField&    phi,
                     const std::vector<scalar>&   magSf,
                     const std::vector<FvPatch>&  patches,
                     scalar                       icAlpha,
                     const std::vector<scalar>&   magUf,
                     scalar                       scAlpha,
                     const std::vector<scalar>&   shearf,
                     SurfaceScalarField&          phic)
{
    const std::size_t nIf = phi.internal.size();
    // magSf is the mesh's own face array -- internal faces first, then the boundary patches -- so the
    // internal faces index straight through. Only the boundary half is unused, because phic's boundary
    // is zeroed below whatever the areas are.
    if (magSf.size() < nIf)
        throw std::runtime_error(
            "brae interFoam alphaEqn: magSf is shorter than the internal face count.");
    if (icAlpha > scalar(0) && magUf.size() != nIf)
        throw std::runtime_error(
            "brae interFoam alphaEqn: icAlpha > 0 needs interpolate(mag(U)) on every internal face "
            "(alphaEqn.H:65).");
    if (scAlpha > scalar(0) && shearf.size() != nIf)
        throw std::runtime_error(
            "brae interFoam alphaEqn: scAlpha > 0 needs mag(delta() & interpolate(symm(grad(U)))) on "
            "every internal face (alphaEqn.H:72).");

    phic.internal.resize(nIf);
    for (std::size_t f = 0; f < nIf; ++f)
        phic.internal[f] = cAlpha * std::fabs(phi.internal[f] / magSf[f]);

    // ORDER MATTERS AND IS OpenFOAM's. The isotropic term REPLACES part of the standard one --
    // `phic *= (1 - icAlpha); phic += (cAlpha*icAlpha)*interpolate(mag(U))` -- so it is a blend. The
    // shear term is added AFTERWARDS and is therefore NOT scaled by (1 - icAlpha): it is an addition,
    // not part of the blend. Applying both as a single weighted sum would be a different model.
    if (icAlpha > scalar(0))
        for (std::size_t f = 0; f < nIf; ++f)
            phic.internal[f] = phic.internal[f] * (scalar(1) - icAlpha) + (cAlpha * icAlpha) * magUf[f];

    if (scAlpha > scalar(0))
        for (std::size_t f = 0; f < nIf; ++f)
            phic.internal[f] += scAlpha * shearf[f];

    // "Do not compress interface at non-coupled boundary faces (inlets, outlets etc.)" --
    // alphaEqn.H:79-89. Interface compression is anti-diffusion; at an open boundary it sharpens an
    // interface the boundary does not have, pulling alpha in through it. Every uncoupled patch is
    // zeroed; a coupled one is not.
    phic.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        phic.boundary[pi].assign(static_cast<std::size_t>(fp.size), scalar(0));
        if (!fp.coupled)
        {
            continue;
        }
        // A COUPLED patch is the one kind alphaEqn.H:79-89 leaves alone: the interface does pass
        // through it, and phic there is what it is on an internal face
        if (icAlpha > scalar(0) || scAlpha > scalar(0))
        {
            throw std::runtime_error(
                "brae interFoam alphaEqn: icAlpha or scAlpha is set and patch `" + fp.name + "` is "
                "coupled; the isotropic and shear compression terms are carried on internal faces only.");
        }
        for (std::size_t i = 0; i < phic.boundary[pi].size(); ++i)
        {
            phic.boundary[pi][i] = cAlpha * std::fabs(phi.boundary[pi][i] / fp.magSf[i]);
        }
    }
}


void offCentredFlux(const SurfaceScalarField& phi,
                    const SurfaceScalarField& phiOld,
                    scalar                    cnCoeff,
                    scalar                    ocCoeff,
                    SurfaceScalarField&       phiCN)
{
    // alphaEqn.H:91-97. OpenFOAM starts with `tmp<surfaceScalarField> phiCN(phi)` and only replaces it
    // when ocCoeff > 0, so on every Euler case phiCN IS phi -- not a copy that happens to be equal.
    if (ocCoeff <= scalar(0))
    {
        phiCN = phi;
        return;
    }
    if (phiOld.internal.size() != phi.internal.size())
        throw std::runtime_error(
            "brae interFoam alphaEqn: the CrankNicolson off-centred flux needs phi.oldTime().");
    phiCN.internal.resize(phi.internal.size());
    for (std::size_t f = 0; f < phi.internal.size(); ++f)
        phiCN.internal[f] = cnCoeff * phi.internal[f] + (scalar(1) - cnCoeff) * phiOld.internal[f];
    phiCN.boundary.resize(phi.boundary.size());
    for (std::size_t pi = 0; pi < phi.boundary.size(); ++pi)
    {
        phiCN.boundary[pi].resize(phi.boundary[pi].size());
        for (std::size_t i = 0; i < phi.boundary[pi].size(); ++i)
            phiCN.boundary[pi][i] =
                cnCoeff * phi.boundary[pi][i] + (scalar(1) - cnCoeff) * phiOld.boundary[pi][i];
    }
}


void fluxWithScheme(const SurfaceScalarField&     psi,
                    const GeometricField<scalar>& vf,
                    AlphaFluxScheme               scheme,
                    const PrimitiveMesh&          m,
                    const FvGeometry&             g,
                    const std::vector<FvPatch>&   patches,
                    SurfaceScalarField&           out)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    std::vector<scalar> w;
    std::vector<vector> gradVfLimited;          // vanLeer's, kept for the coupled faces below
    switch (scheme)
    {
        case AlphaFluxScheme::linear:
            w = g.weights();
            break;
        case AlphaFluxScheme::upwind:
            w = limitedSchemes::upwindWeights(psi.internal);
            break;
        case AlphaFluxScheme::interfaceCompression:
            // a PhiScheme: the two cell values alone, no gradient (limitedSchemes_cpp.cuh)
            w = limitedSchemes::interfaceCompressionWeights(psi.internal, vf, m, g);
            break;
        case AlphaFluxScheme::vanLeer:
        default:
        {
            // LimitedScheme::calcLimiter takes fvc::grad(vf) through the case's gradSchemes; 43 of the
            // 44 shipped interFoam tutorials say `default Gauss linear`, and this takes that. The one
            // that says `cellLimited leastSquares 1` is not served by this path yet and would need the
            // scheme resolved at the call site, as rhoSimpleFoam does for grad(U).
            gradVfLimited = fvc::gaussGrad(vf, m, g, patches);
            w = limitedSchemes::vanLeerWeights(psi.internal, vf, gradVfLimited, m, g);
            break;
        }
    }

    // gaussConvectionScheme.C:64-73 -- flux(faceFlux, vf) == faceFlux*interpolate(vf), and the
    // interpolation was constructed WITH faceFlux, so psi both scales the result and picks the upwind
    // side. Passing one flux to the weights and another to the multiply would be a different operator.
    out.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const scalar vfF = w[f] * vf.internal[own[f]] + (scalar(1) - w[f]) * vf.internal[nei[f]];
        out.internal[f] = psi.internal[f] * vfF;
    }

    // At an UNCOUPLED patch surfaceInterpolation returns the patch value itself -- there is no second
    // cell to weight against -- so the face flux is psi_b * alpha_b whatever the scheme.
    out.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& vb = vf.boundary[pi]->value();
        out.boundary[pi].resize(vb.size());
        if (patches[pi].coupled)
        {
            // THE SCHEME'S OWN WEIGHT ON A COUPLED FACE, as on an internal one (LimitedScheme.C,
            // calcLimiter's bLim[patchi].coupled() branch): the limiter from the two cells, their
            // gradients and the patch's delta, blended with the patch's central weight. An uncoupled
            // patch takes limiter 1 and weight 1 -- the patch value, below.
            const FvPatch& fp = patches[pi];
            for (std::size_t i = 0; i < vb.size(); ++i)
            {
                const label P = fp.faceCells[i];
                // patchNeighbourField of alpha and of its gradient: the cell across a cyclic, the AMI's
                // weighted sum across a cyclicAMI
                const scalar vfN = patchNeighbourValue(fp, static_cast<label>(i), vf.internal);
                const scalar pb = psi.boundary[pi][i];
                const scalar up = (pb >= scalar(0)) ? scalar(1) : scalar(0);
                scalar wf = fp.weights[i];
                switch (scheme)
                {
                    case AlphaFluxScheme::linear:
                        break;
                    case AlphaFluxScheme::upwind:
                        wf = up;
                        break;
                    case AlphaFluxScheme::interfaceCompression:
                        throw std::runtime_error(
                            "brae interFoam alphaEqn: `interfaceCompression` across the coupled patch `"
                            + fp.name + "` is not ported.");
                    case AlphaFluxScheme::vanLeer:
                    default:
                    {
                        const scalar r = limitedSchemes::detail::rScalar(pb, vf.internal[P], vfN,
                                                                 gradVfLimited[P],
                                                                 patchNeighbourValue(fp, static_cast<label>(i), gradVfLimited),
                                                                 fp.delta[i]);
                        const scalar lim = limitedSchemes::detail::vanLeerLimiter(r);
                        wf = lim*fp.weights[i] + (scalar(1) - lim)*up;
                        break;
                    }
                }
                out.boundary[pi][i] = pb * (wf*vf.internal[P] + (scalar(1) - wf)*vfN);
            }
            continue;
        }
        for (std::size_t i = 0; i < vb.size(); ++i)
        {
            const scalar pb = (pi < psi.boundary.size() && i < psi.boundary[pi].size())
                            ? psi.boundary[pi][i] : scalar(0);
            out.boundary[pi][i] = pb * vb[i];
        }
    }
}


void alphaPhiUn(const SurfaceScalarField&     phi,
                const SurfaceScalarField&     phir,
                const GeometricField<scalar>& alpha1,
                const GeometricField<scalar>& alpha2,
                AlphaFluxScheme               alphaScheme,
                AlphaFluxScheme               alpharScheme,
                const PrimitiveMesh&          m,
                const FvGeometry&             g,
                const std::vector<FvPatch>&   patches,
                SurfaceScalarField&           out)
{
    // Term 1: fvc::flux(phi, alpha1, alphaScheme) -- plain advection.
    SurfaceScalarField adv;
    fluxWithScheme(phi, alpha1, alphaScheme, m, g, patches, adv);

    // Term 2, alphaEqn.H:171-176, TWO MINUS SIGNS DEEP:
    //
    //     fvc::flux(-fvc::flux(-phir, alpha2, alpharScheme), alpha1, alpharScheme)
    //
    // The inner flux is evaluated with -phir, so alpha2 is upwinded FROM THE OTHER SIDE than phir
    // would pick; the result is then negated and used both to scale and to upwind alpha1. Dropping
    // either minus gives a compressive flux of the same magnitude taken from the wrong cell, which on a
    // limited scheme differs only across the interface -- the only place it matters.
    SurfaceScalarField negPhir;
    negPhir.internal.resize(phir.internal.size());
    for (std::size_t f = 0; f < phir.internal.size(); ++f) negPhir.internal[f] = -phir.internal[f];
    negPhir.boundary.resize(phir.boundary.size());
    for (std::size_t pi = 0; pi < phir.boundary.size(); ++pi)
    {
        negPhir.boundary[pi].resize(phir.boundary[pi].size());
        for (std::size_t i = 0; i < phir.boundary[pi].size(); ++i)
            negPhir.boundary[pi][i] = -phir.boundary[pi][i];
    }

    SurfaceScalarField inner;
    fluxWithScheme(negPhir, alpha2, alpharScheme, m, g, patches, inner);
    for (scalar& s : inner.internal) s = -s;                       // the OUTER minus
    for (std::vector<scalar>& b : inner.boundary) for (scalar& s : b) s = -s;

    SurfaceScalarField comp;
    fluxWithScheme(inner, alpha1, alpharScheme, m, g, patches, comp);

    out.internal.resize(adv.internal.size());
    for (std::size_t f = 0; f < adv.internal.size(); ++f)
        out.internal[f] = adv.internal[f] + comp.internal[f];
    out.boundary.resize(adv.boundary.size());
    for (std::size_t pi = 0; pi < adv.boundary.size(); ++pi)
    {
        out.boundary[pi].resize(adv.boundary[pi].size());
        for (std::size_t i = 0; i < adv.boundary[pi].size(); ++i)
            out.boundary[pi][i] = adv.boundary[pi][i] + comp.boundary[pi][i];
    }
}


void massFlux(const SurfaceScalarField& alphaPhi10,
              const SurfaceScalarField& phiForRho2,
              scalar                    rho1,
              scalar                    rho2,
              SurfaceScalarField&       rhoPhi)
{
    // alphaEqn.H:248 / :260. A linear blend in alphaPhi10: where the alpha flux equals the volumetric
    // flux (pure phase 1) this is phi*rho1, and where it is zero (pure phase 2) it is phi*rho2. The
    // subtraction order is what makes that true, and swapping it keeps both the dimensions and the
    // magnitude while inverting the mixture.
    if (alphaPhi10.internal.size() != phiForRho2.internal.size())
        throw std::runtime_error("brae interFoam alphaEqn: alphaPhi10 and phi differ in length.");
    const scalar dRho = rho1 - rho2;
    rhoPhi.internal.resize(alphaPhi10.internal.size());
    for (std::size_t f = 0; f < alphaPhi10.internal.size(); ++f)
        rhoPhi.internal[f] = alphaPhi10.internal[f] * dRho + phiForRho2.internal[f] * rho2;
    rhoPhi.boundary.resize(alphaPhi10.boundary.size());
    for (std::size_t pi = 0; pi < alphaPhi10.boundary.size(); ++pi)
    {
        rhoPhi.boundary[pi].resize(alphaPhi10.boundary[pi].size());
        for (std::size_t i = 0; i < alphaPhi10.boundary[pi].size(); ++i)
            rhoPhi.boundary[pi][i] =
                alphaPhi10.boundary[pi][i] * dRho + phiForRho2.boundary[pi][i] * rho2;
    }
}


// ---------------------------------------------------------------------------------------------------

void alphaEqnStep(GeometricField<scalar>&                 alpha1,
                  const std::vector<scalar>&              alpha1Old,
                  const AlphaStepInput&                   in,
                  const interfaceProps::InterfaceCoeffs&  ic,
                  const MULES::Controls&                  mulesCtl,
                  const PrimitiveMesh&                    m,
                  const FvGeometry&                       g,
                  const std::vector<FvPatch>&             patches,
                  SurfaceScalarField&                     alphaPhi10,
                  SurfaceScalarField&                     rhoPhi,
                  SurfaceScalarField&                     nHatf,
                  std::vector<scalar>&                    K,
                  SurfaceScalarField*                     prevCorr)
{
    if (!in.phi || !in.phiCN)
        throw std::runtime_error("brae interFoam alphaEqn: phi and phiCN are both required.");
    if (in.nAlphaCorr < 1)
        throw std::runtime_error("brae interFoam alphaEqn: nAlphaCorr must be at least 1.");
    const label nC = m.nCells();

    // alpha1 COMES IN AS IT STANDS AND IS NOT RESET TO ITS OLD TIME. alphaEqn.H has no such assignment
    // anywhere: the old time enters through fvmDdt's source and through MULES' psi.oldTime(), and the
    // CURRENT alpha1 is the pre-solve's initial guess, the field alphaPhiUn is built from and the one
    // the limiter takes its extrema from (MULESTemplates.C:208 against :244). With one outer corrector
    // the two are the same field -- nothing has touched alpha1 since the last step, or the last
    // sub-cycle, ended -- so resetting it was a no-op, and this function did reset it, for as long as it
    // had only ever been run that way. With `nOuterCorrectors 2` OpenFOAM's second pass starts from the
    // FIRST PASS'S RESULT. Measured on damBreak at dt 5e-3 against real OpenFOAM: every first-pass alpha
    // solve exact to 1e-13 and every second-pass one 84% to 170% out in its initial residual, one of
    // them taking 3 sweeps for OpenFOAM's 2, and the fields 4.9e-06 in alpha and 1.0e-03 in U.
    //
    // NOR IS ITS BOUNDARY EVALUATED HERE. OpenFOAM's alphaEqn.H evaluates alpha1 nowhere before its
    // fluxes: the explicit path's alphaPhiUn = fvc::flux(phi, alpha1, ...) reads the patch values as the
    // LAST evaluate left them -- the previous MULES solve's closing correctBoundaryConditions, or at the
    // first step the file's own `value` -- and the pre-solve's matrix reads only coefficients. A
    // flux-conditional patch learns the new flux at its next evaluate, not before. MEASURED on
    // RAS/mixerVesselAMI's first step (explicit MULES, two sub-cycles), with OpenFOAM's sub-cycles
    // instrumented: the outlet's alpha flux is exactly 0 in sub-cycle one, since the file writes `value
    // uniform 0` there under water, and the 62 outlet cells rise to 1.104; brae evaluated here, took the
    // cells' alpha of 1 on the outflow faces, and kept them at 1 -- alpha 9.4e-02 from OpenFOAM after one
    // step. (An earlier version kept this evaluate on a damBreak measurement taken while the file had
    // other defects.)

    // phic = cAlpha*|phi/magSf|, zeroed on every non-coupled boundary -- ONCE, at the top, as alphaEqn.H
    // computes it (:59-89), before the pre-solve and before anything interpolates across the mesh. With
    // fixed geometry it is the same field wherever it is formed. With a cyclicACMI whose scale moves the
    // open area, it is not: OpenFOAM rescales lazily, and this phic is the one formed on the areas the
    // step STARTED with -- see geometryUpdate below.
    SurfaceScalarField phic;
    compressionFlux(ic.cAlpha, *in.phi, g.magSf(), patches,
                    in.icAlpha, {}, in.scAlpha, {}, phic);
    // ...and the step's geometry change lands here, after phic and before the pre-solve assembles.
    // MEASURED on RAS/damBreakLeakage's opening step with OpenFOAM's alphaEqn instrumented: phic on the
    // newly opened faces is 41.7, |phi| over the CLOSED area, against nHatf on the open one, and the
    // limited correction then cancels the pre-solve's flux on the receiving side exactly (its alpha
    // flux 6e-23 where rescaling first gave 5.8e-13).
    if (in.geometryUpdate)
    {
        in.geometryUpdate();
    }

    SurfaceScalarField upwindFlux;              // talphaPhi1UD -- cached for alphaApplyPrevCorr
    if (in.MULESCorr)
    {
        // alphaEqn.H:103-155, THE IMPLICIT PRE-SOLVE.
        //
        //     fvScalarMatrix alpha1Eqn
        //     (
        //         EulerDdtScheme<scalar>(mesh).fvmDdt(alpha1)
        //       + gaussConvectionScheme<scalar>(mesh, phiCN, upwind<scalar>(mesh, phiCN))
        //             .fvmDiv(phiCN, alpha1)
        //      == Su + fvm::Sp(Sp + divU, alpha1)
        //     );
        //
        // THE CONVECTION IS UPWIND, NAMED EXPLICITLY IN THE CODE -- not the case's div(phi,alpha).
        // That is the whole point of the split: the implicit half is first-order and unconditionally
        // bounded, and every bit of the case's scheme lives in the correction CMULES then limits.
        // Running the case's vanLeer here would make the pre-solve itself unbounded and leave CMULES
        // correcting towards a field that had already overshot.
        //
        // Su, Sp and divU are all zeroField for interFoam (interFoam/alphaSuSp.H), so the right-hand
        // side vanishes; it is written out above rather than dropped because interPhaseChangeFoam's
        // own alphaSuSp.H makes all three live.
        // ...and the fvMatrix constructor's updateCoeffs, ahead of the assembly that reads it
        if (in.updateModelledBoundary)
        {
            in.updateModelledBoundary();
        }
        FvScalarMatrix M = fvm::div<scalar>(in.phiCN->internal, in.phiCN->boundary, alpha1, m, patches);

        // fvm::ddt(alpha1), Euler, rho == 1: diag += V/dt, source += V*alpha.oldTime()/dt -- and on
        // a moving mesh the diagonal's V is Vsc and the source's is Vsc0 (EulerDdtScheme.C:383-392)
        const scalar rDeltaT = scalar(1) / in.deltaT;
        for (label c = 0; c < nC; ++c)
        {
            if (in.Vsc)
            {
                M.diag[c]   += rDeltaT * (*in.Vsc)[c];
                M.source[c] += rDeltaT * alpha1Old[c] * (*in.Vsc0)[c];
                continue;
            }
            M.diag[c]   += rDeltaT * g.V()[c];
            M.source[c] += rDeltaT * g.V()[c] * alpha1Old[c];
        }

        const SolverPerformance sp = in.smoothSolver
            ? smoothSolver(M, alpha1.internal, m, patches, in.symmetric,
                           in.tolAlpha, in.relTolAlpha, in.maxIterAlpha, in.minIterAlpha, in.nSweeps)
            : pbicgstab(M, alpha1.internal, m, patches, in.tolAlpha, in.relTolAlpha, in.maxIterAlpha,
                        in.minIterAlpha);
        if (in.solveLog)
        {
            in.solveLog->push_back(LinearSolveRecord{sp.initialResidual, sp.finalResidual, sp.nIterations});
        }
        alpha1.evaluateBoundary();

        // alphaPhi10 = alpha1Eqn.flux(): the CONSERVATIVE face flux of the SOLVED matrix, not a flux
        // rebuilt from the field afterwards. What it buys is the closure
        //     alpha == alpha.oldTime() - (dt/V)*sum_f alphaPhi10
        // to the LINEAR SOLVER's residual. MEASURED (tests/test_device_alpha_presolve.cu): on the
        // INTERNAL faces of this pure-upwind matrix flux() is bit-identical to the upwind flux of the
        // SOLVED field -- faceH collapses to phi*psi[upwind] -- so those two are not two things. What
        // does break the closure is rebuilding the flux from the field the solve STARTED from (5.1e-03)
        // or with any other interpolation (a central flux, 2.3e-02); and at a patch whose coefficients
        // are not (1, 0), flux() carries the prescribed value where a rebuilt flux carries the cell's.
        upwindFlux = matrixFlux(M, alpha1.internal, m, patches);
        alphaPhi10 = upwindFlux;

        // alphaApplyPrevCorr: the previous step's compression flux as a first guess, limited against
        // the field the pre-solve has just produced.
        //
        // THE FLUX ARGUMENT IS alphaPhi10, the ALPHA flux the pre-solve left (alphaEqn.H:136-144), and
        // not phiCN. limiterCorr reads it in exactly one place, the boundary outlet test
        // `(phi_b + phiCorr_b) > SMALL*SMALL` (CMULESTemplates.C:546), and this call passed the
        // VOLUMETRIC flux there -- the same slip the corrector loop below had, fixed there when its
        // gate measured 3.99e-09 -> 1.07e-10, and left here because nothing ran this branch.
        //
        // IT TOOK THREE FIXTURES TO MEASURE, and the two that failed say when it matters. On damBreak
        // the arguments agree to five digits: the boundary half of the cached correction is zero
        // unless alpha's patch value moves between the pre-solve and the correctors, which needs water
        // LEAVING through an outflow face. With the water column raised to the atmosphere it does
        // leave, and the two tests then DECIDE DIFFERENTLY on up to 13 boundary faces a step, on
        // corrections worth 15% of the face flux -- and the answer was still bit-identical at dt 5e-3,
        // because on those faces the cell limiter returned 1 and made the decision moot. It needs the
        // decision to differ AND the limiter to be biting: the same fixture at dt 1e-2, where one
        // application moves one cell's alpha by 0.10. Against real OpenFOAM, four steps: alpha 5.5e-03,
        // p_rgh 5.7e-03 and U 2.2e-02 with phiCN; 4.6e-13, 2.7e-13 and 5.5e-13 with alphaPhi10.
        if (in.alphaApplyPrevCorr && prevCorr && !prevCorr->internal.empty())
        {
            MULES::Fields mf0;
            mf0.Vsc = in.Vsc;
            mf0.Vsc0 = in.Vsc0;
            MULES::correctLimited(rDeltaT, alpha1,
                                  in.controlPrevCorrOutletOnPhiCN ? *in.phiCN : alphaPhi10,
                                  *prevCorr, mf0, mulesCtl, m, g, patches);
            for (std::size_t f = 0; f < alphaPhi10.internal.size(); ++f)
                alphaPhi10.internal[f] += prevCorr->internal[f];
            for (std::size_t pi = 0; pi < alphaPhi10.boundary.size(); ++pi)
                for (std::size_t i = 0; i < alphaPhi10.boundary[pi].size(); ++i)
                    alphaPhi10.boundary[pi][i] += prevCorr->boundary[pi][i];
        }

        // alphaEqn.H:151-153 -- alpha2 = 1 - alpha1, then mixture.correct(), INSIDE the MULESCorr
        // block and before the corrector loop. That is a calculateK pass of its own, and since the
        // curvature is a fixed point in its passes (see interface_properties_cpp.cu) one pass short is
        // not a rounding difference. Missing it put damBreak at 1.05e-07 against OpenFOAM where the
        // full sequence gives 3.4e-09 -- thirty times worse, and only visible because damBreak is the
        // case that sets MULESCorr.
        interfaceProps::calculateK(alpha1, ic, m, g, patches, /*gradLeastSquares=*/false, nHatf, K);
    }

    for (label aCorr = 0; aCorr < in.nAlphaCorr; ++aCorr)
    {
        // phir uses the nHatf mixture.correct() left LAST TIME (alphaEqn.H:162), not one computed
        // here. That ordering is not cosmetic: calculateK reads alpha's wall gradient, which its own
        // previous pass wrote, so an extra pass changes the curvature. Measured against OpenFOAM's own
        // K on capillaryRise, one pass too few is 18% low at the contact line.

        // phir = phic*nHatf, with the phic formed at the top. nHatf is already a FLUX (nHatfv & Sf), not
        // a unit vector, so this is a product of two face fields and needs no further area weighting.
        SurfaceScalarField phir;
        phir.internal.resize(phic.internal.size());
        for (std::size_t f = 0; f < phic.internal.size(); ++f)
            phir.internal[f] = phic.internal[f] * nHatf.internal[f];
        phir.boundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            phir.boundary[pi].resize(phic.boundary[pi].size());
            for (std::size_t i = 0; i < phic.boundary[pi].size(); ++i)
                phir.boundary[pi][i] = phic.boundary[pi][i] * nHatf.boundary[pi][i];
        }

        // alpha2 = 1 - alpha1, rebuilt from the CURRENT alpha1 -- the compressive term reads it.
        GeometricField<scalar> alpha2;
        alpha2.internal.resize(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) alpha2.internal[c] = scalar(1) - alpha1.internal[c];
        for (const FvPatch& q : patches)
        {
            if (q.coupled)
            {
                alpha2.boundary.push_back(std::make_unique<CoupledCyclicPatchField<scalar>>(q));
                continue;
            }
            alpha2.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        }
        alpha2.evaluateBoundary();

        SurfaceScalarField un;
        alphaPhiUn(*in.phi, phir, alpha1, alpha2, in.alphaScheme, in.alpharScheme, m, g, patches, un);

        MULES::Fields mf;                       // all null: rho == 1, Sp == Su == 0, bounds [0,1]
        mf.Vsc = in.Vsc;                        // ...and the volumes of a mesh that moves
        mf.Vsc0 = in.Vsc0;
        if (in.MULESCorr)
        {
            // alphaEqn.H:178-205. The correction is what the high-order flux adds to the upwind one
            // the implicit solve already applied, and CMULES limits THAT.
            SurfaceScalarField corr = un;
            for (std::size_t f = 0; f < corr.internal.size(); ++f)
                corr.internal[f] -= alphaPhi10.internal[f];
            for (std::size_t pi = 0; pi < corr.boundary.size(); ++pi)
                for (std::size_t i = 0; i < corr.boundary[pi].size(); ++i)
                    corr.boundary[pi][i] -= alphaPhi10.boundary[pi][i];

            const std::vector<scalar> alpha10 = alpha1.internal;      // saved for the relaxation
            MULES::correctLimited(scalar(1)/in.deltaT, alpha1, un, corr, mf, mulesCtl, m, g, patches);

            // UNDER-RELAXED FOR EVERY CORRECTOR BUT THE FIRST, and BOTH halves are relaxed: the field
            // by averaging with its pre-correction value, the flux by taking half the correction.
            // Relaxing one and not the other leaves alpha and alphaPhi10 describing different states,
            // and rhoPhi is built from the flux while UEqn is built on the field.
            const scalar w = (aCorr == 0) ? scalar(1) : scalar(0.5);
            if (aCorr != 0)
            {
                for (label c = 0; c < nC; ++c)
                    alpha1.internal[c] = scalar(0.5)*alpha1.internal[c] + scalar(0.5)*alpha10[c];
                alpha1.evaluateBoundary();
            }
            for (std::size_t f = 0; f < corr.internal.size(); ++f)
                alphaPhi10.internal[f] += w * corr.internal[f];
            for (std::size_t pi = 0; pi < corr.boundary.size(); ++pi)
                for (std::size_t i = 0; i < corr.boundary[pi].size(); ++i)
                    alphaPhi10.boundary[pi][i] += w * corr.boundary[pi][i];
        }
        else
        {
            // MULES::explicitSolve(geometricOneField(), alpha1, phiCN, alphaPhi10, 0, 0, 1, 0) --
            // alphaEqn.H:208-220. alphaPhi10 IS alphaPhiUn on the explicit path, limited in place.
            alphaPhi10 = un;
            // `un` above read alpha's patch values as the LAST update left them; the limiter below
            // reads them as this one leaves them -- see AlphaStepInput::updateModelledBoundary
            if (in.updateModelledBoundary)
            {
                in.updateModelledBoundary();
            }
            MULES::explicitSolveLimited(scalar(1)/in.deltaT, alpha1, alpha1Old, *in.phiCN, alphaPhi10,
                                        mf, mulesCtl, m, g, patches);
        }

        // alpha2 = 1.0 - alpha1 (alphaEqn.H:223), a whole-field assignment, so alpha2's PATCH values
        // are fixed here -- one line above the mixture.correct() that rewrites alpha1's.
        if (in.alpha2BndOut)
        {
            in.alpha2BndOut->resize(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const std::vector<scalar>& ab = alpha1.boundary[pi]->value();
                (*in.alpha2BndOut)[pi].resize(ab.size());
                for (std::size_t i = 0; i < ab.size(); ++i)
                {
                    (*in.alpha2BndOut)[pi][i] = scalar(1) - ab[i];
                }
            }
        }

        // ...and mixture.correct() at the BOTTOM of the corrector, alphaEqn.H:225. The next corrector
        // (or the next sub-cycle) compresses towards where MULES has just put the interface.
        interfaceProps::calculateK(alpha1, ic, m, g, patches, /*gradLeastSquares=*/false, nHatf, K);
    }

    // alphaEqn.H:228-236: the cache for the NEXT step is alphaPhi10 minus the upwind flux -- i.e. the
    // compression the correctors ended up applying. Cleared otherwise, so a case that turns
    // alphaApplyPrevCorr off does not carry a stale flux forward.
    if (prevCorr)
    {
        if (in.alphaApplyPrevCorr && in.MULESCorr)
        {
            *prevCorr = alphaPhi10;
            for (std::size_t f = 0; f < prevCorr->internal.size(); ++f)
                prevCorr->internal[f] -= upwindFlux.internal[f];
            for (std::size_t pi = 0; pi < prevCorr->boundary.size(); ++pi)
                for (std::size_t i = 0; i < prevCorr->boundary[pi].size(); ++i)
                    prevCorr->boundary[pi][i] -= upwindFlux.boundary[pi][i];
        }
        else
        {
            prevCorr->internal.clear();
            prevCorr->boundary.clear();
        }
    }

    // rhoPhi = alphaPhi10*(rho1 - rho2) + phiCN*rho2, alphaEqn.H:248. The Euler branch multiplies
    // rho2 by phiCN; the CrankNicolson one by phi, and they are the same field only when ocCoeff is 0.
    massFlux(alphaPhi10, *in.phiCN, in.rho1, in.rho2, rhoPhi);
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
