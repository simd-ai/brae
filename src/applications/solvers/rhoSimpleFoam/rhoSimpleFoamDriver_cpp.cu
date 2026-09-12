// rhoSimpleFoamDriver_cpp.cu -- see the header for what this is and why the parse is shared.
#include "io_precision.cuh"   // setIOPrecision: OF ties every Info line to writePrecision
#include "rhoSimpleFoamDriver_cpp.cuh"

#include "brae_notice.cuh"
#include "brae_time.cuh"
#include "rhoScalarTransportFO.cuh"   // functionObjects::scalarTransport on this arm (item 15c)
#include "dict_audit.cuh"
#include <memory>   // DictAuditScope: every dictionary entry read off disk and never applied, reported on every exit
#include "foam_field_writer.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "linear_solver_setup.cuh"
#include "residual_control.cuh"
#include "scheme_parse.cuh"
#include "solver_controls.cuh"
#include "write_control.cuh"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include "start_time.cuh"   // openFoamNSteps: OF Time::run's own step count

namespace brae {
namespace cpu {
namespace rhoSimple {

StepInput buildStepInput(
    const std::string&     caseDir,
    const RhoSimpleFields& f,
    const FoamDict&        fvSolution,
    const PrimitiveMesh&   m,
    CaseRefusals&          refusals,
    bool                   verbose)
{
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");
    const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
    const FoamDict* re  = rf ? rf->subDict("equations") : nullptr;
    const FoamDict* rfl = rf ? rf->subDict("fields") : nullptr;

    StepInput in;

    // fvOptions and MRF, derived by the SHARED helper so the CUDA harness gets the same flags -- the
    // device-twin guards were reachable only from fail-proofs before it existed. `refusals` is the
    // caller's, because in.fvOpts points into it and must not dangle when this function returns.
    refusals               = deriveCaseRefusals(caseDir, m);
    in.hasMRF              = refusals.hasMRF;
    in.hasFvOptions        = refusals.hasFvOptions;
    in.fvOptionUnsupported = refusals.fvOptionUnsupported;
    in.limitT              = refusals.limitT;
    in.limitTmin           = refusals.limitTmin;
    in.limitTmax           = refusals.limitTmax;
    in.limitTname          = refusals.limitTname;
    // The per-patch nut wall function, read from 0/nut's boundaryField types by createFields. The
    // caller owns `f` for the whole run, so the pointer outlives the loop.
    in.nutWallKind         = &f.nutWallKind;
    if (!refusals.hasFvOptions && !refusals.opts.empty()) in.fvOpts = &refusals.opts;

    in.consistent = simpleDict && simpleDict->wordOr("consistent", "no") == "yes";
    in.transonic  = simpleDict && simpleDict->wordOr("transonic",  "no") == "yes";
    // getOrDefault<label>(..., 0) -- solutionControl.C:47. The step used to solve the pressure equation
    // exactly once whatever the case named here, which on a corrected non-orthogonal case is a different
    // trajectory than OpenFOAM's (tests/rho_nonorth_corrector_vs_openfoam.sh).
    in.nNonOrthogonalCorrectors =
        simpleDict ? (label)simpleDict->scalarOr("nNonOrthogonalCorrectors", 0) : 0;

    // The schemes, PARSED from the case. Stating a fixture's own schemes was safe while there was one
    // fixture and became a silent substitution the moment a second case appeared: aerofoilNACA0012 asks
    // for `bounded Gauss linearUpwind limited` on div(phi,U) where sbMatched asks for plain upwind, and
    // linearUpwind's deferred correction is a SOURCE term -- running upwind instead left the wall-cell
    // momentum source at 2.4e-02 against OpenFOAM's 2.5e+00.
    {
        // U is a VECTOR: the V forms and LUST are legal on it, and the step assembles all six
        // (rhoUEqn_cpp.cu / rhoUEqn.cu switch on every DivScheme). This mapping used to know only
        // linearUpwind / limitedLinear / upwind, so squareBend's `Gauss limitedLinearV 1` ran as plain
        // limitedLinear -- the direction limiter dropped, in silence (item 16i). The scalars are parsed as
        // scalars, where the parser refuses the V forms and LUST by name.
        const FieldDivScheme dU  = parseFieldDivScheme(caseDir, "U", /*vectorField=*/true);
        const FieldDivScheme dHe = parseFieldDivScheme(caseDir, f.heName);
        auto toScheme = [](const FieldDivScheme& d)
        {
            if (d.limitedLinearV) return DivScheme::limitedLinearV;
            if (d.lust)           return DivScheme::LUST;
            if (d.linearUpwindV)  return DivScheme::linearUpwindV;
            if (d.linearUpwind)   return DivScheme::linearUpwind;
            if (d.limited)        return DivScheme::limitedLinear;
            return DivScheme::upwind;
        };
        in.schemeU  = toScheme(dU);
        in.schemeHe = toScheme(dHe);
        // THE KINETIC-ENERGY TERM'S OWN ENTRY. EEqn.H builds fvc::div(phi, Ekp) on an e-thermo and
        // fvc::div(phi, K) on an h-thermo, and OpenFOAM resolves each under its own key, div(phi,Ekp) or
        // div(phi,K). This used to copy the energy entry with the note "follows the energy entry in
        // every tutorial" -- true of every tutorial and every fixture, which is exactly why a case that
        // separates the two was never seen: it ran the energy scheme on K under the case's own name.
        // Parsed like the others; a scheme the energy equation has not ported refuses there by name.
        const std::string keName = (f.heName == "e") ? "Ekp" : "K";
        const FieldDivScheme dKE = parseFieldDivScheme(caseDir, keName);
        in.schemeKE = toScheme(dKE);
        in.boundedU      = dU.bounded;
        in.boundedHe     = dHe.bounded;
        in.boundedKE     = dKE.bounded;
        in.schemeCoeffU  = dU.coeff;      // RAW k: the weights functions compute twoByk (scheme_parse.cuh)
        in.schemeCoeffHe = dHe.coeff;     // ...and so do the energy pair's; raw, for the same reason
        in.schemeCoeffKE = dKE.coeff;

        // THE LIMITER'S GRADIENT, resolved and CHECKED. OpenFOAM builds limitedLinear's limiter from
        // fvc::grad(lPhi) (LimitedScheme.C:56-59), which goes through the case's own gradSchemes under
        // `grad(e)` / `grad(Ekp)`. brae computes Gauss linear gradients and nothing else, so a case
        // resolving that key to anything else would get a limiter built from a different gradient --
        // and the scheme would carry the case's name while computing something else.
        //
        // Measured on validation/rhoLU at a developed state: swapping the limiter gradient from Gauss
        // linear to leastSquares moves the assembled energy diagonal by 9.1e-03 and its source by
        // 2.8e-03. That is a different discretisation, not an approximation, so it refuses. gasMixing
        // is the case this stops: it says `gradSchemes { default leastSquares; }`, and without this
        // check clearing the div-scheme blocker would have made it run and be quietly wrong.
        // THE LIMITER'S GRADIENT, resolved through the case's own gradSchemes. `Gauss linear` and
        // `leastSquares` are both computed (fvc::gaussGrad / fvc::leastSquaresGrad, and the matching
        // device kernels); anything else still refuses, because the limiter would then be built from a
        // gradient the case did not ask for. leastSquares is not a variation on Gauss linear -- measured
        // against OpenFOAM's own grad(T) on validation/rhoSST, the two differ by 2.5e-01 on the same
        // field, while each matches OpenFOAM's answer for its own scheme to ~1e-13
        // (tests/leastsquares_grad_vs_openfoam). leastSquares is computed AND reached by default: the
        // BRAE_LEASTSQUARES=1 opt-in that used to sit here existed because gasMixing/injectorPipe, the
        // case it unblocks, parted from OpenFOAM by U 1.2e-01 with the gradient matched on both sides.
        // That gap has since been named and closed -- fvPatch::delta()'s projection, limitedLinearV's
        // pre-updateCoeffs boundary, the corrected laplacian's own gradient scheme and the closures'
        // fvm::ddt -- and the case reads U 1.7e-12 / T 9.6e-13 / k 2.2e-12 restarted from OpenFOAM's
        // iteration 5 (tests/rho_gasmixing_vs_openfoam.sh). An opt-in left in place would be a refusal
        // whose reason no longer exists.
        // `force`: the field's OWN gradSchemes entry is also what correctedSnGrad::fullGradCorrection
        // resolves for a `corrected` laplacian (correctedSnGrad.C: mesh.gradScheme("grad(" + name + ')')),
        // so it is needed whether or not the div scheme is limitedLinear. Resolved only for the limiter,
        // an upwind-div case with a corrected laplacian ran its non-orthogonal correction off a Gauss
        // gradient under a leastSquares name -- the substitution this refusal exists to prevent.
        auto resolveLimiterGrad = [&](DivScheme sc, const std::string& fld, scalar& out, bool& lsq,
                                      bool force = false)
        {
            if (!force && sc != DivScheme::limitedLinear) return;
            const FieldGradScheme gs = parseFieldGradScheme(caseDir, fld);
            if (!gs.gaussLinear && !gs.leastSquares)
                throw std::runtime_error(
                    "rhoSimpleFoam buildStepInput: div(phi," + fld + ") is `Gauss limitedLinear`, whose "
                    "limiter OpenFOAM builds from fvc::grad(" + fld + ") through the case's gradSchemes "
                    "(LimitedScheme.C:56-59). This case resolves grad(" + fld + ") to `" + gs.raw +
                    "`. brae computes `Gauss linear` and `leastSquares` and nothing else. Measured on "
                    "validation/rhoLU at a developed state, swapping that gradient moves the assembled "
                    "energy diagonal by 9.1e-03 and its source by 2.8e-03 -- a different discretisation, "
                    "not an approximation. Refusing rather than running the limiter off the wrong "
                    "gradient.");
            // ...and a limiter brae does not apply (cellMDLimited, faceLimited, faceMDLimited) is refused
            // too, by name: the shared reader only WARNED and the gradient ran unlimited under the
            // case's scheme name -- tests/eeqn_limitedlinear_vs_openfoam.sh's cellMDLimited arm found it.
            if (!gs.unsupportedLimiter.empty())
                throw std::runtime_error(
                    "rhoSimpleFoam buildStepInput: gradSchemes resolves grad(" + fld + ") to `" + gs.raw +
                    "` with the `" + gs.unsupportedLimiter + "` limiter, which neither arm applies "
                    "(cellLimited is the one ported). Refusing rather than running the unlimited gradient "
                    "under the case's scheme name.");
            out = gs.cellLimitK;
            lsq = gs.leastSquares;
        };
        // THE GRADIENT linearUpwind / linearUpwindV NAMES in div(phi,U), for the momentum convection
        // correction: mesh.gradScheme(<name>) (linearUpwind.C:61-68), named-then-default. It is NOT
        // grad(U)'s own scheme (in.gradULimitK, which divDevRhoReff and correctedSnGrad take), and the
        // two used to be one coefficient -- see rhoUEqn_cpp.cu. Resolved strictly: the correction is
        // computed from a Gauss linear gradient with an optional cellLimited limiter, and a name that
        // resolves to anything else is refused rather than approximated by the nearest one brae has.
        if (in.schemeU == DivScheme::linearUpwind || in.schemeU == DivScheme::linearUpwindV)
        {
            const FieldGradScheme gl = parseNamedGradScheme(caseDir, dU.luGradName);
            if (!gl.gaussLinear || gl.leastSquares || !gl.unsupportedLimiter.empty())
                throw std::runtime_error(
                    "rhoSimpleFoam buildStepInput: div(phi,U) names `" + dU.luGradName + "` as the "
                    "gradient of its linearUpwind correction, and gradSchemes resolves that to `" +
                    gl.raw + "`. The momentum correction is built from a Gauss linear gradient, "
                    "optionally cellLimited, on both arms; refusing rather than building it from another.");
            in.gradULULimitK = gl.cellLimitK;
        }
        resolveLimiterGrad(in.schemeHe, f.heName, in.limGradHeK, in.limGradHeLeastSq,
                           /*force=*/in.correctedLaplacian);
        resolveLimiterGrad(in.schemeKE, keName,   in.limGradKEK, in.limGradKELeastSq);

        DeviceSimpleControls sctl;
        parseFvSchemesControls(caseDir, sctl);
        // laplacianSchemes: `corrected` and `limited 1` are the uncapped non-orthogonal correction;
        // `limited <psi>` with 0 < psi < 1 caps it per face (fv::limitedSnGrad, limiter =
        // min(psi*|snGrad|/((1 - psi)*|corr| + SMALL), 1)); `limited 0` is NO correction at all -- the
        // limiter is identically 0 -- which the incompressible V2 driver once mapped onto the full
        // correction. The parser reports both as nonOrth with the coefficient in nonOrthLimit (1.0
        // for `corrected`), so the three regimes are separated here. Until this landed the mirror
        // forwarded nonOrth alone: `limited 0.5` ran the uncapped correction under the limited name.
        // `limited 0` is a THIRD regime and not `orthogonal`: limitedSnGrad derives from correctedSnGrad,
        // so its implicit coefficients stay nonOrthDeltaCoeffs while the explicit correction is zeroed.
        // Measured, OpenFOAM's own answers on rhoBoxSym (4 degrees) at 20 iterations: `limited 0` is
        // U 1.3e-04 from `orthogonal` and 1.9e-03 from `corrected`. brae's laplacian takes one flag for
        // both halves and a limiter coefficient whose 0 means UNCAPPED, so this regime is not
        // representable; refused by name rather than mapped onto either neighbour.
        if (sctl.nonOrth && sctl.nonOrthLimit <= 0.0)
            throw std::runtime_error(
                "rhoSimpleFoam buildStepInput: laplacianSchemes asks for `limited 0`, which OpenFOAM's "
                "limitedSnGrad makes nonOrthDeltaCoeffs WITHOUT the explicit correction -- neither "
                "`orthogonal` nor `corrected` (U 1.3e-04 and 1.9e-03 from them on rhoBoxSym). The mirror "
                "represents only those two and the capped `limited <psi>`; refusing rather than running "
                "one of them under the case's name.");
        in.correctedLaplacian = sctl.nonOrth;
        in.snGradLimitCoeff   = (sctl.nonOrth && sctl.nonOrthLimit < 1.0) ? sctl.nonOrthLimit : 0.0;
        in.gradULimitK        = sctl.gradULimitK;
        in.gradKLimitK        = sctl.gradKLimitK;
        // grad(U)'s and grad(p)'s BASE scheme, each field's own gradSchemes entry, named-then-default:
        // leastSquares runs on the host (fvc::leastSquaresGrad, vector and scalar forms). The shared
        // parser above only WARNS on leastSquares and hands out the Gauss coefficient; the mirror resolves
        // the scheme itself, as it does for grad(k). Any other base scheme still falls to the parser's
        // reading and its warning -- recorded in PORT.md as open, not widened here.
        {
            const FieldGradScheme gU = parseFieldGradScheme(caseDir, "U");
            const FieldGradScheme gP = parseFieldGradScheme(caseDir, "p");
            // A limiter neither arm applies (cellMDLimited, faceLimited, faceMDLimited) is refused by
            // name rather than run unlimited under the case's scheme name -- the parse records it
            // (FieldGradScheme::unsupportedLimiter) and the shared reader only WARNED.
            for (const FieldGradScheme* g : {&gU, &gP})
                if (!g->unsupportedLimiter.empty())
                    throw std::runtime_error(
                        "rhoSimpleFoam buildStepInput: gradSchemes resolves `" + g->raw + "` with the `" +
                        g->unsupportedLimiter + "` limiter, which neither arm applies (cellLimited is the "
                        "one ported). Refusing rather than running the unlimited gradient under the case's "
                        "scheme name.");
            in.gradULeastSq = gU.leastSquares;
            in.gradPLeastSq = gP.leastSquares;
            in.gradPLimitK  = gP.cellLimitK;   // applied at every grad(p) consumer, snGrad's correction included
            // ...and the limiter's own field for `Gauss limitedLinear` on a VECTOR: magSqr(U), whose
            // gradient OpenFOAM resolves under `grad(magSqr(U))` (LimitFuncs.C:34-39).
            const FieldGradScheme gM = parseFieldGradScheme(caseDir, "magSqr(U)");
            in.gradMagSqrULeastSq = gM.leastSquares;
            in.gradMagSqrULimitK  = gM.cellLimitK;
            if (!gM.unsupportedLimiter.empty())
                throw std::runtime_error(
                    "rhoSimpleFoam buildStepInput: gradSchemes resolves grad(magSqr(U)) to `" + gM.raw +
                    "` with the `" + gM.unsupportedLimiter + "` limiter, which neither arm applies. Refusing "
                    "rather than running the unlimited gradient under the case's scheme name.");
        }
        {
            const DdtSchemeEntry ddt = parseDdtScheme(caseDir);
            in.ddtEuler = ddt.euler;
            if (ddt.euler)
            {
                if (!(f.deltaT > 0.0))
                    throw std::runtime_error("brae: ddtSchemes default Euler needs a positive deltaT in controlDict.");
                in.rDeltaT = 1.0 / f.deltaT;   // controlDict's LAST deltaT, as OpenFOAM's dictionary reads it
            }
        }
        // The ENERGY gradient limiters, which the parser has carried all along and this never forwarded:
        // gradHeLimitK is the cellLimited coefficient of the gradient the energy's linearUpwind NAMES
        // (OF's linearUpwind takes mesh.gradScheme(gradSchemeName_), e.g. aerofoilNACA0012's
        // `linearUpwind limited` -> `limited cellLimited Gauss linear 1`), else the grad(h|e) entry's;
        // gradKinLimitK the same for div(phi,K|Ekp), falling back to the energy's. Without them the
        // deferred correction ran an UNLIMITED gradient under a case that limits it -- on NACA that is
        // a first-iteration T of [233.71, 301.64] against OpenFOAM's [297.95, 298.01].
        in.gradHeLimitK       = sctl.gradHeLimitK;
        in.gradKELimitK       = sctl.gradKinLimitK;
        if (verbose)
            std::printf("  schemes: div(phi,U) lu=%d bounded=%d | div(phi,%s) lu=%d bounded=%d | "
                        "div(phi,%s) lu=%d bounded=%d | grad(U) cellLimited k=%g | grad(%s) k=%g | grad(%s) k=%g"
                        " | laplacian corrected=%d limited=%g\n",
                        (int)dU.linearUpwind, (int)dU.bounded,
                        f.heName.c_str(), (int)dHe.linearUpwind, (int)dHe.bounded,
                        keName.c_str(), (int)dKE.linearUpwind, (int)dKE.bounded,
                        (double)in.gradULimitK, f.heName.c_str(), (double)in.gradHeLimitK,
                        keName.c_str(), (double)in.gradKELimitK,
                        (int)in.correctedLaplacian, (double)in.snGradLimitCoeff);
    }

    // RELAXATION: "the case NAMES a factor" is OpenFOAM's predicate, and a `default` entry counts as
    // naming it. solution::relaxEquation(name) is eqnRelaxDict_.found(name) || found("default")
    // (solution.C:330-334) and relaxField(name) the same on fieldRelaxDict_ (solution.C:320-327);
    // fvMatrix::relax() (fvMatrix.C:1250-1263) and GeometricField::relax() (GeometricField.C:1099-1114)
    // relax if and only if that predicate holds, with the NAMED entry first (regex keys included --
    // dictionary.H:545-549 matches keyType::REGEX) and the default otherwise (solution.C:337-375,
    // :379-416). This read only the name, so a case relaxing through `default 0.7;` ran UNRELAXED on
    // both arms; the tutorials' `".*" 0.7;` idiom is a regex the dict already matched, which is why no
    // fixture saw it. A factor of 1 is still "named" (fvMatrix::relax(1) applies the dominance clamp).
    struct RelaxEntry
    {
        scalar factor     = 1.0;
        bool   named      = false;   // the OpenFOAM predicate: an entry for the name, or a default
        bool   viaDefault = false;
    };
    const auto relaxEntry = [](const FoamDict* d, const std::string& name) -> RelaxEntry
    {
        RelaxEntry e;
        if (d == nullptr) return e;
        if (d->found(name))
        {
            e.named  = true;
            e.factor = d->scalarOr(name, 1.0);
        }
        else if (d->found("default"))
        {
            e.named      = true;
            e.viaDefault = true;
            e.factor     = d->scalarOr("default", 1.0);
        }
        return e;
    };
    const RelaxEntry eU    = relaxEntry(re,  "U");
    const RelaxEntry eHe   = relaxEntry(re,  f.heName);
    const RelaxEntry ePEqn = relaxEntry(re,  "p");
    const RelaxEntry eK    = relaxEntry(re,  "k");
    const RelaxEntry eEps  = relaxEntry(re,  "epsilon");
    const RelaxEntry eOm   = relaxEntry(re,  "omega");   // kOmegaSST's second scalar
    const RelaxEntry fP    = relaxEntry(rfl, "p");
    const RelaxEntry fRho  = relaxEntry(rfl, "rho");

    in.relaxU             = eU.factor;
    in.relaxHe            = eHe.factor;
    in.relaxPEqn          = ePEqn.factor;
    in.relaxPEqnSpecified = ePEqn.named;
    // The field helpers treat a factor of 1 as "do nothing", which is what OpenFOAM does when the
    // predicate is false (p.relax() is never entered); so an unnamed field factor stays at 1.
    in.relaxP             = fP.factor;
    in.relaxRho           = fRho.factor;
    in.relaxK             = eK.factor;
    in.relaxEpsilon       = eEps.factor;
    in.relaxOmega         = eOm.factor;
    in.relaxEquationU     = eU.named;
    in.relaxEquationHe    = eHe.named;
    in.relaxEquationK     = eK.named;
    in.relaxEquationOmega = eOm.named;
    in.relaxEquationEps   = eEps.named;
    // SAID, not assumed (the V2 precedent): the factor each arm APPLIES, with where it came from, so a
    // default that silently stood in -- or one that silently did not -- is visible in the log.
    if (verbose)
    {
        const auto show = [](const RelaxEntry& e) -> std::string
        {
            if (!e.named) return "none";
            char buf[48];
            std::snprintf(buf, sizeof(buf), "%g%s", (double)e.factor, e.viaDefault ? " (default)" : "");
            return buf;
        };
        std::printf("  relaxation: equations U %s | %s %s | p %s | k %s | epsilon %s | omega %s ;"
                    " fields p %s | rho %s\n",
                    show(eU).c_str(), f.heName.c_str(), show(eHe).c_str(), show(ePEqn).c_str(),
                    show(eK).c_str(), show(eEps).c_str(), show(eOm).c_str(),
                    show(fP).c_str(), show(fRho).c_str());
    }

    // div(phi,k) and div(phi,epsilon|omega) FROM THE CASE. This was a hardcode, so neither the bounded
    // flag nor a non-upwind scheme ever reached the step from the case's own fvSchemes.
    if (f.turbulent && !f.turbulenceFrozen && !f.k.internal.empty())
    {
        const std::string secondT = (f.rasModel == "kOmegaSST") ? "omega" : "epsilon";

        // grad(k) and grad(epsilon|omega): EACH FIELD'S OWN gradSchemes entry, named-then-default.
        // That is the gradient correctedSnGrad's non-orthogonal correction takes
        // (`mesh.gradScheme("grad(" + vf.name() + ')')`, correctedSnGrad.C:52-55) on the corrected
        // turbulence laplacians, and the one kOmegaSST's CDkOmega is built from -- and it is what
        // gradKLimitK means to both closures.
        //
        // NOT sctl.gradKLimitK, which is wrong here twice over. The shared parser writes the gradient
        // linearUpwind NAMES in div(phi,epsilon|omega) into that same slot (scheme_parse.cuh, the line
        // marked EXPERIMENT), so a case writing `linearUpwind limited` had its laplacian corrections
        // cell-limited where OpenFOAM's are not: on squareBendLiq at iteration 1 the epsilon source
        // before relax() read 1.56e-04 off OpenFOAM's in 144 cells at the non-orthogonal block
        // junction, while the convection term itself was exact to 5.6e-15 -- and epsilon 2.6e-06 after
        // the solve. It also reads only explicit grad(k)/grad(epsilon) lines, so a cellLimited
        // `default` never reached the closures at all. The shared parser serves the incompressible
        // drivers too and is left as it is; the mirror resolves its own.
        {
            const FieldGradScheme gK = parseFieldGradScheme(caseDir, "k");
            const FieldGradScheme gS = parseFieldGradScheme(caseDir, secondT);
            // Only where a gradient of k or the second scalar is actually taken: the corrected
            // laplacian's correction (both models) and CDkOmega (SST). A case using neither is not
            // refused over a gradient scheme nothing reads.
            const bool used = in.correctedLaplacian || f.rasModel == "kOmegaSST";
            // Gauss linear or leastSquares, either optionally cellLimited: the two base schemes the host
            // closures compute (fvc::gaussGrad, fvc::leastSquaresGrad -- the latter gated against
            // OpenFOAM's own grad by tests/leastsquares_grad_vs_openfoam.sh). Anything else is refused.
            auto computed = [](const FieldGradScheme& gs)
            {
                return (gs.gaussLinear || gs.leastSquares) && gs.unsupportedLimiter.empty();
            };
            if (used && (!computed(gK) || !computed(gS)))
                in.turbDivUnsupported =
                    "a grad(k)/grad(" + secondT + ") scheme brae does not compute for the turbulence "
                    "laplacian correction and CDkOmega -- they resolve to `" + gK.raw + "` and `" + gS.raw +
                    "`, where the host closures take Gauss linear or leastSquares, optionally cellLimited";
            else if (used && (gK.cellLimitK != gS.cellLimitK || gK.leastSquares != gS.leastSquares))
                in.turbDivUnsupported =
                    "grad(k) and grad(" + secondT + ") with different schemes or cellLimited coefficients "
                    "(the closures carry one gradient scheme for both)";
            else
            {
                in.gradKLimitK  = gK.cellLimitK;
                in.gradKLeastSq = used && gK.leastSquares;
            }
        }

        const FieldDivScheme dK = parseFieldDivScheme(caseDir, "k");
        const FieldDivScheme dS = parseFieldDivScheme(caseDir, secondT);
        in.boundedTurb = dK.bounded;
        if (dK.bounded != dS.bounded)
            in.turbDivUnsupported = "bounded on one of div(phi,k)/div(phi," + secondT
                                  + ") and not the other (brae carries one flag for both)";
        // limitedLinear is ASSEMBLED (both closures take the weights), but only as one scheme for both
        // scalars -- the closures carry a single flag and coefficient, so entries that disagree refuse.
        if (dK.limited != dS.limited || (dK.limited && dK.coeff != dS.coeff))
            in.turbDivUnsupported = "limitedLinear on div(phi,k) and div(phi," + secondT
                                  + ") with different schemes or coefficients (brae carries one for both)";
        in.limitedLinearTurb = dK.limited && dS.limited;
        in.turbLimiterCoeff  = dK.coeff;   // RAW k of `limitedLinear k` -- see scheme_parse.cuh
        // linearUpwind: ONE scheme for both scalars, as limitedLinear is, and a gradient brae computes.
        // The gradient is the one the entry NAMES (`linearUpwind limited` -> gradSchemes `limited`), not
        // grad(k) -- OpenFOAM's linearUpwind builds it from mesh.gradScheme(gradSchemeName_)
        // (linearUpwind.C:61-68). The host closures take a Gauss linear gradient with an optional
        // cellLimited limiter; anything else the name resolves to is refused rather than approximated.
        if (dK.linearUpwind != dS.linearUpwind)
            in.turbDivUnsupported = "linearUpwind on one of div(phi,k)/div(phi," + secondT
                                  + ") and not the other (brae carries one scheme for both)";
        else if (dK.linearUpwind && dK.luGradName != dS.luGradName)
            in.turbDivUnsupported = "linearUpwind on div(phi,k) and div(phi," + secondT + ") naming "
                                  "different gradient schemes (`" + dK.luGradName + "`, `" + dS.luGradName
                                  + "`; brae carries one gradient for both)";
        else if (dK.linearUpwind)
        {
            const FieldGradScheme gl = parseNamedGradScheme(caseDir, dK.luGradName);
            if (!gl.gaussLinear || gl.leastSquares || !gl.unsupportedLimiter.empty())
                in.turbDivUnsupported =
                    "Gauss linearUpwind " + dK.luGradName + ", whose gradient resolves to `" + gl.raw
                    + "` -- the turbulence closures build linearUpwind's correction from a Gauss linear "
                    "gradient, optionally cellLimited, and nothing else";
            else
            {
                in.linearUpwindTurb = true;
                in.turbLUGradK      = gl.cellLimitK;
            }
        }

        // THE LIMITER'S GRADIENT, for the turbulence pair, on the same rule as the energy pair above:
        // OpenFOAM builds limitedLinear's limiter from fvc::grad(<field>) resolved through the case's
        // gradSchemes (LimitedScheme.C:56-59), so `grad(k)` and `grad(epsilon|omega)` decide it, and
        // brae computes Gauss linear and leastSquares. Both closures took a plain unlimited Gauss
        // gradient here regardless of what the case asked for, so a `grad(k) cellLimited Gauss linear 1`
        // was read into gradKLimitK, used for the corrected laplacian, and dropped for the limiter.
        //
        // leastSquares needed BRAE_LEASTSQUARES=1 here until the closures' divWithScheme took it
        // (kOmegaSST_cpp.cu, kEpsilon_cpp.cu, under gradKLeastSq). It is computed and gated now --
        // tests/rho_leastsquares_closure_vs_openfoam.sh's lsqko_komega arm, 8.7e-12 against real OpenFOAM
        // on validation/rhoSST restarted from its iteration 5 -- so the opt-in is gone from this guard.
        // The ENERGY limiter keeps its own opt-in: its end-to-end gap is a separate, open measurement.
        if (in.limitedLinearTurb)
        {
            const FieldGradScheme gK = parseFieldGradScheme(caseDir, "k");
            const FieldGradScheme gS = parseFieldGradScheme(caseDir, secondT);
            const bool kOk = (gK.gaussLinear || gK.leastSquares) && gK.unsupportedLimiter.empty();
            const bool sOk = (gS.gaussLinear || gS.leastSquares) && gS.unsupportedLimiter.empty();
            if (!kOk || !sOk)
                in.turbDivUnsupported =
                    "a limiter gradient brae does not compute -- div(phi,k)/div(phi," + secondT +
                    ") are `Gauss limitedLinear`, whose limiter OpenFOAM builds from fvc::grad of each "
                    "field through the case's gradSchemes (LimitedScheme.C:56-59), and this case "
                    "resolves them to `" + (kOk ? gS.raw : gK.raw) + "` where brae has `Gauss linear` "
                    "and `leastSquares`";
            // ONE coefficient AND ONE SCHEME for both, as the closures carry one limiter gradient for
            // both. Entries that disagree refuse rather than silently taking k's.
            else if (gK.cellLimitK != gS.cellLimitK)
                in.turbDivUnsupported =
                    "grad(k) and grad(" + secondT + ") name different cellLimited coefficients (brae "
                    "carries one limiter gradient for both)";
            else if (gK.leastSquares != gS.leastSquares)
                in.turbDivUnsupported =
                    "grad(k) and grad(" + secondT + ") name different gradient SCHEMES (brae carries one "
                    "limiter gradient for both, and running one field's limiter off the other's gradient "
                    "would be a substituted discretisation)";
            else
            {
                in.turbLimGradK      = gK.cellLimitK;
                in.turbLimGradLeastSq = gK.leastSquares;
            }
        }
    }

    return in;
}

namespace {

// The boundary values in the layout every writer expects: flat, patch order, COUPLED PATCHES EXCLUDED
// (foam_field_writer.cuh advances its offset only on non-coupled patches, because a cyclic's values live
// on the interface object). `empty` patches ARE in the layout even though the writer emits no value for
// them, so this must not skip them or every field after the first empty patch is written shifted.
template <typename T>
std::vector<T> flatBoundary(const GeometricField<T>& gf, const std::vector<FvPatch>& patches)
{
    std::vector<T> out;
    for (std::size_t pi = 0; pi < patches.size() && pi < gf.boundary.size(); ++pi)
    {
        if (isCoupledInterfaceType(patches[pi].type)) continue;
        const std::vector<T> v = gf.boundary[pi]->value();
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

// Per-patch value arrays that are not a GeometricField (a surface field's boundary, a model's stored
// patch values) into the same layout.
std::vector<scalar> flatPatchValues(const std::vector<std::vector<scalar>>& bnd,
                                    const std::vector<FvPatch>& patches)
{
    std::vector<scalar> out;
    for (std::size_t pi = 0; pi < patches.size() && pi < bnd.size(); ++pi)
    {
        if (isCoupledInterfaceType(patches[pi].type)) continue;
        out.insert(out.end(), bnd[pi].begin(), bnd[pi].end());
    }
    return out;
}

std::vector<scalar> flatSurfaceBoundary(const SurfaceScalarField& sf,
                                        const std::vector<FvPatch>& patches)
{
    return flatPatchValues(sf.boundary, patches);
}

} // namespace

int runMirror(const std::string& caseDir)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    setIOPrecision(controlDict.intOr("writePrecision", 6));   // OF TimeIO.C:375-383

    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    // The unread-entry safety net the legacy drivers have had since item E5, absent on the mirror until
    // queue item 15: an input this arm parses and never applies is reported at scope exit, on the
    // normal return AND on a refusal (marked PARTIAL there, since an entry may simply not have been
    // reached). Declared AFTER the dicts it points at, so it is destroyed first. fvSchemes is audited
    // through the shared consumption choke point, so it needs no instance here. What this cannot see:
    // thermophysicalProperties and turbulenceProperties are read as private copies inside createFields
    // (rhoCreateFields_cpp.cu), and an audit holds pointers -- queued as 15b.
    // thermophysicalProperties and the turbulence dictionary are read HERE and handed to createFields
    // (item 15b): FoamDict records the keys its consumers query, so the audit can only report on the
    // instance the reads happen on. turbulenceDictPath is createFields' own resolution of OpenFOAM's
    // two names for that dictionary; "" is a case with neither, which createFields refuses.
    const FoamDict thermoProps = readDict(caseDir + "/constant/thermophysicalProperties");
    const std::string turbPath = cpu::rhoSimple::turbulenceDictPath(caseDir);
    std::unique_ptr<FoamDict> turbProps;
    if (!turbPath.empty()) turbProps = std::make_unique<FoamDict>(readDict(turbPath));
    DictAuditScope audit;
    audit.add(controlDict, "system/controlDict");
    audit.add(fvSolution,  "system/fvSolution");
    audit.add(thermoProps, "constant/thermophysicalProperties");
    if (turbProps) audit.add(*turbProps, turbPath.substr(caseDir.size() + 1));
    audit.addFvSchemes(caseDir);
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    // `writeFormat binary` used to be refused here. It is a NOTICE now, emitted once from WriteControl
    // for every driver -- see the note there. The refusal claimed the output would not be what the case
    // asked for; measured, real OpenFOAM restarts from brae's ascii output with binary still set,
    // because the format is read from each FILE's header and never from controlDict.

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    // Time owns startFrom/latestTime resolution, the write cadence and the functionObjects, exactly as
    // OF's Time does -- none of it is a solver's business (OF's rhoSimpleFoam.C mentions none of it).
    // The scalarTransport factory is handed in here (OF's runtime selection table); the objects it
    // builds resolve the fields from `tracerCtx` at their first execute(), after createFields has
    // filled it -- Time has to come first, because the start directory is its to resolve.
    brae::tracer::TracerHostContext tracerCtx;
    std::vector<brae::tracer::RhoTracerHostFO*> tracers;
    std::vector<std::pair<std::string, FunctionObjectList::Factory>> foTypes;
    foTypes.emplace_back("scalarTransport", brae::tracer::rhoTracerHostFactory(caseDir, fvSolution, tracerCtx, tracers));
    Time time(caseDir, controlDict, foTypes);
    const std::string startName = time.startName();
    WriteControl& wc = time.writeControl();
    tracerCtx.fieldDir = caseDir + "/" + startName;

    // This driver's wall treatments read Cmu/kappa/E per patch (item 16h-port): the reader must not
    // announce those entries as unhonoured.
    brae::perPatchWallCoeffsHonoured() = true;
    RhoSimpleFields f = createFields(caseDir + "/" + startName, caseDir, simpleDict, &fvSolution,
                                     m, g, patches, &thermoProps, turbProps.get(), wc.startTime());
    tracerCtx.f = &f;
    tracerCtx.m = &m;
    tracerCtx.g = &g;
    tracerCtx.patches = &patches;

    std::printf("brae rhoSimpleFoam (OF-mirror): %ld cells, start %s, %s\n",
                (long)nC, startName.c_str(),
                f.turbulent ? (f.turbulenceFrozen ? (f.rasModel + " (frozen)").c_str()
                                                  : f.rasModel.c_str())
                            : "laminar");
    std::printf("  energy '%s', consistent %s, transonic %s\n", f.heName.c_str(),
                simpleDict && simpleDict->wordOr("consistent", "no") == "yes" ? "yes" : "no",
                simpleDict && simpleDict->wordOr("transonic", "no") == "yes" ? "yes" : "no");

    CaseRefusals refusals;
    StepInput in = buildStepInput(caseDir, f, fvSolution, m, refusals);

    // THE CASE'S OWN LINEAR-SOLVER TOLERANCES, which the harness deliberately does not read: a gate
    // pins them at 1e-12 so the linear solve is out of the brae-vs-OpenFOAM comparison, while a SOLVER
    // must run what the case asks for -- OF reads tolerance/relTol/maxIter per field from
    // fvSolution/solvers and a looser p tolerance is a different trajectory, not just a cheaper one.
    // This is the ONE deliberate difference between the gated configuration and the shipped one, made
    // in one place and named here rather than drifting apart silently.
    {
        DeviceSimpleControls lctl;
        // The reader's k/epsilon block is gated on this flag and nothing set it, so tolKE/relTolKE sat at
        // the struct defaults 1e-8/0 whatever fvSolution said -- and the ENERGY tolerance was then
        // copied from that same turbulence slot. Every equation now reads its own entry, as OF does.
        lctl.turbulent = f.turbulent;
        const std::string secondName = (f.rasModel == "kOmegaSST") ? "omega" : "epsilon";
        // The HOST arm solves every field through pbicgstab.cuh, which is a DILU-preconditioned
        // BiCGStab whatever the dict says -- pressure included, where the device arm runs an
        // AMG-preconditioned CG. So a case naming `diagonal` or `none` here IS being substituted, in the
        // opposite direction from the usual one, and the reader can only know that if it is told.
        SolverRunsAs runsAs;
        runsAs.alwaysDilu = true;
        runsAs.diluOnEnergy = true;
        runsAs.pSolver = "PBiCGStab";
        runsAs.pPrecon = "DILU";
        readLinearSolverControls(fvSolution, secondName, lctl, "SIMPLE", f.heName, runsAs);
        in.tolU    = lctl.tolU;    in.relTolU    = lctl.relTolU;    in.maxIterU    = lctl.maxIterU;    in.minIterU    = lctl.minIterU;
        in.tolP    = lctl.tolP;    in.relTolP    = lctl.relTolP;    in.maxIterP    = lctl.maxIterP;    in.minIterP    = lctl.minIterP;
        in.tolHe   = lctl.tolHe;   in.relTolHe   = lctl.relTolHe;   in.maxIterHe   = lctl.maxIterHe;   in.minIterHe   = lctl.minIterHe;
        in.tolTurb = lctl.tolKE;   in.relTolTurb = lctl.relTolKE;   in.maxIterTurb = lctl.maxIterKE;   in.minIterTurb = lctl.minIterKE;
        printLinearSolverControls(in, f.heName, secondName, f.turbulent);
    }

    // SIMPLE residualControl. OF's rule -- an empty dict never converges, and `achieved && checked` --
    // lives in the shared ResidualControl, whose dict lookup is regex-aware: every stock tutorial writes
    // its turbulence criteria as a pattern, e.g. `"(k|omega|e)" 1e-4`.
    ResidualControl resControl(simpleDict ? simpleDict->subDict("residualControl") : nullptr);
    std::printf("  residualControl=%s\n", resControl.active() ? "on" : "off");

    // endTime is ABSOLUTE, not a run length: OF's Time::run() tests `value() < endTime - 0.5*deltaT`,
    // so a case restarted at 10 with endTime 20 runs TEN more steps and finishes at 20. Looping
    // `iter <= endTime` from 1 would run twenty and finish at 30 -- silently changing the iteration
    // count, the write times, and any comparison of a restarted run against a continuous one.
    const scalar endTime = controlDict.scalarOr("endTime", 0.0);
    const scalar tStart  = wc.startTime();
    // OF Time::run tests `value() < endTime - 0.5*deltaT` and operator++ ACCUMULATES the value
    // (Time.C:785, :1067). std::lround on the quotient disagrees at ratio n + 0.5: measured, real
    // OpenFOAM runs 2 steps at startTime 0 / endTime 1 / deltaT 0.4 where lround gives 3.
    const long nSteps = openFoamNSteps(static_cast<double>(tStart),
                                       static_cast<double>(endTime),
                                       static_cast<double>(wc.deltaT()));
    if (nSteps < 1)
        throw std::runtime_error(
            "brae rhoSimpleFoam (mirror): controlDict endTime (" + std::to_string((double)endTime)
            + ") is not beyond the start time (" + startName + "): there is nothing to run. endTime is "
              "an ABSOLUTE time, not a number of iterations -- on a restart set it past the time you "
              "are restarting from.");
    time.setSteps(static_cast<int>(nSteps));
    f.deltaT = wc.deltaT();   // the continuity error is dt-scaled (incompressible/continuityErrs.H)

    const std::string wsrc = caseDir + "/" + startName + "/";
    const std::string second = (f.rasModel == "kOmegaSST") ? "omega" : "epsilon";

    auto writeTimeDir = [&](const std::string& tname)
    {
        const std::string outDir = caseDir + "/" + tname;
        std::filesystem::create_directories(outDir);
        // The SOLVED boundary values, not the start directory's. Echoing the template's boundary is the
        // gap the V2 writer has: a written field then carries the 0/ seed on every patch the solve
        // moved, and a restart from it is a restart from the wrong state.
        writeVolField(wsrc + "U", outDir + "/U", f.U.internal, patches, 12,
                      flatBoundary(f.U, patches));
        writeVolField(wsrc + "p", outDir + "/p", f.p.internal, patches, 12,
                      flatBoundary(f.p, patches));
        writeVolField(wsrc + "T", outDir + "/T", f.T.internal, patches, 12,
                      flatBoundary(f.T, patches));
        // scalarTransport's transportedField(), written beside the solved fields on the same cadence.
        for (const brae::tracer::RhoTracerHostFO* st : tracers)
            if (st->ready())
                writeVolField(wsrc + st->fieldName(), outDir + "/" + st->fieldName(), st->hostField(), patches, 12,
                              st->boundaryFlat());
        {
            // rho off the T template: 0/T supplies the FoamFile header shape only -- the identity, the
            // dimensions and every boundary entry are declared here, not inherited, or rho comes out as
            // `object T` with temperature dimensions and an inlet density of 300.
            static const DerivedFieldSpec rhoSpec{"rho", "dimensions      [1 -3 0 0 0 0 0];"};
            writeVolField(wsrc + "T", outDir + "/rho", f.rho.internal, patches, 12,
                          flatBoundary(f.rho, patches), &rhoSpec);
        }
        if (f.turbulent && !f.k.internal.empty())
        {
            writeVolField(wsrc + "k", outDir + "/k", f.k.internal, patches, 12,
                          flatBoundary(f.k, patches));
            const GeometricField<scalar>& sf = (second == "omega") ? f.omega : f.epsilon;
            if (!sf.internal.empty())
                writeVolField(wsrc + second, outDir + "/" + second, sf.internal, patches, 12,
                              flatBoundary(sf, patches));
            writeVolField(wsrc + "nut", outDir + "/nut", f.nut.internal, patches, 12,
                          flatBoundary(f.nut, patches));
            // alphat: OF's EddyDiffusivity registers it AUTO_WRITE, so OF writes it in every time
            // directory and a restart reads it back. No brae driver wrote it before this one.
            if (!f.alphat.internal.empty())
                writeVolField(wsrc + "alphat", outDir + "/alphat", f.alphat.internal, patches, 12,
                              flatBoundary(f.alphat, patches));
        }
        // generalizedNewtonian's nu_ is IOobject AUTO_WRITE under the name "generalizedNewtonian:nu"
        // (generalizedNewtonian.C:79-86), so OpenFOAM writes it in every time directory. Written here
        // under the same name, off the T template like rho, cells and patch values both.
        if (f.generalizedNewtonian)
        {
            static const DerivedFieldSpec nuSpec{"generalizedNewtonian:nu", "dimensions      [0 2 -1 0 0 0 0];"};
            writeVolField(wsrc + "T", outDir + "/generalizedNewtonian:nu", f.gnNu, patches, 12,
                          flatPatchValues(f.gnNuBnd, patches), &nuSpec);
        }
        // phi, so a restart RESUMES the conservative mass flux instead of rebuilding it from
        // interpolate(rho*U)&Sf -- a DIFFERENT field from the corrected phi. Compressible MASS flux
        // dimensions (kg/s), and 17 digits because this one is read back to seed a restart and wants an
        // exact double round-trip rather than display precision.
        writeSurfaceField(outDir + "/phi", f.phi.internal, flatSurfaceBoundary(f.phi, patches),
                          patches, 17, "[1 0 -1 0 0 0 0]");
        std::printf("written %s\n", outDir.c_str());
        wc.recordWritten(caseDir, tname);
    };

    int  nIter = static_cast<int>(nSteps);
    bool converged = false;
    int nStepsThisProcess = 0;   // StepInput::firstIteration: rho.oldTime() semantics
    while (time.loop())
    {
        const int iter = time.timeIndex();
        // The iteration's time value, already advanced by loop() as OpenFOAM's is when the body runs:
        // what a time-dependent boundary Function1 is evaluated at.
        in.time = time.timeValue();
        in.deltaT = time.deltaT();
        in.firstIteration = (nStepsThisProcess++ == 0);
        const Residuals r = rhoSimpleStep(f, in, m, g, patches);

        auto res = [&](const char* k) { return r.count(k) ? (double)r.at(k) : 0.0; };
        std::printf("Time = %s   U %.4e   %s %.4e   p %.4e",
                    WriteControl::timeName(wc.timeValue(iter)).c_str(),
                    res("U"), f.heName.c_str(), res(f.heName.c_str()), res("p"));
        if (r.count("k")) std::printf("   k %.4e   %s %.4e", res("k"), second.c_str(), res(second.c_str()));
        std::printf("\n");

        // OF evaluates EVERY criterion with no short circuit: each entry also has to be COUNTED, and its
        // `checked` flag is what stops an empty or unmatched dict from declaring convergence on nothing.
        // The continuity entries the step reports (contLocal/contGlobal/contCumulative) are diagnostics,
        // not solved fields, and are deliberately not offered to the control.
        resControl.beginIteration();
        bool achieved = resControl.ok(r.count("p") ? r.at("p") : scalar(0), "p");
        achieved = resControl.ok(r.count("U") ? r.at("U") : scalar(0), "U") && achieved;
        if (r.count(f.heName))
            achieved = resControl.ok(r.at(f.heName), f.heName) && achieved;
        if (r.count("k"))      achieved = resControl.ok(r.at("k"), "k") && achieved;
        if (r.count(second))   achieved = resControl.ok(r.at(second), second) && achieved;
        if (resControl.converged(achieved)) { converged = true; nIter = iter; time.stop(); break; }

        // Intermediate write. Time::writeTime() returns false on the last step -- that one is the final
        // write below, as in OF's writeAndEnd.
        if (time.writeTime()) writeTimeDir(WriteControl::timeName(wc.timeValue(iter)));
    }
    time.end();
    std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                          : "SIMPLE reached endTime (%d iterations)\n", nIter);

    // The final state is always written, as OF's writeAndEnd does, and named from the TIME VALUE rather
    // than the iteration count so a case with deltaT != 1 gets OpenFOAM's directory names.
    writeTimeDir(WriteControl::timeName(wc.timeValue(nIter)));
    std::printf("End\n");
    return 0;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
