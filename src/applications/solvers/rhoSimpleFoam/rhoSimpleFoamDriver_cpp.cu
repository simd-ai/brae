// rhoSimpleFoamDriver_cpp.cu -- see the header for what this is and why the parse is shared.
#include "rhoSimpleFoamDriver_cpp.cuh"

#include "brae_notice.cuh"
#include "brae_time.cuh"
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
        const FieldDivScheme dU  = parseFieldDivScheme(caseDir, "U");
        const FieldDivScheme dHe = parseFieldDivScheme(caseDir, f.heName);
        in.schemeU  = dU.linearUpwind  ? DivScheme::linearUpwind
                    : (dU.limited      ? DivScheme::limitedLinear : DivScheme::upwind);
        in.schemeHe = dHe.linearUpwind ? DivScheme::linearUpwind
                    : (dHe.limited     ? DivScheme::limitedLinear : DivScheme::upwind);
        // THE KINETIC-ENERGY TERM'S OWN ENTRY. EEqn.H builds fvc::div(phi, Ekp) on an e-thermo and
        // fvc::div(phi, K) on an h-thermo, and OpenFOAM resolves each under its own key, div(phi,Ekp) or
        // div(phi,K). This used to copy the energy entry with the note "follows the energy entry in
        // every tutorial" -- true of every tutorial and every fixture, which is exactly why a case that
        // separates the two was never seen: it ran the energy scheme on K under the case's own name.
        // Parsed like the others; a scheme the energy equation has not ported refuses there by name.
        const std::string keName = (f.heName == "e") ? "Ekp" : "K";
        const FieldDivScheme dKE = parseFieldDivScheme(caseDir, keName);
        in.schemeKE = dKE.linearUpwind ? DivScheme::linearUpwind
                    : (dKE.limited     ? DivScheme::limitedLinear : DivScheme::upwind);
        in.boundedU      = dU.bounded;
        in.boundedHe     = dHe.bounded;
        in.boundedKE     = dKE.bounded;
        in.schemeCoeffU  = dU.coeff;      // RAW k: the weights functions compute twoByk (scheme_parse.cuh)

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
        if (dK.linearUpwind || dS.linearUpwind) in.turbDivUnsupported = "Gauss linearUpwind";
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

std::vector<scalar> flatSurfaceBoundary(const SurfaceScalarField& sf,
                                        const std::vector<FvPatch>& patches)
{
    std::vector<scalar> out;
    for (std::size_t pi = 0; pi < patches.size() && pi < sf.boundary.size(); ++pi)
    {
        if (isCoupledInterfaceType(patches[pi].type)) continue;
        out.insert(out.end(), sf.boundary[pi].begin(), sf.boundary[pi].end());
    }
    return out;
}

} // namespace

int runMirror(const std::string& caseDir)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    // REFUSED, not silently downgraded: every writer in the tree emits ASCII and force-rewrites a binary
    // template header to `format ascii` (foam_field_writer.cuh). A case asking for binary output would
    // get ascii under its own setting -- readable by OpenFOAM, but not what the case asked for, and
    // silently larger and slower on the big meshes that ask for binary in the first place.
    if (controlDict.wordOr("writeFormat", "ascii") == "binary")
        throw std::runtime_error(
            "brae rhoSimpleFoam (mirror): controlDict writeFormat is `binary`, which brae's field "
            "writer does not emit -- it writes ASCII only. Refusing rather than writing ascii under a "
            "binary setting. Set `writeFormat ascii;` to run this case.");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    // Time owns startFrom/latestTime resolution, the write cadence and the functionObjects, exactly as
    // OF's Time does -- none of it is a solver's business (OF's rhoSimpleFoam.C mentions none of it).
    Time time(caseDir, controlDict);
    const std::string startName = time.startName();
    WriteControl& wc = time.writeControl();

    RhoSimpleFields f = createFields(caseDir + "/" + startName, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);

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
    const long nSteps = std::lround((static_cast<double>(endTime) - static_cast<double>(tStart))
                                    / static_cast<double>(wc.deltaT()));
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
    while (time.loop())
    {
        const int iter = time.timeIndex();
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
