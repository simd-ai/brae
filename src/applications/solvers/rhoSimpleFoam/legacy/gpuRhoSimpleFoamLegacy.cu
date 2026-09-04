// ===================================================================================================
// LEGACY SNAPSHOT -- frozen 2026-08-10. DO NOT DELETE, DO NOT "FIX".
//
// This is the GPU-first rhoSimpleFoam as it stood when the ground-truth rewrite was decided. It is
// kept building and runnable as brae_rhoSimpleFoam_legacy so the new CPU-first path always has
// something to diff against, and so the five validated tutorials keep a working reference.
//
// It validates on squareBend (~1e-06), angledDuct, squareBendLiq, squareBendLiqNoNewtonian, naca0012.
// It DIVERGES on gasMixing/injectorPipe. The cause is NOT known: the boundary-muEff defect that once
// looked like the cause was fixed and wired (184/184 green) and gasMixing still goes NaN at iteration
// 332 -- logged as retracted finding #7 in ../../../../../rhosimplefoam-ground-truth-port.md.
//
// INVARIANT FOR THE REWRITE: the new path must not edit device_simple_foam.{cu,cuh}. That file is
// shared with simpleFoam, which works, and is what keeps this snapshot stable. New code gets new files.
// ===================================================================================================

// brae_rhoSimpleFoam -- steady COMPRESSIBLE SIMPLE, single-GPU device-resident.
//
// Reads a standard OpenFOAM rhoSimpleFoam case and runs the whole loop on the GPU via
// DeviceSimpleSolver::rhoSimpleStep -- the same three composable phases the steady and PIMPLE solvers
// use, with the energy equation and the thermo update inserted:
//
//     UEqn -> EEqn -> pEqn -> thermo.correct() -> rho.relax() -> turbulence
//
// Scope today is SUBSONIC, perfectGas + hConst + (sutherland | const), laminar. Anything outside that is
// refused at start-up rather than run with the wrong physics -- see readThermoCoeffs, which rejects an
// unsupported thermoType by name, and the transonic check below.
//
//   brae_rhoSimpleFoam -case <caseDir>
//
// Note on p: this solver reads ABSOLUTE pressure in Pa, not the kinematic p/rho the incompressible
// solvers use. A case copied from simpleFoam with p in m2/s2 will run and give nonsense, so the
// dimensions line is checked.

#include "primitive_mesh.cuh"
#include "../common/read_surface_field.cuh"   // OF READ_IF_PRESENT for phi on restart
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "device_simple_foam.cuh"
#include "thermo_parse.cuh"
#include "rho_simple_controls.cuh"
#include "turbulence_setup.cuh"   // readTurbulenceModel + readTurbulenceFields (shared with simpleFoam/pimpleFoam)
#include "patch_entry_lookup.cuh"   // findPatchEntry: OF patch/group/regex resolution
#include "brae_notice.cuh"   // noticeIgnored/Approximated/Defaulted: never drop an input silently
#include "start_time.cuh"   // OF startFrom latestTime/firstTime (shared with simpleFoam)
#include "residual_control.cuh"   // OF simpleControl::criteriaSatisfied (shared with simpleFoam)
#include "dict_audit.cuh"   // name every dict entry brae read and then ignored
#include "write_control.cuh"   // OF writeControl/writeInterval/purgeWrite cadence (shared with simpleFoam)
#include "mrf_read.cuh"          // readCellZones (shared with the incompressible driver)
#include "fv_options.cuh"       // OF fv::options: the SAME framework the incompressible driver uses
#include "of_residual_log.cuh"   // BRAE_OF_LOG=1: OF-format per-solve residuals, for iteration-by-iteration diffing
#include "scheme_parse.cuh"        // parseFvSchemesControls
#include "linear_solver_setup.cuh" // readLinearSolverControls + readEnergySolverControls (shared with gpuSimpleFoam)
#include <cstdio>
#include <chrono>
#include <filesystem>   // READ_IF_PRESENT probe for rho
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <filesystem>
#include <cmath>
#include <algorithm>

using namespace brae;

int main(int argc, char** argv)
{
    try
    {
        // BRAE_SETUP_TIMING=1: elapsed seconds at each set-up milestone. Set-up is a long sequence of
        // one-off steps and when one hangs there is nothing in the log between the dictionary notices and
        // the first iteration to say which -- measured on gasMixing/injectorPipe, which sat in set-up for
        // over 12 minutes on a 74650-cell snappyHexMesh with no output at all.
        const auto t0__ = std::chrono::steady_clock::now();
        const bool timeSetup__ = (std::getenv("BRAE_SETUP_TIMING") != nullptr);
        auto mark__ = [&](const char* what)
        {
            if (!timeSetup__) return;
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0__).count();
            std::fprintf(stderr, "  [setup %7.2fs] %s\n", dt, what);
        };
        std::string caseDir = ".";
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "-case" && i + 1 < argc) caseDir = argv[++i];
        }

        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        mark__("controlDict read");
        const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
        mark__("fvSolution read");
        // Read here rather than at its point of use so the audit guard below can outlive it -- the guard
        // holds pointers and must be destroyed BEFORE the dicts it reports on.
        const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");

        // E5: report on EVERY exit, refusals included. Declared after all three dicts so it is destroyed
        // first. The two stock tutorials most worth auditing (aerofoilNACA0012, angledDuct) both refuse on
        // fvOptions, so before this they were the only cases never audited.
        DictAuditScope audit;
        audit.add(controlDict, "system/controlDict");
        audit.add(fvSolution, "system/fvSolution");
        audit.add(turbProps, "constant/turbulenceProperties");

        mark__("turbulenceProperties read");
        const ThermoCoeffs tc = readThermoCoeffs(caseDir, &fvSolution);
        mark__("thermo read");   // share the dict so dict_audit sees these lookups
        const RhoSimpleControls rc = readRhoSimpleControls(fvSolution, tc.internalEnergy);

        PrimitiveMesh m;
        m.read(caseDir + "/constant/polyMesh");
        mark__("mesh read");
        FvGeometry g;
        g.build(m);
        mark__("geometry built");
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        const label nC = m.nCells();

        // C6: startFrom was ignored here -- the start directory was hardcoded to "0", so `startFrom
        // latestTime` (the standard way to CONTINUE a compressible run) silently restarted from scratch
        // and then converged to a perfectly good answer, having discarded the restart.
        mark__("patches built");
        const std::string startName = resolveStartTime(
            caseDir,
            controlDict.wordOr("startFrom", "startTime"),
            controlDict.wordOr("startTime", "0"));
        const std::string t0 = caseDir + "/" + startName;
        // The ABSOLUTE time this run begins at. endTime is absolute too (OF's Time::run() tests
        // value() < endTime - 0.5*deltaT), so the run length is endTime - startTime -- see the loop.
        const scalar tStart = static_cast<scalar>(std::strtod(startName.c_str(), nullptr));
        GeometricField<vector> U = buildField<vector>(readField<vector>(t0 + "/U"), fvp, nC);
        GeometricField<scalar> p = buildField<scalar>(readField<scalar>(t0 + "/p"), fvp, nC);
        GeometricField<scalar> T = buildField<scalar>(readField<scalar>(t0 + "/T"), fvp, nC);
        U.evaluateBoundary();
        p.evaluateBoundary();
        T.evaluateBoundary();

        // Initial density, so the starting flux is a MASS flux like every later one. Without this the
        // first pressure equation sees a volumetric phiHbyA and the first iteration is inconsistent.
        //
        // OF createFields.H builds rho with IOobject::READ_IF_PRESENT:
        //     volScalarField rho(IOobject("rho", runTime.timeName(), mesh, READ_IF_PRESENT, AUTO_WRITE),
        //                        thermo.rho());
        // so a restart resumes the STORED density -- which carries rho.relax()'s history and is NOT
        // reproducible from p/(R*T). Only visible when rho is actually relaxed: with relaxRho = 1 the
        // stored and recomputed densities are identical, which is why every rho-1.0 duct in validation/
        // is blind to this. On naca0012 (rho 0.01), 5 iterations restarted from OF's 100/, reading vs
        // recomputing is worth 4131x on rho, 1381x on p, 583x on U, 155x on T -- validation/
        // restart_vs_openfoam.sh, which carries that comparison as its own negative control.
        //
        // Both HALVES matter: the internal field feeds th.rho (what phiHbyA interpolates) and the
        // boundary feeds rhoBndP_ (NOT rhoBnd_, which is thermo.rho() at the boundary and stays live).
        const std::string rhoPath = t0 + "/rho";
        const bool haveRhoFile = std::filesystem::exists(rhoPath);
        GeometricField<scalar> rhoIO;
        if (haveRhoFile) rhoIO = buildField<scalar>(readField<scalar>(rhoPath), fvp, nC);

        // OF createFields.H: `volScalarField rho(IOobject(..., READ_IF_PRESENT, ...), thermo.rho())`.
        // thermo.rho() -- NOT p/(R*T). They coincide only for a perfect gas; for `properties liquid` the
        // thermo density is the NSRDS rho(T) correlation and has no pressure dependence at all.
        //
        // THIS SEEDS TWO THINGS, AND BOTH WERE WRONG ON THE LIQUID PATH: the flowRateInletVelocity patch
        // (gSum(rho*magSf) below) and the initial mass flux fvc::rhoFlux(rho0, U, ...). squareBendLiq
        // ships no 0/rho, so rho0 fell back to the gas EOS and gave 1e5/(287.058*300) = 1.1612 kg/m3 --
        // air -- against water's 994.51. Measured as an inlet velocity, and hence a grad(U) at the 400
        // inlet cells, exactly 861.174x OF's, which is 1000/1.1612 to seven digits.
        std::vector<scalar> rho0(nC);
        if (haveRhoFile) rho0 = rhoIO.internal;
        else if (tc.model == ThermoModel::liquidH2O)
            for (label i = 0; i < nC; ++i)
                rho0[i] = H2OLiquid::rho(std::min(std::max(T.internal[i], H2OLiquid::Tt), H2OLiquid::Tc));
        else for (label i = 0; i < nC; ++i) rho0[i] = p.internal[i] / (tc.R * T.internal[i]);

        // Flat boundary half in DeviceBoundary order (patch order = DeviceMesh bndCell order), empty on
        // a fresh start so revalidateAfterThermo() takes OF's thermo.rho() fallback.
        std::vector<scalar> rhoBnd0;
        if (haveRhoFile)
            for (const auto& pf : rhoIO.boundary)
                rhoBnd0.insert(rhoBnd0.end(), pf->value().begin(), pf->value().end());

        // flowRateInletVelocity, seeded with the REGISTERED density -- OF's ordering, reproduced.
        //
        // createFields.H builds `rho` (line 12) BEFORE `U` (line 26), so when OF constructs the patch and
        // evaluate()s it, updateCoeffs() finds the registered rho and takes
        //     if (db().foundObject<volScalarField>(rhoName_)) updateValues(rhop);
        // i.e. the REAL patch density. `rhoInlet` is the OTHER branch, reached only when no rho field is
        // registered -- rhoSimpleFoam always registers one, so OF never uses it here. squareBend's
        // `rhoInlet 0.5` is even commented "Guess for rho", and OF ignores it.
        //
        // brae reads U before p and T, so at construction it had no density and fell back to rhoInlet.
        // Measured on squareBend: avgU 467.9 against OF's 611.7, ratio 1.3074, and the inlet momentum
        // boundaryCoeffs 0.765x OF's while every internalCoeff matched to 8 s.f.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (U.boundary[pi]->bcCategory() != 9) continue;          // 9 = mass-form flowRateInletVelocity
            const scalar mdot = U.boundary[pi]->flowRateValue();
            scalar sumRhoA = 0.0;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label c = fvp[pi].faceCells[i];
                sumRhoA += rho0[c] * fvp[pi].magSf[i];                // gSum(rho*magSf), OF updateValues()
            }
            if (sumRhoA <= 0.0) continue;
            const scalar avgU = -mdot / sumRhoA;
            std::vector<vector> v(fvp[pi].size);
            for (label i = 0; i < fvp[pi].size; ++i) v[i] = avgU * fvp[pi].nf[i];
            U.boundary[pi]->setValue(v);
        }

        // OF compressibleCreatePhi.H builds phi with IOobject::READ_IF_PRESENT, so a restart resumes the
        // stored conservative MASS flux; rhoFlux(rho0,U) is only OF's fresh-start fallback. Recomputing it
        // unconditionally began every restart from a flux the pressure equation had never corrected.
        const SurfaceScalarField phi = readPhiIfPresent(t0, fvp, m.nInternalFaces(),
                                                        fvc::rhoFlux(rho0, U, m, g, fvp));

        // Resolve pMaxFactor/pMinFactor into absolute limits the way OF does: the reference is taken from
        // the p BOUNDARY values (pressureControl.C scans p.boundaryField() over patches that fix a value).
        // Without a fixed-pressure patch OF errors out and tells the user to give pMax/pMin directly, so
        // brae does the same rather than inventing a reference from the internal field.
        {
            RhoSimpleControls& rcm = const_cast<RhoSimpleControls&>(rc);
            if (rcm.pMaxFactor > 0.0 || rcm.pMinFactor > 0.0)
            {
                // OF scans ONLY the patches that FIX a value (pressureControl.C:75-77:
                // `if (pbf[patchi].fixesValue())`), because pMaxFactor/pMinFactor scale a KNOWN reference
                // pressure -- a zeroGradient or calculated patch carries whatever the field currently
                // holds there, which is not a reference at all. brae scanned every patch, so on a case
                // with a non-uniform initial p the zeroGradient patches dragged the reference and the
                // limits came out different from OF's. Identical on a uniform initial p, which is why
                // every gate agreed.
                scalar pRefMax = -1e300;
                scalar pRefMin = 1e300;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    if (!p.boundary[pi]->fixesValue()) continue;
                    const std::vector<scalar>& bv = p.boundary[pi]->value();
                    for (scalar v : bv)
                    {
                        pRefMax = std::max(pRefMax, v);
                        pRefMin = std::min(pRefMin, v);
                    }
                }
                if (pRefMax <= -1e299)
                {
                    throw std::runtime_error(
                        "brae: SIMPLE/pMaxFactor or pMinFactor given, but NO pressure patch fixes a value, "
                        "so there is no reference pressure to scale. OpenFOAM refuses the same case "
                        "(pressureControl.C: \"pressure limits are not set\"). Specify absolute pMax/pMin "
                        "instead, or give the case a fixedValue pressure patch.");
                }
                if (rcm.pMaxFactor > 0.0) { rcm.pMaxLimit = pRefMax * rcm.pMaxFactor; rcm.limitMaxP = true; }
                if (rcm.pMinFactor > 0.0) { rcm.pMinLimit = pRefMin * rcm.pMinFactor; rcm.limitMinP = true; }
                std::printf("brae_rhoSimpleFoam: pressureControl limits p to [%g, %g] Pa\n",
                            rcm.limitMinP ? rcm.pMinLimit : -1e300,
                            rcm.limitMaxP ? rcm.pMaxLimit : 1e300);
            }
        }

        DeviceSimpleControls ctl;
        // The pressure needs a reference iff NO p patch fixes the value -- otherwise the all-Neumann
        // system is singular. gpuSimpleFoam has always done this scan; the compressible driver never did,
        // so needRef stayed at its default false and a closed compressible domain (all walls, or
        // fixedFluxPressure everywhere) would solve a singular pressure equation with no reference and
        // no adjustPhi. Same rule as the incompressible driver: a fixedValue p, or a freestreamPressure
        // (bcCategory 4, fixedValue on outflow), references the pressure.
        ctl.needRef = true;
        for (const auto& bf : p.boundary)
            if (bf->fixesValue() || bf->bcCategory() == 4)
            {
                ctl.needRef = false;
                break;
            }
        if (ctl.needRef)
        {
            // rc already carries these, read from the SIMPLE sub-dict by readRhoSimpleControls.
            ctl.pRefCell = rc.pRefCell;
            ctl.pRefValue = rc.pRefValue;
        }
        // ctl.nu is the KINEMATIC viscosity [m^2/s]; tc.mu0 is DYNAMIC [Pa s]. Assigning one to the other
        // was a units error off by rho -- ~1.2x for air at STP, but rho spans 0.87..1.16 even on the small
        // heated duct and far more on a real compressible case.
        //
        // The old comment said "only the seed matters", and that was true until E5: with `turbulence on`
        // the per-iteration solve overwrites nut from th_.mu, so the bad seed washes out in one iteration
        // and no gate could see it. With OF's `RAS { turbulence off; }` the model never runs, so the
        // startup correctNut IS the answer -- and its F2 (arg2 = max(2*sqrt(k)/(betaStar*omega*y),
        // 500*nu/(y^2*omega))) then uses mu as nu, activating the Bradshaw limiter where OF leaves it
        // inactive. Measured on rhoTI with the switch off: 40 inlet-column cells wrong by 75%.
        //
        // Seeded from the case's OWN initial state rather than a constant. Still a single scalar where OF
        // has a field -- validateTurbulence() runs before the thermo is seeded, so a per-cell nu is not
        // available there. Tracked as E7.
        {
            scalar rhoMean = 0.0;
            for (label i = 0; i < nC; ++i) rhoMean += p.internal[i] / (tc.R * T.internal[i]);
            rhoMean = (nC > 0) ? rhoMean / static_cast<scalar>(nC) : scalar(1);
            ctl.nu = tc.mu0 / rhoMean;
        }
        // B1: the transonic branch (phid, the phiHbyA subtraction, implicit fvm::div(phid,p) folded into
        // the pressure matrix, pEqn.relax(), and a BiCGStab solve because the resulting matrix is not
        // symmetric). It ran behind a refusal until it could be shown right on a case that discriminates.
        //
        // The refusal is now LIFTED, on this evidence:
        //   * squareBend -- the actual tutorial B1 exists for, Mach ~0.96, transonic AND consistent
        //     (SIMPLEC) -- converges in 160 iterations against OF's 156 and agrees to
        //     p 1.8e-03, U 1.4e-03, T 6.0e-04, rho 1.6e-03, k 7.4e-03, epsilon 8.3e-03, nut 4.9e-03.
        //     Gate `transonic_vs_openfoam`.
        //   * a low-Mach compressible duct converges in 105 against OF's 104 (p 7.0e-09, T 1.8e-07).
        //     That case does NOT discriminate -- the subsonic branch gives the same answer to 1e-11 --
        //     so it only shows the branch does not BREAK a subsonic case. It is not the evidence here.
        //
        // What actually made squareBend work was NOT a transonic fix. It diverged (contGlobal -2.83e+02
        // at iteration 1, NaN by 50) because of defects in the SHARED compressible path -- most recently
        // the thermo-type-dependent rho update below. The transonic assembly had been right for a while
        // behind an unrelated bug, which is the argument for fixing the chain before trusting a branch.
        const_cast<RhoSimpleControls&>(rc).rhoLagsPressure = tc.rhoThermoType;   // heRhoThermo rho timing
        ctl.transonic = rc.transonic;   // B1: OF pEqn.H transonic branch (guarded above)
        parseFvSchemesControls(caseDir, ctl);

        // Turbulence, through the SAME readers simpleFoam and pimpleFoam use, so a compressible case gets
        // exactly the model selection, coefficient set and wall-function guards an incompressible one does.
        // (turbProps itself is read above, next to the other dicts, so the audit guard can hold it.)
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType != "RAS" && simType != "laminar")
            throw std::runtime_error("brae: unsupported simulationType '" + simType + "' for rhoSimpleFoam (RAS or laminar)");
        ctl.turbulent = (simType == "RAS");
        readTurbulenceModel(turbProps, ctl);
        // kOmegaSST and kEpsilon are both rho-weighted (every RHS term, the diffusivity, the volumetric
        // divU and the per-face wall nu). SA and the kOmegaSST variants are not, so they stay refused:
        // running one down the incompressible path converges to a wrong answer rather than failing.
        // Standard kEpsilon = not SST, not SA, not realizableKE. realizableKE stays refused because its
        // epsilon reaction is a DIFFERENT expression (deviceEpsReactionRealizable, strain-based) that has
        // NOT been rho-weighted -- accepting it here would run an unweighted reaction and converge wrong.
        const bool keStandard = !ctl.sst && !ctl.sa && !ctl.keCoeffs.realizable;
        if (ctl.turbulent && !ctl.sst && !keStandard)
            throw std::runtime_error(
                "brae: rhoSimpleFoam supports kOmegaSST and kEpsilon so far. SpalartAllmaras and "
                "realizableKE are not rho-weighted, and running one down the incompressible path gives a "
                "converged wrong answer, so they are refused instead.");

        // div(phi,h|e) linearUpwind is HONOURED: measured against OF on the heated duct it agrees to
        // 8.2e-7, i.e. as well as upwind does. An earlier measurement said otherwise (7.1e-2) and was
        // wrong -- it used a one-line fvSchemes, which the old line-based parser let leak the energy
        // scheme onto div(phi,U). The parser now splits on statements, so layout cannot do that.
        //
        // Turbulence-scalar linearUpwind is HONOURED here. That opt-out was inherited from gpuSimpleFoam
        // on an unmeasured claim ("linearUpwind degrades k/omega vs OF"); measured on this path it is the
        // DOWNGRADE that is the error. validation/luturb_vs_openfoam.sh (the rhoSST duct with
        // div(phi,k|omega) = linearUpwind) against OF v2412, converged fields:
        //     honoured : k 2.0e-06   omega 3.2e-06   nut 8.1e-07
        //     upwind   : k 6.8e-03   omega 8.9e-03   nut 1.6e-02
        // so running upwind against a case that asked for linearUpwind cost 1.6% on nut.
        //
        // gpuSimpleFoam keeps its guard for a DIFFERENT reason -- pitzDaily SST diverges from a cold start
        // with linearUpwind on k -- and that is a convergence-path problem, not this discretisation: ONE
        // iteration from OF's own converged pitzDaily state agrees to 1.6e-06 on k. See the note there.
        // BRAE_SCALAR_LINEARUPWIND=0 forces upwind as an escape hatch.
        if (const char* luEnv = std::getenv("BRAE_SCALAR_LINEARUPWIND"))
        {
            if (std::atoi(luEnv) == 0) { ctl.luK = false; ctl.luEps = false; }
        }

        // fvOptions, read through the SAME framework the incompressible driver uses (readFvOptions +
        // solver.setFvOptions). OF's fv::options is not a per-solver feature: every solver calls the same
        // three hooks -- fvOptions(...) as an equation source, fvOptions.constrain(eqn), and
        // fvOptions.correct(field) -- and rhoSimpleFoam differs from simpleFoam only in that its momentum
        // source is rho-weighted (fvOptions(rho, U) vs fvOptions(U)) and it has energy hooks as well.
        // So this reuses the framework rather than adding a compressible copy of it.
        //
        // Anything the reader recognises but cannot apply still REFUSES: the reason the compressible driver
        // used to reject the whole file was that dropping a source silently converges to a different
        // problem, and that argument is unchanged for the sources brae does not implement.
        std::map<std::string, std::vector<label>> fvoZones;
        {
            std::ifstream fa(caseDir + "/system/fvOptions"), fb(caseDir + "/constant/fvOptions");
            if (fa.good() || fb.good()) fvoZones = readCellZones(caseDir + "/constant/polyMesh");
        }
        const FvOptionsData fvo = readFvOptions(caseDir, fvoZones, g.V(), nC, g.C());
        if (!fvo.unsupported.empty())
        {
            std::string msg = "brae: fvOptions contains source(s) brae cannot apply (they would be SILENTLY "
                              "dropped -> wrong physics):";
            for (const auto& u : fvo.unsupported) msg += "\n  - " + u;
            msg += "\nRemove/disable them, or use a supported form. Supported on the compressible solver: "
                   "limitTemperature, explicitPorositySource[DarcyForchheimer|fixedCoeff], "
                   "vectorSemiImplicitSource, limitVelocity, velocityDampingConstraint; "
                   "selectionMode all|cellZone.";
            throw std::runtime_error(msg);
        }

        const std::string second = ctl.sst ? "omega" : "epsilon";
        readRelaxationFactors(fvSolution, ctl);   // shared; adds the alpha<=0 guard this copy lacked

        // fvSolution -> ctl, through the SHARED reader. This driver previously read only tolP/tolU/tolKE
        // by hand and silently dropped relTol{P,U,KE}, consistent (SIMPLEC), nNonOrthogonalCorrectors,
        // the smoothSolver selection and the perf knobs -- see linear_solver_setup.cuh.
        readLinearSolverControls(fvSolution, second, ctl);
        const EnergySolverControls eSolve = readEnergySolverControls(fvSolution, tc.internalEnergy);

        TurbulenceFields tf;
        if (ctl.turbulent) tf = readTurbulenceFields(t0, fvp, nC, ctl, second, U);

        // Per-boundary-face Prt from 0/alphat. OF keeps two DIFFERENT turbulent Prandtl numbers in one
        // case: alphatWallFunction reads its own from the patch (default 0.85) while the turbulence model
        // uses the one from its coeffs dict (default 1.0). Using either one everywhere is wrong somewhere.
        std::vector<scalar> prtFace;
        if (ctl.turbulent)
        {
            FieldData<scalar> alphatFd;
            bool haveAlphat = true;
            try { alphatFd = readField<scalar>(t0 + "/alphat"); }
            catch (const std::exception&) { haveAlphat = false; }

            for (const FvPatch& q : fvp)
            {
                if (isCoupledInterfaceType(q.type)) continue;   // DeviceBoundary skips these
                scalar prt = tc.Prt;   // the MODEL's Prt away from an alphat wall function
                // OF-style resolution (exact name, then group, then regex) -- NOT `pb.name == q.name`.
                // squareBend* key this entry as "(?i).*walls" against a patch literally called `walls`,
                // so exact matching missed it and Prt silently reverted to the model default 1.0 instead
                // of the wall function's 0.85: wall alphat, and the wall heat flux, ~15% low.
                if (haveAlphat)
                {
                    const PatchFieldData<scalar>* pb = findPatchEntry(alphatFd.boundary, q);
                    if (pb && (pb->type == "compressible::alphatWallFunction" || pb->type == "alphatWallFunction"))
                        prt = pb->Prt;
                }
                prtFace.insert(prtFace.end(), static_cast<std::size_t>(q.size), prt);
            }
        }

        const int endTime = static_cast<int>(controlDict.scalarOr("endTime", 1000));

        mark__("fields+controls read");
        DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl,
                                  ctl.turbulent ? &tf.k : nullptr,
                                  ctl.turbulent ? &tf.eps : nullptr,
                                  ctl.turbulent ? &tf.nut : nullptr);

        // he boundary: built from the case's 0/T, then converted. brae never reads a 0/he, exactly as OF
        // never asks a user to write one.
        DeviceBoundary dbT = buildDeviceBoundary(T, fvp, g);
        DeviceBoundary dbHe = buildDeviceBoundary(T, fvp, g);
        // The liquid energy form needs p at the FACES (e = h(T) - p/rho(T)); the gas form ignores it.
        // Flattened in DeviceBoundary order (patch order = DeviceMesh bndCell order), as for rhoBnd0.
        DeviceBuffer<scalar> pBndD;
        if (tc.model == ThermoModel::liquidH2O)
        {
            std::vector<scalar> pb;
            for (const auto& pf : p.boundary)
                pb.insert(pb.end(), pf->value().begin(), pf->value().end());
            pBndD.copyFrom(pb);
        }
        deviceEnergyBoundaryFromT(dbT, tc, dbHe, pBndD.size() ? &pBndD : nullptr);
        // OF re-evaluates the turbulent-inlet BCs every updateCoeffs; give the solver the per-face
        // masks so it refreshes them each iteration instead of freezing the set-up value.
        solver.setTurbulentInlets(tf.turbInletMasks.tiMask, tf.turbInletMasks.tiIntensity,
                                  tf.turbInletMasks.mlMask, tf.turbInletMasks.mlLength);
        // OF fvPatchField::fixesValue(), captured BEFORE any per-face inletOutlet resolution: bcType 1 is
        // fixedValue, and only there is T prescribed rather than derived. Those faces must keep T_b exact
        // and have he_b rebuilt from the live p_b every correct() -- for a liquid under
        // sensibleInternalEnergy, e = h(T) - p/rho(T), so an he_b converted once at set-up goes stale the
        // moment the pressure moves.
        //
        // NOTE mixedFvPatchField::fixesValue() is also true in OF, so inletOutlet counts as
        // temperature-authoritative there too; brae still blends that patch in he-space and inverts. The
        // difference is second order (the blend is between two nearby temperatures) and is NOT covered
        // here -- only bcType 1 is claimed.
        {
            const std::vector<label>  tType = dbT.bcType.host();
            const std::vector<scalar> tRef  = dbT.refValue.host();
            if (tType.size() == tRef.size())
            {
                std::vector<label> fixMask(tType.size());
                for (std::size_t i = 0; i < tType.size(); ++i) fixMask[i] = (tType[i] == 1) ? 1 : 0;
                solver.setFixedBoundaryT(fixMask, tRef);
            }
        }
        mark__("solver constructed");
        solver.setCompressible(tc, rc, std::move(dbHe));
        mark__("setCompressible done");
        solver.setAlphatPrt(prtFace);
        // Hand the parsed options to the solver. Reading them and NOT doing this is precisely the silent
        // drop the old refusal existed to prevent -- and it is what happened on the first attempt here:
        // the file parsed, the case ran, and limitTemperature with min=280/max=300 let T reach 314.5.
        if (!fvo.empty())
        {
            solver.setFvOptions(fvo);
            std::printf("  fvOptions: %d source(s)%s%s%s%s%s\n", fvo.count,
                        fvo.limTActive ? " limitTemperature" : "",
                        fvo.fixTActive ? " fixedTemperatureConstraint" : "",
                        fvo.porActive  ? " explicitPorositySource" : "",
                        fvo.hasMomentum ? " momentum" : "",
                        fvo.limUActive ? " limitVelocity" : "");
        }
        solver.setEnergySolver(eSolve.tol, eSolve.relTol, eSolve.useGS);

        // Seed the thermo: T from the case, he from T, then one thermo.correct() so rho/psi/mu/alpha are
        // consistent before the first momentum predictor, and rhoPrev so the first relax has a partner.
        DeviceThermo& th = solver.thermo();
        th.T.copyFrom(T.internal);
        deviceThermoHeFromT(th, tc, &solver.pDevice());
        deviceThermoCorrect(th, solver.pDevice(), tc);
        // OF createFields.H: `volScalarField rho(IOobject("rho", ...), thermo.rho())`. rho is initialised
        // from thermo.rho() as its OWN statement, because hePsiThermo::calculate never writes a rho --
        // psiThermo has no rho_ field at all. Folding this into the correct() call is what the old
        // deviceThermoUpdate did, and it is exactly the conflation that made picking `updateRho` a guess.
        // Seed the THERMO's rho_ too: heRhoThermo::calculate() would normally have written it, but the
        // first correct() above ran before any pressure solve, so both densities start from the same
        // state -- exactly as OF's createFields.H leaves them.
        deviceThermoRho(th, solver.pDevice(), tc, th.rho);
        // READ_IF_PRESENT: the stored field overrides thermo.rho() on a restart. Written AFTER the call
        // above so the thermo's own rho_ still starts from the correct()ed state -- OF keeps `rho` and
        // `thermo.rho()` as two distinct fields and only the former is read back.
        if (haveRhoFile) th.rho.copyFrom(rhoIO.internal);
        // BRAE_DUMP_INIT=<dir>: the INITIALIZED thermo state, before any equation is solved. Step 6 of
        // the liquid bring-up -- "the solver started" is not the same claim as "the solver started from
        // OpenFOAM's state", and the difference is what decides whether an iteration-1 failure is the
        // equation or the initial condition.
        if (const char* initDir = std::getenv("BRAE_DUMP_INIT"))
        {
            std::error_code iec;
            std::filesystem::create_directories(initDir, iec);
            auto dumpv = [&](const char* nm, const std::vector<scalar>& v)
            {
                std::ofstream o(std::string(initDir) + "/" + nm);
                o.precision(17);
                o << "n " << v.size() << " 1\n";
                for (scalar x : v) o << x << '\n';
            };
            dumpv("init_T",   th.T.host());
            dumpv("init_he",  th.he.host());
            dumpv("init_p",   solver.pDevice().host());
            dumpv("init_rhoThermo", th.rhoThermo.host());
            dumpv("init_rhoSolver", th.rho.host());
            if (th.CpField.size()) dumpv("init_Cp", th.CpField.host());
            if (th.kappa.size())   dumpv("init_kappa", th.kappa.host());
            dumpv("init_mu",    th.mu.host());
            dumpv("init_alpha", th.alpha.host());
            {   // grad(U) on the initial U: same input OF's `postProcess -func grad(U) -time 0` sees.
                const std::vector<scalar> gu = solver.gradUField();
                std::ofstream o(std::string(initDir) + "/init_gradU");
                o.precision(17);
                o << "n " << gu.size() << " 1\n";
                for (scalar x : gu) o << x << '\n';
            }
            dumpv("init_Tb",   solver.TBoundary());
            dumpv("init_rhob", solver.rhoBoundary());
            // THE PATCH MANIFEST, without which the boundary dumps above are just a flat list of numbers.
            // A comparison script had to guess where each patch started, and a wrong guess degraded to
            // comparing NOTHING and reporting perfect agreement -- which is exactly what happened to
            // step 6's "boundary matches OF to 0.00e+00". Emitting name and face count per patch, in the
            // same order and with the same cyclic/cyclicAMI exclusions DeviceBoundary applies, lets the
            // gate assert it found every patch it expected and compared every face it found.
            {
                std::ofstream o(std::string(initDir) + "/init_patches");
                o << "# name nFaces type   (DeviceBoundary order; cyclic/cyclicAMI excluded)\n";
                for (const auto& q : fvp)
                {
                    if (isCoupledInterfaceType(q.type)) continue;
                    o << q.name << ' ' << q.size << ' ' << q.type << '\n';
                }
            }
            {   // cell centres, so a failing-cell list can be located in space without another tool
                const std::vector<vector>& CC = g.C();
                std::ofstream o(std::string(initDir) + "/init_C");
                o.precision(17);
                o << "n " << CC.size() << " 3\n";
                for (const vector& v : CC) o << v.x << ' ' << v.y << ' ' << v.z << '\n';
            }
            std::printf("brae: wrote initialized thermo state to %s\n", initDir);
        }
        deviceRhoSeedPrev(th);
        // E7: OF validates the turbulence model AFTER the thermo is constructed; brae's solver ctor did it
        // before, so correctNut ran against a placeholder inlet density. Redo it now that rho is real.
        mark__("pre-revalidate");
        solver.revalidateAfterThermo(haveRhoFile ? &rhoBnd0 : nullptr);

        // What brae RESOLVED from the case, not what the dict says. A relaxation factor silently left at
        // 1.0 is invisible in the fields and fatal on a stiff case.
        std::printf("brae_rhoSimpleFoam: solve  relTol p=%g U=%g k/%s=%g e|h=%g   tol p=%g U=%g e|h=%g   "
                    "GS U=%d k=%d %s=%d e|h=%d   SIMPLEC=%d nNonOrth=%d\n",
                    ctl.relTolP, ctl.relTolU, second.c_str(), ctl.relTolKE, eSolve.relTol,
                    ctl.tolP, ctl.tolU, eSolve.tol,
                    (int)ctl.gsU, (int)ctl.gsK, second.c_str(), (int)ctl.gsEps, (int)eSolve.useGS,
                    (int)ctl.consistent, ctl.nNonOrth);
        std::printf("brae_rhoSimpleFoam: relax  p=%g U=%g e|h=%g k=%g %s=%g rho=%g   pLimit=[%g, %g]\n",
                    ctl.relaxP, ctl.relaxU, rc.relaxHe, ctl.relaxK, second.c_str(), ctl.relaxEps, tc.relaxRho,
                    rc.limitMinP ? rc.pMinLimit : -1.0, rc.limitMaxP ? rc.pMaxLimit : -1.0);
        mark__("revalidate done");
        std::printf("brae_rhoSimpleFoam: %ld cells, subsonic %s, R=%.3f Cp=%.1f\n",
                    (long)nC, ctl.turbulent ? (ctl.sst ? "kOmegaSST" : "kEpsilon") : "laminar", tc.R, tc.Cp);

        // C5: SIMPLE residualControl. The compressible driver had none at all -- it always ran to endTime,
        // so a case asking to stop at p 1e-4 burned every remaining iteration and, more importantly, brae
        // reported a different iteration count than OF for the same input while claiming to run the same
        // case. OF's rule (empty dict never converges; `achieved && checked`) lives in the shared
        // ResidualControl, and its dict lookup is regex-aware, which matters here: every stock tutorial
        // writes its turbulence criteria as a pattern, e.g. `"(k|omega|e)" 1e-4`.
        const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");
        ResidualControl resControl(simpleDict ? simpleDict->subDict("residualControl") : nullptr);
        // OF names the energy field by the case's own energy variable; the solve reports it as "he".
        const std::string heName = tc.internalEnergy ? "e" : "h";
        std::printf("  residualControl=%s\n", resControl.active() ? "on" : "off");

        // OF controlDict write cadence. dict_audit found that this driver read NONE of writeControl /
        // writeInterval / purgeWrite / deltaT / stopAt: it wrote exactly one time directory, at the end,
        // so a case asking to write every N iterations silently got nothing until convergence. The policy
        // is shared with gpuSimpleFoam (write_control.cuh); only the payload below is solver-specific.
        WriteControl wc(controlDict);
        // Time values are measured from where this run actually STARTS, which `startFrom latestTime`
        // can make different from controlDict's startTime.
        wc.setStartTime(tStart);
        const std::string wsrc = t0 + "/";
        auto writeTimeDir = [&](const std::string& tname)
        {
            const std::string outDir = caseDir + "/" + tname;
            std::filesystem::create_directories(outDir);
            writeVolField(wsrc + "U", outDir + "/U", solver.U(), fvp, 12, solver.UBoundary());
            writeVolField(wsrc + "p", outDir + "/p", solver.p(), fvp, 12, solver.pBoundary());
            {
                std::vector<scalar> Tout(nC);
                th.T.copyTo(Tout);
                writeVolField(wsrc + "T", outDir + "/T", Tout, fvp, 12, solver.TBoundary());
            }
            {
                // rho, so the gate can compare the EOS result directly rather than inferring it from p and T.
                // 0/T is only a TEMPLATE for the FoamFile header here -- the identity, the dimensions and every
                // boundary entry are declared, not inherited. Written from T alone, rho came out as `object T`,
                // dimensions of temperature, an inlet density of 300 and (once B5 landed) a fixedGradient
                // density of 20000 kg/m^4, all of which OF reads back without complaint.
                static const DerivedFieldSpec rhoSpec{"rho", "dimensions      [1 -3 0 0 0 0 0];"};
                std::vector<scalar> rhoOut(nC);
                th.rho.copyTo(rhoOut);
                writeVolField(wsrc + "T", outDir + "/rho", rhoOut, fvp, 12, solver.rhoBoundary(), &rhoSpec);
            }
            if (ctl.turbulent)
            {
                writeVolField(wsrc + "k", outDir + "/k", solver.k(), fvp, 12, solver.kBoundary());
                // the 2nd turbulence scalar shares one slot: omega on kOmegaSST, epsilon on kEpsilon
                writeVolField(wsrc + second, outDir + "/" + second, solver.eps(), fvp, 12, solver.epsBoundary());
                writeVolField(wsrc + "nut", outDir + "/nut", solver.nut(), fvp, 12, solver.nutBoundary());
            }
            // phi, so a restart RESUMES the conservative mass flux instead of rebuilding it. OF's
            // compressibleCreatePhi.H pairs READ_IF_PRESENT with AUTO_WRITE; brae read it (see
            // read_surface_field.cuh) but never wrote it, so a brae->brae restart always fell back to
            // interpolate(rho*U)&Sf -- a DIFFERENT field from the corrected phi, which is exactly the
            // discrepancy the phi read was added to remove. Dimensions are the compressible MASS flux
            // [1 0 -1 0 0 0 0] (kg/s), not the incompressible volumetric default.
            // 17 digits, not the 12 the vol fields use: this field is read back to seed a restart, so it
            // wants an exact double round-trip rather than display precision.
            writeSurfaceField(outDir + "/phi", solver.phiInternal(), solver.phiBoundary(), fvp,
                              17, "[1 0 -1 0 0 0 0]");
            std::printf("written %s/{U,p,T,rho,phi%s}\n", outDir.c_str(),
                        ctl.turbulent ? (ctl.sst ? ",k,omega,nut" : ",k,epsilon,nut") : "");
            wc.recordWritten(caseDir, tname);
        };

        // endTime is ABSOLUTE, not a run length. OF's Time::run() tests `value() < endTime - 0.5*deltaT`,
        // so a case restarted at 10 with endTime 20 runs TEN more steps and finishes at 20. brae looped
        // `iter <= endTime` from 1, which on that restart ran TWENTY steps and finished at 30 -- silently
        // changing the iteration count, the write times, and any comparison of a restarted run against a
        // continuous one. Only correct when startTime is 0, which is why every fresh-start case hid it.
        // OF Time::run tests `value() < endTime - 0.5*deltaT` and operator++ ACCUMULATES the value
        // (Time.C:785, :1067). std::lround on the quotient disagrees at ratio n + 0.5: measured, real
        // OpenFOAM runs 2 steps at startTime 0 / endTime 1 / deltaT 0.4 where lround gives 3.
        const long nSteps = openFoamNSteps(static_cast<double>(tStart),
                                           static_cast<double>(endTime),
                                           static_cast<double>(wc.deltaT()));
        if (nSteps < 1)
            throw std::runtime_error(
                "controlDict endTime (" + std::to_string(endTime) + ") is not beyond the start time ("
                + startName + "): there is nothing to run. endTime is an ABSOLUTE time, not a number of "
                "iterations -- on a restart set it past the time you are restarting from.");
        int nIter = static_cast<int>(nSteps);
        bool converged = false;
        scalar cumulativeCont = 0;   // OF's "cumulative =" in the continuity-error line
        for (int iter = 1; iter <= nSteps; ++iter)
        {
            clearTurbulenceReport();
            const DeviceSimpleResidual r = solver.rhoSimpleStep();
            // OF prints a cumulative continuity error; brae's residual carries the per-step one.
            cumulativeCont += r.contGlobal;
            printOfResidualLog(iter, r, cumulativeCont);   // no-op unless BRAE_OF_LOG=1
            if (iter % 50 == 0 || iter == 1)
            {
                // OF prints the TIME NAME (runTime.timeName()), not the iteration index. They coincide
                // only at startTime 0 with deltaT 1, which every fixture in validation/ is -- the blind
                // spot that hid the same thing on the V2 driver until queue item 39. This driver keeps
                // no Time object, so the value is built the way Time::operator++ builds it: startTime
                // plus iter steps of deltaT (Time.C:1067), named by WriteControl::timeName.
                std::printf("Time = %s   Ux %.4e  p %.4e  contGlobal %.4e\n",
                            WriteControl::timeName(static_cast<scalar>(tStart)
                                                   + static_cast<scalar>(iter) * wc.deltaT()).c_str(),
                            r.Ux, r.p, r.contGlobal);
            }
            resControl.beginIteration();
            // U is gated on Ux alone, matching gpuSimpleFoam: brae tracks no solved-directions mask, so the
            // out-of-plane component of a 2D/empty or wedge case carries a degenerate residual that would
            // wrongly block convergence on every 2D case.
            // EVERY criterion is evaluated, with no short circuit and no early break. OF does the same:
            // solutionControl loops the whole residualControl_ list and ANDs the outcomes, because each
            // entry also has to be COUNTED -- its `checked` flag is what stops an empty or unmatched dict
            // from declaring convergence on nothing.
            //
            // `a && b` skips b whenever a is false, so on any iteration where p had not yet converged the
            // U / e / turbulence targets were never looked up at all. The converged DECISION was the same
            // (false either way), but the criteria were never counted and dict_audit correctly reported
            // residualControl/U, /e and /(k|epsilon) as entries brae never read.
            bool achieved = resControl.ok(r.p,  "p");
            achieved      = resControl.ok(r.Ux, "U") && achieved;   // U gated on Ux alone (see below)
            // The ENERGY, from the residual struct. correctTurbulence() clears the shared report store
            // before the turbulence solves, so the EEqn's entry is gone by the time the loop below runs --
            // `residualControl { e 1e-3; }` was therefore never evaluated and never counted.
            achieved      = resControl.ok(r.he, heName) && achieved;
            for (const auto& e : turbulenceReport())
                achieved = resControl.ok(e.perf.initialResidual, e.field == "he" ? heName : e.field) && achieved;
            if (resControl.converged(achieved)) { converged = true; nIter = iter; break; }
            // Intermediate write. Skipped on the last iteration, which the final write below covers.
            const scalar tval = wc.timeValue(iter);
            if (iter != nSteps && wc.isWriteTime(iter, tval)) writeTimeDir(WriteControl::timeName(tval));
        }
        std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                              : "SIMPLE reached endTime (%d iterations)\n", nIter);

        // Always write the final (converged / endTime) state, as OF's writeAndEnd does. Named from the
        // TIME VALUE, not the iteration count, so a case with deltaT != 1 gets OF's directory names.
        const std::string finalName = WriteControl::timeName(wc.timeValue(nIter));
        writeTimeDir(finalName);
        std::printf("brae_rhoSimpleFoam: wrote %s\n", (caseDir + "/" + finalName).c_str());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
