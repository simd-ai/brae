// rhoSimpleFoamDriver.cu -- see the header for what is shared with the host driver and why.
#include "brae_notice.cuh"
#include "rhoSimpleFoamDriver.cuh"

#include "brae_time.cuh"
#include "rhoScalarTransportFO.cuh"   // functionObjects::scalarTransport on this arm (item 15c)
#include "dict_audit.cuh"
#include <memory>   // DictAuditScope: every dictionary entry read off disk and never applied, reported on every exit
#include "foam_field_writer.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "linear_solver_setup.cuh"
#include "of_residual_log.cuh"           // momentumSolverName / printOfSolveLine: the OpenFOAM-format momentum lines
#include "residual_control.cuh"
#include "rhoSimpleFoamDriver_cpp.cuh"   // buildStepInput: the SHARED case -> StepInput parse
#include "rhoThermoDevice.cuh"           // effectiveTransport, device-resident
#include "thermo_model.cuh"              // hConstTToHe: limitTemperature is a T limit, the device clamps he
#include "rhoTurbulenceHook.cuh"         // correctTurbulence, device-resident
#include "solution_directions.cuh"       // polyMesh::solutionD(): which U components are solved
#include "device_mesh.cuh"               // deviceDiv: continuityErrs.H on the device arm
#include "device_blas.cuh"               // deviceHadamard / deviceDot / deviceSumMag / deviceOnes
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
namespace gpu {
namespace rhoSimple {

RhoStepInput buildDeviceStepInput(
    const cpu::rhoSimple::StepInput&       hin,
    const cpu::rhoSimple::RhoSimpleFields& hf,
    const cpu::rhoSimple::CaseRefusals&    refusals,
    const RhoDeviceFields&                 dev,
    const std::vector<FvPatch>&            patches,
    DevicePorosity&                        porosity,
    DeviceConstraints&                     constraints,
    label                                  nCells)
{
    RhoStepInput in;

    in.hasMRF              = refusals.hasMRF;
    in.hasFvOptions        = refusals.hasFvOptions;
    in.fvOptionUnsupported = refusals.fvOptionUnsupported;

    // THE POROUS ZONE, projected onto the device. rhoUEqn.cu applies it (in.porosity) and says so in
    // its own refusal -- "explicitPorositySource (fixedCoeff) IS implemented" -- but nothing built a
    // DevicePorosity for this driver, so every fvOption the host arm implements was reported here as
    // "implemented on the host arm only" and the run refused. That turned OpenFOAM's own
    // angledDuctExplicitFixedCoeff tutorial into a case the host arm ran and the device arm would not,
    // for no reason but a missing projection.
    //
    // What is NOT projected still refuses, by name and per option: anything the host parse marked
    // unsupported, and any implemented option that is not a porosity (there is no device consumer for
    // the temperature/scalar constraints -- the host arm carries those).
    for (const auto& o : refusals.opts.options)
    {
        if (!o.active) continue;
        if (!o.unsupported.empty())
        {
            in.hasFvOptions = true;
            if (in.fvOptionUnsupported.empty()) in.fvOptionUnsupported = o.unsupported;
            continue;
        }
        // THE CONSTRAINTS. Not source terms: OpenFOAM applies them with fvMatrix::setValues, which also
        // strips the coupling out of the neighbours' equations, and the device equations now do the
        // same (deviceSetValues). Both were refused here as "implemented on the host arm only", which
        // is what kept OpenFOAM's own angledDuctExplicitFixedCoeff off this arm.
        if (o.type == "fixedTemperatureConstraint" || o.type == "scalarFixedValueConstraint")
        {
            std::vector<label>  mask(static_cast<std::size_t>(nCells), 0);
            if (o.allCells) std::fill(mask.begin(), mask.end(), 1);
            else for (label c : o.cells) if (c >= 0 && c < nCells) mask[static_cast<std::size_t>(c)] = 1;

            if (o.type == "fixedTemperatureConstraint")
            {
                // OpenFOAM pins he(p, Tuniform), NOT the temperature: setValues on the energy equation
                // takes an energy, and putting a temperature there is a 400x error that still converges.
                const scalar heVal = hConstTToHe(o.Tuniform, hf.thermo);
                constraints.heMask.copyFrom(mask);
                constraints.heVal.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nCells), heVal));
                constraints.hasHe = true;
                std::printf("  fvOptions `%s`: fixedTemperatureConstraint T=%g K -> he=%g on %d cells\n",
                            o.name.c_str(), (double)o.Tuniform, (double)heVal,
                            o.allCells ? (int)nCells : (int)o.cells.size());
                continue;
            }
            // scalarFixedValueConstraint: one entry per field it names. k and epsilon are the two the
            // device closure can pin; any other field has no device consumer and refuses by name.
            for (const auto& fv : o.fieldValues)
            {
                const std::vector<scalar> vals(static_cast<std::size_t>(nCells), fv.second);
                if (fv.first == "k")
                {
                    constraints.kMask.copyFrom(mask); constraints.kVal.copyFrom(vals);
                    constraints.hasK = true;
                }
                else if (fv.first == "epsilon")
                {
                    constraints.epsMask.copyFrom(mask); constraints.epsVal.copyFrom(vals);
                    constraints.hasEps = true;
                }
                else
                {
                    in.hasFvOptions = true;
                    if (in.fvOptionUnsupported.empty())
                        in.fvOptionUnsupported =
                            "scalarFixedValueConstraint on `" + fv.first + "` (the device equations "
                            "constrain he, k and epsilon; this field has no device consumer)";
                    continue;
                }
                std::printf("  fvOptions `%s`: scalarFixedValueConstraint %s=%g on %d cells\n",
                            o.name.c_str(), fv.first.c_str(), (double)fv.second,
                            o.allCells ? (int)nCells : (int)o.cells.size());
            }
            continue;
        }
        if (o.type != "explicitPorositySource")
        {
            in.hasFvOptions = true;
            if (in.fvOptionUnsupported.empty())
                in.fvOptionUnsupported = o.type + " (implemented on the host arm only)";
            continue;
        }
        // A ROTATED coordinate system is refused rather than silently flattened: the device kernel
        // takes DIAGONAL Darcy coefficients, and dropping D's off-diagonals applies the resistance
        // along the wrong axes. fixedCoeff carries FULL transformed tensors, so it has no such limit.
        if (!o.fixedCoeff)
        {
            const scalar offD = std::fabs(o.D.xy) + std::fabs(o.D.xz) + std::fabs(o.D.yz)
                              + std::fabs(o.D.yx) + std::fabs(o.D.zx) + std::fabs(o.D.zy);
            const scalar offF = std::fabs(o.F.xy) + std::fabs(o.F.xz) + std::fabs(o.F.yz)
                              + std::fabs(o.F.yx) + std::fabs(o.F.zx) + std::fabs(o.F.zy);
            const scalar sc = std::fabs(o.D.xx) + std::fabs(o.D.yy) + std::fabs(o.D.zz) + 1.0;
            if (offD + offF > scalar(1e-10) * sc)
            {
                in.hasFvOptions = true;
                if (in.fvOptionUnsupported.empty())
                    in.fvOptionUnsupported =
                        "explicitPorositySource `" + o.name + "` with a ROTATED coordinateSystem (D and "
                        "F carry off-diagonals); the device porosity kernel takes diagonal coefficients";
                continue;
            }
        }
        porosity.active = true;
        porosity.cells.copyFrom(o.cells);
        porosity.d = vector{o.D.xx, o.D.yy, o.D.zz};
        // The kernel applies the 0.5 of OF's 0.5*rho*|U|*F itself, so the RAW F goes across.
        porosity.f = vector{scalar(2) * o.F.xx, scalar(2) * o.F.yy, scalar(2) * o.F.zz};
        porosity.fixed = o.fixedCoeff;
        if (o.fixedCoeff)
        {
            // fixedCoeff's alpha and beta are FULL tensors: calcTransformModelData rotates diag(alpha)
            // into the coordinate system's frame, so all nine components matter.
            const scalar a[9] = {o.alpha.xx, o.alpha.xy, o.alpha.xz,
                                 o.alpha.yx, o.alpha.yy, o.alpha.yz,
                                 o.alpha.zx, o.alpha.zy, o.alpha.zz};
            const scalar b[9] = {o.beta.xx, o.beta.xy, o.beta.xz,
                                 o.beta.yx, o.beta.yy, o.beta.yz,
                                 o.beta.zx, o.beta.zy, o.beta.zz};
            for (int k = 0; k < 9; ++k) { porosity.fa[k] = a[k]; porosity.fb[k] = b[k]; }
            // fixedCoeff::correct reads rhoRef only when the equation is in FORCE units, which the
            // compressible momentum equation is.
            porosity.rhoRef = o.rhoRef;
        }
        std::printf("  fvOptions `%s`: explicitPorositySource/%s on %d cells\n", o.name.c_str(),
                    o.fixedCoeff ? "fixedCoeff" : "DarcyForchheimer", (int)o.cells.size());
    }
    in.porosity = porosity.active ? &porosity : nullptr;
    if (constraints.hasHe) { in.fvoHeMask = &constraints.heMask; in.fvoHeVal = &constraints.heVal; }

    // limitTemperature, projected onto the device's OWN form. deriveCaseRefusals resolves this option
    // OUT of the option list (it sets limitT/limitTmin/limitTmax and fvOptions::read never lists it),
    // so the loop above cannot see it -- and the device applies its limit to he, not T
    // (rhoSimpleFoam.cu limitEnergyKernel), which is why it needs a conversion rather than a copy.
    // Without this the host arm clamped the temperature and the device arm silently did not: an
    // fvOption the case declares, honoured on one arm only, with nothing saying so.
    if (refusals.limitT)
    {
        in.limitHe = true;
        in.heMin   = hConstTToHe(refusals.limitTmin, hf.thermo);
        in.heMax   = hConstTToHe(refusals.limitTmax, hf.thermo);
        std::printf("  fvOption limitTemperature [%g, %g] K -> he [%g, %g]\n",
                    (double)refusals.limitTmin, (double)refusals.limitTmax,
                    (double)in.heMin, (double)in.heMax);
    }
    for (const FvPatch& p : patches)
        if (isCoupledInterfaceType(p.type) || p.type == "processor") in.hasCoupledPatches = true;

    in.takeUAtBoundary = &dev.takeUAtBoundary;
    in.adjustable      = &dev.adjustable;

    in.consistent = hin.consistent;
    in.transonic  = hin.transonic;
    in.isE        = (hf.heName == "e");
    in.nNonOrthogonalCorrectors = hin.nNonOrthogonalCorrectors;

    in.pRefCell  = hf.pressureControl.refCell;
    in.pRefValue = hf.pressureControl.refValue;
    in.limitMaxP = hf.pressureControl.limitMaxP;
    in.pMaxLimit = hf.pressureControl.pMax;
    in.limitMinP = hf.pressureControl.limitMinP;
    in.pMinLimit = hf.pressureControl.pMin;

    in.hasMixed = dev.hasMixed;
    in.frMagSf  = &dev.frMagSf;
    in.frMdot   = &dev.frMdot;
    in.frNx     = &dev.frNx;
    in.frNy     = &dev.frNy;
    in.frNz     = &dev.frNz;

    in.relaxEquationU  = hin.relaxEquationU;   in.relaxU  = hin.relaxU;
    in.relaxEquationHe = hin.relaxEquationHe;  in.relaxHe = hin.relaxHe;
    in.relaxP    = hin.relaxP;
    in.relaxRho  = hin.relaxRho;
    in.relaxPEqn = hin.relaxPEqn;
    in.relaxPEqnSpecified = hin.relaxPEqnSpecified;

    in.boundedU  = hin.boundedU;
    in.boundedHe = hin.boundedHe;
    in.boundedKE = hin.boundedKE;
    in.schemeU   = hin.schemeU;
    in.schemeHe  = hin.schemeHe;
    in.schemeKE  = hin.schemeKE;
    in.schemeCoeffU = hin.schemeCoeffU;
    in.correctedLaplacian = hin.correctedLaplacian;
    in.snGradLimitCoeff   = hin.snGradLimitCoeff;
    // The energy gradient limiters too: the device energy equation has honoured both since it was
    // written (deviceCellLimitGrad on he and on K|Ekp), and this driver never handed them over -- on
    // rhoKE2 with `grad(e|Ekp) cellLimited Gauss linear 1` the CUDA arm read T 2.35e-02 from OpenFOAM,
    // the whole limited-vs-unlimited gap, while the host arm read 1.5e-12.
    in.gradHeLimitK       = hin.gradHeLimitK;
    in.gradKELimitK       = hin.gradKELimitK;
    in.gradULimitK        = hin.gradULimitK;

    in.tolU = hin.tolU;  in.relTolU = hin.relTolU;
    in.tolHe = hin.tolHe; in.relTolHe = hin.relTolHe;
    in.tolP = hin.tolP;  in.relTolP = hin.relTolP;
    in.maxIterU  = hin.maxIterU;  in.minIterU  = hin.minIterU;
    in.maxIterP  = hin.maxIterP;  in.minIterP  = hin.minIterP;
    in.maxIterHe = hin.maxIterHe; in.minIterHe = hin.minIterHe;

    // The momentum components OpenFOAM solves, from the same empty patches polyMesh::calcDirections
    // reads. Derived here rather than in the host StepInput so the harness, which builds its host
    // input by hand, cannot leave it unset: this function is the one path every device input takes.
    const SolutionDirections solutionD = solutionDirections(patches);
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        in.solutionD[cmpt] = solutionD.d[cmpt];
    }

    return in;
}

TurbulenceHookOptions buildTurbulenceHookOptions(
    const cpu::rhoSimple::StepInput&       hin,
    const cpu::rhoSimple::RhoSimpleFields& hf,
    const DeviceConstraints&               constraints)
{
    TurbulenceHookOptions opt;
    opt.co                    = hf.keCoeffs;
    opt.co.correctedLaplacian = hin.correctedLaplacian;
    opt.co.snGradLimitCoeff   = hin.snGradLimitCoeff;
    opt.co.gradULimitK        = hin.gradULimitK;
    opt.co.gradKLimitK        = hin.gradKLimitK;
    opt.Prt                   = hf.Prt;
    opt.bounded               = hin.boundedTurb;
    opt.correctedLaplacian    = hin.correctedLaplacian;
    // limitedLinear is assembled on the HOST closure and not on the device one, so the device arm
    // must keep refusing it by name: the flag the host parse produces is carried through unchanged,
    // plus the device's own scheme limit.
    opt.divSchemeUnsupported = !hin.turbDivUnsupported.empty()
                             ? hin.turbDivUnsupported
                             : (hin.limitedLinearTurb
                                ? std::string("Gauss limitedLinear (device closure is upwind-only)")
                                : std::string());
    opt.relaxEquationK   = hin.relaxEquationK;
    opt.relaxK           = hin.relaxK;
    opt.relaxEquationEps = hin.relaxEquationEps;
    opt.relaxEps         = hin.relaxEpsilon;
    opt.tol              = hin.tolTurb;
    opt.relTol           = hin.relTolTurb;
    opt.maxIter          = hin.maxIterTurb;
    opt.minIter          = hin.minIterTurb;
    if (constraints.hasK)
    {
        opt.fvoKMask = &constraints.kMask;
        opt.fvoKVal  = &constraints.kVal;
    }
    if (constraints.hasEps)
    {
        opt.fvoEpsMask = &constraints.epsMask;
        opt.fvoEpsVal  = &constraints.epsVal;
    }
    return opt;
}

namespace {

// A device buffer -> the flat host vector the writers take. The device boundary layout already EXCLUDES
// coupled patches (buildDeviceMesh keeps them out and this driver refuses them anyway), which is the
// layout foam_field_writer expects, so this is a copy and not a re-ordering.
std::vector<scalar> host(const DeviceBuffer<scalar>& b)
{
    return b.size() ? b.host() : std::vector<scalar>();
}

} // namespace

int runMirrorCuda(const std::string& caseDir)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
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

    if (controlDict.wordOr("writeFormat", "ascii") == "binary")
        throw std::runtime_error(
            "brae rhoSimpleFoam (mirror, CUDA): controlDict writeFormat is `binary`, which brae's field "
            "writer does not emit -- it writes ASCII only. Refusing rather than writing ascii under a "
            "binary setting. Set `writeFormat ascii;` to run this case.");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    // The scalarTransport factory, as on the host arm: built by Time, fields resolved from `tracerCtx`
    // at the first execute(), after the device fields exist (item 15c).
    brae::tracer::TracerCudaContext tracerCtx;
    std::vector<brae::tracer::RhoTracerCudaFO*> tracers;
    std::vector<std::pair<std::string, FunctionObjectList::Factory>> foTypes;
    foTypes.emplace_back("scalarTransport", brae::tracer::rhoTracerCudaFactory(caseDir, fvSolution, tracerCtx, tracers));
    Time time(caseDir, controlDict, foTypes);
    const std::string startName = time.startName();
    WriteControl& wc = time.writeControl();
    tracerCtx.fieldDir = caseDir + "/" + startName;

    // The HOST field set first, exactly as the harness does: createDeviceFields projects the device
    // state from it, and every refusal createFields carries (thermo, RAS model, boundary conditions,
    // coupled patches) fires here before a single byte reaches the GPU.
    // This driver's wall treatments read Cmu/kappa/E per patch (item 16h-port): the reader must not
    // announce those entries as unhonoured.
    brae::perPatchWallCoeffsHonoured() = true;
    cpu::rhoSimple::RhoSimpleFields hf =
        cpu::rhoSimple::createFields(caseDir + "/" + startName, caseDir, simpleDict, &fvSolution,
                                     m, g, patches, &thermoProps, turbProps.get());

    std::printf("brae rhoSimpleFoam (OF-mirror, CUDA): %ld cells, start %s, %s\n",
                (long)nC, startName.c_str(),
                hf.turbulent ? (hf.turbulenceFrozen ? (hf.rasModel + " (frozen)").c_str()
                                                    : hf.rasModel.c_str())
                             : "laminar");

    cpu::rhoSimple::CaseRefusals refusals;
    cpu::rhoSimple::StepInput hin =
        cpu::rhoSimple::buildStepInput(caseDir, hf, fvSolution, m, refusals);

    // fvSolution's `preconditioner` on the three fields this driver solves with BiCGStab. Hoisted out of
    // the block below because the preconditioner they select is built once, from the mesh, and lives as
    // long as the run.
    bool diluU = false, diluHe = false, diluKE = false;
    // DILU on the transonic pressure's BiCGStab (see RhoStepInput::preconP) is OPT-IN, BRAE_DILU_P=1,
    // and only where the case's own p entry names it (`solver PBiCGStab; preconditioner DILU;`, as
    // sbMatched does). The default keeps the diagonal and announces the substitution, because the
    // value is the same and the diagonal is faster: on sbMatched (112k cells, tolerance 1e-12 relTol 0,
    // so both solves converge fully and the p residual trajectories agree to the printed digits) DILU
    // took the p solve from 668 to 202 BiCGStab iterations and the phase from 68 to 474 ms per
    // iteration; on the squareBend tutorial at 896k cells from ~700 to ~300 iterations and from 315 to
    // 567 ms. The level-scheduled apply (item 70's per-level floor) costs more than the iterations it
    // saves; the lever is a faster apply, not the default. tests/transonic_p_dilu.sh holds both arms.
    const FoamDict* solversDict = fvSolution.subDict("solvers");
    const FoamDict* pEntry = solversDict ? solversDict->subDict("p") : nullptr;
    const bool caseAsksDiluP = pEntry && pEntry->wordOr("preconditioner", "") == "DILU";
    const bool diluP = hin.transonic && caseAsksDiluP
                    && std::getenv("BRAE_DILU_P") && std::string(std::getenv("BRAE_DILU_P")) == "1";
    // The case's smoothSolver selection per field, read below and carried into the step and the
    // turbulence hook (item 58).
    bool gsU = false, gsUSym = true, gsHe = false, gsHeSym = true;
    bool gsK = false, gsEps = false, gsKESym = true;
    int  nSweepsU = 1, nSweepsHe = 1, nSweepsKE = 1;

    // BRAE_U_SOLVER, the momentum solver, read ONCE here. Every consumer below -- the shared reader's
    // notice routing, this driver's own notices, the workspace colouring, the step input and the
    // residual-log name -- takes this one decision, so none of them can disagree.
    //   colourGS  (THE DEFAULT) a multicolour Gauss-Seidel smoothSolver whatever the entry names,
    //             under the stop rule the entry names. Measured on the squareBend tutorial at 305,760
    //             cells, momentum ms per outer iteration: 10.9 against 26.0 for the diagonal BiCGStab
    //             this driver ran before, 58.5 for a DILU one, 57.2 for OpenFOAM's own index-ordered
    //             sweep on the host and 36.2 for it level-scheduled on the device
    //             (bench/results/rhoSimpleFoam_squareBend_gb10.md).
    //   ofOrder   OpenFOAM's OWN index order on a case that names a GaussSeidel smoothSolver (the
    //             level-scheduled sweep), and the diagonal BiCGStab on any other entry: what this
    //             driver ran before. EXACT where colourGS approximates, and the opt-out for a case
    //             whose momentum must reproduce OpenFOAM's iterate under a loose relTol, not only its
    //             converged answer.
    // Both run the case's tolerance, relTol, maxIter, minIter and nSweeps. The difference colourGS
    // makes is the ORDER of the sweep, which changes where a solve stopped short of convergence stops
    // -- announced below, per entry, and gated by tests/u_colour_gs_vs_openfoam.sh.
    bool uColourGS = true;
    if (const char* e = std::getenv("BRAE_U_SOLVER"))
    {
        const std::string sel(e);
        if (sel == "colourGS")
        {
            uColourGS = true;
        }
        else if (sel == "ofOrder")
        {
            uColourGS = false;
        }
        else if (!sel.empty())
        {
            throw std::runtime_error(
                "brae rhoSimpleFoam (mirror): BRAE_U_SOLVER='" + sel + "' names no momentum solver this "
                "driver runs. Accepted values: `colourGS` (the default: a multicolour Gauss-Seidel "
                "smoothSolver) or `ofOrder` (OpenFOAM's own index order where the case names a "
                "GaussSeidel smoothSolver, the diagonal BiCGStab otherwise).");
        }
    }

    // The case's own linear-solver tolerances, for the same reason the host driver reads them: a gate
    // pins them so the linear solve is out of the comparison, a SOLVER runs what the case asks for.
    {
        DeviceSimpleControls lctl;
        lctl.turbulent = hf.turbulent;   // gates the reader's k/epsilon block -- see the host driver
        const std::string secondName = (hf.rasModel == "kOmegaSST") ? "omega" : "epsilon";
        // What THIS driver runs, so the substitution notices describe it rather than the legacy one.
        // It wires DILU on the energy solve as well as on U and the turbulence pair; and its pressure
        // solve is the case's own PBiCGStab on the transonic branch (pcEqn.H's phid matrix is asymmetric,
        // so there is no CG to run there) and the AMG-preconditioned CG otherwise.
        SolverRunsAs runsAs;
        runsAs.diluOnEnergy = true;
        // This arm runs OpenFOAM's own sweep wherever the case names a smoothSolver (item 58).
        // BRAE_RHO_SMOOTHSOLVER=0 restores the previous behaviour -- BiCGStab on every field -- with the
        // notices then announcing each substitution, which is the identity gate's control arm.
        {
            const char* e = std::getenv("BRAE_RHO_SMOOTHSOLVER");
            const bool honour = !(e && std::string(e) == "0");
            runsAs.smoothSolverOnEnergy     = honour;
            // ...except on momentum under BRAE_U_SOLVER=colourGS, where OpenFOAM's sweep does NOT run.
            // The reader's U notice is driven by ctl.gsU = useSymGS("U") && smoothSolverOnMomentum:
            // true means "OpenFOAM's own sweep runs, nothing to say" and silences it, false makes it
            // announce `brae runs PBiCGStab preconditioned with ...` against any other entry, an
            // `ignored` line for the smoother and a preconditioner substitution. In colourGS mode
            // neither is true -- the colour sweep is not OpenFOAM's and no BiCGStab runs -- so gsU is
            // kept honest (false) rather than borrowed as a silencer, and the reader is told the caller
            // owns the U notice. The driver prints exactly one truthful set of lines below, after the
            // read; tests/u_colour_gs_vs_openfoam.sh asserts both that presence and the reader's silence.
            runsAs.smoothSolverOnMomentum   = honour && !uColourGS;
            runsAs.momentumNoticedByCaller  = uColourGS;
            // ...and it is the one path that runs OpenFOAM's fixed-count nSweeps branch on U, so the
            // reader hands it a negative nSweeps raw instead of refusing it (linear_solver_setup.cuh).
            if (uColourGS)
            {
                runsAs.fixedSweepsField = "U";
            }
            runsAs.smoothSolverOnTurbulence = honour;
        }
        if (hin.transonic)
        {
            // The notice must say what RUNS: the case's DILU when BRAE_DILU_P=1 opted in, the diagonal
            // otherwise. One decision, read here and used at the wiring below, so the two cannot
            // disagree (tests/transonic_p_dilu.sh holds both arms to their words).
            runsAs.pSolver = "PBiCGStab";
            runsAs.pPrecon = diluP ? "DILU" : "diagonal";
        }
        readLinearSolverControls(fvSolution, secondName, lctl, "SIMPLE", hf.heName, runsAs);
        hin.tolU    = lctl.tolU;    hin.relTolU    = lctl.relTolU;    hin.maxIterU    = lctl.maxIterU;    hin.minIterU    = lctl.minIterU;
        hin.tolP    = lctl.tolP;    hin.relTolP    = lctl.relTolP;    hin.maxIterP    = lctl.maxIterP;    hin.minIterP    = lctl.minIterP;
        hin.tolHe   = lctl.tolHe;   hin.relTolHe   = lctl.relTolHe;   hin.maxIterHe   = lctl.maxIterHe;   hin.minIterHe   = lctl.minIterHe;
        hin.tolTurb = lctl.tolKE;   hin.relTolTurb = lctl.relTolKE;   hin.maxIterTurb = lctl.maxIterKE;   hin.minIterTurb = lctl.minIterKE;
        // Under BRAE_U_SOLVER=colourGS the momentum branch never reads preconU, so a DILU entry must
        // not make the driver build the level schedule for it (review, round 2).
        diluU  = lctl.diluU && !uColourGS;
        diluHe = lctl.diluHe;
        diluKE = lctl.diluKE;
        // The case's own smoothSolver selection, carried into the step (item 58). Without these the
        // driver ran BiCGStab on U, he, k and epsilon while the shared notice -- which takes the flag as
        // proof the caller honours the dict -- announced nothing for U and the pair.
        gsU = lctl.gsU;       gsUSym = lctl.gsUSym;     nSweepsU  = lctl.nSweepsU;
        gsHe = lctl.gsHe;     gsHeSym = lctl.gsHeSym;   nSweepsHe = lctl.nSweepsHe;
        gsK = lctl.gsK;       gsEps = lctl.gsEps;
        gsKESym = lctl.gsKESym;                          nSweepsKE = lctl.nSweepsKE;
        cpu::rhoSimple::printLinearSolverControls(hin, hf.heName, secondName, hf.turbulent);
    }

    if (uColourGS)
    {
        // The U notice for the colour-order momentum solve (the default), from the driver and AFTER the read, since the reader
        // was told to leave U to the caller (runsAs.momentumNoticedByCaller above). Two cases, one line
        // each: the case asked for a Gauss-Seidel smoothSolver and gets it in a different ORDER, or it
        // asked for anything else and gets a different SOLVER. Both run under the stop rule the entry
        // names (tolerance, relTol, maxIter, minIter, nSweeps); neither leaves the iterate under a
        // loose relTol where OpenFOAM's would, which is what makes this `approximated` and not
        // `equivalent`.
        const FoamDict* uEntry = solversDict ? solversDict->subDict("U") : nullptr;
        const std::string want = uEntry ? uEntry->wordOr("solver", "") : "";
        const std::string smoo = uEntry ? uEntry->wordOr("smoother", "") : "";
        const std::string prec = uEntry ? uEntry->wordOr("preconditioner", "") : "";
        const bool asksGaussSeidel = want == "smoothSolver"
                                  && (smoo == "GaussSeidel" || smoo == "symGaussSeidel");
        if (asksGaussSeidel)
        {
            noticeApproximated("solvers/U smoother",
                               "case asks '" + smoo + "' in OpenFOAM's index order; brae sweeps in COLOUR "
                               "order -- same stop rule, a different iterate after n sweeps (tests/gs_ladder "
                               "measured 1.36x/2.76x/6.88x behind after 1/5/10 sweeps on T3A), 5x faster "
                               "(10.9 against 57.2 ms per iteration at 306k cells). BRAE_U_SOLVER=ofOrder "
                               "runs OpenFOAM's own order instead");
        }
        else
        {
            std::string asked = want.empty() ? std::string("no solver") : "'" + want + "'";
            if (!smoo.empty())
            {
                asked += " with smoother '" + smoo + "'";
            }
            if (!prec.empty())
            {
                asked += " with preconditioner '" + prec + "'";
            }
            // Name the VARIANT that runs: with no smoother in the entry gsIsSymmetric() answers true, so
            // the sweep is the symmetric one (device_sym_gauss_seidel.cuh: symGaussSeidel and GaussSeidel
            // are different smoothers, not settings of one), and the start-up line says the same.
            const std::string variant = gsUSym ? "symGaussSeidel" : "GaussSeidel";
            noticeApproximated("solvers/U solver",
                               "case asks " + asked + ", brae runs a multicolour " + variant + " smoothSolver "
                               "to the same tolerance, relTol, maxIter and minIter, nSweeps "
                             + std::to_string(nSweepsU) + " -- a different solver: the converged answer is "
                               "the same, the iterate under relTol is not. BRAE_U_SOLVER=ofOrder runs the "
                               "diagonal-preconditioned BiCGStab this driver ran before");
        }
    }

    RhoDeviceFields dev = createDeviceFields(hf, m, g, patches);
    tracerCtx.dev = &dev;
    tracerCtx.m = &m;
    tracerCtx.g = &g;
    tracerCtx.patches = &patches;
    DevicePorosity porosity;        // outlives gin: RhoStepInput::porosity points into it
    DeviceConstraints constraints;  // ...and so do the fvOptions constraint masks
    RhoStepInput gin =
        buildDeviceStepInput(hin, hf, refusals, dev, patches, porosity, constraints, nC);
    // The AMG hierarchy cache: built once per mesh, reloaded on every later run in the same case
    // directory. The agglomeration is the AMG build cost and is static per mesh, so this is pure
    // set-up time -- and it is now safe to use: the cache's load path did not rebuild the Galerkin
    // gather lists, so run 1 wrote .brae_amgcache and run 2 died in galDiagGatherK reading index 0 of a
    // zero-length buffer (surfacing as "amul: an illegal memory access"). Fixed in the loader
    // (rebuildGalerkinGather), and a cache-loaded hierarchy now reproduces a cold one BIT-IDENTICALLY
    // on p, T, U, rho and phi over 30 iterations -- which is the property that matters: this may cost
    // nothing but time, and it must change no answer.
    gin.amgCacheDir = caseDir + "/constant/polyMesh";
    // The momentum and energy solvers the case named (item 58).
    gin.uSymGaussSeidel  = gsU;   gin.uGaussSeidelSymmetric  = gsUSym;   gin.nSweepsU  = nSweepsU;
    gin.heSymGaussSeidel = gsHe;  gin.heGaussSeidelSymmetric = gsHeSym;  gin.nSweepsHe = nSweepsHe;

    RhoSolverWorkspace w;
    // OpenFOAM preconditions k and epsilon with DILU wherever the case says so, and the host reference
    // always has (pbicgstab.cuh). The device closure ran Jacobi: on sbMatched that left k 5.4e-09 from
    // OpenFOAM where the host arm sat at 8.4e-12, with both arms' assembled systems agreeing to 1e-11 --
    // the gap was the solver's stopping point, not the discretisation (queue item 27). buildDeviceDilu is
    // a level schedule over the mesh, so it is built once here and refreshed per solve inside the solver.
    if (diluU || diluHe || diluKE || diluP)
    {
        w.dilu = buildDeviceDilu(m.owner(), m.neighbour(), nC);
    }
    gin.preconU  = (diluU  && w.dilu.valid) ? &w.dilu : nullptr;
    gin.preconHe = (diluHe && w.dilu.valid) ? &w.dilu : nullptr;
    // The transonic pressure's BiCGStab takes the same DILU when BRAE_DILU_P=1 opted in (see diluP).
    // The notice above already said DILU, so a schedule that failed to build is refused, not
    // silently replaced by the diagonal it just denied.
    if (diluP && !w.dilu.valid)
        throw std::runtime_error("brae rhoSimpleFoam (mirror): the DILU level schedule for the transonic pressure "
                                 "could not be built; refusing rather than running the diagonal the notice "
                                 "denied. BRAE_DILU=0 selects the diagonal explicitly.");
    gin.preconP  = diluP ? &w.dilu : nullptr;

    if (uColourGS)
    {
        // The colouring the colour-order sweep visits the cells in, once per mesh like w.dilu.
        // PrimitiveMesh::owner() spans the boundary faces as well while neighbour() stops at the
        // internal ones (primitive_mesh.cuh:99, and the slice buildDeviceDilu takes for the same
        // reason), so owner is cut to neighbour().size() before the two are paired face by face.
        const std::vector<label>& ownerAll = m.owner();
        const std::vector<label>& nei = m.neighbour();
        const std::vector<label> ownerInternal(
            ownerAll.begin(),
            ownerAll.begin() + static_cast<std::ptrdiff_t>(nei.size()));
        w.uColouring = buildDeviceCellColouring(ownerInternal, nei, static_cast<int>(nC));
        // The notice above already named the colour sweep, so a colouring that failed to build is
        // refused, not silently replaced by the BiCGStab the notice just denied.
        if (!w.uColouring.valid)
            throw std::runtime_error(
                "brae rhoSimpleFoam (mirror): the multicolour Gauss-Seidel momentum solve (the default) "
                "needs a cell colouring and it could not be built; refusing rather than running the solver "
                "the notice denied. BRAE_U_SOLVER=ofOrder runs OpenFOAM's own order instead.");
        gin.uColourGaussSeidel = true;
        // Never both: the step branches `if (uSymGaussSeidel) ... else if (uColourGaussSeidel)`, and
        // clearing the first here is what keeps it from shadowing the second on a smoothSolver entry.
        gin.uSymGaussSeidel = false;
        gin.uColouring = &w.uColouring;
        std::string sizes;
        for (int c = 0; c + 1 < static_cast<int>(w.uColouring.startH.size()); ++c)
        {
            if (!sizes.empty()) sizes += ' ';
            sizes += std::to_string(w.uColouring.startH[static_cast<std::size_t>(c) + 1]
                                  - w.uColouring.startH[static_cast<std::size_t>(c)]);
        }
        std::printf("  momentum: multicolour Gauss-Seidel smoothSolver, %d colours (sizes %s), "
                    "tolerance/relTol/maxIter/minIter/nSweeps from the case's U entry, %s sweeps\n",
                    w.uColouring.nColours,
                    sizes.c_str(),
                    gin.uGaussSeidelSymmetric ? "symGaussSeidel (ascending then descending)"
                                              : "GaussSeidel (ascending only)");
    }

    // The name on the OpenFOAM-format `Solving for Ux` line (of_residual_log.cuh, BRAE_OF_LOG=1): what
    // RUNS, decided from the same flags the step branches on, so the line cannot name a solver the
    // step did not run. The legacy drivers never set this and keep printing JacobiBiCGStab.
    momentumSolverName() = gin.uColourGaussSeidel ? "colourGaussSeidel"
                         : gin.uSymGaussSeidel    ? (gin.uGaussSeidelSymmetric ? "symGaussSeidel"
                                                                               : "GaussSeidel")
                         : gin.preconU            ? "DILUPBiCGStab"
                                                  : "JacobiBiCGStab";

    // THE THERMO HOOKS, device-resident. The step takes them as hooks because EEqn.H ends in
    // thermo.correct() -- which moves T and therefore psi, and every consumer below that point reads
    // the result -- and pcEqn.H opens with rho = thermo.rho(); the step refuses to run without them
    // rather than solve the whole iteration against the state it started with. Both call the same
    // BRAE_HD inline functions the host reference does, so what differs is where, not what.
    gin.thermoCorrect = [&]() { thermoCorrect(dev.f, dev.dbT, hf.thermo); };
    gin.updateRho     = [&]() { updateRho(dev.f, hf.thermo); };

    TurbulenceHookBuffers turbBuf;
    TurbulenceHookOptions turbOpt;
    if (hf.turbulent && !hf.turbulenceFrozen && !hf.k.internal.empty())
    {
        // The SAME options the CUDA harness's turbulent arm drives (buildTurbulenceHookOptions above).
        turbOpt = buildTurbulenceHookOptions(hin, hf, constraints);
        turbOpt.precon = (diluKE && w.dilu.valid) ? &w.dilu : nullptr;
        turbOpt.gsK = gsK;  turbOpt.gsEps = gsEps;  turbOpt.gsSymmetric = gsKESym;  turbOpt.nSweepsKE = nSweepsKE;

        gin.correct = [&]()
        {
            correctTurbulence(dev.f, dev, dev.dm, dev.dbU, hf.thermo, turbOpt, turbBuf);
        };
    }

    ResidualControl resControl(simpleDict ? simpleDict->subDict("residualControl") : nullptr);
    std::printf("  residualControl=%s\n", resControl.active() ? "on" : "off");

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
            "brae rhoSimpleFoam (mirror, CUDA): controlDict endTime (" + std::to_string((double)endTime)
            + ") is not beyond the start time (" + startName + "): there is nothing to run. endTime is "
              "an ABSOLUTE time, not a number of iterations.");
    time.setSteps(static_cast<int>(nSteps));

    const std::string wsrc = caseDir + "/" + startName + "/";
    const std::string second = (hf.rasModel == "kOmegaSST") ? "omega" : "epsilon";

    auto writeTimeDir = [&](const std::string& tname)
    {
        const std::string outDir = caseDir + "/" + tname;
        std::filesystem::create_directories(outDir);
        // The ONLY device-to-host traffic in the run, and it happens on the write cadence rather than
        // every iteration -- which is the whole point of the device path.
        const std::vector<scalar> ux = host(dev.f.Ux), uy = host(dev.f.Uy), uz = host(dev.f.Uz);
        const std::vector<scalar> uxb = host(dev.f.UxBnd), uyb = host(dev.f.UyBnd), uzb = host(dev.f.UzBnd);
        std::vector<vector> U(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) U[c] = vector{ux[c], uy[c], uz[c]};
        std::vector<vector> UB(uxb.size());
        for (std::size_t i = 0; i < uxb.size(); ++i) UB[i] = vector{uxb[i], uyb[i], uzb[i]};
        writeVolField(wsrc + "U", outDir + "/U", U, patches, 12, UB);
        writeVolField(wsrc + "p", outDir + "/p", host(dev.f.p), patches, 12, host(dev.f.pBnd));
        writeVolField(wsrc + "T", outDir + "/T", host(dev.f.T), patches, 12, host(dev.f.TBnd));
        // scalarTransport's transportedField(), written beside the solved fields on the same cadence;
        // its boundary is evaluated from the device field here, as k's is below.
        for (const brae::tracer::RhoTracerCudaFO* st : tracers)
            if (st->ready())
                writeVolField(wsrc + st->fieldName(), outDir + "/" + st->fieldName(), st->hostField(), patches, 12,
                              st->boundaryFlat());
        {
            static const DerivedFieldSpec rhoSpec{"rho", "dimensions      [1 -3 0 0 0 0 0];"};
            writeVolField(wsrc + "T", outDir + "/rho", host(dev.f.rho), patches, 12,
                          host(dev.f.rhoBnd), &rhoSpec);
        }
        if (hf.turbulent && dev.f.k.size())
        {
            // k and epsilon have no materialised boundary buffer on the device -- their patch values
            // live in the DeviceBoundary objects the closure solves against -- so they are evaluated
            // here rather than written with the start directory's boundary echoed back.
            DeviceBuffer<scalar> kB, eB;
            deviceBCValue(dev.dbK, dev.f.k, kB);
            deviceBCValue(dev.dbEps, dev.f.epsilon, eB);
            writeVolField(wsrc + "k", outDir + "/k", host(dev.f.k), patches, 12, host(kB));
            writeVolField(wsrc + second, outDir + "/" + second, host(dev.f.epsilon), patches, 12,
                          host(eB));
            writeVolField(wsrc + "nut", outDir + "/nut", host(dev.f.nut), patches, 12,
                          host(dev.f.nutBnd));
            if (dev.f.alphat.size())
                writeVolField(wsrc + "alphat", outDir + "/alphat", host(dev.f.alphat), patches, 12,
                              host(dev.f.alphatBnd));
        }
        writeSurfaceField(outDir + "/phi", host(dev.f.phiInt), host(dev.f.phiBnd), patches, 17,
                          "[1 0 -1 0 0 0 0]");
        std::printf("written %s\n", outDir.c_str());
        wc.recordWritten(caseDir, tname);
    };

    int  nIter = static_cast<int>(nSteps);
    bool converged = false;
    scalar cumulativeContErr = 0.0;   // continuityErrs.H: summed over the run
    while (time.loop())
    {
        const int iter = time.timeIndex();

        // muEff and alphaEff, on the device. They are the only route by which the thermo and the
        // closure reach the momentum and energy equations, and computing them on the host would mean
        // pulling U, p, T and rho down and pushing four arrays back every iteration.
        DeviceBuffer<scalar> dMu, dMuB, dAl, dAlB;
        effectiveTransport(dev.f, hf.thermo, hf.turbulent, dMu, dMuB, dAl, dAlB);
        gin.muEffCell = &dMu;       gin.muEffBndFace = &dMuB;
        gin.alphaEffCell = &dAl;    gin.alphaEffBndFace = &dAlB;

        Residuals r = rhoSimpleStep(dev.f, w, dev.dm, dev.dbU, dev.dbP, dev.dbHe, dev.dbT, gin);
        // THE CLOSURE'S RESIDUALS, which the step cannot return: its turbulence hook is a
        // std::function<void()>, so k and epsilon never reached `r` and the two residualControl
        // branches below were dead code on this arm -- a case naming "(k|epsilon)" (angledDuct,
        // squareBend) was declared converged on p, U and e alone, a strict subset of OpenFOAM's
        // criteria (simpleControl::criteriaSatisfied walks every field solved this step,
        // simpleControl.C:58-85), and so stopped no later than OpenFOAM and in practice earlier. The
        // values were always computed: kEpsilon.cu writes them into the stages this driver owns, and
        // the hook has run by the time the step returns. Measured on rhoKE with "(k|epsilon)" 1e-3 /
        // 1e-5 alongside p/U/h 1e-3: the run stops at the same iteration either way with this block
        // absent, and later under the tighter criterion with it present.
        if (hf.turbulent)
        {
            r["k"]    = turbBuf.stages.kResidual;
            r[second] = turbBuf.stages.epsResidual;
        }

        // OpenFOAM's own `Solving for Ux` line per SOLVED component (BRAE_OF_LOG=1), named for the
        // solver that ran (momentumSolverName()). The knocked-out component has no entry in `r`, as
        // OpenFOAM prints no line for it (fvMatrixSolve.C:164). Only the momentum lines: they are the
        // ones whose final residual and iteration count this arm's step returns.
        if (ofResidualLog())
        {
            for (const char* c : {"Ux", "Uy", "Uz"})
            {
                if (!r.count(c)) continue;
                printOfSolveLine(momentumSolverName(),
                                 c,
                                 r.at(c),
                                 r.at(std::string(c) + "Final"),
                                 static_cast<int>(r.at(std::string(c) + "Iters")));
            }
        }
        auto res = [&](const char* k) { return r.count(k) ? (double)r.at(k) : 0.0; };
        std::printf("Time = %s   U %.4e   %s %.4e   p %.4e",
                    WriteControl::timeName(wc.timeValue(iter)).c_str(),
                    res("U"), hf.heName.c_str(), res(hf.heName.c_str()), res("p"));
        if (r.count("k")) std::printf("   k %.4e   %s %.4e", res("k"), second.c_str(), res(second.c_str()));
        if (r.count("pIters")) std::printf("   pIters %.0f", res("pIters"));
        if (r.count("uIters")) std::printf("   uIters %.0f", res("uIters"));
        std::printf("\n");
        // OpenFOAM's continuityErrs.H (rhoSimpleFoam's pEqn.H:81 / pcEqn.H:94 include it every
        // iteration): contErr = fvc::div(phi) on the corrected MASS flux, sum local = deltaT * the
        // volume-weighted average of |contErr|, global = the same of contErr, cumulative summed over
        // the run. The host step prints it (rhoSimpleFoam_cpp.cu); this arm printed nothing (item 16a).
        // Two reductions per iteration, beside the residual reads the summary line above already makes;
        // tests/mirror_continuity.sh holds the line to the host arm and to a recomputation from the
        // written phi.
        {
            DeviceBuffer<scalar> contErr, R;
            deviceDiv(dev.dm, dev.f.phiInt, dev.f.phiBnd, contErr);
            deviceHadamard(R, contErr, dev.dm.V);
            const DeviceBuffer<scalar>& ones = deviceOnes(dev.dm.nCells);
            const scalar sumV     = deviceDot(dev.dm.V, ones);
            const scalar sumLocal = hf.deltaT * deviceSumMag(R) / sumV;
            const scalar global   = hf.deltaT * deviceDot(R, ones) / sumV;
            cumulativeContErr += global;
            std::printf("time step continuity errors : sum local = %g, global = %g, cumulative = %g\n",
                        sumLocal, global, cumulativeContErr);
        }

        resControl.beginIteration();
        bool achieved = resControl.ok(r.count("p") ? r.at("p") : scalar(0), "p");
        achieved = resControl.ok(r.count("U") ? r.at("U") : scalar(0), "U") && achieved;
        if (r.count(hf.heName)) achieved = resControl.ok(r.at(hf.heName), hf.heName) && achieved;
        if (r.count("k"))       achieved = resControl.ok(r.at("k"), "k") && achieved;
        if (r.count(second))    achieved = resControl.ok(r.at(second), second) && achieved;
        if (resControl.converged(achieved)) { converged = true; nIter = iter; time.stop(); break; }

        if (time.writeTime()) writeTimeDir(WriteControl::timeName(wc.timeValue(iter)));
    }
    time.end();
    std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                          : "SIMPLE reached endTime (%d iterations)\n", nIter);
    rhoPhaseTimeReport(nIter);   // BRAE_PHASE_TIME (item 67); silent otherwise

    writeTimeDir(WriteControl::timeName(wc.timeValue(nIter)));
    std::printf("End\n");
    return 0;
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
