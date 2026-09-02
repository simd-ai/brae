// rhoSimpleFoamDriver.cu -- see the header for what is shared with the host driver and why.
#include "rhoSimpleFoamDriver.cuh"

#include "brae_time.cuh"
#include "foam_field_writer.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "linear_solver_setup.cuh"
#include "residual_control.cuh"
#include "rhoSimpleFoamDriver_cpp.cuh"   // buildStepInput: the SHARED case -> StepInput parse
#include "rhoThermoDevice.cuh"           // effectiveTransport, device-resident
#include "thermo_model.cuh"              // hConstTToHe: limitTemperature is a T limit, the device clamps he
#include "rhoTurbulenceHook.cuh"         // correctTurbulence, device-resident
#include "solver_controls.cuh"
#include "write_control.cuh"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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
    in.gradULimitK        = hin.gradULimitK;

    in.tolU = hin.tolU;  in.relTolU = hin.relTolU;
    in.tolHe = hin.tolHe; in.relTolHe = hin.relTolHe;
    in.tolP = hin.tolP;  in.relTolP = hin.relTolP;
    in.maxIter = hin.maxIter;

    return in;
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

    Time time(caseDir, controlDict);
    const std::string startName = time.startName();
    WriteControl& wc = time.writeControl();

    // The HOST field set first, exactly as the harness does: createDeviceFields projects the device
    // state from it, and every refusal createFields carries (thermo, RAS model, boundary conditions,
    // coupled patches) fires here before a single byte reaches the GPU.
    cpu::rhoSimple::RhoSimpleFields hf =
        cpu::rhoSimple::createFields(caseDir + "/" + startName, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);

    std::printf("brae rhoSimpleFoam (OF-mirror, CUDA): %ld cells, start %s, %s\n",
                (long)nC, startName.c_str(),
                hf.turbulent ? (hf.turbulenceFrozen ? (hf.rasModel + " (frozen)").c_str()
                                                    : hf.rasModel.c_str())
                             : "laminar");

    cpu::rhoSimple::CaseRefusals refusals;
    cpu::rhoSimple::StepInput hin =
        cpu::rhoSimple::buildStepInput(caseDir, hf, fvSolution, m, refusals);

    // The case's own linear-solver tolerances, for the same reason the host driver reads them: a gate
    // pins them so the linear solve is out of the comparison, a SOLVER runs what the case asks for.
    {
        DeviceSimpleControls lctl;
        const std::string secondName = (hf.rasModel == "kOmegaSST") ? "omega" : "epsilon";
        readLinearSolverControls(fvSolution, secondName, lctl, "SIMPLE");
        hin.tolU    = lctl.tolU;    hin.relTolU  = lctl.relTolU;
        hin.tolP    = lctl.tolP;    hin.relTolP  = lctl.relTolP;
        hin.tolHe   = lctl.tolKE;   hin.relTolHe = lctl.relTolKE;
        hin.tolTurb = lctl.tolKE;   hin.relTolTurb = lctl.relTolKE;
        hin.maxIter = lctl.maxIterP;
        std::printf("  linear solver (from fvSolution): p tol %.1e relTol %.3g maxIter %d\n",
                    (double)hin.tolP, (double)hin.relTolP, hin.maxIter);
    }

    RhoDeviceFields dev = createDeviceFields(hf, m, g, patches);
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

    RhoSolverWorkspace w;

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
        turbOpt.co                   = hf.keCoeffs;
        turbOpt.co.correctedLaplacian = hin.correctedLaplacian;
        turbOpt.co.snGradLimitCoeff   = hin.snGradLimitCoeff;
        turbOpt.Prt                  = hf.Prt;
        turbOpt.bounded              = hin.boundedTurb;
        turbOpt.correctedLaplacian   = hin.correctedLaplacian;
        // limitedLinear is assembled on the HOST closure and not on the device one, so the device arm
        // must keep refusing it by name: the flag the host parse produces is carried through unchanged,
        // plus the device's own scheme limit.
        turbOpt.divSchemeUnsupported = !hin.turbDivUnsupported.empty()
                                     ? hin.turbDivUnsupported
                                     : (hin.limitedLinearTurb
                                        ? std::string("Gauss limitedLinear (device closure is upwind-only)")
                                        : std::string());
        turbOpt.relaxEquationK   = hin.relaxEquationK;   turbOpt.relaxK   = hin.relaxK;
        turbOpt.relaxEquationEps = hin.relaxEquationEps; turbOpt.relaxEps = hin.relaxEpsilon;
        turbOpt.tol = hin.tolTurb; turbOpt.relTol = hin.relTolTurb; turbOpt.maxIter = hin.maxIter;
        if (constraints.hasK)   { turbOpt.fvoKMask   = &constraints.kMask;   turbOpt.fvoKVal   = &constraints.kVal; }
        if (constraints.hasEps) { turbOpt.fvoEpsMask = &constraints.epsMask; turbOpt.fvoEpsVal = &constraints.epsVal; }

        gin.correct = [&]()
        {
            correctTurbulence(dev.f, dev, dev.dm, dev.dbU, hf.thermo, turbOpt, turbBuf);
        };
    }

    ResidualControl resControl(simpleDict ? simpleDict->subDict("residualControl") : nullptr);
    std::printf("  residualControl=%s\n", resControl.active() ? "on" : "off");

    const scalar endTime = controlDict.scalarOr("endTime", 0.0);
    const scalar tStart  = wc.startTime();
    const long nSteps = std::lround((static_cast<double>(endTime) - static_cast<double>(tStart))
                                    / static_cast<double>(wc.deltaT()));
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

        const Residuals r = rhoSimpleStep(dev.f, w, dev.dm, dev.dbU, dev.dbP, dev.dbHe, dev.dbT, gin);

        auto res = [&](const char* k) { return r.count(k) ? (double)r.at(k) : 0.0; };
        std::printf("Time = %s   U %.4e   %s %.4e   p %.4e",
                    WriteControl::timeName(wc.timeValue(iter)).c_str(),
                    res("U"), hf.heName.c_str(), res(hf.heName.c_str()), res("p"));
        if (r.count("k")) std::printf("   k %.4e   %s %.4e", res("k"), second.c_str(), res(second.c_str()));
        std::printf("\n");

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

    writeTimeDir(WriteControl::timeName(wc.timeValue(nIter)));
    std::printf("End\n");
    return 0;
}

} // namespace rhoSimple
} // namespace gpu
} // namespace brae
