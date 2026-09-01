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
    const std::vector<FvPatch>&            patches)
{
    RhoStepInput in;

    // The refusals, as the DEVICE arm must see them. fvOptions is the one that differs by construction:
    // the host arm implements a set of them (refusals.opts) and there is no device consumer, so an
    // implemented option is still an UNPORTED one here and says so rather than being dropped.
    in.hasMRF       = refusals.hasMRF;
    in.hasFvOptions = refusals.hasFvOptions || !refusals.opts.empty();
    in.fvOptionUnsupported =
        !refusals.fvOptionUnsupported.empty() ? refusals.fvOptionUnsupported
      : (!refusals.opts.empty() ? std::string("implemented on the host arm only") : std::string());
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
    RhoStepInput gin = buildDeviceStepInput(hin, hf, refusals, dev, patches);
    // The AMG hierarchy cache, which the compressible path could never reach because nothing gave the
    // driver a directory to keep it in -- so every run rebuilt the hierarchy from cold.
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
