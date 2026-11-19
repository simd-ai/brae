// rhoCentralFoam.cu -- the driver: rhoCentralFoam.C's main() for both arms. See the header.
#include "rhoCentralFoam.cuh"
#include "rhoCentralCreateFields_cpp.cuh"
#include "rhoCentralStep.cuh"
#include "rhoCentralStep_cpp.cuh"

#include "brae_time.cuh"
#include "device_mesh.cuh"   // the limiter selectors
#include "device_ldu.cuh"    // setDeviceResidualFused
#include "purge_write.cuh"
#include "run_clock.cuh"
#include "time_dict_writer.cuh"
#include "time_controls.cuh"
#include "foam_dict.cuh"
#include "foam_field_writer.cuh"
#include "dynamic_motion_solver_fv_mesh_cpp.cuh"   // milestone K
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "io_precision.cuh"
#include "write_control.cuh"
#include "inter_phase_time.cuh"   // BRAE_INTER_PHASE_TIME: where a step goes (off by default)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace brae {
namespace rhoCentral {

namespace {

void printIntegrals(const char* tag, const Integrals& I)
{
    std::printf("diag %s mass %.17Le momentum %.17Le %.17Le %.17Le energy %.17Le\n", tag, I.mass, I.mom[0], I.mom[1],
                I.mom[2], I.energy);
    std::printf("diag %s min rho %.9e p %.9e T %.9e e %.9e max rho %.9e p %.9e T %.9e |U| %.9e\n", tag, I.minRho,
                I.minP, I.minT, I.minE, I.maxRho, I.maxP, I.maxT, I.maxMagU);
}

} // namespace


Integrals integrals(const Case& cs, const State& s)
{
    const std::vector<scalar>& V = cs.g.V();
    Integrals r;
    r.minRho = r.minP = r.minT = r.minE = 1e300;
    r.maxRho = r.maxP = r.maxT = r.maxMagU = -1e300;
    for (label i = 0; i < cs.nC; ++i)
    {
        r.mass += static_cast<long double>(V[i])*s.rho[i];
        r.energy += static_cast<long double>(V[i])*s.rhoE[i];
        r.mom[0] += static_cast<long double>(V[i])*s.rhoU[i].x;
        r.mom[1] += static_cast<long double>(V[i])*s.rhoU[i].y;
        r.mom[2] += static_cast<long double>(V[i])*s.rhoU[i].z;
        const auto ulp = [](scalar q) { const scalar a = std::fabs(q); return std::nextafter(a, 1e308) - a; };
        r.ulpMass += static_cast<long double>(V[i])*ulp(s.rho[i]);
        r.ulpEnergy += static_cast<long double>(V[i])*ulp(s.rhoE[i]);
        // a momentum component's update sums fluxes of the whole vector's size (p*Sf, rhoU*phi), so its rounding is
        // on |rhoU|'s scale -- a component near 0 (or an empty direction's, never solved) has no smaller one
        const long double um = static_cast<long double>(V[i])*ulp(mag(s.rhoU[i]));
        r.ulpMom[0] += um;
        r.ulpMom[1] += um;
        r.ulpMom[2] += um;
        r.minRho = std::min(r.minRho, s.rho[i]);
        r.maxRho = std::max(r.maxRho, s.rho[i]);
        r.minP = std::min(r.minP, s.p[i]);
        r.maxP = std::max(r.maxP, s.p[i]);
        r.minT = std::min(r.minT, s.T[i]);
        r.maxT = std::max(r.maxT, s.T[i]);
        r.minE = std::min(r.minE, s.e[i]);
        r.maxMagU = std::max(r.maxMagU, mag(s.U[i]));
    }
    return r;
}


// the moving mesh's AUTO_WRITE objects (milestone K), as OpenFOAM writes them: polyMesh/points (vectorField), meshPhi
// (surfaceScalarField: calculated or wedge with values, empty without), cellMotionU<c> (cellMotion with its value, the
// point-derived constraint types without) and pointMotionU<c> (the point patches as read: uniformFixedValue's value and
// uniformValue, the constraint types bare). A list whose entries are all equal is written `uniform` (Field::writeEntry).
void writeMotion(const Case& cs, const std::string& out, const std::string& timeName, int prec)
{
    const DynamicMotionSolverFvMesh& mo = *cs.motion;
    const VelocityComponentLaplacianFvMotionSolver* vc = mo.velocityComponent();
    auto header = [&](std::ostream& os, const char* cls, const std::string& loc, const std::string& obj) {
        os << "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       " << cls
           << ";\n    location    \"" << loc << "\";\n    object      " << obj << ";\n}\n\n";
    };
    auto list = [&](std::ostream& os, const char* key, const std::vector<scalar>& v) {
        bool same = !v.empty();
        for (scalar x : v) same = same && (x == v.front());
        if (same) { os << key << "uniform " << v.front() << ";\n"; return; }
        os << key << "nonuniform List<scalar> \n" << v.size() << "\n(\n";
        for (scalar x : v) os << x << '\n';
        os << ")\n;\n";
    };
    std::filesystem::create_directories(out + "/polyMesh");
    {
        std::ofstream os(out + "/polyMesh/points");
        os << std::setprecision(prec);
        header(os, "vectorField", timeName + "/polyMesh", "points");
        os << "\n" << cs.m.points().size() << "\n(\n";
        for (const vector& q : cs.m.points()) os << '(' << q.x << ' ' << q.y << ' ' << q.z << ")\n";
        os << ")\n\n\n// ************************************************************************* //\n";
    }
    {
        std::ofstream os(out + "/meshPhi");
        os << std::setprecision(prec);
        header(os, "surfaceScalarField", timeName, "meshPhi");
        os << "dimensions      [0 3 -1 0 0 0 0];\n\n";
        list(os, "internalField   ", mo.meshPhi().internal);
        os << "\nboundaryField\n{\n";
        for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
        {
            const FvPatch& p = cs.patches[pi];
            os << "    " << p.name << "\n    {\n";
            if (p.type == "empty") os << "        type            empty;\n";
            else
            {
                os << "        type            " << (p.type == "wedge" ? "wedge" : "calculated") << ";\n";
                list(os, "        value           ", mo.meshPhi().boundary[pi]);
            }
            os << "    }\n";
        }
        os << "}\n\n\n// ************************************************************************* //\n";
    }
    if (!vc) return;
    const std::string c = vc->cmptName();
    {
        std::ofstream os(out + "/cellMotionU" + c);
        os << std::setprecision(prec);
        header(os, "volScalarField", timeName, "cellMotionU" + c);
        os << "dimensions      [0 1 -1 0 0 0 0];\n\n";
        list(os, "internalField   ", vc->cellMotionU());
        os << "\nboundaryField\n{\n";
        for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
        {
            os << "    " << cs.patches[pi].name << "\n    {\n        type            " << vc->cellMotionUTypes()[pi]
               << ";\n";
            if (vc->cellMotionUTypes()[pi] == "cellMotion")
                list(os, "        value           ", vc->cellMotionUBoundary()[pi]);
            os << "    }\n";
        }
        os << "}\n\n\n// ************************************************************************* //\n";
    }
    {
        std::ofstream os(out + "/pointMotionU" + c);
        os << std::setprecision(prec);
        header(os, "pointScalarField", timeName, "pointMotionU" + c);
        os << "dimensions      [0 1 -1 0 0 0 0];\n\n";
        list(os, "internalField   ", vc->pointMotionU());
        os << "\nboundaryField\n{\n";
        for (const VelocityComponentLaplacianFvMotionSolver::PointPatch& pp : vc->pointPatches())
        {
            os << "    " << pp.name << "\n    {\n        type            " << pp.typeName << ";\n";
            if (pp.fixesValue())
            {
                os << "        value           uniform " << pp.value << ";\n";
                if (pp.typeName == "uniformFixedValue")
                    os << "        uniformValue    constant " << pp.value << ";\n";
            }
            os << "    }\n";
        }
        os << "}\n\n\n// ************************************************************************* //\n";
    }
}

void writeTime(const Case& cs, const State& s, const RunClock& clock, bool uniformTime,
               const std::vector<scalar>* rDeltaT = nullptr, const Arm::TurbulenceOut* turb = nullptr)
{
    const std::string& timeName = clock.timeName();
    const std::string src = cs.caseDir + "/" + cs.startName + "/";
    const std::string out = cs.caseDir + "/" + timeName;
    std::filesystem::create_directories(out);
    // Time::writeObject: writeTimeDict() first, then the registered objects (TimeIO.C:548-555)
    if (uniformTime)
        writeUniformTime(cs.caseDir, timeName, static_cast<double>(clock.value()), clock.timeIndex(),
                         static_cast<double>(clock.deltaT()), static_cast<double>(clock.deltaT0()), cs.writePrecision);
    const int prec = cs.writePrecision;
    // every patch value, in patch order (writeVolField's flat layout; no coupled patch is accepted here)
    writeVolField(src + "U", out + "/U", s.U, cs.patches, prec, s.UB);
    writeVolField(src + "p", out + "/p", s.p, cs.patches, prec, s.pB);
    writeVolField(src + "T", out + "/T", s.T, cs.patches, prec, s.TB);
    // rho is NO_READ, AUTO_WRITE with calculated patches (createFields.H:24-35)
    static const DerivedFieldSpec rhoSpec{"rho", "dimensions      [1 -3 0 0 0 0 0];"};
    writeVolField(src + "T", out + "/rho", s.rho, cs.patches, prec, s.rhoB, &rhoSpec);
    // LTS (milestone I): rDeltaT is READ_IF_PRESENT, AUTO_WRITE, extrapolatedCalculated (createRDeltaT.H): its patch
    // values are the face cells' after fvc::smooth's correctBoundaryConditions
    if (rDeltaT)
    {
        static const DerivedFieldSpec rdtSpec{"rDeltaT", "dimensions      [0 0 -1 0 0 0 0];", "extrapolatedCalculated"};
        std::vector<scalar> b(static_cast<std::size_t>(cs.nB));
        for (label k = 0; k < cs.nB; ++k) b[static_cast<std::size_t>(k)] = (*rDeltaT)[static_cast<std::size_t>(cs.bCell[k])];
        writeVolField(src + "T", out + "/rDeltaT", *rDeltaT, cs.patches, prec, b, &rdtSpec);
    }
    // RAS (milestone J): the model's fields, AUTO_WRITE (kOmegaSSTBase's k_, omega_; eddyViscosity's nut_;
    // EddyDiffusivity's alphat_), their patch classes as read
    if (turb)
    {
        writeVolField(src + "k", out + "/k", turb->k, cs.patches, prec, turb->kB);
        writeVolField(src + "omega", out + "/omega", turb->omega, cs.patches, prec, turb->omegaB);
        writeVolField(src + "nut", out + "/nut", turb->nut, cs.patches, prec, turb->nutB);
        writeVolField(src + "alphat", out + "/alphat", turb->alphat, cs.patches, prec, turb->alphatB);
    }
    if (cs.motion) writeMotion(cs, out, timeName, prec);   // milestone K
}


int runRhoCentralFoam(const std::string& caseDir, bool onDevice, int control)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    setIOPrecision(controlDict.intOr("writePrecision", 6));
    // The viscous solves' loop residual (lduMatrix::residual) rounded as the reference OpenFOAM build on this host
    // architecture rounds it: GCC contracts its multiply-subtracts into fmsub on aarch64 (objdump of v2412's
    // libOpenFOAM.so: three fmsub, no separate fmul/fsub), and x86-64's baseline has no FMA to contract into. Found in
    // milestone I on reservoir_visc: with the products rounded apart, brae's GaussSeidel fixed points sat a median
    // 12% above OpenFOAM's on the 20 e solves both ran to maxIter, and 4 solves stopped at another sweep.
#if defined(__aarch64__)
    setDeviceResidualFused(control != controlResidualUnfused);
#else
    setDeviceResidualFused(false);
#endif
    Case csRead = readRhoCentralCase(caseDir);
    if (control == controlJanafLowOnly) csRead.gas.janafLowOnly = 1;   // the negative control (milestone I)
    // not const: a moving mesh (milestone K) moves cs.m, cs.g and cs.patches in place (fvMesh::movePoints)
    Case cs = std::move(csRead);
    if (cs.motion) cs.motion->attach(cs.m, cs.g, cs.patches);
    // brae's Time reads controlDict's `functions` and says which it does not run (none is registered here); the
    // loop runs on OpenFOAM's accumulating clock instead, whose deltaT their write times still trim (WriteCadence)
    Time time(caseDir, controlDict);
    RunClock clock = *cs.clock;
    const TimeControls& tc = cs.timeControls;
    // Info's precision is IOstream::defaultPrecision(), which controlDict's writePrecision sets (TimeIO.C:377-389)
    const int logPrec = cs.writePrecision;

    std::printf("brae rhoCentralFoam (%s arm): %ld cells, %ld internal faces, start %s, endTime %.*g, deltaT %.*g, "
                "adjustTimeStep %s",
                onDevice ? "cuda" : "host reference", (long)cs.nC, (long)cs.nIF, cs.startName.c_str(), logPrec,
                (double)clock.endTime(), logPrec, (double)clock.deltaT(), tc.adjustTimeStep ? "yes" : "no");
    if (tc.adjustTimeStep) std::printf(" (maxCo %.*g, maxDeltaT %.*g)", logPrec, (double)tc.maxCo, logPrec,
                                       (double)tc.maxDeltaT);
    if (cs.lts) std::printf("\nUsing LTS: localEuler, rDeltaT = max(1/maxDeltaT, surfaceSum(amaxSf)/(2 maxCo V)) "
                            "(maxCo %.*g, maxDeltaT %.*g), fvc::smooth coefficient %.*g", logPrec, (double)tc.maxCo,
                            logPrec, (double)tc.maxDeltaT, logPrec, (double)cs.rDeltaTSmoothingCoeff);
    if (cs.gas.janaf)
        std::printf("\nthermo: janaf, Tlow %.*g, Tcommon %.*g, Thigh %.*g (Cp, Cv, gamma and alphahe at each T)", logPrec,
                    (double)cs.gas.Tlow, logPrec, (double)cs.gas.Tcommon, logPrec, (double)cs.gas.Thigh);
    std::printf("\nfluxScheme: %s\nreconstruct: rho %s, T %s, U %s\n", cs.tadmor ? "Tadmor" : "Kurganov",
                cs.reconRho.c_str(), cs.reconT.c_str(), cs.reconU.c_str());
    if (cs.viscous)
    {
        auto ctl = [&](const char* f, const SmoothSolveControls& c) {
            std::printf("  %s: smoothSolver %s, nSweeps %d, tolerance %.*g, relTol %.*g, maxIter %d, minIter %d\n", f,
                        c.smoother.c_str(), c.nSweeps, logPrec, (double)c.tolerance, logPrec, (double)c.relTol,
                        c.maxIter, c.minIter);
        };
        if (cs.gas.sutherland)
            std::printf("viscous: sutherland As %.*g, Ts %.*g (laminar)\n", logPrec, (double)cs.gas.As, logPrec,
                        (double)cs.gas.Ts);
        else if (cs.gas.janaf)
            std::printf("viscous: mu %.*g, Pr %.*g, alphaEff = CpByCpv(T)*mu/Pr per cell (laminar, const transport, "
                        "janaf)\n", logPrec, (double)cs.gas.mu, logPrec, (double)(1.0/cs.gas.rPr));
        else
            std::printf("viscous: mu %.*g, Pr %.*g, alphaEff %.*g (laminar, const transport)\n", logPrec,
                        (double)cs.gas.mu, logPrec, (double)(1.0/cs.gas.rPr), logPrec, (double)gasAlphaEff(cs.gas));
        ctl("U", cs.solveU);
        ctl("e", cs.solveE);
    }
    else
        std::printf("inviscid: every mu is 0\n");
    if (cs.ras) std::printf("RAS: kOmegaSST (compressible::turbulenceModel: muEff = mu + rho*nut, alphaEff = "
                            "CpByCpv*(alpha + alphat))\n");
    std::printf("solution directions: (%d %d %d)\n", cs.sd.d[0], cs.sd.d[1], cs.sd.d[2]);

    // a restart's time state (Time::setControls): the start's index, and its deltaT where uniform/time gave it
    std::printf("diag start time %s index %ld deltaT %.17g\n", cs.startName.c_str(), clock.startTimeIndex(),
                (double)clock.deltaT());

    PatchFields pf;
    State s = cpu::rhoCentral::createFields(cs, pf, control);
    const Integrals I0 = integrals(cs, s);
    printIntegrals("start", I0);

    const bool clockControl = control == controlUntrimmed || control == controlLaggedCourant;
    const bool restartControl = control == controlRestartIndexZero || control == controlRestartGridFromZero
                             || control == controlRestartIoExtrapolate || control == controlNoUniformTime;
    const bool driverControl = clockControl || control == controlNoPurge || restartControl;   // either arm
    const bool ioControl = control == controlInletOutletInverted || control == controlLaggedPhi
                        || control == controlInletOutletNewT;
    const bool limiterControl = control == controlLimiterVanLeer || control == controlLimiterScalarOnV;
    const bool patchControl = control == controlNoTransform || control == controlLaggedRhoB || ioControl
                           || limiterControl;   // both arms
    if (control != controlNone && !driverControl && !patchControl && !onDevice)
        throw std::runtime_error("BRAE_CONTROL_RCF upwind and noPressureJump are device-arm controls; the host "
                                 "reference has none.");
    if (clockControl && !tc.adjustTimeStep)
        throw std::runtime_error("BRAE_CONTROL_RCF untrimmed and laggedCourant are controls of adjustTimeStep; this "
                                 "case runs a fixed deltaT.");
    // a control that cannot change this case would pass any gate: refuse it
    if (ioControl && !(cs.ioU || cs.ioP || cs.ioT))
        throw std::runtime_error("BRAE_CONTROL_RCF inletOutletInverted, laggedPhi and inletOutletNewT are controls of "
                                 "inletOutlet; no patch of this case's U, p or T is one.");
    if (control == controlInletOutletNewT && !cs.ioT)
        throw std::runtime_error("BRAE_CONTROL_RCF inletOutletNewT is a control of T's inletOutlet; this case's T has "
                                 "none.");
    if (control == controlLimiterVanLeer && cs.limRho == kVanLeerTwoByk && cs.limT == kVanLeerTwoByk
        && cs.limU == kVanLeerTwoByk)
        throw std::runtime_error("BRAE_CONTROL_RCF limiterVanLeer is a control of the other limiters; this case "
                                 "reconstructs with vanLeer and vanLeerV already.");
    if ((control == controlNoDynamicHead || control == controlNozzleFlowInverted) && cs.tpFaces.empty()
        && cs.ttFaces.empty())
        throw std::runtime_error("BRAE_CONTROL_RCF noDynamicHead and nozzleFlowInverted are controls of totalPressure and "
                                 "totalTemperature; this case has neither.");
    if ((control == controlWaveNoRelax || control == controlWaveVolumeFlux) && cs.wtFaces.empty())
        throw std::runtime_error("BRAE_CONTROL_RCF waveNoRelax and waveVolumeFlux are controls of waveTransmissive; this "
                                 "case has none.");
    if (control == controlWaveNoRelax)
    {
        bool relaxes = false;
        for (scalar l : cs.wtLInf) relaxes = relaxes || l > 0;
        if (!relaxes)
            throw std::runtime_error("BRAE_CONTROL_RCF waveNoRelax is a control of waveTransmissive's lInf; no patch of "
                                     "this case sets one.");
    }
    if (control == controlTotalTemperatureOnce && (cs.ttFaces.empty() || !cs.viscous))
        throw std::runtime_error("BRAE_CONTROL_RCF totalTemperatureOnce is a control of totalTemperature in a viscous "
                                 "run (the e matrix's construction); this case has no such patch or no e corrector.");
    if (control == controlNoPurge && cs.purgeWrite <= 0)
        throw std::runtime_error("BRAE_CONTROL_RCF noPurge is a control of purgeWrite; this case's is 0.");
    if ((control == controlRestartIndexZero && clock.startTimeIndex() == 0)
        || ((control == controlRestartGridFromZero || control == controlRestartIoExtrapolate) && cs.startTime == 0))
        throw std::runtime_error("BRAE_CONTROL_RCF restartIndexZero, restartGridFromZero and restartIoExtrapolate are "
                                 "controls of a restart; this case starts from 0 (or with no index to drop).");
    if (control == controlRestartIoExtrapolate && !(cs.ioU || cs.ioP || cs.ioT))
        throw std::runtime_error("BRAE_CONTROL_RCF restartIoExtrapolate is a control of inletOutlet; this case has "
                                 "none.");
    if (control == controlRestartIndexZero) clock.forgetStartIndex();
    if (control == controlRestartGridFromZero) clock.measureWritesFromZero();
    // Time::writeObject's purge after each write (TimeIO.C:557-582); the control keeps every directory
    PurgeWrite purge(control == controlNoPurge ? 0 : cs.purgeWrite, /*warnNegative=*/true);

    std::unique_ptr<Arm> arm;
    if (onDevice)
    {
        arm = gpu::rhoCentral::makeDeviceArm(cs, s, pf, driverControl ? controlNone : control);
        std::printf("diag device-memory %.3f MiB\n", arm->deviceBytes()/1048576.0);
    }
    else
    {
        if (cs.ras)
            throw std::runtime_error("brae rhoCentralFoam: the host reference arm (BRAE_RHOCENTRALFOAM_ARM=cpu) does not "
                                     "run a RAS model; milestone J ports it on the CUDA arm.");
        if (cs.motion)
            throw std::runtime_error("brae rhoCentralFoam: the host reference arm (BRAE_RHOCENTRALFOAM_ARM=cpu) does not "
                                     "move a mesh; milestone K ports mesh motion on the CUDA arm.");
        if (cs.viscous)
            throw std::runtime_error("brae rhoCentralFoam: the host reference arm (BRAE_RHOCENTRALFOAM_ARM=cpu) runs "
                                     "inviscid cases only; mu > 0 runs the viscous correctors, which milestone G "
                                     "ports on the CUDA arm.");
        if (cs.lts)
            throw std::runtime_error("brae rhoCentralFoam: the host reference arm (BRAE_RHOCENTRALFOAM_ARM=cpu) does not "
                                     "run local time stepping (ddtSchemes localEuler); milestone I ports it on the CUDA "
                                     "arm.");
        if (cs.nozzleBoundaries())
            throw std::runtime_error("brae rhoCentralFoam: the host reference arm (BRAE_RHOCENTRALFOAM_ARM=cpu) does not "
                                     "run totalPressure, totalTemperature or waveTransmissive; milestone H ports them on "
                                     "the CUDA arm.");
        arm = cpu::rhoCentral::makeHostArm(cs, s, pf, patchControl ? control : controlNone);
    }

    // rhoCentralFoam.C:80-81: CoNum is 0 until the first step's centralCourantNo.H, so the first setDeltaT.H grows
    // deltaT by the full 1.2
    scalar CoNum = 0;
    double stepSeconds = 0;
    double motionSeconds = 0;   // mesh.update(), outside step-seconds (milestone K)
    long steps = 0;
    // the log of the steps the arm has not reported yet: their deltaT and time name (the device arm reports a window
    // of steps at once at a fixed deltaT)
    std::vector<std::pair<scalar, std::string>> unlogged;
    // ...and, on a moving mesh (milestone K), each step's motion solve: its log line comes between setDeltaT.H's and
    // centralCourantNo.H's, where mesh.update() runs (rhoCentralFoam.C:96)
    std::vector<SolverPerformance> unloggedMotion;
    const std::string motionField = (cs.motion && cs.motion->velocityComponent())
        ? "cellMotionU" + cs.motion->velocityComponent()->cmptName() : std::string();
    std::printf("\nStarting time loop\n\n");
    // the log, OpenFOAM's lines a step: setDeltaT.H's "deltaT = " under adjustTimeStep, centralCourantNo.H's, Time
    auto flushArm = [&]()
    {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<StepReport> reps = arm->flush();
        stepSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (reps.size() != unlogged.size())
            throw std::logic_error("brae rhoCentralFoam: the arm reported a different number of steps than it ran.");
        for (std::size_t i = 0; i < reps.size(); ++i)
        {
            const StepReport& rep = reps[i];
            const std::string& tn = unlogged[i].second;
            if (tc.adjustTimeStep && !cs.lts) std::printf("deltaT = %.*g\n", logPrec, (double)unlogged[i].first);
            if (!motionField.empty())   // PCG's line (SolverPerformance.C), DIC-preconditioned
                std::printf("DICPCG:  Solving for %s, Initial residual = %.*g, Final residual = %.*g, No Iterations %d\n",
                            motionField.c_str(), logPrec, (double)unloggedMotion[i].initialResidual, logPrec,
                            (double)unloggedMotion[i].finalResidual, unloggedMotion[i].nIterations);
            std::printf("Mean and max Courant Numbers = %.*g %.*g\n", logPrec, (double)rep.meanCoNum, logPrec,
                        (double)rep.CoNum);
            if (cs.lts)   // setRDeltaT.H:26-28
                std::printf("Flow time scale min/max = %.*g, %.*g\n", logPrec, (double)rep.flowMin, logPrec,
                            (double)rep.flowMax);
            std::printf("Time = %s\n\n", tn.c_str());
            // the viscous correctors' solves, OpenFOAM's lines (SolverPerformance.C:95-117) at Info's precision,
            // writePrecision (printOfSolveLine's %g is 6 digits)
            for (int j = 0; j < rep.nSolves; ++j)
                std::printf("smoothSolver:  Solving for %s, Initial residual = %.*g, Final residual = %.*g, "
                            "No Iterations %d\n", rep.solves[j].field, logPrec, (double)rep.solves[j].initialResidual,
                            logPrec, (double)rep.solves[j].finalResidual, rep.solves[j].nIterations);
            if (rep.status)
                throw std::runtime_error(
                    "step to time " + tn + " left an unphysical state (status " + std::to_string(rep.status)
                    + ": 1 negative temperature seed, 2 e->T inversion not converged in 100 iterations -- both "
                      "FatalErrors in OpenFOAM's thermo -- 4 rho, p or T not positive, or e not finite).");
            CoNum = rep.CoNum;
            // the control: the Courant number as it would be at the step before's deltaT
            if (control == controlLaggedCourant) CoNum = rep.CoNum/unlogged[i].first*clock.deltaT0();
        }
        unlogged.clear();
        unloggedMotion.clear();
    };
    // rhoCentralFoam.C:85-97 and 288-294, !LTS: Time::run, setDeltaT.H, ++runTime, the step, runTime.write(). LTS
    // (milestone I): no setDeltaT.H -- runTime advances by the deltaT it holds (controlDict's, or the start time's
    // uniform/time) -- and ++runTime after setRDeltaT.H, :186-190; nothing in the step reads the time between
    interPhase::start();
    while (clock.run())
    {
        if (tc.adjustTimeStep && !cs.lts)
        {
            const scalar dt = setDeltaT(clock.deltaT(), CoNum, tc);   // setDeltaT.H
            if (control == controlUntrimmed) clock.setDeltaTNoAdjust(dt);
            else clock.setDeltaT(dt);                                  // Time::setDeltaT: adjustDeltaT
        }
        ++clock;
        unlogged.emplace_back(clock.deltaT(), clock.timeName());
        // mesh.update() (rhoCentralFoam.C:96), !LTS: the motion at the new time, before the step
        interPhase::mark("loop: setDeltaT.H, ++runTime");
        if (cs.motion)
        {
            const auto m0 = std::chrono::steady_clock::now();
            unloggedMotion.push_back(arm->moveMesh(clock.value(), clock.deltaT(), clock.timeIndex()));
            motionSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - m0).count();
            interPhase::mark("loop: mesh.update()");
        }
        const auto t0 = std::chrono::steady_clock::now();
        arm->step(clock.deltaT());
        stepSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        interPhase::mark("loop: the step");
        ++steps;
        const bool write = clock.writeTime();
        if (tc.adjustTimeStep || write || arm->pending() >= arm->window() || !clock.run()) flushArm();
        interPhase::mark("loop: flush (reports, log)");
        if (write)
        {
            arm->download(s);
            Arm::TurbulenceOut to;
            const bool turb = arm->turbulenceFields(to);
            writeTime(cs, s, clock, control != controlNoUniformTime, arm->rDeltaT(), turb ? &to : nullptr);
            purge.written(cs.caseDir, clock.timeName());
            std::printf("written %s\n", clock.timeName().c_str());
            interPhase::mark("loop: write");
        }
    }
    flushArm();
    time.end();
    // no write here: rhoCentralFoam.C writes only through runTime.write(), on write steps
    arm->download(s);
    scalar bndMass = 0, bndEnergy = 0;
    vector bndMom{0, 0, 0};
    arm->boundaryOutflow(bndMass, bndMom, bndEnergy);
    std::printf("diag end time %s value %.17g steps %ld\n", clock.timeName().c_str(), (double)clock.value(), steps);
    if (cs.ioU || cs.ioP || cs.ioT)
    {
        long long inflow = 0, faceSteps = 0;
        arm->inletOutletInflow(inflow, faceSteps);
        std::printf("diag inletOutlet inflow face-steps %lld of %lld\n", inflow, faceSteps);
    }

    // ---- diagnostics ------------------------------------------------------------------------------------------------
    const Integrals I1 = integrals(cs, s);
    printIntegrals("end", I1);
    // conservation: total(end) - total(start) + integral of the boundary outflow; zero to rounding for a conservative
    // scheme, whatever the boundary does
    const long double eM = (I1.mass - I0.mass) + bndMass;
    const long double eE = (I1.energy - I0.energy) + bndEnergy;
    const long double eU[3] = {(I1.mom[0] - I0.mom[0]) + bndMom.x, (I1.mom[1] - I0.mom[1]) + bndMom.y,
                               (I1.mom[2] - I0.mom[2]) + bndMom.z};
    std::printf("diag boundary-outflow mass %.17e momentum %.17e %.17e %.17e energy %.17e\n", (double)bndMass,
                (double)bndMom.x, (double)bndMom.y, (double)bndMom.z, (double)bndEnergy);
    std::printf("diag balance mass %.6Le energy %.6Le momentum %.6Le %.6Le %.6Le "
                "(relative to the start totals: mass %.3Le energy %.3Le)\n",
                eM, eE, eU[0], eU[1], eU[2], fabsl(eM)/fabsl(I0.mass), fabsl(eE)/fabsl(I0.energy));
    // the round-off a conservative explicit update leaves: each cell's total is rounded a few times a step (the flux sum
    // over V, times deltaT, added to the old value): four ulps of each cell's total a step, summed over the cells of
    // the start or the end state, whichever sums larger. A non-conservative scheme misses by the fluxes' size.
    const long double k = 4.0L*steps;
    if (cs.viscous)
    {
        vector rm;
        scalar re;
        arm->solveResidual(rm, re);
        std::printf("diag solve-residual momentum %.6e %.6e %.6e energy %.6e (deltaT*sum(b - A x) of the implicit "
                    "correctors: the balance is minus this, to round-off)\n", (double)rm.x, (double)rm.y, (double)rm.z,
                    (double)re);
        vector lm, im;
        scalar le, ie;
        arm->solveResidualL1(lm, le);
        arm->solveIdentity(im, ie);
        std::printf("diag solve-residual-l1 momentum %.6e %.6e %.6e energy %.6e (deltaT*sum|b - A x|: the "
                    "unnormalized algebraic residual)\n", (double)lm.x, (double)lm.y, (double)lm.z, (double)le);
        std::printf("diag balance-accounted energy %.6Le momentum %.6Le %.6Le %.6Le (the balance plus the solves' "
                    "remainder)\n", eE + (long double)re, eU[0] + (long double)rm.x, eU[1] + (long double)rm.y,
                    eU[2] + (long double)rm.z);
        std::printf("diag solve-identity momentum %.3e %.3e %.3e energy %.3e (the largest error over the solves of "
                    "sum rho V (x - x_p) = deltaT*(patch fluxes - sum(b - A x)), over sum rho V |x|)\n", (double)im.x,
                    (double)im.y, (double)im.z, (double)ie);
    }
    std::printf("diag roundoff mass %.3Le energy %.3Le momentum %.3Le %.3Le %.3Le (four ulps a cell a step)\n",
                k*std::max(I0.ulpMass, I1.ulpMass), k*std::max(I0.ulpEnergy, I1.ulpEnergy),
                k*std::max(I0.ulpMom[0], I1.ulpMom[0]), k*std::max(I0.ulpMom[1], I1.ulpMom[1]),
                k*std::max(I0.ulpMom[2], I1.ulpMom[2]));
    if (cs.motion)
        std::printf("diag gcl max |V - V0 - deltaT*sum(meshPhi)|/V %.6e (every cell, every step)\n",
                    (double)arm->gclMax());
    std::printf("diag steps %ld step-seconds %.6f per-step-us %.3f cell-steps-per-second %.6e\n", steps, stepSeconds,
                1e6*stepSeconds/(steps ? steps : 1), stepSeconds > 0 ? (double)cs.nC*steps/stepSeconds : 0.0);
    if (cs.motion)   // step-seconds times arm->step alone; mesh.update() runs before it
        std::printf("diag motion-seconds %.6f per-step-us %.3f (mesh.update(), not in step-seconds)\n", motionSeconds,
                    1e6*motionSeconds/(steps ? steps : 1));
    interPhase::report(steps);
    std::printf("End\n");
    return 0;
}

} // namespace rhoCentral
} // namespace brae
