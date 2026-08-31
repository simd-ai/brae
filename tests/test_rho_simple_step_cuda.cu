// The CUDA rhoSimpleFoam DRIVER against the _cpp driver, over MULTIPLE ITERATIONS.
//
// The reference driver is itself gated end-to-end against real OpenFOAM
// (tests/rho_simple_end_to_end_vs_openfoam.sh), so this closes OpenFOAM -> _cpp -> CUDA for the whole
// solver rather than for one equation at a time.
//
// WHY MULTIPLE ITERATIONS, AND WHY THAT IS THE POINT OF THIS FILE. Every component this driver composes
// already has a per-stage gate passing at 1e-12 or better. What those cannot see is what the driver
// FEEDS them on the second pass: the AMG-cache defect in the incompressible twin was EXACT at iteration
// 1 and 1.3e-01 wrong at iteration 2, with every per-stage gate green throughout, because the pressure
// buffers were reallocated each iteration and the persistent hierarchy attached to a different fine
// matrix each time. A one-iteration driver test would have certified it. This one runs N iterations and
// compares after EVERY one, so the first divergent iteration is named rather than the last.
//
// LAMINAR, deliberately. The closure has its own device gate (test_rho_kepsilon_cuda) which pins its
// arithmetic at 1e-16; running it inside this loop would fold two more Krylov solves -- host pbicgstab
// against device BiCGStab -- into a comparison that is meant to be about the ORDER of the iteration.
// The turbulent composition is what rho_simple_end_to_end_vs_openfoam covers.
//
// THE THERMO IS SUPPLIED TO BOTH SIDES FROM THE SAME FUNCTION. The device driver takes thermoCorrect and
// updateRho as hooks precisely because they are thermo operations, so the gate hands both sides
// cpu::rhoSimple::thermoCorrect and cpu::rhoSimple::updateRho. Anything else would be comparing two
// thermodynamic models as well as two drivers, and a disagreement could not be attributed.
//
// WHAT THIS GATE FOUND, on its first run, and what no per-stage gate could see. Ux disagreed by
// 5.406e-04 at iteration 1 and grew monotonically to 3.9e-03 by iteration 8, with phi tracking it while
// p, T, he and rho all sat at ~1e-7. Three measurements localised it:
//   * INSENSITIVE to the linear-solver tolerance -- 5.406e-04 at 1e-16, 1e-14 and 1e-10 alike -- so a
//     discretisation or wiring difference, not a Krylov stopping point;
//   * with the FLUX ZEROED on both sides, leaving only divDevRhoReff, the two matrices sat at a CONSTANT
//     ratio of 1.161 across diag, source and internalCoeffs -- a systematic scaling, not round-off;
//   * 1.161 is exactly p/(R*T) = 100000/(287.1*300), this fixture's density.
// The driver was wiring the DYNAMIC muEff into RhoMomentumInput's KINEMATIC nuEff slot, so the momentum
// module formed rho*nuEff and multiplied by rho a second time. It is the same defect class the rhoUEqn
// gate's own control was built to catch, in the opposite direction -- and it was invisible to that gate,
// which injects the viscosity itself and therefore never exercises the driver's choice of slot.
// Afterwards: Ux 1.47e-11, p 7.32e-14, T 3.10e-13 over the same 8 iterations.
//
// Run: test_rho_simple_step_cuda <caseDir> <timeDir> [iterations]
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "rhoSimpleFoam_cpp.cuh"
#include "rhoSimpleFoam.cuh"
#include "rhoCreateFields.cuh"
#include "rhoThermoDevice.cuh"
#include "kEpsilon.cuh"
#include "scheme_parse.cuh"     // parseFieldDivScheme -- div(phi,k|epsilon|omega) from the case          // gpu::kEpsilonRAS -- the device closure the turbulent arm drives
#include "transport_model.cuh"   // transportMu: nu = mu(T)/rho for the closure inputs
#include "linearViscousStress_cpp.cuh"   // effectiveFaceViscosity -- the host driver's own rho interpolation
#include "thermo_model.cuh"
#include "equation_of_state.cuh"   // perfectGasPsi -- the oracle for the device thermo control
#include "device_fvoptions.cuh"   // DevicePorosity   // hConstTToHe -- the reference's OWN T->he conversion

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

static double relL2(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    double num = 0.0, den = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

static void report(const char* what, double got, double bound)
{
    const bool ok = got < bound;
    if (!ok) ++g_fails;
    std::printf("     %-40s %.6e   %s\n", what, got, ok ? "ok" : "FAIL");
}

static void check(const char* what, bool ok)
{
    if (!ok) ++g_fails;
    std::printf("     %-40s %s\n", what, ok ? "ok" : "FAIL");
}

// The flattening is rhoCreateFields.cu's, not a fourth private copy of it -- the padding convention is
// part of the contract and drifts the moment it is duplicated.
static std::vector<scalar> flat(const std::vector<std::vector<scalar>>& v,
                                const std::vector<FvPatch>& fvp, int nBnd, scalar pad)
{ return gpu::rhoSimple::flattenBoundary(v, fvp, nBnd, pad); }

static std::vector<scalar> flatBnd(const GeometricField<scalar>& f,
                                   const std::vector<FvPatch>& fvp, int nBnd, scalar pad)
{ return gpu::rhoSimple::flattenFieldBoundary(f, fvp, nBnd, pad); }

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <timeDir> [iterations]\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], startT = argv[2];
    int  iters       = 8;
    bool boundaryArm  = false;
    bool deviceThermo = false;
    bool turbulentArm = false;
    for (int a = 3; a < argc; ++a)
    {
        const std::string arg = argv[a];
        if      (arg == "--boundary")      boundaryArm  = true;
        else if (arg == "--device-thermo") deviceThermo = true;
        else if (arg == "--turbulent")     turbulentArm = true;
        else                               iters = std::atoi(argv[a]);
    }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    // TWO INDEPENDENT field sets from the SAME files. `hf` is the host driver's state; `df` is the host
    // mirror the device driver's thermo hooks operate on. Building them separately rather than copying
    // matters: RhoSimpleFields owns unique_ptr patch fields and is not copyable, and a shallow share
    // would let one run's thermo write into the other's state.
    cpu::rhoSimple::RhoSimpleFields hf =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution, m, g, fvp);
    cpu::rhoSimple::RhoSimpleFields df =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution, m, g, fvp);

    // THE BOUNDARY ARM. The updateCoeffs() block this driver grew can only be exercised by a fixture
    // carrying patches whose coefficients move with the solution, and the only compressible one is
    // sbMatched -- four inletOutlet patches and a flowRateInletVelocity inlet, and a kEpsilon case.
    //
    // ONE ITERATION, and turbulent. Both are forced by what this file is: the device driver takes
    // turbulence->correct() as a hook and this gate leaves it null, so from iteration 2 the host is
    // running a closure the device is not and the trajectories separate on that, not on the boundaries.
    // Laminarising instead is worse -- sbMatched carries `transonic yes` and its nut is 30x the laminar
    // viscosity, so removing the closure makes the case unstable and it drove rho negative by iteration
    // 3 ("gSum(rho*magSf) is not positive"). Iteration 1 is where the boundary conditions are the ONLY
    // difference, and it is where the measurement below was taken.
    //
    // WHAT IT CATCHES, measured by disabling updateBoundaryCoeffs and rerunning: Ux at iteration 1 goes
    // from 4.496e-12 to 1.276e-02, nine orders. Disabling only the flowRate half leaves it at 4.607e-12,
    // so on this fixture the whole of that is the INLETOUTLET SWITCH: sbMatched seeds
    // `internalField uniform (0 0 0)`, so phi is zero on every boundary face at the start, the switch
    // must read that as outflow and extrapolate, and the device seeds those faces fixedValue instead.
    // The flowRate half moves nothing at iteration 1 -- rho has not moved yet -- which is exactly why it
    // gets an arithmetic control of its own at the end of this file rather than a trajectory bound.
    if (boundaryArm && iters == 8) iters = 1;
    std::printf("rhoSimpleFoam DRIVER: CUDA vs _cpp  (%d cells, %d iterations, %s%s)\n",
                (int)nC, iters, hf.turbulent ? hf.rasModel.c_str() : "laminar",
                boundaryArm ? ", BOUNDARY arm" : (deviceThermo ? ", DEVICE thermo" : ""));
    if (!boundaryArm && !turbulentArm) check("the fixture is LAMINAR (see the header)", !hf.turbulent);
    if (turbulentArm) check("the TURBULENT arm was given a turbulent fixture", hf.turbulent);
    {
        label coupled = 0;
        for (const FvPatch& p : fvp)
            if (p.type == "cyclic" || p.type == "cyclicAMI" || p.type == "processor") ++coupled;
        check("no coupled patches (the driver refuses them)", coupled == 0);
    }

    // THE CASE'S OWN CONTROLS, on both sides. Running SIMPLE unrelaxed is not a neutral simplification:
    // with relaxU = 1 and relaxP = 1 this fixture diverges to NaN by iteration 7 -- on the HOST as much
    // as on the device -- and a gate comparing two divergent trajectories measures nothing. The
    // relaxation factors, the `bounded` div schemes and the ORTHOGONAL laplacian all come from the case
    // file, because a driver gate that silently substitutes its own numerics is testing something the
    // solver never runs.
    const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
    const FoamDict* req = rf ? rf->subDict("equations") : nullptr;
    const FoamDict* rfl = rf ? rf->subDict("fields") : nullptr;

    cpu::rhoSimple::StepInput hin;
    hin.consistent = simpleDict && simpleDict->wordOr("consistent", "no") == "yes";
    hin.transonic  = simpleDict && simpleDict->wordOr("transonic", "no") == "yes";
    hin.tolU = hin.tolHe = hin.tolP = 1e-14;
    hin.maxIter = 2000;
    hin.relaxU  = req ? req->scalarOr("U", 1.0) : 1.0;
    hin.relaxHe = req ? req->scalarOr(hf.heName, 1.0) : 1.0;
    hin.relaxEquationU  = (req != nullptr) && req->found("U");
    hin.relaxEquationHe = (req != nullptr) && req->found(hf.heName);
    hin.relaxP  = rfl ? rfl->scalarOr("p", 1.0) : 1.0;
    hin.relaxRho = rfl ? rfl->scalarOr("rho", 1.0) : 1.0;
    hin.relaxPEqn = req ? req->scalarOr("p", 1.0) : 1.0;
    hin.relaxPEqnSpecified = (req != nullptr) && req->found("p");
    // THE CLOSURE'S CONTROLS, on the turbulent arm. The gate set none of these, which was harmless while
    // every arm ran laminar and is not once the closure runs: StepInput defaults relaxK/relaxEpsilon to
    // 1.0 with relaxEquationK/Eps TRUE, so the host closure would apply the dominance clamp the case
    // never asked for while the device side used whatever it was handed.
    if (turbulentArm)
    {
        hin.relaxK          = req ? req->scalarOr("k", 1.0) : 1.0;
        hin.relaxEpsilon    = req ? req->scalarOr("epsilon", 1.0) : 1.0;
        hin.relaxEquationK  = (req != nullptr) && req->found("k");
        hin.relaxEquationEps= (req != nullptr) && req->found("epsilon");
        hin.tolTurb         = 1e-12;   // the case's own tolerance (fvSolution)
        hin.relTolTurb      = 0.0;
        // FROM THE CASE, not hardcoded -- see the cpp harness. The device closure's own refusal
        // (kEpsilon.cu hasNonUpwindDivScheme) becomes reachable through exactly this parse.
        {
            const char* secondT = (hf.rasModel == "kOmegaSST") ? "omega" : "epsilon";
            const FieldDivScheme dK = parseFieldDivScheme(caseDir, "k");
            const FieldDivScheme dS = parseFieldDivScheme(caseDir, secondT);
            hin.boundedTurb = dK.bounded;
            if (dK.limited || dS.limited)           hin.turbDivUnsupported = "Gauss limitedLinear";
            if (dK.linearUpwind || dS.linearUpwind) hin.turbDivUnsupported = "Gauss linearUpwind";
            if (dK.bounded != dS.bounded)
                hin.turbDivUnsupported = "bounded on only one of the two turbulence entries";
        }
    }
    hin.boundedU = hin.boundedHe = hin.boundedKE = true;   // `bounded Gauss upwind` on all three
    hin.correctedLaplacian = false;                        // `Gauss linear orthogonal`

    // THE HOST REFERENCE HAS NO NON-ORTHOGONAL CORRECTOR LOOP -- cpu::rhoSimple::StepInput carries no
    // nNonOrthogonalCorrectors at all and solves the pressure equation exactly once. The device driver
    // does implement the loop (solutionControlI.H runs it nNonOrth+1 times), so this gate runs it with
    // one pass and asserts the fixture asks for that, rather than comparing a one-pass reference against
    // a multi-pass device and calling the difference a defect. A case with nNonOrthogonalCorrectors > 0
    // is outside what this gate can compare until the reference grows the loop.
    const label caseNonOrth =
        simpleDict ? (label)simpleDict->scalarOr("nNonOrthogonalCorrectors", 0) : 0;
    check("the case asks for 0 non-orthogonal correctors", caseNonOrth == 0);

    // ---- the DEVICE side ---------------------------------------------------------------------
    // THE DEVICE PROJECTION, from the module that owns it. Every line this replaces was hand-rolled
    // here before rhoCreateFields.cu existed -- the mesh, the four boundary objects, the field upload,
    // the boundary flattening and the two masks. Calling it is also the integration proof: if the driver
    // still lands where it did on a hand-rolled state, the projection is equivalent to it.
    gpu::rhoSimple::RhoDeviceFields dev =
        gpu::rhoSimple::createDeviceFields(df, m, g, fvp);
    const DeviceMesh&           dm   = dev.dm;
    DeviceVectorBoundary&       dbU  = dev.dbU;
    DeviceBoundary&             dbP  = dev.dbP;
    DeviceBoundary&             dbHe = dev.dbHe;
    DeviceBoundary&             dbT  = dev.dbT;
    gpu::rhoSimple::RhoSolverFields& gf = dev.f;

    // The workspace lives ACROSS iterations, deliberately: allocating the pressure buffers fresh each
    // step gives the persistent AMG hierarchy a different fine matrix every time, and the cached V-cycle
    // and PCG graphs are keyed on that matrix.
    gpu::rhoSimple::RhoSolverWorkspace w;

    // THE HOOKS. Both sides get the same thermo, for the reason in the header. Each pulls the device
    // state into df, runs the reference's own function, and pushes the result back -- so what is being
    // compared is the ORDER the driver calls them in and the state it calls them on, which is exactly
    // what a per-stage gate cannot see.
    auto pullPT = [&]()
    {
        df.p.internal  = gf.p.host();
        df.he.internal = gf.he.host();
        // The patch values too: thermoCorrect reads he's boundary and writes T's and psi's.
        const std::vector<scalar> pb = gf.pBnd.host(), hb = gf.heBnd.host();
        std::size_t k = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            std::vector<scalar> a(fvp[pi].size), b(fvp[pi].size);
            for (label i = 0; i < fvp[pi].size; ++i, ++k) { a[i] = pb[k]; b[i] = hb[k]; }
            df.p.boundary[pi]->setStoredValues(std::move(a));
            df.he.boundary[pi]->setStoredValues(std::move(b));
        }
    };
    auto pushThermo = [&]()
    {
        gf.T.copyFrom(df.T.internal);
        gf.TBnd.copyFrom(flatBnd(df.T, fvp, dm.nBndFaces, 0.0));
        gf.psi.copyFrom(df.psi);
        gf.psiBnd.copyFrom(flat(df.psiBnd, fvp, dm.nBndFaces, 0.0));
    };

    gpu::rhoSimple::RhoStepInput gin;
    // constrainHbyA's mask and adjustPhi's mask, from the projection. They answer DIFFERENT questions
    // and rhoCreateFields.cu is where that distinction is made once.
    gin.takeUAtBoundary = &dev.takeUAtBoundary;
    gin.adjustable      = &dev.adjustable;
    gin.consistent = hin.consistent;
    gin.transonic  = hin.transonic;
    gin.isE = (hf.heName == "e");
    gin.tolU = gin.tolHe = gin.tolP = 1e-14;
    gin.maxIter = 2000;
    gin.pRefCell  = hf.pressureControl.refCell;
    gin.pRefValue = hf.pressureControl.refValue;

    // THE CASE'S OWN pressureControl, on every arm. The forced-limiter block below overrides these with
    // bounds tightened to make the clamp bite, and it is SKIPPED on the boundary and turbulent arms --
    // which left those arms giving the limits to the host (through hf.pressureControl, which
    // rhoSimpleStep reads directly) and not to the device. sbMatched names `pMin 1000`, so the host
    // clamped and the device did not: measured on the turbulent arm as p 1.790e-03 at iteration 2 while
    // T, he and rho were all at 1e-12.
    gin.limitMaxP = hf.pressureControl.limitMaxP;
    gin.pMaxLimit = hf.pressureControl.pMax;
    gin.limitMinP = hf.pressureControl.limitMinP;
    gin.pMinLimit = hf.pressureControl.pMin;

    // pressureControl::limit, from the case -- AND made to bind, because the case's own limit does not.
    // rhoBox names `pMin 1000` while p sits near 1e5, so the shipped limit never clips and a driver that
    // dropped it entirely passed this gate for as long as it did. The bounds below are tightened around
    // the field's own range so the clamp is exercised on both paths; the host reference is given exactly
    // the same ones, so what is compared is still device-against-host and not one code's limiter against
    // another's absence.
    // The floor comes from a PRELIMINARY unlimited run, not from the initial field. p starts uniform at
    // 1e5 here and RISES, so a floor taken from the start state clips nothing -- which the control below
    // caught when this gate first tried exactly that.
    // SKIPPED ON THE BOUNDARY ARM. Forcing the limiters to bind is calibrated for rhoBox, whose shipped
    // `pMin 1000` never clips a field sitting near 1e5. The same forcing on sbMatched clamps 111600 of
    // 112000 cells to the energy floor and 55600 to the pressure floor, moves U by 1.5e+05 relative in a
    // single iteration, and leaves the two codes amplifying solver tolerance in a state neither case
    // asked for -- Ux read 4.5e-04 there against 4.5e-12 with the case running as it ships. The boundary
    // arm measures the BOUNDARY, so it needs the interior running normally.
    std::vector<scalar> pUnlimited, heUnlimited, tUnlimited;
    if (!boundaryArm && !turbulentArm)
    {
        cpu::rhoSimple::RhoSimpleFields fn =
            cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                         m, g, fvp);
        for (int it = 1; it <= iters; ++it)
            (void)cpu::rhoSimple::rhoSimpleStep(fn, hin, m, g, fvp);
        pUnlimited = fn.p.internal;
        heUnlimited = fn.he.internal;
        tUnlimited  = fn.T.internal;
    }
    if (!boundaryArm && !turbulentArm)
    {
        scalar pLo = 1e300, pHi = -1e300;
        for (label c = 0; c < nC; ++c)
        { pLo = std::fmin(pLo, pUnlimited[c]); pHi = std::fmax(pHi, pUnlimited[c]); }
        const scalar mid = 0.5 * (pLo + pHi);
        hf.pressureControl.limitMinP = true;
        hf.pressureControl.pMin      = mid;          // clips the entire lower half of the field
        df.pressureControl.limitMinP = true;
        df.pressureControl.pMin      = mid;
        gin.limitMinP  = true;
        gin.pMinLimit  = mid;
        hin.pMinProbe  = mid;

        // ...and the same for the energy limiter. The floor is chosen in TEMPERATURE and converted with
        // the reference's OWN hConstTToHe, because the host takes limitTmin/limitTmax in temperature
        // while the device takes energy. Picking it in energy and inverting by hand would be a second
        // implementation of the conversion, and the two clamping at slightly different energies is
        // exactly the kind of difference that would read as a device defect.
        scalar tLo = 1e300, tHi = -1e300;
        for (label c = 0; c < nC; ++c)
        { tLo = std::fmin(tLo, tUnlimited[c]); tHi = std::fmax(tHi, tUnlimited[c]); }
        const scalar tMid = 0.5 * (tLo + tHi);
        hin.limitT     = true;
        hin.limitTmin  = tMid;
        hin.limitTmax  = 1e30;
        hin.heMinProbe = hConstTToHe(tMid, hf.thermo);
        gin.limitHe = true;
        gin.heMin   = hConstTToHe(tMid, hf.thermo);
        gin.heMax   = hConstTToHe(1e30, hf.thermo);
        std::printf("  limitTemperature: Tmin forced to %.6g K (range %.6g .. %.6g) -> he floor %.6g\n",
                    (double)tMid, (double)tLo, (double)tHi, (double)gin.heMin);
        gin.limitMaxP  = hf.pressureControl.limitMaxP;
        gin.pMaxLimit  = hf.pressureControl.pMax;
        std::printf("  pressureControl: pMin forced to %.6g (field range %.6g .. %.6g) so the clamp binds\n",
                    (double)mid, (double)pLo, (double)pHi);
    }
    // The SAME controls the host got. The sentinel is "the case NAMES a factor", not "the factor is
    // below 1": fvMatrix::relax early-returns only on alpha <= 0, so relax(1.0) still applies the
    // dominance clamp and moves the source.
    // The updateCoeffs() metadata the projection gathered. Passing it is not optional dressing: without
    // it the device driver leaves every flux-switched, freestream and flowRate patch on its seeded
    // coefficients, which is what this gate's sbMatched arm exists to catch.
    gin.hasMixed = dev.hasMixed;
    gin.frMagSf  = &dev.frMagSf;
    gin.frMdot   = &dev.frMdot;
    gin.frNx     = &dev.frNx;
    gin.frNy     = &dev.frNy;
    gin.frNz     = &dev.frNz;

    gin.relaxEquationU  = (req != nullptr) && req->found("U");
    gin.relaxU          = hin.relaxU;
    gin.relaxEquationHe = (req != nullptr) && req->found(hf.heName);
    gin.relaxHe         = hin.relaxHe;
    gin.relaxP    = hin.relaxP;
    gin.relaxRho  = hin.relaxRho;
    gin.relaxPEqn = hin.relaxPEqn;
    gin.relaxPEqnSpecified = hin.relaxPEqnSpecified;
    gin.boundedU = gin.boundedHe = gin.boundedKE = true;
    gin.correctedLaplacian = false;
    gin.nNonOrthogonalCorrectors = 0;   // see the note above: the reference solves p once
    if (deviceThermo)
    {
        // THE DEVICE-RESIDENT HOOKS. Same two operations, never leaving the GPU. This is the arm that
        // makes the driver device-resident in the sense that matters: the hooks above are correct and
        // are what the host-against-host comparison needs, but they copy p, he, T, psi and rho across
        // PCIe twice per iteration, so a solver built on them would spend the run on transfers.
        //
        // The comparison is not two thermodynamic models. Both call the SAME BRAE_HD inline functions --
        // hConstHeToT, perfectGasPsi, perfectGasRho -- so what is under test is the wiring: that the
        // device hook writes every field the host hook writes, on the boundary as well as the cells, and
        // that the driver sees no difference. A field the device hook forgot would show up here as the
        // reference moving where the device did not.
        gin.thermoCorrect = [&]() { gpu::rhoSimple::thermoCorrect(gf, dbT, hf.thermo); };
        gin.updateRho     = [&]() { gpu::rhoSimple::updateRho(gf, hf.thermo); };
    }
    else
    {
        gin.thermoCorrect = [&]()
        {
            pullPT();
            cpu::rhoSimple::thermoCorrect(df, fvp);
            pushThermo();
        };
        gin.updateRho = [&]()
        {
            pullPT();
            df.T.internal = gf.T.host();
            cpu::rhoSimple::updateRho(df, fvp);
            gf.rho.copyFrom(df.rho.internal);
            gf.rhoBnd.copyFrom(flatBnd(df.rho, fvp, dm.nBndFaces, 1.0));
        };
    }
    // THE TURBULENCE HOOK. gin.correct was nullptr in EVERY registered arm, so the device kEpsilon
    // closure ran in no gate at all -- its own standalone gate (test_rho_kepsilon_cuda) assembles ONE
    // system from OpenFOAM's adopted fields, which says the arithmetic is right GIVEN that state and
    // nothing about the closure's POSITION in the iteration or the nut/alphat -> muEff/alphaEff feedback.
    //
    // The host driver runs its own host closure inside rhoSimpleStep, and the two closures are already
    // pinned against each other (test_rho_kepsilon_cuda) and against instrumented OpenFOAM
    // (rho_kepsilon_vs_openfoam). So what this arm adds is exactly the part neither can see: whether the
    // DRIVER composes them in the same order and on the same state.
    //
    // The inputs are built HOST-SIDE from the device state and uploaded, which is the same arrangement
    // this file already uses for muEff/alphaEff. Two of them have no device producer yet -- the
    // volumetric flux and the laminar nu -- so a device-resident version of this hook waits on those,
    // exactly as the thermo hooks waited on rhoThermoDevice.cu.
    // The stages struct lives ACROSS iterations for the same reason the pressure workspace does: it owns
    // device buffers, and reallocating them every call would defeat any caching keyed on them.
    if (turbulentArm)
    {
        const std::vector<label> am = dev.alphatWallMask.size() ? dev.alphatWallMask.host() : std::vector<label>();
        label n = 0; for (std::size_t i = 0; i < am.size(); ++i) n += (am[i] != 0);
        std::printf("  turbulent arm: alphat wall faces %d, turbInlet %s, closure inputs from device\n",
                    (int)n, dev.hasTurbulentInlet ? "yes" : "no");
    }
    gpu::kEpsilonRAS::KEpsilonStages kst;
    if (turbulentArm)
    {
        gin.correct = [&]()
        {
            const label nBF = dm.nBndFaces;
            const std::vector<scalar> hT   = gf.T.host();
            const std::vector<scalar> hRho = gf.rho.host();
            const std::vector<scalar> hTB  = gf.TBnd.host();
            const std::vector<scalar> hRhoB= gf.rhoBnd.host();

            // nu = mu(T)/rho, cells and boundary faces. transportMu is the SAME function the reference
            // uses -- one transport model in the tree, called from both sides.
            std::vector<scalar> nuC(nC), nuB(nBF);
            for (label c = 0; c < nC; ++c)  nuC[c] = transportMu(hT[c], hf.thermo) / hRho[c];
            for (label i = 0; i < nBF; ++i) nuB[i] = transportMu(hTB[i], hf.thermo)
                                                   / (hRhoB[i] > 0 ? hRhoB[i] : scalar(1));

            // The wall-face gather: nu in WALL-face order, which is not boundary-face order.
            std::vector<scalar> nuWall(dev.wfFaceOfBnd.size());
            for (std::size_t i = 0; i < dev.wfFaceOfBnd.size(); ++i)
                nuWall[i] = nuB[static_cast<std::size_t>(dev.wfFaceOfBnd[i])];

            // compressibleTurbulenceModel::phi() -- the VOLUMETRIC flux, phi/fvc::interpolate(rho).
            // divU is a dilatation and must come from this, not from the mass flux the div operator uses.
            const std::vector<scalar> hPhiI = gf.phiInt.host(), hPhiB = gf.phiBnd.host();
            //
            // effectiveFaceViscosity, NOT fvc::interpolate, and the BOUNDARY divided by the interpolated
            // FACE value rather than by the patch rho -- because that is what the host driver does
            // (rhoSimpleFoam_cpp.cu:518-527). The two differ on boundary faces, and using the patch rho
            // here put the device closure on a different volumetric flux from the host's, which showed
            // up as the whole field separating at iteration 2 while iteration 1 stayed at 4.5e-12.
            std::vector<scalar> pbrI(hPhiI.size()), pbrB(hPhiB.size());
            {
                std::vector<std::vector<scalar>> rbP(fvp.size());
                label bi = 0;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    rbP[pi].resize(fvp[pi].size);
                    for (label i = 0; i < fvp[pi].size; ++i, ++bi)
                        rbP[pi][i] = (bi < (label)hRhoB.size()) ? hRhoB[bi] : scalar(1);
                }
                const SurfaceScalarField rhof = cpu::effectiveFaceViscosity(hRho, rbP, m, g, fvp);
                for (std::size_t f = 0; f < hPhiI.size() && f < rhof.internal.size(); ++f)
                    pbrI[f] = hPhiI[f] / rhof.internal[f];
                bi = 0;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    for (label i = 0; i < fvp[pi].size; ++i, ++bi)
                        if (bi < (label)pbrB.size())
                            pbrB[bi] = hPhiB[bi] / rhof.boundary[pi][i];
            }

            DeviceBuffer<scalar> dNuC(nuC), dNuB(nuB), dNuW(nuWall), dPbrI(pbrI), dPbrB(pbrB);

            gpu::kEpsilonRAS::KEpsilonInput kin;
            kin.phiInt = &gf.phiInt;         kin.phiBnd = &gf.phiBnd;
            kin.phiByRhoInt = &dPbrI;        kin.phiByRhoBnd = &dPbrB;
            kin.rhoCell = &gf.rho;           kin.rhoBndFace = &gf.rhoBnd;
            kin.nuCell = &dNuC;              kin.nuBndFace = &dNuB;
            kin.nuWallFace = &dNuW;
            // A SCRATCH COPY, not &gf.nutBnd itself. nutBndFace is the ENTERING wall viscosity the
            // diffusivity is built from, and gf.nutBnd is also the OUTPUT correct() overwrites -- alias
            // them and the closure reads a value it has already replaced partway through. The standalone
            // gate keeps the two separate (gNutBnd vs dNutBnd) and this arm has to as well.
            DeviceBuffer<scalar> nutBndIn(gf.nutBnd.host());
            kin.nutBndFace = &nutBndIn;
            kin.wfBndMask = &dev.wfBndMask;  kin.wallYBndFace = &dev.wallYBndFace;
            kin.Ux = &gf.Ux; kin.Uy = &gf.Uy; kin.Uz = &gf.Uz;
            // The masks built by createDeviceFields. Passing null here is NOT "no turbulent inlet" --
            // it is silently no turbulent inlet at all, on a case whose 0/k asks for one.
            if (dev.hasTurbulentInlet)
            {
                kin.turbInletKMask   = &dev.turbInletKMask;
                kin.turbInletKInt    = &dev.turbInletKInt;
                kin.turbInletEpsMask = &dev.turbInletEpsMask;
                kin.turbInletEpsLen  = &dev.turbInletEpsLen;
            }
            kin.alphatWallMask = &dev.alphatWallMask;
            kin.alphatPrtFace  = &dev.alphatPrtFace;
            kin.co = hf.keCoeffs;
            kin.Prt = hf.Prt;
            kin.boundedK = kin.boundedEps = hin.boundedTurb;
            kin.hasNonUpwindDivScheme = !hin.turbDivUnsupported.empty();
            kin.divSchemeUnsupported  = hin.turbDivUnsupported;
            kin.correctedLaplacian = gin.correctedLaplacian;
            kin.relaxEquationK   = hin.relaxEquationK;   kin.relaxK   = hin.relaxK;
            kin.relaxEquationEps = hin.relaxEquationEps; kin.relaxEps = hin.relaxEpsilon;
            kin.tol = hin.tolTurb; kin.relTol = hin.relTolTurb; kin.maxIter = hin.maxIter;

            gpu::kEpsilonRAS::correct(gf.k, gf.epsilon, gf.nut, gf.nutBnd, &gf.alphat, &gf.alphatBnd,
                                      kst, dm, dbU, dev.dbK, dev.dbEps, dev.wall, kin);
        };
    }
    else
    {
        gin.correct = nullptr;   // laminar
    }

    // The energy the run STARTS from, for the device-thermo oracle control at the end of this file: it
    // has to show that he actually moved, or T tracking he proves nothing.
    const std::vector<scalar> heAtStart = gf.he.host();

    // ---- the loop ----------------------------------------------------------------------------
    std::printf("  per-iteration agreement (relL2, CUDA against _cpp)\n");
    double worstU = 0, worstP = 0, worstT = 0, worstRho = 0;
    double worstAlphatB = 0; label alphatBFaces = 0;
    double worstK = 0, worstE = 0, worstN = 0;
    int firstBad = -1;
    for (int it = 1; it <= iters; ++it)
    {
        const cpu::rhoSimple::Residuals hr = cpu::rhoSimple::rhoSimpleStep(hf, hin, m, g, fvp);

        // muEff / alphaEff for THIS iteration, from the DEVICE state. The driver takes them as inputs
        // because they come from the thermo and the closure; the reference recomputes them internally
        // from the same state, so both see the same transport.
        std::vector<scalar> muEff, alphaEff;
        std::vector<std::vector<scalar>> muEffBnd, alphaEffBnd;
        DeviceBuffer<scalar> dMu, dAl, dMuB, dAlB;
        // THE TURBULENT ARM MUST TAKE THE DEVICE TRANSPORT. muEff and alphaEff are the only route by
        // which the closure reaches the momentum and energy equations, and the host-side branch below
        // builds them from `df` -- the host mirror the THERMO hooks maintain. Nothing syncs df.nut or
        // df.alphat from the device, so with the device closure wired the device driver would run on the
        // nut createFields wrote and never see its own closure's output at all. That is what made the
        // arm's numbers bit-identical across three different volumetric-flux constructions: the inputs
        // were changing and nothing downstream was reading the result.
        if (deviceThermo || turbulentArm)
        {
            // The THIRD round-trip, and the last one in the loop. muEff and alphaEff are the only place
            // the closure and the thermo enter the momentum and energy equations, and computing them on
            // the host means pulling U, p, T and rho down and pushing four arrays back, every iteration.
            gpu::rhoSimple::effectiveTransport(gf, hf.thermo, hf.turbulent, dMu, dMuB, dAl, dAlB);
        }
        else
        {
            df.U.internal.resize(nC);
            const std::vector<scalar> ux = gf.Ux.host(), uy = gf.Uy.host(), uz = gf.Uz.host();
            for (label c = 0; c < nC; ++c) df.U.internal[c] = vector{ux[c], uy[c], uz[c]};
            df.p.internal = gf.p.host();
            df.T.internal = gf.T.host();
            df.rho.internal = gf.rho.host();
            cpu::rhoSimple::effectiveTransport(df, fvp, muEff, muEffBnd, alphaEff, alphaEffBnd);
            dMu.copyFrom(muEff);
            dAl.copyFrom(alphaEff);
            dMuB.copyFrom(flat(muEffBnd, fvp, dm.nBndFaces, 0.0));
            dAlB.copyFrom(flat(alphaEffBnd, fvp, dm.nBndFaces, 0.0));
        }
        gin.muEffCell = &dMu;          gin.muEffBndFace = &dMuB;
        gin.alphaEffCell = &dAl;       gin.alphaEffBndFace = &dAlB;

        gpu::rhoSimple::rhoSimpleStep(gf, w, dm, dbU, dbP, dbHe, dbT, gin);

        // DISSECTION: every field, not just the ones the summary line carries. The drift appears at
        // iteration 2 while everything reported at iteration 1 agrees to ~1e-11, and an amplification of
        // 3e5 inside one iteration is not what smooth arithmetic does -- so a field NOT being watched
        // has almost certainly already parted at iteration 1 and is feeding forward. This looks at all
        // of them, boundaries included, and names the first one that has.
        if (turbulentArm)
        {
            auto bndOf = [&](const GeometricField<scalar>& f) { return flatBnd(f, fvp, dm.nBndFaces, 0.0); };
            std::vector<scalar> hUxB(dm.nBndFaces, 0), hUyB(dm.nBndFaces, 0), hUzB(dm.nBndFaces, 0);
            {
                label bi = 0;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    const std::vector<vector>& b = hf.U.boundary[pi]->value();
                    for (label i = 0; i < fvp[pi].size && bi < dm.nBndFaces; ++i, ++bi)
                    { hUxB[bi] = b[i].x; hUyB[bi] = b[i].y; hUzB[bi] = b[i].z; }
                }
            }
            std::vector<scalar> hPhiB(dm.nBndFaces, 0);
            {
                label bi = 0;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                    for (label i = 0; i < fvp[pi].size && bi < dm.nBndFaces; ++i, ++bi)
                        hPhiB[bi] = hf.phi.boundary[pi][i];
            }
            const struct { const char* n; double v; } row[] = {
                { "psi",        relL2(gf.psi.host(),       hf.psi) },
                { "psiBnd",     relL2(gf.psiBnd.host(),    flat(hf.psiBnd, fvp, dm.nBndFaces, 0.0)) },
                { "rhoThermo",  relL2(gf.rhoThermo.host(), hf.rhoThermo) },
                { "nutBnd",     relL2(gf.nutBnd.host(),    bndOf(hf.nut)) },
                { "alphat",     relL2(gf.alphat.host(),    hf.alphat.internal) },
                { "UxBnd",      relL2(gf.UxBnd.host(),     hUxB) },
                { "UyBnd",      relL2(gf.UyBnd.host(),     hUyB) },
                { "UzBnd",      relL2(gf.UzBnd.host(),     hUzB) },
                { "pBnd",       relL2(gf.pBnd.host(),      bndOf(hf.p)) },
                { "TBnd",       relL2(gf.TBnd.host(),      bndOf(hf.T)) },
                { "heBnd",      relL2(gf.heBnd.host(),     bndOf(hf.he)) },
                { "rhoBnd",     relL2(gf.rhoBnd.host(),    bndOf(hf.rho)) },
                { "phiBnd",     relL2(gf.phiBnd.host(),    hPhiB) },
            };
            std::printf("       (unwatched)");
            for (const auto& r : row) if (r.v > 1e-9) std::printf("  %s %.2e", r.n, r.v);
            std::printf("   [only >1e-9 shown]\n");
        }

        // THE CLOSURE'S OWN OUTPUTS, per iteration. The trajectory says the two drivers separate from
        // iteration 2 -- which is the first iteration whose inputs carry the closure's result -- but not
        // WHICH of k, epsilon or nut moved first. A whole-field U number cannot answer that; these can.
        if (turbulentArm)
        {
            const double rk = relL2(gf.k.host(),       hf.k.internal);
            const double re = relL2(gf.epsilon.host(), hf.epsilon.internal);
            const double rn = relL2(gf.nut.host(),     hf.nut.internal);
            worstK = std::max(worstK, rk);
            worstE = std::max(worstE, re);
            worstN = std::max(worstN, rn);
            std::printf("       %-30s k %.3e  epsilon %.3e  nut %.3e\n", "(closure output)", rk, re, rn);
        }

        // The alphat BOUNDARY, device against host, on the turbulent arm. Compared directly rather than
        // through the trajectory: OF writes it inside every correctNut (EddyDiffusivity.C:38) as
        // rho_b*nut_b/Prt_patch, the host driver now does, and the device closure now does -- but on this
        // fixture the difference does not reach U or T at the level this arm resolves, so a trajectory
        // bound would not test it. This does.
        if (turbulentArm)
        {
            const std::vector<scalar> dAB = gf.alphatBnd.host();
            const std::vector<scalar> hAB = flatBnd(hf.alphat, fvp, dm.nBndFaces, 0.0);
            const std::vector<label>  am  = dev.alphatWallMask.host();
            double num = 0.0, den = 0.0; label nf = 0;
            for (std::size_t i = 0; i < am.size() && i < dAB.size() && i < hAB.size(); ++i)
            {
                if (!am[i]) continue;
                const double d = (double)dAB[i] - (double)hAB[i];
                num += d * d; den += (double)hAB[i] * (double)hAB[i]; ++nf;
            }
            worstAlphatB = std::max(worstAlphatB, den > 0 ? std::sqrt(num / den) : std::sqrt(num));
            alphatBFaces = nf;
        }

        // Compare AFTER EVERY iteration, so the FIRST divergent one is named. A final-state-only
        // comparison reports where the two ended up, not where they parted.
        std::vector<scalar> hu(nC), hv(nC), hw(nC);
        for (label c = 0; c < nC; ++c)
        { hu[c] = hf.U.internal[c].x; hv[c] = hf.U.internal[c].y; hw[c] = hf.U.internal[c].z; }
        const double rUy  = relL2(gf.Uy.host(), hv);
        const double rUz  = relL2(gf.Uz.host(), hw);
        const double rPhi = relL2(gf.phiInt.host(), hf.phi.internal);
        const double rHe  = relL2(gf.he.host(), hf.he.internal);
        const double rU   = relL2(gf.Ux.host(), hu);
        const double rP   = relL2(gf.p.host(),  hf.p.internal);
        const double rT   = relL2(gf.T.host(),  hf.T.internal);
        const double rRho = relL2(gf.rho.host(), hf.rho.internal);
        worstU = std::fmax(worstU, rU);       worstP = std::fmax(worstP, rP);
        worstT = std::fmax(worstT, rT);       worstRho = std::fmax(worstRho, rRho);
        if (firstBad < 0 && (rU > 1e-9 || rP > 1e-11)) firstBad = it;
        if (it == 1)
        {
            // Magnitudes, once: a relative measure on a field that is ZERO by construction (the
            // empty-patch direction) divides noise by noise and reports O(1) for nothing at all.
            auto mx = [](const std::vector<scalar>& v)
            { scalar r = 0; for (scalar x : v) r = std::fmax(r, std::fabs(x)); return (double)r; };
            std::printf("       magnitudes: host |Ux|=%.3e |Uy|=%.3e |Uz|=%.3e   dev |Uy|=%.3e |Uz|=%.3e\n",
                        mx(hu), mx(hv), mx(hw), mx(gf.Uy.host()), mx(gf.Uz.host()));
        }
        std::printf("     iter %3d   Ux %.3e  Uy %.3e  Uz %.3e  he %.3e  phi %.3e  p %.3e  T %.3e  rho %.3e\n",
                    it, rU, rUy, rUz, rHe, rPhi, rP, rT, rRho);
    }

    std::printf("  worst over all %d iterations\n", iters);
    // Two different BiCGStab implementations sit between the two paths on U, he and p, so the fields
    // cannot reach 1e-16 -- what is asserted is that the gap does not GROW into the trajectory, which is
    // the failure mode this file exists for.
    // Bounds set to what the composition actually achieves, with ~30-60x headroom for the fact that
    // two different BiCGStab implementations sit between the two paths. Measured over 8 iterations:
    // Ux 1.474e-11, p 7.317e-14, T 3.099e-13, rho 3.116e-13.
    report("Ux", worstU, 1e-9);
    report("p", worstP, 1e-11);
    report("T", worstT, 1e-11);
    report("rho", worstRho, 1e-11);
    check("no iteration diverged", firstBad < 0);

    if (turbulentArm)
    {
        // The device closure now writes alphat's boundary. Before it did, this read 1.0 exactly on every
        // one of these faces -- brae's device wall alphat was identically zero where the host's (and
        // OpenFOAM's) is not, so alphaEff at the wall carried none of the turbulent diffusivity.
        std::printf("     %-40s over %d alphat wall faces\n", "alphat BOUNDARY: device vs host",
                    (int)alphatBFaces);
        check("the fixture HAS alphat wall faces (else this is vacuous)", alphatBFaces > 0);
        // 1e-3, not machine precision: this inherits the closure drift the arm has not yet explained
        // (see the header). It TIGHTENS when that is understood. Measured 1.143903e-04; with the device
        // boundary write disabled it reads exactly 1.000000e+00, which is what it caught.
        report("alphat boundary (device vs host)", worstAlphatB, 1e-3);
    }

    // ---- CONTROL: the POROSITY reaches the momentum equation -----------------------------------
    // rhoUEqn.cu has been able to apply an explicitPorositySource since it was written, but the DRIVER
    // had no field to carry one and never set uin.porosity -- so a porous case ran with the porosity
    // silently absent, which drives the duct at the wrong speed and still converges. validation/angledDuct
    // and OpenFOAM's own angledDuctExplicitFixedCoeff are exactly that case.
    //
    // This asserts the WIRING, not the physics: a synthetic Darcy resistance over half the cells must
    // change the answer. The physics of the porosity model is rhoUEqn's own to gate; what could not be
    // seen from inside that module is whether the driver ever hands it one.
    {
        DevicePorosity por;
        por.active = true;
        std::vector<label> pc;
        for (label c = 0; c < nC / 2; ++c) pc.push_back(c);
        por.cells.copyFrom(pc);
        // fixedCoeff, NOT DarcyForchheimer -- and the module's own refusal is what said so: on a
        // force-dimensioned momentum equation DarcyForchheimer needs the per-cell laminar mu and rho to
        // build Cd, which this assembly is not given, so it throws rather than solving the kinematic
        // form with nu = 0. fixedCoeff takes rhoRef from the dictionary and is the branch the
        // compressible tutorials use (angledDuctExplicitFixedCoeff is one).
        por.fixed  = true;
        por.rhoRef = 1.0;
        for (int i = 0; i < 9; ++i) { por.fa[i] = 0.0; por.fb[i] = 0.0; }
        por.fa[0] = por.fa[4] = por.fa[8] = 5.0e2;   // diag(alpha), large enough to bite in one iteration

        gpu::rhoSimple::RhoDeviceFields dv =
            gpu::rhoSimple::createDeviceFields(df, m, g, fvp);
        gpu::rhoSimple::RhoSolverWorkspace wp;
        gpu::rhoSimple::RhoStepInput pin2 = gin;
        pin2.porosity = &por;
        pin2.takeUAtBoundary = &dv.takeUAtBoundary;
        pin2.adjustable      = &dv.adjustable;
        // The transport for iteration 1, from the same state the main run started from.
        std::vector<scalar> mu2, al2;
        std::vector<std::vector<scalar>> muB2, alB2;
        cpu::rhoSimple::effectiveTransport(df, fvp, mu2, muB2, al2, alB2);
        DeviceBuffer<scalar> dMu2(mu2), dAl2(al2);
        DeviceBuffer<scalar> dMuB2(flat(muB2, fvp, dm.nBndFaces, 0.0));
        DeviceBuffer<scalar> dAlB2(flat(alB2, fvp, dm.nBndFaces, 0.0));
        pin2.muEffCell = &dMu2;       pin2.muEffBndFace = &dMuB2;
        pin2.alphaEffCell = &dAl2;    pin2.alphaEffBndFace = &dAlB2;
        pin2.thermoCorrect = [](){};  // the projection is fresh; one iteration needs no thermo round-trip
        pin2.updateRho     = [](){};
        gpu::rhoSimple::rhoSimpleStep(dv.f, wp, dv.dm, dv.dbU, dv.dbP, dv.dbHe, dv.dbT, pin2);
        const std::vector<scalar> withPor = dv.f.Ux.host();

        gpu::rhoSimple::RhoDeviceFields dv0 =
            gpu::rhoSimple::createDeviceFields(df, m, g, fvp);
        gpu::rhoSimple::RhoSolverWorkspace w0;
        gpu::rhoSimple::RhoStepInput pin0 = pin2;
        pin0.porosity = nullptr;
        pin0.takeUAtBoundary = &dv0.takeUAtBoundary;
        pin0.adjustable      = &dv0.adjustable;
        gpu::rhoSimple::rhoSimpleStep(dv0.f, w0, dv0.dm, dv0.dbU, dv0.dbP, dv0.dbHe, dv0.dbT, pin0);
        const std::vector<scalar> noPor = dv0.f.Ux.host();

        const double r = relL2(withPor, noPor);
        std::printf("     %-58s rel=%.3e\n", "control: a porosity CHANGES the momentum answer", r);
        check("the driver passes the porosity to the momentum equation", r > 1e-6);
    }

    // ---- CONTROL: the ENERGY limiter must actually BIND ----------------------------------------
    // Same trap as the pressure limiter, and the same remedy. limitTemperature is a CORRECTION applied
    // to he after the solve; the CUDA driver dropped it entirely and refused nothing, so a case naming
    // it got the clamp on the host reference and not on the device. A bound taken from the start state
    // would clip nothing, so the floor comes from the unlimited run's own he range and the control
    // requires it to clip a substantial share. Both limiter controls belong to the rhoBox arm: the
    // boundary arm does not force the limiters at all (see above), so there is nothing there to assert.
    if (!boundaryArm && !turbulentArm)
    {
        std::size_t atFloor = 0;
        for (label c = 0; c < nC; ++c)
            if (hf.he.internal[c] <= hin.heMinProbe * (1.0 + 1e-12)) ++atFloor;
        std::printf("     %-58s %zu of %d cells at the floor\n",
                    "control: the energy limiter BINDS on this run", atFloor, (int)nC);
        check("the energy limiter clips a substantial share", atFloor > (std::size_t)(nC / 10));
    }

    // ---- CONTROL: the pressure limiter must actually BIND --------------------------------------
    // Forcing pMin only matters if it clips. If it does not, every agreement above is agreement about a
    // no-op -- which is exactly how a driver missing the limiter entirely passed this gate until an audit
    // found it. The unlimited run computed above is the comparison.
    if (!boundaryArm && !turbulentArm)
    {
        const double moved = relL2(pUnlimited, hf.p.internal);
        std::size_t atFloor = 0;
        for (label c = 0; c < nC; ++c)
            if (hf.p.internal[c] <= hin.pMinProbe * (1.0 + 1e-12)) ++atFloor;
        std::printf("     %-58s rel=%.3e   %zu of %d cells at the floor\n",
                    "control: the pressure limiter BINDS on this run", moved, atFloor, (int)nC);
        check("the limiter clips something (else the comparison is a no-op)", moved > 1e-12);
        check("...and clips a substantial share of the field", atFloor > (std::size_t)(nC / 10));
    }

    // ---- CONTROL: the trajectory must MOVE ---------------------------------------------------
    // If the case sat still, every agreement above would be the agreement of two codes that did nothing.
    {
        cpu::rhoSimple::RhoSimpleFields f0 =
            cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution, m, g, fvp);
        std::vector<scalar> u0(nC), u1(nC);
        for (label c = 0; c < nC; ++c) { u0[c] = f0.U.internal[c].x; u1[c] = hf.U.internal[c].x; }
        const double movedU = relL2(u1, u0);
        const double movedP = relL2(hf.p.internal, f0.p.internal);
        std::printf("     %-40s Ux %.3e   p %.3e\n", "control: the run actually moved", movedU, movedP);
        check("the solver moved U (control)", movedU > 1e-6);
        check("the solver moved p (control)", movedP > 1e-8);
    }

    // ---- CONTROL: iteration 2 onwards is REACHED, and is what discriminates -------------------
    // The defect this file exists for is invisible at iteration 1. Assert the loop actually ran past it,
    // so a fixture or an early return cannot make this gate vacuous.
    if (!boundaryArm) check("more than one iteration ran (control)", iters > 1);

    // ---- CONTROL: the AMG workspace persisted --------------------------------------------------
    // The hierarchy is built once and reused; if it were rebuilt each iteration the cached V-cycle would
    // attach to a different fine matrix every time -- exact at iteration 1, wrong at iteration 2.
    if (!gin.transonic) check("the AMG hierarchy was built and kept (control)", w.amgBuilt);
    else                check("transonic: no hierarchy built (control)", !w.amgBuilt);

    // ---- CONTROL: the DEVICE thermo computed the right numbers, against an independent oracle ----
    // Not a comparison of the two hook implementations -- they agree, and that is what the bounds above
    // already assert. This recomputes the thermo relations on the HOST from the device's OWN he and p
    // after the loop has run, so what is checked is that the state the device carries actually satisfies
    // them. A hook that silently did nothing would leave T at the value createFields wrote while he moved
    // eight iterations away from it, and that is precisely what this catches: the residual is taken
    // against a he the solve has changed, not against the seed.
    if (deviceThermo)
    {
        const std::vector<scalar> he = gf.he.host(), T = gf.T.host(), psi = gf.psi.host();
        double worstT = 0.0, worstPsi = 0.0, moved = 0.0;
        for (label c = 0; c < nC; ++c)
        {
            const double tWant = hConstHeToT(he[c], hf.thermo);
            const double pWant = perfectGasPsi((scalar)tWant, hf.thermo);
            worstT   = std::max(worstT,   std::fabs((double)T[c]   - tWant) / std::fabs(tWant));
            worstPsi = std::max(worstPsi, std::fabs((double)psi[c] - pWant) / std::fabs(pWant));
            moved    = std::max(moved, std::fabs((double)he[c] - (double)heAtStart[c])
                                       / std::max(1e-30, std::fabs((double)heAtStart[c])));
        }
        std::printf("     %-40s T %.3e  psi %.3e  (he moved %.3e)\n",
                    "control: device thermo vs host oracle", worstT, worstPsi, moved);
        // The energy must have MOVED, or T tracking he would be the agreement of two unchanged fields.
        check("the energy field moved over the run (control)", moved > 1e-10);
        check("device T is hConstHeToT(he) to round-off (control)", worstT < 1e-14);
        check("device psi is perfectGasPsi(T) to round-off (control)", worstPsi < 1e-14);
    }

    // ---- CONTROL: updateCoeffs() is WIRED, and does what OpenFOAM's does ------------------------
    // The driver grew an updateBoundaryCoeffs() call because it had none: every patch whose coefficients
    // are a function of the solution -- the inletOutlet/outletInlet flux switch, the freestream flow-angle
    // blend, flowRateInletVelocity's velocity from the live boundary density -- kept the coefficients it
    // was seeded with for the whole run. rhoUEqn.cuh:64-83 states that contract and calls it "not
    // advisory"; nothing satisfied it, and the rhoBox arm above cannot see it, because rhoBox carries
    // only fixedValue, zeroGradient, noSlip and empty.
    //
    // Asserted against the ARITHMETIC rather than against a trajectory, so the control is exact and needs
    // no bound of its own:
    //   flowRateInletVelocity  avgU = -mdot/gSum(rho*magSf)  ->  DOUBLE the boundary density and the
    //                          inlet velocity must HALVE, to round-off.
    //   inletOutlet            bcType is the sign of phi     ->  REVERSE the flux and every io face must
    //                          switch between fixedValue (1) and zeroGradient (0).
    // A driver that never calls updateCoeffs leaves both unchanged, which is what makes them a control.
    if (dev.hasFlowRate || dev.hasMixed)
    {
        gpu::rhoSimple::RhoDeviceFields dc =
            gpu::rhoSimple::createDeviceFields(df, m, g, fvp);
        gpu::rhoSimple::RhoStepInput cin = gin;
        cin.hasMixed = dc.hasMixed;
        cin.frMagSf  = &dc.frMagSf;
        cin.frMdot   = &dc.frMdot;
        cin.frNx     = &dc.frNx;
        cin.frNy     = &dc.frNy;
        cin.frNz     = &dc.frNz;

        gpu::rhoSimple::updateBoundaryCoeffs(dc.f, dc.dbU, dc.dbP, dc.dbHe, dc.dbT, cin);
        const std::vector<scalar> ref1 = dc.dbU.comp[0].refValue.host();
        const std::vector<label>  io1  = dc.dbHe.bcType.host();

        if (dev.hasFlowRate)
        {
            // rho_b *= 2 everywhere. gSum(rho*magSf) doubles on the flowRate patch, so avgU halves.
            std::vector<scalar> rb = dc.f.rhoBnd.host();
            for (std::size_t i = 0; i < rb.size(); ++i) rb[i] *= 2.0;
            dc.f.rhoBnd.copyFrom(rb);
            gpu::rhoSimple::updateBoundaryCoeffs(dc.f, dc.dbU, dc.dbP, dc.dbHe, dc.dbT, cin);
            const std::vector<scalar> ref2 = dc.dbU.comp[0].refValue.host();

            // Only the flowRate faces are asserted on: every other face's refValue is untouched by this
            // update, and demanding a halving there would be asserting the wrong thing.
            const std::vector<scalar> mask = dc.frMagSf.empty()
                                           ? std::vector<scalar>()
                                           : dc.frMagSf[0].host();
            double worst = 0.0, biggest = 0.0;
            label  nFaces = 0;
            for (std::size_t i = 0; i < mask.size() && i < ref1.size() && i < ref2.size(); ++i)
            {
                if (mask[i] <= 0.0) continue;
                ++nFaces;
                const double want = 0.5 * (double)ref1[i];
                const double den  = std::fabs(want) > 1e-30 ? std::fabs(want) : 1.0;
                worst = std::max(worst, std::fabs((double)ref2[i] - want) / den);
                biggest = std::max(biggest, std::fabs((double)ref1[i]));
            }
            // NON-VACUOUS: an inlet velocity that is zero halves to zero, and the check above would pass
            // on a driver that never called updateCoeffs at all. The magnitude is asserted separately so
            // the halving is a statement about a number that exists.
            std::printf("     %-40s worst %.3e over %d faces, |U_in| %.4g\n",
                        "control: 2x rho_b halves the inlet U", worst, (int)nFaces, biggest);
            check("the flowRate inlet carries a velocity at all (control)", biggest > 1e-30);
            check("flowRateInletVelocity reads the LIVE boundary density (control)",
                  nFaces > 0 && worst < 1e-12);

            std::vector<scalar> rb0 = dc.f.rhoBnd.host();
            for (std::size_t i = 0; i < rb0.size(); ++i) rb0[i] *= 0.5;
            dc.f.rhoBnd.copyFrom(rb0);
        }

        // The flux switch. phi is DRIVEN to a known sign here rather than taken from the fixture: this
        // case seeds `internalField uniform (0 0 0)`, so the flux through every boundary face is exactly
        // zero at the start, `phi < 0` is false whichever sign is written, and a control built on
        // negating the fixture's own phi would have passed on a switch that never moved. Driving it makes
        // the assertion the arithmetic one -- outflow is zeroGradient, inflow is fixedValue -- and
        // independent of where in the trajectory the control happens to run.
        {
            const std::vector<scalar> phiSaved = dc.f.phiBnd.host();

            dc.f.phiBnd.copyFrom(std::vector<scalar>(phiSaved.size(), scalar(+1)));
            gpu::rhoSimple::updateBoundaryCoeffs(dc.f, dc.dbU, dc.dbP, dc.dbHe, dc.dbT, cin);
            const std::vector<label> outflow = dc.dbHe.bcType.host();

            dc.f.phiBnd.copyFrom(std::vector<scalar>(phiSaved.size(), scalar(-1)));
            gpu::rhoSimple::updateBoundaryCoeffs(dc.f, dc.dbU, dc.dbP, dc.dbHe, dc.dbT, cin);
            const std::vector<label> inflow = dc.dbHe.bcType.host();

            dc.f.phiBnd.copyFrom(phiSaved);

            const std::vector<label> mask = dc.dbHe.ioMask.host();
            label nIo = 0, nRight = 0;
            for (std::size_t i = 0; i < mask.size() && i < outflow.size() && i < inflow.size(); ++i)
            {
                if (!mask[i]) continue;
                ++nIo;
                // OF inletOutlet: inflow (phi < 0) takes inletValue as a fixedValue, outflow extrapolates.
                if (outflow[i] == 0 && inflow[i] == 1) ++nRight;
            }
            std::printf("     %-40s %d of %d io faces switch both ways\n",
                        "control: the flux sign drives the switch", (int)nRight, (int)nIo);
            check("inletOutlet switches on the CURRENT flux (control)", nIo > 0 && nRight == nIo);
        }
    }
    else
    {
        std::printf("     %-40s no such patch on this fixture\n", "updateCoeffs control: SKIPPED");
    }

    std::printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
