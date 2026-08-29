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
    const int iters = (argc > 3) ? std::atoi(argv[3]) : 8;

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

    std::printf("rhoSimpleFoam DRIVER: CUDA vs _cpp  (%d cells, %d iterations, %s)\n",
                (int)nC, iters, hf.turbulent ? hf.rasModel.c_str() : "laminar");
    check("the fixture is LAMINAR (see the header)", !hf.turbulent);
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
    // The SAME controls the host got. The sentinel is "the case NAMES a factor", not "the factor is
    // below 1": fvMatrix::relax early-returns only on alpha <= 0, so relax(1.0) still applies the
    // dominance clamp and moves the source.
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
    gin.correct = nullptr;   // laminar

    // ---- the loop ----------------------------------------------------------------------------
    std::printf("  per-iteration agreement (relL2, CUDA against _cpp)\n");
    double worstU = 0, worstP = 0, worstT = 0, worstRho = 0;
    int firstBad = -1;
    for (int it = 1; it <= iters; ++it)
    {
        const cpu::rhoSimple::Residuals hr = cpu::rhoSimple::rhoSimpleStep(hf, hin, m, g, fvp);

        // muEff / alphaEff for THIS iteration, from the DEVICE state. The driver takes them as inputs
        // because they come from the thermo and the closure; the reference recomputes them internally
        // from the same state, so both see the same transport.
        std::vector<scalar> muEff, alphaEff;
        std::vector<std::vector<scalar>> muEffBnd, alphaEffBnd;
        {
            df.U.internal.resize(nC);
            const std::vector<scalar> ux = gf.Ux.host(), uy = gf.Uy.host(), uz = gf.Uz.host();
            for (label c = 0; c < nC; ++c) df.U.internal[c] = vector{ux[c], uy[c], uz[c]};
            df.p.internal = gf.p.host();
            df.T.internal = gf.T.host();
            df.rho.internal = gf.rho.host();
            cpu::rhoSimple::effectiveTransport(df, fvp, muEff, muEffBnd, alphaEff, alphaEffBnd);
        }
        DeviceBuffer<scalar> dMu(muEff), dAl(alphaEff);
        DeviceBuffer<scalar> dMuB(flat(muEffBnd, fvp, dm.nBndFaces, 0.0));
        DeviceBuffer<scalar> dAlB(flat(alphaEffBnd, fvp, dm.nBndFaces, 0.0));
        gin.muEffCell = &dMu;          gin.muEffBndFace = &dMuB;
        gin.alphaEffCell = &dAl;       gin.alphaEffBndFace = &dAlB;

        gpu::rhoSimple::rhoSimpleStep(gf, w, dm, dbU, dbP, dbHe, gin);

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
    check("more than one iteration ran (control)", iters > 1);

    // ---- CONTROL: the AMG workspace persisted --------------------------------------------------
    // The hierarchy is built once and reused; if it were rebuilt each iteration the cached V-cycle would
    // attach to a different fine matrix every time -- exact at iteration 1, wrong at iteration 2.
    if (!gin.transonic) check("the AMG hierarchy was built and kept (control)", w.amgBuilt);
    else                check("transonic: no hierarchy built (control)", !w.amgBuilt);

    std::printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
