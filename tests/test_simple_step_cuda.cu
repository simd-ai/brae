// END-TO-END: the CUDA SIMPLE driver against the _cpp driver, one iteration, same case.
//
// The _cpp driver is validated against OpenFOAM's own dumpSimpleStep output to 2.5e-11 on p
// (test_simple_step_cpp), so this closes OpenFOAM -> _cpp -> CUDA for the whole solver rather than for
// individual stages.
//
// The tolerance here is NOT machine precision and should not be: the two paths run DIFFERENT linear
// solvers (host GAMG + host BiCGStab vs device AMG-PCG + device BiCGStab). Both converge to the requested
// tolerance, so they agree to about that tolerance and no further -- a stricter gate would be asserting
// that two different Krylov methods take the same path, which is not true and not required. The stage
// tests (test_ueqn_cuda, test_peqn_cuda) are what pin the arithmetic at 1e-16; this pins the composition.
//
// MULTI-ITERATION MODE (4th arg): run N iterations of both drivers and require they still agree.
//
// This exists because a single iteration cannot see the bug it was written for. The CUDA driver allocated
// its pressure matrix and folded diagonal fresh every iteration while the AMG hierarchy -- and the V-cycle
// and PCG graph caches keyed on that matrix -- persisted across them. Iteration 1 was exact (8e-14) and
// iteration 2 was wrong by 1.3e-01, from identical inputs. Every per-stage test passed throughout.
//
// Run: test_simple_step_cuda <caseDir> <timeDir> [laminar] [iters]
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "simpleFoam_cpp.cuh"
#include "linearViscousStress_cpp.cuh"
#include "simpleFoam.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

// `scale` is the magnitude the error should be measured AGAINST, and it is an explicit argument because
// self-normalising is wrong for a component that is identically zero.
//
// This case is 2D (empty front/back patches), so Uz is zero to round-off: max|Uz| came out 1.2e-14 on one
// path and 2.3e-14 on the other. Dividing by max|Uz| turned a 1.4e-14 absolute difference into a reported
// relative error of 1.197 and failed the test on a field that is not there. The meaningful denominator for
// a velocity COMPONENT is the magnitude of the velocity, not of that component -- an error in Uz matters
// relative to |U|. Passing 0 keeps the old self-normalising behaviour for fields that have a real scale.
static void cmp(const std::vector<scalar>& a, const std::vector<scalar>& b, const char* nm, scalar tol,
                scalar scale = 0.0)
{
    scalar mx = 0, mg = scale;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(a[i] - b[i]));
        if (scale <= 0.0) mg = std::fmax(mg, std::fabs(b[i]));
    }
    const scalar rel = mg > 0 ? mx / mg : mx;
    const bool ok = rel <= tol;
    if (!ok) ++g_fails;
    std::printf("  %-24s n=%6zu rel=%.3e  (tol %.0e)  %s\n", nm, b.size(), rel, tol, ok ? "OK" : "FAIL");
}

static void check(bool ok, const char* what)
{
    std::printf("  %-56s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

static std::vector<scalar> flatten(const std::vector<std::vector<scalar>>& b, label n, scalar fill)
{
    std::vector<scalar> f;
    for (const auto& v : b) for (scalar x : v) f.push_back(x);
    f.resize(n, fill);
    return f;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <timeDir> [laminar]\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];
    const bool forceLaminar = (argc > 3 && std::string(argv[3]) == "laminar");
    const int iters = (argc > 4) ? std::atoi(argv[4]) : 1;
    const scalar nu = 1e-5;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, fvp);

    const bool turbulent = !forceLaminar
                        && (std::filesystem::exists(caseDir + "/" + t + "/nut")
                            || std::filesystem::exists(caseDir + "/" + t + "/nut.gz"));
    std::vector<scalar> nuEffC(nC, nu);
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nuEffB[pi].assign(fvp[pi].size, nu);
    if (turbulent)
    {
        GeometricField<scalar> nutF =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), fvp, nC);
        nutF.evaluateBoundary();
        for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nutF.internal[c];
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& nb = nutF.boundary[pi]->value();
            for (label i = 0; i < fvp[pi].size; ++i) nuEffB[pi][i] = nu + nb[i];
        }
    }

    std::printf("test_simple_step_cuda:  (%s)\n", turbulent ? "TURBULENT" : "laminar");

    // The fixture may declare SIMPLEC; both drivers refuse it, and step.dat was dumped with plain SIMPLE
    // (see test_simple_step_cpp). Run the comparison the same way.
    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    cd.consistent = false;
    cpu::SimpleControl ctl(cd);

    const scalar relaxU = 0.7, relaxP = 0.3;

    // ---- the _cpp driver ---------------------------------------------------------------------
    cpu::StepInput cin;
    cin.nu = nu;
    cin.nuEff = nuEffC; cin.nuEffBnd = nuEffB;
    cin.relaxU = relaxU; cin.relaxP = relaxP;
    // Tight solves in multi-iteration mode: at the case's usual relTol 0.1 the two Krylov methods stop at
    // genuinely different iterates and the drift is theirs, not the driver's. The point here is the
    // DRIVER, so remove the solver as a variable.
    if (iters > 1) { cin.tolU = cin.tolP = 1e-12; cin.relTolU = cin.relTolP = 0.0; }
    cpu::Residuals cres;
    for (int it = 0; it < iters; ++it) cres = cpu::simpleStep(f, ctl, cin, m, g, fvp);

    // ---- the CUDA driver, from the SAME starting fields --------------------------------------
    cpu::SimpleFields f0 = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, fvp);
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(f0.U, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(f0.p, fvp, g);

    gpu::SolverFields gf;
    {
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        { ux[c] = f0.U.internal[c].x; uy[c] = f0.U.internal[c].y; uz[c] = f0.U.internal[c].z; }
        gf.Ux.copyFrom(ux); gf.Uy.copyFrom(uy); gf.Uz.copyFrom(uz);
        gf.p.copyFrom(f0.p.internal);
        gf.phiInt.copyFrom(f0.phi.internal);
        gf.phiBnd.copyFrom(flatten(f0.phi.boundary, dm.nBndFaces, 0.0));
    }

    const SurfaceScalarField nuFace = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
    DeviceBuffer<scalar> dNuCell(nuEffC), dNuFace(nuFace.internal);
    DeviceBuffer<scalar> dNuBnd(flatten(nuEffB, dm.nBndFaces, nu));

    std::vector<label> takeU, adjustable;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            takeU.push_back(f0.U.boundary[pi]->assignable() ? 0 : 1);
            adjustable.push_back(f0.U.boundary[pi]->fixesValue() ? 0 : 1);
        }
    takeU.resize(dm.nBndFaces, 0);
    adjustable.resize(dm.nBndFaces, 0);
    DeviceBuffer<label> dTakeU(takeU), dAdjust(adjustable);

    gpu::StepInput gin;
    gin.nuEffCell = &dNuCell; gin.nuEffFace = &dNuFace; gin.nuEffBndFace = &dNuBnd;
    gin.relaxU = relaxU; gin.relaxP = relaxP;
    gin.momentumPredictor = cd.momentumPredictor;
    gin.nNonOrthogonalCorrectors = cd.nNonOrthogonalCorrectors;
    gin.pRefCell = f0.pRefCell; gin.pRefValue = f0.pRefValue;
    gin.takeUAtBoundary = &dTakeU; gin.adjustable = &dAdjust;

    if (iters > 1) { gin.tolU = gin.tolP = 1e-12; gin.relTolU = gin.relTolP = 0.0; }
    gpu::SolverWorkspace ws;
    gpu::Residuals gres;
    for (int it = 0; it < iters; ++it) gres = gpu::simpleStep(gf, ws, dm, dbU, dbP, gin);
    if (iters > 1) std::printf("  (%d iterations, tight solves)\n", iters);

    std::printf("  initial residuals   _cpp: U=%.3e p=%.3e   cuda: U=%.3e p=%.3e\n",
                cres.count("U") ? cres.at("U") : -1.0, cres.count("p") ? cres.at("p") : -1.0,
                gres.count("U") ? gres.at("U") : -1.0, gres.count("p") ? gres.at("p") : -1.0);

    // ---- compare -----------------------------------------------------------------------------
    // The initial residual is computed from the SAME matrix and the SAME starting field on both paths, so
    // it must agree far more tightly than the solved fields do -- it is measured before either Krylov
    // method takes a step. If this one drifts, the assembly disagrees, not the solver.
    if (cres.count("p") && gres.count("p"))
    {
        const scalar rc = cres.at("p"), rg = gres.at("p");
        // ABSOLUTE, not relative. The initial residual is sum|A*psi - b|/normFactor, and on a converged
        // case that numerator is a near-total cancellation of large terms -- so a 1e-16 difference in the
        // matrix shows up as a large RELATIVE difference in a residual that is itself ~1e-6. Measured:
        // laminar 2.9e-11 absolute on a residual of 0.79 (rel 3.6e-11); turbulent 2.6e-12 absolute on a
        // residual of 1.4e-6 (rel 1.8e-06). The absolute agreement is the same order in both; the relative
        // figure only tracks how converged the case happens to be.
        //
        // This is a consistency check, not the assembly gate. The assembly gate is test_peqn_cuda, which
        // compares diag/upper/lower/source/patch-coefficients directly at 1e-13.
        const scalar absDiff = std::fabs(rc - rg);
        std::printf("  %-56s |d|=%.3e\n", "initial p residual (same matrix, before solving)", absDiff);
        check(absDiff < 1e-9, "the two paths assemble a consistent pressure system");
    }

    std::printf("  -- solved fields (two different Krylov methods, both to tol 1e-10)\n");
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    { ux[c] = f.U.internal[c].x; uy[c] = f.U.internal[c].y; uz[c] = f.U.internal[c].z; }
    // Velocity components are measured against |U|, not against themselves -- see the note on cmp.
    scalar Umag = 0;
    for (label c = 0; c < nC; ++c)
        Umag = std::fmax(Umag, std::fmax(std::fabs(ux[c]),
                                std::fmax(std::fabs(uy[c]), std::fabs(uz[c]))));
    cmp(gf.Ux.host(), ux, "U x", 1e-7, Umag);
    cmp(gf.Uy.host(), uy, "U y", 1e-7, Umag);
    cmp(gf.Uz.host(), uz, "U z", 1e-7, Umag);   // ~0 on this 2D case (empty front/back patches)
    cmp(gf.p.host(),  f.p.internal, "p", 1e-7);
    cmp(gf.phiInt.host(), f.phi.internal, "phi internal", 1e-7);
    cmp(gf.phiBnd.host(), flatten(f.phi.boundary, dm.nBndFaces, 0.0), "phi boundary", 1e-7);

    // CONTROL: the iteration must have MOVED the fields on the CUDA path. Starting from a converged or
    // near-converged state the changes are small, so a driver that did nothing would pass every line above.
    {
        const std::vector<scalar> pNow = gf.p.host();
        scalar moved = 0;
        for (label c = 0; c < nC; ++c) moved = std::fmax(moved, std::fabs(pNow[c] - f0.p.internal[c]));
        std::printf("  %-56s max|dp|=%.3e\n", "control: the CUDA iteration changed p", moved);
        check(moved > 0.0, "the CUDA SIMPLE step actually did something (control)");
    }

    // CONTROL: the turbulence hook is called exactly once, at the end.
    {
        gpu::SolverFields gf2 = {};
        {
            std::vector<scalar> a(nC), b(nC), c2(nC);
            for (label c = 0; c < nC; ++c)
            { a[c] = f0.U.internal[c].x; b[c] = f0.U.internal[c].y; c2[c] = f0.U.internal[c].z; }
            gf2.Ux.copyFrom(a); gf2.Uy.copyFrom(b); gf2.Uz.copyFrom(c2);
            gf2.p.copyFrom(f0.p.internal);
            gf2.phiInt.copyFrom(f0.phi.internal);
            gf2.phiBnd.copyFrom(flatten(f0.phi.boundary, dm.nBndFaces, 0.0));
        }
        int called = 0;
        gpu::StepInput gin2 = gin;
        gin2.correct = [&called]() { ++called; };
        gpu::SolverWorkspace ws2;
        gpu::simpleStep(gf2, ws2, dm, dbU, dbP, gin2);
        check(called == 1, "turbulence->correct() hook is called exactly once");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
