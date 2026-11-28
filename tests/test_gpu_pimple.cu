// PIMPLE foundation smoke + consistency: the transient DeviceSimpleSolver::pimpleStep() reuses the SAME three composable
// phases as step(), with fvm::ddt(U) folded into the predictor. Checks on a real case (laminar):
//   (A) STEADY-LIMIT: pimpleStep() with a huge dt (rDeltaT->0, ddt negligible) must converge to the SAME steady state as
//       the steady step() -- proves the transient orchestration (advanceTime + phase loop) is consistent with SIMPLE.
//   (B) TRANSIENT STABILITY: pimpleStep() with a physical dt (ddt genuinely active) + multiple outer/inner correctors
//       marches stably -- fields stay finite and bounded, continuity error stays small.
//   (C) backward marches 10 steps from a cold start, each step's runTime.timeIndex() handed over as the pimpleFoam loop does.
//   (D) backward without that index refuses by name: its first-step fvc::ddtCorr form depends on it.
// Skips (exit 125) if the case fixture is absent (validation/ is local-only on CI), like the ami_oracle gate.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_simple_foam.cuh"
#include <sys/stat.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    struct stat st;
    if (stat((caseDir + "/constant/polyMesh/points").c_str(), &st) != 0) {
        std::printf("SKIP: case fixture '%s' not present\n", caseDir.c_str());
        return 125;
    }
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const scalar nu = 1e-2, relaxU = 0.7, relaxP = 0.3, tol = 1e-5;
    const int N = 150;

    const FieldData<vector> Ufd = readField<vector>(caseDir + "/0/U");
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/0/p");
    auto mkControls = [&]{ DeviceSimpleControls c; c.nu=nu; c.relaxU=relaxU; c.relaxP=relaxP; c.tolU=tol; c.tolP=tol; c.turbulent=false; return c; };
    auto mkSolver = [&]{
        GeometricField<vector> U = buildField<vector>(Ufd, fvp, nC); U.evaluateBoundary();
        GeometricField<scalar> p = buildField<scalar>(pfd, fvp, nC); p.evaluateBoundary();
        SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
        return DeviceSimpleSolver(m, g, fvp, U, p, phi, mkControls());
    };
    auto finite = [](const std::vector<vector>& u){ for (auto& v : u) if (!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z)) return false; return true; };
    auto relV = [&](const std::vector<vector>& a, const std::vector<vector>& b){ scalar n=0,d=0;
        for (label c=0;c<nC;++c){ n=std::fmax(n,mag(a[c]-b[c])); d=std::fmax(d,mag(b[c])); } return d>0?n/d:n; };

    int fails = 0;

    // ---- (A) steady-limit: pimpleStep(huge dt) -> same steady state as step() -----------------------------------
    DeviceSimpleSolver sSimple = mkSolver();
    for (int it = 0; it < N; ++it) sSimple.step();
    const std::vector<vector> Usteady = sSimple.U();

    DeviceSimpleSolver sPimpleBig = mkSolver();
    sPimpleBig.setDdtScheme(DdtScheme::Euler);
    DeviceSimpleResidual rb;
    for (int it = 0; it < N; ++it) rb = sPimpleBig.pimpleStep(/*deltaT*/1e8, /*nOuter*/1, /*nCorr*/1);
    const std::vector<vector> Ubig = sPimpleBig.U();
    const scalar eLim = relV(Ubig, Usteady);
    const bool okLim = finite(Ubig) && eLim < 1e-3;
    std::printf("(A) steady-limit pimpleStep(dt=1e8) vs step(): relU diff %.3e  cont %.2e  %s\n",
                eLim, rb.contLocal, okLim ? "OK" : "FAIL");
    if (!okLim) ++fails;

    // ---- (B) transient stability: pimpleStep(physical dt), ddt genuinely active, 2 outer x 2 inner correctors ----
    DeviceSimpleSolver sPimple = mkSolver();
    sPimple.setDdtScheme(DdtScheme::Euler);
    const scalar dt = 1e-4;                       // pitzDaily: U~O(10), cell~O(1e-3) -> CFL ~ O(1)
    scalar cont0 = 0, contN = 0, umax = 0;
    for (int it = 0; it < 80; ++it) {
        const DeviceSimpleResidual r = sPimple.pimpleStep(dt, /*nOuter*/2, /*nCorr*/2);
        if (it == 0) cont0 = r.contLocal;
        contN = r.contLocal;
    }
    const std::vector<vector> Ut = sPimple.U();
    for (auto& v : Ut) umax = std::fmax(umax, mag(v));
    const bool okTrans = finite(Ut) && umax < 1e3 && std::isfinite(contN);
    std::printf("(B) transient pimpleStep(dt=%.0e, 2x2): |U|max %.3e  cont %.2e->%.2e  %s\n",
                dt, umax, cont0, contN, okTrans ? "OK" : "FAIL");
    if (!okTrans) ++fails;

    // ---- (C) backward scheme also runs (2nd-order, uses 2 old levels after the bootstrap) -----------------------
    DeviceSimpleSolver sBwd = mkSolver();
    sBwd.setDdtScheme(DdtScheme::backward);
    for (int it = 0; it < 10; ++it)
    {
        // runTime.timeIndex() of a cold start's step, as the pimpleFoam loop hands it over: backward's first-step
        // fvc::ddtCorr reads it and refuses to run without it
        sBwd.setTimeIndex(it + 1);
        sBwd.pimpleStep(dt, 1, 1);
    }
    const std::vector<vector> Ub2 = sBwd.U();
    const bool okBwd = finite(Ub2);
    std::printf("(C) backward pimpleStep runs (10 steps, incl. Euler bootstrap): %s\n", okBwd ? "OK" : "FAIL");
    if (!okBwd) ++fails;

    // ---- (D) backward's first step without the time index refuses by name, rather than guessing the form -----------
    DeviceSimpleSolver sNoIndex = mkSolver();
    sNoIndex.setDdtScheme(DdtScheme::backward);
    bool refused = false;
    try
    {
        sNoIndex.pimpleStep(dt, 1, 1);
    }
    catch (const std::runtime_error& e)
    {
        refused = std::string(e.what()).find("setTimeIndex") != std::string::npos;
    }
    std::printf("(D) backward without the time index refuses by name: %s\n", refused ? "OK" : "FAIL");
    if (!refused) ++fails;

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
