// TURBULENT end-to-end: the _cpp SIMPLE loop with k-epsilon, on OpenFOAM's own converged solution.
//
// Method is the repo's "cap-and-measure": start from OpenFOAM's CONVERGED fields and take ONE iteration.
// A correct solver leaves a converged solution where it is, so the fields must move by no more than the
// convergence floor the case itself asked for (residualControl 1e-6). This tests the whole coupled loop
// without a thousand-iteration run, and -- unlike a run-to-convergence comparison -- a defect shows up as
// a field that MOVES rather than as a slightly different converged state.
//
// What this adds over the component tests, all of which already pass:
//   * nuEff = nu + nut with the boundary value taken from nut's OWN boundary field (a wall function's
//     nut_wall), not the owner cell -- the defect class that made boundary viscosity 2000x too small once;
//   * the LAGGED ordering: turbulence->correct() runs at the END of the iteration (simpleFoam.C:93-94),
//     so UEqn uses the previous iteration's nut. Correcting first is a different algorithm that still
//     converges to something plausible.
//
// Run: test_simple_turbulent_cpp <caseDir> <convergedTimeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "simpleFoam_cpp.cuh"
#include "simple_foam.cuh"   // the pre-existing, OpenFOAM-validated host path, for equivalence

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

static scalar relS(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(a[i] - b[i]));
        mg = std::fmax(mg, std::fabs(b[i]));
    }
    return mg > 0 ? mx / mg : mx;
}

static scalar relV(const std::vector<vector>& a, const std::vector<vector>& b)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        mx = std::fmax(mx, std::fmax(std::fabs(a[i].x - b[i].x),
                       std::fmax(std::fabs(a[i].y - b[i].y), std::fabs(a[i].z - b[i].z))));
        mg = std::fmax(mg, std::fmax(std::fabs(b[i].x),
                       std::fmax(std::fabs(b[i].y), std::fabs(b[i].z))));
    }
    return mg > 0 ? mx / mg : mx;
}

static void report(scalar rel, const char* nm, scalar tol)
{
    const bool ok = rel <= tol;
    if (!ok) ++g_fails;
    std::printf("  %-12s rel=%.3e  (tol %.0e)  %s\n", nm, rel, tol, ok ? "OK" : "FAIL");
}

static void check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <convergedTime>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar nu = 1e-5;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    std::printf("test_simple_turbulent_cpp:\n");

    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    check(!cd.consistent, "this case is plain SIMPLE (consistent no)");
    cpu::SimpleControl ctl(cd);

    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, patches);

    GeometricField<scalar> k =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/k"), patches, nC);
    GeometricField<scalar> eps =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/epsilon"), patches, nC);
    GeometricField<scalar> nut =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), patches, nC);
    k.evaluateBoundary(); eps.evaluateBoundary(); nut.evaluateBoundary();

    // OpenFOAM's converged state, kept for comparison.
    const std::vector<vector> U0 = f.U.internal;
    const std::vector<scalar> p0 = f.p.internal, k0 = k.internal, e0 = eps.internal, n0 = nut.internal;
    const std::vector<scalar> phi0 = f.phi.internal;

    cpu::TurbulenceState turb;
    turb.k = &k; turb.epsilon = &eps; turb.nut = &nut;
    turb.relaxK = 0.7; turb.relaxEpsilon = 0.7;

    cpu::StepInput in;
    in.nu = nu;
    in.turb = &turb;
    in.relaxU = 0.7; in.relaxP = 0.3;

    const cpu::Residuals res = cpu::simpleStep(f, ctl, in, m, g, patches);
    std::printf("  -- one turbulent SIMPLE iteration from OpenFOAM's converged %s\n", t.c_str());
    std::printf("     initial residuals: U=%.3e p=%.3e\n",
                res.count("U") ? res.at("U") : -1.0, res.count("p") ? res.at("p") : -1.0);

    // A gross-error check, NOT the sharp gate. The sharp gate is the equivalence block below.
    //
    // A first version of this test asserted k and nut would move by less than 1e-3, on the reasoning that
    // the case's residualControl is 1e-6. That reasoning is wrong: residualControl bounds the linear
    // solver's INITIAL RESIDUAL, not the per-iteration field change, and the turbulence fields move
    // considerably more than the momentum fields from a converged state. Measured here, and reproduced
    // EXACTLY (bit for bit) by the pre-existing OpenFOAM-validated host path:
    //
    //     U 2.2e-07   p 1.2e-06   phi 9.2e-08   |   k 3.6e-03   epsilon 8.8e-04   nut 1.5e-03
    //
    // So the bounds below are set above the measured behaviour of a KNOWN-GOOD path, and exist only to
    // catch a solver that has come apart. Tightening them would be asserting something untrue about the
    // case; the real regression detector is the bit-for-bit equivalence check.
    std::printf("  -- gross-error bound (the sharp gate is the equivalence block below)\n");
    report(relV(f.U.internal, U0),   "U",       1e-4);
    report(relS(f.p.internal, p0),   "p",       1e-4);
    report(relS(f.phi.internal, phi0), "phi",   1e-4);
    report(relS(k.internal, k0),     "k",       1e-2);
    report(relS(eps.internal, e0),   "epsilon", 1e-2);
    report(relS(nut.internal, n0),   "nut",     1e-2);

    // EQUIVALENCE: the rebuilt loop against the PRE-EXISTING host path.
    //
    // src/applications/solvers/simpleFoam/simple_foam.cuh is already validated against OpenFOAM
    // end-to-end (ctest: simple_turbulent_full, pitzDailyTurb from 0/ to 1576). Running both for the SAME
    // single iteration from the SAME converged state separates two questions that the "stays put" check
    // above conflates: "is the rebuilt loop right?" and "does one iteration from OpenFOAM's converged
    // state move the fields at all?". If the two paths agree, any residual movement is a property of the
    // case and the model, not of the rebuild.
    {
        cpu::SimpleFields fo = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, patches);
        GeometricField<scalar> ko =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/k"), patches, nC);
        GeometricField<scalar> eo =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/epsilon"), patches, nC);
        GeometricField<scalar> no =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), patches, nC);
        ko.evaluateBoundary(); eo.evaluateBoundary(); no.evaluateBoundary();

        SimpleControls oc;
        oc.nu = nu; oc.relaxU = 0.7; oc.relaxP = 0.3;
        const FieldData<vector> Udata = readField<vector>(caseDir + "/" + t + "/U");
        simpleStep(fo.U, fo.p, fo.phi, Udata, m, g, patches, oc, &no);
        kepsilon::correct(fo.U, ko, eo, no, fo.phi, nu, m, g, patches, 0.7, 0.7, 1e-10, 0.0, 2000);

        std::printf("  -- rebuilt _cpp loop vs the pre-existing OpenFOAM-validated host path\n");
        report(relV(f.U.internal, fo.U.internal),   "U vs old",   1e-9);
        report(relS(f.p.internal, fo.p.internal),   "p vs old",   1e-9);
        report(relS(f.phi.internal, fo.phi.internal), "phi vs old", 1e-9);
        report(relS(k.internal, ko.internal),       "k vs old",   1e-9);
        report(relS(eps.internal, eo.internal),     "eps vs old", 1e-9);
        report(relS(nut.internal, no.internal),     "nut vs old", 1e-9);
    }

    // CONTROL 1: the turbulence must actually be coupled in. Re-run the same iteration LAMINAR (nuEff = nu,
    // no nut anywhere). If that landed in the same place, nut was never reaching the momentum equation and
    // every "stays put" line above would be meaningless.
    {
        cpu::SimpleFields fl = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, patches);
        cpu::SimpleControl ctl2(cd);
        cpu::StepInput lam;
        lam.nu = nu;
        lam.nuEff.assign(nC, nu);
        lam.nuEffBnd.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) lam.nuEffBnd[pi].assign(patches[pi].size, nu);
        lam.relaxU = 0.7; lam.relaxP = 0.3;
        cpu::simpleStep(fl, ctl2, lam, m, g, patches);
        const scalar drift = relV(fl.U.internal, U0);
        std::printf("  %-58s rel=%.3e\n", "control: the same step LAMINAR drifts far more", drift);
        check(drift > 1e-2, "nut genuinely reaches the momentum equation (control)");
    }

    // CONTROL 2: turbulence->correct() must have run and changed nothing much -- but it must have RUN.
    // A no-op correct() would leave nut bit-identical, which a converged case cannot distinguish from
    // "converged", so assert it moved at all.
    scalar nutMoved = 0;
    for (label c = 0; c < nC; ++c) nutMoved = std::fmax(nutMoved, std::fabs(nut.internal[c] - n0[c]));
    std::printf("  %-58s max|dnut|=%.3e\n", "control: turbulence->correct() ran", nutMoved);
    check(nutMoved > 0.0, "turbulence->correct() actually executed (control)");

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
