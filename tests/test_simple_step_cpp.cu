// END-TO-END: one SIMPLE iteration built from the ported _cpp components, against OpenFOAM's own
// dumpSimpleStep output (step.dat).
//
// This is the gate that says the composition is right, not just the parts. Each component already has its
// own test against an OpenFOAM oracle; this one starts from OpenFOAM's converged step-282 fields, runs a
// single iteration through createFields -> simpleControl -> UEqn -> pEqn, and compares p, U and phi
// (internal AND every boundary patch) with what OpenFOAM produced from the same input.
//
// It also pins the things a driver can get wrong that no component test can see:
//   * the momentum predictor must solve a COPY of UEqn, or rAU and HbyA come from the wrong matrix;
//   * p is relaxed BETWEEN the flux correction and the velocity correction;
//   * phi read from disk must be used as read (READ_IF_PRESENT), not recomputed.
//
// Run: test_simple_step_cpp <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "simpleFoam_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

using namespace brae;

static int g_fails = 0;

// 1e-9, not the 1e-5 the older step test uses. The composed _cpp path reproduces OpenFOAM to 2.5e-11 on
// p, 1.6e-12 on U and 1.2e-11 on phi, so a 1e-5 gate would let a four-order regression through unnoticed.
// The tolerance is set to what the code actually achieves plus margin; if it ever has to be loosened,
// that is a finding, not a maintenance step.
static void cmp(const std::vector<scalar>& a, const std::vector<scalar>& of, const char* nm,
                scalar tol = 1e-9)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < of.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(a[i] - of[i]));
        mg = std::fmax(mg, std::fabs(of[i]));
    }
    const scalar rel = mg > 0 ? mx / mg : mx;
    if (!(rel <= tol)) ++g_fails;
    std::printf("  %-18s n=%6zu maxAbs=%.3e rel=%.3e %s\n",
                nm, of.size(), mx, rel, rel <= tol ? "OK" : "FAIL");
}

static void check(bool ok, const char* what)
{
    std::printf("  %-56s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    std::printf("test_simple_step_cpp:\n");

    // ---- createFields --------------------------------------------------------------------
    FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");
    cpu::SimpleFields f = cpu::createFields(caseDir + "/282", simpleDict, m, g, patches);

    std::printf("  -- createFields\n");
    check(f.phiWasRead, "phi was READ from disk, not recomputed from U");
    check(f.p.internal.size() == static_cast<std::size_t>(nC), "p read with the right size");
    // This case fixes p at the outlet, so it needs no reference cell -- and adjustPhi must therefore
    // not run. If that ever flips, the pressure level is being pinned somewhere it should not be.
    check(!cpu::needReference(f.p), "p does NOT need a reference (outlet fixes its value)");
    check(f.pRefCell == -1, "no reference cell was set");

    // ---- simpleControl -------------------------------------------------------------------
    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    std::printf("  -- simpleControl (from system/fvSolution SIMPLE)\n");
    std::printf("     nNonOrthogonalCorrectors=%d momentumPredictor=%d consistent=%d residualControl=%zu\n",
                (int)cd.nNonOrthogonalCorrectors, (int)cd.momentumPredictor, (int)cd.consistent,
                cd.residualControl.size());
    // The parser must SEE this case's settings, including the regex residualControl key.
    check(cd.consistent, "the SIMPLE dict's `consistent yes` was parsed");
    check(cd.nNonOrthogonalCorrectors == 0, "nNonOrthogonalCorrectors read as 0");
    check(cd.residualControl.size() == 3, "three residualControl entries read");
    check(cpu::applyToField(cd.residualControl, "epsilon") == 2,
          "the regex key \"(k|epsilon|omega|f|v2)\" matches field 'epsilon'");
    check(cpu::applyToField(cd.residualControl, "T") == -1, "...and does not match 'T' (control)");

    // THE FIXTURE'S ORACLE IS PLAIN SIMPLE, NOT SIMPLEC.
    //
    // This case's dictionary says `consistent yes`, but step.dat was produced by the dumpSimpleStep app,
    // which ran the plain SIMPLE corrector: the pre-existing test_simple_step reproduces step.dat to 1e-5
    // with no H1/snGrad terms anywhere. So the comparison below is run with SIMPLEC switched OFF, and
    // that is a property of the FIXTURE, recorded here rather than papered over.
    //
    // This used to assert that SIMPLEC was REFUSED, which was the honest thing while it was unported.
    // SIMPLEC is implemented now, so the refusal is gone and the mismatch is no longer self-enforcing --
    // hence the explicit assertion below that the flag really is off before step.dat is compared against.
    // A SIMPLEC oracle is still needed to test the SIMPLEC path at THIS granularity; end to end it is
    // covered by ctest stock_pitzdaily_vs_openfoam.
    {
        cpu::SimpleControl simplec(cd);
        check(simplec.consistent(), "the fixture really does ask for SIMPLEC (control on the note above)");
    }
    cd.consistent = false;                  // match how step.dat was generated -- see the note above
    check(!cd.consistent, "SIMPLEC is OFF for the step.dat comparison (fixture guard)");
    cpu::SimpleControl ctl(cd);

    // The corrector loop must run exactly nNonOrthogonalCorrectors+1 times and then reset.
    {
        int n = 0;
        cpu::SimpleControl probe(cd);
        while (probe.correctNonOrthogonal()) { ++n; if (n > 20) break; }
        check(n == cd.nNonOrthogonalCorrectors + 1, "correctNonOrthogonal runs nNonOrth+1 times");
        int n2 = 0;
        while (probe.correctNonOrthogonal()) { ++n2; if (n2 > 20) break; }
        check(n2 == n, "...and resets, so the next iteration gets the same count");
    }

    // ---- one SIMPLE step -----------------------------------------------------------------
    cpu::StepInput in;
    in.nuEff.assign(nC, nu);
    in.nuEffBnd.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) in.nuEffBnd[pi].assign(patches[pi].size, nu);
    in.relaxU = 0.7; in.relaxP = 0.3;

    const cpu::Residuals res = cpu::simpleStep(f, ctl, in, m, g, patches);
    std::printf("  -- one SIMPLE iteration: initial residuals U=%.3e p=%.3e\n",
                res.count("U") ? res.at("U") : -1.0, res.count("p") ? res.at("p") : -1.0);

    // ---- against OpenFOAM ----------------------------------------------------------------
    std::ifstream sf(caseDir + "/step.dat");
    if (!sf) { std::printf("FAIL no step.dat\n"); return 1; }
    label snC, snIf, snp; sf >> snC >> snIf >> snp;
    std::vector<scalar> pof(snC), phiIntOf(snIf), Uof(3*snC), Ucf(3*snC);
    for (auto& v : pof) sf >> v;
    for (label c = 0; c < snC; ++c)
    {
        sf >> Uof[3*c] >> Uof[3*c+1] >> Uof[3*c+2];
        Ucf[3*c] = f.U.internal[c].x; Ucf[3*c+1] = f.U.internal[c].y; Ucf[3*c+2] = f.U.internal[c].z;
    }
    for (auto& v : phiIntOf) sf >> v;

    std::printf("  -- vs OpenFOAM dumpSimpleStep\n");
    cmp(f.p.internal, pof, "p");
    cmp(Ucf, Uof, "U");
    cmp(f.phi.internal, phiIntOf, "phi:internal");
    for (label pp = 0; pp < snp; ++pp)
    {
        std::string name; label sz; sf >> name >> sz;
        std::vector<scalar> fb(sz);
        for (label i = 0; i < sz; ++i) sf >> fb[i];
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k) if (patches[k].name == name) { pk = k; break; }
        cmp(f.phi.boundary[pk], fb, (name + ":phi").c_str());
    }

    // CONTROL: the step must have MOVED the fields. Starting from OpenFOAM's step-282 solution the
    // changes are small, so a driver that did nothing at all would pass every comparison above.
    const cpu::SimpleFields f0 = cpu::createFields(caseDir + "/282", simpleDict, m, g, patches);
    scalar moved = 0;
    for (label c = 0; c < nC; ++c) moved = std::fmax(moved, std::fabs(f.p.internal[c] - f0.p.internal[c]));
    std::printf("  %-56s max|dp|=%.3e\n", "control: the iteration changed p", moved);
    check(moved > 0.0, "the SIMPLE step actually did something (control)");

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
