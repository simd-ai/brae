// The ENERGY boundary conditions, against REAL OpenFOAM's OWN coefficients.
//
// THE ORACLE is tools/dumpEnergyBC: it constructs the case's fluidThermo and calls
// he.boundaryFieldRef().updateCoeffs() -- the one call the energy matrix makes before it reads a single
// coefficient -- then prints what OpenFOAM's fixedEnergy, gradientEnergy and mixedEnergy wrote. So the
// comparison is against OpenFOAM's own numbers on OpenFOAM's own developed fields, with no instrumented
// build and nothing hand-derived.
//
// WHY THE COEFFICIENTS AND NOT A CONVERGED FIELD. For perfectGas + hConst the live update is
// arithmetically the number createFields already stored once: he(p,T) is p-independent and Cpv is a
// constant, so the static construction-time mapping and OpenFOAM's per-iteration one agree face for
// face. An end-to-end gas run therefore cannot tell a correct transcription from a wrong one -- it is a
// no-op either way, which is exactly the claim the ten compressible gates are evidence for. These four
// numbers per face CAN tell them apart, and the two controls below are the proof:
//
//   control A  gradientEnergy's gradient built with he(T) instead of Cpv*T.snGrad(). he is AFFINE
//              (Cp*(T-Tref)+Href); applying an offset to a slope is the trap validation/hf_vs_openfoam
//              measured as T 7.97e-03 converged on this very fixture.
//   control B  mixedEnergy's refValue built from Tw.value() instead of Tw.refValue(). On a mixed patch
//              those are different fields -- the blend and the dictionary's own number -- and they
//              coincide only when the valueFraction is 1.
//   control C  mixedEnergy's refGrad copied across unscaled instead of multiplied by Cpv. Same family of
//              mistake as A, on the class next door, which is what validation/mx_vs_openfoam measures.
//
// WHAT THIS GATE CANNOT SEE, said plainly rather than left for someone to discover: it cannot separate
// LIVE from STATIC, because on this thermo there is nothing to separate. A p-dependent or T-dependent
// thermo makes the two diverge from the second iteration; that is stage H3.4's fixture and H3.5's
// end-to-end run, and it is the reason this file exists ahead of them.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_dict.cuh"
#include "energy_boundary.cuh"
#include "rhoCreateFields_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;

static int failures = 0;

static void check(
    const std::string& what,
    bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-52s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

static void report(
    const std::string& what,
    double got,
    double bound)
{
    const bool ok = (bound > 0.0) ? (got < bound) : (got <= 0.0);
    if (!ok) ++failures;
    std::printf("     %-52s %.6e   %s\n", what.c_str(), got,
                ok ? "ok" : ("FAIL (bound " + std::to_string(bound) + ")").c_str());
}

// One row of the oracle: which patch, which face, and the coefficients OpenFOAM wrote there.
struct OracleRow
{
    std::string kind;        // MIXED | GRAD | VALUE | OTHER
    label       patch = 0;
    label       face  = 0;
    double      a = 0.0;     // MIXED refValue | GRAD gradient | VALUE value
    double      b = 0.0;     // MIXED refGrad
    double      c = 0.0;     // MIXED valueFraction
};

// Relative difference against OpenFOAM, scaled by the oracle's own magnitude so that a coefficient which
// is legitimately zero (a zeroGradient's gradient) is compared absolutely rather than blowing up.
static double relDiff(
    double got,
    double want)
{
    const double d = std::fabs(got - want);
    const double s = std::fabs(want);
    return s > 1.0 ? d / s : d;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <timeDir> <oracleFile>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string timeDir = argv[2];
    const std::string oraclePath = argv[3];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    std::printf("energy boundary conditions vs OpenFOAM (%s @ %s)\n", caseDir.c_str(), timeDir.c_str());

    // ---- the oracle ----
    std::vector<OracleRow> rows;
    std::map<label, std::string> ofType;   // patch index -> the he patch type OpenFOAM derived
    {
        std::ifstream in(oraclePath);
        if (!in) { std::printf("SKIP: cannot read oracle %s\n", oraclePath.c_str()); return 77; }
        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream ss(line);
            std::string tag;
            ss >> tag;
            if (tag == "PATCH")
            {
                label pi = 0, nf = 0;
                std::string name, type;
                ss >> pi >> name >> type >> nf;
                ofType[pi] = type;
            }
            else if (tag == "MIXED")
            {
                OracleRow r; r.kind = tag;
                ss >> r.patch >> r.face >> r.a >> r.b >> r.c;
                rows.push_back(r);
            }
            else if (tag == "GRAD" || tag == "VALUE" || tag == "OTHER")
            {
                OracleRow r; r.kind = tag;
                ss >> r.patch >> r.face >> r.a;
                rows.push_back(r);
            }
        }
    }
    // FAIL-PROOF. An oracle that failed to parse, or one from a case whose energy conditions are all
    // constraint types, would leave every loop below empty and every bound trivially satisfied.
    check("oracle parsed at least one coefficient row", !rows.empty());
    if (rows.empty()) return 1;

    // ---- brae, on the SAME fields OpenFOAM's thermo read ----
    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(timeDir, caseDir, simpleDict, &fvSolution, m, g, patches);
    // The flux switch first, exactly where the step runs it (rhoSimpleFoam_cpp.cu: one pass over every
    // patch, before anything is assembled, from the phi the iteration starts with). OpenFOAM's
    // inletOutlet does this inside its own updateCoeffs, which mixedEnergy triggers through Tw.evaluate();
    // omitting it here would leave an outlet's valueFraction at brae's construction seed and the oracle
    // would be compared against a boundary condition the solver never runs.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        f.he.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
        f.T.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
    }
    // POISON THE SEED FIRST, and this is what makes the gate bite rather than merely agree.
    //
    // Measured: with the call to updateEnergyBoundaryCoeffs removed entirely, every bound below still
    // read 0.000000e+00 on all three fixtures. That is not a flaw in the oracle -- it is the no-op the
    // header describes, and on perfectGas + hConst the construction-time mapping happens to hold the
    // right numbers already. A gate that a deleted call passes measures nothing about the call.
    //
    // So the coefficients are overwritten with a value no thermo could produce before the live update
    // runs, and the comparison can then only succeed if updateEnergyBoundaryCoeffs wrote every one of
    // them. This is not an artificial state: OpenFOAM's own he boundary arrives at updateCoeffs with
    // refValue and valueFraction never initialised at all (heThermo::init sets only the VALUE and, via
    // heBoundaryCorrection, refGrad -- heThermo.C:56-90), so a mixedEnergy that failed to write them
    // would read uninitialised memory. OpenFOAM relies on nothing being carried over here, and neither
    // may brae.
    const scalar poison = -1.0e30;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::size_t n = static_cast<std::size_t>(patches[pi].size);
        fvPatchField<scalar>& hp = *f.he.boundary[pi];
        if (auto* mp = dynamic_cast<MixedPatchField<scalar>*>(&hp))
        {
            mp->setRefValues(std::vector<scalar>(n, poison));
            mp->setRefGrad(std::vector<scalar>(n, poison));
            mp->setValueFraction(std::vector<scalar>(n, poison));
        }
        else if (auto* gp = dynamic_cast<FixedGradientPatchField<scalar>*>(&hp))
        {
            gp->setGradient(std::vector<scalar>(n, poison));
        }
        else if (auto* vp = dynamic_cast<FixedValuePatchField<scalar>*>(&hp))
        {
            vp->setStoredValues(std::vector<scalar>(n, poison));
        }
    }

    // The two lines the solver runs at the head of the energy assembly, in that order.
    f.T.evaluateBoundary();
    cpu::updateEnergyBoundaryCoeffs(f.he, f.T, f.p, f.thermo, patches);

    // ---- compare, coefficient by coefficient ----
    std::size_t nMixed = 0, nGrad = 0, nValue = 0;
    double worstRefValue = 0.0, worstRefGrad = 0.0, worstVf = 0.0, worstGrad = 0.0, worstValue = 0.0;
    for (const OracleRow& r : rows)
    {
        const std::size_t pi = static_cast<std::size_t>(r.patch);
        if (pi >= patches.size()) { check("oracle patch index within the mesh", false); continue; }
        const fvPatchField<scalar>& hp = *f.he.boundary[pi];
        const std::size_t fi = static_cast<std::size_t>(r.face);
        if (fi >= hp.value().size()) { check("oracle face index within the patch", false); continue; }

        if (r.kind == "MIXED")
        {
            const auto* mp = dynamic_cast<const MixedPatchField<scalar>*>(&hp);
            if (!mp) { check("he patch is mixed where OpenFOAM's is", false); continue; }
            const std::vector<scalar>  ref = mp->refValues();
            const std::vector<scalar>* vf  = mp->valueFractionPtr();
            const std::vector<scalar>* rg  = mp->refGradPtr();
            worstRefValue = std::max(worstRefValue, relDiff((double)ref[fi], r.a));
            worstRefGrad  = std::max(worstRefGrad,  relDiff(rg ? (double)(*rg)[fi] : 0.0, r.b));
            worstVf       = std::max(worstVf,       relDiff(vf ? (double)(*vf)[fi] : 0.0, r.c));
            ++nMixed;
        }
        else if (r.kind == "GRAD")
        {
            // OpenFOAM's gradientEnergy is a fixedGradient. brae builds a zeroGradient where T's is a
            // zeroGradient, whose prescribed gradient is identically zero -- the same four matrix
            // coefficients and the same evaluate (see energy_boundary.cuh). Both are compared here
            // against OpenFOAM's gradient, so the equivalence is asserted rather than assumed.
            const auto* gp = dynamic_cast<const FixedGradientPatchField<scalar>*>(&hp);
            const auto* zp = dynamic_cast<const ZeroGradientPatchField<scalar>*>(&hp);
            if (!gp && !zp) { check("he patch is a gradient family where OpenFOAM's is", false); continue; }
            double got = 0.0;
            if (gp)
            {
                const std::vector<scalar>* rg = gp->refGradPtr();
                got = rg ? (double)(*rg)[fi] : 0.0;
            }
            worstGrad = std::max(worstGrad, relDiff(got, r.a));
            ++nGrad;
        }
        else if (r.kind == "VALUE")
        {
            worstValue = std::max(worstValue, relDiff((double)hp.value()[fi], r.a));
            ++nValue;
        }
        // OTHER rows are the constraint and calculated patches, which heBoundaryTypes leaves with T's own
        // type and no energy condition to update. Nothing is compared there because nothing is computed.
    }

    std::printf("  1. OpenFOAM's own coefficients, after he.boundaryField().updateCoeffs()\n");
    std::printf("     (rows: %zu mixed, %zu gradient, %zu fixedValue)\n", nMixed, nGrad, nValue);
    // 1e-13 rather than 0: the oracle is ASCII at 17 significant digits, which is round-trip exact for a
    // double, but the fields brae reads back are the same 17-digit text and the two codes reach the
    // products by different orderings.
    if (nMixed) report("mixedEnergy  refValue",      worstRefValue, 1e-13);
    if (nMixed) report("mixedEnergy  refGrad",       worstRefGrad,  1e-13);
    if (nMixed) report("mixedEnergy  valueFraction", worstVf,       1e-13);
    if (nGrad)  report("gradientEnergy gradient",    worstGrad,     1e-13);
    if (nValue) report("fixedEnergy  value",         worstValue,    1e-13);
    check("at least one energy coefficient was compared", (nMixed + nGrad + nValue) > 0);

    // ---- the controls: three transcriptions that must NOT reproduce OpenFOAM ----
    std::printf("  2. controls -- the wrong transcriptions must NOT match\n");
    double ctlA = 0.0, ctlB = 0.0, ctlC = 0.0;
    std::size_t nA = 0, nB = 0, nC = 0;
    for (const OracleRow& r : rows)
    {
        const std::size_t pi = static_cast<std::size_t>(r.patch);
        if (pi >= patches.size()) continue;
        const std::size_t fi = static_cast<std::size_t>(r.face);
        const fvPatchField<scalar>& Tp = *f.T.boundary[pi];
        if (fi >= Tp.value().size()) continue;
        const scalar pw = f.p.boundary[pi]->value()[fi];
        const scalar Tw = Tp.value()[fi];

        if (r.kind == "GRAD")
        {
            const std::vector<scalar> Tsn = Tp.snGrad(f.T.internal);
            if (std::fabs((double)Tsn[fi]) < 1e-30) continue;   // zeroGradient cannot discriminate
            ctlA = std::max(ctlA, relDiff((double)thermoHeOf(pw, Tsn[fi], f.thermo), r.a));
            ++nA;
        }
        else if (r.kind == "MIXED")
        {
            ctlB = std::max(ctlB, relDiff((double)thermoHeOf(pw, Tw, f.thermo), r.a));
            ++nB;
            const std::vector<scalar>* Tg = Tp.refGradPtr();
            if (Tg && std::fabs((double)(*Tg)[fi]) > 1e-30)
            {
                ctlC = std::max(ctlC, relDiff((double)(*Tg)[fi], r.b));
                ++nC;
            }
        }
    }
    if (nA)
    {
        check("A: gradient as he(snGrad T) differs from OpenFOAM", ctlA > 1e-6);
        std::printf("     %-52s %.6e\n", "  (its error, for the record)", ctlA);
    }
    if (nB)
    {
        check("B: refValue from T's VALUE differs from OpenFOAM", ctlB > 1e-6);
        std::printf("     %-52s %.6e\n", "  (its error, for the record)", ctlB);
    }
    if (nC)
    {
        check("C: refGrad copied unscaled differs from OpenFOAM", ctlC > 1e-6);
        std::printf("     %-52s %.6e\n", "  (its error, for the record)", ctlC);
    }
    // Which controls a fixture can run depends on which boundary conditions it carries; the gate script
    // asserts the coverage across its three fixtures, and this line says what THIS one exercised.
    std::printf("     controls exercised: A=%zu B=%zu C=%zu\n", nA, nB, nC);

    std::printf("%s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
