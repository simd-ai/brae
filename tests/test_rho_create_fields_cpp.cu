// rhoSimpleFoam createFields.H / compressibleCreatePhi.H / createFieldRefs.H / pressureControl,
// against REAL OpenFOAM.
//
// The oracle is OpenFOAM's own rhoSimpleFoam run with `-postProcess -func "writeObjects(phi,rho)"`.
// rhoSimpleFoam.C includes postProcess.H, so that path constructs createFields.H's field set and writes
// it WITHOUT solving anything -- which is exactly the state this component is responsible for. The gate
// script tests/rho_createfields_vs_openfoam.sh produces it.
//
// WHAT THIS IS REALLY TESTING, and it is not the plumbing:
//
//   compressibleCreatePhi.H is  linearInterpolate(rho*U) & Sf   -- the PRODUCT is interpolated
//   rhoSimpleFoam's pEqn.H is   interpolate(rho)*flux(HbyA)     -- the FACTORS are interpolated
//
// Both forms appear in the same solver, one file apart, and on a non-uniform mesh they differ. Getting
// this wrong changes the mass flux the very first iteration starts from, and it is invisible in the
// dictionaries -- exactly the class of defect that case-by-case porting finds three weeks later as a
// fourth-digit pressure mismatch. So the test does not merely check that brae's phi matches OpenFOAM's;
// it ALSO computes the other form and asserts that one does NOT match. Without that control, a test that
// passes proves only that the two forms happen to agree on this mesh.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "rhoCreateFields_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static int failures = 0;

static void report(
    const std::string& what,
    double got,
    double bound)
{
    // A bound of exactly zero means EXACT equality is required.
    const bool ok = (bound > 0.0) ? (got < bound) : (got <= 0.0);
    if (!ok) ++failures;
    std::printf("     %-46s %.6e   %s\n", what.c_str(), got,
                ok ? "ok" : ("FAIL (bound " + std::to_string(bound) + ")").c_str());
}

static void check(
    const std::string& what,
    bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-46s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

static double relL2(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    double num = 0.0, den = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <coldTimeDir> <oracleTimeDir> [badEnergyCaseDir]\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];   // mesh + system + constant
    const std::string cold    = argv[2];   // p, T, U only -- the cold-start input
    const std::string oracle  = argv[3];   // OpenFOAM's rho and phi, from the SAME p, T, U

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    std::printf("rhoSimpleFoam createFields vs OpenFOAM (%s, %d cells)\n", caseDir.c_str(), (int)nC);

    // ---- 1. COLD START: no rho, no phi on disk. Both must be COMPUTED, and must match OpenFOAM. ----
    std::printf("  1. cold start (input dir has p, T, U only)\n");
    cpu::rhoSimple::RhoSimpleFields f = cpu::rhoSimple::createFields(cold, caseDir, simpleDict, &fvSolution, m, g, patches);
    check("rho was COMPUTED, not read", !f.rhoWasRead);
    check("phi was COMPUTED, not read", !f.phiWasRead);
    check("energy variable is 'e' (sensibleInternalEnergy)", f.heName == "e");
    // The fixture must actually be able to tell the two flux forms apart. A uniform rho makes
    // interpolate(rho*U) and interpolate(rho)*interpolate(U) algebraically identical, so a fixture that
    // starts uniform passes check 1 no matter which form is implemented.
    {
        scalar rmin = f.rho.internal[0], rmax = f.rho.internal[0];
        for (label c = 0; c < nC; ++c)
        {
            rmin = std::min(rmin, f.rho.internal[c]);
            rmax = std::max(rmax, f.rho.internal[c]);
        }
        check("fixture rho is NON-uniform (else nothing below discriminates)",
              (double)(rmax - rmin) / (double)rmax > 1e-6);
    }

    // rho = thermo.rho(): the equation of state on the SAME p and T the thermo read.
    const FieldData<scalar> ofRhoFd = readField<scalar>(oracle + "/rho");
    GeometricField<scalar> ofRho = buildField<scalar>(ofRhoFd, patches, nC);
    report("rho vs OpenFOAM (L2 rel)", relL2(f.rho.internal, ofRho.internal), 1e-12);

    // phi = linearInterpolate(rho*U) & Sf.
    const FieldData<scalar> ofPhiFd = readField<scalar>(oracle + "/phi");
    const std::vector<scalar>& ofPhi = ofPhiFd.internalField;
    const double phiErr = relL2(f.phi.internal, ofPhi);
    report("phi vs OpenFOAM (L2 rel)", phiErr, 1e-12);

    // ---- 2. THE CONTROL. The OTHER flux form -- interpolate(rho)*flux(U), which is what pEqn.H uses
    //         for phiHbyA and what brae's fvc::rhoFlux computes -- must NOT reproduce OpenFOAM's initial
    //         phi. If it does, this mesh cannot tell the two apart and check 1's bound proves nothing.
    std::printf("  2. control -- the OTHER flux form must NOT match\n");
    const SurfaceScalarField wrong = fvc::rhoFlux(f.rho.internal, f.U, m, g, patches);
    const double wrongErr = relL2(wrong.internal, ofPhi);
    check("interpolate(rho)*flux(U) differs from OpenFOAM", wrongErr > 1e-6);
    std::printf("     %-46s %.6e\n", "  (its error, for the record)", wrongErr);
    check("and it is worse than the product form", wrongErr > phiErr);

    // ---- 3. RESTART: rho and phi ARE on disk (the oracle dir), so both must be READ verbatim. ----
    std::printf("  3. restart (rho and phi present on disk)\n");
    cpu::rhoSimple::RhoSimpleFields r = cpu::rhoSimple::createFields(oracle, caseDir, simpleDict, &fvSolution, m, g, patches);
    check("rho was READ", r.rhoWasRead);
    check("phi was READ", r.phiWasRead);
    report("read rho is the file's, exactly", relL2(r.rho.internal, ofRho.internal), 0.0);
    report("read phi is the file's, exactly", relL2(r.phi.internal, ofPhi), 0.0);

    // ---- 4. createFieldRefs.H + domainIntegrate. psi = 1/(R T), initialMass = sum(rho*V). ----
    std::printf("  4. psi and initialMass\n");
    double psiErr = 0.0;
    for (label c = 0; c < nC; ++c)
    {
        const double want = 1.0 / (f.thermo.R * (double)f.T.internal[c]);
        psiErr = std::max(psiErr, std::fabs((double)f.psi[c] - want) / want);
    }
    report("psi == 1/(R T) (max rel)", psiErr, 1e-14);
    double mass = 0.0;
    for (label c = 0; c < nC; ++c) mass += (double)ofRho.internal[c] * (double)g.V()[c];
    report("initialMass == sum(rho*V) (rel)",
           std::fabs((double)f.initialMass - mass) / mass, 1e-12);

    // ---- 5. REFUSALS. Each must throw, and the message must NAME what is missing. A refusal that
    //         says nothing is only marginally better than a silent substitution.
    std::printf("  5. refusals\n");
    {
        // thermo.validate(args.executable(), "h", "e") -- rhoSimpleFoam transports exactly these two.
        // The gate script writes badEnergyDir as a copy of the case whose `energy` is something else.
        bool threw = false;
        std::string msg;
        if (argc > 4)
        {
            try
            {
                (void)cpu::rhoSimple::createFields(caseDir + "/0.orig", std::string(argv[4]), simpleDict,
                                        &fvSolution, m, g, patches);
            }
            catch (const std::exception& e) { threw = true; msg = e.what(); }
            check("an unsupported thermo energy is refused", threw);
            check("and the refusal names the energy it found",
                  msg.find("absoluteEnthalpy") != std::string::npos);
        }

        // ---- THE CASE'S OWN kEpsilon COEFFICIENTS REACH construction-time correctNut ----------
        // createFields is where OpenFOAM's turbulence->validate() equivalent runs: correctNut writes
        // nut = Cmu*k^2/eps and EddyDiffusivity follows with alphat = rho*nut/Prt, and those two fields
        // are what the FIRST momentum and energy solves consume. brae used to default-construct the
        // coefficients here and hardcode `const scalar Prt = 1.0;` -- discarding a Prt it had already
        // parsed into f.thermo.Prt from the dict OpenFOAM reads it from.
        //
        // No shipped fixture declares either, which is exactly why it survived. argv[5] is the same case
        // with `kEpsilonCoeffs { Cmu 0.05; Prt 0.5; }` added, so the two must produce DIFFERENT nut and
        // alphat. Ratios rather than bounds, because the relationship is exact: nut scales with Cmu and
        // alphat with Cmu/Prt, so a substitution shows up as a ratio of 1 instead.
        if (argc > 5)
        {
            const cpu::rhoSimple::RhoSimpleFields fc =
                cpu::rhoSimple::createFields(caseDir + "/0.orig", std::string(argv[5]), simpleDict,
                                             &fvSolution, m, g, patches);
            const cpu::rhoSimple::RhoSimpleFields fd =
                cpu::rhoSimple::createFields(caseDir + "/0.orig", caseDir, simpleDict,
                                             &fvSolution, m, g, patches);
            double nutRatio = 0.0, alphatRatio = 0.0;
            std::size_t nNut = 0, nAl = 0;
            for (std::size_t c = 0; c < fd.nut.internal.size() && c < fc.nut.internal.size(); ++c)
            {
                if (std::fabs(fd.nut.internal[c]) > 0.0)
                { nutRatio += (double)(fc.nut.internal[c] / fd.nut.internal[c]); ++nNut; }
            }
            for (std::size_t c = 0; c < fd.alphat.internal.size() && c < fc.alphat.internal.size(); ++c)
            {
                if (std::fabs(fd.alphat.internal[c]) > 0.0)
                { alphatRatio += (double)(fc.alphat.internal[c] / fd.alphat.internal[c]); ++nAl; }
            }
            if (nNut) nutRatio /= (double)nNut;
            if (nAl)  alphatRatio /= (double)nAl;
            std::printf("     %-34s nut x%.4f (expect 0.5556)   alphat x%.4f (expect 1.1111)\n",
                        "case coefficients reach correctNut", nutRatio, alphatRatio);
            // Cmu 0.05/0.09 = 0.5556 on nut; alphat carries that over Prt 0.5/1.0, i.e. x2 -> 1.1111.
            check("the case's Cmu reaches nut at construction",
                  nNut > 0 && std::fabs(nutRatio - 0.05/0.09) < 1e-6);
            check("the case's Prt reaches alphat at construction",
                  nAl > 0 && std::fabs(alphatRatio - (0.05/0.09)/0.5) < 1e-6);
        }
        else
        {
            std::printf("     %-46s %s\n", "unsupported-energy fixture not supplied", "SKIP");
        }
    }
    {
        bool threw = false;
        try
        {
            (void)cpu::rhoSimple::createFields(caseDir + "/0.orig", caseDir + "/__no_such_case__", simpleDict,
                                    &fvSolution, m, g, patches);
        }
        catch (const std::exception&) { threw = true; }
        check("a missing thermophysicalProperties is refused", threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
