// CUDA rhoPcEqn (SIMPLEC) against the _cpp reference, stage by stage, BOTH BRANCHES.
//
// The reference is itself gated against OpenFOAM's own dumps (tests/rho_pceqn_vs_openfoam.sh), so this
// closes OpenFOAM -> _cpp -> CUDA for the consistent pressure equation. Fourth of the device twins.
//
// WHAT THIS GATE HAS THAT test_rho_peqn_cuda DOES NOT, and it is the reason SIMPLEC needs its own:
//
//   * rAtU = 1/(1/rAU - UEqn.H1()) is compared SEPARATELY from rAU, and a control asserts the two
//     actually differ. If H1 came back zero, rAtU would silently equal rAU and every downstream number
//     would still agree -- the equation would just be SIMPLE wearing SIMPLEC's name.
//   * The SIMPLEC flux correction and the HbyA correction are asserted to move their fields
//     INDEPENDENTLY. They are one change: in the incompressible lineage, keeping rAtU while omitting the
//     flux correction made the converged velocity WORSE than doing neither, so a gate that only checked
//     one of them would pass the configuration that is worse than not trying.
//
// rho and psi ARE SYNTHESIZED non-uniform, and pRefCell is DERIVED from p's boundary conditions rather
// than asserted -- both for the reasons test_rho_peqn_cuda.cu records, the second of which cost that gate
// a false failure: forcing a reference on a case whose p is fixed made the host skip adjustPhi while the
// device ran it.
//
// Run: test_rho_pceqn_cuda <caseDir> <timeDir> [laminar]     env: BRAE_TEST_TRANSONIC=1
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "rhoUEqn_cpp.cuh"
#include "rhoUEqn.cuh"
#include "rhoPEqn_cpp.cuh"
#include "rhoPcEqn_cpp.cuh"
#include "rhoPcEqn.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

static void cmp(const std::vector<scalar>& gpu,
                const std::vector<scalar>& ref,
                const char*                nm,
                scalar                     tol)
{
    if (gpu.size() != ref.size())
    {
        std::printf("  %-34s SIZE MISMATCH %zu vs %zu  FAIL\n", nm, gpu.size(), ref.size());
        ++g_fails;
        return;
    }
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(gpu[i] - ref[i]));
        mg = std::fmax(mg, std::fabs(ref[i]));
    }
    const scalar rel = mg > 0 ? mx / mg : mx;
    const bool ok = rel <= tol;
    if (!ok) ++g_fails;
    std::printf("  %-34s n=%6zu rel=%.3e  %s\n", nm, ref.size(), rel, ok ? "OK" : "FAIL");
}

static void check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <timeDir> [laminar]\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar nu = 1e-5;

    const char* trEnv = std::getenv("BRAE_TEST_TRANSONIC");
    const bool transonic = trEnv && trEnv[0] == '1';

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    U.evaluateBoundary();
    GeometricField<scalar> p =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    p.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    std::vector<std::vector<scalar>> phiBnd(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phiBnd[pi].assign(fvp[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
        {
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
            {
                phiBnd[pi] = b.values;
            }
        }
    }

    const bool forceLaminar = (argc > 3 && std::string(argv[3]) == "laminar");
    const bool turbulent = !forceLaminar && std::filesystem::exists(caseDir + "/" + t + "/nut");
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

    scalar xMin = 1e300, xMax = -1e300;
    for (label c = 0; c < nC; ++c)
    {
        xMin = std::fmin(xMin, g.C()[c].x);
        xMax = std::fmax(xMax, g.C()[c].x);
    }
    const scalar xSpan = (xMax > xMin) ? (xMax - xMin) : 1.0;
    auto frac = [&](const vector& x) { return (x.x - xMin) / xSpan; };

    std::vector<scalar> rhoC(nC), psiC(nC);
    for (label c = 0; c < nC; ++c)
    {
        rhoC[c] = scalar(0.8) + scalar(0.6) * frac(g.C()[c]);
        psiC[c] = scalar(7.0e-6) + scalar(4.0e-6) * frac(g.C()[c]);
    }
    std::vector<std::vector<scalar>> rhoB(fvp.size()), psiB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        rhoB[pi].resize(fvp[pi].size);
        psiB[pi].resize(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            rhoB[pi][i] = scalar(0.8) + scalar(0.6) * frac(fvp[pi].Cf[i]);
            psiB[pi][i] = scalar(7.0e-6) + scalar(4.0e-6) * frac(fvp[pi].Cf[i]);
        }
    }
    scalar rMin = 1e300, rMax = -1e300;
    for (label c = 0; c < nC; ++c) { rMin = std::fmin(rMin, rhoC[c]); rMax = std::fmax(rMax, rhoC[c]); }

    std::printf("test_rho_pceqn_cuda:  SIMPLEC  branch=%s  (%s)  rho in [%.4f .. %.4f]\n",
                transonic ? "TRANSONIC" : "subsonic",
                turbulent ? "TURBULENT" : "laminar", (double)rMin, (double)rMax);
    check((rMax - rMin) / rMax > 0.1, "rho varies across the domain (fixture discriminates)");

    // ---- the momentum matrix both paths consume ---------------------------------------------
    const scalar relaxU = 0.7;
    cpu::rhoSimple::RhoMomentumInput mi;
    mi.phi = &phiF.internalField;  mi.phiBnd = &phiBnd;
    mi.rho = &rhoC;                mi.rhoBnd = &rhoB;
    mi.nuEff = &nuEffC;            mi.nuEffBnd = &nuEffB;
    mi.relaxU = relaxU;
    mi.relaxEquationU = true;   // the gate NAMES a factor, as the device side does
    mi.bounded = true;
    mi.correctedLaplacian = true;
    const FvVectorMatrix refU = cpu::rhoSimple::assembleUEqn(U, mi, m, g, fvp);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(p, fvp, g);

    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dP(p.internal);

    std::vector<scalar> nuBndFlat, rhoBndFlat, psiBndFlat, phiBndFlat;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            nuBndFlat.push_back(nuEffB[pi][i]);
            rhoBndFlat.push_back(rhoB[pi][i]);
            psiBndFlat.push_back(psiB[pi][i]);
            phiBndFlat.push_back(phiBnd[pi][i]);
        }
    }
    nuBndFlat.resize(dm.nBndFaces, nu);
    rhoBndFlat.resize(dm.nBndFaces, 1.0);
    psiBndFlat.resize(dm.nBndFaces, 7.0e-6);
    phiBndFlat.resize(dm.nBndFaces, 0.0);

    DeviceBuffer<scalar> dPhiInt(phiF.internalField), dPhiBnd(phiBndFlat);
    DeviceBuffer<scalar> dRhoCell(rhoC), dRhoBnd(rhoBndFlat);
    DeviceBuffer<scalar> dPsiCell(psiC), dPsiBnd(psiBndFlat);
    DeviceBuffer<scalar> dNuCell(nuEffC), dNuBnd(nuBndFlat);

    gpu::rhoSimple::RhoMomentumInput gmi;
    gmi.phiInt = &dPhiInt;        gmi.phiBnd = &dPhiBnd;
    gmi.rhoCell = &dRhoCell;      gmi.rhoBndFace = &dRhoBnd;
    gmi.nuEffCell = &dNuCell;     gmi.nuEffBndFace = &dNuBnd;
    gmi.relaxEquationU = true;    gmi.relaxU = relaxU;
    gmi.bounded = true;
    gmi.correctedLaplacian = true;
    gpu::MomentumMatrix MU;
    gpu::rhoSimple::assembleUEqn(MU, dm, dbU, dUx, dUy, dUz, gmi);

    // ---- the two masks, each answering its own question --------------------------------------
    std::vector<label> takeU, adjustable;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            takeU.push_back(U.boundary[pi]->assignable() ? 0 : 1);
            // fixesValue() && !isInletOutlet() -- both halves, per adjustPhi. mixedFvPatchField's
            // fixesValue() is TRUE and inletOutlet inherits it, so testing fixesValue alone would mark an
            // inletOutlet OUTLET as fixed outflow and leave adjustPhi nothing to balance against.
            const bool fixed = U.boundary[pi]->fixesValue() && !U.boundary[pi]->isInletOutlet();
            adjustable.push_back(fixed ? 0 : 1);
        }
    }
    takeU.resize(dm.nBndFaces, 0);
    adjustable.resize(dm.nBndFaces, 0);
    DeviceBuffer<label> dTakeU(takeU), dAdjust(adjustable);

    // p NEEDS A REFERENCE only when no p patch fixes a value -- the same condition adjustPhi returns
    // early on. Derived, not asserted; see the header.
    bool pNeedsRef = true;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        if (p.boundary[pi]->fixesValue()) { pNeedsRef = false; break; }
    const label pRefCell = pNeedsRef ? 0 : -1;
    std::printf("  p needs a reference: %s (pRefCell %d)\n", pNeedsRef ? "yes" : "no", (int)pRefCell);

    // ---- the reference ----------------------------------------------------------------------
    cpu::rhoSimple::PressureInput pin;
    pin.rho = &rhoC;   pin.rhoBnd = &rhoB;
    pin.psi = &psiC;   pin.psiBnd = &psiB;
    pin.transonic = transonic;
    pin.relaxP = 0.3;
    pin.relaxPSpecified = true;
    pin.pRefCell = pRefCell;
    pin.pRefValue = 0.0;
    pin.correctedLaplacian = true;
    const cpu::rhoSimple::ConsistentPressureStages rst =
        cpu::rhoSimple::consistentPressurePredictor(refU, U, p, pin, m, g, fvp);
    const FvScalarMatrix refP = cpu::rhoSimple::assemblePcEqn(rst, p, pin, m, g, fvp);

    // ---- the CUDA path ----------------------------------------------------------------------
    gpu::rhoSimple::RhoPressureInput gpin;
    gpin.rhoCell = &dRhoCell;  gpin.rhoBndFace = &dRhoBnd;
    gpin.psiCell = &dPsiCell;  gpin.psiBndFace = &dPsiBnd;
    gpin.transonic = transonic;
    gpin.relaxP = 0.3;
    gpin.relaxPSpecified = true;
    gpin.pRefCell = pRefCell;
    gpin.pRefValue = 0.0;
    gpin.correctedLaplacian = true;
    gpin.takeUAtBoundary = &dTakeU;
    gpin.adjustable = &dAdjust;

    gpu::rhoSimple::ConsistentPressureStages gst;
    gpu::rhoSimple::consistentPressurePredictor(gst, dm, dbU, dbP, MU, dUx, dUy, dUz, dP, gpin);
    gpu::PressureMatrix GP;
    gpu::rhoSimple::assemblePcEqn(GP, gst, dm, dbP, dP, gpin);

    // ---- stage by stage ---------------------------------------------------------------------
    check(gst.transonic == rst.transonic, "both paths took the same branch");
    cmp(gst.rAU.host(),  rst.rAU,  "rAU  = 1/UEqn.A()", 1e-12);
    cmp(gst.rAtU.host(), rst.rAtU, "rAtU = 1/(1/rAU - H1)  [SIMPLEC]", 1e-11);
    cmp(gst.rhorAtU.host(), rst.rhorAtU, "rhorAtU = rho*rAtU", 1e-11);

    {
        const char* h0[3] = {"HbyA0 x (before corr)", "HbyA0 y", "HbyA0 z"};
        const char* h1[3] = {"HbyA  x (after corr)",  "HbyA  y", "HbyA  z"};
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> r0(nC), r1(nC);
            for (label c = 0; c < nC; ++c)
            {
                r0[c] = component(rst.HbyA0[c], k);
                r1[c] = component(rst.HbyA[c], k);
            }
            cmp(gst.HbyA0[k].host(), r0, h0[k], 1e-11);
            cmp(gst.HbyA[k].host(),  r1, h1[k], 1e-11);
        }
    }
    cmp(gst.phiHbyA0Int.host(), rst.phiHbyA0.internal, "phiHbyA BEFORE the branch", 1e-11);
    cmp(gst.phiHbyAInt.host(),  rst.phiHbyA.internal,  "phiHbyA AFTER the branch",  1e-11);
    {
        std::vector<scalar> b0, b1;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                b0.push_back(rst.phiHbyA0.boundary[pi][i]);
                b1.push_back(rst.phiHbyA.boundary[pi][i]);
            }
        }
        std::vector<scalar> g0 = gst.phiHbyA0Bnd.host(), g1 = gst.phiHbyABnd.host();
        g0.resize(b0.size());
        g1.resize(b1.size());
        cmp(g0, b0, "phiHbyA BOUNDARY before branch", 1e-11);
        cmp(g1, b1, "phiHbyA BOUNDARY after branch",  1e-11);
    }
    if (transonic)
    {
        cmp(gst.phidInt.host(), rst.phid.internal, "phid (transonic only)", 1e-11);
    }

    cmp(GP.diag.host(),   refP.diag,   "pcEqn diag",   1e-11);
    cmp(GP.upper.host(),  refP.upper,  "pcEqn upper",  1e-11);
    cmp(GP.lower.host(),  refP.lower,  "pcEqn lower",  1e-11);
    cmp(GP.source.host(), refP.source, "pcEqn source", 1e-10);
    {
        std::vector<scalar> ric, rbc;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                ric.push_back(refP.internalCoeffs[pi][i]);
                rbc.push_back(refP.boundaryCoeffs[pi][i]);
            }
        }
        std::vector<scalar> gic = GP.iC.host(), gbc = GP.bC.host();
        gic.resize(ric.size());
        gbc.resize(rbc.size());
        cmp(gic, ric, "pcEqn internalCoeffs", 1e-11);
        cmp(gbc, rbc, "pcEqn boundaryCoeffs", 1e-11);
    }

    // ---- CONTROL: SIMPLEC is actually doing something ---------------------------------------
    // If H1 came back zero, rAtU would equal rAU, every number above would still agree, and the
    // equation would be SIMPLE wearing SIMPLEC's name.
    {
        scalar d = 0, mg = 0;
        for (label c = 0; c < nC; ++c)
        {
            d = std::fmax(d, std::fabs(rst.rAtU[c] - rst.rAU[c]));
            mg = std::fmax(mg, std::fabs(rst.rAU[c]));
        }
        const scalar r = mg > 0 ? d / mg : d;
        std::printf("  %-58s rel=%.3e\n", "control: rAtU differs from rAU (H1 is non-zero)", (double)r);
        check(r > 1e-6, "the consistent term changes rAU (control)");
    }

    // ---- CONTROL: the SIMPLEC flux correction moves phiHbyA ---------------------------------
    // The other half of the same change. Keeping rAtU while dropping this made the converged velocity
    // WORSE than doing neither in the incompressible lineage, so both are asserted separately.
    {
        scalar d = 0, mg = 0;
        for (std::size_t f = 0; f < rst.phiHbyA.internal.size(); ++f)
        {
            d = std::fmax(d, std::fabs(rst.phiHbyA.internal[f] - rst.phiHbyA0.internal[f]));
            mg = std::fmax(mg, std::fabs(rst.phiHbyA0.internal[f]));
        }
        const scalar r = mg > 0 ? d / mg : d;
        std::printf("  %-58s rel=%.3e\n", "control: the SIMPLEC flux correction moves phiHbyA", (double)r);
        check(r > 1e-9, "phiHbyA carries the (rAtU - rAU) correction (control)");
    }

    // ---- CONTROL: the HbyA correction moves HbyA --------------------------------------------
    {
        scalar d = 0, mg = 0;
        for (label c = 0; c < nC; ++c)
        {
            d = std::fmax(d, std::fabs(rst.HbyA[c].x - rst.HbyA0[c].x));
            mg = std::fmax(mg, std::fabs(rst.HbyA0[c].x));
        }
        const scalar r = mg > 0 ? d / mg : d;
        std::printf("  %-58s rel=%.3e\n", "control: HbyA -= (rAU - rAtU)*grad(p) moves HbyA", (double)r);
        check(r > 1e-9, "HbyA carries its SIMPLEC correction (control)");
    }

    // ---- CONTROL: the branch is a different equation ----------------------------------------
    {
        cpu::rhoSimple::PressureInput other = pin;
        other.transonic = !transonic;
        const cpu::rhoSimple::ConsistentPressureStages ost =
            cpu::rhoSimple::consistentPressurePredictor(refU, U, p, other, m, g, fvp);
        const FvScalarMatrix oP = cpu::rhoSimple::assemblePcEqn(ost, p, other, m, g, fvp);
        scalar dD = 0, mD = 0;
        for (std::size_t c = 0; c < refP.diag.size(); ++c)
        {
            dD = std::fmax(dD, std::fabs(refP.diag[c] - oP.diag[c]));
            mD = std::fmax(mD, std::fabs(refP.diag[c]));
        }
        const scalar r = mD > 0 ? dD / mD : dD;
        std::printf("  %-58s rel=%.3e\n", "control: the OTHER branch is a different matrix", (double)r);
        check(r > 1e-9, "the transonic/subsonic branch changes the equation (control)");
    }

    // ---- CONTROL: `corrected` moves BOTH the coefficients and the source --------------------
    {
        cpu::rhoSimple::PressureInput noC = pin;
        noC.correctedLaplacian = false;
        const cpu::rhoSimple::ConsistentPressureStages cst =
            cpu::rhoSimple::consistentPressurePredictor(refU, U, p, noC, m, g, fvp);
        const FvScalarMatrix cP = cpu::rhoSimple::assemblePcEqn(cst, p, noC, m, g, fvp);
        scalar dD = 0, mD = 0, dS = 0, mS = 0;
        for (std::size_t c = 0; c < refP.diag.size(); ++c)
        {
            dD = std::fmax(dD, std::fabs(refP.diag[c] - cP.diag[c]));
            mD = std::fmax(mD, std::fabs(refP.diag[c]));
            dS = std::fmax(dS, std::fabs(refP.source[c] - cP.source[c]));
            mS = std::fmax(mS, std::fabs(refP.source[c]));
        }
        const scalar rD = mD > 0 ? dD / mD : dD, rS = mS > 0 ? dS / mS : dS;
        std::printf("  %-58s diag=%.3e src=%.3e\n",
                    "control: `corrected` moves coefficients AND source", (double)rD, (double)rS);
        check(rD > 1e-12, "the implicit half of the non-orth correction contributes (control)");
        check(rS > 1e-12, "the explicit half of the non-orth correction contributes (control)");
    }

    // ---- refusals ---------------------------------------------------------------------------
    {
        struct { const char* what; int which; } cases[] = {
            {"MRF is refused on the CUDA path",        0},
            {"an unported fvOptions is refused",       1},
            {"fixedFluxPressure is refused",           2},
            {"a mesh with coupled patches is refused", 3},
        };
        for (const auto& cse : cases)
        {
            gpu::rhoSimple::RhoPressureInput bad = gpin;
            if (cse.which == 0) bad.hasMRF = true;
            if (cse.which == 1) bad.hasFvOptions = true;
            if (cse.which == 2) bad.hasFixedFluxPressure = true;
            if (cse.which == 3) bad.hasCoupledPatches = true;
            gpu::rhoSimple::ConsistentPressureStages sb;
            bool threw = false;
            try
            {
                gpu::rhoSimple::consistentPressurePredictor(
                    sb, dm, dbU, dbP, MU, dUx, dUy, dUz, dP, bad);
            }
            catch (const std::runtime_error&)
            {
                threw = true;
            }
            check(threw, cse.what);
        }
        {
            gpu::rhoSimple::RhoPressureInput bad = gpin;
            bad.takeUAtBoundary = nullptr;
            gpu::rhoSimple::ConsistentPressureStages sb;
            bool threw = false;
            try
            {
                gpu::rhoSimple::consistentPressurePredictor(
                    sb, dm, dbU, dbP, MU, dUx, dUy, dUz, dP, bad);
            }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "a missing constrainHbyA `assignable` mask is refused");
        }
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
