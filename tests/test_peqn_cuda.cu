// CUDA pEqn against the _cpp reference, stage by stage.
//
// The reference is validated against OpenFOAM's own dumps (test_peqn_cpp: rAU/HbyA vs ops.dat, the
// Laplacian vs peqn.dat, setReference asserted to be fvMatrix.C:1011-1023 exactly). This closes
// OpenFOAM -> _cpp -> CUDA for the whole pressure corrector.
//
// Compared stage by stage rather than end to end, because pEqn.H has seven places it can be wrong and one
// number over `p` cannot say which. In particular constrainHbyA and adjustPhi branch on DIFFERENT patch
// questions -- `assignable` and `fixesValue` -- and conflating them is a silent error, so both masks are
// built here from the reference's own predicates and both stages are checked.
//
// Run: test_peqn_cuda <caseDir> <timeDir> [laminar]
#include "primitive_mesh.cuh"
#include "solution_directions.cuh"   // fvMatrix<Type>::H()'s validComponents mask
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"
#include "linearViscousStress_cpp.cuh"
#include "UEqn.cuh"
#include "pEqn.cuh"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

static void cmp(const std::vector<scalar>& gpu, const std::vector<scalar>& ref,
                const char* nm, scalar tol)
{
    if (gpu.size() != ref.size())
    {
        std::printf("  %-32s SIZE %zu vs %zu  FAIL\n", nm, gpu.size(), ref.size());
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
    std::printf("  %-32s n=%6zu rel=%.3e  %s\n", nm, ref.size(), rel, ok ? "OK" : "FAIL");
}

static void check(bool ok, const char* what)
{
    std::printf("  %-54s %s\n", what, ok ? "OK" : "FAIL");
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
    const scalar nu = 1e-5;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
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
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
                phiBnd[pi] = b.values;
    }

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

    std::printf("test_peqn_cuda:  (%s: nuEff %s)\n", turbulent ? "TURBULENT" : "laminar",
                turbulent ? "varies" : "is constant");

    // ---- reference UEqn + pressure stages ---------------------------------------------------
    const scalar relaxU = 0.7;
    cpu::MomentumInput mi;
    mi.phi = &phiF.internalField; mi.phiBnd = &phiBnd;
    mi.nuEff = &nuEffC;           mi.nuEffBnd = &nuEffB;
    mi.relaxU = relaxU;
    const FvVectorMatrix refU = cpu::assembleUEqn(U, mi, m, g, fvp);

    cpu::PressureInput rpin;
    rpin.pRefCell = -1;
    rpin.correctedLaplacian = true;   // exercised on both paths -- see the control below
    rpin.consistent = true;           // SIMPLEC -- exercised on both paths, with its own control
    const cpu::PressureStages rst = cpu::pressurePredictor(refU, U, p, rpin, m, g, fvp);
    const FvScalarMatrix refP = cpu::assemblePEqn(rst, p, rpin, m, g, fvp);

    // ---- CUDA -------------------------------------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(p, fvp, g);

    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz);

    const SurfaceScalarField nuFace = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
    DeviceBuffer<scalar> dP(p.internal);
    DeviceBuffer<scalar> dPhiInt(phiF.internalField);
    DeviceBuffer<scalar> dPhiBnd(flatten(phiBnd, dm.nBndFaces, 0.0));
    DeviceBuffer<scalar> dNuCell(nuEffC), dNuFace(nuFace.internal);
    DeviceBuffer<scalar> dNuBnd(flatten(nuEffB, dm.nBndFaces, nu));

    gpu::MomentumInput gmi;
    gmi.phiInt = &dPhiInt;     gmi.phiBnd = &dPhiBnd;
    gmi.nuEffCell = &dNuCell;  gmi.nuEffFace = &dNuFace;  gmi.nuEffBndFace = &dNuBnd;
    gmi.relaxU = relaxU;
    gpu::MomentumMatrix MU;
    gpu::assembleUEqn(MU, dm, dbU, dUx, dUy, dUz, gmi);

    // The two masks come from the reference's OWN predicates, so a change to either semantic shows up
    // here rather than being silently re-derived on the device.
    std::vector<label> takeU, adjustable;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            takeU.push_back(U.boundary[pi]->assignable() ? 0 : 1);
            adjustable.push_back(U.boundary[pi]->fixesValue() ? 0 : 1);
        }
    takeU.resize(dm.nBndFaces, 0);
    adjustable.resize(dm.nBndFaces, 0);
    DeviceBuffer<label> dTakeU(takeU), dAdjust(adjustable);
    // Control: the two masks must actually DIFFER on this case, or checking both proves nothing.
    bool masksDiffer = false;
    for (std::size_t i = 0; i < takeU.size(); ++i) if (takeU[i] != adjustable[i]) { masksDiffer = true; break; }

    gpu::PressureInput gpin;
    gpin.pRefCell = -1;
    gpin.correctedLaplacian = true;
    gpin.consistent = true;
    gpin.takeUAtBoundary = &dTakeU;
    gpin.adjustable = &dAdjust;
    // The SAME validComponents mask the host reference derives inside matrixH from the patch list
    // (fvMatrix<Type>::H()'s closing block). This input is built BY HAND, which is exactly the harness
    // hazard the driver's own comment warns about: without this the device arm leaves H_z live where the
    // reference zeroes it, and `HbyA z` reads 6.885e-03 against a reference of identically zero.
    {
        const SolutionDirections sd = solutionDirections(fvp);
        for (int cmpt = 0; cmpt < 3; ++cmpt) gpin.solutionD[cmpt] = sd.d[cmpt];
    }

    gpu::PressureStages gst;
    gpu::pressurePredictor(gst, dm, dbU, MU, dUx, dUy, dUz, gpin, &dbP, &dP);

    // ---- stages 1-3 -------------------------------------------------------------------------
    std::printf("  -- stages 1-3: rAU, HbyA (constrained), phiHbyA\n");
    cmp(gst.rAU.host(), rst.rAU, "rAU", 1e-13);

    const char* hn[3] = {"HbyA x", "HbyA y", "HbyA z"};
    for (int k = 0; k < 3; ++k)
    {
        std::vector<scalar> r(nC);
        for (label c = 0; c < nC; ++c) r[c] = component(rst.HbyA[c], k);
        cmp(gst.HbyA[k].host(), r, hn[k], 1e-12);
    }
    cmp(gst.phiHbyAInt.host(), rst.phiHbyA.internal, "phiHbyA internal", 1e-12);
    cmp(gst.phiHbyABnd.host(), flatten(rst.phiHbyA.boundary, dm.nBndFaces, 0.0),
        "phiHbyA boundary", 1e-12);
    check(!gst.phiAdjusted, "adjustPhi did NOT run (p needs no reference)");

    // ---- stage 4-5 --------------------------------------------------------------------------
    std::printf("  -- stages 4-5: laplacian(rAU,p) == div(phiHbyA), setReference\n");
    // rAtU, not rAU: pEqn.H's laplacian diffusivity is rAtU, which SIMPLEC makes different. Building it
    // from rAU here compared the device's rAtU laplacian against a reference rAU one -- a harness
    // disagreement, not a solver one, and it showed up as 6.4e-01 on every coefficient.
    DeviceBuffer<scalar> dRAUface(fvc::interpolate(rst.rAtU, m, g, fvp).internal);
    gpu::PressureMatrix P;
    gpu::assemblePEqn(P, gst, dm, dbP, dRAUface, gpin, &dP);
    cmp(P.upper.host(),  refP.upper,  "pEqn upper",  1e-13);
    cmp(P.lower.host(),  refP.lower,  "pEqn lower",  1e-13);
    cmp(P.diag.host(),   refP.diag,   "pEqn diag",   1e-13);
    cmp(P.source.host(), refP.source, "pEqn source", 1e-11);
    {
        std::vector<scalar> ric, rbc;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            { ric.push_back(refP.internalCoeffs[pi][i]); rbc.push_back(refP.boundaryCoeffs[pi][i]); }
        std::vector<scalar> gic = P.iC.host(), gbc = P.bC.host();
        gic.resize(ric.size()); gbc.resize(rbc.size());
        cmp(gic, ric, "pEqn internalCoeffs", 1e-13);
        cmp(gbc, rbc, "pEqn boundaryCoeffs", 1e-13);
    }

    // setReference: a second assembly with a reference cell must differ exactly as the reference does.
    {
        cpu::PressureInput r2 = rpin; r2.pRefCell = 11; r2.pRefValue = 1.75;
        const FvScalarMatrix refR = cpu::assemblePEqn(rst, p, r2, m, g, fvp);
        gpu::PressureInput g2 = gpin; g2.pRefCell = 11; g2.pRefValue = 1.75;
        g2.adjustable = &dAdjust;
        gpu::PressureMatrix P2;
        gpu::assemblePEqn(P2, gst, dm, dbP, dRAUface, g2, &dP);
        cmp(P2.diag.host(),   refR.diag,   "setReference diag",   1e-13);
        cmp(P2.source.host(), refR.source, "setReference source", 1e-11);
        check(refR.diag[11] != refP.diag[11], "the reference cell really changed (control)");
    }

    // ---- stage 7: phi = phiHbyA - pEqn.flux() -----------------------------------------------
    std::printf("  -- stage 7: flux correction\n");
    {
        DeviceBuffer<scalar> dp(p.internal), phiI, phiB;
        gpu::correctFlux(phiI, phiB, gst, P, dm, dbP, dp);
        const SurfaceScalarField ref = cpu::correctFlux(rst, refP, p.internal, m, fvp);
        cmp(phiI.host(), ref.internal, "phi internal", 1e-12);
        cmp(phiB.host(), flatten(ref.boundary, dm.nBndFaces, 0.0), "phi boundary", 1e-12);
    }

    // ---- p.relax() and the momentum corrector -----------------------------------------------
    std::printf("  -- p.relax(), U = HbyA - rAU*grad(p)\n");
    {
        std::vector<scalar> pPrev(nC, 1.0);
        std::vector<scalar> rp = p.internal;
        cpu::relaxField(rp, pPrev, 0.3);
        DeviceBuffer<scalar> dp(p.internal), dprev(pPrev);
        gpu::relaxField(dp, dprev, 0.3);
        cmp(dp.host(), rp, "p.relax(alpha=0.3)", 1e-14);
    }
    {
        const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);
        const std::vector<vector> ref = cpu::correctVelocity(rst, p, m, g, fvp);
        std::vector<scalar> gx(nC), gy(nC), gz(nC);
        for (label c = 0; c < nC; ++c) { gx[c] = gradP[c].x; gy[c] = gradP[c].y; gz[c] = gradP[c].z; }
        DeviceBuffer<scalar> dGx(gx), dGy(gy), dGz(gz), oUx, oUy, oUz;
        gpu::correctVelocity(oUx, oUy, oUz, gst, dGx, dGy, dGz);
        const char* un[3] = {"U corrector x", "U corrector y", "U corrector z"};
        const DeviceBuffer<scalar>* o[3] = {&oUx, &oUy, &oUz};
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> r(nC);
            for (label c = 0; c < nC; ++c) r[c] = component(ref[c], k);
            cmp(o[k]->host(), r, un[k], 1e-12);
        }
    }

    // CONTROL: `corrected` must actually change the pressure equation, or comparing it proves nothing.
    // Both halves again -- the implicit one lands on the coefficients, the explicit one on the source.
    {
        cpu::PressureInput noC = rpin; noC.correctedLaplacian = false;
        const FvScalarMatrix refNoC = cpu::assemblePEqn(rst, p, noC, m, g, fvp);
        scalar dD = 0, mD = 0, dS = 0, mS = 0;
        for (std::size_t c = 0; c < refP.diag.size(); ++c)
        {
            dD = std::fmax(dD, std::fabs(refP.diag[c] - refNoC.diag[c]));
            mD = std::fmax(mD, std::fabs(refP.diag[c]));
            dS = std::fmax(dS, std::fabs(refP.source[c] - refNoC.source[c]));
            mS = std::fmax(mS, std::fabs(refP.source[c]));
        }
        const scalar rD = mD > 0 ? dD / mD : dD, rS = mS > 0 ? dS / mS : dS;
        std::printf("  %-54s rel=%.3e\n", "control: `corrected` moves the p coefficients", rD);
        std::printf("  %-54s rel=%.3e\n", "control: `corrected` moves the p source", rS);
        check(rD > 1e-12, "the implicit half of the p non-orth correction contributes (control)");
        check(rS > 1e-12, "the explicit half of the p non-orth correction contributes (control)");
    }

    // A missing pressure field with the correction on must be REFUSED, not silently skipped: grad(p) is
    // required to evaluate it, and a nullptr that quietly meant "no correction" would reproduce exactly
    // the class of defect this port exists to avoid.
    {
        gpu::PressureMatrix Pn;
        bool threw = false;
        try { gpu::assemblePEqn(Pn, gst, dm, dbP, dRAUface, gpin, nullptr); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "corrected laplacian without p is refused on the CUDA path");
    }

    // ---- masks and refusals -----------------------------------------------------------------
    std::printf("  -- masks and refusals\n");
    check(masksDiffer, "assignable and fixesValue masks DIFFER on this case (control)");
    // `consistent` is no longer here: SIMPLEC is implemented and is exercised above. fixedFluxPressure
    // takes its place. fixedFluxPressure left too: deviceConstrainPressure is wired at pEqn.H:21,
    // and a never-updated patch refuses on the HOST at coefficient time (requireUpdated).
    const char* names[2] = {"MRF", "fvOptions"};
    for (int which = 0; which < 2; ++which)
    {
        gpu::PressureInput bad = gpin;
        if (which == 0) bad.hasMRF = true;
        else bad.hasFvOptions = true;
        gpu::PressureStages s2;
        bool threw = false;
        try { gpu::pressurePredictor(s2, dm, dbU, MU, dUx, dUy, dUz, bad, &dbP, &dP); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, (std::string(names[which]) + " is refused on the CUDA path").c_str());
    }

    // SIMPLEC without the pressure field must be REFUSED, not silently downgraded to plain SIMPLE:
    // rAtU's corrections need snGrad(p) and grad(p), and a nullptr quietly meaning "skip them" would
    // solve SIMPLE while the case asked for SIMPLEC.
    {
        gpu::PressureStages s3;
        bool threw = false;
        try { gpu::pressurePredictor(s3, dm, dbU, MU, dUx, dUy, dUz, gpin, nullptr, nullptr); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "SIMPLEC without p is refused on the CUDA path");
    }

    // CONTROL: SIMPLEC must actually change the answer, or comparing it proves nothing. rAtU differs
    // from rAU by the off-diagonal row sum, so this is a large effect, not a subtle one.
    {
        cpu::PressureInput noC = rpin; noC.consistent = false;
        const cpu::PressureStages r2 = cpu::pressurePredictor(refU, U, p, noC, m, g, fvp);
        scalar dR = 0, mR = 0, dF = 0, mF = 0;
        for (std::size_t c = 0; c < rst.rAtU.size(); ++c)
        { dR = std::fmax(dR, std::fabs(rst.rAtU[c] - r2.rAtU[c])); mR = std::fmax(mR, std::fabs(rst.rAtU[c])); }
        for (std::size_t f = 0; f < rst.phiHbyA.internal.size(); ++f)
        { dF = std::fmax(dF, std::fabs(rst.phiHbyA.internal[f] - r2.phiHbyA.internal[f]));
          mF = std::fmax(mF, std::fabs(rst.phiHbyA.internal[f])); }
        std::printf("  %-54s rel=%.3e\n", "control: SIMPLEC changes rAtU",   mR > 0 ? dR / mR : dR);
        std::printf("  %-54s rel=%.3e\n", "control: SIMPLEC changes phiHbyA", mF > 0 ? dF / mF : dF);
        check(dR > 0.0, "rAtU differs from rAU under SIMPLEC (control)");
        check(dF > 0.0, "the SIMPLEC flux correction contributes (control)");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
