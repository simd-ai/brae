// CUDA vs the _cpp REFERENCE, stage by stage.
//
// Every existing GPU test compares the device against CPU code written inline in that same test. This one
// compares it against the _cpp reference components, which are themselves validated against OpenFOAM's own
// dumps (test_peqn_cpp, test_ueqn_cpp, test_simple_step_cpp). That closes the chain the rebuild is built
// around:
//
//     OpenFOAM  ->  _cpp reference  ->  CUDA
//
// and it is compared at STAGE granularity, so a disagreement names the stage rather than the iteration.
// Debugging from a final residual is what this replaces.
//
// Run: test_gpu_vs_cpp <caseDir> <timeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_simple.cuh"
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"
#include "linearViscousStress_cpp.cuh"
#include "device_boundary.cuh"
#include "device_divdevreff.cuh"
#include "device_kepsilon.cuh"
#include "k_epsilon.cuh"

#include <cmath>
#include <filesystem>
#include <cstdio>
#include <string>
#include <vector>
#include "solution_directions.cuh"   // fvMatrix<Type>::H()'s validComponents mask

using namespace brae;

static int g_fails = 0;

static void cmp(const std::vector<scalar>& gpu, const std::vector<scalar>& ref,
                const char* nm, scalar tol)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(gpu[i] - ref[i]));
        mg = std::fmax(mg, std::fabs(ref[i]));
    }
    const scalar rel = mg > 0 ? mx / mg : mx;
    const bool ok = rel <= tol;
    if (!ok) ++g_fails;
    std::printf("  %-30s n=%6zu rel=%.3e  %s\n", nm, ref.size(), rel, ok ? "OK" : "FAIL");
}

static std::vector<scalar> flattenB(const std::vector<std::vector<scalar>>& b, label n)
{
    std::vector<scalar> f;
    for (const auto& v : b) for (scalar x : v) f.push_back(x);
    f.resize(n, 0.0);
    return f;
}

static void check(bool ok, const char* what)
{
    std::printf("  %-52s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];
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

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    std::printf("test_gpu_vs_cpp:\n");

    // ---- the _cpp reference UEqn and its pressure stages ------------------------------------
    //
    // nuEff VARIES when the case has a nut field. That matters: on a laminar case nuEff is constant, and a
    // kernel that mishandles a per-face diffusivity -- or takes the owner cell's value on a boundary face
    // instead of the patch value -- still agrees perfectly. Running this on a turbulent case as well is
    // what makes the divDevReff and Laplacian comparisons load-bearing.
    std::vector<scalar> nuEffC(nC, nu);
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nuEffB[pi].assign(fvp[pi].size, nu);

    const bool turbulent = std::filesystem::exists(caseDir + "/" + t + "/nut")
                        || std::filesystem::exists(caseDir + "/" + t + "/nut.gz");
    GeometricField<scalar> nutF;
    if (turbulent)
    {
        nutF = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), fvp, nC);
        nutF.evaluateBoundary();
        for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nutF.internal[c];
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& nb = nutF.boundary[pi]->value();
            for (label i = 0; i < fvp[pi].size; ++i) nuEffB[pi][i] = nu + nb[i];
        }
    }
    std::printf("  (%s case: nuEff %s)\n", turbulent ? "TURBULENT" : "laminar",
                turbulent ? "varies per cell and per boundary face" : "is constant");

    cpu::MomentumInput mi;
    mi.phi = &phiF.internalField; mi.phiBnd = &phiBnd;
    mi.nuEff = &nuEffC;           mi.nuEffBnd = &nuEffB;
    mi.relaxU = 1.0;
    const FvVectorMatrix UEqn = cpu::assembleUEqn(U, mi, m, g, fvp);

    cpu::PressureInput pin;
    pin.pRefCell = -1;
    const cpu::PressureStages st = cpu::pressurePredictor(UEqn, U, p, pin, m, g, fvp);
    const FvScalarMatrix pEqn = cpu::assemblePEqn(st, p, pin, m, g, fvp);

    // ---- stage: rAU = 1/A(), via deviceReciprocalV ------------------------------------------
    // The device takes the already-folded diagonal (D = diag + cmptAv(internalCoeffs)) and divides by V.
    // Build that diagonal from the reference matrix so the comparison isolates the KERNEL, not the
    // assembly that feeds it -- otherwise a disagreement could come from either and name neither.
    {
        std::vector<scalar> D = UEqn.diag;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                D[fvp[pi].faceCells[i]] += cmptAv(UEqn.internalCoeffs[pi][i]);

        DeviceBuffer<scalar> dD, drAU;
        dD.copyFrom(D);
        deviceReciprocalV(dm, dD, drAU);
        cmp(drAU.host(), st.rAU, "rAU  (deviceReciprocalV)", 1e-14);
    }

    // ---- stage: the pressure Laplacian coefficients -----------------------------------------
    {
        const SurfaceScalarField rAUf = fvc::interpolate(st.rAU, m, g, fvp);
        DeviceBuffer<scalar> dgamma;
        dgamma.copyFrom(rAUf.internal);
        DeviceBuffer<scalar> dDiag, dUp, dLo;
        deviceLaplacianCoeffs(dm, dgamma, dDiag, dUp, dLo);
        cmp(dUp.host(),   pEqn.upper, "pEqn upper (deviceLaplacian)", 1e-13);
        cmp(dLo.host(),   pEqn.lower, "pEqn lower (deviceLaplacian)", 1e-13);
        cmp(dDiag.host(), pEqn.diag,  "pEqn diag  (deviceLaplacian)", 1e-13);
    }

    // ---- stage: pEqn.flux() internal --------------------------------------------------------
    // faceH(p) = upper*p[nei] - lower*p[own]; the same expression the reference matrixFlux uses.
    {
        DeviceBuffer<scalar> dUp, dLo, dDiag, dp, dflux;
        dUp.copyFrom(pEqn.upper); dLo.copyFrom(pEqn.lower);
        dDiag.copyFrom(pEqn.diag); dp.copyFrom(p.internal);
        DeviceLduView A{};
        A.nCells = nC; A.nInternalFaces = m.nInternalFaces();
        A.diag = dDiag.data(); A.upper = dUp.data(); A.lower = dLo.data();
        A.owner = dm.owner.data(); A.nei = dm.nei.data();
        A.ownerStart = dm.ownerStart.data();
        A.losort = dm.losort.data(); A.losortStart = dm.losortStart.data();
        deviceMatrixFluxInternal(A, dp, dflux);

        const SurfaceScalarField ref = matrixFlux(pEqn, p.internal, m, fvp);
        cmp(dflux.host(), ref.internal, "pEqn.flux() (deviceMatrixFlux)", 1e-13);
    }

    // ---- stage: setReference ----------------------------------------------------------------
    // fvMatrix.C:1011-1023 DOUBLES the diagonal. Assert the kernel does the same thing the reference
    // does, on the same cell, and touches nothing else.
    {
        const label refCell = 7;
        const scalar refValue = 2.25;
        DeviceBuffer<scalar> dDiag, dSrc;
        dDiag.copyFrom(pEqn.diag); dSrc.copyFrom(pEqn.source);
        deviceSetReference(dDiag, dSrc, refCell, refValue);

        std::vector<scalar> rDiag = pEqn.diag, rSrc = pEqn.source;
        rSrc[refCell] += rDiag[refCell] * refValue;
        rDiag[refCell] += rDiag[refCell];

        cmp(dDiag.host(), rDiag, "setReference diag", 1e-15);
        cmp(dSrc.host(),  rSrc,  "setReference source", 1e-15);
        // Control: the reference cell must actually have changed, or both sides agree trivially.
        check(rDiag[refCell] != pEqn.diag[refCell], "the reference cell really changed (control)");
    }

    // =========================== MOMENTUM STAGES ============================================
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz);

    std::printf("  -- momentum stages\n");

    // fvm::div(phi, U): the UPWIND implicit coefficients. This is the operator whose implicit weights hid
    // brae's LUST defect, so it is compared coefficient by coefficient rather than through a residual.
    {
        DeviceBuffer<scalar> dphi(phiF.internalField);
        DeviceBuffer<scalar> dDiag, dUp, dLo;
        deviceDivUpwindCoeffs(dm, dphi, dDiag, dUp, dLo);
        const FvVectorMatrix divRef = fvm::div(phiF.internalField, phiBnd, U, m, fvp);
        cmp(dUp.host(),   divRef.upper, "div(phi,U) upper", 1e-13);
        cmp(dLo.host(),   divRef.lower, "div(phi,U) lower", 1e-13);
        cmp(dDiag.host(), divRef.diag,  "div(phi,U) diag",  1e-13);
    }

    // turbulence->divDevReff(U), explicit half. The device returns the EXTENSIVE source V*div(sigma);
    // the reference returns -div(sigma) per volume, so the identity is  dev = -V * ref.
    {
        // Flatten the boundary nuEff in the device's boundary-face order (patch by patch), so the wall
        // value really is the wall value -- not the owner cell's.
        std::vector<scalar> nuBndFlat;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i) nuBndFlat.push_back(nuEffB[pi][i]);
        nuBndFlat.resize(dm.nBndFaces, nu);
        DeviceBuffer<scalar> nuCell(nuEffC);
        DeviceBuffer<scalar> nuBnd(nuBndFlat);
        DeviceBuffer<scalar> sx, sy, sz;
        deviceDivDevReff(dm, dbU, dUx, dUy, dUz, nuCell, nuBnd, sx, sy, sz);

        const std::vector<vector> ref = cpu::divDevReffExplicit(U, nuEffC, nuEffB, m, g, fvp);
        const std::vector<scalar>& V = g.V();
        std::vector<scalar> rx(nC), ry(nC), rz(nC);
        for (label c = 0; c < nC; ++c)
        { rx[c] = -V[c]*ref[c].x; ry[c] = -V[c]*ref[c].y; rz[c] = -V[c]*ref[c].z; }
        cmp(sx.host(), rx, "divDevReff source x", 1e-11);
        cmp(sy.host(), ry, "divDevReff source y", 1e-11);
        cmp(sz.host(), rz, "divDevReff source z", 1e-11);
    }

    // fvMatrix::H(), per component -- the numerator of HbyA.
    {
        std::vector<scalar> D = UEqn.diag;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                D[fvp[pi].faceCells[i]] += cmptAv(UEqn.internalCoeffs[pi][i]);
        DeviceBuffer<scalar> dDiag(D), dUp(UEqn.upper), dLo(UEqn.lower);
        DeviceLduView A{};
        A.nCells = nC; A.nInternalFaces = m.nInternalFaces();
        A.diag = dDiag.data(); A.upper = dUp.data(); A.lower = dLo.data();
        A.owner = dm.owner.data(); A.nei = dm.nei.data();
        A.ownerStart = dm.ownerStart.data();
        A.losort = dm.losort.data(); A.losortStart = dm.losortStart.data();

        // The same directions matrixH applies inside itself, so the device arm can be given them too.
        const SolutionDirections solutionD = solutionDirections(fvp);
        const std::vector<vector> Href = matrixH(UEqn, U, m, g, fvp);
        const char* nm[3] = {"H(U) x", "H(U) y", "H(U) z"};
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> psiK(nC), srcK(nC), bdDiagK, bdSrcK, ref(nC);
            for (label c = 0; c < nC; ++c)
            {
                psiK[c] = component(U.internal[c], k);
                srcK[c] = component(UEqn.source[c], k);
                ref[c]  = component(Href[c], k);
            }
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    const vector ic = UEqn.internalCoeffs[pi][i];
                    bdDiagK.push_back(cmptAv(ic) - component(ic, k));
                    bdSrcK.push_back(component(UEqn.boundaryCoeffs[pi][i], k));
                }
            DeviceBuffer<scalar> dPsi(psiK), dSrc(srcK), dBd(bdDiagK), dBs(bdSrcK), dH;
            // The SAME validComponents mask the host matrixH derives from the patch list
            // (fvMatrix<Type>::H()'s closing block). Both arms must apply it or this comparison stops
            // being between two implementations of one function: on a 2D fixture the reference would be
            // identically zero, and cmp's relative check would degenerate to an absolute one on the
            // device's own round-off (measured 3.99e-11 on matrixDumpAsym against a 1e-11 tolerance).
            deviceMatrixH(A, dm, dPsi, dSrc, dBd, dBs, dH, solutionD.valid(k));
            cmp(dH.host(), ref, nm[k], 1e-11);
        }
    }

    // fvc::flux(HbyA) internal -- phiHbyA.
    {
        std::vector<scalar> hx(nC), hy(nC), hz(nC);
        for (label c = 0; c < nC; ++c)
        { hx[c] = st.HbyA[c].x; hy[c] = st.HbyA[c].y; hz[c] = st.HbyA[c].z; }
        DeviceBuffer<scalar> dHx(hx), dHy(hy), dHz(hz), dPhi;
        deviceVectorFlux(dm, dHx, dHy, dHz, dPhi);
        cmp(dPhi.host(), st.phiHbyA.internal, "phiHbyA (deviceVectorFlux)", 1e-12);
    }

    // U = HbyA - rAU*grad(p), the momentum corrector.
    {
        const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);
        const std::vector<vector> ref = cpu::correctVelocity(st, p, m, g, fvp);
        std::vector<scalar> hx(nC), gx(nC), rx(nC);
        for (label c = 0; c < nC; ++c) { hx[c] = st.HbyA[c].x; gx[c] = gradP[c].x; rx[c] = ref[c].x; }
        DeviceBuffer<scalar> dH(hx), dRAU(st.rAU), dG(gx), dU;
        deviceCorrector(dH, dRAU, dG, dU);
        cmp(dU.host(), rx, "U corrector x (deviceCorrector)", 1e-13);
    }

    // =========================== TURBULENCE STAGES ==========================================
    std::printf("  -- turbulence stages\n");
    {
        // GbyNu = gradU && devTwoSymm(gradU) -- the production term.
        DeviceBuffer<scalar> dG;
        deviceGbyNu(dm, dbU, dUx, dUy, dUz, dG);
        const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, fvp);
        cmp(dG.host(), kepsilon::GbyNu(gradU), "GbyNu (deviceGbyNu)", 1e-11);
    }
    {
        // nut = Cmu k^2/epsilon. Uses this case's own k/epsilon if present, else a synthetic pair -- the
        // kernel is the same either way and a synthetic field still exercises every cell.
        std::vector<scalar> kk(nC), ee(nC);
        if (turbulent && std::filesystem::exists(caseDir + "/" + t + "/k"))
        {
            const GeometricField<scalar> kf =
                buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/k"), fvp, nC);
            const GeometricField<scalar> ef =
                buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/epsilon"), fvp, nC);
            kk = kf.internal; ee = ef.internal;
        }
        else
        {
            for (label c = 0; c < nC; ++c) { kk[c] = 0.1 + 0.01*(c % 17); ee[c] = 1.0 + 0.05*(c % 23); }
        }
        DeviceBuffer<scalar> dk(kk), de(ee), dnut;
        deviceNut(dk, de, dnut);
        cmp(dnut.host(), kepsilon::nut(kk, ee), "nut = Cmu k^2/eps (deviceNut)", 1e-14);
    }

    // ---- stage: fvc::div(phi) -- the `bounded` term's ingredient ---------------------------
    // -fvm::Sp(fvc::div(phi),U) is built from this. The CUDA bounded term measured as a no-op against the
    // reference, and the kernel sequence matches the existing GPU driver exactly, so the divergence itself
    // is the thing to check.
    {
        DeviceBuffer<scalar> dPhiI(phiF.internalField);
        DeviceBuffer<scalar> dPhiB(flattenB(phiBnd, dm.nBndFaces));
        DeviceBuffer<scalar> dDiv;
        deviceDiv(dm, dPhiI, dPhiB, dDiv);

        SurfaceScalarField phis;
        phis.internal = phiF.internalField;
        phis.boundary = phiBnd;
        const std::vector<scalar> ref = fvc::div(phis, m, g, fvp);

        // Normalised against the FLUX scale, not against div(phi)'s own maximum. div(phi) IS the
        // near-total cancellation of a cell's face fluxes -- that is what convergence means -- so on a
        // converged field its maximum is round-off-sized and dividing by it inflates a 4e-12 absolute
        // difference into 2.8e-09. The quantity the bounded term actually uses is V*div(phi), and the
        // scale it should be judged against is the face flux it is built from.
        const std::vector<scalar> got = dDiv.host();
        const std::vector<scalar>& V = g.V();
        scalar mxAbs = 0, phiScale = 0, divMax = 0;
        for (std::size_t c = 0; c < ref.size(); ++c)
        {
            mxAbs = std::fmax(mxAbs, std::fabs((got[c] - ref[c]) * V[c]));
            divMax = std::fmax(divMax, std::fabs(ref[c]));
        }
        for (scalar v : phiF.internalField) phiScale = std::fmax(phiScale, std::fabs(v));
        const scalar rel = phiScale > 0 ? mxAbs / phiScale : mxAbs;
        const bool ok = rel <= 1e-12;
        if (!ok) ++g_fails;
        std::printf("  %-30s n=%6zu rel=%.3e  %s   (max|div(phi)|=%.2e, max|phi|=%.2e)\n",
                    "V*div(phi) vs max|phi|", ref.size(), rel, ok ? "OK" : "FAIL", divMax, phiScale);
    }

    // ---- stage: fvc::grad(p) -- deviceGaussGrad vs the host gaussGrad -----------------------
    // The DRIVER computes grad(p) with deviceGaussGrad; the earlier corrector check supplied gradP from
    // the host, so this kernel was never compared. It runs twice per iteration (momentum predictor and
    // velocity corrector), so a systematic bias here biases every field every iteration.
    {
        const DeviceBoundary dbPloc = buildDeviceBoundary(p, fvp, g);
        DeviceBuffer<scalar> dp(p.internal), pb, gx2, gy2, gz2;
        deviceBCValue(dbPloc, dp, pb);
        deviceGaussGrad(dm, dp, pb, gx2, gy2, gz2);

        const std::vector<vector> ref = fvc::gaussGrad(p, m, g, fvp);
        std::vector<scalar> rx(nC), ry(nC), rz(nC);
        for (label c = 0; c < nC; ++c) { rx[c] = ref[c].x; ry[c] = ref[c].y; rz[c] = ref[c].z; }
        cmp(gx2.host(), rx, "grad(p) x (deviceGaussGrad)", 1e-12);
        cmp(gy2.host(), ry, "grad(p) y (deviceGaussGrad)", 1e-12);
        cmp(gz2.host(), rz, "grad(p) z (deviceGaussGrad)", 1e-12);
    }

    // ---- control: the comparison can detect a difference ------------------------------------
    // Perturb one reference value and require cmp to report a failure. Without this every OK above is
    // only evidence that the comparator ran.
    {
        std::vector<scalar> a = pEqn.diag, b = pEqn.diag;
        b[0] *= 1.001;
        scalar mx = 0, mg = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            mx = std::fmax(mx, std::fabs(a[i] - b[i]));
            mg = std::fmax(mg, std::fabs(b[i]));
        }
        check((mg > 0 ? mx / mg : mx) > 1e-13, "a 0.1% perturbation is detected (control)");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
