// CUDA UEqn against the _cpp reference, field by field.
//
// The reference is itself validated against OpenFOAM's own momentum dump (test_ueqn_cpp), so this closes
// OpenFOAM -> _cpp -> CUDA for the whole momentum assembly rather than for individual kernels.
//
// The matrix is compared in its DECOMPOSED form -- diag, upper, lower, the three sources, and every
// boundary coefficient on every patch -- because a folded or fused comparison collapses convection,
// diffusion, the explicit stress, the boundary coefficients and relaxation into one number that cannot
// say which of them is wrong.
//
// Run on both a laminar and a turbulent case: with constant nuEff a kernel that mishandles a per-face
// diffusivity, or reads the owner cell's value on a wall instead of the patch value, still agrees.
//
// Run: test_ueqn_cuda <caseDir> <timeDir> [laminar]
//
// `laminar` forces nuEff = nu everywhere even when the case ships a nut field. Both fixtures here DO ship
// one, so without the flag every run would exercise only the varying-viscosity path -- and the constant
// one is what the OpenFOAM momentum dump was built with, so it is the arithmetic the reference itself was
// proven on. Both are worth covering, and the flag is what makes the difference visible in the ctest name
// rather than hidden in a directory listing.
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
#include "UEqn_cpp.cuh"
#include "UEqn.cuh"
#include "linearViscousStress_cpp.cuh"   // effectiveFaceViscosity -- the same face-nuEff rule as the reference

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
        std::printf("  %-32s SIZE MISMATCH %zu vs %zu  FAIL\n", nm, gpu.size(), ref.size());
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

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    std::vector<std::vector<scalar>> phiBnd(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phiBnd[pi].assign(fvp[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
                phiBnd[pi] = b.values;
    }

    // nuEff: constant on a laminar case, nu + nut where the case has a nut field.
    const bool forceLaminar = (argc > 3 && std::string(argv[3]) == "laminar");
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

    std::printf("test_ueqn_cuda:  (%s: nuEff %s)\n",
            turbulent ? "TURBULENT" : "laminar",
            turbulent ? "varies per cell and per boundary face" : "is constant");

    // ---- the reference ----------------------------------------------------------------------
    const scalar relaxU = 0.7;
    cpu::MomentumInput mi;
    mi.phi = &phiF.internalField; mi.phiBnd = &phiBnd;
    mi.nuEff = &nuEffC;           mi.nuEffBnd = &nuEffB;
    mi.relaxU = relaxU;
    mi.bounded = true;
    mi.correctedLaplacian = true;   // exercised on both paths -- see the controls below
    mi.linearUpwind = true;
    // The scheme under comparison, selectable so one binary covers all of them. Default limitedLinear
    // (a weights change); BRAE_TEST_SCHEME=linearUpwindV picks the limited-correction one instead.
    const char* schEnv = std::getenv("BRAE_TEST_SCHEME");
    const std::string sch = schEnv ? schEnv : "limitedLinear";
    mi.scheme = sch == "linearUpwindV" ? cpu::DivScheme::linearUpwindV
              : sch == "LUST"          ? cpu::DivScheme::LUST
              : sch == "limitedLinearV"? cpu::DivScheme::limitedLinearV
                                       : cpu::DivScheme::limitedLinear;
    mi.schemeCoeff = 1.0;
    std::printf("  div(phi,U) scheme under test: %s\n", sch.c_str());
    const FvVectorMatrix ref = cpu::assembleUEqn(U, mi, m, g, fvp);

    // ---- the CUDA path ----------------------------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);

    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz);

    // Face nuEff, built the same way the reference does (effectiveFaceViscosity): linear interior, the
    // PATCH value on boundary faces. Flattened for the device in bndCell order.
    const SurfaceScalarField nuEffFaceRef =
        cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
    std::vector<scalar> nuBndFlat;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i) nuBndFlat.push_back(nuEffB[pi][i]);
    nuBndFlat.resize(dm.nBndFaces, nu);

    std::vector<scalar> phiBndFlat;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i) phiBndFlat.push_back(phiBnd[pi][i]);
    phiBndFlat.resize(dm.nBndFaces, 0.0);

    DeviceBuffer<scalar> dPhiInt(phiF.internalField), dPhiBnd(phiBndFlat);
    DeviceBuffer<scalar> dNuCell(nuEffC), dNuFace(nuEffFaceRef.internal), dNuBnd(nuBndFlat);

    gpu::MomentumInput gi;
    gi.phiInt = &dPhiInt;         gi.phiBnd = &dPhiBnd;
    gi.nuEffCell = &dNuCell;      gi.nuEffFace = &dNuFace;  gi.nuEffBndFace = &dNuBnd;
    gi.relaxU = relaxU;
    gi.bounded = true;              // exercised on both paths -- see the control below
    gi.correctedLaplacian = true;
    gi.linearUpwind = true;
    gi.scheme = mi.scheme;
    gi.schemeCoeff = mi.schemeCoeff;

    gpu::MomentumMatrix M;
    gpu::assembleUEqn(M, dm, dbU, dUx, dUy, dUz, gi);

    // ---- compare ----------------------------------------------------------------------------
    // The reference's `diag` is the RELAXED diagonal (relaxMatrix writes it in place); the device keeps
    // the raw one and the relaxed one separately, so the relaxed buffer is what corresponds.
    check(M.relaxed, "relaxation ran on the device path");
    // TOLERANCE WITH A LIMITED SCHEME. Upwind weights are exact on both paths (pos0 of the same flux), so
    // those comparisons sit at 0 or 1e-16. A LIMITER does not: r = 2*(gradcf/gradf) - 1 divides by the
    // face difference of magSqr(U), which approaches zero in smooth regions, so the ~1e-16 disagreement
    // between the host and device Gauss gradients (different summation order over faces) is amplified.
    // Measured 5.5e-12 on the off-diagonals with `limitedLinear 1`. That is the arithmetic of the scheme,
    // not a porting defect -- the control below proves the limiter is doing real work (4.4e-01 on the
    // off-diagonals), so this is not a tolerance hiding an absent term.
    // Only the r-RATIO limiters need the looser bound, and the measurement says which: limitedLinear
    // 5.5e-12 and limitedLinearV 1.4e-13, because r = 2*(gradcf/gradf) - 1 divides by a face difference
    // that approaches zero. linearUpwindV and LUST hit 1e-16 -- neither divides by anything small -- so
    // giving them the loose bound would hide a real defect in them.
    const bool ratioLimiter = (mi.scheme == cpu::DivScheme::limitedLinear
                            || mi.scheme == cpu::DivScheme::limitedLinearV);
    const scalar mTol = ratioLimiter ? 5e-11 : 1e-13;
    cmp(M.relaxedDiag.host(), ref.diag,  "diag (relaxed)", mTol);
    cmp(M.upper.host(),       ref.upper, "upper",          mTol);
    cmp(M.lower.host(),       ref.lower, "lower",          mTol);

    const char* sn[3] = {"source x", "source y", "source z"};
    for (int k = 0; k < 3; ++k)
    {
        std::vector<scalar> r(nC);
        for (label c = 0; c < nC; ++c) r[c] = component(ref.source[c], k);
        cmp(M.source[k].host(), r, sn[k], ratioLimiter ? 5e-10 : 1e-11);
    }

    // Boundary coefficients, flattened in the device's bndCell order.
    const char* bn[3] = {"internalCoeffs x", "internalCoeffs y", "internalCoeffs z"};
    const char* cn[3] = {"boundaryCoeffs x", "boundaryCoeffs y", "boundaryCoeffs z"};
    for (int k = 0; k < 3; ++k)
    {
        std::vector<scalar> ric, rbc;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                ric.push_back(component(ref.internalCoeffs[pi][i], k));
                rbc.push_back(component(ref.boundaryCoeffs[pi][i], k));
            }
        std::vector<scalar> gic = M.iC[k].host(), gbc = M.bC[k].host();
        gic.resize(ric.size()); gbc.resize(rbc.size());
        cmp(gic, ric, bn[k], 1e-12);
        cmp(gbc, rbc, cn[k], 1e-12);
    }

    // ---- addPressureGradient ----------------------------------------------------------------
    {
        GeometricField<scalar> p =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
        p.evaluateBoundary();
        const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);

        FvVectorMatrix refP = ref;
        cpu::addPressureGradient(refP, p, m, g, fvp);

        std::vector<scalar> gx(nC), gy(nC), gz(nC);
        for (label c = 0; c < nC; ++c) { gx[c] = gradP[c].x; gy[c] = gradP[c].y; gz[c] = gradP[c].z; }
        DeviceBuffer<scalar> dGx(gx), dGy(gy), dGz(gz);
        gpu::addPressureGradient(M, dm, dGx, dGy, dGz);

        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> r(nC);
            for (label c = 0; c < nC; ++c) r[c] = component(refP.source[c], k);
            cmp(M.source[k].host(), r, k == 0 ? "source x + -grad(p)V"
                                     : k == 1 ? "source y + -grad(p)V" : "source z + -grad(p)V",
                                     ratioLimiter ? 5e-10 : 1e-11);
        }
        // Control: the gradient must have changed the source, or the check above is vacuous.
        scalar moved = 0;
        for (label c = 0; c < nC; ++c)
            moved = std::fmax(moved, std::fabs(component(refP.source[c], 0) - component(ref.source[c], 0)));
        check(moved > 0.0, "the pressure gradient actually changed the source (control)");
    }

    // CONTROL: `bounded` must actually contribute, or comparing it proves nothing. It is SMALL here
    // (~2e-08 of the diagonal) because pitzDailyTurb/1576 is converged and the term is -V*div(phi), which
    // vanishes exactly at convergence -- that is the property that makes it invisible to a converged
    // comparison and the reason it needs its own check rather than being inferred from agreement.
    {
        cpu::MomentumInput noB = mi; noB.bounded = false;
        const FvVectorMatrix refNoB = cpu::assembleUEqn(U, noB, m, g, fvp);
        scalar mx = 0, mg = 0;
        for (std::size_t c = 0; c < ref.diag.size(); ++c)
        { mx = std::fmax(mx, std::fabs(ref.diag[c] - refNoB.diag[c])); mg = std::fmax(mg, std::fabs(ref.diag[c])); }
        const scalar r = mg > 0 ? mx / mg : mx;
        std::printf("  %-54s rel=%.3e\n", "control: `bounded` changes the diagonal", r);
        check(r > 1e-12, "the bounded term actually contributes (control)");
    }

    // CONTROL: the limited scheme must move the MATRIX (it is a weights change, unlike linearUpwind
    // which is a source-only correction), or comparing CUDA against the reference proves nothing about it.
    {
        cpu::MomentumInput noS = mi; noS.scheme = cpu::DivScheme::upwind;
        const FvVectorMatrix refNoS = cpu::assembleUEqn(U, noS, m, g, fvp);
        scalar dU = 0, mU = 0;
        for (std::size_t f = 0; f < ref.upper.size(); ++f)
        { dU = std::fmax(dU, std::fabs(ref.upper[f] - refNoS.upper[f])); mU = std::fmax(mU, std::fabs(ref.upper[f])); }
        const scalar r = mU > 0 ? dU / mU : dU;
        std::printf("  %-54s rel=%.3e\n", "control: the scheme moves the off-diagonals", r);
        // linearUpwindV is a CORRECTION-only scheme (it derives from upwind), so its matrix is upwind's
        // by construction -- asserting it moved would be asserting the opposite of the port.
        if (mi.scheme == cpu::DivScheme::linearUpwindV)
            check(r == 0.0, "linearUpwindV leaves the matrix at upwind's -- correction only (control)");
        else
            check(r > 1e-12, "the limited scheme changes the matrix, not just the source (control)");
    }

    // CONTROL: and for `linearUpwind`. It touches ONLY the source -- linearUpwind derives from `upwind`,
    // so the matrix is unchanged -- and it does NOT vanish at convergence, unlike `bounded`. Both facts
    // are asserted: a diagonal that moved would mean the implicit weights had been changed too, which is
    // not what OpenFOAM does.
    {
        cpu::MomentumInput noL = mi; noL.linearUpwind = false;
        const FvVectorMatrix refNoL = cpu::assembleUEqn(U, noL, m, g, fvp);
        scalar dD = 0, dS = 0, mS = 0;
        for (std::size_t c = 0; c < ref.diag.size(); ++c)
        {
            dD = std::fmax(dD, std::fabs(ref.diag[c] - refNoL.diag[c]));
            dS = std::fmax(dS, std::fabs(component(ref.source[c], 0) - component(refNoL.source[c], 0)));
            mS = std::fmax(mS, std::fabs(component(ref.source[c], 0)));
        }
        const scalar rS = mS > 0 ? dS / mS : dS;
        std::printf("  %-54s rel=%.3e\n", "control: `linearUpwind` moves the source", rS);
        check(rS > 1e-12, "the linearUpwind correction contributes (control)");
        check(dD == 0.0, "linearUpwind leaves the matrix alone -- it is a deferred source only");
    }

    // CONTROL: same argument for `corrected`. Unlike `bounded` this does NOT vanish at convergence -- it
    // is a property of the mesh, not of the solution -- so it must move both the coefficients (implicit
    // half: nonOrthDeltaCoeffs instead of deltaCoeffs) and the source (explicit half). Checking only one
    // would pass with the other half missing, which is precisely the failure the reference had to fix.
    {
        cpu::MomentumInput noC = mi; noC.correctedLaplacian = false;
        const FvVectorMatrix refNoC = cpu::assembleUEqn(U, noC, m, g, fvp);
        scalar dD = 0, mD = 0, dS = 0, mS = 0;
        for (std::size_t c = 0; c < ref.diag.size(); ++c)
        {
            dD = std::fmax(dD, std::fabs(ref.diag[c] - refNoC.diag[c]));
            mD = std::fmax(mD, std::fabs(ref.diag[c]));
            dS = std::fmax(dS, std::fabs(component(ref.source[c], 0) - component(refNoC.source[c], 0)));
            mS = std::fmax(mS, std::fabs(component(ref.source[c], 0)));
        }
        const scalar rD = mD > 0 ? dD / mD : dD, rS = mS > 0 ? dS / mS : dS;
        std::printf("  %-54s rel=%.3e\n", "control: `corrected` moves the coefficients", rD);
        std::printf("  %-54s rel=%.3e\n", "control: `corrected` moves the source", rS);
        check(rD > 1e-12, "the implicit half of the non-orth correction contributes (control)");
        check(rS > 1e-12, "the explicit half of the non-orth correction contributes (control)");
    }

    // ---- refusals ---------------------------------------------------------------------------
    for (int which = 0; which < 2; ++which)
    {
        gpu::MomentumInput bad = gi;
        (which == 0 ? bad.hasMRF : bad.hasFvOptions) = true;
        gpu::MomentumMatrix Mb;
        bool threw = false;
        try { gpu::assembleUEqn(Mb, dm, dbU, dUx, dUy, dUz, bad); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, which == 0 ? "MRF is refused on the CUDA path"
                                : "fvOptions is refused on the CUDA path");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
