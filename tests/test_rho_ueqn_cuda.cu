// CUDA rhoUEqn against the _cpp reference, field by field.
//
// The reference is itself gated against OpenFOAM's own momentum dump (tests/rho_ueqn_vs_openfoam.sh), so
// this closes OpenFOAM -> _cpp -> CUDA for the whole COMPRESSIBLE momentum assembly rather than for
// individual kernels. It is the device twin of tests/test_ueqn_cuda.cu and is meant to be read beside it.
//
// The matrix is compared in its DECOMPOSED form -- diag, upper, lower, the three sources, and every
// boundary coefficient on every patch -- because a folded or fused comparison collapses convection,
// diffusion, the explicit stress, the boundary coefficients and relaxation into one number that cannot
// say which of them is wrong.
//
// WHY rho IS SYNTHESIZED HERE, and why that is the right call rather than a shortcut.
//
// No committed fixture ships a rho field: the compressible _cpp gates generate their state by running
// real OpenFOAM from a shell script, which is what makes them oracles. This gate is a different
// instrument -- it measures whether the DEVICE reproduces the HOST on byte-identical inputs, and for that
// the inputs need to exercise every path, not to be a physical solution. Physical validity is the
// reference's job and the reference has its own gate for it.
//
// rho MUST BE NON-UNIFORM, and that is the whole point. mu_eff = rho*nu_eff is the ONE thing separating
// this equation from simpleFoam's. With a uniform rho, mu_eff is a constant multiple of nu_eff, so a
// device kernel that dropped the rho weighting from ONE term and kept it in another would still agree to
// machine precision on a rescaled matrix. A rho that varies across the domain is what makes the terms
// discriminable from each other. It is asserted below rather than assumed, because a fixture that cannot
// discriminate turns every number here into decoration -- and this project has shipped exactly that
// before (a uniform-rho fixture made two different flux forms algebraically identical).
//
// WHAT THIS GATE DOES NOT CLAIM.
//
//   * It does not claim the compressible equation is right -- that is rho_ueqn_vs_openfoam.sh's claim,
//     against OpenFOAM's own assembled matrix. This one claims only that CUDA == _cpp.
//   * It does not cover relaxU == 1.0, though the two paths now AGREE there. Both carry
//     `relaxEquationU`, which distinguishes "the case named no factor" (OpenFOAM does not relax) from
//     "the case named 1" (OpenFOAM DOES relax -- relax(1.0) still applies the diagonal-dominance clamp
//     and the (D - D0)*psi source step; only alpha <= 0 early-returns). The reference used to use
//     relaxU == 1.0 as the sentinel for both and has been fixed. No validation case names a factor of
//     exactly 1 for U, so this gate runs every case at 0.7 where both relax, and the agreement at 1.0 is
//     asserted by construction rather than measured here.
//   * It does not cover coupled patches. buildDeviceMesh keeps cyclic/AMI/processor faces out of the LDU,
//     and the device module REFUSES a mesh that has them (asserted below). The host does not refuse.
//   * It does not cover DarcyForchheimer porosity. The device refuses it; the host runs it through the
//     shared addSup with nu = 0.0, which drops the viscous Darcy term on a force-dimensioned equation.
//     Also in PORT.md.
//
// Run: test_rho_ueqn_cuda <caseDir> <timeDir> [laminar]
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
    std::printf("  %-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

// The largest relative difference between two assembled matrices, over the diagonal and the x source.
// Used by the controls, which have to show a term MOVES the answer rather than assert it was applied.
static void matrixSpread(const FvVectorMatrix& a,
                         const FvVectorMatrix& b,
                         scalar&               relDiag,
                         scalar&               relSrc)
{
    scalar dD = 0, mD = 0, dS = 0, mS = 0;
    for (std::size_t c = 0; c < a.diag.size(); ++c)
    {
        dD = std::fmax(dD, std::fabs(a.diag[c] - b.diag[c]));
        mD = std::fmax(mD, std::fabs(a.diag[c]));
        dS = std::fmax(dS, std::fabs(a.source[c].x - b.source[c].x));
        mS = std::fmax(mS, std::fabs(a.source[c].x));
    }
    relDiag = mD > 0 ? dD / mD : dD;
    relSrc  = mS > 0 ? dS / mS : dS;
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

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
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
        {
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
            {
                phiBnd[pi] = b.values;
            }
        }
    }

    // nuEff: constant on a laminar case, nu + nut where the case ships a nut field. Both matter -- with a
    // constant nuEff a kernel that reads the owner cell's viscosity on a wall instead of the patch value
    // still agrees, so the varying case is what exercises the face rule.
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

    // rho, smooth in x across the bounding box and spanning the range a real compressible duct covers.
    // Deterministic so two runs of this gate compare the same matrices. See the header for why it is
    // synthesized and why it must not be uniform.
    scalar xMin = 1e300, xMax = -1e300;
    for (label c = 0; c < nC; ++c)
    {
        xMin = std::fmin(xMin, g.C()[c].x);
        xMax = std::fmax(xMax, g.C()[c].x);
    }
    const scalar xSpan = (xMax > xMin) ? (xMax - xMin) : 1.0;
    auto rhoAt = [&](const vector& x) { return scalar(0.8) + scalar(0.6) * (x.x - xMin) / xSpan; };

    std::vector<scalar> rhoC(nC);
    for (label c = 0; c < nC; ++c) rhoC[c] = rhoAt(g.C()[c]);
    std::vector<std::vector<scalar>> rhoB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        rhoB[pi].resize(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i) rhoB[pi][i] = rhoAt(fvp[pi].Cf[i]);
    }

    scalar rMin = 1e300, rMax = -1e300;
    for (label c = 0; c < nC; ++c)
    {
        rMin = std::fmin(rMin, rhoC[c]);
        rMax = std::fmax(rMax, rhoC[c]);
    }
    std::printf("test_rho_ueqn_cuda:  (%s: nuEff %s)  rho in [%.4f .. %.4f]\n",
                turbulent ? "TURBULENT" : "laminar",
                turbulent ? "varies per cell and per boundary face" : "is constant",
                (double)rMin, (double)rMax);
    // THE FIXTURE MUST BE ABLE TO DISCRIMINATE. A uniform rho makes mu_eff a constant multiple of nu_eff
    // and every comparison below vacuous -- see the header.
    check((rMax - rMin) / rMax > 0.1, "rho varies across the domain (fixture discriminates)");

    // ---- the reference ----------------------------------------------------------------------
    const scalar relaxU = 0.7;
    cpu::rhoSimple::RhoMomentumInput mi;
    mi.phi = &phiF.internalField;
    mi.phiBnd = &phiBnd;
    mi.rho = &rhoC;
    mi.rhoBnd = &rhoB;
    mi.nuEff = &nuEffC;
    mi.nuEffBnd = &nuEffB;
    mi.relaxU = relaxU;
    mi.relaxEquationU = true;   // matches gi.relaxEquationU on the device side
    mi.bounded = true;
    mi.correctedLaplacian = true;   // both halves are exercised -- see the control below
    mi.linearUpwind = true;
    const char* schEnv = std::getenv("BRAE_TEST_SCHEME");
    const std::string sch = schEnv ? schEnv : "limitedLinear";
    mi.scheme = sch == "linearUpwindV"  ? cpu::rhoSimple::DivScheme::linearUpwindV
              : sch == "LUST"           ? cpu::rhoSimple::DivScheme::LUST
              : sch == "limitedLinearV" ? cpu::rhoSimple::DivScheme::limitedLinearV
              : sch == "upwind"         ? cpu::rhoSimple::DivScheme::upwind
                                        : cpu::rhoSimple::DivScheme::limitedLinear;
    mi.schemeCoeff = 1.0;
    // `grad(U) cellLimited Gauss linear <k>`, when the case asks for one. This is the ONE input that
    // makes the EMPTY-patch question visible: emptyFvPatchField is zero-sized in OpenFOAM, so those
    // faces do not exist there, and brae has to skip them explicitly in both the min/max range and the
    // face limit. In the face loop it is not harmless -- Cf - C for an empty face points out of the 2D
    // plane, so the extrapolate is round-off and r = maxDelta/extrapolate can clamp the limiter far
    // below what any real face asks for. pitzDailyTurb is 2D, so this arm is where the two paths have to
    // agree about faces that OpenFOAM does not have.
    const char* limEnv = std::getenv("BRAE_TEST_GRADLIMIT");
    mi.gradULimitK = limEnv ? std::atof(limEnv) : 0.0;
    std::printf("  div(phi,U) scheme under test: %s   grad(U) cellLimited k = %g\n",
                sch.c_str(), (double)mi.gradULimitK);
    const FvVectorMatrix ref = cpu::rhoSimple::assembleUEqn(U, mi, m, g, fvp);

    // ---- the CUDA path ----------------------------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);

    // The boundary arrays below are flattened in PATCH order and handed to the device as its boundary
    // order. That is only the same array when no patch has been dropped from the gather, which is what
    // buildDeviceMesh does to coupled patches -- so it is asserted rather than assumed.
    label nBndPatchFaces = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nBndPatchFaces += fvp[pi].size;
    check(nBndPatchFaces == dm.nBndFaces,
          "device boundary order matches patch order (no dropped patch)");

    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    {
        ux[c] = U.internal[c].x;
        uy[c] = U.internal[c].y;
        uz[c] = U.internal[c].z;
    }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz);

    std::vector<scalar> nuBndFlat, rhoBndFlat, phiBndFlat;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            nuBndFlat.push_back(nuEffB[pi][i]);
            rhoBndFlat.push_back(rhoB[pi][i]);
            phiBndFlat.push_back(phiBnd[pi][i]);
        }
    }
    nuBndFlat.resize(dm.nBndFaces, nu);
    rhoBndFlat.resize(dm.nBndFaces, 1.0);
    phiBndFlat.resize(dm.nBndFaces, 0.0);

    DeviceBuffer<scalar> dPhiInt(phiF.internalField), dPhiBnd(phiBndFlat);
    DeviceBuffer<scalar> dRhoCell(rhoC), dRhoBnd(rhoBndFlat);
    DeviceBuffer<scalar> dNuCell(nuEffC), dNuBnd(nuBndFlat);

    gpu::rhoSimple::RhoMomentumInput gi;
    gi.phiInt = &dPhiInt;
    gi.phiBnd = &dPhiBnd;
    gi.rhoCell = &dRhoCell;
    gi.rhoBndFace = &dRhoBnd;
    gi.nuEffCell = &dNuCell;
    gi.nuEffBndFace = &dNuBnd;
    // The host now carries the same flag (RhoMomentumInput::relaxEquationU); at 0.7 both paths relax and
    // the two agree. See the header for the one input this gate therefore does not cover.
    gi.relaxEquationU = true;
    gi.relaxU = relaxU;
    gi.bounded = true;
    gi.correctedLaplacian = true;
    gi.linearUpwind = true;
    gi.scheme = mi.scheme;
    gi.schemeCoeff = mi.schemeCoeff;
    gi.gradULimitK = mi.gradULimitK;

    gpu::MomentumMatrix M;
    gpu::rhoSimple::assembleUEqn(M, dm, dbU, dUx, dUy, dUz, gi);

    // ---- compare ----------------------------------------------------------------------------
    // The reference's `diag` is the RELAXED diagonal (relaxMatrix writes it in place); the device keeps
    // the raw and the relaxed one separately, so relaxedDiag is what corresponds.
    check(M.relaxed, "relaxation ran on the device path");

    // TOLERANCE BY SCHEME, and the reason is arithmetic rather than porting. Upwind weights are exact on
    // both paths (pos0 of the same flux). An r-RATIO limiter is not: r = 2*(gradcf/gradf) - 1 divides by
    // a face difference that approaches zero in smooth regions, so the ~1e-16 disagreement between the
    // host and device Gauss gradients (different summation order over faces) is amplified. The
    // incompressible twin measured 5.5e-12 on the off-diagonals with `limitedLinear 1` and 1.4e-13 with
    // limitedLinearV, while linearUpwindV and LUST hit 1e-16 -- so giving those the loose bound would
    // hide a real defect in them. The controls below prove the limiter is doing real work, so this is not
    // a tolerance standing in for an absent term.
    const bool ratioLimiter = (mi.scheme == cpu::rhoSimple::DivScheme::limitedLinear
                            || mi.scheme == cpu::rhoSimple::DivScheme::limitedLinearV);
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

    const char* bn[3] = {"internalCoeffs x", "internalCoeffs y", "internalCoeffs z"};
    const char* cn[3] = {"boundaryCoeffs x", "boundaryCoeffs y", "boundaryCoeffs z"};
    for (int k = 0; k < 3; ++k)
    {
        std::vector<scalar> ric, rbc;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                ric.push_back(component(ref.internalCoeffs[pi][i], k));
                rbc.push_back(component(ref.boundaryCoeffs[pi][i], k));
            }
        }
        std::vector<scalar> gic = M.iC[k].host(), gbc = M.bC[k].host();
        gic.resize(ric.size());
        gbc.resize(rbc.size());
        cmp(gic, ric, bn[k], 1e-12);
        cmp(gbc, rbc, cn[k], 1e-12);
    }

    // ---- THE CONTROL THIS SOLVER EXISTS FOR -------------------------------------------------
    // mu_eff = rho*nu_eff is the one thing separating this equation from simpleFoam's. Injecting the
    // KINEMATIC nu_eff in its place -- the incompressible divDevReff -- must produce a MEASURABLY
    // DIFFERENT matrix, or every comparison above would pass just as well with the rho weighting missing.
    // The manifest records 6.2e-01 for this against OpenFOAM's own matrix.
    //
    // It is run on BOTH paths, which makes it two controls in one: it proves the fixture discriminates
    // the dynamic form from the kinematic one, AND it exercises the muEff injection path that the gates
    // against real OpenFOAM depend on to measure the assembly without a ported closure in the way.
    {
        cpu::rhoSimple::RhoMomentumInput kin = mi;
        kin.muEff = &nuEffC;
        kin.muEffBnd = &nuEffB;
        const FvVectorMatrix refKin = cpu::rhoSimple::assembleUEqn(U, kin, m, g, fvp);

        scalar rD = 0, rS = 0;
        matrixSpread(ref, refKin, rD, rS);
        std::printf("  %-58s diag=%.3e src=%.3e\n",
                    "control: the KINEMATIC form differs from the dynamic", (double)rD, (double)rS);
        check(rD > 1e-3, "mu_eff = rho*nu_eff actually changes the matrix (control)");

        DeviceBuffer<scalar> dMuCell(nuEffC), dMuBnd(nuBndFlat);
        gpu::rhoSimple::RhoMomentumInput gkin = gi;
        gkin.muEffCell = &dMuCell;
        gkin.muEffBndFace = &dMuBnd;
        gpu::MomentumMatrix Mkin;
        gpu::rhoSimple::assembleUEqn(Mkin, dm, dbU, dUx, dUy, dUz, gkin);
        cmp(Mkin.relaxedDiag.host(), refKin.diag, "injected-muEff diag (both paths)", mTol);
    }

    // ---- addPressureGradient ----------------------------------------------------------------
    {
        GeometricField<scalar> p =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
        p.evaluateBoundary();
        const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);

        FvVectorMatrix refP = ref;
        cpu::rhoSimple::addPressureGradient(refP, p, m, g, fvp);

        std::vector<scalar> gx(nC), gy(nC), gz(nC);
        for (label c = 0; c < nC; ++c)
        {
            gx[c] = gradP[c].x;
            gy[c] = gradP[c].y;
            gz[c] = gradP[c].z;
        }
        DeviceBuffer<scalar> dGx(gx), dGy(gy), dGz(gz);
        gpu::addPressureGradient(M, dm, dGx, dGy, dGz);

        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> r(nC);
            for (label c = 0; c < nC; ++c) r[c] = component(refP.source[c], k);
            cmp(M.source[k].host(), r,
                k == 0 ? "source x + -grad(p)V" : k == 1 ? "source y + -grad(p)V"
                                                         : "source z + -grad(p)V",
                ratioLimiter ? 5e-10 : 1e-11);
        }
        scalar moved = 0;
        for (label c = 0; c < nC; ++c)
        {
            moved = std::fmax(moved, std::fabs(refP.source[c].x - ref.source[c].x));
        }
        check(moved > 0.0, "the pressure gradient actually changed the source (control)");
    }

    // ---- the term controls ------------------------------------------------------------------
    // `bounded`: -V*div(phi), which vanishes exactly at convergence. That is the property making it
    // invisible to a converged comparison, and the reason it needs its own check.
    {
        cpu::rhoSimple::RhoMomentumInput noB = mi;
        noB.bounded = false;
        const FvVectorMatrix refNoB = cpu::rhoSimple::assembleUEqn(U, noB, m, g, fvp);
        scalar rD = 0, rS = 0;
        matrixSpread(ref, refNoB, rD, rS);
        std::printf("  %-58s rel=%.3e\n", "control: `bounded` changes the diagonal", (double)rD);
        check(rD > 1e-12, "the bounded term actually contributes (control)");
    }

    // The div scheme: a WEIGHTS change for the limited family, so it must move the off-diagonals.
    // linearUpwindV derives from upwind and is correction-only, so for it the opposite is asserted.
    {
        cpu::rhoSimple::RhoMomentumInput noS = mi;
        noS.scheme = cpu::rhoSimple::DivScheme::upwind;
        const FvVectorMatrix refNoS = cpu::rhoSimple::assembleUEqn(U, noS, m, g, fvp);
        scalar dU = 0, mU = 0;
        for (std::size_t f = 0; f < ref.upper.size(); ++f)
        {
            dU = std::fmax(dU, std::fabs(ref.upper[f] - refNoS.upper[f]));
            mU = std::fmax(mU, std::fabs(ref.upper[f]));
        }
        const scalar r = mU > 0 ? dU / mU : dU;
        std::printf("  %-58s rel=%.3e\n", "control: the scheme moves the off-diagonals", (double)r);
        if (mi.scheme == cpu::rhoSimple::DivScheme::linearUpwindV)
        {
            check(r == 0.0, "linearUpwindV leaves the matrix at upwind's -- correction only (control)");
        }
        else if (mi.scheme != cpu::rhoSimple::DivScheme::upwind)
        {
            check(r > 1e-12, "the limited scheme changes the matrix, not just the source (control)");
        }
    }

    // `linearUpwind` touches ONLY the source, and unlike `bounded` it does not vanish at convergence.
    // Both facts are asserted: a diagonal that moved would mean the implicit weights were changed too,
    // which is not what OpenFOAM does.
    {
        cpu::rhoSimple::RhoMomentumInput noL = mi;
        noL.linearUpwind = false;
        const FvVectorMatrix refNoL = cpu::rhoSimple::assembleUEqn(U, noL, m, g, fvp);
        scalar dD = 0, dS = 0, mS = 0;
        for (std::size_t c = 0; c < ref.diag.size(); ++c)
        {
            dD = std::fmax(dD, std::fabs(ref.diag[c] - refNoL.diag[c]));
            dS = std::fmax(dS, std::fabs(ref.source[c].x - refNoL.source[c].x));
            mS = std::fmax(mS, std::fabs(ref.source[c].x));
        }
        const scalar rS = mS > 0 ? dS / mS : dS;
        std::printf("  %-58s rel=%.3e\n", "control: `linearUpwind` moves the source", (double)rS);
        check(rS > 1e-12, "the linearUpwind correction contributes (control)");
        check(dD == 0.0, "linearUpwind leaves the matrix alone -- deferred source only");
    }

    // `corrected`: unlike `bounded` this does NOT vanish at convergence -- it is a property of the mesh,
    // not of the solution -- so it must move BOTH the coefficients (implicit half: nonOrthDeltaCoeffs in
    // place of deltaCoeffs) and the source (explicit half). Checking only one would pass with the other
    // half missing, which is exactly the defect this port paid for on the energy and pressure equations.
    {
        cpu::rhoSimple::RhoMomentumInput noC = mi;
        noC.correctedLaplacian = false;
        const FvVectorMatrix refNoC = cpu::rhoSimple::assembleUEqn(U, noC, m, g, fvp);
        scalar rD = 0, rS = 0;
        matrixSpread(ref, refNoC, rD, rS);
        std::printf("  %-58s diag=%.3e src=%.3e\n",
                    "control: `corrected` moves coefficients AND source", (double)rD, (double)rS);
        check(rD > 1e-12, "the implicit half of the non-orth correction contributes (control)");
        check(rS > 1e-12, "the explicit half of the non-orth correction contributes (control)");
    }

    // ---- refusals ---------------------------------------------------------------------------
    // Refusing by name is half the port's contract, so the gate asserts the refusals as well as the
    // numbers. hasCoupledPatches is a refusal the DEVICE has and the host does not -- see the header.
    {
        struct { const char* what; int which; } cases[] = {
            {"MRF is refused on the CUDA path", 0},
            {"an unported fvOptions is refused on the CUDA path", 1},
            {"a mesh with coupled patches is refused on the CUDA path", 2},
        };
        for (const auto& cse : cases)
        {
            gpu::rhoSimple::RhoMomentumInput bad = gi;
            if (cse.which == 0) bad.hasMRF = true;
            if (cse.which == 1) bad.hasFvOptions = true;
            if (cse.which == 2) bad.hasCoupledPatches = true;
            gpu::MomentumMatrix Mb;
            bool threw = false;
            try
            {
                gpu::rhoSimple::assembleUEqn(Mb, dm, dbU, dUx, dUy, dUz, bad);
            }
            catch (const std::runtime_error&)
            {
                threw = true;
            }
            check(threw, cse.what);
        }
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
