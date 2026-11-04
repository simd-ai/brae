// Two things kEpsilon::correct() has to get right, both found by tracing the k/epsilon equations term by
// term against OpenFOAM on pimpleFoam/RAS/oscillatingInletACMI2D.
//
// 1. THE PRODUCTION GRADIENT TAKES THE NAMED grad(U) SCHEME.
//    OF's kEpsilon::correct() opens with `tmp<volTensorField> tgradU = fvc::grad(U)`, which resolves
//    gradSchemes `grad(U)` -- `cellLimited Gauss linear 1` on that case. brae's kOmegaSST path had
//    honoured it for a long time; kEpsilon's never did, and `gradULimitK` appeared nowhere in
//    device_kepsilon.cu. An unlimited production gradient is LARGER exactly where the limiter would have
//    bitten, and the trace showed it: reconstructing OF's own step-1 production from its written fields
//    (U and phi at the step, nut/k/epsilon from the step before, which is all correct() ever reads),
//
//        GbyNu, inlet channel : 2.06e+01 (L2 rel) -> 1.32e-02
//        GbyNu, duct          : 3.42e+00          -> 7.54e-08
//
//    the duct going exact to 7.5e-08 being the proof that the gradient OPERATOR was always right and only
//    the scheme was wrong. Over ten steps that took k from 1.76e-02 to 6.92e-04 on the static case, and
//    on the moving one k 1.26 -> 9.76e-02, epsilon 2.69 -> 4.25e-01, U 2.47e-02 -> 9.05e-03.
//
// 2. THE MATRIX CONSTRAINT TAKES THE BLENDED epsilon, NOT THE RAW WALL VALUE.
//    epsilonWallFunctionFvPatchScalarField::updateCoeffs(weights) blends
//        epsilon[celli] = (1 - w)*epsilon[celli] + w*epsilon0[celli]
//    and manipulateMatrix then appends `epsilon[celli]` -- the value it has just blended -- into
//    matrix.setValues. brae blended the FIELD but constrained the MATRIX to epsilon0, the unblended wall
//    value. On an ordinary wall w = 1 and the blend is the identity, so this was invisible everywhere
//    except the cells a cyclicACMI partially covers, where w = 1 - mask. On the moving case that is a
//    different set of cells every step: epsilon 4.25e-01 -> 5.46e-02, worst cell 9.7e-01 -> 7.7e-02.
//
// Leg 2's identity is what separates the two behaviours cleanly. epsilon0 depends only on the INCOMING k
// (deviceWallEpsG0 runs before any solve), so it is the same in both runs, and a hard constraint gives
//     w = 1   : eps_final = eps0
//     w = 0.5 : eps_final = 0.5*epsIn + 0.5*eps0 = 0.5*(epsIn + eps_final(w=1))
// whereas constraining to the raw eps0 would give eps_final(w=0.5) == eps_final(w=1). The two predictions
// differ by exactly the amount the blend is worth, so the leg cannot pass for the wrong implementation.
#include "box_mesh.cuh"
#include "device_kepsilon.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_buffer.cuh"
#include "cyclic_field.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

scalar worst(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar d = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::fmax(d, std::fabs(a[i] - b[i]));
    return d;
}

}   // namespace

int main()
{
    // 3D, and big enough to have cells that touch NO wall. On a single-layer box every cell is a wall
    // cell, the wall function overwrites G everywhere (w = 1 => G = G0), and the production gradient
    // never survives to be compared -- which is exactly how the first draft of leg 1 failed.
    const label N = 6;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const scalar nu = 1e-3, relaxEps = 1.0, relaxK = 1.0, tol = 1e-12;

    // A SHARP velocity field: the limiter is a no-op on a gradient the Gauss sum already reproduces, so a
    // smooth field would make leg 1 vacuous.
    std::vector<vector> Uc(nC);
    for (label c = 0; c < nC; ++c) Uc[c] = vector{g.C()[c].x > 0.5*N ? 10.0 : 1.0, 0.3*g.C()[c].y, 0.0};
    GeometricField<vector> U = buildCyclicField<vector>(Uc, fvp, {}, /*wallNoSlip*/true);
    U.evaluateBoundary();

    std::vector<scalar> kIn(nC), eIn(nC), nIn(nC);
    for (label c = 0; c < nC; ++c)
    {
        kIn[c] = 0.15 + 0.03*std::sin(0.7*c);
        eIn[c] = 1.5  + 0.50*std::sin(1.1*c);
        nIn[c] = 0.09*kIn[c]*kIn[c]/eIn[c];
    }
    GeometricField<scalar> kf = buildCyclicField<scalar>(kIn, fvp, {});   kf.evaluateBoundary();
    GeometricField<scalar> ef = buildCyclicField<scalar>(eIn, fvp, {});   ef.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    const DeviceBoundary dbK = buildDeviceBoundary(kf, fvp, g), dbEps = buildDeviceBoundary(ef, fvp, g);
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = Uc[c].x; uy[c] = Uc[c].y; uz[c] = Uc[c].z; }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz);
    std::vector<scalar> phiB;
    for (const auto& a : phi.boundary) phiB.insert(phiB.end(), a.begin(), a.end());

    const DeviceWallData wall0 = buildDeviceWallData(m, g, fvp, U);
    const std::vector<label> isW = wall0.isWallCell.host();
    std::size_t nWall = 0;
    for (label c = 0; c < nC; ++c) if (isW[c]) ++nWall;
    std::printf("  fixture: %d cells, %zu wall-constrained\n", (int)nC, nWall);
    if (!nWall)
    { std::printf("  FAIL vacuous: no constrained cells, so leg 2 tests nothing\n"); ++failures; }

    // One correct() from the same state, with a chosen grad(U) limiter coefficient and wall weight.
    auto run = [&](scalar gradULimitK, scalar wallWeight, std::vector<scalar>& kOut, std::vector<scalar>& eOut)
    {
        DeviceWallData w = buildDeviceWallData(m, g, fvp, U);
        w.wallW.copyFrom(std::vector<scalar>(nC, wallWeight));
        DeviceBuffer<scalar> dk(kIn), de(eIn), dnut(nIn);
        // both fields under one rule, the cap the solves had before each field's own was handed down
        const ScalarStopRule stop{tol, 0.0, kUnreadEntryMaxIter, 0};
        // grad(k) and grad(epsilon) unlimited, which the one scalar 0.0 passed here meant before each took its own entry
        TurbScalarGradLimits unlimited;
        unlimited.kField = 0.0;
        unlimited.eField = 0.0;
        unlimited.kLU    = 0.0;
        unlimited.eLU    = 0.0;
        deviceKEpsilonCorrect(dm, w, dbEps, dbK, dbU, dUx, dUy, dUz, dk, de, dnut,
                              DeviceBuffer<scalar>(phi.internal), DeviceBuffer<scalar>(phiB),
                              nu, relaxEps, relaxK, stop, stop,
                              /*bounded*/false, /*boundedEps*/false, /*limitedK*/false, /*limitedEps*/false,
                              2.0, 2.0, KEpsilonCoeffs{}, /*keCheckEvery*/1,
                              /*luK*/false, /*luEps*/false, /*nonOrth*/false, /*gsK*/false, /*gsEps*/false,
                              /*ami*/nullptr, /*cyc*/nullptr, /*nutWall*/0, /*atmZ0*/0.0, /*atmBoundNut*/true,
                              ScalarDdt{}, ScalarDdt{},
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              unlimited, gradULimitK);
        kOut = dk.host();
        eOut = de.host();
    };

    // ---- 1. the production gradient must read the named grad(U) coefficient ----
    {
        std::vector<scalar> kUn, eUn, kLim, eLim;
        run(0.0, 1.0, kUn, eUn);
        run(1.0, 1.0, kLim, eLim);
        const scalar dk = worst(kUn, kLim);
        scalar scale = 0;
        for (scalar v : kUn) scale = std::fmax(scale, std::fabs(v));
        std::printf("  grad(U) limiter: |k(limited) - k(unlimited)| = %.3e  (k peak %.3e)\n",
                    (double)dk, (double)scale);
        if (dk <= 1e-14)
        {
            std::printf("  FAIL the coefficient was IGNORED: on a step velocity profile a cellLimited\n"
                        "       grad(U) must change the production, so k cannot come out identical\n");
            ++failures;
        }
        if (scale <= 0 || dk/scale < 1e-4)
        {
            std::printf("  FAIL vacuous: the limiter moved k by only %.3e of its own size\n",
                        (double)(scale > 0 ? dk/scale : 0));
            ++failures;
        }
        // determinism of the unlimited path, so leg 1's difference is the coefficient and not noise
        std::vector<scalar> kUn2, eUn2;
        run(0.0, 1.0, kUn2, eUn2);
        if (worst(kUn, kUn2) != scalar(0))
        { std::printf("  FAIL the kc = 0 path is not reproducible\n"); ++failures; }
    }

    // ---- 2. the epsilon constraint value is the BLENDED epsilon ----
    {
        std::vector<scalar> kFull, eFull, kHalf, eHalf;
        run(1.0, 1.0, kFull, eFull);   // w = 1: the constraint is eps0 itself
        run(1.0, 0.5, kHalf, eHalf);   // w = 0.5: it must be halfway from the incoming epsilon
        scalar dBlend = 0, dRaw = 0;
        std::size_t n = 0;
        for (label c = 0; c < nC; ++c)
        {
            if (!isW[c]) continue;
            ++n;
            const scalar want = 0.5*eIn[c] + 0.5*eFull[c];   // OF: (1-w)*eps + w*eps0, constrained hard
            dBlend = std::fmax(dBlend, std::fabs(eHalf[c] - want));
            dRaw   = std::fmax(dRaw,   std::fabs(eHalf[c] - eFull[c]));   // what the defect would give
        }
        std::printf("  blend at w = 0.5 over %zu cells: |eps - (1-w)eps_in - w*eps0| = %.3e\n", n, (double)dBlend);
        std::printf("                                  |eps - eps0| (the defect's answer) = %.3e\n", (double)dRaw);
        if (dBlend > 1e-9)
        {
            std::printf("  FAIL the matrix was not constrained to the blended epsilon. OF's manipulateMatrix\n"
                        "       appends epsilon[celli] AFTER updateCoeffs(weights) has blended it\n");
            ++failures;
        }
        // VACUITY GUARD: the two predictions have to be distinguishable on this fixture.
        if (dRaw < 1e-6)
        {
            std::printf("  FAIL vacuous: blending changes nothing here (eps_in is too close to eps0), so the\n"
                        "       leg would pass for an implementation that constrained to the raw wall value\n");
            ++failures;
        }
    }

    std::printf("kepsilon_production_and_blend: %d failures\n", failures);
    return failures ? 1 : 0;
}
