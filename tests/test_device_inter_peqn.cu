// interFoam's pressure corrector on the device -- the four things that are interFoam's own.
//
// EVERY FIXTURE CHOICE HERE EXISTS TO STOP AN ARM BEING VACUOUS, and each was learned from a gate in
// this tree that was vacuous first:
//
//   rAU IS NON-UNIFORM, and physically so: rAU = dt/rho with rho jumping 1000x across the interface.
//   Note 3's two forms -- rAU*reconstruct(f/rAUf) and reconstruct(f) -- coincide EXACTLY when rAU is
//   uniform, which it is on every single-phase fixture with a uniform mesh. Arm 3b measures how far
//   apart they are here.
//
//   rho AND rAU BOTH VARY ACROSS THE SAME FACES. Note 2 is that interpolate(rho*rAU) is not
//   interpolate(rho)*interpolate(rAU); if one varied in x and the other in y, no face would see both
//   and the gap would sit at 1e-14. An earlier version of the host gate did exactly that.
//
//   THE CELLS ARE 2:1:0.5. On a cube every face has the same area, so anything carrying a magSf
//   factor is indistinguishable from the same thing without one.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "inter_peqn_cpp.cuh"
#include "device_inter_peqn.cuh"
#include "device_fvc_reconstruct.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

using namespace brae;
namespace ifm = brae::cpu::interFoam;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
scalar worst(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar w = 0;
    for (std::size_t i = 0; i < a.size(); ++i) w = std::fmax(w, std::fabs(a[i] - b[i]));
    return w;
}
}   // namespace

int main()
{
    std::printf("== interFoam pressure corrector: device\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    const label N = 8;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N, scalar(0), scalar(2), scalar(1), scalar(0.5));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const label nFaces = static_cast<label>(g.magSf().size());
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // a VoF-like state: rho jumps 1000x across a plane, and rAU = dt/rho follows it
    const scalar dt = scalar(1e-3);
    std::vector<scalar> rho(static_cast<std::size_t>(nC)), rAU(static_cast<std::size_t>(nC)),
                        gh(static_cast<std::size_t>(nC)), p_rgh(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        rho[c]   = (C.y < scalar(4)) ? scalar(1000) : scalar(1);
        rAU[c]   = dt / rho[c];                       // the physical rAU, so both vary on the same faces
        gh[c]    = scalar(-9.81) * C.y;
        p_rgh[c] = scalar(3)*std::sin(C.x) + scalar(2)*C.z;
    }

    DeviceBuffer<scalar> dRho(rho), dRAU(rAU), dGh(gh), dPrgh(p_rgh);

    // ---- 1. phig -- and that it carries NO snGrad(p_rgh) ------------------------------------------
    std::vector<scalar> stf(nFaces), ghf(nFaces), snRho(nFaces), rAUf(nFaces), snP(nFaces);
    for (label f = 0; f < nFaces; ++f)
    {
        stf[f]   = scalar(0.07)*std::sin(scalar(f));
        ghf[f]   = scalar(-9.81)*g.Cf()[f].y;
        snRho[f] = scalar(999)*std::cos(scalar(0.3)*scalar(f));
        rAUf[f]  = dt / (scalar(1) + scalar(999)*scalar(0.5)*(scalar(1) + std::sin(scalar(f))));
        // NEARLY HYDROSTATIC: snGrad(p_rgh) balances ghf*snGrad(rho) to within 10%, which is what the
        // p_rgh formulation makes true and what decides the scale of the term. The first version of
        // this fixture used 40*sin, three orders below ghf*snGradRho, and the note-1 control came back
        // at 7.6e-02 against a phig of 1.3e+02 -- comparing a term to one it could never rival.
        snP[f]   = ghf[f]*snRho[f] * (scalar(1) + scalar(0.1)*std::sin(scalar(0.17)*scalar(f)));
    }
    {
        std::vector<scalar> hostPhig;
        ifm::buoyancyFlux(stf, ghf, snRho, rAUf, g.magSf(), hostPhig);

        DeviceBuffer<scalar> dStf(stf), dGhf(ghf), dSnRho(snRho), dRAUf(rAUf), dMagSf(g.magSf()), dPhig;
        deviceBuoyancyFlux(static_cast<int>(nFaces), dStf, dGhf, dSnRho, dRAUf, dMagSf, dPhig);
        if (cudaDeviceSynchronize() != cudaSuccess)
        { std::printf("  FAIL: kernels did not complete\n"); return 1; }
        std::vector<scalar> got;
        dPhig.copyTo(got);
        std::printf("  phig: worst |device - host| = %.3e\n", (double)worst(got, hostPhig));
        check("phig matches the host bit for bit", worst(got, hostPhig) == scalar(0));

        // THE CONTROL FOR NOTE 1: what phig would be if snGrad(p_rgh) were carried into it as well --
        // which is UEqn's source expression, and the one a port that shares the two is left with. It
        // converges; it converges to the wrong balance at the interface.
        std::vector<scalar> withP(nFaces);
        for (label f = 0; f < nFaces; ++f)
            withP[f] = (stf[f] - ghf[f]*snRho[f] - snP[f]) * rAUf[f] * g.magSf()[f];
        scalar d = 0, scale = 0;
        for (label f = 0; f < nFaces; ++f)
        {
            d     = std::fmax(d, std::fabs(withP[f] - hostPhig[f]));
            scale = std::fmax(scale, std::fabs(hostPhig[f]));
        }
        std::printf("  ...and carrying snGrad(p_rgh) into it too would move phig by %.3e of %.3e\n",
                    (double)d, (double)scale);
        check("the pressure gradient is IMPLICIT here, and including it is a different flux",
              d > scalar(0.5)*scale);
    }

    // ---- 2. interpolate(rho*rAU) is not interpolate(rho)*interpolate(rAU) -------------------------
    {
        std::vector<scalar> hostOut;
        ifm::rhoRAUf(rho, rAU, m, g, hostOut);
        DeviceBuffer<scalar> dOut;
        deviceRhoRAUf(dm, dRho, dRAU, dOut);
        std::vector<scalar> got;
        dOut.copyTo(got);
        std::printf("  interpolate(rho*rAU): worst |device - host| = %.3e\n",
                    (double)worst(got, hostOut));
        check("the interpolated product matches the host bit for bit", worst(got, hostOut) == scalar(0));

        // THE CONTROL FOR NOTE 2: two interpolations multiplied, the obvious economy.
        scalar d = 0, scale = 0;
        for (label f = 0; f < nIf; ++f)
        {
            const scalar w = g.weights()[f];
            const label o = m.owner()[f], n = m.neighbour()[f];
            const scalar twice = (w*rho[o] + (scalar(1) - w)*rho[n])
                               * (w*rAU[o] + (scalar(1) - w)*rAU[n]);
            d     = std::fmax(d, std::fabs(twice - hostOut[f]));
            scale = std::fmax(scale, std::fabs(hostOut[f]));
        }
        // rAU = dt/rho, so rho*rAU is EXACTLY dt in every cell and the interpolated product is dt on
        // every face -- a value with no numerical error in it at all. The two-interpolation form is
        // 500.5 * 5.005e-04 at an interface face, i.e. 250x too large, and that is the whole content
        // of note 2 stated as a number.
        std::printf("  ...against interpolate(rho)*interpolate(rAU): %.3e of %.3e (rho*rAU is dt "
                    "exactly, so the right answer is flat)\n", (double)d, (double)scale);
        check("interpolation does not commute with multiplication across the interface",
              d > scalar(0.1)*scale);
    }

    // ---- 3. U = HbyA + rAU*reconstruct((phig - flux)/rAUf) ---------------------------------------
    {
        std::vector<vector> HbyA(static_cast<std::size_t>(nC));
        std::vector<scalar> hx(nC), hy(nC), hz(nC);
        for (label c = 0; c < nC; ++c)
        {
            HbyA[c] = vector{scalar(0.1)*g.C()[c].x, scalar(-0.05)*g.C()[c].y, scalar(0.02)};
            hx[c] = HbyA[c].x; hy[c] = HbyA[c].y; hz[c] = HbyA[c].z;
        }
        std::vector<scalar> ffInt(static_cast<std::size_t>(nIf)), rfInt(static_cast<std::size_t>(nIf));
        for (label f = 0; f < nIf; ++f)
        {
            ffInt[f] = scalar(1e-3)*std::sin(scalar(0.4)*scalar(f)) * g.magSf()[f];
            rfInt[f] = rAUf[f];
        }
        std::vector<std::vector<scalar>> ffBndH(fvp.size()), rfBndH(fvp.size());
        std::vector<scalar> ffBnd, rfBnd;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label gf = fvp[pi].start + i;
                const scalar v = scalar(1e-3)*std::sin(scalar(0.4)*scalar(gf)) * g.magSf()[gf];
                ffBndH[pi].push_back(v);
                rfBndH[pi].push_back(rAUf[gf]);
                ffBnd.push_back(v);
                rfBnd.push_back(rAUf[gf]);
            }
        }
        std::vector<vector> hostU;
        ifm::correctVelocity(HbyA, rAU, ffInt, rfInt, ffBndH, rfBndH, m, g, fvp, hostU);

        DeviceBuffer<scalar> dHx(hx), dHy(hy), dHz(hz), dFf(ffInt), dRf(rfInt),
                             dFfB(ffBnd), dRfB(rfBnd), dUx, dUy, dUz;
        deviceCorrectVelocity(dm, dHx, dHy, dHz, dRAU, dFf, dRf, dFfB, dRfB, dUx, dUy, dUz);
        std::vector<scalar> ux, uy, uz;
        dUx.copyTo(ux);
        dUy.copyTo(uy);
        dUz.copyTo(uz);
        scalar w3 = 0, scale = 0;
        for (label c = 0; c < nC; ++c)
        {
            w3 = std::fmax(w3, std::fmax(std::fabs(ux[c] - hostU[c].x),
                           std::fmax(std::fabs(uy[c] - hostU[c].y), std::fabs(uz[c] - hostU[c].z))));
            scale = std::fmax(scale, std::fabs(hostU[c].x));
        }
        std::printf("  U correction: worst |device - host| = %.3e of %.3e\n",
                    (double)w3, (double)scale);
        check("the velocity correction matches the host", w3 < scalar(1e-13)*scale);

        // THE CONTROL FOR NOTE 3: reconstruct(f) without the division, then no rAU outside -- the two
        // forms a uniform-rAU fixture cannot tell apart.
        DeviceBuffer<scalar> rx, ry, rz;
        deviceReconstruct(dm, dFf, dFfB, rx, ry, rz);
        std::vector<scalar> nx;
        rx.copyTo(nx);
        scalar d = 0;
        for (label c = 0; c < nC; ++c) d = std::fmax(d, std::fabs((hx[c] + nx[c]) - ux[c]));
        std::printf("  ...against reconstruct(f) with no rAUf division and no rAU outside: %.3e\n",
                    (double)d);
        check("dividing by rAUf inside and multiplying by rAU outside is not the same operator on a "
              "non-uniform rAU", d > scalar(1e-6));
    }

    // ---- 4. ddtCorr: the limiter, and the zeroing where U fixes a value ---------------------------
    {
        GeometricField<vector> U;
        U.internal.assign(static_cast<std::size_t>(nC), vector{0,0,0});
        for (const FvPatch& q : fvp)
        {
            if (q.type == "wall")
                U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                    q, true, vector{0,0,0}, std::vector<vector>{}));
            else
                U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
        }
        U.evaluateBoundary();

        std::vector<vector> UOld(static_cast<std::size_t>(nC));
        std::vector<scalar> uox(nC), uoy(nC), uoz(nC);
        for (label c = 0; c < nC; ++c)
        {
            UOld[c] = vector{scalar(0.4)*std::sin(g.C()[c].x), scalar(0.2)*g.C()[c].y, scalar(-0.1)};
            uox[c] = UOld[c].x; uoy[c] = UOld[c].y; uoz[c] = UOld[c].z;
        }
        SurfaceScalarField phiOld;
        phiOld.internal.resize(static_cast<std::size_t>(nIf));
        std::vector<scalar> phiOldBnd;
        for (label f = 0; f < nIf; ++f)
        {
            // deliberately NOT the interpolated flux -- phi and U are separate state, and the whole
            // point of ddtCorr is the part of phi that has no cell-centred representation
            const scalar w = g.weights()[f];
            const label o = m.owner()[f], n = m.neighbour()[f];
            const vector& S = g.Sf()[f];
            const scalar interp = (w*UOld[o].x + (1-w)*UOld[n].x)*S.x
                                + (w*UOld[o].y + (1-w)*UOld[n].y)*S.y
                                + (w*UOld[o].z + (1-w)*UOld[n].z)*S.z;
            phiOld.internal[f] = interp * (scalar(1) + scalar(0.3)*std::sin(scalar(f)));
        }
        phiOld.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label gf = fvp[pi].start + i;
                const vector& S = g.Sf()[gf];
                const vector& uo = UOld[fvp[pi].faceCells[i]];
                const scalar v = (uo.x*S.x + uo.y*S.y + uo.z*S.z)
                               * (scalar(1) + scalar(0.3)*std::sin(scalar(gf)));
                phiOld.boundary[pi].push_back(v);
                phiOldBnd.push_back(v);
            }

        ifm::DdtCorrInput in;
        in.phiOld = &phiOld;
        in.UOld = &UOld;
        in.ddtPhiCoeff = -1;
        in.deltaT = dt;
        SurfaceScalarField hostOut;
        ifm::ddtCorr(in, U, m, g, fvp, hostOut);

        std::vector<int> fixes;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                fixes.push_back(U.boundary[pi]->fixesValue() ? 1 : 0);

        DeviceBuffer<scalar> dPhiOld(phiOld.internal), dPhiOldB(phiOldBnd),
                             dUox(uox), dUoy(uoy), dUoz(uoz), dOutI, dOutB;
        DeviceBuffer<int> dFixes(fixes);
        deviceDdtCorr(dm, dPhiOld, dPhiOldB, dUox, dUoy, dUoz, dFixes, scalar(-1), dt, dOutI, dOutB);
        std::vector<scalar> gi, gb;
        dOutI.copyTo(gi);
        dOutB.copyTo(gb);

        std::vector<scalar> hb;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (scalar v : hostOut.boundary[pi]) hb.push_back(v);

        std::printf("  ddtCorr: internal worst %.3e, boundary worst %.3e\n",
                    (double)worst(gi, hostOut.internal), (double)worst(gb, hb));
        check("ddtCorr matches the host, internal faces and boundary alike",
              worst(gi, hostOut.internal) == scalar(0) && worst(gb, hb) == scalar(0));

        // THE CONTROL FOR NOTE 4a: a constant coefficient of 1, which applies the correction hardest
        // where OpenFOAM applies it least.
        DeviceBuffer<scalar> cI, cB;
        deviceDdtCorr(dm, dPhiOld, dPhiOldB, dUox, dUoy, dUoz, dFixes, scalar(1), dt, cI, cB);
        std::vector<scalar> ci;
        cI.copyTo(ci);
        scalar d = 0, scale = 0;
        for (label f = 0; f < nIf; ++f)
        {
            d     = std::fmax(d, std::fabs(ci[f] - gi[f]));
            scale = std::fmax(scale, std::fabs(ci[f]));
        }
        std::printf("  ...with a CONSTANT coefficient of 1 instead of the limiter: %.3e of %.3e\n",
                    (double)d, (double)scale);
        check("the default coefficient is a limiter, and a constant 1 is a different correction",
              d > scalar(0.1)*scale);

        // NOTE 4b: zero on every wall, which fixes U's value
        scalar onWalls = 0, elsewhere = 0;
        label off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
                (U.boundary[pi]->fixesValue() ? onWalls : elsewhere) =
                    std::fmax(U.boundary[pi]->fixesValue() ? onWalls : elsewhere,
                              std::fabs(gb[off + i]));
            off += fvp[pi].size;
        }
        std::printf("  ...and on the value-fixing patches |ddtCorr| = %.3e, elsewhere %.3e\n",
                    (double)onWalls, (double)elsewhere);
        check("ddtCorr is zero where U fixes a value", onWalls == scalar(0));
        check("...and is not zero where it does not, so that is not vacuous", elsewhere > scalar(1e-6));
    }

    // ---- 5. p = p_rgh + rho*gh --------------------------------------------------------------------
    {
        std::vector<scalar> hostP;
        ifm::staticPressure(p_rgh, rho, gh, hostP);
        DeviceBuffer<scalar> dP;
        deviceStaticPressure(static_cast<int>(nC), dPrgh, dRho, dGh, dP);
        std::vector<scalar> got;
        dP.copyTo(got);
        std::printf("  p = p_rgh + rho*gh: worst %.3e\n", (double)worst(got, hostP));
        check("the static pressure matches the host bit for bit", worst(got, hostP) == scalar(0));
    }

    // ---- 6. THE p_rgh MATRIX: fvm::laplacian(rAUf, p_rgh) == fvc::div(phiHbyA) --------------------
    // Compared as a MATRIX -- diagonal, both off-diagonals and the source -- and not as a solved field,
    // where the linear solver's tolerance would absorb any difference. Two arms beneath it measure
    // things a solved field could not show at all: the sign of the == and what setReference does.
    {
        GeometricField<scalar> prgh;
        prgh.internal = p_rgh;
        for (const FvPatch& q : fvp)
            prgh.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        prgh.evaluateBoundary();

        // rAUf on the internal faces, from the physical rAU = dt/rho
        std::vector<scalar> rAUfInt(static_cast<std::size_t>(nIf));
        for (label f = 0; f < nIf; ++f)
        {
            const scalar w = g.weights()[f];
            rAUfInt[f] = w*rAU[m.owner()[f]] + (scalar(1) - w)*rAU[m.neighbour()[f]];
        }

        // phiHbyA: a flux that is NOT divergence-free, so the source is not uniformly zero
        std::vector<scalar> phiHInt(static_cast<std::size_t>(nIf)), phiHBnd;
        for (label f = 0; f < nIf; ++f)
            phiHInt[f] = scalar(0.4)*std::sin(scalar(0.31)*scalar(f)) * g.magSf()[f];
        std::vector<std::vector<scalar>> phiHBndH(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label gf = fvp[pi].start + i;
                const scalar v = scalar(0.4)*std::sin(scalar(0.31)*scalar(gf)) * g.magSf()[gf];
                phiHBndH[pi].push_back(v);
                phiHBnd.push_back(v);
            }

        SurfaceScalarField phiH;
        phiH.internal = phiHInt;
        phiH.boundary = phiHBndH;

        // fvm::laplacian takes the WHOLE surface field: the boundary half of rAUf is what sets the
        // patches' contribution to the diagonal, and a matrix built from the internal faces alone is
        // a different one at every boundary cell.
        SurfaceScalarField rAUfField;
        rAUfField.internal = rAUfInt;
        rAUfField.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                rAUfField.boundary[pi].push_back(rAU[fvp[pi].faceCells[i]]);

        FvScalarMatrix hostPe =
            fvm::laplacian<scalar>(rAUfField, prgh, m, g, fvp, /*corrected=*/false);
        const std::vector<scalar> divH = fvc::div(phiH, m, g, fvp);
        for (label c = 0; c < nC; ++c) hostPe.source[c] += divH[c] * g.V()[c];

        DeviceBuffer<scalar> dRAUf(rAUfInt), dPhiHI(phiHInt), dPhiHB(phiHBnd);
        DevicePressureMatrix P;
        deviceInterAssemblePEqn(dm, dRAUf, dPhiHI, dPhiHB, /*needReference=*/false, 0, scalar(0), P);
        std::vector<scalar> dd, du, dl, ds;
        P.diag.copyTo(dd);
        P.upper.copyTo(du);
        P.lower.copyTo(dl);
        P.source.copyTo(ds);

        auto rel = [&](const std::vector<scalar>& a, const std::vector<scalar>& b, scalar& sc)
        {
            scalar w = 0;
            sc = 0;
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
            {
                w  = std::fmax(w, std::fabs(a[i] - b[i]));
                sc = std::fmax(sc, std::fabs(b[i]));
            }
            return w;
        };
        scalar s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        const scalar wd = rel(dd, hostPe.diag,   s1);
        const scalar wu = rel(du, hostPe.upper,  s2);
        const scalar wl = rel(dl, hostPe.lower,  s3);
        const scalar ws = rel(ds, hostPe.source, s4);
        std::printf("  p_rgh matrix: diag %.3e of %.3e, upper %.3e of %.3e, lower %.3e of %.3e, "
                    "source %.3e of %.3e\n",
                    (double)wd, (double)s1, (double)wu, (double)s2,
                    (double)wl, (double)s3, (double)ws, (double)s4);
        check("the laplacian matches the host", wd <= scalar(1e-14)*s1 && wu <= scalar(1e-14)*s2
                                             && wl <= scalar(1e-14)*s3);
        check("...and so does the divergence source", ws <= scalar(1e-13)*s4);
        check("...and the source is not zero, so fvc::div(phiHbyA) is really in it", s4 > scalar(1e-6));

        // THE SIGN OF ==. fvMatrix::operator== is source += V*R, a PLUS. rhoSimpleFoam's momentum path
        // carries the minus inside R = -grad(p), so both conventions live in this tree and the wrong
        // one here still converges -- to a pressure that drives the flow backwards.
        scalar flipped = 0;
        for (label c = 0; c < nC; ++c)
            flipped = std::fmax(flipped, std::fabs((-divH[c]*g.V()[c]) - hostPe.source[c]));
        std::printf("  ...with the == taken as a MINUS instead: %.3e of %.3e\n",
                    (double)flipped, (double)s4);
        check("the sign of fvMatrix::operator== is a plus, and a minus is a different equation",
              flipped > s4);

        // setReference DOUBLES the diagonal entry rather than replacing the row.
        DevicePressureMatrix R;
        deviceInterAssemblePEqn(dm, dRAUf, dPhiHI, dPhiHB, /*needReference=*/true,
                                /*pRefCell=*/3, /*pRefValue=*/scalar(7), R);
        std::vector<scalar> rd, rs;
        R.diag.copyTo(rd);
        R.source.copyTo(rs);
        std::printf("  setReference at cell 3: diag %.6g -> %.6g, source %.6g -> %.6g\n",
                    (double)dd[3], (double)rd[3], (double)ds[3], (double)rs[3]);
        check("setReference DOUBLES the diagonal entry", rd[3] == scalar(2)*dd[3]);
        check("...and adds diag*refValue to the source, the ORIGINAL diag and not the doubled one",
              std::fabs(rs[3] - (ds[3] + dd[3]*scalar(7))) <= scalar(1e-12)*std::fabs(rs[3]));
        int nOther = 0;
        for (label c = 0; c < nC; ++c) if (c != 3 && rd[c] != dd[c]) ++nOther;
        check("...and touches no other cell", nOther == 0);
    }

    std::printf("test_device_inter_peqn: %d failures\n", failures);
    return failures ? 1 : 0;
}
