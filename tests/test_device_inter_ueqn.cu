// interFoam's momentum predictor on the device: fvc::reconstruct, the two-rho ddt, and the face force.
//
// THREE ORACLES, and the first one is an identity with no tolerance in it.
//
//   RECONSTRUCT IS PINNED BY ITS DEFINING PROPERTY: for any uniform vector V, reconstruct(V & Sf) == V
//   exactly, on any closed cell whatever its shape. That is what reconstruct is FOR. Both ways to get
//   it wrong break it and neither looks wrong: taking the neighbour with a MINUS (the div convention
//   every other gather in this tree uses) leaves the tensor symmetric and invertible and merely
//   returns a different vector; dropping the SfHat normalisation reweights each face by magSf, which
//   is invisible on a cube because every face has the same area. The fixture is therefore anisotropic
//   and arm 1c computes the div-convention answer here rather than asserting it is wrong.
//
//   THE ddt IS PINNED BY ITS TWO DENSITY FIELDS. On a converged smooth field rho and rho.oldTime()
//   agree and no ordinary gate separates them. Arm 2 builds the fixture a VoF interface actually
//   presents -- half the cells at rho = 1000 and rho_old = 1, the other half reversed -- and the
//   control passes rho twice, which is the port a careless reading writes.
//
//   THE FACE FORCE IS PINNED AGAINST THE HOST, and then against the physics: on a hydrostatic start
//   the three terms cancel everywhere except at the interface, which is the whole content of the p_rgh
//   formulation. Arm 3b measures that the reconstructed force is zero in the bulk and not at the
//   interface -- a version that put rho*g in the source instead would be uniformly non-zero and would
//   still converge.
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc_reconstruct_cpp.cuh"
#include "inter_ueqn_cpp.cuh"
#include "device_fvc_reconstruct.cuh"
#include "device_inter_ueqn.cuh"
#include "device_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

using namespace brae;
namespace fr  = brae::cpu::fvcReconstruct;
namespace ifm = brae::cpu::interFoam;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
}   // namespace

int main()
{
    std::printf("== interFoam momentum predictor: device\n");
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    // ANISOTROPIC cells, 2:1:0.5, so the three face areas take three distinct values while the mesh
    // stays orthogonal. On a unit cube the SfHat normalisation is invisible.
    const label N = 8;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N, scalar(0), scalar(2), scalar(1), scalar(0.5));
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    label nBf = 0;
    for (const FvPatch& q : fvp) nBf += q.size;

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // ---- 1. fvc::reconstruct -----------------------------------------------------------------------
    const vector V{scalar(2.5), scalar(-1.25), scalar(0.75)};
    std::vector<scalar> ssfInt(static_cast<std::size_t>(nIf)), ssfBnd;
    for (label f = 0; f < nIf; ++f)
    {
        const vector& S = g.Sf()[f];
        ssfInt[f] = V.x*S.x + V.y*S.y + V.z*S.z;
    }
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const vector& S = g.Sf()[fvp[pi].start + i];
            ssfBnd.push_back(V.x*S.x + V.y*S.y + V.z*S.z);
        }

    DeviceBuffer<scalar> dSsfInt(ssfInt), dSsfBnd(ssfBnd), rx, ry, rz;
    deviceReconstruct(dm, dSsfInt, dSsfBnd, rx, ry, rz);
    if (cudaDeviceSynchronize() != cudaSuccess)
    { std::printf("  FAIL: kernels did not complete\n"); return 1; }
    std::vector<scalar> hx, hy, hz;
    rx.copyTo(hx);
    ry.copyTo(hy);
    rz.copyTo(hz);

    // 1a: THE IDENTITY
    {
        scalar worst = 0;
        for (label c = 0; c < nC; ++c)
            worst = std::fmax(worst, std::fmax(std::fabs(hx[c] - V.x),
                              std::fmax(std::fabs(hy[c] - V.y), std::fabs(hz[c] - V.z))));
        std::printf("  reconstruct(V & Sf) - V on 2:1:0.5 cells: worst %.3e\n", (double)worst);
        check("reconstruct recovers a uniform V exactly, interior and boundary cells alike",
              worst <= scalar(1e-13));
    }

    // 1b: against the HOST, on a flux that is NOT a uniform field -- the identity alone would pass for
    // any operator that happened to be exact on constants.
    {
        std::vector<scalar> rough(static_cast<std::size_t>(nIf));
        for (label f = 0; f < nIf; ++f)
            rough[f] = std::sin(scalar(3)*g.Cf()[f].x) * std::cos(scalar(2)*g.Cf()[f].y)
                     + scalar(0.5)*g.Cf()[f].z;
        std::vector<scalar> roughB;
        std::vector<fr::BoundaryFace> hb;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label gf = fvp[pi].start + i;
                const scalar s = std::sin(scalar(3)*g.Cf()[gf].x) * std::cos(scalar(2)*g.Cf()[gf].y)
                               + scalar(0.5)*g.Cf()[gf].z;
                roughB.push_back(s);
                if (fvp[pi].type != "empty")
                    hb.push_back(fr::BoundaryFace{static_cast<int>(fvp[pi].faceCells[i]), g.Sf()[gf], s});
            }
        std::vector<int> own(m.owner().begin(), m.owner().begin() + nIf);
        std::vector<int> nei(m.neighbour().begin(), m.neighbour().begin() + nIf);
        std::vector<vector> SfInt(g.Sf().begin(), g.Sf().begin() + nIf);
        std::vector<vector> hostOut;
        fr::reconstruct(static_cast<int>(nC), own, nei, SfInt, rough, hb, hostOut);

        DeviceBuffer<scalar> dI(rough), dB(roughB), dx, dy, dz;
        deviceReconstruct(dm, dI, dB, dx, dy, dz);
        std::vector<scalar> ox, oy, oz;
        dx.copyTo(ox);
        dy.copyTo(oy);
        dz.copyTo(oz);
        scalar worst = 0, scale = 0;
        for (label c = 0; c < nC; ++c)
        {
            worst = std::fmax(worst, std::fmax(std::fabs(ox[c] - hostOut[c].x),
                              std::fmax(std::fabs(oy[c] - hostOut[c].y), std::fabs(oz[c] - hostOut[c].z))));
            scale = std::fmax(scale, std::fabs(hostOut[c].x));
        }
        std::printf("  a non-uniform flux, device vs host: worst %.3e of %.3e\n",
                    (double)worst, (double)scale);
        check("...and matches the host on a flux that is not a uniform field",
              worst < scalar(1e-13)*scale);
    }

    // 1c: THE SIGN CONTROL, computed here rather than asserted. surfaceSum adds to owner AND neighbour
    // with the same sign; the div convention negates the neighbour.
    {
        std::vector<tensor> T(static_cast<std::size_t>(nC), tensor{0,0,0,0,0,0,0,0,0});
        std::vector<vector> v(static_cast<std::size_t>(nC), vector{0,0,0});
        for (label f = 0; f < nIf; ++f)
        {
            fr::accumulate(g.Sf()[f],  ssfInt[f], T[m.owner()[f]],     v[m.owner()[f]]);
            fr::accumulate(g.Sf()[f], -ssfInt[f], T[m.neighbour()[f]], v[m.neighbour()[f]]);  // NEGATED
        }
        label off = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
                if (fvp[pi].type != "empty")
                    fr::accumulate(g.Sf()[fvp[pi].start + i], ssfBnd[off + i],
                                   T[fvp[pi].faceCells[i]], v[fvp[pi].faceCells[i]]);
            off += fvp[pi].size;
        }
        scalar worst = 0;
        for (label c = 0; c < nC; ++c)
        {
            const vector w = fr::dot(fr::inv(T[c]), v[c]);
            worst = std::fmax(worst, std::fmax(std::fabs(w.x - V.x),
                              std::fmax(std::fabs(w.y - V.y), std::fabs(w.z - V.z))));
        }
        std::printf("  ...with the neighbour NEGATED (the div convention): %.3e\n", (double)worst);
        check("the div convention breaks the identity, so arm 1a discriminates", worst > scalar(1e-2));
    }

    // ---- 2. fvm::ddt(rho, U): TWO DENSITY FIELDS ---------------------------------------------------
    // The fixture a VoF interface actually presents: half the cells water-now/air-before, the other
    // half the reverse. A converged smooth field would make rho and rho.oldTime() agree and this arm
    // vacuous.
    {
        std::vector<scalar> rho(static_cast<std::size_t>(nC)), rhoOld(static_cast<std::size_t>(nC));
        std::vector<scalar> uox(nC), uoy(nC), uoz(nC);
        for (label c = 0; c < nC; ++c)
        {
            const bool water = (c % 2) == 0;
            rho[c]    = water ? scalar(1000) : scalar(1);
            rhoOld[c] = water ? scalar(1)    : scalar(1000);      // the interface crossed this cell
            uox[c] = scalar(0.3)*scalar(c % 7) - scalar(1);
            uoy[c] = scalar(0.2)*scalar(c % 5);
            uoz[c] = scalar(-0.1)*scalar(c % 3);
        }
        const scalar dt = scalar(1e-3);

        // the host, through the shared FvVectorMatrix
        FvVectorMatrix M;
        M.diag.assign(static_cast<std::size_t>(nC), scalar(0));
        M.source.assign(static_cast<std::size_t>(nC), vector{0,0,0});
        std::vector<vector> UOld(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) UOld[c] = vector{uox[c], uoy[c], uoz[c]};
        ifm::addEulerDdtRhoU(M, rho, rhoOld, UOld, g.V(), dt);

        std::vector<scalar> zeroC(static_cast<std::size_t>(nC), scalar(0));
        DeviceBuffer<scalar> dRho(rho), dRhoOld(rhoOld), dUox(uox), dUoy(uoy), dUoz(uoz);
        DeviceBuffer<scalar> dDiag(zeroC), dSx(zeroC), dSy(zeroC), dSz(zeroC);
        deviceInterEulerDdtRhoU(dm, dRho, dRhoOld, dUox, dUoy, dUoz, dt, dDiag, dSx, dSy, dSz);
        std::vector<scalar> gd, gx2, gy2, gz2;
        dDiag.copyTo(gd);
        dSx.copyTo(gx2);
        dSy.copyTo(gy2);
        dSz.copyTo(gz2);

        scalar wd = 0, ws = 0, sScale = 0;
        for (label c = 0; c < nC; ++c)
        {
            wd = std::fmax(wd, std::fabs(gd[c] - M.diag[c]));
            ws = std::fmax(ws, std::fmax(std::fabs(gx2[c] - M.source[c].x),
                           std::fmax(std::fabs(gy2[c] - M.source[c].y),
                                     std::fabs(gz2[c] - M.source[c].z))));
            sScale = std::fmax(sScale, std::fabs(M.source[c].x));
        }
        std::printf("  ddt(rho,U): diag worst %.3e, source worst %.3e of %.3e\n",
                    (double)wd, (double)ws, (double)sScale);
        check("the device ddt matches the host's, diagonal and source", wd == scalar(0) && ws == scalar(0));

        // THE CONTROL: rho passed twice, which is what a careless reading of fvm::ddt(rho, U) writes.
        DeviceBuffer<scalar> cDiag(zeroC), cSx(zeroC), cSy(zeroC), cSz(zeroC);
        deviceInterEulerDdtRhoU(dm, dRho, dRho, dUox, dUoy, dUoz, dt, cDiag, cSx, cSy, cSz);
        std::vector<scalar> cx;
        cSx.copyTo(cx);
        scalar ratio = 0;
        for (label c = 0; c < nC; ++c)
            if (std::fabs(gx2[c]) > scalar(1e-30))
                ratio = std::fmax(ratio, std::fabs(cx[c]/gx2[c]));
        std::printf("  ...and with rho.oldTime() replaced by rho, the source is up to %.1fx out\n",
                    (double)ratio);
        check("rho.oldTime() in the source is load-bearing -- the density ratio, not a rounding choice",
              ratio > scalar(100));
    }

    // ---- 3. the face force, and what it looks like on a hydrostatic start -------------------------
    {
        const label nFaces = static_cast<label>(g.magSf().size());
        std::vector<scalar> stf(nFaces), ghf(nFaces), snRho(nFaces), snP(nFaces);
        for (label f = 0; f < nFaces; ++f)
        {
            stf[f]   = scalar(0.01)*std::sin(scalar(f));
            ghf[f]   = scalar(-9.81)*g.Cf()[f].y;
            snRho[f] = scalar(500)*std::cos(scalar(0.5)*scalar(f));
            snP[f]   = scalar(3)*std::sin(scalar(0.25)*scalar(f));
        }
        std::vector<scalar> hostOut;
        ifm::momentumSourceFlux(stf, ghf, snRho, snP, g.magSf(), hostOut);

        DeviceBuffer<scalar> dStf(stf), dGhf(ghf), dSnRho(snRho), dSnP(snP), dMagSf(g.magSf()), dOut;
        deviceMomentumSourceFlux(static_cast<int>(nFaces), dStf, dGhf, dSnRho, dSnP, dMagSf, dOut);
        std::vector<scalar> got;
        dOut.copyTo(got);
        scalar worst = 0, scale = 0;
        for (label f = 0; f < nFaces; ++f)
        {
            worst = std::fmax(worst, std::fabs(got[f] - hostOut[f]));
            scale = std::fmax(scale, std::fabs(hostOut[f]));
        }
        std::printf("  the face force: worst %.3e of %.3e\n", (double)worst, (double)scale);
        check("the momentum source flux matches the host bit for bit", worst == scalar(0));

        // 3b: THE p_rgh PROPERTY. A hydrostatic start -- p_rgh uniform, rho piecewise constant, alpha
        // sharp -- must leave the momentum predictor's body force IDENTICALLY ZERO away from the
        // interface. A port that put rho*g in the source instead would be uniformly non-zero there and
        // would still converge, to a water column accelerating under gravity twice over.
        std::vector<scalar> zInt(static_cast<std::size_t>(nIf), scalar(0)), zBnd;
        const scalar yInterface = scalar(4);
        for (label f = 0; f < nIf; ++f)
        {
            // snGrad(rho) is zero except across the one layer of faces the interface sits on
            const scalar yo = g.C()[m.owner()[f]].y, yn = g.C()[m.neighbour()[f]].y;
            const bool crosses = (yo < yInterface) != (yn < yInterface);
            zInt[f] = crosses ? (ghf[f] * scalar(-999) * scalar(1)) * g.magSf()[f] : scalar(0);
        }
        for (label b = 0; b < nBf; ++b) zBnd.push_back(scalar(0));
        DeviceBuffer<scalar> dzI(zInt), dzB(zBnd), fx, fy, fz;
        deviceReconstruct(dm, dzI, dzB, fx, fy, fz);
        std::vector<scalar> ax, ay, az;
        fx.copyTo(ax);
        fy.copyTo(ay);
        fz.copyTo(az);
        scalar bulk = 0, atInterface = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar mag = std::sqrt(ax[c]*ax[c] + ay[c]*ay[c] + az[c]*az[c]);
            const scalar dy  = std::fabs(g.C()[c].y - yInterface);
            if (dy < scalar(1.01)) atInterface = std::fmax(atInterface, mag);
            else                   bulk        = std::fmax(bulk, mag);
        }
        std::printf("  hydrostatic start: |body force| %.3e in the bulk, %.3e at the interface\n",
                    (double)bulk, (double)atInterface);
        check("the p_rgh body force is IDENTICALLY zero away from the interface", bulk == scalar(0));
        check("...and is not zero at it, so the fixture is not uniformly empty",
              atInterface > scalar(1));
    }

    std::printf("test_device_inter_ueqn: %d failures\n", failures);
    return failures ? 1 : 0;
}
