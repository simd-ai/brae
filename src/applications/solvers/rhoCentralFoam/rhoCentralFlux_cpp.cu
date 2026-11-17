// rhoCentralFlux_cpp.cu -- the central-upwind flux assembly on the host. See the header.
//
// Shared infrastructure it runs on:
//   fvc::gaussGrad (fvc.cu)                         the limiters' gradients, OpenFOAM's gaussGrad::calcGrad order
//   cpu::limitedSchemes::detail (limitedSchemes_cpp) NVDTVD/NVDVTVDV r, the vanLeer limiter, the limited weight
//   centralFaceFlux (rhoCentralFaceFlux.cuh)        the per-face flux both arms evaluate
#include "rhoCentralFlux_cpp.cuh"
#include "rhoCentralFaceFlux.cuh"
#include "rhoCentralThermo.cuh"

#include "fvc.cuh"
#include "limitedSchemes_cpp.cuh"

#include <algorithm>
#include <cmath>

namespace brae {
namespace cpu {
namespace rhoCentral {

using namespace brae::rhoCentral;

namespace {

// fvc::gaussGrad's array form takes the patch values patch by patch
std::vector<vector> grad(const Case& cs, const std::vector<scalar>& vf, const std::vector<scalar>& flat)
{
    std::vector<std::vector<scalar>> bnd(cs.patches.size());
    std::size_t k = 0;
    for (std::size_t pi = 0; pi < cs.patches.size(); ++pi)
    {
        bnd[pi].assign(flat.begin() + k, flat.begin() + k + cs.patches[pi].size);
        k += cs.patches[pi].size;
    }
    return fvc::gaussGrad(vf, bnd, cs.m, cs.g, cs.patches);
}

// interpolate(vf, dir, "reconstruct(...)") on an internal face (directionInterpolate.H, LimitedScheme.C):
// weight = limiter*CDweight + (1 - limiter)*pos0(dir), value = weight*(P - N) + N
inline scalar reconstructScalar(scalar dir, scalar cdw, scalar qP, scalar qN, const vector& gP, const vector& gN,
                                const vector& d)
{
    namespace ls = cpu::limitedSchemes::detail;
    const scalar w = ls::blend(ls::vanLeerLimiter(ls::rScalar(dir, qP, qN, gP, gN, d)), cdw, dir);
    return w*(qP - qN) + qN;
}

// grad(rhoU) in OpenFOAM's packing, gradc_ij = d(rhoU_j)/d(x_i), from the three component gradients
inline tensor gradTensor(const std::vector<vector> g[3], label c)
{
    return tensor{g[0][c].x, g[1][c].x, g[2][c].x,
                  g[0][c].y, g[1][c].y, g[2][c].y,
                  g[0][c].z, g[1][c].z, g[2][c].z};
}

} // namespace


HostFluxes centralFluxes(const Case& cs, const State& s)
{
    namespace ls = cpu::limitedSchemes::detail;
    const GasCoeffs& gas = cs.gas;
    const label nC = cs.nC, nIF = cs.nIF, nB = cs.nB;
    const std::vector<label>& own = cs.m.owner();
    const std::vector<label>& nei = cs.m.neighbour();
    const std::vector<scalar>& w = cs.g.weights();
    const std::vector<vector>& Sf = cs.g.Sf();
    const std::vector<scalar>& magSf = cs.g.magSf();
    const std::vector<scalar>& V = cs.g.V();
    const std::vector<vector>& C = cs.g.C();
    HostFluxes out;

    // ---- rPsi = 1/psi and c = sqrt(Cp/Cv*rPsi), cells and patches (rhoCentralFoam.C:107, 135) -----------------
    const scalar CpByCv = gasCp(gas)/gasCv(gas);
    std::vector<scalar> rPsi(nC), c(nC), rPsiB(nB), cB(nB);
    for (label i = 0; i < nC; ++i)
    {
        rPsi[i] = 1.0/s.psi[i];
        c[i] = std::sqrt(CpByCv*rPsi[i]);
    }
    for (label k = 0; k < nB; ++k)
    {
        rPsiB[k] = 1.0/s.psiB[k];
        cB[k] = std::sqrt(CpByCv*rPsiB[k]);
    }

    // ---- the limiters' gradients: grad(rho), grad(rhoU), grad(rPsi), grad(e), grad(c) (LimitedScheme.C:56-59) ----
    std::vector<scalar> rhoUc[3], rhoUbc[3];
    for (int j = 0; j < 3; ++j)
    {
        rhoUc[j].resize(nC);
        rhoUbc[j].resize(nB);
    }
    for (label i = 0; i < nC; ++i)
    {
        rhoUc[0][i] = s.rhoU[i].x;
        rhoUc[1][i] = s.rhoU[i].y;
        rhoUc[2][i] = s.rhoU[i].z;
    }
    for (label k = 0; k < nB; ++k)
    {
        rhoUbc[0][k] = s.rhoUB[k].x;
        rhoUbc[1][k] = s.rhoUB[k].y;
        rhoUbc[2][k] = s.rhoUB[k].z;
    }
    const std::vector<vector> gRho = grad(cs, s.rho, s.rhoB);
    const std::vector<vector> gRhoU[3] = {grad(cs, rhoUc[0], rhoUbc[0]), grad(cs, rhoUc[1], rhoUbc[1]),
                                          grad(cs, rhoUc[2], rhoUbc[2])};
    const std::vector<vector> gRPsi = grad(cs, rPsi, rPsiB);
    const std::vector<vector> gE = grad(cs, s.e, s.eB);
    const std::vector<vector> gC = grad(cs, c, cB);

    // ---- directed interpolation and the central-upwind flux, internal faces --------------------------------------
    std::vector<scalar> phi(nIF), phiEp(nIF), amaxSf(nIF);
    std::vector<vector> phiUp(nIF);
    for (label f = 0; f < nIF; ++f)
    {
        const label P = own[f], N = nei[f];
        const vector d = C[N] - C[P];
        const scalar cdw = w[f];
        const tensor gUP = gradTensor(gRhoU, P), gUN = gradTensor(gRhoU, N);
        FaceStates in;
        for (int k = 0; k < 2; ++k)
        {
            const scalar dir = (k == 0) ? 1.0 : -1.0;   // the `pos` and `neg` surface fields, createFields.H
            in.rho[k] = reconstructScalar(dir, cdw, s.rho[P], s.rho[N], gRho[P], gRho[N], d);
            const scalar wU = ls::blend(ls::vanLeerLimiter(ls::rVector(dir, s.rhoU[P], s.rhoU[N], gUP, gUN, d)), cdw,
                                        dir);
            in.rhoU[k] = vector{wU*(s.rhoU[P].x - s.rhoU[N].x) + s.rhoU[N].x,
                                wU*(s.rhoU[P].y - s.rhoU[N].y) + s.rhoU[N].y,
                                wU*(s.rhoU[P].z - s.rhoU[N].z) + s.rhoU[N].z};
            in.rPsi[k] = reconstructScalar(dir, cdw, rPsi[P], rPsi[N], gRPsi[P], gRPsi[N], d);
            in.e[k] = reconstructScalar(dir, cdw, s.e[P], s.e[N], gE[P], gE[N], d);
            in.c[k] = reconstructScalar(dir, cdw, c[P], c[N], gC[P], gC[N], d);
        }
        const FaceFlux o = centralFaceFlux(in, Sf[f].x, Sf[f].y, Sf[f].z, magSf[f], cs.tadmor, controlNone);
        phi[f] = o.phi;
        phiUp[f] = vector{o.phiUx, o.phiUy, o.phiUz};
        phiEp[f] = o.phiEp;
        amaxSf[f] = o.amaxSf;
    }
    // ...boundary faces: an uncoupled patch's interpolate is the patch value in both directions
    std::vector<scalar> phiB(nB, 0), phiEpB(nB, 0), amaxSfB(nB, 0);
    std::vector<vector> phiUpB(nB, vector{0, 0, 0});
    for (label k = 0; k < nB; ++k)
    {
        if (cs.bEmpty[k]) continue;
        FaceStates in;
        for (int j = 0; j < 2; ++j)
        {
            in.rho[j] = s.rhoB[k];
            in.rhoU[j] = s.rhoUB[k];
            in.rPsi[j] = rPsiB[k];
            in.e[j] = s.eB[k];
            in.c[j] = cB[k];
        }
        const vector& S = Sf[nIF + k];
        const FaceFlux o = centralFaceFlux(in, S.x, S.y, S.z, magSf[nIF + k], cs.tadmor, controlNone);
        phiB[k] = o.phi;
        phiUpB[k] = vector{o.phiUx, o.phiUy, o.phiUz};
        phiEpB[k] = o.phiEp;
        amaxSfB[k] = o.amaxSf;
        out.bndMass += o.phi;
        out.bndMom += phiUpB[k];
        out.bndEnergy += o.phiEp;
    }

    // ---- centralCourantNo.H -----------------------------------------------------------------------------------------
    if (nIF)
    {
        std::vector<scalar> sumA(nC, 0);
        for (label f = 0; f < nIF; ++f)
        {
            sumA[own[f]] += amaxSf[f];
            sumA[nei[f]] += amaxSf[f];
        }
        for (label k = 0; k < nB; ++k)
            if (!cs.bEmpty[k]) sumA[cs.bCell[k]] += amaxSfB[k];
        scalar mx = -1e300, sA = 0, sV = 0;
        for (label i = 0; i < nC; ++i)
        {
            mx = std::max(mx, sumA[i]/V[i]);
            sA += sumA[i];
            sV += V[i];
        }
        out.CoNum = 0.5*mx*cs.deltaT;
        out.meanCoNum = 0.5*(sA/sV)*cs.deltaT;
    }

    // ---- fvc::div = surfaceIntegrate (fvcSurfaceIntegrate.C:45-78) ---------------------------------------------------
    out.divPhi.assign(nC, 0);
    out.divPhiEp.assign(nC, 0);
    out.divPhiUp.assign(nC, vector{0, 0, 0});
    for (label f = 0; f < nIF; ++f)
    {
        out.divPhi[own[f]] += phi[f];
        out.divPhi[nei[f]] -= phi[f];
        out.divPhiUp[own[f]] += phiUp[f];
        out.divPhiUp[nei[f]] -= phiUp[f];
        out.divPhiEp[own[f]] += phiEp[f];
        out.divPhiEp[nei[f]] -= phiEp[f];
    }
    for (label k = 0; k < nB; ++k)
    {
        if (cs.bEmpty[k]) continue;
        out.divPhi[cs.bCell[k]] += phiB[k];
        out.divPhiUp[cs.bCell[k]] += phiUpB[k];
        out.divPhiEp[cs.bCell[k]] += phiEpB[k];
    }
    for (label i = 0; i < nC; ++i)
    {
        out.divPhi[i] /= V[i];
        out.divPhiUp[i] = out.divPhiUp[i]/V[i];
        out.divPhiEp[i] /= V[i];
    }
    return out;
}

} // namespace rhoCentral
} // namespace cpu
} // namespace brae
