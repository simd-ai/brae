#include "cyclicAMI_cpp.cuh"
#include "limitedSchemes_cpp.cuh"

#include <cmath>

namespace brae {
namespace cpu {
namespace cyclicAMI {

std::vector<scalar> interpolate(const AMIInterface& a, const std::vector<scalar>& psi)
{
    const std::size_t n = a.ownCell.size();
    std::vector<scalar> out(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar s = 0;
        for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k) s += a.weight[k] * psi[a.nbrCell[k]];
        out[i] = s;
    }
    return out;
}


std::vector<vector> interpolateVec(const AMIInterface& a, const std::vector<vector>& U)
{
    const std::size_t n = a.ownCell.size();
    const tensor& T = a.forwardT;
    std::vector<vector> out(n, vector{0, 0, 0});
    for (std::size_t i = 0; i < n; ++i)
    {
        vector s{0, 0, 0};
        for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k)
        {
            const vector& v = U[a.nbrCell[k]];
            // forwardT applied BEFORE the weighted sum, matching amiInterpVecKernel. The transform is
            // constant per interface so the order does not change the value; it changes whether this is
            // a transcription of the kernel or a paraphrase of it.
            const vector r{T.xx*v.x + T.xy*v.y + T.xz*v.z,
                           T.yx*v.x + T.yy*v.y + T.yz*v.z,
                           T.zx*v.x + T.zy*v.y + T.zz*v.z};
            const scalar w = a.weight[k];
            s.x += w * (a.translational ? v.x : r.x);
            s.y += w * (a.translational ? v.y : r.y);
            s.z += w * (a.translational ? v.z : r.z);
        }
        out[i] = s;
    }
    return out;
}


std::vector<tensor> interpolateTensor(const AMIInterface& a, const std::vector<tensor>& G)
{
    const std::size_t n = a.ownCell.size();
    const tensor& T = a.forwardT;
    const scalar R[9] = {T.xx, T.xy, T.xz, T.yx, T.yy, T.yz, T.zx, T.zy, T.zz};
    std::vector<tensor> out(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar acc[9] = {0,0,0,0,0,0,0,0,0};
        for (label kk = a.srcOffset[i]; kk < a.srcOffset[i+1]; ++kk)
        {
            const tensor& q = G[a.nbrCell[kk]];
            const scalar g[9] = {q.xx, q.xy, q.xz, q.yx, q.yy, q.yz, q.zx, q.zy, q.zz};
            const scalar wk = a.weight[kk];
            if (a.translational)
            {
                for (int e = 0; e < 9; ++e) acc[e] += wk * g[e];
            }
            else
            {
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                    {
                        scalar sum = 0;
                        for (int x = 0; x < 3; ++x)
                            for (int y = 0; y < 3; ++y)
                                sum += R[3*r + x] * g[3*x + y] * R[3*c + y];   // (R G R^T)[r][c]
                        acc[3*r + c] += wk * sum;
                    }
            }
        }
        out[i] = tensor{acc[0],acc[1],acc[2], acc[3],acc[4],acc[5], acc[6],acc[7],acc[8]};
    }
    return out;
}


std::vector<scalar> limitedLinearVWeights(
    const AMIInterface&        a,
    const std::vector<scalar>& phi,
    const std::vector<vector>& U,
    const std::vector<tensor>& gradU,
    const scalar               k)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<vector> UN = interpolateVec(a, U);            // patchNeighbourField (rotated)
    const std::vector<tensor> GN = interpolateTensor(a, gradU);     // ...for the gradient, R G R^T

    // OpenFOAM's limiter reads gradc_ij = d(U_j)/d(x_i); brae's device packs (row, col) = (component,
    // derivative). Transpose BOTH sides here, once, so limitedSchemes sees exactly the packing its
    // NVDVTVDV transcription documents. A missing transpose here is not a small error on a sheared
    // cell -- it swaps d(U_x)/dy for d(U_y)/dx.
    auto tr = [](const tensor& q) -> tensor
    { return tensor{q.xx, q.yx, q.zx, q.xy, q.yy, q.zy, q.xz, q.yz, q.zz}; };

    std::vector<vector> UP(n);
    std::vector<tensor> GP(n), GNt(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        UP[i]  = U[a.ownCell[i]];                                   // patchInternalField
        GP[i]  = tr(gradU[a.ownCell[i]]);
        GNt[i] = tr(GN[i]);
    }
    return limitedSchemes::limitedLinearVWeightsCoupled(phi, a.weights, a.delta, UP, UN, GP, GNt, k);
}


std::vector<scalar> limitedLinearWeights(
    const AMIInterface&        a,
    const std::vector<scalar>& phi,
    const std::vector<scalar>& f,
    const std::vector<vector>& gradF,
    const scalar               k)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<scalar> fN = interpolate(a, f);          // a scalar is NOT rotated...
    const std::vector<vector> gN = interpolateVec(a, gradF);   // ...its gradient is
    std::vector<scalar> fP(n);
    std::vector<vector> gP(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        fP[i] = f[a.ownCell[i]];
        gP[i] = gradF[a.ownCell[i]];
    }
    return limitedSchemes::limitedLinearWeightsCoupled(phi, a.weights, a.delta, fP, fN, gP, gN, k);
}


std::vector<scalar> faceValue(const AMIInterface& a, const std::vector<scalar>& cell)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<scalar> nbr = interpolate(a, cell);
    std::vector<scalar> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a.weights[i] * cell[a.ownCell[i]] + (1.0 - a.weights[i]) * nbr[i];
    return out;
}


void assembleMomentum(
    const AMIInterface&        a,
    const std::vector<scalar>& nuEffCell,
    const std::vector<scalar>& phi,
    State&                     st,
    std::vector<scalar>&       diag,
    const std::vector<scalar>* wsch)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<scalar> nuN = interpolate(a, nuEffCell);
    st.ifCoeff.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        const label o = a.ownCell[i];
        // The face diffusivity is the INTERPOLATED one -- w*nuEff[own] + (1-w)*interp(nuEff) -- not the
        // owner cell's, exactly as an internal face takes fvc::interpolate(nuEff). Using the owner value
        // is the same class of error that put 90% of pitzDaily's epsilon residual on one patch.
        const scalar lap = (a.weights[i] * nuEffCell[o] + (1.0 - a.weights[i]) * nuN[i])
                         * a.deltaCoeffs[i] * a.magSf[i];
        const scalar p = phi[i];
        // OpenFOAM assembles a coupled patch with the DIV SCHEME's interpolation weights:
        //     internalCoeffs = phi*w,  boundaryCoeffs = -phi*(1-w)
        // and the solver applies -boundaryCoeffs against the neighbour, so the off-diagonal is phi*(1-w).
        // Upwind is the special case w = pos0(phi), giving max(phi,0) and min(phi,0) -- which is what
        // both interface assemblies hardcoded for EVERY case, including ones asking for limitedLinearV.
        const scalar ws = wsch ? (*wsch)[i] : (p > 0.0 ? scalar(1) : scalar(0));
        st.ifCoeff[i] = -lap + p * (1.0 - ws);
        diag[o]      += lap + p * ws;
    }
}


void assembleLaplacian(
    const AMIInterface&        a,
    const std::vector<scalar>& gammaCell,
    State&                     st,
    std::vector<scalar>&       diag,
    const bool                 addToDiag)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<scalar> gN = interpolate(a, gammaCell);
    st.ifCoeff.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        const label o = a.ownCell[i];
        const scalar c = (a.weights[i] * gammaCell[o] + (1.0 - a.weights[i]) * gN[i])
                       * a.deltaCoeffs[i] * a.magSf[i];
        // THE SIGN IS THE OPPOSITE OF assembleMomentum's, and that is not an inconsistency: the two
        // equations carry the laplacian with opposite signs. Momentum is
        //     fvm::div(phi,U) - fvm::laplacian(nuEff,U)          -> ifCoeff = -lap, diag += lap
        // while the pressure equation is
        //     fvm::laplacian(rAUf,p) == fvc::div(phiHbyA)        -> ifCoeff = +c,   diag -= c
        // This reference had it backwards on the first writing, copied across from the momentum case,
        // and the stage gate caught it on the first run -- which is the whole argument for having one.
        st.ifCoeff[i] = c;
        if (addToDiag) diag[o] -= c;
    }
}


void amul(
    const AMIInterface&        a,
    const State&               st,
    const std::vector<scalar>& psi,
    std::vector<scalar>&       Apsi)
{
    const std::vector<scalar> pn = interpolate(a, psi);
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        Apsi[a.ownCell[i]] += st.ifCoeff[i] * pn[i];
}


void addH(
    const AMIInterface&        a,
    const State&               st,
    const std::vector<scalar>& UN,
    const std::vector<scalar>& V,
    std::vector<scalar>&       H)
{
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
    {
        const label o = a.ownCell[i];
        H[o] -= st.ifCoeff[i] * UN[i] / V[o];
    }
}


void flux(const AMIInterface& a, const std::vector<vector>& HbyA, State& st)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<vector> Hn = interpolateVec(a, HbyA);
    st.phi.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        const label o = a.ownCell[i];
        const scalar w = a.weights[i];
        const vector f{w * HbyA[o].x + (1.0 - w) * Hn[i].x,
                       w * HbyA[o].y + (1.0 - w) * Hn[i].y,
                       w * HbyA[o].z + (1.0 - w) * Hn[i].z};
        st.phi[i] = f.x * a.Sf[i].x + f.y * a.Sf[i].y + f.z * a.Sf[i].z;
    }
}


void correctFlux(
    const AMIInterface&        a,
    const State&               st,
    const std::vector<scalar>& p,
    State&                     phiOut)
{
    const std::vector<scalar> pn = interpolate(a, p);
    phiOut.phi.resize(a.ownCell.size());
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        phiOut.phi[i] = st.phi[i] - st.ifCoeff[i] * (pn[i] - p[a.ownCell[i]]);
}


void addGrad(
    const AMIInterface&        a,
    const std::vector<scalar>& Uown,
    const std::vector<scalar>& UN,
    const std::vector<scalar>& V,
    std::vector<vector>&       grad)
{
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
    {
        const label o = a.ownCell[i];
        const scalar fv = (a.weights[i] * Uown[o] + (1.0 - a.weights[i]) * UN[i]) / V[o];
        grad[o].x += a.Sf[i].x * fv;
        grad[o].y += a.Sf[i].y * fv;
        grad[o].z += a.Sf[i].z * fv;
    }
}


void lapCorr(
    const AMIInterface&        a,
    const int                  comp,
    const std::vector<scalar>& gammaCell,
    const std::vector<tensor>& gradU,
    std::vector<scalar>&       corr)
{
    const std::vector<scalar> gN = interpolate(a, gammaCell);
    const tensor& T = a.forwardT;
    const scalar R[9] = {T.xx, T.xy, T.xz, T.yx, T.yy, T.yz, T.zx, T.zy, T.zz};
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
    {
        const label o = a.ownCell[i];
        const scalar w0 = a.weights[i], w1 = 1.0 - w0;
        scalar gn[3] = {0, 0, 0};
        for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k)
        {
            const tensor& q = gradU[a.nbrCell[k]];
            const scalar G[9] = {q.xx, q.xy, q.xz, q.yx, q.yy, q.yz, q.zx, q.zy, q.zz};
            const scalar wk = a.weight[k];
            if (!a.translational)
            {
                // (R G R^T)[comp][j]: the neighbour's velocity gradient is a TENSOR, so a rotational
                // interface transforms it on BOTH indices. R G alone rotates the vector index and leaves
                // the derivative index in the neighbour's frame.
                for (int j = 0; j < 3; ++j)
                {
                    scalar sum = 0;
                    for (int x = 0; x < 3; ++x)
                        for (int y = 0; y < 3; ++y)
                            sum += R[3*comp + x] * G[3*x + y] * R[3*j + y];
                    gn[j] += wk * sum;
                }
            }
            else
            {
                for (int j = 0; j < 3; ++j) gn[j] += wk * G[3*comp + j];
            }
        }
        const tensor& qo = gradU[o];
        const scalar go[3] = {comp == 0 ? qo.xx : comp == 1 ? qo.yx : qo.zx,
                              comp == 0 ? qo.xy : comp == 1 ? qo.yy : qo.zy,
                              comp == 0 ? qo.xz : comp == 1 ? qo.yz : qo.zz};
        const scalar gf[3] = {w0*go[0] + w1*gn[0], w0*go[1] + w1*gn[1], w0*go[2] + w1*gn[2]};
        const scalar gammaf = w0 * gammaCell[o] + w1 * gN[i];
        corr[o] -= gammaf * a.magSf[i]
                 * (a.corrVec[i].x*gf[0] + a.corrVec[i].y*gf[1] + a.corrVec[i].z*gf[2]);
    }
}


void addDiv(
    const AMIInterface&        a,
    const State&               st,
    const std::vector<scalar>& V,
    std::vector<scalar>&       div)
{
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
    {
        const label o = a.ownCell[i];
        div[o] += st.phi[i] / V[o];
    }
}

} // namespace cyclicAMI
} // namespace cpu
} // namespace brae
