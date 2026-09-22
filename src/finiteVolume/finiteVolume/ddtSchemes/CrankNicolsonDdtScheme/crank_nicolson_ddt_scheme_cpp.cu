// OpenFOAM's CrankNicolson ddt scheme, the host reference -- see the header.
#include "crank_nicolson_ddt_scheme_cpp.cuh"
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace fv {

namespace {

// vector arithmetic in the component order OpenFOAM's Field ops take it
inline vector times(
    scalar a,
    const vector& v)
{
    return vector{a*v.x, a*v.y, a*v.z};
}
inline vector minus(
    const vector& a,
    const vector& b)
{
    return vector{a.x - b.x, a.y - b.y, a.z - b.z};
}
inline vector plus(
    const vector& a,
    const vector& b)
{
    return vector{a.x + b.x, a.y + b.y, a.z + b.z};
}
// ...and the scalar forms, so the body below reads the same for both T (declared before the template:
// a fundamental type brings no namespace to the lookup at instantiation)
inline scalar times(
    scalar a,
    scalar v)
{
    return a*v;
}
inline scalar minus(
    scalar a,
    scalar b)
{
    return a - b;
}
inline scalar plus(
    scalar a,
    scalar b)
{
    return a + b;
}
inline scalar zeroOf(scalar) { return scalar(0); }
inline vector zeroOf(const vector&) { return vector{0, 0, 0}; }

// the body shared by the two fvmDdt overloads: T is scalar or vector, and every product is written in
// the order CrankNicolsonDdtScheme.C:1000-1087 evaluates it (the static branch)
template <typename T>
void fvmDdtBody(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<T>& ddt0,
    const std::vector<scalar>* rho,
    const std::vector<scalar>* rhoOld,
    const std::vector<scalar>* rhoOO,
    const std::vector<T>& vfOld,
    const std::vector<T>& vfOO,
    const std::vector<scalar>& V,
    std::vector<scalar>& diag,
    std::vector<T>& source)
{
    const std::size_t nC = V.size();
    if (clock.deltaT <= scalar(0) || clock.deltaT0 <= scalar(0))
        throw std::runtime_error(
            "brae CrankNicolson fvm::ddt: deltaT and deltaT0 must both be positive (rDtCoef = coef/deltaT, "
            "rDtCoef0 = coef0/deltaT0).");
    if (vfOld.size() != nC || vfOO.size() != nC || diag.size() != nC || source.size() != nC)
        throw std::runtime_error(
            "brae CrankNicolson fvm::ddt(" + ddt0.name + "): vf.oldTime(), vf.oldTime().oldTime(), V and "
            "the matrix must be one value per cell.");
    const bool withRho = (rho != nullptr);
    if (withRho != (rhoOld != nullptr) || withRho != (rhoOO != nullptr))
        throw std::runtime_error(
            "brae CrankNicolson fvm::ddt(" + ddt0.name + "): rho, rho.oldTime() and rho.oldTime().oldTime() "
            "come together or not at all.");
    if (withRho && (rho->size() != nC || rhoOld->size() != nC || rhoOO->size() != nC))
        throw std::runtime_error(
            "brae CrankNicolson fvm::ddt(" + ddt0.name + "): the three densities must be one value per cell.");
    ddt0.lookupOrCreate(clock, nC, {});

    const scalar rDtCoef = ddt0.rDtCoef(clock);
    // fvm.diag() = rDtCoef*rho.primitiveField()*mesh().V()
    for (std::size_t c = 0; c < nC; ++c)
    {
        const scalar r = withRho ? (*rho)[c] : scalar(1);
        diag[c] += (rDtCoef*r)*V[c];
    }
    // vf.oldTime().oldTime(), rho.oldTime().oldTime(): ensure the old-old levels exist -- the caller's
    if (ddt0.evaluate(clock))
    {
        // ddt0 = rDtCoef0*(rho.oldTime()*vf.oldTime() - rho.oldTime().oldTime()*vf.oldTime().oldTime())
        //      - offCentre_(ddt0())
        const scalar rDtCoef0 = ddt0.rDtCoef0(clock);
        for (std::size_t c = 0; c < nC; ++c)
        {
            const scalar ro = withRho ? (*rhoOld)[c] : scalar(1);
            const scalar roo = withRho ? (*rhoOO)[c] : scalar(1);
            ddt0.internal[c] = minus(times(rDtCoef0, minus(times(ro, vfOld[c]), times(roo, vfOO[c]))),
                                     offCentre(clock, ddt0.internal[c]));
        }
    }
    // fvm.source() = (rDtCoef*rho.oldTime().primitiveField()*vf.oldTime().primitiveField()
    //               + offCentre_(ddt0.primitiveField()))*mesh().V()
    for (std::size_t c = 0; c < nC; ++c)
    {
        const scalar ro = withRho ? (*rhoOld)[c] : scalar(1);
        const T term = plus(times(rDtCoef*ro, vfOld[c]), offCentre(clock, ddt0.internal[c]));
        source[c] = plus(source[c], times(V[c], term));
    }
}

}   // namespace


template <typename T>
void CrankNicolsonDdt0<T>::lookupOrCreate(
    const CrankNicolsonClock& clock,
    std::size_t nInternal,
    const std::vector<std::size_t>& patchSizes)
{
    if (exists)
    {
        if (internal.size() != nInternal)
            throw std::runtime_error(
                "brae CrankNicolson: the ddt0 field `" + name + "` has " + std::to_string(internal.size())
                + " values and the mesh " + std::to_string(nInternal) + ".");
        return;
    }
    // DDt0Field(io, mesh, Zero, dims): startTimeIndex_ = mesh.time().timeIndex(), and the
    // GeometricField's own time index is the same, so evaluate() is false for the rest of this step
    internal.assign(nInternal, zeroOf(T{}));
    boundary.resize(patchSizes.size());
    for (std::size_t pi = 0; pi < patchSizes.size(); ++pi)
    {
        boundary[pi].assign(patchSizes[pi], zeroOf(T{}));
    }
    startTimeIndex = clock.timeIndex;
    timeIndex = clock.timeIndex;
    exists = true;
}
template struct CrankNicolsonDdt0<scalar>;
template struct CrankNicolsonDdt0<vector>;


void fvmDdt(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<vector>& ddt0,
    const std::vector<scalar>* rho,
    const std::vector<scalar>* rhoOld,
    const std::vector<scalar>* rhoOO,
    const std::vector<vector>& vfOld,
    const std::vector<vector>& vfOO,
    const std::vector<scalar>& V,
    FvVectorMatrix& M)
{
    fvmDdtBody<vector>(clock, ddt0, rho, rhoOld, rhoOO, vfOld, vfOO, V, M.diag, M.source);
}

void fvmDdt(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<scalar>& ddt0,
    const std::vector<scalar>* rho,
    const std::vector<scalar>* rhoOld,
    const std::vector<scalar>* rhoOO,
    const std::vector<scalar>& vfOld,
    const std::vector<scalar>& vfOO,
    const std::vector<scalar>& V,
    FvScalarMatrix& M)
{
    fvmDdtBody<scalar>(clock, ddt0, rho, rhoOld, rhoOO, vfOld, vfOO, V, M.diag, M.source);
}


void fvcDdtPhiCorr(
    const CrankNicolsonClock& clock,
    CrankNicolsonDdt0<vector>& ddt0,
    CrankNicolsonDdt0<scalar>& dphidt0,
    const std::vector<vector>& UOld,
    const std::vector<vector>& UOO,
    const std::vector<std::vector<vector>>& UOldBnd,
    const std::vector<std::vector<vector>>& UOOBnd,
    const SurfaceScalarField& phiOld,
    const SurfaceScalarField& phiOO,
    const std::vector<bool>& patchFixesU,
    scalar ddtPhiCoeff,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    SurfaceScalarField& out)
{
    const label nIf = m.nInternalFaces();
    const std::size_t nC = static_cast<std::size_t>(m.nCells());
    if (clock.deltaT <= scalar(0) || clock.deltaT0 <= scalar(0))
        throw std::runtime_error("brae CrankNicolson ddtCorr: deltaT and deltaT0 must both be positive.");
    if (UOld.size() != nC || UOO.size() != nC || UOldBnd.size() != patches.size() || UOOBnd.size() != patches.size()
     || phiOld.internal.size() != static_cast<std::size_t>(nIf) || phiOO.internal.size() != static_cast<std::size_t>(nIf)
     || phiOld.boundary.size() != patches.size() || phiOO.boundary.size() != patches.size()
     || patchFixesU.size() != patches.size())
        throw std::runtime_error(
            "brae CrankNicolson ddtCorr: U.oldTime(), U.oldTime().oldTime() (cells and patches), phi.oldTime(), "
            "phi.oldTime().oldTime() and the fixes-value mask must all be the mesh's.");
    std::vector<std::size_t> patchSizes(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        patchSizes[pi] = static_cast<std::size_t>(patches[pi].size);
        if (UOldBnd[pi].size() != patchSizes[pi] || UOOBnd[pi].size() != patchSizes[pi]
         || phiOld.boundary[pi].size() != patchSizes[pi] || phiOO.boundary[pi].size() != patchSizes[pi])
            throw std::runtime_error(
                "brae CrankNicolson ddtCorr: patch `" + patches[pi].name + "` has " + std::to_string(patchSizes[pi])
                + " faces and an old-time field disagrees.");
    }
    ddt0.lookupOrCreate(clock, nC, patchSizes);
    dphidt0.lookupOrCreate(clock, static_cast<std::size_t>(nIf), patchSizes);

    // dimensionedScalar rDtCoef = rDtCoef_(ddt0)  -- ddt0's, and dphidt0's coefficient is never asked for
    const scalar rDtCoef = ddt0.rDtCoef(clock);
    if (ddt0.evaluate(clock))
    {
        // ddt0 = rDtCoef0_(ddt0)*(U.oldTime() - U.oldTime().oldTime()) - offCentre_(ddt0()), a
        // GeometricField assignment: cells and patch values alike
        const scalar rDtCoef0 = ddt0.rDtCoef0(clock);
        for (std::size_t c = 0; c < nC; ++c)
        {
            ddt0.internal[c] = minus(times(rDtCoef0, minus(UOld[c], UOO[c])), offCentre(clock, ddt0.internal[c]));
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (std::size_t i = 0; i < patchSizes[pi]; ++i)
            {
                ddt0.boundary[pi][i] = minus(times(rDtCoef0, minus(UOldBnd[pi][i], UOOBnd[pi][i])),
                                             offCentre(clock, ddt0.boundary[pi][i]));
            }
        }
    }
    if (dphidt0.evaluate(clock))
    {
        const scalar rDtCoef0 = dphidt0.rDtCoef0(clock);
        for (label f = 0; f < nIf; ++f)
        {
            dphidt0.internal[f] = rDtCoef0*(phiOld.internal[f] - phiOO.internal[f]) - offCentre(clock, dphidt0.internal[f]);
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (std::size_t i = 0; i < patchSizes[pi]; ++i)
            {
                dphidt0.boundary[pi][i] = rDtCoef0*(phiOld.boundary[pi][i] - phiOO.boundary[pi][i])
                                        - offCentre(clock, dphidt0.boundary[pi][i]);
            }
        }
    }

    // W = rDtCoef*U.oldTime() + offCentre_(ddt0()), the vol field dotInterpolate is handed
    std::vector<vector> W(nC);
    for (std::size_t c = 0; c < nC; ++c)
    {
        W[c] = plus(times(rDtCoef, UOld[c]), offCentre(clock, ddt0.internal[c]));
    }
    std::vector<std::vector<vector>> Wb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        Wb[pi].resize(patchSizes[pi]);
        for (std::size_t i = 0; i < patchSizes[pi]; ++i)
        {
            Wb[pi][i] = plus(times(rDtCoef, UOldBnd[pi][i]), offCentre(clock, ddt0.boundary[pi][i]));
        }
    }

    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& lambda = g.weights();
    const std::vector<vector>& Sf = g.Sf();
    // doubleScalar.H's SMALL
    const scalar kSmall = scalar(1e-15);
    // dotInterpolate on an internal face: Sfi & (lambda*(vfi[P] - vfi[N]) + vfi[N])
    auto dotInterp = [&](label f, const std::vector<vector>& vf)
    {
        const vector& P = vf[own[f]];
        const vector& N = vf[nei[f]];
        const vector t = plus(times(lambda[f], minus(P, N)), N);
        return Sf[f].x*t.x + Sf[f].y*t.y + Sf[f].z*t.z;
    };
    // fvcDdtPhiCoeff(U, phi) with phiCorr = phi - dotInterpolate(Sf, U): 1 - min(|phiCorr|/(|phi| + SMALL), 1)
    auto coeffOf = [&](scalar phi, scalar phiCorr)
    {
        return (ddtPhiCoeff < scalar(0))
             ? scalar(1) - std::fmin(std::fabs(phiCorr)/(std::fabs(phi) + kSmall), scalar(1))
             : ddtPhiCoeff;
    };

    out.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const scalar phiCorr = phiOld.internal[f] - dotInterp(f, UOld);
        const scalar corr = (rDtCoef*phiOld.internal[f] + offCentre(clock, dphidt0.internal[f])) - dotInterp(f, W);
        out.internal[f] = coeffOf(phiOld.internal[f], phiCorr)*corr;
    }
    out.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        out.boundary[pi].assign(patchSizes[pi], scalar(0));
        // ddtCouplingCoeff is zero on every patch whose U fixes a value and on every cyclicAMI patch
        // (ddtScheme.C:178-181); the correction there is zero whatever the flux says
        if (patchFixesU[pi] || q.type == "cyclicAMI") continue;
        for (label i = 0; i < q.size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(i);
            const vector& S = Sf[q.start + i];
            // the patch half of dotInterpolate: lerp(N, P, lambda) on a coupled patch, the patch value
            // on an uncoupled one
            const vector uo = q.coupled ? coupledLinear(q, i, UOld) : UOldBnd[pi][k];
            const vector wf = q.coupled ? coupledLinear(q, i, W) : Wb[pi][k];
            const scalar phiCorr = phiOld.boundary[pi][k] - (S.x*uo.x + S.y*uo.y + S.z*uo.z);
            const scalar corr = (rDtCoef*phiOld.boundary[pi][k] + offCentre(clock, dphidt0.boundary[pi][k]))
                              - (S.x*wf.x + S.y*wf.y + S.z*wf.z);
            out.boundary[pi][k] = coeffOf(phiOld.boundary[pi][k], phiCorr)*corr;
        }
    }
}


scalar readOcCoeff(const std::string& entry)
{
    // the entry as fvSchemes hands it: "CrankNicolson 0.5", "CrankNicolson", or "CrankNicolson { ... }"
    const std::string key = "CrankNicolson";
    const std::size_t k = entry.find(key);
    if (k == std::string::npos)
        throw std::runtime_error("brae CrankNicolson: `" + entry + "` does not name the scheme.");
    std::string rest = entry.substr(k + key.size());
    const std::size_t b = rest.find_first_not_of(" \t\n\r");
    if (b == std::string::npos)
    {
        // no coefficient: Function1Types::Constant("ocCoeff", 1), CrankNicolsonDdtScheme.C:274
        return scalar(1);
    }
    rest = rest.substr(b);
    if (rest[0] == '{' || rest.find("ocCoeff") != std::string::npos)
        throw std::runtime_error(
            "brae CrankNicolson: ddtSchemes gives `" + entry + "`, an off-centring coefficient that is a "
            "Function1 of time (CrankNicolsonDdtScheme.C:297-303, a `ramp` from Euler to CrankNicolson). "
            "brae carries the constant form only. Refused rather than run the end value from the start.");
    char* end = nullptr;
    const double v = std::strtod(rest.c_str(), &end);
    if (end == rest.c_str())
        throw std::runtime_error("brae CrankNicolson: `" + entry + "`: the off-centring coefficient did not parse.");
    if (v < 0 || v > 1)
        throw std::runtime_error(
            "brae CrankNicolson: off-centreing coefficient = " + std::to_string(v)
            + " should be >= 0 and <= 1 (CrankNicolsonDdtScheme.C:293-297, OpenFOAM's FatalIOError).");
    return static_cast<scalar>(v);
}

}   // namespace fv
}   // namespace cpu
}   // namespace brae
