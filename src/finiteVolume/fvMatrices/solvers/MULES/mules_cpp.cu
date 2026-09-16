// MULES, the host reference -- see mules_cpp.cuh for the provenance and for the six things a
// transcription of this gets wrong.
#include "mules_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace MULES {

namespace {

// OpenFOAM's ROOTVSMALL, the constant the limiter's two divisors carry. Named rather than inlined
// because those divisors run in EVERY cell -- sumPhip and mSumPhim are zero wherever the correction
// does not reach -- so this is what lambda is in the untouched majority of the domain, not a guard
// against a rare zero.
constexpr scalar kRootVSmall = scalar(1.0e-150);

inline scalar at(const std::vector<scalar>* f, std::size_t i, scalar constantValue)
{
    return f ? (*f)[i] : constantValue;
}

inline scalar clamp01(scalar x) { return x < scalar(0) ? scalar(0) : (x > scalar(1) ? scalar(1) : x); }

}   // namespace


Controls readControls(const FoamDict& fvSolution, const std::string& psiName)
{
    Controls c;
    const FoamDict* solvers = fvSolution.subDict("solvers");
    const FoamDict* sd = solvers ? solvers->subDict(psiName) : nullptr;
    if (!sd)
        throw std::runtime_error(
            "brae MULES: fvSolution has no `solvers/" + psiName + "` entry. mesh.solverDict(psi.name()) "
            "is where every MULES control lives, and it is the same entry alphaControls reads.");
    c.nLimiterIter  = static_cast<label>(sd->scalarOr("nLimiterIter",  scalar(3)));
    c.smoothLimiter = sd->scalarOr("smoothLimiter", scalar(0));
    c.extremaCoeff  = sd->scalarOr("extremaCoeff",  scalar(0));
    // boundaryExtremaCoeff DEFAULTS TO extremaCoeff, not to 0 (MULESTemplates.C:218-226). A case that
    // sets extremaCoeff and not boundaryExtremaCoeff means both.
    c.boundaryExtremaCoeff = sd->scalarOr("boundaryExtremaCoeff", c.extremaCoeff);
    if (c.nLimiterIter < 1)
        throw std::runtime_error("brae MULES: nLimiterIter must be at least 1.");
    return c;
}


void boundedDonorFlux(const SurfaceScalarField&     phi,
                      const GeometricField<scalar>& psi,
                      const SurfaceScalarField&     phiPsi,
                      const PrimitiveMesh&          m,
                      const std::vector<FvPatch>&   patches,
                      SurfaceScalarField&           phiBD)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    // upwind<scalar>(mesh, phi).flux(psi): the weights are pos0(phi), so the face value is the owner's
    // where the flux leaves the owner and the neighbour's where it enters.
    phiBD.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const scalar p = phi.internal[f];
        phiBD.internal[f] = p * ((p >= scalar(0)) ? psi.internal[own[f]] : psi.internal[nei[f]]);
    }

    // THE OVERWRITE. On every non-coupled patch phiBD becomes phiPsi, so phiCorr is identically zero
    // there and the boundary flux passes through the limiter untouched -- see note 3 in the header.
    // brae has no coupled patch in a VoF case yet, so this is every patch today; it is written per
    // patch so that adding cyclic changes only the coupled ones.
    phiBD.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        phiBD.boundary[pi] = phiPsi.boundary[pi];
}


void limiter(Limiter&                      lambda,
             scalar                        rDeltaT,
             const GeometricField<scalar>& psi,
             const std::vector<scalar>&    psiOld,
             const SurfaceScalarField&     phiBD,
             const SurfaceScalarField&     phiCorr,
             const Fields&                 f,
             const Controls&               c,
             const PrimitiveMesh&          m,
             const FvGeometry&             g,
             const std::vector<FvPatch>&   patches)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const std::vector<scalar>& V   = g.V();
    const std::vector<scalar>& psiIf = psi.internal;

    const scalar boundaryDeltaExtremaCoeff =
        std::fmax(c.boundaryExtremaCoeff - c.extremaCoeff, scalar(0));

    // lambda starts at 1 on every face and only ever decreases -- see note 5.
    lambda.internal.assign(static_cast<std::size_t>(nIf), scalar(1));
    lambda.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        lambda.boundary[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(1));

    // THE SWAPPED INITIALISATION -- note 1. psiMaxn starts at the LOWER bound and psiMinn at the upper,
    // so the neighbour scan below raises one and lowers the other from the far end.
    std::vector<scalar> psiMaxn(static_cast<std::size_t>(nC));
    std::vector<scalar> psiMinn(static_cast<std::size_t>(nC));
    for (label ci = 0; ci < nC; ++ci)
    {
        psiMaxn[ci] = at(f.psiMin, ci, scalar(0));
        psiMinn[ci] = at(f.psiMax, ci, scalar(1));
    }

    std::vector<scalar> sumPhiBD(static_cast<std::size_t>(nC), scalar(0));
    std::vector<scalar> sumPhip (static_cast<std::size_t>(nC), scalar(0));
    std::vector<scalar> mSumPhim(static_cast<std::size_t>(nC), scalar(0));

    for (label fi = 0; fi < nIf; ++fi)
    {
        const label o = own[fi], n = nei[fi];

        // note 2: the cell's OWN value never enters its own extrema.
        psiMaxn[o] = std::fmax(psiMaxn[o], psiIf[n]);
        psiMinn[o] = std::fmin(psiMinn[o], psiIf[n]);
        psiMaxn[n] = std::fmax(psiMaxn[n], psiIf[o]);
        psiMinn[n] = std::fmin(psiMinn[n], psiIf[o]);

        sumPhiBD[o] += phiBD.internal[fi];
        sumPhiBD[n] -= phiBD.internal[fi];

        const scalar pc = phiCorr.internal[fi];
        if (pc > scalar(0)) { sumPhip[o]  += pc; mSumPhim[n] += pc; }
        else                { mSumPhim[o] -= pc; sumPhip[n]  -= pc; }
    }

    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        const std::vector<scalar>& pv = psi.boundary[pi]->value();
        const bool fixesValue = psi.boundary[pi]->fixesValue();

        for (label i = 0; i < q.size; ++i)
        {
            const label ci = q.faceCells[i];

            // Three branches, and they are not interchangeable. A COUPLED patch contributes its
            // neighbour cell's value; a patch that FIXES A VALUE contributes that value, because psi
            // really can reach it there; anything else (zeroGradient, inletOutlet on outflow)
            // contributes NOTHING -- its own extrema are already represented by the interior.
            if (fixesValue)
            {
                psiMaxn[ci] = std::fmax(psiMaxn[ci], pv[i]);
                psiMinn[ci] = std::fmin(psiMinn[ci], pv[i]);
            }
            else if (boundaryDeltaExtremaCoeff > scalar(0))
            {
                const scalar extrema = boundaryDeltaExtremaCoeff
                                     * (at(f.psiMax, ci, scalar(1)) - at(f.psiMin, ci, scalar(0)));
                psiMaxn[ci] += extrema;
                psiMinn[ci] -= extrema;
            }

            sumPhiBD[ci] += phiBD.boundary[pi][i];

            const scalar pc = phiCorr.boundary[pi][i];
            if (pc > scalar(0)) sumPhip[ci]  += pc;
            else                mSumPhim[ci] -= pc;
        }
    }

    for (label ci = 0; ci < nC; ++ci)
    {
        const scalar pMax = at(f.psiMax, ci, scalar(1));
        const scalar pMin = at(f.psiMin, ci, scalar(0));
        psiMaxn[ci] = std::fmin(psiMaxn[ci] + c.extremaCoeff * (pMax - pMin), pMax);
        psiMinn[ci] = std::fmax(psiMinn[ci] - c.extremaCoeff * (pMax - pMin), pMin);
        if (c.smoothLimiter > scalar(1e-15))
        {
            psiMaxn[ci] = std::fmin(c.smoothLimiter*psiIf[ci]
                                  + (scalar(1) - c.smoothLimiter)*psiMaxn[ci], pMax);
            psiMinn[ci] = std::fmax(c.smoothLimiter*psiIf[ci]
                                  + (scalar(1) - c.smoothLimiter)*psiMinn[ci], pMin);
        }
    }

    // The bounds become FLUX-SPACE BUDGETS (MULESTemplates.C:418-436, the fixed-mesh branch). After
    // this psiMaxn is no longer a value of psi: it is how much net antidiffusive flux the cell can
    // still absorb before it would exceed its upper bound, and psiMinn the same for the lower one.
    for (label ci = 0; ci < nC; ++ci)
    {
        const scalar rhoC  = at(f.rho,    ci, scalar(1));
        const scalar rhoO  = at(f.rhoOld, ci, scalar(1));
        const scalar SpC   = at(f.Sp,     ci, scalar(0));
        const scalar SuC   = at(f.Su,     ci, scalar(0));
        const scalar a = (rhoC*rDeltaT - SpC);
        const scalar b = (rhoO*rDeltaT)*psiOld[ci];
        const scalar mx = V[ci]*(a*psiMaxn[ci] - SuC - b) + sumPhiBD[ci];
        const scalar mn = V[ci]*(SuC - a*psiMinn[ci] + b) - sumPhiBD[ci];
        psiMaxn[ci] = mx;
        psiMinn[ci] = mn;
    }

    std::vector<scalar> sumlPhip (static_cast<std::size_t>(nC));
    std::vector<scalar> mSumlPhim(static_cast<std::size_t>(nC));

    for (label it = 0; it < c.nLimiterIter; ++it)
    {
        std::fill(sumlPhip.begin(),  sumlPhip.end(),  scalar(0));
        std::fill(mSumlPhim.begin(), mSumlPhim.end(), scalar(0));

        for (label fi = 0; fi < nIf; ++fi)
        {
            const label o = own[fi], n = nei[fi];
            const scalar lpc = lambda.internal[fi] * phiCorr.internal[fi];
            if (lpc > scalar(0)) { sumlPhip[o]  += lpc; mSumlPhim[n] += lpc; }
            else                 { mSumlPhim[o] -= lpc; sumlPhip[n]  -= lpc; }
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            for (label i = 0; i < q.size; ++i)
            {
                const label ci = q.faceCells[i];
                const scalar lpc = lambda.boundary[pi][i] * phiCorr.boundary[pi][i];
                if (lpc > scalar(0)) sumlPhip[ci]  += lpc;
                else                 mSumlPhim[ci] -= lpc;
            }
        }

        // OpenFOAM reuses the two accumulators as the two per-cell limiters, and then aliases them to
        // names that read the other way round: lambdam IS sumlPhip and lambdap IS mSumlPhim
        // (MULESTemplates.C:501-502). The crossing is deliberate -- a face carrying a POSITIVE
        // correction out of the owner is constrained by the owner's lower-bound limiter and the
        // neighbour's upper-bound one -- and it is the single easiest line in MULES to transcribe
        // straight and get backwards.
        for (label ci = 0; ci < nC; ++ci)
        {
            sumlPhip[ci]  = clamp01((sumlPhip[ci]  + psiMaxn[ci]) / (mSumPhim[ci] + kRootVSmall));
            mSumlPhim[ci] = clamp01((mSumlPhim[ci] + psiMinn[ci]) / (sumPhip[ci]  + kRootVSmall));
        }
        const std::vector<scalar>& lambdam = sumlPhip;
        const std::vector<scalar>& lambdap = mSumlPhim;

        for (label fi = 0; fi < nIf; ++fi)
        {
            const label o = own[fi], n = nei[fi];
            lambda.internal[fi] = (phiCorr.internal[fi] > scalar(0))
                ? std::fmin(lambda.internal[fi], std::fmin(lambdap[o], lambdam[n]))
                : std::fmin(lambda.internal[fi], std::fmin(lambdam[o], lambdap[n]));
        }

        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            // note 4: a wedge patch is zeroed outright, before any coupled logic. 2D axisymmetric
            // cases are wedges, so this is not an exotic branch.
            if (q.type == "wedge")
            {
                std::fill(lambda.boundary[pi].begin(), lambda.boundary[pi].end(), scalar(0));
                continue;
            }
            // An UNCOUPLED patch is left alone: phiCorr is zero there by construction (see
            // boundedDonorFlux), so lambda on it multiplies nothing. Only coupled patches take the
            // per-cell limiters, and brae has none in a VoF case yet.
        }
        // syncTools::syncFaceList(minEqOp) -- a parallel reduction over coupled faces, and a no-op in
        // serial. It belongs here and is absent because the coupled branch above is.
    }
}


void limit(scalar                        rDeltaT,
           const GeometricField<scalar>& psi,
           const std::vector<scalar>&    psiOld,
           const SurfaceScalarField&     phi,
           SurfaceScalarField&           phiPsi,
           const Fields&                 f,
           const Controls&               c,
           const PrimitiveMesh&          m,
           const FvGeometry&             g,
           const std::vector<FvPatch>&   patches,
           Limiter*                      lambdaOut)
{
    SurfaceScalarField phiBD;
    boundedDonorFlux(phi, psi, phiPsi, m, patches, phiBD);

    SurfaceScalarField phiCorr;
    phiCorr.internal.resize(phiPsi.internal.size());
    for (std::size_t fi = 0; fi < phiPsi.internal.size(); ++fi)
        phiCorr.internal[fi] = phiPsi.internal[fi] - phiBD.internal[fi];
    phiCorr.boundary.resize(phiPsi.boundary.size());
    for (std::size_t pi = 0; pi < phiPsi.boundary.size(); ++pi)
    {
        phiCorr.boundary[pi].resize(phiPsi.boundary[pi].size());
        for (std::size_t i = 0; i < phiPsi.boundary[pi].size(); ++i)
            phiCorr.boundary[pi][i] = phiPsi.boundary[pi][i] - phiBD.boundary[pi][i];
    }

    Limiter lambda;
    limiter(lambda, rDeltaT, psi, psiOld, phiBD, phiCorr, f, c, m, g, patches);

    for (std::size_t fi = 0; fi < phiPsi.internal.size(); ++fi)
        phiPsi.internal[fi] = phiBD.internal[fi] + lambda.internal[fi]*phiCorr.internal[fi];
    for (std::size_t pi = 0; pi < phiPsi.boundary.size(); ++pi)
        for (std::size_t i = 0; i < phiPsi.boundary[pi].size(); ++i)
            phiPsi.boundary[pi][i] =
                phiBD.boundary[pi][i] + lambda.boundary[pi][i]*phiCorr.boundary[pi][i];

    if (lambdaOut) *lambdaOut = lambda;
}


void explicitSolve(scalar                      rDeltaT,
                   std::vector<scalar>&        psi,
                   const std::vector<scalar>&  psiOld,
                   const SurfaceScalarField&   phiPsi,
                   const Fields&               f,
                   const PrimitiveMesh&        m,
                   const FvGeometry&           g,
                   const std::vector<FvPatch>& patches)
{
    // fvc::surfaceIntegrate(phiPsi) -- the divergence, per unit volume.
    const std::vector<scalar> divPhiPsi = fvc::div(phiPsi, m, g, patches);

    const label nC = m.nCells();
    psi.resize(static_cast<std::size_t>(nC));
    for (label ci = 0; ci < nC; ++ci)
    {
        // rho.oldTime() above the line and rho below it -- the same split fvm::ddt(rho,U) carries.
        const scalar num = at(f.rhoOld, ci, scalar(1))*psiOld[ci]*rDeltaT
                         + at(f.Su, ci, scalar(0))
                         - divPhiPsi[ci];
        const scalar den = at(f.rho, ci, scalar(1))*rDeltaT - at(f.Sp, ci, scalar(0));
        psi[ci] = num / den;
    }
}


void explicitSolveLimited(scalar                        rDeltaT,
                          GeometricField<scalar>&       psi,
                          const std::vector<scalar>&    psiOld,
                          const SurfaceScalarField&     phi,
                          SurfaceScalarField&           phiPsi,
                          const Fields&                 f,
                          const Controls&               c,
                          const PrimitiveMesh&          m,
                          const FvGeometry&             g,
                          const std::vector<FvPatch>&   patches,
                          Limiter*                      lambdaOut)
{
    // MULESTemplates.C:166-182: correctBoundaryConditions, THEN limit, THEN solve. The limiter reads
    // psi's boundary values for the fixesValue extrema, so a stale boundary would widen or narrow the
    // bounds it enforces.
    psi.evaluateBoundary();
    limit(rDeltaT, psi, psiOld, phi, phiPsi, f, c, m, g, patches, lambdaOut);
    explicitSolve(rDeltaT, psi.internal, psiOld, phiPsi, f, m, g, patches);
    psi.evaluateBoundary();
}

} // namespace MULES
} // namespace cpu
} // namespace brae
