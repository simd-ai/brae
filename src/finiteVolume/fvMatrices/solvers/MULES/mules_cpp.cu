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

// OpenFOAM's SMALL*SMALL, the threshold CMULES tests the boundary flux against to decide whether a face
// is an outlet. Not zero: a face whose total flux is numerically nothing counts as an inlet and is left
// unlimited, which is the conservative side to err on.
constexpr scalar kSmallSquared = scalar(1.0e-15) * scalar(1.0e-15);

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
    // A COUPLED face keeps upwind's own flux, from the cell the flux leaves (MULESTemplates.C:605, `if
    // (!phiBDPf.coupled())`).
    phiBD.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        phiBD.boundary[pi] = phiPsi.boundary[pi];
        if (!patches[pi].coupled)
        {
            continue;
        }
        const FvPatch& q = patches[pi];
        for (std::size_t i = 0; i < phiBD.boundary[pi].size(); ++i)
        {
            const scalar p = phi.boundary[pi][i];
            phiBD.boundary[pi][i] = p * ((p >= scalar(0)) ? psi.internal[q.faceCells[i]]
                                                           : psi.internal[q.nbrFaceCells[i]]);
        }
    }
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
    // mesh.Vsc(): the mesh's V unless the mesh moves (Fields::Vsc)
    const std::vector<scalar>& V = f.Vsc ? *f.Vsc : g.V();
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
        // AN EMPTY PATCH CONTRIBUTES NOTHING. emptyFvPatch::size() is 0 in OpenFOAM
        // (emptyFvPatch.H:79), so every one of these loop bodies is simply never entered there --
        // the extrema, sumPhiBD and the phiCorr accumulation all skip it. brae's FvPatch keeps the
        // faces, so the skip has to be explicit, and fvc::div already carries the same line for the
        // same reason. Without it a 2D case -- which is damBreak, capillaryRise and most VoF
        // tutorials -- folds its front and back faces into every cell's flux budget.
        if (q.type == "empty") continue;
        const std::vector<scalar>& pv = psi.boundary[pi]->value();
        const bool fixesValue = psi.boundary[pi]->fixesValue();

        for (label i = 0; i < q.size; ++i)
        {
            const label ci = q.faceCells[i];

            // Three branches, and they are not interchangeable. A COUPLED patch contributes its
            // neighbour cell's value; a patch that FIXES A VALUE contributes that value, because psi
            // really can reach it there; anything else (zeroGradient, inletOutlet on outflow)
            // contributes NOTHING -- its own extrema are already represented by the interior.
            if (q.coupled)
            {
                psiMaxn[ci] = std::fmax(psiMaxn[ci], psiIf[q.nbrFaceCells[i]]);
                psiMinn[ci] = std::fmin(psiMinn[ci], psiIf[q.nbrFaceCells[i]]);
            }
            else if (fixesValue)
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
        if (f.Vsc0)
        {
            // MULESTemplates.C:397-417, the moving-mesh branch: the old value's term carries the OLD
            // volume and stands outside the bracket
            const scalar b = ((*f.Vsc0)[ci]*rDeltaT)*rhoO*psiOld[ci];
            psiMaxn[ci] = V[ci]*(a*psiMaxn[ci] - SuC) - b + sumPhiBD[ci];
            psiMinn[ci] = V[ci]*(SuC - a*psiMinn[ci]) + b - sumPhiBD[ci];
            continue;
        }
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
            if (q.type == "empty") continue;            // see the note above
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
            // boundedDonorFlux), so lambda on it multiplies nothing. A COUPLED patch takes this side's
            // per-cell limiter (MULESTemplates.C:543); the sync below brings the other side's.
            if (q.coupled)
            {
                for (label i = 0; i < q.size; ++i)
                {
                    const label ci = q.faceCells[i];
                    lambda.boundary[pi][i] = (phiCorr.boundary[pi][i] > scalar(0))
                        ? std::fmin(lambda.boundary[pi][i], lambdap[ci])
                        : std::fmin(lambda.boundary[pi][i], lambdam[ci]);
                }
            }
        }
        // syncTools::syncFaceList(mesh, allLambda, minEqOp<scalar>()). NOT a no-op in serial: it syncs
        // a cyclic pair too, and leaves both sides of a coupled face with the SMALLER of their limiters
        // -- which is an internal face's one limiter, min(lambda, lambdap of the cell the correction
        // leaves, lambdam of the cell it enters). It does NOT sync an AMI pair (FvPatch::ami): each side
        // keeps its own limiter and the limited flux is not equal and opposite across it. MEASURED on
        // damBreakLeakage's opening step against OpenFOAM's written alphaPhi0: the receiving side's
        // flux 0.75 of the giving side's in OpenFOAM, and the receiving cell's alpha 33% high when brae
        // synced it.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            if (!q.coupled || !q.owner || q.ami)
            {
                continue;
            }
            std::vector<scalar>& mine = lambda.boundary[pi];
            std::vector<scalar>& theirs = lambda.boundary[static_cast<std::size_t>(q.nbrPatch)];
            for (std::size_t i = 0; i < mine.size(); ++i)
            {
                const scalar lo = std::fmin(mine[i], theirs[i]);
                mine[i] = lo;
                theirs[i] = lo;
            }
        }
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
    // fvc::surfaceIntegrate(phiPsi) -- the divergence, per unit volume, and the volume is Vsc
    const std::vector<scalar> divPhiPsi = f.Vsc ? fvc::div(phiPsi, m, patches, *f.Vsc)
                                                : fvc::div(phiPsi, m, g, patches);

    const label nC = m.nCells();
    psi.resize(static_cast<std::size_t>(nC));
    for (label ci = 0; ci < nC; ++ci)
    {
        // rho.oldTime() above the line and rho below it -- the same split fvm::ddt(rho,U) carries.
        // On a moving mesh (MULESTemplates.C:60-68) the old value is weighted by Vsc0/Vsc as well.
        const scalar old = f.Vsc0
            ? (*f.Vsc0)[ci]*at(f.rhoOld, ci, scalar(1))*psiOld[ci]*rDeltaT/(*f.Vsc)[ci]
            : at(f.rhoOld, ci, scalar(1))*psiOld[ci]*rDeltaT;
        const scalar num = old
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

// ---------------------------------------------------------------------------------------------------
// CMULES -- see the header for A, B, C and D, the four things that make this not the explicit path.

Controls readControlsCorr(const FoamDict& fvSolution, const std::string& psiName)
{
    const FoamDict* solvers = fvSolution.subDict("solvers");
    const FoamDict* sd = solvers ? solvers->subDict(psiName) : nullptr;
    if (!sd)
        throw std::runtime_error(
            "brae CMULES: fvSolution has no `solvers/" + psiName + "` entry.");

    Controls c;
    // D: nLimiterIter is get<label> here, not getOrDefault. A case that asks for MULESCorr and omits it
    // is a FatalError in OpenFOAM, and defaulting it to 3 would run that case with an iteration count
    // nobody chose.
    const scalar n = sd->scalarOr("nLimiterIter", scalar(-1));
    if (n < 0)
        throw std::runtime_error(
            "brae CMULES: `nLimiterIter` is missing from solvers/" + psiName + ". The semi-implicit "
            "path reads it with get<label> and has NO default (CMULESTemplates.C:225), unlike the "
            "explicit limiter which defaults it to 3. OpenFOAM FatalErrors here.");
    c.nLimiterIter = static_cast<label>(n);
    if (c.nLimiterIter < 1)
        throw std::runtime_error("brae CMULES: nLimiterIter must be at least 1.");
    c.smoothLimiter = sd->scalarOr("smoothLimiter", scalar(0));
    c.extremaCoeff  = sd->scalarOr("extremaCoeff",  scalar(0));
    c.boundaryExtremaCoeff = sd->scalarOr("boundaryExtremaCoeff", c.extremaCoeff);
    return c;
}


void limiterCorr(Limiter&                      lambda,
                 scalar                        rDeltaT,
                 const GeometricField<scalar>& psi,
                 const SurfaceScalarField&     phi,
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
    // mesh.Vsc(): the mesh's V unless the mesh moves (Fields::Vsc)
    const std::vector<scalar>& V = f.Vsc ? *f.Vsc : g.V();
    const std::vector<scalar>& psiIf = psi.internal;

    const scalar boundaryDeltaExtremaCoeff =
        std::fmax(c.boundaryExtremaCoeff - c.extremaCoeff, scalar(0));

    lambda.internal.assign(static_cast<std::size_t>(nIf), scalar(1));
    lambda.boundary.assign(patches.size(), std::vector<scalar>{});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        lambda.boundary[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(1));

    // The swapped initialisation, exactly as in the explicit limiter.
    std::vector<scalar> psiMaxn(static_cast<std::size_t>(nC));
    std::vector<scalar> psiMinn(static_cast<std::size_t>(nC));
    for (label ci = 0; ci < nC; ++ci)
    {
        psiMaxn[ci] = at(f.psiMin, ci, scalar(0));
        psiMinn[ci] = at(f.psiMax, ci, scalar(1));
    }

    // B: NO sumPhiBD. There is no donor flux here -- it went through the implicit matrix.
    std::vector<scalar> sumPhip (static_cast<std::size_t>(nC), scalar(0));
    std::vector<scalar> mSumPhim(static_cast<std::size_t>(nC), scalar(0));

    for (label fi = 0; fi < nIf; ++fi)
    {
        const label o = own[fi], n = nei[fi];
        psiMaxn[o] = std::fmax(psiMaxn[o], psiIf[n]);
        psiMinn[o] = std::fmin(psiMinn[o], psiIf[n]);
        psiMaxn[n] = std::fmax(psiMaxn[n], psiIf[o]);
        psiMinn[n] = std::fmin(psiMinn[n], psiIf[o]);

        const scalar pc = phiCorr.internal[fi];
        if (pc > scalar(0)) { sumPhip[o]  += pc; mSumPhim[n] += pc; }
        else                { mSumPhim[o] -= pc; sumPhip[n]  -= pc; }
    }

    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        if (q.type == "empty") continue;                // see limiter(), above
        const std::vector<scalar>& pv = psi.boundary[pi]->value();
        const bool fixesValue = psi.boundary[pi]->fixesValue();
        for (label i = 0; i < q.size; ++i)
        {
            const label ci = q.faceCells[i];
            if (q.coupled)
            {
                // CMULESTemplates.C:327 -- the cell on the other side, as on an internal face
                psiMaxn[ci] = std::fmax(psiMaxn[ci], psiIf[q.nbrFaceCells[i]]);
                psiMinn[ci] = std::fmin(psiMinn[ci], psiIf[q.nbrFaceCells[i]]);
            }
            else if (fixesValue)
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

    // A and B together: the budget is measured against psi AS IT STANDS -- rho*psi, the current values
    // -- with no donor term. CMULESTemplates.C:400-412.
    for (label ci = 0; ci < nC; ++ci)
    {
        const scalar rhoC = at(f.rho, ci, scalar(1));
        const scalar SpC  = at(f.Sp,  ci, scalar(0));
        const scalar SuC  = at(f.Su,  ci, scalar(0));
        const scalar a = (rhoC*rDeltaT - SpC);
        const scalar b = rhoC*psiIf[ci]*rDeltaT;
        const scalar mx = V[ci]*(a*psiMaxn[ci] - SuC - b);
        const scalar mn = V[ci]*(SuC - a*psiMinn[ci] + b);
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
            if (q.type == "empty") continue;            // see the note above
            for (label i = 0; i < q.size; ++i)
            {
                const label ci = q.faceCells[i];
                const scalar lpc = lambda.boundary[pi][i] * phiCorr.boundary[pi][i];
                if (lpc > scalar(0)) sumlPhip[ci]  += lpc;
                else                 mSumlPhim[ci] -= lpc;
            }
        }

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
            if (q.type == "wedge")
            {
                std::fill(lambda.boundary[pi].begin(), lambda.boundary[pi].end(), scalar(0));
                continue;
            }
            // C: the branch the explicit limiter does not have. phiCorr is genuinely non-zero on the
            // boundary here, so it has to be limited -- but ONLY where the total flux LEAVES the
            // domain. OpenFOAM's own comment is "Limit outlet faces only". Limiting an inlet would
            // throttle a prescribed inflow, and the threshold is SMALL*SMALL, not 0, so a face with a
            // numerically-zero flux counts as an inlet and is left alone.
            for (label i = 0; i < q.size; ++i)
            {
                const scalar total = phi.boundary[pi][i] + phiCorr.boundary[pi][i];
                // ...on an UNCOUPLED patch. A coupled face is limited whichever way its flux goes
                // (CMULESTemplates.C:516).
                if (!q.coupled && total <= kSmallSquared) continue;
                const label ci = q.faceCells[i];
                lambda.boundary[pi][i] = (phiCorr.boundary[pi][i] > scalar(0))
                    ? std::fmin(lambda.boundary[pi][i], lambdap[ci])
                    : std::fmin(lambda.boundary[pi][i], lambdam[ci]);
            }
        }
        // syncTools::syncFaceList(mesh, allLambda, minEqOp<scalar>()). NOT a no-op in serial: it syncs
        // a cyclic pair too, and leaves both sides of a coupled face with the SMALLER of their limiters
        // -- which is an internal face's one limiter, min(lambda, lambdap of the cell the correction
        // leaves, lambdam of the cell it enters). It does NOT sync an AMI pair (FvPatch::ami): each side
        // keeps its own limiter and the limited flux is not equal and opposite across it. MEASURED on
        // damBreakLeakage's opening step against OpenFOAM's written alphaPhi0: the receiving side's
        // flux 0.75 of the giving side's in OpenFOAM, and the receiving cell's alpha 33% high when brae
        // synced it.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            if (!q.coupled || !q.owner || q.ami)
            {
                continue;
            }
            std::vector<scalar>& mine = lambda.boundary[pi];
            std::vector<scalar>& theirs = lambda.boundary[static_cast<std::size_t>(q.nbrPatch)];
            for (std::size_t i = 0; i < mine.size(); ++i)
            {
                const scalar lo = std::fmin(mine[i], theirs[i]);
                mine[i] = lo;
                theirs[i] = lo;
            }
        }
    }
}


void limitCorr(scalar                        rDeltaT,
               const GeometricField<scalar>& psi,
               const SurfaceScalarField&     phi,
               SurfaceScalarField&           phiCorr,
               const Fields&                 f,
               const Controls&               c,
               const PrimitiveMesh&          m,
               const FvGeometry&             g,
               const std::vector<FvPatch>&   patches,
               Limiter*                      lambdaOut)
{
    Limiter lambda;
    limiterCorr(lambda, rDeltaT, psi, phi, phiCorr, f, c, m, g, patches);

    // phiCorr *= lambda, IN PLACE. No blended flux is formed: there is nothing to blend against.
    for (std::size_t fi = 0; fi < phiCorr.internal.size(); ++fi)
        phiCorr.internal[fi] *= lambda.internal[fi];
    for (std::size_t pi = 0; pi < phiCorr.boundary.size(); ++pi)
        for (std::size_t i = 0; i < phiCorr.boundary[pi].size(); ++i)
            phiCorr.boundary[pi][i] *= lambda.boundary[pi][i];

    if (lambdaOut) *lambdaOut = lambda;
}


void correct(scalar                      rDeltaT,
             std::vector<scalar>&        psi,
             const SurfaceScalarField&   phiCorr,
             const Fields&               f,
             const PrimitiveMesh&        m,
             const FvGeometry&           g,
             const std::vector<FvPatch>& patches)
{
    // CMULESTemplates.C:50-75: the same formula whether the mesh moves or not, but surfaceIntegrate
    // divides by Vsc either way
    const std::vector<scalar> divPhiCorr = f.Vsc ? fvc::div(phiCorr, m, patches, *f.Vsc)
                                                 : fvc::div(phiCorr, m, g, patches);
    const label nC = m.nCells();
    for (label ci = 0; ci < nC; ++ci)
    {
        // A: rho*psi, both CURRENT. explicitSolve's rho.oldTime()*psi.oldTime() would re-do the time
        // step from the old state carrying only the correction, discarding the implicit solve.
        const scalar rhoC = at(f.rho, ci, scalar(1));
        const scalar num  = rhoC*psi[ci]*rDeltaT + at(f.Su, ci, scalar(0)) - divPhiCorr[ci];
        const scalar den  = rhoC*rDeltaT - at(f.Sp, ci, scalar(0));
        psi[ci] = num / den;
    }
}


void correctLimited(scalar                        rDeltaT,
                    GeometricField<scalar>&       psi,
                    const SurfaceScalarField&     phi,
                    SurfaceScalarField&           phiCorr,
                    const Fields&                 f,
                    const Controls&               c,
                    const PrimitiveMesh&          m,
                    const FvGeometry&             g,
                    const std::vector<FvPatch>&   patches,
                    Limiter*                      lambdaOut)
{
    limitCorr(rDeltaT, psi, phi, phiCorr, f, c, m, g, patches, lambdaOut);
    correct(rDeltaT, psi.internal, phiCorr, f, m, g, patches);
    psi.evaluateBoundary();
}

} // namespace MULES
} // namespace cpu
} // namespace brae
