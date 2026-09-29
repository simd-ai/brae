// setRDeltaT.H -- see inter_set_rdeltat_cpp.cuh for the provenance and the four things in it that are not
// what they look like.
#include "inter_set_rdeltat_cpp.cuh"
#include "fvc_smooth_cpp.cuh"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

const char* const WHO = "brae interFoam setRDeltaT: ";

// Foam::max and Foam::min on two doubleScalars (Scalar.H): a comparison, not fmax
scalar maxOf(
    scalar a,
    scalar b)
{
    return (a > b) ? a : b;
}

scalar minOf(
    scalar a,
    scalar b)
{
    return (a < b) ? a : b;
}

// Foam::pos0 (Scalar.H:250-253)
scalar pos0(scalar s)
{
    return (s >= 0) ? scalar(1) : scalar(0);
}

// fvc::surfaceSum (fvcSurfaceIntegrate.C:163-182) of a face field the caller forms face by face: every
// internal face adds to its owner and then its neighbour, in face order, and then every patch face to its
// face cell, patch by patch. An `empty` patch is a zero-sized fvPatch and adds nothing.
template<class FaceValue>
std::vector<scalar> surfaceSum(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    FaceValue faceValue)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<scalar> s(static_cast<std::size_t>(m.nCells()), scalar(0));
    for (label f = 0; f < nIf; ++f)
    {
        const scalar v = faceValue(-1, f);
        s[static_cast<std::size_t>(own[f])] += v;
        s[static_cast<std::size_t>(nei[f])] += v;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        if (q.type == "empty") continue;
        for (label i = 0; i < q.size; ++i)
        {
            s[static_cast<std::size_t>(q.faceCells[static_cast<std::size_t>(i)])] +=
                faceValue(static_cast<label>(pi), i);
        }
    }
    return s;
}

// The patch half of a surface field, checked for the patch's size -- a field whose boundary list is
// short would otherwise read past it
const std::vector<scalar>& patchOf(
    const SurfaceScalarField& sf,
    std::size_t pi,
    const FvPatch& q,
    const char* what)
{
    if (pi >= sf.boundary.size() || static_cast<label>(sf.boundary[pi].size()) != q.size)
    {
        throw std::runtime_error(
            std::string(WHO) + what + " has no values on patch '" + q.name + "'.");
    }
    return sf.boundary[pi];
}

// gMin/gMax of 1/rDeltaT over the cells
void timeScaleRange(
    const std::vector<scalar>& rDeltaT,
    scalar& lo,
    scalar& hi)
{
    lo = scalar(1)/rDeltaT[0];
    hi = lo;
    for (const scalar r : rDeltaT)
    {
        const scalar t = scalar(1)/r;
        lo = minOf(lo, t);
        hi = maxOf(hi, t);
    }
}

} // namespace


LocalEulerControls readLocalEulerControls(const FoamDict& pimple)
{
    LocalEulerControls c;
    c.maxCo = pimple.scalarOr("maxCo", c.maxCo);
    c.maxAlphaCo = pimple.scalarOr("maxAlphaCo", c.maxAlphaCo);
    c.rDeltaTSmoothingCoeff = pimple.scalarOr("rDeltaTSmoothingCoeff", c.rDeltaTSmoothingCoeff);
    c.nAlphaSpreadIter = static_cast<label>(pimple.scalarOr("nAlphaSpreadIter",
                                                            static_cast<scalar>(c.nAlphaSpreadIter)));
    c.alphaSpreadDiff = pimple.scalarOr("alphaSpreadDiff", c.alphaSpreadDiff);
    c.alphaSpreadMax = pimple.scalarOr("alphaSpreadMax", c.alphaSpreadMax);
    c.alphaSpreadMin = pimple.scalarOr("alphaSpreadMin", c.alphaSpreadMin);
    c.nAlphaSweepIter = static_cast<label>(pimple.scalarOr("nAlphaSweepIter",
                                                           static_cast<scalar>(c.nAlphaSweepIter)));
    c.rDeltaTDampingCoeff = pimple.scalarOr("rDeltaTDampingCoeff", c.rDeltaTDampingCoeff);
    c.maxDeltaT = pimple.scalarOr("maxDeltaT", c.maxDeltaT);
    return c;
}


void refuseUnportedLocalEuler(
    const LocalEulerControls& c,
    const std::vector<FvPatch>& patches)
{
    if (c.nAlphaSpreadIter > 0)
    {
        throw std::runtime_error(
            std::string(WHO) + "PIMPLE nAlphaSpreadIter is " + std::to_string(c.nAlphaSpreadIter)
            + " (OpenFOAM's default is 1 when the key is absent). setRDeltaT.H:95-106 then calls "
            "fvc::spread, which is not ported; the case must set `nAlphaSpreadIter 0;`.");
    }
    if (c.nAlphaSweepIter > 0)
    {
        throw std::runtime_error(
            std::string(WHO) + "PIMPLE nAlphaSweepIter is " + std::to_string(c.nAlphaSweepIter)
            + " (OpenFOAM's default is 5 when the key is absent). setRDeltaT.H:108-111 then calls "
            "fvc::sweep, which is not ported; the case must set `nAlphaSweepIter 0;`.");
    }
    for (const FvPatch& q : patches)
    {
        if (q.coupled || q.type == "cyclic" || q.type == "cyclicAMI" || q.type == "cyclicACMI"
         || q.type == "processor")
        {
            throw std::runtime_error(
                std::string(WHO) + "the mesh has the coupled patch '" + q.name + "'. setRDeltaT.H's sums "
                "and fvc::smooth's wave cross coupled patches, which the localEuler port does not.");
        }
    }
}


SetRDeltaTReport setRDeltaT(
    std::vector<scalar>& rDeltaT,
    const LocalEulerControls& c,
    const SetRDeltaTInput& in,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!in.rhoPhi || !in.phi || !in.alpha1 || !in.rho)
    {
        throw std::runtime_error(std::string(WHO) + "rhoPhi, phi, alpha1 and rho are all required.");
    }
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    if (static_cast<label>(rDeltaT.size()) != nC || static_cast<label>(in.rho->size()) != nC
     || static_cast<label>(in.alpha1->internal.size()) != nC)
    {
        throw std::runtime_error(std::string(WHO) + "a cell field is not the mesh's cells.");
    }
    if (static_cast<label>(in.rhoPhi->internal.size()) != nIf
     || static_cast<label>(in.phi->internal.size()) != nIf)
    {
        throw std::runtime_error(std::string(WHO) + "a face field is not the mesh's internal faces.");
    }
    const std::vector<scalar>& V = g.V();
    const std::vector<scalar>& rho = *in.rho;
    SetRDeltaTReport rep;

    // S0, setRDeltaT.H:56 -- rDeltaT0, the whole field as the previous step left it
    const std::vector<scalar> rDeltaT0 = rDeltaT;

    // S1, :59-64
    {
        const std::vector<scalar> sumRhoPhi = surfaceSum(m, patches, [&](label pi, label f)
        {
            if (pi < 0) return std::fabs(in.rhoPhi->internal[static_cast<std::size_t>(f)]);
            const std::size_t p = static_cast<std::size_t>(pi);
            return std::fabs(patchOf(*in.rhoPhi, p, patches[p], "rhoPhi")[static_cast<std::size_t>(f)]);
        });
        const scalar floorValue = scalar(1)/c.maxDeltaT;
        for (label ci = 0; ci < nC; ++ci)
        {
            const std::size_t cc = static_cast<std::size_t>(ci);
            rDeltaT[cc] = maxOf(floorValue, sumRhoPhi[cc]/(((2*c.maxCo)*V[cc])*rho[cc]));
        }
    }

    // S2, :66-81
    if (c.maxAlphaCo < c.maxCo)
    {
        // fvc::average(linearInterpolate(alpha1)) (fvcAverage.C:72-76, 108-116): the linear face values
        // -- lambda*(P - N) + N inside, the STORED patch values outside -- weighted by magSf
        const GeometricField<scalar>& alpha1 = *in.alpha1;
        const std::vector<label>& own = m.owner();
        const std::vector<label>& nei = m.neighbour();
        const std::vector<scalar>& w = g.weights();
        const std::vector<scalar>& magSf = g.magSf();
        std::vector<std::vector<scalar>> alphaB(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].type == "empty") continue;
            alphaB[pi] = alpha1.boundary[pi]->value();
            if (static_cast<label>(alphaB[pi].size()) != patches[pi].size)
            {
                throw std::runtime_error(
                    std::string(WHO) + "alpha1 has no stored values on patch '" + patches[pi].name + "'.");
            }
        }
        const std::vector<scalar> sumMagSfAlpha = surfaceSum(m, patches, [&](label pi, label f)
        {
            if (pi < 0)
            {
                const std::size_t ff = static_cast<std::size_t>(f);
                const scalar P = alpha1.internal[static_cast<std::size_t>(own[ff])];
                const scalar N = alpha1.internal[static_cast<std::size_t>(nei[ff])];
                const scalar af = w[ff]*(P - N) + N;
                return magSf[ff]*af;
            }
            const std::size_t p = static_cast<std::size_t>(pi);
            const std::size_t face = static_cast<std::size_t>(patches[p].start + f);
            return magSf[face]*alphaB[p][static_cast<std::size_t>(f)];
        });
        const std::vector<scalar> sumMagSf = surfaceSum(m, patches, [&](label pi, label f)
        {
            if (pi < 0) return magSf[static_cast<std::size_t>(f)];
            return magSf[static_cast<std::size_t>(patches[static_cast<std::size_t>(pi)].start + f)];
        });
        const std::vector<scalar> sumPhi = surfaceSum(m, patches, [&](label pi, label f)
        {
            if (pi < 0) return std::fabs(in.phi->internal[static_cast<std::size_t>(f)]);
            const std::size_t p = static_cast<std::size_t>(pi);
            return std::fabs(patchOf(*in.phi, p, patches[p], "phi")[static_cast<std::size_t>(f)]);
        });
        for (label ci = 0; ci < nC; ++ci)
        {
            const std::size_t cc = static_cast<std::size_t>(ci);
            const scalar aBar = sumMagSfAlpha[cc]/sumMagSf[cc];
            const scalar mask = pos0(aBar - c.alphaSpreadMin)*pos0(c.alphaSpreadMax - aBar);
            rDeltaT[cc] = maxOf(rDeltaT[cc], (mask*sumPhi[cc])/((2*c.maxAlphaCo)*V[cc]));
        }
    }

    // S3, :84 -- the patch values follow the cells (header note d); S4, :86-88
    timeScaleRange(rDeltaT, rep.flowMin, rep.flowMax);

    // S5, :90-93
    if (c.rDeltaTSmoothingCoeff < 1.0)
    {
        fvc::smooth(rDeltaT, c.rDeltaTSmoothingCoeff, m, patches);
    }

    // S6, :95-111 -- refused where the case is read; asserted here so a caller that skipped the refusal
    // cannot run without them
    if (c.nAlphaSpreadIter > 0 || c.nAlphaSweepIter > 0)
    {
        throw std::runtime_error(std::string(WHO) + "fvc::spread and fvc::sweep are not ported.");
    }

    // S7, :113-115
    timeScaleRange(rDeltaT, rep.smoothedMin, rep.smoothedMax);

    // S8, :120-135
    if (c.rDeltaTDampingCoeff < 1.0 && in.damp)
    {
        const scalar keep = scalar(1) - c.rDeltaTDampingCoeff;
        for (label ci = 0; ci < nC; ++ci)
        {
            const std::size_t cc = static_cast<std::size_t>(ci);
            rDeltaT[cc] = maxOf(rDeltaT[cc], keep*rDeltaT0[cc]);
        }
        rep.damped = true;
        timeScaleRange(rDeltaT, rep.dampedMin, rep.dampedMax);
    }
    return rep;
}


void applySetRDeltaTControls(
    LocalEulerControls& c,
    bool& damp)
{
    if (std::getenv("BRAE_CONTROL_LTS_NODAMP"))
    {
        std::printf("  *** CONTROL MODE: setRDeltaT's damping is off. This run is deliberately wrong. ***\n");
        damp = false;
    }
    if (std::getenv("BRAE_CONTROL_LTS_NOSMOOTH"))
    {
        std::printf("  *** CONTROL MODE: setRDeltaT's smoothing is off. This run is deliberately wrong. ***\n");
        c.rDeltaTSmoothingCoeff = scalar(1);
    }
}


std::set<std::string> readLtsScalarControl()
{
    std::set<std::string> names;
    const char* e = std::getenv("BRAE_CONTROL_LTS_SCALAR");
    if (!e)
    {
        return names;
    }
    const std::string all = e;
    std::size_t b = 0;
    while (b <= all.size())
    {
        std::size_t c = all.find(',', b);
        if (c == std::string::npos) c = all.size();
        const std::string n = all.substr(b, c - b);
        if (n != "alpha" && n != "ueqn" && n != "ddtcorr" && n != "turbulence")
            throw std::runtime_error(
                "brae interFoam: BRAE_CONTROL_LTS_SCALAR names `" + n + "`; it takes alpha, ueqn, ddtcorr or "
                "turbulence, comma-separated, and an unknown name would make the control vacuous.");
        names.insert(n);
        b = c + 1;
    }
    std::printf("  *** CONTROL MODE: the localEuler consumer(s) `%s` read 1/deltaT, not the local rDeltaT. "
                "This run is deliberately wrong. ***\n", all.c_str());
    return names;
}


SurfaceScalarField interpolateRDeltaT(
    const std::vector<scalar>& rDeltaT,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w = g.weights();
    SurfaceScalarField sf;
    sf.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        const std::size_t ff = static_cast<std::size_t>(f);
        const scalar P = rDeltaT[static_cast<std::size_t>(own[ff])];
        const scalar N = rDeltaT[static_cast<std::size_t>(nei[ff])];
        sf.internal[ff] = w[ff]*(P - N) + N;
    }
    sf.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        sf.boundary[pi].resize(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i)
        {
            sf.boundary[pi][static_cast<std::size_t>(i)] =
                rDeltaT[static_cast<std::size_t>(q.faceCells[static_cast<std::size_t>(i)])];
        }
    }
    return sf;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
