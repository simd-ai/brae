// brae's FLUX corrections against REAL OpenFOAM's own, through one refinement step and one
// unrefinement step -- unit 6b of the dynamicRefineFvMesh port.
//
// WHAT IS UNDER TEST: everything that happens to a flux between the face mapper finishing and
// `update()` returning. THREE sites, not one:
//   mapFields:257-422  four write sites keyed on faceMap/reverseFaceMap and on the derived masterFaces
//   mapFields:424-437  mapNewInternalFaces, on the ORIENTED branch for a flux
//   unrefine:610-689   a correction only unrefinement has, keyed on faceToSplitPoint/reversePointMap
// MEASURED: the first overwrites 2,156 faces on the refine step and ZERO on the unrefine step, where
// the third overwrites 158. A port that finds only the first is green on every refinement and silently
// wrong on every unrefinement, which is why both arms exist.
//
// WHAT IS NOT UNDER TEST, deliberately: the FACE MAPPER itself. Its `direct()` predicate and
// addressing build are their own unit, so step one is taken from OpenFOAM's own addressing in the dump
// and what is measured is the corrections ON TOP of it. Said here rather than implied by a green run.
//
// usage: test_flux_map_vs_openfoam <oracleDump> <refine|unrefine>
#include "dynamic_refine_fv_mesh_cpp.cuh"
#include "map_poly_mesh_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf("  %s:   %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

template <typename T>
void put(std::vector<T>& v, label i, const T& x)
{
    if (static_cast<std::size_t>(i) >= v.size()) v.resize(static_cast<std::size_t>(i) + 1, T{});
    v[static_cast<std::size_t>(i)] = x;
}

template <typename T>
void putPatch(std::vector<std::vector<T>>& v, label p, label i, const T& x)
{
    if (static_cast<std::size_t>(p) >= v.size()) v.resize(static_cast<std::size_t>(p) + 1);
    put(v[static_cast<std::size_t>(p)], i, x);
}

struct Oracle
{
    label nInternalFaces = -1, nFaces = -1, nMasterFace = -1, nOldInternalFaces = -1;
    std::vector<label> oldPatchStarts;
    label nFluxOverwritten = -1, nUnrefineFluxFace = -1, nNewInternalFaceHull = -1;
    std::vector<label> patchStart, patchSize, owner, neighbour;
    std::vector<std::vector<label>> cells;
    std::vector<vector> Sf;
    std::vector<std::vector<vector>> SfBnd;
    std::vector<scalar> magSf, prePhi, postPhi, phiU;
    std::vector<std::vector<scalar>> magSfBnd, prePhiBnd, postPhiBnd, phiUBnd;
    std::vector<std::pair<label, label>> faceToSplitPoint;
    std::vector<label> masterFaceOf;      // OpenFOAM's own master-face list, for the derivation check
};

Oracle readOracle(const std::string& path)
{
    Oracle o;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.rfind("[brae] ", 0) != 0) continue;
        std::istringstream is(line.substr(7));
        std::string w;
        is >> w;
        if (w == "post")
        {
            std::string tag;
            is >> tag;
            if (tag == "nInternalFaces") { is >> o.nInternalFaces; continue; }
            if (tag == "nFaces")         { is >> o.nFaces; continue; }
            label i = 0;
            if (tag == "patchStart" || tag == "patchSize" || tag == "faceOwner"
             || tag == "faceNeighbour")
            {
                label v = 0;
                is >> i >> v;
                put(tag == "patchStart" ? o.patchStart
                  : tag == "patchSize"  ? o.patchSize
                  : tag == "faceOwner"  ? o.owner : o.neighbour, i, v);
            }
            else if (tag == "cellFaces")
            {
                label n = 0;
                is >> i >> n;
                std::vector<label> fs(static_cast<std::size_t>(n));
                for (label& x : fs) is >> x;
                if (static_cast<std::size_t>(i) >= o.cells.size()) o.cells.resize(static_cast<std::size_t>(i) + 1);
                o.cells[static_cast<std::size_t>(i)] = std::move(fs);
            }
            else if (tag == "Sf")
            {
                scalar x = 0, y = 0, z = 0;
                is >> i >> x >> y >> z;
                put(o.Sf, i, vector{x, y, z});
            }
            else if (tag == "SfPatch")
            {
                label p = 0;
                scalar x = 0, y = 0, z = 0;
                is >> p >> i >> x >> y >> z;
                putPatch(o.SfBnd, p, i, vector{x, y, z});
            }
            else if (tag == "magSf" || tag == "phi" || tag == "phiU")
            {
                scalar v = 0;
                is >> i >> v;
                put(tag == "magSf" ? o.magSf : tag == "phi" ? o.postPhi : o.phiU, i, v);
            }
            else if (tag == "magSfPatch" || tag == "phiPatch" || tag == "phiUPatch")
            {
                label p = 0;
                scalar v = 0;
                is >> p >> i >> v;
                putPatch(tag == "magSfPatch" ? o.magSfBnd
                       : tag == "phiPatch"   ? o.postPhiBnd : o.phiUBnd, p, i, v);
            }
        }
        else if (w == "pre")
        {
            std::string tag;
            is >> tag;
            if (tag == "phi")
            {
                label i = 0;
                scalar v = 0;
                is >> i >> v;
                put(o.prePhi, i, v);
            }
            else if (tag == "phiPatch")
            {
                label p = 0, i = 0;
                scalar v = 0;
                is >> p >> i >> v;
                putPatch(o.prePhiBnd, p, i, v);
            }
        }
        else if (w == "map")
        {
            label k = -1;
            std::string tag;
            is >> k >> tag;
            if (k != 0) continue;
            if (tag == "nMasterFace")           is >> o.nMasterFace;
            else if (tag == "nOldInternalFaces") is >> o.nOldInternalFaces;
            else if (tag == "oldPatchStarts")
            {
                label i = 0, v = 0;
                is >> i >> v;
                put(o.oldPatchStarts, i, v);
            }
            else if (tag == "nFluxOverwritten") is >> o.nFluxOverwritten;
            else if (tag == "nUnrefineFluxFace")is >> o.nUnrefineFluxFace;
            else if (tag == "nNewInternalFaceHull") is >> o.nNewInternalFaceHull;
            else if (tag == "masterFace")       { label f = 0; is >> f; o.masterFaceOf.push_back(f); }
            else if (tag == "faceToSplitPoint")
            {
                label f = 0, p = 0;
                is >> f >> p;
                o.faceToSplitPoint.emplace_back(f, p);
            }
        }
    }
    return o;
}

struct Diff
{
    scalar worst = 0, refMax = 0;
    std::size_t nAbove = 0;
};

Diff cmp(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const scalar w = std::fabs(a[i] - b[i]);
        d.worst = std::fmax(d.worst, w);
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
        if (w > scalar(0)) ++d.nAbove;
    }
    return d;
}

Diff cmpBnd(
    const std::vector<std::vector<scalar>>& a,
    const std::vector<std::vector<scalar>>& b)
{
    Diff d;
    for (std::size_t p = 0; p < a.size() && p < b.size(); ++p)
    {
        const Diff e = cmp(a[p], b[p]);
        d.worst = std::fmax(d.worst, e.worst);
        d.refMax = std::fmax(d.refMax, e.refMax);
        d.nAbove += e.nAbove;
    }
    return d;
}

}   // namespace


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <oracleDump> <refine|unrefine>\n", argv[0]);
        return 2;
    }
    const std::string dumpPath = argv[1];
    const std::string phase = argv[2];

    std::printf("== brae flux corrections vs OpenFOAM: %s ==\n", dumpPath.c_str());

    const MapPolyMesh mpm = readMapPolyMesh(dumpPath);
    const Oracle o = readOracle(dumpPath);

    check("the oracle carried the new mesh's face addressing",
          o.nInternalFaces > 0 && !o.owner.empty() && !o.cells.empty()
       && o.patchStart.size() == o.patchSize.size() && !o.patchStart.empty());
    check("...its face areas", o.Sf.size() == static_cast<std::size_t>(o.nInternalFaces)
                            && o.magSf.size() == static_cast<std::size_t>(o.nInternalFaces));
    check("...and the flux before and after, with the interpolated one it corrects from",
          !o.prePhi.empty() && !o.postPhi.empty() && !o.phiU.empty());
    if (failures) { std::printf("test_flux_map_vs_openfoam: %d failures\n", failures); return 1; }

    dynamicRefine::FluxMeshView m;
    m.nInternalFaces = o.nInternalFaces;
    m.patchStart = o.patchStart;
    m.patchSize = o.patchSize;
    m.owner = o.owner;
    m.neighbour = o.neighbour;
    m.cells = o.cells;

    // THE MASTER FACES, derived as OpenFOAM derives them -- its own bitSet is local to mapFields and
    // dies with the block. Its COUNT is the cross-check: OpenFOAM prints it at :295.
    const std::vector<char> master = dynamicRefine::masterFaces(mpm, o.nFaces);
    std::size_t nMaster = 0;
    for (const char c : master) nMaster += (c != 0);
    std::printf("  master faces: brae %zu, OpenFOAM %d\n", nMaster, (int)o.nMasterFace);
    check("brae derives OpenFOAM's own master-face count", nMaster == static_cast<std::size_t>(o.nMasterFace));
    {
        std::vector<char> ofSet(master.size(), 0);
        for (const label f : o.masterFaceOf) ofSet[static_cast<std::size_t>(f)] = 1;
        check("...and the same faces, not merely as many", ofSet == master);
    }

    // STEP 1: the flux as OpenFOAM's FACE MAPPER left it, from the dump's own addressing.
    //
    // The OLD flux has to be flattened over every old face first -- internal then boundary at the old
    // patch starts -- because a new boundary face can map from an old face of either kind, and the
    // hull average in mapNewInternalFaces reads the boundary values too. Reconstructing the boundary
    // from the internal field alone left 256 boundary faces at 8.9e-04 and, through the hull, 398
    // internal faces at 1.1e-19: the boundary error walked inwards.
    std::vector<scalar> flatOld(static_cast<std::size_t>(mpm.faceMap.size()), scalar(0));
    std::vector<scalar> oldAll;
    {
        oldAll.assign(static_cast<std::size_t>(o.nOldInternalFaces), scalar(0));
        for (std::size_t i = 0; i < oldAll.size() && i < o.prePhi.size(); ++i) oldAll[i] = o.prePhi[i];
        // ...then the old boundary values at the OLD patch starts
        std::size_t nOldAll = static_cast<std::size_t>(o.nOldInternalFaces);
        for (std::size_t pp = 0; pp < o.prePhiBnd.size(); ++pp)
        {
            nOldAll = std::max(nOldAll,
                               static_cast<std::size_t>(o.oldPatchStarts[pp]) + o.prePhiBnd[pp].size());
        }
        oldAll.resize(nOldAll, scalar(0));
        for (std::size_t pp = 0; pp < o.prePhiBnd.size(); ++pp)
        {
            for (std::size_t i = 0; i < o.prePhiBnd[pp].size(); ++i)
            {
                oldAll[static_cast<std::size_t>(o.oldPatchStarts[pp]) + i] = o.prePhiBnd[pp][i];
            }
        }
        // the INTERNAL half through OpenFOAM's own face-mapper addressing
        mapFieldWith(flatOld, oldAll, mpm.faceMapperDirect, mpm.faceDirectAddressing,
                     mpm.faceAddressing, mpm.faceWeights);
    }
    std::vector<scalar> phi(static_cast<std::size_t>(o.nInternalFaces), scalar(0));
    for (std::size_t i = 0; i < phi.size() && i < flatOld.size(); ++i) phi[i] = flatOld[i];
    std::vector<std::vector<scalar>> phiBnd(o.postPhiBnd.size());
    for (std::size_t p = 0; p < phiBnd.size(); ++p)
    {
        phiBnd[p].assign(o.postPhiBnd[p].size(), scalar(0));
        // The BOUNDARY half goes through faceMap against the flat old field. It is NOT the class
        // OpenFOAM used -- that is fvPatchMapper, per patch, and the oracle does not dump it -- and
        // the face mapper's own addressing covers INTERNAL faces only (its size is nInternalFaces).
        // On both arms this reproduces OpenFOAM's boundary values exactly, which is the check; on a
        // mesh where a patch's faces are split it would not be enough, and the gate says so.
        for (std::size_t i = 0; i < phiBnd[p].size(); ++i)
        {
            const std::size_t facei = static_cast<std::size_t>(o.patchStart[p]) + i;
            if (facei >= mpm.faceMap.size()) continue;
            const label oldFacei = mpm.faceMap[facei];
            if (oldFacei >= 0 && static_cast<std::size_t>(oldFacei) < oldAll.size())
            {
                phiBnd[p][i] = oldAll[static_cast<std::size_t>(oldFacei)];
            }
        }
    }
    const std::vector<scalar> step1 = phi;
    const std::vector<std::vector<scalar>> step1Bnd = phiBnd;

    // STEP 3: the mapFields correction
    const label nWritten = dynamicRefine::correctFluxes(phi, phiBnd, o.phiU, o.phiUBnd, mpm, master, m);
    std::printf("  mapFields correction wrote %d faces; OpenFOAM's own count is %d\n",
                (int)nWritten, (int)o.nFluxOverwritten);
    check("brae's flux correction touches the faces OpenFOAM touched",
          nWritten == o.nFluxOverwritten);

    // STEP 4: mapNewInternalFaces, on the ORIENTED branch -- which is what `phi` takes
    dynamicRefine::mapNewInternalFacesOriented(phi, phiBnd, o.Sf, o.SfBnd, o.magSf, o.magSfBnd, mpm, m);

    // STEP 5: and, on an unrefinement, the site only unrefinement has
    label nUnref = 0;
    if (!o.faceToSplitPoint.empty())
    {
        nUnref = dynamicRefine::correctFluxesUnrefine(phi, phiBnd, o.phiU, o.phiUBnd,
                                                      o.faceToSplitPoint, mpm, m);
        std::printf("  unrefine's own correction wrote %d faces; OpenFOAM's own count is %d\n",
                    (int)nUnref, (int)o.nUnrefineFluxFace);
        check("brae's second flux correction touches the faces OpenFOAM touched",
              nUnref == o.nUnrefineFluxFace);
    }
    else
    {
        std::printf("  (no faceToSplitPoint rows, so this arm has no second correction -- which is "
                    "what a refinement step looks like)\n");
        check("...and a refinement step is what this arm claims to be", phase == "refine");
    }

    const Diff d = cmp(phi, o.postPhi);
    const Diff db = cmpBnd(phiBnd, o.postPhiBnd);
    std::printf("  phi internal: worst %.4e of %.4e (%zu of %d faces differ) | boundary: worst %.4e "
                "(%zu faces)\n",
                (double)d.worst, (double)d.refMax, d.nAbove, (int)o.nInternalFaces,
                (double)db.worst, db.nAbove);
    check("brae's corrected flux is OpenFOAM's on every internal face", d.worst == scalar(0));
    check("...and on every boundary face", db.worst == scalar(0));

    // CONTROL 1: step one alone. Both arms must fail it, or the corrections are not what this gate is
    // measuring.
    {
        const Diff c = cmp(step1, o.postPhi);
        std::printf("  CONTROL: the mapped flux with NO correction at all: worst %.4e of %.4e (%zu "
                    "faces differ)\n", (double)c.worst, (double)c.refMax, c.nAbove);
        check("...is a different answer", c.worst > scalar(0));
    }

    // CONTROL 2: the corrections without mapNewInternalFaces. On the refine arm the oriented round
    // trip rewrites EVERY face, so leaving it out must show; on the unrefine arm it is what moves the
    // 1,278 faces no correction touched.
    {
        std::vector<scalar> p2 = step1;
        std::vector<std::vector<scalar>> b2 = step1Bnd;
        dynamicRefine::correctFluxes(p2, b2, o.phiU, o.phiUBnd, mpm, master, m);
        if (!o.faceToSplitPoint.empty())
        {
            dynamicRefine::correctFluxesUnrefine(p2, b2, o.phiU, o.phiUBnd, o.faceToSplitPoint, mpm, m);
        }
        const Diff c = cmp(p2, o.postPhi);
        std::printf("  CONTROL: the corrections WITHOUT mapNewInternalFaces: worst %.4e (%zu faces "
                    "differ)\n", (double)c.worst, c.nAbove);
        check("...is a different answer, so the oriented round trip is measured", c.worst > scalar(0));
    }

    std::printf("test_flux_map_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
