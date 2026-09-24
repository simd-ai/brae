// brae's pointPatchDist against OpenFOAM's own, on the mesh RAS/floatingObject builds -- and with it
// the scale field rigidBodyMeshMotion makes of it.
//
// usage: test_point_patch_dist_vs_openfoam <caseDir> <ofTimeDir> <patch> [innerDistance outerDistance]
#include "foam_field_reader.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "mesh_edges_cpp.cuh"
#include "point_patch_dist_cpp.cuh"
#include "primitive_mesh.cuh"
#include "primitive_patch_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf("  %s:   %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

struct Diff
{
    scalar worst = 0;
    scalar refMax = 0;
    std::size_t at = 0;
    std::size_t nAbove = 0;
    scalar rel() const { return worst/std::fmax(refMax, scalar(1e-300)); }
};

Diff compare(const std::vector<scalar>& a, const std::vector<scalar>& b, scalar tol)
{
    Diff d;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t k = 0; k < n; ++k)
    {
        const scalar e = std::fabs(a[k] - b[k]);
        if (e > d.worst) { d.worst = e; d.at = k; }
        if (e > tol) ++d.nAbove;
        d.refMax = std::fmax(d.refMax, std::fabs(b[k]));
    }
    return d;
}

std::vector<scalar> readPointField(const std::string& path, label nPoints)
{
    const FieldData<scalar> fd = readField<scalar>(path);
    const std::size_t n = static_cast<std::size_t>(nPoints);
    return fd.internalUniform ? std::vector<scalar>(n, fd.internalUniformValue) : fd.internalField;
}

}   // namespace


int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <ofTimeDir> <patch> [innerDistance outerDistance]\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string ofDir = argv[2];
    const std::string pname = argv[3];
    const bool withScale = (argc > 5);
    const scalar di = withScale ? static_cast<scalar>(std::atof(argv[4])) : scalar(0);
    const scalar dOut = withScale ? static_cast<scalar>(std::atof(argv[5])) : scalar(0);

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nPoints = m.nPoints();

    std::printf("== brae pointPatchDist vs OpenFOAM: %s, patch `%s` ==\n", caseDir.c_str(), pname.c_str());

    const MeshEdges e = buildMeshEdges(m);
    std::printf("  mesh: %d points, %d faces (%d internal), %d edges\n",
                (int)nPoints, (int)m.nFaces(), (int)m.nInternalFaces(), (int)e.nEdges());
    // THE EDGE LIST ITSELF, before anything walks it: every edge joins two DIFFERENT points with the
    // lower first (getEdge orders them), every point's edges are ascending (Foam::sort on the
    // renumbered list), and each edge appears in exactly the two points it joins.
    {
        bool ordered = true;
        bool paired = true;
        std::vector<label> count(static_cast<std::size_t>(e.nEdges()), 0);
        for (label ei = 0; ei < e.nEdges(); ++ei)
        {
            ordered = ordered && (e.start[static_cast<std::size_t>(ei)]
                                < e.end[static_cast<std::size_t>(ei)]);
        }
        for (label p = 0; p < nPoints; ++p)
        {
            const std::vector<label>& pe = e.pointEdges[static_cast<std::size_t>(p)];
            for (std::size_t i = 1; i < pe.size(); ++i)
            {
                ordered = ordered && (pe[i - 1] < pe[i]);
            }
            for (const label ei : pe)
            {
                ++count[static_cast<std::size_t>(ei)];
                paired = paired && (e.start[static_cast<std::size_t>(ei)] == p
                                 || e.end[static_cast<std::size_t>(ei)] == p);
            }
        }
        bool twice = true;
        for (const label c : count) twice = twice && (c == 2);
        check("every edge joins two different points, lower first, and each point's edges ascend",
              ordered);
        check("...and every edge appears in exactly the two points it joins", paired && twice);
    }

    std::vector<label> ids;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].name == pname) ids.push_back(static_cast<label>(pi));
    }
    check("the case has the patch this arm measures to", !ids.empty());
    if (ids.empty()) { std::printf("test_point_patch_dist_vs_openfoam: %d failures\n", ++failures); return 1; }

    const PointPatchDist d = pointPatchDist(m, e, patches, ids);
    check("the wave reached every point, as OpenFOAM's did", d.nUnset == 0);

    const std::vector<scalar> ofDist = readPointField(ofDir + "/pointPatchDist.dump", nPoints);
    check("OpenFOAM's dump has one value per mesh point",
          ofDist.size() == static_cast<std::size_t>(nPoints));
    const Diff dd = compare(d.distance, ofDist, scalar(1e-12));
    std::printf("  pointPatchDist: worst %.4e of %.4e (relative %.3e), %zu of %d points above 1e-12\n",
                (double)dd.worst, (double)dd.refMax, (double)dd.rel(), dd.nAbove, (int)nPoints);
    check("brae's point-to-patch distance is OpenFOAM's", dd.rel() < scalar(1e-13) && dd.nAbove == 0);

    // THE CONTROL, and it is the whole reason this is a wave and not a search: seed every patch point
    // and take the EXACT nearest one. PointEdgeWave does not, because externalPointEdgePoint refuses
    // to propagate an improvement smaller than 1% of the squared distance a point already holds, so
    // the answer it leaves can be up to half a per cent too large.
    {
        std::vector<label> seeds;
        for (const label pi : ids)
        {
            const FvPatch& q = patches[static_cast<std::size_t>(pi)];
            const PrimitivePatchAddressing addr = primitivePatch(m, faceRange(q.start, q.size));
            for (const label mp : addr.meshPoints) seeds.push_back(mp);
        }
        std::vector<scalar> brute(static_cast<std::size_t>(nPoints), scalar(0));
        for (label p = 0; p < nPoints; ++p)
        {
            scalar best = std::numeric_limits<scalar>::max();
            for (const label s : seeds)
            {
                const vector v = m.points()[static_cast<std::size_t>(p)]
                               - m.points()[static_cast<std::size_t>(s)];
                best = std::fmin(best, v.x*v.x + v.y*v.y + v.z*v.z);
            }
            brute[static_cast<std::size_t>(p)] = std::sqrt(best);
        }
        const Diff db = compare(brute, ofDist, scalar(1e-12));
        std::printf("  CONTROL: the EXACT nearest patch point, against OpenFOAM's wave: worst %.4e "
                    "(relative %.3e) on %zu of %d points\n",
                    (double)db.worst, (double)db.rel(), db.nAbove, (int)nPoints);
        // Stated per arm by the script: on the `floatingObject` patch the wave converges to the exact
        // answer and this control cannot witness anything, which is said rather than hidden; on
        // `atmosphere` it separates them on 60 points.
        if (db.nAbove > 0)
        {
            check("...and it is a DIFFERENT answer, so this arm measures the wave and not a search",
                  db.rel() > scalar(1e5)*std::fmax(dd.rel(), scalar(1e-16)));
        }
        else
        {
            std::printf("  (on this patch the wave reaches the exact nearest point everywhere, so the "
                        "control above cannot separate the two -- the `atmosphere` arm is where it "
                        "does)\n");
        }
    }

    if (withScale)
    {
        const std::vector<scalar> ofScale = readPointField(ofDir + "/rbmScale.dump", nPoints);
        const std::vector<scalar> scale = rigidBodyMeshMotionScale(d.distance, di, dOut);
        const Diff ds = compare(scale, ofScale, scalar(1e-12));
        std::printf("  rigidBodyMeshMotion scale (di %g, do %g): worst %.4e (relative %.3e), %zu above "
                    "1e-12\n", (double)di, (double)dOut, (double)ds.worst, (double)ds.rel(), ds.nAbove);
        check("brae's scale field is OpenFOAM's", ds.rel() < scalar(1e-13) && ds.nAbove == 0);

        // THE CONTROL: the LINEAR ramp, which is what the scale is before the cosine. Both are 0
        // outside the band and 1 inside it, so only the band between innerDistance and outerDistance
        // can witness the difference -- and it does.
        std::vector<scalar> linear(static_cast<std::size_t>(nPoints));
        std::size_t inBand = 0;
        for (label p = 0; p < nPoints; ++p)
        {
            const scalar r = (dOut - d.distance[static_cast<std::size_t>(p)])/(dOut - di);
            linear[static_cast<std::size_t>(p)] = std::fmin(std::fmax(r, scalar(0)), scalar(1));
            if (r > scalar(0) && r < scalar(1)) ++inBand;
        }
        const Diff dl = compare(linear, ofScale, scalar(1e-12));
        std::printf("  CONTROL: the ramp WITHOUT the cosine: worst %.4e on %zu of %d points "
                    "(%zu lie in the blend band)\n",
                    (double)dl.worst, dl.nAbove, (int)nPoints, inBand);
        check("...the cosine is what makes the scale, and leaving it out is a different field",
              inBand > 0 && dl.worst > scalar(0.01));
    }

    std::printf("test_point_patch_dist_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
