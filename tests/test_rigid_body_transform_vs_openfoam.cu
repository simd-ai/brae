// brae's rigid-body point transform against OpenFOAM's own pointDisplacement, on RAS/floatingObject.
//
// The DYNAMICS are OpenFOAM's: `q` is read from <time>/uniform/rigidBodyMotionState, so what this
// measures is the chain a `composite (Py Ry)` joint makes, X0(bodyID).inv() & X00(bodyID), and the
// septernion slerp that weights it by the mesh-motion scale -- not the Newmark integrator, which is a
// later unit.
//
// usage: test_rigid_body_transform_vs_openfoam <caseDir> <ofTimeDir>
#include "foam_field_reader.cuh"
#include "foam_token_reader.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "mesh_edges_cpp.cuh"
#include "point_patch_dist_cpp.cuh"
#include "primitive_mesh.cuh"
#include "primitive_patch_cpp.cuh"
#include "rigid_body_motion_cpp.cuh"
#include "point_constraints_cpp.cuh"
#include "septernion_cpp.cuh"
#include <cmath>
#include <cstdio>
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

Diff compare(const std::vector<vector>& a, const std::vector<vector>& b, scalar tol)
{
    Diff d;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t k = 0; k < n; ++k)
    {
        const scalar e = mag(a[k] - b[k]);
        if (e > d.worst) { d.worst = e; d.at = k; }
        if (e > tol) ++d.nAbove;
        d.refMax = std::fmax(d.refMax, mag(b[k]));
    }
    return d;
}


}   // namespace


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <ofTimeDir>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string ofDir = argv[2];

    std::printf("== brae rigid-body transform vs OpenFOAM: %s at %s ==\n", caseDir.c_str(), ofDir.c_str());

    const RBD::MotionSpec spec = RBD::readMotionSpec(caseDir + "/constant/dynamicMeshDict");
    std::printf("  body `%s`: %d links, %d degrees of freedom, innerDistance %g outerDistance %g\n",
                spec.bodyName.c_str(), (int)spec.model.links.size(), (int)spec.model.nDoF(),
                (double)spec.innerDistance, (double)spec.outerDistance);
    // THE CHAIN the composite joint makes, which is what a single-body reading would get wrong
    check("a composite (Py Ry) joint is TWO links: a jointBody through Py, then the body through Ry",
          spec.model.links.size() == 2
       && spec.model.links[0].joint == RBD::JointType::Py
       && spec.model.links[1].joint == RBD::JointType::Ry
       && spec.model.links[0].lambda == 0 && spec.model.links[1].lambda == 1
       && spec.model.links[0].qIndex == 0 && spec.model.links[1].qIndex == 1);
    check("...the body's own transform sits on the FIRST link and the identity on the second",
          mag(spec.model.links[1].XT.r) == scalar(0)
       && mag(spec.model.links[0].XT.r) > scalar(0));

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nPoints = m.nPoints();

    std::vector<label> ids;
    for (const std::string& pn : spec.patches)
    {
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].name == pn) ids.push_back(static_cast<label>(pi));
        }
    }
    check("the case has the body's patches", ids.size() == spec.patches.size() && !ids.empty());

    const MeshEdges e = buildMeshEdges(m);
    const PointPatchDist pd = pointPatchDist(m, e, patches, ids);
    const std::vector<scalar> weight =
        rigidBodyMeshMotionScale(pd.distance, spec.innerDistance, spec.outerDistance);

    const std::string statePath = ofDir + "/uniform/rigidBodyMotionState";
    const std::vector<scalar> q = RBD::readJointStateList(statePath, "q").value_or(std::vector<scalar>{});
    check("OpenFOAM's state file gives a q of the chain's size",
          q.size() == static_cast<std::size_t>(spec.model.nDoF()));
    if (q.size() != static_cast<std::size_t>(spec.model.nDoF()))
    {
        std::printf("test_rigid_body_transform_vs_openfoam: %d failures\n", ++failures);
        return 1;
    }
    std::printf("  OpenFOAM's joint state: q = (%.15g %.15g)\n", (double)q[0], (double)q[1]);
    check("...and it is not the rest state, so the transform under test is not the identity",
          mag(vector{q[0], q[1], 0}) > scalar(1e-9));

    const std::vector<vector>& points0 = m.points();
    const std::vector<vector> moved = spec.model.transformPoints(q, weight, points0);
    std::vector<vector> disp(static_cast<std::size_t>(nPoints));
    for (label p = 0; p < nPoints; ++p)
    {
        disp[static_cast<std::size_t>(p)] = moved[static_cast<std::size_t>(p)]
                                          - points0[static_cast<std::size_t>(p)];
    }

    // ...and the constraint rigidBodyMeshMotion::solve ends in, which pins every point a fixedValue
    // point patch owns -- the tank's own walls lie inside the body's outerDistance here
    PointConstraints::build(caseDir + "/0/pointDisplacement", m, patches, g.Sf()).constrainDisplacement(disp);

    const FieldData<vector> fd = readField<vector>(ofDir + "/pointDisplacement");
    const std::vector<vector> ofDisp = fd.internalUniform
        ? std::vector<vector>(static_cast<std::size_t>(nPoints), fd.internalUniformValue)
        : fd.internalField;
    check("OpenFOAM's pointDisplacement has one value per mesh point",
          ofDisp.size() == static_cast<std::size_t>(nPoints));

    const Diff dd = compare(disp, ofDisp, scalar(1e-12));
    std::printf("  pointDisplacement: worst %.4e of %.4e (relative %.3e), %zu of %d points above "
                "1e-12\n", (double)dd.worst, (double)dd.refMax, (double)dd.rel(), dd.nAbove, (int)nPoints);
    check("brae's point displacement is OpenFOAM's", dd.rel() < scalar(1e-11) && dd.nAbove == 0);

    // CONTROL 1: the same transform with NO weight -- the body's motion applied to the whole mesh,
    // which is what leaving the scale out of transformPoints gives.
    {
        const std::vector<scalar> one(static_cast<std::size_t>(nPoints), scalar(1));
        const std::vector<vector> all = spec.model.transformPoints(q, one, points0);
        std::vector<vector> d2(static_cast<std::size_t>(nPoints));
        for (label p = 0; p < nPoints; ++p)
        {
            d2[static_cast<std::size_t>(p)] = all[static_cast<std::size_t>(p)]
                                            - points0[static_cast<std::size_t>(p)];
        }
        const Diff dc = compare(d2, ofDisp, scalar(1e-12));
        std::printf("  CONTROL: the body's transform applied to EVERY point: worst %.4e (relative "
                    "%.3e) on %zu of %d points\n",
                    (double)dc.worst, (double)dc.rel(), dc.nAbove, (int)nPoints);
        check("...the weight is what confines the motion, and dropping it is a different mesh",
              dc.rel() > scalar(100)*std::fmax(dd.rel(), scalar(1e-16)) && dc.nAbove > 0);
    }

    // CONTROL 2: the slerp replaced by a LINEAR interpolation of the transform -- the same endpoints,
    // the same weight, a different path between them. It is the one thing in transformPoints that a
    // reading of "interpolate the transform" would get wrong.
    {
        const std::vector<RBD::SpatialTransform> x0 = spec.model.X0(q);
        const std::vector<RBD::SpatialTransform> x00 =
            spec.model.X0(std::vector<scalar>(static_cast<std::size_t>(spec.model.nDoF()), scalar(0)));
        const RBD::SpatialTransform X = RBD::inv(x0[static_cast<std::size_t>(spec.model.bodyID())])
                                      & x00[static_cast<std::size_t>(spec.model.bodyID())];
        std::vector<vector> d3(static_cast<std::size_t>(nPoints));
        std::size_t blended = 0;
        for (label p = 0; p < nPoints; ++p)
        {
            const std::size_t k = static_cast<std::size_t>(p);
            const scalar w = weight[k];
            const vector full = RBD::transformPoint(X, points0[k]);
            d3[k] = w*(full - points0[k]);
            if (w > scalar(1e-15) && w < scalar(1) - scalar(1e-15)) ++blended;
        }
        const Diff dc = compare(d3, ofDisp, scalar(1e-12));
        std::printf("  CONTROL: a LINEAR blend instead of the septernion slerp: worst %.4e (relative "
                    "%.3e) on %zu of %d points (%zu lie in the blend band)\n",
                    (double)dc.worst, (double)dc.rel(), dc.nAbove, (int)nPoints, blended);
        check("...the slerp is not a linear blend of the displacement",
              blended > 0 && dc.rel() > scalar(100)*std::fmax(dd.rel(), scalar(1e-16)));
    }

    // CONTROL 3: the rest state moves nothing at all
    {
        const std::vector<scalar> zero(static_cast<std::size_t>(spec.model.nDoF()), scalar(0));
        const std::vector<vector> still = spec.model.transformPoints(zero, weight, points0);
        scalar worst = 0;
        for (label p = 0; p < nPoints; ++p)
        {
            worst = std::fmax(worst, mag(still[static_cast<std::size_t>(p)]
                                       - points0[static_cast<std::size_t>(p)]));
        }
        std::printf("  CONTROL: q = 0 moves the mesh by %.4e\n", (double)worst);
        check("...at the rest state the transform is the identity", worst < scalar(1e-15));
    }

    std::printf("test_rigid_body_transform_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
