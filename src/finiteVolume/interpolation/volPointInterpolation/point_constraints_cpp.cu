#include "point_constraints_cpp.cuh"

#include "foam_field_reader.cuh"
#include "patch_entry_lookup.cuh"
#include "primitive_patch_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace brae {

namespace {

const char* const WHO = "brae pointConstraints: ";

// OpenFOAM's pointConstraint, Tuple2<label, vector>: the number of constraints and a direction
struct PointConstraint
{
    label  first = 0;
    vector second{0, 0, 0};
};

// pointConstraintI.H:60-86
void applyConstraint(
    PointConstraint& pc,
    const vector&    cd)
{
    if (pc.first == 0)
    {
        pc.first = 1;
        pc.second = cd;
    }
    else if (pc.first == 1)
    {
        // cd ^ second(), VectorI.H:163-172
        const vector planeNormal = cross(cd, pc.second);
        const scalar magPlaneNormal =
            std::sqrt(planeNormal.x*planeNormal.x + planeNormal.y*planeNormal.y + planeNormal.z*planeNormal.z);
        if (magPlaneNormal > scalar(1e-3))
        {
            pc.first = 2;
            pc.second = vector{planeNormal.x/magPlaneNormal, planeNormal.y/magPlaneNormal,
                               planeNormal.z/magPlaneNormal};
        }
    }
    else if (pc.first == 2)
    {
        const scalar d = cd.x*pc.second.x + cd.y*pc.second.y + cd.z*pc.second.z;
        if (std::fabs(d) > scalar(1e-3))
        {
            pc.first = 3;
            pc.second = vector{0, 0, 0};
        }
    }
}

// pointConstraintI.H:130-148. sqr(v) is the symmTensor (x*x, x*y, x*z, y*y, y*z, z*z) (SymmTensorI.H:612-620),
// I - S negates the off-diagonals (:679-687), and the tensor is its symmetric fill (TensorI.H:74-79).
tensor constraintTransformation(const PointConstraint& pc)
{
    const vector& v = pc.second;
    if (pc.first == 1)
    {
        const scalar xx = v.x*v.x;
        const scalar xy = v.x*v.y;
        const scalar xz = v.x*v.z;
        const scalar yy = v.y*v.y;
        const scalar yz = v.y*v.z;
        const scalar zz = v.z*v.z;
        return tensor{scalar(1) - xx, -xy, -xz,
                      -xy, scalar(1) - yy, -yz,
                      -xz, -yz, scalar(1) - zz};
    }
    if (pc.first == 2)
    {
        const scalar xx = v.x*v.x;
        const scalar xy = v.x*v.y;
        const scalar xz = v.x*v.z;
        const scalar yy = v.y*v.y;
        const scalar yz = v.y*v.z;
        const scalar zz = v.z*v.z;
        return tensor{xx, xy, xz, xy, yy, yz, xz, yz, zz};
    }
    return tensor{0, 0, 0, 0, 0, 0, 0, 0, 0};
}

// tensor & vector, row by row, summed left to right (TensorI.H:1207-1214)
vector dotTV(
    const tensor& T,
    const vector& v)
{
    return vector{T.xx*v.x + T.xy*v.y + T.xz*v.z,
                  T.yx*v.x + T.yy*v.y + T.yz*v.z,
                  T.zx*v.x + T.zy*v.y + T.zz*v.z};
}

} // namespace


PointConstraints PointConstraints::build(
    const std::string&          fieldPath,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    const std::vector<vector>&  Sf)
{
    const FieldData<vector> fd = readField<vector>(fieldPath);
    PointConstraints pc;
    std::vector<PatchEdgeAddressing> edges;
    pc.patches_.reserve(patches.size());
    for (const FvPatch& q : patches)
    {
        PointPatchConstraint c;
        c.name = q.name;
        const PatchFieldData<vector>* entry = findPatchEntry(fd.boundary, q);
        if (q.type == "symmetryPlane")
        {
            // pointPatchFieldNew.C:143-161: the constraint type replaces the entry unless `patchType` keeps it
            if (entry && !entry->patchType.empty())
            {
                throw std::runtime_error(
                    std::string(WHO) + fieldPath + ": patch `" + q.name + "` names `patchType "
                    + entry->patchType + "`, which keeps the entry's own type on a constraint patch. Not "
                    "ported.");
            }
            c.kind = PointPatchConstraint::Kind::symmetryPlane;
        }
        else if (q.type == "patch" || q.type == "wall")
        {
            if (!entry)
            {
                throw std::runtime_error(
                    std::string(WHO) + fieldPath + " has no entry for the patch `" + q.name + "`.");
            }
            if (entry->type == "calculated")
            {
                // pointPatchField::evaluate only toggles `updated` (pointPatchField.C:302-310)
                c.kind = PointPatchConstraint::Kind::calculated;
            }
            else if (entry->type == "fixedValue")
            {
                if (!entry->valueUniform)
                {
                    throw std::runtime_error(
                        std::string(WHO) + "the point patch `" + q.name + "` is a fixedValue with a per-point "
                        "value; only a uniform one is ported.");
                }
                c.kind = PointPatchConstraint::Kind::fixedValue;
                c.value = entry->uniformValue;
            }
            else
            {
                throw std::runtime_error(
                    std::string(WHO) + "the point patch `" + q.name + "` is `" + entry->type + "`. Only "
                    "fixedValue (which pins the points it owns), calculated (which does not evaluate) and a "
                    "symmetryPlane mesh patch are ported; every other pointPatchField writes something of "
                    "its own into the shared point field and would move the mesh differently.");
            }
        }
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "the mesh patch `" + q.name + "` is `" + q.type + "`. Only patch, wall and "
                "symmetryPlane are ported: symmetry, wedge, empty and the cyclics apply constraints of their "
                "own (or are coupled), and a mesh with an empty or wedge patch is corrected by "
                "twoDPointCorrector as well (pointConstraints.C:419-423).");
        }

        const PrimitivePatchAddressing addr = primitivePatch(m, faceRange(q.start, q.size));
        c.meshPoints = addr.meshPoints;

        if (c.kind == PointPatchConstraint::Kind::symmetryPlane && q.size > 0)
        {
            // symmetryPlanePolyPatch.C:48-58: gSum(faceAreas()) from Zero, in face order, then
            // normalise(ROOTVSMALL) -- mag, and divide each component, or Zero below ROOTVSMALL
            const scalar rootVSmall = scalar(1e-150);
            vector s{0, 0, 0};
            for (label f = 0; f < q.size; ++f)
            {
                const vector& a = Sf[static_cast<std::size_t>(q.start + f)];
                s.x += a.x;
                s.y += a.y;
                s.z += a.z;
            }
            const scalar ms = std::sqrt(s.x*s.x + s.y*s.y + s.z*s.z);
            c.n = (ms < rootVSmall) ? vector{0, 0, 0} : vector{s.x/ms, s.y/ms, s.z/ms};
            // :61-90, the planarity check: a FatalError in OpenFOAM
            for (label f = 0; f < q.size; ++f)
            {
                const vector& a = Sf[static_cast<std::size_t>(q.start + f)];
                const scalar ma = std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
                if (ma > rootVSmall)
                {
                    const vector nf{a.x/ma, a.y/ma, a.z/ma};
                    const vector dn{c.n.x - nf.x, c.n.y - nf.y, c.n.z - nf.z};
                    if (dn.x*dn.x + dn.y*dn.y + dn.z*dn.z > scalar(1e-15))
                    {
                        throw std::runtime_error(
                            std::string(WHO) + "symmetry plane `" + q.name + "` is not planar: face "
                            + std::to_string((long)f) + "'s normal differs from the average normal. "
                            "OpenFOAM stops here (symmetryPlanePolyPatch.C:74-89).");
                    }
                }
            }
        }
        pc.patches_.push_back(std::move(c));
        edges.push_back(patchEdges(addr));
    }

    // pointConstraints.C:108-146: every non-empty, non-coupled face patch's boundaryPoints (the points of its
    // boundary edges, ascending by local label -- sortedToc), in patch order, each mesh point taking a slot at
    // its first appearance; only a constraint patch's applyConstraint changes the slot
    std::unordered_map<label, std::size_t> slotOf;
    std::vector<label> slotPoint;
    std::vector<PointConstraint> slotConstraint;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const PatchEdgeAddressing& pe = edges[pi];
        std::vector<label> bp;
        for (std::size_t e = static_cast<std::size_t>(pe.nInternalEdges); e < pe.start.size(); ++e)
        {
            bp.push_back(pe.start[e]);
            bp.push_back(pe.end[e]);
        }
        std::sort(bp.begin(), bp.end());
        bp.erase(std::unique(bp.begin(), bp.end()), bp.end());
        const PointPatchConstraint& c = pc.patches_[pi];
        for (const label lp : bp)
        {
            const label mp = c.meshPoints[static_cast<std::size_t>(lp)];
            auto it = slotOf.find(mp);
            std::size_t slot = 0;
            if (it != slotOf.end())
            {
                slot = it->second;
            }
            else
            {
                slot = slotPoint.size();
                slotOf.emplace(mp, slot);
                slotPoint.push_back(mp);
                slotConstraint.push_back(PointConstraint{});
            }
            // symmetryPlanePointPatch.C:80-87: the plane's normal, whatever the point
            if (c.kind == PointPatchConstraint::Kind::symmetryPlane)
            {
                applyConstraint(slotConstraint[slot], c.n);
            }
        }
    }
    // pointConstraints.C:314-331: only the non-trivial ones are kept, in slot order
    for (std::size_t k = 0; k < slotPoint.size(); ++k)
    {
        if (slotConstraint[k].first != 0)
        {
            pc.cornerPoints_.push_back(slotPoint[k]);
            pc.cornerTensors_.push_back(constraintTransformation(slotConstraint[k]));
            pc.cornerCounts_.push_back(slotConstraint[k].first);
        }
    }
    return pc;
}


void PointConstraints::evaluate(std::vector<vector>& d) const
{
    // GeometricBoundaryField::evaluate, nonBlocking: every initEvaluate (none here) and then every evaluate,
    // in boundary order (GeometricBoundaryField.C:576-587)
    for (const PointPatchConstraint& c : patches_)
    {
        if (c.kind == PointPatchConstraint::Kind::fixedValue)
        {
            // valuePointPatchField.C:203-211, setInInternalField
            for (const label mp : c.meshPoints)
            {
                d[static_cast<std::size_t>(mp)] = c.value;
            }
        }
        else if (c.kind == PointPatchConstraint::Kind::symmetryPlane)
        {
            // symmetryPlanePointPatchField.C:103-123: (pif + transform(I - 2.0*sqr(n), pif))/2.0, the whole
            // patch gathered before any point is written back
            const vector& n = c.n;
            const scalar txx = scalar(1) - scalar(2.0)*(n.x*n.x);
            const scalar txy = -(scalar(2.0)*(n.x*n.y));
            const scalar txz = -(scalar(2.0)*(n.x*n.z));
            const scalar tyy = scalar(1) - scalar(2.0)*(n.y*n.y);
            const scalar tyz = -(scalar(2.0)*(n.y*n.z));
            const scalar tzz = scalar(1) - scalar(2.0)*(n.z*n.z);
            const tensor T{txx, txy, txz, txy, tyy, tyz, txz, tyz, tzz};
            std::vector<vector> vals(c.meshPoints.size());
            for (std::size_t i = 0; i < c.meshPoints.size(); ++i)
            {
                const vector& v = d[static_cast<std::size_t>(c.meshPoints[i])];
                const vector t = dotTV(T, v);
                vals[i] = vector{(v.x + t.x)/scalar(2.0), (v.y + t.y)/scalar(2.0), (v.z + t.z)/scalar(2.0)};
            }
            for (std::size_t i = 0; i < c.meshPoints.size(); ++i)
            {
                d[static_cast<std::size_t>(c.meshPoints[i])] = vals[i];
            }
        }
    }
}


void PointConstraints::constrainCorners(std::vector<vector>& d) const
{
    for (std::size_t k = 0; k < cornerPoints_.size(); ++k)
    {
        vector& v = d[static_cast<std::size_t>(cornerPoints_[k])];
        v = dotTV(cornerTensors_[k], v);
    }
}


void PointConstraints::constrainDisplacement(std::vector<vector>& d) const
{
    evaluate(d);
    constrainCorners(d);
}

} // namespace brae
