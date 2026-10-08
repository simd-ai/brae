// cyclicPeriodicAMI -- OF meshTools/AMIInterpolation/patches/cyclicPeriodicAMI.
//
// An AMI whose two sides need not span the same sector. OpenFOAM covers the shortfall by applying the
// transform of a NAMED periodic patch to one side repeatedly, accumulating the extra overlaps, until the
// weights sum to 1 (cyclicPeriodicAMIPolyPatch::resetAMI + AMIInterpolation::append). Coupling a source
// face to the periodic IMAGE of a target face is exact rather than approximate: the transform is a
// symmetry of the solution, so the value at the image IS the value at the face.
//
// THE FIXTURE IS SYNTHETIC AND DELIBERATELY SO. A box has exactly one translation per axis, so a
// physically-arranged periodic AMI cannot be built from one -- the pair that needs tiling and the pair
// that supplies the transform would have to be different patches separated along the SAME axis. Here the
// x-normal pair is both, which is not a mesh OpenFOAM would accept but is the exact code path: two
// patches with no overlap at all, and a period that brings them onto each other in one image. That makes
// Leg 3 unambiguous in a way a real mesh would not -- coverage is 0 before tiling and 1 after, with no
// partial overlap muddying which of the two produced it.
//
// LEG 2 IS THE ONE THAT MATTERS MOST. cyclicAMI infers a translational period from the two patches'
// centroid difference, and for a periodic AMI that inference is fatal: its sides are CO-LOCATED, so the
// offset between them IS the sector mismatch the tiling exists to cover. Subtracting it re-aligns the
// patches, the raw overlap comes out fully covered, and the tiling loop then runs ZERO images because it
// has nothing left to do -- an interface that reports perfect coverage while coupling the wrong faces.
// That is not hypothetical: it is what the first working build of this feature did, and the tell was
// `images:0 srcSum:1` on a patch pair that visibly needed images.
#include "box_mesh.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "interface/ami_interface.cuh"
#include "foam_dict.cuh"   // isCoupledInterfaceType / isConstraintPatchType
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

struct Spec
{
    std::string type;            // what `inlet`/`outlet` become
    std::string periodicPatch;   // periodicPatch entry on both (empty = none)
    label       maxIter = 36;
    scalar      shear = 0;       // slide the outlet's points this far in +y (see below)
};

PrimitiveMesh rigged(const PrimitiveMesh& src, const Spec& s)
{
    // THE SHEAR IS WHAT MAKES THE FIXTURE MEAN ANYTHING. faceAreaWeightAMI projects each source/target
    // pair onto the plane perpendicular to their pair normal, which DISCARDS any separation along that
    // normal -- correct for a real AMI, whose sides are co-located, but it means two box faces a whole
    // box-length apart still project onto each other perfectly. A fixture that relies on them "not
    // overlapping" therefore tests nothing, and reports full coverage with zero images.
    //
    // Sliding the outlet's points in +y gives a genuine TANGENTIAL offset instead: the two patches now
    // overlap over (Ly - shear) of their height and miss over `shear`, which is exactly the sector
    // mismatch a periodic AMI exists to close -- and the y-period that closes it is the one wallYmin
    // and wallYmax define.
    std::vector<vector> pts = src.points();
    if (s.shear != scalar(0))
    {
        for (const PatchInfo& q : src.patches())
            if (q.name == "outlet")
                for (label f = q.start; f < q.start + q.size; ++f)
                    for (label k = src.faceOffsets()[f]; k < src.faceOffsets()[f+1]; ++k)
                        pts[src.faceVerts()[k]].y += s.shear;
    }
    std::vector<PatchInfo> pp = src.patches();
    for (PatchInfo& q : pp)
    {
        if (q.name == "inlet")  { q.type = s.type; q.neighbourPatch = "outlet"; }
        if (q.name == "outlet") { q.type = s.type; q.neighbourPatch = "inlet";  }
        // wallYmin/wallYmax become the PERIODIC pair: a plain cyclic whose two halves are a box height
        // apart in y, which is the transform the tiling below applies. A wall defines no transform, so
        // without this the periodic AMI has nothing to tile with and says so.
        if (q.name == "wallYmin") { q.type = "cyclic"; q.neighbourPatch = "wallYmax"; q.transform = "unknown"; }
        if (q.name == "wallYmax") { q.type = "cyclic"; q.neighbourPatch = "wallYmin"; q.transform = "unknown"; }
        if (q.name == "inlet" || q.name == "outlet")
        {
            q.periodicPatch = s.periodicPatch;
            q.maxIter = s.maxIter;
            q.matchTolerance = 1e-4;
            q.transform = "unknown";
        }
    }
    PrimitiveMesh out;
    out.assign(std::move(pts), src.faceVerts(), src.faceOffsets(), src.owner(), src.neighbour(),
               std::move(pp), src.nCells());
    return out;
}

// Build the AMI interfaces of a rigged mesh; returns the error text if it refused.
std::string build(const PrimitiveMesh& m, std::vector<AMIInterface>& out)
{
    try
    {
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        out = buildAMIInterfaces(m, g, fvp);
        return "";
    }
    catch (const std::exception& e) { return e.what(); }
}

const AMIInterface* find(const std::vector<AMIInterface>& v, const std::vector<FvPatch>& fvp,
                         const std::string& name)
{
    for (const AMIInterface& a : v)
        if (a.patch >= 0 && a.patch < (label)fvp.size() && fvp[a.patch].name == name) return &a;
    return nullptr;
}

scalar meanCoverage(const AMIInterface& a)
{
    if (a.coverage.empty()) return 0;
    scalar s = 0;
    for (const scalar w : a.coverage)
    {
        s += w;
    }
    return s/(scalar)a.coverage.size();
}
} // namespace

int main()
{
    std::printf("== cyclicPeriodicAMI (periodic tiling of an AMI) ==\n");

    const PrimitiveMesh base = boxtest::boxMesh(4, 3, 2);
    FvGeometry gBase; gBase.build(base);
    const std::vector<FvPatch> fvpBase = buildPatches(base, gBase);

    // ---- Leg 1: the type is coupled at all ---------------------------------------------------------
    {
        check(isCoupledInterfaceType("cyclicPeriodicAMI"),
              "cyclicPeriodicAMI is a COUPLED interface type (so the boundary machinery skips it)");
        check(isConstraintPatchType("cyclicPeriodicAMI"),
              "...and a constraint type, so a field file may omit its boundaryField entry");
    }

    // ---- Leg 2: the period is NOT inferred from the centroids --------------------------------------
    // Same geometry, two types, opposite answers -- which is the whole point.
    {
        std::vector<AMIInterface> ami, per;
        const std::string e1 = build(rigged(base, {"cyclicAMI", "", 36, 0.0}), ami);
        const std::string e2 = build(rigged(base, {"cyclicPeriodicAMI", "wallYmin", 36, 0.0}), per);

        FvGeometry g; g.build(base);
        const std::vector<FvPatch> fvp = buildPatches(base, g);

        // cyclicAMI: the x-separated pair gets a period inferred, which re-aligns them
        const AMIInterface* a = e1.empty() ? find(ami, fvp, "inlet") : nullptr;
        check(a != nullptr, "vacuity guard: the cyclicAMI control built an interface");
        if (a) check(mag(a->separation) > 0.5,
                     "cyclicAMI DOES infer a translational period from the centroid difference");

        const AMIInterface* b = e2.empty() ? find(per, fvp, "inlet") : nullptr;
        if (!e2.empty()) std::printf("        (refused: %s)\n", e2.c_str());
        check(b != nullptr, "cyclicPeriodicAMI builds an interface");
        if (b) check(mag(b->separation) == scalar(0),
                     "cyclicPeriodicAMI infers NO period -- its sides are co-located by definition");
    }

    // ---- Leg 3: the tiling is what achieves coverage -----------------------------------------------
    // The sheared fixture overlaps over (Ly - 0.5)/Ly of its height untransformed. maxIter 0 forbids
    // every image, leaving that partial coverage for the refusal to catch; with images allowed, the
    // y-period wraps the missing strip round and the coverage closes to 1.
    {
        std::vector<AMIInterface> none, tiled;
        const std::string e0 = build(rigged(base, {"cyclicPeriodicAMI", "wallYmin", 0, 0.5}), none);
        check(!e0.empty(), "with maxIter 0 the untransformed overlap is only partial and the coverage refusal fires");
        check(e0.find("coverage") != std::string::npos,
              "...and it says so -- the sheared patches miss each other over the wrapped strip");

        const std::string e1 = build(rigged(base, {"cyclicPeriodicAMI", "wallYmin", 36, 0.5}), tiled);
        if (!e1.empty()) std::printf("        (refused: %s)\n", e1.c_str());
        check(e1.empty(), "with images allowed the same pair builds cleanly");
        FvGeometry g; g.build(base);
        const std::vector<FvPatch> fvp = buildPatches(base, g);
        const AMIInterface* t = e1.empty() ? find(tiled, fvp, "inlet") : nullptr;
        check(t != nullptr, "vacuity guard: the tiled interface exists");
        if (t) check(std::fabs(meanCoverage(*t) - scalar(1)) < 1e-9,
                     "the periodic image covers the source face completely (mean coverage 1)");
    }

    // ---- Leg 4: refusals ---------------------------------------------------------------------------
    {
        std::vector<AMIInterface> v;
        const std::string e = build(rigged(base, {"cyclicPeriodicAMI", "notAPatch", 36, 0.0}), v);
        check(!e.empty() && e.find("notAPatch") != std::string::npos,
              "refusal: a periodicPatch that is not a patch of this mesh, named in the message");
    }
    {
        std::vector<AMIInterface> v;
        const std::string e = build(rigged(base, {"cyclicPeriodicAMI", "", 36, 0.0}), v);
        check(!e.empty(), "refusal: a cyclicPeriodicAMI with no periodicPatch has no transform to tile with");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
