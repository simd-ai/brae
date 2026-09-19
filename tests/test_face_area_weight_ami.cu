// brae's faceAreaWeightAMI against the weights OpenFOAM's own cyclicAMI holds, on RAS/mixerVesselAMI as
// its rotor turns.
//
// THE ORACLE is what a coded function object printed from OpenFOAM's running interFoam --
// AMI1's srcAddress, srcWeights and srcWeightsSum, and tgtAddress, tgtWeights and tgtWeightsSum -- after
// the step whose moved points OpenFOAM wrote to <timeDir>/polyMesh/points. brae recomputes the weights
// from those same points. tests/cyclic_ami_vs_openfoam.sh stages both.
//
// THE MOTION is brae's own: DynamicMotionSolverFvMesh turns the cellZone `rotating` step by step, as
// solidBodyMotionSolver does for the zone's points alone, and every moved point is held against the ones
// OpenFOAM wrote. The weights are then computed twice -- from OpenFOAM's points, which isolates the AMI,
// and from brae's, which is what a run would use.
//
// WHAT IS HELD: every face's set of partners is OpenFOAM's -- the target faces its advancing front
// reaches and keeps, which is NOT every overlapping face (see the control below) -- and every weight and
// weight sum agrees, at round-off.
#include "primitive_mesh.cuh"
#include "face_area_weight_ami_cpp.cuh"
#include "dynamic_motion_solver_fv_mesh_cpp.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

std::vector<vector> readPoints(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    static const std::regex head(R"(\n\s*([0-9]+)\s*\n?\()");
    std::smatch mh;
    std::vector<vector> out;
    if (!std::regex_search(text, mh, head)) return out;
    const std::size_t n = static_cast<std::size_t>(std::atol(mh[1].str().c_str()));
    const char* p = text.c_str() + mh.position(0) + mh.length(0);
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        while (*p && *p != '(')
        {
            ++p;
        }
        if (!*p) break;
        ++p;
        char* end = nullptr;
        vector v;
        v.x = std::strtod(p, &end);
        p = end;
        v.y = std::strtod(p, &end);
        p = end;
        v.z = std::strtod(p, &end);
        p = end;
        out.push_back(v);
    }
    return out;
}

struct Side
{
    std::vector<scalar> sum;
    std::vector<std::map<label, scalar>> w;
};

// "src N", then N lines "i sum n  a0 w0  a1 w1 ...", then the same for "tgt"
bool readDump(
    const std::string& path,
    Side& src,
    Side& tgt)
{
    std::ifstream in(path);
    if (!in) return false;
    for (Side* s : {&src, &tgt})
    {
        std::string tag;
        std::size_t n = 0;
        in >> tag >> n;
        s->sum.resize(n);
        s->w.resize(n);
        for (std::size_t k = 0; k < n; ++k)
        {
            std::size_t i = 0;
            std::size_t m = 0;
            double sum = 0;
            in >> i >> sum >> m;
            s->sum[i] = sum;
            for (std::size_t j = 0; j < m; ++j)
            {
                long a = 0;
                double w = 0;
                in >> a >> w;
                s->w[i][static_cast<label>(a)] = w;
            }
        }
    }
    return static_cast<bool>(in);
}

cpu::ami::Patch patchOf(
    const PrimitiveMesh& m,
    const std::string& name,
    const std::vector<vector>& points)
{
    cpu::ami::Patch p;
    p.points = &points;
    for (const PatchInfo& pi : m.patches())
    {
        if (pi.name != name) continue;
        for (label f = pi.start; f < pi.start + pi.size; ++f)
        {
            std::vector<label> face(static_cast<std::size_t>(m.faceSize(f)));
            for (label k = 0; k < m.faceSize(f); ++k)
            {
                face[static_cast<std::size_t>(k)] = m.faceVert(f, k);
            }
            p.faces.push_back(face);
        }
    }
    return p;
}

// the worst difference, the partner sets and the count of faces compared
void compareSide(
    const char* name,
    const std::vector<std::vector<label>>& addr,
    const std::vector<std::vector<scalar>>& wt,
    const std::vector<scalar>& sum,
    const Side& of,
    std::size_t& setsDiffer,
    scalar& worstW,
    scalar& worstSum)
{
    setsDiffer = 0;
    worstW = 0;
    worstSum = 0;
    for (std::size_t i = 0; i < addr.size() && i < of.w.size(); ++i)
    {
        worstSum = std::fmax(worstSum, std::fabs(sum[i] - of.sum[i]));
        if (addr[i].size() != of.w[i].size())
        {
            ++setsDiffer;
            continue;
        }
        for (std::size_t j = 0; j < addr[i].size(); ++j)
        {
            const auto it = of.w[i].find(addr[i][j]);
            if (it == of.w[i].end())
            {
                ++setsDiffer;
                break;
            }
            worstW = std::fmax(worstW, std::fabs(wt[i][j] - it->second));
        }
    }
    std::printf("  %s: %zu faces, %zu with a different partner set; weights %.3e apart, weight sums %.3e\n",
                name, addr.size(), setsDiffer, (double)worstW, (double)worstSum);
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae faceAreaWeightAMI vs OpenFOAM's cyclicAMI weights ==\n");
    if (argc < 4)
    {
        std::printf("  SKIP: usage: %s <caseDir> <timeDir> <amiDump> [<timeDir> <amiDump> ...]\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> patches = buildPatches(m, g);
    std::unique_ptr<DynamicMotionSolverFvMesh> motion = DynamicMotionSolverFvMesh::New(caseDir, caseDir + "/0");
    check("the case's mesh moves", motion != nullptr);
    if (!motion)
    {
        return 1;
    }
    motion->attach(m, g, patches);
    const std::size_t nZonePoints = motion->pointIDs().size();
    std::printf("  the cellZone motion moves %zu of %zu points\n", nZonePoints, m.points().size());
    check("part of the mesh moves, not all of it", nZonePoints > 0 && nZonePoints < m.points().size());
    scalar tPrev = 0;
    label timeIndex = 0;
    // THE CONTROL: faces OpenFOAM's advancing front leaves less than 99% covered. The walk reaches a
    // target face only through the neighbours of one it has already found, so a partner can be missed
    // where the rotor has slid a sliver of a face over: an all-pairs search covers every face in full.
    std::size_t partlyCovered = 0;
    for (int a = 2; a + 1 < argc; a += 2)
    {
        const std::string timeDir = argv[a];
        const std::string dump = argv[a + 1];
        std::printf("  -- %s\n", timeDir.c_str());
        const std::vector<vector> points = readPoints(timeDir + "/polyMesh/points");
        check("OpenFOAM wrote the moved points", points.size() == m.points().size());
        if (points.size() != m.points().size()) continue;

        // brae's own motion to the same time; the directory's name is the time OpenFOAM wrote
        const std::string base = timeDir.substr(timeDir.find_last_of('/') + 1);
        const scalar t = std::strtod(base.c_str(), nullptr);
        ++timeIndex;
        motion->update(t, t - tPrev, timeIndex);
        tPrev = t;
        scalar worstPoint = 0;
        scalar worstStill = 0;
        std::vector<char> moves(points.size(), 0);
        for (const label pointi : motion->pointIDs())
        {
            moves[static_cast<std::size_t>(pointi)] = 1;
        }
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            const scalar d = mag(m.points()[i] - points[i]);
            worstPoint = std::fmax(worstPoint, d);
            if (!moves[i])
            {
                worstStill = std::fmax(worstStill, mag(points[i] - motion->points0()[i]));
            }
        }
        std::printf("  moved points %.3e from OpenFOAM's; the points outside the zone moved %.3e in OpenFOAM\n",
                    (double)worstPoint, (double)worstStill);
        check("every moved point is OpenFOAM's", worstPoint < 1e-15);
        check("OpenFOAM moved no point outside the zone", worstStill == 0);

        Side ofSrc;
        Side ofTgt;
        check("OpenFOAM's weights were read", readDump(dump, ofSrc, ofTgt));
        std::size_t partly = 0;
        for (const scalar ws : ofSrc.sum)
        {
            partly += (ws < scalar(0.99)) ? 1 : 0;
        }
        std::printf("  OpenFOAM leaves %zu source faces less than 99%% covered\n", partly);
        partlyCovered += partly;
        for (int arm = 0; arm < 2; ++arm)
        {
            const std::vector<vector>& pts = (arm == 0) ? points : m.points();
            std::printf("  weights from %s\n", arm == 0 ? "OpenFOAM's points" : "brae's points");
            const cpu::ami::Patch src = patchOf(m, "AMI1", pts);
            const cpu::ami::Patch tgt = patchOf(m, "AMI2", pts);
            check("the pair has faces, as many as OpenFOAM's", !src.faces.empty() && src.faces.size() == ofSrc.w.size()
                  && tgt.faces.size() == ofTgt.w.size());
            const cpu::ami::Weights W = cpu::ami::faceAreaWeight(src, tgt);
            std::size_t multi = 0;
            for (const auto& ad : W.srcAddress)
            {
                multi += (ad.size() > 1) ? 1 : 0;
            }
            std::printf("  %zu of %zu source faces couple to more than one target face\n", multi, W.srcAddress.size());
            std::size_t dS = 0;
            std::size_t dT = 0;
            scalar wS = 0;
            scalar sS = 0;
            scalar wT = 0;
            scalar sT = 0;
            compareSide("source", W.srcAddress, W.srcWeights, W.srcWeightsSum, ofSrc, dS, wS, sS);
            compareSide("target", W.tgtAddress, W.tgtWeights, W.tgtWeightsSum, ofTgt, dT, wT, sT);
            check("the interface is non-conformal here", multi > 0);
            check("every face has OpenFOAM's partners", dS == 0 && dT == 0);
            check("every weight is OpenFOAM's", wS < 5e-13 && wT < 5e-13);
            check("every weight sum is OpenFOAM's", sS < 5e-13 && sT < 5e-13);
        }
    }
    check("OpenFOAM's walk leaves some face partly covered, so the partner sets can tell a walk from a search",
          partlyCovered > 0);
    std::printf("test_face_area_weight_ami: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
