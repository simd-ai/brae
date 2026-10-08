// brae's PointConstraints against REAL OpenFOAM's pointConstraints::constrainDisplacement, on one staged
// pointDisplacement. The script (tests/point_constraints_vs_openfoam.sh) says what each arm stages and why.
//
// THE ORACLE is tools/dumpPointConstraints: OpenFOAM's own class on the same mesh and the same input
// (pointDisplacement.input), printing every symmetryPlane normal and every constrained point with its count
// and tensor, and writing the field after correctBoundaryConditions alone (.evaluated) and after the whole
// constrainDisplacement (.constrained).
//
// usage: test_point_constraints_vs_openfoam <caseDir> <oracleLog> [measure]
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "point_constraints_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;
bool measureOnly = false;

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

std::vector<vector> readPointField(const std::string& path)
{
    const FieldData<vector> fd = readField<vector>(path);
    return fd.internalField;
}

// points whose value differs in any bit, and the largest difference
struct Diff
{
    long   nBits = 0;
    scalar worst = 0;
};

Diff compare(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z)
        {
            ++d.nBits;
        }
        d.worst = std::fmax(d.worst, std::fabs(a[i].x - b[i].x));
        d.worst = std::fmax(d.worst, std::fabs(a[i].y - b[i].y));
        d.worst = std::fmax(d.worst, std::fabs(a[i].z - b[i].z));
    }
    if (a.size() != b.size())
    {
        d.nBits = -1;
    }
    return d;
}

// a bound on a count of points differing in any bit: 0, unless the harness only measures
void exact(
    const std::string& what,
    const Diff& d)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%-52s %ld points differ, worst %.4e", what.c_str(), d.nBits,
                  (double)d.worst);
    if (measureOnly)
    {
        std::printf("  meas: %s\n", buf);
        return;
    }
    check(buf, d.nBits == 0);
}

// a CONTROL: the broken form must differ from OpenFOAM in at least one point
void control(
    const std::string& what,
    const Diff& d)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), "CONTROL: %-43s %ld points differ, worst %.4e", what.c_str(), d.nBits,
                  (double)d.worst);
    check(buf, d.nBits > 0);
}

// `[brae] <key> ...` lines, parens as separators
std::vector<std::vector<std::string>> oracleLines(
    const std::string& path,
    const std::string& key)
{
    std::vector<std::vector<std::string>> out;
    std::ifstream in(path);
    std::string line;
    const std::string want = "[brae] " + key + " ";
    while (std::getline(in, line))
    {
        if (line.rfind(want, 0) != 0) continue;
        std::string rest = line.substr(want.size());
        for (char& ch : rest)
        {
            if (ch == '(' || ch == ')') ch = ' ';
        }
        std::istringstream is(rest);
        std::vector<std::string> toks;
        std::string t;
        while (is >> t) toks.push_back(t);
        out.push_back(toks);
    }
    return out;
}

} // namespace


int main(
    int argc,
    char** argv)
{
    std::printf("== brae pointConstraints vs OpenFOAM pointConstraints ==\n");
    if (argc < 3)
    {
        std::printf("  SKIP: usage: %s <caseDir> <oracleLog> [measure]\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string log = argv[2];
    measureOnly = (argc > 3 && std::string(argv[3]) == "measure");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    std::printf("  mesh: %d cells, %d points\n", (int)m.nCells(), (int)m.nPoints());

    const PointConstraints pc = PointConstraints::build(caseDir + "/0/pointDisplacement", m, patches, g.Sf());

    // THE NORMALS, bit for bit: gSum(faceAreas).normalise, frozen at construction
    {
        const auto lines = oracleLines(log, "normal");
        int nSym = 0;
        int nSame = 0;
        for (const PointPatchConstraint& c : pc.patchConstraints())
        {
            if (c.kind != PointPatchConstraint::Kind::symmetryPlane) continue;
            ++nSym;
            for (const auto& l : lines)
            {
                if (l.size() != 4 || l[0] != c.name) continue;
                const scalar x = std::stod(l[1]);
                const scalar y = std::stod(l[2]);
                const scalar z = std::stod(l[3]);
                const bool same = (x == c.n.x && y == c.n.y && z == c.n.z);
                std::printf("  normal %-10s brae (%.17g %.17g %.17g)%s\n", c.name.c_str(), (double)c.n.x,
                            (double)c.n.y, (double)c.n.z, same ? "" : "  <- OpenFOAM differs");
                nSame += same ? 1 : 0;
            }
        }
        check("OpenFOAM printed a normal for every symmetryPlane patch brae built",
              static_cast<int>(lines.size()) == nSym);
        check("...and every normal is OpenFOAM's, bit for bit", nSame == nSym && nSym > 0);
    }

    // THE CONSTRAINED POINTS, in OpenFOAM's order, with their counts and tensors
    {
        const auto lines = oracleLines(log, "corner");
        const std::vector<label>& pts = pc.cornerPoints();
        const std::vector<label>& cnt = pc.cornerCounts();
        const std::vector<tensor>& ten = pc.cornerTensors();
        long nOrder = 0;
        long nCount = 0;
        long nTensor = 0;
        long hist[4] = {0, 0, 0, 0};
        for (std::size_t k = 0; k < pts.size() && k < lines.size(); ++k)
        {
            const auto& l = lines[k];
            if (l.size() != 11)
            {
                ++nOrder;
                continue;
            }
            nOrder += (std::stol(l[0]) == pts[k]) ? 0 : 1;
            nCount += (std::stol(l[1]) == cnt[k]) ? 0 : 1;
            const tensor& T = ten[k];
            const scalar t[9] = {T.xx, T.xy, T.xz, T.yx, T.yy, T.yz, T.zx, T.zy, T.zz};
            bool same = true;
            for (int c = 0; c < 9; ++c)
            {
                same = same && (std::stod(l[2 + c]) == t[c]);
            }
            nTensor += same ? 0 : 1;
            hist[std::min<label>(cnt[k], 3)] += 1;
        }
        std::printf("  constrained points: OpenFOAM %zu, brae %zu (brae's counts: %ld one, %ld two, %ld three)\n",
                    lines.size(), pts.size(), hist[1], hist[2], hist[3]);
        check("brae constrains OpenFOAM's points, in OpenFOAM's order",
              lines.size() == pts.size() && nOrder == 0);
        check("...each with OpenFOAM's constraint count", nCount == 0);
        char buf[160];
        std::snprintf(buf, sizeof(buf), "...and OpenFOAM's tensor, bit for bit (%ld differ)", nTensor);
        check(buf, nTensor == 0);
    }

    const std::vector<vector> input = readPointField(caseDir + "/0/pointDisplacement.input");
    const std::vector<vector> ofEval = readPointField(caseDir + "/0/pointDisplacement.evaluated");
    const std::vector<vector> ofCons = readPointField(caseDir + "/0/pointDisplacement.constrained");
    check("the oracle wrote the input, the evaluated and the constrained fields, one value per point",
          input.size() == static_cast<std::size_t>(m.nPoints()) && ofEval.size() == input.size()
       && ofCons.size() == input.size());

    std::vector<vector> ev = input;
    pc.evaluate(ev);
    exact("correctBoundaryConditions, against .evaluated", compare(ev, ofEval));
    std::vector<vector> co = input;
    pc.constrainDisplacement(co);
    exact("constrainDisplacement, against .constrained", compare(co, ofCons));
    const Diff moved = compare(input, ofEval);
    const Diff cornered = compare(ofEval, ofCons);
    std::printf("  OpenFOAM against itself: the evaluate moves %ld points (worst %.4e); the corners %ld more "
                "(worst %.4e)\n", moved.nBits, (double)moved.worst, cornered.nBits, (double)cornered.worst);

    // THE CONTROLS, each a broken form that must NOT be OpenFOAM's
    {
        // the symmetryPlane evaluate skipped: only the pins
        std::vector<vector> v = input;
        for (const PointPatchConstraint& c : pc.patchConstraints())
        {
            if (c.kind != PointPatchConstraint::Kind::fixedValue) continue;
            for (const label p : c.meshPoints) v[static_cast<std::size_t>(p)] = c.value;
        }
        control("the symmetryPlane evaluate skipped", compare(v, ofEval));
    }
    {
        // the corners skipped: the evaluate alone, against the constrained field
        control("constrainCorners skipped", compare(ev, ofCons));
    }
    {
        // the pins written LAST, after every symmetryPlane: a point on both keeps the pin, not its projection
        std::vector<vector> v = input;
        pc.evaluate(v);
        for (const PointPatchConstraint& c : pc.patchConstraints())
        {
            if (c.kind != PointPatchConstraint::Kind::fixedValue) continue;
            for (const label p : c.meshPoints) v[static_cast<std::size_t>(p)] = c.value;
        }
        const Diff d = compare(v, ofEval);
        std::printf("  CONTROL (arm-dependent): the pins written last: %ld points differ, worst %.4e\n",
                    d.nBits, (double)d.worst);
        if (std::getenv("BRAE_EXPECT_PIN_ORDER"))
        {
            control("the pins written after the symmetryPlanes", d);
        }
    }
    {
        // the three-constraint corners left alone (identity for count 3)
        long n3 = 0;
        for (const label c : pc.cornerCounts()) n3 += (c == 3) ? 1 : 0;
        if (n3 > 0)
        {
            std::vector<vector> v = input;
            pc.evaluate(v);
            for (std::size_t k = 0; k < pc.cornerPoints().size(); ++k)
            {
                if (pc.cornerCounts()[k] == 3) continue;
                const tensor& T = pc.cornerTensors()[k];
                vector& x = v[static_cast<std::size_t>(pc.cornerPoints()[k])];
                x = vector{T.xx*x.x + T.xy*x.y + T.xz*x.z, T.yx*x.x + T.yy*x.y + T.yz*x.z,
                           T.zx*x.x + T.zy*x.y + T.zz*x.z};
            }
            control("the three-plane corners left free", compare(v, ofCons));
        }
        else
        {
            std::printf("  (no three-plane corner on this mesh: that control cannot run here)\n");
        }
    }

    std::printf("test_point_constraints_vs_openfoam: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
