// brae's solid-body mesh motion against REAL OpenFOAM's, with no flow solver in the way.
//
// THE ORACLE is OpenFOAM's moveDynamicMesh, which runs mesh.update() in a time loop and writes, per
// time step, the moved polyMesh/points and the mesh-motion flux meshPhi -- byte for byte what interFoam
// writes for the same case (checked once, by hand, on testTubeMixer) -- and postProcess's
// writeCellVolumes and writeCellCentres run on those time directories, which give V and C of the moved
// mesh as OpenFOAM's own primitiveMesh computes them.
//
// tests/mesh_motion_vs_openfoam.sh says what each profile is for and what it measured.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "dynamic_motion_solver_fv_mesh_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

// an ASCII vectorField file: a count, then that many (x y z)
std::vector<vector> readPointsFile(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    // the list starts at the first line that is a bare count followed by "("
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

bool fileExists(const std::string& path)
{
    return std::ifstream(path).good();
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae solid-body mesh motion vs OpenFOAM moveDynamicMesh ==\n");
    if (argc < 5)
    {
        std::printf("  SKIP: usage: %s <caseDir> <profile> <deltaT> <timeDir>...\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string profile = argv[2];
    const scalar deltaT = std::atof(argv[3]);
    std::printf("  profile: %s\n", profile.c_str());

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> patches = buildPatches(m, g);
    const std::vector<scalar> VStart = g.V();

    std::unique_ptr<DynamicMotionSolverFvMesh> mesh = DynamicMotionSolverFvMesh::New(caseDir, caseDir + "/0");
    check("the case asks for a mesh motion, and brae built one", mesh != nullptr);
    if (!mesh) return 1;
    mesh->attach(m, g, patches);
    std::printf("  motion function: %s\n", mesh->motionType().c_str());

    // the scale every absolute point difference is read against
    scalar lengthScale = 0;
    for (const vector& p : m.points())
    {
        lengthScale = std::fmax(lengthScale, mag(p));
    }

    // THE BASELINE: V and C of the mesh BEFORE it moves, against OpenFOAM's for the same points. It
    // says how much of any later V or C difference is the geometry code's own and not the motion's.
    scalar startV = -1;
    scalar startC = -1;
    if (fileExists(caseDir + "/0/V") && fileExists(caseDir + "/0/C"))
    {
        const FieldData<scalar> ofV = readField<scalar>(caseDir + "/0/V");
        const FieldData<vector> ofC = readField<vector>(caseDir + "/0/C");
        startV = 0;
        startC = 0;
        for (std::size_t c = 0; c < g.V().size() && c < ofV.internalField.size(); ++c)
        {
            startV = std::fmax(startV, std::fabs(g.V()[c] - ofV.internalField[c])/ofV.internalField[c]);
            startC = std::fmax(startC, mag(g.C()[c] - ofC.internalField[c])/lengthScale);
        }
        std::printf("  before any motion: V %.3e relative, C %.3e of the mesh's extent\n", (double)startV,
                    (double)startC);
    }

    scalar worstPoint = 0;
    scalar worstPhi = 0;
    scalar worstV = 0;
    scalar worstC = 0;
    scalar worstV0 = 0;
    scalar largestMove = 0;
    scalar largestPhi = 0;
    int nV = 0;
    // Time::operator++: the value ACCUMULATES, and the motion is a function of that value
    scalar t = 0;
    for (int k = 4; k < argc; ++k)
    {
        const std::string timeDir = argv[k];
        const label timeIndex = static_cast<label>(k - 3);
        const std::vector<scalar> VBefore = g.V();
        t = t + deltaT;
        mesh->update(t, deltaT, timeIndex);

        const std::vector<vector> ofPoints = readPointsFile(timeDir + "/polyMesh/points");
        check("OpenFOAM wrote as many points as the mesh has", ofPoints.size() == m.points().size());
        scalar dPoint = 0;
        for (std::size_t i = 0; i < ofPoints.size() && i < m.points().size(); ++i)
        {
            dPoint = std::fmax(dPoint, mag(m.points()[i] - ofPoints[i]));
            largestMove = std::fmax(largestMove, mag(m.points()[i] - mesh->points0()[i]));
        }
        worstPoint = std::fmax(worstPoint, dPoint/lengthScale);

        // V0 is the volume the mesh had BEFORE this update, not the start's
        for (std::size_t c = 0; c < VBefore.size(); ++c)
        {
            worstV0 = std::fmax(worstV0, std::fabs(mesh->V0()[c] - VBefore[c]));
        }

        const FieldData<scalar> ofPhi = readField<scalar>(timeDir + "/meshPhi");
        scalar dPhi = 0;
        scalar phiScale = 0;
        const SurfaceScalarField& phi = mesh->meshPhi();
        for (std::size_t f = 0; f < phi.internal.size() && f < ofPhi.internalField.size(); ++f)
        {
            dPhi = std::fmax(dPhi, std::fabs(phi.internal[f] - ofPhi.internalField[f]));
            phiScale = std::fmax(phiScale, std::fabs(ofPhi.internalField[f]));
        }
        std::size_t nBoundaryCompared = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (const auto& b : ofPhi.boundary)
            {
                if (b.name != patches[pi].name) continue;
                if (b.valueUniform || b.values.size() != phi.boundary[pi].size()) continue;
                for (std::size_t i = 0; i < b.values.size(); ++i)
                {
                    dPhi = std::fmax(dPhi, std::fabs(phi.boundary[pi][i] - b.values[i]));
                    phiScale = std::fmax(phiScale, std::fabs(b.values[i]));
                    ++nBoundaryCompared;
                }
            }
        }
        largestPhi = std::fmax(largestPhi, phiScale);
        worstPhi = std::fmax(worstPhi, dPhi/std::fmax(phiScale, scalar(1e-300)));
        std::printf("  step %d  t = %.17g   points %.3e of the mesh's extent   meshPhi %.3e relative "
                    "(%zu internal, %zu boundary faces; |meshPhi| up to %.3e)\n",
                    (int)timeIndex, (double)t, (double)(dPoint/lengthScale),
                    (double)(dPhi/std::fmax(phiScale, scalar(1e-300))), phi.internal.size(),
                    nBoundaryCompared, (double)phiScale);
        check("...OpenFOAM's meshPhi has this mesh's internal faces",
              ofPhi.internalField.size() == phi.internal.size());

        if (fileExists(timeDir + "/V") && fileExists(timeDir + "/C"))
        {
            const FieldData<scalar> ofV = readField<scalar>(timeDir + "/V");
            const FieldData<vector> ofC = readField<vector>(timeDir + "/C");
            scalar dV = 0;
            scalar dC = 0;
            for (std::size_t c = 0; c < g.V().size() && c < ofV.internalField.size(); ++c)
            {
                dV = std::fmax(dV, std::fabs(g.V()[c] - ofV.internalField[c])/ofV.internalField[c]);
                dC = std::fmax(dC, mag(g.C()[c] - ofC.internalField[c])/lengthScale);
            }
            worstV = std::fmax(worstV, dV);
            worstC = std::fmax(worstC, dC);
            ++nV;
            std::printf("           V %.3e relative, C %.3e of the mesh's extent\n", (double)dV, (double)dC);
            check("...OpenFOAM's V has this mesh's cells", ofV.internalField.size() == g.V().size());
        }
    }

    // A RIGID MOTION KEEPS EVERY VOLUME, up to the round-off of recomputing it from rotated points.
    // That round-off is the reason V0 exists at all on such a mesh: V/V0 differs from 1 in the last
    // digits and MULES divides by it.
    scalar dRigid = 0;
    for (std::size_t c = 0; c < VStart.size(); ++c)
    {
        dRigid = std::fmax(dRigid, std::fabs(g.V()[c] - VStart[c])/VStart[c]);
    }
    std::printf("  the motion moved a point by up to %.3e (extent %.3e); |meshPhi| up to %.3e\n",
                (double)largestMove, (double)lengthScale, (double)largestPhi);
    std::printf("  worst over the run: points %.3e, meshPhi %.3e, V %.3e, C %.3e (V and C at %d steps)\n",
                (double)worstPoint, (double)worstPhi, (double)worstV, (double)worstC, nV);
    std::printf("  V at the end against V at the start: %.3e relative\n", (double)dRigid);

    check("the mesh actually moved", largestMove > scalar(1e-6)*lengthScale && largestPhi > scalar(0));
    // EXACTLY OpenFOAM's, not merely close: MEASURED 0.000e+00 on every profile, at writePrecision 18.
    // A round-off bound would let through every decision this port made that is about the ORDER of
    // operations rather than their meaning, and the script lists what each of them reads when it is
    // wrong: 2e-16 to 2e-14, all of them under any bound a round-off argument could defend.
    // BRAE_MESH_MOTION_ROUNDOFF=1 is for a compiler that contracts or orders floating point
    // differently from the one this was measured with (GCC 13 under nvcc 12, aarch64); it holds the
    // points to 1e-15 of the mesh's extent and meshPhi to 1e-12 instead, and says so.
    if (std::getenv("BRAE_MESH_MOTION_ROUNDOFF"))
    {
        std::printf("  (BRAE_MESH_MOTION_ROUNDOFF: holding points and meshPhi to round-off, not to zero)\n");
        check("the points are OpenFOAM's to round-off", worstPoint < scalar(1e-15));
        check("meshPhi is OpenFOAM's to round-off", worstPhi < scalar(1e-12));
    }
    else
    {
        check("the points are OpenFOAM's, to the bit", worstPoint == scalar(0));
        check("meshPhi is OpenFOAM's, to the bit", worstPhi == scalar(0));
    }
    check("V and C were compared at one step at least", nV > 0);
    // V AND C ARE EXACT TOO, since fv_geometry's face centres took primitiveMeshTools.C's operation
    // order ((1/3)*sumAc/sumA, and (1/3)*(a + b + c) on a triangle). Before that they read 1.4e-14 and
    // 5.7e-16 at worst before any motion and 1.7e-14 and 8.0e-16 after it -- and on sloshingTank2D the
    // 1e-16 in a cell centre chose the wrong cell for pRefPoint (0 0 0.15), which lies on a face.
    if (std::getenv("BRAE_MESH_MOTION_ROUNDOFF"))
    {
        check("V of the moved mesh is OpenFOAM's to round-off", worstV < scalar(1e-13));
        check("C of the moved mesh is OpenFOAM's to round-off", worstC < scalar(5e-15));
    }
    else
    {
        check("V of the moved mesh is OpenFOAM's, to the bit", worstV == scalar(0));
        check("C of the moved mesh is OpenFOAM's, to the bit", worstC == scalar(0));
        check("...and so were V and C before it moved", startV == scalar(0) && startC == scalar(0));
    }
    check("V0 is the volume the mesh had before each update, exactly", worstV0 == scalar(0));
    check("a rigid motion kept every volume to round-off", dRigid < scalar(1e-10));

    std::printf("test_mesh_motion_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
