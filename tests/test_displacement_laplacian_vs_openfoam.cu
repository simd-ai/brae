// brae's displacementLaplacian mesh motion against REAL OpenFOAM's, with no flow solver in the way.
//
// THE ORACLE is OpenFOAM's moveDynamicMesh, which runs mesh.update() in a time loop and writes, per time
// step, the moved polyMesh/points, the mesh-motion flux meshPhi and the motion solver's own two fields,
// cellDisplacement and pointDisplacement; its log carries every GAMG solve of the displacement equation,
// one line per solved component. postProcess's writeCellVolumes and writeCellCentres on the last time
// give V and C of the moved mesh.
//
// tests/displacement_laplacian_vs_openfoam.sh says what each profile is for and what it measured.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "dynamic_motion_solver_fv_mesh_cpp.cuh"
#include "gamg_solver_cpp.cuh"
#include <cmath>
#include <exception>
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

// the internalField of a vector field file, uniform or not, sized n
std::vector<vector> readVectorInternal(
    const std::string& path,
    std::size_t n)
{
    const FieldData<vector> fd = readField<vector>(path);
    if (fd.internalUniform)
    {
        return std::vector<vector>(n, fd.internalUniformValue);
    }
    return fd.internalField;
}

bool fileExists(const std::string& path)
{
    return std::ifstream(path).good();
}

// one GAMG line of OpenFOAM's log
struct LoggedSolve
{
    char component = 0;
    double initialResidual = 0;
    double finalResidual = 0;
    int nIterations = 0;
};

std::vector<LoggedSolve> readGamgLog(const std::string& path)
{
    std::ifstream in(path);
    std::string line;
    std::vector<LoggedSolve> out;
    static const std::regex re(
        R"(GAMG:\s+Solving for cellDisplacement([xyz]), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations ([0-9]+))");
    while (std::getline(in, line))
    {
        std::smatch mm;
        if (!std::regex_search(line, mm, re)) continue;
        LoggedSolve s;
        s.component = mm[1].str()[0];
        s.initialResidual = std::strtod(mm[2].str().c_str(), nullptr);
        s.finalResidual = std::strtod(mm[3].str().c_str(), nullptr);
        s.nIterations = std::atoi(mm[4].str().c_str());
        out.push_back(s);
    }
    return out;
}

scalar maxDiff(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    scalar d = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d = std::fmax(d, mag(a[i] - b[i]));
    }
    return d;
}

scalar maxMag(const std::vector<vector>& a)
{
    scalar d = 0;
    for (const vector& v : a)
    {
        d = std::fmax(d, mag(v));
    }
    return d;
}
}   // namespace

int run(
    int argc,
    char** argv)
{
    std::printf("== brae displacementLaplacian mesh motion vs OpenFOAM moveDynamicMesh ==\n");
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

    std::unique_ptr<DynamicMotionSolverFvMesh> mesh = DynamicMotionSolverFvMesh::New(caseDir, caseDir + "/0");
    check("the case asks for a mesh motion, and brae built one", mesh != nullptr);
    if (!mesh) return 1;
    check("...a displacementLaplacian one", mesh->displacementSolver() != nullptr);
    if (!mesh->displacementSolver()) return 1;
    mesh->attach(m, g, patches);
    const DisplacementLaplacianFvMotionSolver& solver = *mesh->displacementSolver();

    const std::vector<LoggedSolve> logged = readGamgLog(caseDir + "/log.moveDynamicMesh");
    std::size_t nextLogged = 0;

    // the scale every absolute point difference is read against
    scalar lengthScale = 0;
    for (const vector& p : m.points())
    {
        lengthScale = std::fmax(lengthScale, mag(p));
    }

    GamgAgglomerationCache agglomeration;
    // absolute, over the run; the displacement fields are read against the run's largest displacement
    // at the end, the points against the mesh's extent
    scalar worstPoint = 0;
    scalar worstPointDisp = 0;
    scalar worstCellDisp = 0;
    scalar worstPhi = 0;
    scalar largestMove = 0;
    scalar largestPhi = 0;
    int nIterationMismatches = 0;
    int nSolvesCompared = 0;
    int nSolvesBothConverged = 0;
    // Time::operator++: the value ACCUMULATES
    scalar t = 0;
    for (int k = 4; k < argc; ++k)
    {
        const std::string timeDir = argv[k];
        const label timeIndex = static_cast<label>(k - 3);
        t = t + deltaT;
        mesh->update(t, deltaT, timeIndex, false, &agglomeration);

        // the GAMG solves of this step, against the log's, component by component
        const DisplacementSolveRecord& rec = solver.lastSolve();
        for (int cmpt = 0; cmpt < 3; ++cmpt)
        {
            if (!rec.solved[cmpt]) continue;
            const char name = "xyz"[cmpt];
            if (nextLogged >= logged.size() || logged[nextLogged].component != name)
            {
                std::printf("  FAIL: OpenFOAM's log has no solve of cellDisplacement%c at step %d\n", name,
                            (int)timeIndex);
                ++failures;
                continue;
            }
            const LoggedSolve& of = logged[nextLogged++];
            const SolverPerformance& bp = rec.perf[cmpt];
            ++nSolvesCompared;
            if (of.nIterations != bp.nIterations)
            {
                ++nIterationMismatches;
            }
            if (of.finalResidual < 1e-5 && bp.finalResidual < 1e-5)
            {
                ++nSolvesBothConverged;
            }
            std::printf("  step %d  cellDisplacement%c: initial %.6e / %.6e, final %.3e / %.3e, "
                        "iterations %d / %d  (brae / OpenFOAM)\n",
                        (int)timeIndex, name, (double)bp.initialResidual, of.initialResidual,
                        (double)bp.finalResidual, of.finalResidual, (int)bp.nIterations, of.nIterations);
        }

        const std::vector<vector> ofPoints = readPointsFile(timeDir + "/polyMesh/points");
        check("OpenFOAM wrote as many points as the mesh has", ofPoints.size() == m.points().size());
        const std::vector<vector>& pd = solver.pointDisplacement();
        const scalar motion = maxMag(pd);
        largestMove = std::fmax(largestMove, motion);
        const scalar dPoint = maxDiff(m.points(), ofPoints);
        worstPoint = std::fmax(worstPoint, dPoint);

        const std::vector<vector> ofPd = readVectorInternal(timeDir + "/pointDisplacement", pd.size());
        const scalar dPd = maxDiff(pd, ofPd);
        worstPointDisp = std::fmax(worstPointDisp, dPd);

        const std::vector<vector>& cd = solver.cellDisplacement();
        const std::vector<vector> ofCd = readVectorInternal(timeDir + "/cellDisplacement", cd.size());
        const scalar dCd = maxDiff(cd, ofCd);
        worstCellDisp = std::fmax(worstCellDisp, dCd);

        const FieldData<scalar> ofPhi = readField<scalar>(timeDir + "/meshPhi");
        scalar dPhi = 0;
        scalar phiScale = 0;
        const SurfaceScalarField& phi = mesh->meshPhi();
        for (std::size_t f = 0; f < phi.internal.size() && f < ofPhi.internalField.size(); ++f)
        {
            dPhi = std::fmax(dPhi, std::fabs(phi.internal[f] - ofPhi.internalField[f]));
            phiScale = std::fmax(phiScale, std::fabs(ofPhi.internalField[f]));
        }
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
                }
            }
        }
        largestPhi = std::fmax(largestPhi, phiScale);
        // absolute, read against the run's largest |meshPhi| at the end: a step's own largest goes to
        // zero where the paddle turns, and a ratio to it measures the turn rather than the code
        worstPhi = std::fmax(worstPhi, dPhi);
        std::printf("  step %d  t = %.17g  |pointDisplacement| up to %.3e; brae - OpenFOAM: points %.3e, "
                    "pointDisplacement %.3e, cellDisplacement %.3e; meshPhi %.3e of the step's largest\n",
                    (int)timeIndex, (double)t, (double)motion, (double)dPoint, (double)dPd, (double)dCd,
                    (double)(dPhi/std::fmax(phiScale, scalar(1e-300))));
        check("...OpenFOAM's meshPhi has this mesh's internal faces",
              ofPhi.internalField.size() == phi.internal.size());
    }

    // V and C at the last step, from OpenFOAM's postProcess
    const std::string last = argv[argc - 1];
    scalar dV = -1;
    scalar dC = -1;
    if (fileExists(last + "/V") && fileExists(last + "/C"))
    {
        const FieldData<scalar> ofV = readField<scalar>(last + "/V");
        const FieldData<vector> ofC = readField<vector>(last + "/C");
        dV = 0;
        dC = 0;
        for (std::size_t c = 0; c < g.V().size() && c < ofV.internalField.size(); ++c)
        {
            dV = std::fmax(dV, std::fabs(g.V()[c] - ofV.internalField[c])/ofV.internalField[c]);
            dC = std::fmax(dC, mag(g.C()[c] - ofC.internalField[c]));
        }
        dC = dC/lengthScale;
    }
    worstPoint = worstPoint/lengthScale;
    worstPhi = worstPhi/std::fmax(largestPhi, scalar(1e-300));
    worstPointDisp = worstPointDisp/std::fmax(largestMove, scalar(1e-300));
    worstCellDisp = worstCellDisp/std::fmax(largestMove, scalar(1e-300));

    std::printf("  the motion moved a point by up to %.3e (extent %.3e); |meshPhi| up to %.3e\n",
                (double)largestMove, (double)lengthScale, (double)largestPhi);
    std::printf("  worst over the run: points %.3e of the extent; pointDisplacement %.3e and "
                "cellDisplacement %.3e of the largest displacement; meshPhi %.3e of the largest |meshPhi|; at the end "
                "V %.3e relative, C %.3e of the extent\n",
                (double)worstPoint, (double)worstPointDisp, (double)worstCellDisp, (double)worstPhi,
                (double)dV, (double)dC);
    std::printf("  GAMG: %d solves compared, %d with a different iteration count, %d converged in both\n",
                nSolvesCompared, nIterationMismatches, nSolvesBothConverged);

    check("the mesh actually moved", largestMove > scalar(0) && largestPhi > scalar(0));
    check("every GAMG solve of OpenFOAM's log was met, in order", nextLogged == logged.size() && nSolvesCompared > 0);
    check("every GAMG solve took OpenFOAM's iteration count", nIterationMismatches == 0);
    check("V and C were compared at the last step", dV >= 0 && dC >= 0);

    // THE BOUNDS ARE WHAT TWO SHARED OPERATORS COST, MEASURED, and not a tolerance. brae's fvm::laplacian
    // forms its face coefficient as (deltaCoeffs*gamma)*magSf where gaussLaplacianScheme forms
    // gammaMagSf = gamma*magSf first, and its linear interpolation is w*P + (1 - w)*N where
    // surfaceInterpolationScheme::dotInterpolate is lambda*(P - N) + N. With both in OpenFOAM's order
    // every profile is EXACT -- points, both displacement fields and meshPhi 0.000e+00 -- for its first
    // four to seven steps, and a few ulps apart after that. With them as they are, the worst of seven
    // profiles: points 2.9e-16 of the extent, the displacement fields 1.3e-14 of the largest
    // displacement, meshPhi 2.4e-13 of the largest |meshPhi|, V 1.1e-13 relative, C 5.9e-16 of the extent.
    const scalar pointBound = 5e-16;
    const scalar displacementBound = 2e-14;
    const scalar meshPhiBound = 5e-13;
    const scalar VBound = 2e-13;
    const scalar CBound = 1e-15;
    check("the points are OpenFOAM's, to 5e-16 of the extent", worstPoint <= pointBound);
    check("pointDisplacement is OpenFOAM's, to 2e-14 of the largest displacement", worstPointDisp <= displacementBound);
    check("cellDisplacement is OpenFOAM's, to 2e-14 of the largest displacement", worstCellDisp <= displacementBound);
    check("meshPhi is OpenFOAM's, to 5e-13 of the largest |meshPhi|", worstPhi <= meshPhiBound);
    check("V of the moved mesh is OpenFOAM's, to 2e-13 relative", dV <= VBound);
    check("C of the moved mesh is OpenFOAM's, to 1e-15 of the extent", dC <= CBound);

    std::printf("test_displacement_laplacian_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}

int main(
    int argc,
    char** argv)
{
    // A refusal is an answer, and the script's refusal arms read it: what brae will not run, by name
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::printf("  REFUSED: %s\n", e.what());
        return 3;
    }
}
