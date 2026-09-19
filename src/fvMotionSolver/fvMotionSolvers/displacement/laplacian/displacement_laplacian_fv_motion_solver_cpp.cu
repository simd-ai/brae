#include "displacement_laplacian_fv_motion_solver_cpp.cuh"
#include "face_cpp.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "fvm.cuh"
#include "geometric_field.cuh"
#include "patch_wave_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include "solution_directions.cuh"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <regex>
#include <stdexcept>

namespace brae {

namespace {

const char* const WHO = "brae displacementLaplacian: ";

bool fileExists(const std::string& path)
{
    return std::filesystem::exists(path) || std::filesystem::exists(path + ".gz");
}

// the tokens of an fvSchemes entry, the specific key first and `default` after it
std::vector<std::string> schemeTokens(
    const FoamDict& fvSchemes,
    const std::string& section,
    const std::string& key)
{
    const FoamDict* d = fvSchemes.subDict(section);
    if (!d)
    {
        throw std::runtime_error(std::string(WHO) + "system/fvSchemes has no `" + section + "`.");
    }
    const std::vector<std::string>* v = d->find(key);
    if (!v)
    {
        v = d->find("default");
    }
    if (!v || v->empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "system/fvSchemes' " + section + " has neither `" + key + "` nor a default.");
    }
    return *v;
}

std::string joined(const std::vector<std::string>& tokens)
{
    std::string s;
    for (const std::string& t : tokens)
    {
        s += (s.empty() ? "" : " ") + t;
    }
    return s;
}

// a `uniform (x y z)` entry, which is the only form read here
vector uniformVector(
    const FoamDict& d,
    const std::string& key,
    const std::string& where)
{
    const std::vector<std::string>* v = d.find(key);
    if (!v || v->empty() || v->front() != "uniform")
    {
        throw std::runtime_error(
            std::string(WHO) + where + "'s `" + key + "` is not `uniform (x y z)`. Only a uniform value "
            "is read.");
    }
    const std::vector<scalar> c = d.scalarListOr(key, {});
    if (c.size() != 3)
    {
        throw std::runtime_error(std::string(WHO) + where + "'s `" + key + "` is not a vector.");
    }
    return vector{c[0], c[1], c[2]};
}

bool patchMatches(
    const FvPatch& p,
    const std::string& key)
{
    if (key == p.name) return true;
    for (const std::string& grp : p.inGroups)
    {
        if (key == grp) return true;
    }
    try
    {
        const std::regex re = compileFoamRegex(key);
        if (std::regex_match(p.name, re)) return true;
        for (const std::string& grp : p.inGroups)
        {
            if (std::regex_match(grp, re)) return true;
        }
    }
    catch (...)
    {
    }
    return false;
}

GamgControls readSolverEntry(
    const FoamDict& solvers,
    const std::string& name)
{
    const FoamDict* d = solvers.subDict(name);
    if (!d)
    {
        throw std::runtime_error(
            std::string(WHO) + "system/fvSolution has no solver entry for `" + name + "`. "
            "solution::solverDict reads it with a mandatory lookup.");
    }
    const std::string solver = d->wordOr("solver", "");
    if (solver != "GAMG")
    {
        throw std::runtime_error(
            std::string(WHO) + "system/fvSolution solves `" + name + "` with `solver " + solver + "`. Only "
            "GAMG is ported for the displacement equation.");
    }
    return readGamgControls(
        *d,
        d->scalarOr("tolerance", scalar(1e-6)),
        d->scalarOr("relTol", scalar(0)),
        static_cast<int>(d->scalarOr("maxIter", scalar(1000))),
        std::string(WHO) + "system/fvSolution's GAMG entry for " + name + " ");
}

} // namespace

std::unique_ptr<DisplacementLaplacianFvMotionSolver> DisplacementLaplacianFvMotionSolver::New(
    const FoamDict& coeffs,
    const std::string& caseDir,
    const std::string& startDir)
{
    std::unique_ptr<DisplacementLaplacianFvMotionSolver> s(new DisplacementLaplacianFvMotionSolver());
    s->caseDir_ = caseDir;

    // motionDiffusivity::New(mesh, coeffDict().lookup("diffusivity"))
    const std::vector<std::string> diffusivity = coeffs.wordListOr("diffusivity", {});
    if (diffusivity.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "displacementLaplacianCoeffs has no `diffusivity`. OpenFOAM reads it with "
            "a mandatory lookup and stops without it.");
    }
    if (diffusivity.front() != "inverseDistance")
    {
        throw std::runtime_error(
            std::string(WHO) + "displacementLaplacianCoeffs asks for `diffusivity " + joined(diffusivity) +
            "`. Only inverseDistance is ported; another diffusivity distributes the motion differently.");
    }
    s->diffusivityPatches_.assign(diffusivity.begin() + 1, diffusivity.end());
    if (s->diffusivityPatches_.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "`diffusivity inverseDistance` names no patches. OpenFOAM then takes a "
            "uniform distance of 1 (inverseDistanceDiffusivity.C, the empty patchSet), which is not "
            "ported.");
    }
    for (const char* key : {"interpolation", "frozenPointsZone"})
    {
        if (coeffs.found(key))
        {
            throw std::runtime_error(
                std::string(WHO) + "displacementLaplacianCoeffs sets `" + key + "`, which is not ported.");
        }
    }
    for (const char* key : {"cellZone", "cellSet"})
    {
        const std::string name = coeffs.wordOr(key, "");
        if (!name.empty() && name != "none")
        {
            throw std::runtime_error(
                std::string(WHO) + "displacementLaplacianCoeffs names `" + key + " " + name + "`, which "
                "is not ported.");
        }
    }
    for (const char* field : {"pointLocation", "cellDisplacement"})
    {
        if (fileExists(startDir + "/" + field))
        {
            throw std::runtime_error(
                std::string(WHO) + startDir + "/" + field + " exists. OpenFOAM reads it (" +
                (std::string(field) == "pointLocation"
                    ? "and applies its boundary conditions to the new point locations"
                    : "as the displacement to start from") + "), which is not ported.");
        }
    }
    const std::string pdPath = startDir + "/pointDisplacement";
    if (!std::filesystem::exists(pdPath))
    {
        throw std::runtime_error(
            std::string(WHO) + pdPath + " does not exist. displacementMotionSolver reads it with "
            "MUST_READ.");
    }
    s->pointDisplacementDict_ = readDict(pdPath);

    // the run's start time, the waveMaker's default `startTime`
    {
        const std::string base = std::filesystem::path(startDir).filename().string();
        char* end = nullptr;
        const double t = std::strtod(base.c_str(), &end);
        if (base.empty() || end == base.c_str() || *end != '\0')
        {
            throw std::runtime_error(std::string(WHO) + "the start directory `" + startDir + "` is not a time.");
        }
        s->startTime_ = t;
    }

    // constant/g, which the waveMaker reads through meshObjects::gravity
    const std::string gPath = caseDir + "/constant/g";
    if (std::filesystem::exists(gPath))
    {
        const FoamDict gd = readDict(gPath);
        const std::vector<scalar> gv = gd.scalarListOr("value", {});
        if (gv.size() == 3)
        {
            s->g_ = vector{gv[0], gv[1], gv[2]};
        }
    }

    // the equation's solver: fvMatrix::solverDict(), the Final entry under PIMPLE's last iteration
    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* solvers = fvSolution.subDict("solvers");
    if (!solvers)
    {
        throw std::runtime_error(std::string(WHO) + "system/fvSolution has no `solvers`.");
    }
    s->controls_ = readSolverEntry(*solvers, "cellDisplacement");
    s->controlsFinal_ = solvers->subDict("cellDisplacementFinal")
        ? readSolverEntry(*solvers, "cellDisplacementFinal")
        : GamgControls();
    if (!solvers->subDict("cellDisplacementFinal"))
    {
        s->controlsFinal_.smoother.clear();
    }

    // the schemes on the equation's path
    const FoamDict fvSchemes = readDict(caseDir + "/system/fvSchemes");
    const std::vector<std::string> lap =
        schemeTokens(fvSchemes, "laplacianSchemes", "laplacian(diffusivity,cellDisplacement)");
    if (joined(lap) != "Gauss linear corrected")
    {
        throw std::runtime_error(
            std::string(WHO) + "laplacian(diffusivity,cellDisplacement) is `" + joined(lap) + "`. Only "
            "`Gauss linear corrected` is ported for the displacement equation.");
    }
    const std::vector<std::string> grad = schemeTokens(fvSchemes, "gradSchemes", "grad(cellDisplacement)");
    if (joined(grad) != "Gauss linear")
    {
        throw std::runtime_error(
            std::string(WHO) + "grad(cellDisplacement), which the corrected laplacian's non-orthogonal "
            "correction takes, is `" + joined(grad) + "`. Only `Gauss linear` is ported there.");
    }
    // wallDist's y is named "y" & "patch" -- word::operator& capitalises -- and interpolated by name
    const std::vector<std::string> interp = schemeTokens(fvSchemes, "interpolationSchemes", "interpolate(yPatch)");
    if (joined(interp) != "linear")
    {
        throw std::runtime_error(
            std::string(WHO) + "interpolate(yPatch), which the inverseDistance diffusivity takes, is `" +
            joined(interp) + "`. Only `linear` is ported there.");
    }
    // ...and the wall distance itself: inverseDistanceDiffusivity::correct asks for
    // wallDist::New(mesh, meshWave, patchSet), whose patch type name is "patch", so it reads fvSchemes'
    // `patchDist` sub-dictionary (wallDist.C:98-101, subOrEmptyDict) -- meshWave unless it names another
    // method (patchDistMethod.C:64-76), meshWave's correctWalls (default true), and updateInterval
    // (default 1: recomputed at every motion). diffusivityCorrect runs meshWave with correctWalls, every
    // step; anything else the dictionary asks for is refused.
    if (const FoamDict* pd = fvSchemes.subDict("patchDist"))
    {
        const std::string method = pd->wordOr("method", "meshWave");
        if (method != "meshWave")
        {
            throw std::runtime_error(
                std::string(WHO) + "fvSchemes names `patchDist { method " + method + "; }`, the wall distance "
                "the inverseDistance diffusivity takes. Only meshWave is ported.");
        }
        const std::string cw = pd->wordOr("correctWalls", "true");
        if (cw == "false" || cw == "no" || cw == "off")
        {
            throw std::runtime_error(
                std::string(WHO) + "fvSchemes sets `patchDist { correctWalls " + cw + "; }`; the ported "
                "meshWave always corrects the cells beside the patches.");
        }
        const scalar interval = pd->scalarOr("updateInterval", scalar(1));
        if (interval != scalar(1))
        {
            throw std::runtime_error(
                std::string(WHO) + "fvSchemes sets `patchDist { updateInterval " + pd->wordOr("updateInterval", "")
                + "; }`; only 1, the default, is ported (the distance is recomputed at every motion).");
        }
    }
    return s;
}

void DisplacementLaplacianFvMotionSolver::attach(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    points0_ = m.points();

    const FoamDict* bf = pointDisplacementDict_.subDict("boundaryField");
    if (!bf)
    {
        throw std::runtime_error(std::string(WHO) + "pointDisplacement has no boundaryField.");
    }
    const vector internalValue = uniformVector(pointDisplacementDict_, "internalField", "pointDisplacement");
    pointDisplacement_.assign(static_cast<std::size_t>(m.nPoints()), internalValue);

    pointPatches_.clear();
    pointPatches_.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        PointPatch& pp = pointPatches_[pi];
        pp.name = p.name;
        if (isCoupledInterfaceType(p.type))
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + p.name + "` is " + p.type + ". A coupled patch couples the "
                "point field and the interpolation across it, which is not ported.");
        }
        const FoamDict* d = bf->subDict(p.name);
        if (!d)
        {
            for (const std::string& grp : p.inGroups)
            {
                d = bf->subDict(grp);
                if (d) break;
            }
        }
        std::string type;
        if (d)
        {
            type = d->wordOr("type", "");
        }
        else if (p.type == "empty")
        {
            type = "empty";
        }
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "pointDisplacement has no boundaryField entry for patch `" + p.name + "`.");
        }
        pp.dict = d;
        pp.meshPoints = primitivePatch(m, faceRange(p.start, p.size)).meshPoints;
        std::vector<vector> localPoints(pp.meshPoints.size());
        for (std::size_t i = 0; i < pp.meshPoints.size(); ++i)
        {
            localPoints[i] = m.points()[static_cast<std::size_t>(pp.meshPoints[i])];
        }
        if (type == "fixedValue")
        {
            pp.type = PointPatchType::fixedValue;
            pp.value.assign(pp.meshPoints.size(), uniformVector(*d, "value", "pointDisplacement patch " + p.name));
        }
        else if (type == "zeroGradient")
        {
            pp.type = PointPatchType::zeroGradient;
        }
        else if (type == "empty")
        {
            if (p.type != "empty")
            {
                throw std::runtime_error(
                    std::string(WHO) + "pointDisplacement's patch `" + p.name + "` is `empty` on a " +
                    p.type + " patch.");
            }
            pp.type = PointPatchType::empty;
        }
        else if (type == "waveMaker")
        {
            pp.type = PointPatchType::waveMaker;
            pp.waveMaker.reset(new WaveMakerPointPatchVectorField(*d, p.name, localPoints, g_, startTime_));
            pp.value.assign(pp.meshPoints.size(), uniformVector(*d, "value", "pointDisplacement patch " + p.name));
        }
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "pointDisplacement's patch `" + p.name + "` is `" + type + "`. Only "
                "fixedValue, zeroGradient, empty and waveMaker are ported; a constraint type (slip, "
                "symmetry, ...) also constrains the corner points (pointConstraints), which is not.");
        }
        if (p.type == "empty" && pp.type != PointPatchType::empty)
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + p.name + "` is empty and its point condition is not.");
        }
    }

    // the cell displacement: zero, and zero on every patch (GeometricField's `boundaryField_ == value`)
    cellDisplacement_.assign(static_cast<std::size_t>(m.nCells()), vector{0, 0, 0});
    cellDisplacementBoundary_.assign(patches.size(), std::vector<vector>());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (pointPatches_[pi].type == PointPatchType::empty) continue;
        cellDisplacementBoundary_[pi].assign(static_cast<std::size_t>(patches[pi].size), vector{0, 0, 0});
    }

    // polyBoundaryMesh::patchSet(patchNames): names, groups and patterns
    diffusivityPatchIDs_.clear();
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (const std::string& key : diffusivityPatches_)
        {
            if (patchMatches(patches[pi], key))
            {
                diffusivityPatchIDs_.push_back(static_cast<label>(pi));
                break;
            }
        }
    }
    if (diffusivityPatchIDs_.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "`diffusivity inverseDistance` names no patch of this mesh.");
    }

    twoDCorrector_.build(m, g, patches);
    // the pointFaces branch of calcPointCells: the wall distance asks for pointFaces first
    pointCells_ = pointCellsFromPointFaces(m, meshPointFaces(m));
    attached_ = true;
}

void DisplacementLaplacianFvMotionSolver::diffusivityCorrect(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    // wallDist::New(mesh, meshWave, patchSet).y(), which the MeshObject keeps current as the mesh moves
    const PatchWave wave = patchWave(m, g, patches, diffusivityPatchIDs_, true);
    if (wave.nUnset > 0)
    {
        throw std::runtime_error(
            std::string(WHO) + "the wall distance did not reach " + std::to_string(wave.nUnset) + " cells "
            "and faces. OpenFOAM carries on with -GREAT there; brae does not.");
    }
    y_ = wave.distance;

    // meshWave::correct: the wave's patch distances on every non-empty patch, then
    // correctBoundaryConditions -- fixedValue on the named patches keeps them, zeroGradient elsewhere
    // takes the cell's
    faceDiffusivity_ = fvc::interpolate(y_, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        std::vector<scalar>& b = faceDiffusivity_.boundary[pi];
        if (p.type == "empty")
        {
            std::fill(b.begin(), b.end(), scalar(0));
            continue;
        }
        const bool fixed =
            std::find(diffusivityPatchIDs_.begin(), diffusivityPatchIDs_.end(), static_cast<label>(pi))
         != diffusivityPatchIDs_.end();
        for (label i = 0; i < p.size; ++i)
        {
            b[static_cast<std::size_t>(i)] = fixed
                ? wave.patchDistance[pi][static_cast<std::size_t>(i)]
                : y_[static_cast<std::size_t>(p.faceCells[static_cast<std::size_t>(i)])];
        }
    }
    // faceDiffusivity_ = one/fvc::interpolate(y)
    for (scalar& v : faceDiffusivity_.internal)
    {
        v = 1.0/v;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "empty") continue;
        for (scalar& v : faceDiffusivity_.boundary[pi])
        {
            v = 1.0/v;
        }
    }
}

void DisplacementLaplacianFvMotionSolver::solve(
    scalar time,
    bool finalIteration,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    GamgAgglomerationCache& agglomeration)
{
    diffusivityCorrect(m, g, patches);

    // pointDisplacement_.boundaryFieldRef().updateCoeffs(): valuePointPatchField::updateCoeffs writes
    // each fixing patch's values into the point field, patch by patch
    for (PointPatch& pp : pointPatches_)
    {
        if (pp.type == PointPatchType::waveMaker)
        {
            std::vector<vector> localPoints(pp.meshPoints.size());
            for (std::size_t i = 0; i < pp.meshPoints.size(); ++i)
            {
                localPoints[i] = m.points()[static_cast<std::size_t>(pp.meshPoints[i])];
            }
            pp.value = pp.waveMaker->updateCoeffs(time, localPoints);
        }
        if (pp.fixesValue())
        {
            for (std::size_t i = 0; i < pp.meshPoints.size(); ++i)
            {
                pointDisplacement_[static_cast<std::size_t>(pp.meshPoints[i])] = pp.value[i];
            }
        }
    }

    // cellDisplacement_.boundaryFieldRef().updateCoeffs(): cellMotion is the face average of the
    // POINT field -- not of the patch's own values -- over the mesh's current points
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!pointPatches_[pi].fixesValue()) continue;
        const FvPatch& p = patches[pi];
        for (label i = 0; i < p.size; ++i)
        {
            cellDisplacementBoundary_[pi][static_cast<std::size_t>(i)] =
                faceAverage(m, p.start + i, m.points(), pointDisplacement_);
        }
    }

    // the field the matrix is assembled on: cellMotion is fixedValue, the rest what the point
    // condition was
    GeometricField<vector> D;
    D.internal = cellDisplacement_;
    D.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        switch (pointPatches_[pi].type)
        {
            case PointPatchType::fixedValue:
            case PointPatchType::waveMaker:
            {
                D.boundary[pi].reset(new FixedValuePatchField<vector>(
                    p,
                    false,
                    vector{0, 0, 0},
                    cellDisplacementBoundary_[pi]));
                D.boundary[pi]->evaluate(D.internal);
                break;
            }
            case PointPatchType::zeroGradient:
            {
                D.boundary[pi].reset(new ZeroGradientPatchField<vector>(p));
                D.boundary[pi]->setStoredValues(cellDisplacementBoundary_[pi]);
                break;
            }
            case PointPatchType::empty:
            {
                D.boundary[pi].reset(new EmptyPatchField<vector>(p));
                break;
            }
        }
    }

    // fvm::laplacian(1*diffusivity, cellDisplacement), Gauss linear corrected
    FvMatrix<vector> M = fvm::laplacian<vector>(faceDiffusivity_, D, m, g, patches, true);
    {
        std::vector<std::vector<vector>> bnd(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            bnd[pi] = (pointPatches_[pi].type == PointPatchType::empty)
                ? std::vector<vector>(static_cast<std::size_t>(patches[pi].size), vector{0, 0, 0})
                : cellDisplacementBoundary_[pi];
        }
        const std::vector<tensor> gradD = fvc::gaussGrad(D.internal, bnd, m, g, patches);
        const std::vector<vector> corr = fvm::laplacianNonOrthSource<vector, tensor>(
            faceDiffusivity_, D, gradD, m, g, patches);
        for (std::size_t c = 0; c < corr.size(); ++c)
        {
            M.source[c] -= corr[c];
        }
    }

    // fvMatrix<vector>::solveSegregated: one GAMG solve per solved component
    const GamgControls& controls = finalIteration ? controlsFinal_ : controls_;
    if (controls.smoother.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "the last PIMPLE iteration solves with the `cellDisplacementFinal` entry, "
            "and system/fvSolution has none.");
    }
    const GamgAgglomeration& a = agglomeration.get(m, g, controls.nCellsInCoarsestLevel);
    const SolutionDirections sd = solutionDirections(patches);
    lastSolve_ = DisplacementSolveRecord();
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        if (!sd.valid(cmpt)) continue;
        auto component = [cmpt](const vector& v)
        {
            return cmpt == 0 ? v.x : (cmpt == 1 ? v.y : v.z);
        };
        FvScalarMatrix Mc;
        Mc.diag = M.diag;
        Mc.upper = M.upper;
        Mc.lower = M.lower;
        Mc.source.resize(M.source.size());
        for (std::size_t c = 0; c < M.source.size(); ++c)
        {
            Mc.source[c] = component(M.source[c]);
        }
        Mc.internalCoeffs.resize(patches.size());
        Mc.boundaryCoeffs.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (const vector& v : M.internalCoeffs[pi])
            {
                Mc.internalCoeffs[pi].push_back(component(v));
            }
            for (const vector& v : M.boundaryCoeffs[pi])
            {
                Mc.boundaryCoeffs[pi].push_back(component(v));
            }
        }
        std::vector<scalar> psi(cellDisplacement_.size());
        for (std::size_t c = 0; c < psi.size(); ++c)
        {
            psi[c] = component(cellDisplacement_[c]);
        }
        lastSolve_.perf[cmpt] = gamgSolve(Mc, psi, m, patches, a, controls, nullptr);
        lastSolve_.solved[cmpt] = true;
        for (std::size_t c = 0; c < psi.size(); ++c)
        {
            if (cmpt == 0)
            {
                cellDisplacement_[c].x = psi[c];
            }
            else if (cmpt == 1)
            {
                cellDisplacement_[c].y = psi[c];
            }
            else
            {
                cellDisplacement_[c].z = psi[c];
            }
        }
    }

    // psi.correctBoundaryConditions(): zeroGradient takes the new cell values, cellMotion keeps its own
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (pointPatches_[pi].type != PointPatchType::zeroGradient) continue;
        const FvPatch& p = patches[pi];
        for (label i = 0; i < p.size; ++i)
        {
            cellDisplacementBoundary_[pi][static_cast<std::size_t>(i)] =
                cellDisplacement_[static_cast<std::size_t>(p.faceCells[static_cast<std::size_t>(i)])];
        }
    }
}

std::vector<vector> DisplacementLaplacianFvMotionSolver::curPoints(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    // volPointInterpolation::New(mesh).interpolate(cellDisplacement, pointDisplacement), the weights
    // remade on the mesh as it stands (volPointInterpolation::movePoints)
    interpolation_.makeWeights(m, g, patches, pointCells_);
    interpolation_.interpolateInternalField(cellDisplacement_, pointDisplacement_);
    interpolation_.interpolateBoundaryField(cellDisplacementBoundary_, pointDisplacement_);
    // pointConstraints::constrain(pf, false): correctBoundaryConditions, patch by patch -- the fixing
    // patches write their values back over the interpolated ones; no constraint patch, so no corners
    for (const PointPatch& pp : pointPatches_)
    {
        if (!pp.fixesValue()) continue;
        for (std::size_t i = 0; i < pp.meshPoints.size(); ++i)
        {
            pointDisplacement_[static_cast<std::size_t>(pp.meshPoints[i])] = pp.value[i];
        }
    }

    std::vector<vector> curPoints(points0_.size());
    for (std::size_t i = 0; i < points0_.size(); ++i)
    {
        curPoints[i] = points0_[i] + pointDisplacement_[i];
    }
    twoDCorrector_.correctPoints(m.points(), curPoints);
    return curPoints;
}

std::vector<vector> DisplacementLaplacianFvMotionSolver::newPoints(
    scalar time,
    bool finalIteration,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    GamgAgglomerationCache& agglomeration)
{
    if (!attached_)
    {
        throw std::runtime_error(std::string(WHO) + "newPoints() before attach().");
    }
    solve(time, finalIteration, m, g, patches, agglomeration);
    return curPoints(m, g, patches);
}

} // namespace brae
