#include "inter_phase_time.cuh"
#include "displacement_laplacian_fv_motion_solver_cpp.cuh"
#include "face_cpp.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "fvm.cuh"
#include "geometric_field.cuh"
#include "patch_set.cuh"
#include "patch_wave_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include "solution_directions.cuh"
#include <cstring>
#include <cstdio>
#include <optional>
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
        // a Switch (meshWavePatchDistMethod.C:59): read by hand, `0` and `none` ran corrected
        if (!pd->switchOr("correctWalls", true))
        {
            throw std::runtime_error(
                std::string(WHO) + "fvSchemes sets `patchDist { correctWalls " + pd->wordOr("correctWalls", "")
                + "; }`; the ported meshWave always corrects the cells beside the patches.");
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
        // pointPatchField::New (pointPatchFieldNew.C:144-165, the dictionary constructor): on a
        // constraint patch the patch's own constraint type replaces the dictionary's unless `patchType`
        // names the patch's type, so a zeroGradient written on a symmetryPlane runs as symmetryPlane (a
        // slip, plus pointConstraints at its corners) -- neither of which is ported. Running the
        // dictionary's type would move the mesh differently from OpenFOAM and write the wrong type.
        if (p.type == "symmetryPlane" || p.type == "symmetry" || p.type == "wedge"
            || p.type == "cyclic" || p.type == "cyclicSlip" || p.type == "cyclicAMI"
            || p.type == "cyclicACMI" || p.type == "nonuniformTransformCyclic")
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + p.name + "` is " + p.type + ", a constraint patch whose "
                "point condition OpenFOAM replaces with its own " + p.type + " type; that point "
                "constraint is not ported.");
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

    // polyBoundaryMesh::patchSet(patchNames): names, groups and patterns -- the same selection interFoam
    // makes for kOmegaSST, which shares this wallDist (InterTurbulence::wallDistPatchIDs)
    diffusivityPatchIDs_ = patchSet(patches, diffusivityPatches_);
    if (diffusivityPatchIDs_.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "`diffusivity inverseDistance` names no patch of this mesh.");
    }

    twoDCorrector_.build(m, g, patches);
    // the pointFaces branch of calcPointCells: the wall distance asks for pointFaces first
    meshPointFaces_ = meshPointFaces(m);
    pointCells_ = pointCellsFromPointFaces(m, meshPointFaces_);
    // ...and what the two per-step passes keep from the addressing: the wave's cell-to-face lists and the
    // interpolation's own (vol_point_interpolation_cpp.cuh)
    meshCells_ = meshCells(m);
    interpolation_.clearAddressing();
    attached_ = true;
}

void DisplacementLaplacianFvMotionSolver::diffusivityCorrect(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    // wallDist::New(mesh, meshWave, patchSet).y(), which the MeshObject keeps current as the mesh moves
    // BRAE_CONTROL_MOTION_CELLS_REBUILT=1 has the wave build its cell-to-face lists itself, as before
    static const bool cellsRebuilt = std::getenv("BRAE_CONTROL_MOTION_CELLS_REBUILT") != nullptr;
    const PatchWave wave = patchWave(
        m,
        g,
        patches,
        diffusivityPatchIDs_,
        true,
        cellsRebuilt ? nullptr : &meshCells_,
        waveRunner_ ? &waveRunner_ : nullptr,
        cellsRebuilt ? nullptr : &meshPointFaces_);
    if (wave.nUnset > 0)
    {
        throw std::runtime_error(
            std::string(WHO) + "the wall distance did not reach " + std::to_string(wave.nUnset) + " cells "
            "and faces. OpenFOAM carries on with -GREAT there; brae does not.");
    }
    y_ = wave.distance;
    interPhase::Nested timedFace("wave: the face diffusivity from the distance (host interpolate, 1/y)");

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

FvMatrix<vector> DisplacementLaplacianFvMotionSolver::assembleOnHost(
    const GeometricField<vector>& D,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches) const
{
    FvMatrix<vector> M;
    {
        interPhase::Nested timedMatrix("motion: assembly, the laplacian matrix");
        M = fvm::laplacian<vector>(faceDiffusivity_, D, m, g, patches, true);
    }
    interPhase::Nested timedCorr("motion: assembly, the gradient and the non-orthogonal source");
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
    return M;
}

void DisplacementLaplacianFvMotionSolver::sameAssembly(
    const FvMatrix<vector>& got,
    const FvMatrix<vector>& host) const
{
    // bytes, not values: -0 against +0 and one rounding's difference are both differences
    auto same = [](
        const char* what,
        const scalar* a,
        std::size_t na,
        const scalar* b,
        std::size_t nb)
    {
        if (na != nb)
        {
            throw std::runtime_error(
                std::string(WHO) + "the assembly run elsewhere has " + std::to_string(na) + " entries of " + what
                + ", the host's " + std::to_string(nb) + ".");
        }
        if (na == 0 || std::memcmp(a, b, na*sizeof(scalar)) == 0) return;
        std::size_t at = 0;
        while (std::memcmp(a + at, b + at, sizeof(scalar)) == 0)
        {
            ++at;
        }
        char line[160];
        std::snprintf(line, sizeof(line), " entry %zu is %.17g, the host's %.17g.", at, a[at], b[at]);
        throw std::runtime_error(
            std::string(WHO) + "the assembly run elsewhere is not the host's: " + what + line);
    };
    same("upper", got.upper.data(), got.upper.size(), host.upper.data(), host.upper.size());
    same("lower", got.lower.data(), got.lower.size(), host.lower.data(), host.lower.size());
    same("diag", got.diag.data(), got.diag.size(), host.diag.data(), host.diag.size());
    same("source (3 a cell)", &got.source.data()->x, 3*got.source.size(), &host.source.data()->x,
         3*host.source.size());
    if (got.internalCoeffs.size() != host.internalCoeffs.size()
     || got.boundaryCoeffs.size() != host.boundaryCoeffs.size())
    {
        throw std::runtime_error(std::string(WHO) + "the assembly run elsewhere has another number of patches.");
    }
    for (std::size_t pi = 0; pi < host.internalCoeffs.size(); ++pi)
    {
        const std::vector<vector>& gi = got.internalCoeffs[pi];
        const std::vector<vector>& hi = host.internalCoeffs[pi];
        const std::vector<vector>& gb = got.boundaryCoeffs[pi];
        const std::vector<vector>& hb = host.boundaryCoeffs[pi];
        same("internalCoeffs (3 a face)", gi.empty() ? nullptr : &gi.data()->x, 3*gi.size(),
             hi.empty() ? nullptr : &hi.data()->x, 3*hi.size());
        same("boundaryCoeffs (3 a face)", gb.empty() ? nullptr : &gb.data()->x, 3*gb.size(),
             hb.empty() ? nullptr : &hb.data()->x, 3*hb.size());
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
    {
        interPhase::Nested timed("motion: inverseDistance diffusivity (meshWave)");
        diffusivityCorrect(m, g, patches);
    }
    interPhase::Nested timedRest("motion: assemble and solve cellDisplacement");

    // pointDisplacement_.boundaryFieldRef().updateCoeffs(): valuePointPatchField::updateCoeffs writes
    // each fixing patch's values into the point field, patch by patch
    std::optional<interPhase::Nested> timedPart;
    timedPart.emplace("motion: assembly, the boundary values and the field");
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
    FvMatrix<vector> M;
    if (assemblyRunner_)
    {
        // the interior elsewhere, the patches' coefficients from the code fvm::laplacian runs, and the source
        // as below: zero, minus the correction
        timedPart.emplace("motion: assembly, the interior elsewhere and the patches' coefficients");
        std::vector<std::vector<vector>> bnd(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (pointPatches_[pi].type != PointPatchType::empty)
            {
                bnd[pi] = cellDisplacementBoundary_[pi];
            }
        }
        std::vector<vector> corr;
        assemblyRunner_(patches, faceDiffusivity_.internal, D.internal, bnd, M.upper, M.diag, corr);
        if (M.upper.size() != static_cast<std::size_t>(m.nInternalFaces())
         || M.diag.size() != static_cast<std::size_t>(m.nCells())
         || corr.size() != static_cast<std::size_t>(m.nCells()))
        {
            throw std::runtime_error(std::string(WHO) + "the assembly run elsewhere returned another mesh's.");
        }
        M.lower = M.upper;
        M.source.assign(corr.size(), vector{0, 0, 0});
        fvm::laplacianBoundaryCoeffs<vector>(M, faceDiffusivity_, D, g, patches, true, false);
        for (std::size_t c = 0; c < corr.size(); ++c)
        {
            M.source[c] -= corr[c];
        }
        timedPart.reset();
        // BRAE_CONTROL_MOTION_ASSEMBLY_CHECK=1: the host assembles too, and one bit's difference anywhere in the
        // matrix stops the run and names the entry -- the identity gate's oracle, at every step
        static const bool check = std::getenv("BRAE_CONTROL_MOTION_ASSEMBLY_CHECK") != nullptr;
        if (check)
        {
            sameAssembly(M, assembleOnHost(D, m, g, patches));
        }
    }
    else
    {
        M = assembleOnHost(D, m, g, patches);
    }

    // fvMatrix<vector>::solveSegregated: one GAMG solve per solved component
    const GamgControls& controls = finalIteration ? controlsFinal_ : controls_;
    if (controls.smoother.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "the last PIMPLE iteration solves with the `cellDisplacementFinal` entry, "
            "and system/fvSolution has none.");
    }
    timedPart.emplace("motion: the GAMG agglomeration of the moved mesh");
    const GamgAgglomeration& a = agglomeration.get(m, g, controls.nCellsInCoarsestLevel);
    timedPart.reset();
    const SolutionDirections sd = solutionDirections(patches);
    lastSolve_ = DisplacementSolveRecord();
    interPhase::Nested timedSolves("motion: the component solves");
    // ONE scalar system for the components: the matrix is theirs in common and is taken from M whole, and
    // each component brings only its source and its patches' coefficients, written over the last one's. It
    // was copied afresh for every component, the coefficients pushed back a face at a time. MEASURED on
    // waveMakerPiston refined to 896,000 cells: the component solves 122 -> 94 ms a step, every written file
    // of waveMakerFlap and waveMakerSolitary the same bytes as before.
    FvScalarMatrix Mc;
    Mc.diag = std::move(M.diag);
    Mc.upper = std::move(M.upper);
    Mc.lower = std::move(M.lower);
    Mc.source.resize(M.source.size());
    Mc.internalCoeffs.resize(patches.size());
    Mc.boundaryCoeffs.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        Mc.internalCoeffs[pi].resize(M.internalCoeffs[pi].size());
        Mc.boundaryCoeffs[pi].resize(M.boundaryCoeffs[pi].size());
    }
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        if (!sd.valid(cmpt)) continue;
        auto component = [cmpt](const vector& v)
        {
            return cmpt == 0 ? v.x : (cmpt == 1 ? v.y : v.z);
        };
        for (std::size_t c = 0; c < M.source.size(); ++c)
        {
            Mc.source[c] = component(M.source[c]);
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<vector>& ic = M.internalCoeffs[pi];
            const std::vector<vector>& bc = M.boundaryCoeffs[pi];
            for (std::size_t i = 0; i < ic.size(); ++i)
            {
                Mc.internalCoeffs[pi][i] = component(ic[i]);
            }
            for (std::size_t i = 0; i < bc.size(); ++i)
            {
                Mc.boundaryCoeffs[pi][i] = component(bc[i]);
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
    interPhase::Nested timed("motion: cells to points (volPointInterpolation)");
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
