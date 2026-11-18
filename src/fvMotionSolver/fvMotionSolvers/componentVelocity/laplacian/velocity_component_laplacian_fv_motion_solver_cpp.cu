#include "velocity_component_laplacian_fv_motion_solver_cpp.cuh"
#include "face_cpp.cuh"
#include "foam_field_reader.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "geometric_field.cuh"
#include "primitive_patch_cpp.cuh"
#include "scheme_parse.cuh"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace brae {

namespace {

const char* const WHO = "brae velocityComponentLaplacian: ";

bool fileExists(const std::string& path)
{
    return std::filesystem::exists(path) || std::filesystem::exists(path + ".gz");
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

// a number token, its parentheses stripped (`(1`, `0)`)
scalar number(std::string t, const std::string& where)
{
    while (!t.empty() && t.front() == '(') t.erase(t.begin());
    while (!t.empty() && t.back() == ')') t.pop_back();
    char* end = nullptr;
    const double v = std::strtod(t.c_str(), &end);
    if (t.empty() || *end != '\0')
    {
        throw std::runtime_error(std::string(WHO) + where + ": `" + t + "` is not a number.");
    }
    return v;
}

} // namespace

std::unique_ptr<VelocityComponentLaplacianFvMotionSolver> VelocityComponentLaplacianFvMotionSolver::New(
    const FoamDict& coeffs,
    const std::string& caseDir,
    const std::string& startDir)
{
    std::unique_ptr<VelocityComponentLaplacianFvMotionSolver> s(new VelocityComponentLaplacianFvMotionSolver());
    s->caseDir_ = caseDir;
    s->startDir_ = startDir;

    // componentVelocityMotionSolver: cmptName_(coeffDict().get<word>("component"))
    s->cmptName_ = coeffs.wordOr("component", "");
    if (s->cmptName_ == "x") s->cmpt_ = 0;
    else if (s->cmptName_ == "y") s->cmpt_ = 1;
    else if (s->cmptName_ == "z") s->cmpt_ = 2;
    else
    {
        throw std::runtime_error(
            std::string(WHO) + "`component " + s->cmptName_ + "`: OpenFOAM reads x, y or z (vector::component), "
            "with a mandatory get<word>.");
    }

    // motionDiffusivity::New(fvMesh_, coeffDict().lookup("diffusivity")): uniform, or directional (Dx Dy Dz)
    const std::vector<std::string> diffusivity = coeffs.wordListOr("diffusivity", {});
    if (diffusivity.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "no `diffusivity`. OpenFOAM reads it with a mandatory lookup.");
    }
    if (diffusivity.front() == "uniform")
    {
        s->directional_ = false;
    }
    else if (diffusivity.front() == "directional")
    {
        if (diffusivity.size() != 4)
        {
            throw std::runtime_error(
                std::string(WHO) + "`diffusivity " + joined(diffusivity) + "`: directional reads one vector.");
        }
        s->directional_ = true;
        s->diffusivityVector_ = vector{number(diffusivity[1], "diffusivity"), number(diffusivity[2], "diffusivity"),
                                       number(diffusivity[3], "diffusivity")};
    }
    else
    {
        throw std::runtime_error(
            std::string(WHO) + "`diffusivity " + joined(diffusivity) + "`. Only uniform and directional are ported "
            "for this solver; another diffusivity distributes the motion differently.");
    }
    if (coeffs.found("interpolation"))
    {
        throw std::runtime_error(std::string(WHO) + "`interpolation` is set, which is not ported (motionInterpolation "
                                 "default: volPointInterpolation).");
    }
    for (const char* key : {"cellZone", "cellSet"})
    {
        const std::string name = coeffs.wordOr(key, "");
        if (!name.empty() && name != "none")
        {
            throw std::runtime_error(std::string(WHO) + "`" + key + " " + name + "` is not ported.");
        }
    }

    // pointMotionU<c>: MUST_READ
    const std::string pm = startDir + "/pointMotionU" + s->cmptName_;
    if (!fileExists(pm))
    {
        throw std::runtime_error(
            std::string(WHO) + pm + " does not exist. componentVelocityMotionSolver reads it with MUST_READ.");
    }

    // fvSolution's cellMotionU<c>: PCG with DIC
    {
        const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
        const FoamDict* solvers = fvSolution.subDict("solvers");
        const std::string name = "cellMotionU" + s->cmptName_;
        const FoamDict* d = solvers ? solvers->subDict(name) : nullptr;
        if (!d)
        {
            throw std::runtime_error(
                std::string(WHO) + "system/fvSolution has no solver entry for `" + name + "`. "
                "solution::solverDict reads it with a mandatory lookup.");
        }
        const std::string solver = d->wordOr("solver", "");
        const std::string pre = d->wordOr("preconditioner", "");
        if (solver != "PCG" || pre != "DIC")
        {
            throw std::runtime_error(
                std::string(WHO) + "system/fvSolution solves `" + name + "` with `solver " + solver +
                "; preconditioner " + pre + "`. Only PCG with DIC is ported for this equation.");
        }
        s->tolerance_ = d->scalarOr("tolerance", scalar(1e-6));
        s->relTol_ = d->scalarOr("relTol", scalar(0));
        s->maxIter_ = static_cast<int>(d->scalarOr("maxIter", scalar(1000)));
        s->minIter_ = static_cast<int>(d->scalarOr("minIter", scalar(0)));
    }

    // fvSchemes' laplacian(diffusivity,cellMotionU) (the name fvm::laplacian is given)
    {
        // fvm::laplacian(..., "laplacian(diffusivity,cellMotionU)"): the scheme by that name, else the default
        const std::string lap = fvSchemesStatement(caseDir, "laplacianSchemes", "laplacian(diffusivity,cellMotionU)");
        if (lap == "Gauss linear uncorrected" || lap == "Gauss linear limited 0")
        {
            s->orthogonal_ = false;
        }
        else if (lap == "Gauss linear orthogonal")
        {
            s->orthogonal_ = true;
        }
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "laplacian(diffusivity,cellMotionU) is `" + lap + "`. Ported: Gauss linear "
                "uncorrected (and limited 0), Gauss linear orthogonal; a corrected scheme adds the non-orthogonal "
                "correction, which is not ported for this equation.");
        }
    }
    return s;
}

void VelocityComponentLaplacianFvMotionSolver::attach(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const std::string pmPath = startDir_ + "/pointMotionU" + cmptName_;
    const FieldData<scalar> pfd = readField<scalar>(pmPath);
    if (!pfd.internalUniform)
    {
        // a nonuniform point field is a restart's: its values are the last curPoints', overwritten by the next one
        // except where a point is on no patch and no cell -- read it all the same
        if (pfd.internalField.size() != m.points().size())
        {
            throw std::runtime_error(std::string(WHO) + pmPath + " has " + std::to_string(pfd.internalField.size()) +
                                     " values for " + std::to_string(m.points().size()) + " points.");
        }
        pointMotionU_ = pfd.internalField;
    }
    else
    {
        pointMotionU_.assign(m.points().size(), pfd.internalUniformValue);
    }

    auto entryFor = [&](const FvPatch& p) -> const PatchFieldData<scalar>* {
        for (const PatchFieldData<scalar>& e : pfd.boundary)
            if (e.name == p.name) return &e;
        for (const std::string& grp : p.inGroups)
            for (const PatchFieldData<scalar>& e : pfd.boundary)
                if (e.name == grp) return &e;
        return nullptr;
    };

    pointPatches_.clear();
    pointPatches_.resize(patches.size());
    cellMotionUTypes_.assign(patches.size(), "");
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        PointPatch& pp = pointPatches_[pi];
        pp.name = p.name;
        if (isCoupledInterfaceType(p.type))
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + p.name + "` is " + p.type + ". A coupled patch couples the point "
                "field and the interpolation across it, which is not ported.");
        }
        const PatchFieldData<scalar>* e = entryFor(p);
        std::string type;
        if (e) type = e->type;
        else if (p.type == "empty") type = "empty";
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "pointMotionU" + cmptName_ + " has no boundaryField entry for patch `" + p.name +
                "`.");
        }
        // pointPatchField::New: on a constraint patch the patch's own constraint type replaces the dictionary's
        // (pointPatchFieldNew.C), unless `patchType` names it
        if (p.type == "wedge" || p.type == "symmetryPlane" || p.type == "symmetry" || p.type == "empty")
        {
            type = p.type;
        }
        pp.typeName = type;
        pp.meshPoints = primitivePatch(m, faceRange(p.start, p.size)).meshPoints;
        if (type == "fixedValue")
        {
            pp.type = PointPatchType::fixedValue;
            if (!e || !e->hasValue || !e->valueUniform)
            {
                throw std::runtime_error(std::string(WHO) + "pointMotionU" + cmptName_ + "'s patch `" + p.name +
                                         "` is fixedValue without a uniform `value`; only a uniform one is read.");
            }
            pp.value = e->uniformValue;
        }
        else if (type == "uniformFixedValue")
        {
            pp.type = PointPatchType::uniformFixedValue;
            if (!e->unsupportedFunction1.empty() || e->hasPatchExpr)
            {
                throw std::runtime_error(std::string(WHO) + "pointMotionU" + cmptName_ + "'s patch `" + p.name +
                                         "` is uniformFixedValue with a `" + e->unsupportedFunction1 +
                                         "` Function1; only a constant is ported.");
            }
            pp.value = e->uniformValue;   // `constant <v>` or a bare <v> (Function1 shorthand)
        }
        else if (type == "zeroGradient") pp.type = PointPatchType::zeroGradient;
        else if (type == "slip") pp.type = PointPatchType::slip;
        else if (type == "symmetryPlane") pp.type = PointPatchType::symmetryPlane;
        else if (type == "symmetry") pp.type = PointPatchType::symmetry;
        else if (type == "wedge") pp.type = PointPatchType::wedge;
        else if (type == "empty") pp.type = PointPatchType::empty;
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "pointMotionU" + cmptName_ + "'s patch `" + p.name + "` is `" + type + "`. "
                "Ported: fixedValue, uniformFixedValue (constant), zeroGradient, slip, symmetryPlane, symmetry, "
                "wedge, empty.");
        }
        // cellMotionBoundaryTypes: a fixedValue-derived point patch gives the cell field cellMotion; any other keeps
        // its point type's name (slip, wedge, ... as fvPatchField<scalar>: zeroGradient for a scalar)
        cellMotionUTypes_[pi] = pp.fixesValue() ? "cellMotion" : type;
    }

    // cellMotionU<c>: READ_IF_PRESENT, else zero (dimensionedScalar Zero); its patches zero until evaluated
    cellMotionU_.assign(static_cast<std::size_t>(m.nCells()), scalar(0));
    cellMotionURead_ = false;
    const std::string cmPath = startDir_ + "/cellMotionU" + cmptName_;
    if (fileExists(cmPath))
    {
        const FieldData<scalar> cfd = readField<scalar>(cmPath);
        if (cfd.internalUniform) cellMotionU_.assign(static_cast<std::size_t>(m.nCells()), cfd.internalUniformValue);
        else
        {
            if (cfd.internalField.size() != static_cast<std::size_t>(m.nCells()))
                throw std::runtime_error(std::string(WHO) + cmPath + " has another mesh's cell count.");
            cellMotionU_ = cfd.internalField;
        }
        cellMotionURead_ = true;
    }
    cellMotionUBoundary_.assign(patches.size(), std::vector<scalar>());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (pointPatches_[pi].type == PointPatchType::empty) continue;
        cellMotionUBoundary_[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(0));
        if (!pointPatches_[pi].fixesValue())   // GeometricField's construction evaluates a zeroGradient-like patch
        {
            for (label i = 0; i < patches[pi].size; ++i)
                cellMotionUBoundary_[pi][static_cast<std::size_t>(i)] =
                    cellMotionU_[static_cast<std::size_t>(patches[pi].faceCells[static_cast<std::size_t>(i)])];
        }
    }

    twoDCorrector_.build(m, g, patches);
    pointCells_ = pointCellsFromPointFaces(m, meshPointFaces(m));
    interpolation_.clearAddressing();
    attached_ = true;
}

void VelocityComponentLaplacianFvMotionSolver::diffusivityCorrect(
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    // uniformDiffusivity: 1 everywhere; directionalDiffusivity::correct: n = Sf/magSf, faceDiffusivity_ ==
    // (n & cmptMultiply(diffusivityVector_, n)) on the internal faces and every patch
    auto dd = [&](std::size_t f) {
        const vector sf = g.Sf()[f];
        const scalar ms = g.magSf()[f];
        const vector n{sf.x/ms, sf.y/ms, sf.z/ms};
        const vector c{diffusivityVector_.x*n.x, diffusivityVector_.y*n.y, diffusivityVector_.z*n.z};
        return n.x*c.x + n.y*c.y + n.z*c.z;
    };
    const std::size_t nIF = g.Sf().size() - [&] {
        std::size_t nb = 0;
        for (const FvPatch& p : patches) nb += static_cast<std::size_t>(p.size);
        return nb;
    }();
    faceDiffusivity_.internal.resize(nIF);
    for (std::size_t f = 0; f < nIF; ++f) faceDiffusivity_.internal[f] = directional_ ? dd(f) : scalar(1);
    faceDiffusivity_.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        std::vector<scalar>& b = faceDiffusivity_.boundary[pi];
        b.assign(static_cast<std::size_t>(p.size), scalar(0));
        if (p.type == "empty") continue;
        for (label i = 0; i < p.size; ++i)
            b[static_cast<std::size_t>(i)] = directional_ ? dd(static_cast<std::size_t>(p.start + i)) : scalar(1);
    }
}

void VelocityComponentLaplacianFvMotionSolver::solve(
    scalar time,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    (void)time;   // uniformFixedValue's Function1 is a constant (the only one ported)
    diffusivityCorrect(g, patches);

    // pointMotionU_.boundaryFieldRef().updateCoeffs(): each fixing patch writes its values into the point field
    for (const PointPatch& pp : pointPatches_)
    {
        if (!pp.fixesValue()) continue;
        for (const label pt : pp.meshPoints) pointMotionU_[static_cast<std::size_t>(pt)] = pp.value;
    }

    // the field the matrix is assembled on; fvMatrix's construction runs cellMotion's updateCoeffs: the face average
    // of the POINT field over the mesh's current points
    GeometricField<scalar> U;
    U.internal = cellMotionU_;
    U.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        const PointPatch& pp = pointPatches_[pi];
        if (pp.fixesValue())
        {
            for (label i = 0; i < p.size; ++i)
                cellMotionUBoundary_[pi][static_cast<std::size_t>(i)] =
                    faceAverage(m, p.start + i, m.points(), pointMotionU_);
            U.boundary[pi].reset(new FixedValuePatchField<scalar>(p, false, scalar(0), cellMotionUBoundary_[pi]));
            U.boundary[pi]->evaluate(U.internal);
        }
        else if (pp.type == PointPatchType::empty)
        {
            U.boundary[pi].reset(new EmptyPatchField<scalar>(p));
        }
        else   // zeroGradient, and slip/symmetry/wedge, which a scalar sees as zeroGradient (transformFvPatchFields.C)
        {
            U.boundary[pi].reset(new ZeroGradientPatchField<scalar>(p));
            U.boundary[pi]->setStoredValues(cellMotionUBoundary_[pi]);
        }
    }

    // fvm::laplacian(1*diffusivity, cellMotionU<c>): Gauss linear uncorrected (nonOrthDeltaCoeffs, no correction)
    // or orthogonal (deltaCoeffs)
    FvScalarMatrix M = fvm::laplacian<scalar>(faceDiffusivity_, U, m, g, patches, false, !orthogonal_);
    std::vector<scalar> psi = cellMotionU_;
    lastSolve_ = pcg(M, psi, m, patches, tolerance_, relTol_, maxIter_, minIter_);
    cellMotionU_ = psi;

    // psi.correctBoundaryConditions(): the zeroGradient-like patches take the new cell values, cellMotion keeps its own
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const PointPatch& pp = pointPatches_[pi];
        if (pp.fixesValue() || pp.type == PointPatchType::empty) continue;
        const FvPatch& p = patches[pi];
        for (label i = 0; i < p.size; ++i)
            cellMotionUBoundary_[pi][static_cast<std::size_t>(i)] =
                cellMotionU_[static_cast<std::size_t>(p.faceCells[static_cast<std::size_t>(i)])];
    }
}

std::vector<vector> VelocityComponentLaplacianFvMotionSolver::curPoints(
    scalar deltaT,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    // interpolationPtr_->interpolate(cellMotionU_, pointMotionU_): volPointInterpolation, the weights for the mesh as
    // it stands (movePoints)
    interpolation_.makeWeights(m, g, patches, pointCells_);
    interpolation_.interpolateInternalField(cellMotionU_, pointMotionU_);
    interpolation_.interpolateBoundaryField(cellMotionUBoundary_, pointMotionU_);
    // pointConstraints::constrain(pf, false): correctBoundaryConditions, patch by patch -- the fixing patches write
    // their values back; slip, symmetry and wedge on a scalar evaluate to the value they hold (a scalar's transform is
    // itself), and so do the corner constraints
    for (const PointPatch& pp : pointPatches_)
    {
        if (!pp.fixesValue()) continue;
        for (const label pt : pp.meshPoints) pointMotionU_[static_cast<std::size_t>(pt)] = pp.value;
    }

    // tcurPoints.replace(cmpt_, points.component(cmpt_) + deltaT*pointMotionU_): two Field operations, each its own
    // loop through memory -- the product rounded before the sum, never one fused multiply-add -- and so two loops here
    std::vector<scalar> d(pointMotionU_.size());
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = deltaT*pointMotionU_[i];
    std::vector<vector> cur = m.points();
    for (std::size_t i = 0; i < cur.size(); ++i)
    {
        if (cmpt_ == 0) cur[i].x = cur[i].x + d[i];
        else if (cmpt_ == 1) cur[i].y = cur[i].y + d[i];
        else cur[i].z = cur[i].z + d[i];
    }
    twoDCorrector_.correctPoints(m.points(), cur);
    return cur;
}

std::vector<vector> VelocityComponentLaplacianFvMotionSolver::newPoints(
    scalar time,
    scalar deltaT,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!attached_)
    {
        throw std::runtime_error(std::string(WHO) + "newPoints() before attach().");
    }
    solve(time, m, g, patches);
    return curPoints(deltaT, m, g, patches);
}

} // namespace brae
