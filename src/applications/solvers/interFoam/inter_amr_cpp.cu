// interFoam on an adaptive mesh. See inter_amr_cpp.cuh.
#include "inter_amr_cpp.cuh"
#include "foam_dict.cuh"
#include "mesh_cell_cells_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include <filesystem>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

constexpr const char* WHO = "brae interFoam (adaptive mesh): ";

// A solver carries more than fields. Each of these is state a topology change invalidates and that no
// unit has mapped yet, so it throws and names itself rather than surviving into the next step.
void refuseUnmappedState(const InterFields& f)
{
    if (f.turbulence.on)
        throw std::runtime_error(
            std::string(WHO) + "the case runs a turbulence model, whose fields (k, omega or epsilon, nut "
            "and their old times) are not carried through a mesh change yet. RAS/motorBike is the case "
            "that needs it.");
    if (f.waves.any)
        throw std::runtime_error(
            std::string(WHO) + "the case has wave boundary conditions, whose per-patch models hold their "
            "own state and are not carried through a mesh change.");
    if (!f.mrfZones.empty())
        throw std::runtime_error(
            std::string(WHO) + "the case has " + std::to_string(f.mrfZones.size()) + " MRF zone(s), each "
            "resolved against the OLD cell numbering. A change would leave them naming other cells.");
    if (!f.fvOptions.empty())
        throw std::runtime_error(
            std::string(WHO) + "the case has active fvOptions, whose cell sets are resolved against the "
            "OLD numbering.");
    if (f.dynamicMesh)
        throw std::runtime_error(
            std::string(WHO) + "the case asks for a motion solver AND refinement. The mesh would both "
            "move and change topology in one step, and brae carries neither Uf nor the mesh flux through "
            "a topology change yet. laminar/oscillatingBox is the case that needs it.");
    if (f.ddtAlpha == AlphaDdt::CrankNicolson || f.ddtU == DdtScheme::CrankNicolson)
        throw std::runtime_error(
            std::string(WHO) + "the case runs CrankNicolson, whose ddt0 levels are fields of their own and "
            "are not carried through a mesh change yet.");
    if (f.pRef.needReference)
        throw std::runtime_error(
            std::string(WHO) + "the case needs a pressure reference, and pRefCell is a CELL INDEX: after a "
            "change it names a different cell. Renumbering it through the map is its own unit.");
    if (!f.cnRestart.dir.empty())
        throw std::runtime_error(
            std::string(WHO) + "the start directory holds CrankNicolson ddt0 fields, which are not carried "
            "through a mesh change.");
}

}   // namespace

InterAmr readInterAmr(
    const std::string&          caseDir,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    const FvGeometry&           g)
{
    InterAmr amr;
    const std::string path = caseDir + "/constant/dynamicMeshDict";
    if (!std::filesystem::exists(path)) return amr;
    const FoamDict d = readDict(path);
    const std::string meshType = d.wordOr("dynamicFvMesh", "");
    if (meshType != "dynamicRefineFvMesh") return amr;

    amr.active = true;
    amr.controls = dynamicRefine::readRefineControls(d);

    // the mesh this run starts from, and the state that goes with it. A mesh that has never been refined
    // carries no cellLevel on disk and starts at level 0 everywhere; its history is the identity, which is
    // what refinementHistory's own constructor leaves (see the note in hex_ref8_cpp.cuh).
    amr.state.m = m;
    amr.state.patches = patches;
    amr.state.levels.cellLevel.assign(static_cast<std::size_t>(m.nCells()), label(0));
    amr.state.levels.pointLevel.assign(static_cast<std::size_t>(m.nPoints()), label(0));
    amr.state.history = cpu::hexRef8::freshHistory(m.nCells());
    const std::vector<std::vector<label>> cells = meshCells(m);
    const std::vector<std::vector<label>> pointCells = pointCellsFromCells(m, cells);
    amr.state.protectedCell = dynamicRefine::initProtectedCells(
        amr.state.levels.cellLevel, amr.state.levels.pointLevel, pointCells, cells, m, patches);
    (void)g;
    return amr;
}

bool interAmrUpdate(
    InterAmr&              amr,
    InterFields&           f,
    const MutableMesh&     mm,
    label                  timeIndex,
    const InterAmrOldTime& old)
{
    if (!amr.active) return false;
    if (!mm.m || !mm.g || !mm.patches)
        throw std::runtime_error(
            std::string(WHO) + "an adaptive case needs the mesh, its geometry and its patches mutable, and "
            "the driver was handed none. Running it on a copy the fields cannot see would solve a "
            "different problem at every step.");
    refuseUnmappedState(f);

    // The fields the change carries. alpha1, U and p_rgh go whole -- cells and patch fields -- and the
    // three surface fields go through the face mapper: phi and rhoPhi are FLUXES and are oriented, nHatf
    // carries an area and is not.
    amr.state.carriedScalarFields.clear();
    amr.state.carriedVectorFields.clear();
    amr.state.surfaceScalars.clear();
    amr.state.surfaceScalarVelocity.clear();
    amr.state.surfaceVectors.clear();
    amr.state.cellScalars.clear();
    amr.state.carriedScalarFields.push_back(&f.alpha1);
    amr.state.carriedScalarFields.push_back(&f.p_rgh);
    amr.state.carriedVectorFields.push_back(&f.U);
    const auto pushSurface = [&](const SurfaceScalarField& s, bool oriented)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceField c;
        c.field = s.internal;
        c.bnd = s.boundary;
        c.oriented = oriented;
        amr.state.surfaceScalars.push_back(std::move(c));
        // every interFoam tutorial with an adaptive mesh maps every flux to `none`, so no correction runs
        // unless the case names a velocity -- which is read from the dictionary, not assumed
        amr.state.surfaceScalarVelocity.push_back(std::string());
    };
    pushSurface(f.phi, true);
    pushSurface(f.rhoPhi, true);
    pushSurface(f.nHatf, false);

    // Uf AND rAU ARE REQUIREMENTS OF THIS CASE, not moving-mesh extras, and that is measured rather than
    // assumed: `correctPhi` defaults to mesh.dynamic() and a REFINING mesh is dynamic, so OpenFOAM's own
    // run of damBreakWithObstacle writes a Uf and an rAU beside every time directory and solves pcorr at
    // every step. Uf is a surface VECTOR and is NOT oriented -- it is a velocity, not a flux -- so no flip.
    // rAU is a cell field the next CorrectPhi interpolates, so it goes through the cell mapper.
    if (!f.Uf.internal.empty())
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceVectorField uf;
        uf.field = f.Uf.internal;
        uf.bnd = f.Uf.boundary;
        uf.oriented = false;
        amr.state.surfaceVectors.push_back(std::move(uf));
    }
    const bool carryRAU = !f.rAU.empty();
    if (carryRAU) amr.state.cellScalars.push_back(f.rAU);

    // THE OLD-TIME LEVELS, mapped like any other field. The cell ones go through the cell mapper; the
    // per-patch value lists go through the patch mapping, which is what a patch field's own autoMap uses
    // -- they are the same numbers without the object around them.
    const std::size_t scalarsBefore = amr.state.cellScalars.size();
    const std::size_t vectorsBefore = amr.state.cellVectors.size();
    if (old.alphaOld) amr.state.cellScalars.push_back(*old.alphaOld);
    if (old.rhoOld)   amr.state.cellScalars.push_back(*old.rhoOld);
    if (old.rhoOO)    amr.state.cellScalars.push_back(*old.rhoOO);
    if (old.UOld)     amr.state.cellVectors.push_back(*old.UOld);
    if (old.UOO)      amr.state.cellVectors.push_back(*old.UOO);
    const bool carryPhiOld = old.phiOld != nullptr;
    if (carryPhiOld)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceField c;
        c.field = old.phiOld->internal;
        c.bnd = old.phiOld->boundary;
        c.oriented = true;                    // phi.oldTime() is a flux like phi
        amr.state.surfaceScalars.push_back(std::move(c));
        amr.state.surfaceScalarVelocity.push_back(std::string());
    }
    // ...and U's old-time PATCH values, which ddtCorr's boundary half reads. They are boundary lists
    // without a surface field around them, so they ride in as surface VECTORS whose internal half is
    // empty: the patch mapping is the same addressing either way, and leaving them unmapped left ddtCorr
    // reading the OLD patch sizes.
    const std::size_t surfVecBefore = amr.state.surfaceVectors.size();
    const auto pushBoundaryOnly = [&](const std::vector<std::vector<vector>>& b)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceVectorField c;
        c.bnd = b;
        c.oriented = false;
        amr.state.surfaceVectors.push_back(std::move(c));
    };
    if (old.UOldBnd) pushBoundaryOnly(*old.UOldBnd);
    if (old.UOOBnd)  pushBoundaryOnly(*old.UOOBnd);

    const dynamicRefine::RefineUpdateStep step =
        dynamicRefine::refineUpdate(amr.state, amr.controls, f.alpha1.internal, timeIndex);
    amr.nRefined = static_cast<label>(step.cellsToRefine.size());
    amr.nUnrefined = static_cast<label>(step.pointsToUnrefine.size());
    if (!step.hasChanged) return false;

    if (amr.state.surfaceScalars.size() != (carryPhiOld ? 4u : 3u))
        throw std::runtime_error(
            std::string(WHO) + "the change carried " + std::to_string(amr.state.surfaceScalars.size())
            + " surface scalars where three went in (phi, rhoPhi, nHatf). Reading them back by index "
            "would put one field's values into another.");

    // the surface fields back out, in the order they went in
    f.phi.internal = amr.state.surfaceScalars[0].field;
    f.phi.boundary = amr.state.surfaceScalars[0].bnd;
    f.rhoPhi.internal = amr.state.surfaceScalars[1].field;
    f.rhoPhi.boundary = amr.state.surfaceScalars[1].bnd;
    f.nHatf.internal = amr.state.surfaceScalars[2].field;
    f.nHatf.boundary = amr.state.surfaceScalars[2].bnd;
    if (!amr.state.surfaceVectors.empty())
    {
        f.Uf.internal = amr.state.surfaceVectors[0].field;
        f.Uf.boundary = amr.state.surfaceVectors[0].bnd;
    }
    if (carryRAU) f.rAU = amr.state.cellScalars.at(0);
    {
        std::size_t si = scalarsBefore;
        std::size_t vi = vectorsBefore;
        if (old.alphaOld) *old.alphaOld = amr.state.cellScalars.at(si++);
        if (old.rhoOld)   *old.rhoOld   = amr.state.cellScalars.at(si++);
        if (old.rhoOO)    *old.rhoOO    = amr.state.cellScalars.at(si++);
        if (old.UOld)     *old.UOld     = amr.state.cellVectors.at(vi++);
        if (old.UOO)      *old.UOO      = amr.state.cellVectors.at(vi++);
        if (carryPhiOld)
        {
            old.phiOld->internal = amr.state.surfaceScalars.at(3).field;
            old.phiOld->boundary = amr.state.surfaceScalars.at(3).bnd;
        }
        std::size_t bi = surfVecBefore;
        if (old.UOldBnd) *old.UOldBnd = amr.state.surfaceVectors.at(bi++).bnd;
        if (old.UOOBnd)  *old.UOOBnd  = amr.state.surfaceVectors.at(bi++).bnd;
    }

    // ...and the mesh the caller's fields reference. The patches are assigned IN PLACE by the driver, so
    // `*mm.patches` is already the new one; the mesh and geometry are copied into the caller's objects for
    // the same reason -- every FvPatch, every patch field and every operator reads those.
    *mm.m = amr.state.m;
    mm.g->build(*mm.m);
    *mm.patches = amr.state.patches;

    // WHAT interFoam REBUILDS AFTER A CHANGE (interFoam.C:139-147), and it is recomputed rather than
    // mapped because every one of these is a function of the new mesh or of alpha1:
    //   gh and ghf off the new cell and face centres, p from them, and the mixture -- alpha2, rho, mu, nu
    //   and the curvature -- from the mapped alpha1.
    // The driver's own post-change stage does that work; this returns `true` so it runs.
    return true;
}

void interAfterMeshChange(
    InterFields&            f,
    const MutableMesh&      mm,
    GamgAgglomerationCache& gamgCache,
    RunReport&              rep)
{
    const PrimitiveMesh& m = *mm.m;
    const FvGeometry& g = *mm.g;
    const std::vector<FvPatch>& patches = *mm.patches;
    const std::size_t nC = static_cast<std::size_t>(m.nCells());

    // the agglomeration was built for the old mesh
    gamgCache = GamgAgglomerationCache{};

    // gh and ghf off the NEW centres (interFoam.C:130-131), assigned rather than written into
    ghField(f.g, f.ghRefValue, g.C(), f.gh);
    {
        std::vector<vector> Cf(g.Cf().begin(), g.Cf().begin() + m.nInternalFaces());
        ghField(f.g, f.ghRefValue, Cf, f.ghfInternal);
        f.ghfBoundary.assign(patches.size(), std::vector<scalar>());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            std::vector<vector> bCf(g.Cf().begin() + q.start, g.Cf().begin() + q.start + q.size);
            ghField(f.g, f.ghRefValue, bCf, f.ghfBoundary[pi]);
        }
    }

    // the mixture from the MAPPED alpha1 (mixture.correct()), every one of these assigning its own result
    f.alpha2.assign(nC, scalar(0));
    for (std::size_t c = 0; c < nC; ++c) f.alpha2[c] = scalar(1) - f.alpha1.internal[c];
    cpu::twoPhase::mixtureRho(f.alpha1.internal, f.alpha2, f.mixture.phases, f.rho);
    cpu::twoPhase::mixtureMu(f.alpha1.internal, f.mixture.phases, f.mu);
    cpu::twoPhase::mixtureNu(f.alpha1.internal, f.mu, f.mixture.phases, f.nu);
    updateMixtureBoundary(f, patches);

    // p = p_rgh + rho*gh (createFields.H:88), on the new mesh
    f.p.assign(nC, scalar(0));
    for (std::size_t c = 0; c < nC; ++c)
    {
        f.p[c] = f.p_rgh.internal[c] + f.rho[c]*f.gh[c];
    }

    // ...and the interface: nHatf and K are mapped, and then rebuilt on the new geometry, which is what
    // mixture.correct() does last
    interfaceProps::calculateK(f.alpha1, f.interface, m, g, patches, false, f.nHatf, f.K);

    (void)rep;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
