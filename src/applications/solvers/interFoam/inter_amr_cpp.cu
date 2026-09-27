// interFoam on an adaptive mesh. See inter_amr_cpp.cuh.
#include "inter_amr_cpp.cuh"
#include "foam_dict.cuh"
#include "mesh_cell_cells_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include <cstdio>
#include <cstdlib>
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
    if (!f.correctPhi)
        throw std::runtime_error(
            std::string(WHO) + "the case sets correctPhi off. interFoam.C:130-142 puts mixture.correct() "
            "INSIDE the correctPhi branch, so with it off OpenFOAM keeps the MAPPED rho, mu, nu, nHatf and "
            "curvature rather than recomputing them -- and brae carries neither the curvature nor the "
            "mixture through a change, so it would recompute what OpenFOAM mapped.");
    if (!f.cnRestart.dir.empty())
        throw std::runtime_error(
            std::string(WHO) + "the start directory holds CrankNicolson ddt0 fields, which are not carried "
            "through a mesh change.");
}

// dynamicRefineFvMesh.C:309-336: WHICH carried flux gets the mapFields correction is the CASE's own
// decision, read out of correctFluxes by the field's name -- not a property of the field. OpenFOAM's
// three outcomes: "none" maps the field and corrects nothing; a name OpenFOAM cannot find in the table
// warns and also corrects nothing; anything else names the velocity to re-interpolate the flux from.
// "NaN" is a fourth, a debugging aid that fills the changed faces with NaN.
//
// Only the first is ported. The correction itself is gated (unit 7b-2) but it needs an INTERPOLATED FLUX
// on the new mesh, which only the gate injects, so a case that names a velocity is refused by name rather
// than quietly mapped -- which would be the silent substitution: a mapped flux where OpenFOAM
// re-interpolates one. All three interFoam tutorials with an adaptive mesh say `none` to every flux.
std::string fluxVelocityFor(
    const dynamicRefine::RefineControls& c,
    const std::string&                   fieldName)
{
    for (const std::pair<std::string, std::string>& pr : c.correctFluxes)
    {
        if (pr.first != fieldName) continue;
        if (pr.second == "none") return std::string();
        if (pr.second == "NaN")
            throw std::runtime_error(
                std::string(WHO) + "the case maps flux " + fieldName + " to NaN in correctFluxes, which "
                "asks OpenFOAM to POISON the changed faces as a debugging aid. brae does not.");
        throw std::runtime_error(
            std::string(WHO) + "the case asks for flux " + fieldName + " to be re-interpolated from "
            + pr.second + " on every changed face (correctFluxes). The correction is ported and gated, but "
            "it reads an interpolated flux on the NEW mesh that only the gate injects, so this case would "
            "run with a mapped flux where OpenFOAM re-interpolates one.");
    }
    // OpenFOAM warns and maps only; the warning is the behaviour, so it is printed rather than thrown
    std::fprintf(stderr,
                 "%sthe case's correctFluxes does not name %s, so it is mapped and not corrected. "
                 "OpenFOAM prints the same warning at every change.\n", WHO, fieldName.c_str());
    return std::string();
}

}   // namespace

bool caseAsksForAdaptiveMesh(const std::string& caseDir)
{
    const std::string path = caseDir + "/constant/dynamicMeshDict";
    if (!std::filesystem::exists(path)) return false;
    return readDict(path).wordOr("dynamicFvMesh", "") == "dynamicRefineFvMesh";
}

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
    if (d.wordOr("dynamicFvMesh", "") != "dynamicRefineFvMesh") return amr;

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
    // ...and the VECTORS, which held the previous change's old-time levels: left in place, the read-back
    // indices below count past them and every old-time level comes back as another step's field.
    amr.state.cellVectors.clear();
    amr.state.carriedScalarFields.push_back(&f.alpha1);
    amr.state.carriedScalarFields.push_back(&f.p_rgh);
    amr.state.carriedVectorFields.push_back(&f.U);
    const auto pushSurface = [&](const SurfaceScalarField& s, bool oriented, const char* ofName)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceField c;
        c.field = s.internal;
        c.bnd = s.boundary;
        c.oriented = oriented;
        amr.state.surfaceScalars.push_back(std::move(c));
        // ...and whether the CASE asks for this flux to be corrected, looked up by the name OpenFOAM
        // registers the field under, which is the name its correctFluxes table uses
        amr.state.surfaceScalarVelocity.push_back(fluxVelocityFor(amr.controls, ofName));
    };
    pushSurface(f.phi, true, "phi");
    pushSurface(f.rhoPhi, true, "rhoPhi");
    // nHatf IS ORIENTED, and that is OpenFOAM's own flag rather than a reading of the name: it is
    // assigned `nHatfv & Sf` (interfaceProperties.C), Sf is setOriented (fvMeshGeometry.C:71), a product
    // XORs the two flags (orientedType::operator*=) and GeometricField::operator=(tmp) copies the result
    // onto the field (GeometricField.C:1378). So it is negated on a flipped face and hull-averaged as an
    // intensive vector, exactly as a flux is.
    pushSurface(f.nHatf, true, "nHatf");

    // Uf AND rAU ARE REQUIREMENTS OF THIS CASE, not moving-mesh extras, and that is measured rather than
    // assumed: `correctPhi` defaults to mesh.dynamic() and a REFINING mesh is dynamic, so OpenFOAM's own
    // run of damBreakWithObstacle writes a Uf and an rAU beside every time directory and solves pcorr at
    // every step. Uf is a surface VECTOR and is NOT oriented -- it is a velocity, not a flux -- so no flip.
    // rAU is a cell field the next CorrectPhi interpolates, so it goes through the cell mapper.
    const auto pushSurfaceVector = [&](const SurfaceVectorField& s)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceVectorField c;
        c.field = s.internal;
        c.bnd = s.boundary;
        c.oriented = false;
        amr.state.surfaceVectors.push_back(std::move(c));
    };
    const bool carryUf = !f.Uf.internal.empty();
    const std::size_t ufIndex = amr.state.surfaceVectors.size();
    if (carryUf) pushSurfaceVector(f.Uf);
    // ...and Uf.oldTime(), the level ddtCorr reads on a dynamic mesh. Carried on the same flag as Uf:
    // the driver holds it as a local and it exists exactly when Uf does.
    const bool carryUfOld = carryUf && old.UfOld != nullptr && !old.UfOld->internal.empty();
    const std::size_t ufOldIndex = amr.state.surfaceVectors.size();
    if (carryUfOld) pushSurfaceVector(*old.UfOld);
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
    // ...AND alpha2's OWN PATCH VALUES, which are not 1 - alpha1's: `alpha2 = 1 - alpha1` runs one line
    // above the corrector's mixture.correct() (alphaEqn.H:223), so at a contact-angle wall alpha2's patch
    // value is the alpha1 patch value from BEFORE that call re-evaluated it. OpenFOAM's alpha2 is a
    // registered volScalarField and its patch fields are autoMapped like any other; brae keeps the
    // numbers without the object, so they ride through the same patch mapping.
    //
    // WHY IT MATTERS EVEN THOUGH IT IS OVERWRITTEN: updateMixtureBoundary reads it at the post-change
    // rebuild (rhoBnd = alpha1_b*rho1 + alpha2_b*rho2) with a guard that falls back to 1 - alpha1_b per
    // face. Left unmapped, a patch that GREW blends the new face's alpha1 with another face's alpha2 for
    // every face the old patch had, and takes the fallback beyond it -- silently, at the old size, which
    // is the defect class this unit exists to prevent. EMPTY is its own state and is faithful: before the
    // first alpha step 1 - alpha1_b is exactly what OpenFOAM's createFields leaves.
    const bool carryAlpha2Bnd = !f.alpha2Bnd.empty();
    const std::size_t alpha2BndIndex = amr.state.surfaceScalars.size();
    if (carryAlpha2Bnd)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceField c;
        c.bnd = f.alpha2Bnd;
        c.oriented = false;
        amr.state.surfaceScalars.push_back(std::move(c));
        amr.state.surfaceScalarVelocity.push_back(std::string());
    }

    // ...and U's old-time PATCH values, which ddtCorr's boundary half reads. They are boundary lists
    // without a surface field around them, so they ride in as surface VECTORS whose internal half is
    // empty: the patch mapping is the same addressing either way, and leaving them unmapped left ddtCorr
    // reading the OLD patch sizes.
    const std::size_t surfVecBefore = amr.state.surfaceVectors.size();
    (void)surfVecBefore;
    const auto pushBoundaryOnly = [&](const std::vector<std::vector<vector>>& b)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceVectorField c;
        c.bnd = b;
        c.oriented = false;
        amr.state.surfaceVectors.push_back(std::move(c));
    };
    if (old.UOldBnd) pushBoundaryOnly(*old.UOldBnd);
    if (old.UOOBnd)  pushBoundaryOnly(*old.UOOBnd);

    // A GATE'S CONTROL, announced at every change: the carried CELL fields are RESIZED to the new count
    // instead of mapped -- the old values kept by index, the new cells left at zero. That is the defect
    // class this whole unit exists to prevent ("resized by its recompute path rather than indexed at the
    // old size"), and it is the control for the mapping half. It makes the answer wrong.
    const bool resizeNotMap = (std::getenv("BRAE_CONTROL_AMR_RESIZE_NOT_MAP") != nullptr);
    std::vector<scalar> preAlpha, prePrgh;
    std::vector<vector> preU;
    if (resizeNotMap)
    {
        std::printf("  *** CONTROL MODE: the carried cell fields are RESIZED, not mapped. This run is "
                    "deliberately wrong. ***\n");
        preAlpha = f.alpha1.internal;
        prePrgh = f.p_rgh.internal;
        preU = f.U.internal;
    }

    const dynamicRefine::RefineUpdateStep step =
        dynamicRefine::refineUpdate(amr.state, amr.controls, f.alpha1.internal, timeIndex);
    amr.nRefined = static_cast<label>(step.cellsToRefine.size());
    amr.nUnrefined = static_cast<label>(step.pointsToUnrefine.size());
    if (!step.hasChanged) return false;

    {
        const std::size_t want = 3u + (carryPhiOld ? 1u : 0u) + (carryAlpha2Bnd ? 1u : 0u);
        if (amr.state.surfaceScalars.size() != want)
            throw std::runtime_error(
                std::string(WHO) + "the change carried " + std::to_string(amr.state.surfaceScalars.size())
                + " surface scalars where " + std::to_string(want) + " went in (phi, rhoPhi, nHatf, and "
                "phi's old time and alpha2's patch values where those exist). Reading them back by index "
                "would put one field's values into another.");
    }

    // the surface fields back out, in the order they went in
    f.phi.internal = amr.state.surfaceScalars[0].field;
    f.phi.boundary = amr.state.surfaceScalars[0].bnd;
    f.rhoPhi.internal = amr.state.surfaceScalars[1].field;
    f.rhoPhi.boundary = amr.state.surfaceScalars[1].bnd;
    f.nHatf.internal = amr.state.surfaceScalars[2].field;
    f.nHatf.boundary = amr.state.surfaceScalars[2].bnd;
    // the surface vectors back out BY THE INDEX EACH WENT IN AT. Reading Uf from slot 0 was wrong the
    // moment Uf was absent: slot 0 is then U's old-time PATCH values, whose internal half is empty, so a
    // case with no Uf would have had its face velocity silently emptied.
    {
        const std::size_t want =
            (carryUf ? 1u : 0u) + (carryUfOld ? 1u : 0u)
          + (old.UOldBnd ? 1u : 0u) + (old.UOOBnd ? 1u : 0u);
        if (amr.state.surfaceVectors.size() != want)
            throw std::runtime_error(
                std::string(WHO) + "the change carried " + std::to_string(amr.state.surfaceVectors.size())
                + " surface vectors where " + std::to_string(want) + " went in. Reading them back by "
                "index would put one field's values into another.");
    }
    if (carryUf)
    {
        f.Uf.internal = amr.state.surfaceVectors[ufIndex].field;
        f.Uf.boundary = amr.state.surfaceVectors[ufIndex].bnd;
    }
    if (carryUfOld)
    {
        old.UfOld->internal = amr.state.surfaceVectors[ufOldIndex].field;
        old.UfOld->boundary = amr.state.surfaceVectors[ufOldIndex].bnd;
    }
    if (carryAlpha2Bnd) f.alpha2Bnd = amr.state.surfaceScalars[alpha2BndIndex].bnd;
    if (carryRAU) f.rAU = amr.state.cellScalars.at(0);
    if (amr.state.cellScalars.size() != scalarsBefore
        + (old.alphaOld ? 1u : 0u) + (old.rhoOld ? 1u : 0u) + (old.rhoOO ? 1u : 0u)
     || amr.state.cellVectors.size() != vectorsBefore
        + (old.UOld ? 1u : 0u) + (old.UOO ? 1u : 0u))
        throw std::runtime_error(
            std::string(WHO) + "the change carried a different number of cell fields than went in, so the "
            "old-time levels below would be read from another field's slot.");
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

        // A GATE'S CONTROL, announced every change it is on (see BRAE_CONTROL_PREVCORR_PHICN in
        // inter_driver_cpp.cu). It throws the MAPPED old-time levels away and re-captures them from the
        // fields as they stand after the change -- which is a field of the right SIZE holding the wrong
        // instant, so nothing throws and every ddt term on the changing step is zero. It makes the
        // answer wrong, and it is the control for the mapping half of this unit.
        if (std::getenv("BRAE_CONTROL_AMR_NO_OLDTIME_MAP"))
        {
            std::printf("  *** CONTROL MODE: the old-time levels are re-captured from the new fields, "
                        "not mapped. This run is deliberately wrong. ***\n");
            // rho and its old-old level are left MAPPED: the mixture is recomputed in
            // interAfterMeshChange, after this point, so f.rho is still the old mesh's size here and a
            // control that assigned it would throw instead of answering wrongly. The levels below are
            // enough -- they are the ones ddt(rho,U) and the alpha equation read.
            if (old.alphaOld) *old.alphaOld = f.alpha1.internal;
            if (old.UOld)     *old.UOld     = f.U.internal;
            if (old.UOO)      *old.UOO      = f.U.internal;
            if (old.phiOld)   *old.phiOld   = f.phi;
            if (old.UfOld)    *old.UfOld    = f.Uf;
            const auto patchValues = [&]()
            {
                std::vector<std::vector<vector>> out(f.U.boundary.size());
                for (std::size_t pi = 0; pi < f.U.boundary.size(); ++pi) out[pi] = f.U.boundary[pi]->value();
                return out;
            };
            if (old.UOldBnd) *old.UOldBnd = patchValues();
            if (old.UOOBnd)  *old.UOOBnd  = patchValues();
        }
    }

    if (resizeNotMap)
    {
        const std::size_t nNew = static_cast<std::size_t>(amr.state.m.nCells());
        preAlpha.resize(nNew, scalar(0));
        prePrgh.resize(nNew, scalar(0));
        preU.resize(nNew, vector{0, 0, 0});
        f.alpha1.internal = preAlpha;
        f.p_rgh.internal = prePrgh;
        f.U.internal = preU;
    }

    // ...and the mesh the caller's fields reference. The patches are assigned IN PLACE by the driver, so
    // `*mm.patches` is already the new one; the mesh and geometry are copied into the caller's objects for
    // the same reason -- every FvPatch, every patch field and every operator reads those.
    *mm.m = amr.state.m;
    mm.g->build(*mm.m);
    // ELEMENT BY ELEMENT, and the count checked: a whole-vector assignment of a DIFFERENT size
    // reallocates, and every patch field in the solver holds a `const FvPatch&` into this buffer. It
    // happens to be safe at equal sizes -- libstdc++ assigns in place then -- and that is exactly the
    // kind of safety that stops being true silently. The same check lives inside the driver's own
    // updatePatchesInPlace; this is the site the SOLVER's fields reference.
    if (amr.state.patches.size() != mm.patches->size())
        throw std::runtime_error(
            std::string(WHO) + "the change left " + std::to_string(amr.state.patches.size())
            + " patches where the solver's fields reference " + std::to_string(mm.patches->size())
            + ". hexRef8 splits faces WITHIN a patch and never adds or removes one; a change that did "
            "would dangle every patch field's reference.");
    for (std::size_t pi = 0; pi < mm.patches->size(); ++pi)
    {
        (*mm.patches)[pi] = amr.state.patches[pi];
    }

    // WHAT interFoam REBUILDS AFTER A CHANGE (interFoam.C:139-147), and it is recomputed rather than
    // mapped because every one of these is a function of the new mesh or of alpha1:
    //   gh and ghf off the new cell and face centres, p from them, and the mixture -- alpha2, rho, mu, nu
    //   and the curvature -- from the mapped alpha1.
    // The driver's own post-change stage does that work; this returns `true` so it runs.
    return true;
}

void interAfterMeshChange(
    InterFields&              f,
    const MutableMesh&        mm,
    GamgAgglomerationCache&   gamgCache,
    const CorrectPhiControls& cpc,
    RunReport&                rep)
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

    // THE FLUX IS REBUILT AND CORRECTED, which is the other half of interFoam.C's mesh-changed block
    // (:134-142) and the half that was missing. A split face inherits its PARENT'S WHOLE FLUX through a
    // quarter of the area -- Field::map copies, it does not divide -- so the mapped flux is not
    // divergence-free, and interFoam's alpha equation has no div(phi) compensation to absorb that.
    //
    // MEASURED, from OpenFOAM's own written fields of this very case: the mapped flux carries
    // max|div(phi)|/V of about 5.0e+02 per second -- 0.5 per 1 ms step, in the newly refined interface
    // cells -- against 2.6e-09 (sum local) after OpenFOAM's own pcorr solve, which its log spends 8 GAMG
    // iterations on at step 2. That is what took max(alpha.water) to 1.0021 where OpenFOAM stays at 1
    // exactly. Rebuilding phi from Sf & Uf alone does not close it either (about 1.3e+02 per second); the
    // pcorr solve is the rest.
    //
    // `meshPhi` is NULL: a refining mesh does not MOVE, so fvc::makeRelative and makeAbsolute are
    // no-ops (fvcMeshPhi.C guards both on mesh.moving()) and there is no mesh flux to make the flux
    // relative to. `meshChanging` is TRUE, which is what runs correctUphiBCs inside CorrectPhi -- the
    // re-evaluation of every U patch that FIXES a value, and phi_b = U_b & Sf_b on those patches.
    // ...and the control that leaves the MAPPED flux in place, which is the defect this half fixes
    const bool skipCorrectPhi = (std::getenv("BRAE_CONTROL_AMR_NO_CORRECTPHI") != nullptr);
    if (skipCorrectPhi)
        std::printf("  *** CONTROL MODE: the flux is left as the mapper wrote it -- no Sf & Uf rebuild "
                    "and no pcorr solve. This run is deliberately wrong. ***\n");
    if (f.correctPhi && !skipCorrectPhi)
    {
        if (f.Uf.internal.size() != static_cast<std::size_t>(m.nInternalFaces()))
            throw std::runtime_error(
                std::string(WHO) + "correctPhi is on and Uf has " + std::to_string(f.Uf.internal.size())
                + " internal faces where the new mesh has " + std::to_string(m.nInternalFaces())
                + ". The flux rebuild reads Uf, so a missing or unmapped one would set phi to zero.");
        for (label face = 0; face < m.nInternalFaces(); ++face)
        {
            f.phi.internal[static_cast<std::size_t>(face)] =
                dot(g.Sf()[static_cast<std::size_t>(face)], f.Uf.internal[static_cast<std::size_t>(face)]);
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            if (q.type == "empty") continue;
            for (label i = 0; i < q.size; ++i)
            {
                f.phi.boundary[pi][static_cast<std::size_t>(i)] =
                    dot(g.Sf()[static_cast<std::size_t>(q.start + i)],
                        f.Uf.boundary[pi][static_cast<std::size_t>(i)]);
            }
        }
        const SurfaceScalarField rAUf = fvc::interpolate(f.rAU, m, g, patches);
        CorrectPhiInput cin;
        cin.rAUf = &rAUf;
        cin.meshChanging = true;
        cin.meshPhi = nullptr;
        cin.rhoPhi = &f.rhoPhi;
        cin.solveLog = &rep.pcorrSolves;
        correctPhi(f.U, f.phi, f.p_rgh, cin, cpc, m, g, patches);
        pushFluxToPatches(f, patches);
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
