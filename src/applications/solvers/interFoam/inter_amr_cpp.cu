// interFoam on an adaptive mesh. See inter_amr_cpp.cuh.
#include "inter_amr_cpp.cuh"
#include "inter_phase_time.cuh"
#include "foam_dict.cuh"
#include "mesh_cell_cells_cpp.cuh"
#include "primitive_patch_cpp.cuh"
#include "mrf_read.cuh"   // readCellZones: the only place brae holds a mesh's zones
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

constexpr const char* WHO = "brae interFoam (adaptive mesh): ";

// polyMesh/<kind>Zones as ZoneMesh writes it (ZoneMesh.C:1155-1180): `0()`, or N, `(`, then per zone its
// name and a dictionary of `type`, its members (`cellLabels`/`faceLabels`/`pointLabels`, as
// `List<label> N(...)` in ascii, `List<label> N` in binary) and a face zone's `flipMap`. A missing file is
// an empty zone list -- OpenFOAM writes `0()` for it all the same (oscillatingBox has none in constant/).
std::vector<ZoneEntry> readZoneEntries(const std::string& path)
{
    std::vector<ZoneEntry> out;
    if (!std::filesystem::exists(path) && !std::filesystem::exists(path + ".gz"))
    {
        return out;
    }
    TokenStream ts(path);
    const label n = ts.nextLabel();
    ts.expect("(");
    for (label z = 0; z < n; ++z)
    {
        ZoneEntry e;
        e.name = ts.next();
        ts.expect("{");
        int depth = 1;
        while (!ts.eof() && depth > 0)
        {
            const std::string k = ts.next();
            if (k == "{")
            {
                ++depth;
                continue;
            }
            if (k == "}")
            {
                --depth;
                continue;
            }
            if (depth != 1 || k == ";")
            {
                continue;
            }
            if (k == "type")
            {
                e.type = ts.next();
            }
            else if (k == "cellLabels" || k == "faceLabels" || k == "pointLabels")
            {
                std::string t = ts.next();
                if (t.rfind("List<", 0) == 0)
                {
                    t = ts.next();
                }
                e.nMembers = static_cast<label>(std::stol(t));
            }
            else
            {
                e.extraKeys = true;
            }
            // the rest of the entry, to its `;`
            while (!ts.eof() && ts.peek() != ";" && ts.peek() != "}")
            {
                ts.next();
            }
        }
        out.push_back(e);
    }
    return out;
}

// Instrument: BRAE_STAGE_DUMP_DIR=<dir> (+ BRAE_STAGE_DUMP_ITER=n, default 1) writes the mesh-change
// block's THREE flux stages, one value per line, under the names tools/dumpInterFoam writes OpenFOAM's --
// UfIn, phiAbsPre, phiAbsPost. That is the split this unit needs: the mapped Uf that goes in, the
// `phi = Sf & Uf` rebuild (interFoam.C:134) and the pcorr solve, in the one place where the restart
// profile's difference could be born. The end-of-step fields cannot separate them, because pEqn's
// ddtCorr reads Uf.oldTime() and correctUf writes Uf from U, so every stage feeds the next.
struct AmrStageDump
{
    std::string dir;
    bool on = false;

    void vectors(const char* name, const std::vector<vector>& v) const
    {
        if (!on) return;
        std::ofstream o(dir + "/" + name);
        o.precision(17);
        for (const vector& x : v) o << x.x << " " << x.y << " " << x.z << "\n";
    }

    void scalars(const char* name, const std::vector<scalar>& v) const
    {
        if (!on) return;
        std::ofstream o(dir + "/" + name);
        o.precision(17);
        for (const scalar x : v) o << x << "\n";
    }
};

AmrStageDump openAmrStageDump()
{
    AmrStageDump d;
    const char* dd = std::getenv("BRAE_STAGE_DUMP_DIR");
    if (!dd) return d;
    static int changes = 0;
    const char* it = std::getenv("BRAE_STAGE_DUMP_ITER");
    if (++changes != (it && *it ? std::atoi(it) : 1)) return d;
    std::error_code ec;
    std::filesystem::create_directories(dd, ec);
    d.dir = dd;
    d.on = !ec;
    return d;
}

// A solver carries more than fields. Each of these is state a topology change invalidates and that no
// unit has mapped yet, so it throws and names itself rather than surviving into the next step.
void refuseUnmappedState(const InterFields& f)
{
    // TURBULENCE IS CARRIED NOW -- k, the second scalar and nut through the cell and patch mappers like
    // any other field, their per-step old times as raw cell arrays, and the two WALL DISTANCES recomputed
    // rather than mapped because that is what OpenFOAM does (see updateMeshInterTurbulence). What is
    // refused is the state no fixture holds:
    //
    //   * THE CLOSURE UNDER CrankNicolson. Its ddt0 levels (ddt0K, ddt0Eps) and old-OLD levels (kOO,
    //     epsOO) are registered DDt0Fields that MapGeometricFields autoMaps in OpenFOAM, exactly as the
    //     solver's are -- and the solver's took two units of their own (11 host, 13 device). No adaptive
    //     tutorial runs CrankNicolson on a turbulent case, so this is refused rather than carried blind.
    //   * LES. kEqn's filter WIDTH is a function of the cell volume and is recomputed on a changing mesh
    //     (cubeRootVolDelta.C:128-134), which updateMeshInterTurbulence does -- but the only LES fixture
    //     in the tree is LES/nozzleFlow2D, which is not adaptive, so nothing would hold it.
    if (f.turbulence.on && f.ddtU == DdtScheme::CrankNicolson)
        throw std::runtime_error(
            std::string(WHO) + "the case runs a turbulence model AND ddtSchemes names CrankNicolson. The "
            "closure's own ddt0 levels (ddt0K, ddt0Eps) and old-old levels are registered fields OpenFOAM "
            "autoMaps, and no adaptive tutorial pairs the two, so nothing would hold the carry. The "
            "SOLVER's CrankNicolson state IS carried (InterAmrCn).");
    // ...and a SECOND LINE OF DEFENCE on the state itself rather than on the scheme, because a `ddt(k)`
    // entry could form these levels under a default this port read as Euler.
    //
    // IT TESTS THE ddt0 LEVELS AND NOT kOO/epsOO, and that distinction is measured rather than assumed:
    // advanceTurbulenceOldTime rotates kOO and epsOO at EVERY step under EVERY scheme
    // (inter_turbulence_cpp.cu:902-919), mirroring OpenFOAM's oldTime().oldTime(), and only CrankNicolson
    // READS them. A first cut of this refusal tested them for emptiness and fired on the Euler fixture at
    // its second change, with `kOO 2814` -- a refusal that named CrankNicolson on a case that does not run
    // it. They are ordinary mesh-sized arrays and are carried with the rest.
    if (f.turbulence.on && (f.turbulence.cn.ddt0K.exists || f.turbulence.cn.ddt0Eps.exists))
        throw std::runtime_error(
            std::string(WHO) + "the turbulence closure has formed a CrankNicolson ddt0 level (ddt0K "
            + std::string(f.turbulence.cn.ddt0K.exists ? "exists" : "absent") + ", ddt0Eps "
            + std::string(f.turbulence.cn.ddt0Eps.exists ? "exists" : "absent")
            + "), which a change does not carry.");
    if (f.turbulence.on && f.turbulence.model == InterRasModel::KEqnLES)
        throw std::runtime_error(
            std::string(WHO) + "the case runs the LES kEqn closure. Its filter width IS recomputed on a "
            "changing mesh and this adapter does that, but the only LES fixture in the tree "
            "(LES/nozzleFlow2D) is not adaptive, so no gate would hold it.");
    if (f.waves.any)
        throw std::runtime_error(
            std::string(WHO) + "the case has wave boundary conditions, whose per-patch models hold their "
            "own state and are not carried through a mesh change.");
    // MRF IS CARRIED NOW (interAfterMeshChange calls MRF::update), so the refusal that stood here is
    // gone. What remains is the one thing the carry cannot supply: a zone whose spec was not kept, which
    // would leave the rebuild nothing to rebuild from and is a caller error rather than a case's.
    if (f.mrfZones.size() != f.mrfSpecs.size())
        throw std::runtime_error(
            std::string(WHO) + "the case has " + std::to_string(f.mrfZones.size()) + " MRF zone(s) but "
            + std::to_string(f.mrfSpecs.size()) + " kept spec(s). A change rebuilds each zone's face "
            "lists from its own spec (MRFZone::update, MRFZone.C:598-604) and cannot do it from none.");
    // REFINEMENT AND MOTION IN ONE STEP (laminar/oscillatingBox): points0 is carried through the change
    // (RefineUpdateState::points0), V0 is the change's (DynamicMotionSolverFvMesh::topoChanged), and the
    // move follows the change in OpenFOAM's order. What is NOT carried is refused by name:
    if (f.dynamicMesh)
    {
        // the motion the list factory built -- anything else reaching here is a caller's error
        if (!f.dynamicMesh->listForm())
            throw std::runtime_error(
                std::string(WHO) + "the refining mesh carries a motion that was not built as a "
                "dynamicMotionSolverListFvMesh member (DynamicMotionSolverFvMesh::NewForRefine).");
        //   * CrankNicolson: its moving branch reads V00 and the mesh flux's old-time level, and a change
        //     maps the first (fvMesh.C:896-925) and DROPS the second (fvMesh.C:1056) -- neither is ported
        if (f.ddtU == DdtScheme::CrankNicolson)
            throw std::runtime_error(
                std::string(WHO) + "the refining mesh also moves (a motion solver) under CrankNicolson. The "
                "scheme's moving branch reads V00 and the mesh flux's old-time level; a topology change maps "
                "the first and drops the second, and neither is carried here.");
        //   * moveMeshOuterCorrectors: a second update in the same time index can refine AGAIN, with
        //     storeOldVol and the oldPoints store both skipped and the mixed-time oldPoints of the first
        //     change feeding the mesh flux (polyMeshUpdate.C:67-118)
        if (f.moveMeshOuterCorrectors)
            throw std::runtime_error(
                std::string(WHO) + "the refining mesh also moves (a motion solver) with "
                "moveMeshOuterCorrectors. A second mesh update in one time index can refine again against "
                "the mixed-time old points the first change left, which is not carried here.");
    }
    if (!f.correctPhi)
        throw std::runtime_error(
            std::string(WHO) + "the case sets correctPhi off. interFoam.C:130-142 puts mixture.correct() "
            "INSIDE the correctPhi branch, so with it off OpenFOAM keeps the MAPPED rho, mu, nu, nHatf and "
            "curvature rather than recomputing them -- and brae carries neither the curvature nor the "
            "mixture through a change, so it would recompute what OpenFOAM mapped.");
    // CrankNicolson ITSELF is carried (see InterAmrCn), but a RESTART that SEEDS the levels from disk is
    // not gated: those fields are read on the START mesh with startTimeIndex -2, and whether the levels a
    // change then maps are the ones OpenFOAM would have is a question no fixture here asks.
    //
    // THE TEST IS THE FILES, NOT THE DIRECTORY NAME. `cnRestart.dir` is set to the start directory for
    // EVERY CrankNicolson case -- it is where the seeder looks, not a statement that anything is there --
    // so a test on it being non-empty refused every CN case, restart or not. Measured: it refused the
    // gate's own three-step `cn` profile, which starts from 0 and has no ddt0 field anywhere.
    if (!f.cnRestart.dir.empty())
    {
        std::string present;
        for (const char* nm : {"ddt0(rho,U)", "ddtCorrDdt0(U)", "ddtCorrDdt0(Uf)", "ddtCorrDdt0(phi)"})
        {
            if (std::filesystem::exists(f.cnRestart.dir + "/" + nm)
             || std::filesystem::exists(f.cnRestart.dir + "/" + nm + ".gz"))
            {
                present += (present.empty() ? "" : ", ") + std::string(nm);
            }
        }
        if (!present.empty())
            throw std::runtime_error(
                std::string(WHO) + "the start directory holds the CrankNicolson level(s) " + present
                + ". The scheme's levels are carried through a change, but a RESTART that seeds them from "
                "disk and then refines is gated by nothing, so it is refused rather than run blind.");
    }
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
    const std::string&          facesPolyMeshDir,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    const FvGeometry&           g)
{
    InterAmr amr;
    const std::string path = caseDir + "/constant/dynamicMeshDict";
    if (!std::filesystem::exists(path)) return amr;
    const FoamDict d = readDict(path);
    if (d.wordOr("dynamicFvMesh", "") != "dynamicRefineFvMesh") return amr;

    // REFINEMENT AND MOTION ARE ONE CLASS IN v2412: dynamicRefineFvMesh derives from
    // dynamicMotionSolverListFvMesh (dynamicRefineFvMesh.H:92-95), whose init reads a `solvers`
    // SUB-DICTIONARY and builds one motionSolver per sub-dictionary inside it
    // (dynamicMotionSolverListFvMesh.C:98-127), with `mandatory` false so zero of them is legal
    // (dynamicRefineFvMesh.C:1107). update() runs updateTopology() FIRST and the motion after it
    // (dynamicRefineFvMesh.C:1468-1474). The motion is built by DynamicMotionSolverFvMesh::NewForRefine and
    // carried through each change; the driver runs the change and then the move.
    //
    // IT WAS REFUSED HERE, and before that it was DROPPED: MEASURED on laminar/oscillatingBox, brae refined
    // exactly as OpenFOAM did and reported max|U| 1.2e-04 m/s where OpenFOAM reads 2.73 -- the motion lost
    // in silence behind a refusal that could not fire. What is still refused here is a 2-D mesh:
    // points0MotionSolver::updateMesh ends in twoDCorrectPoints (points0MotionSolver.C:209), which puts an
    // added point back on the mesh's two planes, and it is not ported. oscillatingBox is 3-D.
    {
        const FoamDict* solvers = d.subDict("solvers");
        const bool moves = solvers && !solvers->subs.empty();
        bool twoD = false;
        for (const FvPatch& q : patches)
        {
            twoD = twoD || q.type == "empty" || q.type == "wedge";
        }
        if (moves && twoD)
            throw std::runtime_error(
                std::string(WHO) + "constant/dynamicMeshDict asks for refinement AND a motion solver on a "
                "2-D mesh. points0 is carried through each topology change, but OpenFOAM then corrects an "
                "added point back onto the mesh's planes (twoDCorrectPoints, points0MotionSolver.C:209), "
                "and that is not ported; a 3-D mesh runs (laminar/oscillatingBox).");
    }

    amr.active = true;
    amr.controls = dynamicRefine::readRefineControls(d);

    // A 2-D CASE RUNS, and it did not for a while. Porting autoMap for the `empty` and `noSlip` patch
    // fields let one through, and it agreed with OpenFOAM only while the fields it mapped were trivial:
    // with every solve pinned to 1e-14 relTol 0, the first change that mapped a NON-TRIVIAL state read
    // alpha 5.2e-03 and U 3.7e-01 from OpenFOAM (laminar/damBreak), where the 3-D fixture is exact.
    //
    // THE CAUSE was the hull average, and OpenFOAM's own dynamicRefineFvMesh named it in one run
    // (tools/dumpRefineUpdate on a 2-D mesh, now the third arm of tests/refine_update_vs_openfoam.sh):
    // a refined 2-D mesh has INTERNAL faces in the EMPTY direction -- the mesh is one cell thick, so its
    // only z-normal faces are the empty patch's, and hexRef8 splits each cell into eight, including
    // across z -- and those faces are INJECTED, with no old face to map from. OpenFOAM averages the hull
    // out of a flat array it fills from each patch field in turn, and an emptyFvPatchField is
    // ZERO-SIZED on a patch that has faces, so its faces keep the zero and still count. brae's is sized,
    // and its values were going into the average: braePhi 161.0 where OpenFOAM had -6.8e-30.
    // FluxMeshView::patchHoldsNoValues carries the distinction now, and the 2-D case reads alpha
    // 1.8e-15 and U 4.5e-12 at that same step.

    // the mesh this run starts from, and the state that goes with it. A mesh that has never been refined
    // carries no cellLevel on disk and starts at level 0 everywhere; its history is the identity, which is
    // what refinementHistory's own constructor leaves (see the note in hex_ref8_cpp.cuh).
    amr.state.m = m;
    amr.state.patches = patches;
    // THE REFINEMENT STATE STARTS AT ZERO, which is OpenFOAM's own fallback and not a placeholder:
    // cellLevel and pointLevel are READ_IF_PRESENT with `labelList(n, Zero)` behind them, and a history
    // with no file is still ACTIVE with every cell visible and no parents (hexRef8.C:1908-1990). Every
    // blockMesh case takes exactly this path.
    amr.state.levels.cellLevel.assign(static_cast<std::size_t>(m.nCells()), label(0));
    amr.state.levels.pointLevel.assign(static_cast<std::size_t>(m.nPoints()), label(0));
    amr.state.history = cpu::hexRef8::freshHistory(m.nCells());
    // ...AND THEN OFF DISK, where the mesh carries it. A snappyHexMesh mesh is at levels 1 to 3 and a
    // resumed refined one at whatever it reached, and `cellLevel[celli] < maxRefinement`
    // (dynamicRefineFvMesh.C:861) means something different in the two codes until this is read. THE
    // DIRECTORY IS THE FACES INSTANCE'S, which is where OpenFOAM reads them from (mesh_.facesInstance(),
    // hexRef8.C:1912-1990) and which for a genuine restart is the time directory and not constant/ --
    // cpu::timePaths resolves it, and this used to hard-code constant/polyMesh.
    {
        const bool blind = std::getenv("BRAE_CONTROL_AMR_NO_LEVELS") != nullptr;
        if (blind)
        {
            // A GATE'S CONTROL, and the strongest kind: it restores exactly what brae SHIPPED before this
            // unit -- level 0 everywhere, whatever the files say -- which is also what OpenFOAM does when
            // they are absent. So the control is a real port rather than an invented one.
            std::printf("  *** CONTROL MODE: the mesh's cellLevel, pointLevel and refinementHistory are NOT "
                        "read; every cell is taken as level 0. This run is deliberately wrong. ***\n");
        }
        else if (cpu::hexRef8::readRefinementState(facesPolyMeshDir, m.nCells(), m.nPoints(),
                                                   amr.state.levels, amr.state.history))
        {
            label mx = 0;
            for (const label l : amr.state.levels.cellLevel) mx = (l > mx) ? l : mx;
            // THE SPLIT CELLS ARE THE ENTRIES WITH A PARENT, not the length of the list: `parent` has one
            // entry per cell of the history (refinementHistory.C:392-412 sizes it to nCells and fills it
            // with -1), so printing its SIZE called every cell of a mesh with no history a split cell --
            // motorBike, whose Allrun.pre removes the history, reported "8655 split cell(s)" on 8655 cells.
            std::size_t nSplit = 0;
            for (const label pr : amr.state.history.parent) { if (pr >= 0) ++nSplit; }
            std::printf("  the mesh carries a refinement state: cellLevel up to %ld, %zu split cell(s) in "
                        "its history\n", (long)mx, nSplit);
        }
    }
    const std::vector<std::vector<label>> cells = meshCells(m);
    const std::vector<std::vector<label>> pointCells = pointCellsFromCells(m, cells);
    // THE ZONES THE MESH CARRIES, counted from the polyMesh directory -- the only place they exist, since
    // brae's PrimitiveMesh does not hold them. It makes changeMesh's own refusal reachable: resetZones
    // renumbers all three kinds through a change and is not ported, so a case with any of them stops by
    // name instead of running with the zone still in the old numbering (which an MRF zone or an
    // fvOption's cellZone would then apply itself to).
    {
        const std::string pm = facesPolyMeshDir + "/";
        // cellZones ARE CARRIED (dynamicRefine::renumberCellZones), so they are NOT counted here.
        label nZ = 0;
        // ...the other two kinds by their MEMBERS, and not by the file being there NOR by the number of
        // zone ENTRIES. Two narrowings, each measured on a real mesh:
        //
        //   * subsetMesh writes cellZones, faceZones and pointZones for every mesh it makes, each holding
        //     `0()`, so a test on the file's EXISTENCE counted two zones on a case that has none.
        //   * snappyHexMesh writes a pointZone `frozenPoints` with `pointLabels List<label> 0` -- an entry
        //     with NO POINTS IN IT -- so a test on the ENTRY COUNT refused motorBike, whose mesh is exactly
        //     that. An empty zone has nothing to renumber: resetZones sizes the new addressing from the
        //     zone's own membership (polyTopoChange.C:1612-1700), so a zone with no members comes out of the
        //     change empty, with its name and type, whatever the map says. What the refusal is FOR is a zone
        //     that HAS members, whose labels the change must renumber and whose faceZone flip map goes with
        //     them -- so that is what is counted.
        for (const char* other : {"faceZones", "pointZones"})
        {
            // plain or .gz, as readZoneEntries below reads them
            if (!std::filesystem::exists(pm + other) && !std::filesystem::exists(pm + other + ".gz"))
            {
                continue;
            }
            const std::vector<char> raw = gzSlurp(pm + other);
            std::string text(raw.begin(), raw.end());
            // Each zone entry carries its members under a `...Labels` key, whose count is the next integer
            // (`pointLabels List<label> 0;` as well as `faceLabels 3(1 2 3);`). A faceZone's `flipMap` has a
            // count of its own and is NOT one of these keys, so it is not counted twice.
            std::size_t at = 0;
            while ((at = text.find("Labels", at)) != std::string::npos)
            {
                at += 6;
                std::size_t i = at;
                while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
                std::size_t j = i;
                while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
                if (j > i && std::stol(text.substr(i, j - i)) > 0) ++nZ;
            }
        }
        amr.state.nZones = nZ;
        amr.cellZoneEntries = readZoneEntries(pm + "cellZones");
        amr.faceZoneEntries = readZoneEntries(pm + "faceZones");
        amr.pointZoneEntries = readZoneEntries(pm + "pointZones");
    }
    amr.level0Edge = cpu::hexRef8::readLevel0Edge(facesPolyMeshDir, m, amr.state.levels.cellLevel);
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
    const InterAmrOldTime& old,
    const InterAmrCn&      cn)
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
    // BRAE_INTER_PHASE_TIME: this stage part by part
    std::optional<interPhase::Nested> amrPart;
    amrPart.emplace("amr: the carried state gathered (a copy of every field, every step)");
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
    // THE TURBULENCE FIELDS, whole -- cells AND patch fields, so every wall function's own value list
    // goes through the patch mapper with it. They are pushed here rather than beside alpha1 so the
    // read-back indices of the three solver fields do not move.
    const bool carryTurb = f.turbulence.on;
    if (carryTurb)
    {
        amr.state.carriedScalarFields.push_back(&f.turbulence.k);
        amr.state.carriedScalarFields.push_back(
            f.turbulence.model == InterRasModel::KOmegaSST ? &f.turbulence.omega
                                                           : &f.turbulence.epsilon);
        amr.state.carriedScalarFields.push_back(&f.turbulence.nut);
    }

    const std::size_t scalarsBefore = amr.state.cellScalars.size();
    const std::size_t vectorsBefore = amr.state.cellVectors.size();
    if (old.alphaOld) amr.state.cellScalars.push_back(*old.alphaOld);
    if (old.rhoOld)   amr.state.cellScalars.push_back(*old.rhoOld);
    if (old.rhoOO)    amr.state.cellScalars.push_back(*old.rhoOO);
    if (old.alphaOO)  amr.state.cellScalars.push_back(*old.alphaOO);
    // THE CLOSURE'S PER-STEP OLD TIMES. brae keeps these per TIME INDEX rather than per call
    // (InterTurbulence::kOldStep), which is what makes them state a change has to carry: OpenFOAM's are
    // the registered fields' own oldTime() levels, autoMapped with the fields. Pushed only when they have
    // been captured -- at the FIRST change of a run no closure correct() has run and they are empty.
    const bool carryKOld = carryTurb && !f.turbulence.kOldStep.empty();
    const bool carrySecondOld = carryTurb && !f.turbulence.epsOldStep.empty();
    // ...and the old-OLD levels, which rotate at every step under every scheme and are read only by
    // CrankNicolson. Carried anyway: they are mesh-sized, and leaving them at the old count would be a
    // buffer of the wrong length waiting for the one scheme that reads it.
    const bool carryKOO = carryTurb && !f.turbulence.cn.kOO.empty();
    const bool carrySecondOO = carryTurb && !f.turbulence.cn.epsOO.empty();
    if (carryKOld)      amr.state.cellScalars.push_back(f.turbulence.kOldStep);
    if (carrySecondOld) amr.state.cellScalars.push_back(f.turbulence.epsOldStep);
    if (carryKOO)       amr.state.cellScalars.push_back(f.turbulence.cn.kOO);
    if (carrySecondOO)  amr.state.cellScalars.push_back(f.turbulence.cn.epsOO);
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

    // THE CrankNicolson LEVELS, each through the mapper its own type uses. They are REGISTERED fields in
    // OpenFOAM and are autoMapped like any other; the cell ones carry a patch list beside them, which
    // rides through the patch mapping as U's old-time patch values do. startTimeIndex and timeIndex are
    // the FIELD's state in OpenFOAM too (DDt0Field's only extra member), so they are not touched here --
    // re-seeding them would restart the scheme as Euler at every change, which is what this unit's
    // control does on purpose.
    const std::size_t cnVecBefore = amr.state.cellVectors.size();
    const std::size_t cnSurfVecBefore = amr.state.surfaceVectors.size();
    const std::size_t cnSurfBefore = amr.state.surfaceScalars.size();
    // A ddt0 LEVEL'S BOUNDARY IS OPTIONAL, and that is brae's own shape rather than a shortcut: the
    // fvmDdt path creates its level with NO patch lists (crank_nicolson_ddt_scheme_cpp.cu:91 -- the
    // matrix reads cell values only), while the two ddtCorr paths create theirs with one list per patch
    // (:257, :406-407). So each level says for itself whether a boundary rides with it; a count that is
    // neither zero nor the mesh's is a defect in the caller and throws.
    const auto ddt0Formed = [&](const fv::CrankNicolsonDdt0<vector>* d, const char* what) -> bool
    {
        if (!d || !d->exists) return false;
        if (!d->boundary.empty() && d->boundary.size() != mm.patches->size())
            throw std::runtime_error(
                std::string(WHO) + "the CrankNicolson level " + what + " has "
                + std::to_string(d->boundary.size()) + " patch lists where the mesh has "
                + std::to_string(mm.patches->size()) + ". Half a field cannot be mapped.");
        return true;
    };
    const bool cnRhoU = ddt0Formed(cn.ddt0RhoU, "ddt0(rho,U)");
    const bool cnCorrU = ddt0Formed(cn.ddtCorrU, "ddtCorrDdt0(U)");
    if (cnRhoU) amr.state.cellVectors.push_back(cn.ddt0RhoU->internal);
    if (cnCorrU) amr.state.cellVectors.push_back(cn.ddtCorrU->internal);
    const bool cnRhoUBnd = cnRhoU && !cn.ddt0RhoU->boundary.empty();
    const bool cnCorrUBnd = cnCorrU && !cn.ddtCorrU->boundary.empty();
    const std::size_t cnBndBefore = amr.state.surfaceVectors.size();
    if (cnRhoUBnd) pushBoundaryOnly(cn.ddt0RhoU->boundary);
    if (cnCorrUBnd) pushBoundaryOnly(cn.ddtCorrU->boundary);
    // ...and the two surface VECTORS: the Uf ddt0 level and Uf's old-old level. Neither is oriented -- a
    // face velocity is not a flux.
    const bool cnCorrUf = ddt0Formed(cn.ddtCorrUf, "ddtCorrDdt0(Uf)")
                      && !cn.ddtCorrUf->internal.empty();
    const std::size_t cnUfDdt0Index = amr.state.surfaceVectors.size();
    if (cnCorrUf)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceVectorField c;
        c.field = cn.ddtCorrUf->internal;
        c.bnd = cn.ddtCorrUf->boundary;
        c.oriented = false;
        amr.state.surfaceVectors.push_back(std::move(c));
    }
    const bool cnUfOO = cn.UfOO && !cn.UfOO->internal.empty()
                    && cn.UfOO->boundary.size() == mm.patches->size();
    const std::size_t cnUfOOIndex = amr.state.surfaceVectors.size();
    if (cnUfOO) pushSurfaceVector(*cn.UfOO);
    // ...and the ORIENTED fluxes: phi's old-old level and the alpha flux's two blend levels.
    // A HALF-FORMED FIELD IS NOT CARRIED, and it is not guessed at either: the mapper walks every patch
    // of the new mesh and reads the field's own list for it, so a field with an internal half and no
    // boundary lists runs off the end of that vector (a SIGSEGV in mapCarriedFields, which is how this
    // was found). A level the driver has not built yet has BOTH halves empty; anything in between is a
    // defect in the caller and says so.
    const auto fluxFormed = [&](const SurfaceScalarField* fl, const char* what) -> bool
    {
        if (!fl || (fl->internal.empty() && fl->boundary.empty())) return false;
        if (fl->internal.empty() || fl->boundary.size() != mm.patches->size())
            throw std::runtime_error(
                std::string(WHO) + "the CrankNicolson flux " + what + " has "
                + std::to_string(fl->internal.size()) + " internal faces and "
                + std::to_string(fl->boundary.size()) + " patch lists where the mesh has "
                + std::to_string(mm.patches->size()) + ". Half a field cannot be mapped.");
        return true;
    };
    const bool cnPhiOO = fluxFormed(cn.phiOO, "phi.oldTime().oldTime()");
    const bool cnAPhiEnd = fluxFormed(cn.alphaPhiEnd, "alphaPhi10 at the end of the step");
    const bool cnAPhiOld = fluxFormed(cn.alphaPhiOld, "alphaPhi10.oldTime()");
    const auto pushFlux = [&](const SurfaceScalarField& fl)
    {
        dynamicRefine::RefineUpdateState::CarriedSurfaceField c;
        c.field = fl.internal;
        c.bnd = fl.boundary;
        c.oriented = true;
        amr.state.surfaceScalars.push_back(std::move(c));
        amr.state.surfaceScalarVelocity.push_back(std::string());
    };
    if (cnPhiOO) pushFlux(*cn.phiOO);
    if (cnAPhiEnd) pushFlux(*cn.alphaPhiEnd);
    if (cnAPhiOld) pushFlux(*cn.alphaPhiOld);
    // ddtCorrDdt0(phi) is never created on this path: ddtCorr(U, phi, Uf) takes the Uf branch on a
    // dynamic mesh (fvcDdt.C:219). If it ever exists here, the routing has changed and mapping nothing
    // would be silent.
    if (cn.ddtCorrPhi && cn.ddtCorrPhi->exists)
        throw std::runtime_error(
            std::string(WHO) + "the CrankNicolson level `" + cn.ddtCorrPhi->name + "` exists on a refining "
            "mesh. ddtCorr(U, phi, Uf) takes the Uf branch when the mesh is dynamic, so this level should "
            "never have been created -- and nothing here maps it.");

    amrPart.emplace("amr: refineUpdate, whole");
    const dynamicRefine::RefineUpdateStep step =
        dynamicRefine::refineUpdate(amr.state, amr.controls, f.alpha1.internal, amr.startTimeIndex + timeIndex);
    amr.nRefined = static_cast<label>(step.cellsToRefine.size());
    amr.nUnrefined = static_cast<label>(step.pointsToUnrefine.size());
    if (!step.hasChanged) return false;
    amrPart.emplace("amr: the mapped fields handed back to the solver");
    amr.topoChanged = true;

    {
        const std::size_t want = 3u + (carryPhiOld ? 1u : 0u) + (carryAlpha2Bnd ? 1u : 0u)
                               + (cnPhiOO ? 1u : 0u) + (cnAPhiEnd ? 1u : 0u) + (cnAPhiOld ? 1u : 0u);
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
          + (cnRhoUBnd ? 1u : 0u) + (cnCorrUBnd ? 1u : 0u)
          + (cnCorrUf ? 1u : 0u) + (cnUfOO ? 1u : 0u)
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
        + (old.alphaOO ? 1u : 0u) + (carryKOld ? 1u : 0u) + (carrySecondOld ? 1u : 0u)
        + (carryKOO ? 1u : 0u) + (carrySecondOO ? 1u : 0u)
     || amr.state.cellVectors.size() != vectorsBefore
        + (old.UOld ? 1u : 0u) + (old.UOO ? 1u : 0u)
        + (cnRhoU ? 1u : 0u) + (cnCorrU ? 1u : 0u))
        throw std::runtime_error(
            std::string(WHO) + "the change carried a different number of cell fields than went in, so the "
            "old-time levels below would be read from another field's slot.");
    {
        std::size_t si = scalarsBefore;
        std::size_t vi = vectorsBefore;
        if (old.alphaOld) *old.alphaOld = amr.state.cellScalars.at(si++);
        if (old.rhoOld)   *old.rhoOld   = amr.state.cellScalars.at(si++);
        if (old.rhoOO)    *old.rhoOO    = amr.state.cellScalars.at(si++);
        if (old.alphaOO)  *old.alphaOO  = amr.state.cellScalars.at(si++);
        // ...the closure's per-step old times, out of the slots they went into
        if (carryKOld)      f.turbulence.kOldStep   = amr.state.cellScalars.at(si++);
        if (carrySecondOld) f.turbulence.epsOldStep = amr.state.cellScalars.at(si++);
        if (carryKOO)       f.turbulence.cn.kOO     = amr.state.cellScalars.at(si++);
        if (carrySecondOO)  f.turbulence.cn.epsOO   = amr.state.cellScalars.at(si++);
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

        // ...and the CrankNicolson levels, each out of the slot it went into. `exists`,
        // `startTimeIndex` and `timeIndex` are NOT touched: they are the field's own state in OpenFOAM
        // too, and re-seeding them restarts the scheme.
        std::size_t cvi = cnVecBefore;
        if (cnRhoU)  cn.ddt0RhoU->internal = amr.state.cellVectors.at(cvi++);
        if (cnCorrU) cn.ddtCorrU->internal = amr.state.cellVectors.at(cvi++);
        std::size_t cbi = cnBndBefore;
        if (cnRhoUBnd)  cn.ddt0RhoU->boundary = amr.state.surfaceVectors.at(cbi++).bnd;
        if (cnCorrUBnd) cn.ddtCorrU->boundary = amr.state.surfaceVectors.at(cbi++).bnd;
        if (cnCorrUf)
        {
            cn.ddtCorrUf->internal = amr.state.surfaceVectors.at(cnUfDdt0Index).field;
            cn.ddtCorrUf->boundary = amr.state.surfaceVectors.at(cnUfDdt0Index).bnd;
        }
        if (cnUfOO)
        {
            cn.UfOO->internal = amr.state.surfaceVectors.at(cnUfOOIndex).field;
            cn.UfOO->boundary = amr.state.surfaceVectors.at(cnUfOOIndex).bnd;
        }
        std::size_t csi = cnSurfBefore;
        const auto takeFlux = [&](SurfaceScalarField& fl)
        {
            fl.internal = amr.state.surfaceScalars.at(csi).field;
            fl.boundary = amr.state.surfaceScalars.at(csi).bnd;
            ++csi;
        };
        if (cnPhiOO)   takeFlux(*cn.phiOO);
        if (cnAPhiEnd) takeFlux(*cn.alphaPhiEnd);
        if (cnAPhiOld) takeFlux(*cn.alphaPhiOld);

        // A GATE'S CONTROL: throw the mapped CrankNicolson state away and let the scheme re-create each
        // level at the new size. That is the plausible wrong port -- every field is the right SIZE,
        // nothing throws, the run completes, and the scheme silently RESTARTS (a level created at step k
        // is zero for the whole of step k, and its startTimeIndex becomes k, so coef and coef0 fall back
        // to 1 and the step is Euler).
        if (std::getenv("BRAE_CONTROL_AMR_NO_CN_MAP"))
        {
            std::printf("  *** CONTROL MODE: the CrankNicolson levels are dropped and re-created at the new "
                        "size, not mapped. This run is deliberately wrong. ***\n");
            if (cn.ddt0RhoU)  cn.ddt0RhoU->exists = false;
            if (cn.ddtCorrU)  cn.ddtCorrU->exists = false;
            if (cn.ddtCorrUf) cn.ddtCorrUf->exists = false;
            if (cnUfOO)    *cn.UfOO = f.Uf;
            if (cnPhiOO)   *cn.phiOO = f.phi;
            if (cnAPhiEnd) cn.alphaPhiEnd->internal.assign(f.phi.internal.size(), scalar(0));
            if (cnAPhiOld) cn.alphaPhiOld->internal.assign(f.phi.internal.size(), scalar(0));
        }

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
            if (old.alphaOO)  *old.alphaOO  = f.alpha1.internal;
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

    // TWO GATE CONTROLS FOR THE PRESSURE REFERENCE, and the first one MEASURES ITS OWN VACUITY.
    //
    // BRAE_CONTROL_PREF_RENUMBER follows the reference cell through the change's own reverseCellMap,
    // which is the plausible wrong thing and what a refusal calling the index "stale" implies is needed.
    // UNDER PURE REFINEMENT IT IS THE IDENTITY: hexRef8 MODIFIES the parent cell in place and ADDS the
    // other seven children (hexRef8.C's setRefinement), so every retained cell keeps its own index and
    // reverseCellMap[c] == c. Measured on the closed damBreakWithObstacle profile: 16400 -> 16400, and
    // the run reads p 1.9964e-14 relative -- the gate's own number. So on a refine-only fixture this
    // control cannot witness anything, which is also WHY the unit is a range check and nothing more.
    //
    // BRAE_CONTROL_PREF_ANOTHER_CELL is the one that discriminates: it pins the LAST child of the old
    // reference cell instead of the master. That is what any renumbering which did not reproduce
    // hexRef8's master-in-place numbering would land on, and it proves the gate sees WHICH cell is
    // pinned -- pEqn adds (pRefValue - p[pRefCell]) to the whole field, so the two choices differ by the
    // difference between those two cells' pressures.
    if (f.pRef.needReference && f.pRef.pRefCell >= 0)
    {
        const cpu::polyTopoChange::TopoChangeMap& map =
            step.refined ? step.refineMap : step.unrefineMap;
        if (std::getenv("BRAE_CONTROL_PREF_RENUMBER")
            && f.pRef.pRefCell < static_cast<label>(map.reverseCellMap.size()))
        {
            const label moved = map.reverseCellMap[static_cast<std::size_t>(f.pRef.pRefCell)];
            // -master-2 where merged, -1 where removed: the master cell either way
            const label newCell = (moved >= 0) ? moved : (moved < -1 ? -moved - 2 : f.pRef.pRefCell);
            std::printf("  *** CONTROL MODE: the pressure reference is renumbered %ld -> %ld through the "
                        "cell map. OpenFOAM keeps the index. This run is deliberately wrong. ***\n",
                        (long)f.pRef.pRefCell, (long)newCell);
            f.pRef.pRefCell = newCell;
        }
        // ...and the one that proves the gate SEES which cell is pinned: the last cell of the new mesh,
        // which is a child the change added. A child of the reference cell's OWN parent would be the
        // narrower control, and there is none: the reference point of this fixture is in the AIR, far
        // from the interface, so its cell is never selected for refinement -- measured, the child search
        // returned the cell itself. pEqn adds (pRefValue - p[pRefCell]) to the whole field, so any other
        // cell is a whole-field offset.
        if (std::getenv("BRAE_CONTROL_PREF_ANOTHER_CELL"))
        {
            const label last = amr.state.m.nCells() - 1;
            std::printf("  *** CONTROL MODE: the pressure reference is moved %ld -> %ld, a cell the change "
                        "added. This run is deliberately wrong. ***\n",
                        (long)f.pRef.pRefCell, (long)last);
            f.pRef.pRefCell = last;
        }
    }

    // THE RENUMBERED ZONES BACK TO THE CALLER, before anything resolves a selection against them.
    f.cellZones = amr.state.cellZones;

    // ...and the mesh the caller's fields reference. The patches are assigned IN PLACE by the driver, so
    // `*mm.patches` is already the new one; the mesh and geometry are copied into the caller's objects for
    // the same reason -- every FvPatch, every patch field and every operator reads those.
    amrPart.emplace("amr: the solver's mesh copied");
    *mm.m = amr.state.m;
    // THE GEOMETRY IS THE ONE THE REFINEMENT HAS JUST BUILT. Its kept addressing holds an FvGeometry of this
    // same mesh, built after the step's last change (buildAddressing), and the solver's was then built AGAIN
    // from the same points and faces. MEASURED on damBreakWithObstacle, 2026-10-05: 6.6 ms a call for the
    // refinement's and as much here. Copied where the kept one is of this mesh (compared by content), built
    // otherwise -- the same function of the same mesh, so the same bits.
    //   BRAE_CONTROL_AMR_GEOMETRY_BUILT=1   built here, as before
    //   BRAE_CONTROL_AMR_GEOMETRY_CHECK=1   built as well, and the nine arrays compared with the copy, bitwise
    //   BRAE_CONTROL_AMR_GEOMETRY_WRONG=1   a gate's CONTROL, deliberately wrong: the copy is of the mesh with
    //                                       its first point moved, so the cells on that point are another shape
    static const bool geometryBuilt = std::getenv("BRAE_CONTROL_AMR_GEOMETRY_BUILT") != nullptr;
    static const bool geometryCheck = std::getenv("BRAE_CONTROL_AMR_GEOMETRY_CHECK") != nullptr;
    static const bool geometryWrong = std::getenv("BRAE_CONTROL_AMR_GEOMETRY_WRONG") != nullptr;
    const FvGeometry* keptGeometry = geometryBuilt ? nullptr : dynamicRefine::keptStepGeometry(amr.state, *mm.m);
    if (keptGeometry)
    {
        amrPart.emplace("amr: the solver's geometry copied from the refinement's");
        static bool said = false;
        if (!said)
        {
            said = true;
            std::printf("  refinement: the solver's geometry after a change is the refinement's own, copied; "
                        "BRAE_CONTROL_AMR_GEOMETRY_BUILT=1 builds it again\n");
            if (geometryWrong)
            {
                std::printf("  *** CONTROL MODE: the geometry copied is of the mesh with its first point moved. "
                            "This run is deliberately wrong. ***\n");
            }
        }
        if (geometryWrong)
        {
            PrimitiveMesh moved = *mm.m;
            std::vector<vector> pts = moved.points();
            pts[0] = pts[0] + vector{1.0e-3, 0, 0};
            moved.movePoints(pts);
            FvGeometry other;
            other.build(moved);
            mm.g->copyFrom(other, *mm.m);
        }
        else
        {
            mm.g->copyFrom(*keptGeometry, *mm.m);
        }
        if (geometryCheck)
        {
            FvGeometry fresh;
            fresh.build(*mm.m);
            const auto differs = [](
                const auto& x,
                const auto& y)
            {
                return x.size() != y.size()
                    || (!x.empty() && std::memcmp(x.data(), y.data(), x.size()*sizeof(x[0])) != 0);
            };
            const char* what = differs(fresh.Cf(), mm.g->Cf()) ? "the face centres"
                             : differs(fresh.Sf(), mm.g->Sf()) ? "the face areas"
                             : differs(fresh.magSf(), mm.g->magSf()) ? "the face area magnitudes"
                             : differs(fresh.C(), mm.g->C()) ? "the cell centres"
                             : differs(fresh.V(), mm.g->V()) ? "the cell volumes"
                             : differs(fresh.weights(), mm.g->weights()) ? "the interpolation weights"
                             : differs(fresh.deltaCoeffs(), mm.g->deltaCoeffs()) ? "the delta coefficients"
                             : differs(fresh.nonOrthDeltaCoeffs(), mm.g->nonOrthDeltaCoeffs())
                             ? "the non-orthogonal delta coefficients"
                             : differs(fresh.nonOrthCorrectionVectors(), mm.g->nonOrthCorrectionVectors())
                             ? "the non-orthogonal correction vectors" : nullptr;
            if (what)
            {
                throw std::runtime_error(
                    std::string(WHO) + "BRAE_CONTROL_AMR_GEOMETRY_CHECK: of the geometry copied from the "
                    "refinement's, " + what + " are not what building it from the solver's mesh gives.");
            }
        }
    }
    else
    {
        amrPart.emplace("amr: the solver's geometry built (FvGeometry::build)");
        mm.g->build(*mm.m);
    }
    amrPart.emplace("amr: the solver's patches assigned");
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
    RunReport&                rep,
    label                     timeIndex,
    bool                      motionFollows,
    const PatchWaveRunner*    waveRunner)
{
    // the mesh's own sets, for the one selection mode that re-reads a file
    const std::string polyMeshDir = f.amr ? f.amr->polyMeshDir : std::string();
    const PrimitiveMesh& m = *mm.m;
    const FvGeometry& g = *mm.g;
    const std::vector<FvPatch>& patches = *mm.patches;
    const std::size_t nC = static_cast<std::size_t>(m.nCells());

    // THE AGGLOMERATION WAS BUILT FOR THE OLD MESH, and it is UN-BUILT rather than replaced. The whole-
    // object reset zeroed two members that must survive:
    //
    //   `buildCount` is MONOTONE ON PURPOSE. It is what a copy of the hierarchy elsewhere -- the device's
    //   upload -- keys its validity on, precisely because `built` cannot say: the host's own pcorr GAMG
    //   solve a few lines below rebuilds the hierarchy and leaves it built, so a device upload keyed on
    //   `built` alone is kept across the change. Zeroing the count made the rebuilt hierarchy's count 1
    //   again, which is what the stale upload was stamped with -- the same defect
    //   gamg_solver_cpp.cuh:97-107 records, where it showed up as heap corruption in a coarsest-level
    //   solve. damBreakWithObstacle solves BOTH pcorr and p_rgh with GAMG, so it is on that path.
    //
    //   `forward` is pairGAMGAgglomeration's STATIC pairing direction (pair_gamg_agglomeration_cpp.cu),
    //   which in OpenFOAM outlives every object and every mesh. Restoring it to true at each change is a
    //   substitution, not a reset.
    //
    // This is the same line dynamic_motion_solver_fv_mesh_cpp.cu:382 uses after a move, and for the same
    // reason: OpenFOAM's GAMGAgglomeration::movePoints sets requireUpdate_ and the next New() rebuilds
    // from wherever the static direction was left.
    gamgCache.built = false;

    // gh and ghf off the NEW centres (interFoam.C:130-131), assigned rather than written into
    // BRAE_INTER_PHASE_TIME: this stage part by part
    std::optional<interPhase::Nested> afterPart;
    afterPart.emplace("after: gh and ghf");
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

    // THE fvOPTIONS ARE RE-SELECTED, not renumbered, and that is OpenFOAM's own rule rather than a
    // convenience: cellSetOption::isActive() re-runs setCellSelection() whenever the mesh is topoChanging
    // (cellSetOption.C:383-396) and forces the volume to be printed again. A cellZone selection therefore
    // resolves against the LIVE zone -- which the change has just renumbered, so a split zone cell's seven
    // children are IN -- where keeping the labels would leave the source on an eighth of the zone it names.
    //
    // It sits here because OpenFOAM's refresh is LAZY, at the first fvOptions(...) of the step after the
    // change (fvOptionListTemplates.C:43-80 calls isActive() immediately before addSup), and every point
    // between the change and UEqn is arithmetically the same. That makes the placement a choice; it is
    // stated rather than left to be read off.
    afterPart.emplace("after: fvOptions and MRF zones re-selected");
    if (!f.fvOptions.empty())
    {
        // A GATE'S CONTROL: keep the labels each selection gave last time instead of resolving it again.
        // That is the plausible wrong port -- nothing throws, every index is in range, and the source
        // lands on an EIGHTH of the zone it names wherever the zone was refined. OpenFOAM's own log says
        // so in one line per change: `- selected 7398 cell(s) with volume 0.08349609375` where keeping
        // the labels leaves 2736 cells and a volume eight times smaller on the refined part.
        if (std::getenv("BRAE_CONTROL_AMR_NO_RESELECT"))
        {
            std::printf("  *** CONTROL MODE: the fvOptions selections are KEPT, not resolved again on the "
                        "new mesh. This run is deliberately wrong. ***\n");
        }
        else
        {
            fvOptions::reselect(f.fvOptions, f.cellZones, polyMeshDir);
        }
    }

    // THE MRF ZONES ARE REBUILT, for the same reason and from the same live cellZones. OpenFOAM's
    // MRFZoneList::update() is guarded on mesh.topoChanging() and calls each zone's update(), which calls
    // setMRFFaces() AND NOTHING ELSE (MRFZoneList.C:441-450, MRFZone.C:598-604) -- so the dictionary is
    // not re-read, omega is not re-evaluated and cellZoneID_ is not re-found. Everything buildZone
    // computes past the spec's three scalars IS setMRFFaces, so a rebuild from the kept spec against the
    // renumbered zone is that function, exactly.
    //
    // IT HAS TO SIT AFTER THE PATCHES ARE ASSIGNED IN PLACE (above), not merely after the zones are
    // renumbered: setMRFFaces reads each patch's faceCells and size to sort the boundary faces into the
    // included and excluded lists, and a refined patch has more faces than the one the zone was built on.
    if (!f.mrfZones.empty())
    {
        // A GATE'S CONTROL: keep the face lists the zone was built with. That is the plausible wrong port
        // -- nothing throws, every index is in range, and the frame's flux is removed from an EIGHTH of
        // the zone's faces wherever the zone was refined, while the Coriolis source lands on an eighth of
        // its cells.
        if (std::getenv("BRAE_CONTROL_AMR_NO_MRF_UPDATE"))
        {
            // A CONTROL MAY BE WRONG; IT MAY NOT BE UNDEFINED. The kept lists hold the OLD mesh's cell
            // and face indices, and the consumers walk those lists rather than the mesh -- inter_peqn_cpp's
            // makeRelative over z.internalFaces, correctBoundaryVelocity over z.includedFaces[pi],
            // addCoriolis over z.cells -- so on a mesh that only GREW every index is still in range and the
            // control is merely wrong, which is what it is for. On a mesh that SHRANK it is a read past the
            // end, and this fixture does shrink at the tutorial's own deltaT (6390 -> 6362 cells). So the
            // test is not which direction the count moved: it is whether any kept index is now out of
            // range, per list, which also catches a patch that shrank inside a mesh that grew.
            for (const cpu::MRF::Zone& kz : f.mrfZones)
            {
                const bool cellsOut = !kz.cells.empty() && kz.cells.back() >= m.nCells();
                const bool facesOut = !kz.internalFaces.empty()
                                   && kz.internalFaces.back() >= m.nInternalFaces();
                bool patchOut = false;
                for (std::size_t pi = 0; pi < patches.size(); ++pi)
                {
                    for (const std::vector<std::vector<label>>* l : {&kz.includedFaces, &kz.excludedFaces})
                    {
                        if (pi >= l->size() || (*l)[pi].empty()) continue;
                        patchOut = patchOut || (*l)[pi].back() >= patches[pi].size;
                    }
                }
                if (cellsOut || facesOut || patchOut)
                {
                    throw std::runtime_error(
                        std::string(WHO) + "BRAE_CONTROL_AMR_NO_MRF_UPDATE is set and the change left the "
                        "kept zone naming indices the new mesh does not have (cells past the end: "
                        + std::string(cellsOut ? "yes" : "no") + ", internal faces: "
                        + std::string(facesOut ? "yes" : "no") + ", patch faces: "
                        + std::string(patchOut ? "yes" : "no") + "). The control keeps the stale lists on "
                        "purpose, and reading past the end is undefined rather than wrong. Run it on a "
                        "change that does not remove cells or faces.");
                }
            }
            std::printf("  *** CONTROL MODE: the MRF zones keep the face lists they were built with, not "
                        "rebuilt on the new mesh. This run is deliberately wrong. ***\n");
        }
        else
        {
            MRF::update(f.mrfZones, f.mrfSpecs, f.cellZones, m, patches);
        }
    }

    afterPart.emplace("after: the closure's wall distance and filter width (updateMeshInterTurbulence)");
    // THE TURBULENCE CLOSURE'S MESH-DEPENDENT STATE. The FIELDS were mapped by interAmrUpdate; what is
    // left is what OpenFOAM recomputes rather than maps -- the cell wall distance kOmegaSST's F1 and F2
    // blend on, whose MeshObject forces its own latch on a topology change, and the LES filter width.
    // A control is not offered here because the two halves have separate ones already: the RESIZE_NOT_MAP
    // control covers the fields, and the recompute is what this call IS -- skipping it leaves yCell at the
    // OLD CELL COUNT, which the closure's next correct() reads past the end of rather than reading wrongly.
    if (f.turbulence.on)
    {
        // A GATE'S CONTROL: do not recompute them. That is the plausible wrong port -- treat the wall
        // distance and the filter width as state the change carries rather than state it invalidates, which
        // is what a reader would assume from the fact that every OTHER field here is mapped. OpenFOAM's
        // wallDist is a MeshObject and its updateMesh FORCES a recompute (wallDist.C:224-234), so there is
        // nothing to carry.
        if (std::getenv("BRAE_CONTROL_AMR_NO_TURB_UPDATE"))
        {
            std::printf("  *** CONTROL MODE: the turbulence closure's wall distance and filter width are "
                        "NOT recomputed after the change. This run is deliberately wrong. ***\n");
        }
        else
        {
            updateMeshInterTurbulence(f.turbulence, m, g, patches, timeIndex, waveRunner);
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
    afterPart.emplace("after: the flux from Uf and CorrectPhi");
    const bool skipCorrectPhi = (std::getenv("BRAE_CONTROL_AMR_NO_CORRECTPHI") != nullptr);
    if (skipCorrectPhi)
        std::printf("  *** CONTROL MODE: the flux is left as the mapper wrote it -- no Sf & Uf rebuild "
                    "and no pcorr solve. This run is deliberately wrong. ***\n");
    const AmrStageDump asd = openAmrStageDump();
    if (f.correctPhi && !skipCorrectPhi && !motionFollows)
    {
        asd.vectors("UfIn", f.Uf.internal);
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
        asd.scalars("phiAbsPre", f.phi.internal);
        const SurfaceScalarField rAUf = fvc::interpolate(f.rAU, m, g, patches);
        asd.scalars("rAUfCorr", rAUf.internal);
        CorrectPhiInput cin;
        cin.rAUf = &rAUf;
        cin.meshChanging = true;
        cin.meshPhi = nullptr;
        cin.rhoPhi = &f.rhoPhi;
        cin.solveLog = &rep.pcorrSolves;
        correctPhi(f.U, f.phi, f.p_rgh, cin, cpc, m, g, patches);
        asd.scalars("phiAbsPost", f.phi.internal);
        pushFluxToPatches(f, patches);
    }

    // the mixture from the MAPPED alpha1 (mixture.correct()), every one of these assigning its own result
    afterPart.emplace("after: the mixture, p and the reference cell");
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

    // THE PRESSURE REFERENCE IS KEPT, NOT RENUMBERED, and that is OpenFOAM's behaviour rather than a
    // convenience. setRefCell runs ONCE, in createFields.H:104-113, and no solver in the tree re-runs it
    // on a mesh change (the only two other calls are potentialFoam's -writep output, at the END of a
    // run); pRefCell is a plain `label` local of main() that every pEqn then indexes. So after a
    // refinement OpenFOAM pins the SAME INDEX, which on a refined mesh is a different cell -- and
    // reproducing OpenFOAM is the requirement. Renumbering it through the cell map would be the silent
    // substitution here, and the refusal that used to stand in inter_amr_cpp.cu said so in reverse.
    //
    // WHAT IS REFUSED instead is the one case OpenFOAM cannot survive either: an index past the end.
    // findRefCell range-checks at construction and nothing checks again, so an UNREFINEMENT that shrinks
    // the mesh past pRefCell reads out of bounds in OpenFOAM's own Release build. brae says so by name.
    if (f.pRef.needReference && f.pRef.pRefCell >= m.nCells())
        throw std::runtime_error(
            std::string(WHO) + "the case's pressure reference is cell " + std::to_string(f.pRef.pRefCell)
            + " and the change left " + std::to_string(m.nCells()) + " cells. OpenFOAM keeps the index it "
            "found in createFields and never re-checks it, so this is where it would read past the end of "
            "its own p field.");

    // ...and the interface: nHatf and K are mapped, and then rebuilt on the new geometry, which is what
    // mixture.correct() does last -- on the MOVED mesh, by the motion's block, when one follows
    afterPart.emplace("after: the curvature (calculateK)");
    if (!motionFollows)
    {
        interfaceProps::calculateK(f.alpha1, f.interface, m, g, patches, false, f.nHatf, f.K);
    }
    // ...and what it leaves for the pressure corrector, which is where the restart profile's difference
    // first appears: the surface tension force is interpolate(sigma*K)*snGrad(alpha1), so K, nHatf and the
    // alpha calculateK saw are the three inputs behind it.
    asd.scalars("nHatf", f.nHatf.internal);
    asd.scalars("K", f.K);
    asd.scalars("alphaAtK", f.alpha1.internal);
    // ...and the geometry those are a function of once alpha is fixed, because a refined mesh's face
    // centres and areas are computed and not read: they are the only other input.
    if (asd.on)
    {
        std::vector<vector> sf(g.Sf().begin(), g.Sf().begin() + m.nInternalFaces());
        asd.vectors("Sf", sf);
        asd.vectors("C", std::vector<vector>(g.C().begin(), g.C().begin() + m.nCells()));
        asd.scalars("V", std::vector<scalar>(g.V().begin(), g.V().begin() + m.nCells()));
        asd.scalars("weights", std::vector<scalar>(g.weights().begin(),
                                                   g.weights().begin() + m.nInternalFaces()));
        asd.scalars("deltaCoeffs", std::vector<scalar>(g.deltaCoeffs().begin(),
                                                       g.deltaCoeffs().begin() + m.nInternalFaces()));
    }

    (void)rep;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
