// interFoam's time directories, written as OpenFOAM's interFoam writes them.
//
// OpenFOAM writes at exactly one place, `runTime.write()` after the PIMPLE loop (interFoam.C:175), and only
// when Time::operator++ marked the step a write time (Time.C:1103-1130). What it writes is every
// registered AUTO_WRITE object -- for interFoam alpha.<phase1>, U, p_rgh, p, phi and alphaPhi0.<phase1>,
// the turbulence model's own fields, and uniform/time, uniform/cumulativeContErr and
// uniform/functionObjects/functionObjectProperties -- each patch through its own condition's write()
// (the table in inter_writer_cpp.cu cites every one). brae_interFoam wrote NOTHING: both drivers ran the
// time loop and handed nothing to disk, so a real case ran to its end and left only its 0/.
//
// WHAT THE WRITER READS, AND WHEN. The case's own start-directory files are the templates -- a patch's
// TYPE and the parameters OpenFOAM echoes (p0, inletValue, the Function1 of a flow rate, a wall function's
// blending) come from them -- and the VALUES come from the solver's own patch objects as they stand at the
// write: the STORED values OpenFOAM writes, never a fresh evaluation, which would be a different number
// and, done inside the loop, a different run. fixedFluxPressure's gradient is the one the last assembly
// set, not the template's (fixedGradientFvPatchField.C:240 writes gradient_).
//
// WHAT IT REFUSES, at the first write time rather than at startup: a configuration whose extra files are
// not written yet (sub-cycled alpha's alpha.<phase1>_0, CrankNicolson's ddt0 fields, local time stepping's
// rDeltaT, a moving or refining mesh's Uf / rAU / points / refinement state, waves' waveProperties, LES).
// The run before that write is kept -- a refusal at startup would throw away a case the solver can run
// and only the output of which is incomplete -- and every refused file is named at construction so a long
// run is not lost silently.
#pragma once
#include "cf_types.cuh"
#include "crank_nicolson_ddt_scheme_cpp.cuh"
#include "foam_dict.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "geometric_field.cuh"
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

struct InterTurbulence;
struct InterFields;
struct InterWaves;
struct InterAmr;
}   // namespace interFoam
}   // namespace cpu
class DisplacementLaplacianFvMotionSolver;
class RigidBodyMeshMotion;
namespace cpu {
namespace interFoam {

// CrankNicolson's state on a MOVING mesh, as the registry holds it at a write time (RAS/floatingObject):
// the scheme's ddt0 fields, cells and patches, U's old-time level and the mesh's old volumes. The closure's
// half -- ddt0(k), ddt0(epsilon), k_0, epsilon_0 -- is read off InterTurbulence.
struct InterWriteCrankNicolson
{
    const fv::CrankNicolsonDdt0<vector>* ddt0RhoU = nullptr;
    const fv::CrankNicolsonDdt0<vector>* ddtCorrU = nullptr;
    const fv::CrankNicolsonDdt0<vector>* ddtCorrUf = nullptr;
    const fv::CrankNicolsonDdt0<scalar>* meshPhi0 = nullptr;
    // U.oldTime(): its cells and STORED patch values
    const std::vector<vector>* UOld = nullptr;
    const std::vector<std::vector<vector>>* UOldBnd = nullptr;
    // mesh().V0()
    const std::vector<scalar>* V0 = nullptr;
    // Uf.oldTime(), once OpenFOAM writes it -- null until then. The level takes Uf's writeOpt in
    // storeOldTime only `if (field0Ptr_->field0Ptr_)` (GeometricField.C:922-939), and Uf's old-old level
    // is created by fvcDdtUfCorr's evaluate branch in the SECOND step, after that step's store has run. So
    // the first store that finds it is the third step's, and Uf_0 is written from there on (MEASURED on
    // RAS/floatingObject: absent at steps one and two, present at five).
    const SurfaceVectorField* UfOld = nullptr;
};

// What one write time hands the writer: the solver's own objects, read and never modified.
struct InterWriteState
{
    scalar time = 0;
    // Time::timeIndex(): the step count from the case's own start index (uniform/time `index` on a
    // restart, 0 otherwise) -- `writeControl timeStep` counts on it, and uniform/time records it
    label timeIndex = 0;
    // the step that reached `time`
    scalar deltaT = 0;
    const GeometricField<scalar>* alpha1 = nullptr;
    const GeometricField<vector>* U = nullptr;
    const GeometricField<scalar>* p_rgh = nullptr;
    // p's cells, and its patch values p_rgh_b + rho_b*gh_b (pEqn.H:72) in mesh patch order
    const std::vector<scalar>* p = nullptr;
    const std::vector<std::vector<scalar>>* pBoundary = nullptr;
    const SurfaceScalarField* phi = nullptr;
    // alphaPhi0.<phase1>: the alpha flux the step's last alpha solve left (alphaEqn.H, alphaPhi10)
    const SurfaceScalarField* alphaPhi0 = nullptr;
    // null or `on == false` is laminar
    const InterTurbulence* turbulence = nullptr;
    // The CELL values, where the solver keeps them apart from its patch objects: the device loop holds
    // alpha, U and p_rgh on the GPU and their patch values on the host. Null takes the field's own
    // `internal`.
    const std::vector<scalar>* alpha1Cells = nullptr;
    const std::vector<vector>* UCells = nullptr;
    const std::vector<scalar>* p_rghCells = nullptr;
    // alpha.<phase1>_0, when alpha is sub-cycled: alpha's cells and STORED patch values as they stood at
    // the start of the step -- the old-time level the sub-cycle restores (subCycle.H, ~subCycleField) and
    // OpenFOAM writes at every write time, the first included (measured on laminar/mixerVessel2D)
    const std::vector<scalar>* alpha1OldCells = nullptr;
    const std::vector<std::vector<scalar>>* alpha1OldBoundary = nullptr;
    // ...and alpha's stored patch values as the step's LAST sub-cycle began, which the old level keeps on a
    // patch whose operator= is a no-op (the fixedValue and mixed families; see InterWriter::write)
    const std::vector<std::vector<scalar>>* alpha1SubCycleBoundary = nullptr;
    // local time stepping's rDeltaT cells (createRDeltaT.H), empty on a case without it
    const std::vector<scalar>* rDeltaT = nullptr;
    // a moving mesh: Uf after the last corrector's correctUf, the mesh-motion flux of the step's move and
    // the points after it (null on a mesh that does not move)
    const SurfaceVectorField* Uf = nullptr;
    const SurfaceScalarField* meshPhi = nullptr;
    const std::vector<vector>* points = nullptr;
    // a displacementLaplacian motion's solver, whose pointDisplacement and cellDisplacement are written
    const DisplacementLaplacianFvMotionSolver* displacement = nullptr;
    // a rigidBodyMotion's solver, whose pointDisplacement and uniform/rigidBodyMotionState are written
    const RigidBodyMeshMotion* rigidBody = nullptr;
    // a refining mesh: the live mesh (its faces, owner, neighbour) and the adaptive-mesh state (hexRef8's
    // levels and history, level0Edge, the zones, points0 when it also moves)
    const PrimitiveMesh* mesh = nullptr;
    const InterAmr* amr = nullptr;
    // ...and, when it also moves, the motion solver's points0 as the changes have mapped it
    const std::vector<vector>* points0 = nullptr;
    // correctPhi's rAU cells (initCorrectPhi.H), 1/UEqn.A() of the step's last corrector
    const std::vector<scalar>* rAU = nullptr;
    // CrankNicolson on a moving mesh; null under any other scheme
    const InterWriteCrankNicolson* cn = nullptr;
};

class InterWriter
{
public:
    // Reads controlDict's write settings and the start directory's uniform/ state. Refuses by name a
    // writeControl brae cannot honour (clockTime, cpuTime, an unknown word) and a missing or invalid
    // writeInterval, where OpenFOAM stops too (TimeIO.C:284-297).
    InterWriter(
        const std::string& caseDir,
        const std::string& startDir,
        const std::vector<FvPatch>& patches,
        const std::string& phase1Name);

    // A file OpenFOAM would write for this case that brae does not: thrown at the first write time,
    // listed now.
    void refuseAtFirstWrite(
        const std::string& file,
        const std::string& why);

    // Time::operator++'s writeTime_ for the step that has just reached `timeIndex`. `runTimeIndexMoved`
    // is what WriteCadence::advance returned for that step (runTime and adjustableRunTime).
    bool isWriteTime(
        label timeIndex,
        bool runTimeIndexMoved) const;

    // Time::operator++'s deltaT0_ bookkeeping, which runs on EVERY step, written or not.
    void stepTaken(scalar deltaT);

    // continuityErrs.H: cumulativeContErr += deltaT*weightedAverage(div(phi), V), once per corrector.
    void addContinuityError(
        scalar deltaT,
        const std::vector<scalar>& divPhi,
        const std::vector<scalar>& V);

    // The start index a restart continues from (uniform/time `index`), 0 otherwise.
    label startTimeIndex() const { return startTimeIndex_; }

    // alpha.<phase1>_0 is written (a sub-cycled alpha): the loops capture it at the start of a write step
    void writeAlphaOld() { alphaOld_ = true; }
    bool writesAlphaOld() const { return alphaOld_; }
    // The start directory holds alpha.<phase1>_0: OpenFOAM builds the old level from that FILE
    // (readOldTimeIfPresent, GeometricField.C:120, :151-160), so its patches are the file's -- a contact
    // angle's gradient included (readGradientEntry), which brae does not read.
    bool startHoldsAlphaOld() const { return startHoldsAlphaOld_; }
    const std::vector<FvPatch>& patches() const { return patches_; }
    // the start directory holds this file (plain or .gz)
    bool startHolds(const std::string& file) const;
    // the wave models whose state is written, uniform/waveProperties.<patch> (null: a case without waves)
    void writeWaveState(const InterWaves* waves) { waves_ = waves; }
    // local time stepping's rDeltaT is written (createRDeltaT.H: READ_IF_PRESENT, AUTO_WRITE)
    void writeRDeltaT() { rDeltaT_ = true; }
    // a solidBody-moved mesh's state is written: polyMesh/points, meshPhi, Uf
    void writeMeshMotion() { meshMotion_ = true; }
    // CrankNicolson on a moving mesh: the scheme's ddt0 fields, the old-time levels and V0 are written
    void writeCrankNicolson() { crankNicolson_ = true; }
    bool writesCrankNicolson() const { return crankNicolson_; }
    // ...and a displacementLaplacian motion's own fields: pointDisplacement, cellDisplacement
    void writeDisplacement() { displacement_ = true; }
    // ...or a rigidBodyMotion's: pointDisplacement, uniform/rigidBodyMotionState
    void writeRigidBody() { rigidBody_ = true; }
    // a refining mesh's own state: the polyMesh files after the first topology change, hexRef8's files and
    // the cellLevel field at every write, and Uf -- which a mesh that also moves writes with writeMeshMotion
    void writeRefineMesh(bool alsoMoves)
    {
        refine_ = true;
        refineMoves_ = alsoMoves;
    }
    // correctPhi's rAU (initCorrectPhi.H:3-17, AUTO_WRITE)
    void writeRAU() { rAU_ = true; }
    bool writesRAU() const { return rAU_; }
    // What the old level's patches keep from the moment OpenFOAM creates it -- the first alpha1.oldTime()
    // of the run, subCycleField's constructor (subCycle.H:78) ahead of the first alpha solve: each contact
    // angle's gradient, which later assignments never touch (values only, fvPatchField.C:407-413,
    // :552-558). Called there by both loops; once.
    void noteAlphaOldCreation(const GeometricField<scalar>& alpha1);

    // One time directory. THE STEP PAYS FOR BUILDING IT, NOT FOR FORMATTING ITS NUMBERS OR FOR THE DISK: every
    // file's structure is built here, from the solver's fields as they stand, with each long list of numbers
    // COPIED and left as a marker; the lists' text, the files and purgeWrite's removals are a job that runs on
    // another thread while the next steps do (inter_writer_cpp.cu, flushJob). One job at a time: a write waits
    // for the one before. A job that throws stops the run at the next step boundary that finds it ended, at
    // the next write, or at finish(), whichever comes first.
    //   BRAE_CONTROL_WRITE_IN_STEP=1: everything inside the step, as before -- the identity gate's other arm
    void write(const InterWriteState& s);
    // The write in flight, waited for; what it threw, thrown here. Both time loops call it before they return:
    // a run has not ended while a time directory is still being written.
    void finish();
    ~InterWriter();

    // the directories written so far, oldest first (the purgeWrite FIFO)
    const std::deque<std::string>& written() const { return written_; }

    // Renders `field`'s patches once against the start directory's template and registers whatever
    // write() would refuse -- a condition with no transcribed write() -- so the refusal is named at
    // startup and not found at the first write time. The same code path write() takes, cells left out.
    template <typename T>
    void probeConditions(
        const std::string& field,
        const GeometricField<T>& fld);

private:
    struct Template
    {
        std::string dimensions;   // the full `[...]` text
        FoamDict boundary;        // boundaryField, expanded as the reader expands it
        bool present = false;
    };

    const Template& templateFor(const std::string& fieldName);
    // One volume field's file: the template named `field` for the patch types and the parameters
    // OpenFOAM echoes, `cells` and `bcs` for the values; `derived`, when given, replaces every patch value
    // and writes `calculated` (p).
    template <typename T>
    std::string fieldText(
        const std::string& className,
        const std::string& location,
        const std::string& field,
        const std::string& templateName,
        const std::vector<T>& cells,
        const std::vector<std::unique_ptr<fvPatchField<T>>>& bcs,
        const std::vector<std::vector<T>>* derived,
        const std::vector<std::vector<T>>* stored,
        const std::vector<std::vector<T>>* storedGradient);
    std::string timeName(scalar t, int precision) const;
    std::string header(
        const std::string& className,
        const std::string& location,
        const std::string& object) const;
    std::string header(
        const std::string& className,
        const std::string& location,
        const std::string& object,
        const std::string& note,
        const std::string& metaNames) const;
    // emit() only queues a file; write() writes the queue once every file of the time is built, so a
    // refusal thrown half-way through leaves no partial time directory behind
    void emit(
        const std::string& path,
        const std::string& text,
        bool compressible) const;
    void writeFile(
        const std::string& path,
        const std::string& text,
        bool compressible) const;

    std::string caseDir_;
    std::string startDir_;
    const std::vector<FvPatch>& patches_;
    std::string phase1_;

    std::string control_;
    scalar interval_ = 0;
    int purge_ = 0;
    int precision_ = 6;
    bool compress_ = false;
    std::string timeFormat_ = "general";
    // Time::precision_, which OpenFOAM raises for good once a name would not distinguish two times
    int timePrecision_ = 6;

    label startTimeIndex_ = 0;
    scalar deltaTSave_ = 0;
    scalar deltaT0_ = 0;
    scalar cumulativeContErr_ = 0;
    bool alphaOld_ = false;
    const InterWaves* waves_ = nullptr;
    bool rDeltaT_ = false;
    bool meshMotion_ = false;
    bool crankNicolson_ = false;
    bool displacement_ = false;
    bool rigidBody_ = false;
    bool refine_ = false;
    bool refineMoves_ = false;
    bool rAU_ = false;
    bool startHoldsAlphaOld_ = false;
    bool oldLevelNoted_ = false;
    std::vector<std::vector<scalar>> oldLevelGrad_;

    struct PendingFile
    {
        std::string path;
        std::string text;
        bool compressible = false;
    };
    mutable std::vector<PendingFile> pending_;

    // the write in flight (its two times come back for the phase table), and the control's held one
    struct BackgroundTimes
    {
        double lists = 0;
        double disk = 0;
    };
    struct Job;
    std::future<BackgroundTimes> inFlight_;
    std::shared_ptr<Job> held_;
    void settle();
    void pollWrite();

    std::vector<std::pair<std::string, std::string>> refused_;
    std::map<std::string, Template> templates_;
    std::deque<std::string> written_;
};

// Every file OpenFOAM would write for THIS case that the writer does not, registered on `w` (so the run
// stops at its first write time, having said so at startup).
void registerUnwritten(
    InterWriter& w,
    const InterFields& f);

// p's patch values, p == p_rgh + rho*gh on each patch (pEqn.H:72), from the rho patch values the step
// used -- the construction the rigid-body load already makes (inter_driver_cpp.cu, interMeshUpdate).
std::vector<std::vector<scalar>> staticPressureBoundary(
    const GeometricField<scalar>& p_rgh,
    const std::vector<std::vector<scalar>>& rhoBnd,
    const std::vector<std::vector<scalar>>& ghfBoundary);

} // namespace interFoam
} // namespace cpu
} // namespace brae
