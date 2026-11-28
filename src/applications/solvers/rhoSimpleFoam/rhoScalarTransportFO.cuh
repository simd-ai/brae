#pragma once
// OpenFOAM's functionObjects::scalarTransport on the rhoSimpleFoam OF-mirror, both arms (item 15c).
//
// scalarTransport.C, the mass-flux branch (phi.dimensions() == dimMass/dimTime, :311-333):
//     fvScalarMatrix sEqn( fvm::ddt(rho, s) + fvm::div(phi, s, divScheme) - fvm::laplacian(D, s, laplacianScheme)
//                         == fvOptions(rho, s) );
//     sEqn.relax(relaxCoeff);  fvOptions.constrain(sEqn);  sEqn.solve(schemesField_);
// with D = dimensionedScalar(Dname, phi.dimensions()/dimLength, D_) -- the constant the case writes IS the
// diffusivity of the mass-flux equation, no rho on it (:99) -- divScheme "div(phi,<schemesField>)",
// laplacianScheme "laplacian(D<schemesField>,<schemesField>)" (:250), relaxCoeff from
// mesh.relaxEquation(schemesField) (:254; 0 when the case names none, and fvMatrix::relax(0) is a no-op),
// the linear solver from solvers/<schemesField> (:331), and nCorr + 1 solves, `for (int i = 0; i <= nCorr_;
// ++i)` (:315), each stopping the loop once its initialResidual < tol (both from the functionObject dict;
// defaults nCorr_(0) and tol_(1), :184-185). Only the constant-D branch is ported; nut-based and
// alphaD/alphaDt are refused by name, as the legacy driver's is.
//
// fvm::ddt(rho, s) under `ddtSchemes default Euler` (gasMixing/injectorPipe ships it with a tracer) is
// rDeltaT*rho*V on the diagonal and rDeltaT*rho.oldTime()*s.oldTime()*V on the source (EulerDdtScheme.C:437-470);
// under steadyState it vanishes. The object runs from Time::run() after the iteration's write, still at that
// iteration's time index (Time.C:820-825), so:
//   s.oldTime()   is s as this execute found it: its old level is stored at the first access in the time index,
//                 which is this solve's, and on a process's first execute it is CREATED as a copy of s as it stands
//                 (GeometricField::oldTime);
//   rho.oldTime() is rho as the previous execute found it: only pEqn assigns rho (rho = thermo.rho(), the first
//                 write in the next time index, which stores the old level), and nothing writes it between this
//                 execute and that one. On a process's first execute it is rho itself: the closures' ddt (or
//                 this object's, laminar) creates the level after pEqn has written rho, the same rule
//                 RhoStepInput::firstIteration gives the closures.
// Both arms keep those two levels here rather than in the drivers. BRAE_CONTROL_TRACER_DDT_IGNORED=1 drops the term,
// as before; BRAE_CONTROL_TRACER_NCORR_AS_SOLVES=1 takes nCorr as the number of solves (default 1), as before.
//
// The mirror reads controlDict.functions at Time's construction and reported every entry as skipped
// (item 15). This registers a factory for `scalarTransport`, so the entry is BUILT, solved every
// iteration from the mirror's own corrected mass flux, and written on the write cadence beside the
// solved fields. The objects resolve their fields lazily, at the first execute(), because Time is
// constructed before createFields and the start directory is only known after Time resolves it.
#include "brae_time.cuh"
#include "brae_notice.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "geometric_field.cuh"
#include "fv_patch.cuh"
#include "fv_geometry.cuh"
#include "primitive_mesh.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"        // relaxMatrix
#include "ldu_matrix.cuh"           // addEqual
#include "pbicgstab.cuh"
#include "limitedSchemes_cpp.cuh"   // limitedLinearWeights
#include "scheme_parse.cuh"
#include "linear_solver_setup.cuh"  // relaxLookup
#include "rhoCreateFields_cpp.cuh"  // RhoSimpleFields
#include "rhoCreateFields.cuh"      // RhoDeviceFields
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_scalar_transport.cuh"
#include "device_ddt.cuh"           // deviceEulerFvmDdtRho
#include "foam_field_writer.cuh"    // writeVolField
#include "grad_choice.cuh"           // gradOf
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace brae {
namespace tracer {   // its own namespace: cpu::rhoSimple and gpu::rhoSimple both host a `rhoSimple` the drivers sit in

// solvers/<field> and relaxationFactors/equations/<field>, as solve(schemesField) and relaxEquation read them.
struct TracerSolverControls
{
    scalar tol     = 1e-8;
    scalar relTol  = 0.0;
    int    maxIter = 1000;   // lduMatrix::solver's default
    int    minIter = 0;
    bool   gs      = false;  // smoothSolver (honoured on the device arm; the host arm has no Gauss-Seidel)
    bool   gsSym   = true;
    scalar relax   = 0.0;    // 0 = the equation is not named under relaxationFactors: no relaxation
};

inline TracerSolverControls readTracerSolverControls(const FoamDict& fvSolution, const std::string& field)
{
    TracerSolverControls c;
    const FoamDict* solvers = fvSolution.subDict("solvers");
    const FoamDict* s = solvers ? solvers->subDict(field) : nullptr;
    if (!s)
        throw std::runtime_error("brae: scalarTransport '" + field + "' has no fvSolution solvers/" + field +
                                 " entry; OpenFOAM's sEqn.solve(\"" + field + "\") would refuse the case too.");
    c.tol     = s->scalarOr("tolerance", c.tol);
    c.relTol  = s->scalarOr("relTol", c.relTol);
    c.maxIter = static_cast<int>(s->scalarOr("maxIter", static_cast<scalar>(c.maxIter)));
    c.minIter = static_cast<int>(s->scalarOr("minIter", static_cast<scalar>(c.minIter)));
    const std::string solver = s->wordOr("solver", "");
    if (solver == "smoothSolver")
    {
        const std::string sm = s->wordOr("smoother", "");
        if (sm != "symGaussSeidel" && sm != "GaussSeidel")
            throw std::runtime_error("brae: scalarTransport '" + field + "' asks smoothSolver smoother '" + sm +
                                     "', which brae has not ported (symGaussSeidel, GaussSeidel).");
        c.gs = true;
        c.gsSym = (sm == "symGaussSeidel");
    }
    else if (solver != "PBiCGStab" && solver != "PBiCG")
        throw std::runtime_error("brae: scalarTransport '" + field + "' asks solver '" + solver +
                                 "', which brae has not ported for a scalar (PBiCGStab, PBiCG, smoothSolver).");
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;
    scalar a = 0.0;
    if (relaxLookup(re, field, a)) c.relax = a;
    return c;
}

struct TracerSpec
{
    std::string          name;          // the functionObject's name
    std::string          field;         // the transported field
    std::string          schemesField;  // the name the schemes and solver are looked up under
    scalar               D = 0.0;
    int                  nCorr = 0;     // OpenFOAM's nCorr_: nCorr + 1 solves
    scalar               tol = 1.0;     // the functionObject's own `tolerance`: stop the nCorr loop below it
    FieldDivScheme       scheme;
    FieldGradScheme      grad;          // gradSchemes grad(<field>): the limiter's and the corrected laplacian's gradient
    TracerSolverControls sc;
};

// The parts of the run a tracer needs. Filled by the driver AFTER createFields; the objects read it at
// their first execute(), which is after the first momentum/pressure step and so after the fill.
struct TracerHostContext
{
    const cpu::rhoSimple::RhoSimpleFields* f = nullptr;
    const PrimitiveMesh*        m       = nullptr;
    const FvGeometry*           g       = nullptr;
    const std::vector<FvPatch>* patches = nullptr;
    std::string                 fieldDir;      // <case>/<startTime>, where the tracer's file is read from
    std::string                 caseDir;       // where write() puts <time>/<field>
    scalar                      rDeltaT = 0;   // 1/deltaT under ddtSchemes default Euler, 0 under steadyState
};
struct TracerCudaContext
{
    const gpu::rhoSimple::RhoDeviceFields* dev = nullptr;
    const PrimitiveMesh*        m       = nullptr;
    const FvGeometry*           g       = nullptr;
    const std::vector<FvPatch>* patches = nullptr;
    std::string                 fieldDir;
    std::string                 caseDir;
    scalar                      rDeltaT = 0;   // as the host's
};

// The number of solves an execute makes (see the header), and whether fvm::ddt(rho, s) is dropped: both the
// gate's controls.
inline int tracerSolves(const TracerSpec& sp)
{
    static const bool asSolves = std::getenv("BRAE_CONTROL_TRACER_NCORR_AS_SOLVES") != nullptr;
    return asSolves ? std::max(1, sp.nCorr) : sp.nCorr + 1;
}
inline bool tracerDdtIgnored()
{
    static const bool ignored = std::getenv("BRAE_CONTROL_TRACER_DDT_IGNORED") != nullptr;
    return ignored;
}
// OpenFOAM's tracer is NO_WRITE (scalarTransport.C:66-67): only the object's write() writes it, s.write() into the
// time it ran at (:379-389), right after that step's execute. The drivers wrote it beside the solved fields instead,
// before the object had run: every intermediate <time>/<field> held the previous iteration's solve, and the first
// time after a restart held none (gasMixing/injectorPipe from OpenFOAM's iteration 5: no 6/tracer0). The driver's
// final write still writes it, after the last execute. BRAE_CONTROL_TRACER_WRITE_BEFORE_SOLVE=1: as before.
// BRAE_CONTROL_TRACER_GRAD_GAUSS=1: grad(<field>) as Gauss linear whatever the entry says, as before.
inline bool tracerGradGauss()
{
    static const bool gauss = std::getenv("BRAE_CONTROL_TRACER_GRAD_GAUSS") != nullptr;
    return gauss;
}
inline bool tracerWriteBeforeSolve()
{
    static const bool before = std::getenv("BRAE_CONTROL_TRACER_WRITE_BEFORE_SOLVE") != nullptr;
    return before;
}

// The dictionary half, shared by both arms. Returns false (after a notice) for a tracer brae declines.
inline bool parseTracerSpec(const std::string& foName, const FoamDict& fd, const std::string& caseDir,
                            const FoamDict& fvSolution, TracerSpec& out)
{
    if (!fd.found("D"))
    {
        noticeIgnored("functions/" + foName,
                      "scalarTransport without a constant `D` (nut-based or alphaD/alphaDt diffusivity) is not "
                      "implemented, so this tracer is NOT solved.");
        return false;
    }
    out.name         = foName;
    out.field        = fd.wordOr("field", foName);
    out.schemesField = fd.wordOr("schemesField", out.field);
    out.D            = fd.scalarOr("D", 0.0);
    out.nCorr        = static_cast<int>(fd.scalarOr("nCorr", 0.0));
    if (out.nCorr < 0)
    {
        throw std::runtime_error("brae: scalarTransport '" + out.field + "' asks nCorr " + std::to_string(out.nCorr) +
                                 "; OpenFOAM solves nCorr + 1 times, and a negative count is not a case brae runs.");
    }
    out.tol          = fd.scalarOr("tolerance", 1.0);
    try
    {
        out.scheme = parseFieldDivScheme(caseDir, out.schemesField);
        requireUncappedLaplacian(out.scheme, out.schemesField);
        out.sc     = readTracerSolverControls(fvSolution, out.schemesField);
        // grad(<field>), by the FIELD's name: what limitedLinear's limiter (LimitedScheme.C:54, fvc::grad(lPhi)) and the
        // corrected laplacian's correction (correctedSnGrad.C:52-55) take. gasMixing/injectorPipe resolves it to its
        // `default leastSquares`; both arms took Gauss linear, 1.2e-03 in tracer0 over three iterations from OpenFOAM's
        // iteration 5 (Arrhenius 5877691).
        out.grad   = parseFieldGradScheme(caseDir, out.field);
        if (!out.grad.unsupportedLimiter.empty() || !(out.grad.leastSquares || out.grad.gaussLinear))
        {
            throw std::runtime_error("brae: scalarTransport '" + out.field + "': gradSchemes grad(" + out.field + ") is `" +
                                     out.grad.raw + "`; the tracer carries Gauss linear or leastSquares, optionally "
                                     "cellLimited");
        }
        // linearUpwind's correction takes the gradient its entry NAMES (linearUpwind.C:61-68), a third lookup: carried
        // here only when that and grad(<field>) are both plain Gauss linear, the one gradient both arms build for it
        if (out.scheme.linearUpwind)
        {
            const FieldGradScheme lu = parseNamedGradScheme(caseDir, out.scheme.luGradName);
            const bool plainField = out.grad.gaussLinear && !out.grad.leastSquares && out.grad.cellLimitK <= scalar(0);
            const bool plainLu    = lu.gaussLinear && !lu.leastSquares && lu.cellLimitK <= scalar(0) && lu.unsupportedLimiter.empty();
            if (!plainField || !plainLu)
            {
                throw std::runtime_error("brae: scalarTransport '" + out.field + "': linearUpwind " + out.scheme.luGradName +
                                         " with grad(" + out.field + ") `" + out.grad.raw + "` and `" + lu.raw +
                                         "`; the tracer's linearUpwind carries plain Gauss linear only");
            }
        }
    }
    catch (const std::exception& e)
    {
        noticeIgnored("functions/" + foName, std::string(e.what()) + " -- this tracer is NOT solved.");
        return false;
    }
    return true;
}

inline void printSolve(const TracerSpec& sp, const SolverPerformance& p, bool gs)
{
    std::printf("%s:  Solving for %s, Initial residual = %g, Final residual = %g, No Iterations %d\n",
                gs ? "smoothSolver" : "PBiCGStab", sp.field.c_str(), (double)p.initialResidual,
                (double)p.finalResidual, p.nIterations);
}

// ---------------------------------------------------------------------------------------------
// The HOST arm: the equation assembled with the host fvm operators, exactly as the host closures
// assemble theirs (kEpsilon_cpp.cu), solved with the host PBiCGStab.
class RhoTracerHostFO : public FunctionObject
{
public:
    RhoTracerHostFO(TracerSpec spec, const TracerHostContext& ctx) : spec_(std::move(spec)), ctx_(ctx) {}
    const std::string& name() const override { return spec_.name; }

    bool execute() override
    {
        if (failed_) return true;
        if (!ready_ && !initialise()) return true;
        const PrimitiveMesh& m = *ctx_.m;
        const FvGeometry& g = *ctx_.g;
        const std::vector<FvPatch>& patches = *ctx_.patches;
        const SurfaceScalarField& phi = ctx_.f->phi;   // the corrected MASS flux this iteration produced
        const label nC = m.nCells();
        const std::vector<scalar> divPhi = fvc::div(phi, m, g, patches);
        SurfaceScalarField Df;
        Df.internal.assign(m.nInternalFaces(), spec_.D);
        Df.boundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) Df.boundary[pi].assign(patches[pi].size, spec_.D);
        // fvm::ddt(rho, s)'s two old levels (see the header): s as this execute found it, and rho as the previous
        // execute found it (rho itself on this process's first)
        const std::vector<scalar>& rho = ctx_.f->rho.internal;
        const bool ddt = (ctx_.rDeltaT > scalar(0)) && !tracerDdtIgnored();
        const std::vector<scalar> s0 = ddt ? s_.internal : std::vector<scalar>();
        const std::vector<scalar>& rho0 = rhoPrev_.empty() ? rho : rhoPrev_;
        const int nSolves = tracerSolves(spec_);
        for (int corr = 0; corr < nSolves; ++corr)
        {
            std::vector<std::vector<scalar>> sb(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) sb[pi] = s_.boundary[pi]->value();
            const bool needGrad = spec_.scheme.limited || spec_.scheme.linearUpwind || spec_.scheme.nonOrth;
            GradChoice choice;
            if (!tracerGradGauss())
            {
                choice.leastSquares = spec_.grad.leastSquares;
                choice.cellLimitK   = spec_.grad.cellLimitK;
            }
            const std::vector<vector> gradS = !needGrad ? std::vector<vector>()
                                            : choice.gaussLinear() ? fvc::gaussGrad(s_.internal, sb, m, g, patches)
                                                                   : gradOf(s_, choice, m, g, patches);
            FvScalarMatrix M = spec_.scheme.limited
                ? fvm::div(phi.internal, phi.boundary, s_,
                           cpu::limitedSchemes::limitedLinearWeights(phi.internal, s_, gradS, spec_.scheme.coeff, m, g),
                           m, patches)
                : fvm::div(phi.internal, phi.boundary, s_, m, patches);
            FvScalarMatrix L = fvm::laplacian(Df, s_, m, g, patches, spec_.scheme.nonOrth);
            if (spec_.scheme.nonOrth)
            {
                const std::vector<scalar> nc = fvm::laplacianNonOrthSource<scalar, vector>(Df, s_, gradS, m, g, patches);
                for (label c = 0; c < nC; ++c) L.source[c] -= nc[c];
            }
            addEqual(M, L, -1.0);
            if (spec_.scheme.bounded)
                for (label c = 0; c < nC; ++c) M.diag[c] -= divPhi[c] * g.V()[c];   // -Sp(div(phi), s)
            if (spec_.scheme.linearUpwind)
            {
                const std::vector<scalar> lu = fvm::linearUpwindCorrection<scalar, vector>(phi.internal, gradS, m, g);
                for (label c = 0; c < nC; ++c) M.source[c] -= lu[c];
            }
            if (ddt)   // fvm::ddt(rho, s), EulerDdtScheme.C:452-466, added as the closures add theirs (kEpsilon_cpp.cu:633)
            {
                for (label c = 0; c < nC; ++c)
                {
                    const scalar V = g.V()[c];
                    M.diag[c]   += ctx_.rDeltaT * rho[c] * V;
                    M.source[c] += ctx_.rDeltaT * rho0[c] * s0[c] * V;
                }
            }
            relaxMatrix(M, s_, m, patches, spec_.sc.relax);   // alpha <= 0 is a no-op, as fvMatrix::relax
            const SolverPerformance p = pbicgstab(M, s_.internal, m, patches, spec_.sc.tol, spec_.sc.relTol,
                                                  spec_.sc.maxIter, spec_.sc.minIter);
            s_.evaluateBoundary();
            printSolve(spec_, p, false);
            if (p.initialResidual < spec_.tol) break;
        }
        if (ctx_.rDeltaT > scalar(0))
        {
            rhoPrev_ = rho;
        }
        return true;
    }
    bool write() override
    {
        if (!ready_ || tracerWriteBeforeSolve())
        {
            return true;
        }
        const std::string outDir = ctx_.caseDir + "/" + step().timeName;
        std::filesystem::create_directories(outDir);
        writeVolField(ctx_.fieldDir + "/" + spec_.field, outDir + "/" + spec_.field, s_.internal, *ctx_.patches, 12,
                      boundaryFlat());
        return true;
    }
    bool ready() const { return ready_; }
    const std::string& fieldName() const { return spec_.field; }
    const std::vector<scalar>& hostField() const { return s_.internal; }
    std::vector<scalar> boundaryFlat() const
    {
        std::vector<scalar> out;
        for (std::size_t pi = 0; pi < ctx_.patches->size(); ++pi)
        {
            if (isCoupledInterfaceType((*ctx_.patches)[pi].type)) continue;
            const std::vector<scalar>& v = s_.boundary[pi]->value();
            out.insert(out.end(), v.begin(), v.end());
        }
        return out;
    }

private:
    bool initialise()
    {
        if (!ctx_.f || !ctx_.m || !ctx_.g || !ctx_.patches || ctx_.fieldDir.empty()) return false;
        const std::string path = ctx_.fieldDir + "/" + spec_.field;
        if (!std::filesystem::exists(path))
        {
            noticeIgnored("functions/" + spec_.name, "scalarTransport field '" + spec_.field + "' is not present at " +
                          path + ", so this tracer is NOT solved.");
            failed_ = true;
            return false;
        }
        if (spec_.sc.gs)
            noticeApproximated("solvers/" + spec_.schemesField + " solver",
                               "case asks 'smoothSolver'; the host mirror arm runs PBiCGStab on it (same system and "
                               "tolerance; the device arm honours the smoother)");
        s_ = buildField<scalar>(readField<scalar>(path), *ctx_.patches, ctx_.m->nCells());
        s_.evaluateBoundary();
        ready_ = true;
        return true;
    }
    TracerSpec               spec_;
    const TracerHostContext& ctx_;
    GeometricField<scalar>   s_;
    std::vector<scalar>      rhoPrev_;   // rho as the previous execute found it: the next one's rho.oldTime()
    bool                     ready_ = false, failed_ = false;
};

// ---------------------------------------------------------------------------------------------
// The CUDA arm: the same equation through deviceSolveScalarTransport, the module the device closures
// use, on the device mirror's corrected mass flux.
class RhoTracerCudaFO : public FunctionObject
{
public:
    RhoTracerCudaFO(TracerSpec spec, const TracerCudaContext& ctx) : spec_(std::move(spec)), ctx_(ctx) {}
    const std::string& name() const override { return spec_.name; }

    bool execute() override
    {
        if (failed_) return true;
        if (!ready_ && !initialise()) return true;
        const DeviceMesh& dm = ctx_.dev->dm;
        const gpu::rhoSimple::RhoSolverFields& f = ctx_.dev->f;
        DeviceBuffer<scalar> divPhi;
        deviceDiv(dm, f.phiInt, f.phiBnd, divPhi);   // fvc::div(phi) of the mass flux: the `bounded` Sp term
        // the object's own tolerance and relTol, and the old cap: no maxIter or minIter is handed down here yet
        // (kUnreadEntryMaxIter)
        const ScalarStopRule stop{spec_.sc.tol, spec_.sc.relTol, kUnreadEntryMaxIter, 0};
        // fvm::ddt(rho, s)'s two old levels (see the header): s as this execute found it, and rho as the previous
        // execute found it (rho itself on this process's first)
        const bool ddt = (ctx_.rDeltaT > scalar(0)) && !tracerDdtIgnored();
        DeviceBuffer<scalar> s0;
        if (ddt)
        {
            deviceCopy(s0, field_);
        }
        const DeviceBuffer<scalar>& rho0 = rhoPrev_.size() ? rhoPrev_ : f.rho;
        // ...added where the transport adds a reaction: to the assembled diagonal and source, before relax
        auto ddtTerm = [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src)
        {
            if (!ddt)
            {
                return;   // passive: no reaction
            }
            DeviceBuffer<scalar> dd, ds;
            deviceEulerFvmDdtRho(ctx_.rDeltaT, f.rho, rho0, s0, dm.V, dd, ds);   // EulerDdtScheme.C:452-466
            deviceAxpy(scalar(1), dd, diag);
            deviceAxpy(scalar(1), ds, src);
        };
        const int nSolves = tracerSolves(spec_);
        const bool needGrad = spec_.scheme.limited || spec_.scheme.linearUpwind || spec_.scheme.nonOrth;
        const bool ownGrad  = needGrad && !tracerGradGauss() &&
                              (spec_.grad.leastSquares || spec_.grad.cellLimitK > scalar(0));
        for (int corr = 0; corr < nSolves; ++corr)
        {
            // grad(<field>) under its own entry, from the field as this solve finds it (null: the transport's Gauss)
            DeviceBuffer<scalar> fgx, fgy, fgz;
            if (ownGrad)
            {
                DeviceBuffer<scalar> bv;
                deviceBCValue(db_, field_, bv);
                deviceGradOf(dm, field_, bv, spec_.grad.leastSquares, spec_.grad.cellLimitK, fgx, fgy, fgz);
            }
            deviceSolveScalarTransport(
                dm, db_, field_, spec_.field.c_str(), D_, f.phiInt, f.phiBnd, divPhi,
                spec_.scheme.bounded, spec_.scheme.limited, spec_.scheme.linearUpwind, spec_.scheme.nonOrth,
                spec_.scheme.twoByk, spec_.sc.relax, stop, /*checkEvery*/1, spec_.sc.gs,
                ddtTerm,
                nullptr, nullptr, nullptr, nullptr,
                ScalarDdt{},            // the incompressible bundle: fvm::ddt(rho, s) comes in through ddtTerm
                nullptr, scalar(0),
                false,                  // scalarTransport does not bound its field
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,   // fvoSetMask .. limGradZ: none
                // precon, nSweeps, gsSymmetric, boundFloor, fieldGradLimitK, storedBnd, lapLimit: the defaults this call
                // has always taken. The entry's nSweeps and smoother are not handed down, as its minIter and maxIter are
                // not (the stop rule above); that gap is this arm's, unchanged here.
                nullptr, 1, true, scalar(1e-15), scalar(0), nullptr, scalar(1),
                ownGrad ? &fgx : nullptr, ownGrad ? &fgy : nullptr, ownGrad ? &fgz : nullptr);
            // deviceSolveScalarTransport records its solve in the closure store for the device closures'
            // report; the mirror driver prints that store for the turbulence fields only, so the tracer's
            // line is printed here (and taken off the store, where it does not belong). Its initial residual
            // is what stops the nCorr loop.
            if (turbStore().empty() || turbStore().back().field != spec_.field)
            {
                if (nSolves > 1)
                {
                    throw std::runtime_error("brae: scalarTransport '" + spec_.field + "' (device arm): the solve left no "
                                             "residual record, so the nCorr loop cannot stop where OpenFOAM's does.");
                }
                break;
            }
            const DeviceSolverPerf& perf = turbStore().back().perf;
            SolverPerformance p;
            p.initialResidual = perf.initialResidual;
            p.finalResidual   = perf.finalResidual;
            p.nIterations     = perf.nIterations;
            printSolve(spec_, p, spec_.sc.gs);
            turbStore().pop_back();
            if (p.initialResidual < spec_.tol)
            {
                break;
            }
        }
        if (ctx_.rDeltaT > scalar(0))
        {
            deviceCopy(rhoPrev_, f.rho);
        }
        return true;
    }
    bool write() override
    {
        if (!ready_ || tracerWriteBeforeSolve())
        {
            return true;
        }
        const std::string outDir = ctx_.caseDir + "/" + step().timeName;
        std::filesystem::create_directories(outDir);
        writeVolField(ctx_.fieldDir + "/" + spec_.field, outDir + "/" + spec_.field, hostField(), *ctx_.patches, 12,
                      boundaryFlat());
        return true;
    }
    bool ready() const { return ready_; }
    const std::string& fieldName() const { return spec_.field; }
    std::vector<scalar> hostField() const { return field_.host(); }
    std::vector<scalar> boundaryFlat() const
    {
        DeviceBuffer<scalar> b;
        deviceBCValue(db_, field_, b);
        return b.host();
    }

private:
    bool initialise()
    {
        if (!ctx_.dev || !ctx_.m || !ctx_.g || !ctx_.patches || ctx_.fieldDir.empty()) return false;
        const std::string path = ctx_.fieldDir + "/" + spec_.field;
        if (!std::filesystem::exists(path))
        {
            noticeIgnored("functions/" + spec_.name, "scalarTransport field '" + spec_.field + "' is not present at " +
                          path + ", so this tracer is NOT solved.");
            failed_ = true;
            return false;
        }
        const label nC = ctx_.m->nCells();
        GeometricField<scalar> sf = buildField<scalar>(readField<scalar>(path), *ctx_.patches, nC);
        sf.evaluateBoundary();
        field_.copyFrom(sf.internal);
        D_.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nC), spec_.D));
        db_ = buildDeviceBoundary(sf, *ctx_.patches, *ctx_.g);
        ready_ = true;
        return true;
    }
    TracerSpec               spec_;
    const TracerCudaContext& ctx_;
    DeviceBuffer<scalar>     field_, D_;
    DeviceBuffer<scalar>     rhoPrev_;   // as the host arm's
    DeviceBoundary           db_;
    bool                     ready_ = false, failed_ = false;
};

// The factories Time registers. Each pushes the object it builds onto the driver's list, which is how
// the driver writes the tracers on its write cadence (OF's write() writes transportedField()).
inline FunctionObjectList::Factory rhoTracerHostFactory(const std::string& caseDir, const FoamDict& fvSolution,
                                                        const TracerHostContext& ctx,
                                                        std::vector<RhoTracerHostFO*>& made)
{
    return [&caseDir, &fvSolution, &ctx, &made](const std::string& foName, const FoamDict& fd) -> std::unique_ptr<FunctionObject>
    {
        TracerSpec spec;
        if (!parseTracerSpec(foName, fd, caseDir, fvSolution, spec)) return nullptr;
        auto fo = std::make_unique<RhoTracerHostFO>(std::move(spec), ctx);
        made.push_back(fo.get());
        return fo;
    };
}
inline FunctionObjectList::Factory rhoTracerCudaFactory(const std::string& caseDir, const FoamDict& fvSolution,
                                                        const TracerCudaContext& ctx,
                                                        std::vector<RhoTracerCudaFO*>& made)
{
    return [&caseDir, &fvSolution, &ctx, &made](const std::string& foName, const FoamDict& fd) -> std::unique_ptr<FunctionObject>
    {
        TracerSpec spec;
        if (!parseTracerSpec(foName, fd, caseDir, fvSolution, spec)) return nullptr;
        auto fo = std::make_unique<RhoTracerCudaFO>(std::move(spec), ctx);
        made.push_back(fo.get());
        return fo;
    };
}

}   // namespace tracer
}   // namespace brae
