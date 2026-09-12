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
// the linear solver from solvers/<schemesField> (:331), nCorr sub-iterations until initialResidual < tol
// (:275-291, both from the functionObject dict; defaults 1 and 1). Steady: no ddt. Only the constant-D
// branch is ported; nut-based and alphaD/alphaDt are refused by name, as the legacy driver's is.
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
#include <cmath>
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
    int                  nCorr = 1;
    scalar               tol = 1.0;     // the functionObject's own `tolerance`: stop the nCorr loop below it
    FieldDivScheme       scheme;
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
};
struct TracerCudaContext
{
    const gpu::rhoSimple::RhoDeviceFields* dev = nullptr;
    const PrimitiveMesh*        m       = nullptr;
    const FvGeometry*           g       = nullptr;
    const std::vector<FvPatch>* patches = nullptr;
    std::string                 fieldDir;
};

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
    out.nCorr        = static_cast<int>(fd.scalarOr("nCorr", 1.0));
    out.tol          = fd.scalarOr("tolerance", 1.0);
    try
    {
        out.scheme = parseFieldDivScheme(caseDir, out.schemesField);
        out.sc     = readTracerSolverControls(fvSolution, out.schemesField);
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
        for (int corr = 0; corr < spec_.nCorr; ++corr)
        {
            std::vector<std::vector<scalar>> sb(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) sb[pi] = s_.boundary[pi]->value();
            const bool needGrad = spec_.scheme.limited || spec_.scheme.linearUpwind || spec_.scheme.nonOrth;
            const std::vector<vector> gradS = needGrad ? fvc::gaussGrad(s_.internal, sb, m, g, patches)
                                                       : std::vector<vector>();
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
            relaxMatrix(M, s_, m, patches, spec_.sc.relax);   // alpha <= 0 is a no-op, as fvMatrix::relax
            const SolverPerformance p = pbicgstab(M, s_.internal, m, patches, spec_.sc.tol, spec_.sc.relTol,
                                                  spec_.sc.maxIter, spec_.sc.minIter);
            s_.evaluateBoundary();
            printSolve(spec_, p, false);
            if (p.initialResidual < spec_.tol) break;
        }
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
        if (spec_.nCorr != 1)
            throw std::runtime_error("brae: scalarTransport '" + spec_.field + "' asks nCorr " + std::to_string(spec_.nCorr) +
                                     " on the device arm, which runs one correction per iteration; refusing rather "
                                     "than running fewer.");
        deviceSolveScalarTransport(
            dm, db_, field_, spec_.field.c_str(), D_, f.phiInt, f.phiBnd, divPhi,
            spec_.scheme.bounded, spec_.scheme.limited, spec_.scheme.linearUpwind, spec_.scheme.nonOrth,
            spec_.scheme.twoByk, spec_.sc.relax, spec_.sc.tol, spec_.sc.relTol, /*checkEvery*/1, spec_.sc.gs,
            [](DeviceBuffer<scalar>&, DeviceBuffer<scalar>&){},   // passive: no reaction
            nullptr, nullptr, nullptr, nullptr,
            ScalarDdt{},            // steady: fvm::ddt(rho, s) drops out
            nullptr, scalar(0),
            false);                 // scalarTransport does not bound its field
        // deviceSolveScalarTransport records its solve in the closure store for the device closures'
        // report; the mirror driver prints that store for the turbulence fields only, so the tracer's
        // line is printed here (and taken off the store, where it does not belong).
        if (!turbStore().empty() && turbStore().back().field == spec_.field)
        {
            const DeviceSolverPerf& perf = turbStore().back().perf;
            SolverPerformance p;
            p.initialResidual = perf.initialResidual;
            p.finalResidual   = perf.finalResidual;
            p.nIterations     = perf.nIterations;
            printSolve(spec_, p, spec_.sc.gs);
            turbStore().pop_back();
        }
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
