#pragma once
// functionObjects::scalarTransport -- OF src/functionObjects/solvers/scalarTransport.C.
//
// WHAT OF DOES, and what is reproduced here. OF's compressible branch is
//
//     fvm::ddt(rho, s) + fvm::div(phi, s, divScheme) - fvm::laplacian(D, s, laplacianScheme)
//         == fvOptions(rho, s)
//     sEqn.relax(relaxCoeff);  iterate until initialResidual() < tol
//
// On a STEADY solver the ddt term drops out, leaving div - laplacian, which is exactly the shape
// deviceSolveScalarTransport already assembles for k, epsilon, omega and the energy -- so the tracer
// goes through the case's own discretisation rather than a private one.
//
// LOOKED UP BY NAME, LATE. Everything this needs comes from the ObjectRegistry on FIRST EXECUTE, not
// from references captured at construction. That is OF's own arrangement (Time IS an objectRegistry;
// scalarTransport.C's transportedField() creates or finds its field on first use), and it is what makes
// construction order irrelevant: Time can be built at start-up -- so the functionObject report survives
// a later refusal -- while the solver it will drive is built 200 lines further down.
//
// NOT YET READY is not an error. If the registry has no solver at the first execute() the object simply
// defers; a steady solver registers before its loop, so this resolves on the first iteration.
//
// SCOPE, and what is refused rather than approximated:
//   * D: OF picks between a constant, a named nut field, and alphaD*nu + alphaDt*nut (scalarTransport.C
//     D()). ONLY the constant branch is built; the caller refuses the others BY NAME. Substituting a
//     constant for a nut-based diffusivity would change the answer while appearing to work.
//   * ddt: TRANSIENT IS supported. This object owns the field's old time levels (old/old2/ddt0,
//     device_ddt.cuh:95) and solvePassiveScalar advances them with the SOLVER's time scheme -- a case
//     has one time scheme, and choosing it is not a functionObject's business. On a steady solver the
//     term drops out, which is OF's ddt only under `ddtSchemes default steadyState`: OF's tracer takes
//     fvm::ddt(s) under a steady solver's Euler too (scalarTransport.C:341), so gpuSimpleFoam and
//     gpuRhoSimpleFoam refuse that pairing at start-up.
//
// PASSIVE: nothing here writes back into U, p, T, rho or the turbulence, so a case solves identically
// with the tracer present or absent. That is what makes it safe to run from the functionObject
// lifecycle rather than from inside the SIMPLE loop.

#include "brae_time.cuh"
#include "object_registry.cuh"
#include "device_simple_foam.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "brae_notice.cuh"
#include "scheme_parse.cuh"   // parseFieldDivScheme: OF scalarTransport.C:249
#include <filesystem>
#include <string>
#include <vector>

namespace brae {

class ScalarTransportFO : public FunctionObject
{
public:
    ScalarTransportFO(
        std::string name,
        std::string fieldName,
        std::string fieldPath,
        const ObjectRegistry& registry,
        scalar D,
        scalar relax,
        scalar tol,
        FieldDivScheme scheme)
      : name_(std::move(name)),
        fieldName_(std::move(fieldName)),
        fieldPath_(std::move(fieldPath)),
        registry_(registry),
        D_(D),
        relax_(relax),
        tol_(tol),
        scheme_(scheme)
    {}

    const std::string& name() const override { return name_; }

    // OF solves the transport equation in execute(), every time step.
    bool execute() override
    {
        if (failed_) return true;                     // already reported; do not re-report each step
        if (!ready_ && !initialise()) return true;   // registry not populated yet; try again next step
        // Old time levels are owned HERE, one set per tracer, and advanced inside solvePassiveScalar
        // using the solver's own scheme. A steady solver ignores them; a transient one needs them, which
        // is what makes this object safe to register on pimpleFoam.
        solver_->solvePassiveScalar(field_, boundary_, fieldName_.c_str(), Dbuf_, relax_, tol_,
                                    scheme_.bounded, scheme_.limited, scheme_.linearUpwind,
                                    scheme_.twoByk, scheme_.nonOrth, &old_, &old2_, &ddt0_);
        return true;
    }

    // Host copy for the writer, pulled only when a write is actually due -- the reason OF splits
    // write() from execute() -- so the tracer costs no device-to-host traffic per iteration.
    std::vector<scalar> hostField() const { return ready_ ? field_.host() : std::vector<scalar>(); }
    const std::string&  fieldName() const { return fieldName_; }
    bool                ready()     const { return ready_; }

private:
    // Resolve everything BY NAME, once. Returns false while the registry is still incomplete.
    bool initialise()
    {
        solver_ = registry_.lookupObject<DeviceSimpleSolver>("solver");
        const auto* fvp = registry_.lookupObject<std::vector<FvPatch>>("patches");
        const auto* m   = registry_.lookupObject<PrimitiveMesh>("mesh");
        if (!solver_ || !fvp || !m) return false;

        if (!std::filesystem::exists(fieldPath_))
        {
            noticeIgnored("functions/" + name_,
                          "scalarTransport field '" + fieldName_ + "' is not present at " + fieldPath_ +
                          ", so this tracer is NOT solved.");
            failed_ = true;
            return false;
        }

        const label nC = m->nCells();
        GeometricField<scalar> sf = buildField<scalar>(readField<scalar>(fieldPath_), *fvp, nC);
        sf.evaluateBoundary();
        field_.copyFrom(sf.internal);
        Dbuf_.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nC), D_));

        const auto* g = registry_.lookupObject<FvGeometry>("geometry");
        if (!g) return false;
        boundary_ = buildDeviceBoundary(sf, *fvp, *g);

        ready_ = true;
        return true;
    }

    std::string           name_;
    std::string           fieldName_;
    std::string           fieldPath_;
    const ObjectRegistry& registry_;
    scalar                D_;
    scalar                relax_;
    scalar                tol_;
    FieldDivScheme        scheme_;   // the case's own div(phi,<field>), never the momentum's

    DeviceSimpleSolver*   solver_ = nullptr;
    DeviceBuffer<scalar>  field_;
    DeviceBuffer<scalar>  Dbuf_;
    DeviceBoundary        boundary_;
    DeviceBuffer<scalar>  old_, old2_, ddt0_;   // fvm::ddt(s) levels, one set per tracer
    bool                  ready_  = false;
    bool                  failed_ = false;   // reported once; do not retry or re-report every step
};

}   // namespace brae
