#pragma once
// cf GPU offload (#4): the device-resident SIMPLE(+kEpsilon) solver as a reusable object. U/p/phi/k/eps/nut
// all live on the GPU between iterations; step() runs one SIMPLE iteration entirely on device:
//   faithful momentum  div(phi,U) - laplacian(nuEff,U) - fvc::div(nuEff*dev2(T(grad U)))   (BiCGStab/comp)
//   AMG-PCG pressure correction + conservative flux + corrector
//   (turbulent) device kEpsilon::correct()  [production -> wall functions -> eps/k -> nut]
// The loop body is the one validated machine-precision vs the CPU oracle in test_gpu_resident_turb. The
// AMG agglomeration is built once (static geometry); the coarse matrix is re-Galerkined each step (rAU
// changes). step() returns the OF residualControl signal (initial residuals of the U and p solves).
//
// nuEff at boundary faces = the adjacent-cell value (this path's convention, consistent with its laplacian
// boundary). The fully-faithful boundary-nut (nutkWallFunction) treatment lives in the host simple_foam.cuh.
#include "cf_types.cuh"
#include "function1.cuh"
#include "foam_field_reader.cuh"   // FieldData/PatchFieldData: the parsed p0 Function1
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "thermo_model.cuh"   // hConstHeToT: boundary he -> boundary T for the written field
#include "device_pcg.cuh"
#include "device_dilu.cuh"   // OF DILU preconditioner for the momentum BiCGStab
#include "device_simple.cuh"
#include "device_boundary.cuh"
#include "thermo_types.cuh"
#include "device_thermo.cuh"
#include "device_energy.cuh"
#include "rho_simple_controls.cuh"
#include "device_cyclic.cuh"
#include "device_ami.cuh"
#include "device_interface.cuh"   // interface<Op>() overloads dispatching to the cyclic/AMI backends
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"   // deviceS2/deviceF2/deviceNutSST for the startup validate() correctNut
#include "device_generalized_newtonian.cuh"   // laminar generalizedNewtonian/powerLaw viscosity
#include "device_smagorinsky.cuh"  // deviceSmagorinskyNut for the pure-LES (algebraic sub-grid nut) path
#include "device_coded_bc.cuh"     // NVRTC device-resident coded (runtime-compiled) boundary conditions
#include "cell_wall_dist.cuh"
#include "cell_max_delta.cuh"      // cellMaxDeltaXYZ -> hmax_ (SA-IDDES filter width, maxDeltaxyz)
#include "device_mrf.cuh"
#include "device_divdevreff.cuh"
#include "device_ddt.cuh"          // transient fvm::ddt(U) for the PIMPLE path (no-op in steady SIMPLE)
#include "device_amg.cuh"
#include "fv_options.cuh"
#include "device_fvoptions.cuh"
#include "solver_controls.cuh"
#include "time_controls.cuh"   // CourantNumbers   // NutWall, DeviceSimpleControls, DeviceSimpleResidual (moved out for reuse)
#include <cstdlib>
#include <functional>
#include <map>
#include <vector>
#include <cuda_runtime.h>

namespace brae {

// NutWall, DeviceSimpleControls, DeviceSimpleResidual now live in solver_controls.cuh (included above).

class DeviceSimpleSolver
{
public:
    DeviceSimpleSolver(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& fvp,
        const GeometricField<vector>& U,
        const GeometricField<scalar>& p,
        const SurfaceScalarField& phi,
        const DeviceSimpleControls& ctl,
        const GeometricField<scalar>* k = nullptr,
        const GeometricField<scalar>* eps = nullptr,
        const GeometricField<scalar>* nut = nullptr,
        const GeometricField<scalar>* ReThetat = nullptr,
        const GeometricField<scalar>* gammaInt = nullptr,
        // Whether `phi` came off disk. On a coupled patch that decides whether its boundary values are
        // the previous run's CONSERVATIVE interface flux or an un-coupled placeholder.
        bool                          phiWasRead = false);

    // OF turbulence-model load sequence, ported byte-for-byte (do NOT skip, this is why OF never blows up on a
    // case cf does): (1) the model ctor bounds the read fields  [kEpsilon.C:105-106 bound(k_,kMin_); bound(epsilon_,...);
    // kOmegaSSTBase.C:200-201 bound(k_,kMin_); bound(omega_,omegaMin_)]  and (2) simpleFoam.C:92 turbulence->validate()
    // -> eddyViscosity::validate() -> correctNut(), recomputing the INTERNAL nut from the bounded k/(omega|eps) BEFORE
    // iter 1. Without (2) the FIRST momentum predictor runs with nut straight from 0/nut, which is `uniform 0` in
    // motorBike AND pitzDaily, i.e. fully laminar; harmless on attached flows but the seed of divergence on high-Re
    // skewed meshes. kMin_/epsilonMin_/omegaMin_ default to SMALL (= 1e-15), matching the floors used in correct().
    void validateTurbulence();
    // E7: re-run OF's validate() ONCE the thermo exists, which is where OF does it.
    //
    // OF constructs the thermo, THEN the turbulence model, THEN calls validate() -> correctNut(). brae
    // constructs the solver (validate included) before the compressible driver has seeded the thermo, so
    // its startup correctNut saw a boundary velocity built from a PLACEHOLDER density: a
    // flowRateInletVelocity patch falls back to `rhoInlet` until a real rho exists, and on rhoTI that is
    // 1.0 against a true 1.161, giving an inlet U of 58.85 m/s instead of 50.69 against a uniform internal
    // 50. The resulting du/dx = 354 makes sqrt(S2) = 500.6 and trips the SST Bradshaw limiter, so
    // nut/(k/omega) came out 0.2477 where OF has 1 -- predicted to six decimals by that arithmetic before
    // any code changed. Invisible with `turbulence on` (the next iteration overwrites the seed), and the
    // answer with `turbulence off`.
    //
    // rhoBndSeed, when given, is the BOUNDARY half of a `rho` field read off disk. OF's createFields.H
    // builds rho with IOobject::READ_IF_PRESENT, so a restart resumes the STORED density rather than
    // recomputing thermo.rho(); the stored one carries rho.relax()'s history and is NOT reproducible
    // from p and T. Pass nullptr (or a wrong-sized vector) for OF's fresh-start fallback, thermo.rho().
    void revalidateAfterThermo(const std::vector<scalar>* rhoBndSeed = nullptr);
    // PIMPLE-foundation composable phase: turbulence transport (k/eps/omega/nuTilda -> nut). Uses only members,
    // so SIMPLE's step() and a future PIMPLE outer loop both call it (once per outer corrector) unchanged.
    void correctTurbulence();
    // PIMPLE-foundation composable phase 1: the momentum predictor -- assemble div(phi,U) - laplacian(nuEff,U)
    // - fvc::div(nuEff dev2(grad U)^T) + fvOptions/MRF/limiters + body force, relax, and solve each U component
    // (BiCGStab/GaussSeidel). Fills Uk_ + the shared members mDiagR/mUp/mLo/iC/bCb/relaxSrc (the relaxed matrix
    // + boundary coeffs + relaxed source) that correctPressureVelocity() reads. res gets the U-solve residuals.
    void solveMomentumPredictor(DeviceSimpleResidual& res);
    // PIMPLE-foundation composable phase: pressure-velocity coupling (SIMPLE/SIMPLEC corrector). Reads the
    // predictor outputs (member relaxed matrix + boundary coeffs + source), forms rAU/HbyA, solves the pressure
    // correction (with non-orth passes) and corrects U + the conservative face flux. res gets the p residuals.
    void correctPressureVelocity(DeviceSimpleResidual& res);

    DeviceSimpleResidual step();

    // --- Transient (PIMPLE) interface -- reuses the three composable phases above under an outer/inner-corrector loop,
    // with the implicit fvm::ddt(U) folded into solveMomentumPredictor. setDdtScheme() switches this solver from steady
    // (SIMPLE, the default) to transient; the steady step() path is completely unaffected (ddt stays a no-op).
    void setDdtScheme(DdtScheme s, scalar ocCoeff = scalar(1))
    {
        ddtScheme_ = s;
        ocCoeff_   = ocCoeff;
        if (s == DdtScheme::CrankNicolson)   // allocate + zero each transported variable's stored old ddt (ddt0)
        {
            const std::vector<scalar> z(nC_, scalar(0));
            for (int k = 0; k < 3; ++k) Uddt0_[k].copyFrom(z);
            if (ctl_.turbulent && !ctl_.les)
            {
                kddt0_.copyFrom(z);
                if (!ctl_.sa) e2ddt0_.copyFrom(z);
                if (ctl_.lm) { ReThetatddt0_.copyFrom(z); gammaIntddt0_.copyFrom(z); }
            }
        }
    }
    // Advance one time level: store U.oldTime()[.oldTime()] + roll deltaT -> deltaT0 (OF runTime++ / GeometricField::oldTime).
    // Move the mesh for one time step -- OF pimpleFoam.C:139-160, in OF's order:
    //     mesh.controlledUpdate();          move points, recompute geometry
    //     if (mesh.changing()) { ... fvc::makeRelative(phi, U); }
    // The mesh moves BEFORE the equations, and phi is made relative immediately after, so the
    // convective term never sees an absolute flux on a moving mesh.
    //
    // Every input is already verified independently: the transform (rigid invariants), the geometry
    // refresh (bit-identical to a rebuild, addressing preserved), meshPhi (SCL to 1e-20) and
    // makeRelative (relative flux vanishes, mutation-proven). This only composes them.
    //
    // Returns the per-face meshPhi so the caller can build movingWallVelocity from it.
    std::vector<scalar> moveMesh(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& fvp,
        const std::vector<vector>& oldPoints,
        const std::vector<vector>& newPoints,
        scalar deltaT);

    // fvc::makeRelative(phi, U) -- subtract mesh.phi() from every stored flux, INCLUDING the coupled
    // patches (cyc_/ami_.phi), which brae never reached. Called at the end of each pressure corrector,
    // where OF's pEqn.H calls it; a no-op until moveMesh has supplied a meshPhi. See the definition.
    void makeFluxRelative(const std::vector<FvPatch>& fvp, label nInternalFaces);
    // fvc::correctUf(Uf, U, phi) -- rebuild the face velocity from the current U and the ABSOLUTE flux.
    // Called at the end of each pressure corrector, before makeFluxRelative, exactly as pEqn.H does.
    void correctUf();
    // OF createUfIfPresent.H: the face velocity exists only on a dynamic mesh, and is constructed as
    // fvc::interpolate(U) at case setup -- BEFORE any motion, which is why the driver calls this once
    // before the time loop rather than letting the first mesh move trigger it. No mesh motion, no Uf,
    // and fvc::ddtCorr falls back to its phi.oldTime() form exactly as OF's does.
    void enableUf();
    // OF CorrectPhi (finiteVolume/cfdTools/general/CorrectPhi) + the pimpleFoam.C lines that wrap it.
    // Call after the mesh has moved and U's boundary has been refreshed, before the momentum predictor.
    // Returns the initial continuity residual of the pcorr solve (div(phi) before the projection).
    scalar correctPhi(const std::vector<FvPatch>& fvp, int nNonOrth);
    // How many times the turbulence transport was advanced, and how many outer correctors ran, since
    // the last reset. The PIMPLE contract is a statement about CADENCE -- turbulence once per time step
    // by default, outer loop possibly cut short by residualControl -- and a cadence cannot be asserted
    // from a converged field. Counting is the only way to test it.
    long turbCorrections()  const { return turbCorrections_; }
    long outerIterations()  const { return outerIterations_; }
    void resetLoopCounters() { turbCorrections_ = 0; outerIterations_ = 0; }
private:
    long turbCorrections_ = 0, outerIterations_ = 0;
    // PIMPLE residualControl state: outer-loop convergence flag and iteration-2 reference residuals.
    bool outerConverged_ = false;
    std::map<std::string, scalar> outerInitialResidual_;
    bool outerCriteriaSatisfied(const DeviceSimpleResidual& res, int oc);
public:
    // phi +/- mesh.phi() applied to one set of flux buffers (internal, boundary, cyclic, AMI).
    // sign -1 = fvc::makeRelative, +1 = fvc::makeAbsolute.
    void applyMeshPhi(const std::vector<FvPatch>& fvp, const std::vector<scalar>& mp, scalar sign,
                      DeviceBuffer<scalar>& fInt, DeviceBuffer<scalar>& fBnd,
                      DeviceBuffer<scalar>& fCyc, DeviceBuffer<scalar>& fAmi);

    // fvSolution PIMPLE/ddtCorr. OF defaults it to true and pimpleFoam's pEqn.H guards the whole
    // fvc::ddtCorr term with it, so it is the switch that makes brae's answer comparable to a run with
    // the correction deliberately turned off.
    void setDdtCorr(bool on) { ddtCorr_ = on; }

    // Maxwell: hand over the six components of the sigma field the driver read from 0/sigma (as six
    // scalar fields, which is how they are solved). Must be called before the first step when
    // `laminar { model Maxwell; }` is selected; the solver refuses to run the model without it.
    void setMaxwellSigma(const std::vector<GeometricField<scalar>>& sigma, const std::vector<FvPatch>& fvp,
                         const FvGeometry& g);
    // ...and read it back for writing (six components, cell values).
    std::vector<std::vector<scalar>> maxwellSigma() const
    {
        std::vector<std::vector<scalar>> out;
        for (int k = 0; k < 6; ++k) out.push_back(sig_[k].host());
        return out;
    }
    // The viscoelastic stress transport, OF Maxwell::correct(). Called from correctTurbulence().
    void correctMaxwell();

    // Overwrite one U patch's refValue -- the device half of movingWallVelocity's updateCoeffs.
    // OF does `vectorField::operator=(Uwall())` on the patch field; this is the same assignment, on the
    // buffers the momentum assembly reads.
    void setPatchVelocity(label patchi, const std::vector<vector>& Uw);

    // Rebuild the wall-function geometry from the CURRENT mesh. Everything in DeviceWallData is a
    // function of the geometry -- the near-wall distance, the face deltaCoeffs, and (the reason this
    // exists) the cyclicACMI area fractions that decide which non-overlap faces count as wall at all.
    // A sliding ACMI re-splits those areas every step, so a set built once at construction is stale
    // from the second step on. Call after the mesh has moved AND after any movingWallVelocity patch has
    // been assigned, since the wall velocities are read back out of the device boundary.
    void refreshWallData(const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& fvp);
    // Read-only view of that data, so a test can assert the refresh actually happened.
    const DeviceWallData& wallData() const { return wall_; }
    // Read-only view of the pressure boundary, so a test can check a per-step BC's contract (fixedMean's
    // area-weighted mean) against the values the solver actually imposed.
    const DeviceBoundary& pressureBoundary() const { return dbP_; }

    void advanceTime(scalar deltaT);
    // One PIMPLE time step: advanceTime, then nOuterCorrectors x { momentum predictor (ddt folded in); nCorrectors x
    // pressure-velocity correction; turbulence }, then continuity errors. Returns the outer-loop residual signal.
    // `moveMesh` is the mesh-update callback, invoked from INSIDE the outer loop at OpenFOAM's own
    // condition: `pimple.firstIter() || moveMeshOuterCorrectors` (pimpleFoam.C:140). It receives the
    // 0-based outer-iteration index. Passing nullptr leaves the loop static, which is every non-moving
    // case and every other solver that shares this method.
    //
    // The mesh update HAS to live inside the loop to be faithful. With moveMeshOuterCorrectors true the
    // mesh moves before every outer corrector, so the momentum and pressure equations of iteration 2
    // are assembled on different geometry than iteration 1 -- an outer loop that also converges the
    // mesh position. Hoisting it into the driver, as brae did, makes that flag unimplementable.
    DeviceSimpleResidual pimpleStep(scalar deltaT, int nOuterCorrectors, int nCorrectors,
                                    const std::function<void(int)>& moveMesh = nullptr);

    // Steady COMPRESSIBLE step (rhoSimpleFoam). Composes the same three phases as simpleStep/pimpleStep
    // with the energy equation and the thermo update inserted, in OF's rhoSimpleFoam.C order:
    //
    //     UEqn -> EEqn -> pEqn -> thermo.correct() -> rho.relax() -> turbulence
    //
    // EEqn sits between momentum and pressure because the pressure equation needs the density the NEW
    // temperature implies, not the previous one. Requires setCompressible() first.
    DeviceSimpleResidual rhoSimpleStep();

    // The compressible state, so the driver can seed T/he before the loop and read T back after it.
    // The solver owns the buffers (they are sized against its mesh); the driver owns what goes in them.
    DeviceThermo& thermo() { return th_; }
    const DeviceThermo& thermo() const { return th_; }

    // Switch the solver into compressible mode. dbHe is the enthalpy boundary (built from the case's 0/T
    // via deviceEnergyBoundaryFromT); th is allocated and seeded by the caller so the driver owns field
    // construction, exactly as it already owns U/p.
    // Per-boundary-face Prt from 0/alphat: alphatWallFunction patches carry their own (OF default 0.85),
    // every other face uses the turbulence model's (OF default 1.0). Optional -- absent, the model's is used.
    // OF functionObjects::scalarTransport (scalarTransport.C). Solves ONE passive scalar on the
    // device against the flux this solver already holds:
    //
    //     fvm::div(phi, s) - fvm::laplacian(D, s) == 0        (steady; ddt drops out)
    //
    // PASSIVE means passive: nothing here writes back into U, p, T, rho or the turbulence, so a case
    // that carries a tracer solves identically with and without it. That is also why it is safe to run
    // from a functionObject rather than from inside the SIMPLE loop proper.
    //
    // Reuses deviceSolveScalarTransport -- the same routine k, epsilon, omega and the energy already go
    // through -- so the tracer gets the case's own div/laplacian schemes rather than a private
    // discretisation. boundPositive is FALSE: OF's scalarTransport does not bound, and a tracer is not a
    // positive-definite turbulence quantity.
    void solvePassiveScalar(
        DeviceBuffer<scalar>& field,
        const DeviceBoundary& db,
        const char* fieldName,
        const DeviceBuffer<scalar>& D,   // built ONCE by the caller; a per-step H2D fill would be waste
        scalar relax,
        scalar tol,
        // The FIELD'S OWN convection scheme, from the case's div(phi,<field>) entry
        // (OF scalarTransport.C:249). Borrowing the momentum's flags instead ran a tracer through an
        // unbounded central scheme and reached 5.59 against a bound of 1.0.
        bool   schemeBounded,
        bool   schemeLimited,
        bool   schemeLinearUpwind,
        scalar schemeTwoByk,
        bool   schemeNonOrth,   // from laplacian(D<field>,<field>), not the solver's global setting
        // TRANSIENT support. OF's scalarTransport carries fvm::ddt(s), which needs the field's own old
        // time levels. Pass them and this builds the ddt from the SOLVER's own scheme and deltaT (there
        // is exactly one time scheme in a case, and it is not the functionObject's to choose) and
        // advances them here, mirroring advanceTime(). Omit them and the term drops out, which is
        // correct for a steady solver and WRONG for a transient one -- so a transient caller must pass
        // them or not register scalarTransport at all.
        DeviceBuffer<scalar>* old  = nullptr,
        DeviceBuffer<scalar>* old2 = nullptr,
        DeviceBuffer<scalar>* ddt0 = nullptr);

    void setAlphatPrt(const std::vector<scalar>& prtFace);
    // Energy linear solver from fvSolution solvers.(h|e). Was hardcoded tol=1e-10, relTol=0, BiCGStab.
    void setEnergySolver(scalar tol, scalar relTol, bool useGS)
    { eTol_ = tol; eRelTol_ = relTol; eUseGS_ = useGS; }
    void setCompressible(
        const ThermoCoeffs& tc,
        const RhoSimpleControls& rc,
        DeviceBoundary dbHe);

    // Enable an MRF rotating zone: builds the device frame data + makes the resident flux RELATIVE (so the
    // momentum/pressure are solved in the rotating frame). Call once after construction, before stepping.
    void setMRF(
        const MRFZone& z,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<std::string>& nonRotating = {});

    // Enable fvOptions (system/fvOptions). Empty data -> every hook below stays a no-op (bit-identical to no file),
    // exactly like fv::options::New on a case without the dict. Copies the precomputed per-cell source terms to the
    // device. Call once after construction. The momentum source feeds relaxSrc (so it reaches BOTH the predictor and
    // H()/HbyA, like the body force / OF UEqn.H fvOptions(U)); the implicit Sp lowers the momentum diagonal.
    void setFvOptions(const FvOptionsData& fo);
    // rotorDiskSource (BEM): the device rotor is built from the mesh in gpuSimpleFoam and handed over here.
    void setRotorDisk(DeviceRotorDisk r) { rotor_ = std::move(r); }
    // fvOptions scalar source for a transported scalar field (k/epsilon/...): src += Su, diag -= Sp.  No-op when absent.
    void fvOptionsAddSupScalar(
        const std::string& field,
        DeviceBuffer<scalar>* src,
        DeviceBuffer<scalar>* diag)
    {
        if (auto it = fvoScaSu_.find(field); it != fvoScaSu_.end() && src) deviceAxpy(1.0, it->second, *src);
        if (auto it = fvoScaSp_.find(field); it != fvoScaSp_.end() && diag) deviceAxpy(-1.0, it->second, *diag);
    }

    // pull device fields back to host (reconstruct for output / validation).
    std::vector<vector> U() const
    {
        const auto x=Uk_[0].host(), y=Uk_[1].host(), z=Uk_[2].host();
        std::vector<vector> u(nC_);
        for (label c=0;c<nC_;++c)
            u[c]={x[c],y[c],z[c]};
        return u;
    }
    std::vector<scalar> p()   const { return dp_.host(); }
    // Device-side p, for the compressible thermo update (which must not round-trip through the host).
    const DeviceBuffer<scalar>& pDevice() const { return dp_; }
    // Turbulence magnitude for the divergence tripwire: sum|k| + sum|second scalar|, as a DEVICE reduction
    // (no D2H of the fields). Used only to detect blow-up, never as a convergence measure.
    scalar turbSumMag() const
    {
        scalar s = 0;
        if (dk_.size()) s += deviceSumMag(dk_);
        if (de_.size()) s += deviceSumMag(de_);
        return s;
    }
    // Per-face turbulent-inlet data, so the boundary values can be refreshed per iteration (OF parity).
    void setTurbulentInlets(const std::vector<label>& tiMask, const std::vector<scalar>& tiIntensity,
                            const std::vector<label>& mlMask, const std::vector<scalar>& mlLength)
    {
        label nTi = 0, nMl = 0;
        for (label m : tiMask) if (m) ++nTi;
        for (label m : mlMask) if (m) ++nMl;
        // Positive confirmation, because the failure mode here is SILENT and expensive: a
        // turbulentIntensityKineticEnergyInlet that is never refreshed sits at whatever the file's
        // `value` entry says, and OpenFOAM tutorials routinely write `value $internalField` there. On
        // pipeCyclic that is k = 1, against the 0.0038-0.0067 the intensity actually implies -- a 200x
        // inlet that feeds the whole entrance region and decays only by the pipe exit.
        std::printf("  turbulent inlets: %d face(s) intensity-based k, %d face(s) mixing-length "
                    "epsilon/omega, refreshed every iteration\n", (int)nTi, (int)nMl);
        if (!nTi && !nMl) return;
        tiMask_.copyFrom(tiMask); tiIntensity_.copyFrom(tiIntensity);
        mlMask_.copyFrom(mlMask); mlLength_.copyFrom(mlLength);
        hasTurbInlet_ = true;
    }
    std::vector<scalar> k()   const { return dk_.host(); }
    std::vector<scalar> eps() const { return de_.host(); }
    std::vector<scalar> ReThetat() const { return ReThetat_.host(); }   // kOmegaSSTLM
    std::vector<scalar> gammaInt() const { return gammaInt_.host(); }
    std::vector<scalar> nut() const { return dnut_.host(); }
    // SA: the per-boundary-face nutUSpaldingWallFunction wall nut (empty for non-SA), for the viscous force.
    std::vector<scalar> nutWall() const { return dnutBndWall_.size() ? dnutBndWall_.host() : std::vector<scalar>(); }
    std::vector<scalar> cellY() const { return y_.size() ? y_.host() : std::vector<scalar>(); }   // cell wall distance (SST/SA)
    // diagnostics: the conservative face flux (internal then boundary) for continuity-error localisation.
    // Courant number on the DEVICE, coupled interfaces included, returning two scalars.
    //
    // The driver used to pull phiInternal() and phiBoundary() to the host every step of an adaptive run
    // and loop over faces and cells serially -- and it read only those two arrays, so cyclic and AMI
    // flux never entered the sum at all. That is a correctness bug before it is a performance one: the
    // Courant number was understated on exactly the cells a rotating interface makes fastest, and on an
    // adjustTimeStep case the understated maxCo chooses the NEXT deltaT.
    CourantNumbers courantNumbers(scalar deltaT) const;
    std::vector<scalar> phiInternal() const { return phiInt_.host(); }
    // The SOLVED boundary values, so the written boundaryField reports what the solve used rather than
    // echoing the input file. nut is the one that matters most: on a wall its value comes from the wall
    // function, and post-processing (yPlus, wall shear, force split) reads it straight back.
    std::vector<scalar> nutBoundary() const
    {
        return nutBndAll_.size() ? nutBndAll_.host() : std::vector<scalar>();
    }
    // Generic: evaluate any cell field on its own boundary descriptor.
    std::vector<scalar> boundaryOf(const DeviceBoundary& db, const DeviceBuffer<scalar>& f) const
    {
        if (!db.n || !f.size()) return {};
        DeviceBuffer<scalar> b;
        deviceBCValue(db, f, b);
        return b.host();
    }
    // T on the boundary. he is the solved variable, so the boundary T is T(he_b) -- evaluating he's own
    // BC (which now resolves inletOutlet per iteration) and converting, rather than echoing 0/T.
    // T at boundary faces, for the field WRITER and the initialisation dumps.
    //
    // This inverts he_b for the same reason the solver's own boundary path does, and it must use the same
    // thermodynamics -- it did not. On the liquid path it ran the perfect-gas closed form over a liquid
    // he_b of ~-1.56e7 J/kg and wrote nonsense into every `T` boundaryField brae produced, including the
    // init_Tb dump that step 6's "boundary matches OF" check was read from. The solve was unaffected
    // (nothing reads this back), which is precisely why it survived: a wrong number on an output path
    // that no assertion covered.
    // grad(U) on the CURRENT U, for comparing brae's gradient operator against OF's `postProcess -func
    // grad(U)` on byte-identical input. Called at set-up (before any equation is solved) so both sides
    // see the same field -- a diff taken inside turbulence.correct() compares a post-momentum-predictor
    // U against OF's untouched initial one and measures the solve, not the operator.
    std::vector<scalar> gradUField()
    {
        DeviceBuffer<scalar> g;
        deviceGradU(dm_, dbU_, Uk_[0], Uk_[1], Uk_[2], g,
                    hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
        return g.host();
    }
    std::vector<scalar> TBoundary() const
    {
        if (!compressible_ || !dbHe_.n || !th_.he.size()) return {};
        // Prefer the boundary temperature the thermo already established this correct(). Re-deriving it
        // here would be a SECOND inversion with its own answer: on a fixedValue patch the solver holds
        // the prescribed value exactly, while an independent Newton lands a residual away from it, so
        // brae would write 349.999999999999 for a wall OF writes as 350. Two code paths for one quantity
        // is the same defect as the four boundary property kernels, just smaller.
        if (TBnd_.size() == static_cast<std::size_t>(dbHe_.n)) return TBnd_.host();
        DeviceBuffer<scalar> heB;
        deviceBCValue(dbHe_, th_.he, heB);
        const std::vector<scalar> h = heB.host();
        std::vector<scalar> t(h.size());
        if (tc_.model == ThermoModel::liquidH2O)
        {
            // e = h(T) - p/rho(T) needs p at the same faces; Hs does not, and passes 0.
            std::vector<scalar> pb;
            if (tc_.internalEnergy && dbP_.n == dbHe_.n)
            {
                DeviceBuffer<scalar> pB;
                deviceBCValue(dbP_, dp_, pB);
                pb = pB.host();
            }
            const EnergyForm form = tc_.internalEnergy ? EnergyForm::sensibleInternalEnergy
                                                       : EnergyForm::sensibleEnthalpy;
            for (std::size_t i = 0; i < h.size(); ++i)
                t[i] = h2oEnergyToT(form, h[i], i < pb.size() ? pb[i] : scalar(0), 300.0).T;
        }
        else
        {
            for (std::size_t i = 0; i < h.size(); ++i) t[i] = hConstHeToT(h[i], tc_);
        }
        // fixesValue LAST, and on BOTH paths. A prescribed temperature is an input, so it is copied
        // rather than recovered -- an inversion would land a Newton residual away from it and write
        // 349.999999999999 for a wall OF writes as 350. Applied here and not only in the branch above
        // because this fallback runs before the first correct() has filled TBnd_, which is exactly when
        // the initialisation dump is taken.
        if (TFixMask_.size() == h.size() && TFixBnd_.size() == h.size())
        {
            const std::vector<label>  msk = TFixMask_.host();
            const std::vector<scalar> fix = TFixBnd_.host();
            for (std::size_t i = 0; i < h.size(); ++i) if (msk[i]) t[i] = fix[i];
        }
        return t;
    }
    // U on the boundary: each component evaluated against its own descriptor, then recombined. Needed
    // because a vector BC can be per-component (slip/symmetry set vf = |n_k| per component), so there is
    // no single scalar evaluation that covers it.
    std::vector<vector> UBoundary() const
    {
        if (!dbU_.comp[0].n) return {};
        DeviceBuffer<scalar> b0, b1, b2;
        deviceBCValue(dbU_.comp[0], Uk_[0], b0);
        deviceBCValue(dbU_.comp[1], Uk_[1], b1);
        deviceBCValue(dbU_.comp[2], Uk_[2], b2);
        const std::vector<scalar> h0 = b0.host(), h1 = b1.host(), h2 = b2.host();
        std::vector<vector> v(h0.size());
        for (std::size_t i = 0; i < h0.size(); ++i) v[i] = vector{h0[i], h1[i], h2[i]};
        return v;
    }
    // nuTilda shares the k slot on the SA path.
    std::vector<scalar> nuTildaBoundary() const { return boundaryOf(dbK_, dk_); }
    std::vector<scalar> kBoundary()   const { return boundaryOf(dbK_,   dk_); }
    std::vector<scalar> epsBoundary() const { return boundaryOf(dbEps_, de_); }
    std::vector<scalar> pBoundary()   const { return boundaryOf(dbP_,   dp_); }
    std::vector<scalar> phiBoundary() const { return phiBnd_.host(); }

    // THE INTERFACE FLUX, per coupled patch, for the phi write. A cyclic/AMI patch's flux is NOT in
    // phiBoundary -- it lives on the interface object (cyc_.phi / ami_.phi), because that is what the
    // coupling reads. OpenFOAM keeps it in phi's boundaryField and WRITES it, so its restart resumes the
    // exact flux; brae wrote the patch with a type and no value, so the flux could not survive a restart
    // and was rebuilt from U instead. It is not the same number: on
    // pimpleFoam/RAS/oscillatingInletACMI2D the rebuild differed from the stored flux by 1.3e-04 on the
    // first face, and since the momentum interface coefficient is upwind (max(phi,0)) that landed
    // directly on the diagonal -- 0.95% of rAU on every one of the 40 source-side interface cells, and
    // nothing else in the mesh.
    // Returns (fvPatch index, per-face flux in patch face order); empty when there is no interface.
    std::vector<std::pair<label, std::vector<scalar>>> interfacePatchFlux() const;
    // The inverse, for a restart: seed cyc_.phi / ami_.phi from the values just read out of phi's file.
    void setInterfacePatchFlux(const std::vector<std::pair<label, std::vector<scalar>>>& byPatch);
    // rho on the boundary -- the EOS result the solve actually used, already maintained per face by the
    // pressure equation (phiHbyA is rho-weighted with it). Not derivable from the T/p descriptors here,
    // which is why it needs its own accessor rather than boundaryOf().
    std::vector<scalar> rhoBoundary() const { return compressible_ ? rhoBnd_.host() : std::vector<scalar>{}; }

    // CrankNicolson ddt0 (stored old ddt) persistence -- get/set for seamless restart (the driver writes+reads these).
    bool isCrankNicolson() const { return ddtScheme_ == DdtScheme::CrankNicolson; }
    bool cnWarm() const { return cnWarm_; }
    void setCnWarm(bool w) { cnWarm_ = w; }
    // Prime the CrankNicolson time state on restart: deltaT_=deltaT0 makes the FIRST advanceTime set deltaT0_=deltaT0
    // (so the first post-restart step is full CN, not the Euler bootstrap), and the ddt0 recurrence is already warm.
    void setCnRestart(scalar deltaT0) { deltaT_ = deltaT0; cnWarm_ = true; }

    // NVRTC device-coded BC (codedFixedValue on U): compile `code` once into a device kernel over the patch's boundary
    // faces [offset, offset+count) in the flat boundary order. Applied each step in solveMomentumPredictor (writes the
    // patch's dbU_ refValue on the device). setTime feeds the current time to the snippet's `t`.
    // target: 0=U (vector), 1=p, 2=k/nuTilda, 3=second (omega/epsilon). mixed=true -> codedMixed (also writes valueFraction).
    void addCodedBC(const std::string& name, const std::string& code, int offset, int count, int target = 0, bool mixed = false);
    void setTime(scalar t) { time_ = t; }

    // uniformTotalPressure: p0 is a Function1 of time, so the device p0 must be refreshed each step
    // before deviceUpdateTotalPressure recomputes p_b. OF does the same -- updateCoeffs() calls
    // p0_->value(time) every step (uniformTotalPressureFvPatchScalarField.C:149). One entry per patch
    // that declared a non-constant p0; a constant p0 needs none and keeps the value set at read time.
    struct TimeVaryingP0 { label start = 0, count = 0; Function1 f; };
    // fanPressure: p0 is shifted every step by the fan's pressure rise at the CURRENT patch flow rate.
    //   volFlowRate = dir*sum(phi_patch),  dir = -1 for `direction in`, +1 for `out`
    //   p0_eff      = p0 - dir*fanCurve(max(volFlowRate, 0))
    // (fanPressureFvPatchScalarField::updateCoeffs, which then defers to totalPressure with that p0.)
    // The curve is (flowRate, deltaP) pairs, linearly interpolated and CLAMPED outside its range, which is
    // OF's `outOfBounds clamp` default for a table.
    struct FanPressure
    {
        label  start = 0, count = 0;
        scalar p0 = 0, dir = -1;
        std::vector<std::pair<scalar, scalar>> curve;
        scalar dp(scalar q) const
        {
            if (curve.empty()) return 0;
            if (q <= curve.front().first) return curve.front().second;
            if (q >= curve.back().first)  return curve.back().second;
            for (std::size_t i = 1; i < curve.size(); ++i)
                if (q <= curve[i].first)
                {
                    const scalar t = (q - curve[i-1].first) / (curve[i].first - curve[i-1].first);
                    return curve[i-1].second + t*(curve[i].second - curve[i-1].second);
                }
            return curve.back().second;
        }
    };
    void setFanPressure(std::vector<FanPressure> v) { fanP0_ = std::move(v); }
    // fixedMean (OF fixedMeanFvPatchField): a fixedValue whose face values are the ADJACENT CELL values,
    // shifted or scaled so their area-weighted mean equals a prescribed one:
    //     psi   = patchInternalField()
    //     mean  = sum(magSf*psi)/sum(magSf)
    //     |target| > SMALL and |mean| > 0.5|target| : psi *= |target|/|mean|
    //     otherwise                                : psi += (target - mean)
    // It is built as a plain fixedValue and its refValue recomputed here every step, the same way
    // codedFixedValue and fanPressure are.
    struct FixedMean { label start = 0, count = 0; scalar meanValue = 0; };
    void setFixedMean(std::vector<FixedMean> v) { fixedMean_ = std::move(v); }
    void setTimeVaryingP0(std::vector<TimeVaryingP0> v) { tvP0_ = std::move(v); }

    // Build the list from a pressure field's parsed boundary entries. Shared so all three drivers wire
    // it identically -- three hand-rolled loops over patch offsets is how the copies in this codebase
    // have historically drifted.
    //
    // The offsets must be the DeviceBoundary face order, which buildDeviceBoundary produces by walking
    // `patches` in order and emitting patch.size faces each. Getting that wrong would ramp the wrong
    // patch's p0, so it is derived from the same traversal rather than assumed.
    static std::vector<TimeVaryingP0> collectTimeVaryingP0(
        const FieldData<scalar>& pFd,
        const std::vector<FvPatch>& patches)
    {
        std::vector<TimeVaryingP0> out;
        label start = 0;
        for (const FvPatch& fp : patches)
        {
            // dbP_.p0 is in DeviceBoundary face order, which SKIPS the coupled patches -- so the running
            // offset has to skip them too, or every entry after a cyclic/AMI patch addresses the wrong faces.
            if (isCoupledInterfaceType(fp.type)) continue;
            for (const auto& b : pFd.boundary)
            {
                if (b.name != fp.name || !b.hasP0Function1) continue;
                TimeVaryingP0 e;
                e.start = start;
                e.count = fp.size;
                e.f     = b.p0Function1;
                out.push_back(std::move(e));
                break;
            }
            start += fp.size;
        }
        return out;
    }
    std::vector<vector> Uddt0() const
    {
        const auto x = Uddt0_[0].host(), y = Uddt0_[1].host(), z = Uddt0_[2].host();
        std::vector<vector> r(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) r[i] = vector{x[i], y[i], z[i]};
        return r;
    }
    void setUddt0(const std::vector<vector>& v)
    {
        std::vector<scalar> x(v.size()), y(v.size()), z(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) { x[i] = v[i].x; y[i] = v[i].y; z[i] = v[i].z; }
        Uddt0_[0].copyFrom(x); Uddt0_[1].copyFrom(y); Uddt0_[2].copyFrom(z);
    }
    std::vector<scalar> kddt0()  const { return kddt0_.host(); }
    void setKddt0(const std::vector<scalar>& v)  { kddt0_.copyFrom(v); }
    std::vector<scalar> e2ddt0() const { return e2ddt0_.host(); }
    void setE2ddt0(const std::vector<scalar>& v) { e2ddt0_.copyFrom(v); }
    std::vector<scalar> reThetatddt0() const { return ReThetatddt0_.host(); }
    void setReThetatddt0(const std::vector<scalar>& v) { ReThetatddt0_.copyFrom(v); }
    std::vector<scalar> gammaIntddt0() const { return gammaIntddt0_.host(); }
    void setGammaIntddt0(const std::vector<scalar>& v) { gammaIntddt0_.copyFrom(v); }

private:
    const std::vector<FvPatch>& fvp_;
    DeviceSimpleControls ctl_;
    label nC_, nIf_;
    DeviceMesh dm_;
    DeviceVectorBoundary dbU_;
    DeviceBoundary dbP_, dbK_, dbEps_, dbExtrap_, dbReThetat_, dbGammaInt_;   // dbReThetat_/dbGammaInt_: kOmegaSSTLM
    // pcorr (OF CorrectPhi): p's geometry, zeroGradient except fixedValue 0 where p fixes a value.
    DeviceBoundary dbPcorr_;
    DeviceWallData wall_;
    DeviceMRF mrf_;                       // optional rotating zone (inactive by default)
    // fvOptions (empty by default -> no-op). Momentum source per component (relaxSrc) + implicit Sp (diagonal);
    // scalar sources keyed by field name. See setFvOptions / the unconditional hooks in step().
    bool   hasFvoMom_ = false;
    int fvoCount_ = 0;
    DeviceBuffer<scalar> fvoMomSu_[3], fvoMomSp_;
    std::map<std::string, DeviceBuffer<scalar>> fvoScaSu_, fvoScaSp_;
    DevicePorosity por_;                  // explicitPorositySource (DarcyForchheimer), evaluated each iter
    // meanVelocityForce: accumulated pressure gradient gradP_ driving the mean velocity to Ubar (channel flow).
    bool   mvfActive_ = false;
    vector mvfFlowDir_{0,0,0};
    // meanVelocityForce keeps TWO gradients, exactly as OF does (meanVelocityForce.C:146-247), and the
    // pair is not an implementation detail -- collapsing them into one accumulator makes the channel
    // diverge. See the constrain step in the momentum assembly.
    scalar mvfUbarMag_ = 0, mvfRelax_ = 1.0, mvfGradP0_ = 0, mvfDGradP_ = 0, mvfVtot_ = 0;
    DeviceBuffer<scalar> mvfMaskV_, mvfMask01_;   // V (and 1) in the zone, 0 else (= V / ones for selectionMode all)
    bool   limUActive_ = false;
    scalar limUMax_ = 0;
    DeviceBuffer<label> limUCells_;   // limitVelocity clamp
    // limitTemperature. Held as HE limits, not T limits: OF converts once with the case thermo and clamps
    // the energy variable the equation actually solved (limitTemperature::correct(he)).
    bool   limTActive_ = false;
    bool   limTAllCells_ = false;     // selectionMode all -> the he BOUNDARY is clamped too
    scalar limTMin_ = 0, limTMax_ = 0;
    DeviceBuffer<label> limTCells_;
    // fixedTemperatureConstraint: OF constrains the ENERGY matrix (eqn.setValues), so this is carried as a
    // per-cell mask + per-cell he value and handed to the same setValues path the eps wall function uses.
    bool                fixTActive_ = false;
    DeviceWallData      fixTMask_;    // only isWallCell is used by the setValues kernels
    DeviceBuffer<scalar> fixTHe_;
    // scalarFixedValueConstraint on k / epsilon|omega. Per-cell mask + per-cell value, handed to the same
    // setValues path; OF applies it before the eps wall function (kEpsilon.C:266 then :267).
    bool                 fixScaK_ = false, fixScaE_ = false;
    DeviceBuffer<label>  fixScaMask_;
    DeviceBuffer<scalar> fixScaKVal_, fixScaEVal_;
    bool   vdcActive_ = false;
    scalar vdcUMax_ = 0, vdcC_ = 1;
    DeviceBuffer<label> vdcCells_;   // velocityDampingConstraint
    // actuationDiskSource (Froude): thrust over a disk cellZone, computed each iter from the monitored upstream U.
    // One entry per actuationDiskSource. Each turbine carries its OWN masks: sharing them across disks
    // would apply every turbine's thrust to the union of their cells, which converges to a plausible
    // but wrong wind field rather than failing.
    struct ActuationDiskDev
    {
        vector diskDir{1,0,0};
        scalar area = 0, a = 0, vtot = 0, nmon = 0;
        DeviceBuffer<scalar> maskVDisk, monMask01;   // V in disk cells / 1 in monitor cells
    };
    bool adActive_ = false;
    std::vector<ActuationDiskDev> adDisks_;
    DeviceRotorDisk rotor_;                            // rotorDiskSource (BEM); per-cell force recomputed each iter
    DeviceBuffer<scalar> rotorFx_, rotorFy_, rotorFz_;
    AMGData amg_;
    DeviceBuffer<scalar> Uk_[3], dp_, phiInt_, phiBnd_, dk_, de_, dnut_, y_;   // y_ = cell wall distance (SST/SA)
    DeviceBuffer<scalar> hmax_;   // per-cell maxDeltaxyz (IDDES filter width); built only when ctl_.iddes
    // The LES filter width the case's `delta` entry selects, when it is not cubeRootVol. Empty otherwise,
    // and every consumer then falls back to cbrt(V) in its own kernel -- so a case that says nothing is
    // byte-for-byte unchanged. Shared by the sub-grid model, the DES length scale AND the convection
    // scheme, which must all agree about the resolved scale.
    DeviceBuffer<scalar> lesDelta_;
    // Maxwell viscoelastic stress: the six components of sigma with their own boundaries, plus the
    // per-outer-corrector production. Empty on every other case, and every Maxwell branch is gated on
    // ctl_.maxwell, so a Newtonian run is untouched.
    DeviceBuffer<scalar> sig_[6];
    DeviceBoundary       dbSig_[6];
    DeviceBuffer<scalar> sigOld_[6], sigOld2_[6], sigddt0_[6];   // fvm::ddt(sigma) old levels (backward/CN)
    // OF wallDist::n() packed 3 x nC: the nearest wall face's OUTWARD unit normal. Built only for
    // ZDES2020 shielding, which is the only consumer; empty otherwise.
    DeviceBuffer<scalar> wallN_;
    DeviceBuffer<scalar> hwn_;    // per-cell wall-normal grid spacing (IDDES delta 3rd term); built only when ctl_.iddes
    // Transient (PIMPLE) state: ddt scheme + time steps + old-time velocity levels (U.oldTime()[.oldTime()]), rotated by
    // advanceTime(). steadyState + empty old-time buffers in the default steady SIMPLE path, where ddt is a no-op.
    DdtScheme ddtScheme_ = DdtScheme::steadyState;
    std::vector<TimeVaryingP0> tvP0_;   // uniformTotalPressure p0(t), empty for constant p0
    std::vector<FanPressure>   fanP0_;  // fanPressure: p0 shifted by the fan curve each step
    std::vector<FixedMean>     fixedMean_;   // fixedMean: refValue re-derived from the adjacent cells each step
    scalar    deltaT_ = 0, deltaT0_ = 0;
    scalar    ocCoeff_ = 1;         // CrankNicolson off-centring (fvSchemes "CrankNicolson <oc>"); 0=Euler-like, 1=pure CN
    bool      cnWarm_  = false;     // CrankNicolson: the ddt0 recurrence has done its first (Euler-startup) update
    // NVRTC device-coded BCs (codedFixedValue). bndCf* = flat boundary face centres (patch order, cyclic excluded).
    struct SolverCodedBC { CodedBcKernel kernel; int offset = 0, count = 0; int target = 0; bool mixed = false; };   // target: 0=U 1=p 2=k 3=second
    std::vector<SolverCodedBC> codedBCs_;
    DeviceBuffer<scalar> bndCfX_, bndCfY_, bndCfZ_;
    scalar    time_ = 0;            // current time, fed to the coded-BC snippet's `t`
    DeviceBuffer<scalar> Uold_[3], Uold2_[3];
    // Turbulence old-time levels for the URANS fvm::ddt(k/eps/omega/nuTilda): dk_ (k or nuTilda) + de_ (epsilon or omega).
    DeviceBuffer<scalar> kOld_, kOld2_, e2Old_, e2Old2_;
    DeviceBuffer<scalar> ReThetatOld_, ReThetatOld2_, gammaIntOld_, gammaIntOld2_;   // kOmegaSSTLM transition old-time
    DeviceBuffer<scalar> Uddt0_[3], kddt0_, e2ddt0_, ReThetatddt0_, gammaIntddt0_;   // CrankNicolson stored old ddt (per variable); allocated only for CN
    // Momentum-predictor outputs, shared with the pressure-velocity phase (PIMPLE foundation: solveMomentumPredictor()
    // fills them once per outer corrector; correctPressureVelocity() reads them). Members so both phases see them + to
    // avoid per-step reallocation. mDiagR/mUp/mLo = relaxed momentum matrix; iC/bCb = per-component boundary coeffs;
    // relaxSrc = relaxed momentum source (feeds both the predictor solve AND H()/HbyA).
    DeviceBuffer<scalar> mDiagR, mUp, mLo;
    DeviceBuffer<scalar> iC[3], bCb[3], relaxSrc[3];
    DeviceBuffer<scalar> gx, gy, gz;   // gradient workspace: grad(U) in the predictor, reused as grad(p) in the pressure phase
    DeviceBuffer<scalar> ReThetat_, gammaInt_, gammaIntEff_;                   // kOmegaSSTLM transition fields
    DeviceBuffer<scalar> dnutBndWall_;                                          // persistent Spalding wall-nut (SA warm seed)
    DeviceBuffer<scalar> nuConst_, zeroSrc_, zeroBndU_, ones_;

    // Compressible state (rhoSimpleFoam). compressible_ stays false for every incompressible case, so
    // none of this is reachable from the existing solvers and the steady/PIMPLE paths are untouched.
    bool compressible_ = false;
    ThermoCoeffs tc_{};
    RhoSimpleControls rc_{};
    DeviceThermo th_{};
    DeviceBoundary dbHe_{};
    DeviceBuffer<scalar> pD_, pU_, pL_, diagCp_, bp_;            // persistent pressure matrix (graph-stable addresses)
    DeviceBuffer<label>  bndIsWall_, adjustMask_;              // wall mask (true boundary nut); adjustPhi adjustable mask
    // kEpsilon: 1 on boundary faces whose 0/nut BC is 'calculated'. OF fills those by field assignment
    // (Cmu*k_b^2/eps_b) rather than extrapolating the cell, and the two differ by >12x at a fixed-k inlet.
    DeviceBuffer<label>  nutCalcMask_;
    // OF fvMesh::movePoints does storeOldVol(V()) BEFORE moving: the ddt source needs V^{n-1} while the
    // diagonal uses V^n. Empty on a fixed mesh, where both are the same buffer and nothing changes.
    DeviceBuffer<scalar> Vold_;
    std::vector<label> patchStart_;   // DeviceBoundary face offset per patch
    DeviceBuffer<scalar> nutBndFile_;   // nut's own boundaryField (OF reads nut_b from here, not cells)
    bool hasNutCalc_ = false;
    bool hasNutFixed_ = false;
    DeviceBuffer<label> nutCalcSel_;   // complement of nutCalcMask_ (0 on calc faces), for deviceSelectFixedFlux
    DeviceBuffer<label> nutFixedMask_;   // 0 where nut's BC fixes a value on a non-wall patch
    DeviceBuffer<scalar> bndY_, nuBndConst_;                    // nearWallDist y per boundary face; nu over bnd faces
    // Compressible only: per-boundary-face rho and laminar mu, refreshed each outer iteration from the
    // boundary p/he. OF gets these from rho.boundaryField()[patchi] and transport_.mu(patchi); they feed
    // muEff_b = mu_b + rho_b*nut_b and the kinematic nu_b = mu_b/rho_b the wall functions ask for.
    // SIMPLEC: (rAtU - rAU) for the HbyA correction, rho*(rAtU - rAU) for the FLUX correction on the
    // compressible path (OF pcEqn.H). Identical buffers when incompressible.
    DeviceBuffer<scalar> drAtUFlux_;
    scalar eTol_ = 1e-10, eRelTol_ = 0.0;
    bool   eUseGS_ = false;
    DeviceBuffer<scalar> rhoBnd_, muBnd_, nuWallBnd_;
    // T at boundary faces, refreshed from he_b immediately before the boundary properties that consume
    // it -- OF heRhoThermo::calculate establishes T_b once per patch and then evaluates rho/mu/alphah
    // from it, rather than re-inverting he_b inside each one.
    DeviceBuffer<scalar> TBnd_;
    // OF's fvPatchField::fixesValue(), captured from the T boundary at set-up: 1 where T is PRESCRIBED,
    // with the prescribed value alongside. Those faces keep T_b exact and have he_b re-derived from the
    // live p_b every correct(), instead of inverting an he_b frozen at the initial pressure.
    DeviceBuffer<label>  TFixMask_;
    DeviceBuffer<scalar> TFixBnd_;
public:
    // Both vectors are in DeviceBoundary face order and must be dbHe_.n long, or they are ignored.
    void setFixedBoundaryT(const std::vector<label>& mask, const std::vector<scalar>& value)
    {
        if (mask.size() != value.size()) return;
        TFixMask_.copyFrom(mask);
        TFixBnd_.copyFrom(value);
    }
private:
    // The BOUNDARY half of the solver's own `rho` field -- the one OF relaxes. rhoBnd_ above is
    // thermo.rho() at the boundary, recomputed fresh from p_b and T_b every time it is asked for, which
    // is right for everything that reads thermo.rho() (nu(patchi), alphaEff(patchi), the turbulence
    // model). It is WRONG for `phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA)`, which reads the solver's
    // rho field, and that field is assigned once per outer iteration and then relaxed.
    DeviceBuffer<scalar> rhoBndP_, rhoBndPPrev_;
    // BRAE_PTRACE: which SIMPLE iteration the pressure stage trace is on. Advanced once per call to
    // correctPressureVelocity, read by the pre-limit dump, so P0..P6 of one iteration share an index.
    int pTraceIt_ = 0;
    // pEqn.relax() source (D - D0)*p. Absolute, so it is added to bp_ AFTER deviceFoldPressure (which
    // multiplies divPhi by the cell volume); a member so the fold site can see it.
    DeviceBuffer<scalar> pRelaxSrc_;
    // flowRateInletVelocity, massFlowRate form: OF recomputes avgU = -mdot/gSum(rho*magSf) every call, so
    // it moves with the solution. frMagSf_ is magSf masked to the flowRate patches (0 elsewhere), making
    // the patch sum a single dot product against the live boundary rho; frN_ holds the outward normals.
    struct FlowRatePatch { scalar mdot; };
    std::vector<FlowRatePatch> frPatches_;
    std::vector<DeviceBuffer<scalar>> frMagSf_;
    DeviceBuffer<scalar> frNx_, frNy_, frNz_;
    // turbulentIntensityKineticEnergyInlet / turbulentMixingLength*Inlet: per-face masks + coefficients so
    // the inlet k and epsilon|omega can be REBUILT each outer iteration, as OF's updateCoeffs does. Frozen
    // set-up values are exact only when the inlet U never moves; flowRateInletVelocity moves it.
    DeviceBuffer<label>  tiMask_;      // 1 where k has a turbulentIntensityKineticEnergyInlet face
    DeviceBuffer<scalar> tiIntensity_;
    DeviceBuffer<label>  mlMask_;      // 1 = epsilon, 2 = omega
    DeviceBuffer<scalar> mlLength_;
    bool hasTurbInlet_ = false;
    bool hasFlowRate_ = false;
    DeviceBuffer<label>  wfBndIdx_;   // wall-face -> boundary-face index (built once)
    DeviceBuffer<scalar> wfNu_;       // nu = mu_b/rho_b gathered onto wall faces, for omegaWallFunction/G0
    DeviceBuffer<scalar> nutBndAll_;  // nut at boundary faces (wall-function value on walls), for alphat_b
    DeviceBuffer<scalar> prtBnd_;     // per-face Prt: the alphatWallFunction's on its patches, the model's elsewhere
    // any inletOutlet U patch present -> HbyA must keep its extrapolated boundary there (constrainHbyA
    // skips assignable patches, and inletOutlet is one). Costs three zero-gradient evaluations per
    // corrector, so it is gated on the case actually having such a patch.
    bool   hasInletOutletU_ = false;
    bool   hasMixed_ = false;
    bool   hasWedge_ = false;   // any wedge (axisymmetric constraint) U patch -> per-step rotated value                                  // any freestreamVelocity/Pressure (mixed) patch present
    bool   hasPiov_ = false;                                   // any pressureInletOutletVelocity (directionMixed) patch present
    bool   hasSym_ = false;                                    // any slip/symmetry patch present (general normal)
    bool   hasTotalP_ = false;                                 // any totalPressure p patch present (per-step refValue)
    bool   hasCyclic_ = false;                                 // any cyclic (periodic) interface -> Jacobi-PCG pressure (no AMG)
    DeviceCyclic cyc_;                                          // periodic interface coupling (OF updateInterfaceMatrix)
    // (fvPatch index, face count) of each side stacked into cyc_.phi / ami_.phi, in the order
    // buildDeviceCyclic / buildDeviceAMI concatenated them -- the map interfacePatchFlux() needs.
    // The MOMENTUM interface off-diagonal, kept aside because cyc_/ami_.ifCoeff is shared with the
    // pressure laplacian assembly and UEqn.H() needs the momentum one on every corrector, not just the
    // first. See solveMomentumPredictor for what reading the wrong one costs.
    // fvc::ddtCorr state. phi.oldTime() -- the flux at the previous TIME level, snapshotted in
    // advanceTime and held constant across the step's correctors, exactly as OF's oldTime() is. The
    // boundary mask is 0 wherever OF zeroes the coupling coefficient (U fixesValue, or a coupled patch)
    // and 1 elsewhere; it is built once, since neither condition changes during a run.
    DeviceBuffer<scalar> phiOldInt_, phiOldBnd_, ddtCorrBndMask_;
    // ...and the same two things ON the interface, where OF's ddtCorr is the ONLY place it survives.
    // ddtScheme::fvcDdtPhiCoeff zeroes the coefficient on `isA<cyclicAMIFvPatch>`, and cyclicACMIFvPatch
    // derives from coupledFvPatch, NOT from cyclicAMIFvPatch -- so a plain cyclicAMI gets no correction
    // and a cyclicACMI does. Everywhere else it vanishes on its own: a boundary face's phi IS U_b & Sf,
    // so phiCorr is identically zero there. Measured on oscillatingInletACMI2D at t=0.01, OF's term is
    // 0 on inlet/outlet/walls/blockage and max 1.07e-04 on the two coupled patches.
    DeviceBuffer<scalar> amiPhiOld_, cycPhiOld_, amiFluxUold_, cycFluxUold_, ddtCorrIfOnes_;
    bool   ddtCorr_ = true;         // fvSolution PIMPLE/ddtCorr (OF pimpleControl.C:55, default true)
    bool   ddtCorrNoticed_ = false; // say once if the scheme is one brae has no ddtCorr form for

    // mesh.phi() from the last move (per mesh face, internal then boundary). Held rather than applied at
    // the move: fvc::makeRelative belongs at the end of the pressure corrector. See makeFluxRelative.
    std::vector<scalar> meshPhi_;
    // Uf -- the FACE VELOCITY a moving-mesh pimpleFoam carries beside phi (createUfIfPresent.H, made
    // whenever mesh.dynamic()). Held in brae's four flux compartments: internal faces, non-coupled
    // boundary faces, and the two interfaces. Updated by fvc::correctUf at the end of every pressure
    // corrector, aged into *Old_ at the top of each step, and read by fvc::ddtCorr as
    // phiUf0 = Sf_current & Uf.oldTime().
    //
    // WHY IT EXISTS RATHER THAN A SHORTCUT. phi + mesh.phi() reproduces phiUf0 exactly while Sf holds
    // still, and that covered a translating mesh -- but a cyclicACMI rescales its coupled face areas
    // every step from the overlap mask, and a rotating mesh turns Sf under the stored Uf. On the moving
    // oscillatingInletACMI2D the shortcut was exact at step 1 and 5.1e-03 off by step 10, all of it on
    // the interface columns. Carrying the real field removes the approximation instead of announcing it.
    DeviceBuffer<scalar> UfInt_[3], UfBnd_[3], UfAmi_[3], UfCyc_[3];
    DeviceBuffer<scalar> UfOldInt_[3], UfOldBnd_[3], UfOldAmi_[3], UfOldCyc_[3];
    // Second old level, for backward's fvcDdtUfCorr. OF forms it as (Sf & Uf.oldTime().oldTime()) --
    // CURRENT areas against the two-steps-old face velocity -- so on a moving mesh it cannot be produced
    // by ageing the phi VALUE, which would carry the older Sf along with it.
    DeviceBuffer<scalar> UfOld2Int_[3], UfOld2Bnd_[3];
    DeviceBuffer<scalar> phiOld2Int_, phiOld2Bnd_;
    bool   ufActive_ = false;            // set once the mesh-motion path has run: OF's mesh.dynamic()
    bool   meshPhiValid_ = false;
    DeviceBuffer<scalar> cycIfCoeffMom_, amiIfCoeffMom_;
    // Coupled-interface edges added to the AMG agglomeration, in the order they were appended to the
    // fine edge list (internal faces first). The level-0 Galerkin needs a fine coefficient array of the
    // same extended length, so the interface coefficients are gathered into these positions each step.
    label                nAmgIfEdges_ = 0, nAmgCycEdges_ = 0, nAmgAmiEdges_ = 0;
    DeviceBuffer<label>  amgIfAmiSrc_;   // per appended AMI edge: its source face (index into ami_.ifCoeff)
    DeviceBuffer<scalar> amgIfAmiW_;     // per appended AMI edge: its stencil weight
    std::vector<label>   amgIfOwn_, amgIfNbr_;
    DeviceBuffer<scalar> amgFineUpper_, amgFineLower_;   // [internal faces | interface entries]
    std::vector<std::pair<label, label>> cycRuns_, amiRuns_;
    // DILU for the momentum solves (OF's `preconditioner DILU`). The level schedule depends only on the
    // mesh addressing, so it is built once; rD is refactorised inside every solve.
    DeviceDilu dilu_;   // shared by the momentum and turbulence BiCGStab (schedule depends only on the mesh)
    bool   hasAMI_ = false;                                     // any cyclicAMI interface -> Jacobi-PCG pressure (no AMG)
    bool   amiNonConforming_ = false;                           // ...and a face with >1 partner -> BiCGStab (see below)
    DeviceAMI    ami_;                                          // cyclicAMI weighted-stencil coupling (translational path)
};

} // namespace brae
