// _cpp REFERENCE implementation -- see rhoSimpleFoam_cpp.cuh for the OpenFOAM provenance and the order.
#include "rhoSimpleFoam_cpp.cuh"
#include "kOmegaSST_cpp.cuh"
#include "cell_wall_dist.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "pbicgstab.cuh"
#include "thermo_model.cuh"
#include "transport_model.cuh"
#include "equation_of_state.cuh"
#include "linearViscousStress_cpp.cuh"   // effectiveFaceViscosity: interpolate with the field own boundary
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace rhoSimple {

void thermoCorrect(
    RhoSimpleFields&            f,
    const std::vector<FvPatch>& patches)
{
    // T from he, the exact inverse of the hConst relation createFields used in the other direction, and
    // then psi from THAT T. Doing only the first half is the defect the rhotiming gate exists for: the
    // pressure equation would carry a psi that belongs to the previous iteration's temperature.
    const std::size_t nC = f.he.internal.size();
    for (std::size_t c = 0; c < nC; ++c)
    {
        f.T.internal[c]   = hConstHeToT(f.he.internal[c], f.thermo);
        f.psi[c]          = perfectGasPsi(f.T.internal[c], f.thermo);
        // heRhoThermo::calculate() fills rho_ HERE, from the p and T it sees at correct() time
        // (heRhoThermo.C:88). It is not the same number as p*psi later in the iteration, because the
        // pressure equation has not run yet.
        f.rhoThermo[c]    = perfectGasRho(f.p.internal[c], f.T.internal[c], f.thermo);
    }
    // The boundary half is heRhoThermo::calculate()'s patch loop (heRhoThermo.C:102-142; hePsiThermo.C
    // :106-132 has the same shape), NOT an evaluate of T's own boundary conditions. A patch whose T
    // fixesValue() -- fixedValue, and every mixed one (mixedFvPatchField.H:197; inletOutlet inherits it)
    // -- KEEPS the T it has and gets he_b = HE(p_b, T_b) written into he; every other patch inverts,
    // T_b = THE(he_b, p_b). psi_b and rho_b follow that T_b on both branches. The only place OpenFOAM
    // evaluates a mixed T patch from the cells is Tw.evaluate() inside the energy conditions' own
    // updateCoeffs (mixedEnergyFvPatchScalarField.C:97, gradientEnergy... .C:109, fixedEnergy... .C:108)
    // -- at the energy ASSEMBLY, from the cells as they stood before this correct() -- which the step
    // mirrors just before assembleEEqn. The evaluate that used to sit here put the NEW cell temperature
    // on sbMatched's inletOutlet outlet (1000.82 K against OpenFOAM's 1000.00 at iteration 2) and, through
    // rho_b, on the pressure flux that patch carries (0.38203 vs 0.38235): phiHbyA on the outlet 8e-4
    // off while every internal face was at 2e-11, k 1.2e-2 in the outlet cells at t=2 (item 26).
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& pb = f.p.boundary[pi]->value();
        std::vector<scalar> tb = f.T.boundary[pi]->value();
        if (f.T.boundary[pi]->fixesValue())
        {
            std::vector<scalar> hb(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i) hb[i] = hConstTToHe(tb[i], f.thermo);
            f.he.boundary[pi]->assignValue(std::move(hb));
        }
        else
        {
            const std::vector<scalar>& hb = f.he.boundary[pi]->value();
            for (label i = 0; i < patches[pi].size; ++i) tb[i] = hConstHeToT(hb[i], f.thermo);
            f.T.boundary[pi]->assignValue(tb);
        }
        for (label i = 0; i < patches[pi].size; ++i)
        {
            f.psiBnd[pi][i]        = perfectGasPsi(tb[i], f.thermo);
            f.rhoThermoBnd[pi][i]  = perfectGasRho(pb[i], tb[i], f.thermo);
        }
    }
}


void effectiveTransport(
    const RhoSimpleFields&            f,
    const std::vector<FvPatch>&       patches,
    std::vector<scalar>&              muEff,
    std::vector<std::vector<scalar>>& muEffBnd,
    std::vector<scalar>&              alphaEff,
    std::vector<std::vector<scalar>>& alphaEffBnd)
{
    // laminarModel::divDevRhoReff uses muEff() == mu(), and its alphaEff() is thermo.alphahe(), which
    // heThermo defines as CpByCpv*alpha -- gamma for sensibleInternalEnergy, 1 for sensibleEnthalpy.
    // Dropping the CpByCpv factor is a 40% error in the energy diffusivity for air, and it is the kind
    // that converges quietly to the wrong temperature.
    const scalar cpByCpv = thermoCpByCpv(f.thermo);
    const std::size_t nC = f.T.internal.size();
    const bool turb = f.turbulent && !f.nut.internal.empty();
    muEff.resize(nC);
    alphaEff.resize(nC);
    for (std::size_t c = 0; c < nC; ++c)
    {
        const scalar T = f.T.internal[c];
        // mut = rho*nut, and alphat is the EddyDiffusivity's rho*nut/Prt -- READ from the field the
        // turbulence model maintains rather than recomputed here, so the two cannot drift apart.
        const scalar mut    = turb ? f.rho.internal[c] * f.nut.internal[c] : 0.0;
        const scalar alphat = (turb && !f.alphat.internal.empty()) ? f.alphat.internal[c] : 0.0;
        // transportAlpha takes the VISCOSITY, not the temperature (transport_model.cuh): constTransport
        // is mu/Pr and sutherland is the Eucken kappa/Cp built from mu. Passing T read as a viscosity of
        // ~1000 Pa.s and produced an alphaEff six orders too large -- and it did not crash or diverge,
        // because an over-diffusive energy equation still returns a nearly uniform T, which on a
        // low-speed fixture is the right answer. Both gates INJECT OpenFOAM's alphaEff, so neither could
        // see it; the comparison of brae's own against OpenFOAM's is what found it.
        const scalar muLam = transportMu(T, f.thermo);
        muEff[c]    = muLam + mut;
        alphaEff[c] = cpByCpv * (transportAlpha(muLam, f.thermo) + alphat);
    }
    muEffBnd.assign(patches.size(), {});
    alphaEffBnd.assign(patches.size(), {});
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& tb = f.T.boundary[pi]->value();
        muEffBnd[pi].resize(patches[pi].size);
        alphaEffBnd[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            // The BOUNDARY nut is the wall function's, never the adjacent cell's -- the distinction the
            // momentum path has always made through nuEffBnd, and the one a wall function exists to make.
            const scalar mutB = turb ? f.rho.boundary[pi]->value()[i] * f.nut.boundary[pi]->value()[i] : 0.0;
            const scalar alphatB = (turb && !f.alphat.internal.empty())
                                 ? f.alphat.boundary[pi]->value()[i] : 0.0;
            muEffBnd[pi][i]    = transportMu(tb[i], f.thermo) + mutB;
            alphaEffBnd[pi][i] =
                cpByCpv * (transportAlpha(transportMu(tb[i], f.thermo), f.thermo) + alphatB);
        }
    }
}


// rho = thermo.rho(), internal and boundary, from the CURRENT p and T.
//
// EXPORTED rather than file-local because the CUDA driver takes it as a hook -- `rho = thermo.rho()` is a
// thermo operation and the device driver must not learn to be a thermodynamic model -- and the driver's
// gate has to hand the SAME function to both sides, or it would be comparing two thermos as well as two
// drivers.
void updateRho(
    RhoSimpleFields&            f,
    const std::vector<FvPatch>& patches)
{
    const std::size_t nC = f.rho.internal.size();
    // `rho = thermo.rho()`, and WHICH rho that is depends on the thermo type. For hePsiThermo it is
    // p_*psi_ (psiThermo.C:150), recomputed with the pressure that was just solved. For heRhoThermo it
    // is the stored rho_ (rhoThermo.C:233), which heRhoThermo::calculate() last filled inside
    // thermo.correct() at the end of EEqn.H -- from the pressure BEFORE the pressure equation ran.
    //
    // Recomputing it live for heRhoThermo is wrong wherever p moves appreciably in one iteration.
    // angledDuct is such a case: p is clamped from 100000 to pMaxFactor's 150000 on the first
    // iteration, so live gives mixture.rho(150000, 293) = 1.779455 where OpenFOAM carries
    // mixture.rho(100000, 293) = 1.186304. Relaxed at the case's 0.01 that is 1.192236 against
    // OpenFOAM's 1.186303 -- the whole of the 3.93e-03 rho gap at iteration 1, in one cell, with p
    // and T identical in both codes. squareBend and sbMatched are heRhoThermo too and their gates
    // never caught it: their per-iteration pressure excursions are too small to separate the two.
    if (f.thermo.rhoThermoType)
    {
        f.rho.internal = f.rhoThermo;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            std::vector<scalar> rb = f.rhoThermoBnd[pi];
            f.rho.boundary[pi]->setStoredValues(std::move(rb));
        }
        return;
    }
    for (std::size_t c = 0; c < nC; ++c)
        f.rho.internal[c] = perfectGasRho(f.p.internal[c], f.T.internal[c], f.thermo);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& pb = f.p.boundary[pi]->value();
        const std::vector<scalar>& tb = f.T.boundary[pi]->value();
        std::vector<scalar> rb(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            rb[i] = perfectGasRho(pb[i], tb[i], f.thermo);
        f.rho.boundary[pi]->setStoredValues(std::move(rb));
    }
}

namespace {

// GeometricField::relax(alpha): x = xOld + alpha*(x - xOld).
void relaxField(
    std::vector<scalar>&       x,
    const std::vector<scalar>& xOld,
    scalar                     alpha)
{
    if (alpha <= 0.0 || alpha >= 1.0) return;
    for (std::size_t c = 0; c < x.size(); ++c) x[c] = xOld[c] + alpha * (x[c] - xOld[c]);
}


// The boundary half of GeometricField::relax. operator== assigns both halves (GeometricField.C:1094 and
// :1420, GeometricBoundaryField.C operator== -> fvPatchField::operator== -> Field::operator=), so the
// patch VALUE OpenFOAM carries out of p.relax() is prevIter_b + alpha*(p_b - prevIter_b) on every patch.
// Stored on EVERY class through assignValue (OpenFOAM's operator==), and read as it stands by the
// velocity corrector's grad(p) and, without a pressure limiter, by the next momentum assembly.
void relaxBoundary(
    GeometricField<scalar>&                 p,
    const std::vector<std::vector<scalar>>& prevIter,
    scalar                                  alpha)
{
    if (alpha <= 0.0 || alpha >= 1.0) return;
    for (std::size_t pi = 0; pi < p.boundary.size(); ++pi)
    {
        std::vector<scalar> v = p.boundary[pi]->value();
        const std::vector<scalar>& prev = prevIter[pi];
        for (std::size_t i = 0; i < v.size() && i < prev.size(); ++i)
        {
            v[i] = prev[i] + alpha * (v[i] - prev[i]);
        }
        // assignValue is fvPatchField::operator==: the VALUE, on every class, whatever setStoredValues
        // means for it (the reference on the mixed and extrapolated families). The mixed family used to be
        // re-evaluated here instead, which on a freestreamPressure patch is a different number from the
        // blend (queue item 23); on a zeroGradient face the two are the same arithmetic on the same operands.
        p.boundary[pi]->assignValue(std::move(v));
    }
}


// pressureInletOutletVelocityFvPatchVectorField::updateCoeffs (its .C:170-184): valueFraction from the
// flux the patch currently holds, then directionMixed::evaluate (directionMixedFvPatchField.C:157-175)
// -- value = transform(vf, refValue) + transform(I - vf, patchInternalField()), i.e. n(n & U_cell) on an
// inflow face and U_cell on an outflow one. OpenFOAM runs this wherever it reaches an updateCoeffs or
// an evaluate on U: the momentum fvMatrix constructor, the solve's correctBoundaryConditions and the
// velocity corrector's. Every other patch class ignores the call.
void updatePressureInletOutletVelocity(
    RhoSimpleFields&            f,
    const std::vector<FvPatch>& patches)
{
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        // patchInternalField: U at the cell each face belongs to.
        std::vector<vector> Ucell(static_cast<std::size_t>(patches[pi].size), vector{});
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            if (c >= 0 && c < static_cast<label>(f.U.internal.size())) Ucell[i] = f.U.internal[c];
        }
        f.U.boundary[pi]->updateFromPatchVelocity(
            f.U.boundary[pi]->value(),
            Ucell,
            f.rho.boundary[pi]->value());
    }
}


// totalPressureFvPatchScalarField::updateCoeffs (its .C:152-225, the psiName_ == "none" branch):
// p_b = p0 - 0.5*rho_b*neg(phi_b)*|U_b|^2 from the patch fields AS THEY STAND -- U's patch value, the
// registered phi and rho's patch value (lookupPatchField, :162-174). OpenFOAM reaches it inside the
// pressure fvMatrix constructor and again from the correctBoundaryConditions pEqn.H:100-103 runs after
// the limiter; nowhere else. Every other patch class ignores the call.
void updateTotalPressure(
    RhoSimpleFields&            f,
    const std::vector<FvPatch>& patches,
    bool                        evaluateAll)
{
    // freestreamPressure's updateCoeffs runs in the same two places (the pressure fvMatrix constructor and
    // the limiter's correctBoundaryConditions): valueFraction = 0.5 + 0.5*(Up & nf)/mag(Up) from U's
    // PATCH value as it stands (freestreamPressureFvPatchScalarField.C:109-121). It used to be rebuilt at
    // the TOP of the iteration, from the corrected U of the previous one, and p's boundary re-evaluated
    // there: on naca0012 the host read p 3.4e-05 / U 6.5e-05 against OpenFOAM over t=1..10 with the
    // limiter and 9.8e-05 without it (queue item 23).
    std::vector<std::vector<vector>> Ub(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) Ub[pi] = f.U.boundary[pi]->value();
    updateMixedFreestream(f.p.boundary, Ub, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        f.p.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
        f.p.boundary[pi]->updateFromPatchVelocity(
            f.U.boundary[pi]->value(),
            f.U.boundary[pi]->value(),
            f.rho.boundary[pi]->value());
    }
    // updateCoeffs alone changes no other patch's VALUE (totalPressure's operator== is inside its own
    // updateCoeffs, done above); the evaluate is OpenFOAM's correctBoundaryConditions, which follows only
    // under the limiter (pEqn.H:100-103) -- a mixed or zeroGradient face keeps the relaxed blend until then.
    if (evaluateAll) f.p.evaluateBoundary();
}

} // namespace


// Instrument: BRAE_STAGE_DUMP_DIR=<dir> (+ BRAE_STAGE_DUMP_ITER=n, default 1) writes this step's stages
// at ONE iteration as plain columns (%.17g per line; vectors as three columns; surface fields as the
// internal faces plus one file per patch), named after the stage_* fields tools/dumpPEqn writes. The
// file-restart harnesses (test_rho_ueqn_cpp, test_rho_peqn_cpp) rebuild an iteration from OpenFOAM's
// WRITTEN state and cannot see a defect that lives only in the mirror's own in-memory trajectory: on
// sbMatched the momentum matrix rebuilt from OpenFOAM's t=1 files is exact to 8e-15 while the continuous
// run's outlet cells are 1e-2 off on k at t=2 (item 26). This dumps the continuous run's stages instead.
namespace
{
struct StageDump
{
    std::string dir;
    bool        on = false;

    std::FILE* open(const char* name) const
    {
        return std::fopen((dir + "/" + name).c_str(), "w");
    }
    void scalars(const char* name, const std::vector<scalar>& v) const
    {
        if (!on) return;
        std::FILE* fp = open(name);
        if (!fp) return;
        for (scalar x : v) std::fprintf(fp, "%.17g\n", (double)x);
        std::fclose(fp);
    }
    void vectors(const char* name, const std::vector<vector>& v) const
    {
        if (!on) return;
        std::FILE* fp = open(name);
        if (!fp) return;
        for (const vector& x : v) std::fprintf(fp, "%.17g %.17g %.17g\n", (double)x.x, (double)x.y, (double)x.z);
        std::fclose(fp);
    }
    // A vector field's BOUNDARY values, one file per patch. The mixed family's patch VALUE is a stage in
    // its own right: OpenFOAM updates a mixed valueFraction inside the fvMatrix constructor and does not
    // evaluate until the solve's correctBoundaryConditions, so its value at an assembly is the blend of
    // the PREVIOUS valueFraction -- a lag a mirror that evaluates eagerly does not reproduce.
    void vectorsBnd(const char* name, const std::vector<std::vector<vector>>& vb) const
    {
        if (!on) return;
        for (std::size_t pi = 0; pi < vb.size(); ++pi)
        {
            std::FILE* fp = open((std::string(name) + "_b" + std::to_string(pi)).c_str());
            if (!fp) continue;
            for (const vector& x : vb[pi])
                std::fprintf(fp, "%.17g %.17g %.17g\n", (double)x.x, (double)x.y, (double)x.z);
            std::fclose(fp);
        }
    }
    void surface(const char* name, const SurfaceScalarField& sf) const
    {
        if (!on) return;
        scalars(name, sf.internal);
        for (std::size_t pi = 0; pi < sf.boundary.size(); ++pi)
        {
            scalars((std::string(name) + "_b" + std::to_string(pi)).c_str(), sf.boundary[pi]);
        }
    }
};

StageDump stageDump()
{
    StageDump d;
    const char* dd = std::getenv("BRAE_STAGE_DUMP_DIR");
    if (!dd) return d;
    static int calls = 0;
    const char* it = std::getenv("BRAE_STAGE_DUMP_ITER");
    d.dir = dd;
    d.on  = (++calls == (it ? std::atoi(it) : 1));
    return d;
}
} // namespace

Residuals rhoSimpleStep(
    RhoSimpleFields&            f,
    const StepInput&            in,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    Residuals res;
    const label nC = m.nCells();

    // rho.prevIter(), stored where OpenFOAM stores it. simpleControl::loop() calls storePrevIterFields()
    // at the START of the iteration (simpleControl.C:157) and rho.relax() at the tail is
    // prevIter + alpha*(rho - prevIter) (GeometricField.C:1089-1095); pcEqn.H:1's `rho = thermo.rho()`
    // does not touch prevIter. This capture used to sit at the tail, one line before the tail's own
    // updateRho -- exact on the pEqn branch, where rho does not move in between, and wrong on the
    // SIMPLEC branch, whose pcEqn.H opens with rho = thermo.rho(): the relaxation then blended towards
    // that mid-iteration density instead of the one the iteration started with. No fixture could see it
    // (every consistent+subsonic one relaxes rho at 1.0); the gate is rhoBox with `consistent yes` and
    // `rho 0.5`, both arms against OpenFOAM at a matched iteration count.
    const std::vector<scalar> rhoPrevIter = f.rho.internal;
    std::vector<std::vector<scalar>> rhoBndPrevIter(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) rhoBndPrevIter[pi] = f.rho.boundary[pi]->value();
    const StageDump sd = stageDump();
    // p.prevIter(), banked at the same place for the same reason (simpleControl.C:157), BOTH halves:
    // p.relax() at the tail is the same operator== as rho's and assigns the boundary too
    // (GeometricField.C:1094, :1420). The totalPressure inlet is a patch whose value MOVES between here
    // and the tail -- it is recomputed before the pressure assembly -- so the blend the tail leaves has
    // to be against the value this iteration started with, not against the recomputed one.
    const std::vector<scalar> pPrevIter = f.p.internal;
    std::vector<std::vector<scalar>> pBndPrevIter(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) pBndPrevIter[pi] = f.p.boundary[pi]->value();

    // updateCoeffs() for the flux-conditional boundary conditions. OpenFOAM's inletOutlet reads phi from
    // the object registry inside updateCoeffs, which the matrix assembly calls before asking for any
    // coefficient; brae has no registry, so the flux is pushed in here instead -- once per iteration,
    // before anything is assembled, from the phi this iteration starts with. U and he are the fields that
    // carry one on a compressible case; a patch of any other type ignores it.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        f.U.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
        f.he.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
        f.T.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
    }

    // updateCoeffs() for the FREESTREAM family, which is a different rule from the flux switch above.
    // freestreamVelocity is a mixed patch whose valueFraction OpenFOAM recomputes from the current flow
    // ANGLE -- valueFraction() = 0.5 - 0.5*(Up & nf)/mag(Up) (freestreamVelocityFvPatchVectorField.C) --
    // and freestreamPressure follows it. The incompressible driver has always done this; this one never
    // did, so on a far-field case every such patch kept the 0.5 it was seeded with for the whole run.
    // aerofoilNACA0012 is exactly that case: its inlet and outlet are both in the `freestream` group.
    {
        std::vector<std::vector<vector>> Ub(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) Ub[pi] = f.U.boundary[pi]->value();
        updateMixedFreestream(f.U.boundary, Ub, patches);
        // NOT p. freestreamPressure's updateCoeffs runs inside the pressure fvMatrix constructor
        // (updateTotalPressure below), and -fvc::grad(p) in the momentum equation reads p's boundary AS IT
        // STANDS: the blend p.relax() left, or the limiter's re-evaluation. Rebuilding p's valueFraction
        // here from the corrected U and re-evaluating was what kept naca0012 at 3.4e-05 (queue item 23).
    }
    // KEPT, and it should not be: OF's fvPatchField::updateCoeffs sets a flag and the momentum fvMatrix
    // constructor calls nothing else (fvMatrix.C:396), so a mixed patch assembles holding the NEW
    // valueFraction beside the value its last evaluate left. Removing this line and its twin below takes
    // naca0012 t=2 from U 5.9e-10 to 1.2e-12 -- and it was measured and then reverted, because it
    // separates this arm from the DEVICE by 1.32e-06 on rhoBoxP, which carries the same lag. Queue item
    // 30: both halves land together or neither does.
    // ...AND NOT U. updateCoeffs is not an evaluate: OF fvPatchField::updateCoeffs sets a flag
    // (fvPatchField.C) and the momentum fvMatrix constructor calls updateCoeffs and NOTHING else
    // (fvMatrix.C:396), so a mixed patch enters the assembly holding the NEW valueFraction beside the
    // value its last evaluate left -- the blend of the PREVIOUS one -- and only the solve's
    // correctBoundaryConditions reunites them. Evaluating here collapsed that lag (queue items 25, 30).
    // NOT he and NOT T either. Nothing in OpenFOAM's iteration evaluates them before the energy equation:
    // he's patch values stand as the previous solve's correctBoundaryConditions and thermo.correct()
    // left them (fixesValue patches carry HE(p_b, T_b), heRhoThermo.C:119), and T's are evaluated only
    // inside the energy conditions' updateCoeffs, at the assembly below. Evaluating both here re-derived
    // the mixed outlet's he_b and T_b from the CURRENT cells one stage early -- the transport this
    // momentum assembly reads on that patch (mu_b, alphaEff_b) is thermo.correct()'s, from the T_b it
    // computed with, not from a fresher one (item 26).

    std::vector<scalar>              muEff, alphaEff;
    std::vector<std::vector<scalar>> muEffBnd, alphaEffBnd;
    effectiveTransport(f, patches, muEff, muEffBnd, alphaEff, alphaEffBnd);

    std::vector<std::vector<scalar>> rhoBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) rhoBnd[pi] = f.rho.boundary[pi]->value();

    // updateCoeffs() for flowRateInletVelocity, HERE and not at the top of the iteration, because this is
    // where OpenFOAM does it: the value is replaced when the momentum fvMatrix is constructed, which is
    // after createFields.H has already built phi from the seeded field. It reads rho's PATCH values -- the
    // same object OF's patch().lookupPatchField(rhoName_) returns -- so the prescribed mass flow is held
    // against the density the flux is actually carrying. Feeding it a different rho is what made
    // angledDuct lose its inlet mass flow.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        f.U.boundary[pi]->updateFromDensity(rhoBnd[pi]);
    }

    // updateCoeffs() for pressureInletOutletVelocity, which OpenFOAM reaches inside the momentum
    // fvMatrix constructor (fvMatrix.C:396). Its updateCoeffs is not a coefficient update alone: it sets
    // valueFraction = neg(phi)*(I - nn) from the flux THIS iteration starts with and then calls
    // directionMixed::evaluate itself (pressureInletOutletVelocityFvPatchVectorField.C:180-183), so the
    // patch VALUE at the assembly is n(n & U_cell) on inflow faces and U_cell on outflow ones, from the
    // cell velocity as it stands here. That is the number the previous iteration's post-correction
    // evaluate left (same flux, same cells) -- and at iteration 1, where no evaluate has run, it is what
    // OpenFOAM computes too, so the 0/U seed never reaches a momentum assembly on any iteration.
    //
    // totalPressure is NOT updated here any more. OpenFOAM reaches p's updateCoeffs only inside the
    // PRESSURE equation's constructor (below, once the momentum solve has moved U's patch value), so
    // what -fvc::grad(p) reads at the momentum assembly is the value the previous tail left: the relaxed
    // blend from p.relax(), or the recompute pEqn.H:100-103 runs after the limiter. Updating it here from
    // the pre-solve velocity and the tail's rho put a one-solve-old dynamic head into the pressure
    // equation and a fresh value where OpenFOAM carries the blend: rhoTP at t=1 read U 2.15e-02 relL2
    // against OpenFOAM with the written inlet still at the (5 0 0) seed, OpenFOAM's at (13.44 0 0).
    updatePressureInletOutletVelocity(f, patches);
    // ...and its twin. Only pressureInletOutletVelocity evaluates inside its own updateCoeffs
    // (its .C:180-183), and updateFromPatchVelocity above already sets its value;
    // flowRateInletVelocity assigns its value outright, as OpenFOAM's does. Nothing else is
    // evaluated at an assembly.
    // ---- UEqn.H ----
    RhoMomentumInput uin;
    uin.phi = &f.phi.internal;   uin.phiBnd = &f.phi.boundary;
    // WHICH OpenFOAM STAGE THIS CORRESPONDS TO. dumpPEqn writes U TWICE -- stage_Uass before the matrix
    // is built, stage_Upost after fvMatrix's constructor has run updateCoeffs on the boundaries -- and
    // its header warns that a brae-side boundary which RECOMPUTES itself belongs against the second.
    // This write sits after brae's own updateCoeffs equivalents (the flux switch, the freestream blend,
    // flowRateInletVelocity, piov), so stage_Upost is the one to hold a BOUNDARY against.
    //
    // MEASURED 2026-09-03, and the distinction turned out not to bite on either fixture that has one of
    // those patches: at iteration 2 the two stages are bit-identical on naca0012's freestreamVelocity
    // inlet and on sbMatched's flowRateInletVelocity inlet (max|stage_Uass - stage_Upost| = 0.0). Both
    // patches recompute from state that has not moved since the previous iteration's own updateCoeffs, so
    // the recompute reproduces the stored value. The internal field is identical in both stages by
    // construction. Recorded because the reasoning says they CAN differ and the measurement says they do
    // not here -- so use stage_Upost for a boundary, but do not read a difference into a comparison that
    // was made against stage_Uass on these two.
    sd.vectors("Uass", f.U.internal);
    {
        std::vector<std::vector<vector>> ub(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) ub[pi] = f.U.boundary[pi]->value();
        sd.vectorsBnd("Uass", ub);
    }
    sd.scalars("muEffAss", muEff);
    sd.scalars("rhoU", f.rho.internal);
    sd.surface("phiU", f.phi);
    sd.scalars("nutU", f.nut.internal);
    uin.rho = &f.rho.internal;   uin.rhoBnd = &rhoBnd;
    uin.muEff = &muEff;          uin.muEffBnd = &muEffBnd;
    uin.relaxU             = in.relaxU;
    uin.relaxEquationU     = in.relaxEquationU;
    uin.bounded            = in.boundedU;
    uin.scheme             = in.schemeU;
    uin.schemeCoeff        = in.schemeCoeffU;
    uin.gradULimitK        = in.gradULimitK;
    uin.correctedLaplacian = in.correctedLaplacian;
    uin.snGradLimitCoeff   = in.snGradLimitCoeff;
    uin.hasMRF             = in.hasMRF;
    uin.hasFvOptions       = in.hasFvOptions;
    uin.fvOpts             = in.fvOpts;
    uin.fvOptionUnsupported = in.fvOptionUnsupported;
    // The LAMINAR mu(T), per cell, for DarcyForchheimer's Darcy term (OF resolves "thermo:mu",
    // DarcyForchheimer.C:64) -- NOT muEff: the porosity resistance is a property of the medium and the
    // molecular fluid, and feeding it the turbulent viscosity would grow the Darcy term with nut.
    std::vector<scalar> muLam;
    if (in.fvOpts && !in.fvOpts->empty())
    {
        muLam.resize(f.T.internal.size());
        for (std::size_t c = 0; c < muLam.size(); ++c)
            muLam[c] = transportMu(f.T.internal[c], f.thermo);
        uin.muLaminar = &muLam;
    }
    const FvVectorMatrix UEqn = assembleUEqn(f.U, uin, m, g, patches);
    {
        std::vector<scalar> sx(UEqn.source.size()), sy(sx.size()), sz(sx.size());
        for (std::size_t c = 0; c < sx.size(); ++c)
        {
            sx[c] = UEqn.source[c].x; sy[c] = UEqn.source[c].y; sz[c] = UEqn.source[c].z;
        }
        sd.scalars("UDiag", UEqn.diag);
        sd.scalars("UUpper", UEqn.upper);
        sd.scalars("USrcX", sx);
        sd.scalars("USrcY", sy);
        sd.scalars("USrcZ", sz);
    }
    {
        // solve(UEqn == -fvc::grad(p)) on a COPY: the pressure equation needs the ORIGINAL UEqn for
        // A(), H() and H1(), and addPressureGradient would otherwise leave the source carrying -grad(p).
        FvVectorMatrix Mp = UEqn;
        addPressureGradient(Mp, f.p, m, g, patches);
        // fvMatrix<vector>::solveSegregated solves only the components polyMesh::solutionD() leaves
        // valid (fvMatrixSolve.C:157-164) -- on a 2D case the empty direction is never solved and its
        // SolverPerformance stays at Zero -- and what residualControl compares is cmptMax over the
        // per-component vector it stores (solutionControl.C:232, simpleControl.C:67-71).
        //
        // This step reported component 0. Wrong whenever Uy's initial residual exceeds Ux's, which on
        // rhoBox is iterations 2, 3 and 8 (OpenFOAM at iteration 2: Ux 3.224e-01, Uy 6.042e-01; brae
        // printed 3.224e-01), so a residualControl on U fired at a different iteration from OpenFOAM's.
        // A max over three components SOLVED unconditionally would be the opposite error: the empty
        // direction's system has a ~0 right-hand side and a zero field, and its normFactor-scaled
        // residual reads 1 on every iteration (measured on rhoBox: Uz 1.000e+00 at iterations 1..3 on
        // both arms), which would block convergence on every 2D case. Hence the mask, derived from the
        // empty patches the way calcDirections does, and the skip.
        const SolutionDirections solutionD = solutionDirections(patches);
        SolverPerformance upCmpt[3];
        solveVector(Mp, f.U, m, patches, in.tolU, in.relTolU, in.maxIterU, in.minIterU, &solutionD, upCmpt);
        scalar uInitialResidual = 0.0;
        for (int cmpt = 0; cmpt < 3; ++cmpt)
        {
            uInitialResidual = std::max(uInitialResidual, upCmpt[cmpt].initialResidual);
        }
        res["U"] = uInitialResidual;
    }
    // fvMatrixSolve.C:242 -- every solve ends with psi.correctBoundaryConditions(). On the piov patch
    // that is updateCoeffs -> evaluate again (fvPatchField.C:343 cleared the updated flag after the
    // assembly's evaluate): the NEW cell velocity projected with the flux mask the iteration started
    // with. The energy equation's boundary Ekp and the totalPressure update below both read this value,
    // and a patch class whose evaluate() keeps its stored value needs the projection pushed in.
    updatePressureInletOutletVelocity(f, patches);
    f.U.evaluateBoundary();

    sd.vectors("Upred", f.U.internal);

    // ---- EEqn.H ----
    {

        EnergyInput ein;
        ein.phi = &f.phi.internal;  ein.phiBnd = &f.phi.boundary;
        ein.alphaEff = &alphaEff;   ein.alphaEffBnd = &alphaEffBnd;
        ein.heName            = f.heName;
        ein.relaxHe           = in.relaxHe;
        ein.relaxEquationHe   = in.relaxEquationHe;
        ein.boundedHe         = in.boundedHe;
        ein.boundedKE         = in.boundedKE;
        ein.schemeHe          = in.schemeHe;
        ein.schemeKE          = in.schemeKE;
        ein.gradHeLimitK      = in.gradHeLimitK;
        ein.gradKELimitK      = in.gradKELimitK;
        ein.correctedLaplacian = in.correctedLaplacian;
        ein.snGradLimitCoeff  = in.snGradLimitCoeff;
        ein.hasMRF            = in.hasMRF;
        ein.hasFvOptions      = in.hasFvOptions;
        // Tw.evaluate() -- every energy condition's updateCoeffs evaluates T's patch from the cells as
        // they stand at the energy assembly (fixedEnergy .C:108, gradientEnergy .C:109, mixedEnergy .C:97)
        // and builds its refValue/refGrad/value from that T. This is the ONLY evaluate T's boundary
        // gets in an iteration; thermo.correct() below then keeps it on fixesValue patches.
        f.T.evaluateBoundary();
        FvScalarMatrix E = assembleEEqn(f.he, f.U, f.p, f.rho, ein, m, g, patches);
        // fvOptions.constrain(EEqn), EEqn.H:24. A fixedTemperatureConstraint sets he(p, Tuniform) on its
        // cells -- an ENERGY, not a temperature. The thermo conversion is supplied here because the
        // fvOptions reference carries no thermo.
        if (in.fvOpts && !in.fvOpts->empty())
        {
            static ThermoCoeffs tc;   // the lambda below cannot capture, so the thermo goes through here
            tc = f.thermo;
            static bool isE;
            isE = (f.heName == "e");
            cpu::fvOptions::constrain(
                *in.fvOpts, E, f.he.internal, f.heName, m, patches,
                [](scalar T)
                {
                    const scalar hs = tc.Cp * (T - tc.Tref) + tc.Href;
                    return isE ? hs - tc.R * T : hs;
                });
        }
        const SolverPerformance ep =
            pbicgstab(E, f.he.internal, m, patches, in.tolHe, in.relTolHe, in.maxIterHe, in.minIterHe);
        res[f.heName] = ep.initialResidual;
        f.he.evaluateBoundary();
    }
    // fvOptions.correct(he), EEqn.H:27 -- after the energy solve and BEFORE thermo.correct(), which is
    // what makes it show up in T. limitTemperature clamps he between he(p,Tmin) and he(p,Tmax); on the
    // boundary it does the same for any patch that does not fix a value, then corrects the boundary.
    if (in.limitT)
    {
        const scalar heMin = hConstTToHe(in.limitTmin, f.thermo);
        const scalar heMax = hConstTToHe(in.limitTmax, f.thermo);
        for (label c = 0; c < nC; ++c)
        {
            if      (f.he.internal[c] < heMin) f.he.internal[c] = heMin;
            else if (f.he.internal[c] > heMax) f.he.internal[c] = heMax;
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (f.he.boundary[pi]->fixesValue()) continue;
            std::vector<scalar> hb = f.he.boundary[pi]->value();
            for (auto& v : hb)
            {
                if      (v < heMin) v = heMin;
                else if (v > heMax) v = heMax;
            }
            f.he.boundary[pi]->setStoredValues(std::move(hb));
        }
        f.he.evaluateBoundary();
    }

    // EEqn.H ends with thermo.correct(): T, and therefore psi, move here and everything below sees them.
    thermoCorrect(f, patches);
    sd.scalars("he", f.he.internal);
    sd.scalars("T", f.T.internal);
    sd.scalars("psi", f.psi);

    // ---- pEqn.H or pcEqn.H ----
    PressureInput pin;
    pin.rho = &f.rho.internal;  pin.rhoBnd = &rhoBnd;
    pin.psi = &f.psi;           pin.psiBnd = &f.psiBnd;
    pin.transonic            = in.transonic;
    pin.relaxP               = in.relaxPEqn;
    pin.relaxPSpecified      = in.relaxPEqnSpecified;
    pin.pRefCell             = f.pressureControl.refCell;
    pin.pRefValue            = f.pressureControl.refValue;
    pin.correctedLaplacian   = in.correctedLaplacian;
    pin.snGradLimitCoeff     = in.snGradLimitCoeff;
    pin.hasMRF               = in.hasMRF;
    pin.hasFvOptions         = in.hasFvOptions;

    std::vector<scalar>       rAUorAtU;      // whichever the velocity corrector uses
    std::vector<vector>       HbyA;
    SurfaceScalarField        phiHbyA;
    bool                      closedVolume = false;
    FvScalarMatrix            P;

    if (in.consistent)
    {
        // pcEqn.H opens with `rho = thermo.rho()`. pEqn.H does NOT -- the SIMPLEC pressure equation is
        // built from a density that already reflects the just-solved T.
        updateRho(f, patches);
        for (std::size_t pi = 0; pi < patches.size(); ++pi) rhoBnd[pi] = f.rho.boundary[pi]->value();

        const ConsistentPressureStages st =
            consistentPressurePredictor(UEqn, f.U, f.p, pin, m, g, patches);
        sd.scalars("rAU", st.rAU);
        sd.scalars("rAtU", st.rAtU);
        sd.scalars("rhorAtU", st.rhorAtU);
        sd.vectors("HbyA", st.HbyA0);
        sd.vectors("HbyAc", st.HbyA);
        sd.surface("phiHbyA0", st.phiHbyA0);
        sd.surface("phiHbyAc", st.phiHbyA);
        sd.surface("phid", st.phid);
        sd.scalars("rhoP", f.rho.internal);
        // totalPressure's updateCoeffs, where OpenFOAM runs it: inside the pressure fvMatrix's
        // constructor (fvMatrix.C:396), from U's PATCH value as the momentum solve left it, the flux the
        // iteration started with, and rho's patch value as pcEqn.H:1 just set it.
        updateTotalPressure(f, patches, /*evaluateAll=*/false);
        // `while (simple.correctNonOrthogonal())` -- pcEqn.H:67. Each pass re-assembles from the SAME
        // stages (phiHbyA, rhorAtU are outside the loop in OF too) and re-solves; what changes between
        // passes is the deferred non-orthogonal correction, recomputed from the just-solved p. The
        // boundary refresh before re-assembly is fvMatrixSolve.C:242 -- every OF solve ends with
        // psi.correctBoundaryConditions() -- placed so the nNonOrth=0 path stays bit-identical to the
        // single-solve arithmetic every existing gate was measured on. res["p"] keeps the FIRST pass's
        // initial residual, which is the one residualControl reads.
        for (label corr = 0; corr <= in.nNonOrthogonalCorrectors; ++corr)
        {
            if (corr > 0) f.p.evaluateBoundary();
            P = assemblePcEqn(st, f.p, pin, m, g, patches);
            if (corr == 0) { sd.scalars("pD", P.diag); sd.scalars("pSrc", P.source); }
            const SolverPerformance pp =
                pbicgstab(P, f.p.internal, m, patches, in.tolP, in.relTolP, in.maxIterP, in.minIterP);
            if (corr == 0) res["p"] = pp.initialResidual;
        }
        sd.scalars("pSolved", f.p.internal);
        rAUorAtU     = st.rAtU;
        HbyA         = st.HbyA;
        phiHbyA      = st.phiHbyA;
        closedVolume = st.closedVolume;
    }
    else
    {
        const PressureStages st = pressurePredictor(UEqn, f.U, f.p, pin, m, g, patches);
        // The same updateCoeffs on this branch, with rho's patch value as the previous tail left it.
        updateTotalPressure(f, patches, /*evaluateAll=*/false);
        // The same loop for pEqn.H:56 -- see the pcEqn branch above for why it is shaped this way.
        for (label corr = 0; corr <= in.nNonOrthogonalCorrectors; ++corr)
        {
            if (corr > 0) f.p.evaluateBoundary();
            P = assemblePEqn(st, f.p, pin, m, g, patches);
            const SolverPerformance pp =
                pbicgstab(P, f.p.internal, m, patches, in.tolP, in.relTolP, in.maxIterP, in.minIterP);
            if (corr == 0) res["p"] = pp.initialResidual;
        }
        rAUorAtU     = st.rAU;
        HbyA         = st.HbyA;
        phiHbyA      = st.phiHbyA;
        closedVolume = st.closedVolume;
    }

    // phi = phiHbyA + pEqn.flux(). PLUS, not minus: rhoSimpleFoam writes the pressure equation as
    // `fvc::div(phiHbyA) - fvm::laplacian(...) == 0`, where the incompressible solver writes
    // `fvm::laplacian(...) == fvc::div(phiHbyA)` and correspondingly subtracts. Same physics, opposite
    // sign, and the flux is what makes phi discretely conservative.
    {
        const SurfaceScalarField fl = matrixFlux(P, f.p.internal, m, patches);
        f.phi = phiHbyA;
        for (std::size_t i = 0; i < f.phi.internal.size(); ++i) f.phi.internal[i] += fl.internal[i];
        for (std::size_t pi = 0; pi < f.phi.boundary.size(); ++pi)
            for (std::size_t i = 0; i < f.phi.boundary[pi].size(); ++i)
                f.phi.boundary[pi][i] += fl.boundary[pi][i];
    }

    // #include "incompressible/continuityErrs.H" -- pEqn.H:81, HERE, between the flux correction and
    // p.relax(). NOT compressibleContinuityErrs.H: rhoSimpleFoam includes the INCOMPRESSIBLE one, and
    // the compressible file (rho vs thermo.rho) is a different number entirely -- reading the solver
    // caught what a from-memory transcription would have substituted. contErr = fvc::div(phi); the
    // V-weighted average cancels the /V in the divergence, so sumLocal = dt*sum|netFlux_c|/sumV.
    //
    // WHAT A GATE MAY AND MAY NOT COMPARE: sumLocal/global sit at each solver's LINEAR-TOLERANCE
    // floor once converged (measured rhoBoxF: brae 2.1e-09 vs OF 8.4e-09 -- different Krylov stacks,
    // different floors), and `cumulative` INTEGRATES that floor over the whole trajectory (OF
    // -9.6e-05 vs brae 1.7e-11 on the same case). Comparing cumulatives compares stopping criteria,
    // not physics; the tf gate therefore bounds brae's own sumLocal absolutely and checks OF's is at
    // a like floor, nothing more.
    {
        std::vector<scalar> net(nC, scalar(0));
        const std::vector<label>& own = m.owner();
        const std::vector<label>& nei = m.neighbour();
        for (std::size_t fi = 0; fi < nei.size(); ++fi)
        {
            net[own[fi]] += f.phi.internal[fi];
            net[nei[fi]] -= f.phi.internal[fi];
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
                net[patches[pi].faceCells[i]] += f.phi.boundary[pi][i];
        scalar sumV = 0, sumLoc = 0, sumGlob = 0;
        for (label c = 0; c < nC; ++c)
        {
            sumV    += g.V()[c];
            sumLoc  += std::fabs(net[c]);
            sumGlob += net[c];
        }
        const scalar sumLocalContErr = f.deltaT * sumLoc / sumV;
        const scalar globalContErr   = f.deltaT * sumGlob / sumV;
        f.cumulativeContErr += globalContErr;
        res["contLocal"]      = sumLocalContErr;
        res["contGlobal"]     = globalContErr;
        res["contCumulative"] = f.cumulativeContErr;
        std::printf("time step continuity errors : sum local = %g, global = %g, cumulative = %g\n",
                    (double)sumLocalContErr, (double)globalContErr, (double)f.cumulativeContErr);
    }

    // p.relax() -- the FIELD factor, not the equation one. Both are named `p` in fvSolution and they live
    // in different sub-dictionaries; using the equation factor here relaxes the wrong thing.
    //
    // BOTH HALVES. GeometricField::relax is operator==(prevIter() + alpha*(*this - prevIter()))
    // (GeometricField.C:1094) and operator== assigns the boundary too (:1420), after the solve's own
    // correctBoundaryConditions (fvMatrixSolve.C:309) has put the solved cell value on every zeroGradient
    // face. The blend is what U = HbyA - rAU*grad(p) reads on the next line and, on a case without a
    // pressure limiter, what the next momentum assembly reads. On the totalPressure inlet it is a
    // different number from both the fresh p0 - 0.5*rho*|U_b|^2 and the previous one -- rhoTP at t=1
    // blends the 100200 seed towards 100095 at 0.3 -- where relaxing the internal field alone and
    // re-evaluating had left the fresh value. That variant was measured bit-identical on rhoBox, whose p
    // patches are all fixedValue or zeroGradient and for which the blend IS the re-evaluation; that is
    // why an earlier note here called the boundary half unverifiable. rhoTP's inlet sees it.
    f.p.evaluateBoundary();
    relaxField(f.p.internal, pPrevIter, in.relaxP);
    relaxBoundary(f.p, pBndPrevIter, in.relaxP);
    sd.scalars("pRel", f.p.internal);
    sd.surface("phi", f.phi);

    // U = HbyA - rAtU*fvc::grad(p), with rAtU on the SIMPLEC path and rAU otherwise.
    {
        const std::vector<vector> gradP = fvc::gaussGrad(f.p, m, g, patches);
        for (label c = 0; c < nC; ++c)
        {
            f.U.internal[c] = vector{ HbyA[c].x - rAUorAtU[c]*gradP[c].x,
                                      HbyA[c].y - rAUorAtU[c]*gradP[c].y,
                                      HbyA[c].z - rAUorAtU[c]*gradP[c].z };
        }
        // U.correctBoundaryConditions() -- pEqn.H:87 / pcEqn.H:100. Every flux-switched patch evaluates
        // through its updateCoeffs again here (the updated flag was cleared by the solve's evaluate), and
        // phi is the NEW flux by now (pEqn.H:73), so the piov mask and inletOutlet's valueFraction are
        // rebuilt from it and the piov value from the corrected cells: what gets written, what the
        // closure reads, and what the next momentum assembly finds.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            f.U.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
        }
        updatePressureInletOutletVelocity(f, patches);
        // flowRateInletVelocity too: its updateCoeffs reads rho's patch value every time it runs
        // (flowRateInletVelocityFvPatchVectorField.C:210-220, lookupPatchField(rhoName_)), and it runs
        // again inside this evaluate. On sbMatched (heRhoThermo, SIMPLEC) rho's patch value moved from
        // 0.3823 to 0.4097 between the momentum assembly and this point, so OpenFOAM's inlet reads 488.17
        // here where the mirror kept the assembly's 523.09 -- and the turbulent-intensity inlet, which
        // squares |U_b|, turned that 7% into k 1.8e-02 at t=2 (queue item 26). Recomputed here from the
        // rho patch value AS IT STANDS, which is what OpenFOAM's lookup returns.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            f.U.boundary[pi]->updateFromDensity(f.rho.boundary[pi]->value());
        }
        // freestreamVelocity too: its updateCoeffs reads `Up = *this`, the patch's CURRENT value
        // (freestreamVelocityFvPatchVectorField.C:106), and runs again inside this evaluate because the
        // solve's evaluate cleared the flag -- so OpenFOAM rebuilds the valueFraction TWICE per iteration,
        // here and at the momentum assembly, each time from the value the previous evaluate left. Rebuilt
        // at the top only, the closure read an inlet U 3.9e-04 off on 250 at naca0012's second iteration
        // (5.1e-07 relL2, every other closure input at 1e-10), which fed k 3.7e-06 and p 1.7e-05 by t=3
        // (queue item 25).
        {
            std::vector<std::vector<vector>> Ub(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) Ub[pi] = f.U.boundary[pi]->value();
            updateMixedFreestream(f.U.boundary, Ub, patches);
        }
        f.U.evaluateBoundary();
    }

    sd.vectors("Upost", f.U.internal);

    // pressureControl.limit(p), then the closed-volume mass correction, then the boundary re-evaluation
    // that either of them requires.
    const bool pLimited = f.pressureControl.limit(f.p.internal);
    if (closedVolume)
    {
        // p += (initialMass - fvc::domainIntegrate(psi*p))/fvc::domainIntegrate(psi)
        double num = 0.0, den = 0.0;
        for (label c = 0; c < nC; ++c)
        {
            num += (double)f.psi[c] * (double)f.p.internal[c] * (double)g.V()[c];
            den += (double)f.psi[c] * (double)g.V()[c];
        }
        const scalar dp = (scalar)(((double)f.initialMass - num) / den);
        for (label c = 0; c < nC; ++c) f.p.internal[c] += dp;
    }
    // p.correctBoundaryConditions() -- pEqn.H:100-103, keyed on `limitMaxP_ || limitMinP_` (pressureControl.C
    // limit(), true whenever a limit is CONFIGURED). On the totalPressure inlet this is an updateCoeffs
    // (the flag was cleared by the solve's evaluate), i.e. a recompute from the NEW flux, the corrected
    // U's patch value and rho's patch value as it stands -- BEFORE the tail's rho = thermo.rho() below.
    // It is the value written to disk and the one the next momentum assembly reads through grad(p).
    if (pLimited || closedVolume) updateTotalPressure(f, patches, /*evaluateAll=*/true);

    // rho = thermo.rho(), and rho.relax() only when NOT transonic.
    //
    // THE BOUNDARY IS RELAXED TOO. GeometricField::relax is operator==(prevIter + alpha*(*this -
    // prevIter)), and GeometricField::operator== assigns BOTH halves -- `internalFieldRef() = ...;
    // boundaryFieldRef() == ...` (GeometricField.C). Relaxing only the internal field leaves rho's patch
    // values at the unrelaxed thermo value, which is a different field from the one OpenFOAM carries.
    // It matters directly: flowRateInletVelocity holds the prescribed mass flow against rho's PATCH
    // values, so an unrelaxed boundary sets a different inlet velocity every iteration.
    {
        // Against rhoPrevIter, captured at the top of the step -- see the note there.
        updateRho(f, patches);
        if (!in.transonic)
        {
            relaxField(f.rho.internal, rhoPrevIter, in.relaxRho);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                std::vector<scalar> rb = f.rho.boundary[pi]->value();
                for (std::size_t i = 0; i < rb.size() && i < rhoBndPrevIter[pi].size(); ++i)
                {
                    rb[i] = rhoBndPrevIter[pi][i] + in.relaxRho * (rb[i] - rhoBndPrevIter[pi][i]);
                }
                f.rho.boundary[pi]->setStoredValues(std::move(rb));
            }
        }
        // Refresh the shared boundary snapshot so the closure sees ONE rho, both halves the same age
        // -- the rho OpenFOAM's turbulence->correct() sees, after the tail's `rho = thermo.rho();
        // rho.relax()`. Before this line the closure got the live cell rho beside an older boundary
        // snapshot (:216, or :395's on the consistent branch) while its OWN nuLamBnd read the live
        // boundary -- mixed ages inside one closure. MEASURED INERT at convergence on sbMatched
        // (k 3.46e-06 with and without: the consistent branch's mid-step refresh leaves only the
        // per-iteration tail delta, which vanishes as the run converges) -- kept as the alignment it
        // is, not as a numerics fix.
        for (std::size_t pi = 0; pi < patches.size(); ++pi) rhoBnd[pi] = f.rho.boundary[pi]->value();
    }

    sd.scalars("p", f.p.internal);
    sd.scalars("rhoTail", f.rho.internal);

    // turbulence->correct() -- LAST, after the pressure corrector, so the NEXT iteration's momentum
    // equation uses this iteration's closure. OpenFOAM's lagged coupling; correcting before UEqn instead
    // is a different algorithm that still converges to something plausible.
    // ...unless `turbulence off` froze the model: OpenFOAM's correct() returns on its first line then
    // (kEpsilon.C:216), so k, epsilon, nut and alphat keep their validate()-time values for the whole
    // run while the momentum and energy equations keep transporting rho*nut and alphat.
    if (f.turbulent && !f.turbulenceFrozen && !f.k.internal.empty())
    {
        // div(phi,k)/div(phi,epsilon) come from the CASE. The closures below assemble Gauss upwind and
        // Gauss limitedLinear (each with or without `bounded`) and nothing else, so any other named
        // scheme must refuse here -- running upwind under the case's name is the substitution this
        // project keeps finding.
        if (!in.turbDivUnsupported.empty())
            throw std::runtime_error(
                "rhoSimpleFoam step: div(phi,k)/div(phi,epsilon) asks for `" + in.turbDivUnsupported +
                "`, which the compressible closure does not assemble -- only Gauss upwind and Gauss "
                "limitedLinear, with or without `bounded`, are ported. Refusing rather than running "
                "upwind under the case's scheme name.");
        // The compressible instantiation's inputs. nu is the LAMINAR kinematic viscosity mu(T)/rho, which
        // varies cell by cell here where the incompressible lineage has one number for the case.
        std::vector<scalar> nuLam(nC);
        for (label c = 0; c < nC; ++c)
            nuLam[c] = transportMu(f.T.internal[c], f.thermo) / f.rho.internal[c];
        std::vector<std::vector<scalar>> nuLamBnd(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& tb = f.T.boundary[pi]->value();
            const std::vector<scalar>& rb = f.rho.boundary[pi]->value();
            nuLamBnd[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                nuLamBnd[pi][i] = transportMu(tb[i], f.thermo) / rb[i];
        }

        // compressibleTurbulenceModel::phi() -- the VOLUMETRIC flux, phi/fvc::interpolate(rho). divU is a
        // dilatation and must come from this, not from the mass flux the div operator uses.
        SurfaceScalarField phiByRho = f.phi;
        {
            // rhoBnd was refreshed after the tail's rho update + relax, so it IS the live boundary
            // here -- the separate rhoBndLive copy this block used to build (while the closure struct
            // below still took the stale pre-momentum snapshot) is gone with the defect.
            const SurfaceScalarField rhof =
                effectiveFaceViscosity(f.rho.internal, rhoBnd, m, g, patches);
            for (std::size_t fi = 0; fi < phiByRho.internal.size(); ++fi)
                phiByRho.internal[fi] /= rhof.internal[fi];
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                for (label i = 0; i < patches[pi].size; ++i)
                    phiByRho.boundary[pi][i] /= rhof.boundary[pi][i];
        }

        if (f.rasModel == "kOmegaSST")
        {
            // The same compressible instantiation, for the other closure. Both are ONE templated model in
            // OpenFOAM; what differs here is only which second scalar is transported and that kOmegaSST
            // needs the wall distance for its F1/F2 blends.
            kOmegaSST::Compressible sc;
            sc.rho      = &f.rho.internal;
            sc.rhoBnd   = &rhoBnd;
            sc.nu       = &nuLam;
            sc.nuBnd    = &nuLamBnd;
            sc.phiByRho = &phiByRho;
            if (f.alphat.internal.empty())
                throw std::runtime_error(
                    "rhoSimpleFoam_cpp: the case is RAS but has no alphat field. OpenFOAM's "
                    "EddyDiffusivity reads one when the turbulence model is constructed, and the energy "
                    "equation's alphaEff = CpByCpv*(alpha + alphat) needs it. Refusing rather than "
                    "running with alphat = 0.");
            sc.alphat   = &f.alphat.internal;
            sc.Prt      = f.Prt;   // the CASE's, from the field set -- see the kEpsilon branch

            // The case's gradSchemes for grad(k)/grad(omega), which CDkOmega and therefore F1 depend on.
            KOmegaSSTCoeffs sco = f.sstCoeffs;   // the CASE's, read with keCoeffs in createFields
            sco.gradKLimitK      = in.gradKLimitK;
            sco.gradULimitK      = in.gradULimitK;   // grad(U) for S2/GbyNu0

            const std::vector<scalar> y = cellWallDist(m, g, patches);
            kOmegaSST::SSTResiduals sres;
            // Instrument: BRAE_SST_DUMP_IN=<dir> writes the closure's INPUTS at the first iteration as plain
            // columns, so they can be held against OpenFOAM's stage_sst*In dumps (tools/dumpKOmegaSST)
            // when the closure itself is exact on OpenFOAM's inputs but the mirror's omega is not (item 25).
            if (const char* dd = std::getenv("BRAE_SST_DUMP_IN"))
            {
                static int calls = 0;
                const char* it = std::getenv("BRAE_SST_DUMP_ITER");   // which call to dump (default 1)
                if (++calls == (it ? std::atoi(it) : 1))
                {
                    auto dump = [&](const char* name, const std::vector<scalar>& v)
                    {
                        std::FILE* fp = std::fopen((std::string(dd) + "/" + name).c_str(), "w");
                        if (!fp) return;
                        for (scalar x : v) std::fprintf(fp, "%.17g\n", (double)x);
                        std::fclose(fp);
                    };
                    std::vector<scalar> ux(f.U.internal.size()), uy(ux.size()), uz(ux.size());
                    for (std::size_t c = 0; c < ux.size(); ++c)
                    {
                        ux[c] = f.U.internal[c].x;
                        uy[c] = f.U.internal[c].y;
                        uz[c] = f.U.internal[c].z;
                    }
                    dump("nutIn", f.nut.internal);
                    dump("kIn", f.k.internal);
                    dump("omegaIn", f.omega.internal);
                    dump("Ux", ux);
                    dump("Uy", uy);
                    dump("Uz", uz);
                    dump("rho", f.rho.internal);
                    dump("phi", f.phi.internal);
                    dump("nu", nuLam);
                    dump("y", y);
                    // ...and the boundary values the closure reads, per patch: OpenFOAM's freestream
                    // and flux-switched patches re-evaluate at every correctBoundaryConditions.
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        const std::vector<vector>& ub = f.U.boundary[pi]->value();
                        std::vector<scalar> bx(ub.size()), by(ub.size());
                        for (std::size_t i = 0; i < ub.size(); ++i) { bx[i] = ub[i].x; by[i] = ub[i].y; }
                        dump(("Ubx_" + patches[pi].name).c_str(), bx);
                        dump(("Uby_" + patches[pi].name).c_str(), by);
                        dump(("kb_" + patches[pi].name).c_str(), f.k.boundary[pi]->value());
                        dump(("omegab_" + patches[pi].name).c_str(), f.omega.boundary[pi]->value());
                        dump(("nutb_" + patches[pi].name).c_str(), f.nut.boundary[pi]->value());
                        dump(("phib_" + patches[pi].name).c_str(), f.phi.boundary[pi]);
                        dump(("rhob_" + patches[pi].name).c_str(), f.rho.boundary[pi]->value());
                    }
                    sres.captureStages = true;
                }
            }
            kOmegaSST::correct(f.U, f.k, f.omega, f.nut, f.phi, y, /*nu=*/0.0, m, g, patches,
                               in.relaxOmega, in.relaxK, in.tolTurb, in.relTolTurb, in.maxIterTurb,
                               sco, &sres, in.boundedTurb,
                               in.limitedLinearTurb, in.turbLimiterCoeff, in.linearUpwindTurb,
                               in.correctedLaplacian, in.snGradLimitCoeff, /*lm=*/nullptr, &sc,
                               in.minIterTurb, in.relaxEquationOmega, in.relaxEquationK);
            if (sres.captureStages)
            {
                // ...and the closure's own intermediates after the call, against OpenFOAM's stage_sst*.
                const std::string dd = std::getenv("BRAE_SST_DUMP_IN");
                auto dump = [&](const char* name, const std::vector<scalar>& v)
                {
                    std::FILE* fp = std::fopen((dd + "/" + name).c_str(), "w");
                    if (!fp) return;
                    for (scalar x : v) std::fprintf(fp, "%.17g\n", (double)x);
                    std::fclose(fp);
                };
                dump("divU", sres.divU);   dump("S2", sres.s2);       dump("GbyNu0", sres.gbyNu0); dump("G", sres.G);
                dump("CDkOmega", sres.CD); dump("F1", sres.f1);       dump("F23", sres.f23);
                dump("omD0", sres.omD0);   dump("omSrc0", sres.omSrc0); dump("omD", sres.omD);   dump("omSrc", sres.omSrc);
                dump("kD0", sres.kD0);     dump("kSrc0", sres.kSrc0);   dump("kD", sres.kD);     dump("kSrc", sres.kSrc);
                dump("omegaOut", f.omega.internal); dump("kOut", f.k.internal); dump("nutOut", f.nut.internal);
            }
            res["omega"] = sres.omega;
            res["k"]     = sres.k;
            f.alphat.evaluateBoundary();
            correctAlphatBoundary(f, patches);   // EddyDiffusivity's boundary half -- shared, see the header
            return res;
        }

        // REFUSE rather than silently substitute. Every model that is not kOmegaSST used to fall straight
        // through into the kEpsilon path below, and turbulence_setup.cuh has ALREADY swapped in the
        // SELECTED model's coefficient set by the time it arrives -- realizableKE takes C2 1.9,
        // sigmaEps 1.2, A0 4; RNGkEpsilon takes Cmu 0.0845, C1 1.42, C2 1.68, C3 -0.33 and
        // sigmak/sigmaEps 0.71942 (turbulence_setup.cuh:261-301). kEpsilonRef::correct reads
        // Cmu/C1/C2/C3/sigmaK/sigmaEps and NONE of the model flags, so what ran was standard-kEpsilon
        // arithmetic driven by another model's constants: a closure that exists in no source. No gate
        // could catch it either, because both sides of a brae-vs-brae comparison would run the same
        // fabrication. realizableKE has its own _cpp reference in the tree and is simply not wired here.
        if (f.rasModel != "kEpsilon")
        {
            throw std::runtime_error(
                "rhoSimpleFoam_cpp: RAS model '" + f.rasModel + "' is not ported on this path -- only "
                "kEpsilon and kOmegaSST are. Refusing rather than running kEpsilon in its place: this "
                "model's coefficients have already been substituted into the shared struct, so the "
                "result would be neither model.");
        }

        kEpsilonRef::Compressible comp;
        comp.rho      = &f.rho.internal;
        comp.rhoBnd   = &rhoBnd;
        comp.nu       = &nuLam;
        comp.nuBnd    = &nuLamBnd;
        comp.phiByRho = &phiByRho;
        // alphat is REQUIRED, not fabricated: every compressible RAS case ships one (it carries
        // compressible::alphatWallFunction at the walls, which brae could not invent), and a zeroed
        // stand-in would silently remove the turbulent contribution to the energy equation.
        if (f.alphat.internal.empty())
            throw std::runtime_error(
                "rhoSimpleFoam_cpp: the case is RAS but has no alphat field. OpenFOAM's EddyDiffusivity "
                "reads one when the turbulence model is constructed, and the energy equation's alphaEff "
                "= CpByCpv*(alpha + alphat) needs it. Refusing rather than running with alphat = 0.");
        comp.alphat   = &f.alphat.internal;
        // THE CASE'S Prt, from the field set, not StepInput's. StepInput::Prt defaults to 1.0 and no
        // caller in the tree ever assigned it, so alphat = rho*nut/Prt ran at 1.0 whatever the case
        // asked for -- while createFields had already used the case's own value for the CONSTRUCTION
        // alphat. Initialisation and the loop disagreed. EddyDiffusivity.C:36 reads it from the model's
        // coeffDict with default 1.0, which is what readThermoCoeffs parsed into f.thermo.Prt.
        comp.Prt      = f.Prt;

        // The case's laplacianScheme governs the turbulence diffusion terms too -- OF resolves
        // laplacian(DkEff,k) and laplacian(DepsilonEff,epsilon) against the same laplacianSchemes the
        // momentum, energy and pressure equations use, and every rhoSimpleFoam tutorial sets
        // `default Gauss linear corrected`.
        // THE CASE'S coefficients, same reason. StepInput::keCoeffs is a default-constructed
        // KEpsilonCoeffs that nothing assigned, so a case naming `kEpsilonCoeffs { Cmu 0.1; C2 1.8; }`
        // solved with 0.09 and 1.92. OpenFOAM reads six of them (kEpsilon.C:199-204) and createFields
        // now reads all six.
        KEpsilonCoeffs keco     = f.keCoeffs;
        keco.correctedLaplacian = in.correctedLaplacian;
        keco.snGradLimitCoeff   = in.snGradLimitCoeff;
        keco.gradULimitK        = in.gradULimitK;   // grad(U) for the production
        keco.gradKLimitK        = in.gradKLimitK;   // grad(k)/grad(epsilon) for the laplacian corrections

        sd.scalars("kIn", f.k.internal);
        sd.scalars("epsIn", f.epsilon.internal);
        sd.scalars("nutIn", f.nut.internal);
        kEpsilonRef::KEResiduals kres;
        kres.captureStages = sd.on;   // the as-solved systems, for the device twin's dump to diff against
        kEpsilonRef::correct(f.U, f.k, f.epsilon, f.nut, f.phi, /*nu=*/0.0, m, g, patches,
                             in.relaxEpsilon, in.relaxK, in.tolTurb, in.relTolTurb, in.maxIterTurb,
                             keco, &kres, in.boundedTurb, /*dropTerm=*/0, &comp, in.fvOpts,
                             in.relaxEquationEps, in.relaxEquationK, /*constrainBeforeWall=*/true,
                             in.limitedLinearTurb, in.turbLimiterCoeff, in.minIterTurb);
        res["epsilon"] = kres.epsilon;
        res["k"]       = kres.k;
        sd.scalars("kOut", f.k.internal);
        sd.scalars("epsD", kres.epsD);
        sd.scalars("epsSrc", kres.epsSrc);
        sd.scalars("epsUpper", kres.epsUpper);
        sd.scalars("epsLower", kres.epsLower);
        sd.scalars("kD", kres.kD);
        sd.scalars("kSrc", kres.kSrc);
        sd.scalars("kUpper", kres.kUpper);
        sd.scalars("kLower", kres.kLower);
        sd.scalars("G", kres.G);
        sd.scalars("Gwall", kres.Gwall);
        sd.scalars("eps0", kres.eps0);
        sd.scalars("gByNu", kres.gByNu);
        sd.scalars("divU", kres.divU);
        sd.scalars("DkEff", kres.DkEff);
        sd.scalars("DepsilonEff", kres.DepsilonEff);
        sd.scalars("epsOut", f.epsilon.internal);
        sd.scalars("nutOut", f.nut.internal);
        f.alphat.evaluateBoundary();

        correctAlphatBoundary(f, patches);   // EddyDiffusivity's boundary half -- shared, see the header
    }
    return res;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
