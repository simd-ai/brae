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
#include <cmath>
#include <stdexcept>

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
    // The boundary temperature is NOT recomputed from he here: T's patch values come from T's own
    // boundary conditions, which is what basicThermo::correct() leaves in place -- it recalculates the
    // INTERNAL field and lets the boundary conditions stand. psi's patch values follow that T.
    f.T.evaluateBoundary();
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& tb = f.T.boundary[pi]->value();
        const std::vector<scalar>& pb = f.p.boundary[pi]->value();
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
        // p as well: the momentum equation reads p's boundary VALUE through -fvc::grad(p), and in
        // OpenFOAM that value is the blend the previous iteration's updateCoeffs left behind.
        updateMixedFreestream(f.p.boundary, Ub, patches);
        f.p.evaluateBoundary();
    }
    f.U.evaluateBoundary();
    f.he.evaluateBoundary();
    f.T.evaluateBoundary();

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

    // ---- UEqn.H ----
    RhoMomentumInput uin;
    uin.phi = &f.phi.internal;   uin.phiBnd = &f.phi.boundary;
    uin.rho = &f.rho.internal;   uin.rhoBnd = &rhoBnd;
    uin.muEff = &muEff;          uin.muEffBnd = &muEffBnd;
    uin.relaxU             = in.relaxU;
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
    const FvVectorMatrix UEqn = assembleUEqn(f.U, uin, m, g, patches);
    {
        // solve(UEqn == -fvc::grad(p)) on a COPY: the pressure equation needs the ORIGINAL UEqn for
        // A(), H() and H1(), and addPressureGradient would otherwise leave the source carrying -grad(p).
        FvVectorMatrix Mp = UEqn;
        addPressureGradient(Mp, f.p, m, g, patches);
        const SolverPerformance up = solveVector(Mp, f.U, m, patches, in.tolU, in.relTolU, in.maxIter);
        res["U"] = up.initialResidual;
    }
    f.U.evaluateBoundary();

    // ---- EEqn.H ----
    {

        EnergyInput ein;
        ein.phi = &f.phi.internal;  ein.phiBnd = &f.phi.boundary;
        ein.alphaEff = &alphaEff;   ein.alphaEffBnd = &alphaEffBnd;
        ein.heName            = f.heName;
        ein.relaxHe           = in.relaxHe;
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
            pbicgstab(E, f.he.internal, m, patches, in.tolHe, in.relTolHe, in.maxIter);
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
    pin.hasFixedFluxPressure = in.hasFixedFluxPressure;

    const std::vector<scalar> pOld = f.p.internal;
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
        P = assemblePcEqn(st, f.p, pin, m, g, patches);
        const SolverPerformance pp =
            pbicgstab(P, f.p.internal, m, patches, in.tolP, in.relTolP, in.maxIter);
        res["p"] = pp.initialResidual;
        rAUorAtU     = st.rAtU;
        HbyA         = st.HbyA;
        phiHbyA      = st.phiHbyA;
        closedVolume = st.closedVolume;
    }
    else
    {
        const PressureStages st = pressurePredictor(UEqn, f.U, f.p, pin, m, g, patches);
        P = assemblePEqn(st, f.p, pin, m, g, patches);
        const SolverPerformance pp =
            pbicgstab(P, f.p.internal, m, patches, in.tolP, in.relTolP, in.maxIter);
        res["p"] = pp.initialResidual;
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

    // p.relax() -- the FIELD factor, not the equation one. Both are named `p` in fvSolution and they live
    // in different sub-dictionaries; using the equation factor here relaxes the wrong thing.
    relaxField(f.p.internal, pOld, in.relaxP);
    f.p.evaluateBoundary();

    // U = HbyA - rAtU*fvc::grad(p), with rAtU on the SIMPLEC path and rAU otherwise.
    {
        const std::vector<vector> gradP = fvc::gaussGrad(f.p, m, g, patches);
        for (label c = 0; c < nC; ++c)
        {
            f.U.internal[c] = vector{ HbyA[c].x - rAUorAtU[c]*gradP[c].x,
                                      HbyA[c].y - rAUorAtU[c]*gradP[c].y,
                                      HbyA[c].z - rAUorAtU[c]*gradP[c].z };
        }
        f.U.evaluateBoundary();
    }

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
    if (pLimited || closedVolume) f.p.evaluateBoundary();

    // rho = thermo.rho(), and rho.relax() only when NOT transonic.
    //
    // THE BOUNDARY IS RELAXED TOO. GeometricField::relax is operator==(prevIter + alpha*(*this -
    // prevIter)), and GeometricField::operator== assigns BOTH halves -- `internalFieldRef() = ...;
    // boundaryFieldRef() == ...` (GeometricField.C). Relaxing only the internal field leaves rho's patch
    // values at the unrelaxed thermo value, which is a different field from the one OpenFOAM carries.
    // It matters directly: flowRateInletVelocity holds the prescribed mass flow against rho's PATCH
    // values, so an unrelaxed boundary sets a different inlet velocity every iteration.
    {
        const std::vector<scalar> rhoOld = f.rho.internal;
        std::vector<std::vector<scalar>> rhoBndOld(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) rhoBndOld[pi] = f.rho.boundary[pi]->value();
        updateRho(f, patches);
        if (!in.transonic)
        {
            relaxField(f.rho.internal, rhoOld, in.relaxRho);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                std::vector<scalar> rb = f.rho.boundary[pi]->value();
                for (std::size_t i = 0; i < rb.size() && i < rhoBndOld[pi].size(); ++i)
                {
                    rb[i] = rhoBndOld[pi][i] + in.relaxRho * (rb[i] - rhoBndOld[pi][i]);
                }
                f.rho.boundary[pi]->setStoredValues(std::move(rb));
            }
        }
    }

    // turbulence->correct() -- LAST, after the pressure corrector, so the NEXT iteration's momentum
    // equation uses this iteration's closure. OpenFOAM's lagged coupling; correcting before UEqn instead
    // is a different algorithm that still converges to something plausible.
    if (f.turbulent && !f.k.internal.empty())
    {
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
            sc.Prt      = in.Prt;

            // The case's gradSchemes for grad(k)/grad(omega), which CDkOmega and therefore F1 depend on.
            KOmegaSSTCoeffs sco = in.sstCoeffs;
            sco.gradKLimitK      = in.gradKLimitK;

            const std::vector<scalar> y = cellWallDist(m, g, patches);
            kOmegaSST::SSTResiduals sres;
            kOmegaSST::correct(f.U, f.k, f.omega, f.nut, f.phi, y, /*nu=*/0.0, m, g, patches,
                               in.relaxOmega, in.relaxK, in.tolTurb, in.relTolTurb, in.maxIter,
                               sco, &sres, in.boundedTurb,
                               in.sstLimitedLinear, in.sstLimiterCoeff, in.sstLinearUpwind,
                               in.correctedLaplacian, in.snGradLimitCoeff, /*lm=*/nullptr, &sc);
            res["omega"] = sres.omega;
            res["k"]     = sres.k;
            f.alphat.evaluateBoundary();
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
        comp.Prt      = in.Prt;

        // The case's laplacianScheme governs the turbulence diffusion terms too -- OF resolves
        // laplacian(DkEff,k) and laplacian(DepsilonEff,epsilon) against the same laplacianSchemes the
        // momentum, energy and pressure equations use, and every rhoSimpleFoam tutorial sets
        // `default Gauss linear corrected`.
        KEpsilonCoeffs keco     = in.keCoeffs;
        keco.correctedLaplacian = in.correctedLaplacian;
        keco.snGradLimitCoeff   = in.snGradLimitCoeff;

        kEpsilonRef::KEResiduals kres;
        kEpsilonRef::correct(f.U, f.k, f.epsilon, f.nut, f.phi, /*nu=*/0.0, m, g, patches,
                             in.relaxEpsilon, in.relaxK, in.tolTurb, in.relTolTurb, in.maxIter,
                             keco, &kres, in.boundedTurb, /*dropTerm=*/0, &comp, in.fvOpts);
        res["epsilon"] = kres.epsilon;
        res["k"]       = kres.k;
        f.alphat.evaluateBoundary();
    }
    return res;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
