// brae's interFoam time loop -- see inter_driver_cpp.cuh for why the driver owns no numerics and for
// the four old-time fields that are its actual content.
#include "inter_driver_cpp.cuh"
#include "inter_solve_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include "time_controls.cuh"
#include "foam_field_reader.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

// A zeroGradient copy of a cell field, for the fvc:: calls that need a boundary.
GeometricField<scalar> zgField(const std::vector<scalar>& cells, const std::vector<FvPatch>& patches)
{
    GeometricField<scalar> f;
    f.internal = cells;
    for (const FvPatch& q : patches)
        f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    f.evaluateBoundary();
    return f;
}

}   // namespace


RunReport runInterFoam(
    const std::string& caseDir,
    const std::string& startDir,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    label nSteps,
    bool verbose,
    InterFields* fieldsOut,
    scalar endTime,
    PressureTaps* pressureTaps)
{
    InterFields f = buildInterFields(caseDir, startDir, m, g, patches);
    const label nC = m.nCells();

    // The four old-time fields. See the header: each is read by something different, and each is
    // correct in isolation, which is why losing one is invisible to any single equation's gate.
    std::vector<scalar> alphaOld = f.alpha1.internal;
    std::vector<vector> UOld     = f.U.internal;
    std::vector<scalar> rhoOld   = f.rho;
    SurfaceScalarField  phiOld   = f.phi;

    SurfaceScalarField prevCorr;                 // alphaApplyPrevCorr's cache
    RunReport rep;
    rep.deltaT = f.deltaT;

    for (label step = 0; step < nSteps; ++step)
    {
        // Time::run() (Time.C:1000), and it sits HERE -- above CourantNo.H and setDeltaT.H -- so the
        // deltaT it tests is the one the previous iteration ended with, not the one this iteration is
        // about to choose. The half-step slack is OpenFOAM's: a run overshoots endTime by up to half a
        // step rather than landing on it, unless writeControl is adjustableRunTime and Time::
        // adjustDeltaT() trims the last few.
        if (!(rep.time < endTime - scalar(0.5)*rep.deltaT)) break;

        // The stages, in interFoam.C's order. runTimeStep owns the order; this lambda owns the work.
        // THE CASE'S OWN PIMPLE CONTROLS, read in buildInterFields. These were hardcoded here until
        // damBreak's fvSolution was actually read: it says nOuterCorrectors 1, nCorrectors 3 AND
        // `momentumPredictor no`, and the last of those changes which algorithm runs.
        const LoopControls lc = f.pimple;

        SolverHooks hooks;
        hooks.run = [&](Stage s)
        {
            switch (s)
            {
                case Stage::courantNo:
                {
                    const std::vector<scalar> sumPhi = surfaceSumMagPhi(
                        m.owner(), m.neighbour(), f.phi.internal,
                        [&]{ std::vector<scalar> b;
                             for (const auto& p : f.phi.boundary) b.insert(b.end(), p.begin(), p.end());
                             return b; }(),
                        nC, m.nInternalFaces());
                    rep.CoNum = courantNo(sumPhi, g.V(), rep.deltaT).CoNum;
                    // alphaCourantNo needs the same sumPhi, so it is cached on the report rather than
                    // recomputed -- OpenFOAM's two #includes build it twice, which is the one place
                    // this driver deliberately differs and it changes no number.
                    rep.alphaCoNum = alphaCourantNo(sumPhi, f.alpha1.internal, g.V(), rep.deltaT).CoNum;
                    break;
                }
                case Stage::alphaCourantNo:  break;    // computed above, from the same sumPhi
                case Stage::setDeltaT:
                    rep.deltaT = setDeltaTVoF(rep.deltaT, rep.CoNum, rep.alphaCoNum, f.timeCtl,
                                              rep.time, &f.writeCadence);
                    break;
                case Stage::advanceTime:
                    rep.time += rep.deltaT;
                    ++rep.steps;
                    // Time::operator++ moves writeTimeIndex_ AFTER the time, with the step that took
                    // it -- which is what the next adjustDeltaT measures the distance to.
                    f.writeCadence.advance(rep.time, rep.deltaT);
                    break;

                case Stage::alphaControls: break;      // read once, in buildInterFields

                case Stage::alphaEqnSubCycle:
                {
                    AlphaStepInput ai;
                    ai.phi = &f.phi; ai.phiCN = &f.phi;   // Euler: ocCoeff 0, so phiCN IS phi
                    ai.cAlpha = f.interface.cAlpha;
                    ai.nAlphaCorr = f.alphaCtl.nAlphaCorr;
                    ai.icAlpha = f.alphaCtl.icAlpha;
                    ai.scAlpha = f.alphaCtl.scAlpha;
                    ai.rho1 = f.mixture.phases.rho1; ai.rho2 = f.mixture.phases.rho2;
                    ai.alphaScheme  = f.divPhiAlpha;
                    ai.alpharScheme = f.divPhirbAlpha;
                    ai.MULESCorr = f.alphaCtl.MULESCorr;
                    ai.alphaApplyPrevCorr = f.alphaCtl.alphaApplyPrevCorr;
                    ai.alpha2BndOut = &f.alpha2Bnd;

                    auto step1 = [&](const std::vector<scalar>& aOld, scalar dtSub,
                                     std::vector<scalar>& aNew, SurfaceScalarField& rPhi)
                    {
                        AlphaStepInput sub = ai;
                        sub.deltaT = dtSub;
                        SurfaceScalarField aPhi;
                        alphaEqnStep(f.alpha1, aOld, sub, f.interface, f.mulesCtl,
                                     m, g, patches, aPhi, rPhi, f.nHatf, f.K, &prevCorr);
                        aNew = f.alpha1.internal;
                    };
                    alphaEqnSubCycle(f.alphaCtl.nAlphaSubCycles, rep.deltaT,
                                     f.alpha1.internal, alphaOld, f.rhoPhi, step1);
                    // ...and the boundary with it. Dropping this was tried together with the reset in
                    // alphaEqnStep: damBreak's alpha went thirty times further from OpenFOAM and
                    // capillaryRise did not move, so the extra evaluations are load-bearing rather
                    // than spurious.
                    f.alpha1.evaluateBoundary();
                    break;
                }

                case Stage::mixtureCorrect:
                {
                    // rho == alpha1*rho1 + alpha2*rho2 (alphaEqnSubCycle.H:36), and the viscosities
                    // with it. rho.oldTime() is NOT touched: fvm::ddt's source needs the value from
                    // the start of the step, and this is where it would be lost.
                    for (label c = 0; c < nC; ++c) f.alpha2[c] = scalar(1) - f.alpha1.internal[c];
                    cpu::twoPhase::mixtureRho(f.alpha1.internal, f.alpha2, f.mixture.phases, f.rho);
                    cpu::twoPhase::mixtureMu (f.alpha1.internal, f.mixture.phases, f.mu);
                    cpu::twoPhase::mixtureNu (f.alpha1.internal, f.mu, f.mixture.phases, f.nu);
                    // THE BOUNDARY BLENDS FIRST, from alpha's patch values as they stand NOW. `rho ==`
                    // (alphaEqnSubCycle.H:36) and calcNu() both read them before the curvature below
                    // moves them: immiscibleIncompressibleTwoPhaseMixture::correct() is calcNu() THEN
                    // interfaceProperties::correct() (immiscibleIncompressibleTwoPhaseMixture.H:78-82),
                    // and at a contact-angle wall the second call rewrites alpha's gradient and so its
                    // patch value. Blending after it gave the laplacian a wall viscosity one contact-
                    // angle pass ahead: measured on capillaryRise against OpenFOAM's own UEqn.A() after
                    // one step, exact in every cell off the wall and 1.26% high in the air cells at it,
                    // which was the whole of the 0.2% that step leaves in U.
                    updateMixtureBoundary(f, patches);
                    // ...THEN THE CURVATURE. interFoam.C:154 calls mixture.correct() here, after the
                    // sub-cycle and before UEqn, and interfaceProperties::correct() IS calculateK.
                    // Rebuilding only rho/mu/nu leaves UEqn's surface-tension force one pass behind.
                    interfaceProps::calculateK(f.alpha1, f.interface, m, g, patches, false, f.nHatf, f.K);
                    break;
                }

                case Stage::UEqn:
                case Stage::pEqn:
                {
                    // The face force and the pressure corrector share every field they are built from,
                    // so they are assembled together here and pEqn runs on the matrix UEqn produced.
                    // Splitting them across two hook calls would mean rebuilding the curvature.
                    if (s == Stage::pEqn) break;       // done in the UEqn pass, see above

                    // UEqn.H uses mixture.surfaceTensionForce(), which reads the K the LAST
                    // mixture.correct() left -- it does not recompute one. An extra pass here would
                    // put UEqn's force one iteration ahead of the alpha equation's.
                    std::vector<scalar> sK;
                    interfaceProps::sigmaK(f.K, f.interface.sigma, sK);
                    const SurfaceScalarField sKf = fvc::interpolate(sK, m, g, patches);

                    const GeometricField<scalar> rhoF = zgField(f.rho, patches);
                    const SurfaceScalarField snRho = fvc::snGrad(rhoF, m, g, patches, false);
                    const SurfaceScalarField snA   = fvc::snGrad(f.alpha1, m, g, patches, false);

                    SurfaceScalarField stf;
                    stf.internal.resize(static_cast<std::size_t>(m.nInternalFaces()));
                    for (label fi = 0; fi < m.nInternalFaces(); ++fi)
                        stf.internal[fi] = sKf.internal[fi]*snA.internal[fi];
                    // THE BOUNDARY IS NOT ZERO. surfaceTensionForce() is a surfaceScalarField and at a
                    // contact-angle wall snGrad(alpha1) is the gradient correctContactAngle just wrote
                    // -- 9681 on capillaryRise -- so this is precisely where the contact angle enters
                    // the pressure equation. Zeroing it cost 6% of the velocity at step 1, all of it
                    // at the wall.
                    stf.boundary.assign(patches.size(), std::vector<scalar>{});
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        const FvPatch& q = patches[pi];
                        stf.boundary[pi].resize(static_cast<std::size_t>(q.size));
                        for (label i = 0; i < q.size; ++i)
                            stf.boundary[pi][i] = sKf.boundary[pi][i] * snA.boundary[pi][i];
                    }

                    // constrainPressure before snGrad(p_rgh): a fixedFluxPressure gradient is
                    // PRESCRIBED, and brae refuses to assemble one that has not been set.
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                        if (f.p_rgh.boundary[pi]->updateableSnGrad())
                            f.p_rgh.boundary[pi]->updateSnGrad(
                                std::vector<scalar>(static_cast<std::size_t>(patches[pi].size), scalar(0)));
                    const SurfaceScalarField snP = fvc::snGrad(f.p_rgh, m, g, patches, false);

                    SurfaceScalarField force;
                    {
                        std::vector<scalar> out;
                        momentumSourceFlux(stf.internal, f.ghfInternal, snRho.internal,
                                           snP.internal, g.magSf(), out);
                        force.internal = out;
                        force.boundary.assign(patches.size(), std::vector<scalar>{});
                        for (std::size_t pi = 0; pi < patches.size(); ++pi)
                        {
                            const FvPatch& q = patches[pi];
                            momentumSourceFlux(stf.boundary[pi], f.ghfBoundary[pi],
                                               snRho.boundary[pi], snP.boundary[pi],
                                               q.magSf, force.boundary[pi]);
                        }
                    }

                    std::vector<std::vector<scalar>> rhoB(patches.size()), nuB(patches.size()),
                                                     phB(patches.size());
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        const FvPatch& q = patches[pi];
                        rhoB[pi].resize(static_cast<std::size_t>(q.size));
                        nuB[pi].resize(static_cast<std::size_t>(q.size));
                        phB[pi] = f.rhoPhi.boundary.size() > pi
                                ? f.rhoPhi.boundary[pi]
                                : std::vector<scalar>(static_cast<std::size_t>(q.size), scalar(0));
                        // alpha's PATCH values, not the face cell's -- see InterFields::rhoBnd.
                        rhoB[pi] = f.rhoBnd[pi];
                        nuB[pi]  = f.nuBnd[pi];
                    }

                    InterMomentumInput mi;
                    mi.rhoPhi = &f.rhoPhi.internal; mi.rhoPhiBnd = &phB;
                    mi.rho = &f.rho; mi.rhoOld = &rhoOld; mi.rhoBnd = &rhoB;
                    mi.UOld = &UOld;
                    mi.nuEff = &f.nu; mi.nuEffBnd = &nuB;
                    mi.deltaT = rep.deltaT;
                    mi.scheme = f.divRhoPhiU;
                    mi.schemeCoeff = f.divRhoPhiUCoeff;
                    mi.relaxEquationU = f.relaxEquationU; mi.relaxU = f.relaxU;

                    MomentumSolveControls msc;
                    FvVectorMatrix UEqn;
                    momentumPredictor(f.U, mi, force, msc, m, g, patches,
                                      f.momentumPredictorOn, UEqn);

                    DdtCorrInput dc;
                    dc.phiOld = &phiOld; dc.UOld = &UOld; dc.deltaT = rep.deltaT;

                    PressureStepInput pin;
                    pin.UEqn = &UEqn; pin.rho = &f.rho; pin.gh = &f.gh; pin.ghf = &f.ghfInternal;
                    pin.ghfBnd = &f.ghfBoundary;
                    pin.stf = &stf; pin.snGradRho = &snRho; pin.ddt = &dc;
                    pin.taps = pressureTaps;

                    PressureSolveControls psc;
                    psc.nCorrectors = lc.nCorrectors;
                    psc.nNonOrthogonalCorrectors = f.nNonOrthogonalCorrectors;
                    psc.needReference = false;          // damBreak's atmosphere is totalPressure
                    // the CASE's own solve, not a hardcoded 1e-9. BRAE_PTOL still overrides, because
                    // a device-vs-host gate has to pin both sides to one stopping point.
                    const char* ptol = std::getenv("BRAE_PTOL");
                    psc.tolP = ptol ? std::atof(ptol) : f.pSolve.tol;
                    psc.relTolP = ptol ? scalar(0) : f.pSolve.relTol;
                    psc.maxIterP = f.pSolve.maxIter;
                    psc.pcgDIC = f.pSolve.pcgDIC();
                    psc.tolPFinal = ptol ? std::atof(ptol) : f.pSolveFinal.tol;
                    psc.relTolPFinal = ptol ? scalar(0) : f.pSolveFinal.relTol;
                    psc.maxIterPFinal = f.pSolveFinal.maxIter;
                    psc.pcgDICFinal = f.pSolveFinal.pcgDIC();
                    for (label c = 0; c < lc.nCorrectors; ++c)
                    {
                        psc.finalCorrector = (c == lc.nCorrectors - 1);
                        pressureCorrector(f.p_rgh, f.U, f.phi, f.p, pin, psc, m, g, patches);
                    }
                    // alpha1's inletOutlet reads the flux the step ended on, at the next MULES pass.
                    pushFluxToPatches(f, patches);
                    break;
                }

                case Stage::turbulenceCorrect: break;   // laminar
                case Stage::write:            break;
            }
        };

        runTimeStep(lc, hooks);

        // ...and the old-time set moves forward, all four together.
        alphaOld = f.alpha1.internal;
        UOld     = f.U.internal;
        rhoOld   = f.rho;
        phiOld   = f.phi;

        if (verbose)
        {
            scalar aMin = f.alpha1.internal[0], aMax = f.alpha1.internal[0], mass = 0, maxU = 0;
            for (label c = 0; c < nC; ++c)
            {
                aMin = std::fmin(aMin, f.alpha1.internal[c]);
                aMax = std::fmax(aMax, f.alpha1.internal[c]);
                mass += f.alpha1.internal[c]*g.V()[c];
                const vector& v = f.U.internal[c];
                maxU = std::fmax(maxU, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
            }
            // %.17g on t and dt -- round-trip precision -- because these two are what a clock gate
            // reads back. At %.9g the gate bottomed out at 2.389e-09 on host AND device and stayed
            // there when the fields moved four orders closer to OpenFOAM's: it was measuring this
            // printf. See tests/interfoam_dambreak_clock_vs_openfoam.sh.
            std::printf("  t = %.17g  dt = %.17g  Co %.3f  alphaCo %.3f  "
                        "alpha [%.3e, %.15g]  max|U| %.4f\n",
                        (double)rep.time, (double)rep.deltaT, (double)rep.CoNum,
                        (double)rep.alphaCoNum, (double)aMin, (double)aMax, (double)maxU);
        }
    }

    // the final state
    rep.alphaMin = f.alpha1.internal[0];
    rep.alphaMax = f.alpha1.internal[0];
    rep.alphaMass = 0;
    rep.maxU = 0;
    for (label c = 0; c < nC; ++c)
    {
        rep.alphaMin = std::fmin(rep.alphaMin, f.alpha1.internal[c]);
        rep.alphaMax = std::fmax(rep.alphaMax, f.alpha1.internal[c]);
        rep.alphaMass += f.alpha1.internal[c]*g.V()[c];
        const vector& v = f.U.internal[c];
        rep.maxU = std::fmax(rep.maxU, std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z));
    }
    {
        const std::vector<scalar> d = fvc::div(f.phi, m, g, patches);
        for (scalar v : d) rep.worstDivPhi = std::fmax(rep.worstDivPhi, std::fabs(v));
    }
    if (fieldsOut) *fieldsOut = std::move(f);
    return rep;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
