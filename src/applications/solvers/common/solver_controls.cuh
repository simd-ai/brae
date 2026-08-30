#pragma once
// SIMPLE solver configuration + per-step residual report -- the PODs that parameterise DeviceSimpleSolver.
// Split out of device_simple_foam.cuh so the solver interface, PIMPLE, rhoSimple etc. can share the controls
// without pulling in the full solver implementation. Pure data (+ the model-coeff PODs); no device code.
#include "cf_types.cuh"
#include "kepsilon_coeffs.cuh"
#include "komega_sst_coeffs.cuh"
#include "spalart_coeffs.cuh"
#include "smagorinsky_coeffs.cuh"
#include "deshybrid_coeffs.cuh"
#include <string>
#include <vector>

namespace brae {

// Which nut wall function to apply, chosen from the 0/nut boundaryField TYPE (matching OpenFOAM),
// NOT from the turbulence model. nutk = k-based stepwise log law (nutkWallFunction); Spalding =
// velocity-based Spalding blend (nutUSpaldingWallFunction); Blended = velocity-based binomial n=4
// blend (nutUBlendedWallFunction). All three are honoured on any RAS model, exactly as OF does.
enum class NutWall { Nutk, Spalding, Blended, NutU, LowRe };   // NutU = nutUWallFunction (STEPWISE
// blender); LowRe = nutLowReWallFunction, whose calcNut() returns Zero UNCONDITIONALLY
// (nutLowReWallFunctionFvPatchScalarField.C:38-42 is the whole function). It used to be mapped to
// Nutk with a warning on stderr, on the argument that the two are identical on a resolved mesh --
// which is false as stated, because nutk s yPlus is the K-BASED Cmu^0.25*y*sqrt(k)/nu, not
// u_tau*y/nu, so a mesh resolved in friction units can still take nutk s log branch where OpenFOAM
// returns exactly 0. A warning is not a refusal and a substitution is not a port.

struct DeviceSimpleControls
{
    scalar nu = 1e-5, relaxU = 0.7, relaxP = 0.3, relaxK = 0.7, relaxEps = 0.7;
    // The same factors on the LAST outer corrector (OF's ".*Final" idiom). Default to the ordinary
    // factor so a case without Final entries behaves exactly as before.
    scalar relaxUFinal = 0.7, relaxPFinal = 0.3, relaxKFinal = 0.7, relaxEpsFinal = 0.7;
    vector bodyForce{0, 0, 0};                     // constant momentum source (drives periodic/cyclic channels). +V*g.
    scalar tolU = 1e-8, tolP = 1e-7, tolKE = 1e-8;
    scalar relTolU = 0.0, relTolP = 0.0, relTolKE = 0.0;   // solver relTol (fvSolution solvers.{U,p,k,epsilon}.relTol). 0 = abs tol.
    // fvSolution's `Final` VARIANTS (solvers.pFinal / UFinal / kFinal ...). PIMPLE spends its early outer
    // correctors getting close and its last one getting the answer, so OF gives the last one its own,
    // tighter settings: pimpleControl::loop calls mesh.data().setFinalIteration(true) on the final outer
    // corrector, and fvMatrix::solve() then resolves its dictionary through GeometricField::select(),
    // which appends "Final" to the field name. Every solve inside that outer corrector uses them --
    // including each inner PISO pressure corrector, since the flag is per outer iteration.
    //
    // brae read only the base entries, so a case's pFinal was inert: with the oscillatingInletACMI2D
    // settings (p: 1e-5/relTol 0.01, pFinal: 1e-10/relTol 0) that left the FINAL flux converged to the
    // loose figure and reported time-step continuity of 1.7e-04 where OF reaches 1e-14. It cost a long
    // detour in the cyclicACMI hunt, because a continuity residual that size reads as a coupling defect.
    // brae's dict audit had been printing `solvers/pFinal/ (whole sub-dictionary never read)` throughout.
    //
    // Absent entries fall back to the base ones (OF would FatalError instead; refusing a case that OF
    // runs is worse than solving its last corrector exactly as tightly as the others).
    // fvSolution solvers.<field>.{maxIter,minIter}. OF's lduMatrix::solver reads both (defaults 1000 and
    // 0) and they are NOT cosmetic: LES/NACA4412 caps the pressure solve at `maxIter 10`, and on its
    // impulsive first step GAMG leaves with a residual of 4.26 against an initial 1 -- an unconverged
    // field by construction. A solver that instead runs to convergence produces a different (better, but
    // different) answer, and the two cannot be compared on that step at all. minIter forces iterations
    // even when the initial residual already passes, which is how a case pins down "always do one sweep".
    int    maxIterU = 1000, maxIterP = 1000, maxIterKE = 1000;
    int    minIterU = 0,    minIterP = 0,    minIterKE = 0;
    int    maxIterUFinal = 1000, maxIterPFinal = 1000, maxIterKEFinal = 1000;
    int    minIterUFinal = 0,    minIterPFinal = 0,    minIterKEFinal = 0;
    scalar tolUFinal = 1e-8, tolPFinal = 1e-7, tolKEFinal = 1e-8;
    scalar relTolUFinal = 0.0, relTolPFinal = 0.0, relTolKEFinal = 0.0;
    // TWO flags, because OF selects the Final entry two different ways and the difference is visible in
    // its log on any PISO-mode case (nOuterCorrectors 1, two pressure correctors):
    //
    //   GAMG: Solving for p, Initial residual = 0.0513, Final residual = 2.786e-04, No Iterations 4
    //   GAMG: Solving for p, Initial residual = 0.0042, Final residual = 5.988e-11, No Iterations 16
    //
    // corrector 1 stopped on `p` (relTol 0.01), corrector 2 on `pFinal` (1e-10, relTol 0) -- so pFinal
    // is NOT simply "the last outer iteration". pEqn.H asks for it explicitly:
    //     pEqn.solve(p.select(pimple.finalInnerIter()))
    // and finalInnerIter() = last PISO corrector AND last non-orthogonal pass (pimpleControlI.H:98),
    // gated on the outer iteration only when `finalOnLastPimpleIterOnly` is set, which defaults false.
    //
    // UEqn.H and the turbulence models instead call the bare solve(), which resolves through
    // fvMatrix::solverDict() -> psi_.select(mesh.data().isFinalIteration()) -- the flag pimpleControl
    // sets for the whole of the final OUTER corrector. Same case, same log: Ux converges to 3.1e-07
    // from an initial 3.3e-03, i.e. past the `U` relTol of 0.1, so it is on UFinal every step.
    //
    // Neither is ever set on the steady path: OF's simpleControl has no Final concept either.
    bool   finalInner = false;   // p        -- last pressure corrector, last non-orth pass
    bool   finalIter  = false;   // U, k/eps -- anywhere in the last outer corrector
    int    pMaxIter() const { return finalInner ? maxIterPFinal : maxIterP; }
    int    pMinIter() const { return finalInner ? minIterPFinal : minIterP; }
    int    uMaxIter() const { return finalIter  ? maxIterUFinal : maxIterU; }
    int    uMinIter() const { return finalIter  ? minIterUFinal : minIterU; }
    scalar pTol()     const { return finalInner ? tolPFinal     : tolP; }
    scalar pRelTol()  const { return finalInner ? relTolPFinal  : relTolP; }
    scalar uRelax()   const { return finalIter  ? relaxUFinal   : relaxU; }
    scalar pRelax()   const { return finalIter  ? relaxPFinal   : relaxP; }
    scalar kRelax()   const { return finalIter  ? relaxKFinal   : relaxK; }
    scalar epsRelax() const { return finalIter  ? relaxEpsFinal : relaxEps; }
    scalar uTol()     const { return finalIter  ? tolUFinal     : tolU; }
    scalar uRelTol()  const { return finalIter  ? relTolUFinal  : relTolU; }
    scalar keTol()    const { return finalIter  ? tolKEFinal    : tolKE; }
    scalar keRelTol() const { return finalIter  ? relTolKEFinal : relTolKE; }
    // fvSolution PIMPLE/finalOnLastPimpleIterOnly (pimpleControl.C:53, default false).
    bool   finalOnLastPimpleIterOnly = false;
    int    bicgCheckEvery = 1;      // batched convergence for ALL BiCGStab solves (momentum + k/eps); BRAE_BICG_CHECK_EVERY.
    bool   turbulent = false;
    bool   useGraph  = true;     // replay the pressure V-cycle from a cached CUDA graph (#7c-loop)
    bool   bounded   = false;    // "bounded Gauss upwind": add -Sp(div(phi),U). Set from fvSchemes; default off.
    bool   consistent = false;   // SIMPLEC (rAtU consistency correction, p=1). Set from fvSolution SIMPLE.consistent.
    bool   linearUpwind = false; // div(phi,U) "linearUpwind": add the deferred gradient correction. Set from fvSchemes.
    // div(phi,U) bare "linear" = central differencing (OF linear.H:106 -> the geometric weights).
    // Standard for LES, where upwind dissipation would damp the resolved turbulence the model exists to
    // capture. Unbounded by construction, which is why cases pair it with `bounded`.
    bool   divULinear = false;
    bool   linearUpwindV = false;// div(phi,U) "linearUpwindV": linearUpwind + OF's vector direction limiter (also sets linearUpwind).
    // div(phi,U) "limitedLinearV k": OF's NVDVTVDV vector limiter -- ONE limiter per face built from the
    // whole velocity vector, blending the central and upwind weights IMPLICITLY (no deferred correction).
    // laminar { model Maxwell; } -- the viscoelastic stress transport. nuM is the polymer viscosity and
    // lambda its relaxation time; both are material properties, not closure constants, and OF refuses
    // the model without them.
    // PIMPLE/momentumPredictor (OF pimpleControl, default TRUE). When off, pimpleFoam still ASSEMBLES
    // and relaxes UEqn -- rAU and HbyA come from it -- but never solves it: U is updated only by the
    // pressure corrector. laminar/planarPoiseuille turns it off, and solving anyway made its first step
    // 56% fast (0.00535 against 0.00343 m/s) because the predictor moved U before the corrector did.
    bool   momentumPredictor = true;
    // PIMPLE/correctPhi (OF createDyMControls.H, default = mesh.dynamic()). After the mesh moves, the
    // stored flux belongs to the OLD geometry: OF rebuilds the ABSOLUTE flux from the mapped surface
    // velocity (phi = Sf & Uf) and then PROJECTS it divergence-free by solving a pcorr Poisson, before
    // any momentum or pressure equation sees it. Skipping it leaves the step starting from a flux that
    // does not satisfy continuity on the mesh it is about to be used on.
    // PIMPLE loop contract (OF pimpleControl). These are not tuning knobs -- each selects part of the
    // pressure-velocity algorithm, so an unread one means solving a DIFFERENT algorithm.
    //
    // turbOnFinalIterOnly defaults to TRUE in OpenFOAM: turbulence->correct() runs ONCE per time step,
    // on the final outer corrector, because the outer loop is iterating the pressure-velocity coupling,
    // not the turbulence transport. brae corrected on every outer corrector, which on a case with
    // nOuterCorrectors 5 advanced the turbulence model five times per physical step. A 30-case sweep
    // missed it: most tutorials use nOuterCorrectors 1, where the two cadences coincide.
    bool   turbOnFinalIterOnly = true;
    bool   solveFlow = true;            // OF: skip the momentum/pressure solve entirely when false
    bool   simpleRho = false;           // OF SIMPLErho: compressible rho update cadence; inert here
    // PIMPLE residualControl: outer-loop convergence. When every controlled field meets its absolute or
    // RELATIVE tolerance, OF runs ONE more iteration flagged as final and then leaves the loop early.
    // That changes which iteration is `final`, and therefore which solver settings are used and when
    // turbulence is corrected -- so it is not a cost-only control.
    struct OuterResidualControl { std::string field; scalar absTol = 0; scalar relTol = 0; };
    std::vector<OuterResidualControl> outerResidualControl;
    // WHICH PATCHES CARRY THE TURBULENCE WALL FUNCTION, one entry per fvPatch, 1 = yes. OpenFOAM applies
    // epsilonWallFunction/omegaWallFunction per BOUNDARY CONDITION -- they are BC objects on the
    // epsilon/omega field, so only a patch whose BC is one gets a cornerWeights_ entry and an
    // epsilon0/G0 override. brae selected those cells by PATCH TYPE instead, which is a different set
    // whenever a `wall`-typed patch carries a plain BC. EMPTY means "fall back to the patch type",
    // which is what the SA and LES paths use -- they have no epsilon/omega field to read.
    std::vector<char> turbWallPatch;
    // OF createDyMControls.H, default FALSE: the mesh moves once per time step, on the first outer
    // iteration. When true it moves before EVERY outer corrector, so the outer loop converges the mesh
    // position alongside the pressure-velocity coupling.
    bool   moveMeshOuterCorrectors = false;
    // dynamicFvMesh::controlledUpdate -- dynamicMeshDict updateControl/updateInterval (OF's timeControl
    // with the "update" prefix). Default is every step.
    std::string meshUpdateControl = "always";
    int         meshUpdateInterval = 1;
    bool   correctPhi = false;
    // fvSolution solvers.pcorr -- OF's own tutorials give it a LOOSE tolerance (0.02) and relTol 0,
    // because the projection only has to remove the mapping error, not converge a pressure field.
    scalar tolPcorr = 0.02, relTolPcorr = 0.0;
    int    maxIterPcorr = 1000;
    bool   maxwell = false;
    scalar maxwellNuM = 0.0;
    scalar maxwellLambda = 0.0;
    bool   divSigmaVanAlbada = false;   // div(phi,sigma) Gauss vanAlbada (what both Maxwell tutorials name)
    scalar relaxSigma = 1.0;            // relaxationFactors/equations/sigma (OF sigmaEqn.relax(); absent -> none)
    bool   gsSigma = false;             // solvers/sigma smoothSolver + a GaussSeidel smoother
    bool   divULimitedV = false;
    scalar divUTwoBykV  = 2.0;   // OF limitedLinearLimiter twoByk_ = 2/max(k, SMALL); k = 1 -> 2
    // OF LUST.H's two shares: weights() = 0.75*linear + 0.25*linearUpwind::weights(), and
    // correction() = 0.25*linearUpwind::correction(). Named here so the matrix and the deferred
    // correction cannot drift apart -- they have to sum to 1.
    static constexpr scalar lustCentralFrac = 0.75;
    static constexpr scalar lustUpwindFrac  = 0.25;
    bool   lust = false;         // div(phi,U) "LUST": deferred correction = 0.75*linear + 0.25*linearUpwind (OF LUST.H).
    bool   nonOrth = false;      // laplacian "corrected"|"limited": nonOrthDeltaCoeffs implicit + explicit corrVec.grad correction. Set from fvSchemes.
    scalar nonOrthLimit = 1.0;   // snGrad "limited <psi>" coeff (OF fv::limitedSnGrad); 1.0 = "corrected" (unlimited). Set from fvSchemes.
    int    nNonOrth = 0;         // SIMPLE.nNonOrthogonalCorrectors: extra pressure-correction passes (pEqn re-solved nNonOrth+1 times). Set from fvSolution.
    scalar gradULimitK = 0.0;    // grad(U) "cellLimited Gauss linear <k>" coeff (OF cellLimitedGrad<minmod>); 0 = unlimited. Set from fvSchemes.
    bool   limitedK = false, limitedEps = false;  // div(phi,k|epsilon) "limitedLinear": implicit limited weight. Set from fvSchemes.
    bool   luK = false, luEps = false;             // div(phi,k|epsilon|nuTilda) "linearUpwind": deferred gradient correction. Set from fvSchemes.
    bool   gsK = false, gsEps = false;             // scalar linear solver = smoothSolver+symGaussSeidel (read from fvSolution solvers.{k|nuTilda} / {epsilon|omega}).
    bool   gsU = false;                            // momentum linear solver = smoothSolver+(sym)GaussSeidel (read from fvSolution solvers.U).
    // fvSolution asked for `preconditioner DILU` on U (OF's default for the momentum equations, and the
    // entry brae used to substitute Jacobi for). Only meaningful on the BiCGStab path -- a smoothSolver
    // has no preconditioner in OF either.
    bool   diluU = false;
    // DILU on the TURBULENCE solve (k, epsilon/omega, nuTilda), when the case asks for it. Separate from
    // diluU because the two are different fvSolution entries and a case can name one without the other.
    //
    // This is not a cost knob. Measured on turbulentFlatPlate:kEpsilon at y+ ~ 1, over 60 consecutive k
    // solves: OpenFOAM DILU-preconditioned PBiCGStab reduces the residual to a median 0.0064 of initial,
    // overshooting the case relTol of 0.1 by more than 10x in a single iteration, while brae Jacobi
    // BiCGStab stops right at the threshold, median 0.0726. In the stiff k-epsilon pair that leaves the
    // two fields mutually inconsistent every outer iteration and the case DIVERGES; solving them to
    // 1e-3 instead makes the same code converge to U 1.04e-05 of OpenFOAM.
    bool   diluKE = false;
    scalar twoBykK = 2.0, twoBykEps = 2.0;         // 2/max(k_,SMALL) from the limitedLinear coefficient (k_=1 -> 2).
    // div(phi,h|e) and div(phi,K|Ekp) -- the ENERGY equation's convection scheme (rhoSimpleFoam). Read from
    // fvSchemes like every other div scheme; brae used to hardcode upwind here and silently ignore what the
    // case asked for, which is a first-order downgrade that still converges. The K/Ekp term takes the same
    // scheme because OF's tutorials point div(phi,K) at the same entry ($energy) as div(phi,e).
    // div(phi,h|e) "bounded": OF's boundedConvectionScheme subtracts fvm::Sp(fvc::surfaceIntegrate(phi), he).
    // It vanishes at converged continuity, so every steady duct gate passes without it -- but it is a real
    // STABILISER while div(phi) != 0, and the NACA0012 case at Mach 0.72 diverges in one iteration without it.
    bool   boundedHe = false;
    // Same story for the turbulence scalars. OF's `bounded` is a per-SCHEME wrapper
    // (boundedConvectionScheme wraps one div), so div(phi,k) can be bounded while div(phi,U) is not.
    // brae read `bounded` only off the div(phi,U) line and reused that one flag for every scalar, so
    // demo/delta and demo/f16 (`div(phi,U) Gauss LUST` + `div(phi,nuTilda) bounded ...`) ran nuTilda
    // UNBOUNDED. The term is -Sp(div(phi),f): on an upwind row it takes diag from sum|outflow phi| to
    // sum|inflow phi| = sum|offdiag|, i.e. it guarantees marginal diagonal dominance. Without it any
    // cell with net inflow (everywhere, before continuity converges) is not diagonally dominant.
    bool   boundedK   = false;   // div(phi,k) / div(phi,nuTilda)
    bool   boundedEps = false;   // div(phi,epsilon) / div(phi,omega)
    bool   limitedHe = false;                      // div(phi,h|e) "limitedLinear"
    bool   luHe = false;                           // div(phi,h|e) "linearUpwind"
    scalar twoBykHe = 2.0;
    // div(phi,K|Ekp), the KINETIC term -- a separate fvSchemes entry from div(phi,h|e) and a separate
    // discretisation in OF. All four keys used to OR into the He slots, so a case asking for
    // `div(phi,e) linearUpwind` + `div(phi,K) upwind` ran the kinetic term second-order anyway (and the
    // reverse ran the energy equation first-order). They agree in every stock tutorial, where both are
    // `$energy` -- which is exactly why the merge went unnoticed.
    bool   boundedKin = false;
    bool   limitedKin = false;
    bool   luKin = false;
    scalar twoBykKin = 2.0;
    bool   foundKinScheme = false;                 // false -> mirror the He slots (the old behaviour)
    // OF's RAS/LES `turbulence` switch (RASModel.C:70, getOrDefault<Switch>("turbulence", true)). `off`
    // makes correct() return immediately: k, epsilon/omega and nut stay FROZEN at their initial values
    // while the momentum equation keeps using that frozen nut. It is NOT the same as `simulationType
    // laminar`, where nut is zero and never read. brae never read the switch, so a case asking to freeze
    // the turbulence ran a fully live model. Found by dict_audit (E5) once the audit was made to run on
    // REFUSED cases -- aerofoilNACA0012 sets it, and aerofoilNACA0012 refuses on fvOptions.
    bool   turbulenceOn = true;
    // B1: SIMPLE/transonic. OF's rhoSimpleFoam pEqn.H takes a different branch entirely -- the pressure
    // equation gains an IMPLICIT convection term fvm::div(phid, p) with phid = (psi_f/rho_f)*phiHbyA, the
    // part of phiHbyA now carried implicitly is subtracted off, and the matrix is relaxed (the subsonic
    // branch relaxes only the FIELD, never the equation). Running a transonic case down the subsonic
    // branch converges to a wrong answer in silence, which is why this was a refusal until now.
    bool   transonic = false;
    // OF fvMatrix::relax() applies relaxationFactors/equations/<field> and does NOTHING when the entry is
    // absent (fvMatrix.C:1250-1263 -- it only relaxes if relaxEquation() finds one). Both facts matter:
    // the factor AND whether it was given.
    bool   hasRelaxPEqn = false;
    scalar relaxPEqn = 1.0;
    // grad(<turbulence scalar>) / grad(energy) cellLimited coefficients. Separate from gradULimitK: OF takes
    // one gradScheme entry per field, and aerofoilNACA0012 limits grad(U), grad(k) and grad(omega) alike.
    // The cellLimited coeff of the gradient the linearUpwind DIV STATEMENT names -- distinct from
    // gradULimitK/gradKLimitK, which come from an explicit gradSchemes `grad(U)`/`grad(k)` entry.
    // OF keeps these apart: linearUpwind.C takes mesh.gradScheme(gradSchemeName_), while kEpsilon.C's
    // production uses plain fvc::grad(U) -> gradSchemes `default`. Conflating them limits the production
    // gradient on a case that never asked for it, which collapses GbyNu and with it epsilon.
    // OF laminarModels::generalizedNewtonian + powerLaw viscosity (simulationType laminar, laminar{}).
    // Replaces the molecular viscosity with a strain-rate-dependent one; see device_generalized_newtonian.
    bool   gnPowerLaw = false;
    scalar gnNuMin = 0.0, gnNuMax = 0.0, gnN = 1.0;
    scalar gradULULimitK = 0.0;                    // linearUpwind's own grad(U)
    scalar gradKLULimitK = 0.0;                    // linearUpwind's own grad(k)/grad(epsilon)
    scalar gradKLimitK = 0.0;                      // grad(k) / grad(omega) / grad(epsilon) / grad(nuTilda)
    scalar gradHeLimitK = 0.0;                     // grad(h) / grad(e)
    // The cellLimited coefficient of the gradient the KINETIC term's linearUpwind uses. Separate from
    // gradHeLimitK because fvSchemes gives div(phi,K|Ekp) its own entry, which may name a different
    // gradient from div(phi,h|e); it falls back to the energy's when the case omits it.
    scalar gradKinLimitK = 0.0;                    // grad(K) / grad(Ekp)
    KEpsilonCoeffs keCoeffs;                       // k-eps model coeffs (default = OF); read from turbulenceProperties RAS.kEpsilonCoeffs.
    bool   sst = false;                            // RASModel kOmegaSST (the "second turbulence scalar" eps slot holds omega).
    bool   lm = false;                             // RASModel kOmegaSSTLM (sst + Langtry-Menter gamma-ReThetat transition).
    KOmegaSSTCoeffs ksstCoeffs;                    // kOmegaSST coeffs (default = OF); read from RAS.kOmegaSSTCoeffs.
    bool   sa = false;                             // RASModel SpalartAllmaras (one-equation: the "k" slot holds nuTilda; no 2nd scalar).
    bool   des = false;                            // SpalartAllmarasDDES/kOmegaSSTDDES (simulationType LES): DES length-scale limiter; needs sa/sst=true.
    bool   iddes = false;                          // SpalartAllmarasIDDES: the improved (WMLES-capable) length scale on the SA DES path; implies des+sa.
    SpalartAllmarasCoeffs saCoeffs;                // SA coeffs (default = OF) + nutUSpaldingWallFunction E/kappa + CDES (DES) + IDDES blending constants.
    bool   les = false;                            // pure LES Smagorinsky (simulationType LES): ALGEBRAIC sub-grid nut,
                                                   // NO transport scalar (no k/epsilon/omega/nuTilda). Mutually exclusive with sa/sst/des.
    SmagorinskyCoeffs smagCoeffs;                  // Smagorinsky coeffs (default = OF Ck=0.094, Ce=1.048); read from LES.SmagorinskyCoeffs.
    // WALE: the same algebraic-LES slot, a different velocity scale. `les` stays the flag for
    // "nut is algebraic, there is no transport scalar"; this only picks which formula fills it.
    bool wale = false;
    // fvSchemes wallDist `method exactDistance`: the true Euclidean distance to the wall surface, rather
    // than OF's default connectivity-propagated meshWave. Different y => a different F1/F2/dTilda.
    bool exactWallDist = false;
    // div(phi,U) `Gauss DEShybrid <s1> <s2> <delta> ...`: a per-face blend of a low-dissipation scheme
    // and an upwind-biased one, driven by a DES sensor. See deshybrid_coeffs.cuh.
    // LES `delta maxDeltaxyz`: the filter width is OF's face-normal hmax rather than cubeRootVol.
    // Every consumer of the filter width (Smagorinsky, WALE, the SA/SST DES length scale, DEShybrid)
    // must use the SAME one, or the scheme and the model disagree about the resolved scale.
    bool   lesDeltaMax = false;
    scalar lesDeltaCoeff = 2.0;
    bool desHybrid = false;
    DesHybridCoeffs desCoeffs;
    WaleCoeffs waleCoeffs;
    NutWall nutWall = NutWall::Nutk;               // nut wall function, read from the 0/nut wall BC TYPE (not the model),
                                                   // so nutUBlended/nutUSpalding are honoured on kEpsilon/kOmegaSST like OF.
    scalar atmZ0 = 0.0;                            // atmNutkWallFunction roughness length z0 (>0 -> rough-wall nut on the
    bool   atmBoundNut = true;                     // k-based path); read from the 0/nut wall BC. 0 -> smooth nutkWallFunction.
    bool   needRef = false;                        // no fixedValue-p patch -> singular pressure: adjustPhi + setReference.
    label  pRefCell = 0;
    scalar pRefValue = 0.0;                        // pressure reference (fvSolution SIMPLE.pRefCell/pRefValue).
    int    pcgCheckEvery = 1;      // pressure AMG-PCG residual-read cadence (BRAE_PCG_CHECK_EVERY). 1 = exact per-iter
                                   // (bit-identical); K>1 cuts (K-1)/K of the PCG host syncs (overshoots conv by <K).
    bool   corrScaling = false;    // AMG coarse-correction scaling + flexible CG (BRAE_CORR_SCALING). Cuts AMG cycles
                                   // ~2x at scale (graded meshes); nonlinear precond -> not bit-identical to off.
    std::string caseDir = ".";     // case directory, for the AMG hierarchy cache (constant/polyMesh/.brae_amgcache).
    bool   writeCache = false;     // write the mesh + AMG caches this run (set by -partition or BRAE_MESH_CACHE).
};

// OF-style per-field solve report for one SIMPLE step (matches OpenFOAM's "Solving for Ux/Uy/Uz/p" + continuity block).
struct DeviceSimpleResidual
{
    scalar Ux = 0, Uy = 0, Uz = 0, p = 0, pFinal = 0;
    scalar UxFinal = 0, UyFinal = 0, UzFinal = 0;
    int UxIters = 0, UyIters = 0, UzIters = 0;     // U per-component final + nIter
    int pIters = 0;
    // energy. Captured here rather than left in the turbulence report because correctTurbulence() clears
    // that store before the turbulence solves, which would wipe the EEqn entry the energy already pushed.
    scalar he = 0, heFinal = 0;
    int    heIters = 0;
    scalar contLocal = 0, contGlobal = 0;          // time step continuity errors, raw (driver applies deltaT + cumulative)
};

} // namespace brae
