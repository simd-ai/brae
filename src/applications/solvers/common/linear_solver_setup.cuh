#pragma once
// linear_solver_setup.cuh -- fvSolution -> DeviceSimpleControls, for EVERY solver driver.
//
// This exists because the compressible driver was ported by copying the parts of gpuSimpleFoam it needed
// to make the physics run, and fifteen controls were left behind. Each one then fell back to a struct
// default whose own comment said it should come from fvSolution:
//
//   relTol{P,U,KE}   unset -> 0     => every equation solved to ABSOLUTE tolerance every outer iteration,
//                                      where OF does a loose solve (all six rhoSimpleFoam tutorials set it)
//   consistent       unread         => "consistent yes" silently ran SIMPLE instead of SIMPLEC
//   nNonOrth         unset -> 0     => nNonOrthogonalCorrectors ignored
//   gs{U,K,Eps}      unset -> false => "solver smoothSolver" ignored, always BiCGStab
//   {bicg,pcg}CheckEvery, useGraph, corrScaling, bodyForce
//
// None of that is visible in a converged field on a case that happens not to use them, which is why it
// survived. Reading them in ONE place is the structural fix: a new driver gets the whole set or none.
//
// Everything here is read-only on the dicts and writes only into ctl.

#include "solver_controls.cuh"
#include <regex>
#include "foam_dict.cuh"
#include "brae_notice.cuh"   // noticeApproximated / noticeIgnored
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace brae {

// fvSolution solvers.{p,U,<second>,...} tolerances/relTol + smoothSolver selection, and the SIMPLE
// sub-dict controls (consistent / nNonOrthogonalCorrectors / bodyForce).
//
// secondName is the 2nd turbulence scalar's FIELD name: "omega" on kOmegaSST, "epsilon" on kEpsilon,
// unused when ctl.sa (SA solves nuTilda only). Pass it explicitly -- deriving it here would re-hardcode
// the thing that already went wrong once.
//
// algorithmDict is the fvSolution sub-dict holding the algorithm controls: "SIMPLE" for the steady
// solvers, "PIMPLE" for the transient one. OF parameterises this the same way -- solutionControl reads
// consistent/nNonOrthogonalCorrectors from subOrEmptyDict(algorithmName_) (solutionControl.C:46,51,302)
// and the name is a constructor argument defaulting to "SIMPLE" in simpleControl.H:100 and "PIMPLE" in
// pimpleControl.H:135. Hardcoding "SIMPLE" here would read nNonOrthogonalCorrectors as 0 on every
// transient case, so this is required, not cosmetic.
// WHAT THE CALLING DRIVER ACTUALLY RUNS, in OpenFOAM's own vocabulary, so the notices below compare like
// with like. It cannot be derived from the dictionary: two drivers reading the same fvSolution wire
// different things, and a notice describing the wrong one is as misleading as no notice at all -- it
// teaches the reader to discount them all.
//
// THE NAMES ARE OPENFOAM'S, and that is the whole point. PBiCGStab is "preconditioned bi-conjugate
// gradient stabilized ... using a run-time selectable preconditioner" (PBiCGStab.H), so brae's BiCGStab
// with a diagonal preconditioner IS `solver PBiCGStab; preconditioner diagonal;` and its DILU form IS
// `solver PBiCGStab; preconditioner DILU;`. Calling either "Jacobi-BiCGStab" and comparing that string
// against the dict's `PBiCGStab` announced a substitution on every field brae was running exactly as
// asked -- on essentially every tutorial in existence, since PBiCGStab is what they all name. Likewise
// brae's AMG-PCG is an AMG-preconditioned CG (device_amg.cuh), which is OF's `PCG` +
// `preconditioner GAMG`, NOT OF's `GAMG` -- that is multigrid as the solver, a different algorithm.
struct SolverRunsAs
{
    // The pressure equation. The default is what every driver but the OF-mirror runs. Those drivers fall
    // back to a Jacobi-preconditioned CG on an interface-coupled mesh (cyclic/AMI, where the Galerkin
    // coarse operator cannot represent the interface edges), which this description does not carry: the
    // caller knows at run time, the reader does not.
    //
    // `GAMG` here names OpenFOAM's PRECONDITIONER, and it is the honest name for brae's V-cycle in both
    // places one runs: preconditioning a PCG on the symmetric pressure, and (since 2026-09-08, on the
    // OF-mirror's transonic branch) preconditioning a PBiCGStab on the asymmetric one -- OpenFOAM
    // registers GAMGPreconditioner in the asymmetric constructor table as well
    // (GAMGPreconditioner.C:37-42), so `solver PBiCGStab; preconditioner GAMG;` is a legal setting there.
    // A CALLER THAT SETS THIS PAIR MUST PRINT ITS OWN LINE: noticeSolverChoice below is silent whenever
    // the case names exactly what the caller runs, and brae's V-cycle is not OpenFOAM's GAMG (its own
    // agglomeration, a weighted-Jacobi smoother, one V-cycle where GAMGPreconditioner defaults to
    // nVcycles 2, an FP32 cycle by default), so that silence is a lie unless the caller breaks it.
    // rhoSimpleFoamDriver.cu's `transonic p preconditioner` notice is the worked example.
    std::string pSolver = "PCG";
    std::string pPrecon = "GAMG";
    // Does this driver precondition the ENERGY solve with DILU when the case asks? Only the OF-mirror
    // does; every compressible tutorial writes the energy field into the same regex block as U and k, so
    // a driver honouring two of the three still substitutes on the third.
    bool diluOnEnergy = false;
    // ...and does it honour `solver smoothSolver` on the ENERGY equation? Only the OF-mirror's CUDA arm
    // does. The flag exists because the notice is shared: a driver that substitutes must still say so,
    // and one that honours must not claim a substitution it no longer makes.
    bool smoothSolverOnEnergy = false;
    // ...and on momentum / the turbulence pair. Default true: every other driver has run OpenFOAM's own
    // sweep on those since the level-scheduled smoother landed. The OF-mirror sets all three from one
    // switch so the gate can compare the honoured path against the substituted one with the notices
    // truthful in BOTH modes.
    bool smoothSolverOnMomentum = true;
    bool smoothSolverOnTurbulence = true;
    // ...and does it run DILU whatever the dict says? The OF-mirror's HOST arm does -- pbicgstab.cuh is
    // DILU throughout -- so a case asking `none` or `diagonal` there gets DILU and has a right to be told.
    bool alwaysDilu = false;
    // ...and does the CALLER announce the momentum solve itself? The OF-mirror's colour Gauss-Seidel
    // experiment runs a solver this reader has no name for -- a multicolour Gauss-Seidel smoothSolver,
    // which is neither OpenFOAM's own sweep (what gsU = true asserts, and what silences the U notice)
    // nor the PBiCGStab the gsU = false notice would announce. With this set the reader still reads U's
    // tolerance, relTol, maxIter, minIter and nSweeps exactly as before and prints NO U notice; the
    // driver prints the one that is true. Default false: every other caller keeps the reader's lines.
    bool momentumNoticedByCaller = false;
    // The ONE field whose solver path runs OpenFOAM's fixed-count nSweeps branch (a negative nSweeps,
    // smoothSolver.C:95-119); the reader hands that field its raw value and refuses it on every other.
    std::string fixedSweepsField;
};

// The degree the substituted PBiCGStab's Neumann series runs at on the transported turbulence scalars.
// 10 from the sweep in bench/rhoSimpleFoam/eps_precond_experiment.py: degree 3 leaves min(epsilon) at
// 169.3 where DILU leaves 182.6, degree 6 at 179.1, degree 10 at 180.5, and degree 16 does not improve
// on 10 (it converges the preconditioned residual faster and so stops after fewer BiCGStab iterations,
// at a comparable iterate, for 6 more SpMVs).
constexpr int POLY_DEG_KE_DEFAULT = 10;

// THE ONE RULE for what preconditions a substituted PBiCGStab on a transported turbulence scalar, as a
// free function because there are TWO callers and they used to decide separately: readLinearSolverControls
// below, and the V2 simpleFoam driver, whose copy read `preconditioner` itself, keyed its escape hatch on
// BRAE_DILU rather than BRAE_DILU_KE, and had no Neumann series at all -- so a V2 case naming GAMG on the
// pair still got the bare diagonal that item 78 removed everywhere else. Sharing the decision is the point;
// a second copy is how the two drift.
//
//   the case NAMES a preconditioner   -> honour it (DILU, or the diagonal it asked for)
//   it names none, and the field is relaxed by alpha < 1
//                                     -> the degree-10 Neumann series, whose convergence ratio
//                                        fvMatrix::relax then bounds by alpha (see turbRelaxBound)
//   it names none and there is no such bound
//                                     -> DILU, which needs none
//   the field runs as a smoothSolver  -> neither; there is no preconditioner in that path
struct TurbPreconChoice
{
    bool dilu = false;
    int  polyDeg = 1;                                  // 1 == plain Jacobi (the series' first term)
};
inline TurbPreconChoice turbPreconFor(const FoamDict* solvers,
                                      const FoamDict& fvSolution,
                                      const std::string& field,
                                      bool gs)
{
    TurbPreconChoice c;
    if (gs) return c;                                  // a smoothSolver carries no preconditioner
    const FoamDict* sd = solvers ? solvers->subDict(field) : nullptr;
    const std::string prec = sd ? sd->wordOr("preconditioner", "") : std::string();
    if (prec == "DILU") c.dilu = true;
    if (const char* e = std::getenv("BRAE_DILU_KE")) c.dilu = (std::atoi(e) != 0);
    if (c.dilu) return c;
    if (!prec.empty()) return c;                       // a named non-DILU preconditioner is the case's own
    // The blank a substitution leaves. fvMatrix::relax forces D >= sum|offdiag| and THEN divides by the
    // factor (fvMatrix.C:105-113), so a relaxed equation has sum|offdiag|/|a_ii| <= alpha and the series'
    // ratio is bounded by alpha, mesh-independently. Take the LARGEST factor that will be applied: a
    // `".*Final" 1.0` corrector has no bound even where the ordinary factor does.
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
    const FoamDict* src = eqs ? eqs : rf;
    scalar alpha = 1.0;
    if (src)
    {
        const scalar a0 = src->scalarOr(field, scalar(1));
        alpha = std::fmax(a0, src->found(field + "Final") ? src->scalarOr(field + "Final", a0) : a0);
    }
    if (alpha < scalar(1)) c.polyDeg = POLY_DEG_KE_DEFAULT;
    else                   c.dilu = true;              // no bound -> the factorisation, which needs none
    if (const char* e = std::getenv("BRAE_POLY_KE"))
    {
        c.polyDeg = std::max(1, std::atoi(e));
        if (c.polyDeg > 1) c.dilu = false;
    }
    return c;
}

inline void readLinearSolverControls(
    const FoamDict& fvSolution,
    const std::string& secondName,
    DeviceSimpleControls& ctl,
    const std::string& algorithmDict = "SIMPLE",
    // The energy field's name (h or e), for the compressible callers. Empty = no energy equation.
    const std::string& heName = "",
    // What the CALLER actually runs -- see SolverRunsAs. Defaulted to the legacy/incompressible drivers,
    // so every existing call site keeps describing itself correctly.
    const SolverRunsAs& runsAs = SolverRunsAs{})
{
    const FoamDict* solvers = fvSolution.subDict("solvers");

    auto solverTol = [&](const std::string& f, scalar def)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return s ? s->scalarOr("tolerance", def) : def;
    };
    auto solverRelTol = [&](const std::string& f)   // SIMPLE only needs a loose per-step solve
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return s ? s->scalarOr("relTol", 0.0) : 0.0;
    };
    // OF lduMatrix::solver reads maxIter (default 1000) and minIter (default 0) from the same
    // sub-dictionary. Both change WHERE the solve stops, so an unread `maxIter 10` is not a performance
    // detail -- see the note in DeviceSimpleControls.
    auto solverMaxIter = [&](const std::string& f, int def)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return s ? static_cast<int>(s->scalarOr("maxIter", def)) : def;
    };
    auto solverMinIter = [&](const std::string& f, int def)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return s ? static_cast<int>(s->scalarOr("minIter", def)) : def;
    };
    // Does this driver precondition field f with DILU? Not a property of the dictionary -- see
    // SolverRunsAs -- and the BRAE_DILU / BRAE_DILU_KE hatches move it again, so the notices consult the
    // same answer the solve will use rather than a second copy of the rule.
    auto diluHere = [&](const std::string& f) -> bool
    {
        if (runsAs.alwaysDilu) return true;
        const bool wires = (f == "U" || f == "k" || f == secondName || f == "nuTilda")
                        || (runsAs.diluOnEnergy && !heName.empty() && f == heName);
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        bool on = wires && s && s->wordOr("preconditioner", "") == "DILU";
        const char* e = nullptr;
        if (f == "U") e = std::getenv("BRAE_DILU");
        else if (f == "k" || f == secondName || f == "nuTilda") e = std::getenv("BRAE_DILU_KE");
        if (e) on = wires && (std::atoi(e) != 0);
        return on;
    };
    // ...and the same question for the polynomial: on the transported scalars, a blank `preconditioner`
    // is filled by a truncated Neumann series rather than by the bare diagonal (see the block that sets
    // ctl.polyDegKE). The notices below have to name what RUNS, so they ask the same question here
    // rather than printing "diagonal" over a solve that is not one.
    // THE BOUND THE SERIES STANDS ON. fvMatrix::relax does two things in this order (fvMatrix.C:105-113):
    // it forces D[c] = max(|D[c]|, sum|offdiag|), and THEN divides D by the relaxation factor. So on a
    // relaxed equation sum|offdiag|/|a_ii| <= alpha for every row, which bounds the series' convergence
    // ratio by the RELAXATION FACTOR and not by the mesh. Measured on squareBend, the row bound
    // max(sum|offdiag|/|a_ii|) of the epsilon system brae solves:
    //     24k   112k   307k   896k   1.75M   3.02M     alpha
    //   0.9000 0.9000 0.9000 0.9000  0.9000  0.9000     0.9    -- and rho(I - D^-1 A) 0.86 to 0.89
    // flat across two orders of magnitude in cell count, because it is alpha that pins it.
    //
    // With NO factor, or with a factor of 1 (fvMatrix::relax early-returns only on alpha <= 0, so
    // relax(1.0) still clamps but divides by nothing), the clamp alone gives a bound of exactly 1 and
    // the series has nothing to stand on: measured 0.9986 at 112k -- 0.9986^10 = 0.986, so degree 10
    // does nothing at all -- and 1.0010 at 3.02M, where it amplifies instead of preconditioning. A
    // truncated series is a polynomial in A, so it stays a bounded fixed linear operator either way and
    // cannot produce Inf; it just stops being a preconditioner, silently, which is the substitution this
    // project refuses. So an unrelaxed pair takes DILU instead, and the notice says which.
    //
    // The lookup mirrors readRelaxationFactors' own (`eqs ? eqs : rf`, OF's modern-then-legacy fallback),
    // and takes the LARGEST factor that will be applied -- a `".*Final" 1.0` corrector has no bound even
    // when the ordinary factor does.
    auto turbRelaxBound = [&](const std::string& f) -> scalar
    {
        const FoamDict* rf = fvSolution.subDict("relaxationFactors");
        const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
        const FoamDict* src = eqs ? eqs : rf;
        if (!src) return scalar(1);
        const scalar a0 = src->scalarOr(f, scalar(1));
        const scalar a1 = src->subDict(f + "Final") || src->found(f + "Final")
                        ? src->scalarOr(f + "Final", a0) : a0;
        return std::fmax(a0, a1);
    };
    auto polyHere = [&](const std::string& f, bool gs) -> int
    {
        if (gs || diluHere(f)) return 1;
        if (!(turbRelaxBound(f) < scalar(1))) return 1;    // no bound -> no series (see above)
        if (!(f == "k" || f == secondName || f == "nuTilda")) return 1;
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        int deg = (!s || s->wordOr("preconditioner", "").empty()) ? POLY_DEG_KE_DEFAULT : 1;
        if (const char* e = std::getenv("BRAE_POLY_KE")) deg = std::max(1, std::atoi(e));
        return deg;
    };
    // Is this field's blank `preconditioner` one brae has to fill? OpenFOAM's PBiCGStab and PCG both
    // require the entry (lduMatrix::preconditioner::New throws without one), so a field whose entry names
    // none is one whose solver is not a P-solver at all -- and that is exactly when brae is substituting.
    auto blankHere = [&](const std::string& f, bool gs) -> bool
    {
        if (gs || diluHere(f)) return false;
        if (!(f == "k" || f == secondName || f == "nuTilda")) return false;
        const FoamDict* sd = solvers ? solvers->subDict(f) : nullptr;
        return !sd || sd->wordOr("preconditioner", "").empty();
    };
    // The notices name what RUNS, by asking turbPreconFor -- the same function the policy below assigns
    // from. A notice that said `diagonal` over a Neumann-preconditioned solve, or stayed silent over a
    // DILU the caller never applied, is the defect this pair of readings exists to prevent.
    auto krylovPreconGs = [&](const std::string& f, bool gs) -> std::string
    {
        if (gs) return std::string("diagonal");            // unused: a smoothSolver has no preconditioner
        const FoamDict* sd = solvers ? solvers->subDict(f) : nullptr;
        const bool named = sd && !sd->wordOr("preconditioner", "").empty();
        const TurbPreconChoice c = turbPreconFor(solvers, fvSolution, f, gs);
        if (c.polyDeg > 1)
        {
            return "a degree-" + std::to_string(c.polyDeg) + " truncated Neumann series (the case names none)";
        }
        if (c.dilu && !named)
        {
            return "DILU (the case names none, and relaxes " + f + " by 1 or not at all, which leaves the "
                   "cheaper polynomial preconditioner without a convergence bound)";
        }
        return c.dilu ? std::string("DILU") : std::string("diagonal");
    };
    auto krylovPrecon = [&](const std::string& f) -> std::string
    {
        return diluHere(f) ? "DILU" : "diagonal";
    };

    // E2/E3 (dict_audit): SAY what brae runs when it is not what the case asked for.
    //
    // A substituted linear solver is not a wrong answer -- it solves the same linear system to the same
    // tolerance, so the converged SIMPLE result is unchanged. That is why this notices rather than
    // refuses, per the rule in brae_notice.cuh. What it DOES change is the iteration count and the cost,
    // and at a loose per-step relTol (0.01 on p is the SIMPLE norm) two solvers stop at different points,
    // so the intermediate fields differ. A user comparing brae's "Solving for p" line against OF's has a
    // right to know the solver is not the one they asked for.
    //
    // dict_audit found these unread: solvers/p/solver, solvers/p/smoother, solvers/(U|h|e)/preconditioner.
    // `gs` says brae took the smoothSolver path for this field. That honours the SELECTION and the
    // STOPPING RULE, and substitutes the SWEEP: brae's is multicolour where symGaussSeidelSmoother.C
    // walks cells in index order. It is the same algorithm under a permutation, but Gauss-Seidel is
    // order-dependent, so at the loose relTol a SIMPLE step asks for the two stop in different places --
    // measured on validation/T3A, OpenFOAM reached relTol 0.1 on Ux in ONE sweep (1.6186e-05 ->
    // 6.940e-07) where brae took SEVEN (-> 1.278e-06). This used to read "it HONOURED the request -- in
    // which case there is nothing to report", and that premise made every driver that includes this
    // reader silent about the substitution. Passed explicitly rather than inferred from the label: comparing the
    // dict's "smoothSolver" against a display string of "smoothSolver(symGaussSeidel)" made brae announce a
    // substitution on every field it was in fact running exactly as asked. The negative control in
    // tests/test_solver_notices.cu is what caught that, and it is the reason the test has one.
    auto noticeSolverChoice = [&](const std::string& f, const std::string& braeSolver,
                                  const std::string& braePrecon, bool gs)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        if (!s) return;
        const std::string want = s->wordOr("solver", "");
        const std::string smoo = s->wordOr("smoother", "");
        const std::string prec = s->wordOr("preconditioner", "");
        if (!want.empty() && !gs && want != braeSolver)
        {
            // The usual case: same system, same tolerance, so the CONVERGED answer is the same and only
            // the iteration count differs. An iteration CAP breaks that premise -- both solvers then stop
            // where the cap says, not where the tolerance says, and two different solvers stopped at the
            // same iteration count hold two different residuals. LES/NACA4412 is the live example:
            // `maxIter 10` on p, and at its impulsive first step OF's GAMG leaves at a residual of 4.26
            // against an initial 1 while brae's leaves at 2.55. Neither is converged; they cannot agree.
            const bool capped = s->found("maxIter") && s->scalarOr("maxIter", 1000.0) < 1000.0;
            noticeApproximated("solvers/" + f + " solver",
                               "case asks '" + want + "', brae runs " + braeSolver
                             + " preconditioned with " + braePrecon +
                               (capped
                                ? " AND this entry caps the solve at maxIter " + std::to_string((int)s->scalarOr("maxIter", 1000.0))
                                  + " -- with a cap the two solvers stop at DIFFERENT residuals, so the fields differ"
                                    " by however far the solve is from converged, not just in cost"
                                : " (same linear system and tolerance -- iteration count and cost differ)"));
        }
        // A smoother entry only means anything to brae when it actually took the smoothSolver path.
        if (!smoo.empty() && !gs)
            noticeIgnored("solvers/" + f + " smoother",
                          "'" + smoo + "' -- brae is not running a smoothSolver on this field");
        // ...and when it DID take that path there is nothing left to announce: device_sym_gauss_seidel.cuh
        // runs OpenFOAM's own sweep, level-scheduled, in whichever of the two variants the case named --
        // symGaussSeidelSmoother.C's up-then-down or GaussSeidelSmoother.C's up-only. tests/gs_ladder
        // holds both to OpenFOAM's own per-sweep residual. The `ignored` arm above still fires for a
        // smoother brae does not run at all, so the two cannot both be silent for a field.
        // Against what THIS DRIVER preconditions with, not against a fixed exemption list. The old test
        // was `prec != "diagonal" && prec != "none" && !diluWired`, which had two holes: the wired list
        // named the fields a DIFFERENT driver wires (queue item 27), and `none` is not `diagonal` --
        // noPreconditioner is wA = rA and diagonalPreconditioner is wA = rA/diag (their .C files), so a
        // case asking for no preconditioner at all was answered with Jacobi and told nothing.
        //
        // AND THE SUBSTITUTION IS NOT COST-ONLY, which is what the wording used to imply. Both
        // preconditioners reach the requested relTol; they stop in different places. On
        // turbulentFlatPlate:kEpsilon, over 60 consecutive k solves, OpenFOAM's DILU lands at a median
        // 0.0064 of the initial residual -- one iteration overshooting the case's relTol of 0.1 by more
        // than 10x -- while Jacobi stops at 0.0726. That gap left k and epsilon mutually inconsistent
        // every outer iteration and the case DIVERGED at iteration 171. A preconditioner substitution
        // can change whether a case runs at all.
        if (!prec.empty() && !gs && prec != braePrecon)
            noticeApproximated("solvers/" + f + " preconditioner",
                               "case asks '" + prec + "', brae preconditions with " + braePrecon + ". Both reach the"
                               " requested relTol but stop at DIFFERENT residuals, which can change stability, not"
                               " just cost");
    };

    // OF's selection, exactly: solver smoothSolver + a GaussSeidel-family smoother -> brae's symmetric
    // multicolor deviceSymGaussSeidel. Anything else (PBiCG[Stab]/GAMG/...) keeps BiCGStab.
    auto useSymGS = [&](const std::string& f)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        if (!s || s->wordOr("solver", "") != "smoothSolver") return false;
        const std::string sm = s->wordOr("smoother", "");
        return sm == "symGaussSeidel" || sm == "GaussSeidel";
    };
    // ...and WHICH of the two OF smoothers it named. They are different algorithms, not settings:
    // symGaussSeidelSmoother.C walks the cells up then back down, GaussSeidelSmoother.C walks them up
    // ONLY. Anything other than a bare `GaussSeidel` is the symmetric one (the field either did not take
    // this path at all, in which case the flag is unused, or it named symGaussSeidel).
    auto gsIsSymmetric = [&](const std::string& f)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return !(s && s->wordOr("smoother", "") == "GaussSeidel");
    };

    // fvSolution solvers/<field>/nSweeps (smoothSolver.C:78, default 1, NO clamp). Read for every field
    // that can take the smoothSolver path: OpenFOAM smooths nSweeps times BETWEEN residual evaluations
    // and counts sweeps, not evaluations (:205), so a case asking 2 and answered with 1 stops somewhere
    // else. A NEGATIVE nSweeps is OpenFOAM's fixed-count branch (:95-119: exactly -nSweeps sweeps, no
    // residual evaluated, nIterations = -nSweeps) and nSweeps 0 never advances its do-while (:202-209).
    // The first version clamped both to the default and said nothing; now the raw value goes to the one
    // path that runs the fixed-count branch (SolverRunsAs::fixedSweepsField) and every other path refuses
    // it by name rather than running a different solve under the case's own words.
    auto solverNSweeps = [&](const std::string& f, int dflt)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        if (!s || !s->found("nSweeps")) return dflt;
        // Only smoothSolver reads the key (smoothSolver.C:78 is its one reader in the tree); on a
        // PBiCGStab or GAMG entry the key is dead and OpenFOAM runs as if it were absent, so the
        // default is what runs there -- refusing it would abort a case OpenFOAM accepts.
        if (s->wordOr("solver", "") != "smoothSolver") return dflt;
        const int n = static_cast<int>(s->scalarOr("nSweeps", (scalar)dflt));
        if (n == 0)
        {
            throw std::runtime_error("solvers/" + f + " nSweeps 0: OpenFOAM's smoothSolver never advances "
                                     "nIterations with it and its do-while never terminates "
                                     "(smoothSolver.C:202-209); no solver path runs that. Write a positive nSweeps.");
        }
        if (n > 0 || runsAs.fixedSweepsField == f) return n;
        throw std::runtime_error("solvers/" + f + " nSweeps " + std::to_string(n) + ": a negative nSweeps is "
                                 "OpenFOAM's fixed-count branch (smoothSolver.C:95-119: exactly -nSweeps sweeps, "
                                 "no residual evaluated), which the solver path this driver runs on " + f +
                                 " does not have. Write a positive nSweeps for this field.");
    };
    // lduMatrix::solver::readControls (lduMatrixSolver.C:199): tolerance_ = lduMatrix::defaultTolerance,
    // 1e-6 (lduMatrix.C:45), for EVERY field. U and the turbulence pair defaulted to 1e-8 here, 100x
    // tighter than OpenFOAM on an entry that omits `tolerance` (item 79 review, both rounds).
    constexpr scalar kOpenFoamDefaultTolerance = 1e-6;
    ctl.tolP = solverTol("p", kOpenFoamDefaultTolerance);
    ctl.tolU = solverTol("U", kOpenFoamDefaultTolerance);
    ctl.relTolP = solverRelTol("p");
    ctl.relTolU = solverRelTol("U");
    // The `Final` variants, defaulting to the base entry when the case does not define one (see the
    // note in DeviceSimpleControls). Read unconditionally: a steady case simply never sets finalIter,
    // and reading them here is what stops the dict audit calling pFinal an unimplemented input.
    ctl.tolPFinal = solverTol("pFinal", ctl.tolP);
    ctl.tolUFinal = solverTol("UFinal", ctl.tolU);
    ctl.relTolPFinal = solvers && solvers->subDict("pFinal") ? solverRelTol("pFinal") : ctl.relTolP;
    ctl.relTolUFinal = solvers && solvers->subDict("UFinal") ? solverRelTol("UFinal") : ctl.relTolU;
    ctl.maxIterP = solverMaxIter("p", 1000);
    ctl.maxIterU = solverMaxIter("U", 1000);
    ctl.minIterP = solverMinIter("p", 0);
    // pcorr (CorrectPhi). The tutorials spell the key as the regex "pcorr.*"; FoamDict already does OF's
    // regex-keyword lookup, so this finds it either way. Defaults are OF's lduMatrix ones, not p's --
    // a case that asks for correctPhi without a pcorr entry gets a converged projection, not p's relTol.
    ctl.tolPcorr = solverTol("pcorr", kOpenFoamDefaultTolerance);
    ctl.relTolPcorr = solverRelTol("pcorr");
    ctl.maxIterPcorr = solverMaxIter("pcorr", 1000);
    ctl.minIterU = solverMinIter("U", 0);
    ctl.maxIterPFinal = solverMaxIter("pFinal", ctl.maxIterP);
    ctl.maxIterUFinal = solverMaxIter("UFinal", ctl.maxIterU);
    ctl.minIterPFinal = solverMinIter("pFinal", ctl.minIterP);
    ctl.minIterUFinal = solverMinIter("UFinal", ctl.minIterU);
    if (!heName.empty())
    {
        ctl.tolHe     = solverTol(heName, kOpenFoamDefaultTolerance);
        ctl.relTolHe  = solverRelTol(heName);
        ctl.maxIterHe = solverMaxIter(heName, 1000);
        ctl.minIterHe = solverMinIter(heName, 0);
        ctl.nSweepsHe = solverNSweeps(heName, 1);
        // The energy field takes OpenFOAM's own smoother only where the CALLER runs it; every other
        // driver keeps BiCGStab and must keep announcing the substitution.
        ctl.gsHe    = useSymGS(heName) && runsAs.smoothSolverOnEnergy;
        ctl.gsHeSym = gsIsSymmetric(heName);
        noticeSolverChoice(heName, "PBiCGStab", krylovPrecon(heName), ctl.gsHe);
    }
    ctl.gsU = useSymGS("U") && runsAs.smoothSolverOnMomentum;
    ctl.gsUSym = gsIsSymmetric("U");
    ctl.nSweepsU = solverNSweeps("U", 1);
    if (const char* gsuEnv = std::getenv("BRAE_GS_U"))
        ctl.gsU = (std::atoi(gsuEnv) != 0) && ctl.gsU;
    // DILU on the momentum equations, when the case asks for it and brae is on the BiCGStab path.
    {
        const FoamDict* su = solvers ? solvers->subDict("U") : nullptr;
        ctl.diluU = !ctl.gsU && su && su->wordOr("preconditioner", "") == "DILU";
        if (const char* e = std::getenv("BRAE_DILU"))   // attribution escape hatch, both directions
            ctl.diluU = (std::atoi(e) != 0) && !ctl.gsU;
    }
    // DILU on the energy solve, read from the energy field's own block. Same shape as diluU above, and
    // like it, only meaningful on the BiCGStab path.
    if (!heName.empty())
    {
        const FoamDict* sh = solvers ? solvers->subDict(heName) : nullptr;
        ctl.diluHe = sh && sh->wordOr("preconditioner", "") == "DILU";
    }
    // p is never the case's choice: this driver runs what SolverRunsAs says whatever the dict asks.
    noticeSolverChoice("p", runsAs.pSolver, runsAs.pPrecon, false);   // p never takes the smoothSolver path
    // ...unless the caller owns the momentum notice (SolverRunsAs::momentumNoticedByCaller): every line
    // this call could print describes PBiCGStab or OpenFOAM's sweep, and that caller runs neither.
    if (!runsAs.momentumNoticedByCaller)
        noticeSolverChoice("U", "PBiCGStab", krylovPrecon("U"), ctl.gsU);

    if (ctl.turbulent)
    {
        // brae solves the turbulence pair to ONE tolerance (the tighter of the two), so the Final pair
        // collapses the same way. Falling back to the base value per field keeps a case that defines
        // only one of them (kFinal but no epsilonFinal) from tightening the pair on the strength of it.
        if (ctl.sa)
        {
            ctl.tolKE = solverTol("nuTilda", kOpenFoamDefaultTolerance);
            ctl.relTolKE = solverRelTol("nuTilda");
            ctl.maxIterKE = solverMaxIter("nuTilda", 1000);
            ctl.minIterKE = solverMinIter("nuTilda", 0);
            ctl.tolKEFinal = solverTol("nuTildaFinal", ctl.tolKE);
            ctl.relTolKEFinal = solvers && solvers->subDict("nuTildaFinal") ? solverRelTol("nuTildaFinal") : ctl.relTolKE;
            ctl.gsK = useSymGS("nuTilda") && runsAs.smoothSolverOnTurbulence;
            ctl.gsEps = false;
            noticeSolverChoice("nuTilda", "PBiCGStab", krylovPreconGs("nuTilda", ctl.gsK), ctl.gsK);
            // THIS BRANCH NEVER SET THE PRECONDITIONER. ctl.diluKE was assigned only in the k/epsilon
            // branch below, so it stayed false here and a case naming `preconditioner DILU` on nuTilda --
            // which 37 of the 43 SpalartAllmaras tutorials in OpenFOAM do (4 name smoothSolver, 2 PBiCG,
            // none GAMG) -- had its entry read and then ignored, and the solve ran the DIAGONAL.
            //
            // It said nothing, and the reason it said nothing is the point: noticeSolverChoice compares
            // the case's `preconditioner` against what diluHere() answers, diluHere() wires "nuTilda" and
            // answered DILU, so prec == braePrecon and the notice stayed silent over a solve that was not
            // doing it. A capability the shared reader reports and this branch never applied -- the same
            // shape as item 58. Measured on validation/airFoil2D with its nuTilda entry rewritten to
            // PBiCGStab/DILU: brae printed `Jacobi-BiCGStab: Solving for nuTilda`.
            {
                const TurbPreconChoice ch = turbPreconFor(solvers, fvSolution, "nuTilda", ctl.gsK);
                ctl.diluKE = ch.dilu;
                ctl.polyDegKE = ch.polyDeg;
            }
        }
        else
        {
            ctl.tolKE = std::fmin(solverTol("k", kOpenFoamDefaultTolerance), solverTol(secondName, kOpenFoamDefaultTolerance));
            ctl.relTolKE = std::fmin(solverRelTol("k"), solverRelTol(secondName));
            // One cap for the pair, as one tolerance: the tighter maxIter and the larger minIter. A cap
            // decides where a solve STOPS, so two entries that disagree are announced rather than one
            // being taken silently. Neither was read before this; both sat at the struct defaults.
            ctl.maxIterKE = std::min(solverMaxIter("k", 1000), solverMaxIter(secondName, 1000));
            ctl.minIterKE = std::max(solverMinIter("k", 0), solverMinIter(secondName, 0));
            if (solverMaxIter("k", 1000) != solverMaxIter(secondName, 1000)
             || solverMinIter("k", 0) != solverMinIter(secondName, 0))
                noticeApproximated("solvers/k and solvers/" + secondName + " maxIter/minIter",
                                   "the pair is solved under ONE cap: the tighter maxIter and the larger"
                                   " minIter of the two entries");
            ctl.tolKEFinal = std::fmin(solverTol("kFinal", solverTol("k", kOpenFoamDefaultTolerance)),
                                       solverTol(secondName + "Final", solverTol(secondName, kOpenFoamDefaultTolerance)));
            ctl.relTolKEFinal = std::fmin(
                solvers && solvers->subDict("kFinal") ? solverRelTol("kFinal") : solverRelTol("k"),
                solvers && solvers->subDict(secondName + "Final") ? solverRelTol(secondName + "Final") : solverRelTol(secondName));
            ctl.gsK = useSymGS("k") && runsAs.smoothSolverOnTurbulence;
            ctl.gsEps = useSymGS(secondName) && runsAs.smoothSolverOnTurbulence;
            // ONE smoother variant for the transported pair, as nSweeps is: the model solves both
            // scalars through one call. A case that names symGaussSeidel on one and GaussSeidel on the
            // other is refused rather than run with whichever entry was read first -- running would
            // apply one field's smoother under the other's name.
            ctl.gsKESym = gsIsSymmetric("k");
            // ONE nSweeps for the pair, for the reason the smoother variant is one: k and epsilon are
            // solved through one model call. Two different counts would run one field's setting under
            // the other's name, so they are refused rather than resolved by order.
            ctl.nSweepsKE = solverNSweeps("k", 1);
            if ((ctl.gsK || ctl.gsEps) && solverNSweeps(secondName, 1) != ctl.nSweepsKE)
                throw std::runtime_error(
                    "system/fvSolution gives k and " + secondName + " different `nSweeps` on a "
                    "smoothSolver. OpenFOAM smooths nSweeps times between residual evaluations and "
                    "counts sweeps, so the two entries stop the solves in different places, and this "
                    "driver carries one count for the transported pair.");
            if ((ctl.gsK || ctl.gsEps) && gsIsSymmetric(secondName) != ctl.gsKESym)
                throw std::runtime_error(
                    "system/fvSolution names a `GaussSeidel` smoother on one of k / " + secondName
                    + " and `symGaussSeidel` on the other. Those are different OpenFOAM smoothers "
                      "(GaussSeidelSmoother.C sweeps ascending only; symGaussSeidelSmoother.C also "
                      "sweeps back), and this driver carries one smoother for the transported pair, so "
                      "running would apply one field's setting under the other's name.");
            noticeSolverChoice("k", "PBiCGStab", krylovPreconGs("k", ctl.gsK), ctl.gsK);
            noticeSolverChoice(secondName, "PBiCGStab", krylovPreconGs(secondName, ctl.gsEps), ctl.gsEps);
            // DILU on whichever of the pair runs BiCGStab. subDict is regex-aware (literal first, then
            // last wildcard match, OF semantics), so a case writing its solver block as
            // "(omega|epsilon|k)" -- which is how essentially every tutorial writes it -- resolves here
            // without a special case. Read AFTER gsK/gsEps, since a smoothSolver field has no
            // preconditioner to honour.
            {
                const FoamDict* sk = solvers ? solvers->subDict("k") : nullptr;
                const FoamDict* ss = solvers ? solvers->subDict(secondName) : nullptr;
                // diluHere is the ONE rule (it is what the notices printed above consulted). gsK/gsEps
                // subtract the fields running as smoothSolvers, which have no preconditioner to carry.
                const TurbPreconChoice kc = turbPreconFor(solvers, fvSolution, "k", ctl.gsK);
                const TurbPreconChoice sc = turbPreconFor(solvers, fvSolution, secondName, ctl.gsEps);
                // The pair is solved through ONE model call and carries one preconditioner, so the two
                // fields' answers are merged: DILU if either asks for it (it is the stronger operator),
                // and otherwise the larger degree.
                ctl.diluKE = kc.dilu || sc.dilu;
                ctl.polyDegKE = ctl.diluKE ? 1 : std::max(kc.polyDeg, sc.polyDeg);
            }
        }
    }

    const FoamDict* algo = fvSolution.subDict(algorithmDict);
    {
        const std::string cons = algo ? algo->wordOr("consistent", "no") : "no";
        ctl.consistent = (cons == "yes" || cons == "true" || cons == "on" || cons == "1");   // SIMPLEC
    }
    ctl.nNonOrth = algo ? algo->intOr("nNonOrthogonalCorrectors", 0) : 0;
    {
        // pimpleControl.C:53. When set, pFinal is reserved for the last pressure corrector of the LAST
        // outer iteration; by default every outer iteration's last corrector gets it.
        const std::string fl = algo ? algo->wordOr("finalOnLastPimpleIterOnly", "no") : "no";
        ctl.finalOnLastPimpleIterOnly = (fl == "yes" || fl == "true" || fl == "on" || fl == "1");
    }
    {
        const std::vector<scalar> bf = algo ? algo->scalarListOr("bodyForce", {}) : std::vector<scalar>{};
        if (bf.size() >= 3) ctl.bodyForce = vector{bf[0], bf[1], bf[2]};   // constant momentum source
    }

    // Performance knobs (no effect on the converged answer). Kept here so a driver cannot get the
    // correctness controls and miss these, which is how they drifted apart before.
    ctl.pcgCheckEvery = 4;   // batched PCG residual read; OF-validated identical to K=1
    if (const char* ce = std::getenv("BRAE_PCG_CHECK_EVERY"))
    {
        const int k = std::atoi(ce);
        if (k >= 1) ctl.pcgCheckEvery = k;
    }
    if (const char* be = std::getenv("BRAE_BICG_CHECK_EVERY"))
    {
        const int k = std::atoi(be);
        if (k >= 1) ctl.bicgCheckEvery = k;
    }
    if (const char* cs = std::getenv("BRAE_CORR_SCALING")) ctl.corrScaling = (std::atoi(cs) != 0);
    if (const char* ug = std::getenv("BRAE_USE_GRAPH")) ctl.useGraph = (std::atoi(ug) != 0);
}

// relaxationFactors -> ctl.relax{U,P,K,Eps}. Third copy of this in the tree when it was written, each a
// different subset -- the steady driver had all of it, the compressible one had no alpha<=0 guard, and the
// transient one had neither the guard, nor the legacy form, nor the right key for the second scalar (it
// reused k's factor for epsilon/omega, so `omega 0.4` was ignored and omega ran at k's factor).
//
// Call AFTER the turbulence model is read: it branches on ctl.sa/ctl.sst to pick the field names.
// OF looks a relaxation factor up with keyType::REGEX (solution.C:341,383), which is the ONLY reason the
// near-universal PIMPLE idiom
//     relaxationFactors { equations { U 0.8; ".*Final" 1; } }
// does anything: on the last outer corrector OF appends "Final" to the name (GeometricField::relax and
// fvMatrix::relax both go through psi.select(isFinalIteration())), so "UFinal" matches ".*Final" and the
// final corrector runs UNRELAXED. Match the name literally first, then by regex, exactly as OF does.
// pOnly reproduces the filter in OF's legacy branch (solution.C:82-100): when relaxationFactors is the
// FLAT form, only keys beginning `p` or `rho` become FIELD relaxation, while the whole dict becomes
// EQUATION relaxation. Without the filter a flat `{ ".*" 0.7; }` would field-relax pFinal, which OF
// does not do -- its fieldRelaxDict_ never receives that key, so relax() is skipped entirely.
inline bool relaxLookup(const FoamDict* d, const std::string& name, scalar& out, bool pOnly = false)
{
    if (!d) return false;
    auto eligible = [pOnly](const std::string& k)
    { return !pOnly || k.rfind("p", 0) == 0 || k.rfind("rho", 0) == 0; };
    // A LITERAL probe, deliberately not d->found(): FoamDict's own lookup is regex-aware, so `found("p")`
    // is satisfied by a `".*"` key and the eligibility filter below would never get a say.
    bool literal = false;
    for (const auto& lv : d->leaves) if (lv.first == name) { literal = true; break; }
    if (literal && eligible(name)) { out = d->scalarOr(name, scalar(1)); return true; }
    bool hit = false;
    for (const auto& lv : d->leaves)
    {
        const std::string& key = lv.first;
        if (key.find_first_of("()|*?[].^$") == std::string::npos) continue;   // plain word, already tried
        if (!eligible(key)) continue;
        try
        {
            if (std::regex_match(name, compileFoamRegex(key))) { out = d->scalarOr(key, scalar(1)); hit = true; }
        }
        catch (const std::regex_error&) { /* not a usable regex -> not a match, as OF treats it */ }
    }
    return hit;
}

inline void readRelaxationFactors(const FoamDict& fvSolution, DeviceSimpleControls& ctl)
{
    const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
    const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
    // B1: the pressure EQUATION relaxation, distinct from the pressure FIELD relaxation in fields{}.
    // Only the transonic branch relaxes the pEqn, and only when the entry exists (OF fvMatrix::relax()).
    if (eqs && eqs->found("p")) { ctl.hasRelaxPEqn = true; ctl.relaxPEqn = eqs->scalarOr("p", 1.0); }
    const FoamDict* fld = rf ? rf->subDict("fields") : nullptr;
    // OF accepts BOTH the modern nested {equations{} fields{}} and the legacy FLAT {p ..; U ..;} form. Fall
    // back to the flat keys when a sub-dict is absent, so a legacy case isn't silently left un-relaxed
    // (all factors 1.0 -> the steady SIMPLE loop typically diverges).
    const FoamDict* eqSrc  = eqs ? eqs : rf;
    const FoamDict* fldSrc = fld ? fld : rf;

    const char* kName    = ctl.sa ? "nuTilda" : "k";              // SA: relaxK carries the nuTilda relax
    const std::string sName = ctl.sst ? "omega" : "epsilon";

    ctl.relaxU   = eqSrc  ? eqSrc->scalarOr("U", 1.0) : 1.0;
    ctl.relaxK   = eqSrc  ? eqSrc->scalarOr(kName, 1.0) : 1.0;
    ctl.relaxEps = eqSrc  ? eqSrc->scalarOr(sName, 1.0) : 1.0;
    // p goes through the filtered lookup too, not scalarOr: FoamDict's lookup is regex-aware, so on the
    // LEGACY flat form a catch-all `".*" 0.7;` would otherwise field-relax the pressure. OF's legacy
    // branch never copies that key into fieldRelaxDict_, so it relaxes the equations only.
    ctl.relaxP   = 1.0;
    relaxLookup(fldSrc, "p", ctl.relaxP, /*pOnly*/!fld);
    // ...and the FINAL-corrector factors, which brae had no notion of: it applied the ordinary factor on
    // every outer corrector including the last. OF does not, and in PIMPLE that is not a matter of
    // convergence rate -- the final corrector is what makes the step satisfy momentum and continuity
    // together, so relaxing it leaves a residue that the next step inherits. Measured on LES/vortexShed
    // (`relaxationFactors { nuTilda 0.8; U 0.8; p 0.8; ".*Final" 1.0; }`): the outer loop's initial
    // pressure residual GREW corrector by corrector (0.208 -> 0.256 -> 0.311 -> 0.354) instead of
    // falling, contLocal ran 1e-7 -> 40 over twenty steps, and |U| reached 1.29e+06 against OpenFOAM's
    // 0.0435. Absent from the dict -> no Final entry -> fall back to the ordinary factor, which is also
    // what OF does (the lookup simply misses and relax() is skipped for that name).
    // NO MATCH MEANS NO RELAXATION, which is not the same as "reuse the ordinary factor". OF's relax()
    // is guarded -- `if (relaxField(name)) relax(factor)` in GeometricField::relax, and the identical
    // shape in fvMatrix::relax -- so on the final corrector, where the name carries the "Final" suffix,
    // an unmatched name means relax() is never called and the factor is effectively 1. That is why a
    // PIMPLE case with a bare `equations { U 0.7; }` still ends each step with an unrelaxed corrector.
    // Steady SIMPLE is untouched: finalIter is only ever true inside the PIMPLE outer loop.
    ctl.relaxUFinal = 1.0;   relaxLookup(eqSrc,  "UFinal",                     ctl.relaxUFinal);
    ctl.relaxKFinal = 1.0;   relaxLookup(eqSrc,  kName + std::string("Final"), ctl.relaxKFinal);
    ctl.relaxEpsFinal = 1.0; relaxLookup(eqSrc,  sName + "Final",              ctl.relaxEpsFinal);
    ctl.relaxPFinal = 1.0;   relaxLookup(fldSrc, "pFinal", ctl.relaxPFinal, /*pOnly*/!fld);

    // A relaxation factor <= 0 divides by zero in the diagonal-relaxation kernel (Inf diag -> NaN). OF's
    // fvMatrix::relax skips relaxation for alpha <= 0; match that (treat as 1.0 = no under-relaxation) + warn.
    auto fixRelax = [](scalar& a, const char* nm)
    {
        if (a <= 0.0)
        {
            std::fprintf(stderr,
                "brae WARNING: relaxationFactors %s = %g <= 0; using 1.0 (no under-relaxation)\n", nm, (double)a);
            a = 1.0;
        }
    };
    fixRelax(ctl.relaxU, "U");
    fixRelax(ctl.relaxK, kName);
    fixRelax(ctl.relaxEps, sName.c_str());
    fixRelax(ctl.relaxP, "p");
}

// The energy equation's linear solver, which OF names "h" for sensibleEnthalpy and "e" for
// sensibleInternalEnergy. Was hardcoded tol=1e-10, relTol=0, BiCGStab regardless of the case.
struct EnergySolverControls
{
    scalar tol = 1e-10;
    scalar relTol = 0.0;
    bool   useGS = false;
};

inline EnergySolverControls readEnergySolverControls(
    const FoamDict& fvSolution,
    bool internalEnergy)
{
    EnergySolverControls e;
    const FoamDict* solvers = fvSolution.subDict("solvers");
    if (!solvers) return e;
    // Try the case's own energy-field name first, then the other, so a case listing only one is honoured.
    const char* primary = internalEnergy ? "e" : "h";
    const char* fallback = internalEnergy ? "h" : "e";
    const FoamDict* s = solvers->subDict(primary);
    if (!s) s = solvers->subDict(fallback);
    if (!s) return e;
    e.tol = s->scalarOr("tolerance", e.tol);
    e.relTol = s->scalarOr("relTol", e.relTol);
    if (s->wordOr("solver", "") == "smoothSolver")
    {
        const std::string sm = s->wordOr("smoother", "");
        e.useGS = (sm == "symGaussSeidel" || sm == "GaussSeidel");
    }
    // E2/E3: same reporting as the momentum/turbulence fields. The energy entry is usually a REGEX in the
    // stock tutorials -- `"(U|h|e)" { solver smoothSolver; ... preconditioner DILU; }` -- which is exactly
    // why dict_audit flagged `solvers/(U|h|e)/preconditioner`: brae read the tolerances out of that entry
    // and never looked at the preconditioner in it.
    {
        const std::string want = s->wordOr("solver", "");
        const std::string smoo = s->wordOr("smoother", "");
        const std::string prec = s->wordOr("preconditioner", "");
        const std::string field = std::string("solvers/") + primary;
        // The same OpenFOAM vocabulary as readLinearSolverControls above: this reader's callers (the
        // legacy compressible drivers) run a diagonal-preconditioned BiCGStab on the energy field, which
        // IS `PBiCGStab` + `preconditioner diagonal`. The OF-mirror does not call this function -- it
        // reads the energy entry through readLinearSolverControls, where SolverRunsAs::diluOnEnergy says
        // that it wires DILU there.
        if (!want.empty() && !e.useGS && want != "PBiCGStab")
            noticeApproximated(field + " solver",
                               "case asks '" + want + "', brae runs PBiCGStab preconditioned with diagonal"
                               " (same linear system and tolerance -- iteration count and cost differ)");
        if (!smoo.empty() && !e.useGS)
            noticeIgnored(field + " smoother", "'" + smoo + "' -- brae is not running a smoothSolver on this field");
        if (!prec.empty() && !e.useGS && prec != "diagonal")
            noticeApproximated(field + " preconditioner",
                               "case asks '" + prec + "', brae preconditions with diagonal (Jacobi)");
    }
    return e;
}

}   // namespace brae
