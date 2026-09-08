#!/usr/bin/env bash
# The AMG-preconditioned BiCGStab on the TRANSONIC pressure (the default since 2026-09-08) against real
# OpenFOAM running the same pair.
#
# The transonic pressure matrix is asymmetric -- fvm::div(phid, p) makes lower = -w*phi and upper =
# lower + phi, so upper != lower at every face with flow through it -- so the mirror's CUDA arm solves it
# with BiCGStab. It used to precondition that BiCGStab with the diagonal. OpenFOAM registers
# GAMGPreconditioner in the ASYMMETRIC constructor table as well as the symmetric one
# (GAMGPreconditioner.C:37-42), so `p { solver PBiCGStab; preconditioner GAMG; }` is a legal OpenFOAM
# setting on this matrix: it is the like-for-like pair, not a substituted solver class, and it is what
# this gate stages in BOTH codes.
#
# WHY IT EXISTS. Measured 2026-09-08 on the 305,760-cell squareBend, GB10 + OpenFOAM on 20 cores, with
# OpenFOAM's OWN PBiCGStab on the same transonic p: diagonal preconditioner 380.8 solver iterations mean
# and 278.1 ms per outer iteration, DILU 135.3 and 143.9, GAMG 3.8 (max 6) and 33.8 -- a hundredfold cut
# in iterations for a roughly fourfold heavier apply. brae's diagonal arm on the same case ran 2234
# BiCGStab iterations over 20 outer iterations, 65.9 ms of a 69.5 ms pressure phase. The risk this has to
# beat is item 77 (DILU on this equation): 3x fewer iterations for 10x more apply cost, a measured LOSS,
# made opt-in.
#
# AND WHY IT NEEDS A GATE AT ALL. brae's V-cycle is not OpenFOAM's GAMG (its own agglomeration, a
# weighted-Jacobi smoother, ONE V-cycle where GAMGPreconditioner defaults to nVcycles 2, an FP32 cycle by
# default), so a case naming that pair is NOT matched to the bit -- and naming it makes the shared
# reader's `solvers/p` notice go silent, which would be a lie. The driver prints the difference itself.
# This gate holds the wiring AND the words: EXACT where the linear solve is converged, DIFFERENT where it
# is not, and SAID either way.
#
#   FIXTURE     validation/sbMatched (112,000 cells, `transonic yes`, `consistent yes`, kEpsilon on a
#               non-orthogonal mesh -- the tree's only transonic fixture; the SKIP below re-checks that
#               rather than trusting this line). The p entry is REWRITTEN in both codes to
#                   p { solver PBiCGStab;
#                       preconditioner { preconditioner GAMG; smoother GaussSeidel; nPreSweeps 0;
#                                        nPostSweeps 2; nCellsInCoarsestLevel 64;
#                                        agglomerator faceAreaPair; mergeLevels 1;
#                                        tolerance 1e-05; relTol 0; }
#                       tolerance 1e-12; relTol R; maxIter M; }
#               The SUB-DICTIONARY form is required: OpenFOAM reads the preconditioner's controls from
#               that dict (lduMatrixPreconditioner.C:74-88 passes dictionary::null for a primitive
#               entry), and a bare `preconditioner GAMG;` aborts with
#               `FOAM FATAL IO ERROR ... Entry 'smoother' not found in dictionary` -- run on this very
#               fixture, 2026-09-08. With the sub-dictionary OpenFOAM runs it and logs the pair as
#               `GAMGPBiCGStab:  Solving for p` (24 iterations to 1.7e-13 on the first outer iteration,
#               1.27 s for the whole iteration at 112k), which is what the premise check below matches.
#               "(U|e|k|epsilon)" stays the fixture's own PBiCGStab/DILU, pinned 1e-12 / relTol 0 /
#               maxIter 2000, and residualControl is blanked so both codes run exactly N iterations.
#               Every brae arm runs BRAE_U_SOLVER=ofOrder -- the case's own momentum solver -- so the p
#               preconditioner is the ONLY difference left between the two codes.
#   ARM EXACT   brae's DEFAULT (no BRAE_P_SOLVER set) against OpenFOAM, relTol 0 on p in both: every
#               linear solve converges, so U, p, T, k and epsilon at N must agree within BOUND (rel L2).
#   ARM REF     the same case under BRAE_P_SOLVER=diagonal (the opt-out): must ALSO meet BOUND, so the
#               bound is one this fixture can meet and the EXACT arm's pass is not a loose bound.
#   ARM RAN     the CONTROL that the AMG actually preconditioned anything: the EXACT arm's pIters (the
#               p solver iterations the summary line carries) must be strictly FEWER than the REF arm's
#               on every iteration. Without it, a build that silently ignored the hierarchy would pass
#               EXACT by being identical to REF.
#   CONTROL     relTol 0.1 on p in BOTH codes: the two preconditioners then stop at different iterates,
#               so the fields at N must differ by MORE than BOUND. This is what proves the comparison
#               can see a preconditioner difference at all.
#   FAIL-PROOF  BRAE_P_SOLVER=diagonal with p maxIter 1 in brae ONLY (OpenFOAM keeps 2000): one
#               diagonal-preconditioned BiCGStab iteration per outer iteration against a converged solve
#               must MISS BOUND.
#   SAID        five brae-only 1-iteration arms, one per shape of p entry, asserting that exactly one
#               truthful set of lines prints:
#                 (a) `solver GAMG`            -> the reader announces a different SOLVER, plus the
#                                                 driver's V-cycle notice; no preconditioner line.
#                 (b) PBiCGStab + GAMG         -> the reader is SILENT on solvers/p (brae runs the pair
#                                                 the case named); the driver's notice is the only line,
#                                                 and it is what stops that silence being a lie.
#                 (c) PBiCGStab + DILU         -> the reader announces `case asks 'DILU', brae
#                                                 preconditions with GAMG`, plus the driver's notice.
#                 (d) PBiCGStab + diagonal     -> the reader announces `case asks 'diagonal', brae
#                                                 preconditions with GAMG`, plus the driver's notice.
#                 (e) (d) under BRAE_P_SOLVER=diagonal -> the reader is silent (brae runs what the case
#                                                 asked), the driver's V-cycle notice is ABSENT, and the
#                                                 start-up line names the diagonal. (c) and (d) are also
#                                                 what shows the ABSENCE checks in (b) and (e) can match.
#   BOUND       1e-8 (rel L2, every field). PROVENANCE: rho_sbmatched_transient_vs_openfoam measures this
#               same fixture with the same pinning (1e-12 / relTol 0) and the same momentum solver over
#               t=1..3 and holds its device arm to k 6e-10, epsilon 3e-9, U 2e-10, p 6e-10, T 8e-11; this
#               gate runs 5 iterations rather than 3, so the bound is ~3x the worst of those. The p solve
#               is NOT bit-reproducible run to run on this path (four runs of the sibling gate read
#               1.8e-11, 8.8e-12, 3.6e-12, 4.6e-12 on p), which is the other reason for the margin.
#   MEASURED    OpenFOAM side, 2026-09-08, this staging on this fixture, 3.6 s for the 5 iterations:
#               its GAMG-preconditioned PBiCGStab took 24 / 18 / 18 / 18 / 18 iterations on p to final
#               residuals 1.7e-13 / 7.9e-13 / 2.7e-13 / 1.3e-13 / 8.4e-13 -- converged, nowhere near the
#               2000 cap, which is the EXACT arm's premise. OpenFOAM at relTol 0.1 (1-5 iterations per
#               solve) against OpenFOAM converged, rel L2 at t=5: U 6.60e-02, p 2.29e-02, T 4.72e-03,
#               k 1.80e-01, epsilon 3.25e-01 -- six to seven orders above BOUND, which is what the
#               CONTROL has to see. Non-vacuity, 0/ against OpenFOAM t=5: U 1.00, p 1.20e-01, T 1.45e-02,
#               k 1.00, epsilon 1.00.
#               brae side: NOT YET RECORDED. The engine half of this change (the AMGData argument on
#               deviceJacobiBiCGStab and the asymmetric coarsest solve) was landing in parallel, so the
#               brae arms were written against the API and have not been executed. Record the
#               EXACT / REF / RAN / CONTROL / FAIL-PROOF numbers here on the first green run and TIGHTEN
#               BOUND to them: it is inherited from the sibling gate above, not measured on this one.
# MEASURED (first green run, 2026-09-08, after the coarsest solve was made to CONVERGE -- see below):
#   EXACT AMG vs OpenFOAM   U 2.390e-12  p 4.338e-12  T 2.517e-12  k 2.886e-12  epsilon 3.700e-12
#   REF   diagonal vs OF    U 3.946e-12  p 4.796e-12  T 2.525e-12  k 9.614e-12  epsilon 9.902e-12
#   CONTROL relTol 0.1      6.1e-02 .. 3.5e-01 (more than the bound, as it must be)
#   FAIL-PROOF capped       inf / 9.3e-01
#   p iterations per outer solve: AMG 49 50 43 45 44, diagonal 612 498 503 474 491; OpenFOAM's own
#   PBiCGStab+GAMG took at most 24. The bound 1e-8 is four decades above what the EXACT arm reads and
#   could be tightened to 1e-10; left where it is until a second mesh has been through this gate.
#
# WHAT THIS GATE CAUGHT ON ITS FIRST RUN, and why the coarsest level is solved to a tolerance: with the
# coarsest level solved by a FIXED 16 inner iterations the V-cycle is not a fixed LINEAR operator --
# what an inner Krylov method returns depends on its right-hand side -- and the outer BiCGStab broke on
# it: 187, then 1000 (its cap, unconverged), then 89 iterations on three consecutive outer solves, and
# this arm read U 4.670e-01 against OpenFOAM. Converging the coarsest solve gives the numbers above.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="${BRAE_BIN:-$BUILDDIR/brae_rhoSimpleFoam}"
SRC="${1:-$ROOT/validation/sbMatched}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-5}
BOUND=${BOUND:-1e-8}
[ -x "$BIN" ]  || { echo "SKIP: no brae_rhoSimpleFoam at $BIN"; exit 77; }
[ -d "$SRC/constant/polyMesh" ] || { echo "SKIP: fixture $SRC ships no mesh"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
# The premise of the whole gate: this fixture's pressure matrix is the ASYMMETRIC one. A subsonic
# fixture would run the AMG-preconditioned CG on both arms and every comparison here would be vacuous.
grep -qE "transonic[[:space:]]+yes" "$SRC/system/fvSolution" \
    || { echo "SKIP: $SRC is not transonic -- this gate needs the asymmetric pressure matrix"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-92s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage <dir> <pmode> <relTolP> <maxIterP> <iterations>
#   pmode  gamgprecon | stock | gamgsolver | diagprecon  -- the shape of the p entry (see SAID)
stage()
{
    local d=$1 pmode=$2 rel=$3 mx=$4 n=$5
    rm -rf "$d"
    cp -r "$SRC" "$d"
    rm -rf "$d"/[1-9]* "$d"/0 "$d"/processor* "$d"/log.* 2> /dev/null
    cp -r "$d/0.orig" "$d/0"
    python3 - "$d" "$pmode" "$rel" "$mx" "$n" <<'PY'
import re, sys
d, pmode, rel, mx, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
c = d + '/system/controlDict'
s = open(c).read()
for k, v in [('startFrom', 'startTime'), ('startTime', '0'), ('stopAt', 'endTime'), ('deltaT', '1'),
             ('endTime', n), ('writeControl', 'timeStep'), ('writeInterval', n),
             ('writeFormat', 'ascii'), ('writePrecision', '15')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)

# The p entry, written out in full rather than patched: the GAMG preconditioner needs a sub-dictionary
# (OpenFOAM hands a primitive `preconditioner GAMG;` dictionary::null and then aborts on the missing
# `smoother`), and a blanket tolerance/relTol regex over the file would rewrite that sub-dictionary's
# own tolerance as well as the solve's.
P = {
 'gamgprecon': """    p
    {
        solver          PBiCGStab;
        preconditioner
        {
            preconditioner  GAMG;
            smoother        GaussSeidel;
            nPreSweeps      0;
            nPostSweeps     2;
            nCellsInCoarsestLevel 64;
            agglomerator    faceAreaPair;
            mergeLevels     1;
            tolerance       1e-05;
            relTol          0;
        }
        tolerance       1e-12;
        relTol          REL;
        maxIter         MAX;
    }""",
 'stock':      '    p { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol REL; maxIter MAX; }',
 'diagprecon': '    p { solver PBiCGStab; preconditioner diagonal; tolerance 1e-12; relTol REL; maxIter MAX; }',
 'gamgsolver': ('    p { solver GAMG; smoother GaussSeidel; nCellsInCoarsestLevel 64; '
                'agglomerator faceAreaPair; mergeLevels 1; tolerance 1e-12; relTol REL; maxIter MAX; }'),
}
block = P[pmode].replace('REL', rel).replace('MAX', mx)
f = d + '/system/fvSolution'
s = open(f).read()
s, k = re.subn(r'^[ \t]*p[ \t]*\{[^{}]*\}', block, s, flags=re.M)
assert k == 1, 'expected exactly one flat p entry in the fixture, replaced %d' % k
# The other equations pinned, so the pressure preconditioner is the only difference between the codes.
s, k = re.subn(r'"\(U\|e\|k\|epsilon\)"\s*\{[^}]*\}',
               '"(U|e|k|epsilon)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; '
               'relTol 0; maxIter 2000; }', s)
assert k == 1, 'the fixture no longer carries one "(U|e|k|epsilon)" block (found %d)' % k
# ...and no early stop, or the two codes run different numbers of iterations.
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PY
}

# run <dir> <of|brae> [VAR=value ...]
run()
{
    local d=$1 sel=$2
    shift 2
    if [ "$sel" = of ]; then
        ( set +u; source "$OFBASHRC" > /dev/null 2>&1; cd "$d" && rhoSimpleFoam > run.log 2>&1 )
    else
        # BRAE_U_SOLVER=ofOrder: the case's own momentum solver on both sides, so the ONLY difference
        # left between brae and OpenFOAM here is the pressure preconditioner this gate is about.
        ( cd "$d" && env BRAE_RHOSIMPLEFOAM_MIRROR=cuda BRAE_U_SOLVER=ofOrder "$@" \
              "$BIN" -case "$d" > run.log 2>&1 )
    fi
}

# OpenFOAM: the converged solve and the loose one, both with the GAMG-preconditioned PBiCGStab on p.
stage "$W/of_exact" gamgprecon 0   2000 "$N"
run   "$W/of_exact" of || { tail -20 "$W/of_exact/run.log"; echo "FAIL: OpenFOAM (exact) did not run"; exit 1; }
stage "$W/of_ctrl"  gamgprecon 0.1 2000 "$N"
run   "$W/of_ctrl"  of || { tail -20 "$W/of_ctrl/run.log"; echo "FAIL: OpenFOAM (control) did not run"; exit 1; }

# brae: the default (AMG), the opt-out (diagonal), the loose control and the fail-proof.
stage "$W/b_amg"  gamgprecon 0   2000 "$N"
run   "$W/b_amg"  brae || { tail -20 "$W/b_amg/run.log";  say "EXACT       the AMG run finished" FAIL; }
stage "$W/b_diag" gamgprecon 0   2000 "$N"
run   "$W/b_diag" brae BRAE_P_SOLVER=diagonal || { tail -20 "$W/b_diag/run.log"; say "REF         the diagonal run finished" FAIL; }
# The V-cycle's SMOOTHER on this path is the two-stage Gauss-Seidel, chosen across three mesh sizes
# (the numbers are at useTSGSAsym() in device_amg_detail.cuh). BRAE_AMG_TSGS=0 restores the weighted
# Jacobi: a smoother changes the cost and the iterate under a loose relTol, never the converged answer,
# so this arm must meet the same bound as the default one.
stage "$W/b_jac"  gamgprecon 0   2000 "$N"
run   "$W/b_jac"  brae BRAE_AMG_TSGS=0 || { tail -20 "$W/b_jac/run.log"; say "SMOOTHER    the weighted-Jacobi run finished" FAIL; }
stage "$W/b_ctrl" gamgprecon 0.1 2000 "$N"
run   "$W/b_ctrl" brae || { tail -20 "$W/b_ctrl/run.log"; say "CONTROL     the loose AMG run finished" FAIL; }
stage "$W/b_fp"   gamgprecon 0   1    "$N"
# A p solve capped at ONE diagonal-preconditioned iteration may not merely be inaccurate -- it can leave
# the fixed-point iteration unstable and end the run. That is a miss of the bound too, so the launch is
# not asserted here: the field comparison below reads missing (or non-finite) fields as `inf`, which is
# what the FAIL-PROOF arm needs to see. A build that is broken for every arm still fails EXACT and REF.
run   "$W/b_fp"   brae BRAE_P_SOLVER=diagonal \
    || { tail -5 "$W/b_fp/run.log"; echo "  (the capped-p run did not finish -- itself a miss of the bound)"; }

# The SAID arms: one iteration each, brae only, one per shape of p entry.
stage "$W/n_gamgsolver" gamgsolver 0 2000 1
run   "$W/n_gamgsolver" brae || { tail -20 "$W/n_gamgsolver/run.log"; say "SAID (a)    the 'solver GAMG' run finished" FAIL; }
stage "$W/n_pair"       gamgprecon 0 2000 1
run   "$W/n_pair"       brae || { tail -20 "$W/n_pair/run.log"; say "SAID (b)    the PBiCGStab+GAMG run finished" FAIL; }
stage "$W/n_dilu"       stock      0 2000 1
run   "$W/n_dilu"       brae || { tail -20 "$W/n_dilu/run.log"; say "SAID (c)    the PBiCGStab+DILU run finished" FAIL; }
stage "$W/n_diag"       diagprecon 0 2000 1
run   "$W/n_diag"       brae || { tail -20 "$W/n_diag/run.log"; say "SAID (d)    the PBiCGStab+diagonal run finished" FAIL; }
stage "$W/n_diagsel"    diagprecon 0 2000 1
run   "$W/n_diagsel"    brae BRAE_P_SOLVER=diagonal || { tail -20 "$W/n_diagsel/run.log"; say "SAID (e)    the opt-out run finished" FAIL; }

has() { grep -qF -- "$2" "$W/$1/run.log"; }

NOTICE="brae NOTICE [approximated] transonic p preconditioner: brae runs the case's PBiCGStab preconditioned with ITS OWN AMG V-cycle"
LINE_AMG="transonic pressure: PBiCGStab preconditioned with brae's AMG V-cycle"
LINE_DIAG="transonic pressure: PBiCGStab preconditioned with the diagonal (BRAE_P_SOLVER=diagonal)"

has b_amg "$NOTICE" \
    && say "SAID  EXACT carries the driver's V-cycle notice (brae's V-cycle is not OpenFOAM's GAMG)" ok \
    || say "SAID  EXACT carries the driver's V-cycle notice (brae's V-cycle is not OpenFOAM's GAMG)" FAIL
has b_amg "$LINE_AMG" \
    && say "SAID  EXACT's start-up line names the AMG V-cycle as the pressure preconditioner" ok \
    || say "SAID  EXACT's start-up line names the AMG V-cycle as the pressure preconditioner" FAIL
has b_diag "$LINE_DIAG" \
    && say "SAID  REF's start-up line names the diagonal -- the opt-out says what IT runs" ok \
    || say "SAID  REF's start-up line names the diagonal -- the opt-out says what IT runs" FAIL
has b_diag "$NOTICE" \
    && say "SAID  REF carries NO V-cycle notice (no V-cycle runs there)" FAIL \
    || say "SAID  REF carries NO V-cycle notice (no V-cycle runs there)" ok
# (a) a case asking for GAMG as the SOLVER: a different algorithm, announced by the shared reader, and
#     no preconditioner line -- that entry names no preconditioner.
has n_gamgsolver "solvers/p solver: case asks 'GAMG', brae runs PBiCGStab preconditioned with GAMG" \
    && say "SAID (a)  'solver GAMG' is announced as a different solver" ok \
    || say "SAID (a)  'solver GAMG' is announced as a different solver" FAIL
has n_gamgsolver "$NOTICE" \
    && say "SAID (a)  ...and the driver's V-cycle notice says what that GAMG preconditioner actually is" ok \
    || say "SAID (a)  ...and the driver's V-cycle notice says what that GAMG preconditioner actually is" FAIL
has n_gamgsolver "solvers/p preconditioner" \
    && say "SAID (a)  no preconditioner notice on an entry that names no preconditioner" FAIL \
    || say "SAID (a)  no preconditioner notice on an entry that names no preconditioner" ok
# (b) the pair brae runs: the reader is silent, and the driver's notice is the one truthful line.
if has n_pair "solvers/p solver" || has n_pair "solvers/p preconditioner"; then
    say "SAID (b)  the reader is silent on solvers/p when the case names the pair brae runs" FAIL
else
    say "SAID (b)  the reader is silent on solvers/p when the case names the pair brae runs" ok
fi
has n_pair "$NOTICE" \
    && say "SAID (b)  ...and the driver's notice is what stops that silence being a lie" ok \
    || say "SAID (b)  ...and the driver's notice is what stops that silence being a lie" FAIL
# (c) and (d): the reader announces the substitution -- and these are what show (b)'s and (e)'s absence
#     checks are able to match at all.
has n_dilu "solvers/p preconditioner: case asks 'DILU', brae preconditions with GAMG" \
    && say "SAID (c)  a DILU entry is announced against the GAMG that runs (the absences can match)" ok \
    || say "SAID (c)  a DILU entry is announced against the GAMG that runs (the absences can match)" FAIL
has n_diag "solvers/p preconditioner: case asks 'diagonal', brae preconditions with GAMG" \
    && say "SAID (d)  a diagonal entry is announced against the GAMG that runs" ok \
    || say "SAID (d)  a diagonal entry is announced against the GAMG that runs" FAIL
has n_diag "$NOTICE" \
    && say "SAID (d)  ...with the driver's V-cycle notice beside it" ok \
    || say "SAID (d)  ...with the driver's V-cycle notice beside it" FAIL
# (e) the same entry under the opt-out: brae runs what the case asked, so the reader is silent and the
#     V-cycle notice is gone -- the start-up line is what carries the truth.
if has n_diagsel "solvers/p preconditioner" || has n_diagsel "$NOTICE"; then
    say "SAID (e)  under BRAE_P_SOLVER=diagonal a diagonal entry draws no notice at all" FAIL
else
    say "SAID (e)  under BRAE_P_SOLVER=diagonal a diagonal entry draws no notice at all" ok
fi
has n_diagsel "$LINE_DIAG" \
    && say "SAID (e)  ...and the start-up line still names the diagonal" ok \
    || say "SAID (e)  ...and the start-up line still names the diagonal" FAIL

# The OpenFOAM side of the premise: its GAMG-preconditioned PBiCGStab drove EVERY p solve to the
# tolerance, or the EXACT arm compares two unconverged iterates and the bound means nothing.
python3 - "$W/of_exact/run.log" <<'PY' && say "EXACT  OpenFOAM's PBiCGStab+GAMG converged every p solve (final residual < 1e-12, under the cap)" ok \
                                || say "EXACT  OpenFOAM's PBiCGStab+GAMG converged every p solve (final residual < 1e-12, under the cap)" FAIL
import re, sys
s = open(sys.argv[1]).read()
# `GAMGPBiCGStab` is OpenFOAM's own log name for the pair -- the preconditioner's typeName prefixed to
# the solver's -- so matching it proves OpenFOAM ran the GAMG preconditioner and not something else.
rows = re.findall(r'GAMGPBiCGStab:\s+Solving for p, Initial residual = ([-+0-9.eE]+),'
                  r' Final residual = ([-+0-9.eE]+), No Iterations (\d+)', s)
if not rows:
    print('  no "GAMGPBiCGStab: Solving for p" lines in the OpenFOAM log -- it did not run the staged'
          ' GAMG-preconditioned PBiCGStab')
    sys.exit(1)
worst = max(float(r[1]) for r in rows)
most = max(int(r[2]) for r in rows)
print('  OpenFOAM GAMGPBiCGStab on p: %d solves, worst final residual %.3e, most iterations %d (cap 2000)'
      % (len(rows), worst, most))
sys.exit(0 if worst < 1e-12 and most < 2000 else 1)
PY

# ARM RAN: the AMG must have preconditioned something. brae's summary line carries pIters (the p solver
# iterations of the first solve of each outer iteration), so the AMG arm taking strictly fewer than the
# diagonal arm on every iteration is the control that the hierarchy was actually applied -- without it a
# build that ignored it would pass EXACT by being identical to REF.
python3 - "$W/b_amg/run.log" "$W/b_diag/run.log" "$N" <<'PY' \
    && say "RAN    the AMG arm takes fewer p solver iterations than the diagonal arm on every iteration" ok \
    || say "RAN    the AMG arm takes fewer p solver iterations than the diagonal arm on every iteration" FAIL
import re, sys
def parse(p):
    out = []
    for line in open(p):
        m = re.match(r'Time = (\d+)\s+(.*)$', line)
        if m:
            out.append({k: float(v) for k, v in re.findall(r'(\w+) ([\d.eE+-]+)', m.group(2))})
    return out
a, b, n = parse(sys.argv[1]), parse(sys.argv[2]), int(sys.argv[3])
if len(a) != n or len(b) != n:
    print('  the two arms did not both reach %d iterations (%d / %d)' % (n, len(a), len(b)))
    sys.exit(1)
ok = True
for i in range(n):
    ai, bi = a[i].get('pIters', -1), b[i].get('pIters', -1)
    print('  iteration %d: pIters AMG %d / diagonal %d      p initial residual %.4e / %.4e'
          % (i + 1, ai, bi, a[i].get('p', float('nan')), b[i].get('p', float('nan'))))
    ok = ok and ai > 0 and bi > 0 and ai < bi
sys.exit(0 if ok else 1)
PY

W="$W" N="$N" BOUND="$BOUND" python3 - <<'PY' || fail=1
import os, re, sys
import numpy as np
W, N, BOUND = os.environ['W'], int(os.environ['N']), float(os.environ['BOUND'])
FIELDS = ('U', 'p', 'T', 'k', 'epsilon')
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'internalField\s+uniform\s+\(?([^);]+)\)?;', s)
    return np.array([float(x) for x in u.group(1).split()])
def load(arm, t):
    out = {}
    for f in FIELDS:
        fn = os.path.join(W, arm, str(t), f)
        out[f] = read(fn) if os.path.exists(fn) else None
    return out
def worst(a, b, label):
    parts, w = [], 0.0
    for f in FIELDS:
        if a[f] is None or b[f] is None or np.shape(a[f]) != np.shape(b[f]):
            parts.append('%s missing' % f)
            w = float('inf')
            continue
        e = float(np.linalg.norm(a[f] - b[f]) / max(np.linalg.norm(b[f]), 1e-300))
        # A field that went non-finite is INFINITELY far from OpenFOAM, not zero away from it: nan
        # compares false against every bound, so a diverged arm would silently pass the `> BOUND` arms.
        if not np.isfinite(e):
            e = float('inf')
        parts.append('%s %.3e' % (f, e))
        w = max(w, e)
    print('  %-34s %s' % (label + ':', '  '.join(parts)))
    return w
bad = 0
def say(ok, what):
    global bad
    print('  %-92s %s' % (what, 'ok' if ok else 'FAIL'))
    if not ok:
        bad = 1
of, ofc = load('of_exact', N), load('of_ctrl', N)
print('  (information) OpenFOAM relTol 0.1 vs OpenFOAM converged: worst %.3e' % worst(ofc, of, 'OF ctrl vs OF exact'))
e = worst(load('b_amg', N), of, 'EXACT AMG vs OF')
say(e <= BOUND, 'EXACT       the AMG-preconditioned BiCGStab tracks OpenFOAM at N=%d (bound %.0e)' % (N, BOUND))
e = worst(load('b_diag', N), of, 'REF diagonal vs OF')
say(e <= BOUND, 'REF         the diagonal opt-out also meets the bound (the bound is meetable here)')
e = worst(load('b_jac', N), of, 'SMOOTHER Jacobi vs OF')
say(e <= BOUND, 'SMOOTHER    the weighted-Jacobi V-cycle meets the same bound (the smoother is a cost, not an answer)')
# The CONTROL must have RUN to mean anything: a missing field reads as `inf`, which would sail past a
# `> BOUND` test while proving nothing about whether the comparison can see a preconditioner difference.
bc = load('b_ctrl', N)
if any(bc[f] is None for f in FIELDS):
    say(False, 'CONTROL     the loose run wrote no t=%d -- the control cannot discriminate' % N)
else:
    e = worst(bc, ofc, 'CONTROL AMG vs OF, relTol 0.1')
    say(e > BOUND, 'CONTROL     at relTol 0.1 the two preconditioners stop elsewhere: MORE than the bound')
# The FAIL-PROOF is the one arm where not finishing IS the evidence -- a pressure solve capped at one
# diagonal iteration can leave the fixed-point iteration unstable. Distinguished from an arm that never
# launched (no run.log), which proves nothing and fails.
bf = load('b_fp', N)
if any(bf[f] is None for f in FIELDS):
    say(os.path.exists(os.path.join(W, 'b_fp', 'run.log')),
        'FAIL-PROOF  the capped run did not reach t=%d at all -- itself a miss of the bound' % N)
else:
    e = worst(bf, of, 'FAIL-PROOF capped vs OF')
    say(e > BOUND, 'FAIL-PROOF  the diagonal capped at one iteration (brae only) misses the bound')
# NON-VACUITY: OpenFOAM's t=N must sit >= 10x the bound away from the start state, or a solver that did
# nothing at all would pass every arm above.
s0, sN = load('of_exact', 0), of
for f in FIELDS:
    a, b = s0[f], sN[f]
    if a is None or b is None:
        say(False, 'NON-VACUOUS %-8s the start state could not be read' % f)
        continue
    a = np.broadcast_to(a, np.shape(b)) if np.shape(a) != np.shape(b) else a
    r0 = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
    say(r0 >= 10 * BOUND, 'NON-VACUOUS %-8s OpenFOAM moved %.3e from 0/ (needs >= 10x the bound)' % (f, r0))
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: the transonic pressure's AMG-preconditioned BiCGStab is exact where it converges, different where it does not, and says what it runs"
exit $fail
