#!/usr/bin/env bash
# nSweeps: SMOOTHING SWEEPS BETWEEN RESIDUAL EVALUATIONS, and the count OpenFOAM reports is a multiple.
#
# smoothSolver.C:78 reads `nSweeps` from the field's own solver entry (default 1). The positive branch
# (:179-209) smooths nSweeps_ times per pass of the do-while (:186) and then advances the count by
# nSweeps_ (:205), so the convergence test is consulted only on a multiple of it, the solve OVERSHOOTS
# its relTol by whatever the extra sweeps buy, and maxIter bounds SWEEPS rather than evaluations.
#
# brae read the entry nowhere. validation/airFoil2D's own fvSolution says `nSweeps 2` on U and nuTilda,
# and brae reported ODD counts (7, 3, 3) where OpenFOAM's were even in all 600 of its solves.
#
#   ARM 1  the count brae reports is a MULTIPLE of nSweeps and at least nSweeps. That is the modular
#          fact both codes must satisfy; the EXACT count cannot be OpenFOAM's, because brae's sweep is
#          multicolour where OpenFOAM's is index order -- a substitution that is separately measured and
#          separately announced (tests/gs_smoother_notice.sh), and this gate must not pretend otherwise.
#   ARM 2  raising nSweeps must push the solve FURTHER below its relTol, on both codes. Oracle:
#          OpenFOAM's own first Ux solve on the identical staged case leaves at 9.04154904498e-04 with
#          nSweeps 1 and 1.40100300931e-06 with nSweeps 2 -- 645x lower under the same relTol 0.1.
#   REFUSAL a NEGATIVE nSweeps must be refused by name. It is not a tuning: smoothSolver.C:96-119 deletes
#          the convergence test and reports a residual of ZERO, which residualControl reads as converged.
#          `nSweeps 0` is deliberately NOT refused -- OpenFOAM runs it fine whenever the initial residual
#          already satisfies the tolerance, and brae agrees with it there.
#   CONTROL OpenFOAM's own nSweeps-2 residual must be well below its nSweeps-1 one, or ARM 2 is noise.
#
# Fail-proof, 2026-09-04: changing the accumulator `sweeps += sweepsPer` to `sweeps += 1` leaves the
# extra sweeps in place but breaks the count, and ARM 1's nSweeps-2 row goes red (3, not a multiple of 2).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/simpleBoxIO}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
CONTROL_RATIO=${CONTROL_RATIO:-10}
SA_SRC="${2:-$ROOT/validation/airFoil2D}"

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # stage <dir> <nSweeps>
{
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    if [ -f "$1/system/blockMeshDict" ] && command -v blockMesh > /dev/null 2>&1; then
        ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
    fi
    python3 - "$1" "$2" <<'PY'
import re, sys
d, n = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'; s = open(c).read()
# One-line controlDict: nothing here may be line-anchored.
s = re.sub(r'endTime\s+\S+;', 'endTime         3;', s)
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S)
# The whole U block is rewritten so the fixture's own settings cannot leak into the arm.
s = re.sub(r'U \{[^}]*\}',
           'U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.1; nSweeps %s; }'
           % n, s)
open(f, 'w').write(s)
PY
}

declare -A ofn ofr brn brr
for n in 1 2 3; do
    stage "$W/of$n" "$n"; stage "$W/br$n" "$n"
    ( cd "$W/of$n" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM at nSweeps $n"; exit 1; }
    ( cd "$W/br$n" && BRAE_SIMPLEFOAM_V2=1 BRAE_SOLVER_ITERS=1 "$BRAE" "$W/br$n" > run.log 2>&1 ) \
        || { echo "FAIL: brae at nSweeps $n"; tail -8 "$W/br$n/run.log"; exit 1; }
    ofn[$n]=$(grep -m1 "Solving for Ux" "$W/of$n/run.log" | sed 's/.*No Iterations //')
    ofr[$n]=$(grep -m1 "Solving for Ux" "$W/of$n/run.log" | sed 's/.*Final residual = \([0-9.eE+-]*\).*/\1/')
    brn[$n]=$(grep -m1 -oP '\[U0\] nIter=\K[0-9]+' "$W/br$n/run.log")
    brr[$n]=$(grep -m1 -oP '\[U0\].*final=\K[0-9.eE+-]+' "$W/br$n/run.log")
    printf '  nSweeps %s: OpenFOAM nIter %-3s final %-16s | brae nIter %-3s final %s\n' \
        "$n" "${ofn[$n]}" "${ofr[$n]}" "${brn[$n]}" "${brr[$n]}"
done

for n in 1 2 3; do
    say "ARM 1  OpenFOAM's count ${ofn[$n]} is a multiple of nSweeps $n" \
        "$([ $(( ${ofn[$n]} % n )) -eq 0 ] && [ ${ofn[$n]} -ge $n ] && echo ok || echo FAIL)"
    say "ARM 1  brae's count ${brn[$n]} is a multiple of nSweeps $n and >= $n" \
        "$([ $(( ${brn[$n]} % n )) -eq 0 ] && [ ${brn[$n]} -ge $n ] && echo ok || echo FAIL)"
done
python3 - "${ofr[1]}" "${ofr[2]}" "${brr[1]}" "${brr[2]}" "$CONTROL_RATIO" <<'PY'
import sys
o1, o2, b1, b2 = (float(x) for x in sys.argv[1:5]); ratio = float(sys.argv[5])
def say(m, ok):
    print('  %-70s %s' % (m, 'ok' if ok else 'FAIL'))
    return 0 if ok else 1
rc = 0
rc |= say('CONTROL  OpenFOAM at nSweeps 2 is %.1fx further down than at 1 (>= %.0fx)'
          % (o1 / o2, ratio), o1 / o2 >= ratio)
rc |= say('ARM 2  brae at nSweeps 2 is %.1fx further down than at 1' % (b1 / b2), b2 < b1)
sys.exit(rc)
PY
[ $? -eq 0 ] || fail=1

# ARM 5 -- THE TRANSPORTED TURBULENCE SCALAR. Item 40 wired nSweeps into the momentum solve only, and the
# turbulence half was announced instead. validation/airFoil2D really does ship `nSweeps 2` on nuTilda
# beside `solver smoothSolver`, and brae ran it on Jacobi-BiCGStab at brae's own tolerance -- because the
# whole turbulence solver-control block was guarded on the literal `solvers/k`, which a SpalartAllmaras
# case does not have. Measured before: nuTilda counts 31, 29, 31 (odd, and not even a smoothSolver).
# OpenFOAM's own counts on that fixture at nSweeps 2 were EVEN in all 50 of its solves.
if [ -d "$SA_SRC" ]; then
    rm -rf "$W/sa"; cp -r "$SA_SRC" "$W/sa"; rm -rf "$W"/sa/[1-9]* "$W"/sa/log.*
    python3 - "$W/sa" <<'PYEOF'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;', 'endTime         3;', s)
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
PYEOF
    ( cd "$W/sa" && BRAE_SIMPLEFOAM_V2=1 BRAE_TURB_RESID=1 "$BRAE" "$W/sa" > run.log 2>&1 ) \
        || { echo "FAIL: brae crashed on the SA arm"; tail -8 "$W/sa/run.log"; exit 1; }
    nsw=$(python3 -c "
import re,sys
s=open('$SA_SRC/system/fvSolution').read()
m=re.search(r'nuTilda\s*\{[^}]*\}', s, re.S)
n=re.search(r'nSweeps\s+(\d+)\s*;', m.group(0)) if m else None
print(n.group(1) if n else 1)")
    cnt=$(grep -oP 'Solving for nuTilda,.*No Iterations \K[0-9]+' "$W/sa/run.log" | head -3 | tr '\n' ' ')
    bad=$(python3 -c "
ns=int('$nsw'); c='$cnt'.split()
print(0 if c and all(int(x)%ns==0 and int(x)>=ns for x in c) else 1)")
    say "ARM 5  nuTilda counts [$cnt] are multiples of its own nSweeps $nsw" \
        "$([ "$bad" = "0" ] && echo ok || echo FAIL)"
    grep -q "solver=smoothSolver" "$W/sa/run.log" \
        && say "ARM 5  the SA case's own smoothSolver selection is read (it has no solvers/k)" ok \
        || say "ARM 5  the SA case's own smoothSolver selection is read (it has no solvers/k)" FAIL
    say "CONTROL  that fixture's nuTilda really does set nSweeps > 1" \
        "$([ "$nsw" -gt 1 ] && echo ok || echo FAIL)"
fi

# The refusal.
stage "$W/neg" -3
out=$( cd "$W/neg" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/neg" 2>&1 || true )
echo "$out" | grep -q "negative nSweeps is OpenFOAM's FIXED-SWEEP mode" \
    && say "REFUSAL  a negative nSweeps is refused by name" ok \
    || say "REFUSAL  a negative nSweeps is refused by name" FAIL

[ $fail -eq 0 ] && echo "PASS: nSweeps is read per field, counts sweeps as OpenFOAM does, and the fixed-sweep mode is refused"
exit $fail
