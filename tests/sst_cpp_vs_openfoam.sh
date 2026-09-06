#!/usr/bin/env bash
# kOmegaSST on the MIRRORED _cpp path, against real OpenFOAM. No CUDA is involved anywhere in this gate.
#
# This exists because the _cpp reference could not run an SST case at all, and a single-iteration probe
# did not reveal it: probing from OpenFOAM's converged state gave residuals within 1.2-1.5x of its log
# while the same code, run from 0/, drove omega to 1e+46 by iteration 200. Three defects, all of which
# only an end-to-end run exposes:
#
#   1. div(phi,k)/div(phi,omega) were UPWIND. The SST tutorials ask for `bounded Gauss limitedLinear 1`,
#      and upwind is not a looser tolerance, it is a different matrix: 8.3x (omega) and 82x (k) off
#      OpenFOAM's initial residual, against 1.16x and 1.45x once the scheme is right.
#   2. A negative omega cell was FLOORED to SMALL. OpenFOAM's bound() replaces it with its neighbours'
#      average instead. The next iteration divides CDkOmega by that cell, so a floor contributes ~1e15.
#      OpenFOAM's own log bounds omega 258 times on this case, first at min -2445.7; brae's first
#      negative is -2534.0, the same cell on the same iteration. Upwind never produces one, which is
#      why the floor survived until (1) was fixed.
#   3. correctNut wrote only WALL patches. OF assigns nut_ as a field, so a `calculated` patch carries
#      a1*k_b/max(a1*om_b, b1*F2_b*sqrt(S2_b)). The tutorials ship `calculated; value uniform 0` at the
#      inlet, so from 0/ the inlet eddy viscosity stayed zero for the entire run.
#
# THE ORACLE IS REAL OPENFOAM: validation/pitzDailySST/log.simpleFoam is simpleFoam v2412 output whose
# 2000/ this repo carries, reproduced bit-identically by rerunning OpenFOAM on the checked-in case.
#
# TWO ASSERTIONS. The second is a NEGATIVE CONTROL: re-running with the limited scheme disabled must
# BREACH the same bound, so a passing gate is evidence about the scheme and not about a loose number.
set -u
SRC="${1:?case dir}"
MODE="${2:-fixedpoint}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SST_CPP_BIN:-$ROOT/build/test_simple_sst_cpp}"
[ -x "$BIN" ] || { echo "SKIP: no test_simple_sst_cpp at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -f "$SRC/log.simpleFoam" ] || { echo "SKIP: no OpenFOAM log at $SRC/log.simpleFoam"; exit 77; }
[ -d "$SRC/2000" ]           || { echo "SKIP: no OpenFOAM converged state at $SRC/2000"; exit 77; }

if [ "$MODE" = "full" ]; then
    # End to end from 0/, compared against OpenFOAM's converged fields. The bound is what the ESTABLISHED
    # CUDA solver reaches on the same case (U 1.7e-04, p 5.1e-04, k 6.8e-04, omega 1.3e-02, nut 3.9e-03),
    # rounded out -- the _cpp reference has no licence to be worse than the solver it is the reference for.
    echo "  end-to-end 0 -> 2000 (this takes a few minutes)"
    SST_CPP_TOL=3e-02 "$BIN" "$SRC" 0 2000 2000 || { echo "FAIL: _cpp SST did not match OpenFOAM end to end"; exit 1; }
    echo "  ok:   the _cpp kOmegaSST runs the case end to end and agrees with OpenFOAM"
    exit 0
fi

# ---- fixed point: one iteration from OpenFOAM's own converged state ------------------------------
run_resid() {   # $1 = extra env assignment or "-"
    if [ "$1" = "-" ]; then
        BRAE_SST_DEBUG=1 "$BIN" "$SRC" 2000 2000 1 2>&1
    else
        env "$1" BRAE_SST_DEBUG=1 "$BIN" "$SRC" 2000 2000 1 2>&1
    fi
}

parse() {   # stdin -> "omega k"
    python3 -c '
import re, sys
t = sys.stdin.read()
o = re.search(r"\[omega\] init=([0-9.eE+-]+)", t)
k = re.search(r"\[k\] init=([0-9.eE+-]+)", t)
print(o.group(1) if o else "nan", k.group(1) if k else "nan")
'
}

OFVALS=$(python3 - "$SRC/log.simpleFoam" <<'PY'
import re, sys
seen = False
vals = {}
for line in open(sys.argv[1]):
    if line.startswith('Time = 2000'):
        seen = True
    if not seen:
        continue
    m = re.search(r'Solving for (\w+), Initial residual = ([0-9.eE+-]+)', line)
    if m and m.group(1) not in vals:
        vals[m.group(1)] = float(m.group(2))
print(vals.get('omega', float('nan')), vals.get('k', float('nan')))
PY
)

GOOD=$(run_resid - | parse)
CTRL=$(run_resid SST_NOLIMITED=1 | parse)

python3 - "$OFVALS" "$GOOD" "$CTRL" <<'PY'
import sys
ofo, ofk = [float(x) for x in sys.argv[1].split()]
go,  gk  = [float(x) for x in sys.argv[2].split()]
co,  ck  = [float(x) for x in sys.argv[3].split()]

BOUND = 2.0     # brae reaches 1.16x (omega) and 1.45x (k); the control is 8.3x and 82x
rc = 0
for name, b, o in (("omega", go, ofo), ("k", gk, ofk)):
    r = b / o
    ok = r < BOUND
    print("  %-6s brae %.4e   OpenFOAM %.4e   ratio %.2fx   %s"
          % (name, b, o, r, "ok" if ok else "FAIL (>%.1fx)" % BOUND))
    if not ok:
        rc = 1

# The control must FAIL the same bound, or the bound is not measuring the scheme.
worst = max(co / ofo, ck / ofk)
print("  control (upwind on k/omega): omega %.2fx  k %.2fx" % (co / ofo, ck / ofk))
if worst < BOUND:
    print("  FAIL: the control passes too -- this gate does not discriminate")
    rc = 1
else:
    print("  ok:   the control breaches the bound, so the gate measures the scheme")
sys.exit(rc)
PY
