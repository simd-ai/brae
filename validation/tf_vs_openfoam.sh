#!/bin/bash
# `RAS { turbulence off; }` -- brae's mirror vs real OpenFOAM on the rhoBoxF fixture.
#
# The defect this exists to catch: brae treated `turbulence off` as laminar (nut never read, mut = 0)
# where OpenFOAM transports the FROZEN validate()-time nut = Cmu*k^2/eps -- eighty times the molecular
# viscosity on this fixture. See validation/rhoBoxF/README.md for the oracle numbers.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBoxF" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_tf_vs_of}
if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
[ -n "$OFLAST" ] || { echo "FAIL: OpenFOAM produced no output"; exit 1; }
grep -q "solution converged" log.rhoSimpleFoam || echo "note: OF hit endTime rather than residualControl"

out=$("$BUILD/test_rho_simple_step_cpp" "$WORK" 0 "$OFLAST" 2>&1) || { echo "$out" | tail -30; echo FAIL; exit 1; }
echo "$out" | grep -E "frozen arm|validate|bit-identical|file seed" | head -6
# the frozen arm must have ENGAGED -- a fixture regression (turbulence back on, say) would otherwise
# turn this into a second copy of the ordinary end-to-end gate and prove nothing about freezing
echo "$out" | grep -q "frozen arm -- turbulence off" || { echo "FAIL: the frozen arm never engaged"; exit 1; }
echo "$out" | tail -3
# continuityErrs (pEqn.H:81, the INCOMPRESSIBLE file): brae now computes it at OpenFOAM's exact point.
# Both solvers sit at their linear-tolerance floors here, so the gate bounds brae ABSOLUTELY (measured
# 2.1e-09; a sign error in the boundary half of div(phi) reads ~1e-02) and only sanity-checks OF's.
bcont=$(echo "$out" | grep "continuity errors" | tail -1 | grep -oE "sum local = [-0-9.e+]+" | grep -oE "[-0-9.e+]+$")
ocont=$(grep "continuity errors" log.rhoSimpleFoam | tail -1 | grep -oE "sum local = [-0-9.e+]+" | grep -oE "[-0-9.e+]+$")
python3 - "$bcont" "$ocont" <<'PYEOF'
import sys
b, o = float(sys.argv[1]), float(sys.argv[2])
print(f"  continuity sum local: brae {b:.3e}  OF {o:.3e}")
ok = abs(b) < 1e-7 and abs(o) < 1e-7
print("  both at the converged floor" if ok else "FAIL: continuity error off the floor")
sys.exit(0 if ok else 1)
PYEOF
echo "$out" | grep -q "^PASS" && echo PASS || { echo FAIL; exit 1; }
