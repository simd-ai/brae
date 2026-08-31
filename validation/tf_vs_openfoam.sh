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
echo "$out" | grep -q "^PASS" && echo PASS || { echo FAIL; exit 1; }
