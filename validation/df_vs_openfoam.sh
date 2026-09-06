#!/bin/bash
# DarcyForchheimer porosity, host mirror vs real OpenFOAM -- see validation/rhoBoxDF/README.md.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBoxDF" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_df_vs_of}
if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
topoSet > log.topoSet 2>&1
grep -q "cellZoneSet plug now size 240" log.topoSet || { echo "FAIL: the plug cellZone did not build"; exit 1; }
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
[ -n "$OFLAST" ] || { echo "FAIL: OpenFOAM produced no output"; exit 1; }

out=$("$BUILD/test_rho_simple_step_cpp" "$WORK" 0 "$OFLAST" 2>&1) || { echo "$out" | tail -20; echo FAIL; exit 1; }
# ENGAGEMENT: the porosity must actually be read and implemented, or this is the plain rhoBox gate
echo "$out" | grep -q "fvOptions: 1 option(s), all implemented" || { echo "FAIL: the porosity never engaged"; exit 1; }
echo "$out" | grep -E "^     [UTp] " | head -3
echo "$out" | grep -q "^PASS" && echo PASS || { echo "$out" | tail -5; echo FAIL; exit 1; }
