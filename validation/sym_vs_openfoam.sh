#!/bin/bash
# The tilted symmetry plane: host mirror vs real OpenFOAM, device arm refuses by name -- rhoBoxSym/README.md.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBoxSym" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_sym_vs_of}
if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
[ -n "$OFLAST" ] || { echo "FAIL: OpenFOAM produced no output"; exit 1; }

out=$("$BUILD/test_rho_simple_step_cpp" "$WORK" 0 "$OFLAST" 2>&1) || { echo "$out" | tail -15; echo "FAIL(host)"; exit 1; }
echo "$out" | grep -E "^     [UTp] " | head -3
echo "$out" | grep -q "^PASS" || { echo "$out" | tail -5; echo "FAIL(host)"; exit 1; }
echo "PASS(host)"

# DEVICE arm: the segregated model is exact only axis-aligned -- a tilted plane must REFUSE by name.
dout=$("$BUILD/test_rho_simple_step_cuda" "$WORK" 0 3 2>&1) && { echo "FAIL(cuda): a tilted symmetry ran on the device arm"; exit 1; }
echo "$dout" | grep -q "not aligned to a coordinate axis" \
    && echo "PASS(cuda-refused)" \
    || { echo "$dout" | tail -4; echo "FAIL(cuda): refusal does not name the tilt"; exit 1; }
echo PASS
