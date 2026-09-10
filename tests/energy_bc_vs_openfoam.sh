#!/usr/bin/env bash
# The ENERGY boundary conditions -- fixedEnergy, gradientEnergy, mixedEnergy -- against OpenFOAM's OWN
# coefficients, on three fixtures that between them carry every T boundary condition
# basicThermo::heBoundaryTypes maps onto one.
#
#     validation/rhoBoxQ   fixedValue, zeroGradient, fixedGradient, empty
#     validation/rhoBoxM   fixedValue, zeroGradient, mixed,         empty
#     validation/sbMatched fixedValue, zeroGradient, inletOutlet
#
# THE ORACLE is tools/dumpEnergyBC, an OpenFOAM utility that constructs the case's own fluidThermo and
# calls he.boundaryFieldRef().updateCoeffs() -- what the energy matrix does before reading a coefficient
# -- then prints refValue/refGrad/valueFraction/gradient/value at 17 digits. brae reads the SAME time
# directory, so the comparison has no trajectory in it at all: both codes compute the boundary
# coefficients of the same p and T.
#
# EACH FIXTURE IS DEVELOPED FIRST, not read at 0.orig. A uniform start makes T's patch value equal to its
# refValue on the mixed wall, which collapses control B onto the answer and stops it discriminating.
#
# THE COEFFICIENTS ARE POISONED BEFORE THE CALL, and that is what makes this gate bite. Measured: with
# updateEnergyBoundaryCoeffs deleted outright, every bound below still read 0.000000e+00 on all three
# fixtures, because on this thermo the construction-time mapping already holds the right numbers. The
# probe therefore overwrites every coefficient with -1e30 first -- see tests/test_energy_bc.cu for why
# that mirrors OpenFOAM's own construction state rather than inventing one -- and the same deletion then
# reads 7e+22 to 1e+30 and exits 1.
#
# Measured, OpenFOAM v2412 (see tests/test_energy_bc.cu for what each control is):
#     rhoBoxQ    gradient 0.0   value 1.7e-16                             control A 2.09e-02
#     rhoBoxM    refValue 0.0   refGrad 0.0   vf 0.0   value 1.7e-16      control B 8.68e-01  C 9.99e-01
#     sbMatched  refValue 0.0   refGrad 0.0   vf 0.0   value 0.0          control B 1.22e-01
# The bound is 1e-13. The implemented path reproduces OpenFOAM exactly; the controls miss by 1e-2 to 1;
# and the three fail-proofs -- affine gradient, refValue from T's value, call deleted -- each turn the
# gate red on the fixture that carries the boundary condition.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_energy_bc"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-40}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v dumpEnergyBC  > /dev/null 2>&1 || {
    echo "SKIP: dumpEnergyBC not built -- (cd $ROOT/tools/dumpEnergyBC && wmake)"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
rc=0

for FIX in rhoBoxQ rhoBoxM sbMatched; do
    SRC="$ROOT/validation/$FIX"
    [ -d "$SRC" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }

    C="$W/$FIX"
    cp -r "$SRC" "$C"
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"

    # The dictionaries these fixtures ship are single-line, so the edits are token-wise and not
    # line-anchored -- an anchored sed silently rewrites a whole line of unrelated keys.
    ITERS="$ITERS" python3 - "$C" <<'PYEOF'
import os, re, sys
d, n = sys.argv[1], os.environ['ITERS']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+\S+;',        'endTime %s;' % n,     s)
s = re.sub(r'\bwriteInterval\s+\S+;',  'writeInterval %s;' % n, s)
s = re.sub(r'\bwriteControl\s+\S+;',   'writeControl timeStep;', s)
s = re.sub(r'\bwriteFormat\s+\S+;',    'writeFormat ascii;',  s)
s = re.sub(r'\bwritePrecision\s+\S+;', 'writePrecision 17;',  s)
s = re.sub(r'\bstopAt\s+\S+;',         'stopAt endTime;',     s)
open(c, 'w').write(s)
# residualControl would stop the run before the write, leaving no developed time directory.
p = os.path.join(d, 'system/fvSolution')
s = open(p).read()
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
PYEOF

    # rhoBoxQ and rhoBoxM ship a blockMeshDict, sbMatched ships the mesh itself.
    if [ ! -d "$C/constant/polyMesh" ]; then
        ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || {
            echo "FAIL: $FIX -- blockMesh failed"; tail -20 "$C/log.blockMesh"; rc=1; continue; }
    fi
    ( cd "$C" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ) || {
        echo "FAIL: $FIX -- OpenFOAM did not run"; tail -20 "$C/log.rhoSimpleFoam"; rc=1; continue; }

    T=$( (cd "$C" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1) )
    [ -n "$T" ] || { echo "FAIL: $FIX -- OpenFOAM wrote no time directory"; rc=1; continue; }

    ( cd "$C" && dumpEnergyBC -case . -time "$T" > oracle.txt 2>&1 ) || {
        echo "FAIL: $FIX -- dumpEnergyBC failed"; tail -20 "$C/oracle.txt"; rc=1; continue; }
    grep -q '^END$' "$C/oracle.txt" || {
        echo "FAIL: $FIX -- oracle is truncated"; tail -20 "$C/oracle.txt"; rc=1; continue; }

    echo "== $FIX (OpenFOAM time $T) =="
    "$BIN" "$C" "$C/$T" "$C/oracle.txt" | tee "$W/$FIX.out"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
done

# COVERAGE, asserted rather than assumed: between them the three fixtures must have exercised all three
# controls. A fixture list that silently lost its mixed wall would leave B and C at zero, and every
# remaining bound would still be green -- a gate measuring nothing and reporting a pass.
echo "== coverage =="
for K in A B C; do
    N=$(cat "$W"/*.out 2>/dev/null | sed -n 's/.*controls exercised: //p' \
        | tr ' ' '\n' | sed -n "s/^$K=//p" | paste -sd+ - | bc)
    N=${N:-0}
    if [ "$N" -gt 0 ]; then
        echo "     control $K exercised on $N faces                    ok"
    else
        echo "     control $K exercised on NO faces                    FAIL"
        rc=1
    fi
done

exit $rc
