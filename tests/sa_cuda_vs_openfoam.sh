#!/usr/bin/env bash
# SpalartAllmaras on the CUDA V2 driver, end to end, against real OpenFOAM.
#
# Ported one module at a time onto the working _cpp path (tests/sa_cpp_vs_openfoam.sh): the model on the
# k slot, linearUpwind on div(phi,nuTilda), the freestream valueFraction, the Spalding wall nut, and
# inletOutlet resolved every iteration. Each was measured; the module-by-module numbers are in
# manifest/simpleFoam.yaml.
#
# WHAT THE COLD START COST, and why the fixed-point comparison could not find it: assembled at OpenFOAM's
# own converged fields the two paths' momentum matrices are BIT-IDENTICAL (tests/ueqn_localize.cu:
# 6.904985e-05 on both, same patch split to the digit), at t=0 as well as t=500. The divergence was not in
# the assembly at all but in the nuEff BOUNDARY handed to it, and specifically in the WALL:
#
#   * nut = nuTilda*fv1 is a FIELD ASSIGNMENT, so every boundary face takes nuTilda's patch value. The
#     device took the adjacent cell -- 2.88e-01 out at the outlet.
#   * deviceBoundaryNutSpalding rewrites EVERY face, replacing the non-wall ones with the cell value
#     unless it is handed `nutFile`.
#   * deviceBoundaryNut overwrote the boundary buffer each iteration with cell values, DESTROYING the
#     previous wall nut that Spalding's Newton warm-starts from. OpenFOAM seeds it from nut_ itself; a
#     cold start with 10 iterations and a 1% early-out settled 14% low at convergence, and that alone was
#     the whole cold-start gap: U 1.20e-02 -> 5.99e-05, p 3.32e-02 -> 5.24e-05, nuTilda 3.41e-01 ->
#     1.22e-02, i.e. from 200x worse than the _cpp to matching it.
#
# Solver tolerances are NOT the cause and were ruled out by measurement: tightening every solve from the
# case's relTol 0.1 to relTol 0 leaves the answer bit-identical, as a converged SIMPLE loop should.
#
# THE ORACLE IS REAL OPENFOAM: validation/airFoil2D/log.simpleFoam and its converged 500.
# THE CONTROL removes the turbulence model by pointing the case at laminar, which must fail.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/500" ] || { echo "SKIP: no OpenFOAM converged state at $SRC/500"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

compare() {   # compare <resultDir>
    python3 - "$1" "$SRC/500" <<'PY'
import re, sys, numpy as np
def read(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m: return None
    typ = m.group(1).decode(); n = int(m.group(2)); start = m.end()
    nc = 3 if typ == 'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary':
        return np.frombuffer(b[start:start+n*nc*8], dtype='<f8').reshape(n, nc)
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

got, ref = sys.argv[1], sys.argv[2]
# Set just above where the CUDA path lands (U 6.0e-05, p 5.2e-05, nuTilda/nut 1.2e-02), which is also
# where the _cpp reference lands. The CUDA path has no licence to be worse than the reference it was
# ported from; nuTilda is looser than U/p because it is the transported scalar rather than a derived one.
BOUND = {'U': 3e-04, 'p': 3e-04, 'nuTilda': 3e-02, 'nut': 3e-02}
rc = 0
for f in ('U', 'p', 'nuTilda', 'nut'):
    try:
        a, b = read(f"{got}/{f}"), read(f"{ref}/{f}")
    except OSError:
        # A laminar control writes no nuTilda at all; that IS the field disagreeing.
        print("  %-8s not written   FAIL" % f); rc = 1; continue
    if a is None or b is None:
        print(f"  FAIL: could not read {f}"); rc = 1; continue
    e = np.linalg.norm(a - b) / np.linalg.norm(b)
    ok = e < BOUND[f]
    print("  %-8s %.3e   bound %.1e   %s" % (f, e, BOUND[f], "ok" if ok else "FAIL"))
    if not ok: rc = 1
sys.exit(rc)
PY
}

prep() {   # prep <dir>: run the full 500, no early stop, so the comparison is like-for-like with OF
    python3 - "$1/system/fvSolution" <<'PY'
import io, re, sys
p = sys.argv[1]
s = io.open(p).read()
s = re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl\n    {\n    }', s)
io.open(p, 'w').write(s)
PY
}

echo "== SA on the CUDA V2 driver: must match OpenFOAM end to end =="
cp -r "$SRC" "$W/on"
rm -rf "$W/on/500" "$W/on/log.simpleFoam"
prep "$W/on"
( cd "$W/on" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
    echo "FAIL: brae refused or crashed on the SA case"; tail -n 8 "$W/on/run.log"; exit 1; }
grep -q 'div(phi,nuTilda): bounded linearUpwind' "$W/on/run.log" || {
    echo "FAIL: brae did not resolve div(phi,nuTilda) to \`bounded linearUpwind\`"; exit 1; }
compare "$W/on/500" || { echo "FAIL: the CUDA SA did not match OpenFOAM end to end"; exit 1; }

echo "== control: laminar must NOT match =="
cp -r "$SRC" "$W/off"
rm -rf "$W/off/500" "$W/off/log.simpleFoam"
prep "$W/off"
python3 - "$W/off/constant/turbulenceProperties" <<'PY'
import io, re, sys
p = sys.argv[1]
s = io.open(p).read()
s = re.sub(r'simulationType\s+\w+;', 'simulationType          laminar;', s)
io.open(p, 'w').write(s)
PY
( cd "$W/off" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
    echo "FAIL: the laminar control run itself crashed"; tail -n 5 "$W/off/run.log"; exit 1; }
if compare "$W/off/500" > "$W/ctrl.txt" 2>&1; then
    cat "$W/ctrl.txt"
    echo "FAIL: the case passes WITHOUT the turbulence model -- this gate measures nothing"
    exit 1
fi
sed -n '1,4p' "$W/ctrl.txt"
echo "  ok:   the control fails, so the gate is measuring the model"
