#!/usr/bin/env bash
# MRF on the CUDA V2 driver, against real OpenFOAM and against the _cpp reference it was ported from.
#
# Ported one hook at a time onto the working _cpp path, testing after each. The numbers below are the
# single-iteration momentum residual assembled at OpenFOAM's OWN converged 500, against its 1.536e-05:
#
#   hook 1  MRF.correctBoundaryVelocity(U)   UEqn.H:3   2.53e-02   (1648x -- the other two still missing)
#   hook 2  + MRF.DDt(U)                     UEqn.H:8   1.5256e-05 (0.7%)
#   hook 3  + MRF.makeRelative(phiHbyA)      pEqn.H:5   1.5256e-05 (unchanged -- see below)
#
# HOOK 3 LOOKS INERT IN ONE ITERATION AND IS NOT. Omega x r is a solid-body rotation, which is
# DIVERGENCE-FREE, so subtracting its flux leaves div(phiHbyA) -- the pressure equation's whole source --
# unchanged, and p moves only in the 7th digit. What it changes is phi itself, which is the convecting
# flux for the NEXT iteration. Run end to end the difference is total: U 2.5e-03 with it against 7.2e-01
# without. A gate that only measured one iteration would have called this hook dead code and dropped it.
#
# (An earlier bisect on the _cpp put makeRelative at 30x on the pressure residual. That was measured
# before the noSlip fix, when the rotor wall was still treated as stationary and the boundary flux was
# inconsistent with the interior; it does not hold on the fixed code, and the divergence-free argument
# above is why.)
#
# THE ORACLE IS REAL OPENFOAM: validation/mixerVessel2D/log.simpleFoam and its converged 500.
#
# THE CONTROL removes constant/MRFProperties and changes nothing else. The case then has no driving force
# at all -- the flow is quiescent and every field reads 1.000 -- so a gate that passed without the frame
# terms would be measuring nothing. It exercises the real dictionary path rather than a debug switch.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/500" ] || { echo "SKIP: no OpenFOAM converged state at $SRC/500"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

compare() {   # compare <resultDir> <label>; prints one line per field
    python3 - "$1" "$SRC/500" "$2" <<'PY'
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

got, ref, label = sys.argv[1], sys.argv[2], sys.argv[3]
# The bounds are what the _cpp reference reaches on the same case, rounded out. The CUDA path has no
# licence to be worse than the reference it was ported from.
BOUND = {'U': 6e-03, 'p': 3e-03, 'k': 3e-02, 'epsilon': 3e-02}
rc = 0
for f in ('U', 'p', 'k', 'epsilon'):
    a, b = read(f"{got}/{f}"), read(f"{ref}/{f}")
    if a is None or b is None:
        print(f"  FAIL: could not read {f}"); rc = 1; continue
    e = np.linalg.norm(a - b) / np.linalg.norm(b)
    ok = e < BOUND[f]
    print("  %-8s %.3e   bound %.1e   %s" % (f, e, BOUND[f], "ok" if ok else "FAIL"))
    if not ok: rc = 1
sys.exit(rc)
PY
}

echo "== MRF on: the CUDA V2 driver must match OpenFOAM end to end =="
cp -r "$SRC" "$W/on"
rm -rf "$W/on/500" "$W/on/log.simpleFoam"
( cd "$W/on" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
    echo "FAIL: brae refused or crashed on the MRF case"; tail -n 8 "$W/on/run.log"; exit 1; }
grep -q 'MRF: 1 zone' "$W/on/run.log" || { echo "FAIL: brae did not report resolving the MRF zone"; exit 1; }
compare "$W/on/500" on || { echo "FAIL: the CUDA MRF did not match OpenFOAM end to end"; exit 1; }

echo "== control: the same case with constant/MRFProperties removed must NOT match =="
cp -r "$SRC" "$W/off"
rm -rf "$W/off/500" "$W/off/log.simpleFoam" "$W/off/constant/MRFProperties"
( cd "$W/off" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
    echo "FAIL: the control run itself crashed"; tail -n 5 "$W/off/run.log"; exit 1; }
if compare "$W/off/500" off > "$W/ctrl.txt" 2>&1; then
    cat "$W/ctrl.txt"
    echo "FAIL: the case passes WITHOUT the frame terms -- this gate measures nothing"
    exit 1
fi
sed -n '1,4p' "$W/ctrl.txt"
echo "  ok:   the control fails, so the gate is measuring the frame terms"
