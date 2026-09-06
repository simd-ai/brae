#!/usr/bin/env bash
# kOmegaSST on the CUDA V2 path, against real OpenFOAM AND against the _cpp reference.
#
# Ported one module at a time onto the working _cpp path, testing after each:
#   1. div(phi,k)/div(phi,omega) read from fvSchemes -> limitedLinear + bounded, instead of hardcoded
#      upwind.                                            omega 19x -> 6.8x,  k 125x -> 23x
#   2. the `calculated` nut patches evaluated as a1*k_b/max(a1*om_b, b1*F2_b*sqrt(S2_b)).
#   3. the k/omega patch diffusivity DkEff(patchi) = alphaK(F1)*nut_b + nu from nut's own boundary.
#   4. Foam::bound -- already correct on the device (fvc::average on a negative cell, not a floor).
# 2 and 3 are inert at the fixed point BY CONSTRUCTION (nut's boundary is read from OpenFOAM's converged
# file, where it is already right); the end-to-end mode is what exercises them.
#
# THE REGRESSION THIS EXISTS TO CATCH: the wall mask was keyed on `isEpsilonWallFunction`, and
# omegaWallFunction mapped to a plain zeroGradient field, so EVERY kOmegaSST case ran with no wall faces
# at all -- no wall nut, so the wall shear and everything downstream of it were wrong. Nothing caught it,
# because no gate ran an SST case through the rebuilt driver. It measured omega 6.8x and U 48x off
# OpenFOAM; correct, it is 1.16x and 1.05x.
#
# THE ORACLE IS REAL OPENFOAM: validation/pitzDailySST/log.simpleFoam, whose 2000/ this repo carries.
set -u
SRC="${1:?case dir}"
MODE="${2:-fixedpoint}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -f "$SRC/log.simpleFoam" ] || { echo "SKIP: no OpenFOAM log at $SRC/log.simpleFoam"; exit 77; }
[ -d "$SRC/2000" ]           || { echo "SKIP: no OpenFOAM converged state at $SRC/2000"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case"
rm -f "$W/case/log.simpleFoam"

if [ "$MODE" = "full" ]; then
    rm -rf "$W/case/2000"
    ( cd "$W/case" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
        echo "FAIL: brae refused or crashed on the SST case"; tail -n 8 "$W/case/run.log"; exit 1; }
    python3 - "$W/case/2000" "$SRC/2000" <<'PY'
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
# The bounds are what the ESTABLISHED CUDA solver reaches on this case, rounded out. omega carries a
# looser one because its L2 is dominated by the near-wall values, where the case's own loose
# `tol 1e-05 relTol 0.1` on the omega solve is worth a couple of percent.
BOUND = {'U': 1e-03, 'p': 3e-03, 'k': 3e-03, 'omega': 6e-02, 'nut': 2e-02}
rc = 0
for f in ('U', 'p', 'k', 'omega', 'nut'):
    a, b = read(f"{got}/{f}"), read(f"{ref}/{f}")
    if a is None or b is None:
        print(f"  FAIL: could not read {f}"); rc = 1; continue
    e = np.linalg.norm(a - b) / np.linalg.norm(b)
    ok = e < BOUND[f]
    print("  %-6s %.3e   bound %.1e   %s" % (f, e, BOUND[f], "ok" if ok else "FAIL"))
    if not ok: rc = 1
if rc == 0:
    print("  ok:   the CUDA V2 path runs kOmegaSST end to end and agrees with OpenFOAM")
sys.exit(rc)
PY
    exit $?
fi

# ---- fixed point: one iteration from OpenFOAM's own converged state ------------------------------
python3 - "$W/case/system/controlDict" <<'PY'
import io, re, sys
p = sys.argv[1]
s = io.open(p).read()
s = re.sub(r'startFrom\s+\w+;',        'startFrom       startTime;', s)
s = re.sub(r'startTime\s+[\d.]+;',     'startTime       2000;',      s)
s = re.sub(r'endTime\s+[\d.]+;',       'endTime         2001;',      s)
s = re.sub(r'writeInterval\s+[\d.]+;', 'writeInterval   1000;',      s)
io.open(p, 'w').write(s)
PY

( cd "$W/case" && BRAE_TURB_RESID=1 BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
    echo "FAIL: brae refused or crashed starting from OpenFOAM's converged state"
    tail -n 8 "$W/case/run.log"; exit 1; }

grep -q 'div(phi,k): bounded limitedLinear' "$W/case/run.log" || {
    echo "FAIL: brae did not resolve div(phi,k) to `bounded limitedLinear`"; exit 1; }

python3 - "$W/case/run.log" "$SRC/log.simpleFoam" <<'PY'
import io, re, sys
brae_log, of_log = sys.argv[1], sys.argv[2]

braev = {}
for line in io.open(brae_log):
    m = re.search(r'Solving for (\w+), Initial residual = ([0-9.eE+-]+)', line)
    if m and m.group(1) not in braev:
        braev[m.group(1)] = float(m.group(2))
    m2 = re.search(r'U initial residual = ([0-9.eE+-]+)', line)
    if m2 and 'U' not in braev:
        braev['U'] = float(m2.group(1))

ofv, seen = {}, False
for line in io.open(of_log):
    if line.startswith('Time = 2000'):
        seen = True
    if not seen:
        continue
    m = re.search(r'Solving for (\w+), Initial residual = ([0-9.eE+-]+)', line)
    if m:
        # U's oracle is OpenFOAM's own cmptMax over the components it SOLVED, not Ux alone.
        # solutionControl.C:232 compares cmptMax over the stored per-component vector, and on
        # pitzDailySST at t=2000 Uy is 5.76x Ux (Ux 2.123031232446694e-05, Uy 1.22204041856011e-04), so
        # `Ux` was the wrong oracle by that factor. Taking the max is a TIGHTENING even though the
        # denominator grows: the bound below moved 2.0 -> 1.2 on the measured ratio, not the other way.
        if m.group(1) in ('Ux', 'Uy', 'Uz'):
            ofv['U'] = max(ofv.get('U', 0.0), float(m.group(2)))
        elif m.group(1) not in ofv:
            ofv[m.group(1)] = float(m.group(2))

# Assembling at OpenFOAM's own converged fields makes the initial residual a direct statement about the
# discretisation. 2.0 was far tighter than the 6.8x/48x the wall-mask defect measured; with U's oracle
# corrected to OpenFOAM's own cmptMax the measured ratios are U 1.01x, omega 1.13x, k 0.97x, so the bound
# TIGHTENS to 1.2x -- it never loosens.
BOUND = 1.2
rc = 0
for f in ('U', 'omega', 'k'):
    if f not in braev or f not in ofv:
        print(f"  FAIL: missing {f}"); rc = 1; continue
    r = braev[f] / ofv[f]
    ok = r < BOUND
    print("  %-6s brae %.4e   OpenFOAM %.4e   ratio %.2fx   %s"
          % (f, braev[f], ofv[f], r, "ok" if ok else "FAIL (>%.1fx)" % BOUND))
    if not ok: rc = 1
if rc == 0:
    print("  ok:   the CUDA V2 kOmegaSST matches OpenFOAM at its own converged state")
sys.exit(rc)
PY
