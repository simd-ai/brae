#!/usr/bin/env bash
# cellLimited Gauss linear <k>, on BOTH paths, against real OpenFOAM.
#
# `div(phi,U) bounded Gauss linearUpwind limited;` with `limited  cellLimited Gauss linear 1;` in
# gradSchemes -- windAroundBuildings' scheme, and the reason that tutorial was refused: linearUpwind's
# deferred correction is built from the gradient the scheme NAMES, and that correction does NOT vanish at
# convergence, so running the plain Gauss gradient under a limited name solves a different equation.
#
# THE ORACLE IS REAL OPENFOAM: validation/windAroundBuildingsBox/log.simpleFoam and its t=400 fields.
# Both codes assemble the SAME momentum equation from the SAME state, so the initial residual is a direct
# statement about the discretisation.
#
# WHAT THIS MEASURES, and why it discriminates:
#   unlimited (k=0)   Ux residual 2.98e-02   -- 272x OpenFOAM's own 1.09e-04
#   cellLimited (k=1) Ux residual 1.53e-04   -- 1.40x
# so the gate asserts BOTH that the limited form lands near OpenFOAM AND that the unlimited one does not.
# The ratio assertion is the load-bearing one: it is immune to the one-iteration offset between a logged
# residual (start of the step) and the written fields (end of it).
#
# NOTE ON THE MESH: this is windAroundBuildings' blockMesh BACKGROUND box, not its snappyHexMesh result --
# 5000 cells rather than 185237, so that the case can be checked in at 1.5 MB. It is a genuine OpenFOAM
# run on that exact mesh, so the comparison is sound; it is simply not the buildings. On the real snapped
# mesh the same measurement is 6.25x unlimited against 1.33x limited, i.e. the same conclusion with less
# headroom. Do not read this case as covering the tutorial's geometry.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${UEQN_LOCALIZE_BIN:-$ROOT/build/ueqn_localize}"
[ -x "$BIN" ] || { echo "SKIP: no ueqn_localize at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/400" ] || { echo "SKIP: no OpenFOAM state at $SRC/400"; exit 77; }
[ -f "$SRC/log.simpleFoam" ] || { echo "SKIP: no OpenFOAM log at $SRC/log.simpleFoam"; exit 77; }

run() { UEQN_GRADU_LIMIT_K="$1" "$BIN" "$SRC" 400 2>&1; }

OFUX=$(python3 - "$SRC/log.simpleFoam" <<'PY'
import re, sys
seen, val = False, None
for line in open(sys.argv[1]):
    if line.startswith('Time = 400'):
        seen = True
    if seen:
        m = re.search(r'Solving for Ux, Initial residual = ([0-9.eE+-]+)', line)
        if m:
            val = m.group(1); break
print(val if val else 'nan')
PY
)

LIM=$(run 1 | grep 'host   Ux residual' | awk '{print $NF}')
UNL=$(run 0 | grep 'host   Ux residual' | awk '{print $NF}')
LIMD=$(run 1 | grep 'device Ux residual' | awk '{print $NF}')

python3 - "$OFUX" "$LIM" "$UNL" "$LIMD" <<'PY'
import sys
of, lim, unl, limd = (float(x) for x in sys.argv[1:5])
rc = 0
print("  OpenFOAM's own Ux           %.4e" % of)
print("  cellLimited  host           %.4e   %.2fx" % (lim, lim/of))
print("  cellLimited  device         %.4e   %.2fx" % (limd, limd/of))
print("  UNLIMITED    host (control) %.4e   %.2fx" % (unl, unl/of))

if lim / of >= 2.0:
    print("  FAIL: the limited form is not near OpenFOAM (>= 2x)"); rc = 1
# The two paths must agree; they differ only by the nuEff boundary, not by the limiter.
if abs(limd - lim) / lim >= 0.05:
    print("  FAIL: host and device disagree by more than 5%%"); rc = 1
# THE CONTROL: without the limiter the same assembly must be markedly worse, or this gate is vacuous.
if unl / lim < 4.0:
    print("  FAIL: the unlimited control is not worse -- this gate measures nothing"); rc = 1
if rc == 0:
    print("  ok:   cellLimited lands near OpenFOAM on both paths, and dropping it does not")
sys.exit(rc)
PY
