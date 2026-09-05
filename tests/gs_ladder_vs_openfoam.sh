#!/usr/bin/env bash
# Both of OpenFOAM's GaussSeidel smoothers, measured against its own residual after exactly n sweeps.
#
# brae routes `smoothSolver` + a GaussSeidel-family smoother to deviceSymGaussSeidel. That used to be a
# COLOUR-order sweep for both smoother names -- a different smoother wearing each name. It is now
# OpenFOAM's own index-order sweep, level-scheduled onto the device, in whichever of the two variants the
# case asked for: symGaussSeidelSmoother.C walks the cells up then back down, GaussSeidelSmoother.C walks
# them up ONLY. This gate is what says brae is each of them, rather than one of them twice.
#
# THE ORACLE is tools/dumpSimpleFoam, run for one iteration on the fixture: it writes the momentum system
# in the folded form the linear solver sees, the field the solve starts from, and OpenFOAM's residual after
# exactly n sweeps for n = 1..10 (a COPY of the system solved with `tolerance 0; relTol 0; maxIter n`) --
# once for `symGaussSeidel` and once for `GaussSeidel`, because those are two different smoothers and brae
# has to be each of them, not one of them twice.
# tests/gs_ladder.cu reads all of it and runs both a transcription of OpenFOAM's smoother and brae's own.
#
# This gate needs no brae binary: the case never runs through brae. It is the sweep alone.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${GS_LADDER_BIN:-$ROOT/build/gs_ladder}"
SRC="${1:-$ROOT/validation/T3A}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BIN" ]      || { echo "SKIP: no gs_ladder at $BIN"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
DUMP="$(command -v dumpSimpleFoam || true)"
[ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpSimpleFoam" ] \
    && DUMP="$FOAM_USER_APPBIN/dumpSimpleFoam"
[ -n "$DUMP" ] || { echo "SKIP: dumpSimpleFoam not built -- (cd tools/dumpSimpleFoam && wmake)"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/of"
cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$W/of/"
cp -r "$W/of/0.orig" "$W/of/0"
python3 - "$W/of" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^endTime .*',          'endTime         1;',    s, flags=re.M)
s = re.sub(r'^writeInterval .*',    'writeInterval   1;',    s, flags=re.M)
s = re.sub(r'^writePrecision .*',   'writePrecision  15;',   s, flags=re.M)
s = re.sub(r'^writeCompression .*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PY

( cd "$W/of" && BRAE_DUMP_STAGE_ITER=1 "$DUMP" > dump.log 2>&1 ) \
    || { echo "FAIL: dumpSimpleFoam did not run"; tail -20 "$W/of/dump.log"; exit 1; }
for f in stage_UsolveDiag stage_UsolveSrc stage_Usolve0 stage_UsolveUpper stage_UsolveLower; do
    [ -f "$W/of/1/$f" ] || { echo "FAIL: dumpSimpleFoam wrote no 1/$f"; tail -20 "$W/of/dump.log"; exit 1; }
done
for f in stage_UsmoothLadder.dat stage_UgsLadder.dat; do
    [ -f "$W/of/$f" ] || { echo "FAIL: dumpSimpleFoam wrote no $f"; tail -20 "$W/of/dump.log"; exit 1; }
done

# THE ORACLE'S OWN CONTROL: the ladder must agree with the case's real solve, which the same run logged.
# `Solving for Ux ... Final residual = R, No Iterations N` has to be rung N of the ladder, or the ladder is
# measuring a different system than OpenFOAM solved and every number below is meaningless.
python3 - "$W/of" <<'PY' || exit 1
import re, sys
w = sys.argv[1]
log = open(w + '/dump.log').read()
m = re.search(r'Solving for Ux, Initial residual = ([0-9.eE+-]+), '
              r'Final residual = ([0-9.eE+-]+), No Iterations (\d+)', log)
if not m:
    print("FAIL: dumpSimpleFoam logged no Ux solve"); sys.exit(1)
init, fin, nit = float(m.group(1)), float(m.group(2)), int(m.group(3))
rungs = {}
for line in open(w + '/stage_UsmoothLadder.dat'):
    n, i, f = line.split(); rungs[int(n)] = (float(i), float(f))
if nit not in rungs:
    print(f"SKIP-WORTHY: the real solve took {nit} sweeps, outside the ladder"); sys.exit(1)
ri, rf = rungs[nit]
di = abs(ri - init) / abs(init)
df = abs(rf - fin) / abs(fin)
print(f"  ORACLE CONTROL: the real Ux solve stopped at sweep {nit}, {fin:.15g};"
      f" ladder rung {nit} is {rf:.15g}  (rel {df:.2e}, initial rel {di:.2e})")
if di > 1e-12 or df > 1e-12:
    print("FAIL: the ladder does not reproduce the solve OpenFOAM actually ran"); sys.exit(1)
PY

"$BIN" "$W/of" 1
rc=$?
# The same binary once more with the per-level launches forced, for the timing line only: the
# assertions above already passed on the single-block walk, and the numbers are identical either way.
[ $rc -eq 0 ] && BRAE_GS_PER_LEVEL=1 "$BIN" "$W/of" 1 | grep "TIMING"
[ $rc -eq 0 ] && echo "PASS: both of OpenFOAM's GaussSeidel smoothers are brae's, sweep for sweep"
exit $rc
