#!/usr/bin/env bash
# brae's DEVICE alpha step against brae's HOST solver, on damBreak's own mesh and case.
#
# WHY THE CASE AND NOT A FIXTURE. Every device gate before this ran on a synthetic box: a rotating blob
# on uniform cells, built so each arm could discriminate. Those cannot bring what a real case brings --
# damBreak's mesh is graded, its atmosphere patch is inletOutlet, its frontAndBack are EMPTY (which
# MULES must skip and no box fixture here had), and its fvSolution asks for MULESCorr with
# nLimiterIter 5 and nAlphaCorr 2. This gate runs those.
#
# THE CHAIN. tests/interfoam_dambreak_vs_openfoam.sh holds the HOST solver against real OpenFOAM at
# alpha 3.4346e-09; this holds the DEVICE against that host. Two links, each measured. Comparing the
# device straight to OpenFOAM would fold both differences into one number and make neither readable.
#
# No OpenFOAM RUN is needed here -- only its mesh generator, because damBreak ships 0.orig and no mesh.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_device_inter_dambreak_alpha"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
STEPS=${STEPS:-5}
DT=${DT:-1e-4}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

cp -r "$W/case/0.orig" "$W/case/0"
( cd "$W/case" && blockMesh > log.blockMesh 2>&1 ) || { echo "SKIP: blockMesh failed"; exit 77; }

# A FIXED time step, as interfoam_dambreak_vs_openfoam.sh does and for the same reason. damBreak's own
# controlDict says `adjustTimeStep yes`, so the HOST solver -- which reads it -- grows dt from the
# Courant number while the device loop in the test takes the fixed dt it is given. The two then sit at
# DIFFERENT PHYSICAL TIMES and every field disagrees by an amount that looks like a discretisation
# error and is actually a clock. Measured before this was here: alpha 9.57e-01 out of a field whose
# range is 1, on a device run that had advanced alpha by 1.85e-02 and a host run that had advanced it
# by 9.56e-01.
DT="$DT" BRAE_NCORR="${BRAE_NCORR:-}" python3 - "$W/case" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
p = os.path.join(d, 'system/controlDict')
s = open(p).read()
for k, v in (('adjustTimeStep', 'no'), ('deltaT', os.environ['DT']), ('writeControl', 'runTime')):
    s = re.sub(r'^%s\s+.*' % k, '%-16s %s;' % (k, v), s, flags=re.M) \
        if re.search(r'^%s\s+' % k, s, re.M) else s + '\n%-16s %s;\n' % (k, v)
open(p, 'w').write(s)

# ...AND A TIGHT PRESSURE SOLVE, for the same reason as the fixed time step. damBreak asks for
# `tolerance 1e-07; relTol 0.05` on p_rgh -- the host stops when the residual has fallen to FIVE PER
# CENT of its initial value, while the device loop in the test solves to 1e-12. The comparison then
# measures the two solvers' stopping points and not the discretisation at all: measured before this
# was here, p_rgh was 6.35e+01 of 2.85e+03 (2.2%) and U 23%, spread over the whole field rather than
# any one patch, with phiHbyA -- the pressure equation's entire input -- exact to 2.2e-19.
q = os.path.join(d, 'system/fvSolution')
t = open(q).read()
t = re.sub(r'(p_rgh\w*\s*\{[^}]*?tolerance\s+)[^;]+;', r'\g<1>1e-12;', t)
t = re.sub(r'(p_rgh\w*\s*\{[^}]*?relTol\s+)[^;]+;', r'\g<1>0;', t)
# BISECT: BRAE_NCORR forces the PIMPLE corrector count on BOTH sides, so a difference that only
# appears from the second corrector onward -- a field one side refreshes between passes and the other
# does not -- separates from one present in the first.
if os.environ.get('BRAE_NCORR'):
    t = re.sub(r'(nCorrectors\s+)\d+;', r'\g<1>' + os.environ['BRAE_NCORR'] + ';', t)
open(q, 'w').write(t)
PYEOF
if command -v setFields > /dev/null 2>&1; then
    ( cd "$W/case" && setFields > log.setFields 2>&1 ) || { echo "SKIP: setFields failed"; exit 77; }
fi

# BRAE_PTOL: brae's host driver hardcodes the p_rgh solve to 1e-9 (inter_driver_cpp.cu) rather than
# reading the case's solvers/p_rgh entry, and the device loop in the test solves to 1e-12. Two
# different stopping points make the comparison measure the solvers and not the discretisation, so
# both are pinned here. damBreak's own entry is `tolerance 1e-07; relTol 0.05` -- five per cent of the
# initial residual -- which is looser still.
BRAE_NCORR=${BRAE_NCORR:-} BRAE_PTOL=${BRAE_PTOL:-1e-12} "$BIN" "$W/case" "$W/case/0" "$STEPS" "$DT"
