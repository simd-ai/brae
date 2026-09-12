#!/usr/bin/env bash
# aerofoilNACA0012 (validation/naca0012: kOmegaSST, inletOutlet T/k/omega and freestreamVelocity/Pressure on
# the far field, rho relaxed at 0.01) RESTARTED from OpenFOAM's own written iteration 100, both mirror
# arms against OpenFOAM's own restart, iterations 101-105, every linear solver at 1e-14 relTol 0 -- the
# gasMixing protocol (tests/rho_gasmixing_vs_openfoam.sh) on the fixture whose restart the mirror never
# had a gate for. validation/restart_vs_openfoam.sh restarts the PRE-MIRROR driver (BRAE_RHOSIMPLEFOAM_MIRROR
# unset) at a 1e-03 tolerance and asks a different question (that phi and rho are READ).
#
# WHY. The device keeps no stored patch values, and at a restart an inletOutlet face's first evaluate is
# whatever its construction-time coefficients give. On gasMixing that was the closure's 1.6e-04
# (DeviceBoundary::ioStored, now seeded for the closure fields); whether the SOLVER fields' inletOutlet
# faces (T here) needed the same was an open question this gate answers by measurement: they do not --
# nothing evaluates them before the step's first flux switch, and the seed (BRAE_IO_STORED_SOLVER=1, the
# control) changes no digit of either arm.
#
# MEASURED, worst over U p T k omega nut rho:
#     host  it 101  p 1.8e-09 (U 2.4e-10, T 1.2e-11, k 3.8e-12)   it 105  U 2.2e-10  (all others <= 1.4e-10)
#     CUDA  the same digits; CUDA vs host <= 5.2e-13 at every iteration
#   At the case's OWN tolerances the two arms read 2.5e-05 apart in k -- solver stopping points, not the
#   discretisation, which is why this gate tightens them.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
T0=100; T1=105
BOUND=${BOUND:-1e-08}      # measured: worst 1.8e-09 (p at iteration 101, both arms)

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
[ -f "${FOAM_TUTORIALS:-}/resources/geometry/NACA0012.obj.gz" ] || { echo "SKIP: the NACA0012 geometry is not installed"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
ctl() {   # ctl <case> <start> <end> <writeInterval>
    python3 - "$1/system/controlDict" "$2" "$3" "$4" <<'PY'
import re, sys
p, st, end, wi = sys.argv[1:5]; s = open(p).read()
s = re.sub(r'startFrom\s+\w+;', 'startFrom      startTime;', s); s = re.sub(r'startTime\s+[0-9.eE+-]+;', f'startTime      {st};', s)
s = re.sub(r'endTime\s+[0-9.eE+-]+;', f'endTime        {end};', s); s = re.sub(r'writeControl\s+\w+;', 'writeControl   timeStep;', s)
s = re.sub(r'writeInterval\s+[0-9.eE+-]+;', f'writeInterval  {wi};', s); s = re.sub(r'writeFormat\s+\w+;', 'writeFormat    ascii;', s)
s = re.sub(r'writePrecision\s+\d+;', 'writePrecision 15;', s); s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(p, 'w').write(s)
PY
}
tighten() {
    python3 - "$1/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s); s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s); open(p, 'w').write(s)
PY
}
# 1. the developed state: the tutorial's own meshing, OpenFOAM to T0 with the case's own settings
cp -r "$ROOT/validation/naca0012" "$W/dev" || exit 1
( cd "$W/dev" && mkdir -p constant/geometry && cp -f "$FOAM_TUTORIALS/resources/geometry/NACA0012.obj.gz" constant/geometry/ \
  && mkdir -p 0 && cp 0.orig/* 0/ && blockMesh > log.bm 2>&1 && transformPoints -scale '(1 0 1)' > log.tp 2>&1 \
  && extrudeMesh > log.ex 2>&1 && topoSet > log.ts 2>&1 && rm -f system/fvOptions ) || { echo "SKIP: the tutorial's meshing did not run here"; exit 77; }
ctl "$W/dev" 0 "$T0" "$T0"
( cd "$W/dev" && rhoSimpleFoam > log 2>&1 )
[ -d "$W/dev/$T0" ] || { echo "SKIP: OpenFOAM did not reach iteration $T0"; tail -3 "$W/dev/log"; exit 77; }
grep -q nonuniform "$W/dev/$T0/k" || { echo "FAIL: the restart state is uniform"; exit 1; }
restart() {   # restart <dest>
    rm -rf "$1"; cp -r "$W/dev" "$1"; rm -rf "$1"/log* "$1"/postProcessing
    for t in $(ls "$1" | grep -E '^[0-9]+$'); do [ "$t" = "$T0" ] || rm -rf "$1/$t"; done
    ctl "$1" "$T0" "$T1" 1; tighten "$1"
}
# 2. OpenFOAM's own restart, tightened
restart "$W/of"; ( cd "$W/of" && rhoSimpleFoam > log 2>&1 )
[ -d "$W/of/$T1" ] || { echo "SKIP: OpenFOAM's restart did not reach $T1"; tail -3 "$W/of/log"; exit 77; }
compare() {   # compare <brae> <label>
    ARM="$2" BOUND="$BOUND" T0="$T0" T1="$T1" python3 - "$1" "$W/of" <<'PY'
import math, os, re, sys
b, o = sys.argv[1], sys.argv[2]; arm = os.environ['ARM']; bound = float(os.environ['BOUND']); t0, t1 = int(os.environ['T0']), int(os.environ['T1'])
def rd(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if m: return [float(x) for x in m.group(1).replace('(',' ').replace(')',' ').split()], 'n'
    m = re.search(r'internalField\s+uniform\s+\(?([^;)]+)\)?;', s); return ([float(x) for x in m.group(1).split()], 'u') if m else (None, None)
ok, worst, wn = True, 0.0, ''
for it in range(t0 + 1, t1 + 1):
    for f in ('U', 'p', 'T', 'k', 'omega', 'nut', 'rho'):
        pb, po = f'{b}/{it}/{f}', f'{o}/{it}/{f}'
        if not (os.path.exists(pb) and os.path.exists(po)): print('     it %d %-6s MISSING   FAIL' % (it, f)); ok = False; continue
        a, ka = rd(pb); c, kc = rd(po)
        if kc == 'u': c = c * (len(a) // len(c))
        if ka == 'u': a = a * (len(c) // len(a))
        den = math.sqrt(sum(y*y for y in c))
        if den == 0.0: print('     it %d %-6s DEGENERATE (zero-norm oracle)   FAIL' % (it, f)); ok = False; continue
        e = math.sqrt(sum((x-y)**2 for x, y in zip(a, c))) / den
        if e > worst: worst, wn = e, '%s it %d' % (f, it)
        if e >= bound: print('     it %d %-6s vs OpenFOAM (L2 rel)  %.3e   FAIL (bound %g)' % (it, f, e, bound)); ok = False
print('     %s: worst %s %.3e  (bound %g)   %s' % (arm, wn, worst, bound, 'ok' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PY
}
# 3. both mirror arms from the same state
for M in 1 cuda; do
    restart "$W/br_$M"
    if BRAE_RHOSIMPLEFOAM_MIRROR=$M "$BRAE" -case "$W/br_$M" > "$W/br_$M/log" 2>&1 && [ -d "$W/br_$M/$T1" ]; then
        compare "$W/br_$M" "$([ $M = cuda ] && echo CUDA || echo host)" || fail=1
    else
        echo "     $M: DID NOT RUN   FAIL"; grep -v '^brae NOTICE' "$W/br_$M/log" | tail -3; fail=1
    fi
done
[ $fail = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit $fail
