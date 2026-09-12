#!/usr/bin/env bash
# The k/epsilon PATCH DIFFUSIVITY, against real OpenFOAM's own residuals.
#
# OpenFOAM fills nut by FIELD ASSIGNMENT -- nut_ = Cmu*sqr(k_)/epsilon_ in kEpsilon::correctNut() -- and a
# field assignment writes the BOUNDARY too, from the boundary k and epsilon. correctBoundaryConditions()
# then leaves a `calculated` patch alone, so such a patch carries Cmu*k_b^2/eps_b and NOT the value of the
# cell behind it. That boundary nut is what DkEff(patchi) = nut_b/sigmak + nu and DepsilonEff(patchi) are
# built from, so getting it wrong perturbs the k and epsilon laplacians on every non-wall patch.
#
# brae interpolated the CELL nut to those faces instead. On pitzDaily the inlet's true nut_b is 8.52e-04
# against a cell value twelve times larger, and the error concentrated there: 90.5% of the entire epsilon
# residual sat on that one patch.
#
# THE ORACLE IS REAL OPENFOAM: validation/pitzDailyTurb/log.simpleFoam is simpleFoam v2412 output, and
# 1576 is the converged state that log ends at. Starting brae from OpenFOAM's OWN converged fields makes
# the initial residual a direct measurement of the discretisation -- both codes assemble the same
# equations at the same state, so the residuals must agree. They now do, to under 1%.
#
# WHAT THIS GATE CATCHES, measured by reverting the fix and rerunning:
#   epsilon   1.90e-07 -> 1.20e-05   (63x OpenFOAM's, against 0.6% agreement)
#   k         3.65e-07 -> 6.44e-06   (17x OpenFOAM's, against 0.7% agreement)
# The 5% bound below is far tighter than that gap and far looser than the 0.7% actually achieved.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
OFLOG="$SRC/log.simpleFoam"
[ -f "$OFLOG" ]    || { echo "SKIP: no OpenFOAM log at $OFLOG"; exit 77; }
[ -d "$SRC/1576" ] || { echo "SKIP: no converged OpenFOAM state at $SRC/1576"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case"
rm -rf "$W/case/log.simpleFoam"

python3 - "$W/case/system/controlDict" <<'PY'
import io, re, sys
p = sys.argv[1]
s = io.open(p).read()
s = re.sub(r'startFrom\s+\w+;',      'startFrom       startTime;', s)
s = re.sub(r'startTime\s+[\d.]+;',   'startTime       1576;',      s)
s = re.sub(r'endTime\s+[\d.]+;',     'endTime         1577;',      s)
s = re.sub(r'writeInterval\s+[\d.]+;', 'writeInterval   1000;',    s)
io.open(p, 'w').write(s)
PY

( cd "$W/case" && BRAE_TURB_RESID=1 BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
    echo "FAIL: brae refused or crashed starting from OpenFOAM's converged state"
    tail -n 8 "$W/case/run.log"; exit 1; }

python3 - "$W/case/run.log" "$OFLOG" <<'PY'
import io, re, sys

brae_log, of_log = sys.argv[1], sys.argv[2]

# brae: the FIRST solve of each field, i.e. the one assembled at OpenFOAM's converged state.
braev = {}
for line in io.open(brae_log):
    m = re.search(r'Solving for (\w+), Initial residual = ([0-9.eE+-]+)', line)
    if m and m.group(1) not in braev:
        braev[m.group(1)] = float(m.group(2))

# OpenFOAM: the same fields in the log's LAST time step, which produced those converged fields.
ofv, seen = {}, False
for line in io.open(of_log):
    if line.startswith('Time = 1576'):
        seen = True
    if not seen:
        continue
    m = re.search(r'Solving for (\w+), Initial residual = ([0-9.eE+-]+)', line)
    if m and m.group(1) not in ofv:
        ofv[m.group(1)] = float(m.group(2))

rc = 0
for f in ('epsilon', 'k'):
    if f not in braev:
        print("FAIL: brae never reported a %s solve (BRAE_TURB_RESID not honoured?)" % f)
        rc = 1
        continue
    if f not in ofv:
        print("SKIP-ish: OpenFOAM log has no %s at 1576" % f)
        continue
    b, o = braev[f], ofv[f]
    rel = abs(b - o) / o
    ok = rel < 0.05
    print("  %-8s brae %.6e   OpenFOAM %.6e   rel %.2f%%   %s"
          % (f, b, o, 100.0 * rel, "ok" if ok else "FAIL (>5%)"))
    if not ok:
        rc = 1

if rc == 0:
    print("  ok:   the k/epsilon patch diffusivity matches OpenFOAM at its own converged state")
sys.exit(rc)
PY
