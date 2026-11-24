#!/usr/bin/env bash
# rhoSimpleFoam AS A CUSTOMER RUNS IT: `brae -case <dir>` with no BRAE_* variable, which the front door hands to
# brae_rhoSimpleFoam's default driver (the pre-mirror one: no BRAE_RHOSIMPLEFOAM_MIRROR), against OpenFOAM v2412.
#
# THE CASE AND THE BOUND are validation/rho_vs_openfoam.sh's, reused, not set here (that gate runs
# brae_rhoSimpleFoam directly; this one goes through `brae`): validation/rhoBox, a heated duct with a 700 K and a
# 300 K wall, meshed by OpenFOAM's blockMesh and shared by both codes, each run to its own steady state (OpenFOAM on
# its residualControl, brae to its endTime), T and p compared over the cells.
#   ARM      relative L2 of T and of p at most 1e-5 (validation/rho_vs_openfoam.sh:20, "both fields land at ~2e-7"),
#            with that gate's guard against a flat field (a spread above 1e-6 of the mean)
#   ARM      the front door handed the case to brae_rhoSimpleFoam (its dispatch line); with no
#            BRAE_RHOSIMPLEFOAM_MIRROR in the environment that runs the default driver (gpuRhoSimpleFoam.cu:103)
#
# Run through tests/default_path_run.sh: brae runs with no BRAE_* variable.
set -u
BRAE="${1:?brae binary (the front door)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${2:-$ROOT/validation/rhoBox}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TOL=1e-5   # validation/rho_vs_openfoam.sh:20

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
BRAE="$(cd "$(dirname "$BRAE")" && pwd)/$(basename "$BRAE")"
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
set +u
set --
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# OpenFOAM: the case as validation/rho_vs_openfoam.sh stages it
cp -r "$SRC" "$W/of"
rm -rf "$W/of"/[1-9]* "$W/of"/log.*
mkdir -p "$W/of/0" && cp "$W/of"/0.orig/* "$W/of/0/"
( cd "$W/of" && blockMesh > log.blockMesh 2>&1 && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ) < /dev/null \
    || { echo "FAIL: OpenFOAM did not run"; tail -5 "$W/of/log.rhoSimpleFoam"; exit 1; }
OFLAST=$(cd "$W/of" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)

# brae: a clean copy sharing only OpenFOAM's mesh, as that gate does
cp -r "$SRC" "$W/brae"
rm -rf "$W/brae"/[1-9]* "$W/brae"/log.*
cp -r "$W/of/constant/polyMesh" "$W/brae/constant/"
mkdir -p "$W/brae/0" && cp "$W/brae"/0.orig/* "$W/brae/0/"
( cd "$W/brae" && "$BRAE" -case "$W/brae" > log.brae 2>&1 ) < /dev/null \
    || { echo "FAIL: brae -case did not run"; tail -8 "$W/brae/log.brae"; exit 1; }
BRLAST=$(cd "$W/brae" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)

grep -q "application rhoSimpleFoam -> rhoSimpleFoam (brae_rhoSimpleFoam)" "$W/brae/log.brae" \
    && say "brae handed the case to brae_rhoSimpleFoam, no mirror selected" ok \
    || say "brae handed the case to brae_rhoSimpleFoam, no mirror selected" FAIL

python3 - "$W/of/$OFLAST" "$W/brae/$BRLAST" "$TOL" <<'PY' || fail=1
import math, re, sys
ofd, brd, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def rd(p):
    t = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\(([^)]*)\)', t, re.S)
    return [float(x) for x in m.group(1).split()] if m else None
bad = 0
for f in ('T', 'p'):
    of, br = rd('%s/%s' % (ofd, f)), rd('%s/%s' % (brd, f))
    if of is None or br is None or len(of) != len(br):
        print('  %-78s FAIL' % ('%s: missing, uniform or a different cell count' % f))
        bad += 1
        continue
    l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(of, br)) / sum(a * a for a in of))
    spread = (max(of) - min(of)) / (abs(sum(of) / len(of)) + 1e-300)
    ok = l2 <= tol and spread > 1e-6
    print('  %-78s %s' % ('%s: L2rel %.4e <= %.0e (spread %.3e)' % (f, l2, tol, spread), 'ok' if ok else 'FAIL'))
    bad += 0 if ok else 1
sys.exit(1 if bad else 0)
PY
echo "  OpenFOAM stopped at $OFLAST, brae at $BRLAST"
[ $fail -eq 0 ] && echo "PASS: rhoSimpleFoam through brae -case is within rho_vs_openfoam's bound of OpenFOAM"
exit $fail
