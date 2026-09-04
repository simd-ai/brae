#!/usr/bin/env bash
# turbulence->validate() -- the eddy viscosity OpenFOAM enters its FIRST momentum assembly with.
#
# simpleFoam.C:92 calls turbulence->validate() before the SIMPLE loop, and eddyViscosity::validate()
# (eddyViscosity.C:119-122) is correctNut(). So OpenFOAM's first momentum equation carries nut rebuilt
# from the initial k and omega, NOT whatever 0/nut holds -- and every tutorial ships `nut uniform 0`.
#
# THE ORACLE is tools/dumpSimpleFoam: OpenFOAM's own simpleFoam carrying a stage harness that writes,
# at SIMPLE iteration BRAE_DUMP_STAGE_ITER, stage_nuEff (turbulence->nuEff() as the assembly reads it),
# stage_UDiag0 (the momentum diagonal before relax) and stage_rAU. brae writes the same names as plain
# columns under BRAE_STAGE_DUMP_DIR, from the same polyMesh, so the cell ordering is OpenFOAM's.
#
# THE CONTROL is the case file's own nut, which is `uniform 0`: nuEff would then be the laminar nu
# everywhere. The gate requires that field to DISAGREE with OpenFOAM's by more than half, because that
# is exactly what brae did before this was wired -- a laminar first iteration under a turbulent name --
# and a bound that both forms passed would gate nothing. On T3A correctNut gives nut = 1.8e-04 against
# nu = 1.5e-05, so the momentum equation is thirteen times less viscous when validate() is skipped:
# measured nuEff 9.2e-01, the momentum diagonal 3.6e-01 and rAU 3.8e-01 away from OpenFOAM.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/T3A}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
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
prep() {
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$1/"
    cp -r "$1/0.orig" "$1/0"
    python3 - "$1" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^endTime .*',        'endTime         1;',  s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   1;',  s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;', s, flags=re.M)
s = re.sub(r'^writeCompression .*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PY
}
prep "$W/of"; prep "$W/br"; mkdir -p "$W/br/dump"

( cd "$W/of" && BRAE_DUMP_STAGE_ITER=1 "$DUMP" > dump.log 2>&1 ) \
    || { echo "FAIL: dumpSimpleFoam did not run"; tail -20 "$W/of/dump.log"; exit 1; }
for f in stage_nuEff stage_UDiag0 stage_rAU; do
    [ -f "$W/of/1/$f" ] || { echo "FAIL: dumpSimpleFoam wrote no 1/$f"; tail -20 "$W/of/dump.log"; exit 1; }
done

( cd "$W/br" && BRAE_SIMPLEFOAM_V2=1 BRAE_STAGE_DUMP_DIR="$W/br/dump" BRAE_STAGE_DUMP_ITER=1 \
    "$BRAE" "$W/br" > run.log 2>&1 ) \
    || { echo "FAIL: brae crashed"; tail -20 "$W/br/run.log"; exit 1; }
for f in stage_nuEff stage_UDiag0 stage_rAU; do
    [ -f "$W/br/dump/$f" ] || { echo "FAIL: brae wrote no stage dump $f"; tail -20 "$W/br/run.log"; exit 1; }
done

python3 - "$W/of/1" "$W/br/dump" "$W/br/0/nut" "$SRC/constant/transportProperties" <<'PY'
import re, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
ofdir, brdir, nutfile, transport = sys.argv[1:5]

def of_internal(path):
    s = open(path).read()
    tail = s[s.index('internalField'):]
    m = re.match(r'internalField\s+uniform\s+([^;]+);', tail)
    if m:
        return np.array([float(m.group(1).strip())])
    m = re.match(r'internalField\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\(', tail)
    n = int(m.group(1)); body = tail[m.end():]; body = body[:body.index('\n)')]
    a = np.fromstring(body.replace('\n', ' '), sep=' ')
    assert a.size == n, (path, a.size, n)
    return a

def col(name):
    a = np.loadtxt(brdir + '/' + name)
    return a[:, 0] if a.ndim == 2 else a

def rel(o, b):
    if o.size == 1: o = np.repeat(o, b.size)
    return float(np.linalg.norm(o - b) / max(np.linalg.norm(o), 1e-300))

nu = float(re.search(r'^nu\s+([0-9eE.+-]+)\s*;', open(transport).read(), re.M).group(1))
fail = 0

# 1. THE GATE: brae's nuEff at iteration 1 IS what validate() gives OpenFOAM.
bounds = {'stage_nuEff': 1e-12, 'stage_UDiag0': 1e-10, 'stage_rAU': 1e-10}
meas = {}
for f, tol in bounds.items():
    o, b = of_internal(ofdir + '/' + f), col(f)
    meas[f] = rel(o, b)
    ok = meas[f] < tol
    fail |= (not ok)
    print(f"  {f:14s} brae vs OpenFOAM  {meas[f]:.3e}  (bound {tol:.0e})  {'ok' if ok else 'FAIL'}")

# 2. THE CONTROL: the case file's nut, which is what a driver that skips validate() would use.
#    nuEff would be nu + nut_file; the gate has to be able to tell those apart.
nutf = of_internal(nutfile)
nuEffFile = nu + (np.repeat(nutf, col('stage_nuEff').size) if nutf.size == 1 else nutf)
ctrl = rel(of_internal(ofdir + '/stage_nuEff'), nuEffFile)
print(f"  CONTROL: nuEff from the case's own 0/nut is {ctrl:.3e} away from OpenFOAM's")
if ctrl < 0.5:
    print("  FAIL: the control is not separated -- this fixture's 0/nut already equals correctNut, so a "
          "driver that skipped validate() would pass this gate")
    fail = 1

sys.exit(1 if fail else 0)
PY
rc=$?
[ $rc -eq 0 ] && echo "PASS: turbulence->validate() reproduces OpenFOAM's startup eddy viscosity"
exit $rc
