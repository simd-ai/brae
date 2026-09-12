#!/usr/bin/env bash
# Per-patch wall-function coefficients, against real OpenFOAM, on both rhoSimpleFoam mirror arms -- item 16h-port.
#
# OpenFOAM's wall functions each read Cmu, kappa and E from THEIR OWN patch entry
# (wallFunctionCoefficients.C:73-79, one wallCoeffs_ per patch field: nutkWallFunction's from the nut
# file, epsilonWallFunction's from the epsilon file), defaults 0.09 / 0.41 / 9.8. brae parsed and
# ANNOUNCED them (item 16h) and applied a model-wide value; now every scalar patch field carries its own
# (WallFunctionCoeffs), the host kEpsilon / realizableKE / SA wall treatments read them from the field
# they belong to, and the CUDA kEpsilon takes them per face.
#
#   FIXTURE  validation/rhoKE (3,200 cells, kEpsilon, nutkWallFunction + epsilonWallFunction on hotWall
#            and coldWall), blockMesh at staging, every linear solver at 1e-14 / relTol 0, 5 iterations.
#   ARM BOTH nut walls AND epsilon walls write `Cmu 0.085; kappa 0.4; E 9.0;`.
#   ARM SPLIT nut walls write `kappa 0.4; E 9.0;` only, epsilon walls write `Cmu 0.085;` only -- a port
#            that fed one field's entry to the other's wall function fails here.
#   ORACLE   OpenFOAM's written k, epsilon, nut, U at 5 for each arm; each mirror arm within BOUND (rel L2).
#   CONTROL  OpenFOAM BOTH vs OpenFOAM with no entries (defaults): the mutation moves the fields by
#            decades more than BOUND, so agreement is not vacuous.
#   FAIL-PROOF, RUN when BRAE_PRE16H names a pre-port binary: that binary on ARM BOTH (host) misses the
#            bound -- it announced the entries and ran 0.09 / 0.41 / 9.8.
#   MEASURED (2026-09-07): see the numbers printed; bound set from them below.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/rhoKE}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
PRE="${BRAE_PRE16H:-}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM at $OFBASHRC"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
N=5
stage() {   # stage <dir> <arm: both|split|none>
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* 2>/dev/null; [ -d "$1/0" ] || cp -r "$1/0.orig" "$1/0"
    ( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { tail -3 "$1/log.blockMesh"; echo "FAIL: blockMesh"; exit 1; }
    python3 - "$1" "$N" "$2" <<'PY'
import re, sys
d, n, arm = sys.argv[1], sys.argv[2], sys.argv[3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
s = re.sub(r'functions\s*\{.*?\n\}', '', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'tolerance\s+[^;]*;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[^;]*;', 'relTol 0;', s)
open(f, 'w').write(s)
nutAdd = {'both': ' Cmu 0.085; kappa 0.4; E 9.0;', 'split': ' kappa 0.4; E 9.0;', 'none': ''}[arm]
epsAdd = {'both': ' Cmu 0.085; kappa 0.4; E 9.0;', 'split': ' Cmu 0.085;', 'none': ''}[arm]
for fld, typ, add in (('nut', 'nutkWallFunction', nutAdd), ('epsilon', 'epsilonWallFunction', epsAdd)):
    f = d + '/0/' + fld; s = open(f).read()
    assert re.search(r'\b(Cmu|kappa|E)\s+[-0-9.]', s) is None, fld + ' already writes a coefficient'
    s, k = re.subn(r'(type\s+%s\s*;)' % typ, r'\1' + add, s)
    assert k >= 2, '%s: expected the two wall patches, patched %d' % (fld, k)
    open(f, 'w').write(s)
PY
}
run() {   # run <dir> <of|1|cuda> [binary]
    local d=$1 sel=$2 bin=${3:-$BRAE}
    if [ "$sel" = of ]; then ( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$d" && rhoSimpleFoam > run.log 2>&1 )
    else ( cd "$d" && BRAE_RHOSIMPLEFOAM_MIRROR=$sel "$bin" -case "$d" > run.log 2>&1 ); fi
}
for arm in both split none; do stage "$W/of_$arm" $arm; run "$W/of_$arm" of || { tail -5 "$W/of_$arm/run.log"; echo "FAIL: OpenFOAM $arm"; exit 1; }; done
for arm in both split; do
    for a in "1 host" "cuda cuda"; do set -- $a; stage "$W/${2}_$arm" $arm; run "$W/${2}_$arm" $1 || { tail -5 "$W/${2}_$arm/run.log"; say "$2 $arm  the mirror run finished" FAIL; }; done
done
for arm in both split; do for side in host cuda; do
    grep -q "not honoured per patch" "$W/${side}_$arm/run.log" \
        && say "$side $arm  no 'not honoured per patch' notice (the mirror applies them)" FAIL \
        || say "$side $arm  no 'not honoured per patch' notice (the mirror applies them)" ok
done; done
if [ -n "$PRE" ] && [ -x "$PRE" ]; then
    stage "$W/pre_both" both; run "$W/pre_both" 1 "$PRE" || true
    grep -q "not honoured per patch" "$W/pre_both/run.log" && say "FAIL-PROOF  the pre-port binary announced the entries as not honoured" ok \
                                                             || say "FAIL-PROOF  the pre-port binary announced the entries as not honoured" FAIL
fi
python3 - "$W" "$N" "$PRE" <<'PY' || fail=1
import re, sys, os, numpy as np
W, N, PRE = sys.argv[1], sys.argv[2], sys.argv[3]
def internal(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m: return None
    n = int(m.group(2)); start = m.end(); comps = 3 if m.group(1) == b'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary': return np.frombuffer(b[start:start+n*comps*8], dtype='<f8')
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0])
    return np.array([float(x) for x in vals[:n*comps]])
bad = 0
def say(ok, what):
    global bad
    print('  %-78s %s' % (what, 'ok' if ok else 'FAIL'))
    if not ok: bad = 1
FIELDS = ('k', 'epsilon', 'nut', 'U')
def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
of = {arm: {f: internal('%s/of_%s/%s/%s' % (W, arm, N, f)) for f in FIELDS} for arm in ('both', 'split', 'none')}
ctrl = {f: rel(of['both'][f], of['none'][f]) for f in FIELDS}
print('  CONTROL  OpenFOAM BOTH vs OpenFOAM defaults: ' + '  '.join('%s %.3e' % (f, ctrl[f]) for f in FIELDS))
BOUND = 1e-9
say(min(ctrl.values()) > 1000 * BOUND, 'CONTROL  the mutation moves every field by decades more than the bound')
for arm in ('both', 'split'):
    for side in ('host', 'cuda'):
        worst = 0.0; parts = []
        for f in FIELDS:
            fn = '%s/%s_%s/%s/%s' % (W, side, arm, N, f)
            if not os.path.exists(fn): parts.append('%s missing' % f); worst = float('inf'); continue
            b = internal(fn); e = rel(b, of[arm][f]) if b is not None and b.size == of[arm][f].size else float('inf')
            parts.append('%s %.3e' % (f, e)); worst = max(worst, e)
        print('  %s %-5s vs OpenFOAM at %s: %s' % (side, arm, N, '  '.join(parts)))
        say(worst <= BOUND, '%s %-5s tracks OpenFOAM on k, epsilon, nut, U (bound %.0e)' % (side, arm, BOUND))
if PRE and os.path.exists('%s/pre_both/%s/k' % (W, N)):
    e = max(rel(internal('%s/pre_both/%s/%s' % (W, N, f)), of['both'][f]) for f in FIELDS)
    print('  FAIL-PROOF  pre-port binary (host, BOTH) vs OpenFOAM BOTH: worst %.3e' % e)
    say(e > 100 * BOUND, 'FAIL-PROOF  the pre-port binary misses the bound')
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: per-patch wall-function coefficients run as OpenFOAM runs them, on both mirror arms"
exit $fail
