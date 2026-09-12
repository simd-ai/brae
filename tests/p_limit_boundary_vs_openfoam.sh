#!/usr/bin/env bash
# pressureControl::limit and the boundary, against real OpenFOAM -- item 16c.
#
# OpenFOAM clamps p with `p = min(p, pMax_)` on the GeometricField (pressureControl.C:242,253). The
# queue entry read that as "internal AND boundary, a fixedValue patch's stored value included" and
# asked brae to clamp the value-holding patches too. OpenFOAM's source says otherwise: the assignment
# reaches each patch through fvPatchField::operator=(const fvPatchField&), which
# fixedValueFvPatchField.H:204 and mixedFvPatchField.H:305 (and directionMixed, sliced, slip,
# partialSlip, fixedNormalSlip) override as EMPTY -- a value-fixing patch ignores it. Every other
# patch takes the clamped value and is re-evaluated at once, because limit() returns true whenever a
# limit is configured (pressureControl.C:256) and pEqn.H:100-103 then calls
# p.correctBoundaryConditions(). Both brae arms clamp the cells and re-evaluate the boundary keyed on
# the same condition (rhoSimpleFoam.cu:854-875, rhoSimpleFoam_cpp.cu:862-880): OpenFOAM's semantics.
#
#   FIXTURE  validation/rhoBox (1,200 cells, outlet `fixedValue 100000`; after 5 tight iterations its
#            cells span 100000.004 to 100000.566). Its `pMin 1000` is replaced by `pMin 100000.25`, so
#            the clamp BINDS on every iteration in the half of the cells nearest the outlet, while the
#            outlet's own fixed value lies BELOW the limit -- exactly the configuration the queue entry
#            said OpenFOAM "corrects". Every linear solver at 1e-14 / relTol 0 (as
#            mirror_scalar_transport_vs_openfoam does) so the comparison measures the clamp, not the
#            pressure solvers' stopping points. 5 iterations.
#   ORACLE A OpenFOAM's own output: its log reports `pressureControl: p min` (the clamp fired, the
#            CONTROL), its written p has min >= pMin in the cells AND max > pMin (partly clamped, not
#            a degenerate uniform field), and its outlet keeps uniform 100000 -- the fixedValue was NOT
#            raised to pMin.
#   ORACLE B each mirror arm's written p: internal field within BOUND of OpenFOAM's (relative L2),
#            cells min >= pMin, outlet still 100000.
#   FAIL-PROOF, RUN every time: a host run with the fixture's own pMin 1000 (never binds), compared
#            to the same OpenFOAM output, must miss the bound by decades -- the comparison sees the
#            clamp's 0.25 Pa, so it would see a fixedValue raised to pMin.
#   MEASURED (rhoBox, 5 iterations, 2026-09-07): OpenFOAM raised 280 of 1,200 cells to pMin and wrote the
#   outlet at uniform 100000; host 8.660e-13, cuda 5.774e-13 against OpenFOAM (bound 1e-9); the
#   fail-proof arm 1.558e-06.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/rhoBox}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM at $OFBASHRC"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
N=5
PMIN=100000.25
stage() {   # stage <dir> <pMin or "none">
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* 2>/dev/null; [ -d "$1/0" ] || cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "$N" "$2" <<'PY'
import re, sys
d, n, pmin = sys.argv[1], sys.argv[2], sys.argv[3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
s = re.sub(r'functions\s*\{.*?\n\}', '', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'tolerance\s+[^;]*;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[^;]*;', 'relTol 0;', s)
assert re.search(r'\bpMin\s+[^;]*;', s), 'fixture must name pMin (rhoBox names pMin 1000)'
if pmin != 'none':
    s = re.sub(r'\bpMin\s+[^;]*;', 'pMin %s;' % pmin, s, count=1)
open(f, 'w').write(s)
PY
}
stage "$W/of" $PMIN; stage "$W/host" $PMIN; stage "$W/cuda" $PMIN; stage "$W/nolimit" none
( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$W/of" && rhoSimpleFoam > of.log 2>&1 ) || { tail -6 "$W/of/of.log"; echo "FAIL: OpenFOAM did not run"; exit 1; }
grep -q "pressureControl: p min" "$W/of/of.log" && say "CONTROL: OpenFOAM's limiter fired (pressureControl: p min in its log)" ok \
                                                 || say "CONTROL: OpenFOAM's limiter fired (pressureControl: p min in its log)" FAIL
for arm in "1 host" "cuda cuda" "1 nolimit"; do
    set -- $arm; sel=$1; label=$2
    ( cd "$W/$label" && BRAE_RHOSIMPLEFOAM_MIRROR=$sel "$BRAE" -case "$W/$label" > run.log 2>&1 ) || { tail -6 "$W/$label/run.log"; say "$label  the mirror run finished" FAIL; continue; }
    [ -f "$W/$label/$N/p" ] && say "$label  p written at $N" ok || say "$label  p written at $N" FAIL
done
python3 - "$W" "$N" "$PMIN" <<'PY' || fail=1
import re, sys, os, numpy as np
W, N, PMIN = sys.argv[1], sys.argv[2], float(sys.argv[3])
def internal(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', b)
    if not m: return None    # `uniform`: a wholly clamped field, which the control below rejects
    n = int(m.group(1)); start = m.end()
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary': return np.frombuffer(b[start:start+n*8], dtype='<f8')
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0])
    return np.array([float(x) for x in vals[:n]])
def patch(fn, name):
    # the written value(s) of one patch: 'uniform X' or a nonuniform list; None if absent
    t = open(fn, 'rb').read().decode('latin-1')
    bf = t[t.index('boundaryField'):]
    m = re.search(r'\b%s\s*\{([^{}]*)\}' % name, bf, re.S)
    if not m: return None
    body = m.group(1)
    u = re.search(r'value\s+uniform\s+([-+0-9.eE]+)', body)
    if u: return np.array([float(u.group(1))])
    nu = re.search(r'value\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(([^)]*)\)', body, re.S)
    if nu: return np.array([float(x) for x in nu.group(2).split()])
    return None
bad = 0
def say(ok, what):
    global bad
    print('  %-74s %s' % (what, 'ok' if ok else 'FAIL'))
    if not ok: bad = 1
of = internal(W + '/of/%s/p' % N)
if of is None: say(False, 'CONTROL  OpenFOAM\'s p is nonuniform (partly clamped)'); sys.exit(1)
ob = patch(W + '/of/%s/p' % N, 'outlet')
print('  OpenFOAM p at %s: cells min %.6f max %.6f (pMin %.2f), %d of %d cells at pMin, outlet %s'
      % (N, of.min(), of.max(), PMIN, int(np.sum(np.isclose(of, PMIN, rtol=0, atol=1e-9))), of.size, ob))
say(of.min() >= PMIN * (1 - 1e-12) and of.max() > PMIN + 1e-3, 'ORACLE A  OpenFOAM raised the CELLS below pMin to pMin, and only those (partly clamped)')
say(ob is not None and np.allclose(ob, 100000.0, rtol=1e-12), 'ORACLE A  OpenFOAM left the fixedValue outlet at 100000 (NOT raised to pMin)')
BOUND = 1e-9
for label in ('host', 'cuda'):
    fn = W + '/%s/%s/p' % (label, N)
    if not os.path.exists(fn): say(False, label + '  p vs OpenFOAM (no file)'); continue
    b = internal(fn)
    if b is None or b.size != of.size: say(False, label + '  p vs OpenFOAM (wrong size or uniform)'); continue
    e = np.linalg.norm(b - of) / np.linalg.norm(of)
    pb = patch(fn, 'outlet')
    print('  %s  p at %s vs OpenFOAM: rel L2 %.3e   bound %.1e   cells min %.6f   outlet %s' % (label, N, e, BOUND, b.min(), 'none' if pb is None else '%.6g..%.6g' % (pb.min(), pb.max())))
    say(e <= BOUND, label + '  ORACLE B  p tracks OpenFOAM with the clamp binding')
    say(b.min() >= PMIN * (1 - 1e-12), label + '  ORACLE B  cells raised to pMin')
    say(pb is not None and np.allclose(pb, 100000.0, rtol=1e-12), label + '  ORACLE B  fixedValue outlet left at 100000')
fn = W + '/nolimit/%s/p' % N
if os.path.exists(fn):
    b = internal(fn); e = np.linalg.norm(b - of) / np.linalg.norm(of)
    print('  FAIL-PROOF  host with the never-binding pMin 1000 vs OpenFOAM at pMin %.2f: rel L2 %.3e (must exceed %.1e by decades)' % (PMIN, e, BOUND))
    say(e > 100 * BOUND, 'FAIL-PROOF  the comparison sees the clamp (unlimited run misses the bound)')
else:
    say(False, 'FAIL-PROOF  unlimited host run wrote p')
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: OpenFOAM clamps the cells and leaves the fixedValue boundary; both mirror arms do the same"
exit $fail
