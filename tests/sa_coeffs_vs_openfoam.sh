#!/usr/bin/env bash
# A case's SpalartAllmarasCoeffs reach brae's model, and the trajectory says so -- against real OpenFOAM.
#
# turbulence_setup.cuh read only the DES extras (CDES, Cd1..); the base coefficients kept OpenFOAM's
# defaults whatever the case wrote, with a comment saying so (queue item 16d). OpenFOAM reads all of them
# from the model's coeffDict (SpalartAllmarasBase.C:205-312).
#
#   FIXTURE  validation/airFoil2D at its committed converged 500 (real OpenFOAM's).
#   MUTATION `SpalartAllmarasCoeffs { Cb1 0.16; sigmaNut 0.8; }` (defaults 0.1355, 2/3), then 5
#            iterations from 500 in BOTH codes. From a converged state the pressure solvers stop at once,
#            so what moves nuTilda over those 5 iterations is the coefficient change -- which is exactly
#            what this gate is about, and why it compares at a matched iteration count.
#   ARM      brae's nuTilda at 505 against OpenFOAM's at 505, relative L2 over the field.
#   CONTROL  the mutated OpenFOAM run must have MOVED nuTilda from 500 by clearly more than the
#            unmutated one did (else the 5 iterations would be blind to the coefficient and the arm
#            would pass on nothing).
#   FAIL-PROOF, RUN with the coefficient read removed: see the numbers recorded in REFUSALS.md 16d.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/airFoil2D}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC/500" ] || { echo "SKIP: no converged 500 in $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM at $OFBASHRC"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
stage() {   # stage <dir> <mutate:0|1>
    rm -rf "$1"; mkdir -p "$1"; cp -r "$SRC/constant" "$SRC/system" "$SRC/500" "$1/"
    python3 - "$1" "$2" <<'PY'
import re, sys
d, mut = sys.argv[1], sys.argv[2] == '1'
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom startTime;', s)
s = re.sub(r'\bstartTime\s+[^;]*;', 'startTime 500;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 505;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 5;', s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
if mut:
    t = d + '/constant/turbulenceProperties'; s = open(t).read()
    m = re.search(r'RAS\s*\{', s); assert m
    s = s[:m.end()] + '\n    SpalartAllmarasCoeffs { Cb1 0.16; sigmaNut 0.8; }' + s[m.end():]
    open(t, 'w').write(s)
PY
}
stage "$W/of_mut" 1; stage "$W/of_ref" 0; stage "$W/brae_mut" 1
( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$W/of_mut" && simpleFoam > of.log 2>&1 ) || { tail -5 "$W/of_mut/of.log"; echo "FAIL: OpenFOAM (mutated) did not run"; exit 1; }
( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$W/of_ref" && simpleFoam > of.log 2>&1 ) || { tail -5 "$W/of_ref/of.log"; echo "FAIL: OpenFOAM (reference) did not run"; exit 1; }
( cd "$W/brae_mut" && "$BRAE" "$W/brae_mut" > brae.log 2>&1 ) || { tail -8 "$W/brae_mut/brae.log"; echo "FAIL: brae did not run"; exit 1; }
grep -q "SpalartAllmaras (coeffDict)" "$W/brae_mut/brae.log" && say "brae announces the coefficients came from the coeffDict" ok \
                                                             || say "brae announces the coefficients came from the coeffDict" FAIL
python3 - "$W" "$SRC" <<'PY' || fail=1
import re, sys, numpy as np
W, SRC = sys.argv[1:3]
def read(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', b)
    n = int(m.group(1)); start = m.end()
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary':
        return np.frombuffer(b[start:start+n*8], dtype='<f8')
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n]])
base   = read(SRC + '/500/nuTilda')
of_mut = read(W + '/of_mut/505/nuTilda'); of_ref = read(W + '/of_ref/505/nuTilda'); br_mut = read(W + '/brae_mut/505/nuTilda')
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
moved_mut, moved_ref = rel(of_mut, base), rel(of_ref, base)
e = rel(br_mut, of_mut)
bad = 0
def say(ok, what):
    global bad
    print('  %-74s %s' % (what, 'ok' if ok else 'FAIL'))
    if not ok: bad = 1
print('  OpenFOAM nuTilda moved from 500 over 5 iterations: mutated %.3e, unmutated %.3e' % (moved_mut, moved_ref))
say(moved_mut >= 10 * moved_ref, 'CONTROL: the mutation moves nuTilda >= 10x what the unmutated 5 iterations do')
BOUND = 3e-02
print('  brae vs OpenFOAM, nuTilda at 505 (both mutated): rel L2 %.3e   bound %.1e' % (e, BOUND))
say(e <= BOUND, 'brae tracks OpenFOAM under the mutated coefficients')
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: the case's SpalartAllmarasCoeffs reach the model"
exit $fail
