#!/usr/bin/env bash
# The whole outer iteration, with the one remaining substitution removed: brae against real OpenFOAM on
# T3A with the pressure solve driven to 1e-10 in both.
#
# WHY THIS GATE EXISTS. Queue item 53 reported that under this exact setup brae had declared `SIMPLE
# solution converged` while omega sat at 1.602e+44 and nut had relaminarised -- "convergence" over a
# blown-up field. That was the multicolour Gauss-Seidel sweep (item 32's root cause), and it did NOT
# reproduce once the sweep was OpenFOAM's own: brae and simpleFoam now both converge at ITERATION 418,
# and their fields agree to U 1.2e-07, p 2.0e-07, omega 9.5e-09, phi 9.5e-09.
#
# That number is the point. At the case's own `relTol 0.1` on p, lm_cuda_vs_openfoam lands at U
# 6.7e-05 -- and item 32 established that the outer iteration is OpenFOAM's per step to 2e-14. What
# separates 6.7e-05 from 1.2e-07 is therefore ONE thing: brae runs an AMG-preconditioned PCG where the
# case asks GAMG, and at a loose relTol the two stop at different iterates. Tightening p makes them stop
# at the same place, and everything else that could differ is then measured at 1e-07. This gate is
# what says the pressure solver is the whole of the remaining gap, and nothing else has crept in.
#
#   ARM 1    both codes DECLARE convergence (residualControl semantics: initial residuals of the solved
#            fields, simpleControl::criteriaSatisfied), within two iterations of each other.
#   ARM 2    the converged fields agree, per field, to bounds set from the first green run.
#   ARM 3    omega and nut carry the same extrema -- the specific thing item 53 said had blown up.
#   CONTROL  the case's own `relTol 0.1` on p: the iteration counts must differ MATERIALLY (measured
#            brae 406 against OpenFOAM's 269) and U must be far worse than ARM 2's -- so the tight bounds
#            have resolution, and the loose gap is the pressure solver's stopping point and nothing else.
#
# Fail-proof: item 53's own report is the failing run -- the colour-order sweep under this setup gave
# omega 1.602e+44 at U 1.577e-04, which fails every arm here.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/T3A}"
MAXIT="${2:-3000}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
prep() {   # prep <dir> <tight|loose>
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$1/"
    cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "$2" "$MAXIT" <<'PY'
import re, sys
d, mode, maxit = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^endTime .*',          'endTime         %s;' % maxit, s, flags=re.M)
s = re.sub(r'^writeInterval .*',    'writeInterval   %s;' % maxit, s, flags=re.M)
s = re.sub(r'^stopAt .*',           'stopAt          endTime;', s, flags=re.M)
s = re.sub(r'^writePrecision .*',   'writePrecision  15;', s, flags=re.M)
s = re.sub(r'^writeCompression .*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
if mode == 'tight':
    f = d + '/system/fvSolution'; s = open(f).read()
    old = '        solver          GAMG;\n        tolerance       1e-6;\n        relTol          0.1;'
    assert old in s, 'the p entry is not the one this gate was written against'
    s = s.replace(old, '        solver          GAMG;\n        tolerance       1e-10;\n        relTol          0;')
    open(f, 'w').write(s)
PY
}
prep "$W/of_t" tight;  prep "$W/br_t" tight
prep "$W/of_l" loose;  prep "$W/br_l" loose

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for d in of_t of_l; do ( cd "$W/$d" && simpleFoam > log 2>&1 ) || { echo "FAIL: simpleFoam did not run in $d"; tail -8 "$W/$d/log"; exit 1; }; done
for d in br_t br_l; do ( cd "$W/$d" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/$d" > log 2>&1 ) || { echo "FAIL: brae crashed in $d"; tail -8 "$W/$d/log"; exit 1; }; done

python3 - "$W" <<'PY'
import os, re, sys
import numpy as np
W = sys.argv[1]
fail = 0
def say(msg, ok):
    global fail
    print("  %-76s %s" % (msg, "ok" if ok else "FAIL"))
    if not ok: fail = 1

def converged_at(d):
    log = open(os.path.join(d, 'log'), errors='latin-1').read()
    m = re.search(r'converged in (\d+) iterations', log)
    return int(m.group(1)) if m else None

def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r'[0-9]+(\.[0-9]+)?', x) and x != '0']
    return max(ts, key=float) if ts else None

def read(d, f):
    b = open(os.path.join(d, f), 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m:
        m2 = re.search(rb'internalField\s+uniform\s+\(?([-0-9.eE ]+)\)?\s*;', b)
        return np.array([[float(x) for x in m2.group(1).split()]])
    n = int(m.group(2)); nc = 3 if m.group(1) == b'vector' else 1
    vals = re.findall(r'[-+0-9.eE]+', b[m.end():].decode('latin-1').split(')\n;')[0])
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

def rel(a, b):
    if a.shape[0] == 1: a = np.broadcast_to(a, b.shape)
    if b.shape[0] == 1: b = np.broadcast_to(b, a.shape)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

# ---- ARM 1: both declare convergence, at the same iteration ------------------------------------
n_of, n_br = converged_at(W + '/of_t'), converged_at(W + '/br_t')
print("  tight p:  OpenFOAM converged at %s, brae at %s" % (n_of, n_br))
say("ARM 1    both codes declare convergence", n_of is not None and n_br is not None)
if n_of and n_br:
    say("ARM 1    ...within two iterations of each other (measured: the same, 418)", abs(n_of - n_br) <= 2)

# ---- ARM 2: the converged fields -----------------------------------------------------------------
t_of, t_br = lastTime(W + '/of_t'), lastTime(W + '/br_t')
if not (t_of and t_br):
    say("ARM 2    both wrote their converged state", False); sys.exit(1)
d_of, d_br = os.path.join(W, 'of_t', t_of), os.path.join(W, 'br_t', t_br)
# Set from the first green run (U 1.203e-07, p 2.031e-07, k 3.390e-06, omega 9.497e-09,
# nut 5.606e-06, ReThetat 2.761e-06, gammaInt 2.035e-07, phi 9.529e-09), with ~3x headroom for the
# device reductions' run-to-run order. These tighten; they do not loosen.
BOUND = {'U': 5e-07, 'p': 8e-07, 'k': 1e-05, 'omega': 5e-08, 'nut': 2e-05,
         'ReThetat': 1e-05, 'gammaInt': 8e-07, 'phi': 5e-08}
err = {}
for f, tol in BOUND.items():
    try:
        err[f] = rel(read(d_br, f), read(d_of, f))
    except Exception as e:
        say("ARM 2    %-9s readable in both" % f, False); continue
    say("ARM 2    %-9s brae vs OpenFOAM %.3e   (bound %.0e)" % (f, err[f], tol), err[f] < tol)

# ---- ARM 3: the extrema item 53 said had blown up --------------------------------------------------
for f in ('omega', 'nut'):
    a, b = read(d_br, f), read(d_of, f)
    dmax = abs(a.max() - b.max()) / abs(b.max()); dmin = abs(a.min() - b.min()) / max(abs(b.min()), 1e-300)
    print("  %-6s max  OpenFOAM %.4e  brae %.4e   min  OpenFOAM %.4e  brae %.4e" % (f, b.max(), a.max(), b.min(), a.min()))
    say("ARM 3    %s extrema are OpenFOAM's to 1e-3 (no blow-up, no relaminarisation)" % f, dmax < 1e-3 and dmin < 1e-3)

# ---- CONTROL: the case's own loose p ---------------------------------------------------------------
n_ofl, n_brl = converged_at(W + '/of_l'), converged_at(W + '/br_l')
print("  CONTROL  loose p (the case's relTol 0.1): OpenFOAM converged at %s, brae at %s" % (n_ofl, n_brl))
say("CONTROL  both converge at the case's own settings too", n_ofl is not None and n_brl is not None)
if n_ofl and n_brl:
    say("CONTROL  ...but at MATERIALLY different iteration counts (>= 20 apart), so ARM 1 has resolution",
        abs(n_ofl - n_brl) >= 20)
    t1, t2 = lastTime(W + '/of_l'), lastTime(W + '/br_l')
    e_loose = rel(read(os.path.join(W, 'br_l', t2), 'U'), read(os.path.join(W, 'of_l', t1), 'U'))
    ratio = e_loose / max(err.get('U', 1e-30), 1e-30)
    print("  CONTROL  U at loose p %.3e   at tight p %.3e   -> %.0fx" % (e_loose, err.get('U', float('nan')), ratio))
    say("CONTROL  loose p is >= 50x worse on U: the pressure solver's stopping point is the whole gap", ratio >= 50)

sys.exit(fail)
PY
rc=$?
[ $rc -eq 0 ] && echo "PASS: with the pressure solve tight, brae IS OpenFOAM's outer iteration to 1e-07, converging at the same step"
exit $rc
