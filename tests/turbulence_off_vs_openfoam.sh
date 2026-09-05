#!/usr/bin/env bash
# OF's RAS `turbulence` switch, which is NOT `simulationType laminar`.
#
# RASModel.C:70 reads it (`getOrDefault<Switch>("turbulence", true)`) and every model's correct() opens
# with `if (!this->turbulence_) return;` -- kEpsilon.C:216, kOmegaSSTBase.C:502, kOmegaSSTLM.C:602,
# SpalartAllmarasBase.C:442. The model still EXISTS: k, epsilon|omega|nuTilda and the transition scalars
# keep the values they were constructed with, and nut is whatever validate() made of them, because
# simpleFoam.C:92 calls validate() BEFORE the loop and eddyViscosity::validate() is correctNut(). The
# momentum equation then runs on that frozen eddy viscosity for the whole run. A laminar run is a
# different thing entirely -- it has no nut at all.
#
# WHY THIS GATE EXISTS. brae READ the switch (turbulence_setup.cuh) and ANNOUNCED "the model is FROZEN",
# and the only consumer of that flag was the legacy driver. The V2 driver corrected the turbulence
# anyway, so the notice asserted the opposite of what ran -- a capability claimed in a log and absent
# from the code, which is the defect class this repo has paid for more than once.
#
#   ORACLE     real simpleFoam v2412 on the same case with `turbulence off`.
#   GATE       brae's k and omega must be EXACTLY their initial values (the equations were never solved),
#              and U, p, nut must match OpenFOAM's frozen-model answer.
#   CONTROL /
#   FAIL-PROOF the same case with `turbulence on`. Removing the guard makes brae's `off` run behave like
#              its `on` run, so requiring `on` to be FAR from OpenFOAM's `off` answer is exactly the
#              measurement that fails when the guard is gone. The two coincide here, deliberately.
#
# Fail-proof, verified 2026-09-05: with `if (!turbulenceOn) return;` disabled in simpleFoamV2.cu, the
# `off` run converged at iteration 406 -- the SAME iteration as the `on` run -- k moved 8.941e-01 from
# its initial value, omega 1.181e+03, nut/U/p read 5.9e-01 / 2.4e-01 / 4.4e-01 against their 1e-14 /
# 1.5e-04 / 4e-04 bounds, and both control ratios collapsed to exactly 1.0x. Every leg went red.
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
prep() {   # prep <dir> <on|off>
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$1/"
    cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "$2" "$MAXIT" <<'PY'
import re, sys
d, sw, iters = sys.argv[1], sys.argv[2], sys.argv[3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^endTime .*',          'endTime         %s;' % iters, s, flags=re.M)
s = re.sub(r'^writeInterval .*',    'writeInterval   %s;' % iters, s, flags=re.M)
s = re.sub(r'^stopAt .*',           'stopAt          endTime;', s, flags=re.M)
s = re.sub(r'^writePrecision .*',   'writePrecision  15;',   s, flags=re.M)
s = re.sub(r'^writeCompression .*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
t = d + '/constant/turbulenceProperties'; s = open(t).read()
s2 = re.sub(r'turbulence\s+\w+;', 'turbulence      %s;' % sw, s)
assert s2 != s or ('turbulence      %s;' % sw) in s, 'turbulence switch not found in ' + t
open(t, 'w').write(s2)
PY
}
prep "$W/of_off" off
prep "$W/br_off" off
prep "$W/br_on"  on

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
( cd "$W/of_off" && simpleFoam > log 2>&1 ) || { echo "FAIL: simpleFoam did not run"; tail -15 "$W/of_off/log"; exit 1; }
( cd "$W/br_off" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/br_off" > log 2>&1 ) \
    || { echo "FAIL: brae crashed with turbulence off"; tail -15 "$W/br_off/log"; exit 1; }
( cd "$W/br_on"  && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/br_on"  > log 2>&1 ) \
    || { echo "FAIL: brae crashed with turbulence on"; tail -15 "$W/br_on/log"; exit 1; }

grep -q "the model is FROZEN" "$W/br_off/log" \
    || { echo "FAIL: brae did not announce the frozen model"; exit 1; }
grep -q "the model is FROZEN" "$W/br_on/log" \
    && { echo "FAIL: brae announced a frozen model with turbulence ON"; exit 1; }

# Both codes run to the case's OWN residualControl, and each is read at ITS OWN last time. Comparing at a
# matched iteration would compare TRAJECTORIES -- brae's pressure solve is an AMG-preconditioned PCG where
# the case asks GAMG, so the two take different numbers of outer iterations to the same fixed point, and a
# fixed-iteration read of p was 1.36e-02 out while the converged one is three orders of magnitude closer.
for d in of_off br_off br_on; do
    grep -qE "SIMPLE solution converged|solution converged" "$W/$d/log" \
        || { echo "FAIL: $d did not converge inside $MAXIT iterations"; tail -3 "$W/$d/log"; exit 1; }
done

python3 - "$W" "$SRC" <<'PY'
import os, re, sys
import numpy as np
W, src = sys.argv[1], sys.argv[2]

def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r'[0-9]+(\.[0-9]+)?', x) and x != '0']
    return max(ts, key=float) if ts else None

def read(d, f):
    p = os.path.join(d, f)
    if not os.path.exists(p): return None
    b = open(p, 'rb').read()
    m = re.search(rb'internalField\s+uniform\s+\(?([-0-9.eE ]+)\)?\s*;', b)
    if m:
        v = [float(x) for x in m.group(1).split()]
        return np.array([v])
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m: return None
    typ = m.group(1).decode(); n = int(m.group(2)); nc = 3 if typ == 'vector' else 1
    txt = b[m.end():].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

def rel(a, b):
    if a.shape != b.shape:
        a = np.broadcast_to(a, b.shape) if a.shape[0] == 1 else a
        b = np.broadcast_to(b, a.shape) if b.shape[0] == 1 else b
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

paths = []
for c in ('of_off', 'br_off', 'br_on'):
    base = os.path.join(W, c)
    t = lastTime(base)
    if t is None: print("  FAIL: %s wrote no result" % c); sys.exit(1)
    print("  %-7s converged at iteration %s" % (c, t))
    paths.append(os.path.join(base, t))
ofd, brd, brn = paths

fail = 0
# 1. THE EQUATIONS WERE NEVER SOLVED. k and omega must still BE their initial values, in both codes.
for f in ('k', 'omega'):
    init = read(os.path.join(src, '0.orig'), f)
    o, b = read(ofd, f), read(brd, f)
    if o is None or b is None or init is None: print("  FAIL: cannot read %s" % f); fail = 1; continue
    do, db = rel(o, np.broadcast_to(init, o.shape)), rel(b, np.broadcast_to(init, b.shape))
    print("  %-6s frozen at its initial value:  OpenFOAM %.3e   brae %.3e" % (f, do, db))
    if do > 1e-14:
        print("  FAIL: OpenFOAM itself moved %s -- the premise of this gate is wrong" % f); fail = 1
    if db > 1e-14:
        print("  FAIL: brae solved %s with turbulence off" % f); fail = 1

# 2. ...and the fields the frozen model still drives must be OpenFOAM's.
# Set from the first green run, which reproduces bit-for-bit across repeats here:
# nut 2.711e-15, U 1.113e-04, p 3.041e-04. nut is at the round-off floor because it is
# validate()'s one-off correctNut of two fields both codes left untouched. U and p are not:
# with the model frozen this is still a full SIMPLEC solve, and brae runs an AMG-preconditioned
# PCG where the case asks GAMG, which is why the two converge at 506 and 364 iterations.
BOUND = {'nut': 1e-14, 'U': 1.5e-04, 'p': 4e-04}
err = {}
for f in ('nut', 'U', 'p'):
    o, b = read(ofd, f), read(brd, f)
    if o is None or b is None: print("  FAIL: cannot read %s" % f); fail = 1; continue
    err[f] = rel(b, o)
    ok = err[f] < BOUND[f]
    print("  %-6s brae vs OpenFOAM (both frozen)  %.3e   bound %.1e   %s" % (f, err[f], BOUND[f], "ok" if ok else "FAIL"))
    if not ok: fail = 1

# 3. CONTROL and FAIL-PROOF in one: with the guard gone, `off` would run like `on`.
print("  CONTROL: the same case with turbulence ON, against OpenFOAM's frozen answer:")
for f in ('U', 'nut'):
    o, n = read(ofd, f), read(brn, f)
    e = rel(n, o)
    ratio = e / max(err.get(f, 1e-30), 1e-30)
    ok = ratio >= 20.0
    print("    %-6s ON %.3e   %8.1fx the frozen error   %s" % (f, e, ratio, "ok" if ok else "FAIL (want >=20x)"))
    if not ok: fail = 1

sys.exit(1 if fail else 0)
PY
rc=$?
[ $rc -eq 0 ] && echo "PASS: brae freezes the model on \`turbulence off\` exactly as OpenFOAM does"
exit $rc
