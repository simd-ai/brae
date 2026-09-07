#!/usr/bin/env bash
# scalarTransport on both rhoSimpleFoam mirror arms, against real OpenFOAM -- item 15c.
#
# OpenFOAM's functionObjects::scalarTransport on a mass flux solves, every iteration,
#     fvm::div(phi, s) - fvm::laplacian(D, s) == 0        (steady; D the constant the case writes)
# and writes s beside the solved fields. The mirror used to report the entry as skipped.
#
#   FIXTURE  validation/rhoBox with a tracer added at run time: 0/tracer (1 at the inlet, zeroGradient
#            elsewhere), div(phi,tracer) and laplacian(Dtracer,tracer) schemes, a solvers/tracer entry,
#            and the functionObject. EVERY linear solver is tightened to 1e-14 / relTol 0, so both codes
#            converge each solve and the flux the tracer rides on agrees to round-off -- otherwise the
#            comparison would measure the pressure solvers' different stopping points (item 61).
#   ORACLE   OpenFOAM rhoSimpleFoam on the same case, 5 iterations, its written 5/tracer.
#   ARMS     host mirror and CUDA mirror, 5 iterations, their written 5/tracer: relative L2 <= BOUND.
#   CONTROLS OpenFOAM's tracer must be a REAL field (spread > 0.5, std > 0.05): the first version of
#            this gate held every non-inlet patch zeroGradient, the steady passive scalar was then
#            exactly 1 in every cell, and both arms "agreed" with OpenFOAM to 0.000e+00 on a field that
#            discriminates nothing. A wall held at 0 makes it a convection-diffusion balance. Each arm
#            must also have written the file at all.
#   MEASURED (rhoBox, 5 iterations, sink on hotWall): host 2.019e-12, cuda 2.716e-12, max|diff| 7e-12.
#   FAIL-PROOF, RUN on the pre-port binary: neither arm solved or wrote the tracer (both count lines
#   and both comparisons FAIL, exit 1); OpenFOAM's tracer was a real field.
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
stage() {   # stage <dir>
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* 2>/dev/null; [ -d "$1/0" ] || cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "$N" <<'PY'
import re, sys
d, n = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
s = re.sub(r'functions\s*\{.*?\n\}', '', s, flags=re.S)
s = s.rstrip() + '\nfunctions\n{\n    tracer\n    {\n        type scalarTransport;\n        libs (solverFunctionObjects);\n        field tracer;\n        D 0.01;\n    }\n}\n'
open(c, 'w').write(s)
# the tracer field: 1 at the inlet, 0 on the FIRST wall (a sink), zeroGradient elsewhere, empty where
# p is empty. Without the sink the steady passive scalar is exactly 1 in every cell -- a constant
# satisfies div(phi s) - lap(D s) = 0 with `bounded` removing div(phi) -- and the two codes agreed to
# 0.000e+00 on a field that discriminates nothing (measured before the sink was added).
p = open(d + '/0/p').read()
bf = re.search(r'boundaryField\s*\{(.*)\}', p, re.S).group(1)
entries = []; sink = None
for m in re.finditer(r'(\w+)\s*\{([^{}]*)\}', bf):
    name, body = m.group(1), m.group(2)
    if 'empty' in body: entries.append('    %s { type empty; }' % name)
    elif name == 'inlet': entries.append('    %s { type fixedValue; value uniform 1; }' % name)
    elif name != 'outlet' and sink is None:
        sink = name; entries.append('    %s { type fixedValue; value uniform 0; }' % name)
    else: entries.append('    %s { type zeroGradient; }' % name)
assert sink, 'no wall patch to hold at 0'

open(d + '/0/tracer', 'w').write(
    'FoamFile { version 2.0; format ascii; class volScalarField; object tracer; }\n'
    'dimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n' + '\n'.join(entries) + '\n}\n')
f = d + '/system/fvSchemes'; s = open(f).read()
s = re.sub(r'(divSchemes\s*\{)', r'\1\n    div(phi,tracer) bounded Gauss upwind;', s, count=1)
s = re.sub(r'(laplacianSchemes\s*\{)', r'\1\n    laplacian(Dtracer,tracer) Gauss linear orthogonal;', s, count=1)
open(f, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
# every solver tight, so the flux both codes hand the tracer is the same to round-off
s = re.sub(r'tolerance\s+[^;]*;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[^;]*;', 'relTol 0;', s)
s = re.sub(r'(solvers\s*\{)', r'\1\n  tracer { solver PBiCGStab; preconditioner DILU; tolerance 1e-14; relTol 0; }', s, count=1)
open(f, 'w').write(s)
PY
}
stage "$W/of"; stage "$W/host"; stage "$W/cuda"
( set +u; source "$OFBASHRC" >/dev/null 2>&1; cd "$W/of" && rhoSimpleFoam > of.log 2>&1 ) || { tail -6 "$W/of/of.log"; echo "FAIL: OpenFOAM did not run"; exit 1; }
[ -f "$W/of/$N/tracer" ] && say "OpenFOAM wrote its tracer at $N" ok || { say "OpenFOAM wrote its tracer at $N" FAIL; exit 1; }
for arm in "1 host" "cuda cuda"; do
    set -- $arm; sel=$1; label=$2
    ( cd "$W/$label" && BRAE_RHOSIMPLEFOAM_MIRROR=$sel "$BRAE" -case "$W/$label" > run.log 2>&1 ) || { tail -6 "$W/$label/run.log"; say "$label  the mirror run finished" FAIL; continue; }
    grep -q "Solving for tracer" "$W/$label/run.log" && say "$label  the tracer was solved (Solving for tracer)" ok \
                                                       || say "$label  the tracer was solved (Solving for tracer)" FAIL
    [ -f "$W/$label/$N/tracer" ] && say "$label  the tracer was written at $N" ok || say "$label  the tracer was written at $N" FAIL
done
python3 - "$W" "$N" <<'PY' || fail=1
import re, sys, os, numpy as np
W, N = sys.argv[1], sys.argv[2]
def read(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', b)
    if not m:
        u = re.search(rb'internalField\s+uniform\s+([-0-9.eE+]+)', b); return None if not u else np.full(1, float(u.group(1)))
    n = int(m.group(1)); start = m.end()
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary': return np.frombuffer(b[start:start+n*8], dtype='<f8')
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n]])
bad = 0
def say(ok, what):
    global bad
    print('  %-74s %s' % (what, 'ok' if ok else 'FAIL'))
    if not ok: bad = 1
of = read(W + '/of/%s/tracer' % N)
# a field that is not (nearly) constant: the constant-1 solution has zero spread; a real balance
# between the inlet at 1 and the sink at 0 spreads over most of the range
spread = float(np.max(of) - np.min(of)) if of is not None and of.size > 1 else 0.0
print('  OpenFOAM tracer at %s: min %.3f max %.3f std %.3f' % (N, float(np.min(of)), float(np.max(of)), float(np.std(of))))
say(of is not None and of.size > 1 and spread > 0.5 and float(np.std(of)) > 0.05, 'CONTROL: OpenFOAM\'s tracer is a real field (spread > 0.5, std > 0.05), not a constant')
BOUND = 1e-9   # measured 2.019e-12 (host) / 2.716e-12 (cuda), max|diff| 7e-12: three decades of room
for label in ('host', 'cuda'):
    fn = W + '/%s/%s/tracer' % (label, N)
    if not os.path.exists(fn): say(False, label + '  tracer vs OpenFOAM (no file)'); continue
    b = read(fn)
    if b is None or b.size != of.size: say(False, label + '  tracer vs OpenFOAM (unreadable / wrong size)'); continue
    e = np.linalg.norm(b - of) / np.linalg.norm(of)
    print('  %s  tracer at %s vs OpenFOAM: rel L2 %.3e   bound %.1e' % (label, N, e, BOUND))
    say(e <= BOUND, label + '  tracer tracks OpenFOAM')
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: scalarTransport runs on both mirror arms and tracks OpenFOAM"
exit $fail
