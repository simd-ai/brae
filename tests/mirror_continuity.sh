#!/usr/bin/env bash
# Both rhoSimpleFoam mirror arms print OpenFOAM's `time step continuity errors` line, and it is right.
#
# rhoSimpleFoam's pEqn.H:81 and pcEqn.H:94 include continuityErrs.H every iteration; the mirror printed
# nothing (queue item 16a). The line is a diagnostic, not physics, and an OpenFOAM run is not its
# oracle: after the pressure solve the residual continuity error IS the linear solver's stopping point,
# and brae's AMG-PCG and OpenFOAM's GAMG stop at different points (item 61, declined). The oracle is the
# DEFINITION, recomputed from the phi the mirror writes:
#   ARM 1   host mirror on rhoBox, 3 iterations: one line per iteration; its cumulative is the running
#           sum of its globals (to the 6 digits %g prints); its LAST line equals test_continuity_from_phi
#           on ITS OWN written 3/phi.
#   ARM 2   CUDA mirror: the same three, on its own written phi.
#   The arms are NOT compared to each other, on purpose: at iteration 1 on rhoBox the host reads
#   sum local 8.66e-06 and the CUDA arm 1.27e-05, because the host arm solves p with its host solver
#   and the CUDA arm with AMG-PCG, and each stops the residual continuity at its own tolerance. That is
#   the same reason an OpenFOAM run is not the oracle here. The definition is.
#   CONTROL both arms must reach 3 iterations and write phi (a run that wrote nothing would let the
#           oracle read a stale file).
# FAIL-PROOF: the pre-item-16a CUDA arm printed no such line at all (its log from the item-15 build,
# run on this fixture, has 0 of them -> the count line FAILS); the host step has printed it since the
# port and never went through this gate before.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
TOOL="${TOOL_BIN:-$ROOT/build/test_continuity_from_phi}"
SRC="${1:-$ROOT/validation/rhoBox}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -x "$TOOL" ] || { echo "SKIP: no test_continuity_from_phi at $TOOL"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
N=3
for arm in "1 host" "cuda cuda"; do
    set -- $arm; sel=$1; label=$2; d="$W/$label"
    rm -rf "$d"; cp -r "$SRC" "$d"; [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    python3 - "$d" "$N" <<'PY'
import re, sys
d, n = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
open(c, 'w').write(s)
PY
    ( cd "$d" && BRAE_RHOSIMPLEFOAM_MIRROR=$sel "$BRAE" -case "$d" > log 2>&1 ) || { tail -4 "$d/log"; say "$label  the run finished" FAIL; continue; }
    n=$(grep -c "^time step continuity errors" "$d/log")
    [ "$n" -eq "$N" ] && say "$label  one continuity line per iteration ($n of $N)" ok || say "$label  one continuity line per iteration ($n of $N)" FAIL
    [ -f "$d/$N/phi" ] && say "$label  phi written at $N (the oracle reads it)" ok || say "$label  phi written at $N (the oracle reads it)" FAIL
    grep "^time step continuity errors" "$d/log" > "$d/cont"
done
deltaT=$(grep -oE "^deltaT\s+[0-9.eE+-]+" "$W/host/system/controlDict" | awk '{print $2}'); deltaT=${deltaT:-1}
for label in host cuda; do
    "$TOOL" "$W/$label" "$N" "$deltaT" > "$W/$label/oracle" 2>&1 || { cat "$W/$label/oracle"; say "$label  the oracle recomputed the last line" FAIL; }
done
python3 - "$W" <<'PY' || fail=1
import re, sys
W = sys.argv[1]
def parse(p):
    out = []
    for l in open(p):
        m = re.search(r'sum local = ([-0-9.eE+]+), global = ([-0-9.eE+]+), cumulative = ([-0-9.eE+]+)', l)
        if m: out.append(tuple(float(x) for x in m.groups()))
    return out
bad = 0
def say(ok, what):
    global bad
    print('  %-74s %s' % (what, 'ok' if ok else 'FAIL'))
    if not ok: bad = 1
rel = lambda a, b: abs(a - b) / max(abs(a), abs(b), 1e-300)
for label in ('host', 'cuda'):
    lines = parse(W + '/' + label + '/cont')
    run = 0.0; okc = True
    for (sl, g, cu) in lines:
        run += g
        # %g prints 6 significant digits; the running sum of six-digit globals against a six-digit
        # cumulative can differ in the sixth place, so the check is at 2e-5 relative, not round-off.
        okc = okc and abs(cu - run) <= 2e-5 * max(abs(run), abs(cu), 1e-300)
    say(okc and len(lines) > 0, label + '  cumulative is the running sum of the globals (to the printed digits)')
    m = re.search(r'sum local = ([-0-9.eE+]+), global = ([-0-9.eE+]+)', open(W + '/' + label + '/oracle').read())
    if not m or not lines:
        say(False, label + '  the oracle produced numbers'); continue
    osl, og = float(m.group(1)), float(m.group(2))
    print('  %s last line: sum local %.6e global %.6e | continuityErrs.H from its written phi: %.6e %.6e' % (label, lines[-1][0], lines[-1][1], osl, og))
    say(rel(lines[-1][0], osl) <= 1e-5 and rel(lines[-1][1], og) <= 1e-5,
        label + '  last line equals continuityErrs.H recomputed from its written phi (<= 1e-5)')
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: both mirror arms print OpenFOAM's continuity errors, and the numbers are the definition's"
exit $fail
