#!/usr/bin/env bash
# The stage tests/bound_message_vs_openfoam.sh had until 2026-11-02, kept as a stress test of what a code does when a weak solve wrecks a field.
# sbMatched with PBiCGStab on the bare diagonal on k and epsilon, relTol 0.1: BRAE_POLY_KE=1 BRAE_DILU_KE=0 is item 78's substitute.
#
# WHAT HAPPENS, measured (jobs 4845199, 4845223). At iteration 7 the solve takes k and epsilon below zero in the same outlet cell. bound.C:59 then
# floors the face's stored value for both, at 1e-15, and `nut_ = Cmu*sqr(k)/epsilon` on that calculated face is 3.19e+15, alphat 1.21e+15. The next
# energy and epsilon solves see that diffusivity: brae's residual for epsilon is NaN at iteration 8. The candidate's patch half is the right one (the
# bounded gate's control), and the old one survived only by holding the wrong value on that face.
# OPENFOAM DOES THE SAME. rhoSimpleFoam restarted from brae's written iteration 6 bounds k and epsilon at 7 and aborts at 8, T at -1.9e+16
# ("Maximum number of iterations exceeded"). So this is what the state does, not a defect of the bound. OpenFOAM's own run from the start does not
# reach the state: its solver is not brae's substitute.
#
#   ARM 1      brae ends this run by itself with an error naming the non-finite field, and has written no field holding one
#   ARM 2      the premise of the account above: at iteration 7 an outlet face has k and epsilon on the floor and nut above 1e12
#   ARM 3      OpenFOAM from brae's iteration 6 fails too, at the same iteration
#   CONTROL    BRAE_CONTROL_BOUND_PATCH_LIVE=1, the old patch half: runs all twelve iterations, so arm 1's failure is the patch half's
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
stage() {
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    cp -r "$SRC/0.orig" "$1/0"
    python3 - "$1" <<'PYEOF'
import re, sys
d = sys.argv[1]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
for k, v in (('endTime', '12'), ('writeInterval', '1'), ('writeFormat', 'ascii'), ('writePrecision', '17')):
    s, n = re.subn(r'(?m)(^|\s)%s\s+[^;]*;' % k, r'\1%s %s;' % (k, v), s)
    if n == 0:
        s += '\n%s %s;\n' % (k, v)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
old = '"(U|e|k|epsilon)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }'
assert old in s, 'the fixture fvSolution changed shape'
s = s.replace(old, '"(U|e)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }\n'
                   '    "(k|epsilon)"\n    {\n        solver          GAMG;\n'
                   '        smoother        GaussSeidel;\n        tolerance       1e-08;\n'
                   '        relTol          0.1;\n    }')
s = s.replace('p { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }',
              'p { solver GAMG; smoother GaussSeidel; tolerance 1e-08; relTol 0.1; }')
open(f, 'w').write(s)
PYEOF
}
stage "$W/br"
( cd "$W/br" && BRAE_POLY_KE=1 BRAE_DILU_KE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/br" > run.log 2>&1 ); RC=$?
[ "$RC" != 0 ] && grep -aq -E "brae.*(residual|non-finite|nan|diverged)" "$W/br/run.log" \
    && say "ARM 1: brae ends the run itself, naming what went wrong" ok \
    || say "ARM 1: brae ends the run itself, naming what went wrong (exit $RC)" FAIL
printf '        (%s)\n' "$(grep -a -m1 -E 'residual|non-finite|diverged' "$W/br/run.log" | cut -c1-110)"
BADW=$(grep -a -l -E '(^|[[:space:]])(nan|-nan|inf)([[:space:]]|$)' "$W"/br/[0-9]*/k "$W"/br/[0-9]*/epsilon "$W"/br/[0-9]*/U "$W"/br/[0-9]*/p 2>/dev/null | wc -l)
[ "$BADW" = 0 ] && say "ARM 1: no written field holds a non-finite value" ok || say "ARM 1: $BADW written fields hold a non-finite value" FAIL
python3 - "$W/br/7" <<'PYEOF' && say "ARM 2: iteration 7, an outlet face on both floors and nut above 1e12" ok || say "ARM 2: iteration 7, an outlet face on both floors and nut above 1e12" FAIL
import re, sys
def patch(fn, name):
    s = open(fn, errors='replace').read()
    b = s[s.index('boundaryField'):]
    m = re.search(name + r'\s*\{[^{}]*?value\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n?\(([^)]*)\)', b, flags=re.S)
    return [float(x) for x in m.group(2).split()]
d = sys.argv[1]
k, e, n = patch(d + '/k', 'outlet'), patch(d + '/epsilon', 'outlet'), patch(d + '/nut', 'outlet')
print('        outlet: k min %.3g, epsilon min %.3g, nut max %.3g' % (min(k), min(e), max(n)))
raise SystemExit(0 if min(k) <= 2e-15 and min(e) <= 2e-15 and max(n) > 1e12 else 1)
PYEOF
rm -rf "$W/of"; mkdir -p "$W/of"
cp -r "$W/br/constant" "$W/br/system" "$W/br/6" "$W/of/"
rm -rf "$W/of/6/uniform"
python3 - "$W/of" <<'PYEOF'
import re, sys
d = sys.argv[1]
p = d + '/system/controlDict'; s = open(p).read()
for k, v in (('startFrom', 'startTime'), ('startTime', '6'), ('endTime', '9')):
    s, n = re.subn(r'(?m)(^|\s)%s\s+[^;]*;' % k, r'\1%s %s;' % (k, v), s)
    if n == 0:
        s += '\n%s %s;\n' % (k, v)
open(p, 'w').write(s)
p = d + '/system/fvSolution'; s = open(p).read()
old = '        solver          GAMG;\n        smoother        GaussSeidel;\n        tolerance       1e-08;\n        relTol          0.1;\n    }'
assert old in s
s = s.replace(old, '        solver          PBiCGStab;\n        preconditioner  diagonal;\n        tolerance       1e-08;\n        relTol          0.1;\n    }', 1)
open(p, 'w').write(s)
PYEOF
( set +u; source "$OFBASHRC" > /dev/null 2>&1; cd "$W/of" && rhoSimpleFoam > run.log 2>&1 ); RCO=$?
OT=$(grep -a '^Time = ' "$W/of/run.log" | tail -n 1 | sed 's/Time = //')
[ "$RCO" != 0 ] && [ "$OT" = 8 ] && say "ARM 3: OpenFOAM from brae's iteration 6 fails at iteration 8" ok \
    || say "ARM 3: OpenFOAM from brae's iteration 6 fails at iteration 8 (exit $RCO, last Time $OT)" FAIL
printf '        (%s)\n' "$(grep -a -m1 'Maximum number of iterations' "$W/of/run.log" | cut -c1-110)"
stage "$W/live"
( cd "$W/live" && BRAE_POLY_KE=1 BRAE_DILU_KE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda BRAE_CONTROL_BOUND_PATCH_LIVE=1 "$BRAE" -case "$W/live" > run.log 2>&1 ); RCL=$?
LN=$(grep -a -c '^Time = ' "$W/live/run.log")
[ "$RCL" = 0 ] && [ "$LN" = 12 ] && say "CONTROL: the old patch half runs all twelve iterations" ok \
    || say "CONTROL: the old patch half runs all twelve iterations (exit $RCL, $LN iterations)" FAIL
[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
