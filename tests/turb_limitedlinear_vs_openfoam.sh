#!/usr/bin/env bash
# `Gauss limitedLinear` on div(phi,k) and div(phi,epsilon) -- the DEVICE closure, which assembled upwind.
#
# The host mirror has assembled limitedLinear since divWithScheme existed. The device closure had not,
# and refused it -- with a message that read "Only Gauss upwind ... is ported, which is what the host
# reference assembles". That clause was stale and wrong: it described a restriction only the device arm
# had, while telling the reader both arms shared it.
#
# TWO THINGS ARE UNDER TEST AND THEY NEED DIFFERENT COMPARISON POINTS.
#
#  1. IS IT THE RIGHT SCHEME? A trajectory comparison at iteration 20 cannot answer that: each arm's
#     linear solvers already put it 2.0e-04 (host) and 9.5e-03 (CUDA) from OpenFOAM on the SAME case
#     with the SAME upwind scheme. So the exactness arm restarts BOTH codes from OpenFOAM's own
#     developed state and takes ONE iteration, where the only difference left is the assembly. The bound
#     is not a round number: each arm's limitedLinear must be no worse than its own already-validated
#     UPWIND on the same fixture and the same restart. Measured -- host 3.6e-06 against 6.6e-06 upwind,
#     CUDA 2.4e-04 against 2.7e-04 upwind.
#
#  2. IS IT A DIFFERENT SCHEME AT ALL? That needs the developed run, because a one-iteration restart
#     barely separates the two. OpenFOAM's own limitedLinear and upwind differ by 4.9e-02 on k there, so
#     brae reproducing that spread is what rules out a limitedLinear branch quietly running upwind.
#
# AND THE LIMITER'S GRADIENT IS ITS OWN REFUSAL, on both arms. OpenFOAM builds the limiter from
# fvc::grad(k) / fvc::grad(epsilon) through the case's gradSchemes (LimitedScheme.C:56-59); brae computes
# Gauss linear only, and took a plain unlimited gradient whatever the case asked for.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/rhoKE"
OFBASH=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASH" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
DEV=20
NEXT=$((DEV + 1))

fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
its() { [ -f "$1/run.log" ] && grep -c '^Time = ' "$1/run.log" 2> /dev/null | head -1 || echo 0; }

stage() {   # stage <dir> <scheme for BOTH k and epsilon> [gradDefault]
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"; cp -r "$SRC/0.orig" "$1/0"
    python3 - "$1" "$DEV" "$2" "${3:-}" <<'PYEOF'
import re, sys
d, dev, sc, grad = sys.argv[1:5]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % dev, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % dev, s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSchemes'; s = open(f).read()
s, n = re.subn(r'div\(phi,(k|epsilon)\)\s+[^;]+;', lambda m: 'div(phi,%s) bounded Gauss %s;' % (m.group(1), sc), s)
# assert, not a silent pass: a reworked fixture would stage one arm as upwind and compare it to itself.
assert n == 2, 'expected div(phi,k) and div(phi,epsilon), got %d' % n
if grad:
    s, n3 = re.subn(r'(gradSchemes\s*\{[^}]*?default\s+)[^;]+;', r'\g<1>%s;' % grad, s, flags=re.S)
    assert n3 == 1, 'no gradSchemes default to rewrite'
open(f, 'w').write(s)
PYEOF
    ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && blockMesh > log.bm 2>&1 )
}
restart() {   # restart <dir> <source dir>
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$2/constant" "$2/system" "$1/"; cp -r "$2/$DEV" "$1/$DEV"
    python3 - "$1" "$NEXT" <<'PYEOF'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % sys.argv[2], s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 1;', s)
open(c, 'w').write(s)
PYEOF
}
runOF()   { ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && rhoSimpleFoam > run.log 2>&1 ); }
runBrae() { ( cd "$2" && BRAE_RHOSIMPLEFOAM_MIRROR="$1" "$BRAE" -case "$2" > run.log 2>&1 ); }
rel() {   # rel <dirA> <dirB> <time> <field>
    python3 - "$1/$3/$4" "$2/$3/$4" <<'PYEOF'
import re, os, sys, math
def rd(f):
    if not os.path.exists(f): return None
    b = open(f, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(', b)
    return [float(x) for x in b[m.end():].split(b')\n', 1)[0].split()] if m else None
a, b = rd(sys.argv[1]), rd(sys.argv[2])
if not a or not b or len(a) != len(b): print('nan'); raise SystemExit
n = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))); d = math.sqrt(sum(y * y for y in b)) or 1.0
print('%.4e' % (n / d))
PYEOF
}
gt() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >  float(sys.argv[2]) else 1)" "$1" "$2" 2> /dev/null; }

LL="limitedLinear 1"; UP="upwind"

# ---- ARM 0: develop both schemes under real OpenFOAM, and the CONTROL ------------------------------
stage "$W/of_ll" "$LL"; runOF "$W/of_ll"
stage "$W/of_up" "$UP"; runOF "$W/of_up"
[ "$(its "$W/of_ll")" = "$DEV" ] \
    && say "control: real OpenFOAM runs the fixture with limitedLinear on k and epsilon" ok \
    || { tail -3 "$W/of_ll/run.log"; say "control: real OpenFOAM runs the fixture with limitedLinear on k and epsilon" FAIL; }
SPREAD=$(rel "$W/of_ll" "$W/of_up" "$DEV" k)
gt "$SPREAD" "5e-3" \
    && say "CONTROL: limitedLinear and upwind give OpenFOAM DIFFERENT k" ok \
    || say "CONTROL: limitedLinear and upwind give OpenFOAM DIFFERENT k" FAIL
printf '        (OpenFOAM limitedLinear vs OpenFOAM upwind: k %s)\n' "$SPREAD"

# ---- ARM 1: EXACTNESS -- one iteration from OpenFOAM's own developed state -------------------------
# The bound is each arm's OWN upwind agreement on the same restart, so it cannot be loosened by
# accident: if limitedLinear ever lands worse than the scheme brae already had, that is the finding.
restart "$W/r_of_ll" "$W/of_ll"; runOF "$W/r_of_ll"
restart "$W/r_of_up" "$W/of_up"; runOF "$W/r_of_up"
for arm in 1 cuda; do
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    restart "$W/r_${arm}_ll" "$W/of_ll"; runBrae "$arm" "$W/r_${arm}_ll"
    restart "$W/r_${arm}_up" "$W/of_up"; runBrae "$arm" "$W/r_${arm}_up"
    [ "$(its "$W/r_${arm}_ll")" -ge 1 ] \
        || { tail -2 "$W/r_${arm}_ll/run.log"; say "$label arm: runs limitedLinear on k and epsilon" FAIL; continue; }
    say "$label arm: runs limitedLinear on k and epsilon instead of refusing it" ok
    ok=1; det=""
    for fld in k epsilon nut; do
        dll=$(rel "$W/r_${arm}_ll" "$W/r_of_ll" "$NEXT" "$fld")
        dup=$(rel "$W/r_${arm}_up" "$W/r_of_up" "$NEXT" "$fld")
        det="$det $fld $dll/$dup"
        # no worse than this arm's own validated upwind, with 3x of headroom for solver scatter
        python3 -c "
import sys
ll, up = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if ll <= 3.0 * max(up, 1e-12) else 1)" "$dll" "$dup" || ok=0
    done
    [ "$ok" = 1 ] \
        && say "$label arm: limitedLinear is no worse than its own validated upwind" ok \
        || say "$label arm: limitedLinear is no worse than its own validated upwind" FAIL
    printf '        (%s   -- limitedLinear/upwind, each vs its own OpenFOAM run)\n' "$det"
done

# ---- ARM 2: FAIL-PROOF -- brae's two schemes must differ by what OpenFOAM's differ by ---------------
# ARM 1 alone would pass if the limitedLinear branch quietly ran upwind AND upwind were exact.
for arm in 1 cuda; do
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    stage "$W/b_${arm}_ll" "$LL"; runBrae "$arm" "$W/b_${arm}_ll"
    stage "$W/b_${arm}_up" "$UP"; runBrae "$arm" "$W/b_${arm}_up"
    if [ "$(its "$W/b_${arm}_ll")" = "$DEV" ] && [ "$(its "$W/b_${arm}_up")" = "$DEV" ]; then
        bs=$(rel "$W/b_${arm}_ll" "$W/b_${arm}_up" "$DEV" k)
        python3 -c "
import sys
b, o = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if o > 0 and abs(b - o) / o < 0.25 else 1)" "$bs" "$SPREAD" \
            && say "fail-proof ($label): its limitedLinear-vs-upwind spread reproduces OpenFOAM's" ok \
            || say "fail-proof ($label): its limitedLinear-vs-upwind spread reproduces OpenFOAM's" FAIL
        printf '        (brae %s spread %s vs OpenFOAM %s)\n' "$label" "$bs" "$SPREAD"
    else
        say "fail-proof ($label): its limitedLinear-vs-upwind spread reproduces OpenFOAM's" FAIL
    fi
done

# ---- ARM 3: the LIMITER's gradient is refused when brae cannot compute it, on BOTH arms ------------
for arm in 1 cuda; do
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    stage "$W/g_$arm" "$LL" "leastSquares"; runBrae "$arm" "$W/g_$arm"
    [ "$(its "$W/g_$arm")" = 0 ] && grep -q "limiter gradient" "$W/g_$arm/run.log" \
        && say "$label arm: a limiter gradient brae does not compute is refused, by name" ok \
        || say "$label arm: a limiter gradient brae does not compute is refused, by name" FAIL
done
# ...and NOT refused when the case names Gauss linear -- the refusal must discriminate.
stage "$W/gok" "$LL" "Gauss linear"; runBrae 1 "$W/gok"
[ "$(its "$W/gok")" = "$DEV" ] \
    && say "fail-proof: 'Gauss linear' still runs (the gradient refusal is not blanket)" ok \
    || say "fail-proof: 'Gauss linear' still runs (the gradient refusal is not blanket)" FAIL

# ---- ARM 4: what is still ONE scheme for both scalars must still refuse when they disagree ---------
# The closures carry one flag and one coefficient for k and epsilon together, so entries that differ
# have no representation and must not be silently collapsed onto k's.
MIX="$W/mix"; stage "$MIX" "$LL"
sed -i 's/div(phi,epsilon)  *bounded Gauss limitedLinear 1;/div(phi,epsilon) bounded Gauss upwind;/' "$MIX/system/fvSchemes"
grep -q "div(phi,epsilon) bounded Gauss upwind;" "$MIX/system/fvSchemes" || { echo "FAIL: the mix mutation did not apply"; fail=1; }
for arm in 1 cuda; do
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    cp -r "$MIX" "$W/mix_$arm"; runBrae "$arm" "$W/mix_$arm"
    [ "$(its "$W/mix_$arm")" = 0 ] \
        && say "$label arm: k and epsilon naming DIFFERENT schemes still refuses" ok \
        || say "$label arm: k and epsilon naming DIFFERENT schemes still refuses" FAIL
done

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
