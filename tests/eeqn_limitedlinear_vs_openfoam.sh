#!/usr/bin/env bash
# `Gauss limitedLinear` on the energy convection -- div(phi,e|h) and div(phi,Ekp|K).
#
# Both are SEPARATE fvSchemes entries and both were refused. validation/rhoLU names limitedLinear on all
# four and is otherwise fully ported, so it was refused on nothing but this.
#
# THE COMPARISON POINT IS A DEVELOPED STATE, AND THAT IS THE WHOLE DESIGN OF THIS GATE.
# rhoLU starts from a uniform T = 300, so on iteration 1 the internal field is constant, every internal
# face has `gradf = phiN - phiP = 0`, and OpenFOAM's own NVDTVD ratio takes its degenerate branch:
#     if (mag(gradcf) >= 1000*mag(gradf)) r = 2*1000*sign(gradcf)*sign(gradf) - 1;
# With gradf = 0 that test is always true and r is decided by `sign(gradcf)` alone -- where gradcf is
# the Gauss gradient of a CONSTANT field, i.e. round-off, whose sign is arbitrary in either code. So on
# iteration 1 the two codes flip different faces between central and upwind and disagree by 6.6e-04 with
# nothing wrong in either. Measured. A gate anchored there would read as a defect that does not exist.
#
# So both codes are RESTARTED from OpenFOAM's own developed state (T spans 300..381 K, the limiter fully
# engaged) and take ONE iteration. brae then reproduces OpenFOAM's temperature EXACTLY -- 0.0e+00, not a
# tolerance -- because the two are assembling the same matrix from the same fields.
#
# THE CONTROL is OpenFOAM against itself: limitedLinear vs upwind on the same case. If those agreed,
# every arm here would pass with the upwind this change replaced.
#
# AND THE LIMITER'S GRADIENT IS ITS OWN REFUSAL. OpenFOAM builds the limiter from fvc::grad(<field>)
# through the case's gradSchemes (LimitedScheme.C:56-59). brae computes Gauss linear gradients only, so a
# case resolving that key to anything else is refused -- ARM 4. Without it, clearing this blocker would
# make gasMixing (`gradSchemes { default leastSquares; }`) RUN and be quietly wrong, which is strictly
# worse than the refusal it replaced.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/rhoLU"
OFBASH=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASH" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
DEV=50          # iterations of real OpenFOAM to reach a developed state
NEXT=$((DEV + 1))

fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
its() { [ -f "$1/run.log" ] && grep -c '^Time = ' "$1/run.log" 2> /dev/null | head -1 || echo 0; }

# stage <dir> <div(phi,he) scheme> <div(phi,KE) scheme> <endTime> [gradDefault]
stage() {
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$1/0"; else cp -r "$SRC/0" "$1/0"; fi
    python3 - "$1" "$2" "$3" "$4" "${5:-}" <<'PYEOF'
import re, sys
d, he, ke, end, grad = sys.argv[1:6]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % end, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % end, s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSchemes'; s = open(f).read()
# The fixture writes all four keys; h/e and K/Ekp move together because only one of each pair is live.
s, n1 = re.subn(r'div\(phi,(h|e)\)\s+[^;]+;',   lambda m: 'div(phi,%s) %s;' % (m.group(1), he), s)
s, n2 = re.subn(r'div\(phi,(K|Ekp)\)\s+[^;]+;', lambda m: 'div(phi,%s) %s;' % (m.group(1), ke), s)
# assert, not a silent pass: a reworked fixture would otherwise stage BOTH arms identically and every
# comparison below would agree perfectly while testing nothing.
assert n1 == 2 and n2 == 2, 'expected 2 energy and 2 kinetic div entries, got %d/%d' % (n1, n2)
if grad:
    s, n3 = re.subn(r'(gradSchemes\s*\{[^}]*?default\s+)[^;]+;', r'\g<1>%s;' % grad, s, flags=re.S)
    assert n3 == 1, 'no gradSchemes default to rewrite'
open(f, 'w').write(s)
PYEOF
    ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && blockMesh > log.bm 2>&1 )
}
runOF()   { ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && rhoSimpleFoam > run.log 2>&1 ); }
runBrae() { ( cd "$2" && BRAE_RHOSIMPLEFOAM_MIRROR="$1" "$BRAE" -case "$2" > run.log 2>&1 ); }
relT() {    # relT <dirA> <dirB> <time>
    python3 - "$1/$3/T" "$2/$3/T" <<'PYEOF'
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
le() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$1" "$2" 2> /dev/null; }
gt() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >  float(sys.argv[2]) else 1)" "$1" "$2" 2> /dev/null; }

LL="bounded Gauss limitedLinear 1"
UP="bounded Gauss upwind"

# ---- ARM 0: the developed state, and the CONTROL that the scheme matters ---------------------------
stage "$W/dev" "$LL" "$LL" "$DEV";   runOF "$W/dev"
stage "$W/devU" "$UP" "$UP" "$DEV";  runOF "$W/devU"
[ "$(its "$W/dev")" = "$DEV" ] \
    && say "control: real OpenFOAM develops the fixture with limitedLinear" ok \
    || { tail -3 "$W/dev/run.log"; say "control: real OpenFOAM develops the fixture with limitedLinear" FAIL; }
SPREAD=$(relT "$W/dev" "$W/devU" "$DEV")
gt "$SPREAD" "1e-4" \
    && say "CONTROL: limitedLinear and upwind give OpenFOAM DIFFERENT answers" ok \
    || say "CONTROL: limitedLinear and upwind give OpenFOAM DIFFERENT answers" FAIL
printf '        (OpenFOAM limitedLinear vs OpenFOAM upwind on T: %s)\n' "$SPREAD"

# ---- ARM 1: ONE iteration from OpenFOAM's own developed state --------------------------------------
# Both codes read the SAME fields, so nothing here is a trajectory: any difference is discretisation.
restartFrom() {   # restartFrom <dir> <source dir>
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
restartFrom "$W/r_of" "$W/dev"; runOF "$W/r_of"
restartFrom "$W/r_br" "$W/dev"; runBrae 1 "$W/r_br"
[ "$(its "$W/r_br")" -ge 1 ] \
    && say "host arm: runs limitedLinear instead of refusing it" ok \
    || { tail -2 "$W/r_br/run.log"; say "host arm: runs limitedLinear instead of refusing it" FAIL; }
D1=$(relT "$W/r_br" "$W/r_of" "$NEXT")
le "$D1" "1e-9" \
    && say "host arm: one iteration from OpenFOAM's state reproduces its temperature" ok \
    || say "host arm: one iteration from OpenFOAM's state reproduces its temperature" FAIL
printf '        (brae vs OpenFOAM after one iteration: T %s; the scheme is worth %s)\n' "$D1" "$SPREAD"

# ---- ARM 2: the two entries are INDEPENDENT --------------------------------------------------------
# div(phi,e) and div(phi,Ekp) are separate fvSchemes keys. A port that read one and copied it to the
# other would pass ARM 1 and be wrong on any case that separates them.
for spec in "$LL|$UP|he-limited-KE-upwind" "$UP|$LL|he-upwind-KE-limited"; do
    he="${spec%%|*}"; rest="${spec#*|}"; ke="${rest%%|*}"; tag="${rest#*|}"
    stage "$W/m_$tag" "$he" "$ke" "$DEV"; runOF "$W/m_$tag"
    restartFrom "$W/mo_$tag" "$W/m_$tag"; runOF "$W/mo_$tag"
    restartFrom "$W/mb_$tag" "$W/m_$tag"; runBrae 1 "$W/mb_$tag"
    d=$(relT "$W/mb_$tag" "$W/mo_$tag" "$NEXT")
    le "$d" "1e-9" \
        && say "host arm: $tag matches OpenFOAM (the two entries are read separately)" ok \
        || say "host arm: $tag matches OpenFOAM (the two entries are read separately)" FAIL
    printf '        (%s)\n' "$d"
done

# ---- ARM 3: THE FAIL-PROOF -- upwind must still be exact, and DIFFERENT ----------------------------
# If the limitedLinear branch were silently running upwind, ARM 1 would still pass whenever the two
# happened to agree. This pins both ends: upwind is exact, and brae's own two schemes differ by what
# OpenFOAM's differ by.
restartFrom "$W/ru_of" "$W/devU"; runOF "$W/ru_of"
restartFrom "$W/ru_br" "$W/devU"; runBrae 1 "$W/ru_br"
du=$(relT "$W/ru_br" "$W/ru_of" "$NEXT")
le "$du" "1e-9" \
    && say "fail-proof: the upwind arm is still exact against OpenFOAM" ok \
    || say "fail-proof: the upwind arm is still exact against OpenFOAM" FAIL
BSPREAD=$(relT "$W/dev" "$W/devU" "$DEV")   # OpenFOAM's own, recomputed for the message
stage "$W/bdev" "$LL" "$LL" "$DEV";  runBrae 1 "$W/bdev"
stage "$W/bdevU" "$UP" "$UP" "$DEV"; runBrae 1 "$W/bdevU"
if [ "$(its "$W/bdev")" -ge 1 ] && [ "$(its "$W/bdevU")" -ge 1 ]; then
    bs=$(relT "$W/bdev" "$W/bdevU" "$DEV")
    python3 -c "
import sys
b, o = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if o > 0 and abs(b - o) / o < 0.25 else 1)" "$bs" "$BSPREAD" \
        && say "fail-proof: brae's limitedLinear-vs-upwind spread reproduces OpenFOAM's" ok \
        || say "fail-proof: brae's limitedLinear-vs-upwind spread reproduces OpenFOAM's" FAIL
    printf '        (brae spread %s vs OpenFOAM spread %s)\n' "$bs" "$BSPREAD"
else
    say "fail-proof: brae's limitedLinear-vs-upwind spread reproduces OpenFOAM's" FAIL
fi

# ---- ARM 4: THE LIMITER'S GRADIENT is refused when it is not one brae computes ---------------------
# This is what stops gasMixing running quietly wrong once the div blocker is cleared.
stage "$W/lsq" "$LL" "$LL" 5 "leastSquares"; runBrae 1 "$W/lsq"
[ "$(its "$W/lsq")" = 0 ] && grep -q "LimitedScheme.C" "$W/lsq/run.log" \
    && say "a limiter gradient brae does not compute is refused, by name" ok \
    || say "a limiter gradient brae does not compute is refused, by name" FAIL
# ...and NOT refused when the case does name Gauss linear -- the refusal must discriminate.
stage "$W/gl" "$LL" "$LL" 5 "Gauss linear"; runBrae 1 "$W/gl"
[ "$(its "$W/gl")" = 5 ] \
    && say "fail-proof: 'Gauss linear' still runs (the refusal is not blanket)" ok \
    || say "fail-proof: 'Gauss linear' still runs (the refusal is not blanket)" FAIL
# ...and an upwind case is not refused for its gradient at all, since upwind has no limiter.
stage "$W/lsqU" "$UP" "$UP" 5 "leastSquares"; runBrae 1 "$W/lsqU"
[ "$(its "$W/lsqU")" = 5 ] \
    && say "fail-proof: upwind is unaffected -- it has no limiter and so no limiter gradient" ok \
    || say "fail-proof: upwind is unaffected -- it has no limiter and so no limiter gradient" FAIL

# ---- ARM 5: OpenFOAM's own coefficient range, which brae did not check -----------------------------
# limitedLinear.H:69-76 FatalIOErrors on k outside [0,1]; brae checked it only on the V path, so a
# scalar `limitedLinear 3` ran with twoByk = 2/3 -- a scheme OpenFOAM will not construct.
stage "$W/k3" "bounded Gauss limitedLinear 3" "$LL" 5; runBrae 1 "$W/k3"
[ "$(its "$W/k3")" = 0 ] && grep -q "0 <= k <= 1" "$W/k3/run.log" \
    && say "a limitedLinear coefficient outside [0,1] is refused, as OpenFOAM does" ok \
    || say "a limitedLinear coefficient outside [0,1] is refused, as OpenFOAM does" FAIL

# ---- ARM 6: the CUDA arm assembles it too, on the same restart -------------------------------------
# The device closure took the upwind coefficients unconditionally and refused limitedLinear by name. It
# now dispatches both terms: the implicit one through deviceDivLimitedCoeffs, the explicit Ekp through
# the same limiter's face weights. Measured on the same restart the host arm uses, so the two arms are
# compared to the SAME OpenFOAM run and to each other's upwind baseline.
restartFrom "$W/rc_ll" "$W/dev";  runBrae cuda "$W/rc_ll"
restartFrom "$W/rc_up" "$W/devU"; runBrae cuda "$W/rc_up"
[ "$(its "$W/rc_ll")" -ge 1 ] \
    && say "CUDA arm: runs limitedLinear instead of refusing it" ok \
    || { tail -2 "$W/rc_ll/run.log"; say "CUDA arm: runs limitedLinear instead of refusing it" FAIL; }
dcl=$(relT "$W/rc_ll" "$W/r_of" "$NEXT")
dcu=$(relT "$W/rc_up" "$W/ru_of" "$NEXT")
# No worse than this arm's OWN validated upwind, with 3x of headroom -- not a round number, so it
# cannot be loosened without noticing. Measured: 2.2e-11 against 1.5e-11.
python3 -c "
import sys
ll, up = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if ll <= 3.0 * max(up, 1e-14) else 1)" "$dcl" "$dcu" \
    && say "CUDA arm: limitedLinear is no worse than its own validated upwind" ok \
    || say "CUDA arm: limitedLinear is no worse than its own validated upwind" FAIL
printf '        (CUDA limitedLinear %s vs its upwind %s, each against its own OpenFOAM run)\n' "$dcl" "$dcu"
# ...and the CUDA arm must still refuse a limiter gradient it cannot compute, like the host.
stage "$W/culsq" "$LL" "$LL" 5 "leastSquares"; runBrae cuda "$W/culsq"
[ "$(its "$W/culsq")" = 0 ] \
    && say "CUDA arm: a limiter gradient brae does not compute is refused there too" ok \
    || say "CUDA arm: a limiter gradient brae does not compute is refused there too" FAIL
# ...and the two arms must SEPARATE the two entries the same way the host does.
restartFrom "$W/rc_mix" "$W/m_he-limited-KE-upwind"; runBrae cuda "$W/rc_mix"
dmix=$(relT "$W/rc_mix" "$W/mo_he-limited-KE-upwind" "$NEXT")
python3 -c "
import sys
sys.exit(0 if float(sys.argv[1]) <= 3.0 * max(float(sys.argv[2]), 1e-14) else 1)" "$dmix" "$dcu" \
    && say "CUDA arm: he-limited/KE-upwind matches OpenFOAM (entries read separately)" ok \
    || say "CUDA arm: he-limited/KE-upwind matches OpenFOAM (entries read separately)" FAIL
printf '        (%s)\n' "$dmix"

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
