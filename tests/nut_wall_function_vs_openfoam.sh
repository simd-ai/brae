#!/usr/bin/env bash
# The nut wall-function FAMILY. brae's compressible kEpsilon closure computed nutkWallFunction for every
# wall-function patch, so a case naming any other member got nutk's value under that member's name --
# and the mirror refused the whole family rather than admit it.
#
# OpenFOAM has exactly ONE dispatch point: the nut patch field's own virtual calcNut()
# (nutWallFunctionFvPatchScalarField.C:182). Everything downstream READS the result --
# epsilonWallFunctionFvPatchScalarField.C:333-334 takes turbModel.nut(patchi) into the near-wall
# production G -- so getting the dispatch right at that one place is the whole port.
#
# THE THREE ARE NOT VARIANTS OF ONE FUNCTION:
#   nutkWallFunction      yPlus is k-based, Cmu^0.25*y*sqrt(k)/nu; it never reads U at all.
#   nutUWallFunction      yPlus comes from a fixed-point iteration on the log law driven by
#                         |U_cell - U_wall| (nutUWallFunctionFvPatchScalarField.C:55-59).
#   nutLowReWallFunction  calcNut() returns Zero UNCONDITIONALLY (...LowRe...C:38-42 is the whole
#                         function). NOT "nutk on a resolved mesh": nutk's k-based yPlus can exceed
#                         yPlusLam on a mesh resolved in friction units and take the log branch.
#
# THE CONTROL IS OpenFOAM AGAINST ITSELF, and it is what makes this gate non-vacuous: the same case with
# only the nut patch type changed. Measured on this fixture at 5 iterations, nutU vs nutk differs by
# 3.97e-01 on epsilon and lowRe vs nutk by 7.99e-01, while brae agrees with its OWN oracle arm to ~1e-4.
# A brae that quietly ran nutk under all three names -- the precise defect this removes -- would sit
# three orders of magnitude away and could not pass.
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
IT=5
BOUND=1e-2   # brae vs its own oracle arm; measured 2.9e-05 nut / 1.5e-04 k / 3.9e-04 epsilon

fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
its() { [ -f "$1/run.log" ] && grep -c '^Time = ' "$1/run.log" 2> /dev/null | head -1 || echo 0; }

stage() {   # stage <dir> <nut patch type>
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"; cp -r "$SRC/0.orig" "$1/0"
    python3 - "$1" "$IT" "$2" <<'PYEOF'
import re, sys
d, it, typ = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % it, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % it, s)
open(c, 'w').write(s + '\n')
f = d + '/0/nut'; s = open(f).read()
s2, k = re.subn(r'nutkWallFunction', typ, s)
# assert, not a silent pass: a renamed patch or a reworked fixture would otherwise stage ONE arm as
# nutk and compare it against itself, and every arm below would agree perfectly and mean nothing.
assert k >= 1, 'no nutkWallFunction in 0/nut to substitute -- the fixture changed'
open(f, 'w').write(s2)
PYEOF
    ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && blockMesh > log.bm 2>&1 )
}
runOF()   { ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && rhoSimpleFoam > run.log 2>&1 ); }
runBrae() { ( cd "$2" && BRAE_RHOSIMPLEFOAM_MIRROR="$1" "$BRAE" -case "$2" > run.log 2>&1 ); }
rel() {     # rel <dirA> <dirB> <field> -> relative L2 of A against B
    python3 - "$1/$IT/$3" "$2/$IT/$3" <<'PYEOF'
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

TYPES="nutkWallFunction nutUWallFunction nutLowReWallFunction"

# ---- ARM 0: the ORACLE runs, one per member ------------------------------------------------------
for t in $TYPES; do
    stage "$W/of_$t" "$t"; runOF "$W/of_$t"
    [ "$(its "$W/of_$t")" = "$IT" ] \
        && say "control: real OpenFOAM runs the fixture with $t" ok \
        || { tail -3 "$W/of_$t/run.log"; say "control: real OpenFOAM runs the fixture with $t" FAIL; }
done

# ---- ARM 1: THE CONTROL -- the choice must MATTER to OpenFOAM ------------------------------------
# If the three agreed, every arm below would pass with the unconditional nutk this change removed.
for t in nutUWallFunction nutLowReWallFunction; do
    d=$(rel "$W/of_$t" "$W/of_nutkWallFunction" epsilon)
    gt "$d" "0.05" \
        && say "CONTROL: OpenFOAM's $t differs from its nutk, so the dispatch is testable" ok \
        || say "CONTROL: OpenFOAM's $t differs from its nutk, so the dispatch is testable" FAIL
    printf '        (OpenFOAM %s vs OpenFOAM nutk: epsilon %s, nut %s)\n' \
           "$t" "$d" "$(rel "$W/of_$t" "$W/of_nutkWallFunction" nut)"
done

# ---- ARM 2: the HOST arm dispatches -- each member against ITS OWN oracle -------------------------
for t in $TYPES; do
    stage "$W/b_$t" "$t"; runBrae 1 "$W/b_$t"
    [ "$(its "$W/b_$t")" = "$IT" ] \
        || { tail -2 "$W/b_$t/run.log"; say "host arm: runs $t" FAIL; continue; }
    ok=1; det=""
    for fld in nut k epsilon; do
        d=$(rel "$W/b_$t" "$W/of_$t" "$fld"); det="$det $fld $d"
        le "$d" "$BOUND" || ok=0
    done
    [ "$ok" = 1 ] \
        && say "host arm: $t matches OpenFOAM's own $t run" ok \
        || say "host arm: $t matches OpenFOAM's own $t run" FAIL
    printf '        (%s)\n' "$det"
done

# ---- ARM 3: THE FAIL-PROOF -- brae's members must differ from EACH OTHER the way OpenFOAM's do ----
# Matching per-arm is not enough on its own: if brae ran nutk three times AND the oracle staging were
# broken, all three would match. This asserts brae reproduces OpenFOAM's OWN spread.
for t in nutUWallFunction nutLowReWallFunction; do
    db=$(rel "$W/b_$t" "$W/b_nutkWallFunction" epsilon)
    do_=$(rel "$W/of_$t" "$W/of_nutkWallFunction" epsilon)
    python3 -c "
import sys
b, o = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if o > 0 and abs(b - o) / o < 0.25 else 1)" "$db" "$do_" \
        && say "fail-proof: brae's $t-vs-nutk spread reproduces OpenFOAM's" ok \
        || say "fail-proof: brae's $t-vs-nutk spread reproduces OpenFOAM's" FAIL
    printf '        (brae spread %s vs OpenFOAM spread %s)\n' "$db" "$do_"
done

# ---- ARM 4: the CUDA arm REFUSES what its device closure cannot compute ---------------------------
# Both arms read the same createFields, so narrowing the host refusal let this arm through running nutk
# under every name (measured: epsilon 2.20e-01 for nutU, 2.49e+00 for nutLowRe). It refuses by name
# until the device kernel dispatches too -- an arm that cannot do it must say so, not approximate it.
stage "$W/cu_nutk" nutkWallFunction; runBrae cuda "$W/cu_nutk"
[ "$(its "$W/cu_nutk")" = "$IT" ] \
    && say "CUDA arm: still runs nutkWallFunction, which its closure does compute" ok \
    || { tail -2 "$W/cu_nutk/run.log"; say "CUDA arm: still runs nutkWallFunction, which its closure does compute" FAIL; }
for t in nutUWallFunction nutLowReWallFunction; do
    stage "$W/cu_$t" "$t"; runBrae cuda "$W/cu_$t"
    [ "$(its "$W/cu_$t")" = 0 ] && grep -q "$t" "$W/cu_$t/run.log" \
        && say "CUDA arm: refuses $t by name rather than running nutk under it" ok \
        || say "CUDA arm: refuses $t by name rather than running nutk under it" FAIL
done

# ---- ARM 5: anything still outside the ported set refuses, on BOTH arms ---------------------------
for arm in 1 cuda; do
    stage "$W/sp$arm" nutUSpaldingWallFunction; runBrae "$arm" "$W/sp$arm"
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    [ "$(its "$W/sp$arm")" = 0 ] && grep -q "nutUSpalding" "$W/sp$arm/run.log" \
        && say "fail-proof ($label): an UNPORTED member still refuses, by name" ok \
        || say "fail-proof ($label): an UNPORTED member still refuses, by name" FAIL
done

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
