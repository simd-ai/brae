#!/usr/bin/env bash
# fv::limitTemperature -- an fvOption brae IMPLEMENTS ON BOTH ARMS and refused anyway.
#
# aerofoilNACA0012 is the only rhoSimpleFoam tutorial whose sole blocker was a capability brae already
# had. deriveCaseRefusals resolves limitTemperature OUT of the option list (it sets limitT/limitTmin/
# limitTmax and applies the clamp itself), so fvOptions::read's catch-all marked the type unsupported --
# and TWO separate paths then promoted that mark back into a refusal: firstUnsupported() on the host arm
# and the CUDA driver's own option loop. Fixing one left the other refusing.
#
# THE ORACLE IS REAL OpenFOAM'S OWN REPORT LINES, not a field comparison. limitTemperature.C:200-215
# prints, on every call:
#
#     limitTemperature=limitT, Type=Lower, LimitedCells=508, CellsPercent=0.45, Tmin=900, UnlimitedTmin=849
#
# and LimitedCells is a COUNT -- an integer that either matches OpenFOAM's or does not. It is a far
# sharper oracle than a converged field: it says how many cells the option touched, at that iteration,
# on that arm. brae printed nothing at all, so a clamp that moved every cell looked exactly like one
# that moved none.
#
# NOT VACUOUS: the limits are chosen so the clamp BITES (ARM 0 proves the counts are non-zero and that
# they are zero without them), the fail-proofs cover both refusal paths and the `active` switch, and the
# CUDA arm is checked separately because it refused through its own code path.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/sbMatched"
OFBASH=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASH" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
IT=8
# The case's own T runs 849..1147 at iteration 8, so these two cut into it from both ends. A limit
# OUTSIDE that range is satisfied by a run that ignores the option entirely -- which is how this would
# pass while doing nothing.
TMIN=900
TMAX=1100

fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage() {   # stage <dir> <fvOptions body, or "" for none>
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"; cp -r "$SRC/0.orig" "$1/0"
    python3 - "$1" "$IT" <<'PYEOF'
import re, sys
d, it = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % it, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % it, s)
open(c, 'w').write(s + '\n')
PYEOF
    rm -f "$1/system/fvOptions"
    [ -n "$2" ] && printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\n%s\n' "$2" > "$1/system/fvOptions"
    return 0
}
OPT="limitT { type limitTemperature; min $TMIN; max $TMAX; selectionMode all; }"
OPTOFF="limitT { type limitTemperature; min $TMIN; max $TMAX; selectionMode all; active false; }"
OPTBAD="mvf { type meanVelocityForce; selectionMode all; fields (U); Ubar (10 0 0); }"

runOF()   { ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && rhoSimpleFoam > run.log 2>&1 ); }
runBrae() { ( cd "$2" && BRAE_RHOSIMPLEFOAM_MIRROR="$1" "$BRAE" -case "$2" > run.log 2>&1 ); }
# `grep -c` PRINTS 0 and EXITS 1 when there is no match, so a `|| echo 0` fallback appends a SECOND
# zero and every "did this arm refuse" test compares "0\n0" against "0" and fails. The refusal arms of
# this gate reported FAIL against a brae that was refusing correctly.
its()  { [ -f "$1/run.log" ] && grep -c '^Time = ' "$1/run.log" 2> /dev/null | head -1 || echo 0; }
# The whole report, in order: the sequence of counts is what must match, not just their sum.
counts() { grep '^limitTemperature=' "$1/run.log" 2> /dev/null | sed -E 's/.*Type=([A-Za-z]+), LimitedCells=([0-9]+).*/\1:\2/' | tr '\n' ' '; }
nonzero() { counts "$1" | grep -qE ':[1-9]'; }

# ---- ARM 0: the ORACLE, and the control that proves the limits bite --------------------------------
stage "$W/of" "$OPT";      runOF "$W/of"
stage "$W/ofNo" "";        runOF "$W/ofNo"
[ "$(its "$W/of")" = "$IT" ] \
    && say "control: real OpenFOAM runs the case with the option" ok \
    || { tail -3 "$W/of/run.log"; say "control: real OpenFOAM runs the case with the option" FAIL; }
nonzero "$W/of" \
    && say "control: the limits BITE -- OpenFOAM reports non-zero LimitedCells" ok \
    || say "control: the limits BITE -- OpenFOAM reports non-zero LimitedCells" FAIL
[ -z "$(counts "$W/ofNo")" ] \
    && say "control: with no fvOptions OpenFOAM reports nothing (the option is the cause)" ok \
    || say "control: with no fvOptions OpenFOAM reports nothing (the option is the cause)" FAIL
OFC="$(counts "$W/of")"
printf '        (OpenFOAM: %s)\n' "$OFC"

# ---- ARM 1 + 2: both brae arms run it, and report the SAME COUNTS ----------------------------------
for arm in 1 cuda; do
    stage "$W/b$arm" "$OPT"; runBrae "$arm" "$W/b$arm"
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    [ "$(its "$W/b$arm")" = "$IT" ] \
        && say "$label arm: runs the case instead of refusing the option it implements" ok \
        || { tail -2 "$W/b$arm/run.log"; say "$label arm: runs the case instead of refusing the option it implements" FAIL; }
    [ "$(counts "$W/b$arm")" = "$OFC" ] \
        && say "$label arm: every LimitedCells count equals OpenFOAM's, in order" ok \
        || say "$label arm: every LimitedCells count equals OpenFOAM's, in order" FAIL
    printf '        (%s: %s)\n' "$label" "$(counts "$W/b$arm")"
done

# ---- ARM 3: the report line's FORMAT, byte for byte against OpenFOAM's ------------------------------
# The counts could match while the line is unparseable by anything diffing the two logs.
of1=$(grep -m1 '^limitTemperature=' "$W/of/run.log" | sed -E 's/UnlimitedT(min|max)=.*/UnlimitedT\1=/')
b1=$(grep -m1 '^limitTemperature=' "$W/b1/run.log"  | sed -E 's/UnlimitedT(min|max)=.*/UnlimitedT\1=/')
[ -n "$of1" ] && [ "$of1" = "$b1" ] \
    && say "the report line matches OpenFOAM's format byte for byte" ok \
    || say "the report line matches OpenFOAM's format byte for byte" FAIL
printf '        (OF  : %s)\n        (brae: %s)\n' "$of1" "$b1"

# ---- ARM 4: THE TUTORIAL, as shipped ---------------------------------------------------------------
# aerofoilNACA0012 is what this whole item is about. Its own fvOptions asks min 101 -- a limit that never
# bites -- so it tests the REFUSAL, not the clamp; the counts above test the clamp.
set +u; . "$OFBASH" > /dev/null 2>&1; set -u
TUT="$FOAM_TUTORIALS/compressible/rhoSimpleFoam/aerofoilNACA0012"
if [ -d "$TUT" ] && [ -f "$FOAM_TUTORIALS/resources/geometry/NACA0012.obj.gz" ]; then
    T="$W/naca"; rm -rf "$T"; mkdir -p "$T/constant/geometry"
    cp -r "$TUT/constant" "$TUT/system" "$T/"; cp -r "$TUT/0.orig" "$T/0"
    cp -f "$FOAM_TUTORIALS/resources/geometry/NACA0012.obj.gz" "$T/constant/geometry/"
    python3 - "$T" <<'PYEOF'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 5;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 5;', s)
open(c, 'w').write(s + '\n')
PYEOF
    ( cd "$T" && blockMesh > log.bm 2>&1 && transformPoints -scale '(1 0 1)' > log.tp 2>&1 \
       && extrudeMesh > log.em 2>&1 && topoSet > log.ts 2>&1 )
    if [ -f "$T/constant/polyMesh/owner" ]; then
        runBrae 1 "$T"
        [ "$(its "$T")" = 5 ] \
            && say "aerofoilNACA0012 (host): the tutorial runs as shipped" ok \
            || { tail -2 "$T/run.log"; say "aerofoilNACA0012 (host): the tutorial runs as shipped" FAIL; }
        grep -q '^limitTemperature=limitT,' "$T/run.log" \
            && say "...and names the option by its DICT KEY, as OpenFOAM does" ok \
            || say "...and names the option by its DICT KEY, as OpenFOAM does" FAIL
    else
        say "aerofoilNACA0012 (mesh tools unavailable, skipped)" ok
    fi
else
    say "aerofoilNACA0012 (tutorial or geometry missing, skipped)" ok
fi

# ---- ARM 5: FAIL-PROOF -- the catch-all must still refuse what brae does NOT implement -------------
# Both arms, because both had their own path back to the refusal. If the exemption were unconditional
# this arm would run and every arm above would be worthless.
for arm in 1 cuda; do
    stage "$W/bad$arm" "$OPTBAD"; runBrae "$arm" "$W/bad$arm"
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    [ "$(its "$W/bad$arm")" = 0 ] && grep -q "meanVelocityForce" "$W/bad$arm/run.log" \
        && say "fail-proof ($label): an UNIMPLEMENTED fvOption still refuses, by name" ok \
        || say "fail-proof ($label): an UNIMPLEMENTED fvOption still refuses, by name" FAIL
done

# ---- ARM 6: FAIL-PROOF -- 'active false' --------------------------------------------------------
# OpenFOAM constructs the option and reads its dictionary but fvOptionList gates correct() on
# isActive() (fvOptionListTemplates.C:386), so an inactive option clamps NOTHING. brae's dict walk did
# not read the key at all, so it applied a clamp OpenFOAM skips.
stage "$W/offOF" "$OPTOFF"; runOF "$W/offOF"
stage "$W/off1"  "$OPTOFF"; runBrae 1 "$W/off1"
[ -z "$(counts "$W/offOF")" ] \
    && say "control: 'active false' makes OpenFOAM report nothing" ok \
    || say "control: 'active false' makes OpenFOAM report nothing" FAIL
[ -z "$(counts "$W/off1")" ] && [ "$(its "$W/off1")" = "$IT" ] \
    && say "fail-proof: brae skips an inactive option too, and still runs" ok \
    || say "fail-proof: brae skips an inactive option too, and still runs" FAIL
# ...and it must be the SAME field OpenFOAM gets, not merely a quiet one.
tOF=$(python3 - "$W/offOF/$IT/T" 2>/dev/null <<'PYEOF'
import re, sys
b = open(sys.argv[1], 'rb').read()
m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(', b)
print('%.6g' % min(float(x) for x in b[m.end():].split(b')\n', 1)[0].split()) if m else 'nan')
PYEOF
)
tBR=$(python3 - "$W/off1/$IT/T" 2>/dev/null <<'PYEOF'
import re, sys
b = open(sys.argv[1], 'rb').read()
m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(', b)
print('%.6g' % min(float(x) for x in b[m.end():].split(b')\n', 1)[0].split()) if m else 'nan')
PYEOF
)
python3 -c "
import sys
a, b = sys.argv[1:3]
try: ok = abs(float(a) - float(b)) / max(abs(float(b)), 1e-30) < 5e-3
except Exception: ok = False
sys.exit(0 if ok else 1)" "$tBR" "$tOF" \
    && say "...and its T floor is OpenFOAM's UNCLAMPED one, not the limit" ok \
    || say "...and its T floor is OpenFOAM's UNCLAMPED one, not the limit" FAIL
printf '        (min T inactive: brae %s vs OpenFOAM %s; the limit it must NOT have taken is %s)\n' "$tBR" "$tOF" "$TMIN"

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
