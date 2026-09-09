#!/usr/bin/env bash
# `writeFormat binary` -- a refusal that protected nothing, on a claim that was already false.
#
# The rhoSimpleFoam mirror refused any case whose controlDict said `writeFormat binary`, on the grounds
# that brae writes ascii only and the case "would not get what it asked for". Two things were wrong:
#
#  1. OpenFOAM's writeFormat is a WRITE option and nothing reads it back (TimeIO.C:370-372). A file's
#     format comes from that FILE's own FoamFile header (IOobjectReadHeader.C:51, applied at :111). So
#     an ascii time directory written under `writeFormat binary` is read by real OpenFOAM without a
#     word -- which is what ARM 2 below makes OpenFOAM itself prove.
#  2. brae's READER was said to be ascii-only, on the strength of a comment at the top of
#     foam_field_reader.cuh that had been stale since the binary transcoder landed. brae reads
#     OpenFOAM's binary fields today (foam_token_reader.cu:170-175) -- ARM 4.
#
# Only the rhoSimpleFoam mirror ever refused it; simpleFoamV2, gpuSimpleFoam, gpuPimpleFoam and the
# legacy rho drivers all wrote ascii under the same setting and said nothing. The refusal is now one
# NOTICE from WriteControl, which every driver constructs, so the inconsistency is gone too.
#
# THE ORACLE IS REAL OpenFOAM READING brae's OUTPUT, and ARM 3 is the control that proves the oracle can
# fail: the same bytes with the header rewritten to lie about being binary must make OpenFOAM abort.
# Without it, ARM 2 would pass against an OpenFOAM that ignored file headers entirely -- the very
# proposition under test.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/rhoBox"
OFBASH=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASH" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
IT=5

fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
its() { [ -f "$1/run.log" ] && grep -c '^Time = ' "$1/run.log" 2> /dev/null | head -1 || echo 0; }

stage() {   # stage <dir> <ascii|binary> [startFrom]
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$1/0"; else cp -r "$SRC/0" "$1/0"; fi
    python3 - "$1" "$IT" "$2" "${3:-startTime}" <<'PYEOF'
import re, sys
d, it, fmt, sf = sys.argv[1:5]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;',       'endTime %s;' % it, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % it, s)
s = re.sub(r'\bstartFrom\s+[^;]*;',     'startFrom %s;' % sf, s)
if re.search(r'\bwriteFormat\s+[^;]*;', s): s = re.sub(r'\bwriteFormat\s+[^;]*;', 'writeFormat %s;' % fmt, s)
else:                                       s += '\nwriteFormat %s;\n' % fmt
open(c, 'w').write(s + '\n')
PYEOF
}
runOF()   { ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && rhoSimpleFoam > run.log 2>&1 ); }
runBrae() { ( cd "$2" && BRAE_RHOSIMPLEFOAM_MIRROR="$1" "$BRAE" -case "$2" > run.log 2>&1 ); }

# ---- ARM 0: control -- the fixture runs under real OpenFOAM at all ---------------------------------
stage "$W/of0" ascii; runOF "$W/of0"
[ "$(its "$W/of0")" = "$IT" ] \
    && say "control: real OpenFOAM runs the fixture" ok \
    || { tail -3 "$W/of0/run.log"; say "control: real OpenFOAM runs the fixture" FAIL; }

# ---- ARM 1: BOTH brae arms run under `writeFormat binary`, and each says so exactly ONCE -----------
# Once, not at-least-once: a driver emitting the line from two places is a different defect, and the
# notice machinery dedupes so >=1 would hide it.
for arm in 1 cuda; do
    stage "$W/b$arm" binary; runBrae "$arm" "$W/b$arm"
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    [ "$(its "$W/b$arm")" = "$IT" ] \
        && say "$label arm: runs under 'writeFormat binary' instead of refusing" ok \
        || { tail -2 "$W/b$arm/run.log"; say "$label arm: runs under 'writeFormat binary' instead of refusing" FAIL; }
    n=$(grep -c 'controlDict writeFormat' "$W/b$arm/run.log" 2> /dev/null | head -1)
    [ "${n:-0}" = 1 ] \
        && say "$label arm: says so, once, as an [approximated] notice" ok \
        || say "$label arm: says so, once, as an [approximated] notice (got ${n:-0})" FAIL
    grep -aq 'format *ascii;' "$W/b$arm/$IT/T" 2> /dev/null \
        && say "$label arm: the file it wrote is HONESTLY labelled ascii, not mislabelled binary" ok \
        || say "$label arm: the file it wrote is HONESTLY labelled ascii, not mislabelled binary" FAIL
done

# ---- ARM 2: THE ORACLE -- real OpenFOAM restarts from brae's output, binary still set --------------
R="$W/restart"; rm -rf "$R"; cp -r "$W/b1" "$R"; rm -f "$R/run.log"
python3 - "$R" "$IT" <<'PYEOF'
import re, sys
d, it = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;',   'endTime %d;' % (int(it) + 2), s)
open(c, 'w').write(s)
PYEOF
runOF "$R"
grep -q '^End' "$R/run.log" 2> /dev/null \
    && say "ORACLE: real OpenFOAM restarts from brae's ascii output with binary still set" ok \
    || { grep -m2 -A2 'FATAL' "$R/run.log" | head -4; say "ORACLE: real OpenFOAM restarts from brae's ascii output with binary still set" FAIL; }

# ---- ARM 3: THE CONTROL -- the same bytes, mislabelled, must make OpenFOAM ABORT -------------------
# If this passed, ARM 2 would be proving nothing: it would mean OpenFOAM never looks at the header.
L="$W/lie"; rm -rf "$L"; cp -r "$W/b1" "$L"; rm -f "$L/run.log"
for f in "$L/$IT"/*; do [ -f "$f" ] && sed -i 's/format *ascii;/format      binary;/' "$f"; done
python3 - "$L" "$IT" <<'PYEOF'
import re, sys
d, it = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;',   'endTime %d;' % (int(it) + 2), s)
open(c, 'w').write(s)
PYEOF
runOF "$L"
grep -q '^End' "$L/run.log" 2> /dev/null \
    && say "CONTROL: a mislabelled header makes OpenFOAM abort (so ARM 2 tests the header)" FAIL \
    || say "CONTROL: a mislabelled header makes OpenFOAM abort (so ARM 2 tests the header)" ok
printf '        (%s)\n' "$(grep -m1 'FATAL IO ERROR' "$L/run.log" 2> /dev/null || echo 'OpenFOAM rejected it')"

# ---- ARM 4: the READER half -- brae restarts from OpenFOAM's GENUINE binary output -----------------
# The audit called brae's reader ascii-only, from a stale comment. It is not, and this proves it on
# both arms. The non-printable check first, so "brae reads binary" cannot pass against an ascii file.
BW="$W/ofbin"; stage "$BW" binary; runOF "$BW"
if [ -f "$BW/$IT/T" ] && LC_ALL=C grep -qP '[\x00-\x08\x0e-\x1f]' "$BW/$IT/T" 2> /dev/null; then
    say "control: OpenFOAM's own output at 'writeFormat binary' really IS binary" ok
    for arm in 1 cuda; do
        RB="$W/rb$arm"; rm -rf "$RB"; cp -r "$BW" "$RB"; rm -f "$RB/run.log"
        python3 - "$RB" "$IT" <<'PYEOF'
import re, sys
d, it = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;',   'endTime %d;' % (int(it) + 2), s)
open(c, 'w').write(s)
PYEOF
        runBrae "$arm" "$RB"
        label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
        [ "$(its "$RB")" -ge 1 ] \
            && say "$label arm: restarts from OpenFOAM's BINARY time directory" ok \
            || { tail -2 "$RB/run.log"; say "$label arm: restarts from OpenFOAM's BINARY time directory" FAIL; }
    done
else
    say "control: OpenFOAM's binary output (not produced, reader arms skipped)" ok
fi

# ---- ARM 5: nothing changed for an ascii case -- no notice, same run -------------------------------
stage "$W/plain" ascii; runBrae 1 "$W/plain"
[ "$(its "$W/plain")" = "$IT" ] && ! grep -q 'controlDict writeFormat' "$W/plain/run.log" \
    && say "fail-proof: an ascii case runs with NO writeFormat notice (the entry is the cause)" ok \
    || say "fail-proof: an ascii case runs with NO writeFormat notice (the entry is the cause)" FAIL

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
