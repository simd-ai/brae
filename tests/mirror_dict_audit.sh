#!/usr/bin/env bash
# Both rhoSimpleFoam mirror arms report a dictionary entry they read off disk and never applied.
#
# The legacy drivers have carried DictAuditScope since item E5: every entry of controlDict, fvSolution
# and the turbulence dict that no consumer queried is printed at scope exit, on a normal return and on a
# refusal alike. The OF-mirror arms (BRAE_RHOSIMPLEFOAM_MIRROR=1 and =cuda) built Time with no audit at
# all, so an input the mirror parsed and silently ignored was invisible -- the exact class of defect
# that put a passive tracer at 6.32 against a bound of 1.0 in the legacy lineage (queue item 15).
#
#   ARMS     host mirror and CUDA mirror on validation/rhoBox, 2 iterations.
#   PLANT    `bogusUnreadEntry 1;` inside fvSolution's solvers/p block: no consumer reads it, so the
#            audit must name `solvers/p/bogusUnreadEntry` on both arms.
#   CONTROL  the same run without the plant must NOT name it (the audit reports what is unread, not
#            what exists); and the planted run must still complete its iterations.
#
# FAIL-PROOF, RUN against the pre-item-15 binary: neither arm printed a single `NOTICE [unread]` line.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/rhoBox}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <plant:0|1>
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* "$1"/0.[0-9]* 2>/dev/null
    # rhoBox ships its fields as 0.orig (OpenFOAM's restore0Dir convention); the mirror reads 0/.
    [ -d "$1/0" ] || cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "$2" <<'PY'
import re, sys
d, plant = sys.argv[1], sys.argv[2] == '1'
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 2;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 2;', s)
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
open(c, 'w').write(s)
if plant:
    f = d + '/system/fvSolution'; s = open(f).read()
    # first `p {` block inside solvers: plant right after its opening brace
    m = re.search(r'(solvers\s*\{[^{}]*?\bp\s*\{)', s, re.S)
    assert m, 'no solvers/p block to plant in'
    s = s[:m.end()] + '\n        bogusUnreadEntry 1;' + s[m.end():]
    open(f, 'w').write(s)
PY
}
for arm in "1 host" "cuda cuda"; do
    set -- $arm; sel=$1; label=$2
    P="$W/${label}_plant"; C="$W/${label}_clean"
    prep "$P" 1; prep "$C" 0
    ( cd "$P" && BRAE_RHOSIMPLEFOAM_MIRROR=$sel "$BRAE" -case "$P" > log 2>&1 ) || { tail -4 "$P/log"; say "$label  the planted run finished" FAIL; continue; }
    ( cd "$C" && BRAE_RHOSIMPLEFOAM_MIRROR=$sel "$BRAE" -case "$C" > log 2>&1 ) || { tail -4 "$C/log"; say "$label  the clean run finished" FAIL; continue; }
    n=$(grep -c '^Time = ' "$P/log"); [ "$n" -ge 2 ] && say "$label  the planted run completed its iterations ($n)" ok \
                                                      || say "$label  the planted run completed its iterations ($n)" FAIL
    grep -q 'NOTICE \[unread\] *solvers/p/bogusUnreadEntry' "$P/log" \
        && say "$label  the audit names the planted entry solvers/p/bogusUnreadEntry" ok \
        || say "$label  the audit names the planted entry solvers/p/bogusUnreadEntry" FAIL
    grep -q 'bogusUnreadEntry' "$C/log" \
        && say "$label  CONTROL: the clean run does not name it" FAIL \
        || say "$label  CONTROL: the clean run does not name it" ok
done
[ $fail -eq 0 ] && echo "PASS: both mirror arms report what they read and never applied"
exit $fail
