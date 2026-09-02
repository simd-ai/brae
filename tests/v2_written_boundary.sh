#!/usr/bin/env bash
# simpleFoamV2 writes the SOLVED boundary values, not the start directory's.
#
# Every writeVolField call in V2's write block omitted the writer's `computedBoundary` argument, which
# makes it ECHO the template it reads. So a V2 time directory carried the 0/ boundary on every patch the
# solve had moved: on pitzDailyTurb the walls' nut was written as the 0/ placeholder instead of the
# nutkWallFunction values (measured after the fix: inlet 8.520e-04, outlet up to 1.014e-03, walls up to
# 4.258e-06). A restart from such a directory restarts from the wrong state, and any wall
# post-processing reads a number the solve never produced.
#
# THE NON-VACUITY CHECK IS THE POINT: it is not enough that a value list is present -- the template also
# has values. What proves the fix is that the written boundary DIFFERS from the 0/ file it was spliced
# into. And the restart arm proves the result is still a file OpenFOAM accepts.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-${BUILD:-$ROOT/build}/brae}"
SRC="${CASE:-$ROOT/validation/pitzDailyTurb}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-20}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC/constant" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-62s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

cp -r "$SRC"/* "$W/" 2>/dev/null || true
rm -rf "$W"/[1-9]* "$W"/processor* "$W"/log.*
sed -i "s/^endTime.*/endTime $ITERS;/;s/writeInterval.*/writeInterval $ITERS;/" "$W/system/controlDict"

( cd "$W" && BRAE_SIMPLEFOAM_V2=1 "$BIN" -case "$W" > v2.log 2>&1 ) \
    && say "simpleFoamV2 runs and writes a time directory" ok \
    || { tail -4 "$W/v2.log"; say "simpleFoamV2 runs and writes a time directory" FAIL; }
[ -d "$W/$ITERS" ] || { echo "FAIL: V2 wrote no $ITERS/"; exit 1; }

OUT="$W/$ITERS" TMPL="$W/0" python3 - <<'PYEOF' || fail=1
import os, re, sys
def patches(path):
    try: s = open(path).read()
    except OSError: return None
    i = s.find('boundaryField')
    if i < 0: return None
    out = {}
    for m in re.finditer(r'^    ([A-Za-z"][^\s]*)\s*\n\s*\{(.*?)\n    \}', s[i:], re.S | re.M):
        out[m.group(1)] = m.group(2)
    return out
ok = True
moved = 0
for fld in ('nut', 'k', 'p', 'U'):
    a = patches(os.path.join(os.environ['OUT'], fld))
    b = patches(os.path.join(os.environ['TMPL'], fld))
    if not a or not b:
        continue
    for name, body in a.items():
        tb = b.get(name)
        if tb is None or 'value' not in body:
            continue
        # the written entry must not be the template's entry verbatim wherever the solve moved it
        if body.strip() != tb.strip():
            moved += 1
    solved = sum(1 for body in a.values() if 'nonuniform List' in body)
    print('     %-4s patches with a SOLVED (nonuniform) boundary: %d' % (fld, solved))
    if solved == 0:
        print('     %-4s wrote no solved boundary at all   FAIL' % fld); ok = False
if moved == 0:
    print('     no written boundary differs from the 0/ template -- the echo is back   FAIL'); ok = False
else:
    print('     %d boundary entries differ from the template (non-vacuous)' % moved)
sys.exit(0 if ok else 1)
PYEOF
say "the written boundaries are solved, and differ from the 0/ template" "$([ $fail = 0 ] && echo ok || echo FAIL)"

# ---- and OpenFOAM still accepts the file ----------------------------------------------------------
if [ -f "$OFBASHRC" ]; then
    set +u
    # shellcheck disable=SC1091
    source "$OFBASHRC" > /dev/null 2>&1 || true
    set -u
    if command -v simpleFoam > /dev/null 2>&1; then
        # OUTSIDE $W: copying $W into a path under $W copies the copy.
        R=$(mktemp -d)
        cp -r "$W/0" "$W/constant" "$W/system" "$W/$ITERS" "$R/"
        trap 'rm -rf "$W" "$R"' EXIT
        python3 - "$R/system/controlDict" "$ITERS" <<'PYEOF'
import re, sys
c, n = sys.argv[1], int(sys.argv[2])
s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %d;' % (n + 3), s)
open(c, 'w').write(s)
PYEOF
        ( cd "$R" && simpleFoam > ofr.log 2>&1 ) && grep -q "^End" "$R/ofr.log" \
            && grep -q "^Time = $((ITERS+1))$" "$R/ofr.log" \
            && say "real OpenFOAM restarts from V2's written directory" ok \
            || { tail -5 "$R/ofr.log"; say "real OpenFOAM restarts from V2's written directory" FAIL; }
    else
        say "simpleFoam not on PATH -- restart arm skipped" SKIP
    fi
else
    say "no OpenFOAM -- restart arm skipped" SKIP
fi


# ---- the nut WALL FUNCTION the case selects, honoured rather than substituted ---------------------
# simpleFoamV2 had no selector at all: refreshBoundaryNut ran the k-based deviceBoundaryNut whatever
# 0/nut asked for, and every turbulence correct() was handed `nutWall = 0` (nutk) for its near-wall
# production. On backwardFacingStep2D (kOmegaSST + nutUBlendedWallFunction) that produced a wall nut of
# exactly 0.000e+00; with the BC honoured it reads 3.07e-03. The selector is the SAME selectNutWall the
# other drivers use, so the two cannot disagree about what the case asked for.
BFS="$ROOT/validation/backwardFacingStep2D"
if [ -d "$BFS/constant" ]; then
    B2=$(mktemp -d)
    cp -r "$BFS/0" "$BFS/constant" "$BFS/system" "$B2/"
    sed -i "s/^endTime.*/endTime $ITERS;/;s/writeInterval.*/writeInterval $ITERS;/" "$B2/system/controlDict"
    if ( cd "$B2" && BRAE_SIMPLEFOAM_V2=1 "$BIN" -case "$B2" > bfs.log 2>&1 ); then
        grep -q "nutUBlendedWallFunction (velocity-based, honoured" "$B2/bfs.log" \
            && say "V2 names the case's own nut wall function as honoured" ok \
            || { grep -i "nut wall function" "$B2/bfs.log" | head -2; \
                 say "V2 names the case's own nut wall function as honoured" FAIL; }
        python3 "$ROOT/tests/v2_nut_wallfn_arm.py" "$B2/$ITERS/nut" nutUBlendedWallFunction \
            && say "...and the wall nut it wrote is nonzero (it was exactly 0 before)" ok \
            || say "...and the wall nut it wrote is nonzero (it was exactly 0 before)" FAIL
    else
        tail -4 "$B2/bfs.log"; say "V2 runs the nutUBlended fixture" FAIL
    fi
    rm -rf "$B2"
else
    say "backwardFacingStep2D missing -- wall-function arm skipped" SKIP
fi


# ---- FIELD relaxation follows OpenFOAM's "only when named" rule -----------------------------------
# GeometricField::relax() starts at `relaxCoeff = 1` and applies a factor only when
# solution::relaxField finds one -- `found(name) || found("default")` (solution.C:320-327). V2 defaulted
# to 0.3 for p instead, so a case with no `relaxationFactors/fields` entry (backwardFacingStep2D, as
# most SIMPLEC cases are) had its pressure correction cut to a third of OpenFOAM's. Measured: p 8.47
# after one iteration against OpenFOAM's 28.07 (ratio 3.31 = 1/0.3), the outer p residual stalled at
# 1.33e-01 where OpenFOAM sits at 3.75e-02, and at 400 iterations U was 1.78e-01 from OpenFOAM against
# the legacy path's 9.15e-02. With the rule honoured: p residual 3.66e-02 and U 8.78e-02.
#
# The arm is a PAIR, because either half alone would pass on a bug: the case that names NO factor must
# report 1, and the case that names 0.3 must report 0.3.
for pair in "$ROOT/validation/backwardFacingStep2D:1" "$ROOT/validation/pitzDailyTurb:0.3"; do
    cdir="${pair%%:*}"; want="${pair##*:}"
    [ -d "$cdir/constant" ] || { say "$(basename "$cdir") missing -- relaxation arm skipped" SKIP; continue; }
    RX=$(mktemp -d)
    cp -r "$cdir"/0 "$cdir"/constant "$cdir"/system "$RX/" 2>/dev/null || cp -r "$cdir"/* "$RX/"
    sed -i "s/^endTime.*/endTime 1;/;s/^writeInterval.*/writeInterval 1;/" "$RX/system/controlDict"
    rout=$( cd "$RX" && BRAE_SIMPLEFOAM_V2=1 "$BIN" -case "$RX" 2>&1 || true )
    echo "$rout" | grep -qE "relaxation: U [0-9.]+ \(equations\), p $want \(fields\)" \
        && say "$(basename "$cdir"): p field relaxation is $want, as OpenFOAM reads it" ok \
        || { echo "$rout" | grep -i relaxation | head -1; \
             say "$(basename "$cdir"): p field relaxation is $want, as OpenFOAM reads it" FAIL; }
    rm -rf "$RX"
done

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
