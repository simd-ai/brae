#!/usr/bin/env bash
# WHERE EVERY interFoam TUTORIAL STANDS, host arm and device arm.
#
# Not a validation: it does not compare to OpenFOAM. It answers the prior question -- does brae RUN this
# case at all, or REFUSE it by name, or FAIL -- for all 44 cases OpenFOAM ships, so that "what is left to
# close interFoam" is a list rather than an impression. A refusal is a PASS here: it is the project's
# contract working. Only a crash, a hang or a wrong-looking finish is a gap.
#
# Each case is meshed the way its own Allrun meshes it (blockMesh, then the optional topoSet /
# createBaffles / setFields / extrudeMesh steps its system/ directory asks for), capped at a few steps,
# and run under a timeout. snappyHexMesh cases are skipped by name with the reason, not silently.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae}"
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/multiphase/interFoam
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
STEPS=${STEPS:-2}
CASE_TIMEOUT=${CASE_TIMEOUT:-180}
MESH_TIMEOUT=${MESH_TIMEOUT:-240}
W=${KEEP_W:-$(mktemp -d)}
ONLY=${ONLY:-}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM to mesh with"; exit 77; }
set +u; . "$OFBASHRC" > /dev/null 2>&1 || true; set -u

printf '%-46s %-10s %s\n' CASE ARM OUTCOME
printf '%-46s %-10s %s\n' "---" "---" "---"

for cd_ in $(find "$TUT" -name controlDict -path '*/system/*' | sort); do
    rel=${cd_#"$TUT"/}; rel=${rel%/system/controlDict}
    # ONLY is an EXTENDED REGEX, matched with grep: a `case` pattern cannot take alternation from a
    # variable (the `|` stays literal), which silently selected nothing at all.
    [ -z "$ONLY" ] || printf '%s\n' "$rel" | grep -qE "$ONLY" || continue
    src="$TUT/$rel"
    # snappyHexMesh cases: the mesh alone outruns this sweep's budget, and a sweep that silently
    # dropped them would read as coverage it does not have.
    if [ -f "$src/system/snappyHexMeshDict" ]; then
        printf '%-46s %-10s %s\n' "$rel" "-" "SKIPPED (snappyHexMesh: mesh cost outside this sweep)"
        continue
    fi
    c="$W/$(echo "$rel" | tr / _)"
    rm -rf "$c"; cp -r "$src" "$c" || continue
    rm -rf "$c"/processor* "$c"/log.*
    [ -d "$c/0.orig" ] && { rm -rf "$c/0"; cp -r "$c/0.orig" "$c/0"; }

    mesh_ok=1
    (
        cd "$c" || exit 1
        # the two mesh steps these tutorials' own Allrun scripts take before blockMesh: an m4 template
        # (the sloshingTank family) and a case-supplied Allrun.pre (mixerVessel2D, waterChannel). A
        # sweep that skipped them reported seven cases as "no mesh" when the mesh was simply not built.
        [ -f system/blockMeshDict.m4 ] && timeout "$MESH_TIMEOUT" m4 system/blockMeshDict.m4 > system/blockMeshDict 2> log.m4
        if [ -x ./Allrun.pre ]; then
            timeout "$MESH_TIMEOUT" ./Allrun.pre > log.allrunpre 2>&1
        fi
        [ -f system/blockMeshDict ] || [ -f constant/polyMesh/blockMeshDict ] && [ ! -f constant/polyMesh/owner ] && timeout "$MESH_TIMEOUT" blockMesh > log.blockMesh 2>&1
        [ -f system/extrudeMeshDict ]   && timeout "$MESH_TIMEOUT" extrudeMesh    > log.extrude 2>&1
        [ -f system/topoSetDict ]       && timeout "$MESH_TIMEOUT" topoSet        > log.topoSet 2>&1
        [ -f system/createBafflesDict ] && timeout "$MESH_TIMEOUT" createBaffles -overwrite > log.baffles 2>&1
        [ -f system/setFieldsDict ]     && timeout "$MESH_TIMEOUT" setFields      > log.setFields 2>&1
        exit 0
    ) || mesh_ok=0
    # ...or owner.gz: these tutorials set `writeCompression on`, and looking only for the plain file
    # reported seven MESHED cases as "no mesh".
    if [ ! -f "$c/constant/polyMesh/owner" ] && [ ! -f "$c/constant/polyMesh/owner.gz" ]; then
        printf '%-46s %-10s %s\n' "$rel" "-" "NO MESH (its Allrun needs a step this sweep does not run)"
        continue
    fi

    # a few steps only: this asks whether the case RUNS, not where it ends up
    python3 - "$c" "$STEPS" <<'PY' 2>/dev/null || true
import re, sys
c, n = sys.argv[1], int(sys.argv[2])
p = c + '/system/controlDict'
s = open(p).read()
m = re.search(r'^deltaT\s+([0-9.eE+-]+)\s*;', s, re.M)
dt = float(m.group(1)) if m else 1e-4
s = re.sub(r'^endTime\s+.*$', 'endTime         %.10g;' % (n*dt), s, flags=re.M)
s = re.sub(r'^writeInterval\s+.*$', 'writeInterval   %.10g;' % (n*dt), s, flags=re.M)
s = re.sub(r'^adjustTimeStep\s+.*$', 'adjustTimeStep  no;', s, flags=re.M)
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
open(p, 'w').write(s)
PY

    for arm in host device; do
        [ "$arm" = device ] && flag="-device" || flag=""
        out=$(cd "$c" && timeout "$CASE_TIMEOUT" "$BIN" -case "$c" $flag 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then
            verdict="RUNS"
        elif [ $rc -eq 124 ]; then
            verdict="TIMEOUT (${CASE_TIMEOUT}s)"
        else
            # brae states a refusal as a paragraph opening `brae <component>: ...`. Match THAT, not a
            # keyword: the wording varies ("Only X is ported", "asks for", "Refusing rather than"), and
            # a keyword list silently reported `exit 1` for ten cases whose reason was printed in full.
            why=$(printf '%s\n' "$out" | grep -E "^brae (ERROR|[A-Za-z]+ ?[A-Za-z]*:)" \
                  | grep -vE "controlDict application|\(OF-mirror\)|NOTICE" | head -1 | cut -c1-150)
            [ -n "$why" ] || why=$(printf '%s\n' "$out" | grep -viE "^ |NOTICE" | grep -E "[a-z]" | tail -1 | cut -c1-150)
            [ -n "$why" ] || why="exit $rc"
            verdict="REFUSED/ERR: $why"
        fi
        printf '%-46s %-10s %s\n' "$rel" "$arm" "$verdict"
    done
    rm -rf "$c"
done
