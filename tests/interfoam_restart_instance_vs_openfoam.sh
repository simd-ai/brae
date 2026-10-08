#!/usr/bin/env bash
# brae's adaptive interFoam CONTINUED FROM A TIME DIRECTORY, against real OpenFOAM -- the unit that makes
# brae resolve WHICH time directory each mesh file comes from instead of reading constant/polyMesh.
#
# WHAT THE LEVELS UNIT DID NOT CLAIM, and this one does. That fixture MOVED the refined 0.002/polyMesh into
# constant/ so brae could find it, and said so: "a genuine latestTime restart would need facesInstance, and
# that is named below as not claimed". Here nothing is moved. The seed leaves the case exactly as OpenFOAM
# wrote it -- an ORIGINAL mesh in constant/polyMesh and a REFINED one in the time directory -- and both codes
# continue from there.
#
# WHAT OPENFOAM DOES, and it is a SEARCH per file rather than a path (polyMesh.C:175-295, and
# Time::findInstance -> fileOperation::findInstance, fileOperation.C:1077-1190):
#
#   points                            findInstance(meshDir, "points")
#   faces                             findInstance(meshDir, "faces")
#   owner, neighbour                  faces_.instance()                          READ_IF_PRESENT
#   boundary                          findInstance(..., stop = faces instance)   "allow 'newer' boundary file"
#   pointZones, faceZones, cellZones  faces_.instance()                          READ_IF_PRESENT
#   cellLevel, pointLevel, refinementHistory   mesh_.facesInstance()   (hexRef8.C:1912-1990)
#
# The search tries the CURRENT instance first, then walks back to the last time at or before the start value
# and down from there; `constant` is index 0 of times() with value 0 and is reached last. So POINTS AND FACES
# CAN COME FROM DIFFERENT DIRECTORIES -- on a MOVING mesh they always do, because the run writes `points` into
# every time directory and leaves `faces` in constant.
#
# THE FIXTURE MUST BE ABLE TO WITNESS, and for this unit that is one thing: constant/polyMesh and
# <start>/polyMesh MUST BE DIFFERENT MESHES. The script asserts their cell counts differ (2814 against 4998
# on this seed) rather than trusting it -- a fixture whose two instances hold the same mesh would pass with
# the resolution removed.
#
# THE CONTROL, BRAE_CONTROL_MESH_CONSTANT_ONLY: take every mesh file from constant/, which is exactly what
# brae SHIPPED before this unit. It is the plausible wrong port in the strongest sense -- it was the shipped
# one -- and here it puts the run on a 2814-cell mesh where the oracle is on 4998, so the cell-count arm
# catches it before any field is compared.
#
# ...AND THE SECOND HALF, `startFrom latestTime`: braeInterFoam READ that entry and then dropped it, falling
# back to time 0 for anything that was not `startTime`. So the standard way to continue a run silently
# restarted it. The gate runs brae's solver binary on the case as written -- `startFrom latestTime`, nothing
# else changed -- and asserts it reports the time directory rather than 0.
#
# WHAT THIS GATE DOES NOT CLAIM:
#   * A MOVING mesh restart, where `points` and `faces` resolve to DIFFERENT instances. The resolver handles
#     it and mirrors polyMesh's own separation, but the fixture here refines rather than moves, so the two
#     instances are the same directory and this gate cannot witness the split. It is named in PORT.md.
#   * PARALLEL. Not worked on, by rule.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_amr_vs_openfoam"
SOLVER="${BUILD:-$ROOT/build}/brae_interFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
REFSRC="$TUT/multiphase/interFoam/laminar/damBreakWithObstacle"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -x "$SOLVER" ]   || { echo "SKIP: $SOLVER not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: laminar/damBreak tutorial not found at $SRC"; exit 77; }
[ -d "$REFSRC" ]   || { echo "SKIP: damBreakWithObstacle not found at $REFSRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1
requireFresh "$SOLVER" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for app in blockMesh setFields interFoam; do
    command -v "$app" > /dev/null 2>&1 || { echo "SKIP: $app not on PATH"; exit 77; }
done

DT=0.001
SEEDSTEPS=2
N=2
SEEDEND=$(python3 -c "print('%.10g' % ($SEEDSTEPS*float('$DT')))")
END=$(python3 -c "print('%.10g' % (($SEEDSTEPS+$N)*float('$DT')))")

rc=0

# dicts <dir> <startTime> <endTime> <startFrom>
dicts()
{
    python3 - "$1" "$DT" "$2" "$3" "$4" <<'PYEOF'
import re, sys
C, dt, start, end, startFrom = sys.argv[1:6]
p = C + '/system/controlDict'
s = open(p).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}\n', '\n', s, flags=re.S)
for key, val in [('startFrom', startFrom), ('startTime', start), ('stopAt', 'endTime'),
                 ('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', end),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '18'), ('writeCompression', 'off'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(p, 'w').write(s)

# Pinned, so the comparison is the discretisation and not the Krylov stopping point.
p = C + '/system/fvSolution'
s = open(p).read()
s, n = re.subn(r'tolerance\s+\S+;', 'tolerance       1e-14;', s)
assert n >= 3, 'expected at least 3 tolerance entries, found %d' % n
s, m = re.subn(r'relTol\s+\S+;', 'relTol          0;', s)
assert m >= 3, 'expected at least 3 relTol entries, found %d' % m
open(p, 'w').write(s)
PYEOF
}

# cells <polyMesh dir> -- the cell count, from owner/neighbour, so it needs no OpenFOAM
cells()
{
    python3 - "$1" <<'PYEOF'
import re, sys
d = sys.argv[1]
best = -1
for f in ('owner', 'neighbour'):
    t = open(d + '/' + f).read()
    body = t[t.find('// * * *'):]
    m = re.search(r'(\d+)\s*\(', body)
    j = body.find('(', m.start())
    k = body.find(')', j)
    for v in body[j + 1:k].split():
        best = max(best, int(v))
print(best + 1)
PYEOF
}

# ---- PHASE 1: the seed, and NOTHING IS MOVED afterwards ---------------------------------------------
# Two steps of real OpenFOAM on laminar/damBreak with damBreakWithObstacle's dynamicMeshDict. It refines,
# so each time directory gets a polyMesh of its own while constant/polyMesh keeps the mesh blockMesh made.
S="$W/case"
rm -rf "$S"
cp -r "$SRC" "$S" || exit 1
rm -rf "$S"/0 "$S"/processor* "$S"/log.* "$S"/0.[0-9]*
cp "$REFSRC/constant/dynamicMeshDict" "$S/constant/dynamicMeshDict" || exit 1
grep -q "dynamicFvMesh   dynamicRefineFvMesh" "$S/constant/dynamicMeshDict" \
    || { echo "FAIL: the template dynamicMeshDict no longer selects dynamicRefineFvMesh"; exit 1; }
dicts "$S" 0 "$SEEDEND" startTime || exit 1

SKEY=$(oracleKey "$S" "interfoam_restart_instance_seed" "seed" "$DT" "$SEEDSTEPS")
if oracleRestore "$S" "$SKEY" "$SEEDEND"; then
    echo "[seed] OpenFOAM's $SEEDSTEPS steps reused from the oracle cache"
else
    ( cd "$S" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh"; tail -20 "$S/log.blockMesh"; exit 1; }
    cp -r "$S/0.orig" "$S/0" || exit 1
    ( cd "$S" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; exit 1; }
    ( cd "$S" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam (seed)"; tail -30 "$S/log.interFoam"; exit 1; }
    oracleStore "$S" "$SKEY"
fi

# THE FIXTURE MUST BE ABLE TO WITNESS: the two instances must hold DIFFERENT meshes, or a run that ignored
# the resolution would agree with one that honoured it.
[ -f "$S/constant/polyMesh/faces" ] \
    || { echo "FAIL: constant/polyMesh has no faces, so there is nothing for the wrong answer to be"; exit 1; }
[ -f "$S/$SEEDEND/polyMesh/faces" ] \
    || { echo "FAIL: the seed wrote no $SEEDEND/polyMesh, so this gate has no instance to resolve"; exit 1; }
NC_CONST=$(cells "$S/constant/polyMesh") || exit 1
NC_START=$(cells "$S/$SEEDEND/polyMesh") || exit 1
echo "  constant/polyMesh has $NC_CONST cells; $SEEDEND/polyMesh has $NC_START"
[ "$NC_CONST" != "$NC_START" ] \
    || { echo "FAIL: both instances hold the same $NC_CONST-cell mesh, so this fixture cannot witness the"
         echo "      resolution at all -- the seed did not refine"; exit 1; }
for f in cellLevel pointLevel refinementHistory; do
    [ -f "$S/$SEEDEND/polyMesh/$f" ] \
        || { echo "FAIL: the seed wrote no $SEEDEND/polyMesh/$f, which is the state the restart must read"
             exit 1; }
    [ ! -f "$S/constant/polyMesh/$f" ] \
        || { echo "FAIL: constant/polyMesh carries a $f, so reading the wrong instance would still find one"
             exit 1; }
done

# ---- PHASE 2: the continuation. `startFrom latestTime`, nothing moved, nothing copied. ---------------
dicts "$S" 0 "$END" latestTime || exit 1

# ...AND maxRefinement 3, so the continuation CHANGES THE MESH. With the seed's 2 it selects 0 cells out of
# 4998 -- everything eligible is already at the cap -- and a gate whose run never changes topology cannot
# exercise the mapping or its controls. Raising the cap also makes the restored cellLevel load-bearing in the
# one place it decides anything: `cellLevel[celli] < maxRefinement` (dynamicRefineFvMesh.C:861). A run that
# assumed level 0 would refine cells already at level 2, which is what BRAE_CONTROL_AMR_NO_LEVELS does.
sed -i 's/^\( *maxRefinement *\)2;/\13;/' "$S/constant/dynamicMeshDict"
grep -q "maxRefinement   3;" "$S/constant/dynamicMeshDict" \
    || { echo "FAIL: could not raise maxRefinement for the continuation"; exit 1; }

RKEY=$(oracleKey "$S" "interfoam_restart_instance" "cont" "$DT" "$N")
if oracleRestore "$S" "$RKEY" "$END"; then
    echo "[cont] OpenFOAM's $N further steps reused from the oracle cache"
else
    ( cd "$S" && interFoam > log.cont 2>&1 ) \
        || { echo "FAIL: interFoam (the continuation)"; tail -30 "$S/log.cont"; exit 1; }
    [ -d "$S/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory"; exit 1; }
    oracleStore "$S" "$RKEY"
fi
grep -qE "Refined from|Unrefined from" "$S/log.cont" \
    || { echo "FAIL: the continuation changes no cells, so it exercises the mapping and its controls not at"
         echo "      all -- raise maxRefinement or lengthen the run"; rc=1; }

# ...and its one-ulp twin AT THE INTERFACE, which is what any bound here is worth. See
# tests/interfoam_amr_ulp_cell.py for why the cell is not the first one that is 1.
UR="$W/caseUlp"
rm -rf "$UR"
cp -r "$S" "$UR" || exit 1
rm -rf "$UR/$END" "$UR"/log.cont
python3 "$(dirname "$0")/interfoam_amr_ulp_cell.py" "$UR" alpha.water "$SEEDEND" || exit 1
UKEY=$(oracleKey "$UR" "interfoam_restart_instance" "contUlp" "$DT" "$N")
if oracleRestore "$UR" "$UKEY" "$END"; then
    echo "[contUlp] reused from the oracle cache"
else
    ( cd "$UR" && interFoam > log.cont 2>&1 ) \
        || { echo "FAIL: interFoam (the one-ulp twin)"; tail -20 "$UR/log.cont"; exit 1; }
    oracleStore "$UR" "$UKEY"
fi

# ---- brae, on the case as written -------------------------------------------------------------------
if [ $rc -eq 0 ]; then
    "$BIN" "$S" "$S/$SEEDEND" "$S/$END" "$N" "$S/log.cont" restartInstance "$UR/$END" || rc=1
fi

# ---- ...and the SHIPPED SOLVER, because the harness above is handed the start directory and the solver
# has to find it itself. That is the `startFrom latestTime` half: it was read and dropped.
#
# ON ITS OWN COPY, cut back to what the seed left: the oracle ran in $S and wrote $END there, so
# `latestTime` in that directory is the END time and not the restart point. A solver arm run there would
# resolve to the answer instead of to the question.
B="$W/caseBrae"
rm -rf "$B"
cp -r "$S" "$B" || exit 1
rm -rf "$B/$END" "$B"/log.* "$B"/postProcessing
for d in "$B"/0.[0-9]*; do
    [ "$(basename "$d")" = "$SEEDEND" ] && continue
    [ "$(basename "$d")" = "0.001" ] && continue
    rm -rf "$d"
done
[ -d "$B/$SEEDEND" ] || { echo "FAIL: the solver's copy has no $SEEDEND to restart from"; exit 1; }
[ ! -d "$B/$END" ] || { echo "FAIL: the solver's copy still holds the oracle's $END"; exit 1; }
if [ $rc -eq 0 ]; then
    ( cd "$B" && "$SOLVER" > log.brae 2>&1 ) || true
    grep -q "startFrom latestTime -> starting from time '$SEEDEND'" "$B/log.brae" \
        || { echo "FAIL: the solver did not resolve startFrom latestTime to $SEEDEND"
             grep -iE "start|instance" "$B/log.brae" | head -5; rc=1; }
    grep -q "mesh instances -- points '$SEEDEND', faces '$SEEDEND'" "$B/log.brae" \
        || { echo "FAIL: the solver did not resolve the mesh to the $SEEDEND instance"
             grep -iE "instance" "$B/log.brae" | head -5; rc=1; }
    grep -q "brae interFoam (OF-mirror): $NC_START cells" "$B/log.brae" \
        || { echo "FAIL: the solver did not start on the $NC_START-cell mesh"
             grep -E "OF-mirror" "$B/log.brae" | head -3; rc=1; }
    # ...and the control, on the solver, where it is loudest: the wrong instance is the wrong mesh
    ( cd "$B" && BRAE_CONTROL_MESH_CONSTANT_ONLY=1 "$SOLVER" > log.braeControl 2>&1 ) || true
    grep -q "brae interFoam (OF-mirror): $NC_CONST cells" "$B/log.braeControl" \
        || { echo "FAIL: the control did NOT put the solver on constant/polyMesh's $NC_CONST-cell mesh, so"
             echo "      it is not a control"; grep -E "OF-mirror" "$B/log.braeControl" | head -3; rc=1; }
fi

echo "interfoam_restart_instance_vs_openfoam: rc $rc"
exit $rc
