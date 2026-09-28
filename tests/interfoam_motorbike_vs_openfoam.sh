#!/usr/bin/env bash
# brae's adaptive interFoam on RAS/motorBike ITSELF, against real OpenFOAM -- the case the levels, instance,
# RAS and autoMap units were each built for.
#
# WHAT MAKES IT DIFFERENT from every adaptive fixture before it, and all four are the mesh:
#   * a snappyHexMesh mesh, at levels 0 to 3, where a seeded damBreak is at 0 to 2
#   * cellLevel and pointLevel written BINARY, because the case's own writeFormat is binary
#   * NO refinementHistory: Allrun.pre removes it, so the run starts from a mesh that is refined and has no
#     record of how -- which is hexRef8's READ_IF_PRESENT fallback with the levels PRESENT
#   * 61 patches, 60 of them walls off the motorbike surface, against a damBreak's five
# and one thing that is not the mesh: kOmegaSST, on a mesh that changes under it.
#
# THE MESH IS AN INPUT, PRODUCED ONCE BY REAL OPENFOAM. snappyHexMesh is not ported and is not going to be:
# it is a mesher, not a solver, and brae's subject is interFoam. Allrun.pre ships the serial path commented
# out (`#runApplication snappyHexMesh -overwrite`), which is what this uses -- so no part of this gate is
# parallel. MEASURED: 1.53 s and 8655 cells. maxLocalCells is 100000 and in serial that IS the global cap,
# so this mesh is smaller than the tutorial's parallel 2M-cell one; both codes run the SAME mesh, which is
# what the comparison needs, and the cap is named here rather than left to look like the tutorial's number.
#
# THE FIXTURE MUST BE ABLE TO WITNESS, and for this case that is three things, each asserted below rather
# than trusted: the levels must be NON-UNIFORM (a uniform level makes the whole levels question invisible,
# because consistentRefinement only differs where levels differ), they must be BINARY (the ASCII branch is
# already covered by the levels gate), and the run must CHANGE THE MESH.
#
# WHAT IT FOUND, and neither was a missing port -- both were brae refusing or reporting too much:
#   * snappyHexMesh writes a pointZone `frozenPoints` with `pointLabels List<label> 0` -- an entry with NO
#     POINTS IN IT -- and brae counted zone ENTRIES, so an empty zone refused the case. resetZones sizes the
#     new addressing from each zone's own membership (polyTopoChange.C:1612-1700), so a zone with no members
#     comes out of a change empty with its name and type: there is nothing to renumber. The refusal now
#     counts zones that HAVE members, which is what it is for.
#   * the refinement-state diagnostic printed `parent.size()` as a count of split cells. That list has one
#     entry per cell, filled with -1 (refinementHistory.C:392-412), so a mesh with no history at all reported
#     "8655 split cell(s)" on 8655 cells. It now counts entries that have a parent.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_amr_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/motorBike"
GEOM="$TUT/resources/geometry/motorBike.obj.gz"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/motorBike tutorial not found at $SRC"; exit 77; }
[ -f "$GEOM" ]     || { echo "SKIP: motorBike.obj.gz not found at $GEOM"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for app in surfaceFeatureExtract blockMesh snappyHexMesh setFields interFoam; do
    command -v "$app" > /dev/null 2>&1 || { echo "SKIP: $app not on PATH"; exit 77; }
done

DT=0.001
N=2
END=$(python3 -c "print('%.10g' % ($N*float('$DT')))")
rc=0

C="$W/case"
rm -rf "$C"
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/0 "$C"/processor* "$C"/log.* "$C"/0.[0-9]*
mkdir -p "$C/constant/triSurface"
cp -f "$GEOM" "$C/constant/triSurface/" || exit 1

# The case's own settings, with the two a comparison needs pinned: a FIXED step (so the two codes take the
# same steps rather than each choosing its own from a Courant number) and the linear solvers at 1e-14 with
# relTol 0 (so what is compared is the discretisation and not the Krylov stopping point). `functions{}` goes
# because its fieldMinMax is diagnostic output that brae does not carry.
python3 - "$C" "$DT" "$END" <<'PYEOF' || exit 1
import re, sys
C, dt, end = sys.argv[1], sys.argv[2], sys.argv[3]
p = C + '/system/controlDict'
s = open(p).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('stopAt', 'endTime'),
                 ('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', end),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'),
                 ('writePrecision', '18'), ('writeCompression', 'off'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
# writeFormat is NOT touched: it is `binary`, and that is what makes the mesh's cellLevel binary.
assert re.search(r'^writeFormat\s+binary;', s, flags=re.M), 'the case no longer writes binary'
open(p, 'w').write(s)

p = C + '/system/fvSolution'
s = open(p).read()
s, n = re.subn(r'tolerance\s+\S+;', 'tolerance       1e-14;', s)
assert n >= 3, 'expected at least 3 tolerance entries, found %d' % n
s, m = re.subn(r'relTol\s+\S+;', 'relTol          0;', s)
assert m >= 3, 'expected at least 3 relTol entries, found %d' % m
open(p, 'w').write(s)
PYEOF

MKEY=$(oracleKey "$C" "interfoam_motorbike" "mesh" "$DT" "$N")
if oracleRestore "$C" "$MKEY" "$END"; then
    echo "[motorBike] the mesh and OpenFOAM's $N steps reused from the oracle cache"
else
    ( cd "$C" && surfaceFeatureExtract > log.sfe 2>&1 ) \
        || { echo "FAIL: surfaceFeatureExtract"; tail -20 "$C/log.sfe"; exit 1; }
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh"; tail -20 "$C/log.blockMesh"; exit 1; }
    # SERIAL, which is the path Allrun.pre ships commented out. Nothing here is parallel.
    ( cd "$C" && snappyHexMesh -overwrite > log.snappy 2>&1 ) \
        || { echo "FAIL: snappyHexMesh"; tail -30 "$C/log.snappy"; exit 1; }
    grep -q "Finished meshing without any errors" "$C/log.snappy" \
        || { echo "FAIL: snappyHexMesh did not finish cleanly"; tail -20 "$C/log.snappy"; exit 1; }
    # ...and the history removed, as Allrun.pre removes it
    rm -f "$C"/constant/polyMesh/refinementHistory*
    cp -r "$C/0.orig" "$C/0" || exit 1
    ( cd "$C" && setFields > log.setFields 2>&1 ) \
        || { echo "FAIL: setFields"; tail -20 "$C/log.setFields"; exit 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam (the oracle)"; tail -30 "$C/log.interFoam"; exit 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory"; exit 1; }
    oracleStore "$C" "$MKEY"
fi

# ---- THE THREE WITNESS ARMS, before anything is compared ---------------------------------------------
[ ! -f "$C/constant/polyMesh/refinementHistory" ] \
    || { echo "FAIL: the mesh still carries a refinementHistory, so this fixture is not motorBike's shape"
         exit 1; }
grep -q "format *binary" "$C/constant/polyMesh/cellLevel" \
    || { echo "FAIL: cellLevel is not binary, so the binary branch this fixture exists for is not exercised"
         exit 1; }
python3 - "$C/constant/polyMesh/cellLevel" <<'PYEOF' || exit 1
# The levels must be NON-UNIFORM: consistentRefinement's 2:1 constraint only differs where levels differ, so
# a uniform level -- even uniformly 3 -- would make a port that read nothing agree with one that read this.
# Binary labelList: the count, then '(', then count*4 bytes.
import struct, sys
b = open(sys.argv[1], 'rb').read()
i = b.find(b'// * * *')
j = b.find(b'(', i)
# between the separator and the '(' there is the separator's own line and then the count, so the count is the
# LAST whitespace-separated token there -- taking the text after the nearest newline gives the empty string,
# because that newline is the one immediately before the '('.
n = int(b[i:j].split()[-1])
vals = struct.unpack('<%di' % n, b[j + 1:j + 1 + 4 * n])
seen = sorted(set(vals))
print('  cellLevel over %d cells: levels %s' % (n, seen))
assert len(seen) > 1, 'the mesh is at ONE level, so this fixture cannot witness the levels at all'
PYEOF
grep -qE "Refined from|Unrefined from" "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM changed no cells in $N steps, so the mapping is not exercised"; rc=1; }
grep -E "Refined from|Unrefined from" "$C/log.interFoam" | sed 's/^/  OpenFOAM: /'

if [ $rc -eq 0 ]; then
    "$BIN" "$C" "$C/0" "$C/$END" "$N" "$C/log.interFoam" motorBike || rc=1
fi

echo "interfoam_motorbike_vs_openfoam: rc $rc"
exit $rc
