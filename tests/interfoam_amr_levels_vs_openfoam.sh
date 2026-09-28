#!/usr/bin/env bash
# brae's adaptive interFoam STARTED FROM A MESH THAT IS ALREADY REFINED, against real OpenFOAM -- the unit
# that makes brae read the refinement state off disk instead of assuming a level-0 mesh. It is what
# RAS/motorBike stands on: snappyHexMesh writes a mesh at levels 1 to 3 and brae assigned 0 everywhere, so
# `maxRefinement` did not mean the same thing in the two codes.
#
# WHAT OPENFOAM READS, and it is three files and not four:
#   cellLevel   labelList, from mesh_.facesInstance()/polyMesh, READ_IF_PRESENT, falling back to
#               labelList(nCells, Zero)  (hexRef8.C:1912-1925)
#   pointLevel  the same, falling back to labelList(nPoints, Zero)  (:1926-1937)
#   refinementHistory  READ_IF_PRESENT via typeHeaderOk, and then a FATAL check that
#               visibleCells().size() == mesh_.nCells()  (:1953-1990). Its operator>> reads splitCells then
#               visibleCells and CLEARS freeSplitCells_ (refinementHistory.C:1724-1735); its operator<<
#               calls compact() first (:1737-1747), so a written history has no holes and an empty free
#               list is the consistent pair, not an omission.
#   level0Edge  IS NOT READ BY ANY OF THIS. It is READ_IF_PRESENT with a computed fallback
#               (hexRef8.C:1939-1951) and hexRef8 uses level0EdgeLength() in exactly one place --
#               consistentSlowRefinement2 (hexRef8.C:2784) -- which dynamicRefineFvMesh never calls: it
#               calls consistentRefinement (dynamicRefineFvMesh.C:895) and rolls its own buffer layers
#               (:1415). Every other user is snappyHexMesh. So brae reads three files and says why it
#               ignores the fourth, rather than reading it to look thorough.
#
# SO brae's "all zeros" WAS RIGHT for a fresh blockMesh -- that IS OpenFOAM's fallback -- and wrong only
# when the files are there. That distinction is why this is its own unit and not a bug fix.
#
# THE FIXTURE IS BUILT IN TWO PHASES, because a mesh at mixed levels has to come from somewhere:
#   PHASE 1, the seed: real OpenFOAM runs laminar/damBreak with a dynamicMeshDict for two steps. It writes
#            0.002/polyMesh with cellLevel, pointLevel, level0Edge and refinementHistory in it.
#   PHASE 2, the gate case: that directory becomes constant/polyMesh, and the fields beside it become 0/.
#            Both codes then start from it. brae reads constant/polyMesh and has no facesInstance (it
#            hard-codes that path, braeInterFoam.cu:206), so the fixture is built to put the refined mesh
#            where brae looks -- which is honest for this unit and is NOT a restart: a genuine
#            latestTime restart would need facesInstance, and that is named below as not claimed.
#
# THE FIXTURE MUST BE ABLE TO WITNESS, and for this unit that means ONE thing above all: THE SEEDED
# cellLevel MUST NOT BE UNIFORM. A uniform level -- even uniformly 3 -- makes the whole unit invisible:
# consistentRefinement's 2:1 constraint only differs where levels differ, and the >8-anchor protected set
# only fires at a 2:1 transition, so a port that read nothing would agree. MEASURED on this seed, from
# OpenFOAM's own written file: 4998 cells at levels {0: 2190, 1: 312, 2: 2496} -- three distinct levels
# with real transitions between them. The script asserts the spread rather than trusting it.
#
# AND maxRefinement STAYS AT 2, which the seed has already reached. That is what makes the control bite at
# the FIRST change rather than eventually: OpenFOAM refuses to refine the 2,496 cells already at level 2
# (`cellLevel[celli] < maxRefinement`, dynamicRefineFvMesh.C:861), and a brae that thinks they are level 0
# refines them. With maxRefinement 3 both codes would refine those cells and the two would only part later,
# through the consistency constraint -- a weaker arm for the same work.
#
# THE ORACLE is real OpenFOAM's own continuation of the gate case, and the sharpest arm is not a field: it
# is `every cell's refinement level is OpenFOAM's`, compared against the volScalarField `dumpLevel true`
# writes beside each time directory. A field at 1e-14 on a mesh whose levels have drifted would be a
# coincidence; the levels are the state the NEXT change reads.
#
# THE CONTROL, BRAE_CONTROL_AMR_NO_LEVELS: ignore the files and assign level 0 everywhere, which is exactly
# what brae did before this unit and exactly what OpenFOAM does when the files are absent. It is the
# plausible wrong port in the strongest sense -- it was the shipped one.
#
# WHAT THIS GATE DOES NOT CLAIM:
#   * A GENUINE RESTART from a latestTime directory. OpenFOAM reads the levels from
#     mesh_.facesInstance(), the time directory the FACES came from; brae hard-codes constant/polyMesh in
#     four places and has no facesInstance at all, so it cannot be pointed at 0.002 directly. The fixture
#     copies the refined mesh into constant/ instead. Naming this is the point: the reader is done, the
#     instance resolution is not.
#   * BINARY levels. motorBike's controlDict sets `writeFormat binary`, so its snappy-written cellLevel and
#     pointLevel are binary -- the same shape as the cellZones reader that was ASCII-only. The reader goes
#     through brae's format-detecting label-list path, and this fixture writes ASCII, so the binary branch
#     is covered by tests/test_refinement_state_reader.cu on a round-tripped file rather than here.
#   * A label=64 install. brae's readBinaryLabelList memcpys sizeof(label) without reading the header's
#     `arch "...label=32..."`; pre-existing, recorded, and not this unit's.
#   * motorBike itself, whose prep is parallel-only at maxGlobalCells 2000000.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_amr_vs_openfoam"
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
[ -d "$SRC" ]      || { echo "SKIP: laminar/damBreak tutorial not found at $SRC"; exit 77; }
[ -d "$REFSRC" ]   || { echo "SKIP: damBreakWithObstacle not found at $REFSRC (its dynamicMeshDict is the template)"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

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
SEEDSTEPS=1
N=3
SEEDEND=$(python3 -c "print('%.10g' % ($SEEDSTEPS*float('$DT')))")
END=$(python3 -c "print('%.10g' % ($N*float('$DT')))")

rc=0

# dicts <dir> <endTime>  -- the controlDict and fvSolution both phases share
dicts()
{
    python3 - "$1" "$DT" "$2" "${3:-ascii}" <<'PYEOF'
import re, sys
C, dt, end = sys.argv[1], sys.argv[2], sys.argv[3]
p = C + '/system/controlDict'
s = open(p).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}\n', '\n', s, flags=re.S)
fmt = sys.argv[4] if len(sys.argv) > 4 else 'ascii'
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('stopAt', 'endTime'),
                 ('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', end),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', fmt),
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

# ---- PHASE 1: the seed. Real OpenFOAM refines a fresh mesh into one at mixed levels. -----------------
S="$W/seed"
rm -rf "$S"
cp -r "$SRC" "$S" || exit 1
rm -rf "$S"/0 "$S"/processor* "$S"/log.* "$S"/0.[0-9]*
cp "$REFSRC/constant/dynamicMeshDict" "$S/constant/dynamicMeshDict" || exit 1
grep -q "dynamicFvMesh   dynamicRefineFvMesh" "$S/constant/dynamicMeshDict" \
    || { echo "FAIL: the template dynamicMeshDict no longer selects dynamicRefineFvMesh"; exit 1; }
grep -q "maxRefinement   2;" "$S/constant/dynamicMeshDict" \
    || { echo "FAIL: the template's maxRefinement is not 2, which is what makes the control bite at the"
         echo "      first change -- see the header"; exit 1; }
dicts "$S" "$SEEDEND" || exit 1

SKEY=$(oracleKey "$S" "interfoam_amr_levels_seed" "seed" "$DT" "$SEEDSTEPS")
if oracleRestore "$S" "$SKEY" "$SEEDEND"; then
    echo "[seed] OpenFOAM's $SEEDSTEPS steps reused from the oracle cache"
else
    ( cd "$S" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh"; tail -20 "$S/log.blockMesh"; exit 1; }
    cp -r "$S/0.orig" "$S/0" || exit 1
    ( cd "$S" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; exit 1; }
    [ ! -e "$S/constant/polyMesh/cellLevel" ] \
        || { echo "FAIL: the fresh mesh already carries a cellLevel, so the seed does not start at level 0"
             exit 1; }
    ( cd "$S" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam (seed)"; tail -30 "$S/log.interFoam"; exit 1; }
    oracleStore "$S" "$SKEY"
fi
for f in cellLevel pointLevel refinementHistory; do
    [ -f "$S/$SEEDEND/polyMesh/$f" ] \
        || { echo "FAIL: the seed wrote no $SEEDEND/polyMesh/$f, so there is no refinement state to read"
             exit 1; }
done

# ---- PHASE 2: the gate case. The refined mesh becomes constant/, its fields become 0/. ---------------
G="$W/levels"
rm -rf "$G"
cp -r "$SRC" "$G" || exit 1
rm -rf "$G"/0 "$G"/processor* "$G"/log.* "$G"/0.[0-9]* "$G"/constant/polyMesh
cp "$REFSRC/constant/dynamicMeshDict" "$G/constant/dynamicMeshDict" || exit 1
cp -r "$S/$SEEDEND/polyMesh" "$G/constant/polyMesh" || exit 1
mkdir -p "$G/0" || exit 1
# the FIELDS beside that mesh, and only those: polyMesh is the mesh, and `uniform` is the time state.
for f in "$S/$SEEDEND"/*; do
    b=$(basename "$f")
    [ "$b" = polyMesh ] && continue
    [ "$b" = uniform ] && continue
    # ...AND NOT alphaPhi0.water, deliberately. OpenFOAM reads the previous step's MULES correction flux
    # READ_IF_PRESENT and starts its alpha equation from it; brae's restart path does not, and that is a
    # difference in RESTART semantics rather than in the refinement state. MEASURED with it present: the
    # mesh and every cell LEVEL came out exactly OpenFOAM's while alpha read 8.8625e-02 and U 1.2349e+00 --
    # a levels gate reporting an alphaPhi0 gap. Omitted, neither code has one and the comparison is this
    # unit's. Named here because it is a hole in the fixture, not a property of the port.
    [ "$b" = "alphaPhi0.water" ] && continue
    # ...AND NOT Uf, which is a DEFECT THIS FIXTURE FOUND and not a property of the levels.
    # createUfIfPresent.H builds Uf with `IOobject::READ_IF_PRESENT` and `fvc::interpolate(U)` as the
    # FALLBACK, so OpenFOAM READS a stored face velocity whenever one is there -- and for an adaptive case
    # that is every time directory, because a refining mesh is dynamic and writes Uf beside each one. brae
    # always interpolates (inter_case_cpp.cu, "a `Uf` file in the start directory is a restart's, and a
    # restart of a moving mesh is refused"), which was true for a MOVING mesh and is false for a refining
    # one. MEASURED with the file present: the mesh and EVERY CELL LEVEL came out exactly OpenFOAM's while
    # alpha read 8.8625e-02, p_rgh 8.9647e-02 and U 1.2349e+00 -- phi = Sf & Uf at the change consumes it
    # directly. Omitted here so both codes take the same fallback and this gate measures the levels; the
    # read is its own unit, recorded in PORT.md with that number.
    [ "$b" = Uf ] && continue
    # ...AND NOT phi, THE SAME DEFECT CLASS, found by the same fixture. createPhi.H is READ_IF_PRESENT
    # (createPhi.H:45) with `linearInterpolate(U) & Sf` as the fallback, and brae never reads
    # startDir + "/phi" at all. With Uf omitted and phi present the gate read alpha 1.0501e-09, p_rgh
    # 4.5130e-09 and U 3.6038e-08 -- five orders better than with Uf, and still five orders off the floor.
    # Both reads are one unit of their own, with these two numbers as its red.
    [ "$b" = phi ] && continue
    cp -r "$f" "$G/0/$b" || exit 1
done
dicts "$G" "$END" || exit 1

# THE SEED'S LEVELS MUST BE MIXED, or this whole unit is invisible. Asserted, with the counts printed.
python3 - "$G" <<'PYEOF' || exit 1
import re, sys, collections
G = sys.argv[1]
t = open(G + '/constant/polyMesh/cellLevel').read()
body = t[t.find('}', t.find('FoamFile')) + 1:]
m = re.search(r'(\d+)\s*\((.*?)\)', body, re.S)
if not m:
    raise SystemExit('FAIL: could not parse the seeded cellLevel (a uniform short form?): %r'
                     % body.strip()[:80])
v = [int(x) for x in m.group(2).split()]
c = collections.Counter(v)
print('  the seeded mesh: %d cells at levels %s' % (len(v), dict(sorted(c.items()))))
if len(c) < 2:
    raise SystemExit('FAIL: the seeded cellLevel is UNIFORM, so consistentRefinement\'s 2:1 constraint and '
                     'the >8-anchor protected set cannot differ between a port that reads it and one that '
                     'does not -- this fixture would pass either way')
if max(c) >= 2:
    raise SystemExit('FAIL: the seed is already AT maxRefinement 2 everywhere the band wants, so OpenFOAM\'s '
                     'continuation selects 0 cells and no refinement decision is ever taken. MEASURED with a '
                     'two-step seed: "Selected 0 cells for refinement out of 4998". Seed FEWER steps.')
# ...and the refinementHistory's visibleCells must match the cell count, which is FATAL in OpenFOAM
h = open(G + '/constant/polyMesh/refinementHistory').read()
vis = h[h.find('// visibleCells'):]
mv = re.search(r'(\d+)\s*\(', vis)
assert mv, 'FAIL: could not find the visibleCells list in the seeded refinementHistory'
if int(mv.group(1)) != len(v):
    raise SystemExit('FAIL: the seeded refinementHistory has %s visibleCells against %d cells; OpenFOAM '
                     'stops on this (hexRef8.C:1982-1990)' % (mv.group(1), len(v)))
print('  ...and its refinementHistory carries %s visibleCells, matching' % mv.group(1))
PYEOF

GKEY=$(oracleKey "$G" "interfoam_amr_levels" "levels" "$DT" "$N")
if oracleRestore "$G" "$GKEY" "$END"; then
    echo "[levels] OpenFOAM's $N steps from the refined mesh reused from the oracle cache"
else
    ( cd "$G" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam (the gate case)"; tail -30 "$G/log.interFoam"; exit 1; }
    [ -d "$G/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory"; exit 1; }
    oracleStore "$G" "$GKEY"
fi

# THE FIXTURE MUST CHANGE THE MESH AT ALL, or nothing here reads a level.
grep -qE "Refined from|Unrefined from" "$G/log.interFoam" \
    || { echo "FAIL: OpenFOAM's continuation changes no cells, so no refinement decision is taken and the"
         echo "      levels are never read for anything"
         grep -nE "Selected|Refined" "$G/log.interFoam" | head -5; rc=1; }
grep -E "Refined from|Unrefined from" "$G/log.interFoam" | sed 's/^/  OpenFOAM: /'
[ -f "$G/$END/polyMesh/owner" ] \
    || { echo "FAIL: OpenFOAM wrote no polyMesh into $END"; rc=1; }
[ -f "$G/$END/cellLevel" ] \
    || { echo "FAIL: OpenFOAM wrote no cellLevel FIELD into $END -- `dumpLevel true` is what this gate's"
         echo "      sharpest arm compares against"; rc=1; }
[ $rc -eq 0 ] || { echo "interfoam_amr_levels_vs_openfoam: rc $rc"; exit $rc; }

# ---- THE ONE-ULP TWIN, because the field bounds are 1e-08 and not 1e-14 and that has to be justified by
# something other than the code that must meet it. This fixture starts MID-FLOW -- a developed interface on
# a mesh already refined -- where every other profile of this gate starts from rest, so the question is
# whether the case amplifies. OpenFOAM is run a second time with ONE CELL of the initial alpha.water moved
# by one ulp and compared to ITSELF; the binary asserts that brae is no further from OpenFOAM than that twin
# is, and that the twin itself is above 1e-10 so a fixture that stopped amplifying would fail here rather
# than pass on a slack bound.
#
# THE PERTURBED CELL IS AT THE INTERFACE, and tests/interfoam_amr_ulp_cell.py says why: perturbing the first
# cell that is exactly 1 puts it deep in the water, where every neighbour is 1 as well and the curvature --
# a function of alpha's GRADIENT -- never sees it. That twin moved 12 of 12,487 faces of the surface tension
# force and its p_rgh after three steps moved 8.5e-11, below the 1e-10 this gate asks of an amplifying case;
# the restart profile read 20x to 32x against it and could assert nothing. Against a twin perturbed at the
# interface the same brae runs read 0.98x/0.85x/1.03x on the levels profile and 0.86x/0.95x/0.78x on the
# restart one (alpha/p_rgh/U), which is what lets the restart fields be asserted at all.
U2="$W/levelsUlp"
rm -rf "$U2"
cp -r "$G" "$U2" || exit 1
rm -rf "$U2"/0.[0-9]* "$U2"/log.interFoam
python3 "$(dirname "$0")/interfoam_amr_ulp_cell.py" "$U2" || exit 1
UKEY=$(oracleKey "$U2" "interfoam_amr_levels" "ulp" "$DT" "$N")
if oracleRestore "$U2" "$UKEY" "$END"; then
    echo "[ulp] OpenFOAM against itself reused from the oracle cache"
else
    ( cd "$U2" && interFoam > log.interFoam 2>&1 )         || { echo "FAIL: interFoam (the one-ulp twin)"; tail -20 "$U2/log.interFoam"; exit 1; }
    oracleStore "$U2" "$UKEY"
fi
# the twin must refine the SAME way, or it differs in its mesh as well as in one value and the envelope it
# measures is not round-off
grep -qF "Refined from 2814 to 4998 cells." "$U2/log.interFoam" \
    || { echo "FAIL: the one-ulp twin refined differently, so its distance is not round-off"
         grep -n "Refined from" "$U2/log.interFoam"; exit 1; }

"$BIN" "$G" "$G/0" "$G/$END" "$N" "$G/log.interFoam" levels "$U2/$END" || rc=1

# ---- THE BINARY ARM, which is motorBike's actual shape and the trap the cellZones reader fell into: that
# reader was ASCII-only and threw "read past end" on a binary mesh. motorBike's controlDict says
# `writeFormat binary`, so its snappy-written cellLevel and pointLevel ARE binary -- and its Allrun.pre
# DELETES the refinementHistory, which makes snappy's own refinement permanent (with no file OpenFOAM still
# builds an ACTIVE history with every cell visible and no parents).
#
# THE BINARY FILES ARE OPENFOAM'S OWN, not this script's: foamFormatConvert converts FIELDS and leaves
# polyMesh label lists alone (measured -- it reported "Writing U / rAU / p_rgh / p / cellLevel" and touched
# nothing in polyMesh), so the seed is RUN AGAIN with `writeFormat binary` and only its two label lists are
# taken. The two seeds are the same computation at the same pinned tolerances; what differs is the encoding,
# which is the thing under test. Encoding them here instead would test this script's writer against brae's
# reader and could agree on a shared mistake.
SB="$W/seedBinary"
rm -rf "$SB"
cp -r "$SRC" "$SB" || exit 1
rm -rf "$SB"/0 "$SB"/processor* "$SB"/log.* "$SB"/0.[0-9]*
cp "$REFSRC/constant/dynamicMeshDict" "$SB/constant/dynamicMeshDict" || exit 1
dicts "$SB" "$SEEDEND" binary || exit 1
SBKEY=$(oracleKey "$SB" "interfoam_amr_levels_seed" "seedBinary" "$DT" "$SEEDSTEPS")
if oracleRestore "$SB" "$SBKEY" "$SEEDEND"; then
    echo "[seedBinary] reused from the oracle cache"
else
    ( cd "$SB" && blockMesh > log.blockMesh 2>&1 && cp -r 0.orig 0 && setFields > log.setFields 2>&1 \
          && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: the binary seed"; tail -20 "$SB"/log.* 2>/dev/null; exit 1; }
    oracleStore "$SB" "$SBKEY"
fi

B="$W/binary"
rm -rf "$B"
cp -r "$G" "$B" || exit 1
rm -rf "$B"/0.[0-9]* "$B"/log.*
cp "$SB/$SEEDEND/polyMesh/cellLevel"  "$B/constant/polyMesh/cellLevel"  || exit 1
cp "$SB/$SEEDEND/polyMesh/pointLevel" "$B/constant/polyMesh/pointLevel" || exit 1
rm -f "$B/constant/polyMesh/refinementHistory"
python3 - "$G" "$B" <<'PYEOF' || rc=1
import re, sys
G, B = sys.argv[1], sys.argv[2]
def fmt(p):
    t = open(p, 'rb').read(2000).decode('latin-1')
    m = re.search(r'format\s+(\w+);', t)
    return m.group(1) if m else '?'
for f in ('cellLevel', 'pointLevel'):
    got = fmt(B + '/constant/polyMesh/' + f)
    if got != 'binary':
        raise SystemExit('FAIL: %s is `%s`, not binary, so this arm would test the ASCII path again' % (f, got))
    if fmt(G + '/constant/polyMesh/' + f) != 'ascii':
        raise SystemExit('FAIL: the ASCII gate case\'s %s is not ascii, so the two arms are not two '
                         'encodings of one mesh' % f)
print('  the binary arm: cellLevel and pointLevel are OpenFOAM-written BINARY, refinementHistory removed as '
      'motorBike\'s Allrun.pre removes it')
PYEOF
if [ $rc -eq 0 ]; then
    # The oracle is the SAME OpenFOAM run: the mesh is the same mesh, so brae must reach the same answer from
    # the other encoding of it. Any difference between this arm and the one above IS the decoding.
    "$BIN" "$B" "$B/0" "$G/$END" "$N" "$G/log.interFoam" levelsBinary "$U2/$END" || rc=1
fi

# ---- THE restart PROFILE: the SAME fixture with the written state KEPT, which is what the two profiles above
# deliberately omit. createUfIfPresent.H builds Uf with IOobject::READ_IF_PRESENT and AUTO_WRITE, so every
# time directory a REFINING case writes carries one, and `phi = mesh.Sf() & Uf()` (interFoam.C:131) consumes
# it directly at the change while fvc::ddtCorr reads its oldTime. brae interpolated it unconditionally.
#
# MEASURED, the control against the port on this very case: alpha 8.8625e-02 -> 5.6049e-10, p_rgh 8.9647e-02
# -> 2.7323e-09, U 1.2349e+00 -> 4.2657e-08, rAU 8.0261e-01 -> 1.2132e-08, phi 5.0101e-01 -> 1.6414e-08,
# Uf 1.2746e+00 -> 7.4911e-08. Eight orders.
#
# alphaPhi0.water IS ALSO READ_IF_PRESENT (createAlphaFluxes.H:1-8) and brae reads only its PRESENCE, not its
# values -- and this fixture CANNOT witness that, which is measured rather than assumed: interFoam.C:120-123
# is `if (mesh.topoChanging()) { talphaPhi1Corr0.clear(); }` with OpenFOAM's own comment "Do not apply
# previous time-step mesh compression flux if the mesh topology changed", so on a case whose first change is
# at step 1 the value is discarded before anything reads it. Omitting the file from the earlier profiles
# changed nothing to five figures, which is that. It stays an open finding for a NON-adaptive restart.
R="$W/restart"
rm -rf "$R"
cp -r "$SRC" "$R" || exit 1
rm -rf "$R"/0 "$R"/processor* "$R"/log.* "$R"/0.[0-9]* "$R"/constant/polyMesh
cp "$REFSRC/constant/dynamicMeshDict" "$R/constant/dynamicMeshDict" || exit 1
cp -r "$S/$SEEDEND/polyMesh" "$R/constant/polyMesh" || exit 1
mkdir -p "$R/0" || exit 1
# EVERYTHING the seed wrote, this time: Uf, phi and alphaPhi0 included.
for f in "$S/$SEEDEND"/*; do
    b=$(basename "$f")
    [ "$b" = polyMesh ] && continue
    [ "$b" = uniform ] && continue
    cp -r "$f" "$R/0/$b" || exit 1
done
dicts "$R" "$END" || exit 1
[ -f "$R/0/Uf" ] \
    || { echo "FAIL: the seed wrote no Uf, so this profile cannot witness the read it exists for"; exit 1; }

RKEY=$(oracleKey "$R" "interfoam_amr_levels" "restart" "$DT" "$N")
if oracleRestore "$R" "$RKEY" "$END"; then
    echo "[restart] OpenFOAM's $N steps from the full written state reused from the oracle cache"
else
    ( cd "$R" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam (the restart profile)"; tail -30 "$R/log.interFoam"; exit 1; }
    [ -d "$R/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory (restart)"; exit 1; }
    oracleStore "$R" "$RKEY"
fi
grep -qE "Refined from|Unrefined from" "$R/log.interFoam" \
    || { echo "FAIL: the restart profile changes no cells"; rc=1; }

# ...and its own one-ulp twin, because its bounds are 1e-08 for the same reason the others' are.
UR="$W/restartUlp"
rm -rf "$UR"
cp -r "$R" "$UR" || exit 1
rm -rf "$UR"/0.[0-9]* "$UR"/log.interFoam
python3 "$(dirname "$0")/interfoam_amr_ulp_cell.py" "$UR" || exit 1
URKEY=$(oracleKey "$UR" "interfoam_amr_levels" "restartUlp" "$DT" "$N")
if oracleRestore "$UR" "$URKEY" "$END"; then
    echo "[restartUlp] reused from the oracle cache"
else
    ( cd "$UR" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam (the restart one-ulp twin)"; tail -20 "$UR/log.interFoam"; exit 1; }
    oracleStore "$UR" "$URKEY"
fi

if [ $rc -eq 0 ]; then
    "$BIN" "$R" "$R/0" "$R/$END" "$N" "$R/log.interFoam" restart "$UR/$END" || rc=1
fi

echo "interfoam_amr_levels_vs_openfoam: rc $rc"
exit $rc
