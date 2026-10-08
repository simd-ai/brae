#!/usr/bin/env bash
# brae's interFoam with an MRF ZONE on a REFINING mesh, against real OpenFOAM -- the unit that retires the
# `MRF beside refinement` refusal. It is a separate script from tests/interfoam_amr_vs_openfoam.sh, whose
# four profiles all stage laminar/damBreakWithObstacle and whose prep, cell counts and refinement
# assertions are that case's; this one needs a different case for a reason that is not convenience:
#
#   damBreakWithObstacle CANNOT CARRY AN MRF ZONE. Its walls are `fixedFluxPressure`, and brae refuses MRF
#   beside one by name -- constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF) subtracts MRF.relative(Sf & U_b)
#   there (constrainPressureI.H) and this port's subtracts Sf & U_b, so nothing gates the difference.
#   laminar/mixerVessel2D, the one shipped MRF tutorial, has `zeroGradient` on both walls and is the case
#   the static MRF gate already holds (tests/interfoam_mrf_vs_openfoam.sh, alpha 8.2e-15 over 20 steps).
#
# WHAT OPENFOAM DOES, read rather than assumed. MRFZoneList::update() (MRFZoneList.C:441-450) is guarded on
# mesh.topoChanging() and calls each zone's update(), which is `if (topoChanging) setMRFFaces();` AND
# NOTHING ELSE (MRFZone.C:598-604): the dictionary is not re-read, cellZoneID_ is not re-found, omega is
# not re-evaluated, excludedPatchLabels_ is not re-resolved. It does not need to be -- resetZones preserves
# every cellZone's INDEX and ORDER through a change and moves only its labels (polyTopoChange.C:1900-1925),
# and hexRef8 gives each of a split cell's seven children the parent's zone id. So the whole of MRF's
# mesh-dependent state is THREE FACE LISTS, and interFoam calls the update at interFoam.C:128, inside
# `if (mesh.changing())`, after gh/ghf and BEFORE the correctPhi block.
#
# THE FIXTURE, staged from mixerVessel2D with nothing neutralised:
#   * its own `rotor` cellZone, `omega constant 6.2831853`, `nonRotatingPatches ()`, MRF1 active;
#   * a dynamicMeshDict refining on alpha.water at maxRefinement 2 (damBreakWithObstacle's own entries,
#     every flux mapped to `none`);
#   * deltaT 2.5e-4 rather than the tutorial's 1e-3. NOT a convenience: at 1e-3 the refined mesh runs at
#     Courant 1.6 and both codes overshoot alpha to 1.2998 -- OpenFOAM reads 1.29979960987982257 and brae
#     1.29979960985009, which is agreement to eleven digits on a case whose own answer is out of bounds.
#     At 2.5e-4 the Courant number is 0.446 and alpha stays at 1.0000000067 in both;
#   * EVERY SOLVE PINNED to tolerance 1e-14, relTol 0. The tutorial ships `p_rgh { tolerance 1e-07;
#     relTol 0.05; }`, and at that setting the comparison measures the Krylov STOPPING POINT: the fields
#     read 2.2e-02 on p_rgh and 2.2e-07 on alpha with every one of the 12 iteration counts identical.
#     Pinning takes the same run to 2.4e-06 and 7.9e-11.
#
# THE FIXTURE CAN WITNESS, and for this unit that is not "the mesh refined" but "THE MRF ZONE ITSELF
# refined" -- a zone away from the refinement band would be carried correctly by doing nothing. MEASURED
# from OpenFOAM's own written meshes, the `rotor` zone against the mesh it sits in:
#     construction  3072 cells   rotor 1536
#     t = 0.00025   3660         rotor 1676     (20 zone cells split, so 140 children joined it)
#     t = 0.0005    5956         rotor 2180
#     t = 0.00075   5956         rotor 2180
#     t = 0.001     6068         rotor 2292
# The zone grows at three of the four steps. The refinement band is the alpha interface, and setFields puts
# the water in the quadrant x>0, y>0 of a tank of radius 1 -- so the interface runs along both axes,
# straight through an MRF annulus of radius 0.2 to 0.6. That is why this zone is guaranteed to be split and
# a zone chosen by index would not be.
#
# THE ORACLE IS OPENFOAM'S OWN RENUMBERED ZONE, cell by cell, and it exists only because a topoChanging
# mesh writes its polyMesh into every time directory -- cellZones among them. The fvOptions unit had to
# read a printed count off a log line; MRFZone has no Info<< of its own, and this is the better oracle
# anyway: 2292 labels compared one at a time, plus the ascending order that resetZones' own walk produces.
#
# THE CONTROL, BRAE_CONTROL_AMR_NO_MRF_UPDATE: the zones keep the face lists they were built with. That is
# the port that does not throw -- every index is in range on a mesh that only grew -- and it is caught by
# twelve orders. MEASURED at this gate's four steps: alpha 2.4909e-01 against the port's 7.8697e-11, U
# 4.8591e-01 relative, p_rgh 2.5826e+00 -- and it runs on the DEVICE arm as well, because a device that never
# rebuilt its zones would still track a host that never rebuilt its own, so the device-vs-host bound cannot
# witness this. THE CONTROL IS GUARDED RATHER THAN TRUSTED: it keeps the OLD mesh's cell and face indices,
# in range on a mesh that only grew and PAST THE END on one that shrank, so the adapter throws if any kept
# index is out of range. A control may be wrong; it may not be undefined.
#
# THE BOUNDS ARE 1e-09, NOT 1e-14, AND THAT IS THE CASE'S CONDITIONING RATHER THAN SLACK -- which is a
# claim, so the gate measures it instead of asserting it. It runs OpenFOAM A SECOND TIME with ONE CELL of
# the initial alpha.water moved by ONE ULP (1.11e-16) and compares OpenFOAM to ITSELF:
#     steps   alpha        p_rgh rel    U rel        brae as a MULTIPLE of it: alpha  p_rgh    U
#     1       1.1102e-16   1.7814e-14   9.1034e-15                            0.00x   18.15x   242.06x
#     2       9.8810e-15   2.7981e-09   4.4607e-09                            4.34x   0.82x    0.86x
#     3       5.5537e-11   5.2358e-09   6.6193e-09                            0.58x   0.68x    0.83x
#     4       9.6227e-11   5.7215e-09   8.6395e-09                            0.69x   0.48x    0.64x
# Five orders between step 1 and step 2. At four steps brae reads p_rgh 2.7452e-09 against that 5.7215e-09:
# CLOSER TO OPENFOAM THAN OPENFOAM'S OWN ONE-ULP TWIN, and the gate asserts exactly that on p_rgh and U.
#
# THE TWO CURVES CROSS, which is why the assertion is on p_rgh and U and NOT on alpha. brae's distance starts
# at its own 160-iteration PCG floor and the twin's starts at one ulp in one cell, so the twin is BELOW brae
# early and overtakes it as the case amplifies -- the 18x and 242x in the first row are that, not a defect.
# The gate pins the regime by asserting the twin's own p_rgh is above 1e-10, which FAILS at one step,
# correctly: there the 1e-09 bounds would be slack. Inside the regime p_rgh and U are under 1x at every step
# and alpha is not (4.34x at two steps, non-monotone because brae's alpha is exactly 0 at step one), so alpha
# is held by its absolute bound (1e-10, measured 7.87e-11) and its ratio is printed only. An assertion that
# holds at the step count someone happened to pick is not an assertion.
#
# The one-ulp argument is MANDATORY for this profile. It was optional first, and six arguments then asserted
# 1e-09 bounds with nothing behind them while the whole gate went green.
#
# MEASURED, every arm green, four steps:
#                    host          device        device vs host
#     alpha          7.8697e-11    4.9839e-11    6.6390e-11
#     p_rgh rel      2.5814e-09    6.5233e-09    9.0292e-09
#     U rel          5.1157e-09    4.2647e-09    9.3706e-09
#     phi rel        3.4245e-10    3.2242e-10    6.5173e-10
#     rAU rel        1.2473e-10    Uf rel 4.7302e-09    patch alpha, U and p_rgh all EXACT (0.0e+00)
#   the mesh: the same cell, face, internal-face and point counts, every face's owner, every internal
#   face's neighbour and every cell's refinement level OpenFOAM's, at every one of the three changes;
#   all 12 p_rgh and all 4 pcorr iteration counts OpenFOAM's (163/160/157 180/178/176 178/173/170
#   173/170/168 and 0/0/183/184), every final residual within 1e-15, and the first solve after the first
#   change BIT-IDENTICAL in its initial residual -- which is the sharpest single number here, because it
#   says the mapped state and the pressure system assembled on it are OpenFOAM's before any solver
#   round-off can enter;
#   sum local continuity 5.736483e-18 against OpenFOAM's 5.687700e-18, asserted ABSOLUTELY: a relative
#   comparison of two numbers eighteen orders below the field measures their last digits and nothing else.
#
# WHAT THIS GATE DOES NOT CLAIM:
#   * MRF under a MOVING mesh, still refused by name (inter_case_cpp.cu). A refining mesh renumbers; a
#     moving one also composes MRF.makeRelative with the mesh flux, and nothing here holds that.
#   * MRF under RAS, refused: mixerVessel2D is laminar.
#   * MRF beside a fixedFluxPressure p_rgh patch, refused -- see the top of this file.
#   * A TIME-VARYING omega, refused: `omega constant 6.2831853` is a Function1 whose value never moves, so
#     nothing here would notice a zone that failed to re-evaluate it. OpenFOAM's own update does not
#     re-evaluate it either, which is why the port does not and why this is a coverage hole and not a bug.
#   * UNREFINEMENT beside MRF. At this deltaT the case only refines. AT THE TUTORIAL'S OWN 1e-3 IT DOES
#     UNREFINE -- `Refined from 6124 to 6390 cells.` then `Unrefined from 6390 to 6362 cells.` at step 4,
#     with the rotor zone growing 2348 -> 2544 through it, and brae reaches the same 6362 and the same 4
#     split points. That is the first fixture found that unrefines inside a solver run, and it is recorded
#     here as the lead for that pending unit rather than gated at a Courant number of 1.6.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_amr_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/mixerVessel2D"
REFSRC="$TUT/multiphase/interFoam/laminar/damBreakWithObstacle"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: mixerVessel2D tutorial not found at $SRC"; exit 77; }
[ -d "$REFSRC" ]   || { echo "SKIP: damBreakWithObstacle not found at $REFSRC (its dynamicMeshDict is the template)"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for app in blockMesh topoSet setsToZones setFields interFoam m4; do
    command -v "$app" > /dev/null 2>&1 || { echo "SKIP: $app not on PATH"; exit 77; }
done

DT=0.00025
N=4
END=$(python3 -c "print('%.10g' % ($N*float('$DT')))")

# stage <dir> [<ulp>]
#
# Everything case-specific lives here, which is the reason this is its own script. `ulp` non-empty moves ONE
# cell of the initial alpha.water by one ulp AFTER setFields, so the two runs differ in exactly that.
stage()
{
    local C="$1" ulp="${2:-}"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/0 "$C"/processor* "$C"/log.* "$C"/0.[0-9]*

    # THE REFINEMENT DICT is damBreakWithObstacle's own, at maxRefinement 2 -- copied rather than written
    # so a tutorial that changed its entries would surface here instead of being silently out of date.
    sed -e 's/maxRefinement   2;/maxRefinement   2;/' "$REFSRC/constant/dynamicMeshDict" \
        > "$C/constant/dynamicMeshDict" || return 1
    grep -q "dynamicFvMesh   dynamicRefineFvMesh" "$C/constant/dynamicMeshDict" \
        || { echo "FAIL: the template dynamicMeshDict no longer selects dynamicRefineFvMesh"; return 1; }
    grep -q "field           alpha.water" "$C/constant/dynamicMeshDict" \
        || { echo "FAIL: the template refines on some field other than alpha.water"; return 1; }

    python3 - "$C" "$DT" "$END" <<'PYEOF' || return 1
import re, sys
C, dt, end = sys.argv[1], sys.argv[2], sys.argv[3]
p = C + '/system/controlDict'
s = open(p).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('stopAt', 'endTime'),
                 ('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', end),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '18'), ('writeCompression', 'off'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(p, 'w').write(s)

# EVERY SOLVE PINNED. At the tutorial's own `p_rgh { tolerance 1e-07; relTol 0.05; }` the comparison
# measures where PCG stopped, not what the discretisation produced: the same run reads p_rgh 2.2e-02.
p = C + '/system/fvSolution'
s = open(p).read()
s, n = re.subn(r'tolerance\s+\S+;', 'tolerance       1e-14;', s)
assert n >= 4, 'expected at least 4 tolerance entries, found %d' % n
s, m = re.subn(r'relTol\s+\S+;', 'relTol          0;', s)
assert m >= 4, 'expected at least 4 relTol entries, found %d' % m
open(p, 'w').write(s)

# THE CASE MUST STILL BE THE ONE THIS GATE DESCRIBES: an active MRF zone on `rotor`, a constant omega,
# and p_rgh fixing no value anywhere (which is what keeps it out of brae's fixedFluxPressure refusal and
# what makes its own pRefCell live).
mrf = open(C + '/constant/MRFProperties').read()
assert 'cellZone    rotor;' in mrf, 'mixerVessel2D no longer names the rotor cellZone'
assert 'omega     constant' in mrf, 'mixerVessel2D no longer ships a constant omega'
assert 'active      yes' in mrf, 'mixerVessel2D no longer activates MRF1'
prgh = open(C + '/0.orig/p_rgh').read()
assert 'fixedFluxPressure' not in prgh, 'mixerVessel2D p_rgh now fixes a flux; brae refuses MRF beside it'
PYEOF

    # THE MESH AND THE ZONE: the tutorial's own Allrun.pre, so the zone is the one it ships.
    ( cd "$C" \
        && m4 system/blockMeshDict.m4 > system/blockMeshDict \
        && blockMesh > log.blockMesh 2>&1 \
        && topoSet > log.topoSet 2>&1 \
        && setsToZones -noFlipMap > log.setsToZones 2>&1 ) || {
        echo "FAIL: preparing the mesh"; tail -20 "$C"/log.* 2>/dev/null; return 1; }
    cp -r "$C/0.orig" "$C/0" || return 1
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields"; return 1; }

    # THE MESH IS THE ONE THE MEASUREMENTS ABOVE WERE TAKEN ON, and the zone is in it.
    python3 - "$C" "$ulp" <<'PYEOF' || return 1
import re, sys, math
C, ulp = sys.argv[1], sys.argv[2]
own = open(C + '/constant/polyMesh/owner').read()
n = int(re.search(r'nCells:(\d+)', own).group(1))
assert n == 3072, 'mixerVessel2D is %d cells, not the 3072 every count in this gate is stated against' % n
z = open(C + '/constant/polyMesh/cellZones').read()
m = re.search(r'cellLabels\s+List<label>\s*(\d+)', z)
assert m, 'setsToZones wrote no cellLabels, so there is no MRF zone to carry'
nz = int(m.group(1))
assert nz == 1536, 'the rotor zone is %d cells, not the 1536 this gate is stated against' % nz
assert not __import__('os').path.exists(C + '/constant/polyMesh/cellLevel'), \
    'the fresh mesh already carries a cellLevel, so it does not start from level 0'
if ulp:
    p = C + '/0/alpha.water'
    t = open(p).read()
    i = t.find('internalField'); j = t.find('(', i); k = t.find('\n)', j)
    vals = t[j + 1:k].split()
    # the FIRST water cell, so the choice is reproducible, and a value of 1 so one ulp is 1.11e-16 rather
    # than a denormal. A perturbation the answer ABSORBS would measure nothing, and this one does not: it
    # reaches p_rgh 3.96e-09 by step 4.
    idx = next(q for q, v in enumerate(vals) if float(v) == 1.0)
    vals[idx] = repr(math.nextafter(1.0, 0.0))
    open(p, 'w').write(t[:j + 1] + '\n' + '\n'.join(vals) + '\n' + t[k:])
    print('    one ulp on cell %d of alpha.water: 1 -> %s' % (idx, vals[idx]))
PYEOF
}

# runOf <dir> <tag>  -- OpenFOAM, cached on every byte of the staged case
runOf()
{
    local C="$1" tag="$2" key
    key=$(oracleKey "$C" "interfoam_amr_mrf" "$tag" "$DT" "$N")
    if oracleRestore "$C" "$key" "$END"; then
        echo "[$tag] OpenFOAM's $N steps of deltaT $DT to t = $END reused from the oracle cache"
        return 0
    fi
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam ($tag)"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory ($tag)"; return 1; }
    oracleStore "$C" "$key"
}

rc=0

# ---- THE ORACLE ---------------------------------------------------------------------------------------
G="$W/mrf"
stage "$G" || exit 1
runOf "$G" "mrf" || exit 1

# THE FIXTURE MUST BE ABLE TO WITNESS, asserted on OpenFOAM's own log and its own written zones rather
# than assumed. Three refinements at these counts, a pcorr that does real work at the third, and an MRF
# zone that GREW -- without the last one the control would pass by doing nothing.
for want in "Refined from 3072 to 3660 cells." "Refined from 3660 to 5956 cells." \
            "Refined from 5956 to 6068 cells."; do
    grep -qF "$want" "$G/log.interFoam" \
        || { echo "FAIL: OpenFOAM's refinement sequence is not the one this gate is stated against"
             echo "      missing: $want"; grep -n "Refined from" "$G/log.interFoam"; rc=1; }
done
grep -q "Solving for pcorr, Initial residual = 1," "$G/log.interFoam" \
    || { echo "FAIL: OpenFOAM's pcorr solve does no work on this fixture, so the flux correction cannot be"
         echo "      witnessed and the mapped-flux control would be vacuous"
         grep -n "pcorr" "$G/log.interFoam"; rc=1; }
[ -f "$G/$END/polyMesh/owner" ] \
    || { echo "FAIL: OpenFOAM wrote no polyMesh into $END, so there is no mesh to compare on"; rc=1; }
[ -f "$G/$END/polyMesh/cellZones" ] \
    || { echo "FAIL: OpenFOAM wrote no cellZones into $END. That file IS this gate's oracle for the zone"
         echo "      carry -- without it the comparison would fall back to a count."; rc=1; }
[ -f "$G/$END/Uf" ] \
    || { echo "FAIL: OpenFOAM wrote no Uf, so a refining mesh is not dynamic after all"; rc=1; }
python3 - "$G" "$END" <<'PYEOF' || rc=1
import re, sys
G, END = sys.argv[1], sys.argv[2]
def nz(p):
    m = re.search(r'cellLabels\s+List<label>\s*(\d+)', open(p).read())
    return int(m.group(1)) if m else -1
a = nz(G + '/constant/polyMesh/cellZones')
b = nz(G + '/' + END + '/polyMesh/cellZones')
print('  OpenFOAM\'s own rotor zone: %d cells at construction, %d at t = %s' % (a, b, END))
if b <= a:
    raise SystemExit('FAIL: OpenFOAM\'s MRF zone did not grow through the changes (%d -> %d), so this '
                     'fixture cannot witness the carry and the control would pass by doing nothing' % (a, b))
if b != 2292:
    raise SystemExit('FAIL: the carried zone is %d cells, not the 2292 this gate is stated against' % b)
PYEOF
[ $rc -eq 0 ] || { echo "interfoam_amr_mrf_vs_openfoam: rc $rc"; exit $rc; }

# ---- THE ONE-ULP TWIN, which is where this profile's bounds come from ---------------------------------
U="$W/ulp"
echo "[ulp] OpenFOAM against ITSELF, one ulp on one cell of the initial alpha.water:"
stage "$U" ulp || exit 1
runOf "$U" "ulp" || exit 1
for want in "Refined from 3072 to 3660 cells." "Refined from 5956 to 6068 cells."; do
    grep -qF "$want" "$U/log.interFoam" \
        || { echo "FAIL: the one-ulp twin refined DIFFERENTLY, so the two runs differ in their mesh as"
             echo "      well as in one value and the envelope it measures is not round-off"
             grep -n "Refined from" "$U/log.interFoam"; exit 1; }
done

"$BIN" "$G" "$G/0" "$G/$END" "$N" "$G/log.interFoam" mrf "$U/$END" || rc=1

echo "interfoam_amr_mrf_vs_openfoam: rc $rc"
exit $rc
