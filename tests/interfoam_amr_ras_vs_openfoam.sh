#!/usr/bin/env bash
# brae's interFoam with a RAS TURBULENCE MODEL on a REFINING mesh, against real OpenFOAM -- the unit that
# retires the `turbulence beside refinement` refusal, and the one RAS/motorBike needs before it can be
# attempted. Its own script for the same reason the MRF one is: tests/interfoam_amr_vs_openfoam.sh stages
# laminar/damBreakWithObstacle and hardcodes that case's prep and cell counts, and this needs a case that
# is already turbulent.
#
# THE FIXTURE is RAS/damBreak/damBreak, which brae's static RAS gate already holds
# (tests/interfoam_ras_dambreak_vs_openfoam.sh), plus ONE injected file: a dynamicMeshDict. Nothing is
# neutralised. Two profiles:
#   `ke`  as shipped -- kEpsilon, kqRWallFunction on three walls, inletOutlet atmosphere, 2-D with an
#         `empty` defaultFaces patch.
#   `sst` the same case made kOmegaSST by the recipe the static RAS gate already uses: omega from the
#         epsilon file with omegaWallFunction, `div(phi,omega)`, the (U|k|omega) solver entry, and
#         fvSchemes' mandatory `wallDist { method meshWave; }`. IT IS NOT A DUPLICATE OF `ke`: kEpsilon
#         builds NO cell wall distance, so `ke` cannot witness the half of this unit that recomputes one.
#
# WHAT OPENFOAM DOES, read rather than assumed, and it splits in three:
#   * k, the second transported scalar and nut are registered AUTO_WRITE fields, so MapGeometricFields
#     autoMaps them with everything else -- cells through the cell mapper, each patch field through its own
#     autoMap. brae maps them the same way (interAmrUpdate's carriedScalarFields), and their per-step old
#     times with them.
#   * THE CELL WALL DISTANCE is RECOMPUTED, not mapped. wallDist is a
#     MeshObject<fvMesh, UpdateableMeshObject, wallDist> (wallDist.H:76-78), so a topology change reaches
#     wallDist::updateMesh, which is `pdm_->updateMesh(mpm); requireUpdate_ = true; movePoints();`
#     (wallDist.C:224-234) -- the latch is FORCED, with OpenFOAM's own comment "Force update if performing
#     topology change", where a plain MOVE only sets it when the interval divides the step. kOmegaSST holds
#     a const reference into that object (kOmegaSSTBase.C:381, `y_(wallDist::New(this->mesh_).y())`), whose
#     own header notes it is "different to wall distance in parent RASModel which is for near-wall cells
#     only".
#   * THE NEAR-WALL DISTANCE, that other one, needs nothing on the host. It is turbulenceModel::y_, a
#     nearWallDist and NOT a MeshObject, re-corrected inside
#         void Foam::turbulenceModel::correct() { if (mesh_.changing()) { y_.correct(); } }
#     (turbulenceModel.C:94-100) -- and `changing()` is moving() OR topoChanging(), so a REFINING mesh
#     re-corrects it. That is the same dynamic()-is-not-moving() distinction that decided this port's Uf,
#     correctUf and ddtCorr branches. brae's host closures call nearWallDist(m, g, patches) fresh at every
#     correct(), which is that statement unconditionally. THE DEVICE ARM CACHES IT and does rebuild it.
#
# THE FIXTURE CAN WITNESS, and for a MAPPER that takes two separate things, both measured from OpenFOAM's
# own written output rather than assumed:
#   (1) THE FIELDS MUST NOT BE UNIFORM at the change. This case ships k and epsilon uniform 0.1 and nut
#       uniform 0, and a uniform field is mapped correctly by a broken mapper. They develop in ONE step:
#       at t = 0.001 OpenFOAM's k spans [0.09221, 0.0999], epsilon [0.09981, 8.449] and nut
#       [9.057e-05, 0.008999]. So the FIRST change maps a uniform state and the SECOND maps a developed one,
#       which is why this gate takes four steps and not one. The spread is asserted in the .cu.
#   (2) WALL-FUNCTION FACES MUST BE SPLIT, or the wall half is untested. They are, and the water column
#       sitting against the left wall from t = 0 is why. MEASURED, OpenFOAM's own boundary files:
#           leftWall    50 -> 56 -> 68 faces
#           lowerWall   62 -> 68 -> 80
#           rightWall   50 -> 50 -> 50     (the water has not reached it -- a control inside the fixture)
#           defaultFaces (empty) 4536 -> 5004 -> 5940
#       and the mesh 2268 -> 2814 -> 4998 cells.
#
# EVERY SOLVE IS PINNED, and NOT all to the same value, which is the one thing here that needed measuring
# rather than deciding. At the tutorial's own `p_rgh { tolerance 1e-07; relTol 0.05; }` the comparison
# measures the Krylov stopping point: the fields read alpha 6.5e-11 and p_rgh 1.3e-11 with every iteration
# count already identical. Pinning everything to 1e-14 takes them to 1.1e-14 -- but k's residual CANNOT
# REACH 1e-14 with symGaussSeidel on this field, and both codes then grind to maxIter 1000 on two of the
# four k solves, where the count stops discriminating (OpenFOAM 1000, brae 92, both truncated). So the
# `(U|k|epsilon)` block is pinned at 1e-12, where every solve converges and no solve hits maxIter, and
# everything else at 1e-14.
#
# MEASURED, both profiles, both arms, four steps -- `ke` shown:
#                    host          device        device vs host
#     alpha          1.1425e-14    2.7756e-14    2.8311e-14
#     p_rgh rel      1.0323e-14    2.2903e-14    2.0000e-14
#     U rel          1.9283e-13    1.7083e-12    1.7087e-12
#     k rel          2.2516e-14    2.1519e-13
#     epsilon rel    2.0314e-14    6.1862e-13
#     nut rel        3.7415e-14    9.1907e-14
#   plus p 2.0403e-14, rAU 6.1004e-14, phi 5.8787e-14, Uf 1.4868e-13, patch alpha EXACT, patch U
#   2.4417e-14, patch p_rgh 1.0349e-14; the same cell, face, internal-face and point counts and every
#   owner, neighbour and cell level OpenFOAM's at both changes; ALL 23 ITERATION COUNTS OpenFOAM's -- 12
#   p_rgh, 3 pcorr, 4 k and 4 epsilon -- with no solve at maxIter; sum local continuity 2.443691e-15
#   against 2.443923e-15, asserted ABSOLUTELY because a relative comparison of two numbers fifteen orders
#   below the field measures their last digits (it reads 9.5e-05); and the device arm run twice in one
#   process bit-identical.
#
# ONE DEFECT FOUND, and only the device arm could find it. `dRhoOld` -- rho.oldTime() for the closure's
# `fvm::ddt(rho, k)` -- is captured at the TOP of a step from the previous step's dRho, i.e. BEFORE the
# change, and read after it. So after one change it sat at the OLD cell count while every other closure
# input was at the new one. NOTHING ELSE READS IT: the laminar path has no ddt(rho, k), so the buffer was
# wrong for as long as an adaptive turbulent case was refused. MEASURED before the fix: EVERY ONE of the
# 546 cells the change added (78 refined x 7 children, and the trace named the count exactly) came out of
# the closure's FIRST solve with k collapsed to 1.2e-12 where OpenFOAM's minimum is 0.09221, nut following
# it to 3.8e-25 against 9.057e-05, and by four steps nut reached 5.4e+04 times OpenFOAM's. It reads at the
# floor now. It was named by printing the SIZE of every mesh-sized input the closure is handed
# (BRAE_AMR_TRACE) -- one entry said 2268 where the rest said 2814 -- which is cheaper than bisecting the
# arithmetic and is left in for the next buffer of this class.
#
# WHAT THIS GATE DOES NOT CLAIM:
#   * THE CLOSURE UNDER CrankNicolson. Its ddt0 levels are registered DDt0Fields OpenFOAM autoMaps, exactly
#     as the solver's are -- and the solver's took two units of their own. No adaptive tutorial pairs
#     CrankNicolson with a turbulence model, so it is REFUSED by name (on the scheme AND on the state).
#   * LES. The filter width IS recomputed here, but the only LES fixture in the tree (LES/nozzleFlow2D) is
#     not adaptive, so nothing would hold it. REFUSED by name.
#   * A `wallDist { updateInterval }` other than 1. The forced latch above is what a change does
#     differently from a move, and it can only be witnessed at an interval greater than 1 -- which is
#     refused for want of a fixture, in moveInterTurbulence, and was before this unit.
#   * UNREFINEMENT beside turbulence: this case only refines at these settings.
#   * RAS/motorBike itself, which needs a reader for the cellLevel, pointLevel and level0Edge that
#     snappyHexMesh writes -- brae assigns level 0 everywhere, so maxRefinement would not mean the same
#     thing. That is the next unit, and this one is what it stands on.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_amr_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/damBreak/damBreak"
REFSRC="$TUT/multiphase/interFoam/laminar/damBreakWithObstacle"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/damBreak tutorial not found at $SRC"; exit 77; }
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
N=4
END=$(python3 -c "print('%.10g' % ($N*float('$DT')))")

rc=0

# gate <profile>   -- `ke` as shipped, `sst` the same case made kOmegaSST
gate()
{
    local profile="$1"
    local C="$W/$profile"
    local key
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/0 "$C"/processor* "$C"/log.* "$C"/0.[0-9]*

    # THE REFINEMENT DICT is damBreakWithObstacle's own, copied rather than written so a tutorial that
    # changed its entries surfaces here instead of being silently out of date.
    cp "$REFSRC/constant/dynamicMeshDict" "$C/constant/dynamicMeshDict" || return 1
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

# EVERY SOLVE PINNED -- but not all to 1e-14. See the header: symGaussSeidel cannot drive k to 1e-14 on
# this field, and both codes then grind to maxIter on two of the four k solves, where the iteration count
# stops discriminating. The (U|k|epsilon) block gets 1e-12, where every solve converges.
p = C + '/system/fvSolution'
s = open(p).read()
s, n = re.subn(r'tolerance\s+\S+;', 'tolerance       1e-14;', s)
assert n >= 4, 'expected at least 4 tolerance entries, found %d' % n
s, m = re.subn(r'relTol\s+\S+;', 'relTol          0;', s)
assert m >= 4, 'expected at least 4 relTol entries, found %d' % m
s, q = re.subn(r'("\(U\|k\|epsilon\).\*"\s*\{[^}]*?tolerance\s+)1e-14;', r'\g<1>1e-12;', s, flags=re.S)
assert q == 1, 'the (U|k|epsilon) solver block was not found, so k would be pinned below its floor'
open(p, 'w').write(s)

# THE CASE MUST STILL BE THE ONE THIS GATE DESCRIBES: turbulent, kEpsilon, wall functions on the walls.
t = open(C + '/constant/turbulenceProperties').read()
assert 'simulationType      RAS' in t, 'RAS/damBreak is no longer RAS'
assert 'RASModel        kEpsilon' in t, 'RAS/damBreak no longer ships kEpsilon'
assert 'turbulence      on' in t, 'RAS/damBreak no longer has turbulence on'
k = open(C + '/0.orig/k').read()
assert 'kqRWallFunction' in k, 'k no longer carries a wall function, so the wall half is untested'
assert re.search(r'internalField\s+uniform', k), 'k is no longer uniform at t=0 (see the header)'
PYEOF

    # `sst`: the recipe tests/interfoam_ras_dambreak_vs_openfoam.sh already uses on this case.
    if [ "$profile" = sst ]; then
        python3 - "$C" <<'SSTEOF' || { echo "FAIL: the sst profile was not staged"; return 1; }
import os, re, sys
d = sys.argv[1]

# 1. THE MODEL, and THE DENSITY LINEAGE WITH IT -- the second is a refusal of brae's rather than a
#    convenience. `density variable` with kOmegaSST is REFUSED BY NAME ("that is kOmegaSSTBase.C with rho
#    the mixture density and rhoPhi its flux; no shipped tutorial pairs them, so no gate would hold it"),
#    and RAS/damBreak ships `density variable`. So this profile drops it and takes the uniform lineage --
#    the same recipe the STATIC RAS gate uses for its own `sst` profile, so the pairing run here is one
#    that gate already holds. It is a real limitation, named in the header: the variable-density lineage is
#    witnessed by `ke` alone.
q = os.path.join(d, 'constant/turbulenceProperties')
t = open(q).read()
t, n = re.subn(r'RASModel\s+\w+;', 'RASModel        kOmegaSST;', t)
assert n == 1, 'no RASModel entry'
t, n = re.subn(r'^density\s+variable;\s*\n', '', t, flags=re.M)
assert n == 1, 'the tutorial no longer ships `density variable`, so this profile means something else'
open(q, 'w').write(t)

# 2. THE CLOSURE'S div ENTRIES, now the phi ones, and the SECOND field renamed with them. kOmegaSST's
#    wallDist entry is MANDATORY in fvSchemes, and it is the very object this unit recomputes.
q = os.path.join(d, 'system/fvSchemes')
t = open(q).read()
t, a1 = re.subn(r'div\(rhoPhi,k\)\s+Gauss upwind;', 'div(phi,k)      Gauss upwind;', t)
t, a2 = re.subn(r'div\(rhoPhi,epsilon\)\s+Gauss upwind;', 'div(phi,omega)  Gauss upwind;', t)
assert a1 == 1 and a2 == 1, 'the closure div entries were not renamed (%d, %d)' % (a1, a2)
t = t.rstrip() + '\n\nwallDist\n{\n    method meshWave;\n}\n'
open(q, 'w').write(t)

# 3. THE SOLVER ENTRY
q = os.path.join(d, 'system/fvSolution')
t = open(q).read()
t, n = re.subn(r'\(U\|k\|epsilon\)', '(U|k|omega)', t)
assert n >= 1, 'no (U|k|epsilon) solver entry'
open(q, 'w').write(t)

# 4. omega FROM THE epsilon FILE, so its patch types are the tutorial's own with the wall function swapped
e = open(os.path.join(d, '0.orig/epsilon')).read()
assert 'epsilonWallFunction' in e, 'epsilon carries no wall function to swap'
e = e.replace('epsilonWallFunction', 'omegaWallFunction')
e = re.sub(r'object\s+epsilon;', 'object      omega;', e)
e = e.replace('[0 2 -3 0 0 0 0]', '[0 0 -1 0 0 0 0]')
open(os.path.join(d, '0.orig/omega'), 'w').write(e)
SSTEOF
    fi

    key=$(oracleKey "$C" "interfoam_amr_ras" "$profile" "$DT" "$N")
    if oracleRestore "$C" "$key" "$END"; then
        echo "[$profile] OpenFOAM's $N steps of deltaT $DT to t = $END reused from the oracle cache"
    else
        ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) \
            || { echo "FAIL: blockMesh ($profile)"; tail -20 "$C/log.blockMesh"; return 1; }
        cp -r "$C/0.orig" "$C/0" || return 1
        ( cd "$C" && setFields > log.setFields 2>&1 ) \
            || { echo "FAIL: setFields ($profile)"; return 1; }
        [ ! -e "$C/constant/polyMesh/cellLevel" ] \
            || { echo "FAIL: the fresh mesh already carries a cellLevel, so it does not start from level 0"
                 return 1; }
        ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
            || { echo "FAIL: interFoam ($profile)"; tail -30 "$C/log.interFoam"; return 1; }
        [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory ($profile)"; return 1; }
        oracleStore "$C" "$key"
    fi

    # THE FIXTURE MUST BE ABLE TO WITNESS, on OpenFOAM's own log and its own written meshes.
    for want in "Refined from 2268 to 2814 cells." "Refined from 2814 to 4998 cells."; do
        grep -qF "$want" "$C/log.interFoam" \
            || { echo "FAIL: OpenFOAM's refinement sequence on $profile is not the one this gate is stated"
                 echo "      against -- missing: $want"; grep -n "Refined from" "$C/log.interFoam"
                 return 1; }
    done
    # The flux correction must do REAL WORK, or the mapped-flux control is vacuous. Asserted on the
    # ITERATION COUNT and not on the literal text "Initial residual = 1," -- which is what this check said
    # first, and it passed on `ke` (whose residual prints exactly `1,`) while failing on `sst`, whose
    # 0.99999999999999678 does real work over 110 iterations. A witness check that keys on the formatting
    # of a number is not checking the thing it names.
    grep -E "Solving for pcorr" "$C/log.interFoam" | grep -qvE "No Iterations 0$" \
        || { echo "FAIL: every pcorr solve on $profile took 0 iterations, so the flux correction does no"
             echo "      work here and the mapped-flux control would be vacuous"
             grep -n "Solving for pcorr" "$C/log.interFoam" | head -5; return 1; }
    grep -qE "No Iterations 1000" "$C/log.interFoam" \
        && { echo "FAIL: a solve on $profile hit maxIter 1000, so its iteration count is a truncation and"
             echo "      not a convergence -- see the header on why k is pinned at 1e-12"
             grep -n "No Iterations 1000" "$C/log.interFoam" | head -3; return 1; }
    [ -f "$C/$END/polyMesh/owner" ] \
        || { echo "FAIL: OpenFOAM wrote no polyMesh into $END on $profile"; return 1; }
    [ -f "$C/$END/Uf" ] \
        || { echo "FAIL: OpenFOAM wrote no Uf on $profile, so a refining mesh is not dynamic after all"
             return 1; }
    for f in k nut; do
        [ -f "$C/$END/$f" ] \
            || { echo "FAIL: OpenFOAM wrote no $f on $profile, so there is no turbulence oracle"; return 1; }
    done
    # ...AND THE WALL PATCHES GREW, which is what makes the wall half of this unit testable at all. Read
    # from OpenFOAM's own boundary files rather than assumed.
    python3 - "$C" "$END" <<'PYEOF' || return 1
import re, sys
C, END = sys.argv[1], sys.argv[2]
def walls(pm):
    t = open(pm + '/boundary').read()
    body = t[t.find('}', t.find('FoamFile')) + 1:]
    out = {}
    for m in re.finditer(r'^\s*([A-Za-z_][\w.-]*)\s*\n\s*\{(.*?)\n\s*\}', body, re.S | re.M):
        nf = re.search(r'nFaces\s+(\d+);', m.group(2))
        ty = re.search(r'type\s+(\w+);', m.group(2))
        if nf and ty and ty.group(1) == 'wall':
            out[m.group(1)] = int(nf.group(1))
    return out
a = walls(C + '/constant/polyMesh')
b = walls(C + '/' + END + '/polyMesh')
grown = [n for n in a if b.get(n, 0) > a[n]]
print('  wall patches, at construction -> t = %s: %s' % (END,
      ', '.join('%s %d -> %d' % (n, a[n], b.get(n, -1)) for n in sorted(a))))
if not grown:
    raise SystemExit('FAIL: no wall patch gained faces through the changes, so no wall function was ever '
                     'split and the near-wall half of this unit is not witnessed by this fixture')
PYEOF

    "$BIN" "$C" "$C/0" "$C/$END" "$N" "$C/log.interFoam" "$profile" || rc=1
}

# kEpsilon as shipped. It maps k, epsilon and nut and exercises the near-wall distance; it builds NO cell
# wall distance, which is what the second profile is for.
gate ke || rc=1

# ...and kOmegaSST, whose F1 and F2 blend on the CELL wall distance this unit recomputes.
gate sst || rc=1

echo "interfoam_amr_ras_vs_openfoam: rc $rc"
exit $rc
