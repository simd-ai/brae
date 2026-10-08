#!/usr/bin/env bash
# brae's interFoam against REAL OpenFOAM's on RAS/DTCHullMoving with its mesh FROZEN: the atmosphere's
# pressureInletOutletVelocity carries `tangentialVelocity $internalField;` -- (-1.668 0 0), the ship speed --
# which OpenFOAM turns into refValue = tv - n*(n & tv) at construction and fixes on every INFLOW face
# (valueFraction = neg(phi)*(I - nn)). brae's interFoam host loop claims the entry and hands it to the patch
# field; every other solver's factory and the device loop refuse it.
#
# THE MESH is the tutorial's own, made by real OpenFOAM SERIALLY -- the Allrun's surfaceFeatureExtract,
# blockMesh, six topoSet/refineMesh passes and snappyHexMesh, then setFields and renumberMesh, without its
# parallel run (the oracle is a serial run). 848,022 cells.
#
# STAGED, on both codes, and why:
#   dynamicFvMesh staticFvMesh   the rigid-body motion is the port's next units, not this one; the
#                                atmosphere sits at z = 4, beyond the body's outerDistance 1, and does not
#                                move in the shipped run either
#   cache { active false; }      A CONSEQUENCE OF FREEZING THE MESH, not a wall of the shipped case. OpenFOAM
#                                bypasses the cache on a changing mesh (gradScheme.C:99), and polyMesh sets
#                                moving() at the first movePoints and never clears it (polyMesh.C:1191), so
#                                under the shipped rigid-body motion the cache is inert in every corrector.
#                                FROZEN it is live, and at the second outer corrector's UEqn the cached
#                                grad(U) is stale (nOuterCorrectors 2, turbOnFinalIterOnly) and OpenFOAM's
#                                first request re-forms it, which brae refuses (the operand order is not
#                                modelled). BOTH codes run uncached, so the oracle stays OpenFOAM's own answer
#   adjustTimeStep no, deltaT 1e-4, write every step, ascii, functions {}   the oracle
#
# THE WINDOW CAN SEE THE ENTRY (measured, OpenFOAM against itself, cached and uncached alike): with the entry
# removed U moves 2.0e-05 at the first step and 1.0e-04 by the tenth, p_rgh 2.3e-04; one ulp of the entry's x
# component moves U 4e-14. 474 of the atmosphere's 798 faces take inflow at the first write and about 420 at
# every step after. The first UEqn assembly of the run cannot see it: createPhi takes the file value (0 0 0),
# so phi is 0 on the atmosphere and neg(0) is 0 until the first pressure correction.
#
# MEASURED, ten steps, brae against OpenFOAM WITH the entry: alpha 3.0e-14, p_rgh 1.3e-11, U 2.9e-13,
# k 3.2e-13, omega 6.2e-13, nut 7.7e-12; the atmosphere's U face by face 3.9e-15 and its p_rgh 0 -- where
# totalPressure reads 0.5*rho*|U_b|^2 and the tangential part is 1.668 of it; all 40 p_rgh and 20 alpha counts
# OpenFOAM's, the p_rgh initial residuals 8.3e-12. 427 of the 798 atmosphere faces take inflow at the tenth.
# OpenFOAM against itself with one ulp of the entry: U 4.2e-14, p_rgh 3.4e-12 at the tenth. Bounds in the gate.
#
# THE CONTROLS, three steps against OpenFOAM with the entry, each asserted to FAIL on a number:
#   notv     brae on the case with the entry stripped       U 3.9e-05, p_rgh 7.1e-05, the atmosphere's U and
#                                                            p_rgh 1.0 of their largest
#   flip     brae with the entry's sign flipped, (1.668 0 0)  U 3.0e-05, the atmosphere's U 2.0
#   BRAE_CONTROL_PIOV_TV=value   the refValue dropped from the stored inflow value only: the stripped
#                                entry's digits exactly (U 3.9e-05, the atmosphere 1.0) -- the whole witness
#                                is the value
# NOT DISCRIMINATED, measured: BRAE_CONTROL_PIOV_TV=sngrad (the refValue dropped from snGrad only) reads the
# unbroken three-step run to the LAST DIGIT -- alpha 4.7323e-15, U 1.1911e-13, nut 1.8206e-12, every residual --
# so nothing on this host path reads the atmosphere's snGrad. Its only witness is tests/test_piov_sngrad.cu
# LEG 7, bit for bit against a transcription.
# ...and the ORACLE's own, ten steps: OpenFOAM without the entry against OpenFOAM with it, asserted above
# 1e-6 of U (measured 1.2e-04 uncached; 1.0e-04 on the cached run).
#
# THE FORM: the staged cases run the entry as `uniform (-1.668 0 0)`, because renumberMesh rewrites 0/U and piov
# writes its tangentialVelocity expanded; the shipped `$internalField` form is read by tests/test_piov_sngrad.cu
# LEG 10 only.
#
# NOT CLAIMED: the snGrad half (above); the projection tv - n*(n & tv) and the construction-time normals -- the atmosphere is
# axis-aligned (normal (0 0 1)) and the ship speed is tangential, so neither is visible here; they are
# witnessed only by tests/test_piov_sngrad.cu on a tilted normal. The cached grad(U), the moving mesh and the
# device (refused, interfoam_refusals `device_piovTangential`).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_dtchullmoving_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/DTCHullMoving"
STL="$TUT/resources/geometry/DTC-scaled.stl.gz"
STEPS=${STEPS:-10}
CSTEPS=3
MODE=${MEASURE:+measure}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/DTCHullMoving tutorial not found at $SRC"; exit 77; }
[ -f "$STL" ]      || { echo "SKIP: the DTC hull surface is not at $STL"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}

# real OpenFOAM's runs (and meshes) are cached by a hash of the staged case: tests/of_oracle_cache.sh
. "$(dirname "$0")/of_oracle_cache.sh"
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in snappyHexMesh interFoam foamListTimes; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

grep -q "tangentialVelocity *\$internalField;" "$SRC/0.orig/U" \
    || { echo "FAIL: DTCHullMoving's atmosphere no longer writes tangentialVelocity \$internalField"; exit 1; }

M="$W/mesh"
if [ ! -f "$M/constant/polyMesh/owner" ]; then
    rm -rf "$M"
    cp -r "$SRC" "$M" || exit 1
    rm -rf "$M"/[1-9]* "$M"/0 "$M"/processor* "$M"/log.*
    # ...cached (oracleMesh): the key is the tutorial as copied, this function's text and the hull's STL
    meshDTCHullMoving()
    {
    (
        cd "$M" || exit 1
        mkdir -p constant/triSurface
        cp -f "$STL" constant/triSurface/
        surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 || exit 1
        blockMesh > log.blockMesh 2>&1 || exit 1
        for i in 1 2 3 4 5 6
        do
            topoSet -dict system/topoSetDict.$i > log.topoSet.$i 2>&1 || exit 1
            refineMesh -dict system/refineMeshDict -overwrite > log.refineMesh.$i 2>&1 || exit 1
        done
        snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 || exit 1
        # refineMesh leaves its cellMap in 0/polyMesh; the mesh itself is in constant
        rm -rf 0
        cp -r 0.orig 0
        setFields > log.setFields 2>&1 || exit 1
        renumberMesh -overwrite > log.renumberMesh 2>&1 || exit 1
    )
    }
    oracleMesh "$M" interfoam_dtchullmoving meshDTCHullMoving "$(sha256sum < "$STL" | cut -c1-16)" \
        || { echo "FAIL: meshing DTCHullMoving"; ls "$M"; exit 1; }
fi


# stage <name> <tv: shipped|none|flip>
stage()
{
    local name="$1" tv="$2"
    local C="$W/$name"
    rm -rf "$C"
    cp -r "$M" "$C" || return 1
    rm -f "$C"/log.*
    TV="$tv" STEPS="$STEPS" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s, k = re.subn(r'functions\s*\{.*\}\s*(?=//)', 'functions {}\n\n', s, flags=re.S)
assert k == 1, 'functions'
end = 0.0
for _ in range(n):
    end += 1e-4
for key, val in [('endTime', '%.10g' % end), ('deltaT', '1e-4'), ('adjustTimeStep', 'no'),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '18')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
m = os.path.join(d, 'constant/dynamicMeshDict')
t = open(m).read()
t, k = re.subn(r'^dynamicFvMesh\s+\S+;', 'dynamicFvMesh   staticFvMesh;', t, flags=re.M)
assert k == 1, 'dynamicFvMesh'
open(m, 'w').write(t)
p = os.path.join(d, 'system/fvSolution')
t = open(p).read()
t, k = re.subn(r'\ncache\s*\{\s*grad\(U\);\s*\}', '\ncache\n{\n    active false;\n    grad(U);\n}', t)
assert k == 1, 'the tutorial caches grad(U)'
open(p, 'w').write(t)
# renumberMesh -overwrite has READ AND REWRITTEN 0/U (renumberMesh.C:901, :1161-1219): the entry now reads
# `tangentialVelocity uniform (-1.668 0 0);`, the $internalField form expanded by piov's own write(). latin-1,
# because the header says `format binary` and a nonuniform list would be raw bytes.
u = os.path.join(d, '0/U')
t = open(u, encoding='latin-1').read()
assert len(re.findall(r'tangentialVelocity\s+[^;]*;', t)) == 1, 'one tangentialVelocity entry'
tv = os.environ['TV']
if tv == 'none':
    t, k = re.subn(r'\n\s*tangentialVelocity\s+[^;]*;', '', t)
    assert k == 1, 'tangentialVelocity'
elif tv == 'flip':
    m2 = re.search(r'tangentialVelocity\s+uniform\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s*;', t)
    assert m2, 'the entry is uniform'
    flipped = '(%.17g %.17g %.17g)' % tuple(-float(x) for x in m2.groups())
    t = t[:m2.start()] + 'tangentialVelocity uniform %s;' % flipped + t[m2.end():]
open(u, 'w', encoding='latin-1').write(t)
PYEOF
}

stage tv shipped   || exit 1
stage notv none    || exit 1
stage flip flip    || exit 1
oracleRun "$W/tv" interfoam_dtchullmoving tv &
oracleRun "$W/notv" interfoam_dtchullmoving notv &
wait
for c in tv notv; do
    grep -q "^End" "$W/$c/log.interFoam" || { echo "FAIL: interFoam [$c]"; tail -30 "$W/$c/log.interFoam"; exit 1; }
done
LAST=$(foamListTimes -case "$W/tv" 2>/dev/null | tail -1)
CTIME=$(foamListTimes -case "$W/tv" 2>/dev/null | sed -n "${CSTEPS}p")
[ -n "$LAST" ] && [ -n "$CTIME" ] || { echo "FAIL: OpenFOAM wrote no time directories"; exit 1; }
echo "OpenFOAM ran $STEPS Euler steps of DTCHullMoving (mesh frozen), with and without the entry; last $LAST"

rc=0
echo "== [tv]"
"$BIN" "$W/tv" "$W/tv" "$STEPS" "$LAST" "$W/tv/log.interFoam" $MODE || rc=1

# THE ORACLE'S OWN WITNESS: OpenFOAM without the entry against OpenFOAM with it
python3 - "$W/tv/$LAST/U" "$W/notv/$LAST/U" <<'PYEOF' || rc=1
import re, sys
def cells(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n?\(', s)
    body = s[m.end():]
    body = body[:body.index('\n)\n')]
    return [tuple(float(x) for x in v.split()) for v in re.findall(r'\(([^)]*)\)', body)]
a = cells(sys.argv[1]); b = cells(sys.argv[2])
ref = max(sum(c*c for c in v)**0.5 for v in a)
d = max(sum((x - y)**2 for x, y in zip(u, v))**0.5 for u, v in zip(a, b))/ref
print('  oracle: OpenFOAM without the entry against OpenFOAM with it: U %.4e' % d)
if d > 1e-6:
    print('  ok:   the window sees the entry')
else:
    print('  FAIL: the window cannot see the entry -- the gate would pass without it')
    sys.exit(1)
PYEOF

# control <brae case> <label> [env]: brae on CSTEPS steps against OpenFOAM WITH the entry and its log cut
# before step CSTEPS+1, asserted to fail on a number, not on a structural check
awk -v n="$CSTEPS" '/^Time = / { if (++k > n) exit } { print }' "$W/tv/log.interFoam" > "$W/log.cut"
control()
{
    local C="$W/$1"
    env ${3:-BRAE_GATE_PLAIN=1} "$BIN" "$C" "$W/tv" "$CSTEPS" "$CTIME" "$W/log.cut" > "$W/control.log" 2>&1
    local crc=$?
    local pat='FAIL: (alpha,|p_rgh,|U,|k,|omega,|nut,|atmosphere)'
    if [ $crc -ne 0 ] && grep -qE "$pat" "$W/control.log"; then
        echo "  ok:   control $2 fails on a number: $(grep -m1 -E "$pat" "$W/control.log" | sed 's/^ *FAIL: //' | tr -s ' ')"
        return 0
    fi
    echo "  FAIL: control $2 passed the gate -- the gate cannot see what it breaks"
    tail -20 "$W/control.log"
    return 1
}
control notv "the entry stripped from brae's case" || rc=1
control flip "the entry's sign flipped" || rc=1
control tv "the refValue dropped from the inflow value" BRAE_CONTROL_PIOV_TV=value || rc=1

# --- `moving`: THE TUTORIAL AS SHIPPED, the hull moving ------------------------------------------------------
# rigidBodyMotion (Pz + Ry, linearDamper, sphericalAngularDamper), the symmetryPlane point constraints, the
# atmosphere's tangentialVelocity, the cached grad(U) (inert on a changing mesh), adjustTimeStep. STAGED on both
# codes: writeControl timeStep, write every step, ascii, functions {} -- the oracle. The tutorial's
# `writeControl adjustable` makes Time::adjustDeltaT trim deltaT so a whole number of steps reaches the next
# write (1/round(1/1.2e-4) = 1.2000480019e-4 at step one); measured AS SHIPPED over two steps, brae's deltaT and
# body state equal OpenFOAM's to its last printed digit.
#
# SIX STEPS, BECAUSE AT THE EIGHTH OPENFOAM BRANCHES ON ITSELF: with ONE ulp moved in one alpha cell, its
# p_rgh final residuals agree to 1e-11 through step seven, jump 1.2e-02 at step eight's first solve, and step
# nine's first solve takes 9 iterations instead of 10; by step ten OpenFOAM against itself reads U 7.9e-06,
# p_rgh 6.1e-06, nut 4.8e-05, the body's q 9.0e-09 -- and brae against OpenFOAM reads U 8.7e-06, p_rgh 1.3e-05,
# nut 9.2e-05, q 2.3e-08 there, the same branch. Up to it brae is at machine precision (six steps: U 4.8e-13,
# q 2.2e-15, the moved points 2.5e-17 of the extent -- bounds in the gate).
#
# CONTROL, asserted to fail on a number: brae with the dynamicMeshDict's `restraints` removed, against
# OpenFOAM with them (measured at ten steps: q 8.4e-03, U 3.2e-02, nut 1.7e-01).
MSTEPS=6
MV="$W/moving"
rm -rf "$MV"
cp -r "$M" "$MV" || exit 1
rm -f "$MV"/log.*
MSTEPS="$MSTEPS" python3 - "$MV" <<'PYEOF' || { echo "FAIL: staging moving"; exit 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['MSTEPS'])
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
dt = float(re.search(r'^deltaT\s+([^;]+);', s, flags=re.M).group(1))
assert re.search(r'^adjustTimeStep\s+yes;', s, flags=re.M), 'the tutorial adjusts its time step'
# Courant stays far below maxCo here, so each step is 1.2x the last; Time::run stops once
# value >= endTime - 0.5*deltaT
ts = []
t = 0.0
for _ in range(n):
    dt *= 1.2
    t += dt
    ts.append((t, dt))
end = ts[-1][0] + 0.1*ts[-1][1]
assert ts[-2][0] < end - 0.5*ts[-2][1] and not ts[-1][0] < end - 0.5*ts[-1][1]
s, k = re.subn(r'functions\s*\{.*\}\s*(?=//)', 'functions {}\n\n', s, flags=re.S)
assert k == 1, 'functions'
for key, val in [('endTime', '%.17g' % end), ('writeControl', 'timeStep'), ('writeInterval', '1'),
                 ('writeFormat', 'ascii'), ('writePrecision', '18')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
m = open(os.path.join(d, 'constant/dynamicMeshDict')).read()
assert re.search(r'^dynamicFvMesh\s+dynamicMotionSolverFvMesh;', m, flags=re.M), 'the hull moves'
assert re.search(r'^motionSolver\s+rigidBodyMotion;', m, flags=re.M), 'rigidBodyMotion'
PYEOF
MNR="$W/movingNoRestraints"
rm -rf "$MNR"
cp -r "$MV" "$MNR" || exit 1
python3 - "$MNR/constant/dynamicMeshDict" <<'PYEOF' || { echo "FAIL: staging the no-restraints control"; exit 1; }
import re, sys
p = sys.argv[1]
s = open(p).read()
s, k = re.subn(r'\nrestraints\s*\{(?:[^{}]|\{[^{}]*\})*\}', '\n', s)
assert k == 1, 'restraints'
open(p, 'w').write(s)
PYEOF
oracleRun "$MV" interfoam_dtchullmoving moving
grep -q "^End" "$MV/log.interFoam" || { echo "FAIL: interFoam [moving]"; tail -30 "$MV/log.interFoam"; exit 1; }
grep -q "Selecting motion solver: rigidBodyMotion" "$MV/log.interFoam" \
    || { echo "FAIL: OpenFOAM's moving profile did not select rigidBodyMotion"; exit 1; }
MLAST=$(foamListTimes -case "$MV" 2>/dev/null | tail -1)
[ "$(foamListTimes -case "$MV" 2>/dev/null | wc -l)" = "$MSTEPS" ] \
    || { echo "FAIL: OpenFOAM ran $(foamListTimes -case "$MV" | wc -l) steps of the moving profile, not $MSTEPS"; exit 1; }
echo "OpenFOAM ran $MSTEPS steps of DTCHullMoving as shipped (the hull moving); last $MLAST"
# brae starts from the staged case, not from OpenFOAM's time directories
MB="$W/movingBrae"
rm -rf "$MB"
mkdir -p "$MB"
cp -r "$M/0" "$MB/"
cp -r "$MV/constant" "$MV/system" "$MB/"
echo "== [moving]"
"$BIN" "$MB" "$MV" "$MSTEPS" "$MLAST" "$MV/log.interFoam" $MODE moving || rc=1
"$BIN" "$MNR" "$MV" "$MSTEPS" "$MLAST" "$MV/log.interFoam" moving > "$W/control.log" 2>&1
if [ $? -ne 0 ] && grep -qE 'FAIL: (the joint position|alpha,|p_rgh,|U,)' "$W/control.log"; then
    echo "  ok:   control the restraints removed fails on a number: $(grep -m1 -E 'FAIL: (the joint position|alpha,|p_rgh,|U,)' "$W/control.log" | sed 's/^ *FAIL: //' | tr -s ' ')"
else
    echo "  FAIL: control the restraints removed passed the gate -- the gate cannot see the dampers"
    tail -20 "$W/control.log"
    rc=1
fi

echo "interfoam_dtchullmoving_vs_openfoam: rc $rc"
exit $rc
