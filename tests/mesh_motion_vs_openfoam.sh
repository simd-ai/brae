#!/usr/bin/env bash
# brae's solid-body MESH MOTION against REAL OpenFOAM's, with no flow solver in the way: the moved
# points, the mesh-motion flux meshPhi, and V and C of the moved mesh, per time step.
#
# WHY IT IS GATED ALONE. Everything interFoam does on a moving mesh is built on these four fields --
# the wall velocity is a face-centre displacement, the alpha equation divides by V/V0, the flux is made
# relative with meshPhi -- and a motion that is a little wrong gives a flow that is a little wrong and
# converges. So the motion is held to OpenFOAM's DIGITS before any equation reads it.
#
# THE ORACLE is OpenFOAM's moveDynamicMesh, run SERIALLY with a fixed deltaT and written every step at
# writePrecision 18, which is enough to read a double back exactly; then postProcess -func
# writeCellVolumes and writeCellCentres on the first and last of those time directories. Checked once
# by hand on testTubeMixer: moveDynamicMesh's points and meshPhi are interFoam's, byte for byte.
#
# MEASURED, every profile: points, meshPhi, V and C differ from OpenFOAM's by EXACTLY ZERO -- the same
# operations in the same order, not merely the same motion. V and C read 1e-15 and 1e-16 until
# fv_geometry's face centres took primitiveMeshTools.C's operation order; the test still prints them
# before any motion, so the geometry's own agreement and the motion's cannot be confused.
#
# PROFILES, each the tutorial's own constant/dynamicMeshDict unless it says otherwise:
#   testTubeMixer      multiMotion: a rotatingMotion table carrying an oscillatingRotatingMotion box.
#                      A product of two septernions, where the order is part of the answer.
#   sloshingTank2D     SDA, on the tutorial's m4-generated tank with its chamfered, non-orthogonal cells
#                      and empty front and back.
#   sloshingTank3D6DoF tabulated6DoFMotion with its default SPLINE interpolation, stepped across two of
#                      the table's knots.
#   sloshingCylinder   multiMotion of an oscillatingLinearMotion and a rotatingMotion, on a snappyHexMesh
#                      mesh: polyhedral cells whose faces are not all quadrilaterals, which is what
#                      face::centre and face::sweptVol's central decomposition exist for.
#   linear, axis       linearMotion and axisRotationMotion, which no interFoam tutorial uses, staged
#                      onto testTubeMixer's mesh.
#   tabulatedLinear    sloshingTank3D6DoF with `interpolationScheme linear`.
#
# EVERY DECISION ABOUT THE ORDER OF OPERATIONS WAS BROKEN ONCE, with switches that are not in the tree.
# What the gate read, worst of testTubeMixer, sloshingCylinder and sloshingTank2D -- and the reason the
# comparison is EXACT, since all but one of these sit under any round-off bound:
#   meshPhi as sweptVol/deltaT, not sweptVol*(1.0/deltaT)           meshPhi 2.0e-16
#   the swept volume fanned about the vertex mean, not face::centre  meshPhi 6.7e-15
#   points rotated through the quaternion, not the tensor q.R()      points 4.4e-16, meshPhi 1.8e-14
#   septernion*septernion without normalising the product rotation   points 6.1e-16, meshPhi 2.1e-14
#   the time as index*deltaT, not accumulated as Time::operator++    points 1.8e-16, meshPhi 8.4e-15 (on
#                                                                    sloshingTank2D alone: at deltaT 0.01
#                                                                    five sums ARE the products)
#   oldPoints never refreshed (the swept volume from the START)      meshPhi 4.1e+00 -- the only one a
#                                                                    bound would have caught
#
# THE CONTROL is on brae's side, because the oracle has no wrong answer to offer: testTubeMixer's two
# motions in the OTHER order, against OpenFOAM's in the file's order. It must FAIL, and the script
# asserts that it does and by how much.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_mesh_motion_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
LAM="$TUT/multiphase/interFoam/laminar"

[ -x "$BIN" ]                 || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$LAM/testTubeMixer" ]   || { echo "SKIP: testTubeMixer tutorial not found under $LAM"; exit 77; }
[ -f "$OFBASHRC" ]            || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1       || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v moveDynamicMesh > /dev/null 2>&1 || { echo "SKIP: moveDynamicMesh not on PATH"; exit 77; }

HDR='FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }'

# stage <name> <tutorial> <deltaT> <nSteps> [a dynamicMeshDict body to use in the tutorial's place]
stage()
{
    local name="$1" tutorial="$2" dt="$3" n="$4" body="${5:-}"
    local C="$W/$name"
    cp -r "$LAM/$tutorial" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    [ -z "$body" ] || printf '%s\n%s\n' "$HDR" "$body" > "$C/constant/dynamicMeshDict"
    DT="$dt" N="$n" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d = sys.argv[1]
dt, n = os.environ['DT'], int(os.environ['N'])
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('startTime', '0'), ('adjustTimeStep', 'no'), ('deltaT', dt),
                 ('endTime', '%.10g' % (n*float(dt))), ('writeControl', 'timeStep'),
                 ('writeInterval', '1'), ('writeFormat', 'ascii'), ('writePrecision', '18'),
                 ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
# three of the four tutorials write compressed files, which the comparison does not read
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    if [ -f "$C/system/blockMeshDict.m4" ]; then
        ( cd "$C" && m4 system/blockMeshDict.m4 > system/blockMeshDict ) || { echo "FAIL: m4 [$name]"; return 1; }
    fi
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    if [ -f "$C/system/snappyHexMeshDict" ]; then
        mkdir -p "$C/constant/triSurface"
        cp -f "$TUT/resources/geometry/$tutorial.obj.gz" "$C/constant/triSurface/" || { echo "FAIL: no surface for $tutorial"; return 1; }
        ( cd "$C" && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 ) || { echo "FAIL: snappyHexMesh [$name]"; tail -20 "$C/log.snappyHexMesh"; return 1; }
    fi
    ( cd "$C" && moveDynamicMesh > log.moveDynamicMesh 2>&1 ) || { echo "FAIL: moveDynamicMesh [$name]"; tail -30 "$C/log.moveDynamicMesh"; return 1; }
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    for t in 0 "$end"; do
        ( cd "$C" && postProcess -func writeCellVolumes -time "$t" > log.V 2>&1 \
                  && postProcess -func writeCellCentres -time "$t" > log.C 2>&1 ) \
            || { echo "FAIL: postProcess at $t [$name]"; tail -20 "$C/log.V"; return 1; }
    done
    [ -f "$C/$end/meshPhi" ] || { echo "FAIL: OpenFOAM wrote no $end/meshPhi [$name]"; ls "$C"; return 1; }
    echo "OpenFOAM moved the mesh $n steps of deltaT $dt to t = $end   [$name]"
}

# timeDirs <case> <deltaT> <nSteps>: the time directories, in order, as OpenFOAM named them
timeDirs()
{
    python3 - "$1" "$2" "$3" <<'PYEOF'
import sys
c, dt, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
print(' '.join('%s/%.10g' % (c, k*dt) for k in range(1, n + 1)))
PYEOF
}

# gate <name> <deltaT> <nSteps> [the case brae reads its dynamicMeshDict from, if not the oracle's]
gate()
{
    local name="$1" dt="$2" n="$3" braeCase="${4:-$W/$1}"
    # shellcheck disable=SC2046
    "$BIN" "$braeCase" "$name" "$dt" $(timeDirs "$W/$name" "$dt" "$n")
}

TTM_LINEAR='dynamicFvMesh dynamicMotionSolverFvMesh; motionSolver solidBody; solidBodyMotionFunction linearMotion; velocity (0.3 -0.2 0.11);'
TTM_AXIS='dynamicFvMesh dynamicMotionSolverFvMesh; motionSolver solidBody; solidBodyMotionFunction axisRotationMotion; axisRotationMotionCoeffs { origin (0.01 0.02 -0.03); radialVelocity (35 -80 120); }'

rc=0
stage testTubeMixer      testTubeMixer      0.01  5 || rc=1
stage sloshingTank2D     sloshingTank2D     0.05  6 || rc=1
stage sloshingTank3D6DoF sloshingTank3D6DoF 0.15  6 || rc=1
stage sloshingCylinder   sloshingCylinder   0.01  5 || rc=1
stage linear             testTubeMixer      0.01  4 "$TTM_LINEAR" || rc=1
stage axis               testTubeMixer      0.01  4 "$TTM_AXIS"   || rc=1
# the 6DoF tutorial again, its table read with linear interpolation. (On ITS tank: the table moves a
# 20 m tank by metres, and staged onto testTubeMixer's 5 cm mesh it carried the mesh 45 extents from
# the origin, where V and C lose two digits to the coordinates' size -- in both codes.)
TANK_TABLE='dynamicFvMesh dynamicMotionSolverFvMesh; motionSolver solidBody; solidBodyMotionFunction tabulated6DoFMotion; CofG (0 0 0); timeDataFileName "<constant>/6DoF.dat"; interpolationScheme linear;'
stage tabulatedLinear    sloshingTank3D6DoF 0.15  6 "$TANK_TABLE" || rc=1
[ $rc = 0 ] || { echo "mesh_motion_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path: each log names the motion it built
grep -q "Constructed SBMF 1 : rotatingBox of type oscillatingRotatingMotion" "$W/testTubeMixer/log.moveDynamicMesh" \
    || { echo "FAIL: OpenFOAM's testTubeMixer log does not show the multiMotion's second entry"; exit 1; }
grep -q "Selecting solid-body motion function SDA" "$W/sloshingTank2D/log.moveDynamicMesh" \
    || { echo "FAIL: OpenFOAM's sloshingTank2D log does not select SDA"; exit 1; }
grep -q "Selecting solid-body motion function tabulated6DoFMotion" "$W/sloshingTank3D6DoF/log.moveDynamicMesh" \
    || { echo "FAIL: OpenFOAM's sloshingTank3D6DoF log does not select tabulated6DoFMotion"; exit 1; }

gate testTubeMixer      0.01 5 || rc=1
gate sloshingTank2D     0.05 6 || rc=1
gate sloshingTank3D6DoF 0.15 6 || rc=1
gate sloshingCylinder   0.01 5 || rc=1
gate linear             0.01 4 || rc=1
gate axis               0.01 4 || rc=1
gate tabulatedLinear    0.15 6 || rc=1

# THE CONTROL: the same two motions, multiplied in the other order, must NOT be OpenFOAM's
cp -r "$W/testTubeMixer" "$W/swapped"
python3 - "$W/swapped/constant/dynamicMeshDict" <<'PYEOF' || { echo "FAIL: staging the control"; exit 1; }
import re, sys
p = sys.argv[1]
s = open(p).read()
a = re.search(r'\nrotatingTable\s*\{.*?\n\}\n', s, flags=re.S)
b = re.search(r'\nrotatingBox\s*\{.*?\n\}\n', s, flags=re.S)
assert a and b and a.end() <= b.start(), 'the two motions were not found in the order expected'
s = s[:a.start()] + b.group(0) + s[a.end():b.start()] + a.group(0) + s[b.end():]
open(p, 'w').write(s)
PYEOF
out=$(gate testTubeMixer 0.01 5 "$W/swapped")
if echo "$out" | grep -q "FAIL: the points are OpenFOAM's" && echo "$out" | grep -q "FAIL: meshPhi is OpenFOAM's"; then
    echo "ok:   CONTROL: with the multiMotion's two entries swapped, brae's points and meshPhi are NOT OpenFOAM's"
    echo "$out" | grep "worst over the run" | sed 's/^/      /'
else
    echo "FAIL: CONTROL: the multiMotion's order made no difference the gate could see"
    rc=1
fi

echo "mesh_motion_vs_openfoam: rc $rc"
exit $rc
