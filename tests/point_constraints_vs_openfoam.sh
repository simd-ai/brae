#!/usr/bin/env bash
# brae's PointConstraints against REAL OpenFOAM's pointConstraints::constrainDisplacement -- the call
# rigidBodyMeshMotion::solve ends every mesh motion with (rigidBodyMeshMotion.C:385-388): the point patch
# fields evaluated in patch order (fixedValue pins; a symmetryPlane writes (pif + transform(I - 2nn, pif))/2),
# then every constrained point transformed by its combined constraint (I - nn, ll or 0).
#
# THE ORACLE is tools/dumpPointConstraints: OpenFOAM's own class on the same mesh and the same input. With
# -synthetic it replaces the internal field by a smooth polynomial of the point position, nonzero in every
# component on every patch, and writes it (pointDisplacement.input) so brae starts from the same bits.
#
# THREE ARMS, because the tutorial cannot witness everything:
#   A  RAS/DTCHullMoving's own mesh (848,022 cells, meshed serially as its Allrun does) and its own point
#      patches: three fixedValue (0 0 0), bottom/side/midPlane symmetryPlane, hull calculated. Its normals
#      are NOT exactly axis-aligned -- gSum(faceAreas) leaves bottom (5.1e-21 8.9e-18 -1) and side
#      (1.2e-21 -1 5.3e-29) -- so the normal's bits are witnessed. 1,750 constrained points: 1,664 on one
#      plane, 86 on two (orthogonal ones), none on three. MEASURED, OpenFOAM against itself: the evaluate moves
#      43,401 points (worst 7.3); constrainCorners 82 more, in the LAST BIT only (worst 1.1e-16).
#   B  A with the three fixedValue patches pinned OFF zero, so a point on a pin and a plane is witnessed
#      (the pin written first, then projected): the order of the evaluate.
#   C  a parallelepiped (blockMesh, 8^3) whose three TILTED symmetry planes meet at one corner: two-plane
#      edges that are not orthogonal, a three-plane corner, and two pinned faces after them. What DTCHullMoving
#      cannot show -- the cross-product line, the 1e-3 thresholds, the count-3 zero -- is shown here.
#
# THE STANDARD is bit for bit: every normal, every constrained point in OpenFOAM's order with its count and
# tensor, the field after the evaluate and after the whole constraint. CONTROLS in the binary, each a broken
# form asserted to differ from OpenFOAM: the symmetryPlane evaluate skipped (A 35,821 points, B 36,615,
# C 184); constrainCorners skipped (A 82 in the last bit, B 499, C 69 at up to 2.8e-02); the pins written after
# the planes (B, 794 points, 2.0e-02); the three-plane corner left free (C, 1 point, 1.0e-03).
#
# FAIL-PROOFS, each a scratch build of the module with one edit, run over the three arms:
#   the normal divided by a reciprocal multiply    A, B blind; C: 3 normals and 15 tensors differ
#   the evaluate as v - (n.v)n                     A, B 221 points; C 142
#   the pins never written                         A, B 7,580 points; C 153
#   the three-plane corner never reached           A, B blind; C: its count and its tensor
#   the plane's second normal kept instead of the cross product line
#                                                  A, B 86 tensors; C 24
# So A and B alone cannot see the normal's division or the count-3 case: C is what holds them.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_point_constraints_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/DTCHullMoving"
STL="$TUT/resources/geometry/DTC-scaled.stl.gz"
MODE=${MEASURE:+measure}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/DTCHullMoving tutorial not found at $SRC"; exit 77; }
[ -f "$STL" ]      || { echo "SKIP: the DTC hull surface is not at $STL"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in snappyHexMesh blockMesh dumpPointConstraints; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

# the minimal system/ a mesh-reading utility needs, ascii at full precision for the oracle's fields
sysdir()
{
    mkdir -p "$1/system"
    cat > "$1/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application interFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
writeFormat ascii;
writePrecision 18;
EOF
    cat > "$1/system/fvSchemes" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
EOF
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n' > "$1/system/fvSolution"
}

rc=0
# arm <name> <caseDir> [env]
arm()
{
    local name="$1" C="$2"
    ( dumpPointConstraints -case "$C" -time 0 -synthetic > "$C/log.dump" 2>&1 ) \
        || { echo "FAIL: dumpPointConstraints [$name]"; tail -20 "$C/log.dump"; rc=1; return; }
    echo "== [$name]"
    env ${3:-BRAE_GATE_PLAIN=1} "$BIN" "$C" "$C/log.dump" $MODE || rc=1
}

# --- A and B: DTCHullMoving's mesh ------------------------------------------------------------------------
M="$W/mesh"
if [ ! -f "$M/constant/polyMesh/owner" ]; then
    rm -rf "$M"
    cp -r "$SRC" "$M" || exit 1
    rm -rf "$M"/[1-9]* "$M"/0 "$M"/processor* "$M"/log.*
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
        rm -rf 0
        cp -r 0.orig 0
        setFields > log.setFields 2>&1 || exit 1
        renumberMesh -overwrite > log.renumberMesh 2>&1 || exit 1
    ) || { echo "FAIL: meshing DTCHullMoving"; ls "$M"; exit 1; }
fi

for a in A B; do
    C="$W/arm$a"
    rm -rf "$C"
    mkdir -p "$C/constant" "$C/0"
    cp -r "$M/constant/polyMesh" "$C/constant/"
    sysdir "$C"
    # renumberMesh wrote 0/pointDisplacement with every patch by name (the shipped file keys the symmetry
    # planes by group, through setConstraintTypes)
    cp "$M/0/pointDisplacement" "$C/0/"
    grep -q "symmetryPlane" "$C/0/pointDisplacement" \
        || { echo "FAIL: DTCHullMoving's pointDisplacement names no symmetryPlane"; exit 1; }
done
ARMB="$W/armB/0/pointDisplacement" python3 - <<'PYEOF' || { echo "FAIL: staging arm B"; exit 1; }
import os, re
p = os.environ['ARMB']
t = open(p, encoding='latin-1').read()
pins = {'atmosphere': '(0.01 -0.02 0.03)', 'inlet': '(-0.015 0.005 0.02)', 'outlet': '(0.02 0.01 -0.01)'}
for name, v in pins.items():
    t, k = re.subn(r'(\n\s*%s\s*\{\s*type\s+fixedValue;\s*value\s+)uniform\s*\([^)]*\);' % name,
                   r'\g<1>uniform %s;' % v, t)
    assert k == 1, name
open(p, 'w', encoding='latin-1').write(t)
PYEOF

arm "A: DTCHullMoving" "$W/armA"
arm "B: DTCHullMoving, pins off zero" "$W/armB" BRAE_EXPECT_PIN_ORDER=1

# --- C: three tilted symmetry planes meeting at a corner --------------------------------------------------
C="$W/armC"
rm -rf "$C"
mkdir -p "$C/constant" "$C/0"
sysdir "$C"
# a parallelepiped on a = (1 0.2 0.1), b = (0.3 1 0.15), c = (0.1 0.25 1): every face planar, no two
# adjacent faces orthogonal
cat > "$C/system/blockMeshDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
vertices
(
    (0 0 0)
    (1 0.2 0.1)
    (1.3 1.2 0.25)
    (0.3 1 0.15)
    (0.1 0.25 1)
    (1.1 0.45 1.1)
    (1.4 1.45 1.25)
    (0.4 1.25 1.15)
);
blocks ( hex (0 1 2 3 4 5 6 7) (8 8 8) simpleGrading (1 1 1) );
boundary
(
    xmin { type symmetryPlane; faces ( (0 4 7 3) ); }
    ymin { type symmetryPlane; faces ( (1 5 4 0) ); }
    zmin { type symmetryPlane; faces ( (0 3 2 1) ); }
    xmax { type patch; faces ( (2 6 5 1) ); }
    ymax { type wall; faces ( (3 7 6 2) ); }
    zmax { type patch; faces ( (4 5 6 7) ); }
);
EOF
cat > "$C/0/pointDisplacement" <<'EOF'
FoamFile { version 2.0; format ascii; class pointVectorField; object pointDisplacement; }
dimensions [0 1 0 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    xmin { type symmetryPlane; }
    ymin { type symmetryPlane; }
    zmin { type symmetryPlane; }
    xmax { type fixedValue; value uniform (0.01 0.02 0.03); }
    ymax { type calculated; }
    zmax { type fixedValue; value uniform (-0.02 0.01 0.005); }
}
EOF
( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [C]"; tail -20 "$C/log.blockMesh"; exit 1; }
# no pin-order expectation here: C's pinned patches come AFTER its planes in patch order, so the pins are
# written last in OpenFOAM too and that control reads 0 (measured) -- arm B holds it
arm "C: three tilted planes" "$C"

echo "point_constraints_vs_openfoam: rc $rc"
exit $rc
