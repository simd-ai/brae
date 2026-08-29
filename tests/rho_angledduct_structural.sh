#!/usr/bin/env bash
# Stages OpenFOAM's angledDuctExplicitFixedCoeff tutorial and runs the STRUCTURAL measurements on it.
#
# The tutorial lives in $FOAM_TUTORIALS rather than validation/ because the mesh is generated -- blockMesh
# is deterministic and 28000 cells is not something to carry in the repo. The three named blocks ARE the
# porosity cellZone (the case runs no topoSet), which is what makes the fvOption and the wall function
# land on the same cells; a mesh without them would silently drop the overlap the ordering acts on, so the
# binary asserts it.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_angledduct_structural"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
cd "$W/case" || exit 1
rm -rf 0 [1-9]* processor* log.*

blockMesh > blockMesh.log 2>&1 \
    || { echo "FAIL: blockMesh did not run"; tail -20 blockMesh.log; exit 1; }
[ -f constant/polyMesh/owner ] || { echo "FAIL: blockMesh wrote no mesh"; exit 1; }

# THE POROSITY CELLZONE MUST EXIST. Without it the fvOption selects nothing, the overlap the ordering acts
# on is empty, and every measurement below goes vacuous while still passing.
grep -q "porosity" constant/polyMesh/cellZones 2>/dev/null \
    || { echo "FAIL: blockMesh produced no porosity cellZone"; exit 1; }

"$BIN" "$W/case" 0.orig
