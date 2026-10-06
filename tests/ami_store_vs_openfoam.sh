#!/usr/bin/env bash
# The legacy AMI builder's STORE TEST against REAL OpenFOAM's AMI on a cyclicACMI pair whose two sides are
# offset by a sliver. The builder (ami_interface.cuh; simpleFoam, pimpleFoam and rhoSimpleFoam's AMI) kept a
# pair when its overlap exceeded 1e-14 of the direction's OWN face; OpenFOAM stores one when it exceeds 1e-6 of
# the OWNER patch's face (faceAreaIntersect.C:43, faceAreaWeightAMI.C:228; one AMI, built on the owner).
# THE FIXTURE: block A (0..1, 0..1), 4 cells in y, its x = 1 face the owner patch; block B (1..2), 8 cells in y,
# lowered so that B's first face starts 2.25e-7 below A's second. Each A face then overlaps a third B face by a
# sliver that is 9e-7 of the A face (under OpenFOAM's threshold: dropped) and 1.8e-6 of the B face (over it:
# a builder that divides by its own face keeps it in the neighbour's direction). A's first face and B's last
# two have no other partner: OpenFOAM leaves them UNCOVERED, and the old builder coupled A's first with a mask
# of 9e-7 and a weight re-normalised to 1.
# PREMISE, read from OpenFOAM's own log: its count of uncovered / blended / covered faces on each side.
# HELD by tests/test_ami_store_vs_openfoam.cu, whose header says what its mask bound is and is not.
. "$(dirname "$0")/ami_store_lib.sh"
C="$W/acmi"
shell "$C" ACMI1_couple ACMI2_couple
python3 - "$C/system/blockMeshDict" <<'PY'
import sys
d = 2.25e-7
y0 = 0.25 - d
y1 = y0 + 1.0
open(sys.argv[1], 'w').write('''FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.25) (1 0 0.25) (1 1 0.25) (0 1 0.25)
    (1 %.17g 0) (2 %.17g 0) (2 %.17g 0) (1 %.17g 0) (1 %.17g 0.25) (2 %.17g 0.25) (2 %.17g 0.25) (1 %.17g 0.25)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (1 4 1) simpleGrading (1 1 1)
    hex (8 9 10 11 12 13 14 15) (1 8 1) simpleGrading (1 1 1)
);
edges ();
boundary
(
    walls
    {
        type wall;
        faces
        (
            (0 4 7 3) (3 7 6 2) (1 5 4 0) (0 3 2 1) (4 5 6 7)
            (10 14 13 9) (11 15 14 10) (9 13 12 8) (8 11 10 9) (12 13 14 15)
        );
    }
    ACMI1_couple
    {
        type cyclicACMI;
        neighbourPatch ACMI2_couple;
        nonOverlapPatch ACMI1_blockage;
        faces ((2 6 5 1));
    }
    ACMI1_blockage
    {
        type wall;
        faces ((2 6 5 1));
    }
    ACMI2_couple
    {
        type cyclicACMI;
        neighbourPatch ACMI1_couple;
        nonOverlapPatch ACMI2_blockage;
        faces ((8 12 15 11));
    }
    ACMI2_blockage
    {
        type wall;
        faces ((8 12 15 11));
    }
);
mergePatchPairs ();
''' % (y0, y0, y1, y1, y0, y0, y1, y1))
PY
dump "$C"
SRC=$(sed -n -E 's/^ACMI: Patch source uncovered\/blended\/covered = (.*)$/\1/p' "$C/log.postProcess" | head -1)
TGT=$(sed -n -E 's/^ACMI: Patch target uncovered\/blended\/covered = (.*)$/\1/p' "$C/log.postProcess" | head -1)
[ "$SRC" = "1, 3, 0" ] && [ "$TGT" = "2, 6, 0" ] \
    && say "ok:" "PREMISE  OpenFOAM's uncovered/blended/covered faces: owner $SRC, neighbour $TGT" \
    || say "FAIL:" "PREMISE  OpenFOAM reports owner '${SRC:-?}', neighbour '${TGT:-?}'; 1, 3, 0 and 2, 6, 0 expected"
BRAE_ALLOW_ACMI=1 "$BIN" "$C" "$C/ami_dump.txt" ACMI1_couple ACMI2_couple acmi > "$W/log.brae" 2>&1
e=$?
grep -E "^  (ok:|FAIL:)|CONTROL|PASS|FAIL" "$W/log.brae"
[ $e -eq 0 ] && [ "$(grep -c '^  ok:' "$W/log.brae")" = 3 ] || rc=1
echo "ami_store_vs_openfoam: rc $rc"
exit $rc
