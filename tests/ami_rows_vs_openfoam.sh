#!/usr/bin/env bash
# The legacy AMI builder's ROWS on a plain cyclicAMI against REAL OpenFOAM's: OpenFOAM divides a face's
# weights by their SUM (AMIInterpolation::normaliseWeights with requireMatch, the default); the builder left
# them overlap/|Sf|, which sums to the coverage. With the store test made OpenFOAM's
# (ami_store_vs_openfoam.sh) a face that loses a sliver of fraction f would read (1 - f) of its neighbour's
# value from then on; divided by the sum it reads exactly what OpenFOAM's does.
# THE FIXTURE: block A (0..1, 0..1), 4 equal cells in y; block B (1..2) as four blocks whose y nodes are
# 0, 0.25 - d, 0.5, 0.75 + d, 1 with d = 2.25e-7. A's first and last faces overlap a second B face by a sliver
# of 9e-7 of the face, and B's two middle faces a second A face by the same sliver -- under OpenFOAM's
# threshold on every pair, so OpenFOAM has ONE partner a face, weight exactly 1. The two sides' face centres
# have the same mean (0.5), so the builder's inferred separation is zero and does not move the target.
# PREMISE, read from OpenFOAM's own log: the smallest sum of weights on the owner is below 1 (a sliver was
# dropped), and every face has a partner.
# HELD by tests/test_ami_store_vs_openfoam.cu: the partner sets, the weights within 1e-12, and three controls.
. "$(dirname "$0")/ami_store_lib.sh"
C="$W/ami"
shell "$C" AMI1 AMI2
python3 - "$C/system/blockMeshDict" <<'PY'
import sys
d = 2.25e-7
ys = [0.0, 0.25 - d, 0.5, 0.75 + d, 1.0]
v = ['(0 0 0)', '(1 0 0)', '(1 1 0)', '(0 1 0)', '(0 0 0.25)', '(1 0 0.25)', '(1 1 0.25)', '(0 1 0.25)']
# block B's vertices: index 8 + 4*j + (0: x=1,z=0; 1: x=2,z=0; 2: x=1,z=.25; 3: x=2,z=.25)
for y in ys:
    for x, z in ((1, 0), (2, 0), (1, 0.25), (2, 0.25)):
        v.append('(%d %.17g %g)' % (x, y, z))
def b(j, k):
    return 8 + 4*j + k
blocks = []
ami2 = []
walls = ['(0 4 7 3)', '(3 7 6 2)', '(1 5 4 0)', '(0 3 2 1)', '(4 5 6 7)']
for j in range(4):
    blocks.append('hex (%d %d %d %d %d %d %d %d) (1 1 1) simpleGrading (1 1 1)'
                  % (b(j, 0), b(j, 1), b(j + 1, 1), b(j + 1, 0), b(j, 2), b(j, 3), b(j + 1, 3), b(j + 1, 2)))
    ami2.append('(%d %d %d %d)' % (b(j, 0), b(j, 2), b(j + 1, 2), b(j + 1, 0)))
    walls.append('(%d %d %d %d)' % (b(j + 1, 1), b(j + 1, 3), b(j, 3), b(j, 1)))
    walls.append('(%d %d %d %d)' % (b(j, 0), b(j + 1, 0), b(j + 1, 1), b(j, 1)))
    walls.append('(%d %d %d %d)' % (b(j, 2), b(j, 3), b(j + 1, 3), b(j + 1, 2)))
walls.append('(%d %d %d %d)' % (b(0, 1), b(0, 3), b(0, 2), b(0, 0)))
walls.append('(%d %d %d %d)' % (b(4, 0), b(4, 2), b(4, 3), b(4, 1)))
open(sys.argv[1], 'w').write('''FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
    %s
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (1 4 1) simpleGrading (1 1 1)
    %s
);
edges ();
boundary
(
    walls
    {
        type wall;
        faces
        (
            %s
        );
    }
    AMI1
    {
        type cyclicAMI;
        neighbourPatch AMI2;
        transform noOrdering;
        faces ((2 6 5 1));
    }
    AMI2
    {
        type cyclicAMI;
        neighbourPatch AMI1;
        transform noOrdering;
        faces
        (
            %s
        );
    }
);
mergePatchPairs ();
''' % ('\n    '.join(v), '\n    '.join(blocks), '\n            '.join(walls), '\n            '.join(ami2)))
PY
dump "$C"
LOW=$(sed -n -E 's/^AMI: Patch source sum\(weights\) min:([0-9.e+-]+) .*/\1/p' "$C/log.postProcess" | head -1)
NONE=$(awk '$4 == 0 && NF == 4' "$C/ami_dump.txt" | wc -l)
python3 -c "import sys; sys.exit(0 if 0.99999 < float('${LOW:-0}') < 1.0 else 1)" && [ "$NONE" = 0 ] \
    && say "ok:" "PREMISE  OpenFOAM's smallest owner sum of weights is $LOW (a sliver dropped); no face unpartnered" \
    || say "FAIL:" "PREMISE  OpenFOAM's smallest owner sum of weights is '${LOW:-?}', $NONE faces unpartnered"
"$BIN" "$C" "$C/ami_dump.txt" AMI1 AMI2 ami > "$W/log.brae" 2>&1
e=$?
grep -E "^  (ok:|FAIL:)|CONTROL|PASS|FAIL" "$W/log.brae"
[ $e -eq 0 ] && [ "$(grep -c '^  ok:' "$W/log.brae")" = 3 ] || rc=1
echo "ami_rows_vs_openfoam: rc $rc"
exit $rc
