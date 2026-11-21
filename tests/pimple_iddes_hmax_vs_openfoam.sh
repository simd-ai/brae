#!/usr/bin/env bash
# brae's IDDES hmax refusal against REAL OpenFOAM's hmax.
#
# brae's IDDES takes hmax as the cell's vertex bounding box (cell_max_delta.cuh cellMaxDeltaXYZ); OpenFOAM's IDDESDelta
# takes the LESdelta its IDDESDeltaCoeffs names -- maxDeltaxyz, deltaCoeff * max |n_f . (C_f - C)| (maxDeltaxyz.C:46-80),
# by default, or maxDeltaxyzCubeRoot. brae_pimpleFoam refuses an IDDES case when the two differ on its mesh by more than
# round-off, and says by how much (gpuPimpleFoam.cu refuseLesOnMesh). This gate holds both halves of that against
# OpenFOAM's own hmax field, written by `pimpleFoam -postProcess` with writeObjects:
#   box     a Cartesian box (tests/pimple_box_case.py): OpenFOAM's hmax and the bounding box agree to round-off, and brae
#           sets the case up
#   skew    the same box sheared, x += 1.5 y: they differ; brae refuses, and the largest difference it reports is the one
#           measured here between the bounding box and OpenFOAM's field
# for hmax maxDeltaxyz (IDDESDeltaCoeffs empty) and maxDeltaxyzCubeRoot.
#
# TOLERANCES, declared: agreement is a relative difference below 1e-10 (the refusal's own bound; two sums of the same
# coordinates); brae's reported figure must equal this gate's to 1e-6 relative (it prints six decimals of a percentage).
# MEASURED on the tutorials with the same method (2026-11-21, GB10): LES/surfaceMountedCube/fullCase 2.5e-13 (sets up),
# LES/periodicHill/transient 38.6 %, LES/wallMountedHump 152.9 % -- the figures brae reported at set-up.
# CONTROL, asserted: BRAE_CONTROL_LES_MESH_UNCHECKED=1 sets the sheared case up.
set -u
BIN="${1:?brae_pimpleFoam binary}"
W="${2:?work dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
rm -rf "$W"
mkdir -p "$W"
fail=0

# of_hmax <case>: OpenFOAM's hmax at 0, then the largest relative difference of the bounding box from it
of_hmax()
{
    python3 - "$1/system/controlDict" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
s += ('\nwriteFormat ascii;\nwritePrecision 17;\nfunctions { hm { type writeObjects; libs (utilityFunctionObjects); '
      'objects (hmax); writeOption anyWrite; } }\n')
open(p, 'w').write(s)
PY
    (cd "$1" && nice -n 10 pimpleFoam -postProcess -time 0 > log.of 2>&1) || { echo "OpenFOAM failed"; return 1; }
    python3 - "$1" <<'PY'
import re, sys
c = sys.argv[1]
def lst(path):
    s = open(path).read()
    s = s[s.index('}', s.index('FoamFile')) + 1:]
    m = re.search(r'(\d+)\s*\(', s)
    return s[m.end():s.rindex(')')]
pts = [tuple(map(float, t.split())) for t in re.findall(r'\(([^()]*)\)', lst(c + '/constant/polyMesh/points'))]
faces = [list(map(int, t.split())) for t in re.findall(r'\d+\(([^()]*)\)', lst(c + '/constant/polyMesh/faces'))]
own = list(map(int, lst(c + '/constant/polyMesh/owner').split()))
nei = list(map(int, lst(c + '/constant/polyMesh/neighbour').split()))
n = max(own) + 1
lo = [[1e300] * 3 for _ in range(n)]
hi = [[-1e300] * 3 for _ in range(n)]
for cells in (own, nei):
    for f, cell in enumerate(cells):
        for v in faces[f]:
            for d in range(3):
                lo[cell][d] = min(lo[cell][d], pts[v][d])
                hi[cell][d] = max(hi[cell][d], pts[v][d])
s = open(c + '/0/hmax').read()
seg = s[s.index('internalField'):s.index('boundaryField')]
m = re.search(r'List<scalar>\s*\d+\s*\(', seg)
of = list(map(float, seg[m.end():seg.rindex(')')].split())) if m else [float(re.search(r'uniform\s+(\S+);', seg).group(1))] * n
worst = max(abs(max(hi[i][d] - lo[i][d] for d in range(3)) - of[i]) / of[i] for i in range(n))
print('%.9e' % worst)
PY
}

for hmax in maxDeltaxyz maxDeltaxyzCubeRoot; do
    for shape in box skew; do
        c="$W/${shape}_$hmax"
        if [ "$shape" = skew ]; then
            python3 "$ROOT/tests/pimple_box_case.py" "$c" --model SAIDDES --skew 1.5 || exit 1
        else
            python3 "$ROOT/tests/pimple_box_case.py" "$c" --model SAIDDES || exit 1
        fi
        if [ "$hmax" = maxDeltaxyzCubeRoot ]; then
            sed -i 's/delta IDDESDelta;/delta IDDESDelta;\n    IDDESDeltaCoeffs { hmax maxDeltaxyzCubeRoot; maxDeltaxyzCubeRootCoeffs { } }/' \
                "$c/constant/turbulenceProperties"
        fi
        # OpenFOAM's wallDist reads `method` with no default, and IDDESDelta asks for the wall normals
        printf 'wallDist { method meshWave; nRequired yes; }\n' >> "$c/system/fvSchemes"
        cp -r "$c" "$c.brae"
        rel=$(of_hmax "$c")
        env BRAE_CHECK_SETUP=1 "$BIN" -case "$c.brae" > "$c.brae.log" 2>&1
        rc=$?
        said=$(grep -oE 'by up to [0-9.e+-]+ %' "$c.brae.log" | awk '{print $4}')
        if [ "$shape" = box ]; then
            if python3 -c "import sys; sys.exit(0 if float('$rel') < 1e-10 else 1)" && [ "$rc" -eq 0 ]; then
                printf '  %-34s OpenFOAM vs bounding box %s, brae sets up              ok\n' "$shape $hmax" "$rel"
            else
                printf '  %-34s OpenFOAM vs bounding box %s, brae exit %s   FAIL\n' "$shape $hmax" "$rel" "$rc"
                fail=1
            fi
        else
            if [ "$rc" -ne 0 ] && [ -n "$said" ] \
                && python3 -c "import sys; a=float('$rel')*100; b=float('$said'); sys.exit(0 if a > 1e-8 and abs(a-b) <= 1e-6*a else 1)"; then
                printf '  %-34s OpenFOAM vs bounding box %s, brae refuses at %s %%   ok\n' "$shape $hmax" "$rel" "$said"
            else
                printf '  %-34s OpenFOAM vs bounding box %s, brae exit %s says %s   FAIL\n' "$shape $hmax" "$rel" "$rc" "${said:-nothing}"
                fail=1
            fi
        fi
    done
done
# the control: the check off, the sheared case sets up
env BRAE_CHECK_SETUP=1 BRAE_CONTROL_LES_MESH_UNCHECKED=1 "$BIN" -case "$W/skew_maxDeltaxyz.brae" > "$W/control.log" 2>&1
if [ $? -eq 0 ] && grep -qF "CONTROL MODE" "$W/control.log"; then
    echo "  CONTROL BRAE_CONTROL_LES_MESH_UNCHECKED=1: the sheared case sets up (red, as it must be)"
else
    echo "  CONTROL: the sheared case did not set up with the check off   FAIL"
    fail=1
fi
[ "$fail" -eq 0 ] && echo "PASS: brae refuses IDDES exactly where its hmax is not OpenFOAM's, and says by how much"
exit "$fail"
