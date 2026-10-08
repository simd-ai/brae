#!/usr/bin/env bash
# `snGradSchemes` AND `laplacianSchemes` ARE TWO BLOCKS FOR TWO OPERATORS, checked against real
# OpenFOAM on a case where they DISAGREE.
#
# OpenFOAM, read rather than remembered:
#   - An fvm::laplacian entry carries its OWN snGrad scheme, built from that entry's Istream
#     (laplacianScheme.H:121-141 -> snGradScheme<Type>::New). `Gauss linear orthogonal` in
#     laplacianSchemes governs every laplacian in the solver and nothing else.
#   - fvc::snGrad(vf) looks the field up in snGradSchemes instead (fvcSnGrad.C:56-64 ->
#     schemesLookup.C:249-253). The laplacian never reads that block; snGrad never reads the
#     laplacian's.
#   - snGradSchemes is OPTIONAL and its own default is `corrected` (schemesLookup.C:82,
#     populate(dict, "corrected")), while laplacianSchemes is MANDATORY (:83). So deleting the block
#     does NOT make fvc::snGrad follow the laplacian -- it makes it corrected.
#
# brae collapsed the two: scheme_parse.cuh:695 read `st.block == "laplacianSchemes" ||
# st.block == "snGradSchemes"` into ONE pair of flags. Two consequences, both silent:
#   1. A `corrected` in EITHER block set ctl.nonOrth, and nothing could clear it -- a sticky OR. This
#      fixture's `laplacianSchemes { default Gauss linear orthogonal; }` ran CORRECTED, because the
#      snGradSchemes line below it said so. Every fvm::laplacian in the solver, under the orthogonal
#      name the case asked for.
#   2. fvc::snGrad(p) -- rhoSimpleFoam's only fvc::snGrad, in the SIMPLEC correction
#      (pcEqn.H:64, phiHbyA += interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)*magSf) -- read the
#      LAPLACIAN's flag on both arms. Audit 2026-09-15.
#
# WHY NO EXISTING FIXTURE COULD SEE IT. Three things have to hold at once, and nothing in validation/
# had all three: `consistent yes` (without SIMPLEC this solver calls fvc::snGrad nowhere), a
# NON-ORTHOGONAL mesh (on a box the correction vectors are zero and corrected == orthogonal), and the
# two blocks DISAGREEING. rhoCtl and rhoPM have SIMPLEC and disagreeing blocks -- and are perfect
# boxes. rhoLimKE is tilted -- and is plain SIMPLE with both blocks equal, deliberately so, because
# it gates the laplacian limiter and a disagreement there would confound it. validation/rhoSnGrad is
# rhoLimKE's mesh (16.3 degrees maximum non-orthogonality) with SIMPLEC on and the blocks split.
#
#   ARM p    max |p_brae - p_OF| / max |p_OF| at iteration 1 from OpenFOAM's own developed state.
#   ARM U     the same for U. Gated LOOSELY and deliberately -- see WHAT THIS DOES NOT CLAIM below --
#             because the collapse moves it by two orders and nothing else on this fixture does.
#   CONTROL   OpenFOAM's OWN p must MOVE between `snGradSchemes corrected` and `orthogonal` by at
#             least CONTROL_MIN. If it does not, this mesh cannot tell the two schemes apart and both
#             arms are vacuous. Also printed: BASELINE, the same comparison with the two blocks made
#             to AGREE, which is what this fixture scores when the collapse cannot bite.
#
# MEASURED 2026-09-15, and the two halves of the fix are independently load-bearing:
#
#   | configuration                                  | ARM p (cuda) | ARM p (cpu) | ARM U (cuda) |
#   |------------------------------------------------|--------------|-------------|--------------|
#   | fixed                                          |     4.14e-09 |    3.79e-08 |     2.91e-04 |
#   | FAILPROOF a: inLap takes snGradSchemes again   |     4.05e-06 |           - |     2.77e-02 |
#   | FAILPROOF b: fvc::snGrad(p) reads the laplacian|     2.06e-06 |    2.08e-06 |     1.84e-02 |
#
# 977x on p and 95x on U for the parser collapse; 500x and 63x for the routing alone. OpenFOAM's own
# p moves 2.12e-06 relative between the two snGradSchemes, which is the signal being measured.
#
# WHAT THIS GATE DOES NOT CLAIM. U carries a LARGE pre-existing gap on this mesh -- 2.9e-04 here, and
# 4.2e-04 to 3.1e-03 over `corrected`/`orthogonal`/`uncorrected` laplacians, all measured with the fix
# in. It needs a non-orthogonal mesh: the SAME case on a FLAT mesh gives p 5.04e-12, U 1.92e-09,
# T 1.62e-12. What it is BEYOND that is not established -- replacing the symmetryPlane with `slip`
# left U at 2.37e-04, but slip and symmetryPlane are the same transform family, so that ruled nothing
# out. A no-slip wall on the slanted patch is the experiment that has not been run.
# So the U bound below is set to catch a two-order blow-up and nothing finer, and rhoSimpleFoam's
# iteration-1 U on a non-orthogonal mesh is an open gap this gate does not close.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ARM=${ARM:-cuda}
DEV=${DEV:-500}
# 2e-07 passes the worse of the two arms (cpu 3.79e-08) with 5x margin and the agreeing-blocks
# BASELINE (8.27e-08) with 2.4x, and fails the weaker of the two fail-proofs (2.06e-06) by 10x.
BOUND=${BOUND:-2e-7}
# 1e-03 sits 3.4x above the measured U (2.91e-04) and 18x below the weaker fail-proof (1.84e-02).
BOUND_U=${BOUND_U:-1e-3}
CONTROL_MIN=${CONTROL_MIN:-1e-7}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoSnGrad" ] || { echo "SKIP: validation/rhoSnGrad missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

# $1 dir, $2 snGradSchemes default, $3 endTime, $4 optional seed time dir
prep()
{
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$ROOT/validation/rhoSnGrad/constant" "$ROOT/validation/rhoSnGrad/system" "$1/"
    if [ $# -ge 4 ]; then cp -r "$4" "$1/0"; rm -rf "$1/0/uniform"
    else cp -r "$ROOT/validation/rhoSnGrad/0.orig" "$1/0"; fi
    ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "blockMesh failed"; exit 1; }
    sed -i "s/^snGradSchemes    { default corrected; }/snGradSchemes    { default $2; }/" "$1/system/fvSchemes"
    grep -q "snGradSchemes    { default $2; }" "$1/system/fvSchemes" \
        || { echo "  could not set snGradSchemes to '$2'"; exit 1; }
    # ...and the LAPLACIAN block, for the `uncorrected` arms. Via an env var, so the optional 4th
    # positional (the seed directory) keeps its place.
    if [ "${LAPW:-orthogonal}" != orthogonal ]; then
        sed -i "s/^laplacianSchemes { default Gauss linear orthogonal; }/laplacianSchemes { default Gauss linear ${LAPW}; }/" "$1/system/fvSchemes"
        grep -q "laplacianSchemes { default Gauss linear ${LAPW}; }" "$1/system/fvSchemes" \
            || { echo "  could not set laplacianSchemes to '${LAPW}'"; exit 1; }
    fi
    END="$3" python3 - "$1" <<'PREPPY'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict'); s = open(c).read()
s = re.sub(r'startTime \d+;', 'startTime 0;', s)
s = re.sub(r'endTime \d+;', 'endTime %s;' % os.environ['END'], s)
s = re.sub(r'writeInterval \d+;', 'writeInterval %s;' % os.environ['END'], s)
s = re.sub(r'writePrecision \d+;', 'writePrecision 15;', s)
open(c, 'w').write(s)
if os.environ['END'] == '1':
    v = os.path.join(sys.argv[1], 'system/fvSolution'); s = open(v).read()
    open(v, 'w').write(re.sub(r'residualControl \{[^}]*\}', '', s))
PREPPY
}

echo "rho snGradSchemes vs laplacianSchemes, against real OpenFOAM (arm: $ARM)"

# A uniform field has no gradient, so the non-orthogonal correction is zero and nothing could
# discriminate from 0.orig. Develop first, with OpenFOAM, and seed BOTH codes from the same state.
prep "$W/dev" corrected "$DEV"
( cd "$W/dev" && rhoSimpleFoam > log.run 2>&1 ) || { echo "  developing the seed failed"; tail -4 "$W/dev/log.run"; exit 1; }
SEEDT=$(ls -d "$W/dev"/[1-9]* 2>/dev/null | xargs -n1 basename | sort -n | tail -1)
[ -n "$SEEDT" ] || { echo "  no developed state written"; exit 1; }
printf '  seed state: OpenFOAM at iteration %s\n' "$SEEDT"

# `uncorrected` IS NOT `orthogonal`, AND THAT IS THE POINT OF THE LAST FOUR RUNS.
# uncorrectedSnGrad.H:113-119 returns mesh().nonOrthDeltaCoeffs() exactly as correctedSnGrad.H:108-114
# does and differs only in corrected(); only orthogonalSnGrad.H:113-119 returns deltaCoeffs(). brae's
# SHARED parser tested `hasWord(ln, "corrected")`, which is word-boundaried, so the word `uncorrected`
# matched nothing and the flag kept its default -- rhoSimpleFoam ran ORTHOGONAL under the case's own name,
# in every one of its laplacians AND in the SIMPLEC fvc::snGrad(p).
#
# ONE WORD PER BLOCK, so each arm attributes to one half. THEIR CONTROLS ARE THE RUNS ALREADY STAGED:
#   lapUnco (lap uncorrected, sng corrected)  -- control of_split (lap orthogonal, sng corrected)
#   sngUnco (lap orthogonal,  sng uncorrected) -- control of_agree (lap orthogonal, sng orthogonal)
# MEASURED, OpenFOAM against OpenFOAM at this gate's own seed and single iteration:
#   laplacian half  p 1.187800e-06 (5.9x BOUND), U 5.050600e-03 (5.1x BOUND_U), T 2.53e-09, 1200/1200 cells
#   snGrad half     p 6.520300e-07 (3.3x BOUND), U 2.565500e-03 (2.6x BOUND_U), T EXACTLY 0.0
# T being exactly zero on the snGrad half is itself a check that the blocks stay separate: the energy
# equation reads no fvc::snGrad, so its answer cannot move when only that block changes.
for spec in "of_split|corrected|of|orthogonal" "br_split|corrected|br|orthogonal" \
            "of_agree|orthogonal|of|orthogonal" "br_agree|orthogonal|br|orthogonal" \
            "of_lapUnco|corrected|of|uncorrected" "br_lapUnco|corrected|br|uncorrected" \
            "of_sngUnco|uncorrected|of|orthogonal" "br_sngUnco|uncorrected|br|orthogonal"; do
    IFS='|' read -r tag sng who lapw <<< "$spec"
    LAPW="$lapw" prep "$W/$tag" "$sng" 1 "$W/dev/$SEEDT"
    if [ "$who" = of ]; then
        ( cd "$W/$tag" && rhoSimpleFoam > log.run 2>&1 ) \
            || { echo "  OpenFOAM ($tag) failed"; tail -4 "$W/$tag/log.run"; exit 1; }
    else
        ( cd "$W/$tag" && BRAE_RHOSIMPLEFOAM_MIRROR="$ARM" "$BIN" -case "$W/$tag" > log.run 2>&1 ) \
            || { echo "  brae ($tag) failed"; tail -4 "$W/$tag/log.run"; exit 1; }
    fi
    [ -f "$W/$tag/1/p" ] || { echo "  $tag wrote no 1/p"; exit 1; }
done

python3 - "$W" "$BOUND" "$CONTROL_MIN" "$BOUND_U" <<'CMPPY'
import re, sys, math
W, bound, cmin, boundU = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
def rd(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*\n(\d+)\s*\n\(\n(.*?)\n\)\n;', s, re.S)
    if not m: return None
    if m.group(1) == 'scalar':
        return [float(x) for x in m.group(3).split('\n')]
    return [[float(y) for y in l.strip().strip('()').split()] for l in m.group(3).split('\n')]
def rel(a, b):
    if isinstance(a[0], list):
        num = max(math.dist(x, y) for x, y in zip(a, b)); den = max(math.dist(x, [0,0,0]) for x in a)
    else:
        num = max(abs(x-y) for x, y in zip(a, b)); den = max(abs(x) for x in a)
    return num / max(den, 1e-300)

bad = 0
# CONTROL: OpenFOAM itself must answer differently under the two snGradSchemes, or the fixture is
# blind and the arm below proves nothing.
ops, opa = rd(W + '/of_split/1/p'), rd(W + '/of_agree/1/p')
ofmove = rel(ops, opa)
print('  CONTROL  OpenFOAM p, corrected vs orthogonal snGrad  %-12.6e (needs >= %-9.3e) %s'
      % (ofmove, cmin, 'OK' if ofmove >= cmin else 'FAIL'))
if ofmove < cmin: bad = 1

# FLOOR: the agreeing-blocks run, where the collapse cannot bite -- what this fixture can do at best.
floor = rel(opa, rd(W + '/br_agree/1/p'))
print('  BASELINE blocks agree (both orthogonal), p           %-12.6e' % floor)

arm = rel(ops, rd(W + '/br_split/1/p'))
print('  ARM p    blocks DISAGREE                             %-12.6e (bound %-9.3e) %s'
      % (arm, bound, 'OK' if arm <= bound else 'FAIL'))
if arm > bound: bad = 1

armU = rel(rd(W + '/of_split/1/U'), rd(W + '/br_split/1/U'))
print('  ARM U    blocks DISAGREE                             %-12.6e (bound %-9.3e) %s'
      % (armU, boundU, 'OK' if armU <= boundU else 'FAIL'))
if armU > boundU: bad = 1

print('  reported T                                           %-12.6e  (not gated)'
      % rel(rd(W + '/of_split/1/T'), rd(W + '/br_split/1/T')))

# `uncorrected`: nonOrthDeltaCoeffs with NO correction flux, which brae read as `orthogonal`. One arm per
# block, each against the run that differs from it in that block alone, and each with its own OF-vs-OF
# control -- the shipped `orthogonal` answer, i.e. precisely what brae used to compute under the name.
for label, armTag, ctlTag, ctlName in (('laplacian', 'lapUnco', 'of_split', 'orthogonal laplacian'),
                                       ('snGrad   ', 'sngUnco', 'of_agree', 'orthogonal snGrad')):
    ofArm = rd(W + '/of_' + armTag + '/1/p')
    ctl = rel(ofArm, rd(W + '/' + ctlTag + '/1/p'))
    print('  CONTROL  OpenFOAM p, uncorrected vs %-18s %-12.6e (needs >= %-9.3e) %s'
          % (ctlName, ctl, cmin, 'OK' if ctl >= cmin else 'FAIL'))
    if ctl < cmin: bad = 1
    a = rel(ofArm, rd(W + '/br_' + armTag + '/1/p'))
    print('  ARM p    %s uncorrected                        %-12.6e (bound %-9.3e) %s'
          % (label, a, bound, 'OK' if a <= bound else 'FAIL'))
    if a > bound: bad = 1
    aU = rel(rd(W + '/of_' + armTag + '/1/U'), rd(W + '/br_' + armTag + '/1/U'))
    print('  ARM U    %s uncorrected                        %-12.6e (bound %-9.3e) %s'
          % (label, aU, boundU, 'OK' if aU <= boundU else 'FAIL'))
    if aU > boundU: bad = 1

# ...and the ENERGY equation must be untouched by the snGrad block alone, in OpenFOAM's own answer: it
# assembles no fvc::snGrad, so a non-zero move here would mean the two blocks are not separate after all.
tsng = rel(rd(W + '/of_sngUnco/1/T'), rd(W + '/of_agree/1/T'))
print('  CHECK    OpenFOAM T, snGrad block alone              %-12.6e (must be 0) %s'
      % (tsng, 'OK' if tsng == 0.0 else 'FAIL'))
if tsng != 0.0: bad = 1
sys.exit(bad)
CMPPY
rc=$?
[ "$rc" = 0 ] && echo "PASS" || { echo "FAIL"; exit 1; }
