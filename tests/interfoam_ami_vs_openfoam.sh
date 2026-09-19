#!/usr/bin/env bash
# brae's interFoam across a ROTATING cyclicAMI against REAL OpenFOAM's, on RAS/mixerVesselAMI, field by
# field and solve by solve -- and OpenFOAM's AMI weights themselves, face by face.
#
# THE CASE: a stirred vessel. snappyHexMesh cuts a cylindrical cellZone `rotating` round the stirrer,
# createBaffles and mergeOrSplitBaffles -split turn its surface into the cyclicAMI pair AMI1/AMI2, and
# solidBodyMotionSolver turns the zone's points at omega -5 about z. Every step moves the points,
# recomputes the AMI on them and couples the two sides through its weights. kEpsilon, explicit MULES in
# two sub-cycles, a momentum predictor, two outer correctors, correctPhi, `grad(U) cellLimited Gauss
# linear 1`, `limited corrected 0.33`, and a rotatingWallVelocity shaft. Meshed as Allrun.pre does, the
# background block (50 50 100) coarsened to (22 22 44): 82,510 cells, 8,872 faces on each side of the pair.
#
# STAGED, in BOTH codes, and not claimed:
#   p_rgh (and pcorr, which takes $p_rgh) GAMG -> PCG with DIC. GAMG across an AMI agglomerates the
#   interface (cyclicAMIGAMGInterface), which is not ported; brae refuses GAMG on a coupled patch.
#   THE SOLVES ARE PINNED: p_rgh and pcorr to 1e-13 with relTol 0, U, k and epsilon to 1e-13. At the case's
#   own tolerances (p_rgh 1e-6, relTol 0.02) the comparison measures where two Krylov solvers stopped:
#   MEASURED, 100 steps, U 3.9e-07 apart and 14 of 200 p_rgh counts different; pinned, 3.5e-12 and none.
#   Fixed steps of 2e-4 (the tutorial adjusts to maxCo 1.5).
#
# MEASURED, 100 steps (0.1 rad, the rotor sliding 3.5 faces of the pair past the stator's):
#   alpha 4.4e-13, p_rgh 5.4e-14, U 3.5e-12, k 9.5e-14, epsilon 1.2e-13, nut 2.1e-12; the flux across the
#   pair 1.3e-11 of its largest; the moved points bitwise; all 200 p_rgh, 600 U, 100 k, 100 epsilon and
#   101 pcorr counts OpenFOAM's.
# ...and the AMI (test_face_area_weight_ami, 3 steps of 2e-4 with OpenFOAM's AMI dumped each step): every
#   partner set of all 8,872 faces a side OpenFOAM's, weights and sums 2.5e-14.
# CONTROL, OpenFOAM against itself: the rotor held still (omega 0), U 1.1e-01.
#
# WHAT THE GATE FOUND, each localised against OpenFOAM's own instrumented interFoam:
#   1. OpenFOAM's AMI is an ADVANCING FRONT, not a search: one step in, source face 2354 has two partners
#      covering 96.4% of it where every overlapping face covers all of it. An all-pairs search gave the
#      patch weights 1.7e-04 and deltaCoeffs 1.0e-02 off, pcorr 632 iterations for OpenFOAM's 781, U 5.7e-05.
#   2. alphaEqn.H evaluates alpha's boundary nowhere before its fluxes: the explicit path reads the patch
#      values the last MULES solve left, or at step one the file's `value uniform 0` at the outlet --
#      under water, so OpenFOAM's 62 outlet cells rise to 1.104 in the first sub-cycle. brae evaluated at
#      the top of the alpha step and kept them at 1: alpha 9.4e-02 after one step.
#   3. `nut = Cmu*sqr(k)/epsilon` does not reach a fixedValue patch (its operator= is empty): the gasInlet
#      keeps nut 0 where brae wrote 1.29e-03. U 2.7e-03 after one step.
#
# FAIL-PROOFS, each decision broken once in a scratch copy, the gate red every time (U after 100 steps):
#   every overlapping pair (a search) in place of the walk   6.0e-05  (weights: 47 and 49 partner sets,
#                                                                      sums 3.8e-02)
#   the AMI weights normalised by the face area              non-finite by step 6 (weights 1.0e-01)
#   the AMI not recomputed after the move                    refused by name (a crash before the check)
#   delta without the neighbour's interpolated half          9.7e-03
#   no non-orthogonal correction on the coupled faces        4.0e-02  (alpha 1.0e-01)
#   ddtCorr live on the cyclicAMI                            8.7e-03
#   MULES' limiter synced across the pair                    1.4e-04  (alpha 1.5e-02)
#   cellLimited's coupled range from the stored patch value  1.3e-03
#   grad(U) unlimited                                        2.7e-01
#   alpha's boundary evaluated at the top of the alpha step  1.9e-06
#   nut assigned onto the fixedValue gasInlet                1.1e-02
#   rotatingWallVelocity's omega sign flipped                1.2e-03
# NOT DISCRIMINATED: the segregated solve's coupled source added and taken back (round-off: U 4.1e-12,
# one p_rgh count one apart at an edge stop); kEpsilon's moving-mesh terms, V0 in the ddt and the
# absolute flux in divU (a rigid rotation changes no volume: U 3.8e-12).
# NOT CLAIMED: GAMG across the AMI (staged, above), the tutorial's 1.1M-cell mesh, adjustable time
# steps, a transformed, periodic or low-weight-corrected AMI and every AMI keyword brae does not read
# (refused), requireMatch false (refused), a moving mesh under kOmegaSST or LES (refused), a written
# `value` on rotatingWallVelocity (refused), and the device loop (refused).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_ami_vs_openfoam"
WBIN="${BUILD:-$ROOT/build}/test_face_area_weight_ami"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/mixerVesselAMI"
STEPS=${STEPS:-100}
DT=${DT:-2e-4}
WEIGHT_STEPS=3
# the tutorial's (50 50 100) background block meshes 1.1M cells -- a bench size, not a validation one
BLOCK=${BLOCK:-"22 22 44"}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -x "$WBIN" ]     || { echo "SKIP: $WBIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: mixerVesselAMI tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh surfaceFeatureExtract snappyHexMesh createBaffles mergeOrSplitBaffles setFields interFoam; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

timeAfter()
{
    python3 -c "
t = 0.0
for i in range($1):
    t += float('$DT')
print('%.10g' % t)"
}
END=$(timeAfter "$STEPS")

# THE MESH, once, as Allrun.pre and Allrun make it (serial)
M="$W/mesh"
rm -rf "$M"
cp -r "$SRC" "$M" || exit 1
rm -rf "$M"/[1-9]* "$M"/0 "$M"/processor* "$M"/log.*
grep -q "(50 50 100)" "$M/system/blockMeshDict" || { echo "FAIL: the tutorial's block is no longer (50 50 100)"; exit 1; }
sed -i "s/(50 50 100)/($BLOCK)/" "$M/system/blockMeshDict"
cp -rf "$TUT/resources/geometry/mixerVesselAMI" "$M/constant/triSurface" || exit 1
( cd "$M" && blockMesh > log.blockMesh 2>&1 && surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 \
      && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 && createBaffles -overwrite > log.createBaffles 2>&1 \
      && mergeOrSplitBaffles -split -overwrite > log.mergeOrSplitBaffles 2>&1 && cp -r 0.orig 0 \
      && setFields > log.setFields 2>&1 ) \
    || { echo "FAIL: meshing"; exit 1; }
echo "meshed: $(grep "cells:" "$M/log.snappyHexMesh" | tail -1)"

# stage <profile> <steps> <writeEvery>: copy the mesh, fix the step, pin the solves, apply the profile
stage()
{
    local profile="$1"
    local n="$2"
    local every="$3"
    local C="$W/$profile"
    rm -rf "$C"
    mkdir -p "$C"
    cp -r "$M/0" "$M/constant" "$M/system" "$C/" || return 1
    PROFILE="$profile" N="$n" EVERY="$every" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['N'])
dt = os.environ['DT']
p = os.environ['PROFILE']
t = 0.0
for i in range(n):
    t += float(dt)
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % t),
                 ('writeControl', 'timeStep'), ('writeInterval', os.environ['EVERY']), ('writeFormat', 'ascii'),
                 ('writePrecision', '18')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
if p == 'weights':
    # OpenFOAM's own AMI, source and target, after every step: what cyclicAMIPolyPatch::AMI() holds
    s += r'''
functions
{
    amiDump
    {
        type            coded;
        libs            (utilityFunctionObjects);
        name            amiDump;
        codeInclude
        #{
            #include "cyclicAMIPolyPatch.H"
            #include "OFstream.H"
            #include "IOmanip.H"
        #};
        codeExecute
        #{
            const fvMesh& m = mesh();
            const label idx = m.time().timeIndex();
            const cyclicAMIPolyPatch& ap = refCast<const cyclicAMIPolyPatch>(m.boundaryMesh()[m.boundaryMesh().findPatchID("AMI1")]);
            const auto& ami = ap.AMI();
            OFstream os(m.time().path()/("ami_" + Foam::name(idx) + ".txt"));
            os.precision(18);
            os << "src " << ami.srcAddress().size() << nl;
            forAll(ami.srcAddress(), i)
            {
                os << i << " " << ami.srcWeightsSum()[i] << " " << ami.srcAddress()[i].size();
                forAll(ami.srcAddress()[i], j) { os << " " << ami.srcAddress()[i][j] << " " << ami.srcWeights()[i][j]; }
                os << nl;
            }
            os << "tgt " << ami.tgtAddress().size() << nl;
            forAll(ami.tgtAddress(), i)
            {
                os << i << " " << ami.tgtWeightsSum()[i] << " " << ami.tgtAddress()[i].size();
                forAll(ami.tgtAddress()[i], j) { os << " " << ami.tgtAddress()[i][j] << " " << ami.tgtWeights()[i][j]; }
                os << nl;
            }
            return true;
        #};
    }
}
'''
open(c, 'w').write(s)
v = os.path.join(d, 'system/fvSolution')
s = open(v).read()
# THE STAGING: PCG with DIC for p_rgh (pcorr takes $p_rgh), every solve pinned -- see the header
s, k = re.subn(r'(\n    p_rgh\s*\{\s*solver\s+)GAMG;(\s*tolerance[^;]*;\s*relTol[^;]*;\s*)smoother\s+GaussSeidel;',
               r'\1PCG;\2preconditioner  DIC;', s)
assert k == 1, 'p_rgh GAMG'
for pat, val in [(r'(\n    p_rgh\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-13'), (r'(\n    p_rgh\s*\{[^}]*?relTol\s+)[^;]+;', '0'),
                 (r'("pcorr\.\*"\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-13'),
                 (r'("\(U\|T\|k\|epsilon\)\.\*"\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-13')]:
    s, k = re.subn(pat, r'\g<1>' + val + ';', s)
    assert k == 1, pat
open(v, 'w').write(s)
if p == 'still':
    mdict = os.path.join(d, 'constant/dynamicMeshDict')
    s = open(mdict).read()
    s, k = re.subn(r'omega\s+-5;', 'omega           0;', s)
    assert k == 1, 'omega'
    open(mdict, 'w').write(s)
PYEOF
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    echo "OpenFOAM ran $n steps of deltaT $DT   [$profile]"
}

rc=0
stage weights "$WEIGHT_STEPS" 1 || rc=1
stage still "$STEPS" "$STEPS" || rc=1
stage ami "$STEPS" "$STEPS" || rc=1
[ $rc = 0 ] || { echo "interfoam_ami_vs_openfoam: staging failed"; exit 1; }

# OpenFOAM's AMI, step by step, against brae's motion and brae's AMI
args=()
for i in $(seq 1 "$WEIGHT_STEPS"); do
    args+=("$W/weights/$(timeAfter "$i")" "$W/weights/ami_$i.txt")
done
"$WBIN" "$W/weights" "${args[@]}" || rc=1

"$BIN" "$W/ami" "$W/ami/0" "$W/ami/$END" "$STEPS" "$W/ami/log.interFoam" "$W/still/$END" || rc=1

echo "interfoam_ami_vs_openfoam: rc $rc"
exit $rc
