#!/bin/bash
# Whole time steps across processor faces: brae `-parallel` at np = 2 and np = 4 against brae on the undecomposed
# mesh, every linear solve converged (tolerance 1e-12, relTol 0), so what is left between the two is the operators.
#   tests/mgpu_step_parity.sh <brae_interFoam> [scratch dir]
# Case: tools/mgpu_oracle/cases/tankGraded -- sheared and graded (non-orthogonal), `limited corrected` laplacian and
# snGrad, a 60 degree contact angle on the walls -- with the POOL'S SURFACE PUT ON THE CUT: simple2 and simple4 both
# split the 48 cell layers in z at layer 24 (z = 0.07266 under simpleGrading 3), and the pool is filled to there, so
# alpha is 1 on one side of those processor faces and 0 on the other and the wall cells along the cut hold a free
# surface. The two defects this was written for each need that:
#   * the pressure corrector's grad(p_rgh) on a processor face (no explicit stage of the stage gates contains it);
#   * the interfaceProperties constructor pass, whose wall gradient every later pass builds on (the stage gates take
#     that pass on the device themselves; a run took it on the host, which has no neighbour rank). It only shows
#     where alpha differs ACROSS a processor face at the start: MEASURED with the case's own 0.03 pool, which no cut
#     touches, this gate passed at 1.2e-08 with that defect in.
# The undecomposed run is the reference because single-GPU brae is held to OpenFOAM by its own gates; at loose
# solver tolerances OpenFOAM's own np = 1 and np = 2 runs of such a case differ by percents of U, which is why the
# solves are converged here.
# Skips (exit 77) without OpenFOAM v2606 (blockMesh/topoSet/setFields/decomposePar) or mpirun. Needs one GPU.
set -eo pipefail
BRAE=${1:?usage: $0 <brae_interFoam> [scratch]}
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
W=${2:-$(mktemp -d /tmp/mgpu_parity.XXXXXX)}
# U, p_rgh and phi, largest difference over the field's largest value (compare_fields.py's maxrel).
# MEASURED 2026-11-10 on this case at the END_TIME below: see the table in docs/multi-gpu.md.
TOL=${BRAE_PARITY_TOL:-1e-6}
END_TIME=${BRAE_PARITY_END:-0.002}
OF=/usr/lib/openfoam/openfoam2606/etc/bashrc
[ -f "$OF" ] || { echo "SKIP: no OpenFOAM v2606"; exit 77; }
# Fedora keeps Open MPI off PATH; without the launcher every rank would be rank 0 of 1
[ -d /usr/lib64/openmpi/bin ] && export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
command -v mpirun >/dev/null 2>&1 || { echo "SKIP: no mpirun"; exit 77; }
export BRAE_COMM=mpi BRAE_PCG_CHECK_EVERY=1
unset BRAE_MGPU_STAGE_DUMP BRAE_STAGE_PRESSURE BRAE_STAGE_ALPHA
mkdir -p "$W"; cd "$W"; rm -rf mesh np1 np2 np4
cp -r "$HERE/tools/mgpu_oracle/cases/tankGraded" mesh
grep -q "(1.2 1.1 0.03)" mesh/system/setFieldsDict || { echo "FAIL: tankGraded's pool box is not the one this gate moves"; exit 1; }
sed -i 's/(1.2 1.1 0.03)/(1.2 1.1 0.0727)/' mesh/system/setFieldsDict
( set +eu; source "$OF" >/dev/null 2>&1
  cd mesh; rm -f system/decomposeParDict.*
  blockMesh >/dev/null 2>&1; topoSet >/dev/null 2>&1
  cp -r 0.orig 0; mv 0/alpha.water.orig 0/alpha.water; rm -rf 0.orig; setFields >/dev/null 2>&1 )
# ...and nNonOrthogonalCorrectors 0, as the G4 case has it: pcorr's second pass is refused by name under -parallel.
# The explicit non-orthogonal correction, which is what reads grad(p_rgh), is in every pass.
sed -i -E 's/tolerance[[:space:]]+[^;]+;/tolerance 1e-12;/g; s/relTol[[:space:]]+[^;]+;/relTol 0;/g; s/(nNonOrthogonalCorrectors[[:space:]]+)[0-9]+;/\10;/' mesh/system/fvSolution
sed -i "s/^writeControl .*/writeControl runTime;/; s/^writeInterval .*/writeInterval $END_TIME;/; s/^endTime .*/endTime $END_TIME;/" \
    mesh/system/controlDict
D=$HERE/tools/mgpu_oracle/cases/tankGraded/system
cp -r mesh np1
for N in 2 4; do
    cp -r mesh np$N; cp "$D/decomposeParDict.simple$N" np$N/system/decomposeParDict
    ( set +eu; source "$OF" >/dev/null 2>&1; cd np$N; decomposePar -force >/dev/null 2>&1 )
done
"$BRAE" -case np1 -device > log.np1 2>&1 || { tail -n 5 log.np1; echo "FAIL: the undecomposed run"; exit 1; }
RC=0
for N in 2 4; do
    mpirun --bind-to none --oversubscribe -np "$N" "$BRAE" -case np$N -parallel -device > log.np$N 2>&1 \
        || { tail -n 5 log.np$N; echo "FAIL: the np=$N run"; exit 1; }
    echo "== np=1 against np=$N at endTime $END_TIME (bound $TOL)"
    python3 "$HERE/tools/mgpu_oracle/compare_fields.py" np1 np$N --times last --fields U,p_rgh,phi,alpha.water \
        --tol "$TOL" > compare.np$N.md || RC=1
    grep -E "internal" compare.np$N.md
done
[ $RC = 0 ] || { echo "FAIL: a field differs between np=1 and a decomposed run by more than $TOL"; exit 1; }
echo PASS
