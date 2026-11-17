#!/bin/bash
# `brae decompose` and `brae reconstruct` against OpenFOAM v2606's decomposePar and reconstructPar, file by file, and `brae -parallel`
# on a case nobody decomposed.
#   tests/case_decompose_vs_openfoam.sh <brae> <brae_interFoam> [scratch dir]
# Case: tools/mgpu_oracle/cases/tankGraded (sheared, graded, a contact angle on the walls), 55,296 cells.
#   1. the mesh and the start fields of the case: brae decompose (method simple, the decomposeParDict the oracle uses) against
#      decomposePar. Same cell assignment, so the same processor meshes, the same four addressing files, the same fields with the
#      same processor patch values -- identical, to the last digit. (A physical patch's value may differ by 1e-13 of the field: OpenFOAM
#      re-evaluates a fixedFluxPressure patch's value as it reads it; brae copies it.)
#   2. the same for a time brae's own solver wrote (nonuniform boundary values, phi, oriented surface fields).
#   3. reconstructPar and `brae reconstruct` over the processor directories of a real `-parallel` run: identical global fields.
#   4. `brae_interFoam -parallel` on the case as it comes out of blockMesh/setFields, with no processor directories: brae decomposes,
#      every rank runs, rank 0 puts the case back together; the result is the run of step 3.
#   5. scotch and hierarchical (brae's own cut, not OpenFOAM's): decompose then reconstruct gives back the global fields.
#   6. a decomposition abandoned by a killed process is started again; -noReconstruct leaves the case in the processor directories.
# Skips (exit 77) without OpenFOAM v2606 or mpirun. Needs one GPU.
set -eo pipefail
BRAE=${1:?usage: $0 <brae> <brae_interFoam> [scratch dir]}
SOLVER=${2:?usage: $0 <brae> <brae_interFoam> [scratch dir]}
BRAE=$(readlink -f "$BRAE"); SOLVER=$(readlink -f "$SOLVER")
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
W=${3:-$(mktemp -d /tmp/case_decompose.XXXXXX)}
OF=/usr/lib/openfoam/openfoam2606/etc/bashrc
[ -f "$OF" ] || { echo "SKIP: no OpenFOAM v2606"; exit 77; }
[ -d /usr/lib64/openmpi/bin ] && export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
command -v mpirun >/dev/null 2>&1 || { echo "SKIP: no mpirun"; exit 77; }
export BRAE_COMM=mpi BRAE_PCG_CHECK_EVERY=1
unset BRAE_MGPU_STAGE_DUMP BRAE_STAGE_PRESSURE BRAE_STAGE_ALPHA
CMP=$HERE/tools/mgpu_oracle/compare_decomposition.py
SYS=$HERE/tools/mgpu_oracle/cases/tankGraded/system
END_TIME=0.002
fail() { echo "FAIL: $*"; exit 1; }
ofrun() { ( set +eu; source "$OF" >/dev/null 2>&1; "$@" ); }

mkdir -p "$W"; cd "$W"; rm -rf mesh serial
cp -r "$HERE/tools/mgpu_oracle/cases/tankGraded" mesh
sed -i 's/(1.2 1.1 0.03)/(1.2 1.1 0.0727)/' mesh/system/setFieldsDict
sed -i "s/^writePrecision .*/writePrecision 17;/; s/^writeInterval .*/writeInterval $END_TIME;/; s/^endTime .*/endTime $END_TIME;/; s/^writeControl .*/writeControl runTime;/" mesh/system/controlDict
sed -i -E 's/tolerance[[:space:]]+[^;]+;/tolerance 1e-12;/g; s/relTol[[:space:]]+[^;]+;/relTol 0;/g; s/(nNonOrthogonalCorrectors[[:space:]]+)[0-9]+;/\10;/' mesh/system/fvSolution
rm -f mesh/system/decomposeParDict.*
( set +eu; source "$OF" >/dev/null 2>&1; cd mesh
  blockMesh >/dev/null 2>&1; topoSet >/dev/null 2>&1
  cp -r 0.orig 0; mv 0/alpha.water.orig 0/alpha.water; rm -rf 0.orig; setFields >/dev/null 2>&1 )
[ -f mesh/constant/polyMesh/faces ] || fail "the case did not mesh"
ascii() { sed -i 's/^writeFormat .*/writeFormat ascii;/' "$1/system/controlDict"; }

# a global time written by brae's own solver, with the fields decomposePar will have to cut
cp -r mesh serial
"$SOLVER" -case serial -device > log.serial 2>&1 || { tail -n 5 log.serial; fail "the serial run"; }
T=$(ls -d serial/0.* | head -1 | xargs basename)
[ -n "$T" ] || fail "the serial run wrote no time"

# ---- 1, 2: decompose against decomposePar -------------------------------------------------------------------------------------
for SRC in mesh serial; do
    TIME=0; [ "$SRC" = serial ] && TIME=$T
    rm -rf of_$SRC br_$SRC
    cp -r $SRC of_$SRC; cp -r $SRC br_$SRC
    for d in of_$SRC br_$SRC; do cp "$SYS/decomposeParDict.simple4" $d/system/decomposeParDict; done
    ascii of_$SRC
    ofrun bash -c "cd of_$SRC && decomposePar -force -time $TIME" > log.of_$SRC 2>&1 || { tail -n 5 log.of_$SRC; fail "decomposePar of $SRC"; }
    "$BRAE" decompose -case br_$SRC -force -time $TIME > log.br_$SRC 2>&1 || { tail -n 8 log.br_$SRC; fail "brae decompose of $SRC"; }
    echo "== decompose $SRC ($TIME): meshes"
    python3 "$CMP" mesh of_$SRC br_$SRC || fail "the processor meshes of $SRC differ from decomposePar's"
    echo "== decompose $SRC ($TIME): fields"
    python3 "$CMP" fields of_$SRC br_$SRC $TIME --vtol 1e-13 || fail "the processor fields of $SRC differ from decomposePar's"
done

# ---- 3: reconstruct a real -parallel run ---------------------------------------------------------------------------------------
rm -rf par4; cp -r of_mesh par4
mpirun --bind-to none --oversubscribe -np 4 "$SOLVER" -case par4 -parallel -device > log.par4 2>&1 || { tail -n 5 log.par4; fail "the np=4 run"; }
P=$(ls -d par4/processor0/0.* | head -1 | xargs basename)
rm -rf rec_of rec_br; cp -r par4 rec_of; cp -r par4 rec_br
ofrun bash -c "cd rec_of && reconstructPar -time $P" > log.rec_of 2>&1 || { tail -n 5 log.rec_of; fail "reconstructPar"; }
"$BRAE" reconstruct -case rec_br -time $P > log.rec_br 2>&1 || { tail -n 8 log.rec_br; fail "brae reconstruct"; }
echo "== reconstruct $P"
python3 "$CMP" global rec_of rec_br $P || fail "brae reconstruct differs from reconstructPar"

# ---- 4: -parallel on a case nobody decomposed ----------------------------------------------------------------------------------
rm -rf auto4; cp -r mesh auto4; cp "$SYS/decomposeParDict.simple4" auto4/system/decomposeParDict
mpirun --bind-to none --oversubscribe -np 4 "$SOLVER" -case auto4 -parallel -device > log.auto4 2>&1 || { tail -n 8 log.auto4; fail "the automatic run"; }
grep -q "decomposing it into 4" log.auto4 || fail "the run did not say it decomposed the case"
[ -d auto4/$P ] || fail "the run did not reconstruct $P into the case"
echo "== automatic decomposition, run and reconstruction against step 3"
python3 "$CMP" global rec_br auto4 $P || fail "the automatic run differs from the decomposed one"
# a case that has its processor directories is left alone
rm -rf auto4/$P; N0=$(ls auto4 | wc -l)
mpirun --bind-to none --oversubscribe -np 4 "$SOLVER" -case auto4 -parallel -device > log.auto4b 2>&1 || { tail -n 5 log.auto4b; fail "the second run"; }
grep -q "decomposing it into" log.auto4b && fail "a case with processor directories was decomposed again"

# ---- 5: methods brae cuts itself, decomposed and put back -------------------------------------------------------------------------
for M in scotch hierarchical; do
    rm -rf rt_$M; cp -r serial rt_$M
    cp "$SYS/decomposeParDict.${M/hierarchical/hier}4" rt_$M/system/decomposeParDict
    "$BRAE" decompose -case rt_$M -time $T -quiet > log.rt_$M 2>&1 || { tail -n 5 log.rt_$M; fail "decompose $M"; }
    rm -rf rt_$M/$T.src; mv rt_$M/$T rt_$M/$T.src
    "$BRAE" reconstruct -case rt_$M -time $T -quiet >> log.rt_$M 2>&1 || { tail -n 5 log.rt_$M; fail "reconstruct $M"; }
    mkdir -p cmp_$M; rm -rf cmp_$M/$T; mv rt_$M/$T cmp_$M/$T; mv rt_$M/$T.src rt_$M/$T
    ln -sfn "$W/serial/constant" cmp_$M/constant 2>/dev/null || true
    echo "== $M: decompose then reconstruct"
    python3 "$CMP" global serial cmp_$M $T || fail "$M: decompose then reconstruct does not give the fields back"
done

# ---- 6: a decomposition abandoned by a process that is gone is started again; -noReconstruct leaves the results in the processors -----
rm -rf stale2; cp -r mesh stale2; cp "$SYS/decomposeParDict.simple2" stale2/system/decomposeParDict
mkdir -p stale2/processor0/constant/polyMesh                      # what a killed decomposition leaves
echo "$(hostname) 2147483646" > stale2/.brae_decomposing          # ...and its lock, held by a pid that is not there
mpirun --bind-to none --oversubscribe -np 2 "$SOLVER" -case stale2 -parallel -device -noReconstruct > log.stale2 2>&1 || { tail -n 8 log.stale2; fail "the run after an abandoned decomposition"; }
grep -q "abandoned by a process that is gone" log.stale2 || fail "the abandoned decomposition was not noticed"
[ -d stale2/processor1 ] && [ -z "$(ls -d stale2/0.* 2>/dev/null)" ] || fail "-noReconstruct: the case was put back together, or the decomposition is missing"
[ -d stale2/processor0/$P ] || fail "the np=2 run wrote no time"
echo "== abandoned decomposition and -noReconstruct: ok"
echo "PASS"
