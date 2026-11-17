#!/bin/bash
# The operator gate of the multi-GPU explicit stages: brae's per-rank STAGE DUMP (BRAE_MGPU_STAGE_DUMP) against
# OpenFOAM v2606's own dump of the same quantities (tools/dumpInterFoam stageDump.H, BRAE_STAGE_DUMP) at np = 1, 2, 4, 8.
#
#   tools/mgpu_oracle/stage_gate.sh <brae_interFoam> <case> <np> [method=scotch] [tolerance]
#       case    tank | tankGraded | open            np  1 (serial) | 2 | 4 | 8       method  scotch | simple | hier
#   -> $OUT_ROOT/<case>_np<N>[_<method>]/   (OUT_ROOT default /tmp/mgpu_stage)
#
# What it does: builds the mesh and the initial alpha, runs OpenFOAM ONE step (serial, or decomposePar + mpirun -parallel)
# so that a time directory holds a real state (U, phi, alpha, p_rgh), copies that case twice, then
#   * OpenFOAM:  BRAE_STAGE_DUMP=1 dumpInterFoam [-parallel] (startFrom latestTime)   -> s*.dump in the time directory
#   * brae:      BRAE_MGPU_STAGE_DUMP=<copy> brae_interFoam -device [-parallel]  (one process per rank, mpirun
#                --oversubscribe, MPI-host ProcComm; one GPU)                  -> s*.dump per rank
# and compares the two trees with compare_fields.py (mapped onto the global mesh through the processor addressing).
# Both sides read the SAME files, so no linear solve and no limiter is involved: the numbers are operator differences.
# The tolerance (maxrel, default 1e-11) is the exit status; the table is always printed.
# env: ORACLE_ROOT (default /tmp/brae-oracle; setup.sh builds it), OUT_ROOT, KEEP=1 keeps the trees.
set -eo pipefail
BRAE=${1:?usage: stage_gate.sh <brae_interFoam> <case> <np> [method] [tol]}
CASE=${2:?case}
NP=${3:?np}
METHOD=${4:-scotch}
TOL=${5:-1e-11}
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ORACLE_ROOT=${ORACLE_ROOT:-/tmp/brae-oracle}
OUT_ROOT=${OUT_ROOT:-/tmp/mgpu_stage}
BRAE_STAGE_ALPHA_REQUESTED=${BRAE_STAGE_ALPHA:-0}
CASE_LABEL=$(basename "$CASE")
SUF=""; [ "$NP" = 1 ] || { [ "$METHOD" = scotch ] || SUF="_$METHOD"; }
OUT=$OUT_ROOT/${CASE_LABEL}_np${NP}${SUF}
FIELDS=
if [ "${BRAE_STAGE_ALPHA:-0}" != 1 ]; then
    FIELDS=sAlpha.dump,sPhi.dump,sNHatf.dump,sSigmaK.dump,sStf.dump,sRho.dump,sSnGradRho.dump,sPhic.dump,sPhir.dump,sAlphaPhiUn.dump
fi
[ -x "$ORACLE_ROOT/bin/dumpInterFoam" ] || { echo "no dump tool: ORACLE_ROOT=$ORACLE_ROOT $HERE/setup.sh" >&2; exit 1; }
source "$ORACLE_ROOT/of_env.sh"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_btl_vader_single_copy_mechanism=none
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
SOLVER=$ORACLE_ROOT/bin/dumpInterFoam
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"

CASE_SRC=$CASE; [ -d "$CASE" ] || CASE_SRC=$HERE/cases/$CASE
cp -r "$CASE_SRC" state
( cd state
  rm -f system/decomposeParDict.*

  blockMesh > ../log.blockMesh 2>&1; topoSet > ../log.topoSet 2>&1
  sed -i 's/^writeFormat .*/writeFormat ascii;/' system/controlDict
  if [ "$BRAE_STAGE_ALPHA_REQUESTED" = 1 ]; then
    sed -i '/^PIMPLE$/,/^}/s/pRefPoint .*/pRefPoint (0.01 0.005 0.01);/' system/fvSolution
  fi
  cp -r 0.orig 0; mv 0/alpha.water.orig 0/alpha.water; rm -rf 0.orig
  setFields > ../log.setFields 2>&1 )
if [ "$NP" != 1 ]; then
    DICT=$CASE_SRC/system/decomposeParDict.$METHOD$NP
    [ -f "$DICT" ] || { echo "no $DICT" >&2; exit 1; }
    cp "$DICT" state/system/decomposeParDict
    ( cd state && decomposePar -force > ../log.decompose 2>&1 )
fi
# one real step: the state the stages are taken on
export BRAE_STOP_ITER=1 BRAE_WRITE_ITERS=1
unset BRAE_STAGE_ALPHA
if [ "$NP" = 1 ]; then ( cd state && "$SOLVER" > ../log.step1 2>&1 )
else ( cd state && mpirun --bind-to none -np "$NP" "$SOLVER" -parallel > ../log.step1 2>&1 ); fi
unset BRAE_STOP_ITER BRAE_WRITE_ITERS
if [ "$BRAE_STAGE_ALPHA_REQUESTED" = 1 ]; then
    rm -rf corrected
    cp -r state corrected
    sed -i 's/^startFrom .*/startFrom       latestTime;/' corrected/system/controlDict
    if [ "$NP" = 1 ]; then ( cd corrected && BRAE_STAGE_STATE=1 "$SOLVER" > ../log.corrected 2>&1 )
    else ( cd corrected && BRAE_STAGE_STATE=1 mpirun --bind-to none -np "$NP" "$SOLVER" -parallel > ../log.corrected 2>&1 ); fi
    rm -rf of bd
    cp -r corrected of; cp -r corrected bd
fi
if [ "$BRAE_STAGE_ALPHA_REQUESTED" != 1 ]; then
    cp -r state of; cp -r state bd
fi
# both start from the step-1 time directory
sed -i 's/^startFrom .*/startFrom       latestTime;/' of/system/controlDict bd/system/controlDict
if [ -d ofExplicit ]; then sed -i 's/^startFrom .*/startFrom       latestTime;/' ofExplicit/system/controlDict; fi

# OpenFOAM: first the alpha step in alpha-stage mode, then the explicit stages from the unchanged input.
if [ "$BRAE_STAGE_ALPHA_REQUESTED" = 1 ]; then export BRAE_STAGE_ALPHA=1; fi
if [ "$BRAE_STAGE_ALPHA" = 1 ]; then
    rm -rf ofReal; cp -r state ofReal
    sed -i 's/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime 0.00024;/; s/^writeInterval .*/writeInterval 1;/' ofReal/system/controlDict
    printf '\n#inputMode merge\nfunctions { brae_wr { type writeObjects; libs (utilityFunctionObjects); objects (rho alphaPhi0.water); writeControl timeStep; writeInterval 1; } }\n' >> ofReal/system/controlDict
    if [ "$NP" = 1 ]; then ( cd ofReal && interFoam > ../log.of 2>&1 )
    else ( cd ofReal && mpirun --bind-to none -np "$NP" interFoam -parallel > ../log.of 2>&1 ); fi
fi
ORUN=of; [ -d ofExplicit ] && ORUN=ofExplicit
if [ "$BRAE_STAGE_ALPHA_REQUESTED" != 1 ]; then
    export BRAE_STAGE_DUMP=1
    if [ "$NP" = 1 ]; then ( cd "$ORUN" && "$SOLVER" >> ../log.of 2>&1 )
    else ( cd "$ORUN" && mpirun --bind-to none -np "$NP" "$SOLVER" -parallel >> ../log.of 2>&1 ); fi
    unset BRAE_STAGE_DUMP
fi
unset BRAE_STAGE_ALPHA
unset BRAE_STAGE_DUMP

# brae: the same stages (start from the latest time, as OpenFOAM's -latestTime)
export BRAE_MGPU_STAGE_DUMP=$OUT/bd
if [ "$BRAE_STAGE_ALPHA_REQUESTED" = 1 ]; then export BRAE_STAGE_ALPHA=1; fi
if [ "$NP" = 1 ]; then "$BRAE" -case bd -device > log.brae 2>&1
else mpirun --bind-to none -np "$NP" "$BRAE" -case bd -parallel -device > log.brae 2>&1; fi

unset BRAE_MGPU_STAGE_DUMP
grep -a "stage dump" log.brae | head -n 8

T=$(ls -d state/[0-9]* 2>/dev/null | xargs -n1 basename | sort -g | tail -n 1)
[ -n "$T" ] && [ "$T" != 0 ] || T=$(ls -d state/processor0/[0-9]* | xargs -n1 basename | sort -g | tail -n 1)
if [ "$BRAE_STAGE_ALPHA_REQUESTED" = 1 ]; then
    # real OpenFOAM interFoam, one step from the plain step-1 state, writes alpha, alphaPhi0 and rho; alias them
    for D in $( [ "$NP" = 1 ] && echo ofReal || ls -d ofReal/processor* ); do
        TN=$(ls -d $D/[0-9]* | xargs -n1 basename | sort -g | tail -n 1)
        [ "$TN" != "$T" ] || { echo "real interFoam wrote no step in $D" >&2; exit 1; }
        mkdir -p "$D/$T"; DST=$( [ "$NP" = 1 ] && echo of || echo of/$(basename $D) )
        mkdir -p "$DST/$T"
        cp "$D/$TN/alpha.water" "$DST/$T/sAlphaAfter.dump"
        cp "$D/$TN/alphaPhi0.water" "$DST/$T/sAlphaPhi.dump"
        cp "$D/$TN/rho" "$DST/$T/sRhoAfter.dump"
    done
fi
if [ "$BRAE_STAGE_ALPHA_REQUESTED" = 1 ]; then
    for F in sAlphaAfter.dump sAlphaPhi.dump sRhoAfter.dump; do
        find of -path "*/$T/$F" -print -quit 2>/dev/null | grep -q . || {
            echo "missing oracle post-alpha field of/*/$T/$F" >&2; exit 1; }
        find bd -path "*/$T/$F" -print -quit 2>/dev/null | grep -q . || {
            echo "missing brae post-alpha field bd/*/$T/$F" >&2; exit 1; }
    done
fi
echo "state time $T; case $CASE np $NP $METHOD"
set +e
if [ "$BRAE_STAGE_ALPHA_REQUESTED" = 1 ]; then
    python3 "$HERE/compare_fields.py" of bd --times "$T" --fields sAlphaAfter.dump,sAlphaPhi.dump,sRhoAfter.dump --tol "$TOL" > compare_alpha.md
    RC=$?
    cat compare_alpha.md
else
    python3 "$HERE/compare_fields.py" "${ORUN:-of}" bd --times "$T" --fields "$FIELDS" --tol "$TOL" > compare.md
    RC=$?
    cat compare.md
fi
set -e
[ "${KEEP:-0}" = 1 ] || rm -rf state
exit $RC
