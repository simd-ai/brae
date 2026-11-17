#!/bin/bash
# Regenerates one oracle run from scratch: mesh, serial reference, decomposition, OpenFOAM -parallel run, per-rank
# operator dumps, reconstruction and the serial-vs-parallel comparison.
#
#   tools/mgpu_oracle/run_oracle.sh <case> <np> [method=scotch] [steps=20]
#       case    tank | tankGraded | open        (tools/mgpu_oracle/cases/)
#       np      2 | 4 | 8
#       method  scotch | simple | hier          (system/decomposeParDict.<method><np>)
#   -> /tmp/mgpu_oracle/<case>_np<N>[_<method>]/
#        mesh/            blockMesh + topoSet + setFields, the common start (no time dirs but 0)
#        serial/          interFoam (dumpInterFoam) serial to step <steps>:  times t1, t<steps>, dumps at t<steps>
#        serial_d1/       the same stopped at step 1, dumps at t1
#        par/, par_d1/    the same decomposed (processorN/...) run with mpirun -np N ... -parallel
#        par_rec/         reconstructPar of par/ (global fields), for the self-test
#        report.md        decomposition table + serial-vs-parallel error floors (compare_fields.py)
#        log.*            every log
#   env: TIGHT=1 (all linear solvers to 1e-12 / relTol 0, dir suffix _tight), ORACLE_ROOT (default /tmp/brae-oracle; setup.sh builds it), OUT_ROOT (/tmp/mgpu_oracle),
#        KEEP_DUMPS=1 keeps the per-step .dump fields (default: kept; set 0 to drop them after comparing)
set -eo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CASE=${1:?usage: run_oracle.sh <case> <np> [method] [steps]}
NP=${2:?np}
METHOD=${3:-scotch}
STEPS=${4:-20}
ORACLE_ROOT=${ORACLE_ROOT:-/tmp/brae-oracle}
OUT_ROOT=${OUT_ROOT:-/tmp/mgpu_oracle}
SUF=""; [ "$METHOD" = scotch ] || SUF="_$METHOD"
[ "${TIGHT:-0}" = 1 ] && SUF="${SUF}_tight"
OUT=$OUT_ROOT/${CASE}_np${NP}${SUF}
[ -d "$HERE/cases/$CASE" ] || { echo "no case $CASE" >&2; exit 1; }
DICT=$HERE/cases/$CASE/system/decomposeParDict.$METHOD$NP
[ -f "$DICT" ] || { echo "no $DICT" >&2; exit 1; }
[ -x "$ORACLE_ROOT/bin/dumpInterFoam" ] || ORACLE_ROOT=$ORACLE_ROOT "$HERE/setup.sh"
source "$ORACLE_ROOT/of_env.sh"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_btl_vader_single_copy_mechanism=none
SOLVER=$ORACLE_ROOT/bin/dumpInterFoam

rm -rf "$OUT"; mkdir -p "$OUT"
cd "$OUT"
# ---- mesh + initial fields (shared by every run) ----
cp -r "$HERE/cases/$CASE" mesh
( cd mesh
  rm -f system/decomposeParDict.*                       # a run copies the one it needs
  blockMesh > ../log.blockMesh 2>&1
  topoSet > ../log.topoSet 2>&1
  cp -r 0.orig 0; mv 0/alpha.water.orig 0/alpha.water; rm -rf 0.orig
  setFields > ../log.setFields 2>&1
  checkMesh > ../log.checkMesh 2>&1 || true
  if [ "${TIGHT:-0}" = 1 ]; then      # solve every linear system to roundoff: isolates the operators from solver partial convergence
      sed -i -E 's/tolerance +1e-7; *relTol +0\.05;/tolerance 1e-12; relTol 0;/; s/tolerance +1e-8;/tolerance 1e-12;/' system/fvSolution
  fi )
tail -n 3 log.checkMesh | head -n 1

run_pair() {   # <dir> <parallel 0|1>
    local dir=$1 par=$2
    cp -r mesh "$dir"; cp "$DICT" "$dir/system/decomposeParDict"
    if [ "$par" = 1 ]; then
        ( cd "$dir" && decomposePar -force -cellDist > ../log.decompose.$dir 2>&1 )
    fi
}

run_solver() { # <dir> <parallel> <dumpstep>
    local dir=$1 par=$2 n=$3
    export BRAE_STOP_ITER=$n BRAE_DUMP_ITER=$n BRAE_ORACLE_ITERS=$n BRAE_WRITE_ITERS=1,$n
    if [ "$par" = 1 ]; then
        ( cd "$dir" && mpirun --bind-to none -np "$NP" "$SOLVER" -parallel > ../log.$dir 2>&1 )
    else
        ( cd "$dir" && "$SOLVER" > ../log.$dir 2>&1 )
    fi
    unset BRAE_STOP_ITER BRAE_DUMP_ITER BRAE_ORACLE_ITERS BRAE_WRITE_ITERS
}

for spec in "serial:0:$STEPS" "serial_d1:0:1" "par:1:$STEPS" "par_d1:1:1"; do
    IFS=: read d p n <<< "$spec"
    run_pair "$d" "$p"
    s=$(date +%s.%N)
    run_solver "$d" "$p" "$n"
    printf '%-9s steps=%-3s %.1fs  %s\n' "$d" "$n" "$(echo "$(date +%s.%N) - $s" | bc)" \
        "$(grep -a '^Time = ' log.$d | tail -n 1)"
done

# ---- reconstruct the np=N result (global fields) ----
mkdir par_rec; cp -r mesh/constant mesh/system par_rec/
( cd par && reconstructPar -newTimes > ../log.reconstruct 2>&1 ) || echo "reconstructPar failed, see log.reconstruct"
for t in $(ls par | grep -E '^[0-9][0-9.e+-]*$' | grep -v '^0$'); do mv "par/$t" par_rec/; done

# ---- report ----
python3 "$HERE/decomp_stats.py" par > report.md
{
  echo; echo "## serial vs OpenFOAM -parallel (np=$NP, $METHOD): step 1 and step $STEPS, final states";
  python3 "$HERE/compare_fields.py" serial par --fields alpha.water,p_rgh,U,p,phi --times all || true
} >> report.md
cat report.md
