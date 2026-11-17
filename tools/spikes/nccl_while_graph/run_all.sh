#!/bin/bash
# Build and run the NCCL spike on this node, collect everything into ./spike_results_<host>_<date>/ (tar it and send it back).
#   NCCL_ROOT=/path/with/include+lib  [CUDA_HOME=...] [HOSTCXX=g++-14] [ARCH=70] [MPI_HOME=/usr/lib64/openmpi] ./run_all.sh
# One process drives all GPUs (nccl_spike); with MPI_HOME also the one-process-per-GPU build (nccl_spike_mpi) under mpirun.
# Each run is limited to 150 s; the program has its own 60 s hang watchdog. No root needed.
set -u
cd "$(dirname "$0")"
OUT=spike_results_$(hostname -s)_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
make ${CUDA_HOME:+CUDA_HOME=$CUDA_HOME} ${NCCL_ROOT:+NCCL_ROOT=$NCCL_ROOT} ${HOSTCXX:+HOSTCXX=$HOSTCXX} ${ARCH:+ARCH=$ARCH} ${MPI_HOME:+MPI_HOME=$MPI_HOME} \
    2>&1 | tee "$OUT/build.log" | grep -E "error|Error" && { echo "BUILD FAILED, see $OUT/build.log"; exit 1; }
{ date; hostname; nvidia-smi; nvidia-smi topo -m; nvidia-smi nvlink -s 2>&1 | head -80; nvidia-smi --query-gpu=index,name,clocks.max.sm,clocks.max.mem,persistence_mode --format=csv; } > "$OUT/system.txt" 2>&1
NG=$(nvidia-smi -L | wc -l)
echo "$NG GPU(s)"; nvidia-smi topo -m
rc=0
run() { # name, command...
    local n=$1; shift
    echo "=== $n: $*"
    timeout 150 "$@" > "$OUT/$n.log" 2>&1
    local r=$?
    grep -E "^\[(PASS|FAIL|SKIP)\]|OVERALL|HANG|probe:" "$OUT/$n.log"
    echo "    exit status $r (log: $OUT/$n.log)"
    [ $r -eq 0 ] || rc=1
}
for n in 2 4 8; do
    [ "$n" -le "$NG" ] || continue
    run "single_${n}gpu" ./nccl_spike --ngpu $n
done
# the same with a graph communicator in NCCL's default graph mode: expected to show whether the default is usable in a WHILE body
rc_keep=$rc
run "single_${NG}gpu_default_graphmode" ./nccl_spike --ngpu $NG --graph-mode -1 --tests 3
rc=$rc_keep   # informational: a failure here only documents why graph-mode 0 is used
run "single_${NG}gpu_mixing_env0" env NCCL_GRAPH_MIXING_SUPPORT=0 ./nccl_spike --ngpu $NG --graph-mode -1 --tests 1,2,3
run "single_${NG}gpu_relaxed" ./nccl_spike --ngpu $NG --relaxed --tests 2,3
if [ -x ./nccl_spike_mpi ]; then
    MPIRUN=${MPI_HOME:+$MPI_HOME/bin/}mpirun
    for n in 2 4 8; do
        [ "$n" -le "$NG" ] || continue
        run "mpi_${n}gpu" $MPIRUN -np $n ./nccl_spike_mpi
    done
fi
echo; echo "results in $OUT ; send back: tar czf $OUT.tgz $OUT"
[ $rc -eq 0 ] && echo "ALL RUNS EXIT 0" || echo "SOME RUN FAILED (that is a result too: send the tarball)"
exit $rc
