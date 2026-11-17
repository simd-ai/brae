#!/bin/bash
# Runs the whole ProcComm test set for the backends loopback, mpi and nccl and writes one log per run plus a summary.
#
#   BUILD=/path/to/build ./run_proc_comm_tests.sh                       # backends: loopback mpi nccl
#   BUILD=... BACKENDS="nccl" NCCL_NPS="2 4 8" ./run_proc_comm_tests.sh # only NCCL, only these process counts
#
# Variables (all optional except BUILD):
#   BUILD        build dir holding test_proc_comm, test_proc_comm_graph, test_proc_topology, proc_topology_tool
#   BACKENDS     subset of "loopback mpi nccl topology" (default: all four)
#   MPI_NPS      process counts of the MPI-host runs                (default "1 2 3 4 8")
#   NCCL_NPS     process counts of the NCCL runs                    (default "1 2 4 8", capped at the number of GPUs)
#   LAUNCH       launcher prefix; the default is "mpirun --oversubscribe -np" for mpi and "mpirun -np" for nccl
#                (NCCL needs one rank per GPU: never oversubscribe). Slurm, one node:
#                   LAUNCH="srun -N1 --gpu-bind=none --cpu-bind=none -n"      (a process count follows it)
#   NGPU         GPUs to assume (default: nvidia-smi -L); NCCL runs with np > NGPU are skipped. On a machine with fewer GPUs
#                than ranks: NGPU=8 BRAE_NCCL_ALLOW_SHARED_GPU=1 (slow, tests the data path only)
#   OUT          result directory (default ./proc_comm_results_<host>_<date>)
# NCCL runs need a build with -DBRAE_WITH_NCCL=ON and all GPUs of the node visible to every rank (do NOT bind each
# task to one GPU with CUDA_VISIBLE_DEVICES: the ranks map to device = node-local rank, or BRAE_GPU_MAP).
set -u
: "${BUILD:?set BUILD to the build directory}"
BACKENDS=${BACKENDS:-"loopback mpi nccl topology"}
MPI_NPS=${MPI_NPS:-"1 2 3 4 8"}
NCCL_NPS=${NCCL_NPS:-"1 2 4 8"}
OUT=${OUT:-proc_comm_results_$(hostname -s)_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')}
[ "$NGPU" -gt 0 ] || NGPU=1
{
  echo "host $(hostname) date $(date -Is) GPUs visible to nvidia-smi: $NGPU"
  nvidia-smi 2>&1 | head -20
  nvidia-smi topo -m 2>&1
  nvidia-smi nvlink --status 2>&1 | head -40
  echo "NCCL libraries:"; ldconfig -p 2>/dev/null | grep -i libnccl
  echo "env: BRAE_COMM=${BRAE_COMM:-} BRAE_GPU_MAP=${BRAE_GPU_MAP:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-}"
} > "$OUT/system.txt" 2>&1

fails=0
summary="$OUT/summary.txt"
: > "$summary"
run() {   # run <label> <command...>
  local label=$1; shift
  local log="$OUT/$label.log"
  if timeout 1500 "$@" > "$log" 2>&1; then
    printf 'PASS  %-34s %s\n' "$label" "$(grep -h '^backend:' "$log" | head -1)" | tee -a "$summary"
  else
    rc=$?
    printf 'FAIL  %-34s rc=%s (see %s)\n' "$label" "$rc" "$log" | tee -a "$summary"
    fails=$((fails+1))
  fi
}

for b in $BACKENDS; do
  case $b in
    loopback)
      run loopback_test_proc_comm "$BUILD/test_proc_comm"
      run loopback_test_proc_comm_graph "$BUILD/test_proc_comm_graph" ;;
    mpi)
      L=${LAUNCH:-mpirun --oversubscribe -np}
      for n in $MPI_NPS; do run mpi_np${n}_test_proc_comm $L $n "$BUILD/test_proc_comm" mpi; done
      run mpi_np2_graph_refuses $L 2 "$BUILD/test_proc_comm_graph" mpi ;;
    nccl)
      L=${LAUNCH:-mpirun -np}
      for n in $NCCL_NPS; do
        if [ "$n" -gt "$NGPU" ]; then echo "SKIP  nccl np=$n (only $NGPU GPUs)" | tee -a "$summary"; continue; fi
        run nccl_np${n}_test_proc_comm $L $n "$BUILD/test_proc_comm" nccl
        run nccl_np${n}_test_proc_comm_graph $L $n "$BUILD/test_proc_comm_graph" nccl
      done ;;
    topology)
      run topology_unit_test "$BUILD/test_proc_topology"
      if [ -x "$BUILD/proc_topology_tool" ]; then
        "$BUILD/proc_topology_tool" --topology probe --np "$NGPU" --chain > "$OUT/topology_probe_chain.txt" 2>&1
        "$BUILD/proc_topology_tool" --topology probe --np "$NGPU" --all-to-all > "$OUT/topology_probe_alltoall.txt" 2>&1
        [ "$NGPU" -eq 8 ] && "$BUILD/proc_topology_tool" --topology probe --np 8 --cube > "$OUT/topology_probe_cube.txt" 2>&1
        echo "      (topology advisor output on this node: $OUT/topology_probe_*.txt)" | tee -a "$summary"
      fi ;;
    *) echo "unknown backend '$b'" ;;
  esac
done
echo "----"; cat "$summary"
echo "failures: $fails   (send back: tar czf proc_comm_results.tgz $OUT)"
exit $((fails > 0))
