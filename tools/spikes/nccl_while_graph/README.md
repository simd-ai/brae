# NCCL inside a CUDA graph / a conditional WHILE node: transport spike for multi-GPU brae

Question answered: can the multi-GPU transport be NCCL (grouped `ncclSend/ncclRecv` halos, `ncclAllGather` + a
fixed-rank-order local sum for bit-reproducible reductions) *inside the body of the conditional WHILE graph* that the
AMG-PCG / symGaussSeidel / BiCGStab solvers use, with a loop count that every rank derives from the same reduced
value? The program (`nccl_spike.cu`) builds the WHILE graph the way `src/matrices/lduMatrix/solvers/PCG/device_amg_pcg.cu`
does (prologue captured with `cudaStreamBeginCaptureToGraph`, `cudaGraphAddNode` conditional WHILE node, body captured
into the node's body graph, a counter kernel ends the body with `cudaGraphSetConditional`).

Tests (each prints `[PASS]`/`[FAIL]`/`[SKIP]`; exit status is non-zero on any FAIL; a 60 s no-progress watchdog turns a
hang into `[FAIL] ... HANG`; each program run takes well under 2 minutes):

| # | what |
|---|------|
| 1 | eager: halo to every peer (<= 7) + 3 deterministic reductions, 100 iterations, checked bit for bit against the host |
| 2 | the same iteration captured into a graph, 100 replays, checked |
| 3 | the iteration in a conditional WHILE body, 1000 replays, loop count decided by the allgathered sum (varies 4+ ways), iteration count / sum / state checked bit for bit. Preceded by "probe" lines: which ingredient makes a WHILE body uninstantiable |
| 4 | us per PCG-like iteration (halo + local kernel + 3 reductions): eager vs graph vs graph-in-WHILE, 1 KB..8 MB per peer |
| 5 | every GPU pair: latency (8 B), bidirectional bandwidth (32 MB), data verified, classified from `nvidia-smi topo -m` (NVLink NVx / same-quad / cross-quad / cross-socket SYS) |

Two communicators are created: `cg` (graph only, created with `ncclConfig_t.graphUsageMode = 0`, i.e. the per-communicator
form of `NCCL_GRAPH_MIXING_SUPPORT=0`; `--graph-mode -1|0|1` overrides) and `ce` (eager). The eager tests use `ce`,
all graph tests `cg`.

## Build and run (what the user does on the DGX-1)

Needs CUDA 12.x (nvcc) with a supported host compiler (gcc <= 14 for 12.9), NCCL >= 2.2x (headers + libnccl.so.2; the
pip wheel `nvidia-nccl-cu12` unpacked anywhere works: `pip download nvidia-nccl-cu12 --no-deps; unzip *.whl 'nvidia/*'`
gives `nvidia/nccl/{include,lib}`; run `ln -s libnccl.so.2 nvidia/nccl/lib/libnccl.so` there). NCCL 2.19.3 hung in
communicator init on the RTX 5090 test machine; that is a Blackwell/old-NCCL issue, V100 is not affected as far as known,
but prefer a recent NCCL. OpenMPI is optional.

```bash
cd tools/spikes/nccl_while_graph
export NCCL_ROOT=$HOME/nccl/nvidia/nccl        # dir holding include/nccl.h and lib/libnccl.so.2
export CUDA_HOME=/usr/local/cuda-12.9 HOSTCXX=g++-14 ARCH=70
# (system NCCL in /usr: NCCL_ROOT must contain include/ and lib/; else symlink them into a dir)

./run_all.sh            # builds, captures nvidia-smi topo -m, runs 2/4/8 GPUs, writes spike_results_<host>_<date>/
tar czf spike_results.tgz spike_results_*   # send this back
```

`run_all.sh` runs the single-process variant (one process, N GPUs, `./nccl_spike --ngpu N`) for N = 2, 4, 8 (as many GPUs
as the node has), then three diagnostic variants on all GPUs (NCCL's default graph mode, `NCCL_GRAPH_MIXING_SUPPORT=0`,
`--relaxed` capture mode) and, with `MPI_HOME` set, the one-process-per-GPU variant.

Explicit lines, if you prefer them over the script:

```bash
make ARCH=70 HOSTCXX=g++-14 NCCL_ROOT=$NCCL_ROOT CUDA_HOME=$CUDA_HOME MPI_HOME=/usr/lib64/openmpi   # builds both binaries
./nccl_spike --ngpu 2                      # one process, 2 GPUs (ncclCommInitRankConfig in a group)
./nccl_spike --ngpu 4
./nccl_spike --ngpu 8
mpirun -np 2 ./nccl_spike_mpi              # one rank per GPU, ranks must be on ONE node (device = local rank)
mpirun -np 4 ./nccl_spike_mpi
mpirun -np 8 ./nccl_spike_mpi
# slurm, one node:
srun -N1 -n 8 --gpus-per-task=1 --gpu-bind=none ./nccl_spike_mpi   # (use --gres=gpu:8 instead where gpus-per-task is unsupported)
srun -N1 -n 4 ./nccl_spike_mpi ; srun -N1 -n 2 ./nccl_spike_mpi
```
The MPI variant maps rank -> device `localRank`, so all 8 GPUs must be visible to every rank (do not bind each task to
one GPU with CUDA_VISIBLE_DEVICES; use `--gpu-bind=none`/`--gres=gpu:8`). Options: `--tests 3,5`, `--relaxed` or `--global`
(capture mode; default thread-local like the repo), `--graph-mode`.

## What to send back
The `spike_results_*` directory (logs per run, `system.txt` with `nvidia-smi topo -m`, nvlink status, clocks). Key lines:
the `[PASS]/[FAIL]` verdicts, the `probe:` lines of test 3, the test 4 table and the test 5 per-class summary.

## What the local run (1 GPU) established, and what it cannot
Run on an RTX 5090 laptop (sm_120), NCCL 2.32.3, 1 rank (halo = self send/recv, reductions over 1 rank), CUDA 13.2 and
CUDA 12.9 runtimes: tests 1-4 PASS; NCCL ops captured inside a conditional WHILE body replay 1000x bit-exactly with a
data-dependent loop count. Finding: a graph communicator in NCCL's default graph mode (config unset, or `graphUsageMode=2`)
produces a WHILE body that `cudaGraphInstantiate` rejects (`invalid argument`) on CUDA 13.2; with `graphUsageMode=0` (or 1)
or `NCCL_GRAPH_MIXING_SUPPORT=0` it instantiates and runs. NOT exercised locally: any real peer-to-peer transport
(NVLink/PCIe/SHM kernels), multi-rank bit-reproducibility, the MPI variant at run time (compile-checked only), test 5,
sm_70 execution (compile-checked with nvcc 12.9 only), NCCL 2.19-2.2x behaviour on V100.
