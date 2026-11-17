# ProcComm NCCL backend + topology-aware placement: what to run on a DGX-1 (np = 2, 4, 8) and what to send back

What this is: the NCCL backend of `ProcComm` (`src/parallel/proc_comm.cu`, class `NcclComm`) and the rank -> GPU placement
module (`src/parallel/proc_topology.{cuh,cu}`). Neither has run on more than one *physical* GPU yet (see "What was run"
at the end). This directory has the instructions and scripts to do that on a DGX-1 (8x V100, sm_70) or a DGX-H100.

## Design in 10 lines
- One process per GPU under `mpirun`/`srun`; `Pstream::init` binds node-local rank i to GPU i (or to `BRAE_GPU_MAP[i]`)
  before any CUDA context exists. Bootstrap: rank 0 `ncclGetUniqueId`, `MPI_Bcast`, `ncclCommInitRankConfig`.
- TWO communicators over the same ranks: `ce` (eager) and `cg` (graph-only, `ncclConfig_t.graphUsageMode = 0`).
  An op runs on `cg` when `CommChannel::Graph` is passed or its stream is being captured, else on `ce`.
  `NCCL_GRAPH_MIXING_SUPPORT=0` is also set (overwrite=0) as the fallback for an NCCL without `graphUsageMode`.
- Warm-up at creation (one allgather per communicator) and at every `makePlan` (largest send/recv group to every
  neighbour on both communicators), so no first-use latency lands in a timed region or in a capture.
- `exchangeCells`: pack kernel -> one `ncclGroupStart; ncclRecv/ncclSend per non-empty neighbour; ncclGroupEnd` on the
  op's stream (one message per neighbour carries all components) -> unpack kernel. `exchangeFaces`: same group, sending
  straight from the caller's face array, then the combine kernel. Group semantics make the neighbour order irrelevant.
- `allReduce` (device, in place) = `ncclAllGather` of the n partials + the fixed rank-order fold kernel (Sum, Min and
  Max alike; identical bits on every rank, independent of NCCL's algorithm). Min/Max would also be exact with
  `ncclAllReduce`; one code path was kept. `bcastDevice` = `ncclBroadcast` in place (exact).
- Host-resident ops (`barrier`, `allReduceHost`, `bcastHost`, `exchangeIntLists`, makePlan's size check) and the abort
  protocol are the MPI ones of the MPI-host backend (the abort record is allgathered at every host collective).
  Device ops only enqueue and cannot throw a REMOTE abort: it surfaces at the next host collective, where every rank
  aborts + re-creates both NCCL communicators (so peers hanging in NCCL kernels against the failed rank are released).
  Graphs captured before an abort are invalid afterwards.
- Every op is CUDA-graph capturable; each rank captures its OWN graph (no joint graph needed).

## Build (CUDA 12.9 for V100; one binary for V100/H100: `-DCMAKE_CUDA_ARCHITECTURES="70;90"`)
```bash
module load cuda/12.9 openmpi/4.1.4              # nvcc 12.9 needs gcc <= 14
export NCCL_ROOT=$HOME/nccl/nvidia/nccl          # dir with include/nccl.h and lib/libnccl.so
#   (pip wheel: pip download nvidia-nccl-cu12 --no-deps; unzip *.whl 'nvidia/*'; ln -s libnccl.so.2 nvidia/nccl/lib/libnccl.so)
#   prefer NCCL >= 2.28 (2.19.3 hung in communicator init on Blackwell; V100 not known to be affected)
cmake -S . -B build70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DBRAE_WITH_NVSHMEM=OFF \
      -DBRAE_WITH_NCCL=ON -DNCCL_ROOT=$NCCL_ROOT -DCMAKE_CUDA_HOST_COMPILER=$(command -v g++) \
      -DBRAE_NCCL_TEST_NP="1;2;4;8"             # + the SCOTCH flags of tools/../V100_BUILD.md as for brae_interFoam
cmake --build build70 -j16 --target test_proc_comm test_proc_comm_graph test_proc_topology proc_topology_tool
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
`BRAE_WITH_NCCL` is OFF by default; then `ProcComm::createNccl()` throws "this build has no NCCL".

## Run (all GPUs of the node must be visible to every rank: no per-task `CUDA_VISIBLE_DEVICES`)
Everything at once, with logs and a summary (backends loopback, mpi, nccl, topology):
```bash
BUILD=$PWD/build70 tools/proc_comm_nccl/run_proc_comm_tests.sh          # NCCL for np = 1 2 4 8 (capped at the GPU count)
tar czf proc_comm_results.tgz proc_comm_results_*                        # send this back
```
The explicit lines (what the script runs):
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID              # same GPU numbering as nvidia-smi (the topology tool relies on it)
mpirun -np 2 build70/test_proc_comm nccl         # then -np 4 and -np 8: halo/face exchange, reductions, aborts
mpirun -np 2 build70/test_proc_comm_graph nccl   # capture + 1000 replays; same step inside a conditional WHILE body
mpirun -np 4 build70/test_proc_comm nccl ;  mpirun -np 4 build70/test_proc_comm_graph nccl
mpirun -np 8 build70/test_proc_comm nccl ;  mpirun -np 8 build70/test_proc_comm_graph nccl
# slurm, one node, one task per GPU:
srun -N1 -n 8 --gpus-per-node=8 --gpu-bind=none --cpu-bind=none build70/test_proc_comm nccl
srun -N1 -n 4 --gpus-per-node=8 --gpu-bind=none --cpu-bind=none build70/test_proc_comm_graph nccl
# the same tests on the other backends, for comparison:
build70/test_proc_comm ; build70/test_proc_comm_graph                       # loopback (virtual ranks on one GPU)
mpirun --oversubscribe -np 8 build70/test_proc_comm mpi                    # MPI-host backend
# with ctest (needs -DBRAE_NCCL_TEST_NP="1;2;4;8" at configure time):  ctest -R 'proc_comm_(graph_)?nccl|proc_topology'
```
Do not use `--oversubscribe` with `nccl`: NCCL refuses two ranks on one GPU (the backend stops with a clear message; use
`BRAE_COMM=mpi`). Each test prints `backend: nccl (np=N, rank 0 on GPU g ...)` and prefixes every PASS line with `nccl`.
Expected end: `PASS nccl ...` lines and `test_proc_comm_graph nccl: PASS`; the graph test also prints the WHILE
iteration-count histogram (it must show >= 3 different counts) and the node count of each graph.

Debug aids if something hangs (the tests have a 180-300 s watchdog that prints `FAIL: watchdog ...`):
`NCCL_DEBUG=WARN` (or `INFO`), `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` to see which transport misbehaves,
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH,INIT` for the communicator topology (which pairs use NVLink P2P and which SHM).

## What to send back
1. `proc_comm_results_*/` (summary.txt, one log per run, system.txt with `nvidia-smi topo -m`, nvlink status, driver).
2. For any FAIL/hang: the log with `NCCL_DEBUG=WARN` and the np at which it happens.
3. The `topology_probe_*.txt` files (what the advisor decides on the real box; compare with the mock table below).

## Rank -> GPU placement (topology step)
Without a map, rank i uses GPU i. On a DGX-1 that is wrong for most decompositions (GPUs 3 and 4 are not NVLinked: a
chain decomposition crosses the sockets at ranks 3-4). `proc_topology_tool` computes the best map from the decomposition:
```bash
# case decomposed with decomposePar (reads processorN/constant/polyMesh/boundary: neighbProcNo + nFaces)
build70/proc_topology_tool --topology probe --case $CASE            # prints link table, cut weights, placement, binding advice
export BRAE_GPU_MAP=$(build70/proc_topology_tool --topology probe --case $CASE --map-only)     # e.g. 0,3,2,1,5,6,7,4
mpirun -np 8 --bind-to none ... brae_interFoam -parallel           # Pstream::init: node-local rank i -> GPU map[i]
# without a case: --chain, --cube (2x2x2 hierarchical), --all-to-all (scotch-like), --matrix file --np N
# without the box: --topology dgx1v | nvswitch | smi=<file with nvidia-smi topo -m>
```
`--topology probe` uses `cudaDeviceCanAccessPeer` + `cudaDeviceGetP2PAttribute(NativeAtomicSupported)` for NVLink/PCIe/none and
refines classes, NVLink lanes (NV1/NV2) and the CPU/socket affinity with `nvidia-smi topo -m` when that runs.
Objective: sum over communicating rank pairs of latency(link) + faces x bytesPerFace / bandwidth(link), exhaustive
over the injective rank -> GPU maps (<= 8! = 40320), deterministic lexicographic tie-break (np = 2 -> GPUs 0,3, an NV2
pair). The link model (NV lane 25 GB/s/direction, PCIe P2P 10, host-staged SYS 6 GB/s + 30 us extra latency) is
[INFERENCE] from nominal figures. The report lists every communicating pair that is not on NVLink and how many such hops
no placement can avoid (a 7-neighbour scotch-like pattern: 12 of the 28 pairs). Decomposition lever: `hierarchical`
with `n (2 2 2)` gives a cube rank graph, which embeds in the cube-mesh with every pair on NVLink.
The tool also prints CPU binding advice (`numactl --cpunodebind/--membind` per rank, from the GPU's socket).

DGX-1V mock table used by the unit test (NV1/NV2 = 1/2 NVLink lanes, SYS = across the sockets; GPUs 0-3 on socket 0):
```
      G0  G1  G2  G3  G4  G5  G6  G7
G0     X  NV1 NV1 NV2 NV2 SYS SYS SYS
G1    NV1  X  NV2 NV1 SYS NV2 SYS SYS
G2    NV1 NV2  X  NV2 SYS SYS NV1 SYS
G3    NV2 NV1 NV2  X  SYS SYS SYS NV1
G4    NV2 SYS SYS SYS  X  NV1 NV1 NV2
G5    SYS NV2 SYS SYS NV1  X  NV2 NV1
G6    SYS SYS NV1 SYS NV1 NV2  X  NV2
G7    SYS SYS SYS NV1 NV2 NV1 NV2  X
```
(from the DGX-1V whitepaper structure: 6 NVLinks per GPU = two NV2 + two NV1 peers; compare with `system.txt`.)

## Environment variables
| variable | effect |
|---|---|
| `BRAE_COMM` (then `BRAE_COMM_BACKEND`) | `ProcComm::create()` backend: loopback, mpi, nccl, auto (default auto: NCCL when built with it, np > 1 and every rank of a node can own a GPU, else mpi) |
| `BRAE_GPU_MAP=g0,g1,..` | node-local rank i -> GPU gi, applied in `Pstream::init` before any context; invalid map aborts the job |
| `BRAE_COMM_REDUCE_CAPACITY` | doubles per rank of one device reduction (default 4096; larger throws `length_error`) |
| `BRAE_NCCL_ALLOW_SHARED_GPU=1` | bring-up/CI only: allow several ranks per GPU (sets `NCCL_MULTI_RANK_GPU_ENABLE=1`, NCCL >= 2.3x); slow |

## What was run (honest status) and what is open
Run on ONE GPU (RTX 5090 laptop, sm_120, CUDA 13.2, NCCL 2.32.3 cu12, OpenMPI): the NCCL backend with np = 1, 2, 3, 4, 8 ranks
sharing that GPU (`BRAE_NCCL_ALLOW_SHARED_GPU=1`; for np = 8 under CUDA MPS, otherwise time-slicing of the spinning NCCL
kernels makes it too slow), all of test_proc_comm and test_proc_comm_graph PASS (halo/face exchange, rank-ordered reductions,
collective aborts incl. communicator re-creation, capture + 1000 replays, WHILE-body graph with varying loop counts).
Compile-checked for sm_70 with CUDA 12.9 + gcc 14 (NCCL on and off). NOT run anywhere: peers on distinct physical GPUs
(NVLink P2P / SHM transport, the 3 host-staged DGX-1 pairs), sm_70 execution, np = 8 across two sockets, the placement
against a real `nvidia-smi topo -m`, NCCL older than 2.32 (no `ncclCommRevoke`: the abort path falls back to `ncclCommAbort`).
