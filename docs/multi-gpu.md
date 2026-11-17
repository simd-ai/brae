# Multi-GPU `interFoam`

One case across up to 8 GPUs of one node, one process per GPU, with OpenFOAM's own `-parallel` semantics: the mesh is
split into `processorN/` directories (by `brae decompose`, or by OpenFOAM's `decomposePar`; the layout is the same),
processor patches are coupled interfaces, each rank builds its own preconditioner, and reductions are global. Results can be checked against OpenFOAM `-parallel` on the
same decomposition. Single-GPU runs are unchanged (byte for byte, checked by the test suite).

## Quick start

```bash
cd my-case
mpirun --bind-to none -np 4 brae_interFoam -case . -parallel -device
```

That is all: with no `processor<N>/` directories, brae decomposes the case into as many parts as there are processes
(`system/decomposeParDict` supplies the method, scotch without one), every rank runs its own part, and rank 0 puts the
case back together at the end (`-noReconstruct` keeps the results in the processor directories only). The processor
directories stay, so the next run reuses them, and a case that already has them is left alone.

Doing the two steps yourself:

```bash
brae decompose -case . -n 4             # as decomposePar: processor0..3/ with the mesh, addressing and the start time's fields
mpirun --bind-to none -np 4 brae_interFoam -case . -parallel -device
brae reconstruct -case .                # as reconstructPar: the global time directories from processor0..3
```

OpenFOAM's `decomposePar` (v2606) and `reconstructPar` work on the same directories, in either direction.

## Decomposing the case

`brae decompose` writes what OpenFOAM's `decomposePar` writes for the cases `-parallel` runs, without OpenFOAM:
`processor<N>/constant/polyMesh` (points, faces, owner, neighbour, boundary, `cellZones`, and the four `*ProcAddressing`
files) and the volume and surface fields of the start time (`-time` and `-allTimes` choose others). The tests compare it with
`decomposePar` file by file: same processor meshes and addressing, same field values, processor patch values included.

| `method` | what it does |
|---|---|
| `scotch` | the cell graph cut by SCOTCH (the default): a balanced cut, but not the partition OpenFOAM's `scotch` makes (measured: 15,596 + 15,596 cells against OpenFOAM's 15,473 + 15,719 on a 31k-cell case) |
| `simple` | OpenFOAM's `simple`: each direction cut into `n` equal parts by the sorted cell centres (`n`, `delta` in `simpleCoeffs`); the same cell assignment as OpenFOAM's |
| `hierarchical` | the cells cut along `order`'s first direction, each part along the next, and so on; brae's own cut, which can differ from OpenFOAM's by cells at a cut |
| `manual` | the cell-to-processor list of `manualCoeffs/dataFile` |

Not decomposed: cyclic, cyclicAMI/ACMI and processor patches in the mesh, face and point zones that have members (empty ones are carried over), point fields,
dynamic-refinement files (`cellLevel`, `pointLevel`, ...), and Lagrangian data. A field is cut as a whole: its internal
field, and each patch's lists whose length is the patch's face count; every other entry is carried over unchanged. A
processor patch gets the value from the other side of the face, and a reversed face of an `oriented` surface field its
negative, as `decomposePar` does.

`brae reconstruct` is the other direction: internal fields by cell addressing, patch lists by face addressing, the
faces between processors from the owner side. Both write ASCII at full precision (`decomposePar` and `reconstructPar`
read and write both formats; so does brae).

`-parallel` reads the dictionaries from the case and the mesh, fields and output from each rank's `processor<rank>/`.
`startFrom latestTime` resolves against `processor0`, and every rank reads its own `processor<rank>/<latestTime>`
(including `uniform/time` and `uniform/cumulativeContErr`), so a decomposed run can be restarted.

For the common function objects, `volFieldValue` and `surfaceFieldValue` are globally reduced, probes use the lowest
owning rank, `fieldMinMax` reports the winning rank, and `pulseStop` reads the globally reduced value so all ranks stop
together. Only rank 0 writes `postProcessing/`; every rank writes its own `uniform/`.

## Launching: `brae` and `brae_interFoam`

`brae_interFoam -case . -parallel -device` (under `mpirun`) is the multi-GPU `interFoam`. The `brae` launcher also forwards a
decomposed `interFoam` case to it. If a separate multi-GPU program named `brae-mgpu` is installed next to `brae`, the launcher
hands every `-parallel` run to that program instead, so a `-parallel` run through `brae` uses whichever layer the install has.
Running `brae_interFoam` directly always uses the one in this repository.

## Restrictions

Refused by name under `-parallel`: `processorCyclic`, `cyclicAMI`/`cyclicACMI`, dynamic refinement, topology change and
moving meshes, MRF zones, `fvOptions`, any turbulence model but laminar, and `pcorr` with `nNonOrthogonalCorrectors > 0`.
The `surfaces` function object is noticed but not run; `writeFields true` on a field-value object and
`exprField autowrite true` are refused.

## Options

Command line (`brae_interFoam`; the `brae` launcher forwards them):

| option | effect |
|---|---|
| `-parallel` | read the case as decomposed into `processorN/` (decomposed first when it has none) |
| `-noReconstruct` | with `-parallel`: do not put the case back together at the end, when brae decomposed it |
| `-rank N -nprocs M` | the rank and the process count, when not taken from the launcher. Otherwise read from `OMPI_COMM_WORLD_RANK/SIZE`, `PMI_RANK/SIZE` or `SLURM_PROCID/NTASKS`; the explicit flags win |

Environment:

| variable | effect |
|---|---|
| `BRAE_COMM` (or `BRAE_COMM_BACKEND`) | `loopback`, `mpi`, `nccl` or `auto` (default: NCCL when built, more than one rank, and the ranks of a node fit its GPUs; else MPI-host). `BRAE_COMM_NP` sets the loopback rank count; the loopback backend is for test fixtures, not for `mpirun` |
| `BRAE_GPU_MAP="g0,g1,..."` | node-local rank `i` uses GPU `gi` (see `tools/proc_comm_nccl/` for a placement tool); applied before any CUDA context exists; an invalid map aborts the job |
| `BRAE_NCCL_ALLOW_SHARED_GPU=1` | let several NCCL ranks share one GPU (testing on a single-GPU machine) |
| `BRAE_PROC_GRAPH=0` | run the distributed AMG-PCG / symmetric Gauss-Seidel loops host-driven on the NCCL backend too |
| `BRAE_COND_GRAPH=0` | force the host-driven solver loops (`=1` skips the driver check described below) |

Build:

| CMake option | effect |
|---|---|
| `BRAE_WITH_NCCL` (default OFF) | build the NCCL backend; `-DNCCL_ROOT=<dir with include/nccl.h and lib/libnccl.so>` (or `NCCL_HOME`, `CMAKE_PREFIX_PATH`). NCCL 2.28 or newer is recommended |
| `BRAE_NCCL_TEST_NP` | process counts of the NCCL tests, for example `"1;2;4;8"` (needs that many GPUs) |

## What is in

| piece | what it does | where |
|---|---|---|
| reader and roots | `PatchInfo`/`FvPatch` carry `myProcNo`, `neighbProcNo`, `referPatch`; the case's dictionaries and the rank's `processor<rank>/` mesh, fields, zones and output are separate roots | `src/applications/solvers/common/case_roots.cuh` |
| processor geometry | weights, delta, deltaCoeffs, nonOrthDeltaCoeffs and correction vectors from OpenFOAM's `processorFvPatch` formulas with one setup exchange; bit-identical to OpenFOAM v2606 on the test cases | `src/parallel/processor_coupling.*`, [design note](multigpu-processor-interface.md) |
| device interface | processor faces are the last boundary faces of the device mesh; neighbour reads go through halo slots after the local cells (`nCellsExt = nC + nHalo`) | `src/cuda/device_processor.cuh` |
| `ProcComm` | halo exchange of cell and face fields, deterministic rank-order reductions (host and device-resident), broadcast, collective abort; **loopback**, **MPI-host** and **NCCL** backends; capturable in CUDA graphs | `src/parallel/proc_comm.*` |
| topology | rank-to-GPU placement for NVLink cube-mesh nodes (exhaustive and deterministic), `nvidia-smi topo -m` parser | `src/parallel/proc_topology.*`, `tools/proc_comm_nccl/` |
| distributed solvers | global reductions, processor interface in the matrix product, AMG-PCG with a global stopping decision (all ranks take the same iteration count), symmetric Gauss-Seidel with a halo frozen per sweep, BiCGStab, per-rank AMG hierarchy cache | `src/matrices/lduMatrix/` |
| writer and function objects | per-rank `processor<rank>/<time>` output, byte-identical to OpenFOAM's `decomposePar` output (`reconstructPar` works); the reference cell on one rank; global `deltaN` and end-of-run statistics; function objects reduced across ranks | `inter_writer_cpp.*`, `inter_function_objects.cuh`, `common/proc_globals.cuh` |

Distributed AMG-PCG on a decomposed mesh needs more iterations than one GPU, since each rank's V-cycle is block-local as
in OpenFOAM (OpenFOAM's own `p_rgh` matrix on a 31k-cell case: 58/73/83 iterations at np 2/4/8 against 45 monolithic).

## CUDA graphs and drivers

On the NCCL backend the np > 1 AMG-PCG and symmetric Gauss-Seidel loops run as ONE conditional-`WHILE` graph per rank
(the halo exchange and the deterministic allgather-based reductions inside the body on a graph-only communicator, the
stop decided on the device from the globally reduced residual, so every rank runs the same count). They are
bit-identical to the host-driven loops, which the MPI-host backend runs and `BRAE_PROC_GRAPH=0` selects on NCCL.
`BRAE_PCG_DEVICE=0` and `BRAE_GS_HOST_LOOP=1` also force host loops.

Conditional graph nodes need CUDA 12.4 and a driver r550 or newer. `braeCondGraphsAvailable()` checks the driver's CUDA
version at run time and takes the host-driven loops where it is older. On a data-center GPU whose driver cannot be
upgraded, NVIDIA's `cuda-compat` package (a newer user-mode `libcuda.so` first on `LD_LIBRARY_PATH`, no root) enables
the graph loops; it is NVIDIA's proprietary software, so fetch it yourself. It does not help on driver branches NVIDIA's
forward compatibility does not support (every CUDA call then fails with error 803).

The NCCL communicators are process-lifetime and deliberately leaked at exit: after `WHILE` graphs have run on the graph
communicator, `ncclCommDestroy`/`Abort` can wait forever for a global quiescence, so the captured graphs are destroyed
first (a registry in `proc_hooks.cuh`) and the communicators are left to the process teardown.

## Validation

Compared with OpenFOAM v2606 `-parallel` on the same decomposition (a 31,192-cell snappyHexMesh tank with contact-angle
walls, scotch decompositions, every linear solve converged to 1e-12), largest difference over the field's largest value:

| pair | steps | U | p_rgh | phi | alpha |
|---|---|---|---|---|---|
| OpenFOAM np 1 vs brae np 1 | 100 | 1.9e-9 | 3.1e-10 | 4.1e-9 | 1.8e-11 |
| OpenFOAM np 2 vs brae np 2 | 12 | 9.5e-10 | 2.3e-9 | 6.5e-9 | 8.5e-14 |
| OpenFOAM np 4 vs brae np 4 | 100 | 5.5e-9 | 4.3e-10 | 6.1e-9 | 1.8e-11 |
| brae np 1 vs brae np 2 | 100 | 4.4e-7 | 1.4e-7 | 4.7e-7 | 1.5e-7 |
| brae np 1 vs brae np 4 | 100 | 6.8e-3 | 8.2e-5 | 5.5e-3 | 3.2e-7 |
| OpenFOAM np 1 vs OpenFOAM np 4 | 100 | 6.8e-3 | 8.2e-5 | 5.5e-3 | 3.2e-7 |

A decomposed run differs from the undecomposed one by OpenFOAM's own amount: the np 4 split cuts the free surface,
where the alpha jump across processor faces changes the arithmetic. **Compare with converged solves.** At loose solver
tolerances (`p_rgh` 1e-7, relTol 0.05) OpenFOAM's own np 1, 2 and 4 runs of a case differ from each other by 8e-2 to
3e-1 of U after 12 steps, so such a comparison says nothing about the operators.

Tools and tests:

- `tools/mgpu_oracle/` — the OpenFOAM v2606 oracle: small decomposed cases, serial against `-parallel` runs, per-rank
  operator dumps, a field comparator (`compare_fields.py`), the frozen-state harness (`FROZEN.md`) and the geometry gate.
  `README.md` there describes `setup.sh` and `run_oracle.sh`.
- `tools/spikes/nccl_while_graph/` — NCCL inside a conditional graph `WHILE` body, with a per-GPU-pair sweep.
- `tools/proc_comm_nccl/` — run instructions for the NCCL backend and the placement tool on 2, 4 and 8 GPUs.
- ctest: `proc_comm*` (loopback, MPI, NCCL, graph capture), `dist_solvers_np{2,4,8}`, `mgpu_proc_geometry`,
  `processor_patch_reader`, `mgpu_np1_byte_identical`, and `mgpu_step_parity` — np 2 and 4 against the undecomposed run
  on a sheared, graded tank with converged solves and the pool's surface on the processor cut, bound 1e-6 of U
  (measured 1.2e-9). The oracle tests need OpenFOAM v2606 and a GPU, `mgpu_np1_byte_identical` and `mgpu_step_parity`
  also `mpirun`, and skip otherwise.

## Performance

Measured on one node with 8 Tesla V100-SXM2-32GB (hybrid cube-mesh NVLink, two sockets), NCCL 2.32, CUDA 12.9 with
forward-compatibility libraries (driver 535), ranks placed by the topology tool. Case: a 2,483,933-cell tank refill
(`interFoam`, laminar, MULESCorr, 3 PISO correctors, scotch decompositions), fixed deltaT, 70 steps. Step time is
rank 0's clock between steps 10 and 70, which excludes startup and output; run-to-run noise is about ±1.5 ms at np 8.

| np | ms/step | speedup | parallel efficiency |
|---|---|---|---|
| 1 | 287.6 | 1.00 | 100% |
| 2 | 181.8 | 1.58 | 79% |
| 4 | 98.2 | 2.93 | 73% |
| 8 | 56.7 | 5.07 | 63% |

Converged-solve parity, five fixed steps (tolerance 1e-12, relTol 0), largest difference from np 1 over the field's
largest value:

| np | U | p_rgh | phi | alpha |
|---|---|---|---|---|
| 2 | 4.3e-10 | 1.8e-10 | 2.3e-9 | 8.7e-11 |
| 4 | 1.7e-10 | 2.3e-10 | 7.8e-10 | 1.1e-12 |
| 8 | 3.4e-9 | 4.2e-10 | 1.7e-8 | 3.1e-10 |

Where the time goes at np 8 (Nsight Systems with node-level graph tracing, and the per-phase timer): the AMG V-cycle's
products are the largest kernel group (about 28% of GPU-busy time), then the Gauss gradient and the FP64 matrix-vector
product of the outer PCG; the GPU is busy for most of the step, so at 310,000 cells per GPU the step is bandwidth-bound on
the V100 rather than latency-bound. The distributed AMG-PCG's iteration count grows with the number of blocks (np 1: 17
iterations a solve on average, np 8: 25), which by itself puts the pressure solve at about 4.8x of np 1 instead of 8x.

What each change gave at np 8 (same case, in-run step time, one A/B switch each): the Gauss gradient reading one 32-byte
record a face instead of nine scattered values (`BRAE_CONTROL_GRAD_UNPACKED=1` restores the old kernel): 2.6 ms; the
processor-interface products as one launch (`BRAE_CONTROL_PROC_PER_PLACE_LAUNCH=1`): 0.7 ms; the coarse levels of the FP32
V-cycle as one kernel (`BRAE_AMG_FUSE_BELOW=0` switches it off): 0.9 ms. All three are bit-identical to what they replace;
60.2 ms before, 56.7 after.

Tried at np 8 with no gain beyond the run-to-run noise (about ±1 ms): the PCG convergence-check cadence, running the
solver as one CUDA graph per rank, NCCL buffer registration, channel count, thread count and protocol, CUDA connection
count, smoothed-aggregation or merged-level AMG hierarchies (`BRAE_AMG_SA`, `BRAE_AMG_MERGE`; a larger coarsest level,
`BRAE_AMG_TARGET`, is slower), pinning each rank's threads to cores of its GPU's socket, and other fusion thresholds for the
coarse V-cycle (256 and 4096 cells are no better than 1024). About 55-57 ms a step (5x) is the floor for this case without
changing the algorithm: a stronger decomposed preconditioner, or less memory traffic per V-cycle (more FP32, fewer passes).
