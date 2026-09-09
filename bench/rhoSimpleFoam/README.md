# brae rhoSimpleFoam benchmark: the OF-mirror on one GPU vs OpenFOAM on N cores

Same rules as `../run_benchmark.sh`: total wall for a fixed number of SIMPLE iterations, prep
(blockMesh, decomposePar, reconstructPar) excluded on both sides, meshes generated on the fly
and never committed -- `validation/` stays at or below 112k cells so `ctest` stays short.

```bash
cd bench/rhoSimpleFoam
./run_benchmark.sh                          # squareBend at scale 1 (112k) and 2 (896k), 100 iterations
SIZES="2 3" ITERS=200 CORES=20 ./run_benchmark.sh
MODE=converged ./run_benchmark.sh           # both codes to the tutorial's residualControl
```

The case is OpenFOAM's own `compressible/rhoSimpleFoam/squareBend` (kEpsilon, `limitedLinearV` on U,
e-thermo), with every hex block's cell counts multiplied by the scale factor in all three directions:
scale 1 is the stock 112,000 cells, 2 is 896,000, 3 is 3.02M, 4 is 7.17M.

Two columns matter and they answer different questions. `brae` / `OF-Nc` are wall seconds. The
`rel L2` group is the field disagreement AT THE LAST ITERATION -- a trajectory comparison, because
brae's pressure is AMG-PCG where the tutorial names GAMG (announced), so the two codes walk different
paths to the same fixed point. `MODE=converged` compares the destinations instead.

Results measured here go into `../results/` with the machine named in the file.

---

## `run_multisolver.sh` — brae vs every other way to put rhoSimpleFoam on a GPU

The compressible counterpart of `../H100/run_benchmark.sh` (which is simpleFoam on scaled pitzDaily).
Same case as `run_benchmark.sh` above — OpenFOAM's own `compressible/rhoSimpleFoam/squareBend`, scaled
in all three directions — with five columns instead of two:

| column | what it is |
|---|---|
| `brae` | the OF-mirror, CUDA arm, whole loop device-resident |
| `OF-Nc` | stock OpenFOAM `rhoSimpleFoam` on N cores, native GAMG, MPI |
| `SPUMA` | CINECA's device-resident OpenFOAM-v2412 fork, whole loop on the GPU |
| `OF+AMGX` | stock OpenFOAM with **only the pressure solve** offloaded to AMGX |
| `OF+PETSc` | stock OpenFOAM with **only the pressure solve** offloaded to PETSc/cuSPARSE |

```bash
cd bench/rhoSimpleFoam
CORES=24 SIZES="1 2" ITERS=100 ./run_multisolver.sh
SPUMA_BIN=$HOME/spuma/platforms/linux64NvidiaDPInt32Opt/bin/rhoSimpleFoam ./run_multisolver.sh
```

`brae` and `SPUMA` are the two whole-loop columns; `OF+AMGX` and `OF+PETSc` are OpenFOAM with one
equation moved, and reading them as "OpenFOAM on the GPU" overstates them. `CORES` must be a multiple
of 4 — the decomposition is pinned to `simple (nx 2 2)` because scotch re-partitions differently every
run and about half of its partitions diverge on this case.

**An arm that did not finish is never timed.** Compressible runs abort (the thermo Newton solve fatals
on a diverging pressure), and a wall time from a run that stopped at iteration 4 would be the fastest
number in the table. Every column shows `n/ITERSit` instead when its own log is short.

### What the offload columns actually do on this case

Measured on GB10, 112k cells, 20 iterations, stock tutorial (`transonic yes`, massFlowRate 0.5 kg/s —
near-choked):

| configuration | result |
|---|---|
| OF, GAMG (the tutorial) | 20/20 |
| OF, PBiCGStab + DILU | 20/20 (160, then 9, then 3 inner iterations) |
| PETSc, `pc_type gamg`, cuSPARSE **or** CPU matrix | **diverges** — 5/20, p residual 26.7 after 537 its at outer iteration 2 |
| PETSc, `pc_type ilu`, cuSPARSE | 20/20 |
| PETSc, `pc_type jacobi`, cuSPARSE | 20/20 |
| AMGX, nine configurations | **4/20, every one of them** |

So an aggregation-AMG preconditioner does not survive this pressure operator, on either library —
which is the same verdict brae reaches from the other side, running this pressure with a
diagonal-preconditioned BiCGStab rather than with its AMG. The PETSc arm therefore defaults to
`pc_type ilu`, the configuration that solves the case: a benchmark owes a competitor its working
configuration, not its headline one. `PETSC_PC=` overrides it.

### Open defect: `bench/amgxFoam/amgxSolver.C` under-solves

The AMGX column does not complete, and the fault is in **brae's own wrapper**, not in AMGX or in the
matrix. Three measurements say so:

- **Every** config fails at the same outer iteration — `PBICGSTAB` and `FGMRES` crossed with
  `BLOCK_JACOBI`, `MULTICOLOR_DILU`, `MULTICOLOR_GS`, aggregation AMG, and `NOSOLVER` (no
  preconditioner at all). Nine configurations, all 4/20.
- **The tolerance is not honoured.** At `main:tolerance` 1e-1, 2e-2, 5e-3 and 1e-3 the solve stops at
  13 inner iterations every time, and OpenFOAM measures the same ~0.27 final residual against a
  requested `relTol 0.1`. AMGX reports convergence on a system whose residual OpenFOAM does not see
  fall — the two are not looking at the same system.
- **Asymmetry is not the trigger**: the same `transonic yes` case at massFlowRate 0.1 runs 10/10
  through the same wrapper, and the subsonic variant does too.

Under-solved pressure in a near-choked SIMPLE loop diverges, which is exactly what the log shows.
`AMGX_CFG=<file.json>` takes a hand-written config for anyone who wants to try one; fixing the
wrapper's stopping criterion is its own unit of work.
