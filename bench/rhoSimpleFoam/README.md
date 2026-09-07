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
