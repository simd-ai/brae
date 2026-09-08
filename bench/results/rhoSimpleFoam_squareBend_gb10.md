# rhoSimpleFoam: brae's OF-mirror (CUDA arm) vs OpenFOAM v2412 on 20 Grace cores -- GB10, 2026-09-07

Harness: `bench/rhoSimpleFoam/run_benchmark.sh` (stock `compressible/rhoSimpleFoam/squareBend`, kEpsilon,
`limitedLinearV` on U, e-thermo; every hex block scaled by the factor in all three directions; the
sampling functionObjects stripped from both copies; residualControl removed in fixed mode). Wall time is
the solver run only -- blockMesh, decomposePar and reconstructPar excluded on both sides. One GPU.

## Fixed 100 SIMPLE iterations

| scale | cells   | brae (s) | OF-20c (s) | brae ms/it | OF ms/it | rel L2 at iteration 100: U / p / T / k / epsilon |
|------:|--------:|---------:|-----------:|-----------:|---------:|--------------------------------------------------|
| 1     | 112,000 |      4.5 |        3.6 |         45 |       36 | 8.2e-03 / 8.1e-03 / 6.2e-03 / 1.2e-02 / 1.8e-02 |
| 2     | 896,000 |     54.7 |       27.2 |        547 |      272 | 5.4e-02 / 3.6e-02 / 2.5e-02 / 4.2e-01 / 4.0e-01 |

The rel L2 column is a TRAJECTORY comparison at a matched iteration count: brae solves p with AMG-PCG
where the tutorial names GAMG (a substitution brae announces), so the two codes walk different paths
to the same fixed point. The converged comparison below is the one that says whether they arrive at
the same place.

What the numbers say about speed: at 112k the two are at parity; at 896k brae's compressible mirror
costs 547 ms per iteration against 272 for OpenFOAM on 20 cores -- 0.61 ms per iteration per thousand
cells, where the incompressible simpleFoam bench on this same GB10 reads 0.22 at 4.9M cells
(`totalwall_100iter.txt`). The compressible path is 2.8x more expensive per cell than the
incompressible one on the same machine; the phase split below says where.

## Where the 547 ms go (896k cells, `BRAE_PHASE_TIME=1`, 20 iterations)

| phase      | ms/it | share |
|------------|------:|------:|
| pEqn       | 315.4 |   72% |
| UEqn       |  71.7 |   16% |
| turbulence |  36.6 |    8% |
| EEqn       |  16.7 |    4% |
| (sum)      | 440.5 |       |

The pressure equation is three quarters of the iteration. This is the known scaling behaviour of
brae's default AMG (item 61 declined GAMG-as-solver; the default aggregation is not mesh-independent
and its cycle count grows with the mesh -- 24 to 92 cycles from 4k to 1M cells measured earlier),
which `BRAE_AMG_SA=1` (smoothed aggregation) removes; that arm is measured next.

CORRECTION to the paragraph above, from reading the mirror's own log: on this case the pressure does
NOT go through AMG-PCG at all. squareBend is `transonic yes`, and a transonic pressure matrix is
asymmetric (`fvm::div(phid, p)`: upper != lower at every face with flow through it), so a symmetric
solver is wrong there and the mirror takes its BiCGStab branch (rhoSimpleFoam.cu, `if (in.transonic)`)
with a JACOBI preconditioner -- announced as `solvers/p: case asks 'GAMG', brae runs PBiCGStab
preconditioned with diagonal`. That is the 315 ms: a diagonal-preconditioned BiCGStab on 896,000
cells. `BRAE_AMG_SA` is irrelevant on this path. The candidate levers were DILU as the BiCGStab
preconditioner for the transonic p (the device DILU exists and serves U and k on this mirror since
item 74) and, beyond that, an AMG-preconditioned BiCGStab for asymmetric pressure matrices, which brae
does not have. DILU was built and measured the same day -- see "The DILU lever, measured" below: it
is a LOSS at both sizes.

## Converged (MODE=converged, the tutorial's own residualControl: p 1e-3, U 1e-4, e 1e-3, k/eps 1e-3)

| cells   | brae (s) | brae iterations | OF-20c (s) | OF iterations |
|--------:|---------:|----------------:|-----------:|--------------:|
| 896,000 |    232.1 |             382 |       83.9 |           317 |

Agreement of the two CONVERGED states (brae at 382, OpenFOAM at 317), relative L2 over the field:

| U       | p       | T       | rho     | k       | epsilon |
|--------:|--------:|--------:|--------:|--------:|--------:|
| 3.2e-04 | 2.1e-04 | 3.5e-04 | 3.7e-04 | 2.6e-03 | 1.9e-03 |

So the two codes arrive at the same place: the 4e-01 on k at a fixed 100 iterations was the two paths,
not the two destinations, and the residual disagreement at convergence is the slack the tutorial's own
stopping criteria leave (p 1e-3), reached at different points by the two solvers.

## The DILU lever, measured (2026-09-07): fewer iterations, more time, at both sizes

RhoStepInput::preconP now lets the transonic p BiCGStab take the device DILU (the one U and k use).
Twenty iterations, `BRAE_PHASE_TIME=1`, the summary line's new `pIters` (BiCGStab iterations of the
first p solve of the outer iteration):

| cells   | preconditioner | pIters, iteration 1 | pEqn ms/it | whole iteration ms/it |
|--------:|----------------|--------------------:|-----------:|----------------------:|
| 112,000 | diagonal       |                 668 |       67.7 |                 235.9 |
| 112,000 | DILU           |                 202 |      473.9 |                 717.4 |
| 896,000 | diagonal       |                 668 |      315.4 |                 440.5 |
| 896,000 | DILU           |                 306 |      567.1 |                 687.1 |

(112k is validation/sbMatched, whose p entry is `PBiCGStab; preconditioner DILU; tolerance 1e-12;
relTol 0` -- both solves converge fully there and the p residual trajectories agree to the printed
digits over the first three iterations, gate tests/transonic_p_dilu.sh. 896k is the scaled tutorial
with its `GAMG; tolerance 1e-7; relTol 0.01`, which brae substitutes either way.)

Three times fewer BiCGStab iterations and 1.8x (896k) to 7x (112k) MORE time on the phase: one DILU
apply is a level-scheduled walk (item 70's per-level launch floor) and at ~2 applies per BiCGStab
iteration it costs more than the iterations it saves. The 112k ratio is the worse one because the
diagonal solve is only 68 ms there, so the apply's fixed cost dominates outright.

Decision, under the rule that the same value by a faster method wins: the diagonal stays the default
and is announced (`case asks 'DILU', brae preconditions with diagonal` on sbMatched; `case asks 'GAMG',
brae runs PBiCGStab preconditioned with diagonal` on the tutorial); `BRAE_DILU_P=1` opts into the
case's DILU where the p entry names it. The compressible speed lever is therefore NOT a preconditioner
swap: it is a faster DILU apply (a single-block walk where the level widths allow it, as the smoother
work did) or an AMG-preconditioned BiCGStab for the asymmetric p. Neither is built.

## 504k cells (scale 1.65, 2026-09-07): the middle point

The bench script now takes a fractional scale; 1.65 in all three directions puts the tutorial at
504,207 cells. Same protocol as above (OpenFOAM on 20 cores, brae CUDA mirror, prep excluded).

| mode                          | brae (s) | brae it | OF-20c (s) | OF it | ratio |
|-------------------------------|---------:|--------:|-----------:|------:|------:|
| fixed 100 iterations          |     25.9 |     100 |       15.0 |   100 |  1.7x |
| converged (residualControl)   |    100.0 |     299 |       35.9 |   251 |  2.8x |

Agreement at a fixed 100 (trajectory): U 4.2e-02, p 2.8e-02, T 2.6e-02, k 3.9e-01, epsilon 3.3e-01.
Agreement of the two CONVERGED states: U 4.2e-04, p 3.3e-04, T 1.7e-04, k 9.3e-04, epsilon 7.9e-04 --
same destination, as at 896k.

Phase split (BRAE_PHASE_TIME=1, 20 iterations, the diagonal BiCGStab on the transonic p):

| phase      | ms/it | share |
|------------|------:|------:|
| pEqn       | 134.0 |   66% |
| UEqn       |  37.2 |   18% |
| turbulence |  21.6 |   11% |
| EEqn       |  10.0 |    5% |
| (sum)      | 202.7 |       |

So across 112k / 504k / 896k the transonic pressure is 29% / 66% / 72% of the iteration and the brae-to-
OpenFOAM ratio at a fixed 100 goes 1.25x / 1.7x / 2.0x: the diagonal BiCGStab's iteration count grows with
the mesh where OpenFOAM's GAMG does not (46 -> 7 iterations at 896k). Item 77b is the lever.

### The same 504k case on the second box (RTX 2060, 6 GB, CUDA 13.1, Ubuntu 26.04)

| GPU                 | 100 iterations (s) | device memory |
|---------------------|-------------------:|--------------:|
| GB10 (this file)    |               25.9 |       (unified) |
| RTX 2060            |               34.0 |      1,644 MiB |

Whole-process wall in both rows. A 2019 consumer card is within 1.3x of the GB10 on this solver: brae is
bandwidth-bound and the 2060's GDDR6 (336 GB/s) is not far from the GB10's LPDDR5x. 504k cells of the
compressible mirror take 1.6 GB, so anything up to ~1.5M cells fits the 2060.

## The subsonic path: what does NOT measure it (2026-09-07)

Flipping the tutorial's `transonic yes` to `no` on the 0.5 kg/s case is not a subsonic measurement:
OpenFOAM aborts (MPI abort, non-finite) within 4 to 7 iterations at every size, and brae's AMG-PCG then
grinds on a diverging state (46 s per 100 iterations at 112k against 4.5 s transonic). The case is
transonic; the pressure-equation form is not a knob. The subsonic point uses the same mesh with the
inlet at 0.1 kg/s (`MASSFLOW=0.1 TRANSONIC=no` in run_benchmark.sh), measured below.

## Crossover on the transonic path (2026-09-07): there is none

The question "where does brae overtake OpenFOAM on 20 cores" for the compressible solver, on the one case,
brae CUDA mirror vs OpenFOAM-20c, fixed 100 iterations, downward sweep added to the sizes above:

| cells   | brae (s) | OF-20c (s) | brae / OF |
|--------:|---------:|-----------:|----------:|
|  14,000 |      2.2 |        1.3 |     1.7x  |
|  38,416 |      2.9 |        1.8 |     1.6x  |
|  69,071 |      3.4 |        2.5 |     1.4x  |
| 112,000 |      4.5 |        3.6 |     1.25x |
| 504,207 |     25.9 |       15.0 |     1.7x  |
| 896,000 |     54.7 |       27.2 |     2.0x  |

brae is slower at EVERY size. The ratio bottoms at 1.25x near 112k and opens both ways: below it, brae's
fixed per-iteration cost (kernel launches, the pressure graph, host syncs) does not shrink with the mesh
while OpenFOAM's 20 cores still have work to share; above it, the diagonal-BiCGStab's iteration count on
the transonic p grows with the mesh where GAMG's does not (46 -> 7 iterations at 896k). The incompressible
solver's crossover (bench/results/crossover.csv, motorbike, ~12M cells) does not carry over: that path runs
AMG-PCG on a symmetric p. So the compressible crossover is not a mesh size to wait for; it appears only
when item 77b (an AMG-preconditioned BiCGStab for the asymmetric p) is built, and the subsonic sweep below
says what the symmetric path alone would give.

## Block by block at 306k cells (2026-09-07): OpenFOAM on 20 cores vs the brae CUDA mirror

The user's question: which block is slow. OpenFOAM's side is tools/timeRhoSimpleFoam -- OpenFOAM's own
rhoSimpleFoam with wall-clock timers around each include and each linear solve, MAX over the 20 ranks,
one `PHASE` line per iteration (mean over iterations 2..100). brae's side is `BRAE_PHASE_TIME=1`, which
now also reports the linear solves alone; its whole-iteration wall is (15.23 s at 100 it - 3.16 s at
10 it) / 90. Scale 1.39 of the tutorial, 305,760 cells, fixed 100 iterations (bench/rhoSimpleFoam/
phase_table.py builds the table from the two logs).

| block, ms per iteration     | OF-20c | brae  | brae / OF |
|-----------------------------|-------:|------:|----------:|
| UEqn, assemble + solve      |   26.1 |  29.1 |     1.12  |
|   of which the U solve      |   17.6 |  28.1 |     1.60  |
| EEqn, assemble + solve      |    9.8 |   7.1 |     0.72  |
|   of which the he solve     |    5.4 |   6.3 |     1.17  |
| pEqn, assemble + solve      |   24.0 |  77.8 |     3.24  |
|   of which the p solve      |   10.0 |  74.6 |     7.46  |
| turbulence, k + epsilon     |   22.9 |  18.0 |     0.78  |
| rest of the iteration       |    0.7 |   2.2 |           |
| whole iteration             |   82.5 | 134.2 |     1.63  |

Reading it:
- **The pressure SOLVE is the whole gap.** 74.6 ms of brae's 134 against 10.0 of OpenFOAM's 82: the
  diagonal-preconditioned BiCGStab on the asymmetric transonic matrix (pIters 467 at iteration 2, 94 at
  51, 176 at 100) against GAMG at 46 -> 7 iterations. Take that block to parity and brae's iteration is
  ~70 ms against OpenFOAM's 82. Item 77b.
- **Assembly is not the problem anywhere:** U 1.0 vs 8.5 ms, he 0.8 vs 4.4, p 3.2 vs 14.0 -- brae's
  assembly and corrections are 4-8x faster than 20 cores.
- **The U solve is 1.6x slower** (28.1 vs 17.6): three DILU-BiCGStab component solves, each a
  level-scheduled DILU apply per iteration. Second lever, ten times smaller than the first.
- **Energy and turbulence are already faster** than 20 cores (0.72x, 0.78x).
- brae's prep (mesh read, device upload, DILU build) is ~1.8 s once, excluded from every row.

## The momentum block, taken apart (2026-09-07, 306k cells)

What brae runs for U on this case: the tutorial asks `GAMG` with a GaussSeidel smoother at relTol 0.1;
brae substitutes a diagonal-preconditioned BiCGStab per component, announced. Measured:

| quantity                                              | value |
|-------------------------------------------------------|------:|
| BiCGStab iterations per outer iteration, 3 components | 25-36 after start-up (sum 457 over 20) |
| U solve wall per outer iteration                      | 24-28 ms |
| wall per BiCGStab iteration                           | 1.05 ms |
| amulKernel (LDU SpMV) at 306k, from nsys              | 155 us per call, 43 MB of traffic: 277 GB/s = the GB10's bandwidth |
| kernel time per BiCGStab iteration (2 SpMV + dots + axpys) | ~0.45 ms; the other ~0.6 ms is graph-node gaps (~20 nodes, six of them 1-thread scalar kernels) and per-solve setup |
| OpenFOAM-20c                                          | 1-2 GAMG cycles per component, 17.6 ms for the three |
| DILU-BiCGStab on U (case copy asking it)              | 1.5 iterations per component (5.8x fewer) but 55.9 ms: the level-scheduled apply |

The SpMV is at the roofline; the block is slow because the diagonal preconditioner needs ~10 SpMV pairs
per component and each pair carries a graph iteration's worth of gaps, and because the three components
read the same matrix three times.

Offline, on the SOLVED systems dumped at iteration 10 (BRAE_STAGE_DUMP_DIR, UsolveDiag/UsolveB/UUpper/
ULower; bench/rhoSimpleFoam/u_precond_experiment.py, OpenFOAM's normFactor stopping rule, relTol 0.1;
the mesh is bipartite, so red-black is 2 colours). Iterations to the stopping rule, X / Y / Z:

| candidate                                  | iterations | cost per iteration (SpMV-equivalents) |
|--------------------------------------------|-----------:|---------------------------------------|
| BiCGStab + Jacobi (today)                   |  9 / 11 / 13 | 2 SpMV + ~4 dots + ~20 graph nodes |
| BiCGStab + DILU (OpenFOAM's)                |  1 /  2 /  2 | 2 SpMV + 2 sequential walks (~300 levels) |
| BiCGStab + red-black symmetric GS           |  5 /  8 /  7 | 2 SpMV + 2x(4 half-sweeps) |
| BiCGStab + 3-step Jacobi polynomial         |  3 /  4 /  5 | 6 SpMV |
| red-black Gauss-Seidel sweeps, no Krylov    | 11 / 10 /  9 | 1 SpMV (2 launches), 1 residual reduction |
| Jacobi sweeps, no Krylov                    | 20 / 19 / 18 | 1 SpMV (1 launch), 1 residual reduction |

Diagonal dominance is 1.11 on every row (the 0.7 relaxation), which is why plain sweeps reach relTol
0.1 at all -- and why OpenFOAM's GAMG needs no more than 1-2 cycles of its GaussSeidel smoother here.
DILU's strength is the natural ordering following the flow on a blockMesh; red-black breaks it, so the
GS preconditioner buys less than DILU but each apply is two full-width launches instead of 300.

Confirmed on the systems dumped at iteration 60 (the flow developed): BiCGStab+Jacobi 8 / 6 / 9,
+3-step polynomial 3 / 2 / 3, +red-black SGS 4 / 3 / 4, red-black GS sweeps alone 11 / 11 / 11, Jacobi
sweeps alone 21 / 21 / 20, +DILU 1 / 1 / 1. The counts are a property of the relaxed momentum matrix
(dominance 1.11), not of the start-up.

Cost model per component at 306k (SpMV 0.155 ms; today's BiCGStab iteration 1.05 ms of which ~0.6 ms is
graph-node gaps and dots): today ~8.5 ms; BiCGStab + polynomial ~4.3 ms; BiCGStab + red-black SGS ~5 ms;
red-black GS sweeps ~2.1 ms (11 x (two half-sweep launches + one residual reduction)); the three
components fused in one sweep kernel that reads each matrix row once ~2.4 ms for ALL THREE (matrix
traffic 21.6 MB once instead of three times). That last figure against 25-28 ms today is the lever.

## The momentum arms, measured (2026-09-07, 306k cells, 20 iterations, bench/rhoSimpleFoam/momentum_arms.sh)

Every arm solves the same systems under the tutorial U entry's tolerance 1e-8 / relTol 0.1 / maxIter 1000;
only the solver words change. UEqn is assembly (~1 ms) plus the solve; uIters is the BiCGStab iteration
or Gauss-Seidel sweep count summed over the solved components and the 20 outer iterations.

| arm       | what runs                                                   | UEqn ms/it | uIters |
|-----------|-------------------------------------------------------------|-----------:|-------:|
| jacobi    | PBiCGStab, diagonal (brae's substitute for the case's GAMG) |       28.2 |    530 |
| dilu      | PBiCGStab, DILU (level-scheduled apply)                     |       57.9 |     90 |
| hostGS    | smoothSolver GaussSeidel, OpenFOAM's order, on the CPU      |       58.6 |    543 |
| devGS     | the same, level-scheduled on the device (374 levels)        |       36.3 |    560 |
| colourGS  | the same stop rule, COLOUR order, three components fused    |       20.7 |    501 |

colourGS v1 is the best arm and 1.36x better than today's substitute -- and 4x short of the 5 ms the
sweep count promised. nsys on that arm: colourSweepKernel<3> 262 us per colour launch (524 us per full
sweep of three components) and residualKernel<3> 225 us per pass, against a bandwidth-ideal sweep of
about 260 us. The cause is the layout, not the arithmetic: on a naturally numbered hex mesh the two
colour classes are stride-2 subsets, so every colour launch touches every cache line of the per-cell and
per-face arrays and uses half of it -- 147 MB per sweep, which at 277 GB/s is the 524 us measured. The
block-per-component layout (a row read once per component) measured 30.7 ms/it and is withdrawn. v2 of
the engine sweeps a colour-major permuted copy of the system (permute once per solve, contiguous blocks
per colour, scatter psi back), bit-identical arithmetic, measured below.

Correctness of colourGS, before any of the speed work: tests/test_colour_gs_fused (device vs a host
reference under OpenFOAM's loop: identical sweep counts, residuals and psi to 1e-13; controls: OpenFOAM's
natural order leaves a different iterate, a corrupted colouring is a different solve) and
tests/u_colour_gs_vs_openfoam.sh (rhoKE vs live OpenFOAM, U on a smoothSolver GaussSeidel entry at
tolerance 1e-14: colourGS 9.4e-13 / 1.2e-12 / 6.9e-13 / 5.1e-13 on U / p / k / epsilon, today's own-order
path 8.8e-13 / 1.2e-12 / 6.3e-13 / 4.7e-13, bound 1e-9; at relTol 0.1 the two orders stop elsewhere and
differ by 2e-4 to 5e-4; brae capped at one sweep misses by 4e-3 to 9e-3; every notice truthful).

### colourGS v2 and the wait (2026-09-08)

v2 (colour-major permuted layout, bit-identical iterates): colourSweepKernel<3> 115 us per colour launch,
231 us per full sweep of three components against v1's 524 -- the layout fix worked on the kernel -- yet
the block only went 20.7 -> 18.7 ms/it. nsys, one pass: the GPU finished the residual, the three sums and
the 24-byte readback at 310 us and the next sweep launched at 630; the host had been inside
cudaStreamSynchronize the whole time, returning ~300 us AFTER the work it waited for was done (the OS
side: poll and sem_wait dominate). A pinned readback buffer changed nothing (18.2). So the residual sums
are now PUBLISHED by a one-thread kernel into mapped pinned host memory behind a sequence number and the
host spins on it (a two-second fallback to the sync, then a throw): 14.1 ms/it, iterates unchanged.

| colourGS version                                   | UEqn ms/it | per pass |
|----------------------------------------------------|-----------:|---------:|
| v1: colour classes over the natural numbering      |       20.7 |  0.80 ms |
| v2: colour-major layout                            |       18.7 |  0.72 ms |
| v2 + pinned readback                               |       18.2 |  0.70 ms |
| v2 + mailbox (device publish, host spin)           |       14.1 |  0.48 ms |

At 0.48 ms the pass is the GPU: sweep 0.23 + the residual pass 0.24 + the three reductions. What is
left: the residual pass is a full second read of the matrix per sweep because smoothSolver.C:189-197
evaluates it that way; on a 2-colour mesh the residual of the last colour's rows after a sweep is the
round-off of their own update and the other colour's can be taken inside the next sweep's first kernel
before it updates (a speculative sweep, rolled back exactly if the rule says stop), which would make the
residual free and the block ~8 ms; a conditional WHILE graph would remove the remaining launch gaps.

### The momentum arms with the final engine (2026-09-08, 306k, 20 iterations)

| arm       | what runs                                                   | UEqn ms/it | uIters | vs today |
|-----------|-------------------------------------------------------------|-----------:|-------:|---------:|
| jacobi    | PBiCGStab, diagonal (brae's substitute for the case's GAMG) |       27.6 |    491 |    1.00x |
| dilu      | PBiCGStab, DILU (level-scheduled apply)                     |       59.4 |     89 |    0.46x |
| hostGS    | smoothSolver GaussSeidel, OpenFOAM's order, on the CPU      |       59.6 |    560 |    0.46x |
| devGS     | the same, level-scheduled on the device (374 levels)        |       36.2 |    549 |    0.76x |
| colourGS  | the same stop rule, COLOUR order, fused, colour-major, mailbox |    14.5 |    504 |    1.90x |

Same systems, same tolerance / relTol / maxIter / minIter in every arm. The gate on the final engine:
EXACT 9.363e-13 / 1.185e-12 / 6.858e-13 / 5.054e-13, CONTROL 2.0e-4 .. 5.5e-4, FAIL-PROOF 3.6e-3 .. 9.3e-3
(the same numbers as v1: the layout and the mailbox changed no iterate). For the whole 306k iteration the
momentum block goes from 29 to ~15 ms against OpenFOAM-20c's 26; the pressure solve (75 ms against 10) is
the block that remains.

After the second review round (no-op symmetric launches skipped, normFactors through the mailbox, the
colouring bound to its mesh's addressing): jacobi 27.7, dilu 58.7, hostGS 58.3, devGS 36.8,
colourGS 13.5 ms/it -- 2.05x on the block; gate and unit numbers unchanged (2026-09-08).

### The residual from the sweep (2026-09-08)

On a two-colour mesh the residual pass was redundant: a row's residual after its own update is the
round-off of that update (acc - d x, with acc the value the launch already holds), and the other
colour's residual after the sweep is exactly what the next sweep's first launch computes before it
overwrites those rows. So the block's last launch also writes its rows' residual, the next block's first
launch runs speculatively (saves the old values, writes the other rows' residual, then updates) and is
rolled back for any component the rule stops. The fused rows and the explicit pass produce the SAME
numbers row for row (proven bit for bit in the unit test), so the stop decisions are unchanged.

| arm (306k, 20 iterations) | UEqn ms/it | vs today |
|---------------------------|-----------:|---------:|
| jacobi (today)            |       27.2 |    1.00x |
| colourGS + residual fusion|       11.7 |    2.32x |

One pass now (nsys, GB10): RES_NEW launch 130 us + speculative launch 155 us + three reductions 50 us
+ publish 2 us = 360 us, with no gap above 7 us -- the pass is the GPU. What is left is the reductions
(14%, foldable into the sweep kernels as block partials) and the speculative launch's save (~25 us). A
conditional WHILE graph would now buy almost nothing here.

### The reductions folded into the launches (2026-09-08)

The three sum|r| reductions and the residual-vector writes are gone: each residual-writing launch
reduces |r| per thread block in shared memory (fixed index order, no atomics) and writes one partial per
block and component; one 2.7 us kernel sums the partials and the mailbox publishes as before. The
production path stores no residual vector at all.

| arm (306k, 20 iterations) | UEqn ms/it | vs today |
|---------------------------|-----------:|---------:|
| jacobi (today)            |       26.0 |    1.00x |
| colourGS, all four steps  |       10.9 |    2.39x |

One pass (nsys): 265 us -- the RES_NEW launch 115, the speculative launch 134, the final sum 2.7, the
publish 2.3, no gap above 6.4 us. Against the 524 us the first colour version spent on its two sweep
launches alone. The momentum solve is 7.4 ms per outer iteration inside a block of 10.9.

Progress on the momentum block at 306k, all with the same stop rule and the same converged answer:

| step                                                | UEqn ms/it |
|-----------------------------------------------------|-----------:|
| today (diagonal-preconditioned BiCGStab)            |       26.0 |
| colour-ordered fused Gauss-Seidel, natural numbering|       20.7 |
| + colour-major layout                               |       18.7 |
| + mailbox instead of the stream sync                |       14.1 |
| + no-op launches skipped, normFactors in the mailbox|       13.5 |
| + residual taken from the sweeps                    |       11.7 |
| + reductions folded into the launches               |       10.9 |

### What a case's own smoother entry gets (2026-09-08, 306k, 20 iterations)

OpenFOAM's ASYMMETRIC smoother table -- the momentum matrix is asymmetric -- holds exactly five names
(each smoother's .C and its addasymMatrixConstructorToTable): GaussSeidel, symGaussSeidel,
nonBlockingGaussSeidel, DILU, DILUGaussSeidel. brae ported the two Gauss-Seidel forms exactly and has
none of the others. What the mirror's CUDA arm runs on U now, measured on the same case:

| the case's solvers/U entry            | what runs            | UEqn ms/it | before the default |
|---------------------------------------|----------------------|-----------:|-------------------:|
| GAMG                                  | colour Gauss-Seidel  |       11.1 | 27.9 (diagonal BiCGStab) |
| smoothSolver, smoother GaussSeidel    | colour Gauss-Seidel  |       10.9 | 58.0 host / 36.4 device, OpenFOAM's own order |
| smoothSolver, smoother DILUGaussSeidel| colour Gauss-Seidel  |       11.0 | 27.9 (diagonal BiCGStab: brae has no DILU smoother) |
| smoothSolver, smoother nonBlockingGaussSeidel | colour Gauss-Seidel |  10.9 | 27.9 (same) |
| PBiCGStab / PBiCG / PCG               | the case's own solver|       27.9 | unchanged: brae has that solver |

No entry is slower than it was. Each is announced by name, and a DILU-family smoother additionally gets
"Gauss-Seidel is the WEAKER smoother per sweep ... may need more sweeps and, under a maxIter cap, stop
short of its tolerance", because that is the one family where the substitute is weaker than what the
case asked for. Sweep counts over 20 iterations were 460 (DILUGaussSeidel entry), 461
(nonBlockingGaussSeidel) and 502 (GaussSeidel) -- the same solver in all three, the entry only changes
the stop rule it is given.

### brae's Krylov port against OpenFOAM's, on the entry that names it (2026-09-08, 306k, 20 iterations)

The table above compared brae to OpenFOAM on the tutorial's GAMG entry. This is the other comparison:
the velocity pinned to `PBiCGStab` in BOTH codes, so each runs its own port of the same algorithm.

| solvers/U                        | OpenFOAM-20c UEqn | of which the solve | brae UEqn | of which the solve | brae forced to colourGS |
|----------------------------------|------------------:|-------------------:|----------:|-------------------:|------------------------:|
| PBiCGStab, preconditioner DILU   |              22.6 |               14.6 |      62.4 |               60.7 |                    11.0 |
| PBiCGStab, preconditioner diagonal |            37.7 |               29.0 |      27.1 |               25.4 |                    10.6 |

Read it the other way round from the GAMG table: with the DIAGONAL preconditioner brae's BiCGStab beats
20 cores (25.4 against 29.0 on the solve); with DILU it loses badly (60.7 against 14.6), because DILU's
forward-backward walk is sequential by construction -- 300 levels on this mesh -- while on 20 cores it
is 20 independent local factorisations that cost almost nothing extra and cut the iteration count. That
is the same finding as the transonic pressure's (BRAE_DILU_P): DILU is a CPU preconditioner. Note also
that OpenFOAM is FASTER with DILU and brae is faster WITHOUT it, so the two codes disagree about which
setting is better for the same case.

And the colour sweep beats every one of those four numbers by 2.3x to 5.7x. It is not the default on
these entries because the case named a solver brae implements and substituting it costs agreement with
OpenFOAM (rho_sbmatched_transient: 4.8e-12 becomes 7.7e-10); BRAE_U_SOLVER=colourGS forces it for
anyone who wants the speed on such a case.

### The OpenFOAM reference at 306k is decomposition-sensitive (2026-09-08)

Running the same staged case twice, OpenFOAM finished 100 iterations once and aborted at iteration 4 the
other time: `Maximum number of iterations exceeded: 100 when starting from T0:1001.63 old T:-1.26348e+15`
from the thermo's Newton solve, with every residual back at 1.0 the iteration before. It is the
DECOMPOSITION, not the run: three reruns on one fixed scotch partition all converged, three fresh
`simple` decompositions all converged, and 2 of 4 fresh scotch decompositions diverged. scotch
re-partitions differently every time, the partition changes the reduction order, and this case from a
cold start (transonic, consistent yes, pMinFactor 0.1 / pMaxFactor 2) is marginal enough at its first
iterations for that to decide it. run_benchmark.sh now writes `method simple; n (5 2 2)` so the
reference number comes from a reproducible run. Nothing here is brae's: its own arm ran 100 iterations
in every one of those runs.

## The transonic pressure, preconditioned with the AMG V-cycle (2026-09-08)

The pressure equation was three quarters of the compressible iteration: an asymmetric matrix, a
diagonal-preconditioned BiCGStab, and an iteration count that grows with the mesh where OpenFOAM's GAMG
does not. brae's AMG hierarchy turned out to be asymmetric-correct already (it agglomerates on face
areas, and its coarse operator is OpenFOAM's own asymmetric branch); what it lacked was a valid coarsest
solve and a seam to hang it on a BiCGStab.

| 306k, 20 iterations           | p iterations (first / mean) | p solve ms/it | p phase ms/it |
|-------------------------------|----------------------------:|--------------:|--------------:|
| diagonal (what it was)        |                 680 / 110.7 |          65.7 |          69.4 |
| AMG V-cycle (the default now) |                    20 / 3.5 |          23.6 |          29.4 |

OpenFOAM's own PBiCGStab on the same matrix, 20 cores, for scale: diagonal 380.8 iterations / 278.1 ms,
DILU 135.3 / 143.9, GAMG 3.8 / 33.8. brae's 3.5 sits on OpenFOAM's 3.8.

The whole iteration at 306k, against OpenFOAM-20c's 82.5 ms measured block by block earlier:

| block          | OF-20c | brae before this round | brae now |
|----------------|-------:|-----------------------:|---------:|
| UEqn           |   26.1 |                   29.1 |     11.0 |
| EEqn           |    9.8 |                    7.1 |      7.0 |
| pEqn           |   24.0 |                   77.8 |     29.4 |
| turbulence     |   22.9 |                   18.0 |     14.2 |
| whole iteration|   82.5 |                  134.2 |     ~62  |

Over the SAME range OpenFOAM's column was averaged on (100 iterations, not 20), brae reads: UEqn 10.8,
EEqn 7.0, pEqn 25.5 (solve 20.5), turbulence 17.9 -- p iterations first 20, mean 2.9. So the pressure is
the ONE block where 20 CPU cores are still ahead of one GPU, 24.0 against 25.5, and everything else is
2.4x, 1.4x and 1.3x the other way. The whole iteration is 61.2 against 82.5.
