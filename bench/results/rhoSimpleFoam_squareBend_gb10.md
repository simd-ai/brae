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

## The size sweep after the momentum and pressure work (2026-09-08)

The same campaign that opened this file, re-run with the colour Gauss-Seidel momentum solver and the
AMG-preconditioned transonic pressure in place. 100 iterations, prep excluded, OpenFOAM on 20 cores with
the deterministic decomposition:

| cells   | brae (s) | OF-20c (s) | brae is |
|--------:|---------:|-----------:|--------:|
|  38,416 |      2.1 |        2.2 |   1.05x |
| 112,000 |      3.8 |        4.0 |   1.05x |
| 305,760 |      8.0 |        9.1 |   1.14x |
| 896,000 |     21.9 |       31.1 |   1.42x |

Against the same table at the start of the session -- 112k 4.5 against 3.6, 504k 25.9 against 15.0,
896k 54.7 against 27.2, i.e. brae 1.25x to 2.0x SLOWER and getting worse with size. It is now faster at
every size and the margin GROWS with the mesh, which is the shape the hardware should give: the CPU
side is bandwidth-starved sooner. The crossover the earlier entry looked for and did not find (`the
compressible crossover is not a mesh size to wait for; it appears only when item 77b is built`) is
below the smallest mesh here.

The rel L2 columns are trajectory at a fixed 100 iterations, not destination -- both codes solve the
same equations and the converged states agree (see the converged table above); the k and epsilon
columns grow with the mesh because the two codes take different paths through the same transient.

## Choosing the V-cycle's smoother -- ACROSS THREE MESHES, because one was misleading (2026-09-08)

The transonic pressure's V-cycle smoother, p SOLVE ms per outer iteration and V-cycles per solve, on the
same case at three sizes (20 iterations each):

| smoother, pre/post sweeps        |     112k |     306k |     896k |
|----------------------------------|---------:|---------:|---------:|
| weighted Jacobi 1/1 (what ran)   | 9.6 /2.6 | 23.0/3.5 | 67.1/4.8 |
| weighted Jacobi 0/2              | 9.4 /2.6 | 20.0/2.9 | 68.0/5.0 |
| two-stage Gauss-Seidel 1/1       | 9.6 /2.1 | 19.9/2.7 | 64.2/3.6 |
| two-stage Gauss-Seidel 1/2       | 8.9 /1.6 | 19.6/2.0 | 68.5/3.1 |
| multicolour Gauss-Seidel 0/2     |        - | 25.1/2.5 |        - |
| strength-of-connection filter    |        - |     1885 |        - |

Two of these win on ONE mesh and lose on another. Jacobi with OpenFOAM's own 0-pre/2-post shape is 13%
better than the baseline at 306k and WORSE than it at 896k; two-stage with two post-sweeps is the best of
all at 112k and 306k and the worst at 896k. Tuned on 306k alone, either would have shipped. The default
is two-stage 1/1, the only one at least as good as the baseline at every size and the best at the size
where the time hurts, and at tight tolerance (sbMatched, 1e-12) it takes 45/39/36 outer iterations
against Jacobi's 53/50/43. It is the default on the ASYMMETRIC path only (useTSGSAsym), so the subsonic
AMG-PCG and its gates are untouched; BRAE_AMG_TSGS=0 restores the Jacobi and the gate holds that arm to
the same bound. The multicolour smoother needs the fewest cycles of the Gauss-Seidel family and loses on
apply cost for the reason the momentum solver already met: its sweep walks a cells[] indirection instead
of a colour-major layout. Porting that layout into the AMG smoother is the next lever on this block.

## The colour-major layout in the AMG's multicolour smoother: it works, and it still loses (2026-09-08)

The multicolour Gauss-Seidel smoother inside the V-cycle walked a cells[] indirection over the natural
numbering, the same layout tax the momentum solver shed. It now has the colour-major permuted layout:
the permutation built once per grid (it is a function of that grid's graph), the coefficients gathered
once per outer iteration inside amgGalerkin, the sweep over contiguous colour blocks, and the scatter
fused into the sweep. Bit-identical to the old sweep by memcmp on every grid, forward and backward,
with both fail-proofs run (a mathematically identical but differently rounded division, and a frozen
coefficient gather).

p solve ms per outer iteration and V-cycles per solve, transonic pressure, three sizes:

| smoother                              |     112k |     306k |     896k |
|---------------------------------------|---------:|---------:|---------:|
| two-stage Gauss-Seidel (the default)  | 9.2 /2.1 | 20.2/2.7 | 65.0/3.6 |
| multicolour GS, colour-major (new)    | 14.6/2.3 | 25.3/2.6 | 73.2/3.6 |
| multicolour GS, old indirection       | 13.6/2.3 | 25.3/2.6 | 82.6/3.6 |

The layout does what it was built for -- 82.6 to 73.2 at 896k, 11%, with the cycle counts unchanged,
which is the field-level evidence that the sweeps really are the same solver. But multicolour GS is not
competitive with the two-stage smoother at any size, and the reason is structural rather than a layout
one: a colour sweep costs ONE KERNEL LAUNCH PER COLOUR PER LEVEL, and on a twelve-level hierarchy whose
lower levels hold a few hundred cells those launches cost more than the arithmetic they carry, while the
two-stage smoother needs two passes per level whatever the colouring. The default stays two-stage; the
layout stays as a strict improvement to the opt-in path (BRAE_AMG_GS), which the incompressible AMG
shares. BRAE_AMG_GS_PERM=0 restores the indirection sweep.

LESSON, the general form: a colour-ordered smoother is a poor fit for a DEEP hierarchy even after its
layout is fixed, because its launch count scales with colours times levels. Fixing the layout was still
worth doing -- it is the same 2x traffic saving as everywhere else -- it simply cannot outrun that.

## Fusing the vector gradient (2026-09-08)

With the solvers done the iteration is launch- and bandwidth-bound, not arithmetic-bound: at 305,760
cells it is about 30 ms of GPU-busy time inside about 61-65 ms of wall over ~914 kernel launches. The
largest ASSEMBLY item was the gradient -- 16 launches per outer iteration at 288 us each, 4.5 ms/it,
second only to the matrix-vector product's 5.9 ms over 98 calls.

deviceGaussGrad differentiates ONE scalar, so a velocity gradient was three launches, each re-reading
the entire mesh addressing and geometry (owner, nei, w, the three Sf components, ownerStart, losort,
losortStart, the boundary permutation, V) while the field itself is a small part of that traffic.
deviceGaussGradFused carries up to three fields through one pass, bit-identical per field by memcmp
(tests/test_grad_fused.cu, 28 arms including a one-ulp cross-contamination control on both the interior
and the boundary values, and an empty-patch fixture; fail-proof run: feeding field 0's boundary values
to all three fails 10 arms).

| gradient work per outer iteration | launches | ms/it |
|-----------------------------------|---------:|------:|
| before                            |       16 |  4.50 |
| after (momentum + turbulence memo)|  10 + 2  |  3.66 |
| after (+ the viscous stress term) |   7 + 3  |  3.31 |

One fused call costs 397 us where three separate ones cost 864 -- 2.2x, the mesh read once instead of
three times. Whole-iteration GPU-busy 30.5 -> 29.1 ms/it. The wall figure moves inside run-to-run noise
(64-67 ms/it over three runs either way), which is the honest reading: this is a 5% cut in GPU work on a
path where half the iteration is idle, so the launch overhead is what stands between it and the wall.
Seven single-field gradient launches remain, and three of the fused sites are div-scheme branches that
this case does not select -- a case naming limitedLinearV or linearUpwindV gets more of it.

## Where the iteration's time actually goes, and a measurement error worth recording (2026-09-08)

NVTX phase ranges (BRAE_PHASE_NVTX=1, the boundaries the phase timer already had) put GPU work and idle
against an equation. AND THEY MUST BE READ WITH --cuda-graph-trace=node: without it, kernels executed
inside a CUDA graph do not appear in the timeline at all, and since the pressure, energy and turbulence
solvers run in captured graphs while the momentum one runs a host loop, the first reading made the
graphed phases look 42-52% idle and the momentum phase look uniquely efficient. It was an artefact.
Corrected, per outer iteration at 306k, steady state:

| phase       | wall ms | GPU-busy ms | gaps ms | idle | GPU ops |
|-------------|--------:|------------:|--------:|-----:|--------:|
| pressure    |    21.1 |        15.8 |    4.07 |  25% |     816 |
| turbulence  |    14.2 |         9.6 |    4.18 |  32% |     354 |
| energy      |     6.8 |         4.4 |    2.27 |  36% |     170 |
| momentum    |     9.1 |         8.6 |    0.14 |   5% |     175 |
| iteration   |    51.2 |        38.4 |   ~10.5 |  25% |    1515 |

The iteration is 25% idle, not the 59% the first reading claimed. What survived the correction is the
CAUSE and the ranking: every gap over 5 microseconds is spanned by a blocking device-to-host copy --
9.5 ms per iteration over 48 of them -- and the momentum phase is at 5% because its readback is already
a mailbox (a one-thread kernel publishes into mapped host memory behind a sequence number and the host
spins on it, with a bounded fallback).

deviceReadScalar, the shared readback all eleven solver call sites use, is now that mailbox
(reductions.cu; BRAE_READ_SCALAR_SYNC=1 restores the blocking copy). Microbenchmark on a 306k reduction:
21.3 us against 269.4 with an idle queue, 205 against 522 behind 40 queued kernels. On the real case it
is worth about 2.9 ms per iteration (pressure 33.1 -> 31.4, turbulence 13.6 -> 13.0, energy 7.2 -> 6.7
over 20 iterations including the cold one). The remaining 9.5 ms of gaps are the explicit
cudaMemcpyAsync-and-synchronise pairs INSIDE the solver loops -- the BiCGStab graph path's four host
syncs per solve and the Gauss-Seidel solvers' own readbacks -- which the shared helper does not reach.
That is the next piece of work, and the momentum phase is the proof of what it is worth.

## Every solver-loop readback through the mailbox (2026-09-08)

The shared deviceReadScalar became a mailbox first, worth 2.9 ms per iteration. The rest of the idle was
in readbacks written INLINE in the solver loops -- an explicit cudaMemcpyAsync to host followed by a
cudaStreamSynchronize, twelve of them across the BiCGStab graph path, the AMG-PCG path and the
Gauss-Seidel solvers, several reading a residual AND an iteration count. They now publish a group of up
to eight values through one mailbox behind one sequence number and wait once (deviceReadValues). One
site is deliberately still blocking: the host smoother's once-per-solve sync also makes its bulk
device-to-host downloads visible to the CPU sweeps, which a mailbox wait does not guarantee.

100 iterations at 306k, the same range OpenFOAM's block table was averaged over, with and without
BRAE_READ_SCALAR_SYNC=1 (which restores the blocking copies at every site):

| block           | OF-20c | brae, blocking readbacks | brae, mailbox |
|-----------------|-------:|-------------------------:|--------------:|
| momentum        |   26.1 |                     10.1 |          10.2 |
| energy          |    9.8 |                      6.7 |           4.7 |
| pressure        |   24.0 |                     22.6 |          21.0 |
| turbulence      |   22.9 |                     16.5 |          12.5 |
| the four phases |   82.8 |                     55.9 |          48.4 |

7.5 ms per iteration, and it lands where the profile said it would: energy 30%, turbulence 24%,
pressure 7%, momentum nothing (its readback was already a mailbox). brae is now ahead of 20 cores on
every block of the compressible iteration, the pressure included, and the whole iteration is 48.4
against 82.8 -- 1.7x.

## A direct coarsest solve in the AMG V-cycle (2026-09-08)

With every readback through the mailbox the pressure phase's largest remaining GPU item was the
COARSEST-GRID SOLVE: `coarseBiCGStabKernel`, 706 us per call and three calls per outer iteration, 2.1 ms
of a 15-16 ms phase. It is expensive for a reason -- it has to iterate to COARSE_REL_TOL (1e-12) because
an unconverged coarsest level makes the V-cycle input-dependent and the outer Krylov method breaks on it
(the 187 / 1000 / 89 measurement recorded next to that constant) -- and on 64 cells it typically runs to
its 512-iteration cap trying to reach 1e-12 on a BiCGStab that has already stagnated.

A factorisation removes the question. OpenFOAM offers exactly this at the same level: `directSolveCoarsest`
builds an `LUscalarMatrix` -- a dense LU of the coarsest matrix -- instead of the PBiCGStab/PCG
(GAMGSolver.C:266-278 against :299-328). brae now does the same by default when the coarsest grid is at
most `DENSE_COARSE_MAX` cells, which with the default `BRAE_AMG_TARGET` of 64 it always is:
`amgGalerkin` factorises (dense LU, partial pivoting, one block, in shared memory) at the one point where
the coarse coefficients change and outside every graph capture, and each V-cycle pays only the two
substitutions. `BRAE_AMG_COARSE_LU=0` restores the iterative solvers.

100 iterations at 306k, BRAE_PHASE_TIME, one interleaved pair:

| block            | OF-20c | brae, iterative coarsest | brae, direct coarsest |
|------------------|-------:|-------------------------:|----------------------:|
| momentum         |   26.1 |                     10.3 |                  10.2 |
| energy           |    9.8 |                      4.8 |                   4.8 |
| pressure         |   24.0 |                     19.8 |                  17.1 |
| turbulence       |   22.9 |                     12.5 |                  12.6 |
| the four phases  |   82.8 |                     47.4 |                  44.7 |
| the p SOLVE only |      - |                     16.2 |                  13.3 |

The pressure solve is 18% cheaper. It is not a different preconditioner: the outer BiCGStab takes the
same number of iterations (2.13 against 2.17 mean over the 100), and in the unit gate ten stationary
V-cycles land on the same fine residual to seven figures (6.033179e-06 both) -- the direct solve is the
limit the iterative one was being asked to reach, at a fixed cost that no longer depends on the
right-hand side. Four interleaved end-to-end pairs: 0.367 / 0.180 / 0.267 / 0.226 s saved over the
100 iterations, always in the same direction.

Kernel cost at n = 64, from the graph-node profile: the factorisation is 114 us once per outer iteration
(246 us before the pivot search was given a warp instead of a thread, 336 us before the working matrix
moved into shared memory) and each substitution 44 us, against 3 x 706 us. What is left of the pressure
phase is the SpMV -- 106 `amulKernel` launches per iteration, 3.65 ms -- and 3.0 ms of idle.

## Which preconditioner a substituted PBiCGStab carries on k and epsilon (2026-09-08)

A case that names `solver GAMG` on the turbulence pair names no `preconditioner` -- GAMG takes none --
so brae must choose one for the PBiCGStab it substitutes. It chose `diagonal`, OpenFOAM's weakest. The
alternative is `DILU`, OpenFOAM's default for an asymmetric matrix, whose apply is a level-scheduled
sequential walk: one kernel launch per dependency level, and the level count grows with the mesh. That
is why this was measured against SIZE and against MODEL rather than argued from one case.
`bench/rhoSimpleFoam/turb_precon_scan.sh` and `bench/turb_precon_models.sh` produce both tables.

### Speed -- the compressible squareBend, turbulence block, ms per outer iteration

| cells   | DILU levels | diagonal | DILU | ratio | four phases, DILU |
|--------:|------------:|---------:|-----:|------:|------------------:|
|  24,192 |         160 |      2.0 |  8.8 | 4.40x |              14.4 |
| 112,000 |         268 |      4.1 | 16.0 | 3.90x |              29.4 |
| 307,328 |         376 |     12.5 | 33.3 | 2.66x |              67.7 |
| 896,000 |         538 |     32.2 | 61.5 | 1.91x |             183.2 |

The RELATIVE cost falls as the mesh grows -- 4.40x to 1.91x -- because the per-level launch is amortised
over more cells. At 896,000 the whole of DILU is 29.3 ms of a 183.2 ms iteration, 16%.

### Health -- nut at outer iteration 8, against real OpenFOAM on the same mesh

nut = Cmu k^2/epsilon, so it is what shows a dissipation scalar driven non-positive and floored at 1e-15
by `bound()`. In parentheses: cells whose nut is at or below 1e-14.

| cells   | brae diagonal   | OF diagonal   | brae DILU  | OF DILU    | OF GAMG (the case) |
|--------:|----------------:|--------------:|-----------:|-----------:|-------------------:|
|  24,192 |    5.90e+01 (0) |  7.99e+00 (10)| 1.278 (0)  | 1.130 (0)  |          0.921 (0) |
| 112,000 |    3.76e+01 (16)|  1.37e+01 (2) | 4.445 (0)  | 4.068 (0)  |          1.859 (0) |
| 307,328 |    8.61e+15 (58)|  2.00e+01 (4) | 3.134 (0)  | 3.021 (0)  |          1.709 (0) |
| 896,000 |    1.04e+01 (0) |  9.55e+00 (78)| 1.803 (0)  | 1.760 (0)  |          0.576 (0) |

Two things this says that one mesh could not.

**brae's arithmetic is faithful.** brae-DILU tracks OpenFOAM-DILU to 13%, 9%, 3.7% and 2.4% across the
four sizes, and brae-diagonal degrades exactly as OpenFOAM-diagonal does. There is no defect in the
solve; the preconditioner is the only variable.

**diagonal is unpredictable in BOTH codes, not merely worse.** brae explodes to 8.6e+15 at 307k and is
clean at 896k; OpenFOAM floors 4 cells at 307k and 78 at 896k. There is no monotone trend a bound could
be set against -- which is the argument for not shipping it, more than any single number here is.

### Models -- the incompressible driver, the same loose GAMG condition on the pair

The compressible mirror's CUDA arm is kEpsilon-only (it refuses kOmegaSST by name), so the model axis is
measured on the incompressible driver, which reads the same policy. Whole-run wall for 60 iterations;
these meshes are small enough that start-up is a large part of it, so read the ratio as an upper bound on
the per-iteration difference, not as it.

| case             | model        | cells  | diagonal | DILU | nut max: diagonal / DILU / OpenFOAM |
|------------------|--------------|-------:|---------:|-----:|-------------------------------------|
| pitzDailyTurb    | kEpsilon     | 12,225 |    1.37s |1.46s | 1.84e-2 / 1.71e-2 / 6.32e-3         |
| pitzDailyTurbBig | kEpsilon     | 48,900 |    2.51s |2.81s | 1.04e-2 / 9.98e-3 / 3.47e-3         |
| pitzDailySST     | kOmegaSST    | 12,225 |    0.72s |0.92s | 2.46e-3 / 2.32e-3 / 2.37e-3         |
| pitzDailyRKE     | realizableKE | 12,225 |    0.74s |0.90s | 7.50e-3 / 5.16e-3 / 7.57e-3         |
| lmFlatPlate      | kOmegaSSTLM  | 17,200 |    0.90s |1.29s | 1.80e-4 / 1.80e-4 / 1.80e-4         |

No model collapses under either preconditioner at these sizes, and no cell is floored anywhere. That is
the honest reading: the effect needs SCALE, and every fixture available per model is at or below 49,000
cells -- below where the compressible scan first sees it. The model axis therefore separates nothing,
and the mesh axis is what decides this.

## ...and what actually fills that blank: a truncated Neumann series (2026-09-08)

DILU fixes the collapse and costs a kernel launch per dependency level. The question the tables above left
open is whether a FULLY PARALLEL preconditioner reaches the same place. Two families were tried, both on
the epsilon system brae actually solves (`bench/rhoSimpleFoam/eps_precond_experiment.py` dumps it with
BRAE_STAGE_DUMP_DIR and re-solves it offline), ranked not by iteration count but by WHERE the solve stops
under OpenFOAM's own relTol -- because that iterate is what the next outer iteration inherits, and it is
what compounds. At 112k, outer iteration 6:

| preconditioner            | \|x-x*\|/\|x*\| at the stop | min(epsilon) | cost per apply     |
|---------------------------|---------------------------:|-------------:|--------------------|
| Jacobi (the failure)      |                   1.33e-02 |         73.9 | 0 SpMV             |
| red-black (multicolour) DILU |                1.18e-02 |        117.3 | 4 launches         |
| Neumann series, degree 3  |                   1.07e-02 |        169.3 | 2 SpMV             |
| Neumann series, degree 6  |                   4.62e-03 |        179.1 | 5 SpMV             |
| **Neumann series, degree 10** |               4.55e-03 |    **180.5** | **9 SpMV**         |
| DILU, natural order       |                   9.93e-03 |        182.6 | 376 launches       |

**The multicolour reordering the GPU literature points at does not work here.** It is what OPM and AmgX
do, and a two-colour DILU lands barely better than Jacobi (117.3 against 73.9, where natural-order DILU
reaches 182.6): the reordering destroys most of what makes the factorisation work.

**The polynomial does.** M^-1 = sum_{j<k} (I - D^-1 A)^j D^-1 converges iff rho(I - D^-1 A) < 1, i.e. iff
the matrix is diagonally dominant -- and the epsilon equation is strongly so, because of the Sp reaction
term OpenFOAM makes implicit. The equation that was failing is exactly the equation the series is
guaranteed on. Chebyshev matched it on error at the same degree and was REJECTED: at degree 10 it returned
min(epsilon) = -2.8e+04, because a min-max optimal polynomial is not a positive one and can undershoot
through zero, which is the precise failure being chased.

### End to end, the turbulence block, ms per outer iteration

| cells   | diagonal | series | DILU | series against diagonal | DILU against diagonal |
|--------:|---------:|-------:|-----:|------------------------:|----------------------:|
|  24,192 |      1.9 |    1.7 |  9.1 |                    -11% |                 +379% |
| 112,000 |      4.1 |    4.1 | 16.3 |                      0% |                 +298% |
| 307,328 |     12.0 |   13.4 | 33.3 |                    +12% |                 +178% |
| 896,000 |     33.2 |   41.0 | 61.7 |                    +23% |                  +86% |

At the two smaller meshes it is FREE: it needs fewer BiCGStab iterations than Jacobi, and the extra SpMVs
pay for themselves.

### ...and nut at outer iteration 8, against real OpenFOAM (cells at the floor in parentheses)

| cells   | diagonal        | series       | DILU      | OF GAMG   |
|--------:|----------------:|-------------:|----------:|----------:|
|  24,192 |    5.90e+01 (0) | **1.155** (0)| 1.278 (0) | 0.921 (0) |
| 112,000 |   3.76e+01 (16) |    5.461 (0) | 4.445 (0) | 1.859 (0) |
| 307,328 |   8.61e+15 (58) | **2.024** (0)| 3.134 (0) | 1.709 (0) |
| 896,000 |    1.04e+01 (0) | **1.048** (0)| 1.803 (0) | 0.576 (0) |

Zero cells at the floor at every size, and at three of the four it is CLOSER to OpenFOAM's own GAMG than
DILU is. It is not a compromise between diagonal and DILU; on the quantity that matters it is the better
of the two, at a fraction of DILU's cost.

`BRAE_POLY_KE=<n>` sets the degree (1 = the bare diagonal, the fail-proof arm of
tests/turb_precon_vs_openfoam.sh); `BRAE_DILU_KE=1` selects DILU instead. A case that NAMES a
preconditioner keeps it. nuTilda takes the same substitution and is deliberately NOT wired: the
Spalart-Allmaras branch never sets the degree, and it has not been measured.
