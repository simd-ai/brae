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
