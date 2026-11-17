# Performance & tuning

## The one idea: residency

Most GPU approaches offload only the linear solve. The matrix is rebuilt on the CPU and copied to the GPU every
iteration, so assembly, momentum and turbulence stay on one CPU core.

| | where the work runs |
|---|---|
| OpenFOAM + AMGX / PETSc | assembly on CPU, linear solve on GPU, **matrix copied every iteration** |
| SPUMA | GPU fork on unified memory — full residency, no per-iteration copy |
| **brae** | **mesh, fields and every solve stay on the device from first iteration to last** |

## The fast path (on by default)

Two optimisations, both **accuracy-preserving** and **on by default**. Opt out by setting either to `0`.

| flag (default on) | what it does | accuracy |
|---|---|---|
| `BRAE_PCG_DEVICE` | keeps the pressure PCG's Krylov loop resident on the GPU (removes a per-iteration CPU sync) | bit-identical to the host path at iteration 1; same converged state |
| `BRAE_AMG_FP32` | runs the bandwidth-bound multigrid preconditioner in FP32 (half the bytes); the outer solve stays FP64 | bit-identical (mixed precision is confined to the preconditioner) |

Other tuning knobs (advanced; sensible defaults):

| flag | effect |
|---|---|
| `BRAE_AMG_GS=1` | multicolor Gauss-Seidel smoother (vs the default Jacobi), fewer cycles on some anisotropic meshes |
| `BRAE_AMG_TSGS=1` | two-stage Gauss-Seidel (parallel polynomial) smoother, coloring-free |
| `BRAE_AMG_SA=1` | smoothed-aggregation multigrid (vs default pairwise), stronger on cyclic / graded meshes |
| `BRAE_AMG_FUSE_BELOW=<cells>` | the single-precision V-cycle runs every grid of that many cells or fewer (default 1024, the best of a sweep on an RTX 5090; `0` switches it off), and the coarsest solve, as ONE one-block kernel instead of ~7 dependent launches a grid. Bit-identical to the unfused cycle (`tests/test_amg_fused_coarse.cu`); applies to the weighted-Jacobi cycle with no coupled pair, on the default inverse or the LU coarsest solve. Set it before the run: a captured graph keeps the setting it was captured with |
| `BRAE_GS_DEVICE=1` | device-resident k/ε Gauss-Seidel scalar solve |

## interFoam on one GPU: what stays on the device

`interFoam`'s step used to wait on the host in several places: boundary patches evaluated on the host from whole-field
downloads, balance tests and continuity sums read back after every pressure corrector, and a libm call per contact-angle
wall face. Where the case allows it, these now run on the device or are queued and read once per step. Every change is
checked byte for byte against the old path on the G4 tank case (233 steps: fields, `postProcessing`, `uniform/`).

| what | when it applies | A/B switch that restores the old path |
|---|---|---|
| `adjustPhi` sums the internal flux on the device; with only fixed-value outflows its balance test is queued and read at the end of the step (same message and numbers if it fails, one step late) | `p_rgh` needs a reference | `BRAE_CONTROL_ADJUSTPHI_HOST_SUM=1`, `BRAE_CONTROL_ADJUSTPHI_SYNC=1` |
| continuity-error sums queued on the device, accumulated after the step | a writer exists | `BRAE_CONTROL_CONTERR_SYNC=1` |
| the flux push to the patches is skipped | no patch of `U`, `p_rgh` or alpha reads the flux (`noSlip`, `fixedFluxPressure`, contact-angle walls) | `BRAE_CONTROL_FLUX_PUSH_ALWAYS=1` |
| `p_rgh` patches made on the device (gradient, laplacian coefficients, values); evaluated from the cells under their faces otherwise | every patch `fixedFluxPressure` over `noSlip` `U`, static mesh; the face-cell form needs no coupled patch and no momentum predictor | `BRAE_CONTROL_PRESSURE_PATCHES_HOST=1`, `BRAE_CONTROL_PRESSURE_DOWN_WHOLE=1` |
| the interface forces keep alpha, K and rho on the device (mixture nu and `snGrad(rho)` made there) | contact angle on the device, laminar, no momentum predictor, no coupled or empty patch | `BRAE_CONTROL_FORCES_FIELDS_HOST=1` |
| the contact angle's `cos(acos(a12) - theta)` computed on the device, bit for bit as the host's glibc gives it | contact angle on the device | `BRAE_CONTROL_CA_B2_HOST=1` (host loop, with a cache of unchanged faces), `BRAE_CONTROL_CA_B2_FULL=1` (whole array on the host) |

**The device `acos`/`cos`.** CUDA's own `acos` and `cos` differ from glibc's in the last bit on 13–20% of arguments,
which moved the wall gradient of the tank case by an ulp. `src/transportModels/interfaceProperties/` therefore holds a
port of glibc's algorithms and tables (LGPL-2.1-or-later, notices kept inline), plus a correctly rounded `acos` for
glibc 2.33 and older. glibc has an FMA and a plain build of both and picks one at run time, so brae tests four device
variants against the host's libm on a few million values at the first contact-angle step and uses the one that agrees on
every bit; the log says which (`contact angle: b2 = cos(acos(a12) - theta) on the GPU with ...`). If none agrees (another
architecture, an unusual libm) it says so and keeps the host loop. Measured against glibc 2.43 on 1e9 samples: 0
mismatches for the FMA port; glibc 2.34-2.43 and the older generation were checked with the matching variant.

**Measured** (G4 tank, 2.48M cells, laptop RTX 5090, one GPU; the run to t = 0.008, 89 steps, against the same run to
t = 0.002, 28 steps, so start-up cancels): **0.70 s/step before, 0.24 s/step after (3.0x)** against upstream `main` at `df3cd8cb`; the whole 89-step run
takes 69.8 s before and 29.8 s after (two runs each, alternating, within 1 s of each other), and every field of the `End`
line is the same to the digit. Earlier, with the per-phase timer on and the contact-angle change last: 530 ms/step before, 225–241 ms
after, outputs byte-identical over 233 steps. With `nsys` graph tracing the GPU is active 97.5% of a steady-state window
(88.5% before the contact-angle change). A one-step Nsight Compute profile of the 3,400 kernels gives 68% of peak DRAM
throughput, 70% of kernel time in kernels at 60% or more, and at most 1.15× left if every kernel reached 90% of its own
roofline. These are one laptop and one case; the gain on other GPUs depends on the host and was not measured there.

The one numerical difference: `adjustPhi`'s `totalFlux` is summed on the device (a different order from the host's loop,
so it differs in its last bits). It feeds only the 1e-15 and 1e-8 relative thresholds and the error text.

## Benchmarks

| | |
|---|---|
| **measured** | total wall for 100 SIMPLE iterations, scaled-pitzDaily |
| **hardware** | one NVIDIA GB10 — 20 Grace cores + one Blackwell GPU, unified LPDDR5x, **no HBM** |
| **excluded, both sides** | one-time prep: `decomposePar` for OpenFOAM, `brae -partition` for brae (caches mesh + AMG, see [Run it](../README.md#-run-it)) |

| cells | **brae** (Blackwell GPU) | OpenFOAM (20 Grace cores) | OpenFOAM+AMGX (same GPU) | OpenFOAM+PETSc-GPU (same GPU) | SPUMA (same GPU) |
|---|---:|---:|---:|---:|---:|
| 990k | 21.6 s | 15.9 s | 80.0 s | 89.3 s | 92.4 s |
| 4.89M | 107.7 s | 107.0 s | 572.5 s | 601.6 s | 623.2 s |
| 10.28M | 232.5 s | 228.8 s | 1174.2 s | 1253.0 s | 1056.3 s |
| 14.97M | 339.6 s | 354.4 s | 1696.4 s | 1790.3 s | 1359.3 s |
| 35.6M | 877.1 s | 990.4 s | 4111.0 s | - | —¹ |

SPUMA is the [CINECA / EU-exaFOAM OpenFOAM-GPU port](https://gitlab-hpc.cineca.it/exafoam/spuma) (a full GPU fork
of OpenFOAM on unified memory), the closest full-residency peer to brae. ¹ SPUMA diverged to NaN on the extreme
35.6M scaled mesh at stock solver settings.

**Reading it:**

- **~4-5× faster than every other GPU approach** on the *same* GPU, OpenFOAM's own AMGX/PETSc offloads **and** the
  SPUMA port, at every size, the residency payoff.
- **SPUMA is a full-residency port, not an offload** (a GPU fork of OpenFOAM on unified memory): at ≤5M it is
  roughly tied with AMGX/PETSc, but from ~10M up it *overtakes* them (1056 s vs 1174/1253 at 10.28M) because, like
  brae, it avoids the offloads' per-iteration matrix rebuild-and-copy. It still stays ~4-4.5× behind brae, and its
  GPU sits only ~7-8% utilized at these sizes (its design point is tens of millions of cells per GPU).
- **Parity-to-ahead of a 20-core Grace CPU node**, and the lead grows with mesh size (0.74× at 990k → 1.13× at
  35.6M). Small meshes under-utilize the GPU; large meshes saturate it.

## Why GB10 is a conservative baseline

| | |
|---|---|
| GB10 has **no HBM** | its GPU shares the same ~273 GB/s unified memory as the 20 Grace cores |
| so the GPU **cannot** structurally out-run the CPU there | for a bandwidth-bound solver, parity *is* the ceiling |
| brae clears it at scale anyway | on the most bandwidth-constrained hardware a GPU port can run on |
| on HBM the usable bandwidth is ~10× higher | H100 ~3.3 TB/s, GH200 ~4 TB/s — measured below |

## GH200: the same code on HBM

Same source, **rhoSimpleFoam**, 100 fixed SIMPLE iterations, solver wall only, one GH200 against all 64 Grace cores:

| case | cells | brae | OpenFOAM 64c | SPUMA | vs OF | vs SPUMA |
|---|---:|---:|---:|---:|---:|---:|
| aerofoilNACA0012 | 10,000,000 | 108.2 s | 916.7 s | — | **8.47×** | — |
| aerofoilNACA0012 | 1,024,000 | 11.1 s | 44.5 s | 1495.2 s | 4.00× | **135×** |
| squareBendLiq | 896,000 | 6.8 s | 8.4 s | 92.2 s | 1.24× | **13.6×** |
| squareBend | 112,000 | 1.6 s | 1.8 s | — | 1.14× | — |
| squareBendLiq | 112,000 | 1.7 s | 1.6 s | 22.0 s | 0.92× | **12.9×** |

SPUMA is the [CINECA / EU-exaFOAM OpenFOAM-GPU port](https://gitlab-hpc.cineca.it/exafoam/spuma), the closest
full-residency peer. Two rows are blank because SPUMA was not run at that mesh size, not because it failed.

![rhoSimpleFoam throughput against mesh size, one GH200 versus 64 Grace cores, log-log](../bench/results/rhoSimpleFoam/brae_benchmark_scaling.png)

| | |
|---|---|
| **8.47×** at 10M cells | inside the 8-12× predicted above from the bandwidth ratio alone |
| brae plateaus at **9.2 M cell-iterations/s** | from ~1M cells up |
| **OpenFOAM turns down** past 1M (2.3 -> 1.1) | the widening gap is OpenFOAM losing throughput, not brae gaining it |
| `0.92x` row kept on purpose | below ~10^5 cells a GH200 is not the right tool |
| sm_121 and sm_90 print every digit identical | squareBend, 10 iterations |

### All six tutorials, five codes

![Wall time for 100 SIMPLE iterations across the six rhoSimpleFoam tutorials: brae, brae warm, OpenFOAM on 64 cores, AMGX, PETSc and SPUMA, log scale](../bench/results/rhoSimpleFoam/brae_benchmark_arms.png)

| | |
|---|---|
| brae and brae-warm | left of everything, at every size |
| OpenFOAM 64c | the only CPU arm, and the only one close |
| AMGX / PETSc | the offloads — clustered 400-1600 s, the per-iteration matrix rebuild |
| SPUMA | full residency like brae, but 10-100x behind it here |
| timing asymmetry | **brae's wall includes its start-up; OpenFOAM's excludes `decomposePar`** |

Per-case tables: [rhoSimpleFoam on a GH200](../bench/results/rhoSimpleFoam/rhoSimpleFoam_gh200.md).

## Accuracy is not traded for speed

| | |
|---|---|
| fast path vs reference path | identical iteration count, **U 0.016% / p 0.11%** (turbulent pitzDaily) |
| FP32 in the preconditioner | bit-identical — mixed precision never touches the outer double-precision solve |
| brae run to run | **byte-identical** — deterministic reductions, gated by `rho_run_to_run_identity` |
| brae vs OpenFOAM, operator level | **1e-10 to 1e-13** — the GPU sums in a different order |
| brae vs OpenFOAM, 100 iterations | sub-1% — the above compounding through the nonlinear iteration |

Why not bit-identical: [getting-started.md](getting-started.md#why-the-results-are-not-bit-identical).
