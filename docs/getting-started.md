# Getting started

Brae runs an existing OpenFOAM incompressible case on one GPU. If you can run it with `simpleFoam` (steady) or
`pimpleFoam` (transient), you can usually run it with `brae`.

## 1. Prerequisites

- An NVIDIA GPU, Ampere or newer: Ampere (A100, RTX 30-series), Ada Lovelace (RTX 40-series, L40), Hopper (H100,
  GH200), or Blackwell (GB10, B200, RTX 50-series). Brae is standard CUDA, so newer cards build from the same source.
- CUDA toolkit 12.4+ (13.x recommended).
- `cmake ≥ 3.24`, a C++17 compiler, an MPI (OpenMPI), SCOTCH, zlib.

## 2. Build

```bash
git clone https://github.com/simd-ai/brae.git
cd brae
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<arch>   # 121=GB10, 120=RTX 50xx, 100=B200, 90=H100/GH200, 89=RTX 40xx, 86=RTX 30xx, 80=A100
cmake --build build -j 4 --target brae
```

Keep the `-j` bounded. A bare `-j` starts all of brae's ~210 compile jobs at once, each needing up to about 1 GB of
memory (0.43 GB on median), about 89 GB in all: the kernel then kills a compiler and the build stops with only
`make: *** ... Error 2`. Allow about 2 GB of free memory a job: `-j 4` on a 16 GB machine, `-j 8` on 32 GB.

The command is `build/brae`, and it is the only one you run — steady or transient. Add it to your `PATH` if you
like:

```bash
export PATH="$PWD/build:$PATH"
```

## 3. Run a case

Any standard OpenFOAM case directory works, it needs `0/` (or `0.orig/`), `constant/` (with `polyMesh` and
`turbulenceProperties`), and `system/` (`controlDict`, `fvSchemes`, `fvSolution`):

```bash
brae -case /path/to/yourCase
```

What happens:

1. Brae reads your `controlDict` `application` entry and picks the matching solver (`simpleFoam` -> steady,
   `pimpleFoam` -> transient). A solver it does not have yet stops the run instead of being substituted.
2. It reads your mesh, fields, schemes, and solver settings.
3. It **auto-partitions** the mesh and builds its multigrid hierarchy for your GPU (the `decomposePar` analogue ,
   done once, cached). You can pre-build this cache with `brae -case yourCase -partition`.
4. It runs the solver loop fully on the GPU until `endTime` (or, steady, `residualControl` convergence).
5. It writes standard time directories (`100/U`, `100/p`, …) you open with `paraFoam` or `postProcess`.

## 4. The fast path (on by default)

Two accuracy-preserving optimisations give the full device-resident performance, and both are **on by default**:

- `BRAE_PCG_DEVICE`, keeps the pressure solver's Krylov loop on the GPU (no per-iteration CPU sync).
- `BRAE_AMG_FP32`, runs the multigrid preconditioner in single precision (half the memory traffic; the answer stays
  double precision).

Both are validated to preserve the result (see [performance.md](performance.md)). To fall back to the reference
double-precision, host-driven path, set either to 0:

```bash
BRAE_PCG_DEVICE=0 BRAE_AMG_FP32=0 brae -case yourCase
```

## 5. Verify against OpenFOAM (optional)

Because brae writes standard OpenFOAM fields, you can compare directly. Run the same case with `simpleFoam` and diff
the final fields, or compare force coefficients:

```bash
simpleFoam -case yourCase           # OpenFOAM
brae      -case yourCase_brae       # brae (a copy)
# then compare yourCase/<t>/U vs yourCase_brae/<t>/U
```

On matched schemes, brae agrees with OpenFOAM to under 1% on the fields, and to sub-1% on lift on validated
external-aero cases.

### Why the results are not bit-identical

Brae solves the *same* equations as OpenFOAM but will not match to the last digit, and is not meant to.
Floating-point addition is not associative, and brae runs the arithmetic in parallel on the GPU: it loops over cells
rather than faces and combines partial sums in whatever order the hardware schedules. Same operations, different
order, so the rounding (the truncation error) differs at machine precision and can vary slightly run to run. Carried
through the nonlinear SIMPLE iterations, that shows up as the sub-1% field difference. It is a reshuffling of
rounding error, not a change in the physics, and it is expected of any GPU reimplementation of a CPU solver.

## Troubleshooting

- **"unsupported RASModel '…'"**, brae supports laminar, kEpsilon, realizableKE, kOmegaSST, kOmegaSSTLM,
  SpalartAllmaras. See [solvers/simplefoam.md](solvers/simplefoam.md).
- **"unsupported BC type '…'"**, the boundary condition isn't implemented yet; the error names it.
- **Out of memory**, the mesh is larger than your GPU's memory. Brae is single-GPU (see
  [roadmap.md](roadmap.md)).
- Brae fails *clearly* on anything it doesn't support (it names the model / BC and stops) rather than producing
  wrong results.
