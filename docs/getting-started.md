# Getting started

Brae runs an existing OpenFOAM incompressible case on one GPU. If you can run it with `simpleFoam` (steady) or
`pimpleFoam` (transient), you can usually run it with `brae`.

## 1. Prerequisites

- An NVIDIA GPU, Ampere or newer: Ampere (A100, RTX 30-series), Ada Lovelace (RTX 40-series, L40), Hopper (H100,
  GH200), or Blackwell (GB10, B200, RTX 50-series). Brae is standard CUDA, so newer cards build from the same source.
  Volta (V100) also builds, with CUDA 12.x — see [GPU and toolchain compatibility](#gpu-and-toolchain-compatibility).
- CUDA toolkit 12.4+ (13.x recommended). Conditional graph nodes, which the device-resident solver loops use, need
  CUDA 12.4 and a driver r550 or newer.
- `cmake ≥ 3.24`, a C++17 compiler, an MPI (OpenMPI), SCOTCH, zlib.

## 2. Build

```bash
git clone https://github.com/simd-ai/brae.git
cd brae
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<arch>   # 121=GB10, 120=RTX 50xx, 100=B200, 90=H100/GH200, 89=RTX 40xx, 86=RTX 30xx, 80=A100, 70=V100 (CUDA 12.x)
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

## GPU and toolchain compatibility

| GPU | arch | status |
|---|---:|---|
| Blackwell (GB10, RTX 50-series, B200, B300) | 121 / 120 / 100 / 103 | the development targets; the full test suite runs on a GB10 |
| Hopper (H100, GH200) | 90 | supported; measured in [performance](performance.md) |
| Ada, Ampere (RTX 40/30, L40, A100) | 89 / 86 / 80 | built by CI for sm_80 and sm_90; same source |
| Volta (V100) | 70 | builds with CUDA 12.x (compiled for `70` and for `"70;90;120"` with CUDA 12.9); a 2.48M-cell interFoam case runs on 1-8 V100s ([Multi-GPU](multi-gpu.md#performance)). Not part of CI |

Nothing in brae needs sm_80 or newer: the thread-block-cluster reductions (sm_90+) compile to no-ops below Hopper and are
never launched there.

Toolchain notes:

- **CUDA 12.4 and newer build the device-resident solver loops.** The CUDA-graph `WHILE` loops of the AMG-PCG, the
  symmetric Gauss-Seidel and BiCGStab solvers use conditional graph nodes, which exist from CUDA 12.4. They used to be
  compiled in only on CUDA 13, so a CUDA 12 build silently ran host-driven loops: the same algorithms, but not
  bit-identical to the graph path. `src/cuda/cuda_graph_compat.cuh` absorbs the CUDA 12 / 13 signature difference of
  `cudaGraphAddNode` and `cudaStreamGetCaptureInfo`. At run time these loops need a driver r550 or newer (the first
  conditional-node call fails otherwise; there is no silent fallback). The run-time switches `BRAE_PCG_DEVICE=0`,
  `BRAE_GS_HOST_LOOP=1` and `BRAE_BICG_HOST_LOOP=1` select the host-driven loops again.
- **Host compiler.** nvcc 12.9 supports gcc up to 14; gcc 15's libstdc++ does not parse with it. Configure with
  `-DCMAKE_CUDA_HOST_COMPILER=g++-14` (or any gcc 11-14). With CUDA 13, pass a host compiler that version supports
  if the system default is newer.
- **CUDA 13 cannot target V100** (Volta is dropped); use CUDA 12.x for `70`.
- **One binary for several GPUs:** `-DCMAKE_CUDA_ARCHITECTURES="70;90;120"` (a fat build takes about 3 minutes longer).
- **Run-time libraries:** the toolkit's `libnvrtc` must be found; if the toolkit is not installed system-wide, export
  `LD_LIBRARY_PATH=$CUDA_HOME/lib64`.
- **NVSHMEM** was removed: the multi-GPU code uses ProcComm (MPI-host or NCCL); no NVSHMEM installation or CMake option remains
  (a DGX-1 has no NVSwitch, and its unlinked GPU pairs cannot use it). See [Multi-GPU](multi-gpu.md).

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
