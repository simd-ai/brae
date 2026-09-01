<p align="center">
  <img src="docs/images/brae-banner.png" alt="brae" width="100%">
</p>


<p align="center"><b>GPU-native computational fluid dynamics, OpenFOAM-compatible, fully resident on the GPU.</b></p>

<p align="center">
  <img alt="License AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-blue">
  <img alt="CUDA 12.4+" src="https://img.shields.io/badge/CUDA-12.4%2B-76B900">
  <img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-00599C">
  <img alt="GPU Ampere to Blackwell" src="https://img.shields.io/badge/GPU-Ampere%20%7C%20Ada%20%7C%20Hopper%20%7C%20Blackwell-76B900">
  <img alt="OpenFOAM v2412" src="https://img.shields.io/badge/OpenFOAM-v2412-brightgreen">
  <img alt="ctest 261/261" src="https://img.shields.io/badge/ctest-261%2F261-brightgreen">
</p>

Brae keeps the whole CFD solve on one GPU. The mesh, the fields, and every linear solve stay on the device from the
first iteration to the last, with no per-iteration copies to the CPU. Point it at an existing OpenFOAM case and it
writes standard OpenFOAM results, validated cell-by-cell, so it drops into your workflow unchanged.

> **Up to 30x faster than OpenFOAM's own GPU acceleration** (AMGX, PETSc) on an H100, because brae keeps the *whole*
> SIMPLE loop on the device instead of offloading only the linear solve.

---

## ⚡ How fast

On one **H100**, at matched accuracy (under 1% on the fields):

- **26-30x faster** than OpenFOAM's own GPU offloads (AMGX, PETSc)
- **3.9x faster** than the [SPUMA](https://gitlab-hpc.cineca.it/exafoam/spuma) OpenFOAM-GPU port
- **2.5x faster** than a 24-core CPU node

![Solver runtime, brae vs OpenFOAM-CPU, AMGX, PETSc, and SPUMA on a single NVIDIA GB10, log scale, lower is better](bench/results/solver_runtime_comparison.png)

*The chart is GB10, the conservative baseline: with no HBM its GPU shares the CPU's memory, so there it only reaches
about 5x over the offloads and parity with the CPU. On the H100's HBM the same code widens to 26-30x. Not
bit-identical to OpenFOAM (the GPU reorders the floating-point sums), and not meant to be. Full method and numbers:
[docs/performance.md](docs/performance.md) and the [H100 report](bench/H100/H100_GPU_COMPARISON_REPORT.md).*

---

## 🎯 Same case, same answer

motorBike, 2.9M cells, k-omega SST. brae, OpenFOAM on the CPU, its AMGX and PETSc GPU offloads, and the SPUMA port
produce a visually identical surface-pressure field and agree to ~1.6% on drag.

| brae (Blackwell GPU) | OpenFOAM (Grace CPU) |
|:---:|:---:|
| ![motorBike surface pressure, brae on a Blackwell GPU](bench/results/motorbike_p_brae.png) | ![motorBike surface pressure, OpenFOAM on Grace CPU cores](bench/results/motorbike_p_of.png) |
| **OpenFOAM + AMGX (GPU)** | **OpenFOAM + PETSc (GPU)** |
| ![motorBike surface pressure, OpenFOAM with the AMGX GPU solver](bench/results/motorbike_p_amgx.png) | ![motorBike surface pressure, OpenFOAM with the PETSc GPU solver](bench/results/motorbike_p_petsc.png) |
| **SPUMA (OpenFOAM-GPU port)** |  |
| ![motorBike surface pressure, the SPUMA OpenFOAM-GPU port](bench/results/motorbike_p_spuma.png) |  |

See the [full five-way comparison](bench/results/motorbike_comparison.md) for the drag and lift numbers.

---

## 💡 Why brae

Most "OpenFOAM on GPU" approaches offload only the linear solver: the matrix is rebuilt on the CPU and copied to the
GPU every iteration, and assembly, momentum, and turbulence still run on one CPU core. Brae is **device-resident**,
the entire loop lives on the GPU, so there is no migration tax and no serial-CPU ceiling.

- **Drop-in**, reads your existing `0/ constant/ system/` case (ASCII or binary mesh) and writes standard time
  directories for ParaView / `postProcess`.
- **Faithful**, a clean-room reimplementation of OpenFOAM v2412, validated cell-by-cell (sub-1% on the fields).
- **Fast where it counts**, it pulls ahead as the mesh grows, and the lead scales with the GPU's memory bandwidth.

See [docs/memory-model.md](docs/memory-model.md) for the data-layout rationale (device pool over pinned/`thrust`,
LDU-gather over CSR/cuSPARSE).

---

## 📦 Install

Needs an NVIDIA GPU (Ampere or newer, including H100 / GH200 / B200), CUDA 12.4+ (13.x recommended), and a C++17
toolchain. Brae is standard CUDA, so a newer architecture is just a recompile.

```bash
# deps: cmake >= 3.24, CUDA toolkit, an MPI (OpenMPI), SCOTCH, zlib
git clone https://github.com/simd-ai/brae.git
cd brae
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<your_arch>
cmake --build build -j --target brae
```

<details>
<summary>Set <code>&lt;your_arch&gt;</code> to your GPU's compute capability</summary>

| GPU | `<your_arch>` |
|---|---:|
| GB10 | 121 |
| RTX 50-series | 120 |
| GB300 / B300 | 103 |
| B200 / GB200 | 100 |
| H100 / GH200 | 90 |
| RTX 40-series / L40 | 89 |
| RTX 30-series | 86 |
| A100 | 80 |

</details>

---

## 🚀 Get started

Run brae from inside any OpenFOAM case, exactly as you would run `simpleFoam` or `pimpleFoam` itself:

```bash
cd yourCase                       # your OpenFOAM case (0/  constant/  system/)
brae                              # solve in the current directory (steady or transient)
brae -case /path/to/yourCase      # or run it from anywhere
brae -partition -case yourCase    # optional: cache the mesh + AMG once, then later runs start warm
brae --help                       # all options
```

No `decomposePar`, brae auto-partitions for the GPU. The [fast path](docs/performance.md) (device-resident solver +
mixed-precision multigrid) is on by default; opt out with `BRAE_PCG_DEVICE=0 BRAE_AMG_FP32=0`.

### Run several cases across GPUs

Run a mesh-independence study or a parameter sweep, one case per GPU (extras queue as GPUs free up):

```bash
brae -cases mesh_coarse mesh_medium mesh_fine   # one case per GPU
BRAE_JOBS=2 brae -cases caseA caseB caseC       # cap how many run at once
```

Each case's residual output is tagged `[GPUn case]`, and it ends with a per-case summary. On a single GPU the cases
run back to back. Override the detected GPU count with `BRAE_GPUS`.

---

## 🌊 Solvers

Brae implements OpenFOAM's solvers one at a time, each fully device-resident and validated cell-by-cell:

- [`simpleFoam`](docs/solvers/simplefoam.md) — steady incompressible
- [`pimpleFoam`](docs/solvers/pimplefoam.md) — transient incompressible — **brae 4.5× faster than SPUMA**

You always type `brae`. It reads the `application` entry your case already has in `controlDict` and runs the
matching solver, so a transient case needs no different command. A case asking for a solver brae does not have
stops at start-up, named, rather than being solved with the wrong one.

Coming soon: compressible solver.

---

## 🚧 Roadmap

See [the roadmap](docs/roadmap.md) for the current scope and what is coming next.

---

## 📚 Documentation

- [Getting started](docs/getting-started.md), install, first run, verifying against OpenFOAM
- [Performance & tuning](docs/performance.md), the `BRAE_*` knobs and benchmarks
- [Memory model](docs/memory-model.md), device pool vs pinned, LDU-gather vs CSR
- [Roadmap](docs/roadmap.md), scope and what is coming next
