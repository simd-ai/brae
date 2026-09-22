<p align="center">
  <img src="docs/images/brae-banner.png" alt="brae" width="100%">
</p>


<p align="center"><b>OpenFOAM's solvers, re-ported to CUDA. The whole solve stays on one GPU.</b></p>

<p align="center">
  <img alt="License AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-blue">
  <img alt="CUDA 12.4+" src="https://img.shields.io/badge/CUDA-12.4%2B-76B900">
  <img alt="GPU Ampere to Blackwell" src="https://img.shields.io/badge/GPU-Ampere%20%7C%20Ada%20%7C%20Hopper%20%7C%20Blackwell-76B900">
  <img alt="OpenFOAM v2412" src="https://img.shields.io/badge/OpenFOAM-v2412-brightgreen">
  <img alt="478 validation gates" src="https://img.shields.io/badge/validation%20gates-478-brightgreen">
</p>

```bash
curl -fsSL https://brae.sh/install.sh | sh
cd yourCase && brae
```

Your existing OpenFOAM case. Standard OpenFOAM output. No `decomposePar`, no OpenFOAM install required.


---

## ⚡ Speed

![rhoSimpleFoam throughput against mesh size, brae on one GH200 versus OpenFOAM on 64 Grace cores, log-log](bench/results/rhoSimpleFoam/brae_benchmark_scaling.png)

brae plateaus at **9.2 M cell-iterations/s**; OpenFOAM's curve *turns down* past 1M cells at the memory-bandwidth wall.

**One GH200 against all 64 Grace cores**, 100 SIMPLE iterations, solver wall only:

| case | cells | brae | OpenFOAM 64c | ratio |
|---|---:|---:|---:|---:|
| aerofoilNACA0012 | 10,000,000 | 108.2 s | 916.7 s | **8.47×** |
| aerofoilNACA0012 | 1,024,000 | 11.1 s | 44.5 s | 4.00× |
| squareBendLiq | 896,000 | 6.8 s | 8.4 s | 1.24× |
| squareBend | 112,000 | 1.6 s | 1.8 s | 1.14× |
| squareBendLiq | 112,000 | 1.7 s | 1.6 s | **0.92×** |

Against the closest full-residency peer, the [SPUMA](https://gitlab-hpc.cineca.it/exafoam/spuma) OpenFOAM-GPU
port, on the same GH200: **~13×** (squareBendLiq, 6.8 s vs 92.2 s at 896k; 1.7 s vs 22.0 s at 112k). All six
tutorials against OpenFOAM, AMGX, PETSc and SPUMA: [benchmarks →](docs/performance.md)

Below ~10⁵ cells a GH200 is not the right tool. The advantage is a scaling one and arrives around a million cells.

simpleFoam on an H100 reaches **26–30×** over OpenFOAM's AMGX and PETSc GPU offloads:
[benchmarks →](docs/performance.md)

---

## 🎯 Error against OpenFOAM

All six rhoSimpleFoam tutorials, **as shipped**, against real OpenFOAM iteration by iteration. Worst field, relative:

| tutorial | cells | host arm | CUDA arm |
|---|---:|---:|---:|
| aerofoilNACA0012 | 16,000 | 2.8e-12 | 2.8e-12 |
| angledDuctExplicitFixedCoeff | 28,000 | 1.6e-10 | 1.2e-10 |
| squareBend (transonic) | 112,000 | 5.5e-10 | 6.6e-11 |
| squareBendLiq | 112,000 | 1.9e-12 | 1.9e-12 |
| squareBendLiqNoNewtonian | 112,000 | 5.8e-13 | 5.5e-13 |

Same state, one iteration — this measures the port. Run both codes 100 iterations from the same start and small
differences compound: squareBend reads ~8e-03 on p at iteration 100. Two nearby trajectories, same fixed point.

**478 validation gates.** Every right-hand side is OpenFOAM's own output, never a hand-computed expectation.

---

## 🖼 motorBike, 2.9M cells, k-omega SST

| brae (Blackwell GPU) | OpenFOAM (Grace CPU) |
|:---:|:---:|
| ![motorBike surface pressure, brae on a Blackwell GPU](bench/results/simpleFoam/motorbike_p_brae.png) | ![motorBike surface pressure, OpenFOAM on Grace CPU cores](bench/results/simpleFoam/motorbike_p_of.png) |

Surface pressure, same colour scale. Drag agrees to ~1.6% — with OpenFOAM's AMGX and PETSc GPU offloads and the
SPUMA port too: [full five-way comparison →](bench/results/simpleFoam/motorbike_comparison.md)

---

## 🌊 Will it run my case?

Type `brae`. It reads the `application` entry in your `controlDict` and runs the matching solver.

| solver | |
|---|---|
| [`simpleFoam`](docs/solvers/simplefoam.md) | steady incompressible |
| [`pimpleFoam`](docs/solvers/pimplefoam.md) | transient incompressible — URANS / DES / LES |
| [`rhoSimpleFoam`](docs/solvers/rhosimplefoam.md) | steady compressible, subsonic and transonic |

<details>
<summary><b>Full support matrix</b> — turbulence, thermo, schemes, 25+ boundary conditions, fvOptions</summary>

<br>

| | supported |
|---|---|
| **turbulence** | kEpsilon · realizableKE · kOmegaSST · kOmegaSSTLM · SpalartAllmaras · Smagorinsky · WALE · SA-DDES/IDDES · kOmegaSST-DDES/IDDES · laminar · generalizedNewtonian |
| **thermo** | hePsiThermo · heRhoThermo · perfectGas · hConst · sutherland · const · liquid (NSRDS correlations, OpenFOAM's own `he→T` inversion) |
| **convection** | upwind · linearUpwind · linearUpwindV · LUST · linear · limitedLinear · limitedLinearV · vanAlbada |
| **pressure–velocity** | SIMPLE · SIMPLEC (`consistent yes`) · PIMPLE · transonic |
| **time** | steadyState · Euler · backward · CrankNicolson |
| **fvOptions** | explicitPorositySource · limitTemperature · fixedTemperatureConstraint · scalarFixedValueConstraint · MRF |

**Boundary conditions**

| | |
|---|---|
| basic | fixedValue · zeroGradient · noSlip · calculated |
| geometric | slip · symmetry · symmetryPlane · wedge · empty |
| coupled | cyclic · cyclicAMI · processor |
| inlet / outlet | inletOutlet · outletInlet · flowRateInletVelocity · surfaceNormalFixedValue · freestream · freestreamVelocity |
| pressure | totalPressure · fixedFluxPressure · pressureInletOutletVelocity |
| time-varying | uniformFixedValue · codedFixedValue · fixedMean |
| turbulence | nutkWallFunction · epsilonWallFunction · omegaWallFunction · kqRWallFunction · turbulentIntensityKineticEnergyInlet · turbulentMixingLengthDissipationRateInlet |
| thermal | externalWallHeatFluxTemperature · limitTemperature |

</details>

Coming soon: `interFoam` (two-phase VoF).

---

## 🛑 It tells you when it can't

Anything outside that list **stops at start-up, named**. Anything brae does differently says so on stderr:

| prefix | meaning |
|---|---|
| `[ignored]` | your case asked for it; brae does nothing with it |
| `[approximated]` | related but not equal — a scheme downgrade or formula substitution |
| `[defaulted]` | could not read a value, fell back |
| `[unread]` | dictionary entries brae never consulted |
| **refusal** | brae stops rather than answer wrongly |

> A notice is for *"less than asked, but still a defensible answer"*; a refusal is for *"this answer would be wrong"*.
> When in doubt, throw — brae's contract is that it never guesses.

---

## 📦 Install

```bash
curl -fsSL https://brae.sh/install.sh | sh
```

NVIDIA GPU (Ampere or newer), CUDA 12.4+ (13.x recommended), C++17.

<details>
<summary>Build from source</summary>

```bash
# deps: cmake >= 3.24, CUDA toolkit, an MPI (OpenMPI), SCOTCH, zlib
git clone https://github.com/simd-ai/brae.git
cd brae
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<your_arch>
cmake --build build -j --target brae brae_pimpleFoam brae_rhoSimpleFoam
```

`brae` hands each case to the solver its `application` entry names, so build all three side by side.

| GPU | `<your_arch>` | GPU | `<your_arch>` |
|---|---:|---|---:|
| GB10 | 121 | H100 / GH200 | 90 |
| RTX 50-series | 120 | RTX 40-series / L40 | 89 |
| GB300 / B300 | 103 | RTX 30-series | 86 |
| B200 / GB200 | 100 | A100 | 80 |

</details>

---

## 🚀 Run it

```bash
cd yourCase                       # your OpenFOAM case (0/  constant/  system/)
brae                              # steady, transient or compressible
brae -case /path/to/yourCase      # or from anywhere
brae -partition -case yourCase    # cache mesh + AMG once, later runs start warm
brae -cases coarse medium fine    # one case per GPU, extras queue
brae --help
```

---

## 📐 Known limitations

- **Single GPU** — the whole mesh is resident on one device, which caps problem size at its memory.
- **Non-orthogonal meshes, compressible** — rhoSimpleFoam's iteration-1 velocity sits 2.4e-04 to 3.1e-03 from
  OpenFOAM on a 16° mesh, against 1.9e-09 orthogonal. Not yet localised; the table above is orthogonal.
- **Bit-reproducible run to run, not bit-identical to OpenFOAM.** brae's reductions are deterministic
  (atomic-free per-cell gather), and two runs of one binary write byte-identical fields — gated. Against
  OpenFOAM the GPU sums in a different order, so agreement bottoms out at **1e-10 to 1e-13** rather than
  at zero; that is the table above.

---

## 📚 Documentation

[Getting started](docs/getting-started.md) · [Performance & tuning](docs/performance.md) ·
[Memory model](docs/memory-model.md) · [Roadmap](docs/roadmap.md)
