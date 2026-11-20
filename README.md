<p align="center">
  <img src="docs/images/brae-banner.png" alt="brae" width="100%">
</p>

# brae

OpenFOAM's solvers, re-ported to CUDA. Your OpenFOAM case runs on one NVIDIA GPU, and every solver is checked
against OpenFOAM v2412 itself: field by field, on the same case, by a gate that runs real OpenFOAM beside it.

<p>
  <img alt="License AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-blue">
  <img alt="CUDA 12.4+" src="https://img.shields.io/badge/CUDA-12.4%2B-76B900">
  <img alt="GPU Ampere to Blackwell" src="https://img.shields.io/badge/GPU-Ampere%20%7C%20Ada%20%7C%20Hopper%20%7C%20Blackwell-76B900">
  <img alt="OpenFOAM v2412" src="https://img.shields.io/badge/OpenFOAM-v2412-brightgreen">
</p>

**[Getting started](docs/getting-started.md) · [Solvers](docs/solvers/) · [Benchmarks](docs/performance.md) ·
[Memory model](docs/memory-model.md) · [Architecture](src/README.md) · [Roadmap](docs/roadmap.md) ·
[Contributing](CONTRIBUTING.md)**

## Quick start

```bash
curl -fsSL https://brae.sh/install.sh | sh
cd yourCase          # 0/  constant/  system/
brae                 # reads controlDict's application and runs that solver
```

An OpenFOAM case goes in and OpenFOAM time directories come out. You don't need an OpenFOAM install or `decomposePar`.

## At a glance

All rows below come from one NVIDIA GH200: brae on its GPU (96 GiB HBM3), OpenFOAM v2412 on its 72 Grace cores
(72 MPI ranks). The cases are stock v2412 tutorials, scaled up by their own `blockMesh`. brae was `main` at df3cd8cb
(2026-11-18). A step's time is the median gap between successive step lines over steps 11 to 60 (steady) or 11 to 40
(transient), so start-up and mesh reading fall outside it for both codes; function objects, writes and
`residualControl` were removed from both.

| solver | tutorial, scaled | largest mesh on one GH200 | speed at 20 M cells against 72 cores |
|---|---|---|---|
| simpleFoam | squareBend, kEpsilon | **43.7 M cells**, 88 GiB, 0.41 s an iteration | **7.3×** faster |
| rhoSimpleFoam | aerofoilNACA0012, 2-D | **15.1 M cells**, 91 GiB | **15.1×** faster (at 10 M) |
| interFoam | damBreakWithObstacle, VoF | **26.3 M cells**, 91 GiB | **5.5×** faster |
| pimpleFoam | TJunction, kEpsilon | **28.3 M cells**, 93 GiB | **3.7×** faster with `BRAE_GS_HOST_SMOOTHER=0`¹ |

<p align="center">
  <img src="docs/images/one-gh200-seconds-per-step.png" alt="Seconds per step against mesh size for the four solvers, brae on one GH200 GPU against OpenFOAM on its 72 Grace cores, log-log" width="100%">
</p>

¹ pimpleFoam's U, k and epsilon Gauss-Seidel solves run on the GPU only with `BRAE_GS_HOST_SMOOTHER=0`. The default
on `main` still sweeps them on one CPU thread, which is 2.5× *slower* than OpenFOAM at 19.8 M cells. Both paths write
the same fields, byte for byte.

How it was measured: [benchmarks →](docs/performance.md#one-gh200-four-solvers).

## Solvers

Type `brae`. It reads the `application` entry in your `controlDict` and runs the matching solver.

| solver | flow | checked against OpenFOAM v2412 |
|---|---|---|
| [`simpleFoam`](docs/solvers/simplefoam.md) | steady incompressible, RANS | cell by cell, 5 RAS models; motorBike drag within 1.6 % |
| [`pimpleFoam`](docs/solvers/pimplefoam.md) | transient incompressible: URANS, DES, LES | pitzDaily kOmegaSST, U within 0.07 % at t = 0.3 s |
| [`rhoSimpleFoam`](docs/solvers/rhosimplefoam.md) | steady compressible, subsonic and transonic | 5 tutorials as shipped, worst field 5.5e-10 after one iteration |
| [`interFoam`](docs/solvers/interfoam.md) | two-phase VoF: laminar, RAS, LES, moving and refining meshes, waves | all 42 tutorials run to their end time; steps held to serial OpenFOAM field by field |
| [`rhoCentralFoam`](src/applications/solvers/rhoCentralFoam/RELEASE.md) (**experimental**) | transient compressible, density-based (Kurganov/Tadmor central-upwind): inviscid, laminar, kOmegaSST | the seven v2412 tutorials as shipped, every field at every write; 14 gates, every check passing but 24 registered open checks on staged variants |

<details>
<summary><b>Support matrix</b>: turbulence, thermo, schemes, boundary conditions, fvOptions</summary>

<br>

| | supported |
|---|---|
| **turbulence** | kEpsilon · realizableKE · kOmegaSST · kOmegaSSTLM · SpalartAllmaras · Smagorinsky · WALE · SA-DDES/IDDES · kOmegaSST-DDES/IDDES · laminar · generalizedNewtonian |
| **thermo** | hePsiThermo · heRhoThermo · perfectGas · hConst · sutherland · const · liquid (NSRDS correlations, OpenFOAM's own `he→T` inversion) |
| **convection** | upwind · linearUpwind · linearUpwindV · LUST · linear · limitedLinear · limitedLinearV · vanAlbada |
| **pressure–velocity** | SIMPLE · SIMPLEC (`consistent yes`) · PIMPLE · transonic |
| **time** | steadyState · Euler · backward · CrankNicolson |
| **fvOptions** | explicitPorositySource · limitTemperature · fixedTemperatureConstraint · scalarFixedValueConstraint · MRF |
| **rhoCentralFoam** (experimental) | hePsiThermo with hConst or janaf, const or sutherland · fixed `deltaT`, `adjustTimeStep` or LTS · vanLeer · vanAlbada · Minmod · MUSCL · SuperBee · limitedLinear (and their V forms) · static meshes or `velocityComponentLaplacian` motion. The full list: [release report](src/applications/solvers/rhoCentralFoam/RELEASE.md) |

**Boundary conditions**

| | |
|---|---|
| basic | fixedValue · zeroGradient · noSlip · calculated |
| geometric | slip · symmetry · symmetryPlane · wedge · empty |
| coupled | cyclic · cyclicAMI |
| inlet / outlet | inletOutlet · outletInlet · flowRateInletVelocity · surfaceNormalFixedValue · freestream · freestreamVelocity |
| pressure | totalPressure · fixedFluxPressure · pressureInletOutletVelocity |
| time-varying | uniformFixedValue · codedFixedValue · fixedMean |
| turbulence | nutkWallFunction · epsilonWallFunction · omegaWallFunction · kqRWallFunction · turbulentIntensityKineticEnergyInlet · turbulentMixingLengthDissipationRateInlet |
| thermal | externalWallHeatFluxTemperature · limitTemperature |

</details>

Anything not in this matrix stops the run at start-up, and the message names it.

## Supported platforms

| | |
|---|---|
| GPU | NVIDIA Ampere or newer: A100, RTX 30/40/50, L40, H100, GH200, B200, GB200, GB300, GB10 |
| CUDA | 12.4 or newer (13.x recommended) |
| OS | Linux, x86_64 and aarch64 |
| Case | OpenFOAM v2412 format, ASCII or binary, plain or `.gz` |

## How it works

- The mesh and the fields stay on the GPU from the first step to the last.
- The matrices are OpenFOAM's LDU matrices, assembled one thread per cell. simpleFoam and rhoSimpleFoam write
  byte-identical fields run to run, and the test suite holds them to it.
- Whatever linear solver the case names for the pressure equation, brae runs its own AMG-preconditioned conjugate
  gradient there, and the log says so. Every other equation keeps the case's own solver settings.
- The source mirrors OpenFOAM's own tree (`src/OpenFOAM`, `src/finiteVolume`, `src/TurbulenceModels`, ...), so each
  ported class sits where OpenFOAM keeps it. A solver ported on this layout keeps a host (`_cpp`) reference beside
  its CUDA arm; the two share the case reader, the loop and the writer.

More: [architecture →](src/README.md) · [memory model →](docs/memory-model.md) ·
[performance and tuning →](docs/performance.md)

## How it is checked

Every gate runs OpenFOAM v2412 on the same machine and the same case, then compares brae's output with it, field by
field, the host reference arm first where a port has one, then the CUDA arm. A gate carries a control, a deliberately
wrong variant that must go red. Thresholds come from measurement and are never widened to make a gate pass. The
suite (`ctest`, about 900 registered tests) runs on a self-hosted GPU runner for every change to `main`; the hosted
CI compiles every pull request. The few gates that stay red on purpose carry the `known-red` label, and CI holds them
to their registered failures. See [CONTRIBUTING](CONTRIBUTING.md) to build and run them.

## It tells you when it can't

If brae does something other than what your case asked, it says so on stderr:

| prefix | meaning |
|---|---|
| `[ignored]` | your case asked for it; brae does nothing with it |
| `[approximated]` | related but not equal, for example a scheme downgrade or a formula substitution |
| `[defaulted]` | brae could not read a value and fell back to a default |
| `[unread]` | dictionary entries brae never consulted |
| **refusal** | brae stops rather than give a wrong answer |

> A notice means brae gave less than you asked for, but still a defensible answer. A refusal means the answer would
> have been wrong. When in doubt, brae refuses rather than guess.

## Run it

```bash
cd yourCase                       # your OpenFOAM case (0/  constant/  system/)
brae                              # steady, transient or compressible
brae -case /path/to/yourCase      # or from anywhere
brae -partition -case yourCase    # cache mesh + AMG once, later runs start warm
brae -cases coarse medium fine    # one case per GPU, extras queue
brae --help
```

<details>
<summary><b>Build from source</b></summary>

```bash
# deps: cmake >= 3.24, CUDA toolkit, an MPI (OpenMPI), SCOTCH, zlib
git clone https://github.com/simd-ai/brae.git
cd brae
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=<your_arch>
cmake --build build -j 4 --target brae brae_pimpleFoam brae_rhoSimpleFoam brae_interFoam brae_rhoCentralFoam
```

`brae` hands each case to the solver named by its `application` entry, so build all five side by side. Keep `-j`
bounded to about one job per 2 GB of free memory. A bare `-j` starts every compile job at once and runs out of
memory, and the build then stops with only `Error 2`.

| GPU | `<your_arch>` | GPU | `<your_arch>` |
|---|---:|---|---:|
| GB10 | 121 | H100 / GH200 | 90 |
| RTX 50-series | 120 | RTX 40-series / L40 | 89 |
| GB300 / B300 | 103 | RTX 30-series | 86 |
| B200 / GB200 | 100 | A100 | 80 |

</details>

## Limitations

- **One GPU.** The whole mesh must fit in its memory, so a GPU's memory caps the problem size. On a 96 GiB GH200
  that cap is 43.7 M cells for simpleFoam, and 15 M to 28 M cells for pimpleFoam, rhoSimpleFoam and interFoam
  (table above).
- **Compressible flow on non-orthogonal meshes.** rhoSimpleFoam's iteration-1 velocity is 2.4e-04 to 3.1e-03 away
  from OpenFOAM's on a 16° mesh, against 1.9e-09 on an orthogonal one. The cause hasn't been found yet.
- **Not bit-identical to OpenFOAM.** The GPU sums in a different order from OpenFOAM, so agreement bottoms out at
  1e-10 to 1e-13 (relative) rather than at zero.
- **pimpleFoam is not yet bit-reproducible run to run.** Two runs of pitzDaily with a pressure reference differ by
  up to 1.4e-08 (relative, U) after 10 steps; simpleFoam and rhoSimpleFoam do not differ at all.
- **rhoCentralFoam is experimental.** It runs on one GPU, refuses cyclic patches, LES and RAS models other than
  kOmegaSST at start-up, and writes ascii where a case asks for binary (with a notice). 24 registered checks on staged
  variants stay open; none is on a tutorial as shipped.
  [Release report →](src/applications/solvers/rhoCentralFoam/RELEASE.md)

## Contributing

Pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the two review rules (no regressions against
OpenFOAM v2412, never slower than OpenFOAM on the CPU), how to build and run the tests, and the coding guidelines;
[src/README.md](src/README.md) explains the source layout.

## License

AGPL-3.0. See [LICENSE](LICENSE).
