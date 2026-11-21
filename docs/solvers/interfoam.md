# interFoam

Transient two-phase solver for two incompressible, immiscible fluids (volume of fluid). A port of OpenFOAM
v2412's `interFoam`: the PIMPLE loop, the MULES / CMULES alpha transport, the momentum predictor, the `p_rgh`
pressure correctors and the turbulence transport run on the GPU. Boundary conditions on some patches, mesh
refinement and file output run on the host.

[<- back to all solvers](../../README.md#solvers)

You do not run it by name: a case whose `controlDict` says `application interFoam` is handed to it by `brae`.

```bash
cd yourTwoPhaseCase && brae
```

## At a glance

| | |
|---|---|
| **Algorithm** | PIMPLE with MULES / CMULES interface compression, alpha sub-cycling, `momentumPredictor`, `p_rgh` form |
| **Time** | `Euler`, `CrankNicolson`, `localEuler`; `adjustTimeStep` with `maxCo` / `maxAlphaCo`; restart from a written time |
| **Turbulence** | laminar; RAS k-epsilon and k-omega SST; LES kEqn |
| **Interface** | surface tension, `constantAlphaContactAngle`, curvature smoothing (CPU loop only) |
| **Meshes** | static; rigid-body and 6-DoF motion; deforming (displacement Laplacian wave makers); `dynamicRefineFvMesh` |
| **Coupled patches** | `cyclic`, `cyclicAMI`, `cyclicACMI`, porous baffles |
| **Waves** | OpenFOAM's `waveModels` generation and absorption conditions |
| **Sources** | `explicitPorositySource` (Darcy-Forchheimer), MRF |
| **I/O** | standard OpenFOAM case in, standard time directories out, ASCII or binary |
| **Not yet** | function objects (their `adjustableRunTime` write times do trim the time step, as in OpenFOAM), multi-GPU |

A setting this solver does not carry stops the run at start-up and names itself; it is never replaced by a
default.

## How it is checked

- All 42 `interFoam` tutorials shipped with OpenFOAM v2412 run to their own end time.
- The solver's steps are held to serial OpenFOAM on staged tutorial rows, field by field, in the test suite
  (856 tests on a GB10).
- The pressure equation is the one deliberate difference: whatever linear solver the case names for `p_rgh`,
  brae runs its own AMG-preconditioned conjugate gradient on the GPU and says so in a notice.

## Refused configurations

A configuration below stops the run at start-up, before the first time step, with a message that names the
entry, why brae refuses it and what to use instead. Each one used to run, or to stop only later.

| Configuration | Why it is refused |
|---|---|
| A boundary condition interFoam's writer has no branch for (`mixed`, `fixedGradient`, `outletInlet`, `uniformTotalPressure`, `timeVaryingMappedFixedValue`, a coded or `fixedMean` turbulence patch, ...) | No interFoam test runs it and its output is not written. It used to run to the first write time and stop there with nothing written. |
| `p0 table (...)` on `totalPressure` or `uniformTotalPressure` | OpenFOAM's `totalPressure` reads `p0` as a field and stops on a table. interFoam does not refresh `uniformTotalPressure`'s `p0(t)` yet, so `p0` would stay at its start value. |
| A quoted (pattern) key in `gradSchemes` that gives one of interFoam's gradients another scheme than brae's reader takes | OpenFOAM resolves the literal key, then the last matching pattern, then `default`. brae's gradient readers do not resolve patterns. Write the entry literally. |
| A pattern key in `laplacianSchemes` or `snGradSchemes` that differs from `default` | interFoam takes `default` for every laplacian and snGrad, and does not resolve patterns. |
| An `interpolationSchemes` entry other than `linear`, including one written on a single line | brae interpolates `linear` only. A one-line block used to be skipped. |
| `div(rhoPhi,U) Gauss vanLeerV` or `limitedLinearV` with a `cellLimited` `grad(U)`, on the GPU loop | The GPU momentum assembly builds that limiter from an unlimited gradient. The CPU loop (`brae_interFoam -case <dir>` without `-device`) runs it. |
| A `PIMPLE` entry that OpenFOAM's interFoam reads and brae's case reader does not | The run would use a different algorithm from the one the case names. No such entry exists today: this guard holds the reader to OpenFOAM's list. |

A run that diverges stops by name with a non-zero exit as soon as a solved field (alpha, U, p_rgh, phi or a
turbulence field) or the Courant number is no longer finite. It stops before writing that step, as OpenFOAM does
when it stops on a floating-point exception.

## Behaviour changes

**2026-11-21**:

- **Divergence stop.** A diverging run stops at the end of the time step in which a value becomes NaN or infinite.
  It names the fields and writes nothing of that step. It used to run to `endTime`, write NaN fields and exit 0.
- **`minIter` on `p_rgh`, `p_rghFinal`, `pcorr` and `pcorrFinal`** is now applied, as OpenFOAM's linear solvers
  apply it. No shipped tutorial sets it on these entries.
- **The start-up refusals listed above.** In the v2412 interFoam tutorials, none of these configurations appears,
  so every tutorial that ran before still runs.
- **`div(rhoPhi,U)`, `div(phi,alpha)` and `div(phirb,alpha)`** resolve as OpenFOAM resolves them: the literal key,
  then a matching pattern, then `default`. A pattern or a `default` used to be refused or not seen.
- **Unread entries** of `system/fvSolution`, `constant/turbulenceProperties` and `system/fvSchemes` are listed at
  start-up as `brae NOTICE [unread]` lines, as the other solvers list them. The run is not affected.
  `BRAE_DICT_AUDIT=0` turns the list off.
- **`brae check --setup`**: `brae_interFoam` stops before its first step when `BRAE_CHECK_SETUP` is set. The
  `brae check` command does not hand interFoam cases to it yet.

## Speed

Per time step against OpenFOAM on all 20 cores of the same NVIDIA GB10, as shipped, 30 steps (2026-10-07):

| mesh | brae against 20 cores |
|---|---|
| tutorial size, 42 cases | ahead on 30; behind on 12, all but four of them under 10,000 cells |
| each case at the largest size run (up to 3.6 million cells) | ahead on all 42, 1.1x to 4.9x |

A GPU step on a mesh of a few thousand cells is bound by the number of device calls, not by the work in
them; from roughly 30,000 cells upward brae is ahead and the margin grows with the mesh.
