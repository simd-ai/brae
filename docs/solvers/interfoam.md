# interFoam

Transient two-phase solver for two incompressible, immiscible fluids (volume of fluid). A port of OpenFOAM
v2412's `interFoam`: the PIMPLE loop, the MULES / CMULES alpha transport, the momentum predictor, the `p_rgh`
pressure correctors and the turbulence transport run on the GPU. Boundary conditions on some patches, mesh
refinement and file output run on the host.

[<- back to all solvers](../../README.md#-will-it-run-my-case)

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

## Speed

Per time step against OpenFOAM on all 20 cores of the same NVIDIA GB10, as shipped, 30 steps (2026-10-07):

| mesh | brae against 20 cores |
|---|---|
| tutorial size, 42 cases | ahead on 30; behind on 12, all but four of them under 10,000 cells |
| each case at the largest size run (up to 3.6 million cells) | ahead on all 42, 1.1x to 4.9x |

A GPU step on a mesh of a few thousand cells is bound by the number of device calls, not by the work in
them; from roughly 30,000 cells upward brae is ahead and the margin grows with the mesh.
