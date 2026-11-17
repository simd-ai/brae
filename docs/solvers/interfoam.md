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
| **Function objects** | `volFieldValue`, `surfaceFieldValue`, `probes`, `fieldMinMax`, `exprField` and `runTimeControl` (see below); their `adjustableRunTime` write times also trim the time step, as in OpenFOAM |
| **Not yet** | the other function objects (`surfaces`, `forces`, `fieldAverage`, ... are noticed and not run) |
| **Multi-GPU** | `-parallel -device` runs a case across up to 8 GPUs (decomposed by `brae decompose` or `decomposePar`, or decomposed by the run itself), one process per GPU, and agrees with OpenFOAM `-parallel`; laminar flow only, no AMI or moving meshes under `-parallel`: [Multi-GPU](../multi-gpu.md) |

A setting this solver does not carry stops the run at start-up and names itself; it is never replaced by a
default.

## Function objects

`controlDict`'s `functions { }` entries of these types run at the end of each time step (the common `volIntegrate` shapes are
summed on the device in one batch; everything else is evaluated on the host), with
OpenFOAM's cadence (`writeControl` `timeStep`, `runTime`, `adjustableRunTime`, `writeTime`, `none`; no output at
the start time) and OpenFOAM's file layout, `postProcessing/<object>/<start time>/<file>`:

| type | supported |
|---|---|
| `volFieldValue` | `regionType all` or `cellZone`; `volIntegrate`, `volAverage`, `sum`, `min`, `max` of the phase fraction, `p_rgh` and `exprField` results |
| `surfaceFieldValue` | `regionType patch`; `areaAverage`, `areaIntegrate`, `sum`, `min`, `max` of the phase fraction |
| `probes` | `fixedLocations`, cell values of the phase fraction, `U` and `p_rgh` |
| `fieldMinMax` | `U` (magnitude or components) and scalar fields |
| `exprField` | `action new`, scalar expressions of the phase fraction, `p_rgh`, `U` (`.x() .y() .z()`, `mag`, `magSqr`), `pos().x()/y()/z()`, `sqr`, `sqrt`, other expression fields and numbers; kept in memory (`autowrite false`) |
| `runTimeControl` | `minMax` conditions (`maximum`, `minimum`) on another object's result, `satisfiedAction end`: the run writes the time and ends |

`enabled false` removes an object, as in OpenFOAM. An entry of one of these types that asks for something not
listed (`writeFields`, `weightField`, a vector field in `volFieldValue`, `interpolationScheme cellPoint`,
`autowrite true`, a `timeStart`/`timeEnd` window, ...) stops the run at start-up and names itself. An object of
any other type is noticed and not run. The objects are evaluated on the start mesh, so a case with function
objects and a mesh that moves or refines is refused. The device loop downloads a field for them only in the
steps in which an object is due; `runTimeControl` is evaluated every step, so a condition on a `writeControl
timeStep` object costs a download of the fields it reads each step.

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
