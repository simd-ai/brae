# pimpleFoam

Transient incompressible solver (PIMPLE). A GPU port of OpenFOAM v2412's `pimpleFoam`, fully device-resident: the
time loop, the momentum predictor, the pressure correctors and the turbulence transport all stay on the GPU, and
nothing is copied back between time steps. It shares its three phases with [`simpleFoam`](simplefoam.md) — the same
validated momentum, pressure and turbulence code, with an implicit `fvm::ddt` folded in.

[<- back to all solvers](../../README.md#solvers)

You do not run it by name: a case whose `controlDict` says `application pimpleFoam` is handed to it by `brae`.

```bash
cd yourTransientCase && brae
```

## At a glance

| | |
|---|---|
| **Algorithm** | transient incompressible PIMPLE (outer correctors + PISO inner correctors) |
| **Time schemes** | `Euler`, `backward`, `CrankNicolson` (with off-centring coefficient) |
| **Turbulence** | laminar; URANS: k-epsilon, realizable k-epsilon, k-omega SST, k-omega SST-LM, Spalart-Allmaras; hybrid/LES: SA-DDES, SA-IDDES, k-omega SST-DDES, k-omega SST-IDDES, Smagorinsky |
| **Restart** | writes and re-reads `phi` and the CrankNicolson `ddt0(...)` state; the write grid continues from the restart time as OpenFOAM's does; a moving-mesh restart is refused |
| **I/O** | standard OpenFOAM case in, standard time directories out (U, p, phi, turbulence); ASCII & binary mesh |
| **Not yet** | MRF, multi-GPU; the forms listed under [Refused configurations](#refused-configurations) |

Legend: ✓ supported · ✗ not yet.

## Time integration

| feature | status |
|---|---|
| `Euler` (first order) | ✓ |
| `backward` (second order) | ✓ |
| `CrankNicolson` incl. off-centring coefficient `ocCoeff` | ✓ |
| Implicit `fvm::ddt(U)` in the momentum predictor | ✓ |
| Implicit `fvm::ddt` on **turbulence transport** (k, epsilon, omega, nuTilda) | ✓ true URANS, not quasi-steady |
| `ddt0(...)` state written and re-read on restart (`CrankNicolson`) | ✓ |
| Named `ddt(X)` entries | only when they resolve to the `default` scheme; any other is refused |
| Fixed `deltaT` | ✓ |
| `adjustTimeStep` / `maxCo` / `maxDeltaT` (Courant-limited dt) | ✓ OpenFOAM's clock: `setInitialDeltaT`, `setDeltaT` before every step, `Time::adjustDeltaT` onto the write times, `uniform/time` written (issue #53, M4) |

`writeControl timeStep` and `runTime`/`adjustableRunTime`, `writeInterval` and `purgeWrite` behave as in
`Foam::Time`. After a restart the `runTime` write grid continues from the restart time, as OpenFOAM's does
(`Time::operator++` measures it from the resolved start).

Under `adjustTimeStep yes` brae runs OpenFOAM's clock (work package M4, issue #53), as `pimpleFoam.C` does: the
Courant number and `setInitialDeltaT` before the loop (on a run from its first time only), then before every step the
Courant number on the flux the last step left, `setDeltaT` and `++runTime`, with `Time::adjustDeltaT` shortening steps so
that they land on each write time under `writeControl adjustable`. Each write carries `uniform/time` (value, index,
deltaT, deltaT0) as OpenFOAM's does, so a restart in either code takes up the step where the other left it.
`tests/pimple_adjusttimestep_vs_openfoam.sh` replays brae's log through OpenFOAM's rules (every step's deltaT to the bit
from brae's own Courant numbers), holds the write times and fields against OpenFOAM's, and restarts both ways. The
Courant numbers are each code's own flux's, so the steps agree with OpenFOAM's as closely as the fields do.

## PIMPLE controls (`fvSolution`)

| entry | status |
|---|---|
| `nOuterCorrectors` | ✓ |
| `nCorrectors` (PISO inner) | ✓ |
| `nNonOrthogonalCorrectors` | ✓ |
| `relaxationFactors` (`equations`, `fields`) | ✓ — default 1.0 (no relaxation), as OF does for transient |
| `residualControl` on outer correctors | ✓ |

## Turbulence

Every model the steady solver has, plus the hybrid RANS/LES set — with the transport equations marched in time:

| model | |
|---|---|
| laminar | ✓ |
| k-epsilon, realizable k-epsilon | ✓ |
| k-omega SST, k-omega SST-LM (transition) | ✓ |
| Spalart-Allmaras | ✓ |
| SA-DDES, SA-IDDES | ✓ |
| k-omega SST-DDES, k-omega SST-IDDES | ✓ |
| Smagorinsky, WALE (LES) | ✓ algebraic sub-grid `nut`, `delta cubeRootVol` on a 3-D mesh |
| SA-DES, kOmegaSST-DES (non-delayed), dynamic-k LES, RSM | ✗ refused |

Wall functions, model coefficients and the `turbulenceProperties` parse are shared with the steady solver, so they
behave identically.

## Boundary conditions, schemes, mesh

Shared with [`simpleFoam`](simplefoam.md): the same BC set (including `codedFixedValue` / `codedMixed`, compiled on
the device at run time), the same `fvSchemes` div / laplacian / grad parse, the same ASCII-or-binary `polyMesh`
reader and OpenFOAM writer. Anything unsupported is named at start-up.

## Validation

- **pitzDaily, k-omega SST, transient to steady:** velocity within **0.07%** of OpenFOAM v2412 `pimpleFoam` on the
  same case at t = 0.3 s.
- `pimple_run` / `pimple_dispatch` ctests march a kEpsilon URANS pitzDaily end to end, then restart it from
  `latestTime` and check the run resumes from the written `phi` rather than recomputing it.
- `gpu_pimple` checks the transient step reduces to the steady step in the steady limit.

A transient run is only converged when the *flow* stops changing. A small linear-solver residual early in a
slow-developing transient does not mean the solution has settled — compare successive time directories, not
residuals.

## Performance

1× H100 80GB HBM3. pitzDaily developed to a stable field, then advanced at a fixed `deltaT`. Both solvers run
the same case at the same solver tolerance, and both are fully converged:

| cells | brae | SPUMA (OpenFOAM-GPU, best config) | speedup |
|---|---|---|---|
| 4.9M | 56 s | 255 s | **4.5×** |
| 15M  | 179 s | did not finish in 40 min | — |

brae stays fully device-resident — 0 bytes moved CPU↔GPU per time step — while SPUMA's unified-memory port pays
migration on every iteration. SPUMA is shown at its fastest config for this GPU class (PCG + `aDIC` async
preconditioner; ~30% faster than its `twoStageGaussSeidel` path here). brae is on its fast path (device AMG-PCG +
FP32 V-cycle); FP64 is only ~3.5% slower, so the speedup holds either way.

The comparison is a fixed-step transient sample from the *developed* field — the honest, apples-to-apples setup.
An impulsive start at fixed `deltaT` diverges for **both** solvers (Courant blow-up) and is not a valid benchmark.

## Not supported yet

These stop the run with a message rather than producing a quietly wrong answer:

- `constant/MRFProperties` with an active zone.
- `fvOptions` sources brae cannot apply, and time windows that do not cover the run.
- everything under [Refused configurations](#refused-configurations).

Function objects: `forces`, `forceCoeffs` and `scalarTransport` (constant `D`) run; every other type is named as not
run. Multi-GPU transient runs are not in public brae.

## Refused configurations

Each is refused by name at start-up, before the first step, and under `brae check --setup`: brae would compute it
differently from OpenFOAM v2412. The message names the entry, why, and what to use instead.

| what | why |
|---|---|
| `transportModel` other than `Newtonian`; no `transportModel`; no `nu` | brae carries one constant laminar viscosity; OpenFOAM reads both entries with no default |
| no `constant/turbulenceProperties` (an OpenFOAM.org `momentumTransport` is named); no `simulationType` | OpenFOAM reads the file `MUST_READ` and `simulationType` with no default; brae used to run laminar |
| no `deltaT`, no `endTime` in `controlDict` | OpenFOAM reads deltaT with no default, and endTime under `stopAt endTime` (with no `stopAt` it takes 0 and runs no step); brae used 1e-3 and 1.0 |
| a named `ddtSchemes` entry (literal or regex) resolving a solved field to another scheme than `default` | brae integrates every field with `default` |
| a moving-mesh restart (a start after the case's first time, or a start directory with `polyMesh/points`) | brae reads the mesh from `constant/polyMesh` and writes no points, meshPhi or Uf |
| `solidBody` motion entries missing where OpenFOAM reads them, or given as a Function1 that is not a constant | OpenFOAM stops on a missing one; brae defaulted origin, axis and omega |
| `updateControl` other than `always`, `timeStep`, `none`; `updateInterval` below 1 or not an integer | brae moves the mesh every step or never |
| `moveMeshOuterCorrectors yes` with `nOuterCorrectors` > 1 on a moving mesh | brae's update does not keep the old points once per step: from the second corrector the mesh flux is zero |
| kOmegaSST, kOmegaSSTLM, Spalart-Allmaras and their DES/IDDES models on a moving mesh (unless `wallDist { updateInterval 0; }`) | OpenFOAM recomputes the wall distance after every move; brae computes it once |
| a plain `cyclic` whose cells rotate or deform with the motion, or whose two halves move apart; any plain `cyclic` under `velocityComponentLaplacian` | brae builds the cyclic's weights and vectors once |
| `forces`/`forceCoeffs` on a moving mesh | brae's forces keep the start geometry |
| `forces`/`forceCoeffs`/`scalarTransport` with `adjustTimeStep yes` | brae's function objects run on a clock with controlDict's deltaT |
| `SpalartAllmarasDES`, `kOmegaSSTDES` | brae has the delayed (DDES) variants only |
| LES without `delta` | OpenFOAM reads it with no default |
| `Smagorinsky`/`WALE` with `delta maxDeltaxyz` | brae's algebraic LES kernels take cbrt(V) |
| `cubeRootVol` with `deltaCoeff` other than 1, or on a mesh with an `empty` patch | brae's cubeRootVol is cbrt(V); OpenFOAM's 2-D form is sqrt(V/thickness) |
| IDDES with a delta other than `IDDESDelta`, an `IDDESDeltaCoeffs` `Cw` other than brae's, an `hmax` other than `maxDeltaxyz`/`maxDeltaxyzCubeRoot`, a non-default hmax `deltaCoeff`, or `wallDist exactDistance` | brae's IDDES length scale takes none of these |
| IDDES on a mesh where brae's hmax (the cell's bounding box) differs from OpenFOAM's `maxDeltaxyz` | they agree only on cells whose faces are axis-aligned |

pimpleFoam also stops when a turbulence field goes bad: a turbulence solve's residual that is not finite (every step),
or `sum|k| + sum|epsilon or omega|` (`nuTilda` for Spalart-Allmaras) not finite or 1e12 times its first value for 25
checks in a row (every tenth step and before every write). The step is not written. `BRAE_ALLOW_NONFINITE=1` opts out.

## Behaviour changes

**2026-11-21.** Every configuration in the table above used to run; it is now refused. Of the OpenFOAM v2412
pimpleFoam tutorials, three that the previous build set up are now refused: `RAS/TJunctionArrheniusBirdCarreauTransport`
(its `transportModel ArrheniusBirdCarreau` ran Newtonian at nu 1e-5), `RAS/propeller` (forces on a moving mesh, and under
adjustTimeStep) and `LES/periodicHill/transient` (brae's IDDES hmax differs from OpenFOAM's by up to 39 % on its
curvilinear mesh). `RAS/rotatingFanInRoom` and the `SpalartAllmarasIDDES-IDDESDelta` setup of `LES/wallMountedHump`
were refused already and are now refused earlier, for kOmegaSST on a moving mesh and for exactDistance under IDDES. The
tutorials that run write the same files as before. Two outputs change: after a `latestTime` restart under
`writeControl runTime` or `adjustableRunTime` the write times are OpenFOAM's (the first step after the restart is no
longer written), and a run whose turbulence field goes non-finite stops at that step instead of writing it.
**M4.** `adjustTimeStep yes` runs on OpenFOAM's clock. It used to run on brae's own (no `setDeltaT` before the first
step, no trimming to the write times, the time named `startTime + deltaT*steps`), so every adaptive run's step sequence
and transient fields differed from OpenFOAM's from the first step; for a while it was refused. Adaptive runs now take
different, OpenFOAM's, steps: their write times and fields change, and each write gains `uniform/time`. Fixed-deltaT
runs are unchanged.

---

See [roadmap.md](../roadmap.md) for what comes next.
