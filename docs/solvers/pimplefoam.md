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
| **Restart** | seamless — writes and re-reads `phi` and the `ddt0(...)` state, so a restart continues the same trajectory |
| **I/O** | standard OpenFOAM case in, standard time directories out (U, p, phi, turbulence); ASCII & binary mesh |
| **Not yet** | adaptive `deltaT` (`adjustTimeStep`/`maxCo`), MRF, `fvOptions`, runtime function objects, multi-GPU |

Legend: ✓ supported · ✗ not yet.

## Time integration

| feature | status |
|---|---|
| `Euler` (first order) | ✓ |
| `backward` (second order) | ✓ |
| `CrankNicolson` incl. off-centring coefficient `ocCoeff` | ✓ |
| Implicit `fvm::ddt(U)` in the momentum predictor | ✓ |
| Implicit `fvm::ddt` on **turbulence transport** (k, epsilon, omega, nuTilda) | ✓ true URANS, not quasi-steady |
| `ddt0(...)` state written and re-read on restart (`backward`, `CrankNicolson`) | ✓ |
| Fixed `deltaT` | ✓ |
| `adjustTimeStep` / `maxCo` (Courant-limited dt) | ✗ — refused at start-up, not silently ignored |

`writeControl timeStep` and `runTime`/`adjustableRunTime`, `writeInterval` and `purgeWrite` behave as in
`Foam::Time`.

## PIMPLE controls (`fvSolution`)

| entry | status |
|---|---|
| `nOuterCorrectors` | ✓ |
| `nCorrectors` (PISO inner) | ✓ |
| `nNonOrthogonalCorrectors` | ✓ |
| `relaxationFactors` (`equations`, `fields`) | ✓ — default 1.0 (no relaxation), as OF does for transient |
| `residualControl` on outer correctors | ✗ (runs all `nOuterCorrectors`) |

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
| Smagorinsky (LES) | ✓ algebraic sub-grid `nut` |
| WALE, dynamic-k LES, RSM | ✗ |

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

- `adjustTimeStep yes` — use a fixed `deltaT` under your target Courant number.
- `constant/MRFProperties` with an active zone.
- `system/fvOptions` or `constant/fvOptions`.

Also missing: runtime function objects (probes, forces, fieldAverage) during the time loop, and multi-GPU
transient runs.

---

See [roadmap.md](../roadmap.md) for what comes next.
