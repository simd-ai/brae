# Roadmap

## Shipping today

- [`simpleFoam`](solvers/simplefoam.md) — steady incompressible
- [`pimpleFoam`](solvers/pimplefoam.md) — transient incompressible, URANS / DES / LES
- [`rhoSimpleFoam`](solvers/rhosimplefoam.md) — steady compressible, subsonic and transonic

Single GPU, single region. All six rhoSimpleFoam tutorials OpenFOAM v2412 ships run **as shipped**.

## Next

Ordered by value against cost, not by ambition.

1. **`pisoFoam`** — transient incompressible. `pimpleFoam` already runs the harder superset; PISO is PIMPLE with
   `nOuterCorrectors 1`.
2. **`rhoPimpleFoam`** — transient compressible. Reuses `rhoSimpleFoam`'s thermo, energy equation and turbulence
   closure on `pimpleFoam`'s time loop. No new numerics.
3. ~~**`interFoam`** — two-phase VoF.~~ Shipped in 0.4.1: see [the solver page](solvers/interfoam.md).
4. **Adaptive time stepping** — `adjustTimeStep` / `maxCo`. Refused today; needed by `interFoam` anyway.
5. **MRF and `fvOptions` in the transient solver** — the steady solvers have both.
6. **A general functionObject framework** — `forceCoeffs` is hard-wired today; probes, `fieldAverage`, sampling and
   surfaces are silently ignored.

## Open defects

- **Non-orthogonal meshes, compressible.** `rhoSimpleFoam`'s iteration-1 velocity sits 2.4e-04 to 3.1e-03 from
  OpenFOAM on a 16° mesh, against 1.9e-09 on an orthogonal one. Scheme-independent, not localised.
- **`uniform_function1`** — a non-constant `uniformValue` is accepted and then frozen at its file value. A silent
  substitution; the highest-priority open item.
- **`mean_velocity_force`** — illegal memory access in the scalar read-back path.
- **`pimple_loop_contract`** — `residualControl` does not cut the outer loop where OpenFOAM's does.
- **`eval_scoped`** — `#eval{ 2*$R }` cannot resolve `$variable` references. Refuses rather than guessing.
- **`linear_solver_setup`** — the refusal is correct; the test does not catch it, so the binary aborts.

## Not scheduled

- **Multi-GPU.** The distributed solvers moved to `legacy/` and are out of scope (`legacy/README.md`).
  `brae ... -parallel` refuses at start-up. Single-device residency is what removes the domain-decomposition
  approximation OpenFOAM pays in parallel.
- **Overset** (`overRhoSimpleFoam`, `overPimpleDyMFoam`, `overInterDyMFoam`). Belongs on its own branch — it is not
  a boundary condition, it changes the matrix itself:
  - `cellCellStencil` — donor/acceptor search across disconnected meshes, in four flavours. Geometric search across
    mesh regions is the hard part on a GPU.
  - `fvMeshPrimitiveLduAddressing` — an acceptor's equation is *replaced* by an interpolation from its donors, so the
    sparsity pattern changes. brae's device matrix has fixed LDU addressing built from the mesh.
  - `oversetAdjustPhi` / `oversetPatchPhiErr` — global flux conservation across the overlap.
  - Hole cutting and cell classification.

  Scale is comparable to or larger than a whole solver port, and its value is in *moving* bodies, which is transient.
  Refused by name since the `overset_refused` gate: `overset` was briefly treated as a constraint patch type, so brae
  synthesised a default entry and an overset case **ran, converging to a wrong answer**.

## Accuracy

- **Operator level** — same state, one iteration: **1e-10 to 1e-13** against OpenFOAM on the rhoSimpleFoam
  tutorials. See the [solver pages](solvers/) for the per-case table.
- **End to end** — both codes running their own iteration from the same start: under 1% on the fields, ~8e-03 on p
  at iteration 100 for squareBend. Differences compound through the nonlinear iteration; this is two nearby
  trajectories to the same fixed point, not a discretisation error.
- **Run to run** — byte-identical. Reductions are deterministic (atomic-free per-cell gather), gated by
  `rho_run_to_run_identity`.
- **Near-wall (low y+)** — near-wall turbulence quantities can differ ~10–15% on flat plates; bulk fields and forces
  still track OpenFOAM.
- **Extreme aspect ratio (AR ≳ 1000)** — the steady solve can settle differently on sliver cells; use the same
  under-relaxation as OpenFOAM.
- On hard steady cases (bluff-body aero), brae plateaus its residual exactly as OpenFOAM does.

**Brae never guesses.** If a model, boundary condition or scheme is not supported it stops at start-up naming
exactly what it found, so you never get a silently wrong result.
