# rhoCentralFoam in brae: architecture

rhoCentralFoam is a solver in its own right. It owns its equations (explicit conservative rho, rhoU and rhoE, with
implicit viscous correctors), its central-upwind flux assembly and its time loop. It shares brae's mesh, fields, I/O,
boundary conditions, thermo, gradients, linear smoothers and time controls. Nothing here goes through SIMPLE, PIMPLE or
rhoSimpleFoam's pressure correction, and there is no pressure equation: p = rho/psi.

## 1. OpenFOAM's structure, and where each part lives in brae

The reference is OpenFOAM v2412 `applications/solvers/compressible/rhoCentralFoam`. In the table, (H) marks the host
reference `_cpp` (namespace `brae::cpu::rhoCentral`), and (D) marks the device module (namespace
`brae::gpu::rhoCentral`). Shared types live in `brae::rhoCentral`.

| OpenFOAM | brae module | notes |
|---|---|---|
| `rhoCentralFoam.C` `main()`: the time loop, its order, write | `rhoCentralFoam.cu` (driver) + `gpuRhoCentralFoam.cu` (entry) | one loop for both arms; the arm is a `Stepper` (sec. 2) |
| `createFields.H`, `createFieldRefs.H` | `rhoCentralCreateFields_cpp.cu` | host only, once (and on restart) |
| `createTimeControls.H`, `readTimeControls.H`, `setDeltaT.H` | shared `time_controls.cuh` + `rhoCentralClock.cuh` | accumulating clock; no setInitialDeltaT |
| `readFluxScheme.H` | `rhoCentralControls.cu` | all dictionary reading and refusals in one place |
| `directionInterpolate.H` + `LimitedScheme`/`NVDTVD`/`NVDVTVDV` | `rhoCentralReconstruct.cuh` (BRAE_HD face kernels) on the shared limiter library (sec. 3) | |
| lines 102-210 (flux core) | `rhoCentralFlux.cuh` (BRAE_HD per face) + `rhoCentralFlux_cpp.cu` (H) / `rhoCentralFlux.cu` (D) | the per-face arithmetic is written once; the arms differ in loops and gathers |
| `centralCourantNo.H` | inside the flux modules (H/D) | resident on the device; read back in batches |
| `setRDeltaT.H` (LTS) | `rhoCentralSetRDeltaT_cpp.cu` (H) / `rhoCentralSetRDeltaT.cu` (D) | shared `fvc::smooth` |
| lines 222-245 (rho, rhoU, U, viscous U corrector) | `rhoCentralMomentum_cpp.cu` (H) / `rhoCentralMomentum.cu` (D) | the corrector uses the shared laplacian assembly and the shared Gauss-Seidel |
| lines 247-282 (sigmaDotU, rhoE, e, thermo.correct, e corrector) | `rhoCentralEnergy_cpp.cu` (H) / `rhoCentralEnergy.cu` (D) | |
| lines 284-288 (p, rho_b) and the patch updates | `rhoCentralBoundary_cpp.cu` (H) / `rhoCentralBoundary.cu` (D) | the OpenFOAM order of sec. 4 |
| hePsiThermo + hConst/janaf + const/sutherland | `rhoCentralThermo.cuh` (BRAE_HD) | sec. 3: why it is not rhoSimpleFoam's thermo |
| `BCs/` (maxwellSlipU, smoluchowskiJumpT, fixedRho) | `BCs/` in this directory | solver-owned BCs, as in OpenFOAM |
| `turbulence->correct()`, `muEff`, `alphaEff` | `rhoCentralTurbulence` hook over brae's compressible closures | refused until gated |
| `mesh.update()`, `meshPhi` | `rhoCentralMeshMotion` hook | refused until gated |

## 2. The driver and the arms

`rhoCentralFoam.cu` holds OpenFOAM's loop, in OpenFOAM's order, once:

```
readTimeControls
if !LTS: setDeltaT (CoNum from the previous step), ++runTime, [mesh.update]
stepper.fluxes()            directed interpolation, ap/am/a_pos/aSf, phi, phiUp, phiEp, amaxSf
stepper.courant()           centralCourantNo (kept on the device)
if LTS: stepper.setRDeltaT(); ++runTime
stepper.solveRho(); stepper.solveRhoU(); [viscous U corrector]
stepper.solveRhoE(); [viscous e corrector]
stepper.correctThermoAndPressure()
[turbulence->correct()]
write (download on the device arm) at write times; restart state (uniform/time)
```

`Stepper` is the seam between the arms. `HostStepper` composes the `_cpp` modules; `DeviceStepper` composes the device
modules. Both start from the same `createFields` state. The device arm reads scalars back in batches (the Courant
numbers, the status) and never syncs every step. The exception is adjustTimeStep, whose next deltaT needs CoNum on the
host every step: that costs one 8-byte read per step, or later a one-thread kernel.

## 3. Shared infrastructure: reuse, hoist, do not inherit

**Reused as is:**
- mesh and fields: `PrimitiveMesh`, `FvGeometry`, `buildPatches`, `SolutionDirections`, `DeviceMesh`, `DeviceBuffer`,
  `FoamDict`, `readField`/`buildField`;
- gradients: `deviceGaussGradFused`, and leastSquares/cellLimited where a case asks for them;
- time controls: `time_controls.cuh` (`setDeltaT`, `adjustDeltaT`, `startDeltaT`), `resolveStartTime`;
- `fvc::smooth` (host and device);
- the viscous correctors' pieces: `fvm::laplacian` assembly and the Gauss-Seidel smoothers;
- brae's host boundary-condition classes (`fv_patch_field.cuh`) and device value evaluation (`deviceBCValue` and the
  update kernels: inletOutlet, symmetry, wedge, pressureInletOutletVelocity);
- `writeVolField`.

**Hoisted into shared code**, because rhoCentralFoam needs it and today it exists only privately in another solver:
- a BRAE_HD limited-scheme library under `src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/`. It
  provides NVDTVD r, NVDVTVDV r and the limiters as OpenFOAM's `limitedSchemes` define them. `device_limiter.cuh` stays
  as it is for its current callers;
- an accumulating clock: `value += deltaT`, write switches as `Time::operator++`, `uniform/time`;
- a whole-keyword fvSchemes lookup (`parseFieldGradScheme` matches substrings);
- per-cell rDeltaT for localEuler.

**Not inherited** (each was measured or traced as different from OpenFOAM):
- the `w*P + (1 - w)*N` interpolation form: OpenFOAM uses `w*(P - N) + N`;
- the closed-form e -> T inversion and `R*T`: OpenFOAM runs a Newton loop and uses `p/(p/(R*T))`;
- rhoSimpleFoam's rho/p clamps (`rhoMin`/`rhoMax`/`pMin`);
- `thermo_parse`'s silent defaults (molWeight, Cp, Pr, mu, As, Ts);
- `setInitialDeltaT`;
- brae's `assignable()` on symmetry and wedge (use `ofAssignmentWritesValue()`);
- the per-face normal on symmetryPlane: OpenFOAM uses the patch-averaged normal;
- the writer dropping BC keys (psi, gamma, fieldInf, lInf, T0), so OpenFOAM can restart from brae's output.

## 4. Boundary values within a step (OpenFOAM's order, rhoCentralFoam.C)

- **rho, rhoU, rhoE patches.** They are `calculated` on ordinary patches and take the constraint type on constraint
  patches (symmetryPlane, symmetry, wedge, empty).
  - After their solves, the constraint patches re-evaluate from the new cells (224, 227, 257).
  - `rhoU_b == rho_b*U_b` (233) uses the previous step's rho_b on calculated patches, and the new one on constraint
    patches.
  - `rhoE_b == ...` (267) does the same.
  - `rho_b == psi_b*p_b` (288) comes last.
- **U** is evaluated at 232: inletOutlet and pressureInletOutletVelocity read this step's `phi`. Slip and symmetry use
  per-face normals, symmetryPlane the patch normal.
- **e** is evaluated at 264 (`=`, which evaluates constraint patches) and at 265 (`correctBoundaryConditions`).
  - The energy BCs run `Tw.evaluate()` while the cell T is still the old one.
  - The e patch type follows T: fixedValue -> fixedEnergy, zeroGradient -> gradientEnergy, mixed and its derived types
    -> mixedEnergy, and anything else keeps T's type.
- **thermo.correct()** (266):
  - a T patch that `fixesValue()` (fixedValue, every mixed-derived type, calculated) gets `e_b = HE(p_b, T_b)`;
  - any other T patch gets `T_b = THE(e_b, p_b, T_b)` from its own old T.
  - p_b here is still the previous step's.
- **p** is evaluated at 287.

## 5. The first stage's files: keep, refactor, add

| file | verdict |
|---|---|
| `rhoCentralThermo.cuh` | **keep**; extend with janaf and sutherland as BRAE_HD |
| `rhoCentralFoam_cpp.cuh/.cu` | **split**: controls and refusals -> `rhoCentralControls`; createFields -> `rhoCentralCreateFields_cpp`; reconstruction and flux -> `rhoCentralFlux_cpp` on the shared limiter library and `rhoCentralFlux.cuh`; the solves -> `rhoCentralMomentum_cpp` / `rhoCentralEnergy_cpp`; patch updates -> `rhoCentralBoundary_cpp`; writer -> driver. The ad-hoc `BcKind` enum gives way to brae's patch-field classes, plus a per-patch constraint flag. |
| `rhoCentralFoam_device.cuh/.cu` | **split** the same way; the limiter copies go to the shared library; `bndKernel` becomes `rhoCentralBoundary.cu` over `DeviceBoundary` |
| `gpuRhoCentralFoam.cu` | **shrinks** to the entry; the loop moves to the driver |
| `tests/rho_central_sod_vs_openfoam.sh`, `rho_central_refusals.sh`, `rho_central_sod.py` | **keep**; the Sod gate is the refactor's regression gate |

**Missing (not in brae at all):**
- boundary conditions: waveTransmissive/advective, totalTemperature, totalPressure's psi branch, maxwellSlipU,
  smoluchowskiJumpT;
- janaf thermo;
- face operators: `dotInterpolate(Sf, tensor)` and the vector snGrad;
- `fvm/fvc::ddt(rho, X)` with a variable rho;
- rhoCentralFoam's setRDeltaT;
- velocityComponentLaplacian mesh motion (movingCone).

## 6. Milestones, each closed by a gate against OpenFOAM v2412

Every gate:
- compares brae with OpenFOAM on the same case (same schemes, BCs and time step), at every write time;
- carries a control that must turn red;
- reports conservation, positivity, GPU memory and runtime.

| | milestone | gate (cases) |
|---|---|---|
| A | restructure into the modules above; brae BC classes; shared limiter library; driver + Stepper | the Sod gate and the refusals unchanged |
| B | adjustTimeStep: setDeltaT, adjustable writes, first step x1.2, restart deltaT | shockTube **as shipped**: the deltaT sequence and fields |
| C | constraint and transform BCs: symmetryPlane, symmetry, slip, wedge, with the conservative-state patch semantics of sec. 4 | wedge15Ma5 as shipped; obliqueShock as shipped (shock angle against theta-beta-M) |
| D | inletOutlet (U, T -> mixedEnergy) and purgeWrite | forwardStep as shipped |
| E | restart: startFrom latestTime, `uniform/time`, the writer keeping BC keys | stop/restart against the continuous run and against OpenFOAM's own restart |
| F | other Riemann problems and limiters (the vanAlbada, Minmod, MUSCL, limitedLinear family) | Lax, 123 and a strong shock tube: brae against OpenFOAM and the exact solution |
| G | viscous: muEff, tauMC, U and e correctors (smoothSolver/GaussSeidel), sigmaDotU, sutherland, noSlip | a viscous shock tube and a laminar compressible boundary layer against OpenFOAM |
| H | totalPressure (psi branch), totalTemperature, waveTransmissive | a supersonic nozzle and LadenburgJet60psi |
| I | LTS (setRDeltaT, fvc::smooth), janaf, maxwellSlipU, smoluchowskiJumpT | biconic25-55Run35 |
| J | compressible RAS on the central path | a modified tutorial with kOmegaSST |
| K | mesh motion (velocityComponentLaplacian, meshPhi) | movingCone |

The solver stays labelled **experimental** until A to E pass and at least the four inviscid tutorials run as shipped.
Multi-GPU is designed for, not built: the Stepper's patch evaluation and gradient stages are where halo exchanges will
go.
