# rhoCentralFoam in brae: architecture

rhoCentralFoam is a solver in its own right. It owns its equations (explicit conservative rho, rhoU and rhoE, with
implicit viscous correctors), its central-upwind flux assembly and its time loop. It shares brae's mesh, fields, I/O,
boundary conditions, thermo, gradients, linear smoothers and time controls. Nothing here goes through SIMPLE, PIMPLE or
rhoSimpleFoam's pressure correction, and there is no pressure equation: p = rho/psi.

## 1. OpenFOAM's structure, and where each part lives in brae (after milestone A)

The reference is OpenFOAM v2412 `applications/solvers/compressible/rhoCentralFoam`. In the table, (H) marks the host
reference (namespace `brae::cpu::rhoCentral`), (D) the device arm (namespace `brae::gpu::rhoCentral`), and (HD)
BRAE_HD code both arms run (namespace `brae::rhoCentral`).

| OpenFOAM | brae file | role |
|---|---|---|
| `rhoCentralFoam.C` `main()`: start-up, time loop, write | `gpuRhoCentralFoam.cu` (entry), `rhoCentralFoam.cuh/.cu` (driver) | one loop for both arms; the arm is an `Arm` (sec. 2) |
| `readFluxScheme.H`, the dictionary reads, every refusal | `rhoCentralControls.cuh/.cu` | the `Case`: controls, mesh, flat boundary addressing |
| state | `rhoCentralState.cuh` | `State` (cells and flat boundary), `PatchFields` (brae patch-field objects of U, p, T), `StepReport`, controls |
| `createFields.H`, `createFieldRefs.H`, thermo construction | `rhoCentralCreateFields_cpp.cuh/.cu` (H) | once, on the host; both arms start from its state |
| lines 102-210 + `directionInterpolate.H`: reconstruction and fluxes; `centralCourantNo.H`; fvc::div | `rhoCentralFlux_cpp.cuh/.cu` (H), `rhoCentralFlux.cuh/.cu` (D) | gradients, limited directed interpolation, face fluxes |
| the face flux (114-210) | `rhoCentralFaceFlux.cuh` (HD) | written once |
| 222-286, cell half: rho, rhoU, U, rhoE, e, thermo.correct, p | `rhoCentralCellUpdate.cuh` (HD) | four stages (momentum, energy, thermo, pressure) |
| 232-288, patch half | `rhoCentralPatchUpdate.cuh` (HD), with brae's patch fields | rhoU_b, e_b, T_b, psi_b, rhoE_b; then p_b and rho_b |
| one step, composed | `rhoCentralStep_cpp.cuh/.cu` (H), `rhoCentralStep.cuh/.cu` (D) | the two `Arm`s |
| hePsiThermo + hConst + perfectGas | `rhoCentralThermo.cuh` (HD) | Es, Cv, psi, the Newton e -> T loop |

Planned, not built yet: `setDeltaT` with an accumulating clock (B), the constraint and transform BCs (C), the
viscous correctors (G), `setRDeltaT` (I), the solver's own `BCs/` (maxwellSlipU, smoluchowskiJumpT; G, I),
turbulence (J) and mesh motion (K). Each goes into the module of its OpenFOAM counterpart above, or a new one beside
it.

## 2. The driver and the arms

`rhoCentralFoam.cu` holds the loop once: Time, the write cadence, the log, the writes, the diagnostics. An `Arm` is
the step:

```
step()      fluxes; the cells (four stages); U's patches; T's patches on the old cell T; the patch update; p's
            patches; rho_b
flush()     the Courant numbers and statuses of the steps since the last flush
download()  the state, for a write (the device arm copies it back; the host arm's is already in place)
```

- The host arm runs each stage as a loop over the cells, OpenFOAM's field-at-a-time order.
- The device arm fuses the fvc::div gather and the four stages into one kernel, evaluates U, p and T's patches
  through brae's `DeviceBoundary` (`deviceBCValue`), and keeps the Courant numbers and statuses on the device for up
  to 256 steps.
- What milestone B needs from the arm is finer: CoNum before setDeltaT, and LTS's ++runTime after the fluxes. It will
  split `step()` at the fluxes then, not before.

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

## 5. Milestone A: what moved

| before (stage 1) | after |
|---|---|
| `rhoCentralFoam_cpp.cuh/.cu` (case, createFields, host step, writer) | `rhoCentralControls`, `rhoCentralCreateFields_cpp`, `rhoCentralFlux_cpp`, `rhoCentralStep_cpp`, the driver |
| `rhoCentralFoam_device.cuh/.cu` (kernels, DeviceSolver) | `rhoCentralFlux.cu` (flux kernels), `rhoCentralStep.cu` (DeviceArm, cell/patch/Courant kernels) |
| `gpuRhoCentralFoam.cu` (main with the loop) | the entry; the loop is in `rhoCentralFoam.cu` |
| private host gaussGrad and vanLeer/vanLeerV | shared `fvc::gaussGrad` and `cpu::limitedSchemes::detail` (bit-identical on every Sod run) |
| private device scalar vanLeer | shared `limitedFaceWeight` + `limiterOfR` |
| a per-patch BcKind enum with its own evaluation | brae's patch-field objects (`PatchFields`) on the host, `DeviceBoundary` + `deviceBCValue` on the device |
| two copies of the face flux, cell update and patch update | one BRAE_HD copy each |

**Kept solver-local, on purpose:**
- The NVDVTVDV vector limiter on the device. brae's only other copy is inlined in `divLimitedVFaceKernel`.
- The whole-keyword fvSchemes lookup, because `parseFieldGradScheme` matches substrings.
- The Newton thermo, because the shared thermo inverts in closed form.

**Measured equivalence:**
- **Sod gate:** every written file of the gate's nine runs is byte-identical to the stage-1 binary's (252 of 252).
  That covers the CUDA arm, the host arm, Tadmor, 1000 cells, both controls, and the OpenFOAM runs. The logs are
  identical except the timing and memory lines.
- **1,024,000-cell 3-D tube, 100 steps:** the device arm moved by 5.5e-17 relative (U, near-zero velocities in the
  quiescent region) and stays 2.70e-15 from OpenFOAM, as before.
  - The cause: the shared `limitedFaceWeight` is the same arithmetic as the old private copy, but nvcc fuses its
    three-term `d & grad` differently. That shows only where all three terms are nonzero, which the 1-D tube never
    has. Swapping the private copy back restores byte identity, which isolates the cause.

## 5b. Remaining architectural issues

- `Arm::step()` is one call. Milestone B (setDeltaT needs CoNum first) and milestone I (LTS advances runTime after
  the fluxes) need it split at the fluxes.
- brae's `Time`/`WriteControl` assume a fixed deltaT (`timeValue = start + n*deltaT`). adjustTimeStep needs an
  accumulating clock (sec. 3).
- The device arm costs about 19% more per step on tiny meshes, from about five more small launches (the shared
  boundary evaluation): 132 us/step at 100 cells, against 111. It is unchanged at 1M cells (17.6 against 17.4
  ms/step). Device memory rose 14% at 1M cells (1362 against 1194 MiB), for the boundary buffers.
- The host arm rebuilds per-patch boundary arrays for `fvc::gaussGrad` every step. That is correct but allocates;
  acceptable for the oracle.
- The flat boundary layout assumes no coupled patch. Cyclic and processor patches (multi-GPU) will need the
  DeviceBoundary/DeviceMesh appended-face layout.

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
