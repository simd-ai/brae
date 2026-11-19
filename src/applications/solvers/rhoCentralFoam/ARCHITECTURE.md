# rhoCentralFoam in brae: architecture

rhoCentralFoam is a solver in its own right. It owns its equations (explicit conservative rho, rhoU and rhoE, with
implicit viscous correctors), its central-upwind flux assembly and its time loop. It shares brae's mesh, fields, I/O,
boundary conditions, thermo, gradients, linear smoothers and time controls. Nothing here goes through SIMPLE, PIMPLE or
rhoSimpleFoam's pressure correction, and there is no pressure equation: p = rho/psi.

## 1. OpenFOAM's structure, and where each part lives in brae (after milestone G)

The reference is OpenFOAM v2412 `applications/solvers/compressible/rhoCentralFoam`. In the table, (H) marks the host
reference (namespace `brae::cpu::rhoCentral`), (D) the device arm (namespace `brae::gpu::rhoCentral`), and (HD)
BRAE_HD code both arms run (namespace `brae::rhoCentral`).

| OpenFOAM | brae file | role |
|---|---|---|
| `rhoCentralFoam.C` `main()`: start-up, time loop, write | `gpuRhoCentralFoam.cu` (entry), `rhoCentralFoam.cuh/.cu` (driver) | one loop for both arms; the arm is an `Arm` (sec. 2) |
| `createTimeControls.H`, `setDeltaT.H`, `Foam::Time` (run, setDeltaT/adjustDeltaT, operator++) | shared: `time_controls.cuh` (the step policy, with interFoam) and `run_clock.cuh` (the clock, new in B) | the driver keeps the clock; an arm takes the step's deltaT |
| `readFluxScheme.H`, the dictionary reads, every refusal | `rhoCentralControls.cuh/.cu` | the `Case`: controls, mesh, flat boundary addressing |
| state | `rhoCentralState.cuh` | `State` (cells and flat boundary), `PatchFields` (brae patch-field objects of U, p, T), `StepReport`, controls |
| `createFields.H`, `createFieldRefs.H`, thermo construction | `rhoCentralCreateFields_cpp.cuh/.cu` (H) | once, on the host; both arms start from its state |
| lines 102-210 + `directionInterpolate.H`: reconstruction and fluxes; `centralCourantNo.H`; fvc::div | `rhoCentralFlux_cpp.cuh/.cu` (H), `rhoCentralFlux.cuh/.cu` (D) | gradients, limited directed interpolation, face fluxes |
| the face flux (114-210) | `rhoCentralFaceFlux.cuh` (HD) | written once |
| 222-286, cell half: rho, rhoU, U, rhoE, e, thermo.correct, p | `rhoCentralCellUpdate.cuh` (HD) | four stages (momentum, energy, thermo, pressure) |
| 232-288, patch half | `rhoCentralPatchUpdate.cuh` (HD), with brae's patch fields | rhoU_b, e_b, T_b, psi_b, rhoE_b; then p_b and rho_b; the transform patches' U (`transformPatchU`) and constraint-patch rho_b (`patchRhoMid`) |
| constraint and transform patches: symmetryPlane, symmetry, slip, wedge (milestone C) | `rhoCentralControls.cu` (per-face data on the `Case`), the patch half above | sec. 5e |
| inletOutlet on U, p, T, switched on line 195's `phi` (milestone D) | brae's `InletOutletPatchField` (host, `switchInletOutlet` in `rhoCentralCreateFields_cpp.cuh`) and `deviceUpdateInletOutlet` (device) | sec. 5f |
| `Time::writeObject`'s purgeWrite (milestone D) | shared: `src/OpenFOAM/db/Time/purge_write.cuh` (`PurgeWrite`, hoisted out of `WriteControl`) | the driver calls it after each write |
| `interpolate(vf, dir, "reconstruct(...)")`'s limiters (milestone F) | shared: `src/cuda/device_limiter.cuh` (`limiterOfR`, now BRAE_HD: one copy for both arms and every brae solver), the selectors in `device_mesh.cuh`; the case's choice in `rhoCentralControls.cu` (`reconstructLimiter`) | sec. 5h |
| restart (milestone E): `Time::setControls` reading `<start>/uniform/time`, `Time::writeTimeDict` writing it | shared: `run_clock.cuh` (reads deltaT under adjustTimeStep, deltaT0, index), `src/OpenFOAM/db/Time/time_dict_writer.cuh` (new) | sec. 5g |
| one step, composed | `rhoCentralStep_cpp.cuh/.cu` (H), `rhoCentralStep.cuh/.cu` (D) | the two `Arm`s |
| hePsiThermo + hConst + perfectGas | `rhoCentralThermo.cuh` (HD) | Es, Cv, psi, the Newton e -> T loop; constTransport's alphaEff (G) |
| the viscous correctors (milestone G): 220-221 tauMC, 235-244 the U corrector, 247-255 sigmaDotU, 273-282 the e corrector | `rhoCentralViscous.cuh/.cu` (D), on shared operators (sec. 5i); the order in `rhoCentralStep.cu`'s `stepViscous` | sec. 5i |

Planned, not built yet: `setRDeltaT` (I), the solver's own `BCs/` (maxwellSlipU, smoluchowskiJumpT; G, I),
turbulence (J) and mesh motion (K). Each goes into the module of its OpenFOAM counterpart above, or a new one beside
it.

## 2. The driver and the arms

`rhoCentralFoam.cu` holds the loop once: Time, the write cadence, the log, the writes, the diagnostics. An `Arm` is
the step:

```
step(dt)    fluxes; the cells (four stages); U's patches; T's patches on the old cell T; the patch update; p's
            patches; rho_b. dt is runTime.deltaTValue() after setDeltaT.H and ++runTime
flush()     the Courant numbers and statuses of the steps since the last flush
download()  the state, for a write (the device arm copies it back; the host arm's is already in place)
```

- The host arm runs each stage as a loop over the cells, OpenFOAM's field-at-a-time order.
- The device arm fuses the fvc::div gather and the four stages into one kernel, evaluates U, p and T's patches
  through brae's `DeviceBoundary` (`deviceBCValue`), and keeps the Courant numbers and statuses on the device for up
  to 256 steps.
- The loop is rhoCentralFoam.C's for `!LTS`: `Time::run`, setDeltaT.H on the previous step's Courant number,
  `++runTime`, the step at the new deltaT, `runTime.write()`. So milestone B needed no split of `step()`: the step only
  takes its deltaT. The split at the fluxes is LTS's (milestone I), where `++runTime` comes after them.
- At a fixed deltaT the driver flushes the device arm once a window; under adjustTimeStep once a step, since
  setDeltaT.H reads the step before's Courant number.

## 3. Shared infrastructure: reuse, hoist, do not inherit

**Reused as is:**
- mesh and fields: `PrimitiveMesh`, `FvGeometry`, `buildPatches`, `SolutionDirections`, `DeviceMesh`, `DeviceBuffer`,
  `FoamDict`, `readField`/`buildField`;
- gradients: `deviceGaussGradFused`, and leastSquares/cellLimited where a case asks for them;
- time controls: `time_controls.cuh` (`TimeControls`, `setDeltaT`, `WriteCadence`, `adjustDeltaT`, `startDeltaT`),
  `resolveStartTime`, `openFoamTimeName`;
- `fvc::smooth` (host and device);
- the viscous correctors' pieces: `fvm::laplacian` assembly and the Gauss-Seidel smoothers;
- brae's host boundary-condition classes (`fv_patch_field.cuh`) and device value evaluation (`deviceBCValue` and the
  update kernels: inletOutlet, symmetry, wedge, pressureInletOutletVelocity);
- `writeVolField`.

**Hoisted into shared code**, because rhoCentralFoam needs it and today it exists only privately in another solver:
- a BRAE_HD limited-scheme library under `src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/`. It
  provides NVDTVD r, NVDVTVDV r and the limiters as OpenFOAM's `limitedSchemes` define them. `device_limiter.cuh` stays
  as it is for its current callers;
- an accumulating clock: `value += deltaT`, write switches as `Time::operator++`, `uniform/time`. Done in B:
  `src/OpenFOAM/db/Time/run_clock.cuh` (`RunClock`);
- the purgeWrite FIFO (`Time::writeObject`). Done in D: `src/OpenFOAM/db/Time/purge_write.cuh` (`PurgeWrite`);
  `WriteControl::recordWritten`, which held the only copy, now calls it;
- a whole-keyword fvSchemes lookup (`parseFieldGradScheme` matches substrings);
- per-cell rDeltaT for localEuler.

**Not inherited** (each was measured or traced as different from OpenFOAM):
- the `w*P + (1 - w)*N` interpolation form: OpenFOAM uses `w*(P - N) + N`;
- the closed-form e -> T inversion and `R*T`: OpenFOAM runs a Newton loop and uses `p/(p/(R*T))`;
- rhoSimpleFoam's rho/p clamps (`rhoMin`/`rhoMax`/`pMin`);
- `thermo_parse`'s silent defaults (molWeight, Cp, Pr, mu, As, Ts);
- `setInitialDeltaT`;
- brae's `assignable()` on symmetry and wedge (use `ofAssignmentWritesValue()`);
- the per-face normal on symmetryPlane: OpenFOAM uses the patch-averaged normal (done in C: `Case::bUR`);
- U's value on a transform patch from the shared classes: the host's `SymmetryPlanePatchField` evaluates
  `v - (n.v) n` and the device's mixed slot `|n_k| ref_k + (1 - |n_k|) v_k` with `ref` refreshed by
  `deviceUpdateSymmetry` -- the same number, rounded differently on each arm and lagged on the device, which is right
  for the segregated solvers that own them. rhoCentralFoam evaluates OpenFOAM's `(iF + (I - 2nn)&iF)/2` and
  `faceT & iF` once, in `transformPatchU`, for both arms (sec. 5e). The classes still build p's and T's patches, and
  `wedgeGeometry` gives faceT;
- the writer dropping BC keys (psi, gamma, fieldInf, lInf, T0), so OpenFOAM can restart from brae's output.

## 4. Boundary values within a step (OpenFOAM's order, rhoCentralFoam.C)

- **rho, rhoU, rhoE patches.** They are `calculated` on ordinary patches and take the constraint type on constraint
  patches (symmetryPlane, symmetry, wedge, empty).
  - After their solves, the constraint patches re-evaluate from the new cells (224, 227, 257).
  - `rhoU_b == rho_b*U_b` (233) uses the previous step's rho_b on calculated patches, and the new one on constraint
    patches.
  - `rhoE_b == ...` (267) does the same.
  - `rho_b == psi_b*p_b` (288) comes last.
- **phi** is set at 195, `aphiv_pos*rho_pos + aphiv_neg*rho_neg`, each product rounded on its own before the sum
  (OpenFOAM's field algebra stores both): on a face of fluid at rest the two are exact negatives and phi is exactly 0.
  brae fused them into an FMA until milestone D and got +-5e-15 there (`roundedProduct`, `rhoCentralFaceFlux.cuh`).
- **U** is evaluated at 232: inletOutlet and pressureInletOutletVelocity read this step's `phi`, inletOutlet as
  `valueFraction = 1 - pos0(phi)` -- so phi exactly 0 takes the outflow branch. Slip and symmetry use per-face
  normals, symmetryPlane the patch normal.
- **e** is evaluated at 264 (`=`, which evaluates constraint patches) and at 265 (`correctBoundaryConditions`).
  - The energy BCs run `Tw.evaluate()` while the cell T is still the old one. On an inletOutlet T (mixedEnergy) that
    evaluate is the switch on this step's phi and the blend with the OLD cell T: an outflow face's T_b is last step's
    cell T.
  - The e patch type follows T: fixedValue -> fixedEnergy, zeroGradient -> gradientEnergy, mixed and its derived types
    -> mixedEnergy, and anything else keeps T's type.
- **thermo.correct()** (266):
  - a T patch that `fixesValue()` (fixedValue, every mixed-derived type, calculated) gets `e_b = HE(p_b, T_b)`;
  - any other T patch gets `T_b = THE(e_b, p_b, T_b)` from its own old T.
  - p_b here is still the previous step's.
- **p** is evaluated at 287; an inletOutlet p switches on the same phi.

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

- `Arm::step(dt)` is one call. Milestone I (LTS advances runTime after the fluxes) needs it split at the fluxes.
- Under adjustTimeStep the device arm synchronises once a step to read the Courant number, so launch latency is no
  longer hidden behind the next step. Measured, GB10 2026-11-17, step and flush time per step:
  - 100 cells: 280 to 335 us against 131 at a fixed deltaT (over three runs);
  - 1000 cells: 379 to 411 us against 161;
  - 1,024,000 cells: 18.1 ms against 17.6, about 3%.

  The host arm runs the 100-cell tube at 12 to 27 us a step. Ways out, not taken yet: the step as one CUDA graph,
  or setDeltaT.H and adjustDeltaT on the device with the write steps snapshotted.
- interFoam keeps its own copy of the time-name precision raise (`inter_writer_cpp.cu`); `RunClock` has the other.
  One could go once interFoam adopts the clock; it is not changed here.
- The device arm costs about 19% more per step on tiny meshes, from about five more small launches (the shared
  boundary evaluation): 132 us/step at 100 cells, against 111. It is unchanged at 1M cells (17.6 against 17.4
  ms/step). Device memory rose 14% at 1M cells (1362 against 1194 MiB), for the boundary buffers.
- The host arm rebuilds per-patch boundary arrays for `fvc::gaussGrad` every step. That is correct but allocates;
  acceptable for the oracle.
- The flat boundary layout assumes no coupled patch: cyclic and processor patches are refused at start-up
  (`rhoCentralControls.cu`, mesh patch types). Sec. 5c is the plan for both.

## 5c. Coupled patches: the transition before any multi-GPU work

An earlier note here called brae's coupled layout "appended faces". That is not what the code does.
`DeviceMesh` and `DeviceBoundary` keep coupled faces out of the internal-face list (`device_mesh.cuh`: merging
them breaks the owner-contiguous scatter), and the two kinds of coupled patch take two different routes:

- **cyclic, cyclicAMI** (`isCoupledInterfaceType`): skipped by the boundary gather; carried by `DeviceCyclic`
  (`device_cyclic.cuh`): both sides of every pair, own and neighbour cell (or the AMI stencil through
  `CyclicNbr`), Sf out of the own cell, weights, and `ifRank`, so a cell's interface terms are added in face order.
- **processor**: stays in the boundary gather at its patch-order offset (`procStart`), as category 8 with no
  matrix coefficients. After `DeviceHalo::exchange`, `DeviceHalo::scatterBoundaryValues` writes
  `w*psi_own + (1 - w)*psi_nbr` into that slot (`device_halo.cuh`), and the received neighbour cell values are
  readable per interface.

What rhoCentralFoam needs on a coupled face (OpenFOAM: `limitedSurfaceInterpolationScheme::weights` and
`surfaceInterpolationScheme::interpolate` on a coupled patch, `gaussGrad::correctBoundaryConditions`):

- both cell values (`patchInternalField`, `patchNeighbourField`) of rho, rhoU, rPsi, e and c;
- both cell gradients of the same fields, for the limiter's r. The neighbour's gradient is a gradient
  boundary value, so a processor patch exchanges twice a step: values, then gradients;
- the patch's CD weights and delta;
- then the face flux exactly as on an internal face, added to the own cell only. Each side computes its own
  copy of the face with its own Sf, which is what decomposed OpenFOAM does.

The transition, in order; none of it is scheduled in sec. 6 yet:

1. **Split the face body.** Take `faceKernel`'s per-face body into a device function of (P-side, N-side)
   states and gradients. `faceKernel` calls it with the internal neighbour; a new coupled-face kernel calls it
   with the neighbour from `DeviceCyclic` or from the halo, and scatters to the own cell in `ifRank` order.
   `patchFaceKernel` and the patch updates skip coupled faces.
2. **Keep the flat layout aligned with brae's.** The flat boundary arrays (`Case::bCell`, `bPatch`, `bEmpty`)
   already follow patch order. Make them skip `isCoupledInterfaceType` patches, as `DeviceMesh` does, so a
   processor face's flat index is its `DeviceBoundary` slot and `scatterBoundaryValues` writes into
   rhoCentralFoam's boundary buffers in place. Add a `bCoupled` flag beside `bEmpty`.
3. **Gradients.** `deviceGaussGradFused` already reads processor faces from their boundary slot, so the halo
   fill makes it correct unchanged. Cyclics add `deviceCyclicAddGrad`.
4. **Limiter weights.** `deviceCyclicLimitedWeights` and `deviceCyclicLimitedVWeights` build the weights from
   the cyclic face flux's sign. rhoCentralFoam interpolates in a fixed direction (+1 for pos, -1 for neg), so
   these need a direction argument, or the solver keeps its own coupled limiter call next to `vanLeerV`.
5. **Reductions.** The Courant number that setDeltaT.H reads (milestone B, on one GPU), and the diagnostics totals,
   become global reductions, so every rank picks the same deltaT.
6. **Host arm.** It stays the serial oracle. It takes cyclic patches only if brae's shared host patch fields
   and `fvc::gaussGrad` already carry the coupling. It never takes processor patches: no second CPU
   implementation.

Gates for the transition:

- **cyclic, single GPU:** a periodic 1-D tube carrying a density pulse, against serial OpenFOAM at every write
  time. Control: the pair cut into zeroGradient patches must go red.
- **processor, when multi-GPU is scheduled:** against OpenFOAM's own decomposed run on the same decomposition
  (`decomposePar`, `rhoCentralFoam -parallel`), not against serial OpenFOAM. A processor face's terms are summed
  after the internal faces, in boundary order, so decomposed OpenFOAM differs from serial OpenFOAM at
  round-off. The serial run is a second, looser comparison.

## 5d. Milestone B: adjustTimeStep

- **The clock** (`run_clock.cuh`, shared, new): `Time::run`, `Time::setDeltaT` with `adjustDeltaT`, `Time::operator++`
  (the accumulated value, deltaT0, the snap to 0, `writeTime_` for timeStep, runTime and adjustable, the time-name
  precision raise). The step policy is the shared `time_controls.cuh`, already gated on interFoam: setDeltaT.H with
  OpenFOAM's SMALL and the fused `1 + 0.1*maxDeltaTFact`, Time's trimming to the write times, the function objects'
  trimming, and a restart's `uniform/time` deltaT.
- **The driver** runs rhoCentralFoam.C's `!LTS` loop on it. CoNum starts at 0, so the first step grows by 1.2 (then
  is trimmed). The log prints OpenFOAM's three lines a step at writePrecision, as Info does. It writes only on write
  steps: the unconditional final write is gone, since `rhoCentralFoam.C` has none.
- **The arms** take the step's deltaT: the Euler `1/deltaT`, the Courant numbers' scale, the outflow diagnostics.
  At a fixed deltaT that is controlDict's deltaT, as before. The device flush reads its window with two async
  copies and one synchronisation.
- **Now refused by name:** purgeWrite > 0 (silently ignored before), writeControl clockTime and cpuTime.
- **Measured:**
  - the Sod gate's 252 written files byte-identical to milestone A's;
  - shockTube as shipped, 142 steps: every deltaT and time name identical to OpenFOAM's on both arms, fields within
    3.1e-15 at every write time;
  - Tadmor (189 steps) and 1000 cells (1247 steps): the deltaT and names likewise identical;
  - on 1000 cells this run amplifies round-off, OpenFOAM's own too: OpenFOAM decomposed in two is 6.9e-12 from
    OpenFOAM serial at 0.007, brae 6.3e-12. The gate (`tests/rho_central_shocktube_vs_openfoam.sh`) holds that arm to
    ten times OpenFOAM's own spread, measured in the same run.
- **Gated since C:** maxDeltaT binding (3e-5 on shockTube: 226 of 244 steps at the cap, every deltaT the same as
  OpenFOAM's); writeControl runTime and timeStep under adjustTimeStep (no trimming, so deltaT carries the Courant
  number's last bits: within 3.0e-15 of OpenFOAM's at every step, every time name the same); a start from OpenFOAM's
  own `uniform/time` (the C gate's rs7 arm).
- **Not gated:** a restart from brae's own output (brae writes no `uniform/time` yet; milestone E); the function
  objects' trimming (no rhoCentralFoam tutorial has one).

## 5e. Milestone C: the transform patches

- **What runs:** mesh patches symmetryPlane, symmetry and wedge with their own patch-field types on U, p and T (a
  constraint field on another patch type, or another field type on a constraint patch, is refused as OpenFOAM refuses
  it); `slip` on U, p or T of a plain patch. A symmetryPlane that is not planar is refused (symmetryPlanePolyPatch.C).
- **The per-face data** (`rhoCentralControls.cu`, on the `Case`): `bUTransform` and `bUR` -- for a mirror
  R = I - 2.0*sqr(nHat) as OpenFOAM builds it, nHat the patch's one area-weighted normal on a symmetryPlane and
  Sf/magSf per face on symmetry and slip; for a wedge, faceT from the shared `wedgeGeometry`. `bRhoConstraint`: the
  mesh patch is a constraint type. `bTseedOwn`: T's patch field is a transform.
- **What changes in the step** (sec. 4, `rhoCentralPatchUpdate.cuh`), both arms through the same BRAE_HD functions:
  - U at 232 on a transform face: `transformPatchU`, from the new cell U;
  - 233 and 267 multiply by rho_b as it stands there: on a constraint patch the rho solve's correctBoundaryConditions
    has set it to the NEW cell rho (`patchRhoMid`); elsewhere it is last step's psi_b*p_b;
  - 265-266 on a patch whose T is a transform: he takes T's type (basicThermo::heBoundaryTypes), so e_b = e_P with no
    Tw.evaluate(), and THE is seeded with T_b as the last thermo.correct() left it.
  - p (287) and rho_b (288) as before: a scalar transform is the cell value.
- **Measured** (`tests/rho_central_wedge_vs_openfoam.sh`): wedge15Ma5 and obliqueShock as shipped, a `symmetry`
  variant, `slip` on p and T, and an axisymmetric 15-degree cone on `wedge` patches (`tests/rho_central_cases.py`):
  - every step's time name, deltaT and file the same as OpenFOAM's; fields within ten times OpenFOAM's own
    serial-to-decomposed spread at every write time; restarted from OpenFOAM's own developed obliqueShock at t = 7,
    2.0e-13 from OpenFOAM after one step (OpenFOAM decomposed in four: 1.6e-13);
  - these runs amplify round-off, OpenFOAM's own too (a Mach 5 shock and the e -> T inversion at T = 1 against
    Tstd = 298.15): OpenFOAM decomposed is 7e-06 from serial on wedge15Ma5 by t = 0.14, brae 7.0e-06;
  - shock angles brae = OpenFOAM to 1e-6 degrees: wedge 24.61 (theory 24.32), cone 19.76 (Taylor-Maccoll 20.03),
    obliqueShock incident 29.64 and reflected 23.41 (29.00, 23.28), meeting the wall at x = 1.796 (1.804);
  - controls red: U unprojected on the transform faces (0.3 of the field), rhoU_b and rhoE_b by last step's rho_b on
    the constraint patches (2e-03 on obliqueShock, 5e-04 on the cone; not visible on wedge15Ma5, whose symmetry planes
    see only the uniform stream).
- **The envelope, both measurements kept** (approved with C; `tests/rho_central_gate_lib.sh`): every arm also
  reports itself against each OpenFOAM decomposition alone. Re-run after D's phi fix, obliqueShock stays above ten
  times the two-way split alone at t = 7 and 8 (CUDA 1.24x, host 1.95x; 2.9x at t = 5 to 8 in C's first run), and
  within the four-way one (at most 0.75 of it in one run, 0.32 in the next). The four-way reference is NOT
  reproducible on this OpenFOAM install: scotch partitions the same mesh differently from run to run, so that spread
  moves between runs while brae, OpenFOAM serial and the two-way split are bit-identical (tests/rho_central_gate_lib.sh).
- **Not gated:** viscous flow on these patches; 3-D; a symmetryPlane or wedge split over processors.

## 5f. Milestone D: inletOutlet and purgeWrite

- **What runs:** `inletOutlet` on U, p and T of a plain patch, on the flux `phi` (a `phi` entry naming another field,
  or no `inletValue`, is refused; so are outletInlet and the other mixed types); controlDict `purgeWrite` (a negative
  value warned about and taken as 0, as OpenFOAM does). forwardStep runs as shipped.
- **inletOutlet**, brae's own patch field and device switch, not a solver-local copy:
  - the host arm calls `switchInletOutlet` (`InletOutletPatchField::updateFromFlux`), the device arm
    `deviceUpdateInletOutlet`, on this step's boundary phi, before U (232), T (265) and p (287) are evaluated:
    valueFraction = 1 - pos0(phi), so phi exactly 0 is outflow;
  - T's inletOutlet makes he mixedEnergy (basicThermo::heBoundaryTypes); its `Tw.evaluate()` blends the inletValue
    with the cell's OLD T, and since mixed `fixesValue()`, thermo.correct() sets e_b = HE(p_b, T_b) -- the arithmetic
    `updatePatchFace` already had for a fixed T, so no new patch code was needed there;
  - at construction the file's `value` stands, and without one the face cells (inletOutletFvPatchField's
    `extrapolateInternal()`), as brae's class already did.
- **phi's rounding (a fix, sec. 4).** `aphiv_pos*rho_pos + aphiv_neg*rho_neg` was contracted into an FMA on both
  arms. On a face of fluid at rest the two products are exact negatives: OpenFOAM's phi is exactly 0 (outflow), brae's
  was +-5e-15. Measured on a shockTube with inletOutlet ends: brae took the inflow branch on both ends at step 1 and
  ran away from OpenFOAM. Each product is now rounded on its own (`roundedProduct`). Every brae result moved in its
  last bits: the Sod gate's outputs are no longer byte-identical to milestone A's (the brae files of its nine runs
  differ; OpenFOAM's are untouched), and its distances to OpenFOAM are the same size (CUDA 7.7e-15 before, 7.9e-15
  after; bound 8e-14). The B and C gates pass unchanged in count.
- **purgeWrite:** `Time::writeObject`'s FIFO, hoisted into `src/OpenFOAM/db/Time/purge_write.cuh` and called after
  each write; the start time's directory is never in it.
- **Measured** (`tests/rho_central_forwardstep_vs_openfoam.sh`, 94 checks; GB10, 2026-11-17):
  - forwardStep as shipped (16,128 cells, 6694 steps to t = 4): every deltaT and time name the same as OpenFOAM's,
    the same 11 directories left by purgeWrite 10 on both arms; at 17 digits every one of the 40 write times within
    ten times OpenFOAM's own spread (at most 0.22 of the bound; the four-way reference moves between runs, sec. 5e),
    the outlet's written U, T, rho likewise;
  - forwardStep's outlet takes outflow at every face and step (0 inflow face-steps of 428,416), so two variants
    carry the rest: the inlet as inletOutlet (inflow on every face, every step, on U, p and T), and a shockTube whose
    ends flow out at 50 m/s until the expansion turns the left end inward (rev17; restarted from OpenFOAM's own
    t = 0.013 and written every step through the switch: CUDA 2.4e-15, host 2.8e-15 from OpenFOAM at worst);
  - the bow-shock standoff 0.2915, the shock meeting the top wall at x = 0.6115 and the pitot pressure on the step
    12.0136 equal OpenFOAM's to 6 digits (Rayleigh's pitot 12.061);
  - controls red: the switch fed -phi (2.9e+07 x the bound, then an unphysical state at t = 0.91), T's inletOutlet
    on the new cell T (2.5e+05 and 3.0e+05 x on forwardStep in two runs), the switch on the step before's phi (3.2e+11 x on the reversing
    tube; NOT visible on forwardStep, whose outlet never changes branch), purgeWrite ignored (41 directories
    against 11);
  - two variants that do NOT work in OpenFOAM either, and are not gated: inletOutlet ends with the left state as
    inletValue (the switch flips every step and deltaT collapses to 1e-11 in both codes), and an open top on
    forwardStep (OpenFOAM stops on a negative temperature at t = 1.13; brae refuses the same step's state). A
    switch fed by a velocity at round-off level (a tube at rest) decides on round-off, in OpenFOAM too.
- **Performance** (forwardStep to t = 0.03, nothing else running): 16,128 cells 0.65 ms a step against OpenFOAM's
  5.6 (8.6x); 64,512 1.6 against 24.4 (15x); 258,048 5.6 against 101 (18x); 1,032,192 19.4 against 620 (32x).
  GPU memory 2.9 KiB a cell (2.9 GiB at 1,032,192 cells). On the shipped mesh the step is bound by the per-step
  synchronisation of adjustTimeStep (sec. 5b).
- **Not gated:** a restart from brae's own output (milestone E), inletOutlet on a coupled patch, viscous flow, 3-D.

## 5g. Milestone E: restart

- **OpenFOAM's restart, read in its source.** `Time::setControls` (Time.C:146-307) takes the start time from the
  start DIRECTORY's name (startFrom startTime, firstTime or latestTime); `<start>/uniform/time` gives deltaT under
  adjustTimeStep only (controlDict's otherwise), deltaT0, and the time index, from which timeStep writes count.
  runTime/adjustable writes are measured from the start time (operator++, adjustDeltaT). `Time::writeTimeDict`
  (TimeIO.C:508-539) writes uniform/time first at every write time: value at 18 digits, name, index, deltaT and
  deltaT0 at writePrecision; functionObjectList writes an empty functionObjectProperties beside it. rhoCentralFoam
  itself has no restart code: createFields.H reads p, T, U, re-derives T from he and rebuilds rho, rhoU, rhoE (NO_READ),
  exactly as at 0; Euler keeps no old-time field (no `*_0` written or read); CoNum starts at 0, so the first setDeltaT.H
  grows deltaT by 1.2. A restart under adjustTimeStep is therefore NOT its uninterrupted run continued.
- **What brae had** (milestones B to D): the start-time resolution (`resolveStartTime`), uniform/time's deltaT and
  index (`RunClock`), the file's `value` kept on inletOutlet (brae's class), createFields as OpenFOAM's -- so a restart
  from OpenFOAM's output already ran (the C and D gates' rs7 and rsRev arms).
- **What E added:** brae writes uniform/time and functionObjectProperties as OpenFOAM does (the shared
  `time_dict_writer.cuh`; at all 40 forwardStep write times the same text as OpenFOAM's, value, name, index, deltaT,
  deltaT0), so a restart from brae's own output -- by brae or by OpenFOAM -- has the time state; `RunClock` also reads
  deltaT0 (Time.C:301; overwritten at the first step, as there); controls of the restart's mistakes.
- **Measured** (`tests/rho_central_restart_vs_openfoam.sh`; GB10, 2026-11-17): restarts of forwardStep at t = 1, 2, 3,
  wedge15Ma5 (fixed deltaT) at 0.06 and 0.12, obliqueShock at 3 and 7, the reversing tube at 0.012, 0.013 and 0.014
  (across inletOutlet's switch), shockTube's timeStep writes from index 40, an off-grid restart under adjustable writes,
  forwardStep as shipped continued with startFrom latestTime (6-digit fields and uniform/time, purgeWrite 10), and a
  start directory without uniform/; each started from OpenFOAM's output and from brae's, against OpenFOAM restarted
  from the same directory:
  - every step's deltaT and time name, and every directory, the same as OpenFOAM's (the first deltaT 1.2 times the
    start's, trimmed to the next write time, in both); uniform/time the same text at every write time;
  - fields within ten times OpenFOAM's own spread at every write time, with ONE exception: wedge15Ma5 restarted from
    OpenFOAM's t = 0.06, CUDA arm, at t = 0.12 (6.2e-10 against a bound of 3.4e-10). The largest difference of the CUDA
    arm, the host arm (8.1e-11) and OpenFOAM in four (3.4e-11) is at the same free-stream cell just ahead of the shock
    (rho 1.400005), where the limiter acts on differences of 1e-6; OpenFOAM in 2, 3, 4, 6 and 8 all read 1e-11 there at
    0.12 and 0.2 to 1e-9 at 0.14. The CUDA arm without fused multiply-adds reads the same 6.1e-10. Round-off reaching a
    sensitive cell first, not a restart error; the envelope is not changed for it (a decision for review);
  - restart against uninterrupted: brae's and OpenFOAM's restarts part from their own uninterrupted runs by the same
    amount (forwardStep 8.6e-03, 4.2e-03, 4.2e-02 by one time unit later; obliqueShock 3.4e-04, 9.7e-04; wedge15Ma5 at a
    fixed deltaT by 0.2: from 0.06 OpenFOAM's 2.4e-05, brae's 2.1e-05; from 0.12 both 4.1e-06 -- rho, rhoU, rhoE
    are rebuilt from p, T, U, and the Mach 5 shock amplifies the last bits);
  - controls red: deltaT from controlDict (1.1e+09 x the bound), the index dropped and the write grid from 0 (the
    directories off OpenFOAM's), the inletOutlet value taken from the cells (1.3e+10 x), no uniform/time written
    (OpenFOAM restarted from such output 1.2e+09 x from OpenFOAM restarted from brae's full output).
- **The open check, investigated** (fixed, reproducible references: OpenFOAM decomposed by simple (2 1 1) and (3 1 1),
  hierarchical (4 2 1), the frozen scotch four-way partition; OpenFOAM serial with its start state's p or T moved one
  unit in the last place; every step written from 0.06 to 0.125):
  - at cell 2348 (x 0.109, y 0.058; rho 1.4000050406, a precursor of the shock above the free stream's 1.4) OpenFOAM
    serial with p one ulp up is 6.13e-10 from OpenFOAM serial at t = 0.12, with T one ulp up 4.91e-10 -- brae's CUDA
    arm 6.18e-10, the host arm 8.1e-11 -- all crossing 1e-10 at t = 0.1179 to 0.1182; the decompositions, whose
    perturbation starts at their processor faces, read 1.5e-12 to 3.4e-11 there at 0.12 (the four-way one reaches 1e-10
    at 0.125);
  - the limiter there acts on rho differences of 4e-10 to 2e-9 between neighbours: runs 1e-10 apart choose vanLeer's r
    30 to 50 % apart (0.0377 for OpenFOAM serial on face 4645 at t = 0.1178, 0.0242 with p one ulp up, 0.0245 for the
    CUDA arm, whose rho there agrees with the one-ulp run to 7e-17);
  - so the CUDA arm's trajectory is OpenFOAM's own under a one-ulp perturbation of its state: floating-point
    sensitivity of a near-uniform region, not a restart error. The check stays red, unchanged, until its resolution is
    approved; the restart gate reports OpenFOAM's one-ulp response beside brae's distance at every write time.
  - restart against uninterrupted, a property of OpenFOAM's restart that brae reproduces: on calculated patches an
    uninterrupted run's rhoU_b and rhoE_b carry the step before's rho_b (sec. 4), which no file holds; a restart rebuilds
    them from the current one. After ONE step OpenFOAM's restart is 7.2e-04 (rho) from its uninterrupted run, at the
    outlet's cells; by t = 0.12, 1.1e-04 for OpenFOAM, the CUDA arm and the host arm alike (rho 5.61e-04, 5.62e-04,
    5.61e-04 absolute).
- **A start directory without its fields** is refused by name (`checkStartDirectory`, rhoCentralControls.cu): startFrom
  latestTime or firstTime selects the directory OpenFOAM's findTimes would (by its name alone), and U, p and T must be in
  it. The shared `resolveStartTime` still skips a directory without U for the other solvers; rhoCentralFoam refuses where
  that would start it from an older time (tests/rho_central_refusals.sh, three rows; OpenFOAM stops on each too).
- **Not gated:** restart with function objects (none in these tutorials), binary or compressed time directories, LTS.

## 5h. Milestone F: Riemann problems and the reconstruction limiters

- **What runs:** reconstruct(rho) and reconstruct(T) (rPsi, e, c) with vanLeer, vanAlbada, Minmod, MUSCL, SuperBee,
  limitedLinear k or upwind; reconstruct(U) (rhoU) with their V forms (NVDVTVDV) or upwind. Refused by name: the
  scalar-form limiter on U (LimitedScheme's limitFuncs::magSqr), a V form on a scalar, limitedLinear's k outside [0, 1]
  (OpenFOAM's FatalIOError), and every other scheme (QUICK, UMIST, Gamma, SFCD, filteredLinear, limitedCubic ...).
- **No solver-local limiter:** the formulas are OpenFOAM's v2412 ones (Minmod.H:84 `clamp(r, zero_one)`, MUSCL.H:84,
  SuperBee.H:85, vanAlbada.H:85, vanLeer.H:85, limitedLinear.H:100, std::min/max and Scalar.H's clamp), added to the
  shared `limiterOfR` the device already used for vanLeer, vanAlbada and limitedLinear, and made BRAE_HD so the host
  reference evaluates the limiters its own library (cpu::limitedSchemes) lacks with the same function. The host face
  loop is compiled once per kind of limiter (`VanLeerOnly`, `CaseLimiters` in rhoCentralFlux_cpp.cu): GB10's GCC
  contracts a*b + c into an FMA by the code's shape, and with the limiter chosen at run time the host arm moved by up to
  6e-15 relative on Sod and the 17-digit obliqueShock host arm of milestone C's gate went 1.02 x above its bound (t = 5,
  p 1.455e-09 against 1.430e-09). With vanLeer known at compile time, as through milestone E, every output of the Sod
  gate, CUDA and host, is byte-identical to milestone E's (a rebuild of E's commit reproduces E's files).
- **Measured** (`tests/rho_central_riemann_vs_openfoam.sh`, GB10, 2026-11-18): Toro's tests 1 to 3 and Lax's problem on
  shockTube's tube at 1000 cells (R = 1, gamma = 1.4), each with the six limiters, two with Tadmor, and obliqueShock (2-D)
  with each; OpenFOAM serial, in two (simple) and in four (frozen scotch) as the reference:
  - every limiter changes OpenFOAM's solution (by 0.05 to 0.95 against vanLeer), so none could pass by being ignored;
  - sod, lax (but SuperBee), obliqueShock (but MUSCL and SuperBee): every deltaT, time name and directory the same as
    OpenFOAM's, fields within ten times OpenFOAM's own spread at every write time, CUDA and host; the L1 errors against
    the exact solution equal OpenFOAM's (lax Minmod: rho 2.787e-03, u 1.853e-03, p 9.527e-04; strong limitedLinear 1:
    p* 460.894, u* 19.5975, Toro's);
  - Toro 2 (the 123 problem, near vacuum) breaks down in OpenFOAM v2412 itself with vanLeer, MUSCL, SuperBee and
    limitedLinear (a floating-point exception at step 24 to 27); brae stops at the same step;
  - OPEN, left red: lax SuperBee, obliqueShock MUSCL and SuperBee part from OpenFOAM serial's deltaT sequence late in
    the run -- and so do OpenFOAM's own runs: decomposed, and serial started with p one ulp up. Steps that keep serial's
    deltaT (lax SuperBee, of 1891: OpenFOAM in two 1581, in four 1395, one ulp up 1128, brae 1619; obliqueShock MUSCL,
    of 3252: 1449, 1646, 1483, brae 1476; obliqueShock SuperBee, of 3288: 787, 785, 831, brae 785). Compressive limiters
    make these runs chaotic for every implementation; the step check is not relaxed for it: a decision for review;
  - OPEN, left red: the strong shock (Toro 3) with every limiter but vanLeer, at its first write (t = 0.03), and with
    vanAlbada and Tadmor also rho at 0.06 and 0.12: brae 2e-13 to 3e-11 against bounds from OpenFOAM's decomposed
    spread of 8e-14 to 2e-11. OpenFOAM serial with p one ulp up lands, field by field and at every write, within a
    factor 3.1 of brae's distance from serial (the largest: Minmod p at 0.03, brae 2.2e-13, one ulp up 7.0e-14), and
    farther from serial than brae in 73 of the 112 field-write pairs. As with the wedge restart (sec. 5g), brae sits at
    OpenFOAM's own one-ulp response; the decompositions, whose perturbation starts at their processor faces, have not
    reached the waves yet. The envelope is not changed for it: a decision for review.
  - controls red: vanLeer whatever the case names (2.5e+09 x on sod Minmod, 1.1e+05 x on obliqueShock SuperBee); the V
    limiter of rhoU from rhoU_x alone (2.3e+09 x on obliqueShock Minmod; not visible in 1-D, where rhoU has one component).

## 5i. Milestone G: the viscous correctors

With max(mu) > 0 (createFieldRefs.H) OpenFOAM v2412 adds, in this order (rhoCentralFoam.C): tauMC =
muEff*dev2(T(fvc::grad(U))) from U as the step starts (:220-221); after the explicit rho, rhoU and U, the U corrector
solve(fvm::ddt(rho, U) - fvc::ddt(rho, U) - fvm::laplacian(muEff, U) - fvc::div(tauMC)) and rhoU = rho*U (:234-244);
sigmaDotU = (interpolate(muEff)*magSf*snGrad(U) + dotInterpolate(Sf, tauMC)) & (a_pos*U_pos + a_neg*U_neg) from the
corrected U (:246-253), whose divergence the rhoE equation subtracts (:258-263); after e, its patches and
thermo.correct(), the e corrector solve(fvm::ddt(rho, e) - fvc::ddt(rho, e) - fvm::laplacian(alphaEff, e)), then
thermo.correct() and rhoE = rho*(e + 0.5*magSqr(U)) (:273-282). alphaEff = thermo.alphahe() = CpByCpv*alpha
(heThermo.C:809-813), alpha = mu*rPr (constTransportI.H:101-108), so with constTransport and hConst both diffusivities
are constants and both laplacian matrices are built once.

What is reused, not copied (all shared, all device):
- grad(U) and grad(e): `deviceGaussGradFused`; the patch faces' gradient, `deviceBoundaryGradU` (gaussGrad's
  correctBoundaryConditions);
- muEff*dev2(T(.)): `deviceScaledDev2T`, the kernel brae's divDevReff already ran, now exported;
- the U and e matrices: `deviceLaplacianGammaMagSf` (new beside `deviceLaplacianCoeffs`: OpenFOAM's
  deltaCoeffs*(gamma*magSf) rather than (deltaCoeffs*gamma)*magSf), the corrected scheme's explicit part through
  `deviceLaplacianCorrFlux`/`deviceFaceDivSource` (skipped where every nonOrthCorrectionVector is exactly zero, where it
  is exactly zero), `deviceEulerFvmDdtRho` and `deviceEulerFvcDdtRhoSub` (new in device_ddt: Euler ddt with a per-cell
  rho, each product rounded alone), U's patch coefficients from `deviceBCLaplacianCoeffsFace` and `deviceUpdateSymmetry`,
  `deviceFold`;
- the solves: smoothSolver with GaussSeidel or symGaussSeidel through `deviceNormFactorInto` and
  `deviceSymGaussSeidelFused` -- OpenFOAM's sequential sweep order, the residual normalisation, the stop test, U's
  valid components in one fused call (byte-identical to solving them one after the other);
- div(tauMC) and div(sigmaDotU): `deviceDotInterpolateSf` (new: Sf & the linear interpolate of a tensor) and `deviceDiv`;
  snGrad(U): `deviceSnGradInternal` (new: deltaCoeffs*(U_N - U_P) plus the corrected part).
rhoCentralFoam's own: the order (`stepViscous`, which splits the inviscid step's fused cell kernel into a momentum, an
energy and a thermo-and-pressure kernel; the inviscid step itself is unchanged and its outputs byte-identical), e's
patch coefficients from T's patch fields (fixedEnergy, gradientEnergy whose gradient is 0 for a zeroGradient T,
mixedEnergy; transformFvPatchField<scalar> has none), sigmaDotU's patch faces (each patch field's snGrad) and the face
velocity a_pos*U_pos + a_neg*U_neg, which the device face flux now also returns (the host's code is untouched). The
viscous patch fluxes are counted in the conservation diagnostic.

Refused by name: no `e` entry in fvSolution (OpenFOAM stops at its first e solve; the tutorials name `h`), a solver
other than smoothSolver, a smoother other than GaussSeidel or symGaussSeidel, nSweeps < 1, schemes for grad(U),
div(tauMC), laplacian(muEff,U), laplacian(alphahe,e) (the field is named alphahe), snGrad(U) and interpolate(muEff)
other than Gauss linear, Gauss linear, Gauss linear corrected (twice), corrected and linear; sutherland; a wedge patch
with mu > 0 (wedgeFvPatchField's snGrad takes cellT); and the host reference arm with mu > 0 (milestone G ports the
device arm).

The gate, tests/rho_central_viscous_vs_openfoam.sh, declares its solve requirement before comparing anything: the
tutorials' fvSolution (U: tolerance 1e-9, relTol 0.01; e: 1e-10, 0; GaussSeidel, nSweeps 2) and a converged Sod
(1e-13, 0), with OpenFOAM serial's iteration count required at every solve and the residuals reported. Cases: a shear
layer (momentum diffusion alone), conduction in a tube at rest (energy diffusion alone), Sod and obliqueShock with mu > 0.
Controls: noTauMC, noSigmaDotU.

Found on the gate's first run: the round-off conservation check every gate applies fails on every viscous case for
momentum and energy (mass closes). An implicit corrector solved to a tolerance leaves r = b - A x, and the totals move by
-deltaT*sum(r): OpenFOAM's algorithm does the same. brae now accumulates that remainder (`diag solve-residual`), and the
balance plus it closes to round-off on every case (Sod momentum: -4.560590e-04 and +4.560590e-04). The round-off check
stays red, registered as "linear-solver tolerance", with the residual-counted check beside it; which one a viscous
case should carry is a decision for review.

Open validation items (all gates; tests/data/rho_central_open_checks.tsv): every FAIL a gate prints is followed by its
category, and a FAIL the registry does not name prints as an unexplained numerical difference. The categories and what
is in each (none is turned into a pass, no threshold moved):
- round-off amplification: the restart arm wedge15Ma5 from OpenFOAM's t = 0.06 at t = 0.12 (E); the strong shock with
  every limiter but vanLeer near its first write (F). OpenFOAM serial with p one ulp up, now run beside serial, in two
  and in four on every case with an open check, lands as far from serial as brae does.
- timestep sequence divergence: lax SuperBee, obliqueShock MUSCL and SuperBee (F). The Riemann gate now runs them with
  the step fixed as well, and reports where each run's deltaT parts from serial's, the step each run moves 1e-12, 1e-8
  and 1e-4 away, and the shock each run carries.
- OpenFOAM reference failure: no FAIL (Toro 2, where OpenFOAM v2412 stops, is checked as brae stopping too).
- unexplained numerical difference: none registered.

## 5j. Milestone H: the nozzle boundaries

totalPressure (the isentropic psi form), totalTemperature and waveTransmissive (on p) evaluate where OpenFOAM's
updateCoeffs runs: totalTemperature at every Tw.evaluate() a fixedEnergy patch makes (e.correctBoundaryConditions() and,
viscous, the e matrix's construction), totalPressure and waveTransmissive at p.correctBoundaryConditions(). They live in
the shared device_boundary_flow.cu but are not in the shared factory's list, so the other solvers keep refusing them.
noSlip, sutherland transport and the viscous correctors on wedge patches came with LadenburgJet60psi.
tests/rho_central_nozzle_vs_openfoam.sh.

## 5k. Milestone I: local time stepping, janaf and the rarefied-gas walls (biconic25-55Run35)

- **LTS** (`Case::lts`, ddtSchemes `default localEuler`; a mix with Euler terms is refused). The device arm's
  `setRDeltaT()` runs after the fluxes, as setRDeltaT.H runs after centralCourantNo.H: surfaceSum(amaxSf) (the cell
  kernels' gather, run ahead of them), max(1/maxDeltaT, sum/((2*maxCo)*V)), then brae's shared `deviceSmooth` (the
  FaceCellWave of fvc::smooth, in its own visiting order, to the bit of its host oracle). The cells' explicit updates
  and the viscous correctors read the per-cell rDeltaT (shared `deviceLocalEulerFvmDdtRho` / `FvcDdtRhoSub`, Euler's
  expressions with the field in the scalar's place). No setDeltaT; the clock advances by the deltaT it holds.
  rDeltaT is written (extrapolatedCalculated, through the shared writer's `DerivedFieldSpec::patchType`). The host
  reference arm refuses LTS.
- **janaf** (`GasCoeffs::janaf`): Cp, Hs = Ha - Hc and limit() as janafThermoI.H writes them, the coefficients times R,
  T(e) by the Newton loop on Cv(T); every quantity OpenFOAM takes from thermo.Cp()/Cv() is taken per cell and per face
  (the sound speed, sutherland's kappa, alphahe's CpByCpv). OpenFOAM v2412 builds janaf psiThermo with sutherland
  transport only, and so does this port.
- **maxwellSlipU and smoluchowskiJumpT** (rhoCentralSlipBCs.cu, as OpenFOAM keeps them in rhoCentralFoam/BCs).
  smoluchowskiJumpT is a mixed T patch; e takes mixedEnergy's coefficients from it. maxwellSlipU is a partialSlip,
  carried in brae's per-component mixed descriptors the way slip and wedge are (valueFraction_k = snGradTransformDiag_k,
  refValue_k rebuilt from the cells' U before each use) and evaluated exactly. Both update where OpenFOAM updates them
  (header of rhoCentralSlipBCs.cuh).
- **Found on biconic25-55Run35's first step**: OpenFOAM's build constructs `fvm::ddt(rho, e)` -- whose fvMatrix
  constructor's updateCoeffs re-evaluates every T patch on the cells' T of the first thermo.correct() -- before it calls
  `turbulence->alphaEff()`, so alphahe()'s CpByCpv reads those T_b. With janaf this moves e's matrix (4.4e-7 in its
  initial residual); the port follows it (control alphaheOldT).
- **Found on reservoir_visc (task 1)**: OpenFOAM's aarch64 build fuses lduMatrix::residual's multiply-subtracts; the
  shared device residual rounded them apart (x86-64 OpenFOAM's form). `setDeviceResidualFused` (default off, every
  other solver unchanged); rhoCentralFoam sets it on aarch64.

tests/rho_central_lts_vs_openfoam.sh: biconic25-55Run35 as shipped, LTS on forwardStep and the viscous obliqueShock,
janaf tubes across Tcommon, a rarefied channel; controls ltsUnsmoothed, janafLowOnly, alphaheOldT, maxwellNoCreep,
maxwellNoCurvature, smoluchowskiNoJump.

## 5l. Milestone J: compressible RAS (kOmegaSST)

- **Layering, as OpenFOAM stacks it** (src/TurbulenceModels/compressible): `compressibleRAS_cpp` reads what
  compressible::turbulenceModel::New(rho, U, phi, thermo) reads -- turbulenceProperties, the RAS dictionary and its
  coeffs (Prt from kOmegaSSTCoeffs), k omega nut alphat with their patch classes, meshWave and nearWallDist -- and
  refuses by name what it does not run. `gpu::compressibleRAS::KOmegaSST` is the CompressibleTurbulenceModel /
  eddyViscosity / EddyDiffusivity layer: muEff = rho*nut + mu, alphat = rho*nut/Prt, phi() = phi/interpolate(rho),
  nu() = mu/rho, the constraint patches after correctNut. The model itself is brae's shared kOmegaSSTBase closure
  (kOmegaSST.cuh; its correctNut is now callable alone, for validate()), the one every compressible solver of
  OpenFOAM's runs. Nothing in the layer knows which solver calls it.
- **Where rhoCentralFoam calls it** (the solver's own lines, not another brae solver's driver): validate() at :71 after
  construction; muEff() at :220 into the U corrector (beginStep: rho as the step starts, the nut the last correct()
  left); alphaEff() at :278 through the thermo (CpByCpv*(alpha + alphat)); correct() at :290 after p, p_b and rho_b, on
  the step's central mass flux and the laminar mu of the last thermo.correct(), fvm::ddt(rho, .) on rho.oldTime() with
  the case's ddt scheme (Euler or localEuler). Inviscid is still decided by the laminar mu (:216), and a RAS case with
  a laminar mu of 0 is refused (its muEff and alphaEff would reach no equation). The host reference arm refuses RAS.
- **Case design found on the way**: a flat plate staged at maxCo 0.4 (the jet's own is 0.5) grows a grid-scale wave
  train from the leading edge in OpenFOAM itself, laminar and RAS alike, on its 33:1 near-wall cells; OpenFOAM's two-
  and four-way runs then part from its serial by 10% by 5e-4 s. At maxCo 0.2 the rows stay smooth and the spread grows
  slowly (4e-6 at 5e-4 s, 2e-3 at 3e-3 s): the gate's plate runs at 0.2.

tests/rho_central_turb_vs_openfoam.sh: a supersonic turbulent flat plate with wall functions (Euler and localEuler) and
LadenburgJet60psi with kOmegaSST; k omega nut alphat and the plate's wall values against OpenFOAM's own spread;
controls rasLaminarMuEff, rasNoAlphat, rasCorrectFirst, rasPhiUndivided.

## 5m. Milestone K: mesh motion (movingCone)

- **mesh.update() where rhoCentralFoam.C calls it** (after setDeltaT.H and ++runTime, :96; never under LTS, which is
  refused with a dynamicMeshDict): brae's shared `DynamicMotionSolverFvMesh` (V0 once a time index, oldPoints, meshPhi
  from face.C's swept volumes, the geometry and the patches rebuilt in place) driving a new shared
  `VelocityComponentLaplacianFvMotionSolver` (src/fvMotionSolver/fvMotionSolvers/componentVelocity/laplacian), ported
  from OpenFOAM's: the directional diffusivity, pointMotionU<c>'s fixed values written in patch order, cellMotion's face
  average of the point field, the case's `laplacian(diffusivity,cellMotionU)` (Gauss linear uncorrected) and PCG/DIC
  solve, volPointInterpolation in OpenFOAM's pointCells order, points[c] += deltaT*pointMotionU<c>, and the twoD
  correction with snapToWedge (twoDPointCorrector gained the wedge branch, wedgePolyPatch's own normal arithmetic).
  Not pimpleFoam's helper: it maps the solve's scheme, tolerance and wedge correction differently. The shared class
  takes the velocity solver only when its caller asks (rhoCentralFoam), so its other users are unchanged.
- **What the arm holds of the geometry is refreshed after each move**: the device mesh (in place, sec. 5n), the
  boundaries' areas, deltaCoeffs and normals (the builders' arithmetic, uploaded in place, sec. 5n), V0 and meshPhi for the kernels,
  and the viscous correctors' constant-mu laplacian and non-orthogonal switch (the moved cone's cells are no longer
  orthogonal). Then U.correctBoundaryConditions(), as dynamicMotionSolverFvMesh::update ends.
- **The fluxes and the ddt**: phiv_pos/neg -= meshPhi (:126-133), phiEp += meshPhi*(a_pos*p_pos + a_neg*p_neg) (:213-218);
  fvm::ddt's old-time source on V0 in the explicit updates and the correctors (fvc::ddt(rho, .) =
  rDeltaT*(rho*vf - rho0*vf0*V0/V)); a static mesh takes V for V0 and no meshPhi, its arithmetic unchanged.
- **The boundaries movingCone needs**: movingWallVelocity (Up from face::centre at the old and new points, Un =
  meshPhi_b/(magSf + VSMALL)), pressureInletOutletVelocity (the shared directionMixed update, at the move and at :232),
  totalPressure's rho form (p0 - 0.5*rho_b*neg(phi)*|U_b|^2; psi none).
- **Written and read back**: polyMesh/points, meshPhi, cellMotionUx, pointMotionUx; a restart takes its points from the
  start time's own instance (polyMesh's pointsInstance) and cellMotionUx as the first solve's initial guess.

tests/rho_central_moving_vs_openfoam.sh: movingCone as shipped, with tight solves, and restarted from OpenFOAM's 1e-5;
controls motionFluxAbsolute, motionDdtStatic, motionWallStatic, motionPiovStatic.

## 5n. Performance with the numerics unchanged (after K)

Each change below leaves every gate's written files, the log and the deterministic `diag` lines bit-identical. The two
suites, before and after, were compared file by file: 12,814 files, none different. The rule: a change that moves a bit
is not a performance change.

- **mesh.update() is timed on its own.** `diag step-seconds` never included it; `diag motion-seconds` now reports it.
  `BRAE_INTER_PHASE_TIME=1` turns on the shared interPhase timers in the loop, the viscous step, the U solve,
  mesh.update() and the motion solver. They are off by default, and synchronise at each phase boundary.
- **The moved geometry is uploaded, not rebuilt.**
  - The boundaries' deltaCoeffs, magSf, normals and symmetry normals come from `deviceBoundaryGeometry` (shared,
    device_boundary.cuh). It follows the builders' layout and arithmetic, and writes into the existing device
    boundaries.
  - `buildDeviceMesh`'s geometry loops are split out (`deviceMeshGeometryHost`). The builder and
    `refreshDeviceMeshGeometryInPlace` therefore compute the same values, and the in-place refresh uploads into the
    existing buffers. It is opt-in: brae's other moving solvers still call refreshDeviceMeshGeometry.
  - `PackedUpload` (shared, device_buffer.cuh) packs those arrays into one pinned host block. It then makes one
    host-to-device copy and slices it on the device, all on the per-thread stream.
- **The viscous accounting stays on the device.** Milestone G's solve remainder and identity took five blocking
  reductions per solve. They are now device-scalar reductions with the same kernels (`deviceDotInto`,
  `deviceSumMagInto`). Each result is logged on the device and folded on the host, in call order, when read.
- **fvc::smooth chooses its arm by measurement.** OpenFOAM's smoothData stops propagating below a tolerance, so the
  visiting order is part of the result. The device wave keeps that order, but on 32k cells it is bound by latency.
  The first call runs the host oracle and the device wave on the same field. It refuses unless the two agree to the
  bit, then keeps the faster. `BRAE_RCF_SMOOTH=host|device` forces one.

Still open: the small cases are bound by kernel-launch latency, and under adjustTimeStep by setDeltaT's per-step
host round trip. CUDA Graphs or a device-side setDeltaT would be the next step.

## 5o. The audit before the pull request (2026-11-19)

Two independent read-throughs of the branch against OpenFOAM v2412 and origin/main. The order of operations and the
arithmetic of rhoCentralFoam.C, its includes, the thermo, LTS and mesh motion match. Found and changed:

- Line citations of rhoCentralFoam.C (220-221 tauMC, 224 the rho solve, 240-241 the U corrector's terms, 251-254
  sigmaDotU, 278 alphaEff), createFields.H, LimitedScheme.C, localEulerDdt.C, hConstThermo.C, janafThermoI.H,
  partialSlipFvPatchField.C, symmetryPlaneFvPatchField.C and smoluchowskiJumpTFvPatchScalarField.C had drifted by
  one to a few lines, or pointed at a neighbouring function; corrected.
- The wedge branch of the shared twoDPointCorrector lacked wedgePolyPatch.C's two FatalErrors (a centre plane off
  the coordinate planes, a wedge plane on one); added. displacementLaplacian, the other caller, still refuses wedge
  patches before it builds the corrector, so interFoam's behaviour is unchanged.
- The shared field reader read `fieldInf` as a scalar for every field; a vector's `( ... )` now skips as it did before
  the key was read, so a vector waveTransmissive is refused by name, not by a parse error.
- `diag inletOutlet` counted a viscous step's face-steps twice (the patches update twice there); counted once.
- The writeFormat-binary notice said "12 significant digits" for every solver; it held for rhoSimpleFoam alone
  (write_control.cuh).

Kept, as deliberate differences, and said here:

- **brae stops on an unphysical state** (rho, p or T not positive, or e not finite: physicalStatus,
  rhoCentralCellUpdate.cuh) where OpenFOAM would go on until its thermo's own FatalErrors (thermoI.H: a negative
  temperature seed, the e -> T inversion not converging). A run that passes is unaffected; the gates' controls rely
  on the stop (tests/rho_central_gate_lib.sh, red()), as simpleFoam and pimpleFoam stop on a non-finite residual. The
  device arm reads the status when it flushes its window (256 steps, or at a write, or every step under
  adjustTimeStep), so a failing run stops up to that many steps after OpenFOAM's step would have; nothing past it is
  written.
- **janaf's limit() is silent.** OpenFOAM prints a Warning on every clamp of T to [Tlow, Thigh]; brae clamps alike and
  prints nothing (rhoCentralThermo.cuh).
- **A solver other than `diagonal` for rho, rhoU or rhoE is refused**, where OpenFOAM would use diagonalSolver for any
  name; stricter, never different.
- **courantKernel is one block of 1024 threads**, so the Courant reduction sits on one SM. The max, which deltaT
  reads, is exact in any order; the two sums behind the logged mean Courant number keep one fixed order (1024 strided
  partials, then a tree), so the log is the same from run to run, though not OpenFOAM's sequential sum to the last
  bit. A cost on large meshes (sec. 5n's open items), not a numerical difference.

## 6. Milestones, each closed by a gate against OpenFOAM v2412

Every gate:
- compares brae with OpenFOAM on the same case (same schemes, BCs and time step), at every write time;
- carries a control that must turn red;
- reports conservation, positivity, GPU memory and runtime.

| | milestone | gate (cases) |
|---|---|---|
| A | restructure into the modules above; brae BC classes; shared limiter library; driver + Stepper | the Sod gate and the refusals unchanged |
| B (done) | adjustTimeStep: setDeltaT, adjustable writes, first step x1.2, restart deltaT | shockTube **as shipped**: the deltaT sequence and fields |
| C (done) | constraint and transform BCs: symmetryPlane, symmetry, slip, wedge, with the conservative-state patch semantics of sec. 4 | wedge15Ma5 as shipped; obliqueShock as shipped (shock angle against theta-beta-M); a symmetry and a slip variant; an axisymmetric cone on wedge patches |
| D (done) | inletOutlet (U, p, T; T -> mixedEnergy) and purgeWrite | forwardStep as shipped; the inlet as inletOutlet; a tube whose end turns from outflow to inflow |
| E (done; one arm red, for review) | restart: startFrom latestTime/startTime, `uniform/time` read and written, inletOutlet's value | restarts from OpenFOAM's and brae's output against OpenFOAM's own restart, and against the uninterrupted run |
| F (done; open checks for review) | the reconstruction limiters vanAlbada, Minmod, MUSCL, SuperBee, limitedLinear and their V forms, upwind | Sod, Lax, 123 and a strong shock tube with every limiter, obliqueShock in 2-D: brae against OpenFOAM and the exact solution |
| G (done; for review) | viscous: muEff, tauMC, U and e correctors (smoothSolver/GaussSeidel), sigmaDotU, constant mu (sutherland and noSlip are not in this milestone) | a shear layer, conduction, Sod and obliqueShock with mu > 0 against OpenFOAM |
| H (done; open checks registered) | totalPressure (psi branch), totalTemperature, waveTransmissive, noSlip, sutherland, viscous wedge | LadenburgJet60psi as shipped and nozzle tube cases |
| I (done) | LTS (setRDeltaT, fvc::smooth), janaf, maxwellSlipU, smoluchowskiJumpT | biconic25-55Run35 as shipped, LTS/janaf/wall cases |
| J (done; open checks registered) | compressible RAS on the central path: kOmegaSST, its wall functions, muEff and alphaEff, Euler and localEuler | a turbulent flat plate (Euler, LTS) and LadenburgJet60psi with kOmegaSST |
| K (done; open check registered) | mesh motion: dynamicMotionSolverFvMesh with velocityComponentLaplacian, meshPhi, V0, movingWallVelocity, pressureInletOutletVelocity, totalPressure's rho form, restart | movingCone as shipped, with tight solves and restarted |

The solver is labelled **experimental** (the registry row in solver_dispatch.cuh, a line on stderr at start-up, the
build target's comment). A to K are done and all seven v2412 tutorials run as shipped within the gates' bounds, but it
stays experimental while registered open checks remain (tests/data/rho_central_open_checks.tsv, VALIDATION.md), while
its 3-D coverage is young, and while it runs on one GPU only.
Multi-GPU is designed for, not built: the Stepper's patch evaluation and gradient stages are where halo exchanges will
go.
