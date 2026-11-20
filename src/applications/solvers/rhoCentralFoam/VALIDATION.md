# rhoCentralFoam: validation status and the registered open checks

Every gate compares brae with OpenFOAM v2412 run on the same machine, the same case, schemes, boundary conditions and
time step, at every write time. A field passes when its largest relative distance from OpenFOAM serial is within ten
times OpenFOAM's own spread: the larger of OpenFOAM decomposed in two (simple) and in four (a frozen scotch
partition, `tests/data/rho_central_decomposition`), against OpenFOAM serial. The floor is 8e-14, or 1e-6 for files
written at six digits. Each gate also runs a negative control that must turn red by more than 100 x the bound. The
bound is never widened to make a check pass.

## The gates (the suite at 4d951925, 2026-11-19)

| gate (`tests/`) | what it holds to OpenFOAM | passed | failed | ctest labels |
|---|---|---|---|---|
| rho_central_sod_vs_openfoam | shockTube, fixed deltaT; the exact Riemann solution | 23 | 0 | slow, rhocentral |
| rho_central_shocktube_vs_openfoam | shockTube as shipped (adjustTimeStep) | 60 | 0 | slow, rhocentral |
| rho_central_wedge_vs_openfoam | wedge15Ma5, obliqueShock as shipped; symmetry, slip, cone | 127 | 0 | slow, rhocentral |
| rho_central_forwardstep_vs_openfoam | forwardStep as shipped; inletOutlet, purgeWrite | 94 | 0 | slow, rhocentral |
| rho_central_restart_vs_openfoam | restarts from OpenFOAM's and brae's output | 196 | 1 | slow, rhocentral, known-red |
| rho_central_riemann_vs_openfoam | Sod, Lax, 123, Toro 3 with every limiter; obliqueShock | 221 | 16 | slow, rhocentral, known-red |
| rho_central_viscous_vs_openfoam | shear layer, conduction, viscous Sod and obliqueShock | 62 | 0 (5 historical) | slow, rhocentral |
| rho_central_nozzle_vs_openfoam | LadenburgJet60psi as shipped; nozzle tubes | 84 | 3 | slow, rhocentral, known-red |
| rho_central_lts_vs_openfoam | biconic25-55Run35 as shipped; LTS, janaf, slip walls; read back | 68 | 0 | slow, rhocentral |
| rho_central_turb_vs_openfoam | kOmegaSST flat plate (Euler, LTS); turbulent jet | 36 | 2 | slow, rhocentral, known-red |
| rho_central_moving_vs_openfoam | movingCone as shipped, tight solves, restarted; read back | 30 | 1 | slow, rhocentral, known-red |
| rho_central_3d_vs_openfoam | 3-D: axial corner (inviscid, laminar, kOmegaSST), closed-box blast and its restarts, totalPressure duct | 75 | 1 | slow, rhocentral, known-red |
| rho_central_refusals | everything outside the scope stops by name | 46 | 0 | rhocentral |
1,122 checks pass. The 24 that fail are all registered open checks; none is on a tutorial as shipped. The 5
historical lines of the viscous gate are the raw conservation imbalance of a solve stopped at its tolerance, kept in
the log; the acceptance there is the residual-accounted balance.

## The registered open checks

`tests/data/rho_central_open_checks.tsv` lists each one with its analysis. `tests/rho_central_gate_lib.sh` prints the
category under the matching FAIL. It never turns a FAIL into a pass, and no threshold reads the file. The gates that
carry one stay red, so ctest labels them `known-red`. The GB10 CI (`.github/workflows/gpu-test.yml`) leaves them
out of its ctest step (`-LE known-red`) and runs them in a step of their own through `tools/known_red_check.py`.
That step passes only when what is red is exactly these 24:
- no FAIL outside the registry;
- each registry row matching the number of FAILs that `tests/data/rho_central_known_red.tsv` gives it;
- each gate passing its expected number of checks.

A new failure, a registered check going green, or an arm that stops running fails that step until the registry,
the expectations and this table change together. `ctest -L known-red` runs the six as they are.

| # | gate | case (arm) | what the gate measured (brae > bound) | category |
|---|---|---|---|---|
| 1 | restart | restart wedge17_0.06o (CUDA) | fields above the bound at 0.12:U(5.939e-10>2.963e-10) 0.12:p(3.399e-10>2.284e-10) 0.12:rho(6.181e-10>3.437e-10) | round-off amplification |
| 2 | riemann | lax_SuperBee (CUDA) | 1891 steps against 1891; names 1734; deltaT same 1619 (worst 1.459e-02 at 1827) | timestep sequence divergence |
| 3 | riemann | strong_vanAlbada (CUDA) | fields above the bound at 0.03:U(7.718e-12>4.534e-12) 0.03:p(5.666e-12>3.459e-12) 0.03:T(1.719e-12>1.021e-12) 0.03:rho(1.062e-12>8.044e-13) | round-off amplification |
| 4 | riemann | strong_vanAlbada (host reference) | fields above the bound at 0.03:U(7.717e-12>4.534e-12) 0.03:p(5.666e-12>3.459e-12) 0.03:T(1.719e-12>1.021e-12) 0.03:rho(1.062e-12>8.044e-13) | round-off amplification |
| 5 | riemann | strong_Minmod (CUDA) | fields above the bound at 0.03:p(2.158e-13>1.853e-13) 0.03:rho(1.109e-12>8.000e-14) | round-off amplification |
| 6 | riemann | strong_Minmod (host reference) | fields above the bound at 0.03:p(2.099e-13>1.853e-13) 0.03:rho(1.109e-12>8.000e-14) | round-off amplification |
| 7 | riemann | strong_MUSCL (CUDA) | fields above the bound at 0.03:rho(1.577e-12>4.259e-13) | round-off amplification |
| 8 | riemann | strong_MUSCL (host reference) | fields above the bound at 0.03:rho(1.577e-12>4.259e-13) | round-off amplification |
| 9 | riemann | strong_SuperBee (CUDA) | fields above the bound at 0.03:U(4.621e-13>4.262e-13) 0.03:rho(1.143e-12>7.355e-13) | round-off amplification |
| 10 | riemann | strong_SuperBee (host reference) | fields above the bound at 0.03:U(4.561e-13>4.262e-13) 0.03:rho(1.143e-12>7.355e-13) | round-off amplification |
| 11 | riemann | strong_limitedLinear_1 (CUDA) | fields above the bound at 0.03:U(1.961e-12>4.447e-13) 0.03:p(1.207e-12>3.558e-13) 0.03:T(4.318e-13>1.586e-13) 0.03:rho(1.072e-12>1.468e-13) | round-off amplification |
| 12 | riemann | strong_limitedLinear_1 (host reference) | fields above the bound at 0.03:U(1.954e-12>4.447e-13) 0.03:p(1.219e-12>3.558e-13) 0.03:T(4.374e-13>1.586e-13) 0.03:rho(1.072e-12>1.468e-13) | round-off amplification |
| 13 | riemann | strong_vanAlbada_T (CUDA) | fields above the bound at 0.03:U(2.486e-12>4.064e-13) 0.03:p(1.882e-12>1.120e-13) 0.03:T(5.423e-13>8.000e-14) 0.03:rho(1.582e-12>2.025e-13) 0.06:rho(7.380e-12>4.882e-12) 0.12:rho(2.644e-11>1.647e-11) | round-off amplification |
| 14 | riemann | strong_vanAlbada_T (host reference) | fields above the bound at 0.03:U(2.487e-12>4.064e-13) 0.03:p(1.882e-12>1.120e-13) 0.03:T(5.429e-13>8.000e-14) 0.03:rho(1.582e-12>2.025e-13) 0.06:rho(7.380e-12>4.882e-12) 0.12:rho(2.644e-11>1.647e-11) | round-off amplification |
| 15 | riemann | ob_MUSCL (CUDA) | 3252 steps against 3252; names 1609; deltaT same 1476 (worst 1.359e-02 at 2532) | timestep sequence divergence |
| 16 | riemann | ob_SuperBee (CUDA) | 3289 steps against 3288; names 796; deltaT same 785 (worst 4.725e-02 at 3015) | timestep sequence divergence |
| 17 | riemann | ob_SuperBee (host reference) | 3289 steps against 3288; names 802; deltaT same 788 (worst 4.378e-02 at 1561) | timestep sequence divergence |
| 18 | nozzle | ladenburg_long (CUDA) | max Courant number 4.709e-11 at step 36355, above 6.701e-12 | round-off amplification |
| 19 | nozzle | ladenburg_long (CUDA) | fields above the bound at 8e-05:U(1.344e-11>6.721e-12) 0.0001:U(6.180e-11>1.057e-11) | round-off amplification |
| 20 | nozzle | reservoir_visc (CUDA) | iteration counts differ from solve 5570 (9277 of 9278 the same) | tolerance-limited |
| 21 | turb | plate_sst (CUDA) | iteration counts differ from solve 35 (107430 of 107925 the same) | tolerance-limited |
| 22 | turb | plate_lts (CUDA) | iteration counts differ from solve 290 (7499 of 7500 the same) | tolerance-limited |
| 23 | moving | mc_tight (CUDA) | iteration counts differ from solve 1 (43264 of 43265 the same) | tolerance-limited |
| 24 | 3d | duct_tp (CUDA) | iteration counts differ from solve 76 (5619 of 5620 the same) | tolerance-limited |
What the categories mean, and what OpenFOAM's own runs do on the same cases:

- **Round-off amplification (rows 1, 3-14, 18-19).** A difference at round-off grows where the flow amplifies it.
  OpenFOAM's own runs show the same growth when they are perturbed.
  - **Row 1:** wedge15Ma5 is restarted from OpenFOAM's t = 0.06. The difference sits in one free-stream cell where
    vanLeer acts on neighbour differences of 4e-10 to 2e-9. OpenFOAM serial with p one ulp up lands 6.13e-10 from
    itself there; brae lands 6.18e-10 away.
  - **Rows 3-14:** Toro's test 3 (p 1000 against 0.01) with a limiter other than vanLeer, near its first write.
    brae, on both arms, is within 3.1 x of OpenFOAM serial's response to a one-ulp perturbation. OpenFOAM's
    decompositions perturb only at their processor faces and have not reached the waves yet, so the bound they set
    is tighter than the case's own sensitivity.
  - **Rows 18-19:** LadenburgJet60psi is run five times past its endTime. U_z, the component out of the wedge's
    plane, is zero by symmetry and grows about 4.6 x every 2e-5 s in an unstable mode near the Mach disk.
    - The in-plane U stays 2.85e-13 from OpenFOAM serial; OpenFOAM in two is 2.84e-13 away.
    - Seeded with U_z = +-1e-3, the mode grows identically in both codes (within 8.5e-11).
    - OpenFOAM's own frozen four-way run grows it at the same rate.
- **Timestep sequence divergence (rows 2, 15-17).** With adjustTimeStep, the deltaT sequence parts from OpenFOAM
  serial's after about 800 to 1,600 steps. OpenFOAM's runs in two, in four, and with p one ulp up part from it too,
  some earlier (in four at step 1,396 on Lax with SuperBee, one ulp up at 1,117). With the step fixed, the same cases
  pass (`lax_SuperBee_fixed`, `ob_MUSCL_fixed`, `ob_SuperBee_fixed`).
- **Tolerance-limited (rows 20-24).** An iterative solve stops one sweep apart from OpenFOAM's because its final
  residual is within round-off of the declared tolerance. The fields stay within the bound at every write. OpenFOAM's
  own runs flip more solves than brae does:
  - **Row 20:** 1 of 9,278 energy solves. OpenFOAM stopped at 9.992e-11 and brae at 1.0002e-10, against 1e-10.
    OpenFOAM's one-ulp run flips 1 solve, and its run in four flips 44.
  - **Row 21:** 495 of 107,925 k and omega solves. OpenFOAM's own runs flip 573 to 689.
  - **Row 22:** 1 of 7,500 solves. OpenFOAM's serial run stopped at 0.993 of its tolerance.
  - **Row 23:** the first mesh-motion PCG at the gate's tight 1e-14. brae takes 56 iterations where OpenFOAM takes
    55. Every later solve, 43,264 of them, takes OpenFOAM's count.
  - **Row 24:** 1 of 5,620 solves of the 3-D totalPressure duct, step 19's energy solve at 1e-10. OpenFOAM serial
    stops at 9.81e-11; OpenFOAM in two at 9.92e-11; OpenFOAM in four reaches 1.0079e-10 and runs to maxIter 1000, as
    brae does at 1.0150e-10. Every field is within 0.402 of the bound at every write.

None of these is resolved by a wider bound or a clamp. Each would need a reason to change the gate itself, such as a
different reference spread for amplifying cases, and that is a decision for review, not something the gate does
silently.

## The shared field writer, read back

rhoCentralFoam writes through the field writer that simpleFoam, pimpleFoam and rhoSimpleFoam share
(`src/finiteVolume/fields/foam_field_writer.cuh`). `tests/total_bc_roundtrip_vs_openfoam.sh` (ctest
`total_bc_roundtrip_vs_openfoam`, 63 checks) covers all four solvers and the totalPressure family (totalPressure,
uniformTotalPressure, fanPressure), totalTemperature, waveTransmissive and phi on a symmetry patch. It holds four
things against real OpenFOAM:
- brae's patch entries equal OpenFOAM's on the same case;
- OpenFOAM starts from brae's output and keeps them;
- brae starts from OpenFOAM's output;
- brae restarted from its own output matches its uninterrupted run.

Each arm has a control: the file as an earlier brae wrote it. OpenFOAM must refuse each one. brae must refuse the
uniformTotalPressure and fanPressure ones, and read a totalPressure's `inletValue` as p0, with a notice. ARCHITECTURE.md
sec. 5r lists what the gate found and fixed. On the build before those fixes, 22 of the 63 checks fail.

How to restore a time directory an older brae wrote is in
[docs/restart-older-output.md](../../../../docs/restart-older-output.md). Each edit there was tested on such a directory:
- pimpleFoam's TJunction, with its table and with the fan, restarts identically to the uninterrupted run;
- OpenFOAM starts from each restored file.
