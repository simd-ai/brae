# rhoCentralFoam port

OpenFOAM v2412 `applications/solvers/compressible/rhoCentralFoam`, density-based and explicit. It advances rho, rhoU
and rhoE with central-upwind fluxes (Kurganov-Noelle-Petrova, or Kurganov-Tadmor), built from TVD-limited directed
reconstructions, and gets p = rho/psi. There is no pressure equation, so nothing here reuses SIMPLE or PIMPLE.

## Stage 1 (2026-11-16): the minimum solver, gated on Sod

The scope covers:

- single GPU and a static mesh;
- hePsiThermo, pureMixture, const, hConst, perfectGas, sensibleInternalEnergy;
- laminar flow with every mu = 0 (inviscid);
- a fixed deltaT and Euler time stepping;
- fluxScheme Kurganov or Tadmor;
- `reconstruct(rho|T) vanLeer` and `reconstruct(U) vanLeerV`, with Gauss linear gradients;
- zeroGradient, fixedValue and empty patches on U, p and T.

Everything else is refused at start-up, by name (`readRhoCentralCase`, `tests/rho_central_refusals.sh`).

The layout is in ARCHITECTURE.md (milestone A, 2026-11-17: the stage-1 files were split into controls,
createFields, flux, cell and patch updates and the two arms, on brae's shared gradient, limiter and boundary code,
with every Sod output byte-identical).

### Measured (GB10, 2026-11-16)

- **Against OpenFOAM, shockTube with `adjustTimeStep no`, deltaT 1e-5, 700 steps, written at 17 digits.** The
  relative L2 on U, p, T and rho is 7.7e-15 for the CUDA arm and 7.8e-15 for the host arm, at the worst of 7
  write times. Tadmor gives 7.1e-15. The 1000-cell tube over 7000 steps gives 1.9e-14. The logged Courant numbers
  equal OpenFOAM's to every printed digit.
- **1,024,000-cell 3D tube (1000x32x32 hexes), 100 steps.** brae is 2.7e-15 from OpenFOAM. A step takes
  17.4 ms, which is 5.9e7 cell-steps/s on 1.19 GiB of device memory. OpenFOAM takes 986 ms/step serial and 167
  ms/step on 16 MPI ranks (scotch), so brae is 57 x faster than one core and 9.6 x faster than 16 cores.
- **Exact Riemann solution (Toro ch. 4, gamma 1.400198) at t = 0.007.** On 100 cells the shock is 0.21 cells
  and the contact 0.24 cells from exact. On 1000 cells they are 0.17 and 0.03. L1(rho) falls from 6.28e-03 to
  7.52e-04. OpenFOAM's L1 and positions are the same to 10 digits.
- **Conservation.** The change in total mass and energy plus the boundary outflow is 7e-15 relative on 100 cells
  and 2e-14 on 1000. The momentum change balances the end-wall impulse (2520 N s) to 4e-15.
- **Controls, all red against OpenFOAM.**
  - Limiter forced to 0 (first-order upwind): 9.8e-02. Its L1(rho) of 2.2e-02 is 3.5x worse.
  - phiEp without `aSf*p_pos - aSf*p_neg`: 6.2e-03.
  - Tadmor held against Kurganov: 8.8e-03.

### Findings worth carrying

- **T is re-derived from he even at construction.** hePsiThermo's constructor runs `calculate()`, which calls
  `THE(he, p, T)` on every cell, so the starting T is not the one setFields wrote. The Newton loop's tolerance
  comes from the seed (`T0*1e-4`, thermoI.H:64), and Es carries `p/(p/(R*T))`. brae's other gas paths use the
  closed-form inverse; this solver runs the loop.
- **The gradientEnergy patch evaluates T's patch from the cell before `thermo.correct()`**
  (`Tw.evaluate()` in its updateCoeffs). So a zeroGradient-T patch inverts e -> T from the cell's OLD T, not the
  patch's last T.
- **rho's `calculated` patches are only reset at rhoCentralFoam.C:288.** rhoU_b and rhoE_b are therefore built
  from the previous step's rho_b. On a constraint patch (symmetryPlane, wedge) rho's patch is NOT calculated:
  `correctBoundaryConditions()` after the rho solve resets it to the new cell value. This has to be honoured when
  those patches are ported.
- **Per-step synchronisation was the GPU cost, not arithmetic.** Reading the Courant number and status back
  every step cost 2 x 180 us against 84 us of kernels on 100 cells. They are now kept on the device and read
  every 256 steps and at each write: 455 -> 114 us/step, with an identical log.

### Open, in order

1. symmetryPlane and slip on U, p and T, with the constraint-patch rho/rhoU/rhoE behaviour above. Gate:
   wedge15Ma5 (fixed deltaT as shipped), obliqueShock and forwardStep against OpenFOAM, and the shock angle
   against theta-beta-M.
2. setDeltaT (adjustTimeStep, maxCo, maxDeltaT, the first-step 1.2 growth with no setInitialDeltaT) and
   writeControl adjustable's trimming.
3. totalPressure (isentropic psi branch), totalTemperature and waveTransmissive: a nozzle inlet and outlet.
4. The viscous correctors and sigmaDotU; then LTS, janaf and sutherland, mesh motion, and turbulence.

(Items 1-4, LTS/janaf, turbulence (kOmegaSST) and mesh motion (velocityComponentLaplacian) are done: milestones C-K,
ARCHITECTURE.md sections 5d-5m. All seven v2412 tutorials run as shipped.)

## Qualification (2026-11-19): the seven tutorials as shipped, timed idle on the GB10

Wall seconds for the whole run. OpenFOAM v2412 ran once each: serial, and in four with the gates' frozen scotch
partitions. Four ranks is the gates' decomposition, not OpenFOAM's best on this 20-core CPU. Each brae binary is the
mean of two runs, and the two binaries were interleaved. Before is the commit before ARCHITECTURE.md sec. 5n; after
includes 5n. Every written file is the same from both binaries.

| tutorial | cells | OpenFOAM serial | OpenFOAM in four | brae before | brae after |
|---|---|---|---|---|---|
| shockTube | 100 | 0.25 | 0.67 | 0.46 | 0.46 |
| wedge15Ma5 | 4,800 | 3.69 | 1.88 | 0.84 | 0.84 |
| obliqueShock | 1,800 | 2.49 | 1.83 | 1.91 | 1.86 |
| forwardStep | 16,128 | 39.89 | 13.43 | 6.99 (six runs: 6.60) | 6.29 (six runs: 6.90) |
| LadenburgJet60psi | 1,200 | 14.11 | 6.45 | 27.74 | 17.76 |
| biconic25-55Run35 | 32,385 | 621.17 | 223.06 | 241.48 | 132.01 |
| movingCone | 1,900 | 39.42 | 16.72 | 100.99 | 41.99 |

- Where the cells are many, brae is ahead: forwardStep about 6 x OpenFOAM serial, biconic 4.7 x.
- The small viscous and moving cases are bound by latency:
  - kernel launches;
  - setDeltaT's host round trip;
  - GaussSeidel swept on the host in OpenFOAM's order.

  There brae is behind OpenFOAM in four, and on LadenburgJet60psi and movingCone also behind serial.
- forwardStep's 0.7 s "gain" in the two-run means is run-to-run spread, not a change (2026-11-19). Six interleaved
  runs each: before 6.60 s (6.22 to 7.00), after 6.90 s (6.50 to 7.52). Under Nsight Systems both binaries make the
  same calls: 140,574 kernel launches (21 a step), 6,694 stream synchronisations, 1,155 cudaMemcpy, and the
  kernels take 2.57 s and 2.59 s. Nothing in sec. 5n is on forwardStep's path: it is inviscid, static and not LTS.
  80 % of its CUDA API time is the one synchronisation a step that setDeltaT.H's Courant number needs (0.71 ms a
  step under the profiler, against 0.38 ms of kernels). A wall time on this machine moves about 10 % from run to
  run, so a timing claim here needs several interleaved runs.

## Against experiment (2026-11-19)

Two of the tutorials are experiments. Their public measurements were retrieved and compared on physical quantities,
with OpenFOAM's own post-processing run on both codes' fields (wallHeatFlux, surfaces on the wall patch).

**biconic25-55Run35 against CUBRC LENS Run 35** (surface pressure, 26 gauges, stated +-3 %; heat transfer, 41 gauges,
+-5 %; x from the sharp tip along the axis). The public copy of the sensor table pairs the heat-transfer gauges with
the wrong locations; re-paired by gauge number they agree with the Run 35 points plotted by Greenshields et al. (2010)
within 2.1 kW/m2 at all 41.

- **brae and OpenFOAM agree on the wall:** at the tutorial's last write the cone's pressure differs by 1.3e-6 and its
  heat flux by 4.9e-8 of their largest values. Every comparison below holds for both codes alike.
- **Fore-cone (x < 50 mm):** pressure +7.6 % (rms 8.2 %), heat flux +20 % against the gauges.
- **Separated region (50 to 92 mm):** pressure -9 % mean. The slip velocity reverses from 65.1 mm; the measurements
  bracket separation at 50.6 to 61.0 mm (pressure rise) and 55.2 to 65.6 mm (heat-flux drop).
- **Reattachment on the 55-degree cone:** peak pressure 6889 Pa at 108.6 mm against 4885 Pa at 113.3 mm (+41 %, 4.7 mm
  upstream); peak heat flux 601 kW/m2 at 109.2 mm against 536 kW/m2 at 111.9 mm (+12 %, 2.7 mm upstream).
- **Why they differ** (the tutorial's set-up, not the port): nitrogen as a single equilibrium species with janaf
  thermodynamics, where the LENS free stream is vibrationally frozen (Nompelis et al. give a non-equilibrium set,
  T_v 2562 K); the CUBRC nominal free stream (rho 5.515e-4 kg/m3 +-7 %, U 2713 m/s +-3 %, T 138.9 K +-3 %); slip and
  temperature-jump walls with accommodation 0.85; the axisymmetric wedge at 255 x 127 cells; a short pseudo-time from
  a supplied start field (see the 10-times-longer run below).

- **The tutorial's end time is not a converged state.** Run 10 times longer (LTS to 1e-2), the separation shrinks and
  moves away from the measurement: the slip velocity reverses from 71.4 mm (as shipped 65.1; 5e-3: 69.1), peak pressure
  7528 Pa at 104.1 mm, peak heat flux 675 kW/m2 at 104.7 mm. The fore-cone is unchanged (+7.5 % p, +22 % q). OpenFOAM
  decomposed in eight, run as long, gives 7530 Pa at 104.1 mm, 676 kW/m2 at 104.7 mm and reversal from 71.4 mm; its
  wall pressure differs from brae's by 6.7e-3 of the largest value and its heat flux by 3.4e-4 (the as-shipped gate
  measured OpenFOAM's own decomposed spread at 9.7e-3 in U). brae took 2912 s and OpenFOAM in eight 3603 s, both on a
  loaded machine.

**LadenburgJet60psi against Ladenburg, Van Voorhis and Winckler (1949).** The interferometric density tables are
behind a paywall; what is public is the Mach disc's triple point, x = 13.3 mm and r = 1.7 mm from the 10 mm nozzle
(as quoted by Greenshields et al. 2010, whose simulation on 240 x 80 cells gave 13.7 and 1.64 mm), and no uncertainty.
The tutorial's 60 psi is gauge (its inlet's stagnation pressure is 515 kPa).

- On the tutorial's 60 x 20 cells both codes put the Mach disc on the axis at 16.2 mm and the triple point at
  (16.1, 0.86) mm, steady from 2e-5 to 1e-4 s.
- Refined from the tutorial's coarse field (mapFields) and run to 4e-4 s, in both codes: the jet is not steady by 1e-4 s.
  On 120 x 40 the triple point moves 13.26 to 15.87 mm and oscillates about 13.3 to 14.0 mm after 0.2 ms; on 240 x 80 (the
  2010 paper's mesh) it settles at 13.79 to 13.90 mm from 0.26 to 0.4 ms (r 2.0 mm at 0.3 and 0.4 ms). Against
  Ladenburg's 13.3 mm that is 0.5 to 0.6 mm (4 %) downstream; the 2010 paper's own result is 13.7 mm.
- On 120 x 40, brae and OpenFOAM serial take the same 239,867 steps to 0.4 ms and put the disc and triple point at the
  same place to 0.01 mm at every write. Their fields agree to 2.6e-12 (p) at 0.1 ms; after 0.2 ms the difference grows
  exponentially (p 2.8e-8 at 0.24 ms, 7.0e-4 at 0.4 ms). The cause is the out-of-plane mode of the open checks on the
  as-shipped jet (VALIDATION.md rows 18-19): U_z, zero by symmetry on the wedge, seeded at round-off, grows about four
  times every 20 microseconds in both codes. OpenFOAM serial's largest |U_z| is 6.5e-7 m/s at 0.1 ms and 3.5 m/s at
  0.4 ms; brae's seed is half of it (3.0e-7 m/s) and reaches 3.0 m/s. Until 0.2 ms the in-plane fields stay about
  1e-11 from OpenFOAM serial, as OpenFOAM in four restarted from serial's 0.1 ms fields does (it inherits serial's
  mode, and stays within 1.9e-6 in p to 0.4 ms); once the mode reaches about 1 m/s (0.28 ms) it moves the in-plane
  fields. The triple point is unaffected. On 240 x 80, OpenFOAM in eight to 0.1 ms puts the triple point where brae does,
  (13.83, 1.94) mm, its fields 2.1e-11 (p) from brae's.
- Why the triple point differs from 1949: the tutorial's inlet is a uniform sonic state at the nozzle exit (fixedValue
  p 271.7 kPa, T 247.1 K, a 515 kPa reservoir), not the nozzle's own exit profile and boundary layer; laminar air
  (sutherland, Pr 1) on an axisymmetric wedge; ambient total pressure 1 atm; the interferometry's uncertainty is not
  stated.

## At scale (2026-11-19): the 3-D corner to 4.7M cells against OpenFOAM serial and parallel

The corner of tests/rho_central_3d.py (Mach 3 into two 10-degree ramps meeting at a right angle) at levels 1 to 3:
73,728, 589,824 and 4,718,592 cells. adjustTimeStep (maxCo 0.5 inviscid, 0.2 viscous) started at the CFL deltaT, a
fixed number of steps, one ascii write at the end (writePrecision 8). Both codes take the same number of steps.
OpenFOAM v2412 serial and decomposed with scotch into 4, 8, 16 and 20 ranks (`--bind-to core`); brae on the GB10's
GPU. Idle machine, every other job stopped. brae is the median of three runs (viscous: the mean of two runs of the final head);
OpenFOAM one run each. Wall seconds for the whole run, start-up and write included.

| case | cells | steps | brae | brae, device GaussSeidel | OpenFOAM serial | 4 | 8 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| inviscid | 73,728 | 954 | 2.53 | | 38.93 | 29.23 | 20.49 | 12.13 | 10.89 |
| inviscid | 589,824 | 295 | 5.46 | | 165.30 | 63.30 | 51.73 | 33.40 | 28.54 |
| inviscid | 4,718,592 | 100 | 21.83 | | 801.00 | 186.31 | 152.93 | 98.53 | 82.78 |
| laminar | 73,728 | 264 | 3.01 | 2.82 | 16.83 | 12.01 | 8.66 | 5.18 | 4.80 |
| laminar | 589,824 | 92 | 8.14 | 6.40 | 77.96 | 29.65 | 24.24 | 15.18 | 13.22 |
| laminar | 4,718,592 | 50 | 43.69 | 32.72 | 588.49 | 143.99 | 114.86 | 71.68 | 59.54 |
| kOmegaSST | 589,824 | 90 | 18.84 | 11.41 | 103.93 | 43.45 | 33.94 | 21.17 | 18.38 |

- **Per step** (brae's step-seconds over its steps; OpenFOAM's ExecutionTime): inviscid 2.0, 11.8 and 84 ms against
  OpenFOAM in 20 ranks 10, 93 and 780 ms.
- **Device GaussSeidel** is BRAE_GS_HOST_SMOOTHER=0, brae's switch for the level-scheduled device walk in place of
  OpenFOAM's sweep on the CPU. Every written file and every solve line is identical either way. It is quicker on these
  3-D meshes and slower on the 2-D tutorials (LadenburgJet60psi 21.5 against 18.2 s), so the default stays the CPU sweep
  (ARCHITECTURE.md sec. 5p).
- **Start-up, steps and write** (inviscid, 4.7M cells): about 8 s to read the ascii case and build and upload the mesh,
  8.4 s of steps, 5.4 s for the one write. OpenFOAM serial writes the same fields in about 5.2 s, OpenFOAM in 20 ranks
  in 0.5 s.
- **Memory:** brae's device memory 72 MiB, 526 MiB and 3.9 GiB inviscid, 97 MiB, 717 MiB and 5.4 GiB laminar, 872 MiB
  kOmegaSST; host RSS 4.4 GiB at 4.7M cells. OpenFOAM serial's RSS 9.1 GiB (inviscid) and 9.5 GiB (laminar) at 4.7M;
  in 20 ranks the largest rank's 590 to 605 MiB.
- **CPU:** brae's user time at 4.7M cells is 9 s inviscid and 33 s laminar on the CPU sweep (15 s on the device
  sweep); OpenFOAM in 20 ranks 1140 s and 843 s.
- **Profile** (Nsight Systems, 590k cells): inviscid, 22 kernel launches and one synchronisation a step; the face kernel
  is half the GPU time, the single-block Courant reduction 13.6 %. Laminar on the CPU sweep, 223 launches a step, and
  the synchronous copies around GaussSeidel's CPU sweep are two thirds of the CUDA API time.
- The seven tutorials as shipped (sec. above) are 100 to 32,385 cells, where brae is latency-bound; these runs are
  where it is meant to work.
