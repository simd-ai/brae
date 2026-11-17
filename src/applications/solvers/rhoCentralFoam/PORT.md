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
