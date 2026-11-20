# rhoCentralFoam on the GPU: experimental release report

`brae_rhoCentralFoam` ports OpenFOAM v2412's `rhoCentralFoam`: density-based, explicit, with Kurganov/Tadmor
central-upwind fluxes. This report states what the release covers and what it does not. It also says how the solver
was checked, and against what. The design is in [ARCHITECTURE.md](ARCHITECTURE.md), the gate record in
[VALIDATION.md](VALIDATION.md), and the measurements in [PORT.md](PORT.md).

**Status: experimental, single GPU.** The label is in the solver's registry row, in a line on stderr at start-up, and
in the build target's comment. It stays while registered open checks remain, while its 3-D coverage is young, and
while it runs on one GPU.

**Review decision (2026-11-20): approved for an experimental single-GPU merge, not as a production-ready release.**
The review accepted:
- the shared reader and writer corrections (sec. 8), including the refusal of older `uniformTotalPressure` files whose
  time-dependent p0 was lost;
- the 24 registered open checks, held to exactly that set by the strict CI check (sec. 7). No bound was widened, no
  failure is excluded, and none is reported as passing.

The supported scope and the numerical algorithms are as this report describes them. Multi-GPU work is separate.

## 1. Scope

**Runs, on one GPU.** This release adds no multi-GPU path for rhoCentralFoam.

**Covered:**
- **Physics:** inviscid, laminar, and compressible RAS with `kOmegaSST` and OpenFOAM's wall functions.
- **Thermo:** `hePsiThermo`: `hConst` or `janaf`, `const` or `sutherland`, `sensibleInternalEnergy`.
- **Time:**
  - fixed `deltaT`, `adjustTimeStep`, or LTS (`localEuler`);
  - restart from `latestTime` or `startTime`, with `uniform/time`;
  - `purgeWrite`.
- **Mesh:** 1-D, 2-D, axisymmetric (wedge) and 3-D. Static, or `dynamicMotionSolverFvMesh` with
  `velocityComponentLaplacian`.
- **Boundary conditions:** those of the seven v2412 tutorials, listed in ARCHITECTURE.md sec. 6 and
  `rhoCentralControls.cu`.
- **Reconstruction limiters:** vanLeer, vanAlbada, Minmod, MUSCL, SuperBee and limitedLinear, each with its V form,
  plus upwind.

**Stops at start-up with a message naming the entry:**
- cyclic patches;
- RAS models other than `kOmegaSST`, and LES;
- motion solvers other than `velocityComponentLaplacian`;
- viscous cases on the host reference arm;
- every boundary condition, scheme and thermo the gates do not cover (`tests/rho_central_refusals.sh`, 46 checks).

**Approximated, with a notice:** a case asking for `writeFormat binary` is written in ascii, at its `writePrecision`.

## 2. Two kinds of agreement, kept apart

Two kinds of agreement are reported, and they are never mixed.

- **Numerical agreement:** brae computes what OpenFOAM v2412 computes on the same case. It is shown by comparing
  brae against OpenFOAM, field by field, at every write. Experiment plays no part in it.
- **Experimental agreement:** a simulation of a real experiment matches the measurements. It depends on the case
  set-up, the physical models and the mesh, and on none of the GPU port.

When brae and OpenFOAM agree with each other and both miss a measurement by the same amount, the port is faithful and
the miss belongs to the set-up. Every experimental comparison below therefore states first how closely brae and
OpenFOAM agree on the quantity measured.

## 3. Numerical agreement with OpenFOAM v2412

**How a gate decides.** Each gate runs OpenFOAM v2412 on the same machine, with the same case, schemes, boundary
conditions and time step, and compares every written field at every write.
- **Bound:** a field passes within ten times OpenFOAM's own spread. The spread is OpenFOAM serial against OpenFOAM
  decomposed in two and in four, on a frozen partition.
- **Floor:** 8e-14, or 1e-6 for files written at six digits.
- **Controls:** each gate also runs a deliberately wrong variant, which must turn red by more than 100 times the bound.
- **Fixed in advance:** thresholds were declared before each gate first ran and have never been widened.

**Result:** 13 gates, 1,122 checks passed. The 24 that fail are registered open checks (sec. 6); none is on a tutorial
as shipped. VALIDATION.md has the table.

**The seven tutorials as shipped** (CUDA arm). "Of the bound" is brae's worst distance from OpenFOAM serial over every
write and field, as a fraction of the gate's bound.

| tutorial | cells | worst field (OpenFOAM's own spread) | of the bound | solves at OpenFOAM's iteration count |
|---|---|---|---|---|
| shockTube | 100 | U 4.9e-15 (6-digit files) | < 1e-8 | n/a |
| wedge15Ma5 | 4,800 | T 5.3e-6 (2.7e-6, 6-digit files) | 0.198 | n/a |
| obliqueShock | 1,800 | U 3.4e-8 (1.7e-6) | 0.002 | n/a |
| forwardStep | 16,128 | p 1.9e-5 (8.4e-6, 6-digit files) | 0.230 | n/a |
| LadenburgJet60psi | 1,200 | U 3.9e-13 (3.7e-13) | 0.104 | 30,444 of 30,444 |
| biconic25-55Run35 | 32,385 | U 8.5e-4 (9.7e-3); rDeltaT 3.8e-8 | 0.066 | 40,000 of 40,000 |
| movingCone | 1,900 | p 4.3e-6 (4.4e-5); mesh points 6.7e-7 | 0.168 | 43,264 of 43,265 (tight-solve variant) |

**In 3-D.** v2412 ships no 3-D rhoCentralFoam tutorial, so the 3-D gate builds five cases. Each is compared with
OpenFOAM under the same rule, and OpenFOAM reads brae's output back.

| case | what it exercises | worst field, of the bound |
|---|---|---|
| Mach 3 axial corner, inviscid | two 10-degree ramps meeting at a right angle: oblique shocks and their interaction | 0.333 |
| the corner, laminar | sutherland, noSlip walls graded 50:1 | 0.096 |
| the corner, kOmegaSST | OpenFOAM's wall functions, graded 200:1 | 0.140 |
| closed-box blast | slip walls only: mass and energy conserved to round-off; restarted from OpenFOAM's and brae's output | 0.102 (restarts 0.035, 0.044) |
| totalPressure duct | totalPressure (psi form), totalTemperature, waveTransmissive | 0.402 |

**Under refinement.** The inviscid corner's ramp pressure was refined in both codes to 590k cells. It converges to
oblique-shock theory, and brae and OpenFOAM agree at every level. The viscous corners are not steady at the lengths
run, so no refinement claim is made for wall shear or heat flux.

| cells | ramp p / p_inf, brae | OpenFOAM | theory |
|---|---|---|---|
| 9,216 | 2.177876 | 2.177956 | 2.054655 |
| 73,728 | 2.039676 | 2.039669 | |
| 589,824 | 2.055768 | 2.055383 | |

**Other checks:**
- the host reference arm (serial C++) against the CUDA arm;
- conservation to round-off, or the residual-accounted balance for the implicit solves;
- restarts from OpenFOAM's output and from brae's;
- every iterative solve's iteration count against OpenFOAM's.

## 4. Experimental agreement

Two of the tutorials model experiments. The public measurements were compared with both codes' fields on physical
quantities, through OpenFOAM's own post-processing.

**biconic25-55Run35 against CUBRC LENS Run 35** (surface pressure, stated uncertainty +-3 %; heat transfer, +-5 %):
- **Numerical:** brae and OpenFOAM agree on the cone to 1.3e-6 in pressure and 4.9e-8 in heat flux.
- **Experimental:** both codes give the fore-cone pressure 7.6 % above the gauges and the heat flux 20 % above.
  - Peak pressure is 6,889 Pa at 108.6 mm, against 4,885 Pa measured at 113.3 mm.
  - Peak heat flux is 601 kW/m2 at 109.2 mm, against 536 kW/m2 at 111.9 mm.
- **Why the miss belongs to the set-up:**
  - nitrogen is modelled as one equilibrium species, where the LENS free stream is vibrationally frozen;
  - the walls are slip walls with accommodation 0.85;
  - the mesh is 255 x 127 cells;
  - the end time is not converged. Ten times longer, in both codes, the separation shrinks and the peak pressure moves
    to 7,528 Pa at 104.1 mm.

**LadenburgJet60psi against Ladenburg, Van Voorhis and Winckler (1949).** The public measurement is the Mach disc's
triple point at x = 13.3 mm, with no stated uncertainty.
- **Numerical:** both codes put the triple point at the same place. On 120 x 40 they take the same 239,867 steps to 0.4 ms
  and agree on it to 0.01 mm at every write, while the registered out-of-plane round-off mode (rows 18-19) grows.
- **Experimental:**

  | mesh | triple point, both codes | measured |
  |---|---|---|
  | the tutorial's 60 x 20 cells | 16.1 mm | 13.3 mm |
  | 240 x 80 cells (the mesh of Greenshields et al. 2010) | brae settles at 13.8 to 13.9 mm from 0.26 to 0.4 ms; OpenFOAM in eight, run to 0.1 ms, puts it where brae does (13.83 mm, fields 2.1e-11 apart) | 13.3 mm |

  The 2010 paper's own simulation gave 13.7 mm.
- **Why:** on the tutorial's coarse mesh the disc sits 2.8 mm downstream of the measurement. On the 2010 paper's mesh,
  brae is 0.5 to 0.6 mm (4 %) downstream, and the paper's own simulation 0.4 mm. The mesh, not the port, sets the
  distance from the measurement.

## 5. Performance against OpenFOAM serial and parallel (GB10, idle)

All timings are wall seconds for the whole run, start-up and writing included, and both codes take the same number of
steps. OpenFOAM v2412 ran on the same machine's 20-core CPU, decomposed with scotch and bound to cores.

**The 3-D corner at scale:**

| case | cells | brae | brae, GaussSeidel on the device | OpenFOAM serial | OpenFOAM, 20 ranks |
|---|---|---|---|---|---|
| inviscid | 73,728 | 2.53 | | 38.93 | 10.89 |
| inviscid | 589,824 | 5.46 | | 165.30 | 28.54 |
| inviscid | 4,718,592 | 21.83 | | 801.00 | 82.78 |
| laminar | 73,728 | 3.01 | 2.82 | 16.83 | 4.80 |
| laminar | 589,824 | 8.14 | 6.40 | 77.96 | 13.22 |
| laminar | 4,718,592 | 43.69 | 32.72 | 588.49 | 59.54 |
| kOmegaSST | 589,824 | 18.84 | 11.41 | 103.93 | 18.38 |

**The seven tutorials as shipped:**

| tutorial | cells | brae | OpenFOAM serial | OpenFOAM in four |
|---|---|---|---|---|
| shockTube | 100 | 0.46 | 0.25 | 0.67 |
| wedge15Ma5 | 4,800 | 0.84 | 3.69 | 1.88 |
| obliqueShock | 1,800 | 1.86 | 2.49 | 1.83 |
| forwardStep | 16,128 | 6.90 | 39.89 | 13.43 |
| LadenburgJet60psi | 1,200 | 17.76 | 14.11 | 6.45 |
| biconic25-55Run35 | 32,385 | 132.01 | 621.17 | 223.06 |
| movingCone | 1,900 | 41.99 | 39.42 | 16.72 |

- **Large meshes:** at 4.7M cells inviscid, brae is 37 times as fast as OpenFOAM serial and 3.8 times as fast as
  OpenFOAM on 20 ranks. Laminar at 4.7M cells it is 1.4 times as fast as 20 ranks on the default sweep, and 1.8 times
  with GaussSeidel on the device.
- **Small meshes:** brae is bound by latency, not arithmetic: kernel launches, the one host round trip a step that
  setDeltaT needs, and GaussSeidel swept on the CPU in OpenFOAM's order. It trails OpenFOAM in four on the small
  viscous and moving tutorials, and trails OpenFOAM serial on LadenburgJet60psi and movingCone.
- **Memory:** device memory is 0.87 KiB a cell inviscid, 1.2 KiB laminar and 1.5 KiB with kOmegaSST.
- **Where the time goes at 4.7M cells, inviscid:** 8.4 s of steps, about 8 s of start-up and 5.4 s for the one ascii
  write, which is single-threaded. OpenFOAM in 20 ranks writes in 0.5 s.
- **GaussSeidel on the device** (`BRAE_GS_HOST_SMOOTHER=0`) gives identical files and solve lines. It is faster on
  large 3-D meshes and slower on the small 2-D tutorials, so the default stays the CPU sweep.

## 6. Known limitations

**The 24 registered open checks** (`tests/data/rho_central_open_checks.tsv`; VALIDATION.md has the table and the
analysis):
- **15 round-off amplification:** a difference at round-off grows where the flow amplifies it, as it does in
  OpenFOAM's own perturbed runs.
- **4 time-step sequence divergence:** the adaptive deltaT sequence parts from OpenFOAM serial's after 800 to 1,600
  steps, as OpenFOAM's own decomposed runs do. With deltaT fixed, the same cases pass.
- **5 tolerance-limited:** an iterative solve stops one sweep from OpenFOAM's because its final residual is within
  round-off of the tolerance. The fields stay within the bound.

None of them was resolved by a wider bound or a clamp. The six gates that carry them stay red. In CI they are held to
exactly these 24 (sec. 7).

**Performance limits:**
- GaussSeidel is swept on the CPU by default;
- the writer is single-threaded and ascii;
- the Courant number is reduced in one block, which keeps a fixed summation order;
- small meshes are latency-bound.

**Coverage limits:**
- the 3-D viscous corners are not steady at the lengths run, so there is no 3-D refinement claim for wall shear or
  heat flux;
- the experimental comparisons are limited by the tutorials' set-ups (sec. 4);
- a restart under LTS starts and runs (the LTS gate starts brae and OpenFOAM from brae's biconic output), but its
  fields are not compared with OpenFOAM's own LTS restart.

**Found during the review, outside rhoCentralFoam, not fixed here** (both on `main` too):
- pimpleFoam under `adjustTimeStep` misses the adjustable write times and `endTime`, and misnames its last time
  directory (issue #53);
- rhoSimpleFoam's default driver, restarted from its own write, drifts 1.7e-4 in U from its uninterrupted run, inside
  the 1e-3 its restart gate allows (issue #54).

**Not in this release:** multi-GPU (the coupled-patch plan is in ARCHITECTURE.md sec. 5c), cyclic patches, other RAS
models and LES, other motion solvers, binary output.

## 7. CI

- **Hosted build-check (`ci.yml`):** compiles `brae`, which depends on `brae_rhoCentralFoam`, for sm_80 and sm_90. It
  also runs the known-red check's self-test.
- **GB10 workflow (`gpu-test.yml`):** two steps.
  - **ctest:** runs every test except those labelled `known-red`.
  - **Known-red gates:** runs the six known-red rhoCentralFoam gates through `tools/known_red_check.py`. It passes only
    when what is red is exactly the 24 registered checks, counted per registry row with each gate's pass count
    (`tests/data/rho_central_known_red.tsv`). A new FAIL, a registered check going green, or an arm that stops running
    all fail this step until the registry is updated in review.
- **Labels:** the gates carry `slow` and `rhocentral`. Without OpenFOAM v2412 they skip (exit 77).

**Tests that fail on `main` as well.** The full ctest on the GB10, leaving out `known-red` and `rhocentral`, ran 902
tests: 898 passed, 3 failed and 1 skipped. None of the four is in rhoCentralFoam's path. The branch leaves them as
they are on `main`, except that the manifest tool both manifest tests run gains a rhoCentralFoam entry (additions only;
`manifest_rhocentralfoam_not_drifted` passes).

| test | on this branch | on `main` (df3cd8cb) | cause, and where it is tracked |
|---|---|---|---|
| `manifest_simplefoam_not_drifted` | fails | fails | `manifest/simpleFoam.yaml` is not in the tree (issue #50). It skips on a runner without the `ofscan` checkout, as the GB10 CI's is |
| `manifest_rhosimplefoam_not_drifted` | fails | fails | `manifest/rhoSimpleFoam.yaml` is not in the tree (issue #50) |
| `kepsilon_production_and_blend` | aborts | fails, in the GB10 CI's ctest step | the test does not pass the stored wall nut to the kEpsilon closure. The test-only fix is on branch `fix/kepsilon-test-wall-nut` (213e3506) |
| `simple_freestream_of13_vs_openfoam13` | skipped | skipped | needs OpenFOAM 13 |

## 8. Shared code this release changes

The solver's own code is new. The shared changes that reach other solvers are listed in the pull request, each
checked against those solvers. The field writer and reader carry the ones a user of simpleFoam, pimpleFoam or
rhoSimpleFoam would notice, all in what brae writes and reads back:

- **totalPressure:** written with `p0`, and with rho, psi and gamma, in every form.
- **uniformTotalPressure:** p0 written as OpenFOAM's Function1.
- **fanPressure:** written with its fan curve and direction.
- **phi on a symmetry, symmetryPlane or wedge patch:** written with its constraint type.
- **psi on totalTemperature and waveTransmissive:** a missing psi takes the type's default.
- **A uniformTotalPressure with no `p0`:** refused by name, as OpenFOAM refuses it.
- **A totalPressure whose p0 an earlier brae wrote as `inletValue`:** read as p0, with a notice. The value is p0's,
  so a restart from older brae output continues exactly.
- **phi on a symmetry, symmetryPlane or wedge patch, as earlier brae wrote it:** OpenFOAM still refuses such files; the
  fix applies to output written from now on.

`tests/total_bc_roundtrip_vs_openfoam.sh` holds all of these against real OpenFOAM, in both directions and across
restarts. Its 63 checks pass, and on the build before these fixes 22 of them fail.

**Restarting from a time directory an older brae wrote:** [docs/restart-older-output.md](../../../../docs/restart-older-output.md)
gives the exact edits, each tested on such a directory. A `uniformTotalPressure` or `fanPressure` patch gets its `p0`
definition, and the fan's curve and direction, copied back from the case's own `0/p`. Restored that way, pimpleFoam's
TJunction restarts identically to its uninterrupted run (the fields are equal ten steps later, with the table and with
the fan), and OpenFOAM starts from the same file. A `totalPressure` patch needs no edit for brae. For OpenFOAM, the
older `phi` files need their constraint patches retyped, and an older `totalPressure` needs `inletValue` renamed to `p0`.

## 9. The evidence, and the commit it holds at

- **Gates:** the suite ran at 4d951925, the last commit on this branch that changes code. Every later commit changes
  documentation only, so the executables built at 4d951925 are those of the pull request's head.
  - 1,122 checks passed and the 24 registered open checks failed, the same counts gate by gate as the suite before
    it;
  - the known-red check reported ok;
  - all 13,290 files brae wrote are byte-identical to the previous suite's.
- **Full ctest:** 898 of 902 passed, 3 failed and 1 skipped, as sec. 7 lists.
- **Hosted CI:** green on the head. The pull request's description names the head and its run.
- **`main`:** the branch starts at `main`'s head, df3cd8cb, so a merge adds exactly this branch's changes.
