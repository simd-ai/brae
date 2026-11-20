# Proposal: the CUDA/C++ source as layered modules

Status: proposal. No source moves with this document. Each phase below is its own pull request, with the build,
numerical-identity and performance checks named for it. None of it is mixed with the CMake or README work.

Measured on `main` 2b26cdd6 (2026-11-20).

## What is there now

`src/README.md` sets the rule: solvers consume shared components, and a solver never includes another solver's
private implementation. The tree mostly follows OpenFOAM's layout, but the build and the includes do not hold to
that rule yet.

**One library holds every solver.** `brae_core` is one static library of 197 translation units (129,766 lines). 50
of them (51,922 lines, 40 %) are solver drivers from `src/applications/solvers/`: interFoam's 25 (30,118 lines),
rhoSimpleFoam's 16 (11,513) and simpleFoam's 9 (10,291). Every executable links all of them, and a change to one solver's
driver relinks every executable and every test.

**Shared modules include solver folders.** Includes resolve by basename over every header directory, so nothing
stops a shared file from reaching into a solver. Counted over `src/`:

| from | into | includes | example |
|---|---|---:|---|
| `finiteVolume`, `matrices`, `meshTools`, `dynamicFvMesh`, `fvMotionSolver`, `OpenFOAM` | `solvers/interFoam` | 21 | `device_amg.cu` → `inter_phase_time.cuh` |
| `OpenFOAM`, `finiteVolume`, `cuda`, `TurbulenceModels`, `fvMotionSolver`, `functionObjects`, `thermophysicalModels` | `solvers/common` | 16 | `brae_time.cuh` → `brae_notice.cuh`; `device_divdevreff.cu` → `stage_dump.cuh` |
| `TurbulenceModels`, `functionObjects` | `solvers/simpleFoam` | 5 | `device_les_keqn.cu` → `pEqn.cuh`; `forces_fo.cuh` → `device_simple_foam.cuh` |

**Solvers include each other.** rhoSimpleFoam → simpleFoam 8 (`device_simple_foam.cuh`), rhoCentralFoam →
interFoam 3 (`inter_phase_time.cuh`), interFoam ↔ simpleFoam 2 each way (`UEqn.cuh`, `device_inter_ueqn.cuh`),
pimpleFoam → simpleFoam 1, simpleFoam → rhoSimpleFoam 1 (`rho_simple_controls.cuh`).

Most of the edges come down to four files that are infrastructure, kept in a solver folder:

| file | where | what it is |
|---|---|---|
| `inter_phase_time.cuh` | `solvers/interFoam` | a phase timer every module uses |
| `brae_notice.cuh` | `solvers/common` | the `[ignored]`/`[approximated]`/refusal notices |
| `scheme_parse.cuh` | `solvers/common` | fvSchemes parsing, 1,380 lines, header-only |
| `stage_dump.cuh` | `solvers/common` | the stage-dump instrument the gates compare |

and one driver that became a base class: `device_simple_foam.cu` (5,155 lines) with `device_simple_foam.cuh` (1,035
lines, included by 21 files), which pimpleFoam and rhoSimpleFoam drive.

**`src/cuda/` is a second tree beside the mirror.** Its 47 files (20,951 lines) hold turbulence closures
(`device_komega_sst.cu` 1,946, `device_kepsilon.cu` 1,616, `device_spalart.cu`, `device_smagorinsky.cu`,
`device_deshybrid.cu`, `device_maxwell.cu`), finite-volume operators (`device_fvc.cu` 1,458, `device_fvm.cu`,
`device_ddt.cu`), boundary conditions (`device_boundary.cuh` 1,100, `device_boundary_flow.cu` 1,367) and coupled
patches (`device_cyclic.cu` 1,648, `device_ami.cu` 1,123). Each has a place in the mirror already.

**Large files and heavy headers.** `inter_driver_device.cu` 7,700 lines, `device_simple_foam.cu` 5,155,
`inter_writer_cpp.cu` 3,806. Headers that carry implementations: `fv_patch_field.cuh` 4,703 lines (included by 99
files), `foam_field_reader.cuh` 1,877 (164), `device_boundary.cuh` 1,100 (77), `scheme_parse.cuh` 1,380 (29).
Every file that includes them compiles their bodies again.

## Target

```
                 executables: brae  brae_pimpleFoam  brae_rhoSimpleFoam  brae_interFoam  brae_rhoCentralFoam
                                 │          │               │                 │                 │
solver libraries:      brae_simplefoam  brae_pimplefoam  brae_rhosimplefoam  brae_interfoam  brae_rhocentralfoam
                                 └──────────┴───────┬───────┴─────────────────┴─────────────────┘
shared, one direction only:              brae_turbulence   brae_thermo   brae_dynamic_mesh
                                                   │
                                          brae_finite_volume   (fvc/fvm, fields, BCs, coupled patches, fvOptions)
                                                   │
                                          brae_linear_solvers  (PCG, PBiCG(Stab), DIC/DILU, Gauss-Seidel, GAMG, AMG-PCG)
                                                   │
                                          brae_foundation      (OpenFOAM/: mesh, db, IO, dictionaries, notices,
                                                                 scheme parsing, stage dumps, timing; CUDA primitives:
                                                                 buffers, streams, reductions, BLAS1, graph helpers)
```

- One static library per layer, each with `target_include_directories(... PUBLIC <its own directories>)` and
  `target_link_libraries` on the layer below. A layer that includes a header from a layer above fails to compile,
  which is the check `src/README.md` has had to rely on review for.
- One library per solver, holding its driver files now in `brae_core`. An executable links its solver's library
  and the shared layers; `brae` links simpleFoam's (it hands every other case to the solver beside it).
- What two solvers share moves down to the layer that owns it. The incompressible pressure-velocity pieces
  pimpleFoam and rhoSimpleFoam take from `DeviceSimpleSolver` go to `src/finiteVolume/cfdTools/` (as `constrainHbyA`,
  `adjustPhi` and `setRefCell` already have); the four infrastructure headers above go to `src/OpenFOAM/db/`.
- `src/cuda/` empties into the mirror: closures to `src/TurbulenceModels/...`, operators to
  `src/finiteVolume/finiteVolume/{fvc,fvm,ddtSchemes}`, boundary conditions to `src/finiteVolume/fields/fvPatchFields`,
  coupled patches to `src/finiteVolume/fvMesh/fvPatches/constraint/{cyclic,cyclicAMI}`, primitives to
  `src/OpenFOAM/` and `src/matrices/lduMatrix/lduMatrix/`.
- Interfaces in focused `.cuh` files, bodies in `.cu`. The header-heavy files above keep their declarations and
  inline accessors in the header; the rest moves to a `.cu`, with explicit instantiation for the scalar and vector
  types a template is used with.
- Kept as they are: the `_cpp` reference beside each CUDA arm, `--default-stream per-thread` on every translation
  unit, whole-program compilation (no `-rdc` but NVSHMEM's one translation unit), the CUDA-graph capture and
  replay paths, and the per-target architecture list (sm_80, sm_90, sm_121 and the rest of the README's table).

## Phases, each its own pull request

| phase | change | code changed | how it is checked |
|---|---|---|---|
| 1 | Move the four infrastructure headers out of solver folders; move `src/cuda/` files to their mirror places (`git mv`, contents unchanged) | none | object identity (below) on every executable; inventory of targets and tests identical but for source paths |
| 2 | Split `brae_core` into the layers and solver libraries, target-scoped includes, the global header glob retired | build only | object identity; a deliberately upward include must fail to compile (the layering control) |
| 3 | Break the solver-to-solver edges: shared pieces of `DeviceSimpleSolver` into `finiteVolume/cfdTools`, `UEqn`/`device_inter_ueqn` coupling into a shared assembly | yes, by moving functions | output identity (below) and the full suite |
| 4 | Heavy headers to `.cuh` + `.cu`, explicit instantiation | yes, inlining may change | output identity, full suite, compile time and binary size reported |
| 5 | Split the largest drivers (`inter_driver_device.cu`, `device_simple_foam.cu`) along OpenFOAM's own file boundaries (`UEqn.H`, `pEqn.H`, `alphaEqn.H`, ...) | yes | output identity, full suite, performance |

**Object identity** (phases 1 and 2, where no translation unit changes): nvcc is not byte-reproducible (each
translation unit carries random `_GLOBAL__N_` and `_INTERNAL_` identifiers), so identity is checked as it was for a
release on 2026-11-20: every compiled source identical, `cuobjdump -sass` identical after the random identifiers are
normalised, and the host `.text` of each executable of equal size, any differing byte accounted for.

**Output identity** (phases 3 to 5, where a function may move to another translation unit and its inlining, and so
its floating-point contraction, may change): simpleFoam and rhoSimpleFoam write byte-identical fields run to run
today, so their tutorials and the gates' cases must write the same bytes before and after. pimpleFoam (not
bit-reproducible run to run, 1.4e-08 on pitzDaily) and interFoam are held to their existing gates, unchanged. Where
a byte differs, the phase stops until the difference is traced to a contraction and shown to be within one ulp per
operation.

**CUDA graphs**: `BRAE_CACHE_STATS=1` on a fixed case reports the same captures, replays and cache keys before and
after.

**Performance**: the per-step medians of docs/performance.md (steps 11 to 60 or 11 to 40) on one GPU, before and
after, within the run-to-run spread; compile time and the size of each executable reported.

**Every phase**: the hosted build for sm_80, sm_90 and sm_121, and the GPU suite uncached (gpu-full with
`no_cache`).

## Risks

- **Static-library cycles.** The monolith hides them; the layers cannot. Phase 3's edges are the known ones, and
  phase 2 lists any the linker finds.
- **Contraction changes.** Moving a function across a translation unit boundary can turn `a*b + c` into an FMA or
  back. Output identity catches it; the response is to understand it, not to loosen a bound.
- **Compile time** may rise briefly where a header's bodies move into several `.cu` files, and fall once the
  explicit instantiations are in place. Measured per phase.
- **History.** `git mv` keeps `git log --follow`; phase 1 moves files without editing them, so blame survives.
