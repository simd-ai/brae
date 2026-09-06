# brae source layout — the OpenFOAM mirror

This tree mirrors OpenFOAM v2412 closely enough that provenance is mechanical rather than remembered.

## Why the previous layout was replaced

`src/applications/solvers/simpleFoam/device_simple_foam.cu` grew to **3,578 lines** and became the shared
implementation for `pimpleFoam`, `rhoSimpleFoam`, both `rhoSimpleFoam` variants and five headers in
`solvers/common/`. One solver's private file was the de-facto base class for every other solver. Every
change to it was a change to all of them, and no solver could be reasoned about alone.

The dependency *direction* was never the problem — `src/cuda` has never included anything from
`src/applications`, verified across all 65 files. The problem is that solver-independent infrastructure
lived **inside a solver**: `constrainHbyA` in `simpleFoam/simple_foam.cuh`, `adjustPhi` / `setRefCell` /
`setReference` in `solvers/common/solver_controls.cuh`.

## The rule

```
shared infrastructure
        ↑
   ┌────┼────┐
   │    │    │
simple pimple rhoSimple
```

Solvers consume public reusable components. A solver may **never** include another solver's private
implementation. If you are about to add something to a solver, ask whether it is a *simpleFoam* feature or
an *OpenFOAM finite-volume* feature. `grad`, `div`, `laplacian`, interpolation, matrix assembly, matrix
relaxation, LDU operations, linear-solver dispatch, boundary coefficients, surface flux handling and
dictionary access are all the second kind.

## Layout

| path | mirrors | contents |
|---|---|---|
| `src/OpenFOAM/` | `src/OpenFOAM/` | primitives, meshes, db, matrices/solution |
| `src/matrices/lduMatrix/lduMatrix/` | same | LDU view, BLAS1, deterministic reductions, SpMV |
| `src/matrices/lduMatrix/solvers/PCG/` | same | PCG and the AMG-preconditioned / conditional-graph drivers |
| `src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/` | same | AMG build, Galerkin re-coarsening, V-cycle, smoothers, cache |
| `src/matrices/lduMatrix/preconditioners/DILUPreconditioner/` | same | level-scheduled DILU |
| `src/finiteVolume/` | `src/finiteVolume/` | fields, fvMesh, fvMatrices, fvc/fvm, cfdTools |
| `src/TurbulenceModels/` | `src/TurbulenceModels/` | momentum transport models |
| `src/applications/solvers/` | `applications/solvers/` | thin drivers only |

Paths follow **the OpenFOAM version actually being mirrored**. v2412 puts the turbulence models under
`src/TurbulenceModels/`, not `src/MomentumTransportModels/` — `tools/of_manifest.py` refuses to emit a
manifest citing an OpenFOAM path that does not exist, which is how that error was caught rather than
shipped.

## The `_cpp` reference discipline

Every numerical component has two implementations:

```
linearViscousStress_cpp.cu     host transcription of the OpenFOAM semantics — correctness only
device_divdevreff.cu           the CUDA implementation — validated against the reference
```

The `_cpp` file is a readable, allocation-per-step, host-only transcription whose single job is to be
*obviously* the same as the OpenFOAM text it quotes. It carries the OpenFOAM file and line in its header,
and it is validated against **OpenFOAM's own output**, never against another brae path — a self-comparison
proves nothing.

The comparison boundary that matters:

```
same input fields + same mesh + same dictionary + same equation stage
        ↓
_cpp output   vs   CUDA output
```

so that the *first* divergent intermediate can be isolated. Debugging from a final residual history is what
this replaces.

### Worked example — the first component

`src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress_cpp.cu` transcribes

```cpp
// linearViscousStress.C:107-117
- fvc::div((alpha*rho*nuEff())*dev2(T(fvc::grad(U)))) - fvm::laplacian(alpha*rho*nuEff(), U)
```

reached from `simpleFoam/UEqn.H:9` via `IncompressibleTurbulenceModel.C:123`. It reuses brae's existing
`transpose`/`dev2`/`operator*` rather than restating them — **move working code, do not rewrite it**.

`tests/test_divdevreff_cpp.cu` proves it against `validation/kEpsCorrect/divdevreff.dat`, a dump from
OpenFOAM, at **7.4e-16** over 12,225 cells, and carries two controls that must fail:

- a **wrong-sign** control (rel 2.0) — the sign convention is the one thing a transcription can get wrong
  without failing loudly, so it is asserted rather than reasoned about;
- a **wall-nuEff negative control** (rel 1.03) — re-runs with the boundary viscosity replaced by the owner
  cell value, the exact defect brae has shipped before. A test that cannot detect it is blind.

## Determinism — done for the incompressible simpleFoam path

Two runs of the same binary on the same case now produce **bit-identical** fields. Before:

| iterations | 1 | 5 | 20 | 200 |
|---|---|---|---|---|
| worst rel, before | 1.9e-08 | 1.4e-03 | 3.6e-02 | — |
| worst rel, now | **0** | **0** | **0** | **0** |

That number was the *noise floor* of every `_cpp`-vs-CUDA comparison, and a defect smaller than the floor
cannot be seen — which is how the LUST implicit-weight bug hid behind a plausible residual.

Three scatter sites caused it. All three were converted from `atomicAdd` scatter to a **fixed-order
gather**, which is possible because the addressing they need (`ownerStart`/`losort`/`losortStart`, and the
inverse of the AMG agglomeration) is static and already built:

| site | why it summed in a varying order |
|---|---|
| AMG restriction, `rc[map[c]] += r[c]` | many fine cells per coarse cell; runs every level, every V-cycle, every PCG iteration |
| turbulence wall functions, `G0/eps0/omega0[c]` | cells with more than one wall face (`invNw` exists precisely for those) |
| eps `setValues`, `source[nei] -= lower[f]*eps0[own]` | cells with more than one constrained face |

The last two are **rare**, which made them harder rather than easier: the case came out bit-identical at
1, 5, 8, 10 and 15 iterations and differed at 12. An intermittent seed is still fatal here because the
SIMPLE loop amplifies it — one ULP at iteration 1 became 1.3e-3 by iteration 20.

`tests/determinism_gate.sh` (ctest: `determinism_kepsilon`, `determinism_komega`) runs the same case twice
and requires bit equality, and carries a 1-ULP negative control so a broken comparator cannot report
success. Also verified on `airfoil` and `backwardFacingStep2D`.

**Still open**: the opt-in `BRAE_AMG_SA` path still scatters, and the cyclic/AMI (42 sites) and distributed
(`device_halo`) paths are untouched. Those are outside the incompressible single-GPU simpleFoam path and
are recorded in the manifest rather than silently fixed.

## Provenance and drift

`manifest/simpleFoam.yaml` is generated by `tools/of_manifest.py`. It has two halves, kept strictly apart:

- **DERIVED** — runtime-selection tables, dictionary keys with defaults and `file:line`, and the resolved
  selection list for each validation case. Queried from `ofscan`'s index of the OpenFOAM tree. Never edited.
- **CURATED** — the classification and brae status. A judgement about *our* code, which cannot be derived.

```bash
python3 tools/of_manifest.py simpleFoam                      # regenerate
python3 tools/of_manifest.py simpleFoam --check manifest/simpleFoam.yaml   # non-zero if drifted
```

A hand-written "OpenFOAM does X" claim rots silently. A derived one cannot.
