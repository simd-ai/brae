---
name: of-port
description: Port an OpenFOAM component into brae. Use when transcribing a solver, equation, turbulence model, boundary condition or scheme from OpenFOAM source - covers the _cpp host reference first, its gate against real OpenFOAM, the whole-case end-to-end run, and only then the CUDA module. Invoke before writing any port code.
---

# Porting an OpenFOAM component

The order is not negotiable and it is the whole method: **`_cpp` reference first, the whole case running
end to end, then ONE CUDA module at a time.** A probe is not an end-to-end run. Skipping ahead is what
"naive porting, checked case by case" means, and it leaves the real surface unmeasured.

## 1. Before writing anything

- Read OpenFOAM's own source for the component. Not a comment about it, not a brae file claiming to mirror
  it, not memory. `/usr/lib/openfoam/openfoam2412/src` and `.../applications/solvers`.
- Add or update the component's entry in `manifest/<solver>.yaml` via `tools/of_manifest.py`. The
  `classification` field is where **HOST_ONLY vs GPU_REQUIRED** is recorded — decide it once, there, not
  ad hoc per file. A component that does not need to be on the device stays on the host and that is a
  finished state, not a deferred one.
- Regenerate and check: `python3 tools/of_manifest.py <solver> --check manifest/<solver>.yaml`.

## 2. The `_cpp` reference

Transcribe OpenFOAM's host code into `<name>_cpp.cu` / `<name>_cpp.cuh`. **Transcribe, never copy** — the
shape and the term order follow OpenFOAM's file so the two can be read side by side.

Header basenames are **globally unique** across the whole tree: brae puts every source dir on one include
path, so `UEqn_cpp.cuh` in two solver directories silently resolves to whichever came first. Prefix by
solver (`rhoUEqn_cpp.cuh`). This has already cost a session.

### The checklist that catches the recurring defect

Every defect found in this port so far has been one of these. Walk them explicitly:

1. **`updateCoeffs()`.** For every field the component solves or reads, list every boundary condition the
   validation cases use, open each one's `.C`, and find its `updateCoeffs()`. OpenFOAM calls it from
   `fvMatrix`'s constructor, so it runs at *every assembly*. brae has no object registry, so the port
   must call the equivalent explicitly, at the same point. Five separate defects have been exactly this:
   an inletOutlet evaluated on a seeded `valueFraction`; k and epsilon contributing nothing to their own
   systems; turbulent inlets frozen at the case file's `value` when OpenFOAM recomputes them from U and k;
   `flowRateInletVelocity` frozen at its seed rho.
2. **Which field, exactly.** `thermo.rho()` is not the solver's relaxed `rho`. A patch field's boundary
   values are not its face cells' values. `phi` may be the mass flux in one line and the volumetric flux
   in the next (`compressibleTurbulenceModel::phi()` divides by `interpolate(rho)`).
3. **Scheme flags come from the case.** Resolve `fvSchemes` — `laplacianSchemes`, `divSchemes`,
   `gradSchemes` — and pass what it says. A defaulted flag is a silent substitution: `Gauss linear
   orthogonal` where the case asked for `corrected` changes both the implicit coefficient and an explicit
   source, and on a near-orthogonal mesh it hides at 1e-06.
4. **Interpolate the product or the factors?** `fvc::interpolate(a*b)` and
   `fvc::interpolate(a)*fvc::interpolate(b)` differ on non-uniform fields, and OpenFOAM uses both, in the
   same solver, a few lines apart.
5. **Refuse what is not ported**, by name, with the OpenFOAM file it came from.

## 3. Gate the `_cpp` component

Use the `of-gate` skill. Machine precision (~1e-15) is the standard for a `_cpp` component compared
against OpenFOAM's own intermediates on OpenFOAM's own inputs. If it will not localise, use the
`of-instrument` skill rather than reasoning about the gap.

## 4. Run the whole case end to end

Not a probe. The `_cpp` driver runs the real case with the case's own dictionaries and is compared to real
OpenFOAM. **Compare at convergence, not at a tutorial's `endTime`** — comparing at an arbitrary iteration
compares two trajectories. On squareBend that is the difference between 2.8e-02 and 1.8e-03; on the
turbulent sbMatched fixture, between 7.9e-04 at 100 iterations and 1.08e-04 at 400.

Do not neutralise part of the case to make the gate pass without saying so in the script, in `PORT.md`,
and in the manifest's `validation` text. A neutralised inlet is a hole in the coverage, not a passing gate.

## 5. Only now, CUDA

One module at a time, with the `_cpp` component as the in-repo oracle:

- port ONE module to `.cu`/`.cuh`,
- build the MIXED binary (`_cpp` everywhere else),
- gate the mixed build against both the `_cpp` oracle and real OpenFOAM,
- only then take the next module.

Never port two at once: when the answer moves, you want one candidate.

## 6. Record it

- `PORT.md` for the solver: what was found, with the measurement that found it. A finding without a number
  is an opinion.
- The manifest's `validation` text: what the gate asserts AND what it does not claim.
- Findings that belong to another component go in `PORT.md`'s open-findings section rather than being
  silently fixed out of scope.
