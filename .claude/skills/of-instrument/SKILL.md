---
name: of-instrument
description: Instrument OpenFOAM itself to localise a disagreement. Use when brae and OpenFOAM differ and the gap will not localise from brae's side - when a final field is out by some amount and you are about to reason about which term causes it. Builds a copy of OpenFOAM's own class with writes added and its equations untouched, so every intermediate has an oracle.
---

# Instrumenting OpenFOAM

When a gap will not localise, **stop reasoning and go read OpenFOAM's own numbers.** A term computed at
OpenFOAM's converged fields cannot reveal the sign it is added with; a final field that is 2.8e-02 out
cannot tell you which of eight terms did it. Intermediates can.

This has paid every time it was used. `tools/dumpKEpsilon` turned "epsilon is 2.8e-02 out, and dropping
the production term halves it, but the initial residual matches to 0.16%" — two facts that could not be
reconciled by argument — into four named defects in an afternoon.

## The pattern

Existing examples: `tools/dumpPEqn` (a solver), `tools/dumpKEpsilon` (a turbulence model),
`tools/dumpScalarMatrix`.

1. **Copy OpenFOAM's own source** for the class into `tools/dump<Thing>/`. For a runtime-selected model,
   `sed`-rename the class (`kEpsilon` → `kEpsilonDump`) and register it with `makeRASModel(kEpsilonDump)`
   in a small `<Thing>Models.C`. `Make/files` + `Make/options` from the sibling tool; `wmake libso`.
2. **Add writes ONLY.** Do not alter an equation, a coefficient, or an order. What comes out must be what
   OpenFOAM used, not a reconstruction of it. Insert by line number rather than by text anchor if earlier
   edits have moved things.
3. **Gate the writes** on an iteration read from the environment (`BRAE_DUMP_STAGE_ITER`), so a case can
   run normally and dump one iteration.
4. **Select it from the gate script**: `RASModel kEpsilonDump;` plus `libs ("libdumpKEpsilon.so");` in
   `controlDict`. Skip the term-by-term comparison gracefully when the library is not built, and say so.

## What to dump

Dump more than seems necessary — the cheap ones are what turn "somewhere in this equation" into a name:

- **Every intermediate the equation is built from**: `gradU`, `divU`, `G`, each diffusivity.
- **The assembled system, twice**: before `relax()`/`boundaryManipulate()`, and after. Separating assembly
  from manipulation localised two defects at once.
- **`D()` and the source with `boundaryCoeffs` folded into their face cells** — and match brae's capture to
  the same definition, or the comparison measures the capture rather than the code.
- **The off-diagonals.** A per-cell `D`/source comparison cannot see them. They are what finally named the
  `corrected`-vs-`orthogonal` laplacian.
- **Each `fvm` term on its own**, before they are summed.
- **The mesh factors** the coefficient is a product of: `deltaCoeffs`, `nonOrthDeltaCoeffs`, `weights`,
  `magSf`, the interpolated diffusivity. When every factor agreed and the product did not, that was proof
  a *different factor* was being multiplied — which is exactly what it was.

## Reading the result

**Split the comparison before interpreting it.** A single relative-L2 hides the answer:

- **interior cells vs cells touching a patch** — exact inside and wrong at the boundary is a boundary face
  value, not a scheme. This is what named the stale `inletOutlet`.
- **wall cells vs the rest** — a disagreement on wall-adjacent cells is the wall treatment.
- **per patch** — with each patch's declared BC type printed beside it.
- **magnitudes and the worst cells by name**, with the patches each touches. `sum|brae| = 0` where
  `sum|OF|` is not says the coefficients are absent, not merely wrong — a different diagnosis entirely.

A ratio that is constant across the field (6.09x on every inlet cell) means one scalar factor is wrong,
not an accumulation.

## Afterwards

Keep the tool in the tree and say in `PORT.md` what it found. The next gap in the same component starts
from an instrument that already exists.
