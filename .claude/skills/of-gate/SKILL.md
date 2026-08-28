---
name: of-gate
description: Build or repair a brae validation gate against real OpenFOAM. Use when writing a test that compares brae to OpenFOAM, when choosing a bound, when adding a control, or when a gate passes but you are not sure it would fail. Covers oracles, discriminating controls, fail-proofs, and the traps that have produced false results here.
---

# Building a gate

A gate that cannot fail is not a gate. Three parts, all required.

## 1. An oracle, not an expectation

The right-hand side of the comparison is **something OpenFOAM produced**, never a hand-computed number:

- `<solver> -postProcess -func "writeObjects(...)"` builds a field set without solving — this is how
  `createFields` is gated.
- `tools/dumpPEqn` and `tools/dumpKEpsilon` are instrumented copies of OpenFOAM's own solver and model
  that write intermediate stages. See the `of-instrument` skill to build another.
- A full OpenFOAM run of the same case, from the same start, for the same number of iterations.

Write ASCII at `writePrecision 15`. At the fixture's shipped 6 digits a comparison bottoms out near 1e-06
and hides real disagreement underneath it.

## 2. A control that discriminates

State what the gate would fail on, then **prove it fails on it**. A bound both the right and the wrong
implementation pass proves nothing.

- **Two plausible forms**: assert the right one matches AND the wrong one does not. `createFields` asserts
  `linearInterpolate(rho*U)&Sf` matches and that `interpolate(rho)*flux(U)` — the form the next file uses —
  is 11 orders worse.
- **Term sweep**: drop each term of an equation in turn; every one must move the answer.
- **The fixture must be able to discriminate.** A uniform-rho state makes the two flux forms algebraically
  identical, so a gate run from a uniform `0.orig` passes whichever is implemented. Develop the fields
  first and **assert the non-uniformity** rather than trusting a comment.
- **The start state must fail.** If the initial field passes the bound, the gate is measuring nothing.

**Re-examine controls when the baseline improves.** A control with an absolute bound can go dead: the
kEpsilon `divU` control asserted `> 1e-3` and passed throughout the period the closure itself was 2.8e-02
out — it could not tell a working `divU` from a broken one because everything was broken. Express it
relative to the correct answer (`wrong > 1e3 * right`) with an absolute floor, so a fixture that *cannot*
discriminate fails instead of passing vacuously.

## 3. A fail-proof

Before believing a green gate: break the thing it tests, watch it go red, restore, watch it go green.
Record the red number in the commit or `PORT.md`.

## Bounds

- A `_cpp` component against OpenFOAM's own intermediates: **machine precision**, ~1e-15. Anything looser
  is a defect not yet found, not a tolerance.
- End-to-end, both codes running their own outer iteration: looser, and the bound must be justified in the
  script by what was measured — including **at convergence, not at a tutorial `endTime`**.
- Never round a bound up to accommodate a result. If it does not meet the bound, say so and leave it red
  or unregistered, with the reason in `CMakeLists.txt`.

## What the gate does NOT claim

Write it down, in the script and in the manifest's `validation` text. If p and T barely move on the
fixture so no bound could discriminate, report them and gate on U — and say that is what you did. If part
of the case is neutralised (an inlet replaced with a `fixedValue`), that is a hole in coverage: name it in
the script, in `PORT.md`, and in the manifest.

Where a structural difference is real but benign, **assert around it and report it beside** — for example
OpenFOAM's `epsilonWallFunction` derives from `fixedValue` while brae's rows are written by `setValues`,
so the assembled wall rows differ while the solved values agree. Assert off the wall, print the wall
figure. Asserting the union would assert a difference that is not an error; dropping the cells silently
would hide one that might be.

## Traps that have produced false results here

- **`ctest` does not rebuild.** Always `cmake --build` before measuring. A stale test binary reported a
  failure that had already been fixed.
- **Never rebuild while `ctest` is running.** It invalidates the whole run. Two sweeps were discarded for
  this in one session.
- **A build-cache-only commit breaks a bisect.** A bisect whose predicate depends on code the "culprit"
  commit introduced is invalid; so is one that lands on a CMakeLists-only change.
- **Nested heredocs collide.** Writing a gate script from inside a heredoc with the same marker silently
  truncates. Use the Write tool for gate files.
