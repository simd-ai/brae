# rhoSimpleFoam vertical slice

A self-contained port of `rhoSimpleFoam`, laid out like OpenFOAM's own tree, built as its own target with
its own tests.

## Why this exists

The pimpleFoam port was built by reusing the code `simpleFoam` already had. Most of the defects that cost
the most to find came out of that reuse, and they were all the same failure — an OpenFOAM algorithm
**reinterpreted** rather than transcribed, in shared code that `simpleFoam`'s own cases never exercised:

| defect | what OpenFOAM's source says |
|---|---|
| LUST convection | `LUST.H`: `weights() = 0.75*linear + 0.25*linearUpwind::weights()` — the blend is IMPLICIT. brae built a pure-upwind matrix and carried the blend as a deferred correction, so `rAU` was 11% wrong and the pressure coupling with it. |
| `relaxationFactors` | `solution.C`: `fieldRelaxDict_`/`eqnRelaxDict_`, and `select(isFinalIteration())` appends `Final`. brae had no notion of a final-corrector factor. |
| `meanVelocityForce` | `meanVelocityForce.C`: `gradP0_` + `dGradP_`, and `constrain()` zeroes the increment. brae had one accumulator, so the drive compounded. |
| SA `calculated` nut | `SpalartAllmaras::correctNut()`: `nut_ = nuTilda_*fv1` is a field assignment, so the PATCH value comes from `nuTilda`'s patch value. brae kept the value read from `0/nut` — 360x too large on a case whose `0/nut` is a placeholder. |

None of these announced itself. Each produced a converged, plausible field, and each was found only by
comparing against OpenFOAM and disbelieving the number.

So the rule here is **transcribe, don't paraphrase**, and the layout mirrors OpenFOAM's so that "is this
ported?" is a file-list question rather than a judgement call.

## Tiers

Not everything deserves the same treatment, and re-porting the parts that are already right would be
waste with a chance of new error.

**Tier A — shared, never re-ported.** Mesh, geometry, LDU, AMG-PCG, BiCGStab, DILU, dict and field
readers, device buffers. These are infrastructure rather than transcriptions of OpenFOAM algorithms, and
they are the most heavily tested code in the repo. The pressure solve in particular is done: AMG-PCG is
fast and solver-agnostic. Tier A is *included*, not copied.

**Tier B — copied, then re-validated on this solver's paths.** Boundary conditions, div/laplacian/grad
schemes, turbulence models. These *are* transcriptions. A copy is the right starting point — copying a
validated file cannot introduce a reinterpretation, because nobody is interpreting anything — but a copy
inherits latent defects, and "correct for `simpleFoam`" is not "correct for `rhoSimpleFoam`". The paths
this solver reaches that the incompressible ones do not are `rho` weighting through `phiHbyA`, the energy
equation's boundary set, and `rho.relax()`. Tests go there first.

**Tier C — ported fresh from OpenFOAM, host first.** The solver driver and anything solver-specific:
`UEqn`/`EEqn`/`pEqn`/`pcEqn`, the thermo. This is where reinterpretation bites, so this is where
transcription earns its keep. It is also small: OpenFOAM's whole `rhoSimpleFoam` driver is 446 lines.

## Provenance

Every copied file carries, in its header:

    // COPIED FROM <path> — identical as of <sha>. Tier B.

so that "which copies are still identical, and which have diverged and need a fix by hand" is a script
rather than archaeology. A fix to a Tier-B donor propagates by replacement, not by merge.

## Host first, then device

Each Tier-C piece is transcribed to host C++ and validated before any of it moves to the GPU. That gives
an oracle inside the repo: GPU against host, in-process, field by field — instead of instrumenting
OpenFOAM to dump stages, which is what tracing the pimpleFoam defects actually required.

The order is deliberate:

1. **Slice 0** — this folder builds and runs, reproducing the existing solver's output exactly. Proves the
   seam before anything changes. Nothing is ported yet.
2. **Slice 1+** — replace one Tier-C piece at a time with a fresh transcription, each validated against
   OpenFOAM *and* against the copied path it replaces.

Step 1 is not a formality. A folder that builds but silently diverges from the working solver would make
every later comparison meaningless, so it is checked against a real tutorial (`squareBend`) before any
transcription begins.
