# `UEqn.H` — transcription, phases 1–2

Working document for the momentum equation. **No code yet, deliberately.** Phase 1 is the quote, phase 2
is the enumeration; phase 3 writes host C++ against them. The point of stopping here is that every
expensive defect in the pimpleFoam port was a term that was *paraphrased* or *not reached*, and both are
visible at this stage for the price of reading.

---

## Phase 1 — QUOTE

`applications/solvers/compressible/rhoSimpleFoam/UEqn.H`, OpenFOAM v2412, verbatim and complete:

```cpp
    // Solve the Momentum equation

    MRF.correctBoundaryVelocity(U);

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(U)
     ==
        fvOptions(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);
```

21 lines, 8 operations. This block is the specification; phase 3 is judged against it, not against a
description of it.

Note what is **absent** and would be present in the incompressible `simpleFoam`: nothing. And what is
present here and not there: the `rho` arguments. `MRF.DDt(rho, U)`, `divDevRhoReff`, `fvOptions(rho, U)`
are all density-weighted. That is the whole delta of this file, and it is where a copied incompressible
implementation goes wrong silently.

---

## Phase 2 — ENUMERATE

One row per operation. `tier` follows `slice/README.md`: **A** shared and never re-ported, **B** copied
then re-validated on this solver's paths, **C** transcribed fresh.

| # | OF operation | OF implementation | brae counterpart | tier | status |
|---|---|---|---|---|---|
| 1 | `MRF.correctBoundaryVelocity(U)` | `MRFZone.C` | `device_mrf.cuh` | B | present; **not reached** by any rhoSimpleFoam tutorial (no `MRFProperties` in any of the 6) |
| 2 | `fvm::div(phi, U)` | `gaussConvectionScheme.C:78` `fvmDiv` | `deviceDiv{Upwind,Central,LimitedV}Coeffs` (`device_mesh.cuh`) | B | present. **Weights come from the scheme named in `fvSchemes`** — this is where the LUST defect lived |
| 3 | `MRF.DDt(rho, U)` | `MRFZoneList.C:187,210` | `deviceMrfCoriolis` (`device_mrf.cuh`) | B | **GAP — takes no `rho`.** Signature is `(mrf, V, Ux, Uy, Uz, kk, src)`; OF's is ρ-weighted. Not reached by the tutorials, so latent |
| 4 | `turbulence->divDevRhoReff(U)` | `linearViscousStress.C:107` = `-fvc::div((alpha*rho*nuEff)*dev2(T(grad U))) - fvm::laplacian(alpha*rho*nuEff, U)` | `deviceDivDevReff` + `deviceLaplacianCoeffs` | B | present and **ρ-weighting already handled** — the driver passes `muEff = mu + rho*nut`, cited to `linearViscousStress.C` at `device_simple_foam.cu:891` |
| 5 | `fvOptions(rho, U)` | `fvOptionList.C` | `readFvOptions` / `device_fvoptions.cuh` | B | present. Reached: 2 of 6 tutorials ship `fvOptions` (`aerofoilNACA0012`, `angledDuctExplicitFixedCoeff`) |
| 6 | `UEqn.relax()` | `fvMatrix.C:1250` → `psi_.select(isFinalIteration())`, then `:1102` `relax(alpha)` | `deviceRelaxDiag` (`device_simple.cuh`) | C | present. The `select()` naming is the `*Final` rule fixed this cycle — steady SIMPLE never sets `isFinalIteration`, so it takes the base factor |
| 7 | `fvOptions.constrain(UEqn)` | `fvOptionList.C` | folded into the source assembly | C | present (`meanVelocityForce::constrain` is the one that matters, fixed this cycle) |
| 8 | `solve(UEqn == -fvc::grad(p))` | `fvMatrix::solve` + `gaussGrad.C:44,122` | `deviceGaussGrad` + `deviceJacobiBiCGStab` (Tier A solver) | C | present. `== -fvc::grad(p)` means `source += V*(-grad p)` |
| 9 | `fvOptions.correct(U)` | `fvOptionList.C` | post-solve correction | C | present |

### What phase 2 found, before any code was written

**One gap (#3).** `deviceMrfCoriolis` has no `rho` argument, so a compressible MRF case would get an
incompressible Coriolis term. No rhoSimpleFoam tutorial ships `MRFProperties`, so it is latent rather
than active — which is exactly how the SA `calculated`-nut defect sat unnoticed until a case reached it.
It should be a startup refusal until the ρ-weighted form exists, not a silently wrong term.

**One confirmation (#4).** The ρ-weighting on the viscous stress is already correct and already carries
its OF citation. That is the difference between a term that was transcribed and one that was inferred —
worth noting, because it is the pattern the rest should follow.

**One dependency (#2).** The convection weights come from `fvSchemes`, and that is where the LUST defect
lived: OF's implicit weights vs a deferred correction. Phase 4's oracle comparison must check the
assembled diagonal, not just the interpolated face value — the two agree even when the matrix is wrong.

### Coverage of this file by the tutorials

Of the 6 rhoSimpleFoam tutorials: 5 RAS (4 `kEpsilon`, 1 `kOmegaSST`), 1 laminar; 2 with `fvOptions`;
**0 with MRF**. So rows 1 and 3 are unreachable by the tutorial set and cannot be validated by it —
another reason to refuse rather than approximate them.

---

## Next (phase 3)

Transcribe rows 6–9 (tier C) as host C++, one function per OF operation, same names, no fusion. Rows 2,
4, 5 are tier B: copy with provenance and re-validate against the oracle on the ρ-weighted paths. Row 3
gets a refusal.

Phase 4 then asserts the assembled matrix against the OF oracle's `stage_mDiag` / `stage_mSrc` /
`stage_UIC` / `stage_UBC`. That comparison needs the **post-fold** diagonal (`stage_mDiagFold*`), because
OF's `fvMatrix::D()` includes the boundary `internalCoeffs` and brae's pre-fold dump does not — without
it every boundary cell shows a difference that is the missing term rather than a defect.

It is only as sharp as brae's run-to-run floor, which is now bit-identical at iteration 1 but not beyond
(the remaining scatter atomics). Phase 4 comparisons should therefore be made at **iteration 1**, where
the floor is exact.

---

## Phase 3 — TRANSCRIBE: it turned out to be a verification pass, not a rewrite

Phase 3 was planned as "write tier-C rows 6-9 as host C++". Reading OpenFOAM's source alongside brae's
first — which is the phase's actual first step — showed that **all four are already faithfully
transcribed, and re-porting them would have been waste with a chance of new error.**

| row | OF | verdict |
|---|---|---|
| 6 `UEqn.relax()` | `fvMatrix.C:1102-1250` | **already faithful, including the part most likely to be paraphrased.** OF's boundary handling is ASYMMETRIC: it adds `cmptMax(cmptMag(iCoeffs))` and later removes `cmptMin(iCoeffs)` — different quantities, so the net effect on `D` is deliberately non-zero. `device_simple.cuh:72-73` documents exactly that and `device_simple.cu:157-158` implements both sides. |
| 7 `fvOptions.constrain(UEqn)` | `fvOptionList.C` | already faithful; `meanVelocityForce::constrain`'s `gradP0_ += dGradP_; dGradP_ = 0` was fixed and tested this cycle |
| 8 `solve(UEqn == -fvc::grad(p))` | `fvMatrix::solve`, `gaussGrad.C:44,122` | already faithful; `== -fvc::grad(p)` is `source += V*(-grad p)`, and the assembled RHS was verified against the OF oracle this cycle (matched to 1.7e-07) |
| 9 `fvOptions.correct(U)` | `fvOptionList.C` | already faithful; its call POSITION (the tail of `pEqn.H`) was the second `meanVelocityForce` defect, fixed this cycle |
| 3 `MRF.DDt(rho, U)` | `MRFZoneList.C:210` | **the one real gap — now a REFUSAL.** Not merely unimplemented: the compressible driver never opened `constant/MRFProperties` at all, so an active rotating zone ran with no rotation and said nothing. And it is not a copy of the incompressible term either, because OF's overload here is density-weighted and `deviceMrfCoriolis` takes no `rho`. |

So the deliverable of phase 3 for this file is one refusal, not four transcriptions. That is the method
working as intended: enumerate first, and most of the answer is "already done, here is the citation" —
which is only visible because rows 6-9 carry their OpenFOAM references. A row without a citation cannot
be checked this way, and would have had to be re-ported to be trusted.

### Verified after the change

* `tools/slice_provenance.py` — the slice copy is identical to its donor by **diff**, not by a sha (a sha
  goes stale the moment the donor is touched, and a stale one is indistinguishable from a current one).
* The mixed run: donor and slice on `compressible/rhoSimpleFoam/squareBend`, iteration 1, **bit-identical
  on U, p, T, rho and phi**. That comparison is only meaningful because the reduction fix made iteration 1
  exactly reproducible; before it, the same binary twice differed by 4.18e+01 in U.
* The MRF refusal fires, naming `MRFZoneList.C:210` and both dropped effects.

## Next (phase 4)

Nothing in rows 6-9 needs transcribing, so phase 4 moves to the terms whose *inputs* are compressible and
therefore untested by the incompressible cases: `fvm::div(phi, U)`'s weights (row 2, where the LUST defect
lived) and `divDevRhoReff`'s `muEff` (row 4). Assert the assembled matrix against the oracle's
`stage_mDiagFold*` / `stage_mRhsFold*` / `stage_UIC` / `stage_UBC` at **iteration 1**.
