# Pre-PIMPLE modular refactor plan

**Status:** design note (pre-implementation). Target branch: a fresh `refactor/pre-pimple`
cut *after* the current multi-GPU work is pushed to `main`.
**Reference:** OpenFOAM 2412 (`applications/solvers/incompressible/{simpleFoam,pimpleFoam,pisoFoam}`).
**Goal:** carve the two SIMPLE monoliths into reusable momentum / Rhie–Chow / pressure blocks so
that PIMPLE (and PISO, and later rhoPimpleFoam) is a thin orchestrator over the same blocks —
*without changing any numerics* (must stay bit-identical vs the CPU oracle).

This document is the refactor spec only. It deliberately does **not** implement PIMPLE; it makes the
code PIMPLE-ready. The PIMPLE delta is sketched in the last section so the block boundaries are chosen
to make that delta small.

---

## 1. Why refactor first

Today the solver logic lives in **two independent monoliths** that mirror each other stage-for-stage:

| File | Lines | Role |
|---|---|---|
| `src/applications/solvers/simpleFoam/device_simple_foam.cuh` | ~1,112 | single-GPU `DeviceSimpleSolver::step()` |
| `src/applications/solvers/simpleFoam/parallel_device_simple.cuh` | ~1,988 | multi-GPU `ParallelDeviceSimple::step()` (halo-coupled mirror) |

`step()` is a ~600-line straight-line function in each. There is **no shared "UEqn object" or "pEqn
object"** — the momentum assembly, Rhie–Chow (`rAU`/`HbyA`/`phiHbyA`), pressure loop, and corrector are
inlined. PIMPLE needs to run the momentum+pressure+corrector sequence inside an **outer loop**
(`nOuterCorrectors`), with the pressure part re-run inside an **inner PISO loop** (`nCorrectors`). You
cannot express that cleanly over a straight-line body — you would have to duplicate the inner 400 lines,
twice (single + multi GPU). Extracting the blocks first turns the PIMPLE loop into
`for outer { assemble; predict?; for corr { HbyA; pEqn; correct; } } turbulence`.

**Payoff:** PISO falls out for free (PISO = PIMPLE, `nOuterCorrectors=1`, no relaxation), and every future
transient/compressible solver reuses the same blocks.

---

## 2. The shared 17-stage flow (both monoliths, identical order)

Both `step()`s execute this sequence. Line numbers below are the **single-GPU** file
(`device_simple_foam.cuh`); the parallel file has the same stages with `halo_.exchange(...)` interleaved
(see `parallel_device_simple.cuh:1286` onward, stage banners already present).

| # | Stage | Single-GPU lines | Reusable? | Block |
|---|---|---|---|---|
| 1 | BC updates (inletOutlet / mixed / symmetry / totalP) | 342–367 | yes | **A** |
| 2 | `nuEff = nu + nut`, face interp, wall-function boundary nuEff | 369–393 | yes | **A** |
| 3 | explicit `divDevReff` stress source | 394–397 | yes | **A** |
| 4 | `grad(p)` (+ cyclic/AMI contributions) | 399–404 | yes | **A** |
| 5 | momentum matrix coeffs: `div(phi,U) − laplacian(nuEff,U)`, bounded, interface, fvOptions Sp, porosity | 405–447 | yes | **A** |
| 6 | boundary coeffs + relax diagonal | 448–474 | yes | **A** |
| 7 | explicit source `relaxSrc[3]` (relax·Uold + stress + linUpwind/nonOrth corr + body force + fvOptions + porosity + MRF + AD + rotor) + per-component **predictor solve** | 476–627 | yes | **A** |
| 8 | `cmptAv(iC)` + `rAU` (+ SIMPLEC `rAtU`) | 628–657 | yes | **B** |
| 9 | meanVelocityForce.correct (post-predictor) | 658–675 | yes | **B** |
| 10 | `HbyA = rAU·H(U)` (+ interface H, constrainHbyA, slip/sym) | 676–708 | yes | **B** |
| 11 | `phiHbyA = flux(HbyA)` (+ interface flux, MRF makeRelative, adjustPhi, SIMPLEC flux corr) | 709–764 | yes | **B** |
| 12 | pressure loop: `laplacian(rAtU,p) == div(phiHbyA)`, `nNonOrth` passes, AMG/Jacobi-PCG solve | 765–849 | yes | **C** |
| 13 | flux reconstruction `phi = phiHbyA − pEqn.flux()` (+ nonOrth ffc, interface flux corr) | 850–863 | yes | **C** |
| 14 | `p` relaxation | 864–865 | yes | **C** |
| 15 | velocity corrector `U = HbyA − rAU·grad(p)` (+ limitVelocity) | 866–881 | yes | **C** |
| 16 | turbulence `correct()` (kEps / kOmegaSST(+LM) / SA) | 883–909 | yes | **D** |
| 17 | continuity error report | 910–920 | yes | (driver/report) |

**Observation that makes this cheap:** the stages are *already* well-delimited — each has a banner
comment and most already scope their temporaries in `{ ... }`. The refactor is mostly "lift a stage into a
private method taking/returning the shared device buffers," not a logic rewrite.

---

## 3. Target architecture

### 3.1 A `StepWorkspace` struct (kills the 15-out-param problem)

Bundle the per-step working buffers that cross block boundaries into one struct owned by the solver
(persistent, so addresses stay graph-stable — the pressure V-cycle graph is keyed on `diagCp_`'s address,
see `device_simple_foam.cuh:767`). Keep the *persistent* matrix buffers (`pD_/pU_/pL_/diagCp_/bp_`,
`Uk_`, `dp_`, `phiInt_/phiBnd_`) where they already live as members; the workspace holds the transient
per-step intermediates:

```
struct StepWorkspace {
    DeviceBuffer<scalar> nuEff, nuEff_f, nuEffBnd;      // stage 2
    DeviceBuffer<scalar> gx, gy, gz;                    // grad(p), reused each nonOrth pass
    DeviceBuffer<scalar> mDiagR, mUp, mLo;              // relaxed momentum matrix (stages 5-6)
    DeviceBuffer<scalar> relaxSrc[3], iC[3], bCb[3];    // stage 7 explicit source + boundary coeffs
    DeviceBuffer<scalar> rAU, rAtU, drAtU, cmptAvIC;    // stage 8
    DeviceBuffer<scalar> HbyA[3], phiHi, phiHb;         // stages 10-11 (phiHi=internal, phiHb=boundary phiHbyA)
    // interface off-diag sums (cyclic/AMI/processor) already needed across blocks:
    DeviceBuffer<scalar> cycSumOff, amiSumOff;
    DeviceSolverPerf pp, pp0;                           // pressure perf, for the residual report
};
```

### 3.2 Private block methods (identical signatures in *both* classes)

Decompose each `step()` into these private members. Same names, same order, in `DeviceSimpleSolver`
**and** `ParallelDeviceSimple`, so the eventual PIMPLE orchestrator reads identically in both files
(the parallel bodies just contain the extra `halo_.exchange` calls internally).

```
// Block A — momentum. Assembles + (optionally) solves the predictor. Returns per-component U residuals.
void assembleMomentum(StepWorkspace& w);          // stages 1-6  (matrix + boundary coeffs + relax diag)
void momentumSource  (StepWorkspace& w);          // stage 7 assembly of relaxSrc[3] (no solve)
UResidual momentumPredictor(StepWorkspace& w);    // stage 7 solve loop (skippable -> momentumPredictor off)

// Block B — Rhie-Chow. Builds rAU/rAtU, HbyA, phiHbyA from the current U (+ interface + SIMPLEC).
void constructHbyA(StepWorkspace& w);             // stages 8-11

// Block C — pressure corrector. nNonOrth pressure passes + flux reconstruction + p-relax + U corrector.
PResidual pressureCorrector(StepWorkspace& w, bool finalCorrector);  // stages 12-15

// Block D — turbulence (already a set of free functions; wrap for symmetry).
void correctTurbulence(StepWorkspace& w);         // stage 16

// continuity error -> the Residual struct (stage 17), stays in step()/driver.
```

The existing `step()` becomes:

```
DeviceSimpleResidual step() {          // SIMPLE = one outer iteration
    StepWorkspace w;
    assembleMomentum(w);
    momentumSource(w);
    res.U = momentumPredictor(w);
    constructHbyA(w);
    res.p = pressureCorrector(w, /*finalCorrector=*/true);   // nNonOrth loop lives inside
    correctTurbulence(w);
    res.cont = continuityError();
    return res;
}
```

Behavior is **bit-identical** to today — same calls, same order, just relocated.

### 3.3 Keep two classes, mirror the decomposition (do NOT unify yet)

The single- and multi-GPU paths differ in the halo/interface plumbing throughout, and the pressure solve
differs (AMG-PCG vs distributed Jacobi/AMG-per-rank). Attempting a shared base class in this refactor
would fight the interface differences and risk the bit-identical guarantee. **Mirror the block boundaries
in both classes** now; a shared abstract interface is a *later, optional* step once PIMPLE is proven on
both.

---

## 4. PIMPLE-readiness hooks (choose block boundaries so the PIMPLE delta is tiny)

Each block must expose exactly the seams PIMPLE needs. Build these in during the refactor even though
SIMPLE does not exercise them (they stay no-ops / defaults):

1. **`assembleMomentum` / `momentumSource` — a ddt seam.**
   Add an (initially empty) hook `addDdt(w)` called at the end of matrix assembly (diagonal) and source
   assembly (rhs). For SIMPLE it does nothing. For PIMPLE it adds `V/deltaT` to the diagonal and
   `V/deltaT · Uⁿ` (Euler) — or the backward/CN coefficients — to `relaxSrc`. **No ddt operator exists
   yet** (`device_fvm.cu`/`device_fvc.cu` have div/laplacian/grad/interp/flux but no ddt) — the operator
   is PIMPLE-phase work; the refactor only reserves the call site.

2. **`constructHbyA` — a ddtCorr seam on `phiHbyA`.**
   Reserve a hook to add the Rhie–Chow transient flux correction
   `phiHbyA += fvc::interpolate(rAU)·fvc::ddtCorr(U, phi)` right after `phiHi/phiHb` are built
   (currently `device_simple_foam.cuh:709–716`). No-op for SIMPLE.

3. **`pressureCorrector(w, finalCorrector)` — the `finalCorrector` flag already gates p-relaxation.**
   In SIMPLE it is always `true` (relax every step). In PIMPLE, relaxation is applied on **non-final**
   outer correctors and skipped on the final one — so route the existing `deviceScale(dp_, relaxP)` +
   `deviceAxpy(1-relaxP, pPrev, dp_)` (`:864–865`) through `finalCorrector`. Same for momentum relax in
   `assembleMomentum` (the relax factor becomes 1.0 on the final outer corrector).

4. **`momentumPredictor` returns a skippable stage.** PIMPLE's `momentumPredictor off` simply doesn't
   call it (HbyA is then built from the previous U). Make sure `constructHbyA` does not depend on the
   predictor having run beyond reading `Uk_`.

5. **`correctTurbulence` gated by a `pimple.turbulentFinalIter()` flag** — transient PIMPLE corrects
   turbulence only on the final outer corrector; SIMPLE corrects every step. A bool parameter suffices.

6. **Old-time state.** Reserve members `Uold_[3]`, `Uold2_[3]`, `phiOld_` and a `rotateOldTime()` method
   (no-op / unused in SIMPLE). PIMPLE's time loop calls `rotateOldTime()` once per time step before the
   outer loop. Wiring the buffers in now (even if unused) keeps the PIMPLE diff localized.

If these six seams exist, the PIMPLE orchestrator is ~40 lines wrapping the same blocks (see §7).

---

## 5. Refactor sequence (ordered, each step behavior-preserving)

Do these **one block at a time, re-validating bit-identical after each** against the existing oracle
before moving on. Never combine a relocation with a numeric change.

1. **Introduce `StepWorkspace`** in the single-GPU class; move the transient buffers into it but keep
   `step()` a single function (just referencing `w.` instead of locals). Validate.
2. **Extract Block C** (`pressureCorrector`) first — it is the most self-contained (the `nNonOrth` loop is
   already a delimited `for`, `:782`). Validate.
3. **Extract Block B** (`constructHbyA`). Validate.
4. **Extract Block A** into `assembleMomentum` + `momentumSource` + `momentumPredictor`. This is the
   largest stage; the per-component predictor loop (`:538–627`) becomes `momentumPredictor`. Validate.
5. **Extract Block D** (`correctTurbulence`) — mostly a wrapper over the existing free functions. Validate.
6. **Add the six §4 seams as no-ops** (empty `addDdt`, unused old-time members, `finalCorrector` /
   `turbulentFinalIter` params defaulting to the SIMPLE behavior). Validate bit-identical again.
7. **Repeat 1–6 for `ParallelDeviceSimple`**, mirroring the exact method names/boundaries. Validate on the
   multi-GPU oracle (the halo exchanges stay *inside* each block; e.g. `constructHbyA` owns its
   `halo_.exchange(HbyA[k])` at `parallel_device_simple.cuh:817`).

Each of steps 2–5 is a pure code-motion commit — small diff, trivially reviewable, oracle-verified.

---

## 6. Hard constraints (do not break)

- **Bit-identical numerics.** This project validates to machine precision vs the CPU oracle
  (`test_gpu_resident_turb`, and the multi-GPU predictor tests). Every refactor commit must reproduce the
  same residual trace. If a diff appears, it's a relocation bug, not "acceptable drift."
- **Graph-stable buffer addresses.** The pressure V-cycle CUDA graph is captured once and replayed, keyed
  on persistent buffer addresses (`device_simple_foam.cuh:767–768`). Keep `pD_/pU_/pL_/diagCp_/bp_` (and
  the `Uk_`) as persistent **members**, not inside `StepWorkspace`, or `useGraph` breaks.
- **Interface / rotational ordering.** The rotational cyclic/AMI paths depend on snapshotting `U_old`
  before the per-component solve (`:483–492`) so the x↔y rotation mixing is order-independent. Keep that
  snapshot logic *inside* `momentumSource`/`momentumPredictor` — do not let the block split reorder it.
- **SIMPLEC coupling.** `rAtU`/`drAtU` thread from Block B into Block C (the pEqn diffusivity and the
  one-time flux/HbyA correction). They live in `StepWorkspace`; don't recompute per block.

---

## 7. The PIMPLE delta this refactor unlocks (for reference, not this branch)

After the refactor, adding PIMPLE is roughly:

```
// new operator work (device_fvm.cu / device_fvc.cu): fvm::ddt (Euler/backward/CN) + fvc::ddtCorr
// new state: Uold_[3], Uold2_[3], phiOld_  (+ rotateOldTime)
// new driver: physical time loop + Courant number + adjustableRunTime

DevicePimpleResidual step() {                 // one TIME step
    StepWorkspace w;
    for (int oc = 0; oc < nOuterCorrectors_; ++oc) {
        const bool finalIter = (oc == nOuterCorrectors_ - 1);
        assembleMomentum(w);                  // addDdt() now active: diag += V/dt
        momentumSource(w);                    //   relaxSrc += V/dt * Uold (+ backward/CN)
        if (momentumPredictor_) momentumPredictor(w);
        for (int corr = 0; corr < nCorrectors_; ++corr) {
            constructHbyA(w);                 // ddtCorr seam active: phiHbyA += interp(rAU)*ddtCorr(U,phi)
            pressureCorrector(w, /*finalCorrector=*/corr == nCorrectors_-1 && finalIter);
        }
        correctTurbulence(w, /*turbulentFinalIter=*/finalIter);
    }
    rotateOldTime();                          // Uold2<-Uold<-U, phiOld<-phi  (at end of time step)
    return res;
}
// SIMPLE stays: nOuterCorrectors=1, nCorrectors=1, momentumPredictor=on, finalCorrector always true,
//   addDdt/ddtCorr are no-ops -> byte-for-byte the current SIMPLE.
// PISO = PIMPLE with nOuterCorrectors=1 and no relaxation.
```

That is why the block boundaries in §3.2 are drawn exactly here: the PIMPLE loop is a wrapper, and the
only genuinely new numerics are the ddt operator and ddtCorr flux term.

---

## 8. Deliverables checklist for the `refactor/pre-pimple` branch

- [ ] `StepWorkspace` introduced (single-GPU), buffers relocated, oracle bit-identical.
- [ ] Blocks C, B, A, D extracted (single-GPU), each its own oracle-verified commit.
- [ ] Six §4 seams present as no-ops (single-GPU); SIMPLE still bit-identical.
- [ ] Same decomposition mirrored in `ParallelDeviceSimple`; multi-GPU oracle bit-identical.
- [ ] `step()` reduced to the orchestrator skeleton in both classes.
- [ ] No change to `device_fvm.cu`/`device_fvc.cu` (ddt operators are PIMPLE-phase, not here).
- [ ] Doc: update `docs/solvers/` with the new block API once landed.

**Explicitly out of scope for this branch:** ddt operators, old-time rotation logic, Courant/time loop,
transient turbulence, the PIMPLE control-dict parsing. Those are the PIMPLE branch that builds on top.
