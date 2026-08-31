# The rhoSimpleFoam refusal census, adjudicated

Every refusal on the rhoSimpleFoam paths (host _cpp mirror, CUDA mirror, shipped legacy binary and
the shared envelopes), surveyed by five area agents, checked adversarially, and ADJUDICATED against
the tree as of commit 2ecfe0f -- the disputes and missed items were re-verified file by file, so a
classification here is a read of the code, not a survey answer. The vocabulary:

- **GENUINE**     the refusal is correct and load-bearing; behind it is a real unported feature
- **OVER-BROAD**  refuses more than the actual gap
- **STALE**       the capability exists in the tree; only the refusal was never lifted
- **DEAD**        unreachable, harmless (an earlier guard catches it first)
- **HOLE**        a guard that fails to fire, or a silent divergence with no guard at all --
                  the silent-substitution class this project exists to catch
- **RESOLVED**    fixed, ported or lifted in the 2026-08-31 session (commit named in the note)

The original survey counted 120 refusals (GENUINE 67 / OVER-BROAD 13 / STALE 9 / DEAD 31); the
adversarial check disputed 35 classifications and reported 34 missed items; the adjudication ruled
on every one of those 69: GENUINE 31, HOLE 21, RESOLVED-THIS-SESSION 9, REJECTED (the checker was
wrong) 5, OVER-BROAD 2, STALE 2. Twenty-two entries of the original survey were closed by this
session's work before the adjudication ran; they are marked below.

## createFields

### Adjudicated verdicts (disputes and missed items)

| # | kind | item | verdict |
|---|---|---|---|
| 1 | dispute | cyclic/cyclicAMI/processor T patch refused via the T->he whitelist | **GENUINE** |
| 2 | dispute | T fixedGradient cannot be mapped to he | **RESOLVED-THIS-SESSION** |
| 3 | dispute | T mixed cannot be mapped to he | **RESOLVED-THIS-SESSION** |
| 4 | dispute | properties liquid unguarded on the mirror createFields | **RESOLVED-THIS-SESSION** |
| 5 | dispute | RAS { turbulence off; } silently run as molecular-viscosity laminar | **RESOLVED-THIS-SESSION** |
| 6 | dispute | atmNutkWallFunction escaping the nut wall-function refusal | **RESOLVED-THIS-SESSION** |
| 7 | dispute | nut wall function other than nutkWallFunction refused on the mirror path | **GENUINE** |
| 8 | dispute | compressible RASModel other than kEpsilon/kOmegaSST refused | **GENUINE** |
| 9 | dispute | wall-function BC on a non-'wall' patch refused | **GENUINE** |
| 10 | dispute | pressureControl pMaxFactor/pMinFactor/rhoMax/rhoMin refusals | **GENUINE** |
| 11 | dispute | laminar model other than Stokes/generalizedNewtonian/Maxwell refused | **GENUINE** |
| 12 | dispute | generalizedNewtonian powerLaw guard tested only nuMax | **RESOLVED-THIS-SESSION** |
| 13 | missed | CUDA createFields coupled-patch refusal (cyclic/cyclicAMI/cyclicACMI/processor) | **GENUINE** |
| 14 | missed | absence of a coupled-patch guard on the host _cpp createFields | **HOLE** |
| 15 | missed | 'found no T boundary entry for patch X' refusal | **GENUINE** |
| 16 | missed | neither constant/momentumTransport nor constant/turbulenceProperties refused | **GENUINE** |
| 17 | missed | RAS case with no alphat field refused in the mirror driver | **GENUINE** |
| 18 | missed | thermo_parse structural refusals (no thermoType, no mixture, molWeight<=0, Cp/Pr<=0, Cv<=0) | **GENUINE** |
| 19 | missed | uniformFixedValue with a non-constant Function1 refused | **GENUINE** |
| 20 | missed | LESModel outside the supported set refused | **GENUINE** |

Evidence, one line each (current file:line -- line numbers are as of 2ecfe0f):

1. src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:464-473 whitelist excludes coupled types; the checker is right that this refusal is load-bearing: fv_patch_field.cuh:1795-1797 builds any isCoupledInterfaceType as a ZeroGradient placeholder, the real CyclicFvPatchField/ProcessorFvPatchField are constructed only by simpleFoam-side helpers (cyclic_field.cuh:37, field_distribute.cuh:41, parallel_simple.cuh:69) and never on the rhoSimpleFoam path, and the coupled-patch refusal at rhoCreateFields.cu:65-75 guards only the CUDA arm -- so the surveyor's proposed whitelist widening would let the _cpp driver run a cyclic case with zeroGradient silently substituted on every coupled face. The message misattributes the reason (worth fixing), but the refusal refuses exactly the gap, not more.
2. rhoCreateFields_cpp.cu:465 now whitelists fixedGradient and :484-489 scales the gradient slots by Cpv = Cp (or Cp-R for the e arm), exactly the gradientEnergy mapping (commit 2ecfe0f); the checker's objection (no gradient conversion existed) was correct at census time and is now moot.
3. rhoCreateFields_cpp.cu:465 whitelists mixed; :476-477 runs heOf over refValueUniformValue/refValues (the no-op the checker cited is fixed), and :484-489 Cpv-scales refGradient, which foam_field_reader.cuh:726-729 stores in the gradient slot with hasGradient=true (commit 2ecfe0f). Both of the checker's specific gaps are closed.
4. rhoCreateFields_cpp.cu:253-259 now throws on f.thermo.model != ThermoModel::perfectGas before any p/T read, naming the liquid path and the legacy binary that carries it (commit 69de8b4); the checker's missing-refusal diagnosis was correct and the guard now exists.
5. rhoCreateFields_cpp.cu:528-529 sets f.turbulent=true with f.turbulenceFrozen, all four fields are still read, :719-743 does the one-shot boundary validate() (frozen nut = Cmu*k^2/eps, measured 0.001265625 on rhoBoxF), and rhoSimpleFoam_cpp.cu:501 skips only correct() -- OpenFOAM's frozen semantics the checker demanded (kEpsilon.C:216). Frozen non-kEpsilon refused at :560-564 and frozen+nutkWallFunction at :613-618 (commit efc42e6).
6. rhoCreateFields_cpp.cu:605-606 now tests both the nut and atmNut prefixes so the throw at :620-628 fires for atmNutkWallFunction, with a gate arm at tests/rho_simple_end_to_end_vs_openfoam.sh:122-129 that mutates a fixture to atmNutkWallFunction and asserts the refusal (commit 95b7b8c). The checker's fix-by-making-it-fire is what landed.
7. kEpsilon_cpp.cu:240,568,572 call nutkWallFunction unconditionally and kEpsilon.cu:548 hardcodes /*nutWall=*/0 into the near-wall production, so the mirror closure has no family dispatch; the refusal at rhoCreateFields_cpp.cu:620-628 is what prevents silent substitution there. The value functions the surveyor called STALE live only behind the legacy/shared driver's setNutWall (turbulence_setup.cuh:399-467), which never runs on the mirror path -- a capability on another path does not make this refusal liftable without porting the dispatch.
8. Only kEpsilon_cpp.cuh:131 and kOmegaSST_cpp.cuh:131 declare a `struct Compressible` arm; SpalartAllmaras_cpp.cuh, realizableKE_cpp.cuh and kOmegaSSTLM_cpp.cuh contain no compressible instantiation (grep for Compressible: zero hits), and PORT.md forbids substituting the incompressible arm. The refusal at rhoCreateFields_cpp.cu:552-556 refuses exactly what does not exist; the surveyor's STALE rested on the legacy driver and on incompressible host references, neither of which is the missing compressible lineage.
9. The checker's factual correction is confirmed: OF's nutWallFunctionFvPatchScalarField.C:45-56 checkType() does abort(FatalError) on !isA<wallFvPatch>, called from every constructor (:95,111,126,139,153), so the nut half of brae's guard (turbulence_setup.cuh:374-388) transcribes OpenFOAM's own fatal and the surveyor's 'nothing consults polyPatch::type(), OpenFOAM RUNS these cases' is wrong for the nut family. epsilon/omegaWallFunctionFvPatchScalarField.C carry no wallFvPatch check (grep: zero hits), so that half is brae-specific -- and still needed, since setNutWall (:404) and the wall kernels gate on the geometric type and would leave the BC silently inert. Verdict unchanged, rationale corrected.
10. All six conditions are FatalIOErrorInFunction ... exit(FatalIOError) in OF pressureControl.C (~:104-195, verified: pMaxFactor, rhoMax x2, pMinFactor, rhoMin x2), and brae's throws at rhoCreateFields_cpp.cu:145-150, :158-167, :181-185, :191-200 transcribe them in the same order. Checker and surveyor agree the refusals stand; there is no feature behind them to port, so the classification is unchanged.
11. The checker is right on the facts: /usr/lib/openfoam/openfoam2412/src/TurbulenceModels/turbulenceModels/laminar/ contains exactly Stokes, generalizedNewtonian, Maxwell (plus the base laminarModel), and makeLaminarModel registers exactly those three at turbulentTransportModels.C:57-63 and turbulentFluidThermoModels.C:59-65 -- so OF v2412 would itself fatal on lambdaThixotropic/Giesekus/PTT (those are the openfoam.org fork's models, not this semantic authority's). brae's throw at turbulence_setup.cuh:100-106 is a faithful mirror with nothing to port.
12. turbulence_setup.cuh:65-71 now refuses unless all three of n, nuMin, nuMax are present, naming the missing ones, matching OF powerLaw.C's no-default dimensionedScalar constructs (commit 3121905); the checker's under-broad diagnosis (silent n=1 Newtonian default) was correct and is closed.
13. rhoCreateFields.cu:65-75 refuses any coupled FvPatch by name before building the device projection, because buildDeviceMesh keeps those faces out of the LDU; it is the only dedicated coupled-patch guard the solver has and refuses exactly the gap.
14. No cyclic/processor guard exists anywhere in rhoSimpleFoam_cpp.cu or rhoCreateFields_cpp.cu (grep: only comments), and makePatchField at fv_patch_field.cuh:1795-1797 silently builds every coupled type as a ZeroGradient placeholder; the only thing refusing a cyclic case on the host arm today is the T->he whitelist at rhoCreateFields_cpp.cu:464-473 firing by accident of T carrying a `cyclic` entry. A latent silent-substitution hole -- masked for rhoSimpleFoam because T is MUST_READ, and the reason the OVER-BROAD reclassification of the whitelist had to be rejected.
15. rhoCreateFields_cpp.cu:456-460 throws after name/group/regex resolution via findPatchEntry; OpenFOAM would equally fail to construct the field, so refusing rather than inventing an entry is correct.
16. rhoCreateFields_cpp.cu:508-517 refuses when neither file exists rather than assuming laminar, mirroring rhoSimpleFoam constructing the turbulence model unconditionally in createFields.H.
17. rhoSimpleFoam_cpp.cu:551-556 (kOmegaSST arm) and :604-608 (kEpsilon arm) both throw when f.alphat is empty, naming EddyDiffusivity's construction-time read and alphaEff = CpByCpv*(alpha + alphat); load-bearing for the energy equation as claimed.
18. All five present: thermo_parse.cuh:83-87 (no thermoType dict), :166-170 (no mixture dict), :175-178 (molWeight must be positive), :211-214 (Cp and Pr must be positive), :221-226 (Cv = Cp - R must be positive, gas path only -- the real physical guard, correctly skipped on the liquid parse).
19. fv_patch_field.cuh:1553-1560 refuses uniformFixedValue whose uniformValue is a Function1 brae cannot evaluate, while :1572-1574 builds the constant subset as fixedValue -- exactly the premise that makes the surveyor's 'constant uniformFixedValue is exactly fixedValue' claim exact, so the whitelist entry would inherit this refusal rather than widen it.
20. turbulence_setup.cuh:122-124 throws on any LESModel outside Smagorinsky/WALE/SpalartAllmarasDDES-DES-IDDES/kOmegaSSTDDES-DES-IDDES, reachable from the incompressible gpuSimpleFoam/gpuPimpleFoam drivers and distinct from the mirror path's simulationType refusal at rhoCreateFields_cpp.cu:673-674.

### Original survey entries closed by this session

- T fixedGradient -> he (was STALE): lifted by commit 2ecfe0f -- whitelisted at rhoCreateFields_cpp.cu:465 with the Cpv-scaled gradient mapping at :484-489, exactly the gradientEnergy mapping the entry asked for
- T mixed -> he (was STALE): lifted by the same commit 2ecfe0f -- whitelisted at :465, refValue through heOf at :476-477, refGradient Cpv-scaled at :484-489, valueFraction carried across
- properties liquid unguarded on the mirror createFields (was DEAD, i.e. missing refusal): guard added at rhoCreateFields_cpp.cu:253-259 by commit 69de8b4, refusing before any perfect-gas arithmetic runs on a liquid
- atmNutkWallFunction escaping the nut refusal (was DEAD): closed by commit 95b7b8c -- two-prefix test at rhoCreateFields_cpp.cu:605-606 makes the throw at :620-628 fire, gated at tests/rho_simple_end_to_end_vs_openfoam.sh:122-129
- RAS { turbulence off; } silently run as laminar (was DEAD, i.e. missing refusal): commit efc42e6 implements OpenFOAM's frozen-model semantics (rhoCreateFields_cpp.cu:528-529, one-shot validate at :719-743, driver skip at rhoSimpleFoam_cpp.cu:501) with named refusals for frozen non-kEpsilon (:560-564) and frozen+nutkWallFunction (:613-618)
- generalizedNewtonian powerLaw silently defaulting n and nuMin (was DEAD): commit 3121905 requires all three of n, nuMin, nuMax by name at turbulence_setup.cuh:65-71, as OF powerLaw.C does

### The original survey for this area (as surveyed; see the verdicts above for corrections)

<details><summary><b>[STALE]</b> T patch type `fixedGradient` cannot be mapped to he — a prescribed heat-flux wall is refused wholesale.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422 (whitelist at :419-421)
- **trigger**: any entry in 0/T with `type fixedGradient; gradient uniform &lt;g>;` — e.g. validation/rhoBoxQ's hotWall.
- **of_feature**: basicThermo::heBoundaryTypes maps fixedGradient-derived T -> gradientEnergyFvPatchScalarField (basicThermo.C:210-216); the BC is gradientEnergyFvPatchScalarField.C:111-116.
- **path**: host
- **size**: ~10 lines. The device arm is already built.
- **fixture**: YES — validation/rhoBoxQ plus validation/hf_vs_openfoam.sh, which measures T 4.94e-08 converged against real OF and carries the discriminating control (gradient copied unscaled reads 7.97e-03). Today that gate points at $BUILD/brae_rhoSimpleFoam, i.e. gpuRhoSimpleFoam.cu; it needs a second arm on the mirror binary.
- **blocks_tutorial**: None of the six compressible/rhoSimpleFoam tutorials, but this is the only way to write a heat-flux wall, and the shipped brae_rhoSimpleFoam already supports it (device_energy.cu:31-33).
- **depends_on / middle steps**:
  - Add "fixedGradient" to the whitelist at rhoCreateFields_cpp.cu:419-421.
  - Scale the gradient by Cpv, NOT by heOf(): heOf is affine (Cp*(T-Tref)+Href), and applying its offset to a slope is wrong. b.gradientUniformValue and b.gradientValues are the only fields the mapping block at :428-433 never touches today.
  - Nothing else: FixedGradientPatchField already exists (fv_patch_field.cuh:1448-1453 and it already accepts the type name `gradientEnergy`), the host laplacian already consumes gradientBoundaryCoeffs generically (fvm.cuh:96, fvm.cuh:257), and DeviceBoundary already carries refGrad (device_boundary.cuh:38,64,97) so buildDeviceBoundary in rhoCreateFields.cu:86 needs no change.
  - CORRECTION to the in-file justification: rhoCreateFields_cpp.cu:381-388 says zeroGradient is exact only because he is p-independent, so that a janaf or liquid thermo would break it. That misreads OpenFOAM. gradientEnergy's second term is deltaCoeffs*(he(pw,Tw,patchi) - he(pw,Tw,faceCells)) — the SAME pw and Tw on both sides, differing only in which mixture object is used (heThermo.C:264-296). pureMixture returns the one mixture_ for both cellMixture and patchFaceMixture (pureMixture.H:89-101), so the term is identically zero for ANY thermo, p-dependent or not. The mapping grad(he) = Cpv(pw,Tw)*snGrad(T) is therefore exact for janaf too — it just needs Cpv evaluated at Tw instead of a constant.

</details>
<details><summary><b>[STALE]</b> T patch type `mixed` (Robin) cannot be mapped to he — an external-convection wall is refused.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422
- **trigger**: 0/T carrying `type mixed; refValue ...; refGradient ...; valueFraction ...;` — validation/rhoBoxM's hotWall.
- **of_feature**: mixedFvPatchScalarField-derived T -> mixedEnergyFvPatchScalarField (basicThermo.C:217-220); the BC is mixedEnergyFvPatchScalarField.C:113-127.
- **path**: host
- **size**: ~12 lines.
- **fixture**: YES — validation/rhoBoxM plus validation/mx_vs_openfoam.sh (converged T 8.75e-08, p 5.20e-09, U 2.60e-06 vs OF; control: dropping the (1-vf) weight reads boundaryCoeffs 9.81e-02). Again pointed at the legacy binary only.
- **blocks_tutorial**: None of the six, but it is the standard external-convection wall.
- **depends_on / middle steps**:
  - Add "mixed" to the whitelist. The mapping block already does two of the three things OF does: heOf on refValues (:431) gives refValue = he(pw, Tw.refValue()), and valueFraction is carried across untouched, which is exactly `valueFraction() = Tw.valueFraction()`.
  - The THIRD is missing and is the trap: refGrad() = Cpv*Tw.refGrad(). The block never touches gradientValues, so adding `mixed` to the whitelist ALONE would impose T's refGradient as he's — wrong by Cpv (~718 J/kg/K for air on the `e` arm), the same order of error hf_vs_openfoam measured as 7.97e-03.
  - MixedPatchField with setValueFraction + setRefGrad already exists (fv_patch_field.cuh:1623-1647, and it already accepts the type name `mixedEnergy`); device_boundary_assembly.cu:41 already places refGrad inside the (1-vf) lerp, which is the weighting the fixedGradient path never exercises.

</details>
<details><summary><b>[OVER-BROAD]</b> T patch type `outletInlet` cannot be mapped to he, although it is the already-accepted `inletOutlet` with the flux test inverted.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422
- **trigger**: 0/T with `type outletInlet; outletValue ...;`
- **of_feature**: outletInletFvPatchField.H:90 — `public mixedFvPatchField&lt;Type>` — so basicThermo.C:217-220 maps it to mixedEnergy, exactly as it maps inletOutlet.
- **path**: host
- **size**: 1 line.
- **fixture**: No compressible one. A rhoBox variant with `outletInlet` on its outlet T would gate it against OF; the discriminating control is a reversed-flow face, where outletInlet and inletOutlet disagree.
- **blocks_tutorial**: None (pimpleFoam/LES/NACA4412 uses it on other fields).
- **depends_on / middle steps**:
  - One word in the whitelist. The mapping is bit-for-bit the inletOutlet case brae already accepts: refGrad is 0 so Cpv*0 = 0, refValue = he(outletValue) which heOf on inletValues at :432-433 already produces (fv_patch_field.cuh:1580-1585 reads outletValue into the same slot), and valueFraction is recomputed from phi by OutletInletPatchField.
  - OutletInletPatchField already exists and is already constructed by makePatchField (fv_patch_field.cuh:1583).

</details>
<details><summary><b>[OVER-BROAD]</b> T patch type `freestream` cannot be mapped to he — the natural far-field temperature for external aero.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422
- **trigger**: 0/T with `type freestream; freestreamValue uniform 300;`
- **of_feature**: freestreamFvPatchField.H:103 — `public inletOutletFvPatchField&lt;Type>` -> mixedFvPatchField -> mixedEnergy (basicThermo.C:217-220).
- **path**: host
- **size**: 1 line.
- **fixture**: validation/naca0012 is the natural home (its p is already freestreamPressure); its T is currently inletOutlet, so the fixture would have to be varied rather than reused as shipped.
- **blocks_tutorial**: aerofoilNACA0012 runs today because its T happens to be inletOutlet, not freestream.
- **depends_on / middle steps**:
  - One word in the whitelist. `freestream` on a scalar is a binary-flux inletOutlet in OF, and makePatchField already builds it that way (fv_patch_field.cuh:1601-1602, reading the inletValues slot that heOf already transforms at :432).
  - Note freestreamVelocity/freestreamPressure are the CONTINUOUS-blend variants and are vector/pressure BCs, not T ones — no work needed there.

</details>
<details><summary><b>[OVER-BROAD]</b> T patch type `uniformFixedValue` cannot be mapped to he, although a constant Function1 is exactly the accepted `fixedValue` case.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422
- **trigger**: 0/T with `type uniformFixedValue; uniformValue ...;` — squareBendLiq/0.orig/T writes its walls this way.
- **of_feature**: uniformFixedValueFvPatchField.H:87 — `public fixedValueFvPatchField&lt;Type>` -> fixedEnergyFvPatchScalarField (basicThermo.C:205-208, fixedEnergyFvPatchScalarField.C:95-111).
- **path**: host
- **size**: 1 line for the constant subset; medium (a refresh hook in the SIMPLE loop, host + device) for the time-varying one.
- **fixture**: squareBendLiq exercises the hard half, but it is the liquid path and is blocked earlier. rhoBox with `uniformFixedValue { uniformValue constant 700; }` on hotWall gates the easy half against the existing rhoBox OF run.
- **blocks_tutorial**: squareBendLiq (also blocked by the liquid path).
- **depends_on / middle steps**:
  - For a CONSTANT Function1 the mapping is exact and free: T_b never moves in a steady solve, so fixedEnergy's per-updateCoeffs re-evaluation of he(pw,Tw) is a no-op and brae's frozen fixedValue he is the same number. makePatchField already builds uniformFixedValue as fixedValue and already refuses an unsupported Function1 by name (fv_patch_field.cuh:1488-1500), so the whitelist entry inherits that refusal rather than widening it.
  - For a TIME- or EXPRESSION-driven Function1 it is NOT exact, and the missing piece is a per-outer-iteration he_b = he(p_b, T_b) refresh. The _cpp loop has none: rhoSimpleFoam_cpp.cu:188 refreshes only the inletOutlet valueFraction (updateFromFlux), and he's refValue is written once in createFields. deviceEnergyBoundaryFromT (device_energy.cu:72-110) is the legacy path's version of exactly that hook.

</details>
<details><summary><b>[GENUINE]</b> T patch type `totalTemperature` cannot be mapped to he.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422
- **trigger**: 0/T with `type totalTemperature; T0 uniform 300;` — the standard compressible stagnation inlet.
- **of_feature**: totalTemperatureFvPatchScalarField.H:77-80 — `public fixedValueFvPatchScalarField` -> fixedEnergy (basicThermo.C:205-208). T_b = T0/(1 + 0.5*(gamma-1)*M^2) is recomputed from p and U every updateCoeffs, and fixedEnergy then re-derives he from that MOVING T_b (fixedEnergyFvPatchScalarField.C:104-108).
- **path**: both
- **size**: medium — ~150 lines plus a gate.
- **fixture**: None. Needs a new subsonic nozzle/plenum case; OF ships totalTemperature in the rhoPimpleFoam and sonicFoam trees, none under rhoSimpleFoam.
- **blocks_tutorial**: None of the six.
- **depends_on / middle steps**:
  - A totalTemperature patch field on T itself in makePatchField — needs |U| at the patch, the flux sign, and gamma = Cp/Cv from the thermo at the face.
  - A per-outer-iteration he_b refresh in rhoSimpleFoam_cpp's loop. This is the real blocker and it is shared with items 5 and 7: today he's boundary refValue is frozen at createFields time (only rhoSimpleFoam_cpp.cu:188's updateFromFlux runs per iteration). Without it a moving T_b would silently leave he_b at its cold-start value — the same stale-he_b class of defect already recorded on the liquid boundary path.
  - The same refresh on the device arm; deviceEnergyBoundaryFromT (device_energy.cu:72) already exists for the legacy driver and would be the model.

</details>
<details><summary><b>[GENUINE]</b> T patch type `externalWallHeatFluxTemperature` cannot be mapped to he.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422 (and independently src/finiteVolume/fields/fv_patch_field.cuh:1773-1786)
- **trigger**: 0/T with `type externalWallHeatFluxTemperature; mode flux|power|coefficient;`
- **of_feature**: mixedFvPatchScalarField-derived -> mixedEnergy (basicThermo.C:217-220). Its gradient is q/kappaEff.
- **path**: both
- **size**: medium.
- **fixture**: None.
- **blocks_tutorial**: None of the six; common in buoyant/CHT cases.
- **depends_on / middle steps**:
  - The mixedEnergy mapping of item 2 first — this BC is a `mixed` T whose three entries are recomputed each iteration.
  - A per-iteration BC update that can read alphaEff at the patch: kappaEff = Cp*(alpha + alphat), and alphat at a heated turbulent wall grows by orders of magnitude during the run, so a frozen gradient imposes a flux the case never asked for. That is exactly why fv_patch_field.cuh:1773 refuses it even though FixedGradientPatchField could express it.
  - So: mixedEnergy mapping -> the he_b refresh hook (shared with items 5/6) -> a patch-level kappaEff accessor from the closure.

</details>
<details><summary><b>[GENUINE]</b> T patch types `compressible::turbulentTemperatureCoupledBaffleMixed` / `turbulentTemperatureRadCoupledMixed` cannot be mapped to he.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422
- **trigger**: any conjugate-heat-transfer case, i.e. a multi-region T patch.
- **of_feature**: mixed-derived -> mixedEnergy (basicThermo.C:217-220), but the refValue comes from a NEIGHBOUR REGION through mappedPatchBase.
- **path**: both
- **size**: large.
- **fixture**: None; would be a chtMultiRegionFoam-shaped case, which is a different solver.
- **blocks_tutorial**: None of the six.
- **depends_on / middle steps**:
  - Multi-region support in the mirror tree: a region registry, mapped-patch addressing, and an outer coupling loop. None of that exists — the mirror driver builds one PrimitiveMesh.
  - Then the mixedEnergy mapping of item 2.
  - This is the endpoint, not a next step; items 1-2 and the he_b refresh hook are the prerequisites that pay for themselves independently.

</details>
<details><summary><b>[GENUINE]</b> T patch types `fixedJump` / `fixedJumpAMI` cannot be mapped to he.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422
- **trigger**: a cyclic pair carrying a prescribed temperature jump.
- **of_feature**: basicThermo::heBoundaryBaseTypes (basicThermo.C:164-192) gives he an energyJump/energyJumpAMI BASE type, and heBoundaryTypes (basicThermo.C:221-228) the matching energyJump type — the only place the two-argument GeometricField constructor's base-type list is used.
- **path**: both
- **size**: large.
- **fixture**: None compressible.
- **blocks_tutorial**: None of the six.
- **depends_on / middle steps**:
  - Coupled patches in the mirror tree first. rhoCreateFields.cu:64-75 refuses cyclic/cyclicAMI/cyclicACMI/processor outright on the CUDA arm because buildDeviceMesh keeps those faces out of the LDU.
  - Then the jump BC itself, on both halves of the pair.

</details>
<details><summary><b>[OVER-BROAD]</b> A `cyclic` / `cyclicAMI` / `processor` T patch is refused with a T->he message, when the he mapping is not what is missing.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:422 (whitelist at :419-421)
- **trigger**: any compressible case on a cyclic or decomposed mesh.
- **of_feature**: basicThermo.C:201 `wordList hbt(tbf.types())` — the DEFAULT is T's own type name, and a constraint patch matches none of the five isA&lt;> tests, so cyclic T -> cyclic he unchanged. The mapping for constraint patches is the identity, which is why brae already whitelists empty/symmetry/symmetryPlane/wedge/slip/calculated.
- **path**: host
- **size**: 3 lines for the diagnostic; large for the feature.
- **fixture**: None compressible carries a cyclic patch. validation/cyclicChannel is incompressible.
- **blocks_tutorial**: None of the six (all are single-region non-cyclic).
- **depends_on / middle steps**:
  - Add the coupled types to the whitelist so the case reaches its REAL refusal at rhoCreateFields.cu:64-75 ("buildDeviceMesh keeps those faces out of the LDU"), which names the actual missing feature. Today the diagnostic points the user at the energy boundary condition, which is correct in OpenFOAM and needs no work.
  - makePatchField already builds coupled types as a ZeroGradient placeholder (fv_patch_field.cuh:1691, isCoupledInterfaceType), so the whitelist change cannot silently produce a wrong he — it just moves the refusal to the honest one.
  - Lifting the underlying limitation means coupled faces in the mirror DeviceMesh: a large, separate job.

</details>
<details><summary><b>[DEAD]</b> The energy-variable check in createFields ('sensibleEnthalpy -> h, sensibleInternalEnergy -> e') can never fire.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:253
- **trigger**: nothing. readThermoCoeffs is called eleven lines earlier at rhoCreateFields_cpp.cu:242, and thermo_parse.cuh:150-154 already refuses every `energy` word except sensibleEnthalpy and sensibleInternalEnergy on the gas path, and thermo_parse.cuh:123-125 requires sensibleInternalEnergy on the liquid path. A missing thermoType dies at thermo_parse.cuh:85. There is no reachable input that survives readThermoCoeffs and then fails this block.
- **of_feature**: basicThermo::validate(app, "h", "e") — basicThermo.C:504-525 — which rhoSimpleFoam.C calls and which compares he().name() against "h"/"e", fatalling for absoluteEnthalpy ("ha") and absoluteInternalEnergy ("ea").
- **path**: host
- **size**: n/a
- **fixture**: tests/test_rho_create_fields_cpp.cu:164-176 asserts "an unsupported thermo energy is refused" and that the message names the energy found — but the message it catches comes from thermo_parse.cuh:38, not from this throw. The gate passes either way, which is why the dead branch survived.
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - Nothing to implement. The block still earns its keep for its OTHER job — it is what assigns f.heName, which rhoEEqn's kinetic-energy source branches on (Ekp = 0.5|U|^2 + p/rho for `e` vs K = 0.5|U|^2 for `h`).
  - If it is kept as a refusal, note it is a FAITHFUL MIRROR: OpenFOAM refuses the absolute energy forms for this solver too, so lifting it would diverge from OpenFOAM rather than widen brae.

</details>
<details><summary><b>[DEAD]</b> `properties liquid` is ACCEPTED by the thermo parser but the OF-mirror createFields has no perfectGas guard — it then runs the perfect-gas equation of state on a liquid.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:285 and :296 (perfectGasRho), :359 (perfectGasPsi), :392 (f.thermo.Cp)
- **trigger**: any case with `properties liquid;` and the H2O mixture — squareBendLiq and squareBendLiqNoNewtonian. thermo_parse.cuh:118-139 returns ThermoModel::liquidH2O successfully, and nothing downstream on the _cpp path checks c.model.
- **of_feature**: heRhoThermo&lt;rhoThermo, pureMixture&lt;species::thermo&lt;thermophysicalPropertiesSelector&lt;liquidProperties>, sensibleInternalEnergy>>> (basic/rhoThermo/liquidThermo.H). rho, Cp, mu and kappa are per-cell NSRDS correlations, not p/(R T) and a scalar Cp.
- **path**: host
- **size**: ~8 lines for the guard; large for actually porting the liquid path into the mirror.
- **fixture**: validation/squareBendLiq-shaped case run through the _cpp path; today only the legacy binary is pointed at liquids (gpuRhoSimpleFoam.cu:176,485).
- **blocks_tutorial**: squareBendLiq and squareBendLiqNoNewtonian are the tutorials this silently mis-runs.
- **depends_on / middle steps**:
  - This is a MISSING refusal, not an over-broad one. The CUDA arm has the guard — requirePerfectGas at rhoThermoDevice.cu:22-34 — and its comment says explicitly that it lives in one place because a partial refusal is the failure this project keeps naming. The host _cpp createFields, which runs FIRST, has no equivalent.
  - thermo_parse.cuh:136-137 states that on the liquid path the scalar Cp/mu/kappa/rho members stay at their DEFAULTS and are not read, so heOf at rhoCreateFields_cpp.cu:392 would build he from an uninitialised gas Cp. device_energy.cu:38-42 records what that costs: squareBendLiq's 350 K walls carried he = -48361 J/kg where Es(1e5,350) is -15641742 J/kg, and 406 cells reached an energy no liquid can attain.
  - Fix: one requirePerfectGas-style call at the top of createFields, before the p/T read. ~8 lines.

</details>
<details><summary><b>[GENUINE]</b> `properties liquid` with any mixture other than H2O is refused.</summary>

- **citation**: src/thermophysicalModels/thermo_parse.cuh:129-132
- **trigger**: `thermoType { properties liquid; }` with a `mixture { &lt;substance> { ... } }` block naming anything but H2O.
- **of_feature**: Foam::liquidProperties and its ~40 registered substances (thermophysicalModels/thermophysicalProperties/liquidProperties); the correlation FORMS are NSRDS functions 0-6.
- **path**: both
- **size**: small per substance (~40 lines of coefficients) once the mirror liquid path exists at all; the mirror liquid path is the large piece.
- **fixture**: None beyond H2O.
- **blocks_tutorial**: none of the six (both liquid tutorials are H2O)
- **depends_on / middle steps**:
  - nsrds_functions.cuh already carries the general correlation forms; what each substance needs is its own validated coefficient set (rho, pv, hl, Cp, h, Cpg, mu, mug, kappa, sigma) plus a gate against OF for that substance.
  - Also blocked on item 12: even H2O is unguarded on the mirror path today, so a second substance would inherit that gap.

</details>
<details><summary><b>[OVER-BROAD]</b> `properties liquid` with `energy sensibleEnthalpy` is refused, although OpenFOAM registers the variant and the arithmetic exists in brae.</summary>

- **citation**: src/thermophysicalModels/thermo_parse.cuh:123-125
- **trigger**: a liquid case whose thermoType says `energy sensibleEnthalpy;` instead of sensibleInternalEnergy.
- **of_feature**: liquidThermo.H registers both the sensibleEnthalpy and sensibleInternalEnergy instantiations of heRhoThermo&lt;rhoThermo, pureMixture&lt;...>>.
- **path**: both
- **size**: gate only, if the internal-energy path is otherwise sound.
- **fixture**: squareBendLiq with `energy sensibleEnthalpy;` and a matching OF run.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - The arithmetic is already parameterised: h2oEnergy and h2oCpv both take an EnergyForm (device_energy.cu:50-70), and the enthalpy branch is the SIMPLER one — it drops the -p/rho term and its boundary-pressure dependency, which is the thing device_energy.cu:88-94 currently refuses to run without.
  - What is genuinely absent is a validated end-to-end run, which is what the refusal message says. Lifting it means one gate, not new code.

</details>
<details><summary><b>[GENUINE]</b> `thermo` other than hConst (janaf, eConst, hPolynomial, hRefConst, ePolynomial, hTabulated) is refused.</summary>

- **citation**: src/thermophysicalModels/thermo_parse.cuh:148 (throw at :38)
- **trigger**: `thermoType { thermo janaf; }` and every non-hConst thermo word.
- **of_feature**: Foam::janafThermo / eConstThermo / hPolynomialThermo (thermophysicalModels/specie/thermo/). janaf's Cp is a 7-coefficient polynomial in T with a low/high branch at Tcommon.
- **path**: both
- **size**: large.
- **fixture**: None. OF ships janaf in the reacting/combustion trees; no rhoSimpleFoam tutorial uses it.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - A T-dependent Cp/he/Cv on both host and device — today thermo_model.cuh's thermoCv is a single scalar and equation_of_state.cuh's perfectGasPsi/Rho take only R.
  - A he -> T inversion. hConst inverts in closed form; janaf needs Newton. device_thermo.cu already carries a Newton he->T for the liquid path, so the shape exists.
  - Cpv AT THE PATCH FACE for the gradientEnergy/mixedEnergy mapping — see item 1: the mapping grad(he) = Cpv(pw,Tw)*snGrad(T) stays EXACT for janaf, contrary to the claim at rhoCreateFields_cpp.cu:387; only the constant-Cpv shortcut breaks.
  - CpByCpv, which the transonic pEqn and the energy equation both read.

</details>
<details><summary><b>[GENUINE]</b> `equationOfState` other than perfectGas (rhoConst, perfectFluid, incompressiblePerfectGas, Boussinesq, PengRobinsonGas, icoPolynomial, rPolynomial, adiabaticPerfectFluid) is refused.</summary>

- **citation**: src/thermophysicalModels/thermo_parse.cuh:149 (throw at :38); the companion `type` check at :142-146 fires first for heRhoThermo, with a message about the TYPE rather than the eos
- **trigger**: any thermoType whose equationOfState is not perfectGas.
- **of_feature**: thermophysicalModels/specie/equationOfState/* — each supplies rho(p,T), psi(p,T), CpMCv(p,T) and Z.
- **path**: both
- **size**: medium per model, plus the dispatch.
- **fixture**: None compressible.
- **blocks_tutorial**: none of the six (all six are perfectGas or liquid)
- **depends_on / middle steps**:
  - A dispatch in equation_of_state.cuh (host) and device_thermo.cu (device) — today both are single free functions taking ThermoCoeffs, so the dispatch itself is the work, not the individual models (~30 lines each once it exists).
  - CpMCv per model: thermo_parse.cuh:216-226 hardcodes Cv = Cp - R, which is perfectGas::CpMCv only.
  - psiBnd, which the transonic pEqn branch interpolates to faces and which the manifest records as gated against NOTHING today (createFieldRefs validation note) — so a second eos would land on an unmeasured surface.
  - heRhoThermo vs hePsiThermo stops being a no-op the moment eos != perfectGas: rho is the STORED field lagging the pressure solve, which is what thermo_parse.cuh:97-106 already documents and c.rhoThermoType already records.

</details>
<details><summary><b>[GENUINE]</b> `mixture` other than pureMixture (multiComponentMixture, reactingMixture, egrMixture) is refused.</summary>

- **citation**: src/thermophysicalModels/thermo_parse.cuh:147 (throw at :38)
- **trigger**: `thermoType { mixture multiComponentMixture; }`
- **of_feature**: thermophysicalModels/reactionThermo/mixtures/*. cellMixture(celli) and patchFaceMixture(patchi,facei) stop returning the same object.
- **path**: both
- **size**: large.
- **fixture**: None.
- **blocks_tutorial**: none of the six (gasMixing/injectorPipe transports a passive tracer, not species — it is still pureMixture)
- **depends_on / middle steps**:
  - Species transport — a Y field per specie, its own equations, and a composition-weighted thermo.
  - AND it invalidates the exactness argument every T->he mapping above rests on. gradientEnergy's second term deltaCoeffs*(he(pw,Tw,patchi) - he(pw,Tw,faceCells)) is zero ONLY because pureMixture.H:89-101 returns the one mixture_ for both. For a multi-component mixture that term is the composition jump between the wall face and the adjacent cell and is nonzero. Every mapping in items 1-9 would have to be re-derived, and the second term actually implemented.

</details>
<details><summary><b>[GENUINE]</b> `transport` other than sutherland or const (polynomial, logPolynomial, icoTabulated, tabulated, WLF, Andrade) is refused.</summary>

- **citation**: src/thermophysicalModels/thermo_parse.cuh:157-161 (throw at :38)
- **trigger**: `thermoType { transport polynomial; }` with `transport { muCoeffs&lt;8> (...); kappaCoeffs&lt;8> (...); }`
- **of_feature**: thermophysicalModels/specie/transport/* — mu(p,T) and kappa(p,T).
- **path**: both
- **size**: small per model (~30 lines) once ThermoCoeffs carries arrays.
- **fixture**: validation/rhoSuth gates sutherland; nothing gates a polynomial.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - transport_model.cuh on the host and device_thermo.cu on the device (including the boundary-mu kernel at device_thermo.cu:473, which is a separate call site).
  - ThermoCoeffs is a flat scalar struct (As/Ts/mu0/Pr), so a polynomial needs a coefficient array on it — that is the structural change, and it is shared with items 15 and 16.
  - Nothing else: the consumers all go through transportMu/alphahe.

</details>
<details><summary><b>[STALE]</b> A compressible RASModel other than kEpsilon or kOmegaSST is refused on the OF-mirror path.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:491 (and again in the driver at rhoSimpleFoam_cpp.cu:587)
- **trigger**: `RAS { RASModel SpalartAllmaras; }` (or realizableKE, RNGkEpsilon, kOmegaSSTLM) with `turbulence on`.
- **of_feature**: compressible::RASModel::New — the templated models under src/TurbulenceModels/turbulenceModels/RAS/, instantiated with alpha=1, rho as a field and alphaRhoPhi as the MASS flux.
- **path**: both
- **size**: medium per model.
- **fixture**: YES for the refusal — tests/rho_simple_end_to_end_vs_openfoam.sh:103-106 mutates sbMatched's RASModel to LaunderSharmaKE and asserts the refusal names it, with the unmutated case as the negative control. A POSITIVE gate per model needs its own OF run.
- **blocks_tutorial**: None of the six (all four turbulent tutorials are kEpsilon or kOmegaSST).
- **depends_on / middle steps**:
  - STALE because the shipped brae_rhoSimpleFoam (gpuRhoSimpleFoam.cu:32,334) reads all of them through turbulence_setup.cuh:240-241, and the mirror tree already has host references for three of the four: SpalartAllmaras_cpp.cu, realizableKE_cpp.cu and kOmegaSSTLM_cpp.cu under src/TurbulenceModels/turbulenceModels/RAS/.
  - What is missing is the COMPRESSIBLE instantiation of each, and PORT.md forbids copying the incompressible one. The specific difference the manifest records for kEpsilon is TWO FLUXES: fvm::div takes the mass flux while divU takes the VOLUMETRIC one, because compressibleTurbulenceModel::phi() divides by interpolate(rho). Getting that wrong reads 5.0e-06 against 5.0e-15 on the existing gate.
  - Plus per model: the extra field reads in createFields (nuTilda for SA; ReThetat and gammaInt for LM), the EddyDiffusivity alphat = rho*nut/Prt hook (rhoCreateFields_cpp.cu:618-624 currently runs it only for kEpsilon), and the construction-time correctNut, which rhoCreateFields_cpp.cu:602-604 also gates on rasModel == "kEpsilon" only — so even kOmegaSST, which IS accepted, skips validate()'s correctNut. That is a live gap inside the accepted set.
  - RNGkEpsilon and realizableKE are the cheapest: same fields, same wall functions, different coefficients and one extra source term each.

</details>
<details><summary><b>[STALE]</b> A nut wall function other than nutkWallFunction is refused on the OF-mirror path.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:533
- **trigger**: any 0/nut boundary entry whose type starts with "nut" and is not nutkWallFunction — nutUSpaldingWallFunction, nutUWallFunction, nutLowReWallFunction, nutUBlendedWallFunction.
- **of_feature**: nutWallFunctionFvPatchScalarField.C:181-184 — `operator==(calcNut())`, a virtual call PER PATCH; each family member is a different function of different inputs (nutkWallFunctionFvPatchScalarField.C:71 never reads U; nutLowReWallFunctionFvPatchScalarField.C:38-42 returns Zero unconditionally).
- **path**: both
- **size**: medium — ~200 lines plus gates.
- **fixture**: The REFUSAL is gated: tests/rho_simple_end_to_end_vs_openfoam.sh:113-118 rewrites sbMatched's nutkWallFunction to nutUSpaldingWallFunction and asserts the refusal. A POSITIVE gate needs an OF run with the substituted BC — validation/sbMatched is the natural host.
- **blocks_tutorial**: gasMixing/injectorPipe, whose walls carry nutUWallFunction.
- **depends_on / middle steps**:
  - STALE because the VALUE functions already live in the mirror tree: nut_wall_function.cuh:45 (nutUSpalding, Newton on Spalding's law), :66-82 (nutUWallValue, STEPWISE blender), :100 (nutUBlended), :130-165 (the `nutWall` selector, with 4 = nutLowRe returning exactly zero and atmZ0 > 0 selecting the rough atm branch). All BRAE_HD, host and device.
  - What is missing is the DISPATCH. kEpsilon_cpp.cu:240,568,572 and kEpsilon.cu:379 call nutkWallFunction unconditionally, and kEpsilon.cu:548 hardcodes `/*nutWall=*/0` into deviceWallEpsG0 — so the near-wall PRODUCTION term would still be nutk's even if the wall nut were fixed. Both call sites must move together.
  - The selector must be PER PATCH, not case-wide: OF dispatches per patch object, and turbulence_setup.cuh:419-425 refuses two different ones on the shipped driver for exactly that reason. So RhoSimpleFields needs a per-patch nutWall array, populated at rhoCreateFields_cpp.cu:528 where the dictionary type is still in hand (as its own comment at :524-526 says).
  - The nutU family needs |U| at the patch, which the closure does not currently pass to its wall kernels.

</details>
<details><summary><b>[DEAD]</b> atmNutkWallFunction ESCAPES the nut wall-function refusal that names it, and is then run as the smooth nutkWallFunction.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:531
- **trigger**: 0/nut with `type atmNutkWallFunction; z0 uniform 0.1;`. The guard is `if (b.type.rfind("nut", 0) != 0) continue;` — a PREFIX test — and "atmNutkWallFunction" does not start with "nut", so the loop skips it and the throw at :533 is unreachable for this type.
- **of_feature**: atmNutkWallFunctionFvPatchScalarField (src/atmosphericModels/derivedFvPatchFields/wallFunctions/atmNutkWallFunction/) — the rough-wall nut, a function of the roughness length z0, which OpenFOAM carries as a per-face time-varying PatchFunction1.
- **path**: host
- **size**: ~3 lines to close the hole.
- **fixture**: None compressible carries atmNutkWallFunction. Adding it to sbMatched's walls and asserting the refusal is the cheap gate, with the unmutated case as the control — the same shape as the nutUSpalding arm at tests/rho_simple_end_to_end_vs_openfoam.sh:113-118.
- **blocks_tutorial**: none — it lets a case through that should be refused
- **depends_on / middle steps**:
  - Nothing to implement to close it: replace the prefix test with the family list that turbulence_setup.cuh:369-373 already spells out as isNutWallFn. ~3 lines.
  - It is DEAD in the strict sense — the condition cannot fire for the one type the block comment at rhoCreateFields_cpp.cu:521-522 explicitly names as refused. The patch then builds as CalculatedPatchField (fv_patch_field.cuh:1682) and the mirror closure computes the smooth log law (kEpsilon.cu:379; kEpsilon.cu:548 passes atmZ0 = 0 and nothing plumbs z0 in), i.e. exactly the silent substitution the refusal exists to prevent.
  - Actually SUPPORTING it afterwards is item 20's per-patch selector plus the z0 plumbing that turbulence_setup.cuh:386-392 already does for the shipped driver.

</details>
<details><summary><b>[OVER-BROAD]</b> The nut wall-function refusal walks the FILE's boundary entries rather than the mesh's patches, so it fires on entries that resolve to no patch.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:529
- **trigger**: a 0/nut whose boundaryField carries a group or regex key matching no patch on this mesh — e.g. a `"(lowerWall|upperWall)"` block kept from another mesh, or an included boilerplate block.
- **of_feature**: OpenFOAM resolves each PATCH to an entry by name, then by group, then by regex; an entry matching no patch is simply unused.
- **path**: host
- **size**: ~5 lines.
- **fixture**: sbMatched with a spurious `"noSuchPatch.*" { type nutUSpaldingWallFunction; }` block added to 0/nut: it must still run. Control: the same block renamed to match a real wall must still refuse.
- **blocks_tutorial**: none of the six today, but it is a latent refusal of cases OpenFOAM runs
- **depends_on / middle steps**:
  - Iterate `patches` with findPatchEntry, exactly as the T->he block 120 lines earlier does at rhoCreateFields_cpp.cu:408-416. Its comment at :401-407 records this same bug being fixed for T: walking the entries made brae refuse aerofoilNACA0012 over a `cyclic` entry no patch matched, because every modern tutorial includes caseDicts/setConstraintTypes. The nut loop was not converted.
  - ~5 lines.

</details>
<details><summary><b>[DEAD]</b> `RAS { turbulence off; }` is NOT refused — it silently runs the case as molecular-viscosity laminar.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:467 (consumed at rhoSimpleFoam_cpp.cu:67)
- **trigger**: `RAS { RASModel kEpsilon; turbulence off; }`. Line 467 sets f.turbulent = false, and the whole k/epsilon/nut/alphat read at :488-581 sits inside `if (f.turbulent)`, so nut stays empty and rhoSimpleFoam_cpp.cu:67 `turb = f.turbulent && !f.nut.internal.empty()` makes mut = 0 at every cell and face.
- **of_feature**: RASModel::read/correct return immediately when turbulence_ is off, but the model is still CONSTRUCTED — k, epsilon, nut and alphat are all read (alphat is MUST_READ, EddyDiffusivity.C:73-84) — so OpenFOAM's momentum and energy equations run on the FROZEN file nut and file alphat.
- **path**: both
- **size**: ~15 lines.
- **fixture**: sbMatched with `turbulence off;` added, against an OF run of the same. The discriminator is nut: OF's stays at its 0/nut values, brae's is zero, and on sbMatched rho*nut is ~30x the laminar mu (the same ratio recorded at rhoCreateFields_cpp.cu:593-596 for angledDuct).
- **blocks_tutorial**: none — it mis-runs rather than refuses
- **depends_on / middle steps**:
  - Nothing to implement beyond reading the fields: move the k/epsilon/nut/alphat reads out from under the `if (f.turbulent)` guard and gate only the per-iteration correct() on it. ~15 lines.
  - turbulence_setup.cuh:226-232 already gets this right for the shipped drivers — ctl.turbulenceOn plus a noticeApplied saying the model is FROZEN and that this is NOT the same as simulationType laminar. The mirror path has neither the read nor the notice.
  - This is the project's named failure mode with the sign reversed: not a refusal that is too broad, but a substitution with no refusal at all.

</details>
<details><summary><b>[GENUINE]</b> `simulationType` that is neither laminar nor RAS is refused — i.e. LES and DES on the compressible mirror path.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:586
- **trigger**: `simulationType LES;` in constant/turbulenceProperties.
- **of_feature**: compressible::LESModel::New and the sub-grid models under src/TurbulenceModels/turbulenceModels/LES/ and .../DES/.
- **path**: both
- **size**: large.
- **fixture**: None compressible.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - The READER is not the work: turbulence_setup.cuh:99-220 already parses Smagorinsky, WALE, SpalartAllmarasDES/DDES/IDDES and kOmegaSSTDES/DDES/IDDES with their deltas and blending constants, for the shipped drivers.
  - The compressible instantiation of each sub-grid model, plus the delta (cubeRootVol / maxDeltaxyz / IDDESDelta) in the mirror tree — none of which exists there.
  - Lowest priority of the turbulence items: rhoSimpleFoam is STEADY, so an LES on it is not a configuration any tutorial uses.

</details>
<details><summary><b>[OVER-BROAD]</b> `shielding ZDES2020` under SpalartAllmarasIDDES is refused, but OpenFOAM accepts and runs that case.</summary>

- **citation**: src/applications/solvers/common/turbulence_setup.cuh:167-170
- **trigger**: `LES { LESModel SpalartAllmarasIDDES; SpalartAllmarasIDDESCoeffs { shielding ZDES2020; } }`
- **of_feature**: SpalartAllmarasIDDES.H:69-71 declares `class SpalartAllmarasIDDES : public SpalartAllmarasDES` — NOT SpalartAllmarasDDES. The shielding_ member and its read live only in SpalartAllmarasDDES (SpalartAllmarasDDES.C:206-212), so IDDES has no such member and never looks the key up. OpenFOAM leaves it as an unread dictionary entry and runs the standard IDDES.
- **path**: host
- **size**: ~5 lines.
- **fixture**: None. A DES case with the entry added must run and must print the notice; the control is that removing the notice makes the entry invisible again.
- **blocks_tutorial**: none of the six rhoSimpleFoam tutorials; it is reachable from the DES cases the shipped simpleFoam/pimpleFoam drivers run.
- **depends_on / middle steps**:
  - Nothing. brae's reasoning is right — the entry would have no effect — but the ACTION is wrong: it refuses a case OpenFOAM runs, and the supported behaviour (plain IDDES) is exact.
  - The correct shape is noticeApplied, which brae_notice.cuh already provides and which this same function already uses eleven lines of context away at turbulence_setup.cuh:229-232 for the frozen-turbulence switch. ~5 lines.
  - This is a divergence to fix by DOWNGRADING to a notice, never by going silent — a dictionary entry read off disk and dropped is exactly what the notice mechanism exists for.

</details>
<details><summary><b>[DEAD]</b> generalizedNewtonian powerLaw is refused without nuMax, but nuMin and n are silently defaulted where OpenFOAM refuses both.</summary>

- **citation**: src/applications/solvers/common/turbulence_setup.cuh:58-62
- **trigger**: `laminar { model generalizedNewtonian; viscosityModel powerLaw; nuMax 1; n 0.4; }` — no nuMin. The guard at :61 tests only `ctl.gnNuMax &lt;= 0.0`, so this passes and ctl.gnNuMin silently becomes 0.0 (:59). The same holds for `n`, which defaults to 1.0 at :58 — a Newtonian fluid.
- **of_feature**: powerLaw.C:63-65 constructs n_, nuMin_ and nuMax_ as `dimensionedScalar(name, dims, powerLawCoeffs_)`, which FATALS on a missing entry; powerLaw.C:80-82 re-reads all three with readEntry, which also fatals. All three are required in OpenFOAM.
- **path**: host
- **size**: ~4 lines.
- **fixture**: validation has no non-Newtonian case; squareBendLiqNoNewtonian with `n` or `nuMin` deleted is the fixture, and it must refuse.
- **blocks_tutorial**: none — it mis-runs rather than refuses
- **depends_on / middle steps**:
  - Nothing to implement — extend the guard to n and nuMin. ~4 lines. The n default is the dangerous one: n = 1 makes nu = nu0 identically, i.e. the Newtonian answer, on a case that asked for shear thinning. squareBendLiqNoNewtonian records what that is worth: nu sits AT nuMin over essentially the whole field, about 1120x the Newtonian value (turbulence_setup.cuh:33-40).
  - Use a sentinel rather than a plausible default, the way the Maxwell branch at :75-81 already does (nuM = -1, lambda = -1, then refuse).

</details>
<details><summary><b>[GENUINE]</b> generalizedNewtonian viscosityModel other than powerLaw is refused.</summary>

- **citation**: src/applications/solvers/common/turbulence_setup.cuh:52-54
- **trigger**: `laminar { model generalizedNewtonian; viscosityModel BirdCarreau; }`
- **of_feature**: src/TurbulenceModels/turbulenceModels/laminar/generalizedNewtonian/generalizedNewtonianViscosityModels/ — CrossPowerLaw, BirdCarreau, Casson, HerschelBulkley, strainRateFunction, Newtonian, powerLaw.
- **path**: both
- **size**: small per model (~25 lines).
- **fixture**: None; squareBendLiqNoNewtonian is powerLaw.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - Each is an algebraic nu(strainRate) with a clamp, i.e. the same shape as powerLaw. The strain-rate field the powerLaw branch already computes is the whole prerequisite; each model is ~20 lines of formula plus its coefficient read.
  - ctl carries gnPowerLaw/gnN/gnNuMin/gnNuMax as flat scalars, so a second model wants a small selector on DeviceSimpleControls rather than more booleans.

</details>
<details><summary><b>[GENUINE]</b> A laminar `model` other than Stokes, generalizedNewtonian or Maxwell is refused.</summary>

- **citation**: src/applications/solvers/common/turbulence_setup.cuh:87-92
- **trigger**: `laminar { model lambdaThixotropic; }` (or Giesekus, PTT).
- **of_feature**: src/TurbulenceModels/turbulenceModels/laminar/ — Stokes, generalizedNewtonian, Maxwell, lambdaThixotropic, Giesekus, PTT.
- **path**: both
- **size**: medium per model.
- **fixture**: None; OF ships planarPoiseuille/planarContraction for the viscoelastic family.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - Giesekus and PTT are extensions of the Maxwell stress transport ctl.maxwell already carries — same tensor field, one extra nonlinear term each.
  - lambdaThixotropic is different in kind: a scalar structure-parameter transport equation plus nu(lambda), so it needs its own field read in createFields and its own solve in the loop.

</details>
<details><summary><b>[GENUINE]</b> A wall-function BC (nut*/epsilon/omega) on a patch not typed 'wall' in constant/polyMesh/boundary is refused.</summary>

- **citation**: src/applications/solvers/common/turbulence_setup.cuh:359-362
- **trigger**: a 0/nut or 0/epsilon entry naming a wall function on a patch whose polyPatch type is `patch`, `symmetry`, etc. Only fires when the entry resolves to a concrete patch name — group and regex keys are skipped (:357).
- **of_feature**: OpenFOAM applies the wall function wherever the BC OBJECT is placed: nutWallFunctionFvPatchScalarField.C:181-184 is operator==(calcNut()) on the patch's own object, and nothing consults polyPatch::type(). So OpenFOAM RUNS these cases; the limitation is brae's.
- **path**: both
- **size**: medium.
- **fixture**: None registered. A pitzDaily variant with a wall function on a `patch`-typed boundary, run against OF, is the gate; the control is the same case with the patch retyped `wall`, which must agree.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - Drive the wall set from the BC type rather than from FvPatch::type == "wall". Half of this is already done and is the model to copy: turbulence_setup.cuh:489-495 builds ctl.turbWallPatch with findPatchEntry on the SECOND turbulence field's BC type, precisely because the geometric type was wrong there (the turbulentFlatPlate topWall and backwardFacingStep2D regex findings recorded at :477-488).
  - The nut half still uses the geometric type: setNutWall skips any patch whose type is not 'wall' (:379), and the device kernels gate on isWall.
  - So: extend the BC-driven mask to nut, then make the device wall kernels read that mask instead of the geometric flag, then make `y` (wall distance) available for those patches. Medium, and it touches the device wall masks shared by every model.

</details>
<details><summary><b>[GENUINE]</b> Two DIFFERENT nut wall functions on wall patches in the same case are refused.</summary>

- **citation**: src/applications/solvers/common/turbulence_setup.cuh:419-425
- **trigger**: 0/nut with, say, nutUSpaldingWallFunction on one wall and nutkWallFunction on another.
- **of_feature**: nutWallFunctionFvPatchScalarField.C:181-184 dispatches PER PATCH — a virtual calcNut() on each patch's own object — so OpenFOAM honours both.
- **path**: both
- **size**: medium.
- **fixture**: None. sbMatched with two differently-typed wall patches, against OF, gates the positive case; the existing refusal has no registered arm on this driver.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - ctl.nutWall is ONE case-wide enum and the device kernels rewrite EVERY wall face unconditionally (device_kepsilon.cu spaldingNutKernel/blendedNutKernel/nutUWallKernel all write where isWall), so there is nothing to spare the losing patch. The refusal is correct given that design.
  - Lifting it means a PER-FACE nutWall selector array — the same shape ctl.turbWallPatch already has — threaded into the wall kernels, and the same array threaded into the near-wall PRODUCTION term (which today takes a single nutWall argument; kEpsilon.cu:548 passes a literal 0).
  - This is the SAME dependency as the mirror-path nut refusal (item 20): do the per-patch selector once and both lift together.

</details>
<details><summary><b>[GENUINE]</b> `pRefPoint` is refused; only pRefCell is honoured.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:65-67
- **trigger**: a closed case (no p patch fixes a value) whose SIMPLE dict sets pRefPoint instead of pRefCell.
- **of_feature**: findRefCell.C:69-100 — OpenFOAM calls mesh.findCell(pRefPoint), reduces the answer across processors, and fatals if no processor owns it.
- **path**: host
- **size**: small (~60 lines plus a gate).
- **fixture**: None. A rhoBox variant with all-zeroGradient p and a pRefPoint would gate it: the assertion is that brae picks the same cell OpenFOAM reports, and the control is a pRefPoint in a different cell giving a different p level.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - A point-in-cell search over the mirror PrimitiveMesh. Serially a bounding-box prefilter plus a face-normal containment test is enough; the mirror tree has cell face lists and face centres/normals already in FvGeometry.
  - OpenFOAM's parallel reduction and its 'cannot find owner cell' fatal, if this is ever to run decomposed.
  - Nothing else — refCell/refValue flow straight into PressureControl and are already consumed.

</details>
<details><summary><b>[GENUINE]</b> pressureControl refuses pMaxFactor / pMinFactor / rhoMax / rhoMin when no reference pressure (or density) can be evaluated.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:145, :158, :163, :181, :191, :196
- **trigger**: SIMPLE dict naming pMaxFactor/rhoMax/pMinFactor/rhoMin while no p patch fixes a value and no pRefCell is set.
- **of_feature**: pressureControl.C:100-190 — every one of these six conditions is a FatalIOErrorInFunction ... exit(FatalIOError) in OpenFOAM, in the same order and with the same wording.
- **path**: host
- **size**: n/a — comment fix only, ~2 lines.
- **fixture**: None registered. rhoBox with `pMaxFactor` and all-zeroGradient p refuses in both codes; the control is the same case with a fixedValue outlet, which must run.
- **blocks_tutorial**: none of the six
- **depends_on / middle steps**:
  - Nothing. These are faithful mirrors of OpenFOAM's own fatals and there is no feature behind them to port. Lifting any of them would diverge from OpenFOAM.
  - ONE CORRECTION to the code comment: rhoCreateFields_cpp.cu:155-156 says 'OpenFOAM warns and keeps going; brae keeps the same behaviour but says it out loud'. OpenFOAM issues an IOWarning that rhoMax is deprecated (pressureControl.C:122-127) and THEN fatals on exactly the two conditions brae throws on (:129-147). The code is right; the comment describing it is wrong, and the comment is what a reader would trust when deciding whether the refusal is faithful.

</details>

## boundary-factory

### Adjudicated verdicts (disputes and missed items)

| # | kind | item | verdict |
|---|---|---|---|
| 1 | dispute | wall function on non-'wall' patch (was turbulence_setup.cuh:359, OVER-BROAD) | **GENUINE** |
| 2 | dispute | catch-all entry rotatingWallVelocity (GENUINE -> OVER-BROAD objection) | **REJECTED** |
| 3 | dispute | catch-all entry fixedNormalSlip (GENUINE -> OVER-BROAD objection) | **REJECTED** |
| 4 | missed | mixedEnergy loses refValue through the reader's type gate | **HOLE** |
| 5 | missed | surfaceNormalFixedValue/uniformNormalFixedValue `ramp` key silently dropped | **HOLE** |
| 6 | missed | timeVaryingMappedFixedValue accepted path silently diverges from OF (time dirs, mapMethod, offset, setAverage, perturb) | **HOLE** |
| 7 | missed | atmBoundaryLayerInlet{K,Epsilon} silently drop OF's C1/C2 factor | **HOLE** |
| 8 | missed | hasMRF/hasFvOptions refusal family is dead (checker: 'no driver ever raises them') | **REJECTED** |
| 9 | missed | hasCoupledPatches never derived from the mesh | **HOLE** |

Evidence, one line each (current file:line -- line numbers are as of 2ecfe0f):

1. src/applications/solvers/common/turbulence_setup.cuh:374-390 (guardWallFn, called at :473/:484/:500/:501) throws for nut*/epsilon/omega wall functions on a concrete non-wall patch. The surveyor's premise -- 'OpenFOAM honours a wall function wherever it is declared' -- is false for the entire nut family: OF's nutWallFunctionFvPatchScalarField.C:45-57 checkType() aborts on !isA<wallFvPatch> and is called from every constructor (:95/:111/:126/:139/:153), so brae refuses exactly what OF refuses there. epsilonWallFunction/omegaWallFunction carry no checkType in OF v2412, but nearWallDist.C:59 sets y=0.0 on non-wall patches, so OF 'runs' them as division-by-zero garbage epsilon, not as a capability brae is wrongly refusing. Refusal correct and permanent; the checker's objection is upheld.
2. src/finiteVolume/fields/fv_patch_field.cuh:1905 refuses only the unrecognised type name; grep confirms no code anywhere in brae computes Up=(-om)*((Cf-origin)^axis) (OF rotatingWallVelocityFvPatchVectorField.C:124-135, a fixedValue per .H:80-82). The checker's math is right and makes this a cheap S port in the atmBoundaryLayerInletVelocity in-factory style (fv_patch_field.cuh:1813-1850), but OVER-BROAD means an existing capability is being refused -- here none exists, so refusing the unported name by name IS the port boundary the surveyor classified GENUINE. Ease of porting does not reclassify a refusal.
3. src/finiteVolume/fields/fv_patch_field.cuh:1905. The fixedValue==0 equivalence to brae's SymmetryPlanePatchField (fv_patch_field.cuh:1799-1800) is mathematically real, but brae's reader never parses the class's fixedValue entry, so the conditional exact substitution would itself be new parse+dispatch code, not a lifted gate. The checker's own caveat is decisive: all 6 v2412 occurrences are on pointDisplacement (pointPatchFields with an `n` entry) which the fv factory never reads, so no reachable configuration is refused too broadly. GENUINE stands.
4. src/finiteVolume/fields/foam_field_reader.cuh:732 parses refValue only when p.type == "mixed" (refGradient at :727 and valueFraction at :737 are parsed ungated); src/finiteVolume/fields/fv_patch_field.cuh:1726-1735 builds the mixedEnergy MixedPatchField from the never-filled refValues and :937 returns T{} for the empty vector, so the guard at :1728 (valueFraction only) never fires. OF's mixedEnergyFvPatchScalarField derives from mixed (.H:58-60), sets a real refValue = thermo.he(pw, Tw.refValue()) (.C:121) and mixed's write() persists all three entries -- so a gate reading OF-written he (the stated purpose of this path, fv_patch_field.cuh:1497-1512) silently blends toward refValue 0 wherever valueFraction>0. Silent substitution, no refusal.
5. No 'ramp' key exists anywhere in src/finiteVolume/fields/foam_field_reader.cuh -- it falls into the unhandled-entry skip at :767-772 -- and the contract comment at fv_patch_field.cuh:334 still says 'any time-ramp/Function1 is ignored'. OF multiplies the patch values by ramp_->value(t) every updateCoeffs (surfaceNormalFixedValueFvPatchVectorField.C:162-165; uniformNormal... .C:162-164, ramp_ read via Function1::NewIfPresent at .C:58-59). This session's aac18cd fix covers only the refValue/uniformValue slot (foam_field_reader.cuh:508-540 marking; refusals at fv_patch_field.cuh:1649-1663), so simpleCar 0.orig/U intakeType1 (`refValue uniform 1.2; ramp table ((0 0)(10 1))`) still builds a frozen 1.2*n with no refusal -- and a ramp ending below 1 converges silently to the wrong inlet even in steady state.
6. src/finiteVolume/fields/foam_field_reader.cuh:316-347 (readTimeVaryingMapped) takes only the SMALLEST boundaryData time directory and holds it, and the factory maps nearest-point (fv_patch_field.cuh:1667 comment); mapMethod/offset/setAverage/perturb keys hit the unhandled-entry skip at foam_field_reader.cuh:767. OF's MappedFile.C reads setAverage/perturb/offset (:47-67), validates mapMethod with planar interpolation as the default (:119-128, nearestOnly at :439-441 only when explicitly 'nearest'), and interpolates between bracketing time directories each step. A case with several time dirs, a planar-critical patch, or any of those keys runs silently different from OF with no refusal anywhere.
7. src/finiteVolume/fields/fv_patch_field.cuh:1866-1869 computes k = Ustar^2/sqrt(Cmu) and eps = Ustar^3/(kap*(zr+z0)) with no sqrt(C1*log((z+z0)/z0)+C2) factor, and the reader parses only kappa/Cmu/d/z0/Uref/Zref/flowDir/zDir for ABL patches (foam_field_reader.cuh:440-506) -- C1/C2 fall to the skip at :767, neither parsed nor refused. OF atmBoundaryLayer.C:70-71 reads C1_(getOrDefault 0.0)/C2_(getOrDefault 1.0) and applies the factor at :236-240 (k) and :250-254 (epsilon). A YGCJ-profile case setting C1/C2 silently gets Richards-Hoxey. The brae comment at :1817 even names the default assumption without enforcing it.
8. The mirror path's whole-case driver populates both flags FROM THE CASE: tests/test_rho_simple_step_cpp.cu:197-227 scans system/fvOptions + constant/fvOptions (including cpu::fvOptions firstUnsupported at :205-213) and sets in.hasMRF = has("constant/MRFProperties") at :221 -- landed in commit 0fe6626 (2026-08-26), before the census. So the refusals at rhoPEqn_cpp.cu:27/:32, rhoEEqn_cpp.cu:31 and the UEqn twins are reachable from a real case on the _cpp mirror path; the fail-proof hand-sets the checker cites are the six control tests, not the only setters. The production GPU driver independently refuses MRF and unsupported fvOptions at read time (gpuRhoSimpleFoam.cu:382-416). The 'ONE driver-side populate step' the checker demands substantially exists.
9. The flag exists only on the device mirror structs (rhoSimpleFoam.cuh:219, rhoUEqn.cuh:170, rhoPEqn.cuh:127, rhoEEqn.cuh:111) and is set true only by fail-proof tests; nothing anywhere computes it from patch types. On the CUDA path the harness's own pre-check refuses coupled patches first (tests/test_rho_simple_step_cuda.cu:174-177), so those device throws are dead but guarded. On the _cpp whole-case path there is NO check and no flag at all (rhoSimpleFoam_cpp.cuh:145-146 carries only hasMRF/hasFvOptions, and test_rho_simple_step_cpp.cu greps clean for cyclic/coupled): the factory builds cyclic as a zeroGradient placeholder (fv_patch_field.cuh:1795-1797) whose comment justifies the no-op by 'the device skip' -- which the host mirror does not perform -- so a cyclic case pointed at the _cpp harness runs silently uncoupled. Guard fails to fire; silent substitution.

### Original survey entries closed by this session

- hasFixedFluxPressure DEAD refusal (pEqn_cpp.cu:27 + five siblings) -- resolved by the full fixedFluxPressure port this session: FixedFluxPressurePatchField refuses assembly if updateSnGrad never ran (fv_patch_field.cuh:47-49, :592-605), constrainPressure is transcribed in the three _cpp pressure equations and deviceConstrainPressure runs on the device (pEqn.cu:262, rhoPEqn.cu:273; commits 84828bd, 7d6ffdd), the hasFixedFluxPressure flags and their six dead throws are deleted (zero hits in src/ and tests/), and the V2 envelope substring blocker was lifted with the MRF+ffp combination refused at the device call site (c2ff3e4).
- surfaceNormalFixedValue/uniformNormalFixedValue silent zero-inlet HOLE (foam_field_reader.cuh:530) -- resolved by aac18cd: the reader's else branch now sets p.unsupportedFunction1 instead of skipping (foam_field_reader.cuh:531-540) and the factory refuses both the unevaluable Function1 and a missing refValue/uniformValue by name (fv_patch_field.cuh:1649-1663). Note the adjacent `ramp` key was NOT covered and remains open as a missed HOLE.
- fixedMean/fanPressure/coded unhonoured-contract DEAD guard (fan_pressure.cuh:178, last census entry) -- resolved by 6b4fe1e + a02af1c: frozen_bc_guard.cuh refuses fixedMean, fanPressure, codedFixedValue and codedMixed by name at read time on every driver that cannot maintain them per step -- simpleFoam mirror createFields_cpp.cu, rhoCreateFields_cpp.cu, simpleFoamV2.cu, gpuSimpleFoam.cu (where the coded pair stays maintained and is exempted via codedMaintained) and the turbulence fields via readTurbulenceFields' frozenGuardDriver parameter (turbulence_setup.cuh:353-362) -- making the previously unreachable refusal fire exactly where the freeze would have happened.

### The original survey for this area (as surveyed; see the verdicts above for corrections)

<details><summary><b>[GENUINE]</b> Catch-all: any `type` word the factory does not recognise is refused by name. This is the port boundary itself. ACCEPTED (46 spellings): fixedValue, uniformFixedValue, codedFixedValue, fixedEnergy, fixedGradient, gradientEnergy, mixed, mixedEnergy, codedMixed, fixedMean, zeroGradient, noSlip, calculated, empty, symmetry, symmetryPlane, slip, wedge, cellMotion, movingWallVelocity, cyclic/cyclicAMI/cyclicACMI/cyclicPeriodicAMI (placeholder), inletOutlet, outletInlet, freestream, freestreamVelocity, freestreamPressure, totalPressure, uniformTotalPressure, fanPressure, fixedFluxPressure, flowRateInletVelocity, surfaceNormalFixedValue, uniformNormalFixedValue, timeVaryingMappedFixedValue, pressureInletOutletVelocity, turbulentIntensityKineticEnergyInlet, turbulentMixingLength{DissipationRate,Frequency}Inlet, kqRWallFunction, epsilonWallFunction, omegaWallFunction, nutkWallFunction, nutUWallFunction, nutLowReWallFunction, nutUBlendedWallFunction, nutUSpaldingWallFunction, atmNutkWallFunction, compressible::alphatWallFunction, atmBoundaryLayerInlet{Velocity,K,Epsilon,Omega}. EXACT SUBSTITUTIONS (built as a different class, condition each is exact under): fixedEnergy/gradientEnergy/mixedEnergy -> fixedValue/fixedGradient/mixed, exact because OF's energy patches ARE those base classes for the matrix and OF writes the entry each base needs; uniformFixedValue(constant) -> fixedValue, exact for steady; freestream -> inletOutlet, exact (OF freestreamFvPatchField derives from it); kqRWallFunction -> zeroGradient, exact (OF's class IS zeroGradient); nutk/nutU/nutLowRe/nutUBlended/atmNutk/compressible::alphatWallFunction -> calculated, exact because the model writes the value (the wall-nut KIND is carried separately by ctl.nutWall); omegaWallFunction -> the epsilonWallFunction class, exact as a boundary VALUE (both are zeroGradient with the near-wall cell constrained separately) and load-bearing, since mapping it to a bare ZeroGradient left kOmegaSST with no wall faces; slip/symmetry/symmetryPlane -> one SymmetryPlane class, exact for slip (OF slip derives from basicSymmetry) and for symmetry vs symmetryPlane on a planar patch; cellMotion -> fixedValue, exact (it holds what the motion solver put there); movingWallVelocity -> fixedValue, exact on a STATIC mesh where OF never assigns, and driver-updated when the mesh moves; codedFixedValue/codedMixed -> fixedValue/mixed seed, exact ONLY where the NVRTC kernel is wired (see the last entry); atmBoundaryLayerInlet* -> fixedValue evaluated in-factory from the Richards-Hoxey profile, exact for constant parameters. UNPORTED, ranked by occurrence across every v2412 tutorial 0/ and 0.orig/: adjointWallVelocity 190, adjointOutletPressure 125, compressible::turbulentTemperatureRadCoupledMixed 40, mapped 38, prghPressure 32, waveVelocity 27, MarshakRadiation 26, greyDiffusiveRadiation 22, mappedFile 21, outletMappedUniformInlet 16, waveTransmissive 13, fluxCorrectedVelocity 11, copiedFixedValue 11, prghTotalPressure 10, advective 6, fixedNormalSlip 6, turbulentDigitalFilterInlet 6, rotatingWallVelocity 5, nutkRoughWallFunction 4, turbulentInlet 2.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1802
- **trigger**: any boundaryField entry whose `type` is not one of the 46 accepted spellings; also fires for a constraint patch synthesised from an unhandled mesh patch type (geometric_field.cuh:92-97)
- **of_feature**: fvPatchField run-time selection table — finiteVolume/fields/fvPatchFields/fvPatchField/fvPatchField.C, with each derived class under finiteVolume/fields/fvPatchFields/derived/
- **path**: both
- **size**: per-type: S for the fixedValue-shaped ones (copiedFixedValue, fixedNormalSlip, partialSlip, rotatingWallVelocity, ~50-120 lines each); M for waveTransmissive/advective and nutkRoughWallFunction; L for mapped* and the adjoint family
- **fixture**: none for the catch-all itself; tests/test_alphat_bc.cu exercises the message shape ("unsupported BC type") for one name
- **blocks_tutorial**: none of the six OF rhoSimpleFoam tutorials — aerofoilNACA0012, angledDuctExplicitFixedCoeff, gasMixing, squareBend, squareBendLiq, squareBendLiqNoNewtonian use only accepted types
- **depends_on / middle steps**:
  - nothing generic — each unported name is its own port, and the factory is already the single dispatch point, so adding a row costs the class plus its per-step update hook
  - the adjoint* family (315 of the occurrences) needs the adjoint solver and is out of scope
  - mapped / mappedFile / outletMappedUniformInlet need mappedPatchBase: a patch-to-patch sampling and addressing layer brae has only for cyclicAMI
  - prghPressure / prghTotalPressure need a gravity-hydrostatic p_rgh formulation, i.e. a buoyant solver, before the BC means anything
  - waveTransmissive / advective need an outgoing-wave (Courant-weighted ddt) boundary operator, for which the fvm assembly has no slot

</details>
<details><summary><b>[GENUINE]</b> uniformFixedValue whose `uniformValue` is a Function1 brae cannot evaluate (expression / coded / polynomial / csvFile) is refused rather than degraded to a stale `value` entry.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1492
- **trigger**: `type uniformFixedValue; uniformValue { type expression; ... }` (or polynomial / coded / csvFile). The reader records the Function1's name at foam_field_reader.cuh:602/637/644 and the factory throws on it. `constant`, `uniform` and OF's bare-value shorthand are accepted; `table` is implemented and deliberately NOT recorded as unsupported.
- **of_feature**: uniformFixedValueFvPatchField — finiteVolume/fields/fvPatchFields/derived/uniformFixedValue/uniformFixedValueFvPatchField.C, taking a PatchFunction1; the unevaluable forms are OpenFOAM/expressions/Function1/Function1Expression.C and meshTools/PatchFunction1/CodedField/CodedField.C
- **path**: both
- **size**: S for polynomial/csvFile (~150 lines); L for expression (a small expression compiler, ~600-1000 lines plus its own gate)
- **fixture**: tests/test_uniform_function1.cu — refuses `expression`, accepts `constant`, and carries the negative control that a plain fixedValue is undisturbed
- **blocks_tutorial**: squareBendLiq (0.orig/T: the `(?i).*walls` entry re-declares itself uniformFixedValue with an expression uniformValue over a stale `value uniform 350`)
- **depends_on / middle steps**:
  - polynomial + csvFile first: pure table-shaped Function1s that slot next to the existing Function1::table (foam_field_reader.cuh:570-597) and its per-step refresh — no new machinery
  - expression needs an exprField evaluator: a scalar expression parser over patch fields with OF's variable set (internalField(U), snGrad(T), deltaT(), time(), mag/max). This is the piece squareBendLiq actually needs and the largest item in this area.
  - coded needs the existing NVRTC path (src/cuda/device_coded_bc.cu + src/applications/solvers/common/coded_bc_setup.cuh) generalised from `a patch type` to `a Function1 slot`, AND wired into the rhoSimpleFoam driver, which never calls setupCodedBCs

</details>
<details><summary><b>[GENUINE]</b> flowRateInletVelocity whose massFlowRate / volumetricFlowRate is given as a Function1 DICTIONARY is refused at read time; only `constant &lt;v>` and a bare value are accepted.</summary>

- **citation**: src/finiteVolume/fields/foam_field_reader.cuh:672
- **trigger**: `massFlowRate { type coded; name liquidIn; code #{ return 5; #}; }` — the tokeniser sees `{` after the key and throws instead of descending into the sub-dictionary
- **of_feature**: flowRateInletVelocityFvPatchVectorField — finiteVolume/fields/fvPatchFields/derived/flowRateInletVelocity/flowRateInletVelocityFvPatchVectorField.C:71-83, which builds the rate with Function1&lt;scalar>::NewIfPresent, so ANY Function1 is legal there
- **path**: host
- **size**: S (~60 lines) once the Function1 dictionary reader exists; the reader itself is the M-sized part
- **fixture**: none — tests/test_rho_simple_step_cpp.cu and validation/flowrate_vs_openfoam.sh cover the constant form and give the negative control
- **blocks_tutorial**: squareBendLiq (0.orig/U inlet). This is the SECOND independent refusal that case hits — the expression on T is the first — so lifting either alone does not run it.
- **depends_on / middle steps**:
  - a general Function1 sub-dictionary reader in TokenStream — today constant and table are open-coded inline per key; this is the same prerequisite as the uniformFixedValue entry and should be done once
  - for the coded form: the NVRTC path generalised to a Function1 returning ONE scalar per step, which is strictly easier than a coded patch because the result is a number, not a face field
  - no new update hook: fvPatchField::updateFromDensity (fv_patch_field.cuh:401) is already called at every momentum assembly, so a time-varying rate only needs the value pushed in there

</details>
<details><summary><b>[GENUINE]</b> totalPressure / uniformTotalPressure with a p0 Function1 brae cannot evaluate (polynomial, csvFile, expression) is refused; constant, uniform and table are supported.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1473
- **trigger**: a `p0` entry that is not uniform/nonuniform/constant/table — recorded by the reader at foam_field_reader.cuh:600-604
- **of_feature**: uniformTotalPressureFvPatchScalarField (p0 sampled from a Function1 every updateCoeffs) and totalPressureFvPatchScalarField — finiteVolume/fields/fvPatchFields/derived/totalPressure/totalPressureFvPatchScalarField.C
- **path**: host
- **size**: S — rides on whatever the Function1 work delivers
- **fixture**: none; pimpleFoam/RAS/TJunction is the table case that already runs and is the negative control
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the same Function1 evaluator as the uniformFixedValue entry — this refusal is the narrow residue left after `table` was implemented, so it no longer stands alone
  - nothing else: the per-step p0 refresh already exists (the solver re-reads p0 each step and pushes it through updateFromPatchVelocity and the device tpMask)

</details>
<details><summary><b>[GENUINE]</b> atmBoundaryLayerInlet{Velocity,K,Epsilon,Omega} with a time-varying Function1 parameter (Uref, Zref, z0, flowDir, zDir) is refused rather than silently falling back to the default for that parameter.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1483
- **trigger**: a wind-rose or ramped Uref/flowDir in the case's include/ABLConditions — takeConstantFunction1 (foam_field_reader.cuh:213-225) records the name and the factory throws
- **of_feature**: atmBoundaryLayer — atmosphericModels/derivedFvPatchFields/atmBoundaryLayer/atmBoundaryLayer.C:218-224, the Richards-Hoxey profile brae already evaluates verbatim including the groundMin = zDir & ppMin origin
- **path**: host
- **size**: S (~80 lines) on top of the Function1 evaluator
- **fixture**: none; validation/turbinesiting_cuda_vs_openfoam.sh runs the constant form and is the natural control
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the shared Function1 evaluator
  - a per-step re-evaluation hook on a factory-built fixedValue: the profile is computed once in the constructor (fv_patch_field.cuh:1714-1768), so a time-varying Uref needs it moved behind a virtual update in the updateFromDensity family

</details>
<details><summary><b>[GENUINE]</b> totalPressure with a named `psi` entry — OpenFOAM's isentropic high-speed branch — is refused rather than approximated by the low-speed form brae reproduces exactly.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1533
- **trigger**: `type totalPressure; psi thermo:psi; gamma 1.4;` on a compressible case. psiName defaults to "none" (foam_field_reader.cuh:127), so only a case that names psi fires it.
- **of_feature**: totalPressureFvPatchScalarField::updateCoeffs — finiteVolume/fields/fvPatchFields/derived/totalPressure/totalPressureFvPatchScalarField.C:167-200. Three branches: psi==none -> p0 - 0.5*rho*neg(phi)*magSqr(U) (ported); psi named with gamma>1 -> p0/(1 + 0.5*psi*gM1ByG*neg(phi)*magSqr(U))^(1/gM1ByG); psi named with gamma==1 -> p0/(1 + 0.5*psi*neg(phi)*magSqr(U)).
- **path**: both
- **size**: S-M (~120 lines across host + device + the psi boundary plumb), plus a gate
- **fixture**: none for the isentropic branch; validation/rhoTP carries the low-speed form (both totalPressure and pressureInletOutletVelocity) and is the ready-made control
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - psi at the PATCH: brae's rhoThermo computes psi in the cells (src/thermophysicalModels/device_thermo.cu) but never exposes a boundary psi; it needs the same treatment rho_b already gets (rhoBnd assembled at rhoSimpleFoam_cpp.cu:226/252)
  - the `gamma` entry is already parsed and stored (foam_field_reader.cuh gammaTP) and consumed by nothing — one of the parsed-never-applied inputs, which this port would finally use
  - host: TotalPressurePatchField::updateFromPatchVelocity gains a second formula selected by psiName
  - device: the tpMask kernel in src/cuda/device_boundary.cuh computes only the low-speed form from p0/rho/phi/|U| and needs psi as a fifth per-face input

</details>
<details><summary><b>[GENUINE]</b> flowRateInletVelocity with `extrapolateProfile true` is refused — it rescales the extrapolated internal profile instead of imposing a uniform normal velocity.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1550
- **trigger**: `extrapolateProfile true;` on a flowRateInletVelocity patch (parsed at foam_field_reader.cuh:682-686)
- **of_feature**: flowRateInletVelocityFvPatchVectorField::updateValues — finiteVolume/fields/fvPatchFields/derived/flowRateInletVelocity/flowRateInletVelocityFvPatchVectorField.C:162-192: Up = patchInternalField(); strip the normal part; clamp reverse flow with min(nUp,0); then scale by |flowRate|/|estimatedFlowRate| when estimatedFlowRate > 0.5*flowRate, else shift by (flowRate-estimatedFlowRate)/gSum(rho*magSf)
- **path**: both
- **size**: S on host (~40 lines, the OF branch transcribes directly); S-M on device
- **fixture**: none — validation/flowrate_vs_openfoam.sh and validation/fr_vs_openfoam.sh gate the uniform branch and give the control
- **blocks_tutorial**: none (no rhoSimpleFoam tutorial sets it)
- **depends_on / middle steps**:
  - patchInternalField for U — already available on both paths (the piov and symmetry evaluators use it)
  - a patch-wide gSum with the parallel reduction — already available; the branch point is FlowRateInletVelocityPatchField::updateFromDensity (fv_patch_field.cuh:401)
  - the per-face boundary rho, which the mass-flow branch already demands and refuses without (fv_patch_field.cuh:409)
  - device: a patch reduction followed by a per-face write is a new kernel shape for device_boundary_flow.cu, whose uniform branch is a pure per-face write

</details>
<details><summary><b>[GENUINE]</b> pressureInletOutletVelocity carrying a `tangentialVelocity` entry is refused, because brae's piov kernel sets the tangential refValue to zero — running it would silently solve a swirl-free inlet where the case asked for swirl.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1460
- **trigger**: `type pressureInletOutletVelocity; tangentialVelocity uniform (0 5 0);` (parsed at foam_field_reader.cuh:759). The plain form — every simpleFoam/rhoSimpleFoam tutorial usage — is accepted.
- **of_feature**: pressureInletOutletVelocityFvPatchVectorField::setTangentialVelocity — finiteVolume/fields/fvPatchFields/derived/pressureInletOutletVelocity/pressureInletOutletVelocityFvPatchVectorField.C:130-136: refValue() = tangentialVelocity - n*(n & tangentialVelocity), so the tangential component is DRIVEN; without the entry OF sets refValue = Zero (line 95), which is exactly what brae does
- **path**: both
- **size**: S (~60 lines host + device) — the cheapest genuine refusal in this area
- **fixture**: tests/test_uniform_function1.cu:108-153 — refuses the entry by name AND carries the negative control that the plain form still builds
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - a per-face vector refValue on the piov patch: PressureInletOutletVelocityPatchField (fv_patch_field.cuh:1253, bcCategory 6) carries no refValue array, and the device piovMask kernel writes n*(n & U_cell) with no additive term
  - nothing in the reader — it already stores the tangentialVelocity field
  - device: one extra per-face vector buffer in src/cuda/device_boundary.cuh plus an add; the directionMixed valueFraction (sqr(n)) is unchanged

</details>
<details><summary><b>[GENUINE]</b> alphatJayatillekeWallFunction (bare and `compressible::` spellings) is refused rather than accepted as the simple alphatWallFunction, because its thermal-sublayer P-function gives a different wall heat flux.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1676
- **trigger**: `type compressible::alphatJayatillekeWallFunction;` on 0/alphat — any conjugate-heat or heated-wall compressible case. Plain compressible::alphatWallFunction is accepted and mapped to `calculated` (the model writes rho_w*nut_w/Prt).
- **of_feature**: alphatJayatillekeWallFunctionFvPatchScalarField — thermoTools/derivedFvPatchFields/wallFunctions/alphatWallFunctions/alphatJayatillekeWallFunction/alphatJayatillekeWallFunctionFvPatchScalarField.C:220-324, with Psmooth(Prat) at :95 and a 10-step Newton solve for yPlusTherm at :104-120
- **path**: both
- **size**: M (~250 lines plus a gate against an OF-instrumented alphat), gated on exposing the wall qDot
- **fixture**: tests/test_alphat_bc.cu — asserts compressible::alphatWallFunction loads AND that alphatJayatillekeWallFunction is refused by name, so it is already the fail-proof; lifting the refusal must flip that assertion into a numeric comparison
- **blocks_tutorial**: none of the six (all use compressible::alphatWallFunction)
- **depends_on / middle steps**:
  - wall y+ per face — exists (the nut wall functions compute it; Jayatilleke uses the same yPlus(turbModel))
  - turbModel.alpha(patchi), the LAMINAR thermal diffusivity at the wall, plus mu(patchi) and rho(patchi): brae has alpha and mu in the cells (src/thermophysicalModels/device_thermo.cu) and a per-face Prt_b (device_thermo.cu:545), but not the patch-face laminar alpha
  - he's boundary snGrad and alphaEff(alphatw) — the wall heat flux qDot the whole formula is built on; the energy equation has these but exposes no per-face qDot to the wall-function layer
  - Psmooth + the yPlusTherm Newton iteration — pure arithmetic, ~30 lines, no dependency
  - a PER-PATCH override in the alphat write: deviceAlphat (device_thermo.cu:267-292) writes rho*nut/Prt over every face unconditionally, so this needs the same per-face wall-function dispatch that turbulence_setup.cuh:420 currently refuses to have more than one of

</details>
<details><summary><b>[GENUINE]</b> externalWallHeatFluxTemperature is refused in every mode, including the flux/power modes that reduce to a fixedGradient brae can already express, because the gradient is q/kappaEff and kappaEff changes every outer iteration.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1782
- **trigger**: `type externalWallHeatFluxTemperature;` on 0/T (17 occurrences across v2412 tutorials); any of the three modes fires it
- **of_feature**: externalWallHeatFluxTemperatureFvPatchScalarField::updateCoeffs — thermoTools/derivedFvPatchFields/externalWallHeatFluxTemperature/externalWallHeatFluxTemperatureFvPatchScalarField.C. fixedPower and fixedHeatFlux set refGrad = (q + qr)/kappa(Tp) with valueFraction 0; fixedHeatTransferCoeff builds a Robin blend from h, Ta, thicknessLayers/kappaLayers and an optional radiative hrad. kappa(Tp) is temperatureCoupledBase::kappa (thermoTools/derivedFvPatchFields/temperatureCoupledBase/temperatureCoupledBase.C), which for kappaMethod fluidThermo returns turbulence->kappaEff(patchi) — confirming brae's Cp*(alpha+alphat) reading.
- **path**: both
- **size**: M for flux/power modes (~200 lines plus a gate); L for the full three-mode class with layers and radiation. The device already has a refGrad slot (device_boundary.cuh:36), so storage is not the obstacle.
- **fixture**: none. The workaround the message names (`type fixedGradient; gradient uniform &lt;q/kappa>;`) is discretised exactly and is the ready control.
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - a per-outer-iteration updateCoeffs hook for a T/he patch: brae's MixedPatchField has a FIXED valueFraction and no per-step recompute (stated at fv_patch_field.cuh:1619-1622), so this needs a virtual in the updateFromFlux / updateFromDensity / updateFromPatchVelocity family
  - kappaEff at the boundary faces = Cp*(alpha + alphat)_w: alphat_w exists via the alphatWallFunction path and Cp/alpha come from the thermo, but no assembled per-face kappaEff exists today
  - the he&lt;-T boundary derivation must carry the recomputed gradient: rhoCreateFields derives he's BCs from T's once, so a T patch whose refGrad moves each iteration needs gradientEnergy re-derived each iteration
  - PatchFunction1 for q / h / Ta (the constant subset covers most cases) — the same Function1 dependency as above
  - beyond flux mode: thicknessLayers/kappaLayers (a series resistance, trivial once the hook exists) and the emissivity/hrad radiative term (needs sigma and a per-face nonlinear update)

</details>
<details><summary><b>[GENUINE]</b> timeVaryingMappedFixedValue with no readable constant/boundaryData/&lt;patch> directory is refused rather than degraded to the `value` entry.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1569
- **trigger**: the type is named but readBoundaryData found no points/values pair (hasMapData false, foam_field_reader.cuh:346). With data present the patch IS built, as a nearest-point map onto the faces.
- **of_feature**: timeVaryingMappedFixedValueFvPatchField — finiteVolume/fields/fvPatchFields/derived/timeVaryingMappedFixedValue/timeVaryingMappedFixedValueFvPatchField.C; OF also fatals without boundaryData and treats `value` as only the initial field
- **path**: host
- **size**: the refusal: none (permanent). The time-interpolation gap behind it: M (~200 lines).
- **fixture**: none
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the refusal itself is permanent — it mirrors OF's own fatal
  - the real gap BEHIND it: brae reads the single earliest boundaryData time directory and holds it, so a genuinely time-varying mapped inlet is a silent freeze rather than a refusal. Closing that needs (a) enumerating the time directories, (b) interpolating between the bracketing times each step, (c) a per-step refValue push like the coded/fanPressure hook.
  - the spatial map is nearest-point only; OF's default is planarInterpolation (triangulated 2-D), so an exact port needs that too

</details>
<details><summary><b>[GENUINE]</b> A `fixedGradient` (or gradientEnergy) patch with no `gradient` entry is refused.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1451
- **trigger**: `type fixedGradient;` with the `gradient` keyword absent
- **of_feature**: fixedGradientFvPatchField — finiteVolume/fields/fvPatchFields/basic/fixedGradient/fixedGradientFvPatchField.C, whose dict constructor reads `gradient` as a required entry; OF fatals identically
- **path**: host
- **size**: none (permanent)
- **fixture**: none needed
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing — permanent, mirrors OF's own required-entry fatal

</details>
<details><summary><b>[GENUINE]</b> A `mixed`/`mixedEnergy` patch with no `valueFraction` is refused rather than run with a guessed blend — but the check is NARROWER than OpenFOAM's.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1626
- **trigger**: `type mixed;` without `valueFraction`. brae checks only that one entry and silently defaults refValue and refGradient to 0 when they are absent, where OF fatals on any of the three missing.
- **of_feature**: mixedFvPatchField::readMixedEntries — finiteVolume/fields/fvPatchFields/basic/mixed/mixedFvPatchField.C: `if (!hasValue || !hasGrad || !hasFrac) FatalIOError &lt;&lt; "Required entries:" ...`
- **path**: host
- **size**: S (~6 lines to tighten to OF's own condition)
- **fixture**: none; tests/test_liquid_fixed_bc.cu exercises the accepted mixed path and is the control
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing to lift — but there is a small gap to CLOSE: extend the check to refValue and refGradient so brae refuses exactly where OF fatals. foam_field_reader.cuh already sets hasRefValue and hasGradient, so it is ~6 lines.

</details>
<details><summary><b>[GENUINE]</b> flowRateInletVelocity with neither `volumetricFlowRate` nor `massFlowRate` is refused.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1547
- **trigger**: the type named with no rate entry
- **of_feature**: flowRateInletVelocityFvPatchVectorField — finiteVolume/fields/fvPatchFields/derived/flowRateInletVelocity/flowRateInletVelocityFvPatchVectorField.C:85-91, the identical FatalIOError
- **path**: host
- **size**: none (permanent)
- **fixture**: none
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing — permanent, mirrors OF

</details>
<details><summary><b>[GENUINE]</b> Four field-type guards: flowRateInletVelocity, surfaceNormalFixedValue/uniformNormalFixedValue and atmBoundaryLayerInletVelocity refuse on a non-vector field; atmBoundaryLayerInlet{K,Epsilon,Omega} refuses on a non-scalar field.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1556
- **trigger**: a scalar field (0/T, 0/p, 0/k) naming a vector-only BC, or a vector field naming atmBoundaryLayerInletK. Reachable only from a hand-edited case. Companion citations: :1562, :1746, :1770.
- **of_feature**: the per-Type run-time selection tables — e.g. surfaceNormalFixedValueFvPatchVectorField is registered only for vector (finiteVolume/fields/fvPatchFields/derived/surfaceNormalFixedValue/surfaceNormalFixedValueFvPatchVectorField.C)
- **path**: host
- **size**: none (permanent)
- **fixture**: none
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing — permanent type guards that fall out of brae having ONE templated factory where OF has one table per Type

</details>
<details><summary><b>[GENUINE]</b> Two runtime invariants on flowRateInletVelocity's mass branch: a boundary rho array shorter than the patch, and a non-positive gSum(rho*magSf), each refuse rather than fall back to rho = 1.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:409
- **trigger**: a driver that pushes a boundary rho with fewer faces than the patch into updateFromDensity, or a patch whose rho*area sums to zero. Companion citation :421. These are wiring invariants, not case features — they exist because silently using the volumetric form for a massFlowRate inlet rescales the whole inlet by rho, which is the shape of the angledDuct defect.
- **of_feature**: flowRateInletVelocityFvPatchVectorField::updateCoeffs — finiteVolume/fields/fvPatchFields/derived/flowRateInletVelocity/flowRateInletVelocityFvPatchVectorField.C:200-215, where OF looks rho up in the registry and falls back to rhoInlet only when no rho is registered
- **path**: host
- **size**: none (permanent)
- **fixture**: tests/test_rho_simple_step_cpp.cu and validation/flowrate_vs_openfoam.sh exercise the healthy path
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing to lift — keep. Worth checking instead that every driver building such a patch actually CALLS updateFromDensity; on the mirror path only rhoSimpleFoam_cpp.cu:226 and rhoCreateFields_cpp.cu:312 do.

</details>
<details><summary><b>[GENUINE]</b> A turbulent inlet (turbulentIntensityKineticEnergyInlet / turbulentMixingLength{DissipationRate,Frequency}Inlet) whose driving field (U or k) is supplied for fewer faces than the patch refuses rather than filling the rest with zero.</summary>

- **citation**: src/finiteVolume/fields/fv_patch_field.cuh:1151
- **trigger**: the turbulence model calls updateTurbulentInlet with a short U or k boundary array; the only callers are kEpsilon_cpp.cu:313/482 and kOmegaSST_cpp.cu:415/544
- **of_feature**: turbulentIntensityKineticEnergyInletFvPatchScalarField / turbulentMixingLengthDissipationRateInletFvPatchScalarField / turbulentMixingLengthFrequencyInletFvPatchScalarField — TurbulenceModels/turbulenceModels/derivedFvPatchFields/, all inletOutlet-derived with refValue recomputed in updateCoeffs
- **path**: host
- **size**: none (permanent); the LES/laminar wiring gap behind it is S
- **fixture**: validation/ke_vs_openfoam.sh and the rho_kepsilon / rho_komegasst gates exercise the healthy path
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing to lift — a wiring invariant. The real gap it guards: updateTurbulentInlet is called ONLY from the two _cpp RAS models, so a laminar or LES run carrying these BCs builds them as an inletOutlet seeded from the file `value` and never recomputes.

</details>
<details><summary><b>[GENUINE]</b> fixedMean on any field other than p is refused, because brae maintains the prescribed mean for the pressure only — on any other field the patch would be a plain fixedValue frozen at its initial value.</summary>

- **citation**: src/applications/solvers/common/fan_pressure.cuh:178
- **trigger**: `type fixedMean;` on U/k/epsilon/omega/nuTilda/nut, checked by collectFixedMean against exactly that field list
- **of_feature**: fixedMeanFvPatchField::updateCoeffs — finiteVolume/fields/fvPatchFields/derived/fixedMean/fixedMeanFvPatchField.C; the formula is already ported verbatim as applyFixedMean (fan_pressure.cuh:139-153) and is field-agnostic
- **path**: device
- **size**: S (~80 lines to generalise the hook)
- **fixture**: tests/test_fixed_mean.cu
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - a generic per-step patch-functional hook on the device solver: DeviceSimpleSolver::setFixedMean (device_simple_foam.cuh:575) is typed to the pressure boundary only, and generalising it to any field's DeviceBoundary is the whole job
  - nothing on the arithmetic side — solver and test already share one copy of the formula

</details>
<details><summary><b>[GENUINE]</b> fanPressure refuses five entry shapes: `nonDimensional true`, a fanCurve type other than table/tableFile, a missing fanCurve, a curve with fewer than two points, and a `direction` other than in/out.</summary>

- **citation**: src/applications/solvers/common/fan_pressure.cuh:87
- **trigger**: any of those entries on a `type fanPressure` p patch. Companion citations :83 (direction), :93 (no fanCurve), :96 (fanCurve type), :126 (short curve).
- **of_feature**: fanPressureFvPatchScalarField — finiteVolume/fields/fvPatchFields/derived/fanPressure/fanPressureFvPatchScalarField.C; the dimensional table form is ported (p0 shifted by the interpolated pressure rise at the current patch volumetric flow rate, then totalPressure's own face treatment)
- **path**: host
- **size**: S (~60 lines for nonDimensional)
- **fixture**: none for the refusals; the dimensional path runs under pimpleFoam
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nonDimensional: the fan rpm and mean diameter from the dict plus the non-dimensionalising group OF applies to flow rate and head before the lookup — self-contained arithmetic, no new machinery
  - other fanCurve Function1 types (polynomial, csvFile): the shared Function1 evaluator again

</details>
<details><summary><b>[GENUINE]</b> A `sigma` (symmTensor) patch carrying data the component splitter cannot divide — gradient, inletValue, refValue, valueFraction, mapData, normalRef, flowRate or an unsupported Function1 — is refused rather than read and dropped.</summary>

- **citation**: src/finiteVolume/fields/foam_field_reader.cuh:380
- **trigger**: anything but fixedValue / zeroGradient / a constraint type on 0/sigma in a Maxwell (viscoelastic) case
- **of_feature**: the Maxwell stress model's symmTensor boundary fields — TurbulenceModels/.../Maxwell; brae runs sigma as six independent scalar fields through the same machinery
- **path**: host
- **size**: S per entry kind (~30 lines each) if pursued at all
- **fixture**: none
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - a symmTensor-aware patch field family, or per-component splitting of each carried entry — gradient, refValue and valueFraction are componentwise and easy; mapData and flowRate are not meaningful for a stress and should stay refused

</details>
<details><summary><b>[GENUINE]</b> A mesh patch with no matching boundaryField entry — after literal, group and regex matching, and after constraint-type synthesis — is refused.</summary>

- **citation**: src/finiteVolume/fields/geometric_field.cuh:99
- **trigger**: a case whose 0/&lt;field> omits a non-constraint patch. Constraint patches (empty/symmetry/wedge/cyclic/...) are synthesised from the mesh patch type first, mirroring OF's setConstraintTypes, so only real patches fire it.
- **of_feature**: GeometricBoundaryField construction — OF fatals the same way for a missing non-constraint entry
- **path**: host
- **size**: none (permanent)
- **fixture**: tests/test_fields_bc.cu covers the matching path
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing — permanent. The adjacent risk worth an audit is the regex matching itself (compileFoamRegex, last-pattern-wins at geometric_field.cuh:60-87), which SILENTLY selects an entry rather than refusing — that is where a case can get the wrong BC with no throw at all.

</details>
<details><summary><b>[GENUINE]</b> 0/nut carrying more than one nut wall function across wall patches is refused, because brae holds ONE case-wide selector where OpenFOAM dispatches per patch.</summary>

- **citation**: src/applications/solvers/common/turbulence_setup.cuh:420
- **trigger**: two wall patches with different nut wall functions, e.g. nutkWallFunction on one and nutUSpaldingWallFunction on another. This is the highest-leverage BC refusal outside the factory: the winner's kernel rewrites EVERY wall face unconditionally, so before this refusal the last patch in the boundary list decided for all of them, and nutk could never win back (there is no nutk branch and no restoring else).
- **of_feature**: nutWallFunctionFvPatchScalarField::updateCoeffs — TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutWallFunction/nutWallFunctionFvPatchScalarField.C:181-184, which is operator==(calcNut()) on each patch's own object, so every wall may carry a different one and OF honours each
- **path**: both
- **size**: M (~200 lines across the classifier, the DeviceBoundary buffer and four kernels) — and it unlocks two other refusals
- **fixture**: none for the refusal itself; validation/boundary_nut_vs_openfoam.sh gates the single-function path and is the control
- **blocks_tutorial**: none of the six (each uses one nut wall function)
- **depends_on / middle steps**:
  - a per-boundary-face wall-function KIND buffer alongside the existing DeviceBoundary arrays — today NutWall is a single enum (solver_controls.cuh:20)
  - the nut kernels in src/cuda/device_kepsilon.cu (spaldingNutKernel / blendedNutKernel / nutUWallKernel and the nutk path in nut_wall_function.cu) switched from `write where isWall` to `write where isWall && kind == mine`, or merged into one branching kernel
  - host: turbulence_setup.cuh's setNutWall becomes a per-patch classifier rather than one assignment
  - this same per-face dispatch is the prerequisite for alphatJayatilleke and for per-patch atmNutkWallFunction z0, which simpleFoamV2.cu:946 refuses to have two of for exactly the same reason

</details>
<details><summary><b>[OVER-BROAD]</b> A nut/epsilon/omega wall function declared on a patch whose MESH type is not 'wall' is refused, because brae gates the near-wall model on the geometric patch type and the wall function would otherwise be silently inert.</summary>

- **citation**: src/applications/solvers/common/turbulence_setup.cuh:359
- **trigger**: e.g. `type nutkWallFunction;` on a patch declared `type patch;` in constant/polyMesh/boundary. Deliberately conservative: group and regex boundary names are skipped, so it fires only when the entry resolves to a concrete non-wall patch.
- **of_feature**: nutkWallFunctionFvPatchScalarField and epsilon/omegaWallFunction — TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/. OpenFOAM keys the wall treatment on the BOUNDARY CONDITION type, not the patch type, and honours a wall function wherever it is declared; brae keys on the patch type, which is why this must be a refusal rather than a no-op.
- **path**: both
- **size**: M — it changes what drives wall selection, touching near-wall distance, averaging weights and the nut/epsilon/omega kernels
- **fixture**: none
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the fix brae already applied for epsilon and omega: key the wall treatment on the BC type via the existing predicate (fv_patch_field.cuh:118-125 records that mapping omegaWallFunction to a bare ZeroGradient left every kOmegaSST case with no wall faces). Extending that predicate to nut, and to the averaging-weight path, removes the need for the geometric gate.
  - wall distance y on a non-wall patch: brae's near-wall distance is built from wall patches only, so a wall function on a `patch`-type face has no y to use until that is generalised

</details>
<details><summary><b>[DEAD]</b> DEAD REFUSAL: `hasFixedFluxPressure` gates a refusal at six pEqn sites and is set true by nobody in production — the factory maps fixedFluxPressure to zeroGradient (fv_patch_field.cuh:1517) and never raises the flag.</summary>

- **citation**: src/applications/solvers/simpleFoam/pEqn_cpp.cu:27
- **trigger**: nothing in a case can fire it. The six consumers are simpleFoam/pEqn_cpp.cu:27, simpleFoam/pEqn.cu:27, rhoSimpleFoam/rhoPEqn_cpp.cu:36, rhoSimpleFoam/rhoPEqn.cu:146, rhoSimpleFoam/rhoPcEqn_cpp.cu:44, rhoSimpleFoam/rhoPcEqn.cu:132. The only assignments to true are in the six tests that fail-proof it; the two production propagations (rhoSimpleFoam.cu:410, rhoSimpleFoam_cpp.cu:371) copy a field that is always false.
- **of_feature**: fixedFluxPressureFvPatchScalarField + constrainPressure — finiteVolume/cfdTools/general/constrainPressure/constrainPressure.C:60-77 sets the patch snGrad to (phiHbyA_b - rho_b*MRF.relative(Sf&U_b))/(magSf*rhorAU); constrainHbyA.C:56-69 sets HbyA_b = U_b on every patch where U is NOT assignable
- **path**: both
- **size**: S-M (~120 lines) IF the assignable-U case is ever needed; today the honest action is either to delete the six dead throws or to derive the flag so it can actually fire
- **fixture**: tests/test_peqn_cpp.cu:216, test_peqn_cuda.cu:322, test_rho_peqn_cpp.cu:488, test_rho_pceqn_cpp.cu:409, test_rho_peqn_cuda.cu:507, test_rho_pceqn_cuda.cu:443 — six fail-proofs that set the flag by hand and therefore pass while the production path can never reach it
- **blocks_tutorial**: none — the mapping is what lets angledDuctExplicitFixedCoeff, squareBend and squareBendLiqNoNewtonian run at all
- **depends_on / middle steps**:
  - The zeroGradient mapping is EXACT wherever U's patch is non-assignable and p's patch is not fixedFluxExtrapolatedPressure: constrainHbyA then makes phiHbyA_b = Sf&U_b, MRF.relative cancels the frame flux on both sides, ddtCorr's boundary term vanishes at a fixed-value patch, and constrainPressure's numerator is identically zero, so g = 0, which is zeroGradient's snGrad. Measured over every v2412 tutorial pairing them: 160 co-occurrences with U = fixedValue (49), noSlip (43), waveVelocity (27), movingWallVelocity (22), slip (15), uniformFixedValue (4) — all non-assignable, zero counter-examples.
  - To make the refusal live rather than dead it would have to be RAISED where the U patch IS assignable (zeroGradient, calculated, or inletOutlet, which overrides assignable() back to true — inletOutletFvPatchField.H:164). Only that case needs the real gradient path.
  - That path is a per-face refGrad on the p patch recomputed each pressure corrector from phiHbyA_b, rho_b, magSf and rAU. The storage already exists: DeviceBoundary carries refGrad (device_boundary.cuh:36) and MixedPatchField/FixedGradientPatchField carry setRefGrad, so the work is the per-corrector update call.

</details>
<details><summary><b>[DEAD]</b> HOLE / DEAD GUARD: surfaceNormalFixedValue's `refValue` and uniformNormalFixedValue's `uniformValue` route a table/Function1 straight past the unsupportedFunction1 marker, so the patch is built with an EMPTY value array and every face gets U_b = 0*n — a silent zero inlet where the case prescribed a ramp.</summary>

- **citation**: src/finiteVolume/fields/foam_field_reader.cuh:530
- **trigger**: `type surfaceNormalFixedValue; refValue table ((0 0) (1 -10));` or the uniformNormalFixedValue equivalent. The reader's else-branch comments "table/Function1 -> ramp handles it; treat as 0" and calls skipToSemicolon WITHOUT setting p.unsupportedFunction1, so the factory's Function1 refusals (fv_patch_field.cuh:1473/1483/1492) cannot fire for this key. SurfaceNormalFixedValuePatchField::build (fv_patch_field.cuh:333-335) then evaluates `i &lt; vals.size() ? vals[i] : 0` against an empty vector. The class comment at :316 states the contract outright — "any time-`ramp`/Function1 is ignored" — and nothing enforces it.
- **of_feature**: surfaceNormalFixedValueFvPatchVectorField — finiteVolume/fields/fvPatchFields/derived/surfaceNormalFixedValue/surfaceNormalFixedValueFvPatchVectorField.C, where refValue_ is a PatchFunction1 sampled every updateCoeffs and U_b = refValue*nf
- **path**: host
- **size**: XS to close the hole (2 lines); S-M to actually evaluate the Function1
- **fixture**: none — tests/test_uniform_function1.cu is the template and would extend to this key in ~30 lines, with a bare `refValue uniform -10` as the negative control
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - immediate fix, independent of any Function1 work: set p.unsupportedFunction1 in that else-branch (2 lines) so the existing refusal machinery covers the key — that turns a silent wrong answer into a named refusal
  - full support then needs the same Function1 evaluator as uniformFixedValue, plus a per-step refValue push (the patch is a plain FixedValuePatchField today and has no update hook)

</details>
<details><summary><b>[DEAD]</b> HOLE / DEAD GUARD: fixedMean, fanPressure and codedFixedValue/codedMixed are ACCEPTED by the factory on the strength of a per-step driver update that only two of the drivers actually perform — and the refusal meant to catch that (collectFixedMean) is never called on the others.</summary>

- **citation**: src/applications/solvers/common/fan_pressure.cuh:178
- **trigger**: a case with `type fixedMean`, `type fanPressure` or `type codedFixedValue` run through the rhoSimpleFoam OF-mirror driver (rhoSimpleFoam.cu / rhoSimpleFoam_cpp.cu) or simpleFoamV2.cu. collectFanPressure and collectFixedMean are called ONLY from gpuPimpleFoam.cu:461/469, and setupCodedBCs ONLY from gpuPimpleFoam.cu:500 and gpuSimpleFoam.cu:504. On every other driver the patch is built from the file `value` and never updated — a frozen boundary where the case asked for a maintained mean, a fan curve, or a compiled snippet. The factory's comments assert the opposite ("the solver recomputes refValue every step", fv_patch_field.cuh:1497-1505), which is the unhonoured-contract pattern this codebase keeps finding.
- **of_feature**: fixedMeanFvPatchField (finiteVolume/fields/fvPatchFields/derived/fixedMean/), fanPressureFvPatchScalarField (finiteVolume/fields/fvPatchFields/derived/fanPressure/) and codedFixedValueFvPatchField (finiteVolume/fields/fvPatchFields/derived/codedFixedValue/) — all three recompute in updateCoeffs on every solver, not on a chosen subset
- **path**: both
- **size**: S to make the refusals reachable everywhere; M to give the rhoSimpleFoam mirror the per-step hooks
- **fixture**: tests/test_fixed_mean.cu gates the pimpleFoam path only — it would pass unchanged while a rhoSimpleFoam case froze
- **blocks_tutorial**: none today (no rhoSimpleFoam tutorial uses fixedMean or fanPressure) — but squareBendLiq's `coded` massFlowRate is the same class of input reaching the same unwired driver
- **depends_on / middle steps**:
  - cheapest correct step: call collectFixedMean / collectFanPressure and a coded-BC scan from EVERY driver, so drivers that cannot maintain them refuse by name instead of freezing — a wiring change, ~30 lines per driver, and it makes the existing refusal reachable
  - then, to support them on the rhoSimpleFoam mirror path: the per-step hooks must move out of DeviceSimpleSolver (setFanPressure/setFixedMean at device_simple_foam.cuh:565/575, addCodedBC at :532) into something the compressible driver also owns — the same generalisation the fixedMean-on-any-field refusal needs
  - codedFixedValue additionally needs NVRTC wiring on the compressible path: device_coded_bc.cu is target-typed to U/p/k/second, so he, T and rho have no target id yet

</details>

## equations

### Adjudicated verdicts (disputes and missed items)

| # | kind | item | verdict |
|---|---|---|---|
| 1 | dispute | MRF declared, device momentum twin (rhoUEqn.cu) | **HOLE** |
| 2 | dispute | unported fvOption, device momentum twin | **HOLE** |
| 3 | dispute | MRF on the energy equation, device twin | **HOLE** |
| 4 | dispute | unported fvOption on the energy equation, device twin | **HOLE** |
| 5 | dispute | neither muEff nor (rho,nuEff) supplied to divDevRhoReff | **GENUINE** |
| 6 | dispute | bounded on one of div(phi,k)/div(phi,epsilon) only | **GENUINE** |
| 7 | dispute | non-upwind div(phi,k/epsilon), device closure | **HOLE** |
| 8 | dispute | energy convection scheme refusal -- size objection | **GENUINE** |
| 9 | missed | host DarcyForchheimer runs with nu=0 while claiming implementation | **HOLE** |
| 10 | missed | kEpsilon_cpp assembles upwind whatever fvSchemes names | **HOLE** |
| 11 | missed | no host-side coupled-patch refusal in the _cpp mirror | **HOLE** |
| 12 | missed | processor missing from isCoupledInterfaceType (derivation trap) | **GENUINE** |
| 13 | missed | `limited 0` mapped to the FULL non-orthogonal correction | **HOLE** |
| 14 | missed | device step gate never derives the refusal flags | **GENUINE** |
| 15 | missed | live thermo-energy-keyword refusal at createFields | **GENUINE** |

Evidence, one line each (current file:line -- line numbers are as of 2ecfe0f):

1. src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:61 guards `in.hasMRF && !in.mrf`, but rhoSimpleFoam.cu:290 only forwards the flag and no device caller derives it -- tests/test_rho_simple_step_cuda.cu:370 sets only gin.hasMixed -- so an MRF case run through the device gate proceeds rotation-free with both arms agreeing; the checker's 'armed, identical wiring to the host' is false (the host harness derives it at tests/test_rho_simple_step_cpp.cu:221, the device harness derives nothing), and the surveyor's 'harmless' is false for the same reason.
2. src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:70 guard; the device step gate never reads the case's fvOptions dict -- its only porosity is the synthetic wiring control at tests/test_rho_simple_step_cuda.cu:738-757 -- so hasFvOptions is never derived and a case with any fvOption runs with it silently absent on both arms of a green gate.
3. src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu:148 guards `in.hasMRF`, forwarded at rhoSimpleFoam.cu:366 from a RhoStepInput field no device caller assigns; EEqn.H's fvc::div(MRF.phi(),p) is real and an MRF case on the device path drops it silently rather than being refused.
4. src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu:154 guard, forwarded at rhoSimpleFoam.cu:367 but derived by no device caller; the device energy path also implements no fvOptions.constrain at all, so e.g. angledDuctExplicitFixedCoeff's fixedTemperatureConstraint is dropped silently on CUDA while the host applies it -- the guard that should catch this never fires.
5. src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:131-140 and rhoUEqn_cpp.cu:100-108; the checker's 'it fires today' is factually wrong -- rhoSimpleFoam.cu:279 fills muEffCell/muEffBndFace (hard-required at :239-242) so haveMu holds and the guard never fires -- but the class correction is right: it is the same null-contract species as the structural-guards entry already GENUINE (whose own citations rhoUEqn_cpp.cu:117,129,134 sit beside it), guarding a null deref and unchecked kernel reads for any future or injecting caller.
6. src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:460-467 refuses exactly the configuration its oracle cannot express: kEpsilon_cpp.cu:164 carries ONE `bool bounded` while the device applies the split at kEpsilon.cu:671 (boundedEps) and :700 (boundedK); honouring the split would run an arm no gate can validate, and under this project's rules an unvalidated arm is not a lift -- the refusal matches the gap because the oracle is part of the port. Checker upheld; lifting means splitting the host bool, not dropping the guard.
7. src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:442-451 has the right refusal text but the flag is set only by the fail-proof (tests/test_rho_kepsilon_cuda.cu:492) -- no driver anywhere parses div(phi,k)/div(phi,epsilon) -- and the substitution it names is live and measured: squareBend asks `Gauss limitedLinear 1`, gets upwind, worth ~1.6e-03 at convergence, absorbed by the widened TURB_BOUND 3.0e-3 (tests/test_rho_simple_step_cpp.cu:456-470). Not DEAD (not harmless) and not GENUINE (not armed): a guard that fails to fire over a measured silent substitution.
8. Classification confirmed and unchanged: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu:41-53 and rhoEEqn.cu:139-147 refuse anything beyond upwind/linearUpwind and fire on validation/rhoLU; the checker's size shrink is factually supported (OF LimitFuncs.H:101 specialises magSqr<scalar> to the identity, gradHe already computed at rhoEEqn_cpp.cu:217 and :272, limitedLinearWeights exists at src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSchemes_cpp.cu:40) but it corrects metadata, not the class.
9. src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu:223 passes /*nu=*/0.0 with forceDimensions=true into cpu::fvOptions::addSup, whose DF branch is `cd[k] = nu * dd[k] + magU * ff[k]` (src/finiteVolume/cfdTools/general/fvOptions/fvOptions_cpp.cu:339) -- against OF's Cd = mu[celli]*D + (rho[celli]*mag(U))*F (DarcyForchheimerTemplates.C:52-53) the whole Darcy term is dropped and rho is missing from Forchheimer; worse, the host refusal message at rhoUEqn_cpp.cu:98-99 now reads 'explicitPorositySource (DarcyForchheimer and fixedCoeff) IS implemented'. The device refuses this exact branch at rhoUEqn.cu:151-159. Checker confirmed.
10. src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu contains zero throws and takes no div-scheme parameter; the epsilon and k convection at :317 and :486 are the plain fvm::div overload where the weighted one exists and the momentum path uses it; the cost is measured and absorbed in the gate comment at tests/test_rho_simple_step_cpp.cu:467-470 (~1.6e-03 on squareBend), and the legacy device closure honours limitedLinear (src/cuda/device_kepsilon.cuh:295-297), making the OF-mirror reference the more permissive of the two. Checker confirmed.
11. src/finiteVolume/fields/fv_patch_field.cuh:1796-1798 builds any isCoupledInterfaceType patch as ZeroGradientPatchField (the comment defends it only for the legacy device solver, which appends those faces to the LDU); grep finds no cyclic/coupled throw in rhoUEqn_cpp.cu, rhoEEqn_cpp.cu or rhoCreateFields_cpp.cu (only the setConstraintTypes comment at rhoCreateFields_cpp.cu:449-452), while the device lineage refuses upstream at rhoCreateFields.cu:67-74 -- so the host reference silently solves a periodic mesh with zero-gradient walls. Checker confirmed.
12. src/OpenFOAM/db/foam_dict.cuh:56-59 covers only cyclic/cyclicAMI/cyclicACMI/cyclicPeriodicAMI; src/finiteVolume/cfdTools/general/MRF/MRF_cpp.cu:29 ORs in `processor` by hand, and both the device refusal text (rhoUEqn.cu:121-128) and device createFields (rhoCreateFields.cu:67) promise or handle processor -- a future hasCoupledPatches derivation built on the helper alone would under-report exactly as the checker warns. Verified trap, correctly flagged.
13. src/applications/solvers/simpleFoam/simpleFoamV2.cu:365-368 maps `limited 0` to limitCoeff=0.0 with corrected left true, and fvm.cuh:145 treats only limitCoeff in (0,1) as limited, so brae runs the FULL uncapped correction; OF's limitedSnGrad.C:59-71 gives limiter = min(k*|snGrad|/((1-k)*|corr|+SMALL),1) = 0 at k=0, i.e. NO correction -- opposite behaviour, silent. The file even contradicts itself: simpleFoamV2.cu:308-309 asserts '0 means no limiter (plain corrected)' while :318-321 correctly states 'limited 0 is uncorrected'. One-line fix as the checker says; must land before the STALE energy refusal is lifted.
14. tests/test_rho_simple_step_cuda.cu:370 sets only gin.hasMixed while the host twin derives hasFvOptions/hasMRF from the case at tests/test_rho_simple_step_cpp.cu:192-223; the device driver's forwarding (rhoSimpleFoam.cu:290-293, 366-368, 408-411) is exercised only by module fail-proofs (test_rho_ueqn_cuda.cu:507-508, test_rho_eeqn_cuda.cu:368-369, test_rho_pceqn_cuda.cu:441-442). Verified gate gap and the root cause of the four device-twin HOLE verdicts; the test's own porosity control comment (:729-733) records this exact defect class already shipping once.
15. src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:266-274 refuses any thermophysicalProperties `energy` other than sensibleEnthalpy/sensibleInternalEnergy, mirroring OF's thermo.validate(args.executable(), "h", "e") (openfoam2412 applications/solvers/compressible/rhoSimpleFoam/createFields.H:8); it is the sole producer of f.heName and the reachable refusal that makes rhoEEqn_cpp.cu:19-24 DEAD -- the surveyor listed only the dead twin and missed the live one.

### The original survey for this area (as surveyed; see the verdicts above for corrections)

<details><summary><b>[GENUINE]</b> MRF declared by the case but no resolved zones handed to the momentum assembly (host reference)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu:88
- **trigger**: constant/MRFProperties exists; tests/test_rho_simple_step_cpp.cu:221 sets in.hasMRF = has("constant/MRFProperties") and nothing ever assigns uin.mrf, so the guard `in.hasMRF && !in.mrf` fires on every compressible MRF case
- **of_feature**: MRFZoneList::correctBoundaryVelocity + MRFZoneList::DDt(rho,U) -- src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C:210-217 (`return rho*DDt(U)`), applied at applications/solvers/compressible/rhoSimpleFoam/UEqn.H:3 and :8
- **path**: both
- **size**: M -- the arithmetic exists; this is reader + wiring + one new term (MRF.phi) + a fixture
- **fixture**: none compressible. rotCylTurb (simpleFoam kEpsilon + MRFProperties) is the closest and would have to be re-cast as a rhoSimpleFoam case, or an MRF block added to validation/rhoKE
- **blocks_tutorial**: none -- no OpenFOAM rhoSimpleFoam tutorial ships MRFProperties (find over tutorials/compressible/rhoSimpleFoam returns only aerofoilNACA0012/system/fvOptions and angledDuctExplicitFixedCoeff/constant/fvOptions)
- **depends_on / middle steps**:
  - the DDt term itself is ALREADY written (rhoUEqn_cpp.cu:226-245, rho-weighted addCoriolis) -- nothing to port there
  - a constant/MRFProperties reader on the compressible createFields path: rhoCreateFields_cpp.cu contains no MRF code at all (the incompressible readMRFProperties lives in the legacy gpuRhoSimpleFoam.cu:412)
  - cpu::MRF::ZoneSpec -> resolve against the mesh (MRF_cpp.cuh:60-96) -- exists, just not called from this lineage
  - cpu::MRF::correctBoundaryVelocity(U) called BEFORE the boundary snapshot (the pattern is simpleFoamV2.cu:652-674)
  - MRF.makeRelative on a MASS flux for pEqn (MRFZone::makeRelativeRhoFlux), separately refused at rhoPEqn.cu:131
  - EEqn's fvc::div(MRF.phi(), p) -- needs MRFZoneList::phi(), i.e. makeAbsolute onto a zero flux; brae carries frameFluxInt/frameFluxBnd (device_MRF.cuh:31-32) but no makeAbsolute
  - a COMPRESSIBLE MRF fixture: validation/mrfBox, mrfBox2, rotCylTurb and rotorDuct are all simpleFoam

</details>
<details><summary><b>[DEAD]</b> MRF declared but no zones handed to the momentum assembly (device twin)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:63
- **trigger**: `in.hasMRF && !in.mrf`. RhoStepInput::hasMRF is declared at rhoSimpleFoam.cuh:219 and forwarded at rhoSimpleFoam.cu:291, but no caller assigns it -- tests/test_rho_simple_step_cuda.cu never touches it. The only assignment in the tree is tests/test_rho_ueqn_cuda.cu:507, a refusal control. So on the device path an MRF case runs with no rotation and reports nothing -- the exact failure the comment at rhoUEqn.cu:52 says has already shipped once
- **of_feature**: MRFZoneList::DDt(rho,U) -- src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C:210-217; UEqn.H:8
- **path**: device
- **size**: S to make the refusal live (derive hasMRF in the device driver); M for the feature
- **fixture**: none
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the same list as the host entry
  - plus: the device driver must derive hasMRF from the case, which it does for nothing at all today

</details>
<details><summary><b>[GENUINE]</b> An fvOptions type the port does not implement, on the momentum equation (host reference)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu:93
- **trigger**: cpu::fvOptions::read returns an Option with a non-empty `unsupported`, or the harness sees an unrecognised `type` -- tests/test_rho_simple_step_cpp.cu:192-218. IMPLEMENTED and therefore NOT refused: explicitPorositySource/fixedCoeff (fvOptions_cpp.cu:214-245), explicitPorositySource/DarcyForchheimer (:252-276), fixedTemperatureConstraint mode uniform (:172-183), scalarFixedValueConstraint (:186-196), rotorDiskSource (:137-141), actuationDiskSource Froude (:148-152), limitTemperature (as fvOptions.correct(he), rhoSimpleFoam_cpp.cuh:157). REFUSED BY NAME: every other type -- ofscan counts 46 fv::option implementations (fvOptions_cpp.cuh:18) -- plus fixedTemperatureConstraint mode `lookup`, scalarFixedValueConstraint without fieldValues, any option whose selectionMode brae cannot resolve, and actuationDiskSource with variableScaling or a Function1 Cp/Ct
- **of_feature**: fv::optionList's three hooks -- src/finiteVolume/cfdTools/general/fvOptions/fvOptionList.C, applied at rhoSimpleFoam/UEqn.H:11 (`== fvOptions(rho,U)`), :17 (constrain) and :21 (correct)
- **path**: both
- **size**: S per simple source; M for anything needing constrain() on a vector matrix
- **fixture**: validation/simpleCarPorous and OF's angledDuctExplicitFixedCoeff cover the porosity side; no fixture exists for a refused type -- the refusal control is tests/test_rho_ueqn_cpp.cu:1076
- **blocks_tutorial**: none -- aerofoilNACA0012's only fvOption is limitTemperature (implemented) and angledDuctExplicitFixedCoeff's are explicitPorositySource/fixedCoeff + fixedTemperatureConstraint (both implemented)
- **depends_on / middle steps**:
  - per option type, separately -- the framework (dictionary, cellSetOption selection, the three hooks) is shared and already ported
  - the momentum path implements only hook 1 (addSup); fvOptions.constrain(UEqn) (setValues on a VECTOR matrix) and fvOptions.correct(U) are unimplemented for every option, so any constraint naming U is refused whatever its type
  - the highest-value next types for rhoSimpleFoam are meanVelocityForce, buoyancyForce/buoyancyEnergy, limitVelocity and the semiImplicitSource family; each needs its own OF transcription plus a gate

</details>
<details><summary><b>[DEAD]</b> An unported fvOptions type on the momentum equation (device twin)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:72
- **trigger**: RhoStepInput::hasFvOptions (rhoSimpleFoam.cuh:219) is forwarded at rhoSimpleFoam.cu:291 but derived by no caller; tests/test_rho_simple_step_cuda.cu never sets it. Only tests/test_rho_ueqn_cuda.cu:508 sets it, as a control
- **of_feature**: fv::optionList -- src/finiteVolume/cfdTools/general/fvOptions/fvOptionList.C; UEqn.H:11,17,21
- **path**: device
- **size**: S
- **fixture**: tests/test_rho_ueqn_cuda.cu:508 (control only)
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the device driver deriving hasFvOptions/fvOptionUnsupported from cpu::fvOptions::read, exactly as tests/test_rho_simple_step_cpp.cu:200-218 does on the host

</details>
<details><summary><b>[DEAD]</b> A mesh with a cyclic / cyclicAMI / cyclicACMI / processor patch, refused by the momentum assembly</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:123
- **trigger**: `in.hasCoupledPatches`. rhoUEqn.cuh:165 says "the DRIVER says it, from isCoupledInterfaceType over the patch list" -- no driver does; the flag is set only by tests/test_rho_ueqn_cuda.cu:509. It is also redundant: createDeviceFields already refuses the whole case at rhoCreateFields.cu:69 by walking the patch list itself, so no device assembly can ever be reached with a coupled patch present
- **of_feature**: cyclicFvPatchField / processorFvPatchField coupled interfaces -- src/finiteVolume/fields/fvPatchFields/constraint/cyclic/cyclicFvPatchField.C; OpenFOAM carries them as lduInterfaceFields on the matrix, brae's buildDeviceMesh keeps them out of the LDU (device_mesh.cuh:34-38, 110-123)
- **path**: both
- **size**: L for the feature; S to add the missing HOST refusal
- **fixture**: validation/cyclicChannel, pipeCyclic, pipeCyclicAMI exist but are all incompressible
- **blocks_tutorial**: none of the five rhoSimpleFoam tutorials uses a coupled patch
- **depends_on / middle steps**:
  - THE HOST SIDE IS THE HOLE, NOT THIS ONE: rhoUEqn_cpp/rhoEEqn_cpp have no coupled-patch refusal and fv_patch_field.cuh:1692 maps every coupled type to ZeroGradientPatchField, so the host path silently solves a different equation on a periodic mesh
  - to lift it for real: interface coefficients passed through divDevRhoReff (its cyc/ami hooks already exist, device_divdevreff.cuh:35-37), through the corrected-laplacian source, through the off-diagonals, and into deviceRelaxDiag's sumOff

</details>
<details><summary><b>[GENUINE]</b> explicitPorositySource with the DarcyForchheimer model, on the device momentum path (the host has NO equivalent refusal and silently computes the wrong resistance)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:153
- **trigger**: `in.porosity->active && !in.porosity->fixed` -- a case whose explicitPorositySourceCoeffs name `type DarcyForchheimer` rather than `fixedCoeff`
- **of_feature**: porosityModels::DarcyForchheimer -- src/finiteVolume/cfdTools/general/porosityModel/DarcyForchheimer/DarcyForchheimerTemplates.C:52-53, Cd = mu[celli]*D + (rho[celli]*mag(U))*F, reached through explicitPorositySource::addSup (`eqn -= porosityEqn`)
- **path**: both
- **size**: S once mu_lam and rho reach DevicePorosity
- **fixture**: none -- validation/simpleCarPorous and OF's angledDuctExplicitFixedCoeff are both fixedCoeff, so this branch is covered by no gate at all
- **blocks_tutorial**: none (angledDuctExplicitFixedCoeff is fixedCoeff)
- **depends_on / middle steps**:
  - DevicePorosity gaining a per-cell LAMINAR dynamic viscosity field and a per-cell rho -- it currently takes ONE scalar viscosity and no density (device_fvoptions.cuh)
  - RhoMomentumInput carrying mu_lam or a kinematic nu; it carries neither, only muEff (turbulent)
  - THE HOST IS WORSE AND SILENT: rhoUEqn_cpp.cu:223 passes nu = 0.0 into the shared addSup, which on the DarcyForchheimer branch drops the whole viscous Darcy term AND the rho on the Forchheimer term. The device refuses what the host quietly gets wrong -- so the host needs a refusal or a fix before the device one can be lifted against a trustworthy oracle

</details>
<details><summary><b>[DEAD]</b> Neither a dynamic muEff nor an (rho, nuEff) pair was supplied to divDevRhoReff</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu:103 (device twin rhoUEqn.cu:135)
- **trigger**: `!haveMu && !haveRho`. Both drivers always supply both: rhoSimpleFoam_cpp.cu:262-263 sets rho/rhoBnd AND muEff/muEffBnd, and rhoSimpleFoam.cu:281-283 does the same, with an earlier hard requirement at rhoSimpleFoam.cu:241 that muEff/alphaEff be present
- **of_feature**: linearViscousStress::divDevRhoReff -- src/TurbulenceModels/turbulenceModels/linearViscousStress/linearViscousStress.C:107-117, `- fvc::div((alpha*rho*nuEff)*dev2(T(grad U))) - fvm::laplacian(alpha*rho*nuEff, U)`
- **path**: both
- **size**: n/a
- **fixture**: tests/test_rho_ueqn_cpp.cu exercises the muEff-injection path
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing -- it is an argument contract, not a feature

</details>
<details><summary><b>[DEAD]</b> MRF.DDt(rho,U) asked for while only a pre-formed muEff was supplied, so rho cannot be recovered</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu:234 (device twin rhoUEqn.cu:590)
- **trigger**: `in.mrf && !in.rho`. Unreachable twice over: no driver ever sets `mrf` (see the MRF entries), and both drivers set rho alongside muEff
- **of_feature**: MRFZoneList::DDt(rho,U) -- MRFZoneList.C:210-217
- **path**: both
- **size**: n/a
- **fixture**: none
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing -- it becomes live only if the MRF wiring lands and a caller then uses the gate's muEff-injection path

</details>
<details><summary><b>[GENUINE]</b> Structural guards on the momentum inputs: phi null, phi with the wrong internal/boundary face count, mu_eff with the wrong cell or boundary-face count, rho/nuEff length mismatch</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:91,98,104,190,195,211,217 and rhoUEqn_cpp.cu:117,129,134
- **trigger**: A caller passing a short or absent array. These are not feature refusals -- they stand in for CUDA's absence of bounds checking: divFaceKernel launches nBlocks(dm.nInternalFaces) and reads phi[f] over that whole range regardless of phiInt.size() (device_fvm.cu:538), and deviceAxpy takes its trip count from the SOURCE buffer without resizing the destination (blas1.cu:164-169), so a length mismatch is an out-of-bounds WRITE that produces a plausible-looking matrix
- **of_feature**: compressibleCreatePhi.H (phi is rho*(U & Sf), kg/s) and linearViscousStress.C:107-117; the boundary-viscosity guard mirrors effectiveFaceViscosity's rule that a face value is the PATCH value, not the owner cell's
- **path**: both
- **size**: n/a
- **fixture**: tests/test_rho_ueqn_cuda.cu
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing -- keep them

</details>
<details><summary><b>[DEAD]</b> The energy variable is neither "e" nor "h"</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu:20
- **trigger**: `in.heName != "e" && != "h"`. rhoCreateFields_cpp.cu:249-255 already refuses any thermophysicalProperties `energy` entry other than sensibleEnthalpy/sensibleInternalEnergy and is the only producer of f.heName, so this second gate cannot see anything else. The device twin does not carry it at all -- it takes a required `isE` bool instead (rhoEEqn.cuh:88)
- **of_feature**: basicThermo::validate(..., "h", "e") and the he.name() branch at rhoSimpleFoam/EEqn.H:8-11
- **path**: host
- **size**: n/a
- **fixture**: tests/test_rho_eeqn_cpp.cu
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing; if absoluteEnthalpy / absoluteInternalEnergy are ever wanted, the work is in the thermo, not here

</details>
<details><summary><b>[GENUINE]</b> MRF on the energy equation -- EEqn += fvc::div(MRF.phi(), p) (host reference)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu:31
- **trigger**: `in.hasMRF`, set from constant/MRFProperties by tests/test_rho_simple_step_cpp.cu:221. Note this one is unconditional -- unlike the momentum guard it has no "but zones were supplied" escape, because the term is genuinely unwritten
- **of_feature**: MRFZoneList::phi() -- src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C:220-236, a zero surfaceScalarField with mrf.makeAbsolute(phi) applied per zone, i.e. the VOLUMETRIC frame flux; added at rhoSimpleFoam/EEqn.H:17-20
- **path**: both
- **size**: S once MRF.phi() exists; the term itself is one explicit divergence
- **fixture**: none compressible
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - MRFZone::makeAbsolute -- brae has makeRelative (MRF_cpp.cuh:96) and the device frameFluxInt/frameFluxBnd (device_MRF.cuh:31-32) but no makeAbsolute, and MRF.phi() also needs the INCLUDED faces set to the frame flux, which makeRelative zeroes instead
  - an explicit fvc::div(phiMRF, p) into the energy source -- the operator exists (explicitConvectionDivExtensive, rhoEEqn_cpp.cu:64-131), the field does not
  - the whole momentum MRF wiring above, since the flag is shared

</details>
<details><summary><b>[DEAD]</b> MRF on the energy equation (device twin)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu:150
- **trigger**: RhoEnergyInput::hasMRF (rhoEEqn.cuh:109) is forwarded at rhoSimpleFoam.cu:367 from RhoStepInput::hasMRF, which no device caller sets
- **of_feature**: MRFZoneList::phi() -- MRFZoneList.C:220-236; EEqn.H:17-20
- **path**: device
- **size**: S to make live
- **fixture**: tests/test_rho_eeqn_cuda.cu:368 (control only)
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the device driver deriving hasMRF at all

</details>
<details><summary><b>[GENUINE]</b> An unported fvOptions type on the energy equation (host reference)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu:35
- **trigger**: `in.hasFvOptions`, which the harness sets TRUE only for a type cpu::fvOptions::read could not implement. Note what is NOT refused: the host driver applies fvOptions.constrain(EEqn) itself at rhoSimpleFoam_cpp.cu:310-318, passing an he(p,T) conversion, so fixedTemperatureConstraint and scalarFixedValueConstraint are live, and limitTemperature runs as fvOptions.correct(he) at :332-336
- **of_feature**: fv::optionList applied at rhoSimpleFoam/EEqn.H:14 (`== fvOptions(rho,he)`), :24 (constrain) and :28 (correct)
- **path**: host
- **size**: S per type
- **fixture**: OF's angledDuctExplicitFixedCoeff (fixedTemperatureConstraint) and aerofoilNACA0012 (limitTemperature) cover the implemented side
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - per option type; the energy-side hooks (addSup into a scalar matrix, constrain via setValues, correct after the solve) all already exist for the implemented types
  - the message names the type, which is what makes the next step readable

</details>
<details><summary><b>[DEAD]</b> An unported fvOptions type on the energy equation (device twin)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu:156
- **trigger**: Never set (RhoStepInput::hasFvOptions is derived by no device caller). The consequence is concrete: the device energy path implements NO fvOptions at all -- rhoSimpleFoam.cu has no fvOptions.constrain(EEqn) anywhere, only the limitTemperature clamp at :385 -- so angledDuctExplicitFixedCoeff's fixedTemperatureConstraint on the porosity cellZone is silently dropped on the CUDA path while the host applies it
- **of_feature**: fv::optionList::constrain -> fvMatrix::setValues -- src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C:259-291; EEqn.H:24
- **path**: device
- **size**: S for the refusal, M for device constrain()
- **fixture**: angledDuctExplicitFixedCoeff, via tests/rho_angledduct_vs_openfoam.sh -- host arm only today
- **blocks_tutorial**: angledDuctExplicitFixedCoeff on the device path (silently, which is the defect)
- **depends_on / middle steps**:
  - a device setValues on a scalar matrix -- the device kEpsilon module already has the shape of it (fvoEpsMask/fvoEpsVal, kEpsilon.cuh:152-155), so the kernel pattern exists
  - the he(p,Tuniform) conversion on device (rhoThermoDevice.cu has the thermo)
  - the device driver deriving hasFvOptions so the refusal covers what is not built

</details>
<details><summary><b>[STALE]</b> A `limited &lt;k> corrected` laplacian on the energy equation</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu:46
- **trigger**: `in.snGradLimitCoeff != 0.0`, i.e. any case whose laplacianSchemes say `Gauss linear limited 0.33` rather than `corrected`. The refusal's own text -- "brae's laplacian here implements corrected (uncapped)" -- is no longer true: fvm::laplacianNonOrthSource takes a limitCoeff (src/finiteVolume/finiteVolume/fvm.cuh:187) and applies OpenFOAM's limiter inside laplacianCorrFlux (fvm.cuh:143-153, min(k*|orth|/((1-k)*|corr|+SMALL),1) against nonOrthDeltaCoeffs), and rhoEEqn_cpp.cu:273-274 ALREADY passes in.snGradLimitCoeff into it -- code the throw thirty lines above makes unreachable. The device twin has no such refusal and runs the limited form at rhoEEqn.cu:324-329 via deviceLaplacianCorrFluxLimited, so the reference is the more restrictive of the pair it is meant to be the oracle for. The same stale pair sits in the pressure equations at rhoPEqn_cpp.cu:41 (vs :285) and rhoPcEqn_cpp.cu:49 (vs :270)
- **of_feature**: fv::limitedSnGrad&lt;Type>::correction -- src/finiteVolume/finiteVolume/snGradSchemes/limitedSnGrad/limitedSnGrad.C:49-71; the brae formula matches it term for term
- **path**: host
- **size**: S
- **fixture**: none names `limited &lt;k> corrected`; all five OF rhoSimpleFoam tutorials say `Gauss linear corrected`
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing to build. Delete the throw, then decide the two edge cases the current guard hides: OpenFOAM treats k >= 1 as `corrected` and k &lt;= 0 as `uncorrected`, while brae's `limited = (limitCoeff > 0 && limitCoeff &lt; 1)` silently runs FULL corrected at k = 0
  - a gate: no checked-in case names `limited`, so lifting it needs a fixture built for it (copy validation/rhoKE and change laplacianSchemes) plus a control that shows the limiter actually bites -- a near-orthogonal mesh would read identical either way

</details>
<details><summary><b>[GENUINE]</b> A convection scheme other than `Gauss upwind` or `Gauss linearUpwind &lt;grad>` on div(phi,he) or div(phi,Ekp|K)</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu:52 and the device twin src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu:141
- **trigger**: schemeHe or schemeKE resolving to anything else -- in practice `bounded Gauss limitedLinear 1`. The device guard covers BOTH terms only since the fix recorded at rhoEEqn.cu:119-135; before that div(phi,Ekp|K) fell through to upwind with no throw
- **of_feature**: gaussConvectionScheme + limitedSurfaceInterpolationScheme -- src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedLinear/limitedLinear.C and LimitedScheme.H; reached from rhoSimpleFoam/EEqn.H:6 and :9-10
- **path**: both
- **size**: S -- one weighted explicit divergence, host and device, reusing the existing weight functions
- **fixture**: validation/rhoLU names `bounded Gauss limitedLinear 1` on all four of div(phi,h|e|K|Ekp) and is the case this refusal fires on today
- **blocks_tutorial**: gasMixing/injectorPipe (div(phi,e|Ekp|h|K) Gauss limitedLinear 1); squareBendLiq and squareBendLiqNoNewtonian name `bounded Gauss linearUpwind limited`, which IS ported
- **depends_on / middle steps**:
  - a WEIGHTED implicit scalar fvm::div -- already exists and is already used by the momentum equation (fvm::div(phi, phiBnd, U, weights, ...) with limitedSchemes::limitedLinearWeights, rhoUEqn_cpp.cu:39-63)
  - a WEIGHTED EXPLICIT divergence for the kinetic-energy term -- does NOT exist: explicitConvectionDivExtensive (rhoEEqn_cpp.cu:64-131) hardcodes the upwind face value and offers only a linearUpwind deferred correction on top
  - on device: deviceDivLimitedCoeffs already exists (used at rhoUEqn.cu:311-341) so the implicit half is free; the explicit weighted div for Ekp|K is the new kernel
  - the limiter field here is `he` itself (limitFuncs::null, a scalar), so unlike the momentum vector case no magSqr shim is needed -- this is the SIMPLER instantiation of machinery that already runs

</details>
<details><summary><b>[DEAD]</b> A mesh with coupled patches, refused by the energy assembly</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu:164
- **trigger**: `in.hasCoupledPatches`, set by no driver and made unreachable in any case by rhoCreateFields.cu:69, which refuses the case before the device field set is built
- **of_feature**: cyclicFvPatchField / processorFvPatchField as lduInterfaceFields -- src/finiteVolume/fields/fvPatchFields/constraint/cyclic/cyclicFvPatchField.C
- **path**: device
- **size**: n/a
- **fixture**: tests/test_rho_eeqn_cuda.cu:370 (control only)
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - same as the momentum coupled-patch entry; the missing piece is the HOST refusal, not this one

</details>
<details><summary><b>[GENUINE]</b> Required energy inputs absent: phi, alphaEff (cells and boundary faces), or U/p/rho on cells and boundary faces</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoEEqn_cpp.cu:26,28 and rhoEEqn.cu:171,179
- **trigger**: A caller half-supplying the equation's inputs. The p/rho half is deliberately demanded even on the `h` branch, which does not read them -- so a caller cannot half-supply the inputs of an equation it selected. The alphaEff boundary array is required for the same reason the momentum muEff one is: on a wall carrying compressible::alphatWallFunction the face value differs from the owner cell's by the whole of alphat
- **of_feature**: fvm::laplacian(turbulence->alphaEff(), he) -- rhoSimpleFoam/EEqn.H:12; alphaEff = CpByCpv*(alpha + alphat) from src/TurbulenceModels/compressible/EddyDiffusivity/EddyDiffusivity.C
- **path**: both
- **size**: n/a
- **fixture**: tests/test_rho_eeqn_cpp.cu, tests/test_rho_eeqn_cuda.cu
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing -- keep them

</details>
<details><summary><b>[GENUINE]</b> A RAS model other than kEpsilon or kOmegaSST, refused at field construction -- the LIVE model refusal</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:491
- **trigger**: constant/momentumTransport or constant/turbulenceProperties with simulationType RAS, turbulence on, and RASModel naming anything else. It fires before any equation is assembled, which is why the closure-level realizable/RNG guard below can never be reached
- **of_feature**: RASModel::New over the runtime selection table -- src/TurbulenceModels/turbulenceModels/RAS/RASModel/RASModel.C; OpenFOAM v2412 ships EBRSM, kEpsilonPhitF, kOmega, kOmegaSSTSAS, LaunderSharmaKE, LRR, realizableKE, RNGkEpsilon, SpalartAllmaras, SSG and kOmegaSSTLM in addition to the two brae has
- **path**: both
- **size**: M per model that has a _cpp reference; L for RNGkEpsilon or anything without one
- **fixture**: validation/pitzDailyRKE, rke_of/rke_cf and validation/T3A cover realizableKE and kOmegaSSTLM incompressibly; no compressible fixture exists for any refused model
- **blocks_tutorial**: none -- all five OF rhoSimpleFoam tutorials are kEpsilon or kOmegaSST
- **depends_on / middle steps**:
  - realizableKE is the nearest: realizableKE_cpp.cu (238 lines) already exists in the OF-mirror tree with the variable-Cmu rCmu, S2 = 2*magSqr(devSymm(gradU)) and the strain-based epsilon equation; what is missing is a COMPRESSIBLE instantiation (a kEpsilonRef::Compressible-shaped struct carrying rho, nu, phiByRho, alphat, Prt), a device twin, and two lines of wiring at rhoCreateFields_cpp.cu:490 and rhoSimpleFoam_cpp.cu:584
  - SpalartAllmaras_cpp.cu (280 lines) exists but is incompressible-shaped and needs the same compressible instantiation plus a wall-distance field on this path (cellWallDist is already called for the kOmegaSST arm at rhoSimpleFoam_cpp.cu:562)
  - kOmegaSSTLM_cpp.cu (580 lines) exists, unwired
  - RNGkEpsilon has NO _cpp reference at all -- only the legacy fused kernels (src/cuda/device_kepsilon.cu:560 takes co.rng, eta0, beta). It is the model this lineage would have to transcribe from scratch, and the R = eta*(1-eta/eta0)/(1+beta*eta^3) term applies to the G production alone, not to the SuSp divU term
  - every model also needs its own nut wall function dispatch (see the nut entry below)

</details>
<details><summary><b>[DEAD]</b> A RAS model other than kEpsilon reaching the SIMPLE step's turbulence branch</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam_cpp.cu:586
- **trigger**: `f.rasModel != "kEpsilon"` after the kOmegaSST arm has returned. rhoCreateFields_cpp.cu:491 is the only producer of f.rasModel and already admits exactly {kEpsilon, kOmegaSST}, so this can only fire if a test hand-builds the field set. Worth keeping as insurance -- its own comment records that this was the site where every non-SST model used to fall through into kEpsilon arithmetic driven by another model's substituted coefficients
- **of_feature**: RASModel::New -- src/TurbulenceModels/turbulenceModels/RAS/RASModel/RASModel.C
- **path**: host
- **size**: n/a
- **fixture**: none
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing; it becomes live the moment a third model is admitted at createFields

</details>
<details><summary><b>[DEAD]</b> realizableKE or RNGkEpsilon selected, refused inside the device kEpsilon closure</summary>

- **citation**: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:417
- **trigger**: `in.co.realizable || in.co.rng`. On the OF-mirror path nothing ever sets those bits: rhoCreateFields_cpp.cu:468-486 reads only Cmu/C1/C2/C3/sigmak/sigmaEps into f.keCoeffs and never touches realizable/rng. The only setter in the tree is applications/solvers/common/turbulence_setup.cuh:263,285, which belongs to the LEGACY solver lineage. The case is already stopped upstream at rhoCreateFields_cpp.cu:491
- **of_feature**: RAS::realizableKE -- src/TurbulenceModels/turbulenceModels/RAS/realizableKE/realizableKE.C; RAS::RNGkEpsilon -- src/TurbulenceModels/turbulenceModels/RAS/RNGkEpsilon/RNGkEpsilon.C
- **path**: device
- **size**: n/a as a refusal
- **fixture**: tests/test_rho_kepsilon_cuda.cu:476,480 (controls only)
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - everything in the RASModel entry above
  - the danger this guard describes is real and not hypothetical: if the OF-mirror createFields ever learns to read realizableKECoeffs/RNGkEpsilonCoeffs into the SHARED KEpsilonCoeffs struct before a model exists, standard-kEpsilon arithmetic would run on another model's constants -- a closure that exists in no source and that a brae-vs-brae gate could not detect

</details>
<details><summary><b>[GENUINE]</b> A nut wall function other than nutkWallFunction</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:533
- **trigger**: Any patch in 0/nut whose type starts with "nut" and is not nutkWallFunction. Refused HERE because this is the last place the dictionary type still exists -- buildField discards it, which is why neither the host closure nor the device one could have checked
- **of_feature**: nutWallFunctionFvPatchScalarField::updateCoeffs -> the virtual calcNut() -- src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nutWallFunctions/nutWallFunction/nutWallFunctionFvPatchScalarField.C:181-184. OpenFOAM ships nutkWallFunction, nutkRoughWallFunction, nutLowReWallFunction, nutUWallFunction, nutUBlendedWallFunction, nutURoughWallFunction, nutUSpaldingWallFunction and nutUTabulatedWallFunction; epsilonWallFunction then READS the result (epsilonWallFunctionFvPatchScalarField.C:333-334, `const tmp&lt;scalarField> tnutw = turbModel.nut(patchi)`) rather than recomputing it
- **path**: both
- **size**: M for the dispatch + nutLowRe; M again for the nutU family
- **fixture**: none compressible -- validation/saf_* cover nutUSpalding incompressibly through the legacy path
- **blocks_tutorial**: gasMixing/injectorPipe (0.orig/nut is nutUWallFunction). squareBend, squareBendLiq, angledDuctExplicitFixedCoeff and aerofoilNACA0012 all use nutkWallFunction and pass
- **depends_on / middle steps**:
  - a DISPATCH POINT that brae's closure does not have: kEpsilon_cpp.cu:240 and kEpsilon.cu:786 both call nutkWallFunction unconditionally, and kEpsilon.cu additionally passes /*nutWall=*/0 into deviceWallEpsG0 for the near-wall production. So the first step is a per-patch wall-function kind carried on the closure input (a mask like wfBndMask, but typed), not a new formula
  - then each function: nutUSpaldingWallFunction needs a Newton solve on Spalding's law reading |U| (the incompressible lineage has NutUSpaldingPatchField, fv_patch_field.cuh:1683, and deviceBoundaryNutSpalding, so it is a port across lineages rather than from scratch); nutLowReWallFunction is calcNut() == Zero unconditionally and is a two-line addition ONCE the dispatch exists; nutkRoughWallFunction/atmNutkWallFunction need a per-face, time-varying PatchFunction1 roughness where brae carries one scalar
  - and a second consumer: the near-wall production term must take the DISPATCHED nut, not nutk's

</details>
<details><summary><b>[DEAD]</b> A mesh with coupled patches, refused by the kEpsilon closure</summary>

- **citation**: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:427
- **trigger**: `in.hasCoupledPatches`, set only by tests/test_rho_kepsilon_cuda.cu:484 as a control; the whole-step wiring at tests/test_rho_simple_step_cuda.cu:497-535 never sets it, and rhoCreateFields.cu:69 refuses the case first anyway. The extra hazard the message names is real and NOT covered by that earlier gate on the host side: correctNut's boundary assignment is wrong on a coupled patch, where OpenFOAM's correctBoundaryConditions() overwrites it with the interpolated value and Cmu*k_b^2/eps_b != interpolate(Cmu*k^2/eps)
- **of_feature**: cyclicFvPatchField coupled interfaces plus GeometricField::correctBoundaryConditions -- src/finiteVolume/fields/fvPatchFields/constraint/cyclic/cyclicFvPatchField.C
- **path**: device
- **size**: n/a as a refusal; L for the feature
- **fixture**: tests/test_rho_kepsilon_cuda.cu:484 (control only)
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the same interface work as the momentum entry, plus a coupled-aware correctNut

</details>
<details><summary><b>[DEAD]</b> An fvOption the kEpsilon closure does not implement</summary>

- **citation**: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:436
- **trigger**: `in.hasUnportedFvOption`, set only by tests/test_rho_kepsilon_cuda.cu:488. The whole-step device wiring never sets it AND never populates fvoEpsMask/fvoEpsVal/fvoKMask/fvoKVal either (kEpsilon.cuh:152-155), so on the device path an fvOption constraining k or epsilon is silently absent. The host closure does apply it -- kEpsilon_cpp.cu:399,406,531 call cpu::fvOptions::constrain in OpenFOAM's own order
- **of_feature**: fv::optionList applied to both turbulence equations -- src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C:263,285 (`+ fvOptions(alpha, rho, epsilon_)` / `k_`) and the constrain calls at :266 and :288
- **path**: device
- **size**: S -- the masks exist, the resolution does not
- **fixture**: tests/test_rho_kepsilon_cuda.cu:488 (control only)
- **blocks_tutorial**: none -- neither angledDuctExplicitFixedCoeff nor any other rhoSimpleFoam tutorial constrains k or epsilon
- **depends_on / middle steps**:
  - the device driver resolving a scalarFixedValueConstraint naming k or epsilon into the per-cell masks the module already accepts -- the kernel side is done, only the resolution is missing
  - the ORDER is load-bearing and already encoded on the host: relax() -> fvOptions.constrain() -> setValues(wall) (kEpsilon.C:265-267). setValues does source_[nei] -= coeff*value and then ZEROES that coeff (fvMatrix.C:259-291), so only the FIRST setValues touching a cell reaches its neighbours

</details>
<details><summary><b>[DEAD]</b> A div(phi,k) or div(phi,epsilon) scheme other than Gauss upwind</summary>

- **citation**: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:444
- **trigger**: `in.hasNonUpwindDivScheme`, set only by tests/test_rho_kepsilon_cuda.cu:492. The limitation is REAL -- kEpsilon_cpp.cu's two div terms are plain fvm::div with no scheme argument, and the device uses deviceDivUpwindCoeffs -- but no driver parses div(phi,k)/div(phi,epsilon) at all: tests/test_rho_simple_step_cpp.cu:280 hardcodes `in.boundedTurb = true` and reads no turbulence div entry. So a compressible case naming linearUpwind or limitedLinear on k gets upwind silently, which is precisely the substitution this refusal was written to prevent
- **of_feature**: gaussConvectionScheme with a limited interpolation weight -- src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/; the entries are div(phi,k) and div(phi,epsilon), resolved by kEpsilon.C:255 and :277 through fvm::div(alphaRhoPhi, ...)
- **path**: both
- **size**: S to make the refusal live; S again to implement upwind + linearUpwind + limitedLinear for both terms
- **fixture**: validation/rhoKE names `bounded Gauss upwind` and so passes; validation/rhoLUturb names `bounded Gauss linearUpwind grad(k)` but is kOmegaSST
- **blocks_tutorial**: squareBendLiq (div(phi,k|epsilon) bounded Gauss linearUpwind limited) and gasMixing/injectorPipe (Gauss limitedLinear 1) -- both would be run with upwind today rather than refused
- **depends_on / middle steps**:
  - FIRST: derive the flag -- parse div(phi,k) and div(phi,epsilon) in the driver, which is a strictly smaller job than implementing the schemes and turns a silent substitution into a refusal
  - THEN to lift it: the weighted implicit scalar div already exists on both sides (fvm::div with weights on the host; deviceDivLimitedCoeffs on the device, in use at rhoUEqn.cu:311-341), so the port is reusing them for k and epsilon plus the linearUpwind deferred source, which deviceLinearUpwindCorr already provides

</details>
<details><summary><b>[DEAD]</b> A turbulence wall function on a patch whose type is not `wall`</summary>

- **citation**: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:455
- **trigger**: `in.hasNonWallTurbWallFunc`, set only by tests/test_rho_kepsilon_cuda.cu:496. Nothing derives it from the case, so a 0/epsilon carrying epsilonWallFunction on a `patch`-type boundary would divide by an unset near-wall distance instead of being refused. The host closure has no equivalent guard at all
- **of_feature**: nearWallDist -- src/finiteVolume/fvMesh/wallDist/nearWallDist/nearWallDist.C, which fills y only on isA&lt;wallFvPatch> patches; consumed by epsilonWallFunctionFvPatchScalarField.C and nutkWallFunctionFvPatchScalarField.C
- **path**: both
- **size**: S
- **fixture**: none -- no checked-in case puts a wall function on a non-wall patch
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - the driver comparing, per patch, epsilon.boundary[pi]->isTurbulenceWallFunction() against patches[pi].type == "wall" -- both facts are already in hand at buildDeviceWallData's 5-arg call site (kEpsilon.cuh:39-41), so this is a derivation, not a port

</details>
<details><summary><b>[OVER-BROAD]</b> A case that bounds div(phi,k) but not div(phi,epsilon), or the reverse</summary>

- **citation**: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:462
- **trigger**: `in.boundedK != in.boundedEps`. The DEVICE module can already express the split -- it carries two independent bools and applies them at separate sites -- and refuses only because the host reference it is gated against carries ONE `bounded` flag for both (the `bounded` argument of kEpsilonRef::correct). So the refusal is about the oracle, not the code. It is also currently unreachable: tests/test_rho_simple_step_cuda.cu:524 sets both from one hin.boundedTurb, as does the standalone gate at tests/test_rho_kepsilon_cuda.cu:314
- **of_feature**: boundedConvectionScheme -- src/finiteVolume/finiteVolume/convectionSchemes/boundedConvectionScheme/boundedConvectionScheme.C; div(phi,k) and div(phi,epsilon) are separate fvSchemes entries and OpenFOAM resolves each independently
- **path**: both
- **size**: S
- **fixture**: none; every checked-in case and every OF rhoSimpleFoam tutorial uses `$turbulence` for both entries, which is exactly why the shortcut survived
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - splitting kEpsilonRef::correct's single `bounded` argument into boundedK/boundedEps -- one signature change and two call sites in kEpsilon_cpp.cu
  - then the parser change from the div-scheme entry above, since a driver that reads neither entry cannot produce a split anyway
  - a fixture that actually splits them, and a control showing the two terms move independently

</details>
<details><summary><b>[GENUINE]</b> Required closure inputs absent: the two fluxes (mass and volumetric) on internal and boundary faces; rho and nu on cells, boundary faces and wall faces; nut's boundary field; U; the wall-function face mask and per-face near-wall distance</summary>

- **citation**: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:469,476,483,489,494
- **trigger**: A caller half-supplying the closure. Each guard stands for a specific measured defect rather than defensiveness in general: the two fluxes differ by rho and using one for the other is invisible in the incompressible lineage (fvm::div takes the MASS flux, divU is a dilatation and takes phi/interpolate(rho)); there is no case-constant nu on the compressible path, so every fallback that read a scalar nu was a divide-by-zero; DkEff()/DepsilonEff() are volScalarFields whose PATCH values are not the owner cells' (at an inlet OpenFOAM's nut_b is Cmu*k_b^2/eps_b); and without the per-FACE wall mask, correctNut -- which is a whole-GeometricField assignment -- could not tell a wall face from an inlet face on a cell that touches both
- **of_feature**: compressibleTurbulenceModel::phi() (src/TurbulenceModels/compressible/compressibleTurbulenceModel.C) for the volumetric flux; kEpsilon::DkEff/DepsilonEff and correctNut (src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C:206-222); nutkWallFunctionFvPatchScalarField.C for the wall nut
- **path**: device
- **size**: n/a
- **fixture**: tests/test_rho_kepsilon_cuda.cu
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing -- keep them. They are the contract that let the whole-step arm at tests/test_rho_simple_step_cuda.cu:437-536 be built correctly

</details>
<details><summary><b>[GENUINE]</b> A RAS case with no alphat field</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoSimpleFoam_cpp.cu:550 (kOmegaSST arm) and :603 (kEpsilon arm)
- **trigger**: f.alphat.internal empty on a turbulent case -- i.e. no 0/alphat in the start-time directory. Every compressible RAS tutorial ships one, so this fires only on a malformed or hand-assembled case
- **of_feature**: EddyDiffusivity&lt;BasicTurbulenceModel> -- src/TurbulenceModels/compressible/EddyDiffusivity/EddyDiffusivity.C:33-40, which reads alphat at construction and ends every correctNut with alphat_ = rho*nut/Prt followed by alphat_.correctBoundaryConditions()
- **path**: host
- **size**: S for the refusal; M to give the host an alphatWallFunction patch type
- **fixture**: validation/sbMatched, validation/rhoKE and OF's squareBend all ship alphat with compressible::alphatWallFunction walls
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing to lift -- but the neighbouring gap is worth naming: brae has NO compressible::alphatWallFunction patch type, so such a patch is built as a plain calculated one and evaluating it returns whatever 0/alphat shipped (uniform 0 on the walls of both sbMatched and squareBend). The device closure works around it with alphatWallMask/alphatPrtFace (kEpsilon.cuh:139-141), carrying the PATCH's own Prt_ (default 0.85) rather than the model's (default 1.0); the host arm has only evaluateBoundary(), which is not the same call

</details>
<details><summary><b>[GENUINE]</b> simulationType that is neither laminar nor RAS</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:586
- **trigger**: constant/momentumTransport or constant/turbulenceProperties naming LES (or anything else). Steady rhoSimpleFoam with LES is unusual but constructible, and OpenFOAM would build the model
- **of_feature**: turbulenceModel::New dispatch on simulationType -- src/TurbulenceModels/turbulenceModels/turbulenceModel.C and the RAS|LES|laminar selection in TurbulenceModel.C
- **path**: both
- **size**: L
- **fixture**: none compressible; validation/T3A and the saiddes/ksstiddes tests cover LES incompressibly through the legacy device path
- **blocks_tutorial**: none of the five rhoSimpleFoam tutorials
- **depends_on / middle steps**:
  - a compressible LES lineage that does not exist in the OF-mirror tree at all (the legacy path has Smagorinsky, SA-DDES/IDDES and kOmegaSST-DDES/IDDES device kernels, but no _cpp references and no compressible instantiation)
  - delta calculation (cubeRootVol, vanDriest, IDDESDelta) as its own component
  - this is the largest single item in the area and would not begin until a second RAS model has proved the compressible instantiation pattern

</details>

## pressure

### Adjudicated verdicts (disputes and missed items)

| # | kind | item | verdict |
|---|---|---|---|
| 1 | dispute | Item 3: compressible MRF module refusals (GENUINE -> DEAD) | **REJECTED** |
| 2 | dispute | Item 4: incompressible MRF guard (STALE -> keep null-guard, fix wording) | **GENUINE** |
| 3 | dispute | Item 5: fvOptions shared-flag refusal at six pressure sites (OVER-BROAD -> DEAD) | **OVER-BROAD** |
| 4 | dispute | Item 7: `limited <k> corrected` refusal not free to lift | **STALE** |
| 5 | dispute | Item 12: adjustPhi continuity fatal (GENUINE -> OF-PARITY) | **GENUINE** |
| 6 | dispute | Item 13: pressureControl factor refusals (GENUINE -> OF-PARITY) | **GENUINE** |
| 7 | dispute | Item 15: no SIMPLE dict / neither pRefCell nor pRefPoint (GENUINE -> OF-PARITY) | **GENUINE** |
| 8 | missed | fixedFluxPressure -> zeroGradient substitution at field construction, unguarded on the compressible path | **RESOLVED-THIS-SESSION** |
| 9 | missed | incompressible pRefPoint refused with no cell search (twin of rho item) | **GENUINE** |
| 10 | missed | incompressible no-SIMPLE-dictionary refusal (twin of item 15) | **GENUINE** |
| 11 | missed | pRefValue read with scalarOr default where OF's readEntry is mandatory | **HOLE** |
| 12 | missed | V2 device adjustable mask built from fixesValue alone, missing the inletOutlet half | **HOLE** |
| 13 | missed | pEqn_cpp adjustPhi scale loop disagrees with its own sum loop on inletOutlet | **HOLE** |
| 14 | missed | rho _cpp references never populate M.faceFluxCorrection | **HOLE** |
| 15 | missed | rhoCreateFields.cu live coupled-patch refusal never listed as its own entry | **GENUINE** |
| 16 | missed | gpuRhoSimpleFoam live fvOptions and MRF case-level refusals | **GENUINE** |

Evidence, one line each (current file:line -- line numbers are as of 2ecfe0f):

1. tests/test_rho_simple_step_cpp.cu:221 sets in.hasMRF = has("constant/MRFProperties") and is the only driver of the _cpp mirror lineage, so rhoPEqn_cpp.cu:27 and rhoPcEqn_cpp.cu:35 ARE reachable from a real case; the rho-weighted makeRelative is still unported (MRF_cpp.cuh:96 has only the incompressible signature, no rho overload anywhere), so GENUINE stands. Only the device twins (rhoPEqn.cu:131, rhoPcEqn.cu:117) are backstops behind gpuRhoSimpleFoam.cu:412-419's live by-name refusal -- the checker's 'never assigned true anywhere in src/' ignores the host end-to-end harness.
2. pEqn.cu:19 guards `in.hasMRF && !in.mrf` while :154 runs deviceMrfMakeRelative when zones exist -- deleting it would let a declared-MRF case with unbuilt zones silently no-op, so the checker is right the guard must stay. Only the wording is stale: pEqn.cu:20-22 still says 'Not implemented on this path' (and pEqn.cuh:38 comments '// refused'), while pEqn_cpp.cu:18-22 already carries the accurate 'No zones were supplied' message. simpleFoamV2.cu:891 sets only in.mrf, never hasMRF, so from V2 it cannot fire; tests/mrf_probe.cu:81 and tests/test_simple_mrf_cpp.cu:162 do set it.
3. tests/test_rho_simple_step_cpp.cu:197-208 sets hasFvOptions from OptionList::firstUnsupported() and rhoSimpleFoam_cpp.cu:273/305/370 forwards the ONE flag to UEqn, EEqn and pEqn -- an unimplemented option registered on U alone still refuses the pressure equation (rhoPEqn_cpp.cu:32, rhoPcEqn_cpp.cu:40) on the live host lineage, so the checker's blanket DEAD is wrong. The checker is right about the incompressible half: pEqn.cu:23-26 and pEqn_cpp.cu:23-26 correctly cite fvOptions.correct(U) (pEqn.H:49, verified) and sit behind simpleFoamV2.cu:414-431's by-name refusal, but that does not displace OVER-BROAD for the reachable rho sites.
4. rhoPEqn_cpp.cu:36-41 and rhoPcEqn_cpp.cu:44-47 still refuse snGradLimitCoeff != 0 while the host limiter exists in shared code (fvm.cuh laplacianCorrFlux takes limitCoeff), so STALE stands -- and the checker's caveat is verified: neither rho _cpp assembler fills M.faceFluxCorrection (rhoPEqn_cpp.cu:321-322 and rhoPcEqn_cpp.cu:303-304 negate an always-empty vector; fvm::laplacian at fvm.cuh:53-107 never populates it), so lifting requires building the flux the way pEqn_cpp.cu:223 does or `phi = phiHbyA + pEqn.flux()` drops the correction.
5. device_simple.cu:284-296, pEqn_cpp.cu:113-119 and rhoPEqn_cpp.cu:105-116 transcribe adjustPhi.C:100-119's FatalError including the relative totalFlux test (verified against /usr/lib/openfoam/openfoam2412/src/finiteVolume/cfdTools/general/adjustPhi/adjustPhi.C). 'OF-PARITY' is not a census verdict; GENUINE (refusal correct and needed) already says exactly this, so the relabel changes nothing. The checker's actionable remark -- the V2 mask can make this fatal fire on a case OF solves -- is real and adjudicated as its own missed item (simpleFoamV2.cu:724).
6. rhoCreateFields_cpp.cu:146, :159, :164, :182, :192, :197 -- six throws inside makePressureControl (rhoCreateFields_cpp.cu:82+) mirroring pressureControl.C's exit(FatalIOError) sites (verified at pressureControl.C:108-190) in OpenFOAM's own order. Nothing unported, which is precisely what GENUINE means here; the out-of-vocabulary relabel changes nothing.
7. rhoCreateFields_cpp.cu:49-52 (no SIMPLE dictionary) and :72-75 (neither key) mirror findRefCell.C's FatalIOError (verified: findRefCell.C:97-103 'Please supply either pRefCell or pRefPoint'). Refusal correct and needed; the relabel is commentary, not a reclassification.
8. fv_patch_field.cuh:1588-1601 now constructs FixedFluxPressurePatchField (class at :610, refusing at assembly via :630-640 if updateSnGrad never ran); constrainPressure is transcribed in all three host pressure equations (rhoPEqn_cpp.cu:206+, rhoPcEqn_cpp.cu:124+, pEqn_cpp.cu:175+) and deviceConstrainPressure runs at rhoPEqn.cu:273, rhoPcEqn.cu:255, pEqn.cu:262 with the live boundary rho -- the substitution is gone on both lineages (commits 84828bd, 7d6ffdd).
9. createFields_cpp.cu:81-88 throws on pRefPoint, naming findRefCell.C:69-100's mesh.findCell which brae lacks on this lineage -- same genuine gap as rhoCreateFields_cpp.cu:66, correctly refused rather than guessing a cell.
10. createFields_cpp.cu:67-71 throws when p needs a reference and fvSolution has no SIMPLE dictionary, and :91-95 when neither pRefCell nor pRefPoint is given -- both mirror findRefCell.C:97-103's FatalIOError.
11. rhoCreateFields_cpp.cu:76 `refValue = dict->scalarOr("pRefValue", 0.0)` versus findRefCell.C:105 `dict.readEntry(refValueName, refValue)` which FatalIOErrors when absent (verified in OF source) -- a case with pRefCell and no pRefValue is refused by OpenFOAM and silently pinned to p=0 by brae. The same hole exists at createFields_cpp.cu:97 and gpuSimpleFoam.cu:424.
12. simpleFoamV2.cu:724 `adjustable.push_back(f.U.boundary[pi]->fixesValue() ? 0 : 1)` omits adjustPhi.C:59's `&& !isA<inletOutletFvPatchVectorField>` -- brae's InletOutletPatchField inherits MixedPatchField::fixesValue()==true (fv_patch_field.cuh:1010, :1130-1142), so an inletOutlet outlet is marked non-adjustable on the device, its outflow lands in fixedMassOut and is never scaled, and deviceAdjustPhi (device_simple.cu:284-296) can throw the continuity fatal on a case OpenFOAM solves. rhoCreateFields.cu:216-225 builds the same mask correctly and documents why both halves matter.
13. pEqn_cpp.cu:124 `if (U.boundary[pi]->fixesValue()) continue;` while the sum loop at :98 uses `fixesValue() && !isInletOutlet()` -- inletOutlet outflow is counted in adjustableMassOut (the massCorr denominator) but skipped by the scale loop, so continuity is not restored; OF's scale predicate is `!fixesValue() || isA<inletOutlet>` (adjustPhi.C:127-133, verified). rhoPEqn_cpp.cu:84 and :121 use the correct predicate in both loops.
14. rhoPEqn_cpp.cu:321-322 and rhoPcEqn_cpp.cu:303-304 negate a vector nothing ever fills (fvm::laplacian at fvm.cuh:53-107 does not populate it), so on the ADMITTED `corrected` path `phi = phiHbyA + pEqn.flux()` silently drops the non-orthogonal correction flux that fvMatrix.C:1688 adds back in OpenFOAM -- while the CUDA twins build it (rhoPcEqn.cu:400-414, rhoPEqn.cu ffc path) and pEqn_cpp.cu:214-226 shows the host fix. The host REFERENCE is behind its own device module.
15. rhoCreateFields.cu:65-74 throws on cyclic/cyclicAMI/cyclicACMI/processor patches before any device field exists -- the live refusal that makes the module-level hasCoupledPatches backstops (rhoPEqn.cu:146, rhoPcEqn.cu:132) unreachable. Note the surveyor DID cite it inside item 6's trigger text ('rhoCreateFields.cu:67-74'), so item 6 never read as an open hole; the checker is right only that it was not its own entry.
16. gpuRhoSimpleFoam.cu:386-396 refuses unsupported fvOptions BY NAME (readFvOptions unsupported list, naming each source and the supported set) and :412-419 refuses an active constant/MRFProperties citing the density-weighted MRF.DDt(rho,U) it cannot reproduce -- these are the refusals a real compressible case actually hits, and every module-level hasFvOptions/hasMRF guard on the CUDA arm sits behind them.

### Original survey entries closed by this session

- Item 1 (six-module fixedFluxPressure refusals, classified DEAD) -- obsolete: fixedFluxPressure is fully PORTED this session. The refusals and the hasFixedFluxPressure flags they hung on are deleted (grep finds the flag only in PORT.md history); FixedFluxPressurePatchField lives at fv_patch_field.cuh:610 with an assembly-time refusal for drivers that never run constrainPressure, constrainPressure is transcribed in pEqn_cpp/rhoPEqn_cpp/rhoPcEqn_cpp and deviceConstrainPressure runs at pEqn.cu:262, rhoPEqn.cu:273, rhoPcEqn.cu:255 (commits 84828bd, 7d6ffdd).
- Item 2 (V2 case-level substring refusal of fixedFluxPressure, classified OVER-BROAD) -- obsolete: the envelope blocker was LIFTED by commit c2ff3e4; simpleFoamV2.cu:564 now states fixedFluxPressure is supported via the real patch class, and the only remaining refusal in that family is the narrower MRF+fixedFluxPressure combination at the device call site (pEqn.cu:250-257), which is a different, per-combination guard.

### The original survey for this area (as surveyed; see the verdicts above for corrections)

<details><summary><b>[DEAD]</b> All six pressure modules refuse a case that names fixedFluxPressure on p, because constrainPressure is not ported.</summary>

- **citation**: src/applications/solvers/simpleFoam/pEqn.cu:28; src/applications/solvers/simpleFoam/pEqn_cpp.cu:28; src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu:148; src/applications/solvers/rhoSimpleFoam/rhoPEqn_cpp.cu:37; src/applications/solvers/rhoSimpleFoam/rhoPcEqn.cu:134; src/applications/solvers/rhoSimpleFoam/rhoPcEqn_cpp.cu:45
- **trigger**: in.hasFixedFluxPressure == true. Nothing assigns it: the only non-false writes in the tree are the fail-proof arms of four unit tests (test_peqn_cuda.cu:322, test_peqn_cpp.cu:216, test_rho_peqn_cuda.cu:507, test_rho_pceqn_cuda.cu:443, test_rho_peqn_cpp.cu:488, test_rho_pceqn_cpp.cu:409) plus two pure forwardings (rhoSimpleFoam.cu:410, rhoSimpleFoam_cpp.cu:371) from a RhoStepInput/SimpleStepInput field no driver or harness sets. A real case never reaches the pressure equation with it true, because fv_patch_field.cuh:1517 already built the patch as ZeroGradientPatchField -- the substitution happens at field construction and the flag stays false.
- **of_feature**: fixedFluxPressureFvPatchScalarField (fixedGradient + updateableSnGrad) -- /usr/lib/openfoam/openfoam2412/src/finiteVolume/fields/fvPatchFields/derived/fixedFluxPressure/fixedFluxPressureFvPatchScalarField.H:71-74; driven by constrainPressure(p, rho, U, phiHbyA, rhorAU, MRF), constrainPressure.C:60-77, called at rhoSimpleFoam/pEqn.H:12 and pcEqn.H:16 and simpleFoam/pEqn.H:21.
- **path**: both
- **size**: medium -- ~200 lines host (patch class + constrainPressure) + ~80 device (one kernel + the mask + the refresh hook) + a new fixture and gate
- **fixture**: none exists; grep over validation/ finds fixedFluxPressure only inside of_apps build .dep files, and no OF simpleFoam or rhoSimpleFoam tutorial uses it
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - a p-field patch type that is actually fixedGradient rather than the zeroGradient substitution at fv_patch_field.cuh:1517, plus a per-face isFixedFluxPressure mask alongside takeUAtBoundary/adjustable
  - host FixedGradientPatchField boundary coefficients -- these did NOT exist until the uncommitted change at fv_patch_field.cuh:547-570; before it valueBoundaryCoeffs/gradientBoundaryCoeffs fell through to the zero base, so on the host every fixedGradient patch was discretised as zeroGradient and constrainPressure would have written a gradient the matrix never read
  - a per-outer-iteration refresh of the device boundary snapshot, in the pattern DeviceBoundary already uses for inletOutlet/totalPressure/pressureInletOutletVelocity (device_boundary.cuh:105-232) -- the device boundary is pre-baked and constrainPressure rewrites refGrad every outer iteration; DeviceBoundary::refGrad (device_boundary.cuh:36-38) already carries the storage
  - rho on boundary faces and Sf&U_b at the pEqn.H:12 point (both already in hand: in.rhoBndFace, deviceBoundaryFlux)
  - MRF.relative() only if the same case also declares MRF (constrainPressure.C:72)
  - a fixture: nothing in validation/ or in either tutorial family carries a fixedFluxPressure patch, so one must be built

</details>
<details><summary><b>[OVER-BROAD]</b> The incompressible case-level envelope refuses any case whose 0/p text merely contains the string `fixedFluxPressure`, wholesale.</summary>

- **citation**: src/applications/solvers/simpleFoam/simpleFoamV2.cu:574-579
- **trigger**: A substring match on the text of &lt;startTime>/p. This is the ONLY fixedFluxPressure refusal in the tree that a real case can actually reach -- the six module-level ones above are dead -- and it refuses every patch, not just the ones that need the gradient path.
- **of_feature**: constrainPressure.C:65-76 sets the patch snGrad to (phiHbyA_b - rho_b*MRF.relative(Sf&U_b))/(magSf*rhorAU_b); constrainHbyA.C:56-69 makes HbyA_b == U_b on any patch whose U is not assignable.
- **path**: host
- **size**: small -- one loop over p's patches consulting the U-patch assignable predicate the envelope already builds
- **fixture**: none; would need a case with fixedFluxPressure on both an assignable-U and a non-assignable-U patch to gate both halves
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - resolve the blocker PER PATCH instead of per case: on a patch where U.assignable() is false, constrainHbyA has forced HbyA_b = U_b, so phiHbyA_b = rho_b*(Sf&U_b) and constrainPressure's numerator cancels identically -- the prescribed gradient is exactly 0 and the existing zeroGradient substitution at fv_patch_field.cuh:1517 is exact, not an approximation
  - brae already computes exactly the predicate needed: the per-face takeUAtBoundary mask is !assignable() (rhoPEqn.cuh:119, pEqn.cuh:48), so the split is a mask read, not new numerics
  - keep the refusal only for the assignable-U case (zeroGradient/calculated U on a fixedFluxPressure patch), which then needs the full constrainPressure port of the entry above
  - the same per-patch split must be added to the rho lineage, which today has NO case-level fixedFluxPressure guard at all -- rhoCreateFields.cu / rhoCreateFields_cpp.cu never look for it -- so a compressible case with fixedFluxPressure is silently run as zeroGradient with no message

</details>
<details><summary><b>[GENUINE]</b> The compressible pressure equation refuses any case declaring MRF, because pEqn.H:9 / pcEqn.H:11 call MRF.makeRelative(fvc::interpolate(rho), phiHbyA) before adjustPhi.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu:133-136; rhoPEqn_cpp.cu:28-31; rhoPcEqn.cu:119-122; rhoPcEqn_cpp.cu:36-39
- **trigger**: in.hasMRF == true, derived on the host end-to-end harness from the presence of constant/MRFProperties (tests/test_rho_simple_step_cpp.cu:221). The CUDA end-to-end harness never sets it, so on that arm the flag is currently unreachable too.
- **of_feature**: MRFZoneList::makeRelative(const surfaceScalarField& rho, surfaceScalarField& phi) -- MRFZoneList.C:336-346 into MRFZone::makeRelativeRhoFlux, MRFZoneTemplates.C:36-108: phi_f -= rho_f*((Omega x (Cf-origin)) & Sf) on internal and excluded faces, phi_f = 0 on included faces.
- **path**: both
- **size**: small -- ~40 lines (one extra pointer argument and a multiply in two kernels plus the host twin), dominated by building the fixture
- **fixture**: none for compressible MRF; validation/mixerVessel2D + validation/rotatingCylinders + validation/mrfBox are all incompressible. A compressible fixture would be rhoBox or sbMatched plus a constant/MRFProperties.
- **blocks_tutorial**: none -- no rhoSimpleFoam tutorial declares MRF
- **depends_on / middle steps**:
  - a rho weight on the frame flux: deviceMrfMakeRelative (device_MRF.cu:137-157) subtracts the bare geometric term because DeviceMRFZone pre-bakes frameFluxInt/frameFluxBnd without rho (device_MRF.cu:100-112). The compressible form needs rho_f multiplied in at call time, which means keeping the geometric flux and passing the interpolated rho -- rhoPEqn.cu:262 already has it as rhofInt
  - the same rho overload on the host: cpu::MRF::makeRelative (MRF_cpp.cuh) is the incompressible signature only
  - ordering is already correct in both lineages -- the call site is between fvc::flux(HbyA) and adjustPhi, exactly where simpleFoam/pEqn.cu:155-160 puts the incompressible one
  - constrainPressure only if the case ALSO carries fixedFluxPressure; MRF.relative() appears inside constrainPressure.C:72 and nowhere else on this path
  - a compressible MRF fixture: no OF rhoSimpleFoam tutorial declares MRF, and the existing gate (tests/mrf_cuda_vs_openfoam.sh) runs validation/mixerVessel2D, which is incompressible simpleFoam

</details>
<details><summary><b>[STALE]</b> The incompressible pressure equation refuses MRF with the message "Not implemented on this path" -- but MRF.makeRelative IS implemented on this path, twelve lines below the refusal.</summary>

- **citation**: src/applications/solvers/simpleFoam/pEqn.cu:19-22; src/applications/solvers/simpleFoam/pEqn_cpp.cu:18-22
- **trigger**: in.hasMRF && !in.mrf. The guard is really a null-pointer contract ("the case declared MRF but you handed me no zones"), yet the text asserts the term is unported. pEqn.cu:157-160 calls deviceMrfMakeRelative and pEqn_cpp.cu:84 calls MRF::makeRelative unconditionally when the pointer is non-null; simpleFoamV2.cu:395-407 says in so many words that MRF is implemented and READ rather than refused. Additionally nothing on this lineage ever sets hasMRF -- simpleFoamV2.cu sets only in.mrf (line 902) -- so the condition is also unreachable in practice.
- **of_feature**: MRFZoneList::makeRelative(surfaceScalarField&) -- MRFZoneList.C:249-255, at simpleFoam/pEqn.H:5.
- **path**: both
- **size**: trivial -- a message rewrite, or two deleted fields
- **fixture**: validation/mixerVessel2D via tests/mrf_cuda_vs_openfoam.sh and tests/mrf_cpp_vs_openfoam.sh
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing -- rewrite the message to say what the guard actually checks (zones declared but not supplied), or delete the flag and keep the null check on in.mrf alone
  - if the flag is kept, have the envelope actually set it so the guard can fire

</details>
<details><summary><b>[OVER-BROAD]</b> The pressure equation refuses the whole case whenever an unimplemented fvOption is declared, though fvOptions(psi, p, rho.name()) only picks up options registered against the field name "rho".</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu:140-144; rhoPEqn_cpp.cu:33-35; rhoPcEqn.cu:126-130; rhoPcEqn_cpp.cu:41-43; src/applications/solvers/simpleFoam/pEqn.cu:24-26; pEqn_cpp.cu:24-26
- **trigger**: in.hasFvOptions, set by the harness when cpu::fvOptions::OptionList::firstUnsupported() returns a type name (tests/test_rho_simple_step_cpp.cu:192-208). Any unimplemented option type refuses the PRESSURE equation, regardless of which field that option applies to.
- **of_feature**: fv::optionList::operator()(const volScalarField& rho, GeometricField& field, const word& fieldName) -> source(field, fieldName, ds), fvOptionListTemplates.C:34-80 and :113-121. It calls addSup only for options whose applyToField(fieldName) matches, and pEqn.H:32 / pcEqn.H:40 pass rho.name() -- so an option must be registered on "rho" to contribute a single coefficient. angledDuctExplicitFixedCoeff's three options register on U (explicitPorositySource.C:111-114), on thermo.he() (fixedTemperatureConstraint.C:104) and on k/epsilon respectively, so fvOptions(psi, p, rho.name()) is an identically zero matrix there.
- **path**: both
- **size**: small -- one predicate on OptionList plus three call-site changes; medium if a real rho-registered source is also implemented
- **fixture**: validation/angledDuct via tests/rho_angledduct_vs_openfoam.sh (OF's own tutorial, unmodified) already exercises three options none of which touch p
- **blocks_tutorial**: angledDuctExplicitFixedCoeff -- refused by the pressure equation for options that provably contribute nothing to it
- **depends_on / middle steps**:
  - a per-field unsupported query on cpu::fvOptions::OptionList -- firstUnsupported("rho") / ("U") / ("h") -- so each equation refuses only for its own field name, instead of one boolean shared by UEqn, EEqn and pEqn (rhoSimpleFoam.cu:290-291, 366-367, 408-409 all forward the same flag)
  - the OptionList already parses each option's `fields`/`fieldNames` entry for the implemented types, so this is a lookup, not new parsing
  - note the practical ordering: lifting this alone changes nothing for angledDuct, because rhoUEqn refuses on the same shared flag first. The per-field split has to land in all three modules together to be observable
  - an actual `fvOptions(psi,p,rho.name())` term (a mass source on rho) remains genuinely unported and must still refuse

</details>
<details><summary><b>[DEAD]</b> The compressible pressure modules refuse a mesh with cyclic/AMI/processor patches.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu:155-158; src/applications/solvers/rhoSimpleFoam/rhoPcEqn.cu:140-142
- **trigger**: in.hasCoupledPatches, which nothing derives -- rhoSimpleFoam.cu:411 forwards a RhoStepInput field that only tests/test_rho_peqn_cuda.cu:508 and tests/test_rho_ueqn_cuda.cu:509 ever set. The real, mesh-derived refusal is upstream at rhoCreateFields.cu:67-74, which scans the patch list by type and throws before any device field exists, so the pressure module's copy can never be the one that fires.
- **of_feature**: cyclicFvPatchField/cyclicAMIFvPatchField/processorFvPatchField and their updateInterfaceMatrix contribution to the LDU; brae's DeviceMesh deliberately keeps those faces out of the owner-sorted internal-face LDU (device_mesh.cuh:34-37) and carries them as a separate DeviceCyclic interface.
- **path**: both
- **size**: large for the feature; trivial to remove the duplicate flag
- **fixture**: validation/cyclicChannel, validation/pipeCyclic, validation/cyclicChannelAMI exist but are incompressible
- **blocks_tutorial**: none of the six rhoSimpleFoam tutorials has a coupled patch
- **depends_on / middle steps**:
  - coupled-interface support in the pressure laplacian's boundary coefficients: fvm.cuh:83-88 records that gaussLaplacianScheme's coupled branch (corrected deltaCoeffs on the coupled side) is NOT implemented, and that it must be revisited the moment coupled patches are admitted
  - the DeviceCyclic interface path already exists for the legacy incompressible solver (cyclic_simple.cuh, deviceCyclicOffDiagSum) and would have to be threaded into PressureMatrix::view, deviceNormFactor and the AMG agglomeration
  - if the flag stays, have createFields set it rather than leaving two independent guards of the same fact

</details>
<details><summary><b>[STALE]</b> The two _cpp pressure references refuse `limited &lt;k> corrected` laplacianSchemes, saying brae implements only uncapped `corrected` here.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoPEqn_cpp.cu:41-45; src/applications/solvers/rhoSimpleFoam/rhoPcEqn_cpp.cu:49-52
- **trigger**: in.snGradLimitCoeff != 0.0, parsed from the case's fvSchemes. The limitation it describes was lifted: fvm::laplacianCorrFlux implements OF's limiter on the host (fvm.cuh:123-166, the per-face min(k*|orth|/((1-k)*|corr|+SMALL),1)), fvm::laplacianNonOrthSource forwards limitCoeff to it (fvm.cuh:179-206), and this very file passes in.snGradLimitCoeff into that call 240 lines below the refusal (rhoPEqn_cpp.cu:284-285, rhoPcEqn_cpp.cu:269-270) where it is guaranteed zero. The CUDA twins already run the limited path (rhoPEqn.cu:367-389, rhoPcEqn.cu:393-400) and the incompressible host reference uses it without refusing (pEqn_cpp.cu:203-211).
- **of_feature**: fv::limitedSnGrad (limitedSnGrad.H/.C) wrapping correctedSnGrad, selected by `laplacianSchemes default Gauss linear limited &lt;k> corrected`.
- **path**: host
- **size**: trivial for the guards; small for the missing faceFluxCorrection and the gate arm
- **fixture**: validation/airFoil2D via tests/limitedsngrad_vs_openfoam.sh already gates the shared host limiter; validation/sbMatched with a one-line fvSchemes edit covers the compressible call site
- **blocks_tutorial**: none -- all five rhoSimpleFoam tutorials set `Gauss linear corrected`
- **depends_on / middle steps**:
  - delete the two guards
  - first fix what removing them exposes: rhoPEqn_cpp::assemblePEqn never fills M.faceFluxCorrection at all (it negates an empty vector at line 304), where its incompressible twin does (pEqn_cpp.cu:210-211) -- so on the compressible host path `phi = phiHbyA + pEqn.flux()` drops the non-orthogonal correction the source carries, limited or not
  - extend tests/limitedsngrad_vs_openfoam.sh, or add an arm to tests/rho_peqn_vs_openfoam.sh, with `limited 0.5 corrected` in the fixture's fvSchemes -- the existing gate proves the shared host limiter against OF but only through the incompressible lineage

</details>
<details><summary><b>[DEAD]</b> The pressure modules refuse when rho or psi is missing on cells or on boundary faces (and, on the host, when the transonic branch is asked for without psi).</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu:162-164; rhoPcEqn.cu:146-147; rhoPEqn_cpp.cu:22 and :24-26; rhoPcEqn_cpp.cu:30 and :32-34
- **trigger**: A null in.rhoCell/rhoBndFace/psiCell/psiBndFace. The only callers set all four unconditionally (rhoSimpleFoam.cu:397-398, rhoSimpleFoam_cpp.cu:362-364), so no reachable path leaves one null except a direct unit-test call. These are API contract guards, not feature refusals -- worth keeping as such, but they should not be read as an unported feature.
- **of_feature**: rhoSimpleFoam/pEqn.H:2,8,19,22 and pcEqn.H:10,13,23,28 -- rho and psi enter rhorAUf, phiHbyA, phid and the closed-volume correction; psi is not transonic-only because the closed-volume correction at pEqn.H:96-97 is psi-weighted.
- **path**: both
- **size**: n/a
- **fixture**: tests/test_rho_peqn_cuda.cu and tests/test_rho_peqn_cpp.cu exercise these as fail-proof arms
- **blocks_tutorial**: none

</details>
<details><summary><b>[DEAD]</b> constrainHbyA refuses to run without the per-face `assignable` mask, which cannot be recovered from the device bcType.</summary>

- **citation**: src/applications/solvers/simpleFoam/pEqn.cu:97; src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu:186-189; src/applications/solvers/rhoSimpleFoam/rhoPcEqn.cu:169-171
- **trigger**: in.takeUAtBoundary == nullptr. Both drivers set it from the field projection (rhoSimpleFoam.cu:406 from rhoCreateFields; simpleFoamV2.cu:1464 from dTakeU), so it is never null on any reachable path. Kept correctly as a contract: assignable() is not fixesValue() (fv_patch_field.cuh:37-50) -- slip and inletOutlet are non-assignable without fixing a value -- so a caller cannot derive it locally.
- **of_feature**: constrainHbyA.C:56-69: `if (!U.boundaryField()[patchi].assignable() && !isA&lt;fixedFluxExtrapolatedPressure>(p...)) HbyAbf[patchi] = U.boundaryField()[patchi];`
- **path**: both
- **size**: trivial
- **fixture**: tests/test_rho_peqn_cuda.cu:522-530 has the fail-proof arm
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - one real gap hides here: brae's mask implements only the first half of constrainHbyA.C:58-65. The `&& !isA&lt;fixedFluxExtrapolatedPressureFvPatchScalarField>(p.boundaryField()[patchi])` half is nowhere in the tree, so it becomes live the moment fixedFluxExtrapolatedPressure is admitted as a p patch type

</details>
<details><summary><b>[DEAD]</b> adjustPhi refuses to run without the per-face `adjustable` mask, on a case that needs it (pRefCell >= 0).</summary>

- **citation**: src/applications/solvers/simpleFoam/pEqn.cu:170; src/applications/solvers/rhoSimpleFoam/rhoPEqn.cu:326-328; src/applications/solvers/rhoSimpleFoam/rhoPcEqn.cu:342-344
- **trigger**: in.adjustable == nullptr while pRefCell >= 0. Both drivers always supply it (rhoSimpleFoam.cu:407, simpleFoamV2.cu:1464). NOT made stale or duplicated by the new deviceAdjustPhi guards: those check a different thing (whether the continuity error can be removed at all), and this one checks whether the classification the guards depend on was supplied.
- **of_feature**: adjustPhi.C:52-88 classifies each boundary face by `Up.fixesValue() && !isA&lt;inletOutletFvPatchVectorField>(Up)` -- fixesValue, not assignable; the two predicates appear within a few lines of each other in pEqn.H and are not the same question.
- **path**: both
- **size**: trivial
- **fixture**: tests/test_adjust_phi_guards.cu covers the two new device guards directly
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - deviceAdjustPhi now carries OpenFOAM's own two guards (device_simple.cu:284-301: the relative magAdjustableMassOut/totalFlux > SMALL test and the FatalError). That makes three independent transcriptions of adjustPhi.C:100-119 -- rhoPEqn_cpp.cu:104-122, pEqn_cpp.cu:113-126 and the device one -- which now agree; the consolidation opportunity is real but is not a refusal
  - deviceAdjustPhi still returns only massCorr, not adjustPhi.C:145-147's closedVolume predicate, so the device callers substitute `closedVolume = true` whenever pRefCell >= 0 (rhoPEqn.cu:331, rhoPcEqn.cu:347) where the host twin measures it from the fluxes

</details>
<details><summary><b>[DEAD]</b> The CUDA pressure predictor refuses SIMPLEC without the pressure field and its boundary, and assemblePEqn refuses `corrected` without the pressure field.</summary>

- **citation**: src/applications/solvers/simpleFoam/pEqn.cu:93-95 and :260-262
- **trigger**: in.consistent with dbP or p null; in.correctedLaplacian with p null. simpleFoamV2.cu passes both on every call, so neither is reachable. Pure argument contracts: SIMPLEC needs snGrad(p) and grad(p) (pEqn.H:14-15) and the non-orthogonal correction needs grad(p).
- **of_feature**: simpleFoam/pEqn.H:8-16 (rAtU, phiHbyA += interpolate(rAtU-rAU)*snGrad(p)*magSf, HbyA -= (rAU-rAtU)*grad(p)) and gaussLaplacianScheme's corrected source.
- **path**: device
- **size**: n/a
- **fixture**: tests/test_peqn_cuda.cu
- **blocks_tutorial**: none

</details>
<details><summary><b>[GENUINE]</b> adjustPhi raises a fatal error when the continuity error cannot be removed by adjusting the outflow.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoPEqn_cpp.cu:115-121; src/applications/solvers/simpleFoam/pEqn_cpp.cu:123-125; src/cuda/device_simple.cu:295-301 (reached from rhoPEqn.cu:330, rhoPcEqn.cu:346, pEqn.cu:171)
- **trigger**: |fixedMassOut - massIn|/totalFlux > 1e-8 while the adjustable outflow is negligible (|adjOut| &lt;= VSMALL or |adjOut|/totalFlux &lt;= SMALL). Fires on a case whose velocity boundary conditions admit no solution -- the uninitialised-outflow case OF tells you to fix with potentialFoam.
- **of_feature**: adjustPhi.C:100-119, FatalErrorInFunction "Continuity error cannot be removed by adjusting the outflow". Faithfully mirrored, including OF's relative test against totalFlux = VSMALL + sum(mag(phi)) over the WHOLE surface field, which is why deviceAdjustPhi now takes the internal flux too (device_simple.cuh:62-65).
- **path**: both
- **size**: n/a -- already implemented on all three paths
- **fixture**: tests/test_adjust_phi_guards.cu gates both guards against adjustPhi.C:90-119
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing -- this is OpenFOAM's own refusal and must stay fatal; the device path was the last of the three to gain it and previously carried on with massCorr = 1.0, handing the pressure equation an inconsistent right-hand side that converges to something plausible

</details>
<details><summary><b>[GENUINE]</b> pressureControl refuses pMaxFactor/pMinFactor/rhoMax/rhoMin when the corresponding reference pressure (or density) cannot be evaluated from the boundary conditions.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:145-149, :158-161, :163-166, :181-185, :191-194, :196-199
- **trigger**: system/fvSolution's SIMPLE dict names pMaxFactor (or rhoMax/pMinFactor/rhoMin) while no p patch fixesValue() and no pRefCell was set -- so pLimits stays false. rhoMax/rhoMin additionally need a resolvable reference DENSITY.
- **of_feature**: pressureControl.C:99-105, :118-126, :127-137, :156-162, :173-179, :180-187 -- each an exit(FatalIOError), transcribed line for line. brae's makePressureControl (rhoCreateFields_cpp.cu:79-205) mirrors pressureControl.C:33-190 in OpenFOAM's own order, including the pMax-and-pMin short circuit.
- **path**: host
- **size**: n/a -- already implemented; the device consumes the resolved limits (rhoSimpleFoam.cu:538-543)
- **fixture**: validation/naca0012 (aerofoilNACA0012 ships pMinFactor 0.1 / pMaxFactor 2 and a freestreamPressure patch, which derives from mixedFvPatchScalarField so fixesValue() is true and pLimits resolves)
- **blocks_tutorial**: none -- aerofoilNACA0012 resolves its factors
- **depends_on / middle steps**:
  - nothing -- these are OF's own fatal errors, not brae limitations. The one brae-side note is that OF issues an IOWarning before the rhoMax/rhoMin branch (pressureControl.C:118-124, :172-178) and brae throws instead of warning on the reference-evaluation failures, which is the same outcome OF reaches two lines later

</details>
<details><summary><b>[GENUINE]</b> The pressure reference refuses pRefPoint: OpenFOAM locates the cell with mesh.findCell, brae has no point-location search on this path.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:65-67
- **trigger**: p needs a reference (no patch fixesValue) and the SIMPLE dict names pRefPoint rather than pRefCell.
- **of_feature**: findRefCell.C:69-100 -- setRefCell resolves pRefPoint through mesh.findCell(refPointi) and reduces the owning rank's cell index across processors.
- **path**: host
- **size**: small -- a findCell over cell centres/faces; the rest of the path already exists
- **fixture**: none; would be a one-line fvSolution edit on validation/rhoBox (a closed box, the only shape where the reference is needed at all)
- **blocks_tutorial**: none -- no simpleFoam or rhoSimpleFoam tutorial sets pRefPoint
- **depends_on / middle steps**:
  - a point-in-cell search over the PrimitiveMesh (cell bounding boxes plus a face-normal containment test), which brae has nowhere on this lineage
  - nothing else -- the resolved refCell then feeds the existing setReference path (deviceSetReference, rhoPEqn.cu:454), which is already gated

</details>
<details><summary><b>[GENUINE]</b> createFields refuses when p needs a reference but fvSolution has no SIMPLE dictionary, or has one naming neither pRefCell nor pRefPoint.</summary>

- **citation**: src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:48-50 and :71-73
- **trigger**: No p patch fixesValue() (needReference true) and the case supplies no reference. This is exactly the condition that also makes adjustPhi run and the pressure operator singular.
- **of_feature**: findRefCell.C:33-119 -- setRefCell FatalIOErrors when the reference is required and neither entry is present.
- **path**: host
- **size**: n/a
- **fixture**: validation/rhoBox (closed, references the pressure)
- **blocks_tutorial**: none
- **depends_on / middle steps**:
  - nothing -- OF's own refusal, mirrored. It is worth recording that brae keys adjustPhi and the closed-volume correction off pRefCell >= 0 (rhoPEqn.cu:322, rhoPcEqn.cu:338) where OF keys adjustPhi off p.needReference() (adjustPhi.C:43); the two agree only because this refusal makes pRefCell >= 0 exactly when needReference is true

</details>

## envelope

### Adjudicated verdicts (disputes and missed items)

| # | kind | item | verdict |
|---|---|---|---|
| 1 | dispute | gpuRhoSimpleFoam pMaxFactor/pMinFactor refusal when no p patch fixesValue | **OVER-BROAD** |
| 2 | dispute | simpleFoamV2 RASModel whitelist excluding RNGkEpsilon | **GENUINE** |
| 3 | dispute | functionObject not-implemented report vs Time construction ordering | **STALE** |
| 4 | dispute | MRF cellZone-missing refusal labelled a port gap | **GENUINE** |
| 5 | dispute | laplacianSchemes `limited <k>` refusal | **GENUINE** |
| 6 | missed | kOmegaSSTLM accepted on the compressible path despite 'variants stay refused' comment | **HOLE** |
| 7 | missed | RNGkEpsilon runs unstated on the compressible path | **REJECTED** |
| 8 | missed | fixedFluxPressure unguarded on the shipped brae_simpleFoam/brae_rhoSimpleFoam binaries | **HOLE** |
| 9 | missed | hasFixedFluxPressure assigned nowhere in production (dead flag, six unreachable refusals) | **RESOLVED-THIS-SESSION** |
| 10 | missed | simpleFoamV2 latestTime->'0' making the ffp text guard silently vanish | **RESOLVED-THIS-SESSION** |

Evidence, one line each (current file:line -- line numbers are as of 2ecfe0f):

1. src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:229-257 scans only fixesValue patches and throws at :252-256 claiming OF refuses with 'pressure limits are not set' -- a string absent from all of OF v2412; pressureControl.C:52-58 instead ACCEPTS a closed domain via setRefCell (pMax=pMin=refValue, factors scale it, erroring only when pRefCell/pRefValue are also missing, findRefCell.C:100-105), and brae already holds rc.pRefCell/pRefValue at :275-286 plus a full makePressureControl on the mirror (rhoCreateFields_cpp.cu:81-99) -- so it refuses a case OF accepts and brae could trivially compute.
2. src/applications/solvers/simpleFoam/simpleFoamV2.cu:500-506 refuses outside {kEpsilon,kOmegaSST,realizableKE,SpalartAllmaras,kOmegaSSTLM}; the driver wires its own model flags at :898-914 and no line in the file ever sets keCoeffs.rng (grep: zero hits -- the shared reader turbulence_setup.cuh:294-319 is not called here), so lifting the refusal alone would silently run RNG cases with standard-kEpsilon coefficients; the checker is right, the refusal is needed until ~15 lines of coefficient wiring land.
3. src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:766 constructs Time (which runs the brae_time.cuh:143-148 noticeIgnored report) AFTER every refusal site (thermo :105, RAS :342-346, fvOptions :395, MRF :413, pressure limits :252), while the comment at :99-104 claims 'Time at START-UP, before anything can refuse' above a mere ObjectRegistry declaration; the checker is right that nothing is DEAD -- the report is live for every case reaching the run body -- what is stale is the comment's ordering claim, and refusing cases still never see the report.
4. src/applications/solvers/simpleFoam/simpleFoamV2.cu:409-411 (envelope blocker) and :658-660 (run-body throw) fire when MRFProperties names a zone absent from cellZones, and OF FatalErrors identically at MRFZone.C:579-589 ('cannot find MRF cellZone'); the checker's fact is confirmed and corrects the framing (OF-parity well-formedness guard, nothing to lift or port) but the classification stands -- the refusal is correct and needed.
5. src/applications/solvers/simpleFoam/simpleFoamV2.cu:336-369 honours any parseable coeff in [0,1] (r.limitCoeff at :367, gated by tests/limitedsngrad_vs_openfoam.sh) and the blocker at :468-472 fires only on a missing/unparseable coeff, one outside [0,1] (exact OF parity: limitedSnGrad.H:148-155 FatalIOError), or an unported inner snGrad scheme word outside {corrected,uncorrected,orthogonal}; the refusal fires only where refusing is right -- only the message text 'Only the uncapped corrected is ported' at :470 is stale (~5-line fix).
6. src/applications/solvers/common/turbulence_setup.cuh:249-250 sets ctl.sst=true for kOmegaSSTLM, so the guard at gpuRhoSimpleFoam.cu:341-346 (!ctl.sst) passes it against its own comment; the driver builds DeviceSimpleSolver with only k/eps/nut (gpuRhoSimpleFoam.cu:464-467; ReThetat/gammaInt default nullptr, device_simple_foam.cuh:76-77), so device_simple_foam.cu:595-601 never populates the LM buffers, yet :853 hands gammaIntEff_.data() to the SST kernel and :870-874 runs deviceKOmegaSSTLMCorrect on the empty buffers -- and even populated, the transition transport carries no alpha*rho weighting (OF kOmegaSSTLM.C:529-541, :554-580). Fix is ~1 line: widen the guard with ctl.lm.
7. No refusal is missing -- the compressible RNG path is implemented, not substituted: src/cuda/device_kepsilon.cu:286-303 composes the rho weighting (rw) with C1p=C1-R exactly matching RNGkEpsilon.C:282-290's (C1-R)*alpha*rho*GbyNu*Cmu*k, device_simple_foam.cu:876-895 passes ctl_.keCoeffs (rng/eta0/beta filled by turbulence_setup.cuh:294-319) together with rho/mu/rhoBnd/wall-nu, and the diffusivity/divU are rho-scaled (device_kepsilon.cu:631-639, :744-760, :791-795). Real debts are a missing compressible RNG gate (only tests/test_rng_kepsilon.cu exists) and gpuRhoSimpleFoam.cu:338-341's comment silently lumping RNG under 'standard kEpsilon' -- test/comment debt, not a hole or missing refusal.
8. Amended by this session but the substance stands: src/finiteVolume/fields/fv_patch_field.cuh:1588-1601 now builds the real FixedFluxPressurePatchField for every caller, but its refusal lives only in the host coefficient hooks (requireUpdated, fv_patch_field.cuh:623-641), which the shipped device path never calls -- device_simple_foam.cu contains no deviceConstrainPressure and no host *Coeffs() call, and DeviceBoundary::snGradMask (built at device_boundary.cuh:70) is consumed only by the mirror modules (pEqn.cu:262, rhoPEqn.cu:273, rhoPcEqn.cu:255) -- so the shipped binaries run ffp as a FROZEN construction-time gradient (file entry or zero, fv_patch_field.cuh:1590-1599): silently right on non-assignable-U walls, silently wrong opposite assignable-U patches or under MRF, with no refusal firing.
9. grep over src+tests finds hasFixedFluxPressure only in PORT.md history (src/applications/solvers/rhoSimpleFoam/PORT.md:1690,:1735); the flags and their six dead refusal sites are deleted, replaced by FixedFluxPressurePatchField's everUpdated_ guard (fv_patch_field.cuh:632-641) plus live constrainPressure transcriptions (pEqn_cpp.cu:182, rhoPEqn_cpp.cu:216, rhoPcEqn_cpp.cu:130) and deviceConstrainPressure calls (pEqn.cu:262, rhoPEqn.cu:273, rhoPcEqn.cu:255).
10. The 0/p substring guard the hole undermined is deleted -- simpleFoamV2Envelope now records ffp as supported (simpleFoamV2.cu:564-568) and no scan of <startTime>/p remains, so the ptext-cleared failure mode cannot occur. Residual separate flaw worth its own entry: simpleFoamV2.cu:624-625 still resolves `startFrom latestTime` to the literal '0' for the createFields read (the shared resolveStartTime, start_time.cuh:21, used by gpuRhoSimpleFoam.cu:129, is not called), so a restart with a 0/ dir silently starts from 0/ while a missing 0/ fails loudly (gzSlurp throws 'cannot open', foam_token_reader.cu:559-560).

### Original survey entries closed by this session

- Entry 1 (OVER-BROAD wholesale fixedFluxPressure substring refusal on simpleFoamV2) -- resolved by this session's full ffp port: the <startTime>/p text scan is deleted from simpleFoamV2Envelope (simpleFoamV2.cu:564-568 records the lift), the factory builds FixedFluxPressurePatchField (fv_patch_field.cuh:1588-1601), pEqn.cu:262 runs deviceConstrainPressure each assembly with the MRF+ffp combination refused by name at pEqn.cu:253-257, and validation/ffpi_vs_openfoam.sh gates it (CMakeLists.txt:1636-1638). Caveat: tests/simplefoam_v2_dispatch.sh:158-160 still asserts the LIFTED refusal (try_refusal fixedfluxp) and needs flipping to an accept-arm or it fails; and per the M3 verdict the shipped brae_simpleFoam/brae_rhoSimpleFoam binaries remain a HOLE, so entry 1's resolution covers the V2/mirror path only.
- Entry 1's depends_on item 6 (the DEAD hasFixedFluxPressure flag, pEqn.cuh:36/rhoPEqn.cuh:127) -- resolved: the flags and all six refusal sites they gated are deleted, superseded by the FixedFluxPressurePatchField everUpdated_ guard (fv_patch_field.cuh:632-641).

### The original survey for this area (as surveyed; see the verdicts above for corrections)

<details><summary><b>[OVER-BROAD]</b> A pressure patch of type fixedFluxPressure refuses the whole case, wholesale, on a raw substring scan of &lt;startTime>/p.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/simpleFoam/simpleFoamV2.cu:574-579
- **trigger**: The literal text "fixedFluxPressure" appearing anywhere in the file &lt;caseDir>/&lt;startTime>/p (comments included). startTime is resolved crudely here: `startFrom latestTime` is hardcoded to "0" rather than going through resolveStartTime(), and readFileExpanded (foam_dict.cuh:444-450) neither throws on a missing file nor reads .gz — so on a compressed 0/p.gz, or a restart-only case with no 0/, ptext is empty and the guard silently does NOT fire, leaving brae to substitute zeroGradient exactly as the message warns.
- **of_feature**: fixedFluxPressureFvPatchScalarField driven by constrainPressure(p, U, phiHbyA, rAtU, MRF) — /usr/lib/openfoam/openfoam2412/src/finiteVolume/cfdTools/general/constrainPressure/constrainPressure.C:37-78 (it writes the snGrad as (phiHbyA_b - rho_b*MRF.relative(Sf&U_b))/(magSf*rhorAU_b)); the cancellation comes from constrainHbyA.C:56-69, which sets HbyA_b = U_b on every patch whose U is !assignable().
- **path**: both
- **size**: Narrowing the refusal: ~30 lines, host only, no device work. Full constrainPressure: ~150-250 lines across pEqn.cu/pEqn_cpp.cu/rhoPEqn*.cu plus one device kernel to refresh refGrad.
- **fixture**: tests/simplefoam_v2_dispatch.sh:158-160 (try_refusal fixedfluxp) exists but asserts the wholesale refusal — it mutates upperWall (a noSlip, i.e. non-assignable, patch), which is exactly the case that should be ACCEPTED. To narrow the refusal the gate must be split into an accept-arm (fixedFluxPressure on a wall) and a refuse-arm (fixedFluxPressure opposite an inletOutlet/pressureInletVelocity U patch) as its control.
- **blocks_tutorial**: none — no OpenFOAM v2412 simpleFoam or rhoSimpleFoam tutorial ships fixedFluxPressure (grep over both tutorial trees returns nothing). It fires only on user cases and on anything templated from a pimpleFoam setup.
- **depends_on / middle steps**:
  - Narrowing the trigger only: replace the text scan with a typed test on the already-built fields — f.p.boundary[pi] type == fixedFluxPressure AND f.U.boundary[pi]->assignable() (the assignable() classification is already used at simpleFoamV2.cu:731) OR that patch lies in an MRF zone. Nothing new is needed for this half.
  - Implementing the real gradient path: map fixedFluxPressure to FixedGradientPatchField instead of ZeroGradientPatchField (fv_patch_field.cuh:1517 — the class already exists at fv_patch_field.cuh:528 and its device refGrad hook at fv_patch_field.cuh:986)
  - expose the BOUNDARY halves of phiHbyA and rhorAUf/rAtU out of pEqn.cu / pEqn_cpp.cu, which today only publish internal faces
  - a per-outer-iteration setter that rewrites DeviceBoundary::refGrad (the buffer exists; nothing writes it per iteration yet)
  - MRF.relative() on an included patch, for the MRF-zone case
  - the DEAD hasFixedFluxPressure flag (pEqn.cuh:36, rhoPEqn.cuh:127) must actually be assigned by a driver — no production path sets it today, only tests/test_peqn_cuda.cu:322 etc.

</details>
<details><summary><b>[STALE]</b> gpuRhoSimpleFoam refuses any case with SIMPLE/pMaxFactor or pMinFactor when no p patch fixesValue(), claiming OpenFOAM refuses the same case.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:250-257
- **trigger**: system/fvSolution SIMPLE has pMaxFactor or pMinFactor and every p patch is zeroGradient/calculated (a closed domain, or fixedFluxPressure everywhere). The claimed OF error string "pressure limits are not set" does not exist anywhere in the OpenFOAM v2412 source.
- **of_feature**: Foam::pressureControl — /usr/lib/openfoam/openfoam2412/src/finiteVolume/cfdTools/general/pressureControl/pressureControl.C:52-58: `if (pRefRequired && setRefCell(p, dict, refCell_, refValue_)) { pLimits = true; pMax = pMin = refValue_; }`. pRefRequired defaults true (pressureControl.H:88) and rhoSimpleFoam/createFields.H:41 takes that default, so OpenFOAM ACCEPTS a closed domain with pRefCell/pRefValue and scales the factors off refValue. brae also omits OF's rhoMax/rhoMin -> pMax/pMin back-compat branch (pressureControl.C:118-152, :174-207).
- **path**: host
- **size**: ~40 lines — delete the local block at gpuRhoSimpleFoam.cu:224-262 and call makePressureControl(p, rhoField, simpleDict, nC).
- **fixture**: tests/test_rho_create_fields_cpp.cu already gates the OF-mirror version (it references pMaxFactor). There is NO fixture on the shipping driver's version; a control that a closed-domain + pRefValue + pMaxFactor case RUNS is what is missing.
- **blocks_tutorial**: none of the six today (aerofoilNACA0012 ships pMinFactor/pMaxFactor but has fixedValue p at the farfield). It blocks any closed compressible domain — the exact configuration ctl.needRef at gpuRhoSimpleFoam.cu:265-278 was added for.
- **depends_on / middle steps**:
  - Nothing new — the correct code is already written on the OF-mirror path: makePressureControl in /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:79-205 reproduces pressureControl.C in OF's own order, setRefCell included (rhoCreateFields_cpp.cu:93-99) and the rhoMax/rhoMin inference included (:153-168, :188-201).
  - The shipping driver gpuRhoSimpleFoam.cu is the only file that has not adopted it; brae_rhoSimpleFoam is built from that one file (CMakeLists.txt:1386) while the OF-mirror rhoSimpleFoam.cu/_cpp are library-only and driven by tests.
  - To adopt it: PressureControl/makePressureControl need to be callable without the RhoCreateFields aggregate, or the driver needs to build a GeometricField&lt;scalar> rho boundary before the call (it already has one).

</details>
<details><summary><b>[GENUINE]</b> An active constant/MRFProperties zone refuses the compressible solver outright.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:412-418
- **trigger**: constant/MRFProperties exists with at least one zone whose `active` is not off. Note it is checked with readMRFProperties(...).active only — the zone's cellZone is not validated against the mesh here, unlike simpleFoamV2.cu:405-410.
- **of_feature**: MRFZoneList::DDt(rho, U) — /usr/lib/openfoam/openfoam2412/src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C:210-217, literally `return rho*DDt(U)`; called from rhoSimpleFoam/UEqn.H:8. The pressure side is MRF.makeRelative(fvc::interpolate(rho), phiHbyA) at rhoSimpleFoam/pEqn.H:9 and the MRF argument to constrainPressure at pEqn.H:12.
- **path**: both
- **size**: ~120-180 lines: host zone build is reuse, the device Coriolis kernel needs a rho argument, makeRelative needs a rho-weighted variant.
- **fixture**: None on the compressible path. tests/mrf_cuda_vs_openfoam.sh and tests/mrf_cpp_vs_openfoam.sh gate the incompressible term. No OpenFOAM rhoSimpleFoam tutorial ships MRFProperties, so a new fixture case (or a rotating variant of angledDuct) has to be built before this can be gated — that is the real cost, not the kernel.
- **blocks_tutorial**: none of the six rhoSimpleFoam tutorials ships MRFProperties.
- **depends_on / middle steps**:
  - The incompressible MRF is already fully ported and gated: cpu::MRF::readMRFProperties / buildZone / correctBoundaryVelocity (MRF_cpp.cuh), buildDeviceMRFZone + deviceMrfCoriolis (device_MRF.cuh), driven from simpleFoamV2.cu:657-685.
  - A rho-weighted Coriolis: deviceMrfCoriolis takes no rho — either add a rho array argument or multiply the returned acceleration by the cell rho before it enters the momentum source. This is the only genuinely new arithmetic.
  - MRF.makeRelative on a MASS flux: the incompressible makeRelative subtracts (Omega x r)&Sf; the compressible one must subtract rho_f*((Omega x r)&Sf), i.e. it needs the interpolated face rho that pEqn already builds.
  - The cellZone-exists guard from simpleFoamV2.cu:405-410 should come across with it, otherwise a compressible case naming a missing zone rotates nothing and converges.
  - constrainPressure's MRF argument only matters once fixedFluxPressure is implemented (entry 1), so it can be deferred.

</details>
<details><summary><b>[GENUINE]</b> On the compressible path, SpalartAllmaras and realizableKE are refused because they are not rho-weighted.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:342-346
- **trigger**: `ctl.turbulent && !ctl.sst && !keStandard`, where keStandard = !sst && !sa && !keCoeffs.realizable. So it fires for RASModel SpalartAllmaras and realizableKE. IMPORTANT HOLE: turbulence_setup.cuh:235-236 sets `ctl.sst = (model == "kOmegaSST") || ctl.lm`, so kOmegaSSTLM sets sst=true and slips THROUGH this guard even though the comment two lines above says the kOmegaSST variants "stay refused". A compressible kOmegaSSTLM case therefore runs the un-rho-weighted Langtry-Menter transport silently. Likewise RNGkEpsilon: turbulence_setup.cuh:239-240 accepts it and it is neither sst nor sa nor realizable, so keStandard is true and it passes — running RNG coefficients through whatever rho-weighting the standard kEpsilon path applies.
- **of_feature**: compressible::RASModels::SpalartAllmaras and realizableKE — /usr/lib/openfoam/openfoam2412/src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras.C and .../RAS/realizableKE/realizableKE.C, instantiated for the compressible transport type so every transport term carries alpha*rho and the diffusivity is rho*DnuTildaEff.
- **path**: both
- **size**: Closing the LM/RNG hole: ~5 lines, host. Porting SA compressible: ~200 lines. realizableKE compressible: ~250 lines (the reaction term is the work).
- **fixture**: None. tests/rho_kepsilon_vs_openfoam.sh and tests/rho_komegasst_vs_openfoam.sh gate the two that are done. There is no negative control asserting that a compressible SA/realizableKE/LM/RNG case is refused — which is why the LM hole was never noticed.
- **blocks_tutorial**: none of the six (all are kEpsilon, kOmegaSST or laminar).
- **depends_on / middle steps**:
  - The kEpsilon and kOmegaSST rho-weighting already exists and is the template: rho on every RHS term, rho in the diffusivity, the volumetric divU term, and the per-face wall nu (see the Compressible structs in rhoSimpleFoam_cpp.cu:503-560, :595-620).
  - For realizableKE specifically: deviceEpsReactionRealizable is a strain-based epsilon reaction that has no rho-weighted form at all — that expression must be re-derived against realizableKE.C, not just multiplied through.
  - For SpalartAllmaras: the nuTilda transport must become a rhoNuTilda transport; brae's SA rides the k slot (simpleFoamV2.cu:925-932) so the slot plumbing is reusable.
  - Closing the kOmegaSSTLM / RNGkEpsilon hole needs no new physics — just widening the guard to `ctl.lm || ctl.keCoeffs.rng` until those two are rho-weighted.

</details>
<details><summary><b>[STALE]</b> simpleFoamV2 refuses every RASModel outside {kEpsilon, realizableKE, kOmegaSST, kOmegaSSTLM, SpalartAllmaras}.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/simpleFoam/simpleFoamV2.cu:500-506
- **trigger**: constant/turbulenceProperties RAS/RASModel is any other word. In practice the one that matters is RNGkEpsilon, which brae implements everywhere except here.
- **of_feature**: incompressible::RASModels::RNGkEpsilon — /usr/lib/openfoam/openfoam2412/src/TurbulenceModels/turbulenceModels/RAS/RNGkEpsilon/RNGkEpsilon.C (Cmu 0.0845, C1 1.42, C2 1.68, C3 -0.33, sigmak = sigmaEps = 0.71942, plus the R = eta(1-eta/eta0)/(1+beta*eta^3) production correction).
- **path**: host
- **size**: ~15 lines, host only, zero device work.
- **fixture**: tests/simplefoam_v2_dispatch.sh has try_refusal arms for schemes but none for RASModel. A gate would be tests/kepsilon-style: run pitzDaily with RNGkEpsilon on v2 against OpenFOAM, with the standard-kEpsilon coefficients as the discriminating control (Cmu 0.09 vs 0.0845 is a visible difference in nut).
- **blocks_tutorial**: n/a (incompressible path).
- **depends_on / middle steps**:
  - Nothing on the device: the RNG branch is already in the fused kernel — device_kepsilon.cu:273-275 takes `int rng, scalar eta0, scalar beta`, computes R at :292-295, and is dispatched at :560 from KEpsilonCoeffs::rng.
  - The shared reader already parses it: turbulence_setup.cuh:239-240 accepts RNGkEpsilon and :280-301 fills every coefficient from RNGkEpsilonCoeffs. gpuSimpleFoam gets it for free through that reader.
  - simpleFoamV2 does not call readTurbulenceModel — it sets its own model flags at simpleFoamV2.cu:908-925 and never sets keCoeffs.rng. So the work is: add "RNGkEpsilon" to the envelope list, and mirror turbulence_setup.cuh:280-301 into that block.
  - The header doc at simpleFoamV2.cuh:46 ("only kEpsilon and laminar are wired") is stale by four models and should be corrected in the same change.

</details>
<details><summary><b>[GENUINE]</b> Any patch of type cyclic / cyclicAMI / cyclicACMI / cyclicPeriodicAMI / processor refuses the rebuilt simpleFoam.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/simpleFoam/simpleFoamV2.cu:557-561
- **trigger**: constant/polyMesh/boundary contains any patch whose type isCoupledInterfaceType (foam_dict.cuh:56-59) or is "processor". Necessary for THIS path: simpleFoamV2.cu:688 calls buildDeviceMesh(m, g, fvp) with the cyclics argument defaulted to {} (device_mesh.cuh:60,66), and DeviceBoundary skips coupled faces (device_boundary.cuh:52) — so those faces would simply vanish from both the matrix and the flux, silently.
- **of_feature**: cyclicFvPatchField::updateInterfaceMatrix / cyclicAMIFvPatchField — /usr/lib/openfoam/openfoam2412/src/finiteVolume/fields/fvPatchFields/constraint/cyclic/cyclicFvPatchField.C and .../cyclicAMI/cyclicAMIFvPatchField.C; the ACMI area split is cyclicACMIPolyPatch.C.
- **path**: both
- **size**: ~300-400 lines to reach cyclic parity with the shipping solver, plus the _cpp reference work (which is the larger half, because the gate needs an oracle).
- **fixture**: tests/test_coupled_patch_refusal.cu gates that the refusal fires. pipeCyclic is the natural end-to-end case once it is lifted; tests/pipecyclic-style gates exist for the shipping solver.
- **blocks_tutorial**: n/a for rhoSimpleFoam (no rhoSimpleFoam tutorial has a coupled patch). On the simpleFoam side it blocks pipeCyclic (and rotatingCylinders/mixerVessel2D via cyclicAMI).
- **depends_on / middle steps**:
  - The machinery already exists and is used by the SHIPPING solvers: buildCyclicInterfaces (cyclic_interface.cuh:54) + buildDeviceCyclic (device_cyclic.cuh:43) are called from DeviceSimpleSolver's constructor at device_simple_foam.cu:85,125; cyclicAMI/ACMI come through buildGeometryPatchesAndAMI (acmi_area_scaling.cuh:211), used by gpuSimpleFoam.cu:391, gpuPimpleFoam.cu:263 and gpuRhoSimpleFoam.cu:117.
  - simpleFoamV2 uses buildPatches(m, g), not buildGeometryPatchesAndAMI — swapping that in is step one (it is exactly g.build + buildPatches on a mesh with no ACMI).
  - Thread a std::vector&lt;CyclicInterface> into buildDeviceMesh and a DeviceCyclic/DeviceAMI through StepInput into UEqn.cu and pEqn.cu, which today take neither.
  - The linear solver must apply updateInterfaceMatrix: device_simple_foam.cuh:832,890 also records that a cyclic/AMI mesh forces Jacobi-PCG instead of AMG — the rebuilt path's pressure solve inherits that constraint.
  - The _cpp reference (createFields_cpp.cu, fvc.cuh) has NO cyclic handling at all, so there is no host oracle to gate the rebuilt kernels against until it does.
  - `processor` is a separate concern entirely (decomposed case + MPI) and should stay refused.

</details>
<details><summary><b>[GENUINE]</b> fvOptions declaring a type the reader does not recognise refuses the case, by name.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/simpleFoam/simpleFoamV2.cu:424-431 (envelope, via cpu::fvOptions::read) and /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:386-395 (via readFvOptions)
- **trigger**: system/fvOptions or constant/fvOptions declares an active source whose type is outside the implemented set. The two drivers use DIFFERENT readers with DIFFERENT sets, and both error messages are stale: the shared device framework (fv_options.cuh:192-194, :252-272) supports eleven types — vectorSemiImplicitSource, explicitPorositySource, meanVelocityForce, limitVelocity, actuationDiskSource, rotorDisk/rotorDiskSource, velocityDampingConstraint, limitTemperature, fixedTemperatureConstraint, scalarFixedValueConstraint — while the v2 envelope's cpu::fvOptions::read (fvOptions_cpp.cu:119-205) recognises only five, and the messages list four and five respectively.
- **of_feature**: Foam::fv::option runtime selection table — /usr/lib/openfoam/openfoam2412/src/fvOptions/. ofscan counts 46 registered implementations.
- **path**: both
- **size**: Per option type: 50-300 lines. Fixing the two envelope holes above: ~40 lines, host.
- **fixture**: tests/simplefoam_v2_dispatch.sh:138-140 (fvoptions) and :148 (emptyfvoptions) exist but, as noted, the refusal arm does not exercise the envelope check. tests/fvoptions_vs_openfoam.sh, tests/rotordisk_vs_openfoam.sh and tests/actuationdisk_vs_openfoam.sh gate the implemented ones.
- **blocks_tutorial**: none any more. aerofoilNACA0012 (limitTemperature) and angledDuctExplicitFixedCoeff (explicitPorositySource/fixedCoeff + fixedTemperatureConstraint + scalarFixedValueConstraint) are all supported types now — so the comment at gpuRhoSimpleFoam.cu:91-92 ("aerofoilNACA0012, angledDuct both refuse on fvOptions") is itself stale.
- **depends_on / middle steps**:
  - A second, worse problem in the v2 envelope: fixedTemperatureConstraint and scalarFixedValueConstraint PASS cpu::fvOptions::read (fvOptions_cpp.cu:154-199) and then fall into the porosity loop at simpleFoamV2.cu:1387-1408, where their zero D/F pass the rotated-coordinate check and they are installed as a zero-resistance porosity — i.e. the constraint is SILENTLY DROPPED. The envelope must either refuse them on v2 or the run body must apply them.
  - Similarly actuationDiskSource and rotorDiskSource pass the envelope (fvOptions_cpp.cu:136-152) and are only caught later by readFvOptions at simpleFoamV2.cu:1429-1434 — so the envelope's fvOptions blocker is effectively unreachable for four of the eleven types, and the fixture that claims to exercise it (tests/simplefoam_v2_dispatch.sh:138-140, `type actuationDiskSource`) is actually passing on the LATER refusal, not this one.
  - gpuRhoSimpleFoam reads fvOptions only after the mesh, geometry, patches and every field are built (gpuRhoSimpleFoam.cu:377-395), so a case that will refuse pays the whole set-up first — on gasMixing/injectorPipe that is the >12 minutes the file's own comment at :64-67 documents. preflightFvOptions (fv_options.cuh:166) exists precisely for this and is used only by gpuPimpleFoam.cu:176.
  - Adding a type: the framework, the cellSetOption selection and the `== fvOptions(U)` hook are all in place; each new type is its own source term plus a gate.

</details>
<details><summary><b>[DEAD]</b> The header states that the case's p dimensions are checked so a kinematic p [0 2 -2] cannot be run as an absolute pressure — no such check exists anywhere.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:15-17
- **trigger**: Nothing. The refusal is documented but not implemented: the field reader skips the dimensions line entirely (foam_field_reader.cuh:788, "dimensions and other top-level entries are skipped token-by-token") and neither gpuRhoSimpleFoam.cu nor readThermoCoeffs ever looks at it. A simpleFoam case copied to rhoSimpleFoam with p in m2/s2 runs and, exactly as the comment predicts, gives nonsense.
- **of_feature**: Foam::dimensionSet checking on every fvMatrix operation — /usr/lib/openfoam/openfoam2412/src/OpenFOAM/dimensionSet/dimensionSet.C, and the thermo's own p dimensions in basicThermo.C. OpenFOAM catches this structurally; brae has no dimension type at all, which is why an explicit read-time check is the only place it can be caught.
- **path**: host
- **size**: ~60 lines total (reader + a check per driver).
- **fixture**: None. It is cheap to gate: take any of the six tutorials, rewrite 0/p's dimensions to [0 2 -2], assert non-zero exit; the control is the unmodified case running.
- **blocks_tutorial**: none — and that is the point: it blocks nothing at all, including the cases it was written for.
- **depends_on / middle steps**:
  - FieldData&lt;T> must carry the dimension exponents that foam_field_reader.cuh currently discards (~15 lines in the reader plus a 7-int member).
  - A per-solver expected-dimension table: p [1 -1 -2] for rhoSimpleFoam, [0 2 -2] for simpleFoam/pimpleFoam; also worth applying to phi ([1 0 -1] compressible vs [0 3 -1] incompressible), which gpuRhoSimpleFoam already writes with an explicit string at :747-748 and reads back with readPhiIfPresent without checking.
  - No device work and no OpenFOAM behaviour to reproduce beyond the exponents themselves.

</details>
<details><summary><b>[DEAD]</b> The functionObject report ("type X is not implemented, so this functionObject is NOT run") is documented as firing at start-up before anything can refuse — but Time is constructed at the very end of set-up, after every refusal in the file.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:100-103 (the claim) vs :766 (the actual Time construction); the report itself is /home/ghost/cudafoam/brae/src/OpenFOAM/db/Time/brae_time.cuh:143-147
- **trigger**: For a case that refuses — unsupported thermoType (thermo_parse.cuh:38), simulationType (gpuRhoSimpleFoam.cu:331), RAS model (:342), fvOptions (:386), MRF (:412), pressureControl (:250) — the report never fires, because Time is built ~500 lines later at :766. Only `ObjectRegistry timeRegistry;` is declared early (:104), which is not the same object. So exactly the user the comment describes is the one who never gets the report.
- **of_feature**: Foam::Time owns functionObjectList and reads it in its constructor — /usr/lib/openfoam/openfoam2412/src/OpenFOAM/db/Time/TimeIO.C:443,484; functionObjectList::read is /usr/lib/openfoam/openfoam2412/src/OpenFOAM/db/functionObjects/functionObjectList/functionObjectList.C.
- **path**: host
- **size**: ~25 lines: construct Time after controlDict/startFrom resolve, call readFunctionObjects there, setSteps(nSteps) before the loop.
- **fixture**: None. A gate is one line of grep: run a case that refuses on thermoType and assert the log still names the unimplemented functionObjects; the control is the same case with a supported thermo.
- **blocks_tutorial**: gasMixing/injectorPipe is the case this matters for — it ships `abort`, `scalarTransport` and `sampling`, and only scalarTransport is implemented.
- **depends_on / middle steps**:
  - brae_time.cuh already solves this: the owning constructor (brae_time.cuh:227-239) plus readFunctionObjects() (:246-254) plus setSteps() (:314) exist precisely so Time can be built early and told the step count late. gpuSimpleFoam.cu already uses that shape — Time at :212, readFunctionObjects at :278.
  - gpuRhoSimpleFoam must move to the same shape. The obstacle named in the comment (the scalarTransport factory captures DeviceSimpleSolver& and t0) is already handled: the factory resolves mesh/patches/geometry/solver from the ObjectRegistry on its first execute(), so only the start-directory PATH is still passed eagerly (brae_time.cuh:241-245).
  - No device work.

</details>
<details><summary><b>[GENUINE]</b> scalarTransport without a constant `D` (nut-based or alphaD/alphaDt diffusivity) is declined by name and the tracer is not solved.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:668-674
- **trigger**: A `functions` sub-dictionary of type scalarTransport with no `D` entry. Also declines when the div scheme for the tracer cannot be resolved (:676-683). Note this is a noticeIgnored, not a throw — the run continues without the tracer, which is the correct level only because the tracer is diagnostic and does not feed back into the solved fields.
- **of_feature**: functionObjects::scalarTransport::D — /usr/lib/openfoam/openfoam2412/src/functionObjects/solvers/scalarTransport/scalarTransport.C:84-155: constantD_ first, then nutName_ lookup, then alphaD_*turb->nu() + alphaDt_*turb->nut() (incompressible) or alphaD_*turb->mu() + alphaDt_*turb->mut() (compressible), then zero.
- **path**: both
- **size**: ~80 lines: register nut/mu in the ObjectRegistry, add the three branches to the factory, one device kernel reuse for the face diffusivity.
- **fixture**: None specific. gasMixing/injectorPipe's tracer0 uses `D 0.001`, so the supported branch is exercised end to end; there is no case in the tree that trips the refusal.
- **blocks_tutorial**: none — gasMixing/injectorPipe's system/scalarTransport specifies D 0.001 and is honoured.
- **depends_on / middle steps**:
  - ScalarTransportFO already resolves mesh/patches/geometry/solver from the registry and already builds a div+laplacian on the device flux — the whole transport is in place.
  - The nut/mut branch needs the solver to publish a per-cell AND per-boundary-face nut (or mut) to the functionObject through the registry; solver.nut()/nutBoundary() exist (gpuRhoSimpleFoam.cu:756) but are not registered as ObjectRegistry entries the FO can find by name.
  - The compressible branch additionally needs mu = thermo.mu() and mut = rho*nut on the boundary, both of which DeviceThermo carries.
  - The alphaD/alphaDt coefficients are two scalarOr reads.

</details>
<details><summary><b>[GENUINE]</b> div(phi,U) asking for anything outside {upwind, linearUpwind, linearUpwindV, limitedLinear, limitedLinearV, LUST} refuses the rebuilt simpleFoam.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/simpleFoam/simpleFoamV2.cu:476-481
- **trigger**: The first non-{Gauss,bounded,none,numeric} word in the div(phi,U) entry (or in `default` when there is no div(phi,U) entry) is any other scheme — vanLeer, linear, cubic, filteredLinear, SFCD, limitedCubic, etc. HOLE in the trigger: divUScheme (simpleFoamV2.cu:66-95) skips the token "none", so `divSchemes { default none; }` with the div(phi,U) entry written as a REGEX key (which the literal regex at :77 cannot match) yields an empty scheme word, no refusal, and a silent fall-through to upwind at :1483-1489. OpenFOAM would FatalError on the same case.
- **of_feature**: surfaceInterpolationScheme runtime selection — /usr/lib/openfoam/openfoam2412/src/finiteVolume/interpolation/surfaceInterpolation/. ofscan counts 78 registered schemes; six are ported.
- **path**: both
- **size**: ~80-150 lines per scheme (reference + kernel + gate). Closing the `default none` hole: ~10 lines, host.
- **fixture**: tests/simplefoam_v2_dispatch.sh has a vanLeer refusal arm and section 4e asserts the three newer schemes RUN. tests/divschemes_vs_openfoam.sh and tests/linearupwind_vs_openfoam.sh gate the implemented ones against OpenFOAM.
- **blocks_tutorial**: n/a for rhoSimpleFoam.
- **depends_on / middle steps**:
  - Each new scheme is an explicit face-value limiter plus its implicit weights on both the _cpp reference and the device kernel — the pattern is already established by limitedLinear/limitedLinearV/LUST in UEqn.cu and UEqn_cpp.cu.
  - `vanLeer` and `linear` are the two cheapest: linear is pure central weights (no limiter), vanLeer is a one-line limiter function in the same slot limitedLinear occupies.
  - Closing the `default none` hole is host-only: make divUScheme return a sentinel when the entry is genuinely absent and refuse, rather than falling back to upwind — matching OF's own FatalError.
  - The V-variants (limitedLinearV, linearUpwindV) already exist, so the vector-limiter machinery does not need building.

</details>
<details><summary><b>[GENUINE]</b> div(phi,k)/div(phi,epsilon)/div(phi,omega)/div(phi,nuTilda) asking for anything outside {upwind, limitedLinear, linearUpwind} refuses the case.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/simpleFoam/simpleFoamV2.cu:536-542
- **trigger**: The turbulence scalar's own divSchemes entry (resolved regex-aware through readFileExpanded, so `$turbulence` macros are expanded first) names another scheme. HOLE: when the scheme IS linearUpwind, the envelope does not check its NAMED gradient the way it does for div(phi,U) — linearUpwindGradUnsupported (simpleFoamV2.cu:170-231) only ever inspects the div(phi,U) entry. So `div(phi,k) Gauss linearUpwind grad(k);` with `gradSchemes { grad(k) cellLimited Gauss linear 1; }` runs a plain Gauss gradient silently, which is exactly the defect the div(phi,U) version was written to stop.
- **of_feature**: Same surfaceInterpolationScheme table applied to the turbulence transport equations — kEpsilon.C / kOmegaSST.C build fvm::div(alphaRhoPhi, k) with the case's scheme.
- **path**: both
- **size**: Gradient-hole fix ~20 lines host. New schemes: as entry 11.
- **fixture**: tests/simplefoam_v2_dispatch.sh and tests/sst_cuda_vs_openfoam.sh cover the accepted set; there is no fixture asserting a cellLimited grad(k) is refused or honoured.
- **blocks_tutorial**: n/a for rhoSimpleFoam.
- **depends_on / middle steps**:
  - Extending the accepted set is the same work as entry 11, in the turbulence hook rather than the momentum one (simpleFoamV2.cu:1106-1130 reads limitedK/linearUpwindK/twoBykK/boundedK).
  - Closing the gradient hole: generalise linearUpwindGradUnsupported to take the div entry key as a parameter and call it for each turbulence field — ~20 lines, host, and the device already has deviceCellLimitGrad.

</details>
<details><summary><b>[GENUINE]</b> thermoType outside {hePsiThermo | heRhoThermo+perfectGas} x pureMixture x hConst x perfectGas x {sutherland|const} x {sensibleEnthalpy|sensibleInternalEnergy}, or `properties liquid` that is not H2O + heRhoThermo + pureMixture + sensibleInternalEnergy, refuses at start-up.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/thermophysicalModels/thermo_parse.cuh:36-42 (thermoRequire) and :129-133 (the liquid substance refusal)
- **trigger**: Any of: equationOfState rhoConst / perfectFluid / icoPolynomial / Boussinesq / incompressiblePerfectGas; thermo janaf / eConst / hPolynomial; transport polynomial / logPolynomial / icoTabulated; mixture multiComponentMixture / reactingMixture; `properties liquid` naming any substance other than H2O; or `properties liquid` with sensibleEnthalpy. The refusal is precise and names what it found and what is supported, which is the right shape.
- **of_feature**: Foam::fluidThermo::New's runtime selection over the thermo type tuple — /usr/lib/openfoam/openfoam2412/src/thermophysicalModels/basic/fluidThermo/fluidThermo.C, with the components under .../specie/equationOfState/, .../specie/thermo/, .../specie/transport/ and .../thermophysicalProperties/liquidProperties/.
- **path**: both
- **size**: Per equationOfState ~120 lines (host correlation + device branch + gate). janaf ~200. A second liquid substance ~60 lines of coefficients + one validation case. multiComponentMixture: a solver, not a change.
- **fixture**: tests/rho_createfields_vs_openfoam.sh and tests/rho_squarebend_vs_openfoam.sh cover the perfectGas and H2O-liquid paths against OpenFOAM. There is no negative control asserting an unsupported thermoType is refused by name.
- **blocks_tutorial**: none of the six — all are perfectGas+hConst+(sutherland|const) or the H2O `properties liquid` form. It blocks essentially every combustion, multi-species and non-ideal-gas case outside the tutorial set.
- **depends_on / middle steps**:
  - Each equationOfState is rho(p,T) and psi(p,T), plus a CpMCv — the perfectGas pair is thermo_model.cuh; adding one is that file plus a device branch in rhoThermoDevice.cu.
  - IMPORTANT ordering fact: c.rhoThermoType (heRhoThermo) already changes rho TIMING relative to the pressure solve (rho lags by one outer iteration, rho_simple_controls.cuh:27-30, gpuRhoSimpleFoam.cu:328). Any new non-perfectGas EOS lands on that lagged path, so the lag has to be right before a new EOS can be trusted — that is the middle step people miss.
  - janaf thermo needs a 7-coefficient polynomial Cp(T) and its Newton inversion he->T; the Newton inversion machinery already exists for the liquid path (heToT), so this is coefficients plus a device branch, not new algorithm.
  - A second liquid substance is coefficients only — the NSRDS correlation forms in H2OLiquid are general; what is missing is validation, which the refusal message says explicitly.
  - multiComponentMixture is a different order of work entirely (a Yi transport equation per species + reaction) and is not a thermo-parse change.

</details>
<details><summary><b>[GENUINE]</b> An MRF zone naming a cellZone that constant/polyMesh/cellZones does not contain refuses the case.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/simpleFoam/simpleFoamV2.cu:405-410 (envelope) and :668-671 (the throw in the run body)
- **trigger**: constant/MRFProperties names a cellZone absent from the mesh. This is the right refusal and the right reason — brae would find no cells, rotate nothing, and converge to a confidently wrong answer.
- **of_feature**: MRFZone's cellZone lookup — /usr/lib/openfoam/openfoam2412/src/finiteVolume/cfdTools/general/MRF/MRFZone.C:setMRFFaces via mesh.cellZones().findZoneID(); OpenFOAM also errors on a missing zone.
- **path**: host
- **size**: ~10 lines to replicate on gpuRhoSimpleFoam once its MRF refusal is lifted.
- **fixture**: tests/simplefoam_v2_dispatch.sh:124-126 (try_refusal mrf_missing_zone) with the empty-MRFProperties accept-arm at :131-133 as its control. This is the best-shaped fixture in the envelope area.
- **blocks_tutorial**: none.
- **depends_on / middle steps**:
  - Nothing — this is parity, not a port gap. It should be COPIED to the compressible driver, which checks only readMRFProperties(...).active (gpuRhoSimpleFoam.cu:412) and has no zone-existence test at all.
  - The binary cellZones reader is already fixed (readCellZones handles binary polyMesh).

</details>
<details><summary><b>[OVER-BROAD]</b> controlDict endTime is truncated to int before the step count is computed, and a resulting nSteps &lt; 1 refuses the case as 'nothing to run'.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/gpuRhoSimpleFoam.cu:461 (the int cast) and :761-766 (the refusal)
- **trigger**: Any endTime that is not an integer >= 1 relative to the start time — e.g. `endTime 0.5; deltaT 0.1;` (a legal steady run of 5 iterations) truncates to 0 and refuses. deltaT is honoured as a scalar through wc.deltaT() everywhere else, so the truncation is inconsistent with the rest of the time handling and with WriteControl::timeName, which formats fractional times correctly.
- **of_feature**: Foam::Time::run() — /usr/lib/openfoam/openfoam2412/src/OpenFOAM/db/Time/Time.C, `value() &lt; endTime - 0.5*deltaT`; endTime is a scalar and OpenFOAM places no integrality requirement on it for a steady solver.
- **path**: host
- **size**: ~5 lines.
- **fixture**: None. Trivial to gate: a case with endTime 0.5 / deltaT 0.1 must run 5 iterations; the control is endTime 0 which must still refuse.
- **blocks_tutorial**: none of the six as shipped (all use integer endTime with deltaT 1).
- **depends_on / middle steps**:
  - Nothing external: make endTime a scalar and keep the existing std::lround((endTime - tStart)/deltaT) — the rounding already there does the right thing once the input is not pre-truncated.
  - Keep the nSteps &lt; 1 refusal itself: it is a correct guard against `endTime` at or before the restart time and it names the absolute-vs-relative confusion clearly.
  - One subtlety worth preserving: FoamDict::find (foam_dict.cuh:91-111) returns the LAST literal match, which is what makes gasMixing/injectorPipe's duplicated `endTime 0.4; ... endTime 1200;` resolve to 1200 and not refuse. Any change must not reorder that.

</details>
<details><summary><b>[GENUINE]</b> pRefPoint on the OF-mirror compressible path refuses, because brae has no cell-location search.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/rhoSimpleFoam/rhoCreateFields_cpp.cu:60-67
- **trigger**: A closed domain (no p patch fixesValue) whose SIMPLE dictionary gives pRefPoint instead of pRefCell. Note the shipping driver gpuRhoSimpleFoam has no equivalent guard at all: it reads only pRefCell/pRefValue (rho_simple_controls.cuh:68-69), so a case specifying pRefPoint there silently pins cell 0.
- **of_feature**: Foam::setRefCell / findRefCell — /usr/lib/openfoam/openfoam2412/src/finiteVolume/cfdTools/general/findRefCell/findRefCell.C:69-100, which resolves the point with mesh.findCell(pRefPoint).
- **path**: host
- **size**: ~80 lines for a faithful findCell (face-plane walk), ~15 for the refusal to be replicated on gpuRhoSimpleFoam in the meantime.
- **fixture**: None. Cheap: a closed duct with pRefPoint must refuse on gpuRhoSimpleFoam as it does on the _cpp path; the control is the same case with pRefCell, which must run.
- **blocks_tutorial**: none of the six (all have a fixed-value pressure patch).
- **depends_on / middle steps**:
  - A point-in-cell search over the mesh. brae has cell centres (FvGeometry::C()) and face/owner topology, so a correct implementation is either OF's own findCell (walk from the nearest cell centre testing the face planes) or, for a first cut, nearest-cell-centre — but nearest-centre is NOT the same cell OF picks on a stretched mesh, so it would be a silent substitution and must not be shipped as one.
  - Nothing else: pRefCell/pRefValue plumbing already exists on both drivers.

</details>
<details><summary><b>[GENUINE]</b> laplacianSchemes `limited &lt;k>` whose coefficient cannot be parsed, or lies outside [0,1], refuses the case.</summary>

- **citation**: /home/ghost/cudafoam/brae/src/applications/solvers/simpleFoam/simpleFoamV2.cu:365-369 (r.unsupported set) and :467-471 (the blocker)
- **trigger**: Two distinct conditions with different status. (a) coefficient outside [0,1] — this is exact OpenFOAM parity, not a port gap: limitedSnGrad.H:149-155 FatalErrors on the same input. (b) the inner corrected-scheme word is not one of {corrected, uncorrected, orthogonal} — e.g. `Gauss linear limited faceCorrected 0.5`, which OpenFOAM accepts (limitedSnGrad.H:112-123 builds any registered snGradScheme there) and brae refuses. Only (b) is a real limitation, and it is narrow.
- **of_feature**: Foam::fv::limitedSnGrad — /usr/lib/openfoam/openfoam2412/src/finiteVolume/finiteVolume/snGradSchemes/limitedSnGrad/limitedSnGrad.H:100-156; the limiter itself is min(k*|orth|/((1-k)*|corr| + SMALL), 1).
- **path**: both
- **size**: ~5 lines to fix the stale message. Porting another inner snGrad scheme: ~150 lines and its own gate.
- **fixture**: tests/simplefoam_v2_dispatch.sh:186-196 asserts BOTH `limited 0.33` and `limited corrected 0.33` are accepted and announced with their coefficient, and :199-201 asserts `limited 7` is refused. tests/limitedsngrad_vs_openfoam.sh gates the limiter against OpenFOAM.
- **blocks_tutorial**: n/a for rhoSimpleFoam. turbineSiting's `Gauss linear limited corrected 0.33` is the case the parser at :339-361 was widened for and it is now accepted.
- **depends_on / middle steps**:
  - The limiter is already ported and consumed end to end: laplacianScheme() returns limitCoeff (simpleFoamV2.cu:355-357), it reaches in.snGradLimitCoeff at :1500, the momentum/pressure kernels apply it (fvm::laplacianCorrFlux), and the turbulence hook reads the same value at :1077-1081. So `limited 0.33` is honoured, not refused — the blocker text ("Only the uncapped `corrected` is ported") is stale.
  - For (b): a second snGrad correction scheme (faceCorrected, quadraticFit) would have to be ported before the inner-scheme word could be widened. Nothing in the tree needs it today.

</details>

## The forward queue: the 21 holes

Every HOLE above, gathered. These are silent substitutions or unwired guards found live in the
tree at adjudication time -- the same class the six-unwired-guards campaign fixed, and the natural
next campaign.

1. **[createFields]** absence of a coupled-patch guard on the host _cpp createFields
   No cyclic/processor guard exists anywhere in rhoSimpleFoam_cpp.cu or rhoCreateFields_cpp.cu (grep: only comments), and makePatchField at fv_patch_field.cuh:1795-1797 silently builds every coupled type as a ZeroGradient placeholder; the only thing refusing a cyclic case on the host arm today is the T->he whitelist at rhoCreateFields_cpp.cu:464-473 firing by accident of T carrying a `cyclic` entry. A latent silent-substitution hole -- masked for rhoSimpleFoam because T is MUST_READ, and the reason the OVER-BROAD reclassification of the whitelist had to be rejected.
2. **[boundary-factory]** mixedEnergy loses refValue through the reader's type gate
   src/finiteVolume/fields/foam_field_reader.cuh:732 parses refValue only when p.type == "mixed" (refGradient at :727 and valueFraction at :737 are parsed ungated); src/finiteVolume/fields/fv_patch_field.cuh:1726-1735 builds the mixedEnergy MixedPatchField from the never-filled refValues and :937 returns T{} for the empty vector, so the guard at :1728 (valueFraction only) never fires. OF's mixedEnergyFvPatchScalarField derives from mixed (.H:58-60), sets a real refValue = thermo.he(pw, Tw.refValue()) (.C:121) and mixed's write() persists all three entries -- so a gate reading OF-written he (the stated purpose of this path, fv_patch_field.cuh:1497-1512) silently blends toward refValue 0 wherever valueFraction>0. Silent substitution, no refusal.
3. **[boundary-factory]** surfaceNormalFixedValue/uniformNormalFixedValue `ramp` key silently dropped
   No 'ramp' key exists anywhere in src/finiteVolume/fields/foam_field_reader.cuh -- it falls into the unhandled-entry skip at :767-772 -- and the contract comment at fv_patch_field.cuh:334 still says 'any time-ramp/Function1 is ignored'. OF multiplies the patch values by ramp_->value(t) every updateCoeffs (surfaceNormalFixedValueFvPatchVectorField.C:162-165; uniformNormal... .C:162-164, ramp_ read via Function1::NewIfPresent at .C:58-59). This session's aac18cd fix covers only the refValue/uniformValue slot (foam_field_reader.cuh:508-540 marking; refusals at fv_patch_field.cuh:1649-1663), so simpleCar 0.orig/U intakeType1 (`refValue uniform 1.2; ramp table ((0 0)(10 1))`) still builds a frozen 1.2*n with no refusal -- and a ramp ending below 1 converges silently to the wrong inlet even in steady state.
4. **[boundary-factory]** timeVaryingMappedFixedValue accepted path silently diverges from OF (time dirs, mapMethod, offset, setAverage, perturb)
   src/finiteVolume/fields/foam_field_reader.cuh:316-347 (readTimeVaryingMapped) takes only the SMALLEST boundaryData time directory and holds it, and the factory maps nearest-point (fv_patch_field.cuh:1667 comment); mapMethod/offset/setAverage/perturb keys hit the unhandled-entry skip at foam_field_reader.cuh:767. OF's MappedFile.C reads setAverage/perturb/offset (:47-67), validates mapMethod with planar interpolation as the default (:119-128, nearestOnly at :439-441 only when explicitly 'nearest'), and interpolates between bracketing time directories each step. A case with several time dirs, a planar-critical patch, or any of those keys runs silently different from OF with no refusal anywhere.
5. **[boundary-factory]** atmBoundaryLayerInlet{K,Epsilon} silently drop OF's C1/C2 factor
   src/finiteVolume/fields/fv_patch_field.cuh:1866-1869 computes k = Ustar^2/sqrt(Cmu) and eps = Ustar^3/(kap*(zr+z0)) with no sqrt(C1*log((z+z0)/z0)+C2) factor, and the reader parses only kappa/Cmu/d/z0/Uref/Zref/flowDir/zDir for ABL patches (foam_field_reader.cuh:440-506) -- C1/C2 fall to the skip at :767, neither parsed nor refused. OF atmBoundaryLayer.C:70-71 reads C1_(getOrDefault 0.0)/C2_(getOrDefault 1.0) and applies the factor at :236-240 (k) and :250-254 (epsilon). A YGCJ-profile case setting C1/C2 silently gets Richards-Hoxey. The brae comment at :1817 even names the default assumption without enforcing it.
6. **[boundary-factory]** hasCoupledPatches never derived from the mesh
   The flag exists only on the device mirror structs (rhoSimpleFoam.cuh:219, rhoUEqn.cuh:170, rhoPEqn.cuh:127, rhoEEqn.cuh:111) and is set true only by fail-proof tests; nothing anywhere computes it from patch types. On the CUDA path the harness's own pre-check refuses coupled patches first (tests/test_rho_simple_step_cuda.cu:174-177), so those device throws are dead but guarded. On the _cpp whole-case path there is NO check and no flag at all (rhoSimpleFoam_cpp.cuh:145-146 carries only hasMRF/hasFvOptions, and test_rho_simple_step_cpp.cu greps clean for cyclic/coupled): the factory builds cyclic as a zeroGradient placeholder (fv_patch_field.cuh:1795-1797) whose comment justifies the no-op by 'the device skip' -- which the host mirror does not perform -- so a cyclic case pointed at the _cpp harness runs silently uncoupled. Guard fails to fire; silent substitution.
7. **[equations]** MRF declared, device momentum twin (rhoUEqn.cu)
   src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:61 guards `in.hasMRF && !in.mrf`, but rhoSimpleFoam.cu:290 only forwards the flag and no device caller derives it -- tests/test_rho_simple_step_cuda.cu:370 sets only gin.hasMixed -- so an MRF case run through the device gate proceeds rotation-free with both arms agreeing; the checker's 'armed, identical wiring to the host' is false (the host harness derives it at tests/test_rho_simple_step_cpp.cu:221, the device harness derives nothing), and the surveyor's 'harmless' is false for the same reason.
8. **[equations]** unported fvOption, device momentum twin
   src/applications/solvers/rhoSimpleFoam/rhoUEqn.cu:70 guard; the device step gate never reads the case's fvOptions dict -- its only porosity is the synthetic wiring control at tests/test_rho_simple_step_cuda.cu:738-757 -- so hasFvOptions is never derived and a case with any fvOption runs with it silently absent on both arms of a green gate.
9. **[equations]** MRF on the energy equation, device twin
   src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu:148 guards `in.hasMRF`, forwarded at rhoSimpleFoam.cu:366 from a RhoStepInput field no device caller assigns; EEqn.H's fvc::div(MRF.phi(),p) is real and an MRF case on the device path drops it silently rather than being refused.
10. **[equations]** unported fvOption on the energy equation, device twin
   src/applications/solvers/rhoSimpleFoam/rhoEEqn.cu:154 guard, forwarded at rhoSimpleFoam.cu:367 but derived by no device caller; the device energy path also implements no fvOptions.constrain at all, so e.g. angledDuctExplicitFixedCoeff's fixedTemperatureConstraint is dropped silently on CUDA while the host applies it -- the guard that should catch this never fires.
11. **[equations]** non-upwind div(phi,k|epsilon), device closure
   src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.cu:442-451 has the right refusal text but the flag is set only by the fail-proof (tests/test_rho_kepsilon_cuda.cu:492) -- no driver anywhere parses div(phi,k)/div(phi,epsilon) -- and the substitution it names is live and measured: squareBend asks `Gauss limitedLinear 1`, gets upwind, worth ~1.6e-03 at convergence, absorbed by the widened TURB_BOUND 3.0e-3 (tests/test_rho_simple_step_cpp.cu:456-470). Not DEAD (not harmless) and not GENUINE (not armed): a guard that fails to fire over a measured silent substitution.
12. **[equations]** host DarcyForchheimer runs with nu=0 while claiming implementation
   src/applications/solvers/rhoSimpleFoam/rhoUEqn_cpp.cu:223 passes /*nu=*/0.0 with forceDimensions=true into cpu::fvOptions::addSup, whose DF branch is `cd[k] = nu * dd[k] + magU * ff[k]` (src/finiteVolume/cfdTools/general/fvOptions/fvOptions_cpp.cu:339) -- against OF's Cd = mu[celli]*D + (rho[celli]*mag(U))*F (DarcyForchheimerTemplates.C:52-53) the whole Darcy term is dropped and rho is missing from Forchheimer; worse, the host refusal message at rhoUEqn_cpp.cu:98-99 now reads 'explicitPorositySource (DarcyForchheimer and fixedCoeff) IS implemented'. The device refuses this exact branch at rhoUEqn.cu:151-159. Checker confirmed.
13. **[equations]** kEpsilon_cpp assembles upwind whatever fvSchemes names
   src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu contains zero throws and takes no div-scheme parameter; the epsilon and k convection at :317 and :486 are the plain fvm::div overload where the weighted one exists and the momentum path uses it; the cost is measured and absorbed in the gate comment at tests/test_rho_simple_step_cpp.cu:467-470 (~1.6e-03 on squareBend), and the legacy device closure honours limitedLinear (src/cuda/device_kepsilon.cuh:295-297), making the OF-mirror reference the more permissive of the two. Checker confirmed.
14. **[equations]** no host-side coupled-patch refusal in the _cpp mirror
   src/finiteVolume/fields/fv_patch_field.cuh:1796-1798 builds any isCoupledInterfaceType patch as ZeroGradientPatchField (the comment defends it only for the legacy device solver, which appends those faces to the LDU); grep finds no cyclic/coupled throw in rhoUEqn_cpp.cu, rhoEEqn_cpp.cu or rhoCreateFields_cpp.cu (only the setConstraintTypes comment at rhoCreateFields_cpp.cu:449-452), while the device lineage refuses upstream at rhoCreateFields.cu:67-74 -- so the host reference silently solves a periodic mesh with zero-gradient walls. Checker confirmed.
15. **[equations]** `limited 0` mapped to the FULL non-orthogonal correction
   src/applications/solvers/simpleFoam/simpleFoamV2.cu:365-368 maps `limited 0` to limitCoeff=0.0 with corrected left true, and fvm.cuh:145 treats only limitCoeff in (0,1) as limited, so brae runs the FULL uncapped correction; OF's limitedSnGrad.C:59-71 gives limiter = min(k*|snGrad|/((1-k)*|corr|+SMALL),1) = 0 at k=0, i.e. NO correction -- opposite behaviour, silent. The file even contradicts itself: simpleFoamV2.cu:308-309 asserts '0 means no limiter (plain corrected)' while :318-321 correctly states 'limited 0 is uncorrected'. One-line fix as the checker says; must land before the STALE energy refusal is lifted.
16. **[pressure]** pRefValue read with scalarOr default where OF's readEntry is mandatory
   rhoCreateFields_cpp.cu:76 `refValue = dict->scalarOr("pRefValue", 0.0)` versus findRefCell.C:105 `dict.readEntry(refValueName, refValue)` which FatalIOErrors when absent (verified in OF source) -- a case with pRefCell and no pRefValue is refused by OpenFOAM and silently pinned to p=0 by brae. The same hole exists at createFields_cpp.cu:97 and gpuSimpleFoam.cu:424.
17. **[pressure]** V2 device adjustable mask built from fixesValue alone, missing the inletOutlet half
   simpleFoamV2.cu:724 `adjustable.push_back(f.U.boundary[pi]->fixesValue() ? 0 : 1)` omits adjustPhi.C:59's `&& !isA&lt;inletOutletFvPatchVectorField>` -- brae's InletOutletPatchField inherits MixedPatchField::fixesValue()==true (fv_patch_field.cuh:1010, :1130-1142), so an inletOutlet outlet is marked non-adjustable on the device, its outflow lands in fixedMassOut and is never scaled, and deviceAdjustPhi (device_simple.cu:284-296) can throw the continuity fatal on a case OpenFOAM solves. rhoCreateFields.cu:216-225 builds the same mask correctly and documents why both halves matter.
18. **[pressure]** pEqn_cpp adjustPhi scale loop disagrees with its own sum loop on inletOutlet
   pEqn_cpp.cu:124 `if (U.boundary[pi]->fixesValue()) continue;` while the sum loop at :98 uses `fixesValue() && !isInletOutlet()` -- inletOutlet outflow is counted in adjustableMassOut (the massCorr denominator) but skipped by the scale loop, so continuity is not restored; OF's scale predicate is `!fixesValue() || isA&lt;inletOutlet>` (adjustPhi.C:127-133, verified). rhoPEqn_cpp.cu:84 and :121 use the correct predicate in both loops.
19. **[pressure]** rho _cpp references never populate M.faceFluxCorrection
   rhoPEqn_cpp.cu:321-322 and rhoPcEqn_cpp.cu:303-304 negate a vector nothing ever fills (fvm::laplacian at fvm.cuh:53-107 does not populate it), so on the ADMITTED `corrected` path `phi = phiHbyA + pEqn.flux()` silently drops the non-orthogonal correction flux that fvMatrix.C:1688 adds back in OpenFOAM -- while the CUDA twins build it (rhoPcEqn.cu:400-414, rhoPEqn.cu ffc path) and pEqn_cpp.cu:214-226 shows the host fix. The host REFERENCE is behind its own device module.
20. **[envelope]** kOmegaSSTLM accepted on the compressible path despite 'variants stay refused' comment
   src/applications/solvers/common/turbulence_setup.cuh:249-250 sets ctl.sst=true for kOmegaSSTLM, so the guard at gpuRhoSimpleFoam.cu:341-346 (!ctl.sst) passes it against its own comment; the driver builds DeviceSimpleSolver with only k/eps/nut (gpuRhoSimpleFoam.cu:464-467; ReThetat/gammaInt default nullptr, device_simple_foam.cuh:76-77), so device_simple_foam.cu:595-601 never populates the LM buffers, yet :853 hands gammaIntEff_.data() to the SST kernel and :870-874 runs deviceKOmegaSSTLMCorrect on the empty buffers -- and even populated, the transition transport carries no alpha*rho weighting (OF kOmegaSSTLM.C:529-541, :554-580). Fix is ~1 line: widen the guard with ctl.lm.
21. **[envelope]** fixedFluxPressure unguarded on the shipped brae_simpleFoam/brae_rhoSimpleFoam binaries
   Amended by this session but the substance stands: src/finiteVolume/fields/fv_patch_field.cuh:1588-1601 now builds the real FixedFluxPressurePatchField for every caller, but its refusal lives only in the host coefficient hooks (requireUpdated, fv_patch_field.cuh:623-641), which the shipped device path never calls -- device_simple_foam.cu contains no deviceConstrainPressure and no host *Coeffs() call, and DeviceBoundary::snGradMask (built at device_boundary.cuh:70) is consumed only by the mirror modules (pEqn.cu:262, rhoPEqn.cu:273, rhoPcEqn.cu:255) -- so the shipped binaries run ffp as a FROZEN construction-time gradient (file entry or zero, fv_patch_field.cuh:1590-1599): silently right on non-assignable-U walls, silently wrong opposite assignable-U patches or under MRF, with no refusal firing.
