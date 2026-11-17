# mgpu_oracle: OpenFOAM v2606 serial vs `-parallel` reference for multi-GPU interFoam

Small 3D interFoam cases, OpenFOAM's own decomposed (`mpirun -np N interFoam -parallel`) results, per-rank dumps of
the interFoam operators, and a comparator that maps `processorN` data onto the serial mesh. The later multi-GPU brae
work gates field by field against this.

## What is installed (checked on this host)

* `/usr/lib/openfoam/openfoam2606`: ESI OpenFOAM **v2606**, binaries and libraries only
  (`platforms/linux64GccDPInt32Opt`, `WM_MPLIB=SYSTEMOPENMPI`, `/usr/lib64/openmpi`): `interFoam`, `decomposePar`,
  `reconstructPar`, `blockMesh`, `setFields`, `topoSet`, `checkMesh` all run. **No headers, no `wmake`.**
* `/tmp/OpenFOAM-v2606.tgz`: the matching v2606 source tree (api 2606, patch 0). `setup.sh` unpacks it to
  `$ORACLE_ROOT` (default `/tmp/brae-oracle`), points its `platforms/` at the installed binaries,
  builds `wmake`'s own tools, makes the `lnInclude` directories and builds `tools/dumpInterFoam` with `wmake` against
  the installed libraries (about 2 minutes). The installed `bashrc` is not used (it needs the missing `wmake`); the
  generated `$ORACLE_ROOT/of_env.sh` sources the source tree's and overrides `FOAM_USER_APPBIN` (the stock bashrc
  would put it under `$HOME/OpenFOAM`).
* The v2406 source in a container overlay is irrelevant; the tests/*.sh gates use `/usr/lib/openfoam/openfoam2412`,
  which does not exist on this host.

## Files

| file | what |
|---|---|
| `gen_cases.py` | writes `cases/<case>/{0.orig,constant,system}`; the cases are committed |
| `cases/tank` | sealed 0.2x0.1x0.2 m box, 48x24x48 = 55k cells, `constantAlphaContactAngle` theta0 0 on every wall, pool + water column by `setFields`, `pRefPoint`, p_rgh GAMG/DIC |
| `cases/tankGraded` | same box sheared 0.05 m in x and graded 4/0.4/3 (non-orthogonality avg 11.5 deg, max 14 deg), theta0 60 (walls) / 90 (lid), p_rgh PCG/DIC, `nNonOrthogonalCorrectors 1`, `limited corrected 0.5` |
| `cases/open` | open-top 0.4x0.1x0.2 box, 64x16x40 = 41k cells, atmosphere patch (`totalPressure`, `pressureInletOutletVelocity`, `inletOutlet`), theta0 30, no reference cell |
| all cases | MULESCorr yes, nLimiterIter 5, nAlphaCorr 2, momentumPredictor no, nCorrectors 3, laminar, `adjustTimeStep yes` maxCo/maxAlphaCo 0.3, three cellZones (topoSet) with `volFieldValue`, `probes`, `fieldMinMax`, `runTimeControl` function objects |
| `setup.sh` | one-time build of the dump solver (see above) |
| `run_oracle.sh <case> <np> [scotch\|simple\|hier] [steps]` | one complete run into `/tmp/mgpu_oracle/<case>_np<N>[_<method>][_tight]` |
| `run_matrix.sh` | all cases x np 2/4/8 + simple/hier 8 + tight variants |
| `decomp_stats.py <dir>` | cells / interface faces / neighbours per rank |
| `compare_fields.py A B` | the comparator (below) |
| `foamio.py` | numpy-only OpenFOAM reader (ascii + binary, fields, addressing) |
| `selftest.sh` | checks the whole chain on `tank` |

## Running

```
tools/mgpu_oracle/setup.sh                          # once
tools/mgpu_oracle/run_oracle.sh tankGraded 4        # scotch, 20 steps
tools/mgpu_oracle/run_oracle.sh tank 8 simple       # simple (2 2 2): a DGX-1 cube
TIGHT=1 tools/mgpu_oracle/run_oracle.sh open 8      # every linear solver to 1e-12, relTol 0
OUT_ROOT=/tmp/brae-oracle-runs tools/mgpu_oracle/run_matrix.sh
tools/mgpu_oracle/selftest.sh
```

One run makes (each is 5 to 60 s): `mesh/` (blockMesh, topoSet, setFields, checkMesh), `serial/` and `par/` to step
`steps` (writes at step 1 and `steps`, operator dumps at `steps`), `serial_d1/` and `par_d1/` stopped at step 1 with
dumps at step 1, `par_rec/` (OpenFOAM's `reconstructPar` of `par/`) and `report.md`. About 600 MB per run: `/tmp` is
RAM, delete runs you are done with or set `OUT_ROOT`.

Why serial and parallel stop on the same step: the solver is `tools/dumpInterFoam` (OpenFOAM's interFoam plus writes; with no BRAE_* variable set its `alpha.water`, `U`, `p_rgh`, `p`, `phi` at t=0.005 are byte-identical to the installed `interFoam`'s on `cases/tank`) and the runner sets

| env | effect (dumpInterFoam only) |
|---|---|
| `BRAE_STOP_ITER=n` | `runTime.writeAndEnd()` after step n |
| `BRAE_WRITE_ITERS=1,n` | `writeNow()` at those step indices (the controlDict writes never) |
| `BRAE_DUMP_ITER=n` | the existing single-step dumps (`UEqnDiag/Upper/Lower`, `HbyA`, `rAUf`, `phiHbyA`, `stf`, ...) |
| `BRAE_ORACLE_ITERS=n,m` | the oracle dumps below (a comma list) |

All dumps are ordinary `*.dump` fields in the time directory, so in `-parallel` every rank writes its own copy under
`processorN/<time>/` and `cellProcAddressing` / `faceProcAddressing` map them to the serial mesh.

## Oracle dumps (BRAE_ORACLE_ITERS)

| name | content |
|---|---|
| `oAlphaPhiUD`, `oAlphaPredictor` | semi-implicit upwind predictor flux and alpha |
| `oPhir/oPhic/oAlphaPhiUn/oAlphaPhi10Pre/oAlphaPhiCorrUnlim/oAlphaPreCorr` + `a<k>` | per alpha corrector k: compression flux, high-order flux, upwind flux, unlimited MULES correction flux, alpha before the correction |
| `oAlphaPhiCorrLim`, `oAlphaCorr` + `a<k>` | the limited correction flux and alpha after it. **MULES lambda = oAlphaPhiCorrLim/oAlphaPhiCorrUnlim** face by face (`MULES::correct` scales the flux by lambda in place; nothing else exposes it) |
| `oNHatf`, `oSigmaK`, `oAlphaEnd`, `oAlphaPhi10End`, `oRhoPhi`, `oRhoEnd`, `oPhiEntry` | what the alpha step hands to momentum/pressure (sigmaK = sigma*K; K = sigmaK/sigma) |
| `oPrghEqnc<k>n<j>{Diag,Source,Upper,Lower,IntCoeffs,BouCoeffs}` | p_rgh matrix as handed to the solver, corrector k, non-orthogonal iteration j. `IntCoeffs/BouCoeffs` hold the per-patch coefficients in the patch values: for `processorN_to_M` patches these are the interface coefficients |
| `oPrghPre`, `oDivPhiHbyA` + `c<k>n<j>` | p_rgh before the solve and `div(phiHbyA)` |
| `oPhi`, `oU`, `oPrgh`, `oP` + `c<k>` | phi, U, p_rgh, p after pressure corrector k |
| (existing) `UEqnA*`, `HbyA*`, `rAU*`, `rAUf*`, `phiHbyA*`, `stf*`, `UEqnDiag/Upper/Lower/Source*` | UEqn A and H (H = HbyA/rAU), face fluxes, UEqn matrix |

Caveats of the dumps: a field written with `nNonOrthogonalCorrectors 1` by the older gates (`rAUf*`, `phiHbyA*`) is
that of the last non-orthogonal pass; the `oPrghEqn*`/`oDivPhiHbyA` names carry `n<j>`. `UEqnDiag.dump` does not
contain the cut faces in `-parallel` (they are in the processor patches' internalCoeffs, which UEqn does not dump):
compare `UEqnA` instead.

## compare_fields.py

```
compare_fields.py A B [--times all|last|t1,t2] [--fields a,b] [--skip REGEX] [--dumps] [--tol X]
                      [--tol-field name=X] [--ptfloor F] [--coherence] [--csv FILE]
```

A and B are each a global case (serial, `reconstructPar` output, brae single-GPU output) or a decomposed case
(`processorN/`, `decomposePar` output, brae multi-GPU output). Decomposed trees are mapped to the global mesh through
`cellProcAddressing` and `faceProcAddressing`: oriented surface fields (fluxes, `oriented oriented;` in the file) are
sign-corrected on flipped faces, matrix `Upper/Lower` pairs are swapped on them, cut faces take their value from the
processor patch; the diagonal of a matrix dump gets the processor patches' internalCoeffs added back (it is how
lduMatrix applies them), and `|processor BouCoeffs/IntCoeffs|` is compared with `|Upper|` of the other tree on the cut
faces. Times pair by nearest value. Rows: max abs, `maxrel` = max abs / max|A| over that field's internal and boundary
values, `l2rel` = L2(diff)/max(L2(A), max|A|), `ptrel` pointwise relative with a floor of `--ptfloor` x max|A|.
`--coherence` additionally reports how much the two sides of each processor interface disagree.
Exit status 1 when `--tol` is exceeded. Boundary values of patches that store none (e.g. `zeroGradient`) are NaN and
ignored.

Comparing brae against the oracle: point `A` at the oracle's `serial/` (or `par/`) and `B` at brae's case directory
(global) or its `processorN` tree. For operator-level gates use the `*_tight` runs: with solvers converged to 1e-12 the
serial/parallel difference is round-off, with G4's loose p_rgh settings (`relTol 0.05`) it is the partial-convergence
difference (the DIC preconditioner is block-wise in parallel) and nothing brae does can be tighter than that.

## Findings the gates must know about (numbers in `golden/`)

* **Mapping validated**: with every linear solver at 1e-12 (`TIGHT=1`) step 1 of serial vs `-parallel` agrees to
  <= 2e-9 (alpha to 1e-14), all p_rgh matrix coefficients (diag with the interface part folded back, upper, source) to 1e-14 and
  the processor interface coefficients equal `|upper|` of the serial matrix to 1e-14. After 20 steps the floor is
  alpha 3e-9..3e-8, p_rgh 2e-9..8e-9, U 4e-7..3e-6, phi 3e-7..1e-6 (max|diff|/max|field|) for np 4 and 8 (scotch) on all three cases.
* **With the cases' G4-style loose solvers** (`p_rgh relTol 0.05`) the floor is the partial-convergence difference and is
  much larger: step 1 U 2e-6..1e-4 (tank/open, GAMG; `golden/error_floors.md`), up to O(1) on the graded PCG case; after 20 steps
  U up to 4e-3 (open np=8; tank np=8 8e-4) and 0.1..0.3 on the graded case. A brae np=k vs np=1 comparison on such settings cannot be tighter than that;
  gate operators on `*_tight` runs, and gate the loose runs on conservation/qualitative measures.
* `fvMatrix::setReference` uses the local diag, so the reference cell's diag/source in `-parallel` differs from serial when the
  cell touches a processor interface (harmless for the solution; `oPrghEqn*Diag/Source` of the sealed cases differ there).
* `UEqnDiag.dump` lacks the cut faces in `-parallel`; compare `UEqnA`.
* scotch decompositions are not reproducible run to run (cut faces 2304 vs 2448 for tank np=4): record the
  `processorN` tree you compared against, do not regenerate it. `simple`/`hierarchical (2 2 2)` are deterministic.
* A decomposed tree stores each cut face's flux on both sides with opposite sign; the comparator checks they are
  consistent (`--coherence`).
