# rhoSimpleFoam

Steady-state **compressible** solver (SIMPLE / SIMPLEC), subsonic and transonic. A faithful GPU port of OpenFOAM
v2412's `rhoSimpleFoam`, fully device-resident. All six tutorials OpenFOAM ships run **as shipped**.

[<- back to all solvers](../../README.md#solvers)

## At a glance

| | |
|---|---|
| **Algorithm** | steady compressible SIMPLE / SIMPLEC, subsonic and transonic |
| **Thermo** | hePsiThermo, heRhoThermo · perfectGas · hConst · sutherland, const · liquid (NSRDS correlations) |
| **Energy** | sensibleEnthalpy, sensibleInternalEnergy |
| **Turbulence** | laminar, kEpsilon, realizableKE, kOmegaSST, Spalart-Allmaras, generalizedNewtonian |
| **fvOptions** | explicitPorositySource, limitTemperature, fixedTemperatureConstraint, scalarFixedValueConstraint |
| **I/O** | standard OpenFOAM case in, standard time directories out |

## Validation

Against real OpenFOAM, iteration by iteration, from the same start. Worst field, worst of iterations 1-3, relative:

| tutorial | cells | host arm | CUDA arm |
|---|---:|---:|---:|
| aerofoilNACA0012 | 16,000 | 2.8e-12 | 2.8e-12 |
| angledDuctExplicitFixedCoeff | 28,000 | 1.6e-10 | 1.2e-10 |
| squareBend (transonic, Mach 0.999) | 112,000 | 5.5e-10 | 6.6e-11 |
| squareBendLiq | 112,000 | 1.9e-12 | 1.9e-12 |
| squareBendLiqNoNewtonian | 112,000 | 5.8e-13 | 5.5e-13 |
| injectorPipe | 74,650 | run and gated | run and gated |

Those are **operator-level**: both codes given the identical state, advanced one iteration. Run both to convergence
and small differences compound through the nonlinear iteration — squareBend reads ~8e-03 on p at iteration 100.

## Two arms, both shipped

The port carries a host reference and a CUDA implementation of every component, and both are runnable:

```bash
BRAE_RHOSIMPLEFOAM_MIRROR=cuda brae -case yourCase   # device modules
BRAE_RHOSIMPLEFOAM_MIRROR=cpu  brae -case yourCase   # host reference
```

The host arm exists because it is the oracle the CUDA arm is validated against — a readable transcription of the
OpenFOAM text it quotes, gated against OpenFOAM's own intermediate fields rather than against the other arm.

## Performance

One GH200 against all 64 Grace cores, 100 SIMPLE iterations, solver wall only:

| case | cells | brae | OpenFOAM 64c | ratio |
|---|---:|---:|---:|---:|
| aerofoilNACA0012 | 10,000,000 | 108.2 s | 916.7 s | **8.47×** |
| aerofoilNACA0012 | 1,024,000 | 11.1 s | 44.5 s | 4.00× |
| squareBendLiq | 896,000 | 6.8 s | 8.4 s | 1.24× |
| squareBend | 112,000 | 1.6 s | 1.8 s | 1.14× |
| squareBendLiq | 112,000 | 1.7 s | 1.6 s | 0.92× |

Below ~10<sup>5</sup> cells a GH200 is not the right tool. Full method: [performance](../performance.md).

## Schemes

| block | supported |
|---|---|
| `divSchemes` U | upwind, linearUpwind, linearUpwindV, LUST, linear, limitedLinear, limitedLinearV |
| `divSchemes` h/e, K/Ekp | `Gauss upwind`, `Gauss linearUpwind <grad>`, `Gauss limitedLinear <k>` |
| `divSchemes` k/epsilon/omega | upwind, linearUpwind, limitedLinear |
| `laplacianSchemes` | `orthogonal`, `corrected`, `limited <psi>` |
| `snGradSchemes` | read separately from `laplacianSchemes` — they govern different operators |
| `gradSchemes` | Gauss linear, leastSquares, cellLimited |

`laplacianSchemes` and `snGradSchemes` are two blocks for two operators: an `fvm::laplacian` entry carries its own
snGrad scheme, while `fvc::snGrad` takes the `snGradSchemes` block, which is optional and defaults to `corrected`.
brae honours both independently.

## Not supported yet

Each of these **stops at start-up, named** rather than running something else:

- turbulence models outside the list above, and `RAS { turbulence off; }` on anything but kEpsilon
- `nut` wall functions other than `nutkWallFunction`
- `kOmegaSSTLM` on the compressible path
- fvOptions outside the four listed
- tilted symmetry planes (the segregated model cannot carry them)
- coupled patches on the mirror `createFields`
- `limited 0` laplacians — a third regime that is neither `orthogonal` nor `corrected`
- multi-GPU / distributed

## The default driver (`brae -case`)

Without `BRAE_RHOSIMPLEFOAM_MIRROR`, `brae -case` runs the pre-mirror driver (`gpuRhoSimpleFoam.cu`). It
stops on a diverged run, writes `alphat`, and refuses by name, at start-up and under `brae check --setup`,
every configuration it would compute differently from OpenFOAM.

### Refused configurations

| configuration | why | what to do |
|---|---|---|
| `codedFixedValue`, `codedMixed`, `fixedMean`, `fanPressure` on U, p, T, k, epsilon/omega or nut | built from the file `value` and never updated, where OpenFOAM recomputes them every step | not ported on this driver |
| `uniformFixedValue` with an `expression` uniformValue | the patch stays at its `value`; OpenFOAM evaluates the expression every step | on T, run with `BRAE_RHOSIMPLEFOAM_MIRROR=cuda`, which evaluates it; elsewhere, a fixed value |
| a `rotorDisk` fvOption | read and never applied | not ported to the compressible solver |
| an fvOption whose `timeStart`/`duration` window does not cover every iteration of the run | the driver applies fvOptions on every iteration; OpenFOAM switches them by iteration time | remove the window, or run only the iterations inside it |
| `laminar { model Maxwell; }` | the viscoelastic stress is not wired on this driver | not ported |
| an `alphat` patch other than `calculated` or `compressible::alphatWallFunction` (constraint types aside), `compressible::alphatJayatillekeWallFunction` included | the driver computes `alphat = rho*nut/Prt` on every face | use `calculated` or `compressible::alphatWallFunction` |
| a RAS case with no `0/alphat` | OpenFOAM reads it MUST_READ | add it |
| `SIMPLE/transonic` or `SIMPLE/consistent` that is not a Switch word (both drivers) | OpenFOAM stops on it | true/false, yes/no, on/off, y/n, t/f, 1/0 |
| `solvers/<e or h>` `maxIter` other than 1000 or `minIter` other than 0 | the energy solve does not take either entry yet | remove them, or run the mirror, which honours both |
| p with no fixed-value patch and no `pRefCell`/`pRefPoint`, no `pRefValue`, a `pRefCell` outside the mesh, or a `pRefPoint` in no cell | OpenFOAM's `setRefCell` stops on each | give the entry; `pRefPoint` is located as OpenFOAM's `findCell` does |
| `controlDict` without `deltaT` or `endTime` | OpenFOAM stops (or, with no `stopAt` and no `endTime`, runs nothing) | add them |
| a missing `molWeight`, `Cp`, `Hf`, `mu` and `Pr` (const), or `As` and `Ts` (sutherland) in `thermophysicalProperties` (both drivers) | OpenFOAM reads each with a mandatory lookup | add the entry |
| `fixedFluxPressure` on p (unchanged) | this driver never runs constrainPressure | the mirror |

A run that diverges stops with `solution diverged`, naming the residual or the field, and writes no time
directory holding a NaN, an Inf or a negative temperature (`BRAE_ALLOW_NONFINITE=1` runs on regardless).

### Behaviour changes (2026-11-21)

- **Refused now** (they used to run, silently differently from OpenFOAM): every configuration in the table
  above but the last. Of the six v2412 tutorials on the default driver:
  - **gasMixing/injectorPipe** was already refused, at its first iteration (`grad(k)`/`grad(epsilon)`
    `leastSquares`, which the legacy kEpsilon closure does not compute); it now stops at start-up, on the
    `minIter 1` its `solvers/"(U|e|k|epsilon)"` entry sets, which the energy solve cannot take yet. It runs on
    `BRAE_RHOSIMPLEFOAM_MIRROR=cuda` and `=1`, as before.
  - **squareBendLiq** was already refused at start-up (its coded `massFlowRate`); it now stops one check
    earlier, on its wall-T `expression`. It runs on the mirror, as before.
  - aerofoilNACA0012, angledDuctExplicitFixedCoeff, squareBend and squareBendLiqNoNewtonian run as before: no
    stock tutorial that ran on the default driver is refused now.
- **The energy equation is no longer relaxed by a default 0.7** when `relaxationFactors/equations` names
  neither the energy field (`h` or `e`, whichever the case solves, regex keys included) nor `default`; the
  other energy name no longer counts. OpenFOAM does not relax the equation then. No v2412 tutorial and no
  brae fixture is affected (they all name their own field). Gates: `rho_energy_relax_vs_openfoam` (U, h and e solved
  with GaussSeidel in both codes, so only the relaxation differs) and `rho_default_energy_shipped_vs_openfoam` (the
  fixture as shipped, whose DILU brae's default driver runs as a diagonal preconditioner).
- **`pRefPoint` is honoured** (it pinned cell 0). Gate: `rho_default_controls_vs_openfoam`.
- **`endTime` is a scalar** (it was truncated to an integer): `endTime 4.7` runs 5 iterations, as OpenFOAM.
- **`transonic true|on|y|t|1` and `consistent true|on|y|t|1`** run the branch they name (only `yes` did).
- **Written files.** `alphat` is written (a RAS case), with the start file's patch types and any non-default
  `Prt`, so OpenFOAM can restart from brae's output and brae's own restart keeps each wall's Prt. Every field,
  `phi` included, is written at `writePrecision` (OpenFOAM's default 6), where the driver wrote 12 digits
  (`phi` 17) whatever the case said. The solution is unchanged: a case written at `writePrecision 12` writes
  the same bytes as before, `phi` excepted. Gate: `rho_default_restart_vs_openfoam`.
- **Divergence stop**, as above. Gate: `rho_default_nonfinite_refusal`, which provokes every stop with a controlled
  injection (`BRAE_TEST_CORRUPT`) rather than a case that diverges on one GPU and not on another.

## Known limitation

On a **non-orthogonal mesh**, iteration-1 velocity sits 2.4e-04 to 3.1e-03 from OpenFOAM, against 1.9e-09 on an
orthogonal one. Scheme-independent and not yet localised — the validation table above is orthogonal-mesh figures.
