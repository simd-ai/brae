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

## Known limitation

On a **non-orthogonal mesh**, iteration-1 velocity sits 2.4e-04 to 3.1e-03 from OpenFOAM, against 1.9e-09 on an
orthogonal one. Scheme-independent and not yet localised — the validation table above is orthogonal-mesh figures.
