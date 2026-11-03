# simpleFoam

Steady-state incompressible solver (SIMPLE / SIMPLEC). This is the solver brae ships today, a faithful GPU port of
OpenFOAM v2412's `simpleFoam`, fully device-resident and validated cell-by-cell against OpenFOAM. If your case uses
only what is listed here, it should run and match OpenFOAM to under 1%.

[<- back to all solvers](../../README.md#-will-it-run-my-case)

## At a glance

| | |
|---|---|
| **Algorithm** | steady incompressible SIMPLE / SIMPLEC |
| **Turbulence (RAS)** | laminar, k-epsilon, k-omega SST, Spalart-Allmaras, realizable k-epsilon, k-omega SST-LM (transition) |
| **Physics** | MRF, cyclic / cyclicAMI, porosity, rotor-disk & fvOptions, wall functions, force/moment coefficients |
| **Schemes** | Gauss upwind / linearUpwind / limitedLinear / vanLeer / MUSCL, bounded div, corrected (non-orthogonal) laplacian |
| **BCs** | the standard incompressible set (fixedValue, zeroGradient, inletOutlet, freestream, slip, symmetry, wall functions, ...) |
| **I/O** | standard OpenFOAM case in, standard time directories out; ASCII & binary mesh |

Legend: ✓ supported & validated · ✗ not yet.

## Solver & algorithm

| feature | status |
|---|---|
| SIMPLE / SIMPLEC pressure-velocity coupling | ✓ |
| Momentum: `div(phi,U) − laplacian(nuEff,U) − div(devReff)` (full incompressible stress) | ✓ |
| Pressure reference handling (closed domains, all-Neumann, `fixedFluxPressure`) | ✓ |
| Under-relaxation, `residualControl` convergence | ✓ |
| Non-orthogonal correction in the Laplacian (`corrected`) | ✓ |
| `nNonOrthogonalCorrectors > 0` (extra pressure correctors) | ✓ (with a `corrected` laplacian scheme) |

## Turbulence (RAS), 5 models, all validated vs OpenFOAM v2412

| model | notes |
|---|---|
| laminar | machine-precision vs OF |
| **k-epsilon** | channel / pitzDaily / backward-facing-step, sub-1% |
| **realizable k-epsilon** | coefficients read from dict |
| **k-omega SST** | pitzDaily settled U 0.10% / p 0.18% / k 0.09% |
| **Spalart-Allmaras** | NACA0012 airfoil; lift within ~1% of OF |
| **k-omega SST-LM** (Langtry-Menter transition) | transitional airfoil; transition captured |
| plain k-omega, RNG k-epsilon, LaunderSharma, RSM, LES | ✗ not yet |

Model coefficients are read from `turbulenceProperties` (defaults match OpenFOAM exactly). Wall functions:
`nutkWallFunction`, `nutUSpaldingWallFunction`, and the standard k/epsilon/omega wall functions.

## Boundary conditions

The standard incompressible set: `fixedValue`, `zeroGradient`, `noSlip`, `slip`, `symmetry` / `symmetryPlane`,
`inletOutlet`, `outletInlet`, `pressureInletOutletVelocity`, `freestream` / `freestreamPressure`,
`fixedFluxPressure`, `uniformFixedValue`, `atmBoundaryLayerInlet*`, and the wall-function BCs. `cyclic` and
`cyclicAMI` are supported as coupled interfaces. Unsupported BCs are rejected by name at start-up.

## Discretization schemes (`fvSchemes`)

| category | supported |
|---|---|
| `div(phi,U)` | Gauss upwind, linearUpwind, linearUpwindV, LUST, limitedLinear, vanLeer, MUSCL, `bounded` |
| `div(phi,k\|epsilon\|omega\|nuTilda)` | upwind, limitedLinear |
| gradient | Gauss linear, cellLimited |
| laplacian / snGrad | Gauss linear `corrected` (non-orthogonal), `limited` |
| interpolation | linear |

The vector-limited `limitedLinearV` falls back to its scalar form; `leastSquares` gradient and `cellLimited`-corrected
combinations are not yet available.

The turbulence scalars' gradients come from each field's own `grad(k)`, `grad(epsilon|omega)` entry, falling back to
`default` as OpenFOAM's lookup does: kOmegaSST's CDkOmega, the corrected laplacian's non-orthogonal correction and
limitedLinear's limiter take them, and each `linearUpwind` takes the gradient its own div entry names. A two-equation
case whose `grad(k)` or `grad(epsilon|omega)` resolves to anything but Gauss linear, optionally `cellLimited` (minmod),
is refused. Open: CDkOmega's two gradients do not include the coupled faces of a cyclic or AMI patch (the transport
equations' gradients do); no gated case has both.

## Physics & function objects

| feature | status |
|---|---|
| MRF (multiple reference frames) | ✓ validated (laminar + turbulent) |
| cyclic / cyclicAMI interfaces (incl. turbulence transport across) | ✓ |
| Porosity (Darcy-Forchheimer) | ✓ |
| `fvOptions` (rotorDiskSource, meanVelocityForce, velocityDamping, ...) | ✓ (subset) |
| `forces` / `forceCoeffs` (Cd, Cl, Cm) | ✓ validated (airfoil Cl within ~1% of OF) |
| yPlus, wallShearStress, Q, vorticity, mag(U) writers | ✓ |
| general `fvOptions` framework, arbitrary function objects | ✗ (subset only) |

## Mesh & I/O

- Reads OpenFOAM `polyMesh`, **ASCII or binary**, gzipped (`*.gz`) or plain.
- Reads stock dictionaries and `$macro` expansions.
- Writes standard time directories, post-process with `paraFoam` / `postProcess` exactly as usual.
- True 3D and 2D (empty-patch) meshes; validated on structured, graded, and snappyHexMesh meshes.

---

If you hit an unsupported model, BC, or scheme, brae stops with a clear message naming it, it never silently
substitutes something else. See [roadmap.md](../roadmap.md) for what is on the roadmap.
