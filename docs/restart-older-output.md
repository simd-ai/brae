# Restarting from time directories written by an older brae

Since pull request #52, brae writes four boundary entries the way OpenFOAM v2412 writes them. These are written by the
field writer that simpleFoam, pimpleFoam, rhoSimpleFoam and rhoCentralFoam share. Older brae lost part of these
entries when it wrote a time directory. This page says what happens when you restart from such a directory, and how to
restore what was lost from the case's own `0/` files. Output written from now on needs none of this.

| in a time directory an older brae wrote | brae now | OpenFOAM v2412 | what to do |
|---|---|---|---|
| `totalPressure` with p0 under `inletValue` | reads it as p0, with a notice, and writes `p0` from then on | stops: "Required entry 'p0' missing" | nothing for brae; for OpenFOAM, rename `inletValue` to `p0` |
| `uniformTotalPressure` with `inletValue`, and its `p0` table gone | stops: "... is a uniformTotalPressure with no `p0` entry" | stops: "Missing or invalid Function1 entry: p0" | restore `p0` from `0/` ([below](#uniformtotalpressure)) |
| `fanPressure` with `inletValue`, and its fan curve and direction gone | stops: "fanPressure on patch ... has no `p0` entry" | stops: "Required entry 'p0' missing" | restore the entries from `0/` ([below](#fanpressure)) |
| `phi` typed `calculated` on a `symmetry`, `symmetryPlane` or `wedge` patch | reads it | stops: "inconsistent patch and patchField types" | for OpenFOAM, retype the patch ([below](#phi-on-a-constraint-patch)) |

Make these edits by hand, in a text editor, and touch only the patch named in the message. Avoid `foamDictionary` here,
for two reasons:
- It rewrites the whole file, and without `-precision 17` it rounds every value to six digits.
- It writes a list of equal values in OpenFOAM's compact form `25{10}`, which brae does not read yet.

## uniformTotalPressure

The older writer kept a single value of p0 under `inletValue` and dropped the definition, a table or other Function1.
The original definition is still in the case's `0/p`. The steps below use `0.01/` as the restart time.

1. Open `0/p` and find the patch. Copy every entry in it except `type` and `value`: `p0`, its `p0Coeffs` dictionary if
   it has one, and `rho`, `psi` or `gamma` if they are set.
2. Open `0.01/p` and find the same patch. Replace its `inletValue` line with what you copied. Keep `type` and `value`.
3. Restart.

pimpleFoam's `RAS/TJunction`, before:

```
    inlet
    {
        type            uniformTotalPressure;
        inletValue      uniform 10;
        value           nonuniform List<scalar> 25 ( ... );
    }
```

after, with `p0` copied from `0/p`:

```
    inlet
    {
        type            uniformTotalPressure;
        p0              table
        (
            (0 10)
            (1 40)
        );
        value           nonuniform List<scalar> 25 ( ... );
    }
```

Restored this way, a TJunction write from before the change restarts in brae and continues exactly as the
uninterrupted run. The fields are identical ten steps later, and OpenFOAM v2412 starts from the same file.

If the original `0/p` is gone, write `p0` as the case defined it. The `inletValue` holds only one value of it, and a
restart that held p0 there would run a different boundary condition.

## fanPressure

The older writer kept p0 under `inletValue` and dropped the fan curve and its direction.

1. Open `0/p` and find the patch. Copy every entry except `type` and `value`:
   - `direction`;
   - `fanCurve`, as a dictionary, or as `fanCurve <type>;` with `fanCurveCoeffs`;
   - `p0`;
   - `rho`, `psi` or `gamma`, if set.
2. In the restart time's `p`, replace the patch's `inletValue` line with them. Keep `type` and `value`.
3. Restart.

Restored this way, a fanPressure write from before the change restarts in brae exactly as the uninterrupted run, and
OpenFOAM starts from it.

## totalPressure

brae needs no edit. The `inletValue` an older brae wrote is p0's value, so brae reads it as p0 and says so in a notice.
The restart continues exactly as it would from the same file with the entry spelled `p0`.

Older brae refused a named `psi` on these patches, so the `psi` and `gamma` it dropped were the defaults. One exception:
if the patch in `0/p` names a `rho` field other than `rho`, copy that entry into the restart time's `p` too.

For OpenFOAM, rename `inletValue` to `p0` on that patch.

## phi on a constraint patch

brae reads these files as they are. For OpenFOAM, open the restart time's `phi`. For each patch whose type in
`constant/polyMesh/boundary` is `symmetry`, `symmetryPlane` or `wedge`, change `type calculated;` to that type, and keep
`value`. For example, windAroundBuildingsBox's `frontAndBack` becomes `type symmetry;`.
