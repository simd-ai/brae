#!/usr/bin/env bash
# brae's dynamicRefineFvMeshCoeffs reader against the three shipped AMR tutorials' own dictionaries.
#
# THE ORACLE IS THE DICTIONARY ITSELF -- every asserted value is written in the file, so this cannot
# drift: if a tutorial changes, the assertion fails and names the entry that moved.
#
# WHY ALL THREE. They are written three different ways, and no one of them exercises OpenFOAM's lookup
# (`optionalSubDict(typeName + "Coeffs")`, dynamicRefineFvMesh.C:181 and :1292):
#   laminar/damBreakWithObstacle   the entries FLAT at the top level
#   laminar/oscillatingBox         FLAT, beside a `solvers { VF { ... } }` sub-dict for its motion
#                                  solver -- a reader that took "the sub-dictionary" would read that
#   RAS/motorBike                  WRAPPED in `dynamicRefineFvMeshCoeffs`, and with NO `unrefineLevel`
#
# THE ONE DEFAULT. `unrefineLevel` is the only entry OpenFOAM defaults (getOrDefault, :1350-1354), to
# GREAT = 1.0e+15 (doubleScalar.H:58). motorBike omits it, and reading 0 there would unrefine the whole
# mesh on the first step -- which is why the default is asserted and not assumed.
#
# WHY THIS UNIT EXISTS AT ALL. Until now `refineInterval` and `dumpLevel` appeared NOWHERE in src/, and
# every other entry only as a C++ parameter of the ported decision side -- so no case could drive it.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_refine_controls"
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
for t in laminar/damBreakWithObstacle laminar/oscillatingBox RAS/motorBike; do
    [ -f "$TUT/multiphase/interFoam/$t/constant/dynamicMeshDict" ] \
        || { echo "SKIP: $t/constant/dynamicMeshDict not found under $TUT"; exit 77; }
done

rc=0
"$BIN" "$TUT" || rc=1
echo "refine_controls_vs_openfoam: rc $rc"
exit $rc
