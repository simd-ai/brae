#!/usr/bin/env bash
# The gate: nAlphaSmoothCurvature on the HOST loop. interfaceProperties::calculateK smooths a COPY of alpha1 --
# alpha1L = fvc::average(fvc::interpolate(alpha1L)), n times (interfaceProperties.C:117-127) -- and takes the
# interface normal from the copy's gradient. What the copy's PATCHES do, read from fvcAverage.C:72-86 and the
# patch fields' operator=:
#   empty     an empty patch has no faces, so the average never sees it; brae summed its two faces per cell
#   patches   a patch's value stays what alpha1 had there through every pass -- nothing evaluates the copy --
#             where brae let every patch follow the smoothed cell
#   snGrad    gaussGrad's correction takes the copy's snGrad, the patch against the SMOOTHED cell; brae took
#             alpha1's
# FOUND 2026-10-07 by a reviewer. No shipped tutorial sets the entry, so the rows are two tutorials restaged
# with `nAlphaSmoothCurvature 2`, STEPS pinned steps against real interFoam (the GPU loop refuses the entry by
# name: tests/interfoam_refusals.sh `device_smoothCurvature`):
#   capillaryRise   2-D, a contact angle on the walls, an inletOutlet inlet, a zeroGradient atmosphere
#   weirOverflow    2-D, kEpsilon, a variableHeightFlowRate inlet (mixed) the interface crosses
# Three checks: each row within its bound of OpenFOAM, and the CONTROLS, BRAE_CONTROL_SMOOTH_CURVATURE naming
# one part to run as it ran -- `empty` and `patches` on capillaryRise, `snGrad` on weirOverflow -- beside the
# PREMISE that the entry is live in the oracle: brae WITHOUT the entry against OpenFOAM with it.
# core/smooth_curvature_wedge.sh has the wedge and the fixedValue inlet.
# MEASURED 2026-10-07, worst file but the continuity sum (the bounds are a decade above each row's worst):
#   capillaryRise   host 7.8e-15   before the fix 8.4e-01; without the entry 1.1e+00, `empty` 2.7e-02,
#                                  `patches` 5.9e-01 (`snGrad` moves nothing here: no patch's gradient reads a cell)
#   weirOverflow    host 1.2e-13   `snGrad` 5.4e-09, `empty` 5.1e-04, `patches` 1.4e-03
# 5 s.
. "$(dirname "$0")/../lib.sh"
STEPS=6
PASSES=2
# row <key>: the tutorial's write-gate case restaged with the entry and run by OpenFOAM
row()
{
    local key="$1" o="$W/sc_of_$1" dt ot
    wcase "$key" of > "$W/sc_stage_$key.txt" 2>&1
    [ -d "$W/w_of_$key" ] || { say "PREMISE  $key staged" FAIL; return 1; }
    rm -rf "${o:?}"
    mkdir -p "$o"
    cp -r "$W/w_of_$key/0" "$W/w_of_$key/constant" "$W/w_of_$key/system" "$o/"
    sed -i -E "s/^( *)nAlphaCorr( +)/\1nAlphaSmoothCurvature $PASSES;\n\1nAlphaCorr\2/" "$o/system/fvSolution"
    [ "$(grep -c "nAlphaSmoothCurvature $PASSES;" "$o/system/fvSolution")" = 1 ] \
        || { say "PREMISE  $key carries nAlphaSmoothCurvature $PASSES in alpha's solver entry" FAIL; return 1; }
    dt=$(sed -n -E 's/^deltaT +([^;]+);/\1/p' "$o/system/controlDict")
    sed -i -E "s/^endTime .*/endTime         $(python3 -c "print('%.12g' % ($STEPS*$dt))");/" \
        "$o/system/controlDict"
    sed -i -E "s/^writeInterval .*/writeInterval   $STEPS;/" "$o/system/controlDict"
    runof "$o"
    ot=$(timedirs "$o")
    [ "$(echo $ot | wc -w)" = 1 ] || { say "PREMISE  OpenFOAM writes the last step [$key: $ot]" FAIL; return 1; }
}
# arm <key> <name> [env...]: brae's host loop on a copy of the row's case, compared with OpenFOAM's last step;
# the arm named `none` runs WITHOUT the entry
arm()
{
    local key="$1" name="$2" o="$W/sc_of_$1" e="$W/sc_$1_$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    [ "$name" != none ] || sed -i '/nAlphaSmoothCurvature/d' "$e/system/fvSolution"
    runbrae "$e" host "$@"
    python3 "$CMP" "$o" "$e" $(timedirs "$o") > "$e/cmp.txt" 2>&1
}
# gap <key> <name>: the worst file's relative gap to OpenFOAM, the continuity sum apart; "-" on a structure failure
gap()
{
    python3 - "$W/sc_$1_$2/cmp.txt" <<'PY'
import json, sys
try:
    r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
    print('-' if r['structure'] else '%.1e' % max(v['rel'] for k, v in r['files'].items()
                                                  if 'cumulativeContErr' not in k))
except Exception:
    print('-')
PY
}
BOUND_CAPILLARY=2e-13
BOUND_WEIR=2e-12
row capillaryRise || finish "nAlphaSmoothCurvature on the host loop"
row weirOverflow || finish "nAlphaSmoothCurvature on the host loop"
arm capillaryRise host BRAE_X=1
arm weirOverflow host BRAE_X=1
arm capillaryRise none BRAE_X=1
arm capillaryRise empty BRAE_CONTROL_SMOOTH_CURVATURE=empty
arm capillaryRise patches BRAE_CONTROL_SMOOTH_CURVATURE=patches
arm weirOverflow snGrad BRAE_CONTROL_SMOOTH_CURVATURE=snGrad
what="capillaryRise, $PASSES passes: host $(gap capillaryRise host) from OpenFOAM (bound $BOUND_CAPILLARY)"
within "$(gap capillaryRise host)" $BOUND_CAPILLARY && say "$what" ok || say "$what" FAIL
what="weirOverflow, $PASSES passes: host $(gap weirOverflow host) from OpenFOAM (bound $BOUND_WEIR)"
within "$(gap weirOverflow host)" $BOUND_WEIR && say "$what" ok || say "$what" FAIL
what="CONTROL  capillaryRise without the entry $(gap capillaryRise none), the empty faces summed"
what="$what $(gap capillaryRise empty), every patch following its cell $(gap capillaryRise patches);"
what="$what weirOverflow with alpha's own snGrad $(gap weirOverflow snGrad)"
above "$(gap capillaryRise none)" 1e-4 && above "$(gap capillaryRise empty)" 1e-4 \
    && above "$(gap capillaryRise patches)" 1e-4 && above "$(gap weirOverflow snGrad)" 1e-10 \
    && say "$what" ok || say "$what" FAIL
finish "nAlphaSmoothCurvature smooths the copy of alpha as OpenFOAM's does, on the host loop"
