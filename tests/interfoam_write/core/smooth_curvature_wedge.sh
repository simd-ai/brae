#!/usr/bin/env bash
# The gate: nAlphaSmoothCurvature on a WEDGE, host loop. core/smooth_curvature.sh has the entry and what the
# smoothed copy's patches do; this row is the patch kind that DOES follow its smoothed cell. Assigning the
# average to a transform patch is an evaluate (transformFvPatchField.C:142-148), and a scalar's evaluate on a
# wedge is its cell's value -- so the two wedge faces of every cell re-enter each pass with the cell's value
# after the pass before, where a wall keeps alpha1's.
# The row is LES/nozzleFlow2D -- axisymmetric, a fixedValue inlet of fuel -- restaged with
# `nAlphaSmoothCurvature 2`, STEPS pinned steps against real interFoam.
# Three checks: the host within its bound of OpenFOAM; the CONTROL `transform`
# (BRAE_CONTROL_SMOOTH_CURVATURE=transform keeps the wedge's value through the passes as a wall's is kept)
# beside the PREMISE that the entry is live in the oracle, brae WITHOUT it against OpenFOAM with it; and the
# CONTROL `snGrad`, the fixedValue inlet's normal gradient taken against alpha's own cell.
# MEASURED 2026-10-07, worst file but the continuity sum (the bound is a decade above): host 4.4e-13; the
# wedge's value kept 3.7e-03; without the entry 4.2e-02; alpha's own snGrad 1.2e-07. 11 s.
. "$(dirname "$0")/../lib.sh"
STEPS=6
PASSES=2
KEY=nozzleFlow2D
o="$W/scw_of"
wcase $KEY of > "$W/scw_stage.txt" 2>&1
[ -d "$W/w_of_$KEY" ] || { say "PREMISE  $KEY staged" FAIL; finish "nAlphaSmoothCurvature on a wedge"; }
mkdir -p "$o"
cp -r "$W/w_of_$KEY/0" "$W/w_of_$KEY/constant" "$W/w_of_$KEY/system" "$o/"
grep -q "wedge" "$o/constant/polyMesh/boundary" \
    || { say "PREMISE  $KEY has a wedge patch" FAIL; finish "nAlphaSmoothCurvature on a wedge"; }
sed -i -E "s/^( *)nAlphaCorr( +)/\1nAlphaSmoothCurvature $PASSES;\n\1nAlphaCorr\2/" "$o/system/fvSolution"
[ "$(grep -c "nAlphaSmoothCurvature $PASSES;" "$o/system/fvSolution")" = 1 ] \
    || { say "PREMISE  $KEY carries the entry" FAIL; finish "nAlphaSmoothCurvature on a wedge"; }
dt=$(sed -n -E 's/^deltaT +([^;]+);/\1/p' "$o/system/controlDict")
sed -i -E "s/^endTime .*/endTime         $(python3 -c "print('%.12g' % ($STEPS*$dt))");/" "$o/system/controlDict"
sed -i -E "s/^writeInterval .*/writeInterval   $STEPS;/" "$o/system/controlDict"
runof "$o"
ot=$(timedirs "$o")
[ "$(echo $ot | wc -w)" = 1 ] \
    || { say "PREMISE  OpenFOAM writes the last step [$ot]" FAIL; finish "nAlphaSmoothCurvature on a wedge"; }
# arm <name> [env...]: brae's host loop on a copy of the case, compared with OpenFOAM's last step; the arm named
# `none` runs WITHOUT the entry
arm()
{
    local name="$1" e="$W/scw_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    [ "$name" != none ] || sed -i '/nAlphaSmoothCurvature/d' "$e/system/fvSolution"
    runbrae "$e" host "$@"
    python3 "$CMP" "$o" "$e" $ot > "$e/cmp.txt" 2>&1
}
# gap <name>: the worst file's relative gap to OpenFOAM, the continuity sum apart; "-" on a structure failure
gap()
{
    python3 - "$W/scw_$1/cmp.txt" <<'PY'
import json, sys
try:
    r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
    print('-' if r['structure'] else '%.1e' % max(v['rel'] for k, v in r['files'].items()
                                                  if 'cumulativeContErr' not in k))
except Exception:
    print('-')
PY
}
BOUND=5e-12
arm host BRAE_X=1
arm none BRAE_X=1
arm transform BRAE_CONTROL_SMOOTH_CURVATURE=transform
arm snGrad BRAE_CONTROL_SMOOTH_CURVATURE=snGrad
what="$KEY, $PASSES passes: host $(gap host) from OpenFOAM (bound $BOUND)"
within "$(gap host)" $BOUND && say "$what" ok || say "$what" FAIL
what="CONTROL  the wedge's value kept through the passes $(gap transform); without the entry $(gap none)"
above "$(gap transform)" 1e-5 && above "$(gap none)" 1e-5 && say "$what" ok || say "$what" FAIL
what="CONTROL  the inlet's snGrad against alpha's own cell $(gap snGrad)"
above "$(gap snGrad)" 1e-9 && say "$what" ok || say "$what" FAIL
finish "nAlphaSmoothCurvature lets a wedge follow its smoothed cell, as OpenFOAM's does"
