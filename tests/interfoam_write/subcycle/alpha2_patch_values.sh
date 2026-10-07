#!/usr/bin/env bash
# The gate: alpha2's PATCH VALUES in the compressive flux. alpha2 is a stored field (twoPhaseMixture.C:55-64),
# assigned `1.0 - alpha1` at alphaEqn.H:152 and :223, so its patch values are alpha1's as they stood at the last
# assignment -- one curvature pass older than alpha1's own on a contact angle. Under a limited alpharScheme,
# fvc::flux(-phir, alpha2, alpharScheme) takes its limiter's gradient on them (alphaEqn.H:164-176,
# LimitedScheme.C:51-80). Both loops built alpha2's patch values from the cells: the same number wherever
# alpha1's patch value is its cell's, and not on a contact angle, an inlet or an inletOutlet in inflow.
# FOUND 2026-10-07 by a reviewer reading the alpha step; no shipped tutorial reaches it (the seven that name
# `div(phirb,alpha) Gauss vanLeer` have zeroGradient and empty alpha patches alone), so the rows are laminar/
# capillaryRise -- the contact angle, with its inlet and atmosphere -- restaged with that scheme, STEPS pinned
# steps against real interFoam:
#   explicit      the tutorial's alpha step: the assignment at :223
#   MULESCorr     ...with MULESCorr and two correctors: the assignment at :152 too, read by the first corrector
# Three checks: each row's two loops within the bound of OpenFOAM, and the CONTROL on both rows,
# BRAE_CONTROL_ALPHA2_PATCH_ZERO_GRADIENT=1 -- the cells' value on every face again -- over the bound.
# MEASURED 2026-10-07, worst file but the continuity sum (the bounds are a decade above each row's worst):
#   explicit      host 1.5e-14   device 8.7e-15   before the fix U 3.0e-03 on both loops
#   MULESCorr     host 3.3e-14   device 2.2e-14   before the fix U 8.0e-03
# and the device loop says it takes the stored value on 820 boundary faces (the walls, the inlet, the atmosphere).
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
STEPS=6
wcase capillaryRise of > "$W/a2_stage.txt" 2>&1
[ -d "$W/w_of_capillaryRise" ] || { say "capillaryRise did not stage" FAIL; finish "alpha2's patch values"; }
# the MULESCorr row's entries: the solver the pre-solve needs, under a key that matches alpha.waterFinal too
MC='\n\1MULESCorr       yes;\n\1nLimiterIter    5;\n\1solver          smoothSolver;'
MC="$MC"'\n\1smoother        symGaussSeidel;\n\1tolerance       1e-13;\n\1relTol          0;/'
MCSED='s/^( *)alpha\.water$/\1"alpha.water.*"/;s/^( *)nAlphaCorr( +)1;/\1nAlphaCorr\22;'"$MC"
# row <tag> <sed expression for system/fvSolution, or nothing>: the restaged case run by OpenFOAM, then the
# host arm, the device arm and the device arm with the control, each compared with OpenFOAM's last step
row()
{
    local tag="$1" o="$W/a2_of_$1" arm e ot dt
    rm -rf "${o:?}"
    mkdir -p "$o"
    cp -r "$W/w_of_capillaryRise/0" "$W/w_of_capillaryRise/constant" "$W/w_of_capillaryRise/system" "$o/"
    sed -i -E 's/^( *div\(phirb,alpha\) +)Gauss linear;/\1Gauss vanLeer;/' "$o/system/fvSchemes"
    grep -qE "div\(phirb,alpha\) +Gauss vanLeer;" "$o/system/fvSchemes" \
        || { say "PREMISE  div(phirb,alpha) was restaged to vanLeer [$tag]" FAIL; return 1; }
    [ -z "${2:-}" ] || sed -i -E "$2" "$o/system/fvSolution"
    dt=$(sed -n -E 's/^deltaT +([^;]+);/\1/p' "$o/system/controlDict")
    sed -i -E "s/^endTime .*/endTime         $(python3 -c "print('%.12g' % ($STEPS*$dt))");/" \
        "$o/system/controlDict"
    sed -i -E "s/^writeInterval .*/writeInterval   $STEPS;/" "$o/system/controlDict"
    runof "$o"
    ot=$(timedirs "$o")
    [ "$(echo $ot | wc -w)" = 1 ] || { say "PREMISE  OpenFOAM writes the last step [$tag: $ot]" FAIL; return 1; }
    for arm in host device control; do
        e="$W/a2_${tag}_$arm"
        rm -rf "${e:?}"
        mkdir -p "$e"
        cp -r "$o/0" "$o/constant" "$o/system" "$e/"
        case $arm in
            host) runbrae "$e" host BRAE_X=1 ;;
            device) runbrae "$e" device BRAE_X=1 ;;
            control) runbrae "$e" device BRAE_CONTROL_ALPHA2_PATCH_ZERO_GRADIENT=1 ;;
        esac
        python3 "$CMP" "$o" "$e" $ot > "$e/cmp.txt" 2>&1
    done
}
# gap <tag> <arm>: the worst file's relative gap to OpenFOAM, the continuity sum apart; "-" on a structure failure
gap()
{
    python3 - "$W/a2_$1_$2/cmp.txt" <<'PY'
import json, sys
try:
    r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
    print('-' if r['structure'] else '%.1e' % max(v['rel'] for k, v in r['files'].items()
                                                  if 'cumulativeContErr' not in k))
except Exception:
    print('-')
PY
}
# faces <tag>: the boundary faces the device loop says take alpha2's stored value
faces()
{
    sed -n -E "s/.*alpha2's stored patch values on ([0-9]+) boundary face.*/\1/p" "$W/a2_$1_device/log.brae" | head -1
}
BOUND_EXPLICIT=2e-13
BOUND_MULESCORR=4e-13
row explicit ""
row mulescorr "$MCSED"
what="explicit: host $(gap explicit host), device $(gap explicit device) from OpenFOAM (bound $BOUND_EXPLICIT); the"
what="$what device loop takes the stored value on $(faces explicit) boundary faces"
within "$(gap explicit host)" $BOUND_EXPLICIT && within "$(gap explicit device)" $BOUND_EXPLICIT \
    && [ "$(faces explicit)" -gt 100 ] && say "$what" ok || say "$what" FAIL
what="MULESCorr, two correctors: host $(gap mulescorr host), device $(gap mulescorr device) from OpenFOAM"
what="$what (bound $BOUND_MULESCORR)"
within "$(gap mulescorr host)" $BOUND_MULESCORR && within "$(gap mulescorr device)" $BOUND_MULESCORR \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  the cells' value on every face: explicit $(gap explicit control), MULESCorr $(gap mulescorr control)"
above "$(gap explicit control)" 1e-6 && above "$(gap mulescorr control)" 1e-6 \
    && grep -aq "boundary face" "$W/a2_explicit_device/log.brae" \
    && ! grep -aq "alpha2's stored patch values" "$W/a2_explicit_control/log.brae" && say "$what" ok || say "$what" FAIL
finish "the compressive flux's limiter reads alpha2's stored patch values, as OpenFOAM's does"
