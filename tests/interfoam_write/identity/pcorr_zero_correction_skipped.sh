#!/usr/bin/env bash
# The write gate: CorrectPhi's first pass with the non-orthogonal correction of pcorr NOT computed, against the
# pass that computes it. pcorr starts every CorrectPhi at zero, cells and patches, so at the first pass its
# gradient, the correction's source and the face flux correction are zeros and change no value; they were
# computed all the same. MEASURED 2026-10-04, ms a step with nNonOrthogonalCorrectors 0, where the first pass is
# the only one: 15.0 -> 0.6 on waveMakerPiston at 896,000 cells, 21.3 -> 0.7 on the 845,536-cell moving hull.
# STAGING, stated: oscillatingBox with nNonOrthogonalCorrectors raised 0 -> 1 (brae's copies only), so that there
# IS a pass where pcorr is no longer zero and its correction must be computed -- the pass the CONTROL breaks
# (BRAE_CONTROL_PCORR_CORRECTION_NEVER=1 skips the correction at every pass).
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase oscillatingBox of > "$W/pz_stage.txt" 2>&1
o="$W/w_of_oscillatingBox"
[ -d "$o" ] || { say "oscillatingBox did not stage" FAIL; finish "pcorr zero correction identity"; }
for v in computed skipped never; do
    e="$W/pz_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i -E 's/(nNonOrthogonalCorrectors\s+)0;/\11;/' "$e/system/fvSolution"
    case $v in
        computed) runbrae "$e" device BRAE_CONTROL_PCORR_ZERO_CORRECTION=1 ;;
        skipped)  runbrae "$e" device BRAE_X=1 ;;
        never)    runbrae "$e" device BRAE_CONTROL_PCORR_CORRECTION_NEVER=1 ;;
    esac
done
# differs <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$1"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
n=$(differs "$W/pz_computed" "$W/pz_skipped")
what="[device] the first pass without the correction of a zero pcorr, two passes a CorrectPhi: $n written files differ"
grep -q "nNonOrthogonalCorrectors 1;" "$W/pz_skipped/system/fvSolution" \
    && [ -n "$(timedirs "$W/pz_skipped")" ] && [ "$(timedirs "$W/pz_skipped")" = "$(timedirs "$W/pz_computed")" ] \
    && [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
n=$(differs "$W/pz_computed" "$W/pz_never")
what="CONTROL  with the correction skipped at the second pass too, the written files change ($n of them)"
[ "$n" != 0 ] && say "$what" ok || say "$what" FAIL
finish "CorrectPhi's first pass skips the correction of a zero pcorr and changes nothing"
