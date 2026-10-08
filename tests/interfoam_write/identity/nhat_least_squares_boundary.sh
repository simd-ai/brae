#!/usr/bin/env bash
# The write gate: the alpha hooks' boundary normal where the case's `nHat` gradient is NOT Gauss linear, taken
# from a least-squares gradient at the patches' face cells alone, against calculateK whole
# (BRAE_CONTROL_NHAT_FULL=1).
# The hook wants the normal on the patches' faces. Where the Gauss boundary-only form does not apply it took
# calculateK whole on the host, six times a step: the fit and its limiter over every cell, the gradient at
# every face, the internal faces' normal, the curvature. A least-squares fit and its cell limiter are local to
# a cell, so they are now taken at the patches' face cells alone, by the whole-mesh functions' own bodies
# handed a subset (fvc::GradSubset). MEASURED 2026-10-05 on RAS/electrostaticDeposition (54,390 cells, 6,370 of
# them on a patch), three rounds: 75.0-77.8 ms a step with calculateK whole, 71.3-73.4 with the whole mesh's
# gradient and the boundary half alone (BRAE_CONTROL_NHAT_WHOLE_GRADIENT=1, the form kept for a limited Gauss
# nHat), 58.2-59.6 at the boundary's cells; the hooks' boundary normal 13.8 -> 10.7 -> 2.0 ms a step.
# THE ROWS, and what each can see:
#   RAS/electrostaticDeposition's W row (`default cellLimited leastSquares 1`, 3-D, a mesh that moves -- the
#     one shipped tutorial with such an nHat): the forms' notices and the run on that mesh. Of the gradient's
#     value only what rounding happens to give, and nothing asserts it: on the first step every patch normal
#     is a zero; from the second a cell that sits above 1 by rounding at an inflow face (the patch takes 1)
#     has a normal of about 1e-9, and a face that is not axis-aligned keeps a residue of the cell gradient's
#     tangential part (the check counts the normals that are not zero: 2,110 of 6,642 at most).
#   laminar/waves/stokesI's restaged with `nHat cellLimited leastSquares 0.5;` (2-D; a wave inlet whose fixed
#     values the free surface crosses, so the normal is NOT zero there; the limiter's k < 1 branch).
#   laminar/capillaryRise's restaged with `nHat leastSquares;` (a contact angle, whose wall gradient the pass
#     writes back): files only -- the in-code check would make that pass twice and refuses.
# Nothing written may change. Held to the whole-curvature arm, file for file: the DEFAULT run, the
# whole-gradient run, and each with BRAE_CONTROL_NHAT_BOUNDARY_CHECK=1, which takes calculateK whole as well at
# every call and compares every patch face's normal, bitwise, saying how many calls and how many normals that
# were not zero.
# The CONTROLS, on the stokesI row: every other face of the subset left out
# (BRAE_CONTROL_NHAT_SUBSET_SHORT=1) -- under the check the run must stop at a patch face; without it the run
# must finish and write other files; and the whole-gradient form handing back its first call's normal
# (BRAE_CONTROL_NHAT_WHOLE_GRADIENT_STALE=1) must stop under the check.
# DOES NOT CLAIM a least-squares nHat on a mesh with a coupled pair: the solver refuses that combination by
# name (inter_case_cpp.cu; the arm baffle_leastSquaresNHat of tests/interfoam_refusals.sh), and the subset
# form's coupled branches are held to the whole-mesh ones by tests/test_grad_subset_coupled.cu. The restaged
# rows are brae against brae: their oracle is the whole-curvature arm, not an OpenFOAM run (the W rows' OpenFOAM
# runs are staged and not read). That the whole-mesh gradient functions still return what they did before
# they were made to take
# a subset: the host arm of electrostaticDeposition's row wrote byte-identical files before and after (32
# files, recorded in PORT.md, made once by hand), and tests/test_grad_subset.cu holds subset = whole; their
# gates against OpenFOAM (leastsquares_grad_vs_openfoam, celllimited_vs_openfoam) have bounds and would not
# see one ulp.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
subset="nHat: the alpha hooks take the boundary normal from a least-squares gradient at the boundary's own cells"
wholeGrad="nHat: the alpha hooks take the boundary normal from the whole mesh's gradient"
wholeK="nHat: the alpha hooks take calculateK whole"
counted="nHat boundary-cells gradient check: [1-9][0-9]* calls, each [1-9][0-9]* patch faces' normal"
# same <dirA> <dirB>: how many files of A's written times are not B's, byte for byte; "-" unless both hold the
# same times and the same files at each
same()
{
    local a="$1" b="$2" n=0 d f
    [ -n "$(timedirs "$a")" ] && [ "$(timedirs "$a")" = "$(timedirs "$b")" ] || { echo "-"; return; }
    for d in $(timedirs "$a"); do
        [ "$(filesets "$a" "$d")" = "$(filesets "$b" "$d")" ] || { echo "-"; return; }
        for f in $(cd "$a/$d" && find . -type f | sort); do
            cmp -s "$a/$d/$f" "$b/$d/$f" || n=$((n + 1))
        done
    done
    echo $n
}
# staged <dir> <source dir>: a fresh copy of a staged case
staged()
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$2/0" "$2/constant" "$2/system" "$1/"
}
# restaged <dir> <W row> <nHat entry>: a copy of a W row with `nHat <entry>;` added to its gradSchemes
restaged()
{
    staged "$1" "$W/w_of_$2"
    sed -i -E "s/^( *)default( +)Gauss linear;/\1default\2Gauss linear;\n\1nHat            $3;/" \
        "$1/system/fvSchemes"
    grep -q "nHat  *$3;" "$1/system/fvSchemes"
}
# arms <source dir> <tag> [nocheck]: the whole-curvature arm, the whole-gradient arm and the default arm, and
# the last two again under the check. NOT inside a substitution: an arm that stops ends the gate with runbrae's
# own message and the log's tail
arms()
{
    local v
    for v in full whole default checked wholeChecked; do
        [ "${3:-}" = nocheck ] && { [ $v = checked ] || [ $v = wholeChecked ]; } && continue
        staged "$W/${2}_$v" "$1"
        case $v in
            full) runbrae "$W/${2}_$v" device BRAE_CONTROL_NHAT_FULL=1 ;;
            whole) runbrae "$W/${2}_$v" device BRAE_CONTROL_NHAT_WHOLE_GRADIENT=1 ;;
            default) runbrae "$W/${2}_$v" device ;;
            checked) runbrae "$W/${2}_$v" device BRAE_CONTROL_NHAT_BOUNDARY_CHECK=1 ;;
            wholeChecked) runbrae "$W/${2}_$v" device BRAE_CONTROL_NHAT_BOUNDARY_CHECK=1 \
                              BRAE_CONTROL_NHAT_WHOLE_GRADIENT=1 ;;
        esac
    done
}
# only <log> <notice>: that notice in the log and neither of the other two forms'
only()
{
    local n
    grep -q "$2" "$1" || return 1
    for n in "$subset" "$wholeGrad" "$wholeK"; do
        [ "$n" = "$2" ] || ! grep -q "$n" "$1" || return 1
    done
}
# verdict <tag> [nocheck]: "<ok|no> <files an arm> <whole differs> <default differs> <checked differs>
# <whole-checked differs> <calls checked> <most normals not zero>". ok: each arm says the form it took and no
# other; each check's count is in its own arm, with at least six calls compared, and in no unchecked arm; no
# written file of any arm differs from the whole-curvature arm's
verdict()
{
    local tag="$1" t n1 n2 n3=0 n4=0 calls=0 callsW=0 nz=0 said=ok
    t=$(for d in $(timedirs "$W/${tag}_full"); do find "$W/${tag}_full/$d" -type f; done | wc -l)
    n1=$(same "$W/${tag}_full" "$W/${tag}_whole")
    n2=$(same "$W/${tag}_full" "$W/${tag}_default")
    only "$W/${tag}_full/log.brae" "$wholeK" && only "$W/${tag}_whole/log.brae" "$wholeGrad" \
        && only "$W/${tag}_default/log.brae" "$subset" || said=no
    grep -q "nHat .* check:" "$W/${tag}_default/log.brae" "$W/${tag}_whole/log.brae" && said=no
    if [ "${2:-}" != nocheck ]; then
        n3=$(same "$W/${tag}_full" "$W/${tag}_checked")
        n4=$(same "$W/${tag}_full" "$W/${tag}_wholeChecked")
        calls=$(grep -a "$counted" "$W/${tag}_checked/log.brae" | tail -1 \
                | sed -n -E 's/.*check: ([0-9]+) calls.*/\1/p')
        callsW=$(grep -a "nHat whole-gradient check: [1-9]" "$W/${tag}_wholeChecked/log.brae" | tail -1 \
                 | sed -n -E 's/.*check: ([0-9]+) calls.*/\1/p')
        nz=$(grep -a "$counted" "$W/${tag}_checked/log.brae" | tail -1 \
             | sed -n -E 's/.*at most ([0-9]+) of them not zero.*/\1/p')
        only "$W/${tag}_checked/log.brae" "$subset" && only "$W/${tag}_wholeChecked/log.brae" "$wholeGrad" \
            || said=no
        [ "${calls:-0}" -ge 6 ] && [ "${callsW:-0}" -ge 6 ] || said=no
    fi
    [ "$t" -gt 0 ] && [ "$n1" = 0 ] && [ "$n2" = 0 ] && [ "$n3" = 0 ] && [ "$n4" = 0 ] || said=no
    echo "$said $t $n1 $n2 $n3 $n4 ${calls:-0} ${nz:-0}"
}
for k in electrostaticDeposition stokesI capillaryRise; do
    wcase $k of >> "$W/nl_stage.txt" 2>&1
    [ -d "$W/w_of_$k" ] || { say "$k did not stage" FAIL; finish "nHat least squares"; }
done
grep -q "cellLimited leastSquares" "$W/w_of_electrostaticDeposition/system/fvSchemes" \
    || { say "PREMISE  electrostaticDeposition's gradient is a limited least-squares one" FAIL; finish "nHat"; }
restaged "$W/nl_srcWave" stokesI "cellLimited leastSquares 0.5" \
    && restaged "$W/nl_srcAngle" capillaryRise "leastSquares" \
    || { say "stokesI's or capillaryRise's nHat was not restaged" FAIL; finish "nHat least squares"; }
arms "$W/w_of_electrostaticDeposition" nle
read -r v t n1 n2 n3 n4 calls nz <<< "$(verdict nle)"
what="[device] electrostaticDeposition: $t files an arm; whole gradient, default and each checked: $n1, $n2,"
what="$what $n3, $n4 differ; $calls+ calls, $nz normals not zero"
[ "$v" = ok ] && say "$what" ok || say "$what" FAIL
arms "$W/nl_srcWave" nlw
arms "$W/nl_srcAngle" nla nocheck
read -r v t n1 n2 n3 n4 calls nz <<< "$(verdict nlw)"
read -r va ta a1 a2 a3 a4 ac az <<< "$(verdict nla nocheck)"
what="[device] stokesI, nHat cellLimited leastSquares 0.5: $t files, $n1, $n2, $n3, $n4 differ, $nz normals not"
what="$what zero; capillaryRise, leastSquares (a contact angle): $ta files, $a1, $a2"
[ "$v" = ok ] && [ "${nz:-0}" -ge 1 ] && [ "$va" = ok ] && say "$what" ok || say "$what" FAIL
# the controls. Two are expected to stop and one to finish wrong: run directly, runbrae would end the gate
for v in short shortAlone stale; do
    staged "$W/nlc_$v" "$W/nl_srcWave"
    case $v in
        short) ( cd "$W/nlc_$v" && BRAE_CONTROL_NHAT_BOUNDARY_CHECK=1 BRAE_CONTROL_NHAT_SUBSET_SHORT=1 \
                     "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
        shortAlone) ( cd "$W/nlc_$v" && BRAE_CONTROL_NHAT_SUBSET_SHORT=1 \
                          "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
        stale) ( cd "$W/nlc_$v" && BRAE_CONTROL_NHAT_BOUNDARY_CHECK=1 BRAE_CONTROL_NHAT_WHOLE_GRADIENT=1 \
                     BRAE_CONTROL_NHAT_WHOLE_GRADIENT_STALE=1 "$BIN" -case . -device > log.brae 2>&1
                 echo $? > exit.txt ) ;;
    esac
done
# named <dir> <form>: the patch the check's message names for that form
named()
{
    grep -a "NHAT_BOUNDARY_CHECK: the boundary normal of patch .* from $2 and" "$1/log.brae" | head -1 \
        | sed -E 's/.*the boundary normal of patch ([^,]+), face.*/\1/'
}
p1=$(named "$W/nlc_short" "the gradient at the boundary's cells")
p2=$(named "$W/nlc_stale" "the whole gradient alone")
differs=$(same "$W/nlw_default" "$W/nlc_shortAlone")
what="CONTROL  every other face of the subset left out: the check stops at ${p1:-?}, and unchecked $differs files"
what="$what differ; the whole-gradient form's normal stale: stops at ${p2:-?}"
[ "$(cat "$W/nlc_short/exit.txt")" != 0 ] && [ -n "$p1" ] \
    && grep -q "CONTROL MODE: every other face of the boundary cells' subset" "$W/nlc_short/log.brae" \
    && [ "$(cat "$W/nlc_shortAlone/exit.txt")" = 0 ] && [ "$differs" != - ] && [ "$differs" -gt 0 ] \
    && grep -q "CONTROL MODE: every other face of the boundary cells' subset" "$W/nlc_shortAlone/log.brae" \
    && [ "$(cat "$W/nlc_stale/exit.txt")" != 0 ] && [ -n "$p2" ] \
    && grep -q "CONTROL MODE: the boundary normal handed back is the first call's" "$W/nlc_stale/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "a least-squares nHat's boundary normal, taken at the boundary's cells, is calculateK's"
