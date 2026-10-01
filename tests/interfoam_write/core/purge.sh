#!/usr/bin/env bash
# The write gate on damBreak: purgeWrite, and that writing changes nothing (arms F, H).
. "$(dirname "$0")/../lib.sh"
# F + H: `timeStep 1, purgeWrite 1` against `timeStep N`, to endTime 0.02 -- no trim under timeStep
stage "$LAM" "$W/of_p" 0.02 timeStep 1 1 || exit 1
runof "$W/of_p"
op=$(timedirs "$W/of_p")
# ...and MV, laminar/mixerVessel2D: a SUB-CYCLED alpha, so the old level's captures run -- alpha as the
# step starts and as its last sub-cycle begins, on the host and in a device buffer -- on every step in one
# run and on one step after N-1 without in the other. Both damBreaks have nAlphaSubCycles 1 and run none.
for case in LAM RAS MV; do
    src=$LAM; [ $case = RAS ] && src=$RAS; [ $case = MV ] && src=$MV
    for arm in $ARMS; do
        d1="$W/f1_${case}_$arm"; dn="$W/fn_${case}_$arm"
        stage "$src" "$d1" 0.02 timeStep 1 1 || exit 1
        runbrae "$d1" "$arm"
        last=$(timedirs "$d1" | tr -d ' ')
        n=$(sed -n 's/^index *\([0-9]*\);/\1/p' "$d1/$last/uniform/time")
        if [ $case = LAM ]; then
            [ "$(timedirs "$d1")" = "$op" ] && say "ARM H  [$arm] purgeWrite 1 keeps only OpenFOAM's last directory [$op]" ok \
                                            || say "ARM H  [$arm] purgeWrite 1 keeps only OpenFOAM's last directory [$(timedirs "$d1")] vs [$op]" FAIL
        fi
        stage "$src" "$dn" 0.02 timeStep "$n" 0 || exit 1
        runbrae "$dn" "$arm"
        [ "$(timedirs "$dn")" = "$last " ] && diff -r "$d1/$last" "$dn/$last" > /dev/null \
            && say "ARM F  [$case $arm] writing every step leaves $last/ byte-identical to writing once (index $n)" ok \
            || { say "ARM F  [$case $arm] writing every step leaves $last/ byte-identical to writing once (index $n)" FAIL; diff -rq "$d1/$last" "$dn/$last" | head -5; }
    done
done
finish "arms F, H: writing every step leaves the run byte-identical"
