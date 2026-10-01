# Shared by the files of this folder: laminar damBreak, `adjustable 0.05` to endTime 0.12.
# core_base: OpenFOAM's run (cached) as $W/of with its premise, brae on each arm as $W/br_<arm>, and the
# comparison of the two write times as $W/cmp_<arm>.txt. Seconds on this 2268-cell case, so every file of
# the folder builds it for itself.
core_base()
{
    stage "$LAM" "$W/of"  0.12 adjustable 0.05 0 || { echo "SKIP: staging failed"; exit 77; }
    runof "$W/of"
    o=$(timedirs "$W/of")
    [ "$o" = "0.05 0.1 " ] && say "premise: OpenFOAM writes {0.05, 0.1} and nothing at endTime 0.12" ok \
                           || { say "premise: OpenFOAM writes {0.05, 0.1} and nothing at endTime 0.12 [$o]" FAIL; exit 1; }
    for arm in $ARMS; do
        stage "$LAM" "$W/br_$arm" 0.12 adjustable 0.05 0 || exit 1
        runbrae "$W/br_$arm" "$arm"
        python3 "$CMP" "$W/of" "$W/br_$arm" 0.05 0.1 > "$W/cmp_$arm.txt" 2>&1
    done
}
