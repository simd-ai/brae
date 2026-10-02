#!/usr/bin/env bash
# The write gate: three tutorials whose fluid is nearly at rest, against OpenFOAM's own sensitivity.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
# stokesI, solitaryGrimshaw and damBreakLeakage carry tutorial bounds of 5e-07, 2e-07 and 3e-06. After their two
# steps the velocity is a small residual of large cancelling pressure and gravity terms, so where a converged
# pressure solve STOPS moves it. THE TWIN: OpenFOAM itself with every tolerance one decade tighter (1e-14 for
# 1e-13). MEASURED (2026-10-02), worst field file:
#                      brae host   brae device   OpenFOAM against its twin
#   stokesI            4.8e-08     9.2e-08       6.5e-04
#   solitaryGrimshaw   1.1e-08     1.7e-08       4.1e-07
#   damBreakLeakage    2.4e-07     3.6e-07       7.0e-03
# The check is that brae is CLOSER to OpenFOAM than OpenFOAM is to its own twin, on both arms -- a statement
# about the port that the bound alone does not make.
for key in stokesI solitaryGrimshaw damBreakLeakage; do
    wcase $key host device > "$W/rest_$key.txt" 2>&1
    o="$W/w_of_$key"
    [ -d "$o" ] || { say "$key did not stage" FAIL; continue; }
    t="$W/twin_$key"
    mkdir -p "$t"
    cp -r "$o/0" "$o/constant" "$o/system" "$t/"
    sed -i -E 's/(tolerance\s+)1e-13;/\11e-14;/' "$t/system/fvSolution"
    grep -q "1e-14;" "$t/system/fvSolution" || { say "$key: the twin's tolerances did not tighten" FAIL; continue; }
    runof "$t"
    python3 "$CMP" "$o" "$t" $(timedirs "$o") > "$W/cmp_twin_$key.txt" 2>&1
    tw=$(worstof "$W/cmp_twin_$key.txt")
    bh=$(worstof "$W/cmp_w_${key}_host.txt")
    bd=$(worstof "$W/cmp_w_${key}_device.txt")
    closer "$bh" "$tw" && closer "$bd" "$tw" \
        && say "$key: brae (host $bh, device $bd) is closer to OpenFOAM than OpenFOAM's own twin ($tw)" ok \
        || say "$key: brae (host $bh, device $bd) is closer to OpenFOAM than OpenFOAM's own twin ($tw)" FAIL
done
finish "the at-rest tutorials: brae inside OpenFOAM's own sensitivity"
