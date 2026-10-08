#!/usr/bin/env bash
# The gate: the function objects' STATE DICTIONARY, <time>/uniform/functionObjects/functionObjectProperties.
# functionObjectList::execute writes it after every object has run, at a write time, at 16 digits
# (functionObjectList.C:776-789); a probes object stores average, min, max and size of each field's row in it
# at every sample (probesTemplates.C:100-121), under a section named by the SHA-1 of "results", then the
# object, then the value's type. An entry is added where it is first set, so the ORDER is the run's history.
# brae wrote the file empty before 2026-10-08.
# The row is laminar/damBreak as shipped to t = STOP -- three write times -- with three probes objects:
#   sampled  `sampleOnExecute true; writeControl writeTime; writeInterval 2`: sampled at EVERY step, a row at
#            every second write. Its results are in the first write's dictionary, from that step's execute,
#            ahead of the object that wrote a row there. Scalar, vector and label results.
#   first    `writeControl writeTime`: a row and its results at every write.
#   second   `writeControl writeTime; writeInterval 2`, no sampleOnExecute: nothing until the second write.
# Three checks. (1) THE HOST LOOP: every write time's dictionary is OpenFOAM's text with the numbers masked,
# and every number within HOST_BOUND of its own size. (2) THE GPU LOOP, the same. (3) CONTROL,
# BRAE_CONTROL_FUNCTION_OBJECT_STATE=stale: the dictionary as the step BEFORE left it (written ahead of the
# execute) is other text at the first write and other numbers at the next.
# MEASURED 2026-10-08: three writes (0.05, 0.1, 0.15), 74 numbers; brae's dictionaries OpenFOAM's text, the
# numbers within 4.3e-13 on the host and 3.3e-13 on the GPU. FOUND by this gate: a vector entry is written
# `( x y z )`, its tokens a space apart, where brae first wrote `(x y z)`. Control: 18 lines of other text at
# 0.05, a number 5.6e-01 off at 0.1.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
. "$(dirname "$0")/_common.sh"
STOP=0.16
HOST_BOUND=4e-12
DEVICE_BOUND=4e-12
name="the function objects' state dictionary"
o="$W/st_of"
cat > "$W/st_functions.txt" <<'FO'
functions
{
    sampled
    {
        type            probes;
        libs            (sampling);
        sampleOnExecute true;
        writeControl    writeTime;
        writeInterval   2;
        fields          (p alpha.water U);
        probeLocations  ((0.1 0.1 0.0073) (0.5 0.02 0.0073));
    }
    first
    {
        type            probes;
        libs            (sampling);
        writeControl    writeTime;
        fields          (p_rgh);
        probeLocations  ((0.292 0.05 0.0073) (5 5 5));
    }
    second
    {
        type            probes;
        libs            (sampling);
        writeControl    writeTime;
        writeInterval   2;
        fields          (p);
        probeLocations  ((0.1 0.1 0.0073));
    }
}
FO
fo_stage "$o" "$LAM" "$STOP" adjustable 0.05 "$W/st_functions.txt" || finish "$name"
ot=$(timedirs "$o")
t1=$(echo $ot | awk '{print $1}')
t2=$(echo $ot | awk '{print $2}')
p1="$o/$t1/uniform/functionObjects/functionObjectProperties"
p2="$o/$t2/uniform/functionObjects/functionObjectProperties"
[ "$(echo $ot | wc -w)" = 3 ] && grep -q "^    sampled$" "$p1" && grep -q "^    first$" "$p1" \
    && ! grep -q "^    second$" "$p1" && grep -q "^    second$" "$p2" && grep -q "^        vector$" "$p1" \
    && [ "$(grep -n "^    sampled$" "$p1" | cut -d: -f1)" -lt "$(grep -n "^    first$" "$p1" | cut -d: -f1)" ] \
    || { say "PREMISE  OpenFOAM: three writes [$ot], sampled ahead of first at $t1, second only from $t2" FAIL
         finish "$name"; }
arm()   # arm <run dir> <label> <bound>
{
    local e="$1" ok=1 worst=0 numbers=0 t
    for t in $ot; do
        read -r text count gap <<< "$(fo_state "$o" "$e" "$t")"
        [ "$text" = - ] && { text=1; count=0; gap=1; }
        [ "$text" = 0 ] && within "$gap" "$3" || ok=0
        numbers=$((numbers + count))
        worst=$(python3 -c "print('%.1e' % max(float('$worst'), float('$gap')))")
    done
    what="[$2] the dictionary at $(echo $ot | tr ' ' ',') is OpenFOAM's text with the numbers masked, its"
    what="$what $numbers numbers within $worst (bound $3)"
    [ $ok = 1 ] && say "$what" ok || say "$what" FAIL
}
fo_arm "$o" "$W/st_host" host BRAE_X=1
arm "$W/st_host" host $HOST_BOUND
fo_arm "$o" "$W/st_device" device BRAE_X=1
arm "$W/st_device" device $DEVICE_BOUND
fo_arm "$o" "$W/st_stale" host BRAE_CONTROL_FUNCTION_OBJECT_STATE=stale
read -r text1 count1 gap1 <<< "$(fo_state "$o" "$W/st_stale" "$t1")"
read -r text2 count2 gap2 <<< "$(fo_state "$o" "$W/st_stale" "$t2")"
what="CONTROL  the dictionary the step before left: $text1 lines of other text at $t1, a number $gap2 off at $t2"
[ "${text1/-/0}" -ge 1 ] && above "$gap2" 1e-6 && say "$what" ok || say "$what" FAIL
finish "the state dictionary holds every object's results as OpenFOAM's does at each write, on both loops"
