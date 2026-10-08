#!/usr/bin/env bash
# The gate: a `probes` function object, run by brae as OpenFOAM runs it -- built at the first step, executed
# at every later one on the fields the step left (Time::run, Time.C:781-860) -- against the files OpenFOAM's
# own probes wrote. brae ran no function object before 2026-10-08.
# The row is laminar/damBreak as shipped (adjustTimeStep, `writeControl adjustable; writeInterval 0.05`) to
# t = STOP with three objects:
#   every    no timing entry, so no wrapper: a row a step. p, p_rgh and alpha.water at four locations:
#            (0.292 0.05 0.0073) ON THE FACE between two blocks, where OpenFOAM's octree takes the
#            lower-numbered of the two cells that hold it; one inside a cell; one in the water column; and
#            (5 5 5), outside the mesh: "# Not Found" in the head and -VGREAT in its column.
#   velocity U, a row a step, at the three locations a cell holds. (With one no cell holds OpenFOAM itself
#            stops at the first sample -- function_objects/refusals.sh.)
#   written  `writeControl writeTime`, a row a write time, `includeOutOfBounds false`: the outside location's
#            column is left out.
# Three checks. (1) THE HOST LOOP'S FILES: each of the five has OpenFOAM's head byte for byte, its rows, its
# time column as text, and every value within HOST_BOUND of its own size. (2) THE GPU LOOP'S, where the probes'
# cells are gathered on the GPU. (3) CONTROL, BRAE_CONTROL_PROBE_SEARCH=nearestCentre: the cell whose centre
# is nearest in place of the octree's order puts the face location's column off by more than CONTROL and
# leaves the others'.
# MEASURED 2026-10-08: 37 steps, 150 rows in five files, the heads and the time columns OpenFOAM's bytes;
# the values within 4.0e-13 on the host and 2.9e-12 on the GPU (p at the face location both times, the
# fields' own distance from OpenFOAM's on this case). Control: the face location's p 1.6e+00 off, the
# location inside a cell 1.6e-14.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
. "$(dirname "$0")/_common.sh"
STOP=0.12
HOST_BOUND=4e-12
DEVICE_BOUND=3e-11
CONTROL=1e-3
name="a probes function object on laminar/damBreak"
o="$W/pd_of"
cat > "$W/pd_functions.txt" <<'FO'
functions
{
    every
    {
        type            probes;
        libs            (sampling);
        fields          (p p_rgh alpha.water);
        probeLocations  ((0.292 0.05 0.0073) (0.1 0.1 0.0073) (0.5 0.02 0.0073) (5 5 5));
    }
    velocity
    {
        type            probes;
        libs            (sampling);
        fields          (U);
        probeLocations  ((0.292 0.05 0.0073) (0.1 0.1 0.0073) (0.5 0.02 0.0073));
    }
    written
    {
        type            probes;
        libs            (sampling);
        writeControl    writeTime;
        includeOutOfBounds false;
        fields          (p);
        probeLocations  ((0.1 0.1 0.0073) (5 5 5));
    }
}
FO
fo_stage "$o" "$LAM" "$STOP" adjustable 0.05 "$W/pd_functions.txt" || finish "$name"
files="every/0/p every/0/p_rgh every/0/alpha.water velocity/0/U written/0/p"
steps=$(grep -c "^Time = " "$o/log.interFoam")
[ "$(grep -vc '^#' "$o/postProcessing/every/0/p")" = "$steps" ] && [ "$steps" -ge 20 ] \
    && [ "$(grep -vc '^#' "$o/postProcessing/velocity/0/U")" = "$steps" ] \
    && [ "$(grep -vc '^#' "$o/postProcessing/written/0/p")" = "$(echo $(timedirs "$o") | wc -w)" ] \
    && grep -q "^# Probe 3 (5 5 5)  # Not Found" "$o/postProcessing/every/0/p" \
    && [ "$(sed -n 's/^# Time *//p' "$o/postProcessing/written/0/p" | wc -w)" = 1 ] \
    || { say "PREMISE  OpenFOAM: a row a step ($steps), a row a write, a location not found, a column left out" FAIL
         finish "$name"; }
arm()   # arm <run dir> <label> <bound>: every file of the run against OpenFOAM's
{
    local e="$1" ok=1 worst=0 rows=0 f r
    for f in $files; do
        r=$(fo_file "$o/postProcessing/$f" "$e/postProcessing/$f")
        read -r ro rb hd td gap <<< "$r"
        [ "$ro" = "$rb" ] && [ "$hd" = 0 ] && [ "$td" = 0 ] && within "$gap" "$3" || ok=0
        [ "$rb" = - ] && { rb=0; gap=1; }
        rows=$((rows + rb))
        worst=$(python3 -c "print('%.1e' % max(float('$worst'), float('$gap')))")
    done
    what="[$2] five files: OpenFOAM's heads byte for byte, its $rows rows, its time columns as text, every"
    what="$what value within $worst (bound $3)"
    [ $ok = 1 ] && say "$what" ok || say "$what" FAIL
}
fo_arm "$o" "$W/pd_host" host BRAE_X=1
arm "$W/pd_host" host $HOST_BOUND
fo_arm "$o" "$W/pd_device" device BRAE_X=1
arm "$W/pd_device" device $DEVICE_BOUND
fo_arm "$o" "$W/pd_nearest" host BRAE_CONTROL_PROBE_SEARCH=nearestCentre
read -r ro rb hd td face <<< "$(fo_file "$o/postProcessing/every/0/p" "$W/pd_nearest/postProcessing/every/0/p" 0)"
read -r ro rb hd td cell <<< "$(fo_file "$o/postProcessing/every/0/p" "$W/pd_nearest/postProcessing/every/0/p" 1)"
what="CONTROL  the nearest cell centre for the octree's order: the face location's p off by $face (above"
what="$what $CONTROL), the location inside a cell within $cell"
above "$face" $CONTROL && within "$cell" $HOST_BOUND && say "$what" ok || say "$what" FAIL
finish "a probes function object writes OpenFOAM's files on both loops, a location on a face in its cell"
