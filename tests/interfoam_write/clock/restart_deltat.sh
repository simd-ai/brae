#!/usr/bin/env bash
# The gate: THE STEP A RESTART BEGINS WITH, brae's steps against OpenFOAM's own log. Time::setControls takes
# controlDict's deltaT and then, under `adjustTimeStep` alone, the `deltaT` of <start>/uniform/time
# (Time.C:279-292): an adaptive run continues the step it was written with. Both loops began every restart
# from controlDict's deltaT (and said so in a notice), so a continuation re-grew its step at 1.2 a step.
# THE FIXTURE is laminar/damBreak as its clock ships (adjustTimeStep, maxCo 1, deltaT 0.001, a write every
# 0.05) with every solve pinned: real interFoam runs to 0.1, and then OpenFOAM and both of brae's loops
# restart from OPENFOAM'S 0.1 directory and run to 0.15.
# Three checks. (1) THE PREMISES, on OpenFOAM, as numbers: its first step after the restart is not the one a
# start from controlDict's deltaT takes (1.2 x 0.001); and the file's `deltaT0` is INERT -- Time::operator++
# overwrites it with deltaTSave_ before anything asks -- so OpenFOAM restarted with that entry doubled writes
# the same bytes. That is why brae reads `deltaT` there and not `deltaT0`. (2) Each loop takes OpenFOAM's
# steps, one for one, and its 0.15 is OpenFOAM's within the bound. (3) CONTROL:
# BRAE_CONTROL_RESTART_DELTAT_CONTROLDICT=1 begins from controlDict's deltaT again and re-grows from it.
# MEASURED 2026-10-07: the stored deltaT is 2.2964e-03 and OpenFOAM's first step 0.002 (the write cadence lands
# it), then 29 steps to 0.15; both loops take the same 29, every one OpenFOAM's to the last printed digit
# (0.0e+00), and their 0.15 is 6.6e-15 from OpenFOAM's (p_rgh) -- so the bounds are 1e-13 for a step, headroom
# for another compiler's contraction as the other clock gates keep, and 1e-13 for a file. The control's first
# three steps are 1.1905e-03, 1.3946e-03, 1.5805e-03, and it takes 30.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
o="$W/rd_of"
stage "$LAM" "$o" 0.1 adjustable 0.05 0 > "$W/rd_stage.txt" 2>&1 \
    || { say "laminar/damBreak did not stage" FAIL; finish "the step a restart begins with"; }
python3 - "$o/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t = re.sub(r'(tolerance\s+)[^;]+;', r'\g<1>1e-13;', t)
t = re.sub(r'(relTol\s+)[^;]+;', r'\g<1>0;', t)
open(p, 'w').write(t)
PY
runof "$o"
[ -f "$o/0.1/uniform/time" ] \
    || { say "OpenFOAM wrote no 0.1/uniform/time" FAIL; finish "the step a restart begins with"; }
restage()   # restage <dir> [factor on the file's deltaT0]: OpenFOAM's 0.1, restarted to 0.15
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$o/constant" "$o/system" "$o/0.1" "$1/"
    sed -i 's/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         0.15;/' "$1/system/controlDict"
    [ -z "${2:-}" ] || python3 - "$1/0.1/uniform/time" "$2" <<'PY'
import re, sys
p, k = sys.argv[1], float(sys.argv[2])
t = open(p).read()
m = re.search(r'^deltaT0\s+(\S+);', t, flags=re.M)
t = t.replace(m.group(0), 'deltaT0         %.17g;' % (k*float(m.group(1))), 1)
open(p, 'w').write(t)
PY
}
# stepsOf <OpenFOAM log>, stepsBrae <brae log>: the step sizes taken, one a line
stepsOf() { sed -n -E 's/^deltaT = (\S+)$/\1/p' "$1"; }
stepsBrae() { sed -n -E 's/^ *t = \S+ +dt = (\S+) .*/\1/p' "$1"; }
restage "$W/rd_rs_of"
runof "$W/rd_rs_of"
restage "$W/rd_rs_of_dt0" 2
runof "$W/rd_rs_of_dt0"
stored=$(sed -n -E 's/^deltaT\s+(\S+);/\1/p' "$o/0.1/uniform/time")
first=$(stepsOf "$W/rd_rs_of/log.interFoam" | head -1)
n=0
for f in $(cd "$W/rd_rs_of/0.15" && find . -type f | sort); do
    cmp -s "$W/rd_rs_of/0.15/$f" "$W/rd_rs_of_dt0/0.15/$f" || n=$((n + 1))
done
what="PREMISES OpenFOAM's first step after the restart is $first (stored deltaT $stored; 1.2 x controlDict's is"
what="$what 0.0012); with the file's deltaT0 doubled its 0.15 differs in $n files"
[ -d "$W/rd_rs_of/0.15" ] && above "$first" 0.0014 && [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
for v in host device control; do
    restage "$W/rd_$v"
    case $v in
        host)    runbrae "$W/rd_$v" host BRAE_X=1 ;;
        device)  runbrae "$W/rd_$v" device BRAE_X=1 ;;
        control) runbrae "$W/rd_$v" host BRAE_CONTROL_RESTART_DELTAT_CONTROLDICT=1 ;;
    esac
done
# seq <brae log>: "<steps OpenFOAM> <steps brae> <worst relative gap of a step, over the steps both took>"
seq()
{
    python3 - <(stepsOf "$W/rd_rs_of/log.interFoam") <(stepsBrae "$1") <<'PY'
import sys
a = [float(x) for x in open(sys.argv[1]).read().split()]
b = [float(x) for x in open(sys.argv[2]).read().split()]
print(len(a), len(b), '%.1e' % max(abs(x - y)/x for x, y in zip(a, b)) if a and b else '-')
PY
}
BS=1e-13
BF=1e-13
read -r no nh wh <<< "$(seq "$W/rd_host/log.brae")"
read -r no nd wd <<< "$(seq "$W/rd_device/log.brae")"
python3 "$CMP" "$W/rd_rs_of" "$W/rd_host" 0.15 > "$W/cmp_rd_host.txt" 2>&1
python3 "$CMP" "$W/rd_rs_of" "$W/rd_device" 0.15 > "$W/cmp_rd_device.txt" 2>&1
what="[host, device] OpenFOAM's $no steps, one for one: $nh and $nd taken, the worst step $wh and $wd from"
what="$what OpenFOAM's (bound $BS); 0.15 within $BF"
[ "$no" -gt 10 ] && [ "$nh" = "$no" ] && [ "$nd" = "$no" ] && within "$wh" "$BS" && within "$wd" "$BS" \
    && judge "the restart's 0.15, host" "$W/cmp_rd_host.txt" "$BF" "$W/rd_rs_of/log.interFoam" \
    && judge "the restart's 0.15, device" "$W/cmp_rd_device.txt" "$BF" "$W/rd_rs_of/log.interFoam" \
    && say "$what" ok || say "$what" FAIL
c1=$(stepsBrae "$W/rd_control/log.brae" | head -1)
nc=$(stepsBrae "$W/rd_control/log.brae" | wc -l)
what="CONTROL  from controlDict's deltaT: the first step is $c1, not OpenFOAM's $first, and it takes $nc steps"
python3 -c "import sys; a, b = float('$c1'), float('$first'); sys.exit(0 if a <= 0.0012 and abs(a - b) > 0.1*b else 1)" \
    && [ "$nc" != "$no" ] && say "$what" ok || say "$what" FAIL
finish "a restart under adjustTimeStep begins with the step the run was written with"
