#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on RESTART (milestone E): a run started from a time
# directory OpenFOAM wrote, and from one brae wrote, by either code.
#
# WHAT OPENFOAM DOES (v2412, read in its source):
#   Time::setControls (Time.C:146-307)  the start time is the start DIRECTORY's name (startFrom startTime, firstTime or
#     latestTime); <start>/uniform/time gives deltaT -- under adjustTimeStep only, controlDict's otherwise -- deltaT0
#     and the time index, which timeStep writes count from; runTime/adjustable writes are measured from the start time
#   Time::writeTimeDict (TimeIO.C:508-539)  every write time holds uniform/time: value (18 digits), name, index,
#     deltaT and deltaT0 at writePrecision; uniform/functionObjects/functionObjectProperties beside it
#   rhoCentralFoam.C:80-81 and createFields.H  CoNum starts at 0, so setDeltaT.H grows the restart's first deltaT by
#     1.2; p, T, U are read, T is re-derived from he (hePsiThermo's calculate()), and rho, rhoU, rhoE are NO_READ:
#     rebuilt from p, T and U. Euler keeps no old-time field: no *_0 is written or read.
#   So a restart under adjustTimeStep is NOT its uninterrupted run continued: its deltaT sequence differs from the first
#   step on. brae must restart as OpenFOAM restarts, and is compared with OpenFOAM's restart, never with its own run.
# CASES (the tutorials' files; writePrecision 17 and purgeWrite 0 unless named)
#   fs17      forwardStep (adjustTimeStep, adjustable writes every 0.1, inletOutlet outlet) to t = 4
#   wedge17   wedge15Ma5 (FIXED deltaT 1e-4, runTime writes every 0.02) to 0.2
#   oblique17 obliqueShock (adjustTimeStep, adjustable writes every 1) to 10
#   rev17     a shockTube whose left end turns from outflow to inflow at t = 0.013-0.014 (U, T inletOutlet;
#             tests/rho_central_cases.py sodRevSet, sodRev) to 0.02
#   ts17      shockTube as shipped with writeControl timeStep, writeInterval 20, to 0.007
#   fsShip    forwardStep AS SHIPPED (writePrecision 6, purgeWrite 10) to 4
# RESTARTS, each started from OpenFOAM's continuous run (`o`) and from brae's (`b`):
#   fs17 at t = 1, 2 and 3, each to one time unit later; at 2 also written every step for 12 steps (`e`)
#   wedge17 at 0.06 and 0.12, oblique17 at 3 and 7, rev17 at 0.012, 0.013 and 0.014 (before, at and after the switch),
#   all to the case's endTime
#   ts17 from the write at time index 40, with timeStep writes every 30: OpenFOAM writes at index 60, 90, 120 (its
#             global grid), which an index counted from the restart would put at 70, 100
#   offGrid   ts17's index-40 directory restarted under writeControl adjustable 0.001 (the grid measured from there)
#   latest    fsShip's finished output, startFrom latestTime to 4.5: 6-digit fields and uniform/time, purgeWrite 10
#             again (previousWriteTimes starts empty, so the old 3.1 to 4 stay)
#   noUT      fs17's t = 2 with uniform/ removed: deltaT from controlDict, as OpenFOAM falls back
# REFERENCE  OpenFOAM's own restart of the same directory, serial; its spread decomposed in two (simple) and in four
#   (the frozen scotch partition, tests/rho_central_gate_lib.sh), each reported on its own.
# ENVELOPE  tests/rho_central_gate_lib.sh's, unchanged: every field at every write time within ten times the larger of
#   OpenFOAM's two spreads, never below 8e-14 (fsShip's 6-digit files: 1e-6). deltaT the same digits where adjustDeltaT
#   lands it, within 8e-14 where nothing does (timeStep writes).
# CHECKS beyond each arm's steps, names, deltaT, Courant, directories and fields:
#   TIME STATE    brae's uniform/time at every write time the same text as OpenFOAM's (value, name, index, deltaT,
#                 deltaT0), for the continuous runs on both arms and for every restart; under timeStep writes, where
#                 nothing lands deltaT, name and index the same text and value, deltaT, deltaT0 within 8e-14 (deltaT's rule)
#   CROSS         OpenFOAM restarts from brae's directories as it does from its own (each `b` reference above is
#                 OpenFOAM's run from brae's output)
#   UNINTERRUPTED each restart against its code's uninterrupted run, brae's beside OpenFOAM's (reported): the restart's
#                 first deltaT is 1.2 times the start's, so under adjustTimeStep the two parts in their fields; at a
#                 fixed deltaT only the rebuilt rho, rhoU, rhoE can differ
#   PURGE         after the latest restart, every directory left the same as OpenFOAM's
# CONTROLS, each red against OpenFOAM:
#   restart deltaT from controlDict (BRAE_CONTROL_RESTART_DELTAT_CONTROLDICT, time_controls.cuh): fs17 from o at 2
#   restartIndexZero     the index dropped: ts17's restart writes off OpenFOAM's directories
#   restartGridFromZero  the write grid from 0: offGrid's directories are not OpenFOAM's
#   restartIoExtrapolate the inletOutlet `value` replaced by the cells: rev17 from o at 0.013
#   noUniformTime        brae writes no uniform/time: its time state is not OpenFOAM's, and OpenFOAM restarted from it
#                        parts from OpenFOAM restarted from brae's full output
# NOT CLAIMED  restart with function objects (none in these tutorials), binary or compressed time directories, a
#   start directory with fields missing (brae's resolveStartTime skips a directory without U; OpenFOAM would stop),
#   LTS, mesh motion, multi-GPU.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
CASES="$ROOT/tests/rho_central_cases.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
for t in forwardStep wedge15Ma5 obliqueShock shockTube; do [ -d "$TUT/$t" ] || { echo "SKIP: no $t at $TUT"; exit 77; }; done
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-restart.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
fails=0
passes=0
. "$ROOT/tests/rho_central_gate_lib.sh"
ctl() { sed -i -E "$2" "$1/system/controlDict"; }
J=${BRAE_GATE_JOBS:-8}   # OpenFOAM runs at a time
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do sleep 1; done; }

# stage <dir> <tutorial> [writePrecision]: blockMesh (and setFields for shockTube), purgeWrite 0 at 17 digits
stage()
{
    local d=$1 tut=$2 prec=${3:-}
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$TUT/$tut/system" "$TUT/$tut/constant" "$d/"
    if [ -d "$TUT/$tut/0.orig" ]; then cp -r "$TUT/$tut/0.orig" "$d/0"; else cp -r "$TUT/$tut/0" "$d/"; fi
    [ -z "$prec" ] || ctl "$d" "s/^(writePrecision\s+)\S+;/\1$prec;/; s/^(purgeWrite\s+)\S+;/\10;/"
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || { fail "stage $d: blockMesh"; return 1; }
    if [ "$tut" = shockTube ]; then of "cd '$d' && setFields > log.setFields 2>&1" < /dev/null || { fail "stage $d: setFields"; return 1; }; fi
}
copy0() { rm -rf "$2"; mkdir -p "$2"; cp -r "$1/system" "$1/constant" "$1/0" "$2/"; }
# rstage <dir> <base case> <source run> <start> [controlDict sed]: the base's system and constant, the source run's
# start directory (with its uniform/), startFrom startTime there
rstage()
{
    local d=$1 base=$2 src=$3 t=$4 ed=${5:-}
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$base/system" "$base/constant" "$d/"
    cp -r "$src/$t" "$d/" || { fail "stage $(basename "$d"): no $t in $(basename "$src")"; return 1; }
    ctl "$d" "s/^(startFrom\s+)\S+;/\1startTime;/; s/^(startTime\s+)\S+;/\1$t;/"
    [ -z "$ed" ] || ctl "$d" "$ed"
    grep -qE "^startTime\s+$t;" "$d/system/controlDict" || fail "stage $(basename "$d"): startTime not set"
}
rcopy() { rm -rf "$2"; mkdir -p "$2"; cp -r "$1/system" "$1/constant" "$2/"; local t; for t in $(alltimes "$1"); do cp -r "$1/$t" "$2/"; done; }
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }

# ---- the continuous runs, OpenFOAM and brae ------------------------------------------------------------------------
stage "$W/fs17" forwardStep 17 || exit 1
stage "$W/wedge17" wedge15Ma5 17 || exit 1
stage "$W/oblique17" obliqueShock 17 || exit 1
stage "$W/rev17" shockTube 17 || exit 1
python3 "$CASES" sodRevSet "$W/rev17" && of "cd '$W/rev17' && setFields > log.setFields 2>&1" < /dev/null \
    && python3 "$CASES" sodRev "$W/rev17" || fail "stage rev17"
ctl "$W/rev17" 's/^(endTime\s+)\S+;/\10.02;/'
stage "$W/ts17" shockTube 17 || exit 1
ctl "$W/ts17" 's/^(writeControl\s+)\S+;/\1timeStep;/; s/^(writeInterval\s+)\S+;/\120;/'
stage "$W/fsShip" forwardStep || exit 1
grep -qE '^writeControl\s+timeStep;' "$W/ts17/system/controlDict" && grep -qE '^endTime\s+0.02;' "$W/rev17/system/controlDict" \
    || fail "stage: a controlDict edit did not take"
CONT="fs17 wedge17 oblique17 rev17 ts17 fsShip"
for c in $CONT; do copy0 "$W/$c" "$W/of_$c"; done
for c in $CONT; do ofrun "$W/of_$c" & done
for c in $CONT; do copy0 "$W/$c" "$W/b_$c"; braerun "$W/b_$c"; done
for c in fs17 rev17; do copy0 "$W/$c" "$W/h_$c"; braerun "$W/h_$c" BRAE_RHOCENTRALFOAM_ARM=cpu; done
copy0 "$W/fs17" "$W/cU_fs17"; ctlrun "$W/cU_fs17" BRAE_CONTROL_RCF=noUniformTime
wait
for c in $CONT; do grep -qx End "$W/of_$c/log" || fail "OpenFOAM did not finish $c"; done

# ---- the restarts ---------------------------------------------------------------------------------------------------
TSI=$(for t in $(times "$W/of_ts17"); do i=$(sed -n 's/^index *\([0-9]*\);/\1/p' "$W/of_ts17/$t/uniform/time"); [ "$i" = 40 ] && echo "$t"; done)
[ -n "$TSI" ] || fail "stage: ts17 wrote no time with index 40"
RES=""   # <name> <base> <start> [controlDict sed]: each staged from `o` (OpenFOAM's run) and `b` (brae's)
addr() { RES="$RES|$*"; }
for t in 1 2 3; do addr "fs17_$t fs17 $t s/^(endTime\\s+)\\S+;/\\1$((t + 1));/"; done
addr "fs17e_2 fs17 2 s/^(writeControl\\s+)\\S+;/\\1timeStep;/;s/^(writeInterval\\s+)\\S+;/\\11;/;s/^(endTime\\s+)\\S+;/\\12.0074;/"
for t in 0.06 0.12; do addr "wedge17_$t wedge17 $t"; done
for t in 3 7; do addr "oblique17_$t oblique17 $t"; done
for t in 0.012 0.013 0.014; do addr "rev17_$t rev17 $t"; done
addr "ts17_i40 ts17 $TSI s/^(writeInterval\\s+)\\S+;/\\130;/"
addr "offGrid ts17 $TSI s/^(writeControl\\s+)\\S+;/\\1adjustable;/;s/^(writeInterval\\s+)\\S+;/\\10.001;/"
RNAMES=""
IFS='|' read -r -a RSPECS <<< "${RES#|}"
for spec in "${RSPECS[@]}"
do
    read -r name base t ed <<< "$spec"
    RNAMES="$RNAMES $name"
    for src in o b
    do
        s="$W/of_$base"; [ "$src" = b ] && s="$W/b_$base"
        rstage "$W/${name}$src" "$W/$base" "$s" "$t" "$ed"
    done
done
# latest: the finished as-shipped run, every directory it left, startFrom latestTime to 4.5
for src in o b; do
    s="$W/of_fsShip"; [ "$src" = b ] && s="$W/b_fsShip"
    rcopy "$s" "$W/latest$src"
    ctl "$W/latest$src" 's/^(startFrom\s+)\S+;/\1latestTime;/; s/^(endTime\s+)\S+;/\14.5;/'
done
# noUT: fs17's t = 2 without its uniform/
rstage "$W/noUT" "$W/fs17" "$W/of_fs17" 2 's/^(endTime\s+)\S+;/\12.3;/'
rm -rf "$W/noUT/2/uniform"
ALLR=""
for n in $RNAMES latest; do ALLR="$ALLR ${n}o ${n}b"; done
ALLR="$ALLR noUT"
# OpenFOAM: serial on every restart, decomposed in two and four on every one compared within the envelope
for r in $ALLR; do rcopy "$W/$r" "$W/of_$r"; done
for r in $ALLR; do throttle; ofrun "$W/of_$r" & done
wait
for r in $ALLR; do grep -qx End "$W/of_$r/log" || fail "OpenFOAM did not finish the restart $r"; done
NP=""
for r in $ALLR; do case $r in ts17_i40*|offGrid*) ;; *) NP="$NP $r";; esac; done
for r in $NP; do rcopy "$W/$r" "$W/of_${r}_np2"; rcopy "$W/$r" "$W/of_${r}_np4"; done
for r in $NP; do throttle; ofnp "$W/of_${r}_np2" 2 & done
wait
for r in $NP; do throttle; ofnp "$W/of_${r}_np4" 4 & done
wait
for r in $NP; do for n in 2 4; do grep -qx End "$W/of_${r}_np$n/log" || fail "OpenFOAM in $n did not finish the restart $r"; done; done
# brae: CUDA on every restart, the host reference on fs17 at 2 and the rev17 ones
for r in $ALLR; do rcopy "$W/$r" "$W/b_$r"; braerun "$W/b_$r"; done
for r in fs17_2o fs17_2b fs17e_2o fs17e_2b rev17_0.013o rev17_0.013b; do rcopy "$W/$r" "$W/h_$r"; braerun "$W/h_$r" BRAE_RHOCENTRALFOAM_ARM=cpu; done
# the controls
rcopy "$W/fs17_2o" "$W/cD_fs17_2o";   ctlrun "$W/cD_fs17_2o" BRAE_CONTROL_RESTART_DELTAT_CONTROLDICT=1
rcopy "$W/ts17_i40o" "$W/cI_ts17";    ctlrun "$W/cI_ts17" BRAE_CONTROL_RCF=restartIndexZero
rcopy "$W/offGrido" "$W/cG_offGrid";  ctlrun "$W/cG_offGrid" BRAE_CONTROL_RCF=restartGridFromZero
rcopy "$W/rev17_0.013o" "$W/cX_rev17"; ctlrun "$W/cX_rev17" BRAE_CONTROL_RCF=restartIoExtrapolate
# OpenFOAM restarted from the noUniformTime control's output at 2, and from brae's full output (the reference)
rstage "$W/cU_r" "$W/fs17" "$W/cU_fs17" 2 's/^(endTime\s+)\S+;/\13;/'
rcopy "$W/cU_r" "$W/of_cU_r"; ofrun "$W/of_cU_r"

# ---- the time state: uniform/time as OpenFOAM writes it ---------------------------------------------------------------
tstate()   # tstate <name> <brae dir> <OF dir> [exact|bound]
{
    # name and index always the same text; value, deltaT and deltaT0 the same text where adjustDeltaT lands deltaT on the
    # write times (exact), within BOUND where nothing does (timeStep writes: deltaT carries the Courant number's last
    # bits, and the time value accumulates them) -- the rule every arm already applies to deltaT
    local r n miss v wv nm ix dt wdt dt0 wdt0 mode=${4:-exact}
    r=$(python3 "$FIELDS" timedict "$2" "$3" 2>&1)
    read -r _ n _ miss _ v wv _ nm _ ix _ dt wdt _ dt0 wdt0 <<< "$r"
    local okk=0
    if [ -n "$n" ] && [ "$n" -gt 0 ] && [ "$miss" = 0 ] && [ "$nm" = "$n" ] && [ "$ix" = "$n" ]
    then
        if [ "$mode" = exact ]
        then
            [ "$v" = "$n" ] && [ "$dt" = "$n" ] && [ "$dt0" = "$n" ] && okk=1
        else
            le "$wv" "$BOUND" && le "$wdt" "$BOUND" && le "$wdt0" "$BOUND" && okk=1
        fi
    fi
    if [ "$okk" = 1 ] && [ "$mode" = exact ]
    then
        ok "$1: uniform/time at all $n write times the same text as OpenFOAM's (value, name, index, deltaT, deltaT0)"
    elif [ "$okk" = 1 ]
    then
        ok "$1: uniform/time at all $n write times: name and index the same text as OpenFOAM's, value, deltaT and deltaT0 within $BOUND (worst $wv, $wdt, $wdt0; the same text at $v, $dt, $dt0)"
    else
        fail "$1: uniform/time against OpenFOAM's ($mode): ${r:-nothing}"
    fi
}
tmode() { case $1 in ts17*|fs17e*|offGrid*) echo bound ;; *) echo exact ;; esac; }
for c in $CONT; do tstate "$c (CUDA, uninterrupted)" "$W/b_$c" "$W/of_$c" "$(tmode "$c")"; done
for c in fs17 rev17; do tstate "$c (host reference, uninterrupted)" "$W/h_$c" "$W/of_$c"; done
for r in $ALLR; do tstate "restart $r (CUDA)" "$W/b_$r" "$W/of_$r" "$(tmode "$r")"; done

# ---- each restart against OpenFOAM's restart of the same directory ---------------------------------------------------
for r in $ALLR
do
    case $r in
        ts17_i40*|offGrid*) refs=""; dm=bound ;;
        fs17e*) refs=$(spread "$r"); dm=bound ;;
        latest*) refs=$(spread "$r"); dm=exact ;;
        wedge17*) refs=$(spread "$r"); dm=fixed ;;
        *) refs=$(spread "$r"); dm=exact ;;
    esac
    floor=$BOUND; case $r in latest*) floor=1e-6 ;; esac
    if [ -z "$refs" ]
    then
        # ts17/offGrid: the write cadence's checks (directories, names, steps); fields at the shockTube gates' floor
        k=$(cmpdir "$W/b_$r")
        python3 "$STEPS" compare "$W/b_$r/log" "$W/of_$r/log" > "$k/steps.cmp" 2>&1
        read -r nb no <<< "$(sed -n 's/^steps //p' "$k/steps.cmp")"; read -r nm <<< "$(sed -n 's/^names //p' "$k/steps.cmp")"
        read -r _ wdt _ <<< "$(sed -n 's/^deltaT //p' "$k/steps.cmp")"
        if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$nm" = "$nb" ] && le "$wdt" "$BOUND" && [ "$(times "$W/b_$r")" = "$(times "$W/of_$r")" ]
        then
            ok "restart $r (CUDA): $nb steps, every name and directory as OpenFOAM's ($(times "$W/of_$r" | wc -w) written: $(times "$W/of_$r" | cut -c1-80)); deltaT within $BOUND (worst $wdt)"
        else
            fail "restart $r: steps ${nb:-?}/${no:-?}, names ${nm:-?}, deltaT ${wdt:-?}, wrote '$(times "$W/b_$r")' against '$(times "$W/of_$r")'"
        fi
        worst=0
        for t in $(times "$W/of_$r"); do v=$(python3 "$FIELDS" maxrel "$W/b_$r/$t" "$W/of_$r/$t" | awk '{print $2}'); worst=$(mx "$worst" "$v"); done
        if le "$worst" "$BOUND"; then ok "restart $r (CUDA): U p T rho within $BOUND at every write time (worst $worst)"; else fail "restart $r: fields $worst"; fi
        continue
    fi
    arm "restart $r (CUDA)" "$W/b_$r" "$W/of_$r" "$refs" "$floor" "$dm"
    [ -d "$W/h_$r" ] && arm "restart $r (host reference)" "$W/h_$r" "$W/of_$r" "$refs" "$floor" "$dm"
done
# the inletOutlet patches' written values through the switch
for r in rev17_0.012o rev17_0.013o rev17_0.014o rev17_0.013b fs17_2o fs17_2b
do
    pa=sides; case $r in fs17*) pa=outlet ;; esac
    worst=0; over=""
    for t in $(times "$W/of_$r"); do
        v=$(python3 "$FIELDS" patchdiff "$W/b_$r/$t" "$W/of_$r/$t" "$pa" | awk '{print $4}' | sort -g | tail -n 1)
        sp=0; for rr in $(spread "$r"); do sp=$(mx "$sp" "$(python3 "$FIELDS" patchdiff "$rr/$t" "$W/of_$r/$t" "$pa" | awk '{print $4}' | sort -g | tail -n 1)"); done
        bd=$(mx "$BOUND" "$(python3 -c "print(10*float('$sp'))")"); le "$v" "$bd" || over="$over $t"; worst=$(mx "$worst" "$v")
    done
    if [ -z "$over" ]; then ok "restart $r (CUDA): $pa's written U, T, rho within ten times OpenFOAM's own spread at every write time (worst $worst)"
    else fail "restart $r: $pa's values above the bound at$over"; fi
done

# ---- purgeWrite across the latest restart ------------------------------------------------------------------------------
for src in o b; do
    if [ "$(alltimes "$W/b_latest$src")" = "$(alltimes "$W/of_latest$src")" ]
    then
        ok "latest$src: purgeWrite 10 across the restart leaves OpenFOAM's $(alltimes "$W/of_latest$src" | wc -w) directories ($(alltimes "$W/of_latest$src" | cut -c1-90))"
    else
        fail "latest$src: brae left '$(alltimes "$W/b_latest$src")', OpenFOAM '$(alltimes "$W/of_latest$src")'"
    fi
done

# ---- restart against the uninterrupted run (reported) ----------------------------------------------------------------
for spec in "${RSPECS[@]}"
do
    read -r name base t _ <<< "$spec"
    case $name in ts17_i40|offGrid) continue ;; esac
    tl=$(times "$W/of_${name}o" | awk '{print $NF}')
    [ -d "$W/of_$base/$tl" ] || continue
    eo=$(python3 "$FIELDS" maxrel "$W/of_${name}o/$tl" "$W/of_$base/$tl" | awk '{print $2}')
    eb=$(python3 "$FIELDS" maxrel "$W/b_${name}b/$tl" "$W/b_$base/$tl" | awk '{print $2}')
    d1o=$(awk '/^deltaT = /{print $3; exit}' "$W/of_${name}o/log"); d1c=$(awk -v t="$t" '$0 == "Time = " t {f = 1; next} f && /^deltaT = /{print $3; exit}' "$W/of_$base/log")
    echo "     restart against uninterrupted, $name: at t = $tl OpenFOAM's $eo, brae's $eb; the first deltaT after $t: restart ${d1o:-the fixed one}, uninterrupted ${d1c:-the fixed one}"
done

# ---- CPU against GPU (reported) ----------------------------------------------------------------------------------------
for r in fs17e_2o fs17e_2b rev17_0.013o; do
    t1=$(times "$W/b_$r" | awk '{print $1}'); tl=$(times "$W/b_$r" | awk '{print $NF}')
    echo "     CPU against GPU, restart $r: at t = $t1 $(python3 "$FIELDS" maxrel "$W/h_$r/$t1" "$W/b_$r/$t1" | cut -d' ' -f2-), at t = $tl $(python3 "$FIELDS" maxrel "$W/h_$r/$tl" "$W/b_$r/$tl" | cut -d' ' -f2-)"
done

# ---- conservation and stability --------------------------------------------------------------------------------------
for d in b_fs17_2o b_fs17_2b h_fs17_2o b_wedge17_0.06o b_oblique17_7o b_rev17_0.013o b_latesto; do balance "$d"; done

# ---- controls --------------------------------------------------------------------------------------------------------
red "restart deltaT from controlDict, fs17 from OpenFOAM's t = 2" "$W/cD_fs17_2o" "$W/of_fs17_2o" "$(spread fs17_2o)"
for spec in "restartIndexZero cI_ts17 ts17_i40o" "restartGridFromZero cG_offGrid offGrido"; do
    read -r nm c r <<< "$spec"
    if [ "$(times "$W/$c")" != "$(times "$W/of_$r")" ]
    then
        ok "control $nm red: brae wrote '$(times "$W/$c" | cut -c1-70)', OpenFOAM '$(times "$W/of_$r" | cut -c1-70)'"
    else
        fail "control $nm: the same directories as OpenFOAM's: not red"
    fi
done
red "restartIoExtrapolate, rev17 from OpenFOAM's t = 0.013" "$W/cX_rev17" "$W/of_rev17_0.013o" "$(spread rev17_0.013o)"
r=$(python3 "$FIELDS" timedict "$W/cU_fs17" "$W/of_fs17" 2>&1)   # its `missing` count
if [ "$(echo "$r" | awk '{print $4}')" != 0 ]; then ok "control noUniformTime red: $(echo "$r" | awk '{print $4}') of $(echo "$r" | awk '{print $2}') write times without uniform/time"; else fail "control noUniformTime: $r"; fi
red "noUniformTime, OpenFOAM restarted from it at t = 2 against OpenFOAM restarted from brae's full output" \
    "$W/of_cU_r" "$W/of_fs17_2b" "$(spread fs17_2b)"

# ---- performance (reported) ---------------------------------------------------------------------------------------------
for r in fs17_2o fs17_2b; do
    echo "     runtime restart $r: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$r/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$r/log")"
done
echo "rho_central_restart_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge "${BRAE_RCF_E_MIN_PASSES:-1}" ]
