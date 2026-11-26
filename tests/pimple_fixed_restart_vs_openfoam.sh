#!/usr/bin/env bash
# pimpleFoam's restart state at a FIXED deltaT against OpenFOAM v2412: the write cadence, the time index and the write
# names across a `startFrom latestTime` restart.
#
# WHAT IT HOLDS. OpenFOAM's Time::setControls takes a restart's start time from the directory's NAME and its time index
# from the directory's uniform/time (Time.C:193, 302-307; uniform/time's `value` only feeds a warning, :310-346), every
# write writes uniform/time (TimeIO.C:508-555), and Time::operator++ counts runTime writes from the resolved start and
# timeStep writes on the absolute index (Time.C:1111-1129). At a fixed deltaT brae's pimpleFoam counted runTime writes
# from controlDict's startTime (a restart from 0.1 wrote at its first step, 0.101), wrote no uniform/time and counted
# timeStep writes and functionObject steps from the restart, and named its own final write startTime + deltaT*steps
# (another double than the time it reached: at timePrecision 18 it wrote 0.200000000000000011 beside 0.20000000000000015).
#
# THE CASE. incompressible/pimpleFoam/RAS/TJunction as shipped but for a fixed deltaT 0.001 (adjustTimeStep no), written
# at writePrecision 17, with its time-dependent total-pressure inlet (p0 table (0 10) (1 40)). Restarts from 0.1 (index 100).
#
# ARMS
#   A  RESTART FIDELITY at timePrecision 18 (the 0.1 directory carries the accumulated time 0.10000000000000007): brae
#      stopped at 0.1 and restarted equals brae continuous at 0.2, every entry of U p k epsilon nut (pimple_entry_compare.py)
#      <= 1e-12; OpenFOAM's own restart against its continuous run is asserted alongside (the premise: exact).
#      WITNESS, numerical sensitivity, not a control: at timePrecision 6 (as shipped) the name rounds the time to 0.1,
#      as OpenFOAM's does, and the restart must differ from the continuous run (rel > 0): the arm can see a 6.9e-17
#      difference in the restart's time. It is reported beside OpenFOAM's own (2.9e-14). Left open, not suppressed: brae
#      amplifies that rounding ~6,700 times more than OpenFOAM on this case.
#      timePrecision does not move the continuous fields: brae continuous at 6 and at 18 equal at 0.2 (0).
#   B  runTime ORIGIN: restarted from 0.1 with writeInterval 0.03 to 0.19 (0.1 is not a multiple of 0.03): brae's written
#      times equal OpenFOAM's (0.13 0.16 0.19). Control BRAE_CONTROL_FIXED_WRITE_ORIGIN=1 must differ.
#   C  timeStep and INDEX: restarted from 0.1 with writeControl timeStep, writeInterval 30, to 0.18; a forces function
#      object with executeControl/writeControl timeStep 30. brae from its own 0.1, brae from OpenFOAM's, OpenFOAM from
#      brae's: written times, uniform/time `index` at each write and the forces rows' times all equal OpenFOAM-from-its-own
#      (0.12 0.15 0.18, index 120 150 180). Control BRAE_CONTROL_FIXED_RESTART_INDEX=1 (no uniform/time written or read,
#      steps counted from the restart) must differ.
#      C0, an older case: the same from OpenFOAM's 0.1 with uniform/ removed, to 0.19: OpenFOAM falls back to index 0
#      (0.13 0.16 0.19) and so must brae, writes and forces rows; under the control brae's forces rows land on 0.12 0.15
#      0.18 (an index estimated as startTime/deltaT) and must differ.
#   D  FINAL NAME at timePrecision 18: brae's time directories equal OpenFOAM's, continuous and stopped at 0.1. Control
#      BRAE_CONTROL_FINAL_NAME_PRODUCT=1 must differ.
# Each control is a run that completes; it must go red on the check it guards.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=1e-12
FIELDS="U p k epsilon nut"
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUT="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}/incompressible/pimpleFoam/RAS/TJunction"
[ -d "$TUT/system" ] || { echo "SKIP: no pimpleFoam/RAS/TJunction tutorial at $TUT"; exit 77; }
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
W=${BRAE_FR_KEEP:-$(mktemp -d)}
[ -n "${BRAE_FR_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-96s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

setkey()
{
    sed -i -E "s/^$2[[:space:]][^;]*;/$2 $3;/" "$1/system/controlDict"
    grep -q -E "^$2 $3;" "$1/system/controlDict" || { echo "FAIL: could not set $2 $3 in $1"; exit 1; }
}
ofrun() { ( cd "$1" && pimpleFoam > "log.$2" 2>&1 ) || { echo "FAIL: OpenFOAM in $1 ($2)"; tail -5 "$1/log.$2"; exit 1; }; }
brrun()
{
    local d=$1 tag=$2
    shift 2
    ( cd "$d" && env "$@" "$BIN" -case . > "log.$tag" 2>&1 ) || { echo "FAIL: brae in $d ($tag)"; tail -5 "$d/log.$tag"; exit 1; }
}
written() { (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -vx 0 | sort -g | tr '\n' ' ' | sed 's/ $//'); }
lastdir() { written "$1" | tr ' ' '\n' | tail -n 1; }
utindex() { sed -n 's/^index[[:space:]]\+\([0-9]*\);/\1/p' "$1/uniform/time" 2> /dev/null; }
indices() { local t; for t in $(written "$1"); do [ "$t" = "$2" ] && continue; printf '%s:%s ' "$t" "$(utindex "$1/$t")"; done | sed 's/ $//'; }
forcerows() { awk '!/^#/ && NF { printf "%.12g ", $1 }' "$1"/postProcessing/forces1/*/force.dat 2> /dev/null | sed 's/ $//'; }
rel() { python3 "$CMP" "$@" $FIELDS | tail -n 1; }
relnum() { echo "$1" | awk '{ print $2 }'; }
le() { python3 -c "import sys; sys.exit(0 if float('$1') <= float('$2') else 1)"; }
gt0() { python3 -c "import sys; sys.exit(0 if float('$1') > 0 else 1)"; }

# ---- staging: the tutorial at a fixed deltaT, written every 0.1 to 0.2 at writePrecision 17 ------------------------------
mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
for k in "adjustTimeStep +yes" "deltaT +0.001;" "timePrecision +6;"; do
    grep -qE "^$k" "$W/base/system/controlDict" || { echo "FAIL: the tutorial no longer has \`$k\` -- the gate's premise moved"; exit 1; }
done
grep -q 'table' "$W/base/0/p" || { echo "FAIL: TJunction's inlet p0 is no longer a table -- the time-dependent inlet is the premise"; exit 1; }
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
setkey "$W/base" adjustTimeStep no
setkey "$W/base" endTime 0.2
setkey "$W/base" writePrecision 17
setkey "$W/base" writeControl runTime
setkey "$W/base" writeInterval 0.1
stage() { mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; [ -n "${2:-}" ] && setkey "$1" timePrecision "$2"; true; }
# restart <dir> <source case> <time dir>: the staging with <source>'s <time dir> beside its 0, startFrom latestTime
restart() { stage "$1" "${4:-}"; cp -r "$2/$3" "$1/"; setkey "$1" startFrom latestTime; }
# the forces function object, timeStep 30, in place of the tutorial's probes
forces()
{
    python3 - "$1/system/controlDict" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
i = s.index('\nfunctions')
j = s.index('{', i)
d = 0
for k in range(j, len(s)):
    d += {'{': 1, '}': -1}.get(s[k], 0)
    if d == 0:
        break
fo = '''
functions
{
    forces1
    {
        type            forces;
        libs            (forces);
        patches         (defaultFaces);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0 0 0);
        executeControl  timeStep;
        executeInterval 30;
        writeControl    timeStep;
        writeInterval   30;
    }
}'''
open(p, 'w').write(s[:i] + fo + s[k + 1:])
EOF
}
echo "pimple_fixed_restart_vs_openfoam: TJunction at a fixed deltaT 0.001, restarts from 0.1"

# ---- A and D: timePrecision 18 -----------------------------------------------------------------------------------------
stage "$W/of18" 18; ofrun "$W/of18" run
stage "$W/br18" 18; brrun "$W/br18" run
stage "$W/br18r" 18; setkey "$W/br18r" endTime 0.1; brrun "$W/br18r" first
first18=$(written "$W/br18r")
setkey "$W/br18r" startFrom latestTime; setkey "$W/br18r" endTime 0.2; brrun "$W/br18r" restart
s18=$(written "$W/of18" | cut -d' ' -f1)
restart "$W/of18r" "$W/of18" "$s18" 18; ofrun "$W/of18r" restart
e18=$(lastdir "$W/of18")
r=$(rel "$W/of18" "$W/of18r" "$e18")
le "$(relnum "$r")" $BOUND && say "A  premise: OpenFOAM restarted at $s18 = OpenFOAM continuous at $e18 ($r)" PASS \
                           || say "A  premise: OpenFOAM restarted differs from its continuous run ($r)" FAIL
[ "$(lastdir "$W/br18r")" = "$e18" ] || say "A  brae restarted ends at $(lastdir "$W/br18r"), OpenFOAM at $e18" FAIL
r=$(rel "$W/br18" "$W/br18r" "$e18")
le "$(relnum "$r")" $BOUND && say "A  brae restarted = brae continuous at $e18, every entry <= $BOUND ($r)" PASS \
                           || say "A  brae restarted vs continuous at $e18: $r (bound $BOUND)" FAIL
# the witness: timePrecision 6, both codes
stage "$W/br6" 6; brrun "$W/br6" run
stage "$W/br6r" 6; setkey "$W/br6r" endTime 0.1; brrun "$W/br6r" first
cp -r "$W/br6r" "$W/br6first"
setkey "$W/br6r" startFrom latestTime; setkey "$W/br6r" endTime 0.2; brrun "$W/br6r" restart
stage "$W/of6" 6; setkey "$W/of6" endTime 0.1; ofrun "$W/of6" run
r6=$(rel "$W/br6" "$W/br6r" 0.2)
gt0 "$(relnum "$r6")" && say "A  witness: at timePrecision 6 brae's restart from \"0.1\" differs from continuous at 0.2 ($r6)" PASS \
                      || say "A  witness: at timePrecision 6 the restart equals the continuous run -- the arm cannot see the time" FAIL
mkdir -p "$W/br18as6"; ln -s "$W/br18/$e18" "$W/br18as6/0.2"
r=$(rel "$W/br6" "$W/br18as6" 0.2)
[ "$(relnum "$r")" = "0.000e+00" ] && say "A  timePrecision 6 and 18 give the same continuous fields at 0.2 ($r)" PASS \
                                    || say "A  timePrecision changed the continuous fields: $r" FAIL
# D: the time directories
wo=$(written "$W/of18"); wb=$(written "$W/br18")
[ "$wb" = "$wo" ] && say "D  continuous, timePrecision 18: brae writes OpenFOAM's directories ($wb)" PASS \
                  || say "D  continuous: brae [$wb], OpenFOAM [$wo]" FAIL
[ "$first18" = "$s18" ] && say "D  stopped at 0.1: brae writes $first18 alone, OpenFOAM's name" PASS \
                        || say "D  stopped at 0.1: brae [$first18], OpenFOAM [$s18]" FAIL
wo=$(written "$W/of18r"); wb=$(written "$W/br18r")
[ "$wb" = "$wo" ] && say "D  restarted: brae writes OpenFOAM's directories ($wb)" PASS || say "D  restarted: brae [$wb], OpenFOAM [$wo]" FAIL
stage "$W/br18x" 18; brrun "$W/br18x" control BRAE_CONTROL_FINAL_NAME_PRODUCT=1
wb=$(written "$W/br18x")
[ "$wb" != "$(written "$W/of18")" ] && say "D  control FINAL_NAME_PRODUCT: brae writes [$wb]" PASS \
                                    || say "D  control FINAL_NAME_PRODUCT is not caught: [$wb]" FAIL

# ---- B: runTime from a start that is not a multiple of writeInterval ----------------------------------------------------
bset() { setkey "$1" writeInterval 0.03; setkey "$1" endTime 0.19; }
restart "$W/ofB" "$W/of6" 0.1; bset "$W/ofB"; ofrun "$W/ofB" run
restart "$W/brB" "$W/br6first" 0.1; bset "$W/brB"; brrun "$W/brB" run
wo=$(written "$W/ofB"); wb=$(written "$W/brB")
[ "$wb" = "$wo" ] && say "B  runTime 0.03 from 0.1: brae writes OpenFOAM's times ($wb)" PASS || say "B  brae [$wb], OpenFOAM [$wo]" FAIL
restart "$W/brBx" "$W/br6first" 0.1; bset "$W/brBx"; brrun "$W/brBx" control BRAE_CONTROL_FIXED_WRITE_ORIGIN=1
wb=$(written "$W/brBx")
[ "$wb" != "$wo" ] && say "B  control FIXED_WRITE_ORIGIN: brae writes [$wb]" PASS || say "B  control FIXED_WRITE_ORIGIN is not caught: [$wb]" FAIL

# ---- C: writeControl timeStep and the index -----------------------------------------------------------------------------
cset() { setkey "$1" writeControl timeStep; setkey "$1" writeInterval 30; setkey "$1" endTime "$2"; forces "$1"; }
restart "$W/ofC" "$W/of6" 0.1; cset "$W/ofC" 0.18; ofrun "$W/ofC" run
restart "$W/brC" "$W/br6first" 0.1; cset "$W/brC" 0.18; brrun "$W/brC" run
restart "$W/boC" "$W/of6" 0.1; cset "$W/boC" 0.18; brrun "$W/boC" run
restart "$W/obC" "$W/br6first" 0.1; cset "$W/obC" 0.18; ofrun "$W/obC" run
wo=$(written "$W/ofC"); io=$(indices "$W/ofC" 0.1); fo=$(forcerows "$W/ofC")
echo "     OpenFOAM from its own 0.1 (index $(utindex "$W/of6/0.1")): writes [$wo], index [$io], forces rows [$fo]"
[ -n "$fo" ] || say "C  OpenFOAM's forces function object wrote no rows -- the forces check cannot judge" FAIL
for c in brC:"brae from brae's 0.1" boC:"brae from OpenFOAM's 0.1" obC:"OpenFOAM from brae's 0.1"; do
    d=${c%%:*}; what=${c#*:}
    wb=$(written "$W/$d"); ib=$(indices "$W/$d" 0.1); fb=$(forcerows "$W/$d")
    [ "$wb $ib $fb" = "$wo $io $fo" ] && say "C  $what: writes, uniform/time index and forces rows are OpenFOAM's" PASS \
                                      || say "C  $what: writes [$wb] index [$ib] forces [$fb]" FAIL
done
stage "$W/brCx0" 6; setkey "$W/brCx0" endTime 0.1; brrun "$W/brCx0" first BRAE_CONTROL_FIXED_RESTART_INDEX=1
restart "$W/brCx" "$W/brCx0" 0.1; cset "$W/brCx" 0.18; brrun "$W/brCx" control BRAE_CONTROL_FIXED_RESTART_INDEX=1
wb=$(written "$W/brCx")
[ "$wb" != "$wo" ] && say "C  control FIXED_RESTART_INDEX: brae writes [$wb]" PASS || say "C  control FIXED_RESTART_INDEX is not caught: [$wb]" FAIL
# C0: an older case, no uniform/ at the start
restart "$W/ofC0" "$W/of6" 0.1; rm -rf "$W/ofC0/0.1/uniform"; cset "$W/ofC0" 0.19; ofrun "$W/ofC0" run
restart "$W/brC0" "$W/of6" 0.1; rm -rf "$W/brC0/0.1/uniform"; cset "$W/brC0" 0.19; brrun "$W/brC0" run
wo=$(written "$W/ofC0"); fo=$(forcerows "$W/ofC0"); wb=$(written "$W/brC0"); fb=$(forcerows "$W/brC0")
[ "$wb $fb" = "$wo $fo" ] && say "C0 no uniform/time at the start: index 0 as OpenFOAM's, writes [$wb] forces rows [$fb]" PASS \
                          || say "C0 no uniform/time: brae writes [$wb] forces [$fb]; OpenFOAM [$wo] [$fo]" FAIL
restart "$W/brC0x" "$W/of6" 0.1; rm -rf "$W/brC0x/0.1/uniform"; cset "$W/brC0x" 0.19
brrun "$W/brC0x" control BRAE_CONTROL_FIXED_RESTART_INDEX=1
fb=$(forcerows "$W/brC0x")
[ "$fb" != "$fo" ] && say "C0 control FIXED_RESTART_INDEX: brae's forces rows [$fb]" PASS \
                   || say "C0 control FIXED_RESTART_INDEX is not caught: forces rows [$fb]" FAIL

# ---- reported, not gated: OpenFOAM's own restart at timePrecision 6, and brae against OpenFOAM ----------------------------
stage "$W/of6c" 6; ofrun "$W/of6c" run
restart "$W/of6r" "$W/of6c" 0.1 6; ofrun "$W/of6r" restart
echo "     reported: OpenFOAM restarted vs continuous at 0.2, timePrecision 6: $(rel "$W/of6c" "$W/of6r" 0.2)"
echo "     reported: brae continuous vs OpenFOAM continuous at 0.2: $(rel "$W/of6c" "$W/br6" 0.2)"
[ "$fail" = 0 ] && echo "PASS" || echo "FAIL"
exit "$fail"
