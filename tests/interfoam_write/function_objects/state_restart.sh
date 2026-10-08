#!/usr/bin/env bash
# The gate: the function objects on a RESTART. The state dictionary is READ at the start time if it is there
# (functionObjectList.C:88-108, READ_IF_PRESENT) and written back at every write with whatever it held, so the
# results of an object the continuation no longer runs stay in every later time directory; and a probes
# object's files go under the continuation's own START TIME (probes.C:77-85), beside the first run's.
# The row: laminar/damBreak as shipped with two probes objects, run by OpenFOAM to its first write; then both
# codes continue from OPENFOAM'S directory to t = STOP with the object `gone` disabled. STAGED, in both codes:
# every solve pinned (tolerance 1e-13, relTol 0). A continuation opens with a flux correction solved to its
# tolerance, and at the tutorial's own the two codes' continuations are that far apart, not a probe's width.
# Three checks. (1) THE HOST LOOP's continuation: the kept object's file under the start time has OpenFOAM's
# head, rows, time column and values within HOST_BOUND, and every later dictionary is OpenFOAM's text with
# its numbers within the bound -- `gone`'s results still in it. (2) THE GPU LOOP, the same. (3) CONTROL,
# BRAE_CONTROL_FUNCTION_OBJECT_STATE=fresh: the start's dictionary left unread loses `gone`'s results.
# MEASURED 2026-10-08: from 0.05, 94 rows in kept's two files and the dictionaries at 0.1 and 0.15, all within
# 4.0e-13 on the host and 3.2e-13 on the GPU. FOUND by this gate: the carried keyword `average(p_rgh)` came
# back split at its parenthesis through brae's dictionary reader -- the state dictionary has its own reader
# now. (At the tutorial's own tolerances the two continuations' FIELDS are 4.9e-09 apart at 0.1, U; pinned,
# 1.5e-14.) Control: 25 lines of other text at 0.15.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
. "$(dirname "$0")/_common.sh"
STOP=0.16
HOST_BOUND=4e-12
DEVICE_BOUND=4e-12
name="the function objects on a restart"
cat > "$W/sr_functions.txt" <<'FO'
functions
{
    gone
    {
        type            probes;
        libs            (sampling);
        writeControl    writeTime;
        fields          (p U);
        probeLocations  ((0.1 0.1 0.0073));
    }
    kept
    {
        type            probes;
        libs            (sampling);
        fields          (p_rgh alpha.water);
        probeLocations  ((0.292 0.05 0.0073) (0.5 0.02 0.0073));
    }
}
FO
first="$W/sr_first"
FO_PIN=1 fo_stage "$first" "$LAM" 0.06 adjustable 0.05 "$W/sr_functions.txt" || finish "$name"
t0=$(timedirs "$first" | awk '{print $1}')
o="$W/sr_of"
rm -rf "${o:?}"
mkdir -p "$o"
cp -r "$first/0" "$first/$t0" "$first/constant" "$first/system" "$o/"
sed -i -E "s/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         $STOP;/" "$o/system/controlDict"
python3 - "$o/system/controlDict" <<'PY' || { say "PREMISE  the continuation was staged" FAIL; finish "$name"; }
import sys
p = sys.argv[1]
s = open(p).read()
a = '    gone\n    {\n'
assert s.count(a) == 1 and 'latestTime' in s
open(p, 'w').write(s.replace(a, a + '        enabled         false;\n', 1))
PY
runof "$o"
later=$(timedirs "$o" | tr ' ' '\n' | grep -vx "$t0" | tr '\n' ' ')
last=$(echo $later | awk '{print $NF}')
[ -n "$t0" ] && [ "$(echo $later | wc -w)" -ge 2 ] && [ -f "$o/postProcessing/kept/$t0/p_rgh" ] \
    && [ ! -e "$o/postProcessing/gone" ] \
    && grep -q "^    gone$" "$o/$last/uniform/functionObjects/functionObjectProperties" \
    || { say "PREMISE  OpenFOAM from $t0: writes at [$later], kept's files under $t0, gone's results at $last" FAIL
         finish "$name"; }
arm()   # arm <name> <host|device> [env...]: brae continuing from OpenFOAM's directory
{
    local e="$W/sr_$1" loop="$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$first/0" "$first/$t0" "$first/constant" "$o/system" "$e/"
    runbrae "$e" "$loop" "$@"
}
check()   # check <run dir> <label> <bound>
{
    local e="$1" ok=1 worst=0 rows=0 f t
    for f in kept/$t0/p_rgh kept/$t0/alpha.water; do
        read -r ro rb hd td gap <<< "$(fo_file "$o/postProcessing/$f" "$e/postProcessing/$f")"
        [ "$rb" = - ] && { rb=0; gap=1; }
        [ "$ro" = "$rb" ] && [ "$hd" = 0 ] && [ "$td" = 0 ] && within "$gap" "$3" || ok=0
        rows=$((rows + rb))
        worst=$(python3 -c "print('%.1e' % max(float('$worst'), float('$gap')))")
    done
    for t in $later; do
        read -r text count gap <<< "$(fo_state "$o" "$e" "$t")"
        [ "$text" = - ] && { text=1; gap=1; }
        [ "$text" = 0 ] && within "$gap" "$3" || ok=0
        worst=$(python3 -c "print('%.1e' % max(float('$worst'), float('$gap')))")
    done
    [ ! -e "$e/postProcessing/gone" ] || ok=0
    what="[$2] from $t0: kept's files under $t0 hold OpenFOAM's $rows rows, the dictionaries at"
    what="$what $(echo $later | tr ' ' ',') its text and \`gone\`'s results, all within $worst (bound $3)"
    [ $ok = 1 ] && say "$what" ok || say "$what" FAIL
}
arm host host BRAE_X=1
check "$W/sr_host" host $HOST_BOUND
arm device device BRAE_X=1
check "$W/sr_device" device $DEVICE_BOUND
arm fresh host BRAE_CONTROL_FUNCTION_OBJECT_STATE=fresh
read -r text count gap <<< "$(fo_state "$o" "$W/sr_fresh" "$last")"
what="CONTROL  the start's dictionary left unread: $text lines of other text at $last, \`gone\` in it"
what="$what $(grep -c "^    gone$" "$W/sr_fresh/$last/uniform/functionObjects/functionObjectProperties") times"
[ "${text/-/0}" -ge 1 ] && ! grep -q "^    gone$" "$W/sr_fresh/$last/uniform/functionObjects/functionObjectProperties" \
    && say "$what" ok || say "$what" FAIL
finish "a continuation reads the state dictionary and files its probes under its own start, on both loops"
