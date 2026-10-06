# THE TURBULENCE RULE'S GATES (CLAUDE.md, the user's decision 2026-10-06). In interFoam's device loop a
# Gauss-Seidel smoothSolver on k, epsilon or omega that is asked to converge (relTol 0) is swept in COLOUR order
# on the GPU by default, announced: the same smoother under the same stopping rule, another cell order, so
# another iterate after n sweeps (device_inter_turbulence.cu). Every other test runs the case's own order
# (BRAE_TURBULENCE_CASE_SOLVER=1, CMakeLists.txt and lib.sh) so it stays exact. These files unset it and hold
# the default to real OpenFOAM on a W row (two steps, every solve pinned at 1e-13, relTol 0), at a bound one
# decade above the measured worst: two sweep orders stop at different 1e-13 residuals, and that difference is
# what is measured. The pressure stays the case's own solver here (lib.sh), so the bound is the rule's alone.
#
# turbcolour_gate <key> <bound> <field> [field ...]: three checks -- every written file within <bound> of
# OpenFOAM; the rule announced for each <field> and its solves reported; and the CONTROL,
# BRAE_CONTROL_TURBULENCE_COLOUR_ONE_SWEEP=1 (a solve the rule takes stops after one sweep), over the bound.
turbcolour_gate()
{
    local key="$1" bound="$2" v e f
    shift 2
    unset BRAE_TURBULENCE_CASE_SOLVER
    [ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
    wcase $key of > "$W/tc_stage.txt" 2>&1
    local o="$W/w_of_$key"
    [ -d "$o" ] || { say "$key did not stage" FAIL; return; }
    for v in colour control; do
        e="$W/tc_$v"
        mkdir -p "$e"
        cp -r "$o/0" "$o/constant" "$o/system" "$e/"
        case $v in
            colour)  runbrae "$e" device BRAE_PRINT_TURB_SOLVES=1 ;;
            control) runbrae "$e" device BRAE_CONTROL_TURBULENCE_COLOUR_ONE_SWEEP=1 ;;
        esac
        python3 "$CMP" "$o" "$e" $(timedirs "$o") > "$W/cmp_tc_$v.txt" 2>&1
    done
    judge "$key colour order" "$W/cmp_tc_colour.txt" "$bound" "$o/log.interFoam" \
        && say "[device] $key with k and its partner swept in colour order: every file within $bound of OpenFOAM" ok \
        || say "[device] $key with k and its partner swept in colour order: every file within $bound of OpenFOAM" FAIL
    local told=0 solved=0
    for f in "$@"; do
        grep -aq "^ *$f: system/fvSolution asks for smoothSolver .* brae sweeps it in colour order" \
            "$W/tc_colour/log.brae" && told=$((told + 1))
        [ "$(grep -ac "Solving for $f," "$W/tc_colour/log.brae")" -ge 2 ] && solved=$((solved + 1))
    done
    local what="[device] $key: the log announces the colour order for $# fields ($told) and reports their solves"
    [ "$told" = "$#" ] && [ "$solved" = "$#" ] && say "$what" ok || say "$what" FAIL
    # the control is held on the FIELDS, as the pressure rule's is
    python3 - "$W/cmp_tc_control.txt" "$bound" > "$W/tc_control.txt" <<'PY'
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
f = max(((k, v['rel']) for k, v in r['files'].items() if 'cumulativeContErr' not in k), key=lambda kv: kv[1])
print('%s %.1e' % f)
sys.exit(0 if f[1] > float(sys.argv[2]) else 1)
PY
    local rc=$?
    what="CONTROL  one sweep a solve puts $key's fields over $bound ($(cat "$W/tc_control.txt"))"
    [ $rc -eq 0 ] && say "$what" ok || say "$what" FAIL
}
