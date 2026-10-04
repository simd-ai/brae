# THE PRESSURE RULE'S GATES (CLAUDE.md, user decisions 2026-10-03). p_rgh and pcorr run brae's AMG-preconditioned
# PCG by default whatever the case names -- GAMG, PCG with DIC, PCG with a GAMG preconditioner -- announced; every
# other test runs the case's own solver as ported (BRAE_PRESSURE_CASE_SOLVER=1, CMakeLists.txt and lib.sh) so it
# stays exact. These files unset it and hold the default to OpenFOAM's own solver on a W row (two steps, every
# solve pinned at 1e-13, relTol 0), at a bound one decade above the measured worst: two Krylov methods stop at
# different 1e-13 residuals, and that difference is what is measured.
#
# amgpcg_gate <key> <bound>: three checks -- the fields within <bound> of OpenFOAM, the substitution announced,
# and the CONTROL: BRAE_CONTROL_AMG_PCG_ONE_ITERATION=1 (one AMG-PCG iteration a solve) over the bound.
amgpcg_gate()
{
    local key="$1" bound="$2" v e
    unset BRAE_PRESSURE_CASE_SOLVER
    [ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
    wcase $key of > "$W/amg_stage.txt" 2>&1
    local o="$W/w_of_$key"
    [ -d "$o" ] || { say "$key did not stage" FAIL; return; }
    for v in amg control; do
        e="$W/amg_$v"
        mkdir -p "$e"
        cp -r "$o/0" "$o/constant" "$o/system" "$e/"
        case $v in
            amg)     runbrae "$e" device ;;
            control) runbrae "$e" device BRAE_CONTROL_AMG_PCG_ONE_ITERATION=1 ;;
        esac
        python3 "$CMP" "$o" "$e" $(timedirs "$o") > "$W/cmp_amg_$v.txt" 2>&1
    done
    judge "$key AMG-PCG" "$W/cmp_amg_amg.txt" "$bound" "$o/log.interFoam" \
        && say "[device] $key with the AMG-PCG: every file within $bound of OpenFOAM's own solver" ok \
        || say "[device] $key with the AMG-PCG: every file within $bound of OpenFOAM's own solver" FAIL
    local what="[device] $key: the log announces the AMG-PCG in the case's solver's place"
    grep -q "brae runs its AMG-preconditioned PCG" "$W/amg_amg/log.brae" && say "$what" ok || say "$what" FAIL
    # the control is held on the FIELDS: the continuity error goes over on its own, and must not be what trips it
    python3 - "$W/cmp_amg_control.txt" "$bound" > "$W/amg_control.txt" <<'PY'
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
f = max(((k, v['rel']) for k, v in r['files'].items() if 'cumulativeContErr' not in k), key=lambda kv: kv[1])
print('%s %.1e' % f)
sys.exit(0 if f[1] > float(sys.argv[2]) else 1)
PY
    local rc=$?
    what="CONTROL  one AMG-PCG iteration a solve puts $key's fields over $bound ($(cat "$W/amg_control.txt"))"
    [ $rc -eq 0 ] && say "$what" ok || say "$what" FAIL
}

# amgpcg_pcorr_gate <key> <bound>: pcorr's own share of the rule -- the log announces the AMG-PCG for pcorr, and
# the CONTROL BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION=1 (pcorr's solve alone stopped after one iteration) puts
# the fields over <bound>, so the gate sees pcorr. The fields' own bound is amgpcg_gate's, in its own file.
amgpcg_pcorr_gate()
{
    local key="$1" bound="$2" v e
    unset BRAE_PRESSURE_CASE_SOLVER
    [ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
    wcase $key of > "$W/amg_stage.txt" 2>&1
    local o="$W/w_of_$key"
    [ -d "$o" ] || { say "$key did not stage" FAIL; return; }
    for v in amg control; do
        e="$W/pcorr_$v"
        mkdir -p "$e"
        cp -r "$o/0" "$o/constant" "$o/system" "$e/"
        case $v in
            amg)     runbrae "$e" device ;;
            control) runbrae "$e" device BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION=1 ;;
        esac
    done
    local what="[device] $key: the log announces the AMG-PCG for pcorr"
    grep -q "pcorr: system/fvSolution asks for GAMG; brae runs its AMG-preconditioned PCG" "$W/pcorr_amg/log.brae" \
        && say "$what" ok || say "$what" FAIL
    python3 "$CMP" "$o" "$W/pcorr_control" $(timedirs "$o") > "$W/cmp_pcorr_control.txt" 2>&1
    python3 - "$W/cmp_pcorr_control.txt" "$bound" > "$W/pcorr_control.txt" <<'PY'
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
f = max(((k, v['rel']) for k, v in r['files'].items() if 'cumulativeContErr' not in k), key=lambda kv: kv[1])
print('%s %.1e' % f)
sys.exit(0 if f[1] > float(sys.argv[2]) else 1)
PY
    local rc=$?
    what="CONTROL  pcorr's AMG-PCG stopped after one iteration puts $key's fields over $bound"
    what="$what ($(cat "$W/pcorr_control.txt"))"
    [ $rc -eq 0 ] && say "$what" ok || say "$what" FAIL
}

# amgpcg_pcorr_sa_gate <key> <bound>: pcorr on its own smoothed-aggregation hierarchy (DevicePcorrSolver's
# fixedTopology), on a case whose mesh moves and keeps its topology. Three arms -- the default, the hierarchy
# p_rgh uses (BRAE_PCORR_AMG=plain), and pcorr's solve stopped after one iteration -- and three checks: the
# default announces the smoothed hierarchy and solves pcorr in FEWER iterations than the plain one, its fields
# are within <bound> of OpenFOAM's, and the control goes over <bound>.
amgpcg_pcorr_sa_gate()
{
    local key="$1" bound="$2" v e
    unset BRAE_PRESSURE_CASE_SOLVER
    [ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
    wcase $key of > "$W/amg_stage.txt" 2>&1
    local o="$W/w_of_$key"
    [ -d "$o" ] || { say "$key did not stage" FAIL; return; }
    for v in sa plain control; do
        e="$W/pcsa_$v"
        mkdir -p "$e"
        cp -r "$o/0" "$o/constant" "$o/system" "$e/"
        case $v in
            sa)      runbrae "$e" device BRAE_PRINT_TURB_SOLVES=1 ;;
            plain)   runbrae "$e" device BRAE_PRINT_TURB_SOLVES=1 BRAE_PCORR_AMG=plain ;;
            control) runbrae "$e" device BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION=1 ;;
        esac
        python3 "$CMP" "$o" "$e" $(timedirs "$o") > "$W/cmp_pcsa_$v.txt" 2>&1
    done
    local mark="pcorr: its AMG hierarchy is a smoothed-aggregation one of its own"
    local nsa=$(grep -a "Solving for pcorr" "$W/pcsa_sa/log.brae" | awk '{s += $NF} END {print s + 0}')
    local npl=$(grep -a "Solving for pcorr" "$W/pcsa_plain/log.brae" | awk '{s += $NF} END {print s + 0}')
    local what="[device] $key: pcorr takes its smoothed hierarchy and says so, in $nsa iterations against $npl"
    grep -q "$mark" "$W/pcsa_sa/log.brae" && ! grep -q "$mark" "$W/pcsa_plain/log.brae" \
        && [ "$nsa" -gt 0 ] && [ "$nsa" -lt "$npl" ] && say "$what" ok || say "$what" FAIL
    judge "$key pcorr on the smoothed hierarchy" "$W/cmp_pcsa_sa.txt" "$bound" "$o/log.interFoam" \
        && say "[device] $key with pcorr on the smoothed hierarchy: every file within $bound of OpenFOAM" ok \
        || say "[device] $key with pcorr on the smoothed hierarchy: every file within $bound of OpenFOAM" FAIL
    python3 - "$W/cmp_pcsa_control.txt" "$bound" > "$W/pcsa_control.txt" <<'PY'
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
f = max(((k, v['rel']) for k, v in r['files'].items() if 'cumulativeContErr' not in k), key=lambda kv: kv[1])
print('%s %.1e' % f)
sys.exit(0 if f[1] > float(sys.argv[2]) else 1)
PY
    local rc=$?
    what="CONTROL  pcorr's solve stopped after one iteration puts $key's fields over $bound"
    what="$what ($(cat "$W/pcsa_control.txt"))"
    [ $rc -eq 0 ] && say "$what" ok || say "$what" FAIL
}
