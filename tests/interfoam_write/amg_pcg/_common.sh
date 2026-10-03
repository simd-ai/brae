# THE PRESSURE RULE'S GATES (CLAUDE.md, user decision 2026-10-03). A case's GAMG on p_rgh runs brae's
# AMG-preconditioned PCG by default, announced; every other test runs the GAMG port (BRAE_PRESSURE_GAMG_PORT=1,
# CMakeLists.txt and lib.sh) so it stays exact. These files unset it and hold the default to OpenFOAM's own GAMG
# on a W row (two steps, every solve pinned at 1e-13, relTol 0), at a bound one decade above the measured worst:
# two Krylov methods stop at different 1e-13 residuals, and that difference is what is measured.
#
# amgpcg_gate <key> <bound>: three checks -- the fields within <bound> of OpenFOAM, the substitution announced,
# and the CONTROL: BRAE_CONTROL_AMG_PCG_ONE_ITERATION=1 (one AMG-PCG iteration a solve) over the bound.
amgpcg_gate()
{
    local key="$1" bound="$2" v e
    unset BRAE_PRESSURE_GAMG_PORT
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
        && say "[device] $key with the AMG-PCG: every file within $bound of OpenFOAM's GAMG" ok \
        || say "[device] $key with the AMG-PCG: every file within $bound of OpenFOAM's GAMG" FAIL
    grep -q "brae runs its AMG-preconditioned PCG" "$W/amg_amg/log.brae" \
        && say "[device] $key: the log announces the AMG-PCG in GAMG's place" ok \
        || say "[device] $key: the log announces the AMG-PCG in GAMG's place" FAIL
    # the control is held on the FIELDS: the continuity error goes over on its own, and must not be what trips it
    python3 - "$W/cmp_amg_control.txt" "$bound" > "$W/amg_control.txt" <<'PY'
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
f = max(((k, v['rel']) for k, v in r['files'].items() if 'cumulativeContErr' not in k), key=lambda kv: kv[1])
print('%s %.1e' % f)
sys.exit(0 if f[1] > float(sys.argv[2]) else 1)
PY
    [ $? -eq 0 ] \
        && say "CONTROL  one AMG-PCG iteration a solve puts $key's fields over $bound ($(cat "$W/amg_control.txt"))" ok \
        || say "CONTROL  one AMG-PCG iteration a solve puts $key's fields over $bound ($(cat "$W/amg_control.txt"))" FAIL
}
