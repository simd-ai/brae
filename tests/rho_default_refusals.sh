#!/usr/bin/env bash
# rhoSimpleFoam's DEFAULT driver (`brae -case X`, no BRAE_RHOSIMPLEFOAM_MIRROR) refuses, by name, every
# configuration it used to run differently from OpenFOAM -- and the nearest supported configuration still
# sets up. Each arm stages a copy of a repository fixture (rhoBox laminar, rhoBoxF RAS, rhoBoxP closed p,
# rhoSuth sutherland: all four carry their mesh, so no OpenFOAM is needed), applies ONE mutation, and runs
# the default driver under `brae check --setup` (BRAE_CHECK_SETUP=1), which builds the whole case and stops
# before the first iteration -- the same place the refusals have to fire in a real run.
#
#   refused <key words>   the run fails and its message names what was found and why
#   runs                  the run reaches "set the case up completely"
#
# What each group guards (gpuRhoSimpleFoam.cu unless named):
#   frozen_*     coded*/fixedMean/fanPressure, which this driver builds from `value` and never updates
#   expr_*       uniformFixedValue `expression`, never evaluated here (squareBendLiq's wall T)
#   rotor, win_*, maxwell, alphat_*
#                rotorDisk parsed and never applied; fvOptions timeStart/duration ignored; Maxwell accepted
#                then thrown at iteration 1; alphat types computed as rho*nut/Prt; a RAS case with no alphat
#   sw_*         SIMPLE/transonic and /consistent read as OpenFOAM's Switch (both drivers)
#   emax_*       solvers/<e|h> maxIter/minIter, which the energy solve cannot take yet
#   pref_*       the pressure reference as OpenFOAM's setRefCell resolves it
#   time_*       deltaT and endTime as OpenFOAM's Time reads them
#   thermo_*     the mandatory thermophysicalProperties coefficients (both drivers: thermo_parse.cuh)
# The SWITCH EQUIVALENCE arms run three iterations: `transonic true` must write the same bytes as
# `transonic yes` (and `no` must not), likewise consistent; on the default driver and the mirror's host arm.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in rhoBox rhoBoxF rhoBoxP rhoSuth; do
    [ -f "$ROOT/validation/$f/constant/polyMesh/owner" ] || { echo "SKIP: fixture $f has no mesh"; exit 77; }
done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage <fixture> <dir>: a fresh copy, 0.orig as 0/
stage() {
    rm -rf "$2"; cp -r "$ROOT/validation/$1" "$2"; rm -rf "$2"/[1-9]* "$2"/0 "$2"/log.*
    cp -r "$2/0.orig" "$2/0"
}
# edit <dir> <python statements over helpers>: patch(file, patchName, body) replaces a one-line patch entry,
# sub(file, regex, repl, count) a regex substitution that must apply, write(file, text) a new file.
edit() {
    python3 - "$1" "$2" <<'PY'
import re, sys
d, code = sys.argv[1], sys.argv[2]
def rd(f): return open(d + '/' + f).read()
def wr(f, s): open(d + '/' + f, 'w').write(s)
def sub(f, pat, rep, n=1):
    s, k = re.subn(pat, rep, rd(f))
    assert k >= 1, 'mutation did not apply: %s in %s' % (pat, f)
    wr(f, s)
def patch(f, name, body):
    sub(f, r'(\b%s\s*)\{[^{}]*\}' % re.escape(name), lambda m: m.group(1) + '{ ' + body + ' }')
def write(f, text): wr(f, text)
exec(code)
PY
}
# arm <name> <refused|runs> <grep -E key 1> <grep -E key 2> <fixture> <python edit> [env...]
arm() {
    local name="$1" expect="$2" k1="$3" k2="$4" fix="$5" code="$6"; shift 6
    local d="$W/$name"
    stage "$fix" "$d"
    [ -n "$code" ] && { edit "$d" "$code" || { say "$name: staging" FAIL; return; }; }
    local out rc
    out=$(cd "$d" && env BRAE_CHECK_SETUP=1 "$@" "$BRAE" -case "$d" 2>&1); rc=$?
    if [ "$expect" = runs ]; then
        if [ $rc -eq 0 ] && echo "$out" | grep -q "set the case up completely"; then say "$name: runs" ok
        else say "$name: expected to set up, got rc $rc" FAIL; echo "$out" | grep -v NOTICE | tail -3 | sed 's/^/        /'; fi
    else
        if [ $rc -ne 0 ] && echo "$out" | grep -qE -- "$k1" && echo "$out" | grep -qE -- "$k2"; then
            say "$name: refused, names '$k1'" ok
        else
            say "$name: expected a refusal naming '$k1' / '$k2', got rc $rc" FAIL
            echo "$out" | grep -v NOTICE | tail -3 | sed 's/^/        /'
        fi
    fi
    rm -rf "$d"
}

echo "rho_default_refusals: the default rhoSimpleFoam driver, BRAE_CHECK_SETUP=1"

# ---- positive controls: the fixtures as shipped -------------------------------------------------------
arm ctl_rhoBox  runs "" "" rhoBox  ""
arm ctl_rhoBoxF runs "" "" rhoBoxF ""
# rhoBoxP's outlet p is fixedFluxPressure, which this driver refuses (it never runs constrainPressure);
# zeroGradient there keeps what the pref_* arms need: no p patch fixes a value, so p needs a reference.
PZG="patch('0/p', 'outlet', 'type zeroGradient;'); patch('0/U', 'outlet', 'type zeroGradient;'); "
arm ctl_rhoBoxP runs "" "" rhoBoxP "$PZG"
arm ctl_rhoSuth runs "" "" rhoSuth ""

# ---- 1. per-step boundary conditions this driver never updates ----------------------------------------
arm frozen_codedFixedValue_U refused "codedFixedValue" "never updates" rhoBoxF \
    "patch('0/U', 'inlet', 'type codedFixedValue; value uniform (5 0 0); name inl; code #{ #};')"
arm frozen_codedMixed_T      refused "codedMixed" "never updates" rhoBoxF \
    "patch('0/T', 'hotWall', 'type codedMixed; refValue uniform 700; refGradient uniform 0; valueFraction uniform 1; value uniform 700; name hw; code #{ #};')"
arm frozen_fixedMean_p       refused "fixedMean" "never updates" rhoBoxF \
    "patch('0/p', 'inlet', 'type fixedMean; meanValue constant 100000; value uniform 100000;')"
arm frozen_fanPressure_p     refused "fanPressure" "never updates" rhoBoxF \
    "patch('0/p', 'inlet', 'type fanPressure; value uniform 100000;')"
arm frozen_fixedMean_k       refused "fixedMean" "never updates" rhoBoxF \
    "patch('0/k', 'inlet', 'type fixedMean; meanValue constant 0.375; value uniform 0.375;')"
arm frozen_coded_nut         refused "codedFixedValue" "never updates" rhoBoxF \
    "patch('0/nut', 'inlet', 'type codedFixedValue; value uniform 0.001; name n; code #{ #};')"

# ---- 2. the `expression` PatchFunction1, frozen at `value` here ----------------------------------------
EXPR='type uniformFixedValue; uniformValue { type expression; expression #{ 700 #}; } value uniform 700;'
arm expr_T refused "expression" "BRAE_RHOSIMPLEFOAM_MIRROR=cuda" rhoBoxF "patch('0/T', 'hotWall', '$EXPR')"
arm expr_k refused "expression" "No brae rhoSimpleFoam arm" rhoBoxF \
    "patch('0/k', 'inlet', 'type uniformFixedValue; uniformValue { type expression; expression #{ 0.375 #}; } value uniform 0.375;')"
# The message's pointer must be true: the mirror's CUDA arm takes the same case (one iteration).
d="$W/expr_T_mirror"; stage rhoBoxF "$d"
edit "$d" "patch('0/T', 'hotWall', '$EXPR'); sub('system/controlDict', r'\bendTime\s+[^;]+;', 'endTime 1;'); sub('system/controlDict', r'\bwriteInterval\s+[^;]+;', 'writeInterval 1;')"
if (cd "$d" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > log 2>&1) && [ -d "$d/1" ]; then
    say "expr_T_mirror: BRAE_RHOSIMPLEFOAM_MIRROR=cuda runs the same case" ok
else
    say "expr_T_mirror: the mirror the message points to does not run the case" FAIL
    grep -v NOTICE "$d/log" | tail -3 | sed 's/^/        /'
fi
rm -rf "$d"

# ---- 5. sources, models and alphat the driver does not compute faithfully ------------------------------
FVO_HEAD='FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\n'
arm rotor refused "rotorDisk" "never applies" rhoBox \
    "write('system/fvOptions', '''$FVO_HEAD disk { type rotorDisk; selectionMode all; nBlades 3; tipEffect 0.96; inletFlowType local; inletVelocity (5 0 0); geometryMode specified; origin (1 0.1 0.025); axis (1 0 0); refDirection (0 1 0); rpm 100; trimModel fixedTrim; rhoRef 1; rhoInf 1; fixedTrimCoeffs { theta0 0; theta1c 0; theta1s 0; } flapCoeffs { beta0 0; beta1c 0; beta2s 0; } blade { data ((p1 (0.01 -6 0.02)) (p1 (0.1 -6 0.02))); } profiles { p1 { type lookup; data ((-90 0.21 1.45) (90 0.21 1.45)); } } }''')"
LIMT='limitT { type limitTemperature; selectionMode all; min 200; max 1000;'
arm win_partial refused "timeStart" "does not cover" rhoBox \
    "write('system/fvOptions', '''$FVO_HEAD $LIMT timeStart 5; duration 10; }''')"
arm win_covering runs "" "" rhoBox \
    "write('system/fvOptions', '''$FVO_HEAD $LIMT timeStart 0; duration 1e6; }''')"
arm win_negative runs "" "" rhoBox \
    "write('system/fvOptions', '''$FVO_HEAD $LIMT timeStart -1; duration 2; }''')"
arm maxwell refused "Maxwell" "viscoelastic" rhoBox \
    "sub('constant/turbulenceProperties', r'simulationType\s+laminar;', 'simulationType laminar; laminar { model Maxwell; MaxwellCoeffs { nuM 1e-3; lambda 0.1; } }')"
arm alphat_jayatilleke refused "alphatJayatillekeWallFunction" "P-function" rhoBoxF \
    "patch('0/alphat', 'hotWall', 'type compressible::alphatJayatillekeWallFunction; Prt 0.85; value uniform 0;')"
arm alphat_fixedValue refused "'fixedValue'" "rho\*nut/Prt" rhoBoxF \
    "patch('0/alphat', 'hotWall', 'type fixedValue; value uniform 0;')"
arm alphat_bare_name refused "compressible::alphatWallFunction" "no .alphatWallFunction." rhoBoxF \
    "patch('0/alphat', 'hotWall', 'type alphatWallFunction; value uniform 0;')"
arm alphat_wallfunction runs "" "" rhoBoxF \
    "patch('0/alphat', 'hotWall', 'type compressible::alphatWallFunction; Prt 0.9; value uniform 0;')"
d="$W/alphat_missing"; stage rhoBoxF "$d"; rm -f "$d/0/alphat"
out=$(cd "$d" && BRAE_CHECK_SETUP=1 "$BRAE" -case "$d" 2>&1); rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q "alphat does not exist" && echo "$out" | grep -q "MUST_READ" \
    && say "alphat_missing: refused, names the missing 0/alphat" ok \
    || { say "alphat_missing: expected a refusal, got rc $rc" FAIL; echo "$out" | grep -v NOTICE | tail -3; }
rm -rf "$d"

# ---- 6. Switch words ------------------------------------------------------------------------------------
arm sw_transonic_bad  refused "transonic is 'maybe'" "not a Switch" rhoBox \
    "sub('system/fvSolution', r'transonic\s+no;', 'transonic maybe;')"
arm sw_consistent_bad refused "consistent is 'yess'" "not a Switch" rhoBox \
    "sub('system/fvSolution', r'transonic\s+no;', 'transonic no; consistent yess;')"
arm sw_transonic_bad_mirror refused "transonic is 'maybe'" "not a Switch" rhoBox \
    "sub('system/fvSolution', r'transonic\s+no;', 'transonic maybe;')" BRAE_RHOSIMPLEFOAM_MIRROR=1
arm sw_consistent_bad_mirror refused "consistent is 'yess'" "not a Switch" rhoBox \
    "sub('system/fvSolution', r'transonic\s+no;', 'transonic no; consistent yess;')" BRAE_RHOSIMPLEFOAM_MIRROR=1
# equivalence: three iterations at writePrecision 17, the written fields compared byte for byte
switch_run() {  # switch_run <dir> <transonic word> <consistent word> [env...]
    local d="$1" tr="$2" co="$3"; shift 3
    stage rhoBox "$d"
    edit "$d" "sub('system/controlDict', r'\bendTime\s+[^;]+;', 'endTime 3;'); sub('system/controlDict', r'\bwriteInterval\s+[^;]+;', 'writeInterval 3;'); sub('system/controlDict', r'\bwritePrecision\s+[^;]+;', 'writePrecision 17;'); sub('system/fvSolution', r'transonic\s+no;', 'transonic $tr; consistent $co;')"
    (cd "$d" && env "$@" "$BRAE" -case "$d" > log 2>&1)
}
same() { for f in U p T rho phi; do cmp -s "$1/3/$f" "$2/3/$f" || return 1; done; return 0; }
for arm_ in "default:" "mirror:BRAE_RHOSIMPLEFOAM_MIRROR=1"; do
    lbl=${arm_%%:*}; ev=${arm_#*:}; [ -z "$ev" ] && ev=BRAE_NOTHING=1
    switch_run "$W/s_yes"  yes no  "$ev"; switch_run "$W/s_true" true no "$ev"; switch_run "$W/s_no" no no "$ev"
    if [ -d "$W/s_yes/3" ] && same "$W/s_yes" "$W/s_true" && ! same "$W/s_yes" "$W/s_no"; then
        say "sw_equiv_transonic ($lbl): 'true' writes the bytes 'yes' does, 'no' does not" ok
    else say "sw_equiv_transonic ($lbl): 'true' is not run as 'yes'" FAIL; fi
    switch_run "$W/c_yes" no yes "$ev"; switch_run "$W/c_on" no on "$ev"; switch_run "$W/c_y" no y "$ev"
    if [ -d "$W/c_yes/3" ] && same "$W/c_yes" "$W/c_on" && same "$W/c_yes" "$W/c_y" && ! same "$W/c_yes" "$W/s_no"; then
        say "sw_equiv_consistent ($lbl): 'on' and 'y' write the bytes 'yes' does, 'no' does not" ok
    else say "sw_equiv_consistent ($lbl): 'on'/'y' are not run as 'yes'" FAIL; fi
    rm -rf "$W"/s_* "$W"/c_*
done

# ---- 8. the energy solve's maxIter / minIter ------------------------------------------------------------
arm emax_maxIter refused "maxIter 50" "lduMatrix" rhoBox \
    "sub('system/fvSolution', r'(\"\(U\|h\|e\)\"\s*\{)', r'\1 maxIter 50;')"
arm emax_minIter refused "minIter 2" "lduMatrix" rhoBox \
    "sub('system/fvSolution', r'(\"\(U\|h\|e\)\"\s*\{)', r'\1 minIter 2;')"
arm emax_default runs "" "" rhoBox \
    "sub('system/fvSolution', r'(\"\(U\|h\|e\)\"\s*\{)', r'\1 maxIter 1000; minIter 0;')"

# ---- 9. the pressure reference (rhoBoxP: no p patch fixes a value) -------------------------------------
arm pref_none     refused "neither pRefCell nor pRefPoint" "findRefCell" rhoBoxP \
    "${PZG}sub('system/fvSolution', r'pRefCell\s+\d+;', '')"
arm pref_novalue  refused "no pRefValue" "findRefCell" rhoBoxP \
    "${PZG}sub('system/fvSolution', r'pRefValue\s+[^;]+;', '')"
arm pref_range    refused "pRefCell 99999 is not a cell" "Illegal master cellID" rhoBoxP \
    "${PZG}sub('system/fvSolution', r'pRefCell\s+\d+;', 'pRefCell 99999;')"
arm pref_outside  refused "lies in no cell" "CELL_TETS" rhoBoxP \
    "${PZG}sub('system/fvSolution', r'pRefCell\s+\d+;', 'pRefPoint (100 100 100);')"
arm pref_point    runs "" "" rhoBoxP \
    "${PZG}sub('system/fvSolution', r'pRefCell\s+\d+;', 'pRefPoint (1.51 0.135 0.025);')"

# ---- 10. deltaT and endTime ----------------------------------------------------------------------------
arm time_no_deltaT  refused "no deltaT" "TimeIO.C:274" rhoBox "sub('system/controlDict', r'\bdeltaT\s+[^;]+;', '')"
arm time_no_endTime refused "no endTime" "TimeIO.C" rhoBox "sub('system/controlDict', r'\bendTime\s+[^;]+;', '')"

# ---- 11. mandatory thermophysicalProperties coefficients (const on rhoBox, sutherland on rhoSuth) --------
arm thermo_no_molWeight refused "no .molWeight." "specie.C:46" rhoBox \
    "sub('constant/thermophysicalProperties', r'molWeight\s+[^;]+;', '')"
arm thermo_no_Cp refused "no .Cp." "hConstThermo.C:38" rhoBox \
    "sub('constant/thermophysicalProperties', r'\bCp\s+[^;]+;', '')"
arm thermo_no_Hf refused "no .Hf." "hConstThermo.C:39" rhoBox \
    "sub('constant/thermophysicalProperties', r'\bHf\s+[^;]+;', '')"
arm thermo_no_mu refused "no .mu." "constTransport.C:37" rhoBox \
    "sub('constant/thermophysicalProperties', r'\bmu\s+[^;]+;', '')"
arm thermo_no_Pr refused "no .Pr." "constTransport.C:38" rhoBox \
    "sub('constant/thermophysicalProperties', r'\bPr\s+[^;]+;', '')"
arm thermo_no_As refused "no .As." "sutherlandTransport" rhoSuth \
    "sub('constant/thermophysicalProperties', r'\bAs\s+[^;]+;', '')"
arm thermo_no_Ts refused "no .Ts." "sutherlandTransport" rhoSuth \
    "sub('constant/thermophysicalProperties', r'\bTs\s+[^;]+;', '')"
arm thermo_no_Hf_mirror refused "no .Hf." "hConstThermo.C:39" rhoBox \
    "sub('constant/thermophysicalProperties', r'\bHf\s+[^;]+;', '')" BRAE_RHOSIMPLEFOAM_MIRROR=1

[ "$fail" -eq 0 ] && echo "rho_default_refusals: PASS" || echo "rho_default_refusals: FAIL"
exit "$fail"
