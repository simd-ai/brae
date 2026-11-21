#!/usr/bin/env bash
# pimpleFoam's set-up refusals: configurations brae_pimpleFoam would compute differently from OpenFOAM v2412, refused by
# name before the first step, and the nearest supported configuration of each, which must still set up (run under
# BRAE_CHECK_SETUP=1, `brae check --setup`: the case is set up completely and no step is taken).
#
# Every case is a small box written by tests/pimple_box_case.py (no OpenFOAM needed). A refusal arm passes when the run
# exits non-zero and its log holds every key phrase of the message; an `accepts` arm when the set-up completes. What each
# arm guards, with the OpenFOAM source the refusal follows, is at its throw site in gpuPimpleFoam.cu / pimple_controls.cuh.
#
#   transport     transportModel other than Newtonian, no transportModel, no nu (viscosityModelNew.C:43, Newtonian.C:56)
#   turbulence    no turbulenceProperties (OpenFOAM.org's momentumTransport named), no simulationType; the dictionary audited
#   time          no deltaT, no endTime (TimeIO.C:272-275, :342-356)
#   ddt           a named ddt(X) entry, literal or regex, that resolves a solved field to another scheme than `default`
#   motion        a moving-mesh restart; solid-body entries missing or not constant; updateControl/updateInterval brae does
#                 not reproduce; moveMeshOuterCorrectors with nOuterCorrectors > 1; a wall-distance model on a moving mesh; a
#                 plain cyclic whose cells rotate or deform, or whose halves move apart; forces on a moving mesh
#   adaptive      forces/forceCoeffs/scalarTransport with adjustTimeStep; adjustTimeStep's start-up notice naming issue #53
#   les           SpalartAllmarasDES / kOmegaSSTDES; Smagorinsky/WALE with maxDeltaxyz; cubeRootVol deltaCoeff != 1; LES
#                 without delta; IDDES with another delta, another Cw, another hmax, exactDistance, or on a mesh whose cells
#                 are not boxes
#
# CONTROLS, asserted: the BRAE_CONTROL_* switches put the old behaviour back for two arms (named ddt entries unread, the
# IDDES hmax check off) and the arm must then be accepted. The negative control for the whole script is the parent
# commit's binary, on which every refusal arm fails (recorded in the work package's report).
set -u
BIN="${1:?brae_pimpleFoam binary}"
WORK="${2:?work dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOX="$ROOT/tests/pimple_box_case.py"
command -v python3 > /dev/null 2>&1 || { echo "SKIP: python3 not found"; exit 77; }
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }

fail=0
rm -rf "$WORK"
mkdir -p "$WORK"

box()   # box <dir> <generator args...>
{
    local d=$1
    shift
    rm -rf "$d"
    python3 "$BOX" "$d" "$@" || { echo "FAIL: could not write the box case $d"; exit 1; }
}

# set <file> <key> <value...>: a top-level `key value;` replaced, or appended; `-` as the value removes the entry
set_entry()
{
    python3 - "$@" <<'PY'
import re, sys
p, key, val = sys.argv[1], sys.argv[2], ' '.join(sys.argv[3:])
s = open(p).read()
pat = re.compile(r'(?m)^%s\s[^;{]*;\s*\n' % re.escape(key))
if val == '-':
    s = pat.sub('', s)
elif pat.search(s):
    s = pat.sub(lambda m: '%s %s;\n' % (key, val), s, count=1)
else:
    s += '%s %s;\n' % (key, val)
open(p, 'w').write(s)
PY
}

# append <file> <text>: raw text at the end of a dictionary
append() { printf '%s\n' "$2" >> "$1"; }

refuses()   # refuses <name> <case> <phrase>...
{
    local name=$1 dir=$2
    shift 2
    local log="$WORK/$name.log"
    env -u BRAE_CHECK_SETUP "$BIN" -case "$dir" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "FAIL: $name -- accepted (exit 0); it should be refused"
        fail=1
        return
    fi
    local p
    for p in "$@"; do
        if ! grep -qF -e "$p" "$log"; then
            echo "FAIL: $name -- refused, but the message lacks '$p'"
            grep -m3 -E "error|brae:" "$log" | cut -c1-240
            fail=1
            return
        fi
    done
    echo "ok:   $name (refused)"
}

accepts()   # accepts <name> <case> [env assignments...]
{
    local name=$1 dir=$2
    shift 2
    local log="$WORK/$name.log"
    env BRAE_CHECK_SETUP=1 "$@" "$BIN" -case "$dir" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ] || ! grep -qF "set the case up completely" "$log"; then
        echo "FAIL: $name -- not set up (exit $rc); it should run"
        grep -m3 -E "error|brae:" "$log" | cut -c1-240
        fail=1
        return
    fi
    echo "ok:   $name (sets up)"
}

dyn_rotating()   # dyn_rotating <case> [skip-entry]: solidBody rotatingMotion of cellZone rotor, complete unless an entry is skipped
{
    local d=$1 skip=${2:-}
    {
        echo 'FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }'
        echo 'dynamicFvMesh dynamicMotionSolverFvMesh;'
        echo 'motionSolver solidBody;'
        echo 'cellZone rotor;'
        echo 'solidBodyMotionFunction rotatingMotion;'
        [ "$skip" = origin ] || echo 'origin (0.5 0.5 0);'
        [ "$skip" = axis ]   || echo 'axis (0 0 1);'
        [ "$skip" = omega ]  || echo 'omega 1;'
    } > "$d/constant/dynamicMeshDict"
}

dyn_oscillating()   # dyn_oscillating <case>: solidBody oscillatingLinearMotion of cellZone rotor, coefficients where OF reads them
{
    cat > "$1/constant/dynamicMeshDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }
dynamicFvMesh dynamicMotionSolverFvMesh;
solver solidBody;
solidBodyCoeffs
{
    cellZone rotor;
    solidBodyMotionFunction oscillatingLinearMotion;
    oscillatingLinearMotionCoeffs
    {
        amplitude (0 0.01 0);
        omega 3;
    }
}
EOF
}

forces_fo()   # forces_fo <case> <type>
{
    append "$1/system/controlDict" "functions
{
    wallForces
    {
        type $2;
        libs (forces);
        patches (walls);
        rho rhoInf;
        rhoInf 1;
        CofR (0 0 0);
        liftDir (0 1 0);
        dragDir (1 0 0);
        pitchAxis (0 0 1);
        magUInf 1;
        lRef 1;
        Aref 1;
    }
}"
}

# ---- transport ---------------------------------------------------------------------------------------------------
box "$WORK/lam" --model laminar
accepts laminar_box "$WORK/lam"
cp -r "$WORK/lam" "$WORK/t_cross"
set_entry "$WORK/t_cross/constant/transportProperties" transportModel CrossPowerLaw
refuses transport_non_newtonian "$WORK/t_cross" "transportModel CrossPowerLaw" "Newtonian"
cp -r "$WORK/lam" "$WORK/t_nonu"
set_entry "$WORK/t_nonu/constant/transportProperties" nu -
refuses transport_no_nu "$WORK/t_nonu" "has no \`nu\`" "Newtonian.C:56"
cp -r "$WORK/lam" "$WORK/t_nomodel"
set_entry "$WORK/t_nomodel/constant/transportProperties" transportModel -
refuses transport_no_model "$WORK/t_nomodel" "no \`transportModel\`"

# ---- turbulenceProperties ----------------------------------------------------------------------------------------
cp -r "$WORK/lam" "$WORK/tp_none"
rm "$WORK/tp_none/constant/turbulenceProperties"
refuses turbulence_properties_missing "$WORK/tp_none" "no constant/turbulenceProperties" "MUST_READ"
cp -r "$WORK/tp_none" "$WORK/tp_org"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object momentumTransport; }\nsimulationType laminar;\n' \
    > "$WORK/tp_org/constant/momentumTransport"
refuses turbulence_properties_org_name "$WORK/tp_org" "momentumTransport" "OpenFOAM.org"
cp -r "$WORK/lam" "$WORK/tp_nosim"
set_entry "$WORK/tp_nosim/constant/turbulenceProperties" simulationType -
refuses turbulence_no_simulation_type "$WORK/tp_nosim" "no \`simulationType\`"
# the dictionary is audited now: an entry no reader takes is named at the end of the run
box "$WORK/ke" --model kEpsilon
accepts kepsilon_box "$WORK/ke"
cp -r "$WORK/ke" "$WORK/tp_audit"
python3 - "$WORK/tp_audit/constant/turbulenceProperties" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read().replace('turbulence on;', 'turbulence on;\n    notAnOpenFoamKey 1;')
open(p, 'w').write(s)
PY
accepts turbulence_properties_audited "$WORK/tp_audit"
if grep -qF "notAnOpenFoamKey" "$WORK/turbulence_properties_audited.log" \
    && grep -qF "turbulenceProperties" "$WORK/turbulence_properties_audited.log"; then
    echo "ok:   turbulence_properties_unread_entry_named"
else
    echo "FAIL: turbulence_properties_unread_entry_named -- the unread entry was not reported"
    fail=1
fi

# ---- controlDict deltaT / endTime --------------------------------------------------------------------------------
cp -r "$WORK/lam" "$WORK/c_nodt"
set_entry "$WORK/c_nodt/system/controlDict" deltaT -
refuses controldict_no_deltat "$WORK/c_nodt" "no \`deltaT\`" "TimeIO.C:272-275"
cp -r "$WORK/lam" "$WORK/c_noend"
set_entry "$WORK/c_noend/system/controlDict" endTime -
refuses controldict_no_endtime "$WORK/c_noend" "no \`endTime\`" "stopAt endTime"
cp -r "$WORK/c_noend" "$WORK/c_noend_nostop"
set_entry "$WORK/c_noend_nostop/system/controlDict" stopAt -
refuses controldict_no_endtime_no_stopat "$WORK/c_noend_nostop" "no \`endTime\`" "takes endTime 0"

# ---- named ddt entries -------------------------------------------------------------------------------------------
cp -r "$WORK/lam" "$WORK/d_u"
sed -i 's/ddtSchemes { default Euler; }/ddtSchemes { default Euler; ddt(U) backward; }/' "$WORK/d_u/system/fvSchemes"
refuses ddt_named_U "$WORK/d_u" "ddt(U)" "backward" "default"
accepts ddt_named_U_control "$WORK/d_u" BRAE_CONTROL_DDT_NAMED_UNREAD=1
cp -r "$WORK/lam" "$WORK/d_regex"
sed -i 's/ddtSchemes { default Euler; }/ddtSchemes { default Euler; "ddt(.*)" CrankNicolson 0.9; }/' \
    "$WORK/d_regex/system/fvSchemes"
refuses ddt_regex "$WORK/d_regex" "ddt(U)" "CrankNicolson 0.9"
cp -r "$WORK/lam" "$WORK/d_same"
sed -i 's/ddtSchemes { default Euler; }/ddtSchemes { default Euler; ddt(U) Euler; }/' "$WORK/d_same/system/fvSchemes"
accepts ddt_named_same_as_default "$WORK/d_same"
cp -r "$WORK/lam" "$WORK/d_unsolved"
sed -i 's/ddtSchemes { default Euler; }/ddtSchemes { default Euler; ddt(k) backward; }/' "$WORK/d_unsolved/system/fvSchemes"
accepts ddt_named_field_not_solved "$WORK/d_unsolved"
cp -r "$WORK/ke" "$WORK/d_k"
sed -i 's/ddtSchemes { default Euler; }/ddtSchemes { default Euler; ddt(k) backward; }/' "$WORK/d_k/system/fvSchemes"
refuses ddt_named_k "$WORK/d_k" "ddt(k)" "backward"

# ---- mesh motion -------------------------------------------------------------------------------------------------
box "$WORK/mv" --model laminar --zone all
dyn_rotating "$WORK/mv"
accepts moving_box "$WORK/mv"
cp -r "$WORK/mv" "$WORK/mv_restart"
cp -r "$WORK/mv_restart/0" "$WORK/mv_restart/0.001"
set_entry "$WORK/mv_restart/system/controlDict" startFrom latestTime
refuses moving_restart "$WORK/mv_restart" "starts from time 0.001" "after its first time directory" "polyMesh/points"
cp -r "$WORK/mv" "$WORK/mv_points"
mkdir -p "$WORK/mv_points/0/polyMesh"
cp "$WORK/mv_points/constant/polyMesh/points" "$WORK/mv_points/0/polyMesh/points"
refuses moving_start_moved_points "$WORK/mv_points" "holds polyMesh/points"
for e in origin axis omega; do
    cp -r "$WORK/mv" "$WORK/mv_no_$e"
    dyn_rotating "$WORK/mv_no_$e" "$e"
    refuses "rotating_no_$e" "$WORK/mv_no_$e" "has no \`$e\`" "OpenFOAM reads it there with no default"
done
cp -r "$WORK/mv" "$WORK/mv_table"
set_entry "$WORK/mv_table/constant/dynamicMeshDict" omega 'table ((0 0) (1 10))'
refuses rotating_omega_table "$WORK/mv_table" "is not a constant brae can evaluate"
cp -r "$WORK/mv" "$WORK/mv_sub"
set_entry "$WORK/mv_sub/constant/dynamicMeshDict" omega -
append "$WORK/mv_sub/constant/dynamicMeshDict" "omega { type sine; amplitude 1; frequency 1; scale 1; level 0; }"
refuses rotating_omega_function1_dict "$WORK/mv_sub" "Function1 sub-dictionary"
cp -r "$WORK/mv" "$WORK/mv_const"
set_entry "$WORK/mv_const/constant/dynamicMeshDict" omega 'constant 1'
accepts rotating_omega_constant_function1 "$WORK/mv_const"
box "$WORK/osc" --model laminar --zone all
dyn_oscillating "$WORK/osc"
accepts oscillating_box "$WORK/osc"
cp -r "$WORK/osc" "$WORK/osc_misplaced"
append "$WORK/osc_misplaced/constant/dynamicMeshDict" "phaseShift 0.1;"
refuses oscillating_entry_outside_coeffs "$WORK/osc_misplaced" "phaseShift" "where OpenFOAM reads it"
cp -r "$WORK/mv" "$WORK/u_runtime"
append "$WORK/u_runtime/constant/dynamicMeshDict" "updateControl runTime;
updateInterval 0.1;"
refuses update_control_runtime "$WORK/u_runtime" "updateControl runTime" "not ported"
cp -r "$WORK/mv" "$WORK/u_zero"
append "$WORK/u_zero/constant/dynamicMeshDict" "updateControl timeStep;
updateInterval 0;"
refuses update_interval_zero "$WORK/u_zero" "updateInterval 0" "1 or less"
cp -r "$WORK/mv" "$WORK/u_frac"
append "$WORK/u_frac/constant/dynamicMeshDict" "updateControl timeStep;
updateInterval 2.5;"
refuses update_interval_fraction "$WORK/u_frac" "not an integer"
cp -r "$WORK/mv" "$WORK/u_two"
append "$WORK/u_two/constant/dynamicMeshDict" "updateControl timeStep;
updateInterval 2;"
accepts update_interval_two "$WORK/u_two"
cp -r "$WORK/mv" "$WORK/mmoc"
sed -i 's/nOuterCorrectors 1;/nOuterCorrectors 2;\n    moveMeshOuterCorrectors yes;/' "$WORK/mmoc/system/fvSolution"
refuses move_mesh_outer_correctors "$WORK/mmoc" "moveMeshOuterCorrectors yes" "nOuterCorrectors 2"
cp -r "$WORK/mv" "$WORK/mmoc1"
sed -i 's/nOuterCorrectors 1;/nOuterCorrectors 1;\n    moveMeshOuterCorrectors yes;/' "$WORK/mmoc1/system/fvSolution"
accepts move_mesh_outer_correctors_one_corrector "$WORK/mmoc1"
box "$WORK/mv_sst" --model kOmegaSST --zone all
dyn_rotating "$WORK/mv_sst"
refuses moving_wall_distance_sst "$WORK/mv_sst" "kOmegaSST on a moving mesh" "wall distance"
cp -r "$WORK/mv_sst" "$WORK/mv_sst_frozen_y"
append "$WORK/mv_sst_frozen_y/system/fvSchemes" "wallDist { method meshWave; updateInterval 0; }"
accepts moving_wall_distance_never_updated "$WORK/mv_sst_frozen_y"
box "$WORK/mv_ke" --model kEpsilon --zone all
dyn_rotating "$WORK/mv_ke"
accepts moving_kepsilon "$WORK/mv_ke"
box "$WORK/cy_rot" --model laminar --cyclic-x --zone all
dyn_rotating "$WORK/cy_rot"
refuses moving_cyclic_rotating "$WORK/cy_rot" "plain cyclic patch" "rotates"
box "$WORK/cy_trans" --model laminar --cyclic-x --zone all
dyn_oscillating "$WORK/cy_trans"
accepts moving_cyclic_translating_rigidly "$WORK/cy_trans"
box "$WORK/cy_def" --n 2 4 2 --model laminar --cyclic-x --zone half
dyn_oscillating "$WORK/cy_def"
refuses moving_cyclic_deforming "$WORK/cy_def" "plain cyclic patch" "deforms"
box "$WORK/cy_apart" --n 8 4 2 --model laminar --cyclic-x --zone half
dyn_oscillating "$WORK/cy_apart"
refuses moving_cyclic_halves_apart "$WORK/cy_apart" "part of it moves"
box "$WORK/cy_vcl" --model laminar --cyclic-x
printf 'FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\ndynamicFvMesh dynamicMotionSolverFvMesh;\nmotionSolver velocityComponentLaplacian;\ncomponent x;\ndiffusivity uniform;\n' \
    > "$WORK/cy_vcl/constant/dynamicMeshDict"
refuses moving_cyclic_velocity_laplacian "$WORK/cy_vcl" "plain cyclic" "velocityComponentLaplacian"
cp -r "$WORK/osc" "$WORK/f_moving"
forces_fo "$WORK/f_moving" forces
refuses forces_on_moving_mesh "$WORK/f_moving" "wallForces (forces) on a moving mesh"
cp -r "$WORK/osc" "$WORK/fc_moving"
forces_fo "$WORK/fc_moving" forceCoeffs
refuses force_coeffs_on_moving_mesh "$WORK/fc_moving" "wallForces (forceCoeffs) on a moving mesh"
cp -r "$WORK/lam" "$WORK/f_static"
forces_fo "$WORK/f_static" forces
accepts forces_static_mesh "$WORK/f_static"

# ---- adjustTimeStep ----------------------------------------------------------------------------------------------
cp -r "$WORK/lam" "$WORK/a_notice"
set_entry "$WORK/a_notice/system/controlDict" adjustTimeStep yes
set_entry "$WORK/a_notice/system/controlDict" maxCo 0.5
set_entry "$WORK/a_notice/system/controlDict" writeControl adjustable
set_entry "$WORK/a_notice/system/controlDict" writeInterval 0.001
accepts adjust_time_step_runs "$WORK/a_notice"
if grep -qF "issue #53" "$WORK/adjust_time_step_runs.log" \
    && grep -qF "writes land on the first step past each write time" "$WORK/adjust_time_step_runs.log"; then
    echo "ok:   adjust_time_step_notice"
else
    echo "FAIL: adjust_time_step_notice -- no start-up notice naming issue #53 and the write times"
    fail=1
fi
cp -r "$WORK/f_static" "$WORK/a_forces"
set_entry "$WORK/a_forces/system/controlDict" adjustTimeStep yes
set_entry "$WORK/a_forces/system/controlDict" maxCo 0.5
refuses forces_adjust_time_step "$WORK/a_forces" "wallForces (forces) with \`adjustTimeStep yes\`"
cp -r "$WORK/lam" "$WORK/a_scalar"
set_entry "$WORK/a_scalar/system/controlDict" adjustTimeStep yes
set_entry "$WORK/a_scalar/system/controlDict" maxCo 0.5
append "$WORK/a_scalar/system/controlDict" "functions { tracer { type scalarTransport; libs (solverFunctionObjects); field s; D 1e-3; } }"
sed -i 's/div(phi,U) Gauss linear;/div(phi,U) Gauss linear;\n    div(phi,s) Gauss upwind;/' "$WORK/a_scalar/system/fvSchemes"
sed 's/object U;/object s;/; s/volVectorField/volScalarField/; s/\[0 1 -1 0 0 0 0\]/[0 0 0 0 0 0 0]/; s/(1 0 0)/0/g' \
    "$WORK/a_scalar/0/U" | sed 's/type noSlip;/type zeroGradient;/' > "$WORK/a_scalar/0/s"
refuses scalar_transport_adjust_time_step "$WORK/a_scalar" "tracer (scalarTransport) with \`adjustTimeStep yes\`"
cp -r "$WORK/a_scalar" "$WORK/s_fixed"
set_entry "$WORK/s_fixed/system/controlDict" adjustTimeStep -
accepts scalar_transport_fixed_step "$WORK/s_fixed"

# ---- LES and DES -------------------------------------------------------------------------------------------------
box "$WORK/les" --model LES
accepts smagorinsky_cuberootvol_3d "$WORK/les"
les_dict()   # les_dict <case> <LES body>
{
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\nsimulationType LES;\nLES\n{\n%s\n}\n' \
        "$2" > "$1/constant/turbulenceProperties"
}
cp -r "$WORK/les" "$WORK/l_wale"
les_dict "$WORK/l_wale" "LESModel WALE; turbulence on; delta cubeRootVol;"
accepts wale_cuberootvol_3d "$WORK/l_wale"
for m in Smagorinsky WALE; do
    cp -r "$WORK/les" "$WORK/l_max_$m"
    les_dict "$WORK/l_max_$m" "LESModel $m; turbulence on; delta maxDeltaxyz;"
    refuses "${m}_maxdeltaxyz" "$WORK/l_max_$m" "LESModel $m\` with \`delta maxDeltaxyz" "cbrt(V)"
done
cp -r "$WORK/les" "$WORK/l_coeff"
les_dict "$WORK/l_coeff" "LESModel Smagorinsky; turbulence on; delta cubeRootVol; cubeRootVolCoeffs { deltaCoeff 2; }"
refuses cuberootvol_deltacoeff "$WORK/l_coeff" "cubeRootVol \`deltaCoeff 2" "coefficient of 1"
cp -r "$WORK/les" "$WORK/l_coeff_flat"
les_dict "$WORK/l_coeff_flat" "LESModel Smagorinsky; turbulence on; delta cubeRootVol; deltaCoeff 2;"
refuses cuberootvol_deltacoeff_in_les_dict "$WORK/l_coeff_flat" "deltaCoeff 2"
cp -r "$WORK/les" "$WORK/l_nodelta"
les_dict "$WORK/l_nodelta" "LESModel Smagorinsky; turbulence on;"
refuses les_no_delta "$WORK/l_nodelta" "LES has no \`delta\`"
box "$WORK/des" --model SAIDDES
for m in SpalartAllmarasDES kOmegaSSTDES; do
    cp -r "$WORK/des" "$WORK/l_$m"
    les_dict "$WORK/l_$m" "LESModel $m; turbulence on; delta cubeRootVol;"
    refuses "${m}_refused" "$WORK/l_$m" "LESModel $m\` is not ported" "DDES"
done
cp -r "$WORK/des" "$WORK/l_ddes_exact"
les_dict "$WORK/l_ddes_exact" "LESModel SpalartAllmarasDDES; turbulence on; delta cubeRootVol;"
append "$WORK/l_ddes_exact/system/fvSchemes" "wallDist { method exactDistance; }"
accepts ddes_exact_distance "$WORK/l_ddes_exact"
accepts iddes_box "$WORK/des"
cp -r "$WORK/des" "$WORK/i_cuberoot"
les_dict "$WORK/i_cuberoot" "LESModel SpalartAllmarasIDDES; turbulence on; delta IDDESDelta; IDDESDeltaCoeffs { hmax maxDeltaxyzCubeRoot; maxDeltaxyzCubeRootCoeffs { } }"
accepts iddes_hmax_maxdeltaxyzcuberoot_box "$WORK/i_cuberoot"
cp -r "$WORK/des" "$WORK/i_delta"
les_dict "$WORK/i_delta" "LESModel SpalartAllmarasIDDES; turbulence on; delta cubeRootVol;"
refuses iddes_other_delta "$WORK/i_delta" "\`delta cubeRootVol\`" "IDDESDelta"
cp -r "$WORK/des" "$WORK/i_cw"
les_dict "$WORK/i_cw" "LESModel SpalartAllmarasIDDES; turbulence on; delta IDDESDelta; IDDESDeltaCoeffs { Cw 0.2; }"
refuses iddes_cw "$WORK/i_cw" "IDDES Cw" "IDDESDeltaCoeffs"
cp -r "$WORK/des" "$WORK/i_hmax"
les_dict "$WORK/i_hmax" "LESModel SpalartAllmarasIDDES; turbulence on; delta IDDESDelta; IDDESDeltaCoeffs { hmax cubeRootVol; }"
refuses iddes_hmax_other "$WORK/i_hmax" "hmax cubeRootVol\` is not ported"
cp -r "$WORK/des" "$WORK/i_hmax_coeff"
les_dict "$WORK/i_hmax_coeff" "LESModel SpalartAllmarasIDDES; turbulence on; delta IDDESDelta; IDDESDeltaCoeffs { maxDeltaxyzCoeffs { deltaCoeff 1; } }"
refuses iddes_hmax_deltacoeff "$WORK/i_hmax_coeff" "hmax delta's deltaCoeff"
cp -r "$WORK/des" "$WORK/i_exact"
append "$WORK/i_exact/system/fvSchemes" "wallDist { method exactDistance; }"
refuses iddes_exact_distance "$WORK/i_exact" "exactDistance" "meshWave"
box "$WORK/i_skew" --model SAIDDES --skew 1.5
refuses iddes_hmax_not_a_box "$WORK/i_skew" "IDDES hmax" "differs from OpenFOAM's maxDeltaxyz"
accepts iddes_hmax_not_a_box_control "$WORK/i_skew" BRAE_CONTROL_LES_MESH_UNCHECKED=1

[ "$fail" -eq 0 ] && echo "PASS: pimpleFoam refuses at set-up what it would compute differently, and sets up the rest"
exit "$fail"
