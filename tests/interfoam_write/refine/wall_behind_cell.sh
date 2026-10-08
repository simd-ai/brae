#!/usr/bin/env bash
# The write gate: a wall face that lies BEHIND its own cell's centre -- n.(Cf - C) < 0, a warped wall-corner
# cell, here the child a refinement makes of a snapped one. OpenFOAM's patch delta coefficient is the MAGNITUDE
# 1/|n.(Cf - C)| (basicFvGeometryScheme.C: `deltaCoeffsBf[patchi] = 1.0/mag(p.delta())`, fvPatch.C:156-161);
# brae's was 1/(n.(Cf - C)), signed, so such a face took a negative wall diffusion coefficient and a boundary
# snGrad of the wrong sign. FOUND 2026-10-07 by RAS/motorBike's run to its own end time: it stopped at
# t = 0.436 of 2 with one such cell's velocity at 2,600 m/s, where serial OpenFOAM runs through; no start mesh
# of the 42 tutorials has such a face and every two-step row starts before a refinement makes one.
# THE FIXTURE makes them at once: RAS/motorBike's pinned row with the water block lowered onto the bike
# (z 0.9 to 1.3 for 1.5 to 1.75) and moving down at 3 m/s, six steps -- the interface is on the walls from
# the start, so the first refinements split snapped wall cells. ORACLE: real interFoam on the same staged case.
# Three checks. (1) THE PREMISE, as numbers: OpenFOAM refines the mesh and brae counts faces behind their
# cell's centre on it. (2) Each loop within its own bound of OpenFOAM, every written file. MEASURED 2026-10-07:
# 8 such faces of a mesh refined to 30,824 cells; host worst 4.1e-13 (rAU), so its bound is 5e-12; device
# worst 1.3e-10 (nut -- 1.3e-14 absolute in ten cells under the open top, where the air is all but at rest
# and nut divides by a strain rate formed from differences of 1e-13; every other device file is 4.6e-13 or
# less), so its bound is 1.5e-09. (3) CONTROL: BRAE_CONTROL_PATCH_DELTA_SIGNED=1, the signed form: k 1.1e-01,
# U 5.1e-03.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase motorBike of > "$W/wb_stage.txt" 2>&1
o="$W/w_of_motorBike"
[ -d "$o" ] || { say "motorBike did not stage" FAIL; finish "a wall face behind its cell's centre"; }
restage()   # restage <dir>: the water block on the bike and moving, six steps, one write
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$o/0" "$o/constant" "$o/system" "$1/"
    python3 - "$1" <<'PY' || return 1
import re, sys
d = sys.argv[1]
p = d + '/system/setFieldsDict'
t = open(p).read()
t, a = re.subn(r'box \( 0 -0.5 1.5 \) \( 2 0.5 1.75 \);', 'box ( 0 -0.5 0.9 ) ( 2 0.5 1.3 );', t)
t, b = re.subn(r'(volScalarFieldValue alpha.water 1)', r'\1\n            volVectorFieldValue U ( 0 0 -3 )', t)
assert (a, b) == (1, 1), (a, b)
open(p, 'w').write(t)
c = d + '/system/controlDict'
t = open(c).read()
dt = float(re.search(r'^deltaT\s+([^;]+);', t, re.M).group(1))
t = re.sub(r'^endTime\s.*', 'endTime         %.12g;' % (6*dt), t, flags=re.M)
t = re.sub(r'^writeInterval\s.*', 'writeInterval   6;', t, flags=re.M)
open(c, 'w').write(t)
PY
    ( cd "$1" && setFields > log.setFields.lowered 2>&1 )
}
# behind <log>: the faces brae counted behind their cell's centre, the last count it printed
behind() { grep -a "boundary face(s) lie behind" "$1" | tail -1 | sed -E 's/.*patches: ([0-9]+) boundary.*/\1/'; }
restage "$W/wb_of" \
    || { say "the lowered water block did not stage" FAIL; finish "a wall face behind its cell's centre"; }
runof "$W/wb_of"
ot=$(timedirs "$W/wb_of")
cells=$(grep "Refined from" "$W/wb_of/log.interFoam" | tail -1 | sed -E 's/.* to ([0-9]+) cells.*/\1/')
for v in host device control; do
    restage "$W/wb_$v"
    case $v in
        host)    runbrae "$W/wb_$v" host BRAE_X=1 ;;
        device)  runbrae "$W/wb_$v" device BRAE_X=1 ;;
        control) runbrae "$W/wb_$v" host BRAE_CONTROL_PATCH_DELTA_SIGNED=1 ;;
    esac
    python3 "$CMP" "$W/wb_of" "$W/wb_$v" $ot > "$W/cmp_wb_$v.txt" 2>&1
done
nh=$(behind "$W/wb_host/log.brae")
nd=$(behind "$W/wb_device/log.brae")
what="PREMISE  OpenFOAM refines the mesh to ${cells:-?} cells in [$ot]; brae counts ${nh:-0} boundary faces behind"
what="$what their cell's centre (device ${nd:-0})"
[ "$(echo $ot | wc -w)" = 1 ] && [ "${cells:-0}" -gt 20000 ] && [ "${nh:-0}" -ge 4 ] && [ "$nh" = "$nd" ] \
    && say "$what" ok || say "$what" FAIL
BH=5e-12
BD=1.5e-09
judge "a wall face behind its cell, host" "$W/cmp_wb_host.txt" "$BH" "$W/wb_of/log.interFoam" \
    && judge "a wall face behind its cell, device" "$W/cmp_wb_device.txt" "$BD" "$W/wb_of/log.interFoam" \
    && say "[host, device] every written file within $BH and $BD of OpenFOAM" ok \
    || say "[host, device] every written file within $BH and $BD of OpenFOAM" FAIL
judge "control: the signed coefficient" "$W/cmp_wb_control.txt" "$BD" "$W/wb_of/log.interFoam" > "$W/wb_control.txt"
rc=$?
over=$(grep -oE "over the bound: [0-9.e+-]+/(U|k) [0-9.e+-]+" "$W/wb_control.txt" \
       | sed -E 's/over the bound: [^ ]*\///' | tr '\n' ' ')
what="CONTROL  the signed coefficient: U and k go over the bound (${over:-neither})"
[ $rc != 0 ] && grep -aq "CONTROL MODE: it is signed here" "$W/wb_control/log.brae" \
    && echo "$over" | grep -q "^U " && echo "$over" | grep -q "k " && say "$what" ok || say "$what" FAIL
finish "a wall face behind its cell's centre takes OpenFOAM's delta coefficient, 1/|n.(Cf - C)|"
