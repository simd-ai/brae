#!/usr/bin/env bash
# The gate: probes on a MOVING mesh. polyMesh::movePoints ends by telling the function objects
# (polyMesh.C:1292), and a probes object then searches its cells again -- but only under `fixedLocations`,
# the default (probes.C:560-568). With `fixedLocations false` the cells found on the START mesh are kept, so
# the probe rides with the mesh.
# The row is laminar/sloshingTank2D, a solid-body moved tank, with the tutorial's OWN `functions` entry -- a
# probes object (`fixedLocations false`, `writeControl writeTime`, two locations beside the walls) and a
# `surfaces` object, which brae does not port and must say so by name -- and one object added:
#   fixed   the default `fixedLocations true`, every step, at LOCATIONS -- deep in the water, just under the
#           free surface, just over it, and in the air: the search is repeated after every move, and the
#           cell under a location changes as the tank moves.
#   riding  the same locations with `fixedLocations false`: OpenFOAM's own witness that the cells changed.
# STEPS steps of the tutorial's deltaT, a write every WRITE, solves pinned (the staging of ../lib.sh).
# Three checks. (1) THE HOST LOOP: the three objects' files against OpenFOAM's -- heads, rows, time columns,
# values within HOST_BOUND -- and the `surfaces` object named as not run. (2) THE GPU LOOP, the same.
# (3) CONTROL, BRAE_CONTROL_PROBE_SEARCH=once: the search not repeated puts `fixed` off OpenFOAM's by more
# than CONTROL and leaves `riding` within the bound.
# MEASURED 2026-10-08: 40 steps, 162 rows in five files, heads and time columns OpenFOAM's bytes, the values
# within 6.6e-11 on the host and 3.4e-11 on the GPU (U; p 5.8e-14, alpha.water 1.1e-13). OpenFOAM's own fixed
# and riding p part by 7.3e-02; the control puts brae's fixed p that far off and leaves riding at 5.8e-14. 14 s.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
. "$(dirname "$0")/_common.sh"
STEPS=40
WRITE=20
LOCATIONS="((0 3.3 -6.1) (0 -7.4 -0.3) (0 5.1 0.4) (0 0 9.9))"
HOST_BOUND=7e-10
DEVICE_BOUND=7e-10
CONTROL=1e-4
name="probes on a moving mesh"
src="$TUT/multiphase/interFoam/laminar/sloshingTank2D"
o="$W/mm_of"
stage_allrun "$src" "$o" "" || { say "laminar/sloshingTank2D meshed" FAIL; finish "$name"; }
python3 - "$o/system/controlDict" "$src/system/controlDict" "$STEPS" "$WRITE" "$LOCATIONS" <<'PY' \
    || { say "PREMISE  the tutorial's own functions entry was staged" FAIL; finish "$name"; }
import re, sys
c, shipped, steps, write, locations = sys.argv[1:6]
s = open(c).read()
own = re.search(r'\nfunctions\s*\{.*?\n\}', open(shipped).read(), re.S).group(0).strip()
added = '''
    fixed
    {
        type            probes;
        libs            (sampling);
        fields          (p alpha.water U);
        probeLocations  %s;
    }
    riding
    {
        type            probes;
        libs            (sampling);
        fixedLocations  false;
        fields          (p);
        probeLocations  %s;
    }
}''' % (locations, locations)
assert own.endswith('}') and s.count('functions\n{\n}') == 1
s = s.replace('functions\n{\n}', own[:-1].rstrip() + added, 1)
dt = float(re.search(r'^deltaT\s+([^;]+);', s, re.M).group(1))
s = re.sub(r'^endTime\s.*', 'endTime         %.12g;' % (int(steps)*dt), s, flags=re.M)
s = re.sub(r'^writeInterval\s.*', 'writeInterval   %s;' % write, s, flags=re.M)
open(c, 'w').write(s)
PY
runof "$o"
files="probes/0/p fixed/0/p fixed/0/alpha.water fixed/0/U riding/0/p"
read -r ro rb hd td moved <<< "$(fo_file "$o/postProcessing/riding/0/p" "$o/postProcessing/fixed/0/p")"
! grep -q "loading function object" "$o/log.interFoam" && [ "$(grep -c "^Time = " "$o/log.interFoam")" = $STEPS ] \
    && [ "$(grep -vc '^#' "$o/postProcessing/probes/0/p")" = $((STEPS/WRITE)) ] && [ "$ro" = $STEPS ] \
    && ! grep -q "Not Found" "$o/postProcessing/fixed/0/p" && above "$moved" $CONTROL \
    && above "$(awk '!/^#/{for(i=2;i<=NF;i++) if($i>m) m=$i} END{print m+0}' "$o/postProcessing/fixed/0/alpha.water")" \
        0.5 \
    || { say "PREMISE  OpenFOAM: $STEPS steps, all objects loaded, water sampled, fixed p $moved from riding" FAIL
         finish "$name"; }
arm()   # arm <run dir> <label> <bound>
{
    local e="$1" ok=1 worst=0 rows=0 f
    for f in $files; do
        read -r ro rb hd td gap <<< "$(fo_file "$o/postProcessing/$f" "$e/postProcessing/$f")"
        [ "$rb" = - ] && { rb=0; gap=1; }
        [ "$ro" = "$rb" ] && [ "$hd" = 0 ] && [ "$td" = 0 ] && within "$gap" "$3" || ok=0
        rows=$((rows + rb))
        worst=$(python3 -c "print('%.1e' % max(float('$worst'), float('$gap')))")
    done
    grep -aq "functions/wallPressure.*surfaces.*NOT run" "$e/log.brae" || ok=0
    what="[$2] five files: OpenFOAM's heads, its $rows rows, its time columns, values within $worst (bound"
    what="$what $3); the fixed locations' cells moved (p $moved from the riding ones'); \`surfaces\` named as not run"
    [ $ok = 1 ] && say "$what" ok || say "$what" FAIL
}
fo_arm "$o" "$W/mm_host" host BRAE_X=1
arm "$W/mm_host" host $HOST_BOUND
fo_arm "$o" "$W/mm_device" device BRAE_X=1
arm "$W/mm_device" device $DEVICE_BOUND
fo_arm "$o" "$W/mm_once" host BRAE_CONTROL_PROBE_SEARCH=once
read -r ro rb hd td fx <<< "$(fo_file "$o/postProcessing/fixed/0/p" "$W/mm_once/postProcessing/fixed/0/p")"
read -r ro rb hd td rd <<< "$(fo_file "$o/postProcessing/riding/0/p" "$W/mm_once/postProcessing/riding/0/p")"
what="CONTROL  the search not repeated after a move: the fixed locations' p off by $fx (above $CONTROL), the"
what="$what riding ones' within $rd"
above "$fx" $CONTROL && within "$rd" $HOST_BOUND && say "$what" ok || say "$what" FAIL
finish "probes on a moving mesh search again under fixedLocations and ride with the mesh without, on both loops"
