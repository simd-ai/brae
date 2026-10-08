#!/usr/bin/env bash
# parts.sh <label> <tutorialDir> [deltaT override]: ONE case timed PART BY PART on three arms, as shipped.
#   of1 / of20  timeInterFoam -- OpenFOAM's interFoam.C with wall-clock accumulators around the time-step
#               control, mesh.update()+correctPhi, alpha (alphaEqnSubCycle + mixture.correct), UEqn, the pEqn loop
#               and turbulence->correct(), printed cumulatively each step (tools/timeInterFoam) -- serial and
#               on 20 cores (scotch). N steps exactly (stopAt nextWrite).
#   brae        brae_interFoam -device with BRAE_INTER_PHASE_TIME=1, run to OpenFOAM's step N and step N/2;
#               the per-step cost of each phase is the DIFFERENCE of the two runs' totals, which drops start-up.
# OpenFOAM's parts are read between steps K0 and N-1 the same way. A parallel run is a timing baseline only.
set -u
B=/home/ghost/cudafoam/runs/bench
BRAE=/home/ghost/cudafoam/brae
N=${BENCH_STEPS:-30}
K0=${BENCH_SKIP:-10}
LIMIT=${BENCH_TIMEOUT:-2400}
label="$1"
src="$2"
dtw="${3:-}"
export KEEP_W=$B/parts
export BUILD=$BRAE/build
export BRAE_OF_TUTORIALS=/usr/lib/openfoam/openfoam2412/tutorials
. $BRAE/tests/interfoam_write/lib.sh
unset BRAE_PRESSURE_CASE_SOLVER
set +u; source /usr/lib/openfoam/openfoam2412/etc/bashrc > /dev/null 2>&1; set -u
mkdir -p "$W"

ctl()   # ctl <out> <of|brae> <endTime>
{
    python3 - "$src/system/controlDict" "$1" "$2" "$3" "$dtw" "$N" <<'PY'
import re, sys
srcp, out, mode, endTime, dtOverride, n = sys.argv[1:7]
s = open(srcp).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}', '\nfunctions\n{\n}', s, flags=re.S)
if not re.search(r'^functions', s, re.M):
    s += '\nfunctions\n{\n}\n'
# brae's two runs write nothing: a write at the end of the longer one would land in the difference
kv = [('writeControl', 'timeStep'), ('writeInterval', n if mode == 'of' else '1000000'), ('purgeWrite', '0'),
      ('startFrom', 'startTime'),
      ('stopAt', 'nextWrite' if mode == 'of' else 'endTime'), ('endTime', endTime)]
if dtOverride:
    kv.append(('deltaT', dtOverride))
for k, v in kv:
    if re.search(r'^%s\s' % k, s, flags=re.M):
        s = re.sub(r'^%s\s.*' % k, '%s %s;' % (k.ljust(15), v), s, flags=re.M)
    else:
        s = s.replace('\nfunctions\n', '\n%s %s;\nfunctions\n' % (k.ljust(15), v), 1)
open(out, 'w').write(s)
PY
}

mk()   # mk <arm> <of|brae> <endTime>
{
    local d="$W/p_${label}_$1"
    rm -rf "${d:?}"
    mkdir -p "$d"
    cp -r "$W/s_$label/0" "$W/s_$label/constant" "$W/s_$label/system" "$d/"
    cp "$src/system/fvSolution" "$d/system/fvSolution"
    ctl "$d/system/controlDict" "$2" "$3"
    echo "$d"
}

slim()
{
    rm -rf "${1:?}"/constant "${1:?}"/processor* "${1:?}"/[0-9]* "${1:?}"/dynamicCode 2>/dev/null
}

rm -rf "${W:?}/s_$label"
stage_allrun "$src" "$W/s_$label" "$dtw" > "$W/stage_$label.txt" 2>&1 || { echo "$label: staging failed"; exit 1; }
cells=$(grep -ao "nCells: *[0-9]*" "$W/s_$label/constant/polyMesh/owner" | head -1 | grep -o "[0-9]*$")

d=$(mk of1 of 1e6)
( cd "$d" && timeout "$LIMIT" timeInterFoam > log.of 2>&1 )
tN=$(grep -a "^Time = " "$d/log.of" | sed -n "${N}p" | awk '{print $3}')
tH=$(grep -a "^Time = " "$d/log.of" | sed -n "$((N/2))p" | awk '{print $3}')
slim "$d"
d=$(mk of20 of 1e6)
python3 - "$d/system/decomposeParDict" <<'PY'
import os, re, sys
p = sys.argv[1]
if os.path.exists(p):
    s = open(p).read()
    s = re.sub(r'numberOfSubdomains\s+\d+\s*;', 'numberOfSubdomains 20;', s)
    s = re.sub(r'^method\s+\w+\s*;', 'method          scotch;', s, flags=re.M)
else:
    s = ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n'
         '    object      decomposeParDict;\n}\nnumberOfSubdomains 20;\nmethod          scotch;\n')
open(p, 'w').write(s)
PY
( cd "$d" && decomposePar -force > log.decomposePar 2>&1 \
    && timeout "$LIMIT" mpirun -np 20 --bind-to core timeInterFoam -parallel > log.of 2>&1 )
slim "$d"
if [ -n "$tN" ] && [ -n "$tH" ]; then
    for arm in full:$tN half:$tH; do
        d=$(mk brae_${arm%%:*} brae "${arm#*:}")
        ( cd "$d" && BRAE_INTER_PHASE_TIME=1 timeout "$LIMIT" $BRAE/build/brae_interFoam -case . -device > log.brae 2>&1 )
        slim "$d"
    done
fi
rm -rf "${W:?}/s_$label"
python3 $B/parts_report.py "$label" "${cells:-?}" "$N" "$K0"
