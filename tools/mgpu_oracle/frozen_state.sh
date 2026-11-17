#!/bin/bash
# Makes a FROZEN STATE for the multi-rank I/O harness (tests/mgpu_frozen.cu): a serial OpenFOAM run of one of the
# oracle cases stopped after a few steps, decomposePar of its result time into processor0..N-1, and OpenFOAM's own
# answers for that one state -- the function objects (postProcess -parallel), reconstructPar.
#
#   tools/mgpu_oracle/frozen_state.sh <case> <np> [method=scotch] [steps=5]
#       case    tank | tankGraded | open       (tools/mgpu_oracle/cases/)
#       np      1 | 2 | 4 | 8                  (np 1: no decomposition, the serial state itself)
#       method  scotch | simple | hier         (system/decomposeParDict.<method><np>)
#   -> $OUT_ROOT/<case>_np<N>[_<method>]/      OUT_ROOT default /tmp/brae-frozen
#        serial/        mesh + the serial run (dumpInterFoam, BRAE_STOP_ITER=steps); its last time is the state
#        dec/           the decomposed case: system/, processorN/{constant, <time>}
#        dec/postProcessing   OpenFOAM's function objects evaluated on that state (postProcess -parallel;
#                       np 1: postProcess), a reference for compare_fo.py. It READS the fields back, which re-evaluates a
#                       contact-angle patch (alpha's wall values move by ~1e-9): dec/inrun is OpenFOAM serial's own
#                       output while it made the state (compare_fo.py --last-row), dec_par/inrun OpenFOAM -parallel's
#                       own output while it made ITS state dec_par/ (processor patch values = coupled ones)
#        dec/<time>     reconstructPar of the state: equal to serial/<time> (the identity the writer gate builds on)
#        TIME           the state's time name
#   env: ORACLE_ROOT (the oracle's of_env.sh and bin/dumpInterFoam; default /tmp/brae-oracle), FORMAT=ascii|binary (the decomposed fields; default the case's)
set -eo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CASE=${1:?usage: frozen_state.sh <case> <np> [method] [steps]}
NP=${2:?np}
METHOD=${3:-scotch}
STEPS=${4:-5}
for r in "${ORACLE_ROOT:-}" /tmp/brae-oracle; do
    [ -n "$r" ] && [ -f "$r/of_env.sh" ] && { ORACLE_ROOT=$r; break; }
done
OUT_ROOT=${OUT_ROOT:-/tmp/brae-frozen}
SUF=""; [ "$NP" = 1 ] || { SUF="_np$NP"; [ "$METHOD" = scotch ] || SUF="${SUF}_$METHOD"; }
OUT=$OUT_ROOT/${CASE}${SUF:-_np1}
[ -d "$HERE/cases/$CASE" ] || { echo "no case $CASE" >&2; exit 1; }
source "$ORACLE_ROOT/of_env.sh"
export OMP_NUM_THREADS=1 OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_btl_vader_single_copy_mechanism=none
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"
cp -r "$HERE/cases/$CASE" serial
( cd serial
  rm -f system/decomposeParDict.*
  blockMesh > ../log.blockMesh 2>&1
  topoSet > ../log.topoSet 2>&1
  cp -r 0.orig 0; mv 0/alpha.water.orig 0/alpha.water; rm -rf 0.orig
  setFields > ../log.setFields 2>&1 )
# The function objects the multi-rank gate compares are the case's own plus a set chosen so OpenFOAM and brae agree
# by construction: probes INSIDE cells (the case's own probes sit on cell-face planes, where polyMesh::findCell's
# octree and brae's nearest-centre search legitimately pick different one of two cells), one probe outside the mesh,
# patch and cellZone operations of every kind brae ports, and min/max of cells only (`internal true`: v2606's default
# also looks at the patch faces, v2412's -- the version brae mirrors -- does not).
python3 <<'PYEOF'
import re, sys
p = 'serial/system/controlDict'
s = open(p).read()
extra = """
    gate_probes
    {
        type probes; libs (sampling);
        fields (alpha.water U p_rgh);
        fixedLocations true;
        writeControl timeStep; writeInterval 1;
        probeLocations
        (
            (0.0371 0.0623 0.0127)
            (0.0821 0.0213 0.0471)
            (0.1379 0.0791 0.1113)
            (0.1913 0.0457 0.1871)
            (0.0153 0.0907 0.1511)
            (0.1043 0.0337 0.0961)
        );
    }
    gate_probes_out
    {
        type probes; libs (sampling);
        fields (alpha.water U p_rgh);
        fixedLocations true;
        writeControl timeStep; writeInterval 1;
        probeLocations
        (
            (0.0371 0.0623 0.0127)
            (0.4 0.05 0.1)
            (0.1043 0.0337 0.0961)
        );
    }
    gate_zone_avg
    {
        type volFieldValue; libs (fieldFunctionObjects);
        regionType cellZone; name zoneMid;
        operation volAverage; fields (alpha.water p_rgh);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_zone_sum
    {
        type volFieldValue; libs (fieldFunctionObjects);
        regionType cellZone; name zoneLow;
        operation sum; fields (alpha.water p_rgh);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_zone_min
    {
        type volFieldValue; libs (fieldFunctionObjects);
        regionType cellZone; name zoneLow;
        operation min; fields (alpha.water p_rgh);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_zone_max
    {
        type volFieldValue; libs (fieldFunctionObjects);
        regionType all;
        operation max; fields (alpha.water p_rgh);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_all_avg
    {
        type volFieldValue; libs (fieldFunctionObjects);
        regionType all;
        operation volAverage; fields (alpha.water);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_wall_avg
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        regionType patch; name tankWall;
        operation areaAverage; fields (alpha.water);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_wall_int
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        regionType patch; name tankWall;
        operation areaIntegrate; fields (alpha.water);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_wall_sum
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        regionType patch; name lid;
        operation sum; fields (alpha.water);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_wall_min
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        regionType patch; name tankWall;
        operation min; fields (alpha.water);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_wall_max
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        regionType patch; name tankWall;
        operation max; fields (alpha.water);
        writeFields false; log false; writeControl timeStep; writeInterval 1;
    }
    gate_minmax
    {
        type fieldMinMax; libs (fieldFunctionObjects);
        fields (alpha.water p_rgh); mode component; internal true;
        writeControl timeStep; writeInterval 1; log false;
    }
"""
# the case's own minMaxU gets `internal true` too
s = s.replace('fields (U); mode magnitude;', 'fields (U); mode magnitude; internal true;')
i = s.index('functions')
j = s.index('{', i) + 1
s = s[:j] + extra + s[j:]
open(p, 'w').write(s)
PYEOF
( cd serial; BRAE_STOP_ITER=$STEPS BRAE_WRITE_ITERS=$STEPS "$ORACLE_ROOT/bin/dumpInterFoam" > ../log.serial 2>&1 )
TIME=$(ls serial | grep -E '^[0-9][0-9.e+-]*$' | grep -v '^0$' | sort -g | tail -n 1)
echo "$TIME" > TIME
# the state alone (the dumps dumpInterFoam may have written are not part of it)
mkdir dec
cp -r serial/constant serial/system dec/
# the references made from the state (postProcess) carry 17 digits -- bitwise comparisons. (OpenFOAM's own run stops at its
# last write with `ill defined primitiveEntry ... min(U)` when a probe is in no cell and writePrecision is 17, so the runs
# keep the case's 12: their .dat rows (dec/inrun, dec_par/inrun) are good to 12 digits.)
cp dec/system/controlDict dec/system/controlDict.run    # the case's own: what the runs used (12 digits, the probe outside the mesh)
python3 - <<'PYEOF'
# ...and the controlDict of the references made FROM the state: 17 digits and no probe outside the mesh (OpenFOAM stops
# writing its state with the second)
import re
p = 'dec/system/controlDict'
s = open(p).read()
s = re.sub(r'\n    gate_probes_out\n    \{.*?\n    \}\n', '\n', s, flags=re.S)
s = re.sub(r'(?m)^writePrecision .*', 'writePrecision  17;', s)
open(p, 'w').write(s)
PYEOF
cp -r serial/postProcessing dec/inrun       # OpenFOAM SERIAL's own function objects during the run: the last row is the state
mkdir dec/"$TIME"
for f in alpha.water U p_rgh p phi alphaPhi0.water uniform; do cp -r serial/"$TIME"/$f dec/"$TIME"/ 2>/dev/null || true; done
if [ "$NP" = 1 ]; then
    ( cd dec; postProcess -latestTime -fields '(alpha.water U p_rgh)' > ../log.postProcess 2>&1 )
else
    DICT=$HERE/cases/$CASE/system/decomposeParDict.$METHOD$NP
    [ -f "$DICT" ] || { echo "no $DICT" >&2; exit 1; }
    cp "$DICT" dec/system/decomposeParDict
    [ -n "$FORMAT" ] && sed -i "s/^writeFormat .*/writeFormat $FORMAT;/" dec/system/controlDict
    ( cd dec
      decomposePar -latestTime -force -copyUniform > ../log.decompose 2>&1
      mpirun --bind-to none -np "$NP" postProcess -parallel -latestTime -fields '(alpha.water U p_rgh)' > ../log.postProcess 2>&1
      mkdir -p ../recon; cp -r constant system ../recon/; for r in $(seq 0 $((NP-1))); do mkdir -p ../recon/processor$r; cp -r processor$r/constant ../recon/processor$r/; cp -r processor$r/"$TIME" ../recon/processor$r/; done
      cd ../recon && reconstructPar -latestTime > ../log.reconstruct 2>&1 )
fi
if [ "$NP" != 1 ]; then
    # ...and the state of OpenFOAM's OWN -parallel run of the same steps, whose processor patch values are the
    # coupled ones a -parallel solver leaves (decomposePar's are the cell values): dec_par/ holds its last time
    cp -r serial/constant serial/system par_run 2>/dev/null || { mkdir par_run; cp -r serial/constant serial/system par_run/; }
    mkdir -p par_run/0; cp serial/0/alpha.water serial/0/U serial/0/p_rgh serial/0/p par_run/0/
    cp dec/system/decomposeParDict par_run/system/
    [ -n "$FORMAT" ] && sed -i "s/^writeFormat .*/writeFormat $FORMAT;/" par_run/system/controlDict
    ( cd par_run
      decomposePar -force > ../log.decompose.par 2>&1
      BRAE_STOP_ITER=$STEPS BRAE_WRITE_ITERS=$STEPS mpirun --bind-to none -np "$NP" "$ORACLE_ROOT/bin/dumpInterFoam" -parallel > ../log.par 2>&1 )
    PT=$(ls par_run/processor0 | grep -E '^[0-9][0-9.e+-]*$' | grep -v '^0$' | sort -g | tail -n 1)
    echo "$PT" > TIME_PAR
    mkdir dec_par; cp -r dec/system dec_par/
    for r in $(seq 0 $((NP-1))); do mkdir -p dec_par/processor$r; cp -r par_run/processor$r/constant dec_par/processor$r/; mkdir dec_par/processor$r/"$PT"
        for f in alpha.water U p_rgh p phi alphaPhi0.water uniform; do cp -r par_run/processor$r/"$PT"/$f dec_par/processor$r/"$PT"/; done; done
    ( cd dec_par; mpirun --bind-to none -np "$NP" postProcess -parallel -latestTime -fields '(alpha.water U p_rgh)' > ../log.postProcess.par 2>&1 )
    cp -r par_run/postProcessing dec_par/inrun   # OpenFOAM -parallel's own function objects during its run (last row = the state)
    echo "  -parallel state: $OUT/dec_par   time $PT"
fi
echo "frozen state: $OUT   time $TIME   np $NP"
