#!/usr/bin/env bash
# The write gate: `writeFormat binary` on a mesh that moves -- polyMesh/points, a bare vectorField, and the
# point and face-vector fields a moving mesh writes (pointDisplacement, cellDisplacement, Uf, meshPhi).
# core/binary.sh has the format and the oracle: OpenFOAM's own converter reads brae's binary directory, and its
# rewrite at 17 digits has to be brae's ascii run exactly.
# Three checks, on laminar/waves/waveMakerFlap, two steps of 0.01 on the GPU loop: (1) the converter's rewrite
# is the ascii run, every file exact, points among them. (2) the points file is OpenFOAM's own layout: its
# size, its bytes up to the list's opening parenthesis and its bytes after the closing one are those of the
# file OpenFOAM writes for the same mesh (the coordinates between them are each run's own). (3) CONTROL:
# BRAE_CONTROL_WRITE_BINARY_SHIFTED=1, every list's bytes one value late: the rewrite differs, points too.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
o="$W/bm_of"
stage "$TUT/multiphase/interFoam/laminar/waves/waveMakerFlap" "$o" 0.02 timeStep 2 0 writeFormat=binary \
    deltaT=0.01 adjustTimeStep=no > "$W/bm_stage.txt" 2>&1 \
    || { say "waves/waveMakerFlap did not stage" FAIL; finish "writeFormat binary on a moving mesh"; }
runof "$o"
t=$(timedirs "$o" | awk '{print $NF}')
# arm <name> <binary|ascii> [env...]: brae's GPU loop on a copy of the staged case
arm()
{
    local e="$W/bm_$1" format="$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i "s/^writeFormat .*/writeFormat     $format;/" "$e/system/controlDict"
    runbrae "$e" device "$@"
}
# converted <name>: OpenFOAM's rewrite of that arm's time directories as text, in bm_<name>_text
converted()
{
    local e="$W/bm_$1_text"
    rm -rf "${e:?}"
    cp -r "$W/bm_$1" "$e"
    sed -i 's/^writeFormat .*/writeFormat     ascii;/' "$e/system/controlDict"
    ( cd "$e" && foamFormatConvert > log.convert 2>&1 )
}
# exact <a> <b>: "<files compared> <files not exactly equal> <whether polyMesh/points is among the compared>"
exact()
{
    python3 "$CMP" "$W/bm_$1" "$W/bm_$2" $(timedirs "$o") 2> /dev/null | python3 -c "
import json, sys
lines = [l for l in sys.stdin.read().splitlines() if l.startswith('RESULT ')]
if not lines:
    print('0 1 no')
else:
    files = json.loads(lines[-1][7:])['files']
    points = [k for k in files if k.endswith('polyMesh/points')]
    wrong = [k for k, v in files.items() if v['rel'] != 0]
    print(len(files), len(wrong), ('wrong' if set(points) & set(wrong) else 'exact') if points else 'no')"
}
arm binary binary BRAE_X=1
arm ascii ascii BRAE_X=1
arm shifted binary BRAE_CONTROL_WRITE_BINARY_SHIFTED=1
converted binary; rc=$?
read -r n bad pts <<< "$(exact ascii binary_text)"
what="[device] OpenFOAM's converter reads a moving mesh's binary directory: $bad of $n files differ from brae's"
what="$what ascii at 17 digits (points: $pts)"
[ $rc = 0 ] && [ "$n" -ge 12 ] && [ "$bad" = 0 ] && [ "$pts" = exact ] \
    && grep -q "write: writeFormat binary" "$W/bm_binary/log.brae" && say "$what" ok || say "$what" FAIL
lay=$(python3 - "$o/$t/polyMesh/points" "$W/bm_binary/$t/polyMesh/points" <<'PY'
import sys
a, b = (open(p, 'rb').read() for p in sys.argv[1:3])
def ends(x):
    i = x.index(b'(', x.index(b'// * * *'))
    return x[x.index(b'FoamFile'):i + 1], x[x.rindex(b')'):]
print('%d %d %s' % (len(a), len(b), 'same' if ends(a) == ends(b) else 'other'))
PY
)
what="[device] polyMesh/points is OpenFOAM's layout: sizes and ends (OpenFOAM's, brae's, the ends) $lay"
read -r sa sb se <<< "$lay"
[ -n "$sa" ] && [ "$sa" = "$sb" ] && [ "$se" = same ] && say "$what" ok || say "$what" FAIL
converted shifted
read -r n bad pts <<< "$(exact ascii shifted_text)"
what="CONTROL  every list's bytes one value late: $bad of $n files of the converter's rewrite differ (points: $pts)"
[ "$bad" -ge 6 ] && [ "$pts" = wrong ] && say "$what" ok || say "$what" FAIL
finish "writeFormat binary on a moving mesh: the points and the point fields as their own bytes"
