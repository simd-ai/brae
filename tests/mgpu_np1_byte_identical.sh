#!/bin/bash
# First multi-GPU gate: a ONE-domain decomposition run through `-parallel` is byte-identical to the serial run, and
# np = 2 / 4 decompositions are read (per-rank patch table) and then refused cleanly.
#   tests/mgpu_np1_byte_identical.sh <brae_interFoam> [scratch dir]
# Case: tools/mgpu_oracle/cases/tank (OpenFOAM v2606 blockMesh/topoSet/setFields/decomposePar; skipped, exit 77,
# when /usr/lib/openfoam/openfoam2606 is absent). Needs one GPU.
set -eo pipefail
BRAE=${1:?usage: $0 <brae_interFoam> [scratch]}
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
W=${2:-$(mktemp -d /tmp/mgpu_np1.XXXXXX)}
OF=/usr/lib/openfoam/openfoam2606/etc/bashrc
[ -f "$OF" ] || { echo "SKIP: no OpenFOAM v2606"; exit 77; }
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$W"; cd "$W"; rm -rf mesh serial np1 np2 np4
cp -r "$HERE/tools/mgpu_oracle/cases/tank" mesh
( set +eu; source "$OF" >/dev/null 2>&1
  cd mesh; rm -f system/decomposeParDict.*
  blockMesh >/dev/null 2>&1; topoSet >/dev/null 2>&1
  cp -r 0.orig 0; mv 0/alpha.water.orig 0/alpha.water; rm -rf 0.orig; setFields >/dev/null 2>&1 )
D=$HERE/tools/mgpu_oracle/cases/tank/system
mk() { # <n> <dict>
    cp -r mesh np$1; cp "$2" np$1/system/decomposeParDict
    ( set +eu; source "$OF" >/dev/null 2>&1; cd np$1; decomposePar -force >/dev/null 2>&1 ); }
cp "$D/decomposeParDict.simple4" one.dict
sed -i 's/numberOfSubdomains.*/numberOfSubdomains 1;/; s/n (1 2 2)/n (1 1 1)/' one.dict
mk 1 one.dict; mk 2 "$D/decomposeParDict.simple2"; mk 4 "$D/decomposeParDict.simple4"
cp -r mesh serial
for d in serial np1; do
    sed -i 's/^writeControl .*/writeControl runTime;/; s/^writeInterval .*/writeInterval 0.01;/; s/^endTime .*/endTime 0.01;/' $d/system/controlDict
done
"$BRAE" -case serial -device > log.serial 2>&1
"$BRAE" -case np1 -parallel -rank 0 -nprocs 1 -device > log.np1 2>&1
T=$(ls -d serial/[0-9]* | xargs -n1 basename | sort -g | tail -n 1)
[ "$T" != 0 ] || { echo "FAIL: serial run wrote no time directory"; exit 1; }
n=0
for f in $(ls serial/$T); do
    if [ -d serial/$T/$f ]; then diff -r serial/$T/$f np1/processor0/$T/$f >/dev/null
    else cmp -s serial/$T/$f np1/processor0/$T/$f; fi || { echo "FAIL: $T/$f differs between serial and -parallel np=1"; exit 1; }
    n=$((n+1))
done
diff -r serial/postProcessing np1/postProcessing >/dev/null || { echo "FAIL: postProcessing differs"; exit 1; }
echo "np=1: $n entries of $T and postProcessing byte-identical"
for N in 2 4; do for r in $(seq 0 $((N-1))); do
    set +e
    if command -v mpirun >/dev/null 2>&1; then
        mpirun --bind-to none --oversubscribe -np "$N" "$BRAE" -case np$N -parallel -device > log.np${N}_$r 2>&1
    else
        "$BRAE" -case np$N -parallel -rank $r -nprocs $N -device > log.np${N}_$r 2>&1
    fi
    rc=$?; set -e
    [ $rc = 0 ] || { echo "FAIL: np=$N rank $r exit $rc, expected the processor run to start"; exit 1; }
    grep -q "processor patch(es)" log.np${N}_$r || { echo "FAIL: np=$N rank $r: no per-rank table"; exit 1; }
    grep -q "^   t = .*\[device\]" log.np${N}_$r \
        || { echo "FAIL: np=$N rank $r did not reach a numerical stage"; exit 1; }
    break
done; done
echo "np=2, np=4: per-rank tables printed, processor path starts"
echo PASS
