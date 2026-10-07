#!/usr/bin/env bash
# The write gate: `writeFormat binary`. A field's scalar and vector lists are written as their own bytes, laid
# out as OpenFOAM's binary stream lays them (UList::writeList's binary branch: a newline, the count, a newline,
# the bytes between parentheses), the file labelled `format binary`. brae wrote ascii at writePrecision whatever
# the entry said until 2026-10-07; five shipped tutorials ask for binary.
# ORACLE: OpenFOAM's own reader. foamFormatConvert reads brae's binary directory and writes it back as text at
# 17 digits, and that has to be brae's own ascii run at 17 digits, EXACTLY -- a double printed at 17 digits is
# the double. And OpenFOAM restarts from the directory as it does from its own binary one.
# Three checks, on laminar/damBreak, two steps: (1) [host] [device] the converter's rewrite of brae's binary
# directory is brae's ascii run, every file exact; and each field's FoamFile header and the bytes from
# `internalField` to its list's opening parenthesis are OpenFOAM's own, byte for byte (a file's SIZE is not
# held: a wall's flux is `uniform 0` in OpenFOAM's file and four residues of 1e-37 in the device's).
# (2) OpenFOAM restarted from brae's directory starts as
# from its own (the first p_rgh residual), and brae restarted from its binary directory writes the bytes it
# writes from its ascii one. (3) CONTROL: BRAE_CONTROL_WRITE_BINARY_SHIFTED=1 writes every list's bytes one
# value late -- right sizes, wrong values -- and the converter's rewrite is no longer the ascii run.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
o="$W/bn_of"
stage "$LAM" "$o" 0.002 timeStep 2 0 writeFormat=binary deltaT=0.001 adjustTimeStep=no > "$W/bn_stage.txt" 2>&1 \
    || { say "laminar/damBreak did not stage" FAIL; finish "writeFormat binary"; }
runof "$o"
# arm <name> <host|device> <binary|ascii> [env...]: brae on a copy of the staged case
arm()
{
    local e="$W/bn_$1" loop="$2" format="$3"
    shift 3
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i "s/^writeFormat .*/writeFormat     $format;/" "$e/system/controlDict"
    runbrae "$e" "$loop" "$@"
}
# converted <name>: OpenFOAM's rewrite of that arm's time directories as text, in bn_<name>_text
converted()
{
    local e="$W/bn_$1_text"
    rm -rf "${e:?}"
    cp -r "$W/bn_$1" "$e"
    sed -i 's/^writeFormat .*/writeFormat     ascii;/' "$e/system/controlDict"
    ( cd "$e" && foamFormatConvert > log.convert 2>&1 )
}
# exact <a> <b>: "<files compared> <files that are not exactly equal>" over OpenFOAM's time directories
exact()
{
    python3 "$CMP" "$W/bn_$1" "$W/bn_$2" $(timedirs "$o") 2> /dev/null | python3 -c "
import json, sys
lines = [l for l in sys.stdin.read().splitlines() if l.startswith('RESULT ')]
if not lines:
    print('0 1')
else:
    files = json.loads(lines[-1][7:])['files']
    print(len(files), sum(1 for v in files.values() if v['rel'] != 0))"
}
arm host host binary BRAE_X=1
arm device device binary BRAE_X=1
arm host_ascii host ascii BRAE_X=1
arm device_ascii device ascii BRAE_X=1
arm shifted device binary BRAE_CONTROL_WRITE_BINARY_SHIFTED=1
ok=1
for v in host device; do
    converted $v || ok=0
done
read -r nh bh <<< "$(exact host_ascii host_text)"
read -r nd bd <<< "$(exact device_ascii device_text)"
t=$(timedirs "$o" | awk '{print $NF}')
# layout <dir>: "<fields whose header and list opening are OpenFOAM's bytes> <fields OpenFOAM wrote binary>"
layout()
{
    python3 - "$o/$t" "$1/$t" <<'PY'
import os, sys
of, br = sys.argv[1:3]
same = total = 0
for name in sorted(os.listdir(of)):
    p = os.path.join(of, name)
    if not os.path.isfile(p):
        continue
    a = open(p, 'rb').read()
    if b'format      binary;' not in a or b'internalField   nonuniform' not in a:
        continue
    total += 1
    q = os.path.join(br, name)
    if not os.path.isfile(q):
        continue
    b = open(q, 'rb').read()
    def parts(x):
        h = x[x.index(b'FoamFile'):x.index(b'}', x.index(b'FoamFile')) + 1]
        i = x.find(b'internalField')
        return h, x[i:x.index(b'(', i) + 1] if i >= 0 else b''
    same += 1 if parts(a) == parts(b) else 0
print(same, total)
PY
}
read -r lh th <<< "$(layout "$W/bn_host")"
read -r ld td <<< "$(layout "$W/bn_device")"
what="[host] [device] OpenFOAM's converter reads brae's binary: $bh of $nh and $bd of $nd files differ from"
what="$what brae's ascii at 17 digits; $lh and $ld of OpenFOAM's $td binary fields laid out as its own"
[ $ok = 1 ] && [ "$nh" -ge 6 ] && [ "$bh" = 0 ] && [ "$nd" -ge 6 ] && [ "$bd" = 0 ] && [ "$td" -ge 5 ] \
    && [ "$lh" = "$th" ] && [ "$ld" = "$td" ] && grep -q "write: writeFormat binary" "$W/bn_device/log.brae" \
    && say "$what" ok || say "$what" FAIL
# restart <dir> <from>: OpenFOAM or brae, two more steps from <from>'s last time
restart()
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$o/constant" "$o/system" "$1/"
    cp -r "$2/$t" "$1/"
    sed -i 's/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         0.004;/' "$1/system/controlDict"
}
first() { grep -a "Solving for p_rgh" "$1" | head -1 | sed -E 's/.*Initial residual = ([^,]+),.*/\1/'; }
restart "$W/bn_rs_of" "$o"
restart "$W/bn_rs_brae" "$W/bn_device"
( cd "$W/bn_rs_of" && interFoam > log.interFoam 2>&1 )
( cd "$W/bn_rs_brae" && interFoam > log.interFoam 2>&1 ); rc=$?
gap=$(python3 -c "
a, b = float('$(first "$W/bn_rs_of/log.interFoam")' or 0), float('$(first "$W/bn_rs_brae/log.interFoam")' or 1)
print('%.1e' % (abs(a - b)/abs(a) if a else 1))")
restart "$W/bn_rb_binary" "$W/bn_device"
restart "$W/bn_rb_ascii" "$W/bn_device_ascii"
sed -i 's/^writeFormat .*/writeFormat     ascii;/' "$W/bn_rb_binary/system/controlDict" \
    "$W/bn_rb_ascii/system/controlDict"
runbrae "$W/bn_rb_binary" device
runbrae "$W/bn_rb_ascii" device
n=0
for f in $(cd "$W/bn_rb_ascii/0.004" && find . -type f | sort); do
    cmp -s "$W/bn_rb_ascii/0.004/$f" "$W/bn_rb_binary/0.004/$f" || n=$((n + 1))
done
what="OpenFOAM restarts from brae's binary directory (its first p_rgh residual $gap from its own restart's);"
what="$what brae restarted from binary and from ascii: $n files differ"
[ $rc = 0 ] && grep -q "^End" "$W/bn_rs_brae/log.interFoam" && within "$gap" 1e-9 && [ "$n" = 0 ] \
    && [ -d "$W/bn_rb_binary/0.004" ] && say "$what" ok || say "$what" FAIL
converted shifted
read -r ns bs <<< "$(exact device_ascii shifted_text)"
what="CONTROL  every list's bytes one value late: $bs of $ns files of the converter's rewrite differ"
[ "$bs" -ge 3 ] && say "$what" ok || say "$what" FAIL
finish "writeFormat binary: fields as their own bytes, read by OpenFOAM, exact"
