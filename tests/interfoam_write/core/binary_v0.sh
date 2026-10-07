#!/usr/bin/env bash
# The write gate: `writeFormat binary` and V0, the old cell volumes a CrankNicolson run on a moving mesh
# writes -- a `volScalarField::Internal`, which OpenFOAM reads back whenever the file is there
# (fvMesh.C:306-318) in the format its header names. The writer chose a file's format by a class name ENDING
# in `Field`; this one does not, so its header said ascii above a list written as bytes.
# core/binary.sh has the format and the oracle. Three checks on laminar/waves/waveMakerFlap under
# `CrankNicolson 0.9`, two steps of 0.01 on the GPU loop: (1) V0 is OpenFOAM's own layout: its size, its
# bytes up to the list's opening parenthesis and its bytes after the closing one are those of the file
# OpenFOAM writes for the same case. (2) Its doubles are the values of brae's ascii run at 17 digits,
# exactly, and OpenFOAM's within 2e-13 (MEASURED 2026-10-07: 56,000 doubles, 1.7e-14). (3) CONTROL:
# BRAE_CONTROL_WRITE_V0_ASCII_HEADER=1 names the format as the writer did, and (1) fails on the header.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
o="$W/bv_of"
stage "$TUT/multiphase/interFoam/laminar/waves/waveMakerFlap" "$o" 0.02 timeStep 2 0 writeFormat=binary \
    deltaT=0.01 adjustTimeStep=no > "$W/bv_stage.txt" 2>&1 \
    || { say "waves/waveMakerFlap did not stage" FAIL; finish "writeFormat binary: V0"; }
# ...and one alpha sub-cycle: alphaEqn.H:27-32 stops on sub-cycling under CrankNicolson, and the tutorial names
# 3 (and 2 again under PIMPLE)
python3 - "$o/system" <<'PY' || { say "the CrankNicolson staging did not apply" FAIL; finish "writeFormat binary: V0"; }
import re, sys
d = sys.argv[1]
t = open(d + '/fvSchemes').read()
t, n = re.subn(r'(ddtSchemes\s*\{\s*default\s+)Euler;', r'\1CrankNicolson 0.9;', t)
assert n == 1
open(d + '/fvSchemes', 'w').write(t)
t = open(d + '/fvSolution').read()
t, n = re.subn(r'nAlphaSubCycles\s+\d+;', 'nAlphaSubCycles 1;', t)
assert n >= 1
open(d + '/fvSolution', 'w').write(t)
PY
runof "$o"
t=$(timedirs "$o" | awk '{print $NF}')
# arm <name> <binary|ascii> [env...]: brae's GPU loop on a copy of the staged case
arm()
{
    local e="$W/bv_$1" format="$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i "s/^writeFormat .*/writeFormat     $format;/" "$e/system/controlDict"
    runbrae "$e" device "$@"
}
# layout <OpenFOAM's V0> <brae's V0>: "<size> <size> <same|other> <the format brae's header names>"
layout()
{
    python3 - "$1" "$2" <<'PY'
import re, sys
a, b = (open(p, 'rb').read() for p in sys.argv[1:3])
def ends(x):
    i = x.index(b'(', x.index(b'// * * *'))
    return x[x.index(b'FoamFile'):i + 1], x[x.rindex(b')'):]
try:
    same = 'same' if ends(a) == ends(b) else 'other'
except ValueError:
    same = 'other'
m = re.search(rb'format\s+(\w+);', b)
print('%d %d %s %s' % (len(a), len(b), same, m.group(1).decode() if m else '-'))
PY
}
arm binary binary BRAE_X=1
arm ascii ascii BRAE_X=1
arm asciiheader binary BRAE_CONTROL_WRITE_V0_ASCII_HEADER=1
[ -f "$o/$t/V0" ] && [ -f "$W/bv_binary/$t/V0" ] \
    || { say "V0 was not written at $t" FAIL; finish "writeFormat binary: V0"; }
read -r sa sb se fm <<< "$(layout "$o/$t/V0" "$W/bv_binary/$t/V0")"
what="[device] V0 is OpenFOAM's layout: sizes $sa and $sb, the ends $se, the header's format $fm"
[ -n "$sa" ] && [ "$sa" = "$sb" ] && [ "$se" = same ] && [ "$fm" = binary ] && say "$what" ok || say "$what" FAIL
val=$(python3 - "$o/$t/V0" "$W/bv_binary/$t/V0" "$W/bv_ascii/$t/V0" <<'PY'
import re, struct, sys
def doubles(path):
    x = open(path, 'rb').read()
    m = re.search(rb'value\s+nonuniform List<scalar>\s*(\d+)\s*\(', x)
    n = int(m.group(1))
    return list(struct.unpack('<%dd' % n, x[m.end():m.end() + 8*n]))
def text(path):
    x = open(path).read()
    m = re.search(r'value\s+nonuniform List<scalar>\s*(\d+)\s*\((.*?)\)\s*;', x, re.S)
    return [float(v) for v in m.group(2).split()]
try:
    of, b, a = doubles(sys.argv[1]), doubles(sys.argv[2]), text(sys.argv[3])
    ok = len(of) == len(b) == len(a) and len(b) > 0
    print('%d %d %.1e' % (len(b) if ok else 0, sum(1 for u, v in zip(a, b) if u != v) if ok else -1,
                          max(abs(u - v) for u, v in zip(of, b))/max(abs(u) for u in of) if ok else float('nan')))
except Exception:
    print('0 -1 nan')
PY
)
read -r nv nd rel <<< "$val"
B=2e-13
what="[device] V0's $nv doubles: $nd differ from brae's ascii run at 17 digits; $rel from OpenFOAM's (bound $B)"
[ "$nv" -gt 100 ] && [ "$nd" = 0 ] && within "$rel" "$B" && say "$what" ok || say "$what" FAIL
read -r sa sb se fm <<< "$(layout "$o/$t/V0" "$W/bv_asciiheader/$t/V0")"
what="CONTROL  the format named as the writer did: the header's format $fm, the ends $se"
[ "$fm" = ascii ] && [ "$se" = other ] && say "$what" ok || say "$what" FAIL
finish "writeFormat binary: V0's header names the format its list is written in"
