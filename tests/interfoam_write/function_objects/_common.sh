#!/usr/bin/env bash
# Shared by the function-object gates. Sourced after ../lib.sh.
# THE ORACLE is what real OpenFOAM's own function objects wrote for the same case: its postProcessing/ files
# and the state dictionary in its time directories.
#
# fo_stage <dir> <source case> <endTime> <writeControl> <writeInterval> <file holding a `functions { ... }`
# entry> [key=value ...]: the case staged as ../lib.sh stages it, with that entry, run by OpenFOAM (cached).
# FO_PIN=1 in the environment pins every solve first. Returns 1 with the premise that failed said.
fo_stage()
{
    local o="$1" c="$1/system/controlDict" fns="$6"
    stage "$2" "$o" "$3" "$4" "$5" 0 "${@:7}" > "$o.stage.txt" 2>&1 || { say "the case staged" FAIL; return 1; }
    python3 - "$c" "$fns" <<'PY' || { say "PREMISE  the functions entry was staged" FAIL; return 1; }
import sys
p, fo = sys.argv[1:3]
s = open(p).read()
assert s.count('functions\n{\n}') == 1
open(p, 'w').write(s.replace('functions\n{\n}', open(fo).read().strip(), 1))
PY
    if [ -n "${FO_PIN:-}" ]; then
        # FO_PIN=1: every solve pinned (tolerance 1e-13, relTol 0), so that two codes' runs are as far apart
        # as their arithmetic and not as their linear solvers' stopping points
        sed -i -E 's/(tolerance +)[^;]+;/\11e-13;/; s/(relTol +)[^;]+;/\10;/' "$o/system/fvSolution"
    fi
    runof "$o"
    ! grep -q "loading function object" "$o/log.interFoam" \
        || { say "PREMISE  OpenFOAM loaded every function object" FAIL; return 1; }
}
# fo_arm <staged dir> <run dir> <host|device> [env...]: brae's own run of the staged case
fo_arm()
{
    local o="$1" e="$2" loop="$3"
    shift 3
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    runbrae "$e" "$loop" "$@"
}
# fo_file <OpenFOAM's file> <brae's file> [column]: one probes file against OpenFOAM's --
#   "<OpenFOAM's rows> <brae's rows> <head lines that differ> <time entries that differ as text> <worst gap>"
# The head is every `#` line, byte for byte. A value's gap is relative to its own size, or to a thousandth of
# the largest of the file where it is smaller than that (a pressure crossing zero, an alpha of 1e-30). With
# a column, only that probe's values are compared (0 is the first probe).
fo_file()
{
    python3 - "$1" "$2" "${3:-}" <<'PY'
import os, re, sys
a, b, col = sys.argv[1:4]
if not os.path.isfile(a) or not os.path.isfile(b):
    print('- - - - -')
    sys.exit(0)
def read(p):
    head, rows = [], []
    for line in open(p, 'rb').read().split(b'\n'):
        if line.startswith(b'#'):
            head.append(line)
        elif line.strip():
            t = line.decode().split(None, 1)
            vals = re.findall(r'\(([^)]*)\)|(\S+)', t[1] if len(t) > 1 else '')
            rows.append((t[0], [[float(x) for x in (v[0].split() if v[0] else [v[1]])] for v in vals]))
    return head, rows
ha, ra = read(a)
hb, rb = read(b)
headDiff = sum(1 for x, y in zip(ha, hb) if x != y) + abs(len(ha) - len(hb))
n = min(len(ra), len(rb))
timeDiff = sum(1 for i in range(n) if ra[i][0] != rb[i][0])
big = max([abs(x) for _, vs in ra for v in vs for x in v if abs(x) < 1e299] or [1.0])
worst = 0.0
for i in range(n):
    va, vb = ra[i][1], rb[i][1]
    if len(va) != len(vb):
        worst = 1.0
        continue
    for k, (p, q) in enumerate(zip(va, vb)):
        if col != '' and k != int(col):
            continue
        for x, y in zip(p, q):
            if x == y:
                continue
            worst = max(worst, abs(x - y)/max(abs(x), abs(y), 1e-3*big))
print(len(ra), len(rb), headDiff, timeDiff, '%.1e' % worst)
PY
}
# fo_state <OpenFOAM's case> <brae's case> <time>: the state dictionary of that time directory --
#   "<lines that differ with every number masked> <numbers> <worst gap of a number, relative to its own size>"
fo_state()
{
    python3 - "$1/$3/uniform/functionObjects/functionObjectProperties" \
        "$2/$3/uniform/functionObjects/functionObjectProperties" <<'PY'
import os, re, sys
a, b = sys.argv[1:3]
if not os.path.isfile(a) or not os.path.isfile(b):
    print('- - -')
    sys.exit(0)
num = re.compile(r'(?<![\w.])-?\d+\.?\d*(?:[eE][-+]?\d+)?(?![\w.])')
def body(p):
    s = open(p).read()
    return s[s.index('// * * *'):].split('\n')
la, lb = body(a), body(b)
textDiff = sum(1 for x, y in zip(la, lb) if num.sub('N', x) != num.sub('N', y)) + abs(len(la) - len(lb))
worst, count = 0.0, 0
for x, y in zip(la, lb):
    for p, q in zip(num.findall(x), num.findall(y)):
        count += 1
        p, q = float(p), float(q)
        if p != q:
            worst = max(worst, abs(p - q)/max(abs(p), abs(q)))
print(textDiff, count, '%.1e' % worst)
PY
}
