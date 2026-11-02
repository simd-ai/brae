#!/usr/bin/env bash
# linearUpwind honoured on k, omega and epsilon (BRAE_SCALAR_LINEARUPWIND=1) on a 2-D case from cold, every iteration against real OpenFOAM.
# The legacy driver runs these upwind by default, with a warning, because honoured they diverged on pitzDaily "coupled k-omega" (gpuSimpleFoam.cu). MEASURED: omega
# 0.16% out in the outlet column and 23% out in one inlet cell after iteration 1 with U and p at 1e-13, 3.6% out by iteration 40 (job 4844884), and blow-ups to omega
# 1e14 to 1e27 (jobs 4835375, 4845077). Neither is the scheme: linearUpwind's own numbers are exact once these two are.
#   the average in Foam::bound counted a 2-D mesh's `empty` faces, whose patches have no faces in OpenFOAM: a bounded cell 816.4 against 1059.2. This is the blow-up: reverted
#     alone, the run reaches omega 2.5e27 by iteration 1628 F1 off and 1.6e20 by 838 F1 on, and with it in, all six twins of both reach 10000 (job 4845077).
#   the equation's gradient read a live zeroGradient face beside a wall cell where linearUpwind.C:94 reads the stored one: the outlet column.
# ARMS: pitzDailySST with F1 on (BRAE_SST_F1_PATCH=1, without it the inlet's F1 is its own gap at iteration 2) and pitzDaily kEpsilon, tight, cold, 40 iterations.
#   a wall function's face left out of the relax of the wall row: the fields exact, the residual of omega 0.16% out.
# THE BOUNDS: 1e-9 on U, p, k, the second field and nut at every iteration, and 2e-5 on every initial residual, which brae prints to six figures. MEASURED: see the output, worst field 6.6e-12 on SST.
# kEpsilon does not bound in those 40 iterations (none in either code), so its `start` arm bounds: a negative cell in epsilon's start field, bound at the constructor (kEpsilon.C:182), one iteration.
# PREMISE, asserted: both codes bound where the arm's control needs it, the same number of lines.
# CONTROLS, asserted red: BRAE_CONTROL_BOUND_EMPTY=1 and BRAE_CONTROL_TURB_GRAD_LIVE=1 at the same bound, each alone.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
# stage <dir> <fixture> <second field> <end> <write>
stage() {
    local c=$1 src=$ROOT/validation/$2
    rm -rf "$c"; mkdir -p "$c"
    cp -r "$src/constant" "$src/system" "$src/0" "$c/"
    rm -f "$c"/constant/polyMesh/.brae_*
    python3 - "$c" "$3" "$4" "$5" <<'PYEOF'
import re, sys
c, sec, end, wr = sys.argv[1:]
p = c + '/system/fvSchemes'; s = open(p).read()
s, a = re.subn(r'div\(phi,U\)\s+[^;]*;', 'div(phi,U) bounded Gauss linearUpwind grad(U);', s)
s, b = re.subn(r'div\(phi,k\)\s+[^;]*;', 'div(phi,k) bounded Gauss linearUpwind grad(k);', s)
s, d = re.subn(r'div\(phi,%s\)\s+[^;]*;' % sec, 'div(phi,%s) bounded Gauss linearUpwind grad(%s);' % (sec, sec), s)
assert a == 1 and b == 1 and d == 1, (a, b, d)
s = re.sub(r'\n\s*turbulence\s+bounded[^;]*;', '', s)
s = re.sub(r'div\(phi,%s\)[^;]*;' % ('epsilon' if sec == 'omega' else 'omega'), '', s)
open(p, 'w').write(s)
p = c + '/system/fvSolution'; s = open(p).read()
s, n = re.subn(r'solvers\s*\{.*?\}\s*\n\s*(?=SIMPLE)', '''solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-13; relTol 0; maxIter 5000; }
    Phi { $p; }
    "(U|k|omega|epsilon)" { solver smoothSolver; smoother GaussSeidel; nSweeps 1; tolerance 1e-13; relTol 0; maxIter 5000; }
}
''', s, count=1, flags=re.S)
assert n == 1
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
p = c + '/system/controlDict'; s = open(p).read()
for k, v in (('startFrom', 'startTime'), ('startTime', '0'), ('endTime', end), ('writeInterval', wr), ('writeFormat', 'ascii'), ('writePrecision', '17')):
    s, n = re.subn(r'(?m)(^|\s)%s\s+[^;]*;' % k, r'\1%s %s;' % (k, v), s)
    if n == 0:
        s += '\n%s %s;\n' % (k, v)
s = re.sub(r'\nlibs\s[^;]*;', '\n', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(p, 'w').write(s)
PYEOF
}
br() { local c=$1; shift; (cd "$c" && env BRAE_AMG_CACHE=0 BRAE_SIMPLE_CASE_SOLVER=1 BRAE_SCALAR_LINEARUPWIND=1 "$@" "$BRAE" -case . > log 2>&1 < /dev/null; echo $? > rc); }
cat > "$W/cmp.py" <<'PYEOF'
import re, sys
import numpy as np
of, br, sec, want, end = sys.argv[1:6]
BOUND = 1e-9
RESIDUAL = 2e-5
def rd(fn):
    s = open(fn, errors='replace').read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', s)
    if not m:
        return None
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    b = s[m.end():]
    b = b[:b.index('\n)')]
    return np.array(b.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)
worst = 0.0
where = ''
for t in range(1, int(end) + 1):
    for f in ('U', 'p', 'k', sec, 'nut'):
        try:
            x, y = rd('%s/%d/%s' % (of, t, f)), rd('%s/%d/%s' % (br, t, f))
        except FileNotFoundError:
            print('        no %s at %d' % (f, t))
            raise SystemExit(1)
        if x is None or y is None:
            continue
        e = np.linalg.norm(y - x) / np.linalg.norm(x)
        if e > worst:
            worst, where = e, '%s at %d' % (f, t)
la = sum(1 for l in open(of + '/log', errors='replace') if l.startswith('bounding '))
lc = sum(1 for l in open(br + '/log', errors='replace') if l.startswith('bounding '))
def res(fn):
    out = {}
    for m in re.finditer(r'Solving for (\w+), Initial residual = ([0-9.e+-]+)', open(fn, errors='replace').read()):
        out.setdefault(m.group(1), []).append(float(m.group(2)))
    return out
ra, rb = res(of + '/log'), res(br + '/log')
gap = {f: max(abs(x - y) / max(abs(x), 1e-300) for x, y in zip(ra[f], rb[f])) for f in ra if f in rb}
print('        worst %.3e (%s), bounding lines %d and %d; residuals %s' % (worst, where, la, lc, ' '.join('%s %.2e' % kv for kv in gap.items())))
inside = worst < BOUND and all(v < RESIDUAL for v in gap.values())
if want in ('yes', 'no'):
    raise SystemExit(0 if inside and la == lc and (want == 'no' or la > 0) else 1)
raise SystemExit(0 if not inside else 1)
PYEOF
# run_of, compare, and the two controls on one fixture. arm <name> <fixture> <second> <F1 var> <bounds: yes|no> <end>
arm() {
    local name=$1 fx=$2 sec=$3 extra=$4 bounds=$5 end=$6 D=$W/$1
    mkdir -p "$D"
    stage "$D/of" "$fx" "$sec" "$end" 1
    [ -n "${ARM_PREP:-}" ] && "$ARM_PREP" "$D/of" "$sec"
    ( set +u; source "$OFBASHRC" > /dev/null 2>&1; cd "$D/of" && simpleFoam > log 2>&1 < /dev/null ) || { say "$name: OpenFOAM runs" FAIL; return; }
    stage "$D/br" "$fx" "$sec" "$end" 1
    [ -n "${ARM_PREP:-}" ] && "$ARM_PREP" "$D/br" "$sec"
    br "$D/br" "$extra"
    python3 "$W/cmp.py" "$D/of" "$D/br" "$sec" "$bounds" "$end" && say "$name: every iteration within 1e-9 of OpenFOAM, the same bounding count" ok || say "$name: every iteration within 1e-9 of OpenFOAM, the same bounding count" FAIL
    if [ "$bounds" = yes ]
    then
        stage "$D/emp" "$fx" "$sec" "$end" 1
        [ -n "${ARM_PREP:-}" ] && "$ARM_PREP" "$D/emp" "$sec"
        br "$D/emp" "$extra" BRAE_CONTROL_BOUND_EMPTY=1
        python3 "$W/cmp.py" "$D/of" "$D/emp" "$sec" red "$end" && say "$name: CONTROL the average counting empty faces is red" ok || say "$name: CONTROL the average counting empty faces is red" FAIL
    else
        stage "$D/liv" "$fx" "$sec" "$end" 1
        [ -n "${ARM_PREP:-}" ] && "$ARM_PREP" "$D/liv" "$sec"
        br "$D/liv" "$extra" BRAE_CONTROL_TURB_GRAD_LIVE=1
        python3 "$W/cmp.py" "$D/of" "$D/liv" "$sec" red "$end" && say "$name: CONTROL the gradient reading live faces is red" ok || say "$name: CONTROL the gradient reading live faces is red" FAIL
    fi
}
# A negative cell away from every patch in the second field's start
negative_cell() {
    python3 - "$1" "$2" <<'PYEOF'
import re, sys
c, sec = sys.argv[1:]
s = open(c + '/constant/polyMesh/owner', errors='replace').read()
n = int(re.search(r'nCells:\s*(\d+)', s).group(1))
p = c + '/0/' + sec
b = open(p).read()
m = re.search(r'internalField\s+uniform\s+([-0-9.eE+]+);', b)
x = float(m.group(1))
v = [repr(x)] * n
v[n // 2] = repr(-x)
open(p, 'w').write(b[:m.start()] + 'internalField nonuniform List<scalar> %d\n(\n%s\n);' % (n, '\n'.join(v)) + b[m.end():])
PYEOF
}
ARM_PREP=
arm SST pitzDailySST omega BRAE_SST_F1_PATCH=1 yes 40
# ...and the wall rows' norm factor: the fields exact and omega's residual 0.16% out (job 4845099)
stage "$W/SST/nrm" pitzDailySST omega 40 1
br "$W/SST/nrm" BRAE_SST_F1_PATCH=1 BRAE_CONTROL_WALL_ROW_NORM=1
python3 "$W/cmp.py" "$W/SST/of" "$W/SST/nrm" omega red 40 && say "SST: CONTROL the wall rows' norm factor is red" ok || say "SST: CONTROL the wall rows' norm factor is red" FAIL
# ...and the second control, the gradient reading live faces, on SST
stage "$W/SST/liv" pitzDailySST omega 40 1
br "$W/SST/liv" BRAE_SST_F1_PATCH=1 BRAE_CONTROL_TURB_GRAD_LIVE=1
python3 "$W/cmp.py" "$W/SST/of" "$W/SST/liv" omega red 40 && say "SST: CONTROL the gradient reading live faces is red" ok || say "SST: CONTROL the gradient reading live faces is red" FAIL
arm KE pitzDailyTurb epsilon BRAE_NOTHING=1 no 40
ARM_PREP=negative_cell
arm KE-start pitzDailyTurb epsilon BRAE_NOTHING=1 yes 1
ARM_PREP=
[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
