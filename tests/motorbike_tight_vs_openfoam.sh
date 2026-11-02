#!/usr/bin/env bash
# The coarse motorBike (353,784 cells from snappyHexMesh, kOmegaSST) as a deterministic 3-D correctness gate: serial OpenFOAM and brae from the same 0 on the same mesh, every solve tight, 20 iterations.
# tutorial_motorbike cannot be this. It compares iteration 500 of a loose run of a moving flow, parallel OpenFOAM started by potentialFoam against brae started cold, so a real defect and a
# phase of the flow read alike. MEASURED (job 4845479): here the candidate is U 2.5e-08 at iteration 20 and the baseline 1.6e-01, where at iteration 500 of the loose run they read 1.0e-01 and 5.4e-02,
# the wrong way round: the baseline's non-orthogonal momentum correction read the wrong gradient and its snapshot happened to fall closer.
# THE BOUNDS: U, p, k, omega within 5e-7 at iterations 5, 10, 15, 20, nut within 5e-6 (MEASURED worst k 4.3e-08, nut 1.4e-06), and every iteration's initial residuals for Ux, k and omega within 1e-4.
# PREMISE, asserted: the run's first residuals move (the case develops), both codes print the same number of bounding lines.
# CONTROL, asserted red: BRAE_CONTROL_LAP_CORR_DEFAULT_GRAD=1, the old gradient.
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
# The mesh, the tutorial's own way. One iteration of its solver: Allrun's last two steps are what reconstruct it.
( set +u; source "$OFBASHRC" > /dev/null 2>&1
  cp -r "$FOAM_TUTORIALS/incompressible/simpleFoam/motorBike" "$W/mesh" || exit 1
  sed -i -E 's/^endTime .*/endTime 1;/; s/^writeInterval .*/writeInterval 1;/' "$W/mesh/system/controlDict"
  cd "$W/mesh" && timeout 2400 ./Allrun > allrun.log 2>&1 ) < /dev/null
[ -f "$W/mesh/constant/polyMesh/owner" ] && [ -f "$W/mesh/log.reconstructParMesh" ] || { echo "FAIL: the mesh was not made"; tail -n 3 "$W/mesh/allrun.log"; exit 1; }
stage() {
    local c=$1
    mkdir "$c"
    cp -r "$W/mesh/system" "$c/"
    ln -s "$W/mesh/constant" "$c/constant"
    cp -r "$W/mesh/0.orig" "$c/0"
    python3 - "$c" <<'PYEOF'
import re, sys
c = sys.argv[1]
p = c + '/system/controlDict'; s = open(p).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in (('endTime', '20'), ('writeInterval', '5'), ('writeFormat', 'ascii'), ('writePrecision', '17')):
    s, n = re.subn(r'(?m)^%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(p, 'w').write(s)
p = c + '/system/fvSolution'; s = open(p).read()
s, n = re.subn(r'solvers\s*\{.*?\n\}', """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0; maxIter 5000; }
    Phi { $p; }
    "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; nSweeps 1; tolerance 1e-13; relTol 0; maxIter 5000; }
}""", s, count=1, flags=re.S)
assert n == 1
open(p, 'w').write(s)
PYEOF
}
# The case's dictionaries #includeEtc, so brae needs OpenFOAM's environment as tests/tutorial_vs_openfoam.sh gives it. A run that ends in an error stops here by name.
br()
{
    local c=$1
    shift
    ( set +u; source "$OFBASHRC" > /dev/null 2>&1; set -u; cd "$c" && env BRAE_PRESSURE_CASE_SOLVER=1 BRAE_TURBULENCE_CASE_SOLVER=1 BRAE_SIMPLE_CASE_SOLVER=1 "$@" "$BRAE" . > run.log 2>&1 < /dev/null )
    local rc=$?
    if [ "$rc" != 0 ]
    then
        echo "FAIL: brae exit $rc in $(basename "$c"): $(grep -a -m1 -E 'brae ERROR|error' "$c/run.log" | cut -c1-160)"
        exit 1
    fi
}
cat > "$W/cmp.py" <<'PYEOF'
import re, sys
import numpy as np
of, br, want = sys.argv[1:4]
def rd(fn):
    s = open(fn, errors='replace').read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', s)
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    b = s[m.end():]
    b = b[:b.index('\n)')]
    return np.array(b.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)
worst = {}
for t in (5, 10, 15, 20):
    for f in ('U', 'p', 'k', 'omega', 'nut'):
        x, y = rd('%s/%d/%s' % (of, t, f)), rd('%s/%d/%s' % (br, t, f))
        worst[f] = max(worst.get(f, 0.0), np.linalg.norm(y - x) / np.linalg.norm(x))
def res(fn):
    out = {}
    for m in re.finditer(r'Solving for (Ux|k|omega), Initial residual = ([0-9.e+-]+)', open(fn, errors='replace').read()):
        out.setdefault(m.group(1), []).append(float(m.group(2)))
    return out
a, b = res(of + '/run.log'), res(br + '/run.log')
gap = {f: max(abs(x - y) / abs(x) for x, y in zip(a[f], b[f])) for f in a}
la = sum(1 for l in open(of + '/run.log', errors='replace') if l.startswith('bounding '))
lc = sum(1 for l in open(br + '/run.log', errors='replace') if l.startswith('bounding '))
print('        fields %s' % ' '.join('%s %.2e' % kv for kv in worst.items()))
print('        residuals %s; Ux from %.3e to %.3e; bounding lines %d and %d' % (' '.join('%s %.2e' % kv for kv in gap.items()), a['Ux'][1], a['Ux'][-1], la, lc))
inside = all(worst[f] < (5e-6 if f == 'nut' else 5e-7) for f in worst) and all(v < 1e-4 for v in gap.values()) and la == lc and len(a['Ux']) == len(b['Ux']) == 20
raise SystemExit(0 if inside == (want == 'ok') else 1)
PYEOF
stage "$W/of"
( set +u; source "$OFBASHRC" > /dev/null 2>&1; cd "$W/of" && timeout 3000 simpleFoam > run.log 2>&1 < /dev/null ) || { echo "FAIL: OpenFOAM did not finish"; tail -n 3 "$W/of/run.log"; exit 1; }
stage "$W/br"
br "$W/br"
python3 "$W/cmp.py" "$W/of" "$W/br" ok && say "20 tight iterations on the motorBike against serial OpenFOAM" ok || say "20 tight iterations on the motorBike against serial OpenFOAM" FAIL
stage "$W/ctl"
br "$W/ctl" BRAE_CONTROL_LAP_CORR_DEFAULT_GRAD=1
python3 "$W/cmp.py" "$W/of" "$W/ctl" red && say "CONTROL: the old gradient in the non-orthogonal correction is red" ok || say "CONTROL: the old gradient in the non-orthogonal correction is red" FAIL
[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
