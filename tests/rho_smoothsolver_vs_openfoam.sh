#!/usr/bin/env bash
# The rho mirror runs the linear solver the CASE names -- and the trajectory says it does.
#
# Item 58. OpenFOAM's angledDuctExplicitFixedCoeff asks for `solver smoothSolver; smoother GaussSeidel;
# nSweeps 2` on U and on the turbulence pair, and `symGaussSeidel` on the energy. The CUDA mirror ran
# PBiCGStab on all of them -- and said nothing, because the shared notice takes the flag the dict parser
# sets (`ctl.gsU`) as proof that the caller honours it, and this driver never plumbed it into the step.
# That is the `shared-capability-notice-lies` class: a capability written for one driver, read as a
# promise by another. Now the step carries the selection (uSymGaussSeidel / heSymGaussSeidel / the pair's
# gsK,gsEps, each with its smoother variant and its nSweeps) and BRAE_RHO_SMOOTHSOLVER=0 restores the
# substitution WITH the notices that describe it.
#
# THE ORACLE is real rhoSimpleFoam on the same case, 30 iterations from the same start, read per
# iteration. This is a TRAJECTORY comparison and it is the right one here: with the case's own solvers
# and stopping rules on U, e, k and epsilon, both codes take the same solve at each step, so their
# initial residuals must track. What is left is the pressure solver, which brae substitutes and
# announces (queue item 61, declined by the user), and that is why the bounds are factors and not 1e-12.
#
#   ARM 1    the honoured run announces NO solver substitution on U, e, k or epsilon...
#   ARM 2    ...and announces the smoothSolver path it took.
#   ARM 3    at iteration 30 its U, e, k and epsilon initial residuals are within [0.6, 1.6] of
#            OpenFOAM's (measured: U 1.20x, e 1.00x, k 0.93x, epsilon 0.90x).
#   CONTROL  BRAE_RHO_SMOOTHSOLVER=0 -- the behaviour before this item -- announces all four
#            substitutions and lands >= 2x from OpenFOAM on k and epsilon (measured 2.37x and 3.05x).
#            It is the fail-proof: run ARM 3 against that binary and it fails on both fields.
#
# NOTE on the fixture: the tutorial writes the pair's block as `"(k|epsilon)" { $U; ... }`, an OpenFOAM
# dictionary MERGE of U's entries. brae's parser reads the block's own keys but does not merge the
# referenced sub-dict, so on the shipped file k and epsilon name no solver at all and fall back to
# BiCGStab silently -- a second defect, of the same class and with a different cause, filed as queue
# item 73. This gate spells the pair's block out so that what it measures is the solver SELECTION and
# not the parser; item 73 gets its own gate.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
SRC="${FOAM_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff"
[ -d "$SRC" ] || { echo "SKIP: the angledDuct tutorial is not in this OpenFOAM"; exit 77; }
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

cp -r "$SRC" "$W/case" || exit 1
( cd "$W/case" && rm -rf 0 && cp -r 0.orig 0 ) 2>/dev/null || true
python3 - "$W/case" <<'PY'
import re, sys
d = sys.argv[1]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 30;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 30;', s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
# the pair's block, spelled out (see the note above)
f = d + '/system/fvSolution'; s = open(f).read()
old = '''    "(k|epsilon)"
    {
        $U;
        tolerance       1e-07;
        relTol          0.1;
    }'''
new = '''    "(k|epsilon)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-07;
        relTol          0.1;
    }'''
assert old in s, 'the tutorial no longer writes the pair as `$U;` -- re-read the note in this gate'
open(f, 'w').write(s.replace(old, new))
PY
[ $? -eq 0 ] || { echo "FAIL: the fixture could not be prepared"; exit 1; }
( cd "$W/case" && ./Allrun.pre > pre.log 2>&1 ) || ( cd "$W/case" && blockMesh > bm.log 2>&1 && topoSet > ts.log 2>&1 ) || true
[ -f "$W/case/constant/polyMesh/owner" ] || { echo "FAIL: the tutorial did not mesh"; tail -5 "$W/case"/*.log; exit 1; }

cp -r "$W/case" "$W/of"; rm -rf "$W/of"/[1-9]*
( cd "$W/of" && rhoSimpleFoam > of.log 2>&1 ) || { echo "FAIL: OpenFOAM did not run"; tail -20 "$W/of/of.log"; exit 1; }
run() { ( cd "$1" && rm -rf [1-9]* && env $2 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$1" > "$3" 2>&1 ); }
cp -r "$W/case" "$W/hon"; cp -r "$W/case" "$W/ctl"
run "$W/hon" "" "$W/hon/run.log" || { echo "FAIL: the honoured run crashed"; tail -10 "$W/hon/run.log"; exit 1; }
run "$W/ctl" "BRAE_RHO_SMOOTHSOLVER=0" "$W/ctl/run.log" || { echo "FAIL: the control run crashed"; tail -10 "$W/ctl/run.log"; exit 1; }

# ARM 1 / ARM 2 -- what each run says it did
for f in U e k epsilon; do
    if grep -q "solvers/$f solver: case asks 'smoothSolver'" "$W/hon/run.log"
    then say "ARM 1  the honoured run does not announce a substitution on $f" FAIL
    else say "ARM 1  the honoured run does not announce a substitution on $f" ok; fi
    grep -q "solvers/$f solver: case asks 'smoothSolver'" "$W/ctl/run.log" \
        && say "CONTROL  ...and the substituted run announces it on $f" ok \
        || say "CONTROL  ...and the substituted run announces it on $f" FAIL
done
grep -q "smoothSolver: " "$W/hon/run.log" && say "ARM 2  the honoured run announces the smoothSolver path it took" ok \
                                          || say "ARM 2  the honoured run announces the smoothSolver path it took" FAIL
grep -q "smoothSolver: " "$W/ctl/run.log" && say "CONTROL  ...and the substituted run never enters it" FAIL \
                                          || say "CONTROL  ...and the substituted run never enters it" ok

# ARM 3 / CONTROL -- the trajectory against OpenFOAM
python3 - "$W" <<'PY' || fail=1
import re, sys
W = sys.argv[1]
IT = 30
of, it = {}, 0
for line in open(W + '/of/of.log', errors='latin-1'):
    if line.startswith('Time = '):
        it = int(line.split('=')[1]); continue
    m = re.match(r'\S+:  Solving for (\w+), Initial residual = ([\d.eE+-]+)', line)
    if m and it: of.setdefault(it, {}).setdefault(m.group(1), float(m.group(2)))
def brae(p):
    d = {}
    for line in open(p, errors='latin-1'):
        m = re.match(r'Time = (\d+)\s+(.*)$', line)
        if not m: continue
        d[int(m.group(1))] = {f: float(v) for f, v in re.findall(r'(\w+) ([\d.eE+-]+)', m.group(2))}
    return d
hon, ctl = brae(W + '/hon/run.log'), brae(W + '/ctl/run.log')
if IT not in of or IT not in hon or IT not in ctl:
    print("  the runs did not all reach iteration %d" % IT); sys.exit(1)
ofv = {'U': max(of[IT].get('Ux', 0), of[IT].get('Uy', 0), of[IT].get('Uz', 0)),
       'e': of[IT].get('e'), 'k': of[IT].get('k'), 'epsilon': of[IT].get('epsilon')}
bad = 0
# Bounds from the first green run: U 1.20x, e 1.00x, k 0.93x, epsilon 0.90x. They tighten, not loosen.
LO, HI = 0.6, 1.6
for f in ('U', 'e', 'k', 'epsilon'):
    o, h, c = ofv[f], hon[IT].get(f), ctl[IT].get(f)
    if o is None or h is None or c is None: print("  %s missing" % f); bad = 1; continue
    r, rc = h / o, c / o
    ok = LO <= r <= HI
    print("  ARM 3    %-8s iteration %d: OpenFOAM %.4e   brae %.4e (%.2fx, bound %.1f-%.1f)   %s"
          % (f, IT, o, h, r, LO, HI, "ok" if ok else "FAIL"))
    if not ok: bad = 1
    if f in ('k', 'epsilon'):
        okc = rc >= 2.0
        print("  CONTROL  %-8s substituted %.4e (%.2fx OpenFOAM; must be >= 2.0x)                   %s"
              % (f, c, rc, "ok" if okc else "FAIL"))
        if not okc: bad = 1
sys.exit(bad)
PY
[ $fail -eq 0 ] && echo "PASS: the rho mirror runs the case's own linear solvers, and its trajectory is OpenFOAM's"
exit $fail
