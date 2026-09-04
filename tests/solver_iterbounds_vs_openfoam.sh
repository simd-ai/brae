#!/usr/bin/env bash
# maxIter AND minIter come from EACH FIELD'S OWN solver entry, and both are honoured.
#
# lduMatrix::solver::readControls (lduMatrixSolver.C:190-208) reads `minIter` and `maxIter` out of the
# field's own sub-dictionary, defaulting to 0 and lduMatrix::defaultMaxIter = 1000 (lduMatrix.H:125).
# The V2 driver read ONE maxIter, from the `p` entry, and handed it to the momentum solve as well:
# validation/T3A caps `"(U|k|omega|gammaInt|ReThetat)"` at 10 and says nothing about p, so brae ran U at
# 1000 where OpenFOAM stops at 10. minIter was read nowhere and passed nowhere, and once it was passed it
# turned out the device BiCGStab ignored it at its mid-iteration exit -- OpenFOAM guards that return with
# the floor (PBiCGStab.C:222-224) and brae did not.
#
# A cap is a statement about the ANSWER, not a performance hint: two solvers stopped by a cap hold two
# different residuals, and a floor is the same statement from below.
#
#   ARM 1  maxIter, per field. U gets `maxIter 3` and p gets `maxIter 37` in the SAME case, with U's
#          tolerance tightened so the cap binds. ORACLE: OpenFOAM's own log stops Ux at 3. brae must stop
#          at 3 AND print the two caps separately -- reading maxIter from `p` would print 37 on U.
#          CONTROL: the same case at `maxIter 1000` must take MORE than 3 (measured 51-59), or the cap
#          was never what stopped it.
#   ARM 2  minIter. U gets `minIter 5` with the stock loose relTol, so the solve would converge in 1.
#          ORACLE: OpenFOAM's own log runs Ux for exactly 5. CONTROL: without the floor brae takes 1-2.
#
# The uncapped ITERATION COUNTS are deliberately not compared between the codes -- OpenFOAM's DILU
# reaches 1e-14 in 3 where brae's Jacobi needs ~55, which is the preconditioner substitution and not this.
# Only the pinned counts are compared, because only those are pinned by the same dictionary entry.
#
# Fail-proof, 2026-09-04: reading maxIter from `p` again makes ARM 1 print `U ... maxIter=37` and take 37;
# dropping the `nIter >= minIter` guard at the BiCGStab mid-iteration exit makes ARM 2 read 1.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/simpleBoxIO}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
CAP=${CAP:-3}
PCAP=${PCAP:-37}
FLOOR=${FLOOR:-5}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # stage <dir> <capped|free|floor>
{
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    if [ -f "$1/system/blockMeshDict" ] && command -v blockMesh > /dev/null 2>&1; then
        ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
    fi
    python3 - "$1" "$2" "$CAP" "$PCAP" "$FLOOR" <<'PY'
import re, sys
d, mode, cap, pcap, floor = sys.argv[1:6]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;', 'endTime         3;', s)
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S)
# The U entry, rewritten whole so the fixture's own settings cannot leak into the arm.
if mode == 'floor':
    u = 'U { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0.1; minIter %s; }' % floor
elif mode == 'nofloor':
    u = 'U { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0.1; }'
else:
    # tolerance tightened and relTol removed so the solve does NOT converge before the cap.
    u = ('U { solver PBiCGStab; preconditioner DILU; tolerance 1e-14; relTol 0; maxIter %s; }'
         % (cap if mode == 'capped' else '1000'))
s = re.sub(r'U \{[^}]*\}', u, s)
s = re.sub(r'p \{[^}]*\}',
           'p { solver GAMG; smoother DICGaussSeidel; tolerance 1e-10; relTol 0.01; maxIter %s; }' % pcap, s)
open(f, 'w').write(s)
PY
}

run_of()  { ( cd "$1" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM did not run in $1"; exit 1; }; }
run_br()  { ( cd "$1" && BRAE_SIMPLEFOAM_V2=1 BRAE_SOLVER_ITERS=1 "$BRAE" "$1" > run.log 2>&1 ) \
            || { echo "FAIL: brae crashed in $1"; tail -12 "$1/run.log"; exit 1; }; }

for m in capped free floor nofloor; do
    stage "$W/of_$m" "$m"; stage "$W/br_$m" "$m"
    run_of "$W/of_$m"; run_br "$W/br_$m"
done

python3 - "$W" "$CAP" "$PCAP" "$FLOOR" <<'PY'
import os, re, sys
W, cap, pcap, floor = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
fail = 0
def say(msg, ok):
    global fail
    print('  %-70s %s' % (msg, 'ok' if ok else 'FAIL'))
    if not ok: fail = 1
def ofn(d):
    return [int(x) for x in re.findall(r'Solving for Ux,.*No Iterations (\d+)',
                                       open(os.path.join(W, d, 'run.log')).read())]
def brn(d):
    return [int(x) for x in re.findall(r'\[U0\] nIter=(\d+)',
                                       open(os.path.join(W, d, 'run.log')).read())]
def line(d):
    m = re.search(r'linear solves: (.*)', open(os.path.join(W, d, 'run.log')).read())
    return m.group(1) if m else ''

# ARM 1 -- maxIter, from U's own entry.
o, b = ofn('of_capped'), brn('br_capped')
print('         capped: OpenFOAM Ux %s   brae [U0] %s' % (o, b))
say('ARM 1  OpenFOAM stops Ux at maxIter %d' % cap, bool(o) and all(n == cap for n in o))
say('ARM 1  brae stops U at the SAME cap, read from U\'s own entry',
    bool(b) and all(n == cap for n in b))
l = line('br_capped')
# Whole-token match: `maxIter=3` is a SUBSTRING of `maxIter=37`, and the fail-proof that set U's cap
# from p's passed this row while failing the one above it until the boundary was added.
say('ARM 1  the log carries both caps separately (p %d, U %d)' % (pcap, cap),
    re.search(r'p .*maxIter=%d\b' % pcap, l) is not None
    and re.search(r'U .*maxIter=%d\b' % cap, l) is not None)
bf = brn('br_free')
print('         free:   brae [U0] %s' % bf)
say('CONTROL  uncapped, brae takes more than %d, so the cap is what stopped it' % cap,
    bool(bf) and all(n > cap for n in bf))

# ARM 2 -- minIter.
o2, b2 = ofn('of_floor'), brn('br_floor')
print('         floor:  OpenFOAM Ux %s   brae [U0] %s' % (o2, b2))
say('ARM 2  OpenFOAM runs Ux for the minIter floor of %d' % floor,
    bool(o2) and all(n == floor for n in o2))
say('ARM 2  brae runs U for the same floor', bool(b2) and all(n == floor for n in b2))
b3 = brn('br_nofloor')
print('         nofloor: brae [U0] %s' % b3)
say('CONTROL  without the floor brae takes fewer than %d' % floor,
    bool(b3) and all(n < floor for n in b3))
sys.exit(1 if fail else 0)
PY
rc=$?
[ $rc -eq 0 ] && echo "PASS: maxIter and minIter are read per field and honoured as OpenFOAM does"
exit $rc
