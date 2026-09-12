#!/usr/bin/env bash
# The per-solve report: OpenFOAM's three numbers, printed by brae, matching OpenFOAM's.
#
# OpenFOAM prints one line per linear solve (SolverPerformance.C:99-117):
#     <solver>:  Solving for <field>, Initial residual = a, Final residual = b, No Iterations n
# The V2 driver printed only the INITIAL residual of U and p, so brae's log could say where a solve
# started and never where it STOPPED -- and item 32 showed that where the momentum solve stops is what
# decides whether validation/T3A converges at all. Now it prints all three, at OpenFOAM's own six
# significant figures, so a brae log diffs against an OpenFOAM one line for line.
#
# THE ORACLE is real simpleFoam on the same case, one iteration, its own log. Both codes start from the
# same 0.orig, assemble the same momentum system (tests/gs_ladder LEG 0: 1.2e-12) and run the same
# smoother (LEG 2: 2.8e-12) under the same stopping rule, so their first Ux and Uy lines must be the SAME
# THREE NUMBERS. Measured on T3A: `Initial residual = 1, Final residual = 0.0711019, No Iterations 5` and
# `1, 0.099005, 3` -- byte-identical after the solver-name prefix.
#
#   LEG 1   every U component OpenFOAM printed a line for, brae printed one for -- and no others. T3A is
#           2D, so BOTH must omit Uz (fvMatrixSolve.C:164 skips it; so does brae's solutionD mask).
#   LEG 2   on each of those, Initial, Final and No Iterations agree: the residuals to the six printed
#           digits, the count exactly.
#   LEG 3   p's Initial residual agrees (same system). Its Final and No Iterations are REPORTED beside
#           OpenFOAM's but not asserted: brae runs an AMG-preconditioned PCG where the case asks GAMG,
#           announced as a substitution, and the two stop at different points by construction.
#   FORMAT  brae's lines parse under the same regex as OpenFOAM's, so the diff is line for line.
#   CONTROL the same brae case with `relTol 0.01` on U must print MORE iterations and a SMALLER final
#           residual for Ux. That proves the printed numbers are the solve's own and not echoed
#           settings -- and it is precisely the run LEG 2 would fail on if brae's solve stopped elsewhere.
#
# Fail-proof: the control run IS the fail-proof of the comparison's resolution (a stopping point one
# relTol away is caught). Removing the print leaves LEG 1 with no lines, which fails trivially.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/T3A}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
prep() {   # prep <dir> [uRelTol]
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$1/"
    cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "${2:-}" <<'PY'
import re, sys
d, rt = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^endTime .*', 'endTime         1;', s, flags=re.M)
open(c, 'w').write(s)
if rt:
    f = d + '/system/fvSolution'; s = open(f).read()
    # T3A groups U with the turbulence scalars in one regex entry; give U its own tighter one.
    s = s.replace('solvers\n{', 'solvers\n{\n    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol %s; maxIter 1000; }\n' % rt, 1)
    open(f, 'w').write(s)
PY
}
prep "$W/of"
prep "$W/br"
prep "$W/ctl" 0.01

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
( cd "$W/of"  && simpleFoam > log 2>&1 ) || { echo "FAIL: simpleFoam did not run"; tail -10 "$W/of/log"; exit 1; }
( cd "$W/br"  && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/br"  > log 2>&1 ) || { echo "FAIL: brae crashed"; tail -10 "$W/br/log"; exit 1; }
( cd "$W/ctl" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/ctl" > log 2>&1 ) || { echo "FAIL: brae crashed on the control"; tail -10 "$W/ctl/log"; exit 1; }

python3 - "$W/of/log" "$W/br/log" "$W/ctl/log" <<'PY'
import re, sys
RX = re.compile(r'^(\S+):  Solving for (\w+), Initial residual = ([-+0-9.eE]+), Final residual = ([-+0-9.eE]+), No Iterations (\d+)\s*$')

def first_iteration(path):
    """The solve lines of the FIRST iteration only, keyed by field, as (initial, final, nIter)."""
    out, seen_time = {}, 0
    for line in open(path, errors='latin-1'):
        if line.startswith('Time = '):
            seen_time += 1
            if seen_time > 1: break
        m = RX.match(line)
        if m and m.group(2) not in out:
            out[m.group(2)] = (float(m.group(3)), float(m.group(4)), int(m.group(5)), m.group(1))
    return out

of, br, ctl = (first_iteration(p) for p in sys.argv[1:4])
fail = 0
def say(msg, ok):
    global fail
    print("  %-72s %s" % (msg, "ok" if ok else "FAIL"))
    if not ok: fail = 1

# FORMAT: brae printed at least the lines OpenFOAM did, all parsing under OpenFOAM's shape.
say("FORMAT   brae's solve lines parse under OpenFOAM's own regex", len(br) >= 3)

# LEG 1: the same SET of U components -- and no Uz on a 2D case, in either.
ofU, brU = {f for f in of if f.startswith('U')}, {f for f in br if f.startswith('U')}
say("LEG 1    brae reports exactly the U components OpenFOAM reports: %s" % sorted(ofU), ofU == brU and ofU)
say("LEG 1    ...and neither prints Uz on this 2D case", 'Uz' not in of and 'Uz' not in br)

# LEG 2: the three numbers, per component. Residuals at the printed six digits, the count exactly.
def close(a, b): return abs(a - b) <= 2e-6 * max(abs(a), abs(b), 1e-300)
for f in sorted(ofU & brU):
    oi, ofin, on, _ = of[f]; bi, bfin, bn, _ = br[f]
    print("  %-4s OpenFOAM  init %-12g final %-12g n %-3d   brae  init %-12g final %-12g n %d" % (f, oi, ofin, on, bi, bfin, bn))
    say("LEG 2    %s: Initial, Final and No Iterations are OpenFOAM's" % f, close(oi, bi) and close(ofin, bfin) and on == bn)

# LEG 3: p starts from the same system; where it stops is the announced substitution, reported only.
if 'p' in of and 'p' in br:
    oi, ofin, on, osolv = of['p']; bi, bfin, bn, bsolv = br['p']
    print("  p    OpenFOAM  init %-12g final %-12g n %-3d (%s)   brae  init %-12g final %-12g n %d (%s)"
          % (oi, ofin, on, osolv, bi, bfin, bn, bsolv))
    say("LEG 3    p: Initial residual is OpenFOAM's (final/nIter differ by the announced solver)", close(oi, bi))
    say("LEG 3    ...and brae does not call its p solver GAMG", bsolv != 'GAMG')
else:
    say("LEG 3    both logs carry a p line", False)

# CONTROL: relTol 0.01 on U must stop LATER and LOWER -- the numbers are the solve's, not the settings'.
if 'Ux' in ctl and 'Ux' in br:
    _, cfin, cn, _ = ctl['Ux']; _, bfin, bn, _ = br['Ux']
    print("  CONTROL  Ux at relTol 0.1: final %g n %d   at relTol 0.01: final %g n %d" % (bfin, bn, cfin, cn))
    say("CONTROL  a tighter relTol prints more iterations and a smaller final residual", cn > bn and cfin < bfin)
    say("CONTROL  ...and that run would FAIL LEG 2, so LEG 2 can fail", not (close(cfin, of['Ux'][1]) and cn == of['Ux'][2]))
else:
    say("CONTROL  the control run printed a Ux line", False)

sys.exit(fail)
PY
rc=$?
[ $rc -eq 0 ] && echo "PASS: brae reports every solve with OpenFOAM's three numbers, and they are OpenFOAM's"
exit $rc
