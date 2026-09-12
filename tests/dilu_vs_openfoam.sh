#!/usr/bin/env bash
# brae runs the PRECONDITIONER the case names -- and then its momentum solve IS OpenFOAM's.
#
# Item 74. OpenFOAM's PBiCGStab is a preconditioned BiCGStab and essentially every incompressible
# tutorial asks for DILU; the V2 mirror ran a diagonal-preconditioned one and announced the substitution.
# Announced is not the same as harmless: both reach the requested relTol but stop at different residuals,
# and the compressible mirror measured that difference at k 5.4e-09 from OpenFOAM under Jacobi against
# 8.4e-12 under DILU, on assembled systems agreeing to 1e-11 (queue item 27). It also costs: on
# pitzDailyTurb the first Ux solve took 19 Krylov iterations under the diagonal and takes 5 under DILU.
#
# THE ORACLE is real simpleFoam on the same case, one iteration, its own log. With the case's own solver
# AND preconditioner, brae solves the same system by the same method to the same tolerance, so the three
# numbers OpenFOAM prints must be the three numbers brae prints -- not merely close. Measured:
#     OpenFOAM  DILUPBiCGStab:  Solving for Ux, Initial residual = 1, Final residual = 2.43274914e-11, No Iterations 5
#     brae      DILUPBiCGStab:  Solving for Ux, Initial residual = 1, Final residual = 2.43275e-11,    No Iterations 5
#
#   LEG 1   every U component OpenFOAM printed a line for, brae printed one for -- and no others.
#   LEG 2   on each, Initial, Final and No Iterations are OpenFOAM's (residuals to the printed digits,
#           the count exactly).
#   LEG 3   the SOLVER NAME brae prints is the one OpenFOAM prints, DILUPBiCGStab: the log diffs line for
#           line, and a name that said Jacobi while running DILU would be the same class of defect this
#           item fixed.
#   LEG 4   the turbulence pair too (BRAE_TURB_RESID prints those lines): k and epsilon take OpenFOAM's
#           iteration count exactly -- that is the preconditioner and the stopping rule -- on a system
#           within 1% of its own. k's initial residual is exact; epsilon's is 0.4% out for reasons that
#           predate this item, and a gate about the SOLVER must not fail on the assembly.
#   CONTROL BRAE_DILU=0 -- the behaviour before this item -- must MISS LEG 2 by a wide margin (19
#           iterations against 5), so the legs have resolution and the fail-proof is a real run.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/pitzDailyTurb}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
prep() {
    mkdir -p "$1"; cp -r "$SRC/constant" "$SRC/system" "$1/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$1/0"; else cp -r "$SRC/0" "$1/0"; fi
    python3 - "$1" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 1;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 1;', s)
open(c, 'w').write(s)
PY
}
prep "$W/of"; prep "$W/br"; prep "$W/ctl"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
( cd "$W/of" && simpleFoam > log 2>&1 ) || { echo "FAIL: simpleFoam did not run"; tail -8 "$W/of/log"; exit 1; }
( cd "$W/br"  && BRAE_TURB_RESID=1 BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/br"  > log 2>&1 ) || { echo "FAIL: brae crashed"; tail -8 "$W/br/log"; exit 1; }
( cd "$W/ctl" && BRAE_DILU=0 BRAE_TURB_RESID=1 BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/ctl" > log 2>&1 ) || { echo "FAIL: the control crashed"; tail -8 "$W/ctl/log"; exit 1; }

python3 - "$W/of/log" "$W/br/log" "$W/ctl/log" <<'PY'
import re, sys
RX  = re.compile(r'^(\S+):  Solving for (\w+), Initial residual = ([-+0-9.eE]+), Final residual = ([-+0-9.eE]+), No Iterations (\d+)\s*$')
RXT = re.compile(r'^\s*Solving for (\w+), Initial residual = ([-+0-9.eE]+), No Iterations (\d+)\s*$')
def first(path):
    out, turb, seen = {}, {}, 0
    for line in open(path, errors='latin-1'):
        if line.startswith('Time = '):
            seen += 1
            if seen > 1: break
        m = RX.match(line)
        if m and m.group(2) not in out:
            out[m.group(2)] = (float(m.group(3)), float(m.group(4)), int(m.group(5)), m.group(1))
        t = RXT.match(line)
        if t and t.group(1) not in turb: turb[t.group(1)] = (float(t.group(2)), int(t.group(3)))
    return out, turb
of, oft   = first(sys.argv[1])
br, brt   = first(sys.argv[2])
ctl, _    = first(sys.argv[3])
fail = 0
def say(msg, ok):
    global fail
    print("  %-74s %s" % (msg, "ok" if ok else "FAIL"))
    if not ok: fail = 1
def close(a, b): return abs(a - b) <= 2e-6 * max(abs(a), abs(b), 1e-300)
ofU, brU = {f for f in of if f.startswith('U')}, {f for f in br if f.startswith('U')}
say("LEG 1    brae reports exactly the U components OpenFOAM reports: %s" % sorted(ofU), bool(ofU) and ofU == brU)
for f in sorted(ofU & brU):
    oi, ofin, on, osolv = of[f]; bi, bfin, bn, bsolv = br[f]
    print("  %-4s OpenFOAM  init %-12g final %-12g n %-3d   brae  init %-12g final %-12g n %d" % (f, oi, ofin, on, bi, bfin, bn))
    say("LEG 2    %s: Initial, Final and No Iterations are OpenFOAM's" % f, close(oi, bi) and close(ofin, bfin) and on == bn)
    say("LEG 3    %s: and the solver NAME is OpenFOAM's own (%s)" % (f, osolv), bsolv == osolv)
# OpenFOAM prints the turbulence solves in the same full format as the momentum ones, so they land in
# `of`; brae's come from its own short line (BRAE_TURB_RESID), so they land in `brt`.
for f in ('k', 'epsilon', 'omega'):
    if f not in of or f not in brt: continue
    oi, _ofin, on, _os = of[f]; bi, bn = brt[f]
    print("  %-8s OpenFOAM  init %-12g n %-3d   brae  init %-12g n %d  (init %.2f%% apart)"
          % (f, oi, on, bi, bn, 100.0 * abs(oi - bi) / max(abs(oi), 1e-300)))
    # The ITERATION COUNT is what this gate is about: it is decided by the preconditioner and the
    # stopping rule, and it must be OpenFOAM's exactly. The initial residual is a property of the
    # ASSEMBLED system, and epsilon's differs by 0.4% here for reasons that predate this item (k's is
    # exact), so it is bounded at 1% and reported rather than asserted tight -- a gate that demanded
    # more would be failing on something it does not test.
    say("LEG 4    %s: the turbulence solve takes OpenFOAM's iteration count" % f, on == bn)
    say("LEG 4    %s: ...on a system within 1%% of OpenFOAM's" % f,
        abs(oi - bi) <= 0.01 * max(abs(oi), 1e-300))
if 'Ux' in ctl and 'Ux' in of:
    _, cfin, cn, _ = ctl['Ux']; oi, ofin, on, _ = of['Ux']
    print("  CONTROL  Ux without DILU: final %g n %d   against OpenFOAM's %g n %d" % (cfin, cn, ofin, on))
    say("CONTROL  the diagonal-preconditioned run MISSES LEG 2 (so LEG 2 can fail)",
        not (close(ofin, cfin) and on == cn))
sys.exit(fail)
PY
rc=$?
[ $rc -eq 0 ] && echo "PASS: with the case's own preconditioner, brae's solve is OpenFOAM's -- name, residuals and iteration count"
exit $rc
