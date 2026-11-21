#!/usr/bin/env bash
# A DIVERGED RUN MUST STOP on rhoSimpleFoam's DEFAULT driver too (`brae -case X`, no mirror).
#
# rho_nonfinite_refusal holds the mirror arm to this; the default driver checked nothing for finiteness
# (gpuRhoSimpleFoam.cu's loop): on squareBend run subsonic at 0.1 kg/s it printed `Ux nan  p nan` from
# before iteration 50 to the end, wrote a 100/ full of NaN and exited 0 (measured on main 9b7b1e58). It now
# stops on the first residual that is not a finite non-negative number (solution_health.cuh's
# healthResidual; the residuals are already on the host, so the check adds no device read), and refuses to
# write a time directory holding a NaN/Inf or a negative temperature.
#
# DETERMINISTIC BY CONSTRUCTION. Every stop is provoked by a controlled injection (BRAE_TEST_CORRUPT puts one bad
# value in one cell of one field AFTER that iteration's residual check), never by waiting for a case to diverge. The
# first version of this gate did wait: squareBend run subsonic at 0.1 kg/s diverged by iteration 18 on the GB10
# (sm_121), while on an Arrhenius GH200 (sm_90) the same source ran all 60 iterations healthily (2026-11-21), so its
# premise held on one GPU and not on the other. A divergence that round-off decides is not a reproducer.
#
#   RESIDUAL    rhoBox, p=Inf after iteration 3: the run stops at iteration 4 on a residual, having written nothing.
#   CONTROL     the same injection under BRAE_ALLOW_NONFINITE=1: iteration 4's residuals are non-finite in the log
#               and the run goes on to its end (the old behaviour) -- so it is the stop, and nothing else, that
#               ends RESIDUAL at 4. (The next pressure solve overwrites the one bad cell, so the CONTROL's final
#               field is finite: this is about the residual the stop reads, not about the written field.)
#   CONTINUITY  rhoBox, phi=NaN after iteration 3: stopped at iteration 4 on a residual, nothing written.
#   TURBULENCE  squareBend as shipped (kEpsilon, 112,000 cells), k=NaN after iteration 3: stopped at iteration 4 on
#               a residual (p, the first checked that it reaches through nut), nothing written.
#   WRITE       rhoBox, U=NaN after the last iteration's residual check (no residual ever sees it): the final write
#               refuses, names U, and leaves no directory.
#   FAIL-PROOF  squareBend as shipped (transonic yes, 0.5 kg/s) runs all N iterations with the stop armed, and
#               rhoBox runs to its end: the stop does not fire on a healthy run.
# Labels: slow (squareBend is 112,000 cells). Honours OFBASHRC (OpenFOAM's blockMesh stages squareBend).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-60}
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: OpenFOAM not available ($OFBASHRC)"; exit 77; }
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true   # before set -u: the bashrc reads unset variables
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: OpenFOAM's meshers not on PATH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
TUT="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/squareBend"
[ -d "$TUT" ] || { echo "SKIP: tutorials not found"; exit 77; }
[ -f "$ROOT/validation/rhoBox/constant/polyMesh/owner" ] || { echo "SKIP: fixture rhoBox has no mesh"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
written() { (cd "$1" && ls -d [1-9]* 2>/dev/null | wc -l); }
# iterations a log reached: OpenFOAM's `Time = <t>` line, which BRAE_OF_LOG=1 makes brae print too (brae's own
# summary line, `Time = <t>   Ux ...`, at iteration 1 and every 50th, is not counted)
its() { grep -cE '^Time = [^ ]+$' "$1" 2>/dev/null; }

# stage <dir> <transonic yes|no> <massFlowRate>   (as rho_nonfinite_refusal.sh; called with the tutorial's own values)
stage() {
    local d="$1" tr="$2" mf="$3"
    rm -rf "$d"; cp -r "$TUT" "$d"; rm -rf "$d"/0 "$d"/[1-9]* "$d"/log.*
    cp -r "$d/0.orig" "$d/0"
    python3 - "$d" "$N" "$tr" "$mf" <<'PY'
import re, sys
d, n, tr, mf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*', '\n', s, flags=re.S)
for k, v in (('startFrom','latestTime'), ('stopAt','endTime'), ('endTime',n), ('writeInterval',n)):
    s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
open(c, 'w').write(s + '\n')
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s, k = re.subn(r'\btransonic\s+\w+\s*;', 'transonic       %s;' % tr, s)
assert k == 1, 'expected one transonic entry'
open(p, 'w').write(s)
u = d + '/0/U'; s = open(u).read()
s, k = re.subn(r'(massFlowRate\s+constant\s+)[-0-9.eE]+', r'\g<1>' + mf, s)
assert k == 1, 'expected one massFlowRate entry'
open(u, 'w').write(s)
PY
    ( cd "$d" && blockMesh > log.blockMesh 2>&1 ) || return 1
}
stagebox() {  # stagebox <dir> <endTime> [writeInterval, default 1]
    rm -rf "$1"; cp -r "$ROOT/validation/rhoBox" "$1"; rm -rf "$1"/[1-9]* "$1"/0; cp -r "$1/0.orig" "$1/0"
    sed -i -E "s/\bendTime +[^;]+;/endTime $2;/; s/\bwriteInterval +[^;]+;/writeInterval ${3:-1};/" "$1/system/controlDict"
}

echo "rho_default_nonfinite_refusal: a run that goes non-finite must stop (controlled injections; $N iterations healthy)"

# stopped <name> <dir> <log> <at>: the run exited non-zero at iteration <at> on the residual check, wrote nothing
stopped() {
    local name=$1 d=$2 log=$3 at=$4 rc=$5 reached
    reached=$(its "$log")
    if [ "$rc" -ne 0 ] && grep -q "solution diverged -- invalid residual" "$log" && [ "$reached" -eq "$at" ] \
       && [ "$(written "$d")" -eq 0 ]; then
        say "$name stopped at iteration $at on a residual, nothing written" PASS
        echo "           $(grep -m1 'solution diverged' "$log" | cut -c1-140)"
    else
        say "$name expected a residual stop at iteration $at, nothing written (rc $rc, $reached its, $(written "$d") dirs)" FAIL
        grep -v NOTICE "$log" | tail -2 | sed 's/^/           /'
    fi
}

# ---- the two stops, each on what it claims to read (rhoBox, seconds) -----------------------------------
stagebox "$W/res" 8 8   # the only write at 8: the residual check, not the write-time scan, has to stop it
( cd "$W/res" && BRAE_TEST_CORRUPT=p:inf:3 BRAE_OF_LOG=1 "$BRAE" -case "$W/res" > log 2>&1 ); r_rc=$?
stopped "RESIDUAL p=Inf after iteration 3:" "$W/res" "$W/res/log" 4 "$r_rc"
stagebox "$W/ctl" 8 8
( cd "$W/ctl" && BRAE_TEST_CORRUPT=p:inf:3 BRAE_ALLOW_NONFINITE=1 BRAE_OF_LOG=1 "$BRAE" -case "$W/ctl" > log 2>&1 ); c_rc=$?
c_reached=$(its "$W/ctl/log")
c_bad=$(awk '/^Time = /{t=$3} t == 4 && /Initial residual = -?(nan|inf)/{n++} END{print n+0}' "$W/ctl/log")
if [ "$c_rc" -eq 0 ] && [ "$c_reached" -eq 8 ] && [ "$c_bad" -gt 0 ] && [ -d "$W/ctl/8" ]; then
    say "CONTROL  BRAE_ALLOW_NONFINITE=1: $c_bad non-finite residuals at iteration 4, ran on to 8 (the old behaviour)" PASS
else
    say "CONTROL  expected non-finite residuals at 4 and a run to 8 with the stop off (rc $c_rc, $c_reached its, $c_bad bad)" FAIL
fi
stagebox "$W/phi" 8 8
( cd "$W/phi" && BRAE_TEST_CORRUPT=phi:nan:3 BRAE_OF_LOG=1 "$BRAE" -case "$W/phi" > log 2>&1 ); f_rc=$?
stopped "CONTINUITY phi=NaN after iteration 3:" "$W/phi" "$W/phi/log" 4 "$f_rc"
stagebox "$W/wr" 5
( cd "$W/wr" && BRAE_TEST_CORRUPT=U:nan:5 "$BRAE" -case "$W/wr" > log 2>&1 ); w_rc=$?
if [ "$w_rc" -ne 0 ] && grep -q "non-finite value in U" "$W/wr/log" && grep -q "No field written for 5" "$W/wr/log" \
   && [ ! -d "$W/wr/5" ] && [ -d "$W/wr/4" ]; then
    say "WRITE    U=NaN after the last check: the write refuses, 1..4 written, 5/ not" PASS
else
    say "WRITE    expected the final write to refuse (rc $w_rc)" FAIL
    grep -v NOTICE "$W/wr/log" | tail -2 | sed 's/^/           /'
fi
stagebox "$W/ok_box" 8
( cd "$W/ok_box" && "$BRAE" -case "$W/ok_box" > log 2>&1 ) && [ -d "$W/ok_box/8" ] \
    && say "FAILPROOF rhoBox runs to its end with the stop armed" PASS \
    || say "FAILPROOF the stop fired on rhoBox" FAIL

# ---- the tutorial as shipped: a turbulence field injected, then the healthy run ---------------------------
if stage "$W/turb" yes 0.5; then
    ( cd "$W/turb" && BRAE_TEST_CORRUPT=k:nan:3 BRAE_OF_LOG=1 "$BRAE" -case "$W/turb" > log 2>&1 ); t_rc=$?
    stopped "TURBULENCE k=NaN after iteration 3 (squareBend):" "$W/turb" "$W/turb/log" 4 "$t_rc"
else
    say "TURBULENCE could not stage squareBend" FAIL
fi
if stage "$W/ok" yes 0.5; then
    ( cd "$W/ok" && BRAE_OF_LOG=1 "$BRAE" -case "$W/ok" > log.ok 2>&1 ); ok_rc=$?
    ok_reached=$(its "$W/ok/log.ok")
    [ "$ok_rc" -eq 0 ] && [ "$ok_reached" -eq "$N" ] && [ -d "$W/ok/$N" ] \
        && say "FAILPROOF the healthy tutorial runs all $N iterations and writes $N/" PASS \
        || say "FAILPROOF the stop fired on a healthy run (exit $ok_rc, $ok_reached iterations)" FAIL
else
    say "FAILPROOF could not stage the healthy case" FAIL
fi

[ "$fail" -eq 0 ] && echo "rho_default_nonfinite_refusal: PASS" || echo "rho_default_nonfinite_refusal: FAIL"
exit "$fail"
