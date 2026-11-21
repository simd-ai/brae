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
# OpenFOAM is NOT assumed to abort on a NaN: on aarch64 (the GB10, the GH200) v2412 has no floating-point
# trap and prints `nan` residuals and runs on. What it does on this case, measured on the GB10: its first
# non-finite residual at iteration 18, then a FatalError from the thermo inversion at 19 ("Negative
# initial temperature T0", thermoI.H:57). The gate reads both from OpenFOAM's own log each time it runs.
#
#   ORACLE      OpenFOAM, on the staged squareBend (transonic no, 0.1 kg/s), prints a non-finite value or
#               stops (non-zero exit) within N iterations; S_OF is the first iteration at which it does
#               either. If it does neither, the reproducer stopped reproducing and the gate says so.
#   ARM         the default driver stops, says the run diverged and names the residual, at or before S_OF,
#               and writes no time directory (so no non-finite field).
#   CONTROL     BRAE_ALLOW_NONFINITE=1 restores the old behaviour: all N iterations, and N/ holds a NaN.
#   RESIDUAL    rhoBox with BRAE_TEST_CORRUPT=p:inf:3 (one cell of p set to Inf after iteration 3's check):
#               the run stops at iteration 4 on a residual, having written nothing.
#   WRITE       rhoBox with BRAE_TEST_CORRUPT=U:nan:<last> (after the last iteration's residual check, so
#               no residual ever sees it): the final write refuses, names U, and leaves no directory.
#   FAIL-PROOF  squareBend as shipped (transonic yes, 0.5 kg/s) runs all N iterations with the stop armed,
#               and rhoBox runs to its end: the stop does not fire on a healthy run.
# Labels: slow (squareBend is 112,000 cells; OpenFOAM runs to its failure). Honours OFBASHRC.
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

# stage <dir> <transonic yes|no> <massFlowRate>   (as rho_nonfinite_refusal.sh)
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

echo "rho_default_nonfinite_refusal: a diverged default-driver run must stop ($N iterations, squareBend)"

# ---- the diverging case ---------------------------------------------------------------------------------
stage "$W/div" no 0.1 || { echo "SKIP: blockMesh failed"; exit 77; }
cp -r "$W/div" "$W/div_of"
( cd "$W/div_of" && nice -n 10 rhoSimpleFoam > log.of 2>&1 ); of_rc=$?
of_reached=$(its "$W/div_of/log.of")
# the first iteration whose OpenFOAM log lines carry a nan/inf (residuals, continuity), if any
of_nan=$(awk '/^Time = /{t=$3} /(^|[ =(])-?(nan|inf)([ ,)]|$)/{print t; exit}' "$W/div_of/log.of")
S_OF=""
[ -n "$of_nan" ] && S_OF=$of_nan
[ "$of_rc" -ne 0 ] && { [ -z "$S_OF" ] || [ "$of_reached" -lt "$S_OF" ]; } && S_OF=$of_reached
if [ -n "$S_OF" ]; then
    say "ORACLE   OpenFOAM: first non-finite at ${of_nan:-none}, exit $of_rc after $of_reached its -> S_OF $S_OF" PASS
else
    say "ORACLE   OpenFOAM neither printed a non-finite value nor stopped -- the reproducer no longer diverges" FAIL
fi
grep -m1 'FOAM FATAL' -A1 "$W/div_of/log.of" | tail -1 | sed 's/^/           OpenFOAM: /'
rm -rf "$W/div_of"

cp -r "$W/div" "$W/div_brae"
( cd "$W/div_brae" && BRAE_OF_LOG=1 "$BRAE" -case "$W/div_brae" > log.brae 2>&1 ); b_rc=$?
b_reached=$(its "$W/div_brae/log.brae")
msg=$(grep -m1 'solution diverged' "$W/div_brae/log.brae")
if [ "$b_rc" -ne 0 ] && echo "$msg" | grep -qE "invalid residual for [A-Za-z]+|non-finite continuity"; then
    say "ARM      the default driver stops and names it (exit $b_rc after $b_reached iterations)" PASS
    echo "           ${msg:0:150}"
else
    say "ARM      the default driver did not stop on it (exit $b_rc after $b_reached iterations)" FAIL
fi
[ "$b_reached" -gt 0 ] && [ -n "$S_OF" ] && [ "$b_reached" -le "$S_OF" ] \
    && say "ARM      it stops at iteration $b_reached, at or before OpenFOAM's $S_OF" PASS \
    || say "ARM      expected a stop at or before iteration ${S_OF:-?}, got $b_reached" FAIL
[ "$(written "$W/div_brae")" -eq 0 ] && say "ARM      no time directory written" PASS \
    || say "ARM      it wrote $(written "$W/div_brae") time director(y|ies)" FAIL
rm -rf "$W/div_brae"

mv "$W/div" "$W/div_ctl"
( cd "$W/div_ctl" && BRAE_ALLOW_NONFINITE=1 BRAE_OF_LOG=1 "$BRAE" -case "$W/div_ctl" > log.ctl 2>&1 ); c_rc=$?
c_reached=$(its "$W/div_ctl/log.ctl")
if [ "$c_rc" -eq 0 ] && [ "$c_reached" -eq "$N" ] && grep -qiw nan "$W/div_ctl/$N/U" 2>/dev/null; then
    say "CONTROL  BRAE_ALLOW_NONFINITE=1: all $N iterations and a NaN in $N/U (the old behaviour)" PASS
else
    say "CONTROL  expected $N iterations and a NaN field with the stop off (rc $c_rc, $c_reached its)" FAIL
fi
rm -rf "$W/div_ctl"

# ---- the two stops, each on what it claims to read (rhoBox, seconds) -----------------------------------
stagebox "$W/res" 8 8   # the only write at 8: the residual check, not the write-time scan, has to stop it
( cd "$W/res" && BRAE_TEST_CORRUPT=p:inf:3 BRAE_OF_LOG=1 "$BRAE" -case "$W/res" > log 2>&1 ); r_rc=$?
r_reached=$(its "$W/res/log")
if [ "$r_rc" -ne 0 ] && grep -q "solution diverged -- invalid residual" "$W/res/log" && [ "$r_reached" -eq 4 ] \
   && [ ! -d "$W/res/4" ]; then
    say "RESIDUAL p=Inf after iteration 3: stopped at iteration 4 on a residual, 4/ not written" PASS
else
    say "RESIDUAL expected a residual stop at iteration 4 (rc $r_rc, $r_reached its)" FAIL
    grep -v NOTICE "$W/res/log" | tail -2 | sed 's/^/           /'
fi
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

# ---- FAIL-PROOF: the tutorial as shipped -----------------------------------------------------------------
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
