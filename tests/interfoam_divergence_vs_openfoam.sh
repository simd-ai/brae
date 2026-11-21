#!/usr/bin/env bash
# The DIVERGENCE STOP of brae's interFoam against REAL OpenFOAM's interFoam, on laminar/damBreak forced to blow up.
#
# WHAT OPENFOAM DOES. interFoam has no divergence check of its own. Where the FPU traps (x86) it stops on the first
# floating-point exception (etc/controlDict trapFpe, sigFpe.C). On aarch64 -- this machine and the GH200 -- nothing
# traps, and what happens next depends on the case: OpenFOAM can carry the NaN to endTime, write it and exit 0 (the
# pimpleFoam agent measured that), or stop on reading a NaN solver residual back from its solver-performance
# dictionary ("Wrong token type - expected scalar value, found ... word 'nan'", Scalar.C:169), as it does on THIS
# fixture on the GB10 (exit 1 in its 8th step). So this gate does NOT assert an OpenFOAM abort. It finds the step in
# which OpenFOAM FIRST PRINTS a non-finite value (`nan` or `inf` in its log: a residual, a Courant number, the phase
# fraction) and holds brae to that, and reports OpenFOAM's exit code beside it.
#
# WHAT brae DID (main 9b7b1e58), MEASURED on the fixture below: both loops ran to endTime, wrote NaN fields at
# every write time and printed `End:` with exit 0 (host: alpha [nan, nan] from step 8; device: from step 6).
# Now each loop checks the Courant numbers at the top of a step and the solved fields -- alpha, U, p_rgh, phi
# and the closure's -- at its end, before runTime.write(), and stops by name (divergenceMessage,
# inter_driver_cpp.cuh). The device check is one launch over those arrays and one small copy a step.
#
# THE FIXTURE: damBreak as shipped (blockMesh, setFields), `MULESCorr no` (explicit MULES), a fixed deltaT of
# 0.01 -- a Courant number in the hundreds after two steps -- endTime 0.2, a write every 2 steps. MEASURED on the
# GB10: OpenFOAM first prints a non-finite value in its 8th step, having written 0.02, 0.04 and 0.06. The shipped
# `MULESCorr yes` at the same deltaT is the HEALTHY CONTROL: it does not blow up in either code, and must run to
# endTime.
#
# DECLARED BEFORE THE FIRST RUN (what passes):
#   OpenFOAM:  prints a non-finite value in some step N before endTime (else the fixture does not diverge and the
#              gate is void); its exit code is REPORTED, not asserted.
#   each brae arm (host; -device with the case's own pressure solver, as CI runs it; -device with the AMG-PCG,
#   the default path):
#     exits non-zero, naming "the solution diverged", in step N or earlier;
#     writes no file holding a non-finite value, and no time directory at or after step N's time;
#     the host arm (OpenFOAM's own linear solvers, round-off apart from it until the blow-up) writes exactly the
#     directories OpenFOAM wrote before step N.
#   the healthy control: OpenFOAM and every brae arm reach endTime, exit 0.
# Numbers of the diverging fields are not compared: they are 1e+42 and beyond.
#
# NEGATIVE CONTROL: the binary of main 9b7b1e58 fails every brae assertion above (exit 0, NaN written).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_interFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh setFields interFoam; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done
HAVE_GPU=0
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then HAVE_GPU=1; fi

fails=0
say() { printf "  %-5s %s\n" "$2:" "$1"; [ "$2" = ok ] || fails=$((fails+1)); }

# stage <dir> <MULESCorr yes|no>
stage()
{
    rm -rf "$1"
    cp -r "$SRC" "$1" || return 1
    rm -rf "$1"/0 "$1"/[1-9]* "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    ( cd "$1" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) || return 1
    python3 - "$1" "$2" <<'PY'
import re, sys
d, mules = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'
s = open(c).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', '0.01'), ('endTime', '0.2'), ('writeControl', 'timeStep'),
                 ('writeInterval', '2'), ('writePrecision', '15'), ('writeFormat', 'ascii')]:
    s, n = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert n == 1, key
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'
t = open(f).read()
t, n = re.subn(r'MULESCorr\s+\w+;', 'MULESCorr       %s;' % mules, t)
assert n == 1
open(f, 'w').write(t)
PY
}

# timedirs <case>: the written time directories, sorted
timedirs() { ( cd "$1" && ls -d [0-9]* 2>/dev/null | grep -v '^0$' | grep -E '^[0-9.e+-]+$' | sort -g | tr '\n' ' ' ); }
# nonfinite <case>: files under the written time directories holding nan or inf
nonfinite()
{
    local d
    for d in $(timedirs "$1"); do
        grep -rlwiE 'nan|-?inf|-?infinity' "$1/$d" 2>/dev/null
    done
}
# laststep <log> <pattern>: the number of step lines
steps() { grep -cE "$2" "$1"; }

echo "== interFoam divergence stop vs OpenFOAM (damBreak, MULESCorr no, deltaT 0.01) =="
OF="$W/of"
stage "$OF" no || { echo "SKIP: staging failed"; exit 77; }
( cd "$OF" && nice -n 10 interFoam > log.interFoam 2>&1 ); ofrc=$?
# OpenFOAM's first step with a non-finite value in its log, counted by `Time =` lines (0: none)
ofNaN=$(python3 - "$OF/log.interFoam" <<'PY'
import re, sys
step = 0
for line in open(sys.argv[1], errors='replace'):
    if line.startswith('Time = '):
        step += 1
    elif step and re.search(r'(?i)(?<![A-Za-z])(-?nan|-?inf)(?![A-Za-z])', line):
        print(step)
        sys.exit(0)
print(0)
PY
)
ofSteps=$(steps "$OF/log.interFoam" '^Time = ')
# the time OpenFOAM's step N reached: N fixed steps of 0.01
ofNaNTime=$(python3 -c "print('%.10g' % (0.01*int('$ofNaN')))")
ofDirs=$(timedirs "$OF")
ofBefore=$(for d in $ofDirs; do python3 -c "import sys; sys.exit(0 if float('$d') < float('$ofNaNTime') - 1e-12 else 1)" && printf '%s ' "$d"; done)
echo "  OpenFOAM: first non-finite value printed in step ${ofNaN}; exit $ofrc after $ofSteps step(s) (reported); wrote [ $ofDirs], before step $ofNaN [ $ofBefore]"
if [ "$ofNaN" -gt 0 ]; then
    say "OpenFOAM prints a non-finite value in step $ofNaN (the fixture diverges)" ok
else
    say "OpenFOAM prints a non-finite value before endTime (the fixture diverges)" FAIL
    echo "interfoam_divergence_vs_openfoam: the fixture no longer diverges in OpenFOAM; nothing can be held"
    exit 1
fi

# brae_arm <name> <flags> <env...>
brae_arm()
{
    local name="$1" flags="$2"
    shift 2
    local C="$W/$name"
    stage "$C" no || { say "$name: staging" FAIL; return; }
    # shellcheck disable=SC2086
    ( cd "$C" && env "$@" "$BIN" -case . $flags > log.brae 2>&1 ); local rc=$?
    local stop dirs bad
    stop=$(sed -n 's/.*the solution diverged in time step \([0-9]*\) .*/\1/p' "$C/log.brae" | head -1)
    dirs=$(timedirs "$C")
    bad=$(nonfinite "$C")
    echo "  brae $name: exit $rc, stopped in step ${stop:-none}, wrote [ $dirs]"
    if [ $rc -ne 0 ] && grep -q "the solution diverged" "$C/log.brae" && ! grep -q '^End: t' "$C/log.brae"; then
        say "$name stops by name with a non-zero exit" ok
    else
        say "$name stops by name with a non-zero exit" FAIL
        tail -2 "$C/log.brae" | cut -c1-200 | sed 's/^/        /'
    fi
    if [ -z "$bad" ]; then
        say "$name writes no non-finite value" ok
    else
        say "$name writes no non-finite value ($(echo "$bad" | wc -l) file(s), e.g. $(echo "$bad" | head -1))" FAIL
    fi
    if [ -n "$stop" ] && [ "$stop" -le "$ofNaN" ]; then
        say "$name stops in OpenFOAM's first non-finite step or earlier ($stop <= $ofNaN)" ok
    else
        say "$name stops in OpenFOAM's first non-finite step or earlier (${stop:-none} <= $ofNaN)" FAIL
    fi
    local late=""
    for d in $dirs; do
        python3 -c "import sys; sys.exit(0 if float('$d') < float('$ofNaNTime') - 1e-12 else 1)" || late="$late $d"
    done
    if [ -z "$late" ]; then
        say "$name writes no time directory at or after step $ofNaN's time" ok
    else
        say "$name writes no time directory at or after step $ofNaN's time (late:$late)" FAIL
    fi
    if [ "$name" = host ]; then
        if [ "$dirs" = "$ofBefore" ]; then
            say "host writes exactly OpenFOAM's directories before step $ofNaN" ok
        else
            say "host writes exactly OpenFOAM's directories before step $ofNaN ([ $dirs] against [ $ofBefore])" FAIL
        fi
    fi
}

brae_arm host ""
if [ $HAVE_GPU = 1 ]; then
    brae_arm device_case_solver "-device" BRAE_PRESSURE_CASE_SOLVER=1
    brae_arm device_amg_pcg "-device" -u BRAE_PRESSURE_CASE_SOLVER
else
    echo "  (no GPU: the -device arms are not run)"
fi

echo "-- the healthy control: MULESCorr yes, the same deltaT, to endTime"
OFH="$W/of_healthy"
stage "$OFH" yes || { echo "SKIP: staging failed"; exit 77; }
( cd "$OFH" && nice -n 10 interFoam > log.interFoam 2>&1 ); rc=$?
if [ $rc -eq 0 ] && grep -q '^End' "$OFH/log.interFoam"; then
    say "OpenFOAM runs the healthy control to endTime" ok
else
    say "OpenFOAM runs the healthy control to endTime" FAIL
fi
healthy_arm()
{
    local name="$1" flags="$2"
    shift 2
    local C="$W/healthy_$name"
    stage "$C" yes || { say "healthy $name: staging" FAIL; return; }
    # shellcheck disable=SC2086
    ( cd "$C" && env "$@" "$BIN" -case . $flags > log.brae 2>&1 ); local rc=$?
    if [ $rc -eq 0 ] && grep -q '^End: t = 0.2' "$C/log.brae" && [ -z "$(nonfinite "$C")" ] \
       && [ "$(timedirs "$C")" = "$(timedirs "$OFH")" ]; then
        say "healthy $name runs to endTime with OpenFOAM's directories" ok
    else
        say "healthy $name runs to endTime with OpenFOAM's directories" FAIL
        tail -2 "$C/log.brae" | cut -c1-200 | sed 's/^/        /'
    fi
}
healthy_arm host ""
if [ $HAVE_GPU = 1 ]; then
    healthy_arm device_case_solver "-device" BRAE_PRESSURE_CASE_SOLVER=1
    healthy_arm device_amg_pcg "-device" -u BRAE_PRESSURE_CASE_SOLVER
fi

echo "interfoam_divergence_vs_openfoam: $fails failure(s)"
[ $fails -eq 0 ]
