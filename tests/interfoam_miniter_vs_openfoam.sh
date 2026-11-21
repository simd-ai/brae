#!/usr/bin/env bash
# `minIter` on p_rgh, p_rghFinal, pcorr and pcorrFinal: brae's interFoam against REAL OpenFOAM's, solve by solve.
#
# WHAT OPENFOAM DOES. lduMatrix::solver::readControls reads `minIter` (default 0) from every solver entry
# (lduMatrixSolver.C:195-205), and PCG iterates while nIterations < minIter even with its residual already under
# tolerance: `minIter_ > 0 || !converged` enters the loop, `... || nIterations < minIter_` keeps it (PCG.C:127-200).
# WHAT brae DID (main 9b7b1e58): neither the pressure entries' reader nor any p_rgh/pcorr solve call took it --
# only GAMG entries carried it (readGamgControls) -- so the solves stopped where the tolerance alone stopped them.
# No shipped interFoam tutorial names minIter for p_rgh or pcorr (it is named for alpha and U|k|epsilon|omega
# only), so the tutorials are untouched; this gate stages it.
#
# PROFILES, both on laminar/damBreak (blockMesh, setFields), 10 fixed steps of 1e-3, written at writePrecision 15:
#   prgh   p_rgh `minIter 20` (p_rghFinal takes it through `$p_rgh`), "pcorr.*" `minIter 3`. The non-final
#          solves stop at relTol 0.05 after 1-17 iterations and take 20; pcorr's one solve (initCorrectPhi on a
#          case at rest) is an exactly zero system, where OpenFOAM's PCG breaks on checkSingularity with 0
#          iterations (brae's host pcg has no singularity test, so the call site passes no floor there).
#   pcorr  a start that is not divergence-free (U internalField (0.1 0.05 0)) and "pcorr.*" at tolerance 1e-3 with
#          `minIter 80`: the solve stops at 56 iterations without it, takes 80 with it.
# CONTROLS (OpenFOAM against itself, asserted): each profile's minIter must move OpenFOAM's own fields by more than
# 1e3 x the field bound below, and change its iteration counts -- else a brae ignoring minIter would pass.
#
# BOUNDS, DECLARED BEFORE THE FIRST RUN:
#   iteration counts -- every p_rgh and pcorr solve, in order -- EXACTLY OpenFOAM's, on the host arm and on the
#     device arm with the case's own solver (BRAE_PRESSURE_CASE_SOLVER=1, as CI runs every test);
#   fields alpha.water, U, p_rgh at the last step: relative (foam_time_compare.py, |a-b| over the file's largest
#     |a|) 1e-10. brae's host and device loops hold damBreak to OpenFOAM at 1e-14..3e-12 in
#     tests/interfoam_dambreak_vs_openfoam.sh at fixed steps; 1e-10 leaves that two decades, and sits more than
#     four decades under the smallest control (OpenFOAM's own move from minIter, >= 4e-06 measured).
#   the default device path (the AMG-PCG, BRAE_PRESSURE_CASE_SOLVER unset) is another Krylov method whose counts
#     are not OpenFOAM's (tests/interfoam_write/amg_pcg holds its fields); here it is held to the FLOOR only:
#     every p_rgh solve of the prgh profile and the pcorr profile's pcorr solve take at least minIter iterations.
#
# NEGATIVE CONTROL: the binary of main 9b7b1e58 fails both profiles' counts and fields (measured: the pcorr
# profile's U 4.4e-05 from OpenFOAM, which is OpenFOAM's own control distance).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_interFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
CMP="$ROOT/tools/foam_time_compare.py"
BOUND=1e-10

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

# stage <dir> <profile> <withMinIter 0|1>
stage()
{
    rm -rf "$1"
    cp -r "$SRC" "$1" || return 1
    rm -rf "$1"/0 "$1"/[1-9]* "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    ( cd "$1" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) || return 1
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
d, profile, withMin = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
c = d + '/system/controlDict'
s = open(c).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', '1e-3'), ('endTime', '0.01'), ('writeControl', 'timeStep'),
                 ('writeInterval', '10'), ('writePrecision', '15'), ('writeFormat', 'ascii')]:
    s, n = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert n == 1, key
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSolution'
t = open(f).read()
assert re.search(r'\n    p_rghFinal\n    \{\n        \$p_rgh;', t), 'p_rghFinal no longer inherits $p_rgh'
if profile == 'prgh' and withMin:
    t, n = re.subn(r'(\n    p_rgh\n    \{[^}]*relTol\s+0\.05;)', r'\1\n        minIter         20;', t, count=1)
    assert n == 1
    t, n = re.subn(r'(\n    "pcorr\.\*"\n    \{[^}]*relTol\s+0;)', r'\1\n        minIter         3;', t, count=1)
    assert n == 1
if profile == 'pcorr':
    t, n = re.subn(r'(\n    "pcorr\.\*"\n    \{[^}]*)tolerance\s+1e-5;', r'\1tolerance       1e-3;', t, count=1)
    assert n == 1
    if withMin:
        t, n = re.subn(r'(\n    "pcorr\.\*"\n    \{[^}]*relTol\s+0;)', r'\1\n        minIter         80;', t, count=1)
        assert n == 1
    u = d + '/0/U'
    us = open(u).read()
    us, n = re.subn(r'^internalField\s.*', 'internalField   uniform (0.1 0.05 0);', us, flags=re.M)
    assert n == 1
    open(u, 'w').write(us)
open(f, 'w').write(t)
PY
}

# counts <log>: "<field> <n>" per p_rgh/pcorr solve, in order (OpenFOAM's log, or brae's BRAE_PRINT_TURB_SOLVES
# lines). OpenFOAM's "solution singularity" line is a 0-iteration solve.
counts()
{
    python3 - "$1" <<'PY'
import re, sys
for line in open(sys.argv[1]):
    m = re.search(r'Solving for (p_rgh|pcorr)(.*)', line)
    if not m:
        continue
    if 'singularity' in m.group(2):
        print(m.group(1), 0)
        continue
    k = re.search(r'No Iterations (\d+)', line)
    if k:
        print(m.group(1), k.group(1))
PY
}
# by field: brae prints its p_rgh solves before its pcorr solves; OpenFOAM interleaves them
byfield() { counts "$1" | awk '{print $1, $2}' | sort -s -k1,1; }

# worst <ref> <other>: the largest relative difference over alpha.water, U and p_rgh at 0.01
worst()
{
    python3 "$CMP" "$1" "$2" 0.01 | python3 -c "
import json, sys
r = [l for l in sys.stdin if l.startswith('RESULT ')]
d = json.loads(r[-1][7:])
print(max(d['files'][k]['rel'] for k in ('0.01/alpha.water', '0.01/U', '0.01/p_rgh')) if not d['structure'] else 'structure')"
}

for profile in prgh pcorr; do
    echo "== profile $profile =="
    OF="$W/of_$profile"
    OF0="$W/of0_$profile"
    stage "$OF" "$profile" 1 && stage "$OF0" "$profile" 0 || { echo "SKIP: staging failed"; exit 77; }
    ( cd "$OF" && nice -n 10 interFoam > log.interFoam 2>&1 ) || { say "OpenFOAM runs [$profile]" FAIL; continue; }
    ( cd "$OF0" && nice -n 10 interFoam > log.interFoam 2>&1 ) || { say "OpenFOAM runs without minIter [$profile]" FAIL; continue; }
    ctl=$(worst "$OF" "$OF0")
    if [ "$(byfield "$OF/log.interFoam")" != "$(byfield "$OF0/log.interFoam")" ] \
       && python3 -c "import sys; sys.exit(0 if float('$ctl') > 1e3*$BOUND else 1)" 2>/dev/null; then
        say "control: minIter moves OpenFOAM's own counts and fields ($ctl)" ok
    else
        say "control: minIter moves OpenFOAM's own counts and fields ($ctl)" FAIL
    fi
    arm()
    {
        local name="$1" flags="$2"
        shift 2
        local C="$W/${name}_$profile"
        stage "$C" "$profile" 1 || { say "$name staging" FAIL; return; }
        # shellcheck disable=SC2086
        ( cd "$C" && env "$@" BRAE_PRINT_TURB_SOLVES=1 "$BIN" -case . $flags > log.brae 2>&1 ) \
            || { say "$name runs [$profile]" FAIL; tail -2 "$C/log.brae" | cut -c1-200; return; }
        if [ "$name" = device_amg_pcg ]; then
            # the floor: every p_rgh solve (p_rghFinal inherits the entry's minIter) and, in the pcorr profile, pcorr
            local floor=20 fld=p_rgh
            [ "$profile" = pcorr ] && floor=80 && fld=pcorr
            local low
            if [ "$profile" = prgh ]; then
                low=$(counts "$C/log.brae" | awk '$1=="p_rgh" && $2<20{bad++} END{print bad+0}')
            else
                low=$(counts "$C/log.brae" | awk '$1=="pcorr" && $2<80{bad++} END{print bad+0}')
            fi
            if [ "$low" = 0 ] && grep -q 'brae runs its AMG-preconditioned PCG' "$C/log.brae"; then
                say "$name: every $fld solve that minIter reaches takes >= $floor iterations [$profile]" ok
            else
                say "$name: every $fld solve that minIter reaches takes >= $floor iterations ($low below) [$profile]" FAIL
            fi
            return
        fi
        if [ "$(byfield "$C/log.brae")" = "$(byfield "$OF/log.interFoam")" ]; then
            say "$name: every p_rgh and pcorr iteration count is OpenFOAM's ($(counts "$C/log.brae" | wc -l) solves) [$profile]" ok
        else
            say "$name: every p_rgh and pcorr iteration count is OpenFOAM's [$profile]" FAIL
            diff <(byfield "$OF/log.interFoam") <(byfield "$C/log.brae") | head -6 | sed 's/^/        /'
        fi
        local w
        w=$(worst "$OF" "$C")
        if python3 -c "import sys; sys.exit(0 if float('$w') <= $BOUND else 1)" 2>/dev/null; then
            say "$name: alpha, U, p_rgh within $BOUND of OpenFOAM ($w) [$profile]" ok
        else
            say "$name: alpha, U, p_rgh within $BOUND of OpenFOAM ($w) [$profile]" FAIL
        fi
    }
    arm host ""
    if [ $HAVE_GPU = 1 ]; then
        arm device_case_solver "-device" BRAE_PRESSURE_CASE_SOLVER=1
        arm device_amg_pcg "-device" -u BRAE_PRESSURE_CASE_SOLVER
    fi
done
[ $HAVE_GPU = 1 ] || echo "  (no GPU: the -device arms are not run)"

echo "interfoam_miniter_vs_openfoam: $fails failure(s)"
[ $fails -eq 0 ]
