#!/usr/bin/env bash
# brae_interFoam's time directories against real OpenFOAM's: WHEN it writes, WHAT it writes, and that the
# writing changes nothing.
#
# OpenFOAM writes at one place, runTime.write() after the PIMPLE loop (interFoam.C:175), when
# Time::operator++ marked the step (Time.C:1103-1130); there is no end-of-run write. What goes in is every
# AUTO_WRITE object -- alpha.water, U, p_rgh, p, phi, alphaPhi0.water, the closure's fields -- each patch
# through its own condition's write(), plus uniform/time, uniform/cumulativeContErr and the function
# objects' state file. brae_interFoam wrote nothing before this gate: a case ran to its end and left 0/.
#
# The writer is inter_writer_cpp.cu; the comparison is tools/foam_time_compare.py (structure exactly,
# values after expanding `uniform`, since whether a list is written uniform is a property of the values).
#
# Staging: laminar/damBreak as shipped, blockMesh + setFields, `functions {}` (brae runs no function
# objects and says so), writePrecision 17 so a value gap is the solver's and not the rounding's, and
# endTime 0.12 -- NOT a write time under `adjustable 0.05`, so OpenFOAM writes {0.05, 0.1} and no 0.12.
#
#   ARM A  brae writes exactly OpenFOAM's directories, and nothing at endTime (the end-of-run fail-proof).
#   ARM B  each holds OpenFOAM's file set.
#   ARM C  uniform/time: `name` and `index` exact, value/deltaT/deltaT0 at the clock's bound.
#   ARM D  every file's structure -- header, dimensions, patch order, each patch's keyword list, every word
#          -- exactly OpenFOAM's; every value within the bound measured for it below.
#   ARM E  OpenFOAM restarts from brae's 0.1: its first deltaT is the one brae stored (adjustTimeStep reads
#          uniform/time) and its cumulative continuity error continues brae's.
#   ARM F  writing does not perturb the run: `timeStep 1` against `timeStep N` leaves a byte-identical
#          final directory, host and device, laminar and RAS (the device downloads its closure to write).
#   ARM G  RAS/damBreak (kEpsilon: k, epsilon, nut, the wall functions' entries) through A-D, both arms.
#   ARM H  purgeWrite 1 keeps what OpenFOAM keeps.
#   ARM S  a sub-cycled alpha (mixerVessel2D) writes alpha.water_0, OpenFOAM's old time, at every step.
#   ARM W  eight tutorials whose conditions now write -- capillaryRise, weirOverflow, angledDuct,
#          damBreakLeakage, damBreakPorousBaffle, damBreakPermeable, nozzleFlow2D, eulerianInjection --
#          against OpenFOAM at pinned solves, host and device; the old level's restore rule witnessed.
#   ARM R  a file brae cannot write yet (stokesI's wave-model state) is named at startup, and the run
#          stops at its first write time with nothing written.
#   Every arm runs on the host loop and on `-device` when a GPU is present.
#
#   CONTROLS, each asserted red:
#     0/ handed in as brae's 0.05 fails D.
#     BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT=1 (the gradient the dictionary constructor leaves, the
#       shared writer's defect) fails D on p_rgh -- after asserting OpenFOAM's gradient is non-zero, so
#       the fixture can witness it.
#     brae's alpha.water with its inletValue line deleted fails D (the keyword comparison is live).
#     brae's 0.1 with uniform/time deleted moves E's first deltaT (E witnesses the stored deltaT).
#     R's case with no write time before endTime runs to its end: the refusal is of the output only.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_interFoam"
CMP="$ROOT/tools/foam_time_compare.py"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
LAM="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
RAS="$TUT/multiphase/interFoam/RAS/damBreak/damBreak"
MV="$TUT/multiphase/interFoam/laminar/mixerVessel2D"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$LAM" ]      || { echo "SKIP: damBreak tutorial not found at $LAM"; exit 77; }
[ -d "$RAS" ]      || { echo "SKIP: RAS damBreak tutorial not found at $RAS"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

GPU=0
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then GPU=1; fi
ARMS="host"
[ $GPU -eq 1 ] && ARMS="host device"

fail=0
say() { printf '  %-86s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# BOUNDS, worst |brae - OpenFOAM| over a file / the largest |OpenFOAM| value in it, at writePrecision 17.
# MEASURED 2026-09-30, host / device:
#   laminar 0.05  U 6.2e-14 / 9.3e-14, phi 8.8e-14 / 1.3e-13, cumulativeContErr 1.1e-11 / 2.6e-12
#   laminar 0.1   U 3.8e-14 / 2.1e-14, phi 2.9e-14 / 1.6e-14, cumulativeContErr 1.8e-12 / 6.9e-13
#   RAS 0.05      k 2.5e-12 / 3.5e-12, epsilon 2.3e-12 / 3.4e-12, U 9.0e-13 / 5.3e-13
#   waterChannel  U 2.2e-11 / 2.2e-11, phi 2.0e-11 / 2.0e-11, p 3.6e-12, nut 2.1e-12 / 2.4e-12
#   every other file below these; uniform/time 0 on every arm (the damBreak clock gate holds deltaT to
#   1e-13 per step). cumulativeContErr is a running sum of signed errors, relative to its own size.
# One decade above the worst. The first run of this gate put waterChannel's cumulativeContErr at 1.2e-07:
# brae did not count initCorrectPhi.H's continuityErrs.H, a 2.885e-10 term a 1e-08 bound would have hidden.
BOUND_FIELDS=${BOUND_FIELDS:-2e-10}
BOUND_TIME=${BOUND_TIME:-1e-13}


stage()   # stage <src> <dir> <endTime> <writeControl> <writeInterval> <purgeWrite> [key=value ...]
{
    local src="$1" d="$2"
    cp -r "$src" "$d" || return 1
    rm -rf "$d"/[1-9]* "$d"/0 "$d"/processor* "$d"/log.*
    cp -r "$d/0.orig" "$d/0"
    # the tutorial's own mesh script when it ships one (waterChannel's extrusions, mixerVessel2D's m4,
    # topoSet and setsToZones), blockMesh otherwise -- serial tools either way
    if [ -x "$d/Allrun.pre" ]; then
        ( cd "$d" && ./Allrun.pre > log.allrunpre 2>&1 ) || return 1
        [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    else
        ( cd "$d" && blockMesh > log.blockMesh 2>&1 ) || return 1
    fi
    ( cd "$d" && setFields > log.setFields 2>&1 ) || return 1
    python3 - "$d/system/controlDict" "$3" "$4" "$5" "$6" "${@:7}" <<'PY'
import re, sys
c, end, wc, wi, pw = sys.argv[1:6]
extra = [kv.split('=', 1) for kv in sys.argv[6:]]
s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}', '\nfunctions\n{\n}', s, flags=re.S)
for k, v in [('endTime', end), ('writeControl', wc), ('writeInterval', wi), ('purgeWrite', pw),
             ('writePrecision', '17'), ('writeFormat', 'ascii'), ('writeCompression', 'off')] + extra:
    if re.search(r'^%s\s' % k, s, flags=re.M):
        s = re.sub(r'^%s\s.*' % k, '%-16s%s;' % (k, v), s, flags=re.M)
    else:
        s = s.replace('\nfunctions\n', '\n%-16s%s;\nfunctions\n' % (k, v), 1)
open(c, 'w').write(s)
PY
}
timedirs() { ( cd "$1" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9.e+-]+$' | grep -vE '^0$' | sort -g | tr '\n' ' ' ); }
filesets() { ( cd "$1/$2" && find . -type f | sed 's/\.gz$//' | sort | tr '\n' ' ' ); }
runof()    { ( cd "$1" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam did not run in $1"; tail -5 "$1/log.interFoam"; exit 1; }; }
runbrae()  # runbrae <dir> <arm> [env...]
{
    local d="$1" arm="$2"; shift 2
    local flag=""
    [ "$arm" = device ] && flag="-device"
    ( cd "$d" && env "$@" "$BIN" -case . $flag > log.brae 2>&1 ) \
        || { echo "FAIL: brae_interFoam ($arm) did not run in $d"; tail -5 "$d/log.brae"; exit 1; }
}
# field bounds from the comparer's RESULT line: prints the worst file, fails structure or any rel > bound
# [ofLog]: cumulativeContErr is dt*weightedAverage(div(phi), V) summed per corrector -- a signed sum of
# face fluxes, zero in exact arithmetic on a closed domain and at rounding once the solves are pinned, so
# a bound relative to its own value measures nothing. Its scale is the size of what cancels: the mean
# Courant number OpenFOAM prints IS 0.5*dt*sum|phi|/sum(V) (CourantNo.H), so a corrector's term is
# 2*Co_mean in magnitude. The floor is that total, sum over steps of nCorrectors*2*Co_mean, times the
# larger of 10 eps and the relative gap the FIELDS of the same comparison show -- a continuity error that
# agrees as well as the fluxes it is summed from. MEASURED on the eight W cases at pinned solves: 0.0 to
# 3.6 eps for the cases whose fields agree to rounding; angledDuct (fields 1.3e-09) and damBreakLeakage
# (2.4e-07, the column at rest) needed 8e+03 and 3.8e+06 eps, inside their field gaps. The floor this
# replaced, 1e-14 of OpenFOAM's `sum local` total, was 1e-28 at pinned solves: below every rounding.
# LIMIT, stated: where OpenFOAM's own value is itself at rounding (arms S and W at pinned solves) the floor
# exceeds it, so a dropped accumulation would pass there. Arms D witness that class -- damBreak's 4.6e-04
# and waterChannel's -2.3e-03 are compared relative, with no floor, and caught the missing initCorrectPhi
# term (2.885e-10).
judge()   # judge <label> <resultFile> <bound> [ofLog]
{
    python3 - "$@" <<'PY'
import json, re, sys
label, path, bound = sys.argv[1], sys.argv[2], float(sys.argv[3])
r = json.loads([l for l in open(path) if l.startswith('RESULT ')][-1][7:])
floor = 0.0
if len(sys.argv) > 4:
    log = open(sys.argv[4]).read()
    coScale = 0.0
    for chunk in log.split('\nTime = '):
        co = re.search(r'Courant Number mean: (\S+)', chunk)
        coScale += len(re.findall(r'sum local', chunk)) * 2.0 * (float(co.group(1)) if co else 0.0)
    fieldsRel = max([v['rel'] for k, v in r['files'].items()
                     if not k.endswith('cumulativeContErr') and 'functionObject' not in k] + [0.0])
    floor = max(10 * 2.220446049250313e-16, fieldsRel) * coScale
worst = max(r['files'].items(), key=lambda kv: kv[1]['rel'])
over = [k for k, v in r['files'].items()
        if v['rel'] > bound and not (k.endswith('uniform/cumulativeContErr') and v['abs'] <= floor)]
if floor > 0:
    for k, v in r['files'].items():
        if k.endswith('uniform/cumulativeContErr'):
            print('      %s: |brae - OpenFOAM| %.3e against the rounding floor %.3e' % (k, v['abs'], floor))
print('      %s: %d structure failures; worst %s at %.3e (bound %.0e)'
      % (label, r['structure'], worst[0], worst[1]['rel'], bound))
for k in over:
    print('      over the bound: %s %.3e' % (k, r['files'][k]['rel']))
sys.exit(0 if r['structure'] == 0 and not over else 1)
PY
}

# ---------------------------------------------------------------------------------------------------
# A-D: laminar damBreak, adjustable 0.05 to endTime 0.12
stage "$LAM" "$W/of"  0.12 adjustable 0.05 0 || { echo "SKIP: staging failed"; exit 77; }
runof "$W/of"
o=$(timedirs "$W/of")
[ "$o" = "0.05 0.1 " ] && say "premise: OpenFOAM writes {0.05, 0.1} and nothing at endTime 0.12" ok \
                       || { say "premise: OpenFOAM writes {0.05, 0.1} and nothing at endTime 0.12 [$o]" FAIL; exit 1; }

# the fixture must witness: a developed flow, and an atmosphere with inflow AND outflow faces
python3 - "$W/of/0.1" <<'PY' && say "fixture witnesses: max|U| > 0.1, atmosphere phi both signs" ok \
                             || say "fixture witnesses: max|U| > 0.1, atmosphere phi both signs" FAIL
import re, sys
d = sys.argv[1]
U = open(d + '/U').read()
uint = U[U.find('internalField'):U.find('boundaryField')]
mags = [sum(float(c)**2 for c in m.split())**0.5 for m in re.findall(r'\(([^()]*)\)', uint)]
phi = open(d + '/phi').read()
a = phi.find('atmosphere', phi.find('boundaryField'))
seg = phi[phi.find('(', a) + 1:phi.find(')', a)]
vals = [float(x) for x in seg.split()]
print('      max|U| %.3f, atmosphere phi in [%.3e, %.3e]' % (max(mags), min(vals), max(vals)))
sys.exit(0 if max(mags) > 0.1 and min(vals) < 0 < max(vals) else 1)
PY

for arm in $ARMS; do
    stage "$LAM" "$W/br_$arm" 0.12 adjustable 0.05 0 || exit 1
    runbrae "$W/br_$arm" "$arm"
    b=$(timedirs "$W/br_$arm")
    [ "$b" = "$o" ] && say "ARM A  [$arm] brae writes exactly OpenFOAM's directories, none at endTime [$b]" ok \
                    || say "ARM A  [$arm] brae writes exactly OpenFOAM's directories, none at endTime [$b]" FAIL
    for t in 0.05 0.1; do
        [ "$(filesets "$W/of" $t)" = "$(filesets "$W/br_$arm" $t)" ] \
            && say "ARM B  [$arm] $t/ holds OpenFOAM's file set" ok \
            || { say "ARM B  [$arm] $t/ holds OpenFOAM's file set" FAIL; echo "      OF:   $(filesets "$W/of" $t)"; echo "      brae: $(filesets "$W/br_$arm" $t)"; }
    done
    python3 "$CMP" "$W/of" "$W/br_$arm" 0.05 0.1 > "$W/cmp_$arm.txt" 2>&1
    python3 - "$W/cmp_$arm.txt" "$BOUND_TIME" <<'PY' && say "ARM C  [$arm] uniform/time: name, index exact; value, deltaT, deltaT0 within $BOUND_TIME" ok \
                                                   || say "ARM C  [$arm] uniform/time: name, index exact; value, deltaT, deltaT0 within $BOUND_TIME" FAIL
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
bad = 0
for t in ('0.05', '0.1'):
    e = r['files'].get(t + '/uniform/time')
    print('      %s/uniform/time: structure %s, worst rel %.3e' % (t, e and e['structure'], e['rel'] if e else -1))
    bad += (not e) or (not e['structure']) or e['rel'] > float(sys.argv[2])
sys.exit(1 if bad else 0)
PY
    judge "$arm" "$W/cmp_$arm.txt" "$BOUND_FIELDS" \
        && say "ARM D  [$arm] every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" ok \
        || { say "ARM D  [$arm] every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" FAIL; grep -v RESULT "$W/cmp_$arm.txt" | grep -B1 "^      " | head -20; }
done

# CONTROL: 0/ handed in as brae's 0.05
mkdir -p "$W/ctl0"; cp -r "$W/of/0" "$W/ctl0/0.05"
mkdir -p "$W/ctl0/0.05/uniform"; cp -r "$W/br_host/0.05/uniform/." "$W/ctl0/0.05/uniform/"
sed -i 's/^location .*/location    "0.05";/' "$W"/ctl0/0.05/* 2>/dev/null
python3 "$CMP" "$W/of" "$W/ctl0" 0.05 > "$W/cmp_ctl0.txt" 2>&1
judge "control 0/" "$W/cmp_ctl0.txt" "$BOUND_FIELDS" > /dev/null \
    && say "CONTROL  0/ handed in as brae's 0.05 FAILS D" FAIL \
    || say "CONTROL  0/ handed in as brae's 0.05 FAILS D" ok

# CONTROL: a deleted keyword
mkdir -p "$W/ctlk"; cp -r "$W/br_host/0.05" "$W/ctlk/"
sed -i '/inletValue/d' "$W/ctlk/0.05/alpha.water"
python3 "$CMP" "$W/of" "$W/ctlk" 0.05 > "$W/cmp_ctlk.txt" 2>&1 \
    && say "CONTROL  brae's alpha.water without its inletValue FAILS D" FAIL \
    || say "CONTROL  brae's alpha.water without its inletValue FAILS D" ok

# ---------------------------------------------------------------------------------------------------
# D on RAS/waterChannel (kOmegaSST, omega and nutk wall functions, flowRateInletVelocity), three fixed
# steps of 0.1 -- and the construction-gradient control, which damBreak cannot witness: in the p_rgh
# formulation a zeroGradient-alpha wall has snGrad(rho) = 0 and OpenFOAM's gradient there IS zero. The
# inlet's fixed alpha makes it non-zero.
WCH="$TUT/multiphase/interFoam/RAS/waterChannel"
if [ -d "$WCH" ] && command -v extrudeMesh > /dev/null 2>&1; then
    stage "$WCH" "$W/of_w" 0.3 timeStep 3 0 adjustTimeStep=no deltaT=0.1 || exit 1
    runof "$W/of_w"
    [ "$(timedirs "$W/of_w")" = "0.3 " ] || say "premise: OpenFOAM waterChannel writes {0.3} [$(timedirs "$W/of_w")]" FAIL
    python3 - "$W/of_w/0.3/p_rgh" <<'PY' && say "fixture witnesses: OpenFOAM's waterChannel p_rgh gradient is non-zero" ok \
                                     || say "fixture witnesses: OpenFOAM's waterChannel p_rgh gradient is non-zero" FAIL
import re, sys
grads = re.findall(r'gradient\s+nonuniform List<scalar>\s*\d*\s*\(([^)]*)\)', open(sys.argv[1]).read())
g = [abs(float(x)) for s in grads for x in s.split()]
print('      max|p_rgh gradient| %.3e over %d faces' % (max(g) if g else 0.0, len(g)))
sys.exit(0 if g and max(g) > 1 else 1)
PY
    for arm in $ARMS; do
        stage "$WCH" "$W/br_w_$arm" 0.3 timeStep 3 0 adjustTimeStep=no deltaT=0.1 || exit 1
        runbrae "$W/br_w_$arm" "$arm"
        python3 "$CMP" "$W/of_w" "$W/br_w_$arm" 0.3 > "$W/cmp_w_$arm.txt" 2>&1
        judge "waterChannel $arm" "$W/cmp_w_$arm.txt" "$BOUND_FIELDS" \
            && say "ARM D  [$arm] waterChannel: every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" ok \
            || { say "ARM D  [$arm] waterChannel: every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" FAIL; grep -v RESULT "$W/cmp_w_$arm.txt" | grep -B1 "^      " | head -20; }
    done
    stage "$WCH" "$W/ctlg" 0.3 timeStep 3 0 adjustTimeStep=no deltaT=0.1 || exit 1
    runbrae "$W/ctlg" host BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT=1
    python3 "$CMP" "$W/of_w" "$W/ctlg" 0.3 > "$W/cmp_ctlg.txt" 2>&1
    judge "control gradient" "$W/cmp_ctlg.txt" "$BOUND_FIELDS" | grep -q "over the bound: 0.3/p_rgh" \
        && say "CONTROL  BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT=1 puts p_rgh over the bound" ok \
        || say "CONTROL  BRAE_CONTROL_WRITE_CONSTRUCTION_GRADIENT=1 puts p_rgh over the bound" FAIL
else
    say "waterChannel arm: tutorial or extrudeMesh missing -- the gradient control cannot run" FAIL
fi

# ---------------------------------------------------------------------------------------------------
# E: OpenFOAM restarts from brae's 0.1, and from its own, to 0.11
restart()   # restart <dir> <from case> [drop uniform/time]
{
    mkdir -p "$1"
    cp -r "$W/of/constant" "$W/of/system" "$1/"
    cp -r "$2/0.1" "$1/"
    [ "${3:-}" = drop ] && rm -f "$1/0.1/uniform/time"
    sed -i 's/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         0.11;/' "$1/system/controlDict"
    ( cd "$1" && interFoam > log.interFoam 2>&1 )
}
firsts() { python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
dt = re.findall(r'^deltaT = (\S+)', s, flags=re.M)
cu = re.findall(r'cumulative = (\S+)', s)
print(dt[0] if dt else 'none', cu[0] if cu else 'none')
PY
}
restart "$W/rs_of" "$W/of" || { say "ARM E  OpenFOAM restarts from its own 0.1" FAIL; }
ref=$(firsts "$W/rs_of/log.interFoam")
for arm in $ARMS; do
    if restart "$W/rs_$arm" "$W/br_$arm" && grep -q "^End" "$W/rs_$arm/log.interFoam"; then
        got=$(firsts "$W/rs_$arm/log.interFoam")
        python3 - $ref $got "$BOUND_TIME" "$BOUND_FIELDS" <<'PY' && say "ARM E  [$arm] OpenFOAM restarts from brae's 0.1: first deltaT and cumulative continue brae's" ok \
                                          || say "ARM E  [$arm] OpenFOAM restarts from brae's 0.1: first deltaT and cumulative continue brae's" FAIL
import sys
dto, cuo, dtb, cub, bound, boundc = sys.argv[1:7]
rd = abs(float(dtb) - float(dto)) / float(dto)
rc = abs(float(cub) - float(cuo)) / max(abs(float(cuo)), 1e-300)
print('      first deltaT OF-from-OF %s, OF-from-brae %s (%.3e); first cumulative %s vs %s (%.3e)'
      % (dto, dtb, rd, cuo, cub, rc))
sys.exit(0 if rd < float(bound) and rc < float(boundc) else 1)
PY
    else
        say "ARM E  [$arm] OpenFOAM restarts from brae's 0.1 and runs to 0.11" FAIL
        tail -5 "$W/rs_$arm/log.interFoam" | sed 's/^/      /'
    fi
done
restart "$W/rs_drop" "$W/br_host" drop
got=$(firsts "$W/rs_drop/log.interFoam")
python3 - $ref $got <<'PY' && say "CONTROL  brae's 0.1 without uniform/time moves E's first deltaT" ok \
                           || say "CONTROL  brae's 0.1 without uniform/time moves E's first deltaT" FAIL
import sys
dto, dtb = sys.argv[1], sys.argv[3]
print('      first deltaT OF-from-OF %s, from brae without uniform/time %s' % (dto, dtb))
sys.exit(0 if dtb == 'none' or abs(float(dtb) - float(dto)) / float(dto) > 1e-3 else 1)
PY

# ---------------------------------------------------------------------------------------------------
# E0: a start directory holding a written field's `_0` level. OpenFOAM reads it as an AUTO_WRITE old time
# (readOldTimeIfPresent, GeometricField.C:120, :131-160) and writes it back at every write time; brae
# writes only a sub-cycled alpha's, so a U_0 there must be named at startup and the run stopped at its
# first write. CONTROL: the identical restart without U_0 runs to its end.
restart_brae()   # restart_brae <dir> [with U_0]
{
    mkdir -p "$1"
    cp -r "$W/of/constant" "$W/of/system" "$W/br_host/0.1" "$1/"
    if [ "${2:-}" = withU0 ]; then
        sed 's/^\( *object *\)U;/\1U_0;/' "$1/0.1/U" > "$1/0.1/U_0"
    fi
    sed -i 's/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         0.15;/' "$1/system/controlDict"
    ( cd "$1" && stdbuf -oL -eL "$BIN" -case . > log.brae 2>&1 )
}
restart_brae "$W/r0" withU0; rc=$?
[ $rc -ne 0 ] && grep -q "U_0 will not be written" "$W/r0/log.brae" && [ "$(timedirs "$W/r0")" = "0.1 " ] \
    && say "ARM E0 a start directory's U_0 is named and the run stops at its first write" ok \
    || { say "ARM E0 a start directory's U_0 is named and the run stops at its first write" FAIL; tail -3 "$W/r0/log.brae" | sed 's/^/      /'; }
restart_brae "$W/r0ctl" && [ "$(timedirs "$W/r0ctl")" = "0.1 0.15 " ] \
    && say "CONTROL  the same restart without U_0 runs to its end and writes 0.15" ok \
    || { say "CONTROL  the same restart without U_0 runs to its end and writes 0.15" FAIL; tail -3 "$W/r0ctl/log.brae" | sed 's/^/      /'; }

# ---------------------------------------------------------------------------------------------------
# F + H: `timeStep 1, purgeWrite 1` against `timeStep N`, to endTime 0.02 -- no trim under timeStep
stage "$LAM" "$W/of_p" 0.02 timeStep 1 1 || exit 1
runof "$W/of_p"
op=$(timedirs "$W/of_p")
# ...and MV, laminar/mixerVessel2D: a SUB-CYCLED alpha, so the old level's captures run -- alpha as the
# step starts and as its last sub-cycle begins, on the host and in a device buffer -- on every step in one
# run and on one step after N-1 without in the other. Both damBreaks have nAlphaSubCycles 1 and run none.
for case in LAM RAS MV; do
    src=$LAM; [ $case = RAS ] && src=$RAS; [ $case = MV ] && src=$MV
    for arm in $ARMS; do
        d1="$W/f1_${case}_$arm"; dn="$W/fn_${case}_$arm"
        stage "$src" "$d1" 0.02 timeStep 1 1 || exit 1
        runbrae "$d1" "$arm"
        last=$(timedirs "$d1" | tr -d ' ')
        n=$(sed -n 's/^index *\([0-9]*\);/\1/p' "$d1/$last/uniform/time")
        if [ $case = LAM ]; then
            [ "$(timedirs "$d1")" = "$op" ] && say "ARM H  [$arm] purgeWrite 1 keeps only OpenFOAM's last directory [$op]" ok \
                                            || say "ARM H  [$arm] purgeWrite 1 keeps only OpenFOAM's last directory [$(timedirs "$d1")] vs [$op]" FAIL
        fi
        stage "$src" "$dn" 0.02 timeStep "$n" 0 || exit 1
        runbrae "$dn" "$arm"
        [ "$(timedirs "$dn")" = "$last " ] && diff -r "$d1/$last" "$dn/$last" > /dev/null \
            && say "ARM F  [$case $arm] writing every step leaves $last/ byte-identical to writing once (index $n)" ok \
            || { say "ARM F  [$case $arm] writing every step leaves $last/ byte-identical to writing once (index $n)" FAIL; diff -rq "$d1/$last" "$dn/$last" | head -5; }
    done
done

# ---------------------------------------------------------------------------------------------------
# G: RAS/damBreak (kEpsilon), adjustable 0.05 to 0.06
stage "$RAS" "$W/of_r" 0.06 adjustable 0.05 0 || exit 1
runof "$W/of_r"
[ "$(timedirs "$W/of_r")" = "0.05 " ] || { say "premise: OpenFOAM RAS writes {0.05} [$(timedirs "$W/of_r")]" FAIL; }
for arm in $ARMS; do
    stage "$RAS" "$W/br_r_$arm" 0.06 adjustable 0.05 0 || exit 1
    runbrae "$W/br_r_$arm" "$arm"
    [ "$(timedirs "$W/br_r_$arm")" = "0.05 " ] && [ "$(filesets "$W/of_r" 0.05)" = "$(filesets "$W/br_r_$arm" 0.05)" ] \
        && say "ARM G  [$arm] RAS: OpenFOAM's directories and file set (k, epsilon, nut)" ok \
        || say "ARM G  [$arm] RAS: OpenFOAM's directories and file set (k, epsilon, nut)" FAIL
    python3 "$CMP" "$W/of_r" "$W/br_r_$arm" 0.05 > "$W/cmp_r_$arm.txt" 2>&1
    judge "RAS $arm" "$W/cmp_r_$arm.txt" "$BOUND_FIELDS" \
        && say "ARM G  [$arm] RAS: every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" ok \
        || { say "ARM G  [$arm] RAS: every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" FAIL; grep -v RESULT "$W/cmp_r_$arm.txt" | grep -B1 "^      " | head -20; }
done

# ---------------------------------------------------------------------------------------------------
# S: a SUB-CYCLED alpha (laminar/mixerVessel2D, nAlphaSubCycles 2, MRF) writes alpha.water_0 -- its old
# time, which GeometricField::storeOldTime makes AUTO_WRITE once the sub-cycle has given it an old-old
# level. OpenFOAM writes it at EVERY write time, the first included, and it holds alpha as the step found
# it (bit-identical to the previous step's alpha.water, measured). Three fixed steps of 1e-3.
if [ -d "$MV" ]; then
    stage "$MV" "$W/of_s" 0.003 timeStep 1 0 adjustTimeStep=no || exit 1
    runof "$W/of_s"
    [ "$(timedirs "$W/of_s")" = "0.001 0.002 0.003 " ] && [ -f "$W/of_s/0.001/alpha.water_0" ] \
        && say "premise: OpenFOAM writes alpha.water_0 at every step of a sub-cycled alpha, the first included" ok \
        || say "premise: OpenFOAM writes alpha.water_0 at every step of a sub-cycled alpha, the first included" FAIL
    for arm in $ARMS; do
        stage "$MV" "$W/br_s_$arm" 0.003 timeStep 1 0 adjustTimeStep=no || exit 1
        runbrae "$W/br_s_$arm" "$arm"
        ok=1
        for t in 0.001 0.002 0.003; do
            [ "$(filesets "$W/of_s" $t)" = "$(filesets "$W/br_s_$arm" $t)" ] || ok=0
        done
        [ $ok -eq 1 ] && say "ARM S  [$arm] OpenFOAM's file set at every step, alpha.water_0 included" ok \
                      || say "ARM S  [$arm] OpenFOAM's file set at every step, alpha.water_0 included" FAIL
        python3 "$CMP" "$W/of_s" "$W/br_s_$arm" 0.001 0.002 0.003 > "$W/cmp_s_$arm.txt" 2>&1
        judge "mixerVessel2D $arm" "$W/cmp_s_$arm.txt" "$BOUND_FIELDS" "$W/of_s/log.interFoam" \
            && say "ARM S  [$arm] every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" ok \
            || { say "ARM S  [$arm] every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" FAIL; grep -v RESULT "$W/cmp_s_$arm.txt" | grep -B1 "^      " | head -20; }
        # brae's own alpha_0 IS its previous alpha -- and, the control, not its current one
        python3 - "$W/br_s_$arm" <<'PY' && say "ARM S  [$arm] brae's alpha.water_0 at 0.002 is its 0.001 alpha bit for bit, not its 0.002 one" ok \
                                       || say "ARM S  [$arm] brae's alpha.water_0 at 0.002 is its 0.001 alpha bit for bit, not its 0.002 one" FAIL
import re, sys
d = sys.argv[1]
def cells(p):
    t = open(p).read()
    return t[t.find('internalField'):t.find('boundaryField')].replace('alpha.water_0', '')
old, prev, cur = cells(d + '/0.002/alpha.water_0'), cells(d + '/0.001/alpha.water'), cells(d + '/0.002/alpha.water')
print('      alpha_0(0.002) == alpha(0.001): %s; == alpha(0.002): %s' % (old == prev, old == cur))
sys.exit(0 if old == prev and old != cur else 1)
PY
    done
else
    say "ARM S  mixerVessel2D tutorial missing" FAIL
fi

# ---------------------------------------------------------------------------------------------------
# W: the tutorials each condition write() was transcribed for, AS SHIPPED but for two fixed steps at the
# file's deltaT and PINNED solves (every tolerance 1e-13, relTol 0 -- at a case's own tolerances the
# comparison measures where two Krylov solvers stop, 1e-07 on damBreakLeakage). Meshed by the tutorial's
# own Allrun with the solver line dropped and its parallel steps run serially.
#   capillaryRise         constantAlphaContactAngle (and alpha.water_0's frozen gradient)
#   weirOverflow          variableHeightFlowRate, variableHeightFlowRateInletVelocity, and alpha.water_0
#                         on a mixed-family patch -- the witness for the old level's restore rule
#   angledDuct            turbulentIntensityKineticEnergyInlet, turbulentMixingLengthDissipationRateInlet, slip
#   damBreakLeakage       cyclicACMI on every volume field, and the surface fields' ACMI patches
#   damBreakPorousBaffle  porousBafflePressure (fixedJump's jump on the owner)
#   damBreakPermeable     prghPermeableAlphaTotalPressure, permeableAlphaPressureInletOutletVelocity
#   nozzleFlow2D          LES kEqn's k and nut (deltaT 1e-9: the file's 1e-8 diverges in OpenFOAM itself)
#   eulerianInjection     alpha.water_0 over fixedValue and inletOutlet patches
# CONTROL: BRAE_CONTROL_ALPHA_OLD_START=1 (the old level at the step's start on every patch, this
# writer's first form) puts weirOverflow's alpha.water_0 over the bound -- 5.3e-03 at the inlet.
stage_allrun()   # stage_allrun <src> <dir> <deltaT or ""> -- mesh as the Allrun does, two pinned steps
{
    local src="$1" d="$2" dtOverride="$3"
    cp -r "$src" "$d" || return 1
    rm -rf "$d"/[1-9]* "$d"/processor* "$d"/log.*
    # the Allrun without its solver, its decompose/reconstruct, and with runParallel run serially
    sed -E -e '/decomposePar|reconstructPar|redistributePar/d' -e '/\$\(getApplication\)|runApplication +interFoam|runParallel +interFoam/d' \
        -e 's/runParallel/runApplication/' "$d/Allrun" > "$d/Allrun.mesh"
    ( cd "$d" && bash ./Allrun.mesh > log.allrunmesh 2>&1 ) || return 1
    [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    python3 - "$d" "$dtOverride" <<'PY'
import re, sys
d, dtOverride = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'
s = open(c).read()
dt = float(dtOverride) if dtOverride else float(re.search(r'^deltaT\s+([^;]+);', s, re.M).group(1))
s = re.sub(r'\nfunctions\s*\{.*?\n\}', '\nfunctions\n{\n}', s, flags=re.S)
if not re.search(r'^functions', s, re.M):
    s += '\nfunctions\n{\n}\n'
for k, v in [('deltaT', '%.12g' % dt), ('endTime', '%.12g' % (2*dt)), ('writeControl', 'timeStep'),
             ('writeInterval', '1'), ('purgeWrite', '0'), ('adjustTimeStep', 'no'), ('writePrecision', '17'),
             ('writeFormat', 'ascii'), ('writeCompression', 'off')]:
    if re.search(r'^%s\s' % k, s, flags=re.M):
        s = re.sub(r'^%s\s.*' % k, '%-16s%s;' % (k, v), s, flags=re.M)
    else:
        s = s.replace('\nfunctions\n', '\n%-16s%s;\nfunctions\n' % (k, v), 1)
open(c, 'w').write(s)
q = d + '/system/fvSolution'
t = open(q).read()
t = re.sub(r'(tolerance\s+)[^;]+;', r'\g<1>1e-13;', t)
t = re.sub(r'(relTol\s+)[^;]+;', r'\g<1>0;', t)
open(q, 'w').write(t)
PY
}
# <tutorial>:<deltaT override>:<bound>. The bounds are one decade above the worst of host and device at
# pinned solves (2026-09-30): capillaryRise 3.8e-14, weirOverflow 2.6e-12, damBreakPorousBaffle 3.5e-12,
# nozzleFlow2D 1.3e-12, eulerianInjection 1.1e-13, damBreakPermeable 1.6e-13, angledDuct 1.3e-09 -- and
# damBreakLeakage 3.6e-07, which is NOT a port gap: at step 2 its column stands at rest behind the shut
# baffle and U is round-off on a near-zero scale (the leakage gate's own header; it compares after 520
# steps, at 4.9e-12). The value check there is weak and says so; its structure check is not.
W_CASES="
laminar/capillaryRise::4e-13
RAS/weirOverflow::3e-11
RAS/angledDuct::2e-08
RAS/damBreakLeakage::4e-06
RAS/damBreakPorousBaffle::4e-11
laminar/damBreakPermeable::2e-12
LES/nozzleFlow2D:1e-9:2e-11
laminar/vofToLagrangian/eulerianInjection::2e-12
"
for entry in $W_CASES; do
    rel=${entry%%:*}; rest=${entry#*:}; dtw=${rest%%:*}; BOUND_W=${rest#*:}; key=$(basename "$rel")
    src="$TUT/multiphase/interFoam/$rel"
    [ -d "$src" ] || { say "ARM W  $key: tutorial missing" FAIL; continue; }
    stage_allrun "$src" "$W/w_of_$key" "$dtw" || { say "ARM W  $key: meshing failed (see $W/w_of_$key/log.allrunmesh)" FAIL; continue; }
    runof "$W/w_of_$key"
    ot=$(timedirs "$W/w_of_$key")
    [ "$(echo $ot | wc -w)" = 2 ] || { say "ARM W  $key: premise, OpenFOAM writes two steps [$ot]" FAIL; continue; }
    for arm in $ARMS; do
        d="$W/w_br_${key}_$arm"
        mkdir -p "$d"
        cp -r "$W/w_of_$key/0" "$W/w_of_$key/constant" "$W/w_of_$key/system" "$d/"
        runbrae "$d" "$arm"
        ok=1
        [ "$(timedirs "$d")" = "$ot" ] || ok=0
        for t in $ot; do
            [ "$(filesets "$W/w_of_$key" $t)" = "$(filesets "$d" $t)" ] || ok=0
        done
        [ $ok -eq 1 ] && say "ARM W  [$arm] $key: OpenFOAM's directories and file sets" ok \
                      || say "ARM W  [$arm] $key: OpenFOAM's directories and file sets" FAIL
        python3 "$CMP" "$W/w_of_$key" "$d" $ot > "$W/cmp_w_${key}_$arm.txt" 2>&1
        judge "$key $arm" "$W/cmp_w_${key}_$arm.txt" "$BOUND_W" "$W/w_of_$key/log.interFoam" \
            && say "ARM W  [$arm] $key: every file's structure is OpenFOAM's, every value within $BOUND_W" ok \
            || { say "ARM W  [$arm] $key: every file's structure is OpenFOAM's, every value within $BOUND_W" FAIL; grep -v RESULT "$W/cmp_w_${key}_$arm.txt" | grep -B1 "^      " | head -12; }
    done
done
if [ -d "$W/w_of_weirOverflow" ]; then
    d="$W/w_ctl_weir"
    mkdir -p "$d"
    cp -r "$W/w_of_weirOverflow/0" "$W/w_of_weirOverflow/constant" "$W/w_of_weirOverflow/system" "$d/"
    runbrae "$d" host BRAE_CONTROL_ALPHA_OLD_START=1
    python3 "$CMP" "$W/w_of_weirOverflow" "$d" $(timedirs "$W/w_of_weirOverflow") > "$W/cmp_w_ctl.txt" 2>&1
    judge "control start-only" "$W/cmp_w_ctl.txt" 3e-11 | grep -q "over the bound: .*alpha.water_0" \
        && say "CONTROL  BRAE_CONTROL_ALPHA_OLD_START=1 puts weirOverflow's alpha.water_0 over the bound" ok \
        || say "CONTROL  BRAE_CONTROL_ALPHA_OLD_START=1 puts weirOverflow's alpha.water_0 over the bound" FAIL
fi

# ---------------------------------------------------------------------------------------------------
# R: a file brae cannot write yet -- laminar/waves/stokesI's wave-model state, uniform/waveProperties.<patch>
# (waveModel.C:250-261) -- is named at startup, before the first step, and the run stops at its first write
# time having written nothing. (Until the condition write()s landed this arm used capillaryRise's contact
# angle; that one is written now.) CONTROL: the same case whose only write time lies past endTime runs to
# its end: the refusal is of the OUTPUT, and a run that never reaches a write is not refused.
STK="$TUT/multiphase/interFoam/laminar/waves/stokesI"
if [ -d "$STK" ]; then
    stage "$STK" "$W/stk" 0.05 timeStep 3 0 adjustTimeStep=no deltaT=0.01 || exit 1
    # line-buffered: the refusal goes to stderr, the steps to stdout, and only then is file order time order
    ( cd "$W/stk" && stdbuf -oL -eL "$BIN" -case . > log.brae 2>&1 ); rc=$?
    python3 - "$W/stk/log.brae" "$rc" "$(timedirs "$W/stk")" <<'PY' && say "ARM R  an unwritten file is named before step 1; the run stops at its first write, nothing written" ok \
                                                                    || say "ARM R  an unwritten file is named before step 1; the run stops at its first write, nothing written" FAIL
import re, sys
log, rc, dirs = open(sys.argv[1]).read(), int(sys.argv[2]), sys.argv[3].strip()
named = log.find('uniform/waveProperties.<patch> will not be written')
first = re.search(r'^\s*t = ', log, flags=re.M)
# the third step's line prints after runTime.write(), which is where the run stops
steps = len(re.findall(r'^\s*t = ', log, flags=re.M))
stop = log.find('this is a write time, and OpenFOAM would write files brae does not write')
print('      exit %d, named at offset %d, first step at %s, %d step lines, stop at %d, time directories [%s]'
      % (rc, named, first.start() if first else -1, steps, stop, dirs))
sys.exit(0 if rc != 0 and 0 <= named < (first.start() if first else -1) and steps == 2
              and stop > first.start() and not dirs else 1)
PY
    stage "$STK" "$W/stk_ctl" 0.02 timeStep 1000 0 adjustTimeStep=no deltaT=0.01 || exit 1
    ( cd "$W/stk_ctl" && "$BIN" -case . > log.brae 2>&1 ) \
        && [ -z "$(timedirs "$W/stk_ctl")" ] && grep -q "will not be written" "$W/stk_ctl/log.brae" \
        && say "CONTROL  the same case with no write time before endTime runs to its end (exit 0)" ok \
        || { say "CONTROL  the same case with no write time before endTime runs to its end (exit 0)" FAIL; tail -3 "$W/stk_ctl/log.brae" | sed 's/^/      /'; }
else
    say "ARM R  stokesI tutorial missing" FAIL
fi

[ $fail -eq 0 ] && echo "PASS: brae_interFoam writes OpenFOAM's time directories, when OpenFOAM does, without moving the run"
exit $fail
