# THE SHARED HALF OF THE WRITE GATE, sourced by every tests/interfoam_write/arm_*.sh. Each arm is its own
# file and its own ctest test, so a change is checked by the arm it touches and the arms run side by side.
# The header below is the gate's own, kept whole: what is written, when, and what each arm holds.
#
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
#   ARM E1 a refusal thrown inside write() leaves no time directory (BRAE_CONTROL_WRITE_REFUSE_LATE=1).
#   ARM V  the list-entry wave models' waveProperties byte-identical to OpenFOAM's (irregularMultiDirection,
#          streamFunction): arm W's comparer is blind to the token form.
#          Also a sized `10 ( ... )` list, an empty `extra;` entry, and a quoted string (refused by name),
#          each against OpenFOAM; CONTROL: one token put back raw fails the byte check.
#   ARM Y  a refining mesh beyond W: alpha.water_0 mapped with the mesh (rule, OpenFOAM and brae), controls
#          BRAE_CONTROL_AMR_NO_WRITE_COMPACT=1 and BRAE_CONTROL_AMR_ALPHA0_START=1, sixty steps through an
#          unrefinement, a write before any change with a restart from it (the global refine index), and a
#          restart from a compressed refined write (the .gz lookups).
#   ARM Z  a coupled (cyclicAMI) patch's rAU and p are the result's own cells evaluated; control
#          BRAE_CONTROL_COUPLED_OPERAND_VALUES=1 on mixerVesselAMI (OpenFOAM staged to PCG; brae falls back to it).
#   ARM M  a moving mesh's cumulativeContErr is the absolute flux's (sloshingTank2D), with its control.
#   ARM P  the mesh update's CorrectPhi continuity error counts (waveMakerPiston, loose pcorr), with its
#          control BRAE_CONTROL_NO_CORRECTPHI_CONTERR=1.
#   ARM Q  a rigid body's uniform/rigidBodyMotionState entry by entry and as text (DTCHullMoving moving,
#          floatingObject under Euler at rest, `2 { 0 }`), with controls BRAE_CONTROL_RBSTATE_OLD=1 and
#          BRAE_CONTROL_RBSTATE_PAREN=1; DTCHullMoving's files are arm W's, host only (the device refuses it).
#   ARM R  a file brae cannot write yet (a wave model's entry holding a sub-dictionary, on stokesI) is named
#          at startup, and the run stops at its first write time with nothing written.
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
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
ORACLE_HITS=0
ORACLE_MISSES=0
MESH_HITS=0
MESH_MISSES=0
. "$ROOT/tests/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }
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
        s = re.sub(r'^%s\s.*' % k, '%s %s;' % (k.ljust(15), v), s, flags=re.M)
    else:
        s = s.replace('\nfunctions\n', '\n%s %s;\nfunctions\n' % (k.ljust(15), v), 1)
open(c, 'w').write(s)
PY
}
timedirs() { ( cd "$1" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9.e+-]+$' | grep -vE '^0$' | sort -g | tr '\n' ' ' ); }
filesets() { ( cd "$1/$2" && find . -type f | sed 's/\.gz$//' | sort | tr '\n' ' ' ); }
# runof <dir>: real OpenFOAM on the staged case -- CACHED (tests/of_oracle_cache.sh). The key is a hash of
# every staged byte, the mesh included, and of the interFoam binary, so OpenFOAM's answer is re-used exactly
# when nothing that decides it has changed: editing brae does not re-run OpenFOAM. brae's own runs are never
# cached. BRAE_OF_CACHE=off restores a run every time.
runof()
{
    local d="$1"
    local key
    key=$(oracleKey "$d" "interfoam_write" "run")
    if oracleRestore "$d" "$key" "log.interFoam"; then
        ORACLE_HITS=$((ORACLE_HITS + 1))
        return 0
    fi
    ( cd "$d" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam did not run in $d"; tail -5 "$d/log.interFoam"; exit 1; }
    ORACLE_MISSES=$((ORACLE_MISSES + 1))
    oracleStore "$d" "$key"
}
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
# LIMIT, stated: the floor is at least twice OpenFOAM's summed `sum local` (see judge), which is at
# least |OpenFOAM's own value| -- so in arms S and W this file is held to within the correctors' own
# local error, and a dropped or sign-flipped accumulation would pass there. Arms D witness that class:
# damBreak's 4.6e-04 and waterChannel's -2.3e-03 are compared relative, with no floor, and caught the
# missing initCorrectPhi term (2.885e-10).
judge()   # judge <label> <resultFile> <bound> [ofLog]
{
    python3 - "$@" <<'PY'
import json, re, sys
label, path, bound = sys.argv[1], sys.argv[2], float(sys.argv[3])
r = json.loads([l for l in open(path) if l.startswith('RESULT ')][-1][7:])
# AN OLD-TIME LEVEL IS HELD ON ITS FIELD'S SCALE where that is the larger: <X>_0 is X one step back, and a
# level that is still zero -- floatingObject's U_0 at the first write, whose only non-zero entries are the
# 1.7e-14 m/s round-off of a wall that has not moved -- has no scale of its own to be relative to.
for k, v in r['files'].items():
    parent = r['files'].get(k[:-2]) if k.endswith('_0') else None
    if parent and parent['rel'] > 0 and v['rel'] > 0:
        own = v['abs'] / v['rel']
        v['rel'] = v['abs'] / max(own, parent['abs'] / parent['rel'])
floor = 0.0
if len(sys.argv) > 4:
    log = open(sys.argv[4]).read()
    coScale = 0.0
    for chunk in log.split('\nTime = '):
        co = re.search(r'Courant Number mean: (\S+)', chunk)
        coScale += len(re.findall(r'sum local', chunk)) * 2.0 * (float(co.group(1)) if co else 0.0)
    fieldsRel = max([v['rel'] for k, v in r['files'].items()
                     if not k.endswith('cumulativeContErr') and 'functionObject' not in k] + [0.0])
    # ...and the bound that holds for any two codes: a corrector's |global| is at most its `sum local`
    # (|sum V*div| <= sum V*|div|), so two runs whose p_rgh solves stop at different residuals -- the
    # device's linear solvers against OpenFOAM's -- differ by at most twice the total. MEASURED needing it:
    # stokesI's device, 1.2e-13 against 1.4e-11 of `sum local` (the host, OpenFOAM's iteration for
    # iteration, 1e-15); and DTCHull under localEuler, where OpenFOAM prints no Courant number
    # (interFoam.C, `if (!LTS)`) and the flux-scale floor above is zero.
    sumLocal = sum(float(x) for x in re.findall(r'sum local = (\S+),', log))
    floor = max(max(10 * 2.220446049250313e-16, fieldsRel) * coScale, 2.0 * sumLocal)
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
    # THE MESH IS CACHED, keyed before anything is run: every byte of the tutorial as copied, the time step
    # override, and this function's own text (the staging IS part of the input). snappyHexMesh on the two
    # large tutorials was most of a run's wall clock, and its output cannot change while these do not.
    local mkey
    mkey=$(oracleKey "$d" "interfoam_write" "mesh" "$dtOverride" "$(declare -f stage_allrun | sha256sum | cut -c1-16)")
    if oracleRestore "$d" "$mkey" ".brae-mesh-done"; then
        MESH_HITS=$((MESH_HITS + 1))
        return 0
    fi
    MESH_MISSES=$((MESH_MISSES + 1))
    # the Allrun without its solver, its decompose/reconstruct, and with runParallel run serially
    # ...and its Allrun.pre the same way, where the meshing lives (RAS/motorBike: decomposePar, a parallel
    # snappyHexMesh, the per-processor refinementHistory removal and restore0Dir -processor). Run as shipped
    # it left processor directories and a written time the solver then started from.
    # controlDict_nextWrite is motorBike's second run, a restart this staging does not take.
    sed -E -e '/decomposePar|reconstructPar|redistributePar/d' -e '/\$\(getApplication\)|runApplication +interFoam|runParallel +interFoam/d' \
        -e '/controlDict_nextWrite/d' -e 's/runParallel/runApplication/' -e 's#^\./Allrun\.pre#bash ./Allrun.pre.mesh#' \
        "$d/Allrun" > "$d/Allrun.mesh"
    if [ -f "$d/Allrun.pre" ]; then
        sed -E -e '/decomposePar|reconstructPar|redistributePar/d' -e 's/restore0Dir -processor/restore0Dir/' \
            -e 's#^ls -d processor.*refinementHistory$#rm -f constant/polyMesh/refinementHistory#' \
            -e 's/runParallel/runApplication/' "$d/Allrun.pre" > "$d/Allrun.pre.mesh"
    fi
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
        s = re.sub(r'^%s\s.*' % k, '%s %s;' % (k.ljust(15), v), s, flags=re.M)
    else:
        s = s.replace('\nfunctions\n', '\n%s %s;\nfunctions\n' % (k.ljust(15), v), 1)
open(c, 'w').write(s)
q = d + '/system/fvSolution'
t = open(q).read()
t = re.sub(r'(tolerance\s+)[^;]+;', r'\g<1>1e-13;', t)
t = re.sub(r'(relTol\s+)[^;]+;', r'\g<1>0;', t)
open(q, 'w').write(t)
PY
    local prc=$?
    [ $prc -eq 0 ] || return $prc
    : > "$d/.brae-mesh-done"
    oracleStore "$d" "$mkey"
}
# <tutorial>:<deltaT override>:<host bound>:<device bound>. Each arm has its own bound, one decade above
# that arm's worst: one bound for both let the host borrow the device's looser linear solvers, 100x on
# electrostaticDeposition. The host reproduces run to run (waveMakerFlap 4.15e-10 in two runs of two
# builds), so its bound is one decade above its measured worst, rounded up -- and above an earlier build's
# figure where that was larger (waveMakerFlap 5.7e-10). The device's is one decade above the worst either
# arm ever showed, since it does not reproduce: waveMakerFlap's device read 2.3e-08 in one run, 8.9e-10 in two.
# HOST worst at pinned solves (U2 gate, 2026-09-30), every file but cumulativeContErr: capillaryRise 3.8e-14,
# weirOverflow 4.3e-13, angledDuct 8.3e-10, damBreakLeakage 2.4e-07, damBreakPorousBaffle 4.3e-13,
# damBreakPermeable 1.6e-13, nozzleFlow2D 1.1e-12, eulerianInjection 4.3e-14, cnoidal 3.6e-10, solitary
# 1.5e-10, solitaryGrimshaw 1.1e-08, solitaryMcCowan 1.0e-08, stokesI 4.8e-08, stokesII 1.3e-09, stokesV
# 1.4e-09, mangroveInteraction 7.9e-03, DTCHull 5.7e-13, sloshingTank2D 1.2e-12 (held at the 1e-11 it
# already had, not loosened to 2e-11), testTubeMixer 3.5e-11, sloshingCylinder 9.4e-11,
# electrostaticDeposition 1.8e-07, waveMakerSolitary 6.0e-11, waveMakerPiston 1.3e-09, waveMakerFlap
# 5.7e-10, waveMakerMultiPaddleFlap 7.3e-12, waveMakerMultiPaddlePiston 2.2e-11.
# The DEVICE bounds, and the notes on the cases, from the combined measurements (2026-09-30):
# capillaryRise 3.8e-14, weirOverflow 2.6e-12, damBreakPorousBaffle 3.5e-12,
# nozzleFlow2D 1.3e-12, eulerianInjection 1.1e-13, damBreakPermeable 1.6e-13, angledDuct 1.3e-09 -- and
# damBreakLeakage 3.6e-07, which is NOT a port gap: at step 2 its column stands at rest behind the shut
# baffle and U is round-off on a near-zero scale (the leakage gate's own header; it compares after 520
# steps, at 4.9e-12). The value check there is weak and says so; its structure check is not.
# The wave tutorials (waveAlpha, waveVelocity, uniform/waveProperties.<patch>, alpha.water_0 over a
# waveAlpha inlet): cnoidal 4.2e-10, solitary 1.7e-10, solitaryGrimshaw 1.7e-08, solitaryMcCowan
# 1.5e-08, stokesI 9.2e-08, stokesII 1.5e-09, stokesV 1.6e-09 (the waves gate holds their values);
# mangroveInteraction 8.1e-03, also NOT a port gap: the top inletOutlet's flux is +-1e-20 above water at
# rest, so each code takes the inflow or outflow branch by the sign of round-off (6,705 of 9,800 faces
# differ at 0.1) and k and epsilon follow -- its structure check (slip on k/epsilon/nut/p_rgh) is the point.
# RAS/DTCHull (localEuler's rDeltaT, nutkRoughWallFunction, outletPhaseMeanVelocity, variableHeightFlowRate,
# meshed by its Allrun's snappyHexMesh serially): 5.7e-13 host, 1.0e-11 device.
# The solidBody-moved meshes (Uf, meshPhi, polyMesh/points): sloshingTank2D 9.1e-13, testTubeMixer 3.5e-11,
# sloshingCylinder 9.4e-11 (through its as-shipped first move, which leaves OpenFOAM's own alpha in
# [-1.11, 1.86]); electrostaticDeposition 1.8e-07 host and 1.8e-07 device (2026-10-02). The device
# read 1.7e-05 until then, and the cause recorded here -- a per-face fixesValue mask -- was WRONG: its U
# patches read the RELATIVE flux after the corrector where OpenFOAM's read the absolute one (side-02, all
# 225 faces on the wrong branch). Control BRAE_CONTROL_DEVICE_U_PATCH_RELATIVE=1, sensitivity/.
# The wave makers (displacementLaplacian: pointDisplacement with the waveMaker patch's write() -- the solitary
# branch's rewritten wavePeriod, waveAngle in radians -- cellDisplacement's cellMotion patches, and correctPhi's
# rAU): waveMakerSolitary 7.1e-11, waveMakerPiston 1.8e-09, waveMakerFlap 5.7e-10 host and 2.3e-08 device,
# waveMakerMultiPaddleFlap 7.3e-12, waveMakerMultiPaddlePiston 2.4e-11.
# The list-entry wave models (waveProperties echoed token by token, byte-identical after the banner on both
# arms): irregularMultiDirection 7.9e-10 host, 1.2e-09 device; streamFunction 2.5e-09 host, 2.7e-09 device.
# THE ROWS ABOVE 1e-9 ARE THE CASES' OWN SENSITIVITY, NOT PORT GAPS -- measured 2026-10-02, worst field file,
# brae host / brae device / OpenFOAM against itself with every tolerance 1e-14 for 1e-13 (sensitivity/):
#   streamFunction 2.5e-09 / 2.7e-09 / 4.5e-05     stokesII 1.3e-09 / 1.5e-09 / 1.0e-04
#   stokesV 1.4e-09 / 1.6e-09 / 9.5e-05            cnoidal 3.6e-10 / 4.2e-10 / 1.4e-05
#   irregularMultiDirection 7.9e-10 / 1.2e-09 / 6.2e-06   solitary 1.5e-10 / 1.7e-10 / 7.6e-08
#   solitaryMcCowan 1.0e-08 / 1.5e-08 / 9.0e-07    waveMakerPiston 1.3e-09 / 1.8e-09 / 2.1e-07
#   waveMakerFlap 4.1e-10 / 8.9e-10 / 6.7e-08
# RAS/angledDuct is the other kind: its p_rgh GAMG is capped at 50 cycles and never converges (the first solve
# ENDS at 1.36 from 1.0), so tolerances do not move it and round-off does -- gravity moved by one ulp moves
# OpenFOAM's own phi 7.4e-10, against brae's 8.3e-10 host and 1.3e-09 device (sensitivity/angled_duct_ulp.sh).
# The four sloshing tanks beside sloshingTank2D (2026-10-02, both arms as shipped): 2D3DoF 9.7e-13 / 1.0e-12,
# 3D 2.8e-11 / 2.5e-11, 3D3DoF 4.8e-12 / 1.8e-12, 3D6DoF 3.4e-12 / 2.2e-12.
W_CASES="
laminar/capillaryRise::4e-13:4e-13
RAS/weirOverflow::5e-12:3e-11
RAS/angledDuct::8.3e-09:1.3e-08
RAS/damBreakLeakage::3e-06:4e-06
RAS/damBreakPorousBaffle::5e-12:4e-11
laminar/damBreakPermeable::2e-12:2e-12
LES/nozzleFlow2D:1e-9:2e-11:2e-11
laminar/vofToLagrangian/eulerianInjection::5e-13:2e-12
laminar/waves/cnoidal::3.6e-09:4.2e-09
laminar/waves/solitary::1.5e-09:1.7e-09
laminar/waves/solitaryGrimshaw::2e-07:2e-07
laminar/waves/solitaryMcCowan::1.0e-07:1.5e-07
laminar/waves/stokesI::5e-07:1e-06
laminar/waves/stokesII::1.3e-08:1.5e-08
laminar/waves/stokesV::1.4e-08:1.6e-08
laminar/waves/irregularMultiDirection::7.9e-09:1.2e-08
laminar/waves/streamFunction::2.5e-08:2.7e-08
laminar/waves/mangroveInteraction::8e-02:1e-01
RAS/DTCHull::6e-12:1e-10
laminar/sloshingTank2D::1e-11:1e-11
laminar/sloshingTank2D3DoF::9.7e-12:1.0e-11
laminar/sloshingTank3D::2.8e-10:2.5e-10
laminar/sloshingTank3D3DoF::4.8e-11:1.8e-11
laminar/sloshingTank3D6DoF::3.4e-11:2.2e-11
laminar/testTubeMixer::4e-10:4e-10
laminar/sloshingCylinder::1e-09:1e-09
RAS/electrostaticDeposition::2e-06:1.8e-06
laminar/waves/waveMakerSolitary::7e-10:1e-09
laminar/waves/waveMakerPiston::1.3e-08:1.8e-08
laminar/waves/waveMakerFlap::4.1e-09:8.9e-09
laminar/waves/waveMakerMultiPaddleFlap::8e-11:1e-10
laminar/waves/waveMakerMultiPaddlePiston::3e-10:3e-10
RAS/DTCHullMoving::3e-10:2.4e-10
RAS/DTCHullMovingCoarse::1e-10:3.1e-11
laminar/damBreakWithObstacle::2e-11:3e-11
laminar/oscillatingBox::2e-10:1.8e-10
RAS/motorBike::9e-12:2e-11
RAS/mixerVesselAMI::5e-11:5e-11
RAS/floatingObject::1e-12:1.6e-12
"
# RAS/DTCHullMoving (rigidBodyMotion: pointDisplacement, uniform/rigidBodyMotionState, points, meshPhi, Uf,
# rAU): host 2.4e-11, pointDisplacement at 0.0002 (1.1e-16 absolute on a 4.7e-06 largest); polyMesh/points
# 4.3e-18 (2026-09-30). Its device arm runs since 2026-10-01 -- see BOUND_X3_FIELDS and arm X3.
# Arm Q's bounds: each rigidBodyMotionState entry against its own size -- DTCHullMoving's worst 3.4e-16
# (0.0002/qDdot), floatingObject's 0 -- and floatingObject (Euler)'s fields, worst 8.7e-14 (0.02/phi).
# The refining meshes (U5/U6: polyMesh/* after the first change -- compared as TEXT, exactly -- hexRef8's
# cellLevel, pointLevel, level0Edge and refinementHistory at every write, the cellLevel field, Uf, and
# oscillatingBox's meshPhi and points0; alpha.water_0 mapped with the mesh), 2026-09-30, worst field:
# damBreakWithObstacle 1.5e-12 host / 2.1e-12 device; oscillatingBox 1.8e-11 host / 1.8e-11 device (the device loop
# composes the change and the move since 2026-10-02); motorBike 8.8e-13 host / 1.0e-12 device (snappy's binary
# levels and its level0Edge 0.5 read, the frozenPoints zone written with its meta).
# RAS/mixerVesselAMI (U4: rAU and p on the cyclicAMI pair, the result's own cells evaluated -- every field
# expression ends in correctLocalBoundaryConditions, GeometricFieldFunctionsM.C:50), OpenFOAM staged to PCG: host
# 4.3e-12 (2026-10-01, 895k cells). Its device bound is `-`: the device loop does not couple a cyclicAMI.
# RAS/floatingObject AS SHIPPED (U7: CrankNicolson 0.5 on a mesh a rigid body moves, kEpsilon, three outer
# correctors, correctPhi): the scheme's ddt0 fields with their patches, U_0 / k_0 / epsilon_0, meshPhiCN_0 and
# V0, host worst 1.2e-13 (0.02/phi; 2026-10-01). Its device bound is `-`: the device loop refuses a rigid
# body by name. Arm X holds what this row cannot witness, and its controls.
BOUND_RB_DTC=4e-15
BOUND_FO=9e-13
BOUND_WFO=1e-12
# ...and the same row from the device loop: measured worst 1.6e-13 (0.02/ddt0(k)), 2026-10-02
BOUND_WFO_DEVICE=1.6e-12
# RAS/DTCHullMoving ON THE DEVICE (2026-10-01): the rigid body's load and the atmosphere's tangentialVelocity
# are carried (arm X3 holds both controls). Worst file k 6.6e-09, every other file at the host arm's level
# (pointDisplacement 2.4e-11, U 3.4e-12). THE k GAP IS THE SOLVE, NOT A TERM: the closure's inputs and the
# assembled k system agree with the host's to round-off (diagonal 1.7e-15, source 7.2e-15, from the stage
# dump), and OpenFOAM's own k solve here stops at its 1000-iteration cap unconverged (final residual 1.1e-13
# against 1e-13), so the device smoother's last bits ride the iterate. floatingObject's device bound stays
# `-`: the rigid body runs there too (tests/interfoam_moving_vs_openfoam.sh, `floating`, under Euler), but
# the tutorial is CrankNicolson with kEpsilon on a moving mesh, whose moving branch the device closure does
# not carry -- refused by name at start-up -- and its written state is the host loop's alone.
# ...and arm X3 (device_body/) runs on the COARSENED hull (coarse_dtc_source), where the device's worst file was
# k at 2.9e-10 for the same reason.
# THAT REASON WAS A PORT GAP, CLOSED 2026-10-02, and "the solve, not a term" was half of it: smoothSolver's
# LOOP evaluates lduMatrix::residual (smoothSolver.C:190-197, lduMatrixATmul.C:268-340), where the device
# evaluated source - A.psi. OpenFOAM's k solve at step two stalls at 1.08e-13 against a tolerance of 1e-13 and
# runs its 1000 sweeps; the device read 4.5e-14 and stopped after 2. With deviceResidual: 1000 sweeps, final
# residual 1.077e-13, k 9.4e-14 on the tutorial's mesh (worst file pointDisplacement 2.4e-11, the host arm's)
# and worst file 3.1e-12 on the coarsened hull. Control BRAE_CONTROL_DEVICE_GS_RESIDUAL_AMUL=1:
# device_body/k_residual_control.sh.
BOUND_X3_FIELDS=3.1e-11

declare -A WCASE_OF
# the per-arm half of wcase: brae on each arm against the staged OpenFOAM run ($key, $ot and the two bounds
# are wcase's)
wcase_arms()
{
    local arms="$1"
    for arm in $arms; do
        BOUND_W=$BOUND_WH
        [ "$arm" = device ] && BOUND_W=$BOUND_WD
        d="$W/w_br_${key}_$arm"
        mkdir -p "$d"
        cp -r "$W/w_of_$key/0" "$W/w_of_$key/constant" "$W/w_of_$key/system" "$d/"
        # a device bound of `-`: the device loop refuses the case at startup, by name -- assert THAT refusal,
        # the case's own (several device refusals end in "Run without -device"), and that it wrote nothing
        if [ "$BOUND_W" = "-" ]; then
            case $key in
                floatingObject) why="which the device loop does not keep in its written form" ;;
                *) why="" ;;
            esac
            [ -n "$why" ] || say "ARM W  [$arm] $key: a device bound of - with no expected refusal named" FAIL
            ( cd "$d" && "$BIN" -case . -device > log.brae 2>&1 ); rc=$?
            [ $rc -ne 0 ] && [ -n "$why" ] && grep -qF "$why" "$d/log.brae" && [ -z "$(timedirs "$d")" ] \
                && say "ARM W  [$arm] $key: refused at startup, by name, nothing written" ok \
                || { say "ARM W  [$arm] $key: refused at startup, by name, nothing written" FAIL; tail -3 "$d/log.brae" | sed 's/^/      /'; }
            continue
        fi
        [ "$key" = mixerVesselAMI ] && cp "$d/system/fvSolution.gamg" "$d/system/fvSolution"
        runbrae "$d" "$arm"
        if [ "$key" = mixerVesselAMI ]; then
            grep -q "solver          GAMG;" "$d/system/fvSolution" \
                && grep -q "across which brae's GAMG is not ported" "$d/log.brae" \
                && say "ARM W  [$arm] $key: brae read the tutorial's GAMG entry and announced PCG with DIC in its place" ok \
                || say "ARM W  [$arm] $key: brae read the tutorial's GAMG entry and announced PCG with DIC in its place" FAIL
        fi
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
}

# coarse_dtc_source <dir>: RAS/DTCHullMoving copied to <dir> with every block of its background mesh halved
# in each direction -- 108,833 cells where the tutorial has 848,022 (2026-10-01), the hull's patch (6,685
# faces), the rigid body, the atmosphere's tangentialVelocity and every condition unchanged. The tutorial's
# own row stays at full size; the arms that hold the body's state file and the device's load run here,
# where a run is twenty seconds and not five minutes. MEASURED against OpenFOAM, two steps: host worst
# 1.1e-11 (rAU), device worst 2.9e-10 (k), every other file below 7e-12 on both arms.
coarse_dtc_source()
{
    local d="$1"
    rm -rf "$d"
    cp -r "$TUT/multiphase/interFoam/RAS/DTCHullMoving" "$d" || return 1
    python3 - "$d/system/blockMeshDict" <<'EOF_COARSE'
import re, sys
p = sys.argv[1]
s = open(p).read()
def half(m):
    return '(%d %d %d)' % tuple(max(1, int(x) // 2) for x in m.groups())
s, n = re.subn(r'\((\d+) (\d+) (\d+)\)(?= simpleGrading)', half, s)
open(p, 'w').write(s)
sys.exit(0 if n == 6 else 1)
EOF_COARSE
}

# wcase <key> [arm ...]: ONE row of W_CASES -- the tutorial meshed and pinned, real OpenFOAM run on it (both
# cached), then brae on each named arm (every arm the machine has by default; `of` for none) held against
# it. An arm that depends on a tutorial's run calls this for the tutorial it needs, so each arm file stands
# alone. Idempotent within one script.
declare -A WCASE_DONE
wcase()
{
    local wkey="$1"
    shift
    local arms="${*:-$ARMS}"
    [ "$arms" = of ] && arms=""
    [ -n "${WCASE_DONE[$wkey:$arms]:-}" ] && return 0
    WCASE_DONE[$wkey:$arms]=1
    local entry=""
    local e
    for e in $W_CASES; do
        [ "$(basename "${e%%:*}")" = "$wkey" ] && entry="$e"
    done
    [ -n "$entry" ] || { say "ARM W  $wkey: no such row in W_CASES" FAIL; return 1; }
    # the OpenFOAM half is staged once; a second call for another arm keeps it
    if [ -n "${WCASE_OF[$wkey]:-}" ]; then
        rel=${entry%%:*}; rest=${entry#*:}; dtw=${rest%%:*}; bounds=${rest#*:}; key=$(basename "$rel")
        BOUND_WH=${bounds%%:*}; BOUND_WD=${bounds#*:}
        ot=$(timedirs "$W/w_of_$key")
        wcase_arms "$arms"
        return 0
    fi
    WCASE_OF[$wkey]=1
    rel=${entry%%:*}; rest=${entry#*:}; dtw=${rest%%:*}; bounds=${rest#*:}; key=$(basename "$rel")
    BOUND_WH=${bounds%%:*}; BOUND_WD=${bounds#*:}
    src="$TUT/multiphase/interFoam/$rel"
    # DTCHullMovingCoarse is not a tutorial: it is RAS/DTCHullMoving with its background block halved in each
    # direction (coarse_dtc_source), for the arms that need a hull the fluid moves and not that mesh
    if [ "$key" = DTCHullMovingCoarse ]; then
        src="$W/src_$key"
        coarse_dtc_source "$src" || { say "ARM W  $key: the block was not coarsened" FAIL; return 1; }
    fi
    [ -d "$src" ] || { say "ARM W  $key: tutorial missing" FAIL; return 1; }
    stage_allrun "$src" "$W/w_of_$key" "$dtw" || { say "ARM W  $key: meshing failed (see $W/w_of_$key/log.allrunmesh)" FAIL; return 1; }
    if [ "$key" = mixerVesselAMI ]; then
        # STAGED IN OPENFOAM ONLY, and not claimed: p_rgh (and pcorr, which takes $p_rgh) GAMG -> PCG with DIC.
        # GAMG across an AMI agglomerates the interface (cyclicAMIGAMGInterface), which is not ported; brae
        # reads the tutorial's OWN GAMG entry (kept beside as fvSolution.gamg), runs PCG with DIC in its place
        # and says so. So the oracle is OpenFOAM running the solver brae runs. Against OpenFOAM's own GAMG the
        # same brae run is 1.2e-09 off (p, 2026-10-01, pinned solves: GAMG stops at its 1000-iteration cap).
        cp "$W/w_of_$key/system/fvSolution" "$W/w_of_$key/system/fvSolution.gamg"
        python3 - "$W/w_of_$key/system/fvSolution" <<'EOF_PCG' || say "ARM W  mixerVesselAMI: the PCG staging did not apply" FAIL
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(\n    p_rgh\n    \{\n)\s*solver\s+GAMG;(.*?)\s*smoother\s+GaussSeidel;',
               r'\1        solver          PCG;\n        preconditioner  DIC;\2', s, flags=re.S)
open(p, 'w').write(s)
sys.exit(0 if n == 1 else 1)
EOF_PCG
    fi
    runof "$W/w_of_$key"
    ot=$(timedirs "$W/w_of_$key")
    [ "$(echo $ot | wc -w)" = 2 ] || { say "ARM W  $key: premise, OpenFOAM writes two steps [$ot]" FAIL; return 1; }
    wcase_arms "$arms"
}

# finish <what>: the arm's verdict, and what the caches saved
finish()
{
    echo "  cache: OpenFOAM runs reused $ORACLE_HITS, run $ORACLE_MISSES; meshes reused $MESH_HITS, built $MESH_MISSES"
    [ $fail -eq 0 ] && echo "PASS: $1"
    exit $fail
}
