#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on pressureInletOutletVelocity on a STATIC mesh, on both
# arms.
#
# WHY  movingCone is the only rhoCentralFoam tutorial with a pressureInletOutletVelocity patch, and it moves its mesh, so
#   only the CUDA arm ever ran one against OpenFOAM (the moving gate). On a static mesh the host reference arm
#   (BRAE_RHOCENTRALFOAM_ARM=cpu) accepted the patch and never updated it: U.correctBoundaryConditions() at
#   rhoCentralFoam.C:232 runs its updateCoeffs on this step's phi (valueFraction = neg(phi)*(I - n n)), and the host
#   patch kept the value it was read with for the whole run. The CUDA arm updates it at every step
#   (deviceUpdatePressureInletOutletVelocity). Found 2026-11-20: on piovBox the host arm's top U was the read (0 0 0)
#   at every write, up to 246 m/s from OpenFOAM's, while the CUDA arm's was within 4e-12 m/s.
# CASE  piovBox (tests/rho_central_cases.py piovBox, on the shockTube tutorial's thermo, schemes and solution): a 2-D
#   box at rest with a 3e5 Pa block in 1e5 Pa air, walls `slip`, the TOP open (U pressureInletOutletVelocity, p
#   fixedValue 1e5, T inletOutlet); deltaT 2e-6 fixed, 2000 steps to 4e-3, 16 writes at 17 digits. The top's flux is
#   exactly 0 until the blast arrives; then it leaves, and the expansion behind it draws it back in on part of the patch.
# REFERENCE  OpenFOAM serial; OpenFOAM's own spread decomposed in two (simple (2 1 1)) and in four (scotch, the partition
#   frozen: tests/data/rho_central_decomposition, piovBox).
# ENVELOPE (tests/rho_central_gate_lib.sh, declared here before the first run): every field at every write time within
#   ten times the larger of OpenFOAM's two spreads, never below BOUND (8e-14, the other static gates' floor); the top
#   patch's U, T and rho values by the same rule and the same measure as the fields: the largest difference on the
#   patch's faces over OpenFOAM serial's largest |field| over the cells at that write time. (A patch-relative measure, the
#   forwardStep gate's, divides by the patch's own largest value; here the top is at rest up to round-off, 1e-13 m/s,
#   until the blast arrives, and that measure compared round-off with round-off: OpenFOAM decomposed differed from
#   serial by 10 % of the noise there, and both brae arms by 120 %, before and after the fix alike.)
# ARMS  K the CUDA arm, H the host reference; each against OpenFOAM serial: the step count, time names and deltaT, the
#   Courant number, the write times, U p T rho at every write time, and the top patch's U T rho values.
# PREMISES  read off OpenFOAM's own written top U, not brae's: at some write time faces on the inflow branch (value
#   n*(n & U_cell), no tangential part) with a resolved normal speed (above 1e-3 m/s), and at some write time outflow faces
#   with a resolved tangential speed (above 1 m/s): both branches, and the tangential part an inflow face drops.
# CONTROL  motionPiovStatic (rhoCentralState.cuh: the patch never updated, its read value kept -- what the host arm did)
#   on the CUDA arm: red against OpenFOAM by more than 100 x the bound.
# ALSO  conservation to round-off and positivity (both arms); the two arms against each other (reported).
# NOT CLAIMED  tangentialVelocity (refused by the shared factory), a moving mesh (the moving gate), viscous flow, a
#   pressure other than fixedValue on the patch.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
CASES="$ROOT/tests/rho_central_cases.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam/shockTube"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/0.orig" ] || { echo "SKIP: no shockTube tutorial at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-piov.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_PIOV_BOUND:-8e-14}
fails=0
passes=0
. "$ROOT/tests/rho_central_gate_lib.sh"

# ---- the case and OpenFOAM ----------------------------------------------------------------------------------------
stage()
{
    local d=$1
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$TUT/system" "$TUT/constant" "$TUT/0.orig" "$d/"
    cp -r "$TUT/0.orig" "$d/0"
    python3 "$CASES" piovBox "$d" || { fail "stage $d: piovBox"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1" < /dev/null \
        || { fail "stage $d: blockMesh/setFields"; return 1; }
}
stage "$W/box" || exit 1
copy "$W/box" "$W/of"
copy "$W/box" "$W/of_np2"
copy "$W/box" "$W/of_np4"
ofrun "$W/of" &
ofnp "$W/of_np2" 2 &
ofnp "$W/of_np4" 4 &
wait
REFS="$W/of_np2 $W/of_np4"

# ---- brae: both arms, and the control ------------------------------------------------------------------------------
copy "$W/box" "$W/bK"
copy "$W/box" "$W/bH"
copy "$W/box" "$W/cS"
braerun "$W/bK" BRAE_RHOCENTRALFOAM_ARM=cuda
braerun "$W/bH" BRAE_RHOCENTRALFOAM_ARM=cpu
ctlrun "$W/cS" BRAE_RHOCENTRALFOAM_ARM=cuda BRAE_CONTROL_RCF=motionPiovStatic

# ---- premises, on OpenFOAM's own top U -----------------------------------------------------------------------------
prem=$(python3 - "$W/of" "$ROOT/tests" <<'EOF'
import os, re, sys
o = sys.argv[1]
sys.path.insert(0, sys.argv[2])
from rho_central_fields import patch_value
best_in = (0, 0.0, '')     # the write time with the most inflow faces, all at a resolved normal speed
best_tan = (0.0, '')       # the largest tangential speed on an outflow face
for t in sorted((d for d in os.listdir(o) if re.fullmatch(r'[0-9.e+-]+', d) and d != '0'), key=float):
    rows = patch_value(f'{o}/{t}/U', 'top')
    if len(rows) < 40:
        continue          # uniform: every face at rest
    inflow = [v for v in rows if abs(v[0]) < 1e-20 and v[1] < 0]          # n*(n & U_cell), n = (0 1 0)
    outflow = [v for v in rows if not (abs(v[0]) < 1e-20 and v[1] < 0)]
    if inflow and len(inflow) > best_in[0] and min(abs(v[1]) for v in inflow) > 1e-3:
        best_in = (len(inflow), min(abs(v[1]) for v in inflow), t)
    tan = max((abs(v[0]) for v in outflow), default=0.0)
    if tan > best_tan[0]:
        best_tan = (tan, t)
print(best_in[0], '%.3e' % best_in[1], best_in[2] or '-', '%.3e' % best_tan[0], best_tan[1] or '-')
EOF
)
read -r nin minin tin tan ttan <<< "$prem"
if [ "${nin:-0}" -gt 0 ]
then
    ok "premise: OpenFOAM's top takes the inflow branch at a resolved speed ($nin of 40 faces at t = $tin, normal speed at least $minin m/s, tangential part dropped)"
else
    fail "premise: OpenFOAM's top never takes the inflow branch at a resolved speed: the case does not exercise it"
fi
if le 1 "${tan:-0}"
then
    ok "premise: OpenFOAM's top outflow faces carry a resolved tangential speed (up to $tan m/s, at t = $ttan)"
else
    fail "premise: no resolved tangential speed on the top's outflow faces (${tan:-?} m/s)"
fi

# ---- the arms against OpenFOAM -------------------------------------------------------------------------------------
patcharm()   # patcharm <name> <brae dir> <OF dir> <patch> "<refs>": the patch's U, T, rho values, by the field check's measure
{
    local name=$1 b=$2 o=$3 pa=$4 refs=$5 line
    line=$(python3 - "$ROOT/tests" "$b" "$o" "$pa" "$BOUND" "$(times "$o" | tr ' ' ',')" $refs <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from rho_central_fields import internal, patch_value
b, o, pa, floor = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
times = [t for t in sys.argv[6].split(',') if t]
refs = sys.argv[7:]
def rel(x, t, f):
    # the largest |x - OpenFOAM serial| over the patch's faces, divided by OpenFOAM serial's largest |f| over the cells
    # at t: the denominator of the field check (rho_central_fields.py absdiff), so a patch at rest up to round-off is
    # measured against the flow's scale, not against its own noise
    u, v = np.broadcast_arrays(patch_value(f'{x}/{t}/{f}', pa), patch_value(f'{o}/{t}/{f}', pa))
    d = np.abs(u - v)
    return float(d.max()) / float(np.abs(internal(f'{o}/{t}/{f}')).max())
over, worst = [], (0.0, '')
for t in times:
    for f in ('U', 'T', 'rho'):
        v = rel(b, t, f)
        sp = max((rel(r, t, f) for r in refs), default=0.0)
        bd = max(floor, 10 * sp)
        if v > bd:
            over.append(f'{t}:{f}({v:.3e}>{bd:.3e})')
        if v / bd > worst[0]:
            worst = (v / bd, f'{v:.3e} at t = {t}, {f} (OpenFOAM\'s own {sp:.3e})')
if over:
    print('FAIL', ' '.join(over))
else:
    print('OK', '%.3f' % worst[0], worst[1])
PY
)
    case $line in
        OK*)   read -r _ q rest <<< "$line"
               ok "$name: $pa's U, T, rho values at every write time within ten times OpenFOAM's own spread, relative to the field's scale; closest ($q of its bound): $rest" ;;
        FAIL*) fail "$name: $pa values above the bound at ${line#FAIL }" ;;
        *)     fail "$name: $pa comparison did not run: $line" ;;
    esac
}
arm "piovBox (CUDA)" "$W/bK" "$W/of" "$REFS" "$BOUND" fixed
arm "piovBox (host reference)" "$W/bH" "$W/of" "$REFS" "$BOUND" fixed
patcharm "piovBox (CUDA)" "$W/bK" "$W/of" top "$REFS"
patcharm "piovBox (host reference)" "$W/bH" "$W/of" top "$REFS"
red "motionPiovStatic, piovBox (CUDA: the patch never updated)" "$W/cS" "$W/of" "$REFS"

# ---- conservation, positivity; CPU against GPU (reported) ----------------------------------------------------------
balance bK
balance bH
tl=$(times "$W/bK" | awk '{print $NF}')
echo "     CPU against GPU, piovBox at t = $tl: $(python3 "$FIELDS" maxrel "$W/bH/$tl" "$W/bK/$tl" | sed 's/^maxrel //')"
echo "rho_central_piov_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 17 ]
