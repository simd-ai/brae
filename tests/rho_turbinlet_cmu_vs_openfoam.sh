#!/usr/bin/env bash
# THE TURBULENT MIXING-LENGTH INLETS RESOLVE Cmu THE WAY OPENFOAM DOES, on both mirror arms -- rhoKE
# (epsilon, host and device) and rhoSST (omega, host only) at t=1..N from 0/, solvers pinned, against real
# rhoSimpleFoam's written output at every t.
#
# Queue item 13d claimed the two inlets read their OWN `Cmu` from the patch dictionary and that the
# mirror was wrong to hand them the model's coefficient. Read at the source, the precedence is the
# other way round:
#   * turbulentMixingLengthDissipationRateInletFvPatchScalarField.C:91 reads the patch `Cmu` (default
#     0.09), and :149 then takes `turbModel.coeffDict().getOrDefault<scalar>("Cmu", Cmu_)` -- the
#     MODEL's coeffDict wins whenever it carries a Cmu. kEpsilon's constructor puts one there
#     unconditionally (kEpsilon.C:102-108 getOrAddToDict; dimensionedType.C:377-393 ADDS the default
#     when absent), so under kEpsilon the patch entry is dead and the model's Cmu is the inlet's.
#     The header says as much (.H:84-85: "turbulence model, boundary condition dictionary, and default").
#   * turbulentMixingLengthFrequencyInletFvPatchScalarField.C:137-138 reads NO patch entry at all:
#     `coeffDict().getOrDefault<scalar>("Cmu", 0.09)`. kOmegaSST never adds a Cmu (kOmegaSSTBase.C has
#     none; betaStar is a different key), so the omega inlet takes kOmegaSSTCoeffs { Cmu } or 0.09 --
#     NEVER betaStar, which is what the mirror passed (kOmegaSST_cpp.cu).
#
#   ARM KE-A  rhoKE, epsilon inlet turbulentMixingLengthDissipationRateInlet { mixingLength L; }, no Cmu
#             anywhere -> 0.09 on both sides.                                           host + device
#   ARM KE-B  the same patch with `Cmu 0.12;` and no kEpsilonCoeffs -> OpenFOAM still 0.09 (the model's
#             added default overrides the patch); OpenFOAM's own KE-B must EQUAL its KE-A.  host + device
#   ARM KE-C  no patch Cmu, kEpsilonCoeffs { Cmu 0.12; } -> 0.12 at the inlet and in the model.
#   ARM SST-A rhoSST, omega inlet turbulentMixingLengthFrequencyInlet { mixingLength L; }, no Cmu.   host
#   ARM SST-B the same patch with `Cmu 0.12;` -> ignored by OpenFOAM; its SST-B must EQUAL its SST-A.
#   ARM SST-C kOmegaSSTCoeffs { Cmu 0.12; } -> 0.12 at the inlet, the model untouched (Cmu is not an
#             SST coefficient): a PURE inlet perturbation in OpenFOAM.
#   ARM SST-D kOmegaSSTCoeffs { betaStar 0.11; } and no Cmu -> the inlet keeps 0.09 while the model
#             takes 0.11: the substitution the old code made.
#   CONTROLS  OpenFOAM's own answers: KE-C vs KE-A, SST-C vs SST-A and SST-D vs SST-A must differ by
#             >= CONTROL_RATIO x the bound at t=N, and so must KE-A against a run whose mixingLength is
#             divided by (0.12/0.09)^0.75 -- the inlet-only equivalent of Cmu 0.12 under kEpsilon, since
#             no kEpsilon input can move the inlet's Cmu without moving the model's.
#   LOG       printCoeffs on: OpenFOAM's KE-A log prints `Cmu 0.09;` in its coefficient dict although the
#             case never set one (the entry :149 reads exists), and its SST-A log prints none.
#
#   REFUSALS  `Cmu 0;` and `mixingLength 0;` on the dissipation inlet are FatalIOErrors in OpenFOAM's
#             constructor (:89, :91 getCheck ge(SMALL)); both mirror arms must refuse naming the patch
#             and the entry. KE-B, with its explicit positive Cmu, is the control that a named entry runs.
#
# Fail-proof (2026-09-03, the pre-fix binary, which handed the omega inlet betaStar): SST-C at t=1 omega
# 1.73e-03 / k 9.25e-05 (max over t=1..10 omega 4.84e-03 / k 2.70e-03 / U 1.91e-05 / p 1.13e-05 /
# T 1.11e-05), its inlet omega 399.30 where OpenFOAM writes 371.59 (7.46e-02); SST-D at t=1 omega 1.21e-03 /
# k 7.77e-05 (max omega 3.86e-03 / k 2.39e-03), its inlet 379.76 against 399.30 (4.89e-02); exit 1. The KE
# arms sat at the floor under the old code, as the source says they must. With the item's own proposed
# fix instead (the PATCH Cmu replacing the model's on the epsilon inlet, a host-class mutation), the HOST
# arm fails KE-B (t=1 epsilon 5.23e-03 / k 2.61e-04; max over t=1..10 k 8.45e-03 / epsilon 7.94e-03 /
# U 4.09e-04 / p 2.16e-05 / T 2.31e-05; the inlet epsilon 418.04 where OpenFOAM writes 336.91, 2.41e-01)
# and KE-C (the inlet 336.91 against 418.04, 1.94e-01; epsilon 4.78e-03 at t=1), exit 1, while the device
# arm -- whose closure takes the model's Cmu directly -- stays at the floor. Fixed: PASS.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-10}
# The mixing length: with the fixtures' inlet k = 9.375 it gives epsilon = 0.09^0.75 * 9.375^1.5 / L =
# 337 and omega = sqrt(9.375)/(0.09^0.25 L) = 399, i.e. the seeds the fixtures ship (336.9 and 400).
L=${L:-0.014}
# Floors measured 2026-09-03 (max over t=1..10 and over the arms, against OpenFOAM):
#   rhoKE   host k 8.8e-13, epsilon 2.1e-12, U 7.3e-13, p 2.9e-12, T 9.7e-13
#           device k 3.3e-12, epsilon 5.8e-12, U 1.4e-12, p 3.1e-12, T 9.7e-13
#   rhoSST  host k 9.1e-13, omega 7.4e-13, U 6.7e-13, p 2.9e-12, T 9.8e-13
# Bounds ~30x the larger arm's floor; the inlet patch values (1.2e-12 / 3.4e-13) share the field's bound.
KE_K_BOUND=${KE_K_BOUND:-1e-10}; KE_EPS_BOUND=${KE_EPS_BOUND:-1.8e-10}; KE_U_BOUND=${KE_U_BOUND:-5e-11}
KE_P_BOUND=${KE_P_BOUND:-1e-10}; KE_T_BOUND=${KE_T_BOUND:-3e-11}
SST_K_BOUND=${SST_K_BOUND:-3e-11}; SST_OMEGA_BOUND=${SST_OMEGA_BOUND:-2.5e-11}; SST_U_BOUND=${SST_U_BOUND:-2e-11}
SST_P_BOUND=${SST_P_BOUND:-1e-10}; SST_T_BOUND=${SST_T_BOUND:-3e-11}
# OpenFOAM against OpenFOAM with a dead entry added: bit-identical in the measurement (0.0 on every field).
IDENTITY_BOUND=${IDENTITY_BOUND:-1e-14}
CONTROL_RATIO=${CONTROL_RATIO:-100}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoKE" ]  || { echo "SKIP: fixture validation/rhoKE missing"; exit 77; }
[ -d "$ROOT/validation/rhoSST" ] || { echo "SKIP: fixture validation/rhoSST missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh > /dev/null 2>&1     || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=${W:-$(mktemp -d)}; mkdir -p "$W"; [ -n "${KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# One meshed base per fixture; every arm is a copy with its own inlet block and coefficient subdict.
for fx in rhoKE rhoSST; do
    cp -r "$ROOT/validation/$fx" "$W/base_$fx"; rm -rf "$W"/base_$fx/[1-9]* "$W"/base_$fx/0 "$W"/base_$fx/log.*
    cp -r "$W/base_$fx/0.orig" "$W/base_$fx/0"
    ( cd "$W/base_$fx" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on $fx"; exit 1; }
done

stage()   # $1 fixture  $2 dst  $3 field (epsilon|omega)  $4 inlet block  $5 model  $6 coeffs block ("" = none)
{
    python3 - "$W/base_$1" "$2" "$N" "$3" "$4" "$5" "$6" <<'TURBCMUSTAGE'
import os, re, shutil, sys
src, dst, n, fld, inlet, model, coeffs = sys.argv[1:8]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
c = os.path.join(dst, 'system/controlDict'); s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', n), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(dst, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0; maxIter 2000;', s)
open(f, 'w').write(s)
p = os.path.join(dst, '0', fld); s = open(p).read()
s, k = re.subn(r'\binlet\s*\{[^{}]*\}', lambda m: 'inlet ' + inlet, s)
assert k == 1, 'expected one inlet block in 0/%s, found %d' % (fld, k)
open(p, 'w').write(s)
if coeffs:
    t = os.path.join(dst, 'constant/turbulenceProperties'); s = open(t).read()
    s, k = re.subn(r'(RAS\s*\{)', lambda m: m.group(1) + ' %sCoeffs %s' % (model, coeffs), s, count=1)
    assert k == 1, 'expected one RAS block in turbulenceProperties, found %d' % k
    open(t, 'w').write(s)
TURBCMUSTAGE
}

run()   # $1 dir  $2 arm (of|1|cuda)
{
    case "$2" in
        of)   ( cd "$1" && rhoSimpleFoam > run.log 2>&1 ) ;;
        *)    ( cd "$1" && BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BIN" -case "$1" > run.log 2>&1 ) ;;
    esac || { tail -5 "$1/run.log"; echo "FAIL: $2 did not run in $1"; exit 1; }
    [ -d "$1/$N" ] || { echo "FAIL: $2 wrote no $N/ in $1"; exit 1; }
}

# The inlet-only equivalent of Cmu 0.12 under kEpsilon: epsilon_b = Cmu^0.75 k_b^1.5 / L, so dividing L by
# (0.12/0.09)^0.75 moves the inlet exactly as Cmu 0.12 would while the model keeps 0.09.
LQ=$(python3 -c "print(repr($L / (0.12 / 0.09) ** 0.75))")
KE_INLET="{ type turbulentMixingLengthDissipationRateInlet; mixingLength $L; value uniform 336.9076; }"
KE_INLET_CMU="{ type turbulentMixingLengthDissipationRateInlet; mixingLength $L; Cmu 0.12; value uniform 336.9076; }"
KE_INLET_LQ="{ type turbulentMixingLengthDissipationRateInlet; mixingLength $LQ; value uniform 336.9076; }"
SST_INLET="{ type turbulentMixingLengthFrequencyInlet; mixingLength $L; value uniform 400; }"
SST_INLET_CMU="{ type turbulentMixingLengthFrequencyInlet; mixingLength $L; Cmu 0.12; value uniform 400; }"

stage rhoKE  "$W/of_KE_A"  epsilon "$KE_INLET"     kEpsilon  ""
stage rhoKE  "$W/of_KE_B"  epsilon "$KE_INLET_CMU" kEpsilon  ""
stage rhoKE  "$W/of_KE_C"  epsilon "$KE_INLET"     kEpsilon  "{ Cmu 0.12; }"
stage rhoKE  "$W/of_KE_LQ" epsilon "$KE_INLET_LQ"  kEpsilon  ""
stage rhoSST "$W/of_SST_A" omega   "$SST_INLET"     kOmegaSST ""
stage rhoSST "$W/of_SST_B" omega   "$SST_INLET_CMU" kOmegaSST ""
stage rhoSST "$W/of_SST_C" omega   "$SST_INLET"     kOmegaSST "{ Cmu 0.12; }"
stage rhoSST "$W/of_SST_D" omega   "$SST_INLET"     kOmegaSST "{ betaStar 0.11; }"
for c in of_KE_A of_KE_B of_KE_C of_KE_LQ of_SST_A of_SST_B of_SST_C of_SST_D; do
    grep -q "maxIter 2000" "$W/$c/system/fvSolution" || { echo "FAIL: the solver pin did not apply ($c)"; exit 1; }
    run "$W/$c" of
done
for arm in A B C; do
    eval "inl=\$KE_INLET"; [ $arm = B ] && inl="$KE_INLET_CMU"
    co=""; [ $arm = C ] && co="{ Cmu 0.12; }"
    stage rhoKE "$W/host_KE_$arm" epsilon "$inl" kEpsilon "$co"; run "$W/host_KE_$arm" 1
    stage rhoKE "$W/cuda_KE_$arm" epsilon "$inl" kEpsilon "$co"; run "$W/cuda_KE_$arm" cuda
done
for arm in A B C D; do
    inl="$SST_INLET"; [ $arm = B ] && inl="$SST_INLET_CMU"
    co=""; [ $arm = C ] && co="{ Cmu 0.12; }"; [ $arm = D ] && co="{ betaStar 0.11; }"
    stage rhoSST "$W/host_SST_$arm" omega "$inl" kOmegaSST "$co"; run "$W/host_SST_$arm" 1
done

# The refusals: OpenFOAM's constructor rejects both (getCheck ge(SMALL), :89 and :91); so must both arms,
# by name. A run that got past construction would have written 1/.
KE_INLET_CMU0="{ type turbulentMixingLengthDissipationRateInlet; mixingLength $L; Cmu 0; value uniform 336.9076; }"
KE_INLET_L0="{ type turbulentMixingLengthDissipationRateInlet; mixingLength 0; value uniform 336.9076; }"
stage rhoKE "$W/refuse_cmu0" epsilon "$KE_INLET_CMU0" kEpsilon ""
stage rhoKE "$W/refuse_l0"   epsilon "$KE_INLET_L0"   kEpsilon ""
for r in cmu0:Cmu l0:mixingLength; do
    d=${r%%:*}; key=${r##*:}
    for side in 1 cuda; do
        ( cd "$W/refuse_$d" && BRAE_RHOSIMPLEFOAM_MIRROR=$side "$BIN" -case "$W/refuse_$d" > "run_$side.log" 2>&1 ) && rc=0 || rc=$?
        if [ "$rc" != 0 ] && grep -q "turbulentMixingLengthDissipationRateInlet on patch 'inlet': $key must be >= SMALL" "$W/refuse_$d/run_$side.log"             && [ ! -d "$W/refuse_$d/1" ]; then
            say "refusal ($side): $key 0 on the dissipation inlet is refused by name" ok
        else
            tail -3 "$W/refuse_$d/run_$side.log"
            say "refusal ($side): $key 0 on the dissipation inlet is refused by name" FAIL
        fi
    done
done

# The written OpenFOAM inlet carries the BC type, so the arm ran the inlet it names.
grep -A3 'inlet' "$W/of_KE_A/1/epsilon" | grep -q 'turbulentMixingLengthDissipationRateInlet' \
    && say "OpenFOAM wrote the epsilon inlet as turbulentMixingLengthDissipationRateInlet" ok \
    || say "OpenFOAM wrote the epsilon inlet as turbulentMixingLengthDissipationRateInlet" FAIL
grep -A3 'inlet' "$W/of_SST_A/1/omega" | grep -q 'turbulentMixingLengthFrequencyInlet' \
    && say "OpenFOAM wrote the omega inlet as turbulentMixingLengthFrequencyInlet" ok \
    || say "OpenFOAM wrote the omega inlet as turbulentMixingLengthFrequencyInlet" FAIL
# printCoeffs: the coeffDict entry :149 reads exists under kEpsilon although the case never set it,
# and does not exist under kOmegaSST.
python3 - "$W/of_KE_A/run.log" "$W/of_SST_A/run.log" <<'TURBCMULOG' && say "OpenFOAM's printed kEpsilon coeffs carry Cmu unasked; its kOmegaSST coeffs carry none" ok \
    || say "OpenFOAM's printed kEpsilon coeffs carry Cmu unasked; its kOmegaSST coeffs carry none" FAIL
import re, sys
# printCoeffs prints coeffDict_ right after the model is selected. With no <model>Coeffs subdict in the
# case, optionalSubDict (RASModel.C:72) hands back the RAS dict itself, so the printed block is `RAS {..}`
# and the Cmu it shows under kEpsilon is the one getOrAddToDict wrote (kEpsilon.C:102-108).
def printed(log):
    m = re.search(r'Selecting RAS turbulence model \w+.*?\n\w+\n\{(.*?)\n\}', open(log).read(), re.S)
    return m.group(1) if m else None
ke, sst = printed(sys.argv[1]), printed(sys.argv[2])
ok = ke is not None and re.search(r'\bCmu\s+0\.09;', ke) is not None
ok = ok and sst is not None and re.search(r'\bCmu\b', sst) is None
sys.exit(0 if ok else 1)
TURBCMULOG

W="$W" N="$N" L="$L" CONTROL_RATIO="$CONTROL_RATIO" IDENTITY_BOUND="$IDENTITY_BOUND" \
KE_K_BOUND="$KE_K_BOUND" KE_EPS_BOUND="$KE_EPS_BOUND" KE_U_BOUND="$KE_U_BOUND" KE_P_BOUND="$KE_P_BOUND" KE_T_BOUND="$KE_T_BOUND" \
SST_K_BOUND="$SST_K_BOUND" SST_OMEGA_BOUND="$SST_OMEGA_BOUND" SST_U_BOUND="$SST_U_BOUND" SST_P_BOUND="$SST_P_BOUND" SST_T_BOUND="$SST_T_BOUND" \
python3 - <<'TURBCMUCMP' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], int(os.environ['N'])
ratio, ident = float(os.environ['CONTROL_RATIO']), float(os.environ['IDENTITY_BOUND'])

def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])

def patch_block(s, name):
    i = s.find('boundaryField')
    j = re.search(r'\n\s*' + re.escape(name) + r'\s*\n?\s*\{', s[i:])
    if not j: return None
    k = i + j.end() - 1
    depth, p = 0, k
    while p < len(s):
        if s[p] == '{': depth += 1
        elif s[p] == '}':
            depth -= 1
            if depth == 0: return s[k:p + 1]
        p += 1
    return None

def inlet_value(path, nfaces):
    blk = patch_block(open(path).read(), 'inlet')
    m = re.search(r'\bvalue\s+nonuniform\s+List<scalar>\s*(\d+)\s*\((.*?)\)\s*;', blk, re.S)
    if m:
        return np.array([float(x) for x in m.group(2).split()])
    u = re.search(r'\bvalue\s+uniform\s+([^;]+);', blk)
    return np.full(nfaces, float(u.group(1)))

def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

def relf(a, b, t, f):
    return rel(read(os.path.join(W, a, str(t), f)), read(os.path.join(W, b, str(t), f)))

ok = True
fx = {'KE':  (('k', 'epsilon', 'U', 'p', 'T'), {'k': 'KE_K', 'epsilon': 'KE_EPS', 'U': 'KE_U', 'p': 'KE_P', 'T': 'KE_T'}),
      'SST': (('k', 'omega', 'U', 'p', 'T'),   {'k': 'SST_K', 'omega': 'SST_OMEGA', 'U': 'SST_U', 'p': 'SST_P', 'T': 'SST_T'})}
bounds = {m: {f: float(os.environ[key[f] + '_BOUND']) for f in fields} for m, (fields, key) in fx.items()}

# The precedence, from OpenFOAM alone: a patch Cmu changes nothing under either model.
for m, tag in (('KE', 'the patch Cmu is dead under kEpsilon (the model coeffDict wins, :149)'),
               ('SST', 'the frequency inlet reads no patch Cmu (:137-138)')):
    worst = max(relf('of_%s_B' % m, 'of_%s_A' % m, t, f) for t in range(1, N + 1) for f in fx[m][0])
    good = worst < ident
    print('     identity: OpenFOAM %s-B vs %s-A max over t=1..%d, all fields %.1e   %s' % (m, m, N, worst, 'ok' if good else 'FAIL'))
    print('               %s' % tag)
    ok = ok and good

# Controls: OpenFOAM's own answers separate the arms, or a mirror ignoring the entry passes.
for m, a, b, tag in (('KE', 'of_KE_C', 'of_KE_A', 'kEpsilonCoeffs Cmu 0.12 vs none'),
                     ('KE', 'of_KE_LQ', 'of_KE_A', 'inlet-only equivalent (L / (0.12/0.09)^0.75) vs none'),
                     ('SST', 'of_SST_C', 'of_SST_A', 'kOmegaSSTCoeffs Cmu 0.12 vs none (inlet-only)'),
                     ('SST', 'of_SST_D', 'of_SST_A', 'kOmegaSSTCoeffs betaStar 0.11 vs none')):
    best = max((relf(a, b, N, f) / bounds[m][f], f) for f in fx[m][0])
    good = best[0] > ratio
    print('     control: OpenFOAM %-52s %9.0fx the %s bound at t=%d   %s' % (tag, best[0], best[1], N, 'ok' if good else 'FAIL (inert)'))
    ok = ok and good

# The arms.
arms = [('KE', 'host', a) for a in 'ABC'] + [('KE', 'cuda', a) for a in 'ABC'] + [('SST', 'host', a) for a in 'ABCD']
for m, side, a in arms:
    fields = fx[m][0]
    worst = {f: 0.0 for f in fields}
    first = {}
    for t in range(1, N + 1):
        for f in fields:
            r = relf('%s_%s_%s' % (side, m, a), 'of_%s_%s' % (m, a), t, f)
            worst[f] = max(worst[f], r)
            if t == 1: first[f] = r
    row = '     arm %s-%s %-4s' % (m, a, side)
    good = True
    for f in fields:
        g = worst[f] < bounds[m][f]
        row += '  %s %.2e%s' % (f, worst[f], '' if g else ' FAIL')
        good = good and g
    print(row + '   (max over t=1..%d; t=1: %s)' % (N, ' '.join('%s %.2e' % (f, first[f]) for f in fields)))
    ok = ok and good
    # The inlet itself: OpenFOAM writes the recomputed refValue as the patch `value`.
    f2 = fields[1]
    for t in (1, N):
        of = inlet_value(os.path.join(W, 'of_%s_%s' % (m, a), str(t), f2), 40)
        br = inlet_value(os.path.join(W, '%s_%s_%s' % (side, m, a), str(t), f2), 40)
        r = rel(br, of)
        g = r < bounds[m][f2]
        print('       inlet %s at t=%d: OpenFOAM mean %.6f, %s mean %.6f, rel %.2e   %s' % (f2, t, of.mean(), side, br.mean(), r, 'ok' if g else 'FAIL'))
        ok = ok and g
sys.exit(0 if ok else 1)
TURBCMUCMP
say "the mixing-length inlets take OpenFOAM's Cmu on every arm at t=1..$N" "$([ $fail = 0 ] && echo ok || echo FAIL)"

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
