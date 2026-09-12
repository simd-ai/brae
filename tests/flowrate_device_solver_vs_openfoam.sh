#!/usr/bin/env bash
# flowRateInletVelocity ON THE DeviceSimpleSolver LINEAGE -- both forms, all three step() variants.
#
# OpenFOAM recomputes this inlet inside EVERY momentum matrix constructor (fvMatrix.C:396 ->
# flowRateInletVelocityFvPatchVectorField.C:201-238) and ASSIGNS the patch value there,
# `operator==(avgU*n)` at .C:195-196. The case file's `value` is therefore a SEED that survives only
# until the first assembly (.C:93-97: a `value` entry wins at construction, and only its absence
# triggers an evaluate).
#
# THREE WAYS brae's DeviceSimpleSolver failed to do that, all fixed together because they are one
# missing update:
#   * The mask was keyed on bcCategory() == 9, which is the MASS form only -- a volumetricFlowRate
#     inlet got no mask and was never updated on ANY of the five drivers built on this class.
#   * The only two callers were compressible: revalidateAfterThermo (guarded `if (!compressible_)
#     return;`) and rhoSimpleStep. step() (incompressible SIMPLE) and pimpleStep() called neither, so
#     the mask was built for them and never read -- BOTH forms froze at the seed.
#   * The divisor was always gSum(rho_b*magSf). OpenFOAM uses gSum(magSf) for the volumetric form
#     (.C:208-210, rho is literally one{}) and rhoInlet*gSum(magSf) where no rho field is registered
#     (.C:233), which is every incompressible case.
# The update now lives in DeviceSimpleSolver::solveMomentumPredictor, the one method all three steps
# enter, beside the other updateCoeffs equivalents.
#
#   ARM 1  validation/rhoFRvol, volumetric, on all three LEGACY compressible entry points + the mirror
#   ARM 2  validation/incFR,  incompressible simpleFoam, volumetric AND mass
#   ARM 3  validation/pimFR,  pimpleFoam, volumetric AND mass
#   ARM 4  CONTROL that the gate tests the UPDATE and not the constructor: the same incompressible case
#          with the `value` entry DELETED passes even on the broken binary, because brae's patch-field
#          constructor then computes avgU itself. If this is the only arm passing, the defect is back.
#   ARM 5  CONTROL that the seed is a live variable: OpenFOAM's own answer must differ from the seeded
#          field by far more than the bound, or ARMs 1-3 prove nothing.
#   ARM 6  REFUSAL: massFlowRate on an incompressible solver with no `rhoInlet` -- OpenFOAM FatalErrors
#          (.C:225-231) and so must brae, by name, rather than assuming a density.
#   ARM 8  the REBUILT simpleFoam (BRAE_SIMPLEFOAM_V2=1), which had no flowRate code and no refusal
#          either, against the SAME OpenFOAM oracle ARM 2 uses.
#   ARM 9  REFUSAL, both incompressible arms x both unusable densities: `massFlowRate` with no
#          `rhoInlet`, where OpenFOAM FatalErrors (.C:225-231), and with `rhoInlet 0`, where OpenFOAM
#          passes its own guard, divides by zero and writes inf/nan while exiting 0. brae names both.
#   ARM 7  SEED INDEPENDENCE on the legacy compressible arm. OpenFOAM's dict constructor keeps the case
#          file's `value` and only evaluates when it is absent (.C:93-97), so the seed reaches the
#          INITIAL phi -- compressibleCreatePhi.H builds it before any updateCoeffs runs. The three
#          drivers re-seeded that patch UNCONDITIONALLY, overwriting a `value` OpenFOAM keeps. Each arm
#          is compared against its OWN OpenFOAM run, because OpenFOAM's answer legitimately moves with
#          the seed; what must hold is that brae tracks it whatever the seed is.
#
# FAIL-PROOF, measured in-session on the binary before the fix (relative L2 vs real OpenFOAM, every
# solver pinned at tolerance 1e-14 relTol 0), inlet frozen at the seed 5 against OpenFOAM's 5.166:
#   rhoFRvol legacy      U 1.1049e-02 (t=1)  3.2652e-02 (t=20)
#   ARM 7, validation/rhoFR on the legacy arm, by the seed in 0/U alone:
#     no `value` (OpenFOAM evaluates too)  U 4.8372e-12  1.0548e-11   <- this driver's floor, always passed
#     `value uniform (5 0 0)` (shipped)    U 6.5461e-06  1.5415e-06
#     `value uniform (0 0 0)` (OF tutorial) U 2.7276e-02  6.4649e-03
#   and OpenFOAM's OWN seeded-vs-unseeded answers differ by exactly 6.5461e-06 / 1.5415e-06, i.e. brae
#   with a seed was reproducing OpenFOAM without one. After the fix: 4.9e-12, 4.8e-10, 5.2e-11.
#   incFR volumetric     U 1.1049e-02 (t=1)  3.2770e-02 (t=20)
#   incFR mass rhoInlet 2  U 1.5466e-01 (t=1)  7.0180e-01 (t=20)   <- the worst of them
#   pimFR both forms     U 3.1951e-02        3.1984e-02
#   ARM 8, simpleFoam v2  U 1.1049e-02 (t=1)  3.2770e-02 (t=20), inlet frozen at 5.000 -- and V2 had no
#                         refusal either, so it was silent. After the fix: 2.4e-15 / 7.6e-16.
# and after the fix every one of those is at 1e-11 or below.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDD="${BUILD:-$ROOT/build}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
FLOOR=${FLOOR:-1e-9}
CONTROL_RATIO=${CONTROL_RATIO:-1e4}

for b in brae brae_rhoSimpleFoam brae_rhoSimpleFoam_legacy brae_rhoSimpleFoam_slice brae_pimpleFoam; do
    [ -x "$BUILDD/$b" ] || { echo "SKIP: $BUILDD/$b not built"; exit 77; }
done
for f in rhoFRvol incFR pimFR; do
    [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }
done
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for c in rhoSimpleFoam simpleFoam pimpleFoam blockMesh; do
    command -v $c > /dev/null 2>&1 || { echo "SKIP: $c not on PATH"; exit 77; }
done

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# $1 dst  $2 fixture  $3 sed expression applied to 0/U ("" = none)  $4 endTime  $5 writeInterval
# The linear tolerances are pinned HERE as well as in the fixtures, because validation/rhoFRvol is a
# pre-existing 500-iteration case: without it this gate would compare where two Krylov solvers stopped.
stage()
{
    rm -rf "$W/$1"; cp -r "$ROOT/validation/$2" "$W/$1"
    rm -rf "$W/$1"/[1-9]* "$W/$1"/0; cp -r "$W/$1/0.orig" "$W/$1/0"
    [ -n "$3" ] && sed -i "$3" "$W/$1/0/U"
    END="$4" WI="$5" python3 - "$W/$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', os.environ['END']),
             ('writeInterval', os.environ['WI']), ('writeControl', 'timeStep'),
             ('startFrom', 'startTime'), ('startTime', '0')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
s = re.sub(r'\bfunctions\s*\{.*\}\s*$', '', s, flags=re.S)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
open(f, 'w').write(s)
PYEOF
    ( cd "$W/$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on $1"; exit 1; }
}
MASS='s|volumetricFlowRate constant 0.05166|massFlowRate constant 0.10332; rhoInlet 2|'
NOVAL='s|; value uniform (5 0 0)||'
NORHOI='s|volumetricFlowRate constant 0.05166|massFlowRate constant 0.10332|'

# ---- ARM 1: the compressible legacy lineage, three binaries that share DeviceSimpleSolver ---------
stage fr_of rhoFRvol '' 20 1
( cd "$W/fr_of" && rhoSimpleFoam > log 2>&1 ) || { tail -5 "$W/fr_of/log"; echo "FAIL: OF on rhoFRvol"; exit 1; }
for a in drv bin slice mir; do
    stage "fr_$a" rhoFRvol '' 20 1
    case $a in
      drv)   ( cd "$W/fr_$a" && BRAE_U_SOLVER=ofOrder "$BUILDD/brae_rhoSimpleFoam"        -case "$W/fr_$a" > log 2>&1 ) ;;
      bin)   ( cd "$W/fr_$a" && BRAE_U_SOLVER=ofOrder "$BUILDD/brae_rhoSimpleFoam_legacy" -case "$W/fr_$a" > log 2>&1 ) ;;
      slice) ( cd "$W/fr_$a" && BRAE_U_SOLVER=ofOrder "$BUILDD/brae_rhoSimpleFoam_slice"  -case "$W/fr_$a" > log 2>&1 ) ;;
      mir)   ( cd "$W/fr_$a" && BRAE_U_SOLVER=ofOrder BRAE_RHOSIMPLEFOAM_MIRROR=cuda \
                                "$BUILDD/brae_rhoSimpleFoam" -case "$W/fr_$a" > log 2>&1 ) ;;
    esac || { tail -5 "$W/fr_$a/log"; echo "FAIL: brae arm fr_$a did not run"; exit 1; }
    grep -q "solvers/U solver" "$W/fr_$a/log" && say "fr_$a: no momentum-solver substitution" FAIL \
                                              || say "fr_$a: no momentum-solver substitution" ok
done

# ---- ARMs 2-4: incompressible and PIMPLE, both forms, plus the no-value control -------------------
for fx in incFR pimFR; do
    case $fx in incFR) SOLVER=simpleFoam;  BR="$BUILDD/brae";             END=20;    WI=1 ;;
                pimFR) SOLVER=pimpleFoam; BR="$BUILDD/brae_pimpleFoam"; END=0.002; WI=5 ;; esac
    for v in vol mass noval; do
        case $v in vol) EXPR='' ;; mass) EXPR="$MASS" ;; noval) EXPR="$NOVAL" ;; esac
        stage "${fx}_${v}_of" "$fx" "$EXPR" "$END" "$WI"
        ( cd "$W/${fx}_${v}_of" && $SOLVER > log 2>&1 ) \
            || { tail -5 "$W/${fx}_${v}_of/log"; echo "FAIL: OF $SOLVER on ${fx}_${v}"; exit 1; }
        stage "${fx}_${v}_br" "$fx" "$EXPR" "$END" "$WI"
        ( cd "$W/${fx}_${v}_br" && "$BR" -case "$W/${fx}_${v}_br" > log 2>&1 ) \
            || { tail -5 "$W/${fx}_${v}_br/log"; echo "FAIL: brae on ${fx}_${v}"; exit 1; }
    done
done

# ---- ARM 8: the rebuilt simpleFoam, same fixture and the same OpenFOAM runs ARM 2 already made ----
for v in vol mass noval; do
    case $v in vol) EXPR='' ;; mass) EXPR="$MASS" ;; noval) EXPR="$NOVAL" ;; esac
    stage "v2FR_${v}_br" incFR "$EXPR" 20 1
    ( cd "$W/v2FR_${v}_br" && BRAE_SIMPLEFOAM_V2=1 "$BUILDD/brae" -case "$W/v2FR_${v}_br" > log 2>&1 ) \
        || { tail -5 "$W/v2FR_${v}_br/log"; echo "FAIL: simpleFoam v2 did not run on incFR $v"; exit 1; }
done

W="$W" FLOOR="$FLOOR" CONTROL_RATIO="$CONTROL_RATIO" python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
W = os.environ['W']; FLOOR = float(os.environ['FLOOR']); RATIO = float(os.environ['CONTROL_RATIO'])

def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if not m:
        u = re.search(r'internalField\s+uniform\s+(\(.*?\)|[-+0-9.eE]+)\s*;', s); v = u.group(1)
        return np.array([float(x) for x in v.strip('()').split()]) if v.startswith('(') else np.array([float(v)])
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])

def rel(a, b, t, f):
    return float(np.linalg.norm(read(os.path.join(W, a, t, f)) - read(os.path.join(W, b, t, f)))
                 / np.linalg.norm(read(os.path.join(W, b, t, f))))

def inletUx(case, t):
    s = open(os.path.join(W, case, t, 'U')).read()
    blk = re.search(r'\n    inlet\s*\n    \{(.*?)\n    \}', s[s.find('boundaryField'):], re.S).group(1)
    u = re.search(r'value\s+uniform\s+\(([^)]*)\)', blk)
    if u: return float(u.group(1).split()[0])
    v = re.search(r'value\s+nonuniform\s+List<vector>\s*\n?\d+\s*\n?\(\s*\(([^)]*)\)', blk, re.S)
    return float(v.group(1).split()[0])

def times(d):
    ts = [t for t in os.listdir(os.path.join(W, d))
          if re.fullmatch(r'[0-9.eE+-]+', t) and t != '0' and os.path.isdir(os.path.join(W, d, t))]
    return sorted(ts, key=float)

ok = True
# ARM 1 -- the compressible legacy lineage. OpenFOAM's answer is 0.05166/0.01 = 5.166 exactly.
for a, label in (('fr_drv', 'legacy driver '), ('fr_bin', 'legacy binary '),
                 ('fr_slice', 'legacy slice  '), ('fr_mir', 'OF-mirror cuda')):
    for t in ('1', '20'):
        worst, wf = max(((rel(a, 'fr_of', t, f), f) for f in ('U', 'p', 'T')))
        good = worst < FLOOR
        print('     rhoFRvol %s t=%-3s worst %-2s %.4e  (bound %.1e)  inlet %.9f  %s'
              % (label, t, wf, worst, FLOOR, inletUx(a, t), 'ok' if good else 'FAIL'))
        ok = ok and good
        g2 = abs(inletUx(a, t) - 5.166) / 5.166 < 1e-9
        if not g2: print('        inlet Ux %.9f != OpenFOAM 5.166  FAIL' % inletUx(a, t)); ok = False

# ARMs 2-4 -- incompressible and PIMPLE, volumetric / mass / the no-value control.
for fx, ofx, label in (('incFR', 'incFR', 'simpleFoam   '), ('pimFR', 'pimFR', 'pimpleFoam   '),
                       ('v2FR',  'incFR', 'simpleFoam v2')):
    for v, note in (('vol', 'volumetric      '), ('mass', 'mass + rhoInlet 2'),
                    ('noval', 'no `value` CTRL ')):
        br, of = '%s_%s_br' % (fx, v), '%s_%s_of' % (ofx, v)
        ts = [t for t in times(br) if t in times(of)]
        for t in (ts[0], ts[-1]):
            flds = [f for f in ('U', 'p') if os.path.exists(os.path.join(W, of, t, f))]
            worst, wf = max(((rel(br, of, t, f), f) for f in flds))
            good = worst < FLOOR
            print('     %s %s t=%-8s worst %-2s %.4e  (bound %.1e)  inlet %.9f  %s'
                  % (label, note, t, wf, worst, FLOOR, inletUx(br, t), 'ok' if good else 'FAIL'))
            ok = ok and good

# ARM 5 -- CONTROL: OpenFOAM's own answer is far from the seeded field, so the arms above are live.
seed = np.zeros_like(read(os.path.join(W, 'incFR_vol_of', '1', 'U'))); seed[:, 0] = 5.0
ofU = read(os.path.join(W, 'incFR_vol_of', '20', 'U'))
d = float(np.linalg.norm(ofU - seed) / np.linalg.norm(ofU)) / FLOOR
good = d > RATIO
print('     control: OpenFOAM\'s answer is %.0e x the bound away from the seeded field   %s'
      % (d, 'ok' if good else 'FAIL (the seed is inert; nothing above proves anything)'))
ok = ok and good
sys.exit(0 if ok else 1)
PYEOF

# ---- ARM 7: the answer must not depend on 0/U's `value` seed, which OpenFOAM keeps ---------------
S5='s|value uniform (5 0 0)|value uniform (5 0 0)|'
S0='s|value uniform (5 0 0)|value uniform (0 0 0)|'
SNV='s|; value uniform (5 0 0)||'
for sd in s5 s0 snv; do
    case $sd in s5) EXPR="$S5" ;; s0) EXPR="$S0" ;; snv) EXPR="$SNV" ;; esac
    stage "seed_${sd}_of" rhoFR "$EXPR" 20 1
    ( cd "$W/seed_${sd}_of" && rhoSimpleFoam > log 2>&1 ) \
        || { tail -5 "$W/seed_${sd}_of/log"; echo "FAIL: OF on rhoFR seed $sd"; exit 1; }
    stage "seed_${sd}_br" rhoFR "$EXPR" 20 1
    ( cd "$W/seed_${sd}_br" && BRAE_U_SOLVER=ofOrder "$BUILDD/brae_rhoSimpleFoam" \
        -case "$W/seed_${sd}_br" > log 2>&1 ) \
        || { tail -5 "$W/seed_${sd}_br/log"; echo "FAIL: brae on rhoFR seed $sd"; exit 1; }
done

W="$W" SEED_FLOOR="${SEED_FLOOR:-1e-8}" CONTROL_RATIO="$CONTROL_RATIO" python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
W = os.environ['W']; FLOOR = float(os.environ['SEED_FLOOR']); RATIO = float(os.environ['CONTROL_RATIO'])

def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if not m:
        u = re.search(r'internalField\s+uniform\s+(\(.*?\)|[-+0-9.eE]+)\s*;', s); v = u.group(1)
        return np.array([float(x) for x in v.strip('()').split()]) if v.startswith('(') else np.array([float(v)])
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])

def rel(a, b, t, f):
    return float(np.linalg.norm(read(os.path.join(W, a, t, f)) - read(os.path.join(W, b, t, f)))
                 / np.linalg.norm(read(os.path.join(W, b, t, f))))

ok = True
for sd, note in (('s5',  '`value uniform (5 0 0)`  the shipped fixture '),
                 ('s0',  '`value uniform (0 0 0)`  OpenFOAM\'s tutorial'),
                 ('snv', 'no `value`  CONTROL: brae always passed here')):
    for t in ('1', '20'):
        worst, wf = max(((rel('seed_%s_br' % sd, 'seed_%s_of' % sd, t, f), f) for f in ('U', 'p', 'T')))
        good = worst < FLOOR
        print('     rhoFR seed %s t=%-3s worst %-2s %.4e  (bound %.1e)  %s'
              % (note, t, wf, worst, FLOOR, 'ok' if good else 'FAIL'))
        ok = ok and good

# CONTROL: the seed is a live variable in OpenFOAM's OWN answer, so the three arms above are genuinely
# different measurements and not the same run compared three times.
d = rel('seed_s0_of', 'seed_s5_of', '1', 'U') / FLOOR
good = d > RATIO
print('     control: OpenFOAM\'s own seed-(0 0 0) and seed-(5 0 0) answers differ by %.0e x the bound  %s'
      % (d, 'ok' if good else 'FAIL (the seed is inert; the arms above prove nothing)'))
ok = ok and good
sys.exit(0 if ok else 1)
PYEOF

# ---- ARMs 6 and 9: an unusable density must REFUSE by name, on both incompressible arms -----------
# `rhoInlet 0` is not the same case as an absent one: OpenFOAM's guard is `rhoInlet_ < 0`, so 0 passes
# it, gSum(rho*magSf) is zero, and OpenFOAM writes inf/nan and exits 0 rather than failing. brae names
# both rather than producing a wrong number or a NaN field.
RHOI0='s|volumetricFlowRate constant 0.05166|massFlowRate constant 0.05166; rhoInlet 0|'
for cfg in norhoi rhoi0; do
    case $cfg in norhoi) EXPR="$NORHOI"; WHAT="no rhoInlet" ;; rhoi0) EXPR="$RHOI0"; WHAT="rhoInlet 0 " ;; esac
    for arm in legacy v2; do
        stage "ref_${cfg}_${arm}" incFR "$EXPR" 20 1
        if [ "$arm" = v2 ]; then
            out=$( cd "$W/ref_${cfg}_${arm}" && BRAE_SIMPLEFOAM_V2=1 "$BUILDD/brae" \
                     -case "$W/ref_${cfg}_${arm}" 2>&1 || true )
        else
            out=$( cd "$W/ref_${cfg}_${arm}" && "$BUILDD/brae" -case "$W/ref_${cfg}_${arm}" 2>&1 || true )
        fi
        echo "$out" | grep -qi "rhoInlet" && ! [ -d "$W/ref_${cfg}_${arm}/1" ] \
            && say "massFlowRate with $WHAT is refused by name ($arm arm)" ok \
            || { echo "$out" | tail -3; say "massFlowRate with $WHAT is refused by name ($arm arm)" FAIL; }
    done
done

say "flowRateInletVelocity reaches every DeviceSimpleSolver step, in both forms" "$([ $fail = 0 ] && echo ok || echo FAIL)"
exit $fail
