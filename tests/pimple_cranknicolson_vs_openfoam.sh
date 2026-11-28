#!/usr/bin/env bash
# pimpleFoam's CrankNicolson fvc::ddtCorr against OpenFOAM v2412, step by step.
#
# WHAT IT HOLDS. CrankNicolsonDdtScheme::fvcDdtPhiCorr (.C:1262-1320) keeps two DDt0Fields of its own, ddtCorrDdt0(U) and
# ddtCorrDdt0(phi), created zero on the first step and evaluated once a step after it (ddt0 = rDtCoef0*(old - oldOld) -
# offCentre(ddt0)); the correction is ddtCouplingCoeff*((rDtCoef*phi_o + offCentre(dphidt0)) - (Sf & interpolate(rDtCoef*U_o
# + offCentre(ddt0)))). U's old-old level exists from the first step (fvmDdt asks for it), phi's is created at the first
# evaluation as a copy of phi.oldTime(), so dphidt0 starts from zero while ddt0 does not. brae omitted the term under
# CrankNicolson with a notice: 2.3 (p) from OpenFOAM at step 2 on pitzDaily, 5e-11 with ddtCorr off in both codes.
#
# THE CASE. incompressible/pimpleFoam/RAS/pitzDaily (kEpsilon) from rest at a fixed deltaT 1e-4, six steps written at
# writePrecision 17, every solve tight (tolerance 1e-12, relTol 0), PIMPLE ddtCorr yes.
#
# ARMS, every entry of p U k epsilon nut on its own scale (tests/pimple_entry_compare.py) at steps 1-6 within BOUND:
#   C9  CrankNicolson 0.9 (offCentre multiplies by 0.9)
#   C1  CrankNicolson 1 (offCentre returns ddt0 itself)
# RESTART of C9 from the third step (OpenFOAM's own restart equals its continuous run, the premise):
#   W   brae writes the files OpenFOAM writes there (U_0 phi_0 k_0 epsilon_0, ddt0(U|k|epsilon), ddtCorrDdt0(U|phi)), the _0 and
#       ddtCorrDdt0 values and the ddt0 fields' internal values within BOUND (brae writes ddt0's patch values zero; OpenFOAM's
#       fvmDdt reads the internal field only)
#   R   brae restarted from its own write, O brae from OpenFOAM's, B OpenFOAM from brae's: each against OpenFOAM's continuous run
# PREMISE: OpenFOAM with ddtCorr yes differs from OpenFOAM with ddtCorr no at step 2 (C9), so the arm can see the term.
# CONTROLS, each red on the check it guards:
#   BRAE_CONTROL_CN_NO_DDTCORR=1          the correction omitted, as before: C9
#   BRAE_CONTROL_CN_NO_CORR_RESTART=1     ddtCorrDdt0(U|phi) neither written nor read: O
#   BRAE_CONTROL_BACKWARD_NO_OLD_READ=1   the old levels not seeded at the restart: O
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28, job 5865262), worst entry over steps 1-6:
#   before (develop 191bc2dc): C9 1.0e+02, C1 1.4e+02 (p internal at step 5; phi 9.9e-02, 1.0e-01)
#   after: C9 3.2e-09, C1 3.4e-09 (p internal at step 5); U 6e-12 and phi 1.5e-11 at every step. p's ABSOLUTE difference is flat
#   at 2-4e-7 from step 1 to 6 (the p solves' tolerance), while its scale falls from 1.8e4 to 1.1e2 as the start-up pressure
#   decays, so the relative entry rises to 3.2e-09: a floor, not the correction (phi carries the correction and holds 1.5e-11).
#   BOUND 1e-8, three times it. Premise 2.3 (p at step 2); control 1.0e+02.
#   RESTART (job 5866189): before (develop ecaa29ed) R 2.7e+01, O 2.7e+01, B could not start (brae's ddt0(k|epsilon) written
#   dimensionless: "Different dimensions for '(a - b)'"), W missing U_0 phi_0 k_0 epsilon_0 ddtCorrDdt0(U|phi); after R 3.2e-09,
#   O 1.0e-10, B 3.3e-09 (p internal, the continuous run's floor), W the same files with values within 1.5e-10, ddt0(U|k) 1.6e-11,
#   ddt0(epsilon) 1.5e-12 off the 472 wall-function cells (1.8e-02 at them). Controls 1.8e+01 (ddtCorr fields), 2.7e+01 (levels).
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_CN_BOUND:-1e-8}
PREMISE=1e-6
FIELDS="p U k epsilon nut"
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUT="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}/incompressible/pimpleFoam/RAS/pitzDaily"
[ -d "$TUT/system" ] || { echo "SKIP: no pimpleFoam/RAS/pitzDaily tutorial at $TUT"; exit 77; }
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
W=${BRAE_CN_KEEP:-$(mktemp -d)}
[ -n "${BRAE_CN_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-96s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
# stage <dir> <ddt scheme> <ddtCorr yes|no>
stage()
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"
    python3 - "$1" "$2" "$3" << 'PY'
import re
import sys
d, ddt, corr = sys.argv[1:4]
c = open(d + '/system/controlDict').read()
for k, v in (('adjustTimeStep', 'no'), ('deltaT', '0.0001'), ('endTime', '0.0006'), ('writeControl', 'timeStep'),
             ('writeInterval', '1'), ('writePrecision', '17'), ('timePrecision', '18')):
    c, n = re.subn(r'^%s\s+[^;]*;' % k, '%-16s%s;' % (k, v), c, flags=re.M)
    assert n == 1, k
open(d + '/system/controlDict', 'w').write(c)
s = open(d + '/system/fvSchemes').read()
s2 = re.sub(r'(ddtSchemes\s*\{\s*default\s+)Euler;', r'\g<1>%s;' % ddt, s)
assert s2 != s, 'pitzDaily no longer ships `ddtSchemes { default Euler; }`'
open(d + '/system/fvSchemes', 'w').write(s2)
s = open(d + '/system/fvSolution').read()
i = s.index('solvers')
j = s.index('PIMPLE')
blk = re.sub(r'(\btolerance\s+)[^;]+;', r'\g<1>1e-12;', s[i:j])
blk = re.sub(r'(\brelTol\s+)[^;]+;', r'\g<1>0;', blk)
blk = re.sub(r'(\bsolver\s+[^;]+;)', r'\g<1>\n        maxIter         5000;', blk)
s = s[:i] + blk + s[j:]
s = re.sub(r'(PIMPLE\s*\{)', r'\1\n    ddtCorr         %s;' % corr, s, count=1)
open(d + '/system/fvSolution', 'w').write(s)
PY
}
ofrun() { ( cd "$1" && pimpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM in $1"; tail -5 "$1/log"; exit 1; }; }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ) || { echo "FAIL: brae in $d"; tail -5 "$d/log"; exit 1; }; }
tlist() { (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -vx 0 | sort -g | paste -sd,); }
worst() { python3 "$CMP" "$@" | awk '/^MAX/{print $2, $3, $4, $5}'; }
le() { python3 -c "import sys; sys.exit(0 if float('$1') <= float('$2') else 1)"; }
gt() { python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }
echo "pimple_cranknicolson_vs_openfoam: pitzDaily under CrankNicolson, deltaT 1e-4, six steps, ddtCorr yes"

for a in "C9:CrankNicolson 0.9" "C1:CrankNicolson 1"; do
    tag=${a%%:*}
    ddt=${a#*:}
    stage "$W/$tag-of" "$ddt" yes
    ofrun "$W/$tag-of"
    stage "$W/$tag-br" "$ddt" yes
    brrun "$W/$tag-br"
    T=$(tlist "$W/$tag-of")
    w=$(worst "$W/$tag-of" "$W/$tag-br" "$T" $FIELDS)
    wp=$(worst "$W/$tag-of" "$W/$tag-br" "$T" phi)
    if [ "$BOUND" = report ]; then
        say "$tag ($ddt): worst entry at steps 1-6 $w; phi ${wp%% *}" INFO
        continue
    fi
    le "${w%% *}" "$BOUND" && say "$tag ($ddt): every entry of $FIELDS at steps 1-6 within $BOUND (worst $w)" PASS \
                           || say "$tag ($ddt): every entry of $FIELDS at steps 1-6 within $BOUND (worst $w)" FAIL
    le "${wp%% *}" "$BOUND" && say "$tag ($ddt): phi at steps 1-6 within $BOUND (worst ${wp%% *})" PASS \
                            || say "$tag ($ddt): phi at steps 1-6 within $BOUND (worst ${wp%% *})" FAIL
done

# ---- RESTART under CrankNicolson 0.9, from the third step ---------------------------------------------------------------
# OpenFOAM writes and reads U_0, phi_0 (from the step after ddtCorr creates phi's old-old level), k_0, epsilon_0, ddt0(U|k|
# epsilon) and ddtCorrDdt0(U|phi); a DDt0Field read so is evaluated on the first restarted step with both coefficients
# 1 + ocCoeff (CrankNicolsonDdtScheme.C:48-61, 102-165). Each arm against OpenFOAM's continuous C9 at the steps after the third.
# from <dir> <source case> <time>: the C9 staging with <source>'s <time> beside its 0, startFrom latestTime
from()
{
    stage "$1" "CrankNicolson 0.9" yes
    cp -r "$2/$3" "$1/"
    sed -i -E "s/^startFrom[[:space:]][^;]*;/startFrom       latestTime;/" "$1/system/controlDict"
    grep -q -E "^startFrom[[:space:]]+latestTime;" "$1/system/controlDict" || { echo "FAIL: startFrom in $1"; exit 1; }
}
T3=$(tlist "$W/C9-of" | cut -d, -f3)
AFTER=$(tlist "$W/C9-of" | cut -d, -f4-)
judge()   # <label> <reference case> <case> <times> [fields]
{
    local w
    w=$(worst "$2" "$3" "$4" ${5:-$FIELDS})
    [ -n "$w" ] || { say "$1: nothing compared" FAIL; return; }
    if [ "$BOUND" = report ]; then
        say "$1 (worst $w)" INFO
        return
    fi
    le "${w%% *}" "$BOUND" && say "$1 within $BOUND (worst $w)" PASS || say "$1 within $BOUND (worst $w)" FAIL
}
from "$W/R-of" "$W/C9-of" "$T3"; ofrun "$W/R-of"
w=$(worst "$W/C9-of" "$W/R-of" "$AFTER" $FIELDS)
le "${w%% *}" 0 && say "premise: OpenFOAM restarted from its $T3 equals its continuous run (worst $w)" PASS \
                || say "premise: OpenFOAM restarted from its $T3 equals its continuous run (worst $w)" FAIL
ofw=$(ls "$W/C9-of/$T3" | sort | tr '\n' ' ')
brw=$(ls "$W/C9-br/$T3" | sort | tr '\n' ' ')
[ "$brw" = "$ofw" ] && say "W: brae writes the files OpenFOAM writes at $T3 ($brw)" PASS \
                    || say "W: brae writes the files OpenFOAM writes at $T3 (brae: $brw; OpenFOAM: $ofw)" FAIL
judge "W: brae's U_0 phi_0 k_0 epsilon_0 ddtCorrDdt0(U) ddtCorrDdt0(phi) at $T3 against OpenFOAM's" "$W/C9-of" "$W/C9-br" "$T3" \
      "U_0 phi_0 k_0 epsilon_0 ddtCorrDdt0(U) ddtCorrDdt0(phi)"
# ddt0's patch values are written zero by brae: OpenFOAM's fvmDdt reads the internal field only (.C fvmDdt, the source).
# ddt0(epsilon) is held outside the cells next to an epsilonWallFunction patch: OpenFOAM's epsilon_0 of the first step is the
# copy oldTime() makes after the wall function has set those cells (GeometricField.C:945-975), brae's the start's value, and
# the epsilon equation replaces those rows by the wall function's values, so the difference reaches nothing else. Measured
# there 1.8e-02 (reported), elsewhere 1.5e-12 (Arrhenius 5866005).
w=$(python3 "$CMP" "$W/C9-of" "$W/C9-br" "$T3" "ddt0(U)" "ddt0(k)" | awk '$3 == "internal" {print $10, $2}' | sort -g -r | head -n 1)
we=$(python3 - "$W/C9-of" "$W/C9-br" "$T3" << 'PY'
import re
import sys
of, br, t = sys.argv[1:4]
def internal(path):
    s = open(path).read()
    i = s.index('internalField')
    return [float(x) for x in re.search(r'\(\s*(.*?)\n\)', s[i:], re.S).group(1).split()]
poly = of + '/constant/polyMesh'
owner = [int(x) for x in re.search(r'\(\s*(.*?)\n\)', open(poly + '/owner').read(), re.S).group(1).split()]
bnd = open(poly + '/boundary').read()
eps0 = open(of + '/0/epsilon').read()
walls = set()
for name, blk in re.findall(r'(\w+)\s*\{([^{}]*)\}', bnd):
    m = re.search(name + r'\s*\{[^}]*type\s+(\w+)', eps0)
    if m and m.group(1) == 'epsilonWallFunction':
        n = int(re.search(r'nFaces\s+(\d+)', blk).group(1))
        s0 = int(re.search(r'startFace\s+(\d+)', blk).group(1))
        walls.update(owner[s0:s0 + n])
a = internal('%s/%s/ddt0(epsilon)' % (of, t))
b = internal('%s/%s/ddt0(epsilon)' % (br, t))
scale = max(abs(x) for x in a)
other = max(abs(a[i] - b[i]) for i in range(len(a)) if i not in walls) / scale
wall = max([abs(a[i] - b[i]) for i in walls] or [0.0]) / scale
print('%.3e %.3e %d' % (other, wall, len(walls)))
PY
)
if [ "$BOUND" = report ]; then
    say "W: brae's ddt0(U|k) internal fields at $T3 (worst ${w:-none}); ddt0(epsilon) off/at the wall cells ${we}" INFO
else
    [ -n "$w" ] && le "${w%% *}" "$BOUND" && say "W: brae's ddt0(U|k) internal fields at $T3 within $BOUND (worst $w)" PASS \
        || say "W: brae's ddt0(U|k) internal fields at $T3 within $BOUND (worst ${w:-none})" FAIL
    set -- $we
    [ -n "${1:-}" ] && le "$1" "$BOUND" \
        && say "W: brae's ddt0(epsilon) at $T3 off the $3 wall-function cells within $BOUND ($1; at them $2, reported)" PASS \
        || say "W: brae's ddt0(epsilon) at $T3 off the wall-function cells within $BOUND (${we:-none})" FAIL
    set --
fi
from "$W/R-br" "$W/C9-br" "$T3"; brrun "$W/R-br"
judge "R: brae restarted from its own $T3 against OpenFOAM continuous" "$W/C9-of" "$W/R-br" "$AFTER"
from "$W/O-br" "$W/C9-of" "$T3"; brrun "$W/O-br"
judge "O: brae from OpenFOAM's $T3 against OpenFOAM continuous" "$W/C9-of" "$W/O-br" "$AFTER"
from "$W/B-of" "$W/C9-br" "$T3"; ofrun "$W/B-of"
judge "B: OpenFOAM from brae's $T3 against OpenFOAM continuous" "$W/C9-of" "$W/B-of" "$AFTER"

# PREMISE: the term moves OpenFOAM
stage "$W/C9-of-n" "CrankNicolson 0.9" no
ofrun "$W/C9-of-n"
T2=$(tlist "$W/C9-of" | cut -d, -f2)
w=$(worst "$W/C9-of" "$W/C9-of-n" "$T2" $FIELDS)
gt "${w%% *}" "$PREMISE" && say "premise: OpenFOAM's ddtCorr moves C9 at step 2 ($w)" PASS \
                         || say "premise: OpenFOAM's ddtCorr moves C9 at step 2 ($w)" FAIL

# CONTROL
CB=$BOUND
[ "$CB" = report ] && CB=1e-8
stage "$W/C9-ctl" "CrankNicolson 0.9" yes
brrun "$W/C9-ctl" BRAE_CONTROL_CN_NO_DDTCORR=1
w=$(worst "$W/C9-of" "$W/C9-ctl" "$(tlist "$W/C9-of")" $FIELDS)
gt "${w%% *}" "$CB" && say "control BRAE_CONTROL_CN_NO_DDTCORR=1: C9 goes red ($w)" PASS \
                    || say "control BRAE_CONTROL_CN_NO_DDTCORR=1: C9 goes red ($w)" FAIL

from "$W/ctl-o1" "$W/C9-of" "$T3"
brrun "$W/ctl-o1" BRAE_CONTROL_CN_NO_CORR_RESTART=1
w=$(worst "$W/C9-of" "$W/ctl-o1" "$AFTER" $FIELDS)
gt "${w%% *}" "$CB" && say "control BRAE_CONTROL_CN_NO_CORR_RESTART=1: O goes red ($w)" PASS \
                    || say "control BRAE_CONTROL_CN_NO_CORR_RESTART=1: O goes red ($w)" FAIL
from "$W/ctl-o2" "$W/C9-of" "$T3"
brrun "$W/ctl-o2" BRAE_CONTROL_BACKWARD_NO_OLD_READ=1
w=$(worst "$W/C9-of" "$W/ctl-o2" "$AFTER" $FIELDS)
gt "${w%% *}" "$CB" && say "control BRAE_CONTROL_BACKWARD_NO_OLD_READ=1: O goes red ($w)" PASS \
                    || say "control BRAE_CONTROL_BACKWARD_NO_OLD_READ=1: O goes red ($w)" FAIL

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
