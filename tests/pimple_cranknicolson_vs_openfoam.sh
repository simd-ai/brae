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
# PREMISE: OpenFOAM with ddtCorr yes differs from OpenFOAM with ddtCorr no at step 2 (C9), so the arm can see the term.
# CONTROL BRAE_CONTROL_CN_NO_DDTCORR=1 (the correction omitted, as before): C9 red.
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28, job 5865262), worst entry over steps 1-6:
#   before (develop 191bc2dc): C9 1.0e+02, C1 1.4e+02 (p internal at step 5; phi 9.9e-02, 1.0e-01)
#   after: C9 3.2e-09, C1 3.4e-09 (p internal at step 5); U 6e-12 and phi 1.5e-11 at every step. p's ABSOLUTE difference is flat
#   at 2-4e-7 from step 1 to 6 (the p solves' tolerance), while its scale falls from 1.8e4 to 1.1e2 as the start-up pressure
#   decays, so the relative entry rises to 3.2e-09: a floor, not the correction (phi carries the correction and holds 1.5e-11).
#   BOUND 1e-8, three times it. Premise 2.3 (p at step 2); control 1.0e+02.
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

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
