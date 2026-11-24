#!/usr/bin/env bash
# pimpleFoam's flux-dependent boundary updates on incompressible/pimpleFoam/RAS/TJunction, step by step against
# OpenFOAM v2412.
#
# WHAT IT HOLDS. OpenFOAM updates each boundary condition inside its own field's equation: uniformTotalPressure in the
# constructor of every pressure corrector's fvm::laplacian(rAtU, p), from U's patch values and phi as they stand there;
# pressureInletOutletVelocity and inletOutlet at every U.correctBoundaryConditions(), from the flux the corrector just
# made; turbulentIntensityKineticEnergyInlet, turbulentMixingLengthDissipationRateInlet and k's and epsilon's
# inletOutlet in kEpsilon::correct, from U's and k's PATCH values and the last corrector's flux; and the limitedLinear
# limiter of k and epsilon reads their STORED patch values, which those updates do not move. brae's pimpleStep did the
# updates once, at the momentum assembly, from the flux before the step and from refValue arrays: p_b = p0 exactly at
# step 1, the inlet velocity zeroGradient through step 1, the inlet k and epsilon EXACTLY zero from step 2.
#
# THE CASE. The tutorial as shipped but for a fixed deltaT 0.001 (adjustTimeStep no) to 0.003, every step written at
# writePrecision 17, and every linear solve tight (tolerance 1e-12, relTol 0, maxIter 5000), so that what is left is
# the boundary arithmetic and not two solvers' stopping points (brae's pressure solve is its AMG-PCG, OpenFOAM's GAMG).
#   T1  kEpsilon, the p0 table (0 10) (1 40)
#   T2  T1 laminar: the flow boundaries alone (p, U)
#   T3  T1 with p0 constant 10: the total-pressure update without p0's time dependence
#
# CHECKS, each entry (the internal field and every patch's value) of p U k epsilon nut at 0.001, 0.002, 0.003 on its
# own scale (tests/pimple_entry_compare.py): rel <= BOUND. Controls, each must go red on the entry it guards:
#   BRAE_CONTROL_TOTAL_PRESSURE_AT_MOMENTUM=1  T2, inlet p at 0.001
#   BRAE_CONTROL_PIOV_ASSEMBLY_ONLY=1          T2, inlet U at 0.001
#   BRAE_CONTROL_TURB_INLET_REFVALUE=1         T1, inlet k at 0.002
#   BRAE_CONTROL_TURB_SWITCH_AT_MOMENTUM=1     T1, outlet1 k at 0.001
#   BRAE_CONTROL_TURB_GRAD_AFTER_UPDATE=1      T1, internal k at 0.001
# MEASURED (JUPITER, one GH200, OpenFOAM v2412 serial, 2026-11-25): worst entry over steps 1-3
# T1 4.9e-08, T2 5.6e-08, T3 1.8e-08 (U internal at 0.003, in the outlet2 branch; p within 6.6e-10), so BOUND 1e-6 is a
# decade above. Before these changes: T2 inlet p 1.7e-05 and U 2.4e-04 at step 1, T1 inlet k 1.0 at step 2. Controls:
# 1.7e-05, 6.7e-05, 1.0, 2.5e-01, 1.3e-05, each on its own entry. 31 s.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_TJ_BOUND:-1e-6}
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUT="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}/incompressible/pimpleFoam/RAS/TJunction"
[ -d "$TUT/system" ] || { echo "SKIP: no pimpleFoam/RAS/TJunction tutorial at $TUT"; exit 77; }
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
W=${BRAE_TJ_KEEP:-$(mktemp -d)}
[ -n "${BRAE_TJ_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-86s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
TIMES=0.001,0.002,0.003

setkey()
{
    sed -i -E "s/^$2[[:space:]][^;]*;/$2 $3;/" "$1/system/controlDict"
    grep -q -E "^$2 $3;" "$1/system/controlDict" || { echo "FAIL: could not set $2 $3"; exit 1; }
}
tight()
{
    python3 - "$1/system/fvSolution" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
i = s.index('solvers')
j = s.index('PIMPLE')
blk = re.sub(r'(\btolerance\s+)[^;]+;', r'\g<1>1e-12;', s[i:j])
blk = re.sub(r'(\brelTol\s+)[^;]+;', r'\g<1>0;', blk)
blk = re.sub(r'(\bsolver\s+[^;]+;)', r'\g<1>\n        maxIter         5000;', blk)
open(p, 'w').write(s[:i] + blk + s[j:])
EOF
}
mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
for k in "adjustTimeStep +yes" "deltaT +0.001;"; do
    grep -qE "^$k" "$W/base/system/controlDict" || { echo "FAIL: the tutorial no longer has \`$k\` -- the gate's premise moved"; exit 1; }
done
grep -q 'type            uniformTotalPressure;' "$W/base/0/p" \
    && grep -q 'type            pressureInletOutletVelocity;' "$W/base/0/U" \
    && grep -q 'type            turbulentIntensityKineticEnergyInlet;' "$W/base/0/k" \
    && grep -q 'type            turbulentMixingLengthDissipationRateInlet;' "$W/base/0/epsilon" \
    || { echo "FAIL: the tutorial's inlet conditions moved -- the gate's premise moved"; exit 1; }
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
setkey "$W/base" adjustTimeStep no
setkey "$W/base" endTime 0.003
setkey "$W/base" writeControl timeStep
setkey "$W/base" writeInterval 1
setkey "$W/base" writePrecision 17
tight "$W/base"
stage() { mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; }
stage "$W/T1"
stage "$W/T2"
sed -i -E 's/^simulationType[[:space:]]+RAS;/simulationType laminar;/' "$W/T2/constant/turbulenceProperties"
grep -q '^simulationType laminar;' "$W/T2/constant/turbulenceProperties" || { echo "FAIL: T2 is not laminar"; exit 1; }
stage "$W/T3"
python3 - "$W/T3/0/p" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s2 = re.sub(r'p0\s+table\s*\(\s*\(0 10\)\s*\(1 40\)\s*\);', 'p0              constant 10;', s)
assert s2 != s
open(p, 'w').write(s2)
EOF
ofrun() { ( cd "$1" && nice -n 10 pimpleFoam > log 2>&1 ); }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ); }
fields() { [ "$1" = T2 ] && echo "p U" || echo "p U k epsilon nut"; }

for a in T1 T2 T3; do
    cp -r "$W/$a" "$W/$a-of"
    cp -r "$W/$a" "$W/$a-br"
    ofrun "$W/$a-of" || { echo "FAIL: OpenFOAM $a"; tail -5 "$W/$a-of/log"; exit 1; }
    brrun "$W/$a-br" || { echo "FAIL: brae $a"; tail -5 "$W/$a-br/log"; exit 1; }
    # shellcheck disable=SC2046
    python3 "$CMP" "$W/$a-of" "$W/$a-br" $TIMES $(fields $a) > "$W/$a.cmp"
    sed 's/^/      /' "$W/$a.cmp"
    m=$(awk '/^MAX/{print $2}' "$W/$a.cmp")
    if [ "$BOUND" = report ]; then
        say "$a: worst entry over steps 1-3, rel $m (report only)" INFO
    else
        python3 -c "import sys; sys.exit(0 if float('$m') <= float('$BOUND') else 1)" \
            && say "$a: every entry of $(fields $a | tr ' ' ',') at steps 1-3 within $BOUND (worst $m)" PASS \
            || say "$a: every entry of $(fields $a | tr ' ' ',') at steps 1-3 within $BOUND (worst $m)" FAIL
    fi
done

# Each control must make ITS entry exceed the bound (the bound is the gate's; `report` uses 1e-6 here).
CB=$BOUND
[ "$CB" = report ] && CB=1e-6
control()
{
    local name=$1 arm=$2 t=$3 f=$4 entry=$5
    rm -rf "$W/$arm-$name"
    cp -r "$W/$arm" "$W/$arm-$name"
    brrun "$W/$arm-$name" "BRAE_CONTROL_$name=1" || { say "control $name: brae ran" FAIL; return; }
    local r
    r=$(python3 "$CMP" "$W/$arm-of" "$W/$arm-$name" "$t" "$f" | awk -v e="$entry" '$3 == e {print $10}')
    python3 -c "import sys; sys.exit(0 if '$r' and float('$r') > float('$CB') else 1)" \
        && say "control $name: $arm $f $entry at $t goes red (rel $r)" PASS \
        || say "control $name: $arm $f $entry at $t goes red (rel ${r:-none})" FAIL
}
control TOTAL_PRESSURE_AT_MOMENTUM T2 0.001 p inlet
control PIOV_ASSEMBLY_ONLY T2 0.001 U inlet
control TURB_INLET_REFVALUE T1 0.002 k inlet
control TURB_SWITCH_AT_MOMENTUM T1 0.001 k outlet1
control TURB_GRAD_AFTER_UPDATE T1 0.001 k internal

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
