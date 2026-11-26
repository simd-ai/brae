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
#   R1  T2 on the mesh rotated 30 deg about z (OpenFOAM's transformPoints after blockMesh): the inlet normal is not an axis
#   R2  T1 on the mesh rotated 20 deg about (1 1 1): every patch normal mixes three components
#       The flow starts at rest, there is no gravity and every boundary value is rotation-invariant, so R1 and R2 are T2 and
#       T1 in another frame. The tutorial's probes are left out of R1 and R2: the rotated mesh no longer contains their
#       points, and OpenFOAM's probes then write an entry it cannot read back and stop.
#   F1  T2 with TJunctionFan's fanPressure inlet (its fan curve FluxVsdP.dat, p0 30) and div(phi,U) Gauss linear. The inlet's
#       p0 follows the patch flux at every pressure assembly through the fan curve. Under the tutorial's limitedLinearV the
#       fan case amplifies any difference: OpenFOAM against itself with PCG instead of GAMG on p is 1.7e-05 at step 2 and
#       5.8e-05 at step 3, and against its own one-ulp p0 twin 9.3e-06 at step 3 (Arrhenius 5796712); with Gauss linear
#       brae is 1.2e-10 from OpenFOAM at every step, so that is the staging judged here.
#
# CHECKS, each entry (the internal field and every patch's value) of p U k epsilon nut at 0.001, 0.002, 0.003 on its
# own scale (tests/pimple_entry_compare.py): rel <= BOUND. Controls, each must go red on the entry it guards:
#   BRAE_CONTROL_TOTAL_PRESSURE_AT_MOMENTUM=1  T2, inlet p at 0.001
#   BRAE_CONTROL_PIOV_ASSEMBLY_ONLY=1          T2, inlet U at 0.001
#   BRAE_CONTROL_TURB_INLET_REFVALUE=1         T1, inlet k at 0.002
#   BRAE_CONTROL_TURB_SWITCH_AT_MOMENTUM=1     T1, outlet1 k at 0.001
#   BRAE_CONTROL_TURB_GRAD_AFTER_UPDATE=1      T1, internal k at 0.001
#   BRAE_CONTROL_PIOV_HBYA_DESCRIPTOR=1        R1, inlet U at 0.001 (HbyA_b through U's descriptor, not the cell value)
#   BRAE_CONTROL_PIOV_NO_SOLVE_EVAL=1          R2, inlet nut at 0.002 (U_b not re-evaluated after the momentum solve)
#   BRAE_CONTROL_TOTAL_PRESSURE_AT_MOMENTUM=1  F1 too, inlet p at 0.001 (the fan's p0 from the flux before the step)
# MEASURED (JUPITER, one GH200, OpenFOAM v2412 serial, 2026-11-25): worst entry over steps 1-3
# T1 4.9e-08, T2 5.6e-08, T3 1.8e-08 (U internal at 0.003, in the outlet2 branch; p within 6.6e-10), so BOUND 1e-6 is a
# decade above. Before these changes: T2 inlet p 1.7e-05 and U 2.4e-04 at step 1, T1 inlet k 1.0 at step 2. Controls:
# 1.7e-05, 6.7e-05, 1.0, 2.5e-01, 1.3e-05, each on its own entry. 31 s.
# R1, R2 (Arrhenius, one GH200, 2026-11-26): 4.1e-09 and 1.7e-08; before HbyA_b took the cell value and U_b was re-evaluated
# after the momentum solve, U inlet 3.1e-01 at step 1 on R1 and nut inlet 5.9e-01 on R2. Controls: 3.1e-01 and 3.1e-06.
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
rotate()   # <arm> <axis-angle>: the staged mesh rotated, the probes left out
{
    ( cd "$1" && transformPoints -rotate-angle "$2" > log.transformPoints 2>&1 ) || { echo "FAIL: transformPoints $1"; exit 1; }
    python3 - "$1/system/controlDict" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
i = s.index('\nfunctions')
j = s.index('{', i)
d = 0
for k in range(j, len(s)):
    d += {'{': 1, '}': -1}.get(s[k], 0)
    if d == 0:
        break
open(p, 'w').write(s[:i] + s[k + 1:])
EOF
}
stage "$W/R1"
cp "$W/T2/constant/turbulenceProperties" "$W/R1/constant/turbulenceProperties"
rotate "$W/R1" "((0 0 1) 30)"
stage "$W/R2"
rotate "$W/R2" "((1 1 1) 20)"
stage "$W/F1"
cp "$W/T2/constant/turbulenceProperties" "$W/F1/constant/turbulenceProperties"
cp "${TUT%/TJunction}/TJunctionFan/constant/FluxVsdP.dat" "$W/F1/constant/" || { echo "FAIL: no TJunctionFan fan curve"; exit 1; }
python3 - "$W/F1" <<'EOF'
import re, sys
c = sys.argv[1]
s = open(c + '/0/p').read()
inlet = """inlet
    {
        type            fanPressure;
        direction       in;
        fanCurve
        {
            type table;
            file "<constant>/FluxVsdP.dat";
        }
        p0              uniform 30;
    }"""
s, n = re.subn(r'inlet\s*\{.*?\n    \}', lambda m: inlet, s, count=1, flags=re.S)
assert n == 1
open(c + '/0/p', 'w').write(s)
f = open(c + '/system/fvSchemes').read()
f2, n = re.subn(r'(\n\s*div\(phi,U\)\s+)[^;]+;', r'\1Gauss linear;', f)
assert n == 1
open(c + '/system/fvSchemes', 'w').write(f2)
EOF
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
fields() { case "$1" in T2|R1|F1) echo "p U" ;; *) echo "p U k epsilon nut" ;; esac; }

for a in T1 T2 T3 R1 R2 F1; do
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
control PIOV_HBYA_DESCRIPTOR R1 0.001 U inlet
control PIOV_NO_SOLVE_EVAL R2 0.002 nut inlet
control TOTAL_PRESSURE_AT_MOMENTUM F1 0.001 p inlet

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
