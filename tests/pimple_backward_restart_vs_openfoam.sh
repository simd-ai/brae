#!/usr/bin/env bash
# pimpleFoam's restart under backward against OpenFOAM v2412: the old time levels written and read, and deltaT0.
#
# WHAT IT HOLDS. Under backward every transported field holds a second old level, so OpenFOAM writes its first one: U_0,
# phi_0 (asked for by fvc::ddtCorr), k_0 and epsilon_0 (GeometricField::storeOldTime hands the field's writeOpt to field0
# once field0 has an old level of its own, GeometricField.C:923-940). A restart reads each <field>_0 and makes its own old
# level a copy of it (readOldTimeIfPresent, :130-166), so the first restarted step -- time index 2 or more, where
# backwardDdtScheme takes its full form (deltaT0_(vf), backwardDdtScheme.C:62-71) -- runs on the file's levels; without
# a file the level is a copy of the field when first asked for (oldTime(), :945-975). Its deltaT0 is Time's deltaTSave_
# (Time.C:1059): controlDict's deltaT at a fixed deltaT (Time.C:196), whatever uniform/time's deltaT0 says. brae wrote
# none of the files and took Euler's form, without ddtCorr, on the first restarted step: 0.32 (U) at that step.
#
# THE CASE. incompressible/pimpleFoam/RAS/pitzDaily (kEpsilon) under backward at a fixed deltaT 1e-4, six steps, written
# every step at writePrecision 17 and timePrecision 18 (so a directory's name carries its time), every solve tight
# (tolerance 1e-12, relTol 0).
#
# ARMS, every entry of p U k epsilon nut on its own scale (tests/pimple_entry_compare.py) within BOUND unless stated
#   P  premises: OpenFOAM restarted from its 3e-4 equals its continuous run at 4e-4..6e-4; at 3e-4 it writes U_0 phi_0 k_0
#      epsilon_0
#   C  brae continuous against OpenFOAM continuous, every step
#   W  brae writes the same files as OpenFOAM at 3e-4, and its U_0 phi_0 k_0 epsilon_0 there equal OpenFOAM's
#   R  brae restarted from its own 3e-4 against OpenFOAM continuous, 4e-4..6e-4
#   O  brae from OpenFOAM's 3e-4 against OpenFOAM continuous
#   B  OpenFOAM from brae's 3e-4 against OpenFOAM continuous
#   N  both codes from OpenFOAM's 3e-4 with every _0 file removed (the levels then copies): brae against OpenFOAM, 4e-4..6e-4;
#      premise, OpenFOAM's N differs from its continuous run
#   H  both codes from OpenFOAM's 3e-4 at deltaT 5e-5 (deltaT0 at the first step is then 5e-5, not the 1e-4 uniform/time
#      holds): brae against OpenFOAM at 3.5e-4 and 4e-4
#   X  a start holding another field's _0 (a copy of k_0 named T_0): brae refuses it by name
# CONTROLS, each a run that completes and must go red on the check it guards:
#   BRAE_CONTROL_BACKWARD_NO_OLD_READ=1            O: no old levels read, the first restarted step bootstrapped
#   BRAE_CONTROL_BACKWARD_NO_OLD_WRITE=1           B: brae's write without the _0 files
#   BRAE_CONTROL_BACKWARD_RESTART_DELTAT0_STORED=1 H: the restart's deltaT0 from uniform/time
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28, job 5862559), worst entry:
#   before (PR #104's 79cb0ce5): no _0 file written; R 3.5e-01, O 3.5e-01, B 3.5e-01, N 2.4e-01, H 3.5e-01 (U or p internal at
#     the restart's second step), X ran; C 4.6e-10
#   after: W files equal, values 8.0e-12 (epsilon_0 upperWall); R 1.3e-10, O 9.4e-12, B 1.3e-10, N 1.3e-11, H 2.5e-11, C 4.6e-10
#     (p internal at 3e-4, the continuous run, whose own difference is the largest here). So BOUND 1e-8, twenty times it, the
#     margin the TJunction gates keep. Premises: OpenFOAM's restart equals its continuous run (0); N's start moves it 3.5e-01.
#     Controls 3.5e-01, 3.5e-01, 2.0e-01.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_BR_BOUND:-1e-8}
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
W=${BRAE_BR_KEEP:-$(mktemp -d)}
[ -n "${BRAE_BR_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-96s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

setkey()
{
    if grep -q -E "^$2[[:space:]]" "$1/system/controlDict"; then
        sed -i -E "s/^$2[[:space:]][^;]*;/$2 $3;/" "$1/system/controlDict"
    else
        printf '%-16s%s;\n' "$2" "$3" >> "$1/system/controlDict"
    fi
    grep -q -E "^$2[[:space:]]+$3;" "$1/system/controlDict" || { echo "FAIL: could not set $2 $3 in $1"; exit 1; }
}
ofrun() { ( cd "$1" && pimpleFoam > "log.$2" 2>&1 ) || { echo "FAIL: OpenFOAM in $1 ($2)"; tail -5 "$1/log.$2"; exit 1; }; }
brrun()
{
    local d=$1 tag=$2
    shift 2
    ( cd "$d" && env "$@" "$BIN" -case . > "log.$tag" 2>&1 ) || { echo "FAIL: brae in $d ($tag)"; tail -5 "$d/log.$tag"; exit 1; }
}
tlist() { (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -vx 0 | sort -g | awk -v a="$2" -v b="$3" '$1 + 0 > a + 0 && $1 + 0 < b + 0' \
          | paste -sd,); }
worst() { python3 "$CMP" "$@" | awk '/^MAX/{print $2, $3, $4, $5}'; }
le() { python3 -c "import sys; sys.exit(0 if float('$1') <= float('$2') else 1)"; }
gt() { python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }
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

# ---- staging: pitzDaily under backward at a fixed deltaT 1e-4, six steps, written every step, every solve tight ---------
mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
setkey "$W/base" adjustTimeStep no
setkey "$W/base" deltaT 0.0001
setkey "$W/base" endTime 0.0006
setkey "$W/base" writeControl timeStep
setkey "$W/base" writeInterval 1
setkey "$W/base" writePrecision 17
setkey "$W/base" timePrecision 18
python3 - "$W/base/system" <<'EOF'
import re
import sys
d = sys.argv[1]
s = open(d + '/fvSchemes').read()
s2 = re.sub(r'(ddtSchemes\s*\{\s*default\s+)Euler;', r'\1backward;', s)
assert s2 != s, 'pitzDaily no longer ships `ddtSchemes { default Euler; }`'
open(d + '/fvSchemes', 'w').write(s2)
s = open(d + '/fvSolution').read()
i = s.index('solvers')
j = s.index('PIMPLE')
blk = re.sub(r'(\btolerance\s+)[^;]+;', r'\g<1>1e-12;', s[i:j])
blk = re.sub(r'(\brelTol\s+)[^;]+;', r'\g<1>0;', blk)
blk = re.sub(r'(\bsolver\s+[^;]+;)', r'\g<1>\n        maxIter         5000;', blk)
open(d + '/fvSolution', 'w').write(s[:i] + blk + s[j:])
EOF
stage() { mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; }
# from <dir> <source case> <time>: the staging with <source>'s <time> beside its 0, startFrom latestTime
from() { stage "$1"; cp -r "$2/$3" "$1/"; setkey "$1" startFrom latestTime; }
echo "pimple_backward_restart_vs_openfoam: pitzDaily under backward, deltaT 1e-4, restarts from the third step"

# ---- continuous runs, and each code stopped at the third step ------------------------------------------------------------
stage "$W/of-c"; ofrun "$W/of-c" c
stage "$W/br-c"; brrun "$W/br-c" c
T3=$(tlist "$W/of-c" 0.00029 0.00031)
[ -n "$T3" ] || { echo "FAIL: OpenFOAM wrote no third step"; exit 1; }
AFTER=$(tlist "$W/of-c" 0.00031 1)
ALL=$(tlist "$W/of-c" 0 1)
OLDS="U_0 phi_0 k_0 epsilon_0"

# P: OpenFOAM's own restart is its continuous run, and it writes the four levels
from "$W/of-r" "$W/of-c" "$T3"; ofrun "$W/of-r" r
w=$(worst "$W/of-c" "$W/of-r" "$AFTER" $FIELDS)
le "${w%% *}" 0 && say "premise: OpenFOAM restarted from its $T3 equals its continuous run (worst $w)" PASS \
                || say "premise: OpenFOAM restarted from its $T3 equals its continuous run (worst $w)" FAIL
ofw=$(ls "$W/of-c/$T3" | sort | tr '\n' ' ')
missing=""
for f in $OLDS; do
    [ -f "$W/of-c/$T3/$f" ] || missing="$missing $f"
done
[ -z "$missing" ] && say "premise: OpenFOAM writes $OLDS at $T3" PASS \
                  || say "premise: OpenFOAM writes $OLDS at $T3 (missing:$missing; it wrote: $ofw)" FAIL

# C and W
judge "C: brae continuous against OpenFOAM continuous, every step" "$W/of-c" "$W/br-c" "$ALL"
brw=$(ls "$W/br-c/$T3" | sort | tr '\n' ' ')
[ "$brw" = "$ofw" ] && say "W: brae writes the files OpenFOAM writes at $T3 ($brw)" PASS \
                    || say "W: brae writes the files OpenFOAM writes at $T3 (brae: $brw; OpenFOAM: $ofw)" FAIL
judge "W: brae's $OLDS at $T3 against OpenFOAM's" "$W/of-c" "$W/br-c" "$T3" "$OLDS"

# R, O, B
from "$W/br-r" "$W/br-c" "$T3"; brrun "$W/br-r" r
judge "R: brae restarted from its own $T3 against OpenFOAM continuous" "$W/of-c" "$W/br-r" "$AFTER"
from "$W/br-o" "$W/of-c" "$T3"; brrun "$W/br-o" o
judge "O: brae from OpenFOAM's $T3 against OpenFOAM continuous" "$W/of-c" "$W/br-o" "$AFTER"
from "$W/of-b" "$W/br-c" "$T3"; ofrun "$W/of-b" b
judge "B: OpenFOAM from brae's $T3 against OpenFOAM continuous" "$W/of-c" "$W/of-b" "$AFTER"

# N: no _0 files in the start
from "$W/of-n" "$W/of-c" "$T3"; ( cd "$W/of-n/$T3" && rm -f $OLDS ); ofrun "$W/of-n" n
from "$W/br-n" "$W/of-c" "$T3"; ( cd "$W/br-n/$T3" && rm -f $OLDS ); brrun "$W/br-n" n
w=$(worst "$W/of-c" "$W/of-n" "$AFTER" $FIELDS)
gt "${w%% *}" "$PREMISE" && say "premise: OpenFOAM from its $T3 without the _0 files differs from its continuous run ($w)" PASS \
                         || say "premise: OpenFOAM from its $T3 without the _0 files differs from its continuous run ($w)" FAIL
judge "N: brae against OpenFOAM, both from $T3 without the _0 files" "$W/of-n" "$W/br-n" "$AFTER"

# H: deltaT halved at the restart
for c in of br; do
    from "$W/$c-h" "$W/of-c" "$T3"
    setkey "$W/$c-h" deltaT 0.00005
    setkey "$W/$c-h" endTime 0.0004
done
ofrun "$W/of-h" h
brrun "$W/br-h" h
TH=$(tlist "$W/of-h" 0.00031 1)
[ -n "$TH" ] || { echo "FAIL: OpenFOAM wrote nothing after $T3 at deltaT 5e-5"; exit 1; }
judge "H: brae against OpenFOAM from $T3 at deltaT 5e-5" "$W/of-h" "$W/br-h" "$TH"

# X: a start holding a _0 brae does not carry
from "$W/br-x" "$W/of-c" "$T3"
sed 's/object\([[:space:]]*\)k_0;/object\1T_0;/' "$W/br-x/$T3/k_0" > "$W/br-x/$T3/T_0"
if ( cd "$W/br-x" && "$BIN" -case . > log.x 2>&1 ); then
    say "X: brae refuses a start holding T_0 by name (it ran)" FAIL
else
    grep -q "holds T_0" "$W/br-x/log.x" && say "X: brae refuses a start holding T_0 by name" PASS \
                                         || say "X: brae refuses a start holding T_0 by name ($(grep -a -m1 -i error "$W/br-x/log.x" | cut -c1-60))" FAIL
fi

# ---- controls -------------------------------------------------------------------------------------------------------------
CB=$BOUND
[ "$CB" = report ] && CB=1e-8
control()   # <env> <label> <reference case> <case> <times>
{
    local w
    w=$(worst "$3" "$4" "$5" $FIELDS)
    gt "${w%% *}" "$CB" && say "control $1: $2 goes red ($w)" PASS || say "control $1: $2 goes red (${w:-none})" FAIL
}
from "$W/ctl-o" "$W/of-c" "$T3"; brrun "$W/ctl-o" o BRAE_CONTROL_BACKWARD_NO_OLD_READ=1
control BRAE_CONTROL_BACKWARD_NO_OLD_READ=1 O "$W/of-c" "$W/ctl-o" "$AFTER"
stage "$W/ctl-w"; setkey "$W/ctl-w" endTime 0.0003; brrun "$W/ctl-w" c BRAE_CONTROL_BACKWARD_NO_OLD_WRITE=1
from "$W/ctl-b" "$W/ctl-w" "$T3"; ofrun "$W/ctl-b" b
control BRAE_CONTROL_BACKWARD_NO_OLD_WRITE=1 B "$W/of-c" "$W/ctl-b" "$AFTER"
from "$W/ctl-h" "$W/of-c" "$T3"; setkey "$W/ctl-h" deltaT 0.00005; setkey "$W/ctl-h" endTime 0.0004
brrun "$W/ctl-h" h BRAE_CONTROL_BACKWARD_RESTART_DELTAT0_STORED=1
control BRAE_CONTROL_BACKWARD_RESTART_DELTAT0_STORED=1 H "$W/of-h" "$W/ctl-h" "$TH"

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
