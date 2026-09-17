#!/usr/bin/env bash
# brae's interFoam TURBULENCE against REAL OpenFOAM's, on RAS/damBreak, field by field and solve by solve.
#
# THE METHOD is tests/interfoam_dambreak_vs_openfoam.sh's: exactly N identical FIXED steps, both codes
# read at one instant, because an adaptive step turns any difference into a clock.
#
# THREE PROFILES, because interFoam's turbulence is two models behind one keyword and the shipped case
# exercises neither every setting nor both models.
#
#   variable   RAS/damBreak as shipped: `density variable`. kEpsilon weighted by the mixture rho,
#              convecting with rhoPhi under the key div(rhoPhi,k), divU from the volumetric phi -- and
#              NO validate(), so the first UEqn runs on the case file's nut = 0.
#   uniform    the same case without that line: incompressible::turbulenceModel::New(U, phi, mixture),
#              the ordinary single-phase model, which DOES validate() at construction. OpenFOAM then
#              looks up div(phi,k), which the tutorial does not carry, so the staging renames the two
#              entries; 15 of the 17 turbulent interFoam tutorials are this lineage.
#   custom     uniform, plus every setting the shipped case leaves at a value that cannot be seen:
#              its own kEpsilonCoeffs, equation relaxation 0.7, and a solver tolerance of 0.5 so that
#              `minIter 1` is what forces each sweep. MEASURED on the shipped case with each ignored
#              in turn: minIter and the `".*" 1` relaxation change NOTHING there, to the last digit.
#              On this fixture ignoring minIter is 14% of U and 0 of 5 epsilon iteration counts,
#              ignoring the relaxation 2.2%, the coefficients 3.5% (sigmak/sigmaEps alone 0.3%; C3
#              alone 1.2e-08 of epsilon, because divU is nearly zero in an incompressible flow).
#
# WHAT THE SHIPPED CASE DOES SEE, measured the same way on `variable`: validate() called in the wrong
# lineage 29% of U; rho.oldTime() taken as the current rho 5.6%; divU from the mass flux 53%; the
# inletOutlet patches handed rhoPhi where they look up phi 4.9e-04; PBiCGStab for the case's
# symGaussSeidel 1 of 5 iteration counts and 2.7e-06. brae against OpenFOAM is 1.3e-11.
#
# THE CONTROLS are OpenFOAM's own answers at the same instant: the case run laminar (133% of U away),
# and the run this profile differs from in its one setting.
#
# THE DEVICE LOOP RUNS EVERY PROFILE TWICE, because the closure went onto the device one module at a
# time. With the device closure -- what `brae_interFoam -device` runs -- it is held to OpenFOAM at the
# host's bounds. With the HOST closure in its place (BRAE_INTER_HOST_CLOSURE=1, which the driver
# announces) it is the oracle for the first: against OpenFOAM a disagreement could be the loop's or the
# closure's, between those two only the closure's. MEASURED: device closure against host closure in the
# same loop, U 3.3e-14, k 1.1e-15, epsilon 1.6e-15, nut 1.9e-15, the same sweep counts solve for solve.
# The `custom` profile is where the device closure's one defect showed: it left the wall laplacian
# coefficient out of relax(), as the host reference had, and every epsilon residual was 1.1e-04 from
# OpenFOAM's with the fields unmoved. 5.3e-14 with it.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_ras_dambreak_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/damBreak/damBreak"
STEPS=${STEPS:-5}
DT=${DT:-1e-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the tutorial, apply the profile, fix the step, run real OpenFOAM
stage()
{
    local profile="$1"
    local C="$W/$profile"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "^density  *variable;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the tutorial no longer ships \`density variable\`, so the profiles mean something else"; return 1; }
    case "$profile" in
        laminar)
            sed -i 's/^simulationType .*/simulationType laminar;/' "$C/constant/turbulenceProperties" ;;
        uniform|custom)
            sed -i '/^density /d' "$C/constant/turbulenceProperties"
            sed -i 's/^\( *\)div(rhoPhi,k) .*/\1div(phi,k)      Gauss upwind;/; s/^\( *\)div(rhoPhi,epsilon) .*/\1div(phi,epsilon) Gauss upwind;/' \
                "$C/system/fvSchemes"
            grep -q "div(phi,epsilon)" "$C/system/fvSchemes" || { echo "FAIL: the div entries were not renamed"; return 1; } ;;
    esac
    if [ "$profile" = custom ]; then
        python3 - "$C" <<'PYEOF' || { echo "FAIL: the custom profile was not staged"; return 1; }
import os, re, sys
d = sys.argv[1]
q = os.path.join(d, 'constant/turbulenceProperties')
t = open(q).read()
t = t.replace('    printCoeffs     on;',
              '    printCoeffs     on;\n    kEpsilonCoeffs { Cmu 0.12; C1 1.5; C2 1.8; C3 0.2; sigmak 1.1; sigmaEps 1.2; }', 1)
assert 'kEpsilonCoeffs' in t
open(q, 'w').write(t)
q = os.path.join(d, 'system/fvSolution')
t = open(q).read()
m = re.search(r'("\(U\|k\|epsilon\)\.\*"\s*\{)([^}]*)\}', t)
assert m, 'no (U|k|epsilon).* entry'
body = re.sub(r'tolerance\s+[^;]+;', 'tolerance       0.5;', m.group(2))
assert 'minIter' in body
t = t[:m.start(2)] + body + t[m.end(2):]
t, n = re.subn(r'"\.\*"\s+1;', '"(k|epsilon).*" 0.7;', t)
assert n == 1
open(q, 'w').write(t)
PYEOF
    fi

    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'^adjustTimeStep .*', 'adjustTimeStep  no;',        s, flags=re.M)
s = re.sub(r'^deltaT .*',         'deltaT          %s;' % dt,   s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         %.10g;' % (n*float(dt)), s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;',  s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   %d;' % n,    s, flags=re.M)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',     s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',        s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"

    if [ "$profile" = custom ]; then
        # THE ORACLE HAS TO HAVE TAKEN A SWEEP IT DID NOT NEED, or minIter is not what this gates: a
        # solve whose initial residual is already under the 0.5 tolerance and still reads one iteration
        python3 - "$C/log.interFoam" <<'PYEOF' || { echo "FAIL: minIter never forced a sweep in OpenFOAM's custom run"; return 1; }
import re, sys
forced = 0
for ln in open(sys.argv[1]):
    m = re.search(r'Solving for (k|epsilon), Initial residual = (\S+), Final residual = \S+, No Iterations (\d+)', ln)
    if m and float(m.group(2)) < 0.5 and int(m.group(3)) == 1:
        forced += 1
print('OpenFOAM took %d sweeps that only minIter asked for' % forced)
sys.exit(0 if forced > 0 else 1)
PYEOF
    fi
}

rc=0
for p in laminar variable uniform custom; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_ras_dambreak_vs_openfoam: staging failed"; exit 1; }

# <profile> <the lineage brae must report> <OpenFOAM's run without this profile's one setting>
"$BIN" "$W/variable" "$W/variable/0" "$W/variable/$END" "$STEPS" "$W/variable/log.interFoam" \
       variable variable "$W/laminar/$END" "$W/uniform/$END" || rc=1
"$BIN" "$W/uniform" "$W/uniform/0" "$W/uniform/$END" "$STEPS" "$W/uniform/log.interFoam" \
       uniform uniform "$W/laminar/$END" "$W/variable/$END" || rc=1
"$BIN" "$W/custom" "$W/custom/0" "$W/custom/$END" "$STEPS" "$W/custom/log.interFoam" \
       custom uniform "$W/laminar/$END" "$W/uniform/$END" || rc=1

echo "interfoam_ras_dambreak_vs_openfoam: rc $rc"
exit $rc
