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
# THE `outer` PROFILE (nOuterCorrectors 2, 2026-09-22) is where the device ALPHA STEP's defect showed: it
# reset alpha1 to its old time at the start of every outer pass, which alphaEqn.H never does (the host
# reference had stopped doing it and measured why); the second pass's first p_rgh residual sat 1.2e-06
# from OpenFOAM's, k 6.7e-06 and U 2.0e-06 after two steps, with every first-pass number at the floor.
# It was found under CrankNicolson (tests/interfoam_cn_vs_openfoam.sh), where the same reset reads the
# same numbers. The device arm's alpha-solve bound is wider on this profile alone: the second pass's
# pre-solve starts from the first pass's alpha, its initial residual is 1e-5 of the normFactor, and
# the device's 1e-12 in phi (its VoF floor) reads as 1e-07 relative there.
#
# The `custom` profile is where the device closure's one defect showed: it left the wall laplacian
# coefficient out of relax(), as the host reference had, and every epsilon residual was 1.1e-04 from
# OpenFOAM's with the fields unmoved. 5.3e-14 with it.
#
# PROFILE nutAtmosphere: the uniform lineage with the atmosphere's nut an inletOutlet (inletValue 1e-3),
# which nut.correctBoundaryConditions() evaluates after kEpsilon's field assignment: inflow faces -- 20
# of the 46 at t = 0.005 -- take the inletValue, outflow faces the new cell nut. MEASURED: U 6.0e-14,
# nut 2.7e-15. CONTROL: the atmosphere `calculated`, OpenFOAM's U 2.7e-03 away. BROKEN ONCE EACH: the
# patch written by the assignment, Cmu*k_b^2/epsilon_b (as brae had it), U 2.7e-03; evaluated without
# the flux's valueFraction, U 4.4e-04. The device closure refuses the profile by name.
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

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

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
        uniform|custom|nutAtmosphere|sst|frozen|frozenFloored|frozenSST|splitSolve|splitSolveSST|splitDiv|splitDivSST|lowRe|lowReOff)
            sed -i '/^density /d' "$C/constant/turbulenceProperties"
            sed -i 's/^\( *\)div(rhoPhi,k) .*/\1div(phi,k)      Gauss upwind;/; s/^\( *\)div(rhoPhi,epsilon) .*/\1div(phi,epsilon) Gauss upwind;/' \
                "$C/system/fvSchemes"
            grep -q "div(phi,epsilon)" "$C/system/fvSchemes" || { echo "FAIL: the div entries were not renamed"; return 1; } ;;
    esac
    # nutAtmosphere: the atmosphere's nut an inletOutlet, which correctBoundaryConditions evaluates after
    # kEpsilon's field assignment -- inflow faces take the inletValue, outflow faces the new cell nut
    if [ "$profile" = nutAtmosphere ]; then
        python3 - "$C" <<'PYEOF' || { echo "FAIL: the nutAtmosphere profile was not staged"; return 1; }
import re, sys
p = sys.argv[1] + '/0/nut'
s = open(p).read()
s, n = re.subn(r'atmosphere\s*\{[^}]*\}', 'atmosphere\n    {\n        type            inletOutlet;\n'
               '        inletValue      uniform 0.001;\n        value           uniform 0;\n    }', s)
assert n == 1
open(p, 'w').write(s)
PYEOF
    fi
    # `sst`: the same tutorial made kOmegaSST -- omega from its epsilon file with omegaWallFunction, the
    # closure's own div entries, the solver entry renamed, and fvSchemes' mandatory wallDist method.
    # RAS/damBreak's nut atmosphere is `calculated`, which is what lets the DEVICE closure run it.
    if [ "$profile" = sst ] || [ "$profile" = frozenSST ] || [ "$profile" = splitSolveSST ] \
       || [ "$profile" = splitDivSST ]; then
        python3 - "$C" <<'SSTEOF' || { echo "FAIL: the sst profile was not staged"; return 1; }
import os, re, sys
d = sys.argv[1]
q = os.path.join(d, 'constant/turbulenceProperties')
t = open(q).read()
t, n = re.subn(r'RASModel\s+\w+;', 'RASModel        kOmegaSST;', t)
assert n == 1, 'no RASModel entry'
open(q, 'w').write(t)
q = os.path.join(d, 'system/fvSchemes')
t = open(q).read()
t, n = re.subn(r'div\(phi,epsilon\)\s+Gauss upwind;', 'div(phi,omega)  Gauss upwind;', t)
assert n == 1, 'the epsilon div entry was not renamed'
t = t.rstrip() + '\n\nwallDist\n{\n    method meshWave;\n}\n'
open(q, 'w').write(t)
q = os.path.join(d, 'system/fvSolution')
t = open(q).read()
t, n = re.subn(r'\(U\|k\|epsilon\)', '(U|k|omega)', t)
assert n >= 1, 'no (U|k|epsilon) solver entry'
open(q, 'w').write(t)
e = open(os.path.join(d, '0/epsilon')).read()
e = e.replace('epsilonWallFunction', 'omegaWallFunction')
e = re.sub(r'object\s+epsilon;', 'object      omega;', e)
e = e.replace('[0 2 -3 0 0 0 0]', '[0 0 -1 0 0 0 0]')
open(os.path.join(d, '0/omega'), 'w').write(e)
SSTEOF
    fi
    # `frozen*`: `RAS { turbulence off; }` -- a model that is CONSTRUCTED and VALIDATED and then never
    # corrected again. It is neither laminar nor "keep the file's nut", and THIS FIXTURE SEPARATES THE
    # THREE READINGS because the case ships nut uniform 0 with k and epsilon both 0.1:
    #   laminar                nut stays 0            (no model at all)
    #   "keep the file's nut"  nut stays 0            (the reading brae's old refusal called possible)
    #   OpenFOAM               nut = Cmu*k^2/epsilon  = 0.09*0.01/0.1 = 0.009
    # eddyViscosity.C:119-122 is `correctNut();` with no turbulence_ test, and the uniform lineage calls
    # validate() (incompressibleInterPhaseTransportModel.C:105), so OpenFOAM takes the third. 0.009
    # against a water nu of 1e-6 is a 9000x eddy viscosity, which is why the laminar control is decisive
    # here rather than marginal.
    case "$profile" in
        frozen|frozenFloored|frozenSST)
            sed -i 's/^\( *\)turbulence  *on;/\1turbulence      off;/' "$C/constant/turbulenceProperties"
            grep -q "turbulence      off;" "$C/constant/turbulenceProperties" \
                || { echo "FAIL: the tutorial no longer ships 'turbulence on', so [$profile] staged nothing"; return 1; } ;;
    esac
    # `frozenFloored`: floors ABOVE the case's own k and epsilon, both 0.1. With correct() gated out, the
    # CONSTRUCTOR's bound (kEpsilon.C:182-183) is then the ONLY thing in the entire run that touches
    # either field -- k and epsilon come out at 0.5 and nut at 0.09*0.25/0.5 = 0.045, not 0.009. brae's
    # interFoam reader had that bound in its LES branch alone, so this arm is the one that witnesses it.
    if [ "$profile" = frozenFloored ]; then
        sed -i 's/^\( *\)turbulence      off;/\1turbulence      off;\n\1kMin            0.5;\n\1epsilonMin      0.5;/' \
            "$C/constant/turbulenceProperties"
        grep -q "kMin            0.5;" "$C/constant/turbulenceProperties" \
            || { echo "FAIL: the floors were not staged"; return 1; }
    fi
    # `splitSolve*`: the SECOND equation gets its OWN solver entry. `fvMatrix::solve()` looks the solver
    # dictionary up BY FIELD NAME, so `kFinal` and `epsilonFinal` (or `omegaFinal`) need not agree and
    # OpenFOAM honours each; brae's closures took k's for both and REFUSED a mismatch. Every shipped
    # tutorial writes the pair as one regex key -- `"(U|k|epsilon).*"` here -- which is why no tutorial
    # reaches it and the profile has to be staged.
    #
    # THE SPLIT IS A LITERAL KEY BESIDE THE REGEX. OpenFOAM's dictionary searches hashedEntries_ before
    # patterns, so an explicit `epsilonFinal` wins over the regex that also matches it; brae's reader does
    # the same (foam_dict.cuh:74-90). k keeps the case's own entry, which is what makes this a SPLIT
    # rather than two new settings.
    #
    # WHAT MAKES IT VISIBLE: tolerance 1e-12 against the case's 1e-06, and TWO smoother sweeps against
    # one. Both change the second field's iteration count and final residual in OpenFOAM's own log, which
    # this gate already compares solve for solve -- so a brae that used k's settings for epsilon fails on
    # the counts, not merely on the fields.
    if [ "$profile" = splitSolve ] || [ "$profile" = splitSolveSST ]; then
        SEC=epsilonFinal
        [ "$profile" = splitSolveSST ] && SEC=omegaFinal
        SEC="$SEC" python3 - "$C" <<'PYEOF' || { echo "FAIL: the splitSolve profile was not staged"; return 1; }
import os, re, sys
d = sys.argv[1]
sec = os.environ["SEC"]
q = d + "/system/fvSolution"
t = open(q).read()
# the regex entry must still be there, or k is not keeping the case's own setting
key = '"(U|k|omega)' if sec == "omegaFinal" else '"(U|k|epsilon)'
assert key in t, "the regex solver entry is not where this profile expects it"
block = ("\n    %s\n    {\n        solver          smoothSolver;\n"
         "        smoother        symGaussSeidel;\n        tolerance       1e-12;\n"
         "        relTol          0;\n        minIter         1;\n        nSweeps         2;\n    }\n" % sec)
t, n = re.subn(r"\n\}\s*\n\s*PIMPLE", block + "}\n\nPIMPLE", t, count=1)
assert n == 1, "could not place the entry inside the solvers dictionary"
open(q, "w").write(t)
PYEOF
        grep -q "$SEC" "$C/system/fvSolution" || { echo "FAIL: $SEC was not written"; return 1; }
    fi
    # `splitDiv*`: the two closure equations get DIFFERENT convection schemes. `fvm::div(phi, psi)`
    # resolves `div(phi,<psi>)` by the FIELD's name, so `div(phi,k) Gauss upwind` beside
    # `div(phi,epsilon) Gauss limitedLinear 1` is two different MATRICES in OpenFOAM -- not a looser
    # tolerance. brae's closures carried one scheme for the pair and the reader refused a mismatch.
    #
    # k KEEPS the case's own `Gauss upwind` and only the second field changes, so the arm measures one
    # difference. limitedLinear against upwind on the second equation is a large move, which is why this
    # one shows in the FIELDS as well as in the iteration counts -- unlike `splitSolve`, where a
    # tolerance only moved the stopping point.
    if [ "$profile" = splitDiv ] || [ "$profile" = splitDivSST ]; then
        SECF=epsilon
        [ "$profile" = splitDivSST ] && SECF=omega
        SECF="$SECF" python3 - "$C" <<'DIVEOF' || { echo "FAIL: the splitDiv profile was not staged"; return 1; }
import os, re, sys
d = sys.argv[1]
sec = os.environ["SECF"]
q = d + "/system/fvSchemes"
t = open(q).read()
pat = r"div\(phi,%s\)\s+Gauss upwind;" % sec
assert re.search(pat, t), "div(phi,%s) Gauss upwind is not where this profile expects it" % sec
t = re.sub(pat, "div(phi,%s) Gauss limitedLinear 1;" % sec, t, count=1)
# k MUST still be upwind, or the profile is changing both and measuring nothing
assert re.search(r"div\(phi,k\)\s+Gauss upwind;", t), "div(phi,k) is no longer upwind"
open(q, "w").write(t)
DIVEOF
        grep -q "div(phi,$SECF) Gauss limitedLinear 1;" "$C/system/fvSchemes" \
            || { echo "FAIL: the second field's div entry was not changed"; return 1; }
    fi
    # `lowRe` / `lowReOff`: a PAIR, identical but for `lowReCorrection` on the epsilon wall patches. On a
    # face with y+ < yPlusLam epsilonWallFunction switches epsilon to the RESOLVED form 2*k*nu/y^2 and
    # contributes NO wall production (epsilonWallFunctionFvPatchScalarField.C:242, :338) -- a different
    # BRANCH, not a scaling. brae parsed the entry and NOTHING under interFoam read it, so the branch
    # could never fire.
    #
    # BOTH RAISE THE WATER VISCOSITY TO 1e-2, and that is what makes the arm possible: at the tutorial`s
    # own nu = 1e-6 no face on this mesh has y+ under yPlusLam, so the switch changes NOTHING -- measured,
    # OpenFOAM against itself, 0.0000e+00 on every field. At 1e-2 the wall is resolved and it moves
    # OpenFOAM`s own epsilon on all 2268 cells, worst relative 7.8e+00. The twin is the control, because
    # the raised nu means no other profile is comparable.
    if [ "$profile" = lowRe ] || [ "$profile" = lowReOff ]; then
        LOWRE_ON=0
        [ "$profile" = lowRe ] && LOWRE_ON=1
        LOWRE_ON="$LOWRE_ON" python3 - "$C" <<'LOWEOF' || { echo "FAIL: the $profile profile was not staged"; return 1; }
import os, re, sys
d = sys.argv[1]
q = d + "/constant/transportProperties"
t = open(q).read()
t, n = re.subn(r"(water\s*\{[^}]*?nu\s+)\[?[^;\]]*\]?;", r"\g<1>1e-2;", t, flags=re.S)
if n == 0:
    t, n = re.subn(r"nu\s+1e-06;", "nu              1e-2;", t, count=1)
assert n >= 1, "could not raise the water viscosity"
open(q, "w").write(t)
if os.environ["LOWRE_ON"] == "1":
    q = d + "/0/epsilon"
    t = open(q).read()
    t, n = re.subn(r"(type\s+epsilonWallFunction;\n)", r"\1        lowReCorrection true;\n", t)
    assert n >= 1, "no epsilonWallFunction entry in 0/epsilon"
    open(q, "w").write(t)
    print("  epsilon: lowReCorrection set on %d wall patch(es), water nu raised to 1e-2" % n)
else:
    print("  water nu raised to 1e-2, lowReCorrection left off (the control)")
LOWEOF
        grep -q "1e-2;" "$C/constant/transportProperties" \
            || { echo "FAIL: the viscosity was not raised"; return 1; }
    fi
    # `outer`: the shipped case with nOuterCorrectors 2 -- the second pass starts from the first pass's
    # alpha, U and phi, and every once-per-step update must stay once per step. The DEVICE alpha step
    # reset alpha1 to its old time at the start of every pass until this profile measured it: the
    # second pass's first p_rgh residual 1.2e-06 from OpenFOAM's, k 6.7e-06 and U 2.0e-06 after two
    # steps, while every first-pass number matched (2026-09-22, found under CrankNicolson, where the
    # same reset read the same numbers).
    if [ "$profile" = outer ]; then
        sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 2;/' "$C/system/fvSolution"
        grep -q "nOuterCorrectors 2;" "$C/system/fvSolution" || { echo "FAIL: nOuterCorrectors 1 was not found to replace"; return 1; }
    fi
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
for p in laminar variable uniform custom nutAtmosphere sst outer frozen frozenFloored frozenSST splitSolve splitSolveSST splitDiv splitDivSST lowReOff lowRe; do
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
"$BIN" "$W/nutAtmosphere" "$W/nutAtmosphere/0" "$W/nutAtmosphere/$END" "$STEPS" "$W/nutAtmosphere/log.interFoam" \
       nutAtmosphere uniform "$W/laminar/$END" "$W/uniform/$END" || rc=1
"$BIN" "$W/sst" "$W/sst/0" "$W/sst/$END" "$STEPS" "$W/sst/log.interFoam" \
       sst uniform "$W/laminar/$END" "$W/uniform/$END" || rc=1
"$BIN" "$W/outer" "$W/outer/0" "$W/outer/$END" "$STEPS" "$W/outer/log.interFoam" \
       outer variable "$W/laminar/$END" "$W/variable/$END" || rc=1

# THE FROZEN ARMS. The control is the SAME case with turbulence on -- `uniform` -- so the one setting
# between them is the switch, and `laminar` is the second control: it is the answer a brae that read
# `turbulence off` as "no model" or as "keep the file's nut" would produce, since both leave nut at 0.
"$BIN" "$W/frozen" "$W/frozen/0" "$W/frozen/$END" "$STEPS" "$W/frozen/log.interFoam" \
       frozen uniform "$W/laminar/$END" "$W/uniform/$END" || rc=1
# ...and this one's control is the UNFLOORED frozen run: the floors are the only difference, so any gap
# between the two is the constructor's bound and nothing else.
"$BIN" "$W/frozenFloored" "$W/frozenFloored/0" "$W/frozenFloored/$END" "$STEPS" "$W/frozenFloored/log.interFoam" \
       frozenFloored uniform "$W/laminar/$END" "$W/frozen/$END" || rc=1
"$BIN" "$W/frozenSST" "$W/frozenSST/0" "$W/frozenSST/$END" "$STEPS" "$W/frozenSST/log.interFoam" \
       frozenSST uniform "$W/laminar/$END" "$W/sst/$END" || rc=1

# THE SPLIT-SOLVER ARMS. The control is the same case with ONE entry for both equations -- `uniform` for
# kEpsilon, `sst` for kOmegaSST -- so the only difference is the second field's solver setting.
"$BIN" "$W/splitSolve" "$W/splitSolve/0" "$W/splitSolve/$END" "$STEPS" "$W/splitSolve/log.interFoam" \
       splitSolve uniform "$W/laminar/$END" "$W/uniform/$END" || rc=1
"$BIN" "$W/splitSolveSST" "$W/splitSolveSST/0" "$W/splitSolveSST/$END" "$STEPS" "$W/splitSolveSST/log.interFoam" \
       splitSolveSST uniform "$W/laminar/$END" "$W/sst/$END" || rc=1

# THE SPLIT-SCHEME ARMS. The control is the same case with ONE scheme for both equations.
"$BIN" "$W/splitDiv" "$W/splitDiv/0" "$W/splitDiv/$END" "$STEPS" "$W/splitDiv/log.interFoam" \
       splitDiv uniform "$W/laminar/$END" "$W/uniform/$END" || rc=1
"$BIN" "$W/splitDivSST" "$W/splitDivSST/0" "$W/splitDivSST/$END" "$STEPS" "$W/splitDivSST/log.interFoam" \
       splitDivSST uniform "$W/laminar/$END" "$W/sst/$END" || rc=1
# ...and epsilonWallFunction's `lowReCorrection`, whose control is the `uniform` run in the log law.
"$BIN" "$W/lowRe" "$W/lowRe/0" "$W/lowRe/$END" "$STEPS" "$W/lowRe/log.interFoam" \
       lowRe uniform "$W/laminar/$END" "$W/lowReOff/$END" || rc=1

echo "interfoam_ras_dambreak_vs_openfoam: rc $rc"
exit $rc
