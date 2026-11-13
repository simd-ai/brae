#!/usr/bin/env bash
# laplacianSchemes `limited corrected <psi>` on the legacy driver's momentum laplacian and its closures' (k, omega, epsilon,
# nuTilda), against OpenFOAM serial, on snappyHexMesh's Ahmed body (tests/ahmed_case.py: three prism layers on the body,
# polyhedra where its refinement levels meet, every laplacian `limited corrected 0.33`, every wall under wall functions).
#
#   simple_ahmed_limited_vs_openfoam.sh
#       BRAE_BIN      brae (default <root>/build/brae; it runs simpleFoam and pimpleFoam cases alike)
#       OFBASHRC      OpenFOAM v2412's bashrc
#       AHMED_H       the blockMesh background's cell (default 0.08: 311,172 cells)
#       AHMED_ITERS   OpenFOAM simpleFoam's iteration every start arm begins from (default 150; kEpsilon's after as many of
#                     kOmegaSST's)
#       KEEP_W        keep the work in this directory
#
# WHY. The driver took the cap on the pressure correction only and ran the momentum laplacian's and every closure's correction
# unlimited, on the steady and the transient step alike. From rest OpenFOAM's cap is zero across every face whose two cells hold
# one value (limitedSnGrad.C:48-58). The multi-GPU driver's copy of the steady step, taking the same unlimited correction, was U
# 1.1e-3 and p 6.7e-4 from OpenFOAM serial after one iteration on this body (Arrhenius job 5042510), and on a 259.7 M-cell
# DrivAer from rest it put U 5.94 m/s (16%) below OpenFOAM's in prism cells on the mirror and omega's assembled source up to 9x
# OpenFOAM's beside the walls at the first iteration; that run diverged at iteration 28.
# ARMS (tight solves in both codes: p to 1e-12 at relTol 0, the rest to 1e-12 at relTol 0, maxIter 2000; brae in OpenFOAM's sweep
# order, BRAE_SIMPLE_CASE_SOLVER=1 BRAE_TURBULENCE_CASE_SOLVER=1; every iteration written at 17 digits)
#   rest     kOmegaSST, the case's uniform start, ONE iteration: U, p and phi. Not k, omega and nut: their limitedLinear limiter
#            on a uniform field is decided by round-off ties, which neither code can be held to
#   sst      kOmegaSST from OpenFOAM's own iteration AHMED_ITERS, 3 iterations: U, p, phi, k, omega and nut
#   ke       kEpsilon, the same, under SIMPLEC (`consistent yes`: the flux's fvc::snGrad(p) takes snGradSchemes' cap): U, p,
#            phi, k, epsilon and nut. Its OpenFOAM start runs from kOmegaSST's iteration AHMED_ITERS (epsilon = Cmu k omega),
#            not from the uniform one, from which OpenFOAM's own kEpsilon diverges on this body (ahmed_case.py fromsst)
#   sa       SpalartAllmaras (nutUSpaldingWallFunction), the same: U, p, phi, nuTilda and nut
#   pimple   kOmegaSST transient from the sst arm's start: pimpleFoam, Euler, one outer and two pressure correctors, no
#            relaxation, 2 steps of 2e-4 s: U, p, phi, k, omega and nut
#   zero     `limited corrected 0` against `uncorrected` in brae alone, the sst arm: OpenFOAM's cap at 0 is no correction at
#            all with the corrected scheme's coefficients (limitedSnGrad.C:48-58, limitedSnGrad.H:165-178), which is
#            `uncorrected`; byte for byte, every field of every iteration
# THE ORACLE: OpenFOAM v2412 serial; beside every arm but rest two twins of it: its one-ulp twin (the first cell's Ux one ulp
# higher), how far round-off alone carries it, and its SOLVER TWIN (p by PCG/DIC to the same tolerance), how far the stopping point
# of a converged pressure solve carries it (brae's pressure is its AMG-PCG whatever the case names)
# CHECKS: every field's internal values and every patch's, nut's included, within 1e-10 or 5 x the larger twin's distance, the
# larger. nut's patches are correctNut's of the iteration (refreshNutBoundary): the driver wrote the boundary nut its momentum
# assembly had made before the closure, one correction behind (OpenFOAM's n - 1 at brae's n, outlet 3.3e-2 from OpenFOAM's n)
# CONTROLS, each red against the same bounds:
#   BRAE_CONTROL_LAP_UNLIMITED=1        the unlimited correction on U and the closures, as before: rest, sst and pimple
#   BRAE_CONTROL_LAP_UNLIMITED=closure  U capped, the closure's correction unlimited: sst, ke and sa, on the closure's own fields
#   BRAE_CONTROL_LAP_UNLIMITED=simplec  the SIMPLEC flux's snGrad(p) correction whole: ke, on U, p and phi
#   BRAE_CONTROL_NUT_BOUNDARY_LAG=1     nut's patches one correction behind, as before: sst, on nut alone (and the rest of its
#                                       fields within the bound: the written patches are read by nothing within a run)
#   BRAE_CONTROL_WALL_NUT_ZERO_SEED=1   Spalding's first Newton from zero, not from nut's patch values as read, as before: sa
#   BRAE_CONTROL_PASS0_PREDICTOR_GRAD=1 a transient step's second pressure corrector's laplacian correction from the momentum
#                                       predictor's grad(p), not the current p's, as before: pimple
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
GEN="$ROOT/tests/ahmed_case.py"
CHECK="$ROOT/tests/check_fields.py"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
export HWLOC_COMPONENTS=-gl PYTHONDONTWRITEBYTECODE=1
AHMED_H=${AHMED_H:-0.08}
AHMED_ITERS=${AHMED_ITERS:-150}
ITERS=3
STEPS=2
DT=2e-4
FLOOR=1e-10
FACTOR=5
passes=0
fails=0
fail()
{
    echo "FAIL $*"
    fails=$((fails + 1))
}
ok()
{
    passes=$((passes + 1))
    echo "ok   $*"
}
finish()
{
    echo "== $passes passed, $fails failed (work in $W)"
    [ "$fails" = 0 ] && echo PASS || echo FAIL
    [ "$fails" = 0 ]
    exit $?
}
# of <command>: in OpenFOAM's environment
of()
{
    bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; set -e; $*" < /dev/null
}

# fields_of <model>: the fields an arm of that closure compares, nut held apart
fields_of()
{
    case $1 in
        kOmegaSST) echo "U p phi k omega" ;;
        kEpsilon) echo "U p phi k epsilon" ;;
        SpalartAllmaras) echo "U p phi nuTilda" ;;
    esac
}

# closure_fields <model>: the closure's own transported fields
closure_fields()
{
    case $1 in
        kOmegaSST) echo "k omega" ;;
        kEpsilon) echo "k epsilon" ;;
        SpalartAllmaras) echo "nuTilda" ;;
    esac
}

# stage <dir> <model> <rest|start|pimple> [ulp|solver|zero]: the mesh, the case's entries with tight solves, and the arm's start;
# ulp: the one-ulp twin; solver: the solver twin (p by PCG/DIC); zero: every `limited corrected 0.33` made `limited corrected 0`
stage()
{
    local d=$1 model=$2 arm=$3 twin=${4:-} n=$ITERS
    rm -rf "$d"
    mkdir -p "$d/0"
    cp -r "$W/rans-$model/constant" "$W/rans-$model/system" "$d/"
    rm -f "$d"/constant/polyMesh/.brae_*
    if [ "$arm" = rest ]
    then
        n=1
        cp "$W/rans-$model/0"/* "$d/0/"
    else
        local f
        for f in U p phi nut $(closure_fields "$model")
        do
            cp "$W/rans-$model/$AHMED_ITERS/$f" "$d/0/" || return 1
        done
    fi
    if [ "$arm" = pimple ]
    then
        python3 -B "$GEN" pimple "$d" "$STEPS" "$DT" || return 1
    elif [ "$model" = kEpsilon ]
    then
        sed -i 's/consistent false;/consistent true;/' "$d/system/fvSolution"
        grep -q 'consistent true;' "$d/system/fvSolution" || return 1
    fi
    if [ "$arm" != pimple ]
    then
        python3 -B - "$d" "$n" << 'PY' || return 1
import re, sys
d, n = sys.argv[1], int(sys.argv[2])
c = d + '/system/controlDict'
s = open(c).read()
s2 = re.sub(r'endTime \d+;', 'endTime %d;' % n, s)
s2 = re.sub(r'writeInterval \d+;', 'writeInterval 1;', s2)
assert s2 != s and 'endTime %d;' % n in s2 and 'startTime 0;' in s2, 'controlDict not staged'
open(c, 'w').write(s2)
f = d + '/system/fvSolution'
s = open(f).read()
s2 = s.replace('tolerance 1e-8; relTol 0.01; maxIter 5000;', 'tolerance 1e-12; relTol 0; maxIter 5000;')
s2 = s2.replace('tolerance 1e-10; relTol 0.1; maxIter 2000;', 'tolerance 1e-12; relTol 0; maxIter 2000;')
assert s2.count('relTol 0;') == 2, 'fvSolution not staged'
open(f, 'w').write(s2)
PY
    fi
    case $twin in
        ulp)
            python3 -B "$GEN" ulp "$d" || return 1
            ;;
        solver)
            sed -i 's/p { solver GAMG; smoother GaussSeidel;/p { solver PCG; preconditioner DIC;/' "$d/system/fvSolution"
            sed -i 's/pFinal { solver GAMG; smoother GaussSeidel;/pFinal { solver PCG; preconditioner DIC;/' "$d/system/fvSolution"
            grep -q 'solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0;' "$d/system/fvSolution" \
                && ! grep -q GAMG "$d/system/fvSolution" || return 1
            ;;
        zero)
            sed -i 's/limited corrected 0.33;/limited corrected 0;/g' "$d/system/fvSchemes"
            [ "$(grep -c 'limited corrected 0;' "$d/system/fvSchemes")" = 2 ] || return 1
            ;;
        uncorrected)
            sed -i 's/limited corrected 0.33;/uncorrected;/g' "$d/system/fvSchemes"
            [ "$(grep -c ' uncorrected;' "$d/system/fvSchemes")" = 2 ] || return 1
            ;;
    esac
}

# run_of <case>: the case's application, stopped at its first NaN residual (exit 3). OpenFOAM's floating-point trap does not stop it
# on the Grace CPU: job 5055451's kEpsilon start went NaN at iteration ~30 and solved NaN to every maxIter for the job's remaining
# 3 h, and job 5063071's did the same with FOAM_SIGFPE=true
run_of()
{
    local app=simpleFoam
    grep -q "application pimpleFoam" "$1/system/controlDict" && app=pimpleFoam
    of "cd '$1' && { $app > log 2>&1 & a=\$!; while kill -0 \$a 2> /dev/null; do if grep -q -m1 -E 'residual = -?nan' log; then kill \$a; wait \$a || true; echo 'NaN residual: stopped' >> log; exit 3; fi; sleep 5; done; wait \$a; }"
    echo $? > "$1/rc"
}

# run_brae <dir> [VAR=value ...]
run_brae()
{
    local dir=$1
    shift
    (cd "$dir" && env -u BRAE_CONTROL_LAP_UNLIMITED -u BRAE_CONTROL_NUT_BOUNDARY_LAG -u BRAE_CONTROL_WALL_NUT_ZERO_SEED \
        -u BRAE_CONTROL_PASS0_PREDICTOR_GRAD BRAE_SIMPLE_CASE_SOLVER=1 BRAE_TURBULENCE_CASE_SOLVER=1 "$@" \
        timeout 1800 "$BRAE" -case . > log 2> err < /dev/null; echo $? > rc)
}

ran()
{
    if [ "$(cat "$1/rc" 2> /dev/null)" != 0 ]
    then
        fail "$2 exited $(cat "$1/rc" 2> /dev/null): $(grep -h 'ERROR\|error\|refused\|FOAM FATAL' "$1/log" "$1/err" 2> /dev/null | head -n 1 | cut -c1-200)"
        return 1
    fi
    return 0
}

# times <dir>: the written times after 0, in order
times()
{
    (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -v '^0$' | sort -g)
}

# yardstick <OpenFOAM run> <twins, comma-separated, or -> <time> <fields...>: the largest twin's distance from the run
yardstick()
{
    local ref=$1 twins=$2 t=$3 tw y m=0
    shift 3
    [ "$twins" = - ] && { echo 0; return; }
    for tw in ${twins//,/ }
    do
        y=$(python3 "$CHECK" fields "$tw/$t" "$ref/$t" 1 "$@" 2>&1 | sed -n 's/^worst \([^ ]*\).*/\1/p')
        m=$(python3 -c "print(max($m, float('${y:-inf}')))")
    done
    echo "$m"
}

# compare <run> <OpenFOAM run> <twins or -> <nut: lag|whole|none> <fields...>: the figures of every written time; returns 1 on a FAIL
compare()
{
    local run=$1 ref=$2 twins=$3 nutmode=$4 t prev="" y bound out bad=0 figs="" nut
    shift 4
    for t in $(times "$ref")
    do
        y=$(yardstick "$ref" "$twins" "$t" "$@" nut)
        bound=$(python3 -c "print('%.3g' % max($FLOOR, $FACTOR * float('${y:-inf}')))")
        out=$(python3 "$CHECK" fields "$run/$t" "$ref/$t" "$bound" "$@" 2>&1)
        case $nutmode in
            lag) nut=$(python3 "$CHECK" lag "$run/$t" "$ref/$t" "$ref/${prev:-0}" "$bound" nut 2>&1) ;;
            whole) nut=$(python3 "$CHECK" fields "$run/$t" "$ref/$t" "$bound" nut 2>&1 | tail -n 1 | sed 's/^worst/ok nut/')
                   python3 "$CHECK" fields "$run/$t" "$ref/$t" "$bound" nut > /dev/null 2>&1 || nut="FAIL ${nut#ok }" ;;
            *) nut=ok ;;
        esac
        if echo "$out" | grep -q '^FAIL' || ! echo "$out" | grep -q '^worst' || [ "${nut%% *}" != ok ]
        then
            bad=1
            figs="$figs t $t: $(echo "$out" | grep -m1 -E '^FAIL|rror' | cut -c1-160) ${nut} (bound $bound);"
        else
            figs="$figs t $t $(echo "$out" | tail -n 1 | sed 's/^worst //'), ${nut#ok } (bound $bound);"
        fi
        prev=$t
    done
    [ -n "$figs" ] || { echo " no written time"; return 1; }
    echo "$figs"
    return $bad
}

# control_red <run> <OpenFOAM run> <twins or -> <nut mode> <label> <fields...>
control_red()
{
    local run=$1 ref=$2 twins=$3 nutmode=$4 label=$5 figs
    shift 5
    if [ "$(cat "$run/rc")" != 0 ]
    then
        ok "control $label red: the run stopped (exit $(cat "$run/rc"))"
        return
    fi
    if figs=$(compare "$run" "$ref" "$twins" "$nutmode" "$@")
    then
        fail "control $label: passed:$figs"
    else
        ok "control $label red:$figs"
    fi
}

echo "== the Ahmed mesh"
python3 -B "$GEN" mesh "$W/mesh" "$AHMED_H"
of "cd '$W/mesh' && blockMesh > log.blockMesh 2>&1 && surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 && checkMesh > log.checkMesh 2>&1" \
    || { fail "the Ahmed mesh: $(tail -n 3 "$W/mesh"/log.* | tr '\n' ' ' | cut -c1-300)"; finish; }
poly=$(sed -n 's/^ *polyhedra: *\([0-9]*\).*/\1/p' "$W/mesh/log.checkMesh" | head -n 1)
ncell=$(sed -n 's/^ *cells: *\([0-9]*\).*/\1/p' "$W/mesh/log.checkMesh" | head -n 1)
layers=$(awk '/^patch +faces +layers/ { on = 1; next } on && $1 == "body" { print $3; exit }' "$W/mesh/log.snappyHexMesh")
if [ "${poly:-0}" -gt 0 ] && awk -v l="${layers:-0}" 'BEGIN { exit !(l > 0) }'
then
    ok "the Ahmed mesh: $ncell cells, $poly polyhedra, $layers prism layers on the body (snappyHexMesh's mean); $(grep -E 'non-orthogonality Max' "$W/mesh/log.checkMesh" | sed 's/^ *//' | cut -c1-80)"
else
    fail "the Ahmed mesh: ${poly:-no} polyhedra, ${layers:-no} layers on the body"
fi

echo "== OpenFOAM's starts: simpleFoam to iteration $AHMED_ITERS, each closure (kEpsilon's from kOmegaSST's)"
for model in kOmegaSST kEpsilon SpalartAllmaras
do
    rm -rf "$W/rans-$model"
    mkdir -p "$W/rans-$model/constant"
    cp -r "$W/mesh/constant/polyMesh" "$W/rans-$model/constant/"
    cp -r "$W/mesh/system" "$W/rans-$model/"
    python3 -B "$GEN" rans "$W/rans-$model" "$AHMED_ITERS" "$model"
done
run_of "$W/rans-SpalartAllmaras" &
# kEpsilon from the uniform start diverges in OpenFOAM itself (ahmed_case.py fromsst): its start is kOmegaSST's iteration
# AHMED_ITERS, then AHMED_ITERS of its own
run_of "$W/rans-kOmegaSST"
if [ -f "$W/rans-kOmegaSST/$AHMED_ITERS/phi" ]
then
    python3 -B "$GEN" fromsst "$W/rans-kEpsilon" "$W/rans-kOmegaSST/$AHMED_ITERS" && run_of "$W/rans-kEpsilon"
fi
wait
for model in kOmegaSST kEpsilon SpalartAllmaras
do
    grep -q "default Gauss linear limited corrected 0.33" "$W/rans-$model/system/fvSchemes" \
        || fail "$model: the case's laplacianSchemes are not limited corrected 0.33"
    if [ -f "$W/rans-$model/$AHMED_ITERS/phi" ]
    then
        ok "the $model start: OpenFOAM simpleFoam to iteration $AHMED_ITERS"
    else
        fail "the $model start: OpenFOAM simpleFoam did not reach iteration $AHMED_ITERS ($(tail -n 2 "$W/rans-$model/log" | tr '\n' ' ' | cut -c1-200))"
    fi
done
[ "$fails" = 0 ] || finish

echo "== OpenFOAM serial: every arm and its twins"
stage "$W/of-rest" kOmegaSST rest || fail "the rest arm's case"
run_of "$W/of-rest" &
for arm in sst:kOmegaSST:start ke:kEpsilon:start sa:SpalartAllmaras:start pimple:kOmegaSST:pimple
do
    IFS=: read -r name model kind <<< "$arm"
    stage "$W/of-$name" "$model" "$kind" || fail "the $name arm's case"
    stage "$W/ofu-$name" "$model" "$kind" ulp || fail "the $name arm's one-ulp twin"
    stage "$W/ofs-$name" "$model" "$kind" solver || fail "the $name arm's solver twin"
    run_of "$W/of-$name" &
    run_of "$W/ofu-$name" &
    run_of "$W/ofs-$name" &
done

echo "== brae: every arm and the controls"
# brae_arm <name> <model> <rest|start|pimple> <twin|-> [VAR=value ...]
brae_arm()
{
    local name=$1 model=$2 kind=$3 twin=$4
    shift 4
    if [ "$twin" = - ]
    then
        twin=""
    fi
    stage "$W/$name" "$model" "$kind" $twin || { fail "$name: the case"; echo 1 > "$W/$name/rc"; return; }
    run_brae "$W/$name" "$@"
}
brae_arm b-rest kOmegaSST rest -
brae_arm x-rest kOmegaSST rest - BRAE_CONTROL_LAP_UNLIMITED=1
brae_arm b-sst kOmegaSST start -
brae_arm x-sst kOmegaSST start - BRAE_CONTROL_LAP_UNLIMITED=1
brae_arm xc-sst kOmegaSST start - BRAE_CONTROL_LAP_UNLIMITED=closure
brae_arm xn-sst kOmegaSST start - BRAE_CONTROL_NUT_BOUNDARY_LAG=1
brae_arm b-ke kEpsilon start -
brae_arm xc-ke kEpsilon start - BRAE_CONTROL_LAP_UNLIMITED=closure
brae_arm xs-ke kEpsilon start - BRAE_CONTROL_LAP_UNLIMITED=simplec
brae_arm b-sa SpalartAllmaras start -
brae_arm xc-sa SpalartAllmaras start - BRAE_CONTROL_LAP_UNLIMITED=closure
brae_arm xw-sa SpalartAllmaras start - BRAE_CONTROL_WALL_NUT_ZERO_SEED=1
brae_arm b-pimple kOmegaSST pimple -
brae_arm x-pimple kOmegaSST pimple - BRAE_CONTROL_LAP_UNLIMITED=1
brae_arm x0-pimple kOmegaSST pimple - BRAE_CONTROL_PASS0_PREDICTOR_GRAD=1
brae_arm z-zero kOmegaSST start zero
brae_arm z-unc kOmegaSST start uncorrected
wait

echo "== checks"
for d in of-rest of-sst ofu-sst ofs-sst of-ke ofu-ke ofs-ke of-sa ofu-sa ofs-sa of-pimple ofu-pimple ofs-pimple
do
    ran "$W/$d" "OpenFOAM $d" || finish
done
for d in ofs-sst ofs-ke ofs-sa ofs-pimple
do
    grep -q "^DICPCG:  Solving for p" "$W/$d/log" || fail "$d: the solver twin did not solve p by PCG/DIC"
done
# every OpenFOAM solve stopped at its tolerance, not at maxIter: otherwise the comparison is of two stopping points
hit=$(grep -h -E "No Iterations (2000|5000)$" "$W"/of-*/log "$W"/ofs-*/log | head -n 1)
[ -z "$hit" ] && ok "OpenFOAM's solves all stopped at their tolerance (none at maxIter)" || fail "an OpenFOAM solve stopped at maxIter: $hit"
# the arms' premise: brae ran the case's order and the case's scheme
grep -aq "swept in colour order" "$W/b-sst/log" && fail "brae's sst arm swept in colour order"
for name in rest sst ke sa pimple
do
    model=kOmegaSST
    [ "$name" = ke ] && model=kEpsilon
    [ "$name" = sa ] && model=SpalartAllmaras
    nutmode=whole
    twins="$W/ofu-$name,$W/ofs-$name"
    flds=$(fields_of "$model")
    if [ "$name" = rest ]
    then
        twins=-
        nutmode=none
        flds="U p phi"
    fi
    if ran "$W/b-$name" "brae $name"
    then
        figs=$(compare "$W/b-$name" "$W/of-$name" "$twins" "$nutmode" $flds) \
            && ok "$name ($model), brae against OpenFOAM serial ($flds):$figs" \
            || fail "$name ($model), brae against OpenFOAM serial ($flds):$figs"
    fi
done
# zero: `limited corrected 0` and `uncorrected` value for value (a relative L2 of exactly 0: every value equal, a signed zero
# equal to its unsigned twin), and the 0.33 arm not
if ran "$W/z-zero" "brae limited corrected 0" && ran "$W/z-unc" "brae uncorrected"
then
    nz=0
    nd=0
    for t in $(times "$W/z-unc")
    do
        nz=$((nz + 1))
        python3 "$CHECK" fields "$W/z-zero/$t" "$W/z-unc/$t" 0 U p phi k omega nut > "$W/z-$t.txt" 2>&1 || nd=$((nd + 1))
    done
    [ "$nz" -gt 0 ] && [ "$nd" = 0 ] && ok "zero: limited corrected 0 is uncorrected value for value (U p phi k omega nut, $nz iterations)" \
        || fail "zero: limited corrected 0 differs from uncorrected at $nd of $nz iterations: $(grep -h -m1 FAIL "$W"/z-*.txt | cut -c1-160)"
    python3 "$CHECK" fields "$W/z-zero/$ITERS" "$W/b-sst/$ITERS" 0 U > /dev/null 2>&1 \
        && fail "zero: limited corrected 0 wrote the 0.33 arm's U (the scheme was not read)"
fi
control_red "$W/x-rest" "$W/of-rest" - none "BRAE_CONTROL_LAP_UNLIMITED=1 rest" U p phi
control_red "$W/x-sst" "$W/of-sst" "$W/ofu-sst,$W/ofs-sst" whole "BRAE_CONTROL_LAP_UNLIMITED=1 sst" $(fields_of kOmegaSST)
control_red "$W/xn-sst" "$W/of-sst" "$W/ofu-sst,$W/ofs-sst" whole "BRAE_CONTROL_NUT_BOUNDARY_LAG=1 sst" nut
# ...and its other fields within the bound: the lag is the written file's alone
if [ "$(cat "$W/xn-sst/rc")" = 0 ]
then
    figs=$(compare "$W/xn-sst" "$W/of-sst" "$W/ofu-sst,$W/ofs-sst" lag $(fields_of kOmegaSST)) \
        && ok "control BRAE_CONTROL_NUT_BOUNDARY_LAG=1 sst: every other field, and nut's patches at OpenFOAM's n - 1, within the bound:$figs" \
        || fail "control BRAE_CONTROL_NUT_BOUNDARY_LAG=1 sst: moved more than nut's written patches:$figs"
fi
control_red "$W/x-pimple" "$W/of-pimple" "$W/ofu-pimple,$W/ofs-pimple" whole "BRAE_CONTROL_LAP_UNLIMITED=1 pimple" $(fields_of kOmegaSST)
control_red "$W/x0-pimple" "$W/of-pimple" "$W/ofu-pimple,$W/ofs-pimple" none "BRAE_CONTROL_PASS0_PREDICTOR_GRAD=1 pimple" U p phi
control_red "$W/xw-sa" "$W/of-sa" "$W/ofu-sa,$W/ofs-sa" whole "BRAE_CONTROL_WALL_NUT_ZERO_SEED=1 sa" U
control_red "$W/xs-ke" "$W/of-ke" "$W/ofu-ke,$W/ofs-ke" none "BRAE_CONTROL_LAP_UNLIMITED=simplec ke" U p phi
grep -q "consistent true;" "$W/b-ke/system/fvSolution" || fail "the ke arm is not SIMPLEC"
for name in sst ke sa
do
    model=kOmegaSST
    [ "$name" = ke ] && model=kEpsilon
    [ "$name" = sa ] && model=SpalartAllmaras
    control_red "$W/xc-$name" "$W/of-$name" "$W/ofu-$name,$W/ofs-$name" none "BRAE_CONTROL_LAP_UNLIMITED=closure $name" \
        $(closure_fields "$model")
done
finish
