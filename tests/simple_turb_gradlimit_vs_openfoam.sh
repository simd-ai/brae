#!/usr/bin/env bash
# grad(k) AND grad(omega) TAKE THE CASE'S OWN gradSchemes ENTRIES in the legacy kOmegaSST closure, against real OpenFOAM.
#
# OpenFOAM takes them in three places: CDkOmega = (2*alphaOmega2)*(fvc::grad(k) & fvc::grad(omega))/omega
# (kOmegaSSTBase.C:557), the corrected laplacian's non-orthogonal correction (correctedSnGrad.C:52-55, gradScheme
# "grad(" + field + ")") and limitedLinear's limiter (LimitedScheme.C:56, fvc::grad of the field itself). Each name
# resolves to its own entry, else `default` (schemesLookupDetail.C:76-88). brae took all three unlimited whatever the
# case said, and its parser kept one slot for four fields' entries. MEASURED on the coarse motorBike, cold, every solve
# tight (Arrhenius job 4860833): with grad(k) and grad(omega) cellLimited 1, brae's fields were bit-identical to the
# case without them, while OpenFOAM's omega moved 3.4e-02 and k 6.1e-03 at iteration 1; without the entries the two
# codes agree to 1.9e-13.
#
# MODE 3d: the coarse motorBike (353,784 cells, snappyHexMesh, upwind k and omega), cold, 5 tight iterations; the mesh
#          the tutorial's own way as tests/motorbike_tight_vs_openfoam.sh makes it, or MB_MESH=<a finished copy>.
# MODE 2d: validation/pitzDailySST restarted from OpenFOAM's own t=2000 (limitedLinear k and omega), 5 tight
#          iterations.
# In both modes brae runs BRAE_SST_F1_PATCH=1, the instrument that takes F1's patch half to OpenFOAM's
# (simple_sst_start_vs_openfoam: restarted k 1.55e-06 without it, 1.9e-14 with it), so the arms measure the gradients and
# not that known approximation; the production path is held separately (ARM noF1). MEASURED on 3d (JUPITER login node,
# 2026-11-03), the plain arm at iteration 3 without it: k's assembled system 8.6e-07 off OpenFOAM's, all of it in the
# 48 cells along one row of the background block, D and source off by the same 8e-06 factor; with it, 4e-13.
#
# ARMS, each one OpenFOAM run and one brae run on the same bytes, gradSchemes edited and nothing else:
#   plain    the case as shipped, no grad(k)/grad(omega) entry and an unlimited default
#   limited  grad(k) and grad(omega) cellLimited Gauss linear 1
#   split    grad(k) Gauss linear, grad(omega) cellLimited Gauss linear 1: two entries, two answers
#   widened  grad(k) and grad(omega) cellLimited Gauss linear 0.9: a coefficient below 1 widens the bounds
#            (cellLimitedGrad.C, (1/k - 1)*(max - min)). MEASURED on 2d (JUPITER login node, 2026-11-03): at 0.5 the
#            limiter clipped nothing, OpenFOAM's answer identical to the plain one to the last bit, so an arm there
#            could not tell a limited omega from an unlimited one and its control could not go red.
#   default  default cellLimited Gauss linear 1, with grad(U) and grad(p) named as shipped: the fallback
# THE ORACLE: OpenFOAM with tools/dumpKOmegaSST, OpenFOAM's own kOmegaSSTBase.C with writes added.
# THE BOUNDS. Fields U, p, k, omega at the last iteration: 3d 5e-7 and nut 5e-6, the bounds
#   motorbike_tight_vs_openfoam holds this mesh to (MEASURED there, job 4845479: worst k 4.3e-08, nut 1.4e-06);
#   2d 1e-10 (MEASURED pitzDailySST tight with BRAE_SST_F1_PATCH=1: 6.6e-12 over 40 iterations, job 4850430).
#   Intermediates at the dump iteration -- grad(k), grad(omega), CDkOmega, F1, and omega's and k's assembled systems
#   after relax and boundaryManipulate -- the same bound. The dump iteration is 3 on 3d, where grad(k) has left zero
#   (a cold k is uniform, so CDkOmega is identically zero at iteration 1), and 1 on 2d.
# PREMISE, asserted: OpenFOAM's limited, split, widened and default answers each differ from its plain one, and the
#   split and widened ones from the limited one, by at least SEP x the bound on the largest of U, k and omega, so an
#   arm that ignored its entries, or took one field's for both, could not pass; brae prints the entries it read.
# CONTROLS, asserted red on the limited arm: BRAE_CONTROL_TURB_GRAD_UNLIMITED=1 (the old code), =cd (CDkOmega
#   alone unlimited) and =eqn (the equations' laplacian correction and limitedLinear limiter alone); on the split
#   arm BRAE_CONTROL_TURB_GRAD_ONE_SLOT=1 (omega takes k's entry).
# ARM noF1: the production path, F1 patch half as shipped; the limited arm may sit no further from OpenFOAM than
#   NOF1_RATIO times the plain arm does, which is what the production path holds unlimited.
# ARM refusal (2d): grad(k) leastSquares must be refused by name, not run as Gauss linear.
# The bit-identity of the plain arms to the binary before this change is checked by the job that runs this gate
# against both binaries; a gate cannot hold an old binary.
set -u
MODE=${1:-2d}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SEP=${SEP:-1000}
NOF1_RATIO=${NOF1_RATIO:-4}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
of()
{
    bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"
}
of '[ -f "$FOAM_USER_LIBBIN/libdumpKOmegaSST.so" ]' < /dev/null \
    || { echo "SKIP: libdumpKOmegaSST.so not built -- (cd tools/dumpKOmegaSST && wmake libso)"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

if [ "$MODE" = 3d ]
then
    NITER=5
    DUMPIT=3
    START=0
    FB=5e-7
    NUTB=5e-6
    if [ -n "${MB_MESH:-}" ]
    then
        MESH="$MB_MESH"
    else
        MESH="$W/mesh"
        ( set +u; source "$OFBASHRC" > /dev/null 2>&1
          cp -r "$FOAM_TUTORIALS/incompressible/simpleFoam/motorBike" "$MESH" || exit 1
          sed -i -E 's/^endTime .*/endTime 1;/; s/^writeInterval .*/writeInterval 1;/' "$MESH/system/controlDict"
          cd "$MESH" && timeout 2400 ./Allrun > allrun.log 2>&1 ) < /dev/null
    fi
    [ -f "$MESH/constant/polyMesh/owner" ] && [ -d "$MESH/0.orig" ] || { echo "FAIL: no motorBike mesh at $MESH"; exit 1; }
    ARMS="plain limited split widened default"
elif [ "$MODE" = 2d ]
then
    NITER=5
    DUMPIT=1
    START=2000
    FB=1e-10
    NUTB=1e-10
    MESH="$ROOT/validation/pitzDailySST"
    [ -d "$MESH/2000" ] || { echo "SKIP: fixture $MESH/2000 missing"; exit 77; }
    ARMS="plain limited split widened default"
else
    echo "usage: $0 2d|3d"
    exit 2
fi
END=$((START + NITER))
DUMPT=$((START + DUMPIT))

cat > "$W/gate.py" <<'PY'
import os
import re
import sys

import numpy as np


def edit(fn, fun):
    s = open(fn).read()
    t = fun(s)
    open(fn, 'w').write(t)


def grads(case, arm):
    # The gradSchemes block is the only thing an arm changes. Entries are appended inside the block; `default` is
    # rewritten only on the default arm, with grad(U) and grad(p) named so that only k and omega fall through to it.
    def f(s):
        m = re.search(r'gradSchemes\s*\{([^}]*)\}', s)
        body = m.group(1)
        if arm == 'limited':
            body += '    grad(k)         cellLimited Gauss linear 1;\n    grad(omega)     cellLimited Gauss linear 1;\n'
        elif arm == 'split':
            body += '    grad(k)         Gauss linear;\n    grad(omega)     cellLimited Gauss linear 1;\n'
        elif arm == 'widened':
            body += '    grad(k)         cellLimited Gauss linear 0.9;\n    grad(omega)     cellLimited Gauss linear 0.9;\n'
        elif arm == 'default':
            gu = re.search(r'grad\(U\)\s+([^;]*);', body)
            body = re.sub(r'default\s+[^;]*;', 'default         cellLimited Gauss linear 1;', body)
            if not gu:
                body += '    grad(U)         Gauss linear;\n'
            body += '    grad(p)         Gauss linear;\n'
        elif arm == 'refusal':
            body += '    grad(k)         leastSquares;\n'
        assert re.search(r'default\s', body)
        return s[:m.start(1)] + body + s[m.end(1):]
    edit(case + '/system/fvSchemes', f)


def controls(case, start, end):
    def f(s):
        s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
        for k, v in (('startFrom', 'startTime'), ('startTime', str(start)), ('endTime', str(end)),
                     ('writeControl', 'timeStep'), ('writeInterval', str(end - start)), ('writeFormat', 'ascii'),
                     ('writePrecision', '17'), ('writeCompression', 'off')):
            s, n = re.subn(r'(?m)^%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
            if n == 0:
                s += '\n%s %s;\n' % (k, v)
        return s
    edit(case + '/system/controlDict', f)

    def g(s):
        s, n = re.subn(r'solvers\s*\{.*?\n\}', """solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0; maxIter 5000; }
    Phi { $p; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; nSweeps 1; tolerance 1e-13; relTol 0; maxIter 5000; }
}""", s, count=1, flags=re.S)
        assert n == 1
        return re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S)
    edit(case + '/system/fvSolution', g)


def dumpmodel(case):
    # OpenFOAM's arms run OpenFOAM's own kOmegaSST with writes added (tools/dumpKOmegaSST): same equations, and the
    # intermediates the comparison needs. brae's arms keep the case's kOmegaSST.
    def f(s):
        s, n = re.subn(r'(RASModel|model)(\s+)kOmegaSST\s*;', r'\1\2kOmegaSSTDump;', s)
        assert n == 1
        return s
    edit(case + '/constant/turbulenceProperties', f)
    edit(case + '/system/controlDict', lambda s: s + '\nlibs ("libdumpKOmegaSST.so");\n')


def readfoam(fn, n=None):
    s = open(fn, errors='replace').read()
    t = s[s.index('internalField'):]
    m = re.match(r'internalField\s+uniform\s+([^;]+);', t)
    if m:
        v = np.array(m.group(1).strip().strip('()').split(), dtype=float)
        return np.tile(v, (n, 1))
    m = re.match(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?\s*(\d+)\s*\n?\(', t)
    cnt, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    b = t[m.end():]
    b = b[:b.index('\n)')]
    return np.array(b.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(cnt, nc)


def readbrae(fn):
    lines = open(fn).read().split('\n')
    n, nc = int(lines[0].split()[1]), int(lines[0].split()[2])
    return np.array(' '.join(lines[1:n + 1]).split(), dtype=float).reshape(n, nc)


def rel(a, b):
    na = np.linalg.norm(a)
    return float(np.linalg.norm(b - a) / na) if na > 0 else (0.0 if np.linalg.norm(b) == 0 else float('inf'))


FIELDS = ('U', 'p', 'k', 'omega', 'nut')
STAGES = (('stage_sstGradKfield', 'stage_sstGradKfield'), ('stage_sstGradOmega', 'stage_sstGradOmega'),
          ('stage_sstCDkOmega', 'stage_sstCD'), ('stage_sstF1', 'stage_sstF1'),
          ('stage_sstOmD', 'stage_omegaD'), ('stage_sstOmSrc', 'stage_omegaSrc'),
          ('stage_sstKD', 'stage_kD'), ('stage_sstKSrc', 'stage_kSrc'))


def fields(a, b, t):
    return {f: rel(readfoam('%s/%s/%s' % (a, t, f)), readfoam('%s/%s/%s' % (b, t, f))) for f in FIELDS}


def stages(ofc, brc, t):
    n = readfoam('%s/%s/p' % (ofc, sys.argv[-1])).shape[0]
    out = {}
    for o, b in STAGES:
        out[o.replace('stage_sst', '')] = rel(readfoam('%s/%s/%s' % (ofc, t, o), n), readbrae('%s/stage/%s' % (brc, b)))
    return out


what = sys.argv[1]
if what == 'grads':
    grads(sys.argv[2], sys.argv[3])
elif what == 'controls':
    controls(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
elif what == 'dumpmodel':
    dumpmodel(sys.argv[2])
elif what == 'fields':
    # fields <a> <b> <time> <bound> <nutBound> <want: ok|red|print>
    a, b, t, fb, nb, want = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
    g = fields(a, b, t)
    ok = all(g[f] <= (nb if f == 'nut' else fb) for f in FIELDS)
    print('%s %s' % ('within' if ok else 'MISSES', ' '.join('%s %.2e' % (f, g[f]) for f in FIELDS)))
    sys.exit(0 if want == 'print' or ok == (want == 'ok') else 1)
elif what == 'sep':
    # sep <of-arm> <of-other> <time> <bound> <factor>: the arm's oracle must sit far from the other one
    g = fields(sys.argv[2], sys.argv[3], sys.argv[4])
    need = float(sys.argv[5]) * float(sys.argv[6])
    big = max(g['U'], g['k'], g['omega'])
    print('largest %.2e (needs >= %.1e): U %.2e k %.2e omega %.2e' % (big, need, g['U'], g['k'], g['omega']))
    sys.exit(0 if big >= need else 1)
elif what == 'stages':
    # stages <of> <brae> <dumpTime> <bound> <time of a full field for the cell count>
    g = stages(sys.argv[2], sys.argv[3], sys.argv[4])
    ok = all(v <= float(sys.argv[5]) for v in g.values())
    print('%s %s' % ('within' if ok else 'MISSES', ' '.join('%s %.2e' % kv for kv in g.items())))
    sys.exit(0 if ok else 1)
elif what == 'ratio':
    # ratio <of-lim> <brae-lim-noF1> <of-plain> <brae-plain-noF1> <time> <ratio> <floor>
    gl = fields(sys.argv[2], sys.argv[3], sys.argv[6])
    gp = fields(sys.argv[4], sys.argv[5], sys.argv[6])
    r, floor = float(sys.argv[7]), float(sys.argv[8])
    ok = all(gl[f] <= max(r * gp[f], floor) for f in FIELDS)
    print('%s limited %s | plain %s' % ('within' if ok else 'MISSES', ' '.join('%s %.2e' % (f, gl[f]) for f in FIELDS),
                                          ' '.join('%s %.2e' % (f, gp[f]) for f in FIELDS)))
    sys.exit(0 if ok else 1)
PY

stage()
{
    local c=$1
    local who=$2
    local arm=$3
    mkdir -p "$c/constant"
    cp -r "$MESH/system" "$c/"
    for e in "$MESH"/constant/*
    do
        [ "$(basename "$e")" = turbulenceProperties ] && cp "$e" "$c/constant/" || ln -s "$e" "$c/constant/"
    done
    if [ "$MODE" = 3d ]
    then
        cp -r "$MESH/0.orig" "$c/0"
    else
        cp -r "$MESH/2000" "$c/"
        rm -rf "${c:?}/2000/uniform"
    fi
    python3 -B "$W/gate.py" controls "$c" "$START" "$END"
    python3 -B "$W/gate.py" grads "$c" "$arm"
    [ "$who" = of ] && python3 -B "$W/gate.py" dumpmodel "$c"
    true
}
runof()
{
    local c=$1
    ( cd "$c" && of "BRAE_DUMP_STAGE_ITER=$DUMPIT timeout 3000 simpleFoam > run.log 2>&1" ) < /dev/null \
        || { echo "FAIL: OpenFOAM did not finish in $(basename "$c")"; tail -n 3 "$c/run.log"; exit 1; }
}
# The case's dictionaries #includeEtc, so brae needs OpenFOAM's environment as tests/tutorial_vs_openfoam.sh gives it.
runbr()
{
    local c=$1
    shift
    local f1=()
    f1=(BRAE_SST_F1_PATCH=1)
    ( set +u; source "$OFBASHRC" > /dev/null 2>&1; set -u; cd "$c" && env BRAE_PRESSURE_CASE_SOLVER=1 BRAE_TURBULENCE_CASE_SOLVER=1 \
        BRAE_SIMPLE_CASE_SOLVER=1 BRAE_DUMP_STAGE="$c/stage" BRAE_DUMP_STAGE_ITER="$DUMPIT" "${f1[@]}" "$@" "$BRAE" . > run.log 2>&1 < /dev/null )
}
brok()
{
    runbr "$@" || { echo "FAIL: brae exit $? in $(basename "$1"): $(grep -a -m1 -E 'brae ERROR|error' "$1/run.log" | cut -c1-160)"; exit 1; }
}

echo "== $MODE: $NITER tight iterations from t=$START, intermediates at iteration $DUMPIT =="
for arm in $ARMS
do
    stage "$W/of-$arm" of "$arm"
    runof "$W/of-$arm"
    stage "$W/br-$arm" br "$arm"
    brok "$W/br-$arm"
done

# PREMISE: brae read each arm's entries, and each oracle is far from the plain one.
want_line() { grep -q "turbulence gradients cellLimited (0=unlimited): grad(k)=$2 grad(omega)=$3 " "$W/br-$1/run.log"; }
want_line plain 0 0     && say "PREMISE brae read plain: grad(k)=0 grad(omega)=0" ok       || say "PREMISE brae read plain: grad(k)=0 grad(omega)=0" FAIL
want_line limited 1 1   && say "PREMISE brae read limited: grad(k)=1 grad(omega)=1" ok     || say "PREMISE brae read limited: grad(k)=1 grad(omega)=1" FAIL
want_line split 0 1     && say "PREMISE brae read split: grad(k)=0 grad(omega)=1" ok       || say "PREMISE brae read split: grad(k)=0 grad(omega)=1" FAIL
want_line widened 0.9 0.9 && say "PREMISE brae read widened: grad(k)=0.9 grad(omega)=0.9" ok || say "PREMISE brae read widened: grad(k)=0.9 grad(omega)=0.9" FAIL
want_line default 1 1   && say "PREMISE brae read default: grad(k)=1 grad(omega)=1" ok     || say "PREMISE brae read default: grad(k)=1 grad(omega)=1" FAIL
for pair in limited:plain split:plain widened:plain default:plain split:limited widened:limited
do
    a=${pair%%:*}
    b=${pair##*:}
    r=$(python3 -B "$W/gate.py" sep "$W/of-$a" "$W/of-$b" "$END" "$FB" "$SEP") && v=ok || v=FAIL
    say "PREMISE OpenFOAM $a vs $b: $r" "$v"
done

# THE ARMS: fields at the end, intermediates at the dump iteration.
for arm in $ARMS
do
    r=$(python3 -B "$W/gate.py" fields "$W/of-$arm" "$W/br-$arm" "$END" "$FB" "$NUTB" ok) && v=ok || v=FAIL
    say "$arm fields: $r" "$v"
    r=$(python3 -B "$W/gate.py" stages "$W/of-$arm" "$W/br-$arm" "$DUMPT" "$FB" "$END") && v=ok || v=FAIL
    say "$arm iteration $DUMPIT: $r" "$v"
done

# CONTROLS: each must miss.
for ctl in 1 cd eqn
do
    stage "$W/ctl-$ctl" br limited
    brok "$W/ctl-$ctl" BRAE_CONTROL_TURB_GRAD_UNLIMITED="$ctl"
    grep -q "brae CONTROL: BRAE_CONTROL_TURB_GRAD_UNLIMITED=$ctl" "$W/ctl-$ctl/run.log" || say "CONTROL =$ctl did not announce itself" FAIL
    r=$(python3 -B "$W/gate.py" fields "$W/of-limited" "$W/ctl-$ctl" "$END" "$FB" "$NUTB" red) && v=ok || v=FAIL
    say "CONTROL limited, grad limiter ignored ($ctl) is red: $r" "$v"
done
stage "$W/ctl-slot" br split
brok "$W/ctl-slot" BRAE_CONTROL_TURB_GRAD_ONE_SLOT=1
r=$(python3 -B "$W/gate.py" fields "$W/of-split" "$W/ctl-slot" "$END" "$FB" "$NUTB" red) && v=ok || v=FAIL
say "CONTROL split, omega on k's entry is red: $r" "$v"

# The production path, F1's patch half as shipped: limited no further from OpenFOAM than plain is.
for arm in plain limited
do
    stage "$W/nof1-$arm" br "$arm"
    ( set +u; source "$OFBASHRC" > /dev/null 2>&1; set -u; cd "$W/nof1-$arm" && env BRAE_PRESSURE_CASE_SOLVER=1 \
        BRAE_TURBULENCE_CASE_SOLVER=1 BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" . > run.log 2>&1 < /dev/null ) \
        || { echo "FAIL: brae noF1 $arm"; exit 1; }
done
r=$(python3 -B "$W/gate.py" ratio "$W/of-limited" "$W/nof1-limited" "$W/of-plain" "$W/nof1-plain" "$END" "$NOF1_RATIO" "$FB") && v=ok || v=FAIL
say "noF1 limited within ${NOF1_RATIO}x of plain: $r" "$v"
if [ "$MODE" = 2d ]
then
    # grad(k) leastSquares: refused by name, never run.
    stage "$W/refusal" br refusal
    if runbr "$W/refusal"
    then
        say "REFUSAL grad(k) leastSquares ran instead of refusing" FAIL
    else
        grep -q "grad(k) resolves to \`leastSquares\`" "$W/refusal/run.log" && say "REFUSAL grad(k) leastSquares refused by name" ok \
            || say "REFUSAL grad(k) leastSquares failed without naming the entry: $(grep -a -m1 'brae' "$W/refusal/run.log" | cut -c1-100)" FAIL
    fi
fi
[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
