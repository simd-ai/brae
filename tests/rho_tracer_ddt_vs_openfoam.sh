#!/usr/bin/env bash
# scalarTransport on the rhoSimpleFoam OF-mirror arms under `ddtSchemes default Euler`, against OpenFOAM v2412.
#
# WHAT IT HOLDS. OpenFOAM's scalarTransport on a mass flux solves fvm::ddt(rho, s) + fvm::div(phi, s) -
# fvm::laplacian(D, s) (scalarTransport.C:315-332), nCorr + 1 times (`i <= nCorr_`, default nCorr_(0)), stopping once a
# solve's initial residual is below the object's `tolerance` (default 1). Under Euler the ddt is rDeltaT*rho*V on the
# diagonal and rDeltaT*rho.oldTime()*s.oldTime()*V on the source (EulerDdtScheme.C:437-470), its scheme looked up as
# `ddt(rho,tracer0)`. The field is NO_WRITE (:66-67): the object's write() writes it into the time it ran at, right after
# that execute (:379-389). The mirror arms
#   (a) wrote it beside the solved fields, before the object had run: every intermediate time held the previous
#       iteration's solve, and the first time after a restart held none;
#   (b) dropped the ddt;
#   (c) read nCorr as the number of solves (default 1): an explicit `nCorr 1` solved once where OpenFOAM solves twice;
#   (d) took Gauss linear for grad(tracer0), which limitedLinear's limiter (LimitedScheme.C:54) and the corrected
#       laplacian (correctedSnGrad.C:52-55) take under the case's entry: gasMixing's `default leastSquares`.
#
# THE CASE. compressible/rhoSimpleFoam/gasMixing/injectorPipe, the one rhoSimpleFoam tutorial that ships Euler and a
# tracer (tracer0, D 0.001, `Gauss limitedLinear 1`), meshed by its own Allrun.pre. Its scalarTransport object is
# kept as the tutorial writes it (system/scalarTransport, inlined); its abort and sampling objects are dropped. Every
# linear solver is tightened to 1e-14 relTol 0, and BOTH codes restart from OpenFOAM's own iteration DEV, the protocol
# of tests/rho_gasmixing_vs_openfoam.sh (whose header says why: limitedLinear's 0/0 branch from rest, and
# rho.oldTime()'s first-use rule).
#
# ARMS, host and CUDA, iterations DEV+1..LAST, every field (U p T k epsilon nut tracer0) by relative L2 within BOUND:
#   T1  the tutorial's tracer
#   T2  the same with `nCorr 1; tolerance 1e-30;`: OpenFOAM solves the tracer twice an iteration
# PREMISES (OpenFOAM against itself): `ddt(tracer0) steadyState` moves tracer0 off T1's; T2's tracer0 is off T1's.
# CONTROLS, each on tracer0, both arms, a control that writes no tracer0 to judge being a FAIL:
#   BRAE_CONTROL_TRACER_WRITE_BEFORE_SOLVE=1 (a, as before) red on T1 at DEV+2..LAST, where both codes write the field;
#   BRAE_CONTROL_TRACER_DDT_IGNORED=1 (b, as before) red on T1;
#   BRAE_CONTROL_TRACER_NCORR_AS_SOLVES=1 (c, as before) red on T2;
#   BRAE_CONTROL_TRACER_GRAD_GAUSS=1 (d, as before) red on T1.
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28, jobs 5877691, 5877983), worst field at 6-8:
#   develop ca5451a5: no tracer0 at 6; tracer0 at 7 and 8, T1 1.3e-01 and 1.2e-03, T2 3.3e-01 and 2.8e-01, both arms
#   this change: T1 1.8e-11 host, 1.5e-11 CUDA (U at 8; tracer0 2.1e-12, 1.9e-12); T2 the same fields, tracer0 7.8e-12,
#   2.6e-12. So BOUND 1e-10, the bound tests/rho_gasmixing_vs_openfoam.sh holds the same protocol to.
#   controls: write 1.3e-01 (at 7), ddt 2.4e-06, nCorr 2.8e-01, gradient 1.2e-03; premises 2.4e-06 (the ddt) and 3.3e-01.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
DEV=5
LAST=8
BOUND=${BRAE_TT_BOUND:-1e-10}
PREMISE=1e-6
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set --
set +u
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
TUTS="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}"
SRC="$TUTS/compressible/rhoSimpleFoam/gasMixing/injectorPipe"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }
W=${BRAE_TT_KEEP:-$(mktemp -d)}
[ -n "${BRAE_TT_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-96s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
echo "rho_tracer_ddt_vs_openfoam: gasMixing/injectorPipe's tracer under Euler, restarted from OpenFOAM's iteration $DEV"

# the tutorial's own meshing; snappyHexMesh runs in parallel and is reconstructed to serial
rm -rf "$W/mesh"
cp -r "$SRC" "$W/mesh" || exit 1
( cd "$W/mesh" && ./Allrun.pre > log.pre 2>&1 ) || { echo "SKIP: the tutorial's own meshing did not run here"; exit 77; }
if [ -d "$W/mesh/processor0/constant/polyMesh" ]; then
    ( cd "$W/mesh" && reconstructParMesh -constant > log.rpm 2>&1 && rm -rf processor* )
fi
[ -f "$W/mesh/constant/polyMesh/owner" ] || { echo "SKIP: no mesh produced"; exit 77; }

# stage <dest> <endTime> <arm>
stage()
{
    rm -rf "$1"
    cp -r "$W/mesh" "$1"
    rm -rf "$1"/0 "$1"/[1-9]* "$1"/processor* "$1"/log.* "$1"/postProcessing "$1"/dynamicCode
    cp -r "$1/0.orig" "$1/0"
    N="$2" ARM="$3" python3 - "$1" << 'PY'
import os
import re
import sys
d = sys.argv[1]
arm = os.environ['ARM']
p = os.path.join(d, 'system/controlDict')
s = open(p).read()
for k, v in (('startFrom', 'latestTime'), ('endTime', os.environ['N']), ('writeInterval', '1'), ('writeFormat', 'ascii'),
             ('writePrecision', '15'), ('writeCompression', 'off')):
    if re.search(r'^%s\s+' % k, s, re.M):
        s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
    else:
        s += '\n%-15s %s;\n' % (k, v)
# the tutorial's own scalarTransport object, inlined; abort and sampling dropped
t = open(os.path.join(d, 'system/scalarTransport')).read()
m = re.search(r'\ntracer0\s*\{.*?\n\}', t, re.S)
assert m, 'system/scalarTransport has no tracer0 block'
obj = m.group(0).strip()
if arm == 'T2':
    obj = obj[:-1].rstrip() + '\n    nCorr           1;\n    tolerance       1e-30;\n}'
obj = '\n'.join('    ' + line for line in obj.split('\n'))
s, n = re.subn(r'functions\s*\{.*?\n\}', 'functions\n{\n' + obj + '\n}', s, flags=re.S)
assert n == 1, 'functions'
open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSolution')
s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s)
s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
if arm == 'T1steady':
    p = os.path.join(d, 'system/fvSchemes')
    f = open(p).read()
    f, n = re.subn(r'(ddtSchemes\s*\{\s*default\s+Euler;)', r'\g<1>\n    ddt(rho,tracer0) steadyState;', f)
    assert n == 1, 'ddtSchemes'
    open(p, 'w').write(f)
PY
}

# OpenFOAM develops the field to DEV; both codes restart from its written state
stage "$W/dev" "$DEV" T1
( cd "$W/dev" && rhoSimpleFoam > log 2>&1 )
[ -f "$W/dev/$DEV/tracer0" ] || { echo "SKIP: OpenFOAM did not write tracer0 at $DEV here"; tail -3 "$W/dev/log"; exit 77; }
grep -q nonuniform "$W/dev/$DEV/tracer0" && grep -q nonuniform "$W/dev/$DEV/k" \
    || { echo "FAIL: the restart state is uniform -- this protocol cannot discriminate"; exit 1; }
restart() { stage "$1" "$LAST" "$2"; rm -rf "$1/0"; cp -r "$W/dev/$DEV" "$1/$DEV"; }
ofrun() { ( cd "$1" && rhoSimpleFoam > log 2>&1 ); [ -f "$1/$LAST/tracer0" ] || { echo "FAIL: OpenFOAM in $1"; tail -5 "$1/log"; exit 1; }; }
for a in T1 T2 T1steady; do
    restart "$W/of-$a" "$a"
    ofrun "$W/of-$a"
done

# worst <brae case> <OpenFOAM case> [fields]: the worst field by relative L2 over FIRST..LAST (FIRST defaults to DEV+1),
# "<rel> <field> <it>"; "inf <field> <it> (missing)" when either code did not write it
worst()
{
    FIRST="${FIRST:-$((DEV + 1))}" LAST="$LAST" FIELDS="${3:-U p T k epsilon nut tracer0}" python3 - "$1" "$2" << 'PY'
import math
import os
import re
import sys
b, o = sys.argv[1:3]
first, last = int(os.environ['FIRST']), int(os.environ['LAST'])
def rd(path):
    s = open(path).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if m:
        return [float(x) for x in m.group(1).replace('(', ' ').replace(')', ' ').split()], 'n'
    m = re.search(r'internalField\s+uniform\s+\(?([^;)]+)\)?;', s)
    return ([float(x) for x in m.group(1).split()], 'u') if m else (None, None)
best = (-1.0, 'none', 0)
for it in range(first, last + 1):
    for f in os.environ['FIELDS'].split():
        pb, po = '%s/%d/%s' % (b, it, f), '%s/%d/%s' % (o, it, f)
        if not (os.path.exists(pb) and os.path.exists(po)):
            print('inf %s %d (missing)' % (f, it))
            sys.exit(0)
        a, ka = rd(pb)
        c, kc = rd(po)
        if kc == 'u':
            c = c * (len(a) // len(c))
        if ka == 'u':
            a = a * (len(c) // len(a))
        den = math.sqrt(sum(y * y for y in c))
        if len(a) != len(c) or den == 0.0:
            print('inf %s %d (length or zero oracle)' % (f, it))
            sys.exit(0)
        e = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, c))) / den
        if e > best[0]:
            best = (e, f, it)
print('%.3e %s %d' % best)
PY
}
le() { python3 -c "import sys; sys.exit(0 if float('$1') <= float('$2') else 1)"; }
gt() { python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }
brrun()   # brrun <case> <arm: 1 | cuda> [VAR=value ...]
{
    local d=$1 mode=$2
    shift 2
    ( cd "$d" && env BRAE_RHOSIMPLEFOAM_MIRROR="$mode" "$@" "$BRAE" -case . > log 2>&1 )
    [ -f "$d/$LAST/tracer0" ] && return 0
    echo "     brae ($mode) in $d did not reach $LAST"
    grep -v '^brae NOTICE' "$d/log" | tail -3
    return 1
}

for a in T1 T2; do
    for mode in 1 cuda; do
        lab=$([ $mode = 1 ] && echo host || echo CUDA)
        restart "$W/br-$a-$lab" "$a"
        brrun "$W/br-$a-$lab" "$mode"
        w=$(worst "$W/br-$a-$lab" "$W/of-$a")
        wt=$(worst "$W/br-$a-$lab" "$W/of-$a" tracer0)
        if [ "$BOUND" = report ]; then
            say "$a $lab: worst field $w; tracer0 $wt" INFO
        else
            le "${w%% *}" "$BOUND" && say "$a $lab: every field at $((DEV + 1))-$LAST within $BOUND (worst $w)" PASS \
                                   || say "$a $lab: every field at $((DEV + 1))-$LAST within $BOUND (worst $w)" FAIL
        fi
    done
done

# PREMISES
w=$(worst "$W/of-T1steady" "$W/of-T1" tracer0)
gt "${w%% *}" "$PREMISE" && say "premise: OpenFOAM's tracer0 under ddt(rho,tracer0) steadyState differs from Euler ($w)" PASS \
                         || say "premise: OpenFOAM's tracer0 under ddt(rho,tracer0) steadyState differs from Euler ($w)" FAIL
w=$(worst "$W/of-T2" "$W/of-T1" tracer0)
gt "${w%% *}" "$PREMISE" && say "premise: OpenFOAM's tracer0 with nCorr 1 differs from nCorr 0 ($w)" PASS \
                         || say "premise: OpenFOAM's tracer0 with nCorr 1 differs from nCorr 0 ($w)" FAIL

# CONTROLS
CB=$BOUND
[ "$CB" = report ] && CB=1e-9
# each control on tracer0 over the iterations it is judged at: the write control from DEV+2, where both codes write
# the field (its red is the previous iteration's solve, not the missing first directory)
for c in T1:BRAE_CONTROL_TRACER_DDT_IGNORED:$((DEV + 1)) T2:BRAE_CONTROL_TRACER_NCORR_AS_SOLVES:$((DEV + 1)) \
         T1:BRAE_CONTROL_TRACER_WRITE_BEFORE_SOLVE:$((DEV + 2)) T1:BRAE_CONTROL_TRACER_GRAD_GAUSS:$((DEV + 1)); do
    a=${c%%:*}
    v=${c#*:}
    first=${v#*:}
    v=${v%%:*}
    for mode in 1 cuda; do
        lab=$([ $mode = 1 ] && echo host || echo CUDA)
        cd_=$W/ctl-$a-$lab-${v##*_}
        restart "$cd_" "$a"
        # a control run that did not finish proves nothing: it is a FAIL, not a red
        brrun "$cd_" "$mode" "$v=1" || { say "control $v=1: $a $lab did not run" FAIL; continue; }
        w=$(FIRST=$first worst "$cd_" "$W/of-$a" tracer0)
        case "$w" in
            *missing*) say "control $v=1: $a $lab wrote no tracer0 to judge ($w)" FAIL; continue ;;
        esac
        gt "${w%% *}" "$CB" && say "control $v=1: $a $lab goes red ($w)" PASS \
                            || say "control $v=1: $a $lab goes red ($w)" FAIL
    done
done

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
