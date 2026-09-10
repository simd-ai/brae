#!/usr/bin/env bash
# THE SIX rhoSimpleFoam TUTORIALS OpenFOAM v2412 ships, as shipped, on BOTH OF-mirror arms -- the
# standing answer to "which of them does brae run, and which does it refuse". Meshed by each tutorial's
# OWN Allrun.pre (or blockMesh where it has none), so what runs here is the case OpenFOAM distributes and
# not a fixture trimmed to suit.
#
# THE TWO HALVES ARE EACH OTHER'S CONTROL, which is the whole point of running all six in one gate:
#   * four RUN, and are held against real OpenFOAM iteration by iteration. A brae that started refusing
#     them -- a new guard drawn too wide -- turns this red.
#   * two REFUSE, and must refuse BY NAME. A brae that started RUNNING them would be running a case whose
#     scheme or boundary condition it does not implement, silently substituting something else, and that
#     turns this red too. Without the running half, a brae that refused everything would pass the refusal
#     half; without the refusal half, a brae that ran everything regardless would pass the running half.
#
# BOUNDS, all measured, per case, and each the worst field over iterations 1-3 unless noted:
#   aerofoilNACA0012          host 2.8e-12   CUDA 2.8e-12
#   angledDuctExplicitFixed   host 1.6e-10   CUDA 1.2e-10
#   squareBend                host 5.5e-10   CUDA 6.6e-11
#   squareBendLiqNoNewtonian  ITERATION 1 ONLY, host 5.8e-13, CUDA 5.5e-13
# tests/rho_aerofoil_vs_openfoam.sh is the deep gate on that tutorial and carries the three device defects
# it found and closed; this one asserts the coarser property -- it still runs, and still agrees.
#
# WHY squareBendLiqNoNewtonian IS ITERATION 1 ONLY. The case starts from rest, the model's constructor
# puts nu_ at nuMax over the quiescent field, and the strain rate in the resulting plug is the round-off
# in U: OpenFOAM against ITSELF with only the p solver swapped reads nu_ 6.21e-02 and U 1.04e-03 at
# iteration 2 (tests/rho_generalized_newtonian_vs_openfoam.sh carries that measurement). No bound at
# iteration 2 can separate brae from OpenFOAM's own floor there, so this gate does not pretend to.
#
# WHAT THIS GATE DOES NOT CLAIM. It runs three iterations, not to convergence -- it is a breadth gate, and
# the per-component depth is in the dedicated gates (rho_aerofoil, rho_squarebendliq, rho_generalized_
# newtonian, rho_kepsilon, rho_komegasst). Both codes' linear solvers are tightened to 1e-14 relTol 0
# because at a tutorial's own tolerance the two stop their solves at different points and drift on solver
# noise rather than on the discretisation. Nothing else is touched: schemes, boundary conditions, thermo,
# fvOptions and function objects are the tutorial's own.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=3

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
TUT="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam"
[ -d "$TUT" ] || { echo "SKIP: rhoSimpleFoam tutorials not found"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
SUMMARY="$W/summary"; : > "$SUMMARY"

note() { printf '%-28s %-6s %s\n' "$1" "$2" "$3" >> "$SUMMARY"; }

# mesh <tutorial-relative-path> <name>. Each tutorial's own pre-step; injectorPipe snappies in parallel
# and is reconstructed, which is what makes it a serial case brae can read.
mesh() {
    local src="$TUT/$1" d="$W/mesh_$2"
    [ -d "$src" ] || return 1
    rm -rf "$d"
    cp -r "$src" "$d" || return 1
    if [ -x "$d/Allrun.pre" ]
    then
        ( cd "$d" && ./Allrun.pre > log.pre 2>&1 )
    else
        ( cd "$d" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 )
    fi
    # the tutorial's own topoSet, where its Allrun runs one after Allrun.pre
    [ -f "$d/system/topoSetDict" ] && ( cd "$d" && topoSet > log.topoSet 2>&1 )
    # snappyHexMesh runs in parallel here and writes ONLY processor*/constant/polyMesh; the serial
    # constant/polyMesh left behind is blockMesh's BACKGROUND mesh, which is a different mesh with a
    # defaultFaces patch no 0.orig covers. So the reconstruction is keyed on the processor directories
    # existing, never on the serial mesh being absent -- keyed the other way this gate silently ran the
    # 1024-cell background mesh and reported a boundaryField error instead of the tutorial's real refusal.
    if [ -d "$d/processor0/constant/polyMesh" ]
    then
        ( cd "$d" && reconstructParMesh -constant > log.reconstructParMesh 2>&1 && rm -rf processor* )
    fi
    [ -f "$d/constant/polyMesh/owner" ]
}

stage() {   # stage <meshed> <dest>
    rm -rf "$2"
    cp -r "$1" "$2"
    rm -rf "$2"/0 "$2"/[1-9]* "$2"/processor* "$2"/log.* "$2"/postProcessing "$2"/dynamicCode
    cp -r "$2/0.orig" "$2/0"
    ITERS="$ITERS" python3 - "$2" <<'PYEOF'
import os, re, sys
d, n = sys.argv[1], os.environ['ITERS']
p = os.path.join(d, 'system/controlDict')
s = open(p).read()
for k, v in (('startFrom', 'latestTime'), ('endTime', n), ('writeInterval', '1'),
             ('writeFormat', 'ascii'), ('writePrecision', '15'), ('writeCompression', 'off')):
    if re.search(r'^%s\s+' % k, s, re.M):
        s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
    else:
        s += '\n%-15s %s;\n' % (k, v)
open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSolution')
s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s)
s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
PYEOF
}

# compare <braeDir> <ofDir> <label> <bound> <lastIteration> [<iteration-1 bound>]
compare() {
    LABEL="$3" BOUND="$4" LAST="$5" BOUND1="${6:-$4}" python3 - "$1" "$2" <<'PYEOF'
import math, os, re, sys
brae, of = sys.argv[1], sys.argv[2]
label = os.environ['LABEL']
bound, bound1, last = float(os.environ['BOUND']), float(os.environ['BOUND1']), int(os.environ['LAST'])
def internal(p, n=None):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if m:
        return [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', m.group(1))]
    # OpenFOAM writes a field that never moved as `uniform` -- squareBendLiqNoNewtonian's p at
    # iteration 1 is one. Expanded to the other side's length so the two are comparable.
    m = re.search(r'internalField\s+uniform\s+([^;]+);', s)
    v = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', m.group(1))]
    return v * (n // len(v)) if n and len(v) and n % len(v) == 0 else v
def rel(a, b):
    den = sum(y * y for y in b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / den) if den > 0 else 0.0
bad, worst, worstname = 0, 0.0, ''
for it in range(1, last + 1):
    for f in ('U', 'p', 'T', 'k', 'epsilon', 'omega', 'nut'):
        pb, po = '%s/%d/%s' % (brae, it, f), '%s/%d/%s' % (of, it, f)
        if not (os.path.exists(pb) and os.path.exists(po)):
            continue
        a = internal(pb)
        b = internal(po, len(a))
        if len(a) != len(b):
            print('     %s it %d  %-7s LENGTH MISMATCH %d vs %d   FAIL' % (label, it, f, len(a), len(b)))
            bad = 1
            continue
        e = rel(a, b)
        lim = bound1 if it == 1 else bound
        if e > worst:
            worst, worstname = e, '%s it %d' % (f, it)
        if e >= lim:
            print('     %s it %d  %-7s vs OpenFOAM (L2 rel)  %.6e   FAIL (bound %g)' % (label, it, f, e, lim))
            bad = 1
print('     %s worst %s %.3e   %s' % (label, worstname, worst, 'FAIL' if bad else 'ok'))
sys.exit(1 if bad else 0)
PYEOF
}

# runs <name> <tutorial path> <host bound> <cuda bound> <lastIter> [<cuda it1 bound>]
runs() {
    local name="$1" path="$2" hb="$3" cb="$4" last="$5" cb1="${6:-$4}"
    echo "== $name =="
    if ! mesh "$path" "$name"
    then
        echo "     SKIP: the tutorial's own meshing did not run here"
        note "$name" "-" "SKIPPED (meshing unavailable here)"
        return 0
    fi
    local of="$W/of_$name"
    stage "$W/mesh_$name" "$of"
    ( cd "$of" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )
    if [ ! -d "$of/$last" ]
    then
        echo "     SKIP: OpenFOAM itself did not reach iteration $last here"
        note "$name" "-" "SKIPPED (no OpenFOAM oracle)"
        return 0
    fi
    local ok=1
    for MIRROR in 1 cuda
    do
        local arm bound d
        arm=$([ "$MIRROR" = cuda ] && echo CUDA || echo host)
        bound=$([ "$MIRROR" = cuda ] && echo "$cb" || echo "$hb")
        d="$W/br_${name}_$MIRROR"
        stage "$W/mesh_$name" "$d"
        if ! BRAE_RHOSIMPLEFOAM_MIRROR=$MIRROR "$BRAE" -case "$d" > "$d/log.brae" 2>&1 || [ ! -d "$d/$last" ]
        then
            echo "     $arm: DID NOT RUN the tutorial"
            grep -v '^brae NOTICE' "$d/log.brae" | tail -3
            fail=1
            ok=0
            continue
        fi
        if [ "$MIRROR" = cuda ]
        then
            compare "$d" "$of" "$arm" "$bound" "$last" "$cb1" || { fail=1; ok=0; }
        else
            compare "$d" "$of" "$arm" "$bound" "$last" || { fail=1; ok=0; }
        fi
    done
    [ "$ok" = 1 ] && note "$name" "RUNS" "both arms, vs OpenFOAM to iteration $last" \
                  || note "$name" "BROKE" "see above"
}

# refuses <name> <tutorial path> <expected text> <what it needs>
refuses() {
    local name="$1" path="$2" want="$3" needs="$4"
    echo "== $name (expected to refuse) =="
    if ! mesh "$path" "$name"
    then
        echo "     SKIP: the tutorial's own meshing did not run here"
        note "$name" "-" "SKIPPED (meshing unavailable here)"
        return 0
    fi
    local ok=1
    for MIRROR in 1 cuda
    do
        local arm d
        arm=$([ "$MIRROR" = cuda ] && echo CUDA || echo host)
        d="$W/br_${name}_$MIRROR"
        stage "$W/mesh_$name" "$d"
        if BRAE_RHOSIMPLEFOAM_MIRROR=$MIRROR "$BRAE" -case "$d" > "$d/log.brae" 2>&1
        then
            echo "     $arm: RAN a tutorial it does not implement                                  FAIL"
            fail=1
            ok=0
        elif ! grep -q -- "$want" "$d/log.brae"
        then
            echo "     $arm: refused, but not by name                                              FAIL"
            grep -v '^brae NOTICE' "$d/log.brae" | tail -3
            fail=1
            ok=0
        elif ls -d "$d"/[1-9]* > /dev/null 2>&1
        then
            echo "     $arm: refused, but a time directory was written                             FAIL"
            fail=1
            ok=0
        else
            echo "     $arm: refuses by name                                                       ok"
        fi
    done
    [ "$ok" = 1 ] && note "$name" "REFUSES" "$needs" || note "$name" "BROKE" "see above"
}

runs aerofoilNACA0012        aerofoilNACA0012             1e-11 1e-11 "$ITERS"
runs angledDuctExplicitFixedCoeff angledDuctExplicitFixedCoeff 1e-09 1e-09 "$ITERS"
runs squareBend              squareBend                   5e-09 5e-09 "$ITERS"
runs squareBendLiqNoNewtonian squareBendLiqNoNewtonian     1e-11 1e-11 1
refuses squareBendLiq        squareBendLiq \
        "uniformFixedValue with a non-constant uniformValue" \
        "the expression PatchFunction1 on its T walls"
refuses injectorPipe         gasMixing/injectorPipe \
        'div(phi,e) is `Gauss limitedLinear`' \
        "leastSquares reached by default: div(phi,e)'s limiter, then grad(k)/grad(epsilon)"

echo
echo "==================== rhoSimpleFoam tutorials, OpenFOAM v2412 ===================="
sort "$SUMMARY"
echo "================================================================================"
n_runs=$(grep -c ' RUNS ' "$SUMMARY" || true)
n_ref=$(grep -c ' REFUSES ' "$SUMMARY" || true)
echo "runs on both mirror arms: $n_runs        refuses by name: $n_ref"

[ "$fail" -eq 0 ] && echo "PASSED" || echo "FAILED"
exit $fail
