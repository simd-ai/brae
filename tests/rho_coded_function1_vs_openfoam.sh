#!/usr/bin/env bash
# OpenFOAM's `type coded;` Function1 -- squareBendLiq's inlet massFlowRate -- on both rhoSimpleFoam
# mirror arms, against real OpenFOAM v2412 compiling the same code. Iteration by iteration.
#
# WHAT RUNS VERBATIM: the tutorial's inlet, `massFlowRate { type coded; name liquidIn; code #{ static bool
# reported(false); if (!reported) { Info<< "Using coded value for massFlowRate" << nl; reported = true; }
# return 5; #}; }`. OpenFOAM compiles it with wmake and dlopens it; brae compiles it with the host C++
# compiler against its Foam shim and dlopens it (codedFunction1.cuh). What is REPLACED here: the T walls'
# `expression` PatchFunction1 overlay, by the `fixedValue uniform 350` the file states first -- a different
# feature with its own gate (tests/rho_patch_expression_vs_openfoam.sh runs the unmodified tutorial on both
# arms); replacing it keeps this gate about the coded inlet alone.
#
# WHY A SECOND SNIPPET. The tutorial's body returns 5 for every x, which is numerically identical to
# `constant 5`: a brae that froze the time, or evaluated the code once, or never fed x at all would pass
# it. The discriminating arm runs `return 5*(1 + 0.05*x);` -- OpenFOAM and brae both -- so the inlet mass
# flow is 5.25, 5.5, 5.75 kg/s at iterations 1-3 (and 5 at construction), and the gate asserts that on
# the written phi as well as on the fields. Under kOmegaSST too, because what that arm found lives in
# both closures:
#
#   THE DEVICE k ASSEMBLY READ k's PATCH VALUES LIVE, where the host reference reads the stored ones: the
#   host closures refresh the k inlet with updateTurbulentInlet (setRefValues only) and the flux switch,
#   and their gradients read k.boundary[pi]->value(), the previous solve's evaluate (kEpsilon_cpp.cu:620-
#   679, kOmegaSST_cpp.cu:657-726). The device rebuilt that value for epsilon and omega (stage H3.5) and
#   not for k -- invisible while the inlet velocity never changes. With the rate above, CUDA arm,
#   kEpsilon: k 3.2e-08 off OpenFOAM at iteration 2 in the inlet-layer corner cells, 1.2e-04 by
#   iteration 3; the host 4.7e-13. kEpsilon.cu and kOmegaSST.cu now rebuild k's the same way.
#
# ALSO ASSERTED: each side prints the snippet's own line exactly once (a function-local static in a loaded
# library, OpenFOAM's and brae's); brae's written U carries the coded dictionary verbatim and OpenFOAM,
# restarted from brae's written state, recompiles it and runs; and the refusals below fire by name while
# the verbatim tutorial arm -- their control -- runs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ARMS=${ARMS:-"1 cuda"}
ITERS=3

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/squareBendLiq"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
LINE="Using coded value for massFlowRate"

cp -r "$SRC" "$W/mesh" || exit 1
( cd "$W/mesh" && rm -rf 0 [1-9]* processor* log.* dynamicCode && cp -r 0.orig 0 \
    && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 ) \
    || { echo "FAIL: blockMesh/topoSet did not run"; exit 1; }

# stage <dir> <variant> -- variants: tutorial | xdep | xdepSST | the refusal ones (see below).
stage() {
    rm -rf "$1"; cp -r "$W/mesh" "$1"; rm -rf "$1"/0 "$1"/log.* "$1"/dynamicCode
    ITERS="$ITERS" VARIANT="$2" python3 - "$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]; n = os.environ['ITERS']; v = os.environ['VARIANT']
def edit(rel, fn):
    p = os.path.join(d, rel); s = open(p).read(); s2 = fn(s)
    if s2 == s: raise SystemExit('stage: no change made to ' + rel + ' (' + v + ')')
    open(p, 'w').write(s2)
def fixT(s):
    i = s.index('"(?i).*walls"'); j = s.index('#};', i)
    j = s.index('}', j + 3) + 1          # uniformValue
    j = s.index('}', j) + 1              # the walls entry
    return s[:i] + '"(?i).*walls"\n    {\n        type            fixedValue;\n        value           uniform 350;\n    }' + s[j:]
edit('0.orig/T', fixT)
def fixSol(s):
    s = re.sub(r'solvers\s*\{.*?\n\}', 'solvers\n{\n    p { solver PBiCGStab; preconditioner DIC; '
               'tolerance 1e-14; relTol 0; }\n    "(U|e|k|epsilon|omega)" { solver PBiCGStab; '
               'preconditioner DILU; tolerance 1e-14; relTol 0; }\n}', s, count=1, flags=re.S)
    s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
    return re.sub(r'(epsilon\s+0\.7;)', r'\1\n        omega           0.7;', s)
edit('system/fvSolution', fixSol)
def fixCtl(s):
    s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
    for k, val in (('startFrom', 'latestTime'), ('endTime', n), ('writeInterval', '1'),
                   ('writeFormat', 'ascii'), ('writePrecision', '15')):
        s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, val), s, flags=re.M)
    return s
edit('system/controlDict', fixCtl)
RET = '                return 5;'
code = {
    'xdep':        '                return 5*(1 + 0.05*x);',
    'xdepSST':     '                return 5*(1 + 0.05*x);',
    'refThis':     '                return this->time().value();',
    'refDollar':   '                return $rhoInlet;',
}
if v in code:
    edit('0.orig/U', lambda s: s.replace(RET, code[v]))
if v == 'refInclude':
    edit('0.orig/U', lambda s: s.replace('            name  liquidIn;',
                                         '            name  liquidIn;\n            codeInclude #{ #include "fvCFD.H" #};'))
if v == 'refEmpty':
    edit('0.orig/U', lambda s: re.sub(r'code\s*#\{.*?#\};', 'code #{ #};', s, count=1, flags=re.S))
if v == 'refTable':
    edit('0.orig/U', lambda s: re.sub(r'massFlowRate\s*\{.*?#\};\s*\}', 'massFlowRate    table ((0 5) (10 6));', s,
                                      count=1, flags=re.S))
if v == 'refTableDict':
    edit('0.orig/U', lambda s: re.sub(r'massFlowRate\s*\{.*?#\};\s*\}',
                                      'massFlowRate\n        {\n            type table;\n            values ((0 5) (10 6));\n        }',
                                      s, count=1, flags=re.S))
if v == 'xdepSST':
    edit('constant/turbulenceProperties', lambda s: re.sub(r'RASModel\s+\S+;', 'RASModel            kOmegaSST;', s))
    edit('system/fvSchemes', lambda s: s.replace('    div(phi,k)          $turbulence;',
                                                 '    div(phi,k)          $turbulence;\n    div(phi,omega)      $turbulence;')
                                        .rstrip() + '\n\nwallDist\n{\n    method meshWave;\n}\n')
    open(os.path.join(d, '0.orig/omega'), 'w').write(
        'FoamFile { version 2.0; format ascii; class volScalarField; object omega; }\n'
        'dimensions [0 0 -1 0 0 0 0];\ninternalField uniform 2222.2;\nboundaryField\n{\n'
        '    inlet { type turbulentMixingLengthFrequencyInlet; mixingLength 0.005; value $internalField; }\n'
        '    outlet { type inletOutlet; inletValue $internalField; value $internalField; }\n'
        '    "(?i).*walls" { type omegaWallFunction; value $internalField; }\n}\n')
PYEOF
    cp -r "$1/0.orig" "$1/0"
}

# compare <brae> <of> <label> <second> <xdep>
compare() {
    LABEL="$3" SECOND="$4" XDEP="$5" ITERS="$ITERS" python3 - "$1" "$2" <<'PYEOF'
import math, os, re, sys
brae, of = sys.argv[1], sys.argv[2]
label, second, xdep, n = os.environ['LABEL'], os.environ['SECOND'], os.environ['XDEP'] == '1', int(os.environ['ITERS'])
def nums(t): return [float(x) for x in re.findall(r'-?[\d.]+(?:[eE][-+]?\d+)?', t)]
def internal(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    return nums(m.group(1))
def inletFlux(p):
    s = open(p).read(); bf = s[s.index('boundaryField'):]
    body = re.search(r'\n    inlet\s*\{(.*?)\n    \}', bf, re.S).group(1)
    v = re.search(r'value\s+nonuniform\s+List<scalar>\s*\d+\s*\((.*?)\)\s*;', body, re.S)
    return sum(nums(v.group(1)))
def rel(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b)); den = sum(y * y for y in b)
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)
bad = 0
for it in range(1, n + 1):
    for f, bound in (('U', 1e-10), ('p', 1e-10), ('T', 1e-11), ('k', 1e-10), (second, 1e-10), ('nut', 1e-10)):
        e = rel(internal('%s/%d/%s' % (brae, it, f)), internal('%s/%d/%s' % (of, it, f)))
        ok = e < bound
        print('     %s it %d  %-8s vs OpenFOAM (L2 rel)  %.6e   %s' % (label, it, f, e, 'ok' if ok else 'FAIL (bound %g)' % bound))
        bad |= not ok
    # The inlet mass flow the Function1 prescribes at THIS iteration's time, on the written phi --
    # OpenFOAM's and brae's. The x-independent tutorial body gives 5 at every t.
    mdot = 5 * (1 + 0.05 * it) if xdep else 5.0
    fb, fo = inletFlux('%s/%d/phi' % (brae, it)), inletFlux('%s/%d/phi' % (of, it))
    ok = abs(fb + mdot) < 1e-12 * mdot and abs(fo + mdot) < 1e-12 * mdot
    print('     %s it %d  inlet mass flux  brae %.15g  OpenFOAM %.15g  expected -%.15g   %s'
          % (label, it, fb, fo, mdot, 'ok' if ok else 'FAIL')); bad |= not ok
sys.exit(1 if bad else 0)
PYEOF
}

printedOnce() {   # printedOnce <log> <who>
    local c; c=$(grep -c "$LINE" "$1")
    if [ "$c" = 1 ]; then echo "     $2 printed the snippet's own line exactly once                      ok"
    else echo "     $2 printed the snippet's own line $c times, not once                     FAIL"; fail=1; fi
}

runBrae() {   # runBrae <dir> <mirror>
    BRAE_U_SOLVER=ofOrder BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BRAE" -case "$1" > "$1/log.brae" 2>&1
}

for V in tutorial xdep xdepSST; do
    second=$([ "$V" = xdepSST ] && echo omega || echo epsilon)
    xd=$([ "$V" = tutorial ] && echo 0 || echo 1)
    stage "$W/of_$V" "$V" || { echo "FAIL: $V -- staging"; fail=1; continue; }
    ( cd "$W/of_$V" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )
    if ! grep -q "$LINE" "$W/of_$V/log.rhoSimpleFoam"; then
        if grep -q "Failed wmake\|dynamicCode\|FOAM FATAL" "$W/of_$V/log.rhoSimpleFoam"; then
            echo "SKIP: OpenFOAM cannot compile its dynamic code here, so a coded Function1 has no oracle:"
            grep -m3 "Failed\|FATAL\|wmake" "$W/of_$V/log.rhoSimpleFoam"
            exit 77
        fi
        echo "FAIL: $V -- OpenFOAM ran without the coded path"; fail=1; continue
    fi
    echo "== squareBendLiq, coded massFlowRate, $V -- OpenFOAM =="
    printedOnce "$W/of_$V/log.rhoSimpleFoam" "OpenFOAM"
    for MIRROR in $ARMS; do
        BR="$W/br_${V}_$MIRROR"
        stage "$BR" "$V" || { fail=1; continue; }
        runBrae "$BR" "$MIRROR" || { echo "FAIL: $V ($MIRROR) -- brae did not run"; grep -v '^brae NOTICE' "$BR/log.brae" | tail -8; fail=1; continue; }
        echo "== squareBendLiq, coded massFlowRate, $V -- $([ "$MIRROR" = cuda ] && echo 'CUDA arm' || echo 'host arm') =="
        printedOnce "$BR/log.brae" "brae"
        compare "$BR" "$W/of_$V" "$V" "$second" "$xd" || fail=1
    done
done

# ---- the writer round trip: brae's written U carries the coded dictionary, and OpenFOAM runs from it --
RT="$W/br_tutorial_1"
if [ -d "$RT/$ITERS" ]; then
    echo "== round trip: brae's written state -> OpenFOAM =="
    if grep -q "type            coded;" "$RT/$ITERS/U" && grep -q "Info<< \"$LINE\" << nl;" "$RT/$ITERS/U"; then
        echo "     brae's $ITERS/U carries the coded dictionary, the code verbatim                   ok"
    else
        echo "     brae's $ITERS/U lost the coded dictionary                                        FAIL"; fail=1
    fi
    ( cd "$RT" && rm -rf dynamicCode && for t in [0-9]*; do [ "$t" = "$ITERS" ] || rm -rf "$t"; done \
        && sed -i "s/^endTime .*/endTime         $((ITERS + 1));/" system/controlDict \
        && rhoSimpleFoam > log.of_restart 2>&1 )
    if [ $? -eq 0 ] && grep -q "$LINE" "$RT/log.of_restart"; then
        echo "     OpenFOAM restarted from brae's written state, recompiled the code, ran     ok"
    else
        echo "     OpenFOAM could not restart from brae's written state                        FAIL"
        tail -8 "$RT/log.of_restart"; fail=1
    fi
fi

# ---- refusals, each by name; the verbatim tutorial arm above is their control -------------------------
refuse() {   # refuse <variant> <mirror> <expected text> <what>
    local d="$W/ref_$1"
    stage "$d" "$1" || { fail=1; return; }
    if runBrae "$d" "$2"; then
        echo "     REFUSAL $4: brae RAN                                                             FAIL"; fail=1
    elif ! grep -q -- "$3" "$d/log.brae"; then
        echo "     REFUSAL $4: refused, but not by name                                             FAIL"
        grep -v '^brae NOTICE' "$d/log.brae" | tail -4; fail=1
    elif ls -d "$d"/[1-9]* > /dev/null 2>&1; then
        echo "     REFUSAL $4: refused, but a time directory was written                            FAIL"; fail=1
    else
        echo "     REFUSAL $4                                                                       ok"
    fi
}
echo "== refusals =="
refuse refInclude   1 "carries \`codeInclude\`"                 "codeInclude is not compiled in"
refuse refEmpty     1 "has no \`code\`"                         "empty code, as OpenFOAM"
refuse refThis      1 "did not compile"                         "this->time(): not in brae's scope"
refuse refDollar    1 "uses \`\$\` in its code"                 "\$ expansion inside code"
refuse refTable     1 "starts \`table\`"                        "inline table Function1"
refuse refTableDict 1 "is a \`table\` Function1"                "table Function1 dictionary"

# ---- every driver that takes the rate as ONE number refuses a coded one ------------------------------
# flowRateValue() is the one place they all read it (fv_patch_field.cuh), and it refuses a rate that
# depends on time rather than hand over a number the driver would then freeze. Shown on each entry point,
# on the gas fixtures those drivers run, with the fixture's own constant rate turned into a coded one.
BUILDD="$(dirname "$BRAE")"
legacyRefuses() {   # legacyRefuses <fixture> <label> <env> <binary>
    local d; d="$W/leg_$(printf '%s' "$2" | tr -c 'A-Za-z0-9' '_')"
    rm -rf "$d"; cp -r "$ROOT/validation/$1" "$d"; rm -rf "$d"/0; cp -r "$d/0.orig" "$d/0"
    [ -f "$d/constant/polyMesh/points" ] || ( cd "$d" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "     LEGACY $2: blockMesh did not run                             FAIL"; fail=1; return; }
    sed -i -E 's/(massFlowRate|volumetricFlowRate) constant ([0-9.eE+-]+);/\1 { type coded; name legacyRate; code #{ return \2; #}; }/' "$d/0/U"
    if ! grep -q "type coded;" "$d/0/U"; then echo "     LEGACY $2: staging found no constant rate          FAIL"; fail=1; return; fi
    if [ ! -x "$BUILDD/$4" ]; then echo "     LEGACY $2: $4 is not built                               FAIL"; fail=1; return; fi
    if ( cd "$d" && env $3 "$BUILDD/$4" -case "$d" > log 2>&1 ); then
        echo "     LEGACY $2: ran a coded flow rate as one number                                   FAIL"; fail=1
    elif ! grep -q "takes the rate as ONE number" "$d/log"; then
        echo "     LEGACY $2: refused, but not by the flow-rate choke point                        FAIL"
        grep -v '^brae NOTICE' "$d/log" | tail -3; fail=1
    else
        echo "     LEGACY $2: refuses the coded rate by name                                        ok"
    fi
}
legacyRefuses rhoFR  "rhoSimpleFoam, pre-mirror body" "BRAE_RHOSIMPLEFOAM_MIRROR=" brae_rhoSimpleFoam
legacyRefuses rhoFR  "rhoSimpleFoam_legacy"           "BRAE_RHOSIMPLEFOAM_MIRROR=" brae_rhoSimpleFoam_legacy
legacyRefuses rhoFR  "rhoSimpleFoam_slice"            "BRAE_RHOSIMPLEFOAM_MIRROR=" brae_rhoSimpleFoam_slice
legacyRefuses incFR  "simpleFoam"                     "BRAE_SIMPLEFOAM_V2="        brae
legacyRefuses incFR  "simpleFoam V2"                  "BRAE_SIMPLEFOAM_V2=1"       brae

[ "$fail" -eq 0 ] && echo "PASSED" || echo "FAILED"
exit $fail
