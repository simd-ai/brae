#!/usr/bin/env bash
# The transient driver stops on the turbulence fields, not on p and U alone (gpuPimpleFoam.cu, "THE TURBULENCE FIELDS'
# HEALTH"). Every step, before its write, each turbulence solve's residuals finite (solution_health.cuh); every tenth step
# and before every write, sum|k| + sum|second field| finite and not 1e12 times its first value for 25 checks in a row
# (turb_blowup.cuh).
#
# THE CASES: the committed pitzDaily fixture made transient as tests/pimple_run.sh makes it (Euler, written every step).
#   healthy    kEpsilon at deltaT 2e-5, 15 steps: runs to the end and writes every step
#   natural    kEpsilon at deltaT 1e-3, PIMPLE 1 x 1, upwind turbulence: a Courant number in the hundreds by the third step,
#              and the epsilon solve goes non-finite at the tenth (OpenFOAM's log: `Solving for epsilon ... Final residual =
#              nan` at Time = 0.01). Before this check brae read only p's and U's residuals, which were still finite at that
#              step, and WROTE 0.01 with U and p NaN in every cell and k and epsilon at their floors (Foam::bound's max()
#              turns a NaN into the floor). It must stop at that step by name and write nothing non-finite.
#   injected   one cell of k NaN, of epsilon Inf (kEpsilon) and of omega Inf (kOmegaSST) after step 3
#              (BRAE_TEST_CORRUPT): the run stops at step 3, steps 1 and 2 written, 3 not
# CONTROL, asserted: BRAE_ALLOW_NONFINITE=1 on the k NaN arm, the opt-out: the run goes on to the end, so the stop in the
# injected arm is this check's.
# OPENFOAM, when present (OFBASHRC): the natural case run by pimpleFoam, and the step of its first non-finite value
# reported beside brae's stop. On x86 OpenFOAM aborts there (trapFpe 1, sigFpe); on aarch64 (GB10, Grace) the CPU does
# not trap and OpenFOAM writes NaN fields to the end. Reported, not asserted: which step the divergence reaches first is the
# trajectory's, and the two codes' trajectories are not the same to the last bit.
set -u
BIN="${1:?brae_pimpleFoam binary}"
SRC="${2:?committed pitzDaily case dir}"
W="${3:?work dir}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$SRC/constant/polyMesh/points" ] || { echo "SKIP: fixture '$SRC' not present"; exit 77; }
rm -rf "$W"
mkdir -p "$W"
fail=0

stage()   # stage <dir> <model> <deltaT> <steps> <turbulence scheme> <nOuter> <nCorr>
{
    python3 - "$SRC" "$@" <<'PYEOF'
import os, re, shutil, sys
src, d, model, dt, steps, ts, no, nc = sys.argv[1:9]
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
for x in ('constant', 'system', '0'):
    shutil.copytree(os.path.join(src, x), os.path.join(d, x))
for f in os.listdir(d + '/constant/polyMesh'):
    if f.startswith('.brae_'):
        os.remove(d + '/constant/polyMesh/' + f)
end = '%.10g' % (int(steps) * float(dt))
open(d + '/system/controlDict', 'w').write('''FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application pimpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime %s;
deltaT %s;
writeControl timeStep;
writeInterval 1;
writePrecision 12;
timePrecision 8;
runTimeModifiable false;
''' % (end, dt))
s = open(d + '/system/fvSchemes').read()
s = re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default Euler;\n}', s, flags=re.S)
s, n = re.subn(r'turbulence\s+bounded Gauss limitedLinear 1;', 'turbulence %s;' % ts, s)
assert n == 1, 'the fixture no longer names its turbulence scheme through $turbulence'
open(d + '/system/fvSchemes', 'w').write(s)
s = open(d + '/system/fvSolution').read()
def cut(text, key):
    m = re.search(r'\n%s\s*\{' % key, text)
    if not m:
        return text
    i, depth = m.end() - 1, 0
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[:m.start()] + text[i + 1:]
s = cut(cut(cut(s, 'SIMPLE'), 'PIMPLE'), 'relaxationFactors')
s += '\nPIMPLE\n{\n    nOuterCorrectors %s;\n    nCorrectors %s;\n    nNonOrthogonalCorrectors 0;\n}\n' % (no, nc)
# OpenFOAM's final corrector selects <field>Final: the same settings under that name
s = s.replace('"(U|k|epsilon|omega|f|v2)"', '"(U|k|epsilon|omega|f|v2)(|Final)"').replace('\n    p\n', '\n    "p(|Final)"\n')
open(d + '/system/fvSolution', 'w').write(s)
t = open(d + '/constant/turbulenceProperties').read()
t, n = re.subn(r'RASModel\s+\w+;', 'RASModel %s;' % model, t)
assert n == 1
open(d + '/constant/turbulenceProperties', 'w').write(t)
PYEOF
}

times()   # times <dir>: the written time directories after 0, in order
{
    (cd "$1" && ls -d [0-9]* 2>/dev/null | grep -v '^0$' | sort -g | tr '\n' ' ' | sed 's/ $//')
}

nonfinite()   # nonfinite <dir>: written files holding nan or inf
{
    grep -rliE '(^|[ (])(-?nan|-?inf)([ );]|$)' "$1"/[0-9]* 2>/dev/null | grep -v '/0/' | tr '\n' ' '
}

# ---- healthy ----
stage "$W/healthy" kEpsilon 2e-5 15 "bounded Gauss limitedLinear 1" 2 2 || { echo "FAIL: staging"; exit 1; }
"$BIN" -case "$W/healthy" > "$W/healthy.log" 2>&1
rc=$?
n=$(times "$W/healthy" | wc -w)
if [ "$rc" -eq 0 ] && [ "$n" -eq 15 ]; then
    echo "ok:   healthy -- exit 0, 15 steps written"
else
    echo "FAIL: healthy -- exit $rc, $n of 15 steps written"
    grep -m3 error "$W/healthy.log"
    fail=1
fi

# ---- natural divergence ----
stage "$W/natural" kEpsilon 1e-3 20 "bounded Gauss upwind" 1 1 || { echo "FAIL: staging"; exit 1; }
cp -r "$W/natural" "$W/natural_of"
"$BIN" -case "$W/natural" > "$W/natural.log" 2>&1
rc=$?
bad=$(nonfinite "$W/natural")
stop=$(grep -oE 'solution diverged: .*' "$W/natural.log" | head -1)
if [ "$rc" -ne 0 ] && [ -n "$stop" ] && [ -z "$bad" ]; then
    echo "ok:   natural -- stopped: ${stop:0:150}"
    echo "      last written: $(times "$W/natural" | awk '{print $NF}'), no non-finite value written"
else
    echo "FAIL: natural -- exit $rc, stop '${stop:0:120}', non-finite in: ${bad:-none}"
    fail=1
fi
if [ -f "$OFBASHRC" ]; then
    (
        set +u
        # shellcheck disable=SC1091
        source "$OFBASHRC" > /dev/null 2>&1 || true
        cd "$W/natural_of" && nice -n 10 pimpleFoam > log 2>&1
        echo "rc=$?" > rc
    )
    first=$(awk '/^Time = /{t=$3} /[Nn]a[Nn]|inf/ && !f && t != "" {print t; f=1}' "$W/natural_of/log")
    aborted=$(grep -m1 -oE 'sigFpe|Floating point exception \(core dumped\)|FOAM FATAL' "$W/natural_of/log")
    echo "      OpenFOAM: $(cat "$W/natural_of/rc"), first non-finite value in its log at Time = ${first:-none}," \
         "${aborted:+aborted ($aborted)}${aborted:-ran on to the end (no FPE trap on this CPU)}"
fi

# ---- injected ----
inject()   # inject <name> <model> <BRAE_TEST_CORRUPT> <field named in the message>
{
    stage "$W/$1" "$2" 2e-5 6 "bounded Gauss limitedLinear 1" 2 2 || { echo "FAIL: staging"; exit 1; }
    env BRAE_TEST_CORRUPT="$3" "$BIN" -case "$W/$1" > "$W/$1.log" 2>&1
    local rc=$?
    local t
    t=$(times "$W/$1")
    if [ "$rc" -ne 0 ] && grep -qF "solution diverged" "$W/$1.log" && grep -qF "time step 3" "$W/$1.log" \
        && [ "$t" = "2e-05 4e-05" ]; then
        echo "ok:   $1 -- stopped at step 3, written: $t"
    else
        echo "FAIL: $1 -- exit $rc, written: '$t'"
        grep -m2 -E 'error|diverged' "$W/$1.log" | cut -c1-200
        fail=1
    fi
}
inject injected_k_nan kEpsilon k:nan:3
inject injected_epsilon_inf kEpsilon epsilon:inf:3
inject injected_omega_inf kOmegaSST omega:inf:3

# ---- control: the opt-out ----
stage "$W/control" kEpsilon 2e-5 6 "bounded Gauss limitedLinear 1" 2 2 || { echo "FAIL: staging"; exit 1; }
env BRAE_TEST_CORRUPT=k:nan:3 BRAE_ALLOW_NONFINITE=1 "$BIN" -case "$W/control" > "$W/control.log" 2>&1
rc=$?
n=$(times "$W/control" | wc -w)
if [ "$rc" -eq 0 ] && [ "$n" -eq 6 ]; then
    echo "ok:   CONTROL BRAE_ALLOW_NONFINITE=1 -- the same injection runs to the end (red, as it must be)"
else
    echo "FAIL: CONTROL -- exit $rc, $n of 6 steps written: the opt-out did not switch the check off"
    fail=1
fi

[ "$fail" -eq 0 ] && echo "PASS: pimpleFoam stops on a non-finite or runaway turbulence field before writing it"
exit "$fail"
