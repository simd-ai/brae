#!/usr/bin/env bash
# The `brae` command line: --version, solvers, capabilities, check, run, and the rule for command words beside case directories.
#
#   brae_cli.sh <brae binary> <work dir> <source root> <build dir>
#
# CHECKS
#   --version, solvers and capabilities answer, and what they list is the compiled tables': every solver of the dispatch table,
#   every boundary-condition type of patchFieldTypeNames (read from the source here), JSON that parses, and every gate a
#   capability names as its validation is a registered test (ctest -N in <build dir>)
#   check on a real case (validation/pitzDailySST, simpleFoam) and on it made transient (pimpleFoam, Euler, PIMPLE 1-2-0): exit 0
#   or 1 and "verdict: SUPPORTED"
#   NEGATIVE CONTROLS, each a copy of the steady case with one setting changed, each must exit 2 and name what it refuses:
#     an application brae has no solver for; a boundary condition the factory has no branch for; a named but never implemented
#     one (externalWallHeatFluxTemperature); div(phi,U) vanLeerV; an snGrad scheme brae does not implement (faceCorrected); a RAS
#     model brae does not implement (v2f); `bounded backward` on pimpleFoam; a PIMPLE control pimpleFoam does not read; and a
#     directory that is no case (exit 3)
#   command words beside case directories: a case directory called run, check, solvers or capabilities still runs as that case
#   on `brae <word>` (and brae says so), while `brae <word> -case ...` / `--json` is the command; `brae run -case D` refuses an
#   unknown application exactly as `brae -case D` does
#   check --setup on the steady case (a GPU): the solver sets the case up and stops before its first iteration
#   -parallel with no multi-GPU layer installed is refused, naming brae-mgpu
set -u
BIN="${1:?brae binary}"
WORK="${2:?work dir}"
SRC="${3:?source root}"
BUILD="${4:?build dir}"
fail=0
pass=0
ok() { echo "ok:   $1"; pass=$((pass + 1)); }
bad() { echo "FAIL: $1 -- $2"; fail=$((fail + 1)); }
rm -rf "$WORK"
mkdir -p "$WORK"

# ---- --version, solvers, capabilities ----------------------------------------------------------------------------------------
"$BIN" --version > "$WORK/version.log" 2>&1
rc=$?
if [ $rc = 0 ] && grep -q "^brae [0-9]" "$WORK/version.log" && grep -q "commit" "$WORK/version.log" && grep -q "architectures" "$WORK/version.log"; then
    ok "--version: $(head -n 1 "$WORK/version.log")"
else
    bad "--version" "exit $rc: $(head -n 2 "$WORK/version.log" | tr '\n' ' ')"
fi
"$BIN" solvers > "$WORK/solvers.log" 2>&1
rc=$?
missing=""
for a in simpleFoam pimpleFoam rhoSimpleFoam interFoam; do grep -q "^  $a " "$WORK/solvers.log" || missing="$missing $a"; done
[ $rc = 0 ] && [ -z "$missing" ] && ok "solvers lists the four of the dispatch table" || bad "solvers" "exit $rc, missing:$missing"
"$BIN" capabilities --json > "$WORK/caps.json" 2> "$WORK/caps.err"
rc=$?
python3 - "$WORK/caps.json" "$SRC" "$BUILD" > "$WORK/caps.check" 2>&1 << 'PY'
import json, re, subprocess, sys
caps = json.load(open(sys.argv[1]))['capabilities']
src = open(sys.argv[2] + '/src/finiteVolume/fields/fv_patch_field.cuh').read()
a = src.index('inline const std::vector<std::string>& patchFieldTypeNames()')
table = re.findall(r'"([^"]+)"', src[a:src.index('return names;', a)])
listed = {c['name'] for c in caps if c['category'] == 'boundary condition'}
missing = [t for t in table if t not in listed]
cats = {c['category'] for c in caps}
need = {'solver', 'boundary condition', 'div scheme', 'snGrad', 'ddt', 'turbulence', 'linear solver', 'mesh', 'parallel', 'limitation'}
tests = set(subprocess.run(['ctest', '--test-dir', sys.argv[3], '-N'], capture_output=True, text=True).stdout.split())
gates = sorted({g for c in caps for g in c['gate'].split()})
absent = [g for g in gates if not any(t.endswith(g) or t == g for t in tests)]
bad_status = [c['name'] for c in caps if c['status'] not in ('validated', 'implemented', 'experimental', 'approximated', 'refused')]
validated_without_gate = [c['name'] for c in caps if c['status'] == 'validated' and not c['gate']]
print('entries %d, categories %d, boundary types %d of %d listed, gates %d, absent %s, missing %s, categories missing %s, '
      'bad status %s, validated without a gate %s' % (len(caps), len(cats), len(table) - len(missing), len(table), len(gates),
      absent, missing, sorted(need - cats), bad_status, validated_without_gate))
sys.exit(1 if (missing or absent or (need - cats) or bad_status or validated_without_gate) else 0)
PY
crc=$?
[ $rc = 0 ] && [ $crc = 0 ] && ok "capabilities --json: $(cat "$WORK/caps.check")" || bad "capabilities --json" "exit $rc: $(cat "$WORK/caps.check" "$WORK/caps.err" | head -n 4 | tr '\n' ' ')"
"$BIN" capabilities > "$WORK/caps.txt" 2>&1 && grep -q "^boundary condition" "$WORK/caps.txt" && grep -q "linearUpwindV" "$WORK/caps.txt" \
    && ok "capabilities (text) prints the groups" || bad "capabilities (text)" "$(head -n 3 "$WORK/caps.txt" | tr '\n' ' ')"

# ---- check on real cases -----------------------------------------------------------------------------------------------------
steady=$WORK/steady
cp -r "$SRC/validation/pitzDailySST" "$steady"
rm -rf "$steady"/[1-9]* "$steady/constant/polyMesh/.brae_"*
# the transient case: the steady one as pimpleFoam (no transient validation case ships a mesh), Euler, PIMPLE 1-2-0
transient=$WORK/transient
cp -r "$steady" "$transient"
python3 - "$transient" << 'PY2'
import re, sys
d = sys.argv[1]
def edit(path, pat, rep):
    s = open(path).read()
    s2 = re.sub(pat, rep, s, count=1)
    assert s2 != s, (path, pat)
    open(path, 'w').write(s2)
edit(d + '/system/controlDict', r'application\s+\w+;', 'application pimpleFoam;')
edit(d + '/system/controlDict', r'endTime\s+[^;]+;', 'endTime 0.0002;')
edit(d + '/system/controlDict', r'deltaT\s+[^;]+;', 'deltaT 0.0001;')
edit(d + '/system/fvSchemes', r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes { default Euler; }')
# the SIMPLE block replaced whole (its residualControl and consistent are no PIMPLE entries, which pimpleFoam refuses)
f = d + '/system/fvSolution'
s = open(f).read()
a = re.search(r'\bSIMPLE\b\s*\{', s)
assert a, 'no SIMPLE block'
depth = 0
k = a.end() - 1
while True:
    depth += {'{': 1, '}': -1}.get(s[k], 0)
    k += 1
    if depth == 0:
        break
s = s[:a.start()] + 'PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n}' + s[k:]
open(f, 'w').write(s)
PY2
for c in steady transient; do
    "$BIN" check -case "$WORK/$c" > "$WORK/check-$c.log" 2>&1
    rc=$?
    if { [ $rc = 0 ] || [ $rc = 1 ]; } && grep -q "^verdict: SUPPORTED" "$WORK/check-$c.log"; then
        ok "check $c case: exit $rc, $(grep '^verdict' "$WORK/check-$c.log")"
    else
        bad "check $c case" "exit $rc: $(grep -E 'refused|error|verdict' "$WORK/check-$c.log" | head -n 3 | tr '\n' ' ')"
    fi
done
# a comment holding a "(" between the boundary file's header and its patch list is a comment, as OpenFOAM's lexer reads it:
# check took it for the list and said "lists no patch" (a reconstructed boundary file, 2026-11-10)
commented="$WORK/commented"
rm -rf "$commented"
cp -r "$steady" "$commented"
python3 - "$commented/constant/polyMesh/boundary" << 'PY3'
import sys
p = sys.argv[1]
s = open(p).read()
i = s.index('}', s.index('FoamFile')) + 1
open(p, 'w').write(s[:i] + '\n// a note (with parentheses) before the list\n/* and (another) */\n' + s[i:])
PY3
n=$(grep -cE '^\s+type\s' "$steady/constant/polyMesh/boundary")
"$BIN" check -case "$commented" > "$WORK/check-commented.log" 2>&1
rc=$?
grep -q "mesh .* $n patches" "$WORK/check-commented.log" && [ $rc -lt 2 ] \
    && ok "check reads a boundary file with a commented '(' before its list: $n patches" \
    || bad "commented boundary file" "exit $rc: $(grep -E 'mesh|verdict' "$WORK/check-commented.log" | tr '\n' ' ')"

# ---- NEGATIVE CONTROLS: each must exit 2 and name what it refuses ------------------------------------------------------------
control()   # <name> <expected word> <setup command run inside the copy>
{
    local d=$WORK/ctl-$1
    rm -rf "$d"
    cp -r "$steady" "$d"
    (cd "$d" && eval "$3")
    "$BIN" check -case "$d" > "$WORK/ctl-$1.log" 2>&1
    local rc=$?
    if [ $rc = 2 ] && grep -q -- "$2" "$WORK/ctl-$1.log" && grep -q "^verdict: REFUSED" "$WORK/ctl-$1.log"; then
        ok "CONTROL $1 refused (exit 2, names '$2')"
    else
        bad "CONTROL $1" "exit $rc, '$2' $(grep -c -- "$2" "$WORK/ctl-$1.log") times: $(tail -n 3 "$WORK/ctl-$1.log" | tr '\n' ' ')"
    fi
}
control application fooFoam "sed -i 's/^application .*/application fooFoam;/' system/controlDict"
control bc fancyInletVelocity "python3 -c \"import re;p='0/U';s=open(p).read();s=re.sub(r'type\s+fixedValue;','type fancyInletVelocity;',s,count=1);open(p,'w').write(s)\""
control bc-named externalWallHeatFluxTemperature "python3 -c \"import re;p='0/p';s=open(p).read();s=re.sub(r'type\s+zeroGradient;','type externalWallHeatFluxTemperature;',s,count=1);open(p,'w').write(s)\""
control div vanLeerV "python3 -c \"import re;p='system/fvSchemes';s=open(p).read();s2=re.sub(r'div\(phi,U\)[^;]*;','div(phi,U) bounded Gauss vanLeerV;',s);assert s2!=s;open(p,'w').write(s2)\""
control sngrad faceCorrected "python3 -c \"import re;p='system/fvSchemes';s=open(p).read();s2=re.sub(r'laplacianSchemes\s*\{[^}]*\}','laplacianSchemes { default Gauss linear faceCorrected; }',s);assert s2!=s;open(p,'w').write(s2)\""
control ras v2f "python3 -c \"import re;p='constant/turbulenceProperties';s=open(p).read();s2=re.sub(r'RASModel\s+\w+;','RASModel v2f;',s);assert s2!=s;open(p,'w').write(s2)\""
d=$WORK/ctl-ddt
rm -rf "$d"
cp -r "$transient" "$d"
python3 -c "import re;p='$d/system/fvSchemes';s=open(p).read();s2=re.sub(r'ddtSchemes\s*\{[^}]*\}','ddtSchemes { default bounded backward; }',s);assert s2!=s;open(p,'w').write(s2)"
"$BIN" check -case "$d" > "$WORK/ctl-ddt.log" 2>&1
rc=$?
[ $rc = 2 ] && grep -q "bounded backward" "$WORK/ctl-ddt.log" && ok "CONTROL ddt 'bounded backward' refused (exit 2)" \
    || bad "CONTROL ddt" "exit $rc: $(tail -n 3 "$WORK/ctl-ddt.log" | tr '\n' ' ')"
# a PIMPLE control pimpleFoam does not read is refused by name; the transient arm above (the controls it does read) is this
# control's pair: check refused that arm too while it audited the block without pimpleFoam's own reads of it (lc5, 2026-11-09)
d=$WORK/ctl-pimplekey
rm -rf "$d"
cp -r "$transient" "$d"
python3 -c "import re;p='$d/system/fvSolution';s=open(p).read();s2=re.sub(r'(PIMPLE\s*\{)',r'\1\n    fooControl yes;',s,count=1);assert s2!=s;open(p,'w').write(s2)"
"$BIN" check -case "$d" > "$WORK/ctl-pimplekey.log" 2>&1
rc=$?
[ $rc = 2 ] && grep -q "PIMPLE/fooControl" "$WORK/ctl-pimplekey.log" && ok "CONTROL an unread PIMPLE control refused (exit 2, names PIMPLE/fooControl)" \
    || bad "CONTROL pimplekey" "exit $rc: $(tail -n 3 "$WORK/ctl-pimplekey.log" | tr '\n' ' ')"
mkdir -p "$WORK/notacase"
"$BIN" check -case "$WORK/notacase" > "$WORK/ctl-notacase.log" 2>&1
rc=$?
[ $rc = 3 ] && ok "CONTROL a directory with no controlDict: exit 3" || bad "CONTROL notacase" "exit $rc"

# a dynamicFvMesh brae does not implement, on the transient case: refused by check, as pimpleFoam's setup refuses it
ctl_dyn="$WORK/ctl-dynmesh"
rm -rf "$ctl_dyn"
cp -r "$transient" "$ctl_dyn"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\ndynamicFvMesh dynamicMultiMotionSolverFvMesh;\n' \
    > "$ctl_dyn/constant/dynamicMeshDict"
"$BIN" check -case "$ctl_dyn" > "$WORK/ctl-dynmesh.log" 2>&1
rc=$?
[ $rc = 2 ] && grep -q "dynamicMultiMotionSolverFvMesh" "$WORK/ctl-dynmesh.log" \
    && ok "CONTROL dynamicFvMesh dynamicMultiMotionSolverFvMesh refused by check (exit 2), as the solver refuses it" \
    || bad "CONTROL dynmesh" "exit $rc: $(tail -n 3 "$WORK/ctl-dynmesh.log" | tr '\n' ' ')"

# two active MRF zones on the steady case: simpleFoam rotates one and refuses a second, and check says so (occDrivAerRotMesh's
# RANS stage has four)
ctl_mrf="$WORK/ctl-mrf2"
rm -rf "$ctl_mrf"
cp -r "$steady" "$ctl_mrf"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\nMRF_A { cellZone zA; active yes; nonRotatingPatches (); origin (0 0 0); axis (0 0 1); omega 1; }\nMRF_B { cellZone zB; active yes; nonRotatingPatches (); origin (0 0 0); axis (0 0 1); omega 1; }\n' \
    > "$ctl_mrf/constant/MRFProperties"
"$BIN" check -case "$ctl_mrf" > "$WORK/ctl-mrf2.log" 2>&1
rc=$?
[ $rc = 2 ] && grep -q "2 active zones (MRF_A, MRF_B)" "$WORK/ctl-mrf2.log" \
    && ok "CONTROL two active MRF zones refused by check (exit 2), as simpleFoam refuses them" \
    || bad "CONTROL mrf2" "exit $rc: $(tail -n 3 "$WORK/ctl-mrf2.log" | tr '\n' ' ')"

# ---- command words beside case directories -----------------------------------------------------------------------------------
words=$WORK/words
mkdir -p "$words"
for w in run check solvers capabilities; do
    cp -r "$steady" "$words/$w"
    sed -i 's/^application .*/application fooFoamNamed'"$w"';/' "$words/$w/system/controlDict"
    (cd "$words" && "$BIN" "$w" > "$WORK/word-$w.log" 2>&1)
    if grep -q "fooFoamNamed$w" "$WORK/word-$w.log" && grep -q "is a case directory" "$WORK/word-$w.log"; then
        ok "a case directory called '$w': \`brae $w\` runs that case (and says so)"
    else
        bad "case directory '$w'" "$(head -n 3 "$WORK/word-$w.log" | tr '\n' ' ')"
    fi
done
(cd "$words" && "$BIN" capabilities --json > "$WORK/word-capsjson.log" 2>&1)
grep -q '"capabilities"' "$WORK/word-capsjson.log" && ok "beside a case called 'capabilities', \`brae capabilities --json\` is the command" \
    || bad "capabilities --json beside its case" "$(head -n 2 "$WORK/word-capsjson.log" | tr '\n' ' ')"
(cd "$words" && "$BIN" check -case run > "$WORK/word-checkrun.log" 2>&1)
grep -q "fooFoamNamedrun" "$WORK/word-checkrun.log" && grep -q "^verdict: REFUSED" "$WORK/word-checkrun.log" \
    && ok "\`brae check -case run\` checks the case called run" || bad "check -case run" "$(tail -n 2 "$WORK/word-checkrun.log" | tr '\n' ' ')"
"$BIN" run -case "$words/run" > "$WORK/run-cmd.log" 2>&1
rc1=$?
"$BIN" -case "$words/run" > "$WORK/run-plain.log" 2>&1
rc2=$?
if [ $rc1 = $rc2 ] && [ $rc1 != 0 ] && grep -q "fooFoamNamedrun" "$WORK/run-cmd.log" && grep -q "fooFoamNamedrun" "$WORK/run-plain.log"; then
    ok "\`brae run -case D\` refuses the unknown application as \`brae -case D\` does (exit $rc1)"
else
    bad "run -case" "exit $rc1 vs $rc2"
fi

# ---- check --setup (a GPU) and -parallel -------------------------------------------------------------------------------------
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    "$BIN" check -case "$steady" --setup > "$WORK/setup.log" 2>&1
    rc=$?
    if [ $rc = 0 ] && grep -q "brae check --setup: simpleFoam set the case up completely" "$WORK/setup.log" && ! grep -q "^Time = 1$" "$WORK/setup.log"; then
        ok "check --setup: $(grep 'GPU memory in use' "$WORK/setup.log" | sed 's/.*GPU memory/GPU memory/')"
    else
        bad "check --setup" "exit $rc: $(grep -E 'setup|Time =|rror' "$WORK/setup.log" | head -n 3 | tr '\n' ' ')"
    fi
else
    echo "SKIP: check --setup (no GPU)"
fi
"$BIN" -case "$steady" -parallel > "$WORK/parallel.log" 2>&1
rc=$?
[ $rc != 0 ] && grep -q "brae-mgpu" "$WORK/parallel.log" && ok "-parallel without the multi-GPU layer is refused, naming brae-mgpu" \
    || bad "-parallel" "exit $rc: $(tail -n 2 "$WORK/parallel.log" | tr '\n' ' ')"

echo "== $pass passed, $fail failed"
[ "$fail" = 0 ] && echo PASS || echo FAIL
[ "$fail" = 0 ]
