#!/usr/bin/env bash
# `laminar { model generalizedNewtonian; }` -- a sub-dictionary the rhoSimpleFoam mirror never opened.
#
# `simulationType laminar` DOES NOT MEAN "no model". OpenFOAM's laminarModel::New selects from a table
# of three -- Stokes, generalizedNewtonian and Maxwell -- and only Stokes leaves the molecular viscosity
# alone. generalizedNewtonian's nuEff() RETURNS the model's nu (generalizedNewtonian.C:139-146) rather
# than adding to it, so a code that ignores the block is not approximating the model: it is solving a
# different momentum equation.
#
# THE MIRROR ACCEPTED `laminar` BY FALLING THROUGH and read nothing inside it, on both arms. So
# squareBendLiqNoNewtonian -- whose block is exactly this one -- would have run on the molecular
# viscosity the moment its thermo blocker was lifted. That ordering is the point of this gate: it must
# land BEFORE the liquid thermo, or clearing that blocker turns a correct refusal into a confident
# wrong answer.
#
# AND THE SHARED READER WAS PROMISING A CAPABILITY ON ONE DRIVER'S BEHALF. `ctl.gnPowerLaw` is consumed
# only by the COMPRESSIBLE legacy path (device_simple_foam.cu:3540 in rhoSimpleStep, and :1319 behind
# `compressible_`). Incompressible simpleFoam read the model, PRINTED its coefficients, and ran Stokes.
# ARM 3 is that case: it must now refuse, and ARM 5 keeps the compressible arm that really does apply it
# running, so the refusal is discriminating rather than blanket.
#
# THE ORACLE IS REAL OpenFOAM AGAINST ITSELF (ARM 0): the same mesh and the same 0/, once with the block
# and once without. If those two agreed, refusing the block would be refusing a no-op and every arm
# below would be theatre.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
RHO="$ROOT/validation/rhoBox"
INC="$ROOT/validation/cav3d_cf"
OFBASH=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$RHO" ]  || { echo "SKIP: fixture $RHO missing"; exit 77; }
[ -f "$OFBASH" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
IT=100
# squareBendLiqNoNewtonian's OWN block, verbatim -- the tutorial this item is about.
GN='laminar { model generalizedNewtonian; viscosityModel powerLaw; nuMin 1e-03; nuMax 1; n 0.4; }'
MX='laminar { model Maxwell; nuM 1e-05; lambda 0.1; }'

fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
its() { [ -f "$1/run.log" ] && grep -c '^Time = ' "$1/run.log" 2> /dev/null | head -1 || echo 0; }

stage() {   # stage <dir> <fixture> <laminar body, or "">
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$2/constant" "$2/system" "$1/"
    if [ -d "$2/0.orig" ]; then cp -r "$2/0.orig" "$1/0"; else cp -r "$2/0" "$1/0"; fi
    python3 - "$1" "$IT" "$3" <<'PYEOF'
import re, sys
d, it, body = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % it, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % it, s)
open(c, 'w').write(s + '\n')
open(d + '/constant/turbulenceProperties', 'w').write(
    'FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\n'
    'simulationType laminar;\n' + body + '\n')
PYEOF
}
runOF()   { ( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$1" && rhoSimpleFoam > run.log 2>&1 ); }
# `${1:+VAR=$1} cmd` does NOT work as an environment prefix -- bash expands it and then looks for a
# COMMAND called "BRAE_RHOSIMPLEFOAM_MIRROR=1". An empty first argument means "the legacy driver", so
# the two cases are spelled out.
runBrae() {
    if [ -n "$1" ]; then ( cd "$2" && BRAE_RHOSIMPLEFOAM_MIRROR="$1" "$BRAE" -case "$2" > run.log 2>&1 )
    else                 ( cd "$2" && "$BRAE" -case "$2" > run.log 2>&1 ); fi
}
refusedBy() { grep -q "$2" "$1/run.log" 2> /dev/null && [ "$(its "$1")" = 0 ]; }

# ---- ARM 0: THE ORACLE + THE CONTROL -- OpenFOAM against itself ------------------------------------
stage "$W/ofGN" "$RHO" "$GN";  runOF "$W/ofGN"
stage "$W/ofST" "$RHO" "";     runOF "$W/ofST"
[ "$(its "$W/ofGN")" -ge 1 ] && [ "$(its "$W/ofST")" -ge 1 ] \
    && say "control: real OpenFOAM runs the fixture both with and without the block" ok \
    || { tail -3 "$W/ofGN/run.log"; say "control: real OpenFOAM runs the fixture both with and without the block" FAIL; }
grep -q 'generalizedNewtonian' "$W/ofGN/run.log" \
    && say "control: OpenFOAM confirms it SELECTED the model (not a typo'd key)" ok \
    || say "control: OpenFOAM confirms it SELECTED the model (not a typo'd key)" FAIL
relU() { python3 - "$1" "$2" <<'PYEOF'
import re, os, sys
def latest(d):
    ts = [t for t in os.listdir(d) if t.replace('.', '', 1).isdigit() and float(t) > 0]
    return os.path.join(d, max(ts, key=float), 'U') if ts else None
def rd(f):
    if not f or not os.path.exists(f): return None
    b = open(f, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n\(', b)
    if not m: return None
    out = []
    for ln in b[m.end():].split(b')\n', 1)[0].split(b'\n'):
        ln = ln.strip()
        if ln.startswith(b'('): out.append([float(x) for x in ln.strip(b'()').split()])
    return out
a, b = rd(latest(sys.argv[1])), rd(latest(sys.argv[2]))
if not a or not b or len(a) != len(b): print('nan'); raise SystemExit
mx = max(max(abs(p[i] - q[i]) for i in range(3)) for p, q in zip(a, b))
nm = max(max(abs(v) for v in p) for p in b) or 1.0
print('%.4e' % (mx / nm))
PYEOF
}
REL=$(relU "$W/ofGN" "$W/ofST")
# THE NOISE FLOOR, so the number above is read against something. Real OpenFOAM run twice on the same
# case must give the same answer; anything the model moves above that is signal, not scatter.
stage "$W/ofST2" "$RHO" "";    runOF "$W/ofST2"
NOISE=$(relU "$W/ofST" "$W/ofST2")
# 2.0e-02, set from what this fixture MEASURES (3.92e-02) with room to spare, not from a number
# measured somewhere else. rhoBox is a small buoyant box whose converged velocity is not
# viscosity-dominated, so 4% is what the model is worth HERE; on the tutorial this item is about
# (squareBendLiqNoNewtonian) OpenFOAM's own generalizedNewtonian:nu is 1101x to 2532x the molecular
# mu/rho, which is the number the refusal text quotes. The threshold is not loosened to fit: the arm
# below requires the difference to be at least a thousand times the run-to-run noise as well.
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.02 else 1)" "$REL" 2> /dev/null \
    && say "ORACLE: the block CHANGES OpenFOAM's own answer, so refusing it is not refusing a no-op" ok \
    || say "ORACLE: the block CHANGES OpenFOAM's own answer, so refusing it is not refusing a no-op" FAIL
python3 -c "
import sys
sig, noise = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if sig > 1000.0 * max(noise, 1e-12) else 1)" "$REL" "$NOISE" 2> /dev/null \
    && say "...and it is at least 1000x OpenFOAM's own run-to-run noise, so it is signal" ok \
    || say "...and it is at least 1000x OpenFOAM's own run-to-run noise, so it is signal" FAIL
printf '        (generalizedNewtonian vs Stokes: %s relative on U; OpenFOAM against itself: %s)\n' "$REL" "$NOISE"

# ---- ARM 1 + 2: the rhoSimpleFoam MIRROR refuses, by name, on BOTH arms ----------------------------
for arm in 1 cuda; do
    label=$([ "$arm" = 1 ] && echo "host" || echo "CUDA")
    stage "$W/m$arm" "$RHO" "$GN"; runBrae "$arm" "$W/m$arm"
    refusedBy "$W/m$arm" "generalizedNewtonian" \
        && say "rho mirror ($label): refuses the model it does not apply, by name" ok \
        || { tail -2 "$W/m$arm/run.log"; say "rho mirror ($label): refuses the model it does not apply, by name" FAIL; }
    stage "$W/x$arm" "$RHO" "$MX"; runBrae "$arm" "$W/x$arm"
    refusedBy "$W/x$arm" "Maxwell" \
        && say "rho mirror ($label): refuses Maxwell too" ok \
        || say "rho mirror ($label): refuses Maxwell too" FAIL
done

# ---- ARM 3: legacy INCOMPRESSIBLE simpleFoam -- the driver that announced and ignored --------------
if [ -d "$INC" ]; then
    stage "$W/inc" "$INC" "$GN"; runBrae "" "$W/inc"
    refusedBy "$W/inc" "generalizedNewtonian" \
        && say "legacy simpleFoam: refuses the model it never applied (it used to announce it)" ok \
        || { tail -2 "$W/inc/run.log"; say "legacy simpleFoam: refuses the model it never applied (it used to announce it)" FAIL; }
else
    say "legacy simpleFoam (incompressible fixture missing, skipped)" ok
fi

# ---- ARM 4: THE FAIL-PROOF -- the refusal must be DISCRIMINATING -----------------------------------
# If it fired on any laminar case, every arm above would pass with a one-line blanket throw.
for spec in "explicit Stokes:laminar { model Stokes; }" "no laminar block at all:"; do
    what="${spec%%:*}"; body="${spec#*:}"
    ok=1
    for arm in 1 cuda; do
        stage "$W/ok$arm" "$RHO" "$body"; runBrae "$arm" "$W/ok$arm"
        [ "$(its "$W/ok$arm")" -ge 1 ] || ok=0
    done
    [ "$ok" = 1 ] \
        && say "fail-proof: '$what' still RUNS on both mirror arms" ok \
        || say "fail-proof: '$what' still RUNS on both mirror arms" FAIL
done

# ---- ARM 5: the arm that DOES apply it must still run it ------------------------------------------
# device_simple_foam.cu:3540 (rhoSimpleStep) is a real consumer, so the compressible legacy path keeps
# the capability. A blanket refusal in the shared reader would have taken it away.
stage "$W/leg" "$RHO" "$GN"; runBrae "" "$W/leg"
[ "$(its "$W/leg")" -ge 1 ] && grep -q 'generalizedNewtonian/powerLaw' "$W/leg/run.log" \
    && say "compressible legacy: still RUNS the model it does apply, and announces it" ok \
    || { tail -2 "$W/leg/run.log"; say "compressible legacy: still RUNS the model it does apply, and announces it" FAIL; }

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
