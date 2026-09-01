#!/usr/bin/env bash
# THE OF-MIRROR rhoSimpleFoam AS A SOLVER, not as a step: `brae -case <dir>` against real OpenFOAM.
#
# Every other rhoSimpleFoam gate drives the mirror through tests/test_rho_simple_step_cpp.cu, which
# takes its times from argv, reads no controlDict, honours no residualControl and writes no output. So
# the thing a USER runs -- the dispatch, the loop bounds, the write cadence, the convergence stop and
# the time directories -- was gated nowhere. This gate runs the shipped path end to end and checks the
# four things a solver has to get right beyond the arithmetic:
#
#   1. THE ANSWER. Same case, same iteration count, brae's written time directory against OpenFOAM's.
#      Measured on validation/rhoBox at 200 iterations: p 2.95e-12, T 8.24e-11, U 2.93e-11, rho 7.37e-11
#      -- the linear-solver floor, because the driver reads the case's own tolerances and both codes
#      solve the same systems. Bounds pinned at ~30x that; they tighten, never loosen.
#   2. THE OUTPUT IS OPENFOAM'S FORMAT. Real rhoSimpleFoam restarts from brae's written directory and
#      runs on. This is the only check that proves the written boundaryField entries, dimensions and
#      phi are what OF's reader requires -- brae reading its own output back would prove nothing.
#   3. RESTART. `startFrom latestTime` finds brae's own directory, and endTime is ABSOLUTE: restarting
#      at 20 with endTime 23 runs THREE steps, not twenty-three.
#   4. THE REFUSALS a solver needs: `writeFormat binary` (the writer emits ascii only) by name.
#
# The turbulent arm runs validation/rhoBoxF so the k/epsilon/nut/alphat write payload is exercised
# cheaply -- that fixture's turbulence is frozen, which does not matter here: the question is whether
# the fields are WRITTEN in OF's format and read back, not whether they moved.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
SRC="${CASE:-$ROOT/validation/rhoBox}"
TURBSRC="$ROOT/validation/rhoBoxF"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-200}
P_BOUND=1e-10; T_BOUND=3e-09; U_BOUND=1e-09; RHO_BOUND=3e-09

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-64s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# One case copy, prepared identically for both codes: ascii at 15 digits, no function objects, and
# residualControl removed so BOTH run exactly ITERS iterations from the same start (arm 5 puts it back).
stage()
{
    local dst="$1" src="$2" iters="$3"
    rm -rf "$dst"; cp -r "$src" "$dst"
    rm -rf "$dst"/[1-9]* "$dst"/0 "$dst"/processor* "$dst"/log.*
    cp -r "$dst/0.orig" "$dst/0" 2>/dev/null || true
    ITERS="$iters" python3 - "$dst" <<'PYEOF'
import os, re, sys
d = sys.argv[1]; n = os.environ['ITERS']
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', n),
             ('writeInterval', n), ('writeControl', 'timeStep'), ('startFrom', 'startTime'),
             ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PYEOF
}

# ---- arm 1: the answer, mirror vs OpenFOAM at a matched iteration count ---------------------------
stage "$W/brae" "$SRC" "$ITERS"
stage "$W/of"   "$SRC" "$ITERS"
( cd "$W/brae" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$W/brae" > brae.log 2>&1 ) \
    || { tail -5 "$W/brae/brae.log"; echo "FAIL: the mirror solver did not run"; exit 1; }
( cd "$W/of" && rhoSimpleFoam > of.log 2>&1 ) \
    || { tail -5 "$W/of/of.log"; echo "FAIL: OpenFOAM did not run"; exit 1; }
[ -d "$W/brae/$ITERS" ] || { echo "FAIL: the mirror wrote no $ITERS/ directory"; exit 1; }
[ -d "$W/of/$ITERS" ]   || { echo "FAIL: OpenFOAM wrote no $ITERS/"; exit 1; }

BRAE_DIR="$W/brae/$ITERS" OF_DIR="$W/of/$ITERS" START_DIR="$W/of/0" \
P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" U_BOUND="$U_BOUND" RHO_BOUND="$RHO_BOUND" \
python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
def read(p):
    try: s = open(p).read()
    except OSError: return None
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'internalField\s+uniform\s+\(?([^);]+)\)?;', s)
    return None if not u else np.array([float(x) for x in u.group(1).split()])
ok = True
for fld, key in (('p','P_BOUND'), ('T','T_BOUND'), ('U','U_BOUND'), ('rho','RHO_BOUND')):
    a = read(os.path.join(os.environ['BRAE_DIR'], fld))
    b = read(os.path.join(os.environ['OF_DIR'], fld))
    if a is None or b is None or a.shape != b.shape:
        print(f'     {fld:5s} MISSING or shape mismatch in the written output   FAIL'); ok = False; continue
    r = float(np.linalg.norm(a-b)/np.linalg.norm(b))
    bound = float(os.environ[key])
    good = r < bound
    print(f'     {fld:5s} relL2(mirror vs OpenFOAM) {r:.4e}   (bound {bound:.1e})   {"ok" if good else "FAIL"}')
    ok = ok and good
    # NON-VACUITY: the START state must miss the same bound, or a solver that did nothing would pass.
    s0 = read(os.path.join(os.environ['START_DIR'], fld))
    if s0 is not None and s0.shape == b.shape:
        r0 = float(np.linalg.norm(s0-b)/np.linalg.norm(b))
        if r0 < 10*bound:
            print(f'     {fld:5s} start state {r0:.3e} is inside 10x the bound -- vacuous   FAIL'); ok = False
sys.exit(0 if ok else 1)
PYEOF
say "the mirror's written fields match OpenFOAM's (non-vacuously)" "$([ $fail = 0 ] && echo ok || echo FAIL)"

# ---- arm 2: real OpenFOAM restarts FROM brae's written directory ----------------------------------
R="$W/ofread"; rm -rf "$R"; cp -r "$W/brae" "$R"
python3 - "$R/system/controlDict" "$ITERS" <<'PYEOF'
import re, sys
c, n = sys.argv[1], int(sys.argv[2])
s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %d;' % (n + 3), s)
open(c, 'w').write(s)
PYEOF
# The proof is that OpenFOAM CONTINUED from brae's state: its first step is ITERS+1 and it reaches
# End. Not a written directory -- OF writes on its own cadence (writeInterval ITERS from the stage
# above), which endTime+3 does not land on, so a directory check here fails on a restart that worked.
if ( cd "$R" && rhoSimpleFoam > ofread.log 2>&1 ); then
    grep -q "^End" "$R/ofread.log" && grep -q "^Time = $((ITERS+1))$" "$R/ofread.log" \
        && say "real OpenFOAM restarts from the mirror's written time directory" ok \
        || { tail -6 "$R/ofread.log"; say "real OpenFOAM restarts from the mirror's written time directory" FAIL; }
else
    tail -12 "$R/ofread.log"; say "real OpenFOAM restarts from the mirror's written time directory" FAIL
fi

# ---- arm 3: the mirror restarts from its own output, and endTime is ABSOLUTE ----------------------
B2="$W/brestart"; rm -rf "$B2"; cp -r "$W/brae" "$B2"; rm -f "$B2"/brae.log
python3 - "$B2/system/controlDict" "$ITERS" <<'PYEOF'
import re, sys
c, n = sys.argv[1], int(sys.argv[2])
s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %d;' % (n + 3), s)
open(c, 'w').write(s)
PYEOF
rout=$( cd "$B2" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$B2" 2>&1 )
echo "$rout" | grep -q "start $ITERS" \
    && echo "$rout" | grep -q "reached endTime (3 iterations)" \
    && [ -d "$B2/$((ITERS+3))" ] \
    && say "the mirror restarts at latestTime and runs endTime-startTime steps" ok \
    || { echo "$rout" | tail -5; say "the mirror restarts at latestTime and runs endTime-startTime steps" FAIL; }

# ---- arm 4: the turbulence write payload (k, epsilon, nut, alphat), read back by OpenFOAM ---------
if [ -d "$TURBSRC" ]; then
    stage "$W/turb" "$TURBSRC" 10
    if ( cd "$W/turb" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$W/turb" > turb.log 2>&1 ); then
        missing=""
        for fl in k epsilon nut alphat p T U rho phi; do
            [ -f "$W/turb/10/$fl" ] || missing="$missing $fl"
        done
        [ -z "$missing" ] && say "the turbulent write payload is complete (k,epsilon,nut,alphat,phi)" ok \
                          || say "the turbulent write payload is missing:$missing" FAIL
        T2="$W/turbread"; rm -rf "$T2"; cp -r "$W/turb" "$T2"
        python3 - "$T2/system/controlDict" <<'PYEOF'
import re, sys
c = sys.argv[1]; s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 12;', s)
open(c, 'w').write(s)
PYEOF
        ( cd "$T2" && rhoSimpleFoam > turbread.log 2>&1 ) && grep -q "^End" "$T2/turbread.log" \
            && say "OpenFOAM restarts from the mirror's TURBULENT output" ok \
            || { tail -8 "$T2/turbread.log"; say "OpenFOAM restarts from the mirror's TURBULENT output" FAIL; }
    else
        tail -6 "$W/turb/turb.log"; say "the mirror runs the turbulent fixture" FAIL
    fi
else
    say "turbulent fixture $TURBSRC missing" SKIP
fi

# ---- arm 5: residualControl stops the run before endTime -----------------------------------------
C="$W/conv"; stage "$C" "$SRC" 5000
python3 - "$C/system/fvSolution" <<'PYEOF'
import re, sys
f = sys.argv[1]; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}',
           'residualControl { p 1e-3; U 1e-3; "(h|e)" 1e-3; }', s)
open(f, 'w').write(s)
PYEOF
cout=$( cd "$C" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$C" 2>&1 )
niter=$(echo "$cout" | grep -oE "converged in [0-9]+ iterations" | grep -oE "[0-9]+" | head -1)
if [ -n "$niter" ] && [ "$niter" -lt 5000 ]; then
    echo "$cout" | grep -q "residualControl=on" && [ -d "$C/$niter" ] \
        && say "residualControl stops the run at $niter of 5000 and writes that time" ok \
        || say "residualControl stopped but did not write its own time directory" FAIL
else
    echo "$cout" | tail -4; say "residualControl stops the run before endTime" FAIL
fi

# ---- arm 6: writeFormat binary is refused by name (the writer emits ascii only) -------------------
BF="$W/bin"; stage "$BF" "$SRC" 5
sed -i 's/writeFormat ascii;/writeFormat binary;/' "$BF/system/controlDict"
grep -q "writeFormat binary" "$BF/system/controlDict" || { echo "FAIL: the binary mutation did not apply"; exit 1; }
bout=$( cd "$BF" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$BF" 2>&1 )
echo "$bout" | grep -q "writeFormat is \`binary\`" \
    && [ ! -d "$BF/5" ] \
    && say "a binary writeFormat is refused by name, and nothing is written" ok \
    || { echo "$bout" | tail -4; say "a binary writeFormat is refused by name, and nothing is written" FAIL; }


# ---- arms 7-9: THE CUDA ARM. Same solver, device modules doing the arithmetic ---------------------
# The two arms share the case parse, the loop, the write cadence and every refusal (only the equations
# move), so what these check is that moving them changed no answer. Measured on rhoBox at 200
# iterations -- CUDA vs OpenFOAM: p 3.00e-12, T 1.27e-09, U 4.45e-10, rho 1.19e-09; CUDA vs the host
# mirror: p 2.83e-12, T 1.32e-09, U 4.62e-10, rho 1.24e-09. Looser than the host arm's 1e-11 by exactly
# the linear solvers between them (AMG-PCG against the host's BiCGStab), which is why the bounds differ.
# On the 112k-cell turbulent transonic sbMatched the two arms converge on the SAME iteration (123) and
# their written fields agree to 3.9e-09 (p), 1.3e-08 (U), 6.2e-07 (nut).
stage "$W/cuda" "$SRC" "$ITERS"
if ( cd "$W/cuda" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/cuda" > cuda.log 2>&1 ); then
    BRAE_DIR="$W/cuda/$ITERS" OF_DIR="$W/of/$ITERS" HOST_DIR="$W/brae/$ITERS" \
    python3 "$ROOT/tests/rho_mirror_compare.py" || fail=1
    say "the CUDA mirror matches OpenFOAM and the host mirror" "$([ $fail = 0 ] && echo ok || echo FAIL)"
else
    tail -6 "$W/cuda/cuda.log"; say "the CUDA mirror solver runs" FAIL
fi

# OpenFOAM must restart from the CUDA arm's output too: the device write path materialises k and
# epsilon's boundaries with deviceBCValue rather than echoing the start directory, and this is what
# proves it wrote something OF's reader accepts.
RC="$W/cudaread"; rm -rf "$RC"; cp -r "$W/cuda" "$RC"
python3 "$ROOT/tests/rho_mirror_restart_dict.py" "$RC/system/controlDict" "$ITERS"
( cd "$RC" && rhoSimpleFoam > cudaread.log 2>&1 ) && grep -q "^End" "$RC/cudaread.log" \
    && grep -q "^Time = $((ITERS+1))$" "$RC/cudaread.log" \
    && say "real OpenFOAM restarts from the CUDA mirror's written directory" ok \
    || { tail -6 "$RC/cudaread.log"; say "real OpenFOAM restarts from the CUDA mirror's written directory" FAIL; }

# The selector refuses a value it does not know rather than falling through to the PRE-MIRROR driver,
# which would report the old path's answer under a mirror request.
sout=$( cd "$W/cuda" && BRAE_RHOSIMPLEFOAM_MIRROR=gpu "$BIN" -case "$W/cuda" 2>&1 || true )
echo "$sout" | grep -q "BRAE_RHOSIMPLEFOAM_MIRROR is 'gpu'" \
    && say "an unknown mirror selector is refused by name" ok \
    || { echo "$sout" | tail -3; say "an unknown mirror selector is refused by name" FAIL; }


# ---- arm 10: kOmegaSST is REFUSED on the CUDA arm, and RUNS on the host arm -----------------------
# The device projection gates its whole closure set-up on epsilon being present, and kOmegaSST leaves
# epsilon empty (its second scalar is omega). That skipped the nut upload too, which lives inside the
# same block -- so an SST case ran with muEff = the LAMINAR viscosity while reporting kOmegaSST. A
# wrong run, not a missing feature, and invisible from the device side because every buffer it would
# have filled is simply absent. The host arm carries kOmegaSST, which is what makes this arm
# discriminating: the same case must refuse on one arm and RUN on the other.
if [ -d "$ROOT/validation/sbMatched" ]; then
    SST="$W/sst"; stage "$SST" "$ROOT/validation/sbMatched" 3
    sed -i 's/RASModel *kEpsilon;/RASModel        kOmegaSST;/' "$SST/constant/turbulenceProperties"
    cp "$SST/0/epsilon" "$SST/0/omega"
    sed -i 's/object *epsilon;/object      omega;/' "$SST/0/omega"
    sed -i 's|div(phi,epsilon)    $turbulence;|div(phi,epsilon)    $turbulence;\n    div(phi,omega)      $turbulence;|' "$SST/system/fvSchemes"
    grep -q "kOmegaSST" "$SST/constant/turbulenceProperties" || { echo "FAIL: the SST mutation did not apply"; exit 1; }
    cout=$( cd "$SST" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$SST" 2>&1 || true )
    echo "$cout" | grep -q "RASModel 'kOmegaSST'" && echo "$cout" | grep -q "laminar run under a turbulent model" \
        && say "kOmegaSST is refused by name on the CUDA arm" ok \
        || { echo "$cout" | tail -3; say "kOmegaSST is refused by name on the CUDA arm" FAIL; }
    hout=$( cd "$SST" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$SST" 2>&1 || true )
    echo "$hout" | grep -q "^Time = " \
        && say "...and the SAME case runs on the host arm (the refusal is device-specific)" ok \
        || { echo "$hout" | tail -3; say "...and the SAME case runs on the host arm (the refusal is device-specific)" FAIL; }
else
    say "sbMatched missing -- SST refusal arm skipped" SKIP
fi

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
