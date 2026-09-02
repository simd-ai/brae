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
#   5. THE CRITERIA. residualControl on the CUDA arm reads the closure's residuals too (arm 13).
#   6. THE SOLVER ENTRIES. tolerance/relTol/maxIter/minIter reach each equation from ITS entry (arm 14).
#   7. TWO REFUSALS OpenFOAM's construction would have made: no 0/alphat on a RAS case (arm 15) and a
#      flux-switched pressure patch the mirror never refreshes (arm 16).
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


# ---- arm 11: THE SAME CASE DIRECTORY, RUN TWICE -- and the second run must agree BIT FOR BIT --------
# Every other arm stages a pristine copy, so nothing here ever ran a solver twice over its own leftovers
# -- and that is precisely the hole that hid a crash. The driver caches the AMG hierarchy in
# constant/polyMesh/.brae_amgcache, so run 2 LOADS what run 1 built; the loader did not rebuild the
# Galerkin gather lists it needs, and run 2 died in galDiagGatherK reading index 0 of a zero-length
# buffer ("amul: an illegal memory access was encountered"). Fixed in loadAMGCache.
#
# The bound is EQUALITY, not a tolerance: the cache stores the agglomeration STRUCTURE and the matrix
# values are re-Galerkined every step, so a warm run must reproduce a cold one exactly. Anything else
# means the cached hierarchy is not the built one, which a tolerance would hide.
RP="$W/repeat"; stage "$RP" "$SRC" 30
rm -f "$RP/constant/polyMesh/.brae_amgcache"
for pass in 1 2; do
    if ( cd "$RP" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$RP" > "repeat$pass.log" 2>&1 ) \
       && grep -q "^End" "$RP/repeat$pass.log"; then
        say "the CUDA arm runs in the same directory (pass $pass)" ok
    else
        tail -4 "$RP/repeat$pass.log"; say "the CUDA arm runs in the same directory (pass $pass)" FAIL
    fi
    [ "$pass" = 1 ] && { rm -rf "$W/cold"; cp -r "$RP/30" "$W/cold"; rm -rf "$RP/30"; }
done
[ -f "$RP/constant/polyMesh/.brae_amgcache" ] \
    && say "...and the run actually wrote an AMG cache (else the arm is vacuous)" ok \
    || say "...and the run actually wrote an AMG cache (else the arm is vacuous)" FAIL
COLD="$W/cold" WARM="$RP/30" python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
def read(p):
    try: s = open(p).read()
    except OSError: return None
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if not m: return None
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
ok = True
for fld in ('p', 'T', 'U', 'rho', 'phi'):
    a = read(os.path.join(os.environ['COLD'], fld))
    b = read(os.path.join(os.environ['WARM'], fld))
    if a is None or b is None or a.shape != b.shape:
        print('     cache %-4s MISSING or shape mismatch   FAIL' % fld); ok = False; continue
    d = float(np.max(np.abs(a - b)))
    print('     cache %-4s cold vs warm max|diff| %.3e   %s' % (fld, d, 'ok' if d == 0.0 else 'FAIL'))
    ok = ok and d == 0.0
sys.exit(0 if ok else 1)
PYEOF
say "a cache-loaded hierarchy reproduces a cold one bit for bit" "$([ $fail = 0 ] && echo ok || echo FAIL)"

# ---- arm 12: the porous zone reaches the DEVICE momentum equation ----------------------------------
# rhoUEqn.cu has applied a porous zone since it was written and nothing built a DevicePorosity for this
# driver, so every fvOption the host arm implements was reported as "implemented on the host arm only"
# and the run refused -- OpenFOAM's own angledDuctExplicitFixedCoeff among them. The projection is
# shared with the harness (buildDeviceStepInput). fixedCoeff is the reachable model; DarcyForchheimer
# on a force-dimensioned momentum equation is refused BY NAME by rhoUEqn (it is given neither the
# per-cell mu nor rho that Cd = mu*D + rho|U|*F needs), which validation/rhoBoxDF pins here.
if [ -d "$ROOT/validation/rhoBoxDF" ]; then
    DFC="$W/dfcuda"; stage "$DFC" "$ROOT/validation/rhoBoxDF" 3
    dout=$( cd "$DFC" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$DFC" 2>&1 || true )
    echo "$dout" | grep -q "explicitPorositySource/DarcyForchheimer on" \
        && echo "$dout" | grep -q "DarcyForchheimerTemplates.C" \
        && say "the porosity is projected, and DarcyForchheimer refuses by name on the device" ok \
        || { echo "$dout" | tail -3; say "the porosity is projected, and DarcyForchheimer refuses by name on the device" FAIL; }
    hout=$( cd "$DFC" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$DFC" 2>&1 || true )
    echo "$hout" | grep -q "^End" \
        && say "...and the SAME porous case runs on the host arm" ok \
        || { echo "$hout" | tail -3; say "...and the SAME porous case runs on the host arm" FAIL; }
else
    say "rhoBoxDF missing -- porosity arm skipped" SKIP
fi


# ---- arm 13: residualControl on the CUDA arm SEES the turbulence criteria --------------------------
# The device step returns U, e|h and p residuals and nothing else -- its closure hook is a
# std::function<void()> -- so the driver's `if (r.count("k"))` branches were dead and a case naming
# "(k|epsilon)" (angledDuct, squareBend among the stock tutorials) was declared converged on p, U and
# e alone, a strict SUBSET of OpenFOAM's criteria (simpleControl::criteriaSatisfied walks every field
# solved this step). The residuals were computed all along (kEpsilon.cu writes them into the stages the
# driver owns); the driver now reads them after the step.
#
# The oracle is that the criterion BINDS: with everything else at 1e-3, tightening the turbulence
# criterion alone must move the stopping iteration. Measured on rhoKE (3200 cells, kEpsilon): 67 at
# 1e-3 and 83..94 at 1e-5 across runs (this solver is not run-to-run reproducible near a tight
# criterion, which is why the arm asserts the ORDER and not the count), k and epsilon on every log line. Fail-proof (merge removed): 61 under BOTH, no k
# printed. The third check is the strongest: with the case's linear solvers tightened to 1e-12/0 the
# two arms must converge on the SAME iteration -- they read the same criteria, so they stop together
# (67 = 67, residuals agreeing to five digits). Under the case's own tolerances they stop at 67 and 89:
# that spread is the linear solvers (AMG-PCG against BiCGStab), which is why it is not asserted.
if [ -d "$ROOT/validation/rhoKE" ] && command -v blockMesh > /dev/null 2>&1; then
    stageKE()   # $1 dst  $2 turbulence criterion  $3 tight|loose
    {
        stage "$1" "$ROOT/validation/rhoKE" 3000
        ( cd "$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on rhoKE"; exit 1; }
        TOL="$2" MODE="$3" python3 - "$1/system/fvSolution" <<'PYEOF2'
import os, re, sys
f = sys.argv[1]; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}',
           'residualControl { p 1e-3; U 1e-3; h 1e-3; "(k|epsilon)" %s; }' % os.environ['TOL'], s)
if os.environ['MODE'] == 'tight':
    s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
    s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
open(f, 'w').write(s)
PYEOF2
    }
    nOf()   # the stopping iteration of a log, or empty
    {
        grep -oE "converged in [0-9]+ iterations" "$1" | grep -oE "[0-9]+" | head -1
    }
    stageKE "$W/ke3" 1e-3 loose
    stageKE "$W/ke5" 1e-5 loose
    ( cd "$W/ke3" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/ke3" > run.log 2>&1 )
    ( cd "$W/ke5" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/ke5" > run.log 2>&1 )
    n3=$(nOf "$W/ke3/run.log"); n5=$(nOf "$W/ke5/run.log")
    if [ -z "$n3" ] || [ -z "$n5" ]; then
        tail -3 "$W/ke3/run.log"; say "the CUDA arm converges on rhoKE under residualControl" FAIL
    else
        say "the CUDA arm converges on rhoKE under residualControl ($n3 / $n5 iterations)" ok
        [ "$n5" -gt "$n3" ] \
            && say "tightening the turbulence criterion alone moves the stop (the criterion BINDS)" ok \
            || say "tightening the turbulence criterion alone moves the stop (the criterion BINDS)" FAIL
        grep -qE "^Time = .* k [0-9.e+-]+ +epsilon [0-9.e+-]+" "$W/ke3/run.log" \
            && say "k and epsilon residuals are on the CUDA arm's log line" ok \
            || say "k and epsilon residuals are on the CUDA arm's log line" FAIL
    fi
    stageKE "$W/ket" 1e-3 tight
    stageKE "$W/keh" 1e-3 tight
    ( cd "$W/ket" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/ket" > run.log 2>&1 )
    ( cd "$W/keh" && BRAE_RHOSIMPLEFOAM_MIRROR=1    "$BIN" -case "$W/keh" > run.log 2>&1 )
    nt=$(nOf "$W/ket/run.log"); nh=$(nOf "$W/keh/run.log")
    [ -n "$nt" ] && [ "$nt" = "$nh" ] \
        && say "under tight linear solvers both arms stop on the same iteration ($nt)" ok \
        || say "under tight linear solvers both arms stop on the same iteration (cuda ${nt:-none}, host ${nh:-none})" FAIL
else
    say "rhoKE or blockMesh missing -- turbulence residualControl arm skipped" SKIP
fi


# ---- arm 14: every equation's fvSolution/solvers entry reaches ITS solve, both arms ------------------
# Both drivers took the energy tolerance from the turbulence slot (which nothing filled: the reader's
# k/epsilon block is gated on ctl.turbulent and neither driver set it, so 1e-8/0 whatever the case
# said), every equation's maxIter from p's entry, and forwarded minIter nowhere. OF reads all four per
# field from that field's own sub-dictionary (lduMatrixSolver.C:196-205). The oracle and its controls
# are in tests/rho_solver_entries.py -- measured on rhoKE at 10 iterations, every mutated entry moves
# its field by 2e-04 .. 2.8e-01 on both arms, and the fail-proof (old assignments restored) reads
# 0.000e+00 on every energy and turbulence row while the p control still moves.
if [ -d "$ROOT/validation/rhoKE" ] && command -v blockMesh > /dev/null 2>&1; then
    stage "$W/ent" "$ROOT/validation/rhoKE" 10
    ( cd "$W/ent" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on rhoKE"; exit 1; }
    for arm in 1 cuda; do
        python3 "$ROOT/tests/rho_solver_entries.py" "$W/ent" "$BIN" "$arm" "$W/ent_$arm" h \
            && say "the case's solver entries reach every equation (arm $arm)" ok \
            || say "the case's solver entries reach every equation (arm $arm)" FAIL
    done
else
    say "rhoKE or blockMesh missing -- solver-entries arm skipped" SKIP
fi


# ---- arm 15: a turbulent case without 0/alphat is refused by name, on BOTH arms ---------------------
# OpenFOAM's EddyDiffusivity reads alphat MUST_READ at construction and fatals without it. The host arm
# refused late (at the first closure call); the CUDA arm did not refuse at all and ran the energy
# equation on the laminar diffusivity under the model's name (measured on rhoKE with the file removed:
# `Time = 1 ... k ... epsilon` and on). createFields refuses now, before anything runs, on both arms.
# The control is arm 13 above: the same fixture WITH the file runs to convergence on the CUDA arm.
if [ -d "$ROOT/validation/rhoKE" ] && command -v blockMesh > /dev/null 2>&1; then
    stage "$W/noalphat" "$ROOT/validation/rhoKE" 3
    ( cd "$W/noalphat" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on rhoKE"; exit 1; }
    rm -f "$W/noalphat/0/alphat"
    for arm in 1 cuda; do
        out=$( cd "$W/noalphat" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BIN" -case "$W/noalphat" 2>&1 || true )
        echo "$out" | grep -q "alphat does not exist" && ! echo "$out" | grep -q "^Time = 1" \
            && say "a RAS case without 0/alphat is refused by name before it runs (arm $arm)" ok \
            || { echo "$out" | tail -3; say "a RAS case without 0/alphat is refused by name before it runs (arm $arm)" FAIL; }
    done
else
    say "rhoKE or blockMesh missing -- alphat arm skipped" SKIP
fi

# ---- arm 16: a flux-switched p (inletOutlet / outletInlet) is refused by name, on BOTH arms ---------
# The flux switch is pushed into U, he and T every iteration and never into p, so such a patch would
# keep its seeded switch for the whole run. No fixture carries one; rhoBox's outlet is mutated here.
for bc in outletInlet inletOutlet; do
    stage "$W/pio_$bc" "$SRC" 3
    python3 - "$W/pio_$bc/0/p" "$bc" <<'PYEOF'
import re, sys
p, bc = sys.argv[1], sys.argv[2]; s = open(p).read()
s, n = re.subn(r'(outlet\s*\{)[^}]*\}', r'\1 type %s; %s uniform 100000; value uniform 100000; }' % (bc, 'outletValue' if bc == 'outletInlet' else 'inletValue'), s, count=1)
assert n == 1
open(p, 'w').write(s)
PYEOF
    for arm in 1 cuda; do
        out=$( cd "$W/pio_$bc" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BIN" -case "$W/pio_$bc" 2>&1 || true )
        echo "$out" | grep -q "flux-switched" && echo "$out" | grep -q "'outlet' is \`$bc\`" && ! echo "$out" | grep -q "^Time = 1" \
            && say "a $bc pressure patch is refused by name (arm $arm)" ok \
            || { echo "$out" | tail -3; say "a $bc pressure patch is refused by name (arm $arm)" FAIL; }
    done
done

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
