#!/usr/bin/env bash
# BRAE'S OWN OUTPUT MUST BE READABLE BY REAL OpenFOAM, AND MEAN THE SAME THING.
#
# A boundary condition's dictionary carries two kinds of entry: values the solve produces, and INPUTS the
# solve never changes. OpenFOAM's write() echoes both, because its own reader needs them back. brae's
# writer echoed the first kind and, for three entries, not the second:
#
#   `intensity`     turbulentIntensityKineticEnergyInletFvPatchScalarField.C:76 reads it with
#                   dict.get<scalar>(), which THROWS when absent, and :155 always writes it.
#   `mixingLength`  the same, on turbulentMixingLength{DissipationRate,Frequency}Inlet.
#   `rho`           flowRateInletVelocityFvPatchVectorField.C:247 writes it whenever it is not the
#                   default, under `if (!volumetric_)`. `rho none` selects OpenFOAM's VOLUMETRIC branch
#                   for a massFlowRate (.C:208), so losing it silently changes the prescribed inlet.
#
# The first two made brae's output UNREADABLE -- a hard OpenFOAM error. The third made it readable and
# WRONG, which is worse. Neither was visible to any field-vs-field gate, because those compare the
# internalField of a file nothing ever reads back.
#
#   ARM 1  validation/rhoTI (turbulentIntensityKineticEnergyInlet + turbulentMixingLengthFrequencyInlet):
#          real OpenFOAM must restart from brae's own write.
#   ARM 2  the same, with the key CUT OUT of brae's written file -- OpenFOAM must then FAIL. This is a
#          live control: it proves ARM 1 passes because the key is there, not for some other reason.
#   ARM 3  validation/rhoFR with `rho none`: OpenFOAM restarted from brae's write must compute the SAME
#          inlet velocity brae did.
#   ARM 4  the same with `rho none;` cut out -- OpenFOAM's inlet must MOVE, or ARM 3 proves nothing.
#
# FAIL-PROOF, measured in-session on the binary before the fix:
#   ARM 1  `--> FOAM FATAL IO ERROR: Entry 'intensity' not found in dictionary
#           ".../2/k/boundaryField/inlet"`, OpenFOAM exit 1.
#   ARM 3  brae ran its inlet at 6.000000000 m/s; OpenFOAM restarted from that file computed
#          5.160413015 -- 14% of the prescribed flow rate, with no message on either side.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDD="${BUILD:-$ROOT/build}"
BIN="$BUILDD/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TOL=${TOL:-1e-12}
MOVE=${MOVE:-1e-3}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
for f in rhoTI rhoFR; do
    [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }
done
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# $1 dst  $2 fixture  $3 sed on 0/U ("" = none)
stage()
{
    rm -rf "$W/$1"; cp -r "$ROOT/validation/$2" "$W/$1"
    rm -rf "$W/$1"/[1-9]* "$W/$1"/0; cp -r "$W/$1/0.orig" "$W/$1/0"
    [ -n "$3" ] && sed -i "$3" "$W/$1/0/U"
    python3 - "$W/$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', '2'), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
s = re.sub(r'\bfunctions\s*\{.*\}\s*$', '', s, flags=re.S)
open(c, 'w').write(s)
PYEOF
    ( cd "$W/$1" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on $1"; exit 1; }
}

# Restart real OpenFOAM from a directory brae has already written into. $1 dir -> prints the exit code.
ofRestart()
{
    python3 - "$W/$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]; c = os.path.join(d, 'system/controlDict'); s = open(c).read()
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom latestTime;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 3;', s)
open(c, 'w').write(s)
PYEOF
    ( cd "$W/$1" && rhoSimpleFoam > log.of 2>&1 ); echo $?
}

inletUx() { python3 -c "
import re, sys
s = open(sys.argv[1]).read(); b = s[s.find('boundaryField'):]
blk = re.search(r'\n    inlet\s*\n    \{(.*?)\n    \}', b, re.S).group(1)
u = re.search(r'value\s+uniform\s+\(([^)]*)\)', blk)
if u: print(repr(float(u.group(1).split()[0])))
else:
    v = re.search(r'value\s+nonuniform\s+List<vector>\s*\n?\d+\s*\n?\(\s*\(([^)]*)\)', blk, re.S)
    print(repr(float(v.group(1).split()[0])))" "$1"; }

# ---- ARM 1: OpenFOAM restarts from brae's write of a turbulent-inlet case -------------------------
stage ti rhoTI ''
( cd "$W/ti" && BRAE_U_SOLVER=ofOrder BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/ti" > log 2>&1 ) \
    || { tail -5 "$W/ti/log"; echo "FAIL: brae did not run on rhoTI"; exit 1; }
[ -d "$W/ti/2" ] || { echo "FAIL: brae wrote no time 2 on rhoTI"; exit 1; }
grep -q "intensity" "$W/ti/2/k" \
    && say "the written k inlet carries its \`intensity\`" ok \
    || say "the written k inlet carries its \`intensity\`" FAIL
grep -q "mixingLength" "$W/ti/2/omega" \
    && say "the written omega inlet carries its \`mixingLength\`" ok \
    || say "the written omega inlet carries its \`mixingLength\`" FAIL
cp -r "$W/ti" "$W/ti_rt"; rm -f "$W/ti_rt"/log*
rc=$(ofRestart ti_rt)
[ "$rc" = 0 ] \
    && say "real OpenFOAM restarts from brae's own write of rhoTI" ok \
    || { grep -m1 -A3 "FOAM FATAL" "$W/ti_rt/log.of" | head -4; \
         say "real OpenFOAM restarts from brae's own write of rhoTI" FAIL; }

# ---- ARM 2: LIVE CONTROL -- cut the key out and OpenFOAM must fail --------------------------------
for k in intensity mixingLength; do
    case $k in intensity) F=k ;; mixingLength) F=omega ;; esac
    rm -rf "$W/ctl_$k"; cp -r "$W/ti" "$W/ctl_$k"; rm -f "$W/ctl_$k"/log*
    sed -i "/^        $k /d" "$W/ctl_$k/2/$F"
    grep -q "$k" "$W/ctl_$k/2/$F" && { say "control: $k really was removed from the written file" FAIL; }
    rc=$(ofRestart "ctl_$k")
    [ "$rc" != 0 ] && grep -q "$k" "$W/ctl_$k/log.of" \
        && say "control: without \`$k\` OpenFOAM cannot read brae's file" ok \
        || say "control: without \`$k\` OpenFOAM cannot read brae's file" FAIL
done

# ---- ARM 3: `rho none` must survive the round trip, by VALUE and not just by presence -------------
stage rn rhoFR 's|massFlowRate constant 0.06; rhoInlet 1.0;|massFlowRate constant 0.06; rho none;|'
grep -q "rho none" "$W/rn/0/U" || { echo "FAIL: the rho-none mutation did not apply"; exit 1; }
( cd "$W/rn" && BRAE_U_SOLVER=ofOrder "$BIN" -case "$W/rn" > log 2>&1 ) \
    || { tail -5 "$W/rn/log"; echo "FAIL: brae did not run on the rho-none case"; exit 1; }
grep -q "rho  *none" "$W/rn/2/U" \
    && say "the written U inlet carries its \`rho none\`" ok \
    || say "the written U inlet carries its \`rho none\`" FAIL
braeUx=$(inletUx "$W/rn/2/U")
cp -r "$W/rn" "$W/rn_rt"; rm -f "$W/rn_rt"/log*
rc=$(ofRestart rn_rt)
[ "$rc" = 0 ] || { tail -5 "$W/rn_rt/log.of"; say "OpenFOAM restarts from brae's rho-none write" FAIL; }
ofUx=$(inletUx "$W/rn_rt/3/U")
python3 -c "
import sys
b, o, tol = float('$braeUx'), float('$ofUx'), float('$TOL')
d = abs(b - o) / abs(b)
print('     brae inlet %.9f   OpenFOAM restarted from it %.9f   rel %.3e (bound %.1e)   %s'
      % (b, o, d, tol, 'ok' if d < tol else 'FAIL'))
sys.exit(0 if d < tol else 1)" || fail=1

# ---- ARM 4: LIVE CONTROL -- drop `rho none` and OpenFOAM's inlet must move ------------------------
rm -rf "$W/ctl_rho"; cp -r "$W/rn" "$W/ctl_rho"; rm -f "$W/ctl_rho"/log*
sed -i '/^        rho  *none;/d' "$W/ctl_rho/2/U"
grep -q "rho  *none" "$W/ctl_rho/2/U" && say "control: rho none really was removed" FAIL
rc=$(ofRestart ctl_rho)
[ "$rc" = 0 ] || { tail -5 "$W/ctl_rho/log.of"; say "control: OpenFOAM runs without the rho entry" FAIL; }
ctlUx=$(inletUx "$W/ctl_rho/3/U")
python3 -c "
import sys
b, c, mv = float('$braeUx'), float('$ctlUx'), float('$MOVE')
d = abs(b - c) / abs(b)
print('     control: without \`rho none\` OpenFOAM reads %.9f instead of %.9f, rel %.3e (needs > %.1e)   %s'
      % (c, b, d, mv, 'ok' if d > mv else 'FAIL (the entry is inert; ARM 3 proves nothing)'))
sys.exit(0 if d > mv else 1)" || fail=1

say "brae's written boundary conditions round-trip through real OpenFOAM" "$([ $fail = 0 ] && echo ok || echo FAIL)"
exit $fail
