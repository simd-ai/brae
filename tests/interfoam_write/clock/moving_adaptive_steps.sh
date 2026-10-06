#!/usr/bin/env bash
# The write gate: a MOVING mesh run AS SHIPPED -- adjustTimeStep, the tutorial's own tolerances -- takes serial
# OpenFOAM's time steps. laminar/waves/waveMakerPiston to t = 0.74: 65 steps, the first ten growing by the
# control's 1.2, the rest set by the Courant number of the flux relative to the moving mesh.
# WHY IT EXISTS: the 42-tutorial table read this row as 0.27x of 20-core OpenFOAM "over the same interval",
# brae taking 66 steps to t = 0.740921 for OpenFOAM's 30. The 30 are the 20-CORE run's. Serial OpenFOAM takes
# 66 there (MEASURED 2026-10-06), and with the case's own pressure solver every one of brae's 66 time steps is
# serial OpenFOAM's to 3.5e-09. The 20-core run reaches t = 0.741 in the 30 steps that take the serial run to
# 0.499: it is another trajectory, and the oracle here is the serial one. Every other gate on this case pins
# its steps (adjustTimeStep no), so nothing held the adaptive run.
# HELD: (1) with the case's own PCG + DIC, the step count and each time step against OpenFOAM's log (its
# `deltaT =` lines at writePrecision 17); (2) on the default path, the AMG-PCG, the step count within one --
# another Krylov method stopped at the entry's tolerances is not the same number (MEASURED: a time step up to
# 4.8e-02 off, 4.3e-03 off at the last; to 0.74 it takes 66 steps for 65, to 0.740921 66 for 66) -- and its
# notice.
# The CONTROL leaves out setDeltaT.H's cap of 1.2 on a step's growth: from rest the first step is then
# maxDeltaT (0.05 for 0.006), and the series has to miss.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
src="$TUT/multiphase/interFoam/laminar/waves/waveMakerPiston"
stage_allrun "$src" "$W/ma_stage" "" > "$W/ma_stage.txt" 2>&1 \
    || { say "waveMakerPiston did not stage" FAIL; finish "a moving mesh's adaptive time steps"; }
arm()
{
    local d="$W/ma_$1"
    rm -rf "${d:?}"
    mkdir -p "$d"
    cp -r "$W/ma_stage/0" "$W/ma_stage/constant" "$W/ma_stage/system" "$d/"
    cp "$src/system/fvSolution" "$d/system/fvSolution"
    python3 - "$src/system/controlDict" "$d/system/controlDict" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}', '\nfunctions\n{\n}', s, flags=re.S)
for k, v in [('writeControl', 'timeStep'), ('writeInterval', '1000000'), ('startFrom', 'startTime'),
             ('stopAt', 'endTime'), ('endTime', '0.74'), ('writePrecision', '17')]:
    s, n = re.subn(r'^%s\s.*' % k, '%s %s;' % (k.ljust(15), v), s, flags=re.M)
    assert n == 1, k
open(sys.argv[2], 'w').write(s)
PY
}
# series <brae log> <OpenFOAM log>: brae's steps, OpenFOAM's, the widest |dt - dt_OpenFOAM|/dt_OpenFOAM over the
# steps both made, and brae's first time step
series()
{
    python3 - "$1" "$2" <<'PY'
import re, sys
bd = [float(m.group(1)) for l in open(sys.argv[1], errors='replace')
      for m in [re.match(r'^\s*t = \S+\s+dt = (\S+)\s+Co ', l)] if m]
od = [float(m.group(1)) for l in open(sys.argv[2], errors='replace')
      for m in [re.match(r'^deltaT = (\S+)', l)] if m]
n = min(len(bd), len(od))
worst = max([abs(bd[i] - od[i])/od[i] for i in range(n)] or [9.0])
print(len(bd), len(od), '%.3e' % worst, '%.6g' % (bd[0] if bd else 0))
PY
}
arm of
runof "$W/ma_of"
arm case
runbrae "$W/ma_case" device
arm amg
runbrae "$W/ma_amg" device -u BRAE_PRESSURE_CASE_SOLVER
arm control
runbrae "$W/ma_control" device BRAE_CONTROL_DELTAT_GROWTH_UNCAPPED=1
notice="brae runs its AMG-preconditioned PCG"
read -r nb no gap first <<< "$(series "$W/ma_case/log.brae" "$W/ma_of/log.interFoam")"
what="[device] the case's own PCG+DIC: $nb adaptive steps for serial OpenFOAM's $no, each within $gap (4e-08)"
[ "$no" -ge 60 ] && [ "$nb" = "$no" ] && within "$gap" 4e-08 && ! grep -q "$notice" "$W/ma_case/log.brae" \
    && say "$what" ok || say "$what" FAIL
read -r nb no gap first <<< "$(series "$W/ma_amg/log.brae" "$W/ma_of/log.interFoam")"
what="[device] the AMG-PCG, the default: $nb steps for serial OpenFOAM's $no (within one); widest step gap $gap"
[ "$nb" -le $((no + 1)) ] && [ "$nb" -ge $((no - 1)) ] && within "$gap" 0.5 && grep -q "$notice" "$W/ma_amg/log.brae" \
    && say "$what" ok || say "$what" FAIL
read -r nb no gap first <<< "$(series "$W/ma_control/log.brae" "$W/ma_of/log.interFoam")"
what="CONTROL  the cap of 1.2 left out: the first step is $first for OpenFOAM's 0.006, the series $gap off"
grep -q "CONTROL MODE: the time-step control leaves out" "$W/ma_control/log.brae" && above "$gap" 1e-02 \
    && say "$what" ok || say "$what" FAIL
finish "a moving mesh run as shipped takes serial OpenFOAM's time steps"
