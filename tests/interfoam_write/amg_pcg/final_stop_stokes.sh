#!/usr/bin/env bash
# A RECORDED MEASUREMENT WITH ITS ORDER HELD, not a gate of the default pressure path: what the FINAL p_rgh
# solve leaves on a wave tutorial, brae's default (its AMG-preconditioned PCG) beside OpenFOAM's own two
# answers. The case is the tutorial's with its own fvSolution and time-step control, and ONE thing replaced in
# every run: the write cadence (`timeStep`, never), so the steps are not trimmed to the tutorial's 0.033 write
# interval -- these are 27 steps to t = 1.144, not the first 27 of a shipped run.
# laminar/waves/stokesI asks its Final corrector for `GAMG` to 1e-7. At that tolerance the tutorial's result is
# its SOLVER's: OpenFOAM against OpenFOAM, with nothing changed but p_rghFinal's solver (PCG with DIC at the
# same tolerance), alpha's overshoot above 1 after 27 steps is 1.2e-02 where the shipped entry gives 6.0e-04.
# brae's AMG-PCG, which stands in for every pressure entry, lands between: 6.1e-03, on the same 27 steps.
# MEASURED 2026-10-06, and what was ruled out on the way: the stop is OpenFOAM's predicate on the same norm
# factor, taken every iteration on a double-precision recurrence (read); the double-precision cycle and the
# plain loop give the same bits as the default; two-stage Gauss-Seidel gives 3.3e-03 and a smoothed hierarchy
# 1.6e-03. What brings it under OpenFOAM's shipped answer is asking the Final solve for more: with its
# tolerance x0.1 the overshoot is 8.8e-04 (and the run takes 37 steps), x0.01 1.0e-04 on 27.
# THAT FACTOR IS NOT THE DEFAULT: over the 42 tutorials x0.01 costs 3 to 25% a step (stokesI 11.3 -> 12.8 ms,
# DTCHull 344 -> 393, capillaryRise 9.7 -> 11.3) and changes alpha's overshoot on the wave rows alone
# (runs/bench/readers/census_final_tol). It is an experiment switch, BRAE_EXPERIMENT_PRESSURE_FINAL_TOL_FACTOR,
# held here so that the choice stays a measured one.
# HELD: (1) the premise, as numbers: OpenFOAM's own two overshoots are a factor 5 apart at least; (2) the
# default path takes OpenFOAM's steps (within one) and its overshoot is not above OpenFOAM's own with a Krylov
# Final solve -- a ceiling twice today's number, which is all OpenFOAM's own spread gives; (3) with the Final
# tolerance x0.01 the overshoot is not above OpenFOAM's shipped answer, the switch's notice in that arm's log
# and absent from the default's. (The default's number is printed beside it and NOT held above OpenFOAM's: an
# improvement of the default path must not turn this red.)
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
src="$TUT/multiphase/interFoam/laminar/waves/stokesI"
# the time serial OpenFOAM's 27th step reaches as shipped
STOP=1.144067
stage_allrun "$src" "$W/fs_stage" "" > "$W/fs_stage.txt" 2>&1 \
    || { say "stokesI did not stage" FAIL; finish "the Final pressure solve on a wave tutorial"; }
# arm <name> [pcg]: the staged case as shipped to STOP, no write; `pcg` names PCG with DIC for p_rghFinal at
# the entry's own tolerance
arm()
{
    local d="$W/fs_$1"
    rm -rf "${d:?}"
    mkdir -p "$d"
    cp -r "$W/fs_stage/0" "$W/fs_stage/constant" "$W/fs_stage/system" "$d/"
    cp "$src/system/fvSolution" "$d/system/fvSolution"
    python3 - "$src/system/controlDict" "$d/system/controlDict" "$STOP" "$d/system/fvSolution" "${2:-}" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}', '\nfunctions\n{\n}', s, flags=re.S)
for k, v in [('writeControl', 'timeStep'), ('writeInterval', '1000000'), ('startFrom', 'startTime'),
             ('stopAt', 'endTime'), ('endTime', sys.argv[3])]:
    s = re.sub(r'^%s\s.*' % k, '%s %s;' % (k.ljust(15), v), s, flags=re.M)
open(sys.argv[2], 'w').write(s)
if sys.argv[5] == 'pcg':
    p = sys.argv[4]
    s = open(p).read()
    m = re.search(r'\n\s*"?p_rghFinal"?\s*\n\s*\{.*?\n\s*\}', s, flags=re.S)
    tol = re.search(r'tolerance\s+(\S+);', m.group(0)).group(1)
    new = ('\n    p_rghFinal\n    {\n        solver          PCG;\n        preconditioner  DIC;\n'
           '        tolerance       %s;\n        relTol          0;\n    }' % tol)
    open(p, 'w').write(s.replace(m.group(0), new))
PY
}
# ofexcess <dir>: OpenFOAM's last Max(alpha.water) less 1, and its step count
ofexcess()
{
    python3 - "$1/log.interFoam" <<'PY'
import re, sys
s = open(sys.argv[1], errors='replace').read()
a = re.findall(r'Max\(alpha.water\) = (\S+)', s)
print('%.3e %d' % (float(a[-1]) - 1.0, len(re.findall(r'^Time = ', s, flags=re.M))))
PY
}
# brexcess <dir>: brae's End line's alpha maximum less 1, and its step count
brexcess()
{
    python3 - "$1/log.brae" <<'PY'
import re, sys
s = open(sys.argv[1], errors='replace').read()
m = re.search(r'End: .*alpha in \[\S+ ([0-9.e+-]+)\]', s)
print('%.3e %d' % ((float(m.group(1)) - 1.0) if m else -1.0, len(re.findall(r'^\s*t = \S+\s+dt = ', s, flags=re.M))))
PY
}
arm of
runof "$W/fs_of"
arm ofpcg pcg
grep -q "preconditioner  DIC;" "$W/fs_ofpcg/system/fvSolution" \
    || { say "PREMISE  the second OpenFOAM run names PCG with DIC for p_rghFinal" FAIL; finish "the Final solve"; }
runof "$W/fs_ofpcg"
arm default
runbrae "$W/fs_default" device
arm factor
runbrae "$W/fs_factor" device BRAE_EXPERIMENT_PRESSURE_FINAL_TOL_FACTOR=0.01
read -r eg ng <<< "$(ofexcess "$W/fs_of")"
read -r ep np <<< "$(ofexcess "$W/fs_ofpcg")"
read -r ed nd <<< "$(brexcess "$W/fs_default")"
read -r ef nf <<< "$(brexcess "$W/fs_factor")"
num() { python3 -c "import sys; sys.exit(0 if ($1) else 1)"; }
what="PREMISE  OpenFOAM's own alpha overshoot after $ng steps: $eg as shipped (GAMG), $ep with PCG for p_rghFinal"
num "$eg > 0 and $ep > 5*$eg and $ng >= 20" && say "$what" ok || say "$what" FAIL
what="[device] the default pressure path: $nd steps (OpenFOAM $ng), overshoot $ed, not above OpenFOAM's own $ep"
num "abs($nd - $ng) <= 1 and 0 <= $ed <= $ep" \
    && grep -q "brae runs its AMG-preconditioned PCG" "$W/fs_default/log.brae" \
    && ! grep -q "EXPERIMENT" "$W/fs_default/log.brae" && say "$what" ok || say "$what" FAIL
what="[device] the Final tolerance x0.01: $nf steps, overshoot $ef, not above OpenFOAM's shipped $eg"
what="$what (the default's: $ed)"
num "abs($nf - $ng) <= 1 and 0 <= $ef <= $eg" \
    && grep -q "p_rgh EXPERIMENT: the Final solve's tolerance 1e-07 is taken as 1e-09" "$W/fs_factor/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the default pressure path's overshoot on stokesI is inside OpenFOAM's own solver spread"
