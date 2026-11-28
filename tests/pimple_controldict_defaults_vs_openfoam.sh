#!/usr/bin/env bash
# pimpleFoam's controlDict deltaT and end of run against OpenFOAM v2412: what OpenFOAM requires, and a run with nothing to do.
#
# WHAT IT HOLDS. Time::readDict requires deltaT (TimeIO.C:274, readEntry). Under `stopAt endTime` it requires endTime
# (:342-346); with no stopAt the end is endTime if present, else 0 (:353-356); any other stopAt sets GREAT (:350) and stops
# the run through Time::operator++ (Time.C:1166-1180). A run whose end is not after its start takes no step, executes no
# functionObject and writes nothing (Time::run, Time.C:784-791). brae ran deltaT 1e-3 and endTime 1 in place of the missing
# entries, refused an end before the start, rewrote the start directory of a restart from past the end, and ran a
# `stopAt writeNow` to endTime with a notice.
#
# THE CASE. incompressible/pimpleFoam/RAS/pitzDaily, at a fixed deltaT 1e-4 written every step but for D6.
#
# ARMS, each with OpenFOAM's own behaviour as its premise:
#   D1  no deltaT: OpenFOAM stops naming deltaT; brae stops naming deltaT and writes nothing
#   D2  no stopAt and no endTime: OpenFOAM ends cleanly with no time step and nothing written; brae likewise
#   D3  `stopAt endTime` and no endTime: OpenFOAM stops naming endTime; brae likewise
#   D4  a restart from 0.0003 with endTime 0.0002: OpenFOAM ends cleanly, the start directory unchanged; brae likewise
#   D5  `stopAt writeNow`: OpenFOAM runs one step and writes it; brae, which does not port it, refuses it by name
#   D6  D2 under adjustTimeStep (the case as shipped): OpenFOAM ends cleanly with nothing written; brae likewise
# CONTROL BRAE_CONTROL_CONTROLDICT_DEFAULTS=1 (deltaT 1e-3 and endTime 1 when absent, as before): D1 and D2 must go red.
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28, job 5863619): every premise holds. Before (PR #106's
# a8cce3a6): D1 ran on deltaT 1e-3, D2 and D3 still running toward endTime 1 at the 60 s limit, D4 rewrote 0.0003, D5 ran to
# endTime, D6 stopped on `controlDict has no endTime`; after, every arm as OpenFOAM. The control runs D1 and D2 as before.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
LIMIT=${BRAE_CD_LIMIT:-60}   # seconds a run that should end at once may take before it counts as running on
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUT="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}/incompressible/pimpleFoam/RAS/pitzDaily"
[ -d "$TUT/system" ] || { echo "SKIP: no pimpleFoam/RAS/pitzDaily tutorial at $TUT"; exit 77; }
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
W=${BRAE_CD_KEEP:-$(mktemp -d)}
[ -n "${BRAE_CD_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-96s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

setkey()
{
    if grep -q -E "^$2[[:space:]]" "$1/system/controlDict"; then
        sed -i -E "s/^$2[[:space:]][^;]*;/$2 $3;/" "$1/system/controlDict"
    else
        printf '%-16s%s;\n' "$2" "$3" >> "$1/system/controlDict"
    fi
    grep -q -E "^$2[[:space:]]+$3;" "$1/system/controlDict" || { echo "FAIL: could not set $2 $3 in $1"; exit 1; }
}
dropkey() { sed -i -E "/^$2[[:space:]]/d" "$1/system/controlDict"; ! grep -q -E "^$2[[:space:]]" "$1/system/controlDict"; }
# run <runner> <dir>: the exit code, 124 when it was still running after LIMIT seconds
ofrun() { ( cd "$1" && timeout "$LIMIT" pimpleFoam > log 2>&1 ); echo $?; }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" timeout "$LIMIT" "$BIN" -case . > log 2>&1 ); echo $?; }
written() { (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -vx 0 | sort -g | tr '\n' ' ' | sed 's/ $//'); }
named() { grep -a -q -i -E "$2" "$1/log"; }

mkdir -p "$W/base"
cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
grep -q -E "^adjustTimeStep[[:space:]]+yes;" "$W/base/system/controlDict" \
    || { echo "FAIL: pitzDaily no longer ships adjustTimeStep yes -- D6's premise moved"; exit 1; }
grep -q -E "^stopAt" "$W/base/system/controlDict" || { echo "FAIL: pitzDaily no longer names stopAt"; exit 1; }
stage() { rm -rf "$1"; mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; }
fixed()
{
    stage "$1"
    setkey "$1" adjustTimeStep no
    setkey "$1" deltaT 0.0001
    setkey "$1" endTime 0.0003
    setkey "$1" writeControl timeStep
    setkey "$1" writeInterval 1
}
echo "pimple_controldict_defaults_vs_openfoam: pitzDaily, controlDict deltaT and end of run"

# D1: no deltaT
for c in of br; do fixed "$W/d1-$c"; dropkey "$W/d1-$c" deltaT || { echo "FAIL: deltaT not removed"; exit 1; }; done
rof=$(ofrun "$W/d1-of"); rbr=$(brrun "$W/d1-br")
[ "$rof" != 0 ] && [ "$rof" != 124 ] && named "$W/d1-of" deltaT && say "premise D1: OpenFOAM stops naming deltaT" PASS \
    || say "premise D1: OpenFOAM stops naming deltaT (rc $rof)" FAIL
[ "$rbr" != 0 ] && [ "$rbr" != 124 ] && named "$W/d1-br" deltaT && [ -z "$(written "$W/d1-br")" ] \
    && say "D1: brae stops naming deltaT and writes nothing" PASS \
    || say "D1: brae stops naming deltaT and writes nothing (rc $rbr, wrote: $(written "$W/d1-br"))" FAIL

# D2: no stopAt and no endTime
for c in of br; do fixed "$W/d2-$c"; dropkey "$W/d2-$c" stopAt && dropkey "$W/d2-$c" endTime || { echo "FAIL: D2 staging"; exit 1; }; done
rof=$(ofrun "$W/d2-of"); rbr=$(brrun "$W/d2-br")
[ "$rof" = 0 ] && [ -z "$(written "$W/d2-of")" ] && say "premise D2: OpenFOAM ends cleanly with nothing written" PASS \
    || say "premise D2: OpenFOAM ends cleanly with nothing written (rc $rof, wrote: $(written "$W/d2-of"))" FAIL
[ "$rbr" = 0 ] && [ -z "$(written "$W/d2-br")" ] && say "D2: brae ends cleanly with nothing written" PASS \
    || say "D2: brae ends cleanly with nothing written (rc $rbr, wrote: $(written "$W/d2-br" | cut -c1-40))" FAIL

# D3: stopAt endTime and no endTime
for c in of br; do fixed "$W/d3-$c"; setkey "$W/d3-$c" stopAt endTime; dropkey "$W/d3-$c" endTime || { echo "FAIL: D3 staging"; exit 1; }; done
rof=$(ofrun "$W/d3-of"); rbr=$(brrun "$W/d3-br")
[ "$rof" != 0 ] && [ "$rof" != 124 ] && named "$W/d3-of" endTime && say "premise D3: OpenFOAM stops naming endTime" PASS \
    || say "premise D3: OpenFOAM stops naming endTime (rc $rof)" FAIL
[ "$rbr" != 0 ] && [ "$rbr" != 124 ] && named "$W/d3-br" endTime && [ -z "$(written "$W/d3-br")" ] \
    && say "D3: brae stops naming endTime and writes nothing" PASS \
    || say "D3: brae stops naming endTime and writes nothing (rc $rbr, wrote: $(written "$W/d3-br" | cut -c1-40))" FAIL

# D4: a restart from past the end
fixed "$W/d4-src"; r=$(ofrun "$W/d4-src"); [ "$r" = 0 ] || { echo "FAIL: OpenFOAM to 0.0003 (rc $r)"; exit 1; }
T3=$(written "$W/d4-src" | tr ' ' '\n' | tail -n 1)
for c in of br; do
    fixed "$W/d4-$c"; cp -r "$W/d4-src/$T3" "$W/d4-$c/"; setkey "$W/d4-$c" startFrom latestTime; setkey "$W/d4-$c" endTime 0.0002
    ( cd "$W/d4-$c/$T3" && find . -type f -exec md5sum {} + | sort ) > "$W/d4-$c.before"
done
rof=$(ofrun "$W/d4-of"); rbr=$(brrun "$W/d4-br")
same() { ( cd "$1/$T3" && find . -type f -exec md5sum {} + | sort ) | cmp -s - "$1.before"; }
[ "$rof" = 0 ] && [ "$(written "$W/d4-of")" = "$T3" ] && same "$W/d4-of" \
    && say "premise D4: OpenFOAM from $T3 with endTime 0.0002 ends cleanly, $T3 unchanged" PASS \
    || say "premise D4: OpenFOAM from $T3 with endTime 0.0002 ends cleanly, $T3 unchanged (rc $rof)" FAIL
[ "$rbr" = 0 ] && [ "$(written "$W/d4-br")" = "$T3" ] && same "$W/d4-br" \
    && say "D4: brae from $T3 with endTime 0.0002 ends cleanly, $T3 unchanged" PASS \
    || say "D4: brae from $T3 with endTime 0.0002 ends cleanly, $T3 unchanged (rc $rbr, wrote: $(written "$W/d4-br" | cut -c1-40))" FAIL

# D5: stopAt writeNow
for c in of br; do fixed "$W/d5-$c"; setkey "$W/d5-$c" stopAt writeNow; done
rof=$(ofrun "$W/d5-of"); rbr=$(brrun "$W/d5-br")
[ "$rof" = 0 ] && [ "$(written "$W/d5-of" | wc -w)" = 1 ] && say "premise D5: OpenFOAM runs one step and writes it ($(written "$W/d5-of"))" PASS \
    || say "premise D5: OpenFOAM runs one step and writes it (rc $rof, wrote: $(written "$W/d5-of"))" FAIL
[ "$rbr" != 0 ] && [ "$rbr" != 124 ] && named "$W/d5-br" "stopAt writeNow" && [ -z "$(written "$W/d5-br")" ] \
    && say "D5: brae refuses stopAt writeNow by name and writes nothing" PASS \
    || say "D5: brae refuses stopAt writeNow by name and writes nothing (rc $rbr, wrote: $(written "$W/d5-br" | cut -c1-40))" FAIL

# D6: D2 under adjustTimeStep (the case as shipped)
for c in of br; do stage "$W/d6-$c"; dropkey "$W/d6-$c" stopAt && dropkey "$W/d6-$c" endTime || { echo "FAIL: D6 staging"; exit 1; }; done
rof=$(ofrun "$W/d6-of"); rbr=$(brrun "$W/d6-br")
[ "$rof" = 0 ] && [ -z "$(written "$W/d6-of")" ] && say "premise D6: OpenFOAM under adjustTimeStep ends cleanly with nothing written" PASS \
    || say "premise D6: OpenFOAM under adjustTimeStep ends cleanly with nothing written (rc $rof)" FAIL
[ "$rbr" = 0 ] && [ -z "$(written "$W/d6-br")" ] && say "D6: brae under adjustTimeStep ends cleanly with nothing written" PASS \
    || say "D6: brae under adjustTimeStep ends cleanly with nothing written (rc $rbr, wrote: $(written "$W/d6-br" | cut -c1-40))" FAIL

# CONTROL: the old substitutions
fixed "$W/c1"; dropkey "$W/c1" deltaT; r=$(brrun "$W/c1" BRAE_CONTROL_CONTROLDICT_DEFAULTS=1)
{ [ "$r" = 0 ] || [ "$r" = 124 ]; } && say "control BRAE_CONTROL_CONTROLDICT_DEFAULTS=1: D1 runs on deltaT 1e-3 (rc $r)" PASS \
    || say "control BRAE_CONTROL_CONTROLDICT_DEFAULTS=1: D1 runs on deltaT 1e-3 (rc $r)" FAIL
fixed "$W/c2"; dropkey "$W/c2" stopAt; dropkey "$W/c2" endTime; r=$(brrun "$W/c2" BRAE_CONTROL_CONTROLDICT_DEFAULTS=1)
{ [ "$r" = 124 ] || [ -n "$(written "$W/c2")" ]; } && say "control BRAE_CONTROL_CONTROLDICT_DEFAULTS=1: D2 runs toward endTime 1 (rc $r)" PASS \
    || say "control BRAE_CONTROL_CONTROLDICT_DEFAULTS=1: D2 runs toward endTime 1 (rc $r)" FAIL

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
