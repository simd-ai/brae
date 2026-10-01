#!/usr/bin/env bash
# The write gate on damBreak: a start directory brae cannot continue, and a refusal inside write() (arms E0, E1).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
core_base
# E0: a start directory holding a written field's `_0` level. OpenFOAM reads it as an AUTO_WRITE old time
# (readOldTimeIfPresent, GeometricField.C:120, :131-160) and writes it back at every write time; brae
# writes only a sub-cycled alpha's, so a U_0 there must be named at startup and the run stopped at its
# first write. CONTROL: the identical restart without U_0 runs to its end.
restart_brae()   # restart_brae <dir> [with U_0]
{
    mkdir -p "$1"
    cp -r "$W/of/constant" "$W/of/system" "$W/br_host/0.1" "$1/"
    if [ "${2:-}" = withU0 ]; then
        sed 's/^\( *object *\)U;/\1U_0;/' "$1/0.1/U" > "$1/0.1/U_0"
    fi
    sed -i 's/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         0.15;/' "$1/system/controlDict"
    ( cd "$1" && stdbuf -oL -eL "$BIN" -case . > log.brae 2>&1 )
}
restart_brae "$W/r0" withU0; rc=$?
[ $rc -ne 0 ] && grep -q "U_0 will not be written" "$W/r0/log.brae" && [ "$(timedirs "$W/r0")" = "0.1 " ] \
    && say "ARM E0 a start directory's U_0 is named and the run stops at its first write" ok \
    || { say "ARM E0 a start directory's U_0 is named and the run stops at its first write" FAIL; tail -3 "$W/r0/log.brae" | sed 's/^/      /'; }
restart_brae "$W/r0ctl" && [ "$(timedirs "$W/r0ctl")" = "0.1 0.15 " ] \
    && say "CONTROL  the same restart without U_0 runs to its end and writes 0.15" ok \
    || { say "CONTROL  the same restart without U_0 runs to its end and writes 0.15" FAIL; tail -3 "$W/r0ctl/log.brae" | sed 's/^/      /'; }

# E1: a refusal thrown INSIDE write() leaves no time directory -- every file is built into a queue and
# only then written. No case reaches such a refusal (the start-up check mirrors each one), so
# BRAE_CONTROL_WRITE_REFUSE_LATE=1 injects it after the last file is built. The restart is E0's; its
# control, which writes 0.15, is the one above. FAIL-PROOF (2026-09-30): with emit() writing at once,
# the writer's form before the queue, this run left 0.15/ holding all 9 of its files.
( export BRAE_CONTROL_WRITE_REFUSE_LATE=1; restart_brae "$W/r1" ); rc=$?
[ $rc -ne 0 ] && grep -q "BRAE_CONTROL_WRITE_REFUSE_LATE refuses" "$W/r1/log.brae" && [ "$(timedirs "$W/r1")" = "0.1 " ] \
    && say "ARM E1 a refusal inside write() leaves no time directory behind" ok \
    || { say "ARM E1 a refusal inside write() leaves no time directory behind [$(timedirs "$W/r1")]" FAIL; tail -3 "$W/r1/log.brae" | sed 's/^/      /'; }
finish "arms E0, E1: named at start-up, stopped at the first write, nothing half-written"
