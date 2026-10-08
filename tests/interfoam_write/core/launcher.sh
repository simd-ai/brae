#!/usr/bin/env bash
# The gate: `brae <case>` on an interFoam case runs the GPU loop. `brae` is the one command a user types; it
# reads controlDict's `application` and hands the case to the solver's own binary (solver_dispatch.cuh).
# brae_interFoam holds TWO loops -- the host reference the gates hold to OpenFOAM, and the GPU one, taken with
# `-device` -- and the hand-over forwarded the user's arguments alone: the registry had the row, the case was
# handed over, and it ran on one CPU core. A bare case directory (`brae myCase`) was not read by brae_interFoam
# either: it ran the current directory. FOUND 2026-10-08, the first time the launcher was run on an interFoam
# case rather than on a dictionary-only fake.
# The row is laminar/damBreak, ten steps of 0.001, one write. The ORACLE is brae_interFoam's own GPU run of
# the same case -- that loop is held to OpenFOAM by every other gate here; what is under test is which loop the
# launcher starts, and on which directory.
# Three checks: (1) `brae <bare directory>`, run from the directory above: the hand-over names `-device`, every
# step line is the GPU loop's, and the written files are the direct GPU run's BYTES; (2) `brae -case <dir>` the
# same; (3) CONTROL the host loop's lines carry no such tag and its files are other bytes -- so (1) and (2)
# can fail -- and an option brae_interFoam does not have is refused by name.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device loop"; exit 77; }
LAUNCH="$(dirname "$BIN")/brae"
[ -x "$LAUNCH" ] || { echo "SKIP: no brae launcher beside $BIN"; exit 77; }
STEPS=10
name="brae on an interFoam case"
s="$W/la_staged"
stage "$LAM" "$s" 0.01 timeStep $STEPS 0 deltaT=0.001 adjustTimeStep=no > "$W/la_stage.txt" 2>&1 \
    || { say "laminar/damBreak did not stage" FAIL; finish "$name"; }
grep -q "^application  *interFoam;" "$s/system/controlDict" \
    || { say "PREMISE  the case names interFoam as its application" FAIL; finish "$name"; }
# copy <name>: a fresh copy of the staged case
copy()
{
    rm -rf "${W:?}/$1"
    mkdir -p "$W/$1"
    cp -r "$s/0" "$s/constant" "$s/system" "$W/$1/"
}
# sums <dir>: one line a written file under the time directories, its path and its bytes' sum
sums() { ( cd "$1" && find $(timedirs .) -type f | sort | xargs md5sum 2>/dev/null ); }
# tagged <log>: "<step lines> <of them the GPU loop's>"
tagged() { echo "$(grep -ac '^ *t = .* dt = ' "$1") $(grep -a '^ *t = .* dt = ' "$1" | grep -c '\[device\]')"; }
copy la_direct
runbrae "$W/la_direct" device BRAE_X=1
copy la_bare
( cd "$W" && "$LAUNCH" la_bare > la_bare/log.brae 2>&1 ) || { say "\`brae la_bare\` ran" FAIL; finish "$name"; }
copy la_case
"$LAUNCH" -case "$W/la_case" > "$W/la_case/log.brae" 2>&1 || { say "\`brae -case\` ran" FAIL; finish "$name"; }
copy la_host
runbrae "$W/la_host" host BRAE_X=1
sums "$W/la_direct" > "$W/la_direct.sums"
nfiles=$(wc -l < "$W/la_direct.sums")
read -r nd td <<< "$(tagged "$W/la_direct/log.brae")"
[ "$nd" = "$STEPS" ] && [ "$td" = "$STEPS" ] && [ "$nfiles" -ge 6 ] \
    || { say "PREMISE  the direct GPU run took $STEPS steps and wrote files [$nd $td $nfiles]" FAIL; finish "$name"; }
for arm in bare case; do
    sums "$W/la_$arm" > "$W/la_$arm.sums"
    read -r n t <<< "$(tagged "$W/la_$arm/log.brae")"
    differ=$(diff "$W/la_direct.sums" "$W/la_$arm.sums" | grep -c "^<")
    [ "$arm" = bare ] && how="brae <bare directory>" || how="brae -case <dir>"
    what="[$how] $t of $n step lines are the GPU loop's, $differ of $nfiles written files differ from the"
    what="$what direct GPU run's bytes"
    [ "$n" = "$STEPS" ] && [ "$t" = "$STEPS" ] && [ "$differ" = 0 ] \
        && grep -aq "application interFoam -> interFoam (brae_interFoam -device)" "$W/la_$arm/log.brae" \
        && say "$what" ok || say "$what" FAIL
done
sums "$W/la_host" > "$W/la_host.sums"
read -r nh th <<< "$(tagged "$W/la_host/log.brae")"
hdiffer=$(diff "$W/la_direct.sums" "$W/la_host.sums" | grep -c "^<")
copy la_option
( cd "$W/la_option" && "$BIN" -case . -device -notAnOption > log.brae 2>&1 )
rc=$?
what="CONTROL  the host loop: $th of $nh step lines tagged, $hdiffer written files of other bytes; an option the"
what="$what solver does not have: exit $rc"
[ "$nh" = "$STEPS" ] && [ "$th" = 0 ] && [ "$hdiffer" -ge 2 ] && [ $rc -ne 0 ] \
    && grep -aq "the option \`-notAnOption\` is not one this solver takes" "$W/la_option/log.brae" \
    && ! grep -aq '^ *t = .* dt = ' "$W/la_option/log.brae" && say "$what" ok || say "$what" FAIL
finish "\`brae <case>\` hands an interFoam case to the GPU loop, by either spelling of the directory"
