#!/usr/bin/env bash
# The gate: what a function object entry brae cannot honour does. Refuse rather than silently substitute -- an
# entry OpenFOAM itself stops on stops brae with the same cause, an entry OpenFOAM runs and brae has not
# ported stops brae BY NAME, and an object of a TYPE that is not ported is the one thing left out without
# stopping: the run goes on and says so.
# The row is laminar/damBreak staged to t = STOP with one object a run; ../lib.sh's staging, the entry below.
#   OPENFOAM STOPS TOO (its own log is the oracle):
#     vectorNotFound  U at a location no cell holds: a result of the row that is a vector of -VGREAT is not
#                     an entry the state dictionary can hold, "ill defined primitiveEntry starting at keyword
#                     'average(U)'" -- 'min(U)' when another location does have a cell
#     badControl      writeControl with a word timeControl.C:39-51 does not hold
#     noInterval      writeControl runTime with no writeInterval (timeControl.C:157)
#     noType          no `type` (functionObject::New)
#     noFields        probes with no `fields` (probes.C:402)
#   OPENFOAM RUNS IT, brae does not yet:
#     surfaceField    fields (phi)                       pattern        fields ("p.*")
#     cellPoint       interpolationScheme cellPoint      trigger        controlMode trigger
#     clockTime       writeControl clockTime
# Three checks. (1) Each of the ten stops brae's host loop with its own words and leaves no time directory;
# one of them on the GPU loop too, which builds the objects with the same code. (2) THE ORACLE: OpenFOAM stops
# on the first five and runs the other five to the end. (3) THE OTHER SIDE: an object of a type that is not
# ported (fieldMinMax) beside a probes object does NOT stop the run -- it is named as not run and the probes
# file is written -- and `interpolationScheme cell`, the ported twin of a refused entry, runs.
# MEASURED 2026-10-08: ten entries stop brae with their own words and no time directory; OpenFOAM v2412 stops
# on five of them (FOAM FATAL IO ERROR, no End) and runs the other five to End; the mixed case exits 0 with
# six rows. FOUND while writing it: OpenFOAM names 'average(U)' when no location has a cell and 'min(U)' when
# another has, and brae's message now names the same one. 5 s.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
. "$(dirname "$0")/_common.sh"
STOP=0.01
name="function object entries brae cannot honour"
base="$W/rf_base"
stage "$LAM" "$base" "$STOP" adjustable 0.05 0 > "$W/rf_stage.txt" 2>&1 \
    || { say "the case staged" FAIL; finish "$name"; }
probe="type probes; libs (sampling); probeLocations ((0.1 0.1 0.0073));"
entry()   # entry <arm>: the object's dictionary body
{
    case "$1" in
        vectorNotFound) echo "type probes; libs (sampling); fields (U); probeLocations ((5 5 5));" ;;
        badControl)     echo "$probe fields (p); writeControl sometimes;" ;;
        noInterval)     echo "$probe fields (p); writeControl runTime;" ;;
        noType)         echo "libs (sampling); fields (p); probeLocations ((0.1 0.1 0.0073));" ;;
        noFields)       echo "$probe" ;;
        surfaceField)   echo "$probe fields (phi);" ;;
        pattern)        echo "$probe fields (\"p.*\");" ;;
        cellPoint)      echo "$probe fields (p); interpolationScheme cellPoint;" ;;
        trigger)        echo "$probe fields (p); controlMode trigger; triggerStart 1;" ;;
        clockTime)      echo "$probe fields (p); writeControl clockTime; writeInterval 1;" ;;
        cell)           echo "$probe fields (p); interpolationScheme cell;" ;;
    esac
}
words()   # words <arm>: what brae's refusal must say
{
    case "$1" in
        vectorNotFound) echo "keyword 'average(U)'" ;;
        badControl)     echo "writeControl sometimes\`, which is not one of" ;;
        noInterval)     echo "no \`writeInterval\`" ;;
        noType)         echo "has no \`type\`" ;;
        noFields)       echo "needs \`probeLocations\` and \`fields\`" ;;
        surfaceField)   echo "samples \`phi\`, which this solver does not hand a function object" ;;
        pattern)        echo "has the pattern" ;;
        cellPoint)      echo "interpolationScheme cellPoint" ;;
        trigger)        echo "controlMode trigger" ;;
        clockTime)      echo "writeControl clockTime" ;;
    esac
}
case_of()   # case_of <dir> <arm> [second object's body]: the base with that object as `functions`
{
    rm -rf "${1:?}"
    cp -r "$base" "$1"
    python3 - "$1/system/controlDict" "$(entry "$2")" "${3:-}" <<'PY'
import sys
p, body, second = sys.argv[1:4]
s = open(p).read()
assert s.count('functions\n{\n}') == 1
text = 'functions\n{\n    object\n    {\n        %s\n    }\n' % body
if second:
    text += '    other\n    {\n        %s\n    }\n' % second
open(p, 'w').write(s.replace('functions\n{\n}', text + '}', 1))
PY
}
brae_stops()   # brae_stops <dir> <arm> <host|device>: 0 if brae stopped with the arm's words and wrote nothing
{
    local flag=""
    [ "$3" = device ] && flag="-device"
    ( cd "$1" && env BRAE_X=1 "$BIN" -case . $flag > log.brae 2>&1 )
    local rc=$?
    [ $rc -ne 0 ] && grep -aqF -- "$(words "$2")" "$1/log.brae" && [ -z "$(timedirs "$1")" ]
}
stops="vectorNotFound badControl noInterval noType noFields"
unported="surfaceField pattern cellPoint trigger clockTime"
bad=""
for a in $stops $unported; do
    case_of "$W/rf_$a" "$a"
    brae_stops "$W/rf_$a" "$a" host || bad="$bad $a"
done
case_of "$W/rf_dev" cellPoint
brae_stops "$W/rf_dev" cellPoint device || bad="$bad cellPoint(device)"
what="each of ten entries stops the host loop with its own words and leaves no time directory, one on the"
what="$what GPU loop too [not so:${bad:- none}]"
[ -z "$bad" ] && say "$what" ok || say "$what" FAIL
bad=""
for a in $stops $unported; do
    case_of "$W/rf_of_$a" "$a"
    ( cd "$W/rf_of_$a" && interFoam > log.interFoam 2>&1 )
    fatal=$(grep -c "FOAM FATAL" "$W/rf_of_$a/log.interFoam")
    ended=$(grep -c "^End" "$W/rf_of_$a/log.interFoam")
    case " $stops " in
        *" $a "*) [ "$fatal" -ge 1 ] && [ "$ended" = 0 ] || bad="$bad $a(ran)" ;;
        *)        [ "$fatal" = 0 ] && [ "$ended" = 1 ] || bad="$bad $a(stopped)" ;;
    esac
done
grep -q "ill defined primitiveEntry starting at keyword 'average(U)'" "$W/rf_of_vectorNotFound/log.interFoam" \
    || bad="$bad vectorNotFound(other cause)"
what="ORACLE  OpenFOAM stops on the five brae stops on with it, average(U) by name, and runs the five brae has"
what="$what not ported [not so:${bad:- none}]"
[ -z "$bad" ] && say "$what" ok || say "$what" FAIL
case_of "$W/rf_mixed" cell "type fieldMinMax; libs (fieldFunctionObjects); fields (U);"
( cd "$W/rf_mixed" && env BRAE_X=1 "$BIN" -case . > log.brae 2>&1 )
rc=$?
rows=$(grep -vc '^#' "$W/rf_mixed/postProcessing/object/0/p" 2>/dev/null || echo 0)
what="THE OTHER SIDE  an unported type beside a probes object: exit $rc, fieldMinMax named as not run, the"
what="$what probes file's $rows rows written with \`interpolationScheme cell\`"
[ $rc -eq 0 ] && grep -aq "functions/other.*fieldMinMax.*NOT run" "$W/rf_mixed/log.brae" && [ "$rows" -ge 3 ] \
    && [ ! -e "$W/rf_mixed/postProcessing/other" ] && say "$what" ok || say "$what" FAIL
finish "an entry OpenFOAM stops on stops brae, an unported entry is refused by name, an unported type is said"
