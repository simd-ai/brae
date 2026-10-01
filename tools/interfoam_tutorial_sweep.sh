#!/usr/bin/env bash
# WHERE EVERY interFoam TUTORIAL STANDS, host arm and device arm.
#
# Not a validation: it does not compare to OpenFOAM. It answers the prior question -- does brae RUN this
# case at all, or REFUSE it by name, or FAIL -- for all 44 cases OpenFOAM ships, so that "what is left to
# close interFoam" is a list rather than an impression. A refusal is a PASS here: it is the project's
# contract working. Only a crash, a hang or a wrong-looking finish is a gap.
#
# Each case is meshed the way its own Allrun meshes it (blockMesh, then the optional topoSet /
# createBaffles / setFields / extrudeMesh steps its system/ directory asks for), capped at a few steps,
# and run under a timeout. snappyHexMesh cases are skipped by name with the reason, not silently.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae}"
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/multiphase/interFoam
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
STEPS=${STEPS:-2}
# 600 and 600: the two multi-paddle tutorials mesh 448,000 cells and meshing them is most of the budget.
# The device arm runs their two steps in about 7 s. waveMakerMultiPaddleFlap's device arm used to read
# TIMEOUT here, and the earlier note called that a budget: it was a CRASH (free(): invalid next size, a
# stale device GAMG upload after the hierarchy was rebuilt on the moved mesh) spending its time writing
# a core. A TIMEOUT on a case that runs elsewhere is a gap to localise, not a budget to read past.
CASE_TIMEOUT=${CASE_TIMEOUT:-600}
MESH_TIMEOUT=${MESH_TIMEOUT:-600}
W=${KEEP_W:-$(mktemp -d)}
ONLY=${ONLY:-}
# ARMS="host" halves a census; ALL_REASONS=1 lists EVERY file the writer names at startup under a refused
# case -- the verdict line shows only the first, and a moving case is refused for several
ARMS=${ARMS:-host device}
ALL_REASONS=${ALL_REASONS:-}
mkdir -p "$W"

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM to mesh with"; exit 77; }
set +u; . "$OFBASHRC" > /dev/null 2>&1 || true; set -u

printf '%-46s %-10s %s\n' CASE ARM OUTCOME
printf '%-46s %-10s %s\n' "---" "---" "---"

for cd_ in $(find "$TUT" -name controlDict -path '*/system/*' | sort); do
    rel=${cd_#"$TUT"/}; rel=${rel%/system/controlDict}
    # ONLY is an EXTENDED REGEX, matched with grep: a `case` pattern cannot take alternation from a
    # variable (the `|` stays literal), which silently selected nothing at all.
    [ -z "$ONLY" ] || printf '%s\n' "$rel" | grep -qE "$ONLY" || continue
    src="$TUT/$rel"
    # ...AND ONLY THE CASES THAT ARE interFoam. This tree holds two that are not: the vofToLagrangian
    # workflow STARTS with an interFoam run (eulerianInjection) and then HANDS OVER to sprayFoam, so
    # lagrangianParticleInjection and lagrangianDistributionInjection carry `application sprayFoam` and
    # `startFrom latestTime` -- they continue from the first run's output. Globbing every controlDict under
    # the tree reported those two as interFoam REFUSALS, which counts a solver brae does not claim against
    # the one it does. OpenFOAM's own `application` entry decides.
    app=$(sed -n 's/^application  *\([A-Za-z0-9_]*\).*/\1/p' "$src/system/controlDict" | head -1)
    if [ -n "$app" ] && [ "$app" != "interFoam" ]; then
        printf '%-46s %-10s %s\n' "$rel" "-" "NOT interFoam (application $app)"
        continue
    fi
    # snappyHexMesh cases: the mesh alone outruns this sweep's budget, and a sweep that silently
    # dropped them would read as coverage it does not have. SNAPPY=1 lifts the skip and meshes them --
    # measured on RAS/motorBike, SERIAL snappyHexMesh is 1.53 s for 8655 cells (maxLocalCells is the
    # global cap in serial), so the blanket skip is not the budget it was taken for; the DTCHull pair is
    # what MESH_TIMEOUT is really for.
    if [ -f "$src/system/snappyHexMeshDict" ] && [ -z "${SNAPPY:-}" ]; then
        printf '%-46s %-10s %s\n' "$rel" "-" "SKIPPED (snappyHexMesh: set SNAPPY=1 to mesh and run them)"
        continue
    fi
    c="$W/$(echo "$rel" | tr / _)"
    rm -rf "$c"; cp -r "$src" "$c" || continue
    rm -rf "$c"/processor* "$c"/log.*
    [ -d "$c/0.orig" ] && { rm -rf "$c/0"; cp -r "$c/0.orig" "$c/0"; }

    mesh_ok=1
    (
        cd "$c" || exit 1
        # the two mesh steps these tutorials' own Allrun scripts take before blockMesh: an m4 template
        # (the sloshingTank family) and a case-supplied Allrun.pre (mixerVessel2D, waterChannel). A
        # sweep that skipped them reported seven cases as "no mesh" when the mesh was simply not built.
        [ -f system/blockMeshDict.m4 ] && timeout "$MESH_TIMEOUT" m4 system/blockMeshDict.m4 > system/blockMeshDict 2> log.m4
        if [ -x ./Allrun.pre ]; then
            timeout "$MESH_TIMEOUT" ./Allrun.pre > log.allrunpre 2>&1
        elif [ -f system/snappyHexMeshDict ] && [ -f ./Allrun ]; then
            # ...and a snappy case whose meshing lives in its Allrun (RAS/DTCHull, DTCHullMoving: feature
            # extraction, six topoSet.N/refineMesh rounds, then snappyHexMesh). The steps below alone left
            # the background block with no hull, and brae refused it -- rightly -- for a kOmegaSST case with
            # no wall patch, which read as a brae gap. The Allrun without its solver and with its parallel
            # steps serial, as the write gate's stage_allrun runs it.
            sed -E -e '/decomposePar|reconstructPar|redistributePar/d' \
                   -e '/\$\(getApplication\)|runApplication +interFoam|runParallel +interFoam/d' \
                   -e 's/runParallel/runApplication/' ./Allrun > Allrun.mesh
            timeout "$MESH_TIMEOUT" bash ./Allrun.mesh > log.allrunmesh 2>&1
        fi
        # ...AND restore0Dir AFTER IT, because that is where the tutorials put it. RAS/mixerVesselAMI's
        # Allrun.pre ENDS WITH `rm -rf 0` and leaves restore0Dir and setFields to its Allrun, so a sweep that
        # copies 0.orig only BEFORE the prep has no fields at all afterwards -- brae then says "cannot open
        # <case>/0/alpha.water", correctly, and it reads as a brae gap. This runs before the setFields below,
        # which is the order Allrun uses.
        [ -d 0.orig ] && [ ! -d 0 ] && cp -r 0.orig 0
        # ...AND `owner.gz` COUNTS AS A MESH HERE TOO. A case with `writeCompression on` -- sloshingCylinder
        # is one -- has its Allrun.pre leave a GZIPPED mesh, so a guard that tests only the plain file finds
        # none and re-runs blockMesh, OVERWRITING the snappy mesh its own Allrun.pre just built. The fields
        # stay at the snappy count and brae then refuses the pair by name: "field internalField has 33568
        # values but the mesh has 4900 cells", which reads as a brae gap and is this line. The same .gz blind
        # spot was already found once for the NO MESH check below; this guard was left behind.
        [ -f system/blockMeshDict ] || [ -f constant/polyMesh/blockMeshDict ] && [ ! -f constant/polyMesh/owner ] && [ ! -f constant/polyMesh/owner.gz ] && timeout "$MESH_TIMEOUT" blockMesh > log.blockMesh 2>&1
        [ -f system/extrudeMeshDict ]   && timeout "$MESH_TIMEOUT" extrudeMesh    > log.extrude 2>&1
        [ -f system/topoSetDict ]       && timeout "$MESH_TIMEOUT" topoSet        > log.topoSet 2>&1
        [ -f system/createBafflesDict ] && timeout "$MESH_TIMEOUT" createBaffles -overwrite > log.baffles 2>&1
        [ -f system/setFieldsDict ]     && timeout "$MESH_TIMEOUT" setFields      > log.setFields 2>&1
        exit 0
    ) || mesh_ok=0
    # ...or owner.gz: these tutorials set `writeCompression on`, and looking only for the plain file
    # reported seven MESHED cases as "no mesh".
    if [ ! -f "$c/constant/polyMesh/owner" ] && [ ! -f "$c/constant/polyMesh/owner.gz" ]; then
        printf '%-46s %-10s %s\n' "$rel" "-" "NO MESH (its Allrun needs a step this sweep does not run)"
        continue
    fi

    # a few steps only: this asks whether the case RUNS, not where it ends up
    python3 - "$c" "$STEPS" <<'PY' 2>/dev/null || true
import re, sys
c, n = sys.argv[1], int(sys.argv[2])
p = c + '/system/controlDict'
s = open(p).read()
m = re.search(r'^deltaT\s+([0-9.eE+-]+)\s*;', s, re.M)
dt = float(m.group(1)) if m else 1e-4
s = re.sub(r'^endTime\s+.*$', 'endTime         %.10g;' % (n*dt), s, flags=re.M)
# the last step writes, so the sweep exercises brae_interFoam's writer too: a time under the run-time
# controls, a STEP COUNT under `timeStep` -- where n*dt is a fraction, which the writer rightly refuses,
# and every timeStep case read as refused by the staging rather than by brae
wc = re.search(r'^writeControl\s+(\w+)', s, re.M)
wi = n if (wc and wc.group(1) == 'timeStep') else n*dt
s = re.sub(r'^writeInterval\s+.*$', 'writeInterval   %.10g;' % wi, s, flags=re.M)
s = re.sub(r'^adjustTimeStep\s+.*$', 'adjustTimeStep  no;', s, flags=re.M)
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
open(p, 'w').write(s)
PY

    # every arm starts from the SAME case: brae_interFoam writes time directories now, and under
    # `startFrom latestTime` the device arm started from the host arm's last write -- "controlDict gives no
    # steps to take" on eleven cases in the first sweep after the writer landed
    start_dirs=$(cd "$c" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9.e+-]+$' | sort | tr '\n' ' ')
    for arm in $ARMS; do
        [ "$arm" = device ] && flag="-device" || flag=""
        for t in $(cd "$c" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9.e+-]+$'); do
            case " $start_dirs " in
                *" $t "*) ;;
                *) rm -rf "${c:?}/$t" ;;
            esac
        done
        out=$(cd "$c" && timeout "$CASE_TIMEOUT" "$BIN" -case "$c" $flag 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then
            verdict="RUNS"
        elif [ $rc -eq 124 ]; then
            verdict="TIMEOUT (${CASE_TIMEOUT}s)"
        else
            # brae states a refusal as a paragraph opening `brae <component>: ...`. Match THAT, not a
            # keyword: the wording varies ("Only X is ported", "asks for", "Refusing rather than"), and
            # a keyword list silently reported `exit 1` for ten cases whose reason was printed in full.
            # ...and THE LINE THAT STOPPED THE RUN, which is the LAST one: the writer names what it will not
            # write at start-up, so the first refusal line is a notice, and a run that dies earlier in its
            # steps for another reason was reported under the writer's (RAS/mixerVesselAMI read "rAU will not
            # be written" while GAMG had refused its AMI interface). When the stop IS the write-time refusal,
            # the notice is the informative line.
            refusals=$(printf '%s\n' "$out" | grep -E "^brae (ERROR|[A-Za-z]+ ?[A-Za-z]*:)" \
                       | grep -vE "controlDict application|\(OF-mirror\)|NOTICE")
            why=$(printf '%s\n' "$refusals" | tail -1 | cut -c1-150)
            if printf '%s' "$why" | grep -q "this is a write time"; then
                why=$(printf '%s\n' "$refusals" | grep "will not be written" | head -1 | cut -c1-150)
            fi
            [ -n "$why" ] || why=$(printf '%s\n' "$out" | grep -viE "^ |NOTICE" | grep -E "[a-z]" | tail -1 | cut -c1-150)
            [ -n "$why" ] || why="exit $rc"
            verdict="REFUSED/ERR: $why"
        fi
        printf '%-46s %-10s %s\n' "$rel" "$arm" "$verdict"
        if [ -n "$ALL_REASONS" ] && [ $rc -ne 0 ]; then
            printf '%s\n' "$out" | sed -n 's/^brae interFoam: \(.*\) will not be written (\(.*\)$/      - \1: \2/p' \
                | sed -E 's/brae interFoam writer: [^ ]+ on patch (`[^`]*`): its condition (`[^`]*`) has no transcribed write\(\).*/\2 on \1/' \
                | cut -c1-140 | sort -u
        fi
    done
    rm -rf "$c"
done
