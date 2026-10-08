#!/usr/bin/env bash
# The write gate: the device loop's two controls on laminar/oscillatingBox, a mesh that refines AND moves.
. "$(dirname "$0")/../lib.sh"
wcase oscillatingBox of
[ -d "$W/w_of_oscillatingBox" ] || { say "ARM W  oscillatingBox did not stage" FAIL; finish "oscillatingBox device controls"; }
# The tutorial row holds the device arm against OpenFOAM (lib.sh: worst U and Uf 1.8e-11, 2026-10-02; the
# mesh goes 1000 -> 2400 -> 8000 cells in its two steps). These are what that row would not see broken:
#   BRAE_CONTROL_DEVICE_REFINE_STALE_POINTS=1  the refinement works on the points of its last change, so the
#     change hands the mesh back where it was before the motion: MEASURED p 4.2e+00.
#   BRAE_CONTROL_DEVICE_REFINE_MOVE_REFUSED=1  the refusal the device loop had, put back: nothing written.
BOUND_OB_DEVICE=1.8e-10
ot=$(timedirs "$W/w_of_oscillatingBox")
e="$W/ob_ctl_stale"
mkdir -p "$e"
cp -r "$W/w_of_oscillatingBox/0" "$W/w_of_oscillatingBox/constant" "$W/w_of_oscillatingBox/system" "$e/"
runbrae "$e" device BRAE_CONTROL_DEVICE_REFINE_STALE_POINTS=1
grep -q "CONTROL MODE" "$e/log.brae" \
    && say "CONTROL  [device] BRAE_CONTROL_DEVICE_REFINE_STALE_POINTS=1 ran as a control" ok \
    || say "CONTROL  [device] BRAE_CONTROL_DEVICE_REFINE_STALE_POINTS=1 ran as a control" FAIL
python3 "$CMP" "$W/w_of_oscillatingBox" "$e" $ot > "$W/cmp_ob_stale.txt" 2>&1
judge "control stale points" "$W/cmp_ob_stale.txt" "$BOUND_OB_DEVICE" "$W/w_of_oscillatingBox/log.interFoam" > "$W/ob_stale.txt" \
    && { say "CONTROL  [device] the stale points put oscillatingBox over the bound" FAIL; cat "$W/ob_stale.txt"; } \
    || say "CONTROL  [device] the stale points put oscillatingBox over the bound" ok
e="$W/ob_ctl_refused"
mkdir -p "$e"
cp -r "$W/w_of_oscillatingBox/0" "$W/w_of_oscillatingBox/constant" "$W/w_of_oscillatingBox/system" "$e/"
( cd "$e" && BRAE_CONTROL_DEVICE_REFINE_MOVE_REFUSED=1 "$BIN" -case . -device > log.brae 2>&1 )
grep -q "the mesh refines AND a motion solver moves it" "$e/log.brae" && [ -z "$(timedirs "$e")" ] \
    && say "CONTROL  [device] BRAE_CONTROL_DEVICE_REFINE_MOVE_REFUSED=1 refuses by name and writes nothing" ok \
    || { say "CONTROL  [device] BRAE_CONTROL_DEVICE_REFINE_MOVE_REFUSED=1 refuses by name and writes nothing" FAIL; tail -3 "$e/log.brae" | cut -c1-300; }
finish "oscillatingBox device controls: each defect is seen"
