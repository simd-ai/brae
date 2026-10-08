#!/usr/bin/env bash
# The write gate: CrankNicolson's state on a moving mesh with the body released, FROM THE DEVICE LOOP, and
# the device closure's control.
. "$(dirname "$0")/../lib.sh"
FO="$TUT/multiphase/interFoam/RAS/floatingObject"
wcase floatingObject of
[ -d "$W/w_of_floatingObject" ] || { say "ARM X  floatingObject did not stage" FAIL; finish "arm X"; }
# X1 on the device: the fixture of cn_moving/released.sh -- the body released, five steps, the mesh
# deforming, Uf_0 written -- run through `-device`, which keeps the scheme's state in device buffers and
# brings it down at the write: the ddt0 fields' cells AND patches, the old levels, V0.
# MEASURED (2026-10-02): worst file meshPhiCN_0 1.7e-11, ddt0(rho,U) 1.1e-12, every field below 1e-12.
# CONTROL: BRAE_CONTROL_DEVICE_CN_CLOSURE_STATIC=1 (the device kEpsilon's fvm::ddt on the static branch).
. "$(dirname "$0")/_common.sh"
x1_stage
BOUND_X1D=1.7e-10
for v in run static; do
    e="$W/x1d_br_$v"
    mkdir -p "$e"
    cp -r "$d/0" "$d/constant" "$d/system" "$e/"
done
runbrae "$W/x1d_br_run" device
[ "$(echo $(timedirs "$W/x1d_br_run"))" = "$xt" ] && [ "$(filesets "$d" $xt)" = "$(filesets "$W/x1d_br_run" $xt)" ] \
    && say "ARM X1 [device] floatingObject released: OpenFOAM's directory and file set, Uf_0 included" ok \
    || say "ARM X1 [device] floatingObject released: OpenFOAM's directory and file set, Uf_0 included" FAIL
python3 "$CMP" "$d" "$W/x1d_br_run" $xt > "$W/cmp_x1d.txt" 2>&1
judge "floatingObject released, device" "$W/cmp_x1d.txt" "$BOUND_X1D" "$d/log.interFoam" \
    && say "ARM X1 [device] floatingObject released: every file's structure is OpenFOAM's, every value within $BOUND_X1D" ok \
    || { say "ARM X1 [device] floatingObject released: every file's structure is OpenFOAM's, every value within $BOUND_X1D" FAIL; grep -v RESULT "$W/cmp_x1d.txt" | grep -B1 "^      " | head -12; }
runbrae "$W/x1d_br_static" device BRAE_CONTROL_DEVICE_CN_CLOSURE_STATIC=1
python3 "$CMP" "$d" "$W/x1d_br_static" $xt > "$W/cmp_x1d_static.txt" 2>&1
judge "control device closure static" "$W/cmp_x1d_static.txt" "$BOUND_X1D" "$d/log.interFoam" > "$W/x1d_static.txt" \
    && { say "CONTROL  BRAE_CONTROL_DEVICE_CN_CLOSURE_STATIC=1 puts the released body's k and epsilon over the bound" FAIL; cat "$W/x1d_static.txt"; } \
    || say "CONTROL  BRAE_CONTROL_DEVICE_CN_CLOSURE_STATIC=1 puts the released body's k and epsilon over the bound" ok
grep -qE "over the bound: [0-9.e+-]+/(k|epsilon|ddt0\\(k\\)|ddt0\\(epsilon\\)) " "$W/x1d_static.txt" \
    && say "CONTROL  ...and it is the closure's own files that go over" ok \
    || { say "CONTROL  ...and it is the closure's own files that go over" FAIL; cat "$W/x1d_static.txt"; }
finish "arm X1 device: the released body writes OpenFOAM's CrankNicolson state from the device loop"
