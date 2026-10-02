#!/usr/bin/env bash
# The write gate: CrankNicolson's state on a moving mesh with the body released, and the closure's control.
. "$(dirname "$0")/../lib.sh"
FO="$TUT/multiphase/interFoam/RAS/floatingObject"
wcase floatingObject of
[ -d "$W/w_of_floatingObject" ] || { say "ARM X  floatingObject did not stage" FAIL; finish "arm X"; }
# X: CrankNicolson's state on a MOVING mesh (U7) -- RAS/floatingObject as shipped is arm W's row; these are
# what that row cannot witness, and the controls of what it can.
#   THE TUTORIAL AS SHIPPED keeps its body at rest until t = 4 (accelerationRelaxation is a table that is
#   zero until then), so no cell changes volume and the closure's moving fvm::ddt is the static one
#   arithmetically, and it ends before Uf_0 is first written (the third step). X1 RELEASES THE BODY --
#   `accelerationRelaxation 0.7`, the table's own final value, in both codes -- and runs five steps: the mesh
#   deforms, V0 and V00 differ, and OpenFOAM writes Uf_0. MEASURED (2026-10-01, host): worst file
#   meshPhiCN_0 3.0e-11, every field below 2e-12.
# CONTROLS, each asserted red:
#   BRAE_CONTROL_CN_CLOSURE_STATIC=1 on X1 (kEpsilon's fvm::ddt on the scheme's static branch): ddt0(k)
#     2.3e-03, epsilon 5.9e-04.
#   BRAE_CONTROL_CN_PHIOLD_PREV=1 on the tutorial (the alpha blend's phi.oldTime() left at the previous
#     step's flux in the correctors after the one that created it): U 2.9e-08, rAU 4.4e-08 at step two.
#   BRAE_CONTROL_CN_OLD_AT_ENTRY=1 on the tutorial (epsilon.oldTime() taken before the wall function's
#     update on a cold start): epsilon_0 8.7e-01 at the first write.
#   BRAE_CONTROL_CN_DDT0_CELLS_ONLY=1 on the tutorial (the ddt0 fields' patches left zero): ddt0(k) and
#     ddt0(epsilon) 1.0e+00 at step two.
. "$(dirname "$0")/_common.sh"
x1_stage
for v in run static; do
    e="$W/x1_br_$v"
    mkdir -p "$e"
    cp -r "$d/0" "$d/constant" "$d/system" "$e/"
done
runbrae "$W/x1_br_run" host
[ "$(echo $(timedirs "$W/x1_br_run"))" = "$xt" ] && [ "$(filesets "$d" $xt)" = "$(filesets "$W/x1_br_run" $xt)" ] \
    && say "ARM X1 [host] floatingObject released: OpenFOAM's directory and file set, Uf_0 included" ok \
    || say "ARM X1 [host] floatingObject released: OpenFOAM's directory and file set, Uf_0 included" FAIL
python3 "$CMP" "$d" "$W/x1_br_run" $xt > "$W/cmp_x1.txt" 2>&1
judge "floatingObject released" "$W/cmp_x1.txt" "$BOUND_X1" "$d/log.interFoam" \
    && say "ARM X1 [host] floatingObject released: every file's structure is OpenFOAM's, every value within $BOUND_X1" ok \
    || { say "ARM X1 [host] floatingObject released: every file's structure is OpenFOAM's, every value within $BOUND_X1" FAIL; grep -v RESULT "$W/cmp_x1.txt" | grep -B1 "^      " | head -12; }
runbrae "$W/x1_br_static" host BRAE_CONTROL_CN_CLOSURE_STATIC=1
python3 "$CMP" "$d" "$W/x1_br_static" $xt > "$W/cmp_x1_static.txt" 2>&1
judge "control closure static" "$W/cmp_x1_static.txt" "$BOUND_X1" "$d/log.interFoam" > "$W/x1_static.txt" \
    && { say "CONTROL  BRAE_CONTROL_CN_CLOSURE_STATIC=1 puts the released body's k and epsilon over the bound" FAIL; cat "$W/x1_static.txt"; } \
    || say "CONTROL  BRAE_CONTROL_CN_CLOSURE_STATIC=1 puts the released body's k and epsilon over the bound" ok
grep -qE "over the bound: [0-9.e+-]+/(k|epsilon) " "$W/x1_static.txt" \
    && say "CONTROL  ...and it is k or epsilon that goes over" ok \
    || { say "CONTROL  ...and it is k or epsilon that goes over" FAIL; cat "$W/x1_static.txt"; }
finish "arm X1: the released body writes OpenFOAM's CrankNicolson state"
