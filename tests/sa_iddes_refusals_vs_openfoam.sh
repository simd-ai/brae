#!/usr/bin/env bash
# brae's SA-IDDES against REAL OpenFOAM's SpalartAllmarasIDDES (tests/sa_iddes_lib.sh has the fixture and the
# oracle; tests/test_sa_iddes_vs_openfoam.cu the checks, the bound and what is not claimed).
# THE READER: three switches of the family brae runs at their defaults alone are refused by name wherever
# OpenFOAM would read them -- the model's sub-dictionary or LES{} itself (LESModel.C:72, optionalSubDict) --
# and coefficients written bare in LES{} are read. The dump is not used by this mode beyond being the fixture's.
. "$(dirname "$0")/sa_iddes_lib.sh"
stage_and_dump "$W/case"
"$BIN" "$W/case/saIddes_dump.txt" refusals > "$W/log.brae" 2>&1
e=$?
grep -E "^  (ok:|FAIL:)|PASS|FAIL" "$W/log.brae"
[ $e -eq 0 ] && [ "$(grep -c '^  ok:' "$W/log.brae")" = 3 ] && rc=0 || rc=1
echo "sa_iddes_refusals_vs_openfoam: rc $rc"
exit $rc
