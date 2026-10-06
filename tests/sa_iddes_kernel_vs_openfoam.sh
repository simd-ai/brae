#!/usr/bin/env bash
# brae's SA-IDDES against REAL OpenFOAM's SpalartAllmarasIDDES (tests/sa_iddes_lib.sh has the fixture and the
# oracle; tests/test_sa_iddes_vs_openfoam.cu the checks, the bound and what is not claimed).
# THE KERNEL: brae's SA-IDDES length scale, fed OpenFOAM's own inputs cell by cell with the constants of
# OpenFOAM's constructed model, against OpenFOAM's dTilda. Its control leaves psi out of fe, as the kernel had
# it until 2026-10-06 (SpalartAllmarasIDDES.C:118 multiplies fe by psi).
. "$(dirname "$0")/sa_iddes_lib.sh"
stage_and_dump "$W/case"
"$BIN" "$W/case/saIddes_dump.txt" kernel > "$W/log.brae" 2>&1
e=$?
grep -E "^  (ok:|FAIL:)|PASS|FAIL" "$W/log.brae"
[ $e -eq 0 ] && [ "$(grep -c '^  ok:' "$W/log.brae")" = 3 ] && rc=0 || rc=1
echo "sa_iddes_kernel_vs_openfoam: rc $rc"
exit $rc
