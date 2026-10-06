#!/usr/bin/env bash
# brae's SA-IDDES against REAL OpenFOAM's SpalartAllmarasIDDES (tests/sa_iddes_lib.sh has the fixture and the
# oracle; tests/test_sa_iddes_vs_openfoam.cu the checks, the bound and what is not claimed).
# THE DEFAULTS: the constants brae's reader gives a case that sets none, against the ones OpenFOAM's
# constructed model reports, and the kernel run with them. Its control starts the reader from 20 / 5 / 1.87,
# kOmegaSSTIDDES's, which it used until 2026-10-06 (SpalartAllmarasIDDES.C:165-209 has 8, 3, 3.55, 1.63).
. "$(dirname "$0")/sa_iddes_lib.sh"
stage_and_dump "$W/case"
"$BIN" "$W/case/saIddes_dump.txt" defaults > "$W/log.brae" 2>&1
e=$?
grep -E "^  (ok:|FAIL:)|PASS|FAIL" "$W/log.brae"
[ $e -eq 0 ] && [ "$(grep -c '^  ok:' "$W/log.brae")" = 3 ] && rc=0 || rc=1
echo "sa_iddes_defaults_vs_openfoam: rc $rc"
exit $rc
