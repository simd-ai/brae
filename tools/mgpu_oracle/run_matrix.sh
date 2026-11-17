#!/bin/bash
# The reference matrix used for REPORT numbers: every case at np 2/4/8 (scotch), the DGX-1-shaped simple/hierarchical
# 8-way splits, and the tightened-solver variants.  ~1 h of wall time, ~600 MB per run (OUT_ROOT, default /tmp is a
# RAM disk: point OUT_ROOT at a disk and delete the runs you do not need afterwards).
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
for c in tank tankGraded open; do
  for n in 2 4 8; do "$HERE/run_oracle.sh" $c $n scotch 20 > /dev/null 2>&1; done
  "$HERE/run_oracle.sh" $c 8 simple 20 > /dev/null 2>&1
  "$HERE/run_oracle.sh" $c 8 hier 20 > /dev/null 2>&1
  TIGHT=1 "$HERE/run_oracle.sh" $c 8 scotch 20 > /dev/null 2>&1
  TIGHT=1 "$HERE/run_oracle.sh" $c 4 scotch 20 > /dev/null 2>&1
done
