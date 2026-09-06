#!/usr/bin/env bash
# `limited <k> corrected` -- OF fv::limitedSnGrad -- on BOTH paths.
#
# The non-orthogonal correction, capped per face against the ORTHOGONAL part of the same snGrad:
#     limiter = min( k*|orth| / ((1 - k)*|corr| + SMALL), 1 ),   correction = limiter * corr
# so `limited 1` IS `corrected` (the denominator collapses to SMALL) and `limited 0` is `uncorrected`.
# turbineSiting asks for `Gauss linear limited corrected 0.33`, and running the UNCAPPED correction under
# that name applies a larger correction than the case asked for -- and it does not vanish at convergence.
#
# THREE ASSERTIONS, on validation/airFoil2D because its C-grid is genuinely non-orthogonal so the
# correction is not a rounding term:
#   1. `limited 1` reproduces `corrected` BIT-FOR-BIT. A limiter that quietly scaled everything, or that
#      divided the wrong way round, would not.
#   2. `limited 0.33` CHANGES the answer -- otherwise the scheme is inert here and the gate is vacuous.
#   3. host and device agree bit-for-bit at 0.33. That is the assertion that earned its keep: OF takes
#      mag() of the WHOLE snGrad and of the WHOLE correction, so a VECTOR field gets ONE limiter per face
#      shared by all three components. brae's device limited each component independently -- a different
#      scheme, 0.6% out on this case -- and only comparing it against the host port exposed that.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${UEQN_LOCALIZE_BIN:-$ROOT/build/ueqn_localize}"
[ -x "$BIN" ] || { echo "SKIP: no ueqn_localize at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/500" ] || { echo "SKIP: no state at $SRC/500"; exit 77; }

run() { UEQN_SNGRAD_LIMIT_K="$1" "$BIN" "$SRC" 500 2>&1; }
host() { echo "$1" | grep 'host   Ux residual' | awk '{print $NF}'; }
dev()  { echo "$1" | grep 'device Ux residual' | awk '{print $NF}'; }

U0=$(run 0); U33=$(run 0.33); U1=$(run 1)
h0=$(host "$U0"); h33=$(host "$U33"); d33=$(dev "$U33"); h1=$(host "$U1"); d1=$(dev "$U1")

python3 - "$h0" "$h33" "$d33" "$h1" "$d1" <<'PY'
import sys
h0, h33, d33, h1, d1 = (float(x) for x in sys.argv[1:6])
rc = 0
print("  uncorrected-cap (k=0)   host %.6e" % h0)
print("  limited 0.33            host %.6e   device %.6e" % (h33, d33))
print("  limited 1               host %.6e   device %.6e" % (h1, d1))

# 1. `limited 1` == `corrected`, exactly, on both paths.
for nm, v in (("host", h1), ("device", d1)):
    if v != h0:
        print("  FAIL: `limited 1` is not identical to `corrected` on the %s (%.6e vs %.6e)" % (nm, v, h0))
        rc = 1
# 2. the limiter must actually bite.
if abs(h33 - h0) / h0 < 1e-6:
    print("  FAIL: `limited 0.33` does not change the answer -- this gate measures nothing"); rc = 1
# 3. the two paths must agree; this is what caught the per-component device limiter.
if h33 != d33:
    print("  FAIL: host and device disagree at k=0.33 (%.6e vs %.6e)" % (h33, d33)); rc = 1
if rc == 0:
    print("  ok:   `limited 1` == `corrected`, 0.33 bites, and both paths agree bit-for-bit")
sys.exit(rc)
PY
