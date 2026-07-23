#!/usr/bin/env python3
# Compare a distributed-path field write against the single-GPU reference for the AMI oracle test.
# U is compared directly (L2-relative); p is a SINGULAR all-Neumann field on a closed AMI domain, so it is
# only defined up to an additive constant -- compare it LEVEL-SUBTRACTED (mean removed from both). The
# distributed vs single-GPU comparison is the one that matters: my two AMI bugs (bounded-div missing the
# AMI-face flux; deferred-rotation leaking into H()) lived in the SHARED distributed code, so a np2-vs-np1
# oracle would pass with both ranks wrong -- only distributed-vs-single-GPU catches them.
import sys, re, math

FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def read(fn, vec):
    t = open(fn).read()
    typ = "vector" if vec else "scalar"
    m = re.search(r"internalField\s+nonuniform\s+List<" + typ + r">\s*\n(\d+)\s*\n\(", t)
    if not m:
        raise SystemExit("READERR: no nonuniform internalField in " + fn)
    n = int(m.group(1))
    nums = [float(x) for x in re.findall(FLOAT, t[m.end():])]
    if vec:
        return [nums[3 * i:3 * i + 3] for i in range(n)]
    return nums[:n]

def l2_vec(a, b):
    num = sum((a[i][c] - b[i][c]) ** 2 for i in range(len(a)) for c in range(3))
    den = sum(b[i][c] ** 2 for i in range(len(b)) for c in range(3))
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)

def l2_scalar_level(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    aa = [x - ma for x in a]
    bb = [x - mb for x in b]
    num = sum((aa[i] - bb[i]) ** 2 for i in range(len(a)))
    den = sum(x * x for x in bb)
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)

def main():
    ref_dir, tst_dir, uTol, pTol = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    U = l2_vec(read(ref_dir + "/U", True), read(tst_dir + "/U", True))
    p = l2_scalar_level(read(ref_dir + "/p", False), read(tst_dir + "/p", False))
    ok = (U < uTol) and (p < pTol)
    print("U L2rel=%.3e (tol %.1e)  p level-subtracted L2rel=%.3e (tol %.1e)  -> %s"
          % (U, uTol, p, pTol, "PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
