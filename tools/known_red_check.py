#!/usr/bin/env python3
"""known_red_check.py -- hold the known-red rhoCentralFoam gates to their registered open checks.

Six rhoCentralFoam gates carry the ctest label `known-red` (CMakeLists.txt): they stay red on purpose, because each of
their FAILs is an analysed open check in tests/data/rho_central_open_checks.tsv, which the gate prints as a FAIL with
its category on the line below. The GB10 CI's ctest step leaves the label out (-LE known-red), so on its own it would
never see a NEW failure in those six. This runs them (or reads their saved output) and passes only when what is red is
exactly what is registered, as tests/data/rho_central_known_red.tsv counts it:

  * every FAIL line is matched by a registry row of its gate, by the gate library's own rule (the first row whose glob
    matches the message, tests/rho_central_gate_lib.sh `fail`);
  * every registry row matches as many FAIL lines as the expectations say: fewer is a registered check that went green
    or stopped running, more is a new failure that happens to fit the glob;
  * every gate prints its summary line and passes exactly its expected number of checks: fewer is an arm that stopped
    running (the gates do not assert their own counts), more is a new arm the expectations do not know yet.

It changes nothing about the gates: their exit codes, thresholds and output are untouched, the output is echoed, and
the report lists every registered FAIL by category. A change in either direction fails here until the registry,
the expectations and VALIDATION.md's table are updated together, in review. A gate that skips (exit 77: no OpenFOAM)
makes the whole check skip (77).

  tools/known_red_check.py --test-dir build [-j N]    run the six gates through ctest, then check
  tools/known_red_check.py --log GATE=FILE ...        check saved gate output (a gate's stdout, or ctest -V's)
  tools/known_red_check.py --selftest                 check the checker on synthetic output
"""
import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "tests", "data", "rho_central_open_checks.tsv")
EXPECT = os.path.join(ROOT, "tests", "data", "rho_central_known_red.tsv")


def rows(path):
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            yield line.split("\t")


def load(registry, expect):
    """(registry rows per gate, expected passes per gate, expected FAILs per (gate, glob)), or raise on a mismatch."""
    reg = {}
    for c in rows(registry):
        if len(c) < 3:
            raise ValueError("%s: a row with fewer than three columns: %r" % (registry, "\t".join(c)))
        reg.setdefault(c[0], []).append((c[1], c[2]))
    passes, fails = {}, {}
    for c in rows(expect):
        if len(c) == 3 and c[1] == "passed":
            passes[c[0]] = int(c[2])
        elif len(c) == 4 and c[1] == "failed":
            fails[(c[0], c[3])] = int(c[2])
        else:
            raise ValueError("%s: not a `passed` or `failed` row: %r" % (expect, "\t".join(c)))
    regkeys = {(g, pat) for g, rr in reg.items() for pat, _ in rr}
    if regkeys != set(fails):
        miss = sorted(regkeys - set(fails))
        extra = sorted(set(fails) - regkeys)
        raise ValueError("the expectations and the registry name different rows: no expected count for %s; "
                         "no registry row for %s" % (miss, extra))
    if set(reg) != set(passes):
        raise ValueError("gates with registry rows %s, gates with an expected pass count %s"
                         % (sorted(reg), sorted(passes)))
    return reg, passes, fails


def parse(text, gate):
    """FAIL messages, the gate's summary (passed, failed) or None, and whether it skipped."""
    fl, summary, skipped = [], None, False
    sumre = re.compile(r"^%s: (\d+) passed, (\d+) failed" % re.escape(gate))
    for raw in text.splitlines():
        line = re.sub(r"^\d+: ", "", raw)   # ctest -V prefixes a test's lines with its number
        if line.startswith("FAIL "):
            fl.append(line[5:])
        m = sumre.match(line)
        if m:
            summary = (int(m.group(1)), int(m.group(2)))
        if line.startswith("SKIP"):
            skipped = True
    return fl, summary, skipped


def check(outputs, reg, passes, fails, out=sys.stdout):
    """outputs: gate -> text (None: the gate did not run). Returns 0, 1 or 77."""
    bad, skipped, known = [], [], []
    for gate in sorted(passes):
        text = outputs.get(gate)
        if text is None:
            bad.append("%s: no output (the gate did not run)" % gate)
            continue
        fl, summary, skip = parse(text, gate)
        if skip and summary is None:
            skipped.append(gate)
            continue
        if summary is None:
            bad.append("%s: no summary line (the gate stopped before its end)" % gate)
        count = {pat: 0 for pat, _ in reg[gate]}
        for msg in fl:
            hit = next(((pat, cat) for pat, cat in reg[gate] if fnmatch.fnmatchcase(msg, pat)), None)
            if hit is None:
                bad.append("%s: FAIL not in the registry: %s" % (gate, msg))
            else:
                count[hit[0]] += 1
                known.append((gate, hit[1], msg))
        for pat, n in count.items():
            want = fails[(gate, pat)]
            if n != want:
                bad.append("%s: registered row `%s` matched %d FAIL line(s), expected %d" % (gate, pat, n, want))
        if summary is not None:
            if summary[0] != passes[gate]:
                bad.append("%s: %d checks passed, expected %d" % (gate, summary[0], passes[gate]))
            if summary[1] != len(fl):
                bad.append("%s: the summary counts %d FAILs, %d FAIL lines printed" % (gate, summary[1], len(fl)))
    total = sum(fails.values())
    print("\nknown-red gates: %d registered open checks expected in %d gates" % (total, len(passes)), file=out)
    bycat = {}
    for gate, cat, msg in known:
        bycat.setdefault(cat.split(" (")[0], []).append((gate, msg))
    for cat in sorted(bycat):
        print("  KNOWN-RED  %d  %s" % (len(bycat[cat]), cat), file=out)
        for gate, msg in bycat[cat]:
            print("             %s: %s" % (gate.replace("rho_central_", "").replace("_vs_openfoam", ""), msg[:110]),
                  file=out)
    for b in bad:
        print("  UNEXPECTED %s" % b, file=out)
    for g in skipped:
        print("  SKIPPED    %s" % g, file=out)
    if bad:
        print("known-red check: FAILED -- %d difference(s) from the registered open checks" % len(bad), file=out)
        return 1
    if skipped:
        print("known-red check: SKIPPED -- %d gate(s) did not run" % len(skipped), file=out)
        return 77
    print("known-red check: ok -- red is exactly the %d registered open checks (still FAILs, listed above)" % total,
          file=out)
    return 0


def run_ctest(test_dir, gates, jobs):
    """Run the gates through ctest -V and split the interleaved output by test number."""
    regex = "^(%s)$" % "|".join(re.escape(g) for g in sorted(gates))
    cmd = ["ctest", "--test-dir", test_dir, "-R", regex, "-V", "-j", str(jobs)]
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        sys.stdout.write(line)
        lines.append(line)
    p.wait()
    num = {}
    for line in lines:
        m = re.match(r"^\s*(?:\d+/\d+\s+)?(?:Start|Test)\s+#?(\d+):\s+(\S+)", line)
        if m and m.group(2) in gates:
            num[m.group(1)] = m.group(2)
    outputs = {}
    for line in lines:
        m = re.match(r"^(\d+): ", line)
        if m and m.group(1) in num:
            outputs.setdefault(num[m.group(1)], []).append(line[len(m.group(0)):])
    return {g: "".join(v) for g, v in outputs.items()}


def selftest():
    """The checker against synthetic gate output built from the expectations: exact, and five departures."""
    reg, passes, fails = load(REGISTRY, EXPECT)

    def synth(drop=None, add=None, dpass=0, nosummary=None, skip=None):
        o = {}
        for gate in passes:
            if gate == skip:
                o[gate] = "SKIP: no OpenFOAM\n"
                continue
            fl = []
            for pat, _ in reg[gate]:
                fl += [pat.replace("*", "x")] * fails[(gate, pat)]
            if gate == drop:
                fl = fl[1:]
            if gate == add:
                fl.append("brand_new (CUDA): fields above the bound at 1")
            text = "".join("ok   synthetic %d\n" % i for i in range(passes[gate]))
            for f in fl:
                text += "FAIL %s\n     [category]\n" % f
            if gate != nosummary:
                text += "%s: %d passed, %d failed (work in /tmp)\n" % (gate, passes[gate] + dpass, len(fl))
            o[gate] = text
        return o

    g = sorted(passes)[0]
    sink = open(os.devnull, "w")
    cases = [("exact", synth(), 0),
             ("an unregistered FAIL", synth(add=g), 1),
             ("a registered FAIL gone", synth(drop=g), 1),
             ("one check fewer passed", synth(dpass=-1), 1),
             ("a gate with no summary", synth(nosummary=g), 1),
             ("a gate that skipped", synth(skip=g), 77)]
    ok = True
    for name, o, want in cases:
        got = check(o, reg, passes, fails, out=sink)
        print("%-4s %-28s exit %d (expected %d)" % ("ok" if got == want else "FAIL", name, got, want))
        ok &= got == want
    # the registry and the expectations must name the same rows
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as t:
        t.write(open(REGISTRY).read() + "rho_central_3d_vs_openfoam\tnew_row (CUDA): *\tround-off amplification\tx\n")
    try:
        load(t.name, EXPECT)
        print("FAIL a registry row with no expected count is not refused")
        ok = False
    except ValueError:
        print("ok   a registry row with no expected count is refused")
    finally:
        os.unlink(t.name)
    print("selftest: %s" % ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test-dir", help="a configured build directory: run the gates through ctest")
    ap.add_argument("-j", type=int, default=2, help="gates run at once (ctest -j)")
    ap.add_argument("--log", action="append", default=[], metavar="GATE=FILE", help="a gate's saved output")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    reg, passes, fails = load(REGISTRY, EXPECT)
    if a.log:
        outputs = {}
        for spec in a.log:
            gate, path = spec.split("=", 1)
            outputs[gate] = open(path, errors="replace").read()
    elif a.test_dir:
        outputs = run_ctest(a.test_dir, set(passes), a.j)
    else:
        ap.error("give --test-dir, --log or --selftest")
    return check(outputs, reg, passes, fails)


if __name__ == "__main__":
    sys.exit(main())
