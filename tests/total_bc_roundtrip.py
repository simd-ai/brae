#!/usr/bin/env python3
"""Helpers for tests/total_bc_roundtrip_vs_openfoam.sh: read one patch's dictionary out of a written field, compare two
of them entry by entry, compare fields, and edit a written field the way an earlier brae wrote it (the controls).
  total_bc_roundtrip.py same <fileA> <fileB> <patch>       the patch's entries other than `value`, OpenFOAM's
                                                           write-if-different defaults filled in; exit 1 on a difference
  total_bc_roundtrip.py entry <file> <patch> <key>          one entry's normalised text (nothing: absent)
  total_bc_roundtrip.py fields <dirA> <dirB> <field>...     the largest max|a - b|/max|b| over the internal fields
  total_bc_roundtrip.py cut <file> <patch> <key>...         drop entries (a `<key>Coeffs` with each)
  total_bc_roundtrip.py oldp0 <file> <patch>                the totalPressure-family patch as brae wrote it before
                                                           2026-11-19: type, `inletValue` = p0 (at t = 0), value
  total_bc_roundtrip.py settype <file> <patch> <type>       the patch's `type`
  total_bc_roundtrip.py ctl <controlDict> <key>=<value>...  set controlDict entries (functions removed)
"""
import re
import sys

import numpy as np

# OpenFOAM leaves these out when they hold their default (writeEntryIfDifferent), and brae's writer may not
DEFAULTS = {
    "totalPressure": {"U": "U", "phi": "phi"},
    "uniformTotalPressure": {"U": "U", "phi": "phi"},
    "fanPressure": {"U": "U", "phi": "phi"},
    "totalTemperature": {"U": "U", "phi": "phi", "psi": "thermo:psi"},
    "waveTransmissive": {"phi": "phi", "rho": "rho", "psi": "thermo:psi"},
}


def tokens(s):
    s = re.sub(r"//[^\n]*", "", s)
    return re.findall(r'"[^"]*"|[(){};]|[^\s(){};"]+', s)


def block(text, patch):
    """(start, end) of the patch's `{ ... }` inside boundaryField, braces included."""
    bf = re.search(r"\bboundaryField\s*\{", text)
    m = re.compile(r"\n\s*%s\s*\n?\s*\{" % re.escape(patch)).search(text, bf.end())
    if not m:
        raise SystemExit("no patch %s" % patch)
    i = text.index("{", m.start())
    d = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            d += 1
        elif text[j] == "}":
            d -= 1
            if d == 0:
                return i, j + 1
    raise SystemExit("unbalanced patch %s" % patch)


def parse(toks, i=0):
    """A dictionary's entries from toks[i] (after its '{') to its '}': {key: text or dict}, and the index after."""
    out = {}
    while i < len(toks) and toks[i] != "}":
        key = toks[i]
        i += 1
        if toks[i] == "{":
            sub, i = parse(toks, i + 1)
            out[key] = sub
            continue
        val, depth = [], 0
        while not (depth == 0 and toks[i] == ";"):
            depth += toks[i] == "("
            depth -= toks[i] == ")"
            val.append(toks[i])
            i += 1
        i += 1
        # a list's size before its '(' is optional in OpenFOAM's reader: leave it out of the comparison
        v = " ".join(val)
        out[key] = re.sub(r"(?<![\w.])\d+ \(", "(", v)
    return out, i + 1


def entries(path, patch):
    text = open(path).read()
    a, b = block(text, patch)
    e, _ = parse(tokens(text[a + 1:b - 1] + " }"))
    e.pop("value", None)
    return e


def same(fa, fb, patch):
    ea, eb = entries(fa, patch), entries(fb, patch)
    t = ea.get("type", eb.get("type"))
    for k, v in DEFAULTS.get(t, {}).items():
        ea.setdefault(k, v)
        eb.setdefault(k, v)
    diff = [(k, ea.get(k), eb.get(k)) for k in sorted(set(ea) | set(eb)) if ea.get(k) != eb.get(k)]
    for k, x, y in diff:
        print("    %s: %s | %s" % (k, x, y))
    return 1 if diff else 0


def internal(path):
    s = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*\n?(\d+)\s*\(", s)
    if not m:
        m2 = re.search(r"internalField\s+uniform\s+([^;]+);", s)
        return np.array([float(x) for x in m2.group(1).strip("() ").split()])
    n = int(m.group(2))
    body = s[m.end():]
    if m.group(1) == "scalar":
        return np.array(body.split(None, n)[:n], float)
    v = re.findall(r"\(([^()]*)\)", body[:body.find("\n)\n") + 3])[:n]
    return np.array([list(map(float, x.split())) for x in v])


def fields(da, db, names):
    worst = 0.0
    for f in names:
        x, y = internal("%s/%s" % (da, f)), internal("%s/%s" % (db, f))
        worst = max(worst, float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-300)))
    return worst


def edit(path, patch, fn):
    text = open(path).read()
    a, b = block(text, patch)
    open(path, "w").write(text[:a] + fn(text[a:b]) + text[b:])


def cut(path, patch, keys):
    def f(blk):
        for k in keys:
            for kk in (k + "Coeffs", k):
                blk = re.sub(r"\n\s*%s\s*\{[^{}]*\}" % re.escape(kk), "", blk)
                blk = re.sub(r"\n\s*%s\s[^;{]*;" % re.escape(kk), "", blk)
        return blk
    edit(path, patch, f)


def oldp0(path, patch):
    e = entries(path, patch)
    p0 = e.get("p0", "")
    if isinstance(p0, str) and p0.startswith("uniform"):
        v = p0.split()[1]
    else:   # a table: its first point, as the reader seeded the slot it wrote from
        v = re.findall(r"\( *[^ ()]+ ([^ ()]+) *\)", e["p0Coeffs"]["values"])[0]
    cut(path, patch, ["rho", "psi", "gamma", "p0", "fanCurve", "direction"])
    edit(path, patch, lambda blk: re.sub(r"(\n\s*type\s[^;]*;)", r"\1\n        inletValue      uniform %s;" % v, blk, count=1))


def settype(path, patch, t):
    edit(path, patch, lambda blk: re.sub(r"(\n\s*type\s+)[^;]*;", r"\g<1>%s;" % t, blk, count=1))


def ctl(path, sets):
    c = open(path).read()
    c = re.sub(r"\n\s*#include[^\n]*", "\n", c)
    c = re.sub(r"\nfunctions\s*\{.*?\n\}", "\n", c, flags=re.S)
    for kv in sets:
        k, v = kv.split("=", 1)
        c, n = re.subn(r"\b%s\s+[^;]*;" % re.escape(k), "%s %s;" % (k, v), c)
        if n == 0:
            c += "\n%s %s;\n" % (k, v)
    open(path, "w").write(c)


def main(a):
    cmd = a[0]
    if cmd == "same":
        return same(a[1], a[2], a[3])
    if cmd == "entry":
        e = entries(a[1], a[2]).get(a[3])
        if e is not None:
            print(e)
        return 0
    if cmd == "fields":
        print("%.3e" % fields(a[1], a[2], a[3:]))
        return 0
    if cmd == "cut":
        cut(a[1], a[2], a[3:])
    elif cmd == "oldp0":
        oldp0(a[1], a[2])
    elif cmd == "settype":
        settype(a[1], a[2], a[3])
    elif cmd == "ctl":
        ctl(a[1], a[2:])
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
