#!/usr/bin/env python3
"""Compare two OpenFOAM time directories: structure exactly, values numerically.

    foam_time_compare.py <referenceCase> <otherCase> <time> [<time> ...]

STRUCTURE is what OpenFOAM's write() decides and must match exactly: the set of files (a `.gz` and its
plain twin count as one), every FoamFile header entry, `dimensions`, the patch order, each patch's
keyword list in order, and every non-numeric entry (types, blending words, Function1 words, `name`,
`index`).

VALUES are compared after expanding `uniform v` against the other side's list, because whether a list is
written uniform is a property of the values: OpenFOAM writes `uniform 0` for a wall gradient whose faces
are exactly zero, and a solver whose arithmetic leaves 1e-12 there writes a list. Each file reports its
worst |a - b| over every number in it, divided by the largest |a| in the reference file (the field's own
scale; for uniform/time that is the time value).

uniform/functionObjects/functionObjectProperties is compared as TEXT after the banner: its content is
function-object state, and a case run with `functions {}` has none on either side.

Prints one line per file and a final `RESULT {json}` line: {"structure": <failures>, "files":
{"<time>/<file>": {"structure": bool, "rel": x, "abs": |a - b|, "notes": [...]}}, "fileset": [...]}. Exit status is 1 when
any structure check fails, else 0 -- value bounds belong to the caller, which knows what it measured.
"""
import gzip
import json
import os
import re
import sys

EXACT_KEYS = {"dimensions", "name", "index", "oriented"}


def read(path):
    if os.path.exists(path):
        return open(path).read()
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt").read()
    return None


def files(d):
    out = set()
    for root, _, names in os.walk(d):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), d)
            out.add(rel[:-3] if rel.endswith(".gz") else rel)
    return out


def tokens(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.findall(r'"[^"]*"|[{}();\[\]]|[^\s{}();\[\]]+', text)


def parse_block(toks, i):
    """(list of (key, ('dict', entries) | ('val', tokens)), index past the closing brace)"""
    out = []
    while i < len(toks) and toks[i] != "}":
        key = toks[i]
        i += 1
        if toks[i] == "{":
            sub, i = parse_block(toks, i + 1)
            out.append((key, ("dict", sub)))
            continue
        val = []
        depth = 0
        while not (toks[i] == ";" and depth == 0):
            if toks[i] in "([":
                depth += 1
            if toks[i] in ")]":
                depth -= 1
            val.append(toks[i])
            i += 1
        i += 1
        out.append((key, ("val", val)))
    return out, i + 1


def numbers(val):
    """('uniform'|'list'|'plain', [tuple per entry]) or None when the entry is not numeric"""
    try:
        if val and val[0] == "uniform":
            return ("uniform", [tuple(float(x) for x in val[1:] if x not in "()")])
        if val and val[0] == "nonuniform":
            body = val[2:]
            start = 1 if body[0] != "(" else 0
            inner = body[start + 1:-1]
            ents = []
            cur = None
            for x in inner:
                if x == "(":
                    cur = []
                elif x == ")":
                    ents.append(tuple(cur))
                    cur = None
                elif cur is not None:
                    cur.append(float(x))
                else:
                    ents.append((float(x),))
            return ("list", ents)
        return ("plain", [tuple(float(x) for x in val if x not in "()")])
    except (ValueError, IndexError):
        return None


class FileCompare:
    def __init__(self, tag):
        self.tag = tag
        self.structure = True
        self.notes = []
        self.worst = 0.0
        self.scale = 0.0

    def bad(self, msg):
        self.structure = False
        self.notes.append(msg)

    def values(self, where, va, vb):
        na = numbers(va)
        nb = numbers(vb)
        if na is None or nb is None:
            if va != vb:
                self.bad("%s: %s vs %s" % (where, " ".join(va)[:80], " ".join(vb)[:80]))
            return
        a = na[1]
        b = nb[1]
        if na[0] == "uniform" and nb[0] == "list":
            a = a * len(b)
        if nb[0] == "uniform" and na[0] == "list":
            b = b * len(a)
        if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
            self.bad("%s: %d entries vs %d" % (where, len(a), len(b)))
            return
        for x, y in zip(a, b):
            for p, q in zip(x, y):
                self.scale = max(self.scale, abs(p))
                self.worst = max(self.worst, abs(p - q))

    def entries(self, where, ea, eb):
        ka = [k for k, _ in ea]
        kb = [k for k, _ in eb]
        if ka != kb:
            self.bad("%s: keywords %s vs %s" % (where, ka, kb))
            return
        db = dict(eb)
        for k, (kind, v) in ea:
            wk = "%s/%s" % (where, k) if where else k
            kind_b, vb = db[k]
            if kind != kind_b:
                self.bad("%s: a dictionary on one side only" % wk)
            elif kind == "dict":
                self.entries(wk, v, vb)
            elif k in EXACT_KEYS or where == "FoamFile":
                if v != vb:
                    self.bad("%s: %s vs %s" % (wk, " ".join(v), " ".join(vb)))
            else:
                self.values(wk, v, vb)

    def rel(self):
        return self.worst / self.scale if self.scale > 0 else self.worst


def compare_file(ref_text, other_text, tag, as_text):
    fc = FileCompare(tag)
    if as_text:
        body_a = ref_text[ref_text.find("FoamFile"):]
        body_b = other_text[other_text.find("FoamFile"):]
        if body_a != body_b:
            fc.bad("text differs after the banner")
        return fc
    ta = tokens(ref_text)
    tb = tokens(other_text)
    la = bare_list(ta)
    lb = bare_list(tb)
    if la is not None or lb is not None:
        # a bare list after the header (polyMesh/points, a pointIOField): header exactly, count exactly,
        # every value numerically
        if la is None or lb is None:
            fc.bad("a bare list on one side only")
            return fc
        ha, _ = parse_block(la[0] + ["}"], 0)
        hb, _ = parse_block(lb[0] + ["}"], 0)
        fc.entries("", ha, hb)
        if la[1][0] != lb[1][0]:
            fc.bad("list count %s vs %s" % (la[1][0], lb[1][0]))
        fc.values("list", ["nonuniform", "List"] + la[1], ["nonuniform", "List"] + lb[1])
        return fc
    ea, _ = parse_block(ta + ["}"], 0)
    eb, _ = parse_block(tb + ["}"], 0)
    fc.entries("", ea, eb)
    return fc


def bare_list(toks):
    """(header tokens, list tokens) when the body after FoamFile is `N ( ... )` alone, else None"""
    if len(toks) < 3 or toks[0] != "FoamFile" or toks[1] != "{":
        return None
    depth = 0
    for i in range(1, len(toks)):
        if toks[i] == "{":
            depth += 1
        elif toks[i] == "}":
            depth -= 1
            if depth == 0:
                rest = toks[i + 1:]
                if len(rest) >= 2 and rest[0].isdigit() and rest[1] == "(":
                    return toks[:i + 1], rest
                return None
    return None


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    ref, other = sys.argv[1], sys.argv[2]
    result = {"structure": 0, "files": {}, "fileset": []}
    for t in sys.argv[3:]:
        da = os.path.join(ref, t)
        db = os.path.join(other, t)
        if not os.path.isdir(db):
            print("  [%s] not written" % t)
            result["structure"] += 1
            result["fileset"].append("%s: not written" % t)
            continue
        fa = files(da)
        fb = files(db)
        if fa != fb:
            msg = "%s: only reference %s, only other %s" % (t, sorted(fa - fb), sorted(fb - fa))
            print("  FILESET " + msg)
            result["structure"] += 1
            result["fileset"].append(msg)
        for f in sorted(fa & fb):
            tag = "%s/%s" % (t, f)
            fc = compare_file(
                read(os.path.join(da, f)),
                read(os.path.join(db, f)),
                tag,
                f.endswith("functionObjectProperties"))
            if not fc.structure:
                result["structure"] += 1
            result["files"][tag] = {
                "structure": fc.structure,
                "rel": fc.rel(),
                "abs": fc.worst,
                "notes": fc.notes,
            }
            print("  %-48s structure %-4s worst rel %.3e" % (tag, "ok" if fc.structure else "BAD", fc.rel()))
            for n in fc.notes[:6]:
                print("      " + n)
    print("RESULT " + json.dumps(result))
    return 1 if result["structure"] else 0


if __name__ == "__main__":
    sys.exit(main())
