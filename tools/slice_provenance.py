#!/usr/bin/env python3
"""Check that every provenance-marked slice copy is still identical to its donor.

WHY A SCRIPT AND NOT A SHA. The vertical-slice method (src/applications/solvers/*/slice, see its
README.md) fills a new solver's folder by COPYING files that are already ported rather than rewriting
them, because rewriting is what produced most of the pimpleFoam port's defects -- an OpenFOAM algorithm
paraphrased instead of transcribed. Copying cannot introduce a paraphrase.

What copying does risk is drift: a fix lands in one copy and not the others, and nobody notices. A sha in
a comment does not catch that -- it goes stale the moment the donor is touched, and a stale sha looks
exactly like a current one. A diff does catch it, so the marker is a claim this script verifies:

    // COPIED FROM <path> -- identical as of <anything>.

Everything above the first line of the donor's own content is treated as the copy's header (the
provenance block) and skipped; the rest must match byte for byte.

Exit 0 if every copy matches, 1 otherwise. `--list` just prints what it found.

Usage:  python3 tools/slice_provenance.py [--list]
"""
import os
import re
import sys

BRAE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = re.compile(r'//\s*COPIED FROM\s+(\S+?)\s*--')


def find_copies():
    """Every file under a slice/ directory carrying a COPIED FROM marker."""
    out = []
    for root, _dirs, files in os.walk(os.path.join(BRAE, "src")):
        if os.sep + "slice" not in root:
            continue
        for fn in files:
            path = os.path.join(root, fn)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    head = fh.read(4096)
            except OSError:
                continue
            m = MARK.search(head)
            if m:
                out.append((path, os.path.join(BRAE, m.group(1))))
    return sorted(out)


def body_after_header(copy_text, donor_text):
    """The copy minus its provenance header.

    The header is whatever precedes the donor's first line. Anchoring on the donor rather than counting
    comment lines means the check does not care how long the provenance block is or how it is worded.
    """
    first = donor_text.split("\n", 1)[0]
    i = copy_text.find(first)
    return copy_text[i:] if i >= 0 else copy_text


def main(argv):
    copies = find_copies()
    if not copies:
        print("no provenance-marked slice copies found")
        return 0
    bad = 0
    for copy, donor in copies:
        rel_c = os.path.relpath(copy, BRAE)
        rel_d = os.path.relpath(donor, BRAE)
        if not os.path.exists(donor):
            print(f"  MISSING DONOR  {rel_c}\n                 donor {rel_d} does not exist")
            bad += 1
            continue
        c = open(copy, encoding="utf-8", errors="replace").read()
        d = open(donor, encoding="utf-8", errors="replace").read()
        body = body_after_header(c, d)
        if body == d:
            print(f"  identical      {rel_c}\n                 <- {rel_d}")
        else:
            # Report where they part company, so the fix is a re-copy or a deliberate divergence.
            cl, dl = body.split("\n"), d.split("\n")
            n = next((i for i in range(min(len(cl), len(dl))) if cl[i] != dl[i]), min(len(cl), len(dl)))
            print(f"  DRIFTED        {rel_c}\n                 <- {rel_d}\n"
                  f"                 first difference at donor line {n + 1}")
            bad += 1
    print(f"\n{len(copies) - bad} identical, {bad} drifted")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
