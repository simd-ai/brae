#!/usr/bin/env python3
"""What the mirror tree claims, file by file, and whether the claim resolves: the HOST half, the DEVICE
half, and the tests each one names.

WHY. The port is driven capability by capability -- a scheme, a boundary condition, a model -- but the
question "is this file's host half gated, and its device half" had no answer short of reading the file.
That gap is how the #28 closure defects survived: the host arm of `Gauss limitedLinear` was gated at
7.2e-12 while the device half of the same capability was never wired, and nothing said so out loud.

WHAT IT READS. Every ported file carries a provenance block, and the `tests:` line of that block is the
file's own claim about what holds it:

    // provenance:
    //   openfoam: src/finiteVolume/.../cellLimitedGrad.C
    //   brae:     .../cellLimitedGrad_cpp.cu
    //   tests:    tests/test_celllimited_cpp.cu, tests/celllimited_vs_openfoam.sh

This walks src/, pairs each host half (`*_cpp.*`) with its device twin by name, and prints one row per
pair: the tests each side names, and whether those paths EXIST. A named test that is not there is the
same rot as braeInterFoam.cu's "`-device` refuses LES" -- a claim that was true once, read as current.

    --check   exit 1 if any provenance block names a test path that does not exist
    --gaps    only the rows where one half names tests and the other names none

It reports what the tree SAYS. A file naming a test is not proof that the test exercises it -- that is
what the gates' own controls are for -- but a file naming NOTHING, or naming something that no longer
exists, is a gap you can act on without reading anything.
"""
import os
import re
import sys

BRAE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BRAE, 'src')
EXT = ('.cu', '.cuh')
# a test path the provenance names: tests/<something>.sh or tests/<something>.cu
TESTPATH = re.compile(r'\b(tests/[\w./-]+\.(?:sh|cu|py))')


def provenance_tests(path):
    """(has_block, [test paths]) from the file's provenance block."""
    try:
        with open(path, errors='ignore') as fh:
            head = fh.read(20000)
    except OSError:
        return False, []
    i = head.find('// provenance:')
    if i < 0:
        return False, []
    block = []
    for line in head[i:].split('\n')[1:]:
        if not line.startswith('//'):
            break
        if line.strip() == '//':
            break
        block.append(line)
    text = '\n'.join(block)
    # the tests: line and its continuations -- everything after `tests:` in the block
    j = text.find('tests:')
    tests = sorted(set(TESTPATH.findall(text[j:]))) if j >= 0 else []
    return True, tests


def is_host_half(name):
    return '_cpp' in name


def norm(name):
    """a name the two halves can be matched on: no extension, no `_cpp`, no `device` prefix, no
    underscores, lower case -- because the tree pairs `les_kEqn_cpp.cu` with `device_les_keqn.cu` and an
    exact-name match finds neither that nor `kEpsilon_cpp.cu` -> `kEpsilon.cu` in the same breath."""
    base = os.path.splitext(name)[0]
    base = base.replace('_cpp', '')
    if base.startswith('device_'):
        base = base[len('device_'):]
    return base.replace('_', '').lower()


def main(argv):
    check = '--check' in argv
    gaps_only = '--gaps' in argv

    files = {}
    for root, _, names in os.walk(SRC):
        for n in names:
            if n.endswith(EXT):
                p = os.path.join(root, n)
                files[os.path.relpath(p, BRAE)] = provenance_tests(p)

    # one row per COMPONENT: a .cu and its .cuh are one thing, and the provenance usually lives in the
    # header while the code lives beside it
    comp = {}          # (dir, normalised name) -> {'host': set(tests), 'dev': set(tests), files...}
    for rel, (has, tests) in sorted(files.items()):
        d, n = os.path.split(rel)
        key = (d, norm(n))
        e = comp.setdefault(key, {'host': set(), 'dev': set(), 'hostf': [], 'devf': []})
        side = 'host' if is_host_half(n) else 'dev'
        e[side].update(tests)
        e[side + 'f'].append(n)

    rows = []
    for (d, name), e in sorted(comp.items()):
        if not e['hostf'] and not e['devf']:
            continue
        rows.append((d, name, sorted(e['host']), sorted(e['dev']), e['hostf'], e['devf']))

    missing = []
    for rel, (has, tests) in sorted(files.items()):
        for t in tests:
            if not os.path.exists(os.path.join(BRAE, t)):
                missing.append((rel, t))

    named = sum(1 for _, (h, t) in files.items() if t)
    blocks = sum(1 for _, (h, t) in files.items() if h)
    print("mirror_coverage: %d files, %d with a provenance block, %d naming a test"
          % (len(files), blocks, named))
    print()
    def mark(present, tests):
        if not present:
            return "--"          # that half does not exist in the tree
        if not tests:
            return "?"           # it exists and names no test
        return str(len(tests))   # it names this many

    print("%-58s %-18s %-6s %-6s" % ("component", "directory", "host", "device"))
    print("%-58s %-18s %-6s %-6s" % ("-" * 58, "-" * 18, "-" * 6, "-" * 6))
    shown = 0
    for d, name, htests, dtests, hf, df in rows:
        h, dv = mark(bool(hf), htests), mark(bool(df), dtests)
        if gaps_only and not (bool(hf) and bool(df) and (bool(htests) != bool(dtests))):
            continue
        shown += 1
        print("%-58s %-18s %-6s %-6s" % (name[:58], os.path.basename(d)[:18], h, dv))
    print()
    print("components printed: %d of %d   (`--` that half is not in the tree, `?` it names no test)"
          % (shown, len(rows)))
    both = [r for r in rows if r[4] and r[5]]
    onlyh = [r for r in both if r[2] and not r[3]]
    claimed_no_test = sorted(rel for rel, (h, t) in files.items() if h and not t)
    print("components with BOTH halves in one directory: %d; of those, host names a test and the device "
          "half names none: %d" % (len(both), len(onlyh)))
    print("files with a provenance block that names NO test: %d" % len(claimed_no_test))
    print("files with no provenance block at all: %d -- an older question than this tool answers"
          % (len(files) - blocks))
    if gaps_only:
        print()
        print("a provenance block, and no test named in it:")
        for rel in claimed_no_test:
            print("   %s" % rel)
    print()
    print("NOTE the pairing is by directory and name, which is the only pairing the tree encodes. Most")
    print("device code lives under src/cuda while its host twin sits on the mirrored OpenFOAM path, so a")
    print("`--` here means \"no twin of that name beside it\", not \"no device implementation\".")
    if missing:
        print()
        for rel, t in missing:
            print("MISSING   %s names %s, which does not exist" % (rel, t))
        print("%d provenance test paths do not resolve" % len(missing))
    return 1 if (check and missing) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
