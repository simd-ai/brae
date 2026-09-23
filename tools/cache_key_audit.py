#!/usr/bin/env python3
"""A device cache keyed on a POINTER, without an identity that survives pool recycling.

`DeviceBuffer`'s pool hands equal-sized blocks back AT THE SAME ADDRESS. So a pointer -- `A.diag`,
`A.owner`, `psi.data()` -- is not the identity of the thing it points into: on a moving mesh the GAMG
hierarchy is rebuilt every step and its coarse levels come back at the same addresses with the same
cell and face counts and a DIFFERENT pairing. A cache that compares only the pointer (and the sizes)
replays the previous step's schedule. MEASURED on waves/waveMakerMultiPaddlePiston, 448k cells:
`gsLevelsFor` keyed on `A.owner` gave device U 3.0057e-02 against OpenFOAM at thirty steps where the
PCG twin read 6.4e-11, and a second device run IN THE SAME PROCESS differed from the first by
1.9e-04. `nextDeviceAddressingId()` (device_mesh.cuh) exists for this: it stamps the addressing and
the guards compare `addressingId`.

WHAT IT CHECKS. Every cache guard -- a condition that decides whether to rebuild or replay something
cached -- that compares a DEVICE POINTER. For each, whether the same condition also compares one of
the identities that survive recycling:

    addressingId    the stamp on DeviceMesh / DeviceLduMatrix / AMGLevel
    buildCount      the agglomeration's rebuild counter
    Epoch           deviceReductionScratchEpoch(), for a graph that captured reduction scratch
    generation      deviceGraphGeneration(), for a graph whose buffers were reallocated

A guard with none of them is REPORTED. `tools/cache_key_audit_allow.txt` is the ledger for a cache
whose lifetime makes the pointer sufficient -- one destroyed with the object it belongs to, say --
with the reason; a stale entry fails, so it cannot rot into a suppression file.

usage: cache_key_audit.py [--allow FILE] <dir>...      exit 1 on an unlisted pointer-only guard
"""
import os
import re
import sys

# a comparison against a device pointer: `c.key != psi.data()`, `gc.key != A.diag`, `c.owner != A.owner`
POINTER_CMP = re.compile(
    r'(\w+)\s*\.\s*(\w+)\s*!=\s*(?:\(const\s+void\s*\*\)\s*)?[\w.\->]*\b(diag|owner|nei|upper|lower|psi|data\(\))')
# the identities that survive a pool block coming back at the same address
SURVIVORS = ('addressingId', 'buildCount', 'Epoch', 'epoch', 'generation', 'Generation')


def strip_comments(text):
    """Comments out, lines kept. A guard quoted in a comment is not a guard, and a survivor named
    only in one would mark a pointer-keyed cache safe -- the failure mode that makes an audit look
    like it is working when it is not (tools/resize_audit.py met both directions of this)."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(''.join(c if c == '\n' else ' ' for c in text[i:j]))
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)


def main(argv):
    allow_path = None
    if '--allow' in argv:
        i = argv.index('--allow')
        allow_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    roots = argv or ['src']

    allow = set()
    if allow_path and os.path.exists(allow_path):
        for line in open(allow_path):
            line = line.split('#')[0].strip()
            if line:
                allow.add(line)

    files = []
    for root in roots:
        for r, _, ns in os.walk(root):
            for n in ns:
                if n.endswith(('.cu', '.cuh')):
                    files.append(os.path.join(r, n))
    files.sort()

    guards, hits = 0, []
    for f in files:
        t = strip_comments(open(f, errors='ignore').read())
        # a guard is one `if (...)` or one `= ... ;` condition; take the whole statement around the
        # comparison so a condition spread over five lines is read as one
        for m in POINTER_CMP.finditer(t):
            start = t.rfind(';', 0, m.start())
            start = max(start, t.rfind('{', 0, m.start()))
            start = max(start, t.rfind('}', 0, m.start()))
            end = t.find(';', m.end())
            if end < 0:
                end = m.end()
            stmt = t[start + 1:end]
            if 'if' not in stmt and '=' not in stmt:
                continue
            # NOT A CACHE GUARD: the two sides are members of the SAME object. `Ain.lower !=
            # Ain.upper` and `M.upper != M.lower` are symmetry checks on one matrix, and reading
            # them as a stale-cache question reported two files that hold no cache at all.
            rhs = t[m.start():m.end()]
            rhs_obj = re.match(r'[^!]*!=\s*(?:\(const\s+void\s*\*\)\s*)?(\w+)', rhs)
            if rhs_obj and rhs_obj.group(1) == m.group(1):
                continue
            # ...nor is a REFUSAL. A guard whose body throws is the safe answer to this very question
            # -- device_colour_gauss_seidel.cu refuses a colouring reused with another mesh -- and it
            # is the opposite of silently replaying a stale entry.
            body = t[end:end + 400]
            if 'throw' in body or 'refuse(' in body:
                continue
            guards += 1
            if any(s in stmt for s in SURVIVORS):
                continue
            rel = os.path.relpath(f)
            line = t[:m.start()].count('\n') + 1
            key = '%s %s.%s' % (os.path.basename(f), m.group(1), m.group(2))
            hits.append((rel, line, key, m.group(3)))

    unlisted = [h for h in hits if h[2] not in allow]
    print('cache_key_audit: %d files, %d cache guards comparing a device pointer' % (len(files), guards))
    print('guards with NO identity that survives pool recycling: %d (%d unlisted)'
          % (len(hits), len(unlisted)))
    for rel, line, key, what in hits:
        mark = ' ' if key in allow else '*'
        print('  %s %s:%d  keyed on `%s` alone' % (mark, rel, line, what))
    stale = [a for a in sorted(allow) if not any(h[2] == a for h in hits)]
    for a in stale:
        print('STALE ledger entry, no such guard: %s' % a)
    return 1 if (unlisted or stale) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
