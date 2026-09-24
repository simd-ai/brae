#!/usr/bin/env python3
"""A function DEFINED in a .cuh with external linkage -- one strong symbol per including object.

A header is included by every translation unit that needs what it declares. A function it DEFINES,
without `inline`, `static`, `constexpr` or a template, is emitted as a strong symbol in each of them,
and the link fails the moment two such objects are put together:

    /usr/bin/ld: libbrae_core.a(inter_case_cpp.cu.o): multiple definition of
        `brae::readSurfaceField(...)'; CMakeFiles/brae.dir/.../gpuSimpleFoam.cu.o: first defined here

MEASURED: `readSurfaceField` (read_surface_field.cuh) was included by six TUs, two of which -- the
interFoam case reader and the simpleFoam driver -- reach the same binary. The ordinary build survived
because the archive hands out only one of them; a localiser that pulled the other could not be linked
at all without `-Wl,--allow-multiple-definition`, which picks a definition and hides the question. The
same header family held four `__global__` kernels defined in device_scalar_transport.cuh, which
twenty-four objects define strongly apiece.

WHAT IT CHECKS. Every function DEFINITION at namespace scope in a .cuh -- a signature followed by a
body, not a declaration, not a class member, not a default argument's braces -- carries `inline`,
`static` or `constexpr`, or is a template. `tools/header_odr_audit_allow.txt` is the ledger for one
that must keep external linkage, with the reason; a stale entry fails, so it cannot rot into a
suppression file.

usage: header_odr_audit.py [--allow FILE] <dir>...     exit 1 on an unlisted external definition
"""
import os
import re
import sys

KEYWORDS = (r'(if|for|while|switch|return|else|catch|namespace|class|struct|enum|union|using|'
            r'typedef|template|static_assert|extern|friend|do|try|public|private|protected|operator)')


def strip_comments(src):
    src = re.sub(r'/\*.*?\*/', lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
    return re.sub(r'//[^\n]*', '', src)


def scan(path):
    """Namespace-scope function definitions in `path` that have external linkage."""
    lines = strip_comments(open(path, errors="replace").read()).split("\n")
    out = []
    stack = []      # 'ns' for a namespace body, 'blk' for anything else
    par = 0         # a brace inside a parameter list is a default argument, not a body
    pending = ""    # the text since the last ; { }
    for i, raw in enumerate(lines):
        if raw.strip().startswith("#"):
            pending = ""
            continue
        for ch in raw:
            if ch == '(':
                par += 1
                pending += ch
                continue
            if ch == ')':
                par = max(0, par - 1)
                pending += ch
                continue
            if par > 0:
                pending += ch
                continue
            if ch == '{':
                kind = 'ns' if re.search(r'\bnamespace\b[^;{}]*$', pending) else 'blk'
                if kind == 'blk' and not any(k == 'blk' for k in stack):
                    sig = re.sub(r'\s+', ' ', pending).strip()
                    m = re.match(r'^([A-Za-z_][\w:<>,\s\*&]*?)\s([\w:~]+)\s*\(', sig)
                    if (m
                            and not re.match(r'^%s\b' % KEYWORDS, sig)
                            and 'inline' not in m.group(1)
                            and 'static' not in m.group(1)
                            and 'constexpr' not in m.group(1)
                            and 'template' not in sig
                            and '=' not in sig.split('(')[0]):
                        out.append((i + 1, m.group(2), sig[:110]))
                stack.append(kind)
                pending = ""
            elif ch == '}':
                if stack:
                    stack.pop()
                pending = ""
            elif ch == ';':
                pending = ""
            else:
                pending += ch
        pending += " "
    return out


def main(argv):
    allow_path = None
    roots = []
    it = iter(argv)
    for a in it:
        if a == "--allow":
            allow_path = next(it)
        else:
            roots.append(a)
    if not roots:
        print(__doc__)
        return 2

    allow = {}
    if allow_path and os.path.exists(allow_path):
        for line in open(allow_path):
            line = line.split("#", 1)[0].strip()
            if line:
                parts = line.split(None, 1)
                allow[parts[0]] = parts[1] if len(parts) > 1 else ""

    seen = set()
    reported = 0
    headers = 0
    defs = 0
    for root in roots:
        for d, _, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith(".cuh"):
                    continue
                path = os.path.join(d, fn)
                headers += 1
                for line, name, sig in scan(path):
                    defs += 1
                    key = "%s %s" % (fn, name)
                    seen.add(key)
                    if key in allow:
                        continue
                    print("EXTERNAL  %s:%d  `%s` is defined here with external linkage -- "
                          "one strong symbol per including object" % (path, line, name))
                    print("          %s" % sig)
                    reported += 1

    stale = 0
    for key in sorted(allow):
        if key not in seen:
            print("STALE     %s -- no such definition any more; remove the entry" % key)
            stale += 1

    print("header_odr_audit: %d headers, %d external definitions, %d unlisted, %d stale entries"
          % (headers, defs, reported, stale))
    return 1 if (reported or stale) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
