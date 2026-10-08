#!/usr/bin/env python3
"""ctest_cached.py -- run ctest, leaving out every test that has already PASSED on the same inputs.

    tools/ctest_cached.py [--test-dir build] [ctest arguments ...]
    tools/ctest_cached.py --list              which tests would run, which are held
    tools/ctest_cached.py --explain <test>    every input in that test's fingerprint
    tools/ctest_cached.py --no-cache          run everything (passes are still recorded)

WHY. The suite is 856 tests and about 1 h 45 on a GB10, and ctest has no map from a source file to the tests
it can change. Choosing "the tests that cover what I touched" by hand missed a pimpleFoam regression for four
days (2026-10-04 to 2026-10-08): the change was in the shared AMG code and the selection was interFoam's.

HOW. A test's outcome is a function of its inputs, so each test gets a FINGERPRINT of them:
  * its command line, its ENVIRONMENT property, and every BRAE_* variable this process was started with;
  * the BYTES of every file on the command line and of every build target its DEPENDS property names;
  * for a script: the scripts and files its code names (followed recursively), every build executable its
    code names, and a whole source directory where the code reaches into one by a computed path;
  * for a binary: every file or fixture directory of this repository whose path is compiled into it;
  * the OpenFOAM installation, the tutorials the gates stage, and the GPU and its driver.
brae_core is a STATIC library, so a test binary holds exactly the code it can reach: its bytes change when
that code changes, with no map to keep. A test whose fingerprint has a recorded pass is left out; the rest go
to ctest by name. ONLY A PASS IS RECORDED -- a failure, a timeout or a skip runs again every time -- and a
test whose own run changes its fingerprint (it writes into a directory it reads) is never recorded.

WHAT IT CANNOT SEE. An input no rule above reaches. So: when unsure a rule takes more, --no-cache is one flag
away, and a release is cut from a run made with it.

The store is outside the build tree (BRAE_TEST_CACHE, default ~/.cache/brae/test-results), so a fresh checkout
or a rebuilt tree finds its passes again as long as its binaries come out the same bytes.
BRAE_TEST_CACHE_SALT is mixed into every fingerprint: change it to set every recorded pass aside.
"""
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

FORMAT = 'ctest-cached-2'
# Directories of the repository that hold inputs. A script that reaches into one by a path it computes
# (`$ROOT/validation/$f`) is given the whole directory.
DATA_ROOTS = ('validation', 'demo', 'bench', 'manifest', 'factory', 'src', 'tests', 'tools', 'docs')
SKIP_DIRS = {'__pycache__', '.git', 'build', 'Testing', 'processor0'}
SCRIPT_SUFFIXES = ('.sh', '.py', '.cmake')
FILTER_OPTIONS = {'-R', '--tests-regex', '-E', '--exclude-regex', '-L', '--label-regex', '-LE',
                  '--label-exclude', '-I', '--tests-information'}


def loadedBytesHash(path):
    """An ELF file's fingerprint: the sections it LOADS, by name, and nothing else.

    A binary rebuilt from unchanged sources is not the same file. MEASURED 2026-10-08 on alpha_eqn_cpp.cu
    compiled twice: 4 bytes of 468,824 differ, all in .strtab -- nvcc names its temporary file
    (`tmpxft_<pid>_...cudafe1.cpp`) in the symbol table -- while .text, .rodata, .data and .nv_fatbin are the
    same bytes. Every executable linked from it then differs too, in its symbol table and in the build id
    the linker takes over the whole file. Neither is code a test can run, so neither is in the fingerprint:
    a rebuilt tree finds its recorded passes again. Returns None for a file that is not a 64-bit ELF."""
    with open(path, 'rb') as f:
        head = f.read(64)
        if len(head) < 64 or head[:4] != b'\x7fELF' or head[4] != 2 or head[5] != 1:
            return None
        shoff, = struct.unpack_from('<Q', head, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from('<HHH', head, 0x3A)
        if shoff == 0 or shnum == 0 or shentsize < 64:
            return None
        f.seek(shoff)
        table = f.read(shentsize*shnum)
        sections = []
        for k in range(shnum):
            name, kind, flags, addr, offset, size = struct.unpack_from('<IIQQQQ', table, k*shentsize)
            sections.append((name, kind, flags, offset, size))
        f.seek(sections[shstrndx][3])
        names = f.read(sections[shstrndx][4])
        h = hashlib.sha256()
        for name, kind, flags, offset, size in sections:
            label = names[name:names.index(b'\0', name)]
            # SHF_ALLOC, not SHT_NOBITS (no bytes in the file), and not the linker's id of the whole file
            if not (flags & 2) or kind == 8 or label == b'.note.gnu.build-id':
                continue
            h.update(label + b'\0' + struct.pack('<QQ', flags, size))
            f.seek(offset)
            left = size
            while left > 0:
                block = f.read(min(left, 1 << 22))
                if not block:
                    break
                h.update(block)
                left -= len(block)
        return 'loaded:' + h.hexdigest()


class Inputs:
    """File and directory fingerprints, each computed once a run and remembered across runs by
    (path, size, mtime) -- a file rewritten with the bytes it had is hashed again and comes out the same."""

    def __init__(self, store):
        self.memoPath = os.path.join(store, 'file-memo.json')
        try:
            self.memo = json.load(open(self.memoPath))
        except (OSError, ValueError):
            self.memo = {}
        self.fresh = {}
        self.trees = {}
        self.embedded = {}

    def stamp(self, path):
        st = os.stat(path)
        return '%d:%d' % (st.st_size, st.st_mtime_ns)

    def fileHash(self, path):
        path = os.path.realpath(path)
        stamp = self.stamp(path)
        got = self.fresh.get(path)
        if got and got[0] == stamp:
            return got[1]
        old = self.memo.get(path)
        if old and old[0] == stamp:
            self.fresh[path] = old
            return old[1]
        digest = loadedBytesHash(path)
        if digest is None:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for block in iter(lambda: f.read(1 << 22), b''):
                    h.update(block)
            digest = h.hexdigest()
        self.fresh[path] = [stamp, digest]
        return digest

    def hashMany(self, paths):
        todo = sorted(set(os.path.realpath(p) for p in paths if os.path.isfile(p)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(self.fileHash, todo))

    def treeHash(self, root, byContent):
        """Every file under root: its bytes inside the repository, its size and time outside it (an OpenFOAM
        installation is not rewritten in place, and hashing 2 GB of tutorials would be the slow part)."""
        key = (os.path.realpath(root), byContent)
        if key in self.trees:
            return self.trees[key]
        rows = []
        for base, dirs, files in os.walk(key[0]):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                if name.endswith('.pyc'):
                    continue
                full = os.path.join(base, name)
                if not os.path.isfile(full):
                    continue
                rel = os.path.relpath(full, key[0])
                rows.append('%s %s' % (rel, self.fileHash(full) if byContent else self.stamp(full)))
        self.trees[key] = hashlib.sha256('\n'.join(rows).encode()).hexdigest()
        return self.trees[key]

    def save(self):
        self.memo.update(self.fresh)
        live = dict((p, v) for p, v in self.memo.items() if os.path.exists(p.split(':', 1)[-1]))
        tmp = self.memoPath + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(live, f)
        os.replace(tmp, self.memoPath)


def codeLines(path):
    """A script's text without its comment lines: a gate that only NAMES another gate in a comment does not
    read it, and following comments tied every write gate to every other."""
    try:
        text = open(path, errors='replace').read()
    except OSError:
        return ''
    return '\n'.join(line for line in text.split('\n') if not line.lstrip().startswith('#'))


class Closure:
    """What a script or a binary reaches beyond its own bytes."""

    def __init__(self, repo, build, inputs):
        self.repo = repo
        self.build = build
        self.inputs = inputs
        self.scripts = {}
        self.closures = {}
        names = []
        for name in sorted(os.listdir(build)):
            full = os.path.join(build, name)
            if os.path.isfile(full) and os.access(full, os.X_OK) and '.' not in name:
                names.append(name)
        self.exeNames = set(names)
        # `brae` alone is a word of every comment and message here: it counts as the launcher only where it
        # ends a path (`$BUILD/brae`, `$(dirname "$BIN")/brae`). Every other target name is its own word.
        others = sorted((n for n in names if n != 'brae'), key=len, reverse=True)
        self.exeRe = re.compile(r'(?<![A-Za-z0-9_])(' + '|'.join(map(re.escape, others)) + r')(?![A-Za-z0-9_])') \
            if others else None
        self.braeRe = re.compile(r'/brae(?![A-Za-z0-9_.\-/])')
        self.pathRe = re.compile(r'(?:\.\.?/)*[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*')
        self.repoRe = re.compile(re.escape(repo.encode()) + rb'/[A-Za-z0-9_.\-/]+')

    def inRepo(self, path):
        real = os.path.realpath(path)
        return real == self.repo or real.startswith(self.repo + os.sep)

    def inBuild(self, path):
        real = os.path.realpath(path)
        return real == self.build or real.startswith(self.build + os.sep)

    def direct(self, path):
        """(files, executables, whole directories) ONE script's code names."""
        if path in self.scripts:
            return self.scripts[path]
        files, exes, trees = set(), set(), set()
        text = codeLines(path)
        here = os.path.dirname(path)
        bases = [here, os.path.dirname(here), self.repo, os.path.join(self.repo, 'tests'),
                 os.path.join(self.repo, 'tools')]
        if self.exeRe:
            exes.update(self.exeRe.findall(text))
        if self.braeRe.search(text):
            exes.add('brae')
        for root in DATA_ROOTS:
            # reached by a computed path: `validation/$f`, `src/"$dir"`, `tests/${name}`
            if re.search(r'(?<![A-Za-z0-9_.])' + re.escape(root) + r'/["\']?[$`]', text):
                trees.add(os.path.join(self.repo, root))
        # `$ROOT/tools/x.py` and `${BUILD}/y` read as one word with the variable's name in front: the path
        # is what follows it. (lib.sh's comparer, `$ROOT/tools/foam_time_compare.py`, was missed without this.)
        bare = re.sub(r'\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/', ' ', text)
        for token in set(self.pathRe.findall(text)) | set(self.pathRe.findall(bare)):
            if '/' not in token and '.' not in token:
                continue
            for base in bases:
                cand = os.path.normpath(os.path.join(base, token))
                if not self.inRepo(cand) or self.inBuild(cand):
                    continue
                if os.path.isfile(cand):
                    files.add(cand)
                    break
                if os.path.isdir(cand) and '/' in token and cand != self.repo:
                    rel = os.path.relpath(cand, self.repo).split(os.sep)
                    # a fixture directory named outright; a bare `tests/` or `src/` is not a fixture
                    if rel[0] in DATA_ROOTS and len(rel) >= 2:
                        trees.add(cand)
                    break
        self.scripts[path] = (files, exes, trees)
        return self.scripts[path]

    def ofScript(self, path):
        """...and through every script it names, and the ones those name."""
        path = os.path.realpath(path)
        if path in self.closures:
            return self.closures[path]
        files, exes, trees = set(), set(), set()
        seen, todo = set(), [path]
        while todo:
            one = todo.pop()
            if one in seen:
                continue
            seen.add(one)
            f, e, t = self.direct(one)
            files |= f
            exes |= e
            trees |= t
            todo.extend(x for x in f if x.endswith(SCRIPT_SUFFIXES) and x not in seen)
        self.closures[path] = (files, exes, trees)
        return self.closures[path]

    def ofBinary(self, path):
        """Files and fixture directories of this repository whose path is compiled into a binary."""
        path = os.path.realpath(path)
        stamp = self.inputs.stamp(path)
        got = self.inputs.fresh.get('embedded:' + path) or self.inputs.memo.get('embedded:' + path)
        if not got or got[0] != stamp:
            found = set()
            try:
                data = open(path, 'rb').read()
            except OSError:
                data = b''
            for m in set(self.repoRe.findall(data)):
                cand = os.path.normpath(m.decode(errors='replace'))
                if not self.inBuild(cand) and cand != self.repo:
                    found.add(cand)
            got = [stamp, sorted(found)]
        self.inputs.fresh['embedded:' + path] = got
        files = set(c for c in got[1] if os.path.isfile(c))
        trees = set(c for c in got[1] if os.path.isdir(c))
        return files, trees


def machineIdentity(env):
    """What every test stands on that is not in the tree: OpenFOAM, the tutorials, the GPU and its driver."""
    rows = []
    of = env.get('WM_PROJECT_DIR') or '/usr/lib/openfoam/openfoam2412'
    for sub in ('etc/bashrc', 'platforms'):
        full = os.path.join(of, sub)
        if os.path.isfile(full):
            st = os.stat(full)
            rows.append('of %s %d %d' % (sub, st.st_size, st.st_mtime_ns))
        elif os.path.isdir(full):
            for base, dirs, files in os.walk(full):
                dirs.sort()
                for name in sorted(files):
                    if base.endswith('/bin') or name.startswith(('libOpenFOAM', 'libfiniteVolume')):
                        st = os.stat(os.path.join(base, name))
                        rows.append('of %s %d %d' % (name, st.st_size, st.st_mtime_ns))
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = 'no nvidia-smi'
    rows.append('gpu ' + gpu)
    return rows


def property_(test, name, default=None):
    for p in test.get('properties', []):
        if p['name'] == name:
            return p['value']
    return default


def fingerprint(test, closure, inputs, machine, globalEnv, explain=None):
    """One sha256 over everything the test's outcome depends on. `explain` collects the rows."""
    repo, build = closure.repo, closure.build
    rows = [FORMAT, 'salt ' + os.environ.get('BRAE_TEST_CACHE_SALT', '')]
    command = test.get('command') or []
    rows.append('command ' + json.dumps(command))
    rows.append('environment ' + json.dumps(sorted(property_(test, 'ENVIRONMENT', []))))
    rows.append('cwd ' + str(property_(test, 'WORKING_DIRECTORY', '')))
    rows.extend(globalEnv)
    rows.extend(machine)
    files, trees, outside = set(), set(), set()

    def take(path):
        if os.path.isfile(path):
            files.add(os.path.realpath(path))
            if path.endswith(SCRIPT_SUFFIXES):
                more = closure.ofScript(path)
                files.update(more[0])
                trees.update(more[2])
                for exe in more[1]:
                    take(os.path.join(build, exe))
            elif closure.inBuild(path):
                more = closure.ofBinary(path)
                files.update(more[0])
                trees.update(more[1])
        elif os.path.isdir(path) and not closure.inBuild(path):
            real = os.path.realpath(path)
            if real == repo:
                # a test handed the whole tree (the registration audit): the sources and the test set
                for part in ('CMakeLists.txt', 'src', 'tests', 'tools'):
                    take(os.path.join(repo, part))
            elif closure.inRepo(real):
                trees.add(real)
            else:
                outside.add(real)

    for token in command:
        # NAME=value on the command line too (`cmake -E env BRAE_X=dir prog`)
        for part in (token, token.split('=', 1)[-1]):
            if os.path.isabs(part):
                take(part)
    for entry in property_(test, 'ENVIRONMENT', []):
        if '=' in entry:
            value = entry.split('=', 1)[1]
            if os.path.isabs(value):
                take(value)
    for target in property_(test, 'DEPENDS', []) or []:
        take(os.path.join(build, target))
    for f in sorted(files):
        rows.append('file %s %s' % (f, inputs.fileHash(f)))
    for t in sorted(trees):
        rows.append('tree %s %s' % (t, inputs.treeHash(t, True)))
    for t in sorted(outside):
        rows.append('outside %s %s' % (t, inputs.treeHash(t, False)))
    if explain is not None:
        explain.extend(rows)
    return hashlib.sha256('\n'.join(rows).encode()).hexdigest()


def splitArguments(argv):
    """Ours, the filters (they pick the tests; the run is then by name), and the rest for ctest."""
    ours = {'testDir': 'build', 'noCache': False, 'list': False, 'explain': None, 'store': None}
    filters, rest = [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--test-dir':
            ours['testDir'] = argv[i + 1]
            i += 2
        elif a == '--no-cache':
            ours['noCache'] = True
            i += 1
        elif a == '--list':
            ours['list'] = True
            i += 1
        elif a == '--explain':
            ours['explain'] = argv[i + 1]
            i += 2
        elif a == '--store':
            ours['store'] = argv[i + 1]
            i += 2
        elif a in FILTER_OPTIONS:
            filters += [a, argv[i + 1]]
            i += 2
        else:
            rest.append(a)
            i += 1
    return ours, filters, rest


def main():
    ours, filters, rest = splitArguments(sys.argv[1:])
    build = os.path.realpath(ours['testDir'])
    ctest = os.environ.get('BRAE_CTEST', 'ctest')
    listing = subprocess.run([ctest, '--test-dir', build, '--show-only=json-v1'] + filters,
                             capture_output=True, text=True)
    if listing.returncode != 0:
        sys.stderr.write(listing.stderr)
        return listing.returncode
    tests = [t for t in json.loads(listing.stdout)['tests'] if t.get('command')]
    repo = build
    cache = os.path.join(build, 'CMakeCache.txt')
    for line in open(cache, errors='replace'):
        if line.startswith('CMAKE_HOME_DIRECTORY:'):
            repo = os.path.realpath(line.split('=', 1)[1].strip())
    store = ours['store'] or os.environ.get('BRAE_TEST_CACHE') or os.path.expanduser('~/.cache/brae/test-results')
    os.makedirs(os.path.join(store, 'passed'), exist_ok=True)
    inputs = Inputs(store)
    closure = Closure(repo, build, inputs)
    machine = machineIdentity(os.environ)
    globalEnv = ['env %s=%s' % (k, v) for k, v in sorted(os.environ.items())
                 if (k.startswith('BRAE_') or k == 'CUDA_VISIBLE_DEVICES')
                 and k not in ('BRAE_TEST_CACHE', 'BRAE_TEST_CACHE_SALT', 'BRAE_CTEST')]

    t0 = time.time()
    # the build's executables in one parallel pass: they are most of the bytes
    inputs.hashMany(os.path.join(build, n) for n in closure.exeNames)

    def keysOf(group):
        return dict((t['name'], fingerprint(t, closure, inputs, machine, globalEnv)) for t in group)

    keys = keysOf(tests)
    inputs.save()
    hashSeconds = time.time() - t0

    def recorded(key):
        return os.path.exists(os.path.join(store, 'passed', key[:2], key))

    if ours['explain']:
        for t in tests:
            if t['name'] == ours['explain']:
                rows = []
                key = fingerprint(t, closure, inputs, machine, globalEnv, rows)
                print('\n'.join(rows))
                print('fingerprint %s, %s' % (key, 'a pass is recorded' if recorded(key) else 'no pass recorded'))
                return 0
        sys.stderr.write('no test named %s among the %d selected\n' % (ours['explain'], len(tests)))
        return 2

    held = [t['name'] for t in tests if recorded(keys[t['name']]) and not ours['noCache']]
    heldSet = set(held)
    toRun = [t for t in tests if t['name'] not in heldSet]
    print('ctest_cached: %d tests selected, %d passed before on the same inputs, %d to run (fingerprints in '
          '%.1f s; store %s)' % (len(tests), len(held), len(toRun), hashSeconds, store), flush=True)
    if ours['list']:
        for t in tests:
            print('%s %s' % ('held' if t['name'] in heldSet else 'run ', t['name']))
        return 0
    if not toRun:
        print('ctest_cached: nothing to run')
        return 0

    junit = os.path.join(build, 'Testing', 'ctest_cached_junit.xml')
    os.makedirs(os.path.dirname(junit), exist_ok=True)
    if os.path.exists(junit):
        os.remove(junit)
    pattern = '^(' + '|'.join(re.escape(t['name']) for t in toRun) + ')$'
    run = subprocess.run([ctest, '--test-dir', build, '-R', pattern, '--output-junit', junit] + rest)

    passed, other = [], []
    try:
        for case in ET.parse(junit).getroot().iter('testcase'):
            bad = case.find('failure') is not None or case.find('error') is not None \
                or case.find('skipped') is not None
            if case.get('status') == 'run' and not bad:
                passed.append(case.get('name'))
            else:
                other.append(case.get('name'))
    except (OSError, ET.ParseError):
        sys.stderr.write('ctest_cached: ctest left no result file; nothing is recorded\n')
        return run.returncode or 1

    # a pass is recorded under the fingerprint taken BEFORE the run, and only if the run left it standing
    byName = dict((t['name'], t) for t in tests)
    # ...which is asked of the tree as it stands NOW: what this run remembered of directories and scripts
    # is from before the tests ran
    inputs.trees.clear()
    closure.scripts.clear()
    closure.closures.clear()
    after = keysOf([byName[n] for n in passed if n in byName])
    inputs.save()
    moved = [n for n in passed if after.get(n) != keys.get(n)]
    head = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                          text=True).stdout.strip()
    for n in passed:
        if n in moved or n not in keys:
            continue
        d = os.path.join(store, 'passed', keys[n][:2])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, keys[n]), 'w') as f:
            json.dump({'test': n, 'when': time.strftime('%Y-%m-%d %H:%M'), 'head': head}, f)
    print('ctest_cached: ran %d: %d passed, %d did not; %d held from earlier runs'
          % (len(passed) + len(other), len(passed), len(other), len(held)))
    if moved:
        print('ctest_cached: %d passed and are NOT recorded -- their own run changed their inputs: %s'
              % (len(moved), ' '.join(sorted(moved)[:12]) + (' ...' if len(moved) > 12 else '')))
    json.dump({'selected': len(tests), 'held': len(held), 'ran': len(passed) + len(other),
               'passed': len(passed), 'notPassed': sorted(other), 'notRecorded': sorted(moved)},
              open(os.path.join(build, 'Testing', 'ctest_cached_last.json'), 'w'))
    return run.returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BrokenPipeError:
        # `... --list | head`: the reader left, and that is not an error of this tool's
        os._exit(0)
