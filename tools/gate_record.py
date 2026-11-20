#!/usr/bin/env python3
"""gate_record.py -- run one gate, stream its output unchanged, keep its exit code, and record what ran in result.json.

    tools/gate_record.py [options] -- <gate command> [args ...]

    --out DIR        where the record goes: DIR/output.log (the gate's stdout and stderr, byte for byte as streamed)
                     and DIR/result.json beside it. Default: the current directory.
    --binary PATH    a brae binary the gate runs (repeatable). Without one, every BRAE_*BIN* variable naming a file is
                     taken, and if there is none either, the solver executables in <source>/build.
    --source DIR     the brae source tree whose commit is recorded (default: the tree this script is in)
    --site, --case, --level TEXT
                     free-form labels, copied into the record as given
    --ofbashrc FILE  OpenFOAM's bashrc to read the version from (default: $OF_BASHRC, then $OFBASHRC, then
                     /usr/lib/openfoam/openfoam2412/etc/bashrc, as the gates find it)

The gate runs in this process's environment and working directory; this tool sets nothing for it. Its exit code is
this tool's exit code (128 + N when a signal N stopped it, as a shell reports it). SIGINT and SIGTERM sent to this
tool are passed on to the gate.

result.json, schema "brae-gate-record/1" (every key is always present; null where a value could not be had):

    schema          "brae-gate-record/1"
    command         [argv ...] as given after `--`
    cwd             the working directory the gate ran in
    env             {name: value} of the environment's BRAE_* and OF* variables only
    start, end      ISO 8601 UTC times, to the second
    wall_s          seconds between them, to the millisecond
    exit_code       the gate's exit code as this tool returns it
    outcome         "pass" (0), "skip" (77 or 125, the suites' SKIP_RETURN_CODEs), "fail" (anything else)
    lines           {"ok": n, "FAIL": n, "hist": n, "open": n, "skip": n}: counts of the gate's own verdict lines
    fail_lines      [the FAIL lines, verbatim]
    open_lines      [the open-item lines, verbatim]
    skip_lines      [the SKIP lines, verbatim]
    brae            {"commit": full hash, "describe": `git describe --always --tags --dirty --abbrev=8`,
                     "dirty": true/false (tracked files changed), "source": the tree}; from BRAE_SOURCE_COMMIT when the
                     tree has no .git (as the build stamps it), with "dirty" null
    binaries        [{"path", "sha256", "bytes"}] for each binary found as above; "binaries_from" says how
    gpu             {"name", "driver", "cuda"} from nvidia-smi, null without one
    openfoam        {"bashrc", "version"}: WM_PROJECT_VERSION as the bashrc sets it, null without one
    hostname        this machine's name
    labels          {"site", "case", "level"} as given

THE VERDICT LINES, as the gates print them (a ctest `NNN: ` prefix is ignored, so a ctest -V log reads the same):

    tests/rho_central_gate_lib.sh      `ok   <what>`, `FAIL <what>`, `hist <what>`, and under a registered FAIL
                                       `     [open item, <category>: <note>]`
    the say() of the simpler gates     `  <what, padded>  ok` / `... FAIL`: the verdict is the line's last word
    and their one-line forms           `  ok:   <what>`, `  FAIL: <what>`, `FAIL: <what>`, `FAIL`
    a skip                             `SKIP: <why>`

A line is counted once, by the first of: open, hist, FAIL, ok, skip. The counts are what the gate printed, not a
judgement of it: the exit code is the gate's verdict.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys

SCHEMA = 'brae-gate-record/1'
OF_DEFAULT = '/usr/lib/openfoam/openfoam2412/etc/bashrc'
SOLVERS = ('brae', 'brae_pimpleFoam', 'brae_interFoam', 'brae_rhoSimpleFoam', 'brae_rhoCentralFoam')
SKIP_CODES = (77, 125)

CTEST_PREFIX = re.compile(r'^\d+: ')
OPEN = re.compile(r'^\s*\[open item[,:]')
HIST = re.compile(r'^hist\s')
FAIL = re.compile(r'^FAIL(\s|:|$)|^\s+FAIL:|\sFAIL$|^\s+FAIL$')
OK = re.compile(r'^ok\s|^\s*ok:|\sok$')
SKIP = re.compile(r'^\s*SKIP\b')


def classify(line):
    """the verdict class of one output line, or None"""
    s = CTEST_PREFIX.sub('', line.rstrip('\r\n')).rstrip()
    for name, pattern in (('open', OPEN), ('hist', HIST), ('FAIL', FAIL), ('ok', OK), ('skip', SKIP)):
        if pattern.search(s):
            return name, s
    return None, s


def count(text):
    lines = {'ok': 0, 'FAIL': 0, 'hist': 0, 'open': 0, 'skip': 0}
    kept = {'FAIL': [], 'open': [], 'skip': []}
    for raw in text.splitlines():
        kind, s = classify(raw)
        if kind is None:
            continue
        lines[kind] += 1
        if kind in kept:
            kept[kind].append(s)
    return lines, kept


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t):
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')


def run(cmd, timeout=20, env=None):
    """a helper command's stdout, or None"""
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.decode(errors='replace').strip() if r.returncode == 0 else None


def braeSource(src):
    info = {'commit': None, 'describe': None, 'dirty': None, 'source': src}
    if os.path.exists(os.path.join(src, '.git')):
        info['commit'] = run(['git', '-C', src, 'rev-parse', 'HEAD'])
        info['describe'] = run(['git', '-C', src, 'describe', '--always', '--tags', '--dirty', '--abbrev=8'])
        status = run(['git', '-C', src, 'status', '--porcelain', '--untracked-files=no'])
        info['dirty'] = None if status is None else bool(status)
    else:
        stamp = os.path.join(src, 'BRAE_SOURCE_COMMIT')
        if os.path.isfile(stamp):
            words = open(stamp).read().split()
            if words and re.fullmatch(r'[0-9a-f]+', words[0]):
                info['commit'] = info['describe'] = words[0]
    return info


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def binaries(given, src):
    if given:
        paths, origin = given, '--binary'
    else:
        paths = [v for k, v in sorted(os.environ.items())
                 if k.startswith('BRAE_') and 'BIN' in k and v and os.path.isfile(v)]
        origin = 'environment (BRAE_*BIN*)'
        if not paths:
            build = os.path.join(src, 'build')
            paths = [os.path.join(build, n) for n in SOLVERS if os.path.isfile(os.path.join(build, n))]
            origin = 'the solver executables in %s' % build
    out = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            out.append({'path': p, 'sha256': sha256(p), 'bytes': os.path.getsize(p)})
        else:
            out.append({'path': p, 'sha256': None, 'bytes': None})
    return out, origin


def gpu():
    q = run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])
    if not q:
        return None
    name, _, driver = q.splitlines()[0].rpartition(',')
    head = run(['nvidia-smi']) or ''
    m = re.search(r'CUDA Version:\s*([0-9.]+)', head)
    return {'name': name.strip(), 'driver': driver.strip(), 'cuda': m.group(1) if m else None}


def openfoam(bashrc):
    if not bashrc or not os.path.isfile(bashrc):
        return {'bashrc': bashrc, 'version': None}
    # `set --`: the bashrc evaluates the arguments it is sourced with (etc/config.sh/setup)
    v = run(['bash', '-c', 'set +u; set --; . "$0" > /dev/null 2>&1; printf %s "${WM_PROJECT_VERSION:-}"', bashrc], 60)
    return {'bashrc': bashrc, 'version': v or None}


def main():
    ap = argparse.ArgumentParser(usage=__doc__.split('\n\n')[1].strip(), add_help=True)
    ap.add_argument('--out', default='.')
    ap.add_argument('--binary', action='append', default=[])
    ap.add_argument('--source', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--site')
    ap.add_argument('--case')
    ap.add_argument('--level')
    ap.add_argument('--ofbashrc', default=os.environ.get('OF_BASHRC') or os.environ.get('OFBASHRC') or OF_DEFAULT)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd:
        ap.error('no gate command: give it after --')
    os.makedirs(a.out, exist_ok=True)
    logPath = os.path.join(a.out, 'output.log')

    start = now()
    try:
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as e:
        sys.stderr.write('gate_record: cannot run %s: %s\n' % (cmd[0], e))
        return 127

    def forward(sig, _frame):
        try:
            child.send_signal(sig)
        except OSError:
            pass
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, forward)

    out = sys.stdout.buffer
    with open(logPath, 'wb') as log:
        fd = child.stdout.fileno()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            out.write(chunk)
            out.flush()
            log.write(chunk)
    rc = child.wait()
    end = now()
    code = 128 - rc if rc < 0 else rc

    text = open(logPath, 'rb').read().decode(errors='replace')
    lines, kept = count(text)
    bins, origin = binaries(a.binary, os.path.abspath(a.source))
    record = {
        'schema': SCHEMA,
        'command': cmd,
        'cwd': os.getcwd(),
        'env': {k: v for k, v in sorted(os.environ.items()) if k.startswith('BRAE_') or k.startswith('OF')},
        'start': iso(start),
        'end': iso(end),
        'wall_s': round((end - start).total_seconds(), 3),
        'exit_code': code,
        'outcome': 'pass' if code == 0 else 'skip' if code in SKIP_CODES else 'fail',
        'lines': lines,
        'fail_lines': kept['FAIL'],
        'open_lines': kept['open'],
        'skip_lines': kept['skip'],
        'brae': braeSource(os.path.abspath(a.source)),
        'binaries': bins,
        'binaries_from': origin,
        'gpu': gpu(),
        'openfoam': openfoam(a.ofbashrc),
        'hostname': socket.gethostname(),
        'labels': {'site': a.site, 'case': a.case, 'level': a.level},
    }
    with open(os.path.join(a.out, 'result.json'), 'w') as f:
        json.dump(record, f, indent=1)
        f.write('\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
