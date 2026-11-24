#!/usr/bin/env python3
"""tutorial_sweep.py -- which OpenFOAM tutorials brae sets up, found by running each tutorial's own scripts with
`brae check --setup` standing in for the solver. No OpenFOAM solver is ever run.

    tools/tutorial_sweep.py --solvers simpleFoam,pimpleFoam --tutorials <OpenFOAM tutorials> --work <dir>
                            [--brae <binary>] [--timeout 1800] [--jobs 1] [--skip-mesh-if REGEX] [--match REGEX]
                            [--script auto|pre] [--ofbashrc FILE|none] [--lock FILE] [--out PREFIX] [--keep]

WHAT IS A TUTORIAL. Every directory under --tutorials whose system/controlDict names one of --solvers as its
`application` is a case. A case is run by the nearest directory at or above it, below --tutorials, that has an Allrun
(Allrun-parallel, Allrun-serial), which is how OpenFOAM's own foamRunTutorials picks it: a tutorial whose Allrun runs
several cases, or composes one from setups.orig/, is one tutorial. A case with no such script is one tutorial too.
--match keeps only the tutorials whose path (relative to --tutorials) the regex finds.

HOW ONE IS RUN. The tutorial is copied to --work/<its path> and run in OpenFOAM's environment (--ofbashrc, default
$OF_BASHRC, then $OFBASHRC, then the v2412 install; `none` takes this process's environment as it is) with a
directory of shims first on PATH:
  - each of --solvers is a shim that runs `brae check --setup -case <case>` where the solver would run, and records
    it. Under mpirun (`<solver> -parallel`, OpenFOAM's runParallel) only rank 0 checks; a mesh that exists only in
    processor*/ is first put together with reconstructParMesh -constant and reconstructPar -withZero (OpenFOAM
    utilities), since one GPU reads the undecomposed case;
  - every other OpenFOAM solver (the EXE names under $FOAM_SOLVERS, and the *Foam executables in $FOAM_APPBIN and on
    PATH that are not *ToFoam mesh converters) is a stub that runs nothing and is named in the record: a potentialFoam
    run before the solver is skipped, so the case brae checks is the case as meshed, without it. With no such list the
    sweep stops rather than risk running one. A script that calls a solver by its full path is not caught.
--script auto (the default) runs the tutorial's Allrun, so everything it does before the solver is done; failing
that its Allrun.pre or Allmesh, then each case is checked; failing that, OpenFOAM's default: blockMesh where there
is a blockMeshDict, then the check. --script pre prefers Allrun.pre or Allmesh to Allrun where both exist (meshing
only, faster). --skip-mesh-if REGEX runs nothing for a tutorial whose path or meshing scripts the regex finds (for
example snappyHexMesh), and says so. --timeout is per tutorial (the whole process group is stopped); --jobs runs
that many tutorials at once; --lock FILE runs each brae check under `flock FILE` (one GPU shared by the jobs; the
wait for it counts against --timeout).

WHAT IS RECORDED, one row per check brae made, and one row for a tutorial whose scripts never reached its solver:
    solver, tutorial, case        the application, the tutorial's path, the checked case's path within the work copy
    script                        what ran: Allrun, Allrun.pre, Allmesh, blockMesh, none
    mesh                          ok (a polyMesh at the check, or after the meshing script), failed, skipped, absent
    verdict                       set up      brae built the case and stopped before its first step
                                  checked     brae's checks passed and this solver path has no set-up stop yet
                                              (`--setup: not run`); notices, where brae printed them, in `notes`
                                  refused     brae refused the case: `refusal` holds the text
                                  error       brae stopped without a refusal (`refusal` holds its last lines)
                                  not reached the scripts ended without running the solver
                                  timeout     --timeout stopped the tutorial first
                                  skipped     --skip-mesh-if matched
    refusal                       the refusal or error text
    memory_gib, memory_total_gib  `GPU memory in use X of Y GiB` as brae check --setup reports it (cudaMemGetInfo:
                                  on a unified-memory GPU it counts more than brae's own arrays)
    check_s, wall_s               the check's seconds; the whole tutorial's seconds
    parallel, stubbed, notes      the solver was called with -parallel; the OpenFOAM solvers stubbed; anything else
    setup_notices (JSON only)     what the solver's own set-up said it runs differently (`brae NOTICE`, `brae: skipping`)
Written to <out>.csv and <out>.json (default <work>/tutorial_sweep); each tutorial's logs are kept in
<work>/_logs/<tutorial> and its copy is deleted unless --keep.
"""
import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

OF_DEFAULT = '/usr/lib/openfoam/openfoam2412/etc/bashrc'
SCRIPTS = ('Allrun', 'Allrun-parallel', 'Allrun-serial')
MESHERS = ('Allrun.pre', 'Allmesh')
FIELDS = ('solver', 'tutorial', 'case', 'script', 'mesh', 'verdict', 'refusal', 'memory_gib', 'memory_total_gib',
          'check_s', 'wall_s', 'parallel', 'stubbed', 'notes')

SHIM = r'''#!/usr/bin/env bash
# tutorial_sweep.py: `brae check --setup` where the tutorial runs this solver; the solver is not run
args="$*"
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-0}}}
[ "$rank" = 0 ] || exit 0
case_dir=.
parallel=0
while [ $# -gt 0 ]
do
    case "$1" in
        -case) case_dir=${2:-.}; shift ;;
        -parallel) parallel=1 ;;
    esac
    shift
done
REC='@REC@'
[ -d "$REC" ] || exit 1
n=1
until mkdir "$REC/$n" 2> /dev/null; do n=$((n + 1)); [ $n -lt 100000 ] || exit 1; done
d="$REC/$n"
basename "$0" > "$d/solver"
printf '%s\n' "$args" > "$d/args"
( cd "$case_dir" && pwd ) > "$d/case"
echo $parallel > "$d/parallel"
owner() { [ -f "$1/constant/polyMesh/owner" ] || [ -f "$1/constant/polyMesh/owner.gz" ]; }
if [ $parallel = 1 ] && owner "$case_dir/processor0"
then
    # one GPU reads the undecomposed case: a mesh that exists only in processor*/, or is newer there (snappyHexMesh
    # in parallel over blockMesh's background mesh), is put together first, with the start fields
    p0="$case_dir/processor0/constant/polyMesh"; r="$case_dir/constant/polyMesh"
    if ! owner "$case_dir" || [ "$(ls -t "$p0"/owner* "$r"/owner* 2> /dev/null | head -1 | xargs dirname)" = "$p0" ]
    then
        ( cd "$case_dir" && reconstructParMesh -constant > log.tutorial_sweep.reconstructParMesh 2>&1 \
          && reconstructPar -withZero > log.tutorial_sweep.reconstructPar 2>&1 ) \
            && echo "the decomposed mesh and start fields were reconstructed for the check" >> "$d/notes" \
            || echo "reconstructing the decomposed mesh failed (log.tutorial_sweep.reconstruct*)" >> "$d/notes"
    fi
fi
owner "$case_dir" && echo ok > "$d/mesh" || echo absent > "$d/mesh"
t0=$(date +%s.%N)
if [ -n '@LOCK@' ]
then
    flock -w 7200 '@LOCK@' '@BRAE@' check --setup -case "$case_dir" > "$d/check.log" 2>&1
else
    '@BRAE@' check --setup -case "$case_dir" > "$d/check.log" 2>&1
fi
echo $? > "$d/rc"
echo "$t0 $(date +%s.%N)" > "$d/time"
exit 0
'''

STUB = r'''#!/usr/bin/env bash
# tutorial_sweep.py: this OpenFOAM solver is not run (the sweep never runs one); the tutorial goes on without it
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-0}}}
[ "$rank" = 0 ] && basename "$0" >> '@REC@/../stubbed'
exit 0
'''


def application(controlDict):
    try:
        text = open(controlDict, errors='replace').read()
    except OSError:
        return None
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'//[^\n]*', ' ', text)
    m = re.search(r'(?m)^\s*application\s+([A-Za-z0-9_]+)\s*;', text)
    return m.group(1) if m else None


def discover(root, solvers, match):
    """{tutorial root (relative): [(case relative, solver)]}"""
    units = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if os.path.basename(dirpath) != 'system' or 'controlDict' not in filenames:
            continue
        app = application(os.path.join(dirpath, 'controlDict'))
        if app not in solvers:
            continue
        case = os.path.relpath(os.path.dirname(dirpath), root)
        unit, d = case, case
        while d not in ('', '.'):
            if any(os.path.isfile(os.path.join(root, d, s)) for s in SCRIPTS):
                unit = d
                break
            d = os.path.dirname(d)
        if match and not re.search(match, unit):
            continue
        units.setdefault(unit, []).append((case, app))
    return dict(sorted(units.items()))


def ofEnvironment(bashrc):
    if bashrc == 'none':
        return dict(os.environ)
    if not os.path.isfile(bashrc):
        sys.exit('tutorial_sweep: no OpenFOAM bashrc at %s (--ofbashrc, OF_BASHRC, OFBASHRC)' % bashrc)
    # `set --`: the bashrc evaluates the arguments it is sourced with (etc/config.sh/setup)
    out = subprocess.run(['bash', '-c', 'set +u; set --; . "$0" > /dev/null 2>&1; env -0', bashrc],
                         stdout=subprocess.PIPE, check=True).stdout
    env = {}
    for item in out.split(b'\0'):
        k, sep, v = item.decode(errors='replace').partition('=')
        if sep:
            env[k] = v
    return env


def solverNames(env):
    names = set()
    src = env.get('FOAM_SOLVERS')
    if src and os.path.isdir(src):
        for dirpath, _, filenames in os.walk(src):
            if 'files' in filenames and os.path.basename(dirpath) == 'Make':
                for m in re.finditer(r'EXE\s*=\s*\$\(FOAM_APPBIN\)/(\w+)', open(os.path.join(dirpath, 'files')).read()):
                    names.add(m.group(1))
    # ...and every *Foam executable (not a *ToFoam mesh converter, not one of brae's own) in $FOAM_APPBIN and on PATH:
    # an OpenFOAM run inside a container reaches its solvers through wrappers on PATH named after them
    for d in [env.get('FOAM_APPBIN', '')] + env.get('PATH', '').split(os.pathsep):
        if d and os.path.isdir(d):
            for n in os.listdir(d):
                if n.endswith('Foam') and not n.endswith('ToFoam') and not n.startswith('brae') \
                        and os.access(os.path.join(d, n), os.X_OK):
                    names.add(n)
    return names


def parseCheck(log, rc):
    """(verdict, refusal, memory, total, notes) from one `brae check --setup` log"""
    text = open(log, errors='replace').read() if os.path.isfile(log) else ''
    lines = text.splitlines()
    verdict = re.search(r'(?m)^verdict: (REFUSED|SUPPORTED WITH NOTICES|SUPPORTED)\b', text)
    setup = re.search(r'brae check --setup: \S+ set the case up completely.*?GPU memory in use ([0-9.]+) of ([0-9.]+) GiB',
                      text)
    notrun = re.search(r'(?m)^--setup: not run: (.*)$', text)
    refused = ['%s: %s' % (m.group(1), m.group(2).strip())
               for m in re.finditer(r'(?m)^\s*\[refused\s*\]\s*(\S+)\s+(.*)$', text)]
    notices = ['%s: %s' % (m.group(1), m.group(2).strip())
               for m in re.finditer(r'(?m)^\s*\[notice\s*\]\s*(\S+)\s+(.*)$', text)]
    # a refusal at set-up is the solver's last `brae: ...` line (the throw); the earlier ones are notices
    errors = [l.strip() for l in lines if re.search(r'\bbrae: ', l)][-1:]
    notes = '; '.join(notices)
    if setup:
        return 'set up', '', float(setup.group(1)), float(setup.group(2)), notes
    if verdict and verdict.group(1) == 'REFUSED':
        return 'refused', ' | '.join(refused) or verdict.group(0), None, None, notes
    if verdict and notrun and rc in (0, 1):
        return 'checked', '', None, None, '; '.join(x for x in (notes, 'setup not run: ' + notrun.group(1)) if x)
    if errors:
        return 'refused', ' | '.join(errors), None, None, notes
    tail = ' | '.join(l.strip() for l in lines[-5:] if l.strip())
    return 'error', 'exit %s: %s' % (rc, tail), None, None, notes


def readf(path, default=''):
    try:
        return open(path, errors='replace').read().strip()
    except OSError:
        return default


def hasMesh(case):
    return any(os.path.isfile(os.path.join(case, 'constant', 'polyMesh', f)) for f in ('owner', 'owner.gz'))


def runScript(cmd, cwd, env, log, deadline):
    """run one script in its own process group; (exit code, timed out)"""
    with open(log, 'ab') as out:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        try:
            return p.wait(timeout=max(1.0, deadline - time.time())), False
        except subprocess.TimeoutExpired:
            for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                try:
                    os.killpg(p.pid, sig)
                except OSError:
                    pass
                try:
                    p.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    pass
            return None, True


def sweepOne(unit, cases, a, env, stubs):
    t0 = time.time()
    src = os.path.join(a.tutorials, unit)
    dst = os.path.join(a.work, unit)
    logs = os.path.join(a.work, '_logs', unit)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.rmtree(logs, ignore_errors=True)
    os.makedirs(logs)
    base = {'tutorial': unit, 'script': 'none', 'stubbed': '', 'parallel': '', 'notes': ''}

    def scriptText():
        return '\n'.join(readf(os.path.join(src, s)) for s in SCRIPTS + MESHERS)
    if a.skip_mesh_if and re.search(a.skip_mesh_if, unit + '\n' + scriptText()):
        return [dict(base, solver=s, case=c, mesh='skipped', verdict='skipped',
                     refusal='--skip-mesh-if %s matched' % a.skip_mesh_if, wall_s=0.0) for c, s in cases]

    shutil.copytree(src, dst, symlinks=True)
    sweep = os.path.join(dst, '.tutorial_sweep')
    rec = os.path.join(sweep, 'checks')
    shimdir = os.path.join(sweep, 'bin')
    os.makedirs(rec)
    os.makedirs(shimdir)
    for n in stubs | set(a.solvers):
        body = SHIM if n in a.solvers else STUB
        body = body.replace('@REC@', rec).replace('@BRAE@', a.brae).replace('@LOCK@', a.lock or '')
        path = os.path.join(shimdir, n)
        with open(path, 'w') as f:
            f.write(body)
        os.chmod(path, 0o755)
    env = dict(env, PATH=shimdir + os.pathsep + env.get('PATH', ''))
    deadline = t0 + a.timeout
    scriptLog = os.path.join(logs, 'script.log')

    have = lambda d, names: next((s for s in names if os.path.isfile(os.path.join(dst, d, s))), None)

    def script(d, name):   # a tutorial script by its own #! line (Allrun's `cd "${0%/*}"` needs the full path)
        path = os.path.join(dst, d, name)
        return [path] if os.access(path, os.X_OK) else ['sh', path]
    allrun, mesher = have('', SCRIPTS), have('', MESHERS)
    timedOut, rc = False, None
    if allrun and not (a.script == 'pre' and mesher):
        base['script'] = allrun
        rc, timedOut = runScript(script('', allrun), dst, env, scriptLog, deadline)
    else:
        if mesher:
            base['script'] = mesher
            rc, timedOut = runScript(script('', mesher), dst, env, scriptLog, deadline)
        for case, solver in cases:
            if timedOut:
                break
            crel = os.path.relpath(case, unit)
            cdir = os.path.join(dst, crel)
            own = None if mesher else have(crel, MESHERS)
            if own:
                base['script'] = own
                rc, timedOut = runScript(script(crel, own), cdir, env, scriptLog, deadline)
            elif not mesher and os.path.isfile(os.path.join(cdir, 'system', 'blockMeshDict')):
                base['script'] = 'blockMesh'   # OpenFOAM's default for a case without scripts (foamRunTutorials)
                rc, timedOut = runScript(['blockMesh'], cdir, env, os.path.join(cdir, 'log.blockMesh'), deadline)
            if not timedOut:
                _, timedOut = runScript([os.path.join(shimdir, solver), '-case', cdir], cdir, env, scriptLog, deadline)
    if rc:
        base['notes'] = '%s exited %s' % (base['script'], rc)
    wall = round(time.time() - t0, 1)

    stubbed = ' '.join(sorted(set(readf(os.path.join(sweep, 'stubbed')).split())))
    rows = []
    for n in sorted(os.listdir(rec), key=int):
        d = os.path.join(rec, n)
        rcCheck = readf(os.path.join(d, 'rc'), None)
        rcCheck = int(rcCheck) if rcCheck not in (None, '') else None
        if rcCheck is None:
            verdict, refusal, mem, tot, notes = 'timeout' if timedOut else 'error', 'the check did not finish', None, None, ''
        else:
            verdict, refusal, mem, tot, notes = parseCheck(os.path.join(d, 'check.log'), rcCheck)
        times = readf(os.path.join(d, 'time')).split()
        case = os.path.normpath(os.path.join(unit, os.path.relpath(readf(os.path.join(d, 'case'), dst), dst)))
        extra = readf(os.path.join(d, 'notes'))
        said = readf(os.path.join(d, 'check.log')).splitlines()
        setupNotices = sorted(set(l.strip() for l in said if re.match(r'\s*brae( NOTICE \[(?!unread)|: skipping)', l)))
        rows.append(dict(base, solver=readf(os.path.join(d, 'solver')), case=case, mesh=readf(os.path.join(d, 'mesh')),
                         setup_notices=setupNotices,
                         verdict=verdict, refusal=refusal, memory_gib=mem, memory_total_gib=tot,
                         check_s=round(float(times[1]) - float(times[0]), 1) if len(times) == 2 else None,
                         wall_s=wall, parallel=readf(os.path.join(d, 'parallel')) == '1', stubbed=stubbed,
                         notes='; '.join(x for x in (base['notes'], notes, extra) if x)))
        if os.path.isfile(os.path.join(d, 'check.log')):
            shutil.copy(os.path.join(d, 'check.log'), os.path.join(logs, 'check.%s.log' % n))
    if not rows:
        for case, solver in cases:
            cdir = os.path.join(dst, os.path.relpath(case, unit))
            rows.append(dict(base, solver=solver, case=case,
                             mesh='ok' if hasMesh(cdir) or hasMesh(os.path.join(cdir, 'processor0')) else
                             'failed' if rc else 'absent',
                             verdict='timeout' if timedOut else 'not reached',
                             refusal='' if timedOut else 'the scripts ended (exit %s) without running %s' % (rc, solver),
                             wall_s=wall, stubbed=stubbed))
    elif timedOut:
        for r in rows:
            r['notes'] = '; '.join(x for x in (r['notes'], 'the tutorial was stopped at --timeout after this check') if x)
    for dirpath, _, filenames in os.walk(dst):
        if '.tutorial_sweep' in dirpath.split(os.sep):
            continue
        for f in filenames:
            if f.startswith('log.'):
                rel = os.path.relpath(os.path.join(dirpath, f), dst).replace(os.sep, '__')
                shutil.copy(os.path.join(dirpath, f), os.path.join(logs, rel))
    if not a.keep:
        shutil.rmtree(dst, ignore_errors=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solvers', required=True, help='comma-separated application names')
    ap.add_argument('--tutorials', default=os.environ.get('FOAM_TUTORIALS') or '/usr/lib/openfoam/openfoam2412/tutorials')
    ap.add_argument('--work', required=True)
    ap.add_argument('--brae', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   'build', 'brae'))
    ap.add_argument('--timeout', type=float, default=1800.0, help='seconds per tutorial')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--skip-mesh-if', dest='skip_mesh_if')
    ap.add_argument('--match')
    ap.add_argument('--script', choices=('auto', 'pre'), default='auto')
    ap.add_argument('--ofbashrc', default=os.environ.get('OF_BASHRC') or os.environ.get('OFBASHRC') or OF_DEFAULT)
    ap.add_argument('--lock')
    ap.add_argument('--out')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    a.solvers = [s for s in a.solvers.split(',') if s]
    a.tutorials = os.path.abspath(a.tutorials)
    a.work = os.path.abspath(a.work)
    a.brae = os.path.abspath(a.brae)
    a.lock = os.path.abspath(a.lock) if a.lock else None
    out = a.out or os.path.join(a.work, 'tutorial_sweep')
    if not os.access(a.brae, os.X_OK):
        sys.exit('tutorial_sweep: no brae binary at %s (--brae)' % a.brae)
    if not os.path.isdir(a.tutorials):
        sys.exit('tutorial_sweep: no tutorials at %s (--tutorials)' % a.tutorials)
    env = ofEnvironment(a.ofbashrc)
    stubs = solverNames(env) - set(a.solvers)
    if not stubs:
        sys.exit('tutorial_sweep: no OpenFOAM solver found ($FOAM_SOLVERS, $FOAM_APPBIN, *Foam on PATH) to stand in '
                 'for, so a tutorial script could run one: stopping')
    units = discover(a.tutorials, set(a.solvers), a.match)
    os.makedirs(a.work, exist_ok=True)
    print('tutorial_sweep: %d tutorials of %s under %s' % (len(units), ','.join(a.solvers), a.tutorials), flush=True)

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = {pool.submit(sweepOne, u, c, a, env, stubs): u for u, c in units.items()}
        for f in concurrent.futures.as_completed(futures):
            u = futures[f]
            try:
                got = f.result()
            except Exception as e:   # one tutorial's failure is its row, not the sweep's end
                got = [dict(dict.fromkeys(FIELDS), tutorial=u, solver=s, case=c, verdict='error',
                            refusal='sweep: %r' % e) for c, s in units[u]]
            for r in got:
                print('  %-12s %-60s %-11s %s' % (r.get('solver'), r.get('tutorial') + ('' if r.get('case') == u else
                      ' [' + str(r.get('case')) + ']'), r.get('verdict'), (r.get('refusal') or '')[:120]), flush=True)
            rows.extend(got)
    rows.sort(key=lambda r: (r.get('solver') or '', r.get('tutorial') or '', r.get('case') or ''))
    with open(out + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in FIELDS})
    summary = {}
    for r in rows:
        summary.setdefault(r.get('solver'), {}).setdefault(r.get('verdict'), 0)
        summary[r.get('solver')][r.get('verdict')] += 1
    with open(out + '.json', 'w') as f:
        json.dump({'tutorials': a.tutorials, 'solvers': a.solvers, 'brae': a.brae, 'script': a.script,
                   'skip_mesh_if': a.skip_mesh_if, 'summary': summary, 'rows': rows}, f, indent=1)
        f.write('\n')
    print('tutorial_sweep: %d rows -> %s.csv, %s.json; %s' % (len(rows), out, out, json.dumps(summary)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
