#!/usr/bin/env bash
# The host smoother's scratch is scratch: its memory must not grow with the iteration count.
#
# Item 68 gave the smoother a cache keyed on the field it solves, copying the graph paths, which key on
# psi because a captured GRAPH bakes pointers. The host smoother bakes nothing -- everything it holds is
# working storage for one call -- and the legacy driver hands it a psi whose address MOVES between
# iterations, so every iteration added an entry holding about 27 MB of PINNED host memory that is never
# freed. Measured on the composed 209,825-cell flat plate: 29 MB per iteration, 6.16 GB by iteration 200,
# 11.88 GB by 400, and the turbulentFlatPlate tutorial gate's run reached 54 GB before it was stopped --
# on a 121 GB machine, with page-locked memory, which comes out of the machine rather than the process.
# It is now one leaked scratch, resized when the mesh size changes.
#
# THE GATE IS THE SHAPE, not a number: memory must be FLAT in the iteration count. A leak of any size
# shows as a slope, and a bound in gigabytes would pass a smaller leak silently.
#
# READ INSIDE ONE RUN, AT FIXED ITERATIONS (BRAE_MEMORY_PROBE, solvers/common/memory_probe.cuh). Until
# 2026-11-03 this compared the kernel's peak resident set at EXIT (/usr/bin/time %M) of a 100- and a
# 400-iteration run. On the JUPITER login node (GH200, 72 CPUs, 64 KiB pages) that figure was 189 to 243 MB
# between identical runs of one binary, every value a multiple of 9 MiB (144 pages, 2 x 72 CPUs: a per-CPU
# counter's batch), while VmHWM read from /proc inside the loop was flat to 0.1 MB from iteration ~64 to 361.
# Arrhenius read 0.67x, 1.50x and 3.00x on the same binaries. That was the metric, not a leak.
#
# One run of N=400 iterations an arm, probed after the set-up, at iterations 64, 212 and 361, after the
# loop and after the final write. The steady points are 64, 212 and 361:
#   ARM      between each pair of steady points, the process's resident memory (smaps_rollup Rss, a walk of
#            the page tables) and the device pool's reserved and allocated bytes grow by at most 1.25x, the
#            bound this gate has always had, now on the same run's own figures. Not VmHWM: a high-water mark
#            set at start-up hides any growth below it.
#   CONTROL  every run reaches its iteration count and prints every probe line.
#   CONTROL2 BRAE_CONTROL_MEMORY_LEAK=host:1 and =device:1 -- 1 MiB kept at every iteration, host or device --
#            must FAIL their own check: the gate can see a leak of that size.
# Reported, not gated: the start-up peak (VmHWM and the pool's peak at start-up), the whole device's use and
# the machine's (other processes are in both), and the shutdown: the highest VmRSS polled every 10 ms after
# the last probe until exit, and the kernel's ru_maxrss at exit (the old figure). Each arm runs REPEATS times.
#
# Open, not attributed: the 0018 candidate's V2 400-iteration runs showed one +5.3 MB VmHWM step in their last
# half second, 3 of 3 (its base 0 of 3), outputs byte-identical (JUPITER login node, 2026-11-03).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/T3A}"
N=${N:-400}
PROBES=${PROBES:-64,212,361}
REPEATS=${REPEATS:-3}
BOUND=${BOUND:-1.25}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
W=$(mktemp -d); [ -n "${KEEP_W:-}" ] && echo "work dir $W" || trap 'rm -rf "$W"' EXIT
stage() {   # stage <dir>
    local d="$1"
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$SRC/constant" "$SRC/system" "$d/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$d/0"; else cp -r "$SRC/0" "$d/0"; fi
    python3 - "$d" "$N" <<'PY'
import re, sys
d, n = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
# residualControl would stop the run before the late probes; every arm must run the full count.
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PY
}
for a in legacy v2 leak_host leak_device; do
    for r in $(seq 1 "$REPEATS"); do
        case $a in leak_*) [ "$r" -gt 1 ] && continue ;; esac
        stage "$W/${a}_$r"
    done
done

python3 - "$BRAE" "$W" "$N" "$PROBES" "$REPEATS" "$BOUND" <<'PY'
import os, re, subprocess, sys, time
brae, W, n, probes = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
repeats, bound = int(sys.argv[5]), float(sys.argv[6])
steady = [int(x) for x in probes.split(',')]
fail = 0
def say(msg, ok):
    global fail
    print('  %-92s %s' % (msg, 'ok' if ok else 'FAIL'))
    if not ok: fail = 1
def status_kb(pid, key):
    try:
        for line in open('/proc/%d/status' % pid):
            if line.startswith(key + ':'): return int(line.split()[1])
    except OSError:
        pass
    return None
def run(case, extra):
    """brae on the case: its probe lines {(at, iteration): {key: value}}, its iteration count, the shutdown
    window's figures, and its log"""
    env = dict(os.environ); env.update(extra); env['BRAE_MEMORY_PROBE'] = probes
    log = os.path.join(case, 'log')
    with open(log, 'w') as out:
        p = subprocess.Popen([brae, case], cwd=case, stdout=out, stderr=subprocess.STDOUT, env=env)
    tail, off, ended, shut_rss, shut_hwm = '', 0, False, 0, 0
    while True:
        pid, _, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            break
        if not ended:
            with open(log, errors='replace') as f:
                f.seek(off); chunk = f.read(); off = f.tell()
            tail = (tail + chunk)[-4096:]
            ended = 'brae memory: at=end ' in tail
        if ended:
            # after the last probe: the shutdown window, until the process is gone
            rss, hwm = status_kb(p.pid, 'VmRSS'), status_kb(p.pid, 'VmHWM')
            if rss: shut_rss = max(shut_rss, rss)
            if hwm: shut_hwm = max(shut_hwm, hwm)
        time.sleep(0.01)
    text = open(log, errors='replace').read()
    lines = {}
    for m in re.finditer(r'^brae memory: at=(\S+) iteration=(\d+) (.*)$', text, flags=re.M):
        lines[(m.group(1), int(m.group(2)))] = {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', m.group(3))}
    iters = len(re.findall(r'^Time = ', text, flags=re.M))
    return lines, iters, {'rss_kb': shut_rss, 'hwm_kb': shut_hwm, 'ru_maxrss_kb': ru.ru_maxrss}, text
def growth(lines, key):
    """the worst ratio over every pair of steady points, later over earlier, and the steady values"""
    vals = [lines[('iteration', it)][key] for it in steady]
    worst = max(vals[j] / vals[i] for i in range(len(vals)) for j in range(i + 1, len(vals)) if vals[i] > 0)
    return worst, vals
MB = 1024.0
def one(label, case, extra, checks):
    lines, iters, shut, text = run(case, extra)
    want = [('start-up', 0)] + [('iteration', it) for it in steady]
    have = all(k in lines for k in want) and any(k[0] == 'loop-end' for k in lines) \
           and any(k[0] == 'end' for k in lines)
    say('%s  reached %d of %d iterations and printed every probe' % (label, iters, n), iters >= n and have)
    if not have:
        return None
    s = lines[('start-up', 0)]
    le = [v for k, v in lines.items() if k[0] == 'loop-end'][0]
    en = [v for k, v in lines.items() if k[0] == 'end'][0]
    res = {}
    for key, name, scale in (('rss_exact_kb', 'process Rss', MB), ('pool_reserved', 'pool reserved', MB * MB),
                             ('pool_allocated', 'pool allocated', MB * MB)):
        r, vals = growth(lines, key)
        res[key] = r
        print('    %-15s at %s: %s MB, worst ratio %.4f'
              % (name, '/'.join(map(str, steady)), ' / '.join('%.2f' % (v / scale) for v in vals), r))
    for name, x in (('start-up', s), ('loop end', le)):
        print('    %-15s VmHWM %.1f MB, pool peak %.2f MB, device used %.1f MB, machine used %.0f MB'
              % (name, x['hwm_kb'] / MB, x['pool_peak_reserved'] / MB / MB, x['device_used'] / MB / MB,
                 x['host_used_kb'] / MB))
    print('    %-15s VmHWM %.1f MB, VmRSS %.1f MB' % ('end', en['hwm_kb'] / MB, en['rss_kb'] / MB))
    print('    %-15s highest VmRSS polled %.1f MB, VmHWM %.1f MB; ru_maxrss at exit %.1f MB'
          % ('shutdown', shut['rss_kb'] / MB, shut['hwm_kb'] / MB, shut['ru_maxrss_kb'] / MB))
    for key, name in checks:
        say('%s  %s grows at most %.2fx between iterations %s' % (label, name, bound, probes), res[key] <= bound)
    return res, text
ALL = (('rss_exact_kb', 'process Rss'), ('pool_reserved', 'pool reserved'), ('pool_allocated', 'pool allocated'))
for arm, tag, extra in (('ARM 1  legacy', 'legacy', {}), ('ARM 2  mirror', 'v2', {'BRAE_SIMPLEFOAM_V2': '1'})):
    for r in range(1, repeats + 1):
        one('%s run %d' % (arm, r), os.path.join(W, '%s_%d' % (tag, r)), extra, ALL)
# The controls run the legacy arm and are judged by the check their leak targets, which must fail.
for tag, spec, key, name in (('leak_host', 'host:1', 'rss_exact_kb', 'process Rss'),
                             ('leak_device', 'device:1', 'pool_allocated', 'pool allocated')):
    got = one('CONTROL2 %s' % spec, os.path.join(W, '%s_1' % tag), {'BRAE_CONTROL_MEMORY_LEAK': spec}, ())
    if got is None:
        continue
    res, text = got
    say('CONTROL2 BRAE_CONTROL_MEMORY_LEAK=%s reached brae' % spec, 'BRAE_CONTROL_MEMORY_LEAK=%s' % spec in text)
    say('CONTROL2 BRAE_CONTROL_MEMORY_LEAK=%s fails the %s check (%.4f > %.2f)' % (spec, name, res[key], bound),
        res[key] > bound)
sys.exit(1 if fail else 0)
PY
rc=$?
[ $rc -eq 0 ] && echo "PASS: the smoother's scratch does not grow with the run"
exit $rc
