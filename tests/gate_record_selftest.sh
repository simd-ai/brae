#!/usr/bin/env bash
# tools/gate_record.py against gates whose every output line and exit code is known: a fake gate prints each verdict
# line form the suite's gates print (tests/rho_central_gate_lib.sh's ok/FAIL/hist/[open item], the simpler gates' say()
# and their `ok:`/`FAIL:` lines, a ctest `NNN: ` prefix, a SKIP) and exits with the code it is asked for.
#   1. the output is streamed byte for byte (a partial last line and a non-UTF-8 byte included), and output.log holds it
#   2. the exit code is the gate's: 0, 3 and 77, and 128 + 15 when SIGTERM stops it
#   3. result.json: the counts and the FAIL/open/SKIP lines; only BRAE_* and OF* variables; the binary's sha256; the
#      source tree's commit and dirty flag as git says; WM_PROJECT_VERSION from the bashrc it is given; the labels
#   4. CONTROL: a line that only mentions ok or FAIL in passing is not counted
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOL="${1:-$ROOT/tools/gate_record.py}"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-90s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1; return 0; }

cat > "$W/gate.sh" <<'EOF'
#!/usr/bin/env bash
printf 'ok   premise: OpenFOAM ran\n'
printf 'FAIL U at t = 0.007: 2.1e-13 > 8e-14\n'
printf '     [open item, round-off amplification: fine17 star region]\n'
printf 'hist raw conservation 3e-12\n'
printf '  ARM A  brae writes exactly the directories OpenFOAM writes                 ok\n'
printf '  ARM C  purgeWrite 1 keeps only 14/, as OpenFOAM does                       FAIL\n'
printf '  ok:   U agrees with OpenFOAM (6.900e-04 <= 1.0e-03)\n'
printf '  FAIL: p is 6.000e-03 from OpenFOAM, above the bound 5.0e-03\n'
printf '242: ok   with a ctest prefix\n'
printf 'the solver said ok, nothing FAILed: a mention, not a verdict\n'
printf 'SKIP: nothing to skip, a line to count\n'
printf '\xff raw byte\n'
printf 'no newline at the end'
exit "${GATE_RC:-0}"
EOF
chmod +x "$W/gate.sh"
cat > "$W/bashrc" <<'EOF'
export WM_PROJECT_VERSION=v9999
EOF
printf 'not a real binary\n' > "$W/brae"
chmod +x "$W/brae"

for rc in 0 3 77
do
    ( cd "$W" && env -i PATH="$PATH" HOME="$HOME" GATE_RC=$rc BRAE_TEST_FLAG=1 OF_BASHRC="$W/bashrc" UNRELATED=1 \
        python3 "$TOOL" --out "$W/r$rc" --binary "$W/brae" --source "$ROOT" --site here --case fake --level L0 \
        -- "$W/gate.sh" > "$W/stdout$rc" )
    got=$?
    [ "$got" = "$rc" ] && say "exit code $rc is the gate's" ok || say "exit code $rc is the gate's (got $got)" FAIL
done
GATE_RC=0 "$W/gate.sh" > "$W/expected"
cmp -s "$W/expected" "$W/stdout0" && say "the output is streamed byte for byte" ok || say "the output is streamed byte for byte" FAIL
cmp -s "$W/expected" "$W/r0/output.log" && say "output.log holds the same bytes" ok || say "output.log holds the same bytes" FAIL

python3 - "$W" "$ROOT" <<'PY' || fail=1
import hashlib, json, subprocess, sys
W, ROOT = sys.argv[1], sys.argv[2]
r = json.load(open(W + '/r3/result.json'))
bad = 0
def check(ok, what):
    global bad
    print('  %-90s %s' % (what, 'ok' if ok else 'FAIL'))
    bad += 0 if ok else 1
check(r['schema'] == 'brae-gate-record/1', 'schema brae-gate-record/1')
check(r['exit_code'] == 3 and r['outcome'] == 'fail', 'exit_code 3, outcome fail')
check(json.load(open(W + '/r77/result.json'))['outcome'] == 'skip', 'exit 77 is a skip')
check(r['lines'] == {'ok': 4, 'FAIL': 3, 'hist': 1, 'open': 1, 'skip': 1}, 'line counts %s' % r['lines'])
check(r['fail_lines'] == ['FAIL U at t = 0.007: 2.1e-13 > 8e-14',
                          '  ARM C  purgeWrite 1 keeps only 14/, as OpenFOAM does                       FAIL',
                          '  FAIL: p is 6.000e-03 from OpenFOAM, above the bound 5.0e-03'], 'the FAIL lines verbatim')
check(r['open_lines'] == ['     [open item, round-off amplification: fine17 star region]'], 'the open line verbatim')
check(sorted(r['env']) == ['BRAE_TEST_FLAG', 'OF_BASHRC'], 'env holds BRAE_* and OF* only: %s' % sorted(r['env']))
want = hashlib.sha256(open(W + '/brae', 'rb').read()).hexdigest()
check(len(r['binaries']) == 1 and r['binaries'][0]['sha256'] == want and r['binaries_from'] == '--binary', 'the binary sha256')
head = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, text=True).stdout.strip()
dirty = bool(subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--untracked-files=no'],
                            stdout=subprocess.PIPE, text=True).stdout.strip())
check(r['brae']['commit'] == head and r['brae']['dirty'] == dirty and r['brae']['describe'],
      'brae commit %s dirty %s describe %s' % (head[:8], dirty, r['brae']['describe']))
check(r['openfoam']['version'] == 'v9999', 'OpenFOAM version from the bashrc: %s' % r['openfoam']['version'])
check(r['labels'] == {'site': 'here', 'case': 'fake', 'level': 'L0'}, 'labels')
check(r['command'] == [W + '/gate.sh'] and r['wall_s'] >= 0 and r['start'] and r['end'] and r['hostname'], 'command, times, host')
check('gpu' in r and (r['gpu'] is None or r['gpu']['name']), 'gpu: %s' % r['gpu'])
sys.exit(1 if bad else 0)
PY

# SIGTERM to the recorder reaches the gate, and the code says so
cat > "$W/slow.sh" <<'EOF'
#!/usr/bin/env bash
echo started
exec sleep 30
EOF
chmod +x "$W/slow.sh"
python3 "$TOOL" --out "$W/rs" --binary "$W/brae" --ofbashrc /nonexistent -- "$W/slow.sh" > "$W/stdout_s" &
pid=$!
for _ in $(seq 50); do grep -q started "$W/stdout_s" 2> /dev/null && break; sleep 0.1; done
kill -TERM $pid
wait $pid
got=$?
[ "$got" = 143 ] && grep -q started "$W/rs/output.log" 2> /dev/null && say "SIGTERM is passed on and stops the gate: exit 128 + 15" ok \
    || say "SIGTERM is passed on and stops the gate: exit 128 + 15 (got $got)" FAIL

[ $fail = 0 ] && echo "PASS: gate_record.py streams, keeps the exit code and records what ran" || echo "FAIL"
exit $fail
