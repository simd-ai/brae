#!/bin/bash
# scalarTransport gate. Until this existed, NOTHING in the 184-test suite carried a tracer, so every
# scalarTransport defect found so far was found by hand: the field solved but was never written; the
# convection scheme was hardcoded instead of read from div(phi,<field>); a transient tracer bounded by
# 1.0 reached 6.32 because the functionObject fired before the solver knew deltaT. All three were green
# on the full suite the entire time.
#
# WHAT IT ASSERTS
#   1. the tracer is WRITTEN               (it solved on the device and never reached disk once)
#   2. it is BOUNDED by its inlet value    (the 6.32 defect)
#   3. it is NON-UNIFORM                   (anti-vacuous: a tracer stuck at its initial value, or one
#                                           smeared to the inlet value everywhere, would pass 1+2)
#   4. an absent div(phi,tracer0) under `default none` is REFUSED, not substituted (OF refuses the case)
#
# 4 is the negative control: it proves the gate can go red. Without it a build that silently dropped
# scalarTransport entirely would still pass 1-3 by writing the initial field.
set -u
BRAE=${1:?usage: tracer_transport.sh <brae_rhoSimpleFoam> <caseDir> <workDir>}
SRC=${2:?}
WORK=${3:?}

[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: $SRC absent"; exit 77; }

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/" 2>/dev/null

( cd "$WORK" && "$BRAE" -case . > log.brae 2>&1 )
rc=$?
if [ $rc -ne 0 ]; then echo "FAIL: solver exited $rc"; tail -5 "$WORK/log.brae"; exit 1; fi

LAST=$(ls -d "$WORK"/[0-9]* 2>/dev/null | grep -v '/0$' | sort -t/ -k2 -g | tail -1)
[ -n "$LAST" ] || { echo "FAIL: no time directory written"; exit 1; }

python3 - "$LAST" <<'PY' || exit 1
import re, sys, os
d = sys.argv[1]
p = os.path.join(d, 'tracer0')
if not os.path.isfile(p):
    print(f"FAIL: {p} not written -- the tracer solved but never reached disk")
    raise SystemExit(1)
s = open(p, errors='replace').read()
m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', s)
if not m:
    u = re.search(r'internalField\s+uniform\s+([-\d.eE+]+)', s)
    print(f"FAIL: tracer0 is UNIFORM ({u.group(1) if u else '?'}) -- it was written but never transported")
    raise SystemExit(1)
st = m.end(); en = s.index('\n)', st)
v = [float(x) for x in s[st:en].split()]
lo, hi = min(v), max(v)
print(f"  tracer0: n={len(v)} min={lo:.6e} max={hi:.6f}")
bad = 0
if hi > 1.0 + 1e-3:
    print(f"  FAIL unbounded: max {hi:.6f} exceeds the inlet value 1.0 -- the tracer's own\n"
          f"       div(phi,tracer0) scheme is not being honoured, or fvm::ddt(s) is missing")
    bad += 1
# Symmetric with the upper bound. OF's scalarTransport does not bound either (no Foam::bound call), so
# undershoot at the linear-solver tolerance is expected and not a defect; what matters is that the field
# does not leave [0,1] by a PHYSICALLY meaningful amount. The defect this guards against reached 6.32.
if lo < -1e-3:
    print(f"  FAIL negative: min {lo:.6e} is below 0 by more than round-off")
    bad += 1
if hi - lo < 1e-6:
    print(f"  FAIL vacuous: the field is effectively constant ({lo:.6e}..{hi:.6e}), so this case cannot\n"
          f"       distinguish a working transport from one that never ran")
    bad += 1
raise SystemExit(1 if bad else 0)
PY

# 4. NEGATIVE CONTROL: strip the scheme and require a refusal.
# Built from $SRC, not from $WORK: `rm -rf $WORK/[0-9]*` also matches 0.orig (it starts with a digit),
# which silently emptied the case and made the control pass for the wrong reason.
W2="$WORK.noscheme"
rm -rf "$W2"; mkdir -p "$W2"; cp -r "$SRC"/* "$W2/"
mkdir -p "$W2/0" && cp "$W2"/0.orig/* "$W2/0/" 2>/dev/null
sed -i 's/^\s*div(phi,tracer0).*$//' "$W2/system/fvSchemes"
( cd "$W2" && "$BRAE" -case . > log.brae 2>&1 )
if grep -q "cannot open" "$W2/log.brae"; then
    echo "FAIL negative control: the control case did not load -- it proves nothing about the refusal."
    tail -3 "$W2/log.brae"; exit 1
fi
if ! grep -q "default none" "$W2/log.brae"; then
    echo "FAIL negative control: div(phi,tracer0) removed under \`default none\` and brae did NOT refuse."
    echo "      OpenFOAM rejects that case; running it with a substituted scheme is a silent wrong answer."
    exit 1
fi
LAST2=$(ls -d "$W2"/[0-9]* 2>/dev/null | grep -v '/0$' | sort -t/ -k2 -g | tail -1)
if [ -n "$LAST2" ] && [ -f "$LAST2/tracer0" ]; then
    if grep -q nonuniform "$LAST2/tracer0"; then
        echo "FAIL negative control: the tracer was transported despite having no convection scheme"
        exit 1
    fi
fi
echo "  negative control: no div(phi,tracer0) under \`default none\` -> refused, not substituted"
echo "PASS tracer_transport"
