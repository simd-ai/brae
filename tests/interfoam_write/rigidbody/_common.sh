# Shared by the files of this folder. Q: a rigid body's uniform/rigidBodyMotionState
# (rigidBodyMeshMotion.C:392-417), entry by entry. The tutorial comparer
# comparer cannot hold this file: it is blind to whitespace, and one relative gap per file lets qDdot
# (~499 at DTCHullMoving's 0.0002) set the scale for q (~1e-6). So each entry is held relative to its own
# size, and the text with every number masked must be OpenFOAM's -- the list form included: `N ( a b )`,
# and `N { v }` for a list of equal entries (UListIO.C:119-123), which a body at rest writes.
#   DTCHullMoving (arm W's run, host): a body that moves, q/qDot/qDdot nonuniform.
#   floatingObject with `ddtSchemes default Euler`: its accelerationRelaxation is 0 until t = 4, so the
#     body stays at rest and writes `2 { 0 }`. The scheme is the staging's, stated (this arm predates the
#     CrankNicolson writer; the tutorial as shipped is arm W's row and arm X); the file under test is the same.
# CONTROLS: BRAE_CONTROL_RBSTATE_PAREN=1 (the uniform lists in the paren form) fails the text on
# floatingObject; BRAE_CONTROL_RBSTATE_OLD=1 (motionState0_, the step's start, which OpenFOAM never writes)
# fails DTCHullMoving's entries.
rbstate()   # rbstate <OpenFOAM case> <brae case> <bound> -- prints the worst entry gap, fails text or bound
{
    python3 - "$@" <<'PY'
import re, sys
of, br, bound = sys.argv[1], sys.argv[2], float(sys.argv[3])
num = re.compile(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?')
def body(p):
    s = open(p).read()
    return s[s.find('// * * *'):]
def entries(s):
    out = {}
    for k, v in re.findall(r'^(\w+)\s+([^;]*);', s, re.M):
        vals = [float(x) for x in num.findall(v)]
        # a list's leading count is its size, not a value; `N { v }` stands for N copies of v
        if '(' in v or '{' in v:
            n, rest = int(vals[0]), vals[1:]
            vals = rest * n if '{' in v else rest
        out[k] = vals
    return out
bad, worst = 0, (0.0, '')
times = sorted([t for t in __import__('os').listdir(of) if re.match(r'^[0-9.e+-]+$', t) and t != '0'], key=float)
for t in times:
    po, pb = '%s/%s/uniform/rigidBodyMotionState' % (of, t), '%s/%s/uniform/rigidBodyMotionState' % (br, t)
    try:
        so, sb = body(po), body(pb)
    except OSError as e:
        print('      %s: %s' % (t, e)); bad += 1; continue
    if num.sub('#', so) != num.sub('#', sb):
        print('      %s: the text with its numbers masked is not OpenFOAM\'s' % t); bad += 1
    eo, eb = entries(so), entries(sb)
    if list(eo) != ['q', 'qDot', 'qDdot', 't', 'deltaT'] or list(eb) != list(eo):
        print('      %s: entries %s, OpenFOAM %s' % (t, list(eb), list(eo))); bad += 1; continue
    for k in eo:
        if len(eo[k]) != len(eb[k]):
            print('      %s/%s: %d values, OpenFOAM %d' % (t, k, len(eb[k]), len(eo[k]))); bad += 1; continue
        scale = max([abs(x) for x in eo[k]] + [0.0])
        gap = max([abs(a - b) for a, b in zip(eo[k], eb[k])] + [0.0])
        rel = gap / scale if scale > 0 else (0.0 if gap == 0 else float('inf'))
        if rel > worst[0]:
            worst = (rel, '%s/%s' % (t, k))
        if rel > bound:
            print('      %s/%s: %.3e over %.0e' % (t, k, rel, bound)); bad += 1
print('      worst entry %s at %.3e (bound %.0e)' % (worst[1] or '-', worst[0], bound))
sys.exit(1 if bad else 0)
PY
}

# q_of_case: RAS/floatingObject under Euler, meshed and run by OpenFOAM (both cached), as $W/q_of
q_of_case()
{
    FO="$TUT/multiphase/interFoam/RAS/floatingObject"
    [ -d "$FO" ] || { say "ARM Q  floatingObject tutorial missing" FAIL; finish "arm Q"; }
    stage_allrun "$FO" "$W/q_of" "" || say "ARM Q  floatingObject: meshing failed (see $W/q_of/log.allrunmesh)" FAIL
    sed -i -E 's/^(\s*default\s+)CrankNicolson[^;]*;/\1Euler;/' "$W/q_of/system/fvSchemes"
    grep -qE '^\s*default\s+Euler;' "$W/q_of/system/fvSchemes" || say "ARM Q  floatingObject: the ddt scheme was not switched to Euler" FAIL
    runof "$W/q_of"
    grep -q "2 { 0 }" "$W/q_of/$(timedirs "$W/q_of" | awk '{print $1}')/uniform/rigidBodyMotionState" \
        && say "fixture witnesses: OpenFOAM's floatingObject body at rest writes \`2 { 0 }\`" ok \
        || say "fixture witnesses: OpenFOAM's floatingObject body at rest writes \`2 { 0 }\`" FAIL
}

# ...and READ back (rigidBodyMeshMotion.C:93-118): a 0/uniform/rigidBodyMotionState both codes start
# from -- a restart from a written time is refused before the reader, the moved points being there.
#   A  `q 2 { 0.01 }`, the form OpenFOAM writes for equal entries. FAIL-PROOF (2026-09-30): the reader
#      before this unit threw `expected '(' got '{'`.
#   B  `q 2 ( 0.01 0.02 )` as rigidBodyMotionState.gz, what `writeCompression on` leaves. FAIL-PROOF: the
#      plain-path lookup before this unit started the body from rest, pointDisplacement 1.0 off.
#   D  `qDdot 2 ( 1e-301 0 )`: readScalar rounds it to 0 (Scalar.C:104-110), so OpenFOAM writes `2 { 0 }`.
#      FAIL-PROOF (2026-09-30): with neither the reader's nor the writer's rounding, brae wrote
#      `qDdot 2 ( 1.0000000000000001e-301 0 )`. Either rounding alone passes it: the writer's re-read
#      hides the reader's, as OpenFOAM's own write would.
#   E  B's q with no FoamFile header: typeHeaderOk fails, OpenFOAM warns and starts from the coeffs --
#      rest (IOobjectReadHeader.C:104-125).
# MEASURED: fields 1.1e-13 (A), 8.6e-14 (B); the body displaced 1.0e-02 and 1.1e-02 at 0.01.
# And two inputs OpenFOAM refuses or brae cannot follow, each asserted on BOTH codes:
#   F  `q 0 ( )` on a 2-DoF chain: OpenFOAM stops (rigidBodyModelState.C:58-69); brae must too, by name.
#   G  `q (0.01 0.02)` in rigidBodyMotionCoeffs and no state file: OpenFOAM starts the body there
#      (rigidBodyMeshMotion.C:117); brae does not port it and must refuse by name.
rsfile()   # rsfile <path> <q> <qDdot> <header yes|no>
{
    python3 - "$@" <<'EOF_RS'
import sys
path, q, qDdot, hdr = sys.argv[1:5]
head = ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n'
        '    location    "0/uniform";\n    object      rigidBodyMotionState;\n}\n\n') if hdr == 'yes' else ''
open(path, 'w').write(head + 'q               %s;\n\nqDot            2 { 0 };\n\nqDdot           %s;\n\n'
                      't               0;\n\ndeltaT          0.01;\n' % (q, qDdot))
EOF_RS
}
rscase()   # rscase <dir> -- floatingObject (Euler) from 0/, no state file yet
{
    mkdir -p "$1/0/uniform"
    cp -r "$W/q_of/0/." "$1/0/"
    cp -r "$W/q_of/constant" "$W/q_of/system" "$1/"
}
