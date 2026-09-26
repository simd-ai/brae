#!/usr/bin/env bash
# brae's interFoam under the CRANKNICOLSON ddt scheme against REAL OpenFOAM's, on RAS/damBreak with
# `ddtSchemes { default CrankNicolson 0.5; }` in the shipped file's place, field by field and solve by
# solve, on BOTH arms.
#
# THE METHOD is the other interFoam gates': exactly N identical FIXED steps at the tutorial's deltaT
# (1e-3), both codes read at one instant, on the tutorial's own blockMesh (2268 cells), kEpsilon in the
# `density variable` lineage with MULESCorr and two alpha correctors.
#
# WHY THIS CASE AND NOT floatingObject. RAS/floatingObject is the one shipped interFoam tutorial that
# names CrankNicolson, and it also moves its mesh under rigidBodyMotion -- a body integrated by a
# Newmark solver in the rigidBodyDynamics library, which brae does not carry. The scheme is what this
# gate holds; the body is refused by name (tests/interfoam_refusals.sh, mesh_rigidBody).
#
# WHAT THE SCHEME IS (CrankNicolsonDdtScheme.C, crank_nicolson_ddt_scheme_cpp.cuh): every fvm::ddt keeps
# the previous step's ddt as a field of its own -- "ddt0(rho,U)", "ddt0(rho,k)", "ddt0(rho,epsilon)",
# and ddtCorr's "ddtCorrDdt0(U)" and "ddtCorrDdt0(phi)" -- created zero at its first assembly and
# advanced once per time step; the diagonal takes (1 + oc)/deltaT once the field is a step old, the
# ddt0 estimate is Euler's for one step and (1 + oc)/deltaT0 after, and the source adds oc*ddt0. So a
# cold start is Euler for one step, half-warmed for the second and CrankNicolson from the third; the
# gate's N is long enough to see all three. alphaEqn.H does its own off-centring: phiCN =
# cnCoeff*phi + (1 - cnCoeff)*phi.oldTime() feeds MULES, and -- because ddt(rho,U) is not Euler -- the
# end-of-step alpha flux is un-blended against alphaPhi10.oldTime() and rhoPhi takes phi beside rho2
# (alphaEqn.H:236-262). alphaPhi10.oldTime() is CREATED by that first call, as a copy of the flux as
# it stands, so on the first blended step the un-blend divides a flux by itself; from the next it is
# the previous step's.
#
# FOUR PROFILES, the first three under CrankNicolson and the fourth the control:
#   cn        `CrankNicolson 0.5`, the tutorial's PIMPLE (nOuterCorrectors 1, nCorrectors 3)
#   cnOuter   ...with nOuterCorrectors 2: every ddt0 is asked for twice per step and must advance once,
#             and alphaPhi10.oldTime() must hold within the step
#   cnFull    `CrankNicolson 1`: offCentre_ hands ddt0 back untouched (its `ocCoeff < 1` branch is not
#             taken) and cnCoeff is 1/2
#   euler     the tutorial as shipped -- THE CONTROL: OpenFOAM's own answer under Euler against its
#             answer under CrankNicolson is what the scheme is worth here, and brae must sit far inside it
#
# MEASURED, host and device: see the .cu's bounds, each with its number. Both arms sit at the round-off
# floor on every profile (U 1.5e-14 host, 1.9e-12 device on `cn`), against a control of 3.2e-02.
#
# ...AND TWO MORE, one per closure, because the closure's fvm::ddt comes through ddtSchemes like every
# other term and kEpsilon was the only one that took CrankNicolson:
#   cnSST     the same tutorial with kOmegaSST in kEpsilon's place (and the UNIFORM lineage, which is
#             what the reader admits under that model), CrankNicolson 0.5.  Control: eulerSST.
#   cnLES     ...and with LES kEqn, which solves k alone.                   Control: eulerLES.
# MEASURED, 20 steps: cnSST host k 9.9e-15 / omega 4.1e-15 / nut 3.9e-13, device 5.6e-14 / 4.1e-15 /
# 8.5e-13; cnLES host k 4.4e-15 / nut 9.5e-15, device 3.2e-15 / 8.5e-15. Control 4.8e-02 in U.
# BROKEN ONCE, on the device arm of both (the scheme's term dropped, so the closure forms Euler):
# cnSST k 2.0233e-02, omega 2.7456e-03, nut 1.7637e-03, U 1.4934e-04, 12 failures; cnLES k 9.3629e-02,
# nut 4.5767e-02, U 1.5009e-03, 10 failures.
# AND A VACUITY THIS GATE FOUND IN ITSELF: the closure's second field is ::epsilon under kEpsilon and
# ::omega under kOmegaSST, and the SST arm first read the empty one -- it printed `epsilon 0.0000e+00`
# and passed with nothing on either side. The arm now asserts brae HAS the field before comparing it.
#
# WHAT THE GATE FOUND. (1) phi's old-old level is CREATED on the second step, as a copy of
# phi.oldTime(): fvcDdtPhiCorr asks for phi.oldTime().oldTime() only inside its evaluate branch, and
# GeometricField::oldTime() creates a level that does not exist from the one it hangs off -- so the
# first dphidt0 is rDtCoef0*(phi^1 - phi^1) = 0, and the level rotates only from the third step. U's
# and rho's old-old levels are asked for on the FIRST step, unconditionally, at the top of fvmDdt, so
# they exist as copies of U^0 and rho^0 and rotate from the second. With phi^0 in dphidt0's place the
# host read U 2.3e-01 from OpenFOAM after two steps, alpha 5e-15 (the alpha step reads no old-old
# level). (2) The DEVICE alpha step reset alpha1 to its old time at the start of every outer pass,
# which alphaEqn.H never does and the host reference had stopped doing: cnOuter's second pass sat
# 1.2e-06 out in its first p_rgh residual, k 6.7e-06 and U 2.0e-06 after two steps -- and the same
# numbers under Euler, so it is the RAS gate's new `outer` profile that holds the fix, and this gate
# that found it. (3) The device's pre-solve boundary coefficients were built from phi where OpenFOAM
# builds them from phiCN (alphaEqn.H:114): fixed, NOT DISCRIMINATED here -- damBreak's one
# flux-conditional alpha patch is the atmosphere, whose cells hold exactly 0.
#
# BROKEN ONCE EACH (U, profiles cn / cnOuter / cnFull), the gate red every time:
#   host   coef held at 1 (Euler diagonal under the name)     2.0e+00 / 2.4e+00 / 5.5e+01
#          coef0 = 1 + oc from the first ddt0 (no Euler start) 1.2e-02 / 1.2e-02 / 5.2e-02
#          ddt0 advanced at every assembly                    1.3e-03 / 4.3e-02 / 2.7e-02
#          the off-centre term dropped from the source         4.2e-01 / 4.6e-01 / 5.8e-01
#          ddtCorr run as Euler                                8.9e-02 / 9.9e-02 / 1.0e-01
#          the closure's ddt run as Euler                      1.5e-02 / 1.5e-03 / 4.1e-02 (k 8e-02 / 9e-02 / 5e-01)
#          rhoPhi with phiCN beside rho2 (the Euler branch)     nothing / 2.4e-03 / nothing
#          no un-blend of the end-of-step alpha flux           3.6e-02 / 5.4e-02 / 8.4e-02
#          phi's old-old level rotated from step one           5.3e-02 / 6.9e-02 / 4.7e-01
#          alphaPhi10.oldTime() created as the previous step   5.6e-03 / 2.5e-03 / 5.2e-02
#   device coef held at 1                                      2.0e+00 / 2.4e+00 / 5.5e+01
#          the off-centre term dropped                         4.2e-01 / 4.6e-01 / 5.8e-01
#          ddtCorr run as Euler                                8.9e-02 / 9.9e-02 / 1.0e-01
#          the closure's ddt run as Euler                      1.5e-02 / 1.5e-03 / 4.1e-02
#          no un-blend                                         3.6e-02 / 5.4e-02 / 8.4e-02
#          phi's old-old level rotated from step one           5.8e-03 / 2.9e-03 / 5.4e-02
# The device lines read the host's to two digits where the same thing is broken: a transcription.
# `rhoPhi with phiCN` is blind on cn and cnFull because with ONE outer corrector phi.oldTime() at the
# alpha step IS phi -- the pressure correctors wrote phi, then ++runTime stored it -- so phiCN is phi
# to round-off and alphaEqn.H's blend acts only on a second outer pass. That is what cnOuter is for.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_cn_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/damBreak/damBreak"
STEPS=${STEPS:-20}
DT=${DT:-1e-3}

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v setFields > /dev/null 2>&1 || { echo "SKIP: setFields not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "^density  *variable;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the tutorial no longer ships \`density variable\`"; return 1; }
    grep -q "default  *Euler;" "$C/system/fvSchemes" \
        || { echo "FAIL: the tutorial's ddtSchemes default is no longer Euler, so the control means something else"; return 1; }
    PROFILE="$profile" STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
p = os.environ['PROFILE']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'^adjustTimeStep .*', 'adjustTimeStep  no;',        s, flags=re.M)
s = re.sub(r'^deltaT .*',         'deltaT          %s;' % dt,   s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         %.10g;' % (n*float(dt)), s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;',  s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   %d;' % n,    s, flags=re.M)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',     s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',        s, flags=re.M)
s = re.sub(r'^writeCompression .*', 'writeCompression off;',    s, flags=re.M)
open(c, 'w').write(s)
if not p.startswith('euler'):
    q = os.path.join(d, 'system/fvSchemes')
    t = open(q).read()
    oc = '1' if p == 'cnFull' else '0.5'
    t, k = re.subn(r'(ddtSchemes\s*\{[^}]*default\s+)Euler;', r'\1CrankNicolson %s;' % oc, t)
    assert k == 1, 'the ddtSchemes default was not replaced'
    open(q, 'w').write(t)
# `ddt(alpha)` NAMED SEPARATELY FROM THE DEFAULT. alphaEqn.H:242-259 branches on `ddt(rho,U)` while
# ocCoeff comes from `ddt(alpha)` (alphaEqn.H:6-56), so the mixed cases are well-defined runs rather than
# something OpenFOAM rejects -- it is brae that used to refuse them for want of a gate.
if p == 'cnAlphaEuler':
    q = os.path.join(d, 'system/fvSchemes')
    t = open(q).read()
    t, k = re.subn(r'(ddtSchemes\s*\{\s*\n\s*default\s+CrankNicolson 0\.5;\n)',
                   r'\1    ddt(alpha)      Euler;\n', t)
    assert k == 1, 'ddt(alpha) Euler was not inserted beside a CrankNicolson default'
    open(q, 'w').write(t)
if p == 'eulerAlphaCN':
    q = os.path.join(d, 'system/fvSchemes')
    t = open(q).read()
    t, k = re.subn(r'(ddtSchemes\s*\{\s*\n\s*default\s+Euler;\n)',
                   r'\1    ddt(alpha)      CrankNicolson 0.5;\n', t)
    assert k == 1, 'ddt(alpha) CrankNicolson was not inserted beside an Euler default'
    open(q, 'w').write(t)
if p in ('cnSST', 'cnLES', 'eulerSST', 'eulerLES'):
    # THE OTHER TWO CLOSURES UNDER THE SCHEME. kEpsilon carried CrankNicolson from the start; this
    # stages the tutorial's own damBreak with kOmegaSST and with LES kEqn in its place, so the
    # closure's own fvm::ddt -- kOmegaSSTBase.C:572 and :602, kEqn.C:172 -- is the term under test.
    # The `density variable` lineage is refused under both, so it goes back to the uniform one.
    q = os.path.join(d, 'constant/turbulenceProperties')
    t = open(q).read()
    t, k = re.subn(r'^\s*density\s+variable;\s*$', '', t, flags=re.M)
    assert k == 1, 'the tutorial no longer ships `density variable`'
    if p.endswith('LES'):
        t, k = re.subn(r'simulationType\s+RAS;', 'simulationType  LES;', t)
        assert k == 1
        t, k = re.subn(r'RAS\s*\{[^}]*\}',
                       'LES\n{\n    LESModel        kEqn;\n    turbulence      on;\n'
                       '    printCoeffs     on;\n    delta           cubeRootVol;\n'
                       '    cubeRootVolCoeffs { deltaCoeff 1; }\n}', t, flags=re.S)
        assert k == 1
    else:
        t, k = re.subn(r'RASModel\s+\w+;', 'RASModel        kOmegaSST;', t)
        assert k == 1
    open(q, 'w').write(t)
    # the tutorial convects k and epsilon with rhoPhi, which is the `density variable` lineage's
    # flux; the uniform one convects with phi, and OpenFOAM looks the entry up under that name
    q = os.path.join(d, 'system/fvSchemes')
    t = open(q).read()
    t, k = re.subn(r'div\(rhoPhi,(k|epsilon)\)(\s+)(\S+ \S+);',
                   r'div(phi,\1)\2\3;', t)
    assert k == 2, 'the tutorial no longer names div(rhoPhi,k) and div(rhoPhi,epsilon)'
    t, k = re.subn(r'div\(phi,epsilon\)', 'div(phi,omega)', t)
    assert k == 1
    open(q, 'w').write(t)
    if p.endswith('SST'):
        open(q, 'a').write('\nwallDist\n{\n    method meshWave;\n}\n')
    for fld, dims, val, wall in [
            ('k', '[0 2 -2 0 0 0 0]', '0.0001',
             'type zeroGradient;' if p.endswith('LES') else 'type kqRWallFunction; value uniform 0.0001;'),
            ('omega', '[0 0 -1 0 0 0 0]', '3', 'type omegaWallFunction; value uniform 3;'),
            ('nut', '[0 2 -1 0 0 0 0]', '0',
             'type zeroGradient;' if p.endswith('LES') else 'type nutkWallFunction; value uniform 0;')]:
        if fld == 'omega' and p.endswith('LES'):
            continue
        body = ('FoamFile { version 2.0; format ascii; class volScalarField; object %s; }\n'
                'dimensions      %s;\ninternalField   uniform %s;\nboundaryField\n{\n'
                '    "(leftWall|rightWall|lowerWall)" { %s }\n'
                '    atmosphere   { type zeroGradient; }\n'
                '    defaultFaces { type empty; }\n}\n') % (fld, dims, val, wall)
        open(os.path.join(d, '0', fld), 'w').write(body)
        open(os.path.join(d, '0.orig', fld), 'w').write(body)
    q = os.path.join(d, 'system/fvSolution')
    t = open(q).read()
    t = t.replace('"(U|k|epsilon).*"', '"(U|k|epsilon|omega).*"')
    open(q, 'w').write(t)
if p == 'cnOuter':
    q = os.path.join(d, 'system/fvSolution')
    t = open(q).read()
    t, k = re.subn(r'nOuterCorrectors\s+1;', 'nOuterCorrectors 2;', t)
    assert k == 1, 'nOuterCorrectors 1 was not found'
    open(q, 'w').write(t)
PYEOF
    # THE ORACLE IS CACHED FROM HERE -- see tests/of_oracle_cache.sh. Everything above is staging;
    # everything below is real OpenFOAM, which is what a re-run should not pay for twice.
    local key
    key=$(oracleKey "$C" "interfoam_cn" "$profile" "$STEPS" "$DT")
    if oracleRestore "$C" "$key" "$END"; then
        echo "OpenFOAM's $STEPS steps of deltaT $DT to t = $END reused from the oracle cache   [$profile]"
        return 0
    fi
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    # the oracle took the path: the ddt0 fields exist only under CrankNicolson
    if [ "${profile#euler}" != "$profile" ]; then
        [ ! -e "$C/$END/ddt0(rho,U)" ] || { echo "FAIL: the Euler control wrote a ddt0 field"; return 1; }
    else
        [ -e "$C/$END/ddt0(rho,U)" ] || { echo "FAIL: OpenFOAM wrote no ddt0(rho,U) under CrankNicolson [$profile]"; ls "$C/$END"; return 1; }
    fi
    oracleStore "$C" "$key"
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in euler cn cnOuter cnFull eulerSST cnSST eulerLES cnLES cnAlphaEuler eulerAlphaCN; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_cn_vs_openfoam: staging failed"; exit 1; }

for p in cn cnOuter cnFull; do
    "$BIN" "$W/$p" "$W/$p/0" "$W/$p/$END" "$STEPS" "$W/$p/log.interFoam" "$W/euler/$END" "$p" || rc=1
done
# ...and the OTHER TWO CLOSURES under the scheme, each against the SAME closure under Euler. This is
# where the DEVICE arm holds CrankNicolson in kOmegaSST and in kEqn: the cyclic fixture's `sstCN` and
# `lesCN` profiles hold the host arm but refuse the device one, because CrankNicolson's end-of-step
# alpha flux across a COUPLED PAIR is a separate thing this loop does not carry.
for p in cnSST cnLES; do
    "$BIN" "$W/$p" "$W/$p/0" "$W/$p/$END" "$STEPS" "$W/$p/log.interFoam" "$W/euler${p#cn}/$END" "$p" || rc=1
done

# ...and `ddt(alpha)` NAMED SEPARATELY FROM THE DEFAULT, both directions. BOTH take the BOTH-CrankNicolson
# run as their control, because that is the answer brae produced while it ran "the two under one scheme":
# MEASURED, OpenFOAM against OpenFOAM at this gate's own 20 steps, cnAlphaEuler against cn reads U
# 3.4565e-02 / alpha 1.5801e-03 / epsilon 7.5480e-02, and eulerAlphaCN against cn reads U 3.1624e-02 /
# alpha 8.9971e-03. Against the ALL-EULER run, cnAlphaEuler reads U 3.5786e-02 / alpha 1.0577e-02.
#
# ONE OF THE TWO IS A NO-OP IN OPENFOAM ITSELF, and it is asserted rather than assumed -- see below.
python3 - "$W/eulerAlphaCN/$END" "$W/euler/$END" <<'NOOPEOF' || rc=1
import re, sys
def cells(path, vector):
    t = open(path).read()
    kind = "vector" if vector else "scalar"
    m = re.search(r"internalField\s+nonuniform List<%s>\s*(\d+)\s*\((.*?)\n\)\s*;" % kind, t, re.S)
    if m is None:
        raise SystemExit("FAIL: %s is uniform" % path)
    if vector:
        return [tuple(float(x) for x in v.split()) for v in re.findall(r"\(([^()]*)\)", m.group(2))]
    return [(float(x),) for x in m.group(2).split()]
bad = 0
for fld, vec in (("alpha.water", False), ("U", True)):
    a = cells(sys.argv[1] + "/" + fld, vec)
    b = cells(sys.argv[2] + "/" + fld, vec)
    assert len(a) == len(b) and a
    d = max(max(abs(x - y) for x, y in zip(u, v)) for u, v in zip(a, b))
    ref = max(max(abs(x) for x in u) for u in a) or 1e-300
    n = sum(1 for u, v in zip(a, b) if any(x != y for x, y in zip(u, v)))
    print("  PROPERTY: OpenFOAM `ddt(alpha) CrankNicolson` under an EULER default equals plain Euler, "
          "%s %.4e over %d of %d cells" % (fld, d / ref, n, len(a)))
    # `phiCN` is cnCoeff*phi + (1 - cnCoeff)*phi.oldTime(), storeOldTimes copies the previous step's final
    # flux at the top of the step, and nothing touches phi before the alpha equation -- so the two fields
    # hold the same values and the blend is the identity. ddt(alpha)'s off-centring reaches the answer only
    # through the un-blend in the non-Euler momentum branch. At `CrankNicolson 1` this is EXACTLY 0; at 0.5
    # it is round-off on 2/3 + 1/3, which is why the bound is 1e-13 and not zero.
    if d / ref > 1e-13:
        print("  FAIL: it does not -- the blend is reaching the answer, so the mixed arm below means "
              "something different from what its comment says")
        bad = 1
sys.exit(bad)
NOOPEOF

for p in cnAlphaEuler eulerAlphaCN; do
    "$BIN" "$W/$p" "$W/$p/0" "$W/$p/$END" "$STEPS" "$W/$p/log.interFoam" "$W/cn/$END" "$p" || rc=1
done

echo "interfoam_cn_vs_openfoam: rc $rc"
exit $rc
