#!/usr/bin/env bash
# THE totalPressure FAMILY AND totalTemperature, WRITTEN AND READ BACK, against real OpenFOAM v2412: what brae writes for
# them OpenFOAM starts from and keeps, brae starts from what OpenFOAM writes, and brae restarted from its own output
# continues as the uninterrupted run -- on simpleFoam, pimpleFoam, rhoSimpleFoam and rhoCentralFoam, which share the
# field writer (src/finiteVolume/fields/foam_field_writer.cuh) and reader (foam_field_reader.cuh).
#
# What this found (2026-11-19), each now a check below:
#   * uniformTotalPressure (pimpleFoam/RAS/TJunction's inlet, p0 a table) was written as `type` + `inletValue` +
#     `value`. OpenFOAM stopped on it ("Missing or invalid Function1 entry: p0"), and brae restarted from it with p0
#     held at p0(0) = 10 where the table ramps it: U 2.3 % and p 5.6 % off the uninterrupted run ten steps later, with
#     no message. Now written as OpenFOAM writes it (rho, psi, gamma, `p0 table;` + `p0Coeffs`), and a
#     uniformTotalPressure with no p0 is refused by name, as OpenFOAM refuses it.
#   * fanPressure lost its fan curve and direction and wrote p0 as `inletValue`: brae refused its own output ("has no
#     `fanCurve` entry") and OpenFOAM stopped ("Required entry 'p0' missing"). brae also refused OpenFOAM's own output,
#     whose curve is `fanCurve table;` + `fanCurveCoeffs` (Function1::writeData). Both read and written now.
#   * phi on a symmetry, symmetryPlane or wedge patch was written `calculated`, which OpenFOAM refuses on a restart
#     ("inconsistent patch and patchField types for patch type symmetry and patchField type calculated"):
#     windAroundBuildingsBox's frontAndBack.
#   * totalPressure itself (every form's p0 under `p0`) and totalTemperature's psi default were fixed before this gate
#     (34554e73); the arms here hold them.
#
#   ARM tj    pimpleFoam/RAS/TJunction (uniformTotalPressure, p0 table) at a fixed deltaT 1e-3, 20 steps, written every
#             10 at writePrecision 17: brae's inlet entries equal OpenFOAM's; OpenFOAM starts from brae's 0.01 and keeps
#             the table; brae from brae's 0.01 reaches 0.02 as the uninterrupted run (bound 1e-12; measured 0); brae
#             starts from OpenFOAM's 0.01.
#   ARM fan   the same case with TJunctionFan's fanPressure inlet (its curve file; TJunctionFan's own cyclic fan baffles
#             are refused by brae): the same four checks.
#   ARM wab   validation/windAroundBuildingsBox (simpleFoam, outlet totalPressure, a symmetry patch) from OpenFOAM's 400
#             two iterations: brae's outlet and phi entries equal OpenFOAM's; OpenFOAM starts from brae's 401; brae from
#             brae's 401 reaches 402 as the uninterrupted run (bound 1e-12; measured 0).
#   ARM rtp   validation/rhoTP (rhoSimpleFoam, totalPressure inlet), two iterations: brae's inlet entries equal
#             OpenFOAM's; OpenFOAM starts from brae's 1 and keeps p0; the mirror (BRAE_RHOSIMPLEFOAM_MIRROR=cuda) from
#             its own 1 reaches 2 within 1e-6 of its uninterrupted run (measured 1.7e-8: its fields are written at 12
#             digits). The default driver's restart is reported, not checked here: validation/restart_vs_openfoam.sh holds
#             it to 1e-3 (measured 1.7e-4 here, and 3.8e-4 on rhoBox, which has no totalPressure).
#   ARM duct  tests/rho_central_3d.py duct_tp (rhoCentralFoam: totalPressure psi form, totalTemperature, waveTransmissive)
#             to 1e-3: brae's p and T entries equal OpenFOAM's; OpenFOAM starts from brae's 5e-4; brae from OpenFOAM's
#             5e-4 (T written without psi, its default) and from its own both reach 1e-3 within 1e-8 of OpenFOAM's
#             restart from the same directory (measured 4.5e-11 and 1.2e-10).
#   CONTROLS  each arm's file as brae wrote it before (p0 as `inletValue`; phi `calculated` on the symmetry patch;
#             T's psi read as none): OpenFOAM refuses it -- so the arms pass because the entries are there, not for
#             another reason. brae refuses the uniformTotalPressure and fanPressure ones by name (their table and curve
#             are not in the file), and the psi none one; a plain totalPressure spelled that way it reads as p0 with a
#             notice (the value is p0's) and must continue exactly as from the same file spelled `p0` (wab, rtp).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
RCFBIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
H="$ROOT/tests/total_bc_roundtrip.py"
[ -x "$BRAE" ]   || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -x "$RCFBIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $RCFBIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM at $OFBASHRC"; exit 77; }
for d in incompressible/pimpleFoam/RAS/TJunction incompressible/pimpleFoam/RAS/TJunctionFan \
         compressible/rhoCentralFoam/LadenburgJet60psi; do
    [ -d "$TUT/$d" ] || { echo "SKIP: no $d under $TUT"; exit 77; }
done
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-total-bc.XXXXXX)}
mkdir -p "$W"
[ -z "$(ls -A "$W")" ] || { echo "FAIL: the work directory $W is not empty"; exit 1; }
echo "work in $W"

passes=0; fails=0
ok()   { passes=$((passes + 1)); echo "ok   $*"; }
fail() { fails=$((fails + 1)); echo "FAIL $*"; }
of()   { bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"; }
le()   { python3 -c "import sys; sys.exit(0 if float('${1:-nan}') <= float('$2') else 1)" 2> /dev/null; }
# brae <dir> [env...]: brae in <dir>, its exit code in <dir>/rc
brae() { local d=$1; shift; (cd "$d" && env "$@" "$BRAE" . > log.brae 2>&1 < /dev/null; echo $? > rc); }
ofrun() { of "cd '$1' && $2 > log.of 2>&1" < /dev/null; echo $? > "$1/rc.of"; }
# restartdir <new dir> <case dir for system/constant> <time dir>: a case that starts from that time
restartdir()
{
    mkdir "$1" && cp -r "$2/system" "$2/constant" "$1/" && cp -r "$3" "$1/" \
        && python3 "$H" ctl "$1/system/controlDict" startFrom=latestTime
}
# same <name> <brae file> <OpenFOAM file> <patch>: the patch's entries but `value`, with OpenFOAM's defaults
same()
{
    local out
    if out=$(python3 "$H" same "$2" "$3" "$4")
    then ok "$1: brae writes $(basename "$(dirname "$2")")/$(basename "$2") $4 with OpenFOAM's entries"
    else fail "$1: $(basename "$2") $4 differs from OpenFOAM's: $(echo "$out" | tr '\n' ' ')"; fi
}
started() { [ "$(cat "$2/rc.of" 2>/dev/null)" = 0 ] && ok "$1" || fail "$1: $(grep -m 1 -A 3 FATAL "$2/log.of" | tr '\n' ' ' | cut -c1-240)"; }
braestarted() { [ "$(cat "$2/rc" 2>/dev/null)" = 0 ] && ok "$1" || fail "$1: $(grep -v NOTICE "$2/log.brae" | grep -m 1 -i "error\|brae:" | cut -c1-240)"; }
# refused <name> <dir> <OpenFOAM's message> <brae's message>: both codes stop on the control's file, each saying why
refused()
{
    if [ "$(cat "$2/rc.of")" != 0 ] && grep -q "$3" "$2/log.of"
    then ok "control $1: OpenFOAM stops on it ($3)"
    else fail "control $1: OpenFOAM did not stop with '$3'"; fi
    if [ "$(cat "$2/rc")" != 0 ] && grep -q "$4" "$2/log.brae"
    then ok "control $1: brae refuses it by name ($4)"
    else fail "control $1: brae did not refuse it with '$4'"; fi
}
# legacy <name> <dir> <restart from brae's p0 spelling> <time> <fields...>: (OpenFOAM's run in <dir>_of) a totalPressure whose p0 is spelled
# `inletValue`, as brae wrote it before 2026-11-19. OpenFOAM refuses it; brae reads it as p0 with a notice and must
# continue exactly as from the same file with `p0`, and write the key `p0` back
legacy()
{
    local n=$1 d=$2 ref=$3 t=$4; shift 4
    if [ "$(cat "${d}_of/rc.of")" != 0 ] && grep -q "Required entry 'p0' missing" "${d}_of/log.of"
    then ok "control $n: OpenFOAM stops on it (Required entry 'p0' missing)"
    else fail "control $n: OpenFOAM did not stop with 'Required entry 'p0' missing'"; fi
    if [ "$(cat "$d/rc")" = 0 ] && grep -q "its p0 under \`inletValue\`" "$d/log.brae"
    then ok "control $n: brae reads it as p0, with a notice"
    else fail "control $n: brae did not read it with the notice"; fi
    within "control $n: brae from it against brae from the same file with p0, at $t" \
        "$(python3 "$H" fields "$d/$t" "$ref/$t" "$@" 2>/dev/null)" 0
    if [ -n "$(python3 "$H" entry "$d/$t/p" "$P" p0)" ]
    then ok "control $n: brae's next write spells it p0"
    else fail "control $n: brae's next write has no p0"; fi
}
# within <name> <value> <bound>
within() { if le "$2" "$3"; then ok "$1: $2 (bound $3)"; else fail "$1: $2 above the bound $3"; fi; }

# ---- tj and fan: pimpleFoam on TJunction's mesh, fixed deltaT (brae's adjustTimeStep write times are not this gate's) ----
stagetj()   # stagetj <dir> [fan]
{
    mkdir "$1" && cp -r "$TUT/incompressible/pimpleFoam/RAS/TJunction/0" "$TUT/incompressible/pimpleFoam/RAS/TJunction/system" \
        "$TUT/incompressible/pimpleFoam/RAS/TJunction/constant" "$1/"
    python3 "$H" ctl "$1/system/controlDict" adjustTimeStep=no deltaT=0.001 endTime=0.02 writeControl=timeStep \
        writeInterval=10 writePrecision=17
    if [ "${2:-}" = fan ]
    then
        cp "$TUT/incompressible/pimpleFoam/RAS/TJunctionFan/constant/FluxVsdP.dat" "$1/constant/"
        python3 - "$1/0/p" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s, n = re.subn(r'inlet\s*\{.*?\n    \}', '''inlet
    {
        type            fanPressure;
        direction       in;
        fanCurve
        {
            type table;
            file "<constant>/FluxVsdP.dat";
        }
        p0              uniform 30;
    }''', s, count=1, flags=re.S)
assert n == 1
open(p, 'w').write(s)
PYEOF
    fi
    of "cd '$1' && blockMesh > log.blockMesh 2>&1" < /dev/null
}
pimplearm()   # pimplearm <name> [fan]
{
    local n=$1 m=${2:-}
    stagetj "$W/${n}_case" "$m" || { fail "$n: staging"; return; }
    cp -r "$W/${n}_case" "$W/${n}_b"; cp -r "$W/${n}_case" "$W/${n}_of"
    brae "$W/${n}_b"; ofrun "$W/${n}_of" pimpleFoam
    braestarted "$n: brae runs 0 to 0.02" "$W/${n}_b"
    started "$n: OpenFOAM runs 0 to 0.02" "$W/${n}_of"
    same "$n" "$W/${n}_b/0.01/p" "$W/${n}_of/0.01/p" inlet
    restartdir "$W/${n}_of_from_b" "$W/${n}_case" "$W/${n}_b/0.01"
    python3 "$H" ctl "$W/${n}_of_from_b/system/controlDict" endTime=0.011 writeInterval=1
    ofrun "$W/${n}_of_from_b" pimpleFoam
    started "$n: OpenFOAM starts from brae's 0.01" "$W/${n}_of_from_b"
    same "$n (OpenFOAM's rewrite of brae's)" "$W/${n}_of_from_b/0.011/p" "$W/${n}_b/0.01/p" inlet
    restartdir "$W/${n}_b_from_b" "$W/${n}_case" "$W/${n}_b/0.01"; brae "$W/${n}_b_from_b"
    braestarted "$n: brae starts from brae's 0.01" "$W/${n}_b_from_b"
    within "$n: brae from its 0.01 against its uninterrupted run at 0.02" \
        "$(python3 "$H" fields "$W/${n}_b_from_b/0.02" "$W/${n}_b/0.02" U p k epsilon nut 2>/dev/null)" 1e-12
    restartdir "$W/${n}_b_from_of" "$W/${n}_case" "$W/${n}_of/0.01"; brae "$W/${n}_b_from_of"
    braestarted "$n: brae starts from OpenFOAM's 0.01" "$W/${n}_b_from_of"
    same "$n (brae's rewrite of OpenFOAM's)" "$W/${n}_b_from_of/0.02/p" "$W/${n}_of/0.01/p" inlet
    # control: the inlet as brae wrote it before 2026-11-19
    restartdir "$W/${n}_old" "$W/${n}_case" "$W/${n}_b/0.01"; python3 "$H" oldp0 "$W/${n}_old/0.01/p" inlet
    brae "$W/${n}_old"; ofrun "$W/${n}_old" pimpleFoam
    if [ "$m" = fan ]
    then refused "$n, p0 as inletValue" "$W/${n}_old" "Required entry 'p0' missing" "has no \`p0\` entry"
    else refused "$n, p0 as inletValue" "$W/${n}_old" "Function1 entry: p0" "with no \`p0\` entry"
    fi
}
pimplearm tj
pimplearm fan fan

# ---- wab: simpleFoam, outlet totalPressure, frontAndBack a symmetry patch, from OpenFOAM's converged 400 ----------------
mkdir "$W/wab_case" && cp -r "$ROOT/validation/windAroundBuildingsBox/"{constant,system,400} "$W/wab_case/"
python3 "$H" ctl "$W/wab_case/system/controlDict" startFrom=latestTime endTime=402 writeInterval=1 writeFormat=ascii \
    writePrecision=17
cp -r "$W/wab_case" "$W/wab_b"; cp -r "$W/wab_case" "$W/wab_of"
brae "$W/wab_b"; ofrun "$W/wab_of" simpleFoam
braestarted "wab: brae runs 400 to 402" "$W/wab_b"
started "wab: OpenFOAM runs 400 to 402" "$W/wab_of"
same wab "$W/wab_b/401/p" "$W/wab_of/401/p" outlet
same wab "$W/wab_b/401/phi" "$W/wab_of/401/phi" frontAndBack
restartdir "$W/wab_of_from_b" "$W/wab_case" "$W/wab_b/401"; ofrun "$W/wab_of_from_b" simpleFoam
started "wab: OpenFOAM starts from brae's 401" "$W/wab_of_from_b"
same "wab (OpenFOAM's rewrite of brae's)" "$W/wab_of_from_b/402/p" "$W/wab_b/401/p" outlet
restartdir "$W/wab_b_from_b" "$W/wab_case" "$W/wab_b/401"; brae "$W/wab_b_from_b"
braestarted "wab: brae starts from brae's 401" "$W/wab_b_from_b"
within "wab: brae from its 401 against its uninterrupted run at 402" \
    "$(python3 "$H" fields "$W/wab_b_from_b/402" "$W/wab_b/402" U p k epsilon nut 2>/dev/null)" 1e-12
restartdir "$W/wab_oldphi" "$W/wab_case" "$W/wab_b/401"; python3 "$H" settype "$W/wab_oldphi/401/phi" frontAndBack calculated
ofrun "$W/wab_oldphi" simpleFoam
if [ "$(cat "$W/wab_oldphi/rc.of")" != 0 ] && grep -q "inconsistent patch and patchField types" "$W/wab_oldphi/log.of"
then ok "control wab, phi calculated on the symmetry patch: OpenFOAM stops on it (inconsistent patch and patchField types)"
else fail "control wab, phi calculated on the symmetry patch: OpenFOAM did not stop"; fi
restartdir "$W/wab_old" "$W/wab_case" "$W/wab_b/401"; python3 "$H" oldp0 "$W/wab_old/401/p" outlet
cp -r "$W/wab_old" "$W/wab_old_of"; brae "$W/wab_old"; ofrun "$W/wab_old_of" simpleFoam
P=outlet legacy "wab, p0 as inletValue" "$W/wab_old" "$W/wab_b_from_b" 402 U p k epsilon nut

# ---- rtp: rhoSimpleFoam, inlet totalPressure ----------------------------------------------------------------------------
mkdir "$W/rtp_case" && cp -r "$ROOT/validation/rhoTP/"{constant,system,0.orig} "$W/rtp_case/" && cp -r "$W/rtp_case/0.orig" "$W/rtp_case/0"
python3 "$H" ctl "$W/rtp_case/system/controlDict" endTime=2 writeInterval=1 writePrecision=17 startFrom=latestTime
of "cd '$W/rtp_case' && blockMesh > log.blockMesh 2>&1" < /dev/null
for a in b bm of; do cp -r "$W/rtp_case" "$W/rtp_$a"; done
brae "$W/rtp_b"; brae "$W/rtp_bm" BRAE_RHOSIMPLEFOAM_MIRROR=cuda; ofrun "$W/rtp_of" rhoSimpleFoam
braestarted "rtp: brae runs 0 to 2" "$W/rtp_b"
braestarted "rtp: brae's mirror runs 0 to 2" "$W/rtp_bm"
started "rtp: OpenFOAM runs 0 to 2" "$W/rtp_of"
same rtp "$W/rtp_b/1/p" "$W/rtp_of/1/p" inlet
same "rtp (mirror)" "$W/rtp_bm/1/p" "$W/rtp_of/1/p" inlet
restartdir "$W/rtp_of_from_b" "$W/rtp_case" "$W/rtp_b/1"; ofrun "$W/rtp_of_from_b" rhoSimpleFoam
started "rtp: OpenFOAM starts from brae's 1" "$W/rtp_of_from_b"
same "rtp (OpenFOAM's rewrite of brae's)" "$W/rtp_of_from_b/2/p" "$W/rtp_b/1/p" inlet
restartdir "$W/rtp_bm_from_bm" "$W/rtp_case" "$W/rtp_bm/1"; brae "$W/rtp_bm_from_bm" BRAE_RHOSIMPLEFOAM_MIRROR=cuda
braestarted "rtp: the mirror starts from its 1" "$W/rtp_bm_from_bm"
within "rtp: the mirror from its 1 against its uninterrupted run at 2" \
    "$(python3 "$H" fields "$W/rtp_bm_from_bm/2" "$W/rtp_bm/2" U p T 2>/dev/null)" 1e-6
restartdir "$W/rtp_b_from_b" "$W/rtp_case" "$W/rtp_b/1"; brae "$W/rtp_b_from_b"
braestarted "rtp: brae starts from its 1" "$W/rtp_b_from_b"
echo "info rtp: the default driver from its 1 against its uninterrupted run at 2: $(python3 "$H" fields \
    "$W/rtp_b_from_b/2" "$W/rtp_b/2" U p T 2>/dev/null) (validation/restart_vs_openfoam.sh holds its restart to 1e-3)"
restartdir "$W/rtp_of_from_of" "$W/rtp_case" "$W/rtp_of/1"; ofrun "$W/rtp_of_from_of" rhoSimpleFoam
echo "info rtp: OpenFOAM from its 1 against its uninterrupted run at 2: $(python3 "$H" fields "$W/rtp_of_from_of/2" "$W/rtp_of/2" U p T)"
restartdir "$W/rtp_old" "$W/rtp_case" "$W/rtp_b/1"; python3 "$H" oldp0 "$W/rtp_old/1/p" inlet
cp -r "$W/rtp_old" "$W/rtp_old_of"; brae "$W/rtp_old"; ofrun "$W/rtp_old_of" rhoSimpleFoam
P=inlet legacy "rtp, p0 as inletValue" "$W/rtp_old" "$W/rtp_b_from_b" 2 U p T

# ---- duct: rhoCentralFoam, totalPressure (psi form) + totalTemperature inlet, waveTransmissive outlet -------------------
RCF="$TUT/compressible/rhoCentralFoam"
mkdir "$W/duct_case" && cp -r "$RCF/LadenburgJet60psi/system" "$RCF/LadenburgJet60psi/constant" "$W/duct_case/"
python3 "$ROOT/tests/rho_central_3d.py" setup duct_tp "$W/duct_case"
python3 "$H" ctl "$W/duct_case/system/controlDict" endTime=0.001
of "cd '$W/duct_case' && blockMesh > log.blockMesh 2>&1" < /dev/null
rcf() { (cd "$1" && "$RCFBIN" -case . > log.brae 2>&1 < /dev/null; echo $? > rc); }
cp -r "$W/duct_case" "$W/duct_b"; cp -r "$W/duct_case" "$W/duct_of"
rcf "$W/duct_b"; ofrun "$W/duct_of" rhoCentralFoam
braestarted "duct: brae runs 0 to 1e-3" "$W/duct_b"
started "duct: OpenFOAM runs 0 to 1e-3" "$W/duct_of"
same duct "$W/duct_b/0.0005/p" "$W/duct_of/0.0005/p" inlet
same duct "$W/duct_b/0.0005/T" "$W/duct_of/0.0005/T" inlet
# the outlet less `field`: brae echoes the case's `field p;`, an entry OpenFOAM's waveTransmissive does not read or write
for f in duct_b duct_of; do cp "$W/$f/0.0005/p" "$W/$f/0.0005/p.nofield"; python3 "$H" cut "$W/$f/0.0005/p.nofield" outlet field; done
same duct "$W/duct_b/0.0005/p.nofield" "$W/duct_of/0.0005/p.nofield" outlet
for s in b of; do
    src="$W/duct_$s/0.0005"
    restartdir "$W/duct_of_from_$s" "$W/duct_case" "$src"; ofrun "$W/duct_of_from_$s" rhoCentralFoam
    restartdir "$W/duct_b_from_$s" "$W/duct_case" "$src"; rcf "$W/duct_b_from_$s"
done
started "duct: OpenFOAM starts from brae's 5e-4" "$W/duct_of_from_b"
started "duct: OpenFOAM starts from its own 5e-4" "$W/duct_of_from_of"
braestarted "duct: brae starts from OpenFOAM's 5e-4 (T's psi left out, its default)" "$W/duct_b_from_of"
braestarted "duct: brae starts from its own 5e-4" "$W/duct_b_from_b"
within "duct: brae from OpenFOAM's 5e-4 against OpenFOAM from it, at 1e-3" \
    "$(python3 "$H" fields "$W/duct_b_from_of/0.001" "$W/duct_of_from_of/0.001" p T U rho 2>/dev/null)" 1e-8
within "duct: brae from brae's 5e-4 against OpenFOAM from it, at 1e-3" \
    "$(python3 "$H" fields "$W/duct_b_from_b/0.001" "$W/duct_of_from_b/0.001" p T U rho 2>/dev/null)" 1e-8
# controls: T's psi as none (the reader's default before 34554e73) and the inlet's p0 cut out
restartdir "$W/duct_psinone" "$W/duct_case" "$W/duct_of/0.0005"
python3 - "$W/duct_psinone/0.0005/T" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s, n = re.subn(r'(inlet\s*\{\s*\n\s*type\s+totalTemperature;)', r'\1\n        psi             none;', s, count=1)
assert n == 1
open(p, 'w').write(s)
PYEOF
rcf "$W/duct_psinone"
if [ "$(cat "$W/duct_psinone/rc")" != 0 ] && grep -q "totalTemperature.* with psi \`none\`" "$W/duct_psinone/log.brae"
then ok "control duct, T's psi none: brae refuses it by name"
else fail "control duct, T's psi none: brae did not refuse it"; fi
restartdir "$W/duct_nop0" "$W/duct_case" "$W/duct_b/0.0005"; python3 "$H" cut "$W/duct_nop0/0.0005/p" inlet p0
rcf "$W/duct_nop0"; ofrun "$W/duct_nop0" rhoCentralFoam
refused "duct, no p0" "$W/duct_nop0" "Required entry 'p0' missing" "with no \`p0\` entry"

echo "total_bc_roundtrip_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ]
