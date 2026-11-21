#!/usr/bin/env bash
# brae_rhoCentralFoam's fvSchemes and fvSolution keys as OpenFOAM v2412 names them (wave 1, Q4, 2026-11-21): what is
# refused by name at start-up, and beside each the nearest form that must run. A NEW test: tests/rho_central_refusals.sh
# keeps its rows.
#
#   RAS (jet_sst, tests/rho_central_turb.py): the compressible kOmegaSST's keys -- laplacian((rho*DkEff),k),
#     laplacian((rho*DomegaEff),omega) (were checked as DkEff/DomegaEff), the energy's laplacian(alphaEff,e) under RAS
#     (was alphahe), ddt(rho,k) and ddt(rho,omega) (never checked), interpolate(rho) (never checked); limitedLinear's
#     coefficient range; linearUpwind's named gradient; a relaxation factor that is not a constant; decayControl
#     (never read), beside kInf/omegaInf alone, which OpenFOAM zeroes and must run.
#   shockTube: any solver name on the diagonal rho/rhoU/rhoE solves (was refused unless `diagonal`), a solve entry with
#     no `solver` (refused, as OpenFOAM stops); a quoted reconstruct key (a regular expression to OpenFOAM) and a pattern;
#     upwind reconstruction beside a gradSchemes default brae does not compute (no gradient is formed: runs); the
#     viscous dotInterpolate(S,tauMC) (never checked); janaf's out-of-range warning (log only).
#   the unread-entry audit: fvSolution, turbulenceProperties, thermophysicalProperties and fvSchemes each name an
#     entry no reader asked for, as `brae NOTICE [unread]`; the shipped fluxScheme is not one.
# No OpenFOAM solver runs here (blockMesh and setFields only). CUDA arm, the default path.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
TURB="$ROOT/tests/rho_central_turb.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}"
RCF="$TUT/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC (blockMesh)"; exit 77; }
[ -d "$RCF/LadenburgJet60psi/0.orig" ] && [ -d "$RCF/shockTube/0.orig" ] || { echo "SKIP: tutorials missing"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-keyref.XXXXXX)}
[ -n "${BRAE_GATE_OUT:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
of() { bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"; }
fails=0
passes=0

edit() { python3 - "$@" <<'PY'
import re, sys
p, pat, rep = sys.argv[1], sys.argv[2], sys.argv[3].replace('\\n', '\n')
t = open(p).read()
t2 = re.sub(pat, lambda m: rep, t, count=1, flags=re.S)
assert t2 != t, (p, pat)
open(p, 'w').write(t2)
PY
}
short()   # short <case>: a few steps, the last written
{
    python3 - "$1/system/controlDict" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
dt = float(re.search(r'^deltaT\s+([^;]+);', s, re.M).group(1))
for key, val in [('adjustTimeStep', 'no'), ('endTime', '%.12g' % (5*dt)), ('writeControl', 'timeStep'),
                 ('writeInterval', '5')]:
    s, n = re.subn(r'^%s\s.*$' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert n == 1, key
open(p, 'w').write(s)
PY
}
J="$W/jet"; S="$W/shock"
mkdir -p "$J"; cp -r "$RCF/LadenburgJet60psi/system" "$RCF/LadenburgJet60psi/constant" "$J/"
cp -r "$RCF/LadenburgJet60psi/0.orig" "$J/0"; gunzip -f "$J/0/"*.gz
python3 "$TURB" setup jet_sst "$J" && of "cd '$J' && blockMesh > log.blockMesh 2>&1" || { echo "SKIP: jet staging"; exit 77; }
short "$J"
mkdir -p "$S"; cp -r "$RCF/shockTube/system" "$RCF/shockTube/constant" "$S/"; cp -r "$RCF/shockTube/0.orig" "$S/0"
of "cd '$S' && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1" || { echo "SKIP: shockTube staging"; exit 77; }
short "$S"

# row <name> <base: jet|shock> <expect: refused|runs> <needle in the log, or -> <edits...>: each edit is
# `<file> <regex> <replacement>` (three words) or `append <file> <text>`
row()
{
    local name=$1 base=$2 expect=$3 needle=$4
    shift 4
    local d="$W/r_$name"
    rm -rf "$d"; cp -r "$W/$base" "$d"
    while [ $# -gt 0 ]; do
        if [ "$1" = append ]; then printf '%b' "$3" >> "$d/$2"; shift 3
        else edit "$d/$1" "$2" "$3" || { echo "FAIL $name: the staging edit failed"; fails=$((fails + 1)); return; }; shift 3; fi
    done
    (cd "$d" && "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
    local got=refused
    [ "$(cat "$d/rc")" = 0 ] && got=runs
    local good=1
    [ "$got" = "$expect" ] || good=0
    if [ "$needle" != - ]; then grep -qF -- "$needle" "$d/log" || good=0; fi
    if [ $good = 1 ]; then passes=$((passes + 1)); printf 'ok   %-28s %s\n' "$name" "$got"
    else
        fails=$((fails + 1))
        printf 'FAIL %-28s expected %s%s, got %s: %s\n' "$name" "$expect" "$([ "$needle" != - ] && echo " naming \`$needle\`")" \
            "$got" "$(grep -E '^brae rhoCentralFoam: |^brae compressible RAS: |rhoCentralFoam: brae' "$d/log" | tail -n 1 | cut -c1-200)"
    fi
}
# same <a> <b>: two rows' written directories byte for byte
same()
{
    local t n=0
    for t in $(cd "$W/r_$1" && ls -d [0-9]* | grep -vx 0); do diff -r -q "$W/r_$1/$t" "$W/r_$2/$t" > /dev/null || n=$((n + 1)); done
    if [ $n = 0 ] && [ -n "$(cd "$W/r_$2" && ls -d [0-9]* | grep -vx 0)" ]; then passes=$((passes + 1)); echo "ok   $2 writes $1's bytes"
    else fails=$((fails + 1)); echo "FAIL $2 does not write $1's bytes ($n directories differ)"; fi
}
LAP='system/fvSchemes (laplacianSchemes\s*\{)'

echo "== rhoCentralFoam RAS keys (jet_sst) =="
row jet_plain jet runs -
row lap_rhoDkEff_uncorrected jet refused "resolves laplacian((rho*DkEff),k) to \`Gauss linear uncorrected\`" \
    $LAP 'laplacianSchemes\n{\n    laplacian((rho*DkEff),k) Gauss linear uncorrected;'
row lap_rhoDomegaEff_orthogonal jet refused "carries one laplacian scheme for both equations" \
    $LAP 'laplacianSchemes\n{\n    laplacian((rho*DomegaEff),omega) Gauss linear orthogonal;'
row lap_alphaEff_uncorrected jet refused "resolves laplacian(alphaEff,e) to \`Gauss linear uncorrected\`" \
    $LAP 'laplacianSchemes\n{\n    laplacian(alphaEff,e) Gauss linear uncorrected;'
row lap_named_corrected jet runs - \
    $LAP 'laplacianSchemes\n{\n    laplacian((rho*DkEff),k) Gauss linear corrected;\n    laplacian((rho*DomegaEff),omega) Gauss linear corrected;\n    laplacian(alphaEff,e) Gauss linear corrected;'
same jet_plain lap_named_corrected
row lap_oldName_ignored jet runs - \
    $LAP 'laplacianSchemes\n{\n    laplacian(DkEff,k) Gauss linear uncorrected;'
row ddt_rhoK_backward jet refused "resolves ddt(rho,k) to \`backward\`" \
    system/fvSchemes '(ddtSchemes\s*\{)' 'ddtSchemes\n{\n    ddt(rho,k) backward;'
row ddt_rhoOmega_euler jet runs - \
    system/fvSchemes '(ddtSchemes\s*\{)' 'ddtSchemes\n{\n    ddt(rho,omega) Euler;'
same jet_plain ddt_rhoOmega_euler
row interpolate_rho_cubic jet refused "resolves interpolate(rho) to \`cubic\`" \
    system/fvSchemes '(interpolationSchemes\s*\{)' 'interpolationSchemes\n{\n    interpolate(rho) cubic;'
row limitedLinear_1p5 jet refused "should be >= 0 and <= 1" \
    system/fvSchemes 'div\(phi,k\)\s+Gauss limitedLinear 1;' 'div(phi,k)      Gauss limitedLinear 1.5;' \
    system/fvSchemes 'div\(phi,omega\)\s+Gauss limitedLinear 1;' 'div(phi,omega)  Gauss limitedLinear 1.5;'
row lu_leastSquares jet refused "linearUpwind's correction is ported with" \
    system/fvSchemes 'div\(phi,k\)\s+Gauss limitedLinear 1;' 'div(phi,k)      Gauss linearUpwind limited;' \
    system/fvSchemes 'div\(phi,omega\)\s+Gauss limitedLinear 1;' 'div(phi,omega)  Gauss linearUpwind limited;' \
    system/fvSchemes '(gradSchemes\s*\{)' 'gradSchemes\n{\n    limited leastSquares;'
row lu_limited jet runs - \
    system/fvSchemes 'div\(phi,k\)\s+Gauss limitedLinear 1;' 'div(phi,k)      Gauss linearUpwind limited;' \
    system/fvSchemes 'div\(phi,omega\)\s+Gauss limitedLinear 1;' 'div(phi,omega)  Gauss linearUpwind limited;' \
    system/fvSchemes '(gradSchemes\s*\{)' 'gradSchemes\n{\n    limited cellLimited Gauss linear 1;'
row relax_table jet refused "a value that is not a constant" \
    append system/fvSolution '\nrelaxationFactors\n{\n    equations\n    {\n        k table ((0 0.5) (1 0.9));\n    }\n}\n'
row relax_constant jet runs - \
    append system/fvSolution '\nrelaxationFactors\n{\n    equations\n    {\n        "(k|omega)" 0.7;\n    }\n}\n'
row decayControl jet refused "sets \`decayControl yes\`" \
    constant/turbulenceProperties '(RAS\s*\{)' 'RAS\n{\n    kOmegaSSTCoeffs { decayControl yes; kInf 1; omegaInf 1; }'
row kInf_without_decayControl jet runs - \
    constant/turbulenceProperties '(RAS\s*\{)' 'RAS\n{\n    kOmegaSSTCoeffs { kInf 1; omegaInf 1; }'
same jet_plain kInf_without_decayControl

echo "== rhoCentralFoam solver and scheme keys (shockTube) =="
row shock_plain shock runs -
row diagonal_PCG shock runs - \
    system/fvSolution '"\(rho\|rhoU\|rhoE\)"\s*\{[^}]*\}' '"(rho|rhoU|rhoE)"\n    {\n        solver          PCG;\n        preconditioner  DIC;\n    }'
same shock_plain diagonal_PCG
row diagonal_noSolver shock refused "names no \`solver\`" \
    system/fvSolution '"\(rho\|rhoU\|rhoE\)"\s*\{[^}]*\}' '"(rho|rhoU|rhoE)"\n    {\n        tolerance 1e-6;\n    }'
row reconstruct_quotedLiteral shock runs "reconstruct: rho linear, T vanLeer, U vanLeerV" \
    system/fvSchemes 'reconstruct\(rho\)\s+vanLeer;' '"reconstruct(rho)" Minmod;'
row reconstruct_pattern shock runs "reconstruct: rho Minmod, T Minmod, U vanLeerV" \
    system/fvSchemes 'reconstruct\(rho\)\s+vanLeer;' '"reconstruct\((rho|T)\)" Minmod;' \
    system/fvSchemes 'reconstruct\(T\)\s+vanLeer;' ''
row upwind_leastSquaresDefault shock runs - \
    system/fvSchemes 'reconstruct\(rho\)\s+vanLeer;' 'reconstruct(rho) upwind;' \
    system/fvSchemes 'reconstruct\(U\)\s+vanLeerV;' 'reconstruct(U) upwind;' \
    system/fvSchemes 'reconstruct\(T\)\s+vanLeer;' 'reconstruct(T) upwind;' \
    system/fvSchemes '(gradSchemes\s*\{\s*default\s+)Gauss linear;' 'gradSchemes\n{\n    default leastSquares;'
row vanLeer_leastSquaresDefault shock refused "grad(rho) to \`leastSquares\`" \
    system/fvSchemes '(gradSchemes\s*\{\s*default\s+)Gauss linear;' 'gradSchemes\n{\n    default leastSquares;'
VISC=(constant/thermophysicalProperties '(\bmu\s+)0;' 'mu              1.8e-05;' system/fvSolution '(\n\s*h\s*\n)' '\n    e\n')
row viscous_plain shock runs - "${VISC[@]}"
row dotInterpolate_cubic shock refused "resolves dotInterpolate(S,tauMC) to \`cubic\`" "${VISC[@]}" \
    system/fvSchemes '(interpolationSchemes\s*\{)' 'interpolationSchemes\n{\n    dotInterpolate(S,tauMC) cubic;'
JANAF='thermodynamics\n    {\n        Tlow            TLOW;\n        Thigh           10000;\n        Tcommon         1000;\n        highCpCoeffs    ( 2.9525407 0.0013968838 -4.9262577e-07 7.8600091e-11 -4.6074978e-15 -923.93753 5.8718221 );\n        lowCpCoeffs     ( 3.5309628 -0.0001236595 -5.0299339e-07 2.4352768e-09 -1.4087954e-12 -1046.9637 2.9674391 );\n    }'
# janaf needs sutherland transport (OpenFOAM v2412 has no const-transport janaf psiThermo), so a viscous case
SUTH=(constant/thermophysicalProperties 'transport(\s+)const;' 'transport       sutherland;'
      constant/thermophysicalProperties '(\n\s*transport\s*\{)[^}]*\}' '\n    transport\n    {\n        As              1.458e-06;\n        Ts              110;\n        Pr              0.72;\n    }'
      system/fvSolution '(\n\s*h\s*\n)' '\n    e\n'
      constant/thermophysicalProperties 'thermo(\s+)hConst;' 'thermo          janaf;')
row janaf_outOfRange shock runs "attempt to use janafThermo<EquationOfState> out of temperature range 300 -> 10000" "${SUTH[@]}" \
    constant/thermophysicalProperties 'thermodynamics\s*\{[^}]*\}' "${JANAF//TLOW/300}"
row janaf_inRange shock runs - "${SUTH[@]}" \
    constant/thermophysicalProperties 'thermodynamics\s*\{[^}]*\}' "${JANAF//TLOW/100}"
if grep -q "out of temperature range" "$W/r_janaf_inRange/log"
then fails=$((fails + 1)); echo "FAIL janaf_inRange warns about the range"
else passes=$((passes + 1)); echo "ok   janaf_inRange does not warn"; fi

echo "== the unread-entry audit =="
row audit_fvSolution shock runs "brae NOTICE [unread]   bogusEntry" append system/fvSolution '\nbogusEntry 1;\n'
row audit_thermo shock runs "brae NOTICE [unread]   bogusThermo" append constant/thermophysicalProperties '\nbogusThermo 1;\n'
row audit_fvSchemes shock runs "brae NOTICE [unread]   fluxRequired/ (whole block)" \
    append system/fvSchemes '\nfluxRequired\n{\n    default no;\n}\n'
row audit_turbulence jet runs "brae NOTICE [unread]   RAS/bogusRas" \
    constant/turbulenceProperties '(RAS\s*\{)' 'RAS\n{\n    bogusRas 1;'
if grep -q "fluxScheme" "$W/r_shock_plain/log" | grep -q unread 2> /dev/null || grep -q "NOTICE \[unread\]   fluxScheme" "$W/r_shock_plain/log"
then fails=$((fails + 1)); echo "FAIL shock_plain reports the shipped fluxScheme unread"
else passes=$((passes + 1)); echo "ok   the shipped fluxScheme is not reported unread"; fi

echo "rho_central_keys_refusals: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ]
