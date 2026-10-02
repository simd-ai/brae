# Sourced by cn_moving/released*.sh: the released body's fixture (arm X1), staged once per file with
# OpenFOAM's run from the oracle cache. Sets BOUND_X1, d (OpenFOAM's case) and xt (its one write time).
x1_stage()
{
    BOUND_X1=3e-10
    d="$W/x1_of"
    mkdir -p "$d"
    cp -r "$W/w_of_floatingObject/0" "$W/w_of_floatingObject/constant" "$W/w_of_floatingObject/system" "$d/"
    python3 - "$d" <<'EOF_X1' || say "ARM X1 floatingObject: the released-body staging did not apply" FAIL
import re, sys
d = sys.argv[1]
p = d + '/constant/dynamicMeshDict'
t = open(p).read()
t, k = re.subn(r'accelerationRelaxation\s+table\s*\((?:[^()]|\([^()]*\))*\)\s*;', 'accelerationRelaxation 0.7;', t, flags=re.S)
open(p, 'w').write(t)
p = d + '/system/controlDict'
c = open(p).read()
dt = float(re.search(r'^deltaT\s+([^;]+);', c, re.M).group(1))
c, n1 = re.subn(r'^(endTime\s+)[^;]*;', r'\g<1>%.12g;' % (5*dt), c, flags=re.M)
c, n2 = re.subn(r'^(writeInterval\s+)[^;]*;', r'\g<1>5;', c, flags=re.M)
open(p, 'w').write(c)
sys.exit(0 if (k, n1, n2) == (1, 1, 1) else 1)
EOF_X1
    grep -qE '^\s*default\s+CrankNicolson 0\.5;' "$d/system/fvSchemes" \
        && say "fixture witnesses: floatingObject ships \`CrankNicolson 0.5\`" ok \
        || say "fixture witnesses: floatingObject ships \`CrankNicolson 0.5\`" FAIL
    runof "$d"
    xt=$(echo $(timedirs "$d"))
    [ "$(echo $xt | wc -w)" = 1 ] && [ -f "$d/$xt/Uf_0" ] \
        && say "fixture witnesses: OpenFOAM's released body writes one time, with Uf_0 in it" ok \
        || say "fixture witnesses: OpenFOAM's released body writes one time, with Uf_0 in it" FAIL
}
