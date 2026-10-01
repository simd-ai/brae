# Shared by the files of this folder.
vbytes()   # vbytes <OpenFOAM case> <brae case> -- arm V's byte check on every waveProperties file
{
    python3 - "$1" "$2" <<'EOF_VB'
import os, re, sys
of, br = sys.argv[1], sys.argv[2]
times = sorted([t for t in os.listdir(of) if re.match(r'^[0-9.e+-]+$', t) and t != '0'], key=float)
n = bad = 0
for t in times:
    for f in sorted(os.listdir('%s/%s/uniform' % (of, t))):
        if not f.startswith('waveProperties.'):
            continue
        n += 1
        so = open('%s/%s/uniform/%s' % (of, t, f)).read()
        try:
            sb = open('%s/%s/uniform/%s' % (br, t, f)).read()
        except OSError:
            sb = ''
        bad += so[so.find('// * * *'):] != sb[sb.find('// * * *'):]
print('      %d files, %d differ' % (n, bad))
sys.exit(0 if n and not bad else 1)
EOF_VB
}
