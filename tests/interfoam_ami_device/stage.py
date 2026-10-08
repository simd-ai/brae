# stage.py <case>: the staging tests/interfoam_ami_vs_openfoam.sh applies, for N fixed steps of DT with one
# write at the end -- p_rgh (and pcorr, which takes $p_rgh) GAMG -> PCG with DIC, every solve pinned.
import os
import re
import sys

d = sys.argv[1]
n = int(os.environ['N'])
dt = os.environ['DT']
t = 0.0
for i in range(n):
    t += float(dt)
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % t),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '18')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
v = os.path.join(d, 'system/fvSolution')
s = open(v).read()
s, k = re.subn(r'(\n    p_rgh\s*\{\s*solver\s+)GAMG;(\s*tolerance[^;]*;\s*relTol[^;]*;\s*)smoother\s+GaussSeidel;',
               r'\1PCG;\2preconditioner  DIC;', s)
assert k == 1, 'p_rgh GAMG'
for pat, val in [(r'(\n    p_rgh\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-13'), (r'(\n    p_rgh\s*\{[^}]*?relTol\s+)[^;]+;', '0'),
                 (r'("pcorr\.\*"\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-13'),
                 (r'("\(U\|T\|k\|epsilon\)\.\*"\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-13')]:
    s, k = re.subn(pat, r'\g<1>' + val + ';', s)
    assert k == 1, pat
open(v, 'w').write(s)
