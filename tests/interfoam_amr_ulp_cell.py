# Move ONE value of a case's initial alpha.water by one ulp, in a cell AT THE INTERFACE.
#
# WHY NOT THE FIRST CELL THAT IS 1, which is what this did: on damBreak's developed state that cell is deep
# in the water, every neighbour is 1 as well, and the perturbation dies where it is made. MEASURED on the
# restart fixture with tools/dumpInterFoam: a one-ulp twin perturbed there moves 12 of 12,487 faces of the
# surface tension force, by 3.5e-21, and its p_rgh after three steps moves 8.5e-11 -- BELOW the 1e-10 the
# gate asks of an amplifying case. Against that envelope brae read 20x to 32x and the gate could assert
# nothing. A cell at the interface is a different measurement: the curvature is a function of alpha's
# GRADIENT, so one ulp only reaches nHatf, K and the surface tension force where alpha varies.
#
# WITH THE PERTURBATION AT THE INTERFACE the same twin moves p_rgh 3.7882e-09 and U 6.8228e-08 after three
# steps, and brae sits at 0.72x and 0.63x of it. The 20x was the control's, not the port's.
#
# THE CELL IS THE OWNER OF THE FIRST INTERNAL FACE whose two cells differ in alpha, so the choice is
# reproducible from the mesh and does not depend on a cell numbering. It FAILS rather than falls back if the
# state has no interface at all, because a twin that cannot witness is worse than no twin.
import math
import re
import sys


def labelList(path):
    t = open(path, 'rb').read(4096).decode('latin-1')
    if 'format' in t and 'binary' in t.split('format', 1)[1][:40]:
        raise SystemExit('interfoam_amr_ulp_cell: %s is binary; this reader is ascii only' % path)
    t = open(path).read()
    body = t[t.find('// * * *'):]
    m = re.search(r'(\d+)\s*\(', body)
    if not m:
        raise SystemExit('interfoam_amr_ulp_cell: no list in %s' % path)
    n = int(m.group(1))
    j = body.find('(', m.start())
    k = body.find(')', j)
    v = [int(x) for x in body[j + 1:k].split()]
    if len(v) != n:
        raise SystemExit('interfoam_amr_ulp_cell: %s says %d and carries %d' % (path, n, len(v)))
    return v


case = sys.argv[1]
field = sys.argv[2] if len(sys.argv) > 2 else 'alpha.water'
# the time directory the field lives in, which for a genuine restart is not `0`
timeDir = sys.argv[3] if len(sys.argv) > 3 else '0'

# THE MESH THE FIELD IS ON, which is the addressing this has to use: a case continued from a refined mesh
# carries the mesh in the time directory and the one it STARTED from in constant/, and pairing the field with
# the wrong one would index a cell that is not the neighbour of anything it names. polyMesh resolves each
# file for itself; here one directory is enough, because the field and its mesh are written together.
import os
meshDir = case + '/' + timeDir + '/polyMesh'
if not os.path.exists(meshDir + '/faces'):
    meshDir = case + '/constant/polyMesh'
own = labelList(meshDir + '/owner')
nei = labelList(meshDir + '/neighbour')

p = case + '/' + timeDir + '/' + field
t = open(p).read()
i = t.find('internalField')
j = t.find('(', i)
k = t.find('\n)', j)
vals = t[j + 1:k].split()
x = [float(v) for v in vals]

cell = None
for f, (o, nb) in enumerate(zip(own, nei)):
    if o < len(x) and nb < len(x) and x[o] != x[nb]:
        # perturb the side that is exactly 1: one ulp of 1 is 1.11e-16 and not a denormal
        cell = o if x[o] == 1.0 else (nb if x[nb] == 1.0 else o)
        break
if cell is None:
    raise SystemExit('interfoam_amr_ulp_cell: no internal face has different %s either side -- this state '
                     'has no interface, so a one-ulp twin of it cannot witness anything' % field)

before = x[cell]
vals[cell] = repr(math.nextafter(before, 0.0) if before != 0.0 else math.nextafter(0.0, 1.0))
open(p, 'w').write(t[:j + 1] + '\n' + '\n'.join(vals) + '\n' + t[k:])
print('  one ulp on cell %d of %s/%s, AT THE INTERFACE: %g -> %s'
      % (cell, timeDir, field, before, vals[cell]))
