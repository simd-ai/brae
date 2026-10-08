import re
import sys

# Four fields carrying the patch types the adaptive cases need -- nutkWallFunction, omegaWallFunction,
# kqRWallFunction and movingWallVelocity -- whose internal fields are the CELL INDEX, so a wall function's
# patch value (its cell's value) differs face by face and a comparison of it cannot pass on a constant.
#
# THE PATCH ENTRIES COME FROM THE MESH, not from one tutorial's names. They were hardcoded to
# damBreakWithObstacle's `walls` and `atmosphere`, so pointing the harness at any other case stopped at
# OpenFOAM's own "Cannot find patchField entry for leftWall" -- which is what a 2-D fixture needs, since a
# 2-D adaptive case is the one that has an `empty` patch and the one whose mapping is not yet gated.
#
# The rule per patch, from its TYPE in constant/polyMesh/boundary:
#   empty  -> `empty`, the only entry OpenFOAM accepts there
#   wall   -> the wall-function type, which is what makes the per-face comparison bite
#   else   -> the open-boundary type
case = sys.argv[1]
ncells = int(sys.argv[2])


def header(cls, obj, dims):
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n"
            "    location    \"0\";\n    object      %s;\n}\n\ndimensions      %s;\n\n" % (cls, obj, dims))


def internal_scalar():
    return "internalField   nonuniform List<scalar>\n%d\n(\n%s\n)\n;\n\n" % (
        ncells, "\n".join(str(i) for i in range(ncells)))


def internal_vector():
    return "internalField   nonuniform List<vector>\n%d\n(\n%s\n)\n;\n\n" % (
        ncells, "\n".join("(%d %d %d)" % (i, 2 * i, 3 * i) for i in range(ncells)))


def patches(meshdir):
    """(name, type) for every patch, in the file's own order."""
    text = open(meshdir + '/boundary').read()
    # strip the FoamFile block, then take every `name { ... type X; ... }`
    body = text[text.find('}', text.find('FoamFile')) + 1:]
    out = []
    for m in re.finditer(r'^\s*([A-Za-z_][\w.-]*)\s*\n\s*\{(.*?)\n\s*\}', body, re.S | re.M):
        name, block = m.group(1), m.group(2)
        t = re.search(r'\btype\s+(\w+)\s*;', block)
        if t:
            out.append((name, t.group(1)))
    return out


def boundary_field(pts, wall_entry, open_entry):
    lines = ['boundaryField\n{\n']
    for name, ptype in pts:
        if ptype == 'empty':
            body = '        type            empty;\n'
        elif ptype == 'wall':
            body = wall_entry
        else:
            body = open_entry
        lines.append('    %s\n    {\n%s    }\n' % (name, body))
    lines.append('}\n')
    return ''.join(lines)


pts = patches(case + '/constant/polyMesh')
if not pts:
    raise SystemExit('refine_update_typed_fields: no patches found in %s/constant/polyMesh/boundary' % case)

fields = {
    'braeNut': (header('volScalarField', 'braeNut', '[0 2 -1 0 0 0 0]') + internal_scalar()
                + boundary_field(pts,
                                 '        type            nutkWallFunction;\n'
                                 '        value           uniform 0;\n',
                                 '        type            zeroGradient;\n')),
    'braeOmega': (header('volScalarField', 'braeOmega', '[0 0 -1 0 0 0 0]') + internal_scalar()
                  + boundary_field(pts,
                                   '        type            omegaWallFunction;\n'
                                   '        value           uniform 1;\n',
                                   '        type            inletOutlet;\n'
                                   '        inletValue      uniform 1;\n'
                                   '        value           uniform 1;\n')),
    'braeKq': (header('volScalarField', 'braeKq', '[0 2 -2 0 0 0 0]') + internal_scalar()
               + boundary_field(pts,
                                '        type            kqRWallFunction;\n'
                                '        value           uniform 0;\n',
                                '        type            fixedValue;\n'
                                '        value           uniform 3;\n')),
    'braeUwall': (header('volVectorField', 'braeUwall', '[0 1 -1 0 0 0 0]') + internal_vector()
                  + boundary_field(pts,
                                   '        type            movingWallVelocity;\n'
                                   '        value           uniform (0 0 0);\n',
                                   '        type            pressureInletOutletVelocity;\n'
                                   '        value           uniform (0 0 0);\n')),
}
for name, text in fields.items():
    open(case + '/0/' + name, 'w').write(text)
print("wrote %d typed fields over %d cells, %d patches (%s)"
      % (len(fields), ncells, len(pts), ", ".join("%s:%s" % p for p in pts)))
