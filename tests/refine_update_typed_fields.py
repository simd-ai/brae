import sys
case = sys.argv[1]
ncells = int(sys.argv[2])

def header(cls, obj, dims):
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n"
            "    location    \"0\";\n    object      %s;\n}\n\ndimensions      %s;\n\n" % (cls, obj, dims))

def internal_scalar():
    # the cell INDEX, so a wall function's patch value (the cell value) differs face by face and a
    # comparison of it cannot pass on a constant
    return "internalField   nonuniform List<scalar>\n%d\n(\n%s\n)\n;\n\n" % (
        ncells, "\n".join(str(i) for i in range(ncells)))

def internal_vector():
    return "internalField   nonuniform List<vector>\n%d\n(\n%s\n)\n;\n\n" % (
        ncells, "\n".join("(%d %d %d)" % (i, 2*i, 3*i) for i in range(ncells)))

fields = {
 'braeNut': (header('volScalarField','braeNut','[0 2 -1 0 0 0 0]') + internal_scalar() +
   'boundaryField\n{\n    atmosphere\n    {\n        type            zeroGradient;\n    }\n'
   '    walls\n    {\n        type            nutkWallFunction;\n        value           uniform 0;\n    }\n}\n'),
 'braeOmega': (header('volScalarField','braeOmega','[0 0 -1 0 0 0 0]') + internal_scalar() +
   'boundaryField\n{\n    atmosphere\n    {\n        type            inletOutlet;\n'
   '        inletValue      uniform 1;\n        value           uniform 1;\n    }\n'
   '    walls\n    {\n        type            omegaWallFunction;\n        value           uniform 1;\n    }\n}\n'),
 'braeKq': (header('volScalarField','braeKq','[0 2 -2 0 0 0 0]') + internal_scalar() +
   'boundaryField\n{\n    atmosphere\n    {\n        type            fixedValue;\n        value           uniform 3;\n    }\n'
   '    walls\n    {\n        type            kqRWallFunction;\n        value           uniform 0;\n    }\n}\n'),
 'braeUwall': (header('volVectorField','braeUwall','[0 1 -1 0 0 0 0]') + internal_vector() +
   'boundaryField\n{\n    atmosphere\n    {\n        type            pressureInletOutletVelocity;\n'
   '        value           uniform (0 0 0);\n    }\n'
   '    walls\n    {\n        type            movingWallVelocity;\n        value           uniform (0 0 0);\n    }\n}\n'),
}
for name, text in fields.items():
    open(case + '/0/' + name, 'w').write(text)
print("wrote %d typed fields over %d cells" % (len(fields), ncells))
