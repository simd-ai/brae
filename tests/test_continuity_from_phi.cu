// OpenFOAM's continuityErrs.H, recomputed from a WRITTEN phi with brae's host fvc::div -- the oracle
// tests/mirror_continuity.sh holds the mirror's printed line to.
//     contErr       = fvc::div(phi)
//     sum local     = deltaT * weightedAverage(|contErr|, V) = deltaT * sum(|contErr_c| V_c) / sum(V_c)
//     global        = deltaT * weightedAverage(contErr, V)
// Run: test_continuity_from_phi <caseDir> <timeDir> <deltaT>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    if (argc < 4) { std::printf("usage: %s <caseDir> <timeDir> <deltaT>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar deltaT = static_cast<scalar>(std::atof(argv[3]));
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const FieldData<scalar> pf = readField<scalar>(caseDir + "/" + t + "/phi");
    SurfaceScalarField phi;
    phi.internal = pf.internalField;
    phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        phi.boundary[pi].assign(patches[pi].size, 0.0);   // an empty patch has no values: zero, as brae stores it
        for (const auto& b : pf.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size)
                phi.boundary[pi] = b.values;
    }
    const std::vector<scalar> d = fvc::div(phi, m, g, patches);
    double sumV = 0, sumAbs = 0, sumSigned = 0;
    for (label c = 0; c < m.nCells(); ++c)
    {
        const double rv = static_cast<double>(d[c]) * static_cast<double>(g.V()[c]);
        sumV += g.V()[c];
        sumAbs += std::fabs(rv);
        sumSigned += rv;
    }
    std::printf("sum local = %.12g, global = %.12g\n", deltaT * sumAbs / sumV, deltaT * sumSigned / sumV);
    return 0;
}
