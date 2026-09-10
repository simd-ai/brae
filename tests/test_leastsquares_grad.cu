// fvc::leastSquaresGrad against OpenFOAM's own leastSquares gradient.
//
// OpenFOAM's leastSquares is an inverse-distance-weighted least-squares fit (leastSquaresGrad.C,
// leastSquaresVectors.C), not a Gauss sum, and a case naming it must get it: on validation/rhoLU at a
// developed state, swapping the limitedLinear limiter's gradient from Gauss linear to this moves the
// assembled energy diagonal by 9.1e-03. rhoSimpleFoam's OF-mirror refused such a case by name until now.
//
// Prints BOTH brae gradients against the same OpenFOAM reference, so the caller can assert that the
// least-squares one matches AND that the Gauss one does not -- a gate that only checked the first would
// pass on any mesh where the two schemes happen to agree.
//   Run: test_leastsquares_grad <caseDir> <time> <field>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_mesh.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace brae;

namespace {
scalar relL2(const std::vector<vector>& a, const std::vector<vector>& b)
{
    scalar num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        num += magSqr(a[i] - b[i]);
        den += magSqr(b[i]);
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::printf("usage: %s <caseDir> <time> <field>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2], fld = argv[3];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<scalar> f =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + fld), patches, m.nCells());
    f.evaluateBoundary();

    // The boundary VALUES, in patch order -- the array form both gradients take.
    std::vector<std::vector<scalar>> fb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) fb[pi] = f.boundary[pi]->value();

    const std::vector<vector> ls = fvc::leastSquaresGrad(f.internal, fb, m, g, patches);
    const std::vector<vector> gs = fvc::gaussGrad(f.internal, fb, m, g, patches);
    const std::vector<vector> ref =
        readField<vector>(caseDir + "/" + t + "/grad(" + fld + ")").internalField;

    if ((label)ref.size() != m.nCells())
    { std::printf("FAIL: reference grad(%s) has %zu cells, mesh has %d\n", fld.c_str(), ref.size(), (int)m.nCells()); return 1; }

    // The DEVICE port, against the same OpenFOAM reference. The host arm above is the numerics; this
    // says the CUDA transcription of it agrees, on the same mesh and the same field.
    std::vector<vector> dev;
    {
        const DeviceMesh dm = buildDeviceMesh(m, g, patches);
        DeviceBuffer<scalar> vol, bval, gxd, gyd, gzd;
        vol.copyFrom(f.internal);
        std::vector<scalar> flat;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            flat.insert(flat.end(), fb[pi].begin(), fb[pi].end());
        bval.copyFrom(flat);
        deviceLeastSquaresGrad(dm, vol, bval, gxd, gyd, gzd);
        const std::vector<scalar> hx = gxd.host(), hy = gyd.host(), hz = gzd.host();
        dev.resize(m.nCells());
        for (label c = 0; c < m.nCells(); ++c) dev[c] = vector{hx[c], hy[c], hz[c]};
    }

    std::printf("cells %d\n", (int)m.nCells());
    std::printf("leastSquares relL2 %.6e\n", relL2(ls, ref));
    std::printf("gaussLinear  relL2 %.6e\n", relL2(gs, ref));
    std::printf("device lsq   relL2 %.6e\n", relL2(dev, ref));
    std::printf("device-host  relL2 %.6e\n", relL2(dev, ls));
    return 0;
}
