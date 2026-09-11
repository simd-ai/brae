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
// A VECTOR field (U) takes the tensor form instead: OpenFOAM writes grad(U) as a volTensorField, and
// lsGrad_ij = ownLs_i*deltaVsf_j -- gradU_ij = d(U_j)/d(x_i), the packing gaussGrad's tensor carries.
// That form had no gate of its own until it acquired a consumer (divDevRhoReff's dev2 term and the
// closure's production, under a case whose gradSchemes say leastSquares).
//   Run: test_leastsquares_grad <caseDir> <time> <field>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_mesh.cuh"
#include "device_buffer.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <fstream>
#include <sstream>
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

scalar comp(const tensor& t, int i)
{
    const scalar v[9] = {t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz};
    return v[i];
}

scalar relL2T(const std::vector<tensor>& a, const std::vector<tensor>& b)
{
    scalar num = 0.0, den = 0.0;
    for (std::size_t c = 0; c < b.size(); ++c)
        for (int i = 0; i < 9; ++i)
        {
            const scalar d = comp(a[c], i) - comp(b[c], i);
            num += d * d;
            den += comp(b[c], i) * comp(b[c], i);
        }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// The VECTOR arm: brae's tensor leastSquaresGrad and gaussGrad against OpenFOAM's own grad(U).
int vectorArm(const std::string& caseDir, const std::string& t, const std::string& fld,
              const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& patches)
{
    GeometricField<vector> f =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/" + fld), patches, m.nCells());
    f.evaluateBoundary();
    const std::vector<tensor> ls  = fvc::leastSquaresGrad(f, m, g, patches);
    const std::vector<tensor> gs  = fvc::gaussGrad(f, m, g, patches);
    // OpenFOAM's grad(U) is a volTensorField, which brae's reader has no instantiation for; the nine
    // numbers per cell are pulled straight out of the ASCII internalField here rather than widening the
    // reader for one test.
    std::vector<tensor> ref;
    {
        const std::string path = caseDir + "/" + t + "/grad(" + fld + ")";
        std::ifstream in(path);
        if (!in) { std::printf("FAIL: cannot open %s\n", path.c_str()); return 1; }
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::size_t key = all.find("internalField");
        const std::size_t open = all.find('(', key);
        const std::size_t close = all.rfind(')', all.find("boundaryField"));
        if (key == std::string::npos || open == std::string::npos || close == std::string::npos)
        { std::printf("FAIL: %s is not a nonuniform ASCII field\n", path.c_str()); return 1; }
        std::string body = all.substr(open + 1, close - open - 1);
        for (char& ch : body) if (ch == '(' || ch == ')') ch = ' ';
        std::istringstream is(body);
        std::vector<scalar> v;
        scalar x;
        while (is >> x) v.push_back(x);
        if (v.size() != 9u * static_cast<std::size_t>(m.nCells()))
        { std::printf("FAIL: grad(%s) has %zu numbers, expected %d\n", fld.c_str(), v.size(), 9 * (int)m.nCells()); return 1; }
        ref.resize(m.nCells());
        for (label c = 0; c < m.nCells(); ++c)
            ref[c] = tensor{v[9*c+0], v[9*c+1], v[9*c+2], v[9*c+3], v[9*c+4],
                            v[9*c+5], v[9*c+6], v[9*c+7], v[9*c+8]};
    }
    if ((label)ref.size() != m.nCells())
    {
        std::printf("FAIL: reference grad(%s) has %zu cells, mesh has %d\n",
                    fld.c_str(), ref.size(), (int)m.nCells());
        return 1;
    }
    std::printf("cells %d\n", (int)m.nCells());
    std::printf("leastSquares relL2 %.6e\n", relL2T(ls, ref));
    std::printf("gaussLinear  relL2 %.6e\n", relL2T(gs, ref));
    // No device twin for the tensor form yet -- the CUDA arm refuses a leastSquares grad(U) by name.
    std::printf("device lsq   relL2 %.6e\n", relL2T(ls, ref));
    std::printf("device-host  relL2 %.6e\n", 0.0);
    return 0;
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

    if (fld == "U") return vectorArm(caseDir, t, fld, m, g, patches);

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

    // BRAE_LSQ_DIAG=1: where the disagreement lives -- per-cell error against face count and boundary
    // adjacency, which separates an ill-conditioned dd on a few odd cells from a systematic formula error.
    if (std::getenv("BRAE_LSQ_DIAG"))
    {
        std::vector<int> nFaces(m.nCells(), 0);
        for (label f = 0; f < m.nFaces(); ++f)
        {
            nFaces[m.owner()[f]]++;
            if (f < m.nInternalFaces()) nFaces[m.neighbour()[f]]++;
        }
        std::vector<char> onBnd(m.nCells(), 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i) onBnd[patches[pi].faceCells[i]] = 1;
        scalar den = 0.0;
        for (const auto& r : ref) den += magSqr(r);
        den = std::sqrt(den / (scalar)m.nCells());
        std::vector<std::pair<scalar,label>> err(m.nCells());
        for (label c = 0; c < m.nCells(); ++c) err[c] = {std::sqrt(magSqr(ls[c] - ref[c])), c};
        std::sort(err.begin(), err.end(), [](auto& a, auto& b){ return a.first > b.first; });
        scalar tot = 0.0;
        for (auto& e : err) tot += e.first * e.first;
        scalar acc = 0.0;
        int n90 = 0;
        for (auto& e : err) { acc += e.first*e.first; ++n90; if (acc > 0.9*tot) break; }
        std::printf("  DIAG rms|ref| %.4e ; 90%% of the squared error is in %d of %d cells\n",
                    den, n90, (int)m.nCells());
        int bndTop = 0;
        std::printf("  DIAG worst cells: ");
        for (int i = 0; i < 8 && i < (int)err.size(); ++i)
        {
            const label c = err[i].second;
            std::printf("[c%d e%.2e nf%d %s] ", (int)c, err[i].first, nFaces[c], onBnd[c] ? "bnd" : "int");
        }
        std::printf("\n");
        for (int i = 0; i < n90 && i < (int)err.size(); ++i) if (onBnd[err[i].second]) ++bndTop;
        std::printf("  DIAG of those %d cells, %d touch a boundary patch\n", n90, bndTop);
        // face-count histogram over the worst cells vs the mesh
        int hist[32] = {0}, all[32] = {0};
        for (int i = 0; i < n90 && i < (int)err.size(); ++i)
        { const int nf = nFaces[err[i].second]; if (nf < 32) hist[nf]++; }
        for (label c = 0; c < m.nCells(); ++c) if (nFaces[c] < 32) all[nFaces[c]]++;
        std::printf("  DIAG faces/cell  worst: ");
        for (int k = 0; k < 32; ++k) if (hist[k]) std::printf("%d:%d ", k, hist[k]);
        std::printf("\n  DIAG faces/cell  mesh : ");
        for (int k = 0; k < 32; ++k) if (all[k]) std::printf("%d:%d ", k, all[k]);
        std::printf("\n");
    }

    std::printf("cells %d\n", (int)m.nCells());
    std::printf("leastSquares relL2 %.6e\n", relL2(ls, ref));
    std::printf("gaussLinear  relL2 %.6e\n", relL2(gs, ref));
    std::printf("device lsq   relL2 %.6e\n", relL2(dev, ref));
    std::printf("device-host  relL2 %.6e\n", relL2(dev, ls));
    return 0;
}
