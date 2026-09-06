// cellLimited Gauss linear <k>, host reference against the DEVICE implementation.
//
// deviceCellLimitGrad is the already-OpenFOAM-validated one (it is what the shipped solver runs for
// `grad(U) cellLimited ...`), so agreeing with it to machine precision is what says the new host port is
// the same scheme rather than a plausible neighbour of it. Both the scalar and the vector form are
// checked, because the vector form is where the limiter stops being a scalar: it is one limiter PER
// COMPONENT, scaling COLUMN j of grad(U) since OpenFOAM's grad(U)_ij is d(U_j)/d(x_i). Getting that
// transposed is the mistake this test exists to catch, so a transposed control is run too.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <time>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    int fails = 0;
    auto check = [&](const char* what, scalar err, scalar tol)
    {
        const bool ok = err < tol;
        std::printf("  %-34s %.3e   tol %.1e   %s\n", what, err, tol, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    };

    const scalar k = 1.0;

    // ---- scalar field ----
    {
        GeometricField<scalar> p =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
        p.evaluateBoundary();

        std::vector<vector> gh = fvc::gaussGrad(p, m, g, fvp);
        const std::vector<vector> unlimited = gh;
        cpu::cellLimitGrad(gh, p, k, m, g, fvp);

        const DeviceBoundary dbP = buildDeviceBoundary(p, fvp, g);
        DeviceBuffer<scalar> dP(p.internal), pbv, gx, gy, gz;
        deviceBCValue(dbP, dP, pbv);
        deviceGaussGrad(dm, dP, pbv, gx, gy, gz);
        deviceCellLimitGrad(dm, dP, pbv, gx, gy, gz, k);
        const std::vector<scalar> hx = gx.host(), hy = gy.host(), hz = gz.host();

        scalar mx = 0, mag0 = 0, moved = 0;
        for (label c = 0; c < nC; ++c)
        {
            mx = std::fmax(mx, mag(gh[c] - vector{hx[c], hy[c], hz[c]}));
            mag0 = std::fmax(mag0, mag(gh[c]));
            moved = std::fmax(moved, mag(gh[c] - unlimited[c]));
        }
        check("scalar grad(p) host vs device", mag0 > 0 ? mx / mag0 : mx, 1e-12);
        // A limiter that never fires would make this test vacuous.
        std::printf("  %-34s %.3e (must be > 0)\n", "  ...and the limiter DID bite:", moved);
        if (!(moved > 0)) ++fails;
    }

    // ---- vector field ----
    {
        GeometricField<vector> U =
            buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
        U.evaluateBoundary();

        std::vector<tensor> gh = fvc::gaussGrad(U, m, g, fvp);
        cpu::cellLimitGrad(gh, U, k, m, g, fvp);

        const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        {
            ux[c] = U.internal[c].x;
            uy[c] = U.internal[c].y;
            uz[c] = U.internal[c].z;
        }
        DeviceBuffer<scalar> dU[3] = {DeviceBuffer<scalar>(ux), DeviceBuffer<scalar>(uy),
                                     DeviceBuffer<scalar>(uz)};
        scalar mx = 0, mag0 = 0;
        for (int cmpt = 0; cmpt < 3; ++cmpt)
        {
            DeviceBuffer<scalar> bv, gx, gy, gz;
            deviceBCValue(dbU.comp[cmpt], dU[cmpt], bv);
            deviceGaussGrad(dm, dU[cmpt], bv, gx, gy, gz);
            deviceCellLimitGrad(dm, dU[cmpt], bv, gx, gy, gz, k);
            const std::vector<scalar> hx = gx.host(), hy = gy.host(), hz = gz.host();
            for (label c = 0; c < nC; ++c)
            {
                // Column cmpt of the host tensor is grad(U_cmpt).
                const scalar* T = &gh[c].xx;
                const vector hostCol{T[0 * 3 + cmpt], T[1 * 3 + cmpt], T[2 * 3 + cmpt]};
                mx = std::fmax(mx, mag(hostCol - vector{hx[c], hy[c], hz[c]}));
                mag0 = std::fmax(mag0, mag(hostCol));
            }
        }
        check("vector grad(U) host vs device", mag0 > 0 ? mx / mag0 : mx, 1e-12);

        // Transposed control: reading the limiter as a ROW instead of a column must NOT agree.
        scalar tmx = 0, tmag = 0;
        for (int cmpt = 0; cmpt < 3; ++cmpt)
        {
            DeviceBuffer<scalar> bv, gx, gy, gz;
            deviceBCValue(dbU.comp[cmpt], dU[cmpt], bv);
            deviceGaussGrad(dm, dU[cmpt], bv, gx, gy, gz);
            deviceCellLimitGrad(dm, dU[cmpt], bv, gx, gy, gz, k);
            const std::vector<scalar> hx = gx.host(), hy = gy.host(), hz = gz.host();
            for (label c = 0; c < nC; ++c)
            {
                const scalar* T = &gh[c].xx;
                const vector hostRow{T[cmpt * 3 + 0], T[cmpt * 3 + 1], T[cmpt * 3 + 2]};
                tmx = std::fmax(tmx, mag(hostRow - vector{hx[c], hy[c], hz[c]}));
                tmag = std::fmax(tmag, mag(hostRow));
            }
        }
        const scalar trel = tmag > 0 ? tmx / tmag : tmx;
        std::printf("  %-34s %.3e (must be >> 0)\n", "  transposed control disagrees:", trel);
        if (!(trel > 1e-6)) ++fails;
    }

    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
