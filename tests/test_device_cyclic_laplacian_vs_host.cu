// The DEVICE's cyclic-interface laplacian coefficients against the HOST's fvm::laplacian, at identical
// inputs, on a real periodic mesh.
//
// This is the first module of interFoam's cyclic path and it depends on nothing else in that loop: the
// pressure equation's interface coupling is a matrix question, answerable against the host reference
// before any field is solved. The host is the oracle -- fvm.cuh's coupled branch, which OpenFOAM's own
// gaussLaplacianScheme puts as
//     internalCoeffs = boundaryCoeffs = -(gamma_b*magSf_b) * dc_b
// with dc_b the patch's nonOrthDeltaCoeffs under `corrected` and its deltaCoeffs otherwise, and gamma_b
// the SURFACE field's value on the patch. The device's twin (laplKernel, device_cyclic.cu:16-34) builds
// its own face gamma from the two CELLS and carries one set of deltaCoeffs.
//
// WHAT THIS GATE ASSERTS, per face of the pair: the device's ifCoeff is the host's coefficient, and the
// diagonal it writes is the host's diagonal contribution. WHAT IT DOES NOT: anything solved. A matching
// matrix is the precondition for the rest of the path, not a substitute for it.
//
// IT RUNS ON TWO MESHES, and the second is why. validation/cyclicChannel's pair is axis-aligned, so its
// nonOrthDeltaCoeffs and its deltaCoeffs are the SAME NUMBER (spread 0.0) and the two passes below
// assert identical arithmetic -- the `corrected` arm could not see a device that ignored the flag.
// validation/cyclicChannelSkew is the same channel sheared into a parallelogram, so the periodic faces
// tilt while still matching by the pure translation (1 0 0): the spread is 7.7e-01 there, and it caught
// the defect at once. DeviceCyclic carried CyclicInterface::deltaCoeffs, which IS OpenFOAM's
// nonOrthDeltaCoeffs, and used it for both -- the ORTHOGONAL laplacian's interface coefficient was
// 4.2e-03 out of 5.5e-02, 7.7% of it, while the corrected one was exact. It now takes the host patch's
// own plain deltaCoeffs when the scheme does not correct. MEASURED after that, both meshes, both
// passes: 1.4e-17 or better on coefficients up to 5.9e-02.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_field.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"
#include "geometric_field.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
}   // namespace

int main(int argc, char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicChannel";
    std::printf("== the device's cyclic laplacian coefficients against the host's fvm::laplacian ==\n");

    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    // the host interFoam loop's own coupling, and the one the device path has to match
    attachCyclicCoupling(fvp, m, g);

    std::size_t nCoupledFaces = 0;
    for (const FvPatch& q : fvp)
    {
        if (q.coupled) nCoupledFaces += static_cast<std::size_t>(q.size);
    }
    std::printf("  mesh: %d cells, %d internal faces, %zu coupled faces\n",
                (int)m.nCells(), (int)m.nInternalFaces(), nCoupledFaces);
    check("the fixture HAS a coupled pair, so the arms below are not comparing two empty lists",
          nCoupledFaces > 0);
    if (!nCoupledFaces) { std::printf("test_device_cyclic_laplacian_vs_host: %d failures\n", failures); return 1; }

    const label nC = m.nCells();

    // gamma: a NON-UNIFORM cell field, so an interpolation that weights the two cells differently from
    // the host's cannot hide. interFoam's is interpolate(rAU), which varies cell to cell.
    std::vector<scalar> gammaCell(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        gammaCell[static_cast<std::size_t>(c)] =
            scalar(0.3) + scalar(0.2)*std::sin(scalar(1.7)*C.x) + scalar(0.1)*std::cos(scalar(2.3)*C.y);
    }
    const SurfaceScalarField gammaf = fvc::interpolate(gammaCell, m, g, fvp);

    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
    // a field to hang the matrix on, with the cyclic patches coupled to their neighbour cells; its
    // VALUES do not enter the coefficients under test, only its patch types do
    const GeometricField<scalar> vf = buildCyclicField<scalar>(gammaCell, fvp, cyclics);
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceBuffer<scalar> dGammaCell(gammaCell);

    // CAN THIS FIXTURE TELL THE TWO PASSES APART? The `corrected` arm exists to catch a device that
    // carries one set of delta coefficients where the host switches to nonOrthDeltaCoeffs, and on an
    // ORTHOGONAL mesh the two are the same number, so the arm would pass whatever the device did.
    scalar dcSpread = 0, dcScale = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (!fvp[pi].coupled) continue;
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(i);
            dcSpread = std::fmax(dcSpread, std::fabs(fvp[pi].nonOrthDeltaCoeffs[k] - fvp[pi].deltaCoeffs[k]));
            dcScale  = std::fmax(dcScale, std::fabs(fvp[pi].deltaCoeffs[k]));
        }
    }
    std::printf("  the pair's nonOrthDeltaCoeffs differ from its deltaCoeffs by up to %.4e (deltaCoeffs "
                "up to %.4e)\n", (double)dcSpread, (double)dcScale);
    const bool nonOrthLive = dcSpread > scalar(1e-12)*std::fmax(dcScale, scalar(1e-300));
    if (!nonOrthLive)
    {
        std::printf("  NOT DISCRIMINATED on this mesh: the pair is orthogonal, so the `corrected` pass "
                    "below asserts the same arithmetic as the orthogonal one and cannot see a device "
                    "that ignores the flag. Run it on a skewed periodic mesh to close that.\n");
    }

    for (int pass = 0; pass < 2; ++pass)
    {
        const bool corrected = (pass == 1);
        const char* name = corrected ? "corrected" : "orthogonal";

        const FvScalarMatrix host = fvm::laplacian<scalar>(gammaf, vf, m, g, fvp, corrected);

        DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);
        std::vector<scalar> gfInt(gammaf.internal.begin(), gammaf.internal.begin() + m.nInternalFaces());
        DeviceBuffer<scalar> dGf(gfInt), diag, upper, lower;
        deviceLaplacianCoeffs(dm, dGf, diag, upper, lower, corrected);
        std::vector<scalar> diagBefore;
        diag.copyTo(diagBefore);
        deviceCyclicAssembleLaplacian(cyc, dGammaCell, diag, /*addToDiag=*/true, corrected);

        std::vector<scalar> ifCoeff, diagAfter;
        cyc.ifCoeff.copyTo(ifCoeff);
        diag.copyTo(diagAfter);

        // the host's coefficient on the same faces, in the same order buildDeviceCyclic laid them out
        std::vector<scalar> hostCoeff;
        std::vector<label>  hostOwner;
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& P = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                // -boundaryCoeffs, because the host stores the coupled coefficient with the sign the
                // fvMatrix adds to the diagonal and the device stores the one deviceAmul multiplies by
                hostCoeff.push_back(-host.boundaryCoeffs[c.patch][i]);
                hostOwner.push_back(P.faceCells[i]);
            }
        }
        check("the device laid out as many interface faces as the host has coupled ones",
              hostCoeff.size() == ifCoeff.size());
        if (hostCoeff.size() != ifCoeff.size()) continue;

        scalar worst = 0, scale = 0;
        std::size_t worstAt = 0;
        for (std::size_t j = 0; j < ifCoeff.size(); ++j)
        {
            const scalar d = std::fabs(ifCoeff[j] - hostCoeff[j]);
            if (d > worst) { worst = d; worstAt = j; }
            scale = std::fmax(scale, std::fabs(hostCoeff[j]));
        }
        std::printf("  %s: worst |device - host| over %zu interface faces %.4e (coefficients up to "
                    "%.4e, face %zu)\n", name, ifCoeff.size(), (double)worst, (double)scale, worstAt);
        check("the device's interface coefficient IS the host's", worst <= scalar(1e-14)*std::fmax(scale, scalar(1e-300)));

        // ...and the diagonal it wrote: the host's diagonal gains internalCoeffs on a coupled patch
        std::vector<scalar> hostDiagAdd(static_cast<std::size_t>(nC), scalar(0));
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (!fvp[pi].coupled) continue;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                hostDiagAdd[static_cast<std::size_t>(fvp[pi].faceCells[i])] +=
                    host.internalCoeffs[pi][static_cast<std::size_t>(i)];
            }
        }
        scalar worstDiag = 0, diagScale = 0;
        for (label c = 0; c < nC; ++c)
        {
            const std::size_t k = static_cast<std::size_t>(c);
            worstDiag = std::fmax(worstDiag, std::fabs((diagAfter[k] - diagBefore[k]) - hostDiagAdd[k]));
            diagScale = std::fmax(diagScale, std::fabs(hostDiagAdd[k]));
        }
        std::printf("  %s: worst diagonal difference %.4e (contributions up to %.4e)\n",
                    name, (double)worstDiag, (double)diagScale);
        check("...and the diagonal it writes is the host's coupled internalCoeffs",
              worstDiag <= scalar(1e-14)*std::fmax(diagScale, scalar(1e-300)));
    }

    std::printf("test_device_cyclic_laplacian_vs_host: %d failures\n", failures);
    return failures ? 1 : 0;
}
