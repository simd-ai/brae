// interFoam's DEVICE PRESSURE OPERATOR across a cyclic pair: the assembled matrix applied to a field,
// against the host's own, and against the property a periodic Poisson operator has to have.
//
// The three units before this gated the interface COEFFICIENTS (laplacian, momentum, flux/div/grad).
// This one is where they have to meet something that solves: deviceInterAssemblePEqn now adds the
// pair's laplacian coefficients to the matrix, and the pressure solve's LDU view carries the
// off-diagonal so that every SpMV inside the Krylov loop applies it (device_ldu.cuh:28-33). Assembling
// the coupling and then solving without it is not a slower answer, it is a different operator.
//
// TWO ARMS, because they fail differently:
//   (a) A.psi against the HOST's A.psi, the host matrix being fvm::laplacian's (already gated face by
//       face) plus its coupled boundaryCoeffs applied as OpenFOAM applies them -- result[faceCell] -=
//       boundaryCoeffs*psi_neighbour (fvm.cuh:92).
//   (b) THE ROW SUM, which catches something arm (a) states less sharply: the two halves of the
//       coupling being out of step. A Laplacian's rows sum to zero, and they still do when the pair is
//       left out ENTIRELY -- a mesh with a wall there is also a Laplacian -- so this arm does NOT see
//       that case; MEASURED, dropping the interface from the matrix leaves the row sums at 3.5e-17
//       while arm (a) goes to 5.2e-02. What it does see is a diagonal added without its off-diagonal,
//       or the reverse: 5.4e-02, measured. Both arms are here because neither covers the other.
//   (a2) THE FLUX the solved pressure leaves on the pair, which is what pEqn.H:56 subtracts from
//       phiHbyA. On a coupled patch fvMatrix::flux() is internalCoeffs*pif - boundaryCoeffs*pnf
//       (fvMatrix.C:1483-1512), the same face flux an internal face gives. Device against host:
//       1.4e-17 of fluxes up to 5.6e-02, both meshes, both schemes -- and the two schemes differ
//       (5.16e-02 against 5.55e-02 on the skewed pair), so the flag is live here too. BROKEN once,
//       the pressure gradient taken the other way round: 1.0e-01 to 1.1e-01, twice the flux itself.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_field.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"
#include "device_ldu.cuh"
#include "device_inter_peqn.cuh"
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
    std::printf("== interFoam's device pressure operator across a cyclic pair ==\n");

    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    attachCyclicCoupling(fvp, m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);

    const label nC = m.nCells(), nIf = m.nInternalFaces();
    std::size_t nCoupled = 0;
    for (const FvPatch& q : fvp) if (q.coupled) nCoupled += static_cast<std::size_t>(q.size);
    std::printf("  mesh: %d cells, %d internal faces, %zu coupled faces\n", (int)nC, (int)nIf, nCoupled);
    check("the fixture has a coupled pair", nCoupled > 0);
    if (!nCoupled) { std::printf("test_device_inter_peqn_cyclic_vs_host: %d failures\n", failures); return 1; }

    // rAU as interFoam has it: a cell field, varying, and the pressure laplacian's gamma
    std::vector<scalar> rAU(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        rAU[static_cast<std::size_t>(c)] = scalar(0.4) + scalar(0.15)*std::sin(scalar(1.9)*C.x + C.y);
    }
    const SurfaceScalarField rAUf = fvc::interpolate(rAU, m, g, fvp);
    const GeometricField<scalar> pf = buildCyclicField<scalar>(rAU, fvp, cyclics);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceBuffer<scalar> dRAU(rAU);
    std::vector<scalar> rAUfInt(rAUf.internal.begin(), rAUf.internal.begin() + nIf);
    DeviceBuffer<scalar> dRAUf(rAUfInt);
    DeviceBuffer<scalar> zeroIf(std::vector<scalar>(static_cast<std::size_t>(nIf), scalar(0)));
    std::size_t nBf = 0;
    for (const FvPatch& q : fvp)
    {
        if (!isCoupledInterfaceType(q.type)) nBf += static_cast<std::size_t>(q.size);
    }
    DeviceBuffer<scalar> zeroBf(std::vector<scalar>(nBf, scalar(0)));
    // ...and the PAIR's phiHbyA, which the assembly now requires because fvc::div(phiHbyA) sums a
    // coupled face like any other. This gate's phiHbyA is zero everywhere, so the pair's is too.
    std::size_t nIfFaces = 0;
    for (const CyclicInterface& c : cyclics) nIfFaces += c.faceCells.size();
    DeviceBuffer<scalar> zeroCycIf(std::vector<scalar>(nIfFaces, scalar(0)));

    for (int pass = 0; pass < 2; ++pass)
    {
        const bool corrected = (pass == 1);
        const char* name = corrected ? "corrected" : "orthogonal";

        DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);
        DevicePressureMatrix P;
        deviceInterAssemblePEqn(dm, dRAUf, zeroIf, zeroBf, /*needReference=*/false, 0, nullptr, P,
                                corrected, nullptr, &cyc, &dRAU, &zeroCycIf);

        // psi: something that is NOT periodic-symmetric, so the interface term cannot cancel itself
        std::vector<scalar> psi(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            const vector& C = g.C()[c];
            psi[static_cast<std::size_t>(c)] = std::sin(scalar(2.3)*C.x)*std::cos(scalar(1.7)*C.y)
                                             + scalar(0.4)*C.x;
        }
        DeviceBuffer<scalar> dPsi(psi), dApsi(static_cast<std::size_t>(nC));
        const DeviceLduView A = deviceLduViewCyclic(dm, P.diag, P.upper, P.lower, cyc.n,
                                                    cyc.ownCell.data(), cyc.nbrCell.data(),
                                                    cyc.ifCoeff.data());
        deviceAmul(A, dPsi, dApsi);
        std::vector<scalar> devApsi;
        dApsi.copyTo(devApsi);

        // ---- (a) the host's own operator -----------------------------------------------------------
        const FvScalarMatrix host = fvm::laplacian<scalar>(rAUf, pf, m, g, fvp, corrected);
        std::vector<scalar> hostApsi(static_cast<std::size_t>(nC), scalar(0));
        for (label c = 0; c < nC; ++c)
        {
            hostApsi[static_cast<std::size_t>(c)] = host.diag[static_cast<std::size_t>(c)]
                                                  * psi[static_cast<std::size_t>(c)];
        }
        for (label f = 0; f < nIf; ++f)
        {
            const label o = m.owner()[static_cast<std::size_t>(f)];
            const label n = m.neighbour()[static_cast<std::size_t>(f)];
            hostApsi[static_cast<std::size_t>(o)] += host.upper[static_cast<std::size_t>(f)]
                                                   * psi[static_cast<std::size_t>(n)];
            hostApsi[static_cast<std::size_t>(n)] += host.lower[static_cast<std::size_t>(f)]
                                                   * psi[static_cast<std::size_t>(o)];
        }
        // the coupled patches, as OpenFOAM's updateInterfaceMatrix applies them: the face cell loses
        // boundaryCoeffs times the value on the OTHER side (fvm.cuh:88-95)
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& Pp = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                hostApsi[static_cast<std::size_t>(Pp.faceCells[i])] -=
                    host.boundaryCoeffs[c.patch][i] * psi[static_cast<std::size_t>(c.nbrFaceCells[i])];
            }
        }
        // ...and the diagonal the host keeps in internalCoeffs on a coupled patch
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& Pp = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                const std::size_t k = static_cast<std::size_t>(Pp.faceCells[i]);
                hostApsi[k] += host.internalCoeffs[c.patch][i] * psi[k];
            }
        }
        scalar worst = 0, scale = 0;
        for (label c = 0; c < nC; ++c)
        {
            const std::size_t k = static_cast<std::size_t>(c);
            worst = std::fmax(worst, std::fabs(devApsi[k] - hostApsi[k]));
            scale = std::fmax(scale, std::fabs(hostApsi[k]));
        }
        std::printf("  %s: worst |A.psi device - host| %.4e (|A.psi| up to %.4e)\n",
                    name, (double)worst, (double)scale);
        check("the device's assembled pressure operator IS the host's, interface included",
              worst <= scalar(1e-13)*std::fmax(scale, scalar(1e-300)));

        // ---- (a2) the FLUX the solved pressure leaves on the pair ---------------------------------
        // pEqn.H:56 writes phi = phiHbyA - p_rghEqn.flux(), and on a coupled patch fvMatrix::flux() is
        // internalCoeffs*pif - boundaryCoeffs*pnf (fvMatrix.C:1483-1512, fv_matrix_ops.cuh:28-31) --
        // the same face flux an internal face gives. The device subtracts ifCoeff*(p_nbr - p_own) from
        // the pair's flux, which is that expression with the laplacian's own coefficient.
        {
            std::vector<scalar> phiIf(static_cast<std::size_t>(cyc.n), scalar(0));
            DeviceCyclic cycF = buildDeviceCyclic(cyclics, g, fvp);
            DevicePressureMatrix PF;
            deviceInterAssemblePEqn(dm, dRAUf, zeroIf, zeroBf, /*needReference=*/false, 0, nullptr, PF,
                                    corrected, nullptr, &cycF, &dRAU, &zeroCycIf);
            cycF.phi.copyFrom(phiIf);          // start from zero, so what is left IS the flux
            DeviceBuffer<scalar> dP(psi);      // psi stands in for the solved p_rgh
            deviceCyclicCorrectFlux(cycF, dP);
            std::vector<scalar> devFlux;
            cycF.phi.copyTo(devFlux);

            std::vector<scalar> hostFlux;
            for (const CyclicInterface& c : cyclics)
            {
                const FvPatch& Pp = fvp[c.patch];
                for (std::size_t i = 0; i < c.faceCells.size(); ++i)
                {
                    const scalar pif = psi[static_cast<std::size_t>(Pp.faceCells[i])];
                    const scalar pnf = psi[static_cast<std::size_t>(c.nbrFaceCells[i])];
                    hostFlux.push_back(-(host.internalCoeffs[c.patch][i]*pif
                                       - host.boundaryCoeffs[c.patch][i]*pnf));
                }
            }
            scalar worstF = 0, scaleF = 0;
            for (std::size_t j = 0; j < devFlux.size() && j < hostFlux.size(); ++j)
            {
                worstF = std::fmax(worstF, std::fabs(devFlux[j] - hostFlux[j]));
                scaleF = std::fmax(scaleF, std::fabs(hostFlux[j]));
            }
            std::printf("  %s: worst |flux device - host| %.4e (fluxes up to %.4e)\n",
                        name, (double)worstF, (double)scaleF);
            check("the flux the solved pressure leaves on the pair IS the host's",
                  devFlux.size() == hostFlux.size()
                  && worstF <= scalar(1e-14)*std::fmax(scaleF, scalar(1e-300)));
            check("...and it is not identically zero, which a pair left out would be",
                  scaleF > scalar(0));
        }

        // ---- (b) the row sum ------------------------------------------------------------------------
        std::vector<scalar> ones(static_cast<std::size_t>(nC), scalar(1));
        DeviceBuffer<scalar> dOnes(ones), dRow(static_cast<std::size_t>(nC));
        deviceAmul(A, dOnes, dRow);
        std::vector<scalar> rowSum;
        dRow.copyTo(rowSum);
        scalar worstRow = 0, diagScale = 0;
        std::vector<scalar> diagH;
        P.diag.copyTo(diagH);
        for (label c = 0; c < nC; ++c)
        {
            worstRow = std::fmax(worstRow, std::fabs(rowSum[static_cast<std::size_t>(c)]));
            diagScale = std::fmax(diagScale, std::fabs(diagH[static_cast<std::size_t>(c)]));
        }
        std::printf("  %s: worst row sum %.4e (diagonal up to %.4e)\n",
                    name, (double)worstRow, (double)diagScale);
        check("a constant field is in the operator's null space, so the pair's diagonal and its "
              "off-diagonal are in step",
              worstRow <= scalar(1e-14)*std::fmax(diagScale, scalar(1e-300)));
    }

    std::printf("test_device_inter_peqn_cyclic_vs_host: %d failures\n", failures);
    return failures ? 1 : 0;
}
