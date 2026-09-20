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
//
// THE SCALAR TRANSPORT ASSEMBLER's own call is the last arm: k's and epsilon's equations are the
// momentum's shape, fvm::div(phi, f) - fvm::laplacian(DEff, f), so gpu::turbulence::
// assembleScalarTransport hands the pair the same kernel -- and what that arm gates is the CALL, each
// of whose three arguments is a number that is nearly right. MEASURED, both meshes: interface 2.7e-20
// of 2.0e-02 and diagonal 5.4e-20 of 1.9e-02. BROKEN ONCE: the pair's own convecting flux dropped, so
// the coefficients fall back to cyc.phi -- interface 5.7e-02 against a 2.0e-02 scale, and the arm
// fails. cyc.phi is seeded with a DIFFERENT flux on purpose, which is what makes that witnessable: a
// compressible closure convects k and epsilon with the mass flux and not with the pair's own phi.
//
// THE MOMENTUM MATRIX's interface coefficients are here too, M = fvm::div(phi, U) - fvm::laplacian(nuEff,
// U), with a flux that changes sign over the pair so the upwind split is exercised (9 outflow, 11 inflow
// faces, asserted). Both schemes, both meshes: 2.7e-20 on coefficients up to 2.1e-01. Its diffusion half
// had the SAME defect as the laplacian's and it was fixed the same way -- BROKEN once, with the flag
// ignored: 1.6e-05 on the skewed mesh's orthogonal pass.
//
// THE THREE EXPLICIT OPERATORS the pressure corrector needs across the pair are here too -- fvc::flux,
// fvc::div's boundary sum and fvc::gaussGrad's interface contribution -- each against the host's own
// coupled branch (fvc.cu:50-53, :399-402). div and grad agree to 1.8e-15 and 3.6e-15 of contributions
// up to 8.2e+00 and 2.0e+01. The FLUX is asserted BIT FOR BIT, and the bits are the point: device and
// host differ by 1.7e-18 plain, and all 20 of 20 faces reproduce the device EXACTLY once the host fuses
// the same multiply-adds with std::fma -- so the difference is nvcc's contraction and the arithmetic
// form is identical. That arm is what discriminates: BROKEN once, with the interpolation weight on the
// wrong side, the value bound barely moves (1.9e-17, because this mesh's weights are near 0.5) while
// the bitwise arm falls from 20 of 20 to 2 of 20 and the divergence goes 1.9e-14.
//
// THE UPWIND CONVECTION coefficients are here too, because the alpha equation's implicit pre-solve
// assembles them and a periodic mesh's pair is in neither of its face lists: internalCoeffs = phi*w,
// boundaryCoeffs = -(phi*(1 - w)), w = pos0(phi) (fvm.cuh:536-549). Device against host, both meshes:
// exactly 0.0 on the interface coefficient and on the diagonal, with the flux changing sign over the
// pair so both upwind branches are taken and neither quantity identically zero.
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

    // ---- THE MOMENTUM MATRIX'S INTERFACE COEFFICIENTS ------------------------------------------
    // M = fvm::div(phi, U) - fvm::laplacian(nuEff, U), which is what a segregated momentum equation
    // puts on a coupled patch. The host gives each face
    //     internalCoeffs = phi*w  -  ( -(nuEff_b*magSf_b)*dc_b )
    //     boundaryCoeffs = -(phi*(1 - w))  -  ( -(nuEff_b*magSf_b)*dc_b )
    // with w = pos0(phi), upwind's weight on a coupled patch as on an internal face (fvm.cuh:536-549).
    // The device's twin is momKernel (device_cyclic.cu:59-83), which builds its own face nuEff from the
    // two CELLS and its own split.
    {
        // a flux with BOTH SIGNS on the pair -- an upwind weight that is constant over the interface
        // would let a wrong split through, so the arm asserts both occur
        std::vector<std::vector<scalar>> phiB(fvp.size());
        std::vector<scalar> phiIf(static_cast<std::size_t>(m.nInternalFaces()), scalar(0));
        for (label f = 0; f < m.nInternalFaces(); ++f)
        {
            phiIf[static_cast<std::size_t>(f)] = scalar(0.13)*std::sin(scalar(0.7)*scalar(f));
        }
        std::size_t nPos = 0, nNeg = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            phiB[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
            if (!fvp[pi].coupled) continue;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const scalar v = scalar(0.21)*std::sin(scalar(1.9)*scalar(i) + scalar(0.4)*scalar(pi));
                phiB[pi][static_cast<std::size_t>(i)] = v;
                if (v > 0) ++nPos; else ++nNeg;
            }
        }
        std::printf("  interface flux: %zu outflow faces, %zu inflow faces\n", nPos, nNeg);
        check("the interface flux changes sign over the pair, so the upwind split is exercised",
              nPos > 0 && nNeg > 0);

        std::vector<scalar> nuCell(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            const vector& C = g.C()[c];
            nuCell[static_cast<std::size_t>(c)] =
                scalar(1e-3)*(scalar(1.4) + std::sin(scalar(2.1)*C.x)*std::cos(scalar(1.3)*C.y));
        }
        const SurfaceScalarField nuf = fvc::interpolate(nuCell, m, g, fvp);
        DeviceBuffer<scalar> dNuCell(nuCell);

        for (int mpass = 0; mpass < 2; ++mpass)
        {
        const bool mCorrected = (mpass == 1);
        const char* mName = mCorrected ? "momentum, corrected" : "momentum, orthogonal";
        const FvScalarMatrix hDiv = fvm::div<scalar>(phiIf, phiB, vf, m, fvp);
        const FvScalarMatrix hLap = fvm::laplacian<scalar>(nuf, vf, m, g, fvp, mCorrected);

        DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);
        {   // the interface flux, in buildDeviceCyclic's own face order
            std::vector<scalar> flat;
            for (const CyclicInterface& c : cyclics)
            {
                for (std::size_t i = 0; i < c.faceCells.size(); ++i)
                {
                    flat.push_back(phiB[static_cast<std::size_t>(c.patch)][i]);
                }
            }
            cyc.phi.copyFrom(flat);
        }
        DeviceBuffer<scalar> mdiag(std::vector<scalar>(static_cast<std::size_t>(nC), scalar(0)));
        deviceCyclicAssembleMomentum(cyc, dNuCell, mdiag, nullptr, mCorrected);

        std::vector<scalar> ifCoeff, diagAdd;
        cyc.ifCoeff.copyTo(ifCoeff);
        mdiag.copyTo(diagAdd);

        std::vector<scalar> hostIf;
        std::vector<scalar> hostDiag(static_cast<std::size_t>(nC), scalar(0));
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& P = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                const scalar bc = hDiv.boundaryCoeffs[c.patch][i] - hLap.boundaryCoeffs[c.patch][i];
                const scalar ic = hDiv.internalCoeffs[c.patch][i] - hLap.internalCoeffs[c.patch][i];
                hostIf.push_back(-bc);
                hostDiag[static_cast<std::size_t>(P.faceCells[i])] += ic;
            }
        }
        scalar worst = 0, scale = 0, worstDiag = 0, diagScale = 0;
        for (std::size_t j = 0; j < ifCoeff.size() && j < hostIf.size(); ++j)
        {
            worst = std::fmax(worst, std::fabs(ifCoeff[j] - hostIf[j]));
            scale = std::fmax(scale, std::fabs(hostIf[j]));
        }
        for (label c = 0; c < nC; ++c)
        {
            const std::size_t k = static_cast<std::size_t>(c);
            worstDiag = std::fmax(worstDiag, std::fabs(diagAdd[k] - hostDiag[k]));
            diagScale = std::fmax(diagScale, std::fabs(hostDiag[k]));
        }
        std::printf("  %s: worst |device - host| interface %.4e (up to %.4e), diagonal %.4e "
                    "(up to %.4e)\n", mName, (double)worst, (double)scale, (double)worstDiag,
                    (double)diagScale);
        check("the device's momentum interface coefficient IS the host's",
              ifCoeff.size() == hostIf.size()
              && worst <= scalar(1e-14)*std::fmax(scale, scalar(1e-300)));
        check("...and the momentum diagonal it writes is the host's",
              worstDiag <= scalar(1e-14)*std::fmax(diagScale, scalar(1e-300)));
        }
    }

    // ---- THE UPWIND CONVECTION COEFFICIENTS on the pair ------------------------------------------
    // fvm::div(phi, psi) with upwind weights gives a coupled face internalCoeffs = phi*w and
    // boundaryCoeffs = -(phi*(1 - w)) with w = pos0(phi), as on an internal face (fvm.cuh:536-549).
    // That is what the alpha equation's implicit pre-solve assembles, and deviceCyclicAddConvection is
    // its device twin -- it ADDS to whatever the laplacian left, which is why it runs on a zeroed
    // diagonal here.
    {
        std::vector<scalar> phiB;
        std::size_t nPos = 0, nNeg = 0;
        for (const CyclicInterface& c : cyclics)
        {
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                const scalar v = scalar(0.07)*std::sin(scalar(1.3)*scalar(i) + scalar(c.patch));
                phiB.push_back(v);
                if (v >= 0) ++nPos; else ++nNeg;
            }
        }
        check("the convection arm's flux changes sign, so both upwind branches are taken",
              nPos > 0 && nNeg > 0);

        DeviceCyclic cycC = buildDeviceCyclic(cyclics, g, fvp);
        cycC.phi.copyFrom(phiB);
        DeviceBuffer<scalar> cdiag(std::vector<scalar>(static_cast<std::size_t>(nC), scalar(0)));
        {   // ifCoeff is ADDED to, so it starts at zero as the laplacian would have left it
            std::vector<scalar> zeros(static_cast<std::size_t>(cycC.n), scalar(0));
            cycC.ifCoeff.copyFrom(zeros);
        }
        deviceCyclicAddConvection(cycC, cdiag);
        std::vector<scalar> devIf, devDiag;
        cycC.ifCoeff.copyTo(devIf);
        cdiag.copyTo(devDiag);

        // the host's own, from its upwind div on the same faces
        SurfaceScalarField phiF;
        phiF.internal.assign(static_cast<std::size_t>(m.nInternalFaces()), scalar(0));
        phiF.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            phiF.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        }
        {
            std::size_t j = 0;
            for (const CyclicInterface& c : cyclics)
                for (std::size_t i = 0; i < c.faceCells.size(); ++i, ++j)
                    phiF.boundary[static_cast<std::size_t>(c.patch)][i] = phiB[j];
        }
        const FvScalarMatrix hDiv = fvm::div<scalar>(phiF.internal, phiF.boundary, vf, m, fvp);
        std::vector<scalar> hostIf, hostDiag(static_cast<std::size_t>(nC), scalar(0));
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& Pp = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                hostIf.push_back(hDiv.boundaryCoeffs[c.patch][i]);
                hostDiag[static_cast<std::size_t>(Pp.faceCells[i])] += hDiv.internalCoeffs[c.patch][i];
            }
        }
        scalar wIf = 0, sIf = 0, wD = 0, sD = 0;
        for (std::size_t j = 0; j < devIf.size() && j < hostIf.size(); ++j)
        {
            // the device stores the coefficient deviceAmul multiplies by; the host stores what the
            // matrix subtracts, so they differ by the sign the laplacian arm above also carries
            wIf = std::fmax(wIf, std::fabs(devIf[j] - (-hostIf[j])));
            sIf = std::fmax(sIf, std::fabs(hostIf[j]));
        }
        for (label c = 0; c < nC; ++c)
        {
            const std::size_t k = static_cast<std::size_t>(c);
            wD = std::fmax(wD, std::fabs(devDiag[k] - hostDiag[k]));
            sD = std::fmax(sD, std::fabs(hostDiag[k]));
        }
        std::printf("  convection: worst |device - host| interface %.4e (up to %.4e), diagonal %.4e "
                    "(up to %.4e)\n", (double)wIf, (double)sIf, (double)wD, (double)sD);
        check("the device's upwind convection coefficient on a coupled face IS the host's",
              devIf.size() == hostIf.size() && wIf == scalar(0));
        check("...and so is the diagonal it adds", wD == scalar(0));
        check("...and neither is identically zero", sIf > scalar(0) && sD > scalar(0));
    }

    // ---- THE FLUX, THE DIVERGENCE AND THE GRADIENT ON THE PAIR ----------------------------------
    // Three explicit operators the pressure corrector needs across a cyclic, each against the host's own
    // coupled branch: fvc::flux (fvc.cu:399-402, dot(coupledLinear(H), Sf)), fvc::div's boundary sum
    // (the owner cell gains phi_b/V) and fvc::gaussGrad (fvc.cu:50-53, Sf*coupledLinear(psi)/V).
    // OpenFOAM's own coupled interpolation is lerp(patchNeighbourField, patchInternalField, lambda) =
    // (1 - w)*N + w*P (surfaceInterpolationScheme.C:284-293 with VectorI.H:259-274), which is the form
    // both arms already use -- unlike an INTERNAL face, where OF writes lambda*(P - N) + N and the two
    // differ in the last bit.
    {
        std::vector<scalar> Hx(static_cast<std::size_t>(nC)), Hy(static_cast<std::size_t>(nC)),
                            Hz(static_cast<std::size_t>(nC)), psi(static_cast<std::size_t>(nC));
        std::vector<vector> H(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            const vector& C = g.C()[c];
            const std::size_t k = static_cast<std::size_t>(c);
            Hx[k] = std::sin(scalar(1.1)*C.x) + scalar(0.3)*C.y;
            Hy[k] = std::cos(scalar(0.9)*C.y) - scalar(0.2)*C.x;
            Hz[k] = scalar(0.05)*std::sin(scalar(2.0)*C.x + C.y);
            H[k] = vector{Hx[k], Hy[k], Hz[k]};
            psi[k] = scalar(1.3) + std::sin(scalar(1.7)*C.x)*std::cos(scalar(1.1)*C.y);
        }

        {   // WHICH WEIGHT does each arm interpolate with? The host patch's and the CyclicInterface's
            // are the same quantity from two builders, and one ulp between them is a whole ulp in every
            // flux across the pair.
            scalar wDiff = 0;
            for (const CyclicInterface& c : cyclics)
            {
                const FvPatch& P = fvp[c.patch];
                for (std::size_t i = 0; i < c.weights.size(); ++i)
                {
                    wDiff = std::fmax(wDiff, std::fabs(c.weights[i] - P.weights[i]));
                }
            }
            std::printf("  weights: worst |CyclicInterface - FvPatch| %.4e\n", (double)wDiff);
        }
        DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);
        DeviceBuffer<scalar> dHx(Hx), dHy(Hy), dHz(Hz), dPsi(psi), dV(g.V());
        deviceCyclicFlux(cyc, dHx, dHy, dHz);
        std::vector<scalar> devPhi;
        cyc.phi.copyTo(devPhi);

        // the host's flux on the same faces
        std::vector<scalar> hostPhi;
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& P = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                hostPhi.push_back(dot(coupledLinear<vector>(P, static_cast<label>(i), H),
                                      g.Sf()[P.start + static_cast<label>(i)]));
            }
        }
        // IS THE DIFFERENCE THE COMPILER'S CONTRACTION? The device kernel writes
        // `wj*Hx[o] + wn*Hx[nb]` and then a three-term dot; with -fmad=true nvcc fuses each into an
        // fma, which rounds once where the host rounds twice. The host form is recomputed here WITH
        // std::fma in the same places, and if that reproduces the device bit for bit the difference is
        // the contraction and nothing else.
        std::vector<scalar> hostFma;
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& P = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                const label o = P.faceCells[i];
                const label nb = c.nbrFaceCells[i];
                const scalar wj = P.weights[i], wn = scalar(1) - wj;
                const std::size_t ok = static_cast<std::size_t>(o), nk = static_cast<std::size_t>(nb);
                const scalar fx = std::fma(wj, Hx[ok], wn*Hx[nk]);
                const scalar fy = std::fma(wj, Hy[ok], wn*Hy[nk]);
                const scalar fz = std::fma(wj, Hz[ok], wn*Hz[nk]);
                const vector& S = g.Sf()[P.start + static_cast<label>(i)];
                hostFma.push_back(std::fma(fx, S.x, std::fma(fy, S.y, fz*S.z)));
            }
        }
        std::size_t fmaExact = 0;
        for (std::size_t j = 0; j < devPhi.size() && j < hostFma.size(); ++j)
        {
            if (devPhi[j] == hostFma[j]) ++fmaExact;
        }
        std::printf("  flux: %zu of %zu faces reproduce the device EXACTLY when the host contracts too\n",
                    fmaExact, devPhi.size());

        scalar worstPhi = 0, phiScale = 0;
        for (std::size_t j = 0; j < devPhi.size() && j < hostPhi.size(); ++j)
        {
            worstPhi = std::fmax(worstPhi, std::fabs(devPhi[j] - hostPhi[j]));
            phiScale = std::fmax(phiScale, std::fabs(hostPhi[j]));
        }
        std::printf("  flux: worst |device - host| %.4e over %zu faces (fluxes up to %.4e)\n",
                    (double)worstPhi, devPhi.size(), (double)phiScale);
        // THE STRONG ARM is the contraction one: every face bit-identical once the host fuses the same
        // multiply-adds. A different weight, a different interpolation form or a different dot order
        // would break it, and none of those is a rounding question.
        check("the device's interface flux is the host's arithmetic, bit for bit under the same "
              "contraction", devPhi.size() == hostPhi.size() && fmaExact == devPhi.size());
        check("...and the two agree to a ulp without it",
              worstPhi <= scalar(4e-16)*std::fmax(phiScale, scalar(1e-300)));

        // ...the divergence that flux carries into the owner cell
        DeviceBuffer<scalar> dDiv(std::vector<scalar>(static_cast<std::size_t>(nC), scalar(0)));
        deviceCyclicAddDiv(cyc, dV, dDiv);
        std::vector<scalar> devDiv;
        dDiv.copyTo(devDiv);
        std::vector<scalar> hostDiv(static_cast<std::size_t>(nC), scalar(0));
        {
            std::size_t j = 0;
            for (const CyclicInterface& c : cyclics)
            {
                const FvPatch& P = fvp[c.patch];
                for (std::size_t i = 0; i < c.faceCells.size(); ++i, ++j)
                {
                    const label o = P.faceCells[i];
                    hostDiv[static_cast<std::size_t>(o)] += hostPhi[j]/g.V()[o];
                }
            }
        }
        scalar worstDiv = 0, divScale = 0;
        for (label c = 0; c < nC; ++c)
        {
            const std::size_t k = static_cast<std::size_t>(c);
            worstDiv = std::fmax(worstDiv, std::fabs(devDiv[k] - hostDiv[k]));
            divScale = std::fmax(divScale, std::fabs(hostDiv[k]));
        }
        std::printf("  div: worst |device - host| %.4e (contributions up to %.4e)\n",
                    (double)worstDiv, (double)divScale);
        check("...and the divergence it adds to the owner cell is the host's",
              worstDiv <= scalar(1e-15)*std::fmax(divScale, scalar(1e-300)));

        // ...and gaussGrad's interface contribution
        DeviceBuffer<scalar> gx(std::vector<scalar>(static_cast<std::size_t>(nC), scalar(0))),
                             gy(std::vector<scalar>(static_cast<std::size_t>(nC), scalar(0))),
                             gz(std::vector<scalar>(static_cast<std::size_t>(nC), scalar(0)));
        deviceCyclicAddGrad(cyc, dPsi, dV, gx, gy, gz);
        std::vector<scalar> dgx, dgy, dgz;
        gx.copyTo(dgx); gy.copyTo(dgy); gz.copyTo(dgz);
        std::vector<vector> hostGrad(static_cast<std::size_t>(nC), vector{0, 0, 0});
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& P = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                const label o = P.faceCells[i];
                const scalar fv = coupledLinear<scalar>(P, static_cast<label>(i), psi)/g.V()[o];
                hostGrad[static_cast<std::size_t>(o)] += g.Sf()[P.start + static_cast<label>(i)]*fv;
            }
        }
        scalar worstGrad = 0, gradScale = 0;
        for (label c = 0; c < nC; ++c)
        {
            const std::size_t k = static_cast<std::size_t>(c);
            worstGrad = std::fmax(worstGrad,
                                  std::fmax(std::fabs(dgx[k] - hostGrad[k].x),
                                            std::fmax(std::fabs(dgy[k] - hostGrad[k].y),
                                                      std::fabs(dgz[k] - hostGrad[k].z))));
            gradScale = std::fmax(gradScale, mag(hostGrad[k]));
        }
        std::printf("  gaussGrad: worst |device - host| %.4e (contributions up to %.4e)\n",
                    (double)worstGrad, (double)gradScale);
        check("...and gaussGrad's interface contribution is the host's",
              worstGrad <= scalar(1e-15)*std::fmax(gradScale, scalar(1e-300)));
    }

    // ---- THE SCALAR TRANSPORT ASSEMBLER'S OWN CALL --------------------------------------------
    // k's and epsilon's equations are fvm::div(phi, f) - fvm::laplacian(DEff, f): the momentum's
    // shape, and so the momentum's interface coefficient, which the arm above already gates kernel
    // for kernel. WHAT THIS ARM GATES IS THE CALL -- that gpu::turbulence::assembleScalarTransport
    // hands the pair the right three things, because each of them is a number that is nearly right:
    //   the DIFFUSIVITY as a CELL field, since a coupled face takes fvc::interpolate's value -- the
    //   two cells' -- and the boundary array beside it is built from nut's PATCH values instead
    //   (kEpsilon_cpp.cu:152-158);
    //   the flux on the PAIR's own faces, not the internal-face array and not always cyc->phi;
    //   the sign, since the laplacian enters the equation negated.
    {
        DeviceBuffer<scalar> psi(static_cast<std::size_t>(nC));
        std::vector<scalar> psiH(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) psiH[static_cast<std::size_t>(c)] = scalar(1) + scalar(0.37)*c;
        psi.copyFrom(psiH);
        const GeometricField<scalar> vfPsi = buildCyclicField<scalar>(psiH, fvp, cyclics);

        // DEff on cells, and a flux that changes sign over the pair so the upwind split is live
        std::vector<scalar> DcellH(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c) DcellH[static_cast<std::size_t>(c)] = scalar(1e-3) + scalar(1e-5)*c;
        DeviceBuffer<scalar> Dcell(DcellH);
        DeviceCyclic tc = buildDeviceCyclic(cyclics, g, fvp);
        std::vector<scalar> phiIfH(static_cast<std::size_t>(tc.n));
        for (int j = 0; j < tc.n; ++j) phiIfH[static_cast<std::size_t>(j)] = (j % 2 ? scalar(-1) : scalar(1))*scalar(1e-3)*(j + 1);
        DeviceBuffer<scalar> cycPhi(phiIfH);

        // the DEVICE's interface coefficient, through the assembler's own path
        DeviceBuffer<scalar> mdiag(static_cast<std::size_t>(nC));
        {
            std::vector<scalar> z(static_cast<std::size_t>(nC), scalar(0));
            mdiag.copyFrom(z);
        }
        // cyc.phi holds a DIFFERENT flux on purpose. The equation's convecting flux on the pair is not
        // always the pair's own phi -- a compressible closure convects with the mass flux -- so the
        // assembler takes it as its own argument, and this is what makes that argument witnessable:
        // with it dropped the coefficients fall back to cyc.phi and the comparison below fails.
        std::vector<scalar> otherPhi(phiIfH.size());
        for (std::size_t j = 0; j < otherPhi.size(); ++j) otherPhi[j] = scalar(-3)*phiIfH[j];
        tc.phi.copyFrom(otherPhi);
        deviceCyclicAssembleMomentum(tc, Dcell, mdiag, nullptr, /*corrected=*/true, &cycPhi);
        std::vector<scalar> devIf, devDiag;
        tc.ifCoeff.copyTo(devIf);
        mdiag.copyTo(devDiag);

        // ...and the HOST's, from the public fvm operators: div - laplacian
        SurfaceScalarField phiF;
        phiF.internal.assign(static_cast<std::size_t>(m.nInternalFaces()), scalar(0));
        phiF.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            phiF.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        {
            std::size_t j = 0;
            for (const CyclicInterface& c : cyclics)
                for (std::size_t i = 0; i < c.faceCells.size(); ++i, ++j)
                    phiF.boundary[static_cast<std::size_t>(c.patch)][i] = phiIfH[j];
        }
        const SurfaceScalarField gammaF = fvc::interpolate(DcellH, m, g, fvp);
        const FvScalarMatrix hDiv = fvm::div<scalar>(phiF.internal, phiF.boundary, vfPsi, m, fvp);
        const FvScalarMatrix hLap = fvm::laplacian<scalar>(gammaF, vfPsi, m, g, fvp, /*corrected=*/true);
        std::vector<scalar> hostIf, hostDiag(static_cast<std::size_t>(nC), scalar(0));
        for (const CyclicInterface& c : cyclics)
        {
            const FvPatch& Pp = fvp[c.patch];
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                hostIf.push_back(hDiv.boundaryCoeffs[c.patch][i] - hLap.boundaryCoeffs[c.patch][i]);
                hostDiag[static_cast<std::size_t>(Pp.faceCells[i])] +=
                    hDiv.internalCoeffs[c.patch][i] - hLap.internalCoeffs[c.patch][i];
            }
        }
        scalar wIf = 0, sIf = 0, wD = 0, sD = 0;
        for (std::size_t j = 0; j < devIf.size() && j < hostIf.size(); ++j)
        {
            // the device stores what deviceAmul multiplies by, the host what the matrix subtracts
            wIf = std::fmax(wIf, std::fabs(devIf[j] - (-hostIf[j])));
            sIf = std::fmax(sIf, std::fabs(hostIf[j]));
        }
        for (label c = 0; c < nC; ++c)
        {
            const std::size_t k = static_cast<std::size_t>(c);
            wD = std::fmax(wD, std::fabs(devDiag[k] - hostDiag[k]));
            sD = std::fmax(sD, std::fabs(hostDiag[k]));
        }
        std::printf("  scalar transport: worst |device - host| interface %.4e (up to %.4e), "
                    "diagonal %.4e (up to %.4e)\n", (double)wIf, (double)sIf, (double)wD, (double)sD);
        check("the transport equation's interface coefficient on a coupled face IS the host's",
              devIf.size() == hostIf.size() && !hostIf.empty()
              && wIf <= scalar(1e-15)*std::fmax(sIf, scalar(1e-300)));
        check("...and so is the diagonal it writes",
              wD <= scalar(1e-15)*std::fmax(sD, scalar(1e-300)));
        // THE FLUX HAS BOTH SIGNS on the pair, so the upwind split is exercised and not assumed
        int nOut = 0, nIn = 0;
        for (scalar v : phiIfH) { if (v > 0) ++nOut; else ++nIn; }
        std::printf("  (the pair's flux: %d outflow faces, %d inflow)\n", nOut, nIn);
        check("...on a pair whose flux changes sign, so the upwind split is live", nOut > 0 && nIn > 0);
    }

    std::printf("test_device_cyclic_laplacian_vs_host: %d failures\n", failures);
    return failures ? 1 : 0;
}
