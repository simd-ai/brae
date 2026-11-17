// cellLimitedGrad has to SEE the coupled patches.
//
// A cyclic / cyclicAMI / cyclicACMI face is a boundary face to brae's addressing -- DeviceBoundary skips
// it, so it appears in neither dm.bndCellStart nor Ubnd -- but it is an INTERNAL face to OF's limiter.
// Foam::fv::cellLimitedGrad::calcGrad touches it twice:
//   1. the RANGE: `if (psf.coupled())` folds psf.patchNeighbourField() into maxVsf/minVsf
//   2. the LIMIT: the second loop runs over EVERY boundary face, coupled ones included, and clips the
//      gradient so the value extrapolated to that face stays inside the range
// brae did neither, so a cell touching an interface was limited as though the interface were not there.
//
// HOW IT WAS FOUND, because the route matters more than the fix. The turbulent ACMI case sat at 2.2e-03
// against OpenFOAM while the laminar one was at 1.4e-07, and that looked like a turbulence-model defect
// for a long time. It is not: the two cases also differ in their convection scheme (`Gauss upwind` vs
// `Gauss linearUpwind grad(U)`), and running the case LAMINAR with linearUpwind reproduced the whole gap
// with no turbulence model present at all. From there, one measurement settled it -- the same laminar
// linearUpwind case on an UNLIMITED grad(U):
//
//     grad(U) Gauss linear              : 2.0e-09
//     grad(U) cellLimited Gauss linear 1: 1.9e-03
//
// a factor of a million, which acquits linearUpwind (and the whole interface flux path) and convicts the
// limiter. 91% of the squared error sat on the 136 interface-adjacent cells, 1.25% of the mesh. With the
// coupled patches folded in, that case goes to 1.4e-07 -- the same agreement the upwind case gets -- and
// the static LAMINAR tutorial improves too, 1.4e-07 to 2.3e-09, because its divDevReff gradient is
// cellLimited as well.
//
// WHAT THIS ASSERTS. Not a formula but the algorithm: OF's cellLimitedGrad is re-implemented on the host,
// coupled faces and all, and the device result must match it. Leg 2 then removes the interfaces from the
// device call and requires the answer to CHANGE -- without that, a fixture where the interface happens
// not to bind would let the defect straight back in.
#include "acmi_mesh.cuh"
#include "acmi_area_scaling.cuh"
#include "device_mesh.cuh"
#include "device_ami.cuh"
#include "device_boundary.cuh"
#include "cyclic_field.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

scalar limFaceHost(scalar maxD, scalar minD, scalar ex)
{
    if (ex >  1e-15) return std::fmin(maxD/ex, scalar(1));
    if (ex < -1e-15) return std::fmin(minD/ex, scalar(1));
    return scalar(1);
}

}   // namespace

int main()
{
    PrimitiveMesh m = acmitest::twoBlockACMI(acmitest::ACMI_DY, /*withBlockage*/true, "ACMI1_blockage",
                                             /*frontBackEmpty*/true);
    FvGeometry g;
    std::vector<FvPatch> fvp;
    std::vector<AMIInterface> amis;
    buildGeometryPatchesAndAMI(m, g, fvp, amis);
    const label nC = m.nCells();

    // A field that varies ACROSS the interface, so the coupled neighbour is genuinely outside the range
    // the internal faces alone would give. A field smooth in x would leave the interface non-binding.
    std::vector<scalar> U(nC);
    for (label c = 0; c < nC; ++c) U[c] = 3.0*g.C()[c].x + 1.7*g.C()[c].y*g.C()[c].y;

    GeometricField<scalar> Uf = buildCyclicField<scalar>(U, fvp, {});
    Uf.evaluateBoundary();

    const DeviceMesh  dm  = buildDeviceMesh(m, g, fvp);
    const DeviceAMI   ami = buildDeviceAMI(amis);
    const DeviceBoundary dbU = buildDeviceBoundary(Uf, fvp, g);

    // An arbitrary but deliberately STEEP seed gradient: the limiter is what is under test, not the
    // gradient operator, so feeding it a gradient that must be clipped isolates it.
    std::vector<scalar> gx0(nC), gy0(nC), gz0(nC);
    for (label c = 0; c < nC; ++c)
    { gx0[c] = 40.0 + 3.0*c; gy0[c] = -25.0 - 1.5*c; gz0[c] = 0.0; }

    DeviceBuffer<scalar> dU, dUb, gx, gy, gz;
    dU.copyFrom(U);
    deviceBCValue(dbU, dU, dUb);
    const std::vector<scalar> Ubnd = dUb.host();

    // The AMI neighbour value per source face, and the pieces the host reference needs.
    DeviceBuffer<scalar> nbrValD;
    deviceAmiInterpolate(ami, dU, nbrValD);
    const std::vector<scalar> nbrVal = nbrValD.host();
    const std::vector<label>  aOwn   = ami.ownCell.host();
    const std::vector<scalar> aDx = ami.dOwnX.host(), aDy = ami.dOwnY.host(), aDz = ami.dOwnZ.host();
    std::printf("  fixture: %d cells, %d AMI source faces\n", (int)nC, ami.n);
    if (ami.n <= 0)
    { std::printf("  FAIL vacuous: the fixture built no AMI faces\n"); ++failures; }

    const scalar kc = 1.0;

    // ---------- the host reference: OF cellLimitedGrad<minmod>, coupled patches included ----------
    std::vector<scalar> want[3];
    {
        std::vector<scalar> maxD(nC, 0.0), minD(nC, 0.0);
        for (label f = 0; f < m.nInternalFaces(); ++f)
        {
            const label o = m.owner()[f], n = m.neighbour()[f];
            const scalar dn = U[n] - U[o], dp = U[o] - U[n];
            maxD[o] = std::fmax(maxD[o], dn); minD[o] = std::fmin(minD[o], dn);
            maxD[n] = std::fmax(maxD[n], dp); minD[n] = std::fmin(minD[n], dp);
        }
        std::size_t b = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)   // NON-coupled patches use their face value
        {
            if (isCyclicFamilyType(fvp[pi].type)) continue;
            for (label i = 0; i < fvp[pi].size; ++i, ++b)
            {
                const label c = fvp[pi].faceCells[i];
                const scalar d = Ubnd[b] - U[c];
                maxD[c] = std::fmax(maxD[c], d); minD[c] = std::fmin(minD[c], d);
            }
        }
        for (int i = 0; i < ami.n; ++i)                  // ...and the COUPLED one its neighbour value
        {
            const label c = aOwn[i];
            const scalar d = nbrVal[i] - U[c];
            maxD[c] = std::fmax(maxD[c], d); minD[c] = std::fmin(minD[c], d);
        }

        std::vector<scalar> lim(nC, 1.0);
        auto clip = [&](label c, scalar dx, scalar dy, scalar dz)
        { lim[c] = std::fmin(lim[c], limFaceHost(maxD[c], minD[c], dx*gx0[c] + dy*gy0[c] + dz*gz0[c])); };
        for (label f = 0; f < m.nInternalFaces(); ++f)
        {
            const label o = m.owner()[f], n = m.neighbour()[f];
            const vector& cf = g.Cf()[f];
            clip(o, cf.x - g.C()[o].x, cf.y - g.C()[o].y, cf.z - g.C()[o].z);
            clip(n, cf.x - g.C()[n].x, cf.y - g.C()[n].y, cf.z - g.C()[n].z);
        }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)   // OF's second loop is over ALL boundary faces
        {
            if (isCyclicFamilyType(fvp[pi].type)) continue;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label c = fvp[pi].faceCells[i];
                const vector& cf = g.Cf()[fvp[pi].start + i];
                clip(c, cf.x - g.C()[c].x, cf.y - g.C()[c].y, cf.z - g.C()[c].z);
            }
        }
        for (int i = 0; i < ami.n; ++i) clip(aOwn[i], aDx[i], aDy[i], aDz[i]);   // ...coupled included

        want[0].resize(nC); want[1].resize(nC); want[2].resize(nC);
        for (label c = 0; c < nC; ++c)
        { want[0][c] = gx0[c]*lim[c]; want[1][c] = gy0[c]*lim[c]; want[2][c] = gz0[c]*lim[c]; }

        // VACUITY GUARD: the limiter has to actually bite, or every leg passes on a no-op.
        std::size_t nLim = 0;
        for (label c = 0; c < nC; ++c) if (lim[c] < 1.0 - 1e-12) ++nLim;
        std::printf("  the reference limits %zu of %d cells\n", nLim, (int)nC);
        if (!nLim)
        { std::printf("  FAIL vacuous: nothing is limited, so this fixture tests nothing\n"); ++failures; }
    }

    // ---------- 1. the device result, interfaces declared, must match it ----------
    CellLimitInterface ifs[1] = {
        { ami.n, ami.ownCell.data(), nbrValD.data(), ami.dOwnX.data(), ami.dOwnY.data(), ami.dOwnZ.data() }
    };
    std::vector<scalar> got[3];
    {
        gx.copyFrom(gx0); gy.copyFrom(gy0); gz.copyFrom(gz0);
        deviceCellLimitGrad(dm, dU, dUb, gx, gy, gz, kc, ifs, 1);
        got[0] = gx.host(); got[1] = gy.host(); got[2] = gz.host();
        scalar w = 0;
        for (int k = 0; k < 3; ++k)
            for (label c = 0; c < nC; ++c) w = std::fmax(w, std::fabs(got[k][c] - want[k][c]));
        std::printf("  with interfaces   : max|device - OF reference| = %.3e (must be 0)\n", (double)w);
        if (w > 1e-12)
        {
            std::printf("  FAIL the device limiter does not reproduce OF's algorithm with the coupled\n"
                        "       patches folded in -- see cellLimitedGrad::calcGrad, both of its loops\n");
            ++failures;
        }
    }

    // ---------- 2. without them it must DIFFER -- this is the defect itself ----------
    {
        gx.copyFrom(gx0); gy.copyFrom(gy0); gz.copyFrom(gz0);
        deviceCellLimitGrad(dm, dU, dUb, gx, gy, gz, kc);   // the old blind-to-the-interface call
        const std::vector<scalar> blind = gx.host();
        scalar w = 0;
        std::size_t nDiff = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar d = std::fabs(blind[c] - want[0][c]);
            if (d > 1e-12) ++nDiff;
            w = std::fmax(w, d);
        }
        std::printf("  without them      : differs on %zu cells, by up to %.3e\n", nDiff, (double)w);
        if (!nDiff)
        {
            std::printf("  FAIL vacuous: omitting the interface changes nothing here, so leg 1 would pass\n"
                        "       for an implementation that ignored the coupled patches entirely. The field\n"
                        "       must vary ACROSS the interface for this to bind.\n");
            ++failures;
        }
    }

    // ---------- 3. a declared interface with no faces must be a no-op, not a crash ----------
    {
        gx.copyFrom(gx0); gy.copyFrom(gy0); gz.copyFrom(gz0);
        deviceCellLimitGrad(dm, dU, dUb, gx, gy, gz, kc);
        const std::vector<scalar> plain = gx.host();
        CellLimitInterface none[1] = { { 0, nullptr, nullptr, nullptr, nullptr, nullptr } };
        gx.copyFrom(gx0); gy.copyFrom(gy0); gz.copyFrom(gz0);
        deviceCellLimitGrad(dm, dU, dUb, gx, gy, gz, kc, none, 1);
        const std::vector<scalar> empty = gx.host();
        scalar w = 0;
        for (label c = 0; c < nC; ++c) w = std::fmax(w, std::fabs(plain[c] - empty[c]));
        std::printf("  empty interface   : max|diff| from the plain path = %.3e\n", (double)w);
        if (w != scalar(0))
        {
            std::printf("  FAIL an interface with no faces changed the answer -- the mesh-only path must stay\n"
                        "       bit-for-bit what it was, since every non-coupled case in the suite takes it\n");
            ++failures;
        }
    }

    std::printf("cell_limit_interface: %d failures\n", failures);
    return failures ? 1 : 0;
}
