// The FUSED Gauss gradient is the same gradient, not a similar one.
//
// WHY IT EXISTS. nsys on brae's compressible iteration at 305,760 cells (12 outer iterations): the
// iteration is about 30 ms of GPU-busy time in 914 launches -- launch- and bandwidth-bound -- and
// gradKernel is 16 of those launches at 281 us each, 4.5 ms/it, 15% of all GPU work and second only to
// the matrix-vector product. Nine of the sixteen are velocity components taken in threes, and each of
// the three re-reads the WHOLE of the addressing and geometry to carry one field. deviceGaussGradFused
// reads it once for up to three fields.
//
// WHAT IS AT STAKE. A fused kernel's failure mode is not "wrong formula", it is CROSS-CONTAMINATION:
// one field's face value, weight or accumulator leaking into another's, which looks plausible and
// converges. So the bar here is memcmp, not a tolerance -- fusing N independent fields is a loop
// interchange, the same faces in the same order with the same expressions, so the bits must be equal
// to N separate deviceGaussGrad launches. The reference is the UNCHANGED gradKernel, which is a
// genuinely different kernel and not a rename of the one under test; deviceGaussGrad deliberately does
// NOT forward to the fused path, or these arms would be comparing a thing to itself.
//
// ARMS
//   (a) n = 3 against three deviceGaussGrad calls, memcmp on all nine arrays.
//   (b) n = 1 (each field on its own) and n = 2, likewise -- the N-dependent unrolling must not change
//       any field's arithmetic.
//   (c) skipIf: a raised flag makes the fused launch a no-op, exactly as it does for the single-field
//       one -- the outputs keep a sentinel written before the launch; lowered, it computes.
//   (d) CONTROL. One ulp into field 1's interior, and separately into field 1's boundary values: the
//       three field-1 arrays must move and the field-0 and field-2 arrays must not move by a bit. This
//       is what fails if the fusion crosses fields, and it fails in BOTH directions (a contaminated
//       kernel either moves the wrong outputs or, if it reads field 0 for all three, fails to move the
//       right ones).
//   (e) EMPTY PATCHES. The same mesh with its two z patches typed `empty`: the fused and separate arms
//       must still agree bit for bit, poisoning the empty patch's boundary values must change NEITHER
//       arm (the bndIsEmpty skip is live in the fused loop), and the same poison on a NON-empty patch
//       must change both -- otherwise a skip that dropped every boundary face would pass.
//   (f) The three fields' gradients differ pairwise, and none is all-zero. Without this the memcmp arms
//       could be comparing three copies of the same numbers, and (d) would have nothing to detect.
//
// FAIL-PROOF, RUN: with the fused boundary loop reading field 0's bval for every field
// (`fld.bval[0][kk]` in place of `fld.bval[i][kk]`), arms (a), (b) n=2, (d) and (e) went red and the
// exit code was 1; restoring it turned them green again.
#include "box_mesh.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::printf("  %-74s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

// Bit equality, not a tolerance: the fused kernel visits the same faces in the same order with the same
// expressions, so anything short of equality is a real difference in what is being computed.
bool sameBits(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(scalar)) == 0;
}

struct G3
{
    std::vector<scalar> x, y, z;
};

G3 fetch(const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz)
{
    G3 r;
    r.x = gx.host();
    r.y = gy.host();
    r.z = gz.host();
    return r;
}

bool sameG3(const G3& a, const G3& b)
{
    return sameBits(a.x, b.x) && sameBits(a.y, b.y) && sameBits(a.z, b.z);
}

// The reference: N separate deviceGaussGrad launches, i.e. the untouched gradKernel.
std::vector<G3> separateGrads(
    const DeviceMesh& dm,
    int n,
    const DeviceBuffer<scalar>* vol,
    const DeviceBuffer<scalar>* bval)
{
    std::vector<G3> out;
    for (int i = 0; i < n; ++i)
    {
        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, vol[i], bval[i], gx, gy, gz);
        out.push_back(fetch(gx, gy, gz));
    }
    return out;
}

// The thing under test: one launch for all n.
std::vector<G3> fusedGrads(
    const DeviceMesh& dm,
    int n,
    const DeviceBuffer<scalar>* vol,
    const DeviceBuffer<scalar>* bval)
{
    const DeviceBuffer<scalar>* vp[3] = {&vol[0], &vol[1], &vol[2]};
    const DeviceBuffer<scalar>* bp[3] = {&bval[0], &bval[1], &bval[2]};
    DeviceBuffer<scalar> gx[3], gy[3], gz[3];
    deviceGaussGradFused(dm, n, vp, bp, gx, gy, gz);
    std::vector<G3> out;
    for (int i = 0; i < n; ++i)
    {
        out.push_back(fetch(gx[i], gy[i], gz[i]));
    }
    return out;
}

bool allZero(const std::vector<scalar>& v)
{
    for (const scalar s : v)
    {
        if (s != 0.0) return false;
    }
    return true;
}

// Every arm of the suite, run against whichever DeviceMesh it is handed -- the plain box for (a)-(d),
// the one with empty z patches for (e). `tag` names the mesh in the printed line.
void runArms(
    const DeviceMesh& dm,
    const std::vector<std::vector<scalar>>& hVol,
    const std::vector<std::vector<scalar>>& hBnd,
    const std::string& tag)
{
    DeviceBuffer<scalar> vol[3], bval[3];
    for (int i = 0; i < 3; ++i)
    {
        vol[i].copyFrom(hVol[i]);
        bval[i].copyFrom(hBnd[i]);
    }

    const std::vector<G3> ref = separateGrads(dm, 3, vol, bval);

    // (f) the fields must actually differ, or nothing below could discriminate anything.
    check(!allZero(ref[0].x) && !allZero(ref[1].y) && !allZero(ref[2].z),
          tag + ": the reference gradients are not all zero");
    check(!sameG3(ref[0], ref[1]) && !sameG3(ref[1], ref[2]) && !sameG3(ref[0], ref[2]),
          tag + ": the three reference gradients differ pairwise");

    // (a) n = 3
    {
        const std::vector<G3> f3 = fusedGrads(dm, 3, vol, bval);
        bool ok = true;
        for (int i = 0; i < 3; ++i)
        {
            ok = ok && sameG3(f3[i], ref[i]);
        }
        check(ok, tag + ": fused n=3 == three deviceGaussGrad calls, BIT FOR BIT (9 arrays)");
    }

    // (b) n = 1 and n = 2
    {
        bool ok = true;
        for (int i = 0; i < 3; ++i)
        {
            const DeviceBuffer<scalar>* vp[1] = {&vol[i]};
            const DeviceBuffer<scalar>* bp[1] = {&bval[i]};
            DeviceBuffer<scalar> gx[1], gy[1], gz[1];
            deviceGaussGradFused(dm, 1, vp, bp, gx, gy, gz);
            ok = ok && sameG3(fetch(gx[0], gy[0], gz[0]), ref[i]);
        }
        check(ok, tag + ": fused n=1 == deviceGaussGrad, bit for bit, for each of the three fields");

        const std::vector<G3> f2 = fusedGrads(dm, 2, vol, bval);
        check(f2.size() == 2 && sameG3(f2[0], ref[0]) && sameG3(f2[1], ref[1]),
              tag + ": fused n=2 == two deviceGaussGrad calls, bit for bit");
    }

    // (c) skipIf, against the single-field kernel's own behaviour
    {
        DeviceBuffer<int> flag(std::vector<int>{1});
        const std::vector<scalar> sentinel(hVol[0].size(), -12345.5);

        DeviceBuffer<scalar> sgx, sgy, sgz;
        sgx.copyFrom(sentinel);
        sgy.copyFrom(sentinel);
        sgz.copyFrom(sentinel);
        deviceGaussGrad(dm, vol[0], bval[0], sgx, sgy, sgz, flag.data());
        const bool singleSkipped = sameBits(sgx.host(), sentinel) && sameBits(sgy.host(), sentinel)
                                && sameBits(sgz.host(), sentinel);

        const DeviceBuffer<scalar>* vp[3] = {&vol[0], &vol[1], &vol[2]};
        const DeviceBuffer<scalar>* bp[3] = {&bval[0], &bval[1], &bval[2]};
        DeviceBuffer<scalar> gx[3], gy[3], gz[3];
        for (int i = 0; i < 3; ++i)
        {
            gx[i].copyFrom(sentinel);
            gy[i].copyFrom(sentinel);
            gz[i].copyFrom(sentinel);
        }
        deviceGaussGradFused(dm, 3, vp, bp, gx, gy, gz, flag.data());
        bool fusedSkipped = true;
        for (int i = 0; i < 3; ++i)
        {
            fusedSkipped = fusedSkipped && sameBits(gx[i].host(), sentinel)
                        && sameBits(gy[i].host(), sentinel) && sameBits(gz[i].host(), sentinel);
        }
        check(singleSkipped && fusedSkipped,
              tag + ": a raised skipIf makes BOTH the single and the fused launch a no-op");

        const std::vector<int> lowered{0};
        flag.copyFrom(lowered);
        deviceGaussGradFused(dm, 3, vp, bp, gx, gy, gz, flag.data());
        bool ran = true;
        for (int i = 0; i < 3; ++i)
        {
            ran = ran && sameG3(fetch(gx[i], gy[i], gz[i]), ref[i]);
        }
        check(ran, tag + ": a lowered skipIf computes the same bits as no flag at all");
    }

    // (d) CONTROL: one ulp into field 1 only.
    {
        std::vector<std::vector<scalar>> pVol = hVol;
        pVol[1][pVol[1].size() / 2] = std::nextafter(pVol[1][pVol[1].size() / 2], 1e30);
        DeviceBuffer<scalar> pv[3];
        for (int i = 0; i < 3; ++i)
        {
            pv[i].copyFrom(pVol[i]);
        }
        const std::vector<G3> f = fusedGrads(dm, 3, pv, bval);
        check(!sameG3(f[1], ref[1]),
              tag + ": CONTROL one ulp into field 1's interior MOVES field 1's gradient");
        check(sameG3(f[0], ref[0]) && sameG3(f[2], ref[2]),
              tag + ": CONTROL ...and leaves fields 0 and 2 bit-identical (no cross-contamination)");
    }
    {
        std::vector<std::vector<scalar>> pBnd = hBnd;
        // A face of the FIRST patch, which is never the empty one in either mesh here, so this control
        // is live on both.
        pBnd[1][0] = std::nextafter(pBnd[1][0], 1e30);
        DeviceBuffer<scalar> pb[3];
        for (int i = 0; i < 3; ++i)
        {
            pb[i].copyFrom(pBnd[i]);
        }
        const std::vector<G3> f = fusedGrads(dm, 3, vol, pb);
        check(!sameG3(f[1], ref[1]),
              tag + ": CONTROL one ulp into field 1's BOUNDARY values moves field 1's gradient");
        check(sameG3(f[0], ref[0]) && sameG3(f[2], ref[2]),
              tag + ": CONTROL ...and leaves fields 0 and 2 bit-identical");
    }
}

} // namespace


int main()
{
    std::printf("== fused Gauss gradient: N fields in one launch, bit for bit ==\n");

    // Sheared, so the mesh is non-orthogonal and the face weights are not all 0.5 -- a w that is
    // constant would hide an interpolation read taken from the wrong field.
    const PrimitiveMesh m = boxtest::boxMesh(9, 7, 5, 0.3);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const int nC = static_cast<int>(m.nCells());
    const int nB = static_cast<int>(m.nFaces() - m.nInternalFaces());
    std::printf("  box 9x7x5 sheared: %d cells, %d internal faces, %d boundary faces\n",
                nC, static_cast<int>(m.nInternalFaces()), nB);

    // Three fields that are genuinely different functions of position, with boundary values that are
    // NOT the owner-cell value (an inlet/outlet profile, not a zeroGradient) -- a boundary loop reading
    // the wrong field has to show up somewhere.
    const std::vector<vector>& C = g.C();
    std::vector<std::vector<scalar>> hVol(3, std::vector<scalar>(nC));
    for (int c = 0; c < nC; ++c)
    {
        hVol[0][c] = 1.0 + C[c].x + 0.5 * C[c].y * C[c].y - 0.25 * C[c].z;
        hVol[1][c] = std::sin(0.31 * C[c].x) * (2.0 + C[c].z) - 0.7 * C[c].y;
        hVol[2][c] = 3.0 - 0.4 * C[c].x * C[c].z + std::cos(0.23 * C[c].y);
    }
    std::vector<std::vector<scalar>> hBnd(3, std::vector<scalar>(nB));
    for (int f = 0; f < nB; ++f)
    {
        hBnd[0][f] = 0.9 + 0.13 * f - 0.001 * f * f;
        hBnd[1][f] = std::cos(0.17 * f) * 4.0 + 1.5;
        hBnd[2][f] = -2.0 + std::sin(0.41 * f) * 0.75;
    }

    {
        DeviceMesh dm = buildDeviceMesh(m, g, fvp);
        runArms(dm, hVol, hBnd, "plain box");
    }

    // (e) the same mesh with its two z patches declared `empty`, so bndIsEmpty is set on those faces
    // and the fused boundary loop has to skip them exactly as gradKernel does.
    {
        std::vector<FvPatch> ep = fvp;
        int nEmptyFaces = 0, firstEmpty = -1;
        for (std::size_t pi = 0; pi < ep.size(); ++pi)
        {
            if (ep[pi].name == "wallZmin" || ep[pi].name == "wallZmax")
            {
                ep[pi].type = "empty";
                if (firstEmpty < 0) firstEmpty = static_cast<int>(ep[pi].start - m.nInternalFaces());
                nEmptyFaces += static_cast<int>(ep[pi].size);
            }
        }
        std::printf("  empty-patch variant: %d of %d boundary faces are on an `empty` patch\n",
                    nEmptyFaces, nB);
        check(nEmptyFaces > 0 && firstEmpty >= 0,
              "empty box: the fixture HAS empty faces (else arm (e) is vacuous)");

        DeviceMesh dme = buildDeviceMesh(m, g, ep);
        runArms(dme, hVol, hBnd, "empty box");

        // The skip itself: poison the empty patch and nothing may move, on EITHER arm; poison a
        // non-empty patch and both must move.
        DeviceBuffer<scalar> vol[3], bval[3];
        for (int i = 0; i < 3; ++i)
        {
            vol[i].copyFrom(hVol[i]);
            bval[i].copyFrom(hBnd[i]);
        }
        const std::vector<G3> ref = separateGrads(dme, 3, vol, bval);
        const std::vector<G3> fus = fusedGrads(dme, 3, vol, bval);

        std::vector<std::vector<scalar>> poisonEmpty = hBnd;
        for (int i = 0; i < 3; ++i)
        {
            for (int f = firstEmpty; f < firstEmpty + nEmptyFaces / 2; ++f)
            {
                poisonEmpty[i][f] = 1e30;
            }
        }
        DeviceBuffer<scalar> pe[3];
        for (int i = 0; i < 3; ++i)
        {
            pe[i].copyFrom(poisonEmpty[i]);
        }
        const std::vector<G3> refPE = separateGrads(dme, 3, vol, pe);
        const std::vector<G3> fusPE = fusedGrads(dme, 3, vol, pe);
        bool inertRef = true, inertFus = true;
        for (int i = 0; i < 3; ++i)
        {
            inertRef = inertRef && sameG3(refPE[i], ref[i]);
            inertFus = inertFus && sameG3(fusPE[i], fus[i]);
        }
        check(inertRef, "empty box: poisoning the empty patch is inert in deviceGaussGrad");
        check(inertFus, "empty box: poisoning the empty patch is inert in the FUSED gradient");

        std::vector<std::vector<scalar>> poisonReal = hBnd;
        for (int i = 0; i < 3; ++i)
        {
            poisonReal[i][0] = 1e30;   // face 0 is on `inlet`, a real patch
        }
        DeviceBuffer<scalar> pr[3];
        for (int i = 0; i < 3; ++i)
        {
            pr[i].copyFrom(poisonReal[i]);
        }
        const std::vector<G3> refPR = separateGrads(dme, 3, vol, pr);
        const std::vector<G3> fusPR = fusedGrads(dme, 3, vol, pr);
        check(!sameG3(refPR[0], ref[0]) && !sameG3(fusPR[0], fus[0]),
              "empty box: CONTROL the same poison on a NON-empty patch moves both arms");
    }

    std::printf(failures ? "FAILED (%d)\n" : "PASSED (%d failures)\n", failures);
    return failures ? 1 : 0;
}
