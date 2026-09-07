// fvc::grad on a 2D mesh: an EMPTY patch contributes nothing, on both arms, as in OpenFOAM.
//
// OpenFOAM cannot sum an empty face: emptyFvPatch::size() returns 0 (emptyFvPatch.H:79), so
// gaussGrad::gradf's boundary loop (gaussGrad.C, `forAll(mesh.boundary()[patchi], facei)`) never
// visits one. brae keeps those faces in its addressing and has to skip them explicitly. Its host
// gradient loop and its device gradKernel did not (fvc.cu's div and boundary-gradient loops did,
// and so did the cellLimited kernel beside gradKernel), so both arms summed every empty face with
// the cell's own value. On an extruded mesh an empty face's Sf_x and Sf_y are bitwise zero, so the
// in-plane gradient could not move -- which is why nothing ever saw it -- while the out-of-plane
// component was the cancellation of two large opposite terms, round-off by construction.
//
//   ORACLE   OpenFOAM's own written grad(p) at the same time (writePrecision 16), all three components.
//   ARMS     host fvc::gaussGrad and device deviceGaussGrad on the fixture's p.
//   CONTROL  poisoning the empty patch's values (1e30 on the +z side only, so the two sides cannot
//            cancel) must change NEITHER arm's gradient by a single bit; poisoning one face of a
//            NON-empty patch must change it -- so a skip that dropped every boundary would fail too.
//   z        max|g_z| stays at round-off against the in-plane gradient on both arms; OpenFOAM's own
//            max|g_z| is printed beside it (it is not zero either: internal faces carry a round-off Sf_z).
//
// FAIL-PROOF, RUN with all four skips removed (three host overloads, one device kernel): the
// empty-patch poison moved max|dg| = 1.000e+33 on BOTH arms, both bit-identity lines failed, exit 1.
// The same run showed what the skips are worth without any poison: max|g_z|/max|g_inplane| was
// 8.777e-16 with the empty faces summed and is 1.516e-16 without them -- OpenFOAM's own number to
// four figures, because the same faces are now being summed.
//
// ONE FIXTURE, on purpose: every validation/ case that carries an OpenFOAM-written grad(p) (matrixDump,
// matrixDumpAsym, matrixDumpSimple, matrixDumpGAMG, kEpsCase, kEpsCorrect, pitzDaily282) is byte-for-
// byte the same pitzDaily mesh and p at time 282, so a second registration would be the same test.
//
// Run: test_empty_face_grad <caseDir> <timeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;
static void say(bool ok, const char* what)
{
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fails;
}

struct Grad3
{
    std::vector<scalar> x, y, z;
};

static Grad3 hostGrad(const std::vector<scalar>& internal,
                      const std::vector<std::vector<scalar>>& bnd,
                      const PrimitiveMesh& m,
                      const FvGeometry& g,
                      const std::vector<FvPatch>& patches)
{
    const std::vector<vector> gv = fvc::gaussGrad(internal, bnd, m, g, patches);
    Grad3 r;
    r.x.resize(gv.size());
    r.y.resize(gv.size());
    r.z.resize(gv.size());
    for (std::size_t c = 0; c < gv.size(); ++c)
    {
        r.x[c] = gv[c].x;
        r.y[c] = gv[c].y;
        r.z[c] = gv[c].z;
    }
    return r;
}

static Grad3 deviceGrad(const DeviceMesh& dm,
                        const DeviceBuffer<scalar>& dp,
                        const std::vector<std::vector<scalar>>& bnd)
{
    DeviceBuffer<scalar> bval(flattenBoundary(bnd));
    DeviceBuffer<scalar> gx, gy, gz;
    deviceGaussGrad(dm, dp, bval, gx, gy, gz);
    Grad3 r;
    r.x = gx.host();
    r.y = gy.host();
    r.z = gz.host();
    return r;
}

static scalar maxAbs(const std::vector<scalar>& v)
{
    scalar m = 0;
    for (scalar a : v) m = std::fmax(m, std::fabs(a));
    return m;
}

static scalar maxDiff(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

// max over cells of |g - gOF| divided by max over cells of |gOF|, on the full vector
static scalar relToOracle(const Grad3& g, const std::vector<vector>& of)
{
    scalar mx = 0, mg = 0;
    for (std::size_t c = 0; c < of.size(); ++c)
    {
        const vector d{g.x[c] - of[c].x, g.y[c] - of[c].y, g.z[c] - of[c].z};
        mx = std::fmax(mx, mag(d));
        mg = std::fmax(mg, mag(of[c]));
    }
    return mx / mg;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), patches, nC);
    p.evaluateBoundary();
    std::vector<std::vector<scalar>> bnd;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) bnd.push_back(p.boundary[pi]->value());

    int emptyPatch = -1, otherPatch = -1;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "empty" && patches[pi].size > 0 && emptyPatch < 0) emptyPatch = (int)pi;
        if (patches[pi].type != "empty" && patches[pi].size > 0 && otherPatch < 0) otherPatch = (int)pi;
    }
    std::printf("test_empty_face_grad: %s/%s  nCells=%d  empty patch=%s (%d faces)\n", caseDir.c_str(), t.c_str(), (int)nC,
                emptyPatch >= 0 ? patches[emptyPatch].name.c_str() : "NONE",
                emptyPatch >= 0 ? (int)patches[emptyPatch].size : 0);
    say(emptyPatch >= 0, "the fixture has a non-empty EMPTY patch (else this gate tests nothing)");
    say(otherPatch >= 0, "...and an ordinary patch to prove the poison reaches the sum");
    if (emptyPatch < 0 || otherPatch < 0) return 1;

    const std::vector<vector> gradOf = readField<vector>(caseDir + "/" + t + "/grad(p)").internalField;
    say((label)gradOf.size() == nC, "OpenFOAM's grad(p) is present for this time and sized to the mesh");
    if ((label)gradOf.size() != nC) return 1;

    const DeviceMesh dm = buildDeviceMesh(m, g, patches);
    DeviceBuffer<scalar> dp(p.internal);

    // ---- the arms against OpenFOAM ------------------------------------------------------------
    const Grad3 h = hostGrad(p.internal, bnd, m, g, patches);
    const Grad3 d = deviceGrad(dm, dp, bnd);
    {
        // the GeometricField overload must be the same function as the (internal, boundary) one
        const std::vector<vector> gv = fvc::gaussGrad(p, m, g, patches);
        scalar dd = 0;
        for (label c = 0; c < nC; ++c) dd = std::fmax(dd, mag(gv[c] - vector{h.x[c], h.y[c], h.z[c]}));
        say(dd == 0.0, "host: the two gaussGrad overloads agree bit for bit");
    }
    const scalar relH = relToOracle(h, gradOf), relD = relToOracle(d, gradOf);
    std::printf("  host   vs OpenFOAM grad(p): rel %.3e\n", relH);
    std::printf("  device vs OpenFOAM grad(p): rel %.3e\n", relD);
    // tests/test_fvc_grad holds the host to 1e-9 on this fixture; measured here 6.835e-15 (host) and
    // 6.837e-15 (device), so the bound is 1e-12: two decades of room, three tighter than the older gate.
    say(relH <= 1e-12, "host   matches OpenFOAM's grad(p) to 1e-12");
    say(relD <= 1e-12, "device matches OpenFOAM's grad(p) to 1e-12");
    {
        const scalar hx = maxDiff(h.x, d.x) / maxAbs(h.x), hy = maxDiff(h.y, d.y) / maxAbs(h.y);
        std::printf("  device vs host: x %.3e  y %.3e  (z max|diff| %.3e)\n", hx, hy, maxDiff(h.z, d.z));
        say(hx <= 1e-12 && hy <= 1e-12, "the two arms agree in-plane to 1e-12");
    }

    // ---- the knocked-out direction ------------------------------------------------------------
    const scalar inPlane = std::fmax(std::fmax(maxAbs(h.x), maxAbs(h.y)), 1e-300);
    scalar ofZ = 0;
    for (const vector& v : gradOf) ofZ = std::fmax(ofZ, std::fabs(v.z));
    std::printf("  max|g_z| / max|g_inplane|:  host %.3e  device %.3e  OpenFOAM %.3e\n",
                maxAbs(h.z) / inPlane, maxAbs(d.z) / inPlane, ofZ / inPlane);
    say(maxAbs(h.z) / inPlane <= 1e-12, "host:   g_z is round-off against the in-plane gradient");
    say(maxAbs(d.z) / inPlane <= 1e-12, "device: g_z is round-off against the in-plane gradient");

    // ---- CONTROL: the empty patch is not in the sum -------------------------------------------
    // One side only (the +z faces), so the two sides of a cell cannot cancel: with the faces in the
    // sum, g_z becomes ~1e30 * Sf_z / V. With them skipped, nothing changes -- to the bit.
    std::vector<std::vector<scalar>> poisoned = bnd;
    {
        const FvPatch& fp = patches[emptyPatch];
        int nPoisoned = 0;
        for (label i = 0; i < fp.size; ++i)
            if (g.Sf()[fp.start + i].z > 0) { poisoned[emptyPatch][i] = 1e30; ++nPoisoned; }
        std::printf("  poisoned %d of %d faces of the empty patch with 1e30\n", nPoisoned, (int)fp.size);
        say(nPoisoned > 0 && nPoisoned < fp.size, "the poison covers one side of the empty patch only");
    }
    const Grad3 hp = hostGrad(p.internal, poisoned, m, g, patches);
    const Grad3 dpz = deviceGrad(dm, dp, poisoned);
    std::printf("  empty-patch poison moved:  host max|dg| = %.3e   device max|dg| = %.3e\n",
                std::fmax(std::fmax(maxDiff(h.x, hp.x), maxDiff(h.y, hp.y)), maxDiff(h.z, hp.z)),
                std::fmax(std::fmax(maxDiff(d.x, dpz.x), maxDiff(d.y, dpz.y)), maxDiff(d.z, dpz.z)));
    say(maxDiff(h.x, hp.x) == 0.0 && maxDiff(h.y, hp.y) == 0.0 && maxDiff(h.z, hp.z) == 0.0,
        "host:   empty-patch values do not enter the gradient (bit-identical)");
    say(maxDiff(d.x, dpz.x) == 0.0 && maxDiff(d.y, dpz.y) == 0.0 && maxDiff(d.z, dpz.z) == 0.0,
        "device: empty-patch values do not enter the gradient (bit-identical)");

    // ---- CONTROL: the poison mechanism reaches the sum on an ordinary patch --------------------
    std::vector<std::vector<scalar>> poisonedOther = bnd;
    poisonedOther[otherPatch][0] = 1e30;
    const Grad3 ho = hostGrad(p.internal, poisonedOther, m, g, patches);
    const Grad3 dov = deviceGrad(dm, dp, poisonedOther);
    const scalar movedH = std::fmax(std::fmax(maxDiff(h.x, ho.x), maxDiff(h.y, ho.y)), maxDiff(h.z, ho.z));
    const scalar movedD = std::fmax(std::fmax(maxDiff(d.x, dov.x), maxDiff(d.y, dov.y)), maxDiff(d.z, dov.z));
    std::printf("  ordinary-patch poison moved:  host %.3e   device %.3e\n", movedH, movedD);
    say(movedH > 0.0, "host:   an ordinary patch's value DOES enter the gradient (poison is live)");
    say(movedD > 0.0, "device: an ordinary patch's value DOES enter the gradient (poison is live)");

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
