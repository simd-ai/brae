// fvc::flux and fvc::div on a 2D mesh: an EMPTY patch carries no flux and enters no sum, on both arms.
//
// OpenFOAM has no empty faces at all -- emptyFvPatch::size() is 0 (emptyFvPatch.H:79) -- so a
// surfaceScalarField's patch field there is zero-sized and surfaceIntegrate never visits it. brae keeps
// those faces in its addressing. Its flux stored U_c . Sf on them (U_z*Sf_z on an extruded mesh:
// round-off) and its scalar divergence summed them, on the host and on the device; the tensor div and
// the host boundary gradient already skipped them. Now the flux stores ZERO there and both divergences
// skip them, so no consumer can pick up a number OpenFOAM does not have.
//
//   ORACLE   OpenFOAM's own written phi at the same time: on the empty patch it carries NO values.
//   ARMS     host fvc::flux / fvc::div and device deviceVectorFlux / deviceBoundaryFlux / deviceDiv on
//            the fixture's U. No OpenFOAM oracle for flux(U) itself: a converged phi is the pressure
//            equation's conservative flux, not fvc::flux(U), and the two are not the same field.
//   CONTROL  poisoning the empty patch's velocity (1e30 on the +z side only) must change NEITHER arm's
//            internal flux, boundary flux or divergence by a bit; poisoning one face of an ordinary
//            patch must change both -- so a skip that dropped every boundary would fail too.
//
// FAIL-PROOF, RUN with the four skips removed (host flux + div, device bndFluxKernel + divKernel):
// see the table in REFUSALS.md item 36d.
//
// Run: test_empty_face_flux <caseDir> <timeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_simple.cuh"
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

struct Arms
{
    std::vector<scalar> hInt, hBnd, hDiv;   // host: internal flux, boundary flux (flattened), div
    std::vector<scalar> dInt, dBnd, dDiv;   // device
};

static Arms run(const std::vector<vector>& Uint,
                const std::vector<std::vector<vector>>& Ubnd,
                const PrimitiveMesh& m,
                const FvGeometry& g,
                const std::vector<FvPatch>& patches,
                const DeviceMesh& dm)
{
    Arms a;
    const SurfaceScalarField phi = fvc::flux(Uint, Ubnd, m, g, patches);
    a.hInt = phi.internal;
    a.hBnd = flattenBoundary(phi.boundary);
    a.hDiv = fvc::div(phi, m, g, patches);

    const label nC = m.nCells();
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    {
        ux[c] = Uint[c].x;
        uy[c] = Uint[c].y;
        uz[c] = Uint[c].z;
    }
    std::vector<std::vector<scalar>> bx(patches.size()), by(patches.size()), bz(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (const vector& v : Ubnd[pi])
        {
            bx[pi].push_back(v.x);
            by[pi].push_back(v.y);
            bz[pi].push_back(v.z);
        }
    DeviceBuffer<scalar> dux(ux), duy(uy), duz(uz);
    DeviceBuffer<scalar> dbx(flattenBoundary(bx)), dby(flattenBoundary(by)), dbz(flattenBoundary(bz));
    DeviceBuffer<scalar> phiInt, phiB, dv;
    deviceVectorFlux(dm, dux, duy, duz, phiInt);
    deviceBoundaryFlux(dm, dbx, dby, dbz, phiB);
    deviceDiv(dm, phiInt, phiB, dv);
    a.dInt = phiInt.host();
    a.dBnd = phiB.host();
    a.dDiv = dv.host();
    return a;
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

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), patches, nC);
    U.evaluateBoundary();
    std::vector<std::vector<vector>> bnd;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) bnd.push_back(U.boundary[pi]->value());

    int emptyPatch = -1, otherPatch = -1;
    std::size_t emptyOffset = 0, off = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "empty" && patches[pi].size > 0 && emptyPatch < 0) { emptyPatch = (int)pi; emptyOffset = off; }
        if (patches[pi].type != "empty" && patches[pi].size > 0 && otherPatch < 0) otherPatch = (int)pi;
        off += patches[pi].size;   // flattenBoundary order; no coupled patch on these fixtures
    }
    std::printf("test_empty_face_flux: %s/%s  nCells=%d  empty patch=%s (%d faces)\n", caseDir.c_str(), t.c_str(), (int)nC,
                emptyPatch >= 0 ? patches[emptyPatch].name.c_str() : "NONE",
                emptyPatch >= 0 ? (int)patches[emptyPatch].size : 0);
    say(emptyPatch >= 0, "the fixture has a non-empty EMPTY patch (else this gate tests nothing)");
    say(otherPatch >= 0, "...and an ordinary patch to prove the poison reaches the sums");
    if (emptyPatch < 0 || otherPatch < 0) return 1;

    // ---- ORACLE: OpenFOAM's phi carries nothing on the empty patch ----------------------------
    {
        const FieldData<scalar> phiOf = readField<scalar>(caseDir + "/" + t + "/phi");
        bool found = false, hasValue = true;
        for (const auto& b : phiOf.boundary)
            if (b.name == patches[emptyPatch].name) { found = true; hasValue = b.hasValue && !b.values.empty(); }
        say(found && !hasValue, "OpenFOAM's own phi has NO values on the empty patch (emptyFvPatch::size() == 0)");
    }

    const DeviceMesh dm = buildDeviceMesh(m, g, patches);
    const Arms a = run(U.internal, bnd, m, g, patches, dm);

    // ---- the empty faces read exactly zero on both arms ---------------------------------------
    {
        scalar hz = 0, dz = 0;
        for (label i = 0; i < patches[emptyPatch].size; ++i)
        {
            hz = std::fmax(hz, std::fabs(a.hBnd[emptyOffset + i]));
            dz = std::fmax(dz, std::fabs(a.dBnd[emptyOffset + i]));
        }
        std::printf("  flux on the empty faces:  host max %.3e   device max %.3e\n", hz, dz);
        say(hz == 0.0, "host:   the empty faces carry exactly zero flux");
        say(dz == 0.0, "device: the empty faces carry exactly zero flux");
    }

    // ---- the two arms agree -------------------------------------------------------------------
    {
        const scalar rInt = maxDiff(a.hInt, a.dInt) / maxAbs(a.hInt);
        const scalar rBnd = maxDiff(a.hBnd, a.dBnd) / maxAbs(a.hBnd);
        const scalar rDiv = maxDiff(a.hDiv, a.dDiv) / maxAbs(a.hDiv);
        std::printf("  device vs host: internal flux %.3e  boundary flux %.3e  div %.3e\n", rInt, rBnd, rDiv);
        say(rInt <= 1e-12 && rBnd <= 1e-12 && rDiv <= 1e-12, "the two arms agree to 1e-12 on flux and divergence");
    }

    // ---- CONTROL: poison the empty patch (one side only, so the two sides cannot cancel) --------
    std::vector<std::vector<vector>> poisoned = bnd;
    {
        int nPoisoned = 0;
        const FvPatch& fp = patches[emptyPatch];
        for (label i = 0; i < fp.size; ++i)
            if (g.Sf()[fp.start + i].z > 0) { poisoned[emptyPatch][i] = vector{1e30, 1e30, 1e30}; ++nPoisoned; }
        std::printf("  poisoned %d of %d faces of the empty patch with (1e30 1e30 1e30)\n", nPoisoned, (int)fp.size);
        say(nPoisoned > 0 && nPoisoned < fp.size, "the poison covers one side of the empty patch only");
    }
    const Arms p = run(U.internal, poisoned, m, g, patches, dm);
    const scalar movedH = std::fmax(std::fmax(maxDiff(a.hInt, p.hInt), maxDiff(a.hBnd, p.hBnd)), maxDiff(a.hDiv, p.hDiv));
    const scalar movedD = std::fmax(std::fmax(maxDiff(a.dInt, p.dInt), maxDiff(a.dBnd, p.dBnd)), maxDiff(a.dDiv, p.dDiv));
    std::printf("  empty-patch poison moved:  host %.3e   device %.3e\n", movedH, movedD);
    say(movedH == 0.0, "host:   empty-patch velocity enters neither the flux nor the divergence (bit-identical)");
    say(movedD == 0.0, "device: empty-patch velocity enters neither the flux nor the divergence (bit-identical)");

    // ---- CONTROL: the poison mechanism reaches the sums on an ordinary patch ------------------
    std::vector<std::vector<vector>> poisonedOther = bnd;
    poisonedOther[otherPatch][0] = vector{1e30, 1e30, 1e30};
    const Arms q = run(U.internal, poisonedOther, m, g, patches, dm);
    const scalar oH = std::fmax(maxDiff(a.hBnd, q.hBnd), maxDiff(a.hDiv, q.hDiv));
    const scalar oD = std::fmax(maxDiff(a.dBnd, q.dBnd), maxDiff(a.dDiv, q.dDiv));
    std::printf("  ordinary-patch poison moved:  host %.3e   device %.3e\n", oH, oD);
    say(oH > 0.0, "host:   an ordinary patch's velocity DOES enter the flux and divergence (poison is live)");
    say(oD > 0.0, "device: an ordinary patch's velocity DOES enter the flux and divergence (poison is live)");

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
