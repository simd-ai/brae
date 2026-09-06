// EMPTY PATCHES MUST BE INERT, on the host and on the device.
//
// In OpenFOAM an empty patch contributes nothing to any surface sum because it CANNOT: emptyFvPatchField
// is a zero-sized patch field (emptyFvPatchField.C:41), so there are no faces to sum over. brae keeps
// those faces in its addressing, so every operator that sums over boundary faces has to skip them
// explicitly -- gradUBoundary and the vector div always did; the scalar gaussGrad, the tensor gaussGrad
// and the scalar div did not, and neither did the device gradKernel and divKernel.
//
// WHY NOTHING CAUGHT IT, AND WHY THAT IS THE POINT. brae's EmptyPatchField gives every empty face its
// OWNER CELL's value, so a cell's two empty faces contribute (Sf_front + Sf_back)*v_c -- and on an
// exactly-extruded 2D mesh those area vectors cancel. Measured on the fixtures: a cell's empty-face area
// vectors sum to 1.626303e-16 of their own magnitude on rhoBox, 2.065357e-16 on pitzDailyTurb and
// 2.426016e-14 on pitzDaily. So the contribution was round-off, always -- but it was round-off because of
// a MESH property, not because of the semantics, and the host and device agreed with each other rather
// than with OpenFOAM.
//
// THIS GATE ASSERTS THE PROPERTY, NOT A NUMBER. It perturbs the empty patch's boundary values to
// something absurd and requires every operator's output to be BIT-IDENTICAL. That cannot be satisfied by
// a mesh that happens to cancel; it can only be satisfied by not reading those faces. The control is the
// same perturbation on a NON-empty patch, which MUST change the answer -- otherwise the test is
// measuring nothing at all.
//
// STATUS: THIS GATE IS BUILT BUT NOT REGISTERED, AND IT CURRENTLY FAILS. That is deliberate, and the
// numbers behind it are these:
//
//   * the deviation is real -- brae's scalar gaussGrad, tensor gaussGrad and scalar div read empty faces
//     that OpenFOAM does not have, on the host AND on the device;
//   * it is round-off -- the empty contribution to grad(p) on pitzDailyTurb is 4.562325e-12 against a
//     max|grad(p)| of 5.785680e+03, a ratio of 7.886e-16, because brae gives every empty face its owner
//     cell's value and a cell's empty-face area vectors cancel (measured: 1.6e-16 on rhoBox, 2.1e-16 on
//     pitzDailyTurb, 1.7e-12 on backwardFacingStep2D);
//   * and correcting it is NOT free. Applying the skip moved tutorial_backwardfacingstep2d's converged U
//     from 3.482e-03 to 1.145e-02 against a 1e-02 bound -- deterministic, reproducible to every printed
//     digit across reruns, and traced to the DEVICE half alone by bisection. A backward-facing step's
//     reattachment point is precisely the quantity that amplifies 1e-16 over 2000 SIMPLE iterations.
//
// So the choice was: ship a red gate, loosen backwardFacingStep2D's bound to accommodate a change, or
// revert and write down what is known. The second is the failure mode this port exists to avoid, so the
// code is reverted and this file stays as the executable statement of the requirement. What it needs
// before it can be enforced is a convergence-robust comparison on that tutorial -- the project has the
// precedent already: "comparing brae to OF at a tutorial's endTime compares trajectories".
//
// Run: test_empty_patch_inert <caseDir> <timeDir>
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

static void check(const char* what, bool ok)
{
    if (!ok) ++g_fails;
    std::printf("  %-62s %s\n", what, ok ? "OK" : "FAIL");
}

static std::size_t differing(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) if (a[i] != b[i]) ++n;
    return n;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<scalar> p =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    p.evaluateBoundary();

    label emptyPatch = -1, otherPatch = -1;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type == "empty" && emptyPatch < 0) emptyPatch = (label)pi;
        if (fvp[pi].type != "empty" && fvp[pi].size > 0 && otherPatch < 0) otherPatch = (label)pi;
    }
    std::printf("empty-patch inertness  (%d cells)\n", (int)nC);
    check("the fixture HAS an empty patch (else this gate is vacuous)", emptyPatch >= 0);
    check("...and a non-empty one to serve as the control", otherPatch >= 0);
    if (emptyPatch < 0 || otherPatch < 0) { std::printf("FAIL\n"); return 1; }
    std::printf("  empty patch '%s' (%d faces), control patch '%s' (%d faces)\n",
                fvp[emptyPatch].name.c_str(), (int)fvp[emptyPatch].size,
                fvp[otherPatch].name.c_str(), (int)fvp[otherPatch].size);

    // Per-patch boundary arrays: the truth, and the same with one patch poisoned.
    auto boundaryOf = [&](label poison)
    {
        std::vector<std::vector<scalar>> b(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            b[pi] = p.boundary[pi]->value();
            if ((label)pi == poison)
                for (auto& v : b[pi]) v = 1.0e6;   // absurd, and nothing may notice
        }
        return b;
    };
    const std::vector<std::vector<scalar>> bTruth  = boundaryOf(-1);
    const std::vector<std::vector<scalar>> bEmpty  = boundaryOf(emptyPatch);
    const std::vector<std::vector<scalar>> bOther  = boundaryOf(otherPatch);

    auto gradOf = [&](const std::vector<std::vector<scalar>>& b)
    {
        const std::vector<vector> gr = fvc::gaussGrad(p.internal, b, m, g, fvp);
        std::vector<scalar> flat;
        flat.reserve(gr.size() * 3);
        for (const vector& v : gr) { flat.push_back(v.x); flat.push_back(v.y); flat.push_back(v.z); }
        return flat;
    };

    // ---- HOST: fvc::gaussGrad ------------------------------------------------------------------
    std::printf("  1. host fvc::gaussGrad\n");
    {
        const std::vector<scalar> truth = gradOf(bTruth);
        const std::vector<scalar> poisonedEmpty = gradOf(bEmpty);
        const std::vector<scalar> poisonedOther = gradOf(bOther);
        std::printf("     %-58s %zu of %zu\n", "  (components changed by poisoning the EMPTY patch)",
                    differing(truth, poisonedEmpty), truth.size());
        std::printf("     %-58s %zu of %zu\n", "  (…by poisoning the CONTROL patch)",
                    differing(truth, poisonedOther), truth.size());
        check("the empty patch is INERT in gaussGrad", differing(truth, poisonedEmpty) == 0);
        check("the control patch is NOT inert (control)", differing(truth, poisonedOther) > 0);
    }

    // ---- HOST: fvc::div ------------------------------------------------------------------------
    std::printf("  2. host fvc::div\n");
    {
        auto divOf = [&](const std::vector<std::vector<scalar>>& b)
        {
            SurfaceScalarField sf;
            sf.internal.assign(m.nInternalFaces(), 1.0);
            sf.boundary = b;
            return fvc::div(sf, m, g, fvp);
        };
        const std::vector<scalar> truth = divOf(bTruth);
        std::printf("     %-58s %zu of %zu\n", "  (cells changed by poisoning the EMPTY patch)",
                    differing(truth, divOf(bEmpty)), truth.size());
        std::printf("     %-58s %zu of %zu\n", "  (…by poisoning the CONTROL patch)",
                    differing(truth, divOf(bOther)), truth.size());
        check("the empty patch is INERT in div", differing(truth, divOf(bEmpty)) == 0);
        check("the control patch is NOT inert (control)", differing(truth, divOf(bOther)) > 0);
    }

    // ---- DEVICE: deviceGaussGrad and deviceDiv -------------------------------------------------
    // The same property on the other path. Host and device agreeing with EACH OTHER is what hid this
    // for as long as it was hidden, so each is asserted against OpenFOAM's semantics separately.
    std::printf("  3. device deviceGaussGrad / deviceDiv\n");
    {
        const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
        auto flatten = [&](const std::vector<std::vector<scalar>>& b)
        {
            std::vector<scalar> out;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                for (label i = 0; i < fvp[pi].size; ++i) out.push_back(b[pi][i]);
            out.resize(dm.nBndFaces, 0.0);
            return out;
        };
        DeviceBuffer<scalar> dP(p.internal);
        auto devGrad = [&](const std::vector<std::vector<scalar>>& b)
        {
            DeviceBuffer<scalar> db(flatten(b)), gx, gy, gz;
            deviceGaussGrad(dm, dP, db, gx, gy, gz);
            std::vector<scalar> out = gx.host();
            for (scalar v : gy.host()) out.push_back(v);
            for (scalar v : gz.host()) out.push_back(v);
            return out;
        };
        auto devDiv = [&](const std::vector<std::vector<scalar>>& b)
        {
            DeviceBuffer<scalar> db(flatten(b)), di(std::vector<scalar>(dm.nInternalFaces, 1.0)), d;
            deviceDiv(dm, di, db, d);
            return d.host();
        };
        const std::vector<scalar> gTruth = devGrad(bTruth);
        const std::vector<scalar> dTruth = devDiv(bTruth);
        std::printf("     %-58s grad %zu   div %zu\n", "  (changed by poisoning the EMPTY patch)",
                    differing(gTruth, devGrad(bEmpty)), differing(dTruth, devDiv(bEmpty)));
        std::printf("     %-58s grad %zu   div %zu\n", "  (…by poisoning the CONTROL patch)",
                    differing(gTruth, devGrad(bOther)), differing(dTruth, devDiv(bOther)));
        check("the empty patch is INERT in deviceGaussGrad", differing(gTruth, devGrad(bEmpty)) == 0);
        check("the empty patch is INERT in deviceDiv", differing(dTruth, devDiv(bEmpty)) == 0);
        check("deviceGaussGrad reads the control patch (control)", differing(gTruth, devGrad(bOther)) > 0);
        check("deviceDiv reads the control patch (control)", differing(dTruth, devDiv(bOther)) > 0);
    }

    std::printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
