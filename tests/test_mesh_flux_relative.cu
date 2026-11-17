// meshPhi and fvc::makeRelative on a moving mesh -- including the cyclicACMI interface faces.
//
// THE INVARIANT, and it is the one sentence that defines the whole mechanism: move the mesh and the
// fluid TOGETHER and the relative flux must vanish. A fluid travelling with the mesh convects nothing.
// So for a rigid translation at velocity Uw, with the absolute flux set to that of a uniform U = Uw,
//
//     phi_abs(f) - meshPhi(f) == 0     on EVERY face, internal and boundary alike
//
// This is exact, not a tolerance: both sides are the same swept volume computed two ways, so the only
// difference allowed is round-off on the face-centre arithmetic.
//
// WHY THE ACMI FIXTURE. brae's makeRelative reached phiInternal and phiBoundary but never the coupled
// patches, whose flux lives on cyc_/ami_.phi instead. The interface faces of THIS fixture are the ones
// that would have been missed, so leg 2 asserts the invariant holds there too -- a check that passes
// vacuously on any mesh without an interface, which is why it is on this fixture and not a plain box.
//
// WHAT THIS DOES NOT COVER. The defect that actually cost the moving ACMI case its accuracy was the
// ORDER, not the coverage: brae relativised the PREVIOUS step's flux with THIS step's meshPhi at the
// mesh move, where OF relativises the flux it has just computed at the end of pEqn.H. That one is
// verified against OpenFOAM (step-1 velocity 2.2e-02 -> 8.6e-04, and 6.8e-07 over ten steps on the
// laminar moving case) and not here; asserting it needs a solver stepped twice.
//
// THE Y-TRANSLATION IS ALSO THE ANSWER TO "does the interface need this at all?". On
// oscillatingInletACMI2D the channel slides IN the interface plane, so the ACMI faces sweep no volume
// and their meshPhi is 7.4e-17 -- measured, not assumed. Leg 3 therefore translates in X as well, where
// the interface faces DO sweep volume, so the coverage check has something to catch.
#include "acmi_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include "swept_volume.cuh"
#include "foam_dict.cuh"   // isCyclicFamilyType
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

// The absolute flux of a UNIFORM velocity field: phi(f) = Uw . Sf(f).
std::vector<scalar> uniformFlux(const FvGeometry& g, label nFaces, const vector& Uw)
{
    std::vector<scalar> phi(static_cast<std::size_t>(nFaces), 0);
    for (label f = 0; f < nFaces; ++f)
        phi[f] = Uw.x*g.Sf()[f].x + Uw.y*g.Sf()[f].y + Uw.z*g.Sf()[f].z;
    return phi;
}

// Rigidly translate every point by d.
std::vector<vector> translate(const std::vector<vector>& p, const vector& d)
{
    std::vector<vector> q = p;
    for (vector& v : q) { v.x += d.x; v.y += d.y; v.z += d.z; }
    return q;
}

struct Result { scalar worstAll, worstInterface; label nInterfaceFaces; scalar maxInterfaceMeshPhi; };

Result checkTranslation(const vector& dir, const std::string& label_)
{
    const scalar deltaT = 1e-3;
    const vector d{dir.x*deltaT, dir.y*deltaT, dir.z*deltaT};   // so the mesh velocity is exactly `dir`

    PrimitiveMesh m = acmitest::twoBlockACMI(acmitest::ACMI_DY);
    const std::vector<vector> oldPoints = m.points();
    const std::vector<vector> newPoints = translate(oldPoints, d);

    // meshPhi is defined on the moved geometry, and so is Sf -- OF recomputes the geometry in
    // movePoints() before fvc::makeRelative is ever asked for.
    m.movePoints(newPoints);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);

    const std::vector<scalar> mp = meshPhi(m, oldPoints, newPoints, deltaT);
    std::vector<scalar> phi = uniformFlux(g, m.nFaces(), dir);
    makeRelative(phi, mp);   // phi -= meshPhi

    Result r{0, 0, 0, 0};
    for (label f = 0; f < m.nFaces(); ++f) r.worstAll = std::fmax(r.worstAll, std::fabs(phi[f]));
    for (const FvPatch& p : fvp)
    {
        if (!isCyclicFamilyType(p.type)) continue;
        r.nInterfaceFaces += p.size;
        for (label i = 0; i < p.size; ++i)
        {
            r.worstInterface = std::fmax(r.worstInterface, std::fabs(phi[p.start + i]));
            r.maxInterfaceMeshPhi = std::fmax(r.maxInterfaceMeshPhi, std::fabs(mp[p.start + i]));
        }
    }
    std::printf("  %-22s worst |phi - meshPhi| all faces %.3e, interface %.3e   (interface faces %d,"
                " max|meshPhi| there %.3e)\n",
                label_.c_str(), r.worstAll, r.worstInterface, (int)r.nInterfaceFaces, r.maxInterfaceMeshPhi);
    return r;
}

}   // namespace

int main()
{
    // ---- 1. translation IN the interface plane (the oscillatingInletACMI2D motion) ----
    const Result y = checkTranslation(vector{0, 1.57, 0}, "slide in y");
    if (y.worstAll > 1e-13)
    { std::printf("  FAIL y-slide: a fluid moving with the mesh has a non-zero relative flux\n"); ++failures; }
    if (!y.nInterfaceFaces)
    { std::printf("  FAIL vacuous: the fixture has no coupled-interface faces, so nothing here checks them\n"); ++failures; }
    // The measured fact from the real case, reproduced here: a face sliding in its own plane sweeps
    // nothing, so this motion CANNOT detect a missing makeRelative on the interface. Leg 3 exists
    // because of this line.
    if (y.maxInterfaceMeshPhi > 1e-13)
    {
        std::printf("  FAIL expected ~0 interface meshPhi for an in-plane slide, got %.3e -- if this is\n"
                    "       real the swept volume of a face translating in its own plane is wrong\n",
                    y.maxInterfaceMeshPhi);
        ++failures;
    }

    // ---- 2 + 3. translation THROUGH the interface, where those faces do sweep volume ----
    const Result x = checkTranslation(vector{1.31, 0, 0}, "translate through x");
    if (x.worstAll > 1e-13)
    { std::printf("  FAIL x-translation: relative flux non-zero (worst %.3e)\n", x.worstAll); ++failures; }
    if (x.worstInterface > 1e-13)
    {
        std::printf("  FAIL x-translation: the INTERFACE faces keep a relative flux of %.3e -- meshPhi is\n"
                    "       not reaching them, which is what left cyc_/ami_.phi absolute on a moving mesh\n",
                    x.worstInterface);
        ++failures;
    }
    // VACUITY GUARD for leg 2: unless the interface faces genuinely sweep volume under this motion, the
    // assertion above is satisfied by doing nothing at all.
    if (x.maxInterfaceMeshPhi < 1e-6)
    {
        std::printf("  FAIL vacuous: interface meshPhi is only %.3e under the x-translation, so the\n"
                    "       coverage check cannot distinguish a subtracted meshPhi from a skipped one\n",
                    x.maxInterfaceMeshPhi);
        ++failures;
    }

    // ---- 4. a static mesh must be untouched (every non-moving brae case depends on it) ----
    {
        PrimitiveMesh m = acmitest::twoBlockACMI(acmitest::ACMI_DY);
        FvGeometry g;
        g.build(m);
        const std::vector<scalar> mp = meshPhi(m, m.points(), m.points(), 1e-3);   // no displacement
        std::vector<scalar> phi = uniformFlux(g, m.nFaces(), vector{2, 3, 5});
        const std::vector<scalar> before = phi;
        makeRelative(phi, mp);
        scalar worst = 0;
        for (std::size_t f = 0; f < phi.size(); ++f) worst = std::fmax(worst, std::fabs(phi[f] - before[f]));
        std::printf("  %-22s flux change %.3e (must be 0)\n", "static mesh", worst);
        if (worst != scalar(0))
        { std::printf("  FAIL a mesh that did not move changed the flux\n"); ++failures; }
    }

    std::printf("mesh_flux_relative: %d failures\n", failures);
    return failures ? 1 : 0;
}
