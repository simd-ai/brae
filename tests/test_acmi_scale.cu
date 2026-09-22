// cyclicACMI `scale` -- OF cyclicACMIPolyPatch's srcScalePtr_, the PRESCRIBED open-area fraction that
// opens and closes an interface over time:
//
//     scaledMask = min(1 - tol, max(tol, scale(t)*mask))      (cyclicACMIPolyPatch.C:99-116)
//     coupled Sf = Sf_raw * max(tol, scaledMask)
//     wall    Sf = Sf_raw * (1 - min(max(scaledMask, tol), 1 - tol))
//
// so the geometric overlap mask is multiplied by a Function1 of time before the area split. This is how
// pimpleFoam/RAS/TJunctionSwitching switches flow from one branch to the other on a mesh that never
// moves: the bottom branch carries `table ((0 1)(0.2 1)(0.3 0))` and the top the mirror image.
//
// LEG 3 IS THE ONE THAT COST A DAY. The scale belongs to the PAIR, not to a patch. OF writes it on the
// OWNER half only (the lower patch index) and clones it onto the neighbour; a scale on the slave half
// is discarded with a warning. Reading it per patch instead leaves the two halves of ONE interface
// with different open areas -- on TJunctionSwitching the 5-face side of the closed top branch passed
// 3.0e-15 of flux while its 12-face partner passed 3.0e-05, which is mass created out of nothing:
// continuity went 0.108 at step 1 and the run diverged by t = 0.01. With the clone it is 5.8e-05.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ami_interface.cuh"
#include "acmi_area_scaling.cuh"
#include "acmi_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// The two-block ACMI fixture with a `scale` on the OWNER half, then the SAME propagation rule the
// boundary reader applies (propagateACMIScale) -- not a copy of it, which would test the copy.
PrimitiveMesh acmiWithScale(const Function1& f, bool ownerOnly = false)
{
    const PrimitiveMesh src = acmitest::twoBlockACMI(acmitest::ACMI_DY, /*withBlockage=*/true);
    std::vector<PatchInfo> ps = src.patches();
    for (PatchInfo& p : ps)
        if (p.type == "cyclicACMI") { p.acmiScale = f; break; }     // owner = the first ACMI half
    if (!ownerOnly) propagateACMIScale(ps);
    PrimitiveMesh m;
    m.assign(src.points(), src.faceVerts(), src.faceOffsets(), src.owner(), src.neighbour(),
             std::move(ps), src.nCells());
    return m;
}

} // namespace

int main()
{
    std::printf("== cyclicACMI scale ==\n");

    const PrimitiveMesh plain = acmitest::twoBlockACMI(acmitest::ACMI_DY, /*withBlockage=*/true);

    // ---- Leg 1: scale = 1 is the unscaled interface, to within OF's own tolerance clamp -------------
    // NOT bit for bit, and that is OF's doing: the scaled path clamps the mask from ABOVE as well,
    //     scaledMask = min(1 - tol, max(tol, scale*mask))        tol = 1e-10
    // so a fully open face carries (1 - 1e-10) of its area once a `scale` entry exists at all, where an
    // unscaled interface carries all of it. A 1e-10 relative difference is the signature of that clamp;
    // anything larger would mean the scale is doing something it should not.
    {
        PrimitiveMesh m = acmiWithScale(Function1::table({{0.0, 1.0}, {1.0, 1.0}, {2.0, 0.0}}));
        FvGeometry gA, gB;
        std::vector<FvPatch> fA, fB;
        std::vector<AMIInterface> aA, aB;
        buildGeometryPatchesAndAMI(m, gA, fA, aA, 0.0);          // scaled, but scale(0) = 1
        PrimitiveMesh m0 = plain;
        buildGeometryPatchesAndAMI(m0, gB, fB, aB);              // no scale entry at all
        scalar e = 0;
        label nAcmi = 0;
        for (label f = 0; f < m.nFaces(); ++f) e = std::max(e, std::fabs(gA.magSf()[f] - gB.magSf()[f]));
        for (const FvPatch& p : fA) if (p.type == "cyclicACMI") nAcmi += p.size;
        check(nAcmi > 0, "vacuity guard: the fixture has cyclicACMI faces");
        scalar refMax = 0;
        for (label f = 0; f < m.nFaces(); ++f) refMax = std::max(refMax, gB.magSf()[f]);
        check(e/refMax < 2e-10 && e > 0.0,
              "scale(t) = 1 reproduces the unscaled areas to OF's 1e-10 tolerance clamp");
        std::printf("        (relative difference %.3g -- the min(1 - tol, .) clamp)\n", (double)(e/refMax));
    }

    // ---- Leg 2: scale = 0 shuts the interface and hands the area to the blockage wall ---------------
    {
        PrimitiveMesh m = acmiWithScale(Function1::table({{0.0, 1.0}, {1.0, 1.0}, {2.0, 0.0}}));
        FvGeometry g;
        std::vector<FvPatch> fvp;
        std::vector<AMIInterface> amis;
        buildGeometryPatchesAndAMI(m, g, fvp, amis, 2.0);        // scale(2) = 0
        scalar maxCoupled = 0, minWall = 1e300;
        int pairs = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (fvp[pi].type != "cyclicACMI") continue;
            const std::string& wn = m.patches()[pi].nonOverlapPatch;
            label wi = -1;
            for (std::size_t q = 0; q < fvp.size(); ++q) if (fvp[q].name == wn) wi = (label)q;
            if (wi < 0) continue;
            ++pairs;
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label fc = fvp[pi].start + i, fw = fvp[wi].start + i;
                maxCoupled = std::max(maxCoupled, g.magSf()[fc]/g.rawMagSf(fc));
                minWall    = std::min(minWall,    g.magSf()[fw]/g.rawMagSf(fw));
            }
        }
        check(pairs == 2, "vacuity guard: both ACMI halves have a blockage wall");
        check(maxCoupled < 1e-9, "scale = 0 closes every coupled face to the tolerance area");
        check(minWall > 1.0 - 1e-9, "...and the blockage wall takes essentially all of it");
        std::printf("        (closed: coupled fraction <= %.3g, wall fraction >= %.9f)\n",
                    (double)maxCoupled, (double)minWall);
    }

    // ---- Leg 3: the scale belongs to the PAIR, not to one patch -------------------------------------
    // THE DEFECT THIS TEST EXISTS FOR. Read per patch, the closed half shuts while its partner stays
    // open, and the interface manufactures mass. Both halves must end up with the same open area.
    {
        const Function1 f = Function1::table({{0.0, 1.0}, {1.0, 0.0}});
        PrimitiveMesh shared = acmiWithScale(f);                  // propagateACMIScale applied
        PrimitiveMesh owned  = acmiWithScale(f, /*ownerOnly=*/true);   // the pre-fix behaviour

        auto coupledAreas = [](PrimitiveMesh& mm, scalar t)
        {
            FvGeometry g;
            std::vector<FvPatch> fvp;
            std::vector<AMIInterface> amis;
            buildGeometryPatchesAndAMI(mm, g, fvp, amis, t);
            std::vector<scalar> frac;
            for (const FvPatch& p : fvp)
                if (p.type == "cyclicACMI")
                {
                    scalar s = 0, raw = 0;
                    for (label i = 0; i < p.size; ++i) { s += g.magSf()[p.start+i]; raw += g.rawMagSf(p.start+i); }
                    frac.push_back(s/raw);                        // open fraction of this half
                }
            return frac;
        };

        const std::vector<scalar> sh = coupledAreas(shared, 1.0);
        const std::vector<scalar> ow = coupledAreas(owned,  1.0);
        check(sh.size() == 2 && ow.size() == 2, "vacuity guard: two ACMI halves measured");
        check(sh[0] < 1e-9 && sh[1] < 1e-9, "propagated: BOTH halves are shut at scale = 0");
        check(ow[0] < 1e-9 && ow[1] > 1e-3,
              "negative control: without the clone one half shuts and the other stays open");
        std::printf("        (propagated: %.3g / %.3g   owner-only: %.3g / %.3g)\n",
                    (double)sh[0], (double)sh[1], (double)ow[0], (double)ow[1]);
    }

    // ---- Leg 4: the slave's scale is ignored, the owner's wins --------------------------------------
    {
        std::vector<PatchInfo> ps = plain.patches();
        std::size_t o = ps.size(), sl = ps.size();
        for (std::size_t i = 0; i < ps.size(); ++i)
            if (ps[i].type == "cyclicACMI") { if (o == ps.size()) o = i; else if (sl == ps.size()) sl = i; }
        ps[o].acmiScale  = Function1::constant(0.25);
        ps[sl].acmiScale = Function1::constant(0.75);   // OF discards this one
        propagateACMIScale(ps);
        check(std::fabs(ps[o].acmiScale.value(0.0)  - 0.25) < 1e-15 &&
              std::fabs(ps[sl].acmiScale.value(0.0) - 0.25) < 1e-15,
              "both halves take the OWNER's scale when each carries one");
    }

    // ---- Leg 5: no scale entry -> nothing depends on time -------------------------------------------
    {
        PrimitiveMesh m = plain;
        FvGeometry gA, gB;
        std::vector<FvPatch> fA, fB;
        std::vector<AMIInterface> aA, aB;
        buildGeometryPatchesAndAMI(m, gA, fA, aA, 0.0);
        PrimitiveMesh m2 = plain;
        buildGeometryPatchesAndAMI(m2, gB, fB, aB, 1e6);          // absurd time
        scalar e = 0;
        for (label f = 0; f < m.nFaces(); ++f) e = std::max(e, std::fabs(gA.magSf()[f] - gB.magSf()[f]));
        check(e == 0.0, "no scale entry: the areas do not depend on time at all");
        check(!hasACMITimeScale(m), "...and the solver is not asked to rebuild geometry every step");
        PrimitiveMesh ms = acmiWithScale(Function1::table({{0.0, 1.0}, {1.0, 0.0}}));
        check(hasACMITimeScale(ms), "a scaled interface DOES ask for the per-step rebuild");
    }

    // ---- Leg 6: a per-face CODED scale is not this path's -------------------------------------------
    // The boundary reader captures `scale { type coded; }` for the interFoam host loop, which evaluates
    // it (cyclic_acmi_cpp); here acmiScale stays empty, which would read as "no scale" and run the
    // interface fully open. It must refuse, naming the coded scale.
    {
        const PrimitiveMesh src = acmitest::twoBlockACMI(acmitest::ACMI_DY, /*withBlockage=*/true);
        std::vector<PatchInfo> ps = src.patches();
        for (PatchInfo& p : ps)
        {
            if (p.type == "cyclicACMI")
            {
                p.acmiScaleCoded = true;
                p.acmiScaleCodedSpec.name = "scale";
                p.acmiScaleCodedSpec.code = "return tmp<scalarField>::New(this->patch().size(), 1.0);";
                break;
            }
        }
        propagateACMIScale(ps);
        PrimitiveMesh m;
        m.assign(src.points(), src.faceVerts(), src.faceOffsets(), src.owner(), src.neighbour(),
                 std::move(ps), src.nCells());
        check(hasACMITimeScale(m), "a coded scale asks for the per-step rebuild");
        bool refused = false;
        try
        {
            FvGeometry g;
            std::vector<FvPatch> fvp;
            std::vector<AMIInterface> amis;
            buildGeometryPatchesAndAMI(m, g, fvp, amis, 0.0);
        }
        catch (const std::exception& e)
        {
            refused = std::string(e.what()).find("type coded") != std::string::npos;
            std::printf("        (%s)\n", std::string(e.what()).substr(0, 110).c_str());
        }
        check(refused, "a coded scale is refused by name on this path, not run fully open");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
