// The DEVICE projection of createFields against the host field set it projects.
//
// The host reference is itself gated against OpenFOAM's own createFields.H
// (tests/rho_createfields_vs_openfoam.sh), so this closes OpenFOAM -> _cpp -> CUDA for the field set.
//
// WHAT IS ACTUALLY AT RISK HERE. An upload is a copy, so the field comparisons below are asserted at
// EXACTLY zero -- a tolerance on a memcpy would be admitting a defect rather than measuring one. What can
// genuinely go wrong is everything around the copy: the boundary-face ORDER, the padding of the faces
// DeviceMesh skips, which of the two boundary masks answers which question, and whether the wall set is
// chosen by the BC or by the patch type. Each of those has its own check, and the two that a fixture can
// make vacuous carry a control that says so.
//
// WHAT THIS GATE CANNOT CATCH TODAY, and it is measured rather than supposed. Deriving `adjustable`
// from `assignable` -- the conflation the module's header warns about -- PASSES on both registered
// fixtures. The two rules disagree only where a patch is non-assignable without fixing a value, which is
// SLIP, and no compressible validation case carries a slip patch on U: rhoBox is fixedValue/zeroGradient/
// noSlip/empty, and sbMatched's inletOutlet outlet reads assignable=1, fixesValue=1, isInletOutlet=1, so
// both rules agree there too. The section below therefore reports itself vacuous instead of passing
// quietly, and the fail-proof is recorded here rather than implied.
//
// THE DISCRIMINATING FIXTURE EXISTS AND IS NOW REGISTERED, just not under validation/. OpenFOAM's own
// compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff gives porosityWall `type slip` on U, and slip
// is non-assignable WITHOUT fixing a value -- measured there as assignable=0, fixesValue=0, so
// !assignable = 1 while !(fixesValue && !IO) = 0. tests/rho_angledduct_structural.sh runs it and reports
// the two masks differing on 1600 faces, with a mask derived from the other wrong on all 1600. That is
// where this conflation is caught; here it still cannot be.
//
// Run: test_rho_createfields_cuda <caseDir> <timeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_dict.cuh"
#include "near_wall_dist.cuh"
#include "rhoCreateFields_cpp.cuh"
#include "rhoCreateFields.cuh"

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

// An upload is a copy. Anything but exact equality is a defect, not a tolerance.
static void exact(const std::vector<scalar>& dev, const std::vector<scalar>& host, const char* nm)
{
    if (dev.size() < host.size())
    {
        std::printf("  %-46s SHORT %zu < %zu  FAIL\n", nm, dev.size(), host.size());
        ++g_fails;
        return;
    }
    std::size_t bad = 0;
    scalar worst = 0;
    for (std::size_t i = 0; i < host.size(); ++i)
    {
        if (dev[i] != host[i]) { ++bad; worst = std::fmax(worst, std::fabs(dev[i] - host[i])); }
    }
    if (bad) ++g_fails;
    std::printf("  %-46s n=%7zu mismatched=%zu worst=%.3e  %s\n",
                nm, host.size(), bad, (double)worst, bad ? "FAIL" : "OK");
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

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    cpu::rhoSimple::RhoSimpleFields hf =
        cpu::rhoSimple::createFields(caseDir + "/" + t, caseDir, simpleDict, &fvSolution, m, g, fvp);

    const gpu::rhoSimple::RhoDeviceFields dev =
        gpu::rhoSimple::createDeviceFields(hf, m, g, fvp);

    std::printf("createFields CUDA projection  (%d cells, %d boundary faces, %s)\n",
                (int)nC, dev.nBndFaces, hf.turbulent ? hf.rasModel.c_str() : "laminar");

    // ---- 1. the cell fields, exactly ----------------------------------------------------------
    std::printf("  1. cell fields\n");
    {
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        { ux[c] = hf.U.internal[c].x; uy[c] = hf.U.internal[c].y; uz[c] = hf.U.internal[c].z; }
        exact(dev.f.Ux.host(), ux, "U.x");
        exact(dev.f.Uy.host(), uy, "U.y");
        exact(dev.f.Uz.host(), uz, "U.z");
    }
    exact(dev.f.p.host(),   hf.p.internal,   "p");
    exact(dev.f.he.host(),  hf.he.internal,  "he");
    exact(dev.f.T.host(),   hf.T.internal,   "T");
    exact(dev.f.rho.host(), hf.rho.internal, "rho");
    exact(dev.f.psi.host(), hf.psi,          "psi");
    exact(dev.f.phiInt.host(), hf.phi.internal, "phi (internal faces)");
    check("initialMass carried through", dev.f.initialMass == hf.initialMass);

    // ---- 2. the boundary ORDER, which is where a copy can still be wrong ----------------------
    // Rebuild the per-patch layout from the flat array and compare against the field's own patches. A
    // transposed or offset gather passes every cell-field check above and fails here.
    std::printf("  2. boundary-face order\n");
    {
        const std::vector<scalar> pb = dev.f.pBnd.host(), rb = dev.f.rhoBnd.host();
        std::size_t k = 0, badP = 0, badR = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& hp = hf.p.boundary[pi]->value();
            const std::vector<scalar>& hr = hf.rho.boundary[pi]->value();
            for (label i = 0; i < fvp[pi].size; ++i, ++k)
            {
                if (pb[k] != hp[i]) ++badP;
                if (rb[k] != hr[i]) ++badR;
            }
        }
        check("p boundary is in patch order", badP == 0);
        check("rho boundary is in patch order", badR == 0);

        // The PADDING. DeviceMesh's boundary gather skips cyclic and processor faces, so nBndFaces can
        // exceed the faces walked above; the tail must carry the documented pad and not stale memory.
        bool padOk = true;
        for (std::size_t i = k; i < rb.size(); ++i) if (rb[i] != scalar(1.0)) padOk = false;
        std::printf("     %-58s walked=%zu of %d\n", "  (padding beyond the walked faces)", k, dev.nBndFaces);
        check("rho pads with 1, not 0 (a divisor must not become 0)", padOk);
    }

    // ---- 3. THE TWO MASKS ANSWER DIFFERENT QUESTIONS ------------------------------------------
    // constrainHbyA asks `assignable`; adjustPhi asks `fixesValue() && !isInletOutlet()`. The two are
    // complements on most patches, and a fixture where they are complements EVERYWHERE cannot tell a
    // correct implementation from one that derived either mask from the other.
    //
    // WHICH PATCH SEPARATES THEM IS MEASURED, NOT ASSUMED. This control first claimed inletOutlet was the
    // discriminator, on the strength of the note in test_rho_peqn_cuda that "slip and inletOutlet are
    // non-assignable without fixing a value". sbMatched's inletOutlet outlet reads assignable=1,
    // fixesValue=1, isInletOutlet=1 -- so !assignable = 0 and !(fixesValue && !IO) = 1 - 1 = 0, and the
    // two AGREE. The claim held for slip and not for inletOutlet. So the disagreeing set is computed
    // from the two rules themselves and the control fires only where one exists.
    std::printf("  3. the two boundary masks\n");
    {
        const std::vector<label> takeU = dev.takeUAtBoundary.host();
        const std::vector<label> adj   = dev.adjustable.host();
        std::size_t differ = 0, k = 0;
        bool separable = false;
        std::string sep;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const bool a  = hf.U.boundary[pi]->assignable();
            const bool fv = hf.U.boundary[pi]->fixesValue();
            const bool io = hf.U.boundary[pi]->isInletOutlet();
            // The two masks disagree on this patch exactly when !assignable != (fixesValue && !IO).
            if ((!a) != (fv && !io)) { separable = true; if (!sep.empty()) sep += ", "; sep += fvp[pi].name; }
            for (label i = 0; i < fvp[pi].size; ++i, ++k)
                if (takeU[k] != (1 - adj[k])) ++differ;
        }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            std::printf("     PATCH %-18s assignable=%d fixesValue=%d inletOutlet=%d\n",
                        fvp[pi].name.c_str(),
                        (int)hf.U.boundary[pi]->assignable(),
                        (int)hf.U.boundary[pi]->fixesValue(),
                        (int)hf.U.boundary[pi]->isInletOutlet());
        std::printf("     %-58s %zu faces\n", "  (faces where the two masks are not complements)", differ);
        if (separable)
        {
            std::printf("     %-58s %s\n", "  (patches that separate the two rules)", sep.c_str());
            check("the masks DIFFER where the two rules disagree (control)", differ > 0);
        }
        else
        {
            std::printf("     %-58s %s\n",
                        "  (no patch separates the two rules -- control vacuous here)", "noted");
            std::printf("     %-58s %s\n",
                        "  (so a mask derived from the other would PASS on this fixture)", "recorded");
        }
        // Whatever the fixture, the masks must at least MATCH the two rules face by face. That is not
        // vacuous anywhere: it is what catches one mask being derived from the other.
        std::size_t wrongT = 0, wrongA = 0;
        k = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const bool a  = hf.U.boundary[pi]->assignable();
            const bool fv = hf.U.boundary[pi]->fixesValue();
            const bool io = hf.U.boundary[pi]->isInletOutlet();
            for (label i = 0; i < fvp[pi].size; ++i, ++k)
            {
                if (takeU[k] != (a ? 0 : 1)) ++wrongT;
                if (adj[k]   != ((fv && !io) ? 0 : 1)) ++wrongA;
            }
        }
        check("takeUAtBoundary is exactly !assignable", wrongT == 0);
        check("adjustable is exactly !(fixesValue && !isInletOutlet)", wrongA == 0);
    }

    // ---- 4. THE WALL SET IS THE BC's, NOT THE PATCH TYPE's -------------------------------------
    // buildDeviceWallData's 4-arg overload falls back to `type == "wall"` alone, and that is the defect
    // that once pinned 545 cells on a far-field wall. Compare the set actually built against the one the
    // type alone would give, and report when a fixture cannot tell them apart.
    if (dev.turbulent)
    {
        std::printf("  4. the wall predicate\n");
        std::size_t byType = 0, byBC = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (fvp[pi].type == "wall") byType += fvp[pi].size;
            if (hf.epsilon.boundary[pi]->isTurbulenceWallFunction()) byBC += fvp[pi].size;
        }
        std::printf("     %-58s type=%zu bc=%zu\n", "  (wall faces by patch type vs by epsilon's BC)",
                    byType, byBC);
        check("the wall set is non-empty (else this section is vacuous)", byBC > 0);

        // The wall-face ORDER recorded for deviceGatherWallNu must be the one buildDeviceWallData used.
        // Gather nearWallDist through it and require it to reproduce the wall data's own wfY exactly.
        const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
        std::vector<scalar> yFlat;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                yFlat.push_back(hf.epsilon.boundary[pi]->isTurbulenceWallFunction() ? yW[pi][i] : scalar(0));
        yFlat.resize(static_cast<std::size_t>(dev.nBndFaces), 0.0);
        const std::vector<scalar> gathered =
            gpu::rhoSimple::gatherWallFaces(yFlat, dev.wfFaceOfBnd);
        exact(gathered, dev.wall.wfY.host(), "wall-face gather order == DeviceWallData's");

        // ...and the per-face mask must agree with the per-cell answer wherever a cell has a wall face.
        const std::vector<label> mask = dev.wfBndMask.host();
        std::size_t k = 0, maskFaces = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i, ++k)
                if (mask[k]) ++maskFaces;
        check("the per-FACE wall mask matches the BC-selected face count", maskFaces == byBC);
    }
    else
    {
        std::printf("  4. laminar -- no wall data, and the closure buffers stay empty\n");
        check("no wall faces built on a laminar case", dev.wall.nWF == 0);
        check("no wall-function face mask built", dev.wfBndMask.size() == 0);
    }

    // ---- 5. THE REFUSAL, with its negative control ---------------------------------------------
    // A coupled patch would be kept out of the LDU entirely, so every equation this field set feeds
    // would lose it silently. The NEGATIVE CONTROL is that the unmodified patch list is accepted --
    // without it this passes whenever construction throws for any reason at all.
    std::printf("  5. refusal\n");
    {
        std::vector<FvPatch> coupled = fvp;
        coupled[0].type = "cyclic";
        bool threw = false;
        try { (void)gpu::rhoSimple::createDeviceFields(hf, m, g, coupled); }
        catch (const std::exception&) { threw = true; }
        check("a coupled patch is REFUSED by name", threw);

        bool threw2 = false;
        try { (void)gpu::rhoSimple::createDeviceFields(hf, m, g, fvp); }
        catch (const std::exception&) { threw2 = true; }
        check("the unmodified mesh is ACCEPTED (negative control)", !threw2);
    }

    std::printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
