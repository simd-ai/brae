// assignable() on the transform patches, and the composition it is half of.
//
// WHY THIS EXISTS. brae's SymmetryPlanePatchField and WedgePatchField return assignable() == FALSE
// where OpenFOAM's return TRUE (transformFvPatchField.H:120, which neither symmetry, symmetryPlane
// nor wedge overrides; only `slip` and `partialSlip` do). That is not a transcription slip, it is one
// half of a pair:
//
//   OpenFOAM   HbyA CARRIES the constraint patch type, so its own symmetry patch evaluates to the
//              tangential projection -- normal flux zero -- and constrainHbyA leaves it alone.
//   brae       HbyA has no patch fields at all; the alternative to taking U's value is the raw CELL
//              value, whose normal component is NOT zero. So it takes U's, and `Sf & U_b` is zero.
//
// Both give a zero normal flux, and the flux is the only thing HbyA's boundary feeds. Correcting
// assignable() alone -- the obvious reading of the class hierarchy -- breaks it: MEASURED,
// RAS/damBreakLeakage then reads U 6.1507e-01, p_rgh 1.0143e+00, alpha 5.4567e-01, and
// LES/nozzleFlow2D fails too.
//
// WHAT IS PINNED HERE is the composition, not either half: on a symmetry, slip or wedge patch the
// value brae's pEqn puts into HbyA has ZERO normal flux. It is written as three arms so that moving
// either half alone fails it:
//   (a) the patch is not assignable, so pEqn takes U's boundary value there;
//   (b) that value's normal flux is zero, to round-off;
//   (c) THE CONTROL: the raw cell value -- what pEqn uses on an assignable patch, and what it would
//       use on these if (a) changed -- has a normal flux that is NOT zero on the same faces.
// (c) is what says (a) is doing work rather than agreeing with the alternative.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;


namespace
{
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf("  %s:   %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
}

int main(int argc, char** argv)
{
    std::printf("== assignable() on a transform patch, and the zero normal flux it composes to ==\n");
    if (argc < 2)
    {
        std::printf("  SKIP: usage: %s <caseDir> [<caseDir>...]\n", argv[0]);
        return 77;
    }

    for (int a = 1; a < argc; ++a)
    {
        const std::string caseDir = argv[a];
        PrimitiveMesh m;
        m.read(caseDir + "/constant/polyMesh");
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        const label nC = m.nCells();

        // a field with a NORMAL component everywhere, so a missing projection cannot hide
        GeometricField<vector> U;
        U.internal.resize(static_cast<std::size_t>(nC));
        for (label c = 0; c < nC; ++c)
        {
            const vector& x = g.C()[c];
            U.internal[c] = vector{1.3 + 0.4*std::sin(x.y), -0.7 + 0.3*std::cos(x.z), 0.9 + 0.2*x.x};
        }
        std::vector<std::string> kind(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const FvPatch& q = fvp[pi];
            if (q.type == "empty")
            {
                U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
                kind[pi] = "empty";
            }
            else if (q.type == "symmetry" || q.type == "symmetryPlane")
            {
                U.boundary.push_back(std::make_unique<SymmetryPlanePatchField<vector>>(q));
                kind[pi] = "transform";
            }
            else if (q.type == "wedge")
            {
                const WedgeGeometry w = wedgeGeometry(q);
                U.boundary.push_back(std::make_unique<WedgePatchField<vector>>(q, w.faceT, w.cellT));
                kind[pi] = "transform";
            }
            else
            {
                U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
                kind[pi] = "other";
            }
        }
        U.evaluateBoundary();

        std::size_t nTransform = 0, nNotAssignable = 0;
        // A FLUX'S OWN SCALE IS |Sf|*|U|, not 1. Bounding it against fmax(|Sf|, 1) made the control's
        // threshold absolute, and on validation/halfChannel -- whose faces are 1e-04 -- a cell-value
        // flux of 4.0e-05 read as "not doing work" against a 1e-03 floor. Both bounds scale now.
        scalar worstFlux = 0, controlFlux = 0, scale = 0, uScale = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (kind[pi] != "transform") continue;
            ++nTransform;
            if (!U.boundary[pi]->assignable()) ++nNotAssignable;
            const FvPatch& q = fvp[pi];
            const std::vector<vector>& ub = U.boundary[pi]->value();
            for (label i = 0; i < q.size; ++i)
            {
                const vector& Sf = g.Sf()[q.start + i];
                const vector& uc = U.internal[static_cast<std::size_t>(q.faceCells[i])];
                const scalar fTaken = Sf.x*ub[static_cast<std::size_t>(i)].x
                                    + Sf.y*ub[static_cast<std::size_t>(i)].y
                                    + Sf.z*ub[static_cast<std::size_t>(i)].z;
                const scalar fCell  = Sf.x*uc.x + Sf.y*uc.y + Sf.z*uc.z;
                worstFlux   = std::fmax(worstFlux, std::fabs(fTaken));
                controlFlux = std::fmax(controlFlux, std::fabs(fCell));
                scale = std::fmax(scale, std::sqrt(Sf.x*Sf.x + Sf.y*Sf.y + Sf.z*Sf.z));
                uScale = std::fmax(uScale, std::sqrt(uc.x*uc.x + uc.y*uc.y + uc.z*uc.z));
            }
        }
        const scalar fluxScale = std::fmax(scale*uScale, scalar(1e-300));
        std::printf("  %s: %zu transform patches, the taken value's normal flux %.3e, the raw cell "
                    "value's %.3e (|Sf|*|U| up to %.3e)\n",
                    caseDir.c_str(), nTransform, (double)worstFlux, (double)controlFlux, (double)fluxScale);
        check("the case carries a transform patch at all", nTransform > 0);
        check("...every one of them is NOT assignable, so pEqn takes U's boundary value there",
              nTransform > 0 && nNotAssignable == nTransform);
        check("...and that value's NORMAL FLUX is zero", worstFlux < scalar(1e-12)*fluxScale);
        check("...while the raw cell value's is NOT: the choice is doing work",
              controlFlux > scalar(1e-3)*fluxScale);
    }

    std::printf("test_symmetry_assignable: %d failures\n", failures);
    return failures ? 1 : 0;
}
