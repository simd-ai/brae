// pressureInletOutletVelocity's snGrad() is directionMixed's, built from the valueFraction and the cell.
//
// OpenFOAM (directionMixedFvPatchField.C, snGrad):
//     (transform(vf, refValue) + transform(I - vf, pif + refGrad/deltaCoeffs) - pif)*deltaCoeffs
// with this condition's refValue = refGrad = 0 and vf = neg(phi)*(I - nn), Zero until the first
// updateCoeffs (pressureInletOutletVelocityFvPatchVectorField.C:47-49, :180). That is
//     -neg(phi)*(pif - n*(n & pif))*deltaCoeffs
// and the STORED VALUE IS NOT IN IT. brae's class returned (value - pif)*deltaCoeffs, which is the same
// number once the value has been refreshed from the cell and a different one before: at construction
// interFoam's RAS/waterChannel holds the file's (0 0 0) on its atmosphere over cells moving at (1 0 0),
// kOmegaSST's correctNut takes grad(U)'s boundary value from this snGrad, and the patch's nut came out
// 3.7e-05 against OpenFOAM's 3.33e-02 (tests/interfoam_waterchannel_vs_openfoam.sh has the run).
//
//   LEG 1  before any flux is known, and on an outflow or zero-flux face: snGrad is exactly zero
//          whatever value is stored
//   LEG 2  on an inflow face: minus the TANGENTIAL part of the cell velocity, times deltaCoeffs
//   LEG 3  THE CONTROL: the old form, (stored value - cell)*deltaCoeffs, is NOT zero on leg 1's state,
//          so a class that still returned it would fail leg 1
//   LEG 4  once the value has been refreshed (updateFromPatchVelocity), the two forms agree -- which is
//          why every gate that reads this patch after its first evaluation never saw the difference
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

scalar magOf(const vector& v)
{
    return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}
}   // namespace

int main()
{
    std::printf("== pressureInletOutletVelocity::snGrad ==\n");
    // three faces on a top patch, normal +z, 0.25 from their cells
    FvPatch p;
    p.name = "atmosphere";
    p.type = "patch";
    p.size = 3;
    p.faceCells = {0, 1, 2};
    p.deltaCoeffs = {4.0, 4.0, 4.0};
    p.nf = {vector{0, 0, 1}, vector{0, 0, 1}, vector{0, 0, 1}};
    p.magSf = {1.0, 1.0, 1.0};
    p.Cf = {vector{0, 0, 1}, vector{1, 0, 1}, vector{2, 0, 1}};

    const std::vector<vector> cells = {vector{1, 0, 0}, vector{1, 0.5, -0.2}, vector{2, 0, 0.3}};
    // the file's value, as waterChannel ships it
    PressureInletOutletVelocityPatchField<vector> f(p, true, vector{0, 0, 0}, {});

    // LEG 1: no flux yet
    {
        const std::vector<vector> sn = f.snGrad(cells);
        scalar worst = 0;
        for (const vector& v : sn)
        {
            worst = std::fmax(worst, magOf(v));
        }
        check("LEG 1  before the first updateCoeffs the valueFraction is Zero and snGrad is exactly 0",
              sn.size() == 3 && worst == scalar(0));
    }
    // outflow, zero flux, inflow
    f.updateFromFlux({0.7, 0.0, -0.4});
    const std::vector<vector> sn = f.snGrad(cells);
    check("LEG 1  ...and on an outflow face (phi > 0), whatever value is stored", magOf(sn[0]) == scalar(0));
    check("LEG 1  ...and at phi = 0: neg(0) is 0", magOf(sn[1]) == scalar(0));
    // LEG 2: inflow, cell (2 0 0.3), normal z: tangential part (2 0 0), snGrad -(2 0 0)*4
    check("LEG 2  on an inflow face snGrad is -(the cell's tangential velocity)*deltaCoeffs",
          std::fabs(sn[2].x + 8.0) < 1e-15 && std::fabs(sn[2].y) < 1e-15 && std::fabs(sn[2].z) < 1e-15);

    // LEG 3: the old form on the same state
    {
        const std::vector<vector>& val = f.value();
        const vector old0{(val[0].x - cells[0].x)*4.0, (val[0].y - cells[0].y)*4.0, (val[0].z - cells[0].z)*4.0};
        std::printf("  the old form on face 0: |(value - cell)*deltaCoeffs| = %.3f\n", (double)magOf(old0));
        check("LEG 3  CONTROL: (stored value - cell)*deltaCoeffs is NOT zero there, so leg 1 can fail",
              magOf(old0) > scalar(1));
    }
    // LEG 4: refreshed, the two forms agree
    {
        f.updateFromPatchVelocity({}, cells, {});
        const std::vector<vector>& val = f.value();
        const std::vector<vector> snNew = f.snGrad(cells);
        scalar worst = 0;
        for (std::size_t i = 0; i < 3; ++i)
        {
            const vector old{(val[i].x - cells[i].x)*4.0, (val[i].y - cells[i].y)*4.0, (val[i].z - cells[i].z)*4.0};
            worst = std::fmax(worst, magOf(vector{old.x - snNew[i].x, old.y - snNew[i].y, old.z - snNew[i].z}));
        }
        check("LEG 4  once the value is refreshed from the cell the two forms agree", worst < 1e-14);
    }

    std::printf("test_piov_sngrad: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
