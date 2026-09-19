// permeableAlphaPressureInletOutletVelocity and prghPermeableAlphaTotalPressure, face by face.
//
// WHY A UNIT TEST, when tests/interfoam_permeable_vs_openfoam.sh holds both against OpenFOAM: the
// tutorial names an `alpha`, so the half of each condition that runs WITHOUT one is reached by no gated
// case, and neither is a face that is dry AND takes inflow. What is asserted here is OpenFOAM's own text
// (pressurePermeableAlphaInletOutletVelocityFvPatchVectorField.C:127-178,
// prghPermeableAlphaTotalPressureFvPatchScalarField.C:151-243):
//
//   velocity   refValue = (phi/magSf)*n;  valueFraction = neg(phi);  with an alpha,
//              valueFraction = max(pos(alpha_p - alphaMin), valueFraction) and refValue 0 where that is 1
//   pressure   refValue = p0 - 0.5*rho*neg(phi)*|U|^2 - rho*gh;  refGrad = snGradp;  with an alpha,
//              valueFraction = 1 - pos(alpha_p - alphaMin), and without one it stays at the 0 it was built with
//
//   LEG 1  velocity, dry + outflow: zeroGradient.  dry + inflow: U = 0.  wet, either way: U = 0
//   LEG 2  alpha_p EXACTLY alphaMin is dry -- pos is strict in this OpenFOAM (Scalar.H, s > 0)
//   LEG 3  velocity with `alpha none`: inflow holds (phi/magSf)*n, which is NOT zero; outflow zeroGradient
//   LEG 4  pressure, dry: the total pressure, with the dynamic term on inflow faces ONLY
//   LEG 5  pressure, wet: the cell's value plus snGradp/deltaCoeffs, as fixedFluxPressure
//   LEG 6  pressure with `alpha none`: valueFraction 0 on every face, wet or not
//   LEG 7  THE CONTROL: an inletOutlet of (0 0 0) on the same faces -- the condition this one is when
//          nothing is wet -- differs on the wet outflow face, so leg 1 can fail
//
// BROKEN ONCE EACH: pos made non-strict (>= for >) fails both LEG 2 lines; the dynamic term dropped fails
// LEG 4's inflow line.
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include <cmath>
#include <cstdio>
#include <string>
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

FvPatch makePatch(label n)
{
    FvPatch p;
    p.name = "rightWall";
    p.type = "wall";
    p.size = n;
    for (label i = 0; i < n; ++i)
    {
        p.faceCells.push_back(i);
        p.deltaCoeffs.push_back(scalar(4));
        p.nf.push_back(vector{1, 0, 0});
        p.magSf.push_back(scalar(0.5));
        p.Cf.push_back(vector{0, scalar(i), 0});
    }
    return p;
}

bool same(
    const vector& a,
    const vector& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
}   // namespace

int main()
{
    const FvPatch p = makePatch(5);
    const scalar alphaMin = 0.01;
    // dry+out, dry+in, wet+out, wet+in, and alpha exactly alphaMin with outflow
    const std::vector<scalar> alphap = {0.0, 0.0, 1.0, 1.0, alphaMin};
    const std::vector<scalar> phip = {0.2, -0.2, 0.2, -0.2, 0.2};
    const std::vector<vector> cells(5, vector{3, 1, 0});
    const vector zero{0, 0, 0};

    std::printf("== permeableAlphaPressureInletOutletVelocity ==\n");
    {
        PermeableAlphaPressureInletOutletVelocityPatchField u(p, "alpha.water", alphaMin,
                                                              std::vector<vector>(5, zero));
        u.updateFromAlphaValues(alphap);
        u.updateFromFlux(phip);
        u.evaluate(cells);
        const std::vector<vector>& v = u.value();
        check("LEG 1  dry + outflow is zeroGradient", same(v[0], cells[0]));
        check("LEG 1  dry + inflow holds U = 0", same(v[1], zero));
        check("LEG 1  wet holds U = 0, outflow or inflow", same(v[2], zero) && same(v[3], zero));
        check("LEG 2  alpha_p exactly alphaMin is DRY (pos is strict)", same(v[4], cells[4]));

        PermeableAlphaPressureInletOutletVelocityPatchField bare(p, "none", alphaMin, std::vector<vector>(5, zero));
        check("LEG 3  `alpha none` asks for no phase fraction", !bare.needsAlphaPatchValues());
        bare.updateFromFlux(phip);
        bare.evaluate(cells);
        const vector inflow = vector{1, 0, 0} * (scalar(-0.2) / scalar(0.5));
        check("LEG 3  `alpha none`: inflow holds (phi/magSf)*n, not zero", same(bare.value()[1], inflow));
        check("LEG 3  `alpha none`: a WET outflow face is zeroGradient", same(bare.value()[2], cells[2]));

        // LEG 7
        InletOutletPatchField<vector> io(p, true, zero, {}, std::vector<vector>(5, zero), false);
        io.updateFromFlux(phip);
        io.evaluate(cells);
        std::printf("  an inletOutlet of (0 0 0) on the wet outflow face: (%g %g %g), against (%g %g %g)\n",
                    (double)io.value()[2].x, (double)io.value()[2].y, (double)io.value()[2].z,
                    (double)v[2].x, (double)v[2].y, (double)v[2].z);
        check("LEG 7  CONTROL: an inletOutlet of 0 differs on the wet outflow face, so leg 1 can fail",
              !same(io.value()[2], v[2]));
    }

    std::printf("== prghPermeableAlphaTotalPressure ==\n");
    {
        const scalar p0 = 7.0;
        const std::vector<scalar> rhop = {1.0, 1.0, 1000.0, 1000.0, 1.0};
        const std::vector<scalar> ghp = {-1.0, -2.0, -3.0, -4.0, -5.0};
        const std::vector<vector> Up(5, vector{2, 1, 0});
        const std::vector<scalar> sn = {10.0, 20.0, 30.0, 40.0, 50.0};
        const std::vector<scalar> pc = {100.0, 200.0, 300.0, 400.0, 500.0};

        PrghPermeableAlphaTotalPressurePatchField q(p, p0, "alpha.water", alphaMin, {});
        check("before the first update the value is the constructor's refValue of 1", q.value()[0] == scalar(1));
        q.updateFromAlphaValues(alphap);
        q.updatePermeableTotalPressure(rhop, phip, Up, ghp);
        q.updateSnGrad(sn);
        q.evaluate(pc);
        const std::vector<scalar>& v = q.value();
        check("LEG 4  dry + outflow: p0 - rho*gh, no dynamic term", v[0] == p0 - rhop[0]*ghp[0]);
        check("LEG 4  dry + inflow: p0 - 0.5*rho*|U|^2 - rho*gh",
              v[1] == p0 - scalar(0.5)*rhop[1]*scalar(5) - rhop[1]*ghp[1]);
        check("LEG 5  wet: the cell's value plus snGradp/deltaCoeffs",
              v[2] == pc[2] + sn[2]/scalar(4) && v[3] == pc[3] + sn[3]/scalar(4));
        check("LEG 2  alpha_p exactly alphaMin is DRY here too", v[4] == p0 - rhop[4]*ghp[4]);

        PrghPermeableAlphaTotalPressurePatchField bare(p, p0, "none", alphaMin, {});
        bare.updatePermeableTotalPressure(rhop, phip, Up, ghp);
        bare.updateSnGrad(sn);
        bare.evaluate(pc);
        bool allGradient = true;
        for (std::size_t i = 0; i < 5; ++i)
        {
            allGradient = allGradient && bare.value()[i] == pc[i] + sn[i]/scalar(4);
        }
        check("LEG 6  `alpha none`: valueFraction stays 0, every face takes the gradient", allGradient);
    }

    std::printf("test_permeable_wall_conditions: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
