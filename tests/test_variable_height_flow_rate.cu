// variableHeightFlowRate and variableHeightFlowRateInletVelocity, face by face.
//
// WHY A UNIT TEST, when tests/interfoam_weiroverflow_vs_openfoam.sh holds both against OpenFOAM: that case
// cannot tell the alpha condition from a zeroGradient. With alpha inside [lowerBound, upperBound] = [0, 1]
// the condition's value is the face cell's on inflow AND on outflow, and OpenFOAM itself goes to NaN when
// an inletOutlet of 0 is put in its place. What the case leaves unseen is exactly what is asserted here,
// against OpenFOAM's own text (variableHeightFlowRateFvPatchField.C:125-164):
//
//     phi < -SMALL   valueFraction 1;  refValue 0 where alpha_c < lowerBound, 1 where alpha_c > upperBound,
//                    else alpha_c
//     otherwise      valueFraction 0;  refValue 0
//
//   LEG 1  before any flux is known the file's `value` stands, and without one the face cell's does
//   LEG 2  inflow inside the bounds FIXES the cell's value: valueInternalCoeffs 0, valueBoundaryCoeffs alpha_c
//   LEG 3  inflow below lowerBound fixes 0, above upperBound fixes 1 -- with bounds that are NOT 0 and 1,
//          which is the half no tutorial reaches
//   LEG 4  outflow, and phi of exactly 0 and of -1e-16 (above -SMALL), are zeroGradient
//   LEG 5  THE CONTROL: an inletOutlet of 0 on the same faces gives a different value on every inflow face
//
// And the velocity condition (variableHeightFlowRateInletVelocityFvPatchVectorField.C:103-139):
//     alpha_p clipped to [0, 1];  avgU = -flowRate/gSum(magSf*alpha_p);  U_p = n*avgU*alpha_p
//   LEG 6  the flux it carries is the prescribed flow rate, through the wet faces only
//   LEG 7  an overshoot and an undershoot in alpha_p are clipped before they are used
//   LEG 8  a dry inlet is refused by name -- OpenFOAM divides by zero there
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>
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
    p.name = "inlet";
    p.type = "patch";
    p.size = n;
    for (label i = 0; i < n; ++i)
    {
        p.faceCells.push_back(i);
        p.deltaCoeffs.push_back(scalar(2));
        p.nf.push_back(vector{-1, 0, 0});
        p.magSf.push_back(scalar(0.5) + scalar(0.25)*scalar(i));
        p.Cf.push_back(vector{0, scalar(i), 0});
    }
    return p;
}
}   // namespace

int main()
{
    std::printf("== variableHeightFlowRate ==\n");
    const FvPatch p = makePatch(6);
    // cells: below the lower bound, inside, above the upper bound, and three more for the flux cases
    const std::vector<scalar> cells = {0.05, 0.4, 0.95, 0.4, 0.4, 0.4};
    const scalar lower = 0.1;
    const scalar upper = 0.9;

    // LEG 1
    {
        VariableHeightFlowRatePatchField withValue(p, lower, upper, std::vector<scalar>(6, scalar(0.25)));
        withValue.evaluate({});
        bool stands = true;
        for (scalar v : withValue.value())
        {
            stands = stands && v == scalar(0.25);
        }
        check("LEG 1  before any flux the file's `value` stands", stands);
        VariableHeightFlowRatePatchField noValue(p, lower, upper, {});
        noValue.evaluate(cells);
        bool extrapolated = true;
        for (std::size_t i = 0; i < 6; ++i)
        {
            extrapolated = extrapolated && noValue.value()[i] == cells[i];
        }
        check("LEG 1  ...and without one the face cell's does (extrapolateInternal)", extrapolated);
    }

    VariableHeightFlowRatePatchField f(p, lower, upper, std::vector<scalar>(6, scalar(0)));
    // inflow on the first three, then outflow, exactly zero, and -1e-16 (above -SMALL = -1e-15)
    f.updateFromFlux({-1.0, -1.0, -1.0, 2.0, 0.0, -1e-16});
    f.evaluate(cells);
    const std::vector<scalar> v = f.value();
    const std::vector<scalar> vic = f.valueInternalCoeffs();
    const std::vector<scalar> vbc = f.valueBoundaryCoeffs();

    check("LEG 2  inflow inside the bounds fixes the CELL's value",
          v[1] == scalar(0.4) && vic[1] == scalar(0) && vbc[1] == scalar(0.4));
    check("LEG 3  inflow below lowerBound 0.1 fixes 0", v[0] == scalar(0) && vic[0] == scalar(0) && vbc[0] == scalar(0));
    check("LEG 3  inflow above upperBound 0.9 fixes 1", v[2] == scalar(1) && vic[2] == scalar(0) && vbc[2] == scalar(1));
    check("LEG 4  outflow is zeroGradient", v[3] == scalar(0.4) && vic[3] == scalar(1) && vbc[3] == scalar(0));
    check("LEG 4  phi of exactly 0 is zeroGradient", v[4] == scalar(0.4) && vic[4] == scalar(1));
    check("LEG 4  phi of -1e-16, above -SMALL, is zeroGradient too", v[5] == scalar(0.4) && vic[5] == scalar(1));

    // LEG 5
    {
        InletOutletPatchField<scalar> io(p, true, scalar(0), {}, std::vector<scalar>(6, scalar(0)), false);
        io.updateFromFlux({-1.0, -1.0, -1.0, 2.0, 0.0, -1e-16});
        io.evaluate(cells);
        const bool differs = io.value()[1] != v[1] && io.value()[2] != v[2];
        std::printf("  an inletOutlet of 0 on face 1: %.3f, against %.3f\n", (double)io.value()[1], (double)v[1]);
        check("LEG 5  CONTROL: an inletOutlet of 0 gives another value on the inflow faces, so legs 2-3 can fail",
              differs);
    }

    std::printf("== variableHeightFlowRateInletVelocity ==\n");
    const scalar flowRate = 3.0;
    VariableHeightFlowRateInletVelocityPatchField u(p, Function1::constant(flowRate), "alpha.water", true,
                                                    vector{0, 0, 0}, {});
    // dry, wet, half, an overshoot and an undershoot, dry
    const std::vector<scalar> alphap = {0.0, 1.0, 0.5, 1.2, -0.3, 0.0};
    u.updateFromAlphaPatch(alphap, scalar(0));
    {
        // U_b = n*avgU*a with a the clipped alpha, so sum(U_b & n * magSf) is avgU*sum(magSf*a) = -flowRate
        scalar carried = 0;
        scalar dryFlux = 0;
        for (label i = 0; i < p.size; ++i)
        {
            const vector& ub = u.value()[static_cast<std::size_t>(i)];
            const scalar flux = (ub.x*p.nf[i].x + ub.y*p.nf[i].y + ub.z*p.nf[i].z) * p.magSf[i];
            carried += flux;
            if (alphap[static_cast<std::size_t>(i)] <= scalar(0))
            {
                dryFlux += std::fabs(flux);
            }
        }
        std::printf("  carried through the patch: %.15g for a flowRate of %.15g\n", (double)carried, (double)flowRate);
        check("LEG 6  the dry faces carry nothing", dryFlux == scalar(0));
        check("LEG 6  the patch carries exactly the prescribed rate, inward", std::fabs(carried + flowRate) < 1e-13);
    }
    {
        // the overshoot face must behave as alpha 1 and the undershoot face as alpha 0
        const scalar uWet = u.value()[1].x;
        const scalar uOver = u.value()[3].x;
        const scalar uUnder = u.value()[4].x;
        check("LEG 7  an overshoot of 1.2 is used as 1 and an undershoot of -0.3 as 0",
              uOver == uWet && uUnder == scalar(0));
    }
    {
        bool named = false;
        try
        {
            u.updateFromAlphaPatch(std::vector<scalar>(6, scalar(0)), scalar(0));
        }
        catch (const std::exception& e)
        {
            named = std::string(e.what()).find("holds no water") != std::string::npos;
        }
        check("LEG 8  a dry inlet is refused by name", named);
    }

    std::printf("test_variable_height_flow_rate: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
