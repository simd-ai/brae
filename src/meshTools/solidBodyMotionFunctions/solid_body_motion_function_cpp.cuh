#pragma once
// OpenFOAM's solidBodyMotionFunctions: a rigid transformation of the mesh as a function of time.
//
// provenance:
//   openfoam:  src/meshTools/solidBodyMotionFunctions/solidBodyMotionFunction/
//                  solidBodyMotionFunctionNew.C (the selector; the name is a LITERAL lookup),
//                  solidBodyMotionFunction.C:65 (SBMFCoeffs_ = dict.optionalSubDict(type + "Coeffs"))
//              .../linearMotion/linearMotion.C, .../oscillatingLinearMotion/oscillatingLinearMotion.C,
//              .../rotatingMotion/rotatingMotion.C, .../axisRotationMotion/axisRotationMotion.C,
//              .../oscillatingRotatingMotion/oscillatingRotatingMotion.C, .../SDA/SDA.C,
//              .../tabulated6DoFMotion/tabulated6DoFMotion.C, .../multiMotion/multiMotion.C
//              src/OpenFOAM/interpolations/interpolateSplineXY/interpolateSplineXY.C
//   tests:     tests/test_mesh_motion_vs_openfoam.cu
//
// EVERY ONE IS AN ABSOLUTE FUNCTION OF TIME applied to the mesh's ORIGINAL points, never an increment
// on the current ones. multiMotion is the PRODUCT of its entries' septernions in the order the
// dictionary lists them, and the order is part of the answer: testTubeMixer tilts a box about its own
// origin ON a turntable, and the other order tilts the turntable.
//
// WHAT IS REFUSED, by name: drivenLinearMotion (its displacement is another object's field), and any
// coefficient OpenFOAM reads as a Function1 when the case writes something other than a constant --
// rotatingMotion's omega, and oscillatingLinearMotion's amplitude, omega, phaseShift and
// verticalShift. A constant omega integrates to omega*t; a table does not.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "septernion_cpp.cuh"
#include <memory>
#include <string>

namespace brae {

class SolidBodyMotionFunction
{
public:
    virtual ~SolidBodyMotionFunction() = default;
    // the septernion at time t; transformPoints() applies it to the original points
    virtual Septernion transformation(scalar t) const = 0;
    virtual std::string type() const = 0;

    // solidBodyMotionFunction::New(dict, time): reads `solidBodyMotionFunction` from `dict` and the
    // function's coefficients from dict.optionalSubDict(<type>Coeffs). `caseDir` resolves
    // tabulated6DoFMotion's `<constant>/...` file name.
    static std::unique_ptr<SolidBodyMotionFunction> New(
        const FoamDict& dict,
        const std::string& caseDir);
};

} // namespace brae
