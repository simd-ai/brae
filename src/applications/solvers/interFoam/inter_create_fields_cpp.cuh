#pragma once
// interFoam's gravity fields -- the host reference.
//
// provenance:
//   openfoam:  applications/solvers/multiphase/interFoam/createFields.H:82-98
//              src/finiteVolume/cfdTools/general/include/readGravitationalAcceleration.H
//              src/finiteVolume/cfdTools/general/include/readhRef.H   (READ_IF_PRESENT)
//              src/finiteVolume/cfdTools/general/include/gh.H:2-12
//   cuda:      src/applications/solvers/interFoam/interCreateFields.cu   (not written yet)
//
//     ghRef = mag(g) > SMALL ? g & (cmptMag(g)/mag(g))*hRef : 0      gh.H:2-7
//     gh    = (g & C)  - ghRef                                      gh.H:11   cell centres
//     ghf   = (g & Cf) - ghRef                                      gh.H:12   face centres
//     p     = p_rgh + rho*gh                                        createFields.H:97
//
// ghRef IS NOT mag(g)*hRef. It is g dotted with the UNIT DIRECTION OF g, times hRef -- so it carries
// g's sign. With g = (0, -9.81, 0) and hRef = h, cmptMag(g)/mag(g) = (0, 1, 0) and ghRef = -9.81*h,
// NEGATIVE. Writing mag(g)*hRef flips it, and the error lands in p through `p = p_rgh + rho*gh`,
// as a uniform offset that looks like a datum choice rather than a bug.
//
// AND IT HIDES ON EVERY TUTORIAL THAT OMITS hRef. readhRef.H is READ_IF_PRESENT with a default of 0,
// and damBreak ships no constant/hRef -- so ghRef is 0 there and the sign cannot be observed. Any gate
// built only on damBreak would pass with the formula written either way, which is why the test below
// supplies a non-zero hRef itself.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

// constant/g -- a uniformDimensionedVectorField, so the vector is under `value`.
inline vector readGravity(const std::string& casePath)
{
    const FoamDict d = readDict(casePath + "/constant/g");
    const std::vector<std::string> v = d.wordListOr("value", {});
    if (v.size() != 3)
        throw std::runtime_error(
            "brae interFoam: constant/g needs `value (gx gy gz);`. interFoam is a buoyant solver -- "
            "without gravity p_rgh and p are the same field and the case is a different problem.");
    return vector{static_cast<scalar>(std::stod(v[0])),
                  static_cast<scalar>(std::stod(v[1])),
                  static_cast<scalar>(std::stod(v[2]))};
}

// constant/hRef -- READ_IF_PRESENT, default 0 (readhRef.H). Absent on every interFoam tutorial that
// ships, which is precisely why its sign is easy to get wrong and hard to notice.
inline scalar readHRef(const std::string& casePath)
{
    try
    {
        const FoamDict d = readDict(casePath + "/constant/hRef");
        return d.scalarOr("value", scalar(0));
    }
    catch (const std::exception&) { return scalar(0); }   // not present: OF's default
}

// gh.H:2-7. The unit direction of g, dotted with g, times hRef -- NOT mag(g)*hRef.
inline scalar ghRef(const vector& g, scalar hRef)
{
    const scalar magG = std::sqrt(g.x*g.x + g.y*g.y + g.z*g.z);
    if (!(magG > scalar(1e-300))) return scalar(0);                 // OF: mag(g) > SMALL
    const vector unitDir{std::fabs(g.x)/magG, std::fabs(g.y)/magG, std::fabs(g.z)/magG};   // cmptMag(g)/mag(g)
    return (g.x*unitDir.x + g.y*unitDir.y + g.z*unitDir.z) * hRef;
}

// gh = (g & C) - ghRef over cell centres; ghf = (g & Cf) - ghRef over face centres. Same expression,
// different geometry, so one function takes either.
inline void ghField(const vector& g, scalar ghRefValue,
                    const std::vector<vector>& centres, std::vector<scalar>& out)
{
    out.resize(centres.size());
    for (std::size_t i = 0; i < centres.size(); ++i)
        out[i] = (g.x*centres[i].x + g.y*centres[i].y + g.z*centres[i].z) - ghRefValue;
}

// p = p_rgh + rho*gh (createFields.H:97). The field interFoam WRITES but never solves.
inline void staticPressure(const std::vector<scalar>& p_rgh,
                           const std::vector<scalar>& rho,
                           const std::vector<scalar>& gh,
                           std::vector<scalar>& p)
{
    p.resize(p_rgh.size());
    for (std::size_t i = 0; i < p_rgh.size(); ++i) p[i] = p_rgh[i] + rho[i]*gh[i];
}

}   // namespace interFoam
}   // namespace cpu
}   // namespace brae
