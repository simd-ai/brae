#include "nut_wall_function.cuh"
#include <cmath>

namespace brae {
// yPlusLam: fixed-point solution of yPlusLam = log(E*yPlusLam)/kappa (OF wallFunctionCoefficients).
// EXPORTED rather than file-local: epsilonWallFunction's lowReCorrection branch needs the same
// threshold, and a second transcription of a fixed-point iteration is a second thing to get wrong.
scalar yPlusLam(scalar kappa, scalar E)
{
    scalar ypl = 11.0;
    for (int i = 0; i < 10; ++i) ypl = std::log(std::fmax(E * ypl, 1.0)) / kappa;
    return ypl;
}
namespace {
} // namespace

std::vector<scalar> nutkWallFunction(
    const FvPatch& wall,
    const std::vector<scalar>& y,
    const std::vector<scalar>& kInternal,
    scalar nu,
    scalar Cmu,
    scalar kappa,
    scalar E)
{
    const scalar Cmu25   = std::pow(Cmu, 0.25);
    const scalar yplLam  = yPlusLam(kappa, E);
    std::vector<scalar> nutw(wall.size);
    for (label i = 0; i < wall.size; ++i)
    {
        const scalar kc    = kInternal[wall.faceCells[i]];
        nutw[i] = nutkWallFunctionValue(yPlusWall(Cmu25, y[i], kc, nu), nu, yplLam, kappa, E);
    }
    return nutw;
}

std::vector<scalar> nutkWallFunction(
    const FvPatch& wall,
    const std::vector<scalar>& y,
    const std::vector<scalar>& kInternal,
    const std::vector<scalar>& nuFace,
    scalar Cmu,
    scalar kappa,
    scalar E)
{
    const scalar Cmu25  = std::pow(Cmu, 0.25);
    const scalar yplLam = yPlusLam(kappa, E);
    std::vector<scalar> nutw(wall.size);
    for (label i = 0; i < wall.size; ++i)
    {
        const scalar kc  = kInternal[wall.faceCells[i]];
        const scalar nuw = nuFace[i];
        nutw[i] = nutkWallFunctionValue(yPlusWall(Cmu25, y[i], kc, nuw), nuw, yplLam, kappa, E);
    }
    return nutw;
}

} // namespace brae
