#include "nut_wall_function.cuh"
#include <cmath>
#include <stdexcept>
#include <string>

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

namespace {

// nutkRoughWallFunctionFvPatchScalarField.C:37-55: the roughness function, three regimes by KsPlus (the
// caller divides E by it only when 2.25 < KsPlus). OpenFOAM's shipped binary computes Cs*KsPlus once,
// before the branch, and fuses nothing here; the expressions are kept in its order.
scalar fnRough(
    scalar KsPlus,
    scalar Cs)
{
    if (KsPlus < 90.0)
    {
        return std::pow((KsPlus - 2.25)/87.75 + Cs*KsPlus, std::sin(0.4258*(std::log(KsPlus) - 0.811)));
    }
    return (1.0 + Cs*KsPlus);
}

// Foam::max and Foam::min on doubleScalars: comparisons, as OpenFOAM's binary evaluates them
scalar maxOf(
    scalar a,
    scalar b)
{
    return (a > b) ? a : b;
}

scalar minOf(
    scalar a,
    scalar b)
{
    return (a < b) ? a : b;
}

} // namespace

std::vector<scalar> nutkRoughWallFunction(
    const FvPatch& wall,
    const std::vector<scalar>& y,
    const std::vector<scalar>& kInternal,
    const std::vector<scalar>& nuFace,
    const std::vector<scalar>& nutPrev,
    const std::vector<scalar>& Ks,
    const std::vector<scalar>& Cs,
    scalar Cmu,
    scalar kappa,
    scalar E)
{
    const std::size_t n = static_cast<std::size_t>(wall.size);
    if (y.size() != n || nuFace.size() != n || nutPrev.size() != n || Ks.size() != n || Cs.size() != n)
    {
        throw std::runtime_error(
            "brae nutkRoughWallFunction: patch '" + wall.name + "' has " + std::to_string(n) + " faces and an "
            "input of another length (y, nu, the previous nut, Ks or Cs).");
    }
    // pow025 (Scalar.H:368-371)
    const scalar Cmu25 = std::sqrt(std::sqrt(Cmu));
    std::vector<scalar> nutw(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const scalar uStar = Cmu25*std::sqrt(kInternal[static_cast<std::size_t>(wall.faceCells[i])]);
        const scalar yPlus = uStar*y[i]/nuFace[i];
        const scalar KsPlus = uStar*Ks[i]/nuFace[i];
        scalar Edash = E;
        if (2.25 < KsPlus)
        {
            Edash /= fnRough(KsPlus, Cs[i]);
        }
        const scalar limitingNutw = maxOf(nutPrev[i], nuFace[i]);
        // "To avoid oscillations limit the change in the wall viscosity" (.C:104-106)
        nutw[i] = maxOf
        (
            minOf
            (
                nuFace[i]*(yPlus*kappa/std::log(maxOf(Edash*yPlus, 1+1e-4)) - 1),
                2*limitingNutw
            ),
            0.5*limitingNutw
        );
    }
    return nutw;
}

} // namespace brae
