#pragma once
// One linear solve as the solver itself reports it -- OpenFOAM's
//
//     <solver>:  Solving for <field>, Initial residual = X, Final residual = Y, No Iterations N
//
// It is what says whether a solve STOPPED where OpenFOAM's did, which a converged field cannot. At a
// loose tolerance the iterate handed back is wherever that solver left it, and brae has twice run a
// different solver to the same tolerance with every field gate green: PBiCGStab for p_rgh's PCG+DIC
// (the device 9.2e-04 of |U| from OpenFOAM on capillaryRise, its discretisation exact to 3.5e-08) and
// Jacobi-BiCGStab for alpha's symGaussSeidel (3.3e-06 of alpha on damBreak at the case's own 1e-8).
// The drivers keep one of these per solve and the OpenFOAM gates read them beside OpenFOAM's log.
#include "cf_types.cuh"

namespace brae {
namespace cpu {
namespace interFoam {

struct LinearSolveRecord
{
    scalar initialResidual = 0;
    scalar finalResidual = 0;
    int nIterations = 0;
};

} // namespace interFoam
} // namespace cpu
} // namespace brae
