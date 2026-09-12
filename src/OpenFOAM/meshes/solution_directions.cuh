#pragma once
// brae::solutionDirections -- OF polyMesh::solutionD() (meshes/polyMesh/polyMesh.C, calcDirections).
//
// Which coordinate directions a vector equation is SOLVED in. fvMatrix<Type>::solveSegregated walks the
// components and `continue`s on every one whose validComponents entry is -1 (fvMatrixSolve.C:157-164),
// and fvMesh::validComponents<vector>() is solutionD() itself (fvMeshTemplates.C:32-44). So on a 2D case
// the empty direction is never solved: its SolverPerformance stays default-constructed, initialResidual
// Zero (SolverPerformance.H:117-121), and the scalar residualControl compares -- cmptMax over that vector
// (solutionControl.C:217-239) -- is the larger of the two solved components.
//
// polyMesh::calcDirections derives solutionD_ from the EMPTY patches alone (polyMesh.C:63-121): the
// componentwise magnitude of every empty face's area vector is summed over all empty patches
// (`emptyDirVec += sum(cmptMag(fa))`, :84), the sum is normalised (:108), and any component above 1e-6
// is knocked out (:112-114). Wedge patches knock out geometricD_ only (polyMesh.C:124-145) -- an
// axisymmetric wedge case SOLVES all three components, so `wedge` is deliberately not looked at here.
//
// Single-rank: the parallel reduce of emptyDirVec (polyMesh.C:104-106) has no counterpart because the
// mirror drivers that use this run undecomposed; parallel_simple.cuh carries its own reduction.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <vector>

namespace brae {

struct SolutionDirections
{
    // polyMesh::solutionD_'s own encoding: +1 solved, -1 knocked out by an empty patch.
    int d[3] = {1, 1, 1};

    bool valid(int cmpt) const { return d[cmpt] > 0; }
};

inline SolutionDirections solutionDirections(const std::vector<FvPatch>& patches)
{
    SolutionDirections sd;
    bool hasEmptyPatches = false;
    scalar emptyDir[3] = {0, 0, 0};
    for (const FvPatch& p : patches)
    {
        // `if (pp.size())` -- polyMesh.C:81: a zero-sized empty patch knocks nothing out.
        if (p.type != "empty" || p.size == 0) continue;
        hasEmptyPatches = true;
        for (label i = 0; i < p.size; ++i)
        {
            // cmptMag(fa) with fa = nf*magSf: FvPatch carries the unit normal and the area separately.
            emptyDir[0] += std::fabs(p.nf[i].x) * p.magSf[i];
            emptyDir[1] += std::fabs(p.nf[i].y) * p.magSf[i];
            emptyDir[2] += std::fabs(p.nf[i].z) * p.magSf[i];
        }
    }
    if (!hasEmptyPatches) return sd;

    // vector::normalise(): a magnitude below ROOTVSMALL leaves Zero (VectorSpaceI.H:485-498), and a
    // zero vector knocks nothing out -- every component then reads +1, exactly polyMesh.C:116-118.
    const scalar mag = std::sqrt(emptyDir[0] * emptyDir[0]
                               + emptyDir[1] * emptyDir[1]
                               + emptyDir[2] * emptyDir[2]);
    const scalar rootVSmall = 1e-150;
    for (int cmpt = 0; cmpt < 3; ++cmpt)
    {
        const scalar n = (mag < rootVSmall) ? scalar(0) : emptyDir[cmpt] / mag;
        sd.d[cmpt] = (n > scalar(1e-6)) ? -1 : 1;
    }
    return sd;
}

} // namespace brae
