#include "dynamic_refine_fv_mesh_cpp.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {
namespace dynamicRefine {

namespace {

const char* const WHO = "brae dynamicRefineFvMesh: ";

// OF doubleScalar.H:58 -- GREAT is 1.0e+15, not VGREAT and not the largest double
constexpr scalar GREAT = scalar(1.0e+15);

}   // namespace


std::vector<scalar> cellToPoint(
    const std::vector<scalar>&             vFld,
    const std::vector<std::vector<label>>& pointCells)
{
    std::vector<scalar> pFld(pointCells.size());
    for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
    {
        const std::vector<label>& pCells = pointCells[pointi];
        if (pCells.empty())
        {
            // OpenFOAM divides by pCells.size() unguarded (dynamicRefineFvMesh.C:768) and would
            // produce a NaN here. A NaN that propagates into a refinement decision is a mesh nobody
            // can explain afterwards, so this names the point instead.
            throw std::runtime_error(
                std::string(WHO) + "point " + std::to_string(pointi) + " has no cells, so the "
                "cell-to-point average has nothing to divide by. OpenFOAM divides by zero here.");
        }
        scalar sum = scalar(0);
        for (const label celli : pCells)
        {
            sum += vFld[static_cast<std::size_t>(celli)];
        }
        pFld[pointi] = sum/static_cast<scalar>(pCells.size());
    }
    return pFld;
}


std::vector<scalar> error(
    const std::vector<scalar>& fld,
    scalar                     minLevel,
    scalar                     maxLevel)
{
    std::vector<scalar> c(fld.size(), scalar(-1));
    for (std::size_t i = 0; i < fld.size(); ++i)
    {
        const scalar err = std::fmin(fld[i] - minLevel, maxLevel - fld[i]);
        if (err >= scalar(0))
        {
            c[i] = err;
        }
    }
    return c;
}


std::vector<scalar> maxPointField(
    const std::vector<scalar>&             pFld,
    const std::vector<std::vector<label>>& pointCells,
    label                                  nCells)
{
    std::vector<scalar> vFld(static_cast<std::size_t>(nCells), -GREAT);
    for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
    {
        for (const label celli : pointCells[pointi])
        {
            scalar& v = vFld[static_cast<std::size_t>(celli)];
            v = std::fmax(v, pFld[pointi]);
        }
    }
    return vFld;
}


std::vector<scalar> maxCellField(
    const std::vector<scalar>&             vFld,
    const std::vector<std::vector<label>>& pointCells)
{
    std::vector<scalar> pFld(pointCells.size(), -GREAT);
    for (std::size_t pointi = 0; pointi < pointCells.size(); ++pointi)
    {
        for (const label celli : pointCells[pointi])
        {
            pFld[pointi] = std::fmax(pFld[pointi], vFld[static_cast<std::size_t>(celli)]);
        }
    }
    return pFld;
}


void selectRefineCandidates(
    scalar                                 lowerRefineLevel,
    scalar                                 upperRefineLevel,
    const std::vector<scalar>&             vFld,
    const std::vector<std::vector<label>>& pointCells,
    label                                  nCells,
    std::vector<char>&                     candidateCell)
{
    if (candidateCell.size() != static_cast<std::size_t>(nCells))
    {
        throw std::runtime_error(
            std::string(WHO) + "the candidate marker is " + std::to_string(candidateCell.size())
            + " long where the mesh has " + std::to_string(nCells) + " cells. OpenFOAM's bitSet is "
            "sized nCells() at the call site (dynamicRefineFvMesh.C:1358) and this function only "
            "ever sets bits -- it does not clear them, and it does not resize.");
    }
    const std::vector<scalar> cellError =
        maxPointField(error(cellToPoint(vFld, pointCells), lowerRefineLevel, upperRefineLevel),
                      pointCells, nCells);
    for (std::size_t celli = 0; celli < cellError.size(); ++celli)
    {
        if (cellError[celli] > scalar(0))
        {
            candidateCell[celli] = 1;
        }
    }
}

}   // namespace dynamicRefine
}   // namespace brae
