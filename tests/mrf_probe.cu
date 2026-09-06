// Residual oracle for the MRF momentum equation, localized.
//
// Assembles UEqn at OpenFOAM's OWN converged fields and splits the initial residual by region. Both
// codes discretise the same equations at the same state, so a residual brae carries and OpenFOAM does
// not is a statement about the discretisation -- and WHERE it sits names the term. This is the same
// method that put 90.5% of the kEpsilon epsilon residual on pitzDaily's inlet.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"
#include "MRF_cpp.cuh"
#include "mrf_read.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <time>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar nu = 1e-5;

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + t, fvSolution.subDict("SIMPLE"), m, g, fvp);
    GeometricField<scalar> nut =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), fvp, nC);
    nut.evaluateBoundary();

    const std::vector<cpu::MRF::ZoneSpec> specs = cpu::MRF::readMRFProperties(caseDir + "/constant");
    const std::map<std::string, std::vector<label>> zoneMap = readCellZones(caseDir + "/constant/polyMesh");
    std::vector<cpu::MRF::Zone> mrf;
    for (const cpu::MRF::ZoneSpec& sp : specs)
    {
        const auto it = zoneMap.find(sp.cellZone);
        if (it != zoneMap.end()) mrf.push_back(cpu::MRF::buildZone(sp, it->second, m, fvp));
    }
    if (!mrf.empty()) cpu::MRF::correctBoundaryVelocity(f.U, mrf, fvp);

    std::vector<scalar> nuEffC(nC);
    for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nut.internal[c];
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& nb = nut.boundary[pi]->value();
        nuEffB[pi].resize(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i) nuEffB[pi][i] = nu + nb[i];
    }

    cpu::MomentumInput mi;
    mi.phi = &f.phi.internal;
    mi.phiBnd = &f.phi.boundary;
    mi.nuEff = &nuEffC;
    mi.nuEffBnd = &nuEffB;
    mi.relaxU = 0.5;
    mi.correctedLaplacian = true;
    mi.bounded = true;
    mi.scheme = cpu::DivScheme::limitedLinearV;
    mi.schemeCoeff = 1.0;
    mi.mrf = mrf.empty() ? nullptr : &mrf;
    mi.hasMRF = !specs.empty();

    FvVectorMatrix M = cpu::assembleUEqn(f.U, mi, m, g, fvp);
    cpu::addPressureGradient(M, f.p, m, g, fvp);

    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<bool>& inZone = mrf.empty() ? std::vector<bool>(nC, false) : mrf[0].inZone;

    for (int cmpt = 0; cmpt < 2; ++cmpt)
    {
        std::vector<scalar> diagC = M.diag, b(nC), psi(nC);
        for (label c = 0; c < nC; ++c)
        {
            b[c]   = component(M.source[c], cmpt);
            psi[c] = component(f.U.internal[c], cmpt);
        }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label c = fvp[pi].faceCells[i];
                diagC[c] += component(M.internalCoeffs[pi][i], cmpt);
                b[c]     += component(M.boundaryCoeffs[pi][i], cmpt);
            }
        }
        std::vector<scalar> yA(nC, 0.0), rA(nC), sumA(nC);
        for (label c = 0; c < nC; ++c) yA[c] = diagC[c] * psi[c];
        for (label fa = 0; fa < nIf; ++fa)
        {
            yA[nei[fa]] += M.lower[fa] * psi[own[fa]];
            yA[own[fa]] += M.upper[fa] * psi[nei[fa]];
        }
        for (label c = 0; c < nC; ++c) rA[c] = b[c] - yA[c];

        for (label c = 0; c < nC; ++c) sumA[c] = diagC[c];
        for (label fa = 0; fa < nIf; ++fa)
        {
            sumA[nei[fa]] += M.lower[fa];
            sumA[own[fa]] += M.upper[fa];
        }
        scalar xRef = 0.0;
        for (label c = 0; c < nC; ++c) xRef += psi[c];
        xRef /= nC;
        scalar normFactor = 1e-20;
        for (label c = 0; c < nC; ++c)
        {
            const scalar tr = sumA[c] * xRef;
            normFactor += std::fabs(yA[c] - tr) + std::fabs(b[c] - tr);
        }

        scalar tot = 0, inZ = 0, outZ = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar a = std::fabs(rA[c]);
            tot += a;
            if (inZone[c]) inZ += a;
            else           outZ += a;
        }
        std::printf("  U%c  initial residual %.6e   (normFactor %.4e)\n",
                    cmpt == 0 ? 'x' : 'y', tot / normFactor, normFactor);
        std::printf("        MRF zone cells   %6zu   %6.2f%%\n", mrf.empty() ? 0 : mrf[0].cells.size(),
                    tot > 0 ? 100.0 * inZ / tot : 0.0);
        std::printf("        outside the zone         %6.2f%%\n", tot > 0 ? 100.0 * outZ / tot : 0.0);
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            scalar s = 0;
            for (label i = 0; i < fvp[pi].size; ++i) s += std::fabs(rA[fvp[pi].faceCells[i]]);
            std::printf("        patch %-12s (%-7s) %6.2f%%\n", fvp[pi].name.c_str(),
                        fvp[pi].type.c_str(), tot > 0 ? 100.0 * s / tot : 0.0);
        }
    }
    return 0;
}
