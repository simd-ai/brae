// Localize the MOMENTUM residual by patch, on BOTH paths, at OpenFOAM's own converged fields.
//
// The SpalartAllmaras CUDA port converges elsewhere than its _cpp reference, and that was traced to the
// momentum/pressure assembly rather than to SA: at OpenFOAM's converged airFoil2D the CUDA momentum
// residual is 1.34e-04 and its pressure 1.88e-03, against the host's 6.91e-05 and 4.11e-04 (OpenFOAM's
// own are 5.36e-06 and 7.51e-05). This assembles the SAME equation on both paths from the SAME fields and
// splits each residual by region, so the difference can be attributed to a patch instead of guessed at.
//
// The residual is OpenFOAM's: r = b - A psi, normalised by lduMatrix::solver::normFactor.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "UEqn_cpp.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "UEqn.cuh"
#include "linearViscousStress_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

namespace {

// OF's initial residual + a per-cell split, from an already-folded (diag, b) pair.
struct Split
{
    scalar residual = 0;
    std::vector<scalar> perCell;
};

Split residualOf(
    const std::vector<scalar>& diagC,
    const std::vector<scalar>& b,
    const std::vector<scalar>& upper,
    const std::vector<scalar>& lower,
    const std::vector<scalar>& psi,
    const PrimitiveMesh&       m)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    std::vector<scalar> yA(nC, 0.0), rA(nC), sumA(nC);
    for (label c = 0; c < nC; ++c) yA[c] = diagC[c] * psi[c];
    for (label f = 0; f < nIf; ++f)
    {
        yA[nei[f]] += lower[f] * psi[own[f]];
        yA[own[f]] += upper[f] * psi[nei[f]];
    }
    for (label c = 0; c < nC; ++c) rA[c] = b[c] - yA[c];

    for (label c = 0; c < nC; ++c) sumA[c] = diagC[c];
    for (label f = 0; f < nIf; ++f)
    {
        sumA[nei[f]] += lower[f];
        sumA[own[f]] += upper[f];
    }
    scalar xRef = 0;
    for (label c = 0; c < nC; ++c) xRef += psi[c];
    xRef /= nC;
    scalar nf = 1e-20, tot = 0;
    for (label c = 0; c < nC; ++c)
    {
        const scalar tr = sumA[c] * xRef;
        nf += std::fabs(yA[c] - tr) + std::fabs(b[c] - tr);
        tot += std::fabs(rA[c]);
    }
    Split s;
    s.residual = tot / nf;
    s.perCell.resize(nC);
    for (label c = 0; c < nC; ++c) s.perCell[c] = std::fabs(rA[c]);
    return s;
}

void report(const char* who, const Split& s, const std::vector<FvPatch>& fvp, label nC)
{
    scalar tot = 0;
    for (label c = 0; c < nC; ++c) tot += s.perCell[c];
    std::printf("  %-6s Ux residual %.6e\n", who, s.residual);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type == "empty") continue;   // 2D: every cell touches one, so it says nothing
        scalar sp = 0;
        for (label i = 0; i < fvp[pi].size; ++i) sp += s.perCell[fvp[pi].faceCells[i]];
        std::printf("           patch %-14s (%-7s) %6.2f%%\n", fvp[pi].name.c_str(),
                    fvp[pi].type.c_str(), tot > 0 ? 100.0 * sp / tot : 0.0);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <time>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    scalar nu = 1e-5;
    {
        const FoamDict tp = readDict(caseDir + "/constant/transportProperties");
        nu = tp.scalarOr("nu", nu);
    }
    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + t, fvSolution.subDict("SIMPLE"), m, g, fvp);
    GeometricField<scalar> nut =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), fvp, nC);
    nut.evaluateBoundary();

    // Both paths get the SAME state, with the freestream valueFraction brought up to date exactly as an
    // iteration would -- otherwise the comparison measures the boundary seed, not the assembly.
    {
        std::vector<std::vector<vector>> Ub(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) Ub[pi] = f.U.boundary[pi]->value();
        updateMixedFreestream(f.U.boundary, Ub, fvp);
        updateMixedFreestream(f.p.boundary, Ub, fvp);
        f.U.evaluateBoundary();
        f.p.evaluateBoundary();
    }

    std::vector<scalar> nuEffC(nC);
    for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nut.internal[c];
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& nb = nut.boundary[pi]->value();
        nuEffB[pi].resize(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i) nuEffB[pi][i] = nu + nb[i];
    }

    scalar relaxU = 0.7;
    if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
        if (const FoamDict* eq = rf->subDict("equations")) relaxU = eq->scalarOr("U", relaxU);

    // `grad(U) cellLimited Gauss linear <k>`: the gradient linearUpwind's correction is built from.
    // 0 (the default) is the plain Gauss gradient, i.e. the scheme off.
    const char* lk = std::getenv("UEQN_GRADU_LIMIT_K");
    const scalar gradULimitK = lk ? std::atof(lk) : 0.0;
    // `limited <k> corrected` on the laplacian (OF limitedSnGrad); 0 = uncapped.
    const char* sk = std::getenv("UEQN_SNGRAD_LIMIT_K");
    const scalar snGradLimitCoeff = sk ? std::atof(sk) : 0.0;

    std::printf("ueqn_localize: %s/%s   %d cells   relaxU %.2f   gradU cellLimited k=%.3g\n",
                caseDir.c_str(), t.c_str(), static_cast<int>(nC), relaxU, gradULimitK);

    std::vector<scalar> psi(nC);
    for (label c = 0; c < nC; ++c) psi[c] = f.U.internal[c].x;

    // ---- host ----
    {
        cpu::MomentumInput mi;
        mi.phi = &f.phi.internal;
        mi.phiBnd = &f.phi.boundary;
        mi.nuEff = &nuEffC;
        mi.nuEffBnd = &nuEffB;
        mi.relaxU = relaxU;
        mi.correctedLaplacian = true;
        mi.bounded = true;
        mi.linearUpwind = true;
        mi.scheme = cpu::DivScheme::linearUpwind;
        mi.gradULimitK = gradULimitK;
        mi.snGradLimitCoeff = snGradLimitCoeff;
        FvVectorMatrix M = cpu::assembleUEqn(f.U, mi, m, g, fvp);
        cpu::addPressureGradient(M, f.p, m, g, fvp);

        std::vector<scalar> diagC = M.diag, b(nC);
        for (label c = 0; c < nC; ++c) b[c] = M.source[c].x;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label c = fvp[pi].faceCells[i];
                diagC[c] += M.internalCoeffs[pi][i].x;
                b[c]     += M.boundaryCoeffs[pi][i].x;
            }
        report("host", residualOf(diagC, b, M.upper, M.lower, psi, m), fvp, nC);
    }

    // ---- device, from the SAME fields ----
    {
        const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
        const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(f.U, fvp, g);
        const DeviceBoundary dbP = buildDeviceBoundary(f.p, fvp, g);
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        {
            ux[c] = f.U.internal[c].x;
            uy[c] = f.U.internal[c].y;
            uz[c] = f.U.internal[c].z;
        }
        DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dP(f.p.internal);
        DeviceBuffer<scalar> dPhiInt(f.phi.internal);
        std::vector<scalar> pb;
        for (const auto& v : f.phi.boundary) pb.insert(pb.end(), v.begin(), v.end());
        DeviceBuffer<scalar> dPhiBnd(pb);
        DeviceBuffer<scalar> dNuC(nuEffC);
        std::vector<scalar> nbFlat;
        for (const auto& v : nuEffB) nbFlat.insert(nbFlat.end(), v.begin(), v.end());
        DeviceBuffer<scalar> dNuB(nbFlat);
        const SurfaceScalarField nf2 = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
        DeviceBuffer<scalar> dNuF(nf2.internal);

        gpu::MomentumInput mi;
        mi.phiInt = &dPhiInt;
        mi.phiBnd = &dPhiBnd;
        mi.nuEffCell = &dNuC;
        mi.nuEffFace = &dNuF;
        mi.nuEffBndFace = &dNuB;
        mi.relaxU = relaxU;
        mi.correctedLaplacian = true;
        mi.bounded = true;
        mi.linearUpwind = true;
        mi.scheme = cpu::DivScheme::linearUpwind;
        mi.gradULimitK = gradULimitK;
        mi.snGradLimitCoeff = snGradLimitCoeff;

        gpu::MomentumMatrix M;
        gpu::assembleUEqn(M, dm, dbU, dUx, dUy, dUz, mi);
        // -fvc::grad(p)*V, from p's own boundary -- the device takes the gradient, not the field.
        DeviceBuffer<scalar> pbv, gpx, gpy, gpz;
        deviceBCValue(dbP, dP, pbv);
        deviceGaussGrad(dm, dP, pbv, gpx, gpy, gpz);
        gpu::addPressureGradient(M, dm, gpx, gpy, gpz);

        const std::vector<scalar> hdiag = (M.relaxed ? M.relaxedDiag : M.diag).host();
        const std::vector<scalar> hup = M.upper.host(), hlo = M.lower.host();
        const std::vector<scalar> hsrc = M.source[0].host();
        const std::vector<scalar> hiC = M.iC[0].host(), hbC = M.bC[0].host();

        std::vector<scalar> diagC = hdiag, b = hsrc;
        std::size_t j = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            for (label i = 0; i < fvp[pi].size; ++i, ++j)
            {
                if (j >= hiC.size()) break;
                const label c = fvp[pi].faceCells[i];
                diagC[c] += hiC[j];
                b[c]     += hbC[j];
            }
        }
        report("device", residualOf(diagC, b, hup, hlo, psi, m), fvp, nC);
    }
    return 0;
}
