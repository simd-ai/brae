// END-TO-END _cpp SpalartAllmaras: run a real SA case start to finish on the MIRRORED reference path,
// no CUDA anywhere, and compare against real OpenFOAM.
//
// airFoil2D is the SA tutorial: freestream inlet/outlet, a fixedValue nuTilda wall, and
// `bounded Gauss linearUpwind grad(...)` on BOTH div(phi,U) and div(phi,nuTilda) -- the turbulence
// scalar's own linearUpwind, which no other case in this suite exercises.
//
// THE ORACLE IS REAL OPENFOAM: the case's own converged time directory.
//
// The control (SIMPLE_SA_LAMINAR=1) drops the model and holds nuEff at the molecular value, which is
// what a passing comparison has to be measured against: if the case were insensitive to the closure the
// gate would prove nothing.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "cell_wall_dist.cuh"
#include "simpleFoam_cpp.cuh"
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"
#include "SpalartAllmaras_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static scalar l2rel(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar n = 0, d = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        n += (a[i] - b[i]) * (a[i] - b[i]);
        d += b[i] * b[i];
    }
    return d > 0 ? std::sqrt(n / d) : std::sqrt(n);
}

static scalar l2relVec(const std::vector<vector>& a, const std::vector<vector>& b)
{
    scalar n = 0, d = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        n += magSqr(a[i] - b[i]);
        d += magSqr(b[i]);
    }
    return d > 0 ? std::sqrt(n / d) : std::sqrt(n);
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::printf("usage: %s <caseDir> <startTime> <refTime> <iters>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], startT = argv[2], refT = argv[3];
    const int iters = std::atoi(argv[4]);
    const bool laminar = std::getenv("SIMPLE_SA_LAMINAR") != nullptr;

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    scalar nu = 1e-5;
    {
        const FoamDict tp = readDict(caseDir + "/constant/transportProperties");
        nu = tp.scalarOr("nu", nu);
    }

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    cpu::SimpleControl ctl(cd);
    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + startT, fvSolution.subDict("SIMPLE"),
                                            m, g, patches);

    GeometricField<scalar> nuTilda =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/nuTilda"), patches, nC);
    GeometricField<scalar> nut =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/nut"), patches, nC);
    nuTilda.evaluateBoundary();
    nut.evaluateBoundary();

    scalar relaxU = 0.7, relaxP = 0.3, relaxNut = 0.7;
    if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
    {
        if (const FoamDict* eq = rf->subDict("equations"))
        {
            relaxU   = eq->scalarOr("U", relaxU);
            relaxNut = eq->scalarOr("nuTilda", relaxNut);
        }
        if (const FoamDict* fl = rf->subDict("fields")) relaxP = fl->scalarOr("p", relaxP);
    }

    cpu::TurbulenceState turb;
    turb.k = &nuTilda;       // SA's single transported scalar rides the k slot
    turb.epsilon = nullptr;
    turb.nut = &nut;
    turb.sa = true;
    turb.y = cellWallDist(m, g, patches);
    turb.relaxK = relaxNut;
    turb.tol = 1e-10;
    turb.relTol = 0.0;
    turb.maxIter = 2000;
    turb.boundedTurb = true;
    turb.linearUpwindTurb = true;
    {
        std::string tp = caseDir + "/constant/momentumTransport";
        {
            std::ifstream probe(tp);
            if (!probe.good()) tp = caseDir + "/constant/turbulenceProperties";
        }
        const FoamDict td = readDict(tp);
        cpu::SA::readCoeffs(td.subDict("RAS"), turb.saCoeffs);
    }

    cpu::SA::Residuals sres;
    turb.saRes = &sres;

    cpu::StepInput in;
    in.nu = nu;
    in.turb = laminar ? nullptr : &turb;
    in.relaxU = relaxU;
    in.relaxP = relaxP;
    in.bounded = true;
    in.linearUpwind = true;
    in.scheme = cpu::DivScheme::linearUpwind;
    in.correctedLaplacian = true;
    in.tolU = 1e-10;
    in.tolP = 1e-08;
    if (laminar)
    {
        in.nuEff.assign(nC, nu);
        in.nuEffBnd.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) in.nuEffBnd[pi].assign(patches[pi].size, nu);
    }

    std::printf("test_simple_sa_cpp: %s  %s -> %s, %d iteration(s)%s\n",
                caseDir.c_str(), startT.c_str(), refT.c_str(), iters,
                laminar ? "   [LAMINAR: control]" : "");
    std::printf("  nu %.4g   relax: U %.2f  p %.2f  nuTilda %.2f\n", nu, relaxU, relaxP, relaxNut);
    std::printf("  pRefCell %d  (a p that needs no reference means a patch FIXES p)\n", (int)f.pRefCell);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        std::printf("    patch %-12s (%-8s)  p fixesValue %d  U fixesValue %d  U assignable %d\n",
                    patches[pi].name.c_str(), patches[pi].type.c_str(),
                    (int)f.p.boundary[pi]->fixesValue(), (int)f.U.boundary[pi]->fixesValue(),
                    (int)f.U.boundary[pi]->assignable());

    for (int it = 0; it < iters; ++it)
    {
        const cpu::Residuals res = cpu::simpleStep(f, ctl, in, m, g, patches);
        if (it == 0 || (it + 1) % 100 == 0 || it + 1 == iters)
        {
            scalar ntMax = 0;
            for (label c = 0; c < nC; ++c) ntMax = std::fmax(ntMax, nuTilda.internal[c]);
            std::printf("  it %5d   U %.4e   p %.4e   nuTilda %.4e   max nuTilda %.4e\n", it + 1,
                        res.count("U") ? res.at("U") : -1.0,
                        res.count("p") ? res.at("p") : -1.0, sres.nuTilda, ntMax);
        }
    }

    const std::vector<vector> Uref =
        buildField<vector>(readField<vector>(caseDir + "/" + refT + "/U"), patches, nC).internal;
    const std::vector<scalar> pref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/p"), patches, nC).internal;
    const std::vector<scalar> ntref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/nuTilda"), patches, nC).internal;
    const std::vector<scalar> nutref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/nut"), patches, nC).internal;

    const scalar eU = l2relVec(f.U.internal, Uref);
    const scalar eP = l2rel(f.p.internal, pref);
    const scalar eN = l2rel(nuTilda.internal, ntref);
    const scalar eV = l2rel(nut.internal, nutref);
    std::printf("  vs OpenFOAM %s (L2 rel):  U %.3e  p %.3e  nuTilda %.3e  nut %.3e\n",
                refT.c_str(), eU, eP, eN, eV);

    // Where does the MOMENTUM residual live? Same method that put 90.5% of the kEpsilon epsilon residual
    // on pitzDaily's inlet and 93.8% of the MRF momentum residual on the rotor wall.
    if (std::getenv("SA_LOCALIZE_U"))
    {
        cpu::MomentumInput mi2;
        mi2.phi = &f.phi.internal;
        mi2.phiBnd = &f.phi.boundary;
        std::vector<scalar> nuEffC(nC);
        for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nut.internal[c];
        std::vector<std::vector<scalar>> nuEffB(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& nb = nut.boundary[pi]->value();
            nuEffB[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i) nuEffB[pi][i] = nu + nb[i];
        }
        mi2.nuEff = &nuEffC;
        mi2.nuEffBnd = &nuEffB;
        mi2.relaxU = relaxU;
        mi2.correctedLaplacian = true;
        mi2.bounded = true;
        mi2.linearUpwind = true;
        mi2.scheme = cpu::DivScheme::linearUpwind;
        FvVectorMatrix M2 = cpu::assembleUEqn(f.U, mi2, m, g, patches);
        cpu::addPressureGradient(M2, f.p, m, g, patches);

        const std::vector<label>& own = m.owner();
        const std::vector<label>& nei = m.neighbour();
        const label nIf = m.nInternalFaces();
        for (int cmpt = 0; cmpt < 1; ++cmpt)
        {
            std::vector<scalar> dg = M2.diag, b(nC), psi(nC), yA(nC, 0.0), rA(nC), sumA(nC);
            for (label c = 0; c < nC; ++c)
            {
                b[c] = component(M2.source[c], cmpt);
                psi[c] = component(f.U.internal[c], cmpt);
            }
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const label c = patches[pi].faceCells[i];
                    dg[c] += component(M2.internalCoeffs[pi][i], cmpt);
                    b[c]  += component(M2.boundaryCoeffs[pi][i], cmpt);
                }
            for (label c = 0; c < nC; ++c) yA[c] = dg[c] * psi[c];
            for (label fa = 0; fa < nIf; ++fa)
            {
                yA[nei[fa]] += M2.lower[fa] * psi[own[fa]];
                yA[own[fa]] += M2.upper[fa] * psi[nei[fa]];
            }
            for (label c = 0; c < nC; ++c) rA[c] = b[c] - yA[c];
            for (label c = 0; c < nC; ++c) sumA[c] = dg[c];
            for (label fa = 0; fa < nIf; ++fa)
            {
                sumA[nei[fa]] += M2.lower[fa];
                sumA[own[fa]] += M2.upper[fa];
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
            std::printf("  U%c residual %.6e  (normFactor %.3e)\n", cmpt == 0 ? 'x' : 'y', tot / nf, nf);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                scalar sp = 0;
                for (label i = 0; i < patches[pi].size; ++i) sp += std::fabs(rA[patches[pi].faceCells[i]]);
                std::printf("      patch %-14s (%-7s) %6.2f%%\n", patches[pi].name.c_str(),
                            patches[pi].type.c_str(), tot > 0 ? 100.0 * sp / tot : 0.0);
            }
        }
    }

    // Where does the nuTilda error LIVE? y enters Stilda, fw and the destruction term, and brae's
    // cellWallDist is exact only on the near-wall cells (correctWalls); the interior keeps the meshWave
    // face-centre value. If the error is concentrated far from the wall, that is the lead.
    if (std::getenv("SA_BIN_BY_Y"))
    {
        const std::vector<scalar>& yy = turb.y;
        const scalar edges[6] = {1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e30};
        for (int b = 0; b < 6; ++b)
        {
            const scalar lo = (b == 0) ? 0.0 : edges[b - 1];
            scalar n = 0, d = 0;
            label cnt = 0;
            for (label c = 0; c < nC; ++c)
            {
                if (yy[c] < lo || yy[c] >= edges[b]) continue;
                n += (nuTilda.internal[c] - ntref[c]) * (nuTilda.internal[c] - ntref[c]);
                d += ntref[c] * ntref[c];
                ++cnt;
            }
            if (cnt)
                std::printf("    y in [%8.1e, %8.1e)  %6d cells   nuTilda L2rel %.3e\n",
                            lo, edges[b], (int)cnt, d > 0 ? std::sqrt(n / d) : 0.0);
        }
    }

    const char* bound = std::getenv("SA_CPP_TOL");
    const scalar tol = bound ? std::atof(bound) : 0.0;
    if (tol > 0.0)
    {
        const bool ok = (eU < tol) && (eP < tol) && (eN < tol) && (eV < tol);
        std::printf("  %s (bound %.1e)\n", ok ? "PASS" : "FAIL", tol);
        return ok ? 0 : 1;
    }
    return 0;
}
