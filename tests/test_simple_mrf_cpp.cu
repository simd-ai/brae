// END-TO-END _cpp MRF: run a real multiple-reference-frame case start to finish on the MIRRORED
// reference path, no CUDA anywhere, and compare against real OpenFOAM.
//
// mixerVessel2D is a rotating-zone case (omega 104.72 rad/s about z, cellZone `rotor`) with kEpsilon,
// `bounded Gauss limitedLinearV 1` momentum and `bounded Gauss limitedLinear 1` turbulence.
//
// THE ORACLE IS REAL OPENFOAM: the case's own converged time directory.
//
// The MRF control (SIMPLE_MRF_OFF=1) drops the zones and keeps everything else, so a passing comparison
// is evidence the frame terms are RIGHT rather than evidence the case is insensitive to them.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "simpleFoam_cpp.cuh"
#include "MRF_cpp.cuh"
#include "mrf_read.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    const bool mrfOff = std::getenv("SIMPLE_MRF_OFF") != nullptr;

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");
    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    cpu::SimpleControl ctl(cd);
    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + startT, simpleDict, m, g, patches);

    GeometricField<scalar> k =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/k"), patches, nC);
    GeometricField<scalar> eps =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/epsilon"), patches, nC);
    GeometricField<scalar> nut =
        buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/nut"), patches, nC);
    k.evaluateBoundary();
    eps.evaluateBoundary();
    nut.evaluateBoundary();

    // ---- MRF ------------------------------------------------------------------------------------
    const std::vector<cpu::MRF::ZoneSpec> specs = cpu::MRF::readMRFProperties(caseDir + "/constant");
    const std::map<std::string, std::vector<label>> zones = readCellZones(caseDir + "/constant/polyMesh");
    std::vector<cpu::MRF::Zone> mrf;
    for (const cpu::MRF::ZoneSpec& sp : specs)
    {
        const auto it = zones.find(sp.cellZone);
        if (it == zones.end())
        {
            std::printf("  FAIL: MRF cellZone `%s` is not in constant/polyMesh/cellZones\n",
                        sp.cellZone.c_str());
            return 1;
        }
        mrf.push_back(cpu::MRF::buildZone(sp, it->second, m, patches));
    }
    if (mrfOff) mrf.clear();

    std::printf("test_simple_mrf_cpp: %s  %s -> %s, %d iteration(s)%s\n",
                caseDir.c_str(), startT.c_str(), refT.c_str(), iters, mrfOff ? "   [MRF OFF: control]" : "");
    for (std::size_t i = 0; i < mrf.size(); ++i)
    {
        label nInc = 0, nExc = 0;
        for (const auto& v : mrf[i].includedFaces) nInc += static_cast<label>(v.size());
        for (const auto& v : mrf[i].excludedFaces) nExc += static_cast<label>(v.size());
        std::printf("  zone %zu: Omega (%.4g %.4g %.4g)  cells %zu  internalFaces %zu  included %d  excluded %d\n",
                    i, mrf[i].Omega.x, mrf[i].Omega.y, mrf[i].Omega.z,
                    mrf[i].cells.size(), mrf[i].internalFaces.size(), nInc, nExc);
    }

    scalar relaxU = 0.5, relaxP = 0.3, relaxK = 0.5, relaxEps = 0.5;
    if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
    {
        if (const FoamDict* eq = rf->subDict("equations"))
        {
            relaxU   = eq->scalarOr("U", relaxU);
            relaxK   = eq->scalarOr("k", relaxK);
            relaxEps = eq->scalarOr("epsilon", relaxEps);
        }
        if (const FoamDict* fl = rf->subDict("fields")) relaxP = fl->scalarOr("p", relaxP);
    }

    cpu::TurbulenceState turb;
    turb.k = &k;
    turb.epsilon = &eps;
    turb.nut = &nut;
    turb.relaxK = relaxK;
    turb.relaxEpsilon = relaxEps;
    turb.tol = 1e-10;
    turb.relTol = 0.0;
    turb.maxIter = 2000;
    // div(phi,k) / div(phi,epsilon): `bounded Gauss limitedLinear 1` in this case.
    turb.boundedTurb = true;
    turb.limitedLinearTurb = true;
    turb.turbLimiterCoeff = 1.0;

    cpu::StepInput in;
    in.nu = 1e-5;
    in.turb = &turb;
    // Diagnostic: hold nut at whatever the start-time file carries and never call the model. Run from
    // OpenFOAM's converged state this hands the momentum equation OpenFOAM's OWN eddy viscosity, so any
    // residual left is the momentum/pressure discretisation and not the turbulence closure.
    if (std::getenv("MRF_FROZEN_NUT"))
    {
        in.turb = nullptr;
        in.nuEff.assign(nC, 0.0);
        for (label c = 0; c < nC; ++c) in.nuEff[c] = in.nu + nut.internal[c];
        in.nuEffBnd.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& nb = nut.boundary[pi]->value();
            in.nuEffBnd[pi].assign(patches[pi].size, in.nu);
            for (label i = 0; i < patches[pi].size; ++i) in.nuEffBnd[pi][i] = in.nu + nb[i];
        }
    }
    in.relaxU = relaxU;
    in.relaxP = relaxP;
    in.mrf = mrf.empty() ? nullptr : &mrf;
    in.hasMRF = !specs.empty();
    // `bounded Gauss limitedLinearV 1` momentum, `Gauss linear corrected` laplacian.
    in.bounded = true;
    in.scheme = cpu::DivScheme::limitedLinearV;
    in.schemeCoeff = 1.0;
    in.correctedLaplacian = true;
    in.tolU = 1e-10;
    in.tolP = 1e-08;
    if (mrfOff) in.hasMRF = false;

    for (int it = 0; it < iters; ++it)
    {
        const cpu::Residuals res = cpu::simpleStep(f, ctl, in, m, g, patches);
        if (it == 0 || (it + 1) % 200 == 0 || it + 1 == iters)
        {
            std::printf("  it %5d   U %.4e   p %.4e\n", it + 1,
                        res.count("U") ? res.at("U") : -1.0,
                        res.count("p") ? res.at("p") : -1.0);
        }
    }

    const std::vector<vector> Uref =
        buildField<vector>(readField<vector>(caseDir + "/" + refT + "/U"), patches, nC).internal;
    const std::vector<scalar> pref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/p"), patches, nC).internal;
    const std::vector<scalar> kref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/k"), patches, nC).internal;
    const std::vector<scalar> eref =
        buildField<scalar>(readField<scalar>(caseDir + "/" + refT + "/epsilon"), patches, nC).internal;

    const scalar eU = l2relVec(f.U.internal, Uref);
    const scalar eP = l2rel(f.p.internal, pref);
    const scalar eK = l2rel(k.internal, kref);
    const scalar eE = l2rel(eps.internal, eref);
    std::printf("  vs OpenFOAM %s (L2 rel):  U %.3e  p %.3e  k %.3e  epsilon %.3e\n",
                refT.c_str(), eU, eP, eK, eE);

    const char* bound = std::getenv("MRF_CPP_TOL");
    const scalar tol = bound ? std::atof(bound) : 0.0;
    if (tol > 0.0)
    {
        const bool ok = (eU < tol) && (eP < tol) && (eK < tol) && (eE < tol);
        std::printf("  %s (bound %.1e)\n", ok ? "PASS" : "FAIL", tol);
        return ok ? 0 : 1;
    }
    return 0;
}
