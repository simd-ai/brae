// kOmegaSSTLM: the CUDA modules against the _cpp REFERENCE, one at a time.
//
// The device carries the transition model as four cell-local kernels plus two scalar transports, and the
// reference is exposed at exactly the same four boundaries (strain / reThetatPrep / gammaPrep / gammaEff).
// Comparing them stage by stage at IDENTICAL inputs is what makes a disagreement name a stage instead of
// reporting one number for the whole model.
//
// The inputs are OpenFOAM's own converged T3A state, so neither side is being fed the other's answer.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "cell_wall_dist.cuh"
#include "kOmegaSSTLM_cpp.cuh"
#include "komega_sst_coeffs.cuh"
#include "fvc.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static int report(const char* name, const std::vector<scalar>& host, const std::vector<scalar>& dev,
                  scalar bound)
{
    scalar linf = 0, worstH = 0, worstD = 0;
    int worst = -1;
    for (std::size_t c = 0; c < host.size(); ++c)
    {
        const scalar den = std::fmax(std::fabs(host[c]), 1e-30);
        const scalar r = std::fabs(dev[c] - host[c]) / den;
        if (r > linf) { linf = r; worst = (int)c; worstH = host[c]; worstD = dev[c]; }
    }
    const bool ok = linf < bound;
    std::printf("  %-22s L_inf rel %.3e   bound %.1e   %s", name, linf, bound, ok ? "ok\n" : "FAIL\n");
    if (!ok && worst >= 0)
        std::printf("      worst cell %d:  _cpp %.12e   device %.12e\n", worst, worstH, worstD);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <time>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar nu = 1.5e-5;

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    auto rd = [&](const std::string& f) {
        GeometricField<scalar> x =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + f), patches, nC);
        x.evaluateBoundary();
        return x;
    };
    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), patches, nC);
    U.evaluateBoundary();
    GeometricField<scalar> k = rd("k"), omega = rd("omega"), nut = rd("nut");
    GeometricField<scalar> ReThetat = rd("ReThetat"), gammaInt = rd("gammaInt");
    const std::vector<scalar> y = cellWallDist(m, g, patches);

    cpu::kOmegaSSTLM::Coeffs co;
    {
        std::string tp = caseDir + "/constant/momentumTransport";
        { std::ifstream probe(tp); if (!probe.good()) tp = caseDir + "/constant/turbulenceProperties"; }
        const FoamDict turbProps = readDict(tp);
        if (const FoamDict* ras = turbProps.subDict("RAS")) cpu::kOmegaSSTLM::readCoeffs(ras, co);
    }

    // ---- the SHARED input: one gradU, computed once on the host and uploaded, so the stage comparison
    // is about the transition kernels and not about two gradient implementations (which have their own
    // gate). deviceGradU stores the tensor component-major, nine planes of nC.
    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    std::vector<scalar> gradUFlat(9 * nC);
    for (label c = 0; c < nC; ++c)
    {
        const tensor& q = gradU[c];
        const scalar tv[9] = {q.xx, q.xy, q.xz, q.yx, q.yy, q.yz, q.zx, q.zy, q.zz};
        for (int i = 0; i < 9; ++i) gradUFlat[i * nC + c] = tv[i];
    }

    const DeviceMesh dm = buildDeviceMesh(m, g, patches);
    DeviceBuffer<scalar> dGradU, dUx, dUy, dUz, dK, dOm, dY, dRe, dGi;
    dGradU.copyFrom(gradUFlat);
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    dUx.copyFrom(ux); dUy.copyFrom(uy); dUz.copyFrom(uz);
    dK.copyFrom(k.internal); dOm.copyFrom(omega.internal); dY.copyFrom(y);
    dRe.copyFrom(ReThetat.internal); dGi.copyFrom(gammaInt.internal);

    std::printf("lm_cuda_probe: %s/%s  %d cells\n", caseDir.c_str(), t.c_str(), (int)nC);
    int rc = 0;

    // ---- MODULE 1: the strain state and the ReThetat reaction (lmReThetatPrepKernel) ---------------
    const cpu::kOmegaSSTLM::StrainState st = cpu::kOmegaSSTLM::strain(gradU, U.internal, 1e-15);
    const cpu::kOmegaSSTLM::ReThetatPrep rp =
        cpu::kOmegaSSTLM::reThetatPrep(st, k.internal, omega.internal, y,
                                       ReThetat.internal, gammaInt.internal, nu, co);
    DeviceBuffer<scalar> dFth, dSpR, dSuR;
    deviceLMReThetatPrep(dm, dGradU, dUx, dUy, dUz, dK, dOm, dY, dRe, dGi, nu, dFth, dSpR, dSuR);
    rc |= report("Fthetat",            rp.Fthetat, dFth.host(), 1e-12);
    rc |= report("ReThetat sp",        rp.sp,      dSpR.host(), 1e-12);
    rc |= report("ReThetat su",        rp.su,      dSuR.host(), 1e-12);

    // ---- MODULE 2: the intermittency reaction (lmGammaPrepKernel) ---------------------------------
    const cpu::kOmegaSSTLM::GammaPrep gp =
        cpu::kOmegaSSTLM::gammaPrep(st, k.internal, omega.internal, y,
                                    ReThetat.internal, gammaInt.internal, nu, co);
    DeviceBuffer<scalar> dSpG, dSuG;
    deviceLMGammaPrep(dm, dGradU, dUx, dUy, dUz, dK, dOm, dY, dRe, dGi, nu, dSpG, dSuG);
    rc |= report("gammaInt sp",        gp.sp,      dSpG.host(), 1e-12);
    rc |= report("gammaInt su",        gp.su,      dSuG.host(), 1e-12);

    // ---- MODULE 3: the effective intermittency (lmGammaEffKernel) ---------------------------------
    const std::vector<scalar> ge =
        cpu::kOmegaSSTLM::gammaEff(st, k.internal, omega.internal, y,
                                   ReThetat.internal, gammaInt.internal, rp.Fthetat, nu);
    DeviceBuffer<scalar> dGe;
    deviceLMGammaEff(dm, dGradU, dUx, dUy, dUz, dK, dOm, dY, dRe, dGi, dFth, nu, dGe);
    rc |= report("gammaIntEff",        ge,         dGe.host(), 1e-12);

    // ---- MODULE 4: the ReThetat diffusivity -------------------------------------------------------
    std::vector<scalar> DRe(nC);
    for (label c = 0; c < nC; ++c) DRe[c] = co.sigmaThetat * (nut.internal[c] + nu);
    DeviceBuffer<scalar> dNut, dDRe;
    dNut.copyFrom(nut.internal);
    deviceLMReDiff(dNut, nu, dDRe);
    rc |= report("DReThetatEff",       DRe,        dDRe.host(), 1e-12);

    // ---- MODULE 5: the two transport equations, end to end (deviceKOmegaSSTLMCorrect vs
    // cpu::correctReThetatGammaInt). This is where the cell-local stages stop being enough: the
    // convection matrix, the laplacian, the relaxation, the linear solve and Foam::bound all enter, and
    // ReThetat's solved value feeds gammaInt's prep within the same call.
    {
        SurfaceScalarField phi;
        const std::string phiPath = caseDir + "/" + t + "/phi";
        if (std::ifstream(phiPath).good() || std::ifstream(phiPath + ".gz").good())
        {
            const FieldData<scalar> pf = readField<scalar>(phiPath);
            phi.internal = pf.internalField;
            phi.boundary.resize(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                phi.boundary[pi].assign(patches[pi].size, 0.0);
                for (const auto& b : pf.boundary)
                    if (b.name == patches[pi].name && b.hasValue
                        && static_cast<label>(b.values.size()) == patches[pi].size)
                        phi.boundary[pi] = b.values;
            }
        }
        else
        {
            phi = fvc::flux(U, m, g, patches);
        }

        // The host reference, on its OWN fields. GeometricField holds unique_ptr boundary conditions and
        // is deliberately not copyable, so the fresh pair is re-read rather than copied -- which also
        // guarantees the two paths start from identical file data and not from a partly-updated field.
        GeometricField<scalar> hRe = rd("ReThetat"), hGi = rd("gammaInt");
        std::vector<scalar> hGe;
        cpu::kOmegaSSTLM::Residuals hres;
        cpu::kOmegaSSTLM::correctReThetatGammaInt(
            U, k, omega, nut, hRe, hGi, hGe, phi, y, nu, m, g, patches,
            /*relaxReThetat*/0.9, /*relaxGammaInt*/0.9, /*tol*/1e-12, /*relTol*/0.0, /*maxIter*/2000,
            co, &hres, /*bounded*/true, /*limitedLinear*/false, /*linearUpwind*/true, 1.0,
            /*correctedLaplacian*/true, /*snGradLimitCoeff*/0.0);

        // The device, from the same inputs.
        DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, patches, g);
        DeviceBoundary dbRe = buildDeviceBoundary(ReThetat, patches, g);
        DeviceBoundary dbGi = buildDeviceBoundary(gammaInt, patches, g);
        DeviceBuffer<scalar> dRe2, dGi2, dGe2, dNut2, dPhiInt, dPhiBnd;
        dRe2.copyFrom(ReThetat.internal);
        dGi2.copyFrom(gammaInt.internal);
        dNut2.copyFrom(nut.internal);
        dPhiInt.copyFrom(phi.internal);
        {
            std::vector<scalar> flat;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                flat.insert(flat.end(), phi.boundary[pi].begin(), phi.boundary[pi].end());
            dPhiBnd.copyFrom(flat);
        }
        deviceKOmegaSSTLMCorrect(dm, dbU, dbRe, dbGi, dUx, dUy, dUz, dK, dOm, dNut2, dY,
                                 dRe2, dGi2, dGe2, dPhiInt, dPhiBnd, nu,
                                 /*relax*/0.9, /*tol*/1e-12, /*relTolKE*/0.0, /*keCheckEvery*/1,
                                 /*bounded*/true, /*nonOrth*/true, /*gsEps*/false,
                                 nullptr, nullptr, {}, {},
                                 /*limitedLinear*/false, /*linearUpwind*/true);

        // Bounded at 1e-06, not at the 1e-12 the cell-local stages hold: past this point the two paths
        // run DIFFERENT LINEAR SOLVERS (host pbicgstab, device BiCGStab with its own preconditioner) on
        // the same matrix. Both converge to the same solution to within their tolerance, and the gap
        // that remains is that tolerance, not a discretisation difference -- which the stages above
        // establish independently by agreeing to 1e-13 on the same inputs.
        rc |= report("ReThetat (solved)",  hRe.internal, dRe2.host(), 1e-06);
        rc |= report("gammaInt (solved)",  hGi.internal, dGi2.host(), 1e-06);
        rc |= report("gammaIntEff (step)", hGe,          dGe2.host(), 1e-06);
    }

    std::printf("%s\n", rc == 0 ? "  ok:   every kOmegaSSTLM CUDA module matches the _cpp reference"
                                : "  FAIL: a kOmegaSSTLM CUDA module disagrees with the _cpp reference");
    return rc;
}
