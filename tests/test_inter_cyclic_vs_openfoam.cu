// brae interFoam on a FULLY PERIODIC mesh against REAL OpenFOAM's, on validation/interFoamCyclic.
//
// WHAT THIS CASE EXERCISES that the baffle gate (tests/interfoam_baffle_vs_openfoam.sh) does not: the
// pair is the DOMAIN's x boundary, not an internal baffle, so every field advects across it for the
// whole run and there is no wall behind it to damp what the coupling gets wrong. The water surface is
// staged as a STEP at that boundary -- 0.3 on one side, 0.15 on the other -- so alpha, its gradient,
// nHatf and the pressure all carry a jump through the pair from the first step.
//
// THE ORACLE is OpenFOAM's own fields at the same instant, ten FIXED steps of 2e-3 in, written at 17
// digits. THE CONTROL is OpenFOAM's own answer with the pair replaced by two WALLS: the same mesh,
// the same cells, the coupling gone.
//
// 800 cells on purpose. This fixture exists so the CYCLIC DEVICE PATH can be run end to end cheaply,
// and the device arm's state is asserted here too.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "cyclic_interface.cuh"
#include "inter_driver_cpp.cuh"
#include "inter_solve_log.cuh"
#include "device_gate_finite.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

// MEASURED, ten steps of 2e-3 (the bounds are about 30x):
//   alpha 1.9e-13, p_rgh 9.8e-12 relative, U 7.5e-13 relative, all 30 p_rgh iteration counts OpenFOAM's
const scalar B_ALPHA = 6e-12;
const scalar B_PRGH = 3e-10;
const scalar B_U = 2e-11;
// ...and the DEVICE arm's (about 30x again). MEASURED against OpenFOAM: alpha 4.2e-11, p_rgh 1.98e-11
// relative, U 5.7e-11 relative; against brae's own host arm 4.2e-11, 1.02e-11, 5.8e-11. The alpha
// figure is the device alpha solver's own stopping point -- with every solve pinned at 1e-16 the two
// arms' alpha agrees to 6.1e-14 -- and not a discretisation difference.
const scalar B_ALPHA_DEV = 1.5e-09;
const scalar B_PRGH_DEV = 6e-10;
const scalar B_U_DEV = 2e-09;
const scalar B_ALPHA_DEV_HOST = 1.5e-09;
const scalar B_PRGH_DEV_HOST = 6e-10;
const scalar B_U_DEV_HOST = 2e-09;

namespace {
int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

struct Diff
{
    scalar linf = 0;
    scalar refMax = 0;
    scalar rel() const { return linf/std::fmax(refMax, scalar(1e-300)); }
};

Diff compare(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.linf = std::fmax(d.linf, std::fabs(a[i] - b[i]));
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    return d;
}

Diff compare(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const vector e{a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z};
        d.linf = std::fmax(d.linf, mag(e));
        d.refMax = std::fmax(d.refMax, mag(b[i]));
    }
    return d;
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: a fully periodic two-phase channel ==\n");
    if (argc < 7)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <wallsOfTimeDir>"
                    " [<profile>]\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string wallsDir = argv[6];
    const std::string profile = (argc > 7) ? argv[7] : "MULESCorr";
    const bool jumpProfile = (profile == "jump");
    (void)jumpProfile;

    std::printf("  profile: %s\n", profile.c_str());
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> patches = buildPatches(m, g);
    // the pair, which brae couples exactly as braeInterFoam.cu's own driver couples it
    attachCyclicCoupling(patches, m, g);
    const label nC = m.nCells();

    std::size_t nCoupled = 0;
    for (const FvPatch& q : patches)
    {
        if (q.coupled)
        {
            nCoupled += static_cast<std::size_t>(q.size);
        }
    }
    std::printf("  mesh: %d cells, %zu coupled faces\n", (int)nC, nCoupled);
    check("the mesh carries a periodic pair at all", nCoupled > 0);
    if (!nCoupled)
    {
        std::printf("test_inter_cyclic_vs_openfoam [%s]: %d failures\n", profile.c_str(), failures);
        return 1;
    }

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);

    auto readCells = [&](const std::string& path)
    {
        const FieldData<scalar> fd = readField<scalar>(path);
        std::vector<scalar> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };
    auto readVectorCells = [&](const std::string& path)
    {
        const FieldData<vector> fd = readField<vector>(path);
        std::vector<vector> v;
        if (fd.internalUniform)
        {
            v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        }
        else
        {
            v = fd.internalField;
        }
        return v;
    };

    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    // THE INITIAL-RESIDUAL BOUND IS THE SECOND CORRECTOR'S OWN SCALE. In step one OpenFOAM's three
    // solves start at 1, 9.916e-08 and 7.359e-12: the correctors after the first begin from a residual
    // that has already collapsed, and a residual is a difference of nearly equal O(1) quantities
    // normalised back to O(1), so round-off on it is ~1e-15 ABSOLUTE however small it gets. On 9.9e-08
    // that is 1.6e-08 relative, which is what this run measures -- the two codes' fields agree to
    // 1.9e-13. The ITERATION COUNTS are what discriminates here and they are asserted exactly: walling
    // the pair moves OpenFOAM's own step one from 70/32/4 to 64/50/1.
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh",
                                               scalar(5e-7), scalar(5e-7));

    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/" + fin.alphaName);
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const std::vector<vector> ofU = readVectorCells(ofDir + "/U");
    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dU = compare(fin.U.internal, ofU);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < B_ALPHA);
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < B_PRGH);
    check("U agrees with OpenFOAM's relatively", dU.rel() < B_U);

    // THE FIXTURE CARRIES SOMETHING ACROSS THE PAIR, in OpenFOAM's own numbers: without a flux there
    // the coupling could be dropped entirely and every comparison above would still pass.
    {
        scalar worstU = 0;
        for (const FvPatch& q : patches)
        {
            if (!q.coupled)
            {
                continue;
            }
            for (label i = 0; i < q.size; ++i)
            {
                worstU = std::fmax(worstU, mag(ofU[static_cast<std::size_t>(q.faceCells[i])]));
            }
        }
        std::printf("  OpenFOAM's |U| in the pair's own cells reaches %.4e\n", (double)worstU);
        check("OpenFOAM moves fluid through the periodic pair", worstU > scalar(1e-3));
    }

    // THE CONTROL, on the oracle. For the plain profiles it is the same mesh with the pair replaced by
    // two WALLS; for `jump` it is OpenFOAM's own PLAIN-CYCLIC answer, which is the pair coupled and
    // nothing else -- so what the comparison shows there is the jump itself and not the coupling.
    const std::vector<scalar> wAlpha = readCells(wallsDir + "/alpha.water");
    const std::vector<vector> wU = readVectorCells(wallsDir + "/U");
    const Diff cA = compare(wAlpha, ofAlpha);
    const Diff cU = compare(wU, ofU);
    std::printf("  CONTROL: OpenFOAM with %s: alpha %.4e, U relative %.4e\n",
                jumpProfile ? "the pair a PLAIN CYCLIC (no jump)" : "the pair two WALLS",
                (double)cA.linf, (double)cU.rel());
    check(jumpProfile ? "the JUMP moves OpenFOAM's own alpha far more than brae is from it"
                      : "walling the pair moves OpenFOAM's own alpha far more than brae is from it",
          cA.linf > scalar(1000)*std::fmax(dA.linf, scalar(1e-16)));
    check("...and its U", cU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-16)));

    // THE DEVICE ARM, against the SAME OpenFOAM fields. It is not a refusal any more: the pressure
    // corrector, the alpha step and the momentum source all carry the pair. What it took, in the order
    // this fixture found it -- the host<->device boundary-array layout, the momentum matrix's own
    // interface coefficient in fvMatrix::H(), the Gauss-Seidel smoother applying the interface to its
    // right-hand side every sweep (OpenFOAM's GaussSeidelSmoother.C:117-143), fvc::ddtCorr on the pair,
    // and divDevReff's grad(U) and stress flux there. EVERY ONE of them is identically zero at step one,
    // where U and phi are zero at rest, which is why this gate has to run more than one step.
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  (no CUDA device: the device arm is not exercised)\n");
    }
    else
    {
        InterFields dev;
        const RunReport rd = runInterFoamDevice(caseDir, startDir, m, g, patches, nSteps, false, &dev);
        check("the device arm ran the same number of steps", rd.steps == nSteps);
        failures += brae::gatecheck::nonFinite("device alpha", dev.alpha1.internal);
        failures += brae::gatecheck::nonFinite("device p_rgh", dev.p_rgh.internal);
        failures += brae::gatecheck::nonFinite("device U", dev.U.internal);
        const Diff eA = compare(dev.alpha1.internal, ofAlpha);
        const Diff eP = compare(dev.p_rgh.internal, ofPrgh);
        const Diff eU = compare(dev.U.internal, ofU);
        std::printf("  device alpha:   Linf %.4e\n", (double)eA.linf);
        std::printf("  device p_rgh:   relative %.4e\n", (double)eP.rel());
        std::printf("  device U:       relative %.4e\n", (double)eU.rel());
        check("the device's alpha agrees with OpenFOAM's absolutely", eA.linf < B_ALPHA_DEV);
        check("the device's p_rgh agrees with OpenFOAM's relatively", eP.rel() < B_PRGH_DEV);
        check("the device's U agrees with OpenFOAM's relatively", eU.rel() < B_U_DEV);
        // ...and against the HOST arm, which is the tighter question: the two run the same
        // discretisation, so what separates them is the port and not the scheme.
        const Diff hA = compare(dev.alpha1.internal, fin.alpha1.internal);
        const Diff hP = compare(dev.p_rgh.internal, fin.p_rgh.internal);
        const Diff hU = compare(dev.U.internal, fin.U.internal);
        std::printf("  device vs HOST: alpha %.4e, p_rgh %.4e relative, U %.4e relative\n",
                    (double)hA.linf, (double)hP.rel(), (double)hU.rel());
        check("...and with brae's own host arm, which runs the same discretisation",
              hA.linf < B_ALPHA_DEV_HOST && hP.rel() < B_PRGH_DEV_HOST && hU.rel() < B_U_DEV_HOST);
    }

    std::printf("test_inter_cyclic_vs_openfoam [%s]: %d failures\n", profile.c_str(), failures);
    return failures == 0 ? 0 : 1;
}
