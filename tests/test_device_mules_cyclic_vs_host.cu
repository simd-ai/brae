// MULES's limiter ACROSS A CYCLIC PAIR: the device's lambda against the host's, at identical inputs.
//
// A cyclic patch is not in the device mesh's boundary gather at all (device_mesh.cuh:41-44), so before
// this the device limiter saw a periodic mesh as one with a wall there: the pair's faces contributed no
// extrema, no flux budget and got no limiter. The host does three things with them, and each is
// transcribed into the device kernels rather than reasoned about:
//   the EXTREMA take the neighbour CELL's psi (mules_cpp.cu:181-185), not a stored patch value;
//   the BUDGETS take phiBD and phiCorr with the owner-face sign, as any patch face does (:199-203);
//   the LIMITER is this side's per-cell one (:311-318) and then the PAIR IS SYNCED to the smaller of
//   the two (:321-345) -- syncTools::syncFaceList with minEqOp, which is NOT a no-op in serial and is
//   what makes a periodic face behave as the one internal face it is.
//
// The two layouts differ and the mapping is the fiddly part: the host's boundary arrays hold every
// patch, coupled included; the device's hold only the uncoupled ones and keep the pair in a separate
// interface array in buildDeviceCyclic's order. The gate builds one set of inputs and hands each arm
// its own view of them.
//
// THE BOUND IS NOT BITWISE. The device gathers per cell in a different order from the host's face loops
// (device_mules.cuh says so for the internal faces already), so the budgets differ in the last bits and
// a limiter that is a ratio of them can differ by more. What must hold exactly is the STRUCTURE: the
// pair's two sides carry the SAME limiter after the sync, and a face OpenFOAM would leave at 1 is not
// pulled down.
//
// MEASURED on validation/cyclicChannel: the device's lambda is the host's to 2.8e-16 on the internal
// faces and 1.1e-16 on the pair, the limiter reaches 0 so it is biting, and both sides of all 20
// periodic faces carry the same limiter to the last bit. BROKEN ONCE EACH, and each leaves its own
// signature: the interface extrema and budgets dropped -- internal 3.9e-01, interface 2.5e-01; the
// interface limiter dropped -- interface lambda stays at 1.0 and the internal faces go 9.1e-01; the
// SYNC dropped -- the two sides of a face differ by 7.8e-01, which only the pair arm above sees.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_field.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"
#include "mules_cpp.cuh"
#include "device_mules.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
}   // namespace

int main(int argc, char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicChannel";
    std::printf("== MULES's limiter across a cyclic pair: the device against the host ==\n");

    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    attachCyclicCoupling(fvp, m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);

    const label nC = m.nCells(), nIf = m.nInternalFaces();
    std::size_t nCoupled = 0;
    for (const FvPatch& q : fvp) if (q.coupled) nCoupled += static_cast<std::size_t>(q.size);
    std::printf("  mesh: %d cells, %d internal faces, %zu coupled faces\n", (int)nC, (int)nIf, nCoupled);
    check("the fixture has a coupled pair", nCoupled > 0);
    if (!nCoupled) { std::printf("test_device_mules_cyclic_vs_host: %d failures\n", failures); return 1; }

    // psi in [0, 1] with an interface running through the mesh, as alpha is, and an old value that
    // differs from it -- MULES's budget is the difference.
    std::vector<scalar> psiCell(static_cast<std::size_t>(nC)), psiOld(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar a = scalar(0.5) + scalar(0.5)*std::tanh(scalar(6)*(scalar(0.55) - C.y));
        psiCell[static_cast<std::size_t>(c)] = std::fmin(std::fmax(a, scalar(0)), scalar(1));
        psiOld[static_cast<std::size_t>(c)] =
            std::fmin(std::fmax(a + scalar(0.02)*std::sin(scalar(3)*C.x), scalar(0)), scalar(1));
    }
    const GeometricField<scalar> psi = buildCyclicField<scalar>(psiCell, fvp, cyclics);

    // phiBD and phiCorr: a donor flux and an antidiffusive correction big enough that the limiter
    // actually bites somewhere -- a gate on a lambda that is 1 everywhere measures nothing.
    SurfaceScalarField phiBD, phiCorr;
    phiBD.internal.resize(static_cast<std::size_t>(nIf));
    phiCorr.internal.resize(static_cast<std::size_t>(nIf));
    for (label f = 0; f < nIf; ++f)
    {
        phiBD.internal[static_cast<std::size_t>(f)] = scalar(0.02)*std::sin(scalar(0.9)*scalar(f));
        phiCorr.internal[static_cast<std::size_t>(f)] = scalar(0.05)*std::cos(scalar(1.3)*scalar(f));
    }
    phiBD.boundary.resize(fvp.size());
    phiCorr.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phiBD.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        phiCorr.boundary[pi].assign(static_cast<std::size_t>(fvp[pi].size), scalar(0));
        if (!fvp[pi].coupled) continue;   // an uncoupled patch's phiCorr is zero by construction
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(i);
            phiBD.boundary[pi][k] = scalar(0.02)*std::sin(scalar(1.1)*scalar(i) + scalar(pi));
            phiCorr.boundary[pi][k] = scalar(0.06)*std::cos(scalar(0.8)*scalar(i) + scalar(pi));
        }
    }

    cpu::MULES::Fields hf;           // interFoam's: rho, Sp, Su, psiMax, psiMin all the constants
    cpu::MULES::Controls hc;
    hc.nLimiterIter = 3;
    const scalar rDeltaT = scalar(100);

    cpu::MULES::Limiter hostLambda;
    cpu::MULES::limiter(hostLambda, rDeltaT, psi, psiOld, phiBD, phiCorr, hf, hc, m, g, fvp);

    // ---- the device's view of the same inputs ---------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);

    std::vector<scalar> bndPsi, bndPhiBD, bndPhiCorr;
    std::vector<int> bndFixes, bndFlag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;    // the device mesh skips these
        const std::vector<scalar>& pv = psi.boundary[pi]->value();
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(i);
            bndPsi.push_back(k < pv.size() ? pv[k] : scalar(0));
            bndPhiBD.push_back(phiBD.boundary[pi][k]);
            bndPhiCorr.push_back(phiCorr.boundary[pi][k]);
            bndFixes.push_back(psi.boundary[pi]->fixesValue() ? 1 : 0);
            bndFlag.push_back(fvp[pi].type == "empty" ? 1 : (fvp[pi].type == "wedge" ? 2 : 0));
        }
    }
    std::vector<scalar> ifPhiBD, ifPhiCorr;
    for (const CyclicInterface& c : cyclics)
    {
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            ifPhiBD.push_back(phiBD.boundary[static_cast<std::size_t>(c.patch)][i]);
            ifPhiCorr.push_back(phiCorr.boundary[static_cast<std::size_t>(c.patch)][i]);
        }
    }

    DeviceBuffer<scalar> dPsi(psiCell), dPsiOld(psiOld), dPsiBnd(bndPsi);
    DeviceBuffer<int> dFixes(bndFixes), dFlag(bndFlag);
    DeviceBuffer<scalar> dPhiBDInt(phiBD.internal), dPhiBDBnd(bndPhiBD);
    DeviceBuffer<scalar> dPhiCorrInt(phiCorr.internal), dPhiCorrBnd(bndPhiCorr);
    DeviceBuffer<scalar> dPhiBDIf(ifPhiBD), dPhiCorrIf(ifPhiCorr);
    DeviceMulesFields df;
    DeviceMulesControls dc;
    dc.nLimiterIter = hc.nLimiterIter;
    DeviceBuffer<scalar> lamInt, lamBnd, lamIf;
    deviceMulesLimiter(dm, (int)nIf, (int)bndPhiBD.size(), rDeltaT, dPsi, dPsiOld, dPsiBnd,
                       dFixes, dFlag, dPhiBDInt, dPhiBDBnd, dPhiCorrInt, dPhiCorrBnd, df, dc,
                       lamInt, lamBnd, &cyc, &dPhiBDIf, &dPhiCorrIf, &lamIf);

    std::vector<scalar> devInt, devIf;
    lamInt.copyTo(devInt);
    lamIf.copyTo(devIf);

    // ---- does the limiter BITE? ------------------------------------------------------------------
    scalar minInt = 1, minIf = 1;
    for (const scalar v : devInt) minInt = std::fmin(minInt, v);
    for (const scalar v : devIf)  minIf  = std::fmin(minIf, v);
    std::printf("  lambda reaches %.4f on the internal faces and %.4f on the pair\n",
                (double)minInt, (double)minIf);
    check("the limiter actually bites somewhere, so this gate is not comparing two fields of ones",
          minInt < scalar(0.999) || minIf < scalar(0.999));

    // ---- the pair's two sides agree, which is what the sync is for --------------------------------
    std::vector<label> twin;
    cyc.twin.copyTo(twin);
    scalar worstPair = 0;
    std::size_t nPaired = 0;
    for (std::size_t j = 0; j < devIf.size(); ++j)
    {
        if (twin[j] < 0) continue;
        ++nPaired;
        worstPair = std::fmax(worstPair, std::fabs(devIf[j] - devIf[static_cast<std::size_t>(twin[j])]));
    }
    std::printf("  the pair: %zu of %zu faces have a twin, worst |lambda - lambda_twin| %.4e\n",
                nPaired, devIf.size(), (double)worstPair);
    check("both sides of every periodic face carry the SAME limiter", nPaired == devIf.size());
    check("...to the last bit, which is what syncFaceList's minEqOp leaves", worstPair == scalar(0));

    // ---- against the host -------------------------------------------------------------------------
    scalar worstInt = 0;
    for (std::size_t f = 0; f < devInt.size() && f < hostLambda.internal.size(); ++f)
    {
        worstInt = std::fmax(worstInt, std::fabs(devInt[f] - hostLambda.internal[f]));
    }
    std::vector<scalar> hostIf;
    for (const CyclicInterface& c : cyclics)
    {
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            hostIf.push_back(hostLambda.boundary[static_cast<std::size_t>(c.patch)][i]);
        }
    }
    scalar worstIf = 0;
    for (std::size_t j = 0; j < devIf.size() && j < hostIf.size(); ++j)
    {
        worstIf = std::fmax(worstIf, std::fabs(devIf[j] - hostIf[j]));
    }
    std::printf("  vs the host: internal %.4e, interface %.4e\n", (double)worstInt, (double)worstIf);
    check("the device's internal-face limiter is the host's", worstInt <= scalar(1e-12));
    check("...and its interface limiter is the host's",
          devIf.size() == hostIf.size() && worstIf <= scalar(1e-12));

    std::printf("test_device_mules_cyclic_vs_host: %d failures\n", failures);
    return failures ? 1 : 0;
}
