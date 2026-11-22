// The forces face integration's two paths (src/functionObjects/forces_device.cu) against each other, bit for bit, in one build.
//
// deviceForcePatchSums makes each selected patch's twelve sums one of two ways. The default is one block a patch: each of its
// 256 threads walks a contiguous run of the patch's faces, computing each face's terms as it goes, and the block adds the 256
// partial sums in a fixed tree. The face-parallel path (BRAE_FORCES_FACE_PARALLEL=1) computes each face's twelve terms in a
// face-parallel kernel, adds them with one thread a (run, term) in face order over the same runs, and adds the runs in the same
// tree. Every patch's twelve sums, and the totals the forces object reduces them to (the patches added in order, then the total
// force and moment), must be the same doubles on:
//   - a tutorial's patches: validation/pitzDaily282 (OpenFOAM's pitzDaily, its converged fields), every patch with faces;
//   - adversarial run lengths: patches of 0, 1, 2, 3, 255, 256, 257, 511, 512, 513, ... faces, one of 298,024 (a car's
//     wheel patch, 1,165 faces a run) and more than 2^19 faces in all, so that the face-parallel path's chunk carry is crossed
//     inside a run; random gathers, values over six decades, signed zeros;
// with rhoB null and set, the boundary correction of grad(U) on and off, and the face-parallel path twice on one selection (its
// kept scratch).
// THE ENVIRONMENT. A solver's call leaves the path to BRAE_FORCES_FACE_PARALLEL, read once a process, so the test has two arms
// (tests/cmake): `default`, the variable unset, where such a call must take the one-block-a-patch kernel and leave the
// selection's scratch empty; and `optin`, the variable 1, where it must take the face-parallel path. In both, the two paths are
// then named explicitly (DeviceForceFields::integration) and compared.
// CONTROL: DeviceForceFields::controlReverseRun adds the largest patch's first run in reverse on the face-parallel path. The
// comparison must see it on every data set (each one's largest patch has runs of 3 faces or more, so the reversed sums can
// round differently), and the next call must be exact again. The one-block-a-patch kernel must refuse the control.
// Printed for information, not gated: the scratch the face-parallel path holds, and each path's time on the adversarial set.
// Run: test_forces_face_parallel <caseDir> <time> [default|optin]
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "forces_device.cuh"
#include "device_buffer.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

namespace
{

// the chunk of forces_device.cu (kChunkFaces): the adversarial set must cross it inside a run
constexpr int kChunkFaces = 1 << 19;
constexpr int kRuns = 256;
constexpr int kTerms = 12;

// One data set on the host: a selection's faces and the fields the integration reads, in the solver's layouts
struct HostCase
{
    std::string name;
    std::vector<int> patchOffset{0};
    std::vector<label> bnd;
    std::vector<scalar> Sf, Cf;
    std::vector<label> faceCellB;
    std::vector<scalar> dcB, pB, UB[3], U[3], gradU, nuEffB, rhoB;
    int nCells = 0;
};

struct DeviceCase
{
    DeviceForceFaces faces;
    DeviceBuffer<label> faceCellB;
    DeviceBuffer<scalar> dcB, pB, UB[3], U[3], gradU, nuEffB, rhoB;
    int nCells = 0;

    explicit DeviceCase(const HostCase& h)
    {
        faces.patchOffset = h.patchOffset;
        faces.nFaces = static_cast<int>(h.bnd.size());
        faces.bnd.copyFrom(h.bnd);
        faces.Sf.copyFrom(h.Sf);
        faces.Cf.copyFrom(h.Cf);
        faceCellB.copyFrom(h.faceCellB);
        dcB.copyFrom(h.dcB);
        pB.copyFrom(h.pB);
        for (int k = 0; k < 3; ++k)
        {
            UB[k].copyFrom(h.UB[k]);
            U[k].copyFrom(h.U[k]);
        }
        gradU.copyFrom(h.gradU);
        nuEffB.copyFrom(h.nuEffB);
        rhoB.copyFrom(h.rhoB);
        nCells = h.nCells;
    }

    DeviceForceFields fields(
        bool withRhoB,
        bool correctGrad,
        ForceIntegration integration) const
    {
        DeviceForceFields fl;
        fl.faceCellB = faceCellB.data();
        fl.dcB = dcB.data();
        fl.pB = pB.data();
        for (int k = 0; k < 3; ++k)
        {
            fl.UB[k] = UB[k].data();
            fl.U[k] = U[k].data();
        }
        fl.gradU = gradU.data();
        fl.nCells = nCells;
        fl.nuEffB = nuEffB.data();
        fl.rhoB = withRhoB ? rhoB.data() : nullptr;
        fl.rhoP = 1.204;
        fl.rhoV = 1.204;
        fl.pRef = 0.25;
        fl.origin = vector{0.125, -0.0375, 0.0078125};
        fl.correctGrad = correctGrad;
        fl.integration = integration;
        return fl;
    }

    // the bytes of the face-parallel path's scratch the selection holds
    std::size_t scratchBytes() const
    {
        return faces.dPatchOffset.size() * sizeof(int) + (faces.terms.size() + faces.runSums.size() + faces.sums.size()) * sizeof(double);
    }
};

// What the forces object makes of the patches' sums (forces_fo.cuh execute and forceLine): the twelve sums of the patches added
// in order from 0.0, then the total force fP + fV + fI and moment mP + mV + mI, fI and mI zero here
std::vector<double> objectTotals(const std::vector<double>& perPatch)
{
    double s[kTerms] = {};
    const std::size_t nPatches = perPatch.size() / kTerms;
    for (std::size_t p = 0; p < nPatches; ++p)
        for (int k = 0; k < kTerms; ++k) s[k] += perPatch[kTerms * p + static_cast<std::size_t>(k)];
    std::vector<double> t(s, s + kTerms);
    for (int k = 0; k < 3; ++k) t.push_back(s[k] + s[3 + k] + 0.0);
    for (int k = 0; k < 3; ++k) t.push_back(s[6 + k] + s[9 + k] + 0.0);
    return t;
}

// the doubles of a and b that differ in any bit, and the largest |a - b| among them
int bitDiffs(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double& worst)
{
    worst = 0.0;
    if (a.size() != b.size()) return static_cast<int>(a.size() > b.size() ? a.size() : b.size());
    int n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::memcmp(&a[i], &b[i], sizeof(double)) == 0) continue;
        ++n;
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    }
    return n;
}

// one comparison of the two paths' outputs: the sums a patch and the object's totals, bit for bit
bool sameOutputs(
    const std::vector<double>& ref,
    const std::vector<double>& got,
    const char* what)
{
    double worstSums = 0.0;
    double worstTotals = 0.0;
    const int nSums = bitDiffs(ref, got, worstSums);
    const std::vector<double> tRef = objectTotals(ref);
    const int nTotals = bitDiffs(tRef, objectTotals(got), worstTotals);
    const bool pass = nSums == 0 && nTotals == 0 && !ref.empty();
    std::printf("  %s: %zu sums, %d differ in any bit (largest |difference| %.3e); %zu totals, %d differ (%.3e) %s\n", what, ref.size(),
                nSums, worstSums, tRef.size(), nTotals, worstTotals, pass ? "OK" : "FAIL");
    return pass;
}

// the longest run of the largest patch: the faces the control reverses
int controlRunLength(const std::vector<int>& patchOffset)
{
    int most = 0;
    for (std::size_t p = 0; p + 1 < patchOffset.size(); ++p) most = std::max(most, patchOffset[p + 1] - patchOffset[p]);
    return (most + kRuns - 1) / kRuns;
}

// pitzDaily: every patch with faces, in mesh order; the solver's boundary arrays count every boundary face (no coupled patch here)
HostCase tutorialCase(
    const std::string& caseDir,
    const std::string& t)
{
    HostCase h;
    h.name = "tutorial " + caseDir + "/" + t;
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    GeometricField<vector> Uf = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, m.nCells());
    Uf.evaluateBoundary();
    GeometricField<scalar> pf = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, m.nCells());
    pf.evaluateBoundary();
    const std::vector<scalar> kInt = readField<scalar>(caseDir + "/" + t + "/k").internalField;
    const int nC = m.nCells();
    h.nCells = nC;
    for (int k = 0; k < 3; ++k) h.U[k].resize(static_cast<std::size_t>(nC));
    for (int c = 0; c < nC; ++c)
    {
        h.U[0][c] = Uf.internal[c].x;
        h.U[1][c] = Uf.internal[c].y;
        h.U[2][c] = Uf.internal[c].z;
    }
    // the boundary values: the patch field's, or the owner cell's where the patch holds none (empty)
    std::vector<vector> Ub;
    std::vector<scalar> pb;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& p = fvp[pi];
        const std::vector<vector>& uv = Uf.boundary[pi]->value();
        const std::vector<scalar>& pv = pf.boundary[pi]->value();
        for (label i = 0; i < p.size; ++i)
        {
            const label c = m.owner()[static_cast<std::size_t>(p.start + i)];
            Ub.push_back(static_cast<label>(uv.size()) == p.size ? uv[i] : Uf.internal[c]);
            pb.push_back(static_cast<label>(pv.size()) == p.size ? pv[i] : pf.internal[c]);
        }
    }
    // grad(U), Gauss linear on the internal faces and the boundary values; d U_j / d x_i at (3*i + j)*nCells + cell
    h.gradU.assign(static_cast<std::size_t>(9) * nC, 0.0);
    const int nIF = m.nInternalFaces();
    for (int f = 0; f < m.nFaces(); ++f)
    {
        const label o = m.owner()[f];
        vector uf;
        if (f < nIF)
        {
            const scalar w = g.weights()[f];
            const label n = m.neighbour()[f];
            uf = vector{w * Uf.internal[o].x + (1 - w) * Uf.internal[n].x, w * Uf.internal[o].y + (1 - w) * Uf.internal[n].y,
                        w * Uf.internal[o].z + (1 - w) * Uf.internal[n].z};
        }
        else
        {
            uf = Ub[static_cast<std::size_t>(f - nIF)];
        }
        const vector S = g.Sf()[f];
        const double s3[3] = {S.x, S.y, S.z};
        const double u3[3] = {uf.x, uf.y, uf.z};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                h.gradU[static_cast<std::size_t>(3 * i + j) * nC + o] += s3[i] * u3[j];
                if (f < nIF) h.gradU[static_cast<std::size_t>(3 * i + j) * nC + m.neighbour()[f]] -= s3[i] * u3[j];
            }
    }
    for (int k = 0; k < 9; ++k)
        for (int c = 0; c < nC; ++c) h.gradU[static_cast<std::size_t>(k) * nC + c] /= g.V()[c];
    // the solver's boundary arrays and the selection: every patch with faces
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& p = fvp[pi];
        for (label i = 0; i < p.size; ++i)
        {
            const int f = p.start + i;
            const label b = f - nIF;
            const label c = m.owner()[f];
            const vector S = g.Sf()[f];
            const vector C = g.Cf()[f];
            const vector d{C.x - g.C()[c].x, C.y - g.C()[c].y, C.z - g.C()[c].z};
            h.faceCellB.push_back(c);
            h.dcB.push_back(std::sqrt(S.x * S.x + S.y * S.y + S.z * S.z) / (S.x * d.x + S.y * d.y + S.z * d.z));
            h.pB.push_back(pb[b]);
            h.UB[0].push_back(Ub[b].x);
            h.UB[1].push_back(Ub[b].y);
            h.UB[2].push_back(Ub[b].z);
            h.nuEffB.push_back(1e-5 + 0.09 * kInt[c] * 1e-3);
            h.rhoB.push_back(1.2 + 0.01 * std::sin(0.37 * b));
            h.bnd.push_back(b);
            h.Sf.insert(h.Sf.end(), {S.x, S.y, S.z});
            h.Cf.insert(h.Cf.end(), {C.x, C.y, C.z});
        }
        if (p.size > 0) h.patchOffset.push_back(static_cast<int>(h.bnd.size()));
    }
    return h;
}

// Adversarial run lengths and values. The boundary arrays are longer than the selection, and the selection reaches them through
// a random permutation; the cells are random; the values span six decades with both signs; some faces have p_b == pRef and
// U_b == U_c exactly (zero terms, signed zeros), some an axis-aligned Sf.
HostCase adversarialCase()
{
    HostCase h;
    h.name = "adversarial run lengths";
    const std::vector<int> sizes = {3, 0, 1, 2, 255, 256, 257, 511, 512, 513, 767, 1000, 4095, 65537, 300000, 298024, 17};
    for (int n : sizes) h.patchOffset.push_back(h.patchOffset.back() + n);
    const int nF = h.patchOffset.back();
    const int nB = nF + 4099;
    const int nC = 1 << 18;
    h.nCells = nC;
    std::mt19937_64 rng(20261122);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto wide = [&](double lo, double hi)
    {
        const double v = std::pow(10.0, lo + (hi - lo) * u01(rng)) * (1.0 + u01(rng));
        return u01(rng) < 0.5 ? -v : v;
    };
    std::vector<label> perm(static_cast<std::size_t>(nB));
    for (int b = 0; b < nB; ++b) perm[b] = b;
    std::shuffle(perm.begin(), perm.end(), rng);
    h.bnd.assign(perm.begin(), perm.begin() + nF);
    for (int b = 0; b < nB; ++b)
    {
        h.faceCellB.push_back(static_cast<label>(rng() % nC));
        h.dcB.push_back(std::fabs(wide(0.0, 4.0)));
        h.pB.push_back(u01(rng) < 0.02 ? 0.25 : wide(-2.0, 3.0));
        for (int k = 0; k < 3; ++k) h.UB[k].push_back(wide(-3.0, 1.7));
        h.nuEffB.push_back(std::fabs(wide(-5.0, -2.0)));
        h.rhoB.push_back(1.0 + 0.3 * u01(rng));
    }
    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < nC; ++c) h.U[k].push_back(wide(-3.0, 1.7));
    for (int k = 0; k < 9; ++k)
        for (int c = 0; c < nC; ++c) h.gradU.push_back(wide(-2.0, 4.0));
    for (int f = 0; f < nF; ++f)
    {
        double S[3] = {wide(-7.0, -1.0), wide(-7.0, -1.0), wide(-7.0, -1.0)};
        if (f % 97 == 0) S[f % 3] = 0.0;
        if (f % 89 == 0) S[(f + 1) % 3] = 0.0;
        for (int k = 0; k < 3; ++k) h.Sf.push_back(S[k]);
        for (int k = 0; k < 3; ++k) h.Cf.push_back(wide(-2.0, 0.7));
        // U_b == U_c on some faces: no boundary correction in that direction
        if (f % 53 == 0)
        {
            const label b = h.bnd[f];
            const label c = h.faceCellB[b];
            for (int k = 0; k < 3; ++k) h.UB[k][b] = h.U[k][c];
        }
    }
    return h;
}

// the median wall time of a call (it returns the sums on the host, so it waits for the kernels), five calls after one
double medianCallMs(
    const DeviceCase& d,
    const DeviceForceFields& fl)
{
    deviceForcePatchSums(d.faces, fl);
    std::vector<double> ms;
    for (int i = 0; i < 5; ++i)
    {
        const auto t0 = std::chrono::steady_clock::now();
        deviceForcePatchSums(d.faces, fl);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[2];
}

// One data set: the environment's path first (on a selection no call has touched), then every configuration's two paths, the
// face-parallel one twice, then the control
bool runCase(
    const HostCase& h,
    bool optin,
    bool timed)
{
    const DeviceCase d(h);
    const int ctlRun = controlRunLength(h.patchOffset);
    std::printf("%s: %zu patches, %d faces, %d cells; the control's run %d faces\n", h.name.c_str(), h.patchOffset.size() - 1, d.faces.nFaces,
                d.nCells, ctlRun);
    bool ok = true;
    // THE ENVIRONMENT'S PATH: a solver's call. Unset, it must be the one-block-a-patch kernel and leave the scratch empty
    const std::vector<double> envSums = deviceForcePatchSums(d.faces, d.fields(false, true, ForceIntegration::environment));
    const std::size_t envScratch = d.scratchBytes();
    const bool envPath = optin ? envScratch > 0 : envScratch == 0;
    ok = ok && envPath;
    std::printf("  the environment's call: the face-parallel scratch holds %zu bytes after it: %s\n", envScratch,
                envPath ? (optin ? "the face-parallel path (OK)" : "the one-block-a-patch kernel, nothing new allocated (OK)")
                        : (optin ? "FAIL (the opt-in did not take the face-parallel path)" : "FAIL (the default allocated the scratch)"));
    const std::vector<double> envRef = deviceForcePatchSums(d.faces, d.fields(false, true, ForceIntegration::patchBlocks));
    ok = sameOutputs(envRef, envSums, "the environment's call against the one-block-a-patch kernel") && ok;
    // THE TWO PATHS, named
    const bool configs[3][2] = {{false, true}, {true, true}, {false, false}};
    for (const auto& cfg : configs)
    {
        const std::vector<double> ref = deviceForcePatchSums(d.faces, d.fields(cfg[0], cfg[1], ForceIntegration::patchBlocks));
        for (int call = 1; call <= 2; ++call)
        {
            const std::vector<double> got = deviceForcePatchSums(d.faces, d.fields(cfg[0], cfg[1], ForceIntegration::faceParallel));
            char what[160];
            std::snprintf(what, sizeof what, "rhoB %s, grad(U) boundary correction %s, face-parallel call %d", cfg[0] ? "set " : "null",
                          cfg[1] ? "on " : "off", call);
            ok = sameOutputs(ref, got, what) && ok;
        }
    }
    std::printf("  the face-parallel scratch held: %zu bytes (terms %zu B = 96 B a face x %zu faces, run sums %zu B, sums %zu B, offsets %zu B)\n",
                d.scratchBytes(), d.faces.terms.size() * sizeof(double), d.faces.terms.size() / kTerms, d.faces.runSums.size() * sizeof(double),
                d.faces.sums.size() * sizeof(double), d.faces.dPatchOffset.size() * sizeof(int));
    // CONTROL: the largest patch's first run in reverse must be seen
    DeviceForceFields fl = d.fields(false, true, ForceIntegration::faceParallel);
    const std::vector<double> ref = deviceForcePatchSums(d.faces, d.fields(false, true, ForceIntegration::patchBlocks));
    fl.controlReverseRun = true;
    const std::vector<double> bad = deviceForcePatchSums(d.faces, fl);
    double worst = 0.0;
    const int nd = bitDiffs(ref, bad, worst);
    const bool witness = ctlRun >= 3;
    const bool caught = nd > 0;
    std::printf("  CONTROL (the largest patch's first run of %d faces added in reverse): %d sums differ (largest |difference| %.3e): %s\n", ctlRun, nd,
                worst, !witness ? "FAIL (the data cannot witness it: a run of fewer than 3 faces)" : caught ? "caught" : "FAIL (not caught)");
    // and the control left nothing behind: the next call is exact again
    fl.controlReverseRun = false;
    const bool again = sameOutputs(ref, deviceForcePatchSums(d.faces, fl), "after the control");
    // the one-block-a-patch kernel has no run to reverse: it must refuse the control, not run without it
    bool refused = false;
    try
    {
        DeviceForceFields blocks = d.fields(false, true, ForceIntegration::patchBlocks);
        blocks.controlReverseRun = true;
        deviceForcePatchSums(d.faces, blocks);
    }
    catch (const std::logic_error&)
    {
        refused = true;
    }
    std::printf("  the control on the one-block-a-patch kernel: %s\n", refused ? "refused (OK)" : "FAIL (it ran)");
    if (timed)
    {
        const double msBlocks = medianCallMs(d, d.fields(false, true, ForceIntegration::patchBlocks));
        const double msFaces = medianCallMs(d, d.fields(false, true, ForceIntegration::faceParallel));
        std::printf("  time a call, median of 5 (information, not gated): one block a patch %.3f ms, face-parallel %.3f ms\n", msBlocks,
                    msFaces);
    }
    return ok && witness && caught && again && refused;
}

} // namespace

int main(
    int argc,
    char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily282";
    const std::string t = argc > 2 ? argv[2] : "282";
    const std::string arm = argc > 3 ? argv[3] : "default";
    if (arm != "default" && arm != "optin")
    {
        std::printf("test_forces_face_parallel: the arm is default or optin, not %s\n", arm.c_str());
        return 2;
    }
    const bool optin = arm == "optin";
    std::printf("test_forces_face_parallel (%s arm): deviceForcePatchSums' one-block-a-patch kernel and face-parallel path, bit for bit\n",
                arm.c_str());
    const bool selected = forcesFaceParallelSelected();
    const bool armOk = selected == optin;
    std::printf("BRAE_FORCES_FACE_PARALLEL selects the %s: %s\n", selected ? "face-parallel path" : "one-block-a-patch kernel",
                armOk ? "as this arm expects" : "FAIL (not what this arm expects)");
    bool ok = armOk;
    ok = runCase(tutorialCase(caseDir, t), optin, false) && ok;
    const HostCase adv = adversarialCase();
    // the chunk carry must be crossed inside a run, or this set cannot witness it
    int inRun = -1;
    for (std::size_t p = 0; p + 1 < adv.patchOffset.size(); ++p)
    {
        const int begin = adv.patchOffset[p];
        const int end = adv.patchOffset[p + 1];
        if (kChunkFaces <= begin || kChunkFaces >= end) continue;
        const int per = (end - begin + kRuns - 1) / kRuns;
        inRun = (kChunkFaces - begin) % per;
    }
    const bool carried = inRun > 0;
    std::printf("adversarial set: the chunk boundary (face %d) falls %d faces into a run: %s\n", kChunkFaces, inRun,
                carried ? "the carry is crossed" : "FAIL (not inside a run)");
    ok = ok && carried;
    ok = runCase(adv, optin, true) && ok;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
