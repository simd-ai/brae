// CUDA kEpsilon against the _cpp reference, stage by stage.
//
// The reference is itself gated against OpenFOAM's own dumps (tests/rho_kepsilon_vs_openfoam.sh, an
// instrumented copy of OpenFOAM's kEpsilon writing stage_divU / stage_GbyNu / stage_G and both assembled
// systems), so this closes OpenFOAM -> _cpp -> CUDA for the whole closure. Fifth of the device twins,
// beside test_rho_ueqn / peqn / eeqn / pceqn_cuda.
//
// WHY THE STAGES ARE COMPARED SEPARATELY AND NOT ONLY THE SOLVED FIELDS. The closure folds a production
// term, a wall treatment, two transport operators, a relaxation and two solves into k and epsilon. A
// single number over the solved field cannot say which of them moved, and the wall treatment in
// particular is invisible once folded into the source -- a wall defect and a relaxation defect are then
// the same number. eps0/G0/isWallCell were added to KEResiduals for exactly this.
//
// rho AND nu ARE SYNTHESIZED, for the reason test_rho_ueqn_cuda.cu sets out at length: pitzDailyTurb is
// an INCOMPRESSIBLE fixture and ships neither, while the compressible instantiation is the whole point
// of this module. This instrument measures DEVICE-vs-HOST on byte-identical inputs, not physics --
// physical validity is rho_kepsilon_vs_openfoam.sh's claim, on a fixture that does ship them.
//
// BOTH MUST VARY, and rho must vary on the BOUNDARY independently of the cells. With a uniform rho the
// mass flux and the volumetric flux coincide, so divU could be built from either and nothing would tell;
// with a constant nut boundary a kernel reading the owner cell's nut would agree everywhere. Each of
// those is a control below rather than an assumption.
//
// Run: test_rho_kepsilon_cuda <caseDir> <timeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "near_wall_dist.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"
#include "kEpsilon_cpp.cuh"
#include "kEpsilon.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

static void cmp(const std::vector<scalar>& gpu,
                const std::vector<scalar>& ref,
                const char*                nm,
                scalar                     tol)
{
    if (gpu.size() != ref.size())
    {
        std::printf("  %-38s SIZE MISMATCH %zu vs %zu  FAIL\n", nm, gpu.size(), ref.size());
        ++g_fails;
        return;
    }
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(gpu[i] - ref[i]));
        mg = std::fmax(mg, std::fabs(ref[i]));
    }
    const scalar rel = mg > 0 ? mx / mg : mx;
    const bool ok = rel <= tol;
    if (!ok) ++g_fails;
    std::printf("  %-38s n=%6zu rel=%.3e  max|ref|=%.3e absdiff=%.3e  %s\n",
                nm, ref.size(), rel, mg, mx, ok ? "OK" : "FAIL");
}

// For a quantity that is near zero BY CONSTRUCTION. div(phi) on a converged, conservative mass flux is
// the canonical case: max|div(phi)| here is 1.3e-03 against div(phi/rho)'s 9.7e+01, four orders down, so
// a relative bound on its own magnitude measures the fixture's convergence and not the device's
// arithmetic. The scale is supplied by the caller -- for divPhi it is the SAME operator's output on the
// volumetric flux, same mesh, same kernel. This is a different measurement, not a looser one: the
// absolute agreement demanded (5e-12 * 9.7e+01) is far tighter than the 1e-12 relative bound would have
// been if div(phi) happened to be large.
static void cmpScaled(const std::vector<scalar>& gpu,
                      const std::vector<scalar>& ref,
                      const char*                nm,
                      scalar                     scale,
                      scalar                     tol)
{
    scalar mx = 0;
    for (std::size_t i = 0; i < ref.size() && i < gpu.size(); ++i)
        mx = std::fmax(mx, std::fabs(gpu[i] - ref[i]));
    const scalar rel = scale > 0 ? mx / scale : mx;
    const bool ok = rel <= tol;
    if (!ok) ++g_fails;
    std::printf("  %-38s n=%6zu rel=%.3e  (scale=%.3e absdiff=%.3e)  %s\n",
                nm, ref.size(), rel, scale, mx, ok ? "OK" : "FAIL");
}

static void check(bool ok, const char* what)
{
    std::printf("  %-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

static scalar relDiff(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar d = 0, mg = 0;
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        d  = std::fmax(d, std::fabs(a[i] - b[i]));
        mg = std::fmax(mg, std::fabs(a[i]));
    }
    return mg > 0 ? d / mg : d;
}

// Flatten a per-patch host array into the boundary-face order DeviceMesh uses.
static std::vector<scalar> flatten(const std::vector<std::vector<scalar>>& v,
                                   const std::vector<FvPatch>&             fvp,
                                   int                                     nBnd,
                                   scalar                                  pad)
{
    std::vector<scalar> out;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
            out.push_back(v[pi][i]);
    out.resize(nBnd, pad);
    return out;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    U.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    std::vector<std::vector<scalar>> phiBnd(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phiBnd[pi].assign(fvp[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
                phiBnd[pi] = b.values;
    }
    SurfaceScalarField phi;
    phi.internal = phiF.internalField;
    phi.boundary = phiBnd;

    // ---- the synthesized compressible inputs -------------------------------------------------
    // A smooth spatial ramp, so rho and nu vary cell to cell and the rho-weighting of every term is
    // exercised. The magnitudes are a gas at room conditions; the point is that they are not constant.
    scalar xMin = 1e300, xMax = -1e300;
    for (label c = 0; c < nC; ++c)
    {
        xMin = std::fmin(xMin, g.C()[c].x);
        xMax = std::fmax(xMax, g.C()[c].x);
    }
    const scalar span = (xMax > xMin) ? (xMax - xMin) : 1.0;
    std::vector<scalar> rhoC(nC), nuC(nC);
    for (label c = 0; c < nC; ++c)
    {
        const scalar s = (g.C()[c].x - xMin) / span;
        rhoC[c] = 1.0 + 0.45 * s;
        nuC[c]  = 1.5e-5 * (1.0 + 0.30 * s);
    }
    // The BOUNDARY values are given their own ramp rather than copied from the owner cell. Copying would
    // make the patch-value rule untestable: a kernel reading the cell would agree everywhere.
    std::vector<std::vector<scalar>> rhoB(fvp.size()), nuB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        rhoB[pi].resize(fvp[pi].size);
        nuB[pi].resize(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label c = fvp[pi].faceCells[i];
            rhoB[pi][i] = rhoC[c] * (1.0 + 0.07 * ((i % 5) - 2) * 0.1);
            nuB[pi][i]  = nuC[c]  * (1.0 + 0.05 * ((i % 3) - 1) * 0.1);
        }
    }

    // compressibleTurbulenceModel::phi() -- the VOLUMETRIC flux, phi/fvc::interpolate(rho).
    SurfaceScalarField phiByRho = phi;
    {
        const SurfaceScalarField rhof = fvc::interpolate(rhoC, m, g, fvp);
        for (std::size_t f = 0; f < phiByRho.internal.size(); ++f)
            phiByRho.internal[f] /= rhof.internal[f];
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                phiByRho.boundary[pi][i] /= rhoB[pi][i];
    }

    KEpsilonCoeffs co;
    co.correctedLaplacian = true;    // pitzDaily's fvSchemes says `default Gauss linear corrected`
    const scalar relaxEps = 0.7, relaxK = 0.7;
    const scalar tol = 1e-14, relTol = 0.0;
    const int maxIter = 2000;

    // ---- the HOST run -------------------------------------------------------------------------
    auto freshField = [&](const char* nm)
    {
        GeometricField<scalar> f =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + nm), fvp, nC);
        return f;
    };
    GeometricField<scalar> hk = freshField("k"), he = freshField("epsilon"), hn = freshField("nut");
    hk.evaluateBoundary();
    he.evaluateBoundary();

    std::vector<scalar> hAlphat(nC, 0.0);
    cpu::kEpsilonRef::Compressible comp;
    comp.rho      = &rhoC;
    comp.rhoBnd   = &rhoB;
    comp.nu       = &nuC;
    comp.nuBnd    = &nuB;
    comp.phiByRho = &phiByRho;
    comp.alphat   = &hAlphat;
    comp.Prt      = 0.85;

    cpu::kEpsilonRef::KEResiduals res;
    res.captureStages = true;
    cpu::kEpsilonRef::correct(U, hk, he, hn, phi, /*nu=*/0.0, m, g, fvp,
                              relaxEps, relaxK, tol, relTol, maxIter, co, &res,
                              /*bounded=*/true, /*dropTerm=*/0, &comp, nullptr);

    // ---- the DEVICE run -----------------------------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);

    // Coupled patches would be kept out of the LDU entirely, which is what the module refuses; assert
    // the fixture has none rather than assume it.
    {
        label coupled = 0;
        for (const FvPatch& p : fvp)
            if (p.type == "cyclic" || p.type == "cyclicAMI" || p.type == "processor") ++coupled;
        check(coupled == 0, "fixture has no coupled patches (this gate cannot cover them)");
    }

    GeometricField<scalar> dk = freshField("k"), de = freshField("epsilon"), dn = freshField("nut");
    dk.evaluateBoundary();
    de.evaluateBoundary();
    DeviceBoundary dbK   = buildDeviceBoundary(dk, fvp, g);
    DeviceBoundary dbEps = buildDeviceBoundary(de, fvp, g);

    // THE WALL PREDICATE IS THE BC's, NOT THE PATCH TYPE's. A `wall` carrying a plain zeroGradient
    // epsilon is not constrained by OpenFOAM, and the 4-arg buildDeviceWallData overload -- which falls
    // back to the type alone -- is the one that once pinned 545 cells on a far-field wall.
    std::vector<char> wfPatch(fvp.size(), 0);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        wfPatch[pi] = de.boundary[pi]->isTurbulenceWallFunction() ? 1 : 0;
    const DeviceWallData wall = buildDeviceWallData(m, g, fvp, U, wfPatch);

    // The per-BOUNDARY-FACE wall mask and near-wall distance. correctNut needs the face question, not
    // the cell one: a cell can touch a wall and an inlet at once and those two faces get different nut.
    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
    std::vector<label>  wfMaskH;
    std::vector<scalar> yBndH, nuWallH;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const bool isWF = isTurbWallPatch(fvp, pi, wfPatch);
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            wfMaskH.push_back(isWF ? 1 : 0);
            yBndH.push_back(isWF ? yW[pi][i] : 0.0);
            if (isWF) nuWallH.push_back(nuB[pi][i]);   // WALL-face order, matching DeviceWallData
        }
    }
    wfMaskH.resize(dm.nBndFaces, 0);
    yBndH.resize(dm.nBndFaces, 0.0);

    const std::vector<scalar> nutBndH = flatten([&]{
        std::vector<std::vector<scalar>> v(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) v[pi] = dn.boundary[pi]->value();
        return v;
    }(), fvp, dm.nBndFaces, 0.0);

    DeviceBuffer<scalar> dUx, dUy, dUz;
    {
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c) { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
        dUx.copyFrom(ux); dUy.copyFrom(uy); dUz.copyFrom(uz);
    }
    DeviceBuffer<scalar> dPhiInt(phi.internal), dPhiBnd(flatten(phi.boundary, fvp, dm.nBndFaces, 0.0));
    DeviceBuffer<scalar> dPbrInt(phiByRho.internal), dPbrBnd(flatten(phiByRho.boundary, fvp, dm.nBndFaces, 0.0));
    DeviceBuffer<scalar> dRhoC(rhoC), dNuC(nuC);
    DeviceBuffer<scalar> dRhoB(flatten(rhoB, fvp, dm.nBndFaces, 1.0));
    DeviceBuffer<scalar> dNuB(flatten(nuB, fvp, dm.nBndFaces, 1.5e-5));
    DeviceBuffer<scalar> dNuWall(nuWallH), dNutBnd(nutBndH);
    DeviceBuffer<label>  dWfMask(wfMaskH);
    DeviceBuffer<scalar> dYBnd(yBndH);
    DeviceBuffer<scalar> gK(dk.internal), gE(de.internal), gN(dn.internal), gAlphat;
    DeviceBuffer<scalar> gNutBnd(nutBndH);

    gpu::kEpsilonRAS::KEpsilonInput gin;
    gin.phiInt = &dPhiInt;           gin.phiBnd = &dPhiBnd;
    gin.phiByRhoInt = &dPbrInt;      gin.phiByRhoBnd = &dPbrBnd;
    gin.rhoCell = &dRhoC;            gin.rhoBndFace = &dRhoB;
    gin.nuCell = &dNuC;              gin.nuBndFace = &dNuB;
    gin.nuWallFace = &dNuWall;
    gin.nutBndFace = &dNutBnd;
    gin.wfBndMask = &dWfMask;        gin.wallYBndFace = &dYBnd;
    gin.Ux = &dUx; gin.Uy = &dUy; gin.Uz = &dUz;
    gin.boundedK = true;             gin.boundedEps = true;
    gin.correctedLaplacian = true;
    gin.relaxEquationEps = true;     gin.relaxEps = relaxEps;
    gin.relaxEquationK = true;       gin.relaxK = relaxK;
    gin.tol = tol; gin.relTol = relTol; gin.maxIter = maxIter;
    gin.co = co;
    gin.Prt = 0.85;

    // production() on its own FIRST, so the PRE-wall G has somewhere to be seen. correct() overwrites G
    // in place at wall cells -- which is what OpenFOAM does and what this module must do -- so after it
    // runs, st.G is the post-override field and the pre-override one no longer exists on the device.
    // Both are worth pinning: they differ by 77% here, so a gate comparing only one of them would be
    // blind to the entire wall treatment or to the entire production term depending on which it picked.
    gpu::kEpsilonRAS::KEpsilonStages pre;
    {
        DeviceBuffer<scalar> nPre(dn.internal);
        gpu::kEpsilonRAS::production(pre, dm, dbU, nPre, gin);
    }

    gpu::kEpsilonRAS::KEpsilonStages st;
    gpu::kEpsilonRAS::correct(gK, gE, gN, gNutBnd, &gAlphat, st, dm, dbU, dbK, dbEps, wall, gin);

    std::printf("kEpsilon CUDA vs _cpp   cells=%d  wall cells=%d/%d\n",
                (int)nC, (int)st.wallCells, (int)res.wallCells);

    // ---- stage by stage -----------------------------------------------------------------------
    std::printf("  1. production\n");
    cmp(pre.gByNu.host(), res.gByNu, "GbyNu = gradU && devTwoSymm(gradU)", 1e-12);
    cmp(pre.divU.host(),  res.divU,  "divU (VOLUMETRIC flux)", 1e-12);
    {
        scalar sc = 0;
        for (scalar v : res.divU) sc = std::fmax(sc, std::fabs(v));
        cmpScaled(pre.divPhi.host(), res.divPhi, "divPhi (MASS flux, for bounded)", sc, 1e-12);
    }
    cmp(pre.G.host(), res.G, "G = nut*GbyNu, PRE-wall", 1e-12);

    std::printf("  2. the wall treatment\n");
    check(res.wallCells > 0, "the fixture HAS wall-function cells (else the wall stage is vacuous)");
    cmp(st.eps0.host(), res.eps0, "epsilonWallFunction eps0", 1e-12);
    cmp(st.G0.host(),   res.G0,   "epsilonWallFunction G0",   1e-12);
    cmp(st.G.host(),    res.Gwall, "G AFTER the wall override", 1e-12);
    {
        std::vector<scalar> gm, rm;
        const std::vector<label> gw = st.isWallCell.host();
        for (label c = 0; c < nC; ++c) { gm.push_back((scalar)gw[c]); rm.push_back((scalar)res.isWallCell[c]); }
        cmp(gm, rm, "the constrained cell set", 0.0);
    }

    std::printf("  3. the solved fields\n");
    cmp(gE.host(), he.internal, "epsilon", 1e-9);
    cmp(gK.host(), hk.internal, "k",       1e-9);
    // nut = Cmu*k^2/epsilon, so its relative error is 2*relerr(k) + relerr(epsilon) -- measured here as
    // 2*2.46e-11 + 8.99e-11 = 1.39e-10 against an observed 1.36e-10, which is the propagation and
    // nothing else. Bounding nut BELOW the fields it is computed from would assert an accuracy the
    // inputs cannot supply; it carries k and epsilon's own bound. What pins the ARITHMETIC is the stage
    // block above, at 1e-12 to 1e-17, where no Krylov solver sits between the two paths.
    cmp(gN.host(), hn.internal, "nut = Cmu k^2/eps", 1e-9);
    cmp(gAlphat.host(), hAlphat, "alphat = rho*nut/Prt", 1e-9);

    // ---- CONTROL: the mass flux must NOT serve as divU ----------------------------------------
    // divU is a dilatation and takes the volumetric flux; the div operator takes the mass flux. They
    // differ by rho, and in the incompressible lineage they are the same field -- so without a varying
    // rho this control cannot discriminate, and the check below would pass vacuously.
    {
        GeometricField<scalar> ck = freshField("k"), ce = freshField("epsilon"), cn = freshField("nut");
        ck.evaluateBoundary();
        ce.evaluateBoundary();
        std::vector<scalar> cA(nC, 0.0);
        cpu::kEpsilonRef::Compressible wrong = comp;
        wrong.phiByRho = &phi;                       // the MASS flux, which is the defect
        wrong.alphat   = &cA;
        cpu::kEpsilonRef::KEResiduals cres;
        cpu::kEpsilonRef::correct(U, ck, ce, cn, phi, 0.0, m, g, fvp, relaxEps, relaxK, tol, relTol,
                                  maxIter, co, &cres, true, 0, &wrong, nullptr);
        const scalar rE = relDiff(he.internal, ce.internal);
        const scalar rK = relDiff(hk.internal, ck.internal);
        std::printf("     %-58s eps=%.3e k=%.3e\n", "control: mass-flux divU moves the answer", (double)rE, (double)rK);
        check(rE > 1e-6, "divU takes the VOLUMETRIC flux (control)");
        check(rK > 1e-6, "...and it reaches k too (control)");
    }

    // ---- CONTROL: nut's boundary is the PATCH value, not the owner cell's ---------------------
    // DkEff/DepsilonEff are volScalarFields, so fvm::laplacian takes their patch values. Replacing the
    // patch array with the owner cells' must move the answer, or a kernel reading the cell would pass.
    {
        GeometricField<scalar> ck = freshField("k"), ce = freshField("epsilon"), cn = freshField("nut");
        ck.evaluateBoundary();
        ce.evaluateBoundary();
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            std::vector<scalar> b(fvp[pi].size);
            for (label i = 0; i < fvp[pi].size; ++i) b[i] = cn.internal[fvp[pi].faceCells[i]];
            cn.boundary[pi]->setValue(b);
        }
        std::vector<scalar> cA(nC, 0.0);
        cpu::kEpsilonRef::Compressible c2 = comp;
        c2.alphat = &cA;
        cpu::kEpsilonRef::KEResiduals cres;
        cpu::kEpsilonRef::correct(U, ck, ce, cn, phi, 0.0, m, g, fvp, relaxEps, relaxK, tol, relTol,
                                  maxIter, co, &cres, true, 0, &c2, nullptr);
        const scalar r = relDiff(he.internal, ce.internal);
        std::printf("     %-58s rel=%.3e\n", "control: owner-cell nut boundary moves the answer", (double)r);
        check(r > 1e-8, "the diffusivity's boundary is nut's PATCH value (control)");
    }

    // ---- CONTROL: the wall treatment must MATTER ----------------------------------------------
    // Without it, `G` keeps its interior value and epsilon is not pinned at wall cells. If that changed
    // nothing, every comparison above would be measuring an inert stage.
    {
        GeometricField<scalar> ck = freshField("k"), ce = freshField("epsilon"), cn = freshField("nut");
        ck.evaluateBoundary();
        ce.evaluateBoundary();
        // dropTerm 5 removes the k equation's production, which is where G enters -- the cheapest probe
        // that the wall-overridden G is load-bearing rather than decorative.
        std::vector<scalar> cA(nC, 0.0);
        cpu::kEpsilonRef::Compressible c3 = comp;
        c3.alphat = &cA;
        cpu::kEpsilonRef::KEResiduals cres;
        cpu::kEpsilonRef::correct(U, ck, ce, cn, phi, 0.0, m, g, fvp, relaxEps, relaxK, tol, relTol,
                                  maxIter, co, &cres, true, /*dropTerm=*/5, &c3, nullptr);
        const scalar r = relDiff(hk.internal, ck.internal);
        std::printf("     %-58s rel=%.3e\n", "control: dropping G changes k", (double)r);
        check(r > 1e-6, "the production term G is load-bearing (control)");
    }

    // ---- CONTROL: the SECOND correct() must match the first -----------------------------------
    // DeviceBuffer::resize takes from a pool whose contract is "returned memory is NOT zeroed", and the
    // FIRST call is always right because the pool has no same-size block to hand back yet. Only a second
    // call in the same process reads another stage's leavings, so a single-assembly test cannot see the
    // defect at all. Both sides are re-run from the same inputs and required to agree with each other.
    {
        GeometricField<scalar> h2k = freshField("k"), h2e = freshField("epsilon"), h2n = freshField("nut");
        h2k.evaluateBoundary();
        h2e.evaluateBoundary();
        std::vector<scalar> h2A(nC, 0.0);
        cpu::kEpsilonRef::Compressible c4 = comp;
        c4.alphat = &h2A;
        cpu::kEpsilonRef::KEResiduals r2;
        cpu::kEpsilonRef::correct(U, h2k, h2e, h2n, phi, 0.0, m, g, fvp, relaxEps, relaxK, tol, relTol,
                                  maxIter, co, &r2, true, 0, &c4, nullptr);

        DeviceBuffer<scalar> g2K(dk.internal), g2E(de.internal), g2N(dn.internal), g2A;
        DeviceBuffer<scalar> g2NutBnd(nutBndH);
        DeviceBoundary db2K   = buildDeviceBoundary(dk, fvp, g);
        DeviceBoundary db2Eps = buildDeviceBoundary(de, fvp, g);
        gpu::kEpsilonRAS::KEpsilonStages st2;
        gpu::kEpsilonRAS::correct(g2K, g2E, g2N, g2NutBnd, &g2A, st2, dm, dbU, db2K, db2Eps, wall, gin);

        cmp(g2E.host(), h2e.internal, "epsilon, SECOND correct() in-process", 1e-9);
        cmp(g2K.host(), h2k.internal, "k, SECOND correct() in-process",       1e-9);
        cmp(g2E.host(), gE.host(),    "epsilon, run 2 against run 1",         1e-14);
        cmp(g2K.host(), gK.host(),    "k, run 2 against run 1",               1e-14);
    }

    // ---- THE REFUSALS -------------------------------------------------------------------------
    // A comment is not a refusal. Each of these must throw, and the NEGATIVE CONTROL is that the same
    // call with the flag cleared does not -- otherwise the test would pass because everything throws.
    std::printf("  refusals\n");
    {
        struct Case { const char* name; gpu::kEpsilonRAS::KEpsilonInput in; };
        std::vector<Case> cases;
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.co.rng = true;
            cases.push_back({"RNGkEpsilon", r});
        }
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.co.realizable = true;
            cases.push_back({"realizableKE", r});
        }
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.hasCoupledPatches = true;
            cases.push_back({"coupled patches", r});
        }
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.hasUnportedFvOption = true;
            cases.push_back({"an unported fvOption", r});
        }
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.hasNonUpwindDivScheme = true;
            cases.push_back({"a non-upwind div scheme", r});
        }
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.hasNonWallTurbWallFunc = true;
            cases.push_back({"a wall function on a non-wall patch", r});
        }
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.boundedK = false;
            cases.push_back({"bounded on one equation only", r});
        }
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.nuWallFace = nullptr;
            cases.push_back({"a missing wall-face nu", r});
        }
        {
            gpu::kEpsilonRAS::KEpsilonInput r = gin; r.wfBndMask = nullptr;
            cases.push_back({"a missing wall-face mask", r});
        }
        for (const Case& c : cases)
        {
            bool threw = false;
            try
            {
                gpu::kEpsilonRAS::KEpsilonStages s;
                DeviceBuffer<scalar> a, b, n, nb(nutBndH), al;
                a.copyFrom(dk.internal); b.copyFrom(de.internal); n.copyFrom(dn.internal);
                gpu::kEpsilonRAS::production(s, dm, dbU, n, c.in);
            }
            catch (const std::exception&) { threw = true; }
            std::printf("     %-58s %s\n", c.name, threw ? "REFUSED" : "ACCEPTED");
            if (!threw) ++g_fails;
        }
        // The negative control: the unmodified input must NOT throw, or every line above passes for the
        // wrong reason.
        bool threw = false;
        try
        {
            gpu::kEpsilonRAS::KEpsilonStages s;
            DeviceBuffer<scalar> n(dn.internal);
            gpu::kEpsilonRAS::production(s, dm, dbU, n, gin);
        }
        catch (const std::exception&) { threw = true; }
        check(!threw, "the supported configuration is ACCEPTED (negative control)");
    }

    std::printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
