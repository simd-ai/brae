#include "turbulence_transport.cuh"
#include "device_pbicg.cuh"
#include <algorithm>
#include <memory>   // FieldGrad: one gradient per (scheme, limiter) pair an assembly asks for (FP-3)
#include <vector>   // std::max
#include "device_mesh.cuh"     // deviceDivUpwindCoeffs / deviceDivLimitedCoeffs / deviceLaplacian*
#include "device_kepsilon.cuh" // deviceGaussGrad, deviceCellLimitGrad, deviceBCValue
#include "device_blas.cuh"     // deviceAxpy
#include "device_fvoptions.cuh"   // deviceSetValues: fvMatrix::setValues
#include "device_pcg.cuh"
#include "device_amg.cuh"      // deviceSymGaussSeidel, when the case names a smoothSolver
#include "device_simple.cuh"    // deviceRelaxDiag -- fvMatrix::relax
#include "stage_dump.cuh"   // the laplacian stage dump below; free when BRAE_DUMP_STAGE is unset
#include <cstdio>
#include <string>
#include <map>       // FP-10: the folded system kept per field so the solver's graph reads it in place
#include <utility>
#include <cmath>

namespace brae {
namespace gpu {
namespace turbulence {

namespace {

// FP-2: M -= the laplacian coefficients, five arrays of three lengths in one launch. Each statement is
// axpyKernel's `y += a * x` with a = -1, so nvcc emits the same fused multiply-add and the bits match the
// five separate launches this replaces (held by the SST dumps and written fields on aerofoilNACA0012 and
// the residual lines on squareBend and injectorPipe, all bit-identical before and after).
__global__ void subtractLaplacianKernel(
    int nC, int nF, int nB,
    const scalar* __restrict__ lDiag, const scalar* __restrict__ lUp, const scalar* __restrict__ lLo,
    const scalar* __restrict__ lIC, const scalar* __restrict__ lBC,
    scalar* __restrict__ diag, scalar* __restrict__ upper, scalar* __restrict__ lower,
    scalar* __restrict__ iC, scalar* __restrict__ bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const scalar a = scalar(-1.0);
    if (i < nC) diag[i] += a * lDiag[i];
    if (i < nF) { upper[i] += a * lUp[i]; lower[i] += a * lLo[i]; }
    if (i < nB) { iC[i] += a * lIC[i]; bC[i] += a * lBC[i]; }
}

void subtractLaplacian(
    const DeviceBuffer<scalar>& lDiag, const DeviceBuffer<scalar>& lUp, const DeviceBuffer<scalar>& lLo,
    const DeviceBuffer<scalar>& lIC, const DeviceBuffer<scalar>& lBC, PressureMatrix& M)
{
    const int nC = static_cast<int>(lDiag.size()), nF = static_cast<int>(lUp.size()), nB = static_cast<int>(lIC.size());
    const int n = std::max(nC, std::max(nF, nB));
    if (n <= 0) return;
    constexpr int tpb = 256;
    subtractLaplacianKernel<<<(n + tpb - 1) / tpb, tpb>>>(
        nC, nF, nB, lDiag.data(), lUp.data(), lLo.data(), lIC.data(), lBC.data(),
        M.diag.data(), M.upper.data(), M.lower.data(), M.iC.data(), M.bC.data());
    cudaCheck(cudaGetLastError(), "subtractLaplacian");
}
// Each device module in the tree carries its own copy of this two-liner (rhoEEqn.cu, rhoPEqn.cu,
// rhoPcEqn.cu, kEpsilon.cu). Kept local here for the same reason rather than adding a public symbol
// for a memset.
void zeroed(DeviceBuffer<scalar>& b, int n)
{
    b.resize(static_cast<std::size_t>(n));
    cudaCheck(cudaMemsetAsync(b.data(), 0, static_cast<std::size_t>(n) * sizeof(scalar), cudaStreamPerThread),
              "turbulence transport zero");
}
}

void assembleScalarTransport(
    PressureMatrix&             M,
    const DeviceMesh&           dm,
    const DeviceBoundary&       db,
    const DeviceBuffer<scalar>& field,
    const DeviceBuffer<scalar>& gammaFace,
    const DeviceBuffer<scalar>& gammaBnd,
    const TransportScheme&      sc)
{
    const int nC = dm.nCells;

    // fvm::div(phi, field). The boundary half carries the flux-conditional switch the caller has
    // already applied to db.
    //
    // limitedLinear is a WEIGHT change, not a correction, so it replaces the upwind coefficients rather
    // than adding to the source -- the same shape the host closures' divWithScheme takes. The limiter's
    // gradient is the field's own Gauss gradient, limited by the case's grad(<field>) cellLimited
    // coefficient when it names one; the corrected-laplacian block below builds the same three buffers
    // the same way, and this is deliberately the identical call sequence so the two cannot drift.
    // FP-3: the field's boundary values and its gradient are evaluated ONCE per (base scheme,
    // cellLimited coefficient) pair this assembly asks for, and every site asking for the same pair
    // reads the same buffers. The limitedLinear limiter and the corrected laplacian both take the
    // case's grad(<field>) entry, so on gasMixing/injectorPipe (default leastSquares) each closure
    // fitted grad(k) and grad(epsilon) twice and evaluated the boundary values three times per field.
    // Nothing is reassociated -- the same kernels on the same inputs, so the bits are the ones each
    // site computed for itself -- and `field` is not written between the sites (the matrix is).
    struct FieldGrad
    {
        bool leastSq = false;
        scalar limitK = 0.0;
        DeviceBuffer<scalar> gx, gy, gz;
    };
    DeviceBuffer<scalar> bval;
    bool haveBval = false;
    std::vector<std::unique_ptr<FieldGrad>> grads;
    auto fieldBval = [&]() -> const DeviceBuffer<scalar>&
    {
        if (!haveBval)
        {
            if (sc.bndValues) deviceCopy(bval, *sc.bndValues);
            else              deviceBCValue(db, field, bval);
            haveBval = true;
        }
        return bval;
    };
    auto fieldGrad = [&](bool leastSq, scalar limitK) -> const FieldGrad&
    {
        for (const auto& g : grads)
        {
            if (g->leastSq == leastSq && g->limitK == limitK) return *g;
        }
        const DeviceBuffer<scalar>& bv = fieldBval();
        auto g = std::make_unique<FieldGrad>();
        g->leastSq = leastSq;
        g->limitK = limitK;
        if (leastSq) deviceLeastSquaresGrad(dm, field, bv, g->gx, g->gy, g->gz);
        else         deviceGaussGrad(dm, field, bv, g->gx, g->gy, g->gz);
        if (limitK > scalar(0)) deviceCellLimitGrad(dm, field, bv, g->gx, g->gy, g->gz, limitK);
        grads.push_back(std::move(g));
        return *grads.back();
    };

    if (sc.limitedLinear)
    {
        // The limiter's gradient takes the case's OWN gradScheme for this field -- OpenFOAM builds it
        // through fvc::grad(lPhi) (LimitedScheme.C:56-59), not through a scheme the closure chooses.
        const FieldGrad& g = fieldGrad(sc.limGradLeastSq, sc.limGradK);
        deviceDivLimitedCoeffs(dm, *sc.phiInt, field, g.gx, g.gy, g.gz,
                               scalar(2) / std::fmax(sc.limiterCoeff, scalar(1e-15)),
                               M.diag, M.upper, M.lower);
    }
    else
    {
        deviceDivUpwindCoeffs(dm, *sc.phiInt, M.diag, M.upper, M.lower);
    }
    zeroed(M.source, nC);
    deviceBCDivCoeffs(db, *sc.phiBnd, M.iC, M.bC);

    // linearUpwind's explicit correction, part of the same fvm::div object -- SUBTRACTED, because
    // `fvm += fvc::surfaceIntegrate(...)` is `source -= V*su` (fvMatrix.C:1855-1862), exactly as the
    // momentum equation's deviceAxpy(-corrFac, lu, source). Internal faces only: linearUpwind::correction
    // leaves every uncoupled boundary face at zero.
    if (sc.linearUpwind)
    {
        DeviceBuffer<scalar> lu;
        const FieldGrad& g = fieldGrad(false, sc.luGradK);
        deviceLinearUpwindCorr(dm, *sc.phiInt, g.gx, g.gy, g.gz, lu);
        deviceAxpy(-1.0, lu, M.source);
    }

    // ...AND THE PAIR'S OWN COEFFICIENT, for both halves at once. The equation is
    // fvm::div(phi, field) - fvm::laplacian(gamma, field), which is the momentum equation's shape, so
    // the interface coefficient is the momentum's: -gamma_f*dc*magSf + phi*(1 - w) into ifCoeff and
    // +gamma_f*dc*magSf + phi*w into the diagonal (device_cyclic.cu's momKernel). gamma_f there is the
    // two CELLS interpolated, which is what a coupled face takes (kEpsilon_cpp.cu:152-158) -- a patch
    // value would be a different number. The div scheme's weights reach it the same way they reach the
    // internal faces: null is upwind.
    //
    // It goes in BEFORE the laplacian block below because that block SUBTRACTS its own arrays from M,
    // and the pair's contribution is already signed for the assembled equation.
    if (sc.cyc && sc.cyc->n > 0)
    {
        if (!sc.gammaCell || !sc.cycPhi)
        {
            throw std::runtime_error(
                "brae turbulence transport: the mesh has a periodic pair and the caller gave no CELL "
                "diffusivity or no flux for it. A coupled face takes fvc::interpolate's value -- the "
                "two cells' -- so the face array cannot stand in for the first, and the equation's own "
                "flux on those faces is not the internal-face array.");
        }
        // THE PAIR TAKES UPWIND'S WEIGHT, so a case whose div scheme is not upwind is refused rather
        // than run with one scheme inside and another across the pair. deviceCyclicLimitedWeights
        // exists for the alpha flux and would serve here; nothing has needed it yet.
        if (sc.limitedLinear || sc.linearUpwind)
        {
            throw std::runtime_error(
                "brae turbulence transport: the mesh has a periodic pair and the case's div scheme for "
                "this field is not upwind. The pair's interface coefficient is built with upwind's "
                "weight, so running it would put one scheme on the internal faces and another across "
                "the pair. The host closure carries the case's scheme on both.");
        }
        deviceCyclicAssembleMomentum(*sc.cyc, *sc.gammaCell, M.diag, /*wsch=*/nullptr,
                                     sc.correctedLaplacian, sc.cycPhi);
    }

    // - fvm::laplacian(gamma, field).
    {
        DeviceBuffer<scalar> lDiag, lUp, lLo, lIC, lBC, lapSrc;
        deviceLaplacianCoeffs(dm, gammaFace, lDiag, lUp, lLo, sc.correctedLaplacian);
        deviceBCLaplacianCoeffsFace(db, gammaBnd, lIC, lBC);
        // FP-2: the five `axpy(-1, l, M)` subtractions in one launch (subtractLaplacianKernel above),
        // the same `y += a*x` statement per array, so the same doubles.
        subtractLaplacian(lDiag, lUp, lLo, lIC, lBC, M);

        if (sc.correctedLaplacian)
        {
            DeviceBuffer<scalar> ffc, corr;
            // correctedSnGrad's correction takes the field's OWN grad scheme (correctedSnGrad.C:52-55):
            // its base (leastSquares or Gauss linear) and its cellLimited coefficient, both from the
            // case's grad(<field>) entry. The host twin is kEpsilon_cpp.cu's laplacian block.
            const FieldGrad& g = fieldGrad(sc.gradFieldLeastSq, sc.gradFieldLimitK);
            if (sc.snGradLimitCoeff > scalar(0.0))
            {
                deviceLaplacianCorrFluxLimited(dm, gammaFace, field, g.gx, g.gy, g.gz, sc.snGradLimitCoeff, ffc);
                deviceFaceDivSource(dm, ffc, corr);
            }
            else
            {
                deviceLaplacianCorr(dm, gammaFace, g.gx, g.gy, g.gz, corr);
            }
            // deviceLaplacianCorr returns -V*div(faceFluxCorr) -- already negated -- and the laplacian
            // itself enters this equation with -1, so its explicit source does too. The two signs
            // compose to the reference's `L.source -= corr` followed by `M -= L`.
            deviceAxpy(-1.0, corr, M.source);
            lapSrc = std::move(corr);
        }

        // The laplacian ON ITS OWN, as OpenFOAM's tools/dumpKEpsilon writes it (kEpsilonDump.C:348
        // dumps fvm::laplacian(alpha*rho*DepsilonEff(), epsilon_) as a matrix of its own). SIGNS: the
        // host reference builds L with `L.source -= corr_host` and then does `M -= L`; this arm's
        // `corr` is already negated for the composed `M.source -= corr` above, so L.source in the
        // reference's convention IS this `corr`. Nothing here is derived -- it is the same four arrays
        // the solve uses, scattered the way captureSystem scatters them.
        if (sc.stageTag && stageDumpActive() && stageDumpFirstOnly(sc.stageTag))
        {
            const std::vector<label>  bc  = dm.bndCell.host();
            const std::vector<scalar> hD  = lDiag.host();
            const std::vector<scalar> hIC = lIC.host();
            const std::vector<scalar> hBC = lBC.host();
            const std::vector<scalar> hSr = lapSrc.size() ? lapSrc.host() : std::vector<scalar>(hD.size(), scalar(0));
            std::vector<scalar> D(hD), S(hSr);
            for (std::size_t f = 0; f < bc.size() && f < hIC.size(); ++f)
            {
                D[static_cast<std::size_t>(bc[f])] += hIC[f];
                S[static_cast<std::size_t>(bc[f])] += hBC[f];
            }
            const std::string tag(sc.stageTag);
            stageDump(tag + "LapD",      D);
            stageDump(tag + "LapSrc",    S);
            stageDump(tag + "LapDUpper", lUp);
            stageDump(tag + "LapDLower", lLo);
        }
    }
}

namespace
{
__global__ void wallFacesTakeCellKernel(
    int            nB,
    const label*   wfMask,
    const label*   bndCell,
    const scalar*  cell,
    scalar*        bnd)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nB || !wfMask[f]) return;
    bnd[f] = cell[bndCell[f]];
}
} // namespace

void wallFacesTakeCell(
    const DeviceMesh&           dm,
    const DeviceBuffer<label>&  wfMask,
    const DeviceBuffer<scalar>& field,
    DeviceBuffer<scalar>&       bnd)
{
    const int nB = static_cast<int>(bnd.size());
    if (nB == 0) return;
    const int tpb = 256;
    wallFacesTakeCellKernel<<<(nB + tpb - 1) / tpb, tpb>>>(nB, wfMask.data(), dm.bndCell.data(),
                                                           field.data(), bnd.data());
    cudaCheck(cudaGetLastError(), "turbulence wallFacesTakeCell");
}

void solveScalarEqn(
    PressureMatrix&             M,
    DeviceBuffer<scalar>&       field,
    const DeviceMesh&           dm,
    bool                        relaxEquation,
    scalar                      alpha,
    const DeviceBuffer<label>*  fvoMask,
    const DeviceBuffer<scalar>* fvoVal,
    const DeviceBuffer<label>*  wallMask,
    const DeviceBuffer<scalar>* wallVal,
    const SolveControls&        sv,
    scalar&                     residualOut,
    const std::string&          dumpPrefix,   // "" = no dump; else <dir>/<name> path prefix
    // this field's own solver: the case's smoothSolver, or BiCGStab
    bool gs,
    DeviceSolverPerf* perfOut,
    DeviceCyclic* cyc)
{
    const int nC  = dm.nCells;
    const int nIf = dm.nInternalFaces;
    const int nB  = dm.nBndFaces;

    // The guard is "the case NAMES a factor", not "the factor is below 1": fvMatrix::relax early-returns
    // only on alpha <= 0, so relax(1.0) still applies the dominance clamp and adds (D - D0)*psi.
    if (relaxEquation && alpha > scalar(0.0))
    {
        DeviceBuffer<scalar> relaxedDiag, delta, t;
        deviceRelaxDiag(M.view(dm), dm, M.iC, alpha, relaxedDiag, delta);
        deviceCopy(M.diag, relaxedDiag);
        deviceHadamard(t, delta, field);
        deviceAxpy(1.0, t, M.source);
    }

    // Both constraints go through the SAME four kernels, because in OpenFOAM they are the same call --
    // FixedValueConstraint::constrain and epsilonWallFunction::manipulateMatrix both end in
    // fvMatrix::setValues. Order matters and is OpenFOAM's: whichever runs first is the one whose value
    // reaches the neighbours, since setValues zeroes the coefficient it just transferred through.
    auto applySetValues = [&](const DeviceBuffer<label>* mask, const DeviceBuffer<scalar>* val)
    {
        if (!mask || !val) return;
        deviceSetValues(dm, *mask, *val, M.diag, M.upper, M.lower, M.source, M.iC, M.bC, field, cyc);
    };
    applySetValues(fvoMask, fvoVal);
    applySetValues(wallMask, wallVal);

    // Fold the boundary coefficients in exactly as fvMatrix::solve does, then solve. BiCGStab, not PCG:
    // upwind convection makes upper != lower, so the matrix is asymmetric and a symmetric solver would
    // be solving a different system.
    // FP-10: the solver captures its graph against the addresses it was handed, so a folded system that
    // moves every step is copied into the graph's own buffers instead of read where it lies. The pair is
    // kept per field -- the same key the solver's graph cache uses -- and foldKernel writes every cell,
    // so reusing them changes no bits. 12 -> 4 face-sized copies per iteration at 896k.
    static auto& foldCache = *new std::map<const void*, std::pair<DeviceBuffer<scalar>, DeviceBuffer<scalar>>>();
    auto& fold = foldCache[field.data()];
    DeviceBuffer<scalar>& diagC = fold.first;
    DeviceBuffer<scalar>& b     = fold.second;
    deviceFold(dm, M.diag, M.source, M.iC, M.bC, diagC, b);

    DeviceLduView A{};
    A.nCells = dm.nCells;
    A.nInternalFaces = dm.nInternalFaces;
    A.diag = diagC.data();
    A.upper = M.upper.data();
    A.lower = M.lower.data();
    A.owner = dm.owner.data();
    A.nei = dm.nei.data();
    A.ownerStart = dm.ownerStart.data();
    A.losort = dm.losort.data();
    A.losortStart = dm.losortStart.data();
    // ...and the PAIR's off-diagonal, which deviceAmul applies as Apsi[own] += ifCoeff*psi[nbr]. The
    // fold above adds the boundary diagonal only; a coupled patch has none there (deviceFold walks
    // the device's boundary arrays, which hold no coupled face) because its diagonal went straight
    // into M.diag with the coefficient. Without this the matrix and the solve are different
    // operators -- the defect the pressure step's own view note records.
    if (cyc && cyc->n > 0)
    {
        A.nCyc = cyc->n;
        A.cycOwn = cyc->ownCell.data();
        A.cycNbr = cyc->nbrCell.data();
        A.cycCoeff = cyc->ifCoeff.data();
    }

    // Instrument (BRAE_STAGE_DUMP_DIR, see correct()): the FOLDED system as the solver sees it -- diag
    // with the boundary diagonal folded in, source with the boundary source folded in, the two
    // off-diagonals (<prefix>D/Src/Upper/Lower), and the field before and after the solve
    // (<prefix>SolveIn/SolveOut) -- the same convention the host's
    // captureSystem uses (kEpsilon_cpp.cu), so the two arms' assembled systems diff directly.
    auto dump = [&](const char* what, const DeviceBuffer<scalar>& v)
    {
        if (dumpPrefix.empty()) return;
        const std::vector<scalar> h = v.host();
        std::FILE* fp = std::fopen((dumpPrefix + what).c_str(), "w");
        if (!fp) return;
        for (scalar x : h) std::fprintf(fp, "%.17g\n", (double)x);
        std::fclose(fp);
    };
    dump("D", diagC);
    dump("Src", b);
    dump("Upper", M.upper);
    dump("Lower", M.lower);
    dump("SolveIn", field);

    // the ones vector kept across calls (item 63) and the normFactor kept on the device (item 66)
    DeviceBuffer<scalar> dnf;
    deviceNormFactorInto(A, field, b, deviceOnes(nC), dnf);
    // The solver the case asked for (item 58). The view above is internal-face only, which is what the
    // level-scheduled sweep needs; there is no interface to drop silently.
    DeviceSolverPerf perf;
    if (gs && sv.gsColour)
    {
        // FP-1: the case's smoothSolver swept in COLOUR order, one component through the momentum engine.
        // The driver announced the order; refuse rather than run something else without the colouring.
        if (!sv.colouring || !sv.colouring->valid)
            throw std::runtime_error(
                "brae turbulence: the colour-order smoothSolver was selected for a transported scalar but "
                "SolveControls::colouring is null or invalid; refusing rather than running a solver the "
                "notice did not name");
        GSFusedComponent one;
        one.A = &A; one.b = &b; one.psi = &field; one.normFactor = 1.0; one.dNormFactor = dnf.data();
        deviceColourGaussSeidelFused(1, &one, *sv.colouring, sv.tol, sv.relTol, sv.maxIter, sv.minIter,
                                     sv.nSweeps, sv.gsSymmetric, &perf);
    }
    else if (gs)
        deviceSymGaussSeidel(A, b, field, dnf.data(), sv.tol, sv.relTol, sv.maxIter, &perf, sv.minIter,
                             sv.nSweeps, sv.gsSymmetric);
    else if (sv.pbicg)
    {
        if (!sv.precon || !sv.precon->valid)
            throw std::runtime_error(
                "brae turbulence: PBiCG was selected for a transported scalar and SolveControls::precon "
                "carries no DILU schedule; PBiCG here is OpenFOAM's PBiCG WITH DILU and nothing else.");
        // the normFactor is on the device for the other two solvers; this one's recurrence runs on
        // host scalars (device_pbicg.cu), so it is read once. rD is rebuilt from THIS matrix inside
        // the solve, which is why the schedule is not const there (device_pcg.cu:604 does the same).
        std::vector<scalar> nf;
        dnf.copyTo(nf);
        perf = devicePBiCGDilu(A, b, field, nf.at(0), sv.tol, sv.relTol, sv.maxIter, sv.minIter,
                               *const_cast<DeviceDilu*>(sv.precon));
    }
    else
        perf = deviceJacobiBiCGStab(A, b, field, dnf.data(), sv.tol, sv.relTol, sv.maxIter, /*checkEvery=*/1, sv.minIter,
                                    sv.precon, /*amg=*/nullptr, sv.polyDeg);
    residualOut = perf.initialResidual;
    if (perfOut)
    {
        *perfOut = perf;
    }
    dump("SolveOut", field);
}


} // namespace turbulence
} // namespace gpu
} // namespace brae
