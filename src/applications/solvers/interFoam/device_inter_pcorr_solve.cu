#include "inter_phase_time.cuh"
#include "device_inter_pcorr_solve.cuh"
#include "device_inter_pressure_step.cuh"   // deviceAmgPcgHierarchy
#include "device_blas.cuh"
#include "device_amg_split.cuh"
#include "device_ldu.cuh"
#include "fv_matrix_ops.cuh"   // matrixFlux, the check's oracle
#include "device_mesh.cuh"   // nextDeviceAddressingId
#include "device_pcg.cuh"    // deviceNormFactor
#include <optional>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace brae {

bool DevicePcorrSolver::takes(
    const std::string& asked,
    const std::vector<FvPatch>& patches)
{
    for (const FvPatch& p : patches)
    {
        if (p.coupled && p.type != "cyclic" && p.type != "cyclicAMI")
        {
            if (!announcedCoupled)
            {
                announcedCoupled = true;
                std::printf("  pcorr: the mesh has the coupled patch '%s' of type %s; pcorr keeps the case's own "
                            "solver, which carries the pair\n", p.name.c_str(), p.type.c_str());
            }
            return false;
        }
    }
    if (!announced)
    {
        announced = true;
        std::printf("  pcorr: system/fvSolution asks for %s; brae runs its AMG-preconditioned PCG instead -- "
                    "same operator, different Krylov method and iteration count\n", asked.c_str());
    }
    return true;
}

void DevicePcorrSolver::prepare(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    const bool same = nC == nCells
                   && static_cast<std::size_t>(nIf) == owner.size()
                   && std::equal(owner.begin(), owner.end(), m.owner().begin())
                   && std::equal(neighbour.begin(), neighbour.end(), m.neighbour().begin());
    std::optional<interPhase::Nested> preparePart;
    if (!same)
    {
        preparePart.emplace("pcorr prepare: the gather addressing built and up");
        // the gather addressing buildDeviceMesh derives, over the internal faces
        owner.assign(m.owner().begin(), m.owner().begin() + nIf);
        neighbour.assign(m.neighbour().begin(), m.neighbour().begin() + nIf);
        nCells = nC;
        std::vector<label> ownerStart(static_cast<std::size_t>(nC) + 1, 0);
        std::vector<label> losortStart(static_cast<std::size_t>(nC) + 1, 0);
        for (label f = 0; f < nIf; ++f)
        {
            ++ownerStart[static_cast<std::size_t>(owner[static_cast<std::size_t>(f)]) + 1];
            ++losortStart[static_cast<std::size_t>(neighbour[static_cast<std::size_t>(f)]) + 1];
        }
        for (label c = 0; c < nC; ++c)
        {
            ownerStart[static_cast<std::size_t>(c) + 1] += ownerStart[static_cast<std::size_t>(c)];
            losortStart[static_cast<std::size_t>(c) + 1] += losortStart[static_cast<std::size_t>(c)];
        }
        std::vector<label> losort(static_cast<std::size_t>(nIf));
        std::vector<label> next(losortStart.begin(), losortStart.end() - 1);
        for (label f = 0; f < nIf; ++f)
        {
            const std::size_t n = static_cast<std::size_t>(neighbour[static_cast<std::size_t>(f)]);
            losort[static_cast<std::size_t>(next[n]++)] = f;
        }
        dOwner.copyFrom(owner);
        dNeighbour.copyFrom(neighbour);
        dOwnerStart.copyFrom(ownerStart);
        dLosort.copyFrom(losort);
        dLosortStart.copyFrom(losortStart);
        preparePart.emplace("pcorr prepare: the hierarchy");
        // the hierarchy the p_rgh solve takes (deviceAmgPcgHierarchy), from the disk cache on the start mesh
        const bool firstBuild = addressingId == 0;
        addressingId = nextDeviceAddressingId();
        // which hierarchy: see fixedTopology in the header
        static const int mode = []()
        {
            const char* e = std::getenv("BRAE_PCORR_AMG");
            const std::string v = e ? e : "";
            if (v.empty())
            {
                return 0;
            }
            if (v == "plain")
            {
                return 1;
            }
            if (v == "sa")
            {
                return 2;
            }
            throw std::runtime_error("brae interFoam device pcorr solve: BRAE_PCORR_AMG=" + v
                                     + " is not one of plain, sa.");
        }();
        // ...and twoDimensional: a patch of type empty that has faces
        bool twoDimensional = false;
        // ...and no coupled pair: the hierarchy carries a pair on its grids through its aggregate maps
        // (AMGPair), which a smoothed prolongator is not
        bool coupledPair = false;
        for (const FvPatch& q : patches)
        {
            if (q.type == "empty" && q.size > 0) twoDimensional = true;
            if (q.coupled) coupledPair = true;
        }
        if (coupledPair && mode == 2)
        {
            throw std::runtime_error(
                "brae interFoam device pcorr solve: BRAE_PCORR_AMG=sa on a mesh with a coupled pair. A smoothed-"
                "aggregation hierarchy does not carry the pair on its grids; unset it.");
        }
        if (mode == 2 || (mode == 0 && fixedTopology && firstBuild && twoDimensional && !coupledPair))
        {
            static bool said = false;
            if (!said)
            {
                said = true;
                std::printf("  pcorr: its AMG hierarchy is a smoothed-aggregation one of its own, built once "
                            "(the mesh keeps its topology and is two-dimensional); BRAE_PCORR_AMG=plain takes "
                            "the one p_rgh uses\n");
            }
            // from the case's cache where it is this mesh's (the start mesh only), built otherwise
            interPhase::Nested timedBuild("pcorr: the smoothed-aggregation hierarchy (load or build)");
            amg = deviceAmgPcgHierarchy(m, g, caseDir, firstBuild, nullptr, /*smoothed=*/true);
        }
        else
        {
            amg = deviceAmgPcgHierarchy(m, g, caseDir, firstBuild, amgMemo);
        }
    }
    preparePart.emplace("pcorr prepare: the boundary slots");
    // correct()'s boundary: a slot a face of the patches that are neither empty nor coupled, in patch order, and
    // each cell's slots in that order -- the order the host adds a cell's boundary terms in
    std::size_t nNow = 0;
    for (const FvPatch& q : patches)
    {
        if (q.coupled || q.type == "empty") continue;
        nNow += static_cast<std::size_t>(q.size);
    }
    if (!same || nNow != nSlots)
    {
        nSlots = nNow;
        std::vector<label> slotCell;
        slotCell.reserve(nSlots);
        std::vector<label> start(static_cast<std::size_t>(nC) + 1, 0);
        for (const FvPatch& q : patches)
        {
            if (q.coupled || q.type == "empty") continue;
            for (label i = 0; i < q.size; ++i)
            {
                slotCell.push_back(q.faceCells[i]);
                ++start[static_cast<std::size_t>(q.faceCells[i]) + 1];
            }
        }
        for (label c = 0; c < nC; ++c)
        {
            start[static_cast<std::size_t>(c) + 1] += start[static_cast<std::size_t>(c)];
        }
        std::vector<label> list(nSlots);
        std::vector<label> at(start.begin(), start.end() - 1);
        for (std::size_t k = 0; k < nSlots; ++k)
        {
            const std::size_t c = static_cast<std::size_t>(slotCell[k]);
            list[static_cast<std::size_t>(at[c]++)] = static_cast<label>(k);
        }
        dSlotStart.copyFrom(start);
        dSlotList.copyFrom(list);
    }
}

void DevicePcorrSolver::pairUp(
    const FvScalarMatrix& M,
    const std::vector<FvPatch>& patches)
{
    // A face a row: its own cell, the slots patchNeighbourValue sums (a cyclic's one neighbour cell at weight
    // 1, a cyclicAMI's weighted cells), and the coefficient. fvm::laplacian stores a coupled face's interface
    // coefficient in boundaryCoeffs with lduMatrix's sign -- Amul takes result -= boundaryCoeffs*psi_nbr -- and
    // deviceAmul adds, so it goes up negated.
    std::vector<label> own;
    std::vector<label> off{label(0)};
    std::vector<label> nbr;
    std::vector<scalar> w;
    std::vector<scalar> ifc;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        if (!q.coupled) continue;
        for (label i = 0; i < q.size; ++i)
        {
            const std::size_t k = static_cast<std::size_t>(i);
            own.push_back(q.faceCells[i]);
            ifc.push_back(-M.boundaryCoeffs[pi][k]);
            if (q.amiOffsets.empty())
            {
                nbr.push_back(q.nbrFaceCells[k]);
                w.push_back(scalar(1));
            }
            else
            {
                for (label sl = q.amiOffsets[k]; sl < q.amiOffsets[k + 1]; ++sl)
                {
                    nbr.push_back(q.amiNbrCells[static_cast<std::size_t>(sl)]);
                    w.push_back(q.amiWeights[static_cast<std::size_t>(sl)]);
                }
            }
            off.push_back(static_cast<label>(nbr.size()));
        }
    }
    nPair = static_cast<int>(own.size());
    if (nPair == 0) return;
    dPairOwn.copyFrom(own);
    // a cell's faces of the pair, added in face order by the products (DeviceLduView::pairRank)
    dPairRank.copyFrom(pairOwnerRanks(own, nPairRanks));
    dPairOff.copyFrom(off);
    dPairNbr.copyFrom(nbr);
    dPairW.copyFrom(w);
    dPairIfc.copyFrom(ifc);
}

bool DevicePcorrSolver::solve(
    const std::string& asked,
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    SolverPerformance& perf)
{
    if (!takes(asked, patches)) return false;
    if (M.upper != M.lower)
    {
        throw std::runtime_error(
            "brae interFoam device pcorr solve: the pcorr matrix is asymmetric, and a conjugate gradient needs "
            "a symmetric operator. fvm::laplacian(rAUf, pcorr) is symmetric; reaching here is a defect.");
    }
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    std::optional<interPhase::Nested> timedPart;
    // by parts: this was one row, "the addressing check, the fold and the uploads", 11.0 ms a step on
    // RAS/motorBike at 2.1 calls a step (2026-10-05)
    timedPart.emplace("pcorr solve: the addressing check and the hierarchy (prepare)");
    prepare(m, g, patches);
    timedPart.emplace("pcorr solve: the boundary folded into the diagonal and the source (host)");
    // fvMatrix::solveSegregated: addBoundaryDiag, addBoundarySource (no coupled patch reaches here)
    std::vector<scalar> diag = M.diag;
    std::vector<scalar> source = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const std::size_t c = static_cast<std::size_t>(patches[pi].faceCells[i]);
            diag[c] += M.internalCoeffs[pi][static_cast<std::size_t>(i)];
            // a coupled patch's boundaryCoeffs are the INTERFACE's coefficients, never a source
            if (patches[pi].coupled) continue;
            source[c] += M.boundaryCoeffs[pi][static_cast<std::size_t>(i)];
        }
    }
    timedPart.emplace("pcorr solve: the matrix, the source and pcorr up");
    dDiag.copyFrom(diag);
    dUpper.copyFrom(M.upper);
    dSource.copyFrom(source);
    dPsi.copyFrom(psi);
    pairUp(M, patches);
    DeviceLduView A{nC, nIf, dDiag.data(), dUpper.data(), dUpper.data(), dOwner.data(), dNeighbour.data(),
                    dOwnerStart.data(), dLosort.data(), dLosortStart.data(), 0, nullptr, nullptr, nullptr};
    A.addressingId = addressingId;
    if (nPair > 0)
    {
        A.nAmi = nPair;
        A.amiOwn = dPairOwn.data();
        A.amiOff = dPairOff.data();
        A.amiNbr = dPairNbr.data();
        A.amiW = dPairW.data();
        A.amiIfc = dPairIfc.data();
        A.pairRank = nPairRanks > 1 ? dPairRank.data() : nullptr;
        A.nPairRanks = nPairRanks;
    }
    timedPart.emplace("pcorr: the hierarchy's coarse matrices (Galerkin) and the norm factor");
    amgGalerkin(amg, dDiag, dUpper, dUpper);
    const scalar nf = deviceNormFactor(A, dPsi, dSource, deviceOnes(nC));
    // BRAE_CONTROL_AMG_PCG_ONE_ITERATION=1, the AMG-PCG gates' control, as the p_rgh solve takes it; and
    // BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION=1, which stops pcorr's alone -- the pcorr gate's control
    static const bool oneIteration = std::getenv("BRAE_CONTROL_AMG_PCG_ONE_ITERATION") != nullptr
                                  || std::getenv("BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION") != nullptr;
    timedPart.emplace("pcorr: the AMG-PCG iterations");
    // the solve BRAE_AMG_PCG_SPLIT=pcorr names
    const amgSplit::Name splitName("pcorr");
    // the fast path's knobs, the ones the p_rgh solve takes (amgPcgKnobs). By default deviceAMGPCG runs the
    // whole PCG loop from its captured graph whatever they say (BRAE_PCG_DEVICE); they matter when that is off
    // or coarse-correction scaling is asked for, and then both pressure entries now take the same ones.
    const AmgPcgKnobs& knobs = amgPcgKnobs();
    const DeviceSolverPerf r = deviceAMGPCG(A, amg, dSource, dPsi, nf, tol, relTol, oneIteration ? 1 : maxIter,
                                            knobs.graph, knobs.checkEvery, knobs.corrScaling, minIter);
    timedPart.emplace("pcorr: the download");
    dPsi.copyTo(psi);
    timedPart.reset();
    perf.initialResidual = r.initialResidual;
    perf.finalResidual = r.finalResidual;
    perf.nIterations = r.nIterations;
    return true;
}

namespace {

// fvm::laplacian's face coefficient, dc*gamma*magSf: the two products the host forms, in its order
__global__
void pcorrUpperKernel(
    int nIf,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ gamma,
    const scalar* __restrict__ magSf,
    scalar* __restrict__ upper)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    upper[f] = __dmul_rn(__dmul_rn(dc[f], gamma[f]), magSf[f]);
}

// A CELL'S ROW AND SOURCE, as the host's loops leave them. The host walks the FACES in ascending number and
// writes to both cells of each -- diag[own] -= coeff, diag[nei] -= coeff (fvm::laplacian); d[own] += phi,
// d[nei] -= phi (fvc::div) -- so a cell's terms arrive in ascending face number, whichever side it is on:
// here the faces it is the neighbour of (losort) and the ones it owns are merged in that order. Then, in the
// host's order: the boundary fluxes (patch order), the division by V, source = d*V added to a zero source
// (fused, pe.source[k] += divPhi[k]*V[k]), setReference(0, 0) on the reference cell, and the boundary's
// coefficients folded in (fvMatrix::solveSegregated).
// `ownerFirst` non-zero is a gate's CONTROL: a cell's owned faces before the ones it neighbours.
__global__
void pcorrCellKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ V,
    const label* __restrict__ slotStart,
    const label* __restrict__ slotList,
    const scalar* __restrict__ slotPhi,
    const scalar* __restrict__ slotIC,
    const scalar* __restrict__ slotBC,
    int referenceCell,
    int ownerFirst,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar dg = 0.0;
    scalar d = 0.0;
    label i = losortStart[c];
    const label ie = losortStart[c + 1];
    label j = ownerStart[c];
    const label je = ownerStart[c + 1];
    while (i < ie || j < je)
    {
        bool takeNeighboured = (j >= je) || (i < ie && losort[i] < j);
        if (ownerFirst) takeNeighboured = (j >= je);
        if (takeNeighboured)
        {
            const label f = losort[i++];
            dg = dg - upper[f];
            d = d - phi[f];
        }
        else
        {
            const label f = j++;
            dg = dg - upper[f];
            d = d + phi[f];
        }
    }
    const label ks = slotStart[c];
    const label ke = slotStart[c + 1];
    for (label k = ks; k < ke; ++k)
    {
        d = d + slotPhi[slotList[k]];
    }
    d = d / V[c];
    scalar src = fma(d, V[c], 0.0);
    if (c == referenceCell)
    {
        src = fma(dg, 0.0, src);
        dg = dg + dg;
    }
    for (label k = ks; k < ke; ++k)
    {
        dg = dg + slotIC[slotList[k]];
        src = src + slotBC[slotList[k]];
    }
    diag[c] = dg;
    source[c] = src;
}

// phi -= pcorrEqn.flux() on the internal faces: upper*psi[nei] - lower*psi[own] as the host forms it (the first
// product fused into the subtraction, the second rounded -- fnmsub in matrixFlux), then the subtraction
__global__
void pcorrFluxKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ upper,
    const scalar* __restrict__ psi,
    scalar* __restrict__ phi)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar rounded = __dmul_rn(upper[f], psi[own[f]]);
    const scalar flux = fma(upper[f], psi[nei[f]], -rounded);
    phi[f] = phi[f] - flux;
}

// the first entry of `have` that is not `want`'s VALUE (a zero of either sign is a zero), or -1
long firstUnequal(
    const std::vector<scalar>& have,
    const std::vector<scalar>& want)
{
    if (have.size() != want.size()) return 0;
    for (std::size_t i = 0; i < want.size(); ++i)
    {
        if (!(have[i] == want[i])) return static_cast<long>(i);
    }
    return -1;
}

void refuseUnequal(
    const char* what,
    const std::vector<scalar>& have,
    const std::vector<scalar>& want)
{
    const long at = firstUnequal(have, want);
    if (at < 0) return;
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "brae interFoam device pcorr: BRAE_CONTROL_PCORR_ASSEMBLY_CHECK: %s, entry %ld of %zu: %.17g on "
                  "the GPU, %.17g as the host forms it.", what, at, want.size(),
                  static_cast<std::size_t>(at) < have.size() ? have[static_cast<std::size_t>(at)] : 0.0,
                  static_cast<std::size_t>(at) < want.size() ? want[static_cast<std::size_t>(at)] : 0.0);
    throw std::runtime_error(buf);
}

}   // namespace

bool DevicePcorrSolver::correct(
    const std::string& asked,
    const SurfaceScalarField& rAUf,
    SurfaceScalarField& phi,
    const GeometricField<scalar>& pcorr,
    bool needReference,
    bool nonOrthDeltaCoeffs,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const DeviceFvGeometry* deviceGeometry,
    scalar tol,
    scalar relTol,
    int maxIter,
    int minIter,
    SolverPerformance& perf,
    const FvScalarMatrix* hostMatrix)
{
    // a coupled pair: the host assembles, and solve() takes the system with the pair on its view
    for (const FvPatch& q : patches)
    {
        if (q.coupled) return false;
    }
    if (!takes(asked, patches)) return false;
    static const bool ownerFirst = std::getenv("BRAE_CONTROL_PCORR_ASSEMBLY_OWNER_FIRST") != nullptr;
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    std::optional<interPhase::Nested> timedPart;
    timedPart.emplace("pcorr: the addressing check and the hierarchy");
    prepare(m, g, patches);

    timedPart.emplace("pcorr: rAUf, phi and the patches' terms up");
    if (rAUf.internal.size() != static_cast<std::size_t>(nIf) || phi.internal.size() != static_cast<std::size_t>(nIf))
    {
        throw std::runtime_error("brae interFoam device pcorr: rAUf or phi is not the mesh's internal faces'.");
    }
    dGamma.copyFrom(rAUf.internal);
    dPhi.copyFrom(phi.internal);
    // the geometry: the GPU's own while it is the host's (after a move), the host's uploaded otherwise
    const std::vector<scalar>& dcHost = nonOrthDeltaCoeffs ? g.nonOrthDeltaCoeffs() : g.deltaCoeffs();
    const bool onDevice = deviceGeometry && deviceGeometry->built
                       && deviceGeometry->hostGeneration == g.generation()
                       && deviceGeometry->nC == nC && deviceGeometry->nIf == nIf;
    const scalar* dcD = nullptr;
    const scalar* magSfD = nullptr;
    const scalar* VD = nullptr;
    if (onDevice)
    {
        dcD = nonOrthDeltaCoeffs ? deviceGeometry->nonOrthDeltaCoeffs.data() : deviceGeometry->deltaCoeffs.data();
        magSfD = deviceGeometry->magSf.data();
        VD = deviceGeometry->V.data();
    }
    else
    {
        // before the first step (the start's CorrectPhi) and wherever the mesh changed on the host alone
        interPhase::Nested timedUp("pcorr: the geometry up (the GPU's is not this mesh's)");
        dDc.resize(static_cast<std::size_t>(nIf));
        dMagSf.resize(static_cast<std::size_t>(nIf));
        cudaCheck(cudaMemcpy(dDc.data(), dcHost.data(), static_cast<std::size_t>(nIf)*sizeof(scalar),
                             cudaMemcpyHostToDevice), "pcorr deltaCoeffs");
        cudaCheck(cudaMemcpy(dMagSf.data(), g.magSf().data(), static_cast<std::size_t>(nIf)*sizeof(scalar),
                             cudaMemcpyHostToDevice), "pcorr magSf");
        dV.copyFrom(g.V());
        dcD = dDc.data();
        magSfD = dMagSf.data();
        VD = dV.data();
    }
    // the patches' terms, a slot a face: the flux for div(phi), and fvm::laplacian's two coefficients
    // (laplacianBoundaryCoeffs' uncoupled branch)
    std::vector<scalar> slotPhi(nSlots);
    std::vector<scalar> slotIC(nSlots);
    std::vector<scalar> slotBC(nSlots);
    {
        std::size_t k = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            if (q.coupled || q.type == "empty") continue;
            const std::vector<scalar> gIC = pcorr.boundary[pi]->gradientInternalCoeffs();
            const std::vector<scalar> gBC = pcorr.boundary[pi]->gradientBoundaryCoeffs();
            for (label i = 0; i < q.size; ++i)
            {
                const std::size_t ii = static_cast<std::size_t>(i);
                const scalar pGamma = rAUf.boundary[pi][ii] * g.magSf()[static_cast<std::size_t>(q.start + i)];
                slotPhi[k] = phi.boundary[pi][ii];
                slotIC[k] = pGamma * gIC[ii];
                slotBC[k] = (-pGamma) * gBC[ii];
                ++k;
            }
        }
    }
    dSlotPhi.copyFrom(slotPhi);
    dSlotIC.copyFrom(slotIC);
    dSlotBC.copyFrom(slotBC);

    timedPart.emplace("pcorr: the matrix, the source and the fold (device)");
    dUpper.resize(static_cast<std::size_t>(nIf));
    dDiag.resize(static_cast<std::size_t>(nC));
    dSource.resize(static_cast<std::size_t>(nC));
    dPsi.resize(static_cast<std::size_t>(nC));
    const int tpb = 256;
    if (nIf > 0)
    {
        pcorrUpperKernel<<<(nIf + tpb - 1)/tpb, tpb, 0, cudaStreamPerThread>>>(
            nIf,
            dcD,
            dGamma.data(),
            magSfD,
            dUpper.data());
        cudaCheck(cudaGetLastError(), "pcorrUpperKernel");
    }
    pcorrCellKernel<<<(nC + tpb - 1)/tpb, tpb, 0, cudaStreamPerThread>>>(
        nC,
        dOwnerStart.data(),
        dLosort.data(),
        dLosortStart.data(),
        dUpper.data(),
        dPhi.data(),
        VD,
        dSlotStart.data(),
        dSlotList.data(),
        dSlotPhi.data(),
        dSlotIC.data(),
        dSlotBC.data(),
        needReference ? 0 : -1,
        ownerFirst ? 1 : 0,
        dDiag.data(),
        dSource.data());
    cudaCheck(cudaGetLastError(), "pcorrCellKernel");
    cudaCheck(cudaMemsetAsync(dPsi.data(), 0, static_cast<std::size_t>(nC)*sizeof(scalar), cudaStreamPerThread),
              "pcorr zero start");
    cudaCheck(cudaStreamSynchronize(cudaStreamPerThread), "pcorr assembly");
    if (hostMatrix)
    {
        // the host's system, folded as solve() folds it, against the GPU's
        std::vector<scalar> diag = hostMatrix->diag;
        std::vector<scalar> source = hostMatrix->source;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const std::size_t c = static_cast<std::size_t>(patches[pi].faceCells[i]);
                diag[c] += hostMatrix->internalCoeffs[pi][static_cast<std::size_t>(i)];
                source[c] += hostMatrix->boundaryCoeffs[pi][static_cast<std::size_t>(i)];
            }
        }
        refuseUnequal("the off-diagonal", dUpper.host(), hostMatrix->upper);
        refuseUnequal("the folded diagonal", dDiag.host(), diag);
        refuseUnequal("the folded source", dSource.host(), source);
    }
    DeviceLduView A{nC, nIf, dDiag.data(), dUpper.data(), dUpper.data(), dOwner.data(), dNeighbour.data(),
                    dOwnerStart.data(), dLosort.data(), dLosortStart.data(), 0, nullptr, nullptr, nullptr};
    A.addressingId = addressingId;
    timedPart.emplace("pcorr: the hierarchy's coarse matrices (Galerkin) and the norm factor");
    amgGalerkin(amg, dDiag, dUpper, dUpper);
    const scalar nf = deviceNormFactor(A, dPsi, dSource, deviceOnes(nC));
    static const bool oneIteration = std::getenv("BRAE_CONTROL_AMG_PCG_ONE_ITERATION") != nullptr
                                  || std::getenv("BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION") != nullptr;
    timedPart.emplace("pcorr: the AMG-PCG iterations");
    // the solve BRAE_AMG_PCG_SPLIT=pcorr names
    const amgSplit::Name splitName("pcorr");
    const AmgPcgKnobs& knobs = amgPcgKnobs();
    const DeviceSolverPerf r = deviceAMGPCG(A, amg, dSource, dPsi, nf, tol, relTol, oneIteration ? 1 : maxIter,
                                            knobs.graph, knobs.checkEvery, knobs.corrScaling, minIter);

    timedPart.emplace("pcorr: the flux (device), phi and pcorr down");
    SurfaceScalarField phiBefore;
    if (hostMatrix) phiBefore = phi;
    if (nIf > 0)
    {
        pcorrFluxKernel<<<(nIf + tpb - 1)/tpb, tpb, 0, cudaStreamPerThread>>>(
            nIf,
            dOwner.data(),
            dNeighbour.data(),
            dUpper.data(),
            dPsi.data(),
            dPhi.data());
        cudaCheck(cudaGetLastError(), "pcorrFluxKernel");
    }
    dPhi.copyTo(phi.internal);
    std::vector<scalar> psi;
    dPsi.copyTo(psi);
    // the patches' flux, on the host: internalCoeffs*psi - boundaryCoeffs, the product fused into the
    // subtraction as matrixFlux forms it, and phi -= flux on every patch that is not empty
    {
        std::size_t k = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            if (q.coupled || q.type == "empty") continue;
            for (label i = 0; i < q.size; ++i)
            {
                const scalar flux = std::fma(slotIC[k], psi[static_cast<std::size_t>(q.faceCells[i])], -slotBC[k]);
                phi.boundary[pi][static_cast<std::size_t>(i)] -= flux;
                ++k;
            }
        }
    }
    if (hostMatrix)
    {
        // the host's own flux (matrixFlux) of the GPU's solution, taken from phi as it came in: what correctPhi's
        // last lines do, on every face
        const SurfaceScalarField flux = matrixFlux(*hostMatrix, psi, m, patches);
        SurfaceScalarField want = phiBefore;
        for (std::size_t f = 0; f < want.internal.size(); ++f)
        {
            want.internal[f] -= flux.internal[f];
        }
        refuseUnequal("phi on the internal faces", phi.internal, want.internal);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].type == "empty") continue;
            for (std::size_t i = 0; i < want.boundary[pi].size(); ++i)
            {
                want.boundary[pi][i] -= flux.boundary[pi][i];
            }
            refuseUnequal(("phi on the patch " + patches[pi].name).c_str(), phi.boundary[pi], want.boundary[pi]);
        }
    }
    timedPart.reset();
    perf.initialResidual = r.initialResidual;
    perf.finalResidual = r.finalResidual;
    perf.nIterations = r.nIterations;
    return true;
}

} // namespace brae
