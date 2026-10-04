#include "device_displacement_laplacian_assembly.cuh"
#include "inter_phase_time.cuh"
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace brae {

namespace {

const char* const WHO = "brae deviceDisplacementAssembly: ";

constexpr int dlaTPB = 256;

int dlaBlocks(label n)
{
    return static_cast<int>((n + dlaTPB - 1)/dlaTPB);
}

// a*b + c in ONE rounding, as the host's fmadd, fmla and (with a negated) fmls; the control rounds the product
// first
__device__
inline scalar dlaMulAdd(
    scalar a,
    scalar b,
    scalar c,
    bool unfused)
{
    if (unfused)
    {
        return __dadd_rn(__dmul_rn(a, b), c);
    }
    return fma(a, b, c);
}

// fvm::laplacian's face coefficient, dc*gamma*magSf in that order, with nonOrthDeltaCoeffs as `corrected` takes
__global__
void dlaCoeffKernel(
    label nIf,
    const scalar* dc,
    const scalar* gamma,
    const scalar* magSf,
    scalar* upper)
{
    const label f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf)
    {
        return;
    }
    upper[f] = dc[f]*gamma[f]*magSf[f];
}

// fvc::gaussGrad of the vector field, a cell at a time: its internal faces in ascending number (the faces it
// owns are [ownerStart[c], ownerStart[c + 1]); the ones it neighbours are losort's, ascending), then its
// boundary faces in patch order, an empty patch's left out as the host leaves them, then the division by V
__global__
void dlaGradKernel(
    label nC,
    const label* ownerStart,
    const label* losort,
    const label* losortStart,
    const label* own,
    const label* nei,
    const scalar* w,
    const scalar* Sfx,
    const scalar* Sfy,
    const scalar* Sfz,
    const scalar* D,
    const label* bndCellStart,
    const label* bndPerm,
    const label* bndGFace,
    const label* bndIsEmpty,
    const scalar* Db,
    const scalar* V,
    bool unfused,
    scalar* grad)
{
    const label c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC)
    {
        return;
    }
    scalar a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    label i = ownerStart[c];
    const label iE = ownerStart[c + 1];
    label j = losortStart[c];
    const label jE = losortStart[c + 1];
    while (i < iE || j < jE)
    {
        const label fn = (j < jE) ? losort[j] : INT_MAX;
        const bool owned = (i < iE) && (i < fn);
        const label f = owned ? i : fn;
        const std::size_t o = 3*static_cast<std::size_t>(own[f]);
        const std::size_t n = 3*static_cast<std::size_t>(nei[f]);
        const scalar wf = w[f];
        // Uf = w*(Do - Dn) + Dn
        scalar u[3];
        for (int k = 0; k < 3; ++k)
        {
            u[k] = dlaMulAdd(wf, D[o + k] - D[n + k], D[n + k], unfused);
        }
        // the owner adds Sf (x) Uf and the neighbour subtracts it
        const scalar s[3] =
        {
            owned ? Sfx[f] : -Sfx[f],
            owned ? Sfy[f] : -Sfy[f],
            owned ? Sfz[f] : -Sfz[f]
        };
        for (int r = 0; r < 3; ++r)
        {
            for (int k = 0; k < 3; ++k)
            {
                a[3*r + k] = dlaMulAdd(s[r], u[k], a[3*r + k], unfused);
            }
        }
        if (owned)
        {
            ++i;
        }
        else
        {
            ++j;
        }
    }
    for (label q = bndCellStart[c]; q < bndCellStart[c + 1]; ++q)
    {
        const label b = bndPerm[q];
        if (bndIsEmpty[b]) continue;
        const label gf = bndGFace[b];
        const scalar s[3] = {Sfx[gf], Sfy[gf], Sfz[gf]};
        const scalar* u = Db + 3*static_cast<std::size_t>(b);
        for (int r = 0; r < 3; ++r)
        {
            for (int k = 0; k < 3; ++k)
            {
                a[3*r + k] = dlaMulAdd(s[r], u[k], a[3*r + k], unfused);
            }
        }
    }
    const scalar v = V[c];
    scalar* out = grad + 9*static_cast<std::size_t>(c);
    for (int t = 0; t < 9; ++t)
    {
        out[t] = a[t]/v;
    }
}

// laplacianCorrFlux on an internal face: gamma*magSf*(corrVec & interpolate(grad)), the contraction over the
// gradient's FIRST index (dotCorr)
__global__
void dlaCorrFluxKernel(
    label nIf,
    const label* own,
    const label* nei,
    const scalar* w,
    const scalar* cvx,
    const scalar* cvy,
    const scalar* cvz,
    const scalar* gamma,
    const scalar* magSf,
    const scalar* grad,
    scalar* ffc)
{
    const label f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf)
    {
        return;
    }
    const scalar wf = w[f];
    const scalar w1 = 1.0 - wf;
    const scalar* gO = grad + 9*static_cast<std::size_t>(own[f]);
    const scalar* gN = grad + 9*static_cast<std::size_t>(nei[f]);
    scalar gf[9];
    for (int t = 0; t < 9; ++t)
    {
        gf[t] = fma(wf, gO[t], w1*gN[t]);
    }
    const scalar cx = cvx[f];
    const scalar cy = cvy[f];
    const scalar cz = cvz[f];
    const scalar gm = gamma[f]*magSf[f];
    scalar* out = ffc + 3*static_cast<std::size_t>(f);
    for (int k = 0; k < 3; ++k)
    {
        const scalar corr = fma(cz, gf[6 + k], fma(cx, gf[k], cy*gf[3 + k]));
        out[k] = gm*corr;
    }
}

// the two per-cell sums the host's face loops leave: the diagonal, minus every face coefficient, and
// laplacianNonOrthSource's, plus the flux of an owned face and minus a neighboured one's. Both read their
// addends from memory, so nothing here is a product the compiler could fuse.
__global__
void dlaFoldKernel(
    label nC,
    const label* ownerStart,
    const label* losort,
    const label* losortStart,
    const scalar* upper,
    const scalar* ffc,
    scalar* diag,
    scalar* corr)
{
    const label c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC)
    {
        return;
    }
    scalar d = 0;
    scalar s[3] = {0, 0, 0};
    label i = ownerStart[c];
    const label iE = ownerStart[c + 1];
    label j = losortStart[c];
    const label jE = losortStart[c + 1];
    while (i < iE || j < jE)
    {
        const label fn = (j < jE) ? losort[j] : INT_MAX;
        const bool owned = (i < iE) && (i < fn);
        const label f = owned ? i : fn;
        const scalar* flux = ffc + 3*static_cast<std::size_t>(f);
        d -= upper[f];
        if (owned)
        {
            s[0] += flux[0];
            s[1] += flux[1];
            s[2] += flux[2];
            ++i;
        }
        else
        {
            s[0] -= flux[0];
            s[1] -= flux[1];
            s[2] -= flux[2];
            ++j;
        }
    }
    diag[c] = d;
    scalar* out = corr + 3*static_cast<std::size_t>(c);
    out[0] = s[0];
    out[1] = s[1];
    out[2] = s[2];
}

} // namespace


void deviceDisplacementAssembly(
    const DeviceMesh& dm,
    const std::vector<FvPatch>& patches,
    const std::vector<scalar>& gamma,
    const std::vector<vector>& D,
    const std::vector<std::vector<vector>>& boundary,
    DeviceDisplacementAssembly& w,
    std::vector<scalar>& upper,
    std::vector<scalar>& diag,
    std::vector<vector>& corr)
{
    static_assert(sizeof(vector) == 3*sizeof(scalar), "a vector is read here as three scalars");
    const label nC = dm.nCells;
    const label nIf = dm.nInternalFaces;
    const label nB = dm.nBndFaces;
    if (static_cast<label>(D.size()) != nC || static_cast<label>(gamma.size()) != nIf
     || boundary.size() != patches.size())
    {
        throw std::runtime_error(std::string(WHO) + "the fields handed in are not on the device mesh.");
    }
    interPhase::Nested timed("motion: the equation's interior (device)");
    static const bool unfused = std::getenv("BRAE_CONTROL_MOTION_ASSEMBLY_UNFUSED") != nullptr;
    const std::size_t sC = static_cast<std::size_t>(nC);
    const std::size_t sIf = static_cast<std::size_t>(nIf);
    w.gamma.resize(sIf);
    w.D.resize(3*sC);
    w.Db.resize(3*static_cast<std::size_t>(nB));
    w.upper.resize(sIf);
    w.grad.resize(9*sC);
    w.ffc.resize(3*sIf);
    w.corr.resize(3*sC);
    w.diag.resize(sC);
    if (nIf > 0)
    {
        cudaCheck(cudaMemcpy(w.gamma.data(), gamma.data(), sIf*sizeof(scalar), cudaMemcpyHostToDevice), WHO);
    }
    cudaCheck(cudaMemcpy(w.D.data(), D.data(), 3*sC*sizeof(scalar), cudaMemcpyHostToDevice), WHO);
    // the boundary values into the device mesh's boundary order: the patches in turn. An empty patch's faces are
    // in that order too and are never read, so nothing is sent for them -- on a 2-D mesh they are two a cell.
    label at = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        if (p.coupled || isCoupledInterfaceType(p.type))
        {
            throw std::runtime_error(
                std::string(WHO) + "patch " + p.name + " is coupled (" + p.type + "); the correction's coupled "
                "faces are not assembled here.");
        }
        if (p.type != "empty" && p.size > 0)
        {
            if (boundary[pi].size() != static_cast<std::size_t>(p.size))
            {
                throw std::runtime_error(std::string(WHO) + "patch " + p.name + " was handed no boundary values.");
            }
            cudaCheck(cudaMemcpy(w.Db.data() + 3*static_cast<std::size_t>(at), boundary[pi].data(),
                                 3*static_cast<std::size_t>(p.size)*sizeof(scalar), cudaMemcpyHostToDevice), WHO);
        }
        at += p.size;
    }
    if (at != nB)
    {
        throw std::runtime_error(std::string(WHO) + "the patches do not add up to the device mesh's boundary.");
    }
    if (nIf > 0)
    {
        dlaCoeffKernel<<<dlaBlocks(nIf), dlaTPB, 0, cudaStreamPerThread>>>(
            nIf,
            dm.nonOrthDc.data(),
            w.gamma.data(),
            dm.magSf.data(),
            w.upper.data());
    }
    dlaGradKernel<<<dlaBlocks(nC), dlaTPB, 0, cudaStreamPerThread>>>(
        nC,
        dm.ownerStart.data(),
        dm.losort.data(),
        dm.losortStart.data(),
        dm.owner.data(),
        dm.nei.data(),
        dm.w.data(),
        dm.Sfx.data(),
        dm.Sfy.data(),
        dm.Sfz.data(),
        w.D.data(),
        dm.bndCellStart.data(),
        dm.bndPerm.data(),
        dm.bndGFace.data(),
        dm.bndIsEmpty.data(),
        w.Db.data(),
        dm.V.data(),
        unfused,
        w.grad.data());
    if (nIf > 0)
    {
        dlaCorrFluxKernel<<<dlaBlocks(nIf), dlaTPB, 0, cudaStreamPerThread>>>(
            nIf,
            dm.owner.data(),
            dm.nei.data(),
            dm.w.data(),
            dm.corrVecX.data(),
            dm.corrVecY.data(),
            dm.corrVecZ.data(),
            w.gamma.data(),
            dm.magSf.data(),
            w.grad.data(),
            w.ffc.data());
    }
    dlaFoldKernel<<<dlaBlocks(nC), dlaTPB, 0, cudaStreamPerThread>>>(
        nC,
        dm.ownerStart.data(),
        dm.losort.data(),
        dm.losortStart.data(),
        w.upper.data(),
        w.ffc.data(),
        w.diag.data(),
        w.corr.data());
    upper.resize(sIf);
    diag.resize(sC);
    corr.resize(sC);
    if (nIf > 0)
    {
        cudaCheck(cudaMemcpy(upper.data(), w.upper.data(), sIf*sizeof(scalar), cudaMemcpyDeviceToHost), WHO);
    }
    cudaCheck(cudaMemcpy(diag.data(), w.diag.data(), sC*sizeof(scalar), cudaMemcpyDeviceToHost), WHO);
    cudaCheck(cudaMemcpy(corr.data(), w.corr.data(), 3*sC*sizeof(scalar), cudaMemcpyDeviceToHost), WHO);
}

} // namespace brae
