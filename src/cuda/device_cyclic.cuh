#pragma once
// cf GPU offload, cyclic (periodic) coupling as a SEPARATE lduInterface, exactly like OpenFOAM's
// cyclicFvPatchField. The periodic faces are NOT merged into the owner-sorted internal-face LDU (that breaks
// the amulKernel's owner-contiguous scatter); instead this carries the interface coupling and applies it:
//   - matrix:  diag[own] -= ifCoeff           (folded into the assembled diagonal)
//              Apsi[own] += ifCoeff * psi[nbr] (the off-diagonal, applied inside deviceAmul -> updateInterfaceMatrix)
//   - flux/continuity: phi_cyc = interp(HbyA).Sf ; div[own] += phi_cyc ; phi_cyc -= ifCoeff*(p[nbr]-p[own])
//   - gradient: grad[own] += Sf * (w*psi[own] + (1-w)*psi[nbr]) / V[own]
// Both sides of each periodic pair are stored (buildCyclicInterfaces returns both), so the coupling is symmetric.
// Translational only (R = I): scalars and vector components couple with identity transform. Rotational = Phase 1.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "interface/cyclic_interface.cuh"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// THE NEIGHBOUR OF AN INTERFACE FACE, as a kernel reads it. Across a cyclic that is one cell; across a
// cyclicAMI it is the AMI's weighted sum over several (cyclicAMIFvPatchField::patchNeighbourField,
// AMIInterpolation::weightedSum: the result starts at zero and takes += w*value, slot by slot -- the
// order the host's patchNeighbourValue uses, so the two arms add the same numbers in the same order).
// `off` null = one cell per face, cell[j]; else slots off[j] .. off[j+1] of (cell, w).
struct CyclicNbr
{
    const label*  cell = nullptr;
    const label*  off = nullptr;
    const scalar* w = nullptr;
};

#ifdef __CUDACC__
__host__ __device__ __forceinline__ scalar cyclicNbrValue(
    const CyclicNbr& s,
    const scalar* x,
    int j)
{
    if (!s.off)
    {
        return x[s.cell[j]];
    }
    scalar r = 0;
    for (label k = s.off[j]; k < s.off[j + 1]; ++k)
    {
        r += s.w[k]*x[s.cell[k]];
    }
    return r;
}
#endif

struct DeviceCyclic
{
    // the AMI stencil, present when ANY pair of the mesh is a cyclicAMI: then EVERY face has slots (a
    // cyclic's face one slot of weight 1, which adds 0 + 1*x and is the cell's value to the bit), and
    // nbrCell is not to be read -- it holds -1 on the AMI faces.
    bool stencil = false;
    DeviceBuffer<label>  stOff, stCell;
    DeviceBuffer<scalar> stW;
    // 1 on a cyclicAMI's faces, 0 on a cyclic's: what OpenFOAM tests with isA<cyclicAMIFvPatch>, which
    // a cyclicACMI patch is not (deviceCyclicZeroOnAmi)
    DeviceBuffer<label>  stIsAmi;
    CyclicNbr nbr() const
    {
        CyclicNbr s;
        if (stencil)
        {
            s.cell = stCell.data();
            s.off = stOff.data();
            s.w = stW.data();
        }
        else
        {
            s.cell = nbrCell.data();
        }
        return s;
    }
    int n = 0;                                  // total cyclic faces (BOTH sides of every pair)
    DeviceBuffer<label>  ownCell, nbrCell;      // this-side cell, periodic-neighbour cell
    DeviceBuffer<scalar> deltaCoeffs, weights, magSf;   // face geometry (own weight w; |Sf|; 1/|delta|)
    // ...and the ORTHOGONAL delta coefficient, taken from the host patch rather than re-derived.
    // CyclicInterface::deltaCoeffs is OpenFOAM's nonOrthDeltaCoeffs, 1/(nf & delta), which is what a
    // `corrected` laplacian wants; an `uncorrected` one wants the patch's plain 1/|delta|. MEASURED on a
    // skewed periodic pair (validation/cyclicChannelSkew): using the non-orthogonal set for both put the
    // orthogonal laplacian's interface coefficient 4.2e-03 out of 5.5e-02, 7.7% of it.
    DeviceBuffer<scalar> orthDeltaCoeffs;
    // A PER-CELL MAP of this interface's faces, and each face's TWIN on the other side of the pair.
    // The kernels above scatter with atomics, which is fine for a sum; MULES gathers, because its
    // limiter is a per-cell budget and the explicit path is deliberately atomics-free
    // (device_mules.cuh). ifCellStart is a CSR over cells into ifPerm, which holds interface-face
    // indices; twin[j] is the index of the face j pairs with, or -1 where there is none (an AMI side,
    // which OpenFOAM does NOT sync -- see MULES's sync note).
    DeviceBuffer<label> ifCellStart, ifPerm, twin;
    // THE ORDER A CELL'S FACES ARE SUMMED IN. The kernels that add a face's term to its owner cell do it with
    // atomicAdd, and a cell that owns SEVERAL faces of the pair was summed in the order the threads arrived:
    // another rounding at every run. MEASURED 2026-10-07 on RAS/mixerVesselAMI's pinned row, two runs of one
    // binary: 25 of 34 written files differ from the first step (the host loop: none), the first operation
    // that is not the same twice being deviceCyclicAddGrad in the alpha corrector. 11,416 + 9,976 cells of
    // that mesh own two to six faces of the AMI, and for 6,742 of them the faces are in different thread
    // blocks of the launch. ifRank[j] is face j's place among its owner's faces in face order (ifPerm's);
    // such a kernel is launched once a place, each launch taking the faces of that place alone, so no two
    // faces of a launch share a cell and a cell's terms are added in face order -- the host loop's order.
    // One launch, as before, where no cell owns two (every cyclic tutorial).
    //   BRAE_CONTROL_PAIR_SUMS_UNORDERED=1: one launch whatever the mesh, as it was.
    //   BRAE_CONTROL_PAIR_SUMS_REVERSED=1: a gate's CONTROL -- the places taken last first, another fixed
    //   order, which the written files have to show.
    DeviceBuffer<label> ifRank;
    int nIfRanks = 1;
    static bool sumsUnordered()
    {
        static const bool on = std::getenv("BRAE_CONTROL_PAIR_SUMS_UNORDERED") != nullptr;
        return on;
    }
    static bool sumsReversed()
    {
        static const bool on = std::getenv("BRAE_CONTROL_PAIR_SUMS_REVERSED") != nullptr;
        return on;
    }
    // the number of launches such a kernel takes, the rank list to hand it (null: every face), and the
    // place the k-th launch takes
    int ownerPasses() const
    {
        return (nIfRanks <= 1 || sumsUnordered()) ? 1 : nIfRanks;
    }
    const label* ownerRank() const
    {
        return (nIfRanks <= 1 || sumsUnordered()) ? nullptr : ifRank.data();
    }
    int ownerPass(int k) const
    {
        return sumsReversed() ? ownerPasses() - 1 - k : k;
    }
    DeviceBuffer<scalar> Sfx, Sfy, Sfz;         // face area vector, oriented OUT of ownCell
    DeviceBuffer<scalar> dOwnX, dOwnY, dOwnZ;   // Cf - C[own]          (linearUpwind face delta, own side)
    DeviceBuffer<scalar> dNbrX, dNbrY, dNbrZ;   // Cf_nbr - C[nbr]      (linearUpwind face delta, nbr side, UN-rotated)
    DeviceBuffer<scalar> corrVecX, corrVecY, corrVecZ;   // non-orth correction vector (laplacian "corrected")
    DeviceBuffer<scalar> dX, dY, dZ;            // fvPatch::delta() = dOwn - transform(forwardT, dNbr)
    DeviceBuffer<scalar> ifCoeff;               // assembled per-field off-diagonal: gammaFace*dc*magSf (+ convection)
    DeviceBuffer<scalar> phi;                   // assembled cyclic-face flux (for continuity / corrector)
    // ROTATIONAL (Phase 1): a vector neighbour value is rotated by forwardT (nbr->own) before coupling. Scalars are
    // never rotated. fT packed (3*i+j)*n + face = forwardT[i][j] (identity for translational). ifCoeffC[kk] = the
    // per-component implicit off-diagonal (= ifCoeff * forwardT[kk][kk]; OF transformCoupleField diag-only).
    bool rotational = false;
    DeviceBuffer<scalar> fT;                    // 9*n packed: fT[(3*i+j)*n + face]
    DeviceBuffer<scalar> ifCoeffC[3];
    bool empty() const { return n == 0; }
};

// Flatten every cyclic interface (both patches of each pair) into device arrays.
inline DeviceCyclic buildDeviceCyclic(
    const std::vector<CyclicInterface>& cyclics,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    std::vector<label> oc, nc;
    std::vector<scalar> dc, odc, w, ms, sfx, sfy, sfz;
    std::vector<scalar> dox, doy, doz, dnx, dny, dnz, cvx, cvy, cvz, dlx, dly, dlz;
    bool rot = false;
    for (const auto& c : cyclics)
    {
        if (!c.translational) rot = true;
    }
    for (const auto& c : cyclics)
    {
        const FvPatch& P = fvp[c.patch];
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            const label gf = P.start + (label)i;
            oc.push_back(c.faceCells[i]);
            nc.push_back(c.nbrFaceCells[i]);
            dc.push_back(c.deltaCoeffs[i]);
            // the HOST PATCH's own orthogonal coefficient, so the two arms cannot drift apart
            odc.push_back(static_cast<std::size_t>(i) < P.deltaCoeffs.size()
                          ? P.deltaCoeffs[static_cast<std::size_t>(i)] : c.deltaCoeffs[i]);
            w.push_back(c.weights[i]);
            ms.push_back(g.magSf()[gf]);
            sfx.push_back(g.Sf()[gf].x);
            sfy.push_back(g.Sf()[gf].y);
            sfz.push_back(g.Sf()[gf].z);
            dox.push_back(c.dOwn[i].x);
            doy.push_back(c.dOwn[i].y);
            doz.push_back(c.dOwn[i].z);
            dnx.push_back(c.dNbr[i].x);
            dny.push_back(c.dNbr[i].y);
            dnz.push_back(c.dNbr[i].z);
            cvx.push_back(c.corrVec[i].x);
            cvy.push_back(c.corrVec[i].y);
            cvz.push_back(c.corrVec[i].z);
            dlx.push_back(c.delta[i].x);
            dly.push_back(c.delta[i].y);
            dlz.push_back(c.delta[i].z);
        }
    }
    // the CSR over owner cells, and the twin map. Both sides of a pair are stored in patch order, so
    // side A's face i twins with side B's face i; the offsets are what turns that into an index.
    std::vector<label> cellStart, perm, twin(oc.size(), -1);
    {
        std::map<label, std::size_t> offsetOfPatch;
        std::size_t off = 0;
        for (const auto& c : cyclics)
        {
            offsetOfPatch[static_cast<label>(c.patch)] = off;
            off += c.faceCells.size();
        }
        off = 0;
        for (const auto& c : cyclics)
        {
            const auto it = offsetOfPatch.find(static_cast<label>(c.nbrPatch));
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                // ...except across an AMI-family pair, which MULES' limiter sync does not cover
                // (CyclicInterface::ami says why): its faces keep twin = -1 and their own lambda. The
                // host measured what syncing one costs on RAS/damBreakLeakage -- U 1.7e-01, and only
                // on the step the baffle opens.
                if (it != offsetOfPatch.end() && !c.ami)
                {
                    twin[off + i] = static_cast<label>(it->second + i);
                }
            }
            off += c.faceCells.size();
        }
        // THE CSR IS OVER EVERY CELL, not over the cells the pair happens to touch. Sizing it to the
        // highest owner + 2 leaves every kernel that walks it by cell -- MULES's three, and the
        // reconstruction -- reading ifCellStart[c + 1] past the end for every cell above that owner.
        // g.V() is the cell count here, and the pair's owners are a subset of it.
        const label nCellsAll = static_cast<label>(g.V().size());
        label maxCell = -1;
        for (const label cc : oc) maxCell = std::max(maxCell, cc);
        std::vector<label> count(static_cast<std::size_t>(std::max(maxCell + 1, nCellsAll) + 1), 0);
        for (const label cc : oc) ++count[static_cast<std::size_t>(cc) + 1];
        cellStart.assign(count.size(), 0);
        for (std::size_t i = 1; i < count.size(); ++i) cellStart[i] = cellStart[i - 1] + count[i];
        std::vector<label> fill = cellStart;
        perm.assign(oc.size(), 0);
        for (std::size_t j = 0; j < oc.size(); ++j)
        {
            perm[static_cast<std::size_t>(fill[static_cast<std::size_t>(oc[j])]++)] = static_cast<label>(j);
        }
    }
    // each face's place among its owner's faces, in face order (perm's), and the most one cell owns
    std::vector<label> rank(oc.size(), 0);
    label mostOfACell = 1;
    for (std::size_t c = 0; c + 1 < cellStart.size(); ++c)
    {
        for (label k = cellStart[c]; k < cellStart[c + 1]; ++k)
        {
            rank[static_cast<std::size_t>(perm[static_cast<std::size_t>(k)])] = k - cellStart[c];
        }
        mostOfACell = std::max(mostOfACell, cellStart[c + 1] - cellStart[c]);
    }

    DeviceCyclic d;
    d.n = (int)oc.size();
    d.rotational = rot;
    bool anyStencil = false;
    for (const auto& c : cyclics)
    {
        anyStencil = anyStencil || !c.amiOffsets.empty();
    }
    if (anyStencil)
    {
        std::vector<label> so, sc, isAmi;
        std::vector<scalar> sw;
        so.push_back(0);
        for (const auto& c : cyclics)
        {
            for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            {
                if (c.amiOffsets.empty())
                {
                    sc.push_back(c.nbrFaceCells[i]);
                    sw.push_back(scalar(1));
                }
                else
                {
                    for (label k = c.amiOffsets[i]; k < c.amiOffsets[i + 1]; ++k)
                    {
                        sc.push_back(c.amiNbrCells[static_cast<std::size_t>(k)]);
                        sw.push_back(c.amiWeights[static_cast<std::size_t>(k)]);
                    }
                }
                so.push_back(static_cast<label>(sc.size()));
                isAmi.push_back(c.amiOffsets.empty() ? label(0) : label(1));
            }
        }
        d.stIsAmi.copyFrom(isAmi);
        d.stencil = true;
        d.stOff.copyFrom(so);
        d.stCell.copyFrom(sc);
        d.stW.copyFrom(sw);
    }
    d.ownCell.copyFrom(oc);
    d.nbrCell.copyFrom(nc);
    d.deltaCoeffs.copyFrom(dc);
    d.orthDeltaCoeffs.copyFrom(odc);
    d.ifCellStart.copyFrom(cellStart);
    d.ifPerm.copyFrom(perm);
    d.ifRank.copyFrom(rank);
    d.nIfRanks = static_cast<int>(mostOfACell);
    if (mostOfACell > 1)
    {
        // (said once: a moving mesh lays the pair out again at every step)
        static bool said = false;
        if (!said)
        {
            said = true;
            std::size_t several = 0;
            for (std::size_t c = 0; c + 1 < cellStart.size(); ++c)
            {
                several += (cellStart[c + 1] - cellStart[c] > 1) ? 1 : 0;
            }
            std::printf("  coupled pair: %zu cells own several of its faces, up to %d; a cell's faces are summed in "
                        "face order, a launch a place (BRAE_CONTROL_PAIR_SUMS_UNORDERED=1: in the order the "
                        "threads arrive, which another run does not repeat)\n", several,
                        static_cast<int>(mostOfACell));
        }
    }
    d.twin.copyFrom(twin);
    d.weights.copyFrom(w);
    d.magSf.copyFrom(ms);
    d.Sfx.copyFrom(sfx);
    d.Sfy.copyFrom(sfy);
    d.Sfz.copyFrom(sfz);
    d.dOwnX.copyFrom(dox);
    d.dOwnY.copyFrom(doy);
    d.dOwnZ.copyFrom(doz);
    d.dNbrX.copyFrom(dnx);
    d.dNbrY.copyFrom(dny);
    d.dNbrZ.copyFrom(dnz);
    d.corrVecX.copyFrom(cvx);
    d.corrVecY.copyFrom(cvy);
    d.corrVecZ.copyFrom(cvz);
    d.dX.copyFrom(dlx);
    d.dY.copyFrom(dly);
    d.dZ.copyFrom(dlz);
    d.ifCoeff.resize(d.n);
    d.phi.resize(d.n);
    if (rot)   // pack forwardT (9*n, component-major) + the per-component implicit scratch
    {
        std::vector<scalar> ft((std::size_t)9 * d.n);
        std::size_t f = 0;
        for (const auto& c : cyclics)
        {
            const tensor& T = c.forwardT;
            const scalar tc[9] = { T.xx, T.xy, T.xz, T.yx, T.yy, T.yz, T.zx, T.zy, T.zz };
            for (std::size_t i = 0; i < c.faceCells.size(); ++i, ++f)
                for (int k = 0; k < 9; ++k)
                    ft[(std::size_t)k * d.n + f] = tc[k];
        }
        d.fT.copyFrom(ft);
        for (int k = 0; k < 3; ++k)
            d.ifCoeffC[k].resize(d.n);
    }
    return d;
}

// THE WHOLE PAIR, refreshed IN PLACE after the mesh has MOVED and a cyclicAMI's weights have been
// recomputed on the moved points (cyclicAMIPolyPatch::initMovePoints marks the AMI out of date and the
// next AMI() rebuilds it): every geometric array and the stencil, laid out again by buildDeviceCyclic from
// the re-coupled patches. The face count and the owner cells cannot change under a move, so the per-cell
// map, the twin map, phi and ifCoeff stay -- and every DeviceBuffer OBJECT stays where it is, which is
// what the step's hooks hold pointers to.
inline void refreshDeviceCyclicAfterMove(
    DeviceCyclic& d,
    const std::vector<CyclicInterface>& cyclics,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    DeviceCyclic fresh = buildDeviceCyclic(cyclics, g, fvp);
    if (fresh.n != d.n || fresh.stencil != d.stencil || fresh.rotational != d.rotational)
    {
        throw std::runtime_error(
            "brae device cyclic: the pair's layout changed under a mesh move (" + std::to_string(fresh.n) +
            " faces for " + std::to_string(d.n) + "). A move keeps the topology; this refresh cannot follow "
            "a change of it.");
    }
    d.deltaCoeffs = std::move(fresh.deltaCoeffs);
    d.orthDeltaCoeffs = std::move(fresh.orthDeltaCoeffs);
    d.weights = std::move(fresh.weights);
    d.magSf = std::move(fresh.magSf);
    d.Sfx = std::move(fresh.Sfx);
    d.Sfy = std::move(fresh.Sfy);
    d.Sfz = std::move(fresh.Sfz);
    d.dOwnX = std::move(fresh.dOwnX);
    d.dOwnY = std::move(fresh.dOwnY);
    d.dOwnZ = std::move(fresh.dOwnZ);
    d.dNbrX = std::move(fresh.dNbrX);
    d.dNbrY = std::move(fresh.dNbrY);
    d.dNbrZ = std::move(fresh.dNbrZ);
    d.corrVecX = std::move(fresh.corrVecX);
    d.corrVecY = std::move(fresh.corrVecY);
    d.corrVecZ = std::move(fresh.corrVecZ);
    d.dX = std::move(fresh.dX);
    d.dY = std::move(fresh.dY);
    d.dZ = std::move(fresh.dZ);
    if (d.stencil)
    {
        d.stOff = std::move(fresh.stOff);
        d.stCell = std::move(fresh.stCell);
        d.stW = std::move(fresh.stW);
        d.stIsAmi = std::move(fresh.stIsAmi);
    }
}

// THE PAIR'S FACE AREAS, refreshed IN PLACE after a cyclicACMI rescale: magSf and Sf, in the order
// buildDeviceCyclic laid them out, and NOTHING ELSE. OpenFOAM's rescale (cyclicACMIPolyPatch::
// scalePatchFaceAreas, cyclicACMIFvPatch::resetPatchAreas) moves the areas and leaves the cached
// surfaceInterpolation weights and deltaCoeffs alone -- a static mesh never clears them
// (surfaceInterpolation.C:105-112, cleared only by movePoints/updateGeom) -- and the host's rescale does
// the same, so rebuilding the interface from the rescaled geometry would recompute what neither of them
// does. copyFrom at an unchanged size keeps each buffer's device address, so the layout, the twin map,
// the CSR, phi and ifCoeff -- and every pointer the driver's hooks hold -- stay as they were.
inline void refreshDeviceCyclicAreas(
    DeviceCyclic& d,
    const std::vector<CyclicInterface>& cyclics,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> ms, sfx, sfy, sfz;
    for (const auto& c : cyclics)
    {
        const FvPatch& P = fvp[c.patch];
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            const label gf = P.start + (label)i;
            ms.push_back(g.magSf()[gf]);
            sfx.push_back(g.Sf()[gf].x);
            sfy.push_back(g.Sf()[gf].y);
            sfz.push_back(g.Sf()[gf].z);
        }
    }
    if (static_cast<int>(ms.size()) != d.n)
    {
        throw std::runtime_error(
            "brae device cyclic: the pair's layout changed under an area refresh (" +
            std::to_string(ms.size()) + " faces for " + std::to_string(d.n) + "). The refresh rewrites "
            "areas in place and cannot follow a change of topology.");
    }
    d.magSf.copyFrom(ms);
    d.Sfx.copyFrom(sfx);
    d.Sfy.copyFrom(sfy);
    d.Sfz.copyFrom(sfz);
}

// THE MATRIX VIEW WITH THE PAIR'S OFF-DIAGONAL, in whichever form the pair has: one neighbour cell per
// face (deviceAmul's cyclic term) or the AMI's weighted stencil (its AMI term, cyclicAMIFvPatchField::
// updateInterfaceMatrix). A jump across an AMI is not ported: refused by name.
inline DeviceLduView deviceLduViewPair(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& diag,
    const DeviceBuffer<scalar>& upper,
    const DeviceBuffer<scalar>& lower,
    const DeviceCyclic& cyc,
    const scalar* cycJump = nullptr,
    // the matrix's OWN interface coefficient where the caller kept one: cyc.ifCoeff is scratch that the
    // next assembly on the pair overwrites (MomentumMatrix::cycIfCoeff)
    const DeviceBuffer<scalar>* coeff = nullptr)
{
    const scalar* ifc = (coeff && static_cast<int>(coeff->size()) == cyc.n) ? coeff->data()
                                                                            : cyc.ifCoeff.data();
    if (!cyc.stencil)
    {
        DeviceLduView one = deviceLduViewCyclic(dm, diag, upper, lower, cyc.n, cyc.ownCell.data(),
                                                cyc.nbrCell.data(), ifc, cycJump);
        one.pairRank = cyc.ownerRank();
        one.nPairRanks = cyc.ownerPasses();
        return one;
    }
    if (cycJump)
    {
        throw std::runtime_error(
            "brae device: a jump condition (fixedJump, porousBafflePressure) on a mesh that carries a "
            "cyclicAMI pair. The device applies a jump across a one-to-one pair only.");
    }
    DeviceLduView v = deviceLduViewAmi(dm, diag, upper, lower, cyc.n, cyc.ownCell.data(), cyc.stOff.data(),
                                       cyc.stCell.data(), cyc.stW.data(), ifc);
    v.pairRank = cyc.ownerRank();
    v.nPairRanks = cyc.ownerPasses();
    return v;
}

// THE EXPLICIT NON-ORTHOGONAL CORRECTION ON THE PAIR'S FACES (fvm.cuh, laplacianCorrFluxCoupled):
//     ffc = gamma_f*magSf*lim*(corrVec & (w*grad[own] + (1 - w)*patchNeighbourField(grad)))
// gamma_f the two cells' gamma on the pair's weights, and under `limited <k>` (0 < k < 1) the cap
//     lim = min(k*|dc*(pnf - p_own)| / ((1 - k)*|corr| + 1e-15), 1)
// against coupledFvPatchField::snGrad(nonOrthDeltaCoeffs), the neighbour value LESS the pair's jump.
// A translational cyclic's correction vectors are zero on an orthogonal pair and this is then zero; a
// cyclicAMI's are not. The vector form shares ONE limiter across the three components, as
// limitedSnGrad takes mag() of the whole snGrad and of the whole correction.
void deviceCyclicLapCorrFlux(
    const DeviceCyclic&         cyc,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    scalar                      limitCoeff,
    const DeviceBuffer<scalar>* jump,
    DeviceBuffer<scalar>&       ffcIf);
void deviceCyclicLapCorrFluxVec(
    const DeviceCyclic&         cyc,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>* U[3],
    const DeviceBuffer<scalar>* gx,
    const DeviceBuffer<scalar>* gy,
    const DeviceBuffer<scalar>* gz,
    scalar                      limitCoeff,
    DeviceBuffer<scalar>*       ffcIf);
// cell[own] += sign*f, face by face: a pair face's share of an extensive (un-normalised) face sum
void deviceCyclicAddToOwner(
    const DeviceCyclic&         cyc,
    const DeviceBuffer<scalar>& faceField,
    scalar                      sign,
    DeviceBuffer<scalar>&       cell);

// A per-face field of the pair, zeroed on the cyclicAMI faces: ddtCorr's coupling coefficient is zero
// on every cyclicAMI patch (ddtScheme.C:178-181, 259-262, isA<cyclicAMIFvPatch>). No-op without one.
void deviceCyclicZeroOnAmi(const DeviceCyclic& cyc, DeviceBuffer<scalar>& faceField);

// ifCoeff[j] = gammaFace_j * deltaCoeffs_j * magSf_j  with gammaFace = w*gamma[own] + (1-w)*gamma[nbr]
// (the implicit Laplacian off-diagonal); ALSO folds the diagonal contribution diag[own] -= ifCoeff[j].
// gammaCell is a per-cell diffusivity (rAU for pressure, nuEff for momentum). Pass addToDiag=false to skip
// the diag fold (e.g. when reusing ifCoeff only for the flux corrector).
// `corrected` picks WHICH delta coefficient, as fvm::laplacian's coupled branch does (fvm.cuh:104-113):
// nonOrthDeltaCoeffs when the scheme corrects, the patch's plain deltaCoeffs when it does not. Default
// true, which is what every caller before interFoam was already getting.
void deviceCyclicAssembleLaplacian(DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                                   DeviceBuffer<scalar>& diag, bool addToDiag = true,
                                   bool corrected = true);

// upwind convection on the interface: diag[own] += max(phi,0), ifCoeff[j] += min(phi,0). Adds to the existing
// ifCoeff (call AFTER deviceCyclicAssembleLaplacian) so Apsi[own] += ifCoeff*psi[nbr] carries div-laplacian.
// `phiIf` is the CONVECTING flux on the pair's faces, the caller's (deviceCyclicAssembleMomentum's form): the
// alpha pre-solve convects with phiCN, which is not cyc.phi under CrankNicolson. No default.
void deviceCyclicAddConvection(
    DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& phiIf,
    DeviceBuffer<scalar>& diag,
    const DeviceBuffer<scalar>* wsch = nullptr);

// MOMENTUM matrix interface coupling M = div(phi,U) - laplacian(nuEff,U) (cyc.phi must hold the current flux):
//   ifCoeff[j] = -(nuFace*dc*magSf) + min(phi,0)            (off-diagonal, used in deviceAmul)
//   diag[own] += (nuFace*dc*magSf) + max(phi,0)             (folded into the momentum diagonal)
// with nuFace = w*nuEff[own]+(1-w)*nuEff[nbr]. Mirrors the interior (deviceDivUpwindCoeffs - deviceLaplacianCoeffs).
// `wsch`: the div scheme's face interpolation weight; null = upwind (pos0(phi)), the previous behaviour.
// `corrected` picks the delta coefficient of the DIFFUSION half exactly as the laplacian above does
// (fvm.cuh:104-113): nonOrthDeltaCoeffs when the scheme corrects, the patch's plain deltaCoeffs when it
// does not. Default true, the behaviour every caller before interFoam had.
// fvm::laplacian(nuEff, U) + fvm::div(<flux>, U) across the pair. `convFlux` is the flux the momentum
// equation CONVECTS with, which is not always the pair's own phi: interFoam's UEqn is
// fvm::div(rhoPhi, U), the MASS flux, and cyc.phi is the volumetric one. MEASURED on
// validation/interFoamCyclic: with cyc.phi standing in, rAUf on the pair was 1.1e-03 of 2.0e-03 away
// from the host at step two -- 57% -- and exact at step one, where both fluxes are zero. Null = cyc.phi.
// `convFlux` is the flux the momentum equation CONVECTS with, which is not always the pair's own phi:
// interFoam's UEqn is fvm::div(rhoPhi, U), the MASS flux, and cyc.phi is the volumetric one. MEASURED
// on validation/interFoamCyclic with cyc.phi standing in: rAUf on the pair 1.1e-03 of 2.0e-03 from the
// host at step two -- 57% -- and exact at step one, where both fluxes are zero. Null = cyc.phi.
void deviceCyclicAssembleMomentum(DeviceCyclic& cyc, const DeviceBuffer<scalar>& nuEffCell, DeviceBuffer<scalar>& diag,
                                  const DeviceBuffer<scalar>* wsch = nullptr,
                                  bool corrected = true,
                                  const DeviceBuffer<scalar>* convFlux = nullptr);

// add the cyclic off-diagonal to H (OF fvMatrix::H): H[own] -= ifCoeff[j]*psi[nbr]/V[own]. Call AFTER deviceMatrixH.
// fvMatrix::H(): H[own] -= coeff*psi[nbr]/V[own]. `coeff` is the MATRIX's own interface off-diagonal.
// cyc.ifCoeff is one array that every assembly overwrites, so a caller whose matrix was assembled
// before another one must pass its own copy; null falls back to cyc.ifCoeff.
void deviceCyclicAddH(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& psi, const DeviceBuffer<scalar>& V,
                      DeviceBuffer<scalar>& H,
                      const DeviceBuffer<scalar>* coeff = nullptr);
// per-owner sum of |ifCoeff| (the cyclic off-diagonal magnitude) for the relaxation's diagonal-dominance term.
void deviceCyclicOffDiagSum(const DeviceCyclic& cyc, DeviceBuffer<scalar>& sumOff);

// interface off-diagonal pass for the SpMV: Apsi[own] += ifCoeff[j]*psi[nbr]. Called inside deviceAmul when the
// DeviceLduView carries cyclic data (declared in device_ldu.cuh / applied in device_spmv.cu).

// cyclic-face flux of an interpolated vector field: phi[j] = (w*Hx[own]+(1-w)*Hx[nbr])*Sfx + (y,z).
void deviceCyclicFlux(DeviceCyclic& cyc, const DeviceBuffer<scalar>& Hx, const DeviceBuffer<scalar>& Hy,
                      const DeviceBuffer<scalar>& Hz);
// ...into a caller's array rather than into cyc.phi. phiHbyA is not the pair's flux: phi there is state
// the correctors rewrite, and computing phiHbyA over it would lose the flux the step began with.
void deviceCyclicFluxTo(DeviceCyclic& cyc, const DeviceBuffer<scalar>& Hx, const DeviceBuffer<scalar>& Hy,
                        const DeviceBuffer<scalar>& Hz, DeviceBuffer<scalar>& out);

// ROTATIONAL (Phase 1), only the VECTOR (U/HbyA) couplings differ; scalars use the functions above
// fill the per-component implicit off-diagonal ifCoeffC[kk] = ifCoeff * forwardT[kk][kk] (call after AssembleMomentum).
void deviceCyclicScaleImplicit(DeviceCyclic& cyc);
// rotational HbyA flux: the neighbour vector is rotated, phi = (w*H[own] + (1-w)*forwardT.H[nbr]) . Sf.
void deviceCyclicFluxRot(DeviceCyclic& cyc, const DeviceBuffer<scalar>& Hx, const DeviceBuffer<scalar>& Hy,
                         const DeviceBuffer<scalar>& Hz);
// rotational H (full rotation, mixes components): H[kk][own] -= ifCoeff * (forwardT.U[nbr])[kk] / V[own], all 3 kk.
void deviceCyclicAddHRot(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                         const DeviceBuffer<scalar>& Uz, const DeviceBuffer<scalar>& V,
                         DeviceBuffer<scalar>& Hx, DeviceBuffer<scalar>& Hy, DeviceBuffer<scalar>& Hz);
// rotational gaussGrad of vector component `comp`: grad[own] += Sf * (w*U[comp][own] + (1-w)*(forwardT.U[nbr])[comp]) / V.
void deviceCyclicAddGradRot(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                            const DeviceBuffer<scalar>& Uz, int comp, const DeviceBuffer<scalar>& V,
                            DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz);
// DEFERRED rotation correction (component comp) into the momentum source: src[own] -= ifCoeff*[(forwardT.U[nbr])[comp]
//   - forwardT[comp][comp]*U[comp][nbr]] (the off-diagonal component-MIXING the diag-only implicit drops). With this in
// the source AND the diagonal H (deviceCyclicAddHDiag), the predictor + H both carry the FULL rotation -> consistent ->
// converges. Mirrors the standard deferred-correction split of a non-implementable implicit term. NOT /V (matrixH does).
void deviceCyclicAddDeferredRot(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                                const DeviceBuffer<scalar>& Uz, int comp, DeviceBuffer<scalar>& src);
// diagonal cyclic H for component comp: H[own] -= ifCoeffC[comp]*U[comp][nbr]/V (pairs with the deferred correction).
void deviceCyclicAddHDiag(const DeviceCyclic& cyc, int comp, const DeviceBuffer<scalar>& psi,
                          const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& H);
// continuity: div[own] += phi[j]/V[own]  (deviceDiv returns the volume-normalized divergence Sum phi / V).
void deviceCyclicAddDiv(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div);
// ...and the same for a flux that is NOT the interface's own phi: MULES divides its LIMITED flux, which
// is a different array from the one the pair carries.
void deviceCyclicAddDivFlux(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& phi,
                            const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div);
// epsilon setValues: zero the cyclic interface off-diagonal for wall-cell owners (their eps is fixed = eps0).
void deviceCyclicZeroWallIfCoeff(DeviceCyclic& cyc, const DeviceBuffer<label>& isWallCell);
// pressure-correction flux: phi[j] -= ifCoeff[j]*(p[nbr]-p[own])  (the snGrad(p) flux across the periodic face).
void deviceCyclicCorrectFlux(DeviceCyclic& cyc, const DeviceBuffer<scalar>& p,
                             // the pair's already-signed JUMP (fixedJump, porousBafflePressure), or
                             // null. fvMatrix::flux() reads patchNeighbourField(), which on a jump
                             // cyclic is the cell across LESS the jump.
                             const DeviceBuffer<scalar>* jump = nullptr);
// ...and the SAME flux as a value rather than a subtraction: out[j] = ifCoeff[j]*(p[nbr]-p[own]), which
// is fvMatrix::flux() on that face. interFoam's velocity correction needs it, because
// reconstruct((phig - p_rghEqn.flux())/rAUf) reads the flux and not the corrected phi.
void deviceCyclicPressureFlux(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& p,
                              DeviceBuffer<scalar>& out,
                              const DeviceBuffer<scalar>* jump = nullptr);
// gaussGrad contribution: grad[own] += Sf_j * (w*psi[own]+(1-w)*psi[nbr]) / V[own].
// fvc::snGrad ON THE PAIR'S FACES (fvc.cu, snGradScheme.C's coupled branch): dc*(pnf - psi[own]) with the
// scheme's delta coefficient -- nonOrthDeltaCoeffs under `corrected` and `uncorrected`, the plain 1/|delta| under
// `orthogonal` -- and under `corrected` the correction vector dotted with the two sides' gradients interpolated
// on the pair's weights, capped under `limited <k>` as an internal face's is. No jump: the callers' fields
// (alpha, rho) cross the pair unchanged.
void deviceCyclicSnGrad(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& psi,
                        const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                        const DeviceBuffer<scalar>& gz, bool corrected, scalar limitCoeff, bool nonOrthCoeffs,
                        DeviceBuffer<scalar>& out);
// fvc::interpolate of a CELL field onto the cyclic faces: w*psi[own] + (1-w)*psi[nbr].
void deviceCyclicFaceValue(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& cell, DeviceBuffer<scalar>& out);
// OF patchNeighbourField() for a cyclic patch: the raw neighbour cell value per face (rotated by forwardT
// on a rotational cyclic, which is why it takes all three components). For cellLimitedGrad's range.
void deviceCyclicNbrValue(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& cell,
                          const DeviceBuffer<scalar>& c0, const DeviceBuffer<scalar>& c1,
                          const DeviceBuffer<scalar>& c2, int comp, DeviceBuffer<scalar>& out);

void deviceCyclicAddGrad(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& psi, const DeviceBuffer<scalar>& V,
                         DeviceBuffer<scalar>& gx, DeviceBuffer<scalar>& gy, DeviceBuffer<scalar>& gz,
                         // the pair's already-signed JUMP, or null -- see DeviceLduView::cycJump
                         const DeviceBuffer<scalar>* jump = nullptr);
// linearUpwind deferred correction at the cyclic interface (component comp). Mirrors deviceLinearUpwindCorr but
// the neighbour reconstruction is ROTATED: per face, corr[own] += phi * (gradU_upwind . d_upwind)[comp] with
//   phi>=0 (own upwind):  (grad(U_comp)[own] . dOwn)
//   phi< 0 (nbr upwind):  (forwardT . (gradU[nbr] . dNbr))[comp]  = sum_l forwardT[comp][l] * (grad(U_l)[nbr].dNbr)
// gU{x,y,z}[l] = grad(U_l) (all 3 components, cyclic-inclusive). Caller does relaxSrc[comp] -= corr (as for internal).
// `flux` is the equation's own face flux on the pair -- rhoPhi where the matrix was assembled with
// rhoPhi -- or null to take the interface's volumetric cyc.phi.
void deviceCyclicAddLinUpwindCorr(const DeviceCyclic& cyc, int comp,
                                  const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy,
                                  const DeviceBuffer<scalar>* gUz, DeviceBuffer<scalar>& corr,
                                  const DeviceBuffer<scalar>* flux = nullptr);
// non-orth laplacian "corrected" correction at the cyclic interface (component comp). Per face the explicit face-flux
// correction is ffc = gammaFace*magSf*(corrVec . grad(U_comp)_face), gathered as src[own] -= ffc (owner side), and the
// caller does relaxSrc -= src (mirrors the internal deviceLaplacianCorr). The neighbour gradient ROTATES as a tensor:
// grad((forwardT.U)_comp) = (forwardT . gradU[nbr] . forwardT^T)[comp][:], so it needs ALL 3 component gradients.
// gammaCell = nuEff per cell (interpolated to the face by w). corr is the SAME buffer the internal corr was written to.
// SCALAR overloads (k/epsilon/omega/nuTilda/he): one gradient, never rotated across the interface.
void deviceCyclicAddLinUpwindCorr(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gx,
                                  const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                                  DeviceBuffer<scalar>& corr);
void deviceCyclicAddLapCorr(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                            const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                            const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corr);
void deviceCyclicAddLapCorr(const DeviceCyclic& cyc, int comp, const DeviceBuffer<scalar>& gammaCell,
                            const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy,
                            const DeviceBuffer<scalar>* gUz, DeviceBuffer<scalar>& corr);
// PRESSURE non-orth correction at the cyclic interface (SCALAR field p, no rotation). Per face ffc =
// rAtU_face*magSf*(corrVec . grad(p)_face); adds -ffc to bp[own] (the -V*div pressure source) and returns ffc per
// face in ffcOut so the caller can correct the continuity flux post-solve (cyc.phi -= ffcOut).
void deviceCyclicLapCorrP(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& gammaCell,
                          const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz,
                          DeviceBuffer<scalar>& bp, DeviceBuffer<scalar>& ffcOut);
// cyclic contribution to the divDevReff tensor divergence (the RAW sum V*fvc::div(sigma), no /V, matches
// tensorDivKernel): srcX/Y/Z[own] += (Sf_j & interp(sigmaC))_{x/y/z}, sigmaC packed component-major (i*3+j)*nC+c.
void deviceCyclicAddTensorDiv(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& sigmaC, int nC,
                              DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ);

// The DIV SCHEME's face weight at the cyclic faces -- the cyclic twin of deviceAmiLimitedVWeights, with
// the 1:1 neighbour in place of the AMI stencil. See device_ami.cuh for the packing and the provenance.
void deviceCyclicLimitedVWeights(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& Ux,
                                 const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                                 const DeviceBuffer<scalar>& gradU, int nC, scalar twoByk,
                                 DeviceBuffer<scalar>& out);
void deviceCyclicLimitedWeights(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& f,
                                const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                                const DeviceBuffer<scalar>& gz, scalar twoByk, DeviceBuffer<scalar>& out);

} // namespace brae
