// Out-of-line definitions of brae::DeviceSimpleSolver, moved from device_simple_foam.cuh so brae_core compiles the
// SIMPLE solver bodies ONCE (every consumer TU used to re-parse ~1000 lines of inline CUDA). Bodies are verbatim;
// the class declaration (members + method signatures) stays in the header. Prerequisite for the PIMPLE solver,
// which reuses solveMomentumPredictor / correctPressureVelocity / correctTurbulence under a transient loop.
#include "device_simple_foam.cuh"
#include "swept_volume.cuh"   // meshPhi + makeRelative (OF fvcMeshPhi)
#include "device_scalar_transport.cuh"   // deviceSolveScalarTransport: shared with k/epsilon/omega/energy
#include "device_deshybrid.cuh"   // DEShybrid: per-face blend of linear and linearUpwind (DES sensor)
#include "device_maxwell.cuh"   // Maxwell viscoelastic stress (laminar { model Maxwell; })
#include "fan_pressure.cuh"   // applyFixedMean: OF fixedMeanFvPatchField, shared with its test
#include "brae_notice.cuh"   // never drop an input silently (the ddtCorr scheme gate)
#include "stage_dump.cuh"   // Phase 0 stage harness (cudafoam/rhosimplefoam-restage-plan.md)

namespace brae {

namespace {
// The level-0 fine coefficients over the EXTENDED edge list the AMG hierarchy was built from:
// [internal faces | cyclic entries | AMI stencil entries].
//
// THE SIGN, and it is the whole subtlety. brae's lduMatrix edge means
//     Ax[nei] += lower*x[own] ;  Ax[own] += upper*x[nei]
// while an interface entry means only  Apsi[own] += ifCoeff*psi[nbr]  -- ONE direction. The reverse
// direction is a SEPARATE appended edge, contributed by the matching entry on the other patch. So each
// edge carries upper = ifCoeff and lower = ZERO; writing ifCoeff into both would count every periodic
// connection twice.
//
// It still comes out symmetric on the coarse grid, which is what the V-cycle needs: the two edges (A,B)
// and (B,A) agglomerate to the same coarse face with opposite orientation, and agglomerate()'s faceFlip
// sends the reversed one's upper into cLower. Where both cells land in the same aggregate the pair goes
// into the coarse diagonal instead, which is equally right.
__global__
void amgFineCoeffKernel(
    int nIf, int nCyc, int nAmi,
    const scalar* __restrict__ pU,
    const scalar* __restrict__ pL,
    const scalar* __restrict__ cycIf,
    const scalar* __restrict__ amiIf,
    const label*  __restrict__ amiSrc,
    const scalar* __restrict__ amiW,
    scalar* __restrict__ up,
    scalar* __restrict__ lo)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nIf + nCyc + nAmi) return;
    if (i < nIf)              { up[i] = pU[i];  lo[i] = pL[i]; return; }
    if (i < nIf + nCyc)       { up[i] = cycIf[i - nIf];                    lo[i] = 0; return; }
    const int j = i - nIf - nCyc;
    up[i] = amiIf[amiSrc[j]] * amiW[j];
    lo[i] = 0;
}
}   // namespace



    // BRAE_SETUP_TIMING: same milestone trace as the driver, inside the constructor. The ctor is the
    // longest silent stretch of set-up (mesh upload, AMG hierarchy, wall distance, validate) and a hang
    // in it looks identical to a hang in the driver from the outside.
    static inline void ctorMark(const char* what)
    {
        static const bool on = (std::getenv("BRAE_SETUP_TIMING") != nullptr);
        if (!on) return;
        static const auto t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "  [ctor  %7.2fs] %s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), what);
    }

    DeviceSimpleSolver::DeviceSimpleSolver(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& fvp,
        const GeometricField<vector>& U,
        const GeometricField<scalar>& p,
        const SurfaceScalarField& phi,
        const DeviceSimpleControls& ctl,
        const GeometricField<scalar>* k,
        const GeometricField<scalar>* eps,
        const GeometricField<scalar>* nut,
        const GeometricField<scalar>* ReThetat,
        const GeometricField<scalar>* gammaInt,
    bool                          phiWasRead)
        : fvp_(fvp), ctl_(ctl), nC_(m.nCells()), nIf_(m.nInternalFaces())
    {
        // cyclic (periodic) interfaces: a SEPARATE lduInterface (OF cyclicFvPatchField::updateInterfaceMatrix),
        // NOT merged into the owner-sorted LDU. Both sides of each pair are stored (symmetric coupling).
        const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
        hasCyclic_ = !cyclics.empty();
        // cyclicAMI: non-conforming interface, an area-weighted lduInterface (OF cyclicAMIFvPatchField). Built as a
        // weighted CSR stencil (device_ami); the assembly mirrors the cyclic path with the AMI interpolation.
        const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);
        hasAMI_ = !amis.empty();
        // IS THE INTERFACE CONFORMING? -- i.e. does every source face have exactly ONE partner?
        //
        // This decides the pressure SOLVER, because it decides whether the pressure operator is
        // self-adjoint. The interface contributes ifCoeff_i*w_ij to A[c_i][c_j] and the reverse entry
        // comes from the other direction as ifCoeff_j*w_ji; with w_ij = overlap/|Sf_i| those agree only
        // when deltaCoeffs_i == deltaCoeffs_j. deltaCoeffs is 1/(nf & delta) with
        // delta = patchD - Sum_j w_ij*nbrD_j, so the moment a face has more than one partner its delta
        // is a WEIGHTED AVERAGE over a different set of partners than the reverse direction averages,
        // and the two no longer match. Measured (test_interface_invariants_real): a conforming pair
        // comes out at 4e-16 and 4e-15, a non-conforming one at 1.3e-01 -- fourteen orders apart, so
        // this is structural and not round-off.
        //
        // This is NOT a brae defect to be normalised away: OpenFOAM computes cyclicAMI deltaCoeffs per
        // side by the same formula and inherits the same asymmetry, which is why its AMI tutorials ship
        // GAMG or PBiCGStab for p and none of them ship PCG. The defect was solving it with Jacobi-PCG
        // regardless -- and CG does not merely lose efficiency on a non-symmetric operator, it fails to
        // converge: on pimpleFoam/RAS/oscillatingInletPeriodicAMI2D every solve ran to the 50-iteration
        // cap with the residual GROWING (1.00 -> 1.32) where OF's GAMG converged in 12. Same failure and
        // same remedy as the transonic pressure matrix below.
        for (const AMIInterface& a : amis)
            if (a.nbrCell.size() > a.ownCell.size()) { amiNonConforming_ = true; break; }
        if (std::getenv("BRAE_AMI_PCG")) amiNonConforming_ = false;   // attribution escape hatch
        if (hasCyclic_ || hasAMI_) ctl_.useGraph = false;   // V-cycle graph replay not interface-safe; minor perf only.
        // The DILU level schedule: mesh addressing only, so once per solver. Skipped entirely when the
        // case did not ask for DILU, so no case pays for it unasked.
        // ONE level schedule for every DILU-preconditioned solve on this mesh -- momentum and the
        // turbulence pair alike. The schedule is a property of the addressing, not of the matrix, and
        // diluUpdate recomputes rD from whichever matrix is being solved, so building it twice would
        // duplicate the DAG for nothing.
        if (ctl_.diluU || ctl_.diluKE) dilu_ = buildDeviceDilu(m.owner(), m.neighbour(), nC_);
        turbPrecon() = (ctl_.diluKE && dilu_.valid) ? &dilu_ : nullptr;
        ctorMark("enter");
        dm_   = buildDeviceMesh(m, g, fvp);
        ctorMark("deviceMesh built");
        cyc_  = buildDeviceCyclic(cyclics, g, fvp);
        ami_  = buildDeviceAMI(amis);
        for (const auto& c : cyclics) cycRuns_.push_back({c.patch, (label)c.faceCells.size()});
        for (const auto& a : amis)    amiRuns_.push_back({a.patch, (label)a.ownCell.size()});
        dbU_  = buildDeviceVectorBoundary(U, fvp, g);
        dbP_  = buildDeviceBoundary(p, fvp, g);
        // pcorr's boundary, for CorrectPhi. OF builds it as zeroGradient EVERYWHERE except the patches
        // where p itself fixes a value, which become fixedValue 0 (CorrectPhi.C:56-70). It is p's
        // geometry with p's types thrown away, so it is built here beside dbP_ rather than derived at
        // the call site -- the two must see the same patch order or the flux correction lands on the
        // wrong faces.
        {
            GeometricField<scalar> pc;
            pc.internal.assign(static_cast<std::size_t>(m.nCells()), scalar(0));
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (p.boundary[pi]->fixesValue())
                    pc.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(
                        fvp[pi], true, scalar(0), std::vector<scalar>{}));
                else
                    pc.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(fvp[pi]));
            }
            pc.evaluateBoundary();
            dbPcorr_ = buildDeviceBoundary(pc, fvp, g);
        }
        wall_ = buildDeviceWallData(m, g, fvp, U, ctl_.turbWallPatch);
        {   // DeviceBoundary face offset of each patch, for setPatchVelocity
            label st = 0;
            patchStart_.assign(fvp.size(), 0);
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                patchStart_[pi] = st;
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                st += fvp[pi].size;
            }
        }
        if (nut)
        {
            std::vector<label> cm, cs, fx;
            // nut's OWN boundaryField, in DeviceBoundary face order. OF evaluates a non-wall patch's
            // nut from this, never from the adjacent cell -- extrapolating is 607x wrong at a
            // `calculated` inlet (rhosimplefoam-ground-truth-port.md section 3).
            std::vector<scalar> nb;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                const int nutCat = nut->boundary[pi]->bcCategory();
                const bool calc = (fvp[pi].type != "wall") && (nutCat == 2);
                // A non-wall `fixedValue` nut patch is NOT a `calculated` one and must not be lumped in
                // with it: `calculated` means "correctNut filled this in", so Cmu*k_b^2/eps_b is right
                // there, while `fixedValue` means the case PINNED it -- conventionally to 0 -- and the
                // read value is the only correct answer. brae used the adjacent CELL nut at such a patch.
                // Traced against the OF oracle on pimpleFoam/RAS/oscillatingInletPeriodicAMI2D: nu is
                // 1e-06 but Cmu*k^2/eps is 7.03e-04 in the first cell, so the inlet's momentum
                // internalCoeffs came out 704x OpenFOAM's (predicted 704.1, measured 703.1) -- which is
                // where that case's H(), and everything downstream of it, went wrong.
                const bool nutFixed = (fvp[pi].type != "wall") && (nutCat == 1);
                if (nutFixed) hasNutFixed_ = true;
                if (calc) hasNutCalc_ = true;
                const std::vector<scalar>& pv = nut->boundary[pi]->value();
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    cm.push_back(calc ? 1 : 0);
                    // ...and its complement, because deviceSelectFixedFlux copies where the mask is ZERO.
                    cs.push_back(calc ? 0 : 1);
                    // 0 marks "take nut's own read value here" -- the sense deviceSelectFixedFlux uses.
                    fx.push_back(nutFixed ? 0 : 1);
                    nb.push_back(pv[i]);
                }
            }
            if (hasNutCalc_) { nutCalcMask_.copyFrom(cm); nutCalcSel_.copyFrom(cs); }
            if (hasNutFixed_) nutFixedMask_.copyFrom(fx);
            nutBndFile_.copyFrom(nb);
        }
        {
            // Wall-face -> boundary-face index. buildDeviceWallData walks fvp keeping the TURBULENCE WALL
            // patches (isTurbWallPatch: type=="wall" AND named by the epsilon/omega BC); DeviceBoundary
            // walks the same fvp skipping cyclic/cyclicAMI (a wall is neither), so the wall faces are a
            // subsequence of the boundary faces and this map is just the running count. The two MUST use
            // the same predicate or the map silently points at the wrong faces.
            std::vector<label> wfb;
            label bi = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (label i = 0; i < fvp[pi].size; ++i, ++bi)
                    if (isTurbWallPatch(fvp, pi, ctl_.turbWallPatch)) wfb.push_back(bi);
            }
            if (!wfb.empty()) wfBndIdx_.copyFrom(wfb);
        }
        // all-extrapolated boundary -> gathers a cell field to boundary faces (nuEff_bnd = adjacent cell value)
        {
            std::vector<label> ty, fc;
            std::vector<scalar> ref, dc, ms;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    ty.push_back(0);
                    fc.push_back(fvp[pi].faceCells[i]);
                    ref.push_back(0.0);
                    dc.push_back(fvp[pi].deltaCoeffs[i]);
                    ms.push_back(g.magSf()[fvp[pi].start + i]);
                }
            }
            dbExtrap_.n = (int)ty.size();
            dbExtrap_.bcType.copyFrom(ty);
            dbExtrap_.refValue.copyFrom(ref);
            dbExtrap_.deltaCoeffs.copyFrom(dc);
            dbExtrap_.magSf.copyFrom(ms);
            dbExtrap_.faceCell.copyFrom(fc);
            dbExtrap_.valueFraction.copyFrom(std::vector<scalar>(ty.size(), 0.0));
            dbExtrap_.mixedMask.copyFrom(std::vector<label>(ty.size(), 0));
        }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (U.boundary[pi]->wedgeFaceT())   // axisymmetric wedge: the value is the ROTATED cell value
            {
                hasWedge_ = true;
                break;
            }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (U.boundary[pi]->bcCategory() == 3)   // inletOutlet
            {
                hasInletOutletU_ = true;
                break;
            }
        // any freestreamVelocity/Pressure (mixed) far-field patch present -> run the per-step valueFraction update.
        // A wedge also reports category 5 and must NOT count: its valueFraction is geometry, not a flux sign
        // (see the mixedMask comment in buildDeviceVectorBoundary), so a purely axisymmetric case has no
        // per-step valueFraction to update at all.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if ((U.boundary[pi]->bcCategory() == 5 && !U.boundary[pi]->wedgeFaceT())
             || p.boundary[pi]->bcCategory() == 5)
            {
                hasMixed_ = true;
                break;
            }
        // any pressureInletOutletVelocity (directionMixed) U patch present -> run its per-step updateCoeffs.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (U.boundary[pi]->bcCategory() == 6)
            {
                hasPiov_ = true;
                break;
            }
        // any slip/symmetry U patch -> run its per-step updateCoeffs (general non-axis-aligned normal).
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (U.boundary[pi]->isSymmetry())
            {
                hasSym_ = true;
                break;
            }
        // flowRateInletVelocity (massFlowRate): one masked-magSf buffer per such patch, plus the outward
        // normals over all boundary faces. Both are geometric and built once.
        {
            std::vector<scalar> nx, ny, nz;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    nx.push_back(fvp[pi].nf[i].x);
                    ny.push_back(fvp[pi].nf[i].y);
                    nz.push_back(fvp[pi].nf[i].z);
                }
            }
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (U.boundary[pi]->bcCategory() != 9) continue;
                hasFlowRate_ = true;
                std::vector<scalar> mask(nx.size(), 0.0);
                label bi = 0;
                for (std::size_t pj = 0; pj < fvp.size(); ++pj)
                {
                    if (isCoupledInterfaceType(fvp[pj].type)) continue;
                    for (label i = 0; i < fvp[pj].size; ++i, ++bi)
                        if (pj == pi) mask[bi] = fvp[pj].magSf[i];
                }
                frMagSf_.emplace_back();
                frMagSf_.back().copyFrom(mask);
                // OF re-reads flowRate_->value(t) each call; steady + constant -> the seeded value. It is
                // recovered from the seeded BC rather than re-parsed: avgU*sum(rho_seed*magSf) = -mdot.
                frPatches_.push_back(FlowRatePatch{U.boundary[pi]->flowRateValue()});
            }
            if (hasFlowRate_) { frNx_.copyFrom(nx); frNy_.copyFrom(ny); frNz_.copyFrom(nz); }
        }
        // any totalPressure p patch -> recompute its fixedValue refValue each step from the patch velocity + flux.
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (p.boundary[pi]->bcCategory() == 7)
            {
                hasTotalP_ = true;
                break;
            }
        // per-boundary-face wall mask + nearWallDist y (for the true boundary nut = nutkWallFunction at walls).
        {
            const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
            std::vector<label> isW;
            std::vector<scalar> yv;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                const bool wall = (fvp[pi].type == "wall");
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    isW.push_back(wall ? 1 : 0);
                    yv.push_back(wall ? yW[pi][i] : 0.0);
                }
            }
            bndIsWall_.copyFrom(isW);
            bndY_.copyFrom(yv);
        }
        // flat boundary face centres (patch order, cyclic/cyclicAMI excluded) -- the absolute-position Cf indexing used
        // by the NVRTC coded-BC kernels (aligned to the DeviceBoundary refValue/faceCell order). Built once at setup.
        {
            std::vector<scalar> bx, by, bz;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    const vector& cf = g.Cf()[fvp[pi].start + i];
                    bx.push_back(cf.x); by.push_back(cf.y); bz.push_back(cf.z);
                }
            }
            bndCfX_.copyFrom(bx); bndCfY_.copyFrom(by); bndCfZ_.copyFrom(bz);
        }
        // adjustPhi adjustable mask (patch order = phiBnd_ order): a face is adjustable iff its U patch does NOT
        // fix the flux (zeroGradient/inletOutlet/calculated) -> OF "!fixesValue || isA<inletOutlet>".
        {
            std::vector<label> adj;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                const bool fixed = U.boundary[pi]->fixesValue();
                for (label i = 0; i < fvp[pi].size; ++i)
                    adj.push_back(fixed ? 0 : 1);
            }
            adjustMask_.copyFrom(adj);
        }
        // fvc::ddtCorr's per-boundary-face coefficient mask. OF's ddtScheme::fvcDdtPhiCoeff zeroes the
        // coupling coefficient on exactly two kinds of patch:
        //     if (U.boundaryField()[patchi].fixesValue() || isA<cyclicAMIFvPatch>(...)) ccbf[patchi] = 0
        // The cyclicAMI half is why the missing ddtCorr was never an INTERFACE defect -- OF puts nothing
        // there either. Coupled patches are not in this array at all (phiBnd_ skips them), so they get
        // no correction for free; the mask only has to carry the fixesValue half.
        {
            std::vector<scalar> mask;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                const scalar m01 = U.boundary[pi]->fixesValue() ? scalar(0) : scalar(1);
                for (label i = 0; i < fvp[pi].size; ++i) mask.push_back(m01);
            }
            ddtCorrBndMask_.copyFrom(mask);
        }
        nuBndConst_.copyFrom(std::vector<scalar>(dbExtrap_.n, ctl.nu));
        // AMG hierarchy for the pressure Laplacian (static: faceWeights = |Sf|), built from the mesh internal faces
        // PLUS the coupled-interface edges.
        //
        // The interface edges used to be left out, and the whole AMG with them: a cyclic or AMI mesh fell back to
        // Jacobi-PCG over the interface-coupled fine operator. That is a diagonal preconditioner on a Poisson
        // problem, and it costs what one costs -- measured on pipeCyclic, 1250 cells with a periodic pair, brae
        // took 153/118/130 pressure iterations where OpenFOAM's GAMG took 5/5/2, while brae on the 10x LARGER
        // interface-free pitzDaily took 18. A mesh ten times smaller needing eight times the iterations is the
        // O(sqrt(N)) signature of losing the multigrid, not a property of the mesh.
        //
        // Nothing about a periodic edge actually resists agglomeration: buildAMG takes a plain edge list, and an
        // interface entry IS an edge -- (ownCell, nbrCell) for a conformal 1:1 pair, and (ownCell[i], nbrCell[k])
        // per stencil entry with the AMI weight folded into the face weight for a non-conformal one. The
        // agglomeration is symmetric in owner/nei and carries a faceFlip per fine face, so an edge whose coarse
        // orientation comes out reversed scatters into cLower instead of cUpper by itself.
        //
        // Each physical periodic connection appears TWICE, once from each side's own patch entry, and that is
        // exactly right: the two carry the two directions of one lduMatrix edge. Coarsening sees the pair as one
        // strong connection, which is what it is.
        //
        // OpenFOAM does the same thing by a different route -- a dedicated cyclicGAMGInterface agglomerated at
        // every level -- and the point of both is the same: the coarse grid has to be able to move the periodic
        // mode, or the outer Krylov has to do it one iteration at a time.
        std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf_), neiInt(m.neighbour());
        std::vector<scalar> fwAMG(g.magSf().begin(), g.magSf().begin() + nIf_);   // default: geometric face area (bit-identical)
        if (std::getenv("BRAE_AMG_SOC"))   // SoC ON: weight = LAPLACIAN coupling magSf*deltaCoeffs (the real matrix strength,
        {
            const auto& dcg = g.deltaCoeffs();                                    // up to ~constant rAU) so agglomerate()'s
            for (label f = 0; f < nIf_; ++f)
                fwAMG[f] *= dcg[f];                  // strong-face filter sees the true anisotropy
        }
        // The interface edges, appended after the internal faces so a fine coefficient array is simply
        // [internal faces | interface entries] and level-0 Galerkin reads them with no extra addressing.
        // nAmgIfEdges_ records how many were added; zero on a mesh without interfaces, which leaves the
        // hierarchy and every existing case bit-identical.
        nAmgIfEdges_ = 0;
        amgIfOwn_.clear();
        amgIfNbr_.clear();
        if (hasCyclic_ || hasAMI_)
        {
            const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);
            for (const CyclicInterface& c : cyclics)
                for (std::size_t i = 0; i < c.faceCells.size(); ++i)
                {
                    amgIfOwn_.push_back(c.faceCells[i]);
                    amgIfNbr_.push_back(c.nbrFaceCells[i]);
                    // |Sf| of this patch face, the same geometric weight the internal faces carry.
                    fwAMG.push_back(fvp[c.patch].magSf[i]);
                }
            for (const AMIInterface& a : amis)
                for (std::size_t i = 0; i < a.ownCell.size(); ++i)
                    for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k)
                    {
                        amgIfOwn_.push_back(a.ownCell[i]);
                        amgIfNbr_.push_back(a.nbrCell[k]);
                        fwAMG.push_back(a.magSf[i] * a.weight[k]);   // the AMI weight IS the edge's share
                    }
            {
                std::vector<label>  aSrc;
                std::vector<scalar> aW;
                for (const AMIInterface& a : amis)
                    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
                        for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k)
                        {
                            aSrc.push_back(static_cast<label>(i));
                            aW.push_back(a.weight[k]);
                        }
                nAmgAmiEdges_ = static_cast<label>(aSrc.size());
                if (nAmgAmiEdges_) { amgIfAmiSrc_.copyFrom(aSrc); amgIfAmiW_.copyFrom(aW); }
            }
            ownerInt.insert(ownerInt.end(), amgIfOwn_.begin(), amgIfOwn_.end());
            neiInt.insert(neiInt.end(), amgIfNbr_.begin(), amgIfNbr_.end());
            nAmgIfEdges_  = static_cast<label>(amgIfOwn_.size());
            nAmgCycEdges_ = nAmgIfEdges_ - nAmgAmiEdges_;
            std::printf("  AMG: %d interface edge(s) agglomerated alongside %d internal faces\n",
                        (int)nAmgIfEdges_, (int)nIf_);
        }
        ctorMark("pre-AMG");
        // The hierarchy CACHE keys on the mesh only, so a run that adds interface edges must not reload a
        // hierarchy built without them (or vice versa). Interface meshes therefore build fresh.
        amg_ = nAmgIfEdges_
             ? buildAMG(ownerInt, neiInt, fwAMG, nC_)
             : buildOrLoadAMG(ownerInt, neiInt, fwAMG, nC_, ctl_.caseDir + "/constant/polyMesh", ctl_.writeCache);

        // initial device state.
        {
            std::vector<scalar> ux(nC_), uy(nC_), uz(nC_);
            for (label c = 0; c < nC_; ++c)
            {
                ux[c]=U.internal[c].x;
                uy[c]=U.internal[c].y;
                uz[c]=U.internal[c].z;
            }
            Uk_[0].copyFrom(ux);
            Uk_[1].copyFrom(uy);
            Uk_[2].copyFrom(uz);
        }
        dp_.copyFrom(p.internal);
        phiInt_.copyFrom(phi.internal);             // internal-face flux (cyclic flux is held separately in cyc_.phi)
        // initialise the persistent cyclic-face flux. The host phi was built with the cyclic patch as a zeroGradient
        // PLACEHOLDER (makePatchField), so its cyclic boundary flux is the un-coupled, un-rotated own-cell value (and
        // for rotational it is NOT conservative). Recompute it on the device from U_init with the proper coupled +
        // rotated interpolation (deviceCyclicFlux[Rot] is conservative) so the first momentum convection + continuity
        // see the correct periodic flux. (For a rest start U=0 this is 0 either way; it matters for a swirling init.)
        if (hasCyclic_)
        {
            if (cyc_.rotational) deviceCyclicFluxRot(cyc_, Uk_[0], Uk_[1], Uk_[2]);
            else                 interfaceFlux(cyc_, Uk_[0], Uk_[1], Uk_[2]);
        }
        // The AMI interface flux. Rebuilding it from U is right for a COLD start -- the host phi carries a
        // zeroGradient placeholder on a coupled patch, so its boundary value is neither coupled nor
        // rotated -- but it is wrong on a RESTART, and wrong in a way that hides.
        //
        // A written phi DOES carry the interface flux: OpenFOAM writes it per patch (pipeCyclic's
        // side1/side2 hold 400 and 250 values), and createFields copies those straight into
        // phi.boundary, past the patch-field layer that would otherwise have replaced them. That flux is
        // the CONSERVATIVE one the previous run ended with; fvc::flux(U) is not, because U alone does not
        // know what the pressure correction did. Measured on pipeCyclic restarted from OpenFOAM's own
        // converged state, the two differ by 4.98e-03 relative on side1.
        //
        // It matters most for the residual ORACLE this port leans on. Restart brae at OpenFOAM's
        // converged fields and every internal face gets OpenFOAM's phi (read from file) while every
        // interface face got a rebuilt one -- so the momentum residual came out exactly zero in the
        // interior and enormous on the interface cells, which reads exactly like an interface
        // discretisation defect and is not one.
        if (hasAMI_)
        {
            bool loaded = false;
            if (phiWasRead)
            {
                std::vector<scalar> ifPhi;
                ifPhi.reserve(ami_.n);
                for (const auto& r : amiRuns_)
                {
                    const label pi = r.first;
                    if (pi < 0 || (std::size_t)pi >= phi.boundary.size()
                        || (label)phi.boundary[pi].size() != r.second) { ifPhi.clear(); break; }
                    ifPhi.insert(ifPhi.end(), phi.boundary[pi].begin(), phi.boundary[pi].end());
                }
                if ((label)ifPhi.size() == ami_.n)
                {
                    ami_.phi.copyFrom(ifPhi);
                    loaded = true;
                    std::printf("  AMI: interface flux restored from the written phi (%d faces); "
                                "fvc::flux(U) would not be the conservative one\n", (int)ami_.n);
                }
            }
            if (!loaded) interfaceFlux(ami_, Uk_[0], Uk_[1], Uk_[2]);   // cold start: rebuild from U_init
        // Stage harness: fvc::flux(U) ON the interface, from the U just read off disk. The one interface
        // quantity that can be compared against OpenFOAM with NO circularity -- restart both codes from
        // the same time directory and this is the same function of the same input on both sides, before
        // either has taken a step. Everything downstream (HbyA, p) is contaminated by the other's answer.
        }
        if (hasAMI_ && stageDumpActive()) stageDump("stage_ami_fluxU_init", ami_.phi);
        {
            std::vector<scalar> o;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                o.insert(o.end(), phi.boundary[pi].begin(), phi.boundary[pi].end());
            }
            phiBnd_.copyFrom(o);
        }
        std::vector<scalar> nv(nC_, 0.0);
        if (ctl_.turbulent)
        {
            nv = nut->internal;   // nut is the ONLY turbulence field for pure LES (Smagorinsky); read it for every model.
            if (!ctl_.les)        // RAS/DES transport scalars: pure-LES Smagorinsky is ALGEBRAIC (no k/epsilon/omega/nuTilda, no wall distance).
            {
                dbK_   = buildDeviceBoundary(*k, fvp, g);        // SA: the "k" slot holds nuTilda
                dk_.copyFrom(k->internal);
                if (!ctl_.sa)
                {
                    dbEps_ = buildDeviceBoundary(*eps, fvp, g);
                    de_.copyFrom(eps->internal);   // 2nd scalar (omega/eps); SA has none
                }
                if (ctl_.sst || ctl_.sa)
                {
                    std::vector<vector> wallOrigin;                              // nearest wall-face centre (IDDES wall-normal source)
                    // ZDES2020 shielding reads OF's wallDist::n(), the nearest wall face's OUTWARD unit
                    // normal, and it must come from the SAME method that produced y -- the wave carries
                    // the seed face's normal, exactDistance takes the normal at the hit point. A case
                    // that asks for it writes `nRequired yes` under wallDist (NACA4412 does).
                    const bool needWallN = ctl_.sa && ctl_.saCoeffs.zdes;
                    std::vector<vector> wallNormal;
                    // `delta maxDeltaxyz`: OF's face-normal hmax, computed once (static geometry) and
                    // shared by the sub-grid model, the DES length scale and DEShybrid.
                    if (ctl_.lesDeltaMax) lesDelta_.copyFrom(cellMaxDeltaFaceNormal(m, g, ctl_.lesDeltaCoeff));
                    ctorMark("pre-wallDist");
                    // OF wallDist: `meshWave` (the default) or `exactDistance`. IDDES additionally needs
                    // the wall ORIGIN, which only the wave carries, so exactDistance is not offered there.
                    if (ctl_.exactWallDist && !ctl_.iddes)
                        y_.copyFrom(exactCellWallDist(m, g, fvp, needWallN ? &wallNormal : nullptr));
                    else
                        y_.copyFrom(cellWallDist(m, g, fvp, ctl_.iddes ? &wallOrigin : nullptr,
                                                 needWallN ? &wallNormal : nullptr));   // wall distance (SST F1/F2/F3; SA dTilda)
                    if (needWallN && wallNormal.size() == static_cast<std::size_t>(nC_))
                    {
                        std::vector<scalar> packed(3*static_cast<std::size_t>(nC_));
                        for (label c = 0; c < nC_; ++c)
                        {
                            packed[c]            = wallNormal[c].x;
                            packed[nC_ + c]      = wallNormal[c].y;
                            packed[2*nC_ + c]    = wallNormal[c].z;
                        }
                        wallN_.copyFrom(packed);
                    }
                    if (ctl_.iddes)   // IDDES filter widths (maxDeltaxyz + wall-normal spacing), uploaded once at setup
                    {
                        const std::vector<scalar> hmax = cellMaxDeltaXYZ(m);
                        hmax_.copyFrom(hmax);
                        hwn_.copyFrom(cellWallNormalSpacing(m, g, wallOrigin, hmax));
                    }
                }
                if (ctl_.lm && ReThetat && gammaInt)   // kOmegaSSTLM transition fields
                {
                    dbReThetat_ = buildDeviceBoundary(*ReThetat, fvp, g);
                    ReThetat_.copyFrom(ReThetat->internal);
                    dbGammaInt_ = buildDeviceBoundary(*gammaInt, fvp, g);
                    gammaInt_.copyFrom(gammaInt->internal);
                    gammaIntEff_.copyFrom(std::vector<scalar>(nC_, 0.0));   // OF kOmegaSSTLM ctor: gammaIntEff_ = Zero (no Pk on iter 1, before correctReThetatGammaInt updates it)
                }
            }
        }
        dnut_.copyFrom(nv);
        nuConst_.copyFrom(std::vector<scalar>(nC_, ctl_.nu));
        zeroSrc_.copyFrom(std::vector<scalar>(nC_, 0.0));
        zeroBndU_.copyFrom(std::vector<scalar>(dbU_.n, 0.0));
        ones_.copyFrom(std::vector<scalar>(nC_, 1.0));            // unit vector for the OF normFactor
        if (stageDumpActive() && y_.size()) stageDump("stage_y", y_);   // wall distance (SST F1/F2)
        validateTurbulence();                                    // OF turbulenceModel ctor bound() + simpleFoam.C:92 validate()
    }

    void DeviceSimpleSolver::addCodedBC(const std::string& name, const std::string& code, int offset, int count, int target, bool mixed)
    {
        SolverCodedBC cbc;
        const bool vec = (target == 0);   // U is vector; p/k/second are scalar
        cbc.kernel = mixed ? (vec ? compileCodedMixedVectorBc(name, code) : compileCodedMixedScalarBc(name, code))
                           : (vec ? compileCodedVectorBc(name, code)      : compileCodedScalarBc(name, code));
        cbc.offset = offset;
        cbc.count  = count;
        cbc.target = target;
        cbc.mixed  = mixed;
        codedBCs_.push_back(std::move(cbc));
    }

    void DeviceSimpleSolver::revalidateAfterThermo(const std::vector<scalar>* rhoBndSeed)
    {
        if (!compressible_) return;
        // Same three steps the outer iteration runs, in the same order, so this is OF's validate() with
        // the state OF has at that point -- not a special startup path that could drift from the real one.
        deviceThermoTBoundary(dbP_, dp_, dbHe_, th_.he, tc_, &th_.T, &TFixMask_, &TFixBnd_, TBnd_);
        deviceThermoRhoBoundary(dbP_, dp_, TBnd_, tc_, rhoBnd_);
        // Seed the solver rho field's BOUNDARY half, and its prevIter, so the first rho.relax() blends
        // toward the same state OF's does. Seeding prevIter lazily on first use instead makes the first
        // relaxation a no-op and leaves the unrelaxed density in place for the whole run.
        //
        // OF: `volScalarField rho(IOobject("rho", ..., READ_IF_PRESENT, AUTO_WRITE), thermo.rho())`.
        // Present on disk -> the stored field wins, boundary included; absent -> thermo.rho().
        if (rhoBndSeed && rhoBndSeed->size() == static_cast<std::size_t>(dbP_.n))
        {
            rhoBndP_.copyFrom(*rhoBndSeed);
            rhoBndPPrev_.copyFrom(*rhoBndSeed);
        }
        else
        {
            deviceCopy(rhoBndP_, rhoBnd_);
            deviceCopy(rhoBndPPrev_, rhoBnd_);
        }
        // Same density rule as the per-iteration update in rhoSimpleStep (see the long note there): OF's
        // flowRateInletVelocity looks up the REGISTERED rho, which createFields.H has already built by the
        // time U's patches evaluate, so seed avgU from the solver rho -- rhoBndP_, just set above.
        for (std::size_t k = 0; k < frPatches_.size(); ++k)
        {
            const scalar sumRhoA = deviceDot(rhoBndP_, frMagSf_[k]);
            if (sumRhoA <= 0.0) continue;
            deviceUpdateFlowRateInlet(dbU_, frMagSf_[k], -frPatches_[k].mdot / sumRhoA, frNx_, frNy_, frNz_);
        }
        validateTurbulence();
    }

    void DeviceSimpleSolver::validateTurbulence()
    {
        if (!ctl_.turbulent) return;
        const DeviceMesh& dm = dm_;
        if (ctl_.les)   // pure LES Smagorinsky: nut is algebraic (no bound(k)/bound(omega)); seed it from the initial U.
        {
            DeviceBuffer<scalar> gradU;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            if (ctl_.wale) deviceWaleNut(nC_, gradU, dm.V, ctl_.waleCoeffs, dnut_, &lesDelta_);
            else           deviceSmagorinskyNut(nC_, gradU, dm.V, ctl_.smagCoeffs, dnut_, &lesDelta_);   // nut = Ck*delta*sqrt(k_sgs)
            deviceAlphat(th_, dnut_, tc_);                                    // EddyDiffusivity::correctNut() tail
            return;
        }
        deviceBoundField(dm, dk_, 1e-15);                              // bound(k_, kMin_)  [SA: bound(nuTilda_, 0)]
        if (!ctl_.sa) deviceBoundField(dm, de_, 1e-15);               // bound(omega_|epsilon_, ...Min_)
        // validate() -> correctNut(): recompute internal nut from the bounded fields + the initial U.
        if (ctl_.sa)
        {
            deviceNutSA(dk_, ctl_.nu, ctl_.saCoeffs.Cv1, dnut_);     // nut = nuTilda*fv1(nuTilda)
        }
        else if (ctl_.sst)
        {
            DeviceBuffer<scalar> gradU;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU,
                        hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            if (ctl_.gradULimitK > 0.0) deviceCellLimitGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, ctl_.gradULimitK,
                                                          hasCyclic_ ? &cyc_ : nullptr, hasAMI_ ? &ami_ : nullptr);   // OF grad(U) cellLimited (validate correctNut)
            DeviceBuffer<scalar> S2;
            deviceS2(gradU, nC_, S2);
            DeviceBuffer<scalar> F2;
            deviceF2(dk_, de_, y_, ctl_.nu, ctl_.ksstCoeffs, F2);
            deviceNutSST(dk_, de_, F2, S2, ctl_.ksstCoeffs, dnut_);   // nut = a1*k/max(a1*omega, b1*F2*sqrt(S2))
        }
        else
        {
            deviceNut(dk_, de_, dnut_, ctl_.keCoeffs);               // nut = Cmu*k^2/eps
        }
        // The tail of OF's correctNut() for a COMPRESSIBLE model: EddyDiffusivity::correctNut() sets
        // alphat_ = rho*nut/Prt right after nut, so alphat is already turbulent before the FIRST energy
        // equation -- rhoSimpleFoam.C:64 calls turbulence->validate() before the time loop.
        //
        // brae used to update alphat only at the END of the outer iteration, so iteration 1 solved the
        // energy equation with alphat = 0, i.e. alphaEff = laminar alpha alone. On a slow case that is a
        // small error that the next iteration repairs. On NACA0012 the turbulent part is 65x the laminar
        // one (alphaEff 1.67e-03 vs 3.59e-05): the first energy solve had 1/46th of the real thermal
        // diffusivity, T fell 68 K in the first cell off the aerofoil, and the run diverged from there.
        deviceAlphat(th_, dnut_, tc_);
    }

    void DeviceSimpleSolver::setMaxwellSigma(const std::vector<GeometricField<scalar>>& sigma,
                                            const std::vector<FvPatch>& fvp,
                                            const FvGeometry& g)
    {
        if (sigma.size() != 6)
            throw std::runtime_error("brae: setMaxwellSigma needs the six components of sigma.");
        for (int k = 0; k < 6; ++k)
        {
            sig_[k].copyFrom(sigma[k].internal);
            dbSig_[k] = buildDeviceBoundary(sigma[k], fvp, g);
        }
    }

    // OF laminarModels::Maxwell::correct():
    //
    //   P = twoSymm(sigma & gradU) - nuM/lambda*twoSymm(gradU)
    //   ddt(sigma) + div(phi, sigma) + Sp(1/lambda, sigma) == P
    //
    // Six scalar transports sharing one convection matrix and coupled only through the explicit P --
    // OF's fvSymmTensorMatrix is segregated in exactly the same way, so this is the same solve, not an
    // approximation of it. No laplacian: the stress equation has no diffusion term at all.
    void DeviceSimpleSolver::correctMaxwell()
    {
        const DeviceMesh& dm = dm_;
        if (!sig_[0].size())
            throw std::runtime_error(
                "brae: laminar model Maxwell selected but the sigma field was never handed to the solver. "
                "The viscoelastic stress is a TRANSPORTED field with its own initial condition; running "
                "without it would silently solve a Newtonian fluid.");

        DeviceBuffer<scalar> gradU;
        deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
        if (ctl_.gradULimitK > 0.0)
            deviceCellLimitGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, ctl_.gradULimitK,
                                 hasCyclic_ ? &cyc_ : nullptr, hasAMI_ ? &ami_ : nullptr);

        const scalar rLambda = scalar(1)/ctl_.maxwellLambda;
        DeviceBuffer<scalar> P[6];
        deviceMaxwellP(nC_, sig_, gradU, ctl_.maxwellNuM, rLambda, P);

        // div(phi,sigma) is `Gauss vanAlbada` in both tutorials: the NVDTVD ratio built from
        // magSqr(sigma) (OF's limitFuncs::magSqr for a non-scalar type), so ONE limiter per face serves
        // all six components -- which is also why the limiter field is built once, out here.
        DeviceBuffer<scalar> magSig, mgx, mgy, mgz, magBnd;
        const bool vanAlbada = ctl_.divSigmaVanAlbada;
        if (vanAlbada)
        {
            deviceSymmMagSqr(nC_, sig_, magSig);
            deviceBCValue(dbExtrap_, magSig, magBnd);
            deviceGaussGrad(dm, magSig, magBnd, mgx, mgy, mgz);
        }
        const DdtCoeffs ddtc = ddtCoeffs(ddtScheme_, deltaT_, deltaT0_, ocCoeff_, cnWarm_);
        DeviceBuffer<scalar> zeroD(std::vector<scalar>(static_cast<std::size_t>(nC_), scalar(0)));
        DeviceBuffer<scalar> divU;
        deviceDiv(dm, phiInt_, phiBnd_, divU);
        for (int k = 0; k < 6; ++k)
        {
            const ScalarDdt sDdt{ ddtc, &sigOld_[k],
                                  sigOld2_[k].size() ? &sigOld2_[k] : nullptr,
                                  ddtc.cn ? &sigddt0_[k] : nullptr };
            deviceSolveScalarTransport(
                dm, dbSig_[k], sig_[k], "sigma", zeroD, phiInt_, phiBnd_, divU,
                /*bounded=*/false, /*limited=*/vanAlbada, /*linearUpwind=*/false, /*nonOrth=*/false,
                /*twoByk=*/scalar(0),   // 0 selects vanAlbada in the shared limiter kernel
                ctl_.relaxSigma, ctl_.keTol(), ctl_.keRelTol(), ctl_.bicgCheckEvery, /*useGS=*/ctl_.gsSigma,
                [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src)
                { deviceMaxwellReaction(dm.V, rLambda, P[k], diag, src); },
                nullptr, nullptr, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr, sDdt,
                nullptr, scalar(0), /*boundPositive=*/false,   // sigma is NOT positive-definite: it has negative components
                /*fvoSetMask=*/nullptr, /*fvoSetVal=*/nullptr,
                // the shared magSqr(sigma) limiter -- one per face for all six components (OF limitFuncs)
                vanAlbada ? &magSig : nullptr, vanAlbada ? &mgx : nullptr,
                vanAlbada ? &mgy : nullptr, vanAlbada ? &mgz : nullptr);
        }
    }

    void DeviceSimpleSolver::correctTurbulence()
    {
        const DeviceMesh& dm = dm_;
        // OF calls turbulence->correct() every outer corrector whatever the model is, and for a laminar
        // Maxwell fluid THAT is the stress transport -- there is no nut and no k. It runs before the
        // turbulenceOn gate below, which is a RAS/LES switch and has nothing to say about a material law.
        if (ctl_.maxwell) { correctMaxwell(); return; }
        clearTurbulenceReport();   // OF-style report: collect this step's turbulence solves (Solving for k/omega/...)
        // OF RASModel `turbulence off` -> correct() returns before anything is solved or updated, so k,
        // epsilon/omega and nut stay at the values they were read with and momentum keeps using them
        // (kEpsilon.C:216, and every other model does the same). The report stays empty, which is also
        // what OF prints: no "Solving for k" lines.
        if (!ctl_.turbulenceOn) return;
        // TURBULENT INLETS, refreshed HERE and not in the momentum predictor.
        //
        // OF's rule is structural, not per-case: a boundary condition updates when ITS OWN equation is
        // assembled. `turbulentIntensityKineticEnergyInlet` and `turbulentMixingLengthDissipationRateInlet`
        // are k/epsilon BCs, so their updateCoeffs() fires inside kEqn/epsEqn -- which live in
        // kEpsilon::correct() (kEpsilon.C:252,273), called LAST in the outer iteration
        // (rhoSimpleFoam.C:86, after UEqn/EEqn/pEqn). Momentum therefore always sees the PREVIOUS
        // iteration's k, epsilon and nut boundary values.
        //
        // brae refreshed them at the top of solveMomentumPredictor, one phase early. Measured cost on
        // squareBend at iteration 1: the inlet `calculated` nut is built as Cmu*k_b^2/eps_b, so an inlet
        // k already recomputed from THIS iteration's U (1026, = 1.5*(0.05*523.087)^2) made the inlet
        // boundary diffusivity 3.36e-02 against OF's 2.14e-04 -- 157x. That fed the inlet's momentum
        // internalCoeffs (157x), hence Upred (4.76e-02), hence HbyA, phiHbyA, phid, p and U. The 22400
        // WALL faces and the outlet matched OF exactly throughout; only the 400 inlet faces were wrong.
        //
        // A3 fixed these inlets being FROZEN at set-up. This is that fix landing one phase too early.
        if (hasTurbInlet_)
        {
            deviceUpdateTurbulentInletK(dbU_, tiMask_, tiIntensity_, dbK_);
            deviceUpdateTurbulentInletSecond(dbK_, mlMask_, mlLength_,
                                             ctl_.sst ? ctl_.ksstCoeffs.betaStar : ctl_.keCoeffs.Cmu, dbEps_);
        }
        if (ctl_.les)   // pure LES Smagorinsky: algebraic sub-grid nut = Ck*delta*sqrt(k_sgs) from the current U. No
        {              // transport solve (report stays empty -> no "Solving for k/omega" lines), so no ddt(k/omega) either.
            DeviceBuffer<scalar> gradU;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            if (ctl_.wale) deviceWaleNut(nC_, gradU, dm.V, ctl_.waleCoeffs, dnut_, &lesDelta_);
            else           deviceSmagorinskyNut(nC_, gradU, dm.V, ctl_.smagCoeffs, dnut_, &lesDelta_);
            return;
        }
        if (ctl_.turbulent)
        {
            // transient URANS fvm::ddt(k/eps/omega/nuTilda): a per-scalar bundle (steady -> active==false -> no-op, so
            // the steady SIMPLE turbulence stays byte-for-byte). dk_=k|nuTilda (kOld_), de_=epsilon|omega (e2Old_).
            const DdtCoeffs ddtc = ddtCoeffs(ddtScheme_, deltaT_, deltaT0_, ocCoeff_, cnWarm_);
            const ScalarDdt kDdt{ ddtc, &kOld_,  kOld2_.size()  ? &kOld2_  : nullptr, ddtc.cn ? &kddt0_ : nullptr };
            const ScalarDdt sDdt{ ddtc, &e2Old_, e2Old2_.size() ? &e2Old2_ : nullptr, ddtc.cn ? &e2ddt0_ : nullptr };
            const ScalarDdt reDdt{ ddtc, &ReThetatOld_, ReThetatOld2_.size() ? &ReThetatOld2_ : nullptr, ddtc.cn ? &ReThetatddt0_ : nullptr };   // kOmegaSSTLM
            const ScalarDdt giDdt{ ddtc, &gammaIntOld_, gammaIntOld2_.size() ? &gammaIntOld2_ : nullptr, ddtc.cn ? &gammaIntddt0_ : nullptr };
            if (ctl_.sa)    // one-equation: dk_ slot holds nuTilda; relaxK/limitedK/twoBykK carry the nuTilda settings
                deviceSpalartAllmarasCorrect(dm, dbU_, dbK_, Uk_[0], Uk_[1], Uk_[2], dk_, dnut_, y_, phiInt_, phiBnd_,
                                             ctl_.nu, ctl_.kRelax(), ctl_.keTol(), ctl_.boundedK, ctl_.limitedK, ctl_.twoBykK,
                                             ctl_.saCoeffs, ctl_.keRelTol(), ctl_.bicgCheckEvery, ctl_.luK, ctl_.nonOrth,
                                             ctl_.gsK, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr, kDdt,   // nuTilda ddt (kOld_)
                                             ctl_.des, ctl_.iddes, ctl_.iddes ? &hmax_ : nullptr, ctl_.iddes ? &hwn_ : nullptr, &lesDelta_,
                                             wallN_.size() ? &wallN_ : nullptr);   // SA-DDES/IDDES length-scale limiter (no-op for plain SA-RANS)
            else if (ctl_.sst)   // de_ slot holds omega; relaxEps/limitedEps/twoBykEps carry the omega-equation settings
            {
                deviceKOmegaSSTCorrect(dm, wall_, dbEps_, dbK_, dbU_, Uk_[0], Uk_[1], Uk_[2], dk_, de_, dnut_, y_,
                                       phiInt_, phiBnd_, ctl_.nu, ctl_.epsRelax(), ctl_.kRelax(), ctl_.keTol(), ctl_.boundedK, ctl_.boundedEps,
                                       ctl_.limitedK, ctl_.limitedEps, ctl_.twoBykK, ctl_.twoBykEps, ctl_.ksstCoeffs, ctl_.keRelTol(), ctl_.bicgCheckEvery,
                                       ctl_.luK, ctl_.luEps, ctl_.nonOrth, ctl_.gradULimitK, ctl_.gsK, ctl_.gsEps, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr,
                                       ctl_.lm ? gammaIntEff_.data() : nullptr,   // LM: scale k Pk/epsilonByk by the lagged gammaIntEff
                                       static_cast<int>(ctl_.nutWall),   // near-wall G0 uses the same BC-chosen wall nut as the momentum shear
                                       ctl_.atmZ0, ctl_.atmBoundNut,   // atmNutkWallFunction roughness for the near-wall G0
                                       kDdt, sDdt, ctl_.des, ctl_.iddes, ctl_.iddes ? &hmax_ : nullptr, ctl_.iddes ? &hwn_ : nullptr,   // ddt(k)/ddt(omega) + kOmegaSSTDDES/IDDES DES limiter (no-op for RANS)
                                       compressible_ ? &th_.rho : nullptr,   // rho-weight every k/omega term, as OF's alpha*rho* does
                                       compressible_ ? &th_.mu : nullptr,   // laminar dynamic mu; F1/F2 get nu = mu/rho from it
                                       (compressible_ && wfNu_.size()) ? &wfNu_ : nullptr,   // nu at WALL faces for omegaWallFunction/G0
                                       compressible_ ? &rhoBnd_ : nullptr,   // rho_b: divU is div of the VOLUMETRIC flux, not the mass flux
                                       // Order matters: with an EXTRAPOLATED nut_b this made omega worse
                                       // (3.1e-5 -> 3.2e-3). Enabled only now that nut_b is evaluated.
                                       nutBndAll_.size() ? &nutBndAll_ : nullptr,
                                       compressible_ ? &muBnd_ : nullptr,
                                       ctl_.gradKLULimitK,   // linearUpwind's named grad(k|eps), not gradSchemes grad(k)
                                       // fvOptions scalarFixedValueConstraint (k / omega)
                                       fixScaK_ ? &fixScaMask_ : nullptr, fixScaK_ ? &fixScaKVal_ : nullptr,
                                       fixScaE_ ? &fixScaMask_ : nullptr, fixScaE_ ? &fixScaEVal_ : nullptr,   // grad(k)/grad(omega) cellLimited (C2)
                                       &lesDelta_);   // the case's `delta` for the DDES length scale (empty -> cubeRootVol)
                if (ctl_.lm)   // Langtry-Menter: transport ReThetat + gammaInt, update gammaIntEff for next iter
                    deviceKOmegaSSTLMCorrect(dm, dbU_, dbReThetat_, dbGammaInt_, Uk_[0], Uk_[1], Uk_[2], dk_, de_, dnut_, y_,
                                             ReThetat_, gammaInt_, gammaIntEff_, phiInt_, phiBnd_, ctl_.nu, ctl_.epsRelax(),
                                             ctl_.keTol(), ctl_.keRelTol(), ctl_.bicgCheckEvery, ctl_.bounded, ctl_.nonOrth,
                                             ctl_.gsEps, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr, reDdt, giDdt);   // LM transition ddt
            }
            else
                deviceKEpsilonCorrect(dm, wall_, dbEps_, dbK_, dbU_, Uk_[0], Uk_[1], Uk_[2], dk_, de_, dnut_,
                                      phiInt_, phiBnd_, ctl_.nu, ctl_.epsRelax(), ctl_.kRelax(), ctl_.keTol(), ctl_.boundedK, ctl_.boundedEps,
                                      ctl_.limitedK, ctl_.limitedEps, ctl_.twoBykK, ctl_.twoBykEps, ctl_.keCoeffs, ctl_.keRelTol(), ctl_.bicgCheckEvery,
                                      ctl_.luK, ctl_.luEps, ctl_.nonOrth, ctl_.gsK, ctl_.gsEps, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr,
                                      static_cast<int>(ctl_.nutWall),   // near-wall G0 uses the same BC-chosen wall nut as the momentum shear
                                      ctl_.atmZ0, ctl_.atmBoundNut,   // atmNutkWallFunction roughness for the near-wall G0
                                      kDdt, sDdt,   // transient fvm::ddt(k)/ddt(epsilon)  (sDdt = the 2nd-scalar bundle)
                                      compressible_ ? &th_.rho : nullptr,   // alpha*rho on every k/eps RHS term + diffusivity
                                      compressible_ ? &th_.mu : nullptr,    // laminar dynamic mu
                                      compressible_ ? &rhoBnd_ : nullptr,   // rho_b: divU from the VOLUMETRIC flux
                                      (compressible_ && wfNu_.size()) ? &wfNu_ : nullptr,   // nu at wall faces (OF nu(patchi))
                                      nutBndAll_.size() ? &nutBndAll_ : nullptr,   // nut_b -> DkEff/DepsEff(patchi)
                                      compressible_ ? &muBnd_ : nullptr,
                                      ctl_.gradKLULimitK,   // linearUpwind's named grad(k|eps)
                                      ctl_.gradULimitK,     // kEpsilon::correct()'s own fvc::grad(U) -> gradSchemes grad(U)
                                      // fvOptions scalarFixedValueConstraint (OF eqn.setValues per field)
                                      fixScaK_ ? &fixScaMask_ : nullptr, fixScaK_ ? &fixScaKVal_ : nullptr,
                                      fixScaE_ ? &fixScaMask_ : nullptr, fixScaE_ ? &fixScaEVal_ : nullptr);
        }
    }

    void DeviceSimpleSolver::solveMomentumPredictor(DeviceSimpleResidual& res)
    {
        // The flux this outer iteration INHERITS. OF's dumpPEqn writes the same quantity at the top of
        // pEqn.H, and the momentum predictor does not touch phi, so the two are directly comparable.
        if (stageDumpActive() && stageDumpFirstOnly("phiIn")) stageDump("stage_phiIn", phiInt_);
        if (stageDumpActive() && stageDumpFirstOnly("phiInB")) stageDump("stage_phiInB", phiBnd_);
        const DeviceMesh& dm = dm_;
        const scalar tol = ctl_.uTol();
        // Transient fvm::ddt(U): the ONE term steady SIMPLE lacks. steadyState (SIMPLE) -> ddtc.active==false -> the two
        // deviceFvmDdt* calls below are exact no-ops (this method stays byte-for-byte SIMPLE). pimpleStep() sets the
        // scheme + rotates the old-time fields, so the transient (PIMPLE) path folds ddt into the SAME predictor.
        const DdtCoeffs ddtc = ddtCoeffs(ddtScheme_, deltaT_, deltaT0_, ocCoeff_, cnWarm_);
        // nonOrthLimitP (the pressure non-orth limiter) is declared in step() -- it is read only by the pressure phase.
        const scalar nonOrthRelaxU = std::getenv("BRAE_NONORTH_RELAX_U") ? std::atof(std::getenv("BRAE_NONORTH_RELAX_U")) : 1.0;

        // inletOutlet BCs: resolve each IO face to fixedValue|zeroGradient from the PREVIOUS step's boundary flux
        // (OF updateCoeffs uses the prior corrector's phi). No-op when a boundary has no inletOutlet faces.
        deviceUpdateInletOutlet(dbU_, phiBnd_);
        deviceUpdateInletOutlet(dbP_, phiBnd_);
        // The ENERGY boundary too. dbHe_ inherits T's BC types, so an `inletOutlet` T outlet -- which
        // every stock rhoSimpleFoam tutorial has -- arrives here masked as inletOutlet and MUST be
        // resolved per iteration like the others. It was omitted: the mask was set at build time and
        // never consulted, so the outlet enthalpy stayed clamped at fixedValue(inletValue) with full
        // convection AND laplacian coupling, where OF switches to zeroGradient on outflow. It propagated
        // straight into rhoBnd_ (built from dbHe_), hence the outlet density, the outlet mass flux and
        // the pressure field.
        if (compressible_ && dbHe_.n) deviceUpdateInletOutlet(dbHe_, phiBnd_);
        // NOTE: the turbulent-inlet refresh does NOT belong here. It moved into correctTurbulence(),
        // which is where OF updates it -- see the comment there.
        if (ctl_.turbulent && !ctl_.les)   // pure LES has no k/epsilon/omega boundaries (algebraic nut)
        {
            deviceUpdateInletOutlet(dbK_, phiBnd_);
            deviceUpdateInletOutlet(dbEps_, phiBnd_);
            if (ctl_.lm)
            {
                deviceUpdateInletOutlet(dbReThetat_, phiBnd_);
                deviceUpdateInletOutlet(dbGammaInt_, phiBnd_);
            }
        }
        // mixed freestreamVelocity/Pressure: recompute the per-face valueFraction from the (lagged) flow angle.
        if (hasMixed_) deviceUpdateMixedFreestream(dbU_, dbP_, phiBnd_, Uk_[0], Uk_[1], Uk_[2],
                                                  compressible_ ? &rhoBnd_ : nullptr);   // phiBnd_ is a MASS flux
        if (hasPiov_)  deviceUpdatePressureInletOutletVelocity(dbU_, phiBnd_, Uk_[0], Uk_[1], Uk_[2]);
        if (hasSym_)   deviceUpdateSymmetry(dbU_, Uk_[0], Uk_[1], Uk_[2]);
        if (hasWedge_) deviceUpdateWedge(dbU_, Uk_[0], Uk_[1], Uk_[2]);   // axisymmetric: the rotated cell velocity
        // totalPressure p: recompute refValue = p0 - 0.5*neg(phi)|U_b|^2 from the boundary velocity (deviceBCValue
        // reflects the just-updated U BCs above, e.g. pressureInletOutletVelocity) + the boundary flux.
        if (hasTotalP_)
        {
            // p0(t) BEFORE the total-pressure update, matching OF's updateCoeffs order: p0 is sampled
            // at the current time, then p_b = p0 - 0.5*neg(phi)*magSqr(U) is formed from it.
            if (!tvP0_.empty())
            {
                std::vector<scalar> p0h = dbP_.p0.host();
                for (const auto& e : tvP0_)
                {
                    const scalar v = e.f.value(time_);
                    for (label i = 0; i < e.count; ++i)
                    {
                        const label f = e.start + i;
                        if (f >= 0 && f < (label)p0h.size()) p0h[f] = v;
                    }
                }
                dbP_.p0.copyFrom(p0h);
            }
            // fanPressure: the fan's operating point depends on the flow it is currently passing, so p0
            // has to be re-derived from THIS step's boundary flux before the total-pressure update runs.
            // OF does the same inside fanPressureFvPatchScalarField::updateCoeffs, which then defers to
            // totalPressure with the shifted p0.
            if (!fanP0_.empty())
            {
                const std::vector<scalar> phih = phiBnd_.host();
                std::vector<scalar> p0h = dbP_.p0.host();
                for (const auto& e : fanP0_)
                {
                    scalar sum = 0;
                    for (label i = 0; i < e.count; ++i)
                    {
                        const label f = e.start + i;
                        if (f >= 0 && f < (label)phih.size()) sum += phih[f];
                    }
                    const scalar v = e.p0 - e.dir * e.dp(std::max(e.dir * sum, scalar(0)));
                    for (label i = 0; i < e.count; ++i)
                    {
                        const label f = e.start + i;
                        if (f >= 0 && f < (label)p0h.size()) p0h[f] = v;
                    }
                }
                dbP_.p0.copyFrom(p0h);
            }
            DeviceBuffer<scalar> ubx, uby, ubz;
            deviceBCValue(dbU_.comp[0], Uk_[0], ubx);
            deviceBCValue(dbU_.comp[1], Uk_[1], uby);
            deviceBCValue(dbU_.comp[2], Uk_[2], ubz);
            deviceUpdateTotalPressure(dbP_, phiBnd_, ubx, uby, ubz,
                                      compressible_ ? &rhoBnd_ : nullptr);   // p in Pa -> rho-weighted dynamic head
        }
        // fixedMean: the face values ARE the adjacent cell values, shifted (or scaled) so their
        // area-weighted mean hits the prescribed one. That is a patch-wide reduction, so the small
        // per-boundary-face gather comes back to the host and the sum is done there -- the whole cell
        // field never moves.
        if (!fixedMean_.empty())
        {
            DeviceBuffer<scalar> pin;
            deviceGatherPatchInternal(dbP_, dp_, pin);
            const std::vector<scalar> psi = pin.host();
            const std::vector<scalar> aSf = dbP_.magSf.host();
            std::vector<scalar> rv = dbP_.refValue.host();
            for (const auto& e : fixedMean_)
            {
                if (e.start < 0 || e.start + e.count > (label)psi.size()) continue;
                applyFixedMean(e.meanValue, aSf.data() + e.start, psi.data() + e.start, e.count,
                               rv.data() + e.start);
            }
            dbP_.refValue.copyFrom(rv);
        }
        // NVRTC device-coded BCs (codedFixedValue): run each compiled snippet over its patch faces, overwriting the
        // target field's refValue on the device from position/time/adjacent-cell value. Same updateCoeffs point as the
        // BCs above; the p/k/second boundaries set here persist into the pressure + turbulence phases of this corrector.
        for (const SolverCodedBC& cbc : codedBCs_)
        {
            if (cbc.target == 0)         // U (vector)
            {
                if (cbc.mixed)
                    launchCodedMixedVectorBc(cbc.kernel, cbc.offset, cbc.count, time_, bndCfX_, bndCfY_, bndCfZ_,
                                             Uk_[0], Uk_[1], Uk_[2], dbU_.comp[0].faceCell,
                                             dbU_.comp[0].refValue, dbU_.comp[1].refValue, dbU_.comp[2].refValue,
                                             dbU_.comp[0].valueFraction, dbU_.comp[1].valueFraction, dbU_.comp[2].valueFraction);
                else
                    launchCodedVectorBc(cbc.kernel, cbc.offset, cbc.count, time_, bndCfX_, bndCfY_, bndCfZ_,
                                        Uk_[0], Uk_[1], Uk_[2], dbU_.comp[0].faceCell,
                                        dbU_.comp[0].refValue, dbU_.comp[1].refValue, dbU_.comp[2].refValue);
            }
            else                         // scalar: p (1), k/nuTilda (2), second omega|epsilon (3)
            {
                DeviceBoundary& db = (cbc.target == 1) ? dbP_ : (cbc.target == 2) ? dbK_ : dbEps_;
                const DeviceBuffer<scalar>& fld = (cbc.target == 1) ? dp_ : (cbc.target == 2) ? dk_ : de_;
                if (cbc.mixed)
                    launchCodedMixedScalarBc(cbc.kernel, cbc.offset, cbc.count, time_, bndCfX_, bndCfY_, bndCfZ_, fld, db.faceCell, db.refValue, db.valueFraction);
                else
                    launchCodedScalarBc(cbc.kernel, cbc.offset, cbc.count, time_, bndCfX_, bndCfY_, bndCfZ_, fld, db.faceCell, db.refValue);
            }
        }

        DeviceBuffer<scalar> nuEff;
        if (compressible_)
            // muEff = mu + rho*nut. OF assembles the stress as rho*nuEff() (linearViscousStress.C), and for a
            // compressible model nuEff() = nut() + mu/rho, so rho*nuEff() = rho*nut + mu = mut + mu
            // (CompressibleTurbulenceModel::muEff). dnut_ stays KINEMATIC -- the SST solves for it, every wall
            // function compares it against nu, and alphat = rho*nut/Prt reads it -- so the rho belongs here.
            deviceHadamard(nuEff, th_.rho, dnut_);
        else
            deviceCopy(nuEff, dnut_);
        deviceAxpy(1.0, nuConst_, nuEff);   // + nu (incompressible) / + mu (compressible, nuConst_ = th_.mu)
        // Maxwell: the implicit laplacian coefficient is nu0 = nu + nuM (Maxwell.H:98). nut is identically
        // zero here -- it is a laminar model -- so this is the whole of the difference.
        if (ctl_.maxwell)
        {
            DeviceBuffer<scalar> ones1(std::vector<scalar>(static_cast<std::size_t>(nC_), scalar(1)));
            deviceAxpy(ctl_.maxwellNuM, ones1, nuEff);
        }
        // Stage harness: the ASSEMBLED momentum diffusivity, which is what OF's turbulence->muEff() returns.
        // Dumped here rather than at the predictor because this is where it exists; comparing nuConst_
        // (the laminar part alone) against OF's muEff is not a comparison, it is two different quantities.
        if (compressible_ && stageDumpActive() && stageDumpFirstOnly("muEff")) stageDump("stage_muEff", nuEff);
        DeviceBuffer<scalar> nuEff_f;
        deviceInterpolate(dm, nuEff, nuEff_f);
        // nuEff at boundary FACES: turbulent uses the TRUE wall nut (nutkWallFunction), not the cell value, so the
        // wall shear matches OpenFOAM; laminar/non-wall faces use the adjacent cell value.
        // Boundary turbulent viscosity into muEff_b. OF: mut(patchi) = rho.boundaryField()[patchi]*nut(patchi),
        // so the wall nut -- which every wall function returns KINEMATIC -- is rho_b-weighted here, exactly as
        // the cell value is above. Incompressible keeps the plain sum.
        auto addWallNutToMuEff =
            [&](const DeviceBuffer<scalar>& nutB,
                DeviceBuffer<scalar>& muEffB)
            {
                deviceCopy(nutBndAll_, nutB);   // same nut: alphat_b (compressible) + DkEff/DepsEff(patchi)
                if (!compressible_)
                {
                    deviceAxpy(1.0, nutB, muEffB);
                    return;
                }
                DeviceBuffer<scalar> mutB;
                deviceHadamard(mutB, rhoBnd_, nutB);
                deviceAxpy(1.0, mutB, muEffB);
            };

        DeviceBuffer<scalar> nuEffBnd;
        if (ctl_.turbulent && ctl_.les)
        {
            // pure LES Smagorinsky: NO k-based wall function (there is no k field). Honour a velocity-based nut wall
            // function (nutUSpaldingWallFunction) if the 0/nut BC selects it; otherwise extrapolate the cell nuEff
            // (nu+nut) to the boundary (calculated/zeroGradient/fixedValue nut -> adjacent-cell sub-grid nut), exactly
            // as the laminar path does.
            if (ctl_.nutWall == NutWall::Spalding)
            {
                deviceCopy(nuEffBnd, nuBndConst_);
                deviceBoundaryNutSpalding(dbU_, bndIsWall_, bndY_, Uk_[0], Uk_[1], Uk_[2], dnut_, ctl_.nu, ctl_.saCoeffs, dnutBndWall_,
                                          compressible_ ? &nuWallBnd_ : nullptr);
                deviceAxpy(1.0, dnutBndWall_, nuEffBnd);
            }
            else
                deviceBCValue(dbExtrap_, nuEff, nuEffBnd);
        }
        else if (ctl_.turbulent)
        {
            deviceCopy(nuEffBnd, nuBndConst_);
            // Wall nut chosen by the 0/nut BC TYPE (ctl_.nutWall), matching OpenFOAM, NOT the model.
            if (ctl_.sa || ctl_.nutWall == NutWall::Spalding)   // nutUSpaldingWallFunction (velocity-based Newton uTau): SA always, or the BC on any model
            {
                deviceBoundaryNutSpalding(dbU_, bndIsWall_, bndY_, Uk_[0], Uk_[1], Uk_[2], dnut_, ctl_.nu, ctl_.saCoeffs, dnutBndWall_,
                                          nullptr, nutBndFile_.size() ? &nutBndFile_ : nullptr);
                // SPALART-ALLMARAS: the `calculated` patches carry nuTilda_b*fv1(chi_b), which is what OF's
                // correctNut leaves there -- `nut_ = nuTilda_*fv1` is a GeometricField assignment, so the
                // patch value comes from nuTilda's PATCH value. brae had no SA branch here at all (the
                // guard on the kEpsilon path reads `!ctl_.sa`), so those faces kept whatever 0/nut held.
                //
                // That is not a small error when 0/nut is a placeholder. On pimpleFoam/LES/vortexShed
                // `internalField uniform 1.0e-05` with nu = 1e-05, while the physical value is
                // nuTilda*fv1 = 1e-05*2.786e-03 = 2.79e-08 -- 360x smaller. brae's boundary nuEff came out
                // 2.000e-05 against OpenFOAM's 1.0028e-05, so every non-wall patch's momentum
                // internalCoeffs was 2x: predicted 1.9944, measured 1.9944 against the OF stage oracle.
                // Same defect class as the 704x `fixedValue` inlet above, one turbulence model along.
                if (hasNutCalc_ && ctl_.sa)
                {
                    DeviceBuffer<scalar> ntB, saNutB;
                    deviceBCValue(dbK_, dk_, ntB);            // nuTilda's own patch values (dk_ is nuTilda for SA)
                    deviceNutSABoundary(ntB, compressible_ ? &nuWallBnd_ : nullptr,
                                        ctl_.nu, ctl_.saCoeffs.Cv1, saNutB);
                    deviceSelectFixedFlux(nutCalcSel_, saNutB, dnutBndWall_);   // calc faces only; wall faces keep Spalding
                }
                addWallNutToMuEff(dnutBndWall_, nuEffBnd);
            }
            else if (ctl_.nutWall == NutWall::NutU)   // nutUWallFunction (log-law yPlus, stepwise blend)
            {
                deviceBoundaryNutU(dbU_, bndIsWall_, bndY_, Uk_[0], Uk_[1], Uk_[2], dnut_, ctl_.nu,
                                   ctl_.keCoeffs.kappa, ctl_.keCoeffs.E, dnutBndWall_,
                                   compressible_ ? &nuWallBnd_ : nullptr,
                                   nutBndFile_.size() ? &nutBndFile_ : nullptr);
                addWallNutToMuEff(dnutBndWall_, nuEffBnd);
            }
            else if (ctl_.nutWall == NutWall::Blended)   // nutUBlendedWallFunction (velocity-based binomial n=4 blend) on kEps/kOmegaSST
            {
                deviceBoundaryNutBlended(dbU_, bndIsWall_, bndY_, Uk_[0], Uk_[1], Uk_[2], dnut_, ctl_.nu, ctl_.keCoeffs.kappa, ctl_.keCoeffs.E, dnutBndWall_,
                                         compressible_ ? &nuWallBnd_ : nullptr,
                                   nutBndFile_.size() ? &nutBndFile_ : nullptr);
                addWallNutToMuEff(dnutBndWall_, nuEffBnd);
            }
            else   // k-based wall nut: nutkWallFunction (smooth), or atmNutkWallFunction (rough) when ctl_.atmZ0>0
            {
                DeviceBuffer<scalar> nutBnd;
                // kEpsilon only: the 'calculated' nut patches carry Cmu*k_b^2/eps_b. The SST's nut has a
                // different expression (a1*k/max(a1*omega, b1*F2*sqrt(S2))) that needs boundary F2 and S2,
                // so it keeps the extrapolated value until that is measured and built.
                DeviceBuffer<scalar> kB, eB;
                const bool keNut = hasNutCalc_ && !ctl_.sst && !ctl_.sa && !ctl_.keCoeffs.realizable;
                if (keNut) { deviceBCValue(dbK_, dk_, kB); deviceBCValue(dbEps_, de_, eB); }
                deviceBoundaryNut(dbU_.comp[0], bndIsWall_, bndY_, dk_, dnut_, ctl_.nu, nutBnd, ctl_.keCoeffs, ctl_.atmZ0, ctl_.atmBoundNut,
                                  compressible_ ? &nuWallBnd_ : nullptr,
                                  keNut ? &nutCalcMask_ : nullptr,
                                  keNut ? &kB : nullptr,
                                  keNut ? &eB : nullptr);
                // SST: the 'calculated' patches carry a1*k_b/max(a1*om_b, b1*F2_b*sqrt(S2_b)) -- a different
                // expression from kEpsilon's, needing boundary F2 and S2. Overwrites those faces only.
                if (hasNutCalc_ && ctl_.sst)
                {
                    DeviceBuffer<scalar> kBs, omBs, gradUs;
                    deviceBCValue(dbK_, dk_, kBs);
                    deviceBCValue(dbEps_, de_, omBs);
                    deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradUs,
                                hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
                    deviceSSTNutBoundary(dbU_, kBs, omBs, y_,   // CELL wall distance (bndY_ is 0 off-wall)
                                         compressible_ ? &nuWallBnd_ : nullptr, ctl_.nu,
                                         gradUs, nC_, Uk_[0], Uk_[1], Uk_[2],
                                         nutCalcMask_, ctl_.ksstCoeffs, dnut_, nutBnd);
                }
                // nutLowReWallFunction: calcNut() is Zero UNCONDITIONALLY
                // (nutLowReWallFunctionFvPatchScalarField.C:38-42 is the whole function). It differs from
                // the k-based branch ONLY in the wall value, so the branch above runs in full -- the
                // 'calculated' faces, the SST expression, all of it -- and the WALL faces alone are then
                // zeroed. Writing zeros over the entire boundary array instead cost bump2D:kOmegaSST
                // U 1.781e-03 against a 1.5e-03 bound and nut 2.444e-01 against 2.2e-01, because it
                // destroyed the calculated inlet/outlet values this branch had just computed.
                if (ctl_.nutWall == NutWall::LowRe && bndIsWall_.size() == nutBnd.size())
                {
                    DeviceBuffer<scalar> keep;
                    deviceCopy(keep, nutBnd);
                    cudaCheck(cudaMemsetAsync(nutBnd.data(), 0,
                                              nutBnd.size() * sizeof(scalar), cudaStreamPerThread),
                              "nutLowRe wall nut");
                    // writes `keep` where the mask is 0, i.e. every NON-wall face keeps what the branch
                    // above computed; wall faces stay at the zero OpenFOAM gives them.
                    deviceSelectFixedFlux(bndIsWall_, keep, nutBnd);
                }
                // ...then pin the faces whose nut BC fixes a value to that value. Last, so it overrides
                // whatever the wall/calculated evaluation left there.
                if (hasNutFixed_ && nutBndFile_.size() == nutBnd.size())
                    deviceSelectFixedFlux(nutFixedMask_, nutBndFile_, nutBnd);
                deviceCopy(dnutBndWall_, nutBnd);   // materialised: gpuSimpleFoam reads it for wallForces
                addWallNutToMuEff(nutBnd, nuEffBnd);
            }
        }
        else if (compressible_ && ctl_.gnPowerLaw)
            // generalizedNewtonian: OF's nuEff(patchi) is nu_.boundaryField()[patchi] -- the SAME
            // strain-rate viscosity as the interior, NOT the molecular mu. Taking mu_b = thermo mu here put
            // 1100x too little viscosity on every wall face (interior 0.9945 Pa.s vs thermo 8.9e-4), and the
            // wall shear is where a duct's entire pressure drop comes from: the profile came out flat
            // (U min 1.26 against OF's 0.031), the pressure drop 5x low, and the dissipation heating with it.
            // The face value is the adjacent cell's, as the incompressible laminar branch below does.
            deviceBCValue(dbExtrap_, nuEff, nuEffBnd);
        else if (compressible_)
            deviceCopy(nuEffBnd, muBnd_);   // laminar compressible: mu_b = Sutherland(T_b), OF transport_.mu(patchi)
        else
            deviceBCValue(dbExtrap_, nuEff, nuEffBnd);   // laminar: adjacent-cell value
        // explicit divDevReff stress (uses the incoming U; coupled across the 3 components)
        DeviceBuffer<scalar> ddrX, ddrY, ddrZ;
        // ...with the case's NAMED grad(U) scheme, not a bare Gauss one: OF's divDevReff calls
        // fvc::grad(U) and gets whatever gradSchemes says. See the note on the parameter.
        // MAXWELL splits the coefficient in two. OF's Maxwell::divDevRhoReff is
        //     div(nuM*grad(U)) + div(sigma) - div(nu*dev2(T(grad(U)))) - laplacian(nu + nuM, U)
        // so the dev2 transpose term carries the MOLECULAR nu while the implicit laplacian carries
        // nu0 = nu + nuM. Every other model has one nuEff for both, which is why this is the one place
        // the two are told apart.
        const DeviceBuffer<scalar>& ddrNuCell = ctl_.maxwell ? nuConst_ : nuEff;
        DeviceBuffer<scalar> ddrNuBnd;
        if (ctl_.maxwell) deviceBCValue(dbExtrap_, nuConst_, ddrNuBnd);
        deviceDivDevReff(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], ddrNuCell, ctl_.maxwell ? ddrNuBnd : nuEffBnd,
                         ddrX, ddrY, ddrZ,
                         hasCyclic_ ? &cyc_ : nullptr, hasAMI_ ? &ami_ : nullptr, nullptr, ctl_.gradULimitK);
        // ...and the two terms only Maxwell has. Both are V*fvc::div of a tensor, added to the same
        // source the dev2 term feeds, with the same sign OF gives them (divDevRhoReff is subtracted from
        // the momentum equation, and brae's ddr carries that sign already -- see the loop below).
        if (ctl_.maxwell)
        {
            DeviceBuffer<scalar> gU;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gU, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            if (ctl_.gradULimitK > 0.0)
                deviceCellLimitGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gU, ctl_.gradULimitK,
                                     hasCyclic_ ? &cyc_ : nullptr, hasAMI_ ? &ami_ : nullptr);
            DeviceBuffer<scalar> mX, mY, mZ;
            deviceDivNuMGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gU, ctl_.maxwellNuM, mX, mY, mZ);
            deviceAxpy(-1.0, mX, ddrX);   // ddr holds -divDevReff's dev2 half; div(nuM*grad U) enters with the opposite sign
            deviceAxpy(-1.0, mY, ddrY);
            deviceAxpy(-1.0, mZ, ddrZ);
            DeviceBuffer<scalar> sBnd[6];
            for (int q = 0; q < 6; ++q) deviceBCValue(dbSig_[q], sig_[q], sBnd[q]);
            DeviceBuffer<scalar> sX, sY, sZ;
            deviceDivSymmTensor(dm, sig_, sBnd, sX, sY, sZ);
            deviceAxpy(-1.0, sX, ddrX);
            deviceAxpy(-1.0, sY, ddrY);
            deviceAxpy(-1.0, sZ, ddrZ);
        }
        DeviceBuffer<scalar>* ddr[3] = { &ddrX, &ddrY, &ddrZ };
        if (stageDumpActive() && stageDumpFirstOnly("ddr")) stageDump3("stage_divDevReff", ddrX, ddrY, ddrZ);

        DeviceBuffer<scalar> pbv;
        deviceBCValue(dbP_, dp_, pbv);
        // gx/gy/gz are members (shared with the pressure phase).
        deviceGaussGrad(dm, dp_, pbv, gx, gy, gz);
        if (hasCyclic_) interfaceAddGrad(cyc_, dp_, dm.V, gx, gy, gz);   // cyclic-face contribution to grad(p) in the momentum source
        if (hasAMI_)    interfaceAddGrad(ami_, dp_, dm.V, gx, gy, gz);       // AMI-face contribution to grad(p)
        DeviceBuffer<scalar> mDiag, lD, lU, lL;   // mUp/mLo are now members (shared with the pressure phase)
        // div(phi,U): central differencing when the case asks for a bare `Gauss linear`, otherwise the
        // upwind matrix (with linearUpwind's deferred correction added separately below). Same
        // coefficients either way -- only the interpolation weight differs, exactly as OF's
        // gaussConvectionScheme::fvmDiv takes weights from whichever scheme was named.
        //
        // `Gauss limitedLinearV k` is the third case and the only one that needs a field: OF's
        // LimitedScheme builds a per-face limiter from fvc::grad(U) and blends the central and upwind
        // weights IMPLICITLY (limitedSurfaceInterpolationScheme::weights), so the blend belongs in the
        // matrix, not in a deferred source. The gradient is the one gradSchemes `grad(U)` names --
        // fvc::grad(lPhi) inside calcLimiter -- hence the same cellLimited treatment the other
        // grad(U) consumers get. Uk_ still holds the old velocity here (nothing is solved until the kk
        // loop below), which is what OF interpolates too.
        // Hoisted out of the branch below: the interface assembly further down needs the SAME gradient
        // to build its own limiter, because a coupled face is limited by OF exactly as an internal one is
        // (LimitedScheme::calcLimiter has a coupled() branch; only an UNCOUPLED patch gets limiter 1).
        DeviceBuffer<scalar> gradUlv;
        if (ctl_.divULinear) deviceDivCentralCoeffs(dm, phiInt_, mDiag, mUp, mLo);
        else if (ctl_.divULimitedV)
        {
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradUlv, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            if (ctl_.gradULimitK > 0.0)
                deviceCellLimitGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradUlv, ctl_.gradULimitK,
                                     hasCyclic_ ? &cyc_ : nullptr, hasAMI_ ? &ami_ : nullptr);
            deviceDivLimitedVCoeffs(dm, phiInt_, Uk_, gradUlv, ctl_.divUTwoBykV, mDiag, mUp, mLo);
        }
        else if (ctl_.lust)
        {
            // LUST's blend belongs in the MATRIX, not in a deferred source. OF LUST.H:
            //     weights()    = 0.75*linear.weights() + 0.25*linearUpwind::weights()
            //     correction() = 0.25*linearUpwind::correction()
            // so three quarters of the central weighting is implicit and only a quarter of the
            // gradient term is explicit. brae assembled a PURE UPWIND matrix and carried the whole
            // blend explicitly. The face value is the same either way, and the matrix is not: upwind
            // puts more on the diagonal, so rAU came out too small -- and rAU is what sets the
            // pressure-Laplacian coefficient and HbyA, so the entire pressure-velocity coupling was
            // scaled wrong.
            //
            // Measured against the OF stage oracle on pimpleFoam/LES/vortexShed, ONE momentum assembly
            // from identical inputs: brae's far-field diagonal was 1.1118x OpenFOAM's over the 112640
            // cells that touch no boundary, and mDiag/V was 131.5 against 118.25 with 1/dt = 100 --
            // i.e. the convection+diffusion part was ~1.7x. The same case run with `Gauss linear`,
            // where brae's implicit weights already matched, gave a ratio of 1.0000. Downstream that
            // 11% seeded an alternating-sign pressure mode in the second cell layer off the cylinder
            // which grew 3-4x per step until |U| reached 1e+10 against OpenFOAM's 0.0435.
            //
            // fvm::div's coefficients are LINEAR in the face weight, so the blend is exact as a
            // combination of the two schemes brae already builds -- no third kernel, and no chance of
            // the blended form drifting from the central and upwind ones it is defined against.
            DeviceBuffer<scalar> cD, cU, cL;
            deviceDivCentralCoeffs(dm, phiInt_, cD, cU, cL);
            deviceDivUpwindCoeffs(dm, phiInt_, mDiag, mUp, mLo);
            deviceScale(mDiag, DeviceSimpleControls::lustUpwindFrac);  deviceAxpy(DeviceSimpleControls::lustCentralFrac, cD, mDiag);
            deviceScale(mUp,   DeviceSimpleControls::lustUpwindFrac);  deviceAxpy(DeviceSimpleControls::lustCentralFrac, cU, mUp);
            deviceScale(mLo,   DeviceSimpleControls::lustUpwindFrac);  deviceAxpy(DeviceSimpleControls::lustCentralFrac, cL, mLo);
        }
        else                 deviceDivUpwindCoeffs(dm, phiInt_, mDiag, mUp, mLo);
        deviceLaplacianCoeffs(dm, nuEff_f, lD, lU, lL, ctl_.nonOrth);
        deviceAxpy(-1.0,lD,mDiag);
        deviceAxpy(-1.0,lU,mUp);
        deviceAxpy(-1.0,lL,mLo);
        // bounded Gauss upwind: - fvm::Sp(fvc::div(phi), U). mDiag -= V*div(phi). Stabilises the rest-start
        // transient (vanishes at convergence); matches OF's bounded scheme + the CPU simpleStep.
        // div(phi) MUST include the cyclic-face flux (OF's fvc::div sums ALL faces). Without it the cyclic cells
        // see the un-cancelled cyclic flux -/+phi/V as a spurious divergence, so the bounded term injects -/+phi into the
        // diagonal, asymmetric by 2phi between the inflow/outflow cyclic sides -> asymmetric rAU/HbyA -> non-axisym p.
        if (ctl_.bounded)
        {
            DeviceBuffer<scalar> dphiM;
            deviceDiv(dm, phiInt_, phiBnd_, dphiM);
            if (hasCyclic_) interfaceAddDiv(cyc_, dm.V, dphiM);
            if (hasAMI_)    interfaceAddDiv(ami_, dm.V, dphiM);
            DeviceBuffer<scalar> bM;
            deviceHadamard(bM, dphiM, dm.V);
            deviceAxpy(-1.0, bM, mDiag);
        }
        // cyclic (periodic) momentum coupling M = div(phi,U) - laplacian(nuEff,U): folds the interface diagonal into
        // mDiag + builds cyc_.ifCoeff (off-diag) from the current cyclic flux; cycSumOff feeds the relax dominance.
        //
        // THE INTERFACE'S OWN DIV-SCHEME WEIGHT. Both assemblies below used to split the convective term
        // upwind -- ifCoeff = -lap + min(phi,0), diag += lap + max(phi,0) -- for EVERY case, because
        // that is what w = pos0(phi) is. OpenFOAM instead builds a coupled patch with the weights of the
        // scheme the case NAMED, so on pipeCyclic (`bounded Gauss limitedLinearV 1`) brae's interface
        // off-diagonal never changed sign where OpenFOAM's did: L2 6.86e-01 against OF's own
        // boundaryCoeffs, with the gap equal to phi*(1-w) face by face.
        //
        // LUST is the one blend still assembled upwind at a coupled face (w = 0.75*cd + 0.25*pos0). No
        // simpleFoam tutorial names it -- the fourteen that do are fireFoam and pimpleFoam LES -- so it
        // has no case to be gated on here; see the manifest entry.
        DeviceBuffer<scalar> cycWsch, amiWsch;
        if (ctl_.divULimitedV)
        {
            if (hasCyclic_)
                deviceCyclicLimitedVWeights(cyc_, Uk_[0], Uk_[1], Uk_[2], gradUlv, nC_, ctl_.divUTwoBykV, cycWsch);
            if (hasAMI_)
                deviceAmiLimitedVWeights(ami_, Uk_[0], Uk_[1], Uk_[2], gradUlv, nC_, ctl_.divUTwoBykV, amiWsch);
        }
        // `Gauss linear` -- CENTRAL differencing, and at a coupled face there is nothing to compute: OF's
        // linear scheme returns mesh.surfaceInterpolation::weights() (linear.H:106), whose boundary field
        // on a coupled patch IS the patch's own interpolation weight, which both interface structs already
        // carry and already use for every face value they form. So the weight is a pointer, not a kernel.
        //
        // This is the widest of the three gaps in absolute terms: w = wCD is ~0.5, the furthest a weight
        // can get from upwind's 0 or 1, so a case that asks for `Gauss linear` and is given upwind at its
        // interface is not slightly off there, it is running a different scheme at half strength.
        else if (ctl_.divULinear)
        {
            if (hasCyclic_) deviceCopy(cycWsch, cyc_.weights);
            if (hasAMI_)    deviceCopy(amiWsch, ami_.weights);
        }
        DeviceBuffer<scalar> cycSumOff;
        if (hasCyclic_)
        {
            interfaceAssembleMomentum(cyc_, nuEff, mDiag, cycWsch.size() ? &cycWsch : nullptr);
            cycSumOff.copyFrom(std::vector<scalar>(nC_, 0.0));
            interfaceOffDiagSum(cyc_, cycSumOff);
            if (cyc_.rotational) interfaceScaleImplicit(cyc_);   // per-component ifCoeffC[kk] = ifCoeff*forwardT[kk][kk]
            // The cyclic MOMENTUM off-diagonal, dumped so a conformal cyclicAMI run can be diffed
            // against a cyclic run of the same mesh face for face -- at weight 1 and a 1:1 stencil the
            // two must agree exactly, so any difference localises which path is wrong.
            if (stageDumpActive() && stageDumpFirstOnly("cycIfCoeffMom"))
                stageDump("stage_cyc_ifCoeffMom", cyc_.ifCoeff);
        }
        DeviceBuffer<scalar> amiSumOff;                              // cyclicAMI momentum coupling (translational)
        if (hasAMI_)
        {
            interfaceAssembleMomentum(ami_, nuEff, mDiag, amiWsch.size() ? &amiWsch : nullptr);
            if (stageDumpActive() && stageDumpFirstOnly("amiNuEff"))
            {   // the interface diffusivity the momentum laplacian uses: fvc::interpolate(nuEff) there
                DeviceBuffer<scalar> tmpF, tmpN;
                deviceAmiFaceValue(ami_, nuEff, tmpF);
                deviceAmiInterpolate(ami_, nuEff, tmpN);
                stageDump("stage_ami_nuEffFace", tmpF);
                stageDump("stage_ami_nuEffNbr", tmpN);
                stageDump("stage_ami_nuEffCell", nuEff);
                stageDump("stage_ami_ifCoeffMom", ami_.ifCoeff);   // the MOMENTUM off-diagonal
                stageDump("stage_ami_phiMom", ami_.phi);           // the flux its upwinding reads
            }
            amiSumOff.copyFrom(std::vector<scalar>(nC_, 0.0));
            interfaceOffDiagSum(ami_, amiSumOff);
            if (ami_.rotational) interfaceScaleImplicit(ami_);   // per-component ifCoeffC[kk] = ifCoeff*forwardT[kk][kk]
        }
        // KEEP THE MOMENTUM INTERFACE COEFFICIENT. cyc_/ami_.ifCoeff is ONE buffer written by two different
        // assemblies: interfaceAssembleMomentum here, and interfaceAssembleLaplacian for the pressure in
        // correctPressureVelocity. UEqn.H() needs the MOMENTUM one (interfaceAddH), and on the first
        // pressure corrector it still holds it -- H is built before the pressure is assembled. On the
        // SECOND corrector it does not: the first corrector's pressure assembly has overwritten it, so H
        // picks up gammaFace*deltaCoeffs*magSf instead of -(nuFace*dc*magSf) + min(phi,0). On
        // pimpleFoam/RAS/oscillatingInletACMI2D that is 4.7e-05 in place of -5.3e-08, a factor of ~900,
        // and it lands on H at exactly the interface cells.
        //
        // The symptom is specific enough to name: with nCorrectors 1 brae matched OpenFOAM to 7.6e-08 in
        // U from an identical start; with nCorrectors 2 the same step came out 2.6e-03, as a UNIFORM
        // pressure offset on the whole upstream block (a uniform shift has no gradient, so the bulk
        // velocity stayed exact and only the interface-adjacent columns moved).
        if (hasCyclic_) deviceCopy(cycIfCoeffMom_, cyc_.ifCoeff);
        if (hasAMI_)    deviceCopy(amiIfCoeffMom_, ami_.ifCoeff);
        if (fvoMomSp_.size()) deviceAxpy(-1.0, fvoMomSp_, mDiag);   // fvOptions implicit momentum Sp: diag -= Sp*V (== fvm::Sp)
        // explicitPorositySource (DarcyForchheimer): isotropic resistance into the diagonal (mDiag += V*tr(Cd)); the
        // anisotropic remainder is added to relaxSrc per component below. Uses the LAMINAR nu (OF mu/rho, not nuEff).
        if (por_.active) deviceFvoPorosityDiag(por_, ctl_.nu, dm.V, Uk_[0], Uk_[1], Uk_[2], mDiag);
        DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };
        DeviceBuffer<scalar> r0IC,r0BC,r0lIC,r0lBC;
        deviceBCDivCoeffs(dbU_.comp[0], phiBnd_, r0IC, r0BC);
        deviceBCLaplacianCoeffsFace(dbU_.comp[0], nuEffBnd, r0lIC, r0lBC);
        deviceAxpy(-1.0,r0lIC,r0IC);
        // slip/symmetry: the boundary diagonal differs per component (vf_k=|n_k|), so the SHARED relax/rAU diagonal
        // must use OF's cmptMax(cmptMag(iC0,iC1,iC2)), not comp[0] alone (== comp[0] when components are equal, so
        // every other BC is unaffected). Compute the per-component boundary iC and the per-face max magnitude.
        DeviceBuffer<scalar> iCmaxMag, iCmin;
        if (hasSym_)
        {
            DeviceBuffer<scalar> r1IC,r1BC,r1lIC,r1lBC, r2IC,r2BC,r2lIC,r2lBC;
            deviceBCDivCoeffs(dbU_.comp[1], phiBnd_, r1IC, r1BC);
            deviceBCLaplacianCoeffsFace(dbU_.comp[1], nuEffBnd, r1lIC, r1lBC);
            deviceAxpy(-1.0,r1lIC,r1IC);
            deviceBCDivCoeffs(dbU_.comp[2], phiBnd_, r2IC, r2BC);
            deviceBCLaplacianCoeffsFace(dbU_.comp[2], nuEffBnd, r2lIC, r2lBC);
            deviceAxpy(-1.0,r2lIC,r2IC);
            deviceCmptMaxMag3(r0IC, r1IC, r2IC, iCmaxMag);   // OF fvMatrix::relax adds cmptMax(cmptMag(iC)) to the diagonal
            deviceCmptMin3(r0IC, r1IC, r2IC, iCmin);         // ... and REMOVES cmptMin(iC), a different quantity
        }
        // + implicit fvm::ddt(U) DIAGONAL (rho=1 incompressible): coefft*rDeltaT*V, shared by all 3 components, added
        // to the assembled diagonal BEFORE relax -- OF assembles fvm::ddt into UEqn, then UEqn.relax(). No-op for SIMPLE.
        deviceFvmDdtDiag(dm.V, ddtc, 1.0, mDiag);
        DeviceBuffer<scalar> delta;   // mDiagR is now a member
        deviceRelaxDiag(deviceLduView(dm,mDiag,mUp,mLo), dm, r0IC, ctl_.uRelax(), mDiagR, delta,
                        hasCyclic_ ? cycSumOff.data() : (hasAMI_ ? amiSumOff.data() : nullptr),
                        hasSym_ ? iCmaxMag.data() : nullptr,
                        hasSym_ ? iCmin.data() : nullptr);
        if (stageDumpActive() && stageDumpFirstOnly("mommat"))
        {
            stageDump("stage_mDiag0", mDiag);     // assembled momentum diagonal BEFORE relax (OF D0)
            stageDump("stage_mDiagR", mDiagR);    // ... and AFTER relax (OF D)
        }
        // velocityDampingConstraint: OF fvOptions.constrain(UEqn) runs AFTER UEqn.relax() and adds an implicit diagonal
        // sink (no source) where |U|>UMax. Add to the relaxed diagonal so it feeds the predictor AND rAU/HbyA (= OF A()).
        if (vdcActive_) deviceFvoVelocityDamping(vdcCells_, vdcUMax_, vdcC_, dm.V, Uk_[0], Uk_[1], Uk_[2], mDiagR);
        const DeviceLduView Uview = deviceLduView(dm, mDiagR, mUp, mLo);
        // rotational cyclic: snapshot U_old so the deferred rotation MIXING is built from ONE consistent field for
        // all three components. OF evaluates the cyclic patchNeighbourField (forwardT.U[nbr]) ONCE at UEqn assembly,
        // from U_old, for the whole vector. cf solves the 3 components SEQUENTIALLY in place (Uk_[kk] is the solution
        // vector), so reading the live Uk_ inside the loop would feed component kk the freshly-solved earlier
        // components -> the x<->y rotation mixing becomes asymmetric -> spurious cross-component coupling that breaks
        // the rotational (axi)symmetry and drifts the converged field. The snapshot makes it order-independent.
        DeviceBuffer<scalar> Usnap[3];
        if ((hasCyclic_ && cyc_.rotational) || (hasAMI_ && ami_.rotational))
            for (int kk = 0; kk < 3; ++kk)
                deviceCopy(Usnap[kk], Uk_[kk]);
        // rotational AMI deferred: the UN-rotated AMI-interpolated neighbour of the U_old snapshot (per src face),
        // consistent across components (the deferred mixing reads it, not the live Uk_).
        DeviceBuffer<scalar> Uint[3];
        if (hasAMI_ && ami_.rotational)
            for (int l = 0; l < 3; ++l)
                deviceAmiInterpolate(ami_, Usnap[l], Uint[l]);
        // linearUpwind / non-orth need grad(U). Precompute ALL 3 component gradients HERE (Uk_ is still all-old at this
        // point, none solved yet), cyclic-inclusive: the cyclic linearUpwind reconstruction rotates gradU[nbr], so it
        // needs every component from a consistent field (computing per-kk inside the loop would read freshly-solved
        // earlier components -> the same sequencing leak as the deferred mixing).
        // gU* = gradSchemes `default` (unlimited), for the non-orthogonal laplacian correction.
        // gLU* = the gradient the linearUpwind div statement names, limited when that name resolves to a
        // cellLimited scheme. LUx/LUy/LUz below select between them so the unlimited path costs nothing.
        DeviceBuffer<scalar> gUx[3], gUy[3], gUz[3];
        DeviceBuffer<scalar> gLUx[3], gLUy[3], gLUz[3];
        DeviceBuffer<scalar> cycNbr[3], amiNbr[3];   // interface neighbour value per face, for the limiter
        const bool limitLU = (ctl_.gradULULimitK > 0.0);
        DeviceBuffer<scalar> amiURot[3];   // rotated AMI-interp of the CURRENT U (gradient face value + linearUpwind/non-orth nbr)
        if ((ctl_.linearUpwind || ctl_.lust || ctl_.nonOrth) && hasAMI_ && ami_.rotational)
            deviceAmiInterpolateVec(ami_, Uk_[0], Uk_[1], Uk_[2], amiURot[0], amiURot[1], amiURot[2]);
        if (ctl_.linearUpwind || ctl_.lust || ctl_.nonOrth)
            for (int l = 0; l < 3; ++l)
            {
                DeviceBuffer<scalar> ubv;
                deviceBCValue(dbU_.comp[l], Uk_[l], ubv);
                deviceGaussGrad(dm, Uk_[l], ubv, gUx[l], gUy[l], gUz[l]);
                if (hasCyclic_)
                {
                    if (cyc_.rotational) deviceCyclicAddGradRot(cyc_, Uk_[0], Uk_[1], Uk_[2], l, dm.V, gUx[l], gUy[l], gUz[l]);
                    else interfaceAddGrad(cyc_, Uk_[l], dm.V, gUx[l], gUy[l], gUz[l]);
                }
                if (hasAMI_)
                {
                    if (ami_.rotational) deviceAmiAddGradRot(ami_, Uk_[l], amiURot[l], dm.V, gUx[l], gUy[l], gUz[l]);
                    else interfaceAddGrad(ami_, Uk_[l], dm.V, gUx[l], gUy[l], gUz[l]);
                }
                // grad(U) cellLimited Gauss linear <k>, into a SEPARATE buffer.
                //
                // TWO CONSUMERS, TWO SCHEMES. OF does not have "the" grad(U) here: linearUpwind takes the
                // gradient its own div statement NAMES (`linearUpwind limited` -> cellLimited Gauss linear 1,
                // via mesh.gradScheme(gradSchemeName_) in linearUpwind.C), while the non-orthogonal laplacian
                // correction takes gradSchemes `default`, which in this case is plain Gauss linear. Limiting
                // one buffer in place gave BOTH the same gradient and so had to be wrong for one of them:
                // unlimited was wrong for linearUpwind (the converged squareBendLiq U was 24x off), and
                // limiting it in place then fed an over-limited gradient into the non-orth correction and
                // produced a transient nut excursion. Keeping them apart is the only way both match OF.
                if (limitLU)
                {
                    deviceCopy(gLUx[l], gUx[l]);
                    deviceCopy(gLUy[l], gUy[l]);
                    deviceCopy(gLUz[l], gUz[l]);
                    // ...and the limiter has to SEE the interface. A coupled face is invisible to brae's
                    // boundary addressing but internal to OF's cellLimitedGrad, so without this an
                    // interface cell is limited as if its interface neighbour did not exist. Measured on
                    // oscillatingInletACMI2D, laminar + linearUpwind, 10 free steps: 1.9e-03 against
                    // OpenFOAM with the interface omitted here and 2.0e-09 with the same case run on an
                    // UNLIMITED grad(U) -- which is how the limiter, not the linearUpwind correction,
                    // was identified. 91% of the squared error sat on the 136 interface cells.
                    CellLimitInterface ifs[2];
                    int nIfs = 0;
                    if (hasCyclic_ && cyc_.n > 0)
                    {
                        deviceCyclicNbrValue(cyc_, Uk_[l], Usnap[0], Usnap[1], Usnap[2], l, cycNbr[l]);
                        ifs[nIfs++] = { cyc_.n, cyc_.ownCell.data(), cycNbr[l].data(),
                                        cyc_.dOwnX.data(), cyc_.dOwnY.data(), cyc_.dOwnZ.data() };
                    }
                    if (hasAMI_ && ami_.n > 0)
                    {
                        if (ami_.rotational) deviceCopy(amiNbr[l], amiURot[l]);
                        else                 deviceAmiInterpolate(ami_, Uk_[l], amiNbr[l]);
                        ifs[nIfs++] = { ami_.n, ami_.ownCell.data(), amiNbr[l].data(),
                                        ami_.dOwnX.data(), ami_.dOwnY.data(), ami_.dOwnZ.data() };
                    }
                    deviceCellLimitGrad(dm, Uk_[l], ubv, gLUx[l], gLUy[l], gLUz[l], ctl_.gradULULimitK,
                                        ifs, nIfs);
                }
            }
        // actuationDiskSource (Froude): T = 2*rho*A*(Uref.diskDir)^2*a*(1-a), Uref = mean U over the monitor cells
        // (lagged). Computed once per iter; distributed over the disk cells (V[c]/Vtot) into relaxSrc below.
        std::vector<scalar> adT(adDisks_.size(), 0.0);
        for (std::size_t di = 0; di < adDisks_.size(); ++di)
        {
            const auto& d = adDisks_[di];
            const scalar ud = (deviceDot(Uk_[0], d.monMask01)*d.diskDir.x + deviceDot(Uk_[1], d.monMask01)*d.diskDir.y
                             + deviceDot(Uk_[2], d.monMask01)*d.diskDir.z) / d.nmon;   // this disk's Uref . diskDir
            adT[di] = 2.0 * d.area * ud*ud * d.a * (1.0 - d.a);
        }
        // rotorDiskSource (BEM): per-cell body force from the current U (lagged), recomputed each iter.
        if (rotor_.active) deviceRotorForce(rotor_, nC_, Uk_[0], Uk_[1], Uk_[2], rotorFx_, rotorFy_, rotorFz_);
        DeviceBuffer<scalar>* rotorF[3] = { &rotorFx_, &rotorFy_, &rotorFz_ };
        // linearUpwindV: the V-limiter couples the 3 velocity components, so compute all 3 correction sources once
        // here (the plain per-component linearUpwind path stays inside the kk loop below).
        DeviceBuffer<scalar> corrV[3];
        // The linearUpwind family reads the NAMED (possibly limited) gradient; the non-orth laplacian
        // correction below keeps gU* (gradSchemes `default`). See the split at the gradient build above.
        const DeviceBuffer<scalar>* LUx = limitLU ? gLUx : gUx;
        const DeviceBuffer<scalar>* LUy = limitLU ? gLUy : gUy;
        const DeviceBuffer<scalar>* LUz = limitLU ? gLUz : gUz;
        // DEShybrid: the DES sensor is a function of the CURRENT velocity gradient, nut and the filter
        // width, so sigma is rebuilt once per assembly and shared by the three components. Its gradient is
        // OF's fvc::grad(U), i.e. the gradSchemes `grad(U)` entry -- the same one kEpsilon's production
        // takes, and limited when that entry says cellLimited.
        DeviceBuffer<scalar> desSigma;
        if (ctl_.desHybrid)
        {
            DeviceBuffer<scalar> gradU;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            if (ctl_.gradULimitK > 0.0)
                deviceCellLimitGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU, ctl_.gradULimitK,
                                     hasCyclic_ ? &cyc_ : nullptr, hasAMI_ ? &ami_ : nullptr);
            // delta = the LES filter width the turbulence model itself uses -- lesDelta_ when the case
            // named one (maxDeltaxyz), else empty and every kernel falls back to cubeRootVol (V^(1/3)).
            // The scheme must agree with the model rather than pick a second, different width.
            deviceDesHybridSigma(nC_, gradU, dm.V, dnut_, ctl_.nu, ctl_.desCoeffs, desSigma, &lesDelta_);
        }
        if (ctl_.linearUpwindV) deviceLinearUpwindVCorr(dm, phiInt_, LUx, LUy, LUz, Uk_[0], Uk_[1], Uk_[2], corrV[0], corrV[1], corrV[2]);
        for (int kk = 0; kk < 3; ++kk)   // (#5 OpenMP-parallel momentum was tried + benchmarked -> 136x SLOWER, reverted)
        {
            DeviceBuffer<scalar> dIC,dBC,lIC,lBC;
            deviceBCDivCoeffs(dbU_.comp[kk],phiBnd_,dIC,dBC);
            deviceBCLaplacianCoeffsFace(dbU_.comp[kk],nuEffBnd,lIC,lBC);
            if (kk == 0 && stageDumpActive() && stageDumpFirstOnly("nuEffBnd")) stageDump("stage_nuEffBnd", nuEffBnd);
            deviceAxpy(-1.0,lIC,dIC);
            deviceAxpy(-1.0,lBC,dBC);
            iC[kk]=std::move(dIC);
            bCb[kk]=std::move(dBC);
            // Stage harness: the MOMENTUM boundary coefficients, at the same point OF's UEqn has them
            // (after div + laplacian, before the solve). Per component, flattened in patch order -- the
            // comparison tool sums them per patch against OF's stage_UIC/stage_UBC.
            if (stageDumpActive() && stageDumpFirstOnly(kk == 0 ? "mombc0" : (kk == 1 ? "mombc1" : "mombc2")))
            {
                stageDump(std::string("stage_UIC") + char('0' + kk), iC[kk]);
                stageDump(std::string("stage_UBC") + char('0' + kk), bCb[kk]);
                // ...and U's evaluated boundary values, the quantity bC is built FROM. Without it a bC
                // mismatch cannot be attributed: a wrong wall velocity and a wrong coefficient formed
                // from a right one look identical downstream.
                { DeviceBuffer<scalar> ubv; deviceBCValue(dbU_.comp[kk], Uk_[kk], ubv);
                  stageDump(std::string("stage_Ubnd") + char('0' + kk), ubv); }
            }
            deviceHadamard(relaxSrc[kk], delta, Uk_[kk]);
            deviceAxpy(1.0, *ddr[kk], relaxSrc[kk]);                                  // += explicit divDevReff stress
            // + fvm::ddt(U) SOURCE for this component: rDeltaT*V*(coefft0*Uold - coefft00*Uold2). Into relaxSrc so it
            // feeds BOTH the predictor solve AND H()/HbyA (OF's UEqn.H() includes the ddt source). No-op for SIMPLE.
            // V^{n-1} on the SOURCE, V^n on the diagonal (OF EulerDdtScheme.C:383-388). That pairing
            // is the discrete SCL; one volume for both leaks continuity every step on a moving mesh.
            deviceFvmDdtSource(Vold_.size() ? Vold_ : dm.V, ddtc, 1.0, Uold_[kk], Uold2_[kk], relaxSrc[kk], ddtc.cn ? &Uddt0_[kk] : nullptr);   // CrankNicolson: + ocCoeff*ddt0
            if (mrf_.active) deviceMrfCoriolis(mrf_, dm.V, Uk_[0], Uk_[1], Uk_[2], kk, relaxSrc[kk]);   // MRF Coriolis: -V*(Omega x U)_kk on zone cells
            // Explicit momentum corrections (linearUpwind deferred + non-orth laplacian) go into relaxSrc so they
            // feed BOTH the predictor solve AND H()/HbyA, OF's UEqn.H() includes them. Adding them only to the
            // predictor source leaves the corrector inconsistent and the U residual plateaus (does NOT reach 1e-6).
            if (ctl_.linearUpwind || ctl_.lust)   // deferred correction: source -= div(phi * grad(U_kk)_upwind.(Cf-C_up))
            {
                DeviceBuffer<scalar> corr;
                if (ctl_.desHybrid)     deviceDesHybridCorr(dm, phiInt_, desSigma, Uk_[kk], LUx[kk], LUy[kk], LUz[kk], corr);
                else if (ctl_.linearUpwindV) corr = std::move(corrV[kk]);                     // vector-limited (computed once above)
                else                    deviceLinearUpwindCorr(dm, phiInt_, LUx[kk], LUy[kk], LUz[kk], corr);
                if (hasCyclic_) interfaceAddLinUpwindCorr(cyc_, kk, LUx, LUy, LUz, corr);   // + cyclic-face linearUpwind (rotated nbr)
                if (hasAMI_)    interfaceAddLinUpwindCorr(ami_, kk, LUx, LUy, LUz, corr);      // + AMI-face linearUpwind (rotated nbr stencil)
                // OF LUST.H correction() = 0.25*linearUpwind::correction(), and NOTHING else: the
                // 0.75*linear part is carried by the implicit weights above, not here. Adding a linear
                // deferred correction on top of the blended matrix would apply that share twice.
                if (ctl_.lust) deviceScale(corr, DeviceSimpleControls::lustUpwindFrac);
                deviceAxpy(-1.0, corr, relaxSrc[kk]);
            }
            if (ctl_.nonOrth)   // -fvm::laplacian source correction: -= lapCorr(nuEff, grad(U_kk))
            {
                DeviceBuffer<scalar> lc;
                deviceLaplacianCorr(dm, nuEff_f, gUx[kk], gUy[kk], gUz[kk], lc);
                if (hasCyclic_) interfaceAddLapCorr(cyc_, kk, nuEff, gUx, gUy, gUz, lc);   // + cyclic-face non-orth (rotated tensor)
                if (hasAMI_)    interfaceAddLapCorr(ami_, kk, nuEff, gUx, gUy, gUz, lc);       // + AMI-face non-orth (rotated tensor stencil)
                deviceAxpy(-nonOrthRelaxU, lc, relaxSrc[kk]);
            }
            // constant body force (drives periodic channels) is an explicit momentum SOURCE: it must go into relaxSrc
            // so it feeds BOTH the predictor AND H()/HbyA (OF UEqn.H() includes fvOptions sources). Adding it only to
            // the predictor leaves HbyA short by rAU*g -> the corrector U = HbyA - rAU*grad(p) converges ~(1-relax) low.
            const scalar bf = (kk==0)?ctl_.bodyForce.x:(kk==1)?ctl_.bodyForce.y:ctl_.bodyForce.z;
            if (bf != 0.0) deviceAxpy(bf, dm.V, relaxSrc[kk]);   // += V*g
            if (hasFvoMom_) deviceAxpy(1.0, fvoMomSu_[kk], relaxSrc[kk]);   // == fvOptions(U): explicit momentum source Su*V
            if (por_.active) deviceFvoPorositySource(por_, kk, ctl_.nu, dm.V, Uk_[0], Uk_[1], Uk_[2], relaxSrc[kk]);  // porosity anisotropic remainder
            if (mvfActive_) deviceAxpy((kk==0?mvfFlowDir_.x:kk==1?mvfFlowDir_.y:mvfFlowDir_.z)*(mvfGradP0_ + mvfDGradP_), mvfMaskV_, relaxSrc[kk]);  // meanVelocityForce body force (OF addSup: flowDir*(gradP0+dGradP)*V)
            // actuationDisk: OF adds T*diskDir DIRECTLY to eqn.source() (= eqn -= ., opposite cf's relaxSrc convention,
            // which mirrors meanVelocityForce's eqn += Su) -> relaxSrc -= (V/Vtot)*T*diskDir (a momentum sink/turbine).
            for (std::size_t di = 0; di < adDisks_.size(); ++di)   // turbines superpose into relaxSrc
            {
                const auto& d = adDisks_[di];
                deviceAxpy(-adT[di]*(kk==0?d.diskDir.x:kk==1?d.diskDir.y:d.diskDir.z)/d.vtot, d.maskVDisk, relaxSrc[kk]);
            }
            if (rotor_.active) deviceAxpy(-1.0, *rotorF[kk], relaxSrc[kk]);   // rotorDiskSource: relaxSrc -= force (OF eqn -= force)
            if (stageDumpActive() && stageDumpFirstOnly(kk==0?"msrc0":(kk==1?"msrc1":"msrc2")))
                stageDump(std::string("stage_mSrc") + char('0'+kk), relaxSrc[kk]);   // OF UEqn.source() minus -V*grad(p)
            DeviceBuffer<scalar> s;
            deviceHadamard(s, dm.V, *gg[kk]);
            deviceScale(s,-1.0);
            deviceAxpy(1.0,relaxSrc[kk],s);   // s = -V*grad(p) + relaxSrc (incl. body force)
            // rotational cyclic deferred split (PREDICTOR only): the diag-implicit off-diagonal (ifCoeffC = ifCoeff.
            // forwardT[kk][kk]) plus this off-diagonal MIXING term reconstructs the full-rotation coupling -ifCoeff.
            // (forwardT.U[nbr])[kk] in the predictor RHS, but with a consistent (diagonal-dominant) matrix that
            // converges where OF's full-implicit limit-cycles in cf's segregated SIMPLE. Built from the U_old snapshot
            // (order-independent). H() below uses the FULL rotation on the post-solve U (OF-faithful), NOT this term.
            if (hasCyclic_ && cyc_.rotational) interfaceAddDeferredRot(cyc_, Usnap[0], Usnap[1], Usnap[2], kk, s);
            if (hasAMI_ && ami_.rotational)    interfaceAddDeferredRot(ami_, Uint[0], Uint[1], Uint[2], kk, s);
            DeviceBuffer<scalar> diagC,b;
            deviceFold(dm, mDiagR, s, iC[kk], bCb[kk], diagC, b);
            // The diagonal AFTER folding in the boundary internal coefficients -- the only form
            // comparable to OpenFOAM's fvMatrix::D(), which is diag() + addCmptAvBoundaryDiag(). The
            // pre-fold dump (stage_mDiagR) omits them, so near-wall cells could not be compared against
            // the oracle at all: every boundary-adjacent cell showed a difference that was the missing
            // term rather than a defect. Dumped per component because brae keeps the boundary diagonal
            // per component (slip/symmetry give vf_k = |n_k|, so the three differ); OF's D() is their
            // component average, which is what the comparison has to form.
            const char* mdfTag = (kk == 0) ? "mDiagFold0" : (kk == 1) ? "mDiagFold1" : "mDiagFold2";
            if (stageDumpActive() && stageDumpFirstOnly(mdfTag))
                stageDump(std::string("stage_") + mdfTag, diagC);
            // ...and the matching RHS. deviceFold produces both halves together (diagC gets the boundary
            // internalCoeffs, b gets the boundaryCoeffs), so dumping only the diagonal leaves the other
            // half of the boundary treatment unmeasurable. OF's counterpart is
            // UEqn.source() + addBoundarySource(), which the oracle dumps as stage_mSrc + stage_mBndSrc.
            const char* mbfTag = (kk == 0) ? "mRhsFold0" : (kk == 1) ? "mRhsFold1" : "mRhsFold2";
            if (stageDumpActive() && stageDumpFirstOnly(mbfTag))
                stageDump(std::string("stage_") + mbfTag, b);
            // rotational: the implicit off-diagonal for component kk is scaled by forwardT[kk][kk] (OF transformCoupleField).
            const scalar* mIfc = !hasCyclic_ ? nullptr : (cyc_.rotational ? cyc_.ifCoeffC[kk].data() : cyc_.ifCoeff.data());
            const DeviceLduView mv = hasCyclic_
                ? deviceLduViewCyclic(dm,diagC,mUp,mLo, cyc_.n, cyc_.ownCell.data(), cyc_.nbrCell.data(), mIfc)
                : hasAMI_
                ? deviceLduViewAmi(dm,diagC,mUp,mLo, ami_.n, ami_.ownCell.data(), ami_.off.data(), ami_.nbrCell.data(), ami_.weight.data(),
                                   ami_.rotational ? ami_.ifCoeffC[kk].data() : ami_.ifCoeff.data())
                : deviceLduView(dm,diagC,mUp,mLo);
            // OF UEqn.H: `if (pimple.momentumPredictor()) { solve(UEqn == -fvc::grad(p)); }`. Everything
            // above this point is the matrix ASSEMBLY, which rAU/HbyA need either way; only the solve is
            // conditional. Skipping it leaves U exactly as the previous corrector left it, which is what
            // the case asked for.
            if (!ctl_.momentumPredictor) continue;
            const scalar nf = deviceNormFactor(mv, Uk_[kk], b, ones_);          // OF residualControl normalisation
            // The PER-CELL momentum residual, r = b - A*psi, before the solve moves anything.
            //
            // The global initial residual is one number for the whole field, so it can say that brae's
            // momentum equation disagrees with OpenFOAM's converged state but never WHERE. Every defect
            // found in this port was found by splitting a residual by region -- the pitzDaily inlet, the
            // Spalding wall, turbineSiting's ground -- and on pipeCyclic the disagreement is 777x on Uy
            // against 17x on Ux, which is a shape worth being able to look at rather than infer.
            {
                const char* mrTag = (kk == 0) ? "mResid0" : (kk == 1) ? "mResid1" : "mResid2";
                if (stageDumpActive() && stageDumpFirstOnly(mrTag))
                {
                    DeviceBuffer<scalar> Apsi, r;
                    deviceAmul(mv, Uk_[kk], Apsi);
                    deviceCopy(r, b);
                    deviceAxpy(-1.0, Apsi, r);
                    stageDump(std::string("stage_") + mrTag, r);
                    stageDump(std::string("stage_") + mrTag + "_normFactor", std::vector<scalar>(1, nf));
                }
            }
            // OF fvVectorMatrix::solveSegregated solves each U component with the `U` lduMatrix solver (smoothSolver
            // /GaussSeidel for motorBike). Route through deviceSymGaussSeidel when fvSolution asks for it (robust on
            // anisotropic snappy cells where Jacobi-BiCGStab under-solves the loose relTol); interface LDUs keep BiCGStab.
            scalar ur;
            DeviceSolverPerf uperf;
            if (ctl_.gsU && !hasCyclic_ && !hasAMI_)
                ur = deviceSymGaussSeidel(mv, b, Uk_[kk], nf, tol, ctl_.uRelTol(), 5000, &uperf);
            else
            {
                // DILU when the case asked for it. It matters most exactly where Jacobi is weakest: on a
                // boundary-layer mesh the momentum matrix is dominated by the wall-normal coupling, and a
                // diagonal preconditioner leaves the residual error in that direction for the PIMPLE
                // outer loop to amplify (see device_dilu.cuh).
                uperf = deviceJacobiBiCGStab(mv, b, Uk_[kk], nf, tol, ctl_.uRelTol(), ctl_.uMaxIter(), ctl_.bicgCheckEvery, ctl_.uMinIter(),
                                             (ctl_.diluU && dilu_.valid) ? &dilu_ : nullptr);
                ur = uperf.initialResidual;
            }
            // keep all 3 components + their final residual / nIter (OF prints Solving for Ux/Uy/Uz each iteration)
            if (kk == 0)      { res.Ux = ur; res.UxFinal = uperf.finalResidual; res.UxIters = uperf.nIterations; }
            else if (kk == 1) { res.Uy = ur; res.UyFinal = uperf.finalResidual; res.UyIters = uperf.nIterations; }
            else              { res.Uz = ur; res.UzFinal = uperf.finalResidual; res.UzIters = uperf.nIterations; }
        }
        // meanVelocityForce CONSTRAIN -- OF meanVelocityForce::constrain (meanVelocityForce.C:246-247):
        //     gradP0_ += dGradP_;  dGradP_ = 0.0;
        // once per momentum-matrix build, AFTER addSup has already used gradP0_ + dGradP_ above (OF adds
        // the source while constructing UEqn and constrains it immediately afterwards, so the two agree).
        //
        // This is what bounds the accumulation. correct() may run several times against one matrix -- OF
        // calls it after the momentum predictor and again after U = HbyA - rAU*grad(p) -- and each call
        // OVERWRITES dGradP_, so only the last increment before this line is ever banked. Without it the
        // driving gradient compounds every corrector and the channel diverges rather than holding Ubar.
        if (mvfActive_) { mvfGradP0_ += mvfDGradP_; mvfDGradP_ = 0; }
    }

    void DeviceSimpleSolver::correctPressureVelocity(DeviceSimpleResidual& res)
    {
        // Put the momentum interface coefficient back before H() is built: a previous corrector's pressure
        // assembly leaves the LAPLACIAN coefficient in the same buffer (see solveMomentumPredictor).
        if (hasCyclic_ && cycIfCoeffMom_.size() == cyc_.ifCoeff.size()) deviceCopy(cyc_.ifCoeff, cycIfCoeffMom_);
        if (hasAMI_    && amiIfCoeffMom_.size() == ami_.ifCoeff.size()) deviceCopy(ami_.ifCoeff, amiIfCoeffMom_);
        const DeviceMesh& dm = dm_;
        // P0 of the pressure stage trace (BRAE_PTRACE): p as it ENTERS the pressure correction. The
        // counter advances once per SIMPLE iteration here and is read (not advanced) by the pre-limit
        // dump in rhoSimpleStep, so every stage of one iteration carries the same index.
        if (pTraceActive())
        {
            ++pTraceIt_;
            pTraceDump("P0_pEnter", pTraceIt_, dp_);
            DeviceBuffer<scalar> pb0;
            deviceBCValue(dbP_, dp_, pb0);
            pTraceDump("P0_pEnterB", pTraceIt_, pb0);
        }
        // Non-orth pressure-correction limiter (OF fv::limitedSnGrad), read by the pressure phase below: per-face caps the
        // lagged corrVec.grad(p) correction to (psi/(1-psi))*|orthogonal snGrad| on pathological high-non-orth meshes
        // (T3A: AR 3232, 43.8deg), where the once-lagged correction outruns the over-relaxed diagonal and diverges. Only
        // the pathological faces (|corr|>|orth|) are limited; well-behaved faces keep limiter==1 (full "corrected",
        // OF-accurate). psi=1.0 -> unlimited (bit-identical to the validated cases). The momentum-viscous correction is
        // stable at full strength (it is the PRESSURE correction that destabilizes on T3A), so the predictor keeps psi=1.
        const scalar nonOrthLimitP = std::getenv("BRAE_NONORTH_LIMIT") ? std::atof(std::getenv("BRAE_NONORTH_LIMIT")) : ctl_.nonOrthLimit;
        // The predictor's LDU view + grad-pointer array, rebuilt over the (now member) relaxed-matrix + gradient buffers
        // for the pressure-velocity phase below (H() at deviceMatrixH, and the HbyA non-orth grad term). The predictor
        // does not modify mUp/mLo/gx/gy/gz after filling them, so these are identical pointers -> identical results.
        const DeviceLduView Uview = deviceLduView(dm, mDiagR, mUp, mLo);
        DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };
        // OF A() = (relaxed_diag + cmptAv(internalCoeffs))/V  (fvMatrix::D() -> addCmptAvBoundaryDiag, fvMatrix::A()).
        // For component-INDEPENDENT BCs cmptAv(iC) == iC[0] (bit-identical to before). For slip/symmetry the boundary
        // diagonal is PER-COMPONENT (vf_k=|n_k|), so the SHARED rAU must use the component-AVERAGE, NOT iC[0]: iC[0]
        // misses the constrained-component diagonal (bottom symmetry iC=(0,iC_v,0): iC[0]=0 but cmptAv=iC_v/3), which
        // on high-AR cells made rAU wildly too large -> catastrophic slip blowup. Reused below as the H() cmptAv term.
        DeviceBuffer<scalar> cmptAvIC;
        if (hasSym_)
        {
            deviceCopy(cmptAvIC, iC[0]);
            deviceAxpy(1.0, iC[1], cmptAvIC);
            deviceAxpy(1.0, iC[2], cmptAvIC);
            deviceScale(cmptAvIC, 1.0/3.0);
        }
        DeviceBuffer<scalar> diagA,dumb,rAU;
        deviceFold(dm,mDiagR,zeroSrc_, hasSym_ ? cmptAvIC : iC[0], zeroBndU_,diagA,dumb);
        deviceReciprocalV(dm,diagA,rAU);
        // SIMPLEC: rAtU = 1/(1/rAU - H1) = V/max(A*1, 0.1*diagA) (A*1 = row sum = deviceAmul with ones). drAtU = rAtU - rAU.
        DeviceBuffer<scalar> rAtU, drAtU;
        if (ctl_.consistent)
        {
            DeviceBuffer<scalar> rowSum;
            deviceAmul(hasCyclic_
                ? deviceLduViewCyclic(dm, diagA, mUp, mLo, cyc_.n, cyc_.ownCell.data(), cyc_.nbrCell.data(), cyc_.ifCoeff.data())
                : deviceLduView(dm, diagA, mUp, mLo), ones_, rowSum);
            // ...AND THE cyclicAMI INTERFACE. OF fvMatrix::H1() adds boundaryCoeffs for EVERY coupled patch:
            //     H1_ = lduMatrix::H1();  forAll(psi_.boundaryField(), patchi)
            //         if (ptf.coupled() && ptf.size()) addToInternalField(..., boundaryCoeffs_[patchi].component(0), H1_);
            // so an AMI face belongs in the row sum exactly as a cyclic face does. The line above took the
            // cyclic interface and stopped there, so on a mesh coupled ONLY by an AMI the row sum was the
            // internal faces alone and rAtU = V/max(A*1, 0.1*diagA) came out too large at every interface
            // cell -- and rAtU sets both the pressure Laplacian coefficient and the SIMPLEC flux correction,
            // so the error lands on the pressure equation, not the momentum one.
            //
            // MEASURED on the basic/simpleFoam/implicitAMI tutorial, brae seeded with OpenFOAM's own
            // converged fields: the momentum residual was 2.8e-07 -- brae's UEqn agrees, and its interface
            // off-diagonal matches OpenFOAM's boundaryCoeffs to every digit -- while the pressure residual
            // was 3.4e-02 with a continuity error of 0.128. Converged, U was 8.3e-02 from OpenFOAM with
            // SIMPLEC on and 1.8e-03 with it off, which is what named the term.
            //
            // ifCoeff, not ifCoeffC: OF takes boundaryCoeffs.component(0), and a rotational interface
            // carries its transform in updateInterfaceMatrix rather than in the coefficient, so the scalar
            // coefficient is the one H1 sums. deviceAmiAmul contributes ifCoeff[i]*sum_k w_k -- the
            // coverage-weighted row sum, which is the AMI's `1` -- matching the cyclic view's ifCoeff*1.
            if (hasAMI_) deviceAmiAmul(ami_, ones_, rowSum);
            deviceSimplecRAtU(dm, rowSum, diagA, rAtU);
            deviceCopy(drAtU, rAtU);
            deviceAxpy(-1.0, rAU, drAtU);
            // OF rhoSimpleFoam/pcEqn.H weights the SIMPLEC flux correction by density:
            //     phiHbyA += fvc::interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)*magSf
            // (incompressible simpleFoam/pEqn.H has no rho and is the expression below without this).
            // The HbyA correction on the other hand is NOT weighted in either: HbyA -= (rAU - rAtU)*grad(p),
            // so only drAtU used for the FLUX is scaled -- keep an unscaled copy for that.
            if (compressible_)
            {
                DeviceBuffer<scalar> t;
                deviceHadamard(t, th_.rho, drAtU);
                deviceCopy(drAtUFlux_, t);
            }
            else deviceCopy(drAtUFlux_, drAtU);
        }
        else
            deviceCopy(rAtU, rAU);
        if (limUActive_) deviceFvoLimitVelocity(limUCells_, limUMax_, Uk_[0], Uk_[1], Uk_[2]);   // limitVelocity on the predictor U (OF fvOptions.correct) -> feeds H()/HbyA
        // slip/symmetry H() consistency term: OF UEqn.H() adds (cmptAv(iC) - iC_cmpt)*psi_cmpt per component, where
        // cmptAv = mean of the 3 boundary internalCoeffs (fvMatrix::H + addCmptAvBoundaryDiag). Zero for every
        // component-independent BC (iC_cmpt == cmptAv), nonzero ONLY for slip (per-component vf=|n_k|), it supplies
        // the HbyA<->rAU consistency that a single scalar rAU otherwise breaks at a non-axis-aligned wall.
        DeviceBuffer<scalar>& cmptAvH = cmptAvIC;   // same cmptAv(iC) already computed for the rAU diagonal above
        DeviceBuffer<scalar> HbyA[3];
        {
            DeviceBuffer<scalar> Hk[3];
            // cyclicAMI H: precompute the AMI-interpolated (rotated) neighbour U once for the vector, then H[own] -=
            // ifCoeff*UNbr[kk]/V per component (= OF UEqn.H() with the AMI weighted stencil).
            DeviceBuffer<scalar> UNx, UNy, UNz;
            if (hasAMI_) deviceAmiInterpolateVec(ami_, Uk_[0], Uk_[1], Uk_[2], UNx, UNy, UNz);
            DeviceBuffer<scalar>* UN[3] = { &UNx, &UNy, &UNz };
            for (int kk = 0; kk < 3; ++kk)
            {
                DeviceBuffer<scalar> bdH;   // slip: bdDiag = cmptAv(iC) - iC[kk] (OF H() term); else zero (bit-identical)
                if (hasSym_)
                {
                    deviceCopy(bdH, cmptAvH);
                    deviceAxpy(-1.0, iC[kk], bdH);
                }
                deviceMatrixH(Uview, dm, Uk_[kk], relaxSrc[kk], hasSym_ ? bdH : zeroBndU_, bCb[kk], Hk[kk]);
                if (hasCyclic_ && !cyc_.rotational) interfaceAddH(cyc_, Uk_[kk], dm.V, Hk[kk]);   // translational off-diag
                if (hasAMI_) interfaceAddH(ami_, *UN[kk], dm.V, Hk[kk]);
            }
            // rotational cyclic H: the FULL rotation -ifCoeff.(forwardT.U[nbr])[kk] on the POST-solve U, evaluated once
            // for the whole vector (matches OF UEqn.H(), which re-evaluates patchNeighbourField from the new U). The
            // deferred mixing fed only the PREDICTOR; H is the exact full-rotation explicit coupling, consistent across
            // components, so HbyA/phiHbyA carry the true rotated neighbour momentum, not a per-component-stale value.
            if (hasCyclic_ && cyc_.rotational) deviceCyclicAddHRot(cyc_, Uk_[0], Uk_[1], Uk_[2], dm.V, Hk[0], Hk[1], Hk[2]);
            for (int kk = 0; kk < 3; ++kk)
                deviceHadamard(HbyA[kk], rAU, Hk[kk]);
        }
        DeviceBuffer<scalar> phiHi;
        deviceVectorFlux(dm, HbyA[0], HbyA[1], HbyA[2], phiHi);   // flux of the ORIGINAL HbyA
        // Phase 0 stage harness: the MOMENTUM stage's outputs, at the same point OF's pcEqn.H writes them
        // (rAU/rAtU/HbyA constructed, flux taken, nothing from the pressure step applied yet).
        if (stageDumpActive() && stageDumpFirstOnly("momentum"))
        {
            stageDump("stage_rAU", rAU);
            stageDump("stage_rAtU", ctl_.consistent ? rAtU : rAU);
            stageDump3("stage_HbyA", HbyA[0], HbyA[1], HbyA[2]);
            stageDump("stage_fluxHbyA", phiHi);   // fvc::flux(HbyA), BEFORE the interpolate(rho) weighting
        }
        if (hasCyclic_)
        {
            if (cyc_.rotational) deviceCyclicFluxRot(cyc_, HbyA[0], HbyA[1], HbyA[2]);   // rotate the neighbour HbyA
            else interfaceFlux(cyc_, HbyA[0], HbyA[1], HbyA[2]);   // cyclic-face phiHbyA = interp(HbyA).Sf -> cyc_.phi
        }
        if (hasAMI_) interfaceFlux(ami_, HbyA[0], HbyA[1], HbyA[2]);   // AMI-face phiHbyA = (w*HbyA[own]+(1-w)*interp(HbyA[nbr])).Sf
        // Stage harness: phiHbyA ON the interface. This is the one quantity neither code writes per face, and
        // it is where a coupling defect shows up as a number rather than as a downstream symptom.
        if ((hasAMI_ || hasCyclic_) && stageDumpActive() && stageDumpFirstOnly("amiPhiHbyA"))
        {
            if (hasAMI_)
            {
                stageDump("stage_ami_phiHbyA", ami_.phi);
                stageDump("stage_ami_ifCoeffP", ami_.ifCoeff);   // the PRESSURE (laplacian) off-diagonal
            }
            if (hasCyclic_) stageDump("stage_cyc_ifCoeffP", cyc_.ifCoeff);
            // ...and the CYCLIC faces. These live outside both the internal-face array and the
            // DeviceBoundary, so neither existing dump sees them -- which is exactly why a stage
            // comparison showed div(phiHbyA) wrong on the interface cells while every dumped input
            // looked fine. An undumped quantity is an untested one.
            if (hasCyclic_) stageDump("stage_cyc_phiHbyA", cyc_.phi);
            stageDump("stage_ami_HbyAx", HbyA[0]);
            stageDump("stage_ami_HbyAy", HbyA[1]);
            stageDump("stage_ami_rAU", rAU);
            if (hasCyclic_) stageDump("stage_cyc_phi", cyc_.phi);
        }
        // OF's U.correctBoundaryConditions() -- which solve() runs at the end of the momentum predictor
        // and pEqn.H runs again after the velocity correction. It matters for a WEDGE because its value
        // is the ROTATED cell velocity: brae holds that as a mixed refValue computed from U, so a
        // refValue left over from before the predictor is a value for the wrong U. The error is
        // d*(U_old - U_new) with d = 0.5*(1 - cos 2th) ~ 1.9e-3, which on movingCone's first step (U
        // starting from zero) put a spurious 4e-11 through every one of the 1900 wedge faces -- summing
        // to a leak the pressure equation answered with an equal, entirely fictitious inflow at `left`.
        if (hasWedge_) deviceUpdateWedge(dbU_, Uk_[0], Uk_[1], Uk_[2]);
        DeviceBuffer<scalar> hxb,hyb,hzb;
        deviceBCValue(dbU_.comp[0],HbyA[0],hxb);
        deviceBCValue(dbU_.comp[1],HbyA[1],hyb);
        deviceBCValue(dbU_.comp[2],HbyA[2],hzb);
        // ...and on a wedge the value is HbyA's OWN rotated cell value: the blend above went through a
        // refValue built from U, which is a different field. OF's H() inherits psi's BC types and rotates
        // itself, so this is what pEqn.H's fvc::flux(HbyA) sees on the wedge planes -- and it makes that
        // flux identically zero, as the constraint requires.
        if (hasWedge_) deviceWedgeFaceValue(dbU_, HbyA[0], HbyA[1], HbyA[2], hxb, hyb, hzb);
        // constrainHbyA at mixed velocity faces (OF resets phiHbyA_b = U_b.Sf at fixesValue patches): use U_b not HbyA_b.
        if (hasMixed_) deviceConstrainMixedHbyA(dbU_, Uk_[0], Uk_[1], Uk_[2], hxb, hyb, hzb);
        // ...and the reverse at inletOutlet, which OF marks assignable and therefore does NOT constrain:
        // HbyA_b there is the zero-gradient value it was born with, NOT the clamped inlet velocity. The
        // evaluation above went through U's descriptor, which on a BACKFLOW face returns inletValue (0),
        // so phiHbyA_b would be zero on exactly the faces where OF lets the flux through. Measured on
        // pimpleFoam/RAS/TJunction, whose outlet1 reverses at step 2: p in the boundary-adjacent cell was
        // 0.54 low and the backflow velocity 3x OF's (U L2 1.0e-01).
        if (hasInletOutletU_)
        {
            DeviceBuffer<scalar> ex, ey, ez;
            deviceBCValue(dbExtrap_, HbyA[0], ex);
            deviceBCValue(dbExtrap_, HbyA[1], ey);
            deviceBCValue(dbExtrap_, HbyA[2], ez);
            deviceExtrapolateIOHbyA(dbU_, ex, ey, ez, hxb, hyb, hzb);
        }
        // slip/symmetry: OF constrainHbyA sets HbyA_b = U_b (= U.boundaryField, the slipped velocity) at every
        // non-assignable patch, NOT the projected HbyA. Project the cell U (HbyA_b = U_c - n(n.U_c) = U_b), matching
        // OF byte-for-byte and the mixed path above (both use U_b). Wall flux is still exactly 0 (normal removed).
        if (hasSym_) deviceConstrainSymmetryHbyA(dbU_, Uk_[0], Uk_[1], Uk_[2], hxb, hyb, hzb);
        if (stageDumpActive() && stageDumpFirstOnly("HbyAb")) stageDump3("stage_HbyAb", hxb, hyb, hzb);
        DeviceBuffer<scalar> phiHb;
        deviceBoundaryFlux(dm,hxb,hyb,hzb,phiHb);
        // ---- fvc::ddtCorr, pEqn.H line 2 ----------------------------------------------------------
        //     if (pimple.ddtCorr()) phiHbyA += MRF.zeroFilter(fvc::interpolate(rAU)*fvc::ddtCorr(U, phi, Uf));
        //
        // brae had no ddtCorr at all. It is a ~1% correction on almost every internal face (measured on
        // pimpleFoam/RAS/oscillatingInletACMI2D: 3.5% of the peak flux, 0.31% rms, on 20924 of 21464
        // faces), and it was the entire difference between agreeing with OpenFOAM to 1.5e-03 and to
        // 1.1e-02 on that case once the cyclicACMI defects were out of the way.
        //
        // IT IS NOT AN INTERFACE TERM. OF zeroes its coefficient on every cyclicAMI patch, so a coupled
        // face gets nothing from it in either code -- which is why chasing the ACMI flux profile kept
        // landing on the inlet instead of the interface.
        //
        // Uf: IMPLEMENTED, and not here. pimpleFoam passes fvc::ddtCorr(U, phi, Uf), which dispatches to
        // the Uf form on a dynamic mesh -- phiCorr uses (Sf & Uf.oldTime()), the CURRENT face areas
        // against the OLD face velocity, in place of the stored phi.oldTime(). brae supplies that by
        // BUILDING phiOldInt_/phiOldBnd_ from Uf.oldTime() whenever Uf is active (see the ufActive_
        // block in moveMesh's old-field aging), so the kernel below needs no Uf argument and this term
        // is already the moving-mesh form. Leaving the old "deliberately not implemented" note here sent
        // a later audit hunting a defect that had already been fixed a thousand lines away.
        if (ddtCorr_ && ddtScheme_ != DdtScheme::steadyState && deltaT_ > 0
            && phiOldInt_.size() == phiHi.size())
        {
            // backward's fvc::ddtCorr. Transcribed from backwardDdtScheme::fvcDdtPhiCorr and
            // ::fvcDdtUfCorr, which are algebraically the same term -- they differ only in whether the
            // old flux is phi.oldTime() or (Sf & Uf.oldTime()), and brae already builds phiOld from Uf
            // on a moving mesh, so one code path serves both. 13 of OpenFOAM's 35 pimpleFoam tutorials
            // select `backward`, and brae omitted this term for every one of them.
            const bool backwardCorr = (ddtScheme_ == DdtScheme::backward)
                                   && ddtCoeffs(ddtScheme_, deltaT_, deltaT0_, ocCoeff_, cnWarm_).coefft00 > scalar(0)
                                   && phiOld2Int_.size() == phiOldInt_.size()
                                   && Uold2_[0].size() == Uold_[0].size();
            if (ddtScheme_ == DdtScheme::CrankNicolson)
            {
                // CrankNicolson's fvcDdtUfCorr carries its own off-centred ddt0 recurrence, not just an
                // extra old level, so it is not this shape. Omitted with a notice rather than run under
                // backward's or Euler's form, which would be a silent substitution.
                if (!ddtCorrNoticed_)
                {
                    ddtCorrNoticed_ = true;
                    noticeIgnored("PIMPLE/ddtCorr",
                                  "brae implements fvc::ddtCorr for the Euler and backward ddt schemes; "
                                  "this case uses CrankNicolson, whose form carries the off-centred ddt0 "
                                  "recurrence, so the correction is omitted");
                }
            }
            else if (ddtScheme_ != DdtScheme::Euler && !backwardCorr)
            {
                // backward on its FIRST transient step has no second old level yet, and OF falls back to
                // Euler there (coefft00 is 0).
                // Nothing to report: this is backward's own first-step fallback to Euler.
            }
            else
            {
                DeviceBuffer<scalar> fUoInt, fUoBnd, rAUfDdt, rAUbDdt;
                deviceVectorFlux(dm, Uold_[0], Uold_[1], Uold_[2], fUoInt);   // interpolate(U_old) & Sf
                DeviceBuffer<scalar> uob[3];
                for (int kk = 0; kk < 3; ++kk) deviceBCValue(dbU_.comp[kk], Uold_[kk], uob[kk]);
                deviceBoundaryFlux(dm, uob[0], uob[1], uob[2], fUoBnd);
                deviceInterpolate(dm, rAU, rAUfDdt);        // fvc::interpolate(rAU) -- rAU, not rAtU
                deviceBCValue(dbExtrap_, rAU, rAUbDdt);     // adjacent-cell value on the boundary
                // Stage harness: the correction ALONE, as pre/post pairs, so it can be differenced against
                // OpenFOAM's own fvc::interpolate(rAU)*fvc::ddtCorr without unpicking it from phiHbyA.
                const bool dumpDdt = stageDumpActive() && stageDumpFirstOnly("ddtCorr");
                if (dumpDdt) { stageDump("stage_ddtCorr_pre", phiHi); stageDump("stage_ddtCorr_preB", phiHb); }
                // backward's fvcDdtUfCorr (backwardDdtScheme.C): the corrected quantity is
                //     (coefft0*phi_o - coefft00*phi_oo) - (coefft0*fluxU_o - coefft00*fluxU_oo)
                // while the coupling coefficient stays on the SINGLE old level. coefft0/coefft00 are the
                // same pair the implicit ddt source already uses, rebuilt here from the same ddtCoeffs() call.
                DeviceBuffer<scalar> effInt, effBnd, fUo2Int, fUo2Bnd;
                if (backwardCorr)
                {
                    deviceVectorFlux(dm, Uold2_[0], Uold2_[1], Uold2_[2], fUo2Int);
                    DeviceBuffer<scalar> u2b[3];
                    for (int k = 0; k < 3; ++k) deviceBCValue(dbU_.comp[k], Uold2_[k], u2b[k]);
                    deviceBoundaryFlux(dm, u2b[0], u2b[1], u2b[2], fUo2Bnd);
                    const DdtCoeffs dc = ddtCoeffs(ddtScheme_, deltaT_, deltaT0_, ocCoeff_, cnWarm_);
                    const scalar c0 = dc.coefft0, c00 = dc.coefft00;
                    deviceCopy(effInt, phiOldInt_);   deviceScale(effInt, c0);
                    deviceAxpy(-c00, phiOld2Int_, effInt);
                    deviceAxpy(-c0,  fUoInt,      effInt);
                    deviceAxpy(+c00, fUo2Int,     effInt);
                    deviceCopy(effBnd, phiOldBnd_);   deviceScale(effBnd, c0);
                    deviceAxpy(-c00, phiOld2Bnd_, effBnd);
                    deviceAxpy(-c0,  fUoBnd,      effBnd);
                    deviceAxpy(+c00, fUo2Bnd,     effBnd);
                }
                deviceDdtCorrFlux(dm.nInternalFaces, phiOldInt_, phiOldBnd_, fUoInt, fUoBnd,
                                  rAUfDdt, rAUbDdt, ddtCorrBndMask_, scalar(1)/deltaT_, phiHi, phiHb,
                                  backwardCorr ? &effInt : nullptr, backwardCorr ? &effBnd : nullptr);
                // ...and on the interface. See the note on amiPhiOld_ for why this is the one coupled
                // place OF keeps the correction. deviceDdtCorrFlux's boundary sweep is elementwise, so
                // it serves here with an all-ones mask and no internal part.
                DeviceBuffer<scalar> noInt;
                auto ifDdtCorr = [&](const DeviceBuffer<scalar>& phiOld, const DeviceBuffer<scalar>& fUo,
                                     const DeviceBuffer<scalar>& rAUface, DeviceBuffer<scalar>& phi)
                {
                    if (phi.size() == 0 || phiOld.size() != phi.size() || fUo.size() != phi.size()) return;
                    if (ddtCorrIfOnes_.size() != phi.size())
                        ddtCorrIfOnes_.copyFrom(std::vector<scalar>(phi.size(), scalar(1)));
                    deviceDdtCorrFlux(0, noInt, phiOld, noInt, fUo, noInt, rAUface,
                                      ddtCorrIfOnes_, scalar(1)/deltaT_, noInt, phi);
                };
                if (hasAMI_)
                {
                    DeviceBuffer<scalar> rAUfAmi;
                    deviceAmiFaceValue(ami_, rAU, rAUfAmi);       // fvc::interpolate(rAU) on the interface
                    if (dumpDdt) stageDump("stage_ddtCorr_amiPre", ami_.phi);
                    ifDdtCorr(amiPhiOld_, amiFluxUold_, rAUfAmi, ami_.phi);
                    if (dumpDdt) stageDump("stage_ddtCorr_amiPost", ami_.phi);
                }
                if (dumpDdt)
                {
                    stageDump("stage_ddtCorr_post", phiHi);
                    stageDump("stage_ddtCorr_postB", phiHb);
                    stageDump("stage_ddtCorr_phiOld", phiOldInt_);
                    stageDump("stage_ddtCorr_fluxUold", fUoInt);
                    stageDump("stage_ddtCorr_rAUf", rAUfDdt);
                }
            }
        }
        // OF pcEqn.H line 1: `rho = thermo.rho();` -- the SIMPLEC/transonic path REFRESHES the solver's
        // rho at the TOP, before phiHbyA is built, and again at the bottom. pEqn.H (subsonic) does NOT:
        // it only assigns rho at the end. Counted directly in the two files: pcEqn.H has the statement in
        // its first three lines, pEqn.H has none.
        //
        // brae had no top-of-file refresh on either path, so phiHbyA was weighted with the PREVIOUS
        // iteration's rho. That was masked while thermo.correct() happened to overwrite rho mid-iteration;
        // separating the solver and thermo densities exposed it. Measured on squareBend at iteration 1 with
        // every linear system at machine tolerance: HbyA exact to 7.5e-07 but phiHbyA 1.68e-02 -- the only
        // quantity that would not come down with solver tolerance.
        if (ctl_.consistent) deviceThermoRho(th_, dp_, tc_, th_.rho);
        if (compressible_)
        {
            // OF rhoSimpleFoam pEqn.H: phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA), i.e. a MASS flux.
            // Both halves must be weighted: deviceInterpolate covers internal faces only, and the boundary
            // rho comes from the boundary p and T rather than the adjacent cell -- see
            // deviceThermoRhoBoundary for why extrapolating there is silently wrong at a fixed-T inlet.
            DeviceBuffer<scalar> rhoF;
            deviceInterpolate(dm, th_.rho, rhoF);
            DeviceBuffer<scalar> tmp;
            deviceHadamard(tmp, rhoF, phiHi);
            deviceCopy(phiHi, tmp);

            // OF: fvc::interpolate(rho) on a patch is rho.boundaryField()[patchi] -- the SOLVER's rho
            // field, assigned once per outer iteration at the end of pEqn.H and then RELAXED. Rebuilding
            // it here from the current p_b and T_b bypasses that relaxation exactly as the cell-side
            // thermo.correct() did. Measured on aerofoilNACA0012 (rho relax 0.01) at iteration 2: the
            // unweighted boundary flux(HbyA) matched OF to 1.7e-04, but the flux-weighted density was
            // 1.313 against OF's 1.168, making the inlet mass influx 12.4% too large. That imbalance is
            // the pressure equation's compatibility condition, and this system is nearly pure Neumann
            // (the whole boundary anchors it with sum|iC| ~ 2e-3), so a 7% error in sum(pSrc) moved the
            // pressure LEVEL by tens of kPa per iteration and pinned p at the pMax ceiling.
            DeviceBuffer<scalar> rhoB;
            if (rhoBndP_.size() == static_cast<std::size_t>(dbP_.n)) deviceCopy(rhoB, rhoBndP_);
            else
            {
                deviceThermoTBoundary(dbP_, dp_, dbHe_, th_.he, tc_, &th_.T, &TFixMask_, &TFixBnd_, TBnd_);
                deviceThermoRhoBoundary(dbP_, dp_, TBnd_, tc_, rhoB);
            }
            if (stageDumpActive() && stageDumpFirstOnly("rhoBphi")) stageDump("stage_rhoBphi", rhoB);
            DeviceBuffer<scalar> tmpB;
            deviceHadamard(tmpB, rhoB, phiHb);
            deviceCopy(phiHb, tmpB);
        }
        // The REAL phiHbyA = interpolate(rho)*flux(HbyA), i.e. OF's phiHbyA at the top of pEqn.H. The
        // separate stage_fluxHbyA above is the UNWEIGHTED flux -- comparing that against OF's phiHbyA
        // shows a clean 1/rho scale factor and looks exactly like a defect. It is not one.
        if (stageDumpActive() && stageDumpFirstOnly("phiHbyA")) stageDump("stage_phiHbyA", phiHi);
        if (stageDumpActive() && stageDumpFirstOnly("phiHbyAb")) stageDump("stage_phiHbyAb", phiHb);
        if (mrf_.active) deviceMrfApplyFrameFlux(mrf_, +1.0, phiHi, phiHb);   // MRF.makeRelative(phiHbyA) before pressure
        // adjustPhi (OF order: after flux(HbyA), before the SIMPLEC correction): enforce global continuity by
        // scaling the adjustable outflow when there is no pressure reference (closed/all-velocity domains).
        if (ctl_.needRef) deviceAdjustPhi(adjustMask_, phiHb, &phiHi);
        // B1 + SIMPLEC: OF's pcEqn.H builds phid from the ORIGINAL phiHbyA and subtracts
        // interp(psi*p)*phiHbyA/interp(rho) using that ORIGINAL too -- the SIMPLEC term is ADDED beside it,
        // not folded in first:
        //     phiHbyA += interp(rho*(rAtU-rAU))*snGrad(p)*magSf - interp(psi*p)*phiHbyA/interp(rho);
        // brae applies its SIMPLEC correction to phiHi below, so without this snapshot both the phid and
        // the subtraction would use the already-corrected flux. That is exactly why squareBend (consistent
        // yes) diverged with contGlobal -283 at iteration 1 while a non-SIMPLEC duct was fine.
        DeviceBuffer<scalar> phiHi0, phiHb0;
        if (compressible_ && ctl_.transonic) { deviceCopy(phiHi0, phiHi); deviceCopy(phiHb0, phiHb); }
        if (ctl_.consistent)
        {
            // phiHbyA += interpolate(rAtU-rAU)*snGrad(p)*magSf = laplacian(interp(drAtU), p).flux()  (NOT via HbyA flux)
            DeviceBuffer<scalar> drAtUf;
            deviceInterpolate(dm, drAtUFlux_, drAtUf);   // rho*(rAtU-rAU) when compressible
            DeviceBuffer<scalar> ld, lu, ll;
            deviceLaplacianCoeffs(dm, drAtUf, ld, lu, ll, ctl_.nonOrth);   // over-relaxed nonOrthDeltaCoeffs to match OF's corrected snGrad(p) (was orthogonal dc -> cos(theta) too small on non-orth meshes)
            DeviceBuffer<scalar> fInt;
            deviceMatrixFluxInternal(deviceLduView(dm, ld, lu, ll), dp_, fInt);
            deviceAxpy(1.0, fInt, phiHi);
            // OF's term is interp(rAtU-rAU)*fvc::snGrad(p)*magSf, and snGrad(p) is the CORRECTED snGrad (orth + non-orth).
            // The orth part is the laplacian flux above; add the non-orth part interp(drAtU)*(corrVec.grad(p)_f)*magSf so
            // the SIMPLEC flux correction is complete on non-orthogonal meshes (no-op when orthogonal: corrVec~=0).
            if (ctl_.nonOrth)
            {
                DeviceBuffer<scalar> ffcS;
                deviceLaplacianCorrFlux(dm, drAtUf, gx, gy, gz, ffcS);
                deviceAxpy(1.0, ffcS, phiHi);
            }
            DeviceBuffer<scalar> dIC, dBC;
            deviceBCLaplacianCoeffs(dbP_, drAtUFlux_, dIC, dBC);
            DeviceBuffer<scalar> fBnd;
            deviceMatrixFluxBoundary(dbP_, dIC, dBC, dp_, fBnd);
            deviceAxpy(1.0, fBnd, phiHb);
            // ...AND ON THE COUPLED INTERFACE FACES, which neither call above reaches:
            // deviceMatrixFluxInternal walks the internal faces and deviceMatrixFluxBoundary the
            // NON-coupled patches, and a cyclic or cyclicAMI face is in neither list. OpenFOAM has no such
            // gap -- `fvc::interpolate(rAtU - rAU)*fvc::snGrad(p)*mesh.magSf()` is a surfaceScalarField
            // whose COUPLED boundary values are evaluated like every other patch's, so the SIMPLEC
            // correction reaches the interface flux there.
            //
            // Without it the interface flux is inconsistent with the pressure equation about to be
            // assembled from the same rAtU, and it shows up as a continuity error on exactly the interface
            // cells. Measured on the basic/simpleFoam/implicitAMI tutorial with brae seeded at OpenFOAM's
            // own converged fields -- where the true divergence is zero -- the continuity error was 0.128
            // while the momentum residual was 2.8e-07, which is what pointed at the pressure side.
            //
            // THE DIFFUSIVITY IS NEGATED because the primitive reused here subtracts: interfaceCorrectFlux
            // is `phi -= ifCoeff*(interp(p) - p[own])`, the POST-solve pressure correction, whereas OF's
            // SIMPLEC term ADDS that same laplacian flux. Passing -drAtU turns the one into the other
            // exactly, and reuses a coefficient path that is already gated instead of adding a second one.
            if (hasCyclic_ || hasAMI_)
            {
                DeviceBuffer<scalar> negDrAtU;
                deviceCopy(negDrAtU, drAtUFlux_);
                deviceScale(negDrAtU, -1.0);
                DeviceBuffer<scalar> scratch;
                scratch.copyFrom(std::vector<scalar>(nC_, 0.0));
                // ifCoeff is ONE buffer shared by the momentum and pressure assemblies (see the note where
                // cycIfCoeffMom_/amiIfCoeffMom_ are taken), so borrow it and put it back.
                if (hasCyclic_)
                {
                    DeviceBuffer<scalar> keep;
                    deviceCopy(keep, cyc_.ifCoeff);
                    interfaceAssembleLaplacian(cyc_, negDrAtU, scratch, false);
                    interfaceCorrectFlux(cyc_, dp_);
                    if (ctl_.nonOrth)   // snGrad(p) is the CORRECTED snGrad; add its non-orth half too
                    {
                        DeviceBuffer<scalar> ffcS;
                        interfaceLapCorrP(cyc_, negDrAtU, gx, gy, gz, scratch, ffcS);
                        deviceAxpy(-1.0, ffcS, cyc_.phi);
                    }
                    deviceCopy(cyc_.ifCoeff, keep);
                }
                if (hasAMI_)
                {
                    DeviceBuffer<scalar> keep;
                    deviceCopy(keep, ami_.ifCoeff);
                    interfaceAssembleLaplacian(ami_, negDrAtU, scratch, false);
                    interfaceCorrectFlux(ami_, dp_);
                    if (ctl_.nonOrth)
                    {
                        DeviceBuffer<scalar> ffcS;
                        interfaceLapCorrP(ami_, negDrAtU, gx, gy, gz, scratch, ffcS);
                        deviceAxpy(-1.0, ffcS, ami_.phi);
                    }
                    deviceCopy(ami_.ifCoeff, keep);
                }
            }
            // HbyA adjustment AFTER the flux (feeds the velocity corrector only): HbyA -= (rAU-rAtU)*grad(p)
            for (int kk = 0; kk < 3; ++kk)
            {
                DeviceBuffer<scalar> t;
                deviceHadamard(t, drAtU, *gg[kk]);
                deviceAxpy(1.0, t, HbyA[kk]);
            }
        }
        DeviceBuffer<scalar> rAUf;
        // rho*rAtU. Needed BOTH for the internal laplacian coefficient and for the BOUNDARY ones, so it
        // is built outside the branch rather than local to it -- see the boundary call below.
        DeviceBuffer<scalar> rhoRAtU_p;
        if (compressible_)
        {
            // OF rhoSimpleFoam pEqn.H: rhorAUf = fvc::interpolate(rho*rAU). The laplacian coefficient is
            // the ONLY change the subsonic pressure equation needs -- there is no psi*p term outside the
            // transonic branch, so the system stays symmetric and the AMG-PCG path is unchanged.
            deviceHadamard(rhoRAtU_p, th_.rho, rAtU);
            deviceInterpolate(dm, rhoRAtU_p, rAUf);
        }
        else
        {
            deviceInterpolate(dm,rAtU,rAUf);
        }
        // The pressure laplacian's FACE diffusivity, OF's rhorAUf = fvc::interpolate(rho*rAU).
        if (stageDumpActive() && stageDumpFirstOnly("rhorAUf")) stageDump("stage_rhorAUf", rAUf);
        if (stageDumpActive() && stageDumpFirstOnly("rhorAU"))
        {
            if (compressible_) stageDump("stage_rhorAU", rhoRAtU_p);
            stageDump("stage_rhoP", th_.rho);
            stageDump("stage_rAUp", rAtU);
        }
        // B1 TRANSONIC (OF rhoSimpleFoam pEqn.H). Computed ONCE, before the non-orthogonal loop, exactly
        // where OF computes it -- phid is built from the ORIGINAL phiHbyA, before the subtraction below.
        //
        //     phid     = (interp(psi)/interp(rho)) * phiHbyA
        //     phiHbyA -= interp(psi*p) * phiHbyA / interp(rho)
        //
        // For perfectGas rho IS psi*p cell-wise, so that second line drives phiHbyA to zero and the whole
        // mass flux is carried implicitly by fvm::div(phid,p). It is written as the general expression
        // anyway, because rho is bounded (rhoMin/rhoMax) and relaxed, and under either of those rho and
        // psi*p genuinely differ -- taking the shortcut would be right only until someone set rhoMax.
        DeviceBuffer<scalar> phidI, phidB;
        if (compressible_ && ctl_.transonic)
        {
            DeviceBuffer<scalar> psiF, rhoF, ratio;
            deviceInterpolate(dm, th_.psi, psiF);
            deviceInterpolate(dm, th_.rho, rhoF);
            deviceDivide(ratio, psiF, rhoF);              // psi_f / rho_f
            deviceHadamard(phidI, ratio, phiHi0);         // phid = (psi_f/rho_f) * phiHbyA_ORIGINAL

            // Boundary: psi_b/rho_b = 1/p_b exactly, because deviceThermoRhoBoundary builds
            // rho_b = p_b/(R T_b) from that same relation -- no second boundary thermo evaluation.
            // Computed BEFORE phiHb is zeroed below, matching OF's ordering (phid uses the original flux).
            DeviceBuffer<scalar> pBndV;
            deviceBCValue(dbP_, dp_, pBndV);
            deviceDivide(phidB, phiHb0, pBndV);

            DeviceBuffer<scalar> psip, psipF, fac, tmp;
            deviceHadamard(psip, th_.psi, dp_);
            deviceInterpolate(dm, psip, psipF);
            deviceDivide(fac, psipF, rhoF);               // interp(psi*p)/interp(rho)
            deviceHadamard(tmp, phiHi0, fac);             // the subtraction also uses the ORIGINAL flux
            deviceAxpy(-1.0, tmp, phiHi);                 // phiHbyA -= interp(psi*p)*phiHbyA_orig/interp(rho)
            // At a boundary face psi_b*p_b IS rho_b, so that factor is exactly 1 and the ORIGINAL boundary
            // flux cancels itself: phiHb -= phiHb0. Under SIMPLEC that leaves whatever the SIMPLEC boundary
            // correction added, which is what OF keeps too -- zeroing it outright would discard that term.
            deviceAxpy(-1.0, phiHb0, phiHb);

            if (stageDumpActive())
            {
                stageDump("stage_phiHbyA0", phiHi0);   // pre-SIMPLEC, pre-subtraction: what phid is built from
                stageDump("stage_phid", phidI);
                stageDump("stage_phidB", phidB);
            }
        }

        // pressure matrix buffers are PERSISTENT members: their addresses stay fixed across SIMPLE steps (only the
        // values change), so the V-cycle CUDA graph (keyed on diagCp_) is captured once and replayed every step.
        DeviceBuffer<scalar> pPrev;
        deviceCopy(pPrev, dp_);   // p entering the pressure step (for relaxation), captured once
        // nNonOrthogonalCorrectors (OF SIMPLE, pEqn.H `while (simple.correctNonOrthogonal())`): re-solve
        // laplacian(rAtU,p) == div(phiHbyA) (nNonOrth+1) times, recomputing the EXPLICIT non-orthogonal correction
        // (corrVec.grad(p)) from the UPDATED p each pass; the conservative flux is committed only on the final pass.
        // Orthogonal scheme -> corrVec = 0 so the correction vanishes and one pass suffices (identical to nNonOrth=0).
        // The pEqn is fully re-assembled each pass, exactly as OF rebuilds the fvMatrix (deviceFoldPressure WRITES
        // diagCp_/bp_, and setReference doubles a FRESH diag, so per-pass re-assembly is self-consistent). Unchanged
        // for SIMPLE vs SIMPLEC: rAtU is just the diffusivity (= rAU for SIMPLE, = 1/(1/rAU - H1) for SIMPLEC); the
        // one-time SIMPLEC phiHbyA/HbyA correction above already used the entry p, as in OF (it is outside this loop).
        const int nPass = (ctl_.nonOrth ? ctl_.nNonOrth : 0) + 1;
        DeviceBuffer<scalar> pIC, pBC, ffcP, ffcPcyc, ffcPami;
        DeviceSolverPerf pp, pp0;
        // The other half of OF's finalInnerIter(): `corrNonOrtho_ == nNonOrthCorr_ + 1`. The caller has
        // already decided whether this is the final pressure corrector; only its LAST non-orth pass gets
        // pFinal, the earlier passes being intermediate solves of the same corrector. Restored on the way
        // out so the steady caller, which never sets it, is unaffected.
        const bool finalInnerCorrector = ctl_.finalInner;
        for (int nonOrthPass = 0; nonOrthPass < nPass; ++nonOrthPass)
        {
            ctl_.finalInner = finalInnerCorrector && (nonOrthPass == nPass - 1);
            if (nonOrthPass > 0)   // recompute grad(p) from the just-solved p (pass 0 uses the entry grad in gx/gy/gz)
            {
                DeviceBuffer<scalar> pbvN;
                deviceBCValue(dbP_, dp_, pbvN);
                deviceGaussGrad(dm, dp_, pbvN, gx, gy, gz);
                if (hasCyclic_) interfaceAddGrad(cyc_, dp_, dm.V, gx, gy, gz);
                if (hasAMI_)    interfaceAddGrad(ami_, dp_, dm.V, gx, gy, gz);
            }
            deviceLaplacianCoeffs(dm, rAUf, pD_, pU_, pL_, ctl_.nonOrth);
            // OF: `- fvm::laplacian(rhorAtU, p)` with `rhorAtU = rho*rAtU` (pcEqn.H:13), and an fvMatrix's
            // internalCoeffs/boundaryCoeffs are built from the SAME diffusivity as its internal
            // coefficients. brae passed the unweighted rAtU here while the internal coefficients used
            // interpolate(rho*rAtU) -- so every pressure boundary coefficient was too large by 1/rho.
            // Measured on squareBend's fixedValue outlet: iC and bC both 2.6154x OF's, against
            // 1/rho = 1/0.3823 = 2.6157. The inlet (zeroGradient, coefficients ~0) and the walls hid it.
            deviceBCLaplacianCoeffs(dbP_, compressible_ ? rhoRAtU_p : rAtU, pIC, pBC);
            // Stage harness: the PRESSURE boundary coefficients. Dumped AFTER the transonic div term is
            // folded in below (that is where the dump call sits), on the first non-orthogonal pass only.
            // brae's pressure matrix is the NEGATIVE of OF's (brae solves laplacian(p) = div(phiHbyA),
            // OF solves div(phiHbyA) + div(phid,p) - laplacian(p) == 0), so these compare against OF's
            // with a sign flip -- the comparison reports both so the flip is visible rather than assumed.
            // B1: + fvm::div(phid, p) in OF's arrangement. brae solves laplacian(p) = div(phiHbyA), which
            // is OF's equation multiplied by -1, so the div coefficients SUBTRACT here. Folding them into
            // pU_/pL_/pIC/pBC (rather than a side channel) also makes the flux reconstruction below correct
            // for free: deviceMatrixFluxInternal is lduMatrix::faceH over the ASSEMBLED off-diagonals,
            // exactly as OF's fvMatrix::flux() is.
            if (compressible_ && ctl_.transonic)
            {
                DeviceBuffer<scalar> dD, dU, dL, dIC, dBC;
                deviceDivUpwindCoeffs(dm, phidI, dD, dU, dL);   // fvSchemes: div(phid,p) Gauss upwind
                deviceBCDivCoeffs(dbP_, phidB, dIC, dBC);
                deviceAxpy(-1.0, dD, pD_);
                deviceAxpy(-1.0, dU, pU_);
                deviceAxpy(-1.0, dL, pL_);
                deviceAxpy(-1.0, dIC, pIC);
                deviceAxpy(-1.0, dBC, pBC);
            }
            DeviceBuffer<scalar> divPhiH;
            deviceDiv(dm, phiHi, phiHb, divPhiH);
            if (hasCyclic_) interfaceAddDiv(cyc_, dm.V, divPhiH);            // + cyclic-face flux into continuity div(phiHbyA)
            if (hasAMI_)    interfaceAddDiv(ami_, dm.V, divPhiH);               // + AMI-face flux into continuity div(phiHbyA)
            // Stage harness: div(phiHbyA) BEFORE the pEqn relaxation adds (D-D0)*p to it. The two were
            // cancelling, so only the pre-relax value shows whether div(phiHbyA) itself is wrong.
            if (stageDumpActive() && stageDumpFirstOnly("divPhiH")) stageDump("stage_divPhiH", divPhiH);
            // P1: the pressure RHS before assembly -- div(phiHbyA), internal and boundary kept apart.
            if (pTraceActive() && nonOrthPass == 0)
            {
                pTraceDump("P1_divPhiHbyA", pTraceIt_, divPhiH);
                pTraceDump("P1_phiHbyAi",   pTraceIt_, phiHi);
                pTraceDump("P1_phiHbyAb",   pTraceIt_, phiHb);
            }
            if (stageDumpActive() && stageDumpFirstOnly("pcoeff"))
            {
                stageDump("stage_pIC", pIC);
                stageDump("stage_pBC", pBC);
            }
            // B1: OF relaxes the pEqn in the TRANSONIC branch only ("Relax the pressure equation to
            // ensure diagonal-dominance"); the subsonic branch relaxes only the FIELD. fvMatrix::relax()
            // enforces D = max(|D|, sumOff)/alpha and adds (D - D0)*psi to the source -- not a no-op even
            // at alpha = 1, because the dominance clamp still applies, and that clamp is the point:
            // fvm::div(phid,p) is an upwind term whose off-diagonals can otherwise swamp the laplacian.
            //
            // ORDER MATTERS. It relaxes the RAW diagonal with the boundary internalCoeffs passed
            // SEPARATELY, then folds -- exactly as deviceSolveScalarTransport does. Relaxing the already-
            // folded diagCp_ while also passing pIC counts the boundary diagonal TWICE in the dominance
            // clamp, which is what this code did at first.
            // SIGN CONVENTION, which cost two wrong attempts before it was measured. brae's pressure
            // matrix is A = +laplacian: deviceLaplacianCoeffs gives upper = lower = +c and diag = -sum(c),
            // OF's native fvm::laplacian orientation, so brae's diagonal is NEGATIVE. OF's pEqn matrix is
            // -fvm::laplacian, POSITIVE diagonal, and relaxKernel (like fvMatrix::relax) computes
            // fmax(fabs(d0), sumOff)/alpha -- which FORCES the diagonal positive. Applied straight to
            // brae's matrix it FLIPS the sign: measured iteration-1 contGlobal -9.8e+06 against -18.5 with
            // the relaxation off entirely.
            //
            // So relax in OF's orientation and transform back. A = -M, so negate the diagonal, relax,
            // negate the result and delta. The off-diagonals are left alone because sumOff uses |.| only,
            // and delta negates the same way, which keeps the source update in the scalar transport's own
            // form (b += delta*psi).
            if (compressible_ && ctl_.transonic && ctl_.hasRelaxPEqn)
            {
                DeviceBuffer<scalar> negD, negIC, rdM, deltaM;
                DeviceBuffer<scalar>& dTerm = pRelaxSrc_;
                deviceCopy(negD, pD_);   deviceScale(negD, -1.0);
                deviceCopy(negIC, pIC);  deviceScale(negIC, -1.0);
                deviceRelaxDiag(deviceLduView(dm, negD, pU_, pL_), dm, negIC, ctl_.relaxPEqn, rdM, deltaM);
                deviceScale(rdM, -1.0);        // back to brae's orientation
                deviceScale(deltaM, -1.0);
                deviceHadamard(dTerm, deltaM, dp_);
                // dTerm is applied to bp_ AFTER the fold, NOT to divPhiH before it. foldPressureKernel
                // computes b = V*divPhi + boundaryCoeffs -- divPhiH is a PER-VOLUME divergence that the
                // fold integrates. The relaxation source (D - D0)*p is already absolute, so adding it here
                // scaled it by the cell volume and it vanished: -0.0025 * 1.5625e-08 = -3.906e-11, exactly
                // what bp_ measured at every inlet cell against OF's 0.0025.
                if (stageDumpActive() && stageDumpFirstOnly("prelax"))
                {
                    stageDump("stage_pRelaxDelta", deltaM);   // the (D - D0) the relax produced
                    stageDump("stage_pRelaxTerm", dTerm);     // and its source contribution, delta*p
                }
                deviceCopy(pD_, rdM);
            }
            // WHAT THE SIGN FIX DID, AND DID NOT, DO. It removed the divergence: squareBend (Mach 0.958,
            // transonic + SIMPLEC) went NaN by iteration 50 before, and now runs 500 stable iterations at
            // Ux ~1e-2. But the ANSWER IS STILL WRONG -- p 72% and U 264% off OF, residuals STALLING at
            // 7e-2 where OF reaches 1e-3 in 156 iterations. A stalling residual means the assembled
            // equation is not consistent with the flux/thermo update around it, so at least one defect
            // remains in this branch. Stable-and-wrong is the worst state this project has a name for,
            // which is exactly why transonic stays REFUSED by default.
            deviceFoldPressure(dm, pD_, divPhiH, pIC, pBC, diagCp_, bp_);
            if (pRelaxSrc_.size()) deviceAxpy(1.0, pRelaxSrc_, bp_);   // pEqn.relax() source, absolute -> post-fold
            // Stage harness: the assembled LINEAR SYSTEM. diagCp_ is the diagonal including the boundary
            // internalCoeffs and bp_ the full RHS -- the like-for-like counterparts of OF's D() and
            // source()+boundary. brae's matrix is the NEGATIVE of OF's, so expect a sign flip.
            if (stageDumpActive() && stageDumpFirstOnly("psystem"))
            {
                stageDump("stage_pD", diagCp_);
                stageDump("stage_pSrc", bp_);
            }
            // P2: the assembled linear system. diagCp_ includes the boundary internalCoeffs and bp_ the
            // full RHS including boundaryCoeffs -- the counterparts of OF's D() and source()+boundary.
            // brae's pressure matrix is the NEGATIVE of OF's, so the comparison flips the sign rather
            // than assuming it. pIC/pBC go out separately so a boundary-only defect is visible as one.
            if (pTraceActive() && nonOrthPass == 0)
            {
                pTraceDump("P2_pD",   pTraceIt_, diagCp_);
                pTraceDump("P2_pSrc", pTraceIt_, bp_);
                pTraceDump("P2_pIC",  pTraceIt_, pIC);
                pTraceDump("P2_pBC",  pTraceIt_, pBC);
            }
            // cyclic pressure Laplacian laplacian(rAtU,p): fold the interface diagonal into diagCp_ + set cyc_.ifCoeff.
            if (hasCyclic_) interfaceAssembleLaplacian(cyc_, rAtU, diagCp_, /*addToDiag*/true);
            if (hasAMI_)    interfaceAssembleLaplacian(ami_, rAtU, diagCp_, /*addToDiag*/true);   // AMI pressure laplacian
            // Stage harness: the assembled AMI interface, per SOURCE face. The quantity to check first is the
            // row sum: the pressure operator annihilates constants only if each row's off-diagonal sum
            // (ifCoeff*sum_k w_k) cancels its diagonal term (-ifCoeff), i.e. only if sum_k w_k == 1. That is
            // what the ACMI re-normalisation in ami_interface.cuh restores -- before it, blended faces summed
            // to their coverage and this dump was how the residual showed up as a number.
            if (hasAMI_ && stageDumpActive() && stageDumpFirstOnly("amiP"))
            {
                stageDump("stage_ami_magSf",   ami_.magSf);
                stageDump("stage_ami_dc",      ami_.deltaCoeffs);
                stageDump("stage_ami_weights", ami_.weights);
                stageDump("stage_ami_wsum",    ami_.weightsSum);
                stageDump("stage_ami_ifCoeff", ami_.ifCoeff);
                stageDump("stage_ami_w",       ami_.weight);
                const std::vector<label> oc = ami_.ownCell.host(), of = ami_.off.host(), nc = ami_.nbrCell.host();
                stageDump("stage_ami_own", std::vector<scalar>(oc.begin(), oc.end()));
                stageDump("stage_ami_off", std::vector<scalar>(of.begin(), of.end()));
                stageDump("stage_ami_nbr", std::vector<scalar>(nc.begin(), nc.end()));
                stageDump("stage_ami_rAtU", rAtU);
                stageDump("stage_ami_Sfx", ami_.Sfx);
                stageDump("stage_ami_Sfy", ami_.Sfy);
            }
            // explicit non-orth correction: faceFluxCorr from grad(p) for THIS pass; add -V*div(ffc) to b, and subtract
            // ffc from the reconstructed flux below (so div(phi)=0 on non-orth faces). gx/gy/gz = grad(p) this pass.
            ffcP = DeviceBuffer<scalar>();
            ffcPcyc = DeviceBuffer<scalar>();
            ffcPami = DeviceBuffer<scalar>();
            if (ctl_.nonOrth)
            {
                deviceLaplacianCorrFluxLimited(dm, rAUf, dp_, gx, gy, gz, nonOrthLimitP, ffcP);
                DeviceBuffer<scalar> sc;
                deviceFaceDivSource(dm, ffcP, sc);
                deviceAxpy(1.0, sc, bp_);
                if (hasCyclic_) interfaceLapCorrP(cyc_, rAtU, gx, gy, gz, bp_, ffcPcyc);
                if (hasAMI_)    interfaceLapCorrP(ami_, rAtU, gx, gy, gz, bp_, ffcPami);
            }
            // No fixedValue-p patch -> singular all-Neumann pressure. cyclic/AMI use a tiny SYMMETRIC diagonal shift
            // (the periodic RHS is zero-mean); non-cyclic keeps the OF single-cell setReference (doubles a fresh diag).
            if (ctl_.needRef)
            {
                if (hasCyclic_ || hasAMI_)
                {
                    const scalar eps = -1e-10 * (deviceSumMag(diagCp_) / nC_);
                    deviceAxpy(eps, ones_, diagCp_);
                }
                else deviceSetReference(diagCp_, bp_, ctl_.pRefCell, ctl_.pRefValue);
            }
            if (hasCyclic_ || hasAMI_)
            {
                // Periodic/AMI: interface-coupled operator solved with Jacobi-PCG (no AMG; the internal-face Galerkin
                // coarse operator cannot represent the interface edges).
                // BOTH, when the mesh has both -- see deviceLduViewCyclicAmi. A ternary here dropped one
                // interface's off-diagonal out of the matrix while its diagonal and flux stayed in.
                const DeviceLduView pvc =
                    (hasCyclic_ && hasAMI_)
                    ? deviceLduViewCyclicAmi(dm,diagCp_,pU_,pL_,
                                             cyc_.n, cyc_.ownCell.data(), cyc_.nbrCell.data(), cyc_.ifCoeff.data(),
                                             ami_.n, ami_.ownCell.data(), ami_.off.data(), ami_.nbrCell.data(),
                                             ami_.weight.data(), ami_.ifCoeff.data())
                    : hasCyclic_
                    ? deviceLduViewCyclic(dm,diagCp_,pU_,pL_, cyc_.n, cyc_.ownCell.data(), cyc_.nbrCell.data(), cyc_.ifCoeff.data())
                    : deviceLduViewAmi(dm,diagCp_,pU_,pL_, ami_.n, ami_.ownCell.data(), ami_.off.data(), ami_.nbrCell.data(), ami_.weight.data(), ami_.ifCoeff.data());
                const scalar nfp = deviceNormFactor(pvc, dp_, bp_, ones_);
                // A NON-CONFORMING AMI makes this operator non-self-adjoint (see the constructor), and CG
                // requires symmetry. Conforming interfaces -- every plain cyclic, and an AMI whose two
                // sides happen to match 1:1 -- stay on PCG, which is both faster and the path all the
                // existing cyclic cases are validated on.
                // The interface edges are now IN the agglomeration, so the coarse grids can move the
                // periodic mode and the V-cycle is a valid preconditioner here. Galerkin needs the fine
                // coefficients over the SAME extended edge list the hierarchy was built from:
                // [internal faces | interface entries], the second half gathered from the interface
                // off-diagonals. The pressure matrix is symmetric, so upper and lower agree.
                // A NON-CONFORMING AMI keeps BiCGStab (the operator is not self-adjoint), so it does not
                // use the V-cycle -- and must not pay for a Galerkin re-coarsening it will never read.
                const bool amgHere = nAmgIfEdges_ && !amiNonConforming_;
                if (amgHere)
                {
                    const int nE = static_cast<int>(nIf_ + nAmgIfEdges_);
                    amgFineUpper_.resize(nE);
                    amgFineLower_.resize(nE);
                    amgFineCoeffKernel<<<nBlocks(nE), TPB>>>(
                        (int)nIf_, (int)nAmgCycEdges_, (int)nAmgAmiEdges_,
                        pU_.data(), pL_.data(),
                        hasCyclic_ ? cyc_.ifCoeff.data() : nullptr,
                        hasAMI_    ? ami_.ifCoeff.data() : nullptr,
                        nAmgAmiEdges_ ? amgIfAmiSrc_.data() : nullptr,
                        nAmgAmiEdges_ ? amgIfAmiW_.data()   : nullptr,
                        amgFineUpper_.data(), amgFineLower_.data());
                    cudaCheck(cudaGetLastError(), "amgFineCoeff");
                    amgGalerkin(amg_, diagCp_, amgFineUpper_, amgFineLower_);
                }
                pp = amiNonConforming_
                   ? deviceJacobiBiCGStab(pvc, bp_, dp_, nfp, ctl_.pTol(), ctl_.pRelTol(), ctl_.pMaxIter(),
                                          ctl_.pcgCheckEvery, ctl_.pMinIter())
                   : amgHere
                   ? deviceAMGPCG(pvc, amg_, bp_, dp_, nfp, ctl_.pTol(), ctl_.pRelTol(), ctl_.pMaxIter(),
                                  /*captureVcycle*/false, ctl_.pcgCheckEvery, /*corrScaling*/false, ctl_.pMinIter())
                   : deviceJacobiPCG(pvc, bp_, dp_, nfp, ctl_.pTol(), ctl_.pRelTol(), ctl_.pMaxIter(), ctl_.pMinIter());
            }
            else if (compressible_ && ctl_.transonic)
            {
                // B1: the transonic pressure matrix is NOT SYMMETRIC. fvm::div(phid,p) is an upwind
                // convection term, so pU_ != pL_ -- and AMG-PCG (like any CG) requires symmetry. Left on
                // the PCG path it does not merely lose efficiency: it fails to converge and every SIMPLE
                // step burns the full 3000-iteration cap, which is how this first showed up (squareBend
                // stalled before printing iteration 1). BiCGStab handles the asymmetry; it is the same
                // solver brae already uses for every asymmetric scalar transport.
                const DeviceLduView pv = deviceLduView(dm, diagCp_, pU_, pL_);
                const scalar nfp = deviceNormFactor(pv, dp_, bp_, ones_);
                pp = deviceJacobiBiCGStab(pv, bp_, dp_, nfp, ctl_.pTol(), ctl_.pRelTol(), ctl_.pMaxIter(), ctl_.pcgCheckEvery, ctl_.pMinIter());
            }
            else
            {
                const scalar nfp = deviceNormFactor(deviceLduView(dm,diagCp_,pU_,pL_), dp_, bp_, ones_);
                amgGalerkin(amg_, diagCp_, pU_, pL_);                                      // re-coarsen the pressure matrix
                pp = deviceAMGPCG(deviceLduView(dm,diagCp_,pU_,pL_), amg_, bp_, dp_, nfp, ctl_.pTol(), ctl_.pRelTol(), ctl_.pMaxIter(), ctl_.useGraph, ctl_.pcgCheckEvery, ctl_.corrScaling, ctl_.pMinIter());
            }
            if (nonOrthPass == 0) pp0 = pp;   // SIMPLE residualControl uses the FIRST pass's initial residual (OF convention)
        }
        ctl_.finalInner = finalInnerCorrector;   // undo the per-pass narrowing (see the loop head)
        res.p = pp0.initialResidual;
        res.pFinal = pp.finalResidual;
        res.pIters = pp.nIterations;
        if (std::getenv("BRAE_SOLVER_DEBUG")) std::printf("    p %s iters=%d  init=%.2e final=%.2e\n",
                                                        (hasCyclic_||hasAMI_) ? (amiNonConforming_ ? "Jacobi-BiCGStab (non-conforming AMI)" : "Jacobi-PCG") : ((compressible_ && ctl_.transonic) ? "Jacobi-BiCGStab" : "AMG-PCG"), pp.nIterations, pp.initialResidual, pp.finalResidual);
        const DeviceLduView pview = deviceLduView(dm,diagCp_,pU_,pL_);
        DeviceBuffer<scalar> pfi;
        deviceMatrixFluxInternal(pview, dp_, pfi);
        deviceCopy(phiInt_, phiHi);
        deviceAxpy(-1.0, pfi, phiInt_);
        if (ctl_.nonOrth) deviceAxpy(-1.0, ffcP, phiInt_);   // phi -= faceFluxCorrection (continuity on non-orth faces)
        DeviceBuffer<scalar> pfb;
        deviceMatrixFluxBoundary(dbP_, pIC, pBC, dp_, pfb);
        deviceCopy(phiBnd_, phiHb);
        deviceAxpy(-1.0, pfb, phiBnd_);
        if (hasCyclic_) interfaceCorrectFlux(cyc_, dp_);   // cyc_.phi -= ifCoeff*(p_nbr - p_own) : conservative periodic flux (pre-relax dp_)
        if (hasAMI_)    interfaceCorrectFlux(ami_, dp_);       // ami_.phi -= ifCoeff*(interp(p_nbr) - p_own)
        if (hasCyclic_ && ctl_.nonOrth && ffcPcyc.size()) deviceAxpy(-1.0, ffcPcyc, cyc_.phi);   // cyclic-face non-orth flux correction (matches phiInt_ -= ffcP)
        if (hasAMI_ && ctl_.nonOrth && ffcPami.size()) deviceAxpy(-1.0, ffcPami, ami_.phi);       // AMI-face non-orth flux correction
        if (stageDumpActive() && stageDumpFirstOnly("pSolved"))
        {
            stageDump("stage_pSolved", dp_);         // straight out of the linear solver
            stageDump("stage_phiSolved", phiInt_);
            if (hasAMI_) stageDump("stage_ami_phiCorrected", ami_.phi);   // interface flux after the p correction
        }
        if (stageDumpActive() && stageDumpFirstOnly("pSolved")) stageDump("stage_pSolved", dp_);   // pre-relax, pre-limit
        // P3: straight out of the linear solve, BEFORE field relaxation -- the decisive quantity. If the
        // RHS and matrix still agree here and this does not, the defect is in the solve path (reference
        // handling, AMG-PCG, the boundary contribution in Amul, initial guess or stopping criterion)
        // rather than anywhere in the SIMPLE assembly upstream of it.
        if (pTraceActive())
        {
            pTraceDump("P3_pSolved", pTraceIt_, dp_);
            pTraceNote(pTraceIt_, "p_initialResidual", pp0.initialResidual);
            pTraceNote(pTraceIt_, "p_finalResidual",   pp.finalResidual);
            pTraceNote(pTraceIt_, "p_nIterations",     pp.nIterations);
        }
        deviceScale(dp_, ctl_.pRelax());
        deviceAxpy(1.0 - ctl_.pRelax(), pPrev, dp_);
        if (stageDumpActive() && stageDumpFirstOnly("pRelaxed")) stageDump("stage_pRelaxed", dp_);
        if (pTraceActive()) pTraceDump("P4_pRelaxed", pTraceIt_, dp_);   // after p.relax()
        DeviceBuffer<scalar> pbv2;
        deviceBCValue(dbP_, dp_, pbv2);
        if (pTraceActive()) pTraceDump("P5_pBoundary", pTraceIt_, pbv2);   // after BC evaluation
        DeviceBuffer<scalar> gnx,gny,gnz;
        deviceGaussGrad(dm, dp_, pbv2, gnx,gny,gnz);
        if (hasCyclic_) interfaceAddGrad(cyc_, dp_, dm.V, gnx,gny,gnz);   // + cyclic-face contribution to grad(p)
        if (hasAMI_)    interfaceAddGrad(ami_, dp_, dm.V, gnx,gny,gnz);       // + AMI-face contribution to grad(p)
        DeviceBuffer<scalar>* gn[3]={&gnx,&gny,&gnz};
        for (int kk = 0; kk < 3; ++kk)   // SIMPLEC: rAtU
        {
            DeviceBuffer<scalar> Un;
            deviceCorrector(HbyA[kk], rAtU, *gn[kk], Un);
            Uk_[kk]=std::move(Un);
        }
        if (hasWedge_) deviceUpdateWedge(dbU_, Uk_[0], Uk_[1], Uk_[2]);   // pEqn.H's U.correctBoundaryConditions()
        // limitVelocity (fvOptions.correct): clamp |U| <= max on the corrected (output) velocity. OF clamps after the
        // momentum predictor; cf clamps the post-corrector U so the WRITTEN field is bounded (matches OF's output).
        if (limUActive_) deviceFvoLimitVelocity(limUCells_, limUMax_, Uk_[0], Uk_[1], Uk_[2]);
        // meanVelocityForce.correct -- OF pimpleFoam pEqn.H's LAST fvOptions.correct(U), i.e. after
        // U = HbyA - rAU*grad(p), next to limitVelocity (the other fvOptions constraint applied here).
        //
        // THE POSITION IS THE POINT. This ran before H()/HbyA was built, so HbyA was formed from the
        // corrected velocity and the corrector U = HbyA - rAU*grad(p) then overwrote it -- the step
        // ended with the mean velocity wherever the pressure solve left it rather than pinned at Ubar.
        // The source kept driving to make up a deficit that the next correction had already erased, so
        // the mean climbed anyway: 0.386 against OpenFOAM's 0.129 with Ubar = 0.1335. Applying it last,
        // as OF does, is what actually holds the mean.
        // dGradP = relax*(|Ubar| - magUbarAve)/rAUave; accumulate gradP_; correct U += flowDir*rAU*dGradP. Reductions
        // and the body force/correction use maskV/mask01 so selectionMode all (whole field) and cellZone both work.
        if (mvfActive_)
        {
            const scalar magUbarAve = (mvfFlowDir_.x*deviceDot(Uk_[0], mvfMaskV_) + mvfFlowDir_.y*deviceDot(Uk_[1], mvfMaskV_)
                                     + mvfFlowDir_.z*deviceDot(Uk_[2], mvfMaskV_)) / mvfVtot_;
            const scalar rAUave = deviceDot(rAU, mvfMaskV_) / mvfVtot_;
            const scalar dGradP = mvfRelax_ * (mvfUbarMag_ - magUbarAve) / rAUave;
            // ASSIGN, never accumulate -- OF meanVelocityForce::correct sets dGradP_ outright
            // (meanVelocityForce.C:170), so a second correct() in the same matrix REPLACES the first
            // rather than adding to it. Only constrain() folds an increment into the running total, and
            // it does so once per momentum-matrix build.
            //
            // brae accumulated on every call. With OF's own call pattern -- correct(U) after the
            // momentum predictor AND again after U = HbyA - rAU*grad(p) -- that is two increments
            // banked per step where OF banks one, so the driving pressure gradient grew about twice as
            // fast as the flow could respond. The result is a feedback loop that oscillates with
            // GROWING amplitude rather than settling: measured on LES/periodicPlaneChannel, gradP went
            // 0.116 -> 0.229 -> -0.143 -> -0.382 -> 0.600 while the mean velocity it is supposed to
            // pin at Ubar = 0.1335 ran away to 1.476 -- eleven times the target, L2rel(U) = 1.00e+01
            // against OpenFOAM, which held 0.129 throughout.
            mvfDGradP_ = dGradP;
            if (std::getenv("BRAE_MVF_DEBUG")) std::printf("    mvf: magUbarAve=%.6g rAUave=%.6g dGradP=%.6g gradP=%.6g\n", magUbarAve, rAUave, dGradP, mvfGradP0_ + mvfDGradP_);
            DeviceBuffer<scalar> rAUm;
            deviceHadamard(rAUm, rAU, mvfMask01_);              // rAU in the selection, 0 else
            deviceAxpy(mvfFlowDir_.x*dGradP, rAUm, Uk_[0]);
            deviceAxpy(mvfFlowDir_.y*dGradP, rAUm, Uk_[1]);
            deviceAxpy(mvfFlowDir_.z*dGradP, rAUm, Uk_[2]);
        }
        // ...and the last line of OF's pEqn.H: make the flux this corrector produced relative to the mesh
        // motion. phiHbyA is an ABSOLUTE flux (fvc::flux(HbyA)), so phi = phiHbyA - pEqn.flux() is too;
        // the next corrector rebuilds it absolute again, so this runs per corrector exactly as OF does.
        // No-op on a static mesh.
        // pEqn.H's last two lines, in order: correct the face velocity from the ABSOLUTE flux, THEN make
        // the flux relative. Swapping them would feed Uf a relative flux and lose the mesh motion twice.
        correctUf();
        makeFluxRelative(fvp_, dm_.nInternalFaces);
    }

    DeviceSimpleResidual DeviceSimpleSolver::step()
    {
        const DeviceMesh& dm = dm_;
        DeviceSimpleResidual res;
        solveMomentumPredictor(res);
        correctPressureVelocity(res);

        correctTurbulence();
        // OF continuityErrs.H: contErr = fvc::div(phi) on the CORRECTED flux; sum local = sum(V|contErr|)/sumV,
        // global = sum(V contErr)/sumV. deviceDiv gives Sum(phi)/V per cell, so R = V*div = the cell flux balance.
        {
            DeviceBuffer<scalar> divPhi;
            deviceDiv(dm, phiInt_, phiBnd_, divPhi);
            // The interface faces are NOT in the DeviceBoundary (deviceDiv cannot see them), so their flux has to
            // be added the same way div(phiHbyA) does it for the pressure equation. Omitting it here does not
            // report a smaller error, it reports a LARGER one: the pEqn drives div(phi) INCLUDING the interface
            // to zero, so a residual that leaves the interface out measures the interface flux itself.
            if (hasCyclic_) interfaceAddDiv(cyc_, dm.V, divPhi);
            if (hasAMI_)    interfaceAddDiv(ami_, dm.V, divPhi);
            DeviceBuffer<scalar> R;
            deviceHadamard(R, divPhi, dm.V);
            const scalar sumV = deviceDot(dm.V, ones_) + 1e-300;
            res.contLocal  = deviceSumMag(R) / sumV;
            res.contGlobal = deviceDot(R, ones_) / sumV;
        }
        return res;
    }

    // Advance one time level: roll the old-time velocity fields (t-2 <- t-1 <- current) and the time steps
    // (deltaT0 <- deltaT). Call ONCE at the top of each time step, before the outer-corrector loop -- OF does this at
    // runTime++ (the registry ages every field's oldTime). First call: Uold2_ stays empty + deltaT0_=0, so ddtCoeffs()
    // bootstraps to Euler exactly like pimpleFoam's first step (only one old level exists yet).
    std::vector<scalar> DeviceSimpleSolver::moveMesh(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& fvp,
        const std::vector<vector>& oldPoints,
        const std::vector<vector>& newPoints,
        scalar deltaT)
    {
        // 0. storeOldVol(V()) -- OF fvMesh::movePoints:944, BEFORE the geometry is recomputed. The ddt
        //    source needs the volume the old-time field belongs to.
        Vold_.copyFrom(dm_.V.host());   // DeviceBuffer is non-copyable; this is storeOldVol

        // 1. geometry: only the geometric buffers, addressing untouched (OF clearGeom + clearOut).
        refreshDeviceMeshGeometry(dm_, m, g, fvp);

        // 1b. AMI weights. A sliding interface maps source faces to target faces by OVERLAP, so its
        //     weights are a function of the current geometry -- once the rotor turns they address the
        //     wrong faces. OF rebuilds them on a move (cyclicAMIPolyPatch::movePoints ->
        //     AMIInterpolation::update); brae built them once at construction and never again, which
        //     is why pimpleFoam/RAS/rotatingFanInRoom diverged with a growing continuity error while
        //     its Courant number sat at 0.46 -- flux crossing the interface was being gathered from
        //     faces that had rotated away.
        //
        //     Both halves must re-run: buildAMIInterfaces recomputes the overlap from the MOVED
        //     geometry, buildDeviceAMI re-uploads it. Doing only the upload would re-send stale weights.
        //
        //     ...but the WEIGHTS are the only thing a move invalidates. OF's phi is a surfaceScalarField
        //     and fvMesh::movePoints recomputes the AMI without touching the flux its coupled patch is
        //     carrying. buildDeviceAMI returns a FRESH DeviceAMI whose .phi is a newly allocated buffer,
        //     so assigning it wholesale threw that flux away at the top of every step after the first.
        //     ami_.phi is the interface's convection coefficient (deviceAmiAssembleMomentum takes
        //     min(phi,0)/max(phi,0) from it), so a zeroed one leaves the interface with pure diffusion
        //     and no convective coupling at all. On pimpleFoam/RAS/oscillatingInletACMI2D that showed up
        //     as step 1 matching OF to 6e-04 and step 2 putting the inlet channel 1140 above the duct --
        //     and as a RESTART curing it, because the constructor rebuilds the flux from U while a
        //     continuation has nothing left to rebuild it from.
        //     The patch sizes are fixed (only the overlap moves), so the flux stays face-for-face valid.
        if (hasAMI_)
        {
            DeviceBuffer<scalar> phiKeep = std::move(ami_.phi);
            const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);
            ami_ = buildDeviceAMI(amis);
            if (phiKeep.size() == ami_.phi.size()) ami_.phi = std::move(phiKeep);
            amiRuns_.clear();
            for (const auto& a : amis) amiRuns_.push_back({a.patch, (label)a.ownCell.size()});
        }

        // 2. meshPhi = sweptVol/deltaT, per face (OF fvMesh::movePoints stores this as mesh.phi()).
        //    KEPT, not applied here. See makeFluxRelative() for where fvc::makeRelative belongs and why
        //    doing it at the move was wrong.
        meshPhi_ = meshPhi(m, oldPoints, newPoints, deltaT);
        meshPhiValid_ = true;
        if (stageDumpActive())
        {
            stageDump("stage_meshPhi", meshPhi_);
            // ...and the interface's share of it, which is the quantity this was all about.
            std::vector<scalar> imp;
            for (const auto& r : amiRuns_)
            {
                const label st = (r.first >= 0 && r.first < (label)fvp.size()) ? fvp[r.first].start : -1;
                for (label i = 0; i < r.second; ++i)
                    imp.push_back((st >= 0 && (std::size_t)(st + i) < meshPhi_.size()) ? meshPhi_[st + i] : 0.0);
            }
            if (!imp.empty()) stageDump("stage_ami_meshPhi", imp);
        }
        return meshPhi_;
    }

    // fvc::makeRelative(phi, U): phi -= mesh.phi(). OF calls this at the END of pEqn.H, on the flux the
    // pressure corrector has just produced, using the meshPhi of the move at the top of that same step.
    //
    // WHY NOT AT THE MOVE, where brae used to do it. Two things went wrong there:
    //   - it relativised the PREVIOUS step's flux with THIS step's meshPhi, so fvm::div(phi,U) convected
    //     with phi_abs(n-1) - meshPhi(n) where OF uses phi_abs(n-1) - meshPhi(n-1)
    //   - on the FIRST step it turned the initial phi (0, from U = 0) into -meshPhi, a spurious
    //     convective flux the same order as the physical one. OF leaves phi alone at the move unless the
    //     case asks for correctPhi -- pimpleFoam.C guards that whole block with `if (correctPhi)`, and
    //     oscillatingInletACMI2D says no.
    // Measured on that case: step 1 velocity was 2.2e-02 (L2) from OpenFOAM with the whole moving channel
    // wrong by ~10%, against 4.7e-07 for the same case held static.
    //
    // AND IT INCLUDES THE INTERFACE. phi is one surfaceScalarField in OF, so makeRelative reaches the
    // coupled patches too; in brae that flux lives on cyc_/ami_.phi and was never relativised. On this
    // fixture the interface's own meshPhi is ~0 -- the channel slides IN the interface plane, so those
    // faces sweep no volume -- but a rotor whose AMI is not a surface of revolution, or any interface
    // with a normal component of motion, has a non-zero one.
    // phi +/- mesh.phi(), across every place brae keeps a flux: the internal faces, the non-coupled
    // boundary faces (phiBnd_ skips the coupled ones, like DeviceBoundary), and the interfaces.
    // sign = -1 is fvc::makeRelative, +1 is fvc::makeAbsolute.
    void DeviceSimpleSolver::applyMeshPhi(
        const std::vector<FvPatch>& fvp,
        const std::vector<scalar>& mp,
        scalar sign,
        DeviceBuffer<scalar>& fInt,
        DeviceBuffer<scalar>& fBnd,
        DeviceBuffer<scalar>& fCyc,
        DeviceBuffer<scalar>& fAmi)
    {
        if (mp.empty()) return;
        if (fInt.size())
        {
            std::vector<scalar> pi = fInt.host();
            for (std::size_t f = 0; f < pi.size() && f < mp.size(); ++f) pi[f] += sign*mp[f];
            fInt.copyFrom(pi);
        }
        if (fBnd.size())
        {
            std::vector<scalar> pb = fBnd.host();
            std::size_t b = 0;
            for (std::size_t pj = 0; pj < fvp.size(); ++pj)
            {
                if (isCoupledInterfaceType(fvp[pj].type)) continue;
                for (label i = 0; i < fvp[pj].size; ++i, ++b)
                {
                    const std::size_t f = static_cast<std::size_t>(fvp[pj].start + i);
                    if (b < pb.size() && f < mp.size()) pb[b] += sign*mp[f];
                }
            }
            fBnd.copyFrom(pb);
        }
        auto onInterface = [&](const std::vector<std::pair<label, label>>& runs, DeviceBuffer<scalar>& phi)
        {
            if (runs.empty() || phi.size() == 0) return;
            std::vector<scalar> h = phi.host();
            std::size_t off = 0;
            for (const auto& r : runs)
            {
                const label st = (r.first >= 0 && r.first < (label)fvp.size()) ? fvp[r.first].start : -1;
                for (label i = 0; i < r.second; ++i)
                {
                    const std::size_t f = static_cast<std::size_t>(st + i);
                    if (st >= 0 && off + i < h.size() && f < mp.size()) h[off + i] += sign*mp[f];
                }
                off += static_cast<std::size_t>(r.second);
            }
            phi.copyFrom(h);
        };
        onInterface(cycRuns_, fCyc);
        onInterface(amiRuns_, fAmi);
    }

    // fvc::correctUf(Uf, U, phi) -- fvcMeshPhi.C. Uf = fvc::interpolate(U) with its NORMAL component
    // replaced by the one the conservative flux implies:  Uf += n*(phi/magSf - (n & Uf)).
    // phi must be ABSOLUTE here, which is why this runs before makeFluxRelative, as in pEqn.H.
    void DeviceSimpleSolver::enableUf()
    {
        ufActive_ = true;
        const DeviceMesh& dm = dm_;
        for (int k = 0; k < 3; ++k)
        {
            // Construction is the plain interpolation -- NO flux correction. OF's Uf(fvc::interpolate(U))
            // is only made consistent with phi by the first correctUf at the end of step 1's pressure
            // corrector. Getting this wrong shows up only at step 1, where Uf.oldTime() is this value.
            deviceInterpolate(dm, Uk_[k], UfInt_[k]);
            deviceBCValue(dbU_.comp[k], Uk_[k], UfBnd_[k]);
            if (hasAMI_)    deviceAmiFaceValue(ami_, Uk_[k], UfAmi_[k]);
            if (hasCyclic_) deviceCyclicFaceValue(cyc_, Uk_[k], UfCyc_[k]);
            deviceCopy(UfOldInt_[k], UfInt_[k]);        // oldTime() bootstraps to the field itself
            deviceCopy(UfOldBnd_[k], UfBnd_[k]);
            if (hasAMI_)    deviceCopy(UfOldAmi_[k], UfAmi_[k]);
            if (hasCyclic_) deviceCopy(UfOldCyc_[k], UfCyc_[k]);
        }
    }

    void DeviceSimpleSolver::correctUf()
    {
        if (!ufActive_) return;
        const DeviceMesh& dm = dm_;
        for (int k = 0; k < 3; ++k)
        {
            deviceInterpolate(dm, Uk_[k], UfInt_[k]);            // fvc::interpolate(U), internal faces
            deviceBCValue(dbU_.comp[k], Uk_[k], UfBnd_[k]);      // ...and U_b on the ordinary patches
            if (hasAMI_)    deviceAmiFaceValue(ami_, Uk_[k], UfAmi_[k]);      // ...and the coupled ones,
            if (hasCyclic_) deviceCyclicFaceValue(cyc_, Uk_[k], UfCyc_[k]);   //    where it is the interp
        }
        deviceCorrectUf(dm.nInternalFaces, nullptr, dm.Sfx, dm.Sfy, dm.Sfz, dm.magSf, phiInt_,
                        UfInt_[0], UfInt_[1], UfInt_[2]);
        deviceCorrectUf(dm.nBndFaces, dm.bndGFace.data(), dm.Sfx, dm.Sfy, dm.Sfz, dm.magSf, phiBnd_,
                        UfBnd_[0], UfBnd_[1], UfBnd_[2]);
        if (hasAMI_)
            deviceCorrectUf(ami_.n, nullptr, ami_.Sfx, ami_.Sfy, ami_.Sfz, ami_.magSf, ami_.phi,
                            UfAmi_[0], UfAmi_[1], UfAmi_[2]);
        if (hasCyclic_)
            deviceCorrectUf(cyc_.n, nullptr, cyc_.Sfx, cyc_.Sfy, cyc_.Sfz, cyc_.magSf, cyc_.phi,
                            UfCyc_[0], UfCyc_[1], UfCyc_[2]);
    }

    // OF CorrectPhi (finiteVolume/cfdTools/general/CorrectPhi/CorrectPhi.C) together with the pimpleFoam.C
    // lines that wrap it:
    //
    //     phi = mesh.Sf() & Uf();          // absolute flux from the MAPPED surface velocity
    //     CorrectPhi(U, phi, p, rAUf = 1, divU = 0, pimple);
    //     fvc::makeRelative(phi, U);
    //
    // WHY IT EXISTS. When the mesh moves, the stored phi was computed on the OLD face areas. Reusing it
    // is not a small error on a rotating or sliding interface, where a face's Sf changes direction as
    // well as size: the momentum equation convects with a flux that does not close, and the pressure
    // equation is then asked to fix a continuity error that is really a geometry error. On brae the
    // symptom was unmistakable -- the two worst cases in the pimpleFoam sweep were exactly the two whose
    // continuity residual ran away (axialTurbine contLocal 2.0, oscillatingInletPeriodicAMI2D 3.4).
    //
    // Uf IS THE INPUT, not phi. That is the whole point: Uf is a face VELOCITY, so remapping it onto the
    // new Sf gives a flux consistent with the new geometry, which the old flux is not. This is why the
    // solver refuses to run the projection unless Uf is being maintained.
    scalar DeviceSimpleSolver::correctPhi(const std::vector<FvPatch>& fvp, int nNonOrth)
    {
        const DeviceMesh& dm = dm_;
        if (!ufActive_ || dm.nCells <= 0) return 0;

        // ---- phi = Sf & Uf, on every face family the mesh has ---------------------------------------
        deviceDotSf(dm.nInternalFaces, nullptr, dm.Sfx, dm.Sfy, dm.Sfz,
                    UfInt_[0], UfInt_[1], UfInt_[2], phiInt_);
        deviceDotSf(dm.nBndFaces, dm.bndGFace.data(), dm.Sfx, dm.Sfy, dm.Sfz,
                    UfBnd_[0], UfBnd_[1], UfBnd_[2], phiBnd_);
        if (hasAMI_)    deviceDotSf(ami_.n, nullptr, ami_.Sfx, ami_.Sfy, ami_.Sfz,
                                    UfAmi_[0], UfAmi_[1], UfAmi_[2], ami_.phi);
        if (hasCyclic_) deviceDotSf(cyc_.n, nullptr, cyc_.Sfx, cyc_.Sfy, cyc_.Sfz,
                                    UfCyc_[0], UfCyc_[1], UfCyc_[2], cyc_.phi);

        // ---- correctUphiBCs: phi_b = U_b & Sf wherever U fixes its value ----------------------------
        // OF sets this BEFORE the projection so the prescribed wall/inlet flux is what pcorr has to work
        // around, not something it is free to correct away. adjustMask_ is already !fixesValue(), built
        // from the same patch order as phiBnd_.
        {
            DeviceBuffer<scalar> ubx, uby, ubz, phiFix;
            deviceBCValue(dbU_.comp[0], Uk_[0], ubx);
            deviceBCValue(dbU_.comp[1], Uk_[1], uby);
            deviceBCValue(dbU_.comp[2], Uk_[2], ubz);
            deviceBoundaryFlux(dm, ubx, uby, ubz, phiFix);
            deviceSelectFixedFlux(adjustMask_, phiFix, phiBnd_);
        }

        // ---- pcorr.needReference(): adjustPhi on the RELATIVE flux, as OF does ----------------------
        // The bracket matters. adjustPhi enforces global continuity, and on a moving mesh the quantity
        // that has to balance is the flux RELATIVE to the mesh -- the absolute one legitimately does not
        // sum to zero when the domain volume is changing. OF makes it relative, adjusts, makes it
        // absolute again (CorrectPhi.C:79-84); dropping the round trip would push the volume-change rate
        // into the adjustment.
        if (ctl_.needRef && meshPhiValid_ && !meshPhi_.empty())
        {
            applyMeshPhi(fvp, meshPhi_, scalar(-1), phiInt_, phiBnd_, cyc_.phi, ami_.phi);
            deviceAdjustPhi(adjustMask_, phiBnd_, &phiInt_);
            applyMeshPhi(fvp, meshPhi_, scalar(+1), phiInt_, phiBnd_, cyc_.phi, ami_.phi);
        }
        else if (ctl_.needRef)
        {
            deviceAdjustPhi(adjustMask_, phiBnd_, &phiInt_);
        }

        // ---- the pcorr Poisson: laplacian(1, pcorr) == div(phi) -------------------------------------
        // rAUf is the dimensioned scalar 1 in pimpleFoam's correctPhi.H, so the diffusivity is unity on
        // every face and in every cell -- not rAU. Using rAU here would still project the flux, but to a
        // different weighting than OF's, and the difference does not vanish with mesh refinement.
        DeviceBuffer<scalar> onesF, onesC, pcorr;
        onesF.copyFrom(std::vector<scalar>(static_cast<std::size_t>(std::max(dm.nInternalFaces, 0)), scalar(1)));
        onesC.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nC_), scalar(1)));
        pcorr.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nC_), scalar(0)));

        DeviceSolverPerf pr{};
        const int nPasses = std::max(nNonOrth, 0) + 1;
        for (int pass = 0; pass < nPasses; ++pass)
        {
            DeviceBuffer<scalar> pcD, pcU, pcL, pcIC, pcBC, diagC, b, divPhi, ffc;
            deviceLaplacianCoeffs(dm, onesF, pcD, pcU, pcL, ctl_.nonOrth);
            deviceBCLaplacianCoeffs(dbPcorr_, onesC, pcIC, pcBC);
            deviceDiv(dm, phiInt_, phiBnd_, divPhi);
            if (hasCyclic_) interfaceAddDiv(cyc_, dm.V, divPhi);
            if (hasAMI_)    interfaceAddDiv(ami_, dm.V, divPhi);
            deviceFoldPressure(dm, pcD, divPhi, pcIC, pcBC, diagC, b);
            if (hasCyclic_) interfaceAssembleLaplacian(cyc_, onesC, diagC, /*addToDiag*/true);
            if (hasAMI_)    interfaceAssembleLaplacian(ami_, onesC, diagC, /*addToDiag*/true);

            DeviceBuffer<scalar> ffcCyc, ffcAmi;
            if (ctl_.nonOrth && pass > 0)   // pass 0 has pcorr == 0, so its gradient contributes nothing
            {
                DeviceBuffer<scalar> gx, gy, gz, pbv, sc;
                deviceBCValue(dbPcorr_, pcorr, pbv);
                deviceGaussGrad(dm, pcorr, pbv, gx, gy, gz);
                if (hasCyclic_) interfaceAddGrad(cyc_, pcorr, dm.V, gx, gy, gz);
                if (hasAMI_)    interfaceAddGrad(ami_, pcorr, dm.V, gx, gy, gz);
                deviceLaplacianCorrFlux(dm, onesF, gx, gy, gz, ffc);
                deviceFaceDivSource(dm, ffc, sc);
                deviceAxpy(1.0, sc, b);
                if (hasCyclic_) interfaceLapCorrP(cyc_, onesC, gx, gy, gz, b, ffcCyc);
                if (hasAMI_)    interfaceLapCorrP(ami_, onesC, gx, gy, gz, b, ffcAmi);
            }
            // pcorr inherits p's reference requirement exactly: its patches are fixedValue wherever p's
            // fix a value, so it is singular in precisely the cases p is.
            if (ctl_.needRef)
            {
                if (hasCyclic_ || hasAMI_)
                {
                    const scalar eps = -1e-10 * (deviceSumMag(diagC) / nC_);
                    deviceAxpy(eps, ones_, diagC);
                }
                else deviceSetReference(diagC, b, ctl_.pRefCell, scalar(0));
            }
            // Jacobi-PCG, not the AMG path p uses: amgGalerkin re-coarsens the shared hierarchy in place,
            // and pcorr is a unit-diffusivity Poisson solved to a loose tolerance once per mesh move --
            // not worth handing it p's hierarchy and then rebuilding that for p on the next line.
            const DeviceLduView pv = (hasCyclic_ || hasAMI_)
                ? (hasCyclic_
                    ? deviceLduViewCyclic(dm, diagC, pcU, pcL, cyc_.n, cyc_.ownCell.data(), cyc_.nbrCell.data(), cyc_.ifCoeff.data())
                    : deviceLduViewAmi(dm, diagC, pcU, pcL, ami_.n, ami_.ownCell.data(), ami_.off.data(), ami_.nbrCell.data(), ami_.weight.data(), ami_.ifCoeff.data()))
                : deviceLduView(dm, diagC, pcU, pcL);
            const scalar nf = deviceNormFactor(pv, pcorr, b, ones_);
            const DeviceSolverPerf r = deviceJacobiPCG(pv, b, pcorr, nf, ctl_.tolPcorr, ctl_.relTolPcorr,
                                                        ctl_.maxIterPcorr, 0);
            if (pass == 0) pr = r;

            if (pass == nPasses - 1)   // OF: phi -= pcorrEqn.flux() on the FINAL non-orthogonal pass only
            {
                DeviceBuffer<scalar> fi, fb;
                deviceMatrixFluxInternal(pv, pcorr, fi);
                deviceAxpy(-1.0, fi, phiInt_);
                if (ffc.size()) deviceAxpy(-1.0, ffc, phiInt_);
                deviceMatrixFluxBoundary(dbPcorr_, pcIC, pcBC, pcorr, fb);
                deviceAxpy(-1.0, fb, phiBnd_);
                if (hasCyclic_) interfaceCorrectFlux(cyc_, pcorr);   // cyc_.phi -= ifCoeff*(pcorr_nbr - pcorr_own)
                if (hasAMI_)    interfaceCorrectFlux(ami_, pcorr);
                if (ffcCyc.size()) deviceAxpy(-1.0, ffcCyc, cyc_.phi);
                if (ffcAmi.size()) deviceAxpy(-1.0, ffcAmi, ami_.phi);
            }
        }

        // ---- pimpleFoam.C: make the flux relative to the mesh motion --------------------------------
        makeFluxRelative(fvp, dm.nInternalFaces);
        return pr.initialResidual;
    }

    CourantNumbers DeviceSimpleSolver::courantNumbers(scalar deltaT) const
    {
        CourantNumbers c;
        const DeviceMesh& dm = dm_;
        if (dm.nCells <= 0) return c;
        DeviceBuffer<scalar> sumPhi;
        deviceSurfaceSumMagPhi(dm, phiInt_, phiBnd_,
                               hasCyclic_ ? &cyc_.ownCell : nullptr, hasCyclic_ ? &cyc_.phi : nullptr,
                               hasAMI_    ? &ami_.ownCell : nullptr, hasAMI_    ? &ami_.phi : nullptr,
                               sumPhi);
        // OF CourantNo.H: CoNum = 0.5*max(sumPhi/V)*deltaT, meanCoNum = 0.5*(sum(sumPhi)/sum(V))*deltaT.
        const scalar sumV = deviceSumMag(dm.V);
        c.CoNum     = scalar(0.5)*deviceMaxRatio(sumPhi, dm.V)*deltaT;
        c.meanCoNum = (sumV > 0) ? scalar(0.5)*(deviceSumMag(sumPhi)/sumV)*deltaT : scalar(0);
        return c;
    }


    void DeviceSimpleSolver::makeFluxRelative(const std::vector<FvPatch>& fvp, label nInternalFaces)
    {
        if (!meshPhiValid_ || meshPhi_.empty()) return;
        applyMeshPhi(fvp, meshPhi_, scalar(-1), phiInt_, phiBnd_, cyc_.phi, ami_.phi);
        (void)nInternalFaces;
    }

    // Split the flattened interface flux back out per coupled patch, and put it back. Both directions use
    // the SAME (patch, count) runs recorded where the interface was built, so a layout change cannot make
    // the write and the read disagree silently.
    std::vector<std::pair<label, std::vector<scalar>>> DeviceSimpleSolver::interfacePatchFlux() const
    {
        std::vector<std::pair<label, std::vector<scalar>>> out;
        auto split = [&out](const std::vector<std::pair<label, label>>& runs, const DeviceBuffer<scalar>& phi)
        {
            if (runs.empty() || phi.size() == 0) return;
            const std::vector<scalar> h = phi.host();
            std::size_t off = 0;
            for (const auto& r : runs)
            {
                const std::size_t n = static_cast<std::size_t>(r.second);
                if (off + n > h.size()) break;                      // layout mismatch: write nothing rather than garbage
                out.push_back({r.first, std::vector<scalar>(h.begin() + off, h.begin() + off + n)});
                off += n;
            }
        };
        split(cycRuns_, cyc_.phi);
        split(amiRuns_, ami_.phi);
        return out;
    }

    void DeviceSimpleSolver::setInterfacePatchFlux(const std::vector<std::pair<label, std::vector<scalar>>>& byPatch)
    {
        auto seed = [&byPatch](const std::vector<std::pair<label, label>>& runs, DeviceBuffer<scalar>& phi)
        {
            if (runs.empty() || phi.size() == 0) return;
            std::vector<scalar> h = phi.host();
            std::size_t off = 0;
            for (const auto& r : runs)
            {
                const std::size_t n = static_cast<std::size_t>(r.second);
                if (off + n > h.size()) break;
                for (const auto& b : byPatch)
                    if (b.first == r.first && b.second.size() == n)
                    {
                        std::copy(b.second.begin(), b.second.end(), h.begin() + off);
                        break;
                    }
                off += n;
            }
            phi.copyFrom(h);
        };
        seed(cycRuns_, cyc_.phi);
        seed(amiRuns_, ami_.phi);
    }

    void DeviceSimpleSolver::setPatchVelocity(label patchi, const std::vector<vector>& Uw)
    {
        // dbU_ holds the three components' boundary descriptors in DeviceBoundary face order; a patch
        // occupies a contiguous run starting at its own offset.
        if (patchi < 0 || patchi >= (label)patchStart_.size()) return;
        const label start = patchStart_[patchi];
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> rv = dbU_.comp[k].refValue.host();
            for (std::size_t i = 0; i < Uw.size(); ++i)
            {
                const std::size_t f = static_cast<std::size_t>(start) + i;
                if (f < rv.size()) rv[f] = (k == 0 ? Uw[i].x : k == 1 ? Uw[i].y : Uw[i].z);
            }
            dbU_.comp[k].refValue.copyFrom(rv);
        }
    }

    // See the header for why. The wall velocities are taken from dbU_ rather than from a host U field
    // because on a moving mesh those two disagree: movingWallVelocity is assigned straight into the
    // device boundary by setPatchVelocity, so dbU_ is what the solver actually imposes.
    void DeviceSimpleSolver::refreshWallData(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& fvp)
    {
        const std::vector<scalar> rvx = dbU_.comp[0].refValue.host();
        const std::vector<scalar> rvy = dbU_.comp[1].refValue.host();
        const std::vector<scalar> rvz = dbU_.comp[2].refValue.host();
        std::vector<std::vector<vector>> wallU(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (fvp[pi].type != "wall") continue;
            wallU[pi].resize(fvp[pi].size);
            const std::size_t st = static_cast<std::size_t>(patchStart_[pi]);
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const std::size_t f = st + static_cast<std::size_t>(i);
                if (f < rvx.size()) wallU[pi][i] = vector{rvx[f], rvy[f], rvz[f]};
            }
        }
        wall_ = buildDeviceWallData(m, g, fvp, wallU, ctl_.turbWallPatch);
    }

    void DeviceSimpleSolver::advanceTime(scalar deltaT)
    {
        const bool cn  = (ddtScheme_ == DdtScheme::CrankNicolson);
        const bool two = (ddtScheme_ == DdtScheme::backward) || cn;   // backward (coefft00 term) + CrankNicolson (ddt0 recurrence) both need the 2nd old level
        for (int k = 0; k < 3; ++k)
        {
            if (two && Uold_[k].size()) deviceCopy(Uold2_[k], Uold_[k]);   // t-2 <- t-1  (once t-1 exists)
            deviceCopy(Uold_[k], Uk_[k]);                                  // t-1 <- current
        }
        if (ctl_.maxwell && sig_[0].size())   // fvm::ddt(sigma): the stress carries its own old levels
        {
            for (int k = 0; k < 6; ++k)
            {
                if (two && sigOld_[k].size()) deviceCopy(sigOld2_[k], sigOld_[k]);
                deviceCopy(sigOld_[k], sig_[k]);
            }
        }
        if (ctl_.turbulent && !ctl_.les)   // turbulence old-time levels for fvm::ddt(k/eps): dk_ = k (or nuTilda), de_ = epsilon|omega. Pure LES nut is algebraic -> no ddt.
        {
            if (two && kOld_.size()) deviceCopy(kOld2_, kOld_);
            deviceCopy(kOld_, dk_);
            if (!ctl_.sa)   // SA is one-equation (no second scalar)
            {
                if (two && e2Old_.size()) deviceCopy(e2Old2_, e2Old_);
                deviceCopy(e2Old_, de_);
            }
            if (ctl_.lm)   // kOmegaSSTLM transition fields (ReThetat, gammaInt)
            {
                if (two && ReThetatOld_.size()) deviceCopy(ReThetatOld2_, ReThetatOld_);
                deviceCopy(ReThetatOld_, ReThetat_);
                if (two && gammaIntOld_.size()) deviceCopy(gammaIntOld2_, gammaIntOld_);
                deviceCopy(gammaIntOld_, gammaInt_);
            }
        }
        if (cn)   // CrankNicolson: update each transported variable's stored old ddt (ddt0) from the just-rotated old levels.
        {         // deltaT0 = the PREVIOUS deltaT (deltaT_ not yet overwritten); no-op on the first step (no oldTime.oldTime).
            const DdtCoeffs ddtc = ddtCoeffs(ddtScheme_, deltaT, deltaT_, ocCoeff_, cnWarm_);
            for (int k = 0; k < 3; ++k) deviceFvmDdtUpdateDdt0(ddtc, Uold_[k], Uold2_[k], Uddt0_[k]);
            if (ctl_.maxwell && sig_[0].size())
                for (int k = 0; k < 6; ++k) deviceFvmDdtUpdateDdt0(ddtc, sigOld_[k], sigOld2_[k], sigddt0_[k]);
            if (ctl_.turbulent && !ctl_.les)
            {
                deviceFvmDdtUpdateDdt0(ddtc, kOld_, kOld2_, kddt0_);
                if (!ctl_.sa) deviceFvmDdtUpdateDdt0(ddtc, e2Old_, e2Old2_, e2ddt0_);
                if (ctl_.lm)
                {
                    deviceFvmDdtUpdateDdt0(ddtc, ReThetatOld_, ReThetatOld2_, ReThetatddt0_);
                    deviceFvmDdtUpdateDdt0(ddtc, gammaIntOld_, gammaIntOld2_, gammaIntddt0_);
                }
            }
            if (ddtc.cn && Uold2_[0].size()) cnWarm_ = true;   // first (Euler-startup) ddt0 update done -> subsequent updates use 1+oc
        }
        // phi.oldTime(): the flux at the PREVIOUS time level, which fvc::ddtCorr reads. OF ages it at
        // runTime++ along with every other field, so it is fixed for the whole step -- all of the step's
        // pressure correctors see the same one, not each other's output.
        if (phiOldInt_.size()) { deviceCopy(phiOld2Int_, phiOldInt_); deviceCopy(phiOld2Bnd_, phiOldBnd_); }
        deviceCopy(phiOldInt_, phiInt_);
        deviceCopy(phiOldBnd_, phiBnd_);
        // The interface's phi.oldTime(), and flux(U.oldTime()) across it. Both are fixed for the step, so
        // they are built once here rather than per corrector. interfaceFlux writes into cyc_/ami_.phi --
        // that IS the flux buffer -- so it is saved and put back around the call.
        if (hasCyclic_ && cyc_.phi.size())
        {
            deviceCopy(cycPhiOld_, cyc_.phi);
            interfaceFlux(cyc_, Uold_[0], Uold_[1], Uold_[2]);
            deviceCopy(cycFluxUold_, cyc_.phi);
            deviceCopy(cyc_.phi, cycPhiOld_);
        }
        if (hasAMI_ && ami_.phi.size())
        {
            deviceCopy(amiPhiOld_, ami_.phi);
            interfaceFlux(ami_, Uold_[0], Uold_[1], Uold_[2]);
            deviceCopy(amiFluxUold_, ami_.phi);
            deviceCopy(ami_.phi, amiPhiOld_);
        }
        // ...and on a MOVING mesh the old flux fvc::ddtCorr wants is not phi.oldTime() at all.
        // pimpleFoam passes fvc::ddtCorr(U, phi, Uf), which takes the Uf form whenever the mesh is
        // dynamic, and that form reads
        //
        //     phiUf0 = mesh.Sf() & Uf.oldTime()          (EulerDdtScheme::fvcDdtUfCorr)
        //
        // -- the CURRENT face areas dotted into the face velocity stored at the previous time level. It
        // is also the denominator of the coupling coefficient, not phi.oldTime(), so both halves of the
        // correction come from it.
        //
        // These differ whenever Sf changes between the two levels. On a translating mesh they do not,
        // and phi.oldTime() + mesh.phi() reproduces phiUf0 to 2.7e-10 -- which is what brae used to do.
        // A cyclicACMI breaks that: the overlap mask rescales the coupled areas every step. Measured on
        // the moving oscillatingInletACMI2D, the shortcut was exact at step 1 and 5.1e-03 out by step 10,
        // all of it on the interface columns.
        if (ufActive_)
        {
            const DeviceMesh& dm = dm_;
            for (int k = 0; k < 3; ++k)   // age Uf, as runTime++ ages every registered field
            {
                if (UfOldInt_[k].size()) { deviceCopy(UfOld2Int_[k], UfOldInt_[k]); deviceCopy(UfOld2Bnd_[k], UfOldBnd_[k]); }
                deviceCopy(UfOldInt_[k], UfInt_[k]);
                deviceCopy(UfOldBnd_[k], UfBnd_[k]);
                if (hasAMI_)    deviceCopy(UfOldAmi_[k], UfAmi_[k]);
                if (hasCyclic_) deviceCopy(UfOldCyc_[k], UfCyc_[k]);
            }
            deviceDotSf(dm.nInternalFaces, nullptr, dm.Sfx, dm.Sfy, dm.Sfz,
                        UfOldInt_[0], UfOldInt_[1], UfOldInt_[2], phiOldInt_);
            deviceDotSf(dm.nBndFaces, dm.bndGFace.data(), dm.Sfx, dm.Sfy, dm.Sfz,
                        UfOldBnd_[0], UfOldBnd_[1], UfOldBnd_[2], phiOldBnd_);
            if (UfOld2Int_[0].size())   // backward's second level, on the CURRENT areas
            {
                deviceDotSf(dm.nInternalFaces, nullptr, dm.Sfx, dm.Sfy, dm.Sfz,
                            UfOld2Int_[0], UfOld2Int_[1], UfOld2Int_[2], phiOld2Int_);
                deviceDotSf(dm.nBndFaces, dm.bndGFace.data(), dm.Sfx, dm.Sfy, dm.Sfz,
                            UfOld2Bnd_[0], UfOld2Bnd_[1], UfOld2Bnd_[2], phiOld2Bnd_);
            }
            if (hasAMI_)
                deviceDotSf(ami_.n, nullptr, ami_.Sfx, ami_.Sfy, ami_.Sfz,
                            UfOldAmi_[0], UfOldAmi_[1], UfOldAmi_[2], amiPhiOld_);
            if (hasCyclic_)
                deviceDotSf(cyc_.n, nullptr, cyc_.Sfx, cyc_.Sfy, cyc_.Sfz,
                            UfOldCyc_[0], UfOldCyc_[1], UfOldCyc_[2], cycPhiOld_);
        }
        deltaT0_ = deltaT_;
        deltaT_  = deltaT;
    }

    // One PIMPLE time step -- the transient counterpart of step(): the SAME three composable phases re-orchestrated under
    // outer (momentum<->pressure<->turbulence) + inner (pressure) corrector loops, with fvm::ddt(U) folded into the
    // predictor (active once setDdtScheme() set ddtScheme_ != steadyState). The momentum matrix is re-assembled each
    // outer corrector from the latest U/phi, exactly as OF's while(pimple.loop()) re-forms UEqn.
    void DeviceSimpleSolver::setCompressible(
        const ThermoCoeffs& tc,
        const RhoSimpleControls& rc,
        DeviceBoundary dbHe)
    {
        tc_ = tc;
        rc_ = rc;
        dbHe_ = std::move(dbHe);
        th_.allocate(dm_.nCells);
        compressible_ = true;
    }

    void DeviceSimpleSolver::solvePassiveScalar(
        DeviceBuffer<scalar>& field,
        const DeviceBoundary& db,
        const char* fieldName,
        const DeviceBuffer<scalar>& D,
        scalar relax,
        scalar tol,
        bool   schemeBounded,
        bool   schemeLimited,
        bool   schemeLinearUpwind,
        scalar schemeTwoByk,
        bool   schemeNonOrth,
        DeviceBuffer<scalar>* old,
        DeviceBuffer<scalar>* old2,
        DeviceBuffer<scalar>* ddt0)
    {
        const int nC = dm_.nCells;
        if (static_cast<int>(field.size()) != nC) return;

        // The bounded-convection correction term needs div(phi); build it the same way every other
        // caller of deviceSolveScalarTransport does.
        DeviceBuffer<scalar> divPhi;
        deviceFaceDivSource(dm_, phiInt_, divPhi);

        // fvm::ddt(s), from the solver's own time scheme. Old levels are advanced HERE, with the same
        // two-level rule advanceTime() uses: backward needs the coefft00 term and CrankNicolson the ddt0
        // recurrence, so both carry t-2; Euler does not.
        // A TRANSIENT solver whose deltaT_ is still 0 has not advanced yet: this solver learns deltaT
        // inside advanceTime(), and a functionObject fires BEFORE the first one. Solving here would drop
        // fvm::ddt(s) and solve a STEADY convection-diffusion problem instead -- not an approximation of
        // the transient one but a different, badly-conditioned equation, whose error then persists in
        // the field. Measured on pitzDaily: a tracer bounded by 1.0 reached 6.32. Skipping this single
        // step leaves the tracer at its initial value for one step, which is strictly smaller error.
        if (old && ddtScheme_ != DdtScheme::steadyState && deltaT_ <= scalar(0)) return;

        ScalarDdt sDdt{};
        if (old && ddtScheme_ != DdtScheme::steadyState)
        {
            const bool cn  = (ddtScheme_ == DdtScheme::CrankNicolson);
            const bool two = (ddtScheme_ == DdtScheme::backward) || cn;
            if (old->size() != static_cast<std::size_t>(nC)) { old->resize(nC); deviceCopy(*old, field); }
            if (two && old2)
            {
                if (old2->size() != static_cast<std::size_t>(nC)) old2->resize(nC);
                deviceCopy(*old2, *old);       // t-2 <- t-1
            }
            deviceCopy(*old, field);           // t-1 <- current
            const DdtCoeffs ddtc = ddtCoeffs(ddtScheme_, deltaT_, deltaT0_, ocCoeff_, cnWarm_);
            sDdt = ScalarDdt{ ddtc, old,
                              (two && old2 && old2->size()) ? old2 : nullptr,
                              (ddtc.cn && ddt0) ? ddt0 : nullptr };
        }

        // No reaction: a passive scalar has no source. OF's fvOptions_(rho, s) is not applied -- brae
        // refuses unsupported fvOptions at start-up, so there is nothing here to drop silently.
        deviceSolveScalarTransport(
            dm_, db, field, fieldName, D, phiInt_, phiBnd_, divPhi,
            schemeBounded, schemeLimited, schemeLinearUpwind, schemeNonOrth, schemeTwoByk,
            relax, tol, ctl_.keRelTol(), ctl_.bicgCheckEvery, false,
            [](DeviceBuffer<scalar>&, DeviceBuffer<scalar>&){},   // passive: no reaction
            nullptr, nullptr,
            hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr,
            sDdt,                 // steady -> default-constructed, so fvm::ddt(s) drops out
            nullptr, scalar(0),
            false);               // OF's scalarTransport does NOT bound; a tracer is not positive-definite
    }

    void DeviceSimpleSolver::setAlphatPrt(const std::vector<scalar>& prtFace)
    {
        if (!prtFace.empty()) prtBnd_.copyFrom(prtFace);
    }


    // See the header for why the phases are ordered UEqn -> EEqn -> pEqn -> thermo -> turbulence.
    DeviceSimpleResidual DeviceSimpleSolver::rhoSimpleStep()
    {
        const DeviceMesh& dm = dm_;
        DeviceSimpleResidual res;

        // muEff. The momentum assembly builds nuEff = dnut_ + nuConst_ and nuConst_ has exactly two uses
        // (seeded from ctl_.nu at construction, added here), so handing it the DYNAMIC viscosity turns the
        // same expression into muEff without touching the assembly. mu is T-dependent under Sutherland, so
        // it must be refreshed every outer iteration, not seeded once. dnut_ is zero while the solve is
        // laminar; phase 4 fills it with mut and this line keeps working unchanged.
        deviceCopy(nuConst_, th_.mu);
        // laminar generalizedNewtonian: nuEff() RETURNS nu_, it does not add to the molecular nu, so this
        // overwrites nuConst_ (= muEff's laminar part) rather than adding to it. Placed straight after the
        // thermo copy because nu0 = mu/rho must be the CURRENT thermo viscosity, as OF's
        // `nu_ = viscosityModel_->nu(this->nu(), strainRate())` re-evaluates it every correct().
        if (ctl_.gnPowerLaw)
        {
            DeviceBuffer<scalar> gradU, S2;
            deviceGradU(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], gradU,
                        hasAMI_ ? &ami_ : nullptr, hasCyclic_ ? &cyc_ : nullptr);
            deviceS2(gradU, nC_, S2);            // 2*magSqr(symm(gradU)); sqrt(S2) = OF's strainRate
            deviceGeneralizedNewtonianPowerLawMu(S2, th_.rho, ctl_.gnNuMin, ctl_.gnNuMax, ctl_.gnN, nuConst_);
            if (std::getenv("BRAE_GN_DEBUG"))
            {
                const std::vector<scalar> hm = nuConst_.host(), hr = th_.rho.host(), hs = S2.host();
                double mn=1e300,mx=-1e300,rn=1e300,rx=-1e300,sn=1e300,sx=-1e300;
                for (std::size_t q=0;q<hm.size();++q){ mn=std::min(mn,(double)hm[q]); mx=std::max(mx,(double)hm[q]);
                    rn=std::min(rn,(double)hr[q]); rx=std::max(rx,(double)hr[q]);
                    sn=std::min(sn,std::sqrt(std::max(0.0,(double)hs[q]))); sx=std::max(sx,std::sqrt(std::max(0.0,(double)hs[q]))); }
                std::printf("  [gn] mu %.4e..%.4e  rho %.4e..%.4e  strainRate %.4e..%.4e\n", mn,mx,rn,rx,sn,sx);
            }
        }

        // Boundary rho and mu for this iteration. Both are T-dependent, so they move with the solution and
        // must be refreshed here rather than seeded once. nuBndConst_ becomes mu_b, which turns the boundary
        // expression nuEffBnd = nuBndConst_ + nut_b into muEff_b = mu_b + rho_b*nut_b once the wall nut is
        // rho_b-weighted (see addWallNutToMuEff).
        // One T_b for all three, as in OF's single boundary loop -- not three separate inversions of he_b.
        deviceThermoTBoundary(dbP_, dp_, dbHe_, th_.he, tc_, &th_.T, &TFixMask_, &TFixBnd_, TBnd_);
        deviceThermoRhoBoundary(dbP_, dp_, TBnd_, tc_, rhoBnd_);
        deviceThermoMuBoundary(TBnd_, tc_, muBnd_);
        deviceThermoNuBoundary(dbP_, dp_, TBnd_, tc_, nuWallBnd_);   // OF nu(patchi), for the wall functions
        deviceGatherWallNu(wfBndIdx_, nuWallBnd_, wfNu_);   // same nu, in the wall-face ordering omega/G0 uses
        deviceCopy(nuBndConst_, muBnd_);

        // flowRateInletVelocity (massFlowRate): OF recomputes avgU = -mdot/gSum(rho*magSf) every time the
        // U boundary is updated, so it tracks the inlet density as the solution develops. Done here, before
        // the momentum predictor, so the predictor sees the same inlet OF's does.
        //
        // THE DENSITY HERE MUST BE THE SOLVER'S rho, NOT thermo.rho(). OF's
        // flowRateInletVelocityFvPatchVectorField::updateCoeffs does
        //     patch().lookupPatchField<volScalarField>(rhoName_)   // rhoName_ defaults to "rho"
        // which resolves to the REGISTERED rho -- the relaxed field createFields.H built -- and that is
        // the same field `phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA)` weights the flux with. Using
        // one density for both is what makes the inlet flux telescope exactly:
        //     sum_i rho_i|Sf|_i * (-mdot / sum_j rho_j|Sf|_j)  ==  -mdot
        // Feed avgU a DIFFERENT density from the one weighting the flux and the cancellation breaks, so
        // the inlet no longer delivers the prescribed mass flow. Measured on angledDuct (relaxRho 0.01):
        // OF's inlet flux is pinned at exactly -1.00000e-01 kg/s every iteration, brae's ran
        // -0.100024 -> -0.099690 -> -0.104411 -> -0.068195, and that net mass imbalance is what drove the
        // pressure level to -300 kPa by iteration 4.
        //
        // Invisible on every other compressible case in validation/: they all set rho 1.0, where the
        // relaxed and thermo densities are identical. angledDuct is the one case with BOTH a relaxed rho
        // and a flowRateInletVelocity inlet.
        const DeviceBuffer<scalar>& frRho =
            (rhoBndP_.size() == static_cast<std::size_t>(dbP_.n)) ? rhoBndP_ : rhoBnd_;
        for (std::size_t k = 0; k < frPatches_.size(); ++k)
        {
            const scalar sumRhoA = deviceDot(frRho, frMagSf_[k]);
            if (sumRhoA <= 0.0) continue;
            deviceUpdateFlowRateInlet(dbU_, frMagSf_[k], -frPatches_[k].mdot / sumRhoA, frNx_, frNy_, frNz_);
        }

        solveMomentumPredictor(res);
        // Stage harness: U straight out of the MOMENTUM PREDICTOR. OF's UEqn.H() uses THIS U (not the
        // initial field), so it is its own stage between rAU and HbyA. Missing it was why the first
        // Phase 1 pass could not explain a 4% HbyA error on a case whose 0/U is uniform zero.
        if (stageDumpActive() && stageDumpFirstOnly("Upred"))
        {
            stageDump3("stage_Upred", Uk_[0], Uk_[1], Uk_[2]);
        }

        // EEqn, with OF's kinetic-energy term: EEqn.H is div(phi,he) + div(phi,K) - laplacian(alphaEff,he).
        // K is built from the velocity the momentum predictor just produced, matching OF's ordering.
        // sensibleInternalEnergy convects Ekp = 0.5|U|^2 + p/rho, sensibleEnthalpy just K = 0.5|U|^2.
        // p_b comes from the pressure BC and rho_b from the boundary p and T, so the boundary half of
        // div(phi,Ekp) is evaluated on the patch exactly as OF's fvc::div does, not extrapolated.
        // div(phi) for the energy equation's "bounded" term. Was passed as zeros, which silently
        // disabled the term even when fvSchemes asked for it.
        DeviceBuffer<scalar> divPhiHe;
        deviceDiv(dm, phiInt_, phiBnd_, divPhiHe);
        DeviceBuffer<scalar> kineticSrc;
        DeviceBuffer<scalar> pBndK;
        if (tc_.internalEnergy) deviceBCValue(dbP_, dp_, pBndK);
        deviceEnergyKineticSource(dm, dbU_, Uk_[0], Uk_[1], Uk_[2], phiInt_, phiBnd_, kineticSrc,
                                  tc_.internalEnergy ? &dp_ : nullptr,
                                  tc_.internalEnergy ? &th_.rho : nullptr,
                                  tc_.internalEnergy ? &pBndK : nullptr,
                                  tc_.internalEnergy ? &rhoBnd_ : nullptr,
                                  // div(phi,K|Ekp) scheme. luHe was parsed but never forwarded, so a case
                                  // asking for `div(phi,Ekp) bounded Gauss linearUpwind` silently ran upwind
                                  // on a term that is a large share of the energy under sensibleInternalEnergy
                                  // (Ekp = 0.5|U|^2 + p/rho, and p/rho is order 40% of e).
                                  &dbHe_, ctl_.limitedKin, ctl_.twoBykKin, ctl_.luKin, ctl_.gradKinLimitK,
                                  ctl_.boundedKin);   // parsed since the scheme work, but never forwarded until now
        // alphaEff on the PATCHES, so the alphatWallFunction actually reaches the wall heat flux.
        // ALWAYS when compressible, not just when turbulent. OF's laplacian uses the PATCH diffusivity,
        // alpha_b = alphah(p_b, T_b) evaluated at the BOUNDARY temperature. With transport const that
        // equals the adjacent cell value and the distinction is invisible; under sutherland a 700 K wall
        // against a 370 K first cell makes mu_b ~1.6x the cell mu, and using the cell value understates
        // the wall heat flux by that much while converging perfectly.
        DeviceBuffer<scalar> alphaEffBnd;
        if (compressible_)
            deviceAlphaEffBoundary(muBnd_, rhoBnd_,
                                   ctl_.turbulent ? &nutBndAll_ : nullptr,
                                   prtBnd_.size() ? &prtBnd_ : nullptr,
                                   TBnd_, tc_, alphaEffBnd);
        deviceSolveEnergy(
            dm,
            dbHe_,
            th_,
            tc_,
            phiInt_,
            phiBnd_,
            divPhiHe,
            ctl_.boundedHe,
            ctl_.limitedHe,   // div(phi,h|e) limitedLinear, from fvSchemes -- was hardcoded false
            ctl_.luHe,        // div(phi,h|e) linearUpwind,   from fvSchemes -- was hardcoded false
            ctl_.nonOrth,
            ctl_.twoBykHe,
            rc_.relaxHe,
            eTol_,              // fvSolution solvers.(h|e).tolerance -- was hardcoded 1e-10
            eRelTol_,           //                        .relTol     -- was hardcoded 0.0
            ctl_.bicgCheckEvery,
            eUseGS_,            //                        .solver smoothSolver -- was hardcoded false
            nullptr,
            nullptr,
            &kineticSrc,
            alphaEffBnd.size() ? &alphaEffBnd : nullptr,
            ctl_.gradHeLimitK,
            fixTActive_ ? &fixTMask_ : nullptr,   // fvOptions fixedTemperatureConstraint (OF eqn.setValues)
            fixTActive_ ? &fixTHe_   : nullptr);   // grad(he) cellLimited coeff, from the gradient linearUpwind names

        // Capture the energy solve's residuals before correctTurbulence() clears the shared report store.
        for (const ScalarSolveEntry& se : turbulenceReport())
            if (se.field == "he")
            { res.he = se.perf.initialResidual; res.heFinal = se.perf.finalResidual; res.heIters = se.perf.nIterations; }

        // OF EEqn.H: `EEqn.solve(); fvOptions.correct(he); thermo.correct();` -- the constraint is applied
        // to he BEFORE the thermo update, so T and every property derived from it come out of the clamped
        // energy. Clamping after thermo.correct() would leave T and he disagreeing for the rest of the step.
        if (limTActive_)
        {
            deviceFvoLimitEnergy(limTCells_, limTMin_, limTMax_, th_.he);
            // OF clamps the he boundary too, but only for selectionMode all (`!cellSetOption::useSubMesh()`).
            if (limTAllCells_ && dbHe_.n > 0)
            {
                DeviceBuffer<scalar> heB;
                deviceBCValue(dbHe_, th_.he, heB);
                deviceFvoLimitEnergyBoundary(dbHe_, limTMin_, limTMax_, heB);
            }
        }

        // OF EEqn.H ends with thermo.correct(), so the PRESSURE equation sees psi/mu/alpha from the
        // just-solved he. brae used to update thermo only after the pressure step, leaving the pEqn on
        // the previous iteration's psi; doing it here is OF's ordering.
        //
        // WHICH FIELDS THIS MOVES IS OF'S DEFINITION, not a decision taken here. hePsiThermo::calculate
        // writes T/psi/mu/alpha and has no rho argument at all; heRhoThermo::calculate writes those AND
        // rho. deviceThermoCorrect dispatches exactly as OF's virtual call does.
        deviceThermoCorrect(th_, dp_, tc_);

        correctPressureVelocity(res);

        // P6: p as pressureControl sees it, before any clamping. Dumped unconditionally (not inside the
        // limiter branch) so the trace has the same six stages whether or not limits are configured.
        if (pTraceActive()) pTraceDump("P6_pPreLimit", pTraceIt_, dp_);

        // pressureControl::limit(p), in OF's position: after the pressure solve and the U correction,
        // BEFORE rho is recomputed. Clamping after rho would let one NaN density through per iteration.
        if (rc_.limitMaxP || rc_.limitMinP)
        {
            deviceLimitPressure(dp_,
                                rc_.limitMinP ? rc_.pMinLimit : -1e300,
                                rc_.limitMaxP ? rc_.pMaxLimit : 1e300);
        }

        // End of pEqn.H/pcEqn.H. OF does exactly `rho = thermo.rho()` here -- NO thermo.correct() -- so
        // T, psi, mu and alpha keep the values EEqn's thermo.correct() gave them, and only rho moves.
        // Re-running the full update is harmless for those four (he has not changed since the energy
        // solve, so T/psi/mu/alpha come out identical), but NOT for rho, which depends on the p the
        // pressure equation just changed. And what OF's `thermo.rho()` returns depends on the thermo type:
        //
        //     hePsiThermo:  p_*psi_   -> recomputed with the JUST-SOLVED p    (what this line does)
        //     heRhoThermo:  rho_      -> the STORED field, from BEFORE the solve
        //
        // brae accepted heRhoThermo on the grounds that rho == psi*p exactly for perfectGas. That is true
        // of the arithmetic and false of the timing: a heRhoThermo case must carry a rho that LAGS the
        // pressure by one outer iteration. squareBend is heRhoThermo, and its rho was 7.0e-02 off OF at
        // iteration 1 -- which then propagates into phid (+5%) and the whole transonic pressure equation.
        // OF pEqn.H tail: `rho = thermo.rho(); rho.relax();`. deviceThermoRho IS thermo.rho() --
        // p*psi for hePsiThermo, the stored rho_ for heRhoThermo -- so the one-iteration lag a
        // heRhoThermo case carries falls out of OF's own definition instead of a copy-restore dance.
        deviceThermoRho(th_, dp_, tc_, th_.rho);
        // OF pEqn.H: `rho = thermo.rho(); if (!simple.transonic()) { rho.relax(); }` -- the transonic
        // branch does NOT relax rho, because its pressure equation already carries psi*p implicitly and
        // relaxing rho on top of that lags the two against each other. brae relaxed unconditionally; that
        // was invisible while thermo.correct() was overwriting rho anyway, and became a divergence
        // (squareBend -> NaN) the moment the solver rho was separated and the relaxation started working.
        if (!ctl_.transonic) deviceRhoRelax(th_, tc_);
        // ...and the BOUNDARY half of the same field, with the same factor. OF's rho.relax() relaxes the
        // internal and boundary values of one field together; brae keeps them in two buffers, so both
        // have to be stepped here or the pressure equation reads a relaxed rho inside and an unrelaxed
        // one on the patches.
        if (compressible_)
        {
            deviceThermoTBoundary(dbP_, dp_, dbHe_, th_.he, tc_, &th_.T, &TFixMask_, &TFixBnd_, TBnd_);
            deviceThermoRhoBoundary(dbP_, dp_, TBnd_, tc_, rhoBndP_);
            if (rhoBndPPrev_.size() != rhoBndP_.size()) deviceCopy(rhoBndPPrev_, rhoBndP_);   // never on the
            deviceRhoRelaxBuffer(rhoBndP_, rhoBndPPrev_, tc_);                                 // seeded path
        }

        correctTurbulence();

        // alphat = mut/Prt, for the NEXT iteration's energy equation. Must follow correctTurbulence(),
        // which is what makes nut valid. No-op while laminar (nut empty -> alphat stays zero), so the
        // energy equation is unaffected until phase 4.1 gives the SST a compressible nut.
        deviceAlphat(th_, dnut_, tc_);

        // Continuity. NOTE: with a mass flux this is a MASS residual, not a volumetric one -- the number
        // means something different from the incompressible report even though the code is identical.
        // Flagged in phi-audit.md as an INTERPRET site.
        {
            DeviceBuffer<scalar> divPhi;
            deviceDiv(dm, phiInt_, phiBnd_, divPhi);
            if (hasCyclic_) interfaceAddDiv(cyc_, dm.V, divPhi);   // interface faces are outside the DeviceBoundary
            if (hasAMI_)    interfaceAddDiv(ami_, dm.V, divPhi);   // (see step(): the pEqn's div includes them)
            DeviceBuffer<scalar> R;
            deviceHadamard(R, divPhi, dm.V);
            const scalar sumV = deviceDot(dm.V, ones_) + 1e-300;
            res.contLocal = deviceSumMag(R) / sumV;
            res.contGlobal = deviceDot(R, ones_) / sumV;
        }
        return res;
    }

    // OF pimpleControl::criteriaSatisfied. Every controlled field passes if its FINAL residual is below
    // absTol, or if final/initial is below relTol -- either alone is enough, per field, and all fields
    // must pass. The reference for the relative test is iteration 2's INITIAL residual (OF's
    // storeInitialResiduals() is corr_ == 2), not iteration 1's: iteration 1 starts from the previous
    // time step's field and its residual is not comparable with the outer-loop convergence being tested.
    bool DeviceSimpleSolver::outerCriteriaSatisfied(const DeviceSimpleResidual& res, int oc)
    {
        // OF's maxResidual returns the pair (initial, final) of the LAST solve. The reference stored at
        // corr_ == 2 is that pair's FIRST -- iteration 1's initial residual -- while the test from
        // corr_ == 3 onward uses its LAST, the previous iteration's final residual. Two different halves
        // of the pair; using the initial residual for both makes the relative test compare an iteration
        // against itself and converge on step one.
        auto initialOf = [&](const std::string& f) -> scalar
        {
            if (f == "U" || f == "Ux") return res.Ux;
            if (f == "Uy") return res.Uy;
            if (f == "Uz") return res.Uz;
            if (f == "p")  return res.p;
            return scalar(0);
        };
        auto residualOf = [&](const std::string& f) -> scalar
        {
            if (f == "U" || f == "Ux") return res.UxFinal;
            if (f == "Uy") return res.UyFinal;
            if (f == "Uz") return res.UzFinal;
            if (f == "p")  return res.pFinal;
            // A residualControl entry naming a field brae does not track here would otherwise decide
            // convergence off a residual of zero -- i.e. converge instantly, on every step. Refuse
            // instead: an unimplemented control must not silently change the algorithm.
            throw std::runtime_error(
                "brae: PIMPLE/residualControl names field '" + f + "', whose per-outer-iteration "
                "residual brae does not track. Only U (Ux/Uy/Uz) and p are available. Running would "
                "test convergence against a residual of zero and leave the outer loop immediately.");
        };
        if (oc == 0) return false;                       // OF: corr_ == 1 never satisfies
        if (oc == 1)                                     // corr_ == 2: STORE, do not compare
        {
            for (const auto& rc : ctl_.outerResidualControl)
                outerInitialResidual_[rc.field] = initialOf(rc.field);
            return false;
        }
        bool achieved = true, checked = false;
        for (const auto& rc : ctl_.outerResidualControl)
        {
            const scalar r = residualOf(rc.field);
            const auto it = outerInitialResidual_.find(rc.field);
            const scalar ini = (it == outerInitialResidual_.end() ? scalar(0) : it->second) + scalar(1e-300);
            const bool abs_ = (rc.absTol > 0) && (r < rc.absTol);
            const bool rel_ = (rc.relTol > 0) && (r / ini < rc.relTol);
            achieved = achieved && (abs_ || rel_);
            checked = true;
        }
        return checked && achieved;
    }


    DeviceSimpleResidual DeviceSimpleSolver::pimpleStep(scalar deltaT, int nOuterCorrectors, int nCorrectors,
                                                       const std::function<void(int)>& moveMesh)
    {
        const DeviceMesh& dm = dm_;
        // THE FIRST mesh update runs BEFORE advanceTime, not inside the loop with the others.
        //
        // OpenFOAM ages its fields (++runTime) and then updates the mesh, because phi.oldTime() is a
        // stored field it reads LAZILY in pEqn, by which point the geometry is current. brae builds the
        // old flux EAGERLY -- advanceTime turns Uf.oldTime() into phiOld against the current Sf -- so
        // running it on pre-move geometry pairs an old face velocity with old areas, which is neither
        // form OpenFOAM has. Measured: with the first update moved after advanceTime, movingCone went
        // 3.6e-03 -> 8.0e-03, mixerVesselAMI2D 8.8e-03 -> 1.9e-02 and oscillatingInletACMI2D
        // 1.3e-02 -> 2.0e-02. The additional moveMeshOuterCorrectors updates DO belong inside the loop:
        // by then the fields are aged and only the geometry is being re-converged.
        if (moveMesh) moveMesh(0);
        advanceTime(deltaT);                                  // store oldTime() + set deltaT/deltaT0 for ddtCoeffs()
        DeviceSimpleResidual res;
        const int nOuter = nOuterCorrectors > 0 ? nOuterCorrectors : 1;
        const int nCorr  = nCorrectors      > 0 ? nCorrectors      : 1;
        for (int oc = 0; oc < nOuter; ++oc)
        {
            // OF pimpleControl::loop -> mesh.data().setFinalIteration(true) on the LAST outer corrector.
            // That is the flag the bare solve() in UEqn.H and the turbulence models pick up, so U and
            // k/epsilon are on their Final entries for the whole of this iteration.
            // OF pimpleControl::finalIter() = converged_ || corr_ == nCorrPIMPLE_. residualControl can
            // therefore make an EARLIER iteration the final one: when the criteria are met at the top of
            // an iteration, OF flags that iteration final, runs it, and leaves the loop.
            // OF calls criteriaSatisfied() at the TOP of loop(), reading the residuals the PREVIOUS
            // iteration left in solverPerformanceDict -- so convergence detected here makes THIS
            // iteration the final one, not the next. Checking after the body instead runs one iteration
            // too many, which is what the contract test caught.
            // OF pimpleFoam.C:140 -- mesh.controlledUpdate() runs on the first outer iteration, or on
            // every one when moveMeshOuterCorrectors is set. Everything geometric that follows (AMI
            // rebuild, moving-wall velocity, wall distance, correctPhi) is the callback's business.
            if (moveMesh && oc > 0 && ctl_.moveMeshOuterCorrectors) moveMesh(oc);
            if (!ctl_.outerResidualControl.empty() && !outerConverged_)
                outerConverged_ = outerCriteriaSatisfied(res, oc);
            ctl_.finalIter = (oc == nOuter - 1) || outerConverged_;
            if (ctl_.solveFlow)
            {
            solveMomentumPredictor(res);                     // momentum incl. implicit ddt
            for (int pc = 0; pc < nCorr; ++pc)               // PIMPLE pressure (inner) correctors
            {
                // ...p is different: pEqn.H asks for p.select(pimple.finalInnerIter()), which is the LAST
                // pressure corrector regardless of which outer iteration this is, unless the case set
                // finalOnLastPimpleIterOnly. The non-orth half of finalInnerIter() is applied inside
                // correctPressureVelocity, which owns that loop.
                ctl_.finalInner = (pc == nCorr - 1) && (!ctl_.finalOnLastPimpleIterOnly || ctl_.finalIter);
                correctPressureVelocity(res);
            }
            }   // solveFlow
            // OF pimpleFoam.C: `if (pimple.turbCorr()) { laminarTransport.correct(); turbulence->correct(); }`
            // with turbCorr() = !turbOnFinalIterOnly || finalIter(). The outer loop iterates the
            // pressure-velocity coupling; the turbulence transport is advanced ONCE per time step unless
            // the case asks otherwise. Correcting it every outer iteration integrates the model
            // nOuterCorrectors times per physical step -- a different equation, not a tighter solve.
            ++outerIterations_;
            if (!ctl_.turbOnFinalIterOnly || ctl_.finalIter) { correctTurbulence(); ++turbCorrections_; }
            // residualControl: evaluated from the SECOND outer iteration (OF storeInitialResiduals()
            // returns corr_ == 2), where iteration 2's initial residuals become the reference the
            // relative test divides by. Never on the first iteration, and never once already final.
            if (ctl_.finalIter) break;   // OF leaves the loop after running the final iteration
        }
        outerConverged_ = false;
        outerInitialResidual_.clear();
        ctl_.finalIter = false;    // the steady path and anything outside the loop read the base entries
        ctl_.finalInner = false;
        // continuity errors on the corrected flux (identical to step()).
        {
            DeviceBuffer<scalar> divPhi;
            deviceDiv(dm, phiInt_, phiBnd_, divPhi);
            if (hasCyclic_) interfaceAddDiv(cyc_, dm.V, divPhi);   // interface faces are outside the DeviceBoundary
            if (hasAMI_)    interfaceAddDiv(ami_, dm.V, divPhi);   // (see step(): the pEqn's div includes them)
            DeviceBuffer<scalar> R;
            deviceHadamard(R, divPhi, dm.V);
            const scalar sumV = deviceDot(dm.V, ones_) + 1e-300;
            res.contLocal  = deviceSumMag(R) / sumV;
            res.contGlobal = deviceDot(R, ones_) / sumV;
        }
        return res;
    }

    void DeviceSimpleSolver::setMRF(
        const MRFZone& z,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<std::string>& nonRotating)
    {
        mrf_ = buildDeviceMRF(z, m, g, fvp_, nonRotating);
        deviceMrfApplyFrameFlux(mrf_, +1.0, phiInt_, phiBnd_);     // initial phi -> relative (OF createFields)
        // NOTE: MRF is NOT applied to the cyclic/cyclicAMI interface flux. Ground truth: OF never combines MRF with
        // cyclicAMI (0/12 MRF tutorials use AMI, all use conformal multi-zone meshes; rotating AMI is done by a
        // MOVING MESH in transient solvers, not MRF). So there is no MRF x AMI pattern to support here.
    }

    void DeviceSimpleSolver::setFvOptions(const FvOptionsData& fo)
    {
        hasFvoMom_ = fo.hasMomentum;
        if (fo.hasMomentum)
        {
            for (int c = 0; c < 3; ++c)
                fvoMomSu_[c].copyFrom(fo.momSu[c]);
            if (!fo.momSp.empty()) fvoMomSp_.copyFrom(fo.momSp);
        }
        for (const auto& s : fo.scaSu)
            fvoScaSu_[s.first].copyFrom(s.second);   // scalar (k/epsilon/...) sources
        for (const auto& s : fo.scaSp)
            fvoScaSp_[s.first].copyFrom(s.second);
        if (fo.porActive)
        {
            por_.active = true;
            por_.d = fo.porD;
            por_.f = fo.porF;
            por_.fixed = fo.porFixed;                                  // porosityModels::fixedCoeff
            por_.rhoRef = fo.porRhoRef;
            for (int k = 0; k < 9; ++k) { por_.fa[k] = fo.porAlphaT[k]; por_.fb[k] = fo.porBetaT[k]; }
            por_.cells.copyFrom(fo.porCells);
        }
        if (fo.mvfActive)
        {
            mvfActive_ = true;
            mvfRelax_ = fo.mvfRelax;
            const scalar m = std::sqrt(fo.mvfUbar.x*fo.mvfUbar.x + fo.mvfUbar.y*fo.mvfUbar.y + fo.mvfUbar.z*fo.mvfUbar.z);
            mvfUbarMag_ = m;
            mvfFlowDir_ = m > 0 ? vector{fo.mvfUbar.x/m, fo.mvfUbar.y/m, fo.mvfUbar.z/m} : vector{1,0,0};
            std::vector<scalar> m01(nC_, fo.mvfCells.empty() ? 1.0 : 0.0);   // selectionMode all -> ones; cellZone -> 1 in zone
            for (label c : fo.mvfCells)
                if (c >= 0 && c < nC_) m01[c] = 1.0;
            mvfMask01_.copyFrom(m01);
            deviceHadamard(mvfMaskV_, dm_.V, mvfMask01_);                     // maskV = V in the selection (0 else)
            mvfVtot_ = deviceSumMag(mvfMaskV_);                               // total selection volume
        }
        if (!fo.fixScaVals.empty() && !fo.fixScaCells.empty())
        {
            std::vector<label> mask(nC_, 0);
            for (label c : fo.fixScaCells) if (c >= 0 && c < nC_) mask[c] = 1;
            fixScaMask_.copyFrom(mask);
            const auto itk = fo.fixScaVals.find("k");
            if (itk != fo.fixScaVals.end()) { fixScaK_ = true; fixScaKVal_.copyFrom(std::vector<scalar>(nC_, itk->second)); }
            // the "second" turbulence scalar: epsilon on kEpsilon, omega on kOmegaSST -- brae holds both in de_
            auto ite = fo.fixScaVals.find(ctl_.sst ? "omega" : "epsilon");
            if (ite != fo.fixScaVals.end()) { fixScaE_ = true; fixScaEVal_.copyFrom(std::vector<scalar>(nC_, ite->second)); }
        }
        // BOTH fvOptions below turn a TEMPERATURE into an ENERGY with hConstTToHe, and both then keep the
        // result as a SCALAR. That is exact for a perfect gas, where he is a linear, pressure-independent
        // function of T. It is wrong twice over for a liquid: the correlation is a 5th-order polynomial,
        // and under sensibleInternalEnergy e = h(T) - p/rho(T) varies with pressure, so a single scalar
        // cannot represent the limit across the field at all.
        //
        // REFUSED RATHER THAN APPROXIMATED. Converting properly means carrying per-cell energy limits
        // recomputed against the live p, which is a real change and one no case in the suite exercises
        // yet (squareBendLiq sets no fvOptions). Silently converting with the gas formula is what the
        // rest of this fix exists to eliminate -- a plausible number from an inapplicable equation.
        if (tc_.model == ThermoModel::liquidH2O && (fo.fixTActive || fo.limTActive))
        {
            throw std::runtime_error(
                "brae: fvOptions limitTemperature/fixedTemperatureConstraint are not supported on the "
                "liquid thermo path yet. Both convert a temperature limit into a single energy scalar "
                "using the perfect-gas relation; for a liquid the energy is a polynomial in T and, under "
                "sensibleInternalEnergy, also depends on p, so the limit is a field rather than a "
                "constant. Remove the source, or run this case on the perfect-gas path.");
        }
        if (fo.fixTActive)
        {
            // OF: eqn.setValues(cells_, thermo.he(thermo.p(), Tuni, cells_)). The temperature is converted
            // ONCE with the case thermo, exactly as limitTemperature's limits are.
            fixTActive_ = true;
            std::vector<label> mask(nC_, 0);
            for (label c : fo.fixTCells) if (c >= 0 && c < nC_) mask[c] = 1;
            fixTMask_.isWallCell.copyFrom(mask);
            fixTHe_.copyFrom(std::vector<scalar>(nC_, hConstTToHe(fo.fixTTemp, tc_)));
        }
        if (fo.limTActive)
        {
            limTActive_ = true;
            limTAllCells_ = fo.limTAllCells;
            // OF converts the two TEMPERATURE limits into ENERGY limits once, with the case's own thermo
            // (limitTemperature.C: heMin = thermo.he(p, Tmin, cells)), and clamps he. With hConst the
            // conversion is exact and p-independent, so the two limits are scalars rather than fields.
            limTMin_ = hConstTToHe(fo.limTMin, tc_);
            limTMax_ = hConstTToHe(fo.limTMax, tc_);
            std::vector<label> tc = fo.limTCells;
            if (tc.empty())   // selectionMode all
            {
                tc.resize(nC_);
                for (label c = 0; c < nC_; ++c) tc[c] = c;
            }
            limTCells_.copyFrom(tc);
        }
        if (fo.limUActive)
        {
            limUActive_ = true;
            limUMax_ = fo.limUMax;
            std::vector<label> lc = fo.limUCells;
            if (lc.empty())   // selectionMode all
            {
                lc.resize(nC_);
                for (label c = 0; c < nC_; ++c)
                    lc[c] = c;
            }
            limUCells_.copyFrom(lc);
        }
        if (fo.vdcActive)
        {
            vdcActive_ = true;
            vdcUMax_ = fo.vdcUMax;
            vdcC_ = fo.vdcC;
            std::vector<label> vc = fo.vdcCells;
            if (vc.empty())   // selectionMode all
            {
                vc.resize(nC_);
                for (label c = 0; c < nC_; ++c)
                    vc[c] = c;
            }
            vdcCells_.copyFrom(vc);
        }
        if (fo.adActive)
        {
            adActive_ = true;
            adDisks_.clear();
            adDisks_.resize(fo.adDisks.size());
            for (std::size_t di = 0; di < fo.adDisks.size(); ++di)
            {
                const auto& src = fo.adDisks[di];
                auto& d = adDisks_[di];
                d.diskDir = src.diskDir;
                d.area    = src.area;
                d.a       = src.a;
                std::vector<scalar> dm01(nC_, 0.0);
                for (label c : src.diskCells)
                    if (c>=0 && c<nC_) dm01[c] = 1.0;
                DeviceBuffer<scalar> dmask;
                dmask.copyFrom(dm01);
                deviceHadamard(d.maskVDisk, dm_.V, dmask);
                d.vtot = deviceSumMag(d.maskVDisk);   // V in THIS disk / its own total volume
                std::vector<scalar> mon(nC_, 0.0);
                for (label c : src.monitorCells)
                    if (c>=0 && c<nC_) mon[c] = 1.0;
                d.monMask01.copyFrom(mon);
                d.nmon = (scalar)src.monitorCells.size();
            }
        }
        fvoCount_ = fo.count;
    }

}  // namespace brae
