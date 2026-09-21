// CUDA implementation -- see UEqn.cuh for the provenance and the contract with the _cpp reference.
#include "UEqn.cuh"
#include "device_inter_ueqn.cuh"
#include "device_blas.cuh"
#include "device_divdevreff.cuh"
#include "device_kepsilon.cuh"   // deviceGradUShared: grad(U) once per U state (item 65)
#include "device_simple.cuh"
#include <cmath>
#include <stdexcept>

namespace brae {
namespace gpu {

namespace {

// resize() does NOT zero: DevicePool::take hands back a RECYCLED block, the same contract as cudaMalloc
// (device_buffer.cuh:134-141). A buffer this file ACCUMULATES into rather than assigns must therefore be
// memset first. The limitedLinear branch below did not, and read the pool's leavings as part of
// magSqr(U) from the second assembly onward -- U 5.1e-01 from OpenFOAM on eulerianInjection where the
// host reads 3.9e-12. The compressible twin has carried this helper since its own port (rhoUEqn.cu).
void zeroBuffer(
    DeviceBuffer<scalar>& b,
    int n)
{
    b.resize(static_cast<std::size_t>(n));
    if (n <= 0) return;
    cudaCheck(cudaMemsetAsync(b.data(), 0, static_cast<std::size_t>(n)*sizeof(scalar)), "zero buffer");
}

void refuseUnsupported(const MomentumInput& in)
{
    // Identical wording and citations to the reference: a component that is out of scope must say so, not
    // be absent from the code and therefore from the reader's attention.
    if (in.hasMRF)
        throw std::runtime_error(
            "UEqn(cuda): the case declares MRF, which UEqn.H applies via MRF.correctBoundaryVelocity(U) "
            "and MRF.DDt(U) (simpleFoam/UEqn.H:3,8). Not implemented on this path; refusing rather than "
            "silently solving a different equation.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "UEqn(cuda): the case declares fvOptions, which UEqn.H applies as fvOptions(U), "
            "fvOptions.constrain(UEqn) and fvOptions.correct(U) (simpleFoam/UEqn.H:11,17,23). Not "
            "implemented on this path; refusing rather than silently solving a different equation.");
}

} // namespace


void assembleUEqn(
    MomentumMatrix&              M,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    const DeviceBuffer<scalar>&  Ux,
    const DeviceBuffer<scalar>&  Uy,
    const DeviceBuffer<scalar>&  Uz,
    const MomentumInput&         in)
{
    refuseUnsupported(in);

    // fvc::grad(U) ON A COUPLED MESH. gaussGrad sums EVERY face of a cell and a cyclic face is a face,
    // but deviceGradUShared walks the internal faces and the device's boundary arrays, which carry no
    // coupled face (device_mesh.cuh:41-44) -- so the pair's half was missing from every gradient this
    // assembly takes. The CLOSURE's grad(U) was given the same half in #28 (commit 44a8ba6); this is
    // the momentum path's, and it is applied to the LOCAL copies rather than to the shared memo, which
    // other sites and other iterations read.
    //
    // MEASURED on damBreakPorousBaffle with every solve pinned at 1e-16, at the second outer corrector
    // (the first starts from rest, where grad(U) is zero and this term cannot be seen): the momentum
    // source was 5.9392e-11 of 9.9062e-04 from the host arm, H 5.3135e-05 of 8.6143e+02 once divided
    // by V, HbyA 5.0952e-09 and U 3.6174e-09 -- against 1.5959e-15 for the same case meshed with no
    // pair at all.
    const DeviceBuffer<scalar>* UForGrad[3] = {&Ux, &Uy, &Uz};
    if (in.cyc && in.cyc->n > 0)
    {
        if (in.cyc->rotational)
            throw std::runtime_error(
                "brae device UEqn: grad(U) across a ROTATIONAL cyclic needs the neighbour rotated by "
                "forwardT (deviceCyclicAddGradRot, which takes all three components at once); this "
                "assembly adds the pair per component. Refusing rather than summing an un-rotated "
                "neighbour into the gradient.");
        if (in.gradULimitK > 0.0)
            throw std::runtime_error(
                "brae device UEqn: a cellLimited grad(U) across a coupled patch would limit against a "
                "neighbour it cannot see -- OF's cellLimitedGrad treats a cyclic face as internal, and "
                "this arm's limiter walks the internal faces and the non-coupled patches only. The "
                "Gauss half is summed across the pair here; the limiter's is not ported. Refusing.");
    }
    auto addPairToGrad = [&](int k,
                             DeviceBuffer<scalar>& gx,
                             DeviceBuffer<scalar>& gy,
                             DeviceBuffer<scalar>& gz)
    {
        if (!in.cyc || in.cyc->n == 0) return;
        deviceCyclicAddGrad(*in.cyc, *UForGrad[k], dm.V, gx, gy, gz);
    };

    // ---- fvm::div(phi, U) -------------------------------------------------------------------
    // Upwind implicit weights, matching the reference's fvm::div. The weights of this operator are where
    // brae's LUST defect lived, which is why the CUDA-vs-reference test compares them coefficient by
    // coefficient rather than through a residual.
    // The scheme's own implicit weights. `upwind` and `linearUpwind` share the upwind weights -- the
    // latter is a deferred correction only -- so both take the plain kernel.
    switch (in.scheme)
    {
        case cpu::DivScheme::limitedLinearV:
        case cpu::DivScheme::vanLeerV:
        {
            // The kernel takes CONTIGUOUS 3-arrays (it indexes U[0..2]), not an array of pointers, so
            // the components are gathered into one. Three device copies per assembly; the alternative is
            // a second kernel signature. vanLeerV is the same NVDVTVDV r with vanLeer's limiter, which
            // the kernel selects by the sentinel (device_mesh.cuh, kVanLeerTwoByk).
            const DeviceBuffer<scalar>* Usrc[3] = {&Ux, &Uy, &Uz};
            DeviceBuffer<scalar> Uarr[3], gx[3], gy[3], gz[3];
            const GradUMemo& gm = deviceGradUShared(dm, dbU, Ux, Uy, Uz);   // grad(U) at this U, once (item 65)
            for (int k = 0; k < 3; ++k)
            {
                deviceCopy(Uarr[k], *Usrc[k]);
                deviceCopy(gx[k], gm.gx[k]);
                deviceCopy(gy[k], gm.gy[k]);
                deviceCopy(gz[k], gm.gz[k]);
                addPairToGrad(k, gx[k], gy[k], gz[k]);
            }
            const scalar twoByk = (in.scheme == cpu::DivScheme::vanLeerV)
                ? kVanLeerTwoByk
                : 2.0 / std::fmax(in.schemeCoeff, 1e-15);
            deviceDivLimitedVCoeffs(dm, *in.phiInt, Uarr, gx, gy, gz, twoByk, M.diag, M.upper, M.lower);
            break;
        }
        case cpu::DivScheme::limitedLinear:
        {
            // limitedLinear on a VECTOR limits on the SCALAR magSqr(U) (LimitedScheme.H instantiates it
            // as NVDTVD + limitFuncs::magSqr), not per component and not the V form.
            DeviceBuffer<scalar> mag2, t, ub, gx, gy, gz, m2b;
            const DeviceBuffer<scalar>* U3[3] = {&Ux, &Uy, &Uz};
            // ZEROED, not merely sized: these two are accumulated into, and resize() recycles
            zeroBuffer(mag2, dm.nCells);
            for (int k = 0; k < 3; ++k)
            {
                deviceHadamard(t, *U3[k], *U3[k]);
                deviceAxpy(1.0, t, mag2);
            }
            zeroBuffer(m2b, dm.nBndFaces);
            for (int k = 0; k < 3; ++k)
            {
                deviceBCValue(dbU.comp[k], *U3[k], ub);
                deviceHadamard(t, ub, ub);
                deviceAxpy(1.0, t, m2b);
            }
            deviceGaussGrad(dm, mag2, m2b, gx, gy, gz);
            deviceDivLimitedCoeffs(dm, *in.phiInt, mag2, gx, gy, gz,
                                   2.0 / std::fmax(in.schemeCoeff, 1e-15),
                                   M.diag, M.upper, M.lower);
            break;
        }
        case cpu::DivScheme::linear:
            // central differencing: the mesh's own weights. THIS FELL THROUGH TO `default` -- upwind
            // -- until interFoam's driver was read: its own scheme enum has `linear` (three shipped
            // tutorials name it for div(rhoPhi,U)) and its device mapping had no case for it, so a
            // -device run of such a case would have convected upwind under the name `linear`. None of
            // the three runs yet for other reasons; the dambreak gate's `linear` profile holds it now.
            deviceDivCentralCoeffs(dm, *in.phiInt, M.diag, M.upper, M.lower);
            break;

        case cpu::DivScheme::LUST:
        {
            // weights = 0.75*linear + 0.25*upwind, and the coefficients are LINEAR in the weights
            // (lower = -w*phi, upper = lower + phi, diag = negSumDiag), so blending the two coefficient
            // sets is exact rather than an approximation of the blended-weight kernel.
            DeviceBuffer<scalar> cD, cU, cL, uD, uU, uL;
            deviceDivCentralCoeffs(dm, *in.phiInt, cD, cU, cL);
            deviceDivUpwindCoeffs (dm, *in.phiInt, uD, uU, uL);
            deviceCopy(M.diag,  cD); deviceScale(M.diag,  0.75); deviceAxpy(0.25, uD, M.diag);
            deviceCopy(M.upper, cU); deviceScale(M.upper, 0.75); deviceAxpy(0.25, uU, M.upper);
            deviceCopy(M.lower, cL); deviceScale(M.lower, 0.75); deviceAxpy(0.25, uL, M.lower);
            break;
        }
        default:
            deviceDivUpwindCoeffs(dm, *in.phiInt, M.diag, M.upper, M.lower);
            break;
    }

    // ---- - fvm::laplacian(nuEff, U) ---------------------------------------------------------
    // The implicit half of divDevReff. Face nuEff is passed in already interpolated, and the BOUNDARY
    // faces carry the patch value (nut_wall on a wall function), not the owner cell's.
    {
        DeviceBuffer<scalar> lD, lU, lL;
        deviceLaplacianCoeffs(dm, *in.nuEffFace, lD, lU, lL, in.correctedLaplacian);
        deviceAxpy(-1.0, lD, M.diag);
        deviceAxpy(-1.0, lU, M.upper);
        deviceAxpy(-1.0, lL, M.lower);
    }

    // ---- boundary coefficients, per component -----------------------------------------------
    // A vector boundary is three scalar boundaries. The div and laplacian internalCoeffs are isotropic;
    // only the refValue-dependent boundaryCoeffs differ by component, so the scalar kernels are reused
    // on dbU.comp[k] exactly as the rest of brae does.
    for (int k = 0; k < 3; ++k)
    {
        deviceBCDivCoeffs(dbU.comp[k], *in.phiBnd, M.iC[k], M.bC[k]);
        DeviceBuffer<scalar> lIC, lBC;
        deviceBCLaplacianCoeffsFace(dbU.comp[k], *in.nuEffBndFace, lIC, lBC);
        deviceAxpy(-1.0, lIC, M.iC[k]);
        deviceAxpy(-1.0, lBC, M.bC[k]);
    }

    // ---- `bounded`: - fvm::Sp(fvc::div(phi), U) ---------------------------------------------
    // Before relax, as OpenFOAM does. deviceDiv returns the per-volume divergence, so the extensive
    // diagonal contribution is V*div(phi).
    if (in.bounded)
    {
        DeviceBuffer<scalar> divPhi, t;
        deviceDiv(dm, *in.phiInt, *in.phiBnd, divPhi);
        deviceHadamard(t, divPhi, dm.V);
        deviceAxpy(-1.0, t, M.diag);
    }

    // ---- explicit divDevReff: -fvc::div(nuEff*dev2(T(grad U))) ------------------------------
    // The kernel returns the EXTENSIVE V*div(sigma), which is exactly what the reference adds to `source`
    // (it computes -div(sigma) per volume and then subtracts it times V). So this is the source, directly.
    // ...WITH THE PAIR. divDevReff is an EXPLICIT term and a periodic face is in it twice: in the
    // fvc::grad(U) the deviatoric stress is built from, and in the fvc::div of that stress. The
    // function carries both already; this call handed it a null pair. It is identically zero on a case
    // at rest -- grad(U) is zero -- and live from the second step: MEASURED on
    // validation/interFoamCyclic, UEqn's source 2.4e-05 of 2.5e-01 from the host at step two, which is
    // HbyA 5.3e-04 of 1.9e+00 at the pair's own cells and U 2.1e-04 by the end of the step.
    deviceDivDevReff(dm, dbU, Ux, Uy, Uz, *in.nuEffCell, *in.nuEffBndFace,
                     M.source[0], M.source[1], M.source[2],
                     in.cyc, /*ami*/nullptr, /*proc*/nullptr, in.UbStored,
                     // The gradSchemes `grad(U)` entry, which linearViscousStress.C:114's fvc::grad(U)
                     // resolves. These five arguments fell through to their defaults, so the case's
                     // limiter never reached the dev2 term on this driver -- the legacy one has passed
                     // it since device_simple_foam.cu.
                     in.gradUSchemeLimitK);

    // ---- explicit non-orthogonal correction --------------------------------------------------
    // AFTER divDevReff, which ASSIGNS the source (device_divdevreff.cu: `dX[c] = d[0]`) rather than
    // accumulating into it. Adding the correction first compiles and runs and is silently discarded.
    //
    // deviceLaplacianCorr returns the LAPLACIAN's own source correction (-V*fvc::div(faceFluxCorr)).
    // divDevReff carries MINUS the laplacian, so it enters the momentum source with the opposite sign --
    // the bookkeeping the existing GPU driver does at device_simple_foam.cu:1423-1426, and the sign a
    // measurement against real OpenFOAM had to settle on the reference side (backwards made U worse).
    if (in.correctedLaplacian)
    {
        const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
        DeviceBuffer<scalar> gxc[3], gyc[3], gzc[3];
        const GradUMemo& gm = deviceGradUShared(dm, dbU, Ux, Uy, Uz);       // the same grad(U) as the sites below
        for (int k = 0; k < 3; ++k)
        {
            deviceCopy(gxc[k], gm.gx[k]);
            deviceCopy(gyc[k], gm.gy[k]);
            deviceCopy(gzc[k], gm.gz[k]);
            addPairToGrad(k, gxc[k], gyc[k], gzc[k]);
        }
        if (in.snGradLimitCoeff > 0.0)
        {
            // `limited <k> corrected`. OF's limitedSnGrad takes mag() of the WHOLE snGrad and of the
            // WHOLE correction, so all three components share one per-face limiter -- which is why this
            // cannot be done inside the per-component loop above.
            DeviceBuffer<scalar> ffc[3];
            deviceLaplacianCorrFluxLimitedVec(dm, *in.nuEffFace, *U[0], *U[1], *U[2],
                                              gxc, gyc, gzc, in.snGradLimitCoeff, ffc);
            for (int k = 0; k < 3; ++k)
            {
                DeviceBuffer<scalar> lc;
                deviceFaceDivSource(dm, ffc[k], lc);
                deviceAxpy(-1.0, lc, M.source[k]);
            }
        }
        else
        {
            for (int k = 0; k < 3; ++k)
            {
                DeviceBuffer<scalar> lc;
                deviceLaplacianCorr(dm, *in.nuEffFace, gxc[k], gyc[k], gzc[k], lc);
                deviceAxpy(-1.0, lc, M.source[k]);
            }
        }
    }

    // ---- linearUpwind's deferred correction --------------------------------------------------
    // AFTER divDevReff for the same reason as the block above: that call ASSIGNS the source.
    //
    // OpenFOAM applies this inside fvm::div, but it only ever touches `source`, so its position among the
    // other source contributions is free -- what is NOT free is being after the assignment and before
    // relax(), which reads the source. SUBTRACTED, because `fvm += fvc::surfaceIntegrate(...)` on an
    // fvMatrix means `source -= V*...` (fvMatrix.C:1855-1862).
    //
    // Per component with the SCALAR gradient of that component, which is what OpenFOAM's `vector`
    // specialisation computes as one tensor grad: (d & grad(U))_j = d . grad(U_j) under OpenFOAM's
    // grad(U)_ij = d(U_j)/d(x_i) convention. The two are the same field, not an approximation of it.
    // How much of linearUpwind's correction this scheme carries: 1 for linearUpwind, 0.25 for LUST
    // (LUST.H overrides correction() too), 0 otherwise.
    // linearUpwindV: a DIFFERENT correction, limited across the three components at once, so it cannot
    // be expressed as a factor on linearUpwind's.
    if (in.scheme == cpu::DivScheme::linearUpwindV)
    {
        const DeviceBuffer<scalar>* Usrc[3] = {&Ux, &Uy, &Uz};
        DeviceBuffer<scalar> gx[3], gy[3], gz[3], cx, cy, cz;
        const GradUMemo& gm = deviceGradUShared(dm, dbU, Ux, Uy, Uz);
        for (int k = 0; k < 3; ++k)
        {
            deviceCopy(gx[k], gm.gx[k]);
            deviceCopy(gy[k], gm.gy[k]);
            deviceCopy(gz[k], gm.gz[k]);
            addPairToGrad(k, gx[k], gy[k], gz[k]);
            if (in.gradULimitK > 0.0)
                deviceCellLimitGrad(dm, *Usrc[k], gm.ub[k], gx[k], gy[k], gz[k], in.gradULimitK);
        }
        deviceLinearUpwindVCorr(dm, *in.phiInt, gx, gy, gz, Ux, Uy, Uz, cx, cy, cz);
        const DeviceBuffer<scalar>* cc[3] = {&cx, &cy, &cz};
        for (int k = 0; k < 3; ++k) deviceAxpy(-1.0, *cc[k], M.source[k]);
    }

    const scalar corrFac = (in.scheme == cpu::DivScheme::linearUpwind || in.linearUpwind) ? 1.0
                         : (in.scheme == cpu::DivScheme::LUST)                            ? 0.25
                         : 0.0;
    if (corrFac != 0.0)
    {
        const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
        const GradUMemo& gm = deviceGradUShared(dm, dbU, Ux, Uy, Uz);
        // ALL THREE components' gradients are built first, because the pair's correction reconstructs
        // the neighbour in the NEIGHBOUR's frame: on a rotational cyclic the component that comes back
        // is forwardT . (gradU[nbr] . dNbr), which mixes all three (device_cyclic.cuh). Per-component
        // buffers cannot express that, so they are hoisted out of the loop below.
        DeviceBuffer<scalar> gxA[3], gyA[3], gzA[3];
        for (int k = 0; k < 3; ++k)
        {
            deviceCopy(gxA[k], gm.gx[k]);
            deviceCopy(gyA[k], gm.gy[k]);
            deviceCopy(gzA[k], gm.gz[k]);
            addPairToGrad(k, gxA[k], gyA[k], gzA[k]);
            // `linearUpwind <name>` where <name> resolves to `cellLimited Gauss linear <k>`.
            if (in.gradULimitK > 0.0)
                deviceCellLimitGrad(dm, *U[k], gm.ub[k], gxA[k], gyA[k], gzA[k], in.gradULimitK);
        }
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> lu;
            deviceLinearUpwindCorr(dm, *in.phiInt, gxA[k], gyA[k], gzA[k], lu);
            // ...and the PAIR's faces, which that kernel's internal-face loop does not reach. The host
            // reference adds them (fvm::addLinearUpwindCorrectionCoupled, called from every interFoam
            // UEqn assembly); this arm had the kernel for it since the legacy driver and never called
            // it. It accumulates into the SAME buffer, so both halves leave through one axpy with the
            // scheme's share -- 1 for linearUpwind, 0.25 for LUST.
            if (in.cyc && in.cyc->n > 0)
            {
                if (!in.cycConvFlux)
                    throw std::runtime_error(
                        "brae device UEqn: linearUpwind's deferred correction across a coupled patch "
                        "must be weighted by the SAME flux the matrix was assembled with "
                        "(MomentumInput::cycConvFlux). Refusing rather than weighting it with the "
                        "interface's volumetric phi on an equation that convects with rhoPhi.");
                deviceCyclicAddLinUpwindCorr(*in.cyc, k, gxA, gyA, gzA, lu, in.cycConvFlux);
            }
            deviceAxpy(-corrFac, lu, M.source[k]);
        }
    }

    // ---- == fvOptions(U) ----------------------------------------------------------------------
    // BEFORE relax, as UEqn.H has it. The diagonal takes the isotropic part implicitly and the source
    // the anisotropic remainder, which is what keeps a 5e7 Darcy coefficient stable.
    // + MRF.DDt(U), UEqn.H:8. Part of the LHS expression, so it is in the matrix BEFORE relax -- the
    // same slot fvOptions' source occupies. EXPLICIT in U: OpenFOAM builds a volVectorField of
    // Omega x U from the current U rather than an implicit Coriolis operator, so it is lagged like any
    // other deferred term and lands as source -= V*(Omega x U).
    if (in.mrf && !in.mrf->empty())
    {
        for (int k = 0; k < 3; ++k)
        {
            if (!in.mrfRho)
            {
                deviceMrfCoriolisZone(*in.mrf, dm.V, Ux, Uy, Uz, k, M.source[k]);
                continue;
            }
            // ...and rho*DDt(U) where the caller's equation is rho-weighted, as rhoSimpleFoam's device
            // arm forms it (rhoUEqn.cu:769-777): the kernel writes -V*(Omega x U) into a zeroed
            // accumulator, which is then multiplied by rho and added. acc already carries the volume and
            // the sign, and rho is a cell field, so rho*(a*V) == (rho*a)*V -- a rearrangement, not another
            // term, and the same bits as the host's `source += rho*acc` (inter_ueqn_cpp.cu:210-220).
            // the accumulator is ACCUMULATED into, so it is memset first: DeviceBuffer's pool hands back
            // a same-size block that still holds the last user's numbers (rhoUEqn.cu:28-43)
            DeviceBuffer<scalar> acc, t;
            acc.resize(static_cast<std::size_t>(dm.nCells));
            if (dm.nCells > 0)
            {
                cudaCheck(cudaMemsetAsync(acc.data(), 0,
                                          static_cast<std::size_t>(dm.nCells)*sizeof(scalar),
                                          cudaStreamPerThread),
                          "UEqn MRF acc zero");
            }
            deviceMrfCoriolisZone(*in.mrf, dm.V, Ux, Uy, Uz, k, acc);
            deviceHadamard(t, acc, *in.mrfRho);
            deviceAxpy(1.0, t, M.source[k]);
        }
    }

    // == fvOptions(U): rotorDiskSource. Two operators, not one. addSup does `eqn -= force` with force
    // PER VOLUME, and fvMatrix::operator-=(DimensionedField) is `source() += V*su`, so the OPTION
    // matrix's source gains V*force. simpleFoam then writes `UEqn == fvOptions(U)`, and the free
    // operator== is `UEqn - fvOptions(U)` -- so the MOMENTUM source LOSES it. The blade-element force is
    // the force on the BLADE; what the fluid feels is the reaction, which is what that minus delivers.
    if (in.rotor && in.rotor->active)
    {
        DeviceBuffer<scalar> fx, fy, fz;
        deviceRotorForce(*in.rotor, dm.nCells, Ux, Uy, Uz, fx, fy, fz);
        DeviceBuffer<scalar>* fc[3] = {&fx, &fy, &fz};
        for (int k = 0; k < 3; ++k) deviceAxpy(-1.0, *fc[k], M.source[k]);
    }

    // == fvOptions(U): actuationDiskSource (Froude). Applied AFTER the rotor for no reason but order;
    // the two sources simply superpose, as any number of turbines on one mesh do.
    if (in.actuationDisk && in.actuationDisk->active)
    {
        DeviceBuffer<scalar>* src[3] = {&M.source[0], &M.source[1], &M.source[2]};
        deviceActuationDiskAddSup(*in.actuationDisk, Ux, Uy, Uz, src);
    }

    if (in.porosity && in.porosity->active)
    {
        deviceFvoPorosityDiag(*in.porosity, in.nuLaminar, dm.V, Ux, Uy, Uz, M.diag,
                              in.porosityMu, in.porosityRho);
        for (int k = 0; k < 3; ++k)
            deviceFvoPorositySource(*in.porosity, k, in.nuLaminar, dm.V, Ux, Uy, Uz, M.source[k],
                                    in.porosityMu, in.porosityRho);
    }

    // ---- the PERIODIC PAIR's momentum coupling ----------------------------------------------
    // Gated face by face against the host's own coupled coefficients in
    // tests/test_device_cyclic_laplacian_vs_host.cu. It goes in before relax, with every other
    // coefficient, because relax reads the diagonal it leaves.
    if (in.cyc && in.cyc->n > 0)
    {
        deviceCyclicAssembleMomentum(*in.cyc, *in.nuEffCell, M.diag, nullptr, in.cycCorrected,
                                     in.cycConvFlux);
        // ...and kept, because the next assembly on this pair overwrites cyc.ifCoeff -- see the note
        // on MomentumMatrix::cycIfCoeff.
        deviceCopy(M.cycIfCoeff, in.cyc->ifCoeff);
    }

    // ---- fvm::ddt(rho, U), for a transient momentum equation --------------------------------
    // BEFORE relax(), as the fvMatrix constructor's `+` puts it. rho and rho.oldTime() are separate
    // fields on purpose -- see device_inter_ueqn.cuh.
    if (in.ddtRho)
    {
        if (!in.ddtRhoOld || !in.ddtUOld[0] || !in.ddtUOld[1] || !in.ddtUOld[2])
            throw std::runtime_error(
                "brae momentum: a transient ddt needs rho, rho.oldTime() and all three components of "
                "U.oldTime(). rho.oldTime() is NOT rho at a VoF interface -- they differ by the density "
                "ratio -- so it is a separate argument and cannot be defaulted to the first.");
        deviceInterEulerDdtRhoU(dm, *in.ddtRho, *in.ddtRhoOld,
                                *in.ddtUOld[0], *in.ddtUOld[1], *in.ddtUOld[2], in.ddtDeltaT,
                                M.diag, M.source[0], M.source[1], M.source[2], in.ddtV0);
    }

    // ---- UEqn.relax() -----------------------------------------------------------------------
    // OpenFOAM's fvMatrix::relax is ASYMMETRIC: it ADDS cmptMax(cmptMag(internalCoeffs)) to the diagonal
    // and REMOVES cmptMin(internalCoeffs), which are different quantities and agree only when the three
    // components are equal. Both are supplied here rather than approximated by |iC[0]|, so slip and
    // symmetry patches -- where the components genuinely differ -- stay right.
    M.relaxed = false;
    if (in.relaxU > 0.0 && (in.relaxU < 1.0 || in.relaxEquation))
    {
        DeviceBuffer<scalar> iCmaxMag, iCmin;
        deviceCmptMaxMag3(M.iC[0], M.iC[1], M.iC[2], iCmaxMag);
        deviceCmptMin3   (M.iC[0], M.iC[1], M.iC[2], iCmin);

        const DeviceLduView A = M.view(dm);
        // ...and the pair's off-diagonal in the dominance term: without it the clamp is computed
        // against a row that is missing its periodic neighbour (device_simple.cuh, cycSumOff).
        DeviceBuffer<scalar> cycSumOff;
        if (in.cyc && in.cyc->n > 0)
        {
            cycSumOff.resize(static_cast<std::size_t>(dm.nCells));
            cudaCheck(cudaMemsetAsync(cycSumOff.data(), 0,
                                      static_cast<std::size_t>(dm.nCells)*sizeof(scalar),
                                      cudaStreamPerThread), "cyclic sumOff zero");
            deviceCyclicOffDiagSum(*in.cyc, cycSumOff);
        }
        deviceRelaxDiag(A, dm, M.iC[0], in.relaxU, M.relaxedDiag, M.delta,
                        (in.cyc && in.cyc->n > 0) ? cycSumOff.data() : nullptr,
                        iCmaxMag.data(), iCmin.data());
        M.relaxed = true;

        // source += (relaxedDiag - rawDiag) * psi, per component -- the reference's
        //     M.source[c] += (M.diag[c] - D0[c]) * psi.internal[c]
        const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> t;
            deviceHadamard(t, M.delta, *U[k]);
            deviceAxpy(1.0, t, M.source[k]);
        }
    }
}


void addPressureGradient(
    MomentumMatrix&              M,
    const DeviceMesh&            dm,
    const DeviceBuffer<scalar>&  gradPx,
    const DeviceBuffer<scalar>&  gradPy,
    const DeviceBuffer<scalar>&  gradPz)
{
    // solve(UEqn == -fvc::grad(p)): source -= grad(p)*V. fvc::grad is per-volume and `source` is
    // extensive, so the volume factor is explicit here as it is in the reference.
    const DeviceBuffer<scalar>* gp[3] = {&gradPx, &gradPy, &gradPz};
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> t;
        deviceHadamard(t, *gp[k], dm.V);
        deviceAxpy(-1.0, t, M.source[k]);
    }
}

} // namespace gpu
} // namespace brae
