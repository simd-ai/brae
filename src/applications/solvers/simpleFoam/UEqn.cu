// CUDA implementation -- see UEqn.cuh for the provenance and the contract with the _cpp reference.
#include "UEqn.cuh"
#include "device_blas.cuh"
#include "device_divdevreff.cuh"
#include "device_kepsilon.cuh"   // deviceGradUShared: grad(U) once per U state (item 65)
#include "device_simple.cuh"
#include <cmath>
#include <stdexcept>

namespace brae {
namespace gpu {

namespace {

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

    // ---- fvm::div(phi, U) -------------------------------------------------------------------
    // Upwind implicit weights, matching the reference's fvm::div. The weights of this operator are where
    // brae's LUST defect lived, which is why the CUDA-vs-reference test compares them coefficient by
    // coefficient rather than through a residual.
    // The scheme's own implicit weights. `upwind` and `linearUpwind` share the upwind weights -- the
    // latter is a deferred correction only -- so both take the plain kernel.
    switch (in.scheme)
    {
        case cpu::DivScheme::limitedLinearV:
        {
            // The kernel takes CONTIGUOUS 3-arrays (it indexes U[0..2]), not an array of pointers, so
            // the components are gathered into one. Three device copies per assembly; the alternative is
            // a second kernel signature.
            const DeviceBuffer<scalar>* Usrc[3] = {&Ux, &Uy, &Uz};
            DeviceBuffer<scalar> Uarr[3], gx[3], gy[3], gz[3];
            const GradUMemo& gm = deviceGradUShared(dm, dbU, Ux, Uy, Uz);   // grad(U) at this U, once (item 65)
            for (int k = 0; k < 3; ++k)
            {
                deviceCopy(Uarr[k], *Usrc[k]);
                deviceCopy(gx[k], gm.gx[k]);
                deviceCopy(gy[k], gm.gy[k]);
                deviceCopy(gz[k], gm.gz[k]);
            }
            deviceDivLimitedVCoeffs(dm, *in.phiInt, Uarr, gx, gy, gz,
                                    2.0 / std::fmax(in.schemeCoeff, 1e-15),
                                    M.diag, M.upper, M.lower);
            break;
        }
        case cpu::DivScheme::limitedLinear:
        {
            // limitedLinear on a VECTOR limits on the SCALAR magSqr(U) (LimitedScheme.H instantiates it
            // as NVDTVD + limitFuncs::magSqr), not per component and not the V form.
            DeviceBuffer<scalar> mag2, t, ub, gx, gy, gz, m2b;
            const DeviceBuffer<scalar>* U3[3] = {&Ux, &Uy, &Uz};
            mag2.resize(dm.nCells);
            for (int k = 0; k < 3; ++k)
            {
                deviceHadamard(t, *U3[k], *U3[k]);
                deviceAxpy(1.0, t, mag2);
            }
            m2b.resize(dm.nBndFaces);
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
    deviceDivDevReff(dm, dbU, Ux, Uy, Uz, *in.nuEffCell, *in.nuEffBndFace,
                     M.source[0], M.source[1], M.source[2],
                     /*cyc*/nullptr, /*ami*/nullptr, /*proc*/nullptr, /*UbStored*/nullptr,
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
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> gx, gy, gz, lu;
            deviceCopy(gx, gm.gx[k]);
            deviceCopy(gy, gm.gy[k]);
            deviceCopy(gz, gm.gz[k]);
            // `linearUpwind <name>` where <name> resolves to `cellLimited Gauss linear <k>`.
            if (in.gradULimitK > 0.0)
                deviceCellLimitGrad(dm, *U[k], gm.ub[k], gx, gy, gz, in.gradULimitK);
            deviceLinearUpwindCorr(dm, *in.phiInt, gx, gy, gz, lu);
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
            deviceMrfCoriolisZone(*in.mrf, dm.V, Ux, Uy, Uz, k, M.source[k]);
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
        deviceFvoPorosityDiag(*in.porosity, in.nuLaminar, dm.V, Ux, Uy, Uz, M.diag);
        for (int k = 0; k < 3; ++k)
            deviceFvoPorositySource(*in.porosity, k, in.nuLaminar, dm.V, Ux, Uy, Uz, M.source[k]);
    }

    // ---- UEqn.relax() -----------------------------------------------------------------------
    // OpenFOAM's fvMatrix::relax is ASYMMETRIC: it ADDS cmptMax(cmptMag(internalCoeffs)) to the diagonal
    // and REMOVES cmptMin(internalCoeffs), which are different quantities and agree only when the three
    // components are equal. Both are supplied here rather than approximated by |iC[0]|, so slip and
    // symmetry patches -- where the components genuinely differ -- stay right.
    M.relaxed = false;
    if (in.relaxU > 0.0 && in.relaxU < 1.0)
    {
        DeviceBuffer<scalar> iCmaxMag, iCmin;
        deviceCmptMaxMag3(M.iC[0], M.iC[1], M.iC[2], iCmaxMag);
        deviceCmptMin3   (M.iC[0], M.iC[1], M.iC[2], iCmin);

        const DeviceLduView A = M.view(dm);
        deviceRelaxDiag(A, dm, M.iC[0], in.relaxU, M.relaxedDiag, M.delta,
                        nullptr, iCmaxMag.data(), iCmin.data());
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
