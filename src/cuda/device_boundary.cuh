#pragma once
// cf GPU offload, on-device boundary-condition evaluation. Encodes each boundary face's BC category
// (0=extrapolated, 1=fixedValue, 2=calculated) + its refValue / deltaCoeffs / |Sf| / faceCell, then
// computes on device: the boundary value() (from the internal field) and the matrix boundary coefficients
// (laplacian internalCoeffs/boundaryCoeffs from a diffusivity, div internalCoeffs/boundaryCoeffs from a
// face flux). This removes the host round-trip: the SIMPLE step's boundary coeffs are built on the GPU.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "fv_geometry.cuh"
#include "geometric_field.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>

#include <cstdlib>
#include <string>

namespace brae {

// See DeviceBoundary::ioStored. Read once; "0" disables the stored-value seed.
inline bool ioStoredEnabled()
{
    static const bool on = !(std::getenv("BRAE_IO_STORED") && std::string(std::getenv("BRAE_IO_STORED")) == "0");
    return on;
}

struct DeviceBoundary
{
    int n = 0;                              // total boundary faces (patch order = DeviceMesh bndCell order)
    DeviceBuffer<label>  bcType;            // 0 extrapolated, 1 fixedValue, 2 calculated (3=inletOutlet -> resolved
                                            // to 0|1 per face each step by deviceUpdateInletOutlet; 5=mixed/Robin;
                                            // 8=COUPLED (processor): zero matrix coeffs, value injected by the halo)
    // fvPatchField::assignable() per face -- may an ASSIGNMENT to the field overwrite this patch's value?
    // False for the fixedValue, mixed and transform families; inletOutlet overrides it back to TRUE
    // (fv_patch_field.cuh:57-67). Carried explicitly rather than derived from bcType, because bcType
    // resolves an inletOutlet to fixedValue on an inflow face and that is exactly the case the two
    // questions disagree on. Foam::bound's boundary half (bound.C:59) is an assignment, so it is the
    // consumer: without this the device arms clamped nothing there, and clamping everything would move
    // wall-function faces OpenFOAM leaves alone.
    DeviceBuffer<label>  assignableMask;
    DeviceBuffer<label>  ioMask;            // 1 if the face is inletOutlet (bcType recomputed from the flux sign)
    DeviceBuffer<label>  oioMask;           // 1 if the face is outletInlet (freestreamPressure): opposite flux switch
    // The STORED patch value of an inletOutlet / outletInlet face, and whether it is still what an evaluate
    // must return. OpenFOAM constructs both with valueFraction 0 and the file's `value` (or the cells,
    // absent one) and first runs the flux switch inside the first assembly's updateCoeffs; until then every
    // fvc::grad / interpolate reads that stored value. This arm keeps no stored patch values, so before
    // its first deviceUpdateInletOutlet an evaluate could only return what the construction-time
    // coefficients give -- fixedValue(inletValue) here, which is neither the stored value on an outflow
    // face (the owner cell) nor on an inflow one (the last refValue). Measured on gasMixing/injectorPipe
    // restarted from OpenFOAM's own iteration 5: the closure's assembled epsilon off-diagonals 1.6e-04
    // off the host with 100% of it on outlet-adjacent faces (461 outflow faces, OpenFOAM's written value
    // equal to the owner cell on every one); seeded zeroGradient instead, 8.8e-04 with 100% of it on the
    // two turbulent inlets. From rest (inletValue == internalField) nothing showed. ioFresh is cleared by
    // the first flux switch, after which the coefficients ARE the last evaluate.
    // inletOutlet ONLY, and ONLY where the builder is asked (storedIoSeed): the closure's k and
    // epsilon|omega boundaries, whose stored value the closure reconstructs at its first step (kEpsilon.cu,
    // kOmegaSST.cu) -- measured on the gasMixing restart above, CUDA 1.2e-11 of OpenFOAM after. Seeding
    // the SOLVER fields the same way went the other way on validation/restart_vs_openfoam.sh
    // (aerofoilNACA0012 restarted, T/k/omega inletOutlet on the freestream, thermo-derived T patch):
    // p 1.5e-05 -> 1.16e-03 and T 1.6e-04 -> 4.2e-03 against OpenFOAM, so U/p/he/T keep their
    // fixedValue(inletValue) construction seed. Recorded as OPEN in the manifest: which construction
    // state OpenFOAM's solver fields effectively see at a restart has not been named by a stage
    // measurement yet, and a seed that helps one gate and hurts another is not a port.
    // BRAE_IO_STORED=0 disables the seed (ioFresh all zero): the control that reproduces the
    // fixedValue(inletValue) construction state from a shipped binary, as BRAE_DILU_KE does for the
    // preconditioner. Never the default.
    DeviceBuffer<scalar> ioStored;
    DeviceBuffer<label>  ioFresh;
    DeviceBuffer<label>  mixedMask;         // 1 if the face is mixed/Robin (freestreamVelocity/Pressure): vf recomputed
    DeviceBuffer<label>  piovMask;          // 1 if the face is pressureInletOutletVelocity (bcType+refValue recomputed)
    DeviceBuffer<label>  symMask;            // 1 if the face is slip/symmetry (per-comp vf=|n_k|, ref recomputed each step)
    // wedge (axisymmetric constraint): 1 on a wedge face, with that patch's HALF-angle rotation packed
    // per face (9 per face, row-major). The valueFraction is geometry and set once; only refValue is
    // recomputed each step, from the rotated cell velocity -- see deviceUpdateWedge.
    DeviceBuffer<label>  wedgeMask;
    DeviceBuffer<scalar> wedgeT;             // 9*n, faceT
    // THE GRADIENT FIELD'S CONSTRAINT TYPE ON A SYMMETRY PLANE. fvc::grad(U) is built with
    // extrapolatedCalculated patches, but fvPatchField::New puts a constraint patch's OWN type in their
    // place -- keyed on the MESH patch type, not on U's condition (fvPatchFieldNew.C:57-64) -- and
    // gaussGrad ends in correctBoundaryConditions, so on a `symmetry` or `symmetryPlane` patch the
    // gradient's value is the cell gradient averaged with its mirror image, (G + R G R^T)/2 with
    // R = I - 2 nn (basicSymmetryFvPatchField::evaluate). gradSymMask is 1 on those faces and gradSymN
    // the normal the mirror takes, three per face: the face's own Sf/magSf on a `symmetry`, the
    // patch's ONE area-weighted normal on a `symmetryPlane` (symmetryPlanePolyPatch.C:48-57).
    // NOT symMask: that one follows U's BC and is 1 on a `slip` wall too, where OpenFOAM's gradient
    // keeps the plain cell value.
    DeviceBuffer<label>  gradSymMask;
    DeviceBuffer<scalar> gradSymN;           // 3*n, row per face
    DeviceBuffer<label>  tpMask;             // 1 if the face is totalPressure (fixedValue-p, refValue recomputed each step)
    DeviceBuffer<scalar> valueFraction;     // per-face vf for mixed faces (deviceUpdateMixedFreestream); blends 0->1
    // fixedGradient's prescribed normal gradient, per face. ZERO for every other BC, which is what makes
    // one code path serve both: OF's fixedGradient IS zeroGradient plus a source proportional to g.
    DeviceBuffer<scalar> refGrad;
    // 1 on a fixedFluxPressure face: the SOLVER overwrites refGrad there every pressure assembly
    // (deviceConstrainPressure), exactly as constrainPressure.C hands updateSnGrad the flux-consistent
    // gradient. Empty mask = no such faces, and the kernel is skipped entirely.
    DeviceBuffer<label>  snGradMask;
    label                nSnGradFaces = 0;   // host-side count, for cheap has-any checks at call sites
    DeviceBuffer<scalar> refValue, p0, deltaCoeffs, magSf;   // p0 = totalPressure reference (constant; refValue = p0 - 0.5*neg(phi)|U|^2)
    DeviceBuffer<label>  faceCell;
};

inline DeviceBoundary buildDeviceBoundary(
    const GeometricField<scalar>& f,
    const std::vector<FvPatch>& fvp,
    const FvGeometry& g,
    // Seed inletOutlet faces with the field's stored value until the first flux switch (ioStored). ON for
    // the closure fields only -- see DeviceBoundary::ioStored for the two measurements that bound it.
    bool storedIoSeed = false)
{
    std::vector<label> ty, fc, io, oio, mx, pv, sm, tp, sg, asg, iofr;
    std::vector<scalar> ref, dc, ms, vf, p0, rg, iost;   // rg = fixedGradient normal gradient (0 elsewhere)
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;                     // cyclic = internal-like (handled by appended faces)
        // A processor patch is COUPLED: it stays in the boundary gather (the explicit operators need its bval
        // slot, filled with the halo-interpolated face value by DeviceHalo::scatterBoundaryValues), but must
        // contribute NO matrix coefficients -- its coupling is the interface off-diagonal. bcCategory() is NOT
        // overridden on ProcessorFvPatchField, so it would otherwise report 0 (= zeroGradient) whose
        // valueInternalCoeffs is 1, DOUBLE-COUNTING the interface diagonal.
        const int cat = (fvp[pi].type == "processor") ? 8 : f.boundary[pi]->bcCategory();
        // The REFERENCE value, not the current one: the mixed/inletOutlet evaluators blend towards it, and
        // a restarted field carries the value OpenFOAM wrote in value() -- which is a blend, not a
        // reference. refValues() is value() for every BC that does not distinguish them.
        const std::vector<scalar> val = f.boundary[pi]->refValues();   // totalPressure: p0
        // totalPressure's initial device VALUE is the patch value -- the written `value` on a restart from
        // OpenFOAM's output, p0 on a cold start -- while its p0 buffer takes the reference (queue 20).
        const std::vector<scalar> cur = (cat == 7 || cat == 3) ? f.boundary[pi]->value() : std::vector<scalar>{};
        const std::vector<scalar>* vfp = f.boundary[pi]->valueFractionPtr();   // mixed (cat 5): per-face vf seed
        const std::vector<scalar>* rgp = f.boundary[pi]->refGradPtr();         // fixedGradient: per-face g
        const label sgm = f.boundary[pi]->updateableSnGrad() ? 1 : 0;   // fixedFluxPressure
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            // Categories whose VALUE is resolved per-step but whose TYPE is a plain fixedValue: totalPressure
            // and flowRateInletVelocity(mass) map to 1 here -- pushing the category through as a device
            // bcType leaves an unknown type that no evaluator handles. inletOutlet and outletInlet START
            // AS zeroGradient (0): OpenFOAM constructs both with valueFraction 0 (inletOutletFvPatchField.C
            // and outletInletFvPatchField.C, the dictionary constructors) and the flux switch first runs
            // inside the first assembly's updateCoeffs. This arm keeps no stored patch values, so until
            // that first switch every evaluate is what these construction-time coefficients give: seeded
            // fixedValue(inletValue), a RESTARTED case saw inletValue on every inletOutlet face at its
            // first step where OpenFOAM's stored value is the last evaluate. Measured on gasMixing/
            // injectorPipe restarted from OpenFOAM's iteration 5: the closure's assembled epsilon
            // off-diagonals 1.6e-04 off the host, 100% of it on faces of outlet-adjacent cells (the
            // outlet's k/epsilon are inletOutlet, all 461 faces outflow, OpenFOAM's written value equal to
            // the owner cell on every one), while from rest (inletValue == internalField) nothing showed.
            ty.push_back((cat == 3 || cat == 4 || cat == 7 || cat == 9) ? 1 : cat);   // 5 stays mixed
            iost.push_back((cat == 3 && i < (label)cur.size()) ? cur[i] : 0.0);
            iofr.push_back((cat == 3 && storedIoSeed && ioStoredEnabled()) ? 1 : 0);
            asg.push_back(f.boundary[pi]->assignable() ? 1 : 0);
            io.push_back(cat == 3 ? 1 : 0);
            oio.push_back(cat == 4 ? 1 : 0);
            mx.push_back(cat == 5 ? 1 : 0);
            pv.push_back(0);
            sm.push_back(0);
            tp.push_back(cat == 7 ? 1 : 0);
            p0.push_back(cat == 7 ? val[i] : 0.0);   // totalPressure: mask + the reference p0
            vf.push_back((cat == 5 && vfp) ? (*vfp)[i] : 0.0);
            rg.push_back(rgp ? (*rgp)[i] : 0.0);
            sg.push_back(sgm);
            ref.push_back((cat == 7 && i < (label)cur.size()) ? cur[i] : val[i]);
            dc.push_back(fvp[pi].deltaCoeffs[i]);
            ms.push_back(g.magSf()[fvp[pi].start + i]);
            fc.push_back(fvp[pi].faceCells[i]);
        }
    }
    DeviceBoundary db;
    db.n = static_cast<int>(ty.size());
    db.bcType.copyFrom(ty);
    db.assignableMask.copyFrom(asg);
    db.ioMask.copyFrom(io);
    db.oioMask.copyFrom(oio);
    db.ioStored.copyFrom(iost);
    db.ioFresh.copyFrom(iofr);
    db.mixedMask.copyFrom(mx);
    db.piovMask.copyFrom(pv);
    db.symMask.copyFrom(sm);
    db.tpMask.copyFrom(tp);
    db.p0.copyFrom(p0);
    db.valueFraction.copyFrom(vf);
    db.refGrad.copyFrom(rg);
    db.snGradMask.copyFrom(sg);
    for (label v : sg) db.nSnGradFaces += v;
    db.refValue.copyFrom(ref);
    db.deltaCoeffs.copyFrom(dc);
    db.magSf.copyFrom(ms);
    db.faceCell.copyFrom(fc);
    return db;
}

// inletOutlet: recompute the effective per-face bcType from the patch flux sign (OF valueFraction = neg(phi)):
// inflow phiBnd<0 -> fixedValue(1, =inletValue), outflow phiBnd>=0 -> zeroGradient(0). Only ioMask=1 faces change.
// Call at the start of each SIMPLE step (before assembly), with the previous step's boundary flux (matches OF).
void deviceUpdateInletOutlet(DeviceBoundary& db, const DeviceBuffer<scalar>& phiBnd);
void deviceBCValue(const DeviceBoundary& db, const DeviceBuffer<scalar>& internal, DeviceBuffer<scalar>& value,
                   const int* skipIf = nullptr);   // skipIf: device flag, the launch is a no-op when set
void deviceBCLaplacianCoeffs(const DeviceBoundary& db, const DeviceBuffer<scalar>& gammaCell,
                             DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC);
// variant taking gamma per BOUNDARY FACE (for nuEff with the true wall nut, not the cell value).
void deviceBCLaplacianCoeffsFace(const DeviceBoundary& db, const DeviceBuffer<scalar>& gammaFace,
                                 DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC);
void deviceBCDivCoeffs(const DeviceBoundary& db, const DeviceBuffer<scalar>& phiB,
                       DeviceBuffer<scalar>& iC, DeviceBuffer<scalar>& bC);
// pEqn.flux() at boundary faces = internalCoeffs*p[faceCell] - boundaryCoeffs.
void deviceMatrixFluxBoundary(const DeviceBoundary& db, const DeviceBuffer<scalar>& iC, const DeviceBuffer<scalar>& bC,
                              const DeviceBuffer<scalar>& p, DeviceBuffer<scalar>& fluxB);

// A vector field's BC is three scalar boundaries (one per component): the laplacian/div internalCoeffs are
// isotropic (component-independent), only the refValue-dependent boundaryCoeffs / value differ by component.
// So the per-component momentum boundary coeffs reuse the scalar kernels on comp[k].
struct DeviceVectorBoundary
{
    DeviceBoundary comp[3];
    int n = 0;
    DeviceBuffer<scalar> nx, ny, nz;   // nx/ny/nz = unit face normal (pressureInletOutletVelocity)
    // pressureInletOutletVelocity's refValue per boundary face, EMPTY unless a patch carries a
    // `tangentialVelocity`: OpenFOAM makes it once, tangentialVelocity - n*(n & tangentialVelocity) at
    // construction (pressureInletOutletVelocityFvPatchVectorField.C:120-126), and the inflow value is
    // directionMixed's (vf & refValue) + ((I - vf) & U_cell). Zero on every other face.
    DeviceBuffer<scalar> piovRef[3];
};

// inletOutlet on a vector field: same patch flux for all 3 components -> update each component's bcType.
inline void deviceUpdateInletOutlet(DeviceVectorBoundary& db, const DeviceBuffer<scalar>& phiBnd)
{
    for (int k = 0; k < 3; ++k)
        deviceUpdateInletOutlet(db.comp[k], phiBnd);
}

// mixed (Robin) freestreamVelocity/Pressure updateCoeffs: recompute per-face vf from the flow angle. The normal
// flux U.n = phi_b/|Sf| is exact (lagged like OF + the inletOutlet switch); |U| is the local adjacent-cell speed.
// Writes vf to the 3 U components (velocity sign 0.5-0.5c) and to p (pressure sign 0.5+0.5c); non-mixed faces
// untouched. Call at the start of each step (before assembly), after the io switch.
// `which`: 1 = the velocity patches only, 2 = the pressure patches only, 3 = both. OpenFOAM rebuilds
// freestreamVelocity's valueFraction at every evaluate on U (the momentum assembly and the velocity
// correction's correctBoundaryConditions) and freestreamPressure's inside the pressure fvMatrix
// constructor and under the limiter, so the two move at different points of the iteration; the
// incompressible driver keeps rebuilding both together (3).
void deviceUpdateMixedFreestream(DeviceVectorBoundary& dbU, DeviceBoundary& dbP, const DeviceBuffer<scalar>& phiBnd,
                                 const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                                 const DeviceBuffer<scalar>& Uz,
                                 const DeviceBuffer<scalar>* rhoBnd = nullptr,   // compressible: rho at boundary faces
                                 int which = 3,
                                 // OF's `Up` is the patch field's OWN STORED value -- `const Field<vector>& Up = *this`
                                 // (freestreamVelocityFvPatchVectorField.C) -- i.e. what the last evaluate left, NOT a
                                 // re-evaluation against the cells as they stand now. Pass the stored patch velocity
                                 // (the rhoSimpleFoam mirror carries it as f.UxBnd/UyBnd/UzBnd, refreshed exactly where
                                 // the host reference calls U.evaluateBoundary) and this uses it. Null keeps the old
                                 // reconstruction, which is the same number only while the cells have not moved since.
                                 const DeviceBuffer<scalar>* UbX = nullptr,
                                 const DeviceBuffer<scalar>* UbY = nullptr,
                                 const DeviceBuffer<scalar>* UbZ = nullptr);
// constrainHbyA at mixed velocity faces: OF resets phiHbyA_b = U_b.Sf at fixesValue patches (mixed fixesValue=true).
// cf's HbyA boundary value uses HbyA_cell in the (1-vf) part; at mixed faces replace hb[k] with the boundary value
// of U itself (U_b = (1-vf) U_cell + vf U_freestream) so the boundary flux uses U_b, not HbyA_b. Non-mixed faces
// untouched (pure fixedValue already = U_b; zeroGradient is not fixesValue).
void deviceConstrainMixedHbyA(const DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Ux,
                              const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                              DeviceBuffer<scalar>& hbx, DeviceBuffer<scalar>& hby, DeviceBuffer<scalar>& hbz);
// The OTHER half of constrainHbyA: the patches OF deliberately does NOT constrain. OF's rule is
// `!U.boundaryField()[patchi].assignable()`, and inletOutlet overrides assignable() back to TRUE
// (inletOutletFvPatchField.H:164) even though its mixed base returns false -- so HbyA keeps the value it
// was born with, the extrapolatedCalculated boundary of rAU*UEqn.H(), i.e. the ADJACENT CELL value.
// Evaluating HbyA through U's inletOutlet descriptor instead pins phiHbyA_b to the CLAMPED inlet
// velocity, which on a backflow face is zero: a different pressure equation, not a different rounding.
// `ext` is HbyA evaluated with a zero-gradient boundary; io faces take it, everything else is untouched.
void deviceExtrapolateIOHbyA(const DeviceVectorBoundary& dbU,
                             const DeviceBuffer<scalar>& extx, const DeviceBuffer<scalar>& exty,
                             const DeviceBuffer<scalar>& extz,
                             DeviceBuffer<scalar>& hbx, DeviceBuffer<scalar>& hby, DeviceBuffer<scalar>& hbz);
// constrainHbyA at slip/symmetry faces: HbyA_b = HbyA_c - n(n.HbyA_c) so the wall flux is exactly 0 (no penetration).
void deviceConstrainSymmetryHbyA(const DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Hx,
                                 const DeviceBuffer<scalar>& Hy, const DeviceBuffer<scalar>& Hz,
                                 DeviceBuffer<scalar>& hbx, DeviceBuffer<scalar>& hby, DeviceBuffer<scalar>& hbz);

// pressureInletOutletVelocity updateCoeffs (directionMixed, valueFraction = neg(phi)*(I - n n)): per piov face by the
// boundary flux sign, set each component's bcType, valueFraction and refValue. Outflow (phi>=0) -> zeroGradient;
// inflow (phi<0) -> the mixed (cat 5) form of OpenFOAM's transform coefficients, vf_k = sqrt(1 - n_k^2) with the
// refValue chosen so the blend is n*(n.U_cell) -- see piovUpdateKernel. Call wherever OpenFOAM reaches an
// updateCoeffs or an evaluate on U (the momentum assembly, after its solve, after the velocity correction), with
// the flux registered at that moment and the cell velocity as it stands.
// `directionMixed` selects that form; false keeps the kernel's earlier typing (every inflow component
// fixedValue at n*(n.U_cell)) for the frozen incompressible driver, which reads 1.1459e-04 against
// OpenFOAM on validation/piov with it and 1.4911e-03 with the directionMixed form (bisected 2026-09-03).
void deviceUpdatePressureInletOutletVelocity(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& phiBnd,
                                             const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                                             const DeviceBuffer<scalar>& Uz, bool directionMixed = false);

// slip/symmetry updateCoeffs (OF basicSymmetry, GENERAL non-axis-aligned): reuses the mixed (cat 5) kernels with a
// PER-COMPONENT valueFraction vf_k = |n_k| and ref_k = U_c[k] - sign(n_k)*(n.U_c). Then valueIC_k = 1-|n_k|,
// gradIC_k = -dc*|n_k|, value = U_c - n(n.U_c), identical to OF's snGradTransformDiag=(|nx|,|ny|,|nz|) coeffs. Call
// each step before assembly with the previous step's cell velocity (the cross-component n.U is lagged, as in a
// segregated solve). Reduces bit-exactly to fixedValue-0(normal)+zeroGradient(tangential) for an axis-aligned plane.
// wedge updateCoeffs (OF wedgeFvPatchField<vector>): per wedge face, ref_k is chosen so that the mixed
// blend d_k*ref_k + (1 - d_k)*U_c[k] reproduces OF's value transform(faceT, U_cell)[k], with the
// valueFraction d_k = 0.5*(1 - cellT_kk) already stored. That makes the IMPLICIT coefficients
// (1 - d_k and -deltaCoeffs*d_k) identical to OF's valueInternalCoeffs/gradientInternalCoeffs, and puts
// the cross-component rotation where it belongs: in the explicit value.
void deviceUpdateWedge(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                       const DeviceBuffer<scalar>& Uz);
// b = faceT & f_cell on the wedge faces (other faces untouched) -- for evaluating a field that is NOT the
// one the wedge refValue was built from, i.e. HbyA. See wedgeFaceValueKernel.
void deviceWedgeFaceValue(const DeviceVectorBoundary& dbU,
                          const DeviceBuffer<scalar>& fx, const DeviceBuffer<scalar>& fy,
                          const DeviceBuffer<scalar>& fz,
                          DeviceBuffer<scalar>& bx, DeviceBuffer<scalar>& by, DeviceBuffer<scalar>& bz);
// A laplacian's boundaryCoeffs on the WEDGE faces, component k, overwritten with OpenFOAM's own
// -gamma*magSf*gradientBoundaryCoeffs -- which the mixed slot's single refValue cannot carry, because it
// is spent reproducing the wedge's VALUE (faceT) and OpenFOAM's gradient coefficient takes cellT and
// half the deltaCoeffs. Call it right after deviceBCLaplacianCoeffsFace on the same bC, with the field
// the coefficients are for. A mesh with no wedge returns at once.
void deviceWedgeLaplacianBC(const DeviceVectorBoundary& dbU, int k, const DeviceBuffer<scalar>& gammaFace,
                            const DeviceBuffer<scalar>& fx, const DeviceBuffer<scalar>& fy,
                            const DeviceBuffer<scalar>& fz, DeviceBuffer<scalar>& bC);
void deviceUpdateSymmetry(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                          const DeviceBuffer<scalar>& Uz);
// OF correctUphiBCs: overwrite phiB on the faces where `adjustable` is 0 (i.e. U fixesValue) with phiFixed.
void deviceSelectFixedFlux(const DeviceBuffer<label>& adjustable, const DeviceBuffer<scalar>& phiFixed,
                           DeviceBuffer<scalar>& phiB);

// totalPressure updateCoeffs (OF totalPressureFvPatchScalarField, incompressible rho=none/psi=none): per face
// refValue = p0 - 0.5*neg(phi_b)*magSqr(U_b)  (inflow phi<0 -> p0 - dynamic head; outflow -> p0). bcType stays
// fixedValue(1). Call each step AFTER the U-boundary updates, with the boundary U (deviceBCValue of current U) + flux.
// flowRateInletVelocity (massFlowRate): set U_b = avgU*n on the masked patch faces. See the .cu.
// turbulentIntensityKineticEnergyInlet / turbulentMixingLength{DissipationRate,Frequency}Inlet.
// OF re-evaluates both every updateCoeffs; these refresh the boundary refValue in place from the CURRENT
// boundary U (for k) and the CURRENT boundary k (for epsilon|omega). mask: k side 1 = active;
// second side 1 = epsilon, 2 = omega.
void deviceUpdateTurbulentInletK(const DeviceVectorBoundary& dbU, const DeviceBuffer<label>& mask,
                                 const DeviceBuffer<scalar>& intensity, DeviceBoundary& dbK);
void deviceUpdateTurbulentInletSecond(const DeviceBoundary& dbK, const DeviceBuffer<label>& mask,
                                      const DeviceBuffer<scalar>& len, scalar Cmu, DeviceBoundary& dbSecond);

// The last three are the caller's STORED boundary-value arrays, written on the masked faces alongside
// the refValue. OpenFOAM's flowRateInletVelocity assigns its patch VALUE inside updateCoeffs
// (flowRateInletVelocityFvPatchVectorField.C:194-196, operator==(avgU*n)), so a caller that keeps its
// own U boundary arrays and assembles against them must be given the new value here -- refreshing only
// the refValue left rhoTI's momentum assembly differentiating against 0/U's seed (50) where OpenFOAM
// had 50.687834607787899, worth U 8.9e-06 at iteration 1. Null where the caller re-derives from dbU.
void deviceUpdateFlowRateInlet(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& maskMagSf, scalar avgU,
                               const DeviceBuffer<scalar>& nx, const DeviceBuffer<scalar>& ny,
                               const DeviceBuffer<scalar>& nz,
                               DeviceBuffer<scalar>* UxBnd = nullptr,
                               DeviceBuffer<scalar>* UyBnd = nullptr,
                               DeviceBuffer<scalar>* UzBnd = nullptr);
// FP-9: the same update with the reduction kept on the DEVICE -- gSum(rho*magSf) into a device scalar,
// avgU formed there, and the patch kernel reading it. The host-scalar form above costs a blocking copy
// per inlet patch per iteration (measured: one 463 us gap on squareBend, the GPU idle across it).
void deviceUpdateFlowRateInletDev(DeviceVectorBoundary& dbU, const DeviceBuffer<scalar>& maskMagSf,
                                  scalar mdot, bool isMass, const DeviceBuffer<scalar>& rhoBnd,
                                  const DeviceBuffer<scalar>& nx, const DeviceBuffer<scalar>& ny,
                                  const DeviceBuffer<scalar>& nz,
                                  DeviceBuffer<scalar>* UxBnd = nullptr,
                                  DeviceBuffer<scalar>* UyBnd = nullptr,
                                  DeviceBuffer<scalar>* UzBnd = nullptr);

// patchInternalField for every boundary face (out[i] = cellField[faceCell[i]]), for BCs whose value is a
// patch-wide functional of the adjacent cells -- fixedMean is one.
void deviceGatherPatchInternal(const DeviceBoundary& db, const DeviceBuffer<scalar>& cellField,
                               DeviceBuffer<scalar>& out);

void deviceUpdateTotalPressure(DeviceBoundary& db, const DeviceBuffer<scalar>& phiB, const DeviceBuffer<scalar>& Uxb,
                               const DeviceBuffer<scalar>& Uyb, const DeviceBuffer<scalar>& Uzb,
                               const DeviceBuffer<scalar>* rhoBnd = nullptr);   // compressible: rho at the face
                                                                                // (p in Pa -> 0.5*rho*|U|^2)

// THE HOST HALF of buildDeviceVectorBoundary: every per-face array the device boundary holds, from the field's
// patches as they stand. Split from the upload so a caller whose mesh does not move can refresh only the arrays
// that move (refreshDeviceVectorBoundaryState) from the same arithmetic.
struct DeviceVectorBoundaryHost
{
    std::vector<label> ty[3], fc, io, oio, mx, pv, sm, wdg, iofr, gsm;
    std::vector<scalar> dc, ms, ref[3], vf[3], nrm[3], rg[3], wdgT, iost[3], gsn;   // rg = fixedGradient, per component
    // pressureInletOutletVelocity's refValue per face (DeviceVectorBoundary::piovRef)
    std::vector<scalar> prv[3];
    bool anyPiovRef = false;
};

inline DeviceVectorBoundaryHost deviceVectorBoundaryArrays(
    const GeometricField<vector>& f,
    const std::vector<FvPatch>& fvp,
    const FvGeometry& g,
    bool storedIoSeed,
    // the per-call STATE alone (ty, vf, ref, rg, iost, iofr): every other array is left empty -- the caller has
    // established they are the ones a full build gave (deviceVectorBoundaryShape)
    bool stateOnly = false,
    // ...and, of that state, NOTHING FOR AN `empty` PATCH: the arrays come out compact, the other patches' faces
    // in patch order (deviceNonEmptyBoundaryRanges says where each run belongs), for a caller that keeps the
    // empty faces' entries on the device itself. An empty patch's state is constant but for its refValue, which
    // is its cells' values (EmptyPatchField::evaluate).
    bool skipEmpty = false)
{
    if (skipEmpty && !stateOnly)
    {
        throw std::runtime_error("brae deviceVectorBoundaryArrays: skipEmpty is for a state-only build.");
    }
    DeviceVectorBoundaryHost h;
    std::vector<label> (&ty)[3] = h.ty;
    std::vector<label>& fc = h.fc;
    std::vector<label>& io = h.io;
    std::vector<label>& oio = h.oio;
    std::vector<label>& mx = h.mx;
    std::vector<label>& pv = h.pv;
    std::vector<label>& sm = h.sm;
    std::vector<label>& wdg = h.wdg;
    std::vector<label>& iofr = h.iofr;
    std::vector<label>& gsm = h.gsm;
    std::vector<scalar>& dc = h.dc;
    std::vector<scalar>& ms = h.ms;
    std::vector<scalar> (&ref)[3] = h.ref;
    std::vector<scalar> (&vf)[3] = h.vf;
    std::vector<scalar> (&nrm)[3] = h.nrm;
    std::vector<scalar> (&rg)[3] = h.rg;
    std::vector<scalar>& wdgT = h.wdgT;
    std::vector<scalar> (&iost)[3] = h.iost;
    std::vector<scalar>& gsn = h.gsn;
    std::vector<scalar> (&prv)[3] = h.prv;
    bool& anyPiovRef = h.anyPiovRef;
    // every array at its final size up front: twenty vectors grown a push_back at a time were most of the
    // refresh on a 2-D case, whose two `empty` patches hold as many faces as the mesh has cells
    std::size_t nAll = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (skipEmpty && fvp[pi].type == "empty") continue;
        if (!isCoupledInterfaceType(fvp[pi].type)) nAll += static_cast<std::size_t>(fvp[pi].size);
    }
    iofr.reserve(nAll);
    for (int k = 0; k < 3; ++k)
    {
        ty[k].reserve(nAll);
        vf[k].reserve(nAll);
        ref[k].reserve(nAll);
        rg[k].reserve(nAll);
        iost[k].reserve(nAll);
    }
    if (!stateOnly)
    {
        fc.reserve(nAll);
        io.reserve(nAll);
        oio.reserve(nAll);
        mx.reserve(nAll);
        pv.reserve(nAll);
        sm.reserve(nAll);
        wdg.reserve(nAll);
        gsm.reserve(nAll);
        dc.reserve(nAll);
        ms.reserve(nAll);
        gsn.reserve(3*nAll);
        wdgT.reserve(9*nAll);
        for (int k = 0; k < 3; ++k)
        {
            nrm[k].reserve(nAll);
            prv[k].reserve(nAll);
        }
    }
    // BRAE_CONTROL_BOUNDARY_ARRAYS_PER_FACE=1 takes the per-face loop on every patch, as before -- the identity
    // check's other arm
    static const bool perFaceAlways = std::getenv("BRAE_CONTROL_BOUNDARY_ARRAYS_PER_FACE") != nullptr;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;                     // cyclic = internal-like (handled by appended faces)
        if (skipEmpty && fvp[pi].type == "empty") continue;
        // the gradient's mirror normal, in the host reference's own arithmetic (fvc.cu, boundaryGradU):
        // Sf/magSf by DIVISION per face on a `symmetry`; sumA/mag(sumA) once per patch on a
        // `symmetryPlane`, summed in face order from zero and (0,0,0) below 1e-150
        const bool gradSym = (fvp[pi].type == "symmetry");
        const bool gradSymPlane = (fvp[pi].type == "symmetryPlane");
        vector planeN{0, 0, 0};
        if (gradSymPlane)
        {
            vector sumA{0, 0, 0};
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                sumA = sumA + g.Sf()[fvp[pi].start + i];
            }
            const scalar a = mag(sumA);
            planeN = (a > scalar(1.0e-150)) ? sumA/a : vector{0, 0, 0};
        }
        // A PLAIN PATCH -- no wedge, no symmetry, no inletOutlet stored value, no per-face valueFraction, gradient
        // or tangential reference -- has per-patch constants in every array but the geometry and refValue, and
        // those are written in bulk below; every other patch takes the per-face loops. The arrays are the same.
        const int catEarly = (fvp[pi].type == "processor") ? 8 : f.boundary[pi]->bcCategory();
        const bool plain = !perFaceAlways && !gradSym && !gradSymPlane && catEarly != 3
                        && !f.boundary[pi]->isSymmetry()
                        && !(f.boundary[pi]->wedgeFaceT() && f.boundary[pi]->wedgeCellT())
                        && f.boundary[pi]->tangentialRefPtr() == nullptr
                        && f.boundary[pi]->refGradPtr() == nullptr
                        && !(catEarly == 5 && f.boundary[pi]->valueFractionPtr());
        if (plain)
        {
            const std::size_t np = static_cast<std::size_t>(fvp[pi].size);
            const std::size_t start = static_cast<std::size_t>(fvp[pi].start);
            const std::vector<vector> val = f.boundary[pi]->refValues();
            if (!stateOnly)
            {
                gsm.insert(gsm.end(), np, 0);
                gsn.insert(gsn.end(), 3*np, planeN.x);
                fc.insert(fc.end(), fvp[pi].faceCells.begin(), fvp[pi].faceCells.begin() + fvp[pi].size);
                dc.insert(dc.end(), fvp[pi].deltaCoeffs.begin(), fvp[pi].deltaCoeffs.begin() + fvp[pi].size);
                ms.insert(ms.end(), g.magSf().begin() + fvp[pi].start,
                          g.magSf().begin() + fvp[pi].start + fvp[pi].size);
                io.insert(io.end(), np, 0);
                oio.insert(oio.end(), np, catEarly == 4 ? 1 : 0);
                mx.insert(mx.end(), np, catEarly == 5 ? 1 : 0);
                pv.insert(pv.end(), np, catEarly == 6 ? 1 : 0);
                sm.insert(sm.end(), np, 0);
                wdg.insert(wdg.end(), np, 0);
                const std::size_t t0 = wdgT.size();
                wdgT.resize(t0 + 9*np, scalar(0));
                for (std::size_t i = 0; i < np; ++i)
                {
                    wdgT[t0 + 9*i] = scalar(1);
                    wdgT[t0 + 9*i + 4] = scalar(1);
                    wdgT[t0 + 9*i + 8] = scalar(1);
                }
                for (int k = 0; k < 3; ++k)
                {
                    prv[k].insert(prv[k].end(), np, scalar(0));
                    nrm[k].resize(nrm[k].size() + np);
                }
                const std::size_t n0 = nrm[0].size() - np;
                for (std::size_t i = 0; i < np; ++i)
                {
                    const vector Sf = g.Sf()[start + i];
                    const scalar mg = g.magSf()[start + i];
                    nrm[0][n0 + i] = mg > 0 ? Sf.x / mg : 0.0;
                    nrm[1][n0 + i] = mg > 0 ? Sf.y / mg : 0.0;
                    nrm[2][n0 + i] = mg > 0 ? Sf.z / mg : 0.0;
                }
            }
            iofr.insert(iofr.end(), np, 0);
            const label tyPlain = (catEarly == 4 || catEarly == 9) ? 1 : (catEarly == 6 ? 0 : catEarly);
            for (int k = 0; k < 3; ++k)
            {
                rg[k].insert(rg[k].end(), np, scalar(0));
                ty[k].insert(ty[k].end(), np, tyPlain);
                vf[k].insert(vf[k].end(), np, scalar(0));
                iost[k].insert(iost[k].end(), np, scalar(0));
                ref[k].resize(ref[k].size() + np);
            }
            const std::size_t r0 = ref[0].size() - np;
            for (std::size_t i = 0; i < np; ++i)
            {
                ref[0][r0 + i] = val[i].x;
                ref[1][r0 + i] = val[i].y;
                ref[2][r0 + i] = val[i].z;
            }
            continue;
        }
        for (label i = 0; i < fvp[pi].size && !stateOnly; ++i)
        {
            const label gf = fvp[pi].start + i;
            const vector n = gradSym ? g.Sf()[gf]/g.magSf()[gf] : planeN;
            gsm.push_back((gradSym || gradSymPlane) ? 1 : 0);
            gsn.push_back(n.x);
            gsn.push_back(n.y);
            gsn.push_back(n.z);
        }
        // A processor patch is COUPLED: it stays in the boundary gather (the explicit operators need its bval
        // slot, filled with the halo-interpolated face value by DeviceHalo::scatterBoundaryValues), but must
        // contribute NO matrix coefficients -- its coupling is the interface off-diagonal. bcCategory() is NOT
        // overridden on ProcessorFvPatchField, so it would otherwise report 0 (= zeroGradient) whose
        // valueInternalCoeffs is 1, DOUBLE-COUNTING the interface diagonal.
        const int cat = (fvp[pi].type == "processor") ? 8 : f.boundary[pi]->bcCategory();
        // pressureInletOutletVelocity with a tangentialVelocity: the refValue the host patch made from it
        // at construction goes up beside the normals, and the directionMixed kernel adds it to the inflow
        // value (deviceUpdatePressureInletOutletVelocity). The factory refuses the entry for every solver
        // that has not claimed it, so only interFoam reaches here with one.
        const std::vector<vector>* piovRefHost = f.boundary[pi]->tangentialRefPtr();
        if (piovRefHost && piovRefHost->size() != static_cast<std::size_t>(fvp[pi].size))
        {
            throw std::runtime_error(
                "brae: patch " + fvp[pi].name + " carries a `tangentialVelocity` and its refValue is not one "
                "vector per face.");
        }
        anyPiovRef = anyPiovRef || piovRefHost != nullptr;
        for (label i = 0; i < fvp[pi].size && !stateOnly; ++i)
        {
            const vector rv = piovRefHost ? (*piovRefHost)[static_cast<std::size_t>(i)] : vector{0, 0, 0};
            prv[0].push_back(rv.x);
            prv[1].push_back(rv.y);
            prv[2].push_back(rv.z);
        }
        const bool sym = f.boundary[pi]->isSymmetry();
        // wedge: the patch field carries the two rotation tensors; the valueFraction comes from the
        // FULL-angle one and the per-step refValue from the HALF-angle one.
        const tensor* wfT = f.boundary[pi]->wedgeFaceT();
        const tensor* wcT = f.boundary[pi]->wedgeCellT();
        const bool wedge = (wfT && wcT);
        const scalar wdgVf[3] = { wedge ? scalar(0.5)*(scalar(1) - wcT->xx) : scalar(0),
                                  wedge ? scalar(0.5)*(scalar(1) - wcT->yy) : scalar(0),
                                  wedge ? scalar(0.5)*(scalar(1) - wcT->zz) : scalar(0) };
        // The REFERENCE value, not the current one -- see the scalar builder above.
        const std::vector<vector> val = f.boundary[pi]->refValues();
        // ...and the STORED value of an inletOutlet/outletInlet face -- see DeviceBoundary::ioStored.
        const std::vector<vector> curv = (cat == 3) ? f.boundary[pi]->value() : std::vector<vector>{};
        const std::vector<scalar>* vfp = f.boundary[pi]->valueFractionPtr();   // mixed (cat 5): per-face vf seed
        // fixedGradient on a VECTOR field: the gradient is a vector, so it splits per component -- each
        // DeviceBoundary in comp[] carries its own refGrad, exactly as each carries its own refValue.
        const std::vector<vector>* rgv = f.boundary[pi]->refGradPtr();
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            if (!stateOnly)
            {
                fc.push_back(fvp[pi].faceCells[i]);
                dc.push_back(fvp[pi].deltaCoeffs[i]);
                ms.push_back(g.magSf()[fvp[pi].start + i]);
                io.push_back(cat == 3 ? 1 : 0);
                oio.push_back(cat == 4 ? 1 : 0);   // inletOutlet / outletInlet (same flux for all 3 comps)
            }
            iofr.push_back((cat == 3 && storedIoSeed && ioStoredEnabled()) ? 1 : 0);
            // A WEDGE is NOT in the mixed mask, even though it reports category 5. It borrows the mixed
            // (Robin) slot for its COEFFICIENTS only -- its valueFraction d_k = 0.5*(1 - cellT_kk) is pure
            // geometry, fixed for the run. deviceUpdateMixedFreestream rewrites the valueFraction of every
            // masked face from the flux sign, so leaving a wedge in here overwrote (0, 1.9e-3, 1.9e-3)
            // with a flat 1 -- turning the axisymmetric constraint into a fixedValue wall on both wedge
            // planes. On movingCone that put nuEff*magSf*deltaCoeffs (which scales as 1/r^2 at the axis)
            // into the momentum diagonal: rAU fell 17x in the first radial row, so U = HbyA - rAU*grad(p)
            // came out 7x short of OpenFOAM's on a pressure field that agreed to 2%.
            if (!stateOnly)
            {
                mx.push_back((cat == 5 && !wedge) ? 1 : 0);
                pv.push_back(cat == 6 ? 1 : 0);
                sm.push_back(sym ? 1 : 0);   // mixed / piov / symmetry masks
            }
            const scalar seedVf = (cat == 5 && vfp) ? (*vfp)[i] : 0.0;
            {
                // fvPatch::nf() = Sf()/magSf() (fvPatch.C:150-153), a DIVISION, as the host's patch.nf is
                // (fv_patch.cu). It was Sf*(1/magSf): another last bit, and on a nearly axis-aligned face the
                // pressureInletOutletVelocity transform diagonal sqrt(1 - n_k*n_k) cancels it up by ~1e4 --
                // MEASURED on the piston's top patch as rAU 7.7e-11 from the host's on 40 cells at step one.
                const vector Sf = g.Sf()[fvp[pi].start + i];
                const scalar mg = g.magSf()[fvp[pi].start + i];
                nrm[0].push_back(mg > 0 ? Sf.x / mg : 0.0);
                nrm[1].push_back(mg > 0 ? Sf.y / mg : 0.0);
                nrm[2].push_back(mg > 0 ? Sf.z / mg : 0.0);
            }
            {
                const vector gv = rgv ? (*rgv)[static_cast<std::size_t>(i)] : vector{0, 0, 0};
                rg[0].push_back(gv.x); rg[1].push_back(gv.y); rg[2].push_back(gv.z);
            }
            const scalar rv[3] = { val[i].x, val[i].y, val[i].z };
            if (!stateOnly)
            {
                wdg.push_back(wedge ? 1 : 0);
                const tensor T = wedge ? *wfT : tensor{1,0,0,0,1,0,0,0,1};
                const scalar t9[9] = {T.xx, T.xy, T.xz, T.yx, T.yy, T.yz, T.zx, T.zy, T.zz};
                for (int q = 0; q < 9; ++q) wdgT.push_back(t9[q]);
            }
            if (wedge)   // mixed kernels; vf_k = 0.5*(1 - cellT_kk) (geometry, fixed), ref per step
            {
                for (int k = 0; k < 3; ++k)
                {
                    ty[k].push_back(5);
                    vf[k].push_back(wdgVf[k]);
                    ref[k].push_back(rv[k]);
                    iost[k].push_back(0.0);   // never an io face; every per-face array is indexed alike
                }
            }
            else if (sym)   // mixed kernels; vf_k=|n_k|, ref recomputed per step (init = host value v - n(n.v))
            {
                for (int k = 0; k < 3; ++k)
                {
                    ty[k].push_back(5);
                    vf[k].push_back(std::fabs(nrm[k].back()));
                    ref[k].push_back(rv[k]);
                    // never an io face, but ioStored must have one entry per face like every other array:
                    // a shorter one was read past its end on a case with a symmetry patch (an illegal
                    // address in test_mean_velocity_force).
                    iost[k].push_back(0.0);
                }
            }
            else
            {
                for (int k = 0; k < 3; ++k)
                {
                    // io/oio/flowRateInlet init fixedValue (the device rewrites refValue per step); piov
                    // zeroGradient init. Anything whose VALUE is resolved per-step but whose TYPE is a
                    // plain fixedValue MUST map to 1 -- pushing the category through leaves an unknown
                    // device bcType that no evaluator handles, and the patch then behaves arbitrarily.
                    ty[k].push_back((cat == 3 || cat == 4 || cat == 9) ? 1 : (cat == 6 ? 0 : cat));
                    vf[k].push_back(seedVf);
                    ref[k].push_back(rv[k]);
                    iost[k].push_back((cat == 3 && i < (label)curv.size())
                                      ? (k == 0 ? curv[i].x : k == 1 ? curv[i].y : curv[i].z) : 0.0);
                }
            }
        }
    }
    return h;
}

// the upload half of buildDeviceVectorBoundary, from arrays already built
inline DeviceVectorBoundary uploadDeviceVectorBoundary(const DeviceVectorBoundaryHost& h)
{
    const std::vector<label> (&ty)[3] = h.ty;
    const std::vector<label>& fc = h.fc;
    const std::vector<label>& io = h.io;
    const std::vector<label>& oio = h.oio;
    const std::vector<label>& mx = h.mx;
    const std::vector<label>& pv = h.pv;
    const std::vector<label>& sm = h.sm;
    const std::vector<label>& wdg = h.wdg;
    const std::vector<label>& iofr = h.iofr;
    const std::vector<label>& gsm = h.gsm;
    const std::vector<scalar>& dc = h.dc;
    const std::vector<scalar>& ms = h.ms;
    const std::vector<scalar> (&ref)[3] = h.ref;
    const std::vector<scalar> (&vf)[3] = h.vf;
    const std::vector<scalar> (&nrm)[3] = h.nrm;
    const std::vector<scalar> (&rg)[3] = h.rg;
    const std::vector<scalar>& wdgT = h.wdgT;
    const std::vector<scalar> (&iost)[3] = h.iost;
    const std::vector<scalar>& gsn = h.gsn;
    const std::vector<scalar> (&prv)[3] = h.prv;
    const bool anyPiovRef = h.anyPiovRef;
    DeviceVectorBoundary db;
    db.n = static_cast<int>(fc.size());
    // BRAE_CONTROL_DEVICE_PIOV_NOREF=1 leaves the refValue off the device -- the inflow tangential part is
    // then zero, which is what this builder refused to run before it carried the entry. A gate's control.
    if (anyPiovRef && std::getenv("BRAE_CONTROL_DEVICE_PIOV_NOREF") == nullptr)
    {
        for (int k = 0; k < 3; ++k)
        {
            db.piovRef[k].copyFrom(prv[k]);
        }
    }
    db.nx.copyFrom(nrm[0]);
    db.ny.copyFrom(nrm[1]);
    db.nz.copyFrom(nrm[2]);
    for (int k = 0; k < 3; ++k)
    {
        db.comp[k].n = db.n;
        db.comp[k].bcType.copyFrom(ty[k]);
        db.comp[k].ioMask.copyFrom(io);
        db.comp[k].oioMask.copyFrom(oio);
        db.comp[k].ioStored.copyFrom(iost[k]);
        db.comp[k].ioFresh.copyFrom(iofr);
        db.comp[k].mixedMask.copyFrom(mx);
        db.comp[k].piovMask.copyFrom(pv);
        db.comp[k].symMask.copyFrom(sm);
        db.comp[k].wedgeMask.copyFrom(wdg);
        db.comp[k].wedgeT.copyFrom(wdgT);
        db.comp[k].gradSymMask.copyFrom(gsm);
        db.comp[k].gradSymN.copyFrom(gsn);
        db.comp[k].valueFraction.copyFrom(vf[k]);
        db.comp[k].refValue.copyFrom(ref[k]);
        db.comp[k].refGrad.copyFrom(rg[k]);
        db.comp[k].deltaCoeffs.copyFrom(dc);
        db.comp[k].magSf.copyFrom(ms);
        db.comp[k].faceCell.copyFrom(fc);
    }
    return db;
}

inline DeviceVectorBoundary buildDeviceVectorBoundary(
    const GeometricField<vector>& f,
    const std::vector<FvPatch>& fvp,
    const FvGeometry& g,
    bool storedIoSeed = false)   // as the scalar builder's
{
    return uploadDeviceVectorBoundary(deviceVectorBoundaryArrays(f, fvp, g, storedIoSeed));
}

// WHAT THE NON-STATE ARRAYS ARE BUILT FROM: the boundary's geometry (face cells, the patches' deltaCoeffs, Sf
// and magSf on their faces) and each uncoupled patch's kind -- its type, category, symmetry, wedge tensors and
// tangentialVelocity refValue. Two equal keys give the same non-state arrays, so a refresh can skip building
// them; a cyclicACMI whose open fraction moves changes its faces' areas and so its key.
struct DeviceVectorBoundaryShape
{
    std::vector<label> fc;
    std::vector<scalar> geom;
    std::vector<std::string> kind;
    std::vector<scalar> kindData;

    bool operator==(const DeviceVectorBoundaryShape& o) const
    {
        return fc == o.fc && geom == o.geom && kind == o.kind && kindData == o.kindData;
    }
};

inline DeviceVectorBoundaryShape deviceVectorBoundaryShape(
    const GeometricField<vector>& f,
    const std::vector<FvPatch>& fvp,
    const FvGeometry& g,
    // an `empty` patch's face cells and geometry left out of the key, for a caller that keys them otherwise --
    // on the mesh's addressing and on a count of its geometry's changes. They change only when the mesh does,
    // and on a 2-D mesh they are two faces a cell: MEASURED on waveMakerPiston refined to 896,000 cells, the key
    // alone 29 ms a step, four calls.
    bool skipEmpty = false)
{
    DeviceVectorBoundaryShape k;
    std::size_t nAll = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (skipEmpty && fvp[pi].type == "empty") continue;
        if (!isCoupledInterfaceType(fvp[pi].type)) nAll += static_cast<std::size_t>(fvp[pi].size);
    }
    k.fc.reserve(nAll);
    k.geom.resize(5*nAll);
    std::size_t at = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;
        const int cat = (fvp[pi].type == "processor") ? 8 : f.boundary[pi]->bcCategory();
        k.kind.push_back(fvp[pi].type + "|" + std::to_string(cat) + "|"
                         + (f.boundary[pi]->isSymmetry() ? "s" : "-"));
        const tensor* wfT = f.boundary[pi]->wedgeFaceT();
        const tensor* wcT = f.boundary[pi]->wedgeCellT();
        for (const tensor* T : {wfT, wcT})
        {
            if (T)
            {
                k.kindData.insert(k.kindData.end(), {T->xx, T->xy, T->xz, T->yx, T->yy, T->yz, T->zx, T->zy, T->zz});
            }
        }
        if (const std::vector<vector>* tr = f.boundary[pi]->tangentialRefPtr())
        {
            for (const vector& v : *tr)
            {
                k.kindData.insert(k.kindData.end(), {v.x, v.y, v.z});
            }
        }
        if (skipEmpty && fvp[pi].type == "empty") continue;
        k.fc.insert(k.fc.end(), fvp[pi].faceCells.begin(), fvp[pi].faceCells.begin() + fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const std::size_t gf = static_cast<std::size_t>(fvp[pi].start + i);
            k.geom[at] = fvp[pi].deltaCoeffs[i];
            k.geom[at + 1] = g.magSf()[gf];
            k.geom[at + 2] = g.Sf()[gf].x;
            k.geom[at + 3] = g.Sf()[gf].y;
            k.geom[at + 4] = g.Sf()[gf].z;
            at += 5;
        }
    }
    return k;
}

// ...and on a mesh that does NOT MOVE, the arrays that can change between two builds alone: the per-face type,
// valueFraction, refValue, refGrad and the inletOutlet stored value and seed -- the patches' state, and what the
// device's update kernels rewrite in place. The geometry, the masks and the wedge tensors stay as `db` holds
// them from its full build. MEASURED on RAS/DTCHull (69,887 boundary faces): a full build 16 ms, three a step,
// two thirds of it the uploads. The caller owns the precondition: deviceVectorBoundaryShape equal to the one `db`
// was built with. MEASURED on RAS/damBreakLeakage, where a cyclicACMI's open fraction moves on an unmoved mesh:
// U 3.0e-01 from OpenFOAM on the device arm with the first build's geometry kept, 5.1e-12 rebuilt.
// where the faces of the patches that are neither `empty` nor a coupled interface sit in the device boundary's
// numbering (the non-coupled patches in order), run by run; `nAll` is that numbering's size
struct DeviceBoundaryRange
{
    std::size_t at = 0;
    std::size_t n = 0;
};
inline std::vector<DeviceBoundaryRange> deviceNonEmptyBoundaryRanges(
    const std::vector<FvPatch>& fvp,
    std::size_t& nAll)
{
    std::vector<DeviceBoundaryRange> ranges;
    nAll = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;
        const std::size_t np = static_cast<std::size_t>(fvp[pi].size);
        if (fvp[pi].type != "empty" && np > 0)
        {
            ranges.push_back(DeviceBoundaryRange{nAll, np});
        }
        nAll += np;
    }
    return ranges;
}

// ARRAYS OF ONE STAGED BLOCK TO THEIR PLACES IN THE BOUNDARY'S NUMBERING, RUN BY RUN, IN ONE LAUNCH
// (device_boundary_flow.cu). `stage` holds whole arrays of `nh` compact entries one after another; array from[k]
// of it goes to *to[k], entry i to the place the runs give it. The faces between the runs are left as they are.
// It stands for a device-to-device copy an array a run, which is what a refresh was: MEASURED 2026-10-06 on
// stokesII (four patches that are not empty, so 72 copies for U's eighteen state arrays at each of three calls
// a step), the refresh 0.5 -> 0.1 ms a step and the step 7.73 -> 7.30; on damBreak's 2,268 cells the step
// 6.0 -> 5.3 with the assembly's mirror. At most twelve arrays a call.
// BRAE_CONTROL_SCATTER_RUNS_BY_COPIES=1 makes those copies instead -- the identity gate's other arm;
// BRAE_CONTROL_SCATTER_RUNS_REVERSED=1 is its control, deliberately wrong: the entries land in reverse order.
void deviceScatterRuns(
    const DeviceBuffer<scalar>& stage,
    std::size_t nh,
    const std::vector<DeviceBoundaryRange>& runs,
    const std::vector<DeviceBuffer<scalar>*>& to,
    const std::vector<int>& from);
void deviceScatterRuns(
    const DeviceBuffer<label>& stage,
    std::size_t nh,
    const std::vector<DeviceBoundaryRange>& runs,
    const std::vector<DeviceBuffer<label>*>& to,
    const std::vector<int>& from);

// `ranges` null: h holds every boundary face's state and all of it goes up. `ranges` set: h is COMPACT -- built
// with skipEmpty -- and each run goes to its place, the faces between them (the empty patches') left as they are.
inline void refreshDeviceVectorBoundaryState(
    DeviceVectorBoundary& db,
    const DeviceVectorBoundaryHost& h,
    const std::vector<DeviceBoundaryRange>* ranges = nullptr)
{
    std::size_t nCompact = 0;
    if (ranges)
    {
        for (const DeviceBoundaryRange& r : *ranges)
        {
            if (r.at + r.n > static_cast<std::size_t>(db.n))
            {
                throw std::runtime_error("brae: refreshDeviceVectorBoundaryState: a run past the boundary's end.");
            }
            nCompact += r.n;
        }
    }
    if (static_cast<std::size_t>(h.ty[0].size()) != (ranges ? nCompact : static_cast<std::size_t>(db.n)))
    {
        throw std::runtime_error(
            "brae: refreshDeviceVectorBoundaryState on a boundary of another size -- the device boundary was "
            "built for a different mesh, which needs buildDeviceVectorBoundary.");
    }
    // ONE UPLOAD per type, then the device splits it: eighteen blocking uploads of one boundary's worth each
    // were 23.7 ms a step on RAS/DTCHull (69,887 boundary faces, three calls a step)
    const std::size_t n = static_cast<std::size_t>(db.n);
    const std::size_t nh = ranges ? nCompact : n;
    std::vector<scalar> packS;
    packS.reserve(12*nh);
    std::vector<label> packL;
    packL.reserve(4*nh);
    for (int k = 0; k < 3; ++k)
    {
        packS.insert(packS.end(), h.iost[k].begin(), h.iost[k].end());
        packS.insert(packS.end(), h.vf[k].begin(), h.vf[k].end());
        packS.insert(packS.end(), h.ref[k].begin(), h.ref[k].end());
        packS.insert(packS.end(), h.rg[k].begin(), h.rg[k].end());
        packL.insert(packL.end(), h.ty[k].begin(), h.ty[k].end());
    }
    packL.insert(packL.end(), h.iofr.begin(), h.iofr.end());
    static DeviceBuffer<scalar> stageS;
    static DeviceBuffer<label> stageL;
    if (nh > 0)
    {
        stageS.copyFrom(packS);
        stageL.copyFrom(packL);
    }
    // every array of the two staged blocks to its buffer, whole or run by run, in one launch a block
    const std::vector<DeviceBoundaryRange> whole{DeviceBoundaryRange{0, n}};
    const std::vector<DeviceBoundaryRange>& runs = ranges ? *ranges : whole;
    auto sized = [&](auto& to)
    {
        if (to.size() == n) return;
        if (ranges)
        {
            throw std::runtime_error("brae: refreshDeviceVectorBoundaryState: a run-by-run refresh of a "
                                     "boundary that was never built whole.");
        }
        to.resize(n);
    };
    std::vector<DeviceBuffer<scalar>*> toS;
    std::vector<int> fromS;
    std::vector<DeviceBuffer<label>*> toL;
    std::vector<int> fromL;
    for (int k = 0; k < 3; ++k)
    {
        sized(db.comp[k].ioStored);
        sized(db.comp[k].valueFraction);
        sized(db.comp[k].refValue);
        sized(db.comp[k].refGrad);
        sized(db.comp[k].bcType);
        sized(db.comp[k].ioFresh);
        toS.push_back(&db.comp[k].ioStored);
        fromS.push_back(4*k);
        toS.push_back(&db.comp[k].valueFraction);
        fromS.push_back(4*k + 1);
        toS.push_back(&db.comp[k].refValue);
        fromS.push_back(4*k + 2);
        toS.push_back(&db.comp[k].refGrad);
        fromS.push_back(4*k + 3);
        toL.push_back(&db.comp[k].bcType);
        fromL.push_back(k);
        toL.push_back(&db.comp[k].ioFresh);
        fromL.push_back(3);
    }
    deviceScatterRuns(stageS, nh, runs, toS, fromS);
    deviceScatterRuns(stageL, nh, runs, toL, fromL);
}

} // namespace brae
