#pragma once
// OpenFOAM's compressible RAS kOmegaSST on the device, as a compressible solver holds it: compressible::turbulenceModel::
// New(rho, U, phi, thermo) (rhoCentralFoam's createFields.H). The layers OpenFOAM stacks, and what each is here:
//
//   CompressibleTurbulenceModel  rho_, U_, alphaRhoPhi_ = the solver's MASS flux phi (rhoCentralFoam's central one,
//                                reassigned every step), phi() = phi/fvc::interpolate(rho) (the volumetric flux
//                                kOmegaSSTBase's divU reads); mu() = thermo.mu(), nu() = mu/rho
//   eddyViscosity                nut_; mut() = rho*nut; muEff() = mut() + mu() (cells and patches alike)
//   EddyDiffusivity              alphat_ = rho*nut/Prt (correctNut, EddyDiffusivity.C:34-40; Prt from the coeffDict),
//                                alphaEff() = thermo.alphaEff(alphat) = CpByCpv*(alpha + alphat): the SOLVER's thermo
//                                forms it (heThermo.C:869-876), from alphat() here
//   kOmegaSSTBase                validate() = correctNut() = correctNut(2*magSqr(symm(fvc::grad(U)))) (kOmegaSSTBase.C
//                                :130-145, :420-440); correct() = the omega and k equations and correctNut(S2) -- brae's
//                                shared closure, gpu::kOmegaSSTRAS (kOmegaSST.cuh), which every compressible solver of
//                                OpenFOAM's runs too
//
// The solver calls these where OpenFOAM's solver does (rhoCentralFoam.C: validate() at :71, muEff() at :220,
// alphaEff() at :278, correct() at :290); nothing here knows which solver it serves.
#include "cf_types.cuh"
#include "compressibleRAS_cpp.cuh"
#include "device_boundary.cuh"
#include "device_buffer.cuh"
#include "device_kepsilon.cuh"   // DeviceWallData
#include "device_mesh.cuh"
#include "kOmegaSST.cuh"
#include "limitedSchemes_cpp.cuh"      // cpu::EqnDivScheme, EqnGradScheme
#include "turbulence_transport.cuh"    // turbulence::SolveControls

namespace brae {
namespace gpu {
namespace compressibleRAS {

// the flow the model reads, at the moment it is called
struct Flow
{
    const DeviceBuffer<scalar>* rho = nullptr;      // rho_ (cells) and its patch values
    const DeviceBuffer<scalar>* rhoB = nullptr;
    const DeviceBuffer<scalar>* rhoOld = nullptr;   // rho.oldTime(): rho as the time step started (fvm::ddt's)
    const DeviceBuffer<scalar>* Ux = nullptr;       // U_
    const DeviceBuffer<scalar>* Uy = nullptr;
    const DeviceBuffer<scalar>* Uz = nullptr;
    const DeviceBuffer<scalar>* mu = nullptr;       // thermo.mu(), cells and patches
    const DeviceBuffer<scalar>* muB = nullptr;
    const DeviceBuffer<scalar>* phiInt = nullptr;   // alphaRhoPhi_: the solver's mass flux
    const DeviceBuffer<scalar>* phiBnd = nullptr;
    scalar rDeltaT = 0;                             // Euler's 1/deltaT...
    const DeviceBuffer<scalar>* rDeltaTCells = nullptr;   // ...or localEuler's per-cell rDeltaT
    bool controlPhiUndivided = false;   // a NEGATIVE CONTROL, deliberately wrong: phi() = the mass flux itself
};

class KOmegaSST
{
public:
    KOmegaSST(const brae::compressibleRAS::Case& c, const PrimitiveMesh& m, const FvGeometry& g,
              const std::vector<FvPatch>& patches, const GeometricField<vector>& U, const DeviceMesh& dm);

    // kOmegaSSTBase::validate(): correctNut() on the fields as read
    void validate(const Flow& f, const DeviceVectorBoundary& dbU);
    // CompressibleTurbulenceModel::muEff() = rho*nut + mu, into cells and patch faces (out may be mu itself)
    void muEff(const DeviceBuffer<scalar>& rho, const DeviceBuffer<scalar>& rhoB, const DeviceBuffer<scalar>& mu,
               const DeviceBuffer<scalar>& muB, DeviceBuffer<scalar>& muEffC, DeviceBuffer<scalar>& muEffB) const;
    // EddyDiffusivity::alphat(), cells and patch faces, as the last correctNut left it
    const DeviceBuffer<scalar>& alphat() const { return alphat_; }
    const DeviceBuffer<scalar>& alphatB() const { return alphatB_; }
    // kOmegaSST::correct()
    void correct(const Flow& f, const DeviceVectorBoundary& dbU);

    // the fields, for writing (patch values evaluated as the fields' boundaries hold them)
    const DeviceBuffer<scalar>& k() const { return k_; }
    const DeviceBuffer<scalar>& omega() const { return omega_; }
    const DeviceBuffer<scalar>& nut() const { return nut_; }
    const DeviceBuffer<scalar>& nutB() const { return nutB_; }
    void kB(DeviceBuffer<scalar>& out) const;
    void omegaB(DeviceBuffer<scalar>& out) const;
    const gpu::kOmegaSSTRAS::KOmegaSSTResiduals& residuals() const { return res_; }

private:
    void fill(const Flow& f);   // the closure's input from this call's flow

    const brae::compressibleRAS::Case& c_;
    const DeviceMesh& dm_;
    int nC_ = 0, nB_ = 0;
    DeviceBuffer<scalar> k_, omega_, nut_, nutB_, alphat_, alphatB_;
    DeviceBoundary dbK_, dbOmega_;
    DeviceWallData wall_;
    DeviceBuffer<label> wfMask_, f1One_, nutCalc_, nutKind_, alphatMask_, wfBnd_, constraint_, bCell_;
    void constrainPatches();   // nut's and alphat's constraint patches after correctNut
    DeviceBuffer<scalar> yCell_, yBnd_, nutCmu25_, nutKappa_, nutE_, nutYplLam_, alphatPrt_;
    // per call
    DeviceBuffer<scalar> nuC_, nuB_, nuW_, nutBIn_, nutWIn_, rhoF_, phiByRhoI_, phiByRhoB_;
    cpu::EqnDivScheme omegaDiv_;
    cpu::EqnGradScheme omegaGrad_;
    turbulence::SolveControls svOmega_;
    gpu::kOmegaSSTRAS::KOmegaSSTInput in_;
    gpu::kOmegaSSTRAS::KOmegaSSTResiduals res_;
};

} // namespace compressibleRAS
} // namespace gpu
} // namespace brae
