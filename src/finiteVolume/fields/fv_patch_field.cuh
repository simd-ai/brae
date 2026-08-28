#pragma once
// brae::fvPatchField<T>, boundary-condition base + the core concrete types, mirroring
// OpenFOAM fvPatchField. evaluate() sets the boundary face VALUES from the internal field;
// snGrad() is the surface-normal gradient. Matrix-coupling coefficients (valueInternalCoeffs
// etc.) come with Phase 3 (assembly). Unsupported types throw, no silent fallback.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "wedge_patch.cuh"   // the axisymmetric constraint patch's rotation tensors
#include "foam_field_reader.cuh"
#include "cf_pstream.cuh"
#include "foam_dict.cuh"   // isCoupledInterfaceType
#include <memory>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <cmath>

namespace brae {

template <typename T>
class fvPatchField
{
public:
    explicit fvPatchField(const FvPatch& p) : patch_(p), value_(p.size) {}
    virtual ~fvPatchField() = default;

    // Coupled (processor) patches post their exchange here; others no-op. The owning field
    // calls initEvaluate on all patches, then Pstream::waitAll(), then evaluate on all.
    virtual void initEvaluate(const std::vector<T>& /*internal*/) {}
    virtual void evaluate(const std::vector<T>& internal) = 0;
    virtual bool fixesValue() const = 0;

    // OpenFOAM fvPatchField::assignable() -- "may this patch's value be OVERWRITTEN by an assignment to
    // the field?". Defaults true; false for the fixedValue, mixed and transform families
    // (fvPatchField.H, fixedValueFvPatchField.H:169, mixed/, transform/).
    //
    // This is NOT the same question as fixesValue(), and conflating them is a silent error:
    //   * slip / partialSlip are NOT assignable but do NOT fix a value;
    //   * inletOutlet derives from mixed but OVERRIDES assignable() back to TRUE
    //     (inletOutletFvPatchField.H:164); outletInlet derives from the same base and does not.
    // constrainHbyA (simpleFoam/pEqn.H:3) branches on assignable(); adjustPhi (pEqn.H:6) branches on
    // fixesValue(). Using one for the other changes which patches keep U's boundary value in HbyA.
    virtual bool assignable() const { return true; }

    // Is this specifically an inletOutlet? adjustPhi (pEqn.H) branches on
    // `Up.fixesValue() && !isA<inletOutletFvPatchVectorField>(Up)` -- it needs BOTH questions, because
    // mixedFvPatchField::fixesValue() is TRUE (mixedFvPatchField.H:197) and inletOutlet inherits it, so
    // without the exclusion an inletOutlet outlet would be counted as a FIXED outflow and adjustPhi would
    // have nothing adjustable left to balance against.
    virtual bool isInletOutlet() const { return false; }

    // OpenFOAM's updateCoeffs() for a flux-conditional BC. inletOutlet sets valueFraction = neg(phi) from
    // the patch flux (inletOutletFvPatchField.C), which it reaches through the object registry; brae's
    // host path has no registry, so the SOLVER pushes the flux in once per iteration, exactly where
    // OpenFOAM would have looked it up. A no-op on every other patch type.
    virtual void updateFromFlux(const std::vector<scalar>& /*phip*/) {}

    // OF's flowRateInletVelocity RECOMPUTES its value in updateCoeffs, from the registered rho's PATCH
    // field -- not from the face cells, and not from `rhoInlet`, which is only the fallback for a solver
    // that registers no rho (flowRateInletVelocityFvPatchVectorField.C:updateCoeffs/updateValues):
    //     avgU = -flowRate/gSum(rho*magSf);   value = avgU*n      n = patch().nf(), OUTWARD
    // A patch that is not one does nothing here.
    virtual void updateFromDensity(const std::vector<scalar>& /*rhop*/) {}

    // The CONSTRUCTION-time half of the same thing, and it is a different branch. OF's dict constructor
    // calls evaluate() -> updateCoeffs() only when the case supplies no `value`, so the inlet is already
    // at avgU*n before createFields.H builds phi from it. When the case DOES supply a `value` OF keeps it
    // verbatim and replaces it at the first momentum assembly instead. sbMatched gives no value;
    // angledDuct gives `uniform (0 0 0)`, which makes OF's first inlet mass flux exactly zero. Taking
    // either branch for both was measured at 303x on the convective boundaryCoeffs.
    virtual void updateAtConstruction(const std::vector<scalar>& /*rhop*/) {}

    // OF's turbulent inlets RECOMPUTE their refValue in updateCoeffs, from the patch fields as they stand
    // at that moment -- they do not carry the value the case file was written with:
    //   turbulentIntensityKineticEnergyInlet:      refValue = 1.5*intensity^2*magSqr(Up)
    //   turbulentMixingLengthDissipationRateInlet: refValue = (Cmu^0.75/mixingLength)*kp^1.5
    //   turbulentMixingLengthFrequencyInlet:       refValue = sqrt(kp)/(Cmu^0.25*mixingLength)
    // Any other patch does nothing here.
    virtual void updateTurbulentInlet(
        const std::vector<vector>& /*Up*/,
        const std::vector<scalar>& /*kp*/,
        scalar                     /*Cmu*/) {}

    // Laplacian/gradient matrix coupling: snGrad = gradientInternalCoeffs * phi_P
    // + gradientBoundaryCoeffs. Default (zeroGradient/empty/calculated) contributes nothing.
    virtual std::vector<T> gradientInternalCoeffs() const { return std::vector<T>(patch_.size, T{}); }
    virtual std::vector<T> gradientBoundaryCoeffs() const { return std::vector<T>(patch_.size, T{}); }

    // BC category for the on-device evaluator: 0 = extrapolated (zeroGradient/empty: value tracks the
    // internal cell, valueIC=1, gradIC=0); 1 = fixedValue (fixedValue/noSlip: value=ref, valueIC=0,
    // gradIC=-deltaCoeffs); 2 = calculated (value=ref but extrapolated coeffs).
    virtual int bcCategory() const { return 0; }
    // Is this an epsilonWallFunction? OpenFOAM's epsilon wall treatment keys on the BC TYPE, not on the
    // patch type: createAveragingWeights counts the adjacent faces whose epsilon field carries an
    // epsilonWallFunction, so a `wall` patch with a plain fixedValue or zeroGradient epsilon must NOT
    // contribute a wall constraint or an averaging weight. brae otherwise maps the type to zeroGradient
    // and would lose that distinction.
    virtual bool isTurbulenceWallFunction() const { return false; }
    // Is this nut patch a nutUSpaldingWallFunction? It reads as `calculated` to the matrix -- the value is
    // supplied, not derived from a gradient -- but a model's correctNut must NOT overwrite it with the
    // nut_ = f(nuTilda) field assignment: correctBoundaryConditions() lets the wall function recompute it
    // from Spalding's law, and for SpalartAllmaras (nuTilda fixedValue 0 at a wall) the assignment would
    // otherwise leave the wall with NO eddy viscosity at all.
    virtual bool isNutUSpalding() const { return false; }
    // The patch's REFERENCE value -- inletValue / outletValue / freestreamValue / refValue -- as opposed
    // to its current value(). For most BCs the two are the same object and this returns value(); the
    // read-and-hold family below overrides it. The device boundary builder needs the reference, because
    // that is what its mixed/inletOutlet evaluators blend TOWARDS; handing it value() only worked while
    // the two were kept identical, and stops working the moment a field carries the value OF wrote.
    virtual std::vector<T> refValues() const { return value_; }
    // wedge (axisymmetric constraint): the HALF-angle and FULL-angle rotation tensors, or null on every
    // other patch type. The device builder reads them to set the per-component valueFraction and to
    // recompute the rotated value each step.
    virtual const tensor* wedgeFaceT() const { return nullptr; }
    virtual const tensor* wedgeCellT() const { return nullptr; }
    // flowRateInletVelocity: the dict flow rate, so the solver can recompute avgU against the live rho.
    virtual scalar flowRateValue() const { return 0.0; }

    // symmetryPlane/symmetry needs PER-COMPONENT device categories for vectors (the wall-normal component
    // is reflected = fixedValue 0, the tangential components are zeroGradient). The device builder keys on
    // this flag + the face normal; scalars are plain zeroGradient so it is a no-op for them.
    virtual bool isSymmetry() const { return false; }

    // Convection/value matrix coupling: faceValue = valueInternalCoeffs * phi_P
    // + valueBoundaryCoeffs. Default (zeroGradient/empty/calculated) = (1, 0).
    virtual std::vector<T> valueInternalCoeffs() const { return std::vector<T>(patch_.size, tUniform<T>(1)); }
    virtual std::vector<T> valueBoundaryCoeffs() const { return std::vector<T>(patch_.size, T{}); }

    // Mixed (Robin) BC support for the device builder (bcCategory 5): the per-face valueFraction (null = not mixed)
    // and the freestream sign (true = velocity vf=0.5-0.5 U.n/|U|, false = pressure vf=0.5+0.5 U.n/|U|).
    virtual const std::vector<scalar>* valueFractionPtr() const { return nullptr; }
    virtual bool mixedVelocitySign() const { return true; }
    // fixedGradient's prescribed normal gradient; null for every other BC, so the device refGrad stays 0
    // and zeroGradient behaves exactly as before.
    virtual const std::vector<T>* refGradPtr() const { return nullptr; }

    const std::vector<T>& value() const { return value_; }
    void setValue(const std::vector<T>& v) { value_ = v; }   // e.g. nutkWallFunction writing nut at walls

    // Replace the value the patch HOLDS, not merely the one it currently exposes. OF's operator==(...)
    // on a fixedValue or calculated patch replaces the stored value, so a later evaluate() reproduces it.
    // setValue writes value_ ALONE, and both families rebuild value_ from their stored copy on the next
    // evaluate() -- which silently reverted a corrected flowRateInletVelocity inlet to the `rhoInlet`
    // seed, at a measured 24% of the case's prescribed mass flow. Use this whenever the new value has to
    // survive an evaluateBoundary().
    virtual void setStoredValues(std::vector<T> v) { value_ = std::move(v); }
    // Coupled-patch neighbour (halo) values for the matrix; non-coupled patches return value().
    virtual const std::vector<T>& patchNeighbourField() const { return value_; }
    const FvPatch&        patch() const { return patch_; }

    std::vector<T> patchInternalField(const std::vector<T>& internal) const
    {
        std::vector<T> pif(patch_.size);
        for (label i = 0; i < patch_.size; ++i)
            pif[i] = internal[patch_.faceCells[i]];
        return pif;
    }

protected:
    const FvPatch& patch_;
    std::vector<T> value_;
};

// fixedValue: value is prescribed (uniform or per-face).
template <typename T>
class FixedValuePatchField : public fvPatchField<T>
{
public:
    FixedValuePatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals)
        : fvPatchField<T>(p), uniform_(uniform), uniformValue_(uval), values_(std::move(vals)) {}
    void evaluate(const std::vector<T>&) override
    {
        for (label i = 0; i < this->patch_.size; ++i)
            this->value_[i] = uniform_ ? uniformValue_ : values_[i];
    }
    void setStoredValues(std::vector<T> v) override
    {
        uniform_ = false;
        values_  = std::move(v);
        evaluate({});
    }
    bool fixesValue() const override { return true; }
    bool assignable() const override { return false; }   // OF fixedValueFvPatchField.H:169
    int  bcCategory() const override { return 1; }

    std::vector<T> gradientInternalCoeffs() const override        // -deltaCoeffs
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = tUniform<T>(-this->patch_.deltaCoeffs[i]);
        return r;
    }
    std::vector<T> gradientBoundaryCoeffs() const override        // deltaCoeffs * refValue
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = (uniform_ ? uniformValue_ : values_[i]) * this->patch_.deltaCoeffs[i];
        return r;
    }
    std::vector<T> valueInternalCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); } // 0
    std::vector<T> valueBoundaryCoeffs() const override            // the fixed value
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = uniform_ ? uniformValue_ : values_[i];
        return r;
    }
private:
    bool           uniform_;
    T              uniformValue_;
    std::vector<T> values_;
};

// totalPressure (incompressible, rho=none/psi=none): a fixedValue p whose value is recomputed each step from the patch
// velocity and the boundary flux:  p_b = p0 - 0.5*neg(phi_b)*magSqr(U_b)   (OF totalPressureFvPatchScalarField).
// Inflow (phi<0): p = p0 - 0.5|U|^2 (static = total - dynamic head); outflow (phi>=0): p = p0. The host build uses p0
// as the fixedValue base (value() = p0); the DEVICE recomputes the per-face refValue each step (deviceUpdateTotalPressure).
// bcCategory()=7 marks the face for the device builder. Pressure (scalar) only.
template <typename T>
class TotalPressurePatchField : public FixedValuePatchField<T>
{
public:
    TotalPressurePatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals)
        : FixedValuePatchField<T>(p, uniform, uval, std::move(vals)) {}
    int bcCategory() const override { return 7; }                   // device: totalPressure (per-step refValue)
};

// surfaceNormalFixedValue / uniformNormalFixedValue (velocity U only): U_b = refValue * face_normal
// (OF surfaceNormalFixedValueFvPatchVectorField: refValue_*patch().nf()). refValue is a SCALAR (signed normal-velocity
// magnitude; <0 = inflow). A plain fixedValue once built, the normals are geometric, refValue is constant for steady;
// any time-`ramp`/Function1 is ignored (the converged value, ramp->1 at endTime). Vector field only. bcCategory()=1.
class SurfaceNormalFixedValuePatchField : public FixedValuePatchField<vector>
{
public:
    SurfaceNormalFixedValuePatchField(
        const FvPatch& p,
        bool uniform,
        scalar uval,
        std::vector<scalar> vals)
        : FixedValuePatchField<vector>(p, false, vector{}, build(p, uniform, uval, vals)) {}
private:
    static std::vector<vector> build(
        const FvPatch& p,
        bool uniform,
        scalar uval,
        const std::vector<scalar>& vals)
    {
        std::vector<vector> v(p.size);
        for (label i = 0; i < p.size; ++i)
            v[i] = (uniform ? uval : (i < (label)vals.size() ? vals[i] : scalar(0))) * p.nf[i];
        return v;
    }
};

// flowRateInletVelocity (OF flowRateInletVelocityFvPatchVectorField). The default (extrapolateProfile
// false) branch is, verbatim from updateValues():
//
//     avgU = -flowRate / gSum(rho*patch().magSf());   operator==(avgU*n)      n = patch().nf() (OUTWARD)
//
// so a POSITIVE flow rate gives inflow, because n points out of the domain. Two branches, selected by
// which key the dict carries (updateCoeffs()):
//   volumetricFlowRate -> rho == 1, the sum is pure area and the value is GEOMETRIC (never changes);
//   massFlowRate       -> rho is the registered rho patch field, so the value moves with the solution
//                         and must be recomputed every outer iteration (bcCategory 9 does that).
// rhoInlet is NOT an override: OF uses it only when no rho field is registered. rhoSimpleFoam registers
// one, so it is the seed here and the live rho takes over on the first update.
class FlowRateInletVelocityPatchField : public FixedValuePatchField<vector>
{
public:
    // The patch VALUE starts as the case's own `value` entry, exactly as OF does. OF's
    // flowRateInletVelocity is a fixedValue whose value is only replaced when updateCoeffs() first runs --
    // which happens when the momentum equation is assembled, i.e. AFTER createFields.H has already built
    // phi from the seeded field. On angledDuct (`value uniform (0 0 0)`, massFlowRate 0.1) that makes OF's
    // inlet mass flux EXACTLY 0 in the first momentum equation and -0.1 from the second onward (measured).
    //
    // brae used to compute avgU*n here, in the constructor, so its very first flux was already -0.1. Every
    // coefficient agreed (sum|iC| to 8 s.f., avgU to 5 s.f.) but the convective boundaryCoeffs did not:
    // OF's bC = -phi_b*U_b vanishes at iteration 1 while brae's did not, leaving sum|bC| 303x apart.
    FlowRateInletVelocityPatchField(
        const FvPatch& p,
        scalar flowRate,
        bool isMass,
        scalar rhoInlet,
        bool valueUniform,
        const vector& uniformValue,
        const std::vector<vector>& values)
        : FixedValuePatchField<vector>(
              p,
              // OF's dict constructor: if the case gives a `value`, that IS the starting field; if it does
              // not, OF calls evaluate() -> updateCoeffs() and starts from the computed avgU*n
              // (flowRateInletVelocityFvPatchVectorField.C). Both branches matter and they differ per case:
              // angledDuct gives `value uniform (0 0 0)` (so OF's first inlet flux is 0), squareBend gives
              // no value at all (so OF's first inlet flux is already -mdot). Taking only the first branch
              // segfaulted squareBend on an empty value list; taking only the second was the 303x bC error.
              (valueUniform || !values.empty()) ? valueUniform : false,
              uniformValue,
              (valueUniform || !values.empty()) ? values : build(p, flowRate, isMass, rhoInlet)),
          isMass_(isMass),
          flowRate_(flowRate),
          hadValue_(valueUniform || !values.empty())
    {}

    void updateAtConstruction(const std::vector<scalar>& rhop) override
    {
        if (hadValue_) return;      // OF keeps the case's `value` until the first momentum assembly
        updateFromDensity(rhop);
    }
    // 9 = flowRateInletVelocity: refValue recomputed per step from the live boundary rho (mass form only).
    int bcCategory() const override { return isMass_ ? 9 : 1; }
    scalar flowRateValue() const override { return flowRate_; }

    // OF updateValues(rho), verbatim. Called where OpenFOAM calls updateCoeffs -- when the momentum
    // equation is assembled -- so the inlet moves with the solution instead of staying at the seed the
    // constructor built. The VOLUMETRIC branch is recomputed too: OF passes one{} rather than skipping,
    // which matters when a case supplies both `volumetricFlowRate` and a `value` that disagrees with it.
    void updateFromDensity(const std::vector<scalar>& rhop) override
    {
        const label n = this->patch_.size;
        if (n == 0) return;
        // REFUSE rather than treat a missing density as 1. Silently using the volumetric form for a
        // massFlowRate inlet rescales the whole inlet by rho, which is the shape of the angledDuct defect.
        if (isMass_ && rhop.size() < static_cast<std::size_t>(n))
        {
            throw std::runtime_error(
                std::string("flowRateInletVelocity on patch '") + this->patch_.name
                + "': massFlowRate needs rho on every face of the patch, and was given fewer. "
                  "Refusing rather than falling back to a density this case did not ask for.");
        }
        scalar sumRhoA = 0.0;
        for (label i = 0; i < n; ++i)
        {
            sumRhoA += (isMass_ ? rhop[i] : scalar(1)) * this->patch_.magSf[i];
        }
        if (!(sumRhoA > 0.0))
        {
            throw std::runtime_error(
                std::string("flowRateInletVelocity on patch '") + this->patch_.name
                + "': gSum(rho*magSf) is not positive, so no inlet velocity can be formed from the "
                  "prescribed flow rate.");
        }
        const scalar avgU = -flowRate_ / sumRhoA;
        std::vector<vector> v(n);
        for (label i = 0; i < n; ++i) v[i] = avgU * this->patch_.nf[i];
        this->setStoredValues(std::move(v));
    }

private:
    bool   isMass_;
    scalar flowRate_ = 0.0;
    bool   hadValue_ = false;

    static std::vector<vector> build(
        const FvPatch& p,
        scalar flowRate,
        bool isMass,
        scalar rhoInlet)
    {
        // gSum(rho*magSf). Volumetric -> rho = 1. Mass -> the seed rho (rhoInlet if the case gave one,
        // else 1); the per-step update replaces it with the real patch rho before it matters.
        const scalar rhoSeed = (!isMass) ? scalar(1) : (rhoInlet > 0.0 ? rhoInlet : scalar(1));
        scalar sumRhoA = 0.0;
        for (label i = 0; i < p.size; ++i) sumRhoA += rhoSeed * p.magSf[i];
        const scalar avgU = (sumRhoA > 0.0) ? -flowRate / sumRhoA : scalar(0);
        std::vector<vector> v(p.size);
        for (label i = 0; i < p.size; ++i) v[i] = avgU * p.nf[i];
        return v;
    }
};

// timeVaryingMappedFixedValue: a fixedValue whose per-face value is the external boundaryData profile MAPPED onto the
// faces, NEAREST data point to each face centre (exact when the data is finer than the mesh, as in pitzDailyExptInlet's
// 70-point inlet profile). Time interpolation / offset / setAverage not applied (steady; offset 0, setAverage off here).
// bcCategory()=1. Works for scalar (k/epsilon) and vector (U).
template <typename T>
class TimeVaryingMappedPatchField : public FixedValuePatchField<T>
{
public:
    TimeVaryingMappedPatchField(
        const FvPatch& p,
        const std::vector<vector>& pts,
        const std::vector<T>& vals)
        : FixedValuePatchField<T>(p, false, T{}, mapNearest(p, pts, vals)) {}
private:
    static std::vector<T> mapNearest(
        const FvPatch& p,
        const std::vector<vector>& pts,
        const std::vector<T>& vals)
    {
        std::vector<T> v(p.size);
        for (label i = 0; i < p.size; ++i)
        {
            const vector& c = p.Cf[i];
            scalar best = 1e300;
            label bj = 0;
            for (std::size_t j = 0; j < pts.size(); ++j)
            {
                const scalar dx = c.x - pts[j].x, dy = c.y - pts[j].y, dz = c.z - pts[j].z;
                const scalar d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < best)
                {
                    best = d2;
                    bj = (label)j;
                }
            }
            v[i] = vals.empty() ? T{} : vals[bj];
        }
        return v;
    }
};

// zeroGradient: boundary value == adjacent internal cell value.
template <typename T>
class ZeroGradientPatchField : public fvPatchField<T>
{
public:
    explicit ZeroGradientPatchField(const FvPatch& p) : fvPatchField<T>(p) {}
    void evaluate(const std::vector<T>& internal) override
    {
        this->value_ = this->patchInternalField(internal);
    }
    bool fixesValue() const override { return false; }
};

// epsilonWallFunction: a zeroGradient boundary value, with the near-wall CELL constrained separately by
// the turbulence model (setValues) and the production there replaced. The class exists only so the model
// can tell which patches carry it -- see isTurbulenceWallFunction above. omegaWallFunction maps here too.
template <typename T>
class EpsilonWallFunctionPatchField : public ZeroGradientPatchField<T>
{
public:
    explicit EpsilonWallFunctionPatchField(const FvPatch& p) : ZeroGradientPatchField<T>(p) {}
    bool isTurbulenceWallFunction() const override { return true; }
};

// fixedGradient: the normal gradient is prescribed, the value follows.
//     value = patchInternalField + gradient/deltaCoeffs        (fixedGradientFvPatchField.C:191)
// In the matrix it is EXACTLY zeroGradient plus a source: valueInternalCoeffs 1 and
// gradientInternalCoeffs 0 are the same as zeroGradient's, and only the boundary coefficients pick up the
// gradient (valueBoundaryCoeffs = g/deltaCoeffs, gradientBoundaryCoeffs = g -- .C:205,216,224,232). That is
// why the device side carries a per-face refGrad that is simply ZERO for zeroGradient: one code path, and
// every existing zeroGradient patch keeps its exact behaviour.
template <typename T>
class FixedGradientPatchField : public fvPatchField<T>
{
public:
    FixedGradientPatchField(const FvPatch& p, bool uniform, const T& g, const std::vector<T>& gs)
        : fvPatchField<T>(p), uniform_(uniform), gUniform_(g), gValues_(gs)
    {
        grad_.assign(static_cast<std::size_t>(p.size), T{});
        for (label i = 0; i < p.size; ++i)
            grad_[static_cast<std::size_t>(i)] =
                uniform_ ? gUniform_
                         : (static_cast<std::size_t>(i) < gValues_.size() ? gValues_[static_cast<std::size_t>(i)] : T{});
    }
    void evaluate(const std::vector<T>& internal) override
    {
        const std::vector<T> pif = this->patchInternalField(internal);
        this->value_.resize(pif.size());
        for (std::size_t i = 0; i < pif.size(); ++i)
            this->value_[i] = pif[i] + grad_[i] / this->patch_.deltaCoeffs[i];
    }
    bool fixesValue() const override { return false; }          // the VALUE is not fixed; the gradient is
    const std::vector<T>* refGradPtr() const override { return &grad_; }

private:
    bool uniform_;
    T gUniform_;
    std::vector<T> gValues_;
    std::vector<T> grad_;
};

// noSlip: a fixedValue velocity wall whose value STARTS at zero -- which is not the same thing as a
// wall whose value IS zero, and the difference is not academic.
//
// OpenFOAM's noSlipFvPatchVectorField derives from fixedValueFvPatchVectorField with no updateCoeffs
// override, and fixedValue's coefficients are built from *this -- the LIVE patch value. MRFZone::
// correctBoundaryVelocity then writes Omega x (Cf - origin) onto exactly such patches with
// `Ubf[patchi] == pfld`, and operator== assigns the underlying Field whatever assignable() says (that
// is why OpenFOAM uses == there and not =). So under MRF the rotor wall genuinely MOVES.
//
// This class used to return a hardcoded zero from the base valueBoundaryCoeffs and re-zero value_ on
// every evaluate(), i.e. it baked in "this wall is stationary". Correct for every case that does not
// rotate, and wrong for every case that does: on mixerVessel2D it put 93.8% of the whole momentum
// residual on the rotor patch. For a wall that never gets assigned, value_ stays zero and every
// coefficient below is identical to what the hardcoded version produced.
template <typename T>
class NoSlipPatchField : public fvPatchField<T>
{
public:
    bool assignable() const override { return false; }   // OF fixedValueFvPatchField.H:169
    explicit NoSlipPatchField(const FvPatch& p) : fvPatchField<T>(p) {}
    // fixedValue's evaluate does NOT re-establish the value -- the field IS the value. Zeroing here
    // would discard whatever correctBoundaryVelocity just wrote.
    void evaluate(const std::vector<T>&) override {}
    bool fixesValue() const override { return true; }
    int  bcCategory() const override { return 1; }

    std::vector<T> gradientInternalCoeffs() const override        // -deltaCoeffs
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = tUniform<T>(-this->patch_.deltaCoeffs[i]);
        return r;
    }
    std::vector<T> gradientBoundaryCoeffs() const override        // deltaCoeffs * the LIVE value
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = this->value_[i] * this->patch_.deltaCoeffs[i];
        return r;
    }
    std::vector<T> valueInternalCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); } // 0
    std::vector<T> valueBoundaryCoeffs() const override { return this->value_; }
};

// empty: 2D front/back. Value tracks the internal field (the out-of-plane component is zero,
// so it contributes nothing to in-plane fluxes); the discretisation ignores empty patches.
template <typename T>
class EmptyPatchField : public fvPatchField<T>
{
public:
    explicit EmptyPatchField(const FvPatch& p) : fvPatchField<T>(p) {}
    void evaluate(const std::vector<T>& internal) override
    {
        this->value_ = this->patchInternalField(internal);
    }
    bool fixesValue() const override { return false; }
};

// symmetryPlane / symmetry: a slip plane. For a SCALAR (p,k,epsilon,nut) the normal gradient is zero, so the
// boundary value == the adjacent internal cell value (identical to zeroGradient). For a VECTOR (U) the
// wall-normal component is reflected and the tangential components are mirrored: the boundary value is the
// tangential projection v - n(n.v) (OF basicSymmetry value = (v + (I-2nn)&v)/2 = v - n(n.v)), and the
// snGrad is -n(n.v)*deltaCoeffs. The matrix coupling (per-component fixedValue-0 on the normal axis,
// zeroGradient on the tangential axes) is applied on the device by buildDeviceVectorBoundary, which keys on
// isSymmetry() + the face normal. The vector specialisation below is axis-agnostic for value()/output.
template <typename T>
class SymmetryPlanePatchField : public fvPatchField<T>
{
public:
    bool assignable() const override { return false; }   // OF: symmetry is a transform patch field
    explicit SymmetryPlanePatchField(const FvPatch& p) : fvPatchField<T>(p) {}

    // OF basicSymmetryFvPatchField. For a SCALAR the normal gradient is zero and this is zeroGradient,
    // which is what the base coefficients already give. For a VECTOR it is a TRANSFORM patch field and
    // neither the value nor the coefficients are the scalar ones:
    //
    //   evaluate:  value = (iF + transform(I - 2*sqr(n), iF))/2 = iF - (n & iF)*n   (the TANGENTIAL part)
    //   snGrad  =  (transform(I - 2*sqr(n), iF) - iF)*deltaCoeffs/2 = -(n & iF)*n*deltaCoeffs
    //
    // and, from transformFvPatchField.C with snGradTransformDiag = cmptMag(nf)
    // (basicSymmetryFvPatchField.C:121-126):
    //
    //   valueInternalCoeffs    =  1 - cmptMag(n)
    //   valueBoundaryCoeffs    =  value - cmptMultiply(valueInternalCoeffs, iF)
    //   gradientInternalCoeffs = -deltaCoeffs*cmptMag(n)
    //   gradientBoundaryCoeffs =  snGrad - cmptMultiply(gradientInternalCoeffs, iF)
    //
    // This class used to return the scalar value for every type and the BASE coefficients (1, 0, 0, 0),
    // so a slip patch on a vector field got the full internal velocity and a matrix contribution that
    // ignored the plane entirely. It is invisible to any gate that compares FIELDS, because OpenFOAM
    // writes no `value` for a transform patch; angledDuct's momentum coefficients found it at exactly
    // 1.0 on the slip patch with every other patch at 1e-14.
    void evaluate(const std::vector<T>& internal) override
    {
        pif_ = this->patchInternalField(internal);
        if constexpr (std::is_same<T, vector>::value)
        {
            this->value_.resize(this->patch_.size);
            for (label i = 0; i < this->patch_.size; ++i)
            {
                const vector& n = this->patch_.nf[i];
                const scalar nd = n.x*pif_[i].x + n.y*pif_[i].y + n.z*pif_[i].z;
                this->value_[i] = vector{ pif_[i].x - nd*n.x,
                                          pif_[i].y - nd*n.y,
                                          pif_[i].z - nd*n.z };
            }
        }
        else
        {
            this->value_ = pif_;                               // scalar: zeroGradient
        }
    }

    std::vector<T> valueInternalCoeffs() const override
    {
        if constexpr (!std::is_same<T, vector>::value)
        {
            return fvPatchField<T>::valueInternalCoeffs();
        }
        else
        {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
        {
            const vector& n = this->patch_.nf[i];
            r[i] = vector{ 1.0 - std::fabs(n.x), 1.0 - std::fabs(n.y), 1.0 - std::fabs(n.z) };
        }
        return r;
        }
    }

    std::vector<T> valueBoundaryCoeffs() const override
    {
        if constexpr (!std::is_same<T, vector>::value)
        {
            return fvPatchField<T>::valueBoundaryCoeffs();
        }
        else
        {
        const std::vector<T> vic = valueInternalCoeffs();
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
        {
            const T& p = pifAt(i);
            r[i] = vector{ this->value_[i].x - vic[i].x*p.x,
                           this->value_[i].y - vic[i].y*p.y,
                           this->value_[i].z - vic[i].z*p.z };
        }
        return r;
        }
    }

    std::vector<T> gradientInternalCoeffs() const override
    {
        if constexpr (!std::is_same<T, vector>::value)
        {
            return fvPatchField<T>::gradientInternalCoeffs();
        }
        else
        {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
        {
            const vector& n = this->patch_.nf[i];
            const scalar dc = this->patch_.deltaCoeffs[i];
            r[i] = vector{ -dc*std::fabs(n.x), -dc*std::fabs(n.y), -dc*std::fabs(n.z) };
        }
        return r;
        }
    }

    std::vector<T> gradientBoundaryCoeffs() const override
    {
        if constexpr (!std::is_same<T, vector>::value)
        {
            return fvPatchField<T>::gradientBoundaryCoeffs();
        }
        else
        {
        const std::vector<T> gic = gradientInternalCoeffs();
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
        {
            const vector& n = this->patch_.nf[i];
            const T& p = pifAt(i);
            const scalar dc = this->patch_.deltaCoeffs[i];
            const scalar nd = n.x*p.x + n.y*p.y + n.z*p.z;
            const vector sn { -nd*n.x*dc, -nd*n.y*dc, -nd*n.z*dc };   // snGrad
            r[i] = vector{ sn.x - gic[i].x*p.x, sn.y - gic[i].y*p.y, sn.z - gic[i].z*p.z };
        }
        return r;
        }
    }

    bool fixesValue() const override { return false; }
    bool isSymmetry() const override { return true; }

private:
    // The patch internal field, cached at evaluate(). OpenFOAM's coefficient methods call
    // patchInternalField() directly; brae's take no arguments, so it is kept here instead.
    std::vector<T> pif_;
    const T& pifAt(label i) const
    {
        static const T zero{};
        return (i < static_cast<label>(pif_.size())) ? pif_[i] : zero;
    }
};

// The vector case used to live here as an explicit specialization. It computed the right VALUE
// (v - n(n.v)) but nothing else, so it silently shadowed the in-class evaluate above -- which is the one
// that caches the patch internal field the four transform coefficients are built from. The value agreed
// with OpenFOAM while every coefficient was computed against a zero internal field: valueBoundaryCoeffs
// came out equal to value_ and gradientBoundaryCoeffs came out exactly zero. Kept as a comment because
// "the value is right so evaluate() is right" is what made this take a while to find.

// wedge (OF wedgeFvPatchField) -- the AXISYMMETRIC constraint patch. The mesh is one cell thick in the
// azimuthal direction and the two bounding planes are related by a rotation about the symmetry axis, so
// the patch value is the CELL value ROTATED onto the patch plane:
//
//     value                  = transform(faceT, patchInternalField)
//     snGrad                 = (transform(cellT, pif) - pif)*0.5*deltaCoeffs
//     snGradTransformDiag_k  = 0.5*(1 - cellT_kk)                       [per component]
//     valueInternalCoeffs_k  = 1 - snGradTransformDiag_k
//     gradientInternalCoeffs_k = -deltaCoeffs*snGradTransformDiag_k
//
// FOR A SCALAR IT IS EXACTLY zeroGradient, and that is not an approximation: OF specialises
// wedgeFvPatchField<scalar> to return snGrad 0, value = pif and snGradTransformDiag 0 (a scalar has no
// direction to rotate). Only the VECTOR case carries the rotation -- which is why the coefficients above
// are per-component SCALARS with the cross-component coupling living entirely in the value.
template <typename T>
class WedgePatchField : public fvPatchField<T>
{
public:
    bool assignable() const override { return false; }   // OF: wedge is a transform patch field
    WedgePatchField(const FvPatch& p, const tensor& faceT, const tensor& cellT)
        : fvPatchField<T>(p), faceT_(faceT), cellT_(cellT) {}
    void evaluate(const std::vector<T>& internal) override
    {
        this->value_ = this->patchInternalField(internal);   // scalar: zeroGradient, exactly as OF
    }
    bool fixesValue() const override { return false; }
    int  bcCategory() const override { return 0; }           // scalar: the device's zeroGradient
    const tensor* wedgeFaceT() const override { return &faceT_; }
    const tensor* wedgeCellT() const override { return &cellT_; }

protected:
    tensor faceT_, cellT_;
};

// vector: the rotation is real. value = faceT & U_cell.
template <> inline void WedgePatchField<vector>::evaluate(const std::vector<vector>& internal)
{
    for (label i = 0; i < this->patch_.size; ++i)
    {
        const vector& v = internal[this->patch_.faceCells[i]];
        this->value_[i] = vector{faceT_.xx*v.x + faceT_.xy*v.y + faceT_.xz*v.z,
                                 faceT_.yx*v.x + faceT_.yy*v.y + faceT_.yz*v.z,
                                 faceT_.zx*v.x + faceT_.zy*v.y + faceT_.zz*v.z};
    }
}
// ...and a vector wedge is a MIXED (Robin) boundary per component, with valueFraction
// d_k = 0.5*(1 - cellT_kk) -- which reproduces OF's valueInternalCoeffs/gradientInternalCoeffs exactly
// (mixed gives 1 - vf and -deltaCoeffs*vf). Category 5 is the device's mixed slot.
template <> inline int WedgePatchField<vector>::bcCategory() const { return 5; }

// Shared storage + read-and-hold value() for the OF calculated / inletOutlet / outletInlet /
// pressureInletOutletVelocity / mixed family, they differ ONLY in which device category (bcCategory) claims the
// face; value_[i] is the same uniform-or-per-face read value (= refValue). Each OF-named subclass below stays a
// distinct type so OpenFOAM developers still read them 1:1 against OF's fvPatchField hierarchy.
template <typename T>
class ExtrapolatedValuePatchField : public fvPatchField<T>
{
public:
    ExtrapolatedValuePatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals)
        : fvPatchField<T>(p), uniform_(uniform), uniformValue_(uval), values_(std::move(vals)) { evaluate({}); }
    void evaluate(const std::vector<T>&) override                  // value() = the read value (OF calculated-style)
    {
        for (label i = 0; i < this->patch_.size; ++i)
            this->value_[i] = refValue(i);
    }
    std::vector<T> refValues() const override
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i) r[i] = refValue(i);
        return r;
    }
    bool fixesValue() const override { return false; }
protected:
    T refValue(label i) const { return uniform_ ? uniformValue_ : (values_.empty() ? T{} : values_[i]); }

public:
    // OF writes `this->refValue() = ...`; the turbulent inlets do it every updateCoeffs.
    void setRefValues(std::vector<T> v) { uniform_ = false; values_ = std::move(v); }
    void setStoredValues(std::vector<T> v) override
    {
        setRefValues(std::move(v));
        evaluate({});
    }

protected:
    bool           uniform_;
    T              uniformValue_;
    std::vector<T> values_;
};

// calculated: value is read from file and kept (used by e.g. nut interior/empty patches).
template <typename T>
class CalculatedPatchField : public ExtrapolatedValuePatchField<T>
{
public:
    using ExtrapolatedValuePatchField<T>::ExtrapolatedValuePatchField;   // read-and-hold value(); OF calculatedFvPatchField
    int bcCategory() const override { return 2; }
};

// nutUSpaldingWallFunction: a `calculated` value as far as the matrix is concerned, but one the turbulence
// model has to recompute from the velocity every correctNut rather than overwrite. Same shape as the
// device's deviceBoundaryNutSpalding; this only makes the patch identifiable on the host.
template <typename T>
class NutUSpaldingPatchField : public CalculatedPatchField<T>
{
public:
    using CalculatedPatchField<T>::CalculatedPatchField;
    bool isNutUSpalding() const override { return true; }
};

// mixed (Robin) BC, OF mixedFvPatchField (refGrad = 0). value = (1-vf)*internal + vf*refValue; the per-face
// valueFraction vf blends linearly between zeroGradient (vf=0) and fixedValue (vf=1) in BOTH the value and the
// matrix coeffs (gradientInternalCoeffs = -vf*dc, valueInternalCoeffs = 1-vf, etc.). freestreamVelocity /
// freestreamPressure are mixed with vf = 0.5 -/+ 0.5*(U.n)/|U| (continuous flow-angle blend, NOT the binary
// inletOutlet switch). OF mixedFvPatchField::fixesValue() == true -> references the pressure (needReference false).
// The valueFraction is NOT a constant for the freestream family: OF recomputes it every updateCoeffs from
// the flow angle, and updateMixedFreestream below is the host form of that (the device has
// deviceUpdateMixedFreestream). Leaving it at the seed 0.5 makes every far-field face a half-and-half
// blend regardless of whether it is inflow or outflow, which on a case whose whole far field is
// freestream -- airFoil2D -- measured 665x on the momentum residual and 3551x on pressure against
// OpenFOAM's own converged state.
template <typename T>
class MixedPatchField : public ExtrapolatedValuePatchField<T>     // value() = refValue (freestreamValue); base evaluate() sets it
{
public:
    bool assignable() const override { return false; }   // OF: mixed
    MixedPatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals,
        bool velocitySign,
        // The freestream family (freestreamVelocity/freestreamPressure) is the one whose valueFraction
        // brae actually maintains -- see evaluate() for why that gates the blend.
        bool freestream = false,
        // The file's own `value` entry, per face, when it had one. OF's mixed constructor READS it and
        // does not re-evaluate, so a restart resumes with the field OpenFOAM wrote rather than with a
        // blend of a valueFraction no iteration has computed yet. refValue stays freestreamValue; only
        // value() carries this, which is why the device builder had to be taught to ask for the
        // reference explicitly (refValues()) instead of reading value().
        std::vector<T> readValue = {})
        : ExtrapolatedValuePatchField<T>(p, uniform, uval, std::move(vals)),
          velocitySign_(velocitySign), freestream_(freestream), vf_(p.size, 0.5)
    {
        if (readValue.size() == static_cast<std::size_t>(p.size)) this->value_ = std::move(readValue);
    }
    bool fixesValue() const override { return true; }               // OF mixedFvPatchField::fixesValue() == true
    int  bcCategory() const override { return 5; }                  // device: mixed (per-face valueFraction blend)

    // OF mixedFvPatchField::evaluate -- a BLEND, not the refValue:
    //     value = lerp(patchInternalField + refGrad/deltaCoeffs, refValue, valueFraction)
    // Taking refValue outright puts a far-field patch at freestreamValue everywhere instead of the field
    // OpenFOAM converges to; on airFoil2D its inlet p runs 0.74 to -1.77 across the patch, and the
    // pressure residual at OpenFOAM's own converged state was 17x worse without this.
    //
    // GATED ON `freestream_`, and the reason is not caution. OF's evaluate calls updateCoeffs() FIRST, and
    // the other shape this class serves -- the `mixed` T patch basicThermo maps onto mixedEnergy --
    // overrides updateCoeffs to rebuild refValue/refGrad/valueFraction in ENERGY space from the thermo.
    // brae carries no such conversion, so its seeded valueFraction is not the one OF blends with, and
    // blending on it measurably disagrees (mx_vs_openfoam, hotWall: OF 467.8 against a blended 367.9).
    // The freestream family is maintained -- updateMixedFreestream rewrites vf from the flow angle every
    // iteration -- so the blend runs exactly where its inputs are real.
    void evaluate(const std::vector<T>& internal) override
    {
        // OF's evaluate calls updateCoeffs() FIRST, and for this family updateCoeffs IS the flow-angle
        // valueFraction -- so a blend is only defined once that has run. Until then (construction, and the
        // evaluateBoundary that follows a field read) the value stands as read: on a restart the file's
        // `value` is already OpenFOAM's converged blend, and re-blending it against the 0.5 seed threw it
        // away (restart_vs_openfoam, p L2rel 4.8e-03 against a 1e-03 bound).
        // The gate is `vfUpdated_`, NOT the family. What makes the blend defined is whether a REAL
        // valueFraction has been computed for these faces: the freestream family gets one from
        // updateMixedFreestream, and inletOutlet/outletInlet get one from updateFromFlux. The plain
        // `mixed`/mixedEnergy shape sets neither -- brae carries no thermo conversion for it -- so it
        // still falls through here and keeps the value as read, exactly as before.
        (void)freestream_;
        if (!vfUpdated_ || internal.empty()) return;

        const std::vector<T> pif = this->patchInternalField(internal);
        for (label i = 0; i < this->patch_.size; ++i)
        {
            T inner = pif[i];
            if (!refGrad_.empty()) inner = inner + refGrad_[i] * (1.0 / this->patch_.deltaCoeffs[i]);
            this->value_[i] = inner * (1.0 - vf_[i]) + this->refValue(i) * vf_[i];
        }
    }
    const std::vector<scalar>* valueFractionPtr() const override { return &vf_; }
    // mixedEnergy / an external-convection wall carries refGradient beside refValue. Stored per face and
    // handed to DeviceBoundary::refGrad through the same hook fixedGradient uses (B5); the kernels apply
    // OF's (1-vf) weight, so a mixed patch with refGradient 0 stays bit-identical to before.
    const std::vector<T>* refGradPtr() const override { return refGrad_.empty() ? nullptr : &refGrad_; }
    void setRefGrad(std::vector<T> g) { refGrad_ = std::move(g); }
    // A plain `mixed` patch STATES its valueFraction; only the freestream family recomputes it per step.
    // Recording that it HAS been recomputed is what licenses evaluate() to blend -- see there.
    void setValueFraction(std::vector<scalar> f)
    {
        vf_ = std::move(f);
        vfUpdated_ = true;
    }
    bool mixedVelocitySign() const override { return velocitySign_; }
    // OF mixed coeffs with refGrad = 0 (host correctness; the device blends the same way in its kernels):
    std::vector<T> gradientInternalCoeffs() const override        // -vf*deltaCoeffs
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = tUniform<T>(-vf_[i] * this->patch_.deltaCoeffs[i]);
        return r;
    }
    std::vector<T> gradientBoundaryCoeffs() const override        // vf*deltaCoeffs*refValue
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = this->refValue(i) * (vf_[i] * this->patch_.deltaCoeffs[i]);
        return r;
    }
    std::vector<T> valueInternalCoeffs() const override           // 1-vf
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = tUniform<T>(1.0 - vf_[i]);
        return r;
    }
    std::vector<T> valueBoundaryCoeffs() const override           // vf*refValue
    {
        std::vector<T> r(this->patch_.size);
        for (label i = 0; i < this->patch_.size; ++i)
            r[i] = this->refValue(i) * vf_[i];
        return r;
    }
protected:
    // PROTECTED, not private, so a derived flux-conditional patch can SEED the valueFraction without
    // going through setValueFraction() -- which also sets vfUpdated_, and that flag is what tells
    // evaluate() a real blend is available. Seeding through it would make the first evaluateBoundary()
    // after a field read blend against a fraction no iteration has computed, throwing away the value
    // OpenFOAM wrote on a restart. The seed is a starting point; vfUpdated_ is a statement about
    // whether the flow has been consulted, and the two are not the same thing.
    std::vector<scalar> vf_;            // per-face valueFraction (freestream seed 0.5; flux-conditional 0)

private:
    bool                velocitySign_;  // true: vf=0.5-0.5 U.n/|U| (velocity); false: 0.5+0.5 ... (pressure)
    bool                freestream_;    // this class also serves the plain `mixed`/mixedEnergy shape
    bool                vfUpdated_ = false;   // has a real valueFraction been computed yet?

    std::vector<T> refGrad_;
};

// inletOutlet: flux-conditional mix (OF mixed, valueFraction = neg(phi), refValue = inletValue, refGrad = 0).
// Per face: inflow (phi<0) -> fixedValue = inletValue; outflow (phi>=0) -> zeroGradient. The host evaluate()
// does not see the flux; the DEVICE recomputes the per-face fixedValue|zeroGradient choice every iteration from
// the patch flux (deviceUpdateInletOutlet). value()/refValue = inletValue (also the rest-start boundary value);
// bcCategory()=3 marks the face inletOutlet for the device builder. (Device-resident solver path; the host CPU
// assembly path treats category 3 as its zeroGradient base default, see scope in PORTING_INLETOUTLET_BC.md.)
template <typename T>
class InletOutletPatchField : public MixedPatchField<T>   // value()/refValue = inletValue
{
public:
    // TRUE, and this is NOT an oversight in OpenFOAM. mixedFvPatchField::assignable() is false
    // (mixedFvPatchField.H:200), but inletOutletFvPatchField OVERRIDES it back to true
    // (inletOutletFvPatchField.H:163-164) -- "True: this patch field is altered by assignment".
    // outletInlet, which also derives from mixed, does NOT override and so really is false; the two
    // differ on purpose. Inferring this from the base class instead of reading the derived one made
    // constrainHbyA take U at every inletOutlet patch, which OpenFOAM does not do: measured on
    // rhoSimpleFoam's pcEqn, HbyA's boundary was 1.3e-03 out and the resulting phiHbyA 3.5e-03, while the
    // internal field was exact to 1.4e-15.
    bool assignable() const override { return true; }    // OF inletOutletFvPatchField.H:164
    bool isInletOutlet() const override { return true; }
    // The mixed base takes a velocity-sign flag and a freestream flag for the far-field family; neither
    // applies here, so the flux-conditional families get their own four-argument constructor and the
    // valueFraction comes from updateFromFlux instead.
    InletOutletPatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals)
        : MixedPatchField<T>(p, uniform, uval, std::move(vals), /*velocitySign=*/true)
    {
        // SEED THE VALUE FRACTION AT ZERO, not at the mixed base's 0.5. That 0.5 is the FREESTREAM
        // seed -- a half-and-half flow-angle blend waiting for updateMixedFreestream -- and it is
        // meaningless for a flux-conditional patch, which is either fully zeroGradient or fully
        // fixedValue per face and never in between. Zero is the outflow branch, which is what this
        // patch behaved as before it became mixed, so a field whose flux has not been pushed in yet is
        // unchanged rather than half-fixed. Getting this wrong reports valueInternalCoeffs = 0.5 where
        // OpenFOAM has 1, which measured 2.8e-01 on the momentum internalCoeffs.
        this->vf_.assign(p.size, 0.0);
    }

    // inletOutletFvPatchField::updateCoeffs -- valueFraction = neg(phi): 1 where the flux ENTERS the
    // domain (fixedValue on inletValue), 0 where it leaves (zeroGradient). `neg`, not `1 - pos0`: they
    // agree everywhere except at phi exactly 0, where OpenFOAM takes the outflow branch.
    void updateFromFlux(const std::vector<scalar>& phip) override
    {
        std::vector<scalar> vf(this->patch_.size, 0.0);
        for (label i = 0; i < this->patch_.size && i < (label)phip.size(); ++i)
            vf[i] = (phip[i] < 0.0) ? 1.0 : 0.0;
        this->setValueFraction(std::move(vf));
    }
    int bcCategory() const override { return 3; }                  // inletOutlet (device: per-face fixedValue|zeroGradient)
};

// The two RAS inlets that derive from inletOutlet in OpenFOAM and recompute their refValue every
// updateCoeffs. They differ from a plain inletOutlet in exactly that: the flux switch is the base's, the
// inflow VALUE is not read from the case file but computed from the current U (for k) or the current k
// (for epsilon/omega). Reading the file's `value` and keeping it is what a solver does when it never calls
// updateCoeffs, and it leaves the inlet frozen at whatever state the case was written in.
//   turbulentIntensityKineticEnergyInletFvPatchScalarField.C:      refValue = 1.5*sqr(intensity)*magSqr(Up)
//   turbulentMixingLengthDissipationRateInletFvPatchScalarField.C: refValue = (Cmu^0.75/L)*pow(kp, 1.5)
//   turbulentMixingLengthFrequencyInletFvPatchScalarField.C:       refValue = sqrt(kp)/(Cmu^0.25*L)
template <typename T>
class TurbulentInletPatchField : public InletOutletPatchField<T>
{
public:
    enum Kind { intensityK, mixingLengthEpsilon, mixingLengthOmega };

    TurbulentInletPatchField(
        const FvPatch& p,
        bool           uniform,
        T              uval,
        std::vector<T> vals,
        Kind           kind,
        scalar         coefficient)
        : InletOutletPatchField<T>(p, uniform, uval, std::move(vals)),
          kind_(kind),
          coefficient_(coefficient)
    {}

    void updateTurbulentInlet(
        const std::vector<vector>& Up,
        const std::vector<scalar>& kp,
        scalar                     Cmu) override
    {
        if constexpr (std::is_same<T, scalar>::value)
        {
            const bool fromU = (kind_ == intensityK);
            const std::size_t have = fromU ? Up.size() : kp.size();
            // REFUSE rather than fill what is missing with zeros. A short source array here would leave
            // part of the inlet at refValue = 0 -- a silent no-turbulence inlet, which is exactly the
            // kind of substitution that took four measurements to find the last time it happened.
            if (have < static_cast<std::size_t>(this->patch_.size))
            {
                throw std::runtime_error(
                    std::string("turbulent inlet on patch '") + this->patch_.name
                    + "': updateCoeffs needs " + (fromU ? "U" : "k")
                    + " on every face of the patch, and was given fewer. Refusing rather than "
                      "computing an inlet value from a field that is not there.");
            }
            std::vector<scalar> r(this->patch_.size, scalar{0});
            for (label i = 0; i < this->patch_.size; ++i)
            {
                if (fromU)
                {
                    r[i] = 1.5 * (coefficient_ * coefficient_) * magSqr(Up[i]);
                }
                else if (kind_ == mixingLengthEpsilon)
                {
                    r[i] = (std::pow(Cmu, 0.75) / coefficient_) * std::pow(kp[i], 1.5);
                }
                else
                {
                    r[i] = std::sqrt(kp[i]) / (std::pow(Cmu, 0.25) * coefficient_);
                }
            }
            this->setRefValues(std::move(r));
        }
        else
        {
            (void)Up;
            (void)kp;
            (void)Cmu;
        }
    }

private:
    Kind   kind_;
    scalar coefficient_;      // intensity, or mixingLength
};


// outletInlet (freestreamPressure base): the flux-OPPOSITE of inletOutlet, outflow phi>=0 -> fixedValue(outletValue),
// inflow phi<0 -> zeroGradient. Device category 4 (oioMask). value() = outletValue (= refValue).
template <typename T>
class OutletInletPatchField : public MixedPatchField<T>   // value()/refValue = outletValue (= freestreamValue)
{
public:
    // FALSE, and unlike inletOutlet this one really does inherit it: outletInletFvPatchField declares no
    // assignable() of its own, so mixedFvPatchField's false stands (mixedFvPatchField.H:200).
    bool assignable() const override { return false; }   // OF: outletInlet does NOT override mixed
    OutletInletPatchField(
        const FvPatch& p,
        bool uniform,
        T uval,
        std::vector<T> vals)
        : MixedPatchField<T>(p, uniform, uval, std::move(vals), /*velocitySign=*/true)
    {
        // SEED THE VALUE FRACTION AT ZERO, not at the mixed base's 0.5. That 0.5 is the FREESTREAM
        // seed -- a half-and-half flow-angle blend waiting for updateMixedFreestream -- and it is
        // meaningless for a flux-conditional patch, which is either fully zeroGradient or fully
        // fixedValue per face and never in between. Zero is the outflow branch, which is what this
        // patch behaved as before it became mixed, so a field whose flux has not been pushed in yet is
        // unchanged rather than half-fixed. Getting this wrong reports valueInternalCoeffs = 0.5 where
        // OpenFOAM has 1, which measured 2.8e-01 on the momentum internalCoeffs.
        this->vf_.assign(p.size, 0.0);
    }

    // outletInletFvPatchField::updateCoeffs -- valueFraction = pos0(phi), the exact opposite switch.
    void updateFromFlux(const std::vector<scalar>& phip) override
    {
        std::vector<scalar> vf(this->patch_.size, 0.0);
        for (label i = 0; i < this->patch_.size && i < (label)phip.size(); ++i)
            vf[i] = (phip[i] >= 0.0) ? 1.0 : 0.0;
        this->setValueFraction(std::move(vf));
    }
    int bcCategory() const override { return 4; }                  // outletInlet (device: per-face fixedValue|zeroGradient, opposite switch)
};

// pressureInletOutletVelocity (OF directionMixedFvPatchVectorField): an outlet that allows backflow. Per face by the
// flux sign (valueFraction = neg(phi)*(I - n n); value = vf&refValue + (I-vf)&pif): outflow (phi>=0) -> zeroGradient
// (full extrapolation); inflow (phi<0) -> the TANGENTIAL velocity is fixed to refValue (default 0) while the NORMAL
// component is zeroGradient, so the pressure sets the inflow speed (value = n*(n.U_cell)). The DEVICE recomputes the
// per-face inflow value each step (deviceUpdatePressureInletOutletVelocity); bcCategory()=6 marks it. Vector-only;
// a non-zero `tangentialVelocity` field is NOT supported and is now REFUSED at construction (it was
// only a comment before, so a case carrying one ran with the tangential component silently zeroed).
template <typename T>
class PressureInletOutletVelocityPatchField : public ExtrapolatedValuePatchField<T>   // value() = written seed
{
public:
    bool assignable() const override { return false; }   // OF: pressureInletOutletVelocity derives from directionMixed
    using ExtrapolatedValuePatchField<T>::ExtrapolatedValuePatchField;
    int bcCategory() const override { return 6; }                  // device: pressureInletOutletVelocity (outlet, adjustable flux)
};

// processor: rank<->rank coupled patch. The exchange receives the NEIGHBOUR cells' values (the
// halo, = patchNeighbourField, used by the matrix). If interpolation weights are set, value() is the
// COUPLED FACE value w*ownCell + (1-w)*neighbour (OpenFOAM coupledFvPatchField::evaluate), so the
// serial fvc operators (grad/flux/interpolate) treat a processor face like an internal face. Without
// weights, value() == the halo (the original behaviour). Mirrors OpenFOAM processor/coupled patch.
template <typename T>
class ProcessorFvPatchField : public fvPatchField<T>
{
public:
    ProcessorFvPatchField(
        const FvPatch& p,
        int neighbProcNo,
        int tag = 0)
        : fvPatchField<T>(p), neighbProcNo_(neighbProcNo), tag_(tag) {}

    void setWeights(std::vector<scalar> w) { weights_ = std::move(w); }

    // A processor patch contributes NOTHING to the matrix as a boundary, its coupling is added
    // as an interface (off-diagonal) separately. So all matrix coeffs are zero (overriding the
    // base valueInternalCoeffs default of 1), letting the serial fvm operators run on the local
    // mesh and produce the correct REAL-boundary coeffs while ignoring processor faces.
    std::vector<T> valueInternalCoeffs()    const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> valueBoundaryCoeffs()    const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> gradientInternalCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> gradientBoundaryCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); }

    void initEvaluate(const std::vector<T>& internal) override
    {
        sendBuf_ = this->patchInternalField(internal);             // own interface cells
        halo_.resize(sendBuf_.size());                             // receive neighbour cells into halo_
        const int ns = static_cast<int>(sendBuf_.size() * (sizeof(T) / sizeof(scalar)));
        Pstream::irecv(reinterpret_cast<scalar*>(halo_.data()), ns, neighbProcNo_, tag_);
        Pstream::isend(reinterpret_cast<const scalar*>(sendBuf_.data()), ns, neighbProcNo_, tag_);
    }
    void evaluate(const std::vector<T>& internal) override
    {
        if (weights_.empty())   // value() = halo
        {
            this->value_ = halo_;
            return;
        }
        const std::vector<T> pif = this->patchInternalField(internal);
        this->value_.resize(halo_.size());
        for (std::size_t i = 0; i < halo_.size(); ++i)
            this->value_[i] = weights_[i] * pif[i] + (1.0 - weights_[i]) * halo_[i];  // coupled face value
    }
    bool fixesValue() const override { return false; }

    const std::vector<T>& patchNeighbourField() const override { return halo_; }

private:
    int                 neighbProcNo_;
    int                 tag_;
    std::vector<T>      sendBuf_;
    std::vector<T>      halo_;
    std::vector<scalar> weights_;
};

// cyclic: a periodic coupled patch. Like the processor patch but the neighbour cells are LOCAL (the
// owner cells of the matched neighbour faces, gathered from `internal`, no MPI). value() is the coupled
// face value w*ownCell + (1-w)*neighbour (with the transform applied to the neighbour for rotational
// cyclic), so the serial fvc operators treat a cyclic face like an internal face. The matrix coupling
// is added as an interface (off-diagonal), so all boundary matrix coeffs are zero. Mirrors OpenFOAM
// cyclicFvPatchField. forwardT = identity for translational periodicity.
template <typename T>
class CyclicFvPatchField : public fvPatchField<T>
{
public:
    CyclicFvPatchField(
        const FvPatch& p,
        std::vector<label> nbrFaceCells,
        std::vector<scalar> weights,
        tensor forwardT = tUniform<tensor>(0))
        : fvPatchField<T>(p), nbrFaceCells_(std::move(nbrFaceCells)), weights_(std::move(weights)),
          forwardT_(forwardT), hasTransform_(false) {}

    // Mark a rotational transform (forwardT applied to vector/tensor neighbour values).
    void setTransform(const tensor& fT)
    {
        forwardT_ = fT;
        hasTransform_ = true;
    }

    std::vector<T> valueInternalCoeffs()    const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> valueBoundaryCoeffs()    const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> gradientInternalCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); }
    std::vector<T> gradientBoundaryCoeffs() const override { return std::vector<T>(this->patch_.size, T{}); }

    void evaluate(const std::vector<T>& internal) override
    {
        const std::vector<T> pif = this->patchInternalField(internal);     // own cells
        halo_.resize(nbrFaceCells_.size());
        for (std::size_t i = 0; i < nbrFaceCells_.size(); ++i)
            halo_[i] = transformValue(internal[nbrFaceCells_[i]]);
        this->value_.resize(halo_.size());
        for (std::size_t i = 0; i < halo_.size(); ++i)
            this->value_[i] = weights_[i] * pif[i] + (1.0 - weights_[i]) * halo_[i];   // coupled face value
    }
    bool fixesValue() const override { return false; }

    const std::vector<T>& patchNeighbourField() const override { return halo_; }

private:
    // Apply the cyclic transform to a neighbour value (identity unless rotational; scalars never rotate).
    T transformValue(const T& v) const { return v; }

    std::vector<label>  nbrFaceCells_;
    std::vector<scalar> weights_;
    std::vector<T>      halo_;       // transformed neighbour cell values (= patchNeighbourField)
    tensor              forwardT_;
    bool                hasTransform_;
};

// Rotational transform specialisation for vectors (forwardT & v). Scalars/tensors handled inline above
// (scalar: no rotation; tensor: forwardT & v & forwardT^T, added with C4).
template <> inline vector CyclicFvPatchField<vector>::transformValue(const vector& v) const
{
    return hasTransform_ ? dot(v, transpose(forwardT_)) : v;   // transform(forwardT, v) = forwardT & v = v & forwardT^T
}

// inletOutlet / totalPressure take their operating value from the inletValue slot, falling back to the plain `value`
// when inletValue is omitted. One accessor for that choice (returns a view; the referenced vector outlives the call).
template <typename T>
struct InletOrValue
{
    bool uniform;
    T uniformValue;
    const std::vector<T>& values;
};
template <typename T>
inline InletOrValue<T> inletOrValue(const PatchFieldData<T>& d)
{
    return d.hasInletValue ? InletOrValue<T>{d.inletUniform, d.inletUniformValue, d.inletValues}
                           : InletOrValue<T>{d.valueUniform, d.uniformValue,      d.values};
}

template <typename T>
std::unique_ptr<fvPatchField<T>> makePatchField(const FvPatch& p, const PatchFieldData<T>& d)
{
    // THE ENERGY BOUNDARY FAMILY -- fixedEnergy, gradientEnergy, mixedEnergy -- handled by the three
    // branches they derive from. This is NOT a substitution. OpenFOAM's energy patch fields ARE their
    // base classes as far as the matrix is concerned:
    //     fixedEnergyFvPatchScalarField    : public fixedValueFvPatchScalarField
    //     gradientEnergyFvPatchScalarField : public fixedGradientFvPatchScalarField
    //     mixedEnergyFvPatchScalarField    : public mixedFvPatchScalarField
    // What each adds is an updateCoeffs() that recomputes its value, gradient or refValue from T through
    // the thermo -- so the COEFFICIENTS are the base class's, unchanged, and the numbers those coeffs are
    // built from are written to the file. OpenFOAM writes exactly the entries each base needs: `value`
    // for fixedEnergy, `gradient` for gradientEnergy, and refValue/refGradient/valueFraction for
    // mixedEnergy. Reading them is reading what OpenFOAM used.
    //
    // The solver does not reach this path: it derives he's boundary conditions from T's, which is the
    // same relation from the other side. This exists so a gate can read the `he` OpenFOAM WROTE --
    // test_rho_eeqn_cpp refused angledDuct with `unsupported BC type 'fixedEnergy'` and could not run on
    // any case whose inlet fixes a temperature, which is most of them.
    if (d.type == "fixedGradient" || d.type == "gradientEnergy")
    {
        if (!d.hasGradient)
            throw std::runtime_error("brae: patch " + p.name + " is fixedGradient but has no 'gradient' entry.");
        return std::make_unique<FixedGradientPatchField<T>>(p, d.gradientUniform, d.gradientUniformValue, d.gradientValues);
    }
    if (d.type == "pressureInletOutletVelocity" && d.hasTangentialVelocity)
    {
        // OF: refValue = tangentialVelocity - n*(n & tangentialVelocity), i.e. the tangential component
        // is DRIVEN, not free. brae's piov kernel sets the tangential refValue to zero, so running this
        // would silently solve a different boundary condition (a swirl-free inlet where the case asked
        // for swirl). The header claimed this was unsupported; nothing enforced it until now.
        throw std::runtime_error(
            "brae: patch " + p.name + " is pressureInletOutletVelocity with a `tangentialVelocity` entry, "
            "which brae does not apply -- it would silently run with zero tangential velocity. Remove the "
            "entry (if the tangential component really is zero) or use a BC that fixes the full vector.");
    }
    // `table` is implemented (Function1::table + the solver's per-step p0 refresh), so it is not in
    // unsupportedFunction1 at all. Anything still recorded there -- polynomial, csvFile, expression --
    // is genuinely unevaluated and must stay named rather than silently frozen at its t=0 value.
    if ((d.type == "uniformTotalPressure" || d.type == "totalPressure") && !d.unsupportedFunction1.empty())
    {
        // Same rule as uniformFixedValue below: a time-varying p0 that brae cannot evaluate must be
        // named, not silently replaced by whatever `value` happens to hold -- that would run a
        // constant-pressure outlet where the case asked for a ramp.
        throw std::runtime_error(
            "brae: patch " + p.name + " is " + d.type + " with a non-constant p0 ('"
            + d.unsupportedFunction1 + "'). brae evaluates only `constant`/`uniform` Function1 entries, "
            "so the prescribed time variation would be lost. Replace p0 with a constant, or drive the "
            "case at fixed total pressure.");
    }
    if (d.hasABL && !d.unsupportedFunction1.empty())
    {
        // A time-varying atmBoundaryLayer entry (a wind rose, a ramped Uref). Reading past it would leave
        // brae on the DEFAULT for that parameter -- a silently different inlet profile, not a failure.
        throw std::runtime_error(
            "brae: patch " + p.name + " has " + d.type + " with a non-constant Function1 entry ('"
            + d.unsupportedFunction1 + "'). brae evaluates only `constant`/`uniform` Function1 entries, "
            "so the prescribed variation would be lost. Replace it with a constant.");
    }
    if (d.type == "uniformFixedValue" && !d.unsupportedFunction1.empty())
    {
        // A non-constant uniformValue. Refusing rather than falling back to whatever `value` happens to
        // be present, which would look like a converged run at the wrong boundary value.
        throw std::runtime_error(
            "brae: patch " + p.name + " has uniformFixedValue with a non-constant uniformValue ('"
            + d.unsupportedFunction1 + "'). brae evaluates only `constant`/`uniform` Function1 entries. "
            "Running would silently substitute the patch's `value` entry, so it is refused. Replace it "
            "with a constant, or use codedFixedValue.");
    }
    // fixedMean: a fixedValue whose face values are the adjacent cell values, shifted or scaled so their
    // area-weighted mean equals `meanValue` (OF fixedMeanFvPatchField). Seeded from `value`; the solver
    // recomputes refValue every step, exactly as it does for codedFixedValue and fanPressure.
    if (d.type == "fixedMean")
        return std::make_unique<FixedValuePatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    if (d.type == "fixedValue" || d.type == "uniformFixedValue" || d.type == "codedFixedValue"
     || d.type == "fixedEnergy")   // fixedEnergy: fixedValue on he -- see the energy-family note above
        // uniformFixedValue: steady constant = fixedValue. codedFixedValue: seed with `value`; the NVRTC device kernel
        // (device_coded_bc) OVERWRITES this patch's refValue each step from the compiled `code` snippet (the driver
        // parses the code + wires the solver's coded-BC apply). Building it as fixedValue makes buildField/buildDeviceBoundary succeed.
        return std::make_unique<FixedValuePatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    if (d.type == "codedMixed")   // Robin: seeded as mixed (vf=0.5); the NVRTC device kernel overwrites refValue + valueFraction each step
        return std::make_unique<MixedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values, false);
    if (d.type == "noSlip")          return std::make_unique<NoSlipPatchField<T>>(p);
    if (d.type == "zeroGradient")    return std::make_unique<ZeroGradientPatchField<T>>(p);
    // fixedFluxPressure = fixedGradient p whose gradient constrainPressure sets to (phiHbyA - Sf&U)/(magSf*rAU).
    // At a fixed-velocity patch constrainHbyA gives phiHbyA_b = Sf&U, so that gradient is exactly 0 == zeroGradient.
    // That is its only standard usage (no-flux walls), so we map it to zeroGradient (exact there). See
    // PORTING_PRESSURE_REFERENCE.md; a fixedFluxPressure on a non-fixed-velocity patch would need the gradient path.
    if (d.type == "fixedFluxPressure") return std::make_unique<ZeroGradientPatchField<T>>(p);
    // uniformTotalPressure is totalPressure with a TIME-VARYING p0 -- OF's own class differs only in
    // that p0 comes from a Function1 (uniformTotalPressureFvPatchScalarField.C:149 samples it every
    // updateCoeffs). The face treatment is identical: p_b = p0 - 0.5*neg(phi)*magSqr(U). So it builds
    // the same patch field, and the solver refreshes p0 per step from the table.
    // fanPressure is totalPressure with p0 shifted by a fan curve: OF's updateCoeffs computes the patch
    // volumetric flow rate, looks up the pressure rise, and calls totalPressure::updateCoeffs(p0 - dir*pdFan)
    // (fanPressureFvPatchScalarField.C). The FACE treatment is identical, so it builds the same patch field
    // seeded with p0, and the solver shifts p0 per step. Same split as uniformTotalPressure above.
    if (d.type == "totalPressure" || d.type == "uniformTotalPressure" || d.type == "fanPressure")   // p0 read into the inletValue slot
    {
        // OF has three branches. brae implements the two that share one expression (kinematic, and the
        // compressible low-speed form with rho). The isentropic branch a named psi selects is different
        // physics, so it is refused rather than approximated -- it would converge and be wrong at the inlet.
        if (d.psiName != "none")
        {
            throw std::runtime_error(
                "brae: totalPressure with psi '" + d.psiName + "' on patch " + p.name +
                " selects OpenFOAM's isentropic high-speed branch (p0/(1+0.5*psi*gM1ByG*|U|^2)^(1/gM1ByG)), "
                "which is not implemented. Remove the psi entry to use the low-speed form, which brae "
                "reproduces exactly.");
        }
        const auto v = inletOrValue(d);
        return std::make_unique<TotalPressurePatchField<T>>(p, v.uniform, v.uniformValue, v.values);
    }
    if (d.type == "flowRateInletVelocity")
    {
        if constexpr (std::is_same_v<T, vector>)
        {
            if (!d.hasFlowRate)
                throw std::runtime_error("brae: flowRateInletVelocity on patch " + p.name +
                    " has neither 'volumetricFlowRate' nor 'massFlowRate' (OF fails the same way).");
            if (d.extrapolateProfile)
                throw std::runtime_error("brae: flowRateInletVelocity 'extrapolateProfile true' on patch " +
                    p.name + " is not implemented (it rescales the extrapolated internal profile rather "
                    "than applying a uniform normal velocity). Remove it to use the uniform form.");
            return std::make_unique<FlowRateInletVelocityPatchField>(
                p, d.flowRate, d.flowRateIsMass, d.rhoInlet, d.valueUniform, d.uniformValue, d.values);
        }
        else throw std::runtime_error("brae: flowRateInletVelocity is a velocity (vector) BC");
    }
    if (d.type == "surfaceNormalFixedValue" || d.type == "uniformNormalFixedValue")   // U_b = refValue(scalar) * n
    {
        if constexpr (std::is_same_v<T, vector>)
            return std::make_unique<SurfaceNormalFixedValuePatchField>(p, d.normalRefUniform, d.normalRefUniformValue, d.normalRefValues);
        else throw std::runtime_error("brae: surfaceNormalFixedValue/uniformNormalFixedValue is a velocity (vector) BC");
    }
    if (d.type == "timeVaryingMappedFixedValue")   // boundaryData profile mapped (nearest) onto the faces -> fixedValue
    {
        if (d.hasMapData) return std::make_unique<TimeVaryingMappedPatchField<T>>(p, d.mapPoints, d.mapValues);
        // No mapped data: OF would FATAL here (the `value` entry is only the initial field, not a substitute for
        // boundaryData). Throw rather than silently degrade to fixedValue/zeroGradient -- masks a misconfigured case.
        throw std::runtime_error("brae: timeVaryingMappedFixedValue on patch '" + p.name +
            "' has no boundaryData (constant/boundaryData/" + p.name + "); OF requires it -- not falling back silently");
    }
    if (d.type == "inletOutlet")   // refValue = inletValue (fall back to value if inletValue omitted)
    {
        const auto v = inletOrValue(d);
        return std::make_unique<InletOutletPatchField<T>>(p, v.uniform, v.uniformValue, v.values);
    }
    // outletInlet: the same mixed BC with the flux test inverted -- outflow (phi >= 0) takes the
    // prescribed outletValue, inflow extrapolates. The class and its device category existed already;
    // nothing constructed it, so a case naming it was refused. (pimpleFoam/LES/NACA4412 uses it.)
    if (d.type == "outletInlet")
    {
        const auto v = inletOrValue(d);   // outletValue is read into the same slot
        return std::make_unique<OutletInletPatchField<T>>(p, v.uniform, v.uniformValue, v.values);
    }
    // turbulent-inlet BCs (inletOutlet-derived): inflow value computed from U/k by the solver (applyTurbulentInlets);
    // here we build an inletOutlet placeholder from the written `value` so buildField succeeds.
    if (d.type == "turbulentIntensityKineticEnergyInlet" || d.type == "turbulentMixingLengthDissipationRateInlet"
        || d.type == "turbulentMixingLengthFrequencyInlet")
    {
        // The file's `value` is the SEED only. OF recomputes refValue in updateCoeffs from the current U
        // or k, so the patch carries the rule and its coefficient rather than the written number.
        const bool ki = (d.type == "turbulentIntensityKineticEnergyInlet");
        const auto kind = ki ? TurbulentInletPatchField<T>::intensityK
                             : (d.type == "turbulentMixingLengthDissipationRateInlet"
                                ? TurbulentInletPatchField<T>::mixingLengthEpsilon
                                : TurbulentInletPatchField<T>::mixingLengthOmega);
        return std::make_unique<TurbulentInletPatchField<T>>(
            p, d.valueUniform, d.uniformValue, d.values, kind, ki ? d.intensity : d.mixingLength);
    }
    // base `freestream` (e.g. k/omega) derives from inletOutlet in OF -> BINARY flux switch (kept).
    if (d.type == "freestream")
        return std::make_unique<InletOutletPatchField<T>>(p, d.inletUniform, d.inletUniformValue, d.inletValues);
    // freestreamVelocity / freestreamPressure derive from mixedFvPatchField in OF -> CONTINUOUS Robin blend
    // vf = 0.5 -/+ 0.5*(U.n)/|U| (flow-angle, not the binary switch). The device recomputes vf each step.
    // The file's own `value`, expanded per face. OF's mixed constructor keeps it verbatim on read.
    auto readValueOf = [&]() -> std::vector<T>
    {
        if (!d.hasValue) return {};
        if (d.valueUniform) return std::vector<T>(p.size, d.uniformValue);
        if (d.values.size() == static_cast<std::size_t>(p.size)) return d.values;
        return {};
    };
    if (d.type == "freestreamVelocity")   // mixed, velocity sign (0.5 - 0.5 U.n/|U|)
        return std::make_unique<MixedPatchField<T>>(p, d.inletUniform, d.inletUniformValue, d.inletValues,
                                                    true, /*freestream*/true, readValueOf());
    if (d.type == "freestreamPressure")   // mixed, pressure sign (0.5 + 0.5 U.n/|U|)
        return std::make_unique<MixedPatchField<T>>(p, d.inletUniform, d.inletUniformValue, d.inletValues,
                                                    false, /*freestream*/true, readValueOf());
    // Plain `mixed` (Robin): refValue + refGradient + valueFraction, all given by the case. Unlike
    // freestream*/inletOutlet the valueFraction is FIXED, not recomputed from the flux, so the device does
    // not need a per-step update -- the seeded vf is the answer. This is the shape basicThermo maps a
    // `mixed` T patch onto (-> mixedEnergy), which is how an external-convection wall is written.
    if (d.type == "mixed" || d.type == "mixedEnergy")
    {
        if (!d.hasValueFraction)
            throw std::runtime_error(
                "brae: patch " + p.name + " is `mixed` but has no `valueFraction` entry. OF requires all "
                "three of refValue, refGradient and valueFraction; running with a guessed blend would "
                "silently solve a different boundary condition.");
        auto mp = std::make_unique<MixedPatchField<T>>(p, d.refValueUniform, d.refValueUniformValue,
                                                      d.refValues, /*velocitySign*/ true);
        {
            std::vector<scalar> f(static_cast<std::size_t>(p.size), d.vfUniform ? d.vfUniformValue : scalar(0));
            if (!d.vfUniform)
                for (std::size_t i = 0; i < f.size() && i < d.vfValues.size(); ++i) f[i] = d.vfValues[i];
            mp->setValueFraction(std::move(f));
        }
        if (d.hasGradient)
        {
            std::vector<T> g(static_cast<std::size_t>(p.size),
                             d.gradientUniform ? d.gradientUniformValue : T{});
            if (!d.gradientUniform)
                for (std::size_t i = 0; i < g.size() && i < d.gradientValues.size(); ++i) g[i] = d.gradientValues[i];
            mp->setRefGrad(std::move(g));
        }
        return mp;
    }
    // pressureInletOutletVelocity (directionMixed): device recomputes the inflow normal-projection each step (cat 6).
    if (d.type == "pressureInletOutletVelocity")
        return std::make_unique<PressureInletOutletVelocityPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    if (d.type == "kqRWallFunction")      return std::make_unique<ZeroGradientPatchField<T>>(p); // zeroGradient wrapper
    // boundary face = cell value; the near-wall constraint is applied by the turbulence model
    if (d.type == "epsilonWallFunction")  return std::make_unique<EpsilonWallFunctionPatchField<T>>(p);
    // omegaWallFunction: the same shape as epsilonWallFunction -- a zeroGradient boundary VALUE with the
    // near-wall cell constrained separately (deviceWallOmegaG0 + setValues). It must answer the wall-
    // function predicate for the same reason epsilon's does: the solver keys its wall treatment on the BC
    // TYPE, and mapping this to a bare ZeroGradientPatchField left every kOmegaSST case with NO wall
    // faces at all -- no wall nut, so the wall shear and everything downstream of it were wrong.
    if (d.type == "omegaWallFunction")    return std::make_unique<EpsilonWallFunctionPatchField<T>>(p);
    if (d.type == "nutkWallFunction")     return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // alphatWallFunction: alphat_w = rho_w*nut_w/Prt, i.e. the SAME expression deviceAlphat applies in the
    // cells. Nothing is prescribed at the patch, so it is `calculated` exactly like the nut wall functions
    // -- the model writes the value. Without this row a real OF compressible case is refused at load,
    // because every rhoSimpleFoam tutorial with turbulence ships a 0/alphat.
    // OF v2412 registers this as "compressible::alphatWallFunction"; the unqualified spelling is NOT a
    // valid OF type (OF errors out on it). Both are accepted here so a hand-written case is not rejected
    // for a spelling OF would have caught anyway, but the qualified form is the one real cases use.
    if (d.type == "compressible::alphatWallFunction"
     || d.type == "alphatWallFunction")   return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // alphatJayatillekeWallFunction is NOT the same condition: it adds a thermal-sublayer resistance
    // (the P-function) and gives a different wall heat flux. Accepting it as the simple form would run
    // and converge with the wrong wall heat transfer, so it is refused by name instead.
    if (d.type == "compressible::alphatJayatillekeWallFunction"
     || d.type == "alphatJayatillekeWallFunction")
    {
        throw std::runtime_error(
            "brae: alphatJayatillekeWallFunction is not implemented (patch " + p.name
            + "). It applies a thermal-sublayer P-function that brae does not have, so treating it as "
              "compressible::alphatWallFunction would give the wrong wall heat flux. Use that instead, or wait "
              "for the Jayatilleke port.");
    }
    if (d.type == "atmNutkWallFunction")  return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values); // atmospheric rough-wall nut (z0); device wall nut from deviceBoundaryNut with atmZ0>0
    // SA wall nut. Device: deviceBoundaryNutSpalding. Host: the model's correctNut recomputes it -- see
    // isNutUSpalding, and SpalartAllmaras_cpp.cu for what taking the field assignment there costs.
    if (d.type == "nutUSpaldingWallFunction")
        return std::make_unique<NutUSpaldingPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // nutLowReWallFunction: OF calcNut()=0 (resolved viscous sublayer). cf's nutkWallFunction already yields nut=0 for
    // yPlus<yPlusLam, so on the low-Re mesh this BC is used on it is the same; nutUBlendedWallFunction is the same
    // log-law wall-nut family as nutk. Both -> Calculated (device wall nut from deviceBoundaryNut). Validated vs OF.
    if (d.type == "nutUWallFunction" || d.type == "nutLowReWallFunction" || d.type == "nutUBlendedWallFunction")
        return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // cyclic: the device-resident solver couples it via appended internal faces (DeviceMesh), so the host patch
    // field is a no-op placeholder here (its value is unused; the FvPatch type "cyclic" drives the device skip).
    if (isCoupledInterfaceType(d.type))          return std::make_unique<ZeroGradientPatchField<T>>(p);
    if (d.type == "empty")           return std::make_unique<EmptyPatchField<T>>(p);
    if (d.type == "symmetryPlane" || d.type == "symmetry" || d.type == "slip")
        return std::make_unique<SymmetryPlanePatchField<T>>(p);  // slip = OF basicSymmetry
    if (d.type == "wedge")   // axisymmetric constraint: the geometry IS the boundary condition
    {
        const WedgeGeometry w = wedgeGeometry(p);
        return std::make_unique<WedgePatchField<T>>(p, w.faceT, w.cellT);
    }
    // cellMotion (OF cellMotionFvPatchField): the CELL-motion counterpart of a prescribed point motion.
    // It holds the value the motion solver put there, so as a boundary condition it is a fixedValue --
    // which is what OF's cellMotionBoundaryTypes maps a fixed point motion to. Appears in a written
    // cellMotionU<Cmpt> field, which a restart reads back.
    if (d.type == "cellMotion")      return std::make_unique<FixedValuePatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    if (d.type == "calculated")      return std::make_unique<CalculatedPatchField<T>>(p, d.valueUniform, d.uniformValue, d.values);
    // atmBoundaryLayerInlet{Velocity,K,Epsilon,Omega} (OpenFOAM atmBoundaryLayer.C, v2412): Richards-Hoxey log-law
    // profile evaluated per boundary face from the face-centre height z = Cf.zDir - d, a steady fixedValue. Params come
    // from include/ABLConditions (foam_field_reader parses them). u* = kappa|Uref|/ln((Zref+z0)/z0):
    //   U(z)=(u*/kappa)ln((z+z0)/z0)flowDir ; k=u*^2/sqrt(Cmu) ; eps=u*^3/(kappa(z+z0)) ; omega=u*/(sqrt(Cmu)kappa(z+z0)).
    // OF defaults C1=0,C2=1 -> k is height-constant. z clamped >=0 at/below the ground plane (OF max(z-d,0)).
    if (d.type == "atmBoundaryLayerInletVelocity")
    {
        if constexpr (std::is_same_v<T, vector>)
        {
            const scalar kap = d.ablKappa, z0 = d.ablZ0;
            const scalar Ustar = kap * std::fabs(d.ablUref) / std::log((d.ablZref + z0) / z0);
            vector fh = d.ablFlowDir;
            const scalar fm = std::sqrt(fh.x*fh.x + fh.y*fh.y + fh.z*fh.z);
            if (fm > 0)
            {
                fh.x /= fm;
                fh.y /= fm;
                fh.z /= fm;
            }
            const vector zh = d.ablZDir;
            // OF measures the profile from the PATCH's own lowest point, not from z = 0:
            //   Un = (Ustar/kappa)*log(((zDir & Cf) - groundMin - d + z0)/z0),  groundMin = zDir & ppMin
            // (atmBoundaryLayer.C:218-224). turbineSiting's terrain sits near z = 1000 m, so leaving
            // groundMin out replaces the boundary layer with the logarithm of the altitude -- a ~40%
            // error in the inlet velocity that still converges to a plausible-looking wind field.
            const scalar groundMin = p.ppMin.x*zh.x + p.ppMin.y*zh.y + p.ppMin.z*zh.z;
            std::vector<vector> vals(p.size);
            for (label i = 0; i < p.size; ++i)
            {
                const vector& c = p.Cf[i];
                scalar zr = c.x*zh.x + c.y*zh.y + c.z*zh.z - groundMin - d.ablD;
                if (zr < 0) zr = 0;
                const scalar Um = (Ustar / kap) * std::log((zr + z0) / z0);
                vals[i] = vector{fh.x*Um, fh.y*Um, fh.z*Um};
            }
            return std::make_unique<FixedValuePatchField<vector>>(p, false, vector{}, vals);
        }
        else throw std::runtime_error("brae: atmBoundaryLayerInletVelocity is a velocity (vector) BC");
    }
    if (d.type == "atmBoundaryLayerInletK" || d.type == "atmBoundaryLayerInletEpsilon" || d.type == "atmBoundaryLayerInletOmega")
    {
        if constexpr (std::is_same_v<T, scalar>)
        {
            const scalar kap = d.ablKappa, z0 = d.ablZ0, Cmu = d.ablCmu;
            const scalar Ustar = kap * std::fabs(d.ablUref) / std::log((d.ablZref + z0) / z0);
            const vector zh = d.ablZDir;
            const bool isK = (d.type == "atmBoundaryLayerInletK"), isEps = (d.type == "atmBoundaryLayerInletEpsilon");
            // Same origin as the velocity above: zDir & boundBox(patch points).min().
            const scalar groundMin = p.ppMin.x*zh.x + p.ppMin.y*zh.y + p.ppMin.z*zh.z;
            std::vector<scalar> vals(p.size);
            for (label i = 0; i < p.size; ++i)
            {
                const vector& c = p.Cf[i];
                scalar zr = c.x*zh.x + c.y*zh.y + c.z*zh.z - groundMin - d.ablD;
                if (zr < 0) zr = 0;
                vals[i] = isK   ? Ustar*Ustar / std::sqrt(Cmu)
                        : isEps ? Ustar*Ustar*Ustar / (kap * (zr + z0))
                                : Ustar / (std::sqrt(Cmu) * kap * (zr + z0));
            }
            return std::make_unique<FixedValuePatchField<scalar>>(p, false, scalar{}, vals);
        }
        else throw std::runtime_error("brae: atmBoundaryLayerInlet{K,Epsilon,Omega} is a scalar BC");
    }
    // codedFixedValue is handled above (seeded as fixedValue; the NVRTC device kernel overwrites its refValue per step).
    if (d.type == "externalWallHeatFluxTemperature" || d.type == "compressible::externalWallHeatFluxTemperature")
    {
        // The other half of B5, and deliberately still a refusal. In `flux`/`power` mode this BC is
        // exactly a fixedGradient with g = q/kappa -- which brae can now express -- but kappa is
        // kappaEff, i.e. Cp*(alpha + alphat), so it varies per face AND changes every outer iteration as
        // the turbulence develops. Freezing it at the initial value is not a mild approximation: alphat
        // at a heated turbulent wall grows by orders of magnitude, so the imposed flux would drift far
        // from the requested q while the run still converged and still looked physical. Refuse, and say
        // what to substitute.
        throw std::runtime_error(
            "brae: patch " + p.name + " uses externalWallHeatFluxTemperature, which brae does not "
            "implement: its gradient is q/kappaEff and kappaEff = Cp*(alpha + alphat) changes every "
            "iteration, so a frozen value would impose a flux the case never asked for. If kappa is "
            "effectively constant for your case, compute g = q/kappa yourself and use "
            "`type fixedGradient; gradient uniform <g>;` -- brae discretises that exactly (see B5).");
    }
    // movingWallVelocity -- OF movingWallVelocityFvPatchVectorField. It IS a fixedValue: updateCoeffs()
    // does `vectorField::operator=(Uwall())` and then defers to fixedValueFvPatchVectorField, and on a
    // STATIC mesh (`if (mesh.moving())` is false) it never assigns at all and is exactly the `value`
    // entry. So the patch field is a plain fixedValue here; the solver overwrites its refValue each
    // step from movingWallVelocity() when the mesh moves, which is the same split OF uses.
    //
    // A case that declares it WITHOUT a moving mesh therefore behaves correctly by construction: no
    // move, no overwrite, the value entry stands -- which is what OF does.
    if (d.type == "movingWallVelocity")
    {
        const auto v = inletOrValue(d);
        return std::make_unique<FixedValuePatchField<T>>(p, v.uniform, v.uniformValue, v.values);
    }
    throw std::runtime_error("brae: unsupported BC type '" + d.type + "' on patch " + p.name);
}

// OF freestreamVelocity::updateCoeffs / freestreamPressure::updateCoeffs -- the host form.
//
//   velocity: valueFraction = 0.5 - 0.5*(Up & nf)/mag(Up)
//   pressure: valueFraction = 0.5 + 0.5*(Up & nf)/mag(Up)
//
// `Up` is the patch's CURRENT velocity, i.e. whatever the previous evaluate left there, so calling this
// before the field is re-evaluated reproduces OpenFOAM's own one-iteration lag. A face with no flow
// (mag(Up) == 0) has no flow angle to blend on and keeps 0.5, which is what the seed already is.
//
// Only patches that ARE mixed are touched; everything else is left alone.
template <typename T>
void updateMixedFreestream(
    std::vector<std::unique_ptr<fvPatchField<T>>>& boundary,
    const std::vector<std::vector<vector>>&        Ubnd,
    const std::vector<FvPatch>&                    patches)
{
    for (std::size_t pi = 0; pi < patches.size() && pi < boundary.size(); ++pi)
    {
        MixedPatchField<T>* mp = dynamic_cast<MixedPatchField<T>*>(boundary[pi].get());
        if (!mp) continue;
        // ONLY the freestream family, which is bcCategory 5. This used to be implied by the dynamic_cast
        // -- freestreamVelocity/freestreamPressure were the only mixed patches brae had -- and stopped
        // being implied when inletOutlet and outletInlet were reparented onto MixedPatchField to give
        // them the Robin coefficients. They are governed by the FLUX SIGN, a binary switch, not by the
        // continuous flow-angle blend below; applying this to them overwrites the valueFraction their own
        // updateFromFlux just set. Measured on sbMatched, whose outlet is inletOutlet: rAU 1.24e-02 and
        // the momentum internalCoeffs 2.76e-01 against OpenFOAM.
        if (mp->bcCategory() != 5) continue;
        if (pi >= Ubnd.size() || Ubnd[pi].size() != static_cast<std::size_t>(patches[pi].size)) continue;

        const scalar sign = mp->mixedVelocitySign() ? -1.0 : 1.0;
        std::vector<scalar> vf(patches[pi].size, 0.5);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const vector& Up = Ubnd[pi][i];
            const scalar mu = mag(Up);
            if (mu > 0.0) vf[i] = 0.5 + sign * 0.5 * dot(Up, patches[pi].nf[i]) / mu;
        }
        mp->setValueFraction(std::move(vf));
    }
}

} // namespace brae
