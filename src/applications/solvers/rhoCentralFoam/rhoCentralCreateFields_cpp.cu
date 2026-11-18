// rhoCentralCreateFields_cpp.cu -- createFields.H, createFieldRefs.H and the thermo's construction. See the header.
#include "rhoCentralCreateFields_cpp.cuh"
#include "rhoCentralThermo.cuh"

#include "foam_field_reader.cuh"
#include "geometric_field.cuh"

#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace rhoCentral {

using namespace brae::rhoCentral;

// The nozzle boundaries the shared patch-field factory does not build (milestone H): totalTemperature,
// waveTransmissive and totalPressure's isentropic (psi) form start as the fixedValue of their written `value` (OpenFOAM's dictionary constructors read it;
// advective's mixed starts with valueFraction 0 and refValue = value, the same patch value). The CUDA arm recomputes
// them each step where OpenFOAM's updateCoeffs runs (device_boundary.cuh: deviceUpdateTotalTemperature, and
// deviceUpdateWaveTransmissive on a mixed face). The host reference arm refuses them.
static FieldData<scalar> asBuilt(FieldData<scalar> fd)
{
    for (PatchFieldData<scalar>& e : fd.boundary)
    {
        // totalPressure, both forms rhoCentralFoam runs (the psi form, milestone H; the rho form, milestone K): its
        // dictionary constructor takes `value` when written, else p0 (totalPressureFvPatchScalarField.C)
        if (e.type == "totalPressure" && !e.hasValue)
        {
            e.hasValue = true;
            e.valueUniform = e.inletUniform;
            e.uniformValue = e.inletUniformValue;
            e.values = e.inletValues;
        }
        if (e.type == "totalTemperature" || e.type == "waveTransmissive" || e.type == "smoluchowskiJumpT"
            || e.type == "totalPressure")
            e.type = "fixedValue";
    }
    return fd;
}
// maxwellSlipU (milestone I) starts as its written `value` too (partialSlip's dictionary constructor reads it, and
// refValue/valueFraction beside it); the CUDA arm evaluates it as partialSlip from the step's first U.correctBC on.
// smoluchowskiJumpT above: a mixed patch, fixesValue(), built from its value.
static FieldData<vector> asBuiltU(FieldData<vector> fd)
{
    for (PatchFieldData<vector>& e : fd.boundary)
        if (e.type == "maxwellSlipU") e.type = "fixedValue";
    return fd;
}

State createFields(const Case& cs, PatchFields& pf, int control)
{
    const std::string dir = cs.caseDir + "/" + cs.startName;
    const GasCoeffs& g = cs.gas;
    State s;

    // the thermo reads p and T (basicThermo.C), U is read by createFields.H; each patch is evaluated on construction
    pf.p = buildField(asBuilt(readField<scalar>(dir + "/p")), cs.patches, cs.nC);
    pf.T = buildField(asBuilt(readField<scalar>(dir + "/T")), cs.patches, cs.nC);
    pf.U = buildField(asBuiltU(readField<vector>(dir + "/U")), cs.patches, cs.nC);
    s.p = pf.p.internal;
    s.T = pf.T.internal;
    s.U = pf.U.internal;
    // createFields.H: compressible::turbulenceModel::New(rho, U, phi, thermo) -- the RAS model reads its fields here
    if (cs.ras)
        pf.ras = std::make_shared<brae::compressibleRAS::Case>(
            brae::compressibleRAS::readCase(cs.caseDir, dir, cs.m, cs.g, cs.patches, pf.U, cs.lts));
    evaluatePatches(pf.p, s.p, cs, s.pB);
    evaluatePatches(pf.T, s.T, cs, s.TB);
    evaluatePatches(pf.U, s.U, cs, s.UB);
    for (label k = 0; k < cs.nB; ++k)   // a transform face's construction-time evaluate, in OpenFOAM's arithmetic
        if (cs.bUTransform[k]) s.UB[k] = transformPatchU(cs.bUTransform[k], cs.bUR[k], s.U[cs.bCell[k]], controlNone);
    // An inletOutlet patch keeps the file's `value` (inletOutletFvPatchField's dictionary constructor; brae's class keeps
    // it until a flux has switched it). On a restart that value is the step before's evaluation -- on an outflow face of T,
    // the cell's T a step older than the cell's own. THE CONTROL restartIoExtrapolate takes the cells instead.
    if (control == controlRestartIoExtrapolate)
        for (label k = 0; k < cs.nB; ++k)
            if (cs.bInletOutlet[k])
            {
                const label c = cs.bCell[k];
                s.UB[k] = s.U[c];
                s.pB[k] = s.p[c];
                s.TB[k] = s.T[c];
            }

    // heThermo::init (heThermo.C:58-90): he from (p, T) on cells and patches. heBoundaryCorrection sets a
    // gradientEnergy patch's gradient to snGrad(he), which every later evaluate recomputes before use.
    s.e.resize(cs.nC);
    s.eB.resize(cs.nB);
    for (label c = 0; c < cs.nC; ++c) s.e[c] = gasEs(s.p[c], s.T[c], g);
    for (label k = 0; k < cs.nB; ++k) s.eB[k] = gasEs(s.pB[k], s.TB[k], g);

    // hePsiThermo's constructor: calculate(p, T, he, psi, mu, alpha) -- T is re-derived from he even here. On the
    // patches, a T that fixesValue() keeps T_b and sets he_b = HE(p_b, T_b); any other inverts T_b = THE(he_b, ...).
    s.psi.resize(cs.nC);
    s.psiB.resize(cs.nB);
    for (label c = 0; c < cs.nC; ++c)
    {
        const GasTResult r = gasTEs(s.e[c], s.p[c], s.T[c], g);
        if (r.status)
            throw std::runtime_error("brae rhoCentralFoam: createFields: the e -> T inversion failed in cell "
                                     + std::to_string(c) + ".");
        s.T[c]   = r.T;
        s.psi[c] = gasPsi(s.p[c], s.T[c], g);
    }
    for (label k = 0; k < cs.nB; ++k)
    {
        if (pf.T.boundary[cs.bPatch[k]]->fixesValue())
        {
            s.eB[k] = gasEs(s.pB[k], s.TB[k], g);
        }
        else
        {
            const GasTResult r = gasTEs(s.eB[k], s.pB[k], s.TB[k], g);
            if (r.status)
                throw std::runtime_error("brae rhoCentralFoam: createFields: the e -> T inversion failed on "
                                         "boundary face " + std::to_string(k) + ".");
            s.TB[k] = r.T;
        }
        s.psiB[k] = gasPsi(s.pB[k], s.TB[k], g);
    }

    // createFields.H: rho = thermo.rho() = p*psi; rhoU = rho*U; rhoE = rho*(e + 0.5*magSqr(U))
    s.rho.resize(cs.nC);
    s.rhoU.resize(cs.nC);
    s.rhoE.resize(cs.nC);
    for (label c = 0; c < cs.nC; ++c)
    {
        s.rho[c]  = s.p[c]*s.psi[c];
        s.rhoU[c] = s.rho[c]*s.U[c];
        s.rhoE[c] = s.rho[c]*(s.e[c] + 0.5*magSqr(s.U[c]));
    }
    s.rhoB.resize(cs.nB);
    s.rhoUB.resize(cs.nB);
    s.rhoEB.resize(cs.nB);
    for (label k = 0; k < cs.nB; ++k)
    {
        s.rhoB[k]  = s.pB[k]*s.psiB[k];
        s.rhoUB[k] = s.rhoB[k]*s.UB[k];
        s.rhoEB[k] = s.rhoB[k]*(s.eB[k] + 0.5*magSqr(s.UB[k]));
    }
    return s;
}

} // namespace rhoCentral
} // namespace cpu
} // namespace brae
