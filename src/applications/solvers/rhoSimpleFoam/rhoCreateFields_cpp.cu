// _cpp REFERENCE implementation -- see createFields_cpp.cuh for the OpenFOAM provenance.
#include "rhoCreateFields_cpp.cuh"
#include "frozen_bc_guard.cuh"
#include "kEpsilon_cpp.cuh"    // correctNutField: turbulence->validate() before the first solve
#include "kOmegaSST_cpp.cuh"   // likewise, for the other closure
#include "transport_model.cuh" // transportMu: the construction-time nu = mu(T)/rho
#include "near_wall_dist.cuh"  // nearWallDist: the wall functions' y
#include "cell_wall_dist.cuh"  // cellWallDist: kOmegaSST's F2
#include "patch_entry_lookup.cuh"   // findPatchEntry: OF patch/group/regex resolution
#include "foam_field_reader.cuh"
#include "thermo_parse.cuh"
#include "equation_of_state.cuh"
#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace brae {
namespace cpu {
// rhoSimpleFoam's own components. Namespaced because simpleFoam has an assembleUEqn, a createFields and
// a pEqn of its own: the two solvers transcribe DIFFERENT OpenFOAM files that happen to share names, and
// letting them collide in one namespace would make which one a call site gets an accident of includes.
namespace rhoSimple {

namespace {

const scalar kGreat = 1.0e15;   // OpenFOAM GREAT, as pressureControl.C initialises pMax_/pMin_ with

bool fileExists(const std::string& path)
{
    return std::filesystem::exists(path) || std::filesystem::exists(path + ".gz");
}

// GeometricField.C:1073-1084 -- true unless some patch fixes a value.
bool needReference(const GeometricField<scalar>& p)
{
    for (const auto& b : p.boundary)
        if (b->fixesValue()) return false;
    return true;
}

// findRefCell.C:33-119. Returns true when a reference WAS set, which is what pressureControl uses to
// decide whether a reference pressure is available for the *Factor limits.
bool setRefCell(
    const GeometricField<scalar>& p,
    const FoamDict*               dict,
    label                         nCells,
    label&                        refCell,
    scalar&                       refValue)
{
    if (!needReference(p)) return false;
    if (!dict)
        throw std::runtime_error(
            "rhoSimpleFoam createFields: p needs a reference (no boundary patch fixes its value) but "
            "fvSolution has no SIMPLE dictionary to read pRefCell/pRefValue from (findRefCell.C:33).");

    if (dict->found("pRefCell"))
    {
        refCell = dict->intOr("pRefCell", 0);
        if (refCell < 0 || refCell >= nCells)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: illegal pRefCell " + std::to_string(refCell)
                + "; should be 0.." + std::to_string(nCells) + " (findRefCell.C:55-62).");
    }
    else if (dict->found("pRefPoint"))
    {
        // OpenFOAM locates the cell containing pRefPoint (mesh.findCell). brae has no point-location
        // search on this path, and guessing a cell would silently pin the pressure level somewhere the
        // user did not ask for.
        throw std::runtime_error(
            "rhoSimpleFoam createFields: pRefPoint is set. OpenFOAM resolves it with mesh.findCell "
            "(findRefCell.C:69-100); brae has no cell search here. Use pRefCell instead.");
    }
    else
    {
        throw std::runtime_error(
            "rhoSimpleFoam createFields: p needs a reference (no boundary patch fixes its value) but "
            "neither pRefCell nor pRefPoint is set in the SIMPLE dictionary (findRefCell.C).");
    }
    // findRefCell.C:107 is dict.readEntry(refValueName, refValue) -- a FatalIOError when absent,
    // not a default. scalarOr's silent 0.0 here re-levelled every all-Neumann case whose author
    // forgot the entry, which converges happily at the wrong absolute pressure (and rho with it,
    // compressibly).
    if (!dict->found("pRefValue"))
        throw std::runtime_error(
            "rhoSimpleFoam createFields: p needs a reference and pRefCell/pRefPoint is set, but "
            "`pRefValue` is missing from the SIMPLE dictionary. OpenFOAM refuses this too "
            "(findRefCell.C readEntry).");
    refValue = dict->scalarOr("pRefValue", 0.0);
    return true;
}

// pressureControl.C:33-190, in OpenFOAM's own order.
PressureControl makePressureControl(
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho,
    const FoamDict*               dict,
    label                         nCells)
{
    PressureControl pc;
    pc.pMax = kGreat;
    pc.pMin = 0.0;

    bool   pLimits = false;
    scalar pMax = -kGreat;
    scalar pMin = kGreat;

    if (setRefCell(p, dict, nCells, pc.refCell, pc.refValue))
    {
        pLimits = true;
        pMax = pc.refValue;
        pMin = pc.refValue;
    }
    if (!dict) return pc;

    // pMax AND pMin together short-circuit everything below -- no boundary scan, no factors.
    if (dict->found("pMax") && dict->found("pMin"))
    {
        pc.pMax = dict->scalarOr("pMax", kGreat);
        pc.limitMaxP = true;
        pc.pMin = dict->scalarOr("pMin", 0.0);
        pc.limitMinP = true;
        return pc;
    }

    // Otherwise the reference pressure (and density) come from the patches that FIX a value.
    scalar rhoRefMax = -kGreat;
    scalar rhoRefMin = kGreat;
    bool   rhoLimits = false;
    for (std::size_t pi = 0; pi < p.boundary.size(); ++pi)
    {
        if (!p.boundary[pi]->fixesValue()) continue;
        const std::vector<scalar>& pv = p.boundary[pi]->value();
        if (pv.empty()) continue;
        pLimits   = true;
        rhoLimits = true;
        pMax = std::max(pMax, *std::max_element(pv.begin(), pv.end()));
        pMin = std::min(pMin, *std::min_element(pv.begin(), pv.end()));
        if (pi < rho.boundary.size())
        {
            const std::vector<scalar>& rv = rho.boundary[pi]->value();
            if (!rv.empty())
            {
                rhoRefMax = std::max(rhoRefMax, *std::max_element(rv.begin(), rv.end()));
                rhoRefMin = std::min(rhoRefMin, *std::min_element(rv.begin(), rv.end()));
            }
        }
    }

    // The MAXIMUM: pMax, else pMaxFactor * reference, else the backward-compatible rhoMax.
    if (dict->found("pMax"))
    {
        pc.pMax = dict->scalarOr("pMax", kGreat);
        pc.limitMaxP = true;
    }
    else if (dict->found("pMaxFactor"))
    {
        if (!pLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'pMaxFactor' specified rather than 'pMax', but the "
                "corresponding reference pressure cannot be evaluated from the boundary conditions "
                "(no patch fixes p and no pRefCell). Specify 'pMax' rather than 'pMaxFactor' "
                "(pressureControl.C:96-105).");
        pc.pMax = pMax * dict->scalarOr("pMaxFactor", 1.0);
        pc.limitMaxP = true;
    }
    else if (dict->found("rhoMax"))
    {
        // OpenFOAM warns and keeps going; brae keeps the same behaviour but says it out loud, because
        // the limit that results is scaled off a boundary density rather than one the user wrote.
        if (!pLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'rhoMax' specified rather than 'pMax', but the "
                "corresponding reference pressure cannot be evaluated from the boundary conditions "
                "(pressureControl.C:112-126).");
        if (!rhoLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'rhoMax' specified rather than 'pMaxFactor', but the "
                "corresponding reference density cannot be evaluated from the boundary conditions "
                "(pressureControl.C:127-137).");
        const scalar rhoMax = dict->scalarOr("rhoMax", kGreat);
        pc.pMax = std::max(rhoMax / rhoRefMax, (scalar)1.0) * pMax;
        pc.limitMaxP = true;
    }

    // The MINIMUM: the same three, mirrored.
    if (dict->found("pMin"))
    {
        pc.pMin = dict->scalarOr("pMin", 0.0);
        pc.limitMinP = true;
    }
    else if (dict->found("pMinFactor"))
    {
        if (!pLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'pMinFactor' specified rather than 'pMin', but the "
                "corresponding reference pressure cannot be evaluated from the boundary conditions "
                "(pressureControl.C:145-155).");
        pc.pMin = pMin * dict->scalarOr("pMinFactor", 1.0);
        pc.limitMinP = true;
    }
    else if (dict->found("rhoMin"))
    {
        if (!pLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'rhoMin' specified rather than 'pMin', but the "
                "corresponding reference pressure cannot be evaluated from the boundary conditions "
                "(pressureControl.C:162-176).");
        if (!rhoLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'rhoMin' specified rather than 'pMinFactor', but the "
                "corresponding reference density cannot be evaluated from the boundary conditions "
                "(pressureControl.C:177-187).");
        const scalar rhoMin = dict->scalarOr("rhoMin", 0.0);
        pc.pMin = std::min(rhoMin / rhoRefMin, (scalar)1.0) * pMin;
        pc.limitMinP = true;
    }
    return pc;
}

}   // namespace


bool PressureControl::limit(std::vector<scalar>& p) const
{
    if (!limitMaxP && !limitMinP) return false;
    if (limitMaxP)
    {
        for (scalar& v : p) v = std::min(v, pMax);
    }
    if (limitMinP)
    {
        for (scalar& v : p) v = std::max(v, pMin);
    }
    // OpenFOAM returns true whenever a limit is ACTIVE, not whether a value actually moved -- the caller
    // uses it to decide whether to re-evaluate p's boundary conditions.
    return true;
}


RhoSimpleFields createFields(
    const std::string&          timeDir,
    const std::string&          caseDir,
    const FoamDict*             simpleDict,
    const FoamDict*             fvSolution,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    RhoSimpleFields f;
    const label nC = m.nCells();

    // COUPLED PATCHES, refused on topology alone before any file is read. The patch-field factory
    // builds cyclic/AMI/processor types as zeroGradient PLACEHOLDERS for the device solvers, whose
    // DeviceMesh re-couples the faces -- no rhoSimpleFoam mirror driver does, so every equation this
    // field set feeds would lose the interface's contribution silently, as if it were a wall. The
    // CUDA arm has refused this from the start (rhoCreateFields.cu); the host arm relied on the T->he
    // whitelist firing by ACCIDENT, five field reads later, with a message about energy boundary
    // conditions.
    for (const FvPatch& cp : patches)
        if (isCoupledInterfaceType(cp.type) || cp.type == "processor")
            throw std::runtime_error(
                "rhoSimpleFoam createFields(cpp): the mesh has a coupled patch ('" + cp.name +
                "', type " + cp.type + "). The host factory builds coupled types as zeroGradient "
                "placeholders for the device solvers, and no mirror driver adds the interface "
                "coupling those placeholders stand in for. Refusing rather than solving the case as "
                "if the interface were a wall.");

    // fluidThermo::New(mesh). readThermoCoeffs refuses an unsupported thermo BY NAME rather than falling
    // back to a default, so an unhandled equation of state stops here instead of silently running as a
    // perfect gas.
    f.thermo = readThermoCoeffs(caseDir, fvSolution);
    // runTime.deltaTValue() scales the continuity errors; steady SIMPLE cases carry deltaT 1 but the
    // value is the CASE's, not an assumption.
    try { f.deltaT = readDict(caseDir + "/system/controlDict").scalarOr("deltaT", 1.0); } catch (...) {}

    // ...but a SUPPORTED parse is not a supported path. The parser accepts `properties liquid` because
    // the LEGACY binary carries the NSRDS path; everything below evaluates perfectGas + hConst directly
    // (perfectGasRho for the rho seed, perfectGasPsi, heOf for the energy build), and on the liquid
    // parse the scalar Cp/mu/kappa members STAY AT THEIR DEFAULTS (thermo_parse.cuh notes it at the
    // liquid branch) -- so he would be built from a gas Cp the case never set. device_energy.cu:38-42
    // records what that costs: squareBendLiq's 350 K walls carried he = -48361 J/kg where Es(1e5,350)
    // is -15641742 J/kg. The device hooks have this guard (requirePerfectGas, rhoThermoDevice.cu);
    // the host createFields, which runs FIRST on both arms, did not.
    if (f.thermo.model != ThermoModel::perfectGas)
        throw std::runtime_error(
            "brae: rhoSimpleFoam (OF-mirror) createFields implements perfectGas + hConst only, and "
            "this case selects `properties liquid`. The liquid path replaces Cp, mu, kappa and rho "
            "with per-cell NSRDS correlations and inverts he -> T by Newton; the legacy "
            "gpuRhoSimpleFoam binary carries that path. Refusing rather than running a gas equation "
            "of state against a liquid's coefficients.");

    // thermo.validate(args.executable(), "h", "e") -- rhoSimpleFoam accepts exactly these two energy
    // variables, because EEqn.H's kinetic-energy source is written for both and for nothing else.
    {
        const FoamDict tp = readDict(caseDir + "/constant/thermophysicalProperties");
        const FoamDict* tt = tp.subDict("thermoType");
        const std::string energy = tt ? tt->wordOr("energy", "") : "";
        if (energy == "sensibleEnthalpy")            f.heName = "h";
        else if (energy == "sensibleInternalEnergy") f.heName = "e";
        else
            throw std::runtime_error(
                "brae: rhoSimpleFoam thermo energy '" + (energy.empty() ? std::string("<missing>") : energy)
                + "' is not one this solver transports (sensibleEnthalpy -> h, sensibleInternalEnergy -> e)."
                  " OpenFOAM's thermo.validate(.., \"h\", \"e\") refuses the same set. Refusing rather than"
                  " solving a different energy equation.");
    }

    // NO rhoSimpleFoam mirror driver maintains a per-step boundary -- neither the _cpp loop nor the
    // CUDA arm carries the NVRTC or collect* hooks the factory's fixedMean/fanPressure/coded acceptance
    // is written against -- so every field read is checked here, the last place the dictionary type
    // still exists (frozen_bc_guard.cuh). The device arm builds its structures FROM these host fields,
    // so one guard covers both.
    auto guardRead = [&](auto fd, const char* nm)
    {
        refuseFrozenPerStepBC(fd, nm, "rhoSimpleFoam (mirror)", false);
        return fd;
    };

    // p = thermo.p() and T: both read by the thermo, both MUST_READ. Read together so psi and rho below
    // are derived from the SAME state rather than from two fields that could be an iteration apart.
    const FieldData<scalar> pFd = guardRead(readField<scalar>(timeDir + "/p"), "p");
    // A flux-switched PRESSURE patch is refreshed on neither arm: the flux switch (updateFromFlux on the
    // host, deviceUpdateInletOutlet on the device) is pushed into U, he and T every iteration and never
    // into p, so an inletOutlet/outletInlet p would keep the switch it was seeded with for the whole
    // run -- OpenFOAM's inletOutlet reads phi in updateCoeffs each time (inletOutletFvPatchField.C:72).
    // No fixture carries one (freestreamPressure is a different, refreshed path). Refused by name.
    for (const auto& b : pFd.boundary)
    {
        if (b.type == "inletOutlet" || b.type == "outletInlet")
            throw std::runtime_error(
                "rhoSimpleFoam createFields: p's patch '" + b.name + "' is `" + b.type + "`, a flux-switched "
                "condition whose valueFraction OpenFOAM recomputes from phi every updateCoeffs. The mirror "
                "pushes the flux into U, he and T only, so this patch would keep its seeded switch for the "
                "whole run on both arms. Refusing rather than running a frozen switch under the case's name.");
    }
    f.p = buildField<scalar>(pFd, patches, nC);
    f.p.evaluateBoundary();
    const FieldData<scalar> tFd = guardRead(readField<scalar>(timeDir + "/T"), "T");
    f.T = buildField<scalar>(tFd, patches, nC);
    f.T.evaluateBoundary();

    // rho: READ_IF_PRESENT, else thermo.rho(). A restart continues from the written density; a cold start
    // computes it from the equation of state.
    const std::string rhoPath = timeDir + "/rho";
    f.rhoWasRead = fileExists(rhoPath);
    if (f.rhoWasRead)
    {
        f.rho = buildField<scalar>(guardRead(readField<scalar>(rhoPath), "rho"), patches, nC);
        f.rho.evaluateBoundary();
    }
    else
    {
        // thermo.rho() with the boundary values taken from the boundary p and T, so rho's patch values are
        // the equation of state's and not a copy of the internal cell's.
        FieldData<scalar> fd;
        fd.internalUniform = false;   // defaults TRUE; a hand-built field must say so
        fd.internalField.resize(nC);
        for (label c = 0; c < nC; ++c)
            fd.internalField[c] = perfectGasRho(f.p.internal[c], f.T.internal[c], f.thermo);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            PatchFieldData<scalar> b;
            b.name     = patches[pi].name;
            b.type     = "calculated";
            b.hasValue = true;
            const std::vector<scalar>& pb = f.p.boundary[pi]->value();
            const std::vector<scalar>& tb = f.T.boundary[pi]->value();
            b.values.resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                b.values[i] = perfectGasRho(pb[i], tb[i], f.thermo);
            fd.boundary.push_back(std::move(b));
        }
        f.rho = buildField<scalar>(fd, patches, nC);
    }

    // U: MUST_READ.
    f.U = buildField<vector>(guardRead(readField<vector>(timeDir + "/U"), "U"), patches, nC);

    // createFields.H builds rho (line 22) BEFORE U (line 26), so when OpenFOAM constructs U's patches the
    // rho field is already registered and flowRateInletVelocity's constructor-time updateCoeffs takes the
    // REAL patch density -- not `rhoInlet`, which sbMatched even labels "Guess for rho" and which OF
    // reaches only when no rho is registered at all. This must happen before phi is built below, because
    // compressibleCreatePhi.H builds phi FROM U and the seed would otherwise be carried into the flux.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        f.U.boundary[pi]->updateAtConstruction(f.rho.boundary[pi]->value());
    }
    f.U.evaluateBoundary();

    // compressibleCreatePhi.H: READ_IF_PRESENT, else linearInterpolate(rho*U) & Sf.
    //
    // The PRODUCT is interpolated, not the two factors separately -- see the header. rho*U is built per
    // cell and per boundary face and fluxed through the ordinary linear-interpolation path, so the face
    // weights are the ones fvc::flux already reproduces from OpenFOAM.
    const std::string phiPath = timeDir + "/phi";
    f.phiWasRead = fileExists(phiPath);
    if (f.phiWasRead)
    {
        const FieldData<scalar> pf = readField<scalar>(phiPath);
        f.phi.internal = pf.internalField;
        f.phi.boundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            f.phi.boundary[pi].assign(patches[pi].size, 0.0);
            for (const auto& b : pf.boundary)
                if (b.name == patches[pi].name && b.hasValue
                    && static_cast<label>(b.values.size()) == patches[pi].size)
                    f.phi.boundary[pi] = b.values;
        }
    }
    else
    {
        std::vector<vector> rhoU(nC);
        for (label c = 0; c < nC; ++c)
        {
            const scalar r = f.rho.internal[c];
            rhoU[c] = vector{ r * f.U.internal[c].x, r * f.U.internal[c].y, r * f.U.internal[c].z };
        }
        std::vector<std::vector<vector>> rhoUb(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& rb = f.rho.boundary[pi]->value();
            const std::vector<vector>& ub = f.U.boundary[pi]->value();
            rhoUb[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                rhoUb[pi][i] = vector{ rb[i] * ub[i].x, rb[i] * ub[i].y, rb[i] * ub[i].z };
        }
        f.phi = fvc::flux(rhoU, rhoUb, m, g, patches);
    }

    // createFieldRefs.H: psi is thermo.psi(). Derived from the same T that rho came from.
    f.psi.resize(nC);
    for (label c = 0; c < nC; ++c) f.psi[c] = perfectGasPsi(f.T.internal[c], f.thermo);
    f.psiBnd.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& tb = f.T.boundary[pi]->value();
        f.psiBnd[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            f.psiBnd[pi][i] = perfectGasPsi(tb[i], f.thermo);
    }

    // thermo.he() -- the variable EEqn transports, WITH ITS BOUNDARY CONDITIONS.
    //
    // OpenFOAM does not give he the boundary conditions of T; basicThermo::heBoundaryTypes() derives
    // ENERGY ones from them -- fixedValue becomes fixedEnergy, zeroGradient becomes gradientEnergy,
    // inletOutlet becomes mixedEnergy. Those energy types are not a relabelling: a fixedValue patch's
    // boundaryCoeffs come from its refValue, so an he built by borrowing T's patch fields carries a
    // TEMPERATURE where the coefficients want an ENERGY, which reads as a source error on every
    // inlet-adjacent cell and nowhere else.
    //
    // What is done here is the mapping for the set where it is EXACT for this thermo, and a refusal by
    // name for anything else:
    //     fixedValue    -> fixedValue    on he(T_b)         exact: fixedEnergy IS a fixedValue on he
    //     zeroGradient  -> zeroGradient                      gradientEnergy's second term is
    //                                                        deltaCoeffs*(he(pw,Tw) - he(pw,Tw,cells))
    //                                                        -- SAME p, SAME T, and for any pureMixture
    //                                                        the same coefficients, so it is IDENTICALLY
    //                                                        zero (an earlier version of this comment
    //                                                        credited p-independence, which is not the
    //                                                        reason: the term compares patch and cell
    //                                                        MIXTURES, and pureMixture has one)
    //     fixedGradient -> fixedGradient on Cpv*grad(T)      gradientEnergy with that zero second term:
    //                                                        gradient() = Cpv(pw,Tw)*Tw.snGrad()
    //                                                        (gradientEnergyFvPatchScalarField.C:99-105),
    //                                                        and Cpv is the CONSTANT Cp (or Cv = Cp - R
    //                                                        for sensibleInternalEnergy) under hConst --
    //                                                        so the mapping is static and exact. The
    //                                                        gradient SCALES by Cpv; it must not go
    //                                                        through heOf, which is affine -- applying an
    //                                                        offset to a slope is the trap the mx/hf
    //                                                        controls measure (7.97e-03 on rhoBoxQ)
    //     mixed         -> mixed         with vf unchanged,  mixedEnergy verbatim
    //                      refValue -> he(refValue_T),       (mixedEnergyFvPatchScalarField.C:103-115):
    //                      refGrad  -> Cpv*refGrad_T         same zero second term
    //     inletOutlet   -> inletOutlet  on he(T_inletValue)  exact: mixedEnergy is the same mixed BC with
    //                                                        the energy refValue
    // A multi-species mixture revives the second term, and a thermo whose Cpv varies breaks the static
    // scaling; the refusal below is what stops either from becoming a silent wrong answer.
    {
        auto heOf = [&](scalar T)
        {
            const scalar hs = f.thermo.Cp * (T - f.thermo.Tref) + f.thermo.Href;
            // e = h - p/rho = h - R*T for a perfect gas. Hf, the heat of formation, belongs to the
            // ABSOLUTE enthalpy only and must not appear in the sensible energy EEqn transports.
            return (f.heName == "e") ? hs - f.thermo.R * T : hs;
        };
        FieldData<scalar> heFd;
        heFd.internalUniform = false;
        heFd.internalField.resize(nC);
        for (label c = 0; c < nC; ++c) heFd.internalField[c] = heOf(f.T.internal[c]);
        // OVER THE MESH'S PATCHES, not over the file's entries. OpenFOAM resolves each PATCH to an entry
        // by name, then by group, then by regex, and an entry matching no patch is simply unused. Walking
        // the entries instead refuses on ones that were never going to apply: every modern tutorial
        // carries `#includeEtc "caseDicts/setConstraintTypes"`, which defines an entry for cyclic, wedge,
        // processor and the rest so that constraint patches get the right condition automatically. On
        // aerofoilNACA0012 -- whose only constraint patches are `empty` -- that made brae refuse the case
        // over a `cyclic` entry no patch matched.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const PatchFieldData<scalar>* tbp = findPatchEntry(tFd.boundary, patches[pi]);
            if (!tbp)
                throw std::runtime_error(
                    "brae: rhoSimpleFoam createFields found no T boundary entry for patch '"
                    + patches[pi].name + "'. OpenFOAM resolves a patch to an entry by name, then by "
                    "group, then by regex; none matched. Refusing rather than inventing one.");
            const PatchFieldData<scalar>& tb = *tbp;
            PatchFieldData<scalar> b = tb;          // type and structure carried over
            b.name = patches[pi].name;              // the PATCH's name, so buildField resolves it directly
            if (tb.type != "fixedValue" && tb.type != "zeroGradient" && tb.type != "inletOutlet"
                && tb.type != "fixedGradient" && tb.type != "mixed"
                && tb.type != "calculated" && tb.type != "empty" && tb.type != "symmetry"
                && tb.type != "symmetryPlane" && tb.type != "wedge" && tb.type != "slip")
                throw std::runtime_error(
                    "brae: rhoSimpleFoam createFields cannot derive he's boundary condition from T's '"
                    + tb.type + "' on patch '" + tb.name + "'. OpenFOAM maps T's boundary conditions to "
                    "ENERGY ones (basicThermo::heBoundaryTypes) and brae implements that mapping only "
                    "where it is exact for perfectGas+hConst. Refusing rather than transporting an "
                    "energy under a temperature's boundary condition.");
            b.uniformValue     = heOf(tb.uniformValue);
            for (auto& v : b.values)      v = heOf(v);
            b.refValueUniformValue = heOf(tb.refValueUniformValue);   // mixed's refValue slot
            for (auto& v : b.refValues)   v = heOf(v);
            b.inletUniformValue = heOf(tb.inletUniformValue);
            for (auto& v : b.inletValues) v = heOf(v);
            // The GRADIENT slots (fixedGradient's `gradient`, mixed's `refGradient`) SCALE by Cpv --
            // never heOf, which is affine: an offset applied to a slope was worth 7.97e-03 on rhoBoxQ's
            // T when the mx/hf controls first measured it. Cpv is constant under hConst, which is what
            // makes this static mapping exact (gradientEnergy/mixedEnergy re-evaluate it live).
            if (tb.hasGradient)
            {
                const scalar Cpv = (f.heName == "e") ? f.thermo.Cp - f.thermo.R : f.thermo.Cp;
                b.gradientUniformValue = Cpv * tb.gradientUniformValue;
                for (auto& g : b.gradientValues) g = Cpv * g;
            }
            heFd.boundary.push_back(std::move(b));
        }
        f.he = buildField<scalar>(heFd, patches, nC);
        f.he.evaluateBoundary();
    }

    // The case's own kEpsilon coefficients, read below and used by the construction-time correctNut
    // further down -- declared here because the dictionary block that reads them closes before it.
    KEpsilonCoeffs keCase;

    // compressible::turbulenceModel::New(rho, U, phi, thermo) -- the model is constructed here in
    // createFields.H, and constructing it is what reads k, epsilon, nut and alphat. A laminar case reads
    // none of them, which is why they are gated on the dictionary rather than on the files existing.
    {
        // OpenFOAM renamed turbulenceProperties to momentumTransport; both names are in the wild and a
        // case carries exactly one. Neither present is a real case too -- rhoSimpleFoam constructs the
        // model unconditionally, so a case with no dictionary at all is refused rather than assumed
        // laminar, which would run a turbulent case with no closure and report nothing.
        const std::string mtPath = caseDir + "/constant/momentumTransport";
        const std::string tpPath = caseDir + "/constant/turbulenceProperties";
        std::string dictPath;
        if (fileExists(mtPath))      dictPath = mtPath;
        else if (fileExists(tpPath)) dictPath = tpPath;
        else
            throw std::runtime_error(
                "brae: rhoSimpleFoam found neither constant/momentumTransport nor "
                "constant/turbulenceProperties. OpenFOAM constructs the turbulence model from one of "
                "them in createFields.H; refusing rather than assuming the case is laminar.");
        const FoamDict mt2 = readDict(dictPath);
        const std::string sim = mt2.wordOr("simulationType", "laminar");
        if (sim == "RAS")
        {
            const FoamDict* ras = mt2.subDict("RAS");
            // `turbulence off` does NOT mean laminar: OpenFOAM constructs the model regardless (all
            // four fields read) and validate() still runs correctNut() once -- only the per-iteration
            // correct() is skipped (kEpsilon.C:216 `if (!this->turbulence_) return`). Treating off as
            // laminar dropped rho*nut from every face; on the rhoBoxF oracle the frozen nut is
            // Cmu*k^2/eps = 0.001265625 against mu/rho ~ 1.5e-5 -- eighty times the molecular value.
            f.turbulent = true;
            f.turbulenceFrozen = ras && ras->wordOr("turbulence", "on") == "off";
            f.rasModel  = ras ? ras->wordOr("RASModel", "") : "";
            // ALL SIX, as OpenFOAM reads them (kEpsilon.C:199-204 readIfPresent on Cmu, C1, C2, C3,
            // sigmak, sigmaEps). Only Cmu was read here, so a case naming any of the other five got the
            // model default and no warning. OF spells the diffusivity denominators `sigmak`/`sigmaEps`;
            // brae's struct calls the first sigmaK, and the DICT KEY is OpenFOAM's.
            if (const FoamDict* kec = ras ? ras->subDict("kEpsilonCoeffs") : nullptr)
            {
                keCase.Cmu      = kec->scalarOr("Cmu",      keCase.Cmu);
                keCase.C1       = kec->scalarOr("C1",       keCase.C1);
                keCase.C2       = kec->scalarOr("C2",       keCase.C2);
                keCase.C3       = kec->scalarOr("C3",       keCase.C3);
                keCase.sigmaK   = kec->scalarOr("sigmak",   keCase.sigmaK);
                keCase.sigmaEps = kec->scalarOr("sigmaEps", keCase.sigmaEps);
            }
            // Onto the FIELD SET, here and unconditionally, because the SOLVE reads them every
            // iteration and not just the construction-time correctNut below. Set outside the
            // `validate()` guard on purpose: that guard also requires k and epsilon to be sized, and a
            // case that failed it would have carried the model defaults into the loop silently.
            f.keCoeffs = keCase;
            readKOmegaSSTCoeffs(ras, f.sstCoeffs);   // OpenFOAM's defaults where the dict is absent
            f.Prt      = f.thermo.Prt;
            if (ras && ras->wordOr("RASModel", "") == "kOmegaSST")
                std::printf("  kOmegaSSTCoeffs (case): betaStar=%.4g a1=%.4g gamma1=%.4g beta1=%.4g Prt=%.4g\n",
                            (double)f.sstCoeffs.betaStar, (double)f.sstCoeffs.a1, (double)f.sstCoeffs.gamma1,
                            (double)f.sstCoeffs.beta1, (double)f.Prt);
            if (f.turbulent)
            {
                if (f.rasModel != "kEpsilon" && f.rasModel != "kOmegaSST")
                    throw std::runtime_error(
                        "brae: rhoSimpleFoam RASModel '" + f.rasModel + "' is not ported for the "
                        "compressible lineage (kEpsilon and kOmegaSST are). Refusing rather than running "
                        "a different closure, or none.");
                // Frozen kOmegaSST would need ITS validate() -- correctNut from k, omega and the strain
                // rate -- which the kEpsilon-shaped block below cannot provide. Refusing rather than
                // entering the loop with the file nut where OpenFOAM enters with the model's.
                if (f.turbulenceFrozen && f.rasModel != "kEpsilon")
                    throw std::runtime_error(
                        "brae: rhoSimpleFoam `RAS { turbulence off; }` is implemented for kEpsilon only "
                        "-- the one-shot validate() below is kEpsilon's correctNut. '" + f.rasModel +
                        "' frozen would start from the wrong nut. Refusing.");
                f.k   = buildField<scalar>(guardRead(readField<scalar>(timeDir + "/k"), "k"), patches, nC);

                // THE NUT WALL FUNCTION IS PART OF THE MODEL, and only nutkWallFunction is ported here.
                //
                // OpenFOAM has exactly ONE dispatch point for it: nut's own patch field
                // (nutWallFunctionFvPatchScalarField.C:181-184 `operator==(calcNut())`, a virtual call
                // per patch), and everything downstream READS the result -- epsilonWallFunction's
                // near-wall production is `const tmp<scalarField> tnutw = turbModel.nut(patchi);`
                // (epsilonWallFunctionFvPatchScalarField.C:333-334), not a recomputation. brae's closure
                // has no such dispatch: kEpsilon_cpp.cu and kEpsilon.cu both call nutkWallFunction
                // unconditionally, and kEpsilon.cu additionally passes `/*nutWall=*/0` to deviceWallEpsG0
                // for the production term. So a case naming any other member of the family got nutk's
                // value at BOTH sites, silently.
                //
                // The types are genuinely different functions, not variants:
                //   nutUSpaldingWallFunction   Newton on Spalding's law, a function of |U| -- nutk never
                //                              reads U at all (nutkWallFunctionFvPatchScalarField.C:71).
                //   nutUWallFunction /         likewise U-based; the in-tree claim that nutUBlended
                //   nutUBlendedWallFunction    "is the same" as nutk is about the LEGACY device path's
                //                              own approximation, not about OpenFOAM.
                //   nutLowReWallFunction       calcNut() returns Zero UNCONDITIONALLY
                //                              (nutLowReWallFunctionFvPatchScalarField.C:38-42). It is
                //                              NOT "nutk on a resolved mesh": nutk's y+ is the k-based
                //                              Cmu^.25*y*sqrt(k)/nu, and a mesh resolved in friction
                //                              units can still carry k-based y+ > yPlusLam and take the
                //                              log branch where OpenFOAM returns exactly 0.
                //   atmNutkWallFunction        roughness z0, and OpenFOAM's z0 is a per-face,
                //                              time-varying PatchFunction1 where brae carries one scalar.
                //
                // Refused HERE because this is the last place the dictionary TYPE still exists -- once
                // buildField has run, the patch field object no longer carries it, which is why neither
                // the host closure nor the device one could have checked.
                {
                    const FieldData<scalar> nutRaw = guardRead(readField<scalar>(timeDir + "/nut"), "nut");
                    for (const auto& b : nutRaw.boundary)
                    {
                        // TWO prefixes, because the atm family does not start with "nut":
                        // atmNutkWallFunction -- named in the comment above as refused -- passed a
                        // one-prefix test and ran as the smooth nutk with z0 unplumbed, exactly the
                        // substitution this throw exists to prevent.
                        if (b.type.rfind("nut", 0) != 0
                         && b.type.rfind("atmNut", 0) != 0) continue;        // not a nut wall function
                        if (b.type == "nutkWallFunction")
                        {
                            // Ported for the LIVE closure, which recomputes the wall nut every
                            // iteration. Frozen, OpenFOAM evaluates it exactly ONCE -- inside
                            // validate()'s correctBoundaryConditions -- and brae has no wall-function
                            // evaluation at createFields, so the wall would keep the file value.
                            if (!f.turbulenceFrozen) continue;
                            throw std::runtime_error(
                                "brae: rhoSimpleFoam `RAS { turbulence off; }` with a nutkWallFunction "
                                "patch ('" + b.name + "') is not implemented -- OpenFOAM wall-evaluates "
                                "nut once at validate() and brae cannot yet do that outside the live "
                                "closure. Use a calculated/zeroGradient nut boundary, or turbulence on.");
                        }
                        throw std::runtime_error(
                            "brae: rhoSimpleFoam nut patch '" + b.name + "' carries '" + b.type +
                            "', which the compressible kEpsilon closure does not implement -- it computes "
                            "nutkWallFunction unconditionally, for the wall nut AND for the near-wall "
                            "production. These are different functions of different inputs (the nutU "
                            "family reads |U|, which nutk never does; nutLowRe is identically zero), so "
                            "substituting nutk would converge to a different wall viscosity and a "
                            "different epsilon. Refusing rather than running one wall function under "
                            "another's name.");
                    }
                    f.nut = buildField<scalar>(nutRaw, patches, nC);
                }
                f.k.evaluateBoundary();
                f.nut.evaluateBoundary();
                // The second turbulence scalar is the model's, not the case directory's: reading
                // whichever file happens to be present would run kOmegaSST off an epsilon a previous
                // kEpsilon run left behind.
                if (f.rasModel == "kOmegaSST")
                {
                    f.omega = buildField<scalar>(guardRead(readField<scalar>(timeDir + "/omega"), "omega"), patches, nC);
                    f.omega.evaluateBoundary();
                }
                else
                {
                    f.epsilon = buildField<scalar>(guardRead(readField<scalar>(timeDir + "/epsilon"), "epsilon"), patches, nC);
                    f.epsilon.evaluateBoundary();
                }
                // MUST_READ, as OpenFOAM's EddyDiffusivity constructs it (EddyDiffusivity.C:26): a
                // turbulent compressible case without 0/alphat is a FATAL in OpenFOAM ("cannot find
                // file"). The host step refused it late, at the first closure call; the CUDA arm did not
                // refuse at all and ran the energy equation on the laminar diffusivity while reporting
                // the model -- measured on rhoKE with the file removed, `Time = 1 ... k ... epsilon` and on.
                if (!fileExists(timeDir + "/alphat"))
                    throw std::runtime_error(
                        "rhoSimpleFoam createFields: the case is RAS and " + timeDir + "/alphat does not "
                        "exist. OpenFOAM's EddyDiffusivity reads alphat MUST_READ when the turbulence model "
                        "is constructed and fatals without it; the energy equation's alphaEff = "
                        "CpByCpv*(alpha + alphat) needs it. Refusing rather than running with alphat = 0.");
                if (fileExists(timeDir + "/alphat"))
                {
                    const FieldData<scalar> aFd = guardRead(readField<scalar>(timeDir + "/alphat"), "alphat");
                    f.alphat = buildField<scalar>(aFd, patches, nC);
                    f.alphat.evaluateBoundary();
                    // Which patches carry compressible::alphatWallFunction, and each one's own Prt --
                    // see the note on RhoSimpleFields. This is the ONLY place 0/alphat is read, so it is
                    // the only place the patch types are still in hand.
                    f.alphatWallFn.assign(patches.size(), 0);
                    f.alphatPrt.assign(patches.size(), scalar(0.85));
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        const PatchFieldData<scalar>* pb = findPatchEntry(aFd.boundary, patches[pi]);
                        if (!pb) continue;
                        if (pb->type == "compressible::alphatWallFunction"
                         || pb->type == "alphatWallFunction")
                        {
                            f.alphatWallFn[pi] = 1;
                            f.alphatPrt[pi]    = pb->Prt;
                        }
                    }
                }
            }
        }
        else if (sim != "laminar")
        {
            throw std::runtime_error(
                "brae: rhoSimpleFoam simulationType '" + sim + "' is neither laminar nor RAS. Refusing.");
        }
    }

    // turbulence->validate(), rhoSimpleFoam.C:64 -- BEFORE the SIMPLE loop, and it is not a no-op.
    // eddyViscosity::validate() calls correctNut(), so OpenFOAM enters its FIRST momentum solve with
    // nut = Cmu*k^2/epsilon rather than whatever the case's 0/nut file happens to say. angledDuct's
    // 0/nut is `uniform 0` while Cmu*k^2/eps is 0.09*1/200 = 4.5e-04, and since rho*nut = 5.4e-04
    // against mu = 1.8e-05 that is THIRTY TIMES the laminar viscosity -- so reading the file left
    // brae solving the first iteration as if the flow were laminar.
    //
    // It showed up as OpenFOAM's own laminar and turbulent U at iteration 1 differing by 1.5847e-01
    // while brae's turbulent U was 1.5868e-01 from OpenFOAM's: the same number, because brae's
    // turbulent first iteration WAS the laminar answer. The transient that starts there had not decayed
    // by 8000 iterations.
    //
    // BOTH MODELS, AND THE BOUNDARY HALF. This ran for kEpsilon only and computed the interior only:
    // kOmegaSST entered its first momentum solve on the case file's nut and alphat (validate() skipped),
    // and on either model the wall nut stayed at the file's `uniform 0` where OpenFOAM's
    // nut.correctBoundaryConditions() had already evaluated nutkWallFunction from the initial k -- so
    // the first momentum matrix carried mu at the wall instead of mu + rho*nut_w. Measured (host mirror
    // vs OpenFOAM at iteration 1, linear solvers pinned): rhoKE U 2.1e-03 / nut 4.8e-04; rhoSST nut
    // 1.6e-01 / omega 1.2e-01 / U 1.7e-03. Invisible at convergence, which is where every end-to-end
    // gate compared. The closures' own correctNutField does the whole of it, so construction and the
    // loop cannot drift apart again.
    const bool secondSized = (f.rasModel == "kOmegaSST")
        ? static_cast<label>(f.omega.internal.size()) == nC
        : static_cast<label>(f.epsilon.internal.size()) == nC;
    if (f.turbulent && static_cast<label>(f.k.internal.size()) == nC && secondSized)
    {
        // The compressible instantiation's inputs, exactly as the step builds them: nu is the LAMINAR
        // mu(T)/rho, per cell and per boundary face.
        std::vector<scalar> nuLam(nC);
        for (label c = 0; c < nC; ++c)
            nuLam[c] = transportMu(f.T.internal[c], f.thermo) / f.rho.internal[c];
        std::vector<std::vector<scalar>> nuLamBnd(patches.size()), rhoBnd(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& tb = f.T.boundary[pi]->value();
            const std::vector<scalar>& rb = f.rho.boundary[pi]->value();
            rhoBnd[pi] = rb;
            nuLamBnd[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                nuLamBnd[pi][i] = transportMu(tb[i], f.thermo) / rb[i];
        }
        const std::vector<std::vector<scalar>> yWall = nearWallDist(m, g, patches);
        const bool haveAlphat = static_cast<label>(f.alphat.internal.size()) == nC;
        if (f.rasModel == "kEpsilon")
        {
            cpu::kEpsilonRef::Compressible comp;
            comp.rho = &f.rho.internal;  comp.rhoBnd = &rhoBnd;
            comp.nu  = &nuLam;           comp.nuBnd  = &nuLamBnd;
            comp.alphat = haveAlphat ? &f.alphat.internal : nullptr;
            comp.Prt = f.thermo.Prt;
            cpu::kEpsilonRef::correctNutField(f.k, f.epsilon, f.nut, yWall, /*nu=*/0.0, patches, keCase, &comp);
        }
        else
        {
            const std::vector<scalar> y     = cellWallDist(m, g, patches);
            const std::vector<tensor> gradU = fvc::gaussGrad(f.U, m, g, patches);
            cpu::kOmegaSST::Compressible comp;
            comp.rho = &f.rho.internal;  comp.rhoBnd = &rhoBnd;
            comp.nu  = &nuLam;           comp.nuBnd  = &nuLamBnd;
            comp.alphat = haveAlphat ? &f.alphat.internal : nullptr;
            comp.Prt = f.thermo.Prt;
            cpu::kOmegaSST::correctNutField(f.U, f.k, f.omega, f.nut, gradU, y, yWall, /*nu=*/0.0,
                                            m, g, patches, f.sstCoeffs, &comp);
        }
        if (haveAlphat)
        {
            f.alphat.evaluateBoundary();
            correctAlphatBoundary(f, patches);
        }
    }

    f.pressureControl = makePressureControl(f.p, f.rho, simpleDict, nC);

    // initialMass = fvc::domainIntegrate(rho). pEqn.H's closed-volume correction is measured against it.
    f.initialMass = 0.0;
    // basicThermo's constructor runs calculate() before any solving, so rho_ starts equal to rho.
    f.rhoThermo = f.rho.internal;
    f.rhoThermoBnd.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        f.rhoThermoBnd[pi] = f.rho.boundary[pi]->value();

    for (label c = 0; c < nC; ++c) f.initialMass += f.rho.internal[c] * g.V()[c];

    return f;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
