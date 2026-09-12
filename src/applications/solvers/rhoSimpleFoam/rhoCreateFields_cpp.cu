// _cpp REFERENCE implementation -- see createFields_cpp.cuh for the OpenFOAM provenance.
#include "rhoCreateFields_cpp.cuh"
#include "turbulence_setup.cuh"   // readTurbulenceMinima: the ONE reader of kMin/epsilonMin/omegaMin
#include "liquid_thermo.cuh"   // thermo*Of: ONE branch point between the gas and liquid properties
#include "scheme_parse.cuh"   // parseFvSchemesControls: grad(U)'s cellLimited coefficient for validate()
#include "cellLimitedGrad_cpp.cuh"
#include "frozen_bc_guard.cuh"
#include "kEpsilon_cpp.cuh"    // correctNutField: turbulence->validate() before the first solve
#include "bound_cpp.cuh"       // Foam::bound, which every model constructor applies to its two scalars
#include "kOmegaSST_cpp.cuh"   // likewise, for the other closure
#include "transport_model.cuh" // transportMu: the construction-time nu = mu(T)/rho
#include "near_wall_dist.cuh"  // nearWallDist: the wall functions' y
#include "cell_wall_dist.cuh"  // cellWallDist: kOmegaSST's F2
#include "patch_entry_lookup.cuh"   // findPatchEntry: OF patch/group/regex resolution
#include "foam_field_reader.cuh"
#include "thermo_parse.cuh"
#include <memory>
#include "equation_of_state.cuh"
#include <algorithm>
#include <cstdio>
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


void correctGeneralizedNewtonian(
    RhoSimpleFields&            f,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    // thermo.mu() is the thermo's STORED mu_, last filled by thermo.correct() -- from the p before the
    // pressure equation. Every transport brae runs (const, sutherland, the NSRDS liquid) is a function of
    // T alone, so evaluating it at the current p is the same number; the accessor takes p only because
    // the liquid's signature does.
    const std::size_t nC = f.U.internal.size();
    std::vector<scalar> nu0(nC);
    for (std::size_t c = 0; c < nC; ++c)
    {
        nu0[c] = thermoMuOf(f.p.internal[c], f.T.internal[c], f.thermo) / f.rho.internal[c];
    }
    std::vector<std::vector<scalar>> nu0Bnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& pb = f.p.boundary[pi]->value();
        const std::vector<scalar>& tb = f.T.boundary[pi]->value();
        const std::vector<scalar>& rb = f.rho.boundary[pi]->value();
        nu0Bnd[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            nu0Bnd[pi][i] = thermoMuOf(pb[i], tb[i], f.thermo) / rb[i];
        }
    }
    generalizedNewtonian::correctNu(f.U, nu0, nu0Bnd, f.gnCoeffs, f.gnGradULimitK, m, g, patches,
                                    f.gnNu, f.gnNuBnd);
}


std::string turbulenceDictPath(const std::string& caseDir)
{
    const std::string mtPath = caseDir + "/constant/momentumTransport";
    const std::string tpPath = caseDir + "/constant/turbulenceProperties";
    if (fileExists(mtPath)) return mtPath;
    if (fileExists(tpPath)) return tpPath;
    return "";
}

RhoSimpleFields createFields(
    const std::string&          timeDir,
    const std::string&          caseDir,
    const FoamDict*             simpleDict,
    const FoamDict*             fvSolution,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    const FoamDict*             thermoDict,
    const FoamDict*             turbDict,
    scalar                      startTime)
{
    RhoSimpleFields f;
    f.startTime = startTime;
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
    f.thermo = readThermoCoeffs(caseDir, fvSolution, thermoDict, turbDict);
    // runTime.deltaTValue() scales the continuity errors; steady SIMPLE cases carry deltaT 1 but the
    // value is the CASE's, not an assumption.
    try { f.deltaT = readDict(caseDir + "/system/controlDict").scalarOr("deltaT", 1.0); } catch (...) {}

    // `properties liquid` RUNS on this arm as of stage H3.4, and what made that safe was removing the
    // gas formulae from the path rather than adding a liquid branch beside them. Every property now goes
    // through the eight accessors in liquid_thermo.cuh -- mu, Cp, alpha, rho, psi, he, Cpv and THE -- so
    // there is exactly ONE place that knows which thermo the case selected, and a call site cannot be
    // liquid-aware in nineteen places and gas-only in the twentieth.
    //
    // The refusal that stood here for stages H3.0 to H3.3 is kept in the history for what it measured:
    // the parser accepted `properties liquid` while everything below evaluated perfectGas + hConst
    // directly, and on that parse the scalar Cp/mu/kappa members stay at their GAS defaults, so
    // squareBendLiq's 350 K walls carried he = -48361 J/kg where Es(1e5, 350) is -15641742 J/kg
    // (device_energy.cu:38-42). What is refused now is narrower and lives in the parser: any liquid but
    // H2O, any energy form but sensibleInternalEnergy, and anything but heRhoThermo + pureMixture
    // (thermo_parse.cuh, the `properties liquid` block).

    // thermo.validate(args.executable(), "h", "e") -- rhoSimpleFoam accepts exactly these two energy
    // variables, because EEqn.H's kinetic-energy source is written for both and for nothing else.
    {
        std::unique_ptr<FoamDict> ownTp;
        if (!thermoDict) ownTp = std::make_unique<FoamDict>(readDict(caseDir + "/constant/thermophysicalProperties"));
        const FoamDict& tp = thermoDict ? *thermoDict : *ownTp;
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
        if (!b.hasP0Function1) continue;
        // A p0 TABLE. OpenFOAM's uniformTotalPressure samples its p0 Function1 at the current time in
        // every updateCoeffs (uniformTotalPressureFvPatchScalarField.C:149) and at construction (:72-73);
        // the reader seeds the patch with the table's value at t = 0 and neither mirror arm refreshes it,
        // so a ramp would run frozen at its first value under the case's name. Measured on rhoTP with
        // p0 table ((0 100200) (10 100400)) and the solvers pinned: OpenFOAM's inlet reads p 100302 (mean)
        // at t = 10, the host arm 100140 and the device arm 100090, p0 still 100200 on both. A
        // constant-valued table and `constant` run, because value(0) IS the value at every time. The
        // legacy device driver refreshes p0 per step (gpuRhoSimpleFoam.cu); the mirror does not, yet.
        if (b.type == "uniformTotalPressure" && !b.p0Function1.isConstant())
            throw std::runtime_error(
                "rhoSimpleFoam createFields: p's patch '" + b.name + "' is `uniformTotalPressure` with a "
                "time-varying p0 table. OpenFOAM samples p0(t) every updateCoeffs "
                "(uniformTotalPressureFvPatchScalarField.C:149); the mirror seeds p0 once at t = 0 and never "
                "refreshes it on either arm, so the ramp would run frozen at its first value. Refusing "
                "rather than running a constant p0 under the case's name. A constant p0 (`constant`, a "
                "bare value, or a single-valued table) runs.");
        // totalPressure reads p0 as a FIELD (totalPressureFvPatchScalarField.C:67, `p0_("p0", dict,
        // p.size())`), so a Function1 table there is an OpenFOAM read error that the reader used to seed
        // at t = 0 instead. Refused: OpenFOAM would not have run this case at all.
        if (b.type == "totalPressure")
            throw std::runtime_error(
                "rhoSimpleFoam createFields: p's patch '" + b.name + "' is `totalPressure` with a p0 "
                "table. OpenFOAM's totalPressure reads p0 as a field (totalPressureFvPatchScalarField.C:67) "
                "and would fail to read this entry; a Function1 p0 belongs to `uniformTotalPressure`. "
                "Refusing rather than seeding p0 from the table's first value.");
    }
    f.p = buildField<scalar>(pFd, patches, nC);
    f.p.evaluateBoundary();
    const FieldData<scalar> tFd = guardRead(readField<scalar>(timeDir + "/T"), "T");
    f.T = buildField<scalar>(tFd, patches, nC);
    f.T.evaluateBoundary();

    // T MUST LIE WHERE THE THERMO IS DEFINED, checked before a single property is evaluated from it. A
    // liquid's correlations are fits on [Tt, Tc]; above Tc H2O's rho is a fractional power of a negative
    // number, i.e. a NaN (H2OLiquid::inRange). OpenFOAM has no guard and runs on: measured with H2O on
    // sbMatched's ~1000 K fields, every solve reports `Initial residual = nan` for 1000 iterations and the
    // run dies reading its own output back. brae used to reach the same NaN and refuse three calls later
    // under the wrong name -- "flowRateInletVelocity on patch 'inlet': gSum(rho*magSf) is not positive" --
    // which sends whoever reads it to the inlet instead of the thermo. The gas branch has no range and
    // this block does nothing there.
    {
        const ThermoTRange tr = thermoTRangeOf(f.thermo);
        if (tr.bounded)
        {
            scalar      tMin = f.T.internal.empty() ? scalar(0) : f.T.internal[0];
            scalar      tMax = tMin;
            std::string firstBad;
            scalar      firstBadT = 0;
            auto visit = [&](scalar T, const std::string& where)
            {
                tMin = std::min(tMin, T);
                tMax = std::max(tMax, T);
                if (firstBad.empty() && !thermoTInRange(T, f.thermo))
                {
                    firstBad  = where;
                    firstBadT = T;
                }
            };
            for (label c = 0; c < nC; ++c) visit(f.T.internal[c], "cell " + std::to_string(c));
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                const std::vector<scalar>& tb = f.T.boundary[pi]->value();
                for (std::size_t i = 0; i < tb.size(); ++i)
                    visit(tb[i], "patch '" + patches[pi].name + "' face " + std::to_string(i));
            }
            if (!firstBad.empty())
            {
                char buf[640];
                std::snprintf(buf, sizeof(buf),
                              "brae: rhoSimpleFoam createFields -- T = %.10g K at %s is outside the range "
                              "%s's liquid correlations are defined on, [%.2f, %.2f] K (the triple and "
                              "critical points; the field as read spans %.10g .. %.10g K). Above the "
                              "critical point the density correlation is a fractional power of a negative "
                              "number: OpenFOAM computes the NaN and runs on it. Refusing rather than "
                              "evaluating a liquid's properties at a temperature it does not have.",
                              (double)firstBadT, firstBad.c_str(), tr.substance, (double)tr.lo,
                              (double)tr.hi, (double)tMin, (double)tMax);
                throw std::runtime_error(buf);
            }
        }
    }

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
            fd.internalField[c] = thermoRhoOf(f.p.internal[c], f.T.internal[c], f.thermo);
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
                b.values[i] = thermoRhoOf(pb[i], tb[i], f.thermo);
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
        // ...at the START TIME, the timeOutputValue() OpenFOAM's constructor evaluates the flow rate's
        // Function1 at; a constant rate ignores it, a time-dependent one refuses without it.
        f.U.boundary[pi]->updateAtConstruction(f.rho.boundary[pi]->value(), startTime);
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
    for (label c = 0; c < nC; ++c) f.psi[c] = thermoPsiOf(f.p.internal[c], f.T.internal[c], f.thermo);
    f.psiBnd.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& tb = f.T.boundary[pi]->value();
        const std::vector<scalar>& pb = f.p.boundary[pi]->value();
        f.psiBnd[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            f.psiBnd[pi][i] = thermoPsiOf(pb[i], tb[i], f.thermo);
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
    //                                                        (gradientEnergyFvPatchScalarField.C:99-105).
    //                                                        Under hConst Cpv is the constant Cp (or
    //                                                        Cv = Cp - R for sensibleInternalEnergy), so
    //                                                        the seed below and the live update agree
    //                                                        face for face on the gas path. The
    //                                                        gradient SCALES by Cpv; it must not go
    //                                                        through heOf, which is affine -- applying an
    //                                                        offset to a slope is the trap the mx/hf
    //                                                        controls measure (7.97e-03 on rhoBoxQ)
    //     mixed         -> mixed         with vf unchanged,  mixedEnergy verbatim
    //                      refValue -> he(refValue_T),       (mixedEnergyFvPatchScalarField.C:103-115):
    //                      refGrad  -> Cpv*refGrad_T         same zero second term
    //     inletOutlet   -> inletOutlet  on he(T_inletValue)  exact: mixedEnergy is the same mixed BC with
    //                                                        the energy refValue
    // A multi-species mixture revives the second term; the refusal below is what stops one becoming a
    // silent wrong answer.
    //
    // SEED ONLY, since stage H3.3. What this block writes is he's state at construction -- before the
    // first energy assembly -- and updateEnergyBoundaryCoeffs (energy_boundary.cuh) rebuilds the
    // fixedValue/fixedGradient/mixed coefficients from the live p and T at every assembly thereafter,
    // exactly as OpenFOAM's three energy conditions do. It stays here rather than being deleted because
    // a construction-time he is a real state: `he.evaluateBoundary()` below runs on it, and the patches
    // the live update leaves alone (calculated, and the constraint types) keep it for the whole run.
    // The static mapping and the live one agree number for number on the gas path, which is what makes
    // this stage a no-op there.
    {
        // THE PROPERTY ACCESSOR, not the hConst closed form this block used to inline. thermoHeOf is
        // p-dependent on the liquid arm (Es = h(T) - p/rho(T)), so the seed is now built face by face
        // against p's own boundary values rather than by mapping a dictionary's `uniform` entry, which
        // has no pressure to be evaluated at.
        auto heOf = [&](scalar pAt, scalar T) { return thermoHeOf(pAt, T, f.thermo); };
        FieldData<scalar> heFd;
        heFd.internalUniform = false;
        heFd.internalField.resize(nC);
        for (label c = 0; c < nC; ++c) heFd.internalField[c] = heOf(f.p.internal[c], f.T.internal[c]);
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
            if (tb.type == "uniformFixedValue")
            {
                // heBoundaryTypes dispatches on the CLASS (isA<fixedValueFvPatchScalarField>, basicThermo.C:
                // 203), and uniformFixedValue is one: he gets fixedEnergy, seeded from T's `value` exactly
                // as from a fixedValue's. The uniformValue -- a constant or an `expression` -- belongs to
                // T alone; its result reaches he through updateEnergyBoundaryCoeffs at every assembly.
                b.type          = "fixedValue";
                b.hasUniformFn1 = false;
                b.hasPatchExpr  = false;
                b.patchExpr     = PatchExprSpec{};
            }
            if (tb.type != "fixedValue" && tb.type != "zeroGradient" && tb.type != "inletOutlet"
                && tb.type != "uniformFixedValue"
                && tb.type != "fixedGradient" && tb.type != "mixed"
                && tb.type != "calculated" && tb.type != "empty" && tb.type != "symmetry"
                && tb.type != "symmetryPlane" && tb.type != "wedge" && tb.type != "slip")
                throw std::runtime_error(
                    "brae: rhoSimpleFoam createFields cannot derive he's boundary condition from T's '"
                    + tb.type + "' on patch '" + tb.name + "'. OpenFOAM maps T's boundary conditions to "
                    "ENERGY ones (basicThermo::heBoundaryTypes) and brae implements that mapping only "
                    "where it is exact for perfectGas+hConst. Refusing rather than transporting an "
                    "energy under a temperature's boundary condition.");
            // PER FACE, against this patch's own p and T. A dictionary entry is `uniform 300` or a list;
            // he's is neither once the thermo is p-dependent, so every slot is expanded to the patch's
            // face count and each face converted at its own pressure. On the gas path thermoHeOf ignores
            // p and every face gets the same number, which is the entry this block used to write.
            const std::vector<scalar>& pw = f.p.boundary[pi]->value();
            const std::vector<scalar>& Tw = f.T.boundary[pi]->value();
            const label np = patches[pi].size;
            auto perFace = [&](bool uniform, scalar uval, const std::vector<scalar>& vals)
            {
                std::vector<scalar> r(static_cast<std::size_t>(np));
                for (label i = 0; i < np; ++i)
                    r[static_cast<std::size_t>(i)] =
                        uniform ? uval
                                : (static_cast<std::size_t>(i) < vals.size() ? vals[static_cast<std::size_t>(i)]
                                                                             : scalar(0));
                return r;
            };
            auto mapValues = [&](std::vector<scalar> v)
            {
                for (label i = 0; i < np; ++i)
                    v[static_cast<std::size_t>(i)] = heOf(pw[static_cast<std::size_t>(i)],
                                                          v[static_cast<std::size_t>(i)]);
                return v;
            };
            b.valueUniform     = false;
            b.values           = mapValues(perFace(tb.valueUniform, tb.uniformValue, tb.values));
            b.refValueUniform  = false;                               // mixed's refValue slot
            b.refValues        = mapValues(perFace(tb.refValueUniform, tb.refValueUniformValue, tb.refValues));
            b.inletUniform     = false;
            b.inletValues      = mapValues(perFace(tb.inletUniform, tb.inletUniformValue, tb.inletValues));
            // The GRADIENT slots (fixedGradient's `gradient`, mixed's `refGradient`) SCALE by Cpv --
            // never heOf, which is affine: an offset applied to a slope was worth 7.97e-03 on rhoBoxQ's
            // T when the mx/hf controls first measured it. Cpv comes from thermoCpvOf at this face's own
            // p and T rather than from the hConst constant, which is the same number on the gas path and
            // a correlation on the liquid one.
            if (tb.hasGradient)
            {
                b.gradientUniform = false;
                b.gradientValues  = perFace(tb.gradientUniform, tb.gradientUniformValue, tb.gradientValues);
                for (label i = 0; i < np; ++i)
                    b.gradientValues[static_cast<std::size_t>(i)] *=
                        thermoCpvOf(pw[static_cast<std::size_t>(i)], Tw[static_cast<std::size_t>(i)], f.thermo);
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
        const std::string dictPath = turbulenceDictPath(caseDir);
        if (dictPath.empty())
            throw std::runtime_error(
                "brae: rhoSimpleFoam found neither constant/momentumTransport nor "
                "constant/turbulenceProperties. OpenFOAM constructs the turbulence model from one of "
                "them in createFields.H; refusing rather than assuming the case is laminar.");
        std::unique_ptr<FoamDict> ownMt;
        if (!turbDict) ownMt = std::make_unique<FoamDict>(readDict(dictPath));
        const FoamDict& mt2 = turbDict ? *turbDict : *ownMt;
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
            // optionalSubDict, as RASModel.C:72: a flat `RAS { Cmu 0.12; }` without kEpsilonCoeffs reaches
            // OpenFOAM's model, and reached brae's defaults until it did here (queue item 21).
            if (const FoamDict* kec = ras ? ras->optionalSubDict("kEpsilonCoeffs") : nullptr)
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
            // Foam::bound's floors, from the RAS TOP LEVEL, through the one reader all three of brae's
            // turbulenceProperties parsers call (turbulence_setup.cuh, readTurbulenceMinima).
            readTurbulenceMinima(ras, keCase.kMin, keCase.epsilonMin, f.sstCoeffs.omegaMin);
            f.keCoeffs = keCase;
            readKOmegaSSTCoeffs(ras, f.sstCoeffs);   // OpenFOAM's defaults where the dict is absent
            f.sstCoeffs.kMin = keCase.kMin;          // ...after, so the SST reader cannot overwrite it
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
                    // Default Nutk for every patch: a patch with no wall function never reaches the
                    // dispatch (the loop is gated on the epsilon patch being a wall function), and a
                    // wall-function patch whose type is unrecognised throws below rather than falling
                    // through to a default -- a permissive default here is how nutk gets run under
                    // another name, which is the whole defect.
                    f.nutWallKind.assign(patches.size(), static_cast<int>(NutWall::Nutk));
                    const FieldData<scalar> nutRaw = guardRead(readField<scalar>(timeDir + "/nut"), "nut");
                    for (const auto& b : nutRaw.boundary)
                    {
                        // TWO prefixes, because the atm family does not start with "nut":
                        // atmNutkWallFunction -- named in the comment above as refused -- passed a
                        // one-prefix test and ran as the smooth nutk with z0 unplumbed, exactly the
                        // substitution this throw exists to prevent.
                        if (b.type.rfind("nut", 0) != 0
                         && b.type.rfind("atmNut", 0) != 0) continue;        // not a nut wall function
                        // THE FAMILY brae now dispatches, captured HERE because this is the last
                        // place the dictionary TYPE exists -- once buildField has run the patch object
                        // no longer carries it, which is why neither closure could have checked. The
                        // three are different functions of different inputs, and OpenFOAM picks between
                        // them at one virtual call (nutWallFunctionFvPatchScalarField.C:182).
                        int kind = -1;
                        if      (b.type == "nutkWallFunction")     kind = static_cast<int>(NutWall::Nutk);
                        else if (b.type == "nutUWallFunction")     kind = static_cast<int>(NutWall::NutU);
                        else if (b.type == "nutLowReWallFunction") kind = static_cast<int>(NutWall::LowRe);
                        if (kind >= 0)
                        {
                            // Ported for the LIVE closure, which recomputes the wall nut every
                            // iteration. Frozen, OpenFOAM evaluates it exactly ONCE -- inside
                            // validate()'s correctBoundaryConditions -- and brae has no wall-function
                            // evaluation at createFields, so the wall would keep the file value.
                            if (!f.turbulenceFrozen)
                            {
                                // Through the SHARED resolution, not `entry.name == patch.name`. A case
                                // may key its wall entry by regex or by group -- gasMixing uses
                                // `"wall.*"` for walls_pipe_{air,fuel,main} -- and an exact match then
                                // assigns NOTHING, leaving every one of those patches on the permissive
                                // Nutk default seeded above. That is this block's own stated failure
                                // mode ("a permissive default here is how nutk gets run under another
                                // name"), and the type check cannot catch it because the type IS
                                // recognised; only the name match fails. Measured on gasMixing:
                                // OpenFOAM's wall nut is exactly 0 at iteration 1 (magUp = 0 from a
                                // still start puts nutU in its viscous branch) where brae wrote nutk's
                                // 1.4e-04 from k = 6, worth U 1.7e-03 at iteration 1 and 1.2e-01 by
                                // iteration 5. patch_entry_lookup.cuh's header lists the two earlier
                                // instances of exactly this.
                                for (const FvPatch* pp : patchesResolvingTo(nutRaw.boundary, b, patches))
                                    f.nutWallKind[static_cast<std::size_t>(pp - patches.data())] = kind;
                                continue;
                            }
                            throw std::runtime_error(
                                "brae: rhoSimpleFoam `RAS { turbulence off; }` with a '" + b.type + "' "
                                "patch ('" + b.name + "') is not implemented -- OpenFOAM wall-evaluates "
                                "nut once at validate() and brae cannot yet do that outside the live "
                                "closure. Use a calculated/zeroGradient nut boundary, or turbulence on.");
                        }
                        // Still refused, and by name: nutUSpalding and nutUBlended have shared
                        // kernels but both carry known defects against OpenFOAM (the blended form takes
                        // the full |U_cell - U_wall| where OpenFOAM projects out the normal component,
                        // and both seed the iteration from 0 where OpenFOAM seeds from the stored nut),
                        // and atmNutk's z0 is a per-face PatchFunction1 where brae carries one scalar.
                        // A shared kernel is not a port until it agrees.
                        throw std::runtime_error(
                            "brae: rhoSimpleFoam nut patch '" + b.name + "' carries '" + b.type +
                            "', which the compressible kEpsilon closure does not implement. It has "
                            "nutkWallFunction, nutUWallFunction and nutLowReWallFunction; the rest of "
                            "the family are different functions of different inputs (nutUSpalding and "
                            "nutUBlended iterate on |U| with a warm start brae does not carry; atm's z0 "
                            "is a per-face field where brae has one scalar), so substituting one would "
                            "converge to a different wall viscosity and a different epsilon. Refusing "
                            "rather than running one wall function under another's name.");
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
        else if (sim == "laminar")
        {
            // THE laminar{} SUB-DICTIONARY, which this function never opened. `simulationType laminar`
            // was accepted by falling through, so a case selecting a generalizedNewtonian or Maxwell
            // viscosity ran with the molecular one -- silently, and on both mirror arms, since the CUDA
            // arm builds its fields from this same function. OpenFOAM's generalizedNewtonian nuEff()
            // RETURNS the model's nu instead of adding to it, so the mirror was not approximating the
            // model; it was solving a different momentum equation. Measured on validation/rhoBox with
            // squareBendLiqNoNewtonian's own laminar block: 5.74e-01 relative on U against OpenFOAM.
            //
            // generalizedNewtonian with the powerLaw viscosity is APPLIED (stage S1 Half B); every other
            // viscosity model and Maxwell are still refused by name inside the shared reader.
            DeviceSimpleControls lctl;
            lctl.turbulent = false;
            readLaminarModel(mt2, lctl, {"rhoSimpleFoam (OF-mirror)", true, false});
            if (lctl.gnPowerLaw)
            {
                f.generalizedNewtonian = true;
                f.gnCoeffs.n     = lctl.gnN;
                f.gnCoeffs.nuMin = lctl.gnNuMin;
                f.gnCoeffs.nuMax = lctl.gnNuMax;
                // strainRate() is fvc::grad(this->U()) -> mesh.gradScheme("grad(U)"), named-then-default.
                // Resolved strictly: the gradient is Gauss linear with an optional cellLimited limiter,
                // and anything else is refused rather than approximated by the nearest one brae has.
                const FieldGradScheme gs = parseNamedGradScheme(caseDir, "grad(U)");
                if (!gs.gaussLinear || gs.leastSquares || !gs.unsupportedLimiter.empty())
                    throw std::runtime_error(
                        "brae: rhoSimpleFoam (OF-mirror) -- generalizedNewtonian's strainRate() takes "
                        "fvc::grad(U) (generalizedNewtonian.C:98), and gradSchemes resolves grad(U) to `" +
                        gs.raw + "`. The mirror computes Gauss linear, optionally cellLimited; refusing "
                        "rather than building the viscosity from another gradient.");
                f.gnGradULimitK = gs.cellLimitK;
                // The model's CONSTRUCTOR (generalizedNewtonian.C:87): nu_ from the initial U, rho and T,
                // with U's patches as construction left them -- the same boundary values phi was built
                // from above. On a case starting from rest this is nuMax in every cell grad(U) cannot
                // reach yet, and iteration 1's momentum equation runs on it.
                correctGeneralizedNewtonian(f, m, g, patches);
            }
        }
        else
        {
            throw std::runtime_error(
                "brae: rhoSimpleFoam simulationType '" + sim + "' is neither laminar nor RAS. Refusing.");
        }
    }

    // Foam::bound(k_, kMin_) and bound(<second>, <second>Min_), from the MODEL CONSTRUCTOR --
    // kEpsilon.C:182-183, kOmegaSSTBase.C:438-439, realizableKE.C:211-212. OpenFOAM bounds the two
    // transported scalars the instant it has read them, so validate()'s correctNut below already sees
    // bounded fields and the first momentum matrix carries a nut built from them. brae bounded only
    // inside correct(), so a case whose 0/k or 0/epsilon dips under the floor entered iteration 1 with
    // the file's value: at `RAS { epsilonMin 5000; }` on rhoKE OpenFOAM reports `bounding epsilon` before
    // its first "Time =" line and brae reported nothing, because the first bound() it ran was after the
    // first epsilon solve had already been assembled from the sub-floor field.
    //
    // ORDER IS OpenFOAM'S: k first, then the second scalar. bound() takes an area-weighted neighbour
    // average for a NEGATIVE cell, so bounding epsilon first would feed the k pass a different field.
    // SpalartAllmaras is deliberately absent -- its bound(nuTilda_, 0) lives only in correct()
    // (SpalartAllmarasBase.C:487), there is none in the constructor.
    // BRAE_CTOR_BOUND=0 skips it -- the fail-proof for
    // tests/bound_at_construction_vs_openfoam.sh, on both mirror arms at once since the CUDA arm
    // uploads the field set this function returns.
    const char* ctorBoundEnv = std::getenv("BRAE_CTOR_BOUND");
    const bool  ctorBound    = !(ctorBoundEnv && std::string(ctorBoundEnv) == "0");
    if (ctorBound && f.turbulent && static_cast<label>(f.k.internal.size()) == nC)
    {
        cpu::bound(f.k, keCase.kMin, m, g, patches, "k");
        if (f.rasModel == "kOmegaSST")
        {
            if (static_cast<label>(f.omega.internal.size()) == nC)
                cpu::bound(f.omega, f.sstCoeffs.omegaMin, m, g, patches, "omega");
        }
        else if (static_cast<label>(f.epsilon.internal.size()) == nC)
        {
            cpu::bound(f.epsilon, keCase.epsilonMin, m, g, patches, "epsilon");
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
            nuLam[c] = thermoMuOf(f.p.internal[c], f.T.internal[c], f.thermo) / f.rho.internal[c];
        std::vector<std::vector<scalar>> nuLamBnd(patches.size()), rhoBnd(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& tb = f.T.boundary[pi]->value();
            const std::vector<scalar>& pb = f.p.boundary[pi]->value();
            const std::vector<scalar>& rb = f.rho.boundary[pi]->value();
            rhoBnd[pi] = rb;
            nuLamBnd[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                nuLamBnd[pi][i] = thermoMuOf(pb[i], tb[i], f.thermo) / rb[i];
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
            cpu::kEpsilonRef::NutWallSelection nsel;
            nsel.kind = &f.nutWallKind;
            nsel.U    = &f.U;
            cpu::kEpsilonRef::correctNutField(f.k, f.epsilon, f.nut, yWall, /*nu=*/0.0, patches, keCase,
                                              &comp, &nsel);
        }
        else
        {
            const std::vector<scalar> y     = cellWallDist(m, g, patches);
            // validate()'s correctNut takes fvc::grad(U) through the case's grad(U) scheme (kOmegaSSTBase.C:132).
            // grad(U)'s own gradSchemes entry, base scheme then limiter -- the same resolution the step
            // makes (rhoSimpleFoamDriver_cpp.cu). The shared parser only WARNS on leastSquares and hands
            // out the Gauss coefficient, so the base scheme is read from parseFieldGradScheme here too.
            const bool gradULsq = parseFieldGradScheme(caseDir, "U").leastSquares;
            std::vector<tensor> gradU = gradULsq ? fvc::leastSquaresGrad(f.U, m, g, patches)
                                                 : fvc::gaussGrad(f.U, m, g, patches);
            {
                DeviceSimpleControls sctl;
                parseFvSchemesControls(caseDir, sctl);
                if (sctl.gradULimitK > 0.0) cpu::cellLimitGrad(gradU, f.U, sctl.gradULimitK, m, g, patches);
            }
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

    // An `expression` PatchFunction1 is evaluated by the step on T ALONE, inside the energy conditions'
    // updateCoeffs where OpenFOAM's fixedEnergy evaluates T's patch. On any other field OpenFOAM evaluates
    // it in that field's own updateCoeffs, which no brae step reproduces, and the patch would stay at its
    // `value` for the whole run -- a converged answer under a frozen boundary. Refused by name. U is
    // refused earlier still, by the factory (the evaluator carries vectors no further than mag()).
    auto refuseExpressionOn = [&](const GeometricField<scalar>& fld, const std::string& nm)
    {
        for (std::size_t pi = 0; pi < fld.boundary.size(); ++pi)
        {
            if (!fld.boundary[pi]->patchExpression()) continue;
            throw std::runtime_error(
                "brae: " + nm + " on patch '" + patches[pi].name + "' has a uniformFixedValue `expression`. "
                "brae evaluates that PatchFunction1 only on T, at the energy assembly where OpenFOAM's "
                "fixedEnergy::updateCoeffs evaluates T's patch; on " + nm + " OpenFOAM evaluates it in " + nm
                + "'s own updateCoeffs, which no brae step reproduces. Refusing rather than running the "
                "patch frozen at its `value`.");
        }
    };
    refuseExpressionOn(f.p, "p");
    refuseExpressionOn(f.rho, "rho");
    refuseExpressionOn(f.he, f.heName);
    refuseExpressionOn(f.k, "k");
    refuseExpressionOn(f.epsilon, "epsilon");
    refuseExpressionOn(f.omega, "omega");
    refuseExpressionOn(f.nut, "nut");
    refuseExpressionOn(f.alphat, "alphat");

    return f;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
