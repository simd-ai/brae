#pragma once
// rho_simple_controls.cuh -- the controls rhoSimpleFoam needs on top of the incompressible ones.
//
// Deliberately small. Everything the steady incompressible solver already reads (relaxation factors,
// residualControl, schemes, linear-solver selection) is unchanged and stays in SolverControls; this holds
// only what has no incompressible meaning.
//
// The compressible loop composes the SAME phases the steady and PIMPLE solvers use
// (solveMomentumPredictor -> correctPressureVelocity -> correctTurbulence) with two inserted steps:
//
//     UEqn  ->  EEqn  ->  pEqn  ->  thermo.correct()  ->  rho.relax()  ->  turbulence
//
// which is OF's rhoSimpleFoam.C order. The energy equation sits between momentum and pressure because
// the pressure equation needs the density that the new temperature implies, not the previous one.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "thermo_types.cuh"
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// A SIMPLE-dictionary Switch, read the way OpenFOAM reads it. solutionControl.C:50-51 reads `transonic`
// and `consistent` with getOrDefault<bool>, and a bool is read through Foam::Switch (bool.C:53-57):
// true/yes/on/y/t/1/any and false/no/off/n/f/0/none, and `Unknown switch` FatalError on any other word
// (Switch.C:130-136). FoamDict::switchOr is that table. Both rhoSimpleFoam drivers compared the word with
// "yes" alone, so `transonic true;` ran the subsonic branch and `consistent on;` ran SIMPLE on the mirror,
// with nothing said. Absent: false, OpenFOAM's default.
inline bool readSimpleSwitch(const FoamDict* simple, const char* key, const char* where)
{
    if (!simple) return false;
    try
    {
        return simple->switchOr(key, false);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(
            std::string("brae: ") + where + ": system/fvSolution SIMPLE/" + key + " is '"
            + simple->wordOr(key, "") + "', which is not a Switch. OpenFOAM reads it as one "
            "(solutionControl.C:50-51; true/false, yes/no, on/off, y/n, t/f, 1/0) and stops on any other word "
            "(Switch.C:130-136). Refusing rather than guessing which branch the case meant; spell it as one "
            "of those words. (" + e.what() + ")");
    }
}

struct RhoSimpleControls
{
    // OF pEqn.H's transonic branch, selected by SIMPLE/transonic (a Switch; see readSimpleSwitch).
    bool transonic = false;
    // heRhoThermo: OF's `rho = thermo.rho()` after the pressure solve returns the STORED rho_, so rho lags
    // the just-solved pressure by one outer iteration. hePsiThermo returns p_*psi_ and does not. Set from
    // ThermoCoeffs::rhoThermoType by the driver.
    bool rhoLagsPressure = false;

    // OF pressureControl: the cell whose pressure is pinned when the domain is closed. Unused when a
    // boundary fixes p, which is the common external-aero case.
    //
    // What the case SAID is kept beside it, because OpenFOAM's setRefCell (findRefCell.C:31-115) decides
    // from it: pRefCell, else pRefPoint (located with mesh.findCell), else a FatalIOError -- and pRefValue
    // is a mandatory readEntry. The 0 / 0.0 below are the old silent defaults; the default driver
    // resolves the reference from the has* flags instead (gpuRhoSimpleFoam.cu, resolvePRefCell) and
    // refuses where OpenFOAM stops.
    int pRefCell = 0;
    scalar pRefValue = 0.0;
    bool hasPRefCell = false;
    bool hasPRefPoint = false;
    bool hasPRefValue = false;
    std::vector<scalar> pRefPoint;   // the entry's numbers as read (a point has three)

    // OF pressureControl::limit(p) -- p is clamped to [pMin, pMax] AFTER the pressure solve and before
    // rho is recomputed (rhoSimpleFoam/pEqn.H:90). This is a real stabiliser, not a safety net: the
    // stock aerofoilNACA0012 tutorial runs at Mach 0.72 and ships pMinFactor/pMaxFactor precisely
    // because the pressure iterate overshoots early on. A case that specifies the factors and does not
    // get the clamp can diverge where OF converges.
    //
    // OF accepts either absolute pMax/pMin or pMaxFactor/pMinFactor, the latter scaled by a reference
    // pressure taken from the p boundary values (pressureControl.C). Both are read here.
    bool   limitMaxP = false;
    bool   limitMinP = false;
    scalar pMaxLimit = 0.0;
    scalar pMinLimit = 0.0;
    scalar pMaxFactor = 0.0;
    scalar pMinFactor = 0.0;

    // Energy equation relaxation, the factor EEqn.relax() applies. 0 means NO relax() call at all.
    //
    // OpenFOAM's fvMatrix::relax() (fvMatrix.C:1250-1263) relaxes only when
    // solution::relaxEquation(name, factor) finds the FIELD'S OWN name -- "h" under sensibleEnthalpy, "e"
    // under sensibleInternalEnergy -- or a `default` entry in relaxationFactors/equations
    // (solution.C:379-418); otherwise the matrix is left exactly as assembled. This was 0.7 whenever the
    // case named neither, and it also took the OTHER energy name ("h" for an "e" case), which OpenFOAM
    // never looks at. Measured on validation/rhoBox with the factor removed: brae relaxed at 0.7 and
    // landed 2.6e-02 from OpenFOAM's T at iteration 1 (tests/rho_energy_relax_vs_openfoam.sh).
    //
    // 0 rather than 1.0 is deliberate: relax(1.0) is not a no-op in OpenFOAM -- it still enforces
    // diagonal dominance (fvMatrix.C relax(alpha)) -- and deviceSolveScalarTransport takes alpha <= 0 as
    // "relax() not called", the branch fvMatrix::relax's `if (alpha <= 0) return;` describes.
    scalar relaxHe = 0.0;
    std::string relaxHeFrom;   // where the factor came from, for the log: the key, "default", or empty
};

// The relaxationFactors dictionary OpenFOAM reads EQUATION factors from (solution.C:60-106): the
// `equations` sub-dictionary when either `fields` or `equations` is present, and the whole
// relaxationFactors dictionary in the older flat form (the backwards-compatible branch, which copies it
// into eqnRelaxDict_ verbatim). With `fields` and no `equations` there are no equation factors at all.
inline const FoamDict* equationRelaxDict(const FoamDict& fvSolution)
{
    const FoamDict* relax = fvSolution.subDict("relaxationFactors");
    if (!relax) return nullptr;
    const FoamDict* eqns = relax->subDict("equations");
    if (eqns) return eqns;
    if (relax->subDict("fields")) return nullptr;
    return relax;
}

inline RhoSimpleControls readRhoSimpleControls(
    const FoamDict& fvSolution,
    bool internalEnergy = false)
{
    RhoSimpleControls rc;

    const FoamDict* simple = fvSolution.subDict("SIMPLE");
    if (simple)
    {
        rc.transonic = readSimpleSwitch(simple, "transonic", "rhoSimpleFoam");
        rc.hasPRefCell = simple->found("pRefCell");
        rc.hasPRefPoint = simple->found("pRefPoint");
        rc.hasPRefValue = simple->found("pRefValue");
        rc.pRefCell = simple->intOr("pRefCell", rc.pRefCell);
        rc.pRefValue = simple->scalarOr("pRefValue", rc.pRefValue);
        if (rc.hasPRefPoint) rc.pRefPoint = simple->scalarListOr("pRefPoint", {});

        // Absolute limits win over the factors, matching OF's precedence (pMax/pMin read first).
        if (simple->found("pMax")) { rc.pMaxLimit = simple->scalarOr("pMax", 0.0); rc.limitMaxP = true; }
        else if (simple->found("pMaxFactor")) rc.pMaxFactor = simple->scalarOr("pMaxFactor", 0.0);
        if (simple->found("pMin")) { rc.pMinLimit = simple->scalarOr("pMin", 0.0); rc.limitMinP = true; }
        else if (simple->found("pMinFactor")) rc.pMinFactor = simple->scalarOr("pMinFactor", 0.0);
    }

    // The energy factor, by the energy field's own name, else `default`, else none (see relaxHe above).
    // FoamDict::find is OpenFOAM's lookup: the literal key, else the last matching pattern
    // (`"(U|e)" 0.7;` relaxes e), as solution::relaxEquation's REGEX search does.
    if (const FoamDict* eqns = equationRelaxDict(fvSolution))
    {
        const std::string name = internalEnergy ? "e" : "h";
        if (eqns->found(name))
        {
            rc.relaxHe = eqns->scalarOr(name, rc.relaxHe);
            rc.relaxHeFrom = name;
        }
        else if (eqns->found("default"))
        {
            rc.relaxHe = eqns->scalarOr("default", rc.relaxHe);
            rc.relaxHeFrom = "default";
        }
    }

    return rc;
}

} // namespace brae
