#pragma once
// brae::FvOptions, OpenFOAM fvOptions framework (system/fvOptions or constant/fvOptions).
//
// Mirrors fv::options::New(mesh): if the file is present its option list is read, otherwise the list is EMPTY and
// every equation hook is a no-op, bit-identical to a case with no fvOptions (OF's "No finite volume options
// present"). The solver calls the hooks UNCONDITIONALLY (== fvOptions(U), fvOptions.constrain, fvOptions.correct,
// + fvOptions(k|epsilon)) exactly like UEqn.H / kEpsilon.C; emptiness is handled here, not at the call sites.
//
// Implemented sources (the generic, dict-groundable ones):
//   vectorSemiImplicitSource / scalarSemiImplicitSource, explicit Su [+ implicit Sp] on a cell selection,
//   volumeMode {specific|absolute} (OF: SuValue = Su/VDash, VDash = absolute? V_selection : 1; eqn += Su*V[cell],
//   implicit via fvm::Sp(Sp,psi) -> diag -= Sp*V[cell]). selectionMode {all|cellZone}.
//
// The per-cell explicit/implicit contributions are precomputed on the host (steady -> constant) and the solver
// adds them with deviceAxpy; no new device kernel is needed.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "cell_selection.cuh"   // OF cellSetOption: all|cellZone|cellSet resolved in ONE place
#include "rotor_disk.cuh"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace brae {

// A Function1 entry's leading token is either the function's NAME or, in OF's bare shorthand, the value
// itself. Telling those apart is all that separates `Cp 0.386;` from `Cp table (...)`.
inline bool isFoamScalarToken(const std::string& t)
{
    try
    {
        std::size_t n = 0;
        (void)std::stod(t, &n);
        return n == t.size();
    }
    catch (...) { return false; }
}

struct FvOptionsData
{
    // momentum (vectorSemiImplicitSource on field "U"): explicit per-component source Su*V/VDash, implicit Sp*V.
    bool                hasMomentum = false;
    std::vector<scalar> momSu[3];        // per cell, += into relaxSrc[comp]  (0 outside selection)
    std::vector<scalar> momSp;           // per cell, -= into momentum diagonal (0 outside selection / no Sp)
    // scalar sources (scalarSemiImplicitSource), keyed by field name (k, epsilon, omega, ...).
    std::map<std::string, std::vector<scalar>> scaSu, scaSp;
    // explicitPorositySource (DarcyForchheimer): a per-iteration momentum resistance on a cellZone. NOT precomputable
    // (the Forchheimer term depends on |U|), so we store the zone cells + the adjusted d/f coefficients and the
    // solver evaluates the resistance each iteration. Diagonal D/F only (identity coordinateSystem, simpleCar).
    bool                porActive = false;
    std::vector<label>  porCells;
    vector              porD{0,0,0}, porF{0,0,0};   // adjusted Darcy d [1/m^2] + Forchheimer f [1/m]
    // porosityModels::fixedCoeff -- a SECOND model, not a variant. alpha [1/s], beta [1/m], both passed
    // through adjustNegativeResistance, then diag(alpha)/diag(beta) rotated into the coordinateSystem's
    // frame (fixedCoeff.C calcTransformModelData). Row-major 3x3.
    bool                porFixed = false;
    scalar              porAlphaT[9] = {0,0,0,0,0,0,0,0,0};
    scalar              porBetaT[9]  = {0,0,0,0,0,0,0,0,0};
    scalar              porRhoRef = 1.0;
    // meanVelocityForce: a uniform body force adjusted each iter to drive the mean velocity to Ubar (channel flow).
    bool                mvfActive = false;
    vector              mvfUbar{0,0,0};
    scalar              mvfRelax = 1.0;
    std::vector<label>  mvfCells;                   // empty => all cells (selectionMode all)
    // limitVelocity (fvConstraint): clamp |U| <= max on a selection, post-solve (aero/atmospheric stabiliser).
    bool                limUActive = false;
    scalar              limUMax = 0;
    std::vector<label>  limUCells;                  // empty => all cells
    // fixedTemperatureConstraint (fvConstraint, mode uniform): pin T on a cell selection by CONSTRAINING the
    // energy matrix -- OF does `eqn.setValues(cells_, thermo.he(thermo.p(), Tuni, cells_))`, i.e. the same
    // matrix manipulation the epsilon wall function uses (zero the row's off-diagonals, force the value).
    // Clamping he after the solve instead would be a different operator: the neighbours' rows would still
    // carry the unconstrained coupling.
    // scalarFixedValueConstraint: OF FixedValueConstraint reads a `fieldValues` sub-dict, one entry per
    // field name (FixedValueConstraint.C read()), and constrains each with eqn.setValues.
    std::map<std::string, scalar> fixScaVals;   // field name -> fixed value
    std::vector<label>            fixScaCells;

    bool                fixTActive = false;
    scalar              fixTTemp = 0;               // `temperature` [K]; converted to he by the solver's thermo
    std::vector<label>  fixTCells;

    // limitTemperature (fvConstraint): clamp T into [Tmin, Tmax] on a selection, applied through the ENERGY
    // variable. OF's limitTemperature::correct(he) converts the two temperature limits to he limits with the
    // case's own thermo (heMin = thermo.he(p, Tmin, cells)) and clamps he, NOT T -- he is what the equation
    // solved, so clamping T and converting back would leave he and T inconsistent for the rest of the
    // iteration. It then clamps the he BOUNDARY too on every patch that does not fix a value, but only when
    // the selection is the whole mesh (limitTemperature.C: `if (!cellSetOption::useSubMesh())`).
    bool                limTActive = false;
    scalar              limTMin = 0, limTMax = 0;
    std::vector<label>  limTCells;                  // empty => all cells
    bool                limTAllCells = false;       // selectionMode all -> also clamp the he boundary
    // velocityDampingConstraint (fvConstraint): implicit diagonal sink diag += C*V^(2/3)*(|U|-UMax) where |U|>UMax.
    bool                vdcActive = false;
    scalar              vdcUMax = 0, vdcC = 1;
    std::vector<label>  vdcCells;                   // empty => all cells
    // actuationDiskSource (Froude): wind-turbine/propeller momentum source over a disk cellZone, the thrust computed
    // each iter from the upstream-monitored velocity. T = 2*rho*A*(Uref.diskDir)^2*a*(1-a), a = 1 - Cp/Ct.
    // actuationDiskSource: a LIST. A wind farm is several turbines (simpleFoam/turbineSiting ships two),
    // and each is fully independent -- its own monitor cells, its own Uref, its own thrust, its own disk
    // cells. Their momentum sinks simply superpose. brae kept only the first and refused the rest, so a
    // two-turbine case could not run at all.
    //
    // The masks MUST be per disk. Sharing one mask would apply every turbine's thrust to the union of
    // their cells -- a plausible-looking wrong wind field, which is the failure mode this codebase
    // refuses on principle.
    struct ActuationDisk
    {
        vector             diskDir{1,0,0};
        scalar             area = 0, a = 0;       // diskArea, induction a = 1 - Cp/Ct
        std::vector<label> diskCells, monitorCells;
    };
    bool                       adActive = false;
    std::vector<ActuationDisk> adDisks;
    RotorDiskParams     rotor;                      // rotorDiskSource (BEM); geometry built later from the mesh
    int  count = 0;                      // number of options read (0 => no file / empty => all hooks no-op)
    // OF fv::cellSetOption's TIME WINDOW: a source is active only while
    //     timeStart <= t <= timeStart + duration
    // and unconditionally when `timeStart` is absent (OF's default timeStart_ = -1, see
    // cellSetOption::inTimeLimits). The STEADY solver has no time and ignores these; a TRANSIENT driver
    // cannot, because applying a windowed source outside its window is wrong physics that still
    // converges. Recorded per source so the driver can decide.
    struct TimeWindow { std::string name; scalar start = -1; scalar duration = 0; };
    std::vector<TimeWindow> windows;   // only sources that actually SET timeStart
    // Sources whose NAME brae recognizes but that it CANNOT apply here (unsupported selectionMode cellSet, a
    // parsed-but-unapplied scalarSemiImplicitSource, or an unknown type). Recorded instead of silently dropped;
    // the single-GPU driver throws on a non-empty list so a valid-looking case never runs with wrong physics. The
    // parallel path (readFvOptions is shared) ignores this field, so its behaviour is unchanged.
    std::vector<std::string> unsupported;
    bool empty() const { return count == 0; }
};

namespace fvoptions_detail {

// Pull the (Su, Sp) numbers from an injectionRateSuSp field entry: vector form "U ((sx sy sz) sp)" -> [sx,sy,sz,sp];
// scalar form "k (su sp)" -> [su,sp]. FoamDict::scalarListOr already extracts all numeric tokens in order.
inline std::vector<scalar> suSpNumbers(const FoamDict& inj, const std::string& field)
{
    return inj.scalarListOr(field, {});
}

// Resolve an option's coeffs dict: OF accepts the option dict itself OR a nested "<type>Coeffs" sub-dict.
inline const FoamDict& coeffsOf(const FoamDict& opt, const std::string& type)
{
    if (const FoamDict* c = opt.subDict(type + "Coeffs")) return *c;
    return opt;
}

} // namespace fvoptions_detail

// Read fvOptions for a case. `zones` = cellZone name -> cells (readCellZones). `V` = cell volumes. Absent file -> empty.
// Dict-only PRE-FLIGHT: the source types, their time windows, and whether a semi-implicit source has any
// numbers to read -- all without a mesh. readFvOptions needs cell volumes and so can only run once the
// polyMesh is in, which is far too late to refuse a case: the driver should stop before paying for the
// mesh, and a guard fixture without one should still see the refusal it is testing for.
struct FvOptionsPreflight
{
    bool present = false;
    std::vector<std::string> unsupported;
    std::vector<FvOptionsData::TimeWindow> windows;
};

inline FvOptionsPreflight preflightFvOptions(const std::string& caseDir)
{
    FvOptionsPreflight pf;
    std::string path = caseDir + "/system/fvOptions";
    {
        std::ifstream f(path);
        if (!f.good())
        {
            path = caseDir + "/constant/fvOptions";
            std::ifstream g(path);
            if (!g.good()) return pf;
        }
    }
    pf.present = true;
    const FoamDict d = readDict(path);
    for (const auto& s : d.subs)
    {
        const FoamDict& opt = s.second;
        const std::string type = opt.wordOr("type", "");
        const std::string act  = opt.wordOr("active", "yes");
        if (!(act == "yes" || act == "true" || act == "on" || act == "1")) continue;
        static const char* supported[] = {
            "vectorSemiImplicitSource", "explicitPorositySource", "meanVelocityForce", "limitVelocity",
            "actuationDiskSource", "rotorDisk", "rotorDiskSource", "velocityDampingConstraint",
            "limitTemperature", "fixedTemperatureConstraint", "scalarFixedValueConstraint" };
        bool ok = false;
        for (const char* t : supported) if (type == t) { ok = true; break; }
        if (!ok)
        {
            pf.unsupported.push_back("source '" + s.first + "' has unsupported type '" + type + "'");
            continue;
        }
        const FoamDict& co = fvoptions_detail::coeffsOf(opt, type);
        if (type == "vectorSemiImplicitSource"
            && !co.subDict("injectionRateSuSp") && !opt.subDict("injectionRateSuSp")
            && !co.subDict("sources")           && !opt.subDict("sources"))
            pf.unsupported.push_back(
                "source '" + s.first + "' (" + type + ") has neither a `sources` nor an "
                "`injectionRateSuSp` sub-dictionary, so brae found nothing to apply");
        const scalar ts = opt.scalarOr("timeStart", -1e300);
        if (ts > -1e299 && ts >= scalar(0))
            pf.windows.push_back({s.first, ts, opt.scalarOr("duration", 0.0)});
    }
    return pf;
}


inline FvOptionsData readFvOptions(
    const std::string& caseDir,
    const std::map<std::string, std::vector<label>>& zonesIn,
    const std::vector<scalar>& V,
    label nCells,
    const std::vector<vector>& cellCentres = {})
{
    FvOptionsData fo;
    // Local copy: a cellSet resolved below is adopted into this map under its own name, so every
    // source branch can look it up exactly as it looks up a cellZone -- one lookup path, both modes.
    std::map<std::string, std::vector<label>> zones = zonesIn;
    const std::string polyMeshDir = caseDir + "/constant/polyMesh";
    std::string path = caseDir + "/system/fvOptions";
    {
        std::ifstream f(path);
        if (!f.good())
        {
            path = caseDir + "/constant/fvOptions";
            std::ifstream g(path);
            if (!g.good()) return fo;
        }
    }
    const FoamDict d = readDict(path);

    auto ensure = [&](std::vector<scalar>& v) { if (v.empty()) v.assign(nCells, 0.0); };

    // OF porosityModel::adjustNegativeResistance: a negative component -> val*(-maxComponent) (strong resistance).
    auto adjustNeg = [](vector v) -> vector
    {
        const scalar mx = std::max(v.x, std::max(v.y, v.z));
        scalar* c = &v.x;
        for (int i = 0; i < 3; ++i)
            if (c[i] < 0) c[i] *= -mx;
        return v;
    };
    for (const auto& s : d.subs)
    {
        const FoamDict& opt = s.second;
        const std::string type = opt.wordOr("type", "");
        const bool isVec = (type == "vectorSemiImplicitSource");
        const bool isSca = (type == "scalarSemiImplicitSource");
        const bool isPor = (type == "explicitPorositySource");
        const bool isMvf = (type == "meanVelocityForce");
        const bool isLim = (type == "limitVelocity");
        const bool isAd  = (type == "actuationDiskSource");
        const bool isRot = (type == "rotorDisk" || type == "rotorDiskSource");
        const bool isVdc = (type == "velocityDampingConstraint");
        const bool isLimT = (type == "limitTemperature");
        const bool isFixT = (type == "fixedTemperatureConstraint");
        const bool isFixS = (type == "scalarFixedValueConstraint");
        const std::string act = opt.wordOr("active", "yes");
        if (!(act == "yes" || act == "true" || act == "on" || act == "1")) continue;   // inactive -> skip (OF)
        {   // the time window, if this source sets one (-1e300 sentinel = the key is absent)
            const scalar ts = opt.scalarOr("timeStart", -1e300);
            if (ts > -1e299 && ts >= scalar(0))
                fo.windows.push_back({s.first, ts, opt.scalarOr("duration", 0.0)});
        }
        if (!isVec && !isSca && !isPor && !isMvf && !isLim && !isAd && !isRot && !isVdc && !isLimT && !isFixT && !isFixS)
        {
            fo.unsupported.push_back("source '" + s.first + "' has unsupported type '" + type + "'");
            continue;
        }
        const FoamDict& co = fvoptions_detail::coeffsOf(opt, type);
        // selection cells by selectionMode: cellZone -> named zone; cellSet -> the same-named cellZone (topoSet usually
        // creates both, e.g. turbineSiting actuationDisk1/2); all -> whole domain. Fail loud if a named set/zone is absent.
        auto selZoneName = [&](const FoamDict& c, const FoamDict& o) -> std::string {
            const std::string sm = c.wordOr("selectionMode", o.wordOr("selectionMode", "all"));
            if (sm == "cellZone") return c.wordOr("cellZone", o.wordOr("cellZone", ""));
            if (sm == "cellSet")  return c.wordOr("cellSet",  o.wordOr("cellSet",  ""));   // mapped to the same-named cellZone
            return "";
        };
        // selectionMode cellSet: resolved through the SHARED resolver, as OF does in cellSetOption
        // (cellSetOption.H:175) rather than per source. Previously this required a same-named cellZone
        // to exist, so simpleFoam/turbineSiting -- whose actuationDiskSource[Froude] brae already
        // implements -- was refused outright for want of a file reader. Reading the set makes it work
        // for EVERY source type at once, which is why it belongs here and not in the disk branch.
        if (co.wordOr("selectionMode", opt.wordOr("selectionMode", "all")) == "cellSet")
        {
            const std::string cs = co.wordOr("cellSet", opt.wordOr("cellSet", ""));
            if (zones.find(cs) == zones.end())
            {
                const CellSelection sel = resolveCellSelection(polyMeshDir, "cellSet", cs, zones);
                if (!sel.ok)
                {
                    fo.unsupported.push_back("source '" + s.first + "' " + sel.reason);
                    continue;
                }
                zones[cs] = sel.cells;   // adopt it under its own name; every branch below reads `zones`
            }
        }

        if (isMvf)   // meanVelocityForce (channel-flow driver)
        {
            const std::vector<scalar> ub = co.scalarListOr("Ubar", opt.scalarListOr("Ubar", {}));
            if (ub.size() < 3) continue;
            fo.mvfUbar = vector{ub[ub.size()-3], ub[ub.size()-2], ub[ub.size()-1]};
            fo.mvfRelax = co.scalarOr("relaxation", opt.scalarOr("relaxation", 1.0));
            const std::string selMode = co.wordOr("selectionMode", opt.wordOr("selectionMode", "all"));
            if (selMode == "cellZone")
            {
                const auto it = zones.find(co.wordOr("cellZone", opt.wordOr("cellZone", "")));
                if (it != zones.end()) fo.mvfCells = it->second;
            }   // else (all): empty mvfCells -> whole domain
            fo.mvfActive = true;
            ++fo.count;
            continue;
        }
        if (isFixS)   // scalarFixedValueConstraint
        {
            const std::string selMode = co.wordOr("selectionMode", opt.wordOr("selectionMode", "all"));
            if (selMode == "cellZone")
            {
                const auto it = zones.find(co.wordOr("cellZone", opt.wordOr("cellZone", "")));
                if (it != zones.end()) fo.fixScaCells = it->second;
            }
            else if (selMode == "all")
            {
                fo.fixScaCells.resize(nCells);
                for (label c = 0; c < nCells; ++c) fo.fixScaCells[c] = c;
            }
            else
            {
                fo.unsupported.push_back("scalarFixedValueConstraint '" + s.first + "': selectionMode '"
                                         + selMode + "' -- brae supports all|cellZone");
                continue;
            }
            const FoamDict* fvd = co.subDict("fieldValues");
            if (!fvd || fo.fixScaCells.empty())
            {
                fo.unsupported.push_back("scalarFixedValueConstraint '" + s.first + "': needs a fieldValues sub-dict");
                continue;
            }
            for (const auto& lv : fvd->leaves)   // OF iterates the fieldValues sub-dict entry by entry
            {
                const std::string& fn = lv.first;
                if (fn == "k" || fn == "epsilon" || fn == "omega") fo.fixScaVals[fn] = fvd->scalarOr(fn, 0.0);
                else fo.unsupported.push_back("scalarFixedValueConstraint '" + s.first + "': field '" + fn
                                              + "' -- brae constrains k|epsilon|omega here");
            }
            if (fo.fixScaVals.empty()) continue;
            ++fo.count;
            continue;
        }

        if (isFixT)   // fixedTemperatureConstraint (mode uniform)
        {
            const std::string mode = co.wordOr("mode", opt.wordOr("mode", "uniform"));
            if (mode != "uniform")
            {
                fo.unsupported.push_back("fixedTemperatureConstraint '" + s.first + "': mode '" + mode
                                         + "' -- brae supports `uniform` (lookup reads another T field)");
                continue;
            }
            fo.fixTTemp = co.scalarOr("temperature", opt.scalarOr("temperature", 0.0));
            if (!(fo.fixTTemp > 0))
            {
                fo.unsupported.push_back("fixedTemperatureConstraint '" + s.first + "': needs a positive `temperature`");
                continue;
            }
            const std::string selMode = co.wordOr("selectionMode", opt.wordOr("selectionMode", "all"));
            if (selMode == "cellZone")
            {
                const auto it = zones.find(co.wordOr("cellZone", opt.wordOr("cellZone", "")));
                if (it != zones.end()) fo.fixTCells = it->second;
            }
            else if (selMode == "all")
            {
                fo.fixTCells.resize(nCells);
                for (label c = 0; c < nCells; ++c) fo.fixTCells[c] = c;
            }
            else
            {
                fo.unsupported.push_back("fixedTemperatureConstraint '" + s.first + "': selectionMode '"
                                         + selMode + "' -- brae supports all|cellZone");
                continue;
            }
            if (fo.fixTCells.empty()) continue;
            fo.fixTActive = true;
            ++fo.count;
            continue;
        }

        if (isLimT)   // limitTemperature (clamp T into [min, max] via the energy variable)
        {
            // OF reads them as plain `min`/`max` on the option dict (limitTemperature.C read()).
            fo.limTMin = co.scalarOr("min", opt.scalarOr("min", 0.0));
            fo.limTMax = co.scalarOr("max", opt.scalarOr("max", 0.0));
            if (!(fo.limTMax > fo.limTMin))
            {
                fo.unsupported.push_back("limitTemperature '" + s.first + "': needs max > min");
                continue;
            }
            const std::string selMode = co.wordOr("selectionMode", opt.wordOr("selectionMode", "all"));
            if (selMode == "cellZone")
            {
                const auto it = zones.find(co.wordOr("cellZone", opt.wordOr("cellZone", "")));
                if (it != zones.end()) fo.limTCells = it->second;
            }
            else if (selMode != "all")
            {
                fo.unsupported.push_back("limitTemperature '" + s.first + "': selectionMode '" + selMode
                                         + "' -- brae supports all|cellZone");
                continue;
            }
            fo.limTAllCells = (selMode == "all");
            fo.limTActive = true;
            ++fo.count;
            continue;
        }

        if (isLim)   // limitVelocity (clamp |U| <= max)
        {
            fo.limUMax = co.scalarOr("max", opt.scalarOr("max", 0.0));
            if (fo.limUMax <= 0) continue;
            const std::string selMode = co.wordOr("selectionMode", opt.wordOr("selectionMode", "all"));
            if (selMode == "cellZone")
            {
                const auto it = zones.find(co.wordOr("cellZone", opt.wordOr("cellZone", "")));
                if (it != zones.end()) fo.limUCells = it->second;
            }
            fo.limUActive = true;
            ++fo.count;
            continue;
        }
        if (isVdc)   // velocityDampingConstraint (implicit diag sink)
        {
            fo.vdcUMax = co.scalarOr("UMax", opt.scalarOr("UMax", 0.0));
            if (fo.vdcUMax <= 0) continue;
            fo.vdcC = co.scalarOr("C", opt.scalarOr("C", 1.0));
            const std::string selMode = co.wordOr("selectionMode", opt.wordOr("selectionMode", "all"));
            if (selMode == "cellZone")
            {
                const auto it = zones.find(co.wordOr("cellZone", opt.wordOr("cellZone", "")));
                if (it != zones.end()) fo.vdcCells = it->second;
            }   // else (all): empty vdcCells -> whole domain
            fo.vdcActive = true;
            ++fo.count;
            continue;
        }
        if (type == "actuationDiskSource")   // Froude actuator disk (turbine/propeller)
        {
            // OF's own keyword. variableScaling is a DIFFERENT thrust law, not a refinement of Froude,
            // so dropping the source would run the turbine site with no turbines in it and converge.
            if (co.wordOr("variant", "Froude") != "Froude")
            {
                fo.unsupported.push_back(
                    "source '" + s.first + "' is actuationDiskSource with variant '"
                    + co.wordOr("variant", "Froude") + "'; brae implements only Froude");
                continue;
            }
            FvOptionsData::ActuationDisk disk;
            const std::vector<scalar> dd = co.scalarListOr("diskDir", {});
            if (dd.size() >= 3)
            {
                vector d{dd[dd.size()-3], dd[dd.size()-2], dd[dd.size()-1]};
                const scalar m = std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
                if (m > 0) disk.diskDir = vector{d.x/m, d.y/m, d.z/m};
            }
            disk.area = co.scalarOr("diskArea", 0.0);
            // OF builds Cp and Ct as Function1<scalar> of mag(Uref) and evaluates them EVERY iteration, so
            // a table is a thrust curve, not a decoration. brae reads the constant form only; taking the
            // first knot of a table would run a plausible-looking turbine at the wrong operating point.
            bool curved = false;
            for (const char* key : {"Cp", "Ct"})
            {
                const std::string w = co.wordOr(key, "");
                if (w.empty() || isFoamScalarToken(w) || w == "constant" || w == "uniform") continue;
                curved = true;
                fo.unsupported.push_back(
                    "source '" + s.first + "' has actuationDiskSource " + key + " as a non-constant "
                    "Function1 ('" + w + "'); brae evaluates only `constant`/`uniform`");
            }
            if (curved) continue;
            const scalar Cp = co.scalarOr("Cp", 0.0), Ct = co.scalarOr("Ct", 0.0);
            if (disk.area <= 0 || Ct <= 0 || Cp <= 0)
            {
                fo.unsupported.push_back(
                    "source '" + s.first + "' is actuationDiskSource with diskArea/Cp/Ct missing or "
                    "non-positive; OpenFOAM itself fails on Cp,Ct <= 0");
                continue;
            }
            disk.a = 1.0 - Cp/Ct;                                        // induction (sink cancels: OF scales
                                                                         // BOTH Cp and Ct by sink_)
            {
                const auto it = zones.find(selZoneName(co, opt));
                if (it != zones.end()) disk.diskCells = it->second;
            }
            if (disk.diskCells.empty()) continue;
            // monitor cells: POINTS -> findCell(upstreamPoint) (closest cell centre); cellSet/cellZone -> named zone.
            const std::string mm = co.wordOr("monitorMethod", "points");
            if (mm == "points")
            {
                std::vector<scalar> up = co.scalarListOr("upstreamPoint", {});
                if (up.size() < 3)
                {
                    const FoamDict* mc = co.subDict("monitorCoeffs");
                    if (mc) up = mc->scalarListOr("points", {});
                }
                if (up.size() >= 3 && !cellCentres.empty())
                {
                    const vector p{up[up.size()-3], up[up.size()-2], up[up.size()-1]};
                    label best = 0;
                    scalar bd = 1e300;
                    for (label c = 0; c < (label)cellCentres.size(); ++c)
                    {
                        const vector r{cellCentres[c].x-p.x, cellCentres[c].y-p.y, cellCentres[c].z-p.z};
                        const scalar d2 = r.x*r.x+r.y*r.y+r.z*r.z;
                        if (d2 < bd) { bd = d2; best = c; }
                    }
                    disk.monitorCells = { best };
                }
            }
            else   // cellSet/cellZone monitor
            {
                const auto it = zones.find(co.wordOr("cellSet", co.wordOr("cellZone", "")));
                if (it != zones.end()) disk.monitorCells = it->second;
            }
            if (disk.monitorCells.empty()) continue;
            fo.adDisks.push_back(std::move(disk));   // every turbine, not just the first
            fo.adActive = true;
            ++fo.count;
            continue;
        }
        if (isRot)   // rotorDiskSource (Froude BEM, scoped)
        {
            constexpr scalar PI = 3.14159265358979323846;
            RotorDiskParams& rp = fo.rotor;
            auto v3 = [](const std::vector<scalar>& a, vector df) { return a.size()>=3 ? vector{a[a.size()-3],a[a.size()-2],a[a.size()-1]} : df; };
            rp.nBlades  = (int)co.scalarOr("nBlades", 0);
            rp.tipEffect = co.scalarOr("tipEffect", 1.0);
            rp.omega    = co.scalarOr("rpm", 0.0) * PI/30.0;              // rpm -> rad/s
            rp.origin   = v3(co.scalarListOr("origin", {}), vector{0,0,0});
            rp.axis     = v3(co.scalarListOr("axis", {}), vector{0,1,0});
            rp.refDir   = v3(co.scalarListOr("refDirection", {}), vector{0,0,1});
            rp.localInflow = (co.wordOr("inletFlowType", "local") == "local");
            rp.inletVel = v3(co.scalarListOr("inletVelocity", {}), vector{0,0,0});
            {
                const FoamDict* tc = co.subDict("fixedTrimCoeffs");
                rp.theta0 = (tc ? tc->scalarOr("theta0", 0.0) : co.scalarOr("theta0", 0.0)) * PI/180.0;   // deg -> rad
            }
            // Whole-token numeric filter: scalarListOr's regex would pull the "1" out of a profile name like
            // "profile1"; here we keep ONLY tokens that parse fully as a number (skips profile names + parens).
            auto wholeNums = [](const std::vector<std::string>* toks)
            {
                std::vector<scalar> v;
                if (!toks) return v;
                for (const auto& s : *toks)
                {
                    char* e = nullptr;
                    const double x = std::strtod(s.c_str(), &e);
                    if (e != s.c_str() && *e == '\0') v.push_back(x);
                }
                return v;
            };
            // blade.data: list of (profileName (radius twist chord)) -> numbers in groups of 3 (twist deg->rad)
            if (const FoamDict* bl = co.subDict("blade"))
            {
                const std::vector<scalar> bd = wholeNums(bl->find("data"));
                for (std::size_t k=0; k+2<bd.size(); k+=3)
                {
                    rp.bladeR.push_back(bd[k]);
                    rp.bladeTwist.push_back(bd[k+1]*PI/180.0);
                    rp.bladeChord.push_back(bd[k+2]);
                }
            }
            // profiles: the first profile sub-dict, data = list of (alpha Cd Cl) -> groups of 3 (alpha deg->rad)
            if (const FoamDict* pf = co.subDict("profiles"))
            {
                if (!pf->subs.empty())
                {
                    const std::vector<scalar> pd = wholeNums(pf->subs[0].second.find("data"));
                    for (std::size_t k=0; k+2<pd.size(); k+=3)
                    {
                        rp.pAlpha.push_back(pd[k]*PI/180.0);
                        rp.pCd.push_back(pd[k+1]);
                        rp.pCl.push_back(pd[k+2]);
                    }
                }
            }
            {
                const auto it = zones.find(co.wordOr("cellZone", opt.wordOr("cellZone","")));
                if (it != zones.end()) rp.cells = it->second;
            }
            if (rp.cells.empty() || rp.bladeR.empty() || rp.pAlpha.empty()) continue;
            if (rp.active) fo.unsupported.push_back("source '" + s.first + "': a 2nd rotorDisk -- brae keeps only one");
            rp.active = true;
            ++fo.count;
            continue;
        }

        if (isPor)   // explicitPorositySource (DarcyForchheimer)
        {
            const std::string pm = co.wordOr("type", "");
            if (pm != "DarcyForchheimer" && pm != "fixedCoeff")   // powerLaw would silently apply ZERO resistance
            {
                fo.unsupported.push_back("porosity source '" + s.first + "' uses model '" + pm
                                         + "' -- brae supports DarcyForchheimer and fixedCoeff");
                continue;
            }
            if (pm == "fixedCoeff")
            {
                const std::string zn2 = co.wordOr("cellZone", "");
                const auto it2 = zones.find(zn2);
                if (it2 == zones.end() || it2->second.empty()) continue;
                const FoamDict* fcp = co.subDict("fixedCoeffCoeffs");
                const FoamDict& fc = fcp ? *fcp : co;
                auto last3 = [](const std::vector<scalar>& a) -> vector
                { return a.size() >= 3 ? vector{a[a.size()-3], a[a.size()-2], a[a.size()-1]} : vector{0,0,0}; };
                const vector al = adjustNeg(last3(fc.scalarListOr("alpha", {})));
                const vector be = adjustNeg(last3(fc.scalarListOr("beta",  {})));
                fo.porRhoRef = fc.scalarOr("rhoRef", 1.0);

                // coordinateSystem -> rotation. OF axesRotation (axesRotation.C): the LOCAL AXES ARE THE
                // COLUMNS of R, for axisOrder E1_E2: col0 = e1, col1 = e2 with the e1-collinear part
                // removed, col2 = col0 ^ col1. Then csys().transform(T) = R & T & R.T (transform.H:214).
                scalar R[9] = {1,0,0, 0,1,0, 0,0,1};
                const FoamDict* cs = fc.subDict("coordinateSystem");
                if (cs)
                {
                    const std::vector<scalar> e1v = cs->scalarListOr("e1", {1,0,0});
                    const std::vector<scalar> e2v = cs->scalarListOr("e2", {0,1,0});
                    scalar a1[3] = {e1v.size()>2?e1v[e1v.size()-3]:1, e1v.size()>1?e1v[e1v.size()-2]:0, e1v.size()>0?e1v[e1v.size()-1]:0};
                    scalar a2[3] = {e2v.size()>2?e2v[e2v.size()-3]:0, e2v.size()>1?e2v[e2v.size()-2]:1, e2v.size()>0?e2v[e2v.size()-1]:0};
                    scalar m1 = std::sqrt(a1[0]*a1[0]+a1[1]*a1[1]+a1[2]*a1[2]);
                    if (m1 > 0) { a1[0]/=m1; a1[1]/=m1; a1[2]/=m1; }
                    const scalar dot = a2[0]*a1[0]+a2[1]*a1[1]+a2[2]*a1[2];   // removeCollinear(ax1)
                    for (int k = 0; k < 3; ++k) a2[k] -= dot*a1[k];
                    scalar m2 = std::sqrt(a2[0]*a2[0]+a2[1]*a2[1]+a2[2]*a2[2]);
                    if (m2 > 0) { a2[0]/=m2; a2[1]/=m2; a2[2]/=m2; }
                    const scalar a3[3] = { a1[1]*a2[2]-a1[2]*a2[1], a1[2]*a2[0]-a1[0]*a2[2], a1[0]*a2[1]-a1[1]*a2[0] };
                    for (int r = 0; r < 3; ++r) { R[3*r+0]=a1[r]; R[3*r+1]=a2[r]; R[3*r+2]=a3[r]; }   // axes are COLUMNS
                }
                // T' = R & diag(v) & R.T
                auto rot = [&](const vector& v, scalar* out)
                {
                    const scalar d[3] = {v.x, v.y, v.z};
                    for (int i2 = 0; i2 < 3; ++i2)
                        for (int j2 = 0; j2 < 3; ++j2)
                        {
                            scalar acc = 0;
                            for (int k = 0; k < 3; ++k) acc += R[3*i2+k]*d[k]*R[3*j2+k];
                            out[3*i2+j2] = acc;
                        }
                };
                rot(al, fo.porAlphaT);
                rot(be, fo.porBetaT);
                fo.porCells = it2->second;
                if (fo.porActive) fo.unsupported.push_back("source '" + s.first + "': a 2nd explicitPorositySource -- brae keeps only one");
                fo.porActive = true;
                fo.porFixed  = true;
                ++fo.count;
                continue;
            }
            const std::string zn = co.wordOr("cellZone", "");
            const auto it = zones.find(zn);
            if (it == zones.end() || it->second.empty()) continue;
            const FoamDict* dfp = co.subDict("DarcyForchheimerCoeffs");
            const FoamDict& df = dfp ? *dfp : co;
            // d/f are dimensionedVectors: "d  d [dims] (vx vy vz)"; the value vector is the LAST 3 numbers.
            auto last3 = [](const std::vector<scalar>& a) -> vector
            {
                return a.size() >= 3 ? vector{a[a.size()-3], a[a.size()-2], a[a.size()-1]} : vector{0,0,0};
            };
            fo.porD = adjustNeg(last3(df.scalarListOr("d", {})));
            fo.porF = adjustNeg(last3(df.scalarListOr("f", {})));
            fo.porCells = it->second;
            if (fo.porActive) fo.unsupported.push_back("source '" + s.first + "': a 2nd explicitPorositySource -- brae keeps only one");
            fo.porActive = true;
            ++fo.count;
            continue;
        }

        // selection: all cells, or a named cellZone.
        const std::string selMode = co.wordOr("selectionMode", opt.wordOr("selectionMode", "all"));
        std::vector<label> cells;
        if (selMode == "cellZone")
        {
            const std::string zn = co.wordOr("cellZone", opt.wordOr("cellZone", ""));
            const auto it = zones.find(zn);
            if (it != zones.end()) cells = it->second;
        }
        if (cells.empty() && selMode == "all")
        {
            cells.resize(nCells);
            for (label c = 0; c < nCells; ++c)
                cells[c] = c;
        }
        if (cells.empty()) continue;

        const bool absolute = co.wordOr("volumeMode", opt.wordOr("volumeMode", "specific")) == "absolute";
        scalar Vsel = 0;
        for (label c : cells)
            Vsel += V[c];
        const scalar VDash = absolute ? (Vsel > 0 ? Vsel : 1.0) : 1.0;

        // OF SemiImplicitSource::readCoeffs: `injectionRateSuSp` is the 2112-and-earlier spelling, kept for
        // compatibility, and `sources` is the current one -- findDict the old, else subDict the new. brae
        // looked only for the old name, so a modern case (pimpleFoam/laminar/planarPoiseuille writes
        // `sources { U ((5 0 0) 0); }`) fell through the `continue` below and the source vanished with the
        // driver printing "No finite volume options present". A silently dropped body force is exactly what
        // the transient refusal existed to prevent, so the miss must be LOUD, not a skip.
        const FoamDict* inj = co.subDict("injectionRateSuSp");
        if (!inj) inj = opt.subDict("injectionRateSuSp");
        if (!inj) inj = co.subDict("sources");
        if (!inj) inj = opt.subDict("sources");
        if (!inj)
        {
            fo.unsupported.push_back(
                "source '" + s.first + "' (" + type + ") has neither a `sources` nor an "
                "`injectionRateSuSp` sub-dictionary, so brae found nothing to apply");
            continue;
        }

        ++fo.count;
        if (isVec)   // momentum: field "U"
        {
            const std::vector<scalar> n = fvoptions_detail::suSpNumbers(*inj, "U");
            const vector Su{ n.size() > 0 ? n[0] : 0.0, n.size() > 1 ? n[1] : 0.0, n.size() > 2 ? n[2] : 0.0 };
            const scalar Sp = n.size() > 3 ? n[3] : 0.0;
            fo.hasMomentum = true;
            for (int comp = 0; comp < 3; ++comp)
                ensure(fo.momSu[comp]);
            if (Sp != 0.0) ensure(fo.momSp);
            for (label c : cells)
            {
                const scalar vd = V[c] / VDash;
                fo.momSu[0][c] += Su.x * vd;
                fo.momSu[1][c] += Su.y * vd;
                fo.momSu[2][c] += Su.z * vd;
                if (Sp != 0.0) fo.momSp[c] += Sp * V[c];                  // diag -= Sp*V applied by the solver
            }
        }
        else   // scalarSemiImplicitSource: brae PARSES it but never applies it (fvOptionsAddSupScalar has no callers),
        {      // so the source would be silently ignored. Record it; the single-GPU driver refuses to run.
            fo.unsupported.push_back("source '" + s.first + "' is a scalarSemiImplicitSource (parsed but not applied)");
        }
    }
    return fo;
}

} // namespace brae
