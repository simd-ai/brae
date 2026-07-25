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
#include "rotor_disk.cuh"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace brae {

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
    // meanVelocityForce: a uniform body force adjusted each iter to drive the mean velocity to Ubar (channel flow).
    bool                mvfActive = false;
    vector              mvfUbar{0,0,0};
    scalar              mvfRelax = 1.0;
    std::vector<label>  mvfCells;                   // empty => all cells (selectionMode all)
    // limitVelocity (fvConstraint): clamp |U| <= max on a selection, post-solve (aero/atmospheric stabiliser).
    bool                limUActive = false;
    scalar              limUMax = 0;
    std::vector<label>  limUCells;                  // empty => all cells
    // velocityDampingConstraint (fvConstraint): implicit diagonal sink diag += C*V^(2/3)*(|U|-UMax) where |U|>UMax.
    bool                vdcActive = false;
    scalar              vdcUMax = 0, vdcC = 1;
    std::vector<label>  vdcCells;                   // empty => all cells
    // actuationDiskSource (Froude): wind-turbine/propeller momentum source over a disk cellZone, the thrust computed
    // each iter from the upstream-monitored velocity. T = 2*rho*A*(Uref.diskDir)^2*a*(1-a), a = 1 - Cp/Ct.
    bool                adActive = false;
    vector              adDiskDir{1,0,0};
    scalar              adArea = 0, adA = 0;        // diskArea, induction a = 1 - Cp/Ct
    std::vector<label>  adDiskCells, adMonitorCells;
    RotorDiskParams     rotor;                      // rotorDiskSource (BEM); geometry built later from the mesh
    int  count = 0;                      // number of options read (0 => no file / empty => all hooks no-op)
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
inline FvOptionsData readFvOptions(
    const std::string& caseDir,
    const std::map<std::string, std::vector<label>>& zones,
    const std::vector<scalar>& V,
    label nCells,
    const std::vector<vector>& cellCentres = {})
{
    FvOptionsData fo;
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
        const std::string act = opt.wordOr("active", "yes");
        if (!(act == "yes" || act == "true" || act == "on" || act == "1")) continue;   // inactive -> skip (OF)
        if (!isVec && !isSca && !isPor && !isMvf && !isLim && !isAd && !isRot && !isVdc)
        {
            fo.unsupported.push_back("source '" + s.first + "' has unsupported type '" + type + "'");
            continue;
        }
        const FoamDict& co = fvoptions_detail::coeffsOf(opt, type);
        if (co.wordOr("selectionMode", opt.wordOr("selectionMode", "all")) == "cellSet")   // no cellSet reader
        {
            fo.unsupported.push_back("source '" + s.first + "' uses selectionMode cellSet (use cellZone instead)");
            continue;
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
            if (co.wordOr("variant", "Froude") != "Froude") continue;    // (variableScaling: future)
            const std::vector<scalar> dd = co.scalarListOr("diskDir", {});
            if (dd.size() >= 3)
            {
                vector d{dd[dd.size()-3], dd[dd.size()-2], dd[dd.size()-1]};
                const scalar m = std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
                if (m > 0) fo.adDiskDir = vector{d.x/m, d.y/m, d.z/m};
            }
            fo.adArea = co.scalarOr("diskArea", 0.0);
            const scalar Cp = co.scalarOr("Cp", 0.0), Ct = co.scalarOr("Ct", 0.0);
            if (fo.adArea <= 0 || Ct <= 0 || Cp <= 0) continue;
            fo.adA = 1.0 - Cp/Ct;                                        // induction (sink cancels)
            // disk cells: the selection cellZone (selectionMode cellZone).
            {
                const auto it = zones.find(co.wordOr("cellZone", opt.wordOr("cellZone", "")));
                if (it != zones.end()) fo.adDiskCells = it->second;
            }
            if (fo.adDiskCells.empty()) continue;
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
                    fo.adMonitorCells = { best };
                }
            }
            else   // cellSet/cellZone monitor
            {
                const auto it = zones.find(co.wordOr("cellSet", co.wordOr("cellZone", "")));
                if (it != zones.end()) fo.adMonitorCells = it->second;
            }
            if (fo.adMonitorCells.empty()) continue;
            if (fo.adActive) fo.unsupported.push_back("source '" + s.first + "': a 2nd actuationDiskSource -- brae keeps only one");
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
            if (pm != "DarcyForchheimer")   // powerLaw / fixedCoeff not implemented -> would silently apply ZERO resistance
            {
                fo.unsupported.push_back("porosity source '" + s.first + "' uses model '" + pm
                                         + "' -- only DarcyForchheimer is supported (others would apply no resistance)");
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

        const FoamDict* inj = co.subDict("injectionRateSuSp");
        if (!inj) inj = opt.subDict("injectionRateSuSp");
        if (!inj) continue;

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
