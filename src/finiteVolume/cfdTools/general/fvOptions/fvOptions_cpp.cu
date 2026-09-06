// _cpp REFERENCE implementation -- see fvOptions_cpp.cuh for the OpenFOAM provenance and the sign note.
#include "fvOptions_cpp.cuh"
#include "fv_matrix_ops.cuh"
#include "cell_selection.cuh"   // resolveCellSelection: OF cellSetOption::setSelection, already ported for MRF
#include "mrf_read.cuh"         // readCellZones
#include <cmath>
#include <filesystem>

namespace brae {
namespace cpu {
namespace fvOptions {

namespace {

// A `d [0 -2 0 0 0 0 0] (5e7 -1000 -1000)` entry: OpenFOAM's dimensioned<vector>. The dimension set is
// skipped -- brae carries no dimension checking -- and the three numbers are the vector.
bool readDimensionedVector(const FoamDict& d, const std::string& key, vector& out)
{
    const std::vector<std::string>* v = d.find(key);
    if (!v) return false;
    std::vector<scalar> nums;
    for (const std::string& tok : *v)
    {
        try
        {
            std::size_t pos = 0;
            const scalar x = std::stod(tok, &pos);
            if (pos == tok.size()) nums.push_back(x);
        }
        catch (...)
        {
        }
    }
    // [dimensions](7) then the vector(3); take the LAST three, which is the vector either way.
    if (nums.size() < 3) return false;
    out = vector{ nums[nums.size()-3], nums[nums.size()-2], nums[nums.size()-1] };
    return true;
}

// porosityModel::adjustNegativeResistance. A NEGATIVE resistance component in the dictionary is not a
// literal negative coefficient: OpenFOAM replaces it with `val*(-maxCmpt)`, i.e. it becomes POSITIVE and
// scaled by the largest component. angledDuctExplicitFixedCoeff's `alpha (500 -1000 -1000)` is really
// (500, 500000, 500000) -- a very strong resistance across the duct and a weak one along it -- and
// tr(alpha) goes from -1500 to 1000500. Reading it verbatim put the momentum diagonal 667x out inside
// the porosity zone and exact everywhere else, which is what localised it.
//
// All-negative is a fatal error in OpenFOAM, not a clamp, so it is refused here rather than adjusted.
void adjustNegativeResistance(vector& r, const char* what)
{
    const scalar maxCmpt = std::max(r.x, std::max(r.y, r.z));
    if (maxCmpt < 0.0)
        throw std::runtime_error(
            std::string("porosity: every component of `") + what + "` is negative. OpenFOAM treats that "
            "as a fatal error (porosityModel::adjustNegativeResistance) rather than a resistance.");
    if (r.x < 0.0) r.x *= -maxCmpt;
    if (r.y < 0.0) r.y *= -maxCmpt;
    if (r.z < 0.0) r.z *= -maxCmpt;
}

// csys().transform(T) for a Cartesian system given by e1/e2: R & T & R^T with R's rows the local axes.
tensor transformDiag(const vector& diag, const vector& e1in, const vector& e2in)
{
    auto norm = [](vector v) {
        const scalar m = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        return m > 0 ? vector{v.x/m, v.y/m, v.z/m} : vector{0,0,0};
    };
    const vector e1 = norm(e1in);
    // Gram-Schmidt e2 against e1, then e3 = e1 x e2 -- OpenFOAM's coordinateSystem does the same, so a
    // non-orthogonal e2 in the dictionary gives the same axes here as there.
    vector e2 = e2in;
    const scalar dot = e1.x*e2.x + e1.y*e2.y + e1.z*e2.z;
    e2 = norm(vector{e2.x - dot*e1.x, e2.y - dot*e1.y, e2.z - dot*e1.z});
    const vector e3 { e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x };

    const scalar R[3][3] = { {e1.x, e1.y, e1.z}, {e2.x, e2.y, e2.z}, {e3.x, e3.y, e3.z} };
    const scalar Dl[3]   = { diag.x, diag.y, diag.z };
    scalar T[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            scalar v = 0;
            for (int q = 0; q < 3; ++q) v += R[q][i] * Dl[q] * R[q][j];   // R^T * diag * R
            T[i][j] = v;
        }
    return tensor{ T[0][0], T[0][1], T[0][2],
                   T[1][0], T[1][1], T[1][2],
                   T[2][0], T[2][1], T[2][2] };
}

} // namespace


std::string OptionList::firstUnsupported() const
{
    for (const Option& o : options)
        if (o.active && !o.unsupported.empty()) return o.unsupported;
    return "";
}


OptionList read(const std::string& caseDir, const PrimitiveMesh& m)
{
    (void)m;
    OptionList list;
    namespace fs = std::filesystem;
    std::string path;
    for (const std::string& p : {caseDir + "/system/fvOptions", caseDir + "/constant/fvOptions"})
        if (fs::exists(p))
        {
            path = p;
            break;
        }
    if (path.empty()) return list;

    const FoamDict root = readDict(path);
    const std::string polyMeshDir = caseDir + "/constant/polyMesh";
    const auto zones = readCellZones(polyMeshDir);

    for (const auto& entry : root.subs)
    {
        Option o;
        o.name = entry.first;
        const FoamDict& d = entry.second;
        o.type = d.wordOr("type", "");
        const std::string act = d.wordOr("active", "true");
        o.active = !(act == "false" || act == "no" || act == "off" || act == "0");
        if (!o.active)
        {
            list.options.push_back(o);
            continue;
        }

        // rotorDiskSource is implemented (rotorDiskSource_cpp.cu, gated by
        // tests/rotordisk_vs_openfoam.sh against OpenFOAM's own reported drag/lift/AOA). Its parameters
        // are read by readFvOptions into RotorDiskParams, not here -- this list only decides whether the
        // envelope refuses the case, and the geometry needs a mesh this reader does not take.
        if (o.type == "rotorDisk" || o.type == "rotorDiskSource")
        {
            o.rotorDisk = true;
            list.options.push_back(o);
            continue;
        }
        // actuationDiskSource (Froude) is implemented (actuationDiskSource_cpp.cu + actuation_disk.cu,
        // gated by tests/actuationdisk_vs_openfoam.sh against the Uref and thrust OpenFOAM writes for
        // itself). Like the rotor, its parameters come from readFvOptions, which has the mesh; what is
        // refused there -- variableScaling, a Function1 Cp/Ct -- reaches the driver as an unsupported
        // entry rather than through this list.
        if (o.type == "actuationDiskSource")
        {
            o.actuationDisk = true;
            list.options.push_back(o);
            continue;
        }
        // The CONSTRAINTS. Both resolve a cell list the same way a source does and then apply
        // eqn.setValues on it; the value is what differs.
        if (o.type == "fixedTemperatureConstraint" || o.type == "scalarFixedValueConstraint")
        {
            const FoamDict* cc = d.subDict(o.type + "Coeffs");
            const FoamDict& cs = cc ? *cc : d;
            const CellSelection csel = resolveCellSelection(
                polyMeshDir, cs.wordOr("selectionMode", "all"),
                cs.wordOr("cellZone", cs.wordOr("cellSet", "")), zones);
            if (!csel.ok)
            {
                o.unsupported = o.type + ": " + csel.reason;
                list.options.push_back(o);
                continue;
            }
            o.cells    = csel.cells;
            o.allCells = csel.all;

            if (o.type == "fixedTemperatureConstraint")
            {
                // `lookup` mode takes T from a named field per iteration; only `uniform` is implemented,
                // and the other is refused rather than run as a constant.
                const std::string mode = cs.wordOr("mode", "uniform");
                if (mode != "uniform")
                {
                    o.unsupported = "fixedTemperatureConstraint mode '" + mode + "'";
                    list.options.push_back(o);
                    continue;
                }
                o.constraint = Option::Constraint::fixedTemperature;
                o.Tuniform   = cs.scalarOr("temperature", 0.0);
            }
            else
            {
                const FoamDict* fv = cs.subDict("fieldValues");
                if (!fv)
                {
                    o.unsupported = "scalarFixedValueConstraint without fieldValues";
                    list.options.push_back(o);
                    continue;
                }
                o.constraint = Option::Constraint::scalarFixedValue;
                for (const auto& leaf : fv->leaves)
                    o.fieldValues.emplace_back(leaf.first, fv->scalarOr(leaf.first, 0.0));
            }
            list.options.push_back(o);
            continue;
        }
        if (o.type != "explicitPorositySource")
        {
            o.unsupported = o.type.empty() ? std::string("(no type)") : o.type;
            list.options.push_back(o);
            continue;
        }

        // OpenFOAM allows the coefficients either in <type>Coeffs or inline (v2412 reads both).
        const FoamDict* c = d.subDict("explicitPorositySourceCoeffs");
        const FoamDict& src = c ? *c : d;

        const std::string pType = src.wordOr("type", "");
        if (pType == "fixedCoeff")
        {
            const CellSelection fsel = resolveCellSelection(
                polyMeshDir, src.wordOr("selectionMode", "all"),
                src.wordOr("cellZone", src.wordOr("cellSet", "")), zones);
            if (!fsel.ok)
            {
                o.unsupported = "explicitPorositySource: " + fsel.reason;
                list.options.push_back(o);
                continue;
            }
            o.cells    = fsel.cells;
            o.allCells = fsel.all;

            const FoamDict* fcc = src.subDict("fixedCoeffCoeffs");
            const FoamDict& fc  = fcc ? *fcc : src;
            vector av{0,0,0}, bv{0,0,0};
            readDimensionedVector(fc, "alpha", av);
            readDimensionedVector(fc, "beta",  bv);
            adjustNegativeResistance(av, "alpha");
            adjustNegativeResistance(bv, "beta");

            vector fe1{1,0,0}, fe2{0,1,0};
            if (const FoamDict* cs = fc.subDict("coordinateSystem"))
            {
                const FoamDict* rot = cs->subDict("rotation");
                const FoamDict& r = rot ? *rot : *cs;
                vector t;
                if (readDimensionedVector(r, "e1", t)) fe1 = t;
                if (readDimensionedVector(r, "e2", t)) fe2 = t;
            }
            o.fixedCoeff = true;
            o.alpha  = transformDiag(av, fe1, fe2);
            o.beta   = transformDiag(bv, fe1, fe2);
            o.rhoRef = fc.scalarOr("rhoRef", 1.0);
            list.options.push_back(o);
            continue;
        }
        if (pType != "DarcyForchheimer")
        {
            o.unsupported = "explicitPorositySource/" + (pType.empty() ? std::string("(no type)") : pType);
            list.options.push_back(o);
            continue;
        }

        const CellSelection sel = resolveCellSelection(
            polyMeshDir, src.wordOr("selectionMode", "all"),
            src.wordOr("cellZone", src.wordOr("cellSet", "")), zones);
        if (!sel.ok)
        {
            o.unsupported = "explicitPorositySource: " + sel.reason;
            list.options.push_back(o);
            continue;
        }
        o.cells = sel.cells;
        o.allCells = sel.all;

        const FoamDict* dfc = src.subDict("DarcyForchheimerCoeffs");
        const FoamDict& df = dfc ? *dfc : src;
        vector dv{0,0,0}, fv{0,0,0};
        readDimensionedVector(df, "d", dv);
        readDimensionedVector(df, "f", fv);
        // DarcyForchheimer.C:67-68 calls it on d and f too, so the same shorthand applies there.
        adjustNegativeResistance(dv, "d");
        adjustNegativeResistance(fv, "f");

        vector e1{1,0,0}, e2{0,1,0};
        if (const FoamDict* cs = df.subDict("coordinateSystem"))
        {
            const FoamDict* rot = cs->subDict("rotation");
            const FoamDict& r = rot ? *rot : *cs;
            vector t;
            if (readDimensionedVector(r, "e1", t)) e1 = t;
            if (readDimensionedVector(r, "e2", t)) e2 = t;
        }

        // calcTransformModelData: D = csys(diag(d)), F = csys(diag(0.5*f)). The 0.5 is HERE, not in the
        // resistance -- putting it in both places halves the Forchheimer term twice.
        o.D = transformDiag(dv, e1, e2);
        o.F = transformDiag(vector{0.5*fv.x, 0.5*fv.y, 0.5*fv.z}, e1, e2);
        list.options.push_back(o);
    }
    return list;
}


void addSup(
    const OptionList&             opts,
    FvVectorMatrix&               UEqn,
    const GeometricField<vector>& U,
    scalar                        nu,
    const FvGeometry&             g,
    bool                          forceDimensions,
    const std::vector<scalar>*    rhoCell,
    const std::vector<scalar>*    muCell)
{
    // A force-dimensioned DarcyForchheimer without the per-cell fields cannot be computed -- the old
    // path took nu = 0 and no rho, which zeroes the Darcy half and under-weights the Forchheimer half
    // by the whole density, silently. Refuse by name instead.
    if (forceDimensions && !muCell)
        for (const Option& o : opts.options)
            if (o.active && o.unsupported.empty() && !o.fixedCoeff
                && o.constraint == Option::Constraint::none)
                throw std::runtime_error(
                    "fvOptions addSup: a DarcyForchheimer porosity on a force-dimensioned momentum "
                    "equation needs the per-cell rho and laminar mu (Cd = mu*D + rho*|U|*F, "
                    "DarcyForchheimerTemplates.C:53); the caller supplied neither.");

    const std::vector<scalar>& V = g.V();
    for (const Option& o : opts.options)
    {
        if (!o.active || !o.unsupported.empty()) continue;
        if (o.constraint != Option::Constraint::none) continue;   // constraints are not sources

        // fixedCoeff's rho is the dict's rhoRef on a force-dimensioned equation and 1 otherwise -- it is
        // NOT the local density, which is easy to assume and wrong (fixedCoeff.C:202-207).
        const scalar fcRho = forceDimensions ? o.rhoRef : scalar(1.0);

        const std::size_t n = o.allCells ? U.internal.size() : o.cells.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const label c = o.allCells ? static_cast<label>(i) : o.cells[i];
            const vector& u = U.internal[c];
            const scalar magU = std::sqrt(u.x*u.x + u.y*u.y + u.z*u.z);

            // Cd, by model. fixedCoeff: rho*(alpha + beta*|U|). DarcyForchheimer: mu*D + (rho*magU)*F,
            // with mu = nu and rho = 1 for a kinematic equation. Everything below -- the isotropic split,
            // the diagonal, the sign of the source -- is fixedCoeff.C:apply and is shared by both.
            scalar cd[9];
            if (o.fixedCoeff)
            {
                const scalar* al = &o.alpha.xx;
                const scalar* be = &o.beta.xx;
                for (int k = 0; k < 9; ++k) cd[k] = fcRho * (al[k] + magU * be[k]);
            }
            else
            {
                const scalar* dd = &o.D.xx;
                const scalar* ff = &o.F.xx;
                // Cd = mu*D + (rho*|U|)*F (DarcyForchheimerTemplates.C:53). Incompressibly mu is the
                // kinematic nu and rho is geometricOneField, which the null defaults reproduce.
                const scalar muc  = muCell  ? (*muCell)[c]  : nu;
                const scalar rhoc = rhoCell ? (*rhoCell)[c] : scalar(1);
                for (int k = 0; k < 9; ++k) cd[k] = muc * dd[k] + rhoc * magU * ff[k];
            }
            const scalar isoCd = cd[0] + cd[4] + cd[8];

            UEqn.diag[c] += V[c]*isoCd;
            // (Cd - I*isoCd) & U : the OFF-isotropic part, moved to the source. The diagonal keeps the
            // isotropic part implicit, which is what makes a large Darcy coefficient stable.
            const scalar a[9] = { cd[0]-isoCd, cd[1], cd[2],
                                  cd[3], cd[4]-isoCd, cd[5],
                                  cd[6], cd[7], cd[8]-isoCd };
            UEqn.source[c].x -= V[c]*(a[0]*u.x + a[1]*u.y + a[2]*u.z);
            UEqn.source[c].y -= V[c]*(a[3]*u.x + a[4]*u.y + a[5]*u.z);
            UEqn.source[c].z -= V[c]*(a[6]*u.x + a[7]*u.y + a[8]*u.z);
        }
    }
}

void constrain(
    const OptionList&           opts,
    FvScalarMatrix&             eqn,
    std::vector<scalar>&        psi,
    const std::string&          field,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    scalar                    (*heOfT)(scalar))
{
    for (const Option& o : opts.options)
    {
        if (!o.active || !o.unsupported.empty()) continue;
        if (o.constraint == Option::Constraint::none) continue;

        std::vector<label>  cells;
        std::vector<scalar> vals;

        if (o.constraint == Option::Constraint::fixedTemperature)
        {
            // The energy equation only. OpenFOAM sets he(p, Tuniform), NOT the temperature: putting a
            // temperature where an energy belongs is a 400x error that still converges.
            if (field != "e" && field != "h") continue;
            if (!heOfT)
                throw std::runtime_error(
                    "fvOptions: a fixedTemperatureConstraint is active but no he(T) conversion was "
                    "supplied. OpenFOAM constrains the energy equation to he(p, Tuniform); refusing "
                    "rather than constraining it to a temperature.");
            cells = o.allCells ? std::vector<label>() : o.cells;
            if (o.allCells) { cells.resize(m.nCells()); for (label c = 0; c < m.nCells(); ++c) cells[c] = c; }
            vals.assign(cells.size(), heOfT(o.Tuniform));
        }
        else
        {
            scalar v = 0.0;
            bool   found = false;
            for (const auto& fv : o.fieldValues)
                if (fv.first == field) { v = fv.second; found = true; }
            if (!found) continue;
            cells = o.allCells ? std::vector<label>() : o.cells;
            if (o.allCells) { cells.resize(m.nCells()); for (label c = 0; c < m.nCells(); ++c) cells[c] = c; }
            vals.assign(cells.size(), v);
        }

        if (!cells.empty()) setValues(eqn, psi, m, patches, cells, vals);
    }
}

} // namespace fvOptions
} // namespace cpu
} // namespace brae
