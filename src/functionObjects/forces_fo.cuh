#pragma once
// functionObjects::forces and functionObjects::forceCoeffs, OpenFOAM v2412 (src/functionObjects/forces/forces/forces.C and
// forceCoeffs/forceCoeffs.C), run in the functionObject lifecycle (brae_time.cuh): every step its executeControl selects,
// written every step its writeControl selects (OpenFOAM's default: every step).
//
// WHAT IS COMPUTED. forces.C calcForcesMoments' pressure-field branch on the device (forces_device.cuh): per selected patch
// the pressure and viscous force and their moments about the coordinate system's origin, the patches added in ascending
// order, then summed over the ranks (the multi-GPU layer's hook; none on one GPU), then put in the coordinate system's
// local axes. forceCoeffs.C divides by Aref*pDyn (+SMALL) and Aref*pDyn*lRef (+SMALL), pDyn = 0.5*rhoInf*magUInf^2, and
// reports Cd, Cs, Cl (e1, e2, e3 of the force), CmRoll, CmPitch, CmYaw (of the moment), and the front and rear axle parts
// Cd(f,r) = 0.5*Cd +- CmRoll, Cs(f,r) = 0.5*Cs +- CmYaw, Cl(f,r) = 0.5*Cl +- CmPitch, each as pressure + viscous +
// internal (porous).
//
// WHAT IS WRITTEN. postProcessing/<name>/<start time>/force.dat and moment.dat (forces) or coefficient.dat (forceCoeffs), in
// writeFile's format: the header lines, a column a component or coefficient, the time and the values in scientific notation
// at the object's writePrecision (default: controlDict's), columns writePrecision + 8 wide. A restart opens a new file under
// its own start time, as OpenFOAM's newFileAtStartTime does, and `<file>_<time>` when that file is already there.
//
// WHAT IS REFUSED (the object is then not run, and the run says so): directForceDensity (no force-density field in the
// solvers), writeFields, porosity on a case with porosity models (their resistance force is not computed; with none, the
// porous columns are zero and that is said, as OpenFOAM warns), p or U naming another field, rho other than rhoInf on the
// incompressible solvers (OpenFOAM stops: "Could not find rho"), a missing or non-positive rhoInf/magUInf/lRef/Aref, an
// unknown coefficient, a grad(U) scheme other than the two the solver carries, a coordinate system or rotation type other
// than cartesian with axes or axisAngle. Entries v2412 does not read are said to be unread: binData (binning is the binField
// function object), and pitchAxis/rollAxis/yawAxis/sideDir (the axes are e1 = dragDir, e3 = liftDir, e2 = e3 ^ e1); a given
// pitchAxis that is not e2 is said to differ from the axis OpenFOAM uses.
#include "brae_time.cuh"
#include "brae_notice.cuh"
#include "device_simple_foam.cuh"
#include "forces_device.cuh"
#include "foam_dict.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "object_registry.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace brae
{

// a coordinate system as forces reads it: an origin and the local axes (OpenFOAM's rotation tensor's columns)
struct ForcesFrame
{
    vector origin{0, 0, 0};
    vector e1{1, 0, 0};
    vector e2{0, 1, 0};
    vector e3{0, 0, 1};

    // TEST CONTROL BRAE_CONTROL_FORCES_GLOBAL_AXES: the global components reported whatever the axes
    bool globalAxes = false;

    // coordinateSystem::localVector on a cartesian system: R^T & v, R's columns the axes
    vector local(const vector& v) const
    {
        if (globalAxes) return v;
        return vector{e1.x * v.x + e1.y * v.y + e1.z * v.z, e2.x * v.x + e2.y * v.y + e2.z * v.z, e3.x * v.x + e3.y * v.y + e3.z * v.z};
    }
};

// The multi-GPU layer's hooks: a sum over the ranks (in place, every rank the same answer) and whether this rank writes the
// files (OpenFOAM's master). One GPU: no sum, and it writes.
struct ForcesParallel
{
    std::function<void(double*, int)> sum;
    bool master = true;
};

// What a forces object needs from the run that is not in its own dictionary
struct ForcesContext
{
    std::string caseDir;
    ObjectRegistry* registry = nullptr;   // "solver", "patches", "geometry", looked up at the first execute
    int controlDictPrecision = 6;         // IOstream::defaultPrecision, set from controlDict's writePrecision
    std::string gradUScheme;              // the case's gradSchemes entry for grad(U), as written
    bool casePorosity = false;            // the case has porosity models (porosityProperties or an explicitPorositySource)
    ForcesParallel parallel;
};

// The run's side of a forces object, read once by the driver: controlDict's writePrecision (IOstream::defaultPrecision), the
// case's gradSchemes entry for grad(U) (else its default), and whether the case has porosity models
inline ForcesContext forcesContext(
    const std::string& caseDir,
    ObjectRegistry& registry,
    const FoamDict& controlDict)
{
    ForcesContext c;
    c.caseDir = caseDir;
    c.registry = &registry;
    c.controlDictPrecision = controlDict.intOr("writePrecision", 6);
    try
    {
        const FoamDict schemes = readDict(caseDir + "/system/fvSchemes");
        if (const FoamDict* g = schemes.subDict("gradSchemes"))
        {
            const std::vector<std::string>* t = g->find("grad(U)");
            if (!t) t = g->find("default");
            if (t)
                for (const std::string& w : *t) c.gradUScheme += (c.gradUScheme.empty() ? "" : " ") + w;
        }
    }
    catch (const std::exception&)
    {
    }
    namespace fs = std::filesystem;
    c.casePorosity = fs::exists(caseDir + "/constant/porosityProperties");
    for (const char* f : {"/system/fvOptions", "/constant/fvOptions"})
    {
        std::ifstream in(caseDir + f);
        std::stringstream ss;
        ss << in.rdbuf();
        if (ss.str().find("explicitPorositySource") != std::string::npos) c.casePorosity = true;
    }
    return c;
}

namespace forcesDetail
{

inline vector vec3(
    const std::vector<scalar>& v,
    bool& ok)
{
    if (v.size() != 3)
    {
        ok = false;
        return vector{0, 0, 0};
    }
    return vector{v[0], v[1], v[2]};
}

inline scalar dot3(
    const vector& a,
    const vector& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline vector cross3(
    const vector& a,
    const vector& b)
{
    return vector{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline scalar mag3(const vector& a)
{
    return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

// OpenFOAM coordinateRotations::axes::rotation (axesRotation.C): axis1 dominant, axis2 with its axis1 part removed, both
// normalised; the columns by the order. order 0 E1_E2, 1 E2_E3, 2 E3_E1
inline std::string axesFrame(
    const vector& axis1,
    const vector& axis2,
    int order,
    ForcesFrame& f)
{
    const scalar m1 = mag3(axis1);
    if (m1 < 1e-150) return "the dominant coordinate axis has zero length";
    const vector a1{axis1.x / m1, axis1.y / m1, axis1.z / m1};
    vector a2 = axis2;
    if (mag3(a2) < 1e-150) return "the second coordinate axis has zero length (OpenFOAM then guesses one; not carried)";
    const scalar c = dot3(a1, a2);
    a2 = vector{a2.x - c * a1.x, a2.y - c * a1.y, a2.z - c * a1.z};
    const scalar m2 = mag3(a2);
    if (m2 < 1e-15) return "the two coordinate axes are co-linear";
    a2 = vector{a2.x / m2, a2.y / m2, a2.z / m2};
    const vector a3 = cross3(a1, a2);
    if (order == 0)
    {
        f.e1 = a1;
        f.e2 = a2;
        f.e3 = a3;
    }
    else if (order == 1)
    {
        f.e1 = a3;
        f.e2 = a1;
        f.e3 = a2;
    }
    else
    {
        f.e1 = a2;
        f.e2 = a3;
        f.e3 = a1;
    }
    return "";
}

// OpenFOAM quaternion(axis, theta).R() (quaternionI.H): w = cos(theta/2), v = sin(theta/2)*axis/|axis|
inline void axisAngleFrame(
    const vector& axis,
    scalar angle,
    bool degrees,
    ForcesFrame& f)
{
    if (std::fabs(angle) < 1e-300 || mag3(axis) < 1e-15)
    {
        f.e1 = vector{1, 0, 0};
        f.e2 = vector{0, 1, 0};
        f.e3 = vector{0, 0, 1};
        return;
    }
    const scalar theta = degrees ? angle * M_PI / 180.0 : angle;
    const scalar m = mag3(axis);
    const scalar s = std::sin(0.5 * theta);
    const scalar w = std::cos(0.5 * theta);
    const scalar x = s * (axis.x / m);
    const scalar y = s * (axis.y / m);
    const scalar z = s * (axis.z / m);
    const scalar w2 = w * w;
    const scalar x2 = x * x;
    const scalar y2 = y * y;
    const scalar z2 = z * z;
    const scalar txy = 2 * x * y;
    const scalar twz = 2 * w * z;
    const scalar txz = 2 * x * z;
    const scalar twy = 2 * w * y;
    const scalar tyz = 2 * y * z;
    const scalar twx = 2 * w * x;
    // the tensor's rows are (xx xy xz), (yx yy yz), (zx zy zz); the axes are its columns
    f.e1 = vector{w2 + x2 - y2 - z2, txy + twz, txz - twy};
    f.e2 = vector{txy - twz, w2 - x2 + y2 - z2, tyz + twx};
    f.e3 = vector{txz + twy, tyz - twx, w2 - x2 - y2 + z2};
}

// a rotation from a dictionary holding it: `type` axes / axisAngle / none, or the axes' entries directly
inline std::string rotationFrom(
    const FoamDict& d,
    const std::string& type,
    ForcesFrame& f)
{
    bool ok = true;
    if (type == "none" || type == "identity")
    {
        f.e1 = vector{1, 0, 0};
        f.e2 = vector{0, 1, 0};
        f.e3 = vector{0, 0, 1};
        return "";
    }
    if (type == "axisAngle")
    {
        if (!d.found("axis") || !d.found("angle")) return "rotation axisAngle needs axis and angle (axisAngleRotation.C)";
        const vector axis = vec3(d.scalarListOr("axis", {}), ok);
        if (!ok) return "rotation axisAngle: axis is not a vector";
        axisAngleFrame(axis, d.scalarOr("angle", 0), d.switchOr("degrees", true), f);
        return "";
    }
    if (type.empty() || type == "axes" || type == "axesRotation" || type == "cartesian")
    {
        auto has = [&](const char* k) { return d.found(k); };
        vector a, b;
        int order = -1;
        if (has("e1") && has("e2"))
        {
            a = vec3(d.scalarListOr("e1", {}), ok);
            b = vec3(d.scalarListOr("e2", {}), ok);
            order = 0;
        }
        else if (has("e2") && has("e3"))
        {
            a = vec3(d.scalarListOr("e2", {}), ok);
            b = vec3(d.scalarListOr("e3", {}), ok);
            order = 1;
        }
        else if (has("e3") && has("e1"))
        {
            a = vec3(d.scalarListOr("e3", {}), ok);
            b = vec3(d.scalarListOr("e1", {}), ok);
            order = 2;
        }
        else if (has("axis") && has("direction"))
        {
            a = vec3(d.scalarListOr("axis", {}), ok);
            b = vec3(d.scalarListOr("direction", {}), ok);
            order = 2;
        }
        if (order < 0) return "no entries of the type (e1, e2) or (e2, e3) or (e3, e1) found (axesRotation.C)";
        if (!ok) return "a coordinate axis is not a vector";
        return axesFrame(a, b, order, f);
    }
    return "rotation type '" + type + "' is not carried (axes, axisAngle and none are)";
}

// coordinateSystem::assign on a cartesian system: the origin, then `rotation` (a dictionary, or a word whose entries are in
// this dictionary), its older name `coordinateRotation`, or the axes' entries in this dictionary
inline std::string systemFrom(
    const FoamDict& d,
    bool originOptional,
    ForcesFrame& f)
{
    bool ok = true;
    if (d.found("origin"))
    {
        f.origin = vec3(d.scalarListOr("origin", {}), ok);
        if (!ok) return "origin is not a vector";
    }
    else if (!originOptional)
    {
        return "no CofR, no coordinateSystem and no origin: OpenFOAM reads origin with no default (coordinateSystem.C)";
    }
    const FoamDict* r = d.subDict("rotation");
    if (!r) r = d.subDict("coordinateRotation");
    if (r) return rotationFrom(*r, r->wordOr("type", ""), f);
    if (d.found("rotation")) return rotationFrom(d, d.wordOr("rotation", ""), f);
    return rotationFrom(d, "", f);
}

// polyBoundaryMesh::patchSet(wordRes, useGroups): an entry written in quotes is a regular expression, a bare word a literal
// name; either selects a patch by its name or by a group it is in. brae's dictionary tokens carry no quotes (the tokenizer
// strips them) and split every bare parenthesis, so a token holding a regular expression's characters or a parenthesis can
// only have been quoted, and is taken as one. The one reading that differs from OpenFOAM's: a BARE word holding such
// characters (`wall.*` unquoted), which OpenFOAM takes as a literal name, is taken here as an expression
struct PatchMatcher
{
    std::vector<std::string> literals;
    std::vector<std::regex> patterns;
    std::vector<std::string> written;

    bool matchOne(const std::string& s) const
    {
        for (const std::string& l : literals)
            if (s == l) return true;
        for (const std::regex& r : patterns)
            if (std::regex_match(s, r)) return true;
        return false;
    }

    bool selects(const FvPatch& p) const
    {
        if (matchOne(p.name)) return true;
        for (const std::string& g : p.inGroups)
            if (matchOne(g)) return true;
        return false;
    }
};

inline std::string patchMatcherFrom(
    const FoamDict& d,
    PatchMatcher& m)
{
    const std::vector<std::string>* t = d.find("patches");
    if (!t) return "patches is mandatory (forces.C read)";
    static const std::string special = ".*+?|()[]{}^$\\";
    for (const std::string& tok : *t)
    {
        if (tok == "(" || tok == ")") continue;
        if (tok.find_first_of(special) != std::string::npos)
        {
            const std::string body = tok;
            try
            {
                m.patterns.push_back(compileFoamRegex(body));
            }
            catch (const std::exception&)
            {
                return "patches: '" + body + "' is not a regular expression";
            }
            m.written.push_back(tok);
            continue;
        }
        m.literals.push_back(tok);
        m.written.push_back(tok);
    }
    return "";
}

// writeFile's file and format (writeFile.C): newFileAtTime(name, startTime) under postProcessing/<object>/<start time>/,
// <name>_<start time>.dat when <name>.dat is there; scientific at writePrecision; columns writePrecision + addChars (8) wide
class DatFile
{
public:
    bool open(
        const std::string& caseDir,
        const std::string& object,
        const std::string& startName,
        const std::string& name,
        int precision)
    {
        precision_ = precision;
        const std::string dir = caseDir + "/postProcessing/" + object + "/" + startName;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        std::string file = dir + "/" + name + ".dat";
        if (std::filesystem::exists(file)) file = dir + "/" + name + "_" + startName + ".dat";
        os_.open(file);
        if (!os_) return false;
        os_.setf(std::ios_base::scientific, std::ios_base::floatfield);
        os_.precision(precision_);
        os_.width(width());
        return true;
    }

    std::ofstream& os() { return os_; }
    int width() const { return precision_ + 8; }

    void commented(const std::string& s)
    {
        os_ << std::setw(1) << "#";
        if (!s.empty())
        {
            os_ << std::setw(1) << ' ';
            os_.setf(std::ios_base::left, std::ios_base::adjustfield);
            os_ << std::setw(width() - 2) << s.c_str();
        }
    }

    void header(const std::string& s)
    {
        commented(s);
        os_ << '\n';
    }

    void headerValue(
        const std::string& property,
        const std::string& value)
    {
        os_ << std::setw(1) << '#' << std::setw(1) << ' ';
        os_.setf(std::ios_base::left, std::ios_base::adjustfield);
        os_ << std::setw(width() - 2) << property.c_str() << std::setw(1) << ':' << std::setw(1) << ' ' << value << '\n';
    }

    void tabbed(const std::string& s)
    {
        os_ << '\t' << std::setw(width()) << s.c_str();
    }

    void time(const std::string& timeName)
    {
        os_ << std::setw(width()) << timeName;
    }

    // a value as the stream writes it (for a header's vector or scalar)
    std::string text(scalar v) const
    {
        std::ostringstream s;
        s.setf(std::ios_base::scientific, std::ios_base::floatfield);
        s.precision(precision_);
        s << v;
        return s.str();
    }

    std::string text(const vector& v) const
    {
        return "(" + text(v.x) + " " + text(v.y) + " " + text(v.z) + ")";
    }

private:
    std::ofstream os_;
    int precision_ = 6;
};

} // namespace forcesDetail

class ForcesFO : public FunctionObject
{
public:
    // The object, or null when its dictionary asks for what brae does not carry (said so, and counted as not run)
    static std::unique_ptr<FunctionObject> make(
        const std::string& name,
        const FoamDict& d,
        bool coeffs,
        const ForcesContext& c)
    {
        std::unique_ptr<ForcesFO> fo(new ForcesFO(name, coeffs, c));
        const std::string why = fo->read(d);
        if (!why.empty())
        {
            noticeIgnored("functions/" + name, std::string(coeffs ? "forceCoeffs" : "forces") + ": " + why + ". This functionObject is NOT run.");
            return nullptr;
        }
        return fo;
    }

    const std::string& name() const override { return name_; }

    bool execute() override
    {
        if (!ready_ && !initialise()) return true;
        const std::vector<double> perPatch = solver_->forcePatchSums(faces_, rhoP_, rhoV_, pRefK_, frame_.origin);
        // forces.C addToPatchFields: the patches' sums added in the set's ascending order
        double s[18] = {};
        const std::size_t nPatches = perPatch.size() / 12;
        for (std::size_t p = 0; p < nPatches; ++p)
            for (int k = 0; k < 12; ++k) s[k] += perPatch[12 * p + static_cast<std::size_t>(k)];
        // reduce(sumPatchForcesP_, sumOp<vector>()) and the others, over the ranks
        if (ctx_.parallel.sum) ctx_.parallel.sum(s, 18);
        fP_ = vector{s[0], s[1], s[2]};
        fV_ = vector{s[3], s[4], s[5]};
        mP_ = vector{s[6], s[7], s[8]};
        mV_ = vector{s[9], s[10], s[11]};
        fI_ = vector{s[12], s[13], s[14]};
        mI_ = vector{s[15], s[16], s[17]};
        if (coeffs_) computeCoeffs();
        if (log_ && ctx_.parallel.master) logValues();
        return true;
    }

    bool write() override
    {
        if (!ready_ || !writeToFile_ || !ctx_.parallel.master) return true;
        if (coeffs_) return writeCoeffs();
        return writeForces();
    }

private:
    ForcesFO(
        const std::string& name,
        bool coeffs,
        const ForcesContext& c)
      : name_(name),
        coeffs_(coeffs),
        ctx_(c)
    {
    }

    // forces::read, forceCoeffs::read and setCoordinateSystem: "" when every entry is one brae carries
    std::string read(const FoamDict& d)
    {
        using namespace forcesDetail;
        if (d.switchOr("directForceDensity", false)) return "directForceDensity: the solvers hold no force-density field (fD)";
        if (d.switchOr("writeFields", false)) return "writeFields: the force and moment fields are not written by brae";
        if (d.wordOr("p", "p") != "p") return "p names '" + d.wordOr("p", "p") + "': the solver's pressure is p";
        if (d.wordOr("U", "U") != "U") return "U names '" + d.wordOr("U", "U") + "': the solver's velocity is U";
        const std::string rho = d.wordOr("rho", "rho");
        if (rho != "rhoInf")
            return "rho " + rho + " on an incompressible solver: OpenFOAM stops (forces.C initialise, \"Could not find rho\"); "
                   "set rho rhoInf and rhoInf";
        if (!d.found("rhoInf")) return "rho rhoInf needs rhoInf, which OpenFOAM reads with no default (forces.C read)";
        rhoInf_ = d.scalarOr("rhoInf", 0);
        if (!(rhoInf_ >= 1e-15)) return "rhoInf must be at least SMALL (1e-15)";
        // a kinematic p: forces::rho(p) is rhoInf, and pRef is taken per unit density
        rhoP_ = rhoInf_;
        rhoV_ = rhoInf_;
        pRefK_ = d.scalarOr("pRef", 0) / rhoInf_;
        porosity_ = d.switchOr("porosity", false);
        if (porosity_ && ctx_.casePorosity)
            return "porosity on a case with porosity models: their resistance force is not computed by brae";
        if (porosity_)
            noticeApproximated("functions/" + name_, "porosity: the case has no porosity model, so the porous columns are zero (OpenFOAM warns "
                               "\"Porosity effects requested, but no porosity models found\")");
        // grad(U): the forms the solver carries
        const std::string g = ctx_.gradUScheme;
        static const std::regex gauss(R"(^\s*Gauss\s+linear\s*$)");
        static const std::regex limited(R"(^\s*cellLimited\s+Gauss\s+linear\s+[-+0-9.eE]+\s*$)");
        if (!g.empty() && !std::regex_match(g, gauss) && !std::regex_match(g, limited))
            return "grad(U) '" + g + "': the viscous force takes fvc::grad(U) under the case's scheme, and brae carries Gauss linear and "
                   "cellLimited Gauss linear";
        const std::string why = patchMatcherFrom(d, patches_);
        if (!why.empty()) return why;
        // the coordinate system: a coordinateSystem dictionary, else CofR and the two axes, else this dictionary's own
        std::string cs;
        if (const FoamDict* sys = d.subDict("coordinateSystem"))
        {
            const std::string type = sys->wordOr("type", "cartesian");
            if (type != "cartesian") return "coordinateSystem type " + type + ": forces are evaluated in a cartesian system";
            cs = systemFrom(*sys, true, frame_);
        }
        else if (d.found("CofR"))
        {
            bool ok = true;
            frame_.origin = vec3(d.scalarListOr("CofR", {}), ok);
            vector e3{0, 0, 1};
            vector e1{1, 0, 0};
            if (coeffs_)
            {
                if (!d.found("liftDir") || !d.found("dragDir")) return "forceCoeffs with CofR needs liftDir and dragDir (forceCoeffs.C)";
                e3 = vec3(d.scalarListOr("liftDir", {}), ok);
                e1 = vec3(d.scalarListOr("dragDir", {}), ok);
            }
            if (!ok) return "CofR, liftDir or dragDir is not a vector";
            cs = axesFrame(e3, e1, 2, frame_);
        }
        else
        {
            cs = systemFrom(d, false, frame_);
        }
        if (!cs.empty()) return "coordinate system: " + cs;
        if (std::getenv("BRAE_CONTROL_FORCES_GLOBAL_AXES"))
        {
            frame_.globalAxes = true;
            std::printf("  *** CONTROL BRAE_CONTROL_FORCES_GLOBAL_AXES: %s reports global components. This run is deliberately wrong. ***\n",
                        name_.c_str());
        }
        writePrecision_ = d.intOr("writePrecision", ctx_.controlDictPrecision);
        writeToFile_ = d.switchOr("writeToFile", true);
        log_ = d.switchOr("log", true);
        if (d.found("binData") || d.subDict("binData"))
            noticeIgnored("functions/" + name_ + "/binData", "not an entry of OpenFOAM v2412's forces (binning is the binField function "
                          "object): OpenFOAM does not read it, nor does brae");
        if (coeffs_)
        {
            for (const char* k : {"magUInf", "lRef", "Aref"})
                if (!d.found(k)) return std::string(k) + " is mandatory (forceCoeffs.C read)";
            magUInf_ = d.scalarOr("magUInf", 0);
            lRef_ = d.scalarOr("lRef", 0);
            Aref_ = d.scalarOr("Aref", 0);
            if (std::min(magUInf_, rhoInf_) < 1e-15 || std::min(lRef_, Aref_) < 1e-15)
                return "non-zero values are required for reference scales (forceCoeffs.C)";
            static const std::vector<std::string> all = {"Cd", "Cd(f)", "Cd(r)", "Cl", "Cl(f)", "Cl(r)", "CmPitch", "CmRoll", "CmYaw",
                                                         "Cs", "Cs(f)", "Cs(r)"};
            if (const std::vector<std::string>* t = d.find("coefficients"))
            {
                // the list's tokens: its own parentheses, names, and a bare Cd(f) split as Cd ( f ) by the tokenizer (a quoted one
                // arrives whole), put back together
                const std::vector<std::string>& v = *t;
                for (std::size_t i = 0; i < v.size(); ++i)
                {
                    std::string w = v[i];
                    if (w == "(" || w == ")") continue;
                    if (i + 3 < v.size() && v[i + 1] == "(" && (v[i + 2] == "f" || v[i + 2] == "r") && v[i + 3] == ")")
                    {
                        w += "(" + v[i + 2] + ")";
                        i += 3;
                    }
                    if (std::find(all.begin(), all.end(), w) == all.end())
                        return "unknown coefficient type " + w + " (Cd Cs Cl CmRoll CmPitch CmYaw and Cd(f) Cs(f) Cl(f) Cd(r) Cs(r) Cl(r))";
                    active_.insert(w);
                }
            }
            else
            {
                active_.insert(all.begin(), all.end());
            }
            // the axes v2412 does not read: said, and a pitchAxis that is not e2 said to differ
            for (const char* k : {"pitchAxis", "rollAxis", "yawAxis", "sideDir"})
            {
                if (!d.found(k)) continue;
                bool ok = true;
                const vector a = vec3(d.scalarListOr(k, {}), ok);
                const vector& used = std::string(k) == "rollAxis" ? frame_.e1 : (std::string(k) == "yawAxis" ? frame_.e3 : frame_.e2);
                const scalar m = mag3(a);
                const bool same = ok && m > 0 && std::fabs(dot3(a, used) / m - 1.0) < 1e-12;
                noticeIgnored("functions/" + name_ + "/" + k,
                              std::string("not read by OpenFOAM v2412's forceCoeffs (the axes are e1 = dragDir, e3 = liftDir, e2 = e3 ^ e1)")
                                  + (same ? "; it is the axis used" : "; IT IS NOT the axis OpenFOAM uses, which is " + DatFile().text(used)));
            }
        }
        return "";
    }

    bool initialise()
    {
        if (!ctx_.registry) return false;
        solver_ = ctx_.registry->lookupObject<DeviceSimpleSolver>("solver");
        const auto* fvp = ctx_.registry->lookupObject<std::vector<FvPatch>>("patches");
        const auto* g = ctx_.registry->lookupObject<FvGeometry>("geometry");
        if (!solver_ || !fvp || !g) return false;
        // the selection in OpenFOAM's summation order: the patches by ascending index, each one's faces in order; a face's place
        // in the solver's boundary arrays counts every non-coupled patch, empty ones included (DeviceMesh's bndCell)
        std::vector<label> bnd;
        std::vector<scalar> Sf, Cf;
        std::vector<std::string> picked;
        label offset = 0;
        for (const FvPatch& p : *fvp)
        {
            if (isCoupledInterfaceType(p.type)) continue;
            if (p.type != "empty" && p.size > 0 && patches_.selects(p))
            {
                for (label i = 0; i < p.size; ++i)
                {
                    bnd.push_back(offset + i);
                    const vector S = g->Sf()[static_cast<std::size_t>(p.start + i)];
                    const vector C = g->Cf()[static_cast<std::size_t>(p.start + i)];
                    Sf.insert(Sf.end(), {S.x, S.y, S.z});
                    Cf.insert(Cf.end(), {C.x, C.y, C.z});
                }
                faces_.patchOffset.push_back(static_cast<int>(bnd.size()));
                picked.push_back(p.name);
            }
            offset += p.size;
        }
        faces_.nFaces = static_cast<int>(bnd.size());
        faces_.bnd.copyFrom(bnd);
        faces_.Sf.copyFrom(Sf);
        faces_.Cf.copyFrom(Cf);
        if (picked.empty() && ctx_.parallel.master && !ctx_.parallel.sum)
            noticeApproximated("functions/" + name_, "patches select no patch with faces: the forces are zero (OpenFOAM warns \"Cannot find any "
                               "patch or group names matching\")");
        ready_ = true;
        return true;
    }

    // forceCoeffs::calcForceCoeffs and calcMomentCoeffs
    void computeCoeffs()
    {
        const scalar pDyn = 0.5 * rhoInf_ * (magUInf_ * magUInf_);
        const scalar fs = 1.0 / (Aref_ * pDyn + 1e-15);
        const scalar ms = 1.0 / (Aref_ * pDyn * lRef_ + 1e-15);
        auto scale = [](scalar a, const vector& v) { return vector{a * v.x, a * v.y, a * v.z}; };
        CfP_ = scale(fs, frame_.local(fP_));
        CfV_ = scale(fs, frame_.local(fV_));
        CfI_ = scale(fs, frame_.local(fI_));
        CmP_ = scale(ms, frame_.local(mP_));
        CmV_ = scale(ms, frame_.local(mV_));
        CmI_ = scale(ms, frame_.local(mI_));
    }

    static scalar cmpt(
        const vector& v,
        int i)
    {
        return i == 0 ? v.x : (i == 1 ? v.y : v.z);
    }

    // coeffDesc::value: (pressure, viscous, internal) of one coefficient
    vector coeffValue(const std::string& n) const
    {
        auto f = [&](int c) { return vector{cmpt(CfP_, c), cmpt(CfV_, c), cmpt(CfI_, c)}; };
        auto m = [&](int c) { return vector{cmpt(CmP_, c), cmpt(CmV_, c), cmpt(CmI_, c)}; };
        auto front = [&](int c0, int c1)
        {
            const vector a = f(c0);
            const vector b = m(c1);
            return vector{0.5 * a.x + b.x, 0.5 * a.y + b.y, 0.5 * a.z + b.z};
        };
        auto rear = [&](int c0, int c1)
        {
            const vector a = f(c0);
            const vector b = m(c1);
            return vector{0.5 * a.x - b.x, 0.5 * a.y - b.y, 0.5 * a.z - b.z};
        };
        if (n == "Cd") return f(0);
        if (n == "Cs") return f(1);
        if (n == "Cl") return f(2);
        if (n == "Cd(f)") return front(0, 0);
        if (n == "Cd(r)") return rear(0, 0);
        if (n == "Cs(f)") return front(1, 2);
        if (n == "Cs(r)") return rear(1, 2);
        if (n == "Cl(f)") return front(2, 1);
        if (n == "Cl(r)") return rear(2, 1);
        if (n == "CmRoll") return m(0);
        if (n == "CmPitch") return m(1);
        return m(2);
    }

    void logValues() const
    {
        if (coeffs_)
        {
            std::printf("forceCoeffs %s write:\n    Coefficient\tTotal\tPressure\tViscous\tInternal\n", name_.c_str());
            for (const std::string& n : active_)
            {
                const vector v = coeffValue(n);
                std::printf("    %s:\t%.*g\t%.*g\t%.*g\t%.*g\n", n.c_str(), writePrecision_, v.x + v.y + v.z, writePrecision_, v.x, writePrecision_, v.y,
                            writePrecision_, v.z);
            }
            return;
        }
        auto line = [this](const char* what, const vector& v)
        {
            std::printf("        %s : (%.*g %.*g %.*g)\n", what, writePrecision_, v.x, writePrecision_, v.y, writePrecision_, v.z);
        };
        std::printf("forces %s write:\n    Sum of forces\n", name_.c_str());
        const vector p = frame_.local(fP_);
        const vector v = frame_.local(fV_);
        line("Total   ", vector{p.x + v.x, p.y + v.y, p.z + v.z});
        line("Pressure", p);
        line("Viscous ", v);
        std::printf("    Sum of moments\n");
        const vector mp = frame_.local(mP_);
        const vector mv = frame_.local(mV_);
        line("Total   ", vector{mp.x + mv.x, mp.y + mv.y, mp.z + mv.z});
        line("Pressure", mp);
        line("Viscous ", mv);
    }

    bool writeCoeffs()
    {
        using forcesDetail::DatFile;
        if (!coeffFile_)
        {
            coeffFile_ = std::make_unique<DatFile>();
            if (!coeffFile_->open(ctx_.caseDir, name_, step().startName, "coefficient", writePrecision_)) return false;
            DatFile& f = *coeffFile_;
            f.header("Force and moment coefficients");
            f.headerValue("dragDir", f.text(frame_.e1));
            f.headerValue("sideDir", f.text(frame_.e2));
            f.headerValue("liftDir", f.text(frame_.e3));
            f.headerValue("rollAxis", f.text(frame_.e1));
            f.headerValue("pitchAxis", f.text(frame_.e2));
            f.headerValue("yawAxis", f.text(frame_.e3));
            f.headerValue("magUInf", f.text(magUInf_));
            f.headerValue("lRef", f.text(lRef_));
            f.headerValue("Aref", f.text(Aref_));
            f.headerValue("CofR", f.text(frame_.origin));
            f.header("");
            f.commented("Time");
            for (const std::string& n : active_) f.tabbed(n);
            f.os() << std::endl;
        }
        DatFile& f = *coeffFile_;
        f.time(step().timeName);
        for (const std::string& n : active_)
        {
            const vector v = coeffValue(n);
            f.os() << '\t' << (v.x + v.y + v.z);
        }
        f.os() << std::endl;
        return true;
    }

    void forceHeader(
        forcesDetail::DatFile& f,
        const std::string& what)
    {
        f.header(what);
        f.headerValue("CofR", f.text(frame_.origin));
        f.header("");
        f.commented("Time");
        f.tabbed("total_x total_y total_z");
        f.tabbed("pressure_x pressure_y pressure_z");
        f.tabbed("viscous_x viscous_y viscous_z");
        if (porosity_) f.tabbed("porous_x porous_y porous_z");
        f.os() << std::endl;
    }

    void forceLine(
        forcesDetail::DatFile& f,
        const vector& p,
        const vector& v,
        const vector& i)
    {
        f.time(step().timeName);
        const vector t{p.x + v.x + i.x, p.y + v.y + i.y, p.z + v.z + i.z};
        for (const vector& w : {t, p, v})
            f.os() << ' ' << w.x << ' ' << w.y << ' ' << w.z;
        if (porosity_) f.os() << ' ' << i.x << ' ' << i.y << ' ' << i.z;
        f.os() << std::endl;
    }

    bool writeForces()
    {
        using forcesDetail::DatFile;
        if (!forceFile_)
        {
            forceFile_ = std::make_unique<DatFile>();
            momentFile_ = std::make_unique<DatFile>();
            if (!forceFile_->open(ctx_.caseDir, name_, step().startName, "force", writePrecision_)) return false;
            if (!momentFile_->open(ctx_.caseDir, name_, step().startName, "moment", writePrecision_)) return false;
            forceHeader(*forceFile_, "Force");
            forceHeader(*momentFile_, "Moment");
        }
        forceLine(*forceFile_, frame_.local(fP_), frame_.local(fV_), frame_.local(fI_));
        forceLine(*momentFile_, frame_.local(mP_), frame_.local(mV_), frame_.local(mI_));
        return true;
    }

    std::string name_;
    bool coeffs_ = false;
    ForcesContext ctx_;
    forcesDetail::PatchMatcher patches_;
    ForcesFrame frame_;
    scalar rhoInf_ = 1;
    scalar rhoP_ = 1;
    scalar rhoV_ = 1;
    scalar pRefK_ = 0;
    bool porosity_ = false;
    int writePrecision_ = 6;
    bool writeToFile_ = true;
    bool log_ = true;
    scalar magUInf_ = 0;
    scalar lRef_ = 0;
    scalar Aref_ = 0;
    std::set<std::string> active_;
    bool ready_ = false;
    DeviceSimpleSolver* solver_ = nullptr;
    DeviceForceFaces faces_;
    vector fP_{0, 0, 0};
    vector fV_{0, 0, 0};
    vector fI_{0, 0, 0};
    vector mP_{0, 0, 0};
    vector mV_{0, 0, 0};
    vector mI_{0, 0, 0};
    vector CfP_{0, 0, 0};
    vector CfV_{0, 0, 0};
    vector CfI_{0, 0, 0};
    vector CmP_{0, 0, 0};
    vector CmV_{0, 0, 0};
    vector CmI_{0, 0, 0};
    std::unique_ptr<forcesDetail::DatFile> coeffFile_;
    std::unique_ptr<forcesDetail::DatFile> forceFile_;
    std::unique_ptr<forcesDetail::DatFile> momentFile_;
};

} // namespace brae
