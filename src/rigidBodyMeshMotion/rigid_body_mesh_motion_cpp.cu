#include "rigid_body_mesh_motion_cpp.cuh"

#include "foam_dict.cuh"
#include "mesh_edges_cpp.cuh"
#include "point_patch_dist_cpp.cuh"
#include "foam_token_reader.cuh"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <regex>
#include <stdexcept>

namespace brae {

namespace {

const char* const WHO = "brae rigidBodyMotion: ";

// The class a file's FoamFile header names, or "" when the file does not open with one -- read from the
// raw bytes, since TokenStream drops the header. IOobject::readHeader takes the FIRST token, which must
// be the word FoamFile, then the header dictionary (IOobjectReadHeader.C:104-125).
std::string headerClass(const std::string& path)
{
    const std::vector<char> raw = gzSlurp(path);
    std::string text;
    text.reserve(raw.size());
    // comments are whitespace to the tokenizer
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        if (raw[i] == '/' && i + 1 < raw.size() && raw[i + 1] == '/')
        {
            while (i < raw.size() && raw[i] != '\n')
            {
                ++i;
            }
            text += ' ';
            continue;
        }
        if (raw[i] == '/' && i + 1 < raw.size() && raw[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < raw.size() && !(raw[i] == '*' && raw[i + 1] == '/'))
            {
                ++i;
            }
            ++i;
            text += ' ';
            continue;
        }
        text += raw[i];
    }
    static const std::regex header(R"(^\s*FoamFile\s*\{([^}]*)\})");
    std::smatch m;
    if (!std::regex_search(text, m, header))
    {
        return "";
    }
    static const std::regex classEntry(R"((?:^|[\s;])class\s+([A-Za-z_][\w:<>,]*)\s*;)");
    std::smatch c;
    const std::string body = m[1].str();
    if (!std::regex_search(body, c, classEntry))
    {
        return "";
    }
    return c[1].str();
}

}   // namespace


std::unique_ptr<RigidBodyMeshMotion> RigidBodyMeshMotion::New(
    const std::string& caseDir,
    const std::string& startDir)
{
    std::unique_ptr<RigidBodyMeshMotion> s(new RigidBodyMeshMotion);
    // the body, the chain, the solver and the relaxation, with every unported entry refused by name
    s->spec_ = RBD::readMotionSpec(caseDir + "/constant/dynamicMeshDict");

    // rigidBodyMeshMotion.C:121 -- `rho` names the density field the forces are taken with, and only
    // `rho rhoInf` makes it a constant. interFoam's rho is the live mixture density, which is what the
    // body floats on, so a reference density here would be a different problem.
    {
        const FoamDict dm = readDict(caseDir + "/constant/dynamicMeshDict");
        const FoamDict* coeffs = dm.optionalSubDict("rigidBodyMotionCoeffs");
        const std::string rhoName = coeffs ? coeffs->wordOr("rho", "rho") : std::string("rho");
        if (rhoName != "rho")
        {
            throw std::runtime_error(
                std::string(WHO) + "rigidBodyMotionCoeffs asks for `rho " + rhoName + "`. Only the "
                "live density field is ported: with `rho rhoInf` OpenFOAM takes the force at a "
                "constant reference density, and a floating body's buoyancy is exactly what that "
                "throws away.");
        }
    }

    // rigidBodyMeshMotion.C:260-264, evaluated on EVERY solve: the model's gravity is
    // constant/g's value, and a body whose gravity is silently zero floats forever.
    const std::string gPath = caseDir + "/constant/g";
    if (!std::filesystem::exists(gPath))
    {
        throw std::runtime_error(
            std::string(WHO) + "the case has no constant/g. OpenFOAM leaves rigidBodyModel::g() at "
            "zero when no `g` is registered (rigidBodyModel.C:172), so the body would feel no weight "
            "and the run would be silently wrong rather than refused.");
    }
    const FoamDict gd = readDict(gPath);
    const std::vector<scalar> gv = gd.scalarListOr("value", {});
    if (gv.size() != 3)
    {
        throw std::runtime_error(std::string(WHO) + "constant/g needs `value (gx gy gz);`.");
    }
    s->g_ = vector{gv[0], gv[1], gv[2]};

    // rigidBodyMeshMotion.C:93-118: the state comes from <startTime>/uniform/rigidBodyMotionState when
    // typeHeaderOk<IOdictionary>(true) holds -- the file there, plain or .gz (POSIX.C:870-876, the plain
    // one first as gzSlurp takes it), opening with a FoamFile header whose class is `dictionary`
    // (IOobjectReadHeader.C:104-125, 218-232). Otherwise OpenFOAM warns and takes it from coeffDict().
    const std::string statePath = startDir + "/uniform/rigidBodyMotionState";
    const std::size_t nD = static_cast<std::size_t>(s->spec_.model.nDoF());
    s->state_.q.assign(nD, scalar(0));
    s->state_.qDot.assign(nD, scalar(0));
    s->state_.qDdot.assign(nD, scalar(0));
    s->state_.t = scalar(-1);
    s->state_.deltaT = scalar(0);
    bool stateFromFile = false;
    if (std::filesystem::exists(statePath) || std::filesystem::exists(statePath + ".gz"))
    {
        stateFromFile = (headerClass(statePath) == "dictionary");
        if (!stateFromFile)
        {
            std::fprintf(stderr, "brae rigidBodyMeshMotion: %s has no FoamFile header of class `dictionary`; "
                                 "as OpenFOAM does, it is not read and the state comes from the coeffs\n",
                         statePath.c_str());
        }
    }
    if (stateFromFile)
    {
        for (const auto& pair : {std::make_pair("q", &s->state_.q),
                                 std::make_pair("qDot", &s->state_.qDot),
                                 std::make_pair("qDdot", &s->state_.qDdot)})
        {
            const std::optional<std::vector<scalar>> v = RBD::readJointStateList(statePath, pair.first);
            if (!v)
            {
                continue;
            }
            // rigidBodyModelState.C:58-69: a present list of any other size, empty included, is fatal
            if (v->size() != nD)
            {
                throw std::runtime_error(
                    std::string(WHO) + statePath + " holds a `" + pair.first + "` of "
                    + std::to_string(v->size()) + " where the chain has " + std::to_string(nD)
                    + " degrees of freedom.");
            }
            *pair.second = *v;
        }
        s->state_.t = RBD::readJointStateScalar(statePath, "t", s->state_.t);
        s->state_.deltaT = RBD::readJointStateScalar(statePath, "deltaT", s->state_.deltaT);
    }
    else if (!s->spec_.coeffStateKeys.empty())
    {
        std::string keys;
        for (const std::string& k : s->spec_.coeffStateKeys)
        {
            keys += (keys.empty() ? "`" : ", `") + k + "`";
        }
        throw std::runtime_error(
            std::string(WHO) + "the motion coeffs set " + keys + " and the start time holds no "
            "rigidBodyMotionState, so OpenFOAM starts the body from them (rigidBodyMeshMotion.C:117, "
            "rigidBodyModelState.C:51-55). Not ported: the body would start at rest instead.");
    }
    s->state0_ = s->state_;

    // displacementMotionSolver.C:50-61 reads pointDisplacement MUST_READ: the point patch TYPES are
    // the constraint that ends every solve, and they live in the start time's field.
    s->pointDisplacementPath_ = startDir + "/pointDisplacement";
    if (!std::filesystem::exists(s->pointDisplacementPath_)
        && !std::filesystem::exists(s->pointDisplacementPath_ + ".gz"))
    {
        throw std::runtime_error(
            std::string(WHO) + "the start time has no pointDisplacement. OpenFOAM reads it MUST_READ "
            "(displacementMotionSolver.C:57) and its point patch types are what pins the tank's own "
            "walls where the body's blend would otherwise move them.");
    }
    return s;
}


void RigidBodyMeshMotion::attach(
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    const std::vector<vector>&  points0)
{
    points0_ = points0;
    bodyPatches_.clear();
    for (const std::string& want : spec_.patches)
    {
        bool found = false;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].name != want) continue;
            bodyPatches_.push_back(static_cast<label>(pi));
            found = true;
        }
        if (!found)
        {
            throw std::runtime_error(
                std::string(WHO) + "the body names `" + want + "`, which the mesh has no patch for. "
                "OpenFOAM matches the entry as a wordRe against every patch; brae matches one patch "
                "by its literal name.");
        }
    }
    if (bodyPatches_.empty())
    {
        throw std::runtime_error(std::string(WHO) + "the body names no patch for the force to act on.");
    }

    // rigidBodyMeshMotion.C:172-206, ONCE, on points0: the wave is run over the UNDEFORMED mesh and
    // the blend is never refreshed, so a scale rebuilt on the moved mesh would drift with the body.
    const MeshEdges e = buildMeshEdges(m);
    const PointPatchDist pd = pointPatchDist(m, e, patches, bodyPatches_);
    weight_ = rigidBodyMeshMotionScale(pd.distance, spec_.innerDistance, spec_.outerDistance);
    pointDisplacement_.assign(points0_.size(), vector{0, 0, 0});
    pointConstraints_ = PointConstraints::build(pointDisplacementPath_, m, patches, g.Sf());
    attached_ = true;
}


std::vector<vector> RigidBodyMeshMotion::newPoints(
    scalar                      time,
    scalar                      deltaT,
    label                       timeIndex,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches,
    const BodyLoad&             load)
{
    if (!attached_)
    {
        throw std::runtime_error(std::string(WHO) + "newPoints() before attach().");
    }
    if (!load.U || !load.p || !load.rho || !load.nuEff)
    {
        throw std::runtime_error(
            std::string(WHO) + "the mesh motion was asked to move without the fluid load. A rigid "
            "body's points come from its equations of motion, and those need the pressure and the "
            "shear the flow puts on its patches.");
    }
    if (m.nPoints() != static_cast<label>(points0_.size()))
    {
        throw std::runtime_error(
            std::string(WHO) + "the mesh has " + std::to_string(m.nPoints()) + " points where "
            "points0 has " + std::to_string(points0_.size()) + ".");
    }

    // rigidBodyMeshMotion.C:252-256: ONLY the state roll is guarded by the time index. Everything
    // below runs on every outer corrector, from the same start-of-step state -- which is what makes
    // the three calls of a `moveMeshOuterCorrectors yes` step an ITERATION and not three sub-steps.
    if (!haveTimeIndex_ || curTimeIndex_ != timeIndex)
    {
        state0_ = state_;
        curTimeIndex_ = timeIndex;
        haveTimeIndex_ = true;
    }

    // rigidBodyMeshMotion.C:299-309. The force is taken about the GLOBAL ORIGIN (`CofR (0 0 0)`) with
    // pRef 0 and, because interFoam's p carries pressure dimensions, a reference density of 1
    // (forces.C:344-359) -- so the pressure term is Sf*p and the live mixture density is in the
    // viscous one. `spatialVector(momentEff, forceEff)` is ANGULAR FIRST.
    const ForceResult fr = bodyForces(*load.U, *load.p, *load.rho, *load.nuEff, m, g, patches,
                                      bodyPatches_, vector{0, 0, 0}, scalar(0));
    // Instrument: BRAE_TRACE_BODY_LOAD=1 prints the load at every solve, pressure and viscous apart, so two
    // arms can be held against each other at the first one that differs. Costs one getenv per solve.
    if (std::getenv("BRAE_TRACE_BODY_LOAD") != nullptr)
    {
        std::printf("  [body load] index %d t %.17g  Fp (%.17g %.17g %.17g)  Fv (%.17g %.17g %.17g)  "
                    "Mp (%.17g %.17g %.17g)  Mv (%.17g %.17g %.17g)\n",
                    static_cast<int>(timeIndex), static_cast<double>(time),
                    static_cast<double>(fr.pressure.x), static_cast<double>(fr.pressure.y),
                    static_cast<double>(fr.pressure.z), static_cast<double>(fr.viscous.x),
                    static_cast<double>(fr.viscous.y), static_cast<double>(fr.viscous.z),
                    static_cast<double>(fr.momentP.x), static_cast<double>(fr.momentP.y),
                    static_cast<double>(fr.momentP.z), static_cast<double>(fr.momentV.x),
                    static_cast<double>(fr.momentV.y), static_cast<double>(fr.momentV.z));
    }
    std::vector<RBD::SpatialVector> fx(spec_.model.links.size() + 1);
    lastForce_.w = fr.moment();
    lastForce_.l = fr.total();
    fx[static_cast<std::size_t>(spec_.model.bodyID())] = lastForce_;

    // rigidBodyMotion.C:171-197 and Newmark.C:76-100, in OpenFOAM's own order. The dynamics is
    // evaluated at the CURRENT q and qDot -- the previous corrector's, not the start of the step --
    // while the integrator reads the start of the step at both ends.
    state_.t = time;
    state_.deltaT = deltaT;
    if (state0_.deltaT < scalar(1e-30))
    {
        state0_.t = time;
        state0_.deltaT = deltaT;
    }
    // Newmark.C:81-85: the restraints are accumulated onto a COPY of fx, before the dynamics
    const std::vector<RBD::SpatialVector> rfx = spec_.model.applyRestraints(state_.q, state_.qDot, fx);
    const std::vector<scalar> qDdotPrev = state_.qDdot;
    state_.qDdot = spec_.model.forwardDynamics(state_.q, state_.qDot, rfx, g_);
    RBD::relaxAcceleration(state_.qDdot, qDdotPrev, spec_.accelerationRelaxation.value(time),
                           spec_.accelerationDamping);
    RBD::newmarkSolve(state0_, spec_.newmark, state_);

    // BRAE_RBD_TRACE: the body's state and the force it was handed, PER CORRECTOR. The written
    // state is the last corrector's and only at write times, so a disagreement that starts mid-step
    // is invisible in it -- and this trace is what showed that brae tracked OpenFOAM's body to 2e-13
    // for six steps and then jumped four orders in one, at the step where the two codes' pcorr took a
    // different iteration count. Printing changes no arithmetic.
    if (std::getenv("BRAE_RBD_TRACE"))
    {
        std::fprintf(stderr, "[rbd] t %.17g q %.17g %.17g qDot %.17g %.17g qDdot %.17g %.17g "
                     "fx %.17g %.17g %.17g %.17g %.17g %.17g\n",
                     (double)time, (double)state_.q[0], (double)state_.q[1],
                     (double)state_.qDot[0], (double)state_.qDot[1],
                     (double)state_.qDdot[0], (double)state_.qDdot[1],
                     (double)lastForce_.w.x, (double)lastForce_.w.y, (double)lastForce_.w.z,
                     (double)lastForce_.l.x, (double)lastForce_.l.y, (double)lastForce_.l.z);
    }

    // rigidBodyMeshMotion.C:361-388: the displacement is measured from points0, not from where the
    // mesh stands, and the constraint that follows pins every point a fixedValue point patch owns.
    const std::vector<vector> moved = spec_.model.transformPoints(state_.q, weight_, points0_);
    for (std::size_t i = 0; i < points0_.size(); ++i)
    {
        pointDisplacement_[i] = moved[i] - points0_[i];
    }
    pointConstraints_.constrainDisplacement(pointDisplacement_);

    // rigidBodyMeshMotion.C:219: curPoints() = points0 + pointDisplacement
    std::vector<vector> curPoints(points0_.size());
    for (std::size_t i = 0; i < points0_.size(); ++i)
    {
        curPoints[i] = points0_[i] + pointDisplacement_[i];
    }
    return curPoints;
}

}   // namespace brae
