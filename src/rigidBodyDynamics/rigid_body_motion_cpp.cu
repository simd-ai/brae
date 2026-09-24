#include "rigid_body_motion_cpp.cuh"
#include "foam_field_reader.cuh"
#include "foam_token_reader.cuh"
#include "primitive_patch_cpp.cuh"
#include "septernion_cpp.cuh"
#include <cmath>
#include <stdexcept>

namespace brae {
namespace RBD {

namespace {

// TENSOR PRODUCTS, local rather than added to cf_types.cuh: spatialTransform is the only caller in the
// tree and the shared header is included by every translation unit.
// OF's inner products: (A & B)_ij = A_ik B_kj, (T & v)_i = T_ij v_j.
tensor tdot(const tensor& a, const tensor& b)
{
    return tensor{
        a.xx*b.xx + a.xy*b.yx + a.xz*b.zx,  a.xx*b.xy + a.xy*b.yy + a.xz*b.zy,  a.xx*b.xz + a.xy*b.yz + a.xz*b.zz,
        a.yx*b.xx + a.yy*b.yx + a.yz*b.zx,  a.yx*b.xy + a.yy*b.yy + a.yz*b.zy,  a.yx*b.xz + a.yy*b.yz + a.yz*b.zz,
        a.zx*b.xx + a.zy*b.yx + a.zz*b.zx,  a.zx*b.xy + a.zy*b.yy + a.zz*b.zy,  a.zx*b.xz + a.zy*b.yz + a.zz*b.zz};
}

vector tdotv(const tensor& t, const vector& v)
{
    return vector{t.xx*v.x + t.xy*v.y + t.xz*v.z,
                  t.yx*v.x + t.yy*v.y + t.yz*v.z,
                  t.zx*v.x + t.zy*v.y + t.zz*v.z};
}

}   // namespace


SpatialTransform operator&(const SpatialTransform& a, const SpatialTransform& b)
{
    // spatialTransformI.H:137-143: spatialTransform(E_ & X.E_, X.r_ + (r_ & X.E_))
    SpatialTransform out;
    out.E = tdot(a.E, b.E);
    out.r = b.r + dot(a.r, b.E);
    return out;
}


SpatialTransform inv(const SpatialTransform& x)
{
    // spatialTransformI.H:106-109: spatialTransform(E_.T(), -(E_ & r_))
    SpatialTransform out;
    out.E = transpose(x.E);
    out.r = scalar(-1)*tdotv(x.E, x.r);
    return out;
}


vector transformPoint(const SpatialTransform& x, const vector& p)
{
    // spatialTransformI.H:172-178: E_ & (p - r_)
    return tdotv(x.E, p - x.r);
}


SpatialTransform Xt(const vector& r)
{
    SpatialTransform out;
    out.r = r;
    return out;
}


SpatialTransform Xry(scalar omega)
{
    // transform.H:99-109 -- Ry(omega) = (c 0 -s; 0 1 0; s 0 c)
    const scalar s = std::sin(omega);
    const scalar c = std::cos(omega);
    SpatialTransform out;
    out.E = tensor{c, 0, -s, 0, 1, 0, s, 0, c};
    out.r = vector{0, 0, 0};
    return out;
}


std::vector<SpatialTransform> Model::X0(const std::vector<scalar>& q) const
{
    if (q.size() != static_cast<std::size_t>(nDoF()))
    {
        throw std::runtime_error("brae RBD::Model::X0: the joint state has " + std::to_string(q.size())
                                 + " values and the chain has " + std::to_string((long)nDoF())
                                 + " degrees of freedom.");
    }
    // X0_[0] is the root's, which initializeRootBody leaves the identity
    std::vector<SpatialTransform> x0(links.size() + 1);
    for (std::size_t i = 0; i < links.size(); ++i)
    {
        const Link& l = links[i];
        // joint::jcalc -- J.X
        SpatialTransform JX;
        if (l.joint == JointType::Py)
        {
            // Py.C:44: J.X = Xt(S_[0].l()*q), and S_[0].l() is (0 1 0)
            JX = Xt(vector{0, q[static_cast<std::size_t>(l.qIndex)], 0});
        }
        else
        {
            JX = Xry(q[static_cast<std::size_t>(l.qIndex)]);
        }
        const SpatialTransform Xlambda = JX & l.XT;
        x0[i + 1] = (l.lambda != 0) ? (Xlambda & x0[static_cast<std::size_t>(l.lambda)]) : Xlambda;
    }
    return x0;
}


std::vector<vector> Model::transformPoints(
    const std::vector<scalar>& q,
    const std::vector<scalar>& weight,
    const std::vector<vector>& points0) const
{
    const std::vector<SpatialTransform> x0 = X0(q);
    const std::vector<SpatialTransform> x00 = X0(std::vector<scalar>(static_cast<std::size_t>(nDoF()), scalar(0)));
    const std::size_t b = static_cast<std::size_t>(bodyID());
    // rigidBodyMotion.C:259-262: the transform from the initial state in the global frame to the
    // current one
    const SpatialTransform X = inv(x0[b]) & x00[b];

    // ...and its septernion, for the slerp. septernionI.H:59-63 takes the translation from X.r() and
    // the rotation from X.E() through quaternion(tensor).
    const Septernion s(X.r, Quaternion(X.E));
    const Septernion I;

    // SMALL, OpenFOAM's for a double build
    const scalar small_ = scalar(1e-15);
    std::vector<vector> out(points0);
    for (std::size_t i = 0; i < points0.size() && i < weight.size(); ++i)
    {
        const scalar w = weight[i];
        if (w <= small_) continue;                          // a stationary point is left untouched
        if (w > scalar(1) - small_)
        {
            out[i] = transformPoint(X, points0[i]);         // solid-body motion where the weight is 1
        }
        else
        {
            const Septernion sw = slerp(I, s, w);
            out[i] = sw.r.transform(points0[i] - sw.t);     // septernion::transformPoint
        }
    }
    return out;
}


void constrainPointDisplacement(
    const std::string&          fieldPath,
    const PrimitiveMesh&        m,
    const std::vector<FvPatch>& patches,
    std::vector<vector>&        displacement)
{
    const FieldData<vector> fd = readField<vector>(fieldPath);
    for (const FvPatch& q : patches)
    {
        const PatchFieldData<vector>* entry = nullptr;
        for (const PatchFieldData<vector>& b : fd.boundary)
        {
            if (b.name == q.name) entry = &b;
        }
        if (!entry)
        {
            throw std::runtime_error(
                "brae RBD::constrainPointDisplacement: " + fieldPath + " has no entry for the patch `"
                + q.name + "`.");
        }
        if (entry->type == "calculated")
        {
            // pointPatchField::evaluate is a no-op for it: the transform's value stands
            continue;
        }
        if (entry->type != "fixedValue")
        {
            throw std::runtime_error(
                "brae RBD::constrainPointDisplacement: the point patch `" + q.name + "` is `"
                + entry->type + "`. Only fixedValue (which pins the points it owns) and calculated "
                "(which does not evaluate) are ported; every other pointPatchField writes something "
                "of its own into the shared point field and would move the mesh differently.");
        }
        if (!entry->valueUniform)
        {
            throw std::runtime_error(
                "brae RBD::constrainPointDisplacement: the point patch `" + q.name + "` is a fixedValue "
                "with a per-point value; only a uniform one is ported.");
        }
        const PrimitivePatchAddressing addr = primitivePatch(m, faceRange(q.start, q.size));
        for (const label mp : addr.meshPoints)
        {
            displacement[static_cast<std::size_t>(mp)] = entry->uniformValue;
        }
    }
}


void newmarkSolve(
    const ModelState&    s0,
    const NewmarkCoeffs& c,
    ModelState&          s)
{
    if (s.q.size() != s0.q.size() || s.qDot.size() != s0.qDot.size()
     || s.qDdot.size() != s0.qDdot.size())
    {
        throw std::runtime_error("brae RBD::newmarkSolve: the two states have different sizes.");
    }
    // Newmark.C:91-97, in OpenFOAM's own order: qDot FIRST, then q -- and q reads qDot0, not the qDot
    // this line has just written, so swapping them changes the answer.
    const scalar dt = s.deltaT;
    for (std::size_t i = 0; i < s.q.size(); ++i)
    {
        s.qDot[i] = s0.qDot[i] + dt*(c.gamma*s.qDdot[i] + (scalar(1) - c.gamma)*s0.qDdot[i]);
    }
    for (std::size_t i = 0; i < s.q.size(); ++i)
    {
        s.q[i] = s0.q[i] + dt*s0.qDot[i]
               + dt*dt*(c.beta*s.qDdot[i] + (scalar(0.5) - c.beta)*s0.qDdot[i]);
    }
}


void relaxAcceleration(
    std::vector<scalar>&       qDdot,
    const std::vector<scalar>& qDdotPrev,
    scalar                     aRelax,
    scalar                     aDamp)
{
    if (qDdot.size() != qDdotPrev.size())
    {
        throw std::runtime_error("brae RBD::relaxAcceleration: the two accelerations differ in size.");
    }
    for (std::size_t i = 0; i < qDdot.size(); ++i)
    {
        qDdot[i] = aDamp*(aRelax*qDdot[i] + (scalar(1) - aRelax)*qDdotPrev[i]);
    }
}


MotionSpec readMotionSpec(const std::string& dictPath)
{
    // A DEDICATED READER, and not FoamDict, for one reason: the joint chain is a LIST OF
    // DICTIONARIES -- `joints ( { type Py; } { type Ry; } )` -- and FoamDict's parser flattens it
    // (measured: the two entries come back as a leaf named `{`, and the body's own `patches`,
    // `innerDistance` and `outerDistance` are hoisted to the coeffs level). Teaching the shared parser
    // list-of-dictionaries is its own unit; this walks the tokens for the shape this one needs.
    // expandVars: the tutorial writes `mass #eval{ $rho*$Lx*$Ly*$Lz };`, and the tokeniser evaluates
    // it eagerly -- without the $-macros expanded first it stops on the `$`.
    TokenStream ts(dictPath, /*expandVars=*/true);
    MotionSpec spec;
    std::string motionSolver;

    // walk to `rigidBodyMotionCoeffs {`
    auto skipBlock = [&]()
    {
        label depth = 0;
        do
        {
            const std::string t = ts.next();
            if (t == "{") ++depth;
            else if (t == "}") --depth;
        } while (depth > 0 && !ts.eof());
    };

    while (!ts.eof())
    {
        const std::string key = ts.next();
        if (key == "motionSolver" || key == "solver")
        {
            motionSolver = ts.next();
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "rigidBodyMotionCoeffs")
        {
            ts.expect("{");
            break;
        }
        if (key == "{") { skipBlock(); continue; }
    }
    if (motionSolver != "rigidBodyMotion")
    {
        throw std::runtime_error(
            "brae RBD::readMotionSpec: " + dictPath + " asks for `motionSolver " + motionSolver +
            "`; this reader carries rigidBodyMotion only.");
    }

    auto readVectorParen = [&]()
    {
        ts.expect("(");
        vector v{0, 0, 0};
        v.x = ts.nextScalar();
        v.y = ts.nextScalar();
        v.z = ts.nextScalar();
        ts.expect(")");
        return v;
    };

    std::vector<JointType> joints;
    SpatialTransform bodyXT;
    bool haveBody = false;
    bool haveSolver = false;

    // inside rigidBodyMotionCoeffs
    label depth = 1;
    while (!ts.eof() && depth > 0)
    {
        const std::string key = ts.next();
        if (key == "}") { --depth; continue; }
        if (key == "innerDistance") { spec.innerDistance = ts.nextScalar(); if (ts.peek() == ";") ts.next(); continue; }
        if (key == "outerDistance") { spec.outerDistance = ts.nextScalar(); if (ts.peek() == ";") ts.next(); continue; }
        if (key == "patches")
        {
            ts.expect("(");
            while (ts.peek() != ")") spec.patches.push_back(ts.next());
            ts.expect(")");
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "transform")
        {
            ts.expect("(");
            scalar e[9];
            for (scalar& c : e) c = ts.nextScalar();
            ts.expect(")");
            bodyXT.E = tensor{e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]};
            bodyXT.r = readVectorParen();
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "type" && !joints.empty()) { ts.next(); if (ts.peek() == ";") ts.next(); continue; }
        if (key == "parent")
        {
            const std::string p = ts.next();
            if (p != "root")
            {
                throw std::runtime_error(
                    "brae RBD::readMotionSpec: body `" + spec.bodyName + "` hangs off `" + p +
                    "`; only a body joined to `root` is ported.");
            }
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "mergeWith")
        {
            throw std::runtime_error(
                "brae RBD::readMotionSpec: " + dictPath + " merges a body into another (`mergeWith`), "
                "which gives it the master's joint and a composed transform (rigidBodyModel.C:118-160). "
                "Not ported.");
        }
        if (key == "joints")
        {
            ts.expect("(");
            while (ts.peek() != ")")
            {
                ts.expect("{");
                std::string type;
                while (ts.peek() != "}")
                {
                    const std::string k = ts.next();
                    if (k == "type") { type = ts.next(); if (ts.peek() == ";") ts.next(); }
                    else if (ts.peek() != "}") ts.next();
                }
                ts.expect("}");
                if (type == "Py") joints.push_back(JointType::Py);
                else if (type == "Ry") joints.push_back(JointType::Ry);
                else
                {
                    throw std::runtime_error(
                        "brae RBD::readMotionSpec: the joint `" + type + "` is not ported. This unit "
                        "carries Py and Ry, which is what RAS/floatingObject's composite names; "
                        "OpenFOAM has seventeen and each has its own jcalc.");
                }
            }
            ts.expect(")");
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "bodies")
        {
            ts.expect("{");
            spec.bodyName = ts.next();
            if (spec.bodyName == "}")
            {
                throw std::runtime_error("brae RBD::readMotionSpec: the `bodies` dictionary is empty.");
            }
            haveBody = true;
            ts.expect("{");
            depth += 2;
            continue;
        }
        if (key == "solver")
        {
            // DESCENDED INTO, not skipped: it names the integrator and may set gamma and beta.
            ts.expect("{");
            std::string type;
            scalar gamma = scalar(0.5);
            scalar betaEntry = scalar(0.25);
            while (!ts.eof() && ts.peek() != "}")
            {
                const std::string k = ts.next();
                if (k == "type")       { type = ts.next(); }
                else if (k == "gamma") { gamma = ts.nextScalar(); }
                else if (k == "beta")  { betaEntry = ts.nextScalar(); }
                else if (ts.peek() != "}") { ts.next(); }
                if (ts.peek() == ";") ts.next();
            }
            ts.expect("}");
            if (ts.peek() == ";") ts.next();
            if (type != "Newmark")
            {
                throw std::runtime_error(
                    "brae RBD::readMotionSpec: the rigid-body solver `" + type + "` is not ported. "
                    "Only Newmark is: `symplectic` half-steps the velocity with the PREVIOUS step's "
                    "deltaT and `CrankNicolson` carries its own off-centring, and neither is the same "
                    "integrator.");
            }
            spec.newmark = newmarkCoeffs(gamma, betaEntry);
            haveSolver = true;
            continue;
        }
        if (key == "accelerationRelaxation")
        {
            // a Function1: a bare constant, or the `table ( (t v) ... )` the tutorial writes
            if (ts.peek() == "table")
            {
                ts.next();
                ts.expect("(");
                std::vector<std::pair<scalar, scalar>> pts;
                while (!ts.eof() && ts.peek() != ")")
                {
                    ts.expect("(");
                    const scalar tt = ts.nextScalar();
                    const scalar vv = ts.nextScalar();
                    ts.expect(")");
                    pts.emplace_back(tt, vv);
                }
                ts.expect(")");
                spec.accelerationRelaxation = Function1::table(std::move(pts));
            }
            else
            {
                if (ts.peek() == "constant") ts.next();
                spec.accelerationRelaxation = Function1::constant(ts.nextScalar());
            }
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "accelerationDamping")
        {
            spec.accelerationDamping = ts.nextScalar();
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "report")
        {
            const std::string v = ts.next();
            spec.report = (v == "on" || v == "yes" || v == "true" || v == "1");
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "ramp" || key == "cOfGdisplacement" || key == "bodyIdCofG" || key == "test"
         || key == "nIter" || key == "restraints")
        {
            throw std::runtime_error(
                "brae RBD::readMotionSpec: `" + key + "` is set in " + dictPath + ", and this port "
                "does not carry it. `ramp` scales BOTH gravity and the fluid spatial force; "
                "`cOfGdisplacement` accumulates the body's travel into a registered field; `test` "
                "runs the dynamics with no fluid force at all; `nIter` iterates the force and the "
                "relaxation within one mesh update; `restraints` add their own forces inside the "
                "solver. Each changes the answer and none is ported.");
        }
        if (key == "joint")
        {
            // DESCENDED INTO, not skipped: the composite's `joints` list is inside it
            ts.expect("{");
            ++depth;
            continue;
        }
        if (key == "{") { ++depth; continue; }
        // ANY OTHER ENTRY. A sub-dictionary is skipped WHOLE -- `solver { type Newmark; }` read as a
        // leaf swallowed its closing brace and ended the walk one level early, which is how this
        // reader first reported "names no body" on a dictionary that has one.
        if (ts.peek() == "{") { skipBlock(); continue; }
        while (!ts.eof() && ts.peek() != ";" && ts.peek() != "}")
        {
            if (ts.peek() == "{") skipBlock();
            else ts.next();
        }
        if (ts.peek() == ";") ts.next();
    }

    if (!haveBody)
    {
        throw std::runtime_error("brae RBD::readMotionSpec: " + dictPath + " names no body.");
    }
    if (!haveSolver)
    {
        throw std::runtime_error(
            "brae RBD::readMotionSpec: " + dictPath + " names no `solver`. OpenFOAM's default is "
            "`Newmark` (rigidBodySolver::New), but an unstated integrator is a choice this port will "
            "not make for a case.");
    }
    if (spec.accelerationRelaxation.empty())
    {
        spec.accelerationRelaxation = Function1::constant(scalar(1));
    }
    if (joints.empty())
    {
        throw std::runtime_error(
            "brae RBD::readMotionSpec: body `" + spec.bodyName + "` has no `composite` joint list. A "
            "single-joint body is a different chain (no jointBody is inserted) and is not ported.");
    }

    // rigidBodyModel::join for a composite: a massless jointBody for every joint but the last, the
    // body's own transform on the FIRST link and the identity on the rest
    spec.model.links.resize(joints.size());
    for (std::size_t j = 0; j < joints.size(); ++j)
    {
        Link& l = spec.model.links[j];
        l.joint = joints[j];
        l.qIndex = static_cast<label>(j);
        l.lambda = static_cast<label>(j);          // body j + 1's parent is body j (root is 0)
        l.XT = (j == 0) ? bodyXT : SpatialTransform();
    }
    return spec;
}

} // namespace RBD
} // namespace brae
