#include "rigid_body_motion_cpp.cuh"
#include "foam_field_reader.cuh"
#include "foam_token_reader.cuh"
#include "primitive_patch_cpp.cuh"
#include "septernion_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace brae {
namespace RBD {

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


SpatialVector motionAction(const SpatialTransform& X, const SpatialVector& v)
{
    return SpatialVector{tdotv(X.E, v.w), tdotv(X.E, v.l - cross(X.r, v.w))};
}


SpatialVector dualAction(const SpatialTransform& X, const SpatialVector& f)
{
    return SpatialVector{tdotv(X.E, f.w - cross(X.r, f.l)), tdotv(X.E, f.l)};
}


SpatialVector transposeAction(const SpatialTransform& X, const SpatialVector& f)
{
    const tensor ET = transpose(X.E);
    const vector ETfl = tdotv(ET, f.l);
    return SpatialVector{tdotv(ET, f.w) + cross(X.r, ETfl), ETfl};
}


SpatialTensor motionTensor(const SpatialTransform& X)
{
    const tensor Erx = tdot(X.E, skew(X.r));
    return blockTensor(X.E, tensor{0, 0, 0, 0, 0, 0, 0, 0, 0}, scalar(-1)*Erx, X.E);
}


SpatialTensor transposeTensor(const SpatialTransform& X)
{
    const tensor ET = transpose(X.E);
    const tensor Erx = tdot(X.E, skew(X.r));
    return blockTensor(ET, scalar(-1)*transpose(Erx), tensor{0, 0, 0, 0, 0, 0, 0, 0, 0}, ET);
}


symmTensor inertiaIo(const RigidBodyInertia& rbi)
{
    // Ioc(m, c) = m*(I*magSqr(c) - sqr(c)); Io = Ic + Ioc
    const scalar mc = magSqr(rbi.c);
    const symmTensor cc = sqr(rbi.c);
    const symmTensor ioc{rbi.m*(mc - cc.xx), rbi.m*(scalar(0) - cc.xy), rbi.m*(scalar(0) - cc.xz),
                         rbi.m*(mc - cc.yy), rbi.m*(scalar(0) - cc.yz), rbi.m*(mc - cc.zz)};
    return rbi.Ic + ioc;
}


SpatialTensor inertiaTensor(const RigidBodyInertia& rbi)
{
    const symmTensor io = inertiaIo(rbi);
    const tensor ioT{io.xx, io.xy, io.xz, io.xy, io.yy, io.yz, io.xz, io.yz, io.zz};
    const tensor mcStar = rbi.m*skew(rbi.c);
    const tensor mI{rbi.m, 0, 0, 0, rbi.m, 0, 0, 0, rbi.m};
    return blockTensor(ioT, mcStar, scalar(-1)*mcStar, mI);
}


SpatialVector inertiaAction(const RigidBodyInertia& rbi, const SpatialVector& v)
{
    // rigidBodyInertiaI.H:192-206, transcribed: m*l - m*(c ^ w) is TWO multiplications by m
    const symmTensor io = inertiaIo(rbi);
    return SpatialVector{(io & v.w) + rbi.m*cross(rbi.c, v.l),
                         rbi.m*v.l - rbi.m*cross(rbi.c, v.w)};
}


RigidBodyInertia cuboidInertia(scalar mass, const vector& centreOfMass, const vector& L)
{
    const scalar mBy12 = mass/scalar(12.0);
    const scalar mSqrLx = mBy12*L.x*L.x;
    const scalar mSqrLy = mBy12*L.y*L.y;
    const scalar mSqrLz = mBy12*L.z*L.z;
    RigidBodyInertia rbi;
    rbi.m = mass;
    rbi.c = centreOfMass;
    rbi.Ic = symmTensor{mSqrLy + mSqrLz, 0, 0, mSqrLx + mSqrLz, 0, mSqrLx + mSqrLy};
    return rbi;
}


void jcalc(
    JointType         joint,
    scalar            q,
    scalar            qDot,
    SpatialTransform& JX,
    SpatialVector&    JS1,
    SpatialVector&    Jv)
{
    if (joint == JointType::Py || joint == JointType::Pz)
    {
        // Py.C:92-95, Pz.C:92-95 -- Xt(S_[0].l()*q), J.v = S_[0]*qDot. The translation is built by
        // SCALING the axis: 0*q carries q's sign bit, which a literal 0 would not.
        const vector axis = (joint == JointType::Py) ? vector{0, 1, 0} : vector{0, 0, 1};
        JX = Xt(axis*q);
        JS1 = SpatialVector{vector{0, 0, 0}, axis};
        Jv = JS1*qDot;
    }
    else
    {
        // Ry.C:92-96 -- Xry(q), S_[0] = (0 1 0  0 0 0), and J.v is built by setting wy alone
        JX = Xry(q);
        JS1 = SpatialVector{vector{0, 1, 0}, vector{0, 0, 0}};
        Jv = SpatialVector{};
        Jv.w.y = qDot;
    }
}


std::vector<scalar> Model::forwardDynamics(
    const std::vector<scalar>&        q,
    const std::vector<scalar>&        qDot,
    const std::vector<SpatialVector>& fx,
    const vector&                     g) const
{
    const std::size_t nB = links.size() + 1;
    if (q.size() != static_cast<std::size_t>(nDoF()) || qDot.size() != q.size())
    {
        throw std::runtime_error("brae RBD::forwardDynamics: the joint state is the wrong size.");
    }
    if (!fx.empty() && fx.size() != nB)
    {
        throw std::runtime_error("brae RBD::forwardDynamics: fx must carry one spatial force per BODY, "
                                 "the root included.");
    }
    std::vector<SpatialTransform> Xlambda(nB), X0v(nB);
    std::vector<SpatialVector> vv(nB), cc(nB), pA(nB), S1(nB), U1(nB), uu(nB), aa(nB);
    std::vector<SpatialTensor> IA(nB);
    std::vector<scalar> Dinv(nB, scalar(0));
    std::vector<scalar> qDdot(static_cast<std::size_t>(nDoF()), scalar(0));

    // PASS 1, forward: the kinematics, the articulated inertia seed and the bias force
    for (std::size_t i = 1; i < nB; ++i)
    {
        const Link& L = links[i - 1];
        const std::size_t qi = static_cast<std::size_t>(L.qIndex);
        SpatialTransform JX;
        SpatialVector JS1;
        SpatialVector Jv;
        jcalc(L.joint, q[qi], qDot[qi], JX, JS1, Jv);
        S1[i] = JS1;
        Xlambda[i] = JX & L.XT;
        const std::size_t lam = static_cast<std::size_t>(L.lambda);
        X0v[i] = (lam != 0) ? (Xlambda[i] & X0v[lam]) : Xlambda[i];
        vv[i] = motionAction(Xlambda[i], vv[lam]) + Jv;
        cc[i] = crossMotion(vv[i], Jv);                     // J.c is Zero for both joints
        IA[i] = inertiaTensor(L.inertia);
        pA[i] = crossDual(vv[i], inertiaAction(L.inertia, vv[i]));
        if (!fx.empty())
        {
            // forwardDynamics.C:105 -- MINUS, through the DUAL transform
            pA[i] = pA[i] - dualAction(X0v[i], fx[i]);
        }
    }

    // PASS 2, backward: the articulated inertia and bias force folded onto the parent
    for (std::size_t i = nB - 1; i >= 1; --i)
    {
        const Link& L = links[i - 1];
        const std::size_t qi = static_cast<std::size_t>(L.qIndex);
        U1[i] = IA[i] & S1[i];
        Dinv[i] = scalar(1)/doubleInner(S1[i], U1[i]);
        uu[i].w.x = scalar(0) - doubleInner(S1[i], pA[i]);      // tau is zero on this path
        const std::size_t lam = static_cast<std::size_t>(L.lambda);
        if (lam != 0)
        {
            const SpatialTensor Ia = IA[i] - outerSpatial(U1[i], U1[i]*Dinv[i]);
            const SpatialVector pa = pA[i] + (Ia & cc[i]) + U1[i]*(Dinv[i]*uu[i].w.x);
            IA[lam] += (transposeTensor(Xlambda[i]) & Ia) & motionTensor(Xlambda[i]);
            pA[lam] = pA[lam] + transposeAction(Xlambda[i], pa);
        }
        if (i == 1) break;
    }

    // PASS 3, forward: gravity as a base acceleration, then the joint accelerations
    aa[0] = SpatialVector{vector{0, 0, 0}, scalar(-1)*g};
    for (std::size_t i = 1; i < nB; ++i)
    {
        const Link& L = links[i - 1];
        const std::size_t qi = static_cast<std::size_t>(L.qIndex);
        const std::size_t lam = static_cast<std::size_t>(L.lambda);
        aa[i] = motionAction(Xlambda[i], aa[lam]) + cc[i];
        qDdot[qi] = Dinv[i]*(uu[i].w.x - doubleInner(U1[i], aa[i]));
        aa[i] = aa[i] + S1[i]*qDdot[qi];
    }
    return qDdot;
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
        // joint::jcalc -- J.X, which does not read qDot
        SpatialTransform JX;
        SpatialVector JS1;
        SpatialVector Jv;
        jcalc(l.joint, q[static_cast<std::size_t>(l.qIndex)], scalar(0), JX, JS1, Jv);
        const SpatialTransform Xlambda = JX & l.XT;
        x0[i + 1] = (l.lambda != 0) ? (Xlambda & x0[static_cast<std::size_t>(l.lambda)]) : Xlambda;
    }
    return x0;
}


std::vector<SpatialVector> Model::applyRestraints(
    const std::vector<scalar>&        q,
    const std::vector<scalar>&        qDot,
    const std::vector<SpatialVector>& fx) const
{
    std::vector<SpatialVector> rfx(fx);
    if (restraints.empty())
    {
        return rfx;
    }
    const std::size_t nB = links.size() + 1;
    if (q.size() != static_cast<std::size_t>(nDoF()) || qDot.size() != q.size() || rfx.size() != nB)
    {
        throw std::runtime_error("brae RBD::applyRestraints: the joint state or fx is the wrong size.");
    }
    // forwardDynamicsCorrection.C's first pass (forwardDynamics.C:229-258), v and X0 only: neither
    // reads qDdot, and they are the numbers the restraints read out of the model's cache
    std::vector<SpatialTransform> X0v(nB);
    std::vector<SpatialVector> vv(nB);
    for (std::size_t i = 1; i < nB; ++i)
    {
        const Link& L = links[i - 1];
        const std::size_t qi = static_cast<std::size_t>(L.qIndex);
        SpatialTransform JX;
        SpatialVector JS1;
        SpatialVector Jv;
        jcalc(L.joint, q[qi], qDot[qi], JX, JS1, Jv);
        const SpatialTransform Xlambda = JX & L.XT;
        const std::size_t lam = static_cast<std::size_t>(L.lambda);
        X0v[i] = (lam != 0) ? (Xlambda & X0v[lam]) : Xlambda;
        vv[i] = motionAction(Xlambda, vv[lam]) + Jv;
    }
    for (const Restraint& r : restraints)
    {
        const std::size_t b = static_cast<std::size_t>(r.bodyID);
        SpatialVector f;
        if (r.type == RestraintType::linearDamper)
        {
            // linearDamper.C:73: force = -coeff_*v.l(); fx += X0.T() & spatialVector(Zero, force)
            const vector force = (-r.coeff)*vv[b].l;
            f = SpatialVector{vector{0, 0, 0}, force};
        }
        else
        {
            // sphericalAngularDamper.C:73: moment = -coeff_*v.w(); fx += X0.T() & (moment, Zero)
            const vector moment = (-r.coeff)*vv[b].w;
            f = SpatialVector{moment, vector{0, 0, 0}};
        }
        rfx[b] = rfx[b] + transposeAction(X0v[b], f);
    }
    return rfx;
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


namespace
{

// ISstream reads a number with readScalar (ISstream.C:782), which rounds |x| <= VSMALL to zero
// (Scalar.C:104-110, doubleScalarVSMALL 1e-300, doubleScalar.H:64)
scalar readStateNumber(TokenStream& ts)
{
    const scalar x = ts.nextScalar();
    constexpr scalar vSmall = 1.0e-300;
    if (x >= -vSmall && x <= vSmall)
    {
        return scalar(0);
    }
    return x;
}

}   // namespace


std::optional<std::vector<scalar>> readJointStateList(
    const std::string& path,
    const char* key)
{
    TokenStream ts(path);
    while (!ts.eof())
    {
        if (ts.next() != key)
        {
            continue;
        }
        const label n = ts.nextLabel();
        std::vector<scalar> v(static_cast<std::size_t>(n));
        // ListIO.C:245-285: `N ( ... )` or, the form UListIO.C:119-123 writes for more than one equal
        // entry, `N { v }` -- the state of a body at rest (RAS/floatingObject until t = 4 writes
        // `q 2 { 0 }`). Either delimiter holds no value when N is 0.
        if (ts.peek() == "{")
        {
            ts.expect("{");
            if (n > 0)
            {
                const scalar uniformValue = readStateNumber(ts);
                std::fill(v.begin(), v.end(), uniformValue);
            }
            ts.expect("}");
            return v;
        }
        ts.expect("(");
        for (scalar& x : v)
        {
            x = readStateNumber(ts);
        }
        ts.expect(")");
        return v;
    }
    return std::nullopt;
}


scalar readJointStateScalar(
    const std::string& path,
    const char* key,
    scalar fallback)
{
    TokenStream ts(path);
    while (!ts.eof())
    {
        if (ts.next() != key)
        {
            continue;
        }
        return readStateNumber(ts);
    }
    return fallback;
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
    MotionSpec spec;

    // THE TOP LEVEL, walked once for two facts. motionSolver::New reads the name with
    // getCompat<word>("motionSolver", {{"solver", -1612}}) (motionSolver.C:114): `motionSolver` when it
    // is there and the legacy `solver` only when it is not -- and in a file whose coefficients sit at
    // the top level `solver` is the rigid-body INTEGRATOR's sub-dictionary, never a name (DTCHullMoving
    // writes both; reading `solver {` as the name is how this reader first refused that case).
    // motionSolver.C:91: coeffDict() is optionalSubDict("rigidBodyMotionCoeffs").
    std::string motionSolver;
    std::string legacySolver;
    bool coeffsSubDict = false;
    {
        TokenStream top(dictPath, /*expandVars=*/true);
        while (!top.eof())
        {
            const std::string key = top.next();
            if (top.eof()) break;
            if (top.peek() == "{")
            {
                if (key == "rigidBodyMotionCoeffs") coeffsSubDict = true;
                label depth = 0;
                do
                {
                    const std::string t = top.next();
                    if (t == "{") ++depth;
                    else if (t == "}") --depth;
                } while (depth > 0 && !top.eof());
                continue;
            }
            // a leaf: its first token is the value, and a list or a nested block ends at its own `;`
            const std::string value = top.peek();
            label depth = 0;
            while (!top.eof())
            {
                const std::string t = top.next();
                if (t == "(" || t == "{") ++depth;
                else if (t == ")" || t == "}") --depth;
                else if (t == ";" && depth == 0) break;
            }
            if (key == "motionSolver") motionSolver = value;
            else if (key == "solver") legacySolver = value;
        }
    }
    if (motionSolver.empty())
    {
        motionSolver = legacySolver;
    }
    if (motionSolver != "rigidBodyMotion")
    {
        throw std::runtime_error(
            "brae RBD::readMotionSpec: " + dictPath + " asks for `motionSolver " + motionSolver +
            "`; this reader carries rigidBodyMotion only.");
    }

    TokenStream ts(dictPath, /*expandVars=*/true);

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

    // Into `rigidBodyMotionCoeffs {` when there is one. Otherwise the walk below reads the whole file
    // as the coefficients, which is optionalSubDict's other branch: the walk then ends at the end of
    // the file rather than at a closing brace, and the top-level entries that are not coefficients
    // (`dynamicFvMesh`, `motionSolverLibs`, `motionSolver`, FoamFile) fall to its skip-anything-else.
    if (coeffsSubDict)
    {
        while (!ts.eof())
        {
            const std::string key = ts.next();
            if (key == "rigidBodyMotionCoeffs" && ts.peek() == "{")
            {
                ts.expect("{");
                break;
            }
            if (ts.peek() == "{") { skipBlock(); continue; }
        }
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
    std::string bodyType;
    scalar bodyMass = 0;
    vector bodyCofM{0, 0, 0};
    vector bodyL{0, 0, 0};
    bool haveL = false;
    symmTensor bodyInertia{0, 0, 0, 0, 0, 0};
    bool haveInertia = false;
    // restraint name, type, body and coeff, in the dictionary's order; the body is resolved to its
    // index once the chain exists
    struct RestraintEntry
    {
        std::string name;
        std::string type;
        std::string body;
        scalar coeff = 0;
        bool haveCoeff = false;
    };
    std::vector<RestraintEntry> restraintEntries;

    // inside rigidBodyMotionCoeffs, or at the top of a file whose coefficients are there
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
        if (key == "type")
        {
            const std::string v = ts.next();
            // the BODY's type, which appears before its `joint` sub-dictionary; the joint's own
            // `type` entries are consumed inside the `joints` list above
            if (joints.empty() && bodyType.empty()) bodyType = v;
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "mass")         { bodyMass = ts.nextScalar(); if (ts.peek() == ";") ts.next(); continue; }
        if (key == "centreOfMass") { bodyCofM = readVectorParen(); if (ts.peek() == ";") ts.next(); continue; }
        if (key == "L")            { bodyL = readVectorParen(); haveL = true; if (ts.peek() == ";") ts.next(); continue; }
        if (key == "inertia")
        {
            // rigidBodyInertiaI.H:72 -- a symmTensor, written xx xy xz yy yz zz
            ts.expect("(");
            scalar e[6];
            for (scalar& c : e) c = ts.nextScalar();
            ts.expect(")");
            bodyInertia = symmTensor{e[0], e[1], e[2], e[3], e[4], e[5]};
            haveInertia = true;
            if (ts.peek() == ";") ts.next();
            continue;
        }
        if (key == "restraints")
        {
            // rigidBodyModel.C:84-116: every DICTIONARY entry is one restraint, in order
            ts.expect("{");
            while (!ts.eof() && ts.peek() != "}")
            {
                const std::string name = ts.next();
                if (ts.peek() != "{")
                {
                    // a leaf in `restraints` is not a restraint (dEntry.isDict()), and is skipped
                    while (!ts.eof() && ts.peek() != ";") ts.next();
                    if (ts.peek() == ";") ts.next();
                    continue;
                }
                ts.expect("{");
                RestraintEntry r;
                r.name = name;
                while (!ts.eof() && ts.peek() != "}")
                {
                    const std::string k = ts.next();
                    if (k == "type")       { r.type = ts.next(); }
                    else if (k == "body")  { r.body = ts.next(); }
                    else if (k == "coeff") { r.coeff = ts.nextScalar(); r.haveCoeff = true; }
                    else if (ts.peek() == "{") { skipBlock(); continue; }
                    else
                    {
                        while (!ts.eof() && ts.peek() != ";" && ts.peek() != "}") ts.next();
                    }
                    if (ts.peek() == ";") ts.next();
                }
                ts.expect("}");
                restraintEntries.push_back(r);
            }
            ts.expect("}");
            continue;
        }
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
                else if (type == "Pz") joints.push_back(JointType::Pz);
                else if (type == "Ry") joints.push_back(JointType::Ry);
                else
                {
                    throw std::runtime_error(
                        "brae RBD::readMotionSpec: the joint `" + type + "` is not ported. This port "
                        "carries Py, Pz and Ry, which is what RAS/floatingObject's and DTCHullMoving's "
                        "composites name; OpenFOAM has seventeen and each has its own jcalc.");
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
         || key == "nIter")
        {
            throw std::runtime_error(
                "brae RBD::readMotionSpec: `" + key + "` is set in " + dictPath + ", and this port "
                "does not carry it. `ramp` scales BOTH gravity and the fluid spatial force; "
                "`cOfGdisplacement` accumulates the body's travel into a registered field; `test` "
                "runs the dynamics with no fluid force at all; `nIter` iterates the force and the "
                "relaxation within one mesh update. Each changes the answer and none is ported.");
        }
        if (key == "q" || key == "qDot" || key == "qDdot" || key == "t" || key == "deltaT")
        {
            // read by the mesh motion, which alone knows whether a state file takes precedence
            spec.coeffStateKeys.push_back(key);
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

    if (bodyType != "cuboid" && bodyType != "rigidBody")
    {
        throw std::runtime_error(
            "brae RBD::readMotionSpec: body `" + spec.bodyName + "` is `" + bodyType + "`. Only "
            "`cuboid` and `rigidBody` are ported -- each rigidBody type builds its own inertia about "
            "the centre of mass (bodies/), and reading the wrong one gives a body of the right mass "
            "and the wrong resistance to rotation.");
    }
    if (bodyType == "cuboid" && (!haveL || bodyMass <= scalar(0)))
    {
        throw std::runtime_error(
            "brae RBD::readMotionSpec: body `" + spec.bodyName + "` gives no `L` or no positive "
            "`mass`; cuboid reads exactly L, mass and centreOfMass (cuboidI.H:65-77).");
    }
    if (bodyType == "rigidBody" && (!haveInertia || bodyMass <= scalar(0)))
    {
        throw std::runtime_error(
            "brae RBD::readMotionSpec: body `" + spec.bodyName + "` gives no `inertia` or no "
            "positive `mass`; rigidBody reads exactly mass, centreOfMass and inertia, the last ABOUT "
            "THE CENTRE OF MASS (rigidBodyInertiaI.H:68-73).");
    }
    spec.bodyType = bodyType;
    RigidBodyInertia bodyI;
    if (bodyType == "cuboid")
    {
        bodyI = cuboidInertia(bodyMass, bodyCofM, bodyL);
    }
    else
    {
        bodyI.m = bodyMass;
        bodyI.c = bodyCofM;
        bodyI.Ic = bodyInertia;
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
        // ...and the REAL body only on the last link: every one before it is a massless jointBody
        // (rigidBodyModel.C:279-290), which contributes a degree of freedom and no inertia.
        l.inertia = (j + 1 == joints.size()) ? bodyI : RigidBodyInertia();
    }

    for (const RestraintEntry& r : restraintEntries)
    {
        Restraint out;
        out.name = r.name;
        if (r.type == "linearDamper") out.type = RestraintType::linearDamper;
        else if (r.type == "sphericalAngularDamper") out.type = RestraintType::sphericalAngularDamper;
        else
        {
            throw std::runtime_error(
                "brae RBD::readMotionSpec: restraint `" + r.name + "` is `" + r.type + "`. Only "
                "linearDamper and sphericalAngularDamper are ported (DTCHullMoving's two); the "
                "springs, softWall, externalForce and prescribedRotation each add their own force "
                "inside the solver (restraints/).");
        }
        // rigidBodyRestraint.C:53: model.bodyID(body), which is -1 -- and a FatalError downstream --
        // for a name the model does not have
        if (r.body != spec.bodyName)
        {
            throw std::runtime_error(
                "brae RBD::readMotionSpec: restraint `" + r.name + "` names body `" + r.body + "`, "
                "and the model's one body is `" + spec.bodyName + "`.");
        }
        if (!r.haveCoeff)
        {
            throw std::runtime_error(
                "brae RBD::readMotionSpec: restraint `" + r.name + "` gives no `coeff`; " + r.type +
                ".C:102 reads it with readEntry.");
        }
        out.bodyID = spec.model.bodyID();
        out.coeff = r.coeff;
        spec.model.restraints.push_back(out);
    }
    return spec;
}

} // namespace RBD
} // namespace brae
