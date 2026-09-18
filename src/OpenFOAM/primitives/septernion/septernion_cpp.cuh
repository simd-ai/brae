#pragma once
// OpenFOAM's quaternion and septernion, as far as a solid-body mesh motion uses them: build one from
// an axis and an angle or from Euler angles, compose, and transform a field of points.
//
// provenance:
//   openfoam:  src/OpenFOAM/primitives/quaternion/quaternionI.H:42-45 (axis, angle), :86-186 (Euler
//                  orders), :281-303 (mulq0v, transform, invTransform), :306-317 (the normalising
//                  quaternion-on-quaternion transform), :320-340 (R), :588-593 (operator*=),
//                  :690-706 (conjugate, normalised), :741-753 (operator*)
//              src/OpenFOAM/primitives/septernion/septernionI.H:52-62, :99-107, :213-247, :266-278
//              src/OpenFOAM/fields/Fields/transformField/transformField.C:34-45, :74-99
//   tests:     tests/test_mesh_motion_vs_openfoam.cu holds the POINTS these produce against the
//              polyMesh/points OpenFOAM writes, per time step
//
// WHY A TYPE AND NOT A ROTATION FORMULA. src/OpenFOAM/motion/solid_body_motion.cuh rotates with
// Rodrigues' formula, "the same rotation the quaternion encodes, written without a quaternion type".
// It is the same rotation and not the same arithmetic, and a multiMotion is a PRODUCT of septernions
// whose translations are carried through each other's inverse rotations -- there is no writing that
// without the type. The points are compared to OpenFOAM's to the last digit, so the operations are
// OpenFOAM's, in OpenFOAM's order.
//
// THREE THINGS THAT ARE NOT WHAT THEY LOOK LIKE:
//
//   A SEPTERNION'S TRANSLATION IS SUBTRACTED. transformPoint(v) is r.transform(v - t), which is why
//   every motion function writes septernion(-displacement) for a body that moves BY +displacement.
//
//   septernion*septernion NORMALISES the product rotation (quaternion::transform(quaternion) is
//   normalised((*this)*q)); quaternion*quaternion does not.
//
//   A FIELD OF POINTS IS NOT TRANSFORMED THROUGH THE QUATERNION. transformPoints subtracts the
//   translation only when its magnitude exceeds VSMALL, and rotates -- only when mag(R - I) exceeds
//   SMALL -- by the rotation TENSOR q.R(), one tensor-vector product per point. quaternion::transform
//   of a single vector is a different sequence of operations and is used for translations only.
#include "cf_types.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace brae {

struct Quaternion
{
    scalar w = 1;
    vector v{0, 0, 0};

    enum class EulerOrder
    {
        XYZ
    };

    Quaternion() = default;
    Quaternion(
        scalar wIn,
        const vector& vIn)
    :
        w(wIn),
        v(vIn)
    {}
    // quaternion(const scalar w): a real quaternion; quaternion(1) is the identity
    explicit Quaternion(scalar wIn)
    :
        w(wIn),
        v{0, 0, 0}
    {}
    // rotation by theta about d: (cos(theta/2), sin(theta/2)*normalised(d))
    Quaternion(
        const vector& d,
        scalar theta)
    {
        // ROOTVSMALL
        const scalar s = mag(d);
        const vector n = (s < scalar(1e-150)) ? vector{0, 0, 0} : d/s;
        w = std::cos(0.5*theta);
        v = std::sin(0.5*theta)*n;
    }
    // quaternion(eulerOrder, angles). Only XYZ is built here because only XYZ is asked for by the
    // motion functions (oscillatingRotatingMotion, SDA, tabulated6DoFMotion).
    Quaternion(
        EulerOrder,
        const vector& angles)
    {
        *this = Quaternion(vector{1, 0, 0}, angles.x);
        *this *= Quaternion(vector{0, 1, 0}, angles.y);
        *this *= Quaternion(vector{0, 0, 1}, angles.z);
    }

    Quaternion& operator*=(const Quaternion& q)
    {
        const scalar w0 = w;
        w = w*q.w - dot(v, q.v);
        v = w0*q.v + q.w*v + cross(v, q.v);
        return *this;
    }

    Quaternion mulq0v(const vector& u) const
    {
        return Quaternion(-dot(v, u), w*u + cross(v, u));
    }
    vector transform(const vector& u) const;
    vector invTransform(const vector& u) const;
    // the rotation tensor, row-major: xx xy xz yx yy yz zx zy zz
    tensor R() const
    {
        const scalar w2 = w*w;
        const scalar x2 = v.x*v.x;
        const scalar y2 = v.y*v.y;
        const scalar z2 = v.z*v.z;
        const scalar txy = 2*v.x*v.y;
        const scalar twz = 2*w*v.z;
        const scalar txz = 2*v.x*v.z;
        const scalar twy = 2*w*v.y;
        const scalar tyz = 2*v.y*v.z;
        const scalar twx = 2*w*v.x;
        tensor r;
        r.xx = w2 + x2 - y2 - z2;
        r.xy = txy - twz;
        r.xz = txz + twy;
        r.yx = txy + twz;
        r.yy = w2 - x2 + y2 - z2;
        r.yz = tyz - twx;
        r.zx = txz - twy;
        r.zy = tyz + twx;
        r.zz = w2 - x2 - y2 + z2;
        return r;
    }
};

inline Quaternion operator*(
    const Quaternion& q1,
    const Quaternion& q2)
{
    return Quaternion(
        q1.w*q2.w - dot(q1.v, q2.v),
        q1.w*q2.v + q2.w*q1.v + cross(q1.v, q2.v));
}

inline Quaternion conjugate(const Quaternion& q)
{
    return Quaternion(q.w, scalar(-1)*q.v);
}

inline Quaternion normalised(const Quaternion& q)
{
    const scalar s = std::sqrt(q.w*q.w + magSqr(q.v));
    // ROOTVSMALL
    if (s < scalar(1e-150))
    {
        return Quaternion(scalar(0), vector{0, 0, 0});
    }
    return Quaternion(q.w/s, q.v/s);
}

inline vector Quaternion::transform(const vector& u) const
{
    return (mulq0v(u)*conjugate(*this)).v;
}

inline vector Quaternion::invTransform(const vector& u) const
{
    return (conjugate(*this).mulq0v(u)*(*this)).v;
}

struct Septernion
{
    vector t{0, 0, 0};
    Quaternion r;

    Septernion() = default;
    Septernion(
        const vector& tIn,
        const Quaternion& rIn)
    :
        t(tIn),
        r(rIn)
    {}
    explicit Septernion(const vector& tIn)
    :
        t(tIn),
        r()
    {}
    explicit Septernion(const Quaternion& rIn)
    :
        t{0, 0, 0},
        r(rIn)
    {}

    Septernion& operator*=(const Septernion& tr)
    {
        t = tr.t + tr.r.invTransform(t);
        r *= tr.r;
        return *this;
    }
};

inline Septernion operator*(
    const Septernion& tr,
    const Quaternion& r)
{
    return Septernion(r.invTransform(tr.t), tr.r*r);
}

inline Septernion operator*(
    const Septernion& tr1,
    const Septernion& tr2)
{
    return Septernion(
        tr2.r.invTransform(tr1.t) + tr2.t,
        normalised(tr1.r*tr2.r));
}

// transformPoints(const septernion&, const vectorField&)
inline std::vector<vector> transformPoints(
    const Septernion& tr,
    const std::vector<vector>& fld)
{
    std::vector<vector> result(fld.size());
    // VSMALL
    if (mag(tr.t) > scalar(1e-300))
    {
        for (std::size_t i = 0; i < fld.size(); ++i)
        {
            result[i] = fld[i] - tr.t;
        }
    }
    else
    {
        result = fld;
    }

    const tensor rot = tr.r.R();
    const scalar d[9] = {rot.xx - 1, rot.xy, rot.xz, rot.yx, rot.yy - 1, rot.yz, rot.zx, rot.zy, rot.zz - 1};
    scalar magSqrDiff = 0;
    for (const scalar c : d)
    {
        magSqrDiff += c*c;
    }
    // SMALL
    if (std::sqrt(magSqrDiff) > scalar(1e-15))
    {
        for (vector& p : result)
        {
            p = vector{
                rot.xx*p.x + rot.xy*p.y + rot.xz*p.z,
                rot.yx*p.x + rot.yy*p.y + rot.yz*p.z,
                rot.zx*p.x + rot.zy*p.y + rot.zz*p.z};
        }
    }
    return result;
}

} // namespace brae
