#pragma once
// OpenFOAM's spatial vector algebra -- the six-component motion/force vectors, the 6x6 tensors, and
// the actions a spatialTransform has on each. The host reference.
//
// provenance:
//   openfoam: src/OpenFOAM/primitives/spatialVectorAlgebra/SpatialVector/SpatialVector/
//                 SpatialVector.H:74 (the component order), SpatialVectorI.H:47-80 (the
//                 constructors), :99-109 (w and l), :137-153 (the motion cross product),
//                 :157-175 (the dual/force cross product)
//             src/OpenFOAM/primitives/VectorSpace/VectorSpaceI.H:821-833 (operator&&, sequential)
//             src/OpenFOAM/primitives/spatialVectorAlgebra/SpatialTensor/SpatialTensor/
//                 SpatialTensorI.H:49-75 (the four-block constructor)
//             src/OpenFOAM/primitives/MatrixSpace/MatrixSpaceI.H:163-177 (T()), :214-231 (indexing),
//                 :448-472 (matrix & matrix), :478-497 (matrix & vector), :506-528 (the outer product)
//             src/OpenFOAM/primitives/spatialVectorAlgebra/SpatialTensor/spatialTransform/
//                 spatialTransformI.H:32-35 (Erx), :120-127 (the motion tensor), :146-156 (& on a
//                 motion vector), :181-188 and :191-203 (the transpose), :206-213 and :216-226 (the
//                 dual)
//   brae:
//     reference: this header
//     tests:     tests/test_rigid_body_dynamics_vs_openfoam.cu
//
// THE COMPONENT ORDER IS ANGULAR FIRST. `spatialVector(w, l)` -- and the fluid load the mesh motion
// applies is `spatialVector(momentEff, forceEff)`, so the moment occupies components 0..2. Reading it
// the other way round is silent: both halves are three-vectors of plausible magnitude.
//
// THE TERM ORDER IN THE CROSS PRODUCTS IS TRANSCRIBED, NOT SIMPLIFIED. OpenFOAM writes each line as
// `-a*b + c*d`, and the four-product lines of the dual cross in one fixed order. Reassociating them,
// or hoisting a common factor, changes which multiply the compiler fuses and moves the last ulp --
// see the note in tests/ about a tap that broke a gate by hoisting one product.
#include "cf_types.cuh"

namespace brae {
namespace RBD {

// OF's inner products for the full tensor, which cf_types.cuh carries only for symmTensor:
// (A & B)_ij = A_ik B_kj and (T & v)_i = T_ij v_j. Kept here rather than in the shared header
// because spatial algebra is their only caller in the tree.
inline tensor tdot(const tensor& a, const tensor& b)
{
    return tensor{
        a.xx*b.xx + a.xy*b.yx + a.xz*b.zx,  a.xx*b.xy + a.xy*b.yy + a.xz*b.zy,  a.xx*b.xz + a.xy*b.yz + a.xz*b.zz,
        a.yx*b.xx + a.yy*b.yx + a.yz*b.zx,  a.yx*b.xy + a.yy*b.yy + a.yz*b.zy,  a.yx*b.xz + a.yy*b.yz + a.yz*b.zz,
        a.zx*b.xx + a.zy*b.yx + a.zz*b.zx,  a.zx*b.xy + a.zy*b.yy + a.zz*b.zy,  a.zx*b.xz + a.zy*b.yz + a.zz*b.zz};
}

inline vector tdotv(const tensor& t, const vector& v)
{
    return vector{t.xx*v.x + t.xy*v.y + t.xz*v.z,
                  t.yx*v.x + t.yy*v.y + t.yz*v.z,
                  t.zx*v.x + t.zy*v.y + t.zz*v.z};
}

// *v, the Hodge dual: the skew tensor of a vector (TensorI.H:1058-1066)
inline tensor skew(const vector& v)
{
    return tensor{0, -v.z, v.y,  v.z, 0, -v.x,  -v.y, v.x, 0};
}


// SpatialVector: (angular, linear), components 0..2 then 3..5
struct SpatialVector
{
    vector w{0, 0, 0};
    vector l{0, 0, 0};

    scalar operator[](int i) const
    {
        switch (i)
        {
            case 0: return w.x;
            case 1: return w.y;
            case 2: return w.z;
            case 3: return l.x;
            case 4: return l.y;
            default: return l.z;
        }
    }
    scalar& at(int i)
    {
        switch (i)
        {
            case 0: return w.x;
            case 1: return w.y;
            case 2: return w.z;
            case 3: return l.x;
            case 4: return l.y;
            default: return l.z;
        }
    }
};

inline SpatialVector operator+(const SpatialVector& a, const SpatialVector& b)
{
    return SpatialVector{a.w + b.w, a.l + b.l};
}

inline SpatialVector operator-(const SpatialVector& a, const SpatialVector& b)
{
    return SpatialVector{a.w - b.w, a.l - b.l};
}

inline SpatialVector operator*(const SpatialVector& a, scalar s)
{
    return SpatialVector{s*a.w, s*a.l};
}

// VectorSpaceI.H:821-833 -- sequential, ascending
inline scalar doubleInner(const SpatialVector& a, const SpatialVector& b)
{
    scalar dd = a.w.x*b.w.x;
    dd += a.w.y*b.w.y;
    dd += a.w.z*b.w.z;
    dd += a.l.x*b.l.x;
    dd += a.l.y*b.l.y;
    dd += a.l.z*b.l.z;
    return dd;
}

// SpatialVectorI.H:137-153 -- the MOTION cross product, u ^ v
inline SpatialVector crossMotion(const SpatialVector& u, const SpatialVector& v)
{
    SpatialVector r;
    r.w.x = -u.w.z*v.w.y + u.w.y*v.w.z;
    r.w.y =  u.w.z*v.w.x - u.w.x*v.w.z;
    r.w.z = -u.w.y*v.w.x + u.w.x*v.w.y;
    r.l.x = -u.l.z*v.w.y + u.l.y*v.w.z - u.w.z*v.l.y + u.w.y*v.l.z;
    r.l.y =  u.l.z*v.w.x - u.l.x*v.w.z + u.w.z*v.l.x - u.w.x*v.l.z;
    r.l.z = -u.l.y*v.w.x + u.l.x*v.w.y - u.w.y*v.l.x + u.w.x*v.l.y;
    return r;
}

// SpatialVectorI.H:157-175 -- the DUAL (force) cross product, v ^* f
inline SpatialVector crossDual(const SpatialVector& v, const SpatialVector& f)
{
    SpatialVector r;
    r.w.x = -v.w.z*f.w.y + v.w.y*f.w.z - v.l.z*f.l.y + v.l.y*f.l.z;
    r.w.y =  v.w.z*f.w.x - v.w.x*f.w.z + v.l.z*f.l.x - v.l.x*f.l.z;
    r.w.z = -v.w.y*f.w.x + v.w.x*f.w.y - v.l.y*f.l.x + v.l.x*f.l.y;
    r.l.x = -v.w.z*f.l.y + v.w.y*f.l.z;
    r.l.y =  v.w.z*f.l.x - v.w.x*f.l.z;
    r.l.z = -v.w.y*f.l.x + v.w.x*f.l.y;
    return r;
}


// SpatialTensor: 6x6, row-major, m(i, j) = v[i*6 + j]
struct SpatialTensor
{
    scalar v[36] = {0};

    scalar  operator()(int i, int j) const { return v[i*6 + j]; }
    scalar& operator()(int i, int j)       { return v[i*6 + j]; }
};

// SpatialTensorI.H:49-75 -- the four 3x3 blocks, rows/cols 0-2 then 3-5
inline SpatialTensor blockTensor(
    const tensor& t00,
    const tensor& t01,
    const tensor& t10,
    const tensor& t11)
{
    const scalar a[4][9] = {
        {t00.xx, t00.xy, t00.xz, t00.yx, t00.yy, t00.yz, t00.zx, t00.zy, t00.zz},
        {t01.xx, t01.xy, t01.xz, t01.yx, t01.yy, t01.yz, t01.zx, t01.zy, t01.zz},
        {t10.xx, t10.xy, t10.xz, t10.yx, t10.yy, t10.yz, t10.zx, t10.zy, t10.zz},
        {t11.xx, t11.xy, t11.xz, t11.yx, t11.yy, t11.yz, t11.zx, t11.zy, t11.zz}};
    SpatialTensor s;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            s(i, j)         = a[0][i*3 + j];
            s(i, j + 3)     = a[1][i*3 + j];
            s(i + 3, j)     = a[2][i*3 + j];
            s(i + 3, j + 3) = a[3][i*3 + j];
        }
    }
    return s;
}

// MatrixSpaceI.H:448-472 -- result(i,j) += m1(i,k)*m2(k,j), k ascending, from Zero
inline SpatialTensor operator&(const SpatialTensor& a, const SpatialTensor& b)
{
    SpatialTensor r;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            for (int k = 0; k < 6; ++k)
            {
                r(i, j) += a(i, k)*b(k, j);
            }
        }
    }
    return r;
}

// MatrixSpaceI.H:478-497 -- result[i] += m(i,j)*v[j], j ascending, from Zero
inline SpatialVector operator&(const SpatialTensor& a, const SpatialVector& x)
{
    SpatialVector r;
    for (int i = 0; i < 6; ++i)
    {
        scalar s = 0;
        for (int j = 0; j < 6; ++j)
        {
            s += a(i, j)*x[j];
        }
        r.at(i) = s;
    }
    return r;
}

// MatrixSpaceI.H:506-528 -- the outer product, r(i,j) = u[i]*v[j]
inline SpatialTensor outerSpatial(const SpatialVector& u, const SpatialVector& x)
{
    SpatialTensor r;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            r(i, j) = u[i]*x[j];
        }
    }
    return r;
}

inline SpatialTensor operator-(const SpatialTensor& a, const SpatialTensor& b)
{
    SpatialTensor r;
    for (int i = 0; i < 36; ++i) r.v[i] = a.v[i] - b.v[i];
    return r;
}

inline SpatialTensor& operator+=(SpatialTensor& a, const SpatialTensor& b)
{
    for (int i = 0; i < 36; ++i) a.v[i] += b.v[i];
    return a;
}

} // namespace RBD
} // namespace brae
