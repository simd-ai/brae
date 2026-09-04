#pragma once
// cf core integer/scalar types, mirrors OpenFOAM src/OpenFOAM/primitives.
//   label  = int32  (OpenFOAM default integer / addressing index)
//   scalar = double (fp64; matches OpenFOAM for the validation oracle)
// vector/tensor land with the geometry module (Phase 1, next increment).
#include <cstdint>
#include <cmath>

#define BRAE_HD __host__ __device__

namespace brae {

using label  = std::int32_t;
using scalar = double;

// 3-vector, mirrors OpenFOAM Vector<scalar>. Operators match OF: & = dot, ^ = cross.
struct vector
{
    scalar x, y, z;
};

BRAE_HD inline vector operator+(const vector& a, const vector& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
BRAE_HD inline vector operator-(const vector& a, const vector& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
BRAE_HD inline vector operator*(scalar s, const vector& a)        { return {s*a.x, s*a.y, s*a.z}; }
BRAE_HD inline vector operator*(const vector& a, scalar s)        { return {a.x*s, a.y*s, a.z*s}; }
BRAE_HD inline vector operator/(const vector& a, scalar s)        { return {a.x/s, a.y/s, a.z/s}; }
BRAE_HD inline vector& operator+=(vector& a, const vector& b)
{
    a.x+=b.x;
    a.y+=b.y;
    a.z+=b.z;
    return a;
}
BRAE_HD inline vector& operator-=(vector& a, const vector& b)
{
    a.x-=b.x;
    a.y-=b.y;
    a.z-=b.z;
    return a;
}

BRAE_HD inline scalar dot(const vector& a, const vector& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }   // OF: a & b
BRAE_HD inline vector cross(const vector& a, const vector& b)   // OF: a ^ b
{
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}
BRAE_HD inline scalar magSqr(const vector& a) { return dot(a, a); }
BRAE_HD inline scalar mag(const vector& a)    { return std::sqrt(magSqr(a)); }

// Symmetric 3x3 tensor, mirrors OpenFOAM SymmTensor<scalar> and its component ORDER, which is the
// order the dictionary uses: (xx xy xz yy yz zz). Six numbers, not nine -- a `0/sigma` written
// `uniform (0 0 0 0 0 0)` is exactly this. Used by the Maxwell viscoelastic stress.
struct symmTensor
{
    scalar xx, xy, xz, yy, yz, zz;
};

BRAE_HD inline symmTensor operator+(const symmTensor& a, const symmTensor& b)
{
    return {a.xx+b.xx, a.xy+b.xy, a.xz+b.xz, a.yy+b.yy, a.yz+b.yz, a.zz+b.zz};
}
BRAE_HD inline symmTensor operator-(const symmTensor& a, const symmTensor& b)
{
    return {a.xx-b.xx, a.xy-b.xy, a.xz-b.xz, a.yy-b.yy, a.yz-b.yz, a.zz-b.zz};
}
BRAE_HD inline symmTensor operator*(scalar s, const symmTensor& a)
{
    return {s*a.xx, s*a.xy, s*a.xz, s*a.yy, s*a.yz, s*a.zz};
}
BRAE_HD inline scalar magSqr(const symmTensor& a)   // OF magSqr: every off-diagonal counted TWICE
{
    return a.xx*a.xx + a.yy*a.yy + a.zz*a.zz + 2*(a.xy*a.xy + a.xz*a.xz + a.yz*a.yz);
}

// 3x3 tensor, mirrors OpenFOAM Tensor<scalar>. Used for grad(U) and turbulence production.
struct tensor
{
    scalar xx, xy, xz, yx, yy, yz, zx, zy, zz;
};

BRAE_HD inline tensor operator+(const tensor& a, const tensor& b)
{
    return {a.xx+b.xx, a.xy+b.xy, a.xz+b.xz, a.yx+b.yx, a.yy+b.yy, a.yz+b.yz, a.zx+b.zx, a.zy+b.zy, a.zz+b.zz};
}
BRAE_HD inline tensor operator/(const tensor& a, scalar s)
{
    return {a.xx/s, a.xy/s, a.xz/s, a.yx/s, a.yy/s, a.yz/s, a.zx/s, a.zy/s, a.zz/s};
}
BRAE_HD inline tensor operator-(const tensor& a, const tensor& b)
{
    return {a.xx-b.xx, a.xy-b.xy, a.xz-b.xz, a.yx-b.yx, a.yy-b.yy, a.yz-b.yz, a.zx-b.zx, a.zy-b.zy, a.zz-b.zz};
}
BRAE_HD inline tensor& operator+=(tensor& a, const tensor& b)
{
    a = a + b;
    return a;
}
BRAE_HD inline tensor outer(const vector& a, const vector& b)   // a (x) b : [i][j] = a_i b_j
{
    return {a.x*b.x, a.x*b.y, a.x*b.z, a.y*b.x, a.y*b.y, a.y*b.z, a.z*b.x, a.z*b.y, a.z*b.z};
}
// OF cmptMultiply: the componentwise product. A vector boundary coefficient is per COMPONENT (a mixed
// patch can fix one direction and leave another free -- directionMixed does exactly that), so combining
// a coefficient with a field is not a scalar multiply.
BRAE_HD inline scalar cmptMul(scalar a, scalar b) { return a * b; }
BRAE_HD inline vector cmptMul(const vector& a, const vector& b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
BRAE_HD inline scalar tr(const tensor& t)     { return t.xx + t.yy + t.zz; }
BRAE_HD inline tensor transpose(const tensor& t) { return {t.xx, t.yx, t.zx, t.xy, t.yy, t.zy, t.xz, t.yz, t.zz}; }
BRAE_HD inline scalar doubleDot(const tensor& a, const tensor& b)   // a && b
{
    return a.xx*b.xx + a.xy*b.xy + a.xz*b.xz + a.yx*b.yx + a.yy*b.yy + a.yz*b.yz + a.zx*b.zx + a.zy*b.zy + a.zz*b.zz;
}
BRAE_HD inline tensor operator*(scalar s, const tensor& t)
{
    return {s*t.xx, s*t.xy, s*t.xz, s*t.yx, s*t.yy, s*t.yz, s*t.zx, s*t.zy, s*t.zz};
}
// dev2(t) = t - 2*sph(t) = t - (2/3)*tr(t)*I  (OpenFOAM TensorI.H).
BRAE_HD inline tensor dev2(const tensor& t)
{
    const scalar c = (2.0/3.0) * tr(t);
    return {t.xx-c, t.xy, t.xz, t.yx, t.yy-c, t.yz, t.zx, t.zy, t.zz-c};
}
// v & T : (v & T)_j = sum_i v_i T_ij  (OpenFOAM Vector & Tensor).
BRAE_HD inline vector dot(const vector& v, const tensor& t)
{
    return {v.x*t.xx + v.y*t.yx + v.z*t.zx,
            v.x*t.xy + v.y*t.yy + v.z*t.zy,
            v.x*t.xz + v.y*t.yz + v.z*t.zz};
}

// A Type with every component set to s (OpenFOAM pTraits<Type>::one * s).
template <typename T> BRAE_HD inline T tUniform(scalar s);
template <> BRAE_HD inline scalar tUniform<scalar>(scalar s) { return s; }
template <> BRAE_HD inline vector tUniform<vector>(scalar s) { return {s, s, s}; }
template <> BRAE_HD inline tensor tUniform<tensor>(scalar s) { return {s, s, s, s, s, s, s, s, s}; }

// Component average (OpenFOAM cmptAv): scalar -> itself, vector -> (x+y+z)/3.
BRAE_HD inline scalar cmptAv(scalar s) { return s; }
BRAE_HD inline scalar cmptAv(const vector& v) { return (v.x + v.y + v.z) / 3.0; }

// Component magnitude-max / min (fvMatrix::relax boundary handling).
BRAE_HD inline scalar cmptMagMax(scalar s)        { return std::fabs(s); }
BRAE_HD inline scalar cmptMagMax(const vector& v) { return std::fmax(std::fmax(std::fabs(v.x), std::fabs(v.y)), std::fabs(v.z)); }
BRAE_HD inline scalar cmptMin(scalar s)           { return s; }
BRAE_HD inline scalar cmptMin(const vector& v)    { return std::fmin(std::fmin(v.x, v.y), v.z); }

// Component access (segregated vector solve / fvMatrix component extraction).
BRAE_HD inline scalar component(scalar s, int)        { return s; }
BRAE_HD inline scalar component(const vector& v, int c){ return c == 0 ? v.x : (c == 1 ? v.y : v.z); }
BRAE_HD inline void   setComponent(scalar& s, int, scalar val)       { s = val; }
BRAE_HD inline void   setComponent(vector& v, int c, scalar val)     { if (c == 0) v.x = val; else if (c == 1) v.y = val; else v.z = val; }

} // namespace brae
