#pragma once
// fvc::reconstruct -- turning a face flux back into a cell vector.
//
// provenance:
//   openfoam:  src/finiteVolume/finiteVolume/fvc/fvcReconstruct.C:64-85
//              src/finiteVolume/finiteVolume/fvc/fvcSurfaceIntegrate.C:160-170 (surfaceSum)
//   cuda:      src/finiteVolume/finiteVolume/device_fvc_reconstruct.cu -- same gather addressing as
//              deviceDiv but with the SAME SIGN on the neighbour; the identity below is its gate too.
//
//     SfHat = Sf/magSf
//     reconstruct(ssf) = inv(surfaceSum(SfHat (x) Sf)) & surfaceSum(SfHat*ssf)
//
// interFoam needs it TWICE -- for the momentum predictor's source
//     fvc::reconstruct((surfaceTensionForce() - ghf*snGrad(rho) - snGrad(p_rgh))*magSf)   UEqn.H:19-27
// and again in pEqn.H to rebuild U after the pressure solve. It is the piece brae did not already have.
//
// surfaceSum ADDS TO BOTH SIDES OF A FACE WITH THE SAME SIGN:
//     vf[owner[f]]     += ssf[f];
//     vf[neighbour[f]] += ssf[f];      <- NOT -=
// which is the opposite of every divergence in this tree, where the neighbour subtracts. Reusing the
// div accumulation here is the obvious mistake and it does not look wrong: the tensor stays symmetric
// and invertible, the result is just a different vector. The gate's uniform-field arm is what catches
// it -- reconstruct(V & Sf) must return V exactly, and with a flipped sign it does not.
#include "cf_types.cuh"
#include <cmath>
#include <vector>

namespace brae {
namespace cpu {
namespace fvcReconstruct {

// OpenFOAM's Tensor::safeInv (TensorI.H:608-661), which is what inv(Field<tensor>) actually calls --
// tensorField.C:55 uses safeInv and NOT the plain Tensor::inv. The difference is the whole of 2-D:
//
//   threshold = SMALL*(magSqr(xx) + magSqr(yy) + magSqr(zz))
//   any diagonal below it gets +1 before the inversion and -1 after
//   a determinant below ROOTVSMALL returns the ZERO tensor rather than dividing
//
// ON A 2-D MESH THE TENSOR IS SINGULAR and the plain cofactor inverse gives NaN. surfaceSum skips
// empty patches -- emptyFvPatch::size() is 0 -- so sum(SfHat (x) Sf) has NO contribution in the empty
// direction and its zz is exactly zero. Measured: the device reconstruct, which correctly skips empty
// faces, produced NaN in ALL 2268 cells of damBreak within one step, and brae's correctVelocity only
// escaped it by NOT skipping them, which is a second defect masking the first.
inline tensor safeInv(const tensor& t)
{
    constexpr scalar kSmall = 1.0e-15;          // OF SMALL
    constexpr scalar kRootVSmall = 1.0e-150;    // OF ROOTVSMALL

    const scalar sxx = t.xx*t.xx, syy = t.yy*t.yy, szz = t.zz*t.zz;
    const scalar threshold = kSmall * (sxx + syy + szz);
    const bool smallXX = sxx < threshold, smallYY = syy < threshold, smallZZ = szz < threshold;

    tensor w = t;
    if (smallXX || smallYY || smallZZ)
    {
        if (smallXX) w.xx += scalar(1);
        if (smallYY) w.yy += scalar(1);
        if (smallZZ) w.zz += scalar(1);
    }

    const scalar det = w.xx*(w.yy*w.zz - w.yz*w.zy)
                     + w.xy*(w.yz*w.zx - w.yx*w.zz)
                     + w.xz*(w.yx*w.zy - w.yy*w.zx);
    if (std::fabs(det) < kRootVSmall) return tensor{0,0,0,0,0,0,0,0,0};

    const scalar s = scalar(1) / det;
    tensor r{
        (w.yy*w.zz - w.yz*w.zy)*s, (w.xz*w.zy - w.xy*w.zz)*s, (w.xy*w.yz - w.xz*w.yy)*s,
        (w.yz*w.zx - w.yx*w.zz)*s, (w.xx*w.zz - w.xz*w.zx)*s, (w.xz*w.yx - w.xx*w.yz)*s,
        (w.yx*w.zy - w.yy*w.zx)*s, (w.xy*w.zx - w.xx*w.zy)*s, (w.xx*w.yy - w.xy*w.yx)*s};
    if (smallXX) r.xx -= scalar(1);
    if (smallYY) r.yy -= scalar(1);
    if (smallZZ) r.zz -= scalar(1);
    return r;
}

// 3x3 inverse by cofactors, with no 2-D guard. Kept because the gate's identity arms run on 3-D cells
// where the two agree exactly; every SOLVER path goes through safeInv above, as OpenFOAM's does.
inline tensor inv(const tensor& t)
{
    const scalar c00 = t.yy*t.zz - t.yz*t.zy;
    const scalar c01 = t.yz*t.zx - t.yx*t.zz;
    const scalar c02 = t.yx*t.zy - t.yy*t.zx;
    const scalar d   = t.xx*c00 + t.xy*c01 + t.xz*c02;
    const scalar s   = scalar(1) / d;
    return tensor{
        c00*s,                        (t.xz*t.zy - t.xy*t.zz)*s, (t.xy*t.yz - t.xz*t.yy)*s,
        c01*s,                        (t.xx*t.zz - t.xz*t.zx)*s, (t.xz*t.yx - t.xx*t.yz)*s,
        c02*s,                        (t.xy*t.zx - t.xx*t.zy)*s, (t.xx*t.yy - t.xy*t.yx)*s};
}

inline vector dot(const tensor& t, const vector& v)
{
    return vector{t.xx*v.x + t.xy*v.y + t.xz*v.z,
                  t.yx*v.x + t.yy*v.y + t.yz*v.z,
                  t.zx*v.x + t.zy*v.y + t.zz*v.z};
}

// One face's contribution: SfHat (x) Sf into the tensor, SfHat*ssf into the vector.
inline void accumulate(const vector& Sf, scalar ssf, tensor& T, vector& v)
{
    const scalar magSf = std::sqrt(Sf.x*Sf.x + Sf.y*Sf.y + Sf.z*Sf.z);
    const vector h{Sf.x/magSf, Sf.y/magSf, Sf.z/magSf};       // SfHat
    T.xx += h.x*Sf.x; T.xy += h.x*Sf.y; T.xz += h.x*Sf.z;
    T.yx += h.y*Sf.x; T.yy += h.y*Sf.y; T.yz += h.y*Sf.z;
    T.zx += h.z*Sf.x; T.zy += h.z*Sf.y; T.zz += h.z*Sf.z;
    v.x += h.x*ssf;   v.y += h.y*ssf;   v.z += h.z*ssf;
}

// nCells, internal faces by owner/neighbour, then any boundary faces as (cell, Sf, ssf).
struct BoundaryFace { int cell; vector Sf; scalar ssf; };

inline void reconstruct(int nCells,
                        const std::vector<int>& owner,
                        const std::vector<int>& neighbour,
                        const std::vector<vector>& Sf,
                        const std::vector<scalar>& ssf,
                        const std::vector<BoundaryFace>& bnd,
                        std::vector<vector>& out)
{
    std::vector<tensor> T(static_cast<std::size_t>(nCells), tensor{0,0,0,0,0,0,0,0,0});
    std::vector<vector> v(static_cast<std::size_t>(nCells), vector{0,0,0});
    for (std::size_t f = 0; f < owner.size(); ++f)
    {
        // SAME SIGN on both sides -- fvcSurfaceIntegrate.C:165-166.
        accumulate(Sf[f], ssf[f], T[owner[f]],     v[owner[f]]);
        accumulate(Sf[f], ssf[f], T[neighbour[f]], v[neighbour[f]]);
    }
    for (const BoundaryFace& b : bnd) accumulate(b.Sf, b.ssf, T[b.cell], v[b.cell]);

    out.resize(static_cast<std::size_t>(nCells));
    for (int c = 0; c < nCells; ++c) out[c] = dot(safeInv(T[c]), v[c]);
}

}   // namespace fvcReconstruct
}   // namespace cpu
}   // namespace brae
