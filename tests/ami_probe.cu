// What brae actually built for a case's cyclicAMI patches: how many interfaces, how much of each source
// face the target side covers, and the full transform tensor.
//
// WHY THIS EXISTS. brae has nineteen AMI and cyclic gates and every one of them passes, yet the shipped
// path disagrees with OpenFOAM by 11% on U and 39% on p on pipeCyclic -- LAMINAR, so it is the interface
// coupling and not the turbulence. Those gates are unit-level: weights, geometry, host-vs-device. None
// of them runs a real AMI tutorial end to end against OpenFOAM, which is the same blind spot the
// rotorDiskSource gate had (it matched OpenFOAM's own reported force while the driver applied it with
// the wrong sign). This probe is the first step in closing that: it separates "the geometry brae built"
// from "the answer brae computes with it", so the two can be blamed independently.
//
// On pipeCyclic it reports the geometry as CORRECT: two interfaces, 400 source faces against 250 -- the
// mesh is genuinely non-conformal after refineHexMesh -- coverage exactly 1.000000 on every face in both
// directions, and the two rotational transforms exact transposes of each other (+90 and -90 about x).
// So the disagreement is downstream of the geometry.
//
// THE FULL TENSOR MATTERS: a 90-degree rotation has a zero diagonal in the rotated plane, so a diagonal
// print cannot tell +90 from -90 -- and for a swirling flow those are opposite azimuthal velocities at
// the interface.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ami_interface.cuh"
#include <cstdio>
#include <cmath>
using namespace brae;
int main(int argc, char** argv)
{
    PrimitiveMesh m; m.read(std::string(argv[1]) + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    for (const FvPatch& p : fvp)
        std::printf("  patch %-10s %-12s %5d faces\n", p.name.c_str(), p.type.c_str(), (int)p.size);
    const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);
    std::printf("  buildAMIInterfaces -> %d interface(s)\n", (int)amis.size());
    for (const AMIInterface& a : amis)
    {
        scalar wmin = 1e30, wmax = 0, wsum = 0;
        for (scalar w : a.weightsSum) { wmin = std::fmin(wmin, w); wmax = std::fmax(wmax, w); wsum += w; }
        std::printf("    patch %d -> %d : %d src faces, %d stencil entries, coverage min %.6f max %.6f mean %.6f\n",
                    (int)a.patch, (int)a.nbrPatch, (int)a.ownCell.size(), (int)a.nbrCell.size(),
                    wmin, wmax, a.weightsSum.empty() ? 0.0 : wsum / a.weightsSum.size());
        // The FULL tensor. A 90-degree rotation has a zero diagonal in the rotated plane, so printing
        // only the diagonal cannot tell +90 from -90 -- and for a swirling flow those are opposite
        // azimuthal velocities at the interface.
        std::printf("    transform: %s\n    forwardT = [%9.6f %9.6f %9.6f; %9.6f %9.6f %9.6f; %9.6f %9.6f %9.6f]\n",
                    a.translational ? "translational" : "rotational",
                    a.forwardT.xx, a.forwardT.xy, a.forwardT.xz,
                    a.forwardT.yx, a.forwardT.yy, a.forwardT.yz,
                    a.forwardT.zx, a.forwardT.zy, a.forwardT.zz);
        // deltaCoeffs against RADIUS. In a pipe SECTOR the two cyclicAMI patches meet along the axis,
        // so the faces nearest r = 0 are the ones whose owner and neighbour cells are closest together
        // and most nearly coincident -- exactly where 1/(n & delta) is most delicate. A conformal mesh
        // makes every one of these faces identical in area and interpolation weight, so any spread here
        // is geometric and worth seeing binned rather than as a min/max pair.
        {
            const FvPatch& pp = fvp[a.patch];
            std::printf("    deltaCoeffs by radius (r = sqrt(y^2+z^2) of the face centre):\n");
            const int NB = 6;
            scalar rmn = 1e300, rmx = -1e300;
            for (label i = 0; i < pp.size; ++i)
            {
                const scalar r = std::sqrt(pp.Cf[i].y*pp.Cf[i].y + pp.Cf[i].z*pp.Cf[i].z);
                rmn = std::fmin(rmn, r); rmx = std::fmax(rmx, r);
            }
            for (int b = 0; b < NB; ++b)
            {
                const scalar lo = rmn + (rmx-rmn)*b/NB, hi = rmn + (rmx-rmn)*(b+1)/NB;
                scalar dmn = 1e300, dmx = -1e300;
                int n = 0;
                for (label i = 0; i < pp.size && i < (label)a.deltaCoeffs.size(); ++i)
                {
                    const scalar r = std::sqrt(pp.Cf[i].y*pp.Cf[i].y + pp.Cf[i].z*pp.Cf[i].z);
                    if (r < lo || (b+1 < NB ? r >= hi : r > hi)) continue;
                    dmn = std::fmin(dmn, a.deltaCoeffs[i]); dmx = std::fmax(dmx, a.deltaCoeffs[i]);
                    ++n;
                }
                if (n) std::printf("      r %7.4f..%7.4f  %4d faces   deltaCoeffs %10.4f .. %10.4f\n",
                                   lo, hi, n, dmn, dmx);
            }
        }
        // A face centre and where the transform sends it: the check that the rotation goes the way the
        // geometry does, which the tensor alone does not show.
        if (!a.ownCell.empty())
        {
            const vector& c0 = a.Sf[0];
            std::printf("    src face 0: Sf (%9.6f %9.6f %9.6f)  |Sf| %.6g  deltaCoeff %.6g\n",
                        c0.x, c0.y, c0.z, a.magSf[0], a.deltaCoeffs[0]);
        }
    }
    return 0;
}
