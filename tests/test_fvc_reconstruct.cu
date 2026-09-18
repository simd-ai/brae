// fvc::reconstruct, held against its own defining property.
//
// THE ORACLE IS AN IDENTITY, not a stored field: for any uniform vector V,
//
//     reconstruct(V & Sf) == V,   exactly.
//
// That is what reconstruct is FOR -- it inverts the projection a flux performs. It holds on any closed
// cell whatever its shape, so it needs no fixture and no tolerance beyond round-off, and it is sharp:
// the two ways to get reconstruct wrong both break it.
//
//   1. THE SIGN. surfaceSum adds to owner AND neighbour with the SAME sign
//      (fvcSurfaceIntegrate.C:165-166), which is the opposite of every divergence in this tree. Reusing
//      the div pattern leaves the tensor symmetric and invertible -- nothing looks broken, the answer is
//      just a different vector. Arm 3 below flips it deliberately and measures the damage.
//
//   2. THE NORMALISATION. inv(surfaceSum(SfHat (x) Sf)) is not inv(surfaceSum(Sf (x) Sf)); dropping the
//      hat changes the weighting by magSf per face, which is invisible on a cube where every face has
//      the same area and wrong on anything else. Arm 2 uses a cell with DELIBERATELY UNEQUAL face
//      areas so the difference has somewhere to show.
#include "fvc_reconstruct_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;
using namespace brae::cpu::fvcReconstruct;

namespace {
int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

scalar worstErr(const std::vector<vector>& got, const vector& want)
{
    scalar w = 0;
    for (const vector& g : got)
        w = std::fmax(w, std::fmax(std::fabs(g.x - want.x),
                      std::fmax(std::fabs(g.y - want.y), std::fabs(g.z - want.z))));
    return w;
}

// One cell, six faces, all of them boundary faces of that cell. Face areas are the caller's, so a
// cube and a stretched box can share this.
std::vector<BoundaryFace> oneCell(scalar ax, scalar ay, scalar az, const vector& V)
{
    const std::vector<vector> Sf{
        vector{ ax, 0, 0}, vector{-ax, 0, 0},
        vector{0,  ay, 0}, vector{0, -ay, 0},
        vector{0, 0,  az}, vector{0, 0, -az}};
    std::vector<BoundaryFace> b;
    for (const vector& s : Sf)
        b.push_back(BoundaryFace{0, s, s.x*V.x + s.y*V.y + s.z*V.z});   // ssf = V & Sf
    return b;
}
}   // namespace

int main()
{
    std::printf("== fvc::reconstruct ==\n");
    const vector V{scalar(2.5), scalar(-1.25), scalar(0.75)};

    // ---- 1. a unit cube: the easy case, and the one that hides both mistakes -----------------------
    {
        std::vector<vector> out;
        reconstruct(1, {}, {}, {}, {}, oneCell(1, 1, 1, V), out);
        const scalar e = worstErr(out, V);
        std::printf("  unit cube:            worst |reconstruct(V&Sf) - V| = %.3e\n", (double)e);
        check("reconstruct recovers V exactly on a cube", e <= scalar(1e-15));
    }

    // ---- 2. UNEQUAL face areas: where the SfHat normalisation matters ------------------------------
    // 10:2:1 area ratio. Without the hat the tensor is sum(Sf (x) Sf), which weights each direction by
    // magSf^2 instead of magSf -- still symmetric, still invertible, still wrong.
    {
        std::vector<vector> out;
        reconstruct(1, {}, {}, {}, {}, oneCell(scalar(10), scalar(2), scalar(1), V), out);
        const scalar e = worstErr(out, V);
        std::printf("  10:2:1 face areas:    worst |reconstruct(V&Sf) - V| = %.3e\n", (double)e);
        check("...and on a cell with 10:2:1 face areas", e <= scalar(1e-14));
    }

    // ---- 3. THE SIGN CONTROL -----------------------------------------------------------------------
    // Two cells sharing one internal face. Done correctly the identity still holds; with the neighbour
    // subtracting -- the div convention -- it does not. This is the arm that makes the file worth having.
    {
        // cell 0 and cell 1 side by side in x; the shared face is internal, the other ten are boundary.
        const std::vector<int> own{0}, nei{1};
        const std::vector<vector> SfInt{vector{1, 0, 0}};
        const std::vector<scalar> ssfInt{SfInt[0].x*V.x + SfInt[0].y*V.y + SfInt[0].z*V.z};
        std::vector<BoundaryFace> bnd;
        for (int c = 0; c < 2; ++c)
        {
            const std::vector<vector> Sf{
                vector{(c == 0 ? scalar(-1) : scalar(1)), 0, 0},     // the outer x face
                vector{0, 1, 0}, vector{0, -1, 0}, vector{0, 0, 1}, vector{0, 0, -1}};
            for (const vector& s : Sf)
                bnd.push_back(BoundaryFace{c, s, s.x*V.x + s.y*V.y + s.z*V.z});
        }
        std::vector<vector> out;
        reconstruct(2, own, nei, SfInt, ssfInt, bnd, out);
        const scalar e = worstErr(out, V);
        std::printf("  two cells, shared face: worst |reconstruct(V&Sf) - V| = %.3e\n", (double)e);
        check("the identity survives an internal face", e <= scalar(1e-14));

        // ...and now the div convention, computed here rather than in the implementation, so the file
        // shows what the wrong answer looks like instead of asserting that it is wrong.
        std::vector<tensor> T(2, tensor{0,0,0,0,0,0,0,0,0});
        std::vector<vector> v(2, vector{0,0,0});
        accumulate(SfInt[0],  ssfInt[0], T[0], v[0]);
        accumulate(SfInt[0], -ssfInt[0], T[1], v[1]);      // NEGATED, as a divergence would
        for (const BoundaryFace& b : bnd) accumulate(b.Sf, b.ssf, T[b.cell], v[b.cell]);
        std::vector<vector> wrong{dot(inv(T[0]), v[0]), dot(inv(T[1]), v[1])};
        const scalar ew = worstErr(wrong, V);
        std::printf("  ...with the neighbour NEGATED (the div convention): %.3e\n", (double)ew);
        check("the div convention breaks the identity, so this control discriminates", ew > scalar(1e-3));
    }

    // ---- 4. linearity, which the identity alone does not pin ---------------------------------------
    // reconstruct is linear in ssf, so scaling the flux scales the result. A formula that happened to
    // normalise by something flux-dependent would pass arms 1-3 and fail here.
    {
        const std::vector<BoundaryFace> b1 = oneCell(scalar(3), scalar(1), scalar(2), V);
        std::vector<BoundaryFace> b3 = b1;
        for (BoundaryFace& b : b3) b.ssf *= scalar(3);
        std::vector<vector> o1, o3;
        reconstruct(1, {}, {}, {}, {}, b1, o1);
        reconstruct(1, {}, {}, {}, {}, b3, o3);
        const bool lin = std::fabs(o3[0].x - scalar(3)*o1[0].x) <= scalar(1e-14)
                      && std::fabs(o3[0].y - scalar(3)*o1[0].y) <= scalar(1e-14)
                      && std::fabs(o3[0].z - scalar(3)*o1[0].z) <= scalar(1e-14);
        check("reconstruct is linear in the flux", lin);
    }

    std::printf("test_fvc_reconstruct: %d failures\n", failures);
    return failures ? 1 : 0;
}
