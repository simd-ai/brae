// Curved cyclicAMI coverage -- the case every other AMI test in this suite is blind to.
//
// WHY THIS EXISTS. brae shipped an AMI that flattened BOTH patches onto the source patch's AVERAGE
// plane. That is exact for a planar interface and silently wrong for any other, and every AMI fixture
// in the suite was planar (cyclicChannelAMI, cyc_run_ami, tc_wedge_ami -- 8 to 10 faces, all flat), so
// nothing went red. On pimpleFoam/RAS/rotatingFanInRoom, whose interface is a cylinder, mean coverage
// was 0.4492 with 1481 faces at ZERO: 55% of the interface flux was being dropped. It did not fail as a
// wrong number, it failed as a divergence (contLocal 6.9e-03 -> 1.4e+75) many steps later.
//
// WHAT OF DOES -- faceAreaWeightAMI::calcInterArea (faceAreaWeightAMI.C:402-410):
//
//     vector n(-srcPatch.faceNormals()[srcFacei]);
//     if (reverseTarget_) n -= tgtPatch.faceNormals()[tgtFacei];
//     else                n += tgtPatch.faceNormals()[tgtFacei];
//     scalar magN = mag(n);
//     if (magN > ROOTVSMALL) inter.calc(src, tgt, n/magN, ...);
//
// The projection direction is chosen PER SOURCE/TARGET PAIR. A cylinder is locally planar for every
// pair even though it is globally not planar at all -- that is the whole content of the fix.
//
// THE FIXTURE is a closed polygonal TUBE (tube_ami_mesh.cuh): two concentric annuli joined only by a
// cyclicAMI pair, which is GLOBALLY NON-PLANAR to the maximum degree possible -- the normals sweep a
// full circle, so sum(Sf) cancels to round-off and |sum(Sf)|/sum(|Sf|) ~ 1e-17. The old code's average
// plane is not merely inaccurate there, it is undefined: it normalises a vector of length ~1e-16.
//
// The test runs the fixture in its TWO regimes, which need different kinds of reference:
//
//   COPLANAR (leg 2, Nin == Nout). Both sides are the same polygonal tube subdivided differently only
//   in z, so each source face is exactly coplanar with the target faces covering it. Coverage must be
//   1 TO ROUND-OFF -- an analytic reference with no tolerance to choose. Measured worst |w-1| ~ 1.3e-15.
//
//   NON-COPLANAR (leg 4, Nin != Nout). The two sides are different polygons inscribed in the same
//   cylinder -- the real-world case, a rotor and stator meshed independently. Their facets are not
//   coplanar, so coverage is NOT exactly 1 and no closed form exists. The reference is instead the
//   CONVERGENCE RATE: the residual is faceting error of size O(dtheta^2), so refining must drive it to
//   zero at second order. That is what separates faceting from a defect -- a wrong projection leaves an
//   O(1) error that does not converge at all. Measured orders are 2.01, 2.00, 2.00.
//
// WHY BRUTE FORCE OVER ALL PAIRS IS STILL CORRECT ON A CLOSED TUBE (this surprised me, so it is
// recorded). brae compares every src/tgt pair where OF walks an advancing front, and on a closed tube
// the far side of the pipe is a candidate. It cannot contribute, for two separate reasons:
//   * ANTIPODAL faces have nSrc = +r and nTgt = +r (the outer block's inner face points back at the
//     axis), so n = -nSrc + nTgt = 0 and OF's own ROOTVSMALL guard skips the pair. The guard is not a
//     safety net here, it is what makes the far wall unreachable.
//   * ANY OTHER pair projects along the BISECTOR of the two facet normals, which separates them: facet
//     s lands entirely on one side of the projection origin and facet s' entirely on the other, so the
//     overlap is zero by construction, not by tolerance.
// Both claims are load-bearing and both are checked: pinning the projection direction (the mutation
// this test was validated against) makes the far wall reachable and drives max coverage to exactly 2.
//
// NEGATIVE CONTROL. A test that only ever sees the fixed code cannot show that it would catch the
// defect. Leg 3 recomputes coverage with the OLD single-average-plane projection on this same mesh and
// REQUIRES it to fail. If someone reinstates that projection, legs 2 and 4 go red; if the fixture ever
// stops discriminating between the two, leg 3 goes red and says the test has become vacuous.
#include "tube_ami_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ami_interface.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

// worst |coverage - 1| over each interface of a freshly built fixture (the sum of overlap/|Sf|, before a
// cyclicAMI's rows are divided by their sum -- after it the sum of weights is 1 by construction).
struct Worst { scalar lo, hi, worst; };

std::vector<Worst> coverageOf(const PrimitiveMesh& m, const FvGeometry& g,
                              const std::vector<FvPatch>& fvp)
{
    std::vector<Worst> out;
    for (const AMIInterface& A : buildAMIInterfaces(m, g, fvp))
    {
        Worst w{1e300, -1e300, 0};
        for (const scalar v : A.coverage)
        {
            w.lo    = std::fmin(w.lo, v);
            w.hi    = std::fmax(w.hi, v);
            w.worst = std::fmax(w.worst, std::fabs(v - scalar(1)));
        }
        out.push_back(w);
    }
    return out;
}

}   // namespace

int main()
{
    const label  N = 16, nzA = 3, nzB = 5;
    const scalar Rin = 0.5, R = 1.0, Rout = 1.5, H = 1.0;

    PrimitiveMesh m = tubetest::tubeMesh(N, N, nzA, nzB, Rin, R, Rout, H);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);

    // -------------------------------------------------------------------------------------------
    // 0. The mesh is sane. A miswound face or a broken upper-triangular order would make every
    // number below meaningless, so check the volumes before trusting the coverage.
    {
        scalar vmin = 1e300, vsum = 0;
        for (label c = 0; c < m.nCells(); ++c) { vmin = std::fmin(vmin, g.V()[c]); vsum += g.V()[c]; }
        const scalar want = tubetest::tubeVolume(N, N, Rin, R, Rout, H);
        std::printf("  mesh: %d cells, min V = %.6e, sum V = %.9e (exact %.9e)\n",
                    (int)m.nCells(), vmin, vsum, want);
        if (vmin <= 0)
        { std::printf("  FAIL a cell has non-positive volume -- the fixture is miswound\n"); ++failures; }
        if (std::fabs(vsum - want)/want > 1e-12)
        { std::printf("  FAIL total volume is wrong -- the fixture is not the tube it claims\n"); ++failures; }
    }

    std::vector<label> amiPatch;
    for (std::size_t p = 0; p < fvp.size(); ++p)
        if (fvp[p].type == "cyclicAMI") amiPatch.push_back(static_cast<label>(p));
    if (amiPatch.size() != 2)
    {
        std::printf("  FAIL expected 2 cyclicAMI patches, found %zu\n", amiPatch.size());
        std::printf("test_ami_curved: %d failures\n", ++failures);
        return 1;
    }

    // -------------------------------------------------------------------------------------------
    // 1. ANTI-VACUOUS: the interface really is curved. If a later refactor made this fixture planar
    // it would pass under the OLD code too and would be testing nothing. Assert the hostility.
    for (int a = 0; a < 2; ++a)
    {
        const FvPatch& P = fvp[amiPatch[a]];
        vector sum{0,0,0};
        scalar mags = 0;
        for (label i = 0; i < P.size; ++i)
        { sum += g.Sf()[P.start+i]; mags += g.magSf()[P.start+i]; }
        const scalar planarity = mag(sum)/mags;      // 1 = planar, 0 = normals sweep a full circle
        std::printf("  [%d] %-8s nFaces=%3d  planarity |sum Sf|/sum|Sf| = %.3e\n",
                    a, P.name.c_str(), (int)P.size, planarity);
        if (planarity > 0.05)
        {
            std::printf("  FAIL vacuous: this interface is planar, so the average-plane projection\n"
                        "       would pass too and the test cannot detect the defect it exists for\n");
            ++failures;
        }
    }

    // -------------------------------------------------------------------------------------------
    // 2. COPLANAR SIDES -> AN EXACT REFERENCE. Both sides are the same polygonal tube subdivided
    // differently only in z, so every source face is covered exactly: coverage is 1 to round-off. A
    // face short of 1 loses that fraction of its flux (the rotatingFanInRoom divergence); a face above
    // 1 has reached a target it must not see.
    {
        const std::vector<Worst> cov = coverageOf(m, g, fvp);
        for (std::size_t a = 0; a < cov.size(); ++a)
        {
            std::printf("  [%zu] coplanar sides: coverage min %.15f  max %.15f  worst |w-1| %.3e\n",
                        a, cov[a].lo, cov[a].hi, cov[a].worst);
            if (cov[a].worst > 1e-9)
            {
                std::printf("  FAIL coverage is not 1 on a coplanar-by-construction interface: the AMI\n"
                            "       is %s. OF projects PER PAIR (faceAreaWeightAMI.C:402-410);\n"
                            "       flattening the whole patch onto one average plane loses %.1f%% of it.\n",
                            cov[a].lo < 1 ? "dropping overlap" : "over-counting", 100.0*(1.0 - cov[a].lo));
                ++failures;
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // 3. NEGATIVE CONTROL: the old single-average-plane projection, run on this same mesh, must FAIL.
    // Without this the suite cannot show the fixture discriminates -- only that the current code
    // agrees with itself.
    {
        using namespace brae::ami_detail;
        const FvPatch& S = fvp[amiPatch[0]];
        const FvPatch& T = fvp[amiPatch[1]];

        vector n{0,0,0};
        for (label i = 0; i < S.size; ++i) n += g.Sf()[S.start+i]/g.magSf()[S.start+i];
        n = n/std::fmax(mag(n), scalar(1e-300));           // on a tube this normalises round-off
        vector e1 = std::fabs(n.x) < 0.9 ? cross(n, vector{1,0,0}) : cross(n, vector{0,1,0});
        e1 = e1/std::fmax(mag(e1), scalar(1e-300));
        const vector e2 = cross(n, e1);
        const vector orig = g.Cf()[S.start];
        auto flatten = [&](label f)
        {
            std::vector<vec2> out;
            for (label k = 0; k < m.faceSize(f); ++k)
            {
                const vector d = m.points()[m.faceVert(f,k)] - orig;
                out.push_back({dot(d,e1), dot(d,e2)});
            }
            orientCCW(out);
            return out;
        };
        std::vector<std::vector<vec2>> tgt(T.size);
        for (label j = 0; j < T.size; ++j) tgt[j] = flatten(T.start+j);

        scalar sum = 0, lo = 1e300;
        for (label i = 0; i < S.size; ++i)
        {
            const std::vector<vec2> src = flatten(S.start+i);
            scalar cov = 0;
            for (label j = 0; j < T.size; ++j) cov += overlapArea(src, tgt[j]);
            cov /= g.magSf()[S.start+i];
            sum += cov; lo = std::fmin(lo, cov);
        }
        const scalar mean = sum/scalar(S.size);
        std::printf("  control: the OLD average-plane projection gives mean coverage %.4f (min %.4f)\n",
                    mean, lo);
        if (std::fabs(mean - scalar(1)) < 0.1)
        {
            std::printf("  FAIL negative control: the average-plane projection ALSO gets ~1 on this\n"
                        "       fixture, so leg 2 passing proves nothing. The fixture has stopped\n"
                        "       discriminating and must be made curved again before it is trusted.\n");
            ++failures;
        }
    }

    // -------------------------------------------------------------------------------------------
    // 4. NON-COPLANAR SIDES -- the real rotor/stator case, and the one leg 2 cannot reach.
    //
    // Give the two sides DIFFERENT circumferential resolutions (3:4). They are now different polygons
    // inscribed in the same cylinder, so their facets are not coplanar, a source face is not exactly
    // tiled by its targets, and coverage is NOT exactly 1. There is no closed form to compare against.
    //
    // The reference is the RATE. The mismatch between two inscribed polygons is a radial deviation of
    // O(R*dtheta^2), so the coverage residual must fall as O(dtheta^2) -- second order -- and vanish
    // under refinement. A projection that is merely approximate leaves an error that does not converge,
    // which is what makes this a test rather than a measurement: asserting a tolerance on one mesh
    // would pass for a wrong algorithm tuned to that mesh; asserting the ORDER cannot.
    {
        const label lvl[4][2] = {{12,16}, {24,32}, {48,64}, {96,128}};
        scalar err[4][2] = {};
        std::size_t nAmi = 0;

        for (int L = 0; L < 4; ++L)
        {
            PrimitiveMesh mm = tubetest::tubeMesh(lvl[L][0], lvl[L][1], nzA, nzB, Rin, R, Rout, H);
            FvGeometry gg;
            gg.build(mm);
            const std::vector<FvPatch> pp = buildPatches(mm, gg);
            const std::vector<Worst> cov = coverageOf(mm, gg, pp);
            nAmi = cov.size();
            std::printf("  [nc] Nin=%3d Nout=%3d :", (int)lvl[L][0], (int)lvl[L][1]);
            for (std::size_t a = 0; a < cov.size() && a < 2; ++a)
            {
                err[L][a] = cov[a].worst;
                std::printf("  [%zu] min %.6f max %.6f worst %.3e", a, cov[a].lo, cov[a].hi, cov[a].worst);
            }
            std::printf("\n");
        }

        if (nAmi != 2)
        { std::printf("  FAIL non-conforming fixture built %zu interfaces, expected 2\n", nAmi); ++failures; }

        for (std::size_t a = 0; a < 2; ++a)
        {
            // ANTI-VACUOUS. The coarse error must be a real faceting mismatch. If the two sides ever
            // became coplanar again this collapses to ~1e-15, the ratios below become noise/noise, and
            // a meaningless "order" would be reported as a pass.
            if (err[0][a] < 1e-4)
            {
                std::printf("  FAIL vacuous [%zu]: coarse-level error %.3e is at round-off, so the two\n"
                            "       sides are effectively coplanar and this leg re-tests leg 2\n",
                            a, err[0][a]);
                ++failures;
                continue;
            }
            for (int L = 0; L + 1 < 4; ++L)
            {
                const scalar ratio = err[L][a]/std::fmax(err[L+1][a], scalar(1e-300));
                const scalar order = std::log2(ratio);
                std::printf("  [%zu] refine %3d->%3d : %.3e -> %.3e  ratio %.2f  order %.2f\n",
                            a, (int)lvl[L][0], (int)lvl[L+1][0], err[L][a], err[L+1][a], ratio, order);
                // Second order is 4.0. The floor is 1.8 -- clear of the measured 2.00 and far above
                // anything a non-converging projection could reach.
                if (order < 1.8)
                {
                    std::printf("  FAIL coverage on a non-coplanar curved interface is not converging at\n"
                                "       second order (%.2f). The residual is then NOT faceting error --\n"
                                "       a projection that stays wrong under refinement is a defect.\n", order);
                    ++failures;
                }
            }
            if (err[3][a] > 2e-4)
            {
                std::printf("  FAIL finest-level coverage error %.3e is too large to be faceting alone\n",
                            err[3][a]);
                ++failures;
            }
        }
    }

    std::printf("test_ami_curved: %d failures\n", failures);
    return failures ? 1 : 0;
}
