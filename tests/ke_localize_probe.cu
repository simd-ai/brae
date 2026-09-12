// WHY DOES k NOT DECAY? The k transport equation assembled by the _cpp REFERENCE at OpenFOAM's own
// converged pipeCyclic state, reported term by term and per cell.
//
// The device path's per-cell k residual there is 164x OpenFOAM's own, with 30% of it in the INTERIOR --
// cells that touch no boundary at all. That part is exactly reproducible by the reference even though
// the reference has no cyclicAMI support, because an interior cell's residual r_c = b_c - (A.psi)_c
// involves only that cell and its internal-face neighbours. So the interior is where the two paths can
// be compared without the interface in the way, and where OpenFOAM's own residual (5.7e-05 over the
// whole field) sets the scale.
//
// It prints the k equation's two competing terms as well as the residual: production G = nut*GbyNu and
// destruction epsilon, because "k does not decay" is a statement about their balance, and a residual
// alone cannot say which side of it is wrong.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "k_epsilon.cuh"
#include "kepsilon_coeffs.cuh"
#include "fvc.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <time>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    scalar nu = 1e-6;
    {
        const FoamDict tp = readDict(caseDir + "/constant/transportProperties");
        nu = tp.scalarOr("nu", nu);
    }

    auto rd = [&](const std::string& f) {
        GeometricField<scalar> x =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + f), patches, nC);
        x.evaluateBoundary();
        return x;
    };
    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), patches, nC);
    U.evaluateBoundary();
    GeometricField<scalar> k = rd("k"), eps = rd("epsilon"), nut = rd("nut");

    KEpsilonCoeffs co;
    {
        std::string tp = caseDir + "/constant/momentumTransport";
        { std::ifstream probe(tp); if (!probe.good()) tp = caseDir + "/constant/turbulenceProperties"; }
        const FoamDict turbProps = readDict(tp);
        // RAS.kEpsilonCoeffs, exactly as turbulence_setup.cuh reads them; absent means OpenFOAM's defaults.
        if (const FoamDict* ras = turbProps.subDict("RAS"))
            if (const FoamDict* kec = ras->subDict("kEpsilonCoeffs"))
            {
                co.Cmu      = kec->scalarOr("Cmu",      co.Cmu);
                co.C1       = kec->scalarOr("C1",       co.C1);
                co.C2       = kec->scalarOr("C2",       co.C2);
                co.sigmaK   = kec->scalarOr("sigmak",   co.sigmaK);
                co.sigmaEps = kec->scalarOr("sigmaEps", co.sigmaEps);
            }
    }

    // Cells that touch NO boundary face: the ones the reference can judge without the interface.
    std::vector<char> boundaryTouching(nC, 0);
    for (const FvPatch& p : patches)
        for (label i = 0; i < p.size; ++i) boundaryTouching[p.faceCells[i]] = 1;
    label nInterior = 0;
    for (label c = 0; c < nC; ++c) if (!boundaryTouching[c]) ++nInterior;

    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    const std::vector<scalar> gByNu = kepsilon::GbyNu(gradU);
    const std::vector<scalar> G     = kepsilon::production(nut.internal, gByNu);

    std::printf("ke_localize_probe: %s/%s   %d cells, %d of them interior   nu %.6g\n",
                caseDir.c_str(), t.c_str(), (int)nC, (int)nInterior, nu);
    std::printf("  coefficients: Cmu %.4g C1 %.4g C2 %.4g sigmak %.4g sigmaEps %.4g\n",
                co.Cmu, co.C1, co.C2, co.sigmaK, co.sigmaEps);

    // Wall-adjacent cells, kept apart from the rest: the wall function writes G directly into the
    // near-wall cell (OF epsilonWallFunction/kqRWallFunction), so on a coarse high-Re mesh that single
    // cell layer can dominate the whole field's turbulence budget. An interior-only statistic cannot see it.
    std::vector<char> wallTouching(nC, 0);
    for (const FvPatch& p : patches)
    {
        if (p.type != "wall") continue;
        for (label i = 0; i < p.size; ++i) wallTouching[p.faceCells[i]] = 1;
    }
    label nWall = 0;
    for (label c = 0; c < nC; ++c) if (wallTouching[c]) ++nWall;

    auto wallStats = [&](const char* name, const std::vector<scalar>& v) {
        scalar mn = 1e300, mx = -1e300, sum = 0;
        label n = 0;
        for (label c = 0; c < nC; ++c)
        {
            if (!wallTouching[c]) continue;
            mn = std::fmin(mn, v[c]); mx = std::fmax(mx, v[c]); sum += v[c]; ++n;
        }
        std::printf("  %-26s [wall]     min %12.5e  max %12.5e  mean %12.5e\n",
                    name, mn, mx, n ? sum/n : 0.0);
    };

    auto stats = [&](const char* name, const std::vector<scalar>& v, bool interiorOnly) {
        scalar mn = 1e300, mx = -1e300, sum = 0;
        label n = 0;
        for (label c = 0; c < nC; ++c)
        {
            if (interiorOnly && boundaryTouching[c]) continue;
            mn = std::fmin(mn, v[c]); mx = std::fmax(mx, v[c]); sum += v[c]; ++n;
        }
        std::printf("  %-26s %s min %12.5e  max %12.5e  mean %12.5e\n",
                    name, interiorOnly ? "[interior]" : "[all]     ", mn, mx, n ? sum/n : 0.0);
    };
    stats("k",                    k.internal,   true);
    stats("epsilon",              eps.internal, true);
    stats("nut",                  nut.internal, true);
    stats("GbyNu = gradU&&dev2T", gByNu,        true);
    stats("G = nut*GbyNu",        G,            true);

    // The BALANCE the steady k equation has to strike in the interior: production against destruction.
    // OpenFOAM's converged k is what it is because these two nearly cancel; if brae's G is far from
    // its epsilon, its k cannot sit where OpenFOAM's does whatever the transport does.
    std::vector<scalar> ratio(nC, 0.0);
    for (label c = 0; c < nC; ++c) ratio[c] = G[c] / std::fmax(eps.internal[c], 1e-300);
    stats("G / epsilon",          ratio,        true);

    // Split the wall cells by whether they ALSO touch a coupled interface. brae's wall-cell k spans a
    // factor of 97 where OpenFOAM's spans 1.5, and its MINIMUM matches OpenFOAM -- so most wall cells are
    // right and a subset is not. The corner where a wall meets an AMI is the natural suspect: it is the
    // one place two boundary treatments have to agree about the same cell.
    std::vector<char> amiTouching(nC, 0);
    for (const FvPatch& p : patches)
    {
        if (p.type != "cyclicAMI") continue;
        for (label i = 0; i < p.size; ++i) amiTouching[p.faceCells[i]] = 1;
    }
    auto splitStats = [&](const char* name, const std::vector<scalar>& v, bool wantAMI) {
        scalar mn = 1e300, mx = -1e300, sum = 0;
        label n = 0;
        for (label c = 0; c < nC; ++c)
        {
            if (!wallTouching[c]) continue;
            if (static_cast<bool>(amiTouching[c]) != wantAMI) continue;
            mn = std::fmin(mn, v[c]); mx = std::fmax(mx, v[c]); sum += v[c]; ++n;
        }
        if (!n) return;
        std::printf("  %-18s [wall%s, %3d cells] min %12.5e  max %12.5e  mean %12.5e\n",
                    name, wantAMI ? " n AMI" : " only ", (int)n, mn, mx, sum/n);
    };
    for (int w = 0; w < 2; ++w)
    {
        splitStats("k",       k.internal,   w == 1);
        splitStats("epsilon", eps.internal, w == 1);
        splitStats("G",       G,            w == 1);
    }

    // WHERE along the pipe. brae's wall k spans a factor of 97 and its minimum matches OpenFOAM, so the
    // blow-up is positional; the axial coordinate is the first thing to bin it against, because the swirl
    // this case is driven by decays downstream and an entrance effect would show up as an x-dependence.
    {
        scalar xmn = 1e300, xmx = -1e300;
        for (label c = 0; c < nC; ++c)
            if (wallTouching[c]) { xmn = std::fmin(xmn, g.C()[c].x); xmx = std::fmax(xmx, g.C()[c].x); }
        std::printf("  wall-cell k binned by axial position (x in [%.3g, %.3g]):\n", xmn, xmx);
        const int NB = 8;
        for (int b = 0; b < NB; ++b)
        {
            const scalar lo = xmn + (xmx - xmn) * b / NB, hi = xmn + (xmx - xmn) * (b + 1) / NB;
            scalar mn = 1e300, mx = -1e300, sum = 0;
            label n = 0;
            for (label c = 0; c < nC; ++c)
            {
                if (!wallTouching[c]) continue;
                const scalar x = g.C()[c].x;
                if (x < lo || (b + 1 < NB ? x >= hi : x > hi)) continue;
                mn = std::fmin(mn, k.internal[c]); mx = std::fmax(mx, k.internal[c]);
                sum += k.internal[c]; ++n;
            }
            if (n) std::printf("    x %6.3f..%6.3f  %3d cells   k min %11.4e  max %11.4e  mean %11.4e\n",
                               lo, hi, (int)n, mn, mx, sum / n);
        }
    }

    std::printf("  --- the %d wall-adjacent cells ---\n", (int)nWall);
    wallStats("k",             k.internal);
    wallStats("epsilon",       eps.internal);
    wallStats("nut",           nut.internal);
    wallStats("G = nut*GbyNu", G);
    wallStats("G / epsilon",   ratio);

    // nut recomputed from OpenFOAM's own k and epsilon: Cmu*k^2/epsilon. If this disagrees with the nut
    // OpenFOAM wrote, brae is reading or forming the eddy viscosity differently, and G is wrong through it.
    const std::vector<scalar> nutRecomputed = kepsilon::nut(k.internal, eps.internal, co);
    scalar worst = 0;
    label worstCell = -1;
    for (label c = 0; c < nC; ++c)
    {
        if (boundaryTouching[c]) continue;
        const scalar r = std::fabs(nutRecomputed[c] - nut.internal[c])
                       / std::fmax(std::fabs(nut.internal[c]), 1e-300);
        if (r > worst) { worst = r; worstCell = c; }
    }
    std::printf("  nut vs Cmu*k^2/epsilon [interior]: worst rel %.3e at cell %d "
                "(file %.6e, recomputed %.6e)\n",
                worst, (int)worstCell,
                worstCell >= 0 ? nut.internal[worstCell] : 0.0,
                worstCell >= 0 ? nutRecomputed[worstCell] : 0.0);
    return 0;
}
