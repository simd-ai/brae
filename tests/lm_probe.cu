// Where does the _cpp kOmegaSST(LM) disagree with OpenFOAM? One correct() from OpenFOAM's OWN converged
// state, then the per-cell difference reported BY REGION -- near-wall vs interior, and the worst cells
// named with their wall distance. A single L2 number over 26820 cells cannot say whether a 1% omega
// difference is spread everywhere or sitting on one cell layer, and those have different causes.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "cell_wall_dist.cuh"
#include "kOmegaSST_cpp.cuh"
#include "kOmegaSSTLM_cpp.cuh"
#include "komega_sst_coeffs.cuh"
#include "fvc.cuh"
#include <fstream>

#include <algorithm>
#include <cmath>
#include <cstdio>
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
    const scalar nu = 1.5e-5;

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    auto rd = [&](const std::string& f) {
        GeometricField<scalar> x =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + f), patches, nC);
        x.evaluateBoundary();
        return x;
    };
    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), patches, nC);
    U.evaluateBoundary();
    GeometricField<scalar> k = rd("k"), omega = rd("omega"), nut = rd("nut");
    const std::vector<scalar> k0 = k.internal, om0 = omega.internal, nut0 = nut.internal;

    // phi from the case's own written flux where present, else from U.
    // OpenFOAM's OWN phi where it wrote one. fvc::flux(U) is NOT the same field: OF's phi comes out of
    // the pressure equation and is conservative to machine precision, so div(phi) is zero and the
    // `bounded` Sp term vanishes. Rebuilding it from U leaves a nonzero div(phi) that perturbs both
    // transport equations -- which would show up here as a model disagreement that is not one.
    SurfaceScalarField phi;
    const std::string phiPath = caseDir + "/" + t + "/phi";
    if (std::ifstream(phiPath).good() || std::ifstream(phiPath + ".gz").good())
    {
        const FieldData<scalar> pf = readField<scalar>(phiPath);
        phi.internal = pf.internalField;
        phi.boundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            phi.boundary[pi].assign(patches[pi].size, 0.0);
            for (const auto& b : pf.boundary)
                if (b.name == patches[pi].name && b.hasValue
                    && static_cast<label>(b.values.size()) == patches[pi].size)
                    phi.boundary[pi] = b.values;
        }
        std::printf("  phi: read from %s\n", phiPath.c_str());
    }
    else
    {
        phi = fvc::flux(U, m, g, patches);
        std::printf("  phi: rebuilt from U (the case wrote none)\n");
    }

    const std::vector<scalar> y = cellWallDist(m, g, patches);

    KOmegaSSTCoeffs co;
    {
        std::string tp = caseDir + "/constant/momentumTransport";
        { std::ifstream probe(tp); if (!probe.good()) tp = caseDir + "/constant/turbulenceProperties"; }
        const FoamDict turbProps = readDict(tp);
        if (const FoamDict* ras = turbProps.subDict("RAS")) readKOmegaSSTCoeffs(ras, co);
    }

    cpu::kOmegaSST::SSTResiduals res;
    cpu::kOmegaSST::correct(U, k, omega, nut, phi, y, nu, m, g, patches,
                            0.9, 0.9, 1e-10, 0.0, 2000, co, &res,
                            /*bounded*/true, /*limitedLinear*/false, 1.0, /*linearUpwind*/true,
                            /*correctedLaplacian*/true, /*snGradLimitCoeff*/0.0, nullptr);

    std::printf("lm_probe: %s/%s  %d cells\n", caseDir.c_str(), t.c_str(), (int)nC);
    for (const FvPatch& p : patches)
    {
        if (p.size == 0) { std::printf("  patch %-14s %-10s empty\n", p.name.c_str(), p.type.c_str()); continue; }
        vector lo = p.Cf[0], hi = p.Cf[0], nAv{0, 0, 0};
        for (label i = 0; i < p.size; ++i)
        {
            lo.x = std::fmin(lo.x, p.Cf[i].x); hi.x = std::fmax(hi.x, p.Cf[i].x);
            lo.y = std::fmin(lo.y, p.Cf[i].y); hi.y = std::fmax(hi.y, p.Cf[i].y);
            nAv = nAv + p.nf[i];
        }
        nAv = nAv / static_cast<scalar>(p.size);
        std::printf("  patch %-14s %-10s %5d faces   x [%.4e %.4e]  y [%.4e %.4e]  mean n (%.3f %.3f %.3f)\n",
                    p.name.c_str(), p.type.c_str(), (int)p.size, lo.x, hi.x, lo.y, hi.y, nAv.x, nAv.y, nAv.z);
    }
    std::printf("  initial residuals from OpenFOAM's own state:  omega %.4e   k %.4e\n",
                res.omega, res.k);

    // The near-wall cell layer: every cell that owns a face on a `wall` patch.
    std::vector<char> nearWall(nC, 0);
    for (const FvPatch& p : patches)
    {
        if (p.type != "wall") continue;
        for (label i = 0; i < p.size; ++i) nearWall[p.faceCells[i]] = 1;
    }

    struct Acc { scalar num = 0, den = 0; label n = 0; };
    auto add = [](Acc& a, scalar was, scalar now) {
        a.num += (now - was) * (now - was);
        a.den += was * was;
        ++a.n;
    };
    for (int f = 0; f < 3; ++f)
    {
        const std::vector<scalar>& was = (f == 0) ? om0 : (f == 1) ? k0 : nut0;
        const std::vector<scalar>& now = (f == 0) ? omega.internal : (f == 1) ? k.internal : nut.internal;
        Acc wall, interior, all;
        for (label c = 0; c < nC; ++c)
        {
            add(all, was[c], now[c]);
            add(nearWall[c] ? wall : interior, was[c], now[c]);
        }
        auto rel = [](const Acc& a) { return a.den > 0 ? std::sqrt(a.num / a.den) : 0.0; };
        std::printf("  %-6s  L2 rel: all %.3e   near-wall(%d cells) %.3e   interior %.3e"
                    "   [near-wall share of the numerator %.1f%%]\n",
                    f == 0 ? "omega" : f == 1 ? "k" : "nut", (int)wall.n,
                    rel(all), rel(wall), rel(interior),
                    100.0 * (all.num > 0 ? wall.num / all.num : 0.0));
    }

    // The worst cells, named.
    std::vector<label> idx(nC);
    for (label c = 0; c < nC; ++c) idx[c] = c;
    std::sort(idx.begin(), idx.end(), [&](label a, label b) {
        const scalar da = std::fabs(omega.internal[a] - om0[a]) / std::fmax(om0[a], 1e-30);
        const scalar db = std::fabs(omega.internal[b] - om0[b]) / std::fmax(om0[b], 1e-30);
        return da > db;
    });
    std::printf("  worst omega cells (rel change in one correct()):\n");
    for (int i = 0; i < 6 && i < nC; ++i)
    {
        const label c = idx[i];
        std::printf("    cell %6d  C (%.4e %.4e)  y %.4e  %s  omega %.6e -> %.6e  (%.1f%%)\n",
                    (int)c, g.C()[c].x, g.C()[c].y, y[c], nearWall[c] ? "near-wall" : "interior ",
                    om0[c], omega.internal[c],
                    100.0 * (omega.internal[c] - om0[c]) / std::fmax(om0[c], 1e-30));
    }
    return 0;
}
