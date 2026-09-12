// Does OpenFOAM's converged solution satisfy brae's _cpp kEpsilon equations?
//
// Same residual oracle as kOmegaSST and realizableKE, run on a case brae REPRODUCES (pitzDailyTurb) and
// one it does not (simpleCar). A term that is wrong everywhere shows up on both; a term that is wrong
// only in some regime shows up on one, and the difference between the two cases says which.
//
// Run: kepsilon_probe <caseDir> <ofTimeDir>
#include "kEpsilon_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <ofTimeDir>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    GeometricField<scalar> k =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/k"), fvp, nC);
    GeometricField<scalar> eps =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/epsilon"), fvp, nC);
    GeometricField<scalar> nut =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), fvp, nC);
    U.evaluateBoundary();
    k.evaluateBoundary();
    eps.evaluateBoundary();
    nut.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    SurfaceScalarField phi;
    phi.internal = phiF.internalField;
    phi.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phi.boundary[pi].assign(fvp[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
        {
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
            {
                phi.boundary[pi] = b.values;
            }
        }
    }

    const FoamDict tp = readDict(caseDir + "/constant/transportProperties");
    const scalar nu = tp.scalarOr("nu", 1e-5);
    // The coefficients have no shared reader, so they are read here from RAS.kEpsilonCoeffs; absent keys
    // keep the OpenFOAM defaults the struct already carries.
    KEpsilonCoeffs co;
    {
        const FoamDict turb = readDict(caseDir + "/constant/turbulenceProperties");
        const FoamDict* ras = turb.subDict("RAS");
        const FoamDict* kc = ras ? ras->subDict("kEpsilonCoeffs") : nullptr;
        if (kc)
        {
            co.Cmu = kc->scalarOr("Cmu", co.Cmu);
            co.C1 = kc->scalarOr("C1", co.C1);
            co.C2 = kc->scalarOr("C2", co.C2);
            co.C3 = kc->scalarOr("C3", co.C3);
            co.sigmaK = kc->scalarOr("sigmak", co.sigmaK);
            co.sigmaEps = kc->scalarOr("sigmaEps", co.sigmaEps);
        }
    }

    std::printf("kepsilon_probe %s @ %s: nC=%d nu=%.3g\n", caseDir.c_str(), t.c_str(), (int)nC, nu);

    auto fresh = [&](const std::string& name)
    {
        GeometricField<scalar> f =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + name), fvp, nC);
        f.evaluateBoundary();
        return f;
    };

    // 1. the residual of our equations at OpenFOAM's converged state
    cpu::kEpsilonRef::KEResiduals r0;
    {
        GeometricField<scalar> kk = fresh("k");
        GeometricField<scalar> ee = fresh("epsilon");
        GeometricField<scalar> nn = fresh("nut");
        cpu::kEpsilonRef::correct(U, kk, ee, nn, phi, nu, m, g, fvp,
                                  1.0, 1.0, 1e-12, 0.0, 2000, co, &r0, false);
        std::printf("  residual, UNRELAXED  : epsilon %.4e   k %.4e   (wall cells %d)\n",
                    r0.epsilon, r0.k, (int)r0.wallCells);
    }
    {
        // THE CONTROL THIS PROBE NEEDED FROM THE START. OpenFOAM's logged residual is computed on the
        // RELAXED matrix -- relaxation inflates the diagonal, which changes the normalisation the
        // residual is divided by. Comparing an unrelaxed residual against OpenFOAM's logged one is not a
        // like-for-like comparison, so the case's own relaxation factors are used here.
        GeometricField<scalar> kk = fresh("k");
        GeometricField<scalar> ee = fresh("epsilon");
        GeometricField<scalar> nn = fresh("nut");
        cpu::kEpsilonRef::KEResiduals rr;
        cpu::kEpsilonRef::correct(U, kk, ee, nn, phi, nu, m, g, fvp,
                                  0.7, 0.7, 1e-12, 0.0, 2000, co, &rr, false);
        std::printf("  residual, RELAXED 0.7: epsilon %.4e   k %.4e\n", rr.epsilon, rr.k);
    }
    {
        GeometricField<scalar> kk = fresh("k");
        GeometricField<scalar> ee = fresh("epsilon");
        GeometricField<scalar> nn = fresh("nut");
        cpu::kEpsilonRef::KEResiduals rb;
        cpu::kEpsilonRef::correct(U, kk, ee, nn, phi, nu, m, g, fvp,
                                  1.0, 1.0, 1e-12, 0.0, 2000, co, &rb, true);
        std::printf("  residual, bounded ON : epsilon %.4e   k %.4e\n", rb.epsilon, rb.k);
    }

    // 1c. WHERE does the residual live? Split the per-cell |b - A.psi| by region.
    {
        GeometricField<scalar> kk = fresh("k");
        GeometricField<scalar> ee = fresh("epsilon");
        GeometricField<scalar> nn = fresh("nut");
        cpu::kEpsilonRef::KEResiduals rw;
        cpu::kEpsilonRef::correct(U, kk, ee, nn, phi, nu, m, g, fvp,
                                  0.7, 0.7, 1e-12, 0.0, 2000, co, &rw, false);
        if (!rw.epsCellResidual.empty())
        {
            std::vector<int> tag(nC, 0);   // 0 interior, 1 wall-function cell, 2 other boundary cell
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                // `empty` patches are excluded: on a 2D mesh the front/back pair touches EVERY cell, so
                // counting them tags the whole domain as boundary and the split says nothing.
                if (fvp[pi].type == "empty") continue;

                const int what = ee.boundary[pi]->isTurbulenceWallFunction() ? 1 : 2;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    int& t = tag[fvp[pi].faceCells[i]];
                    if (what == 1 || t == 0) t = what;
                }
            }
            scalar tot = 0;
            scalar sum[3] = {0, 0, 0};
            label  cnt[3] = {0, 0, 0};
            for (label c = 0; c < nC; ++c)
            {
                tot += rw.epsCellResidual[c];
                sum[tag[c]] += rw.epsCellResidual[c];
                ++cnt[tag[c]];
            }
            const char* nm[3] = {"interior", "wall-function cells", "other boundary cells"};
            for (int q = 0; q < 3; ++q)
            {
                std::printf("    %-22s %6d cells  %6.2f%% of the epsilon residual\n",
                            nm[q], (int)cnt[q], tot > 0 ? 100.0*sum[q]/tot : 0.0);
            }

            // ...and per PATCH, which is what names the boundary condition responsible.
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (fvp[pi].type == "empty") continue;
                scalar sp = 0;
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    sp += rw.epsCellResidual[fvp[pi].faceCells[i]];
                }
                std::printf("      patch %-14s (%-22s) %6.2f%%\n",
                            fvp[pi].name.c_str(),
                            ee.boundary[pi]->isTurbulenceWallFunction() ? "epsilonWallFunction" : "other",
                            tot > 0 ? 100.0*sp/tot : 0.0);
            }
        }
    }

    // 1b. TERM BISECT. Drop one epsilon term at a time. A CORRECT term makes the residual worse when
    //     removed; the term that is WRONG is the one whose removal makes it BETTER.
    {
        const char* names[9] =
        {
            "",
            "eps: production C1*GbyNu*Cmu*k",
            "eps: divU SuSp",
            "eps: destruction C2*eps/k",
            "eps: diffusion nut/sigmaEps",
            "k:   production G",
            "k:   divU SuSp",
            "k:   destruction eps/k",
            "k:   diffusion nut/sigmak"
        };
        for (int d = 1; d <= 8; ++d)
        {
            GeometricField<scalar> kk = fresh("k");
            GeometricField<scalar> ee = fresh("epsilon");
            GeometricField<scalar> nn = fresh("nut");
            cpu::kEpsilonRef::KEResiduals rd;
            cpu::kEpsilonRef::correct(U, kk, ee, nn, phi, nu, m, g, fvp,
                                      1.0, 1.0, 1e-12, 0.0, 2000, co, &rd, false, d);
            const scalar r = (d <= 4) ? rd.epsilon : rd.k;
            const scalar base = (d <= 4) ? r0.epsilon : r0.k;
            std::printf("    drop %-32s -> %.4e  (%s)\n", names[d], r,
                        r < base ? "BETTER -- suspect" : "worse (term is doing work)");
        }
    }

    // 2. nut: does OUR correctNut reproduce OpenFOAM's nut from OpenFOAM's own k and epsilon?
    //    This is the one term that can be checked WITHOUT solving anything, so it is checked first --
    //    if Cmu*k^2/eps does not reproduce nut, nothing downstream can agree.
    {
        const std::vector<scalar> ours =
            cpu::kEpsilonRef::correctNut(k.internal, eps.internal, co);
        scalar dmax = 0;
        scalar mg = 0;
        label worst = -1;
        for (label c = 0; c < nC; ++c)
        {
            const scalar d = std::fabs(ours[c] - nut.internal[c]);
            if (d > dmax)
            {
                dmax = d;
                worst = c;
            }
            mg = std::fmax(mg, std::fabs(nut.internal[c]));
        }
        std::printf("  nut = Cmu*k^2/eps vs OpenFOAM's nut: max|d| %.4e  rel %.4e  (cell %d: %.4g vs %.4g)\n",
                    dmax, mg > 0 ? dmax/mg : dmax, (int)worst,
                    worst >= 0 ? ours[worst] : 0.0, worst >= 0 ? nut.internal[worst] : 0.0);
    }

    // 3. G: does nut*GbyNu reproduce a sane production, and how much do the wall cells change it?
    {
        const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, fvp);
        const std::vector<scalar> gb = cpu::kEpsilonRef::GbyNu(gradU);
        scalar lo = 1e30;
        scalar hi = -1e30;
        for (label c = 0; c < nC; ++c)
        {
            lo = std::fmin(lo, gb[c]);
            hi = std::fmax(hi, gb[c]);
        }
        std::printf("  GbyNu in [%.4e, %.4e]\n", lo, hi);
    }
    return 0;
}
