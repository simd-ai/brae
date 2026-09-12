// Phase 5 gate, fvMatrix A()/H() + fvc::div(phi) + face flux vs OpenFOAM (dumpMomentum ops.dat).
// Run: test_fvc_ops <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;
static void cmpS(const std::vector<scalar>& a, const std::vector<scalar>& of, const char* nm, scalar tol = 1e-11) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < of.size(); ++i) { mx = std::fmax(mx, std::fabs(a[i] - of[i])); mg = std::fmax(mg, std::fabs(of[i])); }
    const scalar rel = mg > 0 ? mx / mg : mx; if (rel > tol) ++g_fails;
    std::printf("  %-12s n=%6zu rel=%.3e %s\n", nm, of.size(), rel, rel <= tol ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, m.nCells());
    U.evaluateBoundary();
    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    std::vector<std::vector<scalar>> phiBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phiBnd[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phiBnd[pi] = b.values;
    }

    FvVectorMatrix UEqn = fvm::div(phiF.internalField, phiBnd, U, m, patches);
    addEqual(UEqn, fvm::laplacian(U, 1e-5, m, g, patches), -1.0);

    const std::vector<scalar> A = matrixA(UEqn, m, g, patches);
    const std::vector<vector> H = matrixH(UEqn, U, m, g, patches);

    SurfaceScalarField phiS; phiS.internal = phiF.internalField; phiS.boundary = phiBnd;
    const std::vector<scalar> divPhi = fvc::div(phiS, m, g, patches);
    const SurfaceScalarField fluxU = fvc::flux(U, m, g, patches);

    std::ifstream in(caseDir + "/ops.dat");
    if (!in) { std::printf("FAIL no ops.dat\n"); return 1; }
    label nC, nIf, np; in >> nC >> nIf >> np;
    std::vector<scalar> Aof(nC), divOf(nC), fluxIntOf(nIf);
    std::vector<vector> Hof(nC);
    for (auto& v : Aof) in >> v;
    for (auto& v : Hof) in >> v.x >> v.y >> v.z;
    for (auto& v : divOf) in >> v;
    for (auto& v : fluxIntOf) in >> v;

    // flatten H to compare component-wise.
    std::vector<scalar> Hcf, Hofx;
    for (label c = 0; c < nC; ++c) { Hcf.push_back(H[c].x); Hcf.push_back(H[c].y); Hcf.push_back(H[c].z);
                                     Hofx.push_back(Hof[c].x); Hofx.push_back(Hof[c].y); Hofx.push_back(Hof[c].z); }

    std::printf("test_fvc_ops:\n");
    cmpS(A, Aof, "A");
    cmpS(Hcf, Hofx, "H");
    // H's KNOCKED-OUT COMPONENT, asserted ON ITS OWN, because the flattened compare above cannot see it.
    // fvMatrix<Type>::H() ends by zeroing every component polyMesh::solutionD() knocks out (fvMatrix.C,
    // the validComponents loop after correctBoundaryConditions), and OpenFOAM's own dump here has H_z
    // identically zero in all 12225 cells. brae never ported that block, and on this fixture max|H_x| is
    // 2.0e+05, so a live H_z of order 1e-05 is 1e-10 of the flattened norm and passed unnoticed. What it
    // cost: with the z channel open, dropping the momentum solve OpenFOAM also drops took Uz to 13% of
    // |U| by iteration 200 on pitzDaily; with the block ported it stays at 1.1e-16 relative.
    {
        label ofNz = 0, brNz = 0;
        scalar ofMax = 0.0, brMax = 0.0;
        for (label c = 0; c < nC; ++c)
        {
            if (Hof[c].z != 0.0) ++ofNz;
            if (H[c].z != 0.0)   ++brNz;
            ofMax = std::max(ofMax, std::fabs(Hof[c].z));
            brMax = std::max(brMax, std::fabs(H[c].z));
        }
        std::printf("  %-16s OpenFOAM max %.6e in %d cells | brae max %.6e in %d cells\n",
                    "H z-component", (double)ofMax, (int)ofNz, (double)brMax, (int)brNz);
        if (ofNz != 0)
        {
            std::printf("  SKIP: this fixture is not 2D -- OpenFOAM's own H_z is nonzero, so the arm "
                        "has no knocked-out direction to check\n");
        }
        else if (brNz != 0)
        {
            ++g_fails;
            std::printf("  FAIL H z-component: OpenFOAM zeroes it in every cell (fvMatrix.C's "
                        "validComponents block) and brae leaves it live in %d\n", (int)brNz);
        }
    }
    cmpS(divPhi, divOf, "div(phi)");
    cmpS(fluxU.internal, fluxIntOf, "flux:internal");
    for (label p = 0; p < np; ++p) {
        std::string name; label sz; in >> name >> sz;
        std::vector<scalar> fb(sz); for (label i = 0; i < sz; ++i) in >> fb[i];
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k) if (patches[k].name == name) { pk = k; break; }
        cmpS(fluxU.boundary[pk], fb, (name + ":flux").c_str());
    }
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
