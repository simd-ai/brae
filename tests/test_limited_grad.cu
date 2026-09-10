// brae's cellLimited Gauss linear gradient against OpenFOAM's OWN, cell by cell.
//
// The oracle is tools/dumpLimitedGrad, which makes exactly the call linearUpwind makes for its correction
// -- fv::gradScheme<scalar>::New(mesh, mesh.gradScheme(name))().grad(vf, name) -- on a field read as
// written, patch values included and NOT re-evaluated. This probe reads the same file the same way and
// runs fvc::gaussGrad followed by cellLimitGrad, the two calls the host closures make.
//
// Two comparisons, and the first is the control: the UNLIMITED Gauss linear gradient must already be
// exact, or a disagreement in the limited one cannot be pinned on the limiter.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "patch_entry_lookup.cuh"
#include "fvc.cuh"
#include "cellLimitedGrad_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;

static std::vector<vector> readOracle(const std::string& path, label nC)
{
    std::vector<vector> g(static_cast<std::size_t>(nC));
    std::vector<char> seen(static_cast<std::size_t>(nC), 0);
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag != "GRAD") continue;
        label c = -1;
        double x = 0, y = 0, z = 0;
        ss >> c >> x >> y >> z;
        if (c < 0 || c >= nC) continue;
        g[static_cast<std::size_t>(c)] = vector{x, y, z};
        seen[static_cast<std::size_t>(c)] = 1;
    }
    for (char s : seen)
        if (!s) { std::printf("oracle %s does not cover every cell\n", path.c_str()); std::exit(2); }
    return g;
}

// Worst |brae - OF| over cells, scaled by the largest |OF| component so a near-zero gradient (a uniform
// region) is compared absolutely against the field's own gradient scale.
static double worst(const std::vector<vector>& a, const std::vector<vector>& b, label& at)
{
    double scale = 0.0;
    for (const vector& v : b) scale = std::fmax(scale, std::fmax(std::fabs(v.x), std::fmax(std::fabs(v.y), std::fabs(v.z))));
    double w = 0.0;
    at = -1;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const double d = std::fmax(std::fabs(a[i].x - b[i].x), std::fmax(std::fabs(a[i].y - b[i].y), std::fabs(a[i].z - b[i].z)));
        if (d > w) { w = d; at = static_cast<label>(i); }
    }
    return scale > 0.0 ? w / scale : w;
}

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        std::printf("usage: %s <caseDir> <timeDir> <field> <oracleUnlimited> <oracleLimited> <cellLimitK>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], timeDir = argv[2], field = argv[3];
    const scalar limK = (argc > 6) ? std::atof(argv[6]) : scalar(1);

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FieldData<scalar> fd = readField<scalar>(timeDir + "/" + field);
    if (fd.internalUniform) { std::printf("field %s is uniform -- nothing to compare\n", field.c_str()); return 2; }
    const std::vector<scalar>& vf = fd.internalField;
    std::vector<std::vector<scalar>> vb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const PatchFieldData<scalar>* e = findPatchEntry(fd.boundary, patches[pi]);
        const label n = patches[pi].size;
        if (!e || !e->hasValue)
        {
            std::printf("patch %s has no written value -- the oracle read one\n", patches[pi].name.c_str());
            return 2;
        }
        vb[pi].assign(static_cast<std::size_t>(n), e->uniformValue);
        if (!e->valueUniform)
            for (label i = 0; i < n && i < (label)e->values.size(); ++i) vb[pi][i] = e->values[i];
    }

    std::vector<vector> gU = fvc::gaussGrad(vf, vb, m, g, patches);
    std::vector<vector> gL = gU;
    cpu::cellLimitGrad(gL, vf, vb, limK, m, g, patches);

    const std::vector<vector> ofU = readOracle(argv[4], nC);
    const std::vector<vector> ofL = readOracle(argv[5], nC);
    label atU = -1, atL = -1;
    const double eU = worst(gU, ofU, atU);
    const double eL = worst(gL, ofL, atL);
    std::printf("  Gauss linear            vs OpenFOAM: %.6e  (worst cell %d)\n", eU, (int)atU);
    std::printf("  cellLimited Gauss lin %g vs OpenFOAM: %.6e  (worst cell %d)\n", (double)limK, eL, (int)atL);
    if (atL >= 0)
    {
        const std::size_t c = static_cast<std::size_t>(atL);
        std::printf("    cell %d: brae (%.17g %.17g %.17g)\n", (int)atL, gL[c].x, gL[c].y, gL[c].z);
        std::printf("    cell %d: OF   (%.17g %.17g %.17g)\n", (int)atL, ofL[c].x, ofL[c].y, ofL[c].z);
        std::printf("    cell %d: unlimited (%.17g %.17g %.17g)\n", (int)atL, gU[c].x, gU[c].y, gU[c].z);
    }
    // How many cells the two limited gradients disagree on at all, beyond round-off.
    int nBad = 0;
    for (std::size_t i = 0; i < gL.size(); ++i)
    {
        const double d = std::fmax(std::fabs(gL[i].x - ofL[i].x), std::fmax(std::fabs(gL[i].y - ofL[i].y), std::fabs(gL[i].z - ofL[i].z)));
        const double s = std::fmax(1e-300, std::fmax(std::fabs(ofL[i].x), std::fmax(std::fabs(ofL[i].y), std::fabs(ofL[i].z))));
        if (d > 1e-9 * s && d > 1e-12) ++nBad;
    }
    std::printf("  cells where the LIMITED gradients disagree beyond round-off: %d of %d\n", nBad, (int)nC);
    return (eU < 1e-12 && eL < 1e-12) ? 0 : 1;
}
