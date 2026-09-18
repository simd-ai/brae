// brae's interface curvature against OpenFOAM's OWN, pass for pass.
//
// THE ORACLE IS OpenFOAM'S UNMODIFIED interfaceProperties, read through tools/dumpInterfaceK. That
// utility does not copy or patch the class: nHatf() and sigmaK() are public (interfaceProperties.H:
// 139-144) and immiscibleIncompressibleTwoPhaseMixture inherits them, so what comes out is what
// OpenFOAM computes, with no transcription of its source at all.
//
// WHY "PASS FOR PASS" IS THE WHOLE POINT. calculateK is a FIXED-POINT ITERATION, not a pure function:
// it reads alpha's wall gradient, and correctContactAngle writes that gradient at the end of its own
// previous pass. So the curvature at a contact line depends on HOW MANY TIMES it has run. On
// capillaryRise the wall gradient goes
//
//     pass 1: 7070.5    pass 2: 8659.4    pass 3: 9353.1    pass 4: 9681.2
//
// converging but not converged, and interFoam calls it four times before the first momentum equation
// (the constructor, once per alpha corrector per sub-cycle, and once more after the sub-cycle). A gate
// that compared a single pass would report an 18% error in a correct implementation -- which is
// exactly what happened before this file existed.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_case_cpp.cuh"
#include "interface_properties_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== brae's interface curvature vs OpenFOAM's own ==\n");
    if (argc < 4) { std::printf("  SKIP: usage: %s <case> <timeDir> <passes>\n", argv[0]); return 77; }
    const std::string caseDir = argv[1], t = argv[2];
    const int passes = std::atoi(argv[3]);
    if (!std::filesystem::exists(t + "/K.dump"))
    { std::printf("  SKIP: no K.dump in %s (dumpInterfaceK not run)\n", t.c_str()); return 77; }

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    InterFields f = buildInterFields(caseDir, t, m, g, patches);   // runs pass 1, as the ctor does
    for (int i = 1; i < passes; ++i)
        brae::cpu::interfaceProps::calculateK(f.alpha1, f.interface, m, g, patches, false, f.nHatf, f.K);
    std::printf("  %d calculateK pass(es); brae deltaN = %.6e\n",
                passes, (double)brae::cpu::interfaceProps::deltaN(g.V()));

    const FieldData<scalar> kd = readField<scalar>(t + "/K.dump");
    const std::vector<scalar> ofK = kd.internalUniform
        ? std::vector<scalar>(static_cast<std::size_t>(nC), kd.internalUniformValue) : kd.internalField;
    check("OpenFOAM's K has one value per cell", ofK.size() == static_cast<std::size_t>(nC));

    scalar worst = 0, ref = 0, sumB = 0, sumO = 0;
    label  cw = -1;
    for (label c = 0; c < nC; ++c)
    {
        const scalar e = std::fabs(f.K[c] - ofK[c]);
        if (e > worst) { worst = e; cw = c; }
        ref  = std::fmax(ref, std::fabs(ofK[c]));
        sumB += std::fabs(f.K[c]);
        sumO += std::fabs(ofK[c]);
    }
    std::printf("  K: worst |brae - OpenFOAM| = %.4e  (|K| up to %.4e, relative %.3e)\n",
                (double)worst, (double)ref, (double)(worst/ref));
    std::printf("     sum|brae| %.6e  sum|OpenFOAM| %.6e  ratio %.8f\n",
                (double)sumB, (double)sumO, (double)(sumB/sumO));
    if (cw >= 0)
        std::printf("     worst cell %ld at (%.5f %.5f): brae %.6e  OpenFOAM %.6e\n",
                    (long)cw, (double)g.C()[cw].x, (double)g.C()[cw].y,
                    (double)f.K[cw], (double)ofK[cw]);

    // THE ORACLE'S PRECISION HAD TO BE RAISED BEFORE THESE BOUNDS MEANT ANYTHING. dumpInterfaceK
    // writes ascii, and at OpenFOAM's default writePrecision of 6 a K of order 4e+04 is written to an
    // absolute precision of 0.01 -- which is exactly where the first run of this gate sat. The shell
    // script now sets writePrecision 15, so what is left is the discretisation rather than the file
    // format, and the bounds below are set from that.
    check("brae's curvature matches OpenFOAM's", worst < scalar(1e-9) * ref);
    check("...and so does its total, which no single cell can carry",
          std::fabs(sumB/sumO - scalar(1)) < scalar(1e-10));

    // ...and the contact-angle wall gradient, which is what makes the curvature a fixed point.
    if (std::filesystem::exists(t + "/alphaAfterCorrect.dump"))
    {
        const FieldData<scalar> ad = readField<scalar>(t + "/alphaAfterCorrect.dump");
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (f.alpha1.boundary[pi]->contactAngleTheta0() < scalar(0)) continue;
            const std::vector<scalar> bg = f.alpha1.boundary[pi]->snGrad(f.alpha1.internal);
            const PatchFieldData<scalar>* ofp = nullptr;
            for (const auto& b : ad.boundary) if (b.name == patches[pi].name) ofp = &b;
            if (!ofp || ofp->gradientValues.size() != bg.size()) continue;
            scalar w = 0, r = 0;
            std::size_t nbz = 0, noz = 0;
            for (std::size_t i = 0; i < bg.size(); ++i)
            {
                w = std::fmax(w, std::fabs(bg[i] - ofp->gradientValues[i]));
                r = std::fmax(r, std::fabs(ofp->gradientValues[i]));
                if (std::fabs(bg[i]) > scalar(1e-30)) ++nbz;
                if (std::fabs(ofp->gradientValues[i]) > scalar(1e-30)) ++noz;
            }
            std::printf("  patch '%s': %zu non-zero wall gradients (OpenFOAM %zu); "
                        "worst diff %.4e of %.4e\n",
                        patches[pi].name.c_str(), nbz, noz, (double)w, (double)r);
            check("the contact line is in the SAME PLACE -- the same faces carry a gradient", nbz == noz);
            check("...and carries the same gradient", w < scalar(1e-9) * r);
        }
    }

    std::printf("test_inter_curvature_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
