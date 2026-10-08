// cyclicAMI geometry on a CONFORMING channel: the AMI must reduce to cyclic, each source face overlaps exactly
// ONE target face with weight 1 (weightsSum 1), and the AMI-interpolated coupled geometry must reproduce the cyclic
// values on the uniform 10x10 periodic channel (deltaCoeffs = 1/dx = 10, interp weight = 0.5).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ami_interface.cuh"
#include "cyclic_interface.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicChannelAMI";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<AMIInterface> ai = buildAMIInterfaces(m, g, fvp);
    std::printf("AMI interfaces found: %zu\n", ai.size());

    int fail = (ai.size() != 2) ? 1 : 0;
    int maxStencil = 0; scalar maxWsum = 0, maxDC = 0, maxW = 0, maxWt = 0;
    for (const AMIInterface& a : ai) {
        for (std::size_t i = 0; i < a.ownCell.size(); ++i) {
            const int ne = (int)(a.srcOffset[i+1] - a.srcOffset[i]);
            maxStencil = std::max(maxStencil, ne);
            if (ne != 1) ++fail;                                         // conforming -> exactly one overlap
            // the COVERAGE, sum of overlap/|Sf| before the rows are divided by their sum (after it the sum
            // of weights is 1 by construction and would hold nothing)
            maxWsum = std::fmax(maxWsum, std::fabs(a.coverage[i] - 1.0));
            maxDC   = std::fmax(maxDC,   std::fabs(a.deltaCoeffs[i] - 10.0));
            maxW    = std::fmax(maxW,    std::fabs(a.weights[i] - 0.5));
            for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k)
                maxWt = std::fmax(maxWt, std::fabs(a.weight[k] - 1.0));   // single weight == 1
        }
        std::printf("  patch %d -> nbr %d : %zu src faces, nnz=%zu\n",
                    (int)a.patch, (int)a.nbrPatch, a.ownCell.size(), a.nbrCell.size());
    }
    std::printf("max stencil/face=%d  max|wsum-1|=%.3e  max|w-1|=%.3e  max|deltaCoeffs-10|=%.3e  max|interpw-0.5|=%.3e\n",
                maxStencil, maxWsum, maxWt, maxDC, maxW);
    const bool pass = !fail && maxStencil == 1 && maxWsum < 1e-9 && maxWt < 1e-9 && maxDC < 1e-9 && maxW < 1e-9;
    std::printf("%s\n", pass ? "test_ami_geometry PASS" : "test_ami_geometry FAIL");
    return pass ? 0 : 1;
}
