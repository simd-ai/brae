// atmBoundaryLayerInlet{K,Epsilon} vs the values OpenFOAM itself writes -- the YGCJ factor.
//
// OF's profile carries sqrt(C1*log((z+z0)/z0) + C2) on k (atmBoundaryLayer.C:238-240) and epsilon
// (:252-254); omega has no factor (:258-267). brae computed the DEFAULT (C1=0, C2=1 -> factor 1)
// whatever the case said, because the reader never parsed C1/C2 -- a height-CONSTANT k under a case
// that asked for the curve fit. Parse-level: the BC evaluates in its constructor, so one OF iteration
// writes the exact per-face oracle into <t>/k and <t>/epsilon regardless of the solve.
#include "foam_field_reader.cuh"
#include "fv_patch_field.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include <cmath>
#include <cstdio>
#include <string>

using namespace brae;

static int failures = 0;
static void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("  %-62s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <ofTime>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    for (const char* fld : {"k", "epsilon"})
    {
        const FieldData<scalar> in = readField<scalar>(caseDir + "/0/" + fld);
        const FieldData<scalar> of = readField<scalar>(caseDir + "/" + t + "/" + fld);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].name != "inlet") continue;
            auto byName = [&](const FieldData<scalar>& fd) -> const PatchFieldData<scalar>*
            {
                for (const auto& b : fd.boundary) if (b.name == patches[pi].name) return &b;
                return nullptr;
            };
            const PatchFieldData<scalar>* ib = byName(in);
            const PatchFieldData<scalar>* ob = byName(of);
            check(std::string(fld) + ": fixture carries non-default C1/C2 (engagement)",
                  ib && (ib->ablC1 != 0.0 || ib->ablC2 != 1.0));
            auto pf = makePatchField<scalar>(patches[pi], *ib);
            pf->evaluate({});
            const std::vector<scalar>& bv = pf->value();
            double dMax = 0, vMin = 1e300, vMax = -1e300;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const double o = ob->values[static_cast<std::size_t>(i)];
                dMax = std::max(dMax, std::fabs((double)bv[static_cast<std::size_t>(i)] - o) / std::max(std::fabs(o), 1e-30));
                vMin = std::min(vMin, o); vMax = std::max(vMax, o);
            }
            std::printf("     %-8s max rel |brae - OF| %.3e over %d faces (OF range %.4g..%.4g)\n",
                        fld, dMax, (int)patches[pi].size, vMin, vMax);
            // identical closed form on identical face centres: 1e-8 is generous for parse noise
            check(std::string(fld) + " matches OpenFOAM's written inlet profile", dMax < 1e-8);
            // NON-VACUOUS: with C1 != 0 the profile VARIES with height; the pre-fix flat profile
            // (factor 1) differs from OF by the whole factor and cannot pass the bound above -- and a
            // fixture regression to default C1/C2 flattens OF's own profile, caught here.
            if (std::string(fld) == "k")
                check("...and OF's profile is height-VARYING (else the factor is inert)",
                      (vMax - vMin) / vMax > 0.05);
        }
    }

    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
