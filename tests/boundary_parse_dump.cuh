#pragma once
// The text of what readBoundary made of a polyMesh/boundary, restricted to the PatchInfo fields that existed
// before processor patches were read -- so the same function compiles against the old PatchInfo and the new
// one. tests/data/boundary_parse_golden.txt was produced by it with the reader as it stood at 2a7a5268 (before
// myProcNo/neighbProcNo/referPatch), over every polyMesh/boundary the repository tracks.
#include "primitive_mesh.cuh"
#include <cstdio>
#include <string>
#include <vector>

namespace brae_test {

inline std::string dumpLegacyPatch(const brae::PatchInfo& p)
{
    char buf[1024];
    std::string groups;
    for (const std::string& g : p.inGroups) groups += g + ",";
    std::snprintf(buf, sizeof buf, "%s|%s|%ld|%ld|%s|%s|%s|%ld|%.17g|%s|%s|%.17g,%.17g,%.17g|%.17g,%.17g,%.17g|%d|%d|%d\n",
                  p.name.c_str(), p.type.c_str(), (long)p.start, (long)p.size, p.neighbourPatch.c_str(),
                  p.nonOverlapPatch.c_str(), p.periodicPatch.c_str(), (long)p.maxIter, (double)p.matchTolerance,
                  groups.c_str(), p.transform.c_str(), (double)p.rotationAxis.x, (double)p.rotationAxis.y,
                  (double)p.rotationAxis.z, (double)p.rotationCentre.x, (double)p.rotationCentre.y,
                  (double)p.rotationCentre.z, (int)p.acmiScale.empty(), (int)p.acmiScaleCoded,
                  (int)p.acmiScaleCodedSpec.code.size());
    return buf;
}

// `## <rel>` then one line per patch, or `## <rel> THROWS` when the reader refuses the file.
inline std::string dumpLegacyBoundary(const std::string& repoRoot, const std::string& relBoundaryDir)
{
    brae::PrimitiveMesh m;
    try
    {
        m.readBoundary(repoRoot + "/" + relBoundaryDir);
    }
    catch (const std::exception&)
    {
        return "## " + relBoundaryDir + " THROWS\n";
    }
    std::string s = "## " + relBoundaryDir + "\n";
    for (const brae::PatchInfo& p : m.patches()) s += dumpLegacyPatch(p);
    return s;
}

}   // namespace brae_test
