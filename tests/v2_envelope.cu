// Envelope probe: ask the rebuilt simpleFoam what it CANNOT run in a case, and why.
//
// The envelope is the honest coverage statement -- every blocker below is a component the rebuild
// refuses rather than silently approximating. Used to sweep OpenFOAM's own simpleFoam tutorials.
#include "simpleFoamV2.cuh"

#include <cstdio>
#include <string>

using namespace brae;
using namespace brae::gpu;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s <caseDir>\n", argv[0]);
        return 2;
    }
    const EnvelopeReport r = simpleFoamV2Envelope(argv[1]);
    std::printf("%s\n", r.supported ? "SUPPORTED" : "BLOCKED");
    for (const std::string& b : r.blockers)
    {
        std::printf("  BLOCKER %s\n", b.c_str());
    }
    for (const std::string& n : r.notices)
    {
        std::printf("  NOTICE  %s\n", n.c_str());
    }
    return 0;
}
