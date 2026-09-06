// actuationDiskSource (Froude) against the numbers OpenFOAM writes for itself.
//
// OpenFOAM's actuationDiskSource writes postProcessing/<name>/<t>/actuationDiskSource.dat with, per
// iteration, the monitored Uref, Cp, Ct, the induction factor a and the thrust T. Uref and T between
// them cover the whole model: the monitor-cell selection, the mean, the disk direction projection, the
// induction factor and the disk area. Computing them from OpenFOAM's own converged U is a statement
// about the source term alone.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fv_options.cuh"
#include "actuationDiskSource_cpp.cuh"
#include "mrf_read.cuh"   // readCellZones

#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>

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
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    std::map<std::string, std::vector<label>> zones;
    try { zones = readCellZones(caseDir + "/constant/polyMesh"); }
    catch (const std::exception& e) { std::printf("actuationdisk_probe: could not read constant/polyMesh/cellZones: %s\n", e.what()); return 1; }
    FvOptionsData fvo;
    try { fvo = readFvOptions(caseDir, zones, g.V(), nC, g.C()); }
    catch (const std::exception& e) { std::printf("actuationdisk_probe: could not read the fvOptions dictionary: %s\n", e.what()); return 1; }
    if (!fvo.adActive || fvo.adDisks.empty())
    {
        std::printf("  FAIL: no active actuationDiskSource read from the case\n");
        return 1;
    }

    GeometricField<vector> U;
    try
    {
        U = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
        U.evaluateBoundary();
    }
    catch (const std::exception& e) { std::printf("actuationdisk_probe: could not read the velocity field: %s\n", e.what()); return 1; }

    std::printf("actuationdisk_probe: %s/%s   %zu turbine(s)\n", caseDir.c_str(), t.c_str(),
                fvo.adDisks.size());
    std::vector<vector> source(nC, vector{0, 0, 0});
    for (std::size_t i = 0; i < fvo.adDisks.size(); ++i)
    {
        const auto& d = fvo.adDisks[i];
        const cpu::ActuationDiskState st = cpu::addSupFroude(d, U.internal, g.V(), source);
        std::printf("  disk%zu  cells %zu  monitor %zu  area %.6g  a %.12e\n",
                    i + 1, d.diskCells.size(), d.monitorCells.size(), d.area, st.a);
        std::printf("  disk%zu  Uref (%.12e %.12e %.12e)\n", i + 1, st.Uref.x, st.Uref.y, st.Uref.z);
        std::printf("  disk%zu  T %.12e\n", i + 1, st.T);
    }
    return 0;
}
