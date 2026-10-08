// The INCOMPRESSIBLE registration of kOmegaSSTDump, beside the compressible one in
// kOmegaSSTDumpModels.C. interFoam's uniform lineage builds
// IncompressibleTurbulenceModel<transportModel> (incompressibleInterPhaseTransportModel.C), so a
// library that registers only `fluidThermoCompressibleTurbulenceModel` is invisible to it -- the run
// dies with "Unknown RASModel type kOmegaSSTDump" and lists the incompressible table.
// `turbulentTransportModels.H` defines its OWN makeRASModel against the incompressible base, so this
// is the same one-line registration against a different table, in its own translation unit because
// the two headers define the macro differently.
#include "turbulentTransportModels.H"
#include "kOmegaSSTDump.H"

makeRASModel(kOmegaSSTDump);
