// Registers the instrumented kOmegaSST as a selectable compressible RAS model, so a case reaches it with
// `RASModel kOmegaSSTDump;` and `libs ("libdumpKOmegaSST.so");` -- OpenFOAM's own model, its own
// equations, with writes added.
#include "turbulentFluidThermoModels.H"
#include "kOmegaSSTDump.H"
makeRASModel(kOmegaSSTDump);
