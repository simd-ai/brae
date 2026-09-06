// Registers the instrumented kEpsilon as a selectable compressible RAS model, so a case reaches it with
// `RASModel kEpsilonDump;` and `libs ("libdumpKEpsilon.so");` -- OpenFOAM's own model, its own equations,
// with writes added.
#include "turbulentFluidThermoModels.H"
#include "kEpsilonDump.H"
makeRASModel(kEpsilonDump);
