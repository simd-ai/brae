// Registers the instrumented kEqn as a selectable INCOMPRESSIBLE LES model, so a case reaches it with
// `LESModel kEqnDump;` and `libs ("libdumpKEqn.so");` -- OpenFOAM's own model, its own equations, with
// writes added (BRAE_DUMP_ITER=<timeIndex>).
#include "turbulentTransportModels.H"
#include "kEqnDump.H"
makeLESModel(kEqnDump);
