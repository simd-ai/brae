// Registers the instrumented SpalartAllmarasIDDES as a selectable INCOMPRESSIBLE LES model, so a case reaches it
// with `LESModel SpalartAllmarasIDDESDump;` and `libs ("libdumpSAIDDES.so");` -- OpenFOAM's own model, its own
// statements, with writes added (BRAE_DUMP_ITER=<timeIndex>).
#include "turbulentTransportModels.H"
#include "LESModel.H"
#include "LESeddyViscosity.H"
#include "SpalartAllmarasIDDESDump.H"
makeLESModel(SpalartAllmarasIDDESDump);
