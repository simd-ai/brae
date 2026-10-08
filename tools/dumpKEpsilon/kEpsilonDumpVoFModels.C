// The VoF-PHASE registration of kEpsilonDump, beside the compressible one in kEpsilonDumpModels.C.
//
// interFoam's RAS tutorials do NOT reach either table that file registers. `interFoam` builds its
// turbulence through incompressibleInterPhaseTransportModel, and RAS/damBreak's lineage is
// PhaseIncompressibleTurbulenceModel<transportModel> -- a third selection table, made by
// VoFphaseTurbulentTransportModels.H (src/phaseSystemModels/twoPhaseInter/...), which defines its OWN
// makeRASModel macro against it. A library registering only the compressible table is invisible there:
// the run dies with "Unknown RASModel type kEpsilonDump" and lists the phase-incompressible table.
//
// Same one-line registration, different table, its own translation unit because the two headers define
// the macro differently.
#include "VoFphaseTurbulentTransportModels.H"
#include "kEpsilonDump.H"

makeRASModel(kEpsilonDump);
