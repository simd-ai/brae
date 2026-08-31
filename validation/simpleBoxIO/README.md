# simpleBoxIO -- adjustPhi with an inletOutlet outflow, incompressible

rhoBox's mesh run through simpleFoam with the shape that makes adjustPhi's patch classification
load-bearing: the ONLY outflow is `U inletOutlet` and no p patch fixes a value (pRefCell 0), so
adjustPhi runs every iteration and everything hangs on `fixesValue() && !isA<inletOutlet>`
(adjustPhi.C:59) -- mixed fixesValue() is TRUE and inletOutlet inherits it, so the second half is what
keeps the outflow ADJUSTABLE.

Two defects were live on exactly this shape: pEqn_cpp's SCALE loop tested fixesValue alone (massCorr
computed over one face set, applied to another -- net boundary flux off by the whole outflow), and
simpleFoamV2's device `adjustable` mask had the same missing half, where deviceAdjustPhi then REFUSED
the case OpenFOAM solves ("adjustable mass outflow 0.000000"). Note fixedFluxPressure at such an
outlet makes OpenFOAM ITSELF fatal at startup (see rhoBoxP's README for the compressible contrast);
plain zeroGradient p is what runs. Used by adjm_vs_openfoam.sh, host arm + V2 arm.
