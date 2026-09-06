# simpleBoxP -- incompressible fixedFluxPressure where it does something

rhoBox's mesh run incompressibly (simpleFoam, laminar) with the one boundary shape that both
DISCRIMINATES fixedFluxPressure from zeroGradient and that OpenFOAM itself will start: the ffp patch is
a SIDE VENT (coldWall: U zeroGradient -- assignable, so constrainHbyA leaves HbyA_b and the
constrainPressure numerator is live -- plus p fixedFluxPressure), while the main outlet fixes p = 0 so
no pressure reference is needed and adjustPhi never runs. Every all-Neumann variant of this fixture
dies inside OpenFOAM's own adjustPhi at startup ("Adjustable mass outflow: 0" -- the vent's initial
flux is zero and there is nothing to scale), which is worth remembering before designing another one.

The converged vent gradient is ~-0.08 (OF writes it into the output p file); forcing it to zero
converges to a different field. Used by ffpi_vs_openfoam.sh, which runs the V2 driver
(BRAE_SIMPLEFOAM_V2=1) -- the path whose envelope used to refuse this case on a raw substring of the
0/p text.
