# rhoBoxP -- fixedFluxPressure where it actually does something

rhoBox with the outlet turned into the one combination that DISCRIMINATES fixedFluxPressure from the
zeroGradient it was silently mapped to: `U inletOutlet` (assignable, so constrainHbyA does NOT overwrite
HbyA_b and the constrainPressure numerator does not cancel) + `p fixedFluxPressure`, with no p patch
fixing a value (pRefCell 0, pRefValue 1e5 in SIMPLE). At every fixed-velocity patch the cancellation
phiHbyA_b == rho_b*(Sf&U_b) is EXACT -- X minus X -- which is why a zeroGradient substitution survived
every ordinary fixture; here the converged outlet gradient is -0.73 Pa/m (OF writes it into the output
p file) and forcing it to zero converges to a different field.

Note the fixture also has to drop rhoMin/rhoMax: with no fixed-value p patch, OpenFOAM's own
pressureControl refuses the backward-compat rho bounds ("the corresponding reference density cannot be
evaluated from the boundary conditions").

Used by ffp_vs_openfoam.sh, twice: SIMPLE and SIMPLEC (`consistent yes`), the second reaching
rhoPcEqn's constrainPressure (pcEqn.H:16) with its rhorAtU divisor.
