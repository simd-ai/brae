# rhoBoxDF -- DarcyForchheimer porosity on the compressible mirror

rhoBox with a 240-cell porous plug (cellZone `plug`, built by system/topoSetDict; the cellZones file
ships with the mesh) carrying d = 1e7 and f = 40 -- sized so BOTH halves of the resistance matter:
mu*d ~ 180 and 0.5*rho*|U|*f ~ 116, and the converged pressure drop is ~587 Pa against the case's ~30
Pa baseline.

The defect this exists to catch: the host mirror passed nu = 0 into the DarcyForchheimer branch of
fvOptions addSup, which ZEROED the whole Darcy term, and the Forchheimer half ran without rho -- OF's
form is Cd = mu[celli]*D + (rho[celli]*|U|)*F (DarcyForchheimerTemplates.C:53) with mu the LAMINAR
"thermo:mu" (DarcyForchheimer.C:64), not the effective viscosity. Measured with the old form restored:
U 1.78e-02 and T 8.8e-04 against 3.4e-07 / 2.6e-10 fixed. No porosity fixture existed in validation/
before this one, which is how nu = 0 survived -- the angledDuct work of past sessions compared against
OpenFOAM runs outside the tree.
