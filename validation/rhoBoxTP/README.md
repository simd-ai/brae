# rhoBoxTP -- the pressure-driven inlet (totalPressure + pressureInletOutletVelocity)

rhoBox driven by pressure instead of velocity: inlet p totalPressure (p0 100050), inlet U
pressureInletOutletVelocity, outlet p fixedValue 1e5. This is the shape whose mirror run diverged to
T ~1e+78 for a whole campaign ("rhoTP diverges, driver exonerated, three hypotheses eliminated") --
the cause was one line: TotalPressurePatchField captured p0_ from this->value() IN ITS CONSTRUCTOR,
before any evaluate() had filled value_, so p0 was EMPTY and every update ran p = 0 - 0.5*rho*|U|^2
(-14.52 Pa on a 100050 Pa inlet, iteration 1). The device arm reads its own p0 buffer, and the only
totalPressure gate ran that arm -- which is why it survived. Used by tpm_vs_openfoam.sh (the MIRROR
arm's totalPressure gate); the iteration-1 inlet pressure IS the discriminator: ~100035.5 fixed,
~-14.5 broken, nothing in between.
