# rhoBoxF -- `RAS { turbulence off; }` (the FROZEN turbulence model)

rhoBox with kEpsilon declared and switched off. OpenFOAM still constructs the model (k, epsilon, nut,
alphat all read) and rhoSimpleFoam.C:64 still calls turbulence->validate(), which runs correctNut()
exactly once; every per-iteration correct() then returns on its first line (kEpsilon.C:216). So the run
transports a FROZEN eddy viscosity: nut = Cmu*k^2/eps = 0.09*0.375^2/10 = 0.001265625 -- not the 1e-3
the 0/nut file says, and not zero. The 1e-3 seed is deliberate: a port that freezes at the FILE value
and one that runs laminar are both distinguishable from the validate value, which is what makes this
fixture an oracle (the seed, the validate value and zero are three different numbers).

alphat = rho0*nut/Prt with Prt = 1.0 (EddyDiffusivity's coeffDict default -- NOT the
alphatWallFunction's 0.85; this case carries no alphat wall function precisely so that distinction
cannot blur). nut/alphat boundaries are `calculated`: OF's whole-field assignment overwrites them with
the expression's boundary value, measured 0.001265625 on every patch.

Used by tf_vs_openfoam.sh: real OF runs it (converges ~97 iterations on its residualControl), then the
mirror harness must (a) leave createFields at the validate values, (b) bring nut/alphat through the
loop bit-identical, (c) match OF's converged U/T/p.
