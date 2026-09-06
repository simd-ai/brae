# rhoBoxSym -- a TILTED symmetry plane, compressible

rhoBox's duct with the top replaced by a symmetryPlane whose normal is ~4 degrees off-axis (top edge
rises 0.2 -> 0.35 over the 2 m length). Axis-aligned symmetry can never discriminate a symmetry
defect on a segregated solver -- with n along one axis the per-component treatment (vf = |n_k|) is
exact and a stale snapshot equals a fresh one, which is why the missing deviceUpdateSymmetry/Wedge
calls sat unwired on the rho device driver with nothing able to notice. The tilt couples Ux and Uy.

Measured here: the HOST mirror lands U 8.2e-06 / T 4.7e-09 / p 1.3e-10 from OpenFOAM's converged
state (its live SymmetryPatchField is exact).

The DEVICE arm used to REFUSE this case by name, and the reason it drifted was misattributed twice
before it was found. It was never the segregated model -- that is exact for any normal (OpenFOAM's own
symmetryPlaneFvPatchField::snGradTransformDiag() is the component magnitudes). It was WHEN the mixed
refValue that carries the symmetry value was rebuilt: the driver refreshed U's boundary after the
momentum solve and after the velocity correction without rebuilding that ref first, so both refreshes
blended the new U_c towards a ref built from the U the iteration started with. OpenFOAM's evaluate()
reads patchInternalField at the moment of evaluation and cannot go stale.

The defect is TRANSIENT, which is what makes the fixture's oracle the flux through `slant` rather than
a converged field: with the refresh removed the device leaks max |phi| 7.13e-04 through the plane at
iteration 2 and still converges to the same fixed point (U 7.2e-11 against the host at 200 iterations,
6.8e-11 fixed). With it, 1.96e-19 -- real OpenFOAM reads 8.97e-17 on the same faces. The device arm now
tracks the host to U 6.8e-11 / p 3.0e-13 and inherits the host's 8.2e-06 offset from OpenFOAM exactly
(8.1874e-06). Used by sym_vs_openfoam.sh, four arms.
