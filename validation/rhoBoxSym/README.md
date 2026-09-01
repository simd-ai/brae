# rhoBoxSym -- a TILTED symmetry plane, compressible

rhoBox's duct with the top replaced by a symmetryPlane whose normal is ~4 degrees off-axis (top edge
rises 0.2 -> 0.35 over the 2 m length). Axis-aligned symmetry can never discriminate a symmetry
defect on a segregated solver -- with n along one axis the per-component treatment (vf = |n_k|) is
exact and a stale snapshot equals a fresh one, which is why the missing deviceUpdateSymmetry/Wedge
calls sat unwired on the rho device driver with nothing able to notice. The tilt couples Ux and Uy.

Measured here: the HOST mirror lands U 8.2e-06 / T 4.7e-09 / p 1.3e-10 from OpenFOAM's converged
state (its live SymmetryPatchField is exact); the DEVICE arm drifts from iteration 1 even with the
update calls wired (Uy 2.8e-05 -> 9.5e-01 by iteration 6) -- its segregated model is exact only
axis-aligned, so the device REFUSES a tilted symmetry by name and this fixture's cuda arm asserts
that refusal. Used by sym_vs_openfoam.sh.
