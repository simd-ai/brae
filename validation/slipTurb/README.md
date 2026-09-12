# slipTurb -- the tilted slip channel of `slip`, with a kEpsilon closure

Same 90x30x1 block as `slip`, rotated 30 degrees so the slip walls are NOT axis-aligned, plus k /
epsilon / nut with `slip` on the walls and a RAS kEpsilon model. Re is ~200 on nu = 0.005; the physics
is irrelevant here. It exists because the defect it gates has no consumer without a turbulence closure:
the legacy driver refreshes a symmetry patch's mixed refValue once, at the top of the momentum
predictor, and the closure's grad(U) runs after the pressure corrector -- so on a tilted plane the
boundary velocity the closure reads is blended toward a value for a velocity two solves old. An
axis-aligned plane cannot show it (its normal component's refValue is identically zero).
Gate: tests/legacy_symmetry_refresh.sh (BRAE_SYM_CHECK=1 prints max|n.U_b| on the slip faces at
closure time; OpenFOAM's basicSymmetry value satisfies n.U_b = 0 exactly). 8 iterations.
