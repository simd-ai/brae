# ablBox -- the YGCJ atmospheric-profile factor

A 200-cell 2D box whose inlet carries atmBoundaryLayerInlet{Velocity,K,Epsilon} with NON-DEFAULT
C1 0.17 / C2 0.6. OpenFOAM's k and epsilon inlets carry sqrt(C1*log((z+z0)/z0) + C2)
(atmBoundaryLayer.C:238-254; omega has no factor); at the DEFAULTS C1=0, C2=1 the factor is exactly 1,
which is the only profile brae computed before the reader learned the keys -- turbineSiting, the one
other ABL case in validation/, runs at the defaults, which is precisely why the hole was invisible to
its five green gates. The BC evaluates in its constructor, so ONE OF iteration writes the per-face
oracle whatever the solve does. Used by abl_vs_openfoam.sh.
