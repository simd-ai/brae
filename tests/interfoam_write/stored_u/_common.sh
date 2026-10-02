# Shared by this folder: U's STORED patch values in the device's constrainHbyA, on RAS/DTCHull as the tutorial
# row stages it (845,536 cells, no momentum predictor, an outletPhaseMeanVelocity outlet).
# constrainHbyA assigns `U.boundaryField()[patchi]` (constrainHbyA.C:67) -- what U's last evaluate left. The
# device re-evaluated the patch from its coefficients, which the momentum assembly's updateCoeffs has already
# moved, so the outlet entered the first corrector of every step after the first with the next evaluate's
# values. THE WITNESS is the outlet's phiHbyA at iteration two's first corrector, device against host
# (build/brae_inter_taps_diff): MEASURED worst face 4.4e-16 with the stored values, 2.6e-12 with
# BRAE_CONTROL_DEVICE_HBYA_REEVALUATED=1 -- and every INTERIOR tap at 1e-14 either way, which is why no field
# comparison over two steps shows it (the row reads p_rgh 1.0e-11 against 5.9e-13).
BOUND_OUTLET_FACE=4.4e-15
CONTROL_OUTLET_FLOOR=4.4e-13
TAPS="${BUILD:-$ROOT/build}/brae_inter_taps_diff"

# outlet_face [ENV=1]: the outlet's worst face of phiHbyA, device against host, after two steps; `nan` if the
# instrument did not report it
outlet_face()
{
    [ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
    [ -x "$TAPS" ] || { echo "SKIP: $TAPS not built"; exit 77; }
    wcase DTCHull of > "$W/stored_u_stage.txt" 2>&1
    local o="$W/w_of_DTCHull"
    [ -d "$o" ] || { echo nan; return; }
    env "$@" "$TAPS" "$o" "$o/0" 2 > "$W/taps.txt" 2>&1
    python3 - "$W/taps.txt" <<'EOF_P'
import re, sys
t = open(sys.argv[1]).read()
i = t.find('phiHbyA on the boundary')
m = re.search(r'\n\s+outlet\s+sum\s+\S+\s+net difference\s+\S+\s+worst face\s+(\S+)', t[i:]) if i >= 0 else None
print(m.group(1) if m else 'nan')
EOF_P
}
within()   # within <value> <bound>: value is a number at or below bound
{
    python3 -c "import sys; a, b = float('$1'), float('$2'); sys.exit(0 if a == a and a <= b else 1)"
}
