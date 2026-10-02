#!/usr/bin/env bash
# The write gate: three wave tutorials whose bounds sit above 1e-9, against OpenFOAM's own sensitivity.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
# MEASURED (2026-10-02), worst field file, brae host / brae device / OpenFOAM against its twin:
#   streamFunction   2.5e-09 / 2.7e-09 / 4.5e-05
#   stokesII         1.3e-09 / 1.5e-09 / 1.0e-04
#   cnoidal          3.6e-10 / 4.2e-10 / 1.4e-05
for key in streamFunction stokesII cnoidal; do
    twin_check $key
done
finish "the wave tutorials: brae inside OpenFOAM's own sensitivity"
