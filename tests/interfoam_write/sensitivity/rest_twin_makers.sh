#!/usr/bin/env bash
# The write gate: a solitary wave and two wave makers whose bounds sit above 1e-9, against OpenFOAM's own
# sensitivity.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
# MEASURED (2026-10-02), worst field file, brae host / brae device / OpenFOAM against its twin:
#   solitaryMcCowan   1.0e-08 / 1.5e-08 / 9.0e-07
#   waveMakerFlap     4.1e-10 / 8.9e-10 / 6.7e-08
#   waveMakerPiston   1.3e-09 / 1.8e-09 / 2.1e-07
for key in solitaryMcCowan waveMakerFlap waveMakerPiston; do
    twin_check $key
done
finish "the solitary wave and the wave makers: brae inside OpenFOAM's own sensitivity"
