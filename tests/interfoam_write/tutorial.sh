#!/usr/bin/env bash
# arm W of the write gate for ONE tutorial: tutorial.sh <key> [host|device], a row of W_CASES in lib.sh. One
# ctest test per row -- and per arm on the tutorials whose meshes are large, where the host run and the
# device run are each minutes -- so a tutorial is re-checked alone and the rows run side by side.
. "$(dirname "$0")/lib.sh"
[ $# -ge 1 ] && [ $# -le 2 ] || { echo "usage: tutorial.sh <tutorial key> [host|device]"; exit 2; }
if [ $# -eq 2 ]; then
    [ "$2" = device ] && [ $GPU -ne 1 ] && { echo "SKIP: no GPU for the device arm"; exit 77; }
    wcase "$1" "$2" || true
else
    wcase "$1" || true
fi
finish "arm W: $1 written as OpenFOAM writes it"
