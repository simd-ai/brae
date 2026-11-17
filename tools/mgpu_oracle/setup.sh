#!/bin/bash
# One-time setup for the oracle: unpack the OpenFOAM v2606 SOURCE next to the installed v2606 binaries
# (/usr/lib/openfoam/openfoam2606 ships binaries+libs only: no headers, no wmake) and build tools/dumpInterFoam
# against the installed libraries.  Idempotent.  Everything lands under $ORACLE_ROOT (not in $HOME).
#   ORACLE_ROOT   default /tmp/brae-oracle     (OpenFOAM source tree + built dump tool)
#   OF_TGZ        default /tmp/OpenFOAM-v2606.tgz              (ESI OpenFOAM-v2606 source release)
#   OF_INSTALL    default /usr/lib/openfoam/openfoam2606
set -eo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ORACLE_ROOT=${ORACLE_ROOT:-/tmp/brae-oracle}
OF_TGZ=${OF_TGZ:-/tmp/OpenFOAM-v2606.tgz}
OF_INSTALL=${OF_INSTALL:-/usr/lib/openfoam/openfoam2606}
PLAT=linux64GccDPInt32Opt
SRC=$ORACLE_ROOT/OpenFOAM-v2606
mkdir -p "$ORACLE_ROOT/bin"
if [ ! -d "$SRC/src" ]; then
    [ -f "$OF_TGZ" ] || { echo "need the OpenFOAM-v2606 source tarball at $OF_TGZ" >&2; exit 1; }
    tar xzf "$OF_TGZ" -C "$ORACLE_ROOT"
fi
[ -d "$OF_INSTALL/platforms/$PLAT" ] || { echo "no installed v2606 binaries at $OF_INSTALL/platforms/$PLAT" >&2; exit 1; }
mkdir -p "$SRC/platforms"
ln -sfn "$OF_INSTALL/platforms/$PLAT" "$SRC/platforms/$PLAT"
cat > "$ORACLE_ROOT/of_env.sh" <<EOT
# source me: OpenFOAM-v2606 source tree (wmake, headers) + installed v2606 binaries/libs + system OpenMPI
export WM_COMPILER_TYPE=system WM_COMPILER=Gcc WM_MPLIB=SYSTEMOPENMPI
set +u
source $SRC/etc/bashrc >/dev/null 2>&1 || true
# bashrc points the user dirs into \$HOME/OpenFOAM: override AFTER it
export FOAM_USER_APPBIN=$ORACLE_ROOT/bin FOAM_USER_LIBBIN=$ORACLE_ROOT/lib
export PATH=\$FOAM_USER_APPBIN:\$PATH
command -v mpirun >/dev/null || export PATH=/usr/lib64/openmpi/bin:\$PATH   # SYSTEMOPENMPI: Fedora keeps it off PATH
EOT
source "$ORACLE_ROOT/of_env.sh"
mkdir -p "$FOAM_USER_APPBIN"
[ -x "$SRC/platforms/tools/linux64Gcc/wmkdepend" ] || (cd "$SRC/wmake/src" && make >/dev/null)
# headers: lnInclude for every library (symlink farms, seconds)
if [ ! -d "$SRC/src/finiteVolume/lnInclude" ]; then
    for d in $(cd "$SRC" && find src applications/solvers/multiphase -path '*/Make/files' | sed 's|/Make/files||'); do
        wmakeLnInclude -u "$SRC/$d" > /dev/null 2>&1 || true
    done
    wmakeLnInclude -u "$SRC/src/TurbulenceModels/phaseIncompressible" > /dev/null 2>&1 || true
fi
# the dump tool is built in a scratch copy so that Make/<platform> never lands in the repo
rm -rf "$ORACLE_ROOT/dumpInterFoam"; cp -r "$HERE/../dumpInterFoam" "$ORACLE_ROOT/dumpInterFoam"
(cd "$ORACLE_ROOT/dumpInterFoam" && wmake > "$ORACLE_ROOT/dumpInterFoam.build.log" 2>&1) \
    || { tail -30 "$ORACLE_ROOT/dumpInterFoam.build.log"; exit 1; }
echo "ok: $FOAM_USER_APPBIN/dumpInterFoam   (source $ORACLE_ROOT/of_env.sh)"
