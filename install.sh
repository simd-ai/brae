#!/bin/sh
# brae installer  ,  clones, builds, and installs the brae CFD engine.
#
#   curl -fsSL https://brae.sh/install.sh | sh
#
# Optional environment overrides:
#   BRAE_DIR        where to clone the source     (default: $HOME/brae)
#   BRAE_REF        branch or tag to build        (default: main)
#   BRAE_CUDA_ARCH  GPU compute capability, e.g. 90, 121, or "native"
#                   (default: auto-detected from nvidia-smi)
#   BRAE_INSTALL_CUDA   1 = auto-install the CUDA toolkit (nvcc) if missing,
#                       via the NVIDIA apt repo (Ubuntu/Debian). Installs the
#                       toolkit only, never the GPU driver. (default: off)
#   BRAE_CUDA_TOOLKIT_PKG  toolkit package to install (default: cuda-toolkit-12-6)
#   PREFIX          install prefix for the binary (default: $HOME/.local)
set -eu

REPO="${BRAE_REPO:-https://github.com/simd-ai/brae.git}"
REF="${BRAE_REF:-main}"
DIR="${BRAE_DIR:-$HOME/brae}"
PREFIX="${PREFIX:-$HOME/.local}"
BINDIR="$PREFIX/bin"

# ---- logging ----
if [ -t 1 ]; then
    B=$(printf '\033[1m'); G=$(printf '\033[32m'); Y=$(printf '\033[33m'); R=$(printf '\033[31m'); N=$(printf '\033[0m')
else
    B=''; G=''; Y=''; R=''; N=''
fi
say()  { printf '%s==>%s %s\n' "$G$B" "$N" "$*"; }
warn() { printf '%s!! %s%s\n'  "$Y$B" "$*" "$N" >&2; }
die()  { printf '%serror:%s %s\n' "$R$B" "$N" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# use sudo when not root (auto-install of CUDA needs apt/dpkg privileges)
SUDO=''
if [ "$(id -u)" -ne 0 ] && have sudo; then SUDO="sudo"; fi

# NVIDIA CUDA apt-repo tags for this OS / CPU arch (empty if unsupported for auto-install)
cuda_repo_id() {
    [ -r /etc/os-release ] || return 0
    . /etc/os-release
    case "${ID:-}${VERSION_ID:-}" in
        ubuntu24.04) echo ubuntu2404 ;;
        ubuntu22.04) echo ubuntu2204 ;;
        ubuntu20.04) echo ubuntu2004 ;;
        debian12)    echo debian12  ;;
        debian11)    echo debian11  ;;
        *)           echo ''        ;;
    esac
}
cuda_repo_arch() {
    case "$(uname -m)" in
        x86_64)  echo x86_64 ;;   # H100 / datacenter x86 hosts
        aarch64) echo sbsa   ;;   # Grace / GB10 and other server ARM
        *)       echo ''     ;;
    esac
}
# opt-in (BRAE_INSTALL_CUDA=1): install just the toolkit (nvcc), NOT the `cuda` metapackage,
# so the host's existing GPU driver is left untouched. Returns non-zero on any failure.
install_cuda_toolkit() {
    distro=$(cuda_repo_id); carch=$(cuda_repo_arch)
    pkg="${BRAE_CUDA_TOOLKIT_PKG:-cuda-toolkit-12-6}"
    if [ -z "$distro" ] || [ -z "$carch" ]; then
        warn "CUDA auto-install unsupported on this OS/arch; use the manual steps below"; return 1
    fi
    if ! have wget && ! have curl; then warn "need wget or curl to fetch the CUDA repo key"; return 1; fi
    say "installing CUDA toolkit ($pkg) from the NVIDIA repo ($distro/$carch) - downloads a few GB"
    tmp=$(mktemp -d); key="cuda-keyring_1.1-1_all.deb"
    url="https://developer.download.nvidia.com/compute/cuda/repos/$distro/$carch/$key"
    if have wget; then wget -qO "$tmp/$key" "$url" || { warn "download failed: $url"; return 1; }
    else curl -fsSL -o "$tmp/$key" "$url" || { warn "download failed: $url"; return 1; }
    fi
    $SUDO dpkg -i "$tmp/$key"       || return 1
    $SUDO apt-get update            || return 1
    $SUDO apt-get install -y "$pkg" || return 1
    rm -rf "$tmp"
    [ -x /usr/local/cuda/bin/nvcc ] && { PATH="/usr/local/cuda/bin:$PATH"; export PATH; }
    have nvcc
}

say "installing brae"

# ---- CUDA on PATH (it often lives in /usr/local/cuda/bin) ----
if ! have nvcc && [ -x /usr/local/cuda/bin/nvcc ]; then
    PATH="/usr/local/cuda/bin:$PATH"; export PATH
fi

# ---- CUDA toolkit: optional auto-install (opt-in via BRAE_INSTALL_CUDA=1) ----
if ! have nvcc && [ "${BRAE_INSTALL_CUDA:-0}" = "1" ]; then
    install_cuda_toolkit || warn "CUDA auto-install did not complete; see the manual steps below"
fi

# ---- required tools ----
missing=''
for t in git cmake nvcc; do have "$t" || missing="$missing $t"; done
have c++ || have g++ || missing="$missing c++"
have make || have ninja || missing="$missing make"
if [ -n "$missing" ]; then
    distro=$(cuda_repo_id); carch=$(cuda_repo_arch)
    if [ -n "$distro" ] && [ -n "$carch" ]; then
        cuda_hint="re-run with  BRAE_INSTALL_CUDA=1  to auto-install, or do it manually:
     wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$carch/cuda-keyring_1.1-1_all.deb
     ${SUDO:+$SUDO }dpkg -i cuda-keyring_1.1-1_all.deb && ${SUDO:+$SUDO }apt-get update && ${SUDO:+$SUDO }apt-get install -y cuda-toolkit-12-6
     echo 'export PATH=/usr/local/cuda/bin:\$PATH' >> ~/.bashrc && . ~/.bashrc"
    else
        cuda_hint="https://developer.nvidia.com/cuda-downloads"
    fi
    die "missing required tools:$missing
  base tools (Ubuntu/Debian):  ${SUDO:+$SUDO }apt-get install -y git cmake build-essential
  CUDA toolkit (nvcc) 12.4+ (13.x recommended):
     $cuda_hint"
fi

# ---- cmake >= 3.24 ----
cmake_ver=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n1)
if [ "$(printf '3.24\n%s\n' "$cmake_ver" | sort -V 2>/dev/null | head -n1)" != "3.24" ]; then
    die "cmake $cmake_ver found, brae needs >= 3.24"
fi

# ---- clone or update ----
if [ -d "$DIR/.git" ]; then
    say "updating existing checkout in $DIR"
    git -C "$DIR" fetch --depth 1 origin "$REF"
    git -C "$DIR" reset --hard FETCH_HEAD
else
    say "cloning $REPO ($REF) into $DIR"
    git clone --depth 1 --branch "$REF" "$REPO" "$DIR"
fi

# ---- detect target GPU arch ----
arch="${BRAE_CUDA_ARCH:-}"
if [ -z "$arch" ] && have nvidia-smi; then
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
fi
[ -n "$arch" ] || arch="native"
say "target GPU arch: $arch"

# ---- configure + build ----
cd "$DIR"
say "configuring"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" || die \
"cmake configure failed, usually a missing library.
  On Ubuntu/Debian:  sudo apt-get install -y libopenmpi-dev openmpi-bin libscotch-dev zlib1g-dev"

# ---- which executables must exist next to `brae` ----
# Taken from the dispatch registry itself rather than written out a second time here. `brae` is the only
# command a user types; a case naming `rhoSimpleFoam` in controlDict is exec'd into the sibling binary
# brae_rhoSimpleFoam (solver_dispatch.cuh execSibling), so every executable in that table has to be built
# AND installed or the hand-over dies with "cannot start 'brae_rhoSimpleFoam'".
#
# It did. This list used to read `for b in brae brae_pimpleFoam`, and brae_rhoSimpleFoam -- a registered
# solver since the OF-mirror port -- was built (add_dependencies at CMakeLists.txt:1750) and then left in
# build/. A fresh install could run no compressible case at all. Deriving the list means the next solver
# row cannot repeat it.
SOLVERS=$(sed -n 's/.*{"[A-Za-z]*", *"\(brae[A-Za-z_]*\)".*/\1/p' \
          src/applications/solvers/common/solver_dispatch.cuh | sort -u)
[ -n "$SOLVERS" ] || SOLVERS="brae brae_pimpleFoam brae_rhoSimpleFoam brae_interFoam"   # registry unreadable: known set
say "solvers to build: $(echo $SOLVERS | tr '\n' ' ')"

jobs=$(nproc 2>/dev/null || echo 4)
say "building brae with $jobs jobs (this can take a few minutes)"
# Every solver target by name. `brae` alone happens to pull the others in through add_dependencies today,
# but that is a fact about the current CMakeLists, not a promise, and this install depends on all of them.
cmake --build build -j "$jobs" --target $SOLVERS || die "build failed"

# ---- install the binaries ----
mkdir -p "$BINDIR"
for b in $SOLVERS; do
    [ -f "build/$b" ] || die "build/$b is missing after a successful build.
  It is in the solver registry (src/applications/solvers/common/solver_dispatch.cuh), so `brae` will try
  to hand cases to it and fail. Check that CMakeLists.txt still defines the target."
    install -m 0755 "build/$b" "$BINDIR/$b" 2>/dev/null || cp "build/$b" "$BINDIR/$b"
    say "installed  $BINDIR/$b"
done

# brae-agent is what `brae node ...` execs. It is built only when libcurl's headers are present, and its
# absence must not fail an install of the solver -- someone building brae to run simulations does not need it.
if [ -f build/brae-agent ]; then
    install -m 0755 build/brae-agent "$BINDIR/brae-agent" 2>/dev/null || cp build/brae-agent "$BINDIR/brae-agent"
    say "installed  $BINDIR/brae-agent"
else
    warn "brae-agent was not built (libcurl headers missing), so \`brae node\` will not work.
      Install libcurl4-openssl-dev and rebuild if you want to join the Brae network."
fi

# ---- PATH hint ----
case ":$PATH:" in
    *":$BINDIR:"*) ;;
    *) warn "$BINDIR is not on your PATH. Add it with:"
       printf '      echo '\''export PATH="%s:$PATH"'\'' >> ~/.profile && . ~/.profile\n' "$BINDIR" >&2 ;;
esac

printf '\n%s%sbrae is installed.%s  Run it inside any OpenFOAM case, steady or transient:\n\n' "$G" "$B" "$N"
printf '    cd yourCase && brae\n\n'
printf '  docs: https://brae.sh   source: %s\n' "$DIR"
