# Sourced by the files of this folder: dtchull_gate runs the named parts of the DTCHull gate through its
# engine (tests/interfoam_dtchull_vs_openfoam.sh, which holds the meshing, the staging and what is measured).
#   dtchull_gate run <profile>:<arm> ...        dtchull_gate control <profile>:<arm>:<CTL=1> ...
# At most three to a file.
dtchull_gate()
{
    local kind="$1"
    shift
    if [ "$kind" = run ]; then
        DTCHULL_RUN="$*" DTCHULL_CONTROL="" exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/interfoam_dtchull_vs_openfoam.sh"
    else
        DTCHULL_RUN="" DTCHULL_CONTROL="$*" exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/interfoam_dtchull_vs_openfoam.sh"
    fi
}
