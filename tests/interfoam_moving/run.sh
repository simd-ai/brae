# Sourced by the per-profile tests in this folder's sub-folders. moving_gate <profile>:<control> ...
# stages each profile and its control and gates the profiles alone, through the gate's own engine
# (tests/interfoam_moving_vs_openfoam.sh, which holds the staging and what every profile measures).
# At most three profiles to a file: a test is then one mesh motion, not a family of them.
moving_gate()
{
    local only="" gate="" pc
    for pc in "$@"; do
        only="$only ${pc%%:*} ${pc#*:}"
        gate="$gate ${pc%%:*}"
    done
    MOVING_ONLY="$only" MOVING_GATE="$gate" MOVING_PART="${MOVING_PART:-}" exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/interfoam_moving_vs_openfoam.sh"
}

# moving_profile / moving_control: the same, for a profile whose control is itself a full run of both arms --
# the profile in one file and its control in another, so neither is a ten-minute test
moving_profile()
{
    MOVING_PART=gate moving_gate "$@"
}
moving_control()
{
    MOVING_PART=control moving_gate "$@"
}
