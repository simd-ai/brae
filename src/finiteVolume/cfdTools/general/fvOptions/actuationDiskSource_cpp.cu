#include "actuationDiskSource_cpp.cuh"

#include <cmath>

namespace brae {
namespace cpu {

ActuationDiskState addSupFroude(
    const FvOptionsData::ActuationDisk& d,
    const std::vector<vector>&          U,
    const std::vector<scalar>&          V,
    std::vector<vector>&                source)
{
    ActuationDiskState st;
    if (d.monitorCells.empty() || d.diskCells.empty()) return st;

    // Uref: the mean over the monitor cells. rhoRef is 1 on the incompressible path -- OF sums rho over
    // the same cells and divides by the same count, which for a geometricOneField is exactly 1.
    vector Uref{0, 0, 0};
    for (label c : d.monitorCells) Uref = Uref + U[c];
    Uref = Uref / static_cast<scalar>(d.monitorCells.size());
    st.Uref = Uref;
    st.a = d.a;

    // The disk's total volume, which distributes the thrust by cell volume.
    scalar vtot = 0;
    for (label c : d.diskCells) vtot += V[c];
    if (!(vtot > 0)) return st;

    const scalar UdotN = dot(Uref, d.diskDir);
    const scalar T = 2.0 * d.area * (UdotN * UdotN) * d.a * (1.0 - d.a);
    st.T = T;

    for (label c : d.diskCells)
    {
        source[c] = source[c] + ((V[c] / vtot) * T) * d.diskDir;
    }
    return st;
}

} // namespace cpu
} // namespace brae
