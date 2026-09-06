#include "actuation_disk.cuh"
#include "device_blas.cuh"

namespace brae {

DeviceActuationDisk buildDeviceActuationDisk(
    const FvOptionsData&        fvo,
    const DeviceBuffer<scalar>& V,
    const label                 nCells)
{
    DeviceActuationDisk ad;
    if (!fvo.adActive || fvo.adDisks.empty()) return ad;

    ad.disks.resize(fvo.adDisks.size());
    for (std::size_t di = 0; di < fvo.adDisks.size(); ++di)
    {
        const FvOptionsData::ActuationDisk& src = fvo.adDisks[di];
        DeviceActuationDisk::Disk&          d   = ad.disks[di];

        d.diskDir = src.diskDir;
        d.area    = src.area;
        d.a       = src.a;

        std::vector<scalar> onDisk(nCells, 0.0);
        for (const label c : src.diskCells)
        {
            if (c >= 0 && c < nCells) onDisk[c] = 1.0;
        }
        DeviceBuffer<scalar> mask;
        mask.copyFrom(onDisk);
        // maskVDisk holds V, not 1: the thrust is distributed BY VOLUME, and OF's V() is this disk's own
        // total, not the mesh's -- two turbines on one mesh have two different denominators.
        deviceHadamard(d.maskVDisk, V, mask);
        d.vtot = deviceSumMag(d.maskVDisk);

        std::vector<scalar> onMon(nCells, 0.0);
        for (const label c : src.monitorCells)
        {
            if (c >= 0 && c < nCells) onMon[c] = 1.0;
        }
        d.monMask01.copyFrom(onMon);
        d.nmon = static_cast<scalar>(src.monitorCells.size());
    }
    ad.active = true;
    return ad;
}

void deviceActuationDiskAddSup(
    const DeviceActuationDisk&        ad,
    const DeviceBuffer<scalar>&       Ux,
    const DeviceBuffer<scalar>&       Uy,
    const DeviceBuffer<scalar>&       Uz,
    DeviceBuffer<scalar>*             source[3],
    std::vector<ActuationDiskReport>* report)
{
    if (!ad.active) return;
    if (report) report->assign(ad.disks.size(), ActuationDiskReport{});

    for (std::size_t di = 0; di < ad.disks.size(); ++di)
    {
        const DeviceActuationDisk::Disk& d = ad.disks[di];
        if (!(d.nmon > 0) || !(d.vtot > 0)) continue;

        const vector Uref
        {
            deviceDot(Ux, d.monMask01) / d.nmon,
            deviceDot(Uy, d.monMask01) / d.nmon,
            deviceDot(Uz, d.monMask01) / d.nmon
        };
        const scalar UdotN = Uref.x*d.diskDir.x + Uref.y*d.diskDir.y + Uref.z*d.diskDir.z;
        // rhoRef is 1 on the incompressible path: OF sums geometricOneField over the monitor cells and
        // divides by their count.
        const scalar T = 2.0 * d.area * (UdotN * UdotN) * d.a * (1.0 - d.a);

        const scalar dir[3] = { d.diskDir.x, d.diskDir.y, d.diskDir.z };
        for (int k = 0; k < 3; ++k)
        {
            // THE SIGN. calcFroudeMethod writes `Usource[celli] += ((V[celli]/V())*T)*diskDir` into the
            // OPTION matrix's source, and simpleFoam then assembles `UEqn == fvOptions(U)`, which is
            // `UEqn - fvOptions(U)` (fvMatrix.C, free operator==) -- so the MOMENTUM source LOSES the
            // thrust. T is positive and diskDir points downstream, so that minus is what makes the disk
            // a turbine instead of a propeller. Adding it converges perfectly well to a wind field 22%
            // too fast, which is why this is spelt out rather than inferred.
            if (source[k]) deviceAxpy(-T * dir[k] / d.vtot, d.maskVDisk, *source[k]);
        }
        if (report)
        {
            (*report)[di].Uref = Uref;
            (*report)[di].T    = T;
        }
    }
}

} // namespace brae
