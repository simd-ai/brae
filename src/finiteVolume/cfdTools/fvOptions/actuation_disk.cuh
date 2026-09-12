#pragma once
// fv::actuationDiskSource (Froude), device side -- the same model as the host reference in
// src/finiteVolume/cfdTools/general/fvOptions/actuationDiskSource_cpp.cuh, which carries the derivation
// and the OpenFOAM provenance. Per turbine, per iteration:
//
//   Uref = mean(U) over the monitor cells;  T = 2*area*(Uref & diskDir)^2 * a*(1-a)
//   source[c] += (V[c]/Vdisk) * T * diskDir       over the DISK cells
//
// Both terms are masked reductions/axpys over the whole field rather than gathers over the cell list,
// which is what lets one kernel-free implementation serve any number of turbines.
//
// EACH TURBINE CARRIES ITS OWN MASKS. Sharing them would apply every thrust to the union of the disks
// and converge to a plausible but wrong wind field rather than failing.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "fv_options.cuh"

#include <vector>

namespace brae {

struct DeviceActuationDisk
{
    struct Disk
    {
        vector diskDir{1, 0, 0};
        scalar area = 0, a = 0, vtot = 0, nmon = 0;
        DeviceBuffer<scalar> maskVDisk;   // V[c] on this disk's cells, 0 elsewhere
        DeviceBuffer<scalar> monMask01;   // 1 on this disk's monitor cells, 0 elsewhere
    };
    bool              active = false;
    std::vector<Disk> disks;
};

// What one turbine contributed this iteration -- the same quantities OpenFOAM writes to
// postProcessing/<name>/<t>/actuationDiskSource.dat, so a gate can read one against the other.
struct ActuationDiskReport
{
    vector Uref{0, 0, 0};
    scalar T = 0;
};

DeviceActuationDisk buildDeviceActuationDisk(
    const FvOptionsData&           fvo,
    const DeviceBuffer<scalar>&    V,
    label                          nCells);

// Adds every turbine's thrust into the three momentum sources. `report`, when non-null, is filled with
// one entry per disk in the same order.
void deviceActuationDiskAddSup(
    const DeviceActuationDisk&        ad,
    const DeviceBuffer<scalar>&       Ux,
    const DeviceBuffer<scalar>&       Uy,
    const DeviceBuffer<scalar>&       Uz,
    DeviceBuffer<scalar>*             source[3],
    std::vector<ActuationDiskReport>* report = nullptr);

} // namespace brae
