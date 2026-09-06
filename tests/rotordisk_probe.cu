// rotorDiskSource against OpenFOAM's OWN reported drag and lift.
//
// OpenFOAM's rotorDiskSource prints, every iteration it is applied:
//     Effective drag = sum(rhoRef * localForce.y)     over the disk cells
//     Effective lift = sum(rhoRef * localForce.z)
// where localForce is the (e1, e2, e3) cylindrical-frame force BEFORE the rotation back to Cartesian.
// Those two scalars are a direct, unambiguous oracle for the whole blade-element calculation: the
// coordinate frame, the blade and profile table interpolation, the tip factor and the dynamic pressure
// all feed them. Computing them from OpenFOAM's own converged U is a statement about the model alone.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fv_options.cuh"
#include "rotorDiskSource_cpp.cuh"
#include "mrf_read.cuh"   // readCellZones
#include "rotor_disk.cuh"
#include "device_buffer.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <time>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    const std::map<std::string, std::vector<label>> zones = readCellZones(caseDir + "/constant/polyMesh");
    const FvOptionsData fvo = readFvOptions(caseDir, zones, g.V(), nC, g.C());
    if (!fvo.rotor.active)
    {
        std::printf("  FAIL: no active rotorDisk read from system/fvOptions\n");
        return 1;
    }

    const cpu::RotorDisk rd = cpu::buildRotorDisk(fvo.rotor, m, g);
    std::printf("rotordisk_probe: %s/%s\n", caseDir.c_str(), t.c_str());
    std::printf("  disk cells %zu   rMax %.6f   omega %.6g rad/s   nBlades %.0f   tipEffect %.3f\n",
                rd.cells.size(), rd.rMax, rd.omega, rd.nBlades, rd.tipEffect);

    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    U.evaluateBoundary();

    // rhoRef, as OpenFOAM reads it every addSup.
    scalar rhoRef = 1.0;
    {
        const FoamDict fv = readDict(caseDir + "/system/fvOptions");
        for (const auto& s : fv.subs)
        {
            if (const FoamDict* d = &s.second)
            {
                const scalar r = d->scalarOr("rhoRef", -1.0);
                if (r > 0) rhoRef = r;
            }
        }
    }

    // Re-derive the two sums the same way OpenFOAM accumulates them, from the same force loop.
    std::vector<vector> force;
    cpu::rotorForce(rd, U.internal, force);

    scalar dragEff = 0, liftEff = 0, aoaMin = 1e30, aoaMax = -1e30;
    for (std::size_t i = 0; i < rd.cells.size(); ++i)
    {
        const label c = rd.cells[i];
        if (!(rd.radius[i] > 1e-12) || !(rd.area[i] > 1e-30)) continue;
        // localForce.y and .z are the e2 and e3 components of the Cartesian force.
        dragEff += rhoRef * dot(force[c], rd.e2[i]);
        liftEff += rhoRef * dot(force[c], rd.axis);

        const vector Uin = rd.localInflow ? U.internal[c] : rd.inletVel;
        const scalar Uaz = rd.omega * rd.radius[i] - dot(rd.e2[i], Uin);
        const scalar Uax = dot(rd.axis, Uin);
        scalar a = rd.theta0 + rd.twist[i] - std::atan2(-Uax, Uaz);
        if (a >  3.14159265358979323846) a -= 6.28318530717958647692;
        if (a < -3.14159265358979323846) a += 6.28318530717958647692;
        aoaMin = std::fmin(aoaMin, a);
        aoaMax = std::fmax(aoaMax, a);
    }
    const scalar r2d = 180.0 / 3.14159265358979323846;
    std::printf("  rhoRef %.6g\n", rhoRef);
    std::printf("  min/max(AOA)   = %.6g, %.6g\n", aoaMin * r2d, aoaMax * r2d);
    std::printf("  Effective drag = %.6f\n", dragEff);
    std::printf("  Effective lift = %.6f\n", liftEff);

    // The DEVICE force, from the same fields. brae's device rotorDisk predates any test and its header
    // describes the source sign the other way round (`relaxSrc -= force`), so this is the check that
    // says the two paths push the same direction with the same magnitude.
    {
        const DeviceRotorDisk drd = buildDeviceRotorDisk(fvo.rotor, g.C(), g.Sf(), m.owner(),
                                                         m.neighbour(), m.nInternalFaces());
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        {
            ux[c] = U.internal[c].x;
            uy[c] = U.internal[c].y;
            uz[c] = U.internal[c].z;
        }
        DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), fx, fy, fz;
        deviceRotorForce(drd, static_cast<int>(nC), dUx, dUy, dUz, fx, fy, fz);
        const std::vector<scalar> hx = fx.host(), hy = fy.host(), hz = fz.host();
        scalar mx = 0, mg = 0;
        for (label c = 0; c < nC; ++c)
        {
            mx = std::fmax(mx, mag(force[c] - vector{hx[c], hy[c], hz[c]}));
            mg = std::fmax(mg, mag(force[c]));
        }
        std::printf("  device force vs _cpp: L_inf rel %.3e\n", mg > 0 ? mx / mg : mx);
    }
    return 0;
}
