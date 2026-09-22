// The DEVICE's interface normal flux ON A CYCLIC PAIR against REAL OpenFOAM's own nHatf.
//
// interfaceProperties.C:134-150 is three lines: gradAlphaf = fvc::interpolate(grad(alpha1)), nHatfv =
// gradAlphaf/(mag(gradAlphaf) + deltaN), nHatf = nHatfv & Sf. On a COUPLED patch that interpolation is
// the two CELLS' gradients weighted (surfaceInterpolationScheme.C:284-293), and correctContactAngle
// never reaches a cyclic -- it walks the alphaContactAngle patches, and a periodic pair is not one.
//
// THE ORACLE IS OpenFOAM'S OWN FIELD, not the host arm: `nHatf` is registered in the objectRegistry, so
// a writeObjects function object writes it with the rest of the time directory, cyclic patches
// included. The device is handed OpenFOAM's alpha at that instant and has to reproduce the number
// OpenFOAM wrote on those faces.
//
// THE CONTROL is the same computation with the pair left out of grad(alpha) -- which is what the device
// did before this unit, and what a mesh with a wall in place of the pair would give. It has to differ,
// or this gate would pass whether or not the interface contribution is there.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "cyclic_interface.cuh"
#include "device_mesh.cuh"
#include "device_cyclic.cuh"
#include "device_interface_properties.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}
}   // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("  SKIP: usage: %s <caseDir> <ofTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string timeDir = argv[2];
    std::printf("== the device's nHatf on a cyclic pair against OpenFOAM's own ==\n");

    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    attachCyclicCoupling(fvp, m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);

    const label nC = m.nCells();
    std::size_t nCoupled = 0;
    for (const FvPatch& q : fvp) if (q.coupled) nCoupled += static_cast<std::size_t>(q.size);
    std::printf("  mesh: %d cells, %zu coupled faces\n", (int)nC, nCoupled);
    check("createBaffles left a coupled pair in the mesh", nCoupled > 0);
    if (!nCoupled) { std::printf("test_device_cyclic_nhatf_vs_openfoam: %d failures\n", failures); return 1; }

    // OpenFOAM's OWN alpha at this instant, cells and patch values
    const FieldData<scalar> ofAlpha = readField<scalar>(timeDir + "/alpha.water");
    check("OpenFOAM's alpha has one value per cell",
          ofAlpha.internalField.size() == static_cast<std::size_t>(nC));
    if (ofAlpha.internalField.size() != static_cast<std::size_t>(nC))
    { std::printf("test_device_cyclic_nhatf_vs_openfoam: %d failures\n", failures); return 1; }

    // ...and its nHatf, which is what this gate is against
    const FieldData<scalar> ofNHatf = readField<scalar>(timeDir + "/nHatf");

    // the device's boundary array skips the coupled patches, exactly as the device mesh does
    std::vector<scalar> alphaBnd;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (isCoupledInterfaceType(fvp[pi].type)) continue;
        const PatchFieldData<scalar>* b = nullptr;
        for (const PatchFieldData<scalar>& e : ofAlpha.boundary)
        {
            if (e.name == fvp[pi].name) { b = &e; break; }
        }
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            // A patch OpenFOAM wrote WITHOUT values -- a zeroGradient wall -- holds its face cell's
            // alpha, not zero. Feeding zero there is a wrong gradient in every wall-adjacent cell, and
            // the pair's bottom face is one: MEASURED, that one face read 8.0e-14 against OpenFOAM's
            // 7.8e-06 while the other twelve agreed to 1e-10, because the baffle stands on lowerWall.
            const scalar cellValue = ofAlpha.internalField[static_cast<std::size_t>(fvp[pi].faceCells[i])];
            if (!b) { alphaBnd.push_back(cellValue); continue; }
            if (b->valueUniform) { alphaBnd.push_back(b->uniformValue); continue; }
            alphaBnd.push_back(static_cast<std::size_t>(i) < b->values.size()
                               ? b->values[static_cast<std::size_t>(i)] : cellValue);
        }
    }

    // deltaN, interfaceProperties.C:192-195 -- 1e-8/cbrt(average(V))
    scalar sumV = 0;
    for (label c = 0; c < nC; ++c) sumV += g.V()[c];
    const scalar deltaN = scalar(1e-8)/std::cbrt(sumV/static_cast<scalar>(nC));
    std::printf("  deltaN = %.6e\n", (double)deltaN);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);
    DeviceBuffer<scalar> dAlpha(ofAlpha.internalField), dAlphaBnd(alphaBnd);

    // the device's nHatf on the pair, WITH the pair in grad(alpha)
    DeviceBuffer<scalar> gx, gy, gz, nHatfIf;
    deviceGaussGrad(dm, dAlpha, dAlphaBnd, gx, gy, gz);
    deviceCyclicAddGrad(cyc, dAlpha, dm.V, gx, gy, gz);
    deviceInterfaceNormalFluxCyclic(cyc, gx, gy, gz, deltaN, nHatfIf);
    std::vector<scalar> dev;
    nHatfIf.copyTo(dev);

    // ...and the CONTROL, without it
    DeviceBuffer<scalar> cgx, cgy, cgz, ctrlIf;
    deviceGaussGrad(dm, dAlpha, dAlphaBnd, cgx, cgy, cgz);
    deviceInterfaceNormalFluxCyclic(cyc, cgx, cgy, cgz, deltaN, ctrlIf);
    std::vector<scalar> ctrl;
    ctrlIf.copyTo(ctrl);

    // OpenFOAM's values on the same faces, in buildDeviceCyclic's order
    std::vector<scalar> of;
    for (const CyclicInterface& c : cyclics)
    {
        const PatchFieldData<scalar>* b = nullptr;
        for (const PatchFieldData<scalar>& e : ofNHatf.boundary)
        {
            if (e.name == fvp[c.patch].name) { b = &e; break; }
        }
        check("OpenFOAM wrote nHatf on this coupled patch", b != nullptr && !b->values.empty());
        if (!b) break;
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            of.push_back(i < b->values.size() ? b->values[i] : scalar(0));
        }
    }

    // IS THE PAIR SPECIAL? The same field's INTERNAL faces are an independent reading of the same
    // inputs: OpenFOAM writes nHatf's internalField too, and the device computes those with the kernel
    // that was already gated. If the interior carries the same level, the number below is what this
    // comparison costs -- the alpha OpenFOAM wrote, its geometry against brae's -- and not the pair.
    scalar interior = 0, interiorScale = 0;
    {
        DeviceBuffer<scalar> nHatfInt;
        deviceInterfaceNormalFlux(dm, dm.nInternalFaces, gx, gy, gz, deltaN, nHatfInt);
        std::vector<scalar> di;
        nHatfInt.copyTo(di);
        for (std::size_t f = 0; f < di.size() && f < ofNHatf.internalField.size(); ++f)
        {
            interior = std::fmax(interior, std::fabs(di[f] - ofNHatf.internalField[f]));
            interiorScale = std::fmax(interiorScale, std::fabs(ofNHatf.internalField[f]));
        }
        std::printf("  the INTERIOR of the same field: device vs OpenFOAM %.4e (|nHatf| up to %.4e)\n",
                    (double)interior, (double)interiorScale);
    }

    scalar worst = 0, scale = 0, worstCtrl = 0;
    for (std::size_t j = 0; j < dev.size() && j < of.size(); ++j)
    {
        worst = std::fmax(worst, std::fabs(dev[j] - of[j]));
        worstCtrl = std::fmax(worstCtrl, std::fabs(ctrl[j] - of[j]));
        scale = std::fmax(scale, std::fabs(of[j]));
    }
    std::printf("  nHatf on %zu pair faces: device vs OpenFOAM %.4e, |nHatf| up to %.4e\n",
                dev.size(), (double)worst, (double)scale);
    std::printf("  CONTROL, the pair left out of grad(alpha): %.4e\n", (double)worstCtrl);

    // THE BOUND IS THE FIELD'S OWN INTERIOR, not a number picked to fit. nHatf is a normalised gradient
    // dotted with Sf, so what the comparison can reach is set by the alpha OpenFOAM wrote and by brae's
    // geometry against OpenFOAM's -- MEASURED, the interior of this very field reads 1.4e-12 of a
    // 2.1e-04 scale. The pair has to be as good as that, and it is (3.0e-12 of 4.1e-05, a ratio of 2.2).
    // At writePrecision 15 the same two numbers were 20x worse; 17 digits is what a double round-trips.
    check("the device's nHatf on the pair is as close to OpenFOAM's as the field's own interior is",
          dev.size() == of.size() && worst <= scalar(4)*std::fmax(interior, scalar(1e-300)));
    check("OpenFOAM's nHatf is not identically zero there, so the comparison means something",
          scale > scalar(0));
    check("...and leaving the pair out of the gradient is a DIFFERENT answer",
          worstCtrl > scalar(100)*std::fmax(worst, scalar(1e-300)));

    std::printf("test_device_cyclic_nhatf_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
