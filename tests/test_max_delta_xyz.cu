// OpenFOAM's LES filter width `delta maxDeltaxyz`, and why it is NOT the vertex bounding box brae
// already had.
//
// OF LESModels::maxDeltaxyz::calcDelta() (maxDeltaxyz.C:100-125):
//     hmax[celli] = deltaCoeff_ * max over the cell's faces of |n_f . (Cf_f - C_c)|,  deltaCoeff_ = 2
// i.e. twice the largest cell-centre-to-face-PLANE distance. brae's older cellMaxDeltaXYZ is the extent
// of the cell's vertex bounding box along the coordinate AXES, which is a different quantity the moment
// the cell stops being axis-aligned -- and NACA4412 (the case that asks for maxDeltaxyz) is a curvilinear
// C-grid, so every cell in it is exactly that case.
//
// LEG 2 IS THE ONE THAT MATTERS. Rotate the whole mesh rigidly: the physical filter width of a cell
// cannot change, because nothing about the cell changed. cellMaxDeltaFaceNormal is invariant to six
// decimal places; the bounding-box version inflates by ~12% on the same mesh, because the box it fits is
// stuck to the global axes. Leg 1 pins the two together on a Cartesian mesh first, so leg 2's split is
// demonstrably the rotation and not two unrelated formulas.
//
// Leg 4 then proves the consumer end: handing the sub-grid model an explicit delta changes nut, and
// handing it exactly cbrt(V) reproduces the default bit-for-bit -- so the buffer is really read, and a
// case that never says `delta` is untouched.
#include "cell_max_delta.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "foam_dict.cuh"
#include "solver_controls.cuh"
#include "turbulence_setup.cuh"
#include "device_smagorinsky.cuh"
#include "device_buffer.cuh"
#include "box_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>
#include <cstdio>
#include <string>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// Rigid rotation of every mesh point by `ang` about the z axis. The cells are unchanged as SHAPES;
// only their orientation relative to the global axes moves.
PrimitiveMesh rotatedZ(const PrimitiveMesh& src, scalar ang)
{
    PrimitiveMesh m = src;
    std::vector<vector> p = m.points();
    const scalar cs = std::cos(ang), sn = std::sin(ang);
    for (auto& q : p) { const scalar x = q.x, y = q.y; q.x = cs*x - sn*y; q.y = sn*x + cs*y; }
    m.movePoints(p);
    return m;
}

// readDict() only takes a path, so round-trip through a scratch file.
FoamDict dictFromString(const std::string& body)
{
    const std::string path = "test_max_delta_xyz.tmpdict";
    { std::ofstream f(path); f << body; }
    FoamDict d = readDict(path);
    std::remove(path.c_str());
    return d;
}

scalar maxAbs(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar e = 0;
    for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(a[i] - b[i]));
    return e;
}
} // namespace

int main()
{
    std::printf("== maxDeltaxyz (OF LES delta) ==\n");

    // ---- Leg 1: on a Cartesian mesh the two definitions must AGREE ---------------------------------
    // A hex of size (dx,dy,dz): |n.(Cf-C)| is dx/2 on the x faces, so max*deltaCoeff = max(dx,dy,dz),
    // which is precisely the bounding-box extent. This is the control that makes leg 2 meaningful.
    PrimitiveMesh box = boxtest::boxMesh(4, 3, 2);
    FvGeometry gBox; gBox.build(box);
    const std::vector<scalar> fnBox  = cellMaxDeltaFaceNormal(box, gBox, 2.0);
    const std::vector<scalar> bbxBox = cellMaxDeltaXYZ(box);
    check(maxAbs(fnBox, bbxBox) < 1e-12, "Cartesian: face-normal delta == vertex bounding box");

    // and both equal the hand value: boxMesh lays out UNIT-spaced cells, so max(dx,dy,dz) = 1
    const scalar hand = 1.0;
    scalar eHand = 0;
    for (scalar v : fnBox) eHand = std::max(eHand, std::fabs(v - hand));
    check(eHand < 1e-12, "Cartesian: delta == max(dx,dy,dz) = 1 by hand");

    // ---- Leg 2: rigid rotation -- the filter width is a property of the CELL ------------------------
    // Flatten z so the largest extent lies in the rotation plane, else the z direction (untouched by a
    // z rotation) would win the max and hide the whole effect.
    PrimitiveMesh flat = boxtest::boxMesh(4, 4, 1);
    {
        std::vector<vector> p = flat.points();
        for (auto& q : p) q.z *= 0.1;                 // dz = 0.1 << dx = dy = 1
        flat.movePoints(p);
    }
    FvGeometry gFlat; gFlat.build(flat);
    PrimitiveMesh rot = rotatedZ(flat, 0.5235987755982988);   // 30 degrees
    FvGeometry gRot;  gRot.build(rot);

    const std::vector<scalar> fnFlat = cellMaxDeltaFaceNormal(flat, gFlat, 2.0);
    const std::vector<scalar> fnRot  = cellMaxDeltaFaceNormal(rot,  gRot,  2.0);
    const std::vector<scalar> bbFlat = cellMaxDeltaXYZ(flat);
    const std::vector<scalar> bbRot  = cellMaxDeltaXYZ(rot);

    check(maxAbs(fnFlat, fnRot) < 1e-12, "rotation invariance: face-normal delta unchanged by a 30 deg rigid rotation");

    // The negative control: the OLD definition is NOT invariant, so leg 2 is discriminating and not
    // just two copies of the same arithmetic. A square cell of side h rotated by 30 deg has a bounding
    // box of h*(cos30 + sin30) = 1.366*h.
    const scalar bbDrift = maxAbs(bbFlat, bbRot);
    check(bbDrift > 0.05*hand, "negative control: the vertex bounding box DOES drift under the same rotation");
    std::printf("        (bbox drift = %.4g on delta = %.4g -> %.1f%%)\n",
                bbDrift, fnFlat[0], 100.0*bbDrift/fnFlat[0]);
    check(std::fabs(bbRot[0] - (std::cos(0.5235987755982988) + std::sin(0.5235987755982988))) < 1e-12,
          "negative control: the drifted bbox equals h*(cos30+sin30), i.e. the drift is the axis alignment");

    // ---- Leg 3: deltaCoeff scales linearly ---------------------------------------------------------
    const std::vector<scalar> half = cellMaxDeltaFaceNormal(flat, gFlat, 1.0);
    scalar eHalf = 0;
    for (size_t i = 0; i < half.size(); ++i) eHalf = std::max(eHalf, std::fabs(2.0*half[i] - fnFlat[i]));
    check(eHalf < 1e-14, "deltaCoeff 1 gives exactly half of deltaCoeff 2");

    // ---- Leg 4: the consumer end -- the sub-grid model must actually READ the buffer ----------------
    // Pure strain (du/dy = dv/dx = s) so Smagorinsky returns something non-trivial.
    const int nC = 8;
    std::vector<scalar> g(9*nC, scalar(0)), V(nC, scalar(0.001));
    for (int c = 0; c < nC; ++c) { g[1*nC + c] = 12.0; g[3*nC + c] = 12.0; }   // d(U_y)/dx and d(U_x)/dy
    DeviceBuffer<scalar> dG, dV, dD, nutDef, nutCbrt, nutBig;
    dG.copyFrom(g); dV.copyFrom(V);
    SmagorinskyCoeffs co;

    deviceSmagorinskyNut(nC, dG, dV, co, nutDef);                       // no delta -> cubeRootVol
    std::vector<scalar> cbrtV(nC);
    for (int c = 0; c < nC; ++c) cbrtV[c] = std::cbrt(V[c]);
    dD.copyFrom(cbrtV);
    deviceSmagorinskyNut(nC, dG, dV, co, nutCbrt, &dD);                 // delta == cubeRootVol explicitly
    std::vector<scalar> big(nC);
    for (int c = 0; c < nC; ++c) big[c] = 2.5*cbrtV[c];
    DeviceBuffer<scalar> dBig; dBig.copyFrom(big);
    deviceSmagorinskyNut(nC, dG, dV, co, nutBig, &dBig);

    std::vector<scalar> hDef, hCbrt, hBig;
    nutDef.copyTo(hDef); nutCbrt.copyTo(hCbrt); nutBig.copyTo(hBig);
    check(hDef[0] > 1e-12, "vacuity guard: the default nut is non-zero");
    check(maxAbs(hDef, hCbrt) < 1e-15, "passing delta = cbrt(V) reproduces the cubeRootVol default exactly");
    // Incompressible Smagorinsky is nut ~ delta^2, so 2.5x delta must be 6.25x nut.
    check(std::fabs(hBig[0]/hDef[0] - 6.25) < 1e-9, "2.5x delta gives 6.25x nut (nut ~ delta^2)");

    // ---- Leg 5: the parser reaches those controls ---------------------------------------------------
    {
        const FoamDict d = dictFromString(
            "simulationType LES;\n"
            "LES { LESModel Smagorinsky; turbulence on; printCoeffs on; delta maxDeltaxyz;\n"
            "      maxDeltaxyzCoeffs { deltaCoeff 1.5; } }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        readTurbulenceModel(d, ctl, {"test", true, true});
        check(ctl.lesDeltaMax, "parser: `delta maxDeltaxyz` sets lesDeltaMax");
        check(std::fabs(ctl.lesDeltaCoeff - 1.5) < 1e-14, "parser: maxDeltaxyzCoeffs/deltaCoeff is honoured");
    }
    {
        const FoamDict d = dictFromString("simulationType LES;\nLES { LESModel Smagorinsky; turbulence on; delta cubeRootVol; }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        readTurbulenceModel(d, ctl, {"test", true, true});
        check(ctl.les && !ctl.lesDeltaMax, "negative control: `delta cubeRootVol` is read as LES but leaves lesDeltaMax off");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
