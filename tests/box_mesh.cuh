#pragma once
// Shared test fixture: an in-memory structured hex box (duct) + FieldData helpers.
//
// Used by the tests that need a mesh too large or too specific to ship as a polyMesh on disk:
//   test_gpu_parallel_oom  -- a mesh bigger than one GPU's VRAM (a polyMesh that size would be tens of GB)
//   test_gpu_parallel_duct -- a cut that actually CARRIES FLUX (see that test's header for why this matters)
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include <vector>

namespace brae {
namespace boxtest {


// A structured Nx*Ny*Nz unit-spaced hex box. Faces are emitted in OpenFOAM order: internal faces first in
// upper-triangular order (owner ascending, neighbour ascending within an owner -- the lduAddressing contract),
// then one contiguous block per boundary patch. Face vertex winding is chosen so Sf points owner->neighbour
// for internal faces and outward for boundary faces (right-hand rule).
// `shear` skews the mesh: point x is displaced by shear*y, so cell-centre offsets are no longer aligned with
// the face normals -- i.e. the mesh becomes NON-ORTHOGONAL, and OF's "corrected" laplacian correction stops
// being a no-op. shear = 0 gives the plain orthogonal box.
//
// This exists because EVERY other brae test mesh is orthogonal (cav3d is a cube, the duct a unit box), which
// makes the non-orthogonal correction invisible: on an orthogonal mesh "corrected" == "orthogonal" exactly.
// Testing that correction on an orthogonal mesh is the same class of blindness as testing a cut-face term on
// a zero-flux cut.
//
// `dx/dy/dz` set the CELL SIZE per axis. They default to 1, which is the unit box every existing caller
// gets, and that box has one property worth naming: every face has the same area, so any term carrying a
// magSf factor is indistinguishable from the same term without it. Giving the three axes different sizes
// makes the six face areas take three distinct values while the mesh stays ORTHOGONAL -- so an exact
// identity still holds on it and a missing magSf no longer does.
PrimitiveMesh boxMesh(label Nx, label Ny, label Nz, scalar shear = 0.0,
                      scalar dx = 1.0, scalar dy = 1.0, scalar dz = 1.0)
{
    const label nC = Nx * Ny * Nz;
    auto cell  = [&](label i, label j, label k) { return i + Nx * (j + Ny * k); };
    auto point = [&](label i, label j, label k) { return i + (Nx + 1) * (j + (Ny + 1) * k); };

    std::vector<vector> pts(static_cast<std::size_t>(Nx + 1) * (Ny + 1) * (Nz + 1));
    for (label k = 0; k <= Nz; ++k)
        for (label j = 0; j <= Ny; ++j)
            for (label i = 0; i <= Nx; ++i)
                pts[point(i, j, k)] = vector{dx * scalar(i) + shear * dy * scalar(j),
                                             dy * scalar(j),
                                             dz * scalar(k)};

    // reserve: at ~1e8 cells these reach ~1e9 entries, and push_back's doubling would otherwise transiently
    // double an already multi-GB allocation. nFaces = 3*nC + the 6 boundary planes; nInternal = nFaces - nBnd.
    const std::size_t nBnd   = static_cast<std::size_t>(2) * (Ny * Nz + Nx * Nz + Nx * Ny);
    const std::size_t nFaces = static_cast<std::size_t>(3) * nC
                             - static_cast<std::size_t>(Ny * Nz + Nx * Nz + Nx * Ny) + nBnd;
    std::vector<label> fv, foff, own, nei;
    fv.reserve(4 * nFaces);
    foff.reserve(nFaces + 1);
    own.reserve(nFaces);
    nei.reserve(nFaces - nBnd);
    foff.push_back(0);
    auto quad = [&](label a, label b, label c, label d, label o, label n)
    {
        fv.push_back(a); fv.push_back(b); fv.push_back(c); fv.push_back(d);
        foff.push_back(static_cast<label>(fv.size()));
        own.push_back(o);
        if (n >= 0) nei.push_back(n);
    };
    // face planes with the winding that yields the +x / +y / +z normal
    auto faceX = [&](label i, label j, label k, label o, label n)   // plane x=i
    { quad(point(i,j,k), point(i,j+1,k), point(i,j+1,k+1), point(i,j,k+1), o, n); };
    auto faceY = [&](label i, label j, label k, label o, label n)   // plane y=j
    { quad(point(i,j,k), point(i,j,k+1), point(i+1,j,k+1), point(i+1,j,k), o, n); };
    auto faceZ = [&](label i, label j, label k, label o, label n)   // plane z=k
    { quad(point(i,j,k), point(i+1,j,k), point(i+1,j+1,k), point(i,j+1,k), o, n); };
    // reversed winding -> the opposite normal (for min-side boundary patches, whose outward normal is -)
    auto faceXr = [&](label i, label j, label k, label o)
    { quad(point(i,j,k+1), point(i,j+1,k+1), point(i,j+1,k), point(i,j,k), o, -1); };
    auto faceYr = [&](label i, label j, label k, label o)
    { quad(point(i+1,j,k), point(i+1,j,k+1), point(i,j,k+1), point(i,j,k), o, -1); };
    auto faceZr = [&](label i, label j, label k, label o)
    { quad(point(i,j+1,k), point(i+1,j+1,k), point(i+1,j,k), point(i,j,k), o, -1); };

    // internal faces, upper-triangular: for each cell ascending, neighbours +1 < +Nx < +Nx*Ny
    for (label k = 0; k < Nz; ++k)
        for (label j = 0; j < Ny; ++j)
            for (label i = 0; i < Nx; ++i)
            {
                const label c = cell(i, j, k);
                if (i + 1 < Nx) faceX(i + 1, j, k, c, cell(i + 1, j, k));
                if (j + 1 < Ny) faceY(i, j + 1, k, c, cell(i, j + 1, k));
                if (k + 1 < Nz) faceZ(i, j, k + 1, c, cell(i, j, k + 1));
            }

    std::vector<PatchInfo> patches;
    auto beginPatch = [&](const char* name, const char* type)
    {
        PatchInfo pi;
        pi.name  = name;
        pi.type  = type;
        pi.start = static_cast<label>(own.size());
        patches.push_back(pi);
    };
    auto endPatch = [&]() { patches.back().size = static_cast<label>(own.size()) - patches.back().start; };

    beginPatch("inlet", "patch");                                          // xmin, outward -x
    for (label k = 0; k < Nz; ++k) for (label j = 0; j < Ny; ++j) faceXr(0, j, k, cell(0, j, k));
    endPatch();
    beginPatch("outlet", "patch");                                         // xmax, outward +x
    for (label k = 0; k < Nz; ++k) for (label j = 0; j < Ny; ++j) faceX(Nx, j, k, cell(Nx - 1, j, k), -1);
    endPatch();
    beginPatch("wallYmin", "wall");                                        // outward -y
    for (label k = 0; k < Nz; ++k) for (label i = 0; i < Nx; ++i) faceYr(i, 0, k, cell(i, 0, k));
    endPatch();
    beginPatch("wallYmax", "wall");                                        // outward +y
    for (label k = 0; k < Nz; ++k) for (label i = 0; i < Nx; ++i) faceY(i, Ny, k, cell(i, Ny - 1, k), -1);
    endPatch();
    beginPatch("wallZmin", "wall");                                        // outward -z
    for (label j = 0; j < Ny; ++j) for (label i = 0; i < Nx; ++i) faceZr(i, j, 0, cell(i, j, 0));
    endPatch();
    beginPatch("wallZmax", "wall");                                        // outward +z
    for (label j = 0; j < Ny; ++j) for (label i = 0; i < Nx; ++i) faceZ(i, j, Nz, cell(i, j, Nz - 1), -1);
    endPatch();

    PrimitiveMesh m;
    m.assign(std::move(pts), std::move(fv), std::move(foff), std::move(own), std::move(nei),
             std::move(patches), nC);
    return m;
}

template <typename T>
PatchFieldData<T> pfd(const char* name, const char* type, T v, bool hasValue)
{
    PatchFieldData<T> d;
    d.name         = name;
    d.type         = type;
    d.hasValue     = hasValue;
    d.valueUniform = hasValue;
    d.uniformValue = v;
    return d;
}


}  // namespace boxtest
}  // namespace brae
