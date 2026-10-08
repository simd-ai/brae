// read_surface_field.cuh -- read a written surfaceScalarField (phi) back off disk.
//
// OF createPhi.H / compressibleCreatePhi.H construct phi with IOobject::READ_IF_PRESENT: on a restart the
// stored, CONTINUITY-SATISFYING face flux is read back, and `linearInterpolate(rho*U) & Sf` is only the
// fallback for a fresh start. Recomputing it unconditionally resumes from a flux the pressure equation has
// never corrected -- measured on angledDuct it differs from OF on 80778 of 80800 faces (L2rel 3.1e-03),
// which propagates straight into div(phi,U) and so into the momentum diagonal and rAU.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "../../../finiteVolume/finiteVolume/fvc.cuh"              // SurfaceScalarField
#include "../../../finiteVolume/fvMesh/fv_patch.cuh"               // FvPatch
#include "../../../finiteVolume/fields/foam_field_reader.cuh"      // FieldData / readField

namespace brae
{

// Read a surfaceScalarField (phi) written by writeSurfaceField (or by OpenFOAM) back into a SurfaceScalarField shaped
// exactly like fvc::flux: internal = the nInternalFaces flux list; boundary indexed [patch] with boundary[pi] sized
// patches[pi].size. readField<scalar> parses the file transparently (it does not check the FoamFile class nor the list
// length against nCells -- that check lives only in buildField, which we bypass). Values come from `calculated` patches;
// zeros stand in for empty/cyclic/cyclicAMI/no-value patches (the ctor recomputes cyclic/AMI flux from U + skips them,
// and an empty face flux is 0). Used only for a seamless restart (resume the exact written flux state).
// INLINE, because six translation units include this header and two of them -- interFoam's case
// reader and simpleFoam's driver -- end up in the same binary. Without it each emits a strong
// definition and the link fails the moment both objects are pulled (`multiple definition of
// brae::readSurfaceField`); the ordinary build survives only because the archive hands out one of
// them. tools/header_odr_audit.py asks this of every function defined in a .cuh.
// ONE IMPLEMENTATION FOR BOTH SURFACE TYPES. SurfaceScalarField and SurfaceVectorField are two structs with
// the same layout (fvc.cuh:16-25), and Uf needs exactly this read: createUfIfPresent.H is READ_IF_PRESENT
// with fvc::interpolate(U) only as its FALLBACK, the same shape as createPhi.H. Two copies of a reader is how
// the cellZones reader came to be ASCII-only while the mesh's own lists were not.
namespace detail
{

template <typename Field, typename T>
inline Field readSurfaceFieldImpl(const std::string& path, const std::vector<FvPatch>& patches, label nInternalFaces)
{
    const FieldData<T> fd = readField<T>(path);
    Field ssf;
    if (fd.internalUniform)
        ssf.internal.assign(static_cast<std::size_t>(nInternalFaces), fd.internalUniformValue);
    else if (static_cast<label>(fd.internalField.size()) != nInternalFaces)
        throw std::runtime_error("readSurfaceField: " + path + " internalField has "
            + std::to_string(fd.internalField.size()) + " faces, but the mesh has " + std::to_string(nInternalFaces)
            + " internal faces (wrong mesh / decomposed phi).");
    else
        ssf.internal = fd.internalField;

    ssf.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        std::vector<T> vals(static_cast<std::size_t>(patches[pi].size), T{});
        for (const auto& b : fd.boundary)   // pass-1 exact-name match (written phi lists exact mesh-patch names)
        {
            if (b.name != patches[pi].name) continue;
            // Coupled patches are READ too. They used to be skipped on the grounds that "the ctor
            // recomputes cyclic/AMI flux from U", but that rebuild is not the stored flux: on
            // pimpleFoam/RAS/oscillatingInletACMI2D it differed by 1.3e-04 per face, straight onto the
            // momentum diagonal through the upwind max(phi,0). The caller hands these to
            // setInterfacePatchFlux, and a file without them (brae's own older writes, or a solver with
            // no interface) still leaves the zeros the rebuild then overwrites.
            if (b.hasValue)
            {
                if (b.valueUniform) std::fill(vals.begin(), vals.end(), b.uniformValue);
                else if (b.values.size() == vals.size()) vals = b.values;   // else keep zeros (defensive)
            }
            break;
        }
        ssf.boundary[pi] = std::move(vals);
    }
    return ssf;
}

} // namespace detail

inline SurfaceScalarField readSurfaceField(const std::string& path, const std::vector<FvPatch>& patches, label nInternalFaces)
{
    return detail::readSurfaceFieldImpl<SurfaceScalarField, scalar>(path, patches, nInternalFaces);
}

// ...and Uf, the FACE VELOCITY of a dynamic mesh. createUfIfPresent.H builds it inside `if (mesh.dynamic())`
// -- MOVING OR TOPO-CHANGING -- with IOobject::READ_IF_PRESENT and AUTO_WRITE, so every time directory a
// REFINING case writes carries one, and a run resumed from any of them must read it. `phi = mesh.Sf() & Uf()`
// in interFoam's mesh-changed block (interFoam.C:131) consumes it directly, and fvc::ddtCorr reads its
// oldTime, so an interpolation standing in for it is not a small error.
inline SurfaceVectorField readSurfaceVectorField(const std::string& path, const std::vector<FvPatch>& patches, label nInternalFaces)
{
    return detail::readSurfaceFieldImpl<SurfaceVectorField, vector>(path, patches, nInternalFaces);
}

// OF READ_IF_PRESENT: the stored flux on a restart, else the caller's freshly computed fallback.
// `wasRead`, when given, reports which branch was taken. A caller that only looks at the returned field
// cannot tell: a coupled patch's boundary values are meaningful on the read path (OpenFOAM writes the
// interface flux per patch) and a placeholder on the fallback path, and only the interface knows which.
inline SurfaceScalarField readPhiIfPresent(const std::string& fieldDir, const std::vector<FvPatch>& patches,
                                           label nInternalFaces, SurfaceScalarField&& fallback,
                                           bool* wasRead = nullptr)
{
    const std::string phiPath = fieldDir + "/phi";
    // plain or .gz (writeCompression on): the reader opens either, and OpenFOAM's lookup finds either
    // (POSIX.C:870-876). Probing the plain path alone fell back to the interpolated flux on a restart from a
    // compressed write, with nothing said.
    if (!std::filesystem::exists(phiPath) && !std::filesystem::exists(phiPath + ".gz"))
    {
        if (wasRead) *wasRead = false;
        return std::move(fallback);
    }
    if (wasRead) *wasRead = true;
    return readSurfaceField(phiPath, patches, nInternalFaces);
}

// OF createUfIfPresent.H: the stored face velocity on a restart, else the caller's fvc::interpolate(U).
// `wasRead` reports which branch was taken, so a gate can assert that its fixture actually carries one --
// without it an arm comparing Uf could pass on a case that has no file at all.
inline SurfaceVectorField readUfIfPresent(const std::string& fieldDir, const std::vector<FvPatch>& patches,
                                          label nInternalFaces, SurfaceVectorField&& fallback,
                                          bool* wasRead = nullptr)
{
    const std::string ufPath = fieldDir + "/Uf";
    // plain or .gz, as readPhiIfPresent above. MEASURED on laminar/damBreakWithObstacle restarted from a
    // compressed refined write: the mesh and every level exactly OpenFOAM's, U 3.2e-01 off, because
    // phi = Sf & Uf at the change read the interpolation instead of the stored Uf.
    if (!std::filesystem::exists(ufPath) && !std::filesystem::exists(ufPath + ".gz"))
    {
        if (wasRead) *wasRead = false;
        return std::move(fallback);
    }
    if (wasRead) *wasRead = true;
    return readSurfaceVectorField(ufPath, patches, nInternalFaces);
}

} // namespace brae
