// _cpp REFERENCE implementation -- see createFields_cpp.cuh for the OpenFOAM provenance.
#include "createFields_cpp.cuh"
#include "frozen_bc_guard.cuh"
#include "foam_field_reader.cuh"
#include <filesystem>
#include <stdexcept>

namespace brae {
namespace cpu {

bool needReference(const GeometricField<scalar>& p)
{
    // GeometricField.C:1073-1084 -- true unless some patch fixes a value.
    for (const auto& b : p.boundary)
        if (b->fixesValue()) return false;
    return true;
}


SimpleFields createFields(
    const std::string&          timeDir,
    const FoamDict*             simpleDict,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    SimpleFields f;
    const label nC = m.nCells();

    // p, U: MUST_READ. A missing file is a hard error in OpenFOAM and stays one here.
    // The read is split so the boundary TYPES can be checked before they are erased by buildField:
    // no caller of this createFields maintains a per-step boundary (fixedMean, fanPressure, coded), so
    // building one would freeze it at the file `value` -- see frozen_bc_guard.cuh.
    const FieldData<scalar> pFd = readField<scalar>(timeDir + "/p");
    refuseFrozenPerStepBC(pFd, "p", "simpleFoam (mirror createFields)", false);
    f.p = buildField<scalar>(pFd, patches, nC);
    f.p.evaluateBoundary();
    const FieldData<vector> UFd = readField<vector>(timeDir + "/U");
    refuseFrozenPerStepBC(UFd, "U", "simpleFoam (mirror createFields)", false);
    f.U = buildField<vector>(UFd, patches, nC);
    f.U.evaluateBoundary();

    // createPhi.H: READ_IF_PRESENT, else fvc::flux(U).
    const std::string phiPath = timeDir + "/phi";
    f.phiWasRead = std::filesystem::exists(phiPath) || std::filesystem::exists(phiPath + ".gz");
    if (f.phiWasRead)
    {
        const FieldData<scalar> pf = readField<scalar>(phiPath);
        f.phi.internal = pf.internalField;
        f.phi.boundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            f.phi.boundary[pi].assign(patches[pi].size, 0.0);
            for (const auto& b : pf.boundary)
                if (b.name == patches[pi].name && b.hasValue
                    && static_cast<label>(b.values.size()) == patches[pi].size)
                    f.phi.boundary[pi] = b.values;
        }
    }
    else
    {
        f.phi = fvc::flux(f.U, m, g, patches);
    }

    // setRefCell(p, simple.dict(), pRefCell, pRefValue) -- only when p needs a reference.
    if (needReference(f.p))
    {
        if (!simpleDict)
            throw std::runtime_error(
                "createFields: p needs a reference (no boundary patch fixes its value) but fvSolution has "
                "no SIMPLE dictionary to read pRefCell/pRefValue from (findRefCell.C:33).");

        if (simpleDict->found("pRefCell"))
        {
            f.pRefCell = simpleDict->intOr("pRefCell", 0);
            if (f.pRefCell < 0 || f.pRefCell >= nC)
                throw std::runtime_error(
                    "createFields: illegal pRefCell " + std::to_string(f.pRefCell) + "; should be 0.."
                    + std::to_string(nC) + " (findRefCell.C:55-62).");
        }
        else if (simpleDict->found("pRefPoint"))
        {
            // OpenFOAM locates the cell containing pRefPoint (mesh.findCell). brae has no point-location
            // search on this path, and guessing a cell would silently pin the pressure level somewhere
            // the user did not ask for.
            throw std::runtime_error(
                "createFields: pRefPoint is set. OpenFOAM resolves it with mesh.findCell "
                "(findRefCell.C:69-100); brae has no cell search here. Use pRefCell instead.");
        }
        else
        {
            // findRefCell.C: FatalIOError when a reference is needed and neither key is given.
            throw std::runtime_error(
                "createFields: p needs a reference (no boundary patch fixes its value) but neither "
                "pRefCell nor pRefPoint is set in the SIMPLE dictionary (findRefCell.C).");
        }
        f.pRefValue = simpleDict->scalarOr("pRefValue", 0.0);
    }
    return f;
}

} // namespace cpu
} // namespace brae
