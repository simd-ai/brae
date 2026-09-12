// D2: boundary coefficients vs OpenFOAM, for the BC types that actually caused bugs.
//
// D1 verified the assembled diagonal and source against OF's fvScalarMatrix to 1.76e-06 -- but only with
// plain fixedValue and zeroGradient patches, because diag_compare.cu builds its fields with a local loader
// that understands nothing else. So the one number that decides how a BC enters the matrix was never
// compared for any BC that is not trivial.
//
// That is exactly where group A lived. internalCoeffs goes to the adjacent cell's DIAGONAL and
// boundaryCoeffs to its SOURCE, and four separate defects were "the boundary value exists but never
// reaches the discretisation": A1 (inletOutlet on T never resolved, T 276% off), A12 (grad(K) taking he's
// descriptor), B5 (fixedGradient discretised as zeroGradient -- an adiabatic wall), and the
// face-diffusivity variant that B5's first attempt missed. Each was found separately, by inspection, after
// the fact. This compares the coefficients directly, so the next one shows up as a number.
//
// The field carries FOUR real BC types at once, and OF's own dump gives them four distinct signatures --
// so a comparison that accidentally tested nothing could not come out clean:
//
//     inlet    fixedValue      sum|IC| = 0.0004   sum|BC| = 0.3012
//     outlet   inletOutlet     sum|IC| = 0.1      sum|BC| = 0
//     gradWall fixedGradient   sum|IC| = 0        sum|BC| = 0.025
//     zeroWall zeroGradient    sum|IC| = 0        sum|BC| = 0
//
// Fields are built through brae's PRODUCTION factory (buildField -> makePatchField), not a test-local
// loader, so inletOutlet and fixedGradient take the same path a real run takes. A harness that
// reconstructs the BCs itself tests the harness.
//
//   bcoeff_compare <caseDir> <timeDir> [gamma]

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_blas.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

namespace {

// OF writes ICoeff/BCoeff as volScalarFields whose boundaryField holds the per-face coefficients. A patch
// whose coefficients are all identical is written `uniform <v>`, which is exactly what happens on the
// zeroGradient and fixedGradient patches this case exists to test -- so expanding it here is required, not
// a convenience. Returning the single value unexpanded silently skipped three of the four patches.
std::vector<scalar> ofPatchValues(const FieldData<scalar>& fd, const std::string& patch, std::size_t n)
{
    for (const auto& b : fd.boundary)
        if (b.name == patch)
        {
            if (!b.values.empty()) return b.values;
            if (b.hasValue) return std::vector<scalar>(n, b.uniformValue);
            return std::vector<scalar>(n, scalar(0));   // `calculated` with no value entry -> zero
        }
    return {};
}

}   // namespace

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: bcoeff_compare <caseDir> <timeDir> [gamma]\n"); return 2; }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar gammaVal = (argc > 3) ? std::atof(argv[3]) : 1e-3;

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // Production factory for BOTH fields: this is the path a real run takes.
    GeometricField<scalar> psi = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/psi"), fvp, nC);
    GeometricField<vector> U   = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    psi.evaluateBoundary();
    U.evaluateBoundary();

    // phi = fvc::flux(U), the same flux the OF utility builds -- not a phi read from disk, so both sides
    // derive it identically from the same U.
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceBoundary db = buildDeviceBoundary(psi, fvp, g);
    DeviceBuffer<scalar> phiBnd(flattenBoundary(phi.boundary));   // per-patch lists -> the DeviceBoundary flatten

    // inletOutlet is resolved PER FACE from the flux sign every step (bcCategory 3 -> 0 or 1). Skipping
    // this is precisely defect A1, which left outlet enthalpy clamped to inletValue and put T 276% out.
    deviceUpdateInletOutlet(db, phiBnd);

    // OF: fvm::div(phi,psi) - fvm::laplacian(gamma,psi), so the coefficients subtract the same way
    // deviceSolveScalarTransport subtracts them.
    DeviceBuffer<scalar> aIC, aBC, lIC, lBC;
    deviceBCDivCoeffs(db, phiBnd, aIC, aBC);
    DeviceBuffer<scalar> gammaCell(std::vector<scalar>(static_cast<std::size_t>(nC), gammaVal));
    deviceBCLaplacianCoeffs(db, gammaCell, lIC, lBC);
    deviceAxpy(-1.0, lIC, aIC);
    deviceAxpy(-1.0, lBC, aBC);

    const std::vector<scalar> braeIC = aIC.host(), braeBC = aBC.host();

    // THE HOST ARM. Everything above builds a DeviceBoundary and asks the DEVICE kernels for the
    // coefficients, so this gate has only ever tested the device path -- including for `fixedGradient`,
    // which its own header names as defect B5 ("fixedGradient discretised as zeroGradient -- an adiabatic
    // wall"). The HOST discretisation goes through the patch field's own four coefficient methods, and
    // FixedGradientPatchField overrode none of the boundary two, so they fell through to the base's
    // zeros: fvm::laplacian (fvm.cuh:96) and fvc::snGrad (fvc.cu:339) treated every prescribed gradient
    // as zeroGradient while the device carried it correctly. Two lineages, one boundary condition, no
    // gate able to see the difference.
    //
    // The formulas are fvm.cuh:6-9's, which is what the solver's own assembly uses:
    //     laplacian   IC = gamma*magSf*gradIC          BC = -gamma*magSf*gradBC
    //     div         IC = phi_pf*valueIC              BC = -phi_pf*valueBC
    // and the same subtraction the device arm applies, so both arms face the same OpenFOAM dump.
    std::vector<scalar> hostIC, hostBC;
    {
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const FvPatch& q = fvp[pi];
            if (q.type == "cyclic" || q.type == "cyclicAMI") continue;
            const std::vector<scalar> vIC = psi.boundary[pi]->valueInternalCoeffs();
            const std::vector<scalar> vBC = psi.boundary[pi]->valueBoundaryCoeffs();
            const std::vector<scalar> gIC = psi.boundary[pi]->gradientInternalCoeffs();
            const std::vector<scalar> gBC = psi.boundary[pi]->gradientBoundaryCoeffs();
            for (label i = 0; i < q.size; ++i)
            {
                const scalar phipf = phi.boundary[pi][i];
                const scalar gmS   = gammaVal * q.magSf[i];
                hostIC.push_back(phipf * vIC[i] - gmS * gIC[i]);
                hostBC.push_back(-phipf * vBC[i] + gmS * gBC[i]);
            }
        }
    }

    const FieldData<scalar> ofIC = readField<scalar>(caseDir + "/" + t + "/ICoeff");
    const FieldData<scalar> ofBC = readField<scalar>(caseDir + "/" + t + "/BCoeff");

    std::printf("  %-10s %-15s %8s   %-24s %-24s\n", "patch", "type", "faces", "internalCoeffs", "boundaryCoeffs");
    int bad = 0, compared = 0;
    std::size_t off = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& q = fvp[pi];
        if (q.type == "cyclic" || q.type == "cyclicAMI") continue;   // not in the DeviceBoundary flatten
        const std::size_t n = static_cast<std::size_t>(q.size);

        // OF collapses `empty` patches to zero faces, so there is nothing to compare there.
        const std::vector<scalar> oi = ofPatchValues(ofIC, q.name, n);
        const std::vector<scalar> ob = ofPatchValues(ofBC, q.name, n);
        if (q.type == "empty" || oi.size() != n || ob.size() != n)
        {
            std::printf("  %-10s %-15s %8zu   (skipped: %s)\n", q.name.c_str(), q.type.c_str(), n,
                        q.type == "empty" ? "empty -- OF collapses it to zero faces" : "OF face count differs");
            off += n;
            continue;
        }

        scalar sumOfI = 0, sumOfB = 0, dI = 0, dB = 0, nrmI = 0, nrmB = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const scalar bi = braeIC[off + i], bb = braeBC[off + i];
            sumOfI += std::fabs(oi[i]);
            sumOfB += std::fabs(ob[i]);
            dI += (oi[i] - bi) * (oi[i] - bi);
            dB += (ob[i] - bb) * (ob[i] - bb);
            nrmI += oi[i] * oi[i];
            nrmB += ob[i] * ob[i];
        }
        // A patch whose OF coefficients are all zero (zeroGradient) is compared ABSOLUTELY: a relative
        // error is undefined there, and "brae also produced zero" is the whole claim.
        const scalar eI = (nrmI > 0) ? std::sqrt(dI / nrmI) : std::sqrt(dI);
        const scalar eB = (nrmB > 0) ? std::sqrt(dB / nrmB) : std::sqrt(dB);

        // The HOST arm, against the SAME OpenFOAM dump.
        scalar hdI = 0, hdB = 0;
        for (std::size_t i = 0; i < n && off + i < hostIC.size(); ++i)
        {
            hdI += (oi[i] - hostIC[off + i]) * (oi[i] - hostIC[off + i]);
            hdB += (ob[i] - hostBC[off + i]) * (ob[i] - hostBC[off + i]);
        }
        const scalar hI = (nrmI > 0) ? std::sqrt(hdI / nrmI) : std::sqrt(hdI);
        const scalar hB = (nrmB > 0) ? std::sqrt(hdB / nrmB) : std::sqrt(hdB);
        const bool hostOk = (hI <= 1e-10) && (hB <= 1e-10);

        const bool ok = (eI <= 1e-10) && (eB <= 1e-10) && hostOk;
        std::printf("  %-10s %-15s %8zu   err %.3e (sum|OF| %.4g)  err %.3e (sum|OF| %.4g)  %s\n",
                    q.name.c_str(), psi.boundary[pi]->bcCategory() == 3 ? "inletOutlet" : q.type.c_str(),
                    n, (double)eI, (double)sumOfI, (double)eB, (double)sumOfB, ok ? "OK" : "FAIL");
        std::printf("  %-10s %-15s %8s   host err %.3e                    host err %.3e            %s\n",
                    "", "  (host arm)", "", (double)hI, (double)hB, hostOk ? "OK" : "FAIL");
        compared++;
        if (!ok) bad++;
        off += n;
    }

    if (compared < 4)
    {
        std::printf("  FAIL only %d patches compared; this case is meant to exercise four BC types at once\n", compared);
        bad++;
    }
    std::printf("bcoeff_compare: %d failures over %d patches\n", bad, compared);
    return bad ? 1 : 0;
}
