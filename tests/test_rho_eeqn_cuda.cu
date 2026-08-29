// CUDA rhoEEqn against the _cpp reference, stage by stage and BOTH ENERGY VARIABLES.
//
// The reference is itself gated against OpenFOAM's own stage_Ekp / stage_eD / stage_eSrc dumps
// (tests/rho_eeqn_vs_openfoam.sh), so this closes OpenFOAM -> _cpp -> CUDA for the whole compressible
// energy equation. Third of the three device twins, beside test_rho_ueqn_cuda and test_rho_peqn_cuda.
//
// THE ENERGY VARIABLE IS THE BRANCH. EEqn.H builds the kinetic-energy source as Ekp = 0.5|U|^2 + p/rho
// when he is `e` and as K = 0.5|U|^2 when he is `h`. Those are different conservation laws, not different
// spellings, so the binary runs whichever BRAE_TEST_ENERGY selects and both are registered.
//
// he, rho AND alphaEff ARE SYNTHESIZED, for the reason test_rho_ueqn_cuda.cu sets out at length: no
// committed fixture ships them, and this instrument measures DEVICE-vs-HOST on byte-identical inputs
// rather than physics. `he` reuses the p field's STRUCTURE -- its real, varied boundary conditions -- with
// its own smooth internal values, because what the gate needs from it is a valid boundary and a
// non-degenerate field, not a physical enthalpy. Physical validity is rho_eeqn_vs_openfoam.sh's claim.
//
// alphaEff MUST VARY, and on the boundary independently of the cells: effectiveFaceViscosity overwrites
// the boundary faces with the PATCH value, and with a constant alphaEff a kernel that read the owner
// cell's value instead would still agree everywhere.
//
// Run: test_rho_eeqn_cuda <caseDir> <timeDir>       env: BRAE_TEST_ENERGY=h
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
#include "rhoEEqn_cpp.cuh"
#include "rhoEEqn.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

static void cmp(const std::vector<scalar>& gpu,
                const std::vector<scalar>& ref,
                const char*                nm,
                scalar                     tol)
{
    if (gpu.size() != ref.size())
    {
        std::printf("  %-34s SIZE MISMATCH %zu vs %zu  FAIL\n", nm, gpu.size(), ref.size());
        ++g_fails;
        return;
    }
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(gpu[i] - ref[i]));
        mg = std::fmax(mg, std::fabs(ref[i]));
    }
    const scalar rel = mg > 0 ? mx / mg : mx;
    const bool ok = rel <= tol;
    if (!ok) ++g_fails;
    std::printf("  %-34s n=%6zu rel=%.3e  %s\n", nm, ref.size(), rel, ok ? "OK" : "FAIL");
}

static void check(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    const char* eEnv = std::getenv("BRAE_TEST_ENERGY");
    const std::string heName = (eEnv && eEnv[0] == 'h') ? "h" : "e";
    const bool isE = (heName == "e");

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    U.evaluateBoundary();
    GeometricField<scalar> p =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    p.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    std::vector<std::vector<scalar>> phiBnd(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phiBnd[pi].assign(fvp[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
        {
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
            {
                phiBnd[pi] = b.values;
            }
        }
    }

    scalar xMin = 1e300, xMax = -1e300;
    for (label c = 0; c < nC; ++c)
    {
        xMin = std::fmin(xMin, g.C()[c].x);
        xMax = std::fmax(xMax, g.C()[c].x);
    }
    const scalar xSpan = (xMax > xMin) ? (xMax - xMin) : 1.0;
    auto frac = [&](const vector& x) { return (x.x - xMin) / xSpan; };

    // rho, and the energy field. `he` borrows p's boundary conditions -- see the header.
    std::vector<scalar> rhoC(nC);
    // Built from their own reads rather than copied off p: GeometricField owns unique_ptr patch fields
    // and is deliberately non-copyable. Re-reading gives each field independent boundary objects while
    // still borrowing p's real, varied boundary CONDITIONS -- which is what these two need.
    GeometricField<scalar> rho =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    GeometricField<scalar> he =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    rho.evaluateBoundary();
    for (label c = 0; c < nC; ++c)
    {
        rhoC[c] = scalar(0.8) + scalar(0.6) * frac(g.C()[c]);
        rho.internal[c] = rhoC[c];
        he.internal[c] = scalar(2.0e5) + scalar(6.0e4) * frac(g.C()[c]);
    }
    he.evaluateBoundary();   // AFTER the internal field, so the boundary is what the conditions produce
    // he's BOUNDARY IS EVALUATED FROM ITS BOUNDARY CONDITIONS, not stored by hand. The host reads
    // he.boundary[pi]->value(); the device re-derives it with deviceBCValue from the snapshotted BC
    // types. Those agree only when the stored values ARE what the conditions produce -- forcing a
    // synthesized profile instead made the two gradients differ, and the laplacian's non-orthogonal
    // correction (which is built from that gradient) came out 0.26% off while every other stage sat at
    // 1e-16. rho does not have this problem: its boundary is handed to the device as an explicit array.
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        std::vector<scalar> rb(fvp[pi].size), hb(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            rb[i] = scalar(0.8) + scalar(0.6) * frac(fvp[pi].Cf[i]);
            hb[i] = scalar(2.0e5) + scalar(6.0e4) * frac(fvp[pi].Cf[i]);
        }
        rho.boundary[pi]->setStoredValues(std::move(rb));
        (void)hb;
    }

    // alphaEff: varying per cell AND per boundary face, the boundary independently of the cell so the
    // patch-value rule is actually exercised.
    std::vector<scalar> alphaC(nC);
    std::vector<std::vector<scalar>> alphaB(fvp.size());
    for (label c = 0; c < nC; ++c) alphaC[c] = scalar(2.0e-5) + scalar(1.4e-5) * frac(g.C()[c]);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        alphaB[pi].resize(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i)
            alphaB[pi][i] = scalar(3.1e-5) + scalar(2.2e-5) * frac(fvp[pi].Cf[i]);
    }

    std::printf("test_rho_eeqn_cuda:  he = '%s'  (%s)\n", heName.c_str(),
                isE ? "Ekp = 0.5|U|^2 + p/rho" : "K = 0.5|U|^2");

    // ---- the reference ----------------------------------------------------------------------
    cpu::rhoSimple::EnergyInput ein;
    ein.phi = &phiF.internalField;  ein.phiBnd = &phiBnd;
    ein.alphaEff = &alphaC;         ein.alphaEffBnd = &alphaB;
    ein.heName = heName;
    ein.relaxHe = 0.5;
    ein.relaxEquationHe = true;
    ein.boundedHe = true;
    ein.boundedKE = true;
    ein.correctedLaplacian = true;
    const std::vector<scalar> refKe = cpu::rhoSimple::kineticEnergy(heName, U, p, rho);
    const std::vector<scalar> refKeDiv =
        cpu::rhoSimple::kineticEnergyDivergence(U, p, rho, ein, m, g, fvp);
    const FvScalarMatrix refE = cpu::rhoSimple::assembleEEqn(he, U, p, rho, ein, m, g, fvp);

    // ---- the CUDA path ----------------------------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceBoundary dbHe = buildDeviceBoundary(he, fvp, g);

    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }

    std::vector<scalar> uxb, uyb, uzb, pb, rb, ab, phib;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<vector>& ubv = U.boundary[pi]->value();
        const std::vector<scalar>& pbv = p.boundary[pi]->value();
        const std::vector<scalar>& rbv = rho.boundary[pi]->value();
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            uxb.push_back(ubv[i].x); uyb.push_back(ubv[i].y); uzb.push_back(ubv[i].z);
            pb.push_back(pbv[i]);
            rb.push_back(rbv[i]);
            ab.push_back(alphaB[pi][i]);
            phib.push_back(phiBnd[pi][i]);
        }
    }
    uxb.resize(dm.nBndFaces, 0.0); uyb.resize(dm.nBndFaces, 0.0); uzb.resize(dm.nBndFaces, 0.0);
    pb.resize(dm.nBndFaces, 0.0);
    rb.resize(dm.nBndFaces, 1.0);
    ab.resize(dm.nBndFaces, 2.0e-5);
    phib.resize(dm.nBndFaces, 0.0);

    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dHe(he.internal), dP(p.internal), dRho(rhoC);
    DeviceBuffer<scalar> dAlpha(alphaC), dPhiInt(phiF.internalField);
    DeviceBuffer<scalar> dUxb(uxb), dUyb(uyb), dUzb(uzb), dPb(pb), dRb(rb), dAb(ab), dPhib(phib);

    gpu::rhoSimple::RhoEnergyInput gin;
    gin.phiInt = &dPhiInt;          gin.phiBnd = &dPhib;
    gin.alphaEffCell = &dAlpha;     gin.alphaEffBndFace = &dAb;
    gin.Ux = &dUx; gin.Uy = &dUy; gin.Uz = &dUz;
    gin.pCell = &dP; gin.rhoCell = &dRho;
    gin.UxBnd = &dUxb; gin.UyBnd = &dUyb; gin.UzBnd = &dUzb;
    gin.pBnd = &dPb; gin.rhoBnd = &dRb;
    gin.isE = isE;
    gin.relaxHe = 0.5;
    gin.relaxEquationHe = true;   // the host still uses the 1.0 sentinel; at 0.5 both relax
    gin.boundedHe = true;
    gin.boundedKE = true;
    gin.correctedLaplacian = true;

    DeviceBuffer<scalar> gKe, gKeB, gKeDiv;
    gpu::rhoSimple::kineticEnergy(gKe, gKeB, dm, gin);
    gpu::rhoSimple::kineticEnergyDivergence(gKeDiv, dm, gin);
    gpu::PressureMatrix GE;
    gpu::rhoSimple::assembleEEqn(GE, dm, dbHe, dHe, gin);

    // ---- stage by stage ---------------------------------------------------------------------
    cmp(gKe.host(), refKe, isE ? "Ekp = 0.5|U|^2 + p/rho" : "K = 0.5|U|^2", 1e-13);
    cmp(gKeDiv.host(), refKeDiv, "fvc::div(phi, Ekp|K) extensive", 1e-11);
    cmp(GE.diag.host(),   refE.diag,   "EEqn diag",   1e-11);
    cmp(GE.upper.host(),  refE.upper,  "EEqn upper",  1e-12);
    cmp(GE.lower.host(),  refE.lower,  "EEqn lower",  1e-12);
    cmp(GE.source.host(), refE.source, "EEqn source", 1e-10);
    {
        std::vector<scalar> ric, rbc;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                ric.push_back(refE.internalCoeffs[pi][i]);
                rbc.push_back(refE.boundaryCoeffs[pi][i]);
            }
        }
        std::vector<scalar> gic = GE.iC.host(), gbc = GE.bC.host();
        gic.resize(ric.size());
        gbc.resize(rbc.size());
        cmp(gic, ric, "EEqn internalCoeffs", 1e-11);
        cmp(gbc, rbc, "EEqn boundaryCoeffs", 1e-11);
    }

    // ---- CONTROL: the energy variable is a different equation --------------------------------
    // The manifest's own claim for this component. Ekp and K differ by p/rho, which on a compressible
    // case is the dominant term -- so if the two arms agreed, the branch could be deleted unnoticed.
    {
        cpu::rhoSimple::EnergyInput other = ein;
        other.heName = isE ? "h" : "e";
        const std::vector<scalar> oKe = cpu::rhoSimple::kineticEnergy(other.heName, U, p, rho);
        scalar d = 0, mg = 0;
        for (label c = 0; c < nC; ++c)
        {
            d = std::fmax(d, std::fabs(refKe[c] - oKe[c]));
            mg = std::fmax(mg, std::fabs(refKe[c]));
        }
        const scalar r = mg > 0 ? d / mg : d;
        std::printf("  %-58s rel=%.3e\n", "control: the OTHER energy variable differs", (double)r);
        check(r > 1e-6, "Ekp and K are different fields (control)");
    }

    // ---- CONTROL: `corrected` moves BOTH the coefficients and the source ---------------------
    // This is the equation where implementing only the implicit half was found: source 2.14e-05 out with
    // the diagonal exact at 1.63e-15. Both halves are asserted separately for that reason.
    {
        cpu::rhoSimple::EnergyInput noC = ein;
        noC.correctedLaplacian = false;
        const FvScalarMatrix cE = cpu::rhoSimple::assembleEEqn(he, U, p, rho, noC, m, g, fvp);
        scalar dD = 0, mD = 0, dS = 0, mS = 0;
        for (label c = 0; c < nC; ++c)
        {
            dD = std::fmax(dD, std::fabs(refE.diag[c] - cE.diag[c]));
            mD = std::fmax(mD, std::fabs(refE.diag[c]));
            dS = std::fmax(dS, std::fabs(refE.source[c] - cE.source[c]));
            mS = std::fmax(mS, std::fabs(refE.source[c]));
        }
        const scalar rD = mD > 0 ? dD / mD : dD, rS = mS > 0 ? dS / mS : dS;
        std::printf("  %-58s diag=%.3e src=%.3e\n",
                    "control: `corrected` moves coefficients AND source", (double)rD, (double)rS);
        check(rD > 1e-12, "the implicit half of the non-orth correction contributes (control)");
        check(rS > 1e-12, "the explicit half of the non-orth correction contributes (control)");
    }

    // ---- CONTROL: the boundary alphaEff is not the owner cell's -----------------------------
    // Replacing the patch array with the cell values must change the matrix, or the face rule is
    // untested and a kernel reading the owner cell would pass.
    {
        std::vector<std::vector<scalar>> cellAsBnd(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            cellAsBnd[pi].resize(fvp[pi].size);
            for (label i = 0; i < fvp[pi].size; ++i)
                cellAsBnd[pi][i] = alphaC[fvp[pi].faceCells[i]];
        }
        cpu::rhoSimple::EnergyInput ownerCell = ein;
        ownerCell.alphaEffBnd = &cellAsBnd;
        const FvScalarMatrix oE = cpu::rhoSimple::assembleEEqn(he, U, p, rho, ownerCell, m, g, fvp);
        scalar d = 0, mg = 0;
        for (std::size_t pi = 0, k = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i, ++k)
            {
                d = std::fmax(d, std::fabs(refE.internalCoeffs[pi][i] - oE.internalCoeffs[pi][i]));
                mg = std::fmax(mg, std::fabs(refE.internalCoeffs[pi][i]));
            }
        }
        const scalar r = mg > 0 ? d / mg : d;
        std::printf("  %-58s rel=%.3e\n", "control: patch alphaEff != owner-cell alphaEff", (double)r);
        check(r > 1e-6, "the boundary diffusivity is the PATCH value (control)");
    }

    // ---- CONTROL: `bounded` contributes ------------------------------------------------------
    {
        cpu::rhoSimple::EnergyInput noB = ein;
        noB.boundedHe = false;
        noB.boundedKE = false;
        const FvScalarMatrix bE = cpu::rhoSimple::assembleEEqn(he, U, p, rho, noB, m, g, fvp);
        scalar d = 0, mg = 0;
        for (label c = 0; c < nC; ++c)
        {
            d = std::fmax(d, std::fabs(refE.diag[c] - bE.diag[c]));
            mg = std::fmax(mg, std::fabs(refE.diag[c]));
        }
        const scalar r = mg > 0 ? d / mg : d;
        std::printf("  %-58s rel=%.3e\n", "control: `bounded` changes the diagonal", (double)r);
        check(r > 1e-12, "the bounded terms contribute (control)");
    }

    // ---- refusals ---------------------------------------------------------------------------
    {
        struct { const char* what; int which; } cases[] = {
            {"MRF is refused on the CUDA path",        0},
            {"an unported fvOptions is refused",       1},
            {"a mesh with coupled patches is refused", 2},
            {"an unported div(phi,he) scheme is refused", 3},
        };
        for (const auto& cse : cases)
        {
            gpu::rhoSimple::RhoEnergyInput bad = gin;
            if (cse.which == 0) bad.hasMRF = true;
            if (cse.which == 1) bad.hasFvOptions = true;
            if (cse.which == 2) bad.hasCoupledPatches = true;
            if (cse.which == 3) bad.schemeHe = cpu::rhoSimple::DivScheme::LUST;
            gpu::PressureMatrix Eb;
            bool threw = false;
            try { gpu::rhoSimple::assembleEEqn(Eb, dm, dbHe, dHe, bad); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, cse.what);
        }
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
