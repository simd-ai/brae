// rhoSimpleFoam's pcEqn.H -- the SIMPLEC pressure equation -- against REAL OpenFOAM, BOTH branches.
//
// THE ORACLE is tools/dumpPEqn. The subsonic branch already had a stage harness; the transonic branch had
// none, so one was added (stage_phid, stage_phiHbyA, stage_pICt/pBCt/pDt/pSrct) -- that branch carries
// `fvm::div(phid, p)`, the term that makes the pressure equation convective, and nothing could be
// compared against it before.
//
// The gate script runs OpenFOAM twice on the same fixture, once with `transonic no` and once with
// `transonic yes`, and this binary is told which one it is looking at. Both runs also force
// `consistent no`, because `consistent yes` sends OpenFOAM to pcEqn.H instead -- a different file and a
// different manifest component.
//
// WHAT DIFFERS BETWEEN THE BRANCHES, and why one gate could not cover both by accident:
//
//                        transonic                        subsonic
//   matrix               + fvm::div(phid, p)              (no convective term)
//   phiHbyA              -= interpolate(psi*p)*phiHbyA    adjustPhi(phiHbyA, U, p)
//                           /interpolate(rho)
//   pEqn.relax()         YES                              NO
//   closedVolume         never set                        set by adjustPhi
//
// UEqn's matrix is rebuilt here from OpenFOAM's own muEff and stage_Uass, because rAU = 1/UEqn.A() and
// HbyA = rAU*UEqn.H() both come from it: feeding a momentum matrix brae assembled from its own turbulence
// closure would make a pEqn failure unattributable. That the momentum matrix itself is right is the
// separate business of rho_ueqn_vs_openfoam.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fv_matrix_ops.cuh"
#include "rhoCreateFields_cpp.cuh"
#include "rhoUEqn_cpp.cuh"
#include "rhoPEqn_cpp.cuh"
#include "rhoPcEqn_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace brae;

static std::vector<scalar> rawInternal(
    const FieldData<scalar>& fd,
    label nC)
{
    if (fd.internalUniform) return std::vector<scalar>(nC, fd.internalUniformValue);
    return fd.internalField;
}

template <typename T>
static std::vector<std::vector<T>> rawBoundary(
    const FieldData<T>&         fd,
    const std::vector<FvPatch>& patches)
{
    std::vector<std::vector<T>> out(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        out[pi].assign(patches[pi].size, T{});
        for (const auto& b : fd.boundary)
        {
            if (b.name != patches[pi].name) continue;
            if (b.valueUniform) out[pi].assign(patches[pi].size, b.uniformValue);
            else if (static_cast<label>(b.values.size()) == patches[pi].size) out[pi] = b.values;
            break;
        }
    }
    return out;
}

static int failures = 0;

static void report(const std::string& what, double got, double bound)
{
    const bool ok = got < bound;
    if (!ok) ++failures;
    std::printf("     %-42s %.6e   %s\n", what.c_str(), got, ok ? "ok" : "FAIL");
}

static void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-42s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

static double relL2(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    double num = 0.0, den = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// A SurfaceScalarField compared over internal + every boundary face.
static double relL2Surface(
    const SurfaceScalarField&               a,
    const std::vector<scalar>&              bInt,
    const std::vector<std::vector<scalar>>& bBnd,
    const std::vector<FvPatch>&             patches)
{
    std::vector<scalar> fa = a.internal, fb = bInt;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
        {
            fa.push_back(a.boundary[pi][i]);
            fb.push_back(bBnd[pi][i]);
        }
    return relL2(fa, fb);
}

static std::vector<scalar> matrixD(const FvScalarMatrix& M, const std::vector<FvPatch>& patches)
{
    std::vector<scalar> D = M.diag;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            D[patches[pi].faceCells[i]] += M.internalCoeffs[pi][i];
    return D;
}

static std::vector<scalar> matrixRhs(const FvScalarMatrix& M, const std::vector<FvPatch>& patches)
{
    std::vector<scalar> r = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            r[patches[pi].faceCells[i]] += M.boundaryCoeffs[pi][i];
    return r;
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::printf("usage: %s <caseDir> <startTime> <dumpTime> <transonic:0|1>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string startT  = argv[2];
    const std::string dumpT   = argv[3];
    const bool transonic = (std::strcmp(argv[4], "1") == 0);

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    std::printf("rhoSimpleFoam pcEqn (SIMPLEC) vs OpenFOAM -- %s branch (%d cells)\n",
                transonic ? "TRANSONIC" : "subsonic", (int)nC);

    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);

    const FieldData<scalar> muFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_muEff");
    const std::vector<scalar>              muEff    = rawInternal(muFd, nC);
    const std::vector<std::vector<scalar>> muEffBnd = rawBoundary<scalar>(muFd, patches);
    const FieldData<vector> uaFd = readField<vector>(caseDir + "/" + dumpT + "/stage_Uass");
    const std::vector<std::vector<vector>> uaBnd = rawBoundary<vector>(uaFd, patches);
    if (!uaFd.internalUniform && static_cast<label>(uaFd.internalField.size()) == nC)
        f.U.internal = uaFd.internalField;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) f.U.boundary[pi]->setValue(uaBnd[pi]);

    // rho AS pcEqn.H SEES IT. pcEqn.H opens with `rho = thermo.rho()` -- pEqn.H does not -- so the
    // SIMPLEC pressure equation is built from a density already updated for the just-solved p and T.
    // That call is thermo's business, so OpenFOAM's own value is injected here.
    const FieldData<scalar> rhoFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_rhoP");
    const std::vector<scalar>              rhoP   = rawInternal(rhoFd, nC);
    const std::vector<std::vector<scalar>> rhoBnd = rawBoundary<scalar>(rhoFd, patches);
    const FieldData<scalar> psiFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_psi");
    const std::vector<scalar>              psiP    = rawInternal(psiFd, nC);
    const std::vector<std::vector<scalar>> psiPBnd = rawBoundary<scalar>(psiFd, patches);

    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;

    cpu::rhoSimple::RhoMomentumInput uin;
    uin.phi = &f.phi.internal;  uin.phiBnd = &f.phi.boundary;
    uin.rho = &rhoP;            uin.rhoBnd = &rhoBnd;
    uin.muEff = &muEff;         uin.muEffBnd = &muEffBnd;
    uin.relaxU = re ? re->scalarOr("U", 1.0) : 1.0;
    uin.relaxEquationU = (re != nullptr) && re->found("U");
    uin.bounded = true;
    uin.scheme = cpu::rhoSimple::DivScheme::upwind;
    uin.correctedLaplacian = true;
    const FvVectorMatrix UEqn = cpu::rhoSimple::assembleUEqn(f.U, uin, m, g, patches);

    // UEqn.H() and constrainHbyA see the JUST-SOLVED velocity, not the one the matrix was built from.
    const FieldData<vector> upFd = readField<vector>(caseDir + "/" + dumpT + "/stage_Upred");
    const std::vector<std::vector<vector>> upBnd = rawBoundary<vector>(upFd, patches);
    if (!upFd.internalUniform && static_cast<label>(upFd.internalField.size()) == nC)
        f.U.internal = upFd.internalField;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) f.U.boundary[pi]->setValue(upBnd[pi]);

    cpu::rhoSimple::PressureInput in;
    in.rho = &rhoP;   in.rhoBnd = &rhoBnd;
    in.psi = &psiP;   in.psiBnd = &psiPBnd;
    in.transonic          = transonic;
    in.relaxP             = re ? re->scalarOr("p", 1.0) : 1.0;
    in.relaxPSpecified    = (re != nullptr) && re->found("p");
    in.pRefCell           = f.pressureControl.refCell;
    in.pRefValue          = f.pressureControl.refValue;
    in.correctedLaplacian = true;

    // ---- 1. The SIMPLEC stages. ----
    std::printf("  1. pcEqn stages\n");
    cpu::rhoSimple::ConsistentPressureStages st =
        cpu::rhoSimple::consistentPressurePredictor(UEqn, f.U, f.p, in, m, g, patches);
    check("branch taken matches the case", st.transonic == transonic);

    const std::vector<scalar> ofRAU  = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_rAU"), nC);
    const std::vector<scalar> ofRAtU = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_rAtU"), nC);
    report("rAU = 1/UEqn.A()", relL2(st.rAU, ofRAU), 1e-10);
    report("rAtU = 1/(1/rAU - UEqn.H1())", relL2(st.rAtU, ofRAtU), 1e-10);

    // rAtU MUST differ from rAU, or the SIMPLEC term is inert and nothing below discriminates.
    {
        double d = 0.0, n = 0.0;
        for (label c = 0; c < nC; ++c)
        {
            const double di = (double)ofRAtU[c] - (double)ofRAU[c];
            d += di*di; n += (double)ofRAU[c]*(double)ofRAU[c];
        }
        check("rAtU actually differs from rAU (SIMPLEC is live)", std::sqrt(d)/std::sqrt(n) > 1e-3);
        std::printf("     %-42s %.4e\n", "  |rAtU - rAU| / |rAU|", std::sqrt(d)/std::sqrt(n));
    }

    const FieldData<scalar> rrFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_rhorAtU");
    report("rhorAtU = rho*rAtU", relL2(st.rhorAtU, rawInternal(rrFd, nC)), 1e-10);

    const FieldData<vector> h0Fd = readField<vector>(caseDir + "/" + dumpT + "/stage_HbyA");
    const FieldData<vector> hcFd = readField<vector>(caseDir + "/" + dumpT + "/stage_HbyAc");
    {
        std::vector<scalar> a0, b0, ac, bc;
        for (label c = 0; c < nC; ++c)
        {
            a0.push_back(st.HbyA0[c].x); a0.push_back(st.HbyA0[c].y); a0.push_back(st.HbyA0[c].z);
            b0.push_back(h0Fd.internalField[c].x); b0.push_back(h0Fd.internalField[c].y); b0.push_back(h0Fd.internalField[c].z);
            ac.push_back(st.HbyA[c].x);  ac.push_back(st.HbyA[c].y);  ac.push_back(st.HbyA[c].z);
            bc.push_back(hcFd.internalField[c].x); bc.push_back(hcFd.internalField[c].y); bc.push_back(hcFd.internalField[c].z);
        }
        report("HbyA before its correction", relL2(a0, b0), 1e-10);
        // Did OpenFOAM's own correction do anything? p is uniform on this fixture, so fvc::grad(p) should
        // be zero and `HbyA -= (rAU - rAtU)*grad(p)` a no-op. If OpenFOAM's two dumps differ, the
        // gradient it computed was NOT zero and that is what to explain.
        std::printf("       OF stage_HbyAc vs stage_HbyA: %.4e  (0 => OF's correction was a no-op)\n",
                    relL2(bc, b0));
        // 1e-8, not the 1e-10 used elsewhere, and the looser bound is reasoned rather than fitted.
        // This is the only comparison here that goes through fvc::grad(p), and p is ~1.1e5 while its
        // gradient comes out of a cancelling face sum, so the summation ORDER matters: brae and OpenFOAM
        // add the faces differently. What was ruled out before accepting that -- ASCII write precision
        // (identical at writePrecision 15 and 17), the gradient scheme (the case sets `gradSchemes
        // default Gauss linear`, which is what brae computes), the boundary correction (OpenFOAM's
        // gaussGrad::correctBoundaryConditions alters the gradient's BOUNDARY field, not the cell values
        // this uses), and a wrong p (identical, read from the same written field). The residue is 1.4e-10
        // against a correction that moves HbyA by 2.4e-01, i.e. ~6e-10 of the term itself. The defect
        // this same comparison DID catch -- inletOutlet's assignable() -- showed at 1.3e-03, five orders
        // above this bound, so it is still a gate and not a formality.
        report("HbyA -= (rAU - rAtU)*grad(p)", relL2(ac, bc), 1e-8);
        {
            // Where does it live? A gradient disagreement on boundary cells points at
            // gaussGrad::correctBoundaryConditions (which replaces the wall-normal component by snGrad);
            // one spread through the interior points at the gradient itself.
            std::vector<char> isB(nC, 0);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                for (label i = 0; i < patches[pi].size; ++i) isB[patches[pi].faceCells[i]] = 1;
            double db = 0.0, di = 0.0; long nb = 0;
            for (label c = 0; c < nC; ++c)
            {
                const double dx = (double)st.HbyA[c].x - (double)hcFd.internalField[c].x;
                const double dy = (double)st.HbyA[c].y - (double)hcFd.internalField[c].y;
                const double dz = (double)st.HbyA[c].z - (double)hcFd.internalField[c].z;
                const double q = dx*dx + dy*dy + dz*dz;
                if (isB[c]) { db += q; ++nb; } else di += q;
            }
            std::printf("       HbyA diff: %ld boundary cells %.4e   %ld interior %.4e\n",
                        nb, std::sqrt(db), (long)nC - nb, std::sqrt(di));
        }
    }

    // HbyA's BOUNDARY, which is where constrainHbyA does its work and which the flux below integrates.
    {
        const std::vector<std::vector<vector>> ofHb = rawBoundary<vector>(h0Fd, patches);
        std::vector<scalar> a, b;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const bool takeU = !f.U.boundary[pi]->assignable();
                const vector v = takeU ? f.U.boundary[pi]->value()[i] : st.HbyA0[c];
                a.push_back(v.x); a.push_back(v.y); a.push_back(v.z);
                b.push_back(ofHb[pi][i].x); b.push_back(ofHb[pi][i].y); b.push_back(ofHb[pi][i].z);
            }
        report("HbyA boundary (constrainHbyA)", relL2(a, b), 1e-10);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            std::printf("       %-14s assignable=%d\n", patches[pi].name.c_str(),
                        (int)f.U.boundary[pi]->assignable());
    }

    const FieldData<scalar> p0Fd = readField<scalar>(caseDir + "/" + dumpT + "/stage_phiHbyA0");
    {
        const std::vector<std::vector<scalar>> ofB = rawBoundary<scalar>(p0Fd, patches);
        double di = 0.0, ni = 0.0, db = 0.0, nb = 0.0;
        for (std::size_t fi = 0; fi < st.phiHbyA0.internal.size(); ++fi)
        {
            const double d = (double)st.phiHbyA0.internal[fi] - (double)p0Fd.internalField[fi];
            di += d*d; ni += (double)p0Fd.internalField[fi]*(double)p0Fd.internalField[fi];
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const double d = (double)st.phiHbyA0.boundary[pi][i] - (double)ofB[pi][i];
                db += d*d; nb += (double)ofB[pi][i]*(double)ofB[pi][i];
            }
        std::printf("       phiHbyA0 internal %.4e   boundary %.4e\n",
                    ni > 0 ? std::sqrt(di/ni) : std::sqrt(di), nb > 0 ? std::sqrt(db/nb) : std::sqrt(db));
    }
    report("phiHbyA = interpolate(rho)*flux(HbyA)",
           relL2Surface(st.phiHbyA0, p0Fd.internalField, rawBoundary<scalar>(p0Fd, patches), patches), 1e-10);

    const FieldData<scalar> pcFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_phiHbyAc");
    const std::vector<std::vector<scalar>> pcB = rawBoundary<scalar>(pcFd, patches);
    if (transonic)
    {
        // The transonic statement applies the SIMPLEC correction AND subtracts the psi*p term together,
        // and psi*p is rho for a perfect gas, so the result is a near-total cancellation. Normalised by
        // the pre-correction flux for the same reason as in the pEqn gate.
        double d = 0.0, n0 = 0.0, nres = 0.0;
        for (std::size_t fi = 0; fi < st.phiHbyA.internal.size(); ++fi)
        {
            const double di = (double)st.phiHbyA.internal[fi] - (double)pcFd.internalField[fi];
            d += di*di;
            n0 += (double)st.phiHbyA0.internal[fi]*(double)st.phiHbyA0.internal[fi];
            nres += (double)pcFd.internalField[fi]*(double)pcFd.internalField[fi];
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const double di = (double)st.phiHbyA.boundary[pi][i] - (double)pcB[pi][i];
                d += di*di;
                n0 += (double)st.phiHbyA0.boundary[pi][i]*(double)st.phiHbyA0.boundary[pi][i];
                nres += (double)pcB[pi][i]*(double)pcB[pi][i];
            }
        std::printf("     %-42s residual %.3e of inflow %.3e\n",
                    "  (SIMPLEC corr + psi*p, cancelling)", std::sqrt(nres), std::sqrt(n0));
        report("phiHbyA after both corrections", std::sqrt(d)/std::sqrt(n0), 1e-10);
        const FieldData<scalar> pdFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_phid");
        report("phid, from the UNCORRECTED phiHbyA",
               relL2Surface(st.phid, pdFd.internalField, rawBoundary<scalar>(pdFd, patches), patches), 1e-10);
        check("closedVolume is never set on this branch", !st.closedVolume);
    }
    else
    {
        report("phiHbyA after adjustPhi + SIMPLEC corr",
               relL2Surface(st.phiHbyA, pcFd.internalField, pcB, patches), 1e-10);
    }

    // ---- 2. The assembled SIMPLEC pressure equation. ----
    std::printf("  2. the assembled pressure equation\n");
    const FvScalarMatrix P = cpu::rhoSimple::assemblePcEqn(st, f.p, in, m, g, patches);
    const std::vector<scalar> ofD = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_pD"), nC);
    const std::vector<scalar> ofS = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_pSrc"), nC);
    report("pcEqn.D() vs OpenFOAM", relL2(matrixD(P, patches), ofD), 1e-10);
    // Also 1e-8: the source carries fvc::div(phiHbyA), and phiHbyA carries the SIMPLEC snGrad(p) term,
    // so it inherits the same gradient round-off reasoned about above. D(), which does not, holds 1e-10.
    report("pcEqn source + boundaryCoeffs vs OpenFOAM", relL2(matrixRhs(P, patches), ofS), 1e-8);

    // ---- 3. CONTROLS. ----
    std::printf("  3. controls\n");
    {
        // (a) The plain-SIMPLE pressure equation must NOT reproduce the SIMPLEC one. This is the whole
        //     point of the file: same case, same fixture, rAU where rAtU belongs and no flux correction.
        cpu::rhoSimple::PressureStages sst =
            cpu::rhoSimple::pressurePredictor(UEqn, f.U, f.p, in, m, g, patches);
        const FvScalarMatrix S = cpu::rhoSimple::assemblePEqn(sst, f.p, in, m, g, patches);
        const double sD = relL2(matrixD(S, patches), ofD);
        const double sS = relL2(matrixRhs(S, patches), ofS);
        check("plain SIMPLE's D disagrees", sD > 1e-8);
        check("plain SIMPLE's source disagrees", sS > 1e-8);
        std::printf("     %-42s D %.4e   src %.4e\n", "  (SIMPLE against the SIMPLEC oracle)", sD, sS);
    }
    {
        // (b) The other transonic branch must not reproduce this one.
        cpu::rhoSimple::PressureInput other = in;
        other.transonic = !transonic;
        cpu::rhoSimple::ConsistentPressureStages ost =
            cpu::rhoSimple::consistentPressurePredictor(UEqn, f.U, f.p, other, m, g, patches);
        const FvScalarMatrix O = cpu::rhoSimple::assemblePcEqn(ost, f.p, other, m, g, patches);
        const double oD = relL2(matrixD(O, patches), ofD);
        check("the other transonic branch disagrees", oD > 1e-8);
        std::printf("     %-42s D %.4e\n", "  (its error, for the record)", oD);
    }

    // ---- 4. REFUSALS. ----
    std::printf("  4. refusals\n");
    {
        // fixedFluxPressure is no longer refused: constrainPressure is PORTED (pcEqn.H:16). The
        // never-updated case refuses from FixedFluxPressurePatchField::requireUpdated instead
        // (test_uniform_function1), and the numbers are gated end-to-end by ffp_vs_openfoam.
    }
    {
        cpu::rhoSimple::PressureInput bad = in;
        bad.hasMRF = true;
        bool threw = false;
        try { (void)cpu::rhoSimple::consistentPressurePredictor(UEqn, f.U, f.p, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a declared MRF is refused", threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
