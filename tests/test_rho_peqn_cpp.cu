// rhoSimpleFoam's pEqn.H against REAL OpenFOAM -- BOTH branches, in one run each.
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
#include "scheme_parse.cuh"
#include "fvOptions_cpp.cuh"   // the case's fvOptions: this gate rebuilds UEqn, so it needs them

#include <cmath>
#include <numeric>
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

    std::printf("rhoSimpleFoam pEqn vs OpenFOAM -- %s branch (%d cells)\n",
                transonic ? "TRANSONIC" : "subsonic", (int)nC);

    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);

    // updateCoeffs() for the FREESTREAM family, at the point OpenFOAM calls it. Every momentum
    // coefficient on such a patch is built from its valueFraction, and rAU = 1/UEqn.A() carries it
    // straight into the pressure equation -- on aerofoilNACA0012 a seeded 0.5 read as rAU 1.08e-02
    // against OpenFOAM's, and everything downstream of rAU inherited it.
    {
        std::vector<std::vector<vector>> Ub(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) Ub[pi] = f.U.boundary[pi]->value();
        updateMixedFreestream(f.U.boundary, Ub, patches);
        updateMixedFreestream(f.p.boundary, Ub, patches);
        f.p.evaluateBoundary();
    }

    // Rebuild UEqn from OpenFOAM's own inputs: rAU and HbyA both come from it.
    const FieldData<scalar> muFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_muEff");
    const std::vector<scalar>              muEff    = rawInternal(muFd, nC);
    const std::vector<std::vector<scalar>> muEffBnd = rawBoundary<scalar>(muFd, patches);
    const FieldData<vector> uaFd = readField<vector>(caseDir + "/" + dumpT + "/stage_Uass");
    const std::vector<std::vector<vector>> uaBnd = rawBoundary<vector>(uaFd, patches);
    if (!uaFd.internalUniform && static_cast<label>(uaFd.internalField.size()) == nC)
        f.U.internal = uaFd.internalField;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) f.U.boundary[pi]->setValue(uaBnd[pi]);

    // rho and psi AS pEqn.H SEES THEM. EEqn.H ends with thermo.correct(), which recomputes T from the
    // just-solved he and psi from that T, so pEqn's psi is one energy solve newer than createFields'.
    // Injecting OpenFOAM's own keeps this gate about pEqn.H rather than about thermo.correct(), which is
    // its own manifest component.
    const FieldData<scalar> psiFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_psi");
    const std::vector<scalar>              psiP    = rawInternal(psiFd, nC);
    const std::vector<std::vector<scalar>> psiPBnd = rawBoundary<scalar>(psiFd, patches);
    const FieldData<scalar> rhoFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_rhoP");
    const std::vector<scalar>              rhoP   = rawInternal(rhoFd, nC);
    const std::vector<std::vector<scalar>> rhoBnd = rawBoundary<scalar>(rhoFd, patches);

    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;

    // THE CASE'S fvOptions, and why a pressure gate needs them at all. This gate rebuilds the momentum
    // matrix to get rAU and H(), and HbyA is rAU*H() -- so an fvOptions term missing from that matrix
    // propagates straight into phiHbyA and the pressure source. angledDuct is the case that showed it:
    // its explicitPorositySource adds V*tr(Cd) to the diagonal across 8000 cells, and without it this
    // gate read rAU 7.49e-01 where the momentum gate, on the identical case and iteration, read
    // 1.99e-15. Every number downstream of that was noise, and none of it was brae's.
    cpu::fvOptions::OptionList popts = cpu::fvOptions::read(caseDir, m);
    // REFUSE rather than assemble a different equation. Silently dropping an unimplemented option is
    // exactly the substitution this port exists to catch -- and it would report the gap as a solver
    // defect, which is worse than not running.
    if (!popts.firstUnsupported().empty())
    {
        std::printf("  REFUSED: fvOptions declares '%s', which is not ported. This gate would otherwise\n"
                    "           assemble a momentum matrix the case does not have.\n",
                    popts.firstUnsupported().c_str());
        return 1;
    }
    // pEqn.H applies fvOptions to the PRESSURE equation too -- `fvOptions(psi, p, rho.name())` at
    // pEqn.H:32 and :63. Nothing in angledDuct targets p, and this gate does not model that term, so it
    // refuses if a case ever does rather than quietly leaving it out.
    for (const auto& o : popts.options)
    {
        if (!o.active) continue;
        for (const auto& fv : o.fieldValues)
        {
            if (fv.first == "p" || fv.first == "rho")
            {
                std::printf("  REFUSED: fvOptions '%s' constrains '%s'; pEqn.H applies fvOptions to the\n"
                            "           pressure equation and this gate does not model that term.\n",
                            o.name.c_str(), fv.first.c_str());
                return 1;
            }
        }
    }

    cpu::rhoSimple::RhoMomentumInput uin;
    uin.phi = &f.phi.internal;  uin.phiBnd = &f.phi.boundary;
    uin.rho = &rhoP;            uin.rhoBnd = &rhoBnd;
    uin.muEff = &muEff;         uin.muEffBnd = &muEffBnd;
    uin.relaxU = re ? re->scalarOr("U", 1.0) : 1.0;
    uin.relaxEquationU = (re != nullptr) && re->found("U");
    // PARSED from the case. This gate rebuilds the momentum matrix to get rAU and H(), and HbyA is
    // rAU*H() -- so a momentum matrix assembled under the wrong scheme propagates straight into the
    // pressure equation. Stating sbMatched's schemes here read HbyA 9.1e-04 on aerofoilNACA0012, whose
    // div(phi,U) is `bounded Gauss linearUpwind limited` and whose grad(U) is `cellLimited Gauss linear 1`.
    {
        const FieldDivScheme dU = parseFieldDivScheme(caseDir, "U");
        uin.bounded = dU.bounded;
        uin.scheme  = dU.linearUpwind ? cpu::rhoSimple::DivScheme::linearUpwind
                    : (dU.limited     ? cpu::rhoSimple::DivScheme::limitedLinear
                                      : cpu::rhoSimple::DivScheme::upwind);
        uin.schemeCoeff = dU.coeff;   // RAW k: the weights functions compute twoByk themselves (scheme_parse.cuh)
        DeviceSimpleControls sctl;
        parseFvSchemesControls(caseDir, sctl);
        uin.correctedLaplacian = sctl.nonOrth;
        uin.gradULimitK        = sctl.gradULimitK;
        std::printf("  schemes: div(phi,U) lu=%d bounded=%d | grad(U) cellLimited k=%g\n",
                    (int)dU.linearUpwind, (int)dU.bounded, (double)uin.gradULimitK);
    }
    // updateCoeffs() ON U BEFORE THE MATRIX IS BUILT. fvMatrix's constructor calls
    // psi.boundaryFieldRef().updateCoeffs(), and that is where flowRateInletVelocity recomputes its
    // velocity from rho's PATCH values (avgU = -flowRate/gSum(rho*magSf)) and where inletOutlet reads
    // sign(phi). This gate rebuilt the momentum matrix without either, so its inlet carried a velocity
    // derived from whatever rho createFields left there -- and HbyA, which is rAU*UEqn.H(), came out
    // 8.52e-05 on the inlet-adjacent cells while its deep interior was 1.20e-10. The momentum gate does
    // all three; this one did only the freestream half.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        f.U.boundary[pi]->updateFromDensity(rhoBnd[pi]);
        f.U.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
    }
    f.U.evaluateBoundary();

    if (!popts.empty())
    {
        uin.fvOpts = &popts;
        std::printf("  fvOptions: %zu option(s) applied to the momentum matrix\n", popts.options.size());
        for (const auto& o : popts.options)
            std::printf("     %-24s type=%-26s cells=%zu fixedCoeff=%d tr(alpha)=%g\n",
                        o.name.c_str(), o.type.c_str(), o.cells.size(), (int)o.fixedCoeff,
                        (double)(o.alpha.xx + o.alpha.yy + o.alpha.zz));
    }
    const FvVectorMatrix UEqn = cpu::rhoSimple::assembleUEqn(f.U, uin, m, g, patches);

    // TWO DIFFERENT U's, and pEqn.H uses the second. The momentum matrix is assembled from the velocity
    // the iteration started with (stage_Uass), but `solve(UEqn == -fvc::grad(p))` runs before pEqn.H, so
    // UEqn.H() -- and constrainHbyA, which reads U's boundary -- see the JUST-SOLVED velocity
    // (stage_Upred). Using the assembly-time U for H() leaves HbyA wrong by the whole momentum
    // predictor, which then propagates into phiHbyA, phid and the pressure source.
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

    // ---- 1. The stages, in OpenFOAM's order. ----
    std::printf("  1. pEqn stages\n");
    cpu::rhoSimple::PressureStages st =
        cpu::rhoSimple::pressurePredictor(UEqn, f.U, f.p, in, m, g, patches);
    check("branch taken matches the case", st.transonic == transonic);

    const std::vector<scalar> ofRAU = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_rAU"), nC);
    report("rAU = 1/UEqn.A()", relL2(st.rAU, ofRAU), 1e-10);

    const FieldData<scalar> rrFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_rhorAUf");
    report("rhorAUf = interpolate(rho*rAU)",
           relL2Surface(st.rhorAUf, rrFd.internalField, rawBoundary<scalar>(rrFd, patches), patches), 1e-10);

    const FieldData<vector> hFd = readField<vector>(caseDir + "/" + dumpT + "/stage_HbyA");
    {
        std::vector<scalar> a, b;
        for (label c = 0; c < nC; ++c)
        {
            a.push_back(st.HbyA[c].x); a.push_back(st.HbyA[c].y); a.push_back(st.HbyA[c].z);
            b.push_back(hFd.internalField[c].x); b.push_back(hFd.internalField[c].y); b.push_back(hFd.internalField[c].z);
        }
        report("HbyA = constrainHbyA(rAU*UEqn.H())", relL2(a, b), 1e-10);
        // WHERE it lives. HbyA = rAU*H() in the cells and U on any patch whose BC is not assignable, so a
        // disagreement confined to cells touching a patch is the constrain step or the boundary
        // coefficients H() folds in, and one spread through the interior is H() itself.
        {
            std::vector<char> onB(nC, 0);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                for (label i = 0; i < patches[pi].size; ++i)
                    if (patches[pi].type != "empty") onB[patches[pi].faceCells[i]] = 1;
            std::vector<scalar> ai, bi, ab, bb;
            for (label c = 0; c < nC; ++c)
            {
                const scalar mine[3] = { st.HbyA[c].x, st.HbyA[c].y, st.HbyA[c].z };
                const scalar theirs[3] = { hFd.internalField[c].x, hFd.internalField[c].y,
                                           hFd.internalField[c].z };
                for (int kk = 0; kk < 3; ++kk)
                {
                    if (onB[c]) { ab.push_back(mine[kk]); bb.push_back(theirs[kk]); }
                    else        { ai.push_back(mine[kk]); bi.push_back(theirs[kk]); }
                }
            }
            std::printf("       %-40s interior %.6e   touching a patch %.6e\n", "HbyA split",
                        relL2(ai, bi), relL2(ab, bb));
            // and per patch, so the offending boundary condition names itself
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                if (patches[pi].type == "empty" || !patches[pi].size) continue;
                std::vector<scalar> pa, pb;
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const label c = patches[pi].faceCells[i];
                    pa.push_back(st.HbyA[c].x); pa.push_back(st.HbyA[c].y); pa.push_back(st.HbyA[c].z);
                    pb.push_back(hFd.internalField[c].x); pb.push_back(hFd.internalField[c].y);
                    pb.push_back(hFd.internalField[c].z);
                }
                std::printf("         %-24s %.6e   (%s)\n", patches[pi].name.c_str(), relL2(pa, pb),
                            f.U.boundary[pi]->assignable() ? "assignable" : "NOT assignable");
            }
        }
    }

    const FieldData<scalar> h0Fd = readField<scalar>(caseDir + "/" + dumpT + "/stage_phiHbyA0");
    report("phiHbyA = interpolate(rho)*flux(HbyA)",
           relL2Surface(st.phiHbyA0, h0Fd.internalField, rawBoundary<scalar>(h0Fd, patches), patches), 1e-10);

    // The branch's own treatment of phiHbyA: the psi*p subtraction, or adjustPhi.
    const FieldData<scalar> hFd2 = readField<scalar>(caseDir + "/" + dumpT + "/stage_phiHbyA");
    if (transonic)
    {
        // A NEAR-TOTAL CANCELLATION, so it is normalised by the PRE-subtraction flux. psi*p is rho for a
        // perfect gas, so `phiHbyA -= interpolate(psi*p)*phiHbyA/interpolate(rho)` removes essentially all
        // of phiHbyA; what remains is a small residual, and dividing the difference by that residual would
        // report noise as a large relative error. Measured against phiHbyA0 the number says what it should:
        // how big the disagreement is compared with the flux that went in.
        const std::vector<std::vector<scalar>> ofB = rawBoundary<scalar>(hFd2, patches);
        double d = 0.0, n0 = 0.0, nres = 0.0;
        for (std::size_t fi = 0; fi < st.phiHbyA.internal.size(); ++fi)
        {
            const double di = (double)st.phiHbyA.internal[fi] - (double)hFd2.internalField[fi];
            d += di*di;
            n0 += (double)st.phiHbyA0.internal[fi]*(double)st.phiHbyA0.internal[fi];
            nres += (double)hFd2.internalField[fi]*(double)hFd2.internalField[fi];
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const double di = (double)st.phiHbyA.boundary[pi][i] - (double)ofB[pi][i];
                d += di*di;
                n0 += (double)st.phiHbyA0.boundary[pi][i]*(double)st.phiHbyA0.boundary[pi][i];
                nres += (double)ofB[pi][i]*(double)ofB[pi][i];
            }
        std::printf("     %-42s residual %.3e of inflow %.3e\n",
                    "  (the psi*p subtraction cancels)", std::sqrt(nres), std::sqrt(n0));
        report("phiHbyA after the psi*p subtraction", std::sqrt(d)/std::sqrt(n0), 1e-10);
    }
    else
    {
        report("phiHbyA after adjustPhi",
               relL2Surface(st.phiHbyA, hFd2.internalField, rawBoundary<scalar>(hFd2, patches), patches), 1e-10);
    }

    if (transonic)
    {
        const FieldData<scalar> pdFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_phid");
        report("phid = (interp(psi)/interp(rho))*phiHbyA",
               relL2Surface(st.phid, pdFd.internalField, rawBoundary<scalar>(pdFd, patches), patches), 1e-10);
        check("closedVolume is never set on this branch", !st.closedVolume);
    }

    // ---- 2. The assembled pressure equation. ----
    std::printf("  2. the assembled pressure equation\n");
    const FvScalarMatrix P = cpu::rhoSimple::assemblePEqn(st, f.p, in, m, g, patches);
    const std::string sfx = transonic ? "t" : "";
    const std::vector<scalar> ofD =
        rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_pD" + sfx), nC);
    const std::vector<scalar> ofS =
        rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_pSrc" + sfx), nC);
    report("pEqn.D() vs OpenFOAM", relL2(matrixD(P, patches), ofD), 1e-10);
    report("pEqn source + boundaryCoeffs vs OpenFOAM", relL2(matrixRhs(P, patches), ofS), 1e-10);
    {
        // WITH MAGNITUDES, and split by where it lives. The pressure source is div(phiHbyA), which is the
        // CONTINUITY ERROR -- near zero at convergence, so a relative norm over it can report a large
        // number for a negligible absolute difference. Two readings this session were exactly that.
        const std::vector<scalar> mine = matrixRhs(P, patches);
        auto mg = [](const std::vector<scalar>& v)
        { return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0)); };
        std::vector<char> onB(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            if (patches[pi].type != "empty")
                for (label i = 0; i < patches[pi].size; ++i) onB[patches[pi].faceCells[i]] = 1;
        std::vector<scalar> ai, bi, ab, bb;
        for (label c = 0; c < nC; ++c)
        {
            if (onB[c]) { ab.push_back(mine[c]); bb.push_back(ofS[c]); }
            else        { ai.push_back(mine[c]); bi.push_back(ofS[c]); }
        }
        std::printf("       |brae| %.4e |OF| %.4e   interior %.6e (|%.3e|)   "
                    "touching a patch %.6e (|%.3e|)\n",
                    mg(mine), mg(ofS), relL2(ai, bi), mg(bi), relL2(ab, bb), mg(bb));
        // Per patch: the pressure source at a boundary cell is div(phiHbyA) plus the laplacian's
        // boundaryCoeffs, and those come from p's own BC -- freestreamPressure here. Naming the patch
        // says which boundary condition to read.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (patches[pi].type == "empty" || !patches[pi].size) continue;
            std::vector<scalar> pa, pb;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                pa.push_back(mine[c]); pb.push_back(ofS[c]);
            }
            std::printf("         pEqn src on %-14s %.6e   (|OF| %.3e)\n",
                        patches[pi].name.c_str(), relL2(pa, pb), mg(pb));
        }
    }

    // ---- 3. THE CONTROL: the OTHER branch must not reproduce this one. ----
    std::printf("  3. control -- the other branch must NOT match\n");
    cpu::rhoSimple::PressureInput other = in;
    other.transonic = !transonic;
    cpu::rhoSimple::PressureStages ost =
        cpu::rhoSimple::pressurePredictor(UEqn, f.U, f.p, other, m, g, patches);
    const FvScalarMatrix O = cpu::rhoSimple::assemblePEqn(ost, f.p, other, m, g, patches);
    const double oD = relL2(matrixD(O, patches), ofD);
    const double oS = relL2(matrixRhs(O, patches), ofS);
    // The bar is 1e-8, not 1e-6, and the reason is physical rather than a concession: phid is
    // (psi/rho)*phiHbyA = phiHbyA/p, and p is ~1.1e5 here, so `fvm::div(phid, p)` contributes a diagonal
    // term ~1e-5 of the laplacian's. The branches' MATRICES therefore differ only slightly at this state
    // even though their phiHbyA differ completely -- which is exactly why phid and phiHbyA are gated
    // directly above rather than only through the assembled system.
    check("the other branch's D disagrees", oD > 1e-8);
    check("the other branch's source disagrees", oS > 1e-6);
    std::printf("     %-42s D %.4e   src %.4e\n", "  (its errors, for the record)", oD, oS);

    // ---- 4. REFUSALS. ----
    std::printf("  4. refusals\n");
    {
        // fixedFluxPressure is no longer refused here: constrainPressure is PORTED (pEqn.H:12
        // transcribed above the transonic branch), and the never-updated case now refuses from the
        // patch itself (FixedFluxPressurePatchField::requireUpdated, gated in test_uniform_function1;
        // the flux-consistency numbers are gated end-to-end by ffp_vs_openfoam).
    }
    {
        cpu::rhoSimple::PressureInput bad = in;
        bad.hasMRF = true;
        bool threw = false;
        try { (void)cpu::rhoSimple::pressurePredictor(UEqn, f.U, f.p, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a declared MRF is refused", threw);
    }
    {
        cpu::rhoSimple::PressureInput bad = in;
        bad.transonic = true; bad.psi = nullptr; bad.psiBnd = nullptr;
        bool threw = false;
        try { (void)cpu::rhoSimple::pressurePredictor(UEqn, f.U, f.p, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("transonic without psi is refused", threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
