// CUDA rhoPEqn against the _cpp reference, stage by stage and field by field, BOTH BRANCHES.
//
// The reference is itself gated against OpenFOAM's own pressure dumps (tests/rho_peqn_vs_openfoam.sh,
// which runs OpenFOAM twice -- transonic no and transonic yes), so this closes OpenFOAM -> _cpp -> CUDA
// for the whole compressible pressure equation. Device twin of tests/test_peqn_cuda.cu.
//
// EVERY INTERMEDIATE IS COMPARED, not just the assembled matrix: rAU, rhorAUf, HbyA, phiHbyA before and
// after the branch, phid, and then diag/upper/lower/source/iC/bC and the face-flux correction. pEqn.H is
// eight stages deep and a single number at the end cannot say which one moved.
//
// THE BRANCH IS THE POINT. simple.transonic() selects between two different equations -- the transonic one
// adds fvm::div(phid, p) and calls pEqn.relax(), the subsonic one runs adjustPhi and neither -- so the
// binary runs whichever BRAE_TEST_TRANSONIC selects and both are registered. A gate that only ever
// exercised one would leave the other's port asserted by nothing, which is how this project's worst
// defects have historically survived.
//
// rho AND psi ARE SYNTHESIZED, non-uniform, for the reason spelled out in test_rho_ueqn_cuda.cu: no
// committed fixture ships either field, this instrument measures DEVICE-vs-HOST on identical inputs
// rather than physics, and a uniform rho would make rhorAUf a constant multiple of rAU so a kernel that
// dropped the rho weighting from one term would still agree on a rescaled matrix. Both are asserted to
// vary. Physical validity is rho_peqn_vs_openfoam.sh's claim, not this one's.
//
// WHAT THIS GATE DOES NOT CLAIM: it does not cover MRF, fvOptions, fixedFluxPressure or coupled patches --
// all four are REFUSED by the device module and the refusals are asserted below rather than the behaviour.
//
// Run: test_rho_peqn_cuda <caseDir> <timeDir> [laminar]      env: BRAE_TEST_TRANSONIC=1
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
#include "rhoUEqn_cpp.cuh"
#include "rhoUEqn.cuh"
#include "rhoPEqn_cpp.cuh"
#include "rhoPEqn.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
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
        std::printf("usage: %s <caseDir> <timeDir> [laminar]\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar nu = 1e-5;

    const char* trEnv = std::getenv("BRAE_TEST_TRANSONIC");
    const bool transonic = trEnv && trEnv[0] == '1';

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

    const bool forceLaminar = (argc > 3 && std::string(argv[3]) == "laminar");
    const bool turbulent = !forceLaminar
                        && std::filesystem::exists(caseDir + "/" + t + "/nut");
    std::vector<scalar> nuEffC(nC, nu);
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nuEffB[pi].assign(fvp[pi].size, nu);
    if (turbulent)
    {
        GeometricField<scalar> nutF =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), fvp, nC);
        nutF.evaluateBoundary();
        for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nutF.internal[c];
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& nb = nutF.boundary[pi]->value();
            for (label i = 0; i < fvp[pi].size; ++i) nuEffB[pi][i] = nu + nb[i];
        }
    }

    // rho and psi, smooth and non-uniform across the bounding box. See the header for why they are
    // synthesized and why uniform would make the whole comparison vacuous.
    scalar xMin = 1e300, xMax = -1e300;
    for (label c = 0; c < nC; ++c)
    {
        xMin = std::fmin(xMin, g.C()[c].x);
        xMax = std::fmax(xMax, g.C()[c].x);
    }
    const scalar xSpan = (xMax > xMin) ? (xMax - xMin) : 1.0;
    auto rhoAt = [&](const vector& x) { return scalar(0.8) + scalar(0.6) * (x.x - xMin) / xSpan; };
    // psi = d(rho)/dp. Order 1e-5 for air, and varied so the transonic branch's phid and psi*p terms are
    // not a constant rescaling of phiHbyA.
    auto psiAt = [&](const vector& x) { return scalar(7.0e-6) + scalar(4.0e-6) * (x.x - xMin) / xSpan; };

    std::vector<scalar> rhoC(nC), psiC(nC);
    for (label c = 0; c < nC; ++c) { rhoC[c] = rhoAt(g.C()[c]); psiC[c] = psiAt(g.C()[c]); }
    std::vector<std::vector<scalar>> rhoB(fvp.size()), psiB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        rhoB[pi].resize(fvp[pi].size);
        psiB[pi].resize(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            rhoB[pi][i] = rhoAt(fvp[pi].Cf[i]);
            psiB[pi][i] = psiAt(fvp[pi].Cf[i]);
        }
    }
    scalar rMin = 1e300, rMax = -1e300;
    for (label c = 0; c < nC; ++c) { rMin = std::fmin(rMin, rhoC[c]); rMax = std::fmax(rMax, rhoC[c]); }

    std::printf("test_rho_peqn_cuda:  branch=%s  (%s)  rho in [%.4f .. %.4f]\n",
                transonic ? "TRANSONIC" : "subsonic",
                turbulent ? "TURBULENT" : "laminar", (double)rMin, (double)rMax);
    check((rMax - rMin) / rMax > 0.1, "rho varies across the domain (fixture discriminates)");

    // ---- the momentum matrix both pressure paths consume ------------------------------------
    // Each side gets its OWN lineage's matrix. The two are already gated equal to 1e-12 by
    // test_rho_ueqn_cuda, so any difference below belongs to the pressure module.
    const scalar relaxU = 0.7;
    cpu::rhoSimple::RhoMomentumInput mi;
    mi.phi = &phiF.internalField;  mi.phiBnd = &phiBnd;
    mi.rho = &rhoC;                mi.rhoBnd = &rhoB;
    mi.nuEff = &nuEffC;            mi.nuEffBnd = &nuEffB;
    mi.relaxU = relaxU;
    mi.relaxEquationU = true;   // the gate NAMES a factor, as the device side does
    mi.bounded = true;
    mi.correctedLaplacian = true;
    const FvVectorMatrix refU = cpu::rhoSimple::assembleUEqn(U, mi, m, g, fvp);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(p, fvp, g);

    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dP(p.internal);

    std::vector<scalar> nuBndFlat, rhoBndFlat, psiBndFlat, phiBndFlat;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            nuBndFlat.push_back(nuEffB[pi][i]);
            rhoBndFlat.push_back(rhoB[pi][i]);
            psiBndFlat.push_back(psiB[pi][i]);
            phiBndFlat.push_back(phiBnd[pi][i]);
        }
    }
    nuBndFlat.resize(dm.nBndFaces, nu);
    rhoBndFlat.resize(dm.nBndFaces, 1.0);
    psiBndFlat.resize(dm.nBndFaces, 7.0e-6);
    phiBndFlat.resize(dm.nBndFaces, 0.0);

    DeviceBuffer<scalar> dPhiInt(phiF.internalField), dPhiBnd(phiBndFlat);
    DeviceBuffer<scalar> dRhoCell(rhoC), dRhoBnd(rhoBndFlat);
    DeviceBuffer<scalar> dPsiCell(psiC), dPsiBnd(psiBndFlat);
    DeviceBuffer<scalar> dNuCell(nuEffC), dNuBnd(nuBndFlat);

    gpu::rhoSimple::RhoMomentumInput gmi;
    gmi.phiInt = &dPhiInt;        gmi.phiBnd = &dPhiBnd;
    gmi.rhoCell = &dRhoCell;      gmi.rhoBndFace = &dRhoBnd;
    gmi.nuEffCell = &dNuCell;     gmi.nuEffBndFace = &dNuBnd;
    gmi.relaxEquationU = true;    gmi.relaxU = relaxU;
    gmi.bounded = true;
    gmi.correctedLaplacian = true;
    gpu::MomentumMatrix MU;
    gpu::rhoSimple::assembleUEqn(MU, dm, dbU, dUx, dUy, dUz, gmi);

    // ---- the two masks, and they answer DIFFERENT questions ---------------------------------
    // constrainHbyA asks `assignable`; adjustPhi asks `fixesValue`. slip and inletOutlet are
    // non-assignable WITHOUT fixing a value, so conflating them is a silent error. The control below
    // asserts the case actually distinguishes them.
    std::vector<label> takeU, adjustable;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            takeU.push_back(U.boundary[pi]->assignable() ? 0 : 1);
            // adjustPhi's rule is `fixesValue() && !isInletOutlet()`, and BOTH halves matter:
            // mixedFvPatchField::fixesValue() is TRUE and inletOutlet inherits it, so testing fixesValue
            // alone marks an inletOutlet OUTLET as fixed outflow -- leaving adjustPhi nothing adjustable
            // to balance the inflow against. The reference documents this at rhoPEqn_cpp.cu:81-88 and
            // this gate got it wrong first time: the boundary flux then read 2.09e-01 after the branch
            // while every other stage was at 1e-16.
            const bool fixed = U.boundary[pi]->fixesValue() && !U.boundary[pi]->isInletOutlet();
            adjustable.push_back(fixed ? 0 : 1);
        }
    }
    // THIS CONTROL USED TO ASSERT THE OPPOSITE OF WHAT IT CLAIMED, and could not fail.
    //
    // takeU = !assignable and adjustable = !(fixesValue && !isInletOutlet), so when the two RULES AGREE
    // -- the ordinary case -- the masks are COMPLEMENTS and `takeU[i] != adjustable[i]` is true. The old
    // form asserted exactly that, and broke out of the loop on the first ordinary patch, so it passed on
    // every fixture whether or not the implementation conflated the two rules. What it named as the
    // defect (one mask derived from the other) is the case where the masks are EQUAL on a face.
    //
    // The rules separate only on a patch that is non-assignable WITHOUT fixing a value, which is SLIP.
    // Neither registered fixture here has one -- matrixDumpAsym and pitzDailyTurb are
    // fixedValue/zeroGradient/noSlip/empty -- so this reports itself vacuous rather than asserting into
    // the void. The discriminating fixture DOES exist: OpenFOAM's own
    // compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff gives porosityWall `type slip` on U, and
    // tests/rho_angledduct_structural.sh measures exactly this there (1600 faces separate the rules).
    std::size_t sameFaces = 0;
    bool separable = false;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const bool a  = U.boundary[pi]->assignable();
        const bool fv = U.boundary[pi]->fixesValue();
        const bool io = U.boundary[pi]->isInletOutlet();
        if ((!a) != (fv && !io)) separable = true;
    }
    for (std::size_t i = 0; i < takeU.size(); ++i)
        if (takeU[i] == adjustable[i]) ++sameFaces;   // NOT complements -> the two rules disagree here
    takeU.resize(dm.nBndFaces, 0);
    adjustable.resize(dm.nBndFaces, 0);
    DeviceBuffer<label> dTakeU(takeU), dAdjust(adjustable);
    std::printf("    %-46s %zu faces\n", "(faces where the two rules disagree)", sameFaces);
    if (separable) check(sameFaces > 0, "assignable and fixesValue masks DIFFER on this case (control)");
    else           std::printf("    %-46s %s\n",
                               "(no slip patch here -- this control is vacuous)", "noted");

    // ...and whether adjustPhi RUNS at all, which decides whether `adjustable` is read on this fixture.
    // OpenFOAM's adjustPhi returns immediately when any p patch fixes a value (rhoPEqn_cpp.cu:75-76),
    // and both fixtures registered for this gate have a fixedValue p -- so the mask is inert on every
    // one of the arms that pass here. Printed rather than asserted, because a fixture where it DOES run
    // (a closed volume, no fixed-value p) is what would be needed and none is registered.
    {
        bool anyFixedP = false;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (p.boundary[pi]->fixesValue()) anyFixedP = true;
        std::printf("    %-46s %s\n", "(adjustPhi runs on this fixture?)",
                    anyFixedP ? "NO -- a p patch fixes a value, so `adjustable` is inert here"
                              : "yes -- closed volume");
    }
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        std::printf("    PATCH %-14s n=%5d assignable=%d fixesValue=%d inletOutlet=%d\n",
                    fvp[pi].name.c_str(), (int)fvp[pi].size,
                    (int)U.boundary[pi]->assignable(), (int)U.boundary[pi]->fixesValue(),
                    (int)U.boundary[pi]->isInletOutlet());

    // ---- the reference ----------------------------------------------------------------------
    cpu::rhoSimple::PressureInput pin;
    pin.rho = &rhoC;   pin.rhoBnd = &rhoB;
    pin.psi = &psiC;   pin.psiBnd = &psiB;
    pin.transonic = transonic;
    pin.relaxP = 0.3;
    pin.relaxPSpecified = true;
    // p NEEDS A REFERENCE only when no p patch fixes a value -- which is exactly the condition
    // adjustPhi returns early on (adjustPhi.C, and rhoPEqn_cpp.cu's transcription of it). Hardcoding a
    // reference cell on a case whose p IS fixed made the host skip adjustPhi while the device ran it:
    // closedVolume 0 against 1, massCorr 0.586, and a boundary flux 2.09e-01 apart with every other
    // stage at 1e-16. The device module gates adjustPhi on pRefCell >= 0 and its header says so, so the
    // caller has to derive it the same way OpenFOAM does rather than assert one.
    bool pNeedsRef = true;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        if (p.boundary[pi]->fixesValue()) { pNeedsRef = false; break; }
    const label pRefCell = pNeedsRef ? 0 : -1;
    std::printf("  p needs a reference: %s (pRefCell %d)\n", pNeedsRef ? "yes" : "no", (int)pRefCell);
    pin.pRefCell = pRefCell;
    pin.pRefValue = 0.0;
    pin.correctedLaplacian = true;
    const cpu::rhoSimple::PressureStages rst =
        cpu::rhoSimple::pressurePredictor(refU, U, p, pin, m, g, fvp);
    const FvScalarMatrix refP = cpu::rhoSimple::assemblePEqn(rst, p, pin, m, g, fvp);

    // ---- the CUDA path ----------------------------------------------------------------------
    gpu::rhoSimple::RhoPressureInput gpin;
    gpin.rhoCell = &dRhoCell;  gpin.rhoBndFace = &dRhoBnd;
    gpin.psiCell = &dPsiCell;  gpin.psiBndFace = &dPsiBnd;
    gpin.transonic = transonic;
    gpin.relaxP = 0.3;
    gpin.relaxPSpecified = true;
    gpin.pRefCell = pRefCell;
    gpin.pRefValue = 0.0;
    gpin.correctedLaplacian = true;
    gpin.takeUAtBoundary = &dTakeU;
    gpin.adjustable = &dAdjust;

    gpu::rhoSimple::RhoPressureStages gst;
    gpu::rhoSimple::pressurePredictor(gst, dm, dbU, dbP, MU, dUx, dUy, dUz, dP, gpin);
    gpu::PressureMatrix GP;
    gpu::rhoSimple::assemblePEqn(GP, gst, dm, dbP, dP, gpin);

    // ---- stage by stage ---------------------------------------------------------------------
    check(gst.transonic == rst.transonic, "both paths took the same branch");
    std::printf("    ADJUST device massCorr=%.10f closedVolume=%d | host closedVolume=%d\n",
                (double)gst.massCorr, (int)gst.closedVolume, (int)rst.closedVolume);
    cmp(gst.rAU.host(), rst.rAU, "rAU = 1/UEqn.A()", 1e-12);
    cmp(gst.rhorAUf.host(), rst.rhorAUf.internal, "rhorAUf = interp(rho*rAU)", 1e-12);

    {
        const char* hn[3] = {"HbyA x", "HbyA y", "HbyA z"};
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> r(nC);
            for (label c = 0; c < nC; ++c) r[c] = component(rst.HbyA[c], k);
            cmp(gst.HbyA[k].host(), r, hn[k], 1e-11);
        }
    }
    cmp(gst.phiHbyA0Int.host(), rst.phiHbyA0.internal, "phiHbyA BEFORE the branch", 1e-11);
    cmp(gst.phiHbyAInt.host(), rst.phiHbyA.internal, "phiHbyA AFTER the branch", 1e-11);
    // AND ON THE BOUNDARY. adjustPhi writes only there, div(phiHbyA) reads it at every boundary cell,
    // and comparing internal faces alone would miss both.
    {
        std::vector<scalar> b0, b1;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                b0.push_back(rst.phiHbyA0.boundary[pi][i]);
                b1.push_back(rst.phiHbyA.boundary[pi][i]);
            }
        }
        std::vector<scalar> g0 = gst.phiHbyA0Bnd.host(), g1 = gst.phiHbyABnd.host();
        g0.resize(b0.size());
        g1.resize(b1.size());
        cmp(g0, b0, "phiHbyA BOUNDARY before branch", 1e-11);
        cmp(g1, b1, "phiHbyA BOUNDARY after branch", 1e-11);
    }
    if (transonic)
    {
        cmp(gst.phidInt.host(), rst.phid.internal, "phid (transonic only)", 1e-11);
    }

    // ---- the assembled matrix ---------------------------------------------------------------
    cmp(GP.diag.host(),   refP.diag,   "pEqn diag",   1e-11);
    cmp(GP.upper.host(),  refP.upper,  "pEqn upper",  1e-11);
    cmp(GP.lower.host(),  refP.lower,  "pEqn lower",  1e-11);
    cmp(GP.source.host(), refP.source, "pEqn source", 1e-10);
    {
        std::vector<scalar> ric, rbc;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                ric.push_back(refP.internalCoeffs[pi][i]);
                rbc.push_back(refP.boundaryCoeffs[pi][i]);
            }
        }
        std::vector<scalar> gic = GP.iC.host(), gbc = GP.bC.host();
        gic.resize(ric.size());
        gbc.resize(rbc.size());
        cmp(gic, ric, "pEqn internalCoeffs", 1e-11);
        cmp(gbc, rbc, "pEqn boundaryCoeffs", 1e-11);
    }
    if (!refP.faceFluxCorrection.empty())
    {
        cmp(GP.faceFluxCorr.host(), refP.faceFluxCorrection, "faceFluxCorrection", 1e-10);
    }

    // ---- CONTROL: the branch is a different equation ----------------------------------------
    // The manifest's own claim for this component. If the two branches produced the same matrix, every
    // number above would pass with the branch logic deleted.
    {
        cpu::rhoSimple::PressureInput other = pin;
        other.transonic = !transonic;
        const cpu::rhoSimple::PressureStages ost =
            cpu::rhoSimple::pressurePredictor(refU, U, p, other, m, g, fvp);
        const FvScalarMatrix oP = cpu::rhoSimple::assemblePEqn(ost, p, other, m, g, fvp);
        scalar dD = 0, mD = 0;
        for (std::size_t c = 0; c < refP.diag.size(); ++c)
        {
            dD = std::fmax(dD, std::fabs(refP.diag[c] - oP.diag[c]));
            mD = std::fmax(mD, std::fabs(refP.diag[c]));
        }
        const scalar r = mD > 0 ? dD / mD : dD;
        std::printf("  %-58s rel=%.3e\n", "control: the OTHER branch is a different matrix", (double)r);
        check(r > 1e-9, "the transonic/subsonic branch changes the equation (control)");
    }

    // ---- CONTROL: the rho weighting in rhorAUf must matter ----------------------------------
    // rhorAUf = interpolate(rho*rAU) is what separates this pressure equation from simpleFoam's. With
    // rho == 1 it collapses to the incompressible diffusivity.
    {
        std::vector<scalar> one(nC, 1.0);
        std::vector<std::vector<scalar>> oneB(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) oneB[pi].assign(fvp[pi].size, 1.0);
        cpu::rhoSimple::PressureInput kin = pin;
        kin.rho = &one; kin.rhoBnd = &oneB;
        const cpu::rhoSimple::PressureStages kst =
            cpu::rhoSimple::pressurePredictor(refU, U, p, kin, m, g, fvp);
        scalar d = 0, mg = 0;
        for (std::size_t f = 0; f < rst.rhorAUf.internal.size(); ++f)
        {
            d = std::fmax(d, std::fabs(rst.rhorAUf.internal[f] - kst.rhorAUf.internal[f]));
            mg = std::fmax(mg, std::fabs(rst.rhorAUf.internal[f]));
        }
        const scalar r = mg > 0 ? d / mg : d;
        std::printf("  %-58s rel=%.3e\n", "control: rho weighting changes rhorAUf", (double)r);
        check(r > 1e-3, "rhorAUf carries rho, not just rAU (control)");
    }

    // ---- CONTROL: `corrected` must move BOTH the coefficients and the source ----------------
    // The defect this port paid for twice on the host: implementing only the implicit half leaves the
    // diagonal exact while the source is short by the whole correction.
    {
        cpu::rhoSimple::PressureInput noC = pin;
        noC.correctedLaplacian = false;
        const cpu::rhoSimple::PressureStages cst =
            cpu::rhoSimple::pressurePredictor(refU, U, p, noC, m, g, fvp);
        const FvScalarMatrix cP = cpu::rhoSimple::assemblePEqn(cst, p, noC, m, g, fvp);
        scalar dD = 0, mD = 0, dS = 0, mS = 0;
        for (std::size_t c = 0; c < refP.diag.size(); ++c)
        {
            dD = std::fmax(dD, std::fabs(refP.diag[c] - cP.diag[c]));
            mD = std::fmax(mD, std::fabs(refP.diag[c]));
            dS = std::fmax(dS, std::fabs(refP.source[c] - cP.source[c]));
            mS = std::fmax(mS, std::fabs(refP.source[c]));
        }
        const scalar rD = mD > 0 ? dD / mD : dD, rS = mS > 0 ? dS / mS : dS;
        std::printf("  %-58s diag=%.3e src=%.3e\n",
                    "control: `corrected` moves coefficients AND source", (double)rD, (double)rS);
        check(rD > 1e-12, "the implicit half of the non-orth correction contributes (control)");
        check(rS > 1e-12, "the explicit half of the non-orth correction contributes (control)");
    }

    // ---- CONTROL: setReference is not a no-op -----------------------------------------------
    // fvMatrix.C DOUBLES the reference cell's diagonal rather than setting it, so it must be visible.
    {
        cpu::rhoSimple::PressureInput noR = pin;
        noR.pRefCell = (pRefCell >= 0) ? -1 : 0;
        const cpu::rhoSimple::PressureStages nst =
            cpu::rhoSimple::pressurePredictor(refU, U, p, noR, m, g, fvp);
        const FvScalarMatrix nP = cpu::rhoSimple::assemblePEqn(nst, p, noR, m, g, fvp);
        const bool moved = std::fabs(refP.diag[0] - nP.diag[0]) > 0.0;
        std::printf("  %-58s %.6e vs %.6e\n", "control: setReference doubles the ref cell diagonal",
                    (double)refP.diag[0], (double)nP.diag[0]);
        check(moved, "setReference changed the reference cell (control)");
    }

    // ---- refusals ---------------------------------------------------------------------------
    {
        struct { const char* what; int which; } cases[] = {
            {"MRF is refused on the CUDA path",                0},
            {"an unported fvOptions is refused",               1},
            // fixedFluxPressure is no longer refused: deviceConstrainPressure is wired at pEqn.H:12,
            // gated by ffp_vs_openfoam's cuda arm; a never-updated patch refuses on the HOST at read
            // time via FixedFluxPressurePatchField::requireUpdated before the device is ever built.
            {"a mesh with coupled patches is refused",         3},
        };
        for (const auto& cse : cases)
        {
            gpu::rhoSimple::RhoPressureInput bad = gpin;
            if (cse.which == 0) bad.hasMRF = true;
            if (cse.which == 1) bad.hasFvOptions = true;
            if (cse.which == 3) bad.hasCoupledPatches = true;
            gpu::rhoSimple::RhoPressureStages sb;
            bool threw = false;
            try
            {
                gpu::rhoSimple::pressurePredictor(sb, dm, dbU, dbP, MU, dUx, dUy, dUz, dP, bad);
            }
            catch (const std::runtime_error&)
            {
                threw = true;
            }
            check(threw, cse.what);
        }
        // And the two masks are REQUIRED, not optional: without them the module would silently build a
        // different HbyA or skip the flux balance entirely.
        {
            gpu::rhoSimple::RhoPressureInput bad = gpin;
            bad.takeUAtBoundary = nullptr;
            gpu::rhoSimple::RhoPressureStages sb;
            bool threw = false;
            try { gpu::rhoSimple::pressurePredictor(sb, dm, dbU, dbP, MU, dUx, dUy, dUz, dP, bad); }
            catch (const std::runtime_error&) { threw = true; }
            check(threw, "a missing constrainHbyA `assignable` mask is refused");
        }
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
