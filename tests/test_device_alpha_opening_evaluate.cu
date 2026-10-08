// THE EVALUATE THAT OPENS THE LIMITED EXPLICIT SOLVE, and what the limiter is handed after it, on a fixture
// where both change the answer: the device's alpha step against the host's.
//
// MULES::explicitSolve begins with psi.correctBoundaryConditions() (MULESTemplates.C:168): AFTER the high-order
// flux has been built on the STORED patch values, BEFORE the limiter, which takes its extrema on a patch that
// fixes a value from the patch values (:338-347). And limit() overwrites the bounded flux's boundary with the
// high-order flux's on every uncoupled patch (:599-609), so the correction is zero there whatever the evaluate
// moved. The device's explicit corrector made that evaluate only where a wave model supplies the values, and
// where it did it rebuilt the bounded flux's boundary on the new values. Both were set right 2026-10-06
// (device_inter_alpha_step.cu, device_alpha_step.cu) -- by the source: NO SHIPPED CASE TELLS THE OLD CODE
// FROM THE NEW. MEASURED the same day, device arm against OpenFOAM at pinned solves: capillaryRise 40 steps,
// stokesI 40, weirOverflow 60 and the small static AMI fixture 5 give the same numbers either way (the last
// 7.7e-13 and 8.4e-13 against a control floor of 1e-8).
//
// THIS FIXTURE IS BUILT TO SEE IT. What it takes is a patch value the evaluate moves, on a face that carries
// a flux, of a class the limiter reads (one that fixes a value), next to cells where the limiter's bound is
// active:
//   * the x-min side is an inletOutlet (inletValue 0) and the blob of alpha sits ON that side, so the
//     interface crosses it;
//   * the flux is a rigid rotation whose sense REVERSES every step. At the top of a step the stored patch
//     values are the last step's closing evaluate, on the old flux; the opening evaluate is the first on the
//     new one, and every face of the side whose cell is not at the inlet value moves.
// THE ORACLE IS THE HOST STEP (alpha_eqn_cpp.cu, mules_cpp.cu), which makes the opening evaluate and has the
// overwrite, and which the gates hold to OpenFOAM (capillaryRise forty steps 2.8e-14,
// tests/interfoam_write/subcycle/final_mixture_no_evaluate.sh). The device's hooks evaluate through a host
// scratch field of their own, from the DEVICE's alpha: the two trajectories never touch.
// THE CONTROLS are the code's own switches, each read once a process, so each is this binary run again as a
// child with the switch set (`child` prints the one number): BRAE_CONTROL_MULES_OPENING_EVALUATE_SKIPPED=1
// and BRAE_CONTROL_MULES_BOUNDARY_DONOR_NEW_VALUES=1 must each leave the host by orders.
#include "box_mesh.cuh"
#include "device_gate_finite.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "interface_properties_cpp.cuh"
#include "two_phase_mixture_cpp.cuh"
#include "inter_solve_cpp.cuh"
#include "alpha_eqn_cpp.cuh"
#include "mules_cpp.cuh"
#include "device_inter_alpha_step.cuh"
#include "device_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

using namespace brae;
namespace ip = brae::cpu::interfaceProps;
namespace ifm = brae::cpu::interFoam;

namespace {

int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

std::vector<scalar> flatten(const std::vector<std::vector<scalar>>& b)
{
    std::vector<scalar> v;
    for (const auto& p : b)
    {
        v.insert(v.end(), p.begin(), p.end());
    }
    return v;
}

// what one run of the fixture measures
struct Measured
{
    scalar worst = -1;        // max |device - host| over the cells, after the last step
    long moved = 0;           // patch values the opening evaluates moved by more than 0.1, on faces with a flux
    scalar movedMost = 0;
    long evaluates = 0;       // opening evaluates made
};

Measured run()
{
    const label N = 24;
    const scalar h = scalar(1)/scalar(N);
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1, scalar(0), h, h, h);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const label nIf = m.nInternalFaces();
    std::size_t side = fvp.size();
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].name == "inlet")
        {
            side = pi;
        }
    }
    if (side == fvp.size())
    {
        std::printf("  the box has no patch named inlet\n");
        std::exit(1);
    }

    // the rotation about the box's centre, and its reverse
    const scalar omega = scalar(2);
    auto fluxOf = [&](
        scalar sense,
        SurfaceScalarField& phi)
    {
        auto dotS = [&](label f)
        {
            const vector& P = g.Cf()[f];
            const vector& S = g.Sf()[f];
            return sense*omega*(-(P.y - scalar(0.5))*S.x + (P.x - scalar(0.5))*S.y);
        };
        phi.internal.resize(static_cast<std::size_t>(nIf));
        for (label f = 0; f < nIf; ++f)
        {
            phi.internal[static_cast<std::size_t>(f)] = dotS(f);
        }
        phi.boundary.resize(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            phi.boundary[pi].resize(static_cast<std::size_t>(fvp[pi].size));
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                phi.boundary[pi][static_cast<std::size_t>(i)] = dotS(fvp[pi].start + i);
            }
        }
    };

    // a blob ON the x-min side, its interface a few cells wide
    std::vector<scalar> a0(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c)
    {
        const vector& C = g.C()[c];
        const scalar r = std::hypot(C.x, C.y - scalar(0.5));
        a0[static_cast<std::size_t>(c)] = scalar(0.5)*(scalar(1) - std::tanh((r - scalar(0.25))/scalar(0.05)));
    }

    const scalar rho1 = scalar(1000);
    const scalar rho2 = scalar(1);
    const scalar nu1 = scalar(1e-6);
    const scalar nu2 = scalar(1.48e-5);
    ip::InterfaceCoeffs ic;
    ic.cAlpha = scalar(1);
    ic.deltaN = ip::deltaN(g.V());
    const scalar totalDt = scalar(0.02);
    const int nSub = 2;
    const int nSteps = 6;

    // alpha's patches: the side an inletOutlet, the rest zero-gradient
    auto patchesOf = [&](GeometricField<scalar>& a)
    {
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (pi == side)
            {
                a.boundary.push_back(std::make_unique<InletOutletPatchField<scalar>>(
                    fvp[pi], true, scalar(0), std::vector<scalar>{}));
            }
            else
            {
                a.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(fvp[pi]));
            }
        }
    };

    Measured out;

    // the HOST: its own step, told the step's flux at its top and evaluating where OpenFOAM does
    std::vector<scalar> host;
    {
        SurfaceScalarField phi;
        fluxOf(scalar(1), phi);
        GeometricField<scalar> a;
        a.internal = a0;
        patchesOf(a);
        a.boundary[side]->updateFromFlux(phi.boundary[side]);
        a.evaluateBoundary();
        SurfaceScalarField nHatf;
        SurfaceScalarField alphaPhi;
        SurfaceScalarField rhoPhi;
        std::vector<scalar> K;
        ip::calculateK(a, ic, m, g, fvp, false, nHatf, K);
        brae::cpu::MULES::Controls hctl;
        hctl.nLimiterIter = 5;
        ifm::AlphaStepInput hin;
        hin.phi = &phi;
        hin.phiCN = &phi;
        hin.cAlpha = ic.cAlpha;
        hin.nAlphaCorr = 1;
        hin.rho1 = rho1;
        hin.rho2 = rho2;
        hin.MULESCorr = false;
        hin.alphaScheme = ifm::AlphaFluxScheme::vanLeer;
        hin.alpharScheme = ifm::AlphaFluxScheme::linear;
        ifm::AlphaEqnStep hstep = [&](
            const std::vector<scalar>& old,
            scalar dtSub,
            std::vector<scalar>& result,
            SurfaceScalarField& rp)
        {
            hin.deltaT = dtSub;
            ifm::alphaEqnStep(a, old, hin, ic, hctl, m, g, fvp, alphaPhi, rp, nHatf, K, nullptr);
            result = a.internal;
        };
        std::vector<scalar> cur = a0;
        for (int s = 0; s < nSteps; ++s)
        {
            // the step's flux, told to the patch and NOT evaluated: the values stay the last evaluate's
            fluxOf(s % 2 == 0 ? scalar(1) : scalar(-1), phi);
            a.boundary[side]->updateFromFlux(phi.boundary[side]);
            const std::vector<scalar> old = cur;
            ifm::alphaEqnSubCycle(nSub, totalDt, cur, old, rhoPhi, hstep);
        }
        host = cur;
    }

    // the DEVICE: the step, with the hooks the driver hands it, evaluating through its own scratch
    std::vector<scalar> device;
    {
        SurfaceScalarField phi;
        fluxOf(scalar(1), phi);
        GeometricField<scalar> work;
        work.internal = a0;
        patchesOf(work);
        work.boundary[side]->updateFromFlux(phi.boundary[side]);
        work.evaluateBoundary();
        auto patchValues = [&]()
        {
            std::vector<scalar> v;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                const std::vector<scalar>& b = work.boundary[pi]->value();
                v.insert(v.end(), b.begin(), b.end());
            }
            return v;
        };
        // (the driver's pushFlux: every hook that evaluates tells the patches the flux first)
        auto evaluated = [&](const DeviceBuffer<scalar>& a)
        {
            work.boundary[side]->updateFromFlux(phi.boundary[side]);
            a.copyTo(work.internal);
            work.evaluateBoundary();
        };
        auto normalUp = [&](DeviceBuffer<scalar>& nBnd)
        {
            SurfaceScalarField nHb;
            std::vector<scalar> Kb;
            ip::calculateK(work, ic, m, g, fvp, false, nHb, Kb);
            nBnd.copyFrom(flatten(nHb.boundary));
        };
        DeviceInterAlphaHooks hooks;
        hooks.updateBoundary = [&](
            const DeviceBuffer<scalar>& a,
            DeviceBuffer<scalar>& aBnd,
            DeviceBuffer<scalar>& nBnd)
        {
            evaluated(a);
            aBnd.copyFrom(patchValues());
            normalUp(nBnd);
        };
        hooks.storedBoundary = [&](DeviceBuffer<scalar>& aBnd)
        {
            aBnd.copyFrom(patchValues());
        };
        hooks.refreshBoundary = [&](
            const DeviceBuffer<scalar>& a,
            DeviceBuffer<scalar>& aBnd)
        {
            const std::vector<scalar> before = work.boundary[side]->value();
            evaluated(a);
            aBnd.copyFrom(patchValues());
            const std::vector<scalar>& after = work.boundary[side]->value();
            ++out.evaluates;
            for (std::size_t i = 0; i < after.size(); ++i)
            {
                const scalar d = std::fabs(after[i] - before[i]);
                if (d > scalar(0.1) && phi.boundary[side][i] != scalar(0))
                {
                    ++out.moved;
                    out.movedMost = std::fmax(out.movedMost, d);
                }
            }
        };
        hooks.relaxBoundary = [&](
            const DeviceBuffer<scalar>&,
            const DeviceBuffer<scalar>& relaxed,
            DeviceBuffer<scalar>& aBnd)
        {
            evaluated(relaxed);
            aBnd.copyFrom(patchValues());
        };
        hooks.mixtureCorrect = [&](
            const DeviceBuffer<scalar>& a,
            DeviceBuffer<scalar>& aBnd,
            DeviceBuffer<scalar>& nBnd)
        {
            a.copyTo(work.internal);
            aBnd.copyFrom(patchValues());
            normalUp(nBnd);
        };

        DeviceMesh dm = buildDeviceMesh(m, g, fvp);
        std::vector<int> fixes;
        std::vector<int> flag;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                fixes.push_back(work.boundary[pi]->fixesValue() ? 1 : 0);
                flag.push_back(0);
            }
        }
        DeviceBuffer<scalar> dPhiInt(phi.internal);
        DeviceBuffer<scalar> dPhiBnd(flatten(phi.boundary));
        DeviceBuffer<int> dFixes(fixes);
        DeviceBuffer<int> dFlag(flag);
        DeviceAlphaStepInput din;
        din.phiInt = &dPhiInt;
        din.phiBnd = &dPhiBnd;
        din.phiCNInt = &dPhiInt;
        din.phiCNBnd = &dPhiBnd;
        din.cAlpha = ic.cAlpha;
        din.rho1 = rho1;
        din.rho2 = rho2;
        din.deltaN = ic.deltaN;
        din.alphaScheme = DeviceAlphaScheme::vanLeer;
        din.alpharScheme = DeviceAlphaScheme::linear;
        DeviceMulesControls dmc;
        dmc.nLimiterIter = 5;
        const DevicePhaseProperties props{rho1, nu1, rho2, nu2};
        DeviceInterAlphaControls ctl;
        ctl.nAlphaSubCycles = nSub;
        ctl.nAlphaCorr = 1;
        ctl.MULESCorr = false;

        SurfaceScalarField nH0;
        std::vector<scalar> K0;
        ip::calculateK(work, ic, m, g, fvp, false, nH0, K0);
        DeviceBuffer<scalar> alpha(a0);
        DeviceBuffer<scalar> alphaOld(a0);
        DeviceBuffer<scalar> aBnd(patchValues());
        DeviceBuffer<scalar> nBnd(flatten(nH0.boundary));
        DeviceBuffer<scalar> nHatf(nH0.internal);
        DeviceBuffer<scalar> K(K0);
        DeviceBuffer<scalar> rpInt;
        DeviceBuffer<scalar> rpBnd;
        DeviceBuffer<scalar> alpha2;
        DeviceBuffer<scalar> rho;
        DeviceBuffer<scalar> mu;
        DeviceBuffer<scalar> nu;
        for (int s = 0; s < nSteps; ++s)
        {
            fluxOf(s % 2 == 0 ? scalar(1) : scalar(-1), phi);
            dPhiInt.copyFrom(phi.internal);
            dPhiBnd.copyFrom(flatten(phi.boundary));
            std::vector<scalar> cur;
            alpha.copyTo(cur);
            failures += brae::gatecheck::nonFinite("alpha", cur);
            alphaOld.copyFrom(cur);
            deviceInterAlphaStep(dm, alpha, alphaOld, totalDt, din, dmc, ctl, props, hooks,
                                 aBnd, nBnd, dFixes, dFlag, nHatf, K, rpInt, rpBnd,
                                 alpha2, rho, mu, nu);
        }
        alpha.copyTo(device);
    }

    out.worst = 0;
    for (std::size_t c = 0; c < host.size() && c < device.size(); ++c)
    {
        out.worst = std::fmax(out.worst, std::fabs(device[c] - host[c]));
    }
    if (host.size() != device.size() || host.empty())
    {
        out.worst = -1;
    }
    return out;
}

// this binary again, as a child with one switch set: the number its `child` line prints, or -1
scalar childWorst(
    const std::string& self,
    const std::string& envSwitch)
{
    const std::string cmd = envSwitch + "=1 '" + self + "' child 2>&1";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return scalar(-1);
    char line[1024];
    scalar worst = -1;
    bool controlSaid = false;
    while (std::fgets(line, sizeof(line), p))
    {
        const std::string l(line);
        controlSaid = controlSaid || l.find("CONTROL MODE") != std::string::npos
                   || l.find("(" + envSwitch + ")") != std::string::npos;
        double w = 0;
        if (std::sscanf(line, "CHILD worst=%lf", &w) == 1)
        {
            worst = static_cast<scalar>(w);
        }
    }
    pclose(p);
    // a child that never said it ran the switch has not run it
    return controlSaid ? worst : scalar(-1);
}

} // namespace

int main(
    int argc,
    char** argv)
{
    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess)
    {
        cudaGetLastError();
        nDev = 0;
    }
    if (nDev <= 0)
    {
        std::printf("  SKIP: no CUDA device\n");
        return 77;
    }
    if (argc > 1 && std::string(argv[1]) == "child")
    {
        const Measured c = run();
        std::printf("CHILD worst=%.17g\n", static_cast<double>(c.worst));
        return 0;
    }
    std::printf("== interFoam: the evaluate that opens the limited explicit solve, device vs host, on a side "
                "whose flux reverses\n");
    const Measured d = run();
    std::printf("  six steps of two sub-cycles: %ld opening evaluates, %ld patch values moved by more than 0.1 on "
                "faces carrying a flux (%.3f at most); device against host %.3e\n", d.evaluates, d.moved,
                static_cast<double>(d.movedMost), static_cast<double>(d.worst));
    // MEASURED 2026-10-06: 12 opening evaluates, 66 values moved (by 1.000 at most), device against host
    // 4.4e-16; the opening evaluate skipped 8.6e-04; the boundary rebuilt on the new values 2.4e-02. The bound
    // is a decade above the first, the controls' floor three decades above the bound.
    check("the device step is the host's with the opening evaluate moving fixes-value patch values under the "
          "limiter",
          d.worst >= scalar(0) && d.worst < scalar(5e-15) && d.evaluates == 12 && d.moved > 0
          && d.movedMost > scalar(0.5));
    const scalar skipped = childWorst(argv[0], "BRAE_CONTROL_MULES_OPENING_EVALUATE_SKIPPED");
    std::printf("  CONTROL, the opening evaluate skipped: device against host %.3e\n", static_cast<double>(skipped));
    check("CONTROL: without the opening evaluate the device leaves the host by orders", skipped > scalar(1e-6));
    const scalar donor = childWorst(argv[0], "BRAE_CONTROL_MULES_BOUNDARY_DONOR_NEW_VALUES");
    std::printf("  CONTROL, the bounded flux's boundary rebuilt on the new values: device against host %.3e\n",
                static_cast<double>(donor));
    check("CONTROL: with the bounded flux's boundary rebuilt on the new values the device leaves the host",
          donor > scalar(1e-6));
    std::printf("test_device_alpha_opening_evaluate: %d failures\n", failures);
    return failures ? 1 : 0;
}
