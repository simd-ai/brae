// SpalartAllmarasDDES `shielding ZDES2020` (Deck & Renard 2020), OF v2412 SpalartAllmarasDDES::fd.
//
//     r          = min(nuEff/(max(|gradU|,SMALL) (kappa y)^2), 10)
//     fdStd      = 1 - tanh((Cd1 r)^Cd2)                              <- the standard DDES shielding
//     GnuTilda   = Cd3 max(grad(nuTilda).n, 0)/(max(|gradU|,SMALL) kappa y)
//     fdGnuTilda = 1 - tanh((Cd1 GnuTilda)^Cd2)
//     GOmega     = -(grad(|curl U|).n) sqrt(nuTilda/max(|gradU|^3, SMALL))
//     alpha      = (7/6 Cd4 - GOmega)/(Cd4/6)
//     fR         = pos(Cd4 - GOmega)
//                + 1/(1 + exp(min(-6 alpha/max(1-alpha^2, SMALL), 50))) pos(4Cd4/3 - GOmega) pos(GOmega - Cd4)
//     fd         = fdStd (1 - (1 - fdGnuTilda) fR)
//
// THE SIGN OF n IS THE TRAP. n is OF's wallDist::n(), seeded as `patch.nf()` -- the wall face's OUTWARD
// normal, pointing OUT of the domain, roughly OPPOSITE to (cell centre - wall). Moving along +n therefore
// goes TOWARDS the wall, so the max(.,0) admits only the region where nuTilda grows towards the wall --
// the outer layer, where standard DDES releases to LES too early -- and clips the near-wall side where
// nuTilda still rises away from the wall. Flip n and the model shields the wrong half of the layer.
// Leg 3 pins that by construction: the same field with n reversed must give a DIFFERENT fd.
//
// Leg 2 is the reduction that has to hold: with both wall-normal derivatives zero, ZDES2020 must collapse
// EXACTLY onto the standard fd, so a case that gains nothing from the extra terms is unchanged.
#include "device_buffer.cuh"
#include "device_kepsilon.cuh"
#include "spalart_coeffs.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cell_wall_dist.cuh"
#include "foam_dict.cuh"
#include "solver_controls.cuh"
#include "turbulence_setup.cuh"
#include "box_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// OF's formula, transcribed on the host from SpalartAllmarasDDES.C.
scalar fdHost(scalar y, const scalar g[9], scalar nt, scalar nu,
              const scalar n[3], const scalar gnt[3], const scalar gom[3],
              const SpalartAllmarasCoeffs& co)
{
    constexpr scalar SMALL = 1e-15;
    scalar g2 = 0;
    for (int k = 0; k < 9; ++k) g2 += g[k]*g[k];
    const scalar magGradU = std::sqrt(g2);
    const scalar chi = nt/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const scalar nut = nt*(chi3/(chi3 + Cv13));
    const scalar kd2 = std::max(co.kappa*co.kappa*y*y, scalar(1e-300));
    const scalar r = std::min((nut + nu)/(std::max(magGradU, scalar(1e-300))*kd2), scalar(10));
    const scalar fdStd = 1 - std::tanh(std::pow(co.Cd1*r, co.Cd2));

    const scalar dNt = std::max(gnt[0]*n[0] + gnt[1]*n[1] + gnt[2]*n[2], scalar(0));
    const scalar GnuTilda = co.Cd3*dNt/std::max(std::max(magGradU, SMALL)*co.kappa*y, scalar(1e-300));
    scalar fdG = 1 - std::tanh(std::pow(co.Cd1*GnuTilda, co.Cd2));
    if (co.usefP2)
        fdG *= (1 - std::tanh(std::pow(co.Cd1*co.betaZDES*r, co.Cd2)))/std::max(fdStd, SMALL);

    const scalar mg3 = magGradU*magGradU*magGradU;
    const scalar GOmega = -(gom[0]*n[0] + gom[1]*n[1] + gom[2]*n[2])
                        * std::sqrt(std::max(nt, scalar(0))/std::max(mg3, SMALL));
    const scalar alpha = (7.0/6.0*co.Cd4 - GOmega)/(co.Cd4/6.0);
    const scalar fR = ((co.Cd4 - GOmega > 0) ? 1.0 : 0.0)
                    + 1.0/(1.0 + std::exp(std::min(-6.0*alpha/std::max(1 - alpha*alpha, SMALL), scalar(50))))
                      * ((4.0*co.Cd4/3.0 - GOmega > 0) ? 1.0 : 0.0)
                      * ((GOmega - co.Cd4 > 0) ? 1.0 : 0.0);
    return fdStd*(1 - (1 - fdG)*fR);
}

FoamDict dictFromString(const std::string& body)
{
    const std::string path = "test_zdes_shielding.tmpdict";
    { std::ofstream f(path); f << body; }
    FoamDict d = readDict(path);
    std::remove(path.c_str());
    return d;
}
} // namespace

int main()
{
    std::printf("== SpalartAllmarasDDES shielding ZDES2020 ==\n");
    SpalartAllmarasCoeffs co;
    co.zdes = true;

    // A state in the OUTER part of an attached layer -- which is where ZDES2020 does its work. The
    // standard fd is already well off zero there (0.196 below), nuTilda is on its way DOWN towards the
    // freestream, and the extra shielding pulls fd back towards RANS. Deeper in, where fdStd is already
    // 0, every formula in sight returns 0 and the test would be measuring nothing.
    const scalar y = 0.008, nu = 1e-5, nt = 3e-4;
    scalar g[9] = {0, 0, 0, 220.0, 0, 0, 0, 0, 0};   // dU_x/dy = 220 (pure shear)
    const scalar nWall[3] = {0, -1, 0};              // wall at the BOTTOM: the outward normal points -y
    const scalar gradNt[3] = {0, -0.00285, 0};       // nuTilda falling with +y: grad.n > 0, shielding fires
    const scalar gradOm[3] = {0, -1.2e4, 0};

    // ---- Leg 1: the device kernel against the host transcription -----------------------------------
    // Driven through deviceSADDESdTilda's sibling: build the buffers the solver would.
    {
        const int nC = 1;
        DeviceBuffer<scalar> dY, dG, dNt, dWn[3], dGn[3], dGo[3], dFd;
        dY.copyFrom(std::vector<scalar>{y});
        std::vector<scalar> gp(9);
        for (int k = 0; k < 9; ++k) gp[k] = g[k];
        dG.copyFrom(gp);
        dNt.copyFrom(std::vector<scalar>{nt});
        for (int k = 0; k < 3; ++k)
        {
            dWn[k].copyFrom(std::vector<scalar>{nWall[k]});
            dGn[k].copyFrom(std::vector<scalar>{gradNt[k]});
            dGo[k].copyFrom(std::vector<scalar>{gradOm[k]});
        }
        deviceSAZdesFd(nC, dY, dG, dNt, nu, dWn[0], dWn[1], dWn[2],
                       dGn[0], dGn[1], dGn[2], dGo[0], dGo[1], dGo[2], co, dFd);
        std::vector<scalar> h; dFd.copyTo(h);
        const scalar ref = fdHost(y, g, nt, nu, nWall, gradNt, gradOm, co);
        check(std::fabs(h[0] - ref) < 1e-12, "kernel fd == the host OF formula");
        check(ref > 0 && ref < 1, "vacuity guard: fd is strictly inside (0,1) on this state");
        const scalar zero3[3] = {0, 0, 0};
        const scalar fdStd = fdHost(y, g, nt, nu, nWall, zero3, zero3, co);
        std::printf("        (fd = %.6f, standard fd would be %.6f)\n", ref, fdStd);

        // ...and on the OTHER side of the max(.,0), where the clip is what does the work. Near the wall
        // nuTilda rises AWAY from the wall, so grad(nuTilda).n < 0, GnuTilda is clipped to zero and fd
        // must land exactly on the standard shielding. Without the clip a negative GnuTilda would raise
        // fdGnuTilda above 1 and push fd ABOVE fdStd -- ZDES2020 un-shielding a boundary layer, which is
        // the opposite of what it is for. The kernel is checked here, not just the host formula.
        const scalar gradNtUp[3] = {0, +0.00285, 0};
        for (int k = 0; k < 3; ++k) dGn[k].copyFrom(std::vector<scalar>{gradNtUp[k]});
        DeviceBuffer<scalar> dFd2;
        deviceSAZdesFd(nC, dY, dG, dNt, nu, dWn[0], dWn[1], dWn[2],
                       dGn[0], dGn[1], dGn[2], dGo[0], dGo[1], dGo[2], co, dFd2);
        std::vector<scalar> h2; dFd2.copyTo(h2);
        check(std::fabs(h2[0] - fdStd) < 1e-12, "clip: grad(nuTilda).n < 0 leaves fd exactly at the standard shielding");
        check(std::fabs(h2[0] - h[0]) > 1e-3, "...and that is a different number from the unclipped side");
    }

    // ---- Leg 2: zero wall-normal derivatives collapse ZDES2020 onto the standard fd -----------------
    {
        const scalar zero[3] = {0, 0, 0};
        const scalar zdes = fdHost(y, g, nt, nu, nWall, zero, zero, co);
        scalar g2 = 0; for (int k = 0; k < 9; ++k) g2 += g[k]*g[k];
        const scalar chi = nt/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
        const scalar nut = nt*(chi3/(chi3 + Cv13));
        const scalar r = std::min((nut + nu)/(std::sqrt(g2)*co.kappa*co.kappa*y*y), scalar(10));
        const scalar std_fd = 1 - std::tanh(std::pow(co.Cd1*r, co.Cd2));
        // GnuTilda = 0 -> fdG = 1 -> fd = fdStd*(1 - 0) = fdStd, whatever fR does.
        check(std::fabs(zdes - std_fd) < 1e-14, "no wall-normal gradients: fd is exactly the standard shielding");
    }

    // ---- Leg 3: the direction of n is load-bearing --------------------------------------------------
    {
        const scalar flipped[3] = {0, 1, 0};   // n pointing INTO the fluid -- the plausible wrong choice
        const scalar right = fdHost(y, g, nt, nu, nWall,   gradNt, gradOm, co);
        const scalar wrong = fdHost(y, g, nt, nu, flipped, gradNt, gradOm, co);
        check(std::fabs(right - wrong) > 1e-3, "reversing n changes fd -- the outward convention is not cosmetic");
        std::printf("        (outward n: fd = %.6f   inward n: fd = %.6f)\n", right, wrong);
    }

    // ---- Leg 4: wallDist carries the OUTWARD normal, on a real mesh ---------------------------------
    // boxMesh's Zmin/Zmax patches are the walls here; a cell in the lower half must inherit the Zmin
    // face normal (0,0,-1), i.e. pointing OUT of the domain, and one in the upper half (0,0,+1).
    {
        PrimitiveMesh m = boxtest::boxMesh(3, 3, 6);
        FvGeometry gg; gg.build(m);
        std::vector<FvPatch> fvp = buildPatches(m, gg);
        std::size_t nWall = 0;
        for (FvPatch& p : fvp)
        {
            const bool isZ = (p.name.find("Zmin") != std::string::npos || p.name.find("Zmax") != std::string::npos);
            p.type = isZ ? "wall" : "patch";
            if (isZ) ++nWall;
        }
        check(nWall == 2, "vacuity guard: the fixture has exactly the two z walls");
        std::vector<vector> wn;
        const std::vector<scalar> yy = cellWallDist(m, gg, fvp, nullptr, &wn);
        const std::vector<vector>& C = gg.C();
        const scalar zMid = 3.0;   // unit-spaced cells, 6 deep
        scalar worst = 0;
        int checked = 0;
        for (label c = 0; c < m.nCells(); ++c)
        {
            const vector& n = wn[c];
            const scalar want = (C[c].z < zMid) ? scalar(-1) : scalar(1);
            worst = std::max(worst, std::fabs(n.z - want));
            worst = std::max(worst, std::fabs(n.x) + std::fabs(n.y));
            ++checked;
        }
        check(checked == m.nCells() && worst < 1e-12,
              "cellWallDist: every cell inherits the nearest wall face's OUTWARD normal");
        check(yy[0] > 0, "vacuity guard: the wall distance is non-degenerate");
    }

    // ---- Leg 5: the same, for exactDistance (the method NACA4412 actually names) --------------------
    {
        PrimitiveMesh m = boxtest::boxMesh(3, 3, 6);
        FvGeometry gg; gg.build(m);
        std::vector<FvPatch> fvp = buildPatches(m, gg);
        for (FvPatch& p : fvp)
            p.type = (p.name.find("Zmin") != std::string::npos || p.name.find("Zmax") != std::string::npos) ? "wall" : "patch";
        std::vector<vector> wn;
        const std::vector<scalar> yy = exactCellWallDist(m, gg, fvp, &wn);
        const std::vector<vector>& C = gg.C();
        scalar worst = 0;
        for (label c = 0; c < m.nCells(); ++c)
        {
            const scalar want = (C[c].z < 3.0) ? scalar(-1) : scalar(1);
            worst = std::max(worst, std::fabs(wn[c].z - want));
        }
        check(worst < 1e-12, "exactCellWallDist: same outward normal, from the nearest surface point");
        check(yy[0] > 0, "vacuity guard: the exact wall distance is non-degenerate");
    }

    // ---- Leg 6: selection and refusals --------------------------------------------------------------
    {
        const FoamDict d = dictFromString(
            "simulationType LES;\n"
            "LES { LESModel SpalartAllmarasDDES; delta cubeRootVol; turbulence on;\n"
            "      SpalartAllmarasDDESCoeffs { shielding ZDES2020; Cd3 30; } }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        readTurbulenceModel(d, ctl, {"test", true, true});
        check(ctl.saCoeffs.zdes, "parser: `shielding ZDES2020` selects it");
        check(std::fabs(ctl.saCoeffs.Cd3 - 30.0) < 1e-14, "parser: Cd3 override is honoured");
        check(!ctl.saCoeffs.usefP2, "parser: usefP2 defaults to false, as OF's Switch does");
    }
    {
        const FoamDict d = dictFromString(
            "simulationType LES;\n"
            "LES { LESModel SpalartAllmarasDDES; delta cubeRootVol; turbulence on;\n"
            "      SpalartAllmarasDDESCoeffs { shielding standard; } }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        readTurbulenceModel(d, ctl, {"test", true, true});
        check(!ctl.saCoeffs.zdes, "negative control: `shielding standard` leaves ZDES2020 off");
    }
    {
        const FoamDict d = dictFromString(
            "simulationType LES;\n"
            "LES { LESModel SpalartAllmarasDDES; delta cubeRootVol; turbulence on;\n"
            "      SpalartAllmarasDDESCoeffs { shielding somethingElse; } }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        bool threw = false;
        try { readTurbulenceModel(d, ctl, {"test", true, true}); } catch (const std::exception&) { threw = true; }
        check(threw, "refusal: an unimplemented shielding mode is rejected, not silently substituted");
    }
    {
        const FoamDict d = dictFromString(
            "simulationType LES;\n"
            "LES { LESModel SpalartAllmarasIDDES; delta cubeRootVol; turbulence on;\n"
            "      SpalartAllmarasIDDESCoeffs { shielding ZDES2020; } }\n");
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        bool threw = false;
        try { readTurbulenceModel(d, ctl, {"test", true, true}); } catch (const std::exception&) { threw = true; }
        check(threw, "refusal: ZDES2020 under IDDES, which never calls fd, is rejected");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
