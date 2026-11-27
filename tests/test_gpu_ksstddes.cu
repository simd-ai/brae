// kOmegaSSTDES and kOmegaSSTDDES length scales: deviceKOmegaSSTDESdTildaFactor against a host transcription of OpenFOAM
// v2412's formulas, and their limits by hand. A unit check of the kernel's arithmetic: brae against a transcription, not
// against OpenFOAM. The comparison with OpenFOAM itself, on a case where the limiter acts, is
// tests/pimple_des_step_vs_openfoam.sh.
//   FDES = lRAS/dTilda (the k equation's sqrt(k)/dTilda over betaStar omega), lRAS = sqrt(k)/(betaStar omega),
//   lLES = CDES cbrt(V), CDES = F1 (CDESkom - CDESkeps) + CDESkeps                      (kOmegaSSTDES.C, .H:104-107)
//   DES   dTilda = min(lLES, lRAS)
//   DDES  dTilda = max(lRAS - fd max(lRAS - lLES, 0), SMALL), fd = 1 - tanh((Cd1 r)^Cd2),
//         r = min((nut + nu)/(max(|grad U|, SMALL) (kappa y)^2), 10)                     (kOmegaSSTDDES.C:42-47, 90-102)
#include "device_komega_sst.cuh"   // deviceKOmegaSSTDESdTildaFactor
#include "komega_sst_coeffs.cuh"   // KOmegaSSTCoeffs
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace
{
struct Cell
{
    double k;
    double om;
    double V;
    double F1;
    double g;     // every one of grad U's nine components
    double nut;
    double y;
};

double hostFDES(const Cell& c, double nu, const KOmegaSSTCoeffs& co)
{
    const double small = 1e-15;
    const double lRAS = std::sqrt(c.k) / (co.betaStar * c.om);
    const double CDES = c.F1 * (co.CDESkom - co.CDESkeps) + co.CDESkeps;
    const double lLES = CDES * std::cbrt(c.V);
    double dTilda;
    if (co.desShielded)
    {
        const double magGradU = std::sqrt(9.0 * c.g * c.g);
        const double ky = co.desKappa * c.y;
        const double r = std::fmin((c.nut + nu) / (std::fmax(magGradU, small) * ky * ky), 10.0);
        const double fd = 1.0 - std::tanh(std::pow(co.Cd1 * r, co.Cd2));
        dTilda = std::fmax(lRAS - fd * std::fmax(lRAS - lLES, 0.0), small);
    }
    else
    {
        dTilda = std::fmin(lLES, lRAS);
    }
    return lRAS / dTilda;
}
}

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp, double tol)
    {
        const bool ok = std::fabs(got - exp) <= tol * std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-52s got %.12g  exp %.12g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };
    const double nu = 1e-4;
    // [0] free shear: y large, |grad U| large -> r tiny, fd -> 1, lLES < lRAS: LES
    // [1] near a wall: y tiny, |grad U| small -> r = 10, fd -> 0: RANS, FDES 1
    // [2] lRAS < lLES: RANS under either
    // [3] in between: fd partial, F1 blended
    const Cell cells[] = {
        {1.0, 1.0, 1.0, 1.0, 100.0, 1e-3, 1.0},
        {1.0, 1.0, 1.0, 0.0, 0.01, 1e-3, 1e-6},
        {1e-6, 100.0, 1.0, 1.0, 100.0, 1e-3, 1.0},
        {1.0, 1.0, 1.0, 0.5, 0.05, 1e-3, 0.02},
    };
    const int nC = 4;
    std::vector<scalar> hk(nC), hom(nC), hV(nC), hF1(nC), hg(9 * nC), hnut(nC), hy(nC);
    for (int c = 0; c < nC; ++c)
    {
        hk[c] = cells[c].k;
        hom[c] = cells[c].om;
        hV[c] = cells[c].V;
        hF1[c] = cells[c].F1;
        hnut[c] = cells[c].nut;
        hy[c] = cells[c].y;
        for (int q = 0; q < 9; ++q)
        {
            hg[q * nC + c] = cells[c].g;
        }
    }
    DeviceBuffer<scalar> k, om, V, F1, gradU, nut, y, FDES;
    k.copyFrom(hk);
    om.copyFrom(hom);
    V.copyFrom(hV);
    F1.copyFrom(hF1);
    gradU.copyFrom(hg);
    nut.copyFrom(hnut);
    y.copyFrom(hy);
    for (const bool shielded : {true, false})
    {
        KOmegaSSTCoeffs co;   // betaStar 0.09, CDESkom 0.82, CDESkeps 0.60, Cd1 20, Cd2 3, kappa 0.41
        co.desShielded = shielded;
        deviceKOmegaSSTDESdTildaFactor(nC, k, om, V, F1, gradU, nut, y, nu, co, FDES);
        const auto f = FDES.host();
        std::printf("%s: device kernel against the host transcription\n", shielded ? "DDES" : "DES");
        for (int c = 0; c < nC; ++c)
        {
            char nm[64];
            std::snprintf(nm, sizeof nm, "cell %d FDES", c);
            chk(nm, f[c], hostFDES(cells[c], nu, co), 1e-12);
        }
        const double lRAS0 = 1.0 / 0.09;
        if (shielded)
        {
            // fd = 1 - tanh((Cd1 r)^3) with Cd1 r = 4.4e-4: 1 - 8.4e-11, so FDES is lRAS/lLES less about 1e-9 of it
            chk("free shear, fd -> 1: lRAS/(CDESkom delta)", f[0], lRAS0 / 0.82, 1e-7);
            chk("near the wall, fd -> 0: RANS, FDES 1", f[1], 1.0, 1e-12);
        }
        else
        {
            chk("free shear: lRAS/(CDESkom delta)", f[0], lRAS0 / 0.82, 1e-12);
            chk("near the wall, no delay function: lRAS/(CDESkeps delta)", f[1], lRAS0 / 0.60, 1e-12);
        }
        chk("lRAS < lLES: RANS, FDES 1", f[2], 1.0, 1e-12);
    }
    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
