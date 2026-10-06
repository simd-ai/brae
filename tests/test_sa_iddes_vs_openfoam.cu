// brae's SA-IDDES length scale and its reader against REAL OpenFOAM's SpalartAllmarasIDDES, through a dump of
// OpenFOAM's own dTilda() on a constructed box (tests/sa_iddes_lib.sh stages it; tools/dumpSAIDDES is the model,
// a renamed copy of OpenFOAM's class that writes its terms). The kernel is fed OpenFOAM's own y, hmax, delta,
// gradU, nuTilda and nu, cell by cell, so what is compared is the FORMULA and its constants.
//   test_sa_iddes_vs_openfoam <dump> kernel     the kernel with the constants of the dump's header
//   test_sa_iddes_vs_openfoam <dump> defaults   the kernel with the constants brae's reader gives a case that
//                                               sets none
//   test_sa_iddes_vs_openfoam <dump> refusals   the reader's refusals and where it takes coefficients from
// Three checks a mode. The two controls of the first two modes are switches read once a process, so each runs
// in a child of this binary (`<dump> child <mode>`), which has to say it ran the switch.
// B_DTILDA: the widest |brae - OpenFOAM|/OpenFOAM of dTilda over the cells. MEASURED 2026-10-06 on 320 cells:
// 1.2e-15 (the GPU's tanh, pow and exp against glibc's); the bound is a decade above. With psi left out of
// fe, as the kernel had it, 8.2e-01 (OpenFOAM's dTilda 5.5 times brae's where chi is 0.5, 3.08 where it is 3);
// with kOmegaSSTIDDES's constants, which the reader started from, 3.9e+00 and 33 cells more than 1% off.
// DOES NOT CLAIM: brae's own hmax, wall distance, delta or gradient on a mesh -- they are OpenFOAM's here -- so
// no whole SA-IDDES run is held by this file.
#include "device_buffer.cuh"
#include "device_kepsilon.cuh"
#include "spalart_coeffs.cuh"
#include "foam_dict.cuh"
#include "solver_controls.cuh"
#include "turbulence_setup.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace brae;

namespace {

const double B_DTILDA = 2e-14;

struct Dump
{
    std::map<std::string, double> header;
    int n = 0;
    std::vector<scalar> y, hmax, delta, gradU, nuTilda, nut, psi, alpha, fdTilda, fe, dTilda;
    scalar nu = 0;
};

bool readDump(
    const std::string& path,
    Dump& d)
{
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    std::getline(in, line);
    {
        std::istringstream h(line);
        std::string hash;
        h >> hash;
        std::string key;
        double value = 0;
        while (h >> key >> value)
        {
            d.header[key] = value;
        }
    }
    d.n = static_cast<int>(d.header["cells"]);
    if (d.n <= 0) return false;
    const std::size_t n = static_cast<std::size_t>(d.n);
    d.gradU.resize(9*n);
    for (std::size_t c = 0; c < n; ++c)
    {
        double v[27];
        for (double& x : v)
        {
            if (!(in >> x)) return false;
        }
        d.y.push_back(v[0]);
        d.hmax.push_back(v[1]);
        d.delta.push_back(v[2]);
        for (std::size_t k = 0; k < 9; ++k)
        {
            d.gradU[k*n + c] = v[3 + k];
        }
        d.nuTilda.push_back(v[12]);
        d.nut.push_back(v[13]);
        d.nu = v[14];
        d.psi.push_back(v[15]);
        d.alpha.push_back(v[16]);
        d.fdTilda.push_back(v[21]);
        d.fe.push_back(v[24]);
        d.dTilda.push_back(v[26]);
    }
    return true;
}

SpalartAllmarasCoeffs coeffsOfHeader(const Dump& d)
{
    SpalartAllmarasCoeffs co;
    std::map<std::string, double> h = d.header;
    co.Cdt1 = h["Cdt1"];
    co.Cdt2 = h["Cdt2"];
    co.Cl = h["Cl"];
    co.Ct = h["Ct"];
    co.fe = h["fe"] != 0;
    co.CDES = h["CDES"];
    co.fwStar = h["fwStar"];
    co.kappa = h["kappa"];
    co.Cv1 = h["Cv1"];
    co.Cb1 = h["Cb1"];
    return co;
}

// the fixture's turbulenceProperties as brae reads it: the model under OpenFOAM's own name, nothing set
const char* const FIXTURE_DICT =
    "simulationType LES;\n"
    "LES { LESModel SpalartAllmarasIDDES; printCoeffs on; turbulence on; delta IDDESDelta;\n"
    "      IDDESDeltaCoeffs { hmax maxDeltaxyz; maxDeltaxyzCoeffs {} } }\n";

// (the reader takes a dictionary read from a file: the text goes through one, named for this process)
SpalartAllmarasCoeffs coeffsOfReader(const std::string& text)
{
    const std::string path = "test_sa_iddes_" + std::to_string(static_cast<long>(getpid())) + ".tmpdict";
    {
        std::ofstream f(path);
        f << text;
    }
    FoamDict dict;
    try
    {
        dict = readDict(path);
    }
    catch (...)
    {
        std::remove(path.c_str());
        throw;
    }
    std::remove(path.c_str());
    DeviceSimpleControls ctl;
    ctl.turbulent = true;
    readTurbulenceModel(dict, ctl, {"test", true, true});
    return ctl.saCoeffs;
}

struct Gap
{
    double worst = 0;
    long over1pc = 0;
    // the widest ratio OpenFOAM/brae among the cells of each chi band (0.5, 3, 10, 100)
    double ratio[4] = {1, 1, 1, 1};
};

Gap run(
    const Dump& d,
    const SpalartAllmarasCoeffs& co)
{
    const std::size_t n = static_cast<std::size_t>(d.n);
    DeviceBuffer<scalar> y(d.y);
    DeviceBuffer<scalar> hmax(d.hmax);
    DeviceBuffer<scalar> hwn(d.delta);
    DeviceBuffer<scalar> gradU(d.gradU);
    DeviceBuffer<scalar> nuTilda(d.nuTilda);
    DeviceBuffer<scalar> out;
    deviceSAIDDESdTilda(d.n, y, hmax, hwn, gradU, nuTilda, d.nu, co, out);
    const std::vector<scalar> got = out.host();
    Gap g;
    for (std::size_t c = 0; c < n; ++c)
    {
        const double rel = std::fabs(got[c] - d.dTilda[c])/d.dTilda[c];
        g.worst = std::fmax(g.worst, rel);
        g.over1pc += rel > 0.01 ? 1 : 0;
        const double chi = d.nuTilda[c]/d.nu;
        const int band = chi < 1 ? 0 : (chi < 5 ? 1 : (chi < 50 ? 2 : 3));
        const double r = d.dTilda[c]/got[c];
        if (std::fabs(r - 1) > std::fabs(g.ratio[band] - 1))
        {
            g.ratio[band] = r;
        }
    }
    return g;
}

// the control's child: this binary again with the switch set, its line read back
bool child(
    const char* self,
    const std::string& dump,
    const char* mode,
    const char* envName,
    Gap& g,
    SpalartAllmarasCoeffs* co)
{
    setenv(envName, "1", 1);
    const std::string cmd = std::string(self) + " " + dump + " child " + mode + " 2>&1";
    FILE* p = popen(cmd.c_str(), "r");
    unsetenv(envName);
    if (!p) return false;
    char line[600];
    bool said = false;
    bool have = false;
    while (std::fgets(line, sizeof(line), p))
    {
        const std::string s(line);
        if (s.find(std::string("child ran with ") + envName) != std::string::npos)
        {
            said = true;
        }
        double w = 0;
        long o = 0;
        double r[4];
        double c[3];
        if (std::sscanf(line, "CHILD worst %lg over1pc %ld ratios %lg %lg %lg %lg coeffs %lg %lg %lg", &w, &o,
                        &r[0], &r[1], &r[2], &r[3], &c[0], &c[1], &c[2]) == 9)
        {
            g.worst = w;
            g.over1pc = o;
            for (int k = 0; k < 4; ++k)
            {
                g.ratio[k] = r[k];
            }
            if (co)
            {
                co->Cdt1 = c[0];
                co->Cl = c[1];
                co->Ct = c[2];
            }
            have = true;
        }
    }
    pclose(p);
    return said && have;
}

bool refuses(
    const std::string& text,
    const std::string& key)
{
    try
    {
        coeffsOfReader(text);
    }
    catch (const std::exception& e)
    {
        return std::string(e.what()).find(key) != std::string::npos;
    }
    return false;
}

std::string lesDict(
    const std::string& bare,
    const std::string& inCoeffs)
{
    return "simulationType LES;\nLES { LESModel SpalartAllmarasIDDES; turbulence on; delta IDDESDelta;\n"
           "      IDDESDeltaCoeffs { hmax maxDeltaxyz; maxDeltaxyzCoeffs {} }\n      " + bare + "\n"
           + (inCoeffs.empty() ? "" : "      SpalartAllmarasIDDESCoeffs { " + inCoeffs + " }\n") + "}\n";
}

}   // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <dump> <kernel|defaults|refusals>\n", argv[0]);
        return 2;
    }
    const std::string dumpPath = argv[1];
    const std::string mode = argv[2];
    Dump d;
    if (!readDump(dumpPath, d))
    {
        std::printf("FAIL: %s is not a dump of SpalartAllmarasIDDESDump\n", dumpPath.c_str());
        return 1;
    }
    if (mode == "child")
    {
        const std::string what = argc > 3 ? argv[3] : "";
        const char* envName = what == "kernel" ? "BRAE_CONTROL_SA_IDDES_FE_NOPSI"
                                               : "BRAE_CONTROL_SA_IDDES_SST_DEFAULTS";
        if (std::getenv(envName) == nullptr) return 1;
        std::printf("child ran with %s\n", envName);
        const SpalartAllmarasCoeffs co = what == "kernel" ? coeffsOfHeader(d) : coeffsOfReader(FIXTURE_DICT);
        const Gap g = run(d, co);
        std::printf("CHILD worst %.17g over1pc %ld ratios %.17g %.17g %.17g %.17g coeffs %.17g %.17g %.17g\n",
                    g.worst, g.over1pc, g.ratio[0], g.ratio[1], g.ratio[2], g.ratio[3], co.Cdt1, co.Cl, co.Ct);
        return 0;
    }
    int fails = 0;
    const auto say = [&](
        bool ok,
        const std::string& what)
    {
        std::printf("  %-5s %s\n", ok ? "ok:" : "FAIL:", what.c_str());
        fails += ok ? 0 : 1;
    };
    char buf[600];
    if (mode == "kernel")
    {
        long fePsi = 0;
        long blended = 0;
        long neg = 0;
        double psiLo = 1e300;
        double psiHi = 0;
        double nutGap = 0;
        const double Cv13 = std::pow(d.header["Cv1"], 3);
        for (std::size_t c = 0; c < static_cast<std::size_t>(d.n); ++c)
        {
            fePsi += (d.fe[c] > 0 && std::fabs(d.psi[c] - 1) > 0.1) ? 1 : 0;
            blended += d.fdTilda[c] < 1 ? 1 : 0;
            neg += d.alpha[c] < 0 ? 1 : 0;
            psiLo = std::fmin(psiLo, d.psi[c]);
            psiHi = std::fmax(psiHi, d.psi[c]);
            // the kernel forms nut as nuTilda*fv1; OpenFOAM hands dTilda() its stored nut
            const double chi = d.nuTilda[c]/d.nu;
            const double fv1 = chi*chi*chi/(chi*chi*chi + Cv13);
            nutGap = std::fmax(nutGap, std::fabs(d.nuTilda[c]*fv1 - d.nut[c])/d.nut[c]);
        }
        std::snprintf(buf, sizeof(buf), "PREMISE  of %d cells OpenFOAM has fe > 0 with psi off 1 in %ld, fdTilda < 1 "
                      "in %ld, alpha < 0 in %ld; psi %.3g to %.3g; its stored nut is nuTilda*fv1 to %.1e", d.n,
                      fePsi, blended, neg, psiLo, psiHi, nutGap);
        say(fePsi > 0 && blended > 0 && neg > 0 && neg < d.n && nutGap < 1e-12, buf);
        const Gap g = run(d, coeffsOfHeader(d));
        std::snprintf(buf, sizeof(buf), "dTilda is OpenFOAM's in every cell: widest |brae - OpenFOAM|/OpenFOAM "
                      "%.3e (bound %.0e)", g.worst, B_DTILDA);
        say(g.worst <= B_DTILDA, buf);
        Gap k;
        const bool ran = child(argv[0], dumpPath, "kernel", "BRAE_CONTROL_SA_IDDES_FE_NOPSI", k, nullptr);
        std::snprintf(buf, sizeof(buf), "CONTROL  fe without psi: widest gap %.3e; OpenFOAM/brae up to %.3g where chi "
                      "is 0.5 and %.3g where chi is 3", k.worst, k.ratio[0], k.ratio[1]);
        say(ran && k.worst > 1e3*std::fmax(g.worst, 1e-16) && k.worst > 1e-2, buf);
    }
    else if (mode == "defaults")
    {
        const SpalartAllmarasCoeffs want = coeffsOfHeader(d);
        const SpalartAllmarasCoeffs have = coeffsOfReader(FIXTURE_DICT);
        const bool same = have.Cdt1 == want.Cdt1 && have.Cdt2 == want.Cdt2 && have.Cl == want.Cl
                       && have.Ct == want.Ct && have.fe == want.fe && have.CDES == want.CDES
                       && have.fwStar == want.fwStar && have.kappa == want.kappa && have.Cv1 == want.Cv1
                       && have.Cb1 == want.Cb1;
        std::snprintf(buf, sizeof(buf), "a case that sets no coefficient gets OpenFOAM's constructed model's: Cdt1 "
                      "%g (%g) Cdt2 %g (%g) Cl %g (%g) Ct %g (%g) fe %d (%d) CDES %g fwStar %g", have.Cdt1,
                      want.Cdt1, have.Cdt2, want.Cdt2, have.Cl, want.Cl, have.Ct, want.Ct, have.fe ? 1 : 0,
                      want.fe ? 1 : 0, have.CDES, have.fwStar);
        say(same, buf);
        const Gap g = run(d, have);
        std::snprintf(buf, sizeof(buf), "dTilda with the reader's constants is OpenFOAM's: widest gap %.3e (bound "
                      "%.0e)", g.worst, B_DTILDA);
        say(g.worst <= B_DTILDA, buf);
        Gap k;
        SpalartAllmarasCoeffs old;
        const bool ran = child(argv[0], dumpPath, "defaults", "BRAE_CONTROL_SA_IDDES_SST_DEFAULTS", k, &old);
        std::snprintf(buf, sizeof(buf), "CONTROL  kOmegaSSTIDDES's defaults (Cdt1 %g Cl %g Ct %g): widest gap %.3e, "
                      "%ld of %d cells move by more than 1%%", old.Cdt1, old.Cl, old.Ct, k.worst, k.over1pc, d.n);
        say(ran && old.Cdt1 != want.Cdt1 && k.over1pc > 0 && k.worst > 1e-2, buf);
    }
    else if (mode == "refusals")
    {
        // the three switches brae runs at their defaults alone, each written in the sub-dictionary and bare
        const char* keys[3][2] = {{"lowReCorrection", "false"}, {"useSigma", "true"}, {"ft2", "true"}};
        int refused = 0;
        int accepted = 0;
        for (const auto& k : keys)
        {
            const std::string off = std::string(k[0]) + " " + k[1] + ";";
            const std::string dflt = std::string(k[0]) + " " + (std::string(k[1]) == "true" ? "false" : "true") + ";";
            refused += refuses(lesDict("", off), k[0]) ? 1 : 0;
            refused += refuses(lesDict(off, ""), k[0]) ? 1 : 0;
            accepted += refuses(lesDict("", dflt), k[0]) ? 0 : 1;
            accepted += refuses(lesDict(dflt, ""), k[0]) ? 0 : 1;
        }
        std::snprintf(buf, sizeof(buf), "lowReCorrection false, useSigma true and ft2 true are refused by name, in "
                      "the sub-dictionary and bare in LES{}: %d of 6", refused);
        say(refused == 6, buf);
        std::snprintf(buf, sizeof(buf), "CONTROL  the same keys at their defaults are accepted: %d of 6", accepted);
        say(accepted == 6, buf);
        const SpalartAllmarasCoeffs bare = coeffsOfReader(
            lesDict("Cdt1 10; Cdt2 2; Cl 4; Ct 2; fe false; fwStar 0.5; CDES 0.7;", ""));
        const SpalartAllmarasCoeffs sub = coeffsOfReader(
            lesDict("Cdt1 10;", "Cdt1 11; Cdt2 2.5; Cl 4.5; Ct 2.5; fe no; fwStar 0.45; CDES 0.75;"));
        const bool read = bare.Cdt1 == 10 && bare.Cdt2 == 2 && bare.Cl == 4 && bare.Ct == 2 && !bare.fe
                       && bare.fwStar == 0.5 && bare.CDES == 0.7 && sub.Cdt1 == 11 && sub.Cdt2 == 2.5
                       && sub.Cl == 4.5 && sub.Ct == 2.5 && !sub.fe && sub.fwStar == 0.45 && sub.CDES == 0.75;
        std::snprintf(buf, sizeof(buf), "coefficients written bare in LES{} are read (Cdt1 %g Cdt2 %g Cl %g Ct %g fe "
                      "%d fwStar %g CDES %g), and the sub-dictionary is taken where there is one (Cdt1 %g)",
                      bare.Cdt1, bare.Cdt2, bare.Cl, bare.Ct, bare.fe ? 1 : 0, bare.fwStar, bare.CDES, sub.Cdt1);
        say(read, buf);
    }
    else
    {
        std::printf("FAIL: unknown mode %s\n", mode.c_str());
        return 2;
    }
    std::printf("test_sa_iddes_vs_openfoam %s (%s, %d failed)\n", fails == 0 ? "PASS" : "FAIL", mode.c_str(),
                fails);
    return fails == 0 ? 0 : 1;
}
