// realizableKE's rCmu on the device against the host reference, cell for cell, at strain rates where the
// constant in its denominator decides the answer.
//
// realizableKE.C:53-59 forms W = 2*sqrt(2)*((S&S)&&S)/(magS*S2 + SMALL) and SMALL is 1e-15 in OpenFOAM's
// double build (doubleScalar.H:62). The device kernel added 1e-37 (the float build's VSMALL) while the host
// reference (realizableKE_cpp.cu, which ctest realizablke_cpp holds to OpenFOAM) adds 1e-15 -- found
// 2026-10-05 by a sweep for constants. On a resolved flow the two agree to the last bit or two, which is how
// it stood: every gate of this model runs strain rates of order one and above.
// THE FIXTURE is not a flow: 400 cells whose velocity gradient is a fixed general tensor scaled from 1e-8 to
// 1e+2 /s, so that magS*S2 runs from far below 1e-15 to far above it, and whose turbulent time scale k/eps
// is the strain's own inverse -- so that As*Us*k/eps is of order one in every cell and rCmu = 1/(A0 + that)
// SHOWS As. (With a time scale of 0.025 s, the first fixture tried, a strain of 1e-5 /s leaves rCmu at 1/A0
// whatever As is: the old constant moved it by 8.5e-09 at most, which is also the size of the defect on a
// flow with such a time scale.) Counted as premises: the cells where the two constants give quotients more
// than 1% apart, and the cells where they cannot differ at all.
// THE CONTROL is the kernel's own switch, read once a process, so it is this binary run again as a child
// (`child` prints the one number): BRAE_CONTROL_RKE_SMALL_FLOAT=1 must leave the host by orders.
#include "cf_types.cuh"
#include "device_kepsilon.cuh"
#include "realizableKE_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace brae;

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

struct Measured
{
    // max |device - host|/host of rCmu over the cells
    scalar worst = -1;
    // cells where 1e-15 and 1e-37 give W more than 1% apart
    long decided = 0;
    // cells where magS*S2 is above 1e+3: the constant is rounded away
    long indifferent = 0;
};

Measured run()
{
    const int nC = 400;
    std::vector<tensor> gradU(static_cast<std::size_t>(nC));
    std::vector<scalar> k(static_cast<std::size_t>(nC));
    std::vector<scalar> eps(static_cast<std::size_t>(nC));
    // one general tensor (no symmetry, a non-zero trace), a different sign pattern every third cell
    const tensor base{scalar(0.31), scalar(-0.72), scalar(0.44), scalar(0.18), scalar(-0.27), scalar(0.93),
                      scalar(-0.61), scalar(0.35), scalar(0.52)};
    for (int c = 0; c < nC; ++c)
    {
        const scalar s = std::pow(scalar(10), scalar(-8) + scalar(10)*scalar(c)/scalar(nC - 1));
        const scalar flip = (c % 3 == 0) ? scalar(-1) : scalar(1);
        tensor t = base;
        t.xy *= flip;
        t.zx *= flip;
        const std::size_t i = static_cast<std::size_t>(c);
        gradU[i] = tensor{s*t.xx, s*t.xy, s*t.xz, s*t.yx, s*t.yy, s*t.yz, s*t.zx, s*t.zy, s*t.zz};
        k[i] = scalar(0.375)*(scalar(1) + scalar(0.5)*std::sin(scalar(0.37)*scalar(c)));
        eps[i] = k[i]*s*(scalar(1) + scalar(0.5)*std::cos(scalar(0.23)*scalar(c)));
    }
    RealizableKECoeffs co;
    const std::vector<scalar> s2 = cpu::realizableKE::S2(gradU);
    const std::vector<scalar> host = cpu::realizableKE::rCmu(gradU, s2, k, eps, co);

    // the device's layout: component q of every cell, then the next component
    std::vector<scalar> flat(static_cast<std::size_t>(9*nC));
    for (int c = 0; c < nC; ++c)
    {
        const tensor& t = gradU[static_cast<std::size_t>(c)];
        const scalar q[9] = {t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz};
        for (int j = 0; j < 9; ++j)
        {
            flat[static_cast<std::size_t>(j*nC + c)] = q[j];
        }
    }
    DeviceBuffer<scalar> dGradU(flat);
    DeviceBuffer<scalar> dK(k);
    DeviceBuffer<scalar> dEps(eps);
    DeviceBuffer<scalar> dRCmu;
    DeviceBuffer<scalar> dMagS;
    deviceRealizableStrain(dGradU, dK, dEps, co.A0, nC, dRCmu, dMagS);
    std::vector<scalar> device;
    dRCmu.copyTo(device);

    Measured out;
    out.worst = 0;
    for (int c = 0; c < nC; ++c)
    {
        const std::size_t i = static_cast<std::size_t>(c);
        out.worst = std::fmax(out.worst, std::fabs(device[i] - host[i])/std::fabs(host[i]));
        const scalar denom = std::sqrt(s2[i])*s2[i];
        out.decided += (scalar(1.0e-15) > scalar(0.01)*denom) ? 1 : 0;
        out.indifferent += (denom > scalar(1.0e+3)) ? 1 : 0;
    }
    return out;
}

// this binary again, as a child with the switch set: the number its `child` line prints, or -1
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
        controlSaid = controlSaid || std::string(line).find("CONTROL MODE") != std::string::npos;
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
    std::printf("== realizableKE: rCmu on the device against the host reference, strain rates from 1e-8 to 1e+2\n");
    const Measured d = run();
    std::printf("  400 cells: in %ld the constant decides the quotient by more than 1%%, in %ld it is rounded "
                "away; device against host %.3e\n", d.decided, d.indifferent, static_cast<double>(d.worst));
    // MEASURED 2026-10-06: device against host 4.2e-16 (137 cells decided by the constant, 50 where it is
    // rounded away); with 1e-37, 6.6e-02. The bound is a decade above the first.
    // DOES NOT CLAIM the constant against OpenFOAM at these strain rates: the oracle is the host reference,
    // whose 1e-15 is realizableKE.C:53-59's SMALL by reading -- OpenFOAM's own gate of the model
    // (realizablke_cpp) runs strain rates of order one, where the two constants differ in a last bit at most.
    check("the device's rCmu is the host's in every cell, the ones the constant decides included",
          d.worst >= scalar(0) && d.worst < scalar(5e-15) && d.decided > 50 && d.indifferent > 20);
    const scalar wrong = childWorst(argv[0], "BRAE_CONTROL_RKE_SMALL_FLOAT");
    std::printf("  CONTROL, 1e-37 for SMALL: device against host %.3e\n", static_cast<double>(wrong));
    check("CONTROL: with the float build's constant the device leaves the host by orders", wrong > scalar(1e-3));
    std::printf("test_device_realizable_strain: %d failures\n", failures);
    return failures ? 1 : 0;
}
