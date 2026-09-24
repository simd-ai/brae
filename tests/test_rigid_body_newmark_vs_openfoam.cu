// brae's Newmark integrator and acceleration relaxation against REAL OpenFOAM's own joint state, on
// RAS/floatingObject -- the integrator half of unit 4 of the rigidBodyMotion port.
//
// WHAT IS UNDER TEST. Given the state at the start of a step and the acceleration OpenFOAM's dynamics
// produced for it, reproduce the q and qDot OpenFOAM wrote. The acceleration itself -- the articulated
// body algorithm that turns the fluid force into qDdot -- is the other half and is not this gate's.
//
// usage: test_rigid_body_newmark_vs_openfoam <caseDir> <gamma> <beta>
#include "foam_token_reader.cuh"
#include "rigid_body_motion_cpp.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf("  %s:   %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// <time>/uniform/rigidBodyMotionState: `q N ( ... ); qDot ...; qDdot ...; t ...; deltaT ...;`
bool readState(const std::string& path, RBD::ModelState& s)
{
    if (!std::filesystem::exists(path)) return false;
    TokenStream ts(path);
    auto list = [&](std::vector<scalar>& v)
    {
        const label n = ts.nextLabel();
        ts.expect("(");
        v.assign(static_cast<std::size_t>(n), scalar(0));
        for (scalar& x : v) x = ts.nextScalar();
        ts.expect(")");
    };
    bool haveQ = false;
    while (!ts.eof())
    {
        const std::string k = ts.next();
        if (k == "q")           { list(s.q); haveQ = true; }
        else if (k == "qDot")   { list(s.qDot); }
        else if (k == "qDdot")  { list(s.qDdot); }
        else if (k == "t")      { s.t = ts.nextScalar(); }
        else if (k == "deltaT") { s.deltaT = ts.nextScalar(); }
    }
    return haveQ;
}

scalar worstOf(const std::vector<scalar>& a, const std::vector<scalar>& b, scalar& refMax)
{
    scalar w = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        w = std::fmax(w, std::fabs(a[i] - b[i]));
        refMax = std::fmax(refMax, std::fabs(b[i]));
    }
    return w;
}

}   // namespace


int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <gamma> <beta>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const scalar wantGamma = static_cast<scalar>(std::atof(argv[2]));
    const scalar wantBeta = static_cast<scalar>(std::atof(argv[3]));

    std::printf("== brae Newmark vs OpenFOAM: %s ==\n", caseDir.c_str());

    const RBD::MotionSpec spec = RBD::readMotionSpec(caseDir + "/constant/dynamicMeshDict");
    std::printf("  read: Newmark gamma %.17g beta %.17g, accelerationDamping %.17g, report %d\n",
                (double)spec.newmark.gamma, (double)spec.newmark.beta,
                (double)spec.accelerationDamping, (int)spec.report);
    check("the solver's gamma is the dictionary's", spec.newmark.gamma == wantGamma);
    // beta is CLAMPED from below by gamma: beta = max(0.25*(gamma + 0.5)^2, the entry). The expected
    // value is compared with a tolerance and not for equality: at gamma 0.9 the clamp evaluates
    // 0.25*(1.4)^2 = 0.48999999999999999112, and the decimal 0.49 this arm is given is a different
    // double. Asserting equality there would be asserting how the argument was spelled.
    std::printf("  beta: %.17g (the arm expects %.17g)\n", (double)spec.newmark.beta, (double)wantBeta);
    check("...and beta is the clamp of the entry, not the entry",
          std::fabs(spec.newmark.beta - wantBeta) <= scalar(1e-15)*std::fmax(wantBeta, scalar(1)));

    // the time directories OpenFOAM wrote, in order
    std::vector<std::pair<scalar, std::string>> times;
    for (const auto& e : std::filesystem::directory_iterator(caseDir))
    {
        if (!e.is_directory()) continue;
        const std::string n = e.path().filename().string();
        if (n.empty() || !(std::isdigit(static_cast<unsigned char>(n[0])))) continue;
        if (!std::filesystem::exists(e.path() / "uniform" / "rigidBodyMotionState")) continue;
        times.emplace_back(static_cast<scalar>(std::atof(n.c_str())), n);
    }
    std::sort(times.begin(), times.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::printf("  %zu written states, t = %g .. %g\n", times.size(),
                times.empty() ? 0.0 : (double)times.front().first,
                times.empty() ? 0.0 : (double)times.back().first);
    check("OpenFOAM wrote a state at every step, so consecutive pairs are one deltaT apart",
          times.size() >= 5);
    if (times.size() < 5) { std::printf("test_rigid_body_newmark_vs_openfoam: %d failures\n", ++failures); return 1; }

    std::vector<RBD::ModelState> states(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
    {
        check("...and each one parses",
              readState(caseDir + "/" + times[i].second + "/uniform/rigidBodyMotionState", states[i]));
    }
    check("the chain has the state's number of degrees of freedom",
          states.front().q.size() == static_cast<std::size_t>(spec.model.nDoF()));
    // THE FIRST STEP HAS NO PREDECESSOR ON DISK: OpenFOAM's `0` directory is a copy of `0.orig` and
    // carries no uniform/. The state it integrates from is rigidBodyModelState's own default, which is
    // zero in q, qDot and qDdot (rigidBodyModelState.C:46-71) -- so the first pair below is built from
    // that, and it is a real arm: it is where the body leaves rest.
    RBD::ModelState rest;
    rest.q.assign(states.front().q.size(), scalar(0));
    rest.qDot = rest.q;
    rest.qDdot = rest.q;

    scalar worstQ = 0, worstQDot = 0, refQ = 0, refQDot = 0;
    std::size_t nPairs = 0;
    for (std::size_t i = 0; i < states.size(); ++i)
    {
        const RBD::ModelState& s0 = (i == 0) ? rest : states[i - 1];
        // ...if the run starts at 0 with a written state there, the first entry IS the rest state and
        // has no predecessor to integrate from
        if (i == 0 && std::fabs(times[i].first) < scalar(1e-30)) continue;
        RBD::ModelState s;
        s.q.assign(states[i].q.size(), scalar(0));
        s.qDot = s.q;
        s.qDdot = states[i].qDdot;        // OpenFOAM's own acceleration, relaxed as it wrote it
        s.deltaT = states[i].deltaT;
        s.t = states[i].t;
        RBD::newmarkSolve(s0, spec.newmark, s);
        worstQ = std::fmax(worstQ, worstOf(s.q, states[i].q, refQ));
        worstQDot = std::fmax(worstQDot, worstOf(s.qDot, states[i].qDot, refQDot));
        ++nPairs;
    }
    std::printf("  %zu steps integrated: worst |dq| %.4e of %.4e, worst |dqDot| %.4e of %.4e\n",
                nPairs, (double)worstQ, (double)refQ, (double)worstQDot, (double)refQDot);
    // THE FLOOR IS FMA CONTRACTION, not a different scheme. The identity is exact in exact
    // arithmetic -- checked in decimal against OpenFOAM's written state, residual 0 on all 20
    // (step, component) pairs -- and what is left here is the compiler contracting
    // `dt*qDot0 + dt*dt*(...)` into a fused multiply-add where OpenFOAM's Field operators cannot.
    // MEASURED: 4.9e-17 of q and 6.8e-17 of qDot, a quarter of an ulp.
    const scalar relQ = worstQ/std::fmax(refQ, scalar(1e-300));
    const scalar relQDot = worstQDot/std::fmax(refQDot, scalar(1e-300));
    std::printf("  relative: q %.3e, qDot %.3e\n", (double)relQ, (double)relQDot);
    check("brae's Newmark reproduces OpenFOAM's q, step for step", relQ < scalar(1e-15));
    check("...and its qDot", relQDot < scalar(1e-15));
    check("...over every step of the run, not just the first", nPairs >= 5);

    // CONTROL 1: the acceleration weights swapped -- gamma <-> 1 - gamma and beta <-> 0.5 - beta. At
    // the DEFAULTS these are the same numbers, so this control is only live on the gamma arm; the
    // script says which arm it is on, and the check below states it rather than passing quietly.
    {
        RBD::NewmarkCoeffs flipped;
        flipped.gamma = scalar(1) - spec.newmark.gamma;
        flipped.beta = scalar(0.5) - spec.newmark.beta;
        scalar w = 0, ref = 0;
        for (std::size_t i = 1; i < states.size(); ++i)
        {
            RBD::ModelState s;
            s.q.assign(states[i].q.size(), scalar(0));
            s.qDot = s.q;
            s.qDdot = states[i].qDdot;
            s.deltaT = states[i].deltaT;
            RBD::newmarkSolve(states[i - 1], flipped, s);
            w = std::fmax(w, worstOf(s.q, states[i].q, ref));
        }
        const bool symmetric = (spec.newmark.gamma == scalar(0.5) && spec.newmark.beta == scalar(0.25));
        std::printf("  CONTROL: the two acceleration weights swapped: worst |dq| %.4e%s\n",
                    (double)w, symmetric ? "  (the defaults are symmetric, so this cannot witness)" : "");
        if (!symmetric)
        {
            check("...the weights are not interchangeable", w > scalar(1e-3)*std::fmax(ref, scalar(1e-300)));
        }
    }

    // CONTROL 2: the old acceleration dropped -- q_{n+1} = q_n + dt*qDot_n + beta*dt^2*qDdot_{n+1},
    // which is what a reading of "Newmark uses the new acceleration" gives.
    {
        scalar w = 0, ref = 0;
        for (std::size_t i = 1; i < states.size(); ++i)
        {
            const RBD::ModelState& s0 = states[i - 1];
            const scalar dt = states[i].deltaT;
            for (std::size_t k = 0; k < states[i].q.size(); ++k)
            {
                const scalar qq = s0.q[k] + dt*s0.qDot[k] + dt*dt*spec.newmark.beta*states[i].qDdot[k];
                w = std::fmax(w, std::fabs(qq - states[i].q[k]));
                ref = std::fmax(ref, std::fabs(states[i].q[k]));
            }
        }
        // The body accelerates smoothly, so consecutive accelerations are close and dropping one
        // moves q by 3.0e-04 of itself -- twelve orders above the agreement above, which is what
        // makes it a control, but nothing like 100%. Stated at its measured size.
        const scalar rc2 = w/std::fmax(ref, scalar(1e-300));
        std::printf("  CONTROL: the START-of-step acceleration dropped: worst |dq| %.4e of %.4e "
                    "(relative %.3e)\n", (double)w, (double)ref, (double)rc2);
        check("...both accelerations are in the update",
              rc2 > scalar(1e-5) && rc2 > scalar(1e6)*std::fmax(relQ, scalar(1e-18)));
    }

    // CONTROL 3: the body actually moved, so none of the above is a comparison of zeros
    {
        scalar mq = 0, mqd = 0, mqdd = 0;
        for (const RBD::ModelState& s : states)
        {
            for (std::size_t k = 0; k < s.q.size(); ++k)
            {
                mq = std::fmax(mq, std::fabs(s.q[k]));
                mqd = std::fmax(mqd, std::fabs(s.qDot[k]));
                mqdd = std::fmax(mqdd, std::fabs(s.qDdot[k]));
            }
        }
        std::printf("  CONTROL: the body moved: max |q| %.4e, |qDot| %.4e, |qDdot| %.4e\n",
                    (double)mq, (double)mqd, (double)mqdd);
        check("...the run is not a body at rest", mq > scalar(1e-6) && mqdd > scalar(1e-3));
    }

    // THE RELAXATION, which is the other thing this dictionary carries. It is applied to the
    // acceleration BEFORE the integrator sees it, so this gate holds the formula and the Function1,
    // and the script's `relax1` arm holds the measured 0.7 against an unrelaxed twin.
    {
        std::vector<scalar> a{scalar(2), scalar(-4)};
        const std::vector<scalar> prev{scalar(1), scalar(1)};
        RBD::relaxAcceleration(a, prev, scalar(0.7), scalar(1));
        const bool ok = std::fabs(a[0] - (scalar(0.7)*scalar(2) + scalar(0.3))) < scalar(1e-15)
                     && std::fabs(a[1] - (scalar(0.7)*scalar(-4) + scalar(0.3))) < scalar(1e-15);
        check("the relaxation blends the new acceleration with the one the state carried", ok);
        std::printf("  accelerationRelaxation(t): %.17g at t=0, %.17g at t=3.999, %.17g at t=4.5\n",
                    (double)spec.accelerationRelaxation.value(scalar(0)),
                    (double)spec.accelerationRelaxation.value(scalar(3.999)),
                    (double)spec.accelerationRelaxation.value(scalar(4.5)));
    }

    std::printf("test_rigid_body_newmark_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
