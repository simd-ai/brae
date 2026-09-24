// brae's articulated-body dynamics against REAL OpenFOAM's own rigidBodyMotion::solve, on
// RAS/floatingObject -- unit 4b, the last of the rigidBodyMotion port.
//
// THE ORACLE is tools/dumpRigidBodySolve: OpenFOAM's own RBD::rigidBodyMotion, constructed from the
// case's dictionary, seeded with a written joint state, handed a spatial force, and asked to solve.
// It reproduces the solver's own `report on` block digit for digit, so it is the dynamics isolated
// from the fluid and from the mesh -- which an end-to-end run is not.
//
// usage: test_rigid_body_dynamics_vs_openfoam <caseDir> <oracleLog>
#include "rigid_body_motion_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
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

// the oracle prints `[brae] <key> <numbers...>`; parens are separators
std::vector<scalar> oracleValues(const std::string& path, const std::string& key)
{
    std::ifstream in(path);
    std::string line;
    const std::string want = "[brae] " + key + " ";
    while (std::getline(in, line))
    {
        if (line.rfind(want, 0) != 0) continue;
        std::string rest = line.substr(want.size());
        for (char& ch : rest)
        {
            if (ch == '(' || ch == ')') ch = ' ';
        }
        std::istringstream is(rest);
        std::vector<scalar> v;
        scalar x = 0;
        while (is >> x) v.push_back(x);
        return v;
    }
    return {};
}

scalar worst(const std::vector<scalar>& a, const std::vector<scalar>& b, scalar& refMax)
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
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <oracleLog>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string log = argv[2];

    std::printf("== brae rigid-body dynamics vs OpenFOAM: %s ==\n", log.c_str());

    const RBD::MotionSpec spec = RBD::readMotionSpec(caseDir + "/constant/dynamicMeshDict");
    const std::size_t nD = static_cast<std::size_t>(spec.model.nDoF());

    // THE CHAIN, against the one OpenFOAM built. A `composite` joint is not an n-DoF joint: the model
    // inserts a massless jointBody for every sub-joint but the last, so this one body is THREE bodies
    // and the arrays must line up with that.
    const std::vector<scalar> nB = oracleValues(log, "nBodies");
    const std::vector<scalar> nDoF = oracleValues(log, "nDoF");
    check("the oracle ran and printed its model", !nB.empty() && !nDoF.empty());
    if (nB.empty() || nDoF.empty()) { std::printf("test_rigid_body_dynamics_vs_openfoam: %d failures\n", ++failures); return 1; }
    std::printf("  OpenFOAM: %d bodies, %d degrees of freedom; brae: %zu links, %zu\n",
                (int)nB[0], (int)nDoF[0], spec.model.links.size(), nD);
    check("brae's chain has OpenFOAM's number of bodies",
          spec.model.links.size() + 1 == static_cast<std::size_t>(nB[0]));
    check("...and its degrees of freedom", nD == static_cast<std::size_t>(nDoF[0]));

    // THE SPATIAL INERTIA of the body, row by row against OpenFOAM's own 6x6
    {
        const RBD::SpatialTensor I6 = RBD::inertiaTensor(spec.model.links.back().inertia);
        scalar w = 0, ref = 0;
        bool haveRows = true;
        for (int r = 0; r < 6; ++r)
        {
            const std::vector<scalar> row =
                oracleValues(log, "inertia " + std::to_string((int)nB[0] - 1) + " I row " + std::to_string(r));
            if (row.size() != 6) { haveRows = false; break; }
            for (int c = 0; c < 6; ++c)
            {
                w = std::fmax(w, std::fabs(I6(r, c) - row[static_cast<std::size_t>(c)]));
                ref = std::fmax(ref, std::fabs(row[static_cast<std::size_t>(c)]));
            }
        }
        check("the oracle printed the body's 6x6 spatial inertia", haveRows);
        std::printf("  spatial inertia: worst %.4e of %.4e\n", (double)w, (double)ref);
        check("brae's cuboid inertia is OpenFOAM's, all 36 components", haveRows && w == scalar(0));
    }

    // THE COEFFICIENTS brae READ OUT OF THE SAME DICTIONARY. The dynamics can only be compared at
    // OpenFOAM's own relaxation and damping, so a disagreement here would silently become a
    // disagreement in qDdot and be blamed on the algorithm.
    {
        const std::vector<scalar> ofG = oracleValues(log, "newmarkGamma");
        const std::vector<scalar> ofB = oracleValues(log, "newmarkBeta");
        const std::vector<scalar> ofD = oracleValues(log, "accelerationDamping");
        check("brae read OpenFOAM's Newmark gamma",
              ofG.size() == 1 && std::fabs(ofG[0] - spec.newmark.gamma) < scalar(1e-12));
        check("...its beta, through the clamp",
              ofB.size() == 1 && std::fabs(ofB[0] - spec.newmark.beta) < scalar(1e-12));
        check("...and its acceleration damping",
              ofD.size() == 1 && std::fabs(ofD[0] - spec.accelerationDamping) < scalar(1e-12));
    }

    // THE STATE OpenFOAM SOLVED FROM, and the force it was handed
    RBD::ModelState s0;
    s0.q = oracleValues(log, "preSolve.q");
    s0.qDot = oracleValues(log, "preSolve.qDot");
    s0.qDdot = oracleValues(log, "preSolve.qDdot");
    const std::vector<scalar> fxv = oracleValues(log, "fx " + std::to_string((int)nB[0] - 1));
    const std::vector<scalar> tNew = oracleValues(log, "solve.t");
    const std::vector<scalar> dt = oracleValues(log, "solve.deltaT");
    check("the oracle printed the state it solved from, the force and the clock",
          s0.q.size() == nD && s0.qDot.size() == nD && s0.qDdot.size() == nD
       && fxv.size() == 6 && tNew.size() == 1 && dt.size() == 1);
    if (fxv.size() != 6) { std::printf("test_rigid_body_dynamics_vs_openfoam: %d failures\n", ++failures); return 1; }
    s0.t = tNew[0];
    s0.deltaT = dt[0];

    std::vector<RBD::SpatialVector> fx(spec.model.links.size() + 1);
    fx.back().w = vector{fxv[0], fxv[1], fxv[2]};
    fx.back().l = vector{fxv[3], fxv[4], fxv[5]};
    // constant/g, as rigidBodyMeshMotion sets it on the model each call
    vector g{0, 0, 0};
    {
        const std::vector<scalar> gv = oracleValues(log, "g");
        if (gv.size() == 3) g = vector{gv[0], gv[1], gv[2]};
        else g = vector{0, 0, scalar(-9.81)};
    }

    // THE DYNAMICS, then the relaxation, then the integrator -- the three pieces of one solve
    const std::vector<scalar> raw = spec.model.forwardDynamics(s0.q, s0.qDot, fx, g);
    std::vector<scalar> qDdot = raw;
    const scalar aRelax = spec.accelerationRelaxation.value(tNew[0]);
    RBD::relaxAcceleration(qDdot, s0.qDdot, aRelax, spec.accelerationDamping);
    RBD::ModelState s;
    s.q.assign(nD, scalar(0));
    s.qDot.assign(nD, scalar(0));
    s.qDdot = qDdot;
    s.deltaT = dt[0];
    s.t = tNew[0];
    RBD::newmarkSolve(s0, spec.newmark, s);

    {
        const std::vector<scalar> ofR = oracleValues(log, "accelerationRelaxation");
        check("brae's acceleration relaxation is the one OpenFOAM applied at this time",
              ofR.size() == 1 && std::fabs(ofR[0] - aRelax) < scalar(1e-12));
    }

    std::printf("  aRelax %.17g, raw qDdot %.17g %.17g -> relaxed %.17g %.17g\n",
                (double)aRelax, (double)raw[0], (double)raw[1], (double)qDdot[0], (double)qDdot[1]);

    const std::vector<scalar> ofQDdot = oracleValues(log, "qDdot");
    const std::vector<scalar> ofQ = oracleValues(log, "q");
    const std::vector<scalar> ofQDot = oracleValues(log, "qDot");
    scalar refA = 0, refQ = 0, refV = 0;
    const scalar wA = worst(qDdot, ofQDdot, refA);
    const scalar wQ = worst(s.q, ofQ, refQ);
    const scalar wV = worst(s.qDot, ofQDot, refV);
    std::printf("  qDdot %.4e of %.4e (rel %.3e) | q %.4e of %.4e (rel %.3e) | qDot %.4e of %.4e "
                "(rel %.3e)\n",
                (double)wA, (double)refA, (double)(wA/std::fmax(refA, scalar(1e-300))),
                (double)wQ, (double)refQ, (double)(wQ/std::fmax(refQ, scalar(1e-300))),
                (double)wV, (double)refV, (double)(wV/std::fmax(refV, scalar(1e-300))));
    // THE FLOOR IS A CANCELLATION IN THE SECOND COMPONENT, not the port. The Ry generalised force is
    // the fluid moment about the origin minus the same moment transported to the joint -- two numbers
    // of size 129 whose difference is 0.24, so fifteen digits are lost before the division by the
    // inertia. An independent closed-form derivation of this 2-DoF system puts the same floor at
    // ~1e-13 relative; the first component, which has no cancellation, agrees to the last ulp.
    check("brae's joint acceleration is OpenFOAM's", wA/std::fmax(refA, scalar(1e-300)) < scalar(1e-13));
    check("...its q through the Newmark update", wQ/std::fmax(refQ, scalar(1e-300)) < scalar(1e-13));
    check("...and its qDot", wV/std::fmax(refV, scalar(1e-300)) < scalar(1e-13));

    // THE BODY'S PLACE IN SPACE, from the same q: the chain is what turns the joint state into a
    // transform, and a wrong lambda or XT would show here and nowhere in qDdot.
    {
        const std::vector<RBD::SpatialTransform> x0 = spec.model.X0(s.q);
        const RBD::SpatialTransform& X = x0[static_cast<std::size_t>(spec.model.bodyID())];
        const std::vector<scalar> ofR = oracleValues(log, "X0.r");
        const std::vector<scalar> ofE = oracleValues(log, "X0.E");
        scalar wr = 0, rr = 0;
        if (ofR.size() == 3)
        {
            const std::vector<scalar> mine{X.r.x, X.r.y, X.r.z};
            wr = worst(mine, ofR, rr);
        }
        scalar we = 0, re = 0;
        if (ofE.size() == 9)
        {
            const std::vector<scalar> mine{X.E.xx, X.E.xy, X.E.xz, X.E.yx, X.E.yy,
                                           X.E.yz, X.E.zx, X.E.zy, X.E.zz};
            we = worst(mine, ofE, re);
        }
        std::printf("  X0 of the body: r worst %.4e of %.4e, E worst %.4e\n",
                    (double)wr, (double)rr, (double)we);
        check("the chain puts the body where OpenFOAM puts it",
              ofR.size() == 3 && ofE.size() == 9 && wr < scalar(1e-15) && we < scalar(1e-15));
    }

    // CONTROL 1: gravity dropped. The body's centre of mass sits ABOVE its pivot, so weight is a
    // destabilising moment about the Ry axis -- but at q = 0 with the centre of mass on the axis it
    // contributes nothing, so this control is only live once the body has rotated. The arm says which.
    {
        const std::vector<scalar> noG = spec.model.forwardDynamics(s0.q, s0.qDot, fx, vector{0, 0, 0});
        scalar w = 0, ref = 0;
        const scalar wg = worst(noG, raw, ref);
        w = wg;
        std::printf("  CONTROL: gravity dropped: qDdot moves by %.4e of %.4e (q1 = %.3e)\n",
                    (double)w, (double)ref, (double)s0.q[1]);
        if (std::fabs(s0.q[1]) > scalar(1e-9))
        {
            check("...gravity is in the dynamics", w > scalar(1e-12)*std::fmax(ref, scalar(1e-300)));
        }
        else
        {
            std::printf("  (the body has not rotated yet, so gravity's moment about the Ry axis is "
                        "zero and this control cannot witness it here)\n");
        }
    }

    // CONTROL 2: the fluid load, as an EXACT IDENTITY rather than a norm. The Py joint slides along
    // y, gravity acts along z, and the centre-of-mass offset (0 0 0.25) stays in the x-z plane under
    // any Ry rotation -- so nothing but the fluid force's own y component can accelerate the first
    // joint, and it must do so at exactly F_y/m. A force applied about the wrong axis, to the wrong
    // body, or transported to the joint with the wrong arm each miss this by a factor. A norm would
    // not have said so: once the body has rotated, gravity's moment about the Ry axis is forty times
    // the fluid's contribution to it, and a "the fluid load dominates" control reads as a failure on
    // a state that is perfectly correct.
    {
        const std::vector<RBD::SpatialVector> none(fx.size());
        const std::vector<scalar> noF = spec.model.forwardDynamics(s0.q, s0.qDot, none, g);
        const scalar m = spec.model.links.back().inertia.m;
        const scalar expect = fx.back().l.y/m;
        scalar ref = 0;
        const scalar w = worst(noF, raw, ref);
        std::printf("  CONTROL: the sliding joint %.17g vs F_y/m %.17g; with no fluid force %.17g; "
                    "qDdot moves by %.4e\n",
                    (double)raw[0], (double)expect, (double)noF[0], (double)w);
        check("the fluid force accelerates the sliding joint at exactly F_y/m",
              std::fabs(raw[0] - expect) <= scalar(1e-13)*std::fabs(expect));
        check("...and with no fluid force that joint does not move at all", noF[0] == scalar(0));
        check("...so the fluid load is in the answer",
              w > scalar(1e-6)*std::fmax(ref, scalar(1e-300)));
    }

    // CONTROL 3: the moment read as the linear half and the force as the angular one -- the one
    // mistake the component order invites, and both halves are three-vectors of plausible size.
    {
        std::vector<RBD::SpatialVector> swapped(fx.size());
        swapped.back().w = fx.back().l;
        swapped.back().l = fx.back().w;
        const std::vector<scalar> sw = spec.model.forwardDynamics(s0.q, s0.qDot, swapped, g);
        scalar ref = 0;
        const scalar w = worst(sw, raw, ref);
        std::printf("  CONTROL: the moment and the force exchanged: qDdot moves by %.4e of %.4e\n",
                    (double)w, (double)ref);
        check("...the spatial force is angular-first", w > scalar(1)*std::fmax(ref, scalar(1e-300)));
    }

    std::printf("test_rigid_body_dynamics_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
