// brae's interFoam against REAL OpenFOAM's on RAS/DTCHullMoving with the mesh FROZEN: the tutorial's
// atmosphere `pressureInletOutletVelocity` with its `tangentialVelocity`, and the fields it drives. The
// script (tests/interfoam_dtchullmoving_vs_openfoam.sh) says what is staged and why.
//
// usage: test_inter_dtchullmoving_vs_openfoam <caseDir> <ofCaseDir> <nSteps> <lastTime> <ofLog> [measure]
//   caseDir     the staged case, meshed, with its 0 directory
//   ofCaseDir   the same case after OpenFOAM ran it, one time directory per step
//   lastTime    the name of OpenFOAM's time directory after nSteps
//   ofLog       OpenFOAM's log, for the solves
//   measure     print every number and assert nothing but the path -- for setting bounds
//   moving      the tutorial's own rigid-body motion: the joint state and the moved points are compared
//               too, and grad(U) is cached as the case says (inert on a changing mesh)
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "inter_solve_log.cuh"
#include "device_gate_finite.cuh"
#include "patch_entry_lookup.cuh"
#include "rigid_body_motion_cpp.cuh"
#include "dynamic_motion_solver_fv_mesh_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {

int failures = 0;
bool measureOnly = false;

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

void bound(
    const std::string& what,
    scalar value,
    scalar limit)
{
    if (measureOnly)
    {
        std::printf("  meas: %-56s %.4e\n", what.c_str(), (double)value);
        return;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%-56s %.4e (bound %.1e)", what.c_str(), (double)value, (double)limit);
    check(buf, value <= limit);
}

// THE BOUNDS, from ten steps with the entry, each about three times its measurement: alpha 3.0e-14, p_rgh
// 1.3e-11, U 2.9e-13, k 3.2e-13, omega 6.2e-13, nut 7.7e-12, the atmosphere's U face by face 3.9e-15 and its
// p_rgh 0; every p_rgh and alpha count OpenFOAM's, the p_rgh initial residuals 8.3e-12. OpenFOAM against
// itself with one ulp of the entry reads U 4.2e-14, p_rgh 3.4e-12 at the tenth step.
constexpr scalar BOUND_ALPHA = 1e-13;
constexpr scalar BOUND_PRGH = 4e-11;
constexpr scalar BOUND_U = 1e-12;
constexpr scalar BOUND_K = 1e-12;
constexpr scalar BOUND_OMEGA = 2e-12;
constexpr scalar BOUND_NUT = 3e-11;
constexpr scalar BOUND_ATM_U = 1e-14;
constexpr scalar BOUND_ATM_PRGH = 1e-14;
constexpr scalar BOUND_P_RESIDUAL = 3e-11;
constexpr scalar BOUND_ALPHA_RESIDUAL = 1e-12;
// ...and the `moving` profile: the tutorial's own rigid-body motion, SIX steps -- before the step-8 branch
// OpenFOAM itself takes on one ulp (see the script). Measured: q 2.2e-15, qDot 6.2e-15, qDdot 8.6e-15, the
// moved points 2.5e-17 of the extent, alpha 1.1e-13, p_rgh 1.1e-12, U 4.8e-13, k 3.3e-13, omega 1.0e-12,
// nut 3.2e-12, the atmosphere's U 2.0e-15 and p_rgh 0; every p_rgh and alpha count OpenFOAM's. Each bound
// about three times; the fields not listed here keep the frozen profile's.
constexpr scalar BOUND_BODY = 3e-14;
constexpr scalar BOUND_POINTS = 1e-16;
constexpr scalar MOVING_BOUND_ALPHA = 4e-13;

scalar relInf(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    scalar w = 0;
    scalar ref = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        w = std::fmax(w, std::fabs(a[i] - b[i]));
        ref = std::fmax(ref, std::fabs(b[i]));
    }
    return (a.size() == b.size()) ? w/std::fmax(ref, scalar(1e-300)) : scalar(1);
}

scalar relInf(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    scalar w = 0;
    scalar ref = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        w = std::fmax(w, mag(vector{a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z}));
        ref = std::fmax(ref, mag(b[i]));
    }
    return (a.size() == b.size()) ? w/std::fmax(ref, scalar(1e-300)) : scalar(1);
}

// an ascii polyMesh/points file, as OpenFOAM writes the moved mesh at writeFormat ascii
std::vector<vector> readPointsFile(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    static const std::regex head(R"(\n\s*([0-9]+)\s*\n?\()");
    std::smatch mh;
    std::vector<vector> out;
    if (!std::regex_search(text, mh, head)) return out;
    const std::size_t n = static_cast<std::size_t>(std::atol(mh[1].str().c_str()));
    const char* p = text.c_str() + mh.position(0) + mh.length(0);
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        while (*p && *p != '(')
        {
            ++p;
        }
        if (!*p) break;
        ++p;
        char* end = nullptr;
        vector v;
        v.x = std::strtod(p, &end);
        p = end;
        v.y = std::strtod(p, &end);
        p = end;
        v.z = std::strtod(p, &end);
        p = end;
        out.push_back(v);
    }
    return out;
}

template<class T>
std::vector<T> readCells(
    const std::string& path,
    label nC)
{
    const FieldData<T> fd = readField<T>(path);
    return fd.internalUniform ? std::vector<T>(static_cast<std::size_t>(nC), fd.internalUniformValue)
                              : fd.internalField;
}

template<class T>
std::vector<T> patchValues(
    const FieldData<T>& fd,
    const FvPatch& p)
{
    const PatchFieldData<T>* b = findPatchEntry(fd.boundary, p);
    if (!b)
    {
        return {};
    }
    const std::size_t n = static_cast<std::size_t>(p.size);
    return b->valueUniform ? std::vector<T>(n, b->uniformValue) : b->values;
}

} // namespace


int main(
    int argc,
    char** argv)
{
    if (argc < 6)
    {
        std::printf("  SKIP: usage: %s <caseDir> <ofCaseDir> <nSteps> <lastTime> <ofLog> [measure]\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string ofCase = argv[2];
    const label nSteps = static_cast<label>(std::atol(argv[3]));
    const std::string last = ofCase + "/" + argv[4];
    const std::string logPath = argv[5];
    bool moving = false;
    for (int a = 6; a < argc; ++a)
    {
        measureOnly = measureOnly || std::string(argv[a]) == "measure";
        moving = moving || std::string(argv[a]) == "moving";
    }

    std::printf("== brae interFoam vs OpenFOAM interFoam: RAS/DTCHullMoving, %s ==\n",
                moving ? "as shipped, the hull moving" : "mesh frozen");
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    // non-const, and handed to the driver as a MutableMesh: the rigid body moves them, and the fields hold
    // references to these patches
    std::vector<FvPatch> patches = buildPatches(m, g);
    MutableMesh mutableMesh;
    mutableMesh.m = &m;
    mutableMesh.g = &g;
    mutableMesh.patches = &patches;
    const label nC = m.nCells();
    std::printf("  mesh: %d cells\n", (int)nC);

    std::size_t atm = patches.size();
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].name == "atmosphere") atm = pi;
    }
    check("the mesh has the tutorial's `atmosphere` patch", atm < patches.size());
    if (atm == patches.size())
    {
        std::printf("test_inter_dtchullmoving_vs_openfoam: %d failure(s)\n", failures);
        return 1;
    }

    // does the STAGED case carry the entry? The control strips it from brae's copy only.
    const FieldData<vector> staged = readField<vector>(caseDir + "/0/U");
    const PatchFieldData<vector>* stagedAtm = findPatchEntry(staged.boundary, patches[atm]);
    const bool caseHasTv = stagedAtm && stagedAtm->hasTangentialVelocity;
    std::printf("  the staged atmosphere %s a tangentialVelocity\n", caseHasTv ? "carries" : "does NOT carry");

    InterFields fin;
    const RunReport r = runInterFoam(caseDir, caseDir + "/0", m, g, patches, nSteps, /*verbose=*/true, &fin,
                                     scalar(1.0e300), nullptr, &mutableMesh);

    // THE PATH
    check("brae ran the same number of steps", r.steps == nSteps);
    check("...under Euler, not localEuler", !fin.lts);
    if (moving)
    {
        // the case caches grad(U); on a changing mesh OpenFOAM bypasses the cache (gradScheme.C:99) and
        // moving() stays set from the first move (polyMesh.C:1191)
        check("...with grad(U) cached as the case says", fin.gradUCache.on);
        check("...on a mesh the rigid body moves",
              fin.dynamicMesh && fin.dynamicMesh->rigidBody() != nullptr && fin.dynamicMesh->moving());
    }
    else
    {
        check("...with grad(U) not cached (the gate stages `active false`, both sides)", !fin.gradUCache.on);
    }
    check("...under kOmegaSST", fin.turbulence.on && fin.turbulence.model == InterRasModel::KOmegaSST);
    check("brae's atmosphere carries the tangentialVelocity exactly where the staged case does",
          (fin.U.boundary[atm]->tangentialVelocityPtr() != nullptr) == caseHasTv);

    // THE FIXTURE CAN WITNESS: OpenFOAM's atmosphere takes INFLOW at the gated step, and the entry acts
    // only there (valueFraction = neg(phi)*(I - nn))
    {
        const FieldData<scalar> ofPhi = readField<scalar>(last + "/phi");
        const std::vector<scalar> phiAtm = patchValues(ofPhi, patches[atm]);
        long nIn = 0;
        for (const scalar v : phiAtm)
        {
            nIn += (v < scalar(0)) ? 1 : 0;
        }
        std::printf("  OpenFOAM's atmosphere: %ld of %zu faces inflow at the last step\n", nIn, phiAtm.size());
        check("OpenFOAM's atmosphere has inflow faces, so the entry is live", nIn > 0);
    }

    // THE BODY, from the joint state OpenFOAM writes every step, and THE MESH it moved
    if (moving && fin.dynamicMesh && fin.dynamicMesh->rigidBody())
    {
        const RBD::ModelState& st = fin.dynamicMesh->rigidBody()->state();
        const std::string sp = last + "/uniform/rigidBodyMotionState";
        const std::vector<scalar> ofQ = RBD::readJointStateList(sp, "q");
        const std::vector<scalar> ofV = RBD::readJointStateList(sp, "qDot");
        const std::vector<scalar> ofA = RBD::readJointStateList(sp, "qDdot");
        check("OpenFOAM wrote the joint state", ofQ.size() == st.q.size() && !ofQ.empty());
        scalar wq = 0, wv = 0, wa = 0, rq = 0, rv = 0, ra = 0;
        for (std::size_t i = 0; i < st.q.size() && i < ofQ.size() && i < ofV.size() && i < ofA.size(); ++i)
        {
            wq = std::fmax(wq, std::fabs(st.q[i] - ofQ[i]));
            rq = std::fmax(rq, std::fabs(ofQ[i]));
            wv = std::fmax(wv, std::fabs(st.qDot[i] - ofV[i]));
            rv = std::fmax(rv, std::fabs(ofV[i]));
            wa = std::fmax(wa, std::fabs(st.qDdot[i] - ofA[i]));
            ra = std::fmax(ra, std::fabs(ofA[i]));
        }
        std::printf("  the body: q %.4e of %.4e, qDot %.4e of %.4e, qDdot %.4e of %.4e\n",
                    (double)wq, (double)rq, (double)wv, (double)rv, (double)wa, (double)ra);
        check("the body has moved, so the comparison is not one of zeros", rq > scalar(1e-9));
        bound("the joint position q, relative", wq/std::fmax(rq, scalar(1e-300)), BOUND_BODY);
        bound("...its velocity qDot, relative", wv/std::fmax(rv, scalar(1e-300)), BOUND_BODY);
        bound("...its acceleration qDdot, relative", wa/std::fmax(ra, scalar(1e-300)), BOUND_BODY);

        const std::vector<vector> ofPoints = readPointsFile(last + "/polyMesh/points");
        scalar extent = 0;
        scalar dPoint = 0;
        for (std::size_t i = 0; i < ofPoints.size() && i < m.points().size(); ++i)
        {
            extent = std::fmax(extent, mag(ofPoints[i]));
            dPoint = std::fmax(dPoint, mag(m.points()[i] - ofPoints[i]));
        }
        check("OpenFOAM wrote the moved mesh, one point per brae point",
              !ofPoints.empty() && ofPoints.size() == m.points().size());
        bound("the moved points, relative to the mesh's extent", dPoint/std::fmax(extent, scalar(1e-300)),
              BOUND_POINTS);
    }

    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    bound("alpha, relative to its largest value",
          relInf(fin.alpha1.internal, readCells<scalar>(last + "/alpha.water", nC)),
          moving ? MOVING_BOUND_ALPHA : BOUND_ALPHA);
    bound("p_rgh, relative", relInf(fin.p_rgh.internal, readCells<scalar>(last + "/p_rgh", nC)), BOUND_PRGH);
    bound("U, relative", relInf(fin.U.internal, readCells<vector>(last + "/U", nC)), BOUND_U);
    if (fin.turbulence.on)
    {
        failures += brae::gatecheck::nonFinite("brae k", fin.turbulence.k.internal);
        failures += brae::gatecheck::nonFinite("brae omega", fin.turbulence.omega.internal);
        bound("k, relative", relInf(fin.turbulence.k.internal, readCells<scalar>(last + "/k", nC)), BOUND_K);
        bound("omega, relative",
              relInf(fin.turbulence.omega.internal, readCells<scalar>(last + "/omega", nC)), BOUND_OMEGA);
        bound("nut, relative", relInf(fin.turbulence.nut.internal, readCells<scalar>(last + "/nut", nC)), BOUND_NUT);
    }

    // THE ATMOSPHERE, face by face: U's value is directionMixed's evaluate with the refValue on every
    // inflow face, and p_rgh's totalPressure reads |U_b|^2 there -- 0.5*rho*1.668^2 of it on air
    {
        const std::vector<vector> ofU = patchValues(readField<vector>(last + "/U"), patches[atm]);
        const std::vector<scalar> ofP = patchValues(readField<scalar>(last + "/p_rgh"), patches[atm]);
        failures += brae::gatecheck::nonFinite("brae atmosphere U", fin.U.boundary[atm]->value());
        bound("atmosphere U, face by face, relative to its largest",
              relInf(fin.U.boundary[atm]->value(), ofU), BOUND_ATM_U);
        bound("atmosphere p_rgh, face by face, relative to its largest",
              relInf(fin.p_rgh.boundary[atm]->value(), ofP), BOUND_ATM_PRGH);
    }

    // the solves
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    const std::vector<LinearSolveRecord> ofA = brae::gatecheck::readOfSolves(logPath, fin.alphaName);
    check("OpenFOAM's log gave the p_rgh and alpha solves",
          !ofP.empty() && ofA.size() >= static_cast<std::size_t>(nSteps));
    failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh",
                                               BOUND_P_RESIDUAL, BOUND_P_RESIDUAL, scalar(-1), nullptr,
                                               !measureOnly);
    failures += brae::gatecheck::compareSolves("host", r.alphaSolves, ofA, nSteps, fin.alphaName.c_str(),
                                               BOUND_ALPHA_RESIDUAL, BOUND_ALPHA_RESIDUAL, scalar(1e-5), nullptr,
                                               !measureOnly);

    std::printf("test_inter_dtchullmoving_vs_openfoam: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
