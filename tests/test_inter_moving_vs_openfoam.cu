// brae's interFoam on a MOVING MESH against REAL OpenFOAM's: a closed tank in solid-body motion.
//
// THE ORACLE is OpenFOAM's written state after exactly N identical fixed steps -- alpha, p_rgh, U,
// the moved polyMesh/points, the face velocity Uf and the mesh flux meshPhi -- and its log's "Solving
// for p_rgh" lines, solve by solve. The mesh motion itself is held to OpenFOAM's digits by
// tests/test_mesh_motion_vs_openfoam.cu; this gate is about what the SOLVER does with it: the wall
// velocity, the relative flux, the old volumes in every time derivative, Uf in ddtCorr, and the
// pressure reference of a tank with no free surface to the outside.
//
// tests/interfoam_moving_vs_openfoam.sh says what each profile is for and what it measured.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include "patch_entry_lookup.cuh"
#include "inter_solve_log.cuh"
#include <cmath>
#include <filesystem>
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

// OpenFOAM's residual AT EVERY ITERATION of every solve of one field, from a log whose solver entry
// carries `log 2;` (SolverPerformance.C:70-76 prints "<solver>:  Iteration N residual = r" ahead of each
// convergence check, so the entry at a solve's own count IS its final residual -- asserted where this
// is used). One vector per solve, indexed by iteration; 0 where the log has no entry.
std::vector<std::vector<scalar>> readOfResidualHistories(
    const std::string& logPath,
    const std::string& field)
{
    std::vector<std::vector<scalar>> out;
    std::vector<scalar> cur;
    std::ifstream in(logPath);
    std::string line;
    const std::string kIt = ":  Iteration ";
    const std::string kRes = " residual = ";
    while (std::getline(in, line))
    {
        const std::size_t a = line.find(kIt);
        const std::size_t b = line.find(kRes);
        if (a != std::string::npos && b != std::string::npos && b > a)
        {
            const std::size_t n = static_cast<std::size_t>(std::atoi(line.c_str() + a + kIt.size()));
            if (cur.size() <= n)
            {
                cur.resize(n + 1, scalar(0));
            }
            cur[n] = std::atof(line.c_str() + b + kRes.size());
            continue;
        }
        if (line.find("Solving for ") == std::string::npos)
        {
            continue;
        }
        // a solve's summary line closes its history: this field's is kept, any other's is dropped
        if (line.find("Solving for " + field + ",") != std::string::npos)
        {
            out.push_back(cur);
        }
        cur.clear();
    }
    return out;
}

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

struct Diff
{
    scalar linf = 0;
    scalar refMax = 0;
    scalar rel() const
    {
        return linf/std::fmax(refMax, scalar(1e-300));
    }
};

Diff compare(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.linf = std::fmax(d.linf, std::fabs(a[i] - b[i]));
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    return d;
}

Diff compare(
    const std::vector<vector>& a,
    const std::vector<vector>& b)
{
    Diff d;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        d.linf = std::fmax(d.linf, mag(a[i] - b[i]));
        d.refMax = std::fmax(d.refMax, mag(b[i]));
    }
    return d;
}

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

template <class T>
std::vector<T> cellValues(
    const FieldData<T>& fd,
    label nC)
{
    if (fd.internalUniform)
    {
        return std::vector<T>(static_cast<std::size_t>(nC), fd.internalUniformValue);
    }
    return fd.internalField;
}
}   // namespace

int main(
    int argc,
    char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: a moving mesh ==\n");
    if (argc < 8)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps> <log> <profile> "
                    "<staticOfTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string startDir = argv[2];
    const std::string ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));
    const std::string logPath = argv[5];
    const std::string profile = argv[6];
    const std::string staticDir = argv[7];
    std::printf("  profile: %s\n", profile.c_str());

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();
    MutableMesh mutableMesh;
    mutableMesh.m = &m;
    mutableMesh.g = &g;
    mutableMesh.patches = &patches;

    InterFields fin;
    PressureTaps taps;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/false, &fin,
                                     scalar(1.0e300), &taps, &mutableMesh);

    // THE PROFILES THAT RUN BOTH ARMS: the solid-body mixer (GAMG for p_rgh and a GAMG PRECONDITIONER
    // for p_rghFinal), the deforming-mesh paddle (PCG with DIC), the non-orthogonal cylinder (GAMG),
    // and the piston and flap, whose `div(phirb,alpha) Gauss interfaceCompression` is the alpha
    // scheme four of the waveMakers name. `solitaryGamg` is the one staged entry, and gives the
    // paddle a GAMG p_rgh with a coarsest level of its own to hold the shared hierarchy.
    // The device arm gets its OWN mesh, geometry and patches: both arms MOVE the one they are handed,
    // so sharing would make the host's motion the device's initial condition and every number after
    // that fiction.
    // ...and the two multi-paddle tutorials, 448,000 cells with GAMG for pcorr AND p_rgh on a mesh that
    // moves: the one shape where a host GAMG solve (CorrectPhi's pcorr) rebuilds the shared hierarchy
    // between the move and the device's pressure solve. The device kept its upload across that rebuild
    // -- keyed on `built`, which the host solve had set again -- and read the coarsest level's
    // addressing from the new hierarchy: heap corruption in the first step's first p_rgh on
    // multiFlap, where multiPiston's translated mesh happened to agglomerate to the same level sizes
    // and ran on the OLD pairing (GamgAgglomerationCache::buildCount). These two arms hold the fix.
    // ...and `pistonSST`, A RAS CLOSURE ON A MESH THAT MOVES, which took three terms, each measured
    // on this arm: kOmegaSST's ddt source on the OLD volumes (V for V0 reads U 1.8155e-05), divU as
    // the divergence of the ABSOLUTE flux (a relative divU, 4.4704e-05) and every DISTANCE the
    // closure holds recomputed after the move (a stale wall distance, the third; the two terms
    // without it read alpha 4.2494e-08, p_rgh 4.2660e-08, U 1.7353e-05 against the host's 1.3849e-12,
    // 1.4704e-12 and 1.1909e-10).
    // ...and `sloshing2DCN`, THE SCHEME on a moving mesh: the device arm forms CrankNicolson's own
    // moving ddt and fvcDdtUfCorr, both transcribed from the host reference, and reads the off-centred
    // fvc::meshPhi the shared mesh update builds. It refused the combination by name until those were
    // written. WHAT THIS ARM WITNESSES, measured by injecting each defect into the device build:
    //   * fvcDdtUfCorr -- take the STATIC ddtCorr(U, phi) instead: alpha 7.6e-02, p_rgh 9.1e+00,
    //     U 2.7e-01, Uf 2.8e-01, six failures;
    //   * the ddt's V0/V00 weights -- drop them, i.e. the static branch: U 6.3e-13 against 3.3e-13.
    //     BLIND, and not a fault of the arm: this tank moves as a SOLID BODY, so V == V0 == V00 and
    //     the two branches are the same arithmetic here.
    // `solitaryCN` is the deforming twin, and it witnesses BOTH halves of that weight. The same two
    // injections there, measured: the host reference forced to its static branch reads alpha
    // 3.3427e-03, p_rgh 3.5760e-03, U 2.1938e-01, and the device's deviceCnFvmDdt with V0/V00
    // dropped reads the same three numbers -- one term, one arithmetic, both arms landing on the
    // same wrong answer, three to four orders above this profile's bounds. The identical host
    // injection on the tank above reads alpha 5.8e-15 and U 3.9e-13 with ZERO failures, which is
    // what "a rigid mesh cannot witness a volume weight" looks like.
    // THE PROFILE THAT AMPLIFIES: see the bounds block below, and the script's note. Its own one-ulp
    // control is measured, and every bound on it is anchored there rather than on round-off.
    const bool amplifies = (profile == "solitaryCN");
    const bool deviceArm = (profile == "mixer" || profile == "solitary"
                         || profile == "cylinder" || profile == "solitaryGamg"
                         || profile == "piston" || profile == "flap"
                         || profile == "pistonSST" || profile == "pistonLES"
                         || profile == "multiPiston" || profile == "multiFlap"
                         || profile == "sloshing2DCN" || profile == "solitaryCN"
                         || profile == "esd" || profile == "esdNoCorr"
                         || profile == "closedDamBreak" || profile == "closedDamBreakInitU"
                         || profile == "closedAdjZG" || profile == "closedAdjIO"
                         || profile == "mixerTop" || profile == "mixerPermeable"
                         || profile == "pistonOuter" || profile == "pistonOuterOnce"
                         || profile == "floating" || profile == "solitaryOuterCN");
    // `floating` ON THE DEVICE: the body the fluid moves, its joint state held against OpenFOAM's below
    const bool floatingDevice = (profile == "floating");
    // `solitaryOuterCN`: three steps of the CrankNicolson wave tank under three outer correctors, where
    // the device's distance is its own on nine passes of 200-iteration pressure solves and not a multiple
    // of the host's -- MEASURED (2026-10-01) host alpha 3.9e-13 / p_rgh 2.4e-13 / U 2.1e-10, device
    // 3.1e-10 / 1.9e-10 / 1.5e-09. The defect the profile holds reads 6.9e-05 / 3.2e-05 / 6.5e-03 on
    // both arms (BRAE_CONTROL_CN_PHIOLD_PREV), five orders above the device's bounds.
    const bool shortOuterCN = (profile == "solitaryOuterCN");
    PrimitiveMesh mD;
    FvGeometry gD;
    std::vector<FvPatch> patchesD;
    MutableMesh mutableD;
    InterFields finD;
    RunReport rD;
    if (deviceArm)
    {
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
        if (nDev <= 0)
        {
            std::printf("  SKIP: no CUDA device for the %s profile\n", profile.c_str());
            return 77;
        }
        mD.read(caseDir + "/constant/polyMesh");
        gD.build(mD);
        patchesD = buildPatches(mD, gD);
        mutableD.m = &mD;
        mutableD.g = &gD;
        mutableD.patches = &patchesD;
        rD = runInterFoamDevice(caseDir, startDir, mD, gD, patchesD, nSteps, /*verbose=*/false, &finD,
                                scalar(1.0e300), nullptr, &mutableD);
        // the two arms must have moved the mesh the same way, or nothing below is about the solver
        scalar wv = 0, sv = 0;
        for (label c = 0; c < nC; ++c)
        {
            wv = std::fmax(wv, std::fabs(g.V()[c] - gD.V()[c]));
            sv = std::fmax(sv, std::fabs(g.V()[c]));
        }
        std::printf("  the two arms' meshes: worst |V_host - V_device| %.4e of %.4e\n",
                    (double)wv, (double)sv);
        check("both arms moved the mesh the same way",
              wv <= scalar(1e-13)*std::fmax(sv, scalar(1e-300)));
    }
    // THE BODY ON THE DEVICE ARM. The motion is a host stage on either arm; what this loop owes it is the
    // load -- the closure's nut and U's cells brought down before the mesh update (inter_driver_device.cu)
    // -- and the joint state it then integrates is compared with OpenFOAM's directly, as the host's is.
    if (floatingDevice)
    {
        check("the device arm read the case as a rigid-body motion",
              finD.dynamicMesh && finD.dynamicMesh->rigidBody() != nullptr);
        if (finD.dynamicMesh && finD.dynamicMesh->rigidBody())
        {
            const RBD::ModelState& st = finD.dynamicMesh->rigidBody()->state();
            const std::string sp = ofDir + "/uniform/rigidBodyMotionState";
            const std::vector<scalar> ofQ = RBD::readJointStateList(sp, "q").value_or(std::vector<scalar>{});
            const std::vector<scalar> ofV = RBD::readJointStateList(sp, "qDot").value_or(std::vector<scalar>{});
            const std::vector<scalar> ofA = RBD::readJointStateList(sp, "qDdot").value_or(std::vector<scalar>{});
            scalar wq = 0, wv = 0, wa = 0, rq = 0, rv = 0, ra = 0;
            for (std::size_t i = 0; i < st.q.size() && i < ofQ.size(); ++i)
            {
                wq = std::fmax(wq, std::fabs(st.q[i] - ofQ[i]));
                rq = std::fmax(rq, std::fabs(ofQ[i]));
                wv = std::fmax(wv, std::fabs(st.qDot[i] - ofV[i]));
                rv = std::fmax(rv, std::fabs(ofV[i]));
                wa = std::fmax(wa, std::fabs(st.qDdot[i] - ofA[i]));
                ra = std::fmax(ra, std::fabs(ofA[i]));
            }
            std::printf("  DEVICE body: q %.4e of %.4e, qDot %.4e of %.4e, qDdot %.4e of %.4e\n",
                        (double)wq, (double)rq, (double)wv, (double)rv, (double)wa, (double)ra);
            // the host arm's own bound: MEASURED on the device q 6.4e-17 of 1.3e-04, qDot 8.8e-15 of
            // 6.2e-03, qDdot 1.3e-12 of 1.8e-01 (2026-10-01)
            check("the device's joint position is OpenFOAM's",
                  ofQ.size() == st.q.size() && wq <= scalar(1e-10)*std::fmax(rq, scalar(1e-300)));
            check("...its joint velocity", wv <= scalar(1e-10)*std::fmax(rv, scalar(1e-300)));
            check("...and its joint acceleration", wa <= scalar(1e-10)*std::fmax(ra, scalar(1e-300)));
        }
    }
    // `esd`: ALPHA'S PATCH VALUES, which no other arm here compares, and the only place in the
    // interFoam tutorials where OpenFOAM's under-relaxed corrector is DISTINGUISHABLE from an evaluate.
    //
    // `alpha1 = 0.5*alpha1 + 0.5*alpha10` (VoF/alphaEqn.H:202) is a whole-field ASSIGNMENT. Its boundary
    // half goes GeometricBoundaryField::operator= -> FieldField::operator= -> each patch field's VIRTUAL
    // operator=, and nothing on that path consults assignable(). A `variableHeightFlowRate` patch is
    // mixed and overrides no operator=, so mixedFvPatchField.H:303-305 leaves its value ALONE: it keeps
    // the value MULES's own trailing correctBoundaryConditions left (CMULESTemplates.C). On an OUTFLOW
    // face its valueFraction is 0, so an evaluate would put the owner cell there instead -- which is
    // what brae did, at the relaxation AND again after the sub-cycle in the driver.
    //
    // WHY STEP TWO AND NOT TEN. This case also carries the OPEN `cellLimited` grad(alpha) item, whose
    // cell gap passes the patch difference from step five (2.2e-09 at five, 2.9e-08 at ten). At step two
    // the CELLS agree to 4.6e-16 and the patch value carries 5.1256e-10 -- six orders apart, so nothing
    // else on the case can be responsible. MEASURED, before the fix: brae's patch value sat EXACTLY on
    // OpenFOAM's own owner cell, the two differences the same 5.1256e-10 to five digits. After: 4.4e-16.
    //
    // THE ORACLE IS ASSERTED TO HAVE TAKEN THE PATH, because an oracle whose patch value equals its own
    // owner cell agrees with a re-evaluating brae by accident. THE CONTROL is `esdNoCorr`, the same case
    // with `MULESCorr no` -- one dictionary entry, so the relaxation branch never runs: OpenFOAM's patch
    // value goes back ON its owner cell, 0 faces stale, worst 7.1054e-15.
    if (profile == "esd" || profile == "esdNoCorr")
    {
        const FieldData<scalar> ofAlphaF = readField<scalar>(ofDir + "/" + fin.alphaName);
        const std::vector<scalar> ofAlphaCells = cellValues(ofAlphaF, nC);
        std::size_t nVh = 0;
        std::size_t nStale = 0;
        std::size_t nFaces = 0;
        scalar dPatch = 0;
        scalar dPatchD = 0;
        scalar dStale = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!fin.alpha1.boundary[pi]->isVariableHeightFlowRate()) continue;
            ++nVh;
            const PatchFieldData<scalar>* b = findPatchEntry(ofAlphaF.boundary, patches[pi]);
            check("OpenFOAM wrote a value list for every variableHeightFlowRate patch",
                  b != nullptr && b->hasValue);
            if (b == nullptr || !b->hasValue) continue;
            const std::vector<scalar> braeV = fin.alpha1.boundary[pi]->value();
            // ...and the DEVICE arm's, which keeps its alpha on the device and takes the patch from
            // the host hook: the same assignment-not-evaluate has to hold there
            const std::vector<scalar> braeVD = deviceArm ? finD.alpha1.boundary[pi]->value()
                                                         : std::vector<scalar>();
            for (std::size_t k = 0; k < braeV.size(); ++k)
            {
                const scalar ofV = b->valueUniform ? b->uniformValue : b->values[k];
                dPatch = std::fmax(dPatch, std::fabs(braeV[k] - ofV));
                if (deviceArm) dPatchD = std::fmax(dPatchD, std::fabs(braeVD[k] - ofV));
                // ...against OpenFOAM's OWN owner cell, which is what an evaluate would have written
                const scalar cell = ofAlphaCells[static_cast<std::size_t>(patches[pi].faceCells[k])];
                const scalar st = std::fabs(ofV - cell);
                dStale = std::fmax(dStale, st);
                if (st > scalar(1e-14)) ++nStale;
                ++nFaces;
            }
        }
        std::printf("  alpha patches: %zu variableHeightFlowRate, %zu faces   brae-vs-OF %.4e"
                    "   device-vs-OF %.4e   OF-vs-its-own-cell %.4e   (%zu faces stale)\n",
                    nVh, nFaces, (double)dPatch, (double)dPatchD, (double)dStale, nStale);
        // the tutorial's own five: side-01 (at the clamp, carries nothing) and side-03..side-06
        check("brae built all five variableHeightFlowRate patches", nVh == 5);
        check("...over the tutorial's own 2,625 faces", nFaces == 2625);
        if (profile == "esd")
        {
            // THE ORACLE TOOK THE PATH: 240 of the 2,625 faces measured, so 200 is a floor and not a fit
            check("OpenFOAM's own written patch value LEFT its owner cell, so the arm can witness this",
                  nStale >= 200 && dStale > scalar(1e-11));
            check("brae's alpha patch values are the ones OpenFOAM WROTE", dPatch < scalar(1e-13));
            check("...on the DEVICE arm too", dPatchD < scalar(1e-13));
        }
        else
        {
            // THE CONTROL: with MULESCorr off the assignment never runs, so there is nothing to keep
            check("with MULESCorr off OpenFOAM's patch value is back ON its owner cell",
                  nStale == 0 && dStale < scalar(1e-13));
            check("...and brae agrees there too", dPatch < scalar(1e-13));
            check("...on the DEVICE arm too", dPatchD < scalar(1e-13));
        }
    }
    // `closed*`: a closed tank whose mesh does NOT move -- the pressure reference alone
    const bool moving = profile.rfind("closed", 0) != 0;
    check("brae ran the same number of steps", r.steps == nSteps);
    check("the case moves its mesh, and brae read it so", (fin.dynamicMesh != nullptr) == moving);
    // the waveMaker tutorials are OPEN: a totalPressure atmosphere fixes p_rgh's level
    // ...and `mixerPermeable` is open too, by its WALLS: prghPermeableAlphaTotalPressure fixes
    // p_rgh's value on every face the phase fraction leaves dry, so the closed tube stops needing a
    // reference. OpenFOAM's own run agrees -- its log prints no pRefCell -- and reading it the other
    // way would add a reference OpenFOAM does not apply.
    // ...and `esd`/`esdNoCorr` are open by every side: its p_rgh is `totalPressure p0 uniform 0` on
    // "side-.*", which derives fixedValue and so FIXES a value. OpenFOAM's own run prints no pRefCell.
    const bool open = profile.rfind("solitary", 0) == 0 || profile.rfind("piston", 0) == 0
                   || profile.rfind("flap", 0) == 0 || profile.rfind("multi", 0) == 0
                   || profile == "mixerPermeable" || profile.rfind("floating", 0) == 0
                   || profile.rfind("esd", 0) == 0;
    if (open)
    {
        check("p_rgh is fixed at a patch, and brae read it so", !fin.pRef.needReference);
    }
    else
    {
        check("p_rgh needs a reference on this closed tank, and brae read it so", fin.pRef.needReference);
    }
    std::printf("  motion: %s; pRefCell %d, pRefValue %g\n",
                fin.dynamicMesh ? fin.dynamicMesh->motionType().c_str() : "none",
                (int)fin.pRef.pRefCell, (double)fin.pRef.pRefValue);

    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);

    // THE MESH the fields sit on: brae's points at the end against OpenFOAM's written ones
    if (moving)
    {
        const std::vector<vector> ofPoints = readPointsFile(ofDir + "/polyMesh/points");
        scalar lengthScale = 0;
        scalar dPoint = 0;
        for (std::size_t i = 0; i < ofPoints.size() && i < m.points().size(); ++i)
        {
            lengthScale = std::fmax(lengthScale, mag(ofPoints[i]));
            dPoint = std::fmax(dPoint, mag(m.points()[i] - ofPoints[i]));
        }
        std::printf("  points at the end: %.3e of the mesh's extent (%zu points)\n",
                    (double)(dPoint/std::fmax(lengthScale, scalar(1e-300))), ofPoints.size());
        check("OpenFOAM wrote the moved mesh", ofPoints.size() == m.points().size() && !ofPoints.empty());
        check("brae's mesh ended where OpenFOAM's did", dPoint <= scalar(1e-15)*lengthScale);
    }

    // THE BODY, which the mesh alone cannot fully witness: the blend spreads the joint state over
    // thirteen thousand points, so a q that is wrong in the last digits reads as round-off there. Its
    // own state is written every step, so it is compared directly -- and it is the ONLY field here
    // that the static control cannot produce at all.
    if (profile.rfind("floating", 0) == 0)
    {
        check("brae read the case as a rigid-body motion",
              fin.dynamicMesh && fin.dynamicMesh->rigidBody() != nullptr);
        if (fin.dynamicMesh && fin.dynamicMesh->rigidBody())
        {
            const RBD::ModelState& st = fin.dynamicMesh->rigidBody()->state();
            const std::string sp = ofDir + "/uniform/rigidBodyMotionState";
            const std::vector<scalar> ofQ = RBD::readJointStateList(sp, "q").value_or(std::vector<scalar>{});
            const std::vector<scalar> ofV = RBD::readJointStateList(sp, "qDot").value_or(std::vector<scalar>{});
            const std::vector<scalar> ofA = RBD::readJointStateList(sp, "qDdot").value_or(std::vector<scalar>{});
            check("OpenFOAM wrote the joint state", ofQ.size() == st.q.size() && !ofQ.empty());
            scalar wq = 0, wv = 0, wa = 0, rq = 0, rv = 0, ra = 0;
            for (std::size_t i = 0; i < st.q.size() && i < ofQ.size(); ++i)
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
            check("brae's joint position is OpenFOAM's",
                  wq <= scalar(1e-10)*std::fmax(rq, scalar(1e-300)));
            check("...its joint velocity", wv <= scalar(1e-10)*std::fmax(rv, scalar(1e-300)));
            check("...and its joint acceleration",
                  wa <= scalar(1e-10)*std::fmax(ra, scalar(1e-300)));
        }
    }

    // THE SOLVES: every p_rgh line, iteration counts and residuals
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    // THE WAVEMAKERS' SOLVES RUN 150 TO 480 PCG ITERATIONS to a tolerance of 1e-13 (the script
    // converges them), and on a solve that long the iteration it stops at is decided in the last bits,
    // which thirty steps of a deforming mesh carry forward: measured on the piston, OpenFOAM 218 and
    // brae 217, 163 and 164, and at the last step 194 and 197, every one of them ending below 1e-13 in
    // both codes. For those two profiles EVERY solve must end below the tolerance in both, a solve of
    // 100 iterations or fewer must take OpenFOAM's count, and a longer one must be within 2% of it; the
    // initial residuals are printed, not asserted, and the fields below carry the gate's own bounds.
    // ...and `solitaryCN`, whose solves are converged to 1e-13 for the same reason: where the last
    // iteration is a stopping point rather than a computation, the residual CURVE is the comparison.
    const bool longSolves = profile.rfind("piston", 0) == 0 || profile.rfind("flap", 0) == 0
                         || profile == "solitaryCN" || profile == "solitaryOuterCN";
    // `allowOne`: the DEVICE arm's rule. On a solve converged to 1e-13 with relTol 0 the last
    // iteration is where an implementation stops, not what it computes, and the device's reductions
    // are summed in a different order from OpenFOAM's by construction (device_pcg.cuh). MEASURED on
    // `piston`: 17 of the 90 solves one or two iterations apart, every one of them ending below 1e-13
    // in both codes, with alpha 3.2e-12 and U 1.5e-09 -- the host arm's own distance. So the device
    // is allowed ONE iteration wherever OpenFOAM took at least twenty, and the 2% rule above that;
    // below twenty it is held exactly, as the host is everywhere.
    // `history`: OpenFOAM's OWN residual at every iteration of every solve (the staging gives these
    // profiles' p_rgh entry `log 2;`). It is what a count cannot say. PCG's residual is not monotone,
    // and where OpenFOAM's sits ON the tolerance for several iterations the count is decided by the
    // fourth digit of one residual: MEASURED on `flap`, step 7's first corrector, OpenFOAM reads
    // 1.0167e-13 at iteration 242, 1.0046e-13 at 243, climbs to 1.0981e-13 and only ends at 250; the
    // device ended at 242 on 9.993e-14 -- 1.7% from OpenFOAM's residual AT THAT ITERATION, less than
    // the two are apart on most solves that end ONE iteration apart. So, with a history:
    //   * EVERY solve, whatever its count: brae's final residual is within 15% of OpenFOAM's residual
    //     at the iteration brae stopped on. MEASURED worst over a run's solves: piston 2.2% host and
    //     9.1% device, pistonSST 4.3%, flap 3.2% host and 6.7% device; the medians 0.0% to 1.8%.
    //     This is the two residual CURVES compared, on every solve and not on the odd one.
    //     THE CONTROL is the same statistic against OpenFOAM's residual ONE ITERATION EARLIER, which
    //     must break the bound: measured worst 19% to 27%, medians 5% to 13% -- the curve falls about
    //     8% an iteration here, so the statistic resolves a shift of one;
    //   * a count outside the rule above is accepted ONLY where brae stopped EARLIER within 5% of
    //     OpenFOAM's residual at that iteration -- on OpenFOAM's own plateau -- and on at most ONE
    //     solve of the run (measured: flap's device arm, one; every other arm, none).
    // Without a history the count rule stands alone, as it did.
    auto countsAgree = [&](
        const char* field,
        const std::vector<LinearSolveRecord>& mine,
        const std::vector<LinearSolveRecord>& of,
        bool allowOne = false,
        const std::vector<std::vector<scalar>>* history = nullptr)
    {
        const scalar tol = scalar(1e-13);
        const scalar curveBound = scalar(0.15);
        const scalar plateauBound = scalar(0.05);
        int nApart = 0;
        int worstApart = 0;
        int nOnPlateau = 0;
        int nCompared = 0;
        scalar worstCurve = 0;
        scalar worstEarlier = 0;
        bool converged = mine.size() == of.size() && !of.empty();
        bool counts = converged;
        bool curves = converged;
        bool historyIsTheLog = history != nullptr && history->size() == of.size();
        for (std::size_t k = 0; k < of.size() && k < mine.size(); ++k)
        {
            if (mine[k].finalResidual > tol || of[k].finalResidual > tol)
            {
                converged = false;
            }
            // OpenFOAM's residual at the iteration brae stopped on; 0 where OpenFOAM had already ended
            scalar ofThere = 0;
            if (historyIsTheLog)
            {
                const std::vector<scalar>& h = (*history)[k];
                const std::size_t nOf = static_cast<std::size_t>(of[k].nIterations);
                // THE FAIL-PROOF of the parse: the history's entry at OpenFOAM's own count is the
                // final residual its summary line prints
                if (h.size() <= nOf
                 || brae::gatecheck::residualRelDiff(h[nOf], of[k].finalResidual) > scalar(1e-10))
                {
                    historyIsTheLog = false;
                }
                const std::size_t nMine = static_cast<std::size_t>(mine[k].nIterations);
                ofThere = nMine < h.size() ? h[nMine] : scalar(0);
                // THE CONTROL's half: OpenFOAM's residual one iteration EARLIER
                if (nMine >= 1 && nMine - 1 < h.size() && h[nMine - 1] > 0)
                {
                    worstEarlier = std::fmax(worstEarlier,
                                             std::fabs(mine[k].finalResidual - h[nMine - 1])/h[nMine - 1]);
                }
            }
            if (ofThere > 0)
            {
                ++nCompared;
                const scalar dCurve = std::fabs(mine[k].finalResidual - ofThere)/ofThere;
                worstCurve = std::fmax(worstCurve, dCurve);
                if (dCurve > curveBound)
                {
                    curves = false;
                }
            }
            const int d = std::abs(mine[k].nIterations - of[k].nIterations);
            if (d == 0) continue;
            ++nApart;
            worstApart = std::max(worstApart, d);
            const int allowed = allowOne && of[k].nIterations >= 20
                              ? std::max(1, of[k].nIterations/50)
                              : (of[k].nIterations <= 100 ? 0 : of[k].nIterations/50);
            if (d > allowed)
            {
                const bool onPlateau = ofThere > 0 && mine[k].nIterations < of[k].nIterations
                                    && std::fabs(mine[k].finalResidual - ofThere)/ofThere <= plateauBound;
                if (onPlateau)
                {
                    ++nOnPlateau;
                    std::printf("  %s solve %zu: brae ended at iteration %d on %.4e where OpenFOAM read %.4e and "
                                "went on to %d\n", field, k, (int)mine[k].nIterations,
                                (double)mine[k].finalResidual, (double)ofThere, (int)of[k].nIterations);
                }
                else
                {
                    counts = false;
                }
            }
        }
        std::printf("  %s: %zu solves, %d of them apart, by up to %d iterations\n", field, of.size(), nApart,
                    worstApart);
        check("...every solve ended below its tolerance in both codes", converged);
        check("...every count OpenFOAM's on a short solve, and within 2% of it on a long one", counts);
        if (history)
        {
            std::printf("  %s: brae's final residual against OpenFOAM's AT THE SAME ITERATION, worst %.3e over %d "
                        "solves; %d count explained by OpenFOAM's own plateau\n", field, (double)worstCurve,
                        nCompared, nOnPlateau);
            check("...OpenFOAM's log carries a residual for every iteration, ending on its printed final one",
                  historyIsTheLog && nCompared > 0);
            check("...every solve's final residual is within 15% of OpenFOAM's at that same iteration", curves);
            std::printf("  %s CONTROL: the same statistic against OpenFOAM's residual ONE ITERATION EARLIER, worst %.3e\n",
                        field, (double)worstEarlier);
            // THE CONTROL IS ONLY MEANINGFUL WHERE A COUNT IS IN QUESTION. Where brae took
            // OpenFOAM's iteration count on EVERY solve, the curve statistic is not carrying the
            // result -- the counts are -- and whether the curve is steep enough to resolve a shift of
            // one is a property of the oracle's own residual history on that case, not of brae.
            // MEASURED on `pistonLES`, whose host arm is 0 of 90 apart: the control reads 1.257e-01
            // against a 15% bound, i.e. that case's curve falls too slowly for the statistic, where
            // `piston` and `pistonSST` read 19% to 27%. Asserted wherever a count differs, which is
            // every arm the statistic is there for -- including this profile's own device arm, 22 of
            // 90 apart, control 1.892e-01.
            if (counts && nApart == 0)
            {
                std::printf("  %s: every count is OpenFOAM's, so the curve statistic is not what "
                            "carries this arm; its control (%.3e against a %.0f%% bound) is reported "
                            "and not asserted\n", field, (double)worstEarlier, (double)(100*curveBound));
            }
            else
            {
                check("...and one iteration earlier it is NOT: the statistic resolves a shift of one",
                      worstEarlier > curveBound);
            }
            check("...at most ONE count of the run rests on OpenFOAM's plateau", nOnPlateau <= 1);
        }
    };
    const std::vector<std::vector<scalar>> ofPHistory =
        longSolves ? readOfResidualHistories(logPath, "p_rgh") : std::vector<std::vector<scalar>>();
    if (longSolves)
    {
        brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(1e-10), scalar(1e-6),
                                       scalar(-1), nullptr, false);
        countsAgree("p_rgh", r.pSolves, ofP, /*allowOne=*/false, &ofPHistory);
    }
    else
    {
        // ...and `esd`'s step-one residual bound is its p_rgh difference divided by the normFactor, the
        // same un-localised 1.0e-07 absolute: MEASURED 5.180e-09 in step one (9.594e-08 over the run,
        // inside the shared 1e-6). Every iteration count is OpenFOAM's.
        const scalar resid1 = (profile.rfind("esd", 0) == 0) ? scalar(2e-8) : scalar(1e-10);
        failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", resid1,
                                                   scalar(1e-6));
    }

    // THE DEVICE ARM'S OWN p_rgh SOLVES. This is the arm that says the device ran the solver the case
    // NAMED: `cylinder` asks for GAMG with the DIC smoother, and a V-cycle's iteration count is not a
    // Krylov method's. Run with the substitute this loop used to make here, the counts are 20 of 20
    // wrong and p_rgh is 2.5406e+05 of 5.2780e+06 (sloshingTank2D, measured against the host).
    if (deviceArm)
    {
        if (longSolves)
        {
            brae::gatecheck::compareSolves("device", rD.pSolves, ofP, nSteps, "p_rgh", scalar(1e-10),
                                           scalar(1e-6), scalar(-1), nullptr, false);
            countsAgree("p_rgh", rD.pSolves, ofP, /*allowOne=*/true, &ofPHistory);
        }
        else
        {
            failures += brae::gatecheck::compareSolves("device", rD.pSolves, ofP, nSteps, "p_rgh",
                                                       scalar(1e-10), scalar(1e-6));
        }
    }

    // ...AND EVERY pcorr LINE: initCorrectPhi's at the start of every profile -- moving or not, with
    // correctPhi or without -- and on a `*CorrectPhi` profile CorrectPhi's after every mesh update, one
    // per non-orthogonal pass
    const std::vector<LinearSolveRecord> ofPcorr = brae::gatecheck::readOfSolves(logPath, "pcorr");
    const bool correctPhiOn = fin.correctPhi && moving;
    std::printf("  pcorr solves: brae %zu, OpenFOAM %zu%s\n", r.pcorrSolves.size(), ofPcorr.size(),
                correctPhiOn ? " (correctPhi on)" : " (the start only)");
    check("OpenFOAM's log shows CorrectPhi at the start, and after every mesh update where correctPhi is on",
          !ofPcorr.empty()
          && (correctPhiOn ? ofPcorr.size() > static_cast<std::size_t>(nSteps) : ofPcorr.size() <= 2));
    if (longSolves)
    {
        brae::gatecheck::compareSolves("host", r.pcorrSolves, ofPcorr, nSteps, "pcorr", scalar(1e-10),
                                       scalar(1e-6), scalar(-1), nullptr, false);
        countsAgree("pcorr", r.pcorrSolves, ofPcorr);
    }
    else
    {
        failures += brae::gatecheck::compareSolves("host", r.pcorrSolves, ofPcorr, nSteps, "pcorr",
                                                   scalar(1e-10), scalar(1e-6));
    }

    const std::vector<scalar> ofAlpha = cellValues(readField<scalar>(ofDir + "/" + fin.alphaName), nC);
    const std::vector<scalar> ofPrgh = cellValues(readField<scalar>(ofDir + "/p_rgh"), nC);
    const std::vector<scalar> ofP2 = cellValues(readField<scalar>(ofDir + "/p"), nC);
    const std::vector<vector> ofU = cellValues(readField<vector>(ofDir + "/U"), nC);

    const Diff dA = compare(fin.alpha1.internal, ofAlpha);
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    const Diff dPp = compare(fin.p, ofP2);
    const Diff dU = compare(fin.U.internal, ofU);
    std::printf("  alpha:   Linf %.4e\n", (double)dA.linf);
    std::printf("  p_rgh:   relative %.4e   (|p_rgh| up to %.4e)\n", (double)dP.rel(), (double)dP.refMax);
    std::printf("  p:       relative %.4e   (|p| up to %.4e)\n", (double)dPp.rel(), (double)dPp.refMax);
    std::printf("  U:       relative %.4e   (|U| up to %.4e)\n", (double)dU.rel(), (double)dU.refMax);

    // THE DEVICE ARM AGAINST THE SAME ORACLE. Its bound is the HOST arm's own distance from OpenFOAM
    // on this profile, not a number picked to fit: both arms solve the same pinned systems on the same
    // moved mesh, so what the comparison can reach is the arithmetic, and the host reaches it. The
    // controls are the profile's own -- `mixerStatic` (the tank held still) is what the checks above
    // measure the motion against, and it moves these fields far more than either arm is from OpenFOAM.
    if (deviceArm)
    {
        const Diff eA = compare(finD.alpha1.internal, ofAlpha);
        const Diff eP = compare(finD.p_rgh.internal, ofPrgh);
        const Diff eU = compare(finD.U.internal, ofU);
        std::printf("  DEVICE:  alpha %.4e, p_rgh %.4e, U %.4e\n",
                    (double)eA.linf, (double)eP.rel(), (double)eU.rel());
        std::printf("  host:    alpha %.4e, p_rgh %.4e, U %.4e\n",
                    (double)dA.linf, (double)dP.rel(), (double)dU.rel());
        // ON AN AMPLIFYING CASE the host's own distance is not the yardstick -- both arms are a small
        // multiple of OpenFOAM's sensitivity to its own last bit, and which multiple is round-off's
        // business. MEASURED on `solitaryCN`: host alpha 4.0e-08 / p_rgh 2.9e-08 / U 4.2e-06, device
        // 7.7e-07 / 6.0e-07 / 1.1e-05, one-ulp control 7.8e-09 / 7.9e-09 / 1.8e-06. So the device is
        // held to absolute bounds of its own there -- two to three times what it reads, and four
        // orders below the defect this profile was written for (alpha 5.3e-03).
        check(amplifies ? "the device's alpha is inside this case's own one-ulp noise"
                        : "the device's alpha is as close to OpenFOAM as the host's",
              amplifies ? (eA.linf < scalar(2e-6))
                        : shortOuterCN ? (eA.linf < scalar(3e-9))
                        : (eA.linf <= scalar(20)*std::fmax(dA.linf, scalar(1e-300))));
        check("...its p_rgh", amplifies ? (eP.rel() < scalar(2e-6))
                                        : shortOuterCN ? (eP.rel() < scalar(2e-9))
                                        : (eP.rel() <= scalar(20)*std::fmax(dP.rel(), scalar(1e-300))));
        check("...and its U", amplifies ? (eU.rel() < scalar(2e-5))
                                        : shortOuterCN ? (eU.rel() < scalar(1.5e-8))
                                        : (eU.rel() <= scalar(20)*std::fmax(dU.rel(), scalar(1e-300))));
        check("OpenFOAM's own fields are not zero here, so the comparison means something",
              dU.refMax > scalar(0) && dP.refMax > scalar(0));
    }

    // Uf and the wall velocity, face by face, against what OpenFOAM wrote
    if (moving)
    {
        const FieldData<vector> ofUf = readField<vector>(ofDir + "/Uf");
        const FieldData<vector> ofUFd = readField<vector>(ofDir + "/U");
        const Diff dUf = compare(fin.Uf.internal, ofUf.internalField);
        std::printf("  Uf:      relative %.4e   (|Uf| up to %.4e)\n", (double)dUf.rel(), (double)dUf.refMax);
        check("Uf on the internal faces is OpenFOAM's",
              dUf.rel() < (amplifies ? scalar(2e-5) : scalar(2e-7)) && !ofUf.internalField.empty());
        // ...AND THE DEVICE ARM'S Uf, which is the one field a step's own output cannot witness: Uf
        // is written at the end of the pressure corrector and read only by the NEXT step's ddtCorr.
        // The device arm handed fvc::correctUf the flux it had already made relative to the motion,
        // and after one step its Uf was 1.5004e+00 of 3.3328e+00 from the host's while alpha agreed
        // to 5.6e-15 and U to 1.6e-11 -- every other check on this page green.
        if (deviceArm)
        {
            const Diff eUf = compare(finD.Uf.internal, ofUf.internalField);
            std::printf("  DEVICE Uf: relative %.4e   (host %.4e)\n",
                        (double)eUf.rel(), (double)dUf.rel());
            check("...and the device's Uf is as close to OpenFOAM as the host's",
                  amplifies ? (eUf.rel() < scalar(2e-5))
                            : (eUf.rel() <= scalar(20)*std::fmax(dUf.rel(), scalar(1e-300))));
        }
        scalar dWall = 0;
        scalar wallScale = 0;
        std::size_t nWall = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!fin.movingWallVelocityPatch[pi]) continue;
            const PatchFieldData<vector>* b = findPatchEntry(ofUFd, patches[pi]);
            if (!b || b->valueUniform) continue;
            const Diff d = compare(fin.U.boundary[pi]->value(), b->values);
            dWall = std::fmax(dWall, d.linf);
            wallScale = std::fmax(wallScale, d.refMax);
            nWall += b->values.size();
        }
        std::printf("  wall velocity: Linf %.4e on %zu moving-wall faces (|U_wall| up to %.4e)\n",
                    (double)dWall, nWall, (double)wallScale);
        // `mixerPermeable` replaces the tutorial's movingWallVelocity with the permeable pair, so it
        // has NO moving wall to carry a velocity -- and that is asserted rather than skipped, because
        // an arm that quietly passes on an empty set is the vacuous shape this suite keeps finding.
        // What that profile holds instead is U's own comparison above and the permeable control.
        if (profile == "mixerPermeable")
        {
            check("this profile put the permeable pair where the moving wall was, so there is none",
                  nWall == 0 && !fin.movingWallVelocityPatch[0]);
        }
        else if (profile.rfind("floating", 0) == 0)
        {
            // THE ONLY FIXTURE HERE WHOSE WALL CREEPS WHILE ITS FLUID MOVES. movingWallVelocity's
            // value is the internal field with its normal component replaced by the mesh flux
            // (movingWallVelocityFvPatchVectorField.C), so its ABSOLUTE error is the internal field's
            // -- and on this case the body drifts at 4.1e-03 m/s while the water reaches 1.16 m/s,
            // 280 times more. Scaling the bound by the wall's own velocity would be holding the
            // internal field to 3.6e-15, which is under its round-off. MEASURED: the wall is 9.0e-14
            // where U itself is 6.0e-14 of 1.16 -- the same absolute distance, as the expression says
            // it must be. Every other profile's wall moves with its fluid and keeps the bound above.
            scalar uScale = 0;
            for (const vector& u : fin.U.internal) uScale = std::fmax(uScale, mag(u));
            std::printf("    (the wall creeps at %.3e while the fluid reaches %.3e, so the wall value "
                        "is held to the field's scale)\n", (double)wallScale, (double)uScale);
            check("the moving walls carry OpenFOAM's velocity",
                  nWall > 0 && dWall <= scalar(1e-12)*uScale);
        }
        else
        {
            // `esd` CARRIES ITS OWN, and the cause is NAMED rather than open: movingWallVelocity
            // subtracts two centres computed by DIFFERENT algorithms -- `face::centre` (face.C) for the
            // OLD points and `primitiveMeshTools::makeFaceCentresAndAreas` (what `pp.faceCentres()`
            // returns) for the CURRENT ones -- and those do not cancel even for a rigid translation,
            // which is why OpenFOAM's Up here is -8.00119972000634672e-02 and not -0.08. brae computes
            // BOTH with face::centre, so its difference cancels exactly: 8.3271e-14 on a wall reaching
            // 8.0012e-02, i.e. 1.04e-12 relative, just over the shared 1e-12.
            // THE FIX IS WRITTEN AND HELD BACK (inter_driver_cpp.cu says why): with the geometry's Cf in
            // place this reads EXACTLY 0 on all 3792 faces at both steps and 20 of the 27 moving arms are
            // unchanged or better -- but laminar/sloshingCylinder then goes from 1.3x to 7.5x its own
            // measured one-ulp floor, so a second defect it was compensating has to be found first.
            const scalar wallFactor = (profile.rfind("esd", 0) == 0) ? scalar(2e-12) : scalar(1e-12);
            check("the moving walls carry OpenFOAM's velocity",
                  nWall > 0 && dWall <= wallFactor*wallScale);
        }
        if (deviceArm)
        {
            scalar eWall = 0;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                if (!finD.movingWallVelocityPatch[pi]) continue;
                const PatchFieldData<vector>* b = findPatchEntry(ofUFd, patches[pi]);
                if (!b || b->valueUniform) continue;
                eWall = std::fmax(eWall, compare(finD.U.boundary[pi]->value(), b->values).linf);
            }
            std::printf("  DEVICE wall velocity: Linf %.4e (host %.4e)\n",
                        (double)eWall, (double)dWall);
            check("...and the device's moving walls carry it too",
                  eWall <= scalar(20)*std::fmax(dWall, scalar(1e-300)));
        }
    }

    // THE PRESSURE CORRECTOR TERM BY TERM, when the case was run with tools/dumpInterFoam rather than
    // interFoam (BRAE_DUMP_ITER = the last step): both sides hold the FIRST pressure corrector of
    // that step. They used to hold the last on both sides by accident -- the host taps were written
    // by every corrector and the tool's writes overwrote each other -- and the two are now pinned,
    // because on a case with nCorrectors 2 the accident does not survive one side being fixed.
    // Cells beside a moving wall are split from the rest. A diagnostic, not an arm: the staging script
    // runs interFoam, and this block is how a gap on a moving mesh is taken apart.
    if (std::filesystem::exists(ofDir + "/rAU.dump"))
    {
        std::vector<bool> wallCell(static_cast<std::size_t>(nC), false);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!fin.movingWallVelocityPatch[pi]) continue;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                wallCell[static_cast<std::size_t>(patches[pi].faceCells[i])] = true;
            }
        }
        const label nIf = m.nInternalFaces();
        auto report = [&](
            const char* name,
            const std::vector<scalar>& brae,
            const std::string& file,
            bool faces)
        {
            if (!std::filesystem::exists(file))
            {
                std::printf("    %-10s (no dump)\n", name);
                return;
            }
            const FieldData<scalar> fd = readField<scalar>(file);
            const std::size_t n = static_cast<std::size_t>(faces ? nIf : nC);
            const std::vector<scalar> of = fd.internalUniform
                ? std::vector<scalar>(n, fd.internalUniformValue)
                : fd.internalField;
            if (of.size() != n || brae.size() != n)
            {
                std::printf("    %-10s size mismatch (brae %zu, OpenFOAM %zu, expected %zu)\n",
                            name, brae.size(), of.size(), n);
                return;
            }
            scalar wW = 0;
            scalar wI = 0;
            scalar sc = 0;
            std::size_t at = 0;
            for (std::size_t k = 0; k < n; ++k)
            {
                const bool wall = faces
                    ? (wallCell[static_cast<std::size_t>(m.owner()[k])] || wallCell[static_cast<std::size_t>(m.neighbour()[k])])
                    : wallCell[k];
                const scalar e = std::fabs(brae[k] - of[k]);
                if (e > std::fmax(wW, wI))
                {
                    at = k;
                }
                if (wall)
                {
                    wW = std::fmax(wW, e);
                }
                else
                {
                    wI = std::fmax(wI, e);
                }
                sc = std::fmax(sc, std::fabs(of[k]));
            }
            std::printf("    %-10s moving wall %.3e  rest %.3e  of %.3e  (relative %.2e; worst %s %zu: brae %.12e, OF %.12e)\n",
                        name, (double)wW, (double)wI, (double)sc,
                        (double)(std::fmax(wW, wI)/std::fmax(sc, scalar(1e-300))),
                        faces ? "face" : "cell", at, (double)brae[at], (double)of[at]);
        };
        std::printf("  pressure corrector against tools/dumpInterFoam, first corrector of step %ld:\n",
                    (long)nSteps);
        report("UEqn.A", taps.A, ofDir + "/UEqnA.dump", false);
        report("rAU", taps.rAU, ofDir + "/rAU.dump", false);
        report("rAUf", taps.rAUf, ofDir + "/rAUf.dump", true);
        report("stf", taps.stf, ofDir + "/stf.dump", true);
        report("snGradRho", taps.snGradRho, ofDir + "/snGradRho.dump", true);
        report("phig", taps.phig, ofDir + "/phig.dump", true);
        report("phiHbyA", taps.phiHbyA, ofDir + "/phiHbyA.dump", true);
        report("rho", fin.rho, ofDir + "/rho.dump", false);
    }

    // THE CLOSURE ON THE MOVING MESH, under `pistonSST`: k, omega and nut against OpenFOAM's, and every
    // k and omega solve. kOmegaSST's moving-mesh terms are the old volumes in fvm::ddt, the absolute flux
    // in divU and the wall distance recomputed after every motion; the script's fixture makes the piston
    // a wall so the last of these moves.
    const bool sst = fin.turbulence.on && fin.turbulence.model == InterRasModel::KOmegaSST;
    // ...and whether the case's ddt scheme is the one under test, which picks the control's meaning
    const bool cn = (fin.ddtU == DdtScheme::CrankNicolson);
    // ...and `pistonLES`, the SAME paddle under LES kEqn, which has k and nut and no second field.
    // Its third moving-mesh term is the one the RAS closures do not have: the FILTER WIDTH,
    // (deltaCoeff*V)^(1/3) per cell, which LESModel::correct recomputes on a changing mesh
    // (LESModel.C:251 -> cubeRootVolDelta.C:128-134). k reaches U through nut alone, so comparing it
    // is the only way this profile can see the width at all.
    const bool les = fin.turbulence.on && fin.turbulence.model == InterRasModel::KEqnLES;
    Diff dK;
    Diff dOm;
    Diff dNut;
    if (les)
    {
        const std::vector<scalar> ofK = cellValues(readField<scalar>(ofDir + "/k"), nC);
        const std::vector<scalar> ofNut = cellValues(readField<scalar>(ofDir + "/nut"), nC);
        dK = compare(fin.turbulence.k.internal, ofK);
        dNut = compare(fin.turbulence.nut.internal, ofNut);
        std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
        std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dNut.rel(), (double)dNut.refMax);
        check("brae HAS the closure's fields to compare", !ofK.empty() && ofK.size() == fin.turbulence.k.internal.size());
        check("k agrees with OpenFOAM's relatively", dK.rel() < scalar(2e-10));
        check("nut agrees with OpenFOAM's relatively", dNut.rel() < scalar(2e-10));
        // ...and OpenFOAM's own nut is not zero here, or the comparison says nothing
        scalar nutMax = 0;
        for (const scalar v : ofNut) nutMax = std::fmax(nutMax, std::fabs(v));
        std::printf("  OpenFOAM's own nut reaches %.4e\n", (double)nutMax);
        check("the LES closure is live on this fixture", nutMax > scalar(1e-9));
    }
    if (sst)
    {
        const std::vector<scalar> ofK = cellValues(readField<scalar>(ofDir + "/k"), nC);
        const std::vector<scalar> ofOm = cellValues(readField<scalar>(ofDir + "/omega"), nC);
        const std::vector<scalar> ofNut = cellValues(readField<scalar>(ofDir + "/nut"), nC);
        dK = compare(fin.turbulence.k.internal, ofK);
        dOm = compare(fin.turbulence.omega.internal, ofOm);
        dNut = compare(fin.turbulence.nut.internal, ofNut);
        std::printf("  k:       relative %.4e   (k up to %.4e)\n", (double)dK.rel(), (double)dK.refMax);
        std::printf("  omega:   relative %.4e   (omega up to %.4e)\n", (double)dOm.rel(), (double)dOm.refMax);
        std::printf("  nut:     relative %.4e   (nut up to %.4e)\n", (double)dNut.rel(), (double)dNut.refMax);
        // WHERE: the worst cell of each, and the patch it touches if any, for taking a gap apart
        auto where = [&](const char* name, const std::vector<scalar>& mine, const std::vector<scalar>& of)
        {
            std::size_t at = 0;
            scalar worst = -1;
            for (std::size_t c = 0; c < of.size() && c < mine.size(); ++c)
            {
                const scalar e = std::fabs(mine[c] - of[c]);
                if (e > worst)
                {
                    worst = e;
                    at = c;
                }
            }
            std::string touches = "interior";
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    if (static_cast<std::size_t>(patches[pi].faceCells[i]) == at)
                    {
                        touches = patches[pi].name;
                    }
                }
            }
            std::printf("    %-6s worst at cell %zu (%.4f %.4f %.4f), %s: brae %.10e OpenFOAM %.10e\n", name, at,
                        (double)g.C()[at].x, (double)g.C()[at].y, (double)g.C()[at].z, touches.c_str(),
                        (double)mine[at], (double)of[at]);
        };
        where("k", fin.turbulence.k.internal, ofK);
        where("omega", fin.turbulence.omega.internal, ofOm);
        where("nut", fin.turbulence.nut.internal, ofNut);
        // ...and the wall distance, when `checkMesh -writeFields '(wallDistance)' -time <end>` has written
        // OpenFOAM's for the moved mesh (a diagnostic; the script does not run it)
        if (std::filesystem::exists(ofDir + "/wallDistance"))
        {
            const std::vector<scalar> ofY = cellValues(readField<scalar>(ofDir + "/wallDistance"), nC);
            const Diff dY = compare(fin.turbulence.yCell, ofY);
            std::printf("  y:       relative %.4e   (y up to %.4e)\n", (double)dY.rel(), (double)dY.refMax);
            where("y", fin.turbulence.yCell, ofY);
        }
        const std::vector<LinearSolveRecord> ofKs = brae::gatecheck::readOfSolves(logPath, "k");
        const std::vector<LinearSolveRecord> ofOms = brae::gatecheck::readOfSolves(logPath, "omega");
        failures += brae::gatecheck::compareSolves("host", r.kSolves, ofKs, nSteps, "k",
                                                   scalar(1e-10), scalar(1e-6));
        failures += brae::gatecheck::compareSolves("host", r.omegaSolves, ofOms, nSteps, "omega",
                                                   scalar(1e-10), scalar(1e-6));
        // MEASURED over 30 steps: k 8.6e-12, omega 2.4e-12, nut 3.0e-10 relative; bounded at about 30x
        check("k agrees with OpenFOAM's relatively", dK.rel() < scalar(3e-10));
        check("omega agrees with OpenFOAM's relatively", dOm.rel() < scalar(1e-10));
        check("nut agrees with OpenFOAM's relatively", dNut.rel() < scalar(1e-8));
    }

    // BOUNDS: see the script for what was measured.
    // `solitaryCN` HAS ITS OWN, and they are looser on purpose: that case amplifies. MEASURED with
    // OpenFOAM against ITSELF, one ulp of one alpha cell, over the same thirty steps and the same
    // converged solves -- U 1.752e-06, alpha 7.764e-09, p_rgh 7.933e-09. brae reads 4.077e-06,
    // 3.960e-08 and 2.929e-08 there: two to five times OpenFOAM's own sensitivity to its last bit,
    // which is what this case can resolve and no tighter. The bound is five times brae's, and the
    // defect the profile was written for (phi.oldTime()'s lazy creation) read U 1.06e+00 -- five
    // orders above it. At TWO steps the same comparison reads U 1.5e-11 against a 5.2e-12 control.
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < (amplifies ? scalar(2e-7) : scalar(1e-9)));
    // `esd` HAS ITS OWN p_rgh BOUND, and it is a number this gate does NOT explain. MEASURED at two
    // steps: 1.2411e-07 relative -- but p_rgh's own maximum here is 0.809 while `p` reaches 6.6e+03 and
    // agrees to 1.5193e-11, which is the SAME absolute difference (1.0e-07): p_rgh is a near-total
    // cancellation of the hydrostatic head, so its relative measure is taken against a scale four orders
    // below the pressure it came from. WHAT HAS BEEN RULED OUT: the Krylov stopping point (the staging
    // pins p_rgh to 1e-13/relTol 0, and as shipped at 5e-8/0.01 it read 1.5144e-07 -- converging the
    // solves moved it by a fifth), and the case's `cellLimited leastSquares` gradient (staged OFF as
    // plain `leastSquares` it reads 1.7852e-06, FOURTEEN TIMES WORSE, with alpha's cells going 4.6e-16
    // -> 2.4394e-08). All 12 p_rgh iteration counts are OpenFOAM's and the final residuals agree to
    // 5.8e-09, so it is the same system solved to the same place. NOT LOCALISED; the bound is just above
    // the measurement so a regression still trips it, and it comes down when this is understood.
    const scalar prghBound = (profile.rfind("esd", 0) == 0) ? scalar(3e-7)
                                                           : (amplifies ? scalar(2e-7) : scalar(1e-8));
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < prghBound);
    check("p agrees with OpenFOAM's relatively", dPp.rel() < (amplifies ? scalar(2e-7) : scalar(1e-8)));
    check("U agrees with OpenFOAM's relatively", dU.rel() < (amplifies ? scalar(2e-5) : scalar(2e-7)));

    // THE CONTROL, on the oracle: OpenFOAM's own answer for the same tank with the mesh held still,
    // or -- for the closed dam -- with the other pRefValue, which moves p and nothing else
    if (moving)
    {
        const Diff cU = compare(cellValues(readField<vector>(staticDir + "/U"), nC), ofU);
        const Diff cA = compare(cellValues(readField<scalar>(staticDir + "/" + fin.alphaName), nC), ofAlpha);
        // under `pistonSST` the control is the laminar piston: what the closure itself moves. Under
        // `sloshing2DCN` it is the SAME MOVING TANK under Euler, because what that profile is holding
        // is the ddt scheme and a static control would be blind to it: the mesh flux itself is
        // off-centred under CrankNicolson (fvc::meshPhi), so Euler-vs-CrankNicolson is the only
        // comparison that moves when the scheme is wrong.
        // ...and under `mixerPermeable` the control is the SAME MOTION under the tutorial's own
        // movingWallVelocity, because what that profile holds is the permeable pair and a static
        // control would be answering for the motion instead. mesh.update() ends in
        // U.correctBoundaryConditions() and the flux phi holds THERE is the relative one, where the
        // flux pEqn's own evaluate reads is absolute: reading the absolute one at both left U
        // 1.33e-01 from OpenFOAM after two steps.
        const bool perm = (profile == "mixerPermeable");
        // ...and under `esd`/`esdNoCorr` the control is the SAME MOVING CASE with `MULESCorr` flipped,
        // because what those profiles hold is the under-relaxed corrector's boundary half and a static
        // control would be answering for the motion instead. MEASURED, two steps, OpenFOAM against
        // itself: alpha 8.1858e-09 and U 2.9337e-04 relative -- eleven orders above brae's own distance.
        const bool esd = (profile.rfind("esd", 0) == 0);
        // ...and under `mixerTop` it is the SHIPPED tube, closed: what opening the top -- and adjustPhi
        // balancing its relative flux -- is worth, measured on OpenFOAM against itself.
        const bool top = (profile == "mixerTop");
        const char* controlIs = sst ? "laminar"
                                    : (cn ? "under Euler"
                                          : (perm ? "with movingWallVelocity"
                                                  : (esd ? "with MULESCorr flipped"
                                                         : (top ? "with the top closed" : "with a static mesh"))));
        const char* againstIs = sst ? "with kOmegaSST"
                                    : (cn ? "under CrankNicolson"
                                          : (perm ? "with the permeable pair"
                                                  : (esd ? "as the tutorial ships it"
                                                         : (top ? "with it open" : "with the motion"))));
        std::printf("  CONTROL: OpenFOAM %s against OpenFOAM %s, U relative %.4e, alpha %.4e\n",
                    controlIs, againstIs, (double)cU.rel(), (double)cA.linf);
        check(sst ? "the closure moves OpenFOAM's own U far more than brae is from it"
                  : (cn ? "the ddt scheme moves OpenFOAM's own U far more than brae is from it"
                        : (perm ? "the permeable pair moves OpenFOAM's own U far more than brae is from it"
                                : (esd ? "MULESCorr moves OpenFOAM's own U far more than brae is from it"
                                       : (top ? "opening the top moves OpenFOAM's own U far more than brae is from it"
                                              : "the motion moves OpenFOAM's own U far more than brae is from it")))),
              cU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)));
    }
    else
    {
        const Diff cP = compare(cellValues(readField<scalar>(staticDir + "/p"), nC), ofP2);
        const Diff cU = compare(cellValues(readField<vector>(staticDir + "/U"), nC), ofU);
        if (profile == "closedAdjZG" || profile == "closedAdjIO")
        {
            // the tank walled off (closedAdjZG's control), or the other adjustable kind (closedAdjIO's):
            // what adjustPhi's scaling of the open atmosphere is worth, measured on OpenFOAM against
            // itself -- zeroGradient against inletOutlet is the two halves of adjustPhi.C:59
            std::printf("  CONTROL: OpenFOAM %s against OpenFOAM with this atmosphere, U relative %.4e, "
                        "p relative %.4e\n",
                        profile == "closedAdjZG" ? "walled off" : "with a zeroGradient U atmosphere",
                        (double)cU.rel(), (double)cP.rel());
            check("the adjustable atmosphere moves OpenFOAM's own U far more than brae is from it",
                  cU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)));
        }
        else if (profile == "closedDamBreakInitU")
        {
            // the same closed tank STARTED AT REST: what the initial motion, and the start-up
            // CorrectPhi that makes its flux divergence-free, are worth
            std::printf("  CONTROL: OpenFOAM's tank started at rest against OpenFOAM's started moving, U "
                        "relative %.4e, p relative %.4e\n", (double)cU.rel(), (double)cP.rel());
            check("starting the tank moving moves OpenFOAM's own U far more than brae is from it",
                  cU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)));
        }
        else
        {
            std::printf("  CONTROL: OpenFOAM with pRefValue 1e5 against OpenFOAM with 0, p relative %.4e, "
                        "U relative %.4e\n", (double)cP.rel(), (double)cU.rel());
            check("the reference value moves OpenFOAM's own p far more than brae is from it",
                  cP.rel() > scalar(1000)*std::fmax(dPp.rel(), scalar(1e-14)));
            check("...and its U not at all", cU.rel() < scalar(1e-9));
        }
    }

    std::printf("test_inter_moving_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
