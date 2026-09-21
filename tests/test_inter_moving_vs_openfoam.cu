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
    const bool deviceArm = (profile == "mixer" || profile == "solitary"
                         || profile == "cylinder" || profile == "solitaryGamg"
                         || profile == "piston" || profile == "flap");
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
    // `closed*`: a closed tank whose mesh does NOT move -- the pressure reference alone
    const bool moving = profile.rfind("closed", 0) != 0;
    check("brae ran the same number of steps", r.steps == nSteps);
    check("the case moves its mesh, and brae read it so", (fin.dynamicMesh != nullptr) == moving);
    // the waveMaker tutorials are OPEN: a totalPressure atmosphere fixes p_rgh's level
    const bool open = profile.rfind("solitary", 0) == 0 || profile.rfind("piston", 0) == 0
                   || profile.rfind("flap", 0) == 0 || profile.rfind("multi", 0) == 0;
    if (open)
    {
        check("p_rgh is fixed at the atmosphere, and brae read it so", !fin.pRef.needReference);
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

    // THE SOLVES: every p_rgh line, iteration counts and residuals
    const std::vector<LinearSolveRecord> ofP = brae::gatecheck::readOfPressureSolves(logPath);
    // THE WAVEMAKERS' SOLVES RUN 150 TO 480 PCG ITERATIONS to a tolerance of 1e-13 (the script
    // converges them), and on a solve that long the iteration it stops at is decided in the last bits,
    // which thirty steps of a deforming mesh carry forward: measured on the piston, OpenFOAM 218 and
    // brae 217, 163 and 164, and at the last step 194 and 197, every one of them ending below 1e-13 in
    // both codes. For those two profiles EVERY solve must end below the tolerance in both, a solve of
    // 100 iterations or fewer must take OpenFOAM's count, and a longer one must be within 2% of it; the
    // initial residuals are printed, not asserted, and the fields below carry the gate's own bounds.
    const bool longSolves = profile.rfind("piston", 0) == 0 || profile.rfind("flap", 0) == 0;
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
            check("...and one iteration earlier it is NOT: the statistic resolves a shift of one",
                  worstEarlier > curveBound);
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
        failures += brae::gatecheck::compareSolves("host", r.pSolves, ofP, nSteps, "p_rgh", scalar(1e-10),
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
        check("the device's alpha is as close to OpenFOAM as the host's",
              eA.linf <= scalar(20)*std::fmax(dA.linf, scalar(1e-300)));
        check("...its p_rgh", eP.rel() <= scalar(20)*std::fmax(dP.rel(), scalar(1e-300)));
        check("...and its U", eU.rel() <= scalar(20)*std::fmax(dU.rel(), scalar(1e-300)));
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
        check("Uf on the internal faces is OpenFOAM's", dUf.rel() < scalar(2e-7) && !ofUf.internalField.empty());
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
                  eUf.rel() <= scalar(20)*std::fmax(dUf.rel(), scalar(1e-300)));
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
        check("the moving walls carry OpenFOAM's velocity", nWall > 0 && dWall <= scalar(1e-12)*wallScale);
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
    // interFoam (BRAE_DUMP_ITER = the last step): both sides hold the last corrector of the last step.
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
        std::printf("  pressure corrector against tools/dumpInterFoam, last corrector of step %ld:\n",
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
    Diff dK;
    Diff dOm;
    Diff dNut;
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

    // BOUNDS: see the script for what was measured
    check("alpha agrees with OpenFOAM's absolutely", dA.linf < scalar(1e-9));
    check("p_rgh agrees with OpenFOAM's relatively", dP.rel() < scalar(1e-8));
    check("p agrees with OpenFOAM's relatively", dPp.rel() < scalar(1e-8));
    check("U agrees with OpenFOAM's relatively", dU.rel() < scalar(2e-7));

    // THE CONTROL, on the oracle: OpenFOAM's own answer for the same tank with the mesh held still,
    // or -- for the closed dam -- with the other pRefValue, which moves p and nothing else
    if (moving)
    {
        const Diff cU = compare(cellValues(readField<vector>(staticDir + "/U"), nC), ofU);
        const Diff cA = compare(cellValues(readField<scalar>(staticDir + "/" + fin.alphaName), nC), ofAlpha);
        // under `pistonSST` the control is the laminar piston: what the closure itself moves
        std::printf("  CONTROL: OpenFOAM %s against OpenFOAM %s, U relative %.4e, alpha %.4e\n",
                    sst ? "laminar" : "with a static mesh", sst ? "with kOmegaSST" : "with the motion",
                    (double)cU.rel(), (double)cA.linf);
        check(sst ? "the closure moves OpenFOAM's own U far more than brae is from it"
                  : "the motion moves OpenFOAM's own U far more than brae is from it",
              cU.rel() > scalar(1000)*std::fmax(dU.rel(), scalar(1e-14)));
    }
    else
    {
        const Diff cP = compare(cellValues(readField<scalar>(staticDir + "/p"), nC), ofP2);
        const Diff cU = compare(cellValues(readField<vector>(staticDir + "/U"), nC), ofU);
        if (profile == "closedDamBreakInitU")
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
