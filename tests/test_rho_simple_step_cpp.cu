// rhoSimpleFoam END TO END in _cpp, against real OpenFOAM.
//
// This is the gate PORT.md requires before any .cu is written for this solver: the whole case, run by the
// host reference, compared against OpenFOAM's own rhoSimpleFoam on the same mesh and dictionaries.
//
// LAMINAR, and that is a refusal rather than a simplification. The compressible turbulence closure is a
// separate manifest component and is not ported, so the driver REFUSES a case whose momentumTransport
// names RAS or LES instead of running it as if it were laminar. The gate therefore runs a laminar variant
// of the fixture -- a solver option OpenFOAM honours too -- and additionally asserts that the turbulent
// original IS refused, so "laminar only" is a stated boundary rather than an untested claim.
//
// THE INLET IS NEUTRALISED, as in every other rhoSimpleFoam gate: sbMatched's flowRateInletVelocity
// disagrees with OpenFOAM by ~2.4e-01 (see PORT.md) and would dominate an end-to-end field comparison
// that is meant to be about the solver.
//
// WHAT IS COMPARED. Both codes start from the same fields and run the same number of iterations, so this
// compares trajectories -- which is only meaningful because they are meant to be the SAME trajectory,
// iteration for iteration. The residual history is printed alongside, because a field comparison that
// agrees while the residuals diverge would mean the two are converging to the same place by different
// routes, and that is worth seeing.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "rhoSimpleFoam_cpp.cuh"
#include "scheme_parse.cuh"   // parseFieldDivScheme / parseFvSchemesControls: the CASE's schemes

#include <cmath>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

static int failures = 0;

static void report(const std::string& what, double got, double bound)
{
    const bool ok = got < bound;
    if (!ok) ++failures;
    std::printf("     %-34s %.6e   %s\n", what.c_str(), got, ok ? "ok" : "FAIL");
}

static void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-34s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

static double relL2(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    double num = 0.0, den = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

static std::vector<scalar> readInternal(const std::string& path, label nC)
{
    const FieldData<scalar> fd = readField<scalar>(path);
    if (fd.internalUniform) return std::vector<scalar>(nC, fd.internalUniformValue);
    return fd.internalField;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <startTime> <endTime> [turbulentCaseDir]\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string startT  = argv[2];
    const std::string endT    = argv[3];
    const int iters = std::atoi(endT.c_str()) - std::atoi(startT.c_str());

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;
    const FoamDict* rfl = rf ? rf->subDict("fields") : nullptr;

    std::printf("rhoSimpleFoam END TO END vs OpenFOAM (%d cells, %d iterations)\n", (int)nC, iters);

    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);
    std::printf("  turbulence: %s\n", f.turbulent ? f.rasModel.c_str() : "laminar");
    std::printf("  energy '%s', consistent %s, transonic %s\n", f.heName.c_str(),
                simpleDict && simpleDict->wordOr("consistent", "no") == "yes" ? "yes" : "no",
                simpleDict && simpleDict->wordOr("transonic", "no") == "yes" ? "yes" : "no");

    cpu::rhoSimple::StepInput in;
    // fvOptions and MRF: the components already refuse them BY NAME, but nothing was setting these flags
    // from the case, so a case carrying an fvOption ran with it silently ignored. aerofoilNACA0012's
    // `limitTemperature` is exactly that -- a source term OpenFOAM applies and brae does not.
    {
        auto has = [&](const char* rel)
        {
            std::ifstream f2((caseDir + "/" + rel).c_str());
            return f2.good();
        };
        // Walk the fvOptions dict rather than only noting that one exists: limitTemperature is
        // implemented (it is a correction, not a source), any other type is refused BY NAME. Reading the
        // file and finding nothing but limitTemperature is what lets aerofoilNACA0012 run; a case adding
        // an explicitPorositySource beside it still refuses.
        const std::string fvoPath = has("system/fvOptions") ? caseDir + "/system/fvOptions"
                                  : (has("constant/fvOptions") ? caseDir + "/constant/fvOptions" : "");
        if (!fvoPath.empty())
        {
            const FoamDict fvo = readDict(fvoPath);
            bool anyOther = false;
            for (const auto& entry : fvo.subs)
            {
                const FoamDict* o = &entry.second;
                const std::string ty = o->wordOr("type", "");
                if (ty == "limitTemperature")
                {
                    const std::string sel = o->wordOr("selectionMode", "all");
                    if (sel != "all")
                        throw std::runtime_error(
                            "rhoSimpleFoam: limitTemperature with selectionMode '" + sel
                            + "'. brae applies it over all cells; a cell subset is a different option. "
                              "Refusing rather than limiting the wrong cells.");
                    in.limitT    = true;
                    in.limitTmin = o->scalarOr("min", 0.0);
                    in.limitTmax = o->scalarOr("max", 0.0);
                    std::printf("  fvOption limitTemperature [%g, %g]\n",
                                (double)in.limitTmin, (double)in.limitTmax);
                }
                else if (!ty.empty())
                {
                    anyOther = true;
                }
            }
            in.hasFvOptions = anyOther;
        }
        in.hasMRF = has("constant/MRFProperties");
        if (in.hasFvOptions) std::printf("  the case declares fvOptions\n");
        if (in.hasMRF)       std::printf("  the case declares MRFProperties\n");
    }
    in.consistent = simpleDict && simpleDict->wordOr("consistent", "no") == "yes";
    in.transonic  = simpleDict && simpleDict->wordOr("transonic",  "no") == "yes";
    // The fixture's schemes: `div(phi,*) bounded Gauss upwind`, `laplacianSchemes default Gauss linear
    // corrected`. Stated here rather than parsed, because a scheme brae read wrongly would otherwise be
    // invisible -- the gate would compare two solvers running two discretisations and blame the driver.
    // PARSED from the case, not stated. Stating sbMatched's own schemes was safe while there was one
    // fixture and became a silent substitution the moment this binary was pointed at a second case:
    // aerofoilNACA0012 asks for `bounded Gauss linearUpwind limited` on div(phi,U) and
    // `cellLimited Gauss linear 1` on grad(U), where sbMatched asks for plain upwind and an unlimited
    // gradient. linearUpwind's deferred correction is a SOURCE term, and running upwind instead left the
    // wall-cell momentum source at 2.4e-02 against OpenFOAM's 2.5e+00.
    {
        auto pick = [&](const char* field)
        {
            const FieldDivScheme ds = parseFieldDivScheme(caseDir, field);
            return ds;
        };
        const FieldDivScheme dU  = pick("U");
        const FieldDivScheme dHe = pick(f.heName.c_str());
        in.schemeU  = dU.linearUpwind  ? cpu::rhoSimple::DivScheme::linearUpwind
                    : (dU.limited      ? cpu::rhoSimple::DivScheme::limitedLinear
                                       : cpu::rhoSimple::DivScheme::upwind);
        in.schemeHe = dHe.linearUpwind ? cpu::rhoSimple::DivScheme::linearUpwind
                    : (dHe.limited     ? cpu::rhoSimple::DivScheme::limitedLinear
                                       : cpu::rhoSimple::DivScheme::upwind);
        in.schemeKE      = in.schemeHe;   // div(phi,Ekp) follows the energy entry in every tutorial
        in.boundedU      = dU.bounded;
        in.boundedHe     = dHe.bounded;
        in.boundedKE     = dHe.bounded;
        in.schemeCoeffU  = dU.twoByk;

        DeviceSimpleControls sctl;
        parseFvSchemesControls(caseDir, sctl);
        in.correctedLaplacian = sctl.nonOrth;
        in.gradULimitK        = sctl.gradULimitK;
        std::printf("  schemes: div(phi,U) lu=%d bounded=%d | grad(U) cellLimited k=%g | "
                    "laplacian corrected=%d\n",
                    (int)dU.linearUpwind, (int)dU.bounded, (double)in.gradULimitK,
                    (int)in.correctedLaplacian);
    }
    in.relaxU   = re  ? re->scalarOr("U", 1.0) : 1.0;
    in.relaxHe  = re  ? re->scalarOr(f.heName, 1.0) : 1.0;
    in.relaxPEqn = re ? re->scalarOr("p", 1.0) : 1.0;
    in.relaxPEqnSpecified = (re != nullptr) && re->found("p");
    in.relaxP   = rfl ? rfl->scalarOr("p", 1.0) : 1.0;
    in.relaxRho = rfl ? rfl->scalarOr("rho", 1.0) : 1.0;
    in.relaxK       = re ? re->scalarOr("k", 1.0) : 1.0;
    in.relaxEpsilon = re ? re->scalarOr("epsilon", 1.0) : 1.0;
    in.boundedTurb  = true;   // the fixture's `div(phi,k)`/`div(phi,epsilon)` are `bounded Gauss upwind`

    for (int it = 1; it <= iters; ++it)
    {
        const cpu::rhoSimple::Residuals r = cpu::rhoSimple::rhoSimpleStep(f, in, m, g, patches);
        if (it <= 12 || it == iters || it % 50 == 0)
        {
            // The second turbulence scalar is the MODEL's: kEpsilon reports epsilon, kOmegaSST omega.
            // Looking up "epsilon" unconditionally threw std::out_of_range on an SST case -- a crash
            // where the solver had actually run.
            const char* second = r.count("epsilon") ? "epsilon" : (r.count("omega") ? "omega" : nullptr);
            if (r.count("k") && second)
                std::printf("     iter %4d   U %.3e   %s %.3e   p %.3e   k %.3e   %s %.3e\n", it,
                            (double)r.at("U"), f.heName.c_str(), (double)r.at(f.heName),
                            (double)r.at("p"), (double)r.at("k"), second, (double)r.at(second));
            else
                std::printf("     iter %4d   U %.3e   %s %.3e   p %.3e\n", it,
                            (double)r.at("U"), f.heName.c_str(), (double)r.at(f.heName), (double)r.at("p"));
        }
    }

    // ---- the fields, against OpenFOAM at the same iteration ----
    // THE BOUNDARY CONDITION'S DEFINING PROPERTY, checked before any field comparison: a
    // flowRateInletVelocity inlet must deliver the mass flow the case prescribed. sum(phi) over the patch
    // is exactly -massFlowRate when the velocity is held against the same rho the flux carries, and drifts
    // by whatever factor separates the two when it is not -- 24% on this fixture when the inlet stayed at
    // its `rhoInlet` seed. This is the invariant, so it is asserted rather than inferred from a field norm.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (f.U.boundary[pi]->bcCategory() != 9) continue;
        const scalar mdot = f.U.boundary[pi]->flowRateValue();
        scalar sumPhi = 0.0;
        for (label i = 0; i < patches[pi].size; ++i) sumPhi += f.phi.boundary[pi][i];
        const double rel = std::fabs(sumPhi + mdot) / std::fabs(mdot);
        char what[128];
        std::snprintf(what, sizeof(what), "%s delivers its prescribed mass flow", patches[pi].name.c_str());
        // At the CONVERGENCE level, not machine precision, and that is the honest statement here: the
        // inlet flux can only be as exact as the solution the loop has reached, and brae's U residual at
        // this point is 8.8e-09. The machine-precision form of this assertion lives in
        // rho_ueqn_vs_openfoam, which checks the same invariant on a single assembly at 2.2e-15.
        report(what, rel, 1e-6);
        std::printf("     %-40s sum(phi) %+.9e   prescribed %+.9e\n", "  (the flux it carries)",
                    sumPhi, -mdot);
    }

    std::printf("  fields vs OpenFOAM at t=%s\n", endT.c_str());
    const std::vector<scalar> ofP = readInternal(caseDir + "/" + endT + "/p", nC);
    const std::vector<scalar> ofT = readInternal(caseDir + "/" + endT + "/T", nC);
    // p, T and rho are REPORTED, not gated, and the control below is what decides that: on this fixture
    // they barely move from their initial values (p stays within 3.1e-06 of uniform 110000), so no bound
    // can both pass a correct solver and fail one that did nothing. Reporting a number the test cannot
    // stand behind as if it were a gate is the failure mode this whole port exists to avoid.
    std::printf("     %-34s %.6e   (reported, see the control)\n", "p", relL2(f.p.internal, ofP));
    std::printf("     %-34s %.6e   (reported, see the control)\n", "T", relL2(f.T.internal, ofT));
    {
        const FieldData<vector> uFd = readField<vector>(caseDir + "/" + endT + "/U");
        std::vector<scalar> a, b;
        for (label c = 0; c < nC; ++c)
        {
            a.push_back(f.U.internal[c].x); a.push_back(f.U.internal[c].y); a.push_back(f.U.internal[c].z);
            b.push_back(uFd.internalField[c].x); b.push_back(uFd.internalField[c].y); b.push_back(uFd.internalField[c].z);
        }
        // U IS the gate: it starts at (0,0,0) and ends at the solution, so the control below fails it by
        // a factor of ~7e3 and the bound means something.
        //
        // 5e-4. This was 2e-3 while brae's host path had no inletOutlet flux switch and the error sat
        // almost entirely on the outlet -- 1.5e-02 there against 2.6e-06 at the inlet. With the mixed
        // boundary condition implemented (fv_patch_field.cuh, valueFraction = neg(phi)) the outlet reads
        // 9.9e-06 and the whole field 1.5e-04, so the slack that was being held open for a known gap is
        // no longer there to hold. The per-patch confinement below is asserted for EVERY patch now,
        // including the outlet.
        report("U", relL2(a, b), 5e-4);
        // Where does it live? Every localisation this port has needed started here.
        std::vector<char> isB(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i) isB[patches[pi].faceCells[i]] = 1;
        double db = 0.0, nb2 = 0.0, di = 0.0, ni = 0.0;
        for (label c = 0; c < nC; ++c)
        {
            const double dx = (double)f.U.internal[c].x - (double)uFd.internalField[c].x;
            const double dy = (double)f.U.internal[c].y - (double)uFd.internalField[c].y;
            const double dz = (double)f.U.internal[c].z - (double)uFd.internalField[c].z;
            const double q = dx*dx + dy*dy + dz*dz;
            const double r = (double)uFd.internalField[c].x*(double)uFd.internalField[c].x
                           + (double)uFd.internalField[c].y*(double)uFd.internalField[c].y
                           + (double)uFd.internalField[c].z*(double)uFd.internalField[c].z;
            if (isB[c]) { db += q; nb2 += r; } else { di += q; ni += r; }
        }
        std::printf("       U rel: boundary cells %.4e   interior %.4e\n",
                    nb2 > 0 ? std::sqrt(db/nb2) : 0.0, ni > 0 ? std::sqrt(di/ni) : 0.0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].size) continue;
            double dp = 0.0, np = 0.0;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const double dx = (double)f.U.internal[c].x - (double)uFd.internalField[c].x;
                const double dy = (double)f.U.internal[c].y - (double)uFd.internalField[c].y;
                const double dz = (double)f.U.internal[c].z - (double)uFd.internalField[c].z;
                dp += dx*dx + dy*dy + dz*dz;
                np += (double)uFd.internalField[c].x*(double)uFd.internalField[c].x
                    + (double)uFd.internalField[c].y*(double)uFd.internalField[c].y
                    + (double)uFd.internalField[c].z*(double)uFd.internalField[c].z;
            }
            std::printf("         %-14s %.4e\n", patches[pi].name.c_str(),
                        np > 0 ? std::sqrt(dp/np) : 0.0);
            check(std::string("U confined: ") + patches[pi].name,
                  (np > 0 ? std::sqrt(dp/np) : 0.0) < 5e-4);
        }
        check("U interior is not where it lives", (ni > 0 ? std::sqrt(di/ni) : 0.0) < 5e-4);
    }
    {
        const std::vector<scalar> ofRho = readInternal(caseDir + "/" + endT + "/rho", nC);
        std::printf("     %-34s %.6e   (reported, see the control)\n", "rho",
                    relL2(f.rho.internal, ofRho));
    }

    // ---- THE CONTROL: the initial field must NOT pass those bounds. ----
    std::printf("  control -- the initial field must not pass\n");
    {
        cpu::rhoSimple::RhoSimpleFields z =
            cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                         m, g, patches);
        const FieldData<vector> uFd2 = readField<vector>(caseDir + "/" + endT + "/U");
        std::vector<scalar> za, zb;
        for (label c = 0; c < nC; ++c)
        {
            za.push_back(z.U.internal[c].x); za.push_back(z.U.internal[c].y); za.push_back(z.U.internal[c].z);
            zb.push_back(uFd2.internalField[c].x); zb.push_back(uFd2.internalField[c].y); zb.push_back(uFd2.internalField[c].z);
        }
        const double zu = relL2(za, zb);
        check("the start state fails the U bound", zu > 5e-4);
        std::printf("     %-34s U %.4e   p %.4e   T %.4e\n", "  (start-state errors)",
                    zu, relL2(z.p.internal, ofP), relL2(z.T.internal, ofT));
        std::printf("     %-34s\n", "  p and T start INSIDE their bounds -> not gated, as above");
    }

    // ---- THE REFUSAL: a turbulent case must be refused BY NAME, not run as laminar. ----
    std::printf("  refusal -- an unported RAS model\n");
    if (argc > 4)
    {
        // LaunderSharmaKE is a compressible RAS model brae does not have. It must be refused BY NAME, not run
        // as the kEpsilon it does have and not quietly as laminar -- either would converge to a smooth,
        // plausible, wrong field. The refusal lives in createFields, where OpenFOAM constructs the model.
        bool threw = false;
        std::string msg;
        try
        {
            (void)cpu::rhoSimple::createFields(caseDir + "/" + startT, std::string(argv[4]),
                                               simpleDict, &fvSolution, m, g, patches);
        }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        check("an unported RAS model is refused", threw);
        check("and the refusal names it", msg.find("LaunderSharmaKE") != std::string::npos);
    }
    else
    {
        std::printf("     %-34s %s\n", "unported-model fixture not supplied", "SKIP");
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
