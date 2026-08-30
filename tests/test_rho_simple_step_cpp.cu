// rhoSimpleFoam END TO END in _cpp, against real OpenFOAM.
//
// This is the gate PORT.md requires before any .cu is written for this solver: the whole case, run by the
// host reference, compared against OpenFOAM's own rhoSimpleFoam on the same mesh and dictionaries.
//
// TURBULENT, AND WITH THE FIXTURE'S OWN INLET. This block used to say the opposite -- "LAMINAR, and that
// is a refusal", and "THE INLET IS NEUTRALISED" -- and both were true when written and are not now. The
// driving script says so plainly (rho_simple_end_to_end_vs_openfoam.sh:5 "TURBULENT: the fixture's own
// kEpsilon, as it ships"; :15 "THE FIXTURE'S OWN INLET ... flowRateInletVelocity, not a substitute"), and
// the binary's turbulent branch is live below. The closure is ported and gated separately; the
// flowRateInletVelocity defect the neutralisation existed for is RETIRED (PORT.md: rAU 4.58e-05 ->
// 6.13e-15, momentum boundaryCoeffs 4.15e-01 -> 4.89e-16).
//
// Getting this wrong in a header is not cosmetic: it is the only whole-solver comparison against real
// OpenFOAM, so a stale "laminar, inlet neutralised" note rules out, for the next reader, the two things
// the run actually exercises -- the kEpsilon closure and a real inlet at |U| ~ 523 m/s -- and any residual
// disagreement then gets attributed anywhere but there.
//
// What IS still asserted as a boundary: an UNPORTED RAS model is refused BY NAME. The script builds a
// separate copy with `RASModel LaunderSharmaKE` for exactly that check (:85-88).
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
#include "scheme_parse.cuh"
#include "fvOptions_cpp.cuh"   // parseFieldDivScheme / parseFvSchemesControls: the CASE's schemes

#include "mrf_read.cuh"   // readCellZones: the porosity zone this case constrains
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
        // The IMPLEMENTED options, read through the same reader the incompressible path uses. Its
        // `unsupported` field carries the type name of anything it does not implement, so a case with an
        // explicitPorositySource brae has AND an option it does not is still refused by name.
        {
            static cpu::fvOptions::OptionList opts;
            opts = cpu::fvOptions::read(caseDir, m);
            const std::string bad = opts.firstUnsupported();
            if (!bad.empty())
            {
                in.hasFvOptions = true;
                in.fvOptionUnsupported = bad;
                std::printf("  fvOptions: '%s' is not implemented -- the case will be refused\n",
                            bad.c_str());
                std::fflush(stdout);   // the refusal aborts; an unflushed buffer loses this line
            }
            else if (!opts.empty())
            {
                in.fvOpts = &opts;
                in.hasFvOptions = false;
                std::printf("  fvOptions: %zu option(s), all implemented\n", opts.options.size());
            }
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
        in.gradKLimitK        = sctl.gradKLimitK;
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
    // "the case NAMES a factor", not "the factor is below 1" -- fvMatrix::relax(1.0) still applies the
    // dominance clamp, and OpenFOAM does not relax at all when fvSolution names nothing.
    in.relaxEquationU   = (re != nullptr) && re->found("U");
    in.relaxEquationHe  = (re != nullptr) && re->found(f.heName);
    in.relaxEquationK   = (re != nullptr) && re->found("k");
    in.relaxEquationEps = (re != nullptr) && re->found("epsilon");
    in.boundedTurb  = true;   // the fixture's `div(phi,k)`/`div(phi,epsilon)` are `bounded Gauss upwind`

    double lastUResidual = 1.0;   // the convergence the loop actually reached; bounds below scale with it
    for (int it = 1; it <= iters; ++it)
    {
        const cpu::rhoSimple::Residuals r = cpu::rhoSimple::rhoSimpleStep(f, in, m, g, patches);
        if (r.count("U")) lastUResidual = (double)r.at("U");
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
        // AT THE CONVERGENCE THE LOOP REACHED, expressed against the U residual rather than as a fixed
        // number. The inlet flux can only be as exact as the solution: on sbMatched brae reaches 8.8e-09
        // and delivers the flux to 1.4e-08, while squareBend stops at OpenFOAM's own residualControl
        // (U 1e-04) and delivers it to 5.5e-05. A fixed 1e-6 asserted the first case's convergence on the
        // second and failed a solver that was doing exactly the right thing. The absolute floor keeps the
        // assertion meaningful when the residual is tiny; the machine-precision form of this same
        // invariant lives in rho_ueqn_vs_openfoam, on a single assembly, at 2.2e-15.
        const double massFlowBound = std::max(1e-9, 10.0 * lastUResidual);
        report(what, rel, massFlowBound);
        std::printf("     %-40s bound %.3e (10x the U residual %.3e)\n", "  (what it is held to)",
                    massFlowBound, lastUResidual);
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
        // PER COMPONENT. A uniform whole-field number cannot tell "every component is a little off" from
        // "one component is badly off and the others are exact" -- and on a duct the cross-stream
        // component is both the smallest and the last to converge (OpenFOAM's own Uz residual floors at
        // 8.7e-07 here while Ux reaches 1.1e-08), so it can carry the whole disagreement while
        // contributing almost nothing to |U|.
        {
            const char* cn[3] = {"Ux", "Uy", "Uz"};
            for (int k2 = 0; k2 < 3; ++k2)
            {
                double dc = 0.0, nc = 0.0;
                for (label c = 0; c < nC; ++c)
                {
                    const double a2 = (double)reinterpret_cast<const scalar*>(&f.U.internal[c])[k2];
                    const double b2 = (double)reinterpret_cast<const scalar*>(&uFd.internalField[c])[k2];
                    dc += (a2 - b2)*(a2 - b2); nc += b2*b2;
                }
                std::printf("       %s rel %.4e   (|%s| rms %.4e)\n", cn[k2],
                            nc > 0 ? std::sqrt(dc/nc) : 0.0, cn[k2], std::sqrt(nc/nC));
            }
        }
        std::printf("       U rel: boundary cells %.4e   interior %.4e\n",
                    nb2 > 0 ? std::sqrt(db/nb2) : 0.0, ni > 0 ? std::sqrt(di/ni) : 0.0);
        // AND BY CELLZONE, because the per-patch figures below cannot separate two different causes on
        // this case: porosityWall's face cells are INSIDE the porous block, so "the slip patch disagrees"
        // and "the porosity disagrees" produce the same number there. The zone split does separate them.
        for (const auto& z : readCellZones(caseDir + "/constant/polyMesh"))
        {
            double din = 0.0, nin = 0.0, dout = 0.0, nout = 0.0;
            std::vector<char> inz(nC, 0);
            for (label c : z.second) if (c >= 0 && c < nC) inz[c] = 1;
            for (label c = 0; c < nC; ++c)
            {
                const double dx = (double)f.U.internal[c].x - (double)uFd.internalField[c].x;
                const double dy = (double)f.U.internal[c].y - (double)uFd.internalField[c].y;
                const double dz = (double)f.U.internal[c].z - (double)uFd.internalField[c].z;
                const double q = dx*dx + dy*dy + dz*dz;
                const double r = (double)uFd.internalField[c].x*(double)uFd.internalField[c].x
                               + (double)uFd.internalField[c].y*(double)uFd.internalField[c].y
                               + (double)uFd.internalField[c].z*(double)uFd.internalField[c].z;
                if (inz[c]) { din += q; nin += r; } else { dout += q; nout += r; }
            }
            std::printf("       U in zone '%s' %.4e   outside %.4e\n", z.first.c_str(),
                        nin > 0 ? std::sqrt(din/nin) : 0.0, nout > 0 ? std::sqrt(dout/nout) : 0.0);
        }
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
        // rho's RANGE beside OpenFOAM's. rho.relax() with this case's 0.01 moves rho only 1% of the way
        // to thermo.rho() each iteration, so the min is a direct read on whether the relaxation ran at
        // all: an unrelaxed rho lands on thermo.rho() itself, a relaxed one a hundredth of the way there.
        {
            double bmin = 1e300, bmax = -1e300, omin = 1e300, omax = -1e300;
            for (label c = 0; c < nC; ++c)
            {
                bmin = std::min(bmin, (double)f.rho.internal[c]);
                bmax = std::max(bmax, (double)f.rho.internal[c]);
                omin = std::min(omin, (double)ofRho[c]);
                omax = std::max(omax, (double)ofRho[c]);
            }
            std::printf("       rho brae [%.8g .. %.8g]   OF [%.8g .. %.8g]   relaxRho %g\n",
                        bmin, bmax, omin, omax, (double)in.relaxRho);
            label am = 0;
            for (label c = 0; c < nC; ++c) if ((double)f.rho.internal[c] >= bmax) { am = c; break; }
            const std::vector<scalar> ofPf = readInternal(caseDir + "/" + endT + "/p", nC);
            const std::vector<scalar> ofTf = readInternal(caseDir + "/" + endT + "/T", nC);
            std::printf("       argmax cell %d: brae rho %.8g p %.8g T %.8g | OF rho %.8g p %.8g T %.8g\n",
                        (int)am, (double)f.rho.internal[am], (double)f.p.internal[am],
                        (double)f.T.internal[am], (double)ofRho[am], (double)ofPf[am], (double)ofTf[am]);
        }
    }
    // THE CLOSURE'S OWN FIXED POINT. Nothing above measures it, and it is not a spectator: nut feeds
    // muEff, which is a coefficient of the momentum equation this gate does bound. The closure gates
    // (rho_kepsilon, rho_komegasst) assemble ONE system from OpenFOAM's adopted k, epsilon and U and
    // read machine precision -- which says the assembly is right GIVEN OpenFOAM's state, and says
    // nothing about where brae's own k and epsilon converge to over 8000 iterations of their own.
    // angledDuct is where that mattered: momentum coefficients exact to 2.6e-15 at iteration 200, and
    // U still 9.4e-04 out at convergence with p, T and rho an order of magnitude better.
    if (f.turbulent)
    {
        for (const char* fld : {"k", "epsilon", "omega", "nut", "alphat"})
        {
            const std::string path = caseDir + "/" + endT + "/" + fld;
            if (!std::ifstream(path.c_str()).good()) continue;
            const GeometricField<scalar>* bf = nullptr;
            const std::string n(fld);
            if      (n == "k")       bf = &f.k;
            else if (n == "epsilon") bf = &f.epsilon;
            else if (n == "omega")   bf = &f.omega;
            else if (n == "nut")     bf = &f.nut;
            else if (n == "alphat")  bf = &f.alphat;
            if (!bf || static_cast<label>(bf->internal.size()) != nC) continue;
            const std::vector<scalar> of = readInternal(path, nC);
            std::printf("     %-34s %.6e   (reported, see the control)\n", fld,
                        relL2(bf->internal, of));
            // SPLIT BY CELLZONE. angledDuct's fvOptions pins k = 1, epsilon = 150 and T = 350 across the
            // 8000 porosity cells, and OpenFOAM's converged fields hold every one of them EXACTLY. So a
            // whole-field number here averages 8000 cells that must agree to the last bit with 20000 that
            // are free -- and if brae ever stopped applying the constraint, the in-zone figure is the only
            // place it would show. Printed separately for exactly that reason.
            for (const auto& z : readCellZones(caseDir + "/constant/polyMesh"))
            {
                double din = 0.0, nin = 0.0, dout = 0.0, nout = 0.0;
                std::vector<char> inz(nC, 0);
                for (label c : z.second) if (c >= 0 && c < nC) inz[c] = 1;
                for (label c = 0; c < nC; ++c)
                {
                    const double d = (double)bf->internal[c] - (double)of[c];
                    if (inz[c]) { din += d*d; nin += (double)of[c]*(double)of[c]; }
                    else        { dout += d*d; nout += (double)of[c]*(double)of[c]; }
                }
                std::printf("       %s in zone '%s' %.4e   outside %.4e\n", fld, z.first.c_str(),
                            nin  > 0 ? std::sqrt(din/nin)   : 0.0,
                            nout > 0 ? std::sqrt(dout/nout) : 0.0);
            }
        }
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

    // ---- THE REFUSAL: an unported NUT WALL FUNCTION ------------------------------------------
    // The closure computes nutkWallFunction unconditionally -- for the wall nut AND, through
    // deviceWallEpsG0's `nutWall` selector, for the near-wall production. OpenFOAM dispatches on nut's
    // own patch field (nutWallFunctionFvPatchScalarField.C:181-184) and everything downstream READS the
    // result (epsilonWallFunctionFvPatchScalarField.C:333-334 `turbModel.nut(patchi)`), so a case naming
    // any other member of the family got nutk's value at both sites with nothing to say so.
    //
    // The refusal lives in createFields because that is the last place the dictionary TYPE exists: once
    // buildField has run, the patch-field object no longer carries it, which is why neither closure could
    // have checked. argv[5] is a time directory whose nut names a different wall function.
    std::printf("  refusal -- an unported nut wall function\n");
    if (argc > 5)
    {
        bool threw = false;
        std::string msg;
        try
        {
            (void)cpu::rhoSimple::createFields(std::string(argv[5]), caseDir,
                                               simpleDict, &fvSolution, m, g, patches);
        }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        check("an unported nut wall function is refused", threw);
        if (threw) std::printf("     %s\n", msg.substr(0, 150).c_str());

        // THE NEGATIVE CONTROL. The unmodified time directory must be ACCEPTED -- without it this passes
        // on a createFields that throws for any reason at all, which is not a refusal but a broken read.
        bool threw2 = false;
        try
        {
            (void)cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir,
                                               simpleDict, &fvSolution, m, g, patches);
        }
        catch (const std::exception&) { threw2 = true; }
        check("the case's own nutkWallFunction is ACCEPTED (negative control)", !threw2);
    }
    else
    {
        std::printf("     %-34s %s\n", "unported-nut fixture not supplied", "SKIP");
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
