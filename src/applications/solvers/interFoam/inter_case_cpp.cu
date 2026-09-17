// interFoam's createFields -- see inter_case_cpp.cuh for the provenance and for the four things the
// order of this file encodes.
#include "inter_case_cpp.cuh"
#include "brae_notice.cuh"
#include "foam_field_reader.cuh"
#include "read_surface_field.cuh"
#include "scheme_parse.cuh"
#include <filesystem>

namespace brae {
namespace cpu {
namespace interFoam {

// OpenFOAM's flux-conditional patches LOOK UP phi when they update (pressureInletOutletVelocity,
// inletOutlet, totalPressure: `patch().lookupPatchField<surfaceScalarField, scalar>(phiName_)` inside
// updateCoeffs), so whatever phi is at that moment decides which faces are inflow. brae's patches
// cannot look anything up; they are told, through updateFromFlux, and interFoam never told them. Every
// such face therefore sat at OUTFLOW for the whole run -- on capillaryRise's bottom inlet, where water
// is drawn IN, that dropped pressureInletOutletVelocity's two fixed tangential components and with them
// 2/3 muEff magSf deltaCoeffs of UEqn.A(): 5.33e+05 by that formula, 5.339e+05 measured against
// OpenFOAM's own UEqn.A() dump, in every cell of the bottom row.
void pushFluxToPatches(
    InterFields& f,
    const std::vector<FvPatch>& patches)
{
    for (std::size_t pi = 0; pi < patches.size() && pi < f.phi.boundary.size(); ++pi)
    {
        f.U.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
        f.p_rgh.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
        f.alpha1.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
    }
}

GeometricField<scalar> rhoWithPatchValues(
    const std::vector<scalar>& rhoCells,
    const std::vector<std::vector<scalar>>& rhoBnd,
    const std::vector<FvPatch>& patches)
{
    GeometricField<scalar> r;
    r.internal = rhoCells;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        if (q.type == "empty")
        {
            // OpenFOAM's empty patch field has no faces at all; it contributes nothing anywhere
            r.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
            continue;
        }
        const bool have = pi < rhoBnd.size() && rhoBnd[pi].size() == static_cast<std::size_t>(q.size);
        if (!have)
        {
            throw std::runtime_error(
                "brae interFoam: rho has no patch values on '" + q.name + "'. fvc::snGrad(rho) reads "
                "them -- rho's patches are `calculated`, not zeroGradient -- and defaulting to the cell "
                "value would zero a term that is 100% of the pressure source on an inflow patch.");
        }
        r.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, false, scalar(0), rhoBnd[pi]));
    }
    r.evaluateBoundary();
    return r;
}

void updateMixtureBoundary(InterFields& f, const std::vector<FvPatch>& patches)
{
    f.rhoBnd.resize(patches.size());
    f.muBnd.resize(patches.size());
    f.nuBnd.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& ab = f.alpha1.boundary[pi]->value();
        const std::size_t n = ab.size();
        f.rhoBnd[pi].resize(n);
        f.muBnd[pi].resize(n);
        f.nuBnd[pi].resize(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            // rho takes the RAW alpha and mu/nu the CLAMPED one, exactly as in the interior
            // (two_phase_mixture_cpp.cuh) -- the split is a property of the model, not of where it is
            // evaluated.
            const scalar a  = ab[i];
            const scalar ac = cpu::twoPhase::limitedAlpha(a);
            const auto&  p  = f.mixture.phases;
            const bool haveA2 = pi < f.alpha2Bnd.size() && i < f.alpha2Bnd[pi].size();
            const scalar a2 = haveA2 ? f.alpha2Bnd[pi][i] : scalar(1) - a;
            f.rhoBnd[pi][i] = a*p.rho1 + a2*p.rho2;
            f.muBnd[pi][i]  = ac*p.rho1*p.nu1 + (scalar(1) - ac)*p.rho2*p.nu2;
            f.nuBnd[pi][i]  = f.muBnd[pi][i] / (ac*p.rho1 + (scalar(1) - ac)*p.rho2);
        }
    }
}

namespace {

// fvSchemes' divSchemes entry for div(rhoPhi,U). The shipped tutorials ask for `Gauss linearUpwind
// grad(U)` (24), `Gauss vanLeerV` (8), `Gauss upwind` (6), `Gauss linear` (3) and
// `Gauss limitedLinear 0.2` (1). Anything else -- and vanLeerV, which is not ported -- is refused by
// name rather than run as something similar.
DivScheme parseMomentumDiv(const std::string& entry, scalar& coeff)
{
    coeff = scalar(1);
    std::vector<std::string> tok;
    {
        std::string cur;
        for (char ch : entry)
        {
            if (std::isspace(static_cast<unsigned char>(ch))) { if (!cur.empty()) { tok.push_back(cur); cur.clear(); } }
            else cur += ch;
        }
        if (!cur.empty()) tok.push_back(cur);
    }
    if (tok.empty() || tok[0] != "Gauss")
        throw std::runtime_error(
            "brae interFoam: `div(rhoPhi,U) " + entry + "` -- only `Gauss <scheme>` is ported.");
    const std::string s = (tok.size() > 1) ? tok[1] : "";
    if (s == "upwind")        return DivScheme::upwind;
    if (s == "linear")        return DivScheme::linear;
    if (s == "LUST")          return DivScheme::LUST;
    if (s == "linearUpwind")  return DivScheme::linearUpwind;
    if (s == "linearUpwindV") return DivScheme::linearUpwindV;
    if (s == "limitedLinearV")
    {
        if (tok.size() > 2) coeff = std::stod(tok[2]);
        return DivScheme::limitedLinearV;
    }
    if (s == "limitedLinear")
    {
        if (tok.size() > 2) coeff = std::stod(tok[2]);
        return DivScheme::limitedLinear;    // refused downstream, by name, with the reason
    }
    throw std::runtime_error(
        "brae interFoam: `div(rhoPhi,U) " + entry + "` is not ported. `vanLeerV` (8 shipped tutorials) "
        "is the V-variant of the alpha limiter and limits along the direction of steepest change "
        "rather than per component -- it is a different scheme, not a variant of vanLeer.");
}

AlphaFluxScheme parseAlphaDiv(const std::string& entry, const char* key)
{
    const std::string e = entry;
    if (e.find("vanLeer") != std::string::npos)   return AlphaFluxScheme::vanLeer;
    if (e.find("upwind")  != std::string::npos)   return AlphaFluxScheme::upwind;
    if (e.find("interfaceCompression") != std::string::npos)
        return AlphaFluxScheme::interfaceCompression;   // refused downstream, by name
    if (e.find("linear")  != std::string::npos)   return AlphaFluxScheme::linear;
    throw std::runtime_error(
        std::string("brae interFoam: `") + key + " " + entry + "` is not ported.");
}

AlphaDdt parseAlphaDdt(const std::string& entry)
{
    if (entry.find("CrankNicolson") != std::string::npos) return AlphaDdt::CrankNicolson;
    if (entry.find("localEuler")    != std::string::npos) return AlphaDdt::localEuler;
    if (entry.find("Euler")         != std::string::npos) return AlphaDdt::Euler;
    return AlphaDdt::other;                              // refused by offCentringCoeff, by name
}

// WHAT A CASE CAN ASK FOR THAT interFoam.C HONOURS AND brae DOES NOT EVEN READ. Every one of these ran
// to completion without a word until brae was run over all 44 shipped tutorials and the ones that
// reached `End:` were counted: laminar/damBreakWithObstacle and laminar/oscillatingBox both did, and
// both ask for `dynamicRefineFvMesh` -- adaptive refinement driven by alpha. Seventeen more tutorials
// carry a moving mesh and were only stopped because they hit some OTHER refusal first. braeInterFoam.cu's
// own header listed MRF and fvOptions as refused; nothing refused them.
//
// The rule for each is OpenFOAM's own, read from the source named beside it, because a refusal that
// fires on a case OpenFOAM would run as a static, source-free one is a defect too.
void refuseUnportedCaseInputs(
    const std::string& caseDir,
    const FoamDict& controlDict)
{
    // createDynamicFvMesh.H -> dynamicFvMesh::New (dynamicFvMeshNew.C:65-128): with no dictionary the
    // mesh is static; with one, `dynamicFvMesh` is MANDATORY and names the class.
    const std::string dyn = caseDir + "/constant/dynamicMeshDict";
    if (std::filesystem::exists(dyn))
    {
        const FoamDict d = readDict(dyn);
        const std::string type = d.wordOr("dynamicFvMesh", "");
        if (type.empty())
        {
            throw std::runtime_error(
                "brae interFoam: constant/dynamicMeshDict has no `dynamicFvMesh` entry. OpenFOAM reads "
                "it with get<word> and stops without one.");
        }
        if (type != "staticFvMesh")
        {
            throw std::runtime_error(
                "brae interFoam: constant/dynamicMeshDict asks for `dynamicFvMesh " + type + "`. brae's "
                "mesh does not move or refine -- interFoam.C's mesh.update(), correctPhi and the "
                "mesh-flux terms are not ported -- and running it on the mesh as written would solve a "
                "different problem. 19 of the 44 shipped interFoam tutorials ask for one.");
        }
    }

    // createMRF.H -> IOMRFZoneList (READ_IF_PRESENT); a zone is active unless it says otherwise
    // (MRFZone.C:248, :553).
    const std::string mrf = caseDir + "/constant/MRFProperties";
    if (std::filesystem::exists(mrf))
    {
        const FoamDict d = readDict(mrf);
        for (const auto& z : d.subs)
        {
            const std::string a = z.second.wordOr("active", "true");
            if (a == "false" || a == "no" || a == "off" || a == "0") continue;
            throw std::runtime_error(
                "brae interFoam: constant/MRFProperties has an active zone `" + z.first + "`. MRF adds "
                "a Coriolis source to UEqn and makes every flux relative; none of that is ported here.");
        }
    }

    // createFvOptions.H -> fv::options::createIOobject (fvOptions.C:46-84): constant/ FIRST, then
    // system/. An option is active unless it says otherwise (fvOption.C:72).
    for (const char* where : {"/constant/fvOptions", "/system/fvOptions"})
    {
        const std::string fo = caseDir + where;
        if (!std::filesystem::exists(fo)) continue;
        const FoamDict d = readDict(fo);
        for (const auto& o : d.subs)
        {
            const std::string a = o.second.wordOr("active", "true");
            if (a == "false" || a == "no" || a == "off" || a == "0") continue;
            throw std::runtime_error(
                "brae interFoam: " + std::string(where + 1) + " has an active option `" + o.first
                + "` (type `" + o.second.wordOr("type", "?") + "`). interFoam applies fvOptions to UEqn "
                  "and to U after every corrector; brae's interFoam applies none.");
        }
        // OpenFOAM stops at the first file it finds
        break;
    }

    // A function object does not normally touch the solution, and brae runs none. setTimeStep is the
    // exception: Time::adjustDeltaT ends with functionObjects_.adjustTimeStep(), so it OVERRIDES the
    // deltaT the Courant number chose -- the clock this solver now reproduces bit for bit.
    if (const FoamDict* fns = controlDict.subDict("functions"))
    {
        for (const auto& fo : fns->subs)
        {
            if (fo.second.wordOr("type", "") == "setTimeStep")
            {
                throw std::runtime_error(
                    "brae interFoam: controlDict's function object `" + fo.first + "` is a "
                    "setTimeStep. It overrides deltaT from inside Time::adjustDeltaT, and brae runs no "
                    "function objects.");
            }
        }
    }
}

void refuseUnportedTurbulence(const std::string& caseDir)
{
    const std::string path = caseDir + "/constant/momentumTransport";
    const std::string alt  = caseDir + "/constant/turbulenceProperties";
    const std::string p = std::filesystem::exists(path) ? path
                        : (std::filesystem::exists(alt) ? alt : std::string());
    if (p.empty()) return;                                // no dictionary at all -> laminar
    const FoamDict d = readDict(p);
    const std::string sim = d.wordOr("simulationType", "laminar");
    if (sim != "laminar")
        throw std::runtime_error(
            "brae interFoam: constant/" + std::filesystem::path(p).filename().string()
            + " asks for simulationType `" + sim + "`. interFoam's turbulence is "
              "incompressibleInterPhaseTransportModel, which selects a MIXTURE model and hands it the "
              "blended rho -- not the single-phase model brae already has. Refused rather than run "
              "with the wrong density in the closure.");
}

}   // namespace


InterFields buildInterFields(const std::string&          caseDir,
                             const std::string&          startDir,
                             const PrimitiveMesh&        m,
                             const FvGeometry&           g,
                             const std::vector<FvPatch>& patches)
{
    InterFields f;
    const label nC = m.nCells();

    refuseUnportedTurbulence(caseDir);

    // --- the dictionaries ---------------------------------------------------------------------
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    refuseUnportedCaseInputs(caseDir, controlDict);
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict fvSchemes   = readDict(caseDir + "/system/fvSchemes");

    f.mixture   = cpu::twoPhase::readTransportProperties(caseDir);
    // 1: the alpha field's NAME comes from `phases (water air)`, not from a convention.
    f.alphaName = "alpha." + f.mixture.phase1Name;
    f.interface = interfaceProps::readInterfaceCoeffs(fvSolution, f.alphaName, f.mixture.sigma);
    f.alphaCtl  = readAlphaControls(fvSolution, f.alphaName);
    f.mulesCtl  = f.alphaCtl.MULESCorr ? MULES::readControlsCorr(fvSolution, f.alphaName)
                                       : MULES::readControls(fvSolution, f.alphaName);
    f.timeCtl   = VoFTimeControls::read(controlDict);
    f.writeCadence = WriteCadence::read(controlDict);
    f.deltaT    = controlDict.scalarOr("deltaT", scalar(1e-3));

    {
        // fvSchemes' KEYS CONTAIN PARENTHESES AND COMMAS -- `div(rhoPhi,U)` is one token to OpenFOAM
        // and several to a dictionary tokenizer, so FoamDict cannot look them up. brae already reads
        // them as TEXT, scoped to the block (scheme_parse.cuh's fvSchemesBlock), and this uses the
        // same route rather than a second one that would parse the same file differently.
        const std::string all = readFvSchemesText(caseDir);
        std::string div = fvSchemesBlock(all, "divSchemes");
        if (div.empty())
            throw std::runtime_error("brae interFoam: fvSchemes has no divSchemes block.");
        auto entry = [&](const std::string& key, const std::string& dflt)
        {
            const std::size_t k = div.find(key);
            if (k == std::string::npos) return dflt;
            const std::size_t e = div.find(';', k);
            std::string st = div.substr(k + key.size(),
                                        e == std::string::npos ? std::string::npos : e - k - key.size());
            // trim
            std::size_t b = st.find_first_not_of(" \t\n\r");
            std::size_t f2 = st.find_last_not_of(" \t\n\r");
            return (b == std::string::npos) ? dflt : st.substr(b, f2 - b + 1);
        };
        const std::string uEntry = entry("div(rhoPhi,U)", "");
        if (uEntry.empty())
            throw std::runtime_error(
                "brae interFoam: fvSchemes names no `div(rhoPhi,U)`. interFoam's momentum convection "
                "has no default to fall back to -- every shipped tutorial names one.");
        f.divRhoPhiU   = parseMomentumDiv(uEntry, f.divRhoPhiUCoeff);
        f.divPhiAlpha  = parseAlphaDiv(entry("div(phi,alpha)",   "Gauss vanLeer"), "div(phi,alpha)");
        f.divPhirbAlpha= parseAlphaDiv(entry("div(phirb,alpha)", "Gauss linear"),  "div(phirb,alpha)");

        const std::string ddtBlock = fvSchemesBlock(all, "ddtSchemes");
        auto ddtEntry = [&](const std::string& key, const std::string& dflt)
        {
            const std::size_t k = ddtBlock.find(key);
            if (k == std::string::npos) return dflt;
            const std::size_t e = ddtBlock.find(';', k);
            std::string st = ddtBlock.substr(k + key.size(),
                                             e == std::string::npos ? std::string::npos : e - k - key.size());
            std::size_t b = st.find_first_not_of(" \t\n\r");
            std::size_t f2 = st.find_last_not_of(" \t\n\r");
            return (b == std::string::npos) ? dflt : st.substr(b, f2 - b + 1);
        };
        const std::string dflt = ddtEntry("default", "Euler");
        f.ddtAlpha = parseAlphaDdt(ddtEntry("ddt(alpha)", dflt));
        f.ddtU     = (dflt.find("Euler") != std::string::npos && dflt.find("localEuler") == std::string::npos)
                   ? DdtScheme::Euler
                   : (dflt.find("CrankNicolson") != std::string::npos ? DdtScheme::CrankNicolson
                   : (dflt.find("localEuler")    != std::string::npos ? DdtScheme::localEuler
                   : (dflt.find("backward")      != std::string::npos ? DdtScheme::backward
                                                                      : DdtScheme::steadyState)));
    }

    // fvSolution's PIMPLE block -- see InterFields::pimple for why momentumPredictor is read rather
    // than assumed.
    {
        const FoamDict* pim = fvSolution.subDict("PIMPLE");
        if (!pim)
            throw std::runtime_error(
                "brae interFoam: fvSolution has no PIMPLE block. interFoam is a PIMPLE solver and every "
                "shipped tutorial carries one; there is no default to fall back to.");
        f.pimple.nOuterCorrectors = static_cast<label>(pim->scalarOr("nOuterCorrectors", scalar(1)));
        f.pimple.nCorrectors      = static_cast<label>(pim->scalarOr("nCorrectors", scalar(1)));
        f.nNonOrthogonalCorrectors =
            static_cast<label>(pim->scalarOr("nNonOrthogonalCorrectors", scalar(0)));
        const std::string mp = pim->wordOr("momentumPredictor", "yes");
        f.momentumPredictorOn = !(mp == "no" || mp == "false" || mp == "off" || mp == "0");
        f.pimple.frozenFlow = false;
        f.pimple.turbOnFinalIterOnly = true;
    }

    // solvers/<alpha> -- the linear solve of the MULESCorr pre-solve. A case without MULESCorr never
    // solves for alpha and need not name a solver (capillaryRise does not).
    {
        const FoamDict* sv = fvSolution.subDict("solvers");
        const FoamDict* ad = sv ? sv->subDict(f.alphaName) : nullptr;
        if (ad)
        {
            f.aSolve.solver = ad->wordOr("solver", "");
            f.aSolve.smoother = ad->wordOr("smoother", "");
            f.aSolve.tol = ad->scalarOr("tolerance", scalar(1e-6));
            f.aSolve.relTol = ad->scalarOr("relTol", scalar(0));
            f.aSolve.maxIter = static_cast<int>(ad->scalarOr("maxIter", scalar(1000)));
            f.aSolve.nSweeps = static_cast<int>(ad->scalarOr("nSweeps", scalar(1)));
        }
        if (f.alphaCtl.MULESCorr && !f.aSolve.solver.empty() && !f.aSolve.gaussSeidel())
        {
            // smoothSolver with either Gauss-Seidel smoother is OpenFOAM's own on both paths
            // (smooth_solver_cpp.cuh, deviceSymGaussSeidel). Anything else still substitutes, and says
            // what it costs when the substitute is a poor match for a near-triangular upwind matrix.
            noticeApproximated("interFoam alpha pre-solve",
                "the case asks for `solver " + f.aSolve.solver + "; smoother " + f.aSolve.smoother +
                ";` and brae runs BiCGStab at the same tolerance (DILU-preconditioned on the host, "
                "Jacobi on the device). On damBreak the device's substitute left alpha 3.3e-06 from "
                "OpenFOAM at the case's own 1e-8.");
        }
        if (f.alphaCtl.MULESCorr && f.aSolve.solver.empty())
            throw std::runtime_error(
                "brae interFoam: `MULESCorr yes` solves an implicit alpha equation and fvSolution's `solvers/"
                + f.alphaName + "` names no `solver` for it. OpenFOAM refuses the same case.");
    }

    // solvers/U and solvers/UFinal -- read only when the case solves a momentum predictor, and then
    // REQUIRED exactly where OpenFOAM requires them.
    if (f.momentumPredictorOn)
    {
        const FoamDict* sv = fvSolution.subDict("solvers");
        auto readU = [&](const char* name, bool required, InterFields::AlphaLinearSolve& out)
        {
            const FoamDict* d = sv ? sv->subDict(name) : nullptr;
            if (!d)
            {
                if (!required) return;
                throw std::runtime_error(
                    std::string("brae interFoam: `momentumPredictor yes` and fvSolution has no `solvers/")
                    + name + "` entry. fvMatrix::solve() selects it by the final-iteration flag -- UFinal "
                      "on the last outer corrector, U on the others -- and OpenFOAM stops without it.");
            }
            out.solver = d->wordOr("solver", "");
            out.smoother = d->wordOr("smoother", "");
            out.tol = d->scalarOr("tolerance", scalar(1e-6));
            out.relTol = d->scalarOr("relTol", scalar(0));
            out.maxIter = static_cast<int>(d->scalarOr("maxIter", scalar(1000)));
            out.nSweeps = static_cast<int>(d->scalarOr("nSweeps", scalar(1)));
            if (!out.gaussSeidel())
            {
                noticeApproximated(std::string("interFoam ") + name + " solve",
                    "the case asks for `solver " + out.solver + "` and brae runs BiCGStab at the same "
                    "tolerance. Only smoothSolver with a Gauss-Seidel smoother is OpenFOAM's own here; "
                    "the difference is where the solve stops.");
            }
        };
        readU("UFinal", true, f.uSolveFinal);
        readU("U", f.pimple.nOuterCorrectors > 1, f.uSolve);
    }

    // solvers/p_rgh -- the case's own pressure solve. See InterFields::tolP for why this is read
    // rather than assumed, and why relTol comes from the Final entry.
    {
        const FoamDict* sv = fvSolution.subDict("solvers");
        const FoamDict* pr = sv ? sv->subDict("p_rgh") : nullptr;
        const FoamDict* pf = sv ? sv->subDict("p_rghFinal") : nullptr;
        if (!pr)
            throw std::runtime_error(
                "brae interFoam: fvSolution has no `solvers/p_rgh` entry. OpenFOAM reads the pressure "
                "solve's tolerance from there and every shipped tutorial carries one; assuming a "
                "tolerance would run the case to a convergence nobody asked for, which is exactly the "
                "defect this replaced.");
        // lduMatrix::solver::readControls (lduMatrixSolver.C:195-205): tolerance 1e-6, relTol 0,
        // maxIter 1000 when absent.
        auto readSolve = [&](const FoamDict& d)
        {
            InterFields::PressureLinearSolve s;
            s.solver = d.wordOr("solver", "");
            s.preconditioner = d.wordOr("preconditioner", "");
            s.tol = d.scalarOr("tolerance", scalar(1e-6));
            s.relTol = d.scalarOr("relTol", scalar(0));
            s.maxIter = static_cast<int>(d.scalarOr("maxIter", scalar(1000)));
            if (!s.pcgDIC())
                noticeApproximated("interFoam p_rgh solve",
                    "the case asks for `solver " + s.solver + "; preconditioner " +
                    s.preconditioner + ";` and brae runs PBiCGStab at the same tolerance. Only "
                    "PCG with DIC is OpenFOAM's own solver here; the difference is where the solve "
                    "stops, which on a VoF case is visible as alpha's over-1 excursion.");
            return s;
        };
        f.pSolve = readSolve(*pr);
        if (!pf)
            throw std::runtime_error(
                "brae interFoam: fvSolution has `solvers/p_rgh` but no `p_rghFinal`. pEqn.H solves "
                "the last corrector with p_rgh.select(finalInnerIter()), which OpenFOAM resolves to "
                "the Final entry and refuses to run without; guessing it would pick the last "
                "corrector's stopping point for the case.");
        f.pSolveFinal = readSolve(*pf);
    }

    // relaxationFactors/equations -- see InterFields::relaxEquationU.
    {
        const FoamDict* rf = fvSolution.subDict("relaxationFactors");
        const FoamDict* eq = rf ? rf->subDict("equations") : nullptr;
        if (eq)
        {
            // OpenFOAM resolves the name through the same regex machinery fvSolution uses everywhere;
            // `".*" 1` matches U, and an explicit `U` entry wins over a `default`.
            const scalar u   = eq->scalarOr("U", scalar(-1));
            const scalar any = eq->scalarOr("\".*\"", scalar(-1));
            const scalar def = eq->scalarOr("default", scalar(-1));
            const scalar v = (u >= 0) ? u : ((any >= 0) ? any : def);
            if (v >= 0) { f.relaxEquationU = true; f.relaxU = v; }
        }
    }

    // --- gravity ------------------------------------------------------------------------------
    f.g          = readGravity(caseDir);
    f.hRef       = readHRef(caseDir);
    f.ghRefValue = ghRef(f.g, f.hRef);

    // --- the read fields ----------------------------------------------------------------------
    f.alpha1 = buildField<scalar>(readField<scalar>(startDir + "/" + f.alphaName), patches, nC);
    f.U      = buildField<vector>(readField<vector>(startDir + "/U"),              patches, nC);
    f.p_rgh  = buildField<scalar>(readField<scalar>(startDir + "/p_rgh"),          patches, nC);
    f.alpha1.evaluateBoundary();
    f.U.evaluateBoundary();
    f.p_rgh.evaluateBoundary();

    // 3: createAlphaFluxes.H READS phi when it is there. A restart continues from the WRITTEN flux,
    // which is not fvc::flux(U) -- U and phi disagree by the continuity error the pressure corrector
    // has just driven down, and recomputing it discards exactly that.
    f.phi = readPhiIfPresent(startDir, patches, m.nInternalFaces(),
                             fvc::flux(f.U, m, g, patches), &f.phiWasRead);
    // ...AND EVERY PATCH THAT DECIDES INFLOW FROM OUTFLOW BY IT LEARNS IT. See pushFluxToPatches.
    pushFluxToPatches(f, patches);

    // --- the mixture --------------------------------------------------------------------------
    // 2: rho from the RAW alpha, mu and nu from the CLAMPED one -- see two_phase_mixture_cpp.cuh.
    f.alpha2.resize(static_cast<std::size_t>(nC));
    for (label c = 0; c < nC; ++c) f.alpha2[c] = scalar(1) - f.alpha1.internal[c];
    cpu::twoPhase::mixtureRho(f.alpha1.internal, f.alpha2, f.mixture.phases, f.rho);
    cpu::twoPhase::mixtureMu (f.alpha1.internal, f.mixture.phases, f.mu);
    cpu::twoPhase::mixtureNu (f.alpha1.internal, f.mu, f.mixture.phases, f.nu);

    // ...and the same three blends on every patch, from alpha's own boundary values.
    updateMixtureBoundary(f, patches);

    // --- gh, ghf and p ------------------------------------------------------------------------
    ghField(f.g, f.ghRefValue, g.C(), f.gh);
    {
        std::vector<vector> Cf(static_cast<std::size_t>(m.nInternalFaces()));
        for (label i = 0; i < m.nInternalFaces(); ++i) Cf[i] = g.Cf()[i];
        ghField(f.g, f.ghRefValue, Cf, f.ghfInternal);

        f.ghfBoundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const FvPatch& q = patches[pi];
            std::vector<vector> bCf(static_cast<std::size_t>(q.size));
            for (label i = 0; i < q.size; ++i) bCf[i] = g.Cf()[q.start + i];
            ghField(f.g, f.ghRefValue, bCf, f.ghfBoundary[pi]);
        }
    }
    staticPressure(f.p_rgh.internal, f.rho, f.gh, f.p);

    // interfaceProperties' CONSTRUCTOR calls calculateK (interfaceProperties.C:196-210). That first
    // pass is what leaves alpha's wall gradient non-zero for the second one to build on.
    interfaceProps::calculateK(f.alpha1, f.interface, m, g, patches, false, f.nHatf, f.K);

    // rhoPhi starts at the mass flux implied by the read alpha and phi: alphaEqn overwrites it every
    // step, but UEqn would read it on the very first outer iteration if it did not exist.
    {
        SurfaceScalarField alphaPhi;
        fluxWithScheme(f.phi, f.alpha1, f.divPhiAlpha, m, g, patches, alphaPhi);
        massFlux(alphaPhi, f.phi, f.mixture.phases.rho1, f.mixture.phases.rho2, f.rhoPhi);
    }

    return f;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
