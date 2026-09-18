// interFoam's createFields -- see inter_case_cpp.cuh for the provenance and for the four things the
// order of this file encodes.
#include "inter_case_cpp.cuh"
#include "inter_peqn_cpp.cuh"
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
    // rhoPhi does not exist yet at the first call, from buildInterFields before the mixture is built;
    // nothing reads a flux that early, and the call that closes buildInterFields hands it over
    const SurfaceScalarField* rhoPhi = f.rhoPhi.boundary.size() == patches.size() ? &f.rhoPhi : nullptr;
    for (std::size_t pi = 0; pi < patches.size() && pi < f.phi.boundary.size(); ++pi)
    {
        // each condition the flux its own `phi` entry NAMES -- see namedPatchFlux
        auto fluxFor = [&](const std::string& name) -> const std::vector<scalar>*
        {
            if (name == "rhoPhi" && !rhoPhi) return nullptr;
            return &namedPatchFlux(name, pi, patches[pi].name, f.phi, rhoPhi);
        };
        if (const std::vector<scalar>* q = fluxFor(f.U.boundary[pi]->fluxName()))
        {
            f.U.boundary[pi]->updateFromFlux(*q);
        }
        if (const std::vector<scalar>* q = fluxFor(f.p_rgh.boundary[pi]->fluxName()))
        {
            f.p_rgh.boundary[pi]->updateFromFlux(*q);
        }
        if (const std::vector<scalar>* q = fluxFor(f.alpha1.boundary[pi]->fluxName()))
        {
            f.alpha1.boundary[pi]->updateFromFlux(*q);
        }
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
// `Gauss limitedLinear 0.2` (1). Anything else is refused by name rather than run as something
// similar.
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
    if (s == "vanLeerV")      return DivScheme::vanLeerV;
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
        "brae interFoam: `div(rhoPhi,U) " + entry + "` is not ported. brae has upwind, linear, "
        "linearUpwind, linearUpwindV, limitedLinearV, LUST and vanLeerV here.");
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
    // constant/dynamicMeshDict is read by DynamicMotionSolverFvMesh::New in buildInterFields, which
    // refuses by name every motion that is not the rigid motion of the whole mesh.

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

NonOrthScheme readNonOrthScheme(
    const std::string& fvSchemesText,
    const std::string& block)
{
    const std::string blk = fvSchemesBlock(fvSchemesText, block);
    const std::size_t q = blk.find("default");
    if (blk.empty() || q == std::string::npos)
        throw std::runtime_error(
            "brae interFoam: fvSchemes `" + block + "` has no `default`. interFoam takes every "
            "laplacian and snGrad through it; OpenFOAM stops on a case without one.");
    const std::size_t e = blk.find(';', q);
    NonOrthScheme s;
    s.raw = blk.substr(q + 7, e == std::string::npos ? std::string::npos : e - q - 7);
    const std::size_t b = s.raw.find_first_not_of(" \t\n\r");
    s.raw = (b == std::string::npos) ? std::string() : s.raw.substr(b);
    if (schemeHasWord(s.raw, "limited"))
    {
        // `limited <c>` and `limited corrected <c>` are the same scheme (limitedSnGrad.H:98-124):
        // 1 is fully corrected, 0 is uncorrected.
        double c = -1;
        const char* at = s.raw.c_str() + s.raw.find("limited") + 7;
        while (*at && !(std::isdigit(static_cast<unsigned char>(*at)) || *at == '.'))
        {
            ++at;
        }
        if (std::sscanf(at, "%lf", &c) != 1 || c < 0 || c > 1)
            throw std::runtime_error(
                "brae interFoam: fvSchemes `" + block + "` default is `" + s.raw + "`, a limited "
                "scheme with no coefficient in [0,1].");
        s.corrected = (c > 0);
        s.limitCoeff = (c < 1) ? static_cast<scalar>(c) : scalar(0);
        return s;
    }
    if (schemeHasWord(s.raw, "corrected"))
    {
        s.corrected = true;
        return s;
    }
    if (schemeHasWord(s.raw, "uncorrected") || schemeHasWord(s.raw, "orthogonal")) return s;
    throw std::runtime_error(
        "brae interFoam: fvSchemes `" + block + "` default is `" + s.raw + "`, which names none of "
        "corrected, uncorrected, orthogonal or limited.");
}

// 1 - cos(angle between the cell-centre vector and the face normal), the largest over the internal
// faces. Exactly 0 on a mesh of rectangles; boundary faces do not enter, because v2412's
// fvPatch::delta() is already the patch-NORMAL part of Cf - Cn for every non-coupled patch.
scalar maxNonOrthogonality(
    const PrimitiveMesh& m,
    const FvGeometry& g)
{
    scalar worst = 0;
    for (label fi = 0; fi < m.nInternalFaces(); ++fi)
    {
        const vector d = g.C()[m.neighbour()[fi]] - g.C()[m.owner()[fi]];
        const vector& S = g.Sf()[fi];
        const scalar c = dot(d, S) / (mag(d) * g.magSf()[fi]);
        worst = std::fmax(worst, scalar(1) - c);
    }
    return worst;
}

// interFoam's pressure laplacian, its three snGrads and the momentum laplacian are assembled
// ORTHOGONAL here, whatever the case says -- and 28 of the 44 tutorials say `corrected` or `limited`.
// On a mesh of rectangles that is not a substitution: the correction vector n - d/(n.d) is zero, so
// `corrected` and `orthogonal` are one scheme, which is why damBreak and capillaryRise agree with
// OpenFOAM to 1e-12 under `Gauss linear corrected`. On any other mesh it IS one, it was silent, and
// the only thing that kept it from running was that every such tutorial was refused for something else.
void refuseUncorrectedOnSkewMesh(
    const NonOrthScheme& laplacian,
    const NonOrthScheme& snGrad,
    const PrimitiveMesh& m,
    const FvGeometry& g)
{
    if (!laplacian.corrected && !snGrad.corrected) return;
    const scalar worst = maxNonOrthogonality(m, g);
    // round-off on a mesh of rectangles is 1e-16; one degree is 1.5e-04
    if (worst < scalar(1e-10)) return;
    const scalar degrees = std::acos(scalar(1) - worst) * scalar(180) / scalar(3.14159265358979323846);
    throw std::runtime_error(
        "brae interFoam: fvSchemes asks for a non-orthogonal correction (laplacianSchemes `"
        + laplacian.raw + "`, snGradSchemes `" + snGrad.raw + "`) and the mesh is non-orthogonal by "
        "up to " + std::to_string(degrees) + " degrees. brae's interFoam assembles the pressure "
        "laplacian and its snGrads orthogonal; that is the case's own scheme only where the "
        "correction vanishes. Refused rather than run `orthogonal` under the name `corrected`.");
}


// THE PRESSURE REFERENCE OF A CLOSED CASE -- see InterFields::PressureReference.
//
// setRefCell (findRefCell.C:36-110): needReference() is "no patch fixes a value"; then `pRefCell`,
// else `pRefPoint` located with mesh.findCell(point, FACE_PLANES) -- the nearest cell centre if the
// point is inside that cell by every face plane, else the FIRST cell in index order that is -- and
// `pRefValue`, all mandatory. OpenFOAM falls back to an octree search with the cells decomposed
// into tets when the plane test finds no cell; that fallback is refused here by name.
InterFields::PressureReference readPressureReference(
    const GeometricField<scalar>& p_rgh,
    const FoamDict& fvSolution,
    const PrimitiveMesh& m,
    const FvGeometry& g)
{
    InterFields::PressureReference r;
    r.needReference = true;
    for (const auto& pf : p_rgh.boundary)
    {
        if (pf->fixesValue())
        {
            r.needReference = false;
            break;
        }
    }
    if (!r.needReference) return r;

    const FoamDict* pim = fvSolution.subDict("PIMPLE");
    if (!pim)
    {
        throw std::runtime_error("brae interFoam: fvSolution has no PIMPLE block for the pressure reference.");
    }
    if (pim->found("pRefCell"))
    {
        r.pRefCell = static_cast<label>(pim->scalarOr("pRefCell", scalar(-1)));
        if (r.pRefCell < 0 || r.pRefCell >= m.nCells())
        {
            throw std::runtime_error(
                "brae interFoam: PIMPLE's pRefCell " + std::to_string(r.pRefCell) + " is outside the mesh's "
                + std::to_string(m.nCells()) + " cells. OpenFOAM stops on the same condition.");
        }
    }
    else if (pim->found("pRefPoint"))
    {
        const std::vector<scalar> pv = pim->scalarListOr("pRefPoint", {});
        if (pv.size() != 3)
        {
            throw std::runtime_error("brae interFoam: PIMPLE's pRefPoint is not a point.");
        }
        const vector refPoint{pv[0], pv[1], pv[2]};
        // cell -> faces, for primitiveMesh::pointInCell
        std::vector<std::vector<label>> cellFaces(static_cast<std::size_t>(m.nCells()));
        for (label f = 0; f < m.nFaces(); ++f)
        {
            cellFaces[static_cast<std::size_t>(m.owner()[f])].push_back(f);
            if (f < m.nInternalFaces())
            {
                cellFaces[static_cast<std::size_t>(m.neighbour()[f])].push_back(f);
            }
        }
        auto pointInCell = [&](label celli)
        {
            for (const label nFace : cellFaces[static_cast<std::size_t>(celli)])
            {
                const vector proj = refPoint - g.Cf()[nFace];
                vector normal = g.Sf()[nFace];
                if (m.owner()[nFace] != celli)
                {
                    normal = scalar(-1)*normal;
                }
                if (dot(normal, proj) > 0) return false;
            }
            return true;
        };
        // primitiveMesh::findNearestCell: the first of the nearest centres
        label nearest = 0;
        scalar minProximity = magSqr(g.C()[0] - refPoint);
        for (label celli = 1; celli < m.nCells(); ++celli)
        {
            const scalar proximity = magSqr(g.C()[celli] - refPoint);
            if (proximity < minProximity)
            {
                nearest = celli;
                minProximity = proximity;
            }
        }
        r.pRefCell = -1;
        if (pointInCell(nearest))
        {
            r.pRefCell = nearest;
        }
        else
        {
            for (label celli = 0; celli < m.nCells(); ++celli)
            {
                if (pointInCell(celli))
                {
                    r.pRefCell = celli;
                    break;
                }
            }
        }
        if (r.pRefCell < 0)
        {
            throw std::runtime_error(
                "brae interFoam: PIMPLE's pRefPoint lies in no cell by the face-plane test. OpenFOAM then "
                "searches again with an octree over the cells' tet decomposition (polyMesh::findCell, "
                "CELL_TETS), which is not ported.");
        }
    }
    else
    {
        throw std::runtime_error(
            "brae interFoam: p_rgh fixes its value on no patch, so it needs a reference, and PIMPLE names "
            "neither pRefCell nor pRefPoint. OpenFOAM stops on the same condition (findRefCell.C:91).");
    }
    if (!pim->found("pRefValue"))
    {
        throw std::runtime_error(
            "brae interFoam: p_rgh needs a reference and PIMPLE names no pRefValue. OpenFOAM reads it "
            "with a mandatory lookup (findRefCell.C:100).");
    }
    r.pRefValue = pim->scalarOr("pRefValue", scalar(0));
    return r;
}

// ONE `solver GAMG;` ENTRY. Everything GAMGSolver::readControls and GAMGAgglomeration read from it,
// and a refusal for each control whose branch is not ported -- by name, because every one of them
// changes where the solve stops and none of them changes a converged field.
GamgControls readGamgControls(
    const FoamDict& d,
    scalar tol,
    scalar relTol,
    int maxIter)
{
    const std::string who = "brae interFoam: fvSolution's GAMG entry for p_rgh ";
    auto switchOr = [&](const std::string& key, bool def)
    {
        if (!d.found(key)) return def;
        const std::string w = d.wordOr(key, "");
        if (w == "yes" || w == "true" || w == "on" || w == "y" || w == "t" || w == "1") return true;
        if (w == "no" || w == "false" || w == "off" || w == "n" || w == "f" || w == "0") return false;
        throw std::runtime_error(who + "has `" + key + " " + w + "`, which is not a Switch.");
    };

    GamgControls c;
    c.smoother = d.wordOr("smoother", "");
    c.tolerance = tol;
    c.relTol = relTol;
    c.maxIter = maxIter;
    c.minIter = d.intOr("minIter", 0);
    c.nPreSweeps = d.intOr("nPreSweeps", c.nPreSweeps);
    c.preSweepsLevelMultiplier = d.intOr("preSweepsLevelMultiplier", c.preSweepsLevelMultiplier);
    c.maxPreSweeps = d.intOr("maxPreSweeps", c.maxPreSweeps);
    c.nPostSweeps = d.intOr("nPostSweeps", c.nPostSweeps);
    c.postSweepsLevelMultiplier = d.intOr("postSweepsLevelMultiplier", c.postSweepsLevelMultiplier);
    c.maxPostSweeps = d.intOr("maxPostSweeps", c.maxPostSweeps);
    c.nFinestSweeps = d.intOr("nFinestSweeps", c.nFinestSweeps);
    c.scaleCorrection = switchOr("scaleCorrection", true);
    c.nCellsInCoarsestLevel = static_cast<label>(d.intOr("nCellsInCoarsestLevel", 10));

    if (c.smoother.empty())
    {
        throw std::runtime_error(
            who + "names no `smoother`. lduMatrix::smoother::New reads it with a mandatory lookup and "
            "OpenFOAM stops without it.");
    }
    if (!gamgSmootherPorted(c.smoother))
    {
        throw std::runtime_error(
            who + "asks for `smoother " + c.smoother + "`, which is not ported. brae's GAMG has DIC, "
            "DICGaussSeidel, GaussSeidel and symGaussSeidel (gamg_solver_cpp.cuh).");
    }
    const std::string agglomerator = d.wordOr("agglomerator", "faceAreaPair");
    if (agglomerator != "faceAreaPair")
    {
        throw std::runtime_error(
            who + "asks for `agglomerator " + agglomerator + "`. Only faceAreaPair is ported "
            "(pair_gamg_agglomeration_cpp.cuh); another agglomerator is another hierarchy.");
    }
    if (d.intOr("mergeLevels", 1) != 1)
    {
        throw std::runtime_error(
            who + "asks for `mergeLevels " + std::to_string(d.intOr("mergeLevels", 1)) + "`. "
            "pairGAMGAgglomeration then folds pairs of levels into one (combineLevels, "
            "pairGAMGAgglomerate.C:138), which is not ported.");
    }
    if (d.intOr("updateInterval", 1) != 1)
    {
        throw std::runtime_error(
            who + "sets `updateInterval`. faceAreaPairGAMGAgglomeration then weighs faces by magSf "
            "rather than by the perturbed area vector whenever the time index is not a multiple of it "
            "(faceAreaPairGAMGAgglomeration.C:66-99); only the default's branch is ported.");
    }
    if (!switchOr("cacheAgglomeration", true))
    {
        throw std::runtime_error(
            who + "sets `cacheAgglomeration no`. The hierarchy is then rebuilt at every solve, each "
            "time from wherever the static pairing direction was left; brae builds it once.");
    }
    if (switchOr("interpolateCorrection", false))
    {
        throw std::runtime_error(
            who + "sets `interpolateCorrection yes` (GAMGSolverInterpolate.C), which is not ported.");
    }
    if (switchOr("directSolveCoarsest", false))
    {
        throw std::runtime_error(
            who + "sets `directSolveCoarsest yes`. The coarsest level is then LU-factored rather than "
            "solved by PCG to the entry's tolerance (GAMGSolver.C:303), which is not ported.");
    }
    if (d.subDict("coarsestLevelCorr"))
    {
        throw std::runtime_error(
            who + "carries a `coarsestLevelCorr` sub-dictionary. GAMGSolver then builds the coarsest "
            "level's solver from it (GAMGSolver.C:318) instead of PCG with DIC at the entry's own "
            "tolerance, which is the one brae runs.");
    }
    if (d.found("processorAgglomerator"))
    {
        throw std::runtime_error(
            who + "names a `processorAgglomerator`. This is a serial solver and the entry selects a "
            "different coarse hierarchy in OpenFOAM's parallel runs.");
    }
    return c;
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
        // READ AND THEN NEVER USED: neither driver handed ddtU to the momentum equation, whose
        // input defaults to Euler, so `CrankNicolson 0.5` (RAS/floatingObject) and `localEuler`
        // (RAS/DTCHull) would have run as Euler. Both tutorials were being stopped for other reasons.
        if (f.ddtU != DdtScheme::Euler)
            throw std::runtime_error(
                "brae interFoam: ddtSchemes default is `" + dflt + "`. The momentum equation, the "
                "ddt flux correction in pEqn and the turbulence closure carry Euler only; refusing "
                "rather than running Euler under another scheme's name.");

        f.laplacianScheme = readNonOrthScheme(all, "laplacianSchemes");
        f.snGradScheme = readNonOrthScheme(all, "snGradSchemes");
        refuseUncorrectedOnSkewMesh(f.laplacianScheme, f.snGradScheme, m, g);
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
        // pimpleControl.C:51-52
        const std::string tf = pim->wordOr("turbOnFinalIterOnly", "yes");
        f.pimple.turbOnFinalIterOnly = !(tf == "no" || tf == "false" || tf == "off" || tf == "0");

        // createDyMControls.H: `correctPhi` defaults to mesh.dynamic(), the other two to false
        auto switchOr = [&](const char* key, bool def)
        {
            const std::string w = pim->wordOr(key, def ? "yes" : "no");
            return !(w == "no" || w == "false" || w == "off" || w == "0" || w == "n" || w == "f");
        };
        f.dynamicMesh = DynamicMotionSolverFvMesh::New(caseDir, startDir);
        const bool dynamic = f.dynamicMesh != nullptr;
        f.correctPhi = switchOr("correctPhi", dynamic);
        f.checkMeshCourantNo = switchOr("checkMeshCourantNo", false);
        f.moveMeshOuterCorrectors = switchOr("moveMeshOuterCorrectors", false);
        if (dynamic && f.correctPhi)
        {
            throw std::runtime_error(
                "brae interFoam: the mesh moves and PIMPLE's `correctPhi` is on (it defaults to on for a "
                "moving mesh; the solid-body tutorials write `correctPhi no`). interFoam.C:136-146 then "
                "rebuilds phi from Uf after every mesh update and solves CorrectPhi's pcorr equation "
                "against it, which is not ported. Refused rather than run without the correction.");
        }
    }

    // solvers/<alpha> -- the linear solve of the MULESCorr pre-solve. A case without MULESCorr never
    // solves for alpha and need not name a solver (capillaryRise does not).
    {
        const FoamDict* sv = fvSolution.subDict("solvers");
        const FoamDict* ad = sv ? sv->subDict(f.alphaName) : nullptr;
        if (ad)
        {
            f.aSolve = SmoothLinearSolve::read(*ad);
        }
        // minIter is honoured for k and epsilon and nowhere else yet. Three tutorials name it for
        // alpha (DTCHull, DTCHullMoving, electrostaticDeposition): it forces a sweep on the steps
        // where the pre-solve's initial residual is already under tolerance, which moves alpha.
        if (f.alphaCtl.MULESCorr && f.aSolve.minIter > 0)
            throw std::runtime_error(
                "brae interFoam: `solvers/" + f.alphaName + "` names `minIter "
                + std::to_string(f.aSolve.minIter) + "`, which the alpha pre-solve does not honour "
                "yet. Refused rather than stop a sweep earlier than OpenFOAM does.");
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
            out = SmoothLinearSolve::read(*d);
            if (out.minIter > 0)
                throw std::runtime_error(
                    std::string("brae interFoam: `solvers/") + name + "` names `minIter "
                    + std::to_string(out.minIter) + "`, which the momentum predictor does not honour "
                    "yet. Refused rather than stop a sweep earlier than OpenFOAM does.");
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
            // a `preconditioner { preconditioner GAMG; ... }` sub-dictionary (testTubeMixer and the
            // sloshing tanks, on p_rghFinal and pcorr): lduMatrix::preconditioner::New reads the name
            // from inside it. Named here so the notice below says what the case asked for rather
            // than `preconditioner ;`. The form itself is not ported.
            if (s.preconditioner.empty())
            {
                const FoamDict* pd = d.subDict("preconditioner");
                if (pd)
                {
                    s.preconditioner = "{ " + pd->wordOr("preconditioner", "") + " ... }";
                }
            }
            s.tol = d.scalarOr("tolerance", scalar(1e-6));
            s.relTol = d.scalarOr("relTol", scalar(0));
            s.maxIter = static_cast<int>(d.scalarOr("maxIter", scalar(1000)));
            if (s.gamgSolver())
            {
                s.gamg = readGamgControls(d, s.tol, s.relTol, s.maxIter);
            }
            if (!s.pcgDIC() && !s.gamgSolver())
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
    // THE WAVE CONDITIONS ARE CLAIMED HERE, on this reader's own copy of the file data, and nowhere
    // else: the shared factory refuses both type names, so no other solver can build one frozen.
    FieldData<scalar> alphaData = readField<scalar>(startDir + "/" + f.alphaName);
    FieldData<vector> UData = readField<vector>(startDir + "/U");
    f.waves = readInterWaves(caseDir, startDir, alphaData, UData, patches, f.g, f.alphaName);
    f.alpha1 = buildField<scalar>(alphaData, patches, nC);
    f.U = buildField<vector>(UData, patches, nC);
    f.movingWallVelocityPatch.assign(patches.size(), 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const PatchFieldData<vector>* entry = findPatchEntry(UData, patches[pi]);
        if (entry && entry->type == "movingWallVelocity")
        {
            f.movingWallVelocityPatch[pi] = 1;
        }
    }
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

    // Turbulence. createFields.H:78 constructs it AFTER the mixture, because validate() -- in the one
    // lineage that calls it -- evaluates the nut wall functions with the mixture's nu at the wall.
    f.turbulence = readInterTurbulence(caseDir, startDir, fvSolution, f.ddtU == DdtScheme::Euler,
                                       f.laplacianScheme.corrected, f.laplacianScheme.limitCoeff,
                                       patches, nC);
    if (f.turbulence.on && !f.pimple.turbOnFinalIterOnly && f.pimple.nOuterCorrectors > 1)
        throw std::runtime_error(
            "brae interFoam: `turbOnFinalIterOnly no` with nOuterCorrectors "
            + std::to_string(f.pimple.nOuterCorrectors) + " runs turbulence->correct() more than once "
            "inside a time step. The second call needs k.oldTime() and the non-Final solver entries, "
            "which this port does not carry; no shipped tutorial sets it.");
    validateInterTurbulence(f.turbulence, f.U, f.nu, f.nuBnd, m, g, patches);
    if (f.dynamicMesh && f.turbulence.on)
    {
        throw std::runtime_error(
            "brae interFoam: the mesh moves and the case is turbulent. The closure's wall distance "
            "follows the mesh (wallDist::movePoints) and the closure's own moving-mesh terms are not "
            "ported; refused rather than run the closure on the mesh as it started.");
    }
    if (f.dynamicMesh && f.waves.any)
    {
        throw std::runtime_error(
            "brae interFoam: the mesh moves and a patch carries a wave condition. The wave models take "
            "their geometry once, at construction (waveModel::initialiseGeometry); on a moving patch that "
            "is not the patch the condition is applied to. Not ported; no tutorial combines the two.");
    }

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
    f.pRef = readPressureReference(f.p_rgh, fvSolution, m, g);
    if (f.pRef.needReference)
    {
        // createFields.H:115-124
        applyPressureReference(f.p, f.p_rgh.internal, f.rho, f.gh, f.pRef.pRefCell, f.pRef.pRefValue);
        f.p_rgh.evaluateBoundary();
    }

    // createUfIfPresent.H: the face velocity of a moving mesh, interpolate(U) to begin with. A
    // `Uf` file in the start directory is a restart's, and a restart of a moving mesh is refused
    // where the mesh is read.
    if (f.dynamicMesh)
    {
        std::vector<std::vector<vector>> Ub(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            Ub[pi] = f.U.boundary[pi]->value();
        }
        f.Uf = fvc::interpolate(f.U.internal, Ub, m, g, patches);
    }

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
    // ...and now that rhoPhi exists, the patches that NAME it learn it
    pushFluxToPatches(f, patches);

    return f;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
