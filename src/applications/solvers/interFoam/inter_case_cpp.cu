// interFoam's createFields -- see inter_case_cpp.cuh for the provenance and for the four things the
// order of this file encodes.
#include "inter_case_cpp.cuh"
#include "inter_peqn_cpp.cuh"
#include "brae_notice.cuh"
#include "foam_field_reader.cuh"
#include "mrf_read.cuh"   // readCellZones
#include "read_surface_field.cuh"
#include "scheme_parse.cuh"
#include <filesystem>
#include <map>

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
    pushAlphaToPatches(f, patches);
}

// The two permeable-wall conditions look the PHASE FIELD up, by the name in their `alpha` entry, and read
// its STORED values on their own patch at every updateCoeffs
// (pressurePermeableAlphaInletOutletVelocity...C:160-163, prghPermeableAlphaTotalPressure...C:190-193).
// brae's patches are told, after alpha's last boundary evaluate of the step, so a face whose alpha
// crosses alphaMin switches in the step OpenFOAM switches it. NOT DISCRIMINATED by the gate: no face of
// damBreakPermeable's wall crosses the threshold in either gated run.
void pushAlphaToPatches(
    InterFields& f,
    const std::vector<FvPatch>& patches)
{
    for (std::size_t pi = 0; pi < patches.size() && pi < f.alpha1.boundary.size(); ++pi)
    {
        for (fvPatchField<vector>* ub : {f.U.boundary[pi].get()})
        {
            if (!ub->needsAlphaPatchValues()) continue;
            if (ub->alphaFieldName() != f.alphaName)
                throw std::runtime_error(
                    "brae interFoam: U patch `" + patches[pi].name + "` names `alpha " + ub->alphaFieldName()
                    + "`, and this case's phase field is `" + f.alphaName + "`. OpenFOAM looks the named "
                    "field up and stops without it.");
            ub->updateFromAlphaValues(f.alpha1.boundary[pi]->value());
        }
        for (fvPatchField<scalar>* pb : {f.p_rgh.boundary[pi].get()})
        {
            if (!pb->needsAlphaPatchValues()) continue;
            if (pb->alphaFieldName() != f.alphaName)
                throw std::runtime_error(
                    "brae interFoam: p_rgh patch `" + patches[pi].name + "` names `alpha " + pb->alphaFieldName()
                    + "`, and this case's phase field is `" + f.alphaName + "`. OpenFOAM looks the named "
                    "field up and stops without it.");
            pb->updateFromAlphaValues(f.alpha1.boundary[pi]->value());
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
        if (q.coupled)
        {
            // rho's patch on a cyclic is a cyclic: snGrad(rho) there is deltaCoeffs*(rho_nbr - rho_own)
            r.boundary.push_back(std::make_unique<CoupledCyclicPatchField<scalar>>(q));
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
        if (patches[pi].coupled)
        {
            // ON A COUPLED PATCH the blend is NOT formed from alpha's patch value. OpenFOAM v2412 ends
            // every GeometricField operation with result.correctLocalBoundaryConditions()
            // (GeometricFieldFunctionsM.C; `localConsistency`, on by default, etc/controlDict:225), which
            // re-evaluates a constraint patch of the RESULT: coupledFvPatchField::evaluateLocal is
            // evaluate(), w*pif + (1 - w)*pnf of the result's own cells. For rho, linear in alpha, that
            // is the same number. For nu it is not: measured on RAS/damBreakPorousBaffle with the free
            // surface across the baffle, on the one face whose two cells hold alpha 0.679 and 0.520,
            // nu(alpha_b) against the two cells' nu interpolated is 4.5e-04 of nu -- and
            // porousBafflePressure reads exactly that patch value. It put the jump 2.8e-08 out on that
            // face after 60 steps and U 1.05e-09, where the same case with a plain cyclic held 3e-12;
            // refitting OpenFOAM's own written jump from its own written alpha is exact to 5e-16 with
            // the interpolated cells and 2.8e-08 out without.
            for (std::size_t i = 0; i < n; ++i)
            {
                const label k = static_cast<label>(i);
                f.rhoBnd[pi][i] = coupledLinear(patches[pi], k, f.rho);
                f.muBnd[pi][i] = coupledLinear(patches[pi], k, f.mu);
                f.nuBnd[pi][i] = coupledLinear(patches[pi], k, f.nu);
            }
            continue;
        }
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
    // `Gauss interfaceCompression vanLeer 1` is NOT the PhiScheme `Gauss interfaceCompression`: it is the
    // run-time selectable limited scheme of interfaceCompression.H (interfaceCompressionNew), vanLeer
    // with a compression coefficient. Matching by substring read it as plain vanLeer and ran it without
    // a word -- found while writing a refusal arm for the cyclic baffle. No shipped interFoam tutorial
    // names it; refused, by name.
    {
        const std::size_t at = e.find("interfaceCompression");
        if (at != std::string::npos)
        {
            std::string rest = e.substr(at + std::string("interfaceCompression").size());
            while (!rest.empty() && (rest.back() == ';' || rest.back() == ' ' || rest.back() == '\t')) rest.pop_back();
            while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.erase(rest.begin());
            if (!rest.empty())
                throw std::runtime_error(
                    std::string("brae interFoam: `") + key + " " + entry + "` is interfaceCompression.H's "
                    "limited scheme with a compression coefficient (`" + rest + "`), not the PhiScheme "
                    "`Gauss interfaceCompression`. It is not ported.");
        }
    }
    if (e.find("vanLeer") != std::string::npos)   return AlphaFluxScheme::vanLeer;
    if (e.find("upwind")  != std::string::npos)   return AlphaFluxScheme::upwind;
    if (e.find("interfaceCompression") != std::string::npos)
        return AlphaFluxScheme::interfaceCompression;   // the host's; the device refuses it
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

    // constant/MRFProperties is read in buildInterFields (createMRF.H), which ports it and refuses by
    // name what no gate holds: MRF under a moving mesh, under RAS, and beside a fixedFluxPressure patch.

    // fvOptions are read in buildInterFields (createFvOptions.H), which ports explicitPorositySource /
    // DarcyForchheimer and refuses every other active option by name.

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


// interFoam's pressure laplacian, its three snGrads and the momentum laplacian take the case's
// `corrected` and `limited` schemes now (fvm::laplacian's corrected form, fvc::snGrad's). What is NOT
// theirs is `uncorrected`: OpenFOAM's uncorrectedSnGrad takes nonOrthDeltaCoeffs where orthogonalSnGrad
// takes deltaCoeffs (uncorrectedSnGrad.H:91, orthogonalSnGrad.H:91), and brae's orthogonal assembly
// takes the latter -- the same number on a mesh of rectangles, where the two coefficients coincide.
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
    const bool uncorrected =
        schemeHasWord(laplacian.raw, "uncorrected") || schemeHasWord(snGrad.raw, "uncorrected");
    if (!uncorrected) return;
    const scalar worst = maxNonOrthogonality(m, g);
    // round-off on a mesh of rectangles is 1e-16; one degree is 1.5e-04
    if (worst < scalar(1e-10)) return;
    const scalar degrees = std::acos(scalar(1) - worst) * scalar(180) / scalar(3.14159265358979323846);
    throw std::runtime_error(
        "brae interFoam: fvSchemes asks for `uncorrected` (laplacianSchemes `" + laplacian.raw
        + "`, snGradSchemes `" + snGrad.raw + "`) and the mesh is non-orthogonal by up to "
        + std::to_string(degrees) + " degrees. OpenFOAM's uncorrectedSnGrad divides by the "
        "non-orthogonal delta coefficient where brae's orthogonal assembly divides by |d|; the two "
        "differ on this mesh. Refused rather than run `orthogonal` under the name `uncorrected`.");
}

// gradSchemes. interFoam's gradients -- grad(U) for the momentum scheme and the viscous term's
// explicit half, grad(p_rgh), grad(rho) and grad(alpha) for the non-orthogonal corrections, and
// grad(alpha) for the interface normal -- are all Gauss linear and unlimited here. 40 of the 44
// tutorials write `default Gauss linear;` and nothing else; the four that name a cellLimited or a
// leastSquares gradient are refused, because a gradient the case limits and brae does not is a
// different discretisation that converges. This used to read nothing.
void refuseUnportedGradSchemes(const std::string& fvSchemesText)
{
    const std::string blk = fvSchemesBlock(fvSchemesText, "gradSchemes");
    if (blk.empty())
        throw std::runtime_error("brae interFoam: fvSchemes has no gradSchemes block.");
    // every `key scheme;` line of the block
    std::size_t at = 0;
    while (true)
    {
        const std::size_t e = blk.find(';', at);
        if (e == std::string::npos) break;
        std::string line = blk.substr(at, e - at);
        at = e + 1;
        const std::size_t b = line.find_first_not_of(" \t\n\r{");
        if (b == std::string::npos) continue;
        line = line.substr(b);
        const std::size_t sp = line.find_first_of(" \t");
        if (sp == std::string::npos) continue;
        const std::string key = line.substr(0, sp);
        std::string scheme = line.substr(sp);
        const std::size_t sb = scheme.find_first_not_of(" \t\n\r");
        scheme = (sb == std::string::npos) ? std::string() : scheme.substr(sb);
        while (!scheme.empty() && std::isspace(static_cast<unsigned char>(scheme.back())))
        {
            scheme.pop_back();
        }
        std::string collapsed;
        for (char ch : scheme)
        {
            if (std::isspace(static_cast<unsigned char>(ch)))
            {
                if (!collapsed.empty() && collapsed.back() != ' ')
                {
                    collapsed += ' ';
                }
            }
            else
            {
                collapsed += ch;
            }
        }
        if (collapsed != "Gauss linear")
        {
            throw std::runtime_error(
                "brae interFoam: fvSchemes gradSchemes `" + key + " " + collapsed + "` is not ported. "
                "Every gradient this solver takes -- grad(U), grad(p_rgh), grad(rho), grad(alpha) -- is "
                "Gauss linear and unlimited; a limited or least-squares gradient is a different "
                "discretisation. 40 of the 44 shipped tutorials write `default Gauss linear;` alone.");
        }
    }
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
        refuseUnportedGradSchemes(all);
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
        // initCorrectPhi.H: under correctPhi, rAU is a field kept across steps, READ_IF_PRESENT and
        // 1 when absent; pEqn.H:4 then assigns 1/UEqn.A() into it and never clears it, so every mesh
        // update's CorrectPhi interpolates the LAST corrector's rAU
        f.rAU.assign(static_cast<std::size_t>(m.nCells()), scalar(1));
        if (f.correctPhi
            && (std::filesystem::exists(startDir + "/rAU") || std::filesystem::exists(startDir + "/rAU.gz")))
        {
            throw std::runtime_error(
                "brae interFoam: " + startDir + "/rAU exists. initCorrectPhi.H reads it (READ_IF_PRESENT) "
                "as the rAU the first CorrectPhi interpolates; brae starts from 1, which is what "
                "OpenFOAM does only without the file.");
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
        auto readSolve = [&](
            const FoamDict& d,
            const std::string& field)
        {
            InterFields::PressureLinearSolve s;
            s.solver = d.wordOr("solver", "");
            s.preconditioner = d.wordOr("preconditioner", "");
            s.tol = d.scalarOr("tolerance", scalar(1e-6));
            s.relTol = d.scalarOr("relTol", scalar(0));
            s.maxIter = static_cast<int>(d.scalarOr("maxIter", scalar(1000)));
            // a `preconditioner { preconditioner <name>; ... }` sub-dictionary: lduMatrix::
            // preconditioner::New reads the name from inside it and hands the preconditioner THAT
            // dictionary as its controls (lduMatrixPreconditioner.C). A GAMG one is a GAMGSolver
            // reading its tolerance, relTol, smoother and sweeps from there -- not from the PCG entry.
            if (s.preconditioner.empty())
            {
                const FoamDict* pd = d.subDict("preconditioner");
                if (pd)
                {
                    const std::string name = pd->wordOr("preconditioner", "");
                    if (name == "DIC")
                    {
                        s.preconditioner = "DIC";
                    }
                    else if (name == "GAMG")
                    {
                        s.preconditioner = "{ GAMG ... }";
                        s.gamgPreconditioned = true;
                        // lduMatrix::solver::readControls on the sub-dictionary: its own defaults
                        s.gamgPrecond.gamg = readGamgControls(
                            *pd,
                            pd->scalarOr("tolerance", scalar(1e-6)),
                            pd->scalarOr("relTol", scalar(0)),
                            static_cast<int>(pd->scalarOr("maxIter", scalar(1000))),
                            "brae interFoam: fvSolution's GAMG preconditioner for " + field + " ");
                        s.gamgPrecond.nVcycles = pd->intOr("nVcycles", 2);
                    }
                    else
                    {
                        s.preconditioner = "{ " + name + " ... }";
                    }
                }
            }
            if (s.gamgSolver())
            {
                s.gamg = readGamgControls(
                    d,
                    s.tol,
                    s.relTol,
                    s.maxIter,
                    "brae interFoam: fvSolution's GAMG entry for " + field + " ");
            }
            return s;
        };
        auto noticeUnported = [&](const InterFields::PressureLinearSolve& s)
        {
            if (!s.pcgDIC() && !s.gamgSolver() && !s.pcgGamg())
                noticeApproximated("interFoam p_rgh solve",
                    "the case asks for `solver " + s.solver + "; preconditioner " +
                    s.preconditioner + ";` and brae runs PBiCGStab at the same tolerance. Only "
                    "PCG with DIC is OpenFOAM's own solver here; the difference is where the solve "
                    "stops, which on a VoF case is visible as alpha's over-1 excursion.");
        };
        f.pSolve = readSolve(*pr, "p_rgh");
        noticeUnported(f.pSolve);
        if (!pf)
            throw std::runtime_error(
                "brae interFoam: fvSolution has `solvers/p_rgh` but no `p_rghFinal`. pEqn.H solves "
                "the last corrector with p_rgh.select(finalInnerIter()), which OpenFOAM resolves to "
                "the Final entry and refuses to run without; guessing it would pick the last "
                "corrector's stopping point for the case.");
        f.pSolveFinal = readSolve(*pf, "p_rghFinal");
        noticeUnported(f.pSolveFinal);

        // solvers/pcorr and pcorrFinal: CorrectPhi's solve, which initCorrectPhi.H runs at the start of
        // EVERY case -- moving or not, correctPhi or not -- and interFoam.C:139 after every mesh update
        // under `correctPhi`. pcorr.select(finalNonOrthogonalIter()) takes pcorrFinal on the last
        // non-orthogonal pass and pcorr on the others, and OpenFOAM stops without the one it asks for.
        // No approximation here: every tutorial names PCG with DIC, PCG with a GAMG preconditioner, or
        // GAMG, and a zero right-hand side -- the start of a case at rest -- is exact under all three.
        auto readPcorr = [&](
            const std::string& field,
            InterFields::PressureLinearSolve& out)
        {
            const FoamDict* d = sv->subDict(field);
            if (!d)
            {
                throw std::runtime_error(
                    "brae interFoam: fvSolution has no `solvers/" + field + "` entry. CorrectPhi "
                    "(CorrectPhi.C:111) solves pcorr with it -- at the start of every case, from "
                    "initCorrectPhi.H -- and OpenFOAM stops without it.");
            }
            out = readSolve(*d, field);
            if (!out.pcgDIC() && !out.gamgSolver() && !out.pcgGamg())
            {
                throw std::runtime_error(
                    "brae interFoam: fvSolution solves " + field + " with `solver " + out.solver +
                    "; preconditioner " + out.preconditioner + ";`. Only PCG with DIC, PCG with a "
                    "GAMG preconditioner and GAMG are ported for CorrectPhi's pcorr.");
            }
        };
        readPcorr("pcorrFinal", f.pcorrSolveFinal);
        if (f.nNonOrthogonalCorrectors > 0)
        {
            readPcorr("pcorr", f.pcorrSolve);
        }
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
                                       patches, nC, &m, &g);
    if (f.turbulence.on && !f.pimple.turbOnFinalIterOnly && f.pimple.nOuterCorrectors > 1)
        throw std::runtime_error(
            "brae interFoam: `turbOnFinalIterOnly no` with nOuterCorrectors "
            + std::to_string(f.pimple.nOuterCorrectors) + " runs turbulence->correct() more than once "
            "inside a time step. The second call needs k.oldTime() and the non-Final solver entries, "
            "which this port does not carry; no shipped tutorial sets it.");
    validateInterTurbulence(f.turbulence, f.U, f.nu, f.nuBnd, m, g, patches);

    // createMRF.H -> IOMRFZoneList (READ_IF_PRESENT); a zone is active unless it says otherwise
    // (MRFZone.C:248, :553). createFields.H:129 constructs it AFTER everything above, and nothing here
    // applies it: MRF.correctBoundaryVelocity(U) is UEqn.H's first line, not createFields'.
    {
        const std::vector<MRF::ZoneSpec> specs = MRF::readMRFProperties(caseDir + "/constant");
        if (!specs.empty())
        {
            const std::map<std::string, std::vector<label>> zoneMap =
                readCellZones(caseDir + "/constant/polyMesh");
            for (const MRF::ZoneSpec& sp : specs)
            {
                const auto it = zoneMap.find(sp.cellZone);
                if (it == zoneMap.end())
                    throw std::runtime_error(
                        "brae interFoam: MRF cellZone `" + sp.cellZone + "` is not in "
                        "constant/polyMesh/cellZones. OpenFOAM stops on this (MRFZone.C:258-266).");
                f.mrfZones.push_back(MRF::buildZone(sp, it->second, m, patches));
            }
        }
    }
    // createFvOptions.H -> fv::options (constant/ first, then system/; an option is active unless it
    // says otherwise, fvOption.C:72). interFoam reaches the list in four places: UEqn.H:9
    // `== fvOptions(rho, U)`, :14 constrain(UEqn), :31 and pEqn.H:65 correct(U). A source reaches the
    // first alone, and ONE source is ported here.
    f.fvOptions = fvOptions::read(caseDir, m);
    for (const fvOptions::Option& o : f.fvOptions.options)
    {
        if (!o.active) continue;
        const bool darcyForchheimer = o.unsupported.empty() && !o.rotorDisk && !o.actuationDisk
                                   && !o.fixedCoeff && o.constraint == fvOptions::Option::Constraint::none;
        if (darcyForchheimer) continue;
        const std::string what = !o.unsupported.empty() ? o.unsupported
                               : o.fixedCoeff ? std::string("explicitPorositySource with the fixedCoeff model")
                               : o.type;
        throw std::runtime_error(
            "brae interFoam: fvOptions has an active option `" + o.name + "` (" + what + "). interFoam "
            "applies fvOptions to UEqn as `== fvOptions(rho, U)`, constrains the matrix with them and "
            "corrects U after every corrector; brae's interFoam carries explicitPorositySource with "
            "DarcyForchheimer -- RAS/angledDuct's, gated against OpenFOAM -- and nothing else.");
    }
    bool anyFvOption = false;
    for (const fvOptions::Option& o : f.fvOptions.options)
    {
        anyFvOption = anyFvOption || o.active;
    }
    if (anyFvOption && f.dynamicMesh)
        throw std::runtime_error(
            "brae interFoam: the case has an active fvOption AND a moving mesh. The option's cell "
            "selection and its resistance tensor are taken once; no shipped tutorial pairs them and no "
            "gate holds it.");
    if (anyFvOption && !f.mrfZones.empty())
        throw std::runtime_error(
            "brae interFoam: the case has an active fvOption AND an active MRF zone. Each is gated on "
            "its own tutorial and nothing holds the two together against OpenFOAM.");
    if (!f.mrfZones.empty() && f.dynamicMesh)
        throw std::runtime_error(
            "brae interFoam: the case has an active MRF zone AND a moving mesh. MRF.update() rebuilds "
            "the zone's face lists when the topology changes and makeRelative composes with the mesh "
            "flux (interFoam.C:128, pEqn.H:19-24); no shipped tutorial pairs them and no gate holds it.");
    if (!f.mrfZones.empty() && f.turbulence.on)
        throw std::runtime_error(
            "brae interFoam: the case has an active MRF zone AND is turbulent. The one shipped MRF "
            "tutorial, laminar/mixerVessel2D, is laminar, so no gate holds the closure in a rotating "
            "frame against OpenFOAM. Refused rather than run ungated.");
    for (std::size_t pi = 0; pi < patches.size() && !f.mrfZones.empty(); ++pi)
    {
        if (!f.p_rgh.boundary[pi]->updateableSnGrad()) continue;
        throw std::runtime_error(
            "brae interFoam: the case has an active MRF zone AND p_rgh patch `" + patches[pi].name
            + "` is a fixedFluxPressure. constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF) subtracts "
            "MRF.relative(Sf & U_b) there (constrainPressureI.H), and this port's subtracts Sf & U_b; "
            "mixerVessel2D's walls are zeroGradient, so nothing gates the difference.");
    }
    if (f.dynamicMesh && f.turbulence.on)
    {
        throw std::runtime_error(
            "brae interFoam: the mesh moves and the case is turbulent. The closure's wall distance "
            "follows the mesh (wallDist::movePoints) and the closure's own moving-mesh terms are not "
            "ported; refused rather than run the closure on the mesh as it started.");
    }
    // A WAVE CONDITION ON A MOVING MESH IS NOT REFUSED. OpenFOAM's wave models take their geometry once,
    // at construction (waveModel::initialiseGeometry: the patch's orientation, each face's height and
    // paddle), and read only its magSf and its face cells' alpha as the run goes (waveModel::waterLevel);
    // brae's WaveModel does the same, through the patch the mesh update rebuilds in place. The five
    // waveMaker tutorials absorb at a wall whose points the motion pins, where the two are one geometry.

    // --- coupled patches ----------------------------------------------------------------------
    // A cyclic is a real coupled patch here ONLY when the caller attached its coupling to the mesh patch
    // (attachCyclicCoupling): then the shared factory built a CoupledCyclicPatchField and every operator
    // on this path branches on FvPatch::coupled. Without it the factory's placeholder is a zeroGradient
    // and the pair would run as two walls, silently -- so that is refused, and so is every coupled
    // type the operators do not carry, and every pairing of a cyclic with a part of interFoam that was
    // never run across one.
    {
        const FvPatch* firstCoupled = nullptr;
        for (const FvPatch& q : patches)
        {
            if (isCoupledInterfaceType(q.type) && !q.coupled)
            {
                throw std::runtime_error(
                    "brae interFoam: patch `" + q.name + "` is " + q.type + " and its coupling is not "
                    "attached. The host loop couples a translational `cyclic` (a baffle pair included) "
                    "once attachCyclicCoupling() has filled the mesh patch; cyclicAMI, cyclicACMI and "
                    "processor patches are not ported here. Refused rather than run as two walls.");
            }
            if (q.coupled && !firstCoupled)
            {
                firstCoupled = &q;
            }
        }
        // A JUMP IS THE OWNER'S: fixedJumpFvPatchField::jump() on the other side returns the owner's, so
        // the file's `jump` on the owner is handed across before anything reads it
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].coupled || !patches[pi].owner) continue;
            if (const std::vector<scalar>* j = f.p_rgh.boundary[pi]->coupledJump())
            {
                f.p_rgh.boundary[static_cast<std::size_t>(patches[pi].nbrPatch)]->setOwnerJump(*j);
            }
        }
        f.p_rgh.evaluateBoundary();
        if (firstCoupled)
        {
            const std::string who = "brae interFoam: the case has the cyclic patch `" + firstCoupled->name + "` AND ";
            if (f.dynamicMesh)
                throw std::runtime_error(who + "a moving mesh. The pair's weights and deltas are taken once.");
            if (!f.mrfZones.empty())
                throw std::runtime_error(who + "an active MRF zone. MRF's face lists do not carry coupled faces here.");
            if (anyFvOption)
                throw std::runtime_error(who + "an active fvOption. Nothing holds the two together against OpenFOAM.");
            if (f.waves.any)
                throw std::runtime_error(who + "a wave condition. Nothing holds the two together against OpenFOAM.");
            if (f.turbulence.on && f.turbulence.model != InterRasModel::KEpsilon)
                throw std::runtime_error(who + "a RAS model other than kEpsilon, the one closure carried across a cyclic.");
            // gaussLaplacianScheme adds the deferred non-orthogonal correction on a coupled face too
            // (its correction vectors are not zero there); fvm::laplacian here does not assemble it
            for (const FvPatch& q : patches)
            {
                for (std::size_t i = 0; q.coupled && i < q.nonOrthCorrectionVectors.size(); ++i)
                {
                    if (mag(q.nonOrthCorrectionVectors[i]) > scalar(1e-10))
                        throw std::runtime_error(
                            "brae interFoam: cyclic patch `" + q.name + "` is non-orthogonal (|nf - delta*"
                            "nonOrthDeltaCoeffs| = " + std::to_string((double)mag(q.nonOrthCorrectionVectors[i]))
                            + " on face " + std::to_string(i) + "). The deferred correction of a laplacian on "
                            "a coupled face is not assembled here.");
                }
            }
        }
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
