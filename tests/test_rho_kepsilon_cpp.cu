// The COMPRESSIBLE kEpsilon closure against real OpenFOAM's own turbulence->correct().
//
// THE ORACLE is tools/dumpPEqn, which brackets `turbulence->correct()` and writes every field it READS
// (k, epsilon, nut, alphat, U, phi, rho, mu) and every field it WRITES (k, epsilon, nut, alphat). That
// treats the model as a black box on purpose: the k and epsilon matrices are assembled inside OpenFOAM's
// turbulence library and never surface in the solver, so there is nothing to instrument there without
// patching the library itself. Inputs-in, outputs-out is enough to gate a closure, and it is exactly what
// a closure IS.
//
// WHAT MAKES THIS THE COMPRESSIBLE ONE. OpenFOAM has a single templated kEpsilon.C; the compressible
// instantiation supplies alpha = 1, rho = the density field, alphaRhoPhi = the MASS flux, and a nu that
// varies with temperature:
//
//     epsEqn:  div(alphaRhoPhi, eps) - laplacian(alpha*rho*DepsilonEff, eps)
//           == C1*alpha*rho*GbyNu*Cmu*k - SuSp(((2/3)C1 - C3)*alpha*rho*divU, eps)
//                                       - Sp(C2*alpha*rho*eps/k, eps)
//
// TWO FLUXES. The div operator takes the MASS flux while divU takes the VOLUMETRIC one -- OpenFOAM's
// compressibleTurbulenceModel::phi() returns phi/fvc::interpolate(rho) when the stored flux has mass
// dimensions. In the incompressible lineage the two are the same field. THE CONTROL below feeds the mass
// flux to divU on purpose and requires the answer to change, because that substitution is invisible in
// any incompressible test and is the single easiest thing to get wrong here.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "kEpsilon_cpp.cuh"
#include "near_wall_dist.cuh"
#include "nut_wall_function.cuh"
#include "fvOptions_cpp.cuh"   // the case's fvOptions: kEpsilon.C constrains BOTH equations
#include "linearViscousStress_cpp.cuh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static int failures = 0;

static void report(const std::string& what, double got, double bound)
{
    const bool ok = got < bound;
    if (!ok) ++failures;
    std::printf("     %-40s %.6e   %s\n", what.c_str(), got, ok ? "ok" : "FAIL");
}

static void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-40s %s\n", what.c_str(), ok ? "ok" : "FAIL");
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

// A raw reader for the tensor dump. brae's field reader has no tensor instantiation, and this comparison
// is not a reason to add one to shared code -- the file is ASCII and its internalField is a flat list of
// 9-component entries, so the numbers can be taken directly.
static std::vector<tensor> readTensorInternal(const std::string& path, label nC)
{
    std::ifstream in(path.c_str());
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::size_t at = text.find("internalField");
    if (at == std::string::npos) throw std::runtime_error("no internalField in " + path);
    const bool uniform = text.compare(at + 13, 12, " uniform") > 0
                      && text.find("uniform", at) < text.find("nonuniform", at);
    std::size_t p = text.find('(', at);
    if (p == std::string::npos) throw std::runtime_error("no list in " + path);

    std::vector<scalar> flat;
    const std::size_t want = uniform ? 9 : std::size_t(9) * std::size_t(nC);
    for (std::size_t i = p; i < text.size() && flat.size() < want; )
    {
        const char c = text[i];
        if (c == '(' || c == ')' || std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        char* end = nullptr;
        flat.push_back(std::strtod(text.c_str() + i, &end));
        if (end == text.c_str() + i) break;
        i = std::size_t(end - text.c_str());
    }
    if (flat.size() < want) throw std::runtime_error("short tensor list in " + path);

    std::vector<tensor> out(nC);
    for (label c = 0; c < nC; ++c)
    {
        const std::size_t base = uniform ? 0 : std::size_t(9) * std::size_t(c);
        for (int k = 0; k < 9; ++k) reinterpret_cast<scalar*>(&out[c])[k] = flat[base + std::size_t(k)];
    }
    return out;
}

template<class T>
static std::vector<T> rawInternal(const FieldData<T>& fd, label nC)
{
    if (fd.internalUniform) return std::vector<T>(nC, fd.internalUniformValue);
    return fd.internalField;
}

template <typename T>
static std::vector<std::vector<T>> rawBoundary(const FieldData<T>& fd, const std::vector<FvPatch>& patches)
{
    std::vector<std::vector<T>> out(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        out[pi].assign(patches[pi].size, T{});
        for (const auto& b : fd.boundary)
        {
            if (b.name != patches[pi].name) continue;
            if (b.valueUniform) out[pi].assign(patches[pi].size, b.uniformValue);
            else if (static_cast<label>(b.values.size()) == patches[pi].size) out[pi] = b.values;
            break;
        }
    }
    return out;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <dumpTime>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string dumpT   = argv[2];
    const std::string D       = caseDir + "/" + dumpT + "/";

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    std::printf("compressible kEpsilon vs OpenFOAM (%d cells)\n", (int)nC);

    // Everything OpenFOAM's correct() read, taken from OpenFOAM itself so a disagreement can only be the
    // closure and not the state it was handed.
    GeometricField<scalar> k   = buildField<scalar>(readField<scalar>(D + "stage_kIn"),   patches, nC);
    GeometricField<scalar> eps = buildField<scalar>(readField<scalar>(D + "stage_epsIn"), patches, nC);
    GeometricField<scalar> nut = buildField<scalar>(readField<scalar>(D + "stage_nutIn"), patches, nC);
    GeometricField<vector> U   = buildField<vector>(readField<vector>(D + "stage_Uturb"), patches, nC);
    k.evaluateBoundary(); eps.evaluateBoundary(); nut.evaluateBoundary();

    const FieldData<scalar> rhoFd = readField<scalar>(D + "stage_rhoTurb");
    const std::vector<scalar>              rho    = rawInternal(rhoFd, nC);
    const std::vector<std::vector<scalar>> rhoBnd = rawBoundary<scalar>(rhoFd, patches);
    const FieldData<scalar> muFd = readField<scalar>(D + "stage_muTurb");
    const std::vector<scalar>              mu     = rawInternal(muFd, nC);
    const std::vector<std::vector<scalar>> muBnd  = rawBoundary<scalar>(muFd, patches);

    const FieldData<scalar> phiFd = readField<scalar>(D + "stage_phiTurb");
    SurfaceScalarField phi;
    phi.internal = phiFd.internalField;
    phi.boundary = rawBoundary<scalar>(phiFd, patches);

    // U's flux-conditional boundaries are evaluated only ONCE the flux is in them -- inletOutlet switches
    // on sign(phi), and OpenFOAM had already done this before its turbulence model ran. The driver does
    // exactly this at the top of every iteration; without it the outlet is evaluated against a seeded
    // flux and fvc::grad(U) reads a different face value there, which is what the gradU split above found.
    for (std::size_t pi = 0; pi < patches.size(); ++pi) U.boundary[pi]->updateFromFlux(phi.boundary[pi]);
    U.evaluateBoundary();

    // The boundary values of U are what fvc::grad reads on a boundary face, so before anything downstream
    // is compared, check that brae's evaluation of them reproduces the ones OpenFOAM had. The staged file
    // carries OpenFOAM's own, patch by patch.
    {
        const FieldData<vector> uFd = readField<vector>(D + "stage_Uturb");
        const std::vector<std::vector<vector>> ofUb = rawBoundary<vector>(uFd, patches);
        auto declared = [&](const std::string& nm) -> std::string
        {
            for (const auto& b : uFd.boundary) if (b.name == nm) return b.type;
            return "?";
        };
        // The turbulence fields' declared boundary types matter as much as their values: the assembled
        // matrix takes its boundaryCoeffs from them, and a field read back as `calculated` would carry
        // the right numbers into a system with no boundary contribution at all.
        for (const char* f : {"stage_epsIn", "stage_kIn", "stage_nutIn"})
        {
            const FieldData<scalar> fd = readField<scalar>(D + f);
            std::printf("  1a0. %-14s boundary types:", f);
            for (const auto& b : fd.boundary) std::printf(" %s=%s", b.name.c_str(), b.type.c_str());
            std::printf("\n");
        }
        // The same for the turbulence fields, whose boundaries feed the systems' boundaryCoeffs. The
        // flux is pushed here as correct() will push it, so what is compared is the state the equations
        // are actually assembled from. epsilon's wall is expected to differ: OpenFOAM's epsilonWallFunction
        // derives from fixedValue and carries the wall value, brae's rows are overwritten by setValues.
        for (auto* pr : {&k, &eps})
        {
            const char* nm = (pr == &k) ? "stage_kIn" : "stage_epsIn";
            const FieldData<scalar> fd = readField<scalar>(D + nm);
            const std::vector<std::vector<scalar>> ofb = rawBoundary<scalar>(fd, patches);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                pr->boundary[pi]->updateFromFlux(phi.boundary[pi]);
            pr->evaluateBoundary();
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                std::vector<scalar> a = pr->boundary[pi]->value(), b = ofb[pi];
                std::printf("     %-12s %-28s %.6e\n", nm, patches[pi].name.c_str(), relL2(a, b));
            }
        }
        std::printf("  1a. U on the boundary, patch by patch, against OpenFOAM's own\n");
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            std::vector<scalar> a, b;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                for (int k2 = 0; k2 < 3; ++k2)
                {
                    a.push_back(reinterpret_cast<const scalar*>(&U.boundary[pi]->value()[i])[k2]);
                    b.push_back(reinterpret_cast<const scalar*>(&ofUb[pi][i])[k2]);
                }
            }
            std::printf("     %-28s %-16s %.6e   %s\n", patches[pi].name.c_str(),
                        declared(patches[pi].name).c_str(), relL2(a, b),
                        relL2(a, b) < 1e-10 ? "ok" : "MISMATCH");
        }
    }


    // nu = mu/rho, the LAMINAR kinematic viscosity, per cell and per boundary face.
    std::vector<scalar> nu(nC);
    for (label c = 0; c < nC; ++c) nu[c] = mu[c] / rho[c];
    std::vector<std::vector<scalar>> nuBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        nuBnd[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i) nuBnd[pi][i] = muBnd[pi][i] / rhoBnd[pi][i];
    }

    // compressibleTurbulenceModel::phi() -- the VOLUMETRIC flux.
    SurfaceScalarField phiByRho = phi;
    {
        const SurfaceScalarField rhof = cpu::effectiveFaceViscosity(rho, rhoBnd, m, g, patches);
        for (std::size_t f = 0; f < phiByRho.internal.size(); ++f)
            phiByRho.internal[f] /= rhof.internal[f];
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
                phiByRho.boundary[pi][i] /= rhof.boundary[pi][i];
    }

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;
    const scalar relaxK   = re ? re->scalarOr("k", 1.0) : 1.0;
    const scalar relaxEps = re ? re->scalarOr("epsilon", 1.0) : 1.0;

    std::vector<scalar> alphat(nC, 0.0);

    scalar tolOverride = 1e-12;
    // THE CASE'S fvOptions. kEpsilon.C calls fvOptions.constrain() on BOTH the k and the epsilon
    // equation, and a constraint is not a source: OpenFOAM applies it as setValues(cells, values), which
    // forces the value in those cells AND ZEROES THE OFF-DIAGONAL COUPLING to their neighbours -- one
    // side per face, so upper and lower come out wrong by DIFFERENT amounts. This gate did not pass them,
    // so on angledDuct it compared brae's UNCONSTRAINED matrix against OpenFOAM's constrained one and
    // read `epsilon upper 8.23e-02, lower 6.44e-01` with every individual term exact (`div upper/lower`
    // 0.0, `lapl upper/lower` 2.25e-15) and every mesh factor exact. The asymmetry was the tell.
    cpu::fvOptions::OptionList keOpts = cpu::fvOptions::read(caseDir, m);
    if (!keOpts.firstUnsupported().empty())
    {
        std::printf("  REFUSED: fvOptions declares '%s', which is not ported -- this gate would compare\n"
                    "           a differently-constrained system.\n", keOpts.firstUnsupported().c_str());
        return 1;
    }
    if (!keOpts.empty())
    {
        std::printf("  fvOptions: %zu option(s), constraining k/epsilon as kEpsilon.C does\n",
                    keOpts.options.size());
        for (const auto& o : keOpts.options)
            for (const auto& fv : o.fieldValues)
                std::printf("     %-22s constrains %-8s = %g on %zu cells\n",
                            o.name.c_str(), fv.first.c_str(), (double)fv.second, o.cells.size());
    }

    auto runDrop = [&](bool useVolumetricFluxForDivU, int dropTerm,
                   std::vector<scalar>& kOut,
                   std::vector<scalar>& epsOut,
                   std::vector<scalar>& nutOut,
                   std::vector<scalar>& alphatOut)
    {
        GeometricField<scalar> kk   = buildField<scalar>(readField<scalar>(D + "stage_kIn"),   patches, nC);
        GeometricField<scalar> ee   = buildField<scalar>(readField<scalar>(D + "stage_epsIn"), patches, nC);
        GeometricField<scalar> nn   = buildField<scalar>(readField<scalar>(D + "stage_nutIn"), patches, nC);
        // NOT nn.evaluateBoundary(): nut's wall patches carry nutkWallFunction, which brae maps to a
        // zeroGradient-shaped patch field -- so re-evaluating would replace the wall values OpenFOAM
        // computed with the adjacent cell's, and those wall values are exactly what DepsilonEff reads on
        // a boundary face. The staged field already holds OpenFOAM's.
        kk.evaluateBoundary(); ee.evaluateBoundary();
        std::vector<scalar> at(nC, 0.0);

        cpu::kEpsilonRef::Compressible comp;
        comp.rho      = &rho;
        comp.rhoBnd   = &rhoBnd;
        comp.nu       = &nu;
        comp.nuBnd    = &nuBnd;
        comp.phiByRho = useVolumetricFluxForDivU ? &phiByRho : &phi;
        comp.alphat   = &at;
        comp.Prt      = 1.0;

        // sbMatched's fvSchemes says `laplacianSchemes { default Gauss linear corrected; }`, and that
        // governs the turbulence diffusion terms as much as the momentum ones.
        KEpsilonCoeffs keco;
        keco.correctedLaplacian = true;

        cpu::kEpsilonRef::KEResiduals res;
        res.captureStages = true;   // the intermediates, which the solver does not ask for
        cpu::kEpsilonRef::correct(U, kk, ee, nn, phi, /*nu=*/0.0, m, g, patches,
                                  relaxEps, relaxK, tolOverride, 0.0, 20000, keco, &res,
                                  /*bounded=*/true, dropTerm, &comp, keOpts.empty() ? nullptr : &keOpts);
        kOut = kk.internal; epsOut = ee.internal; nutOut = nn.internal; alphatOut = at;
        return res;
    };

    // ---- 1. The closure, on OpenFOAM's own inputs. ----
    std::printf("  1. one turbulence->correct()\n");
    std::vector<scalar> kO, eO, nO, aO;
    auto run = [&](bool vol, std::vector<scalar>& a, std::vector<scalar>& b,
                   std::vector<scalar>& c, std::vector<scalar>& d)
    { return runDrop(vol, 0, a, b, c, d); };
    const cpu::kEpsilonRef::KEResiduals res = run(true, kO, eO, nO, aO);
    std::printf("     initial residuals: epsilon %.4e   k %.4e   (wall cells %d)\n",
                (double)res.epsilon, (double)res.k, (int)res.wallCells);

    // ---- THE TERM-BY-TERM COMPARISON, against OpenFOAM's OWN internals. tools/dumpKEpsilon is an
    //      instrumented copy of OpenFOAM's kEpsilon -- its equations untouched, writes added -- so divU,
    //      GbyNu, G and both assembled systems have an oracle rather than an argument. This is what turns
    //      "epsilon is 7.4e-02 out in the interior" into a named term.
    std::printf("  1b. the intermediates, against OpenFOAM's own\n");
    {
        // OpenFOAM's epsilonWallFunction derives from fixedValue, so its rows carry that patch's
        // coefficients; brae's derives from zeroGradient and its rows are written by setValues instead.
        // Both end at the same solved epsilon -- the wall figure below is 5.4e-15 -- but the ASSEMBLED
        // rows differ, so the system comparisons are asserted off the wall and the wall cells reported
        // beside them. Asserting the union would be asserting a difference that is not an error;
        // dropping the wall cells silently would be hiding one that might be.
        std::vector<char> isWall(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            if (eps.boundary[pi]->isTurbulenceWallFunction())
                for (label i = 0; i < patches[pi].size; ++i) isWall[patches[pi].faceCells[i]] = 1;

        auto cmp = [&](const char* file, const std::vector<scalar>& mine, const char* what, double bound,
                       bool offWallOnly = false)
        {
            const std::string path = D + file;
            if (!std::ifstream(path.c_str()).good())
            {
                std::printf("     %-40s (no %s -- run with the instrumented model)\n", what, file);
                return;
            }
            const std::vector<scalar> theirs = rawInternal(readField<scalar>(path), nC);
            if (offWallOnly)
            {
                std::vector<scalar> a, b, aw, bw;
                for (label c = 0; c < nC; ++c)
                {
                    if (isWall[c]) { aw.push_back(mine[c]); bw.push_back(theirs[c]); }
                    else           { a .push_back(mine[c]); b .push_back(theirs[c]); }
                }
                report(what, relL2(a, b), bound);
                std::printf("        %-38s %.6e   (rows setValues overwrites)\n",
                            "same, on the wall cells", relL2(aw, bw));
                return;
            }
            const scalar r = relL2(mine, theirs);
            report(what, r, bound);
            if (r >= bound)
            {
                scalar sa = 0.0, sb = 0.0;
                for (label c = 0; c < nC; ++c) { sa += std::fabs(mine[c]); sb += std::fabs(theirs[c]); }
                std::printf("        sum|brae| %.6e   sum|OF| %.6e\n", sa, sb);
                std::vector<label> ord(nC);
                for (label c = 0; c < nC; ++c) ord[c] = c;
                std::sort(ord.begin(), ord.end(), [&](label x, label y)
                          { return std::fabs(mine[x] - theirs[x]) > std::fabs(mine[y] - theirs[y]); });
                for (int n = 0; n < 3 && n < nC; ++n)
                {
                    const label c = ord[n];
                    std::string on;
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                        for (label i = 0; i < patches[pi].size; ++i)
                            if (patches[pi].faceCells[i] == c && on.find(patches[pi].name) == std::string::npos)
                                on += (on.empty() ? "" : "+") + patches[pi].name;
                    std::printf("        cell %-7d brae %+.6e  OF %+.6e   on [%s]\n",
                                int(c), mine[c], theirs[c], on.empty() ? "interior" : on.c_str());
                }
            }
        };
        // gradU FIRST: GbyNu is built from it and from nothing else, so if the gradient disagrees the
        // invariant cannot be the fault. Compared component by component, and localised -- a gradient that
        // is wrong only in cells touching a boundary is a boundary-face-value problem, not a scheme one.
        {
            const std::string path = D + "stage_gradU";
            if (std::ifstream(path.c_str()).good())
            {
                const std::vector<tensor> ofG = readTensorInternal(path, nC);
                std::vector<scalar> mine, theirs;
                for (label c = 0; c < nC; ++c)
                    for (int k = 0; k < 9; ++k)
                    {
                        mine.push_back(reinterpret_cast<const scalar*>(&res.gradU[c])[k]);
                        theirs.push_back(reinterpret_cast<const scalar*>(&ofG[c])[k]);
                    }
                report("gradU = fvc::grad(U)", relL2(mine, theirs), 1e-10);

                std::vector<char> touches(nC, 0);
                for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    for (label i = 0; i < patches[pi].size; ++i) touches[patches[pi].faceCells[i]] = 1;
                std::vector<scalar> iM, iT, bM, bT;
                for (label c = 0; c < nC; ++c)
                    for (int k = 0; k < 9; ++k)
                    {
                        const scalar a = reinterpret_cast<const scalar*>(&res.gradU[c])[k];
                        const scalar b = reinterpret_cast<const scalar*>(&ofG[c])[k];
                        if (touches[c]) { bM.push_back(a); bT.push_back(b); }
                        else            { iM.push_back(a); iT.push_back(b); }
                    }
                std::printf("     %-40s %.6e   (interior-only cells)\n", "  gradU away from boundaries",
                            relL2(iM, iT));
                std::printf("     %-40s %.6e   (cells touching a patch)\n", "  gradU at boundary cells",
                            relL2(bM, bT));
            }
        }
        cmp("stage_divU",   res.divU,   "divU  = div(volumetric flux)", 1e-10);
        cmp("stage_GbyNu",  res.gByNu,  "GbyNu = gradU && devTwoSymm",  1e-10);
        cmp("stage_G",      res.G,      "G     = nut*GbyNu",            1e-10);
        cmp("stage_DepsilonEff", res.DepsilonEff, "DepsilonEff = nut/sigmaEps + nu", 1e-10);
        cmp("stage_DkEff",       res.DkEff,       "DkEff       = nut/sigmak   + nu", 1e-10);
        cmp("stage_epsDivD",   res.epsDivD,   "  div(alphaRhoPhi, eps)  D()",  1e-10);
        cmp("stage_epsDivSrc", res.epsDivSrc, "  div(alphaRhoPhi, eps)  src", 1e-10);
        cmp("stage_epsLapD",   res.epsLapD,   "  laplacian(rho*DepsEff) D()",  1e-10, /*offWallOnly=*/true);
        cmp("stage_epsLapSrc", res.epsLapSrc, "  laplacian(rho*DepsEff) src", 1e-10, /*offWallOnly=*/true);
        cmp("stage_epsD0",   res.epsD0,   "epsilon D()   before relax",  1e-10, /*offWallOnly=*/true);
        cmp("stage_epsSrc0", res.epsSrc0, "epsilon source before relax",  1e-10, /*offWallOnly=*/true);
        cmp("stage_kD0",     res.kD0,     "k D()         before relax",  1e-10);
        cmp("stage_kSrc0",   res.kSrc0,   "k source      before relax",  1e-10);
        auto cmpFace = [&](const char* file, const std::vector<scalar>& mine, const char* what)
        {
            const std::string path = D + file;
            if (!std::ifstream(path.c_str()).good()) return;
            const FieldData<scalar> fd = readField<scalar>(path);
            std::vector<scalar> theirs = fd.internalUniform
                ? std::vector<scalar>(mine.size(), fd.internalUniformValue) : fd.internalField;
            // brae stores some face arrays over ALL faces and OpenFOAM writes only the internal ones;
            // compare the overlap rather than padding one side with zeros, which would read as a defect.
            const std::size_t n = std::min(mine.size(), theirs.size());
            report(what, relL2(std::vector<scalar>(mine.begin(), mine.begin() + n),
                               std::vector<scalar>(theirs.begin(), theirs.begin() + n)), 1e-10);
        };
        // The mesh factors the laplacian coefficient is the product of.
        cmpFace("stage_nonOrthDeltaCoeffs", g.nonOrthDeltaCoeffs(), "  mesh nonOrthDeltaCoeffs");
        cmpFace("stage_deltaCoeffs",        g.deltaCoeffs(),        "  mesh deltaCoeffs");
        cmpFace("stage_weights",            g.weights(),            "  mesh interpolation weights");
        cmpFace("stage_magSf",              g.magSf(),              "  mesh magSf");
        cmpFace("stage_gammaEpsFace", res.gammaEpsFace, "  face gamma = interp(rho*DepsEff)");
        cmpFace("stage_epsDivDUpper", res.epsDivUpper, "  div  upper (off-diagonal)");
        cmpFace("stage_epsDivDLower", res.epsDivLower, "  div  lower (off-diagonal)");
        cmpFace("stage_epsLapDUpper", res.epsLapUpper, "  lapl upper (off-diagonal)");
        cmpFace("stage_epsLapDLower", res.epsLapLower, "  lapl lower (off-diagonal)");
        cmpFace("stage_epsDUpper", res.epsUpper, "epsilon upper (off-diagonal)");
        cmpFace("stage_epsDLower", res.epsLower, "epsilon lower (off-diagonal)");
        cmpFace("stage_kDUpper",   res.kUpper,   "k       upper (off-diagonal)");
        cmpFace("stage_kDLower",   res.kLower,   "k       lower (off-diagonal)");
        cmp("stage_epsD",   res.epsD,   "epsilon D()",                  1e-10, /*offWallOnly=*/true);
        cmp("stage_epsSrc", res.epsSrc, "epsilon source + boundary",    1e-10, /*offWallOnly=*/true);
        cmp("stage_kD",     res.kD,     "k D()",                        1e-10);
        cmp("stage_kSrc",   res.kSrc,   "k source + boundary",          1e-10);
    }

    const std::vector<scalar> ofK   = rawInternal(readField<scalar>(D + "stage_kOut"),      nC);
    const std::vector<scalar> ofE   = rawInternal(readField<scalar>(D + "stage_epsOut"),    nC);
    const std::vector<scalar> ofN   = rawInternal(readField<scalar>(D + "stage_nutOut"),    nC);
    const std::vector<scalar> ofA   = rawInternal(readField<scalar>(D + "stage_alphatOut"), nC);
    // MACHINE PRECISION, not a tolerance chosen to fit. Every intermediate above -- gradU, GbyNu, G,
    // divU, both diffusivities, both off-diagonal sets and both assembled systems -- agrees with
    // OpenFOAM's own to the same order, so the closure is not merely close, it is the same arithmetic.
    report("epsilon after correct()", relL2(eO, ofE), 1e-11);
    report("k after correct()",       relL2(kO, ofK), 1e-11);
    report("nut = Cmu k^2/epsilon",   relL2(nO, ofN), 1e-11);
    report("alphat = rho*nut/Prt",    relL2(aO, ofA), 1e-11);
    {
        // WHERE does it live? A turbulence disagreement concentrated on wall-adjacent cells is the wall
        // treatment (epsilon's setValues, G's replacement, nut's wall function); one spread through the
        // interior is the transport equation.
        std::vector<char> isW(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            if (eps.boundary[pi]->isTurbulenceWallFunction())
                for (label i = 0; i < patches[pi].size; ++i) isW[patches[pi].faceCells[i]] = 1;
        auto split = [&](const std::vector<scalar>& a, const std::vector<scalar>& b, const char* name)
        {
            double dw = 0, nw = 0, di = 0, ni = 0; long cw = 0;
            for (label c = 0; c < nC; ++c)
            {
                const double d = (double)a[c] - (double)b[c];
                if (isW[c]) { dw += d*d; nw += (double)b[c]*b[c]; ++cw; }
                else        { di += d*d; ni += (double)b[c]*b[c]; }
            }
            std::printf("       %-8s wall(%ld) %.4e   interior(%ld) %.4e\n", name, cw,
                        nw > 0 ? std::sqrt(dw/nw) : 0.0, (long)nC - cw, ni > 0 ? std::sqrt(di/ni) : 0.0);
        };
        split(eO, ofE, "epsilon");
        split(kO, ofK, "k");
        split(nO, ofN, "nut");
    }

    // Is the SOLVE converged? A near-identical matrix giving a 7% different field is what an unconverged
    // Krylov solve looks like, so the tolerance is tightened by four orders and the answer re-measured.
    // If it moves, the previous number was the solver stopping, not the discretisation disagreeing.
    {
        std::vector<scalar> kT, eT, nT, aT;
        tolOverride = 1e-16;
        runDrop(true, 0, kT, eT, nT, aT);
        tolOverride = 1e-12;
        std::printf("     tolerance 1e-16: eps %.4e   k %.4e   (against 1e-12: %.4e / %.4e)\n",
                    relL2(eT, ofE), relL2(kT, ofK), relL2(eO, ofE), relL2(kO, ofK));
    }

    // A TERM SWEEP, not an assertion. brae's correct() can drop one term at a time; a term whose REMOVAL
    // moves brae TOWARDS OpenFOAM is a term brae is getting wrong, and one whose removal barely matters
    // cannot be the explanation for a disagreement. This is the cheapest way to turn "epsilon is 7.4e-02
    // out in the interior" into a named term.
    {
        static const char* names[] = { "as-is", "eps production", "eps divU SuSp", "eps destruction",
                                       "eps diffusion", "k production", "k divU SuSp", "k destruction",
                                       "k diffusion" };
        std::printf("     term sweep (drop one, measure epsilon and k):\n");
        for (int dt = 0; dt <= 8; ++dt)
        {
            std::vector<scalar> kD, eD, nD, aD;
            runDrop(true, dt, kD, eD, nD, aD);
            std::printf("       %-16s eps %.4e   k %.4e\n", names[dt],
                        relL2(eD, ofE), relL2(kD, ofK));
        }
    }

    // ---- THE WALL-NUT DEVIATION, measured against OpenFOAM's own stored value ------------------
    // OpenFOAM's near-wall PRODUCTION reads the stored wall nut -- epsilonWallFunctionFvPatchScalarField.C
    // :333-334 is `const tmp<scalarField> tnutw = turbModel.nut(patchi);`, consumed at :342 as
    // (nutw + nuw)*magGradUw. brae RECOMPUTES it instead, from the current k and the current per-face nu
    // (kEpsilon_cpp.cu:240, and kEpsilon.cu passes nutWall=0 to deviceWallEpsG0 for the same reason).
    //
    // For nutkWallFunction the k is the same at that point in the iteration, so the two differ only
    // through nu_w = mu(T_w)/rho_w, which moves whenever the wall's T or p does -- sbMatched's walls are
    // zeroGradient on both. stage_nutIn IS the field OpenFOAM's production reads, so the gap is
    // measurable here without instrumenting anything further.
    //
    // MEASURED AT EXACTLY ZERO over all 22400 wall faces on this state -- abs 0.000000e+00. So the two
    // conventions are not an arithmetic disagreement: handed the same k and the same per-face nu,
    // recomputing nutkWallFunction reproduces OpenFOAM's stored value bit for bit, which is what the
    // formula being shared (nut_wall_function.cuh, one BRAE_HD definition) predicts.
    //
    // WHAT THIS DOES NOT ESTABLISH, and the difference is the whole point of the deviation: this gate
    // runs ONE correct() from OpenFOAM's own dumped inputs, so the stored nut and the nu brae recomputes
    // from are drawn from the same consistent state. In a running solver they are not -- OpenFOAM's
    // stored value was written by the PREVIOUS correctNut, and EEqn has moved T (hence nu_w = mu(T_w)/
    // rho_w, zeroGradient on both here) in between. That staleness needs a multi-iteration comparison
    // and no gate here provides one; dumpKEpsilon writes stage_G but no G0, so the production term has
    // no OpenFOAM oracle at all.
    //
    // On this evidence the restructuring is NOT justified: it would change nothing measurable, and the
    // last structurally-correct change made on that reasoning (skipping empty patches in fvc) moved a
    // real gate's converged U by 3.3x. Reported, not gated -- a bound whose correct value is zero only
    // in the limit would be asserting the wrong thing.
    {
        const KEpsilonCoeffs kw;
        const std::vector<std::vector<scalar>> yW = brae::nearWallDist(m, g, patches);
        double worst = 0.0, worstStored = 0.0;
        std::size_t nWallFaces = 0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!eps.boundary[pi]->isTurbulenceWallFunction()) continue;
            const std::vector<scalar>& stored = nut.boundary[pi]->value();
            std::vector<scalar> nuFace(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i) nuFace[i] = nuBnd[pi][i];
            const std::vector<scalar> recomputed =
                brae::nutkWallFunction(patches[pi], yW[pi], k.internal, nuFace, kw.Cmu, kw.kappa, kw.E);
            for (label i = 0; i < patches[pi].size; ++i, ++nWallFaces)
            {
                const double d = std::fabs((double)recomputed[i] - (double)stored[i]);
                if (d > worst) { worst = d; worstStored = (double)stored[i]; }
            }
        }
        const double rel = worstStored > 0.0 ? worst / worstStored : worst;
        std::printf("     %-40s abs %.6e  rel %.6e  over %zu wall faces\n",
                    "wall nut: recomputed vs OF's stored", worst, rel, nWallFaces);
        std::printf("     %-40s %s\n", "  (reported -- see the note above)",
                    "brae recomputes where OpenFOAM reads");
    }

    // ---- 2. THE CONTROL: divU from the MASS flux instead of the volumetric one. ----
    std::printf("  2. control -- the mass flux must NOT serve as divU\n");
    std::vector<scalar> kW, eW, nW, aW;
    run(false, kW, eW, nW, aW);
    const double wrongE = relL2(eW, ofE);
    const double wrongK = relL2(kW, ofK);
    // The bound is RELATIVE to the correct answer, and this TIGHTENS the control rather than relaxing it.
    // A fixed 1e-3 passed while the closure itself was 2.8e-02 out -- which is to say it could not tell a
    // working divU from a broken one, because everything was broken. What actually discriminates is that
    // the wrong flux is worse by orders of magnitude: 5e-06 against 5e-15 here. The absolute floor stays,
    // so a fixture with uniform rho -- where the two fluxes coincide and nothing COULD discriminate --
    // fails this control instead of passing it vacuously.
    const double rightE = relL2(eO, ofE);
    const double rightK = relL2(kO, ofK);
    check("mass-flux divU disagrees on epsilon", wrongE > 1e3 * rightE && wrongE > 1e-9);
    check("mass-flux divU disagrees on k",       wrongK > 1e3 * rightK && wrongK > 1e-9);
    std::printf("     %-40s eps %.4e   k %.4e\n", "  (its errors, for the record)", wrongE, wrongK);

    // ---- 3. THE CONTROL that the inputs move at all: the fields must CHANGE. A closure that returned
    //         its input unchanged would pass every bound above if OpenFOAM had also barely moved.
    std::printf("  3. control -- correct() actually changed the fields\n");
    const std::vector<scalar> inK = rawInternal(readField<scalar>(D + "stage_kIn"),   nC);
    const std::vector<scalar> inE = rawInternal(readField<scalar>(D + "stage_epsIn"), nC);
    check("OpenFOAM moved epsilon", relL2(inE, ofE) > 1e-3);
    check("OpenFOAM moved k",       relL2(inK, ofK) > 1e-3);
    std::printf("     %-40s eps %.4e   k %.4e\n", "  (how far it moved them)",
                relL2(inE, ofE), relL2(inK, ofK));

    // ---- 4. THE REFUSAL. OF's turbulent inlets compute refValue from another field; brae refuses when
    //         that field does not cover the patch rather than filling the gap with zeros, which would be a
    //         silent no-turbulence inlet. The NEGATIVE CONTROL is the same call with a full-length array:
    //         if that threw too, the test above would pass for the wrong reason.
    std::printf("  4. refusal -- a turbulent inlet with too little of the field it reads\n");
    {
        const FvPatch& fp = patches[0];
        TurbulentInletPatchField<scalar> t(
            fp, true, scalar{1}, {}, TurbulentInletPatchField<scalar>::mixingLengthEpsilon, 0.005);

        bool threw = false;
        bool named = false;
        try
        {
            t.updateTurbulentInlet({}, std::vector<scalar>(fp.size > 0 ? fp.size - 1 : 0, 1.0), 0.09);
        }
        catch (const std::exception& e)
        {
            threw = true;
            named = std::string(e.what()).find(fp.name) != std::string::npos;
        }
        check("a short k refuses", threw);
        check("and the refusal names the patch", named);

        bool controlThrew = false;
        try
        {
            t.updateTurbulentInlet({}, std::vector<scalar>(fp.size, 1.0), 0.09);
        }
        catch (const std::exception&)
        {
            controlThrew = true;
        }
        check("control -- a full-length k does NOT refuse", !controlThrew);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
