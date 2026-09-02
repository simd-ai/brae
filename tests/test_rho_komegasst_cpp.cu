// The COMPRESSIBLE kOmegaSST closure against OpenFOAM's OWN kOmegaSST, term by term.
//
// The oracle is tools/dumpKOmegaSST -- OpenFOAM's kOmegaSSTBase.C with writes added and its equations
// untouched -- run under tools/dumpPEqn so the model's INPUTS (rho, mu, phi) come from the same
// iteration. Every intermediate the port is built from therefore has an oracle rather than an argument:
// divU, gradU, S2, GbyNu0, G, CDkOmega, F1, F23, both assembled systems before and after relax, and their
// off-diagonals, which a per-cell view cannot see.
//
// THE CONTROL is the one thing that separates the compressible reading of this templated model from the
// incompressible one: fvm::div and the `bounded` Sp take the MASS flux, while
// divU = fvc::div(fvc::absolute(this->phi(), U)) takes the VOLUMETRIC one, because
// compressibleTurbulenceModel::phi() divides by fvc::interpolate(rho). Substituting one for the other
// must be measurably worse; if it is not, this fixture cannot tell them apart and the gate proves nothing.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "kOmegaSST_cpp.cuh"
#include "cell_wall_dist.cuh"
#include "fvc.cuh"
#include "linearViscousStress_cpp.cuh"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace brae;

static int failures = 0;

static void report(const std::string& what, double got, double bound)
{
    const bool ok = got < bound;
    if (!ok) ++failures;
    std::printf("     %-44s %.6e   %s\n", what.c_str(), got, ok ? "ok" : "FAIL");
}

static void check(const std::string& what, bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-44s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

static double relL2(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    double d = 0.0, n = 0.0;
    const std::size_t sz = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < sz; ++i)
    {
        const double x = (double)a[i] - (double)b[i];
        d += x * x;
        n += (double)b[i] * (double)b[i];
    }
    return n > 0.0 ? std::sqrt(d / n) : std::sqrt(d);
}

template<class T>
static std::vector<T> rawInternal(const FieldData<T>& fd, label nC)
{
    if (fd.internalUniform) return std::vector<T>(nC, fd.internalUniformValue);
    return fd.internalField;
}

template<class T>
static std::vector<std::vector<T>> rawBoundary(const FieldData<T>& fd,
                                               const std::vector<FvPatch>& patches)
{
    std::vector<std::vector<T>> out(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        out[pi].assign(patches[pi].size, T{});
        for (const auto& b : fd.boundary)
        {
            if (b.name != patches[pi].name) continue;
            if (b.valueUniform)                                     out[pi].assign(patches[pi].size, b.uniformValue);
            else if (b.values.size() == (std::size_t)patches[pi].size) out[pi] = b.values;
        }
    }
    return out;
}

// brae's field reader has no tensor instantiation, and this comparison is not a reason to add one to
// shared code: the file is ASCII and its internalField is a flat list of 9-component entries.
static std::vector<tensor> readTensorInternal(const std::string& path, label nC)
{
    std::ifstream in(path.c_str());
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::size_t at = text.find("internalField");
    if (at == std::string::npos) throw std::runtime_error("no internalField in " + path);
    std::size_t p = text.find('(', at);
    if (p == std::string::npos) throw std::runtime_error("no list in " + path);

    std::vector<scalar> flat;
    const std::size_t want = std::size_t(9) * std::size_t(nC);
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
        for (int kk = 0; kk < 9; ++kk)
            reinterpret_cast<scalar*>(&out[c])[kk] = flat[std::size_t(9) * std::size_t(c) + std::size_t(kk)];
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
    std::printf("rhoSimpleFoam compressible kOmegaSST vs OpenFOAM (%d cells)\n", (int)nC);

    // ---- OpenFOAM's own inputs, as the model had them --------------------------------------------
    const FieldData<scalar> rhoFd = readField<scalar>(D + "stage_rhoTurb");
    const std::vector<scalar>              rho    = rawInternal(rhoFd, nC);
    const std::vector<std::vector<scalar>> rhoBnd = rawBoundary<scalar>(rhoFd, patches);
    const FieldData<scalar> muFd = readField<scalar>(D + "stage_muTurb");
    const std::vector<scalar>              mu     = rawInternal(muFd, nC);
    const std::vector<std::vector<scalar>> muBnd  = rawBoundary<scalar>(muFd, patches);

    // nu = mu/rho, the LAMINAR kinematic viscosity, per cell and per boundary face.
    std::vector<scalar> nu(nC);
    for (label c = 0; c < nC; ++c) nu[c] = mu[c] / rho[c];
    std::vector<std::vector<scalar>> nuBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        nuBnd[pi].resize(patches[pi].size);
        // An EMPTY patch carries rho_b 0 in the dump: guard as test_rho_kepsilon_cpp does, else NaN.
        for (label i = 0; i < patches[pi].size; ++i)
            nuBnd[pi][i] = (patches[pi].type == "empty" || !(rhoBnd[pi][i] > 0)) ? 0.0 : muBnd[pi][i] / rhoBnd[pi][i];
    }

    const FieldData<scalar> phiFd = readField<scalar>(D + "stage_phiTurb");
    SurfaceScalarField phi;
    phi.internal = phiFd.internalField;
    phi.boundary = rawBoundary<scalar>(phiFd, patches);

    // compressibleTurbulenceModel::phi() -- the VOLUMETRIC flux, phi_/fvc::interpolate(rho).
    SurfaceScalarField phiByRho = phi;
    {
        const SurfaceScalarField rhof = cpu::effectiveFaceViscosity(rho, rhoBnd, m, g, patches);
        for (std::size_t f = 0; f < phiByRho.internal.size(); ++f)
            phiByRho.internal[f] /= rhof.internal[f];
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i)
                phiByRho.boundary[pi][i] = (patches[pi].type == "empty" || !(rhof.boundary[pi][i] > 0))
                    ? 0.0 : phiByRho.boundary[pi][i] / rhof.boundary[pi][i];
    }

    // Wall distance, the same meshWave brae uses elsewhere. F1 and F2 are built from it.
    const std::vector<scalar> y = cellWallDist(m, g, patches);

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* rel = fvSolution.subDict("relaxationFactors");
    const FoamDict* eqs = rel ? rel->subDict("equations") : nullptr;
    const scalar relaxOmega = eqs ? eqs->scalarOr("omega", 1.0) : 1.0;
    const scalar relaxK     = eqs ? eqs->scalarOr("k",     1.0) : 1.0;

    // ---- run the closure -------------------------------------------------------------------------
    std::vector<scalar> kOut, omOut, nutOut, alphatOut;
    auto run = [&](bool useVolumetricFluxForDivU,
                   std::vector<scalar>& kO, std::vector<scalar>& omO,
                   std::vector<scalar>& nO, std::vector<scalar>& aO,
                   cpu::kOmegaSST::SSTResiduals* resOut)
    {
        GeometricField<scalar> k   = buildField<scalar>(readField<scalar>(D + "stage_sstKIn"),     patches, nC);
        GeometricField<scalar> om  = buildField<scalar>(readField<scalar>(D + "stage_sstOmegaIn"), patches, nC);
        GeometricField<scalar> nut = buildField<scalar>(readField<scalar>(D + "stage_sstNutIn"),   patches, nC);
        GeometricField<vector> U   = buildField<vector>(readField<vector>(D + "stage_sstU"),       patches, nC);
        k.evaluateBoundary(); om.evaluateBoundary(); nut.evaluateBoundary();

        // The flux-conditional boundaries need the flux before they mean anything: OpenFOAM's fvMatrix
        // constructor calls updateCoeffs(), which is where inletOutlet reads sign(phi). Without this the
        // outlet evaluates on a seeded valueFraction and fvc::grad(U) reads the wrong face value there --
        // measured at 1.5e-01 on the boundary cells during the kEpsilon port.
        for (std::size_t pi = 0; pi < patches.size(); ++pi) U.boundary[pi]->updateFromFlux(phi.boundary[pi]);
        U.evaluateBoundary();

        std::vector<scalar> at(nC, 0.0);
        cpu::kOmegaSST::Compressible comp;
        comp.rho      = &rho;
        comp.rhoBnd   = &rhoBnd;
        comp.nu       = &nu;
        comp.nuBnd    = &nuBnd;
        comp.phiByRho = useVolumetricFluxForDivU ? &phiByRho : &phi;
        comp.alphat   = &at;
        comp.Prt      = 1.0;

        // kOmegaSST takes the laplacian scheme as a function argument, not in its coefficients --
        // sbMatched sets `laplacianSchemes default Gauss linear corrected`, passed below.
        KOmegaSSTCoeffs co;

        cpu::kOmegaSST::SSTResiduals res;
        res.captureStages = (resOut != nullptr);
        cpu::kOmegaSST::correct(U, k, om, nut, phi, y, /*nu=*/0.0, m, g, patches,
                                relaxOmega, relaxK, 1e-16, 0.0, 20000, co, &res,
                                /*bounded=*/true, /*limitedLinear=*/false, 1.0,
                                /*linearUpwind=*/false, /*correctedLaplacian=*/true, 0.0,
                                /*lm=*/nullptr, &comp);
        kO = k.internal; omO = om.internal; nO = nut.internal; aO = at;
        if (resOut) *resOut = res;
    };

    cpu::kOmegaSST::SSTResiduals res;
    run(true, kOut, omOut, nutOut, alphatOut, &res);
    std::printf("  initial residuals: omega %.4e   k %.4e\n", (double)res.omega, (double)res.k);

    // ---- 0. the intermediates, against OpenFOAM's own. G, S2 and the blends are what the two
    //         equations are built from, so a disagreement downstream is either here or in the assembly,
    //         and this separates the two instead of arguing about it.
    std::printf("  0. the intermediates, against OpenFOAM's own\n");
    {
        auto cmp = [&](const char* file, const std::vector<scalar>& mine, const char* what,
                       double bound = 1e-11)
        {
            const std::string path = D + file;
            if (!std::ifstream(path.c_str()).good()) { std::printf("     (no %s)\n", file); return; }
            report(what, relL2(mine, rawInternal(readField<scalar>(path), nC)), bound);
        };
        // gradU FIRST: S2 and GbyNu0 are built from it and from nothing else.
        {
            const std::string path = D + "stage_sstGradU";
            if (std::ifstream(path.c_str()).good())
            {
                const std::vector<tensor> ofG = readTensorInternal(path, nC);
                std::vector<scalar> a, b, ai, bi;
                std::vector<char> onB(nC, 0);
                for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    for (label i = 0; i < patches[pi].size; ++i) onB[patches[pi].faceCells[i]] = 1;
                for (label c = 0; c < nC; ++c)
                    for (int kk = 0; kk < 9; ++kk)
                    {
                        const scalar x = reinterpret_cast<const scalar*>(&res.gradU[c])[kk];
                        const scalar yy = reinterpret_cast<const scalar*>(&ofG[c])[kk];
                        a.push_back(x); b.push_back(yy);
                        if (!onB[c]) { ai.push_back(x); bi.push_back(yy); }
                    }
                report("gradU = fvc::grad(U)", relL2(a, b), 1e-11);
                std::printf("       %-42s %.6e\n", "gradU away from boundaries", relL2(ai, bi));
            }
        }
        // divU is a flux-derived divergence; its floor on this mesh is ~9e-11, the same figure the
        // kEpsilon gate reads for the identical quantity.
        cmp("stage_sstDivU",     res.divU,   "divU  = div(volumetric flux)", 1e-10);
        cmp("stage_sstS2",       res.s2,     "S2");
        cmp("stage_sstGbyNu0",   res.gbyNu0, "GbyNu0");
        cmp("stage_sstG",        res.G,      "G      = nut*GbyNu0");
        cmp("stage_sstCDkOmega", res.CD,     "CDkOmega");
        cmp("stage_sstF1",       res.f1,     "F1 blend");
        {
            // F1 ON THE PATCH FACES against OpenFOAM's own boundary field: the diffusivities' boundary
            // coefficients blend this value, and on the inlet it differs from the owner cell's.
            const FieldData<scalar> fd = readField<scalar>(D + "stage_sstF1");
            const std::vector<std::vector<scalar>> ofB = rawBoundary<scalar>(fd, patches);
            std::vector<scalar> a, b;
            for (std::size_t pi = 0; pi < patches.size() && pi < res.f1Bnd.size(); ++pi)
            {
                if (patches[pi].type == "empty") continue;
                for (label i = 0; i < patches[pi].size; ++i) { a.push_back(res.f1Bnd[pi][i]); b.push_back(ofB[pi][i]); }
            }
            report("F1 on the boundary faces", relL2(a, b), 1e-10);
        }
        cmp("stage_sstF23",      res.f23,    "F23 blend");
    }

    // ---- 0b. the assembled systems. With every intermediate exact, anything left is here.
    std::printf("  0b. the assembled systems, against OpenFOAM's own\n");
    {
        std::vector<char> isWall(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            if (patches[pi].type == "wall")
                for (label i = 0; i < patches[pi].size; ++i) isWall[patches[pi].faceCells[i]] = 1;

        auto cmpSys = [&](const char* file, const std::vector<scalar>& mine, const char* what,
                          bool offWallOnly)
        {
            const std::string path = D + file;
            if (!std::ifstream(path.c_str()).good()) return;
            const std::vector<scalar> theirs = rawInternal(readField<scalar>(path), nC);
            if (offWallOnly)
            {
                // OpenFOAM's omegaWallFunction writes the cell value and brae's rows are written by
                // setValues, so the ASSEMBLED wall rows differ while the solved values agree. Assert off
                // the wall and report the wall figure beside it.
                std::vector<scalar> a, b, aw, bw;
                for (label c = 0; c < nC; ++c)
                {
                    if (isWall[c]) { aw.push_back(mine[c]); bw.push_back(theirs[c]); }
                    else           { a .push_back(mine[c]); b .push_back(theirs[c]); }
                }
                report(what, relL2(a, b), 1e-11);
                std::printf("        %-40s %.6e\n", "same, on the wall cells", relL2(aw, bw));
                return;
            }
            report(what, relL2(mine, theirs), 1e-11);
        };
        auto cmpFace = [&](const char* file, const std::vector<scalar>& mine, const char* what)
        {
            const std::string path = D + file;
            if (!std::ifstream(path.c_str()).good()) return;
            const FieldData<scalar> fd = readField<scalar>(path);
            std::vector<scalar> theirs = fd.internalUniform
                ? std::vector<scalar>(mine.size(), fd.internalUniformValue) : fd.internalField;
            const std::size_t n = std::min(mine.size(), theirs.size());
            report(what, relL2(std::vector<scalar>(mine.begin(), mine.begin() + n),
                               std::vector<scalar>(theirs.begin(), theirs.begin() + n)), 1e-11);
        };
        cmpSys("stage_sstOmD0",   res.omD0,   "omega D()   before relax", true);
        cmpSys("stage_sstOmSrc0", res.omSrc0, "omega src   before relax", true);
        cmpSys("stage_sstKD0",    res.kD0,    "k     D()   before relax", false);
        cmpSys("stage_sstKSrc0",  res.kSrc0,  "k     src   before relax", false);
        cmpFace("stage_sstOmDUpper", res.omUpper, "omega upper (off-diagonal)");
        cmpFace("stage_sstOmDLower", res.omLower, "omega lower (off-diagonal)");
        cmpFace("stage_sstKDUpper",  res.kUpper,  "k     upper (off-diagonal)");
        cmpFace("stage_sstKDLower",  res.kLower,  "k     lower (off-diagonal)");
    }

    // ---- 1. the outputs, against OpenFOAM's own --------------------------------------------------
    std::printf("  1. after correct(), against OpenFOAM's own kOmegaSST\n");
    const std::vector<scalar> ofK   = rawInternal(readField<scalar>(D + "stage_sstKOut"),     nC);
    const std::vector<scalar> ofOm  = rawInternal(readField<scalar>(D + "stage_sstOmegaOut"), nC);
    const std::vector<scalar> ofNut = rawInternal(readField<scalar>(D + "stage_sstNutOut"),   nC);
    // 1e-10, and that is the SOLVE's floor rather than the assembly's. Every intermediate and both
    // assembled systems agree at 1e-14 to 1e-15 above; these three come out the other side of a BiCGStab
    // on a 112000-cell system, and nut is formed from two just-solved fields so it carries both. Measured
    // omega 2.2e-12, k 2.5e-12, nut 1.0e-11. The control below disagrees by 1e-06, four orders outside
    // this bound, so it still discriminates.
    report("omega after correct()", relL2(omOut, ofOm),   1e-10);
    report("k after correct()",     relL2(kOut,  ofK),    1e-10);
    report("nut after correct()",   relL2(nutOut, ofNut), 1e-10);
    {
        const std::string ap = D + "stage_sstAlphatOut";
        if (std::ifstream(ap.c_str()).good())
            report("alphat = rho*nut/Prt", relL2(alphatOut, rawInternal(readField<scalar>(ap), nC)), 1e-10);
    }

    // ---- 2. WHERE it lives, if it does not agree -------------------------------------------------
    {
        std::vector<char> isW(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            if (patches[pi].type == "wall")
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
        split(omOut,  ofOm,  "omega");
        split(kOut,   ofK,   "k");
        split(nutOut, ofNut, "nut");
    }

    // ---- 3. THE CONTROL: the mass flux must not serve as divU -------------------------------------
    std::printf("  2. control -- the mass flux must NOT serve as divU\n");
    {
        std::vector<scalar> kW, omW, nW, aW;
        run(false, kW, omW, nW, aW, nullptr);
        const double wrongOm = relL2(omW, ofOm);
        const double wrongK  = relL2(kW,  ofK);
        const double rightOm = relL2(omOut, ofOm);
        const double rightK  = relL2(kOut,  ofK);
        // RELATIVE to the correct answer, with an absolute floor. A fixed absolute bound goes dead the
        // moment the closure is right -- it passed throughout the period the kEpsilon closure was 2.8e-02
        // out, which is to say it could not tell a working divU from a broken one. The floor makes a
        // fixture with uniform rho, where the two fluxes coincide, FAIL this control rather than pass it.
        check("mass-flux divU disagrees on omega", wrongOm > 1e3 * rightOm && wrongOm > 1e-9);
        check("mass-flux divU disagrees on k",     wrongK  > 1e3 * rightK  && wrongK  > 1e-9);
        std::printf("     %-44s omega %.4e   k %.4e\n", "  (its errors, for the record)", wrongOm, wrongK);
    }

    // ---- 4. THE CONTROL that correct() moved anything at all --------------------------------------
    std::printf("  3. control -- correct() actually changed the fields\n");
    {
        const std::vector<scalar> inK  = rawInternal(readField<scalar>(D + "stage_sstKIn"),     nC);
        const std::vector<scalar> inOm = rawInternal(readField<scalar>(D + "stage_sstOmegaIn"), nC);
        check("OpenFOAM moved omega", relL2(inOm, ofOm) > 1e-3);
        check("OpenFOAM moved k",     relL2(inK,  ofK)  > 1e-3);
        std::printf("     %-44s omega %.4e   k %.4e\n", "  (how far it moved them)",
                    relL2(inOm, ofOm), relL2(inK, ofK));
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
