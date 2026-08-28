// rhoSimpleFoam's UEqn.H against REAL OpenFOAM's own assembled momentum matrix.
//
// THE ORACLE is tools/dumpPEqn -- OpenFOAM's rhoSimpleFoam with a stage harness that writes, at a chosen
// SIMPLE iteration:
//
//   stage_rAU    1/UEqn.A()                       the matrix diagonal, AFTER relax()
//   stage_UIC    UEqn.internalCoeffs(), per patch  the boundary contribution to the diagonal
//   stage_UBC    UEqn.boundaryCoeffs(),  per patch the boundary contribution to the source
//   stage_muEff  turbulence->muEff()               the DYNAMIC viscosity the assembly actually used
//
// WHY stage_muEff MATTERS TO THE DESIGN OF THIS TEST. brae has no ported compressible turbulence model
// yet -- that is a separate manifest component. Injecting OpenFOAM's own muEff makes this a test of the
// ASSEMBLY and of nothing else: if it fails, the momentum equation is wrong, not the closure. Mixing the
// two would produce a number that cannot be attributed, which is the failure mode the whole manifest
// exists to avoid.
//
// WHAT IS ACTUALLY UNDER TEST. linearViscousStress.C defines one operator:
//
//     divDevRhoReff(U) = -fvc::div((alpha*rho*nuEff)*dev2(T(grad U))) - fvm::laplacian(alpha*rho*nuEff, U)
//
// and the compressible solver reaches it with the DYNAMIC viscosity mu_eff = rho*nu_eff where the
// incompressible one reaches it with the kinematic nu_eff. That single factor of rho is the entire
// difference between this file and simpleFoam's UEqn, it is order 1 for air at ambient conditions, and it
// varies with the solution -- so a port that reuses the incompressible form is wrong by an amount that
// looks plausible in every field plot. THE CONTROL below assembles the incompressible way on purpose and
// requires it to disagree with OpenFOAM. Without that, a passing bound would not distinguish the two.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fv_matrix_ops.cuh"
#include "rhoCreateFields_cpp.cuh"
#include "cellLimitedGrad_cpp.cuh"
#include "scheme_parse.cuh"
#include "fvOptions_cpp.cuh"
#include "linearViscousStress_cpp.cuh"
#include "rhoSimpleFoam_cpp.cuh"   // effectiveTransport: the solver's OWN muEff
#include "rhoUEqn_cpp.cuh"

#include <cmath>
#include <numeric>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

// The staged fields are OpenFOAM's own diagnostic output and carry whatever boundary type OpenFOAM gave
// them -- muEff comes out as `extrapolatedCalculated`. brae's patch-field factory has no such type, and
// teaching it one for the sake of reading a test fixture would be changing shared code to suit a test.
// These read the parsed file directly instead: the staged values are what is wanted, not a re-evaluated
// boundary condition.
static std::vector<scalar> rawInternal(
    const FieldData<scalar>& fd,
    label nC)
{
    if (fd.internalUniform) return std::vector<scalar>(nC, fd.internalUniformValue);
    return fd.internalField;
}

template <typename T>
static std::vector<std::vector<T>> rawBoundary(
    const FieldData<T>&         fd,
    const std::vector<FvPatch>& patches)
{
    std::vector<std::vector<T>> out(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        out[pi].assign(patches[pi].size, T{});
        for (const auto& b : fd.boundary)
        {
            if (b.name != patches[pi].name) continue;
            if (b.valueUniform)
            {
                out[pi].assign(patches[pi].size, b.uniformValue);
            }
            else if (static_cast<label>(b.values.size()) == patches[pi].size)
            {
                out[pi] = b.values;
            }
            break;
        }
    }
    return out;
}

static int failures = 0;

static void report(
    const std::string& what,
    double got,
    double bound)
{
    const bool ok = got < bound;
    if (!ok) ++failures;
    std::printf("     %-44s %.6e   %s\n", what.c_str(), got, ok ? "ok" : "FAIL");
}

static void check(
    const std::string& what,
    bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-44s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

static double relL2(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
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

// Boundary coefficients, flattened patch by patch, so one number covers the whole boundary.
static double relL2Boundary(
    const std::vector<std::vector<vector>>& a,
    const std::vector<std::vector<vector>>& b,
    const std::vector<FvPatch>&             patches)
{
    std::vector<scalar> fa, fb;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<vector>& bv = b[pi];
        for (label i = 0; i < patches[pi].size; ++i)
        {
            fa.push_back(a[pi][i].x); fa.push_back(a[pi][i].y); fa.push_back(a[pi][i].z);
            fb.push_back(bv[i].x);    fb.push_back(bv[i].y);    fb.push_back(bv[i].z);
        }
    }
    return relL2(fa, fb);
}

// A raw BOUNDARY reader for a tensor dump. brae's field reader has no tensor instantiation, and this
// comparison is not a reason to add one to shared code: the file is ASCII and a patch's `value` entry is
// either `uniform (9 numbers)` or `nonuniform List<tensor> N ( ... )`. Returns empty if the patch has no
// value entry, which is how an `empty` patch (size 0 in OpenFOAM) shows up.
static std::vector<tensor> readTensorPatch(
    const std::string& path,
    const std::string& patchName,
    label              nFaces)
{
    std::vector<tensor> out;
    std::ifstream in(path.c_str());
    if (!in.good()) return out;
    const std::string txt((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const std::size_t bf = txt.find("boundaryField");
    if (bf == std::string::npos) return out;
    // The patch's own block: its name at the top level of boundaryField, then its braces.
    std::size_t at = txt.find(patchName, bf);
    while (at != std::string::npos)
    {
        const std::size_t ob = txt.find('{', at);
        const std::size_t nl = txt.find('\n', at);
        if (ob != std::string::npos && (nl == std::string::npos || ob < txt.find('}', at)))
        {
            // Scan to the matching close brace so `value` is taken from THIS patch only.
            std::size_t i = ob + 1;
            int depth = 1;
            while (i < txt.size() && depth > 0)
            {
                if (txt[i] == '{') ++depth;
                else if (txt[i] == '}') --depth;
                ++i;
            }
            const std::string blk = txt.substr(ob, i - ob);
            const std::size_t v = blk.find("value");
            if (v == std::string::npos) return out;
            const bool uniform = blk.find("uniform", v) != std::string::npos
                              && blk.find("nonuniform", v) == std::string::npos;
            const std::size_t op = blk.find('(', v);
            if (op == std::string::npos) return out;
            std::vector<scalar> flat;
            const std::size_t want = uniform ? 9 : std::size_t(9) * std::size_t(nFaces);
            for (std::size_t j = op; j < blk.size() && flat.size() < want; )
            {
                const char ch = blk[j];
                if (ch == '(' || ch == ')' || std::isspace((unsigned char)ch)) { ++j; continue; }
                char* end = nullptr;
                flat.push_back(std::strtod(blk.c_str() + j, &end));
                if (end == blk.c_str() + j) break;
                j = std::size_t(end - blk.c_str());
            }
            if (flat.size() < want) return out;
            out.resize(nFaces);
            for (label f2 = 0; f2 < nFaces; ++f2)
                for (int kk = 0; kk < 9; ++kk)
                    out[f2] = out[f2],
                    reinterpret_cast<scalar*>(&out[f2])[kk] =
                        flat[uniform ? std::size_t(kk)
                                     : std::size_t(9)*std::size_t(f2) + std::size_t(kk)];
            return out;
        }
        at = txt.find(patchName, at + 1);
    }
    return out;
}

// The case's fvOptions, at file scope so the diagnostic lambdas above main can name the
// porosity cells they are reporting on.
static cpu::fvOptions::OptionList uopts;

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <startTime> <dumpTime>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string startT  = argv[2];
    const std::string dumpT   = argv[3];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    std::printf("rhoSimpleFoam UEqn vs OpenFOAM (%d cells)\n", (int)nC);

    // The state OpenFOAM assembled from: the start-time fields, and its own initial phi.
    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);

    // OpenFOAM's own muEff, so this measures the assembly and not an unported turbulence closure.
    // T's AND he's updateCoeffs() BEFORE anything is built from them. The solver updates all three
    // flux-conditional families at the top of the iteration (rhoSimpleFoam_cpp.cu:182-184); this gate
    // updated only U, and it did it far below -- after muEff had already been built from a T whose
    // inletOutlet was still sitting at its refValue. On angledDuct that refValue is 293 while the outlet
    // actually runs at ~350.4 (the porous block is pinned to 350), and brae's own laminar muEff read
    // 1.281e-01 against OpenFOAM on the outlet with every other patch at 1e-15. sutherland(293) =
    // 1.8139e-05 against sutherland(350.4) = 2.0805e-05 is exactly that 12.8%: the gate's number, not
    // the solver's. A gate that reports its own omission as a solver defect costs more than it saves.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        f.T.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
        f.he.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
    }
    f.T.evaluateBoundary();

    const FieldData<scalar> muEffFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_muEff");
    const std::vector<scalar> muEffInt = rawInternal(muEffFd, nC);
    const std::vector<std::vector<scalar>> muEffBnd = rawBoundary<scalar>(muEffFd, patches);

    // brae's OWN muEff, which this gate otherwise only INJECTS. The injection above is deliberate -- it
    // keeps the assembly number attributable to the assembly -- but its cost is that the solver's own
    // viscosity is never compared to OpenFOAM's, and a driver that assembles exactly from a muEff of its
    // own making still converges somewhere else. That is not hypothetical: alphaEff, injected into the
    // energy gate for the same reason, was six orders too large because transportAlpha was handed a
    // temperature where it takes a viscosity, and no gate could see it.
    //
    // Asserted against the LAMINAR half, thermo.mu(), because the turbulence model's nut at this point is
    // the model's own state and not something createFields reconstructs.
    {
        std::printf("  0. brae's own transport, against OpenFOAM's\n");
        std::vector<scalar> mineMu, mineAlpha;
        std::vector<std::vector<scalar>> mineMuB, mineAlphaB;
        cpu::rhoSimple::effectiveTransport(f, patches, mineMu, mineMuB, mineAlpha, mineAlphaB);

        const std::string lamPath = caseDir + "/" + dumpT + "/stage_muLam";
        if (std::ifstream(lamPath.c_str()).good())
        {
            const FieldData<scalar> lamFd = readField<scalar>(lamPath);
            const std::vector<scalar> ofLam = rawInternal(lamFd, nC);
            std::vector<scalar> mineLam(nC);
            for (label c = 0; c < nC; ++c)
            {
                const scalar mut = (f.turbulent && !f.nut.internal.empty())
                                 ? f.rho.internal[c] * f.nut.internal[c] : scalar(0);
                mineLam[c] = mineMu[c] - mut;
            }
            report("brae's own laminar muEff == OpenFOAM's", relL2(mineLam, ofLam), 1e-12);
            std::printf("     %-40s brae %.6e   OpenFOAM %.6e\n", "  (the value itself)",
                        mineLam[0], ofLam[0]);

            // The BOUNDARY too: divDevRhoReff's face viscosity takes the patch values, so a muEff that is
            // right in the cell and wrong on the face still poisons the momentum coefficients there.
            const std::vector<std::vector<scalar>> ofLamB = rawBoundary<scalar>(lamFd, patches);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                std::vector<scalar> a, b;
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const scalar mutB = (f.turbulent && !f.nut.internal.empty())
                                      ? f.rho.boundary[pi]->value()[i] * f.nut.boundary[pi]->value()[i]
                                      : scalar(0);
                    a.push_back(mineMuB[pi][i] - mutB);
                    b.push_back(ofLamB[pi][i]);
                }
                std::printf("     laminar muEff on %-16s %.6e\n", patches[pi].name.c_str(), relL2(a, b));
            }
        }
        std::printf("     %-40s %.6e   (turbulent halves differ in this state)\n",
                    "full muEff vs OpenFOAM's", relL2(mineMu, muEffInt));
    }

    // U EXACTLY as OpenFOAM assembled from. sbMatched's inlet is flowRateInletVelocity, whose value
    // depends on which rho it is fed -- a separate component with its own history in this repo. Taking
    // OpenFOAM's evaluated U here keeps this gate measuring the momentum ASSEMBLY; the BC gets its own.
    const FieldData<vector> uAssFd = readField<vector>(caseDir + "/" + dumpT + "/stage_Uass");
    // THE BOUNDARY comes from stage_Upost, taken AFTER fvMatrix's constructor called updateCoeffs, which
    // is the state the matrix was actually assembled from. stage_Uass is taken BEFORE that, so any
    // boundary OpenFOAM recomputes per iteration -- flowRateInletVelocity, the freestream family -- still
    // holds the previous iteration's value there. Comparing brae's recomputed inlet against it read
    // 1.2e-04 on angledDuct: 45.224 against 45.218, which is the lag, not a disagreement. The INTERNAL
    // field is identical in both, since updateCoeffs touches only boundaries.
    const std::string upostPath = caseDir + "/" + dumpT + "/stage_Upost";
    const FieldData<vector> uPostFd = std::ifstream(upostPath.c_str()).good()
                                    ? readField<vector>(upostPath) : uAssFd;
    const std::vector<std::vector<vector>> uAssBnd = rawBoundary<vector>(uPostFd, patches);
    // flowRateInletVelocity: the invariant, and the control that it is not met by accident.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (f.U.boundary[pi]->bcCategory() != 9) continue;
        const scalar mdot = f.U.boundary[pi]->flowRateValue();

        // THE INVARIANT: createFields must leave the inlet carrying the case's prescribed mass flow.
        // OpenFOAM's constructor reaches updateCoeffs before compressibleCreatePhi.H builds phi, so the
        // flux is right from the very first iteration; brae seeded from `rhoInlet` and was 24% short.
        scalar sumPhi = 0.0;
        for (label i = 0; i < patches[pi].size; ++i) sumPhi += f.phi.boundary[pi][i];
        report("createFields' inlet carries the prescribed mass flow",
               std::fabs(sumPhi + mdot) / std::fabs(mdot), 1e-14);

        // THE CONTROL: `rhoInlet` is the fallback OpenFOAM uses only when no rho field is registered, and
        // sbMatched even labels it "Guess for rho". Building the inlet from it instead of the live patch
        // rho must give a MEASURABLY different flux -- otherwise this fixture cannot tell the two apart
        // and the assertion above would pass whichever density were used.
        const std::vector<scalar>& rv = f.rho.boundary[pi]->value();
        scalar sumRhoA = 0.0, sumGuessA = 0.0;
        for (label i = 0; i < patches[pi].size; ++i)
        {
            sumRhoA   += rv[i] * patches[pi].magSf[i];
            sumGuessA += scalar(0.5) * patches[pi].magSf[i];   // sbMatched's rhoInlet
        }
        const double guessFlux = std::fabs(mdot) * (sumRhoA / sumGuessA);
        check("control -- the rhoInlet guess would NOT deliver it",
              std::fabs(guessFlux - std::fabs(mdot)) / std::fabs(mdot) > 1e-2);
        std::printf("     %-40s live rho %+.9e   rhoInlet guess %+.9e\n",
                    "  (the flux each density gives)", sumPhi, -guessFlux);
    }

    // updateCoeffs() for flowRateInletVelocity, at the point OpenFOAM calls it: the momentum assembly.
    // OF holds the prescribed mass flow against the registered rho's PATCH values, which is the RELAXED
    // solver rho of the previous iteration -- dumped as stage_rhoU so this reads the same field OF read
    // rather than one reconstructed from p and T. Without this the inlet stays at the seed the
    // constructor built from `rhoInlet`, which sbMatched even labels "Guess for rho" and OpenFOAM ignores.
    {
        const std::string rhoUPath = caseDir + "/" + dumpT + "/stage_rhoU";
        if (std::ifstream(rhoUPath.c_str()).good())
        {
            const FieldData<scalar> rhoUFd = readField<scalar>(rhoUPath);
            const std::vector<std::vector<scalar>> rhoUBnd = rawBoundary<scalar>(rhoUFd, patches);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                f.U.boundary[pi]->updateFromDensity(rhoUBnd[pi]);
            }
        }
    }
    // updateCoeffs() for the FLUX-CONDITIONAL family, at the point OpenFOAM calls it: fvMatrix's
    // constructor calls psi.boundaryFieldRef().updateCoeffs(), and that is where inletOutlet reads
    // sign(phi) and sets its valueFraction. Until then brae's inletOutlet sits at its refValue -- which
    // is `inletValue`, and angledDuct's is (0 0 0) -- so the outlet read EXACTLY zero against
    // OpenFOAM's (24.5, 24.6, 0.086), a relative error of exactly 1.0.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        f.U.boundary[pi]->updateFromFlux(f.phi.boundary[pi]);
    }
    f.U.evaluateBoundary();

    // updateCoeffs() for the FREESTREAM family, at the point OpenFOAM calls it. freestreamVelocity is a
    // mixed patch whose valueFraction is recomputed from the current flow ANGLE, and every momentum
    // coefficient on such a patch is built from that fraction: valueInternalCoeffs = 1 - vf,
    // valueBoundaryCoeffs = vf*refValue. Adopting OpenFOAM's U VALUES below is not enough -- the value
    // can match exactly while the fraction behind the coefficients is still the seeded 0.5, which on
    // aerofoilNACA0012 read as iC 4.9 and bC 1.6e+03 with the values at 0.0.
    {
        std::vector<std::vector<vector>> Ub(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) Ub[pi] = f.U.boundary[pi]->value();
        updateMixedFreestream(f.U.boundary, Ub, patches);
        updateMixedFreestream(f.p.boundary, Ub, patches);
        f.p.evaluateBoundary();
    }

    // The momentum matrix's SOURCE and OFF-DIAGONALS, which A() and the boundary coefficients do not
    // cover and which are the whole of H() -- and therefore of HbyA and the pressure equation's flux.
    // Compared here because on aerofoilNACA0012 HbyA was 8.6e-06 out with rAU and both coefficient sets
    // exact, which means the disagreement had to be in a part of the matrix nothing was comparing.
    // The assembled matrix BEFORE relax(), so the assembly and the relaxation can be told apart. brae's
    // assembleUEqn relaxes internally, so this rebuilds it with relaxU = 1 -- which is NOT a no-op in
    // OpenFOAM either, but is the closest brae can get to "unrelaxed" without a second entry point.
    auto compareUnrelaxed = [&](const cpu::rhoSimple::RhoMomentumInput& base)
    {
        const std::string sp = caseDir + "/" + dumpT + "/stage_USrc0";
        const std::string dp = caseDir + "/" + dumpT + "/stage_UDiag0";
        if (!std::ifstream(sp.c_str()).good()) return;
        cpu::rhoSimple::RhoMomentumInput un = base;
        un.relaxU = 1.0;
        const FvVectorMatrix U0 = cpu::rhoSimple::assembleUEqn(f.U, un, m, g, patches);

        std::vector<char> isW4(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            if (patches[pi].type == "wall")
                for (label i = 0; i < patches[pi].size; ++i) isW4[patches[pi].faceCells[i]] = 1;

        const FieldData<vector> sFd = readField<vector>(sp);
        std::vector<scalar> a, b, aw, bw;
        for (label c = 0; c < nC; ++c)
        {
            vector mine = U0.source[c];
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                for (label i = 0; i < patches[pi].size; ++i)
                    if (patches[pi].faceCells[i] == c)
                    {
                        mine.x += U0.boundaryCoeffs[pi][i].x;
                        mine.y += U0.boundaryCoeffs[pi][i].y;
                        mine.z += U0.boundaryCoeffs[pi][i].z;
                    }
            const vector& th = sFd.internalField[c];
            const scalar mm[3] = { mine.x, mine.y, mine.z };
            const scalar tt[3] = { th.x, th.y, th.z };
            for (int kk = 0; kk < 3; ++kk)
            {
                a.push_back(mm[kk]); b.push_back(tt[kk]);
                if (isW4[c]) { aw.push_back(mm[kk]); bw.push_back(tt[kk]); }
            }
        }
        auto mg = [](const std::vector<scalar>& v)
        { return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0)); };
        std::printf("       %-40s %.6e   (walls %.6e)\n", "UEqn source BEFORE relax",
                    relL2(a, b), relL2(aw, bw));
        std::printf("         |brae| %.4e |OF| %.4e   walls: |brae| %.4e |OF| %.4e\n",
                    mg(a), mg(b), mg(aw), mg(bw));
        // What brae's own wall-cell source is MADE of, so the missing 100x can be attributed.
        {
            const std::vector<vector> expl = cpu::divDevReffExplicit(
                f.U, muEffInt, muEffBnd, m, g, patches, base.gradULimitK);
            std::vector<scalar> ex, bc;
            for (label c = 0; c < nC; ++c)
            {
                if (!isW4[c]) continue;
                ex.push_back(expl[c].x * g.V()[c]);
                ex.push_back(expl[c].y * g.V()[c]);
                ex.push_back(expl[c].z * g.V()[c]);
                vector s2v{0,0,0};
                for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    for (label i = 0; i < patches[pi].size; ++i)
                        if (patches[pi].faceCells[i] == c)
                        {
                            s2v.x += U0.boundaryCoeffs[pi][i].x;
                            s2v.y += U0.boundaryCoeffs[pi][i].y;
                            s2v.z += U0.boundaryCoeffs[pi][i].z;
                        }
                bc.push_back(s2v.x); bc.push_back(s2v.y); bc.push_back(s2v.z);
            }
            // and the non-orthogonal correction, by difference: the same matrix with `corrected` off.
            cpu::rhoSimple::RhoMomentumInput noc = base;
            noc.relaxU = 1.0;
            noc.correctedLaplacian = false;
            const FvVectorMatrix Uo = cpu::rhoSimple::assembleUEqn(f.U, noc, m, g, patches);
            std::vector<scalar> nonOrth;
            for (label c = 0; c < nC; ++c)
            {
                if (!isW4[c]) continue;
                nonOrth.push_back(U0.source[c].x - Uo.source[c].x);
                nonOrth.push_back(U0.source[c].y - Uo.source[c].y);
                nonOrth.push_back(U0.source[c].z - Uo.source[c].z);
            }
            std::printf("         walls, brae's parts: |dev2*V| %.4e   |boundaryCoeffs| %.4e   "
                        "|nonOrth| %.4e\n", mg(ex), mg(bc), mg(nonOrth));
        }

        if (std::ifstream(dp.c_str()).good())
        {
            const std::vector<scalar> ofD = rawInternal(readField<scalar>(dp), nC);
            std::vector<scalar> da, db, daw, dbw;
            for (label c = 0; c < nC; ++c)
            {
                da.push_back(U0.diag[c]); db.push_back(ofD[c]);
                if (isW4[c]) { daw.push_back(U0.diag[c]); dbw.push_back(ofD[c]); }
            }
            std::printf("       %-40s %.6e   (walls %.6e)\n", "UEqn diag BEFORE relax",
                        relL2(da, db), relL2(daw, dbw));
            // At a cell the porosity acts on, so the term can be seen rather than inferred.
            for (const auto& o : uopts.options)
            {
                if (!o.fixedCoeff || o.cells.empty()) continue;
                const label zc = o.cells[0];
                std::printf("         porosity cell %d: brae diag %.6e   OF diag %.6e   V %.6e   "
                            "V*tr(alpha) %.6e\n", (int)zc, U0.diag[zc], ofD[zc], g.V()[zc],
                            g.V()[zc] * (o.alpha.xx + o.alpha.yy + o.alpha.zz));
                // A cell the porosity does NOT act on. If the diagonal disagrees there too, the porosity
                // is not the cause and the momentum assembly itself is wrong on this case.
                std::vector<char> inZone(nC, 0);
                for (label zz : o.cells) inZone[zz] = 1;
                for (label c2 = 0; c2 < nC; ++c2)
                    if (!inZone[c2])
                    {
                        std::printf("         non-porosity cell %d: brae diag %.6e   OF diag %.6e   "
                                    "ratio %.4f\n", (int)c2, U0.diag[c2], ofD[c2],
                                    ofD[c2] != 0.0 ? (double)(U0.diag[c2]/ofD[c2]) : 0.0);
                        break;
                    }
                break;
            }
        }
    };

    auto compareMomentumMatrix = [&](const FvVectorMatrix& M, scalar gradLimitK)
    {
        std::printf("  1b. the momentum matrix's source and off-diagonals\n");
        const std::string sp = caseDir + "/" + dumpT + "/stage_USrc";
        if (std::ifstream(sp.c_str()).good())
        {
            const FieldData<vector> sFd = readField<vector>(sp);
            std::vector<scalar> a, b, aw, bw;
            std::vector<char> isW(nC, 0);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                if (patches[pi].type == "wall")
                    for (label i = 0; i < patches[pi].size; ++i) isW[patches[pi].faceCells[i]] = 1;
            for (label c = 0; c < nC; ++c)
            {
                vector mine = M.source[c];
                for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    for (label i = 0; i < patches[pi].size; ++i)
                        if (patches[pi].faceCells[i] == c)
                        {
                            mine.x += M.boundaryCoeffs[pi][i].x;
                            mine.y += M.boundaryCoeffs[pi][i].y;
                            mine.z += M.boundaryCoeffs[pi][i].z;
                        }
                const vector& th = sFd.internalField[c];
                const scalar mm[3] = { mine.x, mine.y, mine.z };
                const scalar tt[3] = { th.x, th.y, th.z };
                for (int kk = 0; kk < 3; ++kk)
                {
                    a.push_back(mm[kk]); b.push_back(tt[kk]);
                    if (isW[c]) { aw.push_back(mm[kk]); bw.push_back(tt[kk]); }
                }
            }
            report("UEqn source + boundaryCoeffs", relL2(a, b), 1e-11);
            std::printf("       %-40s %.6e\n", "same, on wall-adjacent cells", relL2(aw, bw));
        }
        // The off-diagonals are SCALARS even for a vector field, as in OpenFOAM: fvMatrix stores one
        // upper/lower per face and the components share them.
        auto face = [&](const char* file, const std::vector<scalar>& mine, const char* what)
        {
            const std::string path = caseDir + "/" + dumpT + "/" + file;
            if (!std::ifstream(path.c_str()).good()) return;
            const FieldData<scalar> fd = readField<scalar>(path);
            const std::size_t n = std::min(mine.size(), fd.internalField.size());
            report(what, relL2(std::vector<scalar>(mine.begin(), mine.begin() + n),
                               std::vector<scalar>(fd.internalField.begin(),
                                                   fd.internalField.begin() + n)), 1e-11);
        };
        // fvc::grad(U): the explicit dev2 term is built from it and from nothing else, so if the
        // gradient agrees the source disagreement is in the term's assembly instead.
        {
            const std::string gp = caseDir + "/" + dumpT + "/stage_UgradU";
            std::ifstream gin(gp.c_str());
            if (gin.good())
            {
                const std::string txt((std::istreambuf_iterator<char>(gin)),
                                      std::istreambuf_iterator<char>());
                const std::size_t at = txt.find("internalField");
                const std::size_t op = txt.find('(', at);
                std::vector<scalar> flat;
                const std::size_t want = std::size_t(9) * std::size_t(nC);
                for (std::size_t i2 = op; i2 < txt.size() && flat.size() < want; )
                {
                    const char ch = txt[i2];
                    if (ch == '(' || ch == ')' || std::isspace((unsigned char)ch)) { ++i2; continue; }
                    char* end = nullptr;
                    flat.push_back(std::strtod(txt.c_str() + i2, &end));
                    if (end == txt.c_str() + i2) break;
                    i2 = std::size_t(end - txt.c_str());
                }
                if (flat.size() >= want)
                {
                    // With the case's gradScheme applied, as OpenFOAM's fvc::grad(U) resolves it.
                    // Comparing brae's UNLIMITED gradient against OpenFOAM's cellLimited one read
                    // 1.9e+00 and measured the harness, not the code.
                    std::vector<tensor> mineG = fvc::gaussGrad(f.U, m, g, patches);
                    if (gradLimitK > 0.0) cpu::cellLimitGrad(mineG, f.U, gradLimitK, m, g, patches);
                    std::vector<char> isW2(nC, 0);
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                        if (patches[pi].type == "wall")
                            for (label i = 0; i < patches[pi].size; ++i) isW2[patches[pi].faceCells[i]] = 1;
                    std::vector<scalar> a, b, aw, bw;
                    for (label c = 0; c < nC; ++c)
                        for (int kk = 0; kk < 9; ++kk)
                        {
                            const scalar mv = reinterpret_cast<const scalar*>(&mineG[c])[kk];
                            const scalar tv = flat[std::size_t(9)*std::size_t(c) + std::size_t(kk)];
                            a.push_back(mv); b.push_back(tv);
                            if (isW2[c]) { aw.push_back(mv); bw.push_back(tv); }
                        }
                    report("fvc::grad(U) vs OpenFOAM", relL2(a, b), 1e-11);
                    std::printf("       %-40s %.6e\n", "same, on wall-adjacent cells", relL2(aw, bw));
                    // and split the REST: cells against a freestream patch, against the deep interior.
                    // The wall is exact, so whatever is left is one of these two, and they have very
                    // different causes -- a boundary value against a limiter or a base scheme.
                    {
                        std::vector<char> onFs(nC, 0);
                        for (std::size_t pi = 0; pi < patches.size(); ++pi)
                            if (patches[pi].type != "wall" && patches[pi].type != "empty")
                                for (label i = 0; i < patches[pi].size; ++i)
                                    onFs[patches[pi].faceCells[i]] = 1;
                        std::vector<scalar> af, bf2, ad, bd;
                        for (label c = 0; c < nC; ++c)
                            for (int kk = 0; kk < 9; ++kk)
                            {
                                const scalar mv = reinterpret_cast<const scalar*>(&mineG[c])[kk];
                                const scalar tv = flat[std::size_t(9)*std::size_t(c) + std::size_t(kk)];
                                if (isW2[c]) continue;
                                if (onFs[c]) { af.push_back(mv); bf2.push_back(tv); }
                                else         { ad.push_back(mv); bd.push_back(tv); }
                            }
                        // With MAGNITUDES beside them. A relative norm over a region where the field is
                        // near zero -- and an aerofoil's far field is uniform freestream, so grad(U) is
                        // near zero there -- reports a large number for a negligible absolute
                        // difference. Two readings this session were exactly that artifact.
                        auto mag2 = [](const std::vector<scalar>& v)
                        {
                            return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
                        };
                        std::printf("       %-40s %.6e   |brae| %.3e |OF| %.3e\n",
                                    "same, on freestream-adjacent cells", relL2(af, bf2),
                                    mag2(af), mag2(bf2));
                        std::printf("       %-40s %.6e   |brae| %.3e |OF| %.3e\n",
                                    "same, deep interior", relL2(ad, bd), mag2(ad), mag2(bd));
                    }
                }
            }
        }
        // The dev2 term, in two halves. divDevReffExplicit returns the DIVERGENCE (extensive, carrying
        // the cell volume as fvMatrix wants it); the tensor it is built from is compared first, so a
        // disagreement can be attributed to the field or to the div operator's boundary faces, not both.
        {
            auto tensorFile = [&](const char* file) -> std::vector<tensor>
            {
                const std::string path = caseDir + "/" + dumpT + "/" + file;
                std::ifstream tin(path.c_str());
                std::vector<tensor> out;
                if (!tin.good()) return out;
                const std::string txt((std::istreambuf_iterator<char>(tin)),
                                      std::istreambuf_iterator<char>());
                const std::size_t at = txt.find("internalField");
                const std::size_t op = txt.find('(', at);
                std::vector<scalar> flat;
                const std::size_t want = std::size_t(9) * std::size_t(nC);
                for (std::size_t i2 = op; i2 < txt.size() && flat.size() < want; )
                {
                    const char ch = txt[i2];
                    if (ch == '(' || ch == ')' || std::isspace((unsigned char)ch)) { ++i2; continue; }
                    char* end = nullptr;
                    flat.push_back(std::strtod(txt.c_str() + i2, &end));
                    if (end == txt.c_str() + i2) break;
                    i2 = std::size_t(end - txt.c_str());
                }
                if (flat.size() < want) return out;
                out.resize(nC);
                for (label c = 0; c < nC; ++c)
                    for (int kk = 0; kk < 9; ++kk)
                        reinterpret_cast<scalar*>(&out[c])[kk] =
                            flat[std::size_t(9)*std::size_t(c) + std::size_t(kk)];
                return out;
            };
            std::vector<char> isW3(nC, 0);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                if (patches[pi].type == "wall")
                    for (label i = 0; i < patches[pi].size; ++i) isW3[patches[pi].faceCells[i]] = 1;

            const std::vector<tensor> ofT = tensorFile("stage_UdevTensor");
            if (!ofT.empty())
            {
                std::vector<tensor> mineT = fvc::gaussGrad(f.U, m, g, patches);
                if (gradLimitK > 0.0) cpu::cellLimitGrad(mineT, f.U, gradLimitK, m, g, patches);
                std::vector<scalar> a, b, aw, bw;
                for (label c = 0; c < nC; ++c)
                {
                    const tensor t = muEffInt[c] * dev2(transpose(mineT[c]));
                    for (int kk = 0; kk < 9; ++kk)
                    {
                        const scalar mv = reinterpret_cast<const scalar*>(&t)[kk];
                        const scalar tv = reinterpret_cast<const scalar*>(&ofT[c])[kk];
                        a.push_back(mv); b.push_back(tv);
                        if (isW3[c]) { aw.push_back(mv); bw.push_back(tv); }
                    }
                }
                report("(rho*nuEff)*dev2(T(grad(U))) tensor", relL2(a, b), 1e-11);
                std::printf("       %-40s %.6e\n", "same, on wall-adjacent cells", relL2(aw, bw));

                // Its BOUNDARY values. fvc::div reads them on every boundary face, and they carry
                // gaussGrad's correction -- the wall-normal component of the gradient replaced by the
                // patch snGrad -- which is how wall shear enters the momentum source. Comparing only the
                // cells cannot see them, and the cells being exact while the source is 6.5e-03 out on
                // wall-adjacent cells is exactly the signature of a boundary-value difference.
                {
                    const std::vector<std::vector<tensor>> mineTb =
                        fvc::gradUBoundary(f.U, mineT, m, g, patches);
                    for (std::size_t pi = 0; pi < patches.size(); ++pi)
                    {
                        if (patches[pi].type == "empty" || !patches[pi].size) continue;
                        const std::vector<tensor> ofTb = readTensorPatch(
                            caseDir + "/" + dumpT + "/stage_UdevTensor",
                            patches[pi].name, patches[pi].size);
                        if (ofTb.empty()) continue;
                        std::vector<scalar> pa, pb;
                        for (label i = 0; i < patches[pi].size; ++i)
                        {
                            const tensor t = muEffBnd[pi][i] * dev2(transpose(mineTb[pi][i]));
                            for (int kk = 0; kk < 9; ++kk)
                            {
                                pa.push_back(reinterpret_cast<const scalar*>(&t)[kk]);
                                pb.push_back(reinterpret_cast<const scalar*>(&ofTb[i])[kk]);
                            }
                        }
                        std::printf("         dev2 tensor on %-14s %.6e   (%ld faces)\n",
                                    patches[pi].name.c_str(), relL2(pa, pb), (long)patches[pi].size);
                    }
                }
            }

            const std::string dp = caseDir + "/" + dumpT + "/stage_UdevDiv";
            if (std::ifstream(dp.c_str()).good())
            {
                const FieldData<vector> dFd = readField<vector>(dp);
                const std::vector<vector> mineD = cpu::divDevReffExplicit(
                    f.U, muEffInt, muEffBnd, m, g, patches, gradLimitK);
                std::vector<scalar> a, b, aw, bw;
                for (label c = 0; c < nC; ++c)
                {
                    // Both are PER UNIT VOLUME: divDevReffExplicit returns the divergence and the
                    // caller multiplies by V when it adds it to the source, so dividing here was wrong
                    // and made this comparison read 6.4e+07 -- a broken comparison, not a defect.
                    // NEGATED: divDevReff returns MINUS both its terms (see linearViscousStress_cpp),
                    // so brae's explicit part is the negative of OpenFOAM's bare fvc::div. Comparing
                    // them unsigned read exactly 2.0, which is relL2 of a field against its own negative
                    // -- a sign convention, not a disagreement.
                    const scalar mm[3] = { -mineD[c].x, -mineD[c].y, -mineD[c].z };
                    const vector& th = dFd.internalField[c];
                    const scalar tt[3] = { th.x, th.y, th.z };
                    for (int kk = 0; kk < 3; ++kk)
                    {
                        a.push_back(mm[kk]); b.push_back(tt[kk]);
                        if (isW3[c]) { aw.push_back(mm[kk]); bw.push_back(tt[kk]); }
                    }
                }
                report("div of that tensor", relL2(a, b), 1e-11);
                std::printf("       %-40s %.6e\n", "same, on wall-adjacent cells", relL2(aw, bw));
            }
        }
        face("stage_UUpper", M.upper, "UEqn upper (off-diagonal)");
        face("stage_ULower", M.lower, "UEqn lower (off-diagonal)");
    };

    // phi as OpenFOAM convects with. The momentum boundaryCoeffs are -phi_b*U_b, so with U's boundary
    // matching, anything left in them is the flux.
    {
        const std::string phiUPath = caseDir + "/" + dumpT + "/stage_phiU";
        if (std::ifstream(phiUPath.c_str()).good())
        {
            const FieldData<scalar> phiUFd = readField<scalar>(phiUPath);
            const std::vector<std::vector<scalar>> ofPhiB = rawBoundary<scalar>(phiUFd, patches);
            std::printf("  0. phi at the momentum assembly, against OpenFOAM's own\n");
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                double d = 0.0, n = 0.0, sa = 0.0, sb = 0.0;
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const double a = f.phi.boundary[pi][i];
                    const double b = ofPhiB[pi][i];
                    d += (a - b) * (a - b);
                    n += b * b;
                    sa += a;
                    sb += b;
                }
                std::printf("     %-18s %.6e   sum(brae) %+.6e  sum(OF) %+.6e\n",
                            patches[pi].name.c_str(), n > 0.0 ? std::sqrt(d / n) : std::sqrt(d), sa, sb);
            }
        }
    }
    {
        // Report how far brae's OWN boundary evaluation is from OpenFOAM's BEFORE overwriting it, so the
        // substitution is visible rather than silent. A measurement, not an assertion: the inlet BC is a
        // separate manifest component and is not what this gate is for.
        double d = 0.0, n = 0.0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            // Patches OpenFOAM writes no `value` for are skipped for the same reason they are not
            // adopted below: there is nothing to compare against, and zeros are not a measurement.
            {
                bool hv = false;
                for (const auto& b : uPostFd.boundary)
                    if (b.name == patches[pi].name && (b.valueUniform || !b.values.empty())) hv = true;
                if (!hv) continue;
            }
            // EMPTY patches are excluded because OpenFOAM excludes them: emptyFvPatch::size() is 0, so
            // its boundaryField has no entries and the written file carries no values. brae keeps the
            // mesh's face count there, which is harmless in the solver -- the base coefficients are zero
            // for the laplacian and the div contributions cancel between front and back, whose Sf are
            // equal and opposite with the same extrapolated value -- but comparing 16000 brae values
            // against nothing reads as an enormous disagreement that is not one.
            if (patches[pi].type == "empty") continue;
            const std::vector<vector>& bv = f.U.boundary[pi]->value();
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const vector& a = bv[i];
                const vector& b = uAssBnd[pi][i];
                d += (double)(a.x-b.x)*(a.x-b.x) + (double)(a.y-b.y)*(a.y-b.y) + (double)(a.z-b.z)*(a.z-b.z);
                n += (double)b.x*b.x + (double)b.y*b.y + (double)b.z*b.z;
            }
        }
        // With the gate's neutralised inlet this must be zero: both codes evaluate the same plain
        // fixedValue. It is asserted rather than printed, because if brae's boundary state differs from
        // OpenFOAM's then everything below is comparing two different problems.
        const double ub = n > 0.0 ? std::sqrt(d/n) : std::sqrt(d);
        report("brae's U boundary == OpenFOAM's", ub, 1e-14);
        // PER PATCH, with the declared type and magnitudes. A boundary disagreement is a boundary
        // CONDITION, and the only way to say which is to name it.
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            std::string ty = "(none written)";
            bool hv = false;
            for (const auto& b : uPostFd.boundary)
                if (b.name == patches[pi].name)
                {
                    ty = b.type;
                    if (b.valueUniform || !b.values.empty()) hv = true;
                }
            if (!hv || !patches[pi].size) { std::printf("       %-16s %-26s (no values written)\n",
                                                        patches[pi].name.c_str(), ty.c_str()); continue; }
            std::vector<scalar> a2, b2;
            const std::vector<vector>& bv2 = f.U.boundary[pi]->value();
            for (label i = 0; i < patches[pi].size; ++i)
            {
                a2.push_back(bv2[i].x); a2.push_back(bv2[i].y); a2.push_back(bv2[i].z);
                b2.push_back(uAssBnd[pi][i].x); b2.push_back(uAssBnd[pi][i].y);
                b2.push_back(uAssBnd[pi][i].z);
            }
            std::printf("       %-16s %-26s %.6e   brae[0]=(%.4e %.4e %.4e) OF[0]=(%.4e %.4e %.4e)\n",
                        patches[pi].name.c_str(), ty.c_str(), relL2(a2, b2),
                        bv2[0].x, bv2[0].y, bv2[0].z,
                        uAssBnd[pi][0].x, uAssBnd[pi][0].y, uAssBnd[pi][0].z);
        }
    }
    // Adopt OpenFOAM's evaluated U in place; GeometricField owns unique_ptr patch fields and cannot be
    // copied, and a copy is not wanted anyway -- the point is to assemble from OpenFOAM's exact input.
    if (!uAssFd.internalUniform && static_cast<label>(uAssFd.internalField.size()) == nC)
        f.U.internal = uAssFd.internalField;
    // ADOPT ONLY WHERE OPENFOAM SUPPLIES VALUES. A transform patch -- slip, noSlip, symmetry, wedge --
    // is written by OpenFOAM with its TYPE and no `value` entry, because its value is reconstructed by
    // evaluate() rather than stored. rawBoundary fills zeros for a patch with no value, so adopting
    // blindly assembles the momentum equation with U = 0 there instead of the projected velocity.
    // On angledDuct, whose porosityWall is `slip`, that corrupted the whole matrix: the U boundary read
    // 1.43e+00, the diagonal 9.93e-01 and rAU 8.23e-01, and the error was UNIFORM across the field
    // rather than confined to any patch -- which is what gave it away, since a real boundary defect
    // shows up in the cells that touch that boundary.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        bool hasValues = false;
        for (const auto& b : uPostFd.boundary)
            if (b.name == patches[pi].name && (b.valueUniform || !b.values.empty())) hasValues = true;
        if (hasValues) f.U.boundary[pi]->setValue(uAssBnd[pi]);
    }
    // AND RE-EVALUATE THE TRANSFORM PATCHES against the internal field just adopted. OpenFOAM's
    // transformFvPatchField coefficients call patchInternalField() LIVE at assembly; brae's take no
    // arguments and read a copy cached by evaluate(), so the cache has to be refreshed whenever the
    // internal field moves under it. These patches carry no OpenFOAM value to overwrite -- that is what
    // hasValues just established -- so re-evaluating them adopts more of OpenFOAM's state, not less.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (f.U.boundary[pi]->isSymmetry()) f.U.boundary[pi]->evaluate(f.U.internal);

    std::vector<std::vector<scalar>> rhoBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) rhoBnd[pi] = f.rho.boundary[pi]->value();

    // The case: `div(phi,U) bounded Gauss upwind`, `laplacianSchemes default Gauss linear corrected`,
    // relaxation U 0.9. Read rather than assumed where it is cheap to read.
    const FoamDict* eqnRelax = simpleDict ? nullptr : nullptr;
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;
    const scalar relaxU = re ? re->scalarOr("U", 1.0) : 1.0;
    (void)eqnRelax;
    std::printf("  relaxation U = %g\n", (double)relaxU);

    cpu::rhoSimple::RhoMomentumInput in;
    in.phi                = &f.phi.internal;
    in.phiBnd             = &f.phi.boundary;
    in.rho                = &f.rho.internal;
    in.rhoBnd             = &rhoBnd;
    in.muEff              = &muEffInt;
    in.muEffBnd           = &muEffBnd;
    in.relaxU             = relaxU;
    // PARSED, like the gradScheme above. sbMatched says `bounded Gauss upwind` for div(phi,U) while
    // aerofoilNACA0012 says `bounded Gauss linearUpwind limited` -- and linearUpwind puts a DEFERRED
    // GRADIENT CORRECTION in the source, which is largest where the gradient is, i.e. at the wall.
    // Stating `upwind` here left brae's wall-cell momentum source at 2.4e-02 against OpenFOAM's 2.5e+00.
    {
        const FieldDivScheme ds = parseFieldDivScheme(caseDir, "U");
        in.bounded = ds.bounded;
        in.scheme  = ds.linearUpwind ? cpu::rhoSimple::DivScheme::linearUpwind
                   : (ds.limited     ? cpu::rhoSimple::DivScheme::limitedLinear
                                     : cpu::rhoSimple::DivScheme::upwind);
        in.schemeCoeff = ds.twoByk;
        std::printf("  divSchemes: div(phi,U) bounded=%d linearUpwind=%d limitedLinear=%d\n",
                    (int)ds.bounded, (int)ds.linearUpwind, (int)ds.limited);
    }
    in.correctedLaplacian = true;
    // PARSED from the case, not stated. divDevRhoReff's explicit term is
    // fvc::div((rho*nuEff)*dev2(T(fvc::grad(U)))) and OpenFOAM resolves that grad against
    // gradSchemes/grad(U): sbMatched says plain `Gauss linear` (unlimited) while aerofoilNACA0012 says
    // `cellLimited Gauss linear 1`. Stating the fixture's own value was safe while there was one fixture
    // and silently ran the wrong discretisation the moment this binary was pointed at a second case --
    // the dev2 tensor read 2.3e-01 against OpenFOAM's purely from that.
    {
        DeviceSimpleControls sctl;
        parseFvSchemesControls(caseDir, sctl);
        in.gradULimitK = sctl.gradULimitK;
        std::printf("  gradSchemes: grad(U) cellLimited k = %g\n", (double)in.gradULimitK);
    }

    // The case's fvOptions, applied as UEqn.H applies them. Without this the gate assembles a momentum
    // equation with no porosity where the case has one, which is a different equation.
    uopts = cpu::fvOptions::read(caseDir, m);
    if (uopts.firstUnsupported().empty() && !uopts.empty())
    {
        in.fvOpts = &uopts;
        std::printf("  fvOptions: %zu option(s) applied\n", uopts.options.size());
        for (const auto& o : uopts.options)
            std::printf("     %-28s type=%-28s cells=%zu allCells=%d fixedCoeff=%d rhoRef=%g "
                        "tr(alpha)=%g\n", o.name.c_str(), o.type.c_str(), o.cells.size(),
                        (int)o.allCells, (int)o.fixedCoeff, (double)o.rhoRef,
                        (double)(o.alpha.xx + o.alpha.yy + o.alpha.zz));
    }

    // ---- 1. The assembled momentum matrix, against OpenFOAM's. ----
    std::printf("  1. UEqn assembled with the COMPRESSIBLE divDevRhoReff (mu_eff = rho*nu_eff)\n");
    FvVectorMatrix M = cpu::rhoSimple::assembleUEqn(f.U, in, m, g, patches);
    compareMomentumMatrix(M, in.gradULimitK);
    compareUnrelaxed(in);

    // rAU = 1/A(). Compared as rAU rather than A so the number is the one pEqn.H actually consumes.
    const std::vector<scalar> A = matrixA<vector>(M, m, g, patches);
    std::vector<scalar> rAU(nC);
    for (label c = 0; c < nC; ++c) rAU[c] = 1.0 / A[c];
    const std::vector<scalar> ofRAU = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_rAU"), nC);
    const double rauErr = relL2(rAU, ofRAU);
    report("rAU = 1/UEqn.A() vs OpenFOAM", rauErr, 1e-10);

    const std::vector<std::vector<vector>> ofUIC =
        rawBoundary<vector>(readField<vector>(caseDir + "/" + dumpT + "/stage_UIC"), patches);
    const std::vector<std::vector<vector>> ofUBC =
        rawBoundary<vector>(readField<vector>(caseDir + "/" + dumpT + "/stage_UBC"), patches);
    report("UEqn.internalCoeffs() vs OpenFOAM",
           relL2Boundary(M.internalCoeffs, ofUIC, patches), 1e-10);
    report("UEqn.boundaryCoeffs() vs OpenFOAM",
           relL2Boundary(M.boundaryCoeffs, ofUBC, patches), 1e-10);
    // Per-patch, so a failure names the patch and its BC rather than one number over the whole boundary.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!patches[pi].size) continue;
        std::vector<std::vector<vector>> a1(patches.size()), b1(patches.size());
        std::vector<std::vector<vector>> a2(patches.size()), b2(patches.size());
        std::vector<FvPatch> one(patches.size());
        for (std::size_t q = 0; q < patches.size(); ++q)
        {
            one[q] = patches[q];
            if (q != pi) one[q].size = 0;
        }
        std::printf("       %-18s iC %.3e   bC %.3e   (%d faces)\n",
                    patches[pi].name.c_str(),
                    relL2Boundary(M.internalCoeffs, ofUIC, one),
                    relL2Boundary(M.boundaryCoeffs, ofUBC, one),
                    (int)patches[pi].size);
    }
    // CONTROL FOR THE TRANSFORM COEFFICIENTS. A slip patch's VALUE is the tangential projection under
    // both the right and the wrong treatment, so every gate that compares fields passes either way --
    // and OpenFOAM writes no `value` for a transform patch, so there is nothing there to compare in the
    // first place. What separates them is only the four coefficient methods. This holds the value fixed
    // and swaps the patch field for a zeroGradient one, whose (1, 0, 0, 0) coefficients are exactly what
    // this class returned before: if that reproduces OpenFOAM's coefficients too, the numbers above are
    // measuring something other than the transform.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!patches[pi].size || !f.U.boundary[pi]->isSymmetry()) continue;
        std::vector<FvPatch> one(patches.size());
        for (std::size_t q = 0; q < patches.size(); ++q)
        {
            one[q] = patches[q];
            if (q != pi) one[q].size = 0;
        }
        const double right = relL2Boundary(M.boundaryCoeffs, ofUBC, one);

        auto zg = std::make_unique<ZeroGradientPatchField<vector>>(patches[pi]);
        zg->evaluate(f.U.internal);
        zg->setValue(f.U.boundary[pi]->value());       // same value, base coefficients
        f.U.boundary[pi].swap(reinterpret_cast<std::unique_ptr<fvPatchField<vector>>&>(zg));
        const FvVectorMatrix Z = cpu::rhoSimple::assembleUEqn(f.U, in, m, g, patches);
        const double wrong = relL2Boundary(Z.boundaryCoeffs, ofUBC, one);
        f.U.boundary[pi].swap(reinterpret_cast<std::unique_ptr<fvPatchField<vector>>&>(zg));

        std::printf("       control: %s as zeroGradient reads bC %.3e (transform reads %.3e)\n",
                    patches[pi].name.c_str(), wrong, right);
        check("  and the transform coefficients are what match, not the base ones",
              wrong > 1e3 * std::max(right, 1e-14));
    }

    // ---- 2. THE CONTROL. Assemble the INCOMPRESSIBLE way -- kinematic nu_eff in place of the dynamic
    //         mu_eff -- and require it to disagree. This is the one thing that distinguishes this
    //         component from simpleFoam's, so a gate that cannot see the difference is not gating it.
    std::printf("  2. control -- the INCOMPRESSIBLE form must NOT reproduce it\n");
    std::vector<scalar> nuEff(nC);
    for (label c = 0; c < nC; ++c) nuEff[c] = muEffInt[c] / f.rho.internal[c];
    std::vector<std::vector<scalar>> nuEffBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        nuEffBnd[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            nuEffBnd[pi][i] = muEffBnd[pi][i] / rhoBnd[pi][i];
    }
    cpu::rhoSimple::RhoMomentumInput wrong = in;
    wrong.muEff    = &nuEff;        // kinematic, as the incompressible solver would pass
    wrong.muEffBnd = &nuEffBnd;
    FvVectorMatrix W = cpu::rhoSimple::assembleUEqn(f.U, wrong, m, g, patches);
    const std::vector<scalar> WA = matrixA<vector>(W, m, g, patches);
    std::vector<scalar> wrAU(nC);
    for (label c = 0; c < nC; ++c) wrAU[c] = 1.0 / WA[c];
    const double wrongErr = relL2(wrAU, ofRAU);
    check("kinematic nu_eff disagrees with OpenFOAM", wrongErr > 1e-3);
    std::printf("     %-44s %.6e\n", "  (its rAU error, for the record)", wrongErr);
    check("and it is worse than the dynamic form", wrongErr > rauErr);

    // ---- 3. dynamicViscosity is exactly rho*nuEff, cells and boundary. ----
    std::printf("  3. mu_eff = rho*nu_eff\n");
    const std::vector<scalar> mu = cpu::rhoSimple::dynamicViscosity(f.rho.internal, nuEff);
    double muErr = 0.0;
    for (label c = 0; c < nC; ++c)
        muErr = std::max(muErr, std::fabs((double)mu[c] - (double)muEffInt[c])
                                / std::max(1e-300, std::fabs((double)muEffInt[c])));
    report("dynamicViscosity round-trips (max rel)", muErr, 1e-14);

    // ---- 4. REFUSALS. ----
    std::printf("  4. refusals\n");
    {
        cpu::rhoSimple::RhoMomentumInput bad = in;
        bad.hasMRF = true;
        bad.mrf    = nullptr;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleUEqn(f.U, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a declared-but-unsupplied MRF is refused", threw);
    }
    {
        cpu::rhoSimple::RhoMomentumInput bad = in;
        bad.hasFvOptions = true;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleUEqn(f.U, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a declared fvOptions is refused", threw);
    }
    {
        cpu::rhoSimple::RhoMomentumInput bad = in;
        bad.muEff = nullptr; bad.muEffBnd = nullptr;
        bad.rho   = nullptr; bad.rhoBnd   = nullptr;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleUEqn(f.U, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("no viscosity at all is refused", threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
