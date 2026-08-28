// rhoSimpleFoam's EEqn.H against REAL OpenFOAM's own assembled energy equation.
//
// THE ORACLE is tools/dumpPEqn's EEqn stage harness, which writes at SIMPLE iteration
// BRAE_DUMP_STAGE_ITER, AFTER EEqn.relax():
//
//   stage_Ekp       the kinetic-energy field OpenFOAM's OWN branch produced
//   stage_he        he as assembled
//   stage_eD        EEqn.D()                      diag + the boundary internalCoeffs
//   stage_eSrc      EEqn.source() + sum(boundaryCoeffs)   the full right-hand side
//   stage_alphaEff  turbulence->alphaEff()
//
// alphaEff is INJECTED, for the same reason muEff is injected into the momentum gate: the compressible
// turbulence closure is a separate manifest component, and a number covering both cannot be attributed
// to either. What is measured here is the ENERGY ASSEMBLY.
//
// WHAT IS ACTUALLY UNDER TEST. EEqn.H branches on he.name():
//
//     e  ->  Ekp = 0.5|U|^2 + p/rho
//     h  ->  K   = 0.5|U|^2
//
// On this fixture (sensibleInternalEnergy, so the `e` arm) p ~ 1.1e5 and rho ~ 0.38, so p/rho ~ 2.9e5
// against 0.5|U|^2 of order 10 -- four orders of magnitude. A solver using the wrong arm converges to a
// smooth, plausible, wrong temperature field, which is why THE CONTROL below builds the `h` arm on
// purpose and requires both the kinetic field and the assembled matrix to disagree with OpenFOAM.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fv_matrix_ops.cuh"
#include "rhoCreateFields_cpp.cuh"
#include "rhoSimpleFoam_cpp.cuh"   // effectiveTransport: the solver's OWN muEff/alphaEff
#include "rhoEEqn_cpp.cuh"
#include "thermo_model.cuh"   // thermoCpByCpv: alphaEff = CpByCpv*(alpha + alphat)

#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

// The staged fields carry OpenFOAM's own boundary types (extrapolatedCalculated among them), which brae's
// patch-field factory has no entry for. These read the parsed file directly: the staged VALUES are what
// is wanted, not a re-evaluated boundary condition.
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
            if (b.valueUniform) out[pi].assign(patches[pi].size, b.uniformValue);
            else if (static_cast<label>(b.values.size()) == patches[pi].size) out[pi] = b.values;
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

// D() = diag + the boundary internalCoeffs folded into their face cells (fvMatrix::D()).
static std::vector<scalar> matrixD(
    const FvScalarMatrix&       M,
    const std::vector<FvPatch>& patches)
{
    std::vector<scalar> D = M.diag;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            D[patches[pi].faceCells[i]] += M.internalCoeffs[pi][i];
    return D;
}

// source() + sum(boundaryCoeffs), which is what the harness writes as stage_eSrc.
static std::vector<scalar> matrixRhs(
    const FvScalarMatrix&       M,
    const std::vector<FvPatch>& patches)
{
    std::vector<scalar> r = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            r[patches[pi].faceCells[i]] += M.boundaryCoeffs[pi][i];
    return r;
}

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

    std::printf("rhoSimpleFoam EEqn vs OpenFOAM (%d cells)\n", (int)nC);

    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);
    std::printf("  energy variable: '%s'\n", f.heName.c_str());

    // OpenFOAM's own alphaEff and he, so this measures the assembly alone.
    const FieldData<scalar> aFd  = readField<scalar>(caseDir + "/" + dumpT + "/stage_alphaEff");
    const std::vector<scalar>              alphaEff    = rawInternal(aFd, nC);
    const std::vector<std::vector<scalar>> alphaEffBnd = rawBoundary<scalar>(aFd, patches);

    // U AFTER the momentum predictor. rhoSimpleFoam runs UEqn before EEqn, so the kinetic-energy source
    // is built from the just-solved velocity, NOT the field the iteration started with. The harness
    // writes it as stage_Upred; using the initial U instead makes 0.5|U|^2 wrong by the whole of the
    // momentum solve, which on this fixture is small next to p/rho and therefore easy to miss.
    const FieldData<vector> upFd = readField<vector>(caseDir + "/" + dumpT + "/stage_Upred");
    const std::vector<std::vector<vector>> upBnd = rawBoundary<vector>(upFd, patches);
    if (!upFd.internalUniform && static_cast<label>(upFd.internalField.size()) == nC)
        f.U.internal = upFd.internalField;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) f.U.boundary[pi]->setValue(upBnd[pi]);

    // he is built from the staged file itself, TYPES INCLUDED. Borrowing T's field and overwriting only
    // its values is not enough: a fixedValue patch's boundaryCoeffs come from its refValue, so a he built
    // that way carries T's 1000 K where the coefficients want he's 4.19e5 J/kg -- which reads as a source
    // error on every inlet-adjacent cell and nowhere else. The gate rewrites OpenFOAM's energy BC types
    // into brae-known equivalents first, and asserts that rewrite is exact.
    GeometricField<scalar> he =
        buildField<scalar>(readField<scalar>(caseDir + "/" + dumpT + "/stage_he"), patches, nC);
    he.evaluateBoundary();

    // ---- 1. THE BRANCH, on its own. ----
    std::printf("  1. the kinetic-energy source (EEqn.H branches on he.name())\n");
    const std::vector<scalar> ke = cpu::rhoSimple::kineticEnergy(f.heName, f.U, f.p, f.rho);
    const std::vector<scalar> ofEkp =
        rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_Ekp"), nC);
    report("Ekp = 0.5|U|^2 + p/rho vs OpenFOAM", relL2(ke, ofEkp), 1e-12);
    {
        // The BOUNDARY values matter as much as the internal ones: EEqn's explicit convection term picks
        // up phi_b*Ekp_b on every boundary face, so an inlet Ekp that is right in the cell and wrong on
        // the face still poisons the source there.
        const FieldData<scalar> ekpFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_Ekp");
        const std::vector<std::vector<scalar>> ofEkpB = rawBoundary<scalar>(ekpFd, patches);
        const std::vector<std::vector<scalar>> keB =
            cpu::rhoSimple::kineticEnergyBoundary(f.heName, f.U, f.p, f.rho, patches);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].size) continue;
            std::vector<scalar> a, b;
            for (label i = 0; i < patches[pi].size; ++i) { a.push_back(keB[pi][i]); b.push_back(ofEkpB[pi][i]); }
            std::printf("       Ekp on %-14s %.4e\n", patches[pi].name.c_str(), relL2(a, b));
        }
    }

    // The control for the branch: the OTHER arm must not reproduce it.
    const std::vector<scalar> keH = cpu::rhoSimple::kineticEnergy("h", f.U, f.p, f.rho);
    const double keHErr = relL2(keH, ofEkp);
    check("the 'h' arm (K = 0.5|U|^2) does NOT match", keHErr > 1e-3);
    std::printf("     %-44s %.6e\n", "  (its error, for the record)", keHErr);

    // ---- 2. The assembled energy matrix. ----
    std::printf("  2. EEqn assembled\n");
    cpu::rhoSimple::EnergyInput in;
    in.phi                = &f.phi.internal;
    in.phiBnd             = &f.phi.boundary;
    in.alphaEff           = &alphaEff;
    in.alphaEffBnd        = &alphaEffBnd;
    in.heName             = f.heName;
    in.boundedHe          = true;
    in.boundedKE          = true;
    in.schemeHe           = cpu::rhoSimple::DivScheme::upwind;
    in.schemeKE           = cpu::rhoSimple::DivScheme::upwind;
    in.correctedLaplacian = true;
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;
    in.relaxHe = re ? re->scalarOr(f.heName, 1.0) : 1.0;
    std::printf("     relaxation %s = %g\n", f.heName.c_str(), (double)in.relaxHe);

    // ---- 0. The three inputs the boundary terms are built from, each against OpenFOAM's own. With the
    //         fixture's real inlet the case runs at ~523 m/s and the boundary flux is ~140x what the
    //         neutralised inlet carries, so anything wrong here is visible only on the real case.
    {
        std::printf("  0. the boundary inputs, against OpenFOAM's own\n");

        // brae's OWN alphaEff, which this gate otherwise only INJECTS. Injecting it keeps the assembly
        // number attributable, but it also means the solver's transport has never been compared to
        // OpenFOAM's here -- and a driver that assembles exactly from an alphaEff of its own making can
        // still converge somewhere else. Asserted, because at this iteration both are built from the same
        // p, T, nut and alphat.
        {
            std::vector<scalar> mineMu, mineAlpha;
            std::vector<std::vector<scalar>> mineMuB, mineAlphaB;
            cpu::rhoSimple::effectiveTransport(f, patches, mineMu, mineMuB, mineAlpha, mineAlphaB);
            // Asserted against the LAMINAR half, thermo.alphahe(), because the turbulence model's nut
            // and alphat at this exact point are the model's own state and not something createFields
            // reconstructs. This is the half that was wrong: transportAlpha takes the VISCOSITY and the
            // driver was handing it the temperature, for an alphaEff six orders too large.
            const std::string lamPath = caseDir + "/" + dumpT + "/stage_alphaEffLam";
            if (std::ifstream(lamPath.c_str()).good())
            {
                const std::vector<scalar> ofLam = rawInternal(readField<scalar>(lamPath), nC);
                // The extraction has to INVERT the formula effectiveTransport actually uses, which is
                //     alphaEff = CpByCpv*(alpha + alphat)
                // so the laminar half is CpByCpv*alpha and what comes off is CpByCpv*alphat, not alphat.
                // Subtracting alphat alone leaves (CpByCpv - 1)*alphat behind -- 40% of it for air on
                // the sensibleInternalEnergy branch. That was invisible while brae's alphat was ZERO at
                // this point, which it was only because createFields did not run turbulence->validate();
                // once validate() set alphat = rho*nut/Prt the way OpenFOAM does, this check failed at
                // 8.10e-01 on a fixture it had always passed. The bound is unchanged.
                const scalar cpByCpv = thermoCpByCpv(f.thermo);
                std::vector<scalar> mineLam(nC);
                for (label c = 0; c < nC; ++c)
                    mineLam[c] = mineAlpha[c] - cpByCpv * (f.alphat.internal.empty()
                                                           ? 0.0 : f.alphat.internal[c]);
                report("brae's own laminar alphaEff == OpenFOAM's", relL2(mineLam, ofLam), 1e-12);
                std::printf("     %-34s brae %.6e   OpenFOAM %.6e\n", "  (the value itself)",
                            mineLam[0], ofLam[0]);
            }
            std::printf("     %-34s %.6e   (turbulent halves differ: this state has nut=0)\n",
                        "full alphaEff vs OpenFOAM's", relL2(mineAlpha, alphaEff));
        }
        auto splitCmp = [&](const std::vector<scalar>& mine, const std::vector<scalar>& theirs,
                            const char* what)
        {
            std::vector<char> onB(nC, 0);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                for (label i = 0; i < patches[pi].size; ++i) onB[patches[pi].faceCells[i]] = 1;
            double di = 0, ni = 0, db = 0, nb = 0;
            for (label c = 0; c < nC; ++c)
            {
                const double d = (double)mine[c] - (double)theirs[c];
                if (onB[c]) { db += d*d; nb += (double)theirs[c]*theirs[c]; }
                else        { di += d*d; ni += (double)theirs[c]*theirs[c]; }
            }
            std::printf("     %-34s interior %.6e   boundary cells %.6e\n", what,
                        ni > 0 ? std::sqrt(di/ni) : 0.0, nb > 0 ? std::sqrt(db/nb) : 0.0);
        };

        const std::string keDivPath = caseDir + "/" + dumpT + "/stage_keDiv";
        if (std::ifstream(keDivPath.c_str()).good())
        {
            // OpenFOAM's own div(phi,Ekp), the whole bounded Gauss upwind term. brae's helper carries the
            // cell VOLUME (the fvMatrix convention `source -= V*field`); OpenFOAM's is per unit volume.
            const std::vector<scalar> ofKeDiv = rawInternal(readField<scalar>(keDivPath), nC);
            const std::vector<scalar> mineExt = cpu::rhoSimple::kineticEnergyDivergence(
                f.U, f.p, f.rho, in, m, g, patches);
            std::vector<scalar> mine(nC);
            for (label c = 0; c < nC; ++c) mine[c] = mineExt[c] / g.V()[c];
            splitCmp(mine, ofKeDiv, "div(phi,Ekp) vs OpenFOAM's");
        }

        const std::string phiEPath = caseDir + "/" + dumpT + "/stage_phiE";
        if (std::ifstream(phiEPath.c_str()).good())
        {
            const FieldData<scalar> phiEFd = readField<scalar>(phiEPath);
            const std::vector<std::vector<scalar>> ofPhiB = rawBoundary<scalar>(phiEFd, patches);
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                double d = 0, n = 0;
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const double a = (*in.phiBnd)[pi][i], b = ofPhiB[pi][i];
                    d += (a-b)*(a-b); n += b*b;
                }
                std::printf("     phi on %-26s %.6e\n", patches[pi].name.c_str(),
                            n > 0 ? std::sqrt(d/n) : std::sqrt(d));
            }
        }

        // he's own boundary values, which the convection and laplacian coefficients are built from.
        const FieldData<scalar> heFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_he");
        const std::vector<std::vector<scalar>> ofHeB = rawBoundary<scalar>(heFd, patches);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& hv = he.boundary[pi]->value();
            double d = 0, n = 0;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const double a = hv[i], b = ofHeB[pi][i];
                d += (a-b)*(a-b); n += b*b;
            }
            std::printf("     he  on %-26s %.6e  (%s)\n", patches[pi].name.c_str(),
                        n > 0 ? std::sqrt(d/n) : std::sqrt(d),
                        he.boundary[pi]->fixesValue() ? "fixesValue" : "not-fixesValue");
        }
    }

    FvScalarMatrix E = cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, in, m, g, patches);
    const std::vector<scalar> ofD   = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_eD"), nC);
    const std::vector<scalar> ofSrc = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_eSrc"), nC);
    // THE SYSTEM BEFORE relax(), separately. fvMatrix::relax clamps the diagonal to
    // max(|D|, sumMagOffDiag) before dividing by alpha, and at a wall cell the wall face contributes
    // NOTHING to the diagonal when he is gradientEnergy -- so sumOff wins, the relaxed D lands on
    // sumOff/alpha, and it carries no trace of the unrelaxed D0. A wrong D0 is then invisible in D and
    // visible only through `S += (D - D0)*psi`. That is exactly the shape angledDuct showed: D agreeing
    // to eleven digits at cell 629 while the source was 7.3e-05 out. Comparing the unrelaxed system
    // directly is the only way to see which of the two it is.
    if (std::ifstream((caseDir + "/" + dumpT + "/stage_eD0").c_str()).good())
    {
        cpu::rhoSimple::EnergyInput un = in;
        un.relaxHe = 1.0;                     // no relaxation: D == D0 and source == the raw right-hand side
        const FvScalarMatrix E0 = cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, un, m, g, patches);
        const std::vector<scalar> ofD0 = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_eD0"), nC);
        const std::vector<scalar> ofS0 = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_eSrc0"), nC);
        std::printf("     EEqn.D()      BEFORE relax                %.6e\n", relL2(matrixD(E0, patches), ofD0));
        std::printf("     EEqn source   BEFORE relax                %.6e\n", relL2(matrixRhs(E0, patches), ofS0));
        std::vector<char> isB2(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i) isB2[patches[pi].faceCells[i]] = 1;
        const std::vector<scalar> mD0 = matrixD(E0, patches);
        double db2 = 0.0, nb2 = 0.0, di2 = 0.0, ni2 = 0.0;
        for (label c = 0; c < nC; ++c)
        {
            const double d = (double)mD0[c] - (double)ofD0[c], r = (double)ofD0[c];
            if (isB2[c]) { db2 += d*d; nb2 += r*r; } else { di2 += d*d; ni2 += r*r; }
        }
        std::printf("       D0 split: boundary cells %.4e   interior %.4e\n",
                    nb2 > 0 ? std::sqrt(db2/nb2) : 0.0, ni2 > 0 ? std::sqrt(di2/ni2) : 0.0);
        const std::vector<scalar> mS0 = matrixRhs(E0, patches);
        double sb = 0.0, snb = 0.0, si = 0.0, sni = 0.0;
        for (label c = 0; c < nC; ++c)
        {
            const double d = (double)mS0[c] - (double)ofS0[c], r = (double)ofS0[c];
            if (isB2[c]) { sb += d*d; snb += r*r; } else { si += d*d; sni += r*r; }
        }
        std::printf("       src0 split: boundary cells %.4e   interior %.4e\n",
                    snb > 0 ? std::sqrt(sb/snb) : 0.0, sni > 0 ? std::sqrt(si/sni) : 0.0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].size) continue;
            double dp = 0.0, np = 0.0;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const double d = (double)mS0[c] - (double)ofS0[c];
                dp += d*d; np += (double)ofS0[c]*(double)ofS0[c];
            }
            std::printf("         src0 on %-14s %.4e\n", patches[pi].name.c_str(),
                        np > 0 ? std::sqrt(dp/np) : 0.0);
        }
        label w0 = 0; double wd0 = -1.0;
        for (label c = 0; c < nC; ++c)
        {
            const double d = std::fabs((double)mS0[c] - (double)ofS0[c]);
            if (d > wd0) { wd0 = d; w0 = c; }
        }
        std::printf("       worst src0 cell %d: brae %.10e  OF %.10e  diff %.4e  (bnd=%d)\n",
                    (int)w0, (double)mS0[w0], (double)ofS0[w0], wd0, (int)isB2[w0]);
        {
            // The PIECES. At a wall phi = 0 and the he gradient is 0, so every boundary coefficient
            // there should vanish and the whole source should be the kinetic term -div(phi,Ekp)*V.
            // Printing them separately says whether that is what brae actually built.
            double bcSum = 0.0;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                for (label i = 0; i < patches[pi].size; ++i)
                    if (patches[pi].faceCells[i] == w0) bcSum += (double)E0.boundaryCoeffs[pi][i];
            const std::vector<scalar> keD2 = cpu::rhoSimple::kineticEnergy(f.heName, f.U, f.p, f.rho);
            std::printf("         pieces at %d: brae source() %.10e   sum(bC) %.10e   V %.6e\n",
                        (int)w0, (double)E0.source[w0], bcSum, (double)g.V()[w0]);
            std::printf("         kineticEnergy field here %.10e   (times V) %.10e\n",
                        (double)keD2[w0], (double)keD2[w0] * (double)g.V()[w0]);
        }
    }

    const double dErr = relL2(matrixD(E, patches), ofD);
    const double sErr = relL2(matrixRhs(E, patches), ofSrc);
    report("EEqn.D() vs OpenFOAM", dErr, 1e-10);
    {
        // Decomposition, so a source gap can be attributed to a term instead of guessed at.
        const std::vector<scalar> mine = matrixRhs(E, patches);
        double nm = 0.0, no = 0.0, nd = 0.0, nke = 0.0, nrel = 0.0;
        const std::vector<scalar> keD = cpu::rhoSimple::kineticEnergy(f.heName, f.U, f.p, f.rho);
        for (label c = 0; c < nC; ++c)
        {
            nm += (double)mine[c]*mine[c];
            no += (double)ofSrc[c]*ofSrc[c];
            const double d = (double)mine[c] - (double)ofSrc[c];
            nd += d*d;
            nrel += (double)ofD[c]*(double)he.internal[c]*(double)ofD[c]*(double)he.internal[c];
            nke += (double)keD[c]*keD[c];
        }
        std::printf("     |brae src| %.4e   |OF src| %.4e   |diff| %.4e   |D*he| %.4e\n",
                    std::sqrt(nm), std::sqrt(no), std::sqrt(nd), std::sqrt(nrel));
        const std::vector<scalar> kd = cpu::rhoSimple::kineticEnergyDivergence(f.U, f.p, f.rho, in, m, g, patches);
        cpu::rhoSimple::EnergyInput ub = in; ub.boundedKE = false;
        const std::vector<scalar> kdu = cpu::rhoSimple::kineticEnergyDivergence(f.U, f.p, f.rho, ub, m, g, patches);
        double nk = 0.0, nku = 0.0;
        for (label c = 0; c < nC; ++c) { nk += (double)kd[c]*kd[c]; nku += (double)kdu[c]*kdu[c]; }
        std::printf("     |KE div| bounded %.4e   unbounded %.4e\n", std::sqrt(nk), std::sqrt(nku));
        // WHERE does the difference live? If it sits on boundary-adjacent cells it is a boundary
        // treatment; if it is spread through the interior it is a volumetric term.
        std::vector<char> isBnd(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i) isBnd[patches[pi].faceCells[i]] = 1;
        double db = 0.0, di = 0.0; long nb = 0;
        for (label c = 0; c < nC; ++c)
        {
            const double d = (double)mine[c] - (double)ofSrc[c];
            if (isBnd[c]) { db += d*d; ++nb; } else di += d*d;
        }
        std::printf("     diff on %ld boundary cells %.4e   on %ld interior cells %.4e\n",
                    nb, std::sqrt(db), (long)nC - nb, std::sqrt(di));
        // THE WORST CELL, with the pieces the difference could come from. A per-patch L2 says where the
        // disagreement lives; it cannot say what it is made of, and the source at a wall cell has no
        // boundary contribution at all here (phi = 0, gradient = 0), so the arithmetic has to be read
        // rather than inferred.
        {
            label wc = 0; double wd = -1.0;
            for (label c = 0; c < nC; ++c)
            {
                const double d = std::fabs((double)mine[c] - (double)ofSrc[c]);
                if (isBnd[c] && d > wd) { wd = d; wc = c; }
            }
            std::string on;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
                for (label i = 0; i < patches[pi].size; ++i)
                    if (patches[pi].faceCells[i] == wc)
                    { if (on.find(patches[pi].name) == std::string::npos) on += patches[pi].name + " "; }
            std::printf("       worst boundary cell %d on [%s]\n", (int)wc, on.c_str());
            std::printf("         brae src %.10e   OF src %.10e   diff %.4e\n",
                        (double)mine[wc], (double)ofSrc[wc], wd);
            std::printf("         brae D   %.10e   OF D   %.10e   he %.10e\n",
                        (double)E.diag[wc], (double)ofD[wc], (double)he.internal[wc]);
            std::printf("         implied dD0 = diff/he %.4e   (relative to D %.4e)\n",
                        wd / (double)he.internal[wc],
                        wd / (double)he.internal[wc] / (double)ofD[wc]);
        }
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].size) continue;
            double dp = 0.0;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const double d = (double)mine[c] - (double)ofSrc[c];
                dp += d*d;
            }
            std::printf("       %-18s diff %.4e  (he BC in brae: %s)\n", patches[pi].name.c_str(),
                        std::sqrt(dp), he.boundary[pi]->fixesValue() ? "fixesValue" : "not-fixesValue");
        }
    }
    report("EEqn source + boundaryCoeffs vs OpenFOAM", sErr, 1e-10);

    // ---- 3. THE CONTROL. ----
    //
    // NOT on the assembled source, and the reason is worth stating: `div(phi,Ekp)` is `bounded`, and
    // boundedConvectionScheme::fvcDiv subtracts surfaceIntegrate(phi)*vf. At iteration 1 continuity is
    // nearly satisfied, so that subtraction very nearly cancels the divergence itself -- measured here,
    // |KE div| falls from 1.4e+01 unbounded to 1.3e-04 bounded. A control asserting that the wrong ARM
    // changes the assembled source would therefore be asserting something this state cannot show, and
    // would pass or fail on rounding. The branch is instead gated where it is actually observable: the
    // kinetic-energy FIELD against OpenFOAM's own stage_Ekp (check 1), and the divergence the two arms
    // produce.
    std::printf("  3. control -- the two arms must produce different equations\n");
    cpu::rhoSimple::EnergyInput wrong = in;
    wrong.heName = (f.heName == "e") ? "h" : "e";
    // Compared UNBOUNDED. The bounded correction removes the near-uniform part of the field, and at this
    // state p/rho -- the entire difference between the arms -- IS near-uniform, so the bounded term is
    // insensitive to the branch here (measured: |KE div| 1.3e-04 bounded against 1.4e+01 unbounded, and
    // the two arms' bounded divergences agree to better than 1e-3 of that). That insensitivity is a
    // property of iteration 1 on this fixture, not of the discretisation, so the control is taken on the
    // convection term the scheme actually builds, before the bounded subtraction.
    cpu::rhoSimple::EnergyInput ue = in;    ue.boundedKE    = false;
    cpu::rhoSimple::EnergyInput uh = wrong; uh.boundedKE    = false;
    const std::vector<scalar> kdE = cpu::rhoSimple::kineticEnergyDivergence(f.U, f.p, f.rho, ue, m, g, patches);
    const std::vector<scalar> kdH = cpu::rhoSimple::kineticEnergyDivergence(f.U, f.p, f.rho, uh, m, g, patches);
    double na = 0.0, nb2 = 0.0;
    for (label c = 0; c < nC; ++c) { na += (double)(kdE[c]-kdH[c])*(kdE[c]-kdH[c]); nb2 += (double)kdE[c]*kdE[c]; }
    check("the arms build different convection terms", std::sqrt(na) > 1e-3 * std::sqrt(nb2));
    std::printf("     %-44s %.4e vs %.4e\n", "  |KE div| unbounded vs |arm difference|",
                std::sqrt(nb2), std::sqrt(na));
    // And the wrong arm's FIELD must not reproduce OpenFOAM's -- asserted in check 1, restated here as
    // the thing that actually distinguishes the two implementations.
    check("wrong arm's field is far from OpenFOAM's", keHErr > 1e-3);

    // ---- 4. REFUSALS. ----
    std::printf("  4. refusals\n");
    {
        cpu::rhoSimple::EnergyInput bad = in;
        bad.heName = "absoluteEnthalpy";
        bool threw = false;
        std::string msg;
        try { (void)cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, bad, m, g, patches); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        check("an energy variable that is neither e nor h is refused", threw);
        check("and the refusal names it", msg.find("absoluteEnthalpy") != std::string::npos);
    }
    {
        cpu::rhoSimple::EnergyInput bad = in;
        bad.hasMRF = true;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a declared MRF is refused", threw);
    }
    {
        cpu::rhoSimple::EnergyInput bad = in;
        bad.schemeKE = cpu::rhoSimple::DivScheme::LUST;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("an unported convection scheme is refused", threw);
    }
    {
        cpu::rhoSimple::EnergyInput bad = in;
        bad.alphaEff = nullptr; bad.alphaEffBnd = nullptr;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a missing alphaEff is refused", threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
