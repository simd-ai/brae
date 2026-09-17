// brae's interFoam against REAL OpenFOAM's, on damBreak, field by field.
//
// THE ORACLE IS OpenFOAM'S OWN WRITTEN STATE after exactly N identical time steps. The shell gate
// rewrites the case with `adjustTimeStep no` and a fixed deltaT before running either solver, because
// an adaptive step makes the two take DIFFERENT steps the moment their Courant numbers differ at all,
// and then every field is compared at a different physical time -- a disagreement that looks like a
// discretisation error and is actually a clock.
//
// WHAT AGREEMENT CAN AND CANNOT MEAN HERE. brae and OpenFOAM do not share a linear solver: OpenFOAM's
// p_rgh runs PCG/DIC at tolerance 1e-07 relTol 0.05, and brae's host path runs DILU-preconditioned
// BiCGStab. So the pressure fields differ by the two solvers' residuals whatever the discretisation
// does, and a tolerance chosen tighter than that would be gating the solver rather than the port. The
// bounds below are therefore stated per field, with the reason each one is what it is.
//
// alpha IS THE FIELD THAT MATTERS. It is bounded, it is conserved, and it is what a VoF solver is for;
// it is also the field least affected by the linear-solver difference, because MULES is explicit.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include "device_gate_finite.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;

void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

struct Diff { scalar linf = 0; scalar l2 = 0; scalar refMax = 0; };

Diff compare(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    Diff d;
    scalar s2 = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        const scalar e = std::fabs(a[i] - b[i]);
        d.linf = std::fmax(d.linf, e);
        s2 += e*e;
        d.refMax = std::fmax(d.refMax, std::fabs(b[i]));
    }
    d.l2 = std::sqrt(s2 / std::fmax(scalar(1), static_cast<scalar>(a.size())));
    return d;
}
}   // namespace

int main(int argc, char** argv)
{
    std::printf("== brae interFoam vs OpenFOAM interFoam: damBreak ==\n");
    if (argc < 5)
    {
        std::printf("  SKIP: usage: %s <caseDir> <startDir> <ofTimeDir> <nSteps>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1], startDir = argv[2], ofDir = argv[3];
    const label nSteps = static_cast<label>(std::atol(argv[4]));

    if (!std::filesystem::exists(ofDir + "/alpha.water"))
    {
        std::printf("  SKIP: OpenFOAM wrote no alpha.water in %s\n", ofDir.c_str());
        return 77;
    }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    // brae, the same N steps, keeping the fields.
    InterFields fin;
    const RunReport r = runInterFoam(caseDir, startDir, m, g, patches, nSteps, /*verbose=*/true, &fin);
    check("brae ran the same number of steps", r.steps == nSteps);

    auto readCells = [&](const std::string& path)
    {
        const FieldData<scalar> fd = readField<scalar>(path);
        std::vector<scalar> v;
        if (fd.internalUniform) v.assign(static_cast<std::size_t>(nC), fd.internalUniformValue);
        else                    v = fd.internalField;
        return v;
    };

    const std::vector<scalar> ofAlpha = readCells(ofDir + "/alpha.water");
    const std::vector<scalar> a0      = readCells(startDir + "/alpha.water");
    check("OpenFOAM's alpha has one value per cell", ofAlpha.size() == static_cast<std::size_t>(nC));

    // HOW FAR THE FIELD MOVED is the yardstick. An absolute tolerance on alpha would be met by a
    // solver that did nothing at all -- 2268 cells of which only a few dozen are near the interface,
    // and five steps of 1e-4 s move it by a few thousandths. The comparison is therefore stated as a
    // fraction of OpenFOAM's OWN change over the same interval.
    scalar ofMoved = 0;
    for (label c = 0; c < nC; ++c) ofMoved = std::fmax(ofMoved, std::fabs(ofAlpha[c] - a0[c]));
    std::printf("  OpenFOAM's alpha moved by up to %.6f over %ld steps\n", (double)ofMoved, (long)nSteps);
    check("OpenFOAM's own run actually moved the interface, so there is something to compare",
          ofMoved > scalar(1e-6));

    // BEFORE any fmax below: compare() accumulates Linf with std::fmax, which DROPS a NaN, so a brae
    // field gone non-finite would read Linf 0 and pass every bound in this file. See
    // tests/device_gate_finite.cuh for the run that printed four green checks on 2268 NaN cells.
    failures += brae::gatecheck::nonFinite("brae alpha", fin.alpha1.internal);
    failures += brae::gatecheck::nonFinite("brae p_rgh", fin.p_rgh.internal);
    failures += brae::gatecheck::nonFinite("brae U", fin.U.internal);
    const Diff dAlpha = compare(fin.alpha1.internal, ofAlpha);
    std::printf("  alpha:  Linf %.4e  L2 %.4e   (OpenFOAM moved %.4e, so Linf is %.2f%% of the change)\n",
                (double)dAlpha.linf, (double)dAlpha.l2, (double)ofMoved,
                (double)(100*dAlpha.linf/ofMoved));

    // THE CONTROL: doing nothing would differ from OpenFOAM by exactly ofMoved. Any claim about alpha
    // is worthless unless brae is much closer than that.
    const Diff dNothing = compare(a0, ofAlpha);
    std::printf("  ...against %.4e for a solver that did NOTHING\n", (double)dNothing.linf);
    check("brae's alpha is far closer to OpenFOAM's than the initial field is",
          dAlpha.linf < scalar(0.05) * dNothing.linf);
    // MEASURED 2.2337e-12. It was 3.43e-09, then 1.24e-08 once the case's own p_rgh tolerance was
    // read, and every one of those was the PRESSURE SOLVER: brae ran PBiCGStab where damBreak names
    // PCG with DIC, and applied p_rghFinal's relTol to all three correctors where pEqn.H selects
    // p_rgh (relTol 0.05) for the first two. With brae::pcg and the per-corrector selection it is
    // 2.2e-12 -- four orders -- and the bound follows it, at about 20x for another compiler's
    // contraction. A gate that would pass at 5% is not measuring the discretisation, it is measuring
    // that something happened; this one would not pass at the old 1e-8.
    check("...and agrees with it to 5e-11 absolute, which is the discretisation and not the control",
          dAlpha.linf < scalar(5e-11));

    // p_rgh and U. BOTH codes now run the case's own PCG with DIC -- brae::pcg is a transcription of
    // lduMatrix PCG + DICPreconditioner, gated in tests/test_pcg.cu -- and both select p_rgh for the
    // first two correctors and p_rghFinal for the last. So the two stop at the same residual, and what
    // is left is floating point amplified through a solve that stops at relTol 0.05, not two solvers'
    // different stopping points. The bounds are relative to each field's own scale.
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    std::printf("  p_rgh:  Linf %.4e  L2 %.4e   (|p_rgh| up to %.4e)\n",
                (double)dP.linf, (double)dP.l2, (double)dP.refMax);
    std::printf("          relative %.3e\n", (double)(dP.linf/std::fmax(dP.refMax, scalar(1e-30))));
    // MEASURED 7.194e-10 relative; 3.35e-06 on PBiCGStab. The old comment here said agreement much
    // below 2.3e-06 "would be luck rather than a claim" -- that was the substituted solver talking.
    check("p_rgh agrees with OpenFOAM's to 1e-8 relative, both running the case's PCG+DIC",
          dP.linf < scalar(1e-8) * std::fmax(dP.refMax, scalar(1e-12)));

    const FieldData<vector> ofUfd = readField<vector>(ofDir + "/U");
    std::vector<vector> ofU;
    if (ofUfd.internalUniform) ofU.assign(static_cast<std::size_t>(nC), ofUfd.internalUniformValue);
    else                       ofU = ofUfd.internalField;
    scalar uLinf = 0, uRef = 0;
    for (label c = 0; c < nC; ++c)
    {
        const vector& a = fin.U.internal[c];
        const vector& b = ofU[c];
        uLinf = std::fmax(uLinf, std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z)));
        uRef  = std::fmax(uRef,  std::sqrt(b.x*b.x + b.y*b.y + b.z*b.z));
    }
    std::printf("  U:      Linf %.4e            (|U| up to %.4e)\n", (double)uLinf, (double)uRef);
    std::printf("          relative %.3e\n", (double)(uLinf/std::fmax(uRef, scalar(1e-30))));
    // MEASURED 1.966e-08 relative; 9.76e-06 on PBiCGStab. U is rebuilt from the pressure flux, so it
    // carries whatever p_rgh's solve leaves.
    check("U agrees with OpenFOAM's to 4e-7 relative, for the same reason",
          uLinf < scalar(4e-7) * std::fmax(uRef, scalar(1e-12)));

    // ...and the conserved quantity, which neither solver's linear tolerance can move.
    scalar ofMass = 0, a0Mass = 0;
    for (label c = 0; c < nC; ++c) { ofMass += ofAlpha[c]*g.V()[c]; a0Mass += a0[c]*g.V()[c]; }
    std::printf("  water volume: initial %.10e, OpenFOAM %.10e, brae %.10e\n",
                (double)a0Mass, (double)ofMass, (double)r.alphaMass);
    check("OpenFOAM conserves the water too (a closed domain)",
          std::fabs(ofMass - a0Mass)/a0Mass < scalar(1e-6));
    check("brae's water volume agrees with OpenFOAM's to 1e-9 -- both conserve, so this IS exact",
          std::fabs(r.alphaMass - ofMass)/ofMass < scalar(1e-9));

    std::printf("test_inter_dambreak_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
