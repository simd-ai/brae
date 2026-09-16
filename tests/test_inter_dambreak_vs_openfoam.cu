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
    // MEASURED 3.43e-09, i.e. one part in a million of OpenFOAM's own change. The bound is set an
    // order of magnitude above that rather than at the 5% the control needs -- a gate that would pass
    // at 5% is not measuring the discretisation, it is measuring that something happened.
    check("...and agrees with it to 1e-7 absolute, which is the discretisation and not the control",
          dAlpha.linf < scalar(1e-7));

    // p_rgh and U. NEITHER solver's linear solve is the other's -- OpenFOAM runs PCG/DIC at tolerance
    // 1e-07 relTol 0.05 on p_rgh, brae's host path runs DILU-preconditioned BiCGStab -- so these
    // fields carry the two solvers' residuals whatever the discretisation does. relTol 0.05 in
    // particular means OpenFOAM stops when the residual has fallen by a factor of 20, which is a long
    // way from converged. The bounds are stated relative to each field's own scale for that reason.
    const std::vector<scalar> ofPrgh = readCells(ofDir + "/p_rgh");
    const Diff dP = compare(fin.p_rgh.internal, ofPrgh);
    std::printf("  p_rgh:  Linf %.4e  L2 %.4e   (|p_rgh| up to %.4e)\n",
                (double)dP.linf, (double)dP.l2, (double)dP.refMax);
    std::printf("          relative %.3e\n", (double)(dP.linf/std::fmax(dP.refMax, scalar(1e-30))));
    // MEASURED 2.3e-06 relative. OpenFOAM's p_rgh runs at relTol 0.05 -- it stops when the residual
    // has fallen by a factor of 20 -- so agreement much below this would be luck rather than a claim.
    check("p_rgh agrees with OpenFOAM's to 1e-4 relative, inside the two linear solves' own residuals",
          dP.linf < scalar(1e-4) * std::fmax(dP.refMax, scalar(1e-12)));

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
    // MEASURED 3.3e-06 relative. U is rebuilt from the pressure flux, so it carries p_rgh's residual.
    check("U agrees with OpenFOAM's to 1e-4 relative, the same bound and for the same reason",
          uLinf < scalar(1e-4) * std::fmax(uRef, scalar(1e-12)));

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
