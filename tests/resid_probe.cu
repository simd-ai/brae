// Does OpenFOAM's converged solution satisfy brae's _cpp equations on this mesh?
//
// The residual oracle, applied to the momentum and pressure equations directly: no driver, no iteration,
// and -- crucially -- nuEff is taken FROM OPENFOAM's own nut rather than modelled, so the turbulence
// model is removed from the comparison. That splits the question:
//
//   both residuals small  -> the momentum/pressure transcription is right on this mesh, and the
//                            disagreement must come from the turbulence model producing a different nut
//   pressure residual big -> the pressure assembly is wrong on this mesh
//
// Run: resid_probe <caseDir> <ofConvergedTimeDir>
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "solve_vector.cuh"
#include "pbicgstab.cuh"
#include "fv_matrix_ops.cuh"
#include "simpleFoam_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "createFields_cpp.cuh"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <ofTimeDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    U.evaluateBoundary(); p.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    std::vector<std::vector<scalar>> phiBnd(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phiBnd[pi].assign(fvp[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
                phiBnd[pi] = b.values;
    }

    const FoamDict tp = readDict(caseDir + "/constant/transportProperties");
    const scalar nu = tp.scalarOr("nu", 1e-5);

    // nuEff = nu + OpenFOAM's OWN nut. The turbulence model is deliberately not run.
    std::vector<scalar> nuEffC(nC, nu);
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nuEffB[pi].assign(fvp[pi].size, nu);
    bool haveNut = false;
    try {
        GeometricField<scalar> nut =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/nut"), fvp, nC);
        nut.evaluateBoundary();
        for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nut.internal[c];
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& nb = nut.boundary[pi]->value();
            for (label i = 0; i < fvp[pi].size; ++i) nuEffB[pi][i] = nu + nb[i];
        }
        haveNut = true;
    } catch (...) { }

    const FoamDict fvSol = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simple = fvSol.subDict("SIMPLE");
    const bool consistent = simple && simple->wordOr("consistent", "no") == "yes";

    std::printf("resid_probe: nC=%d nu=%.3g nut=%s consistent=%d\n",
                (int)nC, nu, haveNut ? "from OpenFOAM" : "ABSENT (laminar)", (int)consistent);

    cpu::MomentumInput mi;
    mi.phi = &phiF.internalField;  mi.phiBnd = &phiBnd;
    mi.nuEff = &nuEffC;            mi.nuEffBnd = &nuEffB;
    mi.relaxU = 1.0;                                   // no relaxation: we want the raw residual
    mi.correctedLaplacian = true;                      // simpleCar and pitzDaily both say `corrected`
    mi.bounded = true;

    FvVectorMatrix UEqn = cpu::assembleUEqn(U, mi, m, g, fvp);
    cpu::addPressureGradient(UEqn, p, m, g, fvp);
    {
        GeometricField<vector> Ucopy =
            buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
        Ucopy.evaluateBoundary();
        const SolverPerformance perf = solveVector(UEqn, Ucopy, m, fvp, 1e-20, 0.0, 0);
        std::printf("  momentum  initial residual = %.4e\n", perf.initialResidual);
    }

    cpu::PressureInput pin;
    pin.relaxP = 1.0;
    pin.pRefCell = -1;
    pin.consistent = consistent;
    pin.correctedLaplacian = true;
    const cpu::PressureStages st = cpu::pressurePredictor(UEqn, U, p, pin, m, g, fvp);
    FvScalarMatrix pEqn = cpu::assemblePEqn(st, p, pin, m, g, fvp);
    {
        std::vector<scalar> pc = p.internal;
        const SolverPerformance perf = pbicgstab(pEqn, pc, m, fvp, 1e-20, 0.0, 0);
        std::printf("  pressure  initial residual = %.4e\n", perf.initialResidual);
    }
    // ---- FROZEN-NUT RUN --------------------------------------------------------------------------
    // The implication of the two residuals above: with OpenFOAM's own nut imposed, its solution nearly
    // satisfies brae's momentum and pressure equations. If that is right, then running brae's SIMPLE loop
    // with nut FROZEN at OpenFOAM's values must converge back to OpenFOAM's U and p -- and any remaining
    // disagreement would then belong to the turbulence model, which this run does not execute at all
    // (in.turb stays null, so kepsilon::correct is never called).
    if (argc > 3)
    {
        const int iters = std::atoi(argv[3]);
        cpu::SimpleFields f = cpu::createFields(caseDir + "/" + t, simple, m, g, fvp);
        cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSol);
        cpu::SimpleControl ctl(cd);
        cpu::StepInput in;
        in.nu = nu;
        in.nuEff = nuEffC;  in.nuEffBnd = nuEffB;      // frozen, from OpenFOAM
        in.turb = nullptr;                              // the turbulence model is NOT run
        in.correctedLaplacian = true;
        in.bounded = true;
        in.relaxU = 0.7; in.relaxP = 0.3;
        in.tolU = 1e-8; in.relTolU = 0.1;
        in.tolP = 1e-6; in.relTolP = 0.1;
        in.maxIterU = 1000;  in.maxIterP = 1000;
        for (int i = 0; i < iters; ++i) cpu::simpleStep(f, ctl, in, m, g, fvp);
        scalar du = 0, mu = 0, dp = 0, mp = 0;
        for (label c = 0; c < nC; ++c)
        {
            const vector& a2 = f.U.internal[c]; const vector& b2 = U.internal[c];
            du = std::fmax(du, std::fmax(std::fabs(a2.x-b2.x), std::fmax(std::fabs(a2.y-b2.y), std::fabs(a2.z-b2.z))));
            mu = std::fmax(mu, std::fmax(std::fabs(b2.x), std::fmax(std::fabs(b2.y), std::fabs(b2.z))));
            dp = std::fmax(dp, std::fabs(f.p.internal[c]-p.internal[c]));
            mp = std::fmax(mp, std::fabs(p.internal[c]));
        }
        std::printf("  frozen-nut, %d iterations: U %.4e   p %.4e  (max-norm, vs OpenFOAM)\n",
                    iters, mu>0?du/mu:du, mp>0?dp/mp:dp);
    }
    return 0;
}
