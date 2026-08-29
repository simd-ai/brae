// STRUCTURAL measurements on OpenFOAM's angledDuctExplicitFixedCoeff tutorial.
//
// WHY THIS FILE EXISTS. Two things in this port were correct-by-construction and measured by nothing,
// and both notes said so honestly -- and both said, wrongly, that no fixture could measure them. This
// tutorial can measure both, and it is the only case reachable from here that does:
//
//   1. THE TWO BOUNDARY MASKS. constrainHbyA asks `assignable`; adjustPhi asks
//      `fixesValue() && !isInletOutlet()`. They disagree only where a patch is non-assignable WITHOUT
//      fixing a value, which is SLIP -- and this tutorial gives porosityWall `type slip` on U. No case
//      under validation/ has one, so deriving either mask from the other passed every registered gate.
//
//   2. THE relax -> constrain -> setValues ORDER. kEpsilon.C:265-267 is relax(), then
//      fvOptions.constrain(), then boundaryManipulate(). The two orders differ only where a cell is BOTH
//      wall-constrained and fvOption-constrained, because fvMatrix::setValues transfers
//      source_[nei] -= coeff*value and then ZEROES that coeff, so only the FIRST setValues touching a
//      cell moves anything into its neighbours. This tutorial's fvOptions carries
//      `porosityTurbulence { scalarFixedValueConstraint; cellZone porosity; fieldValues { k 1;
//      epsilon 150; } }`, its blockMeshDict builds porosityWall from the porosity block's own faces, and
//      0.orig/epsilon gives that wall an epsilonWallFunction -- so the overlap is every one of those
//      cells.
//
// NO CONVERGENCE BOUND, AND THAT IS THE DESIGN. The end-to-end gate on this same tutorial is
// deliberately unregistered because registering it would mean choosing a bound around an unexplained
// 9.43e-04 (PORT.md). Everything here is STRUCTURAL or a SINGLE assembly: the masks are counted, and the
// ordering is measured by running one correct() each way and differencing them. Neither asks the case to
// converge, so neither inherits that open question.
//
// Run: test_rho_angledduct_structural <caseDir> <timeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fvOptions_cpp.cuh"
#include "mrf_read.cuh"
#include "rhoCreateFields_cpp.cuh"
#include "rhoCreateFields.cuh"
#include "kEpsilon_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

static void check(const char* what, bool ok)
{
    if (!ok) ++g_fails;
    std::printf("  %-64s %s\n", what, ok ? "OK" : "FAIL");
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
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(caseDir + "/" + t, caseDir, simpleDict, &fvSolution, m, g, fvp);

    std::printf("angledDuct STRUCTURAL  (%d cells, %s)\n",
                (int)nC, f.turbulent ? f.rasModel.c_str() : "laminar");

    // ---- 0. the fixture properties this file depends on ---------------------------------------
    // Asserted, not assumed: if the tutorial ever loses its slip patch or its turbulence constraint,
    // every measurement below goes vacuous and would otherwise pass in silence.
    std::printf("  0. the fixture\n");
    {
        bool anySlip = false;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const bool a  = f.U.boundary[pi]->assignable();
            const bool fv = f.U.boundary[pi]->fixesValue();
            const bool io = f.U.boundary[pi]->isInletOutlet();
            if ((!a) != (fv && !io)) anySlip = true;
            std::printf("     PATCH %-16s assignable=%d fixesValue=%d inletOutlet=%d\n",
                        fvp[pi].name.c_str(), (int)a, (int)fv, (int)io);
        }
        check("a patch separates the two mask rules (slip on U)", anySlip);
        check("the case is turbulent (the ordering needs the closure)", f.turbulent);
    }

    // ---- 1. THE TWO MASKS, on a case that can tell them apart ---------------------------------
    std::printf("  1. the two boundary masks\n");
    {
        const gpu::rhoSimple::RhoDeviceFields dev =
            gpu::rhoSimple::createDeviceFields(f, m, g, fvp);
        const std::vector<label> takeU = dev.takeUAtBoundary.host();
        const std::vector<label> adj   = dev.adjustable.host();

        std::size_t differ = 0, k = 0;
        std::string where;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            std::size_t here = 0;
            for (label i = 0; i < fvp[pi].size; ++i, ++k)
                if (takeU[k] != (1 - adj[k])) { ++differ; ++here; }
            if (here) { if (!where.empty()) where += ", "; where += fvp[pi].name; }
        }
        std::printf("     %-64s %zu faces on %s\n",
                    "  (faces where the masks are not complements)", differ, where.c_str());
        check("the two masks DIFFER -- this fixture discriminates them", differ > 0);

        // THE FAIL-PROOF, in-test. Deriving `adjustable` from `assignable` is the conflation the module
        // warns about; on every validation fixture it passed. Here it must not.
        std::size_t conflatedWrong = 0;
        k = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const bool a  = f.U.boundary[pi]->assignable();
            const bool fv = f.U.boundary[pi]->fixesValue();
            const bool io = f.U.boundary[pi]->isInletOutlet();
            for (label i = 0; i < fvp[pi].size; ++i, ++k)
                if ((a ? 1 : 0) != ((fv && !io) ? 0 : 1)) ++conflatedWrong;
        }
        std::printf("     %-64s %zu faces\n",
                    "  (faces a mask derived from the other would get wrong)", conflatedWrong);
        check("the conflated mask WOULD be wrong here (fail-proof)", conflatedWrong > 0);
    }

    // ---- 2. THE ORDERING, measured ------------------------------------------------------------
    // One correct() each way from the SAME start. The difference is confined to cells that are both
    // wall-constrained and fvOption-constrained; everywhere else the two orders are identical, which is
    // why a whole-field number is the right way to see whether it reaches the answer at all.
    std::printf("  2. relax -> constrain -> setValues, against the inverted order\n");
    {
        cpu::fvOptions::OptionList keOpts = cpu::fvOptions::read(caseDir, m);
        if (!keOpts.firstUnsupported().empty())
        {
            // An unported option would constrain a DIFFERENT system, so the ordering measured below
            // would not be the one the case asks for.
            std::printf("  REFUSED: fvOptions declares '%s', which is not ported.\n",
                        keOpts.firstUnsupported().c_str());
            return 1;
        }
        std::printf("     %-64s %zu option(s)\n", "  (fvOptions read from the case)",
                    keOpts.options.size());
        std::size_t constrainsTurb = 0;
        for (const auto& o : keOpts.options)
            for (const auto& fvv : o.fieldValues)
                if (fvv.first == "k" || fvv.first == "epsilon")
                {
                    ++constrainsTurb;
                    std::printf("     %-30s constrains %-8s = %g on %zu cells\n",
                                o.name.c_str(), fvv.first.c_str(), (double)fvv.second, o.cells.size());
                }
        check("an fvOption constrains k or epsilon (else the order is inert)", constrainsTurb > 0);

        // Cells that are BOTH wall-function-adjacent AND inside a constrained cell set. The ordering can
        // only act there, so if this is zero the measurement below means nothing.
        std::vector<char> isWall(nC, 0);
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (f.epsilon.boundary[pi]->isTurbulenceWallFunction())
                for (label i = 0; i < fvp[pi].size; ++i) isWall[fvp[pi].faceCells[i]] = 1;
        std::vector<char> isOpt(nC, 0);
        for (const auto& o : keOpts.options)
        {
            bool touchesTurb = false;
            for (const auto& fvv : o.fieldValues)
                if (fvv.first == "k" || fvv.first == "epsilon") touchesTurb = true;
            if (!touchesTurb) continue;
            for (const label c : o.cells) if (c >= 0 && c < nC) isOpt[c] = 1;
        }
        std::size_t overlap = 0;
        for (label c = 0; c < nC; ++c) if (isWall[c] && isOpt[c]) ++overlap;
        std::printf("     %-64s %zu cells\n",
                    "  (cells both wall-constrained AND fvOption-constrained)", overlap);
        check("the two constraints OVERLAP (this is what the order acts on)", overlap > 0);

        // Both arms are run from an independently constructed field set, because RhoSimpleFields owns
        // unique_ptr patch fields and cannot be copied.
        std::vector<scalar> epsA, epsB, kA, kB;
        for (int arm = 0; arm < 2; ++arm)
        {
            cpu::rhoSimple::RhoSimpleFields fa =
                cpu::rhoSimple::createFields(caseDir + "/" + t, caseDir, simpleDict, &fvSolution,
                                             m, g, fvp);
            fa.k.evaluateBoundary();
            fa.epsilon.evaluateBoundary();

            std::vector<std::vector<scalar>> rhoBnd(fvp.size());
            for (std::size_t pi = 0; pi < fvp.size(); ++pi) rhoBnd[pi] = fa.rho.boundary[pi]->value();
            std::vector<scalar> nuLam(nC), alphat(nC, 0.0);
            for (label c = 0; c < nC; ++c) nuLam[c] = 1.5e-05;
            std::vector<std::vector<scalar>> nuLamBnd(fvp.size());
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                nuLamBnd[pi].assign(fvp[pi].size, 1.5e-05);

            cpu::kEpsilonRef::Compressible comp;
            comp.rho = &fa.rho.internal;   comp.rhoBnd = &rhoBnd;
            comp.nu  = &nuLam;             comp.nuBnd  = &nuLamBnd;
            comp.phiByRho = &fa.phi;
            comp.alphat = &alphat;         comp.Prt = 1.0;

            KEpsilonCoeffs co;
            cpu::kEpsilonRef::KEResiduals res;
            cpu::kEpsilonRef::correct(fa.U, fa.k, fa.epsilon, fa.nut, fa.phi, /*nu=*/0.0, m, g, fvp,
                                      /*relaxEps=*/0.7, /*relaxK=*/0.7, 1e-12, 0.0, 2000, co, &res,
                                      /*bounded=*/true, /*dropTerm=*/0, &comp,
                                      keOpts.empty() ? nullptr : &keOpts,
                                      /*relaxEquationEps=*/true, /*relaxEquationK=*/true,
                                      /*constrainBeforeWall=*/(arm == 0));
            if (arm == 0) { epsA = fa.epsilon.internal; kA = fa.k.internal; }
            else          { epsB = fa.epsilon.internal; kB = fa.k.internal; }
        }

        // ...and a THIRD arm at the DEFAULT, with no ordering argument at all. The two arms above prove
        // the order matters; only this one proves brae ships OpenFOAM's. Without it the gate would pass
        // just as happily on a reference that had picked the wrong order and measured it carefully.
        std::vector<scalar> epsDefault;
        {
            cpu::rhoSimple::RhoSimpleFields fd =
                cpu::rhoSimple::createFields(caseDir + "/" + t, caseDir, simpleDict, &fvSolution,
                                             m, g, fvp);
            fd.k.evaluateBoundary();
            fd.epsilon.evaluateBoundary();
            std::vector<std::vector<scalar>> rhoBnd(fvp.size());
            for (std::size_t pi = 0; pi < fvp.size(); ++pi) rhoBnd[pi] = fd.rho.boundary[pi]->value();
            std::vector<scalar> nuLam(nC, 1.5e-05), alphat(nC, 0.0);
            std::vector<std::vector<scalar>> nuLamBnd(fvp.size());
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                nuLamBnd[pi].assign(fvp[pi].size, 1.5e-05);
            cpu::kEpsilonRef::Compressible comp;
            comp.rho = &fd.rho.internal;   comp.rhoBnd = &rhoBnd;
            comp.nu  = &nuLam;             comp.nuBnd  = &nuLamBnd;
            comp.phiByRho = &fd.phi;
            comp.alphat = &alphat;         comp.Prt = 1.0;
            KEpsilonCoeffs co;
            cpu::kEpsilonRef::KEResiduals res;
            cpu::kEpsilonRef::correct(fd.U, fd.k, fd.epsilon, fd.nut, fd.phi, 0.0, m, g, fvp,
                                      0.7, 0.7, 1e-12, 0.0, 2000, co, &res, true, 0, &comp,
                                      keOpts.empty() ? nullptr : &keOpts);
            epsDefault = fd.epsilon.internal;
        }
        const double dDefault = relL2(epsDefault, epsA);
        std::printf("     %-64s %.6e\n", "  (the SHIPPED default against OpenFOAM's order)", dDefault);
        check("brae ships OpenFOAM's order, not the other one", dDefault == 0.0);

        const double dEps = relL2(epsB, epsA);
        const double dK   = relL2(kB, kA);
        std::printf("     %-64s eps %.6e  k %.6e\n",
                    "  (inverted order against OpenFOAM's)", dEps, dK);
        // THE MEASUREMENT. If the two orders agreed, the ordering would be a comment rather than a
        // decision, and the fix that introduced it would be unfalsifiable. This is the number that was
        // missing when the reference's own note said no fixture could produce one.
        check("the ORDER changes the answer -- it is load-bearing", dEps > 1e-12);
    }

    std::printf("%s\n", g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
