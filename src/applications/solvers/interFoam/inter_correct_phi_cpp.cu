// interFoam's correctPhi -- see inter_correct_phi_cpp.cuh for the provenance and the order.
#include "inter_correct_phi_cpp.cuh"
#include "fv_matrix_ops.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "inter_peqn_cpp.cuh"
#include "pcg.cuh"
#include <stdexcept>
#include <string>

namespace brae {
namespace cpu {
namespace interFoam {

namespace {

const char* const WHO = "brae interFoam CorrectPhi: ";

// fvc::makeRelative(phi, U) and makeAbsolute, where the mesh is moving
void relative(
    SurfaceScalarField& phi,
    const SurfaceScalarField* meshPhi,
    scalar sign)
{
    if (!meshPhi) return;
    for (std::size_t f = 0; f < phi.internal.size(); ++f)
    {
        phi.internal[f] -= sign*meshPhi->internal[f];
    }
    for (std::size_t pi = 0; pi < phi.boundary.size() && pi < meshPhi->boundary.size(); ++pi)
    {
        for (std::size_t i = 0; i < phi.boundary[pi].size() && i < meshPhi->boundary[pi].size(); ++i)
        {
            phi.boundary[pi][i] -= sign*meshPhi->boundary[pi][i];
        }
    }
}

// pcorr's patch fields (CorrectPhi.C:52-66): fixedValue 0 where p_rgh fixes a value, zeroGradient
// elsewhere -- and on a constraint patch its constraint type, which fvPatchField::New puts in the
// zeroGradient's place
GeometricField<scalar> makePcorr(
    const GeometricField<scalar>& p_rgh,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches)
{
    GeometricField<scalar> pcorr;
    pcorr.internal.assign(static_cast<std::size_t>(m.nCells()), scalar(0));
    pcorr.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        if (isCoupledInterfaceType(p.type) && !p.coupled)
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + p.name + "` is " + p.type + " and its coupling is not "
                "attached; pcorr across it is ported for an attached `cyclic` only.");
        }
        // an attached cyclic falls to the constraint branch below and gets a PLAIN cyclic: pcorr's
        // patch types come from the mesh patch (CorrectPhi.C:52-66), so even under a p_rgh that
        // carries a jump there, pcorr carries none
        if (p_rgh.boundary[pi]->fixesValue())
        {
            pcorr.boundary[pi].reset(new FixedValuePatchField<scalar>(p, true, scalar(0), {}));
        }
        else if (isConstraintPatchType(p.type))
        {
            PatchFieldData<scalar> d;
            d.name = p.name;
            d.type = p.type;
            pcorr.boundary[pi] = makePatchField<scalar>(p, d);
        }
        else
        {
            pcorr.boundary[pi].reset(new ZeroGradientPatchField<scalar>(p));
        }
    }
    pcorr.evaluateBoundary();
    return pcorr;
}

} // namespace

void correctUphiBCs(
    GeometricField<vector>& U,
    SurfaceScalarField& phi,
    const SurfaceScalarField* rhoPhi,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& q = patches[pi];
        fvPatchField<vector>& Up = *U.boundary[pi];
        if (!Up.fixesValue() || q.type == "empty") continue;
        // evaluate(): updateCoeffs first, which for a flux-conditional condition looks the flux up --
        // the Sf & Uf just assigned, not yet overwritten on this patch
        Up.updateFromFlux(namedPatchFlux(Up.fluxName(), pi, q.name, phi, rhoPhi));
        Up.evaluate(U.internal);
        std::vector<vector> Ucell(static_cast<std::size_t>(q.size));
        for (label i = 0; i < q.size; ++i)
        {
            Ucell[static_cast<std::size_t>(i)] = U.internal[static_cast<std::size_t>(q.faceCells[i])];
        }
        Up.updateFromPatchVelocity(Up.value(), Ucell, {});
        // phibf[patchi] = Ubf[patchi] & mesh.Sf().boundaryField()[patchi]
        const std::vector<vector>& Ub = Up.value();
        for (label i = 0; i < q.size; ++i)
        {
            phi.boundary[pi][static_cast<std::size_t>(i)] =
                dot(Ub[static_cast<std::size_t>(i)], g.Sf()[static_cast<std::size_t>(q.start + i)]);
        }
    }
}

void correctPhi(
    GeometricField<vector>& U,
    SurfaceScalarField& phi,
    const GeometricField<scalar>& p_rgh,
    const CorrectPhiInput& in,
    const CorrectPhiControls& c,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    if (!in.rAUf || !c.pcorrFinal)
    {
        throw std::runtime_error(std::string(WHO) + "rAUf and the pcorrFinal solve are required.");
    }
    if (c.nNonOrthogonalCorrectors > 0 && !c.pcorr)
    {
        throw std::runtime_error(
            std::string(WHO) + "a non-orthogonal corrector solves with the `pcorr` entry, and none was "
            "handed in.");
    }

    if (in.meshChanging)
    {
        correctUphiBCs(U, phi, in.rhoPhi, g, patches);
    }

    GeometricField<scalar> pcorr = makePcorr(p_rgh, m, patches);
    bool needReference = true;
    for (const auto& pf : pcorr.boundary)
    {
        if (pf->fixesValue())
        {
            needReference = false;
            break;
        }
    }

    if (needReference)
    {
        relative(phi, in.meshPhi, scalar(1));
        adjustPhi(phi, U, true, patches);
        relative(phi, in.meshPhi, scalar(-1));
    }

    const label nC = m.nCells();
    // div(phi) is the equation's whole source; it does not change inside the loop
    const std::vector<scalar> divPhi = fvc::div(phi, m, g, patches);
    for (label corr = 0; corr <= c.nNonOrthogonalCorrectors; ++corr)
    {
        const bool finalIter = (corr == c.nNonOrthogonalCorrectors);

        // fvm::laplacian(rAUf, pcorr) == fvc::div(phi) - divU, divU a geometricZeroField
        FvScalarMatrix pe = fvm::laplacian<scalar>(*in.rAUf, pcorr, m, g, patches, c.correctedLaplacian);
        if (c.correctedLaplacian)
        {
            // the fluxRequired branch of gaussLaplacianSchemes.C: CorrectPhi.C:73 sets it for pcorr
            const std::vector<vector> gradP = fvc::gaussGrad(pcorr, m, g, patches);
            const std::vector<scalar> corrSrc = fvm::laplacianNonOrthSource<scalar, vector>(
                *in.rAUf, pcorr, gradP, m, g, patches, c.snGradLimitCoeff);
            for (label cell = 0; cell < nC; ++cell)
            {
                pe.source[static_cast<std::size_t>(cell)] -= corrSrc[static_cast<std::size_t>(cell)];
            }
            pe.faceFluxCorrection = fvm::laplacianCorrFlux<scalar, vector>(
                *in.rAUf, gradP, m, g, c.snGradLimitCoeff, &pcorr);
        }
        // fvMatrix == volScalarField: source += V*su
        for (label cell = 0; cell < nC; ++cell)
        {
            const std::size_t k = static_cast<std::size_t>(cell);
            pe.source[k] += divPhi[k]*g.V()[k];
        }

        // pcorrEqn.setReference(0, 0): source += diag*0, diag += diag, where pcorr needs a reference
        if (needReference && nC > 0)
        {
            pe.source[0] += pe.diag[0]*scalar(0);
            pe.diag[0] += pe.diag[0];
        }

        // pcorr.select(pimple.finalNonOrthogonalIter())
        const InterFields::PressureLinearSolve& s = finalIter ? *c.pcorrFinal : *c.pcorr;
        SolverPerformance sp;
        if (s.pcgGamg())
        {
            if (!c.gamgCache)
            {
                throw std::runtime_error(
                    std::string(WHO) + "a GAMG-preconditioned pcorr solve needs the mesh's hierarchy.");
            }
            sp = pcgGamgSolve(
                pe,
                pcorr.internal,
                m,
                g,
                patches,
                *c.gamgCache,
                s.tol,
                s.relTol,
                s.maxIter,
                0,
                s.gamgPrecond,
                nullptr);
        }
        else if (s.gamgSolver())
        {
            if (!c.gamgCache)
            {
                throw std::runtime_error(
                    std::string(WHO) + "a GAMG pcorr solve needs the mesh's hierarchy.");
            }
            const GamgAgglomeration& a = c.gamgCache->get(m, g, s.gamg.nCellsInCoarsestLevel);
            sp = gamgSolve(
                pe,
                pcorr.internal,
                m,
                patches,
                a,
                s.gamg,
                nullptr);
        }
        else if (s.pcgDIC())
        {
            sp = pcg(pe, pcorr.internal, m, patches, s.tol, s.relTol, s.maxIter);
        }
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "pcorr's solver `" + s.solver + "` is not ported (refused where the "
                "case is read; reaching here is a defect).");
        }
        if (in.solveLog)
        {
            in.solveLog->push_back(LinearSolveRecord{sp.initialResidual, sp.finalResidual, sp.nIterations});
        }
        pcorr.evaluateBoundary();

        if (finalIter)
        {
            // phi -= pcorrEqn.flux()
            const SurfaceScalarField flux = matrixFlux(pe, pcorr.internal, m, patches);
            for (std::size_t f = 0; f < phi.internal.size(); ++f)
            {
                phi.internal[f] -= flux.internal[f];
            }
            for (std::size_t pi = 0; pi < phi.boundary.size(); ++pi)
            {
                if (patches[pi].type == "empty") continue;
                for (std::size_t i = 0; i < phi.boundary[pi].size(); ++i)
                {
                    phi.boundary[pi][i] -= flux.boundary[pi][i];
                }
            }
        }
    }
}

SurfaceScalarField unitFaceField(
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches)
{
    SurfaceScalarField one;
    one.internal.assign(static_cast<std::size_t>(m.nInternalFaces()), scalar(1));
    one.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        one.boundary[pi].assign(static_cast<std::size_t>(patches[pi].size), scalar(1));
    }
    return one;
}

} // namespace interFoam
} // namespace cpu
} // namespace brae
