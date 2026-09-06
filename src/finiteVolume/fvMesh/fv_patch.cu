#include <stdexcept>
#include "fv_patch.cuh"

#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace brae {

std::vector<FvPatch> buildPatches(const PrimitiveMesh& m, const FvGeometry& g)
{
    std::vector<FvPatch> patches;
    patches.reserve(m.patches().size());
    for (const PatchInfo& pi : m.patches())
    {
        // OVERSET is not implemented, and it must not be mistaken for a constraint patch. It was listed in
        // isConstraintPatchType, so a field file omitting its boundaryField entry got a synthesised
        // constraint entry and the case RAN -- producing a converged, wrong answer with no message.
        //
        // Overset is not a boundary condition at all: OpenFOAM's src/overset replaces the matrix addressing
        // (fvMeshPrimitiveLduAddressing), because an acceptor cell's equation becomes an interpolation from
        // donor cells in another mesh region. Without hole cutting, a donor/acceptor search and that matrix
        // surgery, an overset mesh is just several disconnected meshes solved independently. Refuse.
        // cyclicACMI: THE FLOW PATH IS VERIFIED, THE TURBULENCE TRANSPORT IS NOT. Momentum, pressure and
        // the interface flux now reproduce OpenFOAM on pimpleFoam/RAS/oscillatingInletACMI2D run laminar
        // so both codes solve the same equations: 10 steps of velocity agree to 2.3e-09 with the mesh
        // static and 3.4e-07 with the inlet channel sliding, and the per-face interface flux to 6e-08.
        // With `linearUpwind grad(U)` and a cellLimited gradient -- the combination the turbulent case
        // uses, and which no comparison had ever exercised -- 1.4e-07. Those are FREE runs: both solvers
        // from 0, no restart, no probe.
        //
        // k and epsilon were ACMI work too, and are now largely fixed: the ACMI's non-overlap patch is a
        // wall of area (1-mask)*A, so on the covered part it is a `wall` with no wall behind it, and
        // counting those faces both imposed the near-wall epsilon on cells nowhere near a wall and made
        // deviceSolveScalarTransport zero the AMI off-diagonal there.
        //
        // The turbulent cases are much closer than they were. Three separate defects were behind what
        // looked like one turbulence problem: cellLimitedGrad could not see the coupled patches (found by
        // running the case LAMINAR with the turbulent one's linearUpwind scheme, which reproduced the
        // whole gap with no model present); kEpsilon's production gradient ignored the named grad(U)
        // scheme entirely; and the epsilon matrix was constrained to the raw wall value instead of the
        // blended one. Static turbulent is now 1.1e-03 in U and 7.4e-04 in k, moving 9.0e-03 and 9.3e-02.
        // A fourth was nearWallDist searching within one patch where OF searches across all of them
        // (useCombinedWallPatch, default true), which a cyclicACMI exposes because its blockage is a wall
        // patch coincident with the interface. Static turbulent is now U 1.1e-03, k 1.0e-04, eps 8.7e-05;
        // moving is U 9.0e-03, k 9.3e-02. What holds the refusal is the moving case's k and the turbulent
        // U, which is 1.1e-03 against 2.3e-09 for the same case run laminar.
        //
        // AN EARLIER VERSION OF THIS NOTE blamed the interface's VISCOUS coupling, "error proportional to
        // nuEff, the explicit half of UEqn.H()". That was withdrawn: the probe behind it selected its
        // reference with runTime.times().last() and read a stale 0.011 directory while comparing against
        // 0.01. The free run at nu = 1e-3 agrees to 6.7e-08. See ACMI-HANDOFF.md (kept in cudafoam/, outside the repo).
        //
        // THE OLD JUSTIFICATION HERE WAS WRONG, and is kept because it cost a long hunt. It read
        // "contLocal ~0.33 where OF reaches 1e-14, concentrated on the cells touching the interface".
        // Both halves were the instrument: the residual omitted interfaceAddDiv, so it measured the
        // interface flux the pressure equation had just driven to zero. A per-cell continuity residual
        // could not have found any of the real defects anyway -- this interface fails by having both
        // sides balance cell-by-cell while transmitting different totals.
        //
        // What was actually wrong, all found by tracing values against OpenFOAM term by term:
        //   1. the ACMI coverage was applied twice, in the face area AND in the interpolation weights
        //   2. the mesh-move AMI rebuild threw away the interface flux
        //   3. the coupled-patch flux was never written, so a restart rebuilt it from U
        //   4. cyc_/ami_.ifCoeff is shared between the momentum and pressure assemblies, so from the
        //      SECOND pressure corrector on, UEqn.H() read the pressure laplacian coefficient
        //   5. fvc::makeRelative ran at the mesh move instead of at the end of the pressure corrector,
        //      and never reached the interface flux at all
        //   6. fvc::ddtCorr did not exist -- and it is NOT exempt on a cyclicACMI, because
        //      cyclicACMIFvPatch derives from coupledFvPatch, not from cyclicAMIFvPatch
        //   7. no Uf, so the moving-mesh form of ddtCorr had to be approximated from phi + mesh.phi()
        //
        // A case that runs to a confident wrong answer is the one outcome this codebase does not accept,
        // so ACMI stays refused while the turbulence coupling is open. BRAE_ALLOW_ACMI=1 opts in for
        // development; it is deliberately not a dictionary setting, so no case file can turn it on by
        // accident.
        if (pi.type == "cyclicACMI" && !std::getenv("BRAE_ALLOW_ACMI"))
            throw std::runtime_error(
                "brae: patch '" + pi.name + "' is type 'cyclicACMI'. The momentum, pressure and interface "
                "flux path now matches OpenFOAM -- on pimpleFoam/RAS/oscillatingInletACMI2D run laminar "
                "the velocity agrees to 2.3e-09 static and 3.4e-07 with the mesh moving, over 10 free "
                "steps -- but the k/epsilon transport does not: the same case run turbulent is 1.2e-03 "
                "out static and 2.5e-02 moving. Refused rather than solved to a plausible wrong answer. "
                "Set BRAE_ALLOW_ACMI=1 to run it anyway (development only).");

        if (pi.type == "overset")
        {
            throw std::runtime_error(
                "brae: patch '" + pi.name + "' is type 'overset', and overset meshes are not supported. "
                "Overset replaces the matrix addressing (acceptor cells are interpolated from donors in "
                "another region), so running without it would solve the regions as if they were "
                "unconnected and converge to a wrong answer. See docs/roadmap.md.");
        }
        // ANY OTHER COUPLED PATCH. A patch that names a `neighbourPatch` is declaring itself half of a
        // coupled pair: its face values come from the other side, not from a boundary condition. brae
        // builds that coupling for cyclic, cyclicAMI and cyclicACMI. Reaching here with any other such
        // type means the pair would be solved as two ORDINARY boundaries -- silently decoupled.
        //
        // This is not hypothetical. `cyclicPeriodicAMI` (an AMI whose sides span different sectors, tiled
        // by repeatedly applying a periodic transform) got through exactly that way, and the reason it
        // was silent is worth keeping: its mesh entry carries `inGroups 1(cyclicAMI)`, so the FIELD side
        // matched setConstraintTypes' cyclicAMI entry and built a cyclicAMI patch field, while the MESH
        // side tested the real type and skipped it. The field believed it was coupled; nothing coupled
        // it. On pimpleFoam/RAS/oscillatingInletPeriodicAMI2D the downstream half of the interface came
        // out with U identically zero on all 96 faces and carried no flux at all (OpenFOAM: 1.0e-01),
        // and on RAS/axialTurbine the four rotor-stator connections passed flux without conserving it
        // (RUOUTLET +5.19e-03 against DTINLET -4.64e-03, an 11% imbalance where OF matches to 1e-5) --
        // a turbine solved as three disconnected components, converging to a plausible wrong answer.
        //
        // Tested on the mesh type rather than a name list of everything unimplemented: the marker is
        // structural, so a coupled type nobody has thought of yet is refused too, which is the whole
        // point. Deliberately no environment override -- there is nothing here that is right-but-slow.
        if (!pi.neighbourPatch.empty()
         && pi.type != "cyclic" && pi.type != "cyclicAMI" && pi.type != "cyclicACMI"
         && pi.type != "cyclicPeriodicAMI"
         && pi.type != "processor" && pi.type != "processorCyclic")
        {
            throw std::runtime_error(
                "brae: patch '" + pi.name + "' is type '" + pi.type + "', which names a neighbourPatch ('"
                + pi.neighbourPatch + "') and is therefore one half of a COUPLED interface that brae does "
                "not implement. brae couples cyclic, cyclicAMI and cyclicACMI. Running anyway would solve "
                "the two sides as unconnected boundaries and converge to a wrong answer with no message, "
                "so it is refused instead. See docs/roadmap.md.");
        }
        FvPatch p;
        p.name     = pi.name;
        p.type     = pi.type;
        p.inGroups = pi.inGroups;
        p.start    = pi.start;
        p.size     = pi.size;
        p.faceCells.resize(pi.size);
        p.deltaCoeffs.resize(pi.size);
        p.nf.resize(pi.size);
        p.magSf.resize(pi.size);
        p.Cf.resize(pi.size);
        for (label i = 0; i < pi.size; ++i)
        {
            const label f = pi.start + i;
            const label c = m.owner()[f];          // boundary face owner = adjacent cell
            p.faceCells[i] = c;
            // OF fvPatch::delta() is the normal-projected delta; deltaCoeffs = 1/(n.(Cf-C)).
            const vector nHat = g.Sf()[f] / g.magSf()[f];
            p.nf[i] = nHat;
            p.magSf[i] = g.magSf()[f];
            p.Cf[i] = g.Cf()[f];
            p.deltaCoeffs[i] = 1.0 / dot(g.Cf()[f] - g.C()[c], nHat);
        }
        // OF atmBoundaryLayer.C:45 -- boundBox(pp.localPoints()).min(), taken over every point of every
        // face on the patch. An empty patch keeps the zero default; nothing reads it.
        if (pi.size > 0)
        {
            bool first = true;
            for (label i = 0; i < pi.size; ++i)
            {
                const label f = pi.start + i;
                for (label k = m.faceOffsets()[f]; k < m.faceOffsets()[f + 1]; ++k)
                {
                    const vector& pt = m.points()[m.faceVerts()[k]];
                    if (first)
                    {
                        p.ppMin = pt;
                        first   = false;
                    }
                    else
                    {
                        p.ppMin.x = std::min(p.ppMin.x, pt.x);
                        p.ppMin.y = std::min(p.ppMin.y, pt.y);
                        p.ppMin.z = std::min(p.ppMin.z, pt.z);
                    }
                }
            }
        }
        patches.push_back(std::move(p));
    }
    return patches;
}

} // namespace brae
