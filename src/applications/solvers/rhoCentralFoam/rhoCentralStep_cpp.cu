// rhoCentralStep_cpp.cu -- the host reference arm. See the header.
//
// rhoCentralFoam.C, static mesh, !LTS, inviscid:
//   102-210  fluxes                      centralFluxes (rhoCentralFlux_cpp.cu)
//   222-286  cells                       the four stages of rhoCentralCellUpdate.cuh, a loop each
//   232      U.correctBoundaryConditions brae's patch fields
//   233, 265-271  rhoU_b, e_b, T_b, psi_b, rhoE_b   updatePatchFace (rhoCentralPatchUpdate.cuh), with T's patches
//                                        evaluated on the cells' old T, as the energy BCs' Tw.evaluate() sees them
//   287-288  p.correctBoundaryConditions, rho_b = psi_b*p_b
// Every cell update reads only its own cell and the fluxes, so all cells first and then all patches is the same
// arithmetic as OpenFOAM's field-by-field order.
#include "rhoCentralStep_cpp.cuh"
#include "rhoCentralCellUpdate.cuh"
#include "rhoCentralCreateFields_cpp.cuh"
#include "rhoCentralFlux_cpp.cuh"
#include "rhoCentralPatchUpdate.cuh"

#include <vector>

namespace brae {
namespace cpu {
namespace rhoCentral {

using namespace brae::rhoCentral;

namespace {

class HostArm final : public Arm
{
public:
    HostArm(const Case& cs, State& s, PatchFields& pf) : cs_(cs), s_(s), pf_(pf), TfixesValue_(cs.nB)
    {
        for (label k = 0; k < cs.nB; ++k) TfixesValue_[k] = pf.T.boundary[cs.bPatch[k]]->fixesValue() ? 1 : 0;
    }

    void step(scalar deltaT) override
    {
        const Case& cs = cs_;
        State& s = s_;
        const HostFluxes f = centralFluxes(cs, s, deltaT);
        StepReport rep;
        rep.CoNum = f.CoNum;
        rep.meanCoNum = f.meanCoNum;
        bndMass_ += deltaT*f.bndMass;
        bndMom_ += deltaT*f.bndMom;
        bndEnergy_ += deltaT*f.bndEnergy;

        const scalar rDeltaT = 1.0/deltaT;
        const std::vector<scalar>& V = cs.g.V();
        Told_ = s.T;
        // the four stages of rhoCentralCellUpdate.cuh, each over every cell (OpenFOAM's field-at-a-time order)
        for (label i = 0; i < cs.nC; ++i)                                          // :224-231
        {
            const MomentumNew n = momentumStage(rDeltaT, V[i], f.divPhi[i], f.divPhiUp[i].x, f.divPhiUp[i].y,
                                                f.divPhiUp[i].z, cs.sd.valid(0), cs.sd.valid(1), cs.sd.valid(2),
                                                s.rho[i], s.rhoU[i].x, s.rhoU[i].y, s.rhoU[i].z);
            s.rho[i] = n.rho;
            s.rhoU[i] = vector{n.rhoUx, n.rhoUy, n.rhoUz};
            s.U[i] = vector{n.Ux, n.Uy, n.Uz};
        }
        for (label i = 0; i < cs.nC; ++i)                                          // :258-264
        {
            const EnergyNew n = energyStage(rDeltaT, V[i], f.divPhiEp[i], s.rhoE[i], s.rho[i], s.U[i].x, s.U[i].y,
                                            s.U[i].z);
            s.rhoE[i] = n.rhoE;
            s.e[i] = n.e;
        }
        for (label i = 0; i < cs.nC; ++i)                                          // :266
        {
            const ThermoNew n = thermoStage(s.e[i], s.p[i], s.T[i], cs.gas);
            s.T[i] = n.T;
            s.psi[i] = n.psi;
            rep.status |= n.status;
        }
        for (label i = 0; i < cs.nC; ++i)                                          // :284-286
        {
            s.p[i] = pressureStage(s.rho[i], s.psi[i]);
            rep.status |= physicalStatus(s.rho[i], s.p[i], s.T[i], s.e[i]);
        }

        evaluatePatches(pf_.U, s.U, cs, s.UB);       // :232
        evaluatePatches(pf_.T, Told_, cs, TbEval_);  // :265, Tw.evaluate() inside the energy BCs
        for (label k = 0; k < cs.nB; ++k)
        {
            const PatchNew b = updatePatchFace(TfixesValue_[k], s.rhoB[k], s.UB[k].x, s.UB[k].y, s.UB[k].z,
                                               s.e[cs.bCell[k]], TbEval_[k], s.pB[k], cs.gas);
            s.rhoUB[k] = vector{b.rhoUx, b.rhoUy, b.rhoUz};
            s.eB[k] = b.e;
            s.TB[k] = b.T;
            s.psiB[k] = b.psi;
            s.rhoEB[k] = b.rhoE;
            rep.status |= b.status;
        }
        evaluatePatches(pf_.p, s.p, cs, s.pB);       // :287
        for (label k = 0; k < cs.nB; ++k) s.rhoB[k] = s.psiB[k]*s.pB[k];   // :288

        reports_.push_back(rep);
    }

    std::vector<StepReport> flush() override
    {
        std::vector<StepReport> r;
        r.swap(reports_);
        return r;
    }
    int pending() const override { return static_cast<int>(reports_.size()); }
    int window() const override { return 1; }
    void download(State&) override {}
    void boundaryOutflow(scalar& mass, vector& mom, scalar& energy) const override
    {
        mass = bndMass_;
        mom = bndMom_;
        energy = bndEnergy_;
    }

private:
    const Case& cs_;
    State& s_;
    PatchFields& pf_;
    std::vector<int> TfixesValue_;
    std::vector<scalar> Told_, TbEval_;
    std::vector<StepReport> reports_;
    scalar bndMass_ = 0, bndEnergy_ = 0;
    vector bndMom_{0, 0, 0};
};

} // namespace

std::unique_ptr<Arm> makeHostArm(const Case& cs, State& s, PatchFields& pf)
{
    return std::make_unique<HostArm>(cs, s, pf);
}

} // namespace rhoCentral
} // namespace cpu
} // namespace brae
