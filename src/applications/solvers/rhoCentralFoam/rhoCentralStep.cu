// rhoCentralStep.cu -- the device arm. See the header; the arithmetic is the host arm's (rhoCentralStep_cpp.cu).
//
// Per step:
//   centralFluxes (rhoCentralFlux.cu)                 rhoCentralFoam.C:102-210
//   cellKernel   fvc::div gathered in OpenFOAM's face order, then the four cell stages   :222-286
//   deviceUpdateInletOutlet on U, T, p                 :195's phi decides each inletOutlet face (shared, milestone D)
//   deviceBCValue on U (x3)                            :232 U.correctBoundaryConditions (shared DeviceBoundary)
//   deviceBCValue on T with the cells' old T           :265 Tw.evaluate() inside the energy BCs
//   patchKernel  transformPatchU, updatePatchFace      :232 on the transform faces, :233, 265-271
//   deviceBCValue on p                                 :287
//   rhoBKernel   rho_b = psi_b*p_b                     :288
//   courantKernel                                       centralCourantNo.H, kept on the device
#include "rhoCentralStep.cuh"
#include "rhoCentralCellUpdate.cuh"
#include "rhoCentralFlux.cuh"
#include "rhoCentralPatchUpdate.cuh"

#include "device_boundary.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace brae {
namespace gpu {
namespace rhoCentral {

using namespace brae::rhoCentral;

namespace {

constexpr int kTPB = 256;
inline int blocksFor(int n) { return (n + kTPB - 1)/kTPB; }
constexpr int kReportWindow = 256;

struct CellState
{
    scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rhoE, *e, *T, *Told, *p, *psi, *Ux, *Uy, *Uz;
};
struct FluxesIn
{
    const scalar *phi, *phiUx, *phiUy, *phiUz, *phiEp, *amaxSf;
    const scalar *phiB, *phiUxB, *phiUyB, *phiUzB, *phiEpB, *amaxSfB;
};

// fvc::div (surfaceIntegrate, fvcSurfaceIntegrate.C:45-78) gathered in OpenFOAM's scatter order -- owner (+) and
// neighbour (-) faces merged by face index, then the non-empty patch faces, then / V -- and the cell's update.
// surfaceSum(amaxSf) for the Courant number rides along (+ on both sides, fvcSurfaceIntegrate.C:165-166).
__global__ void cellKernel(int nC, int slot, scalar rDeltaT, int v0, int v1, int v2, GasCoeffs gas,
                           const label* __restrict__ ownerStart, const label* __restrict__ losort,
                           const label* __restrict__ losortStart, const label* __restrict__ bndCellStart,
                           const label* __restrict__ bndPerm, const label* __restrict__ bndIsEmpty,
                           const scalar* __restrict__ V, FluxesIn Fx, CellState S,
                           scalar* __restrict__ coCell, int* __restrict__ status)
{
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, sa = 0;
    int fo = ownerStart[c];
    const int foEnd = ownerStart[c + 1];
    int kn = losortStart[c];
    const int knEnd = losortStart[c + 1];
    while (fo < foEnd || kn < knEnd)
    {
        if ((kn >= knEnd) || (fo < foEnd && fo < losort[kn]))
        {
            d0 += Fx.phi[fo];
            d1 += Fx.phiUx[fo];
            d2 += Fx.phiUy[fo];
            d3 += Fx.phiUz[fo];
            d4 += Fx.phiEp[fo];
            sa += Fx.amaxSf[fo];
            ++fo;
        }
        else
        {
            const int f = losort[kn];
            d0 -= Fx.phi[f];
            d1 -= Fx.phiUx[f];
            d2 -= Fx.phiUy[f];
            d3 -= Fx.phiUz[f];
            d4 -= Fx.phiEp[f];
            sa += Fx.amaxSf[f];
            ++kn;
        }
    }
    for (int q = bndCellStart[c]; q < bndCellStart[c + 1]; ++q)
    {
        const int k = bndPerm[q];
        if (bndIsEmpty[k]) continue;
        d0 += Fx.phiB[k];
        d1 += Fx.phiUxB[k];
        d2 += Fx.phiUyB[k];
        d3 += Fx.phiUzB[k];
        d4 += Fx.phiEpB[k];
        sa += Fx.amaxSfB[k];
    }
    const scalar Vc = V[c];
    coCell[c] = sa;
    d0 /= Vc;
    d1 /= Vc;
    d2 /= Vc;
    d3 /= Vc;
    d4 /= Vc;

    // the four stages of rhoCentralCellUpdate.cuh, back to back
    const scalar Told = S.T[c];
    const MomentumNew m = momentumStage(rDeltaT, Vc, d0, d1, d2, d3, v0, v1, v2, S.rho[c], S.rhoUx[c], S.rhoUy[c],
                                        S.rhoUz[c]);
    const EnergyNew en = energyStage(rDeltaT, Vc, d4, S.rhoE[c], m.rho, m.Ux, m.Uy, m.Uz);
    const ThermoNew th = thermoStage(en.e, S.p[c], Told, gas);
    const scalar p = pressureStage(m.rho, th.psi);
    S.rho[c] = m.rho;
    S.rhoUx[c] = m.rhoUx;
    S.rhoUy[c] = m.rhoUy;
    S.rhoUz[c] = m.rhoUz;
    S.Ux[c] = m.Ux;
    S.Uy[c] = m.Uy;
    S.Uz[c] = m.Uz;
    S.rhoE[c] = en.rhoE;
    S.e[c] = en.e;
    S.Told[c] = Told;
    S.T[c] = th.T;
    S.psi[c] = th.psi;
    S.p[c] = p;
    const int st = th.status | physicalStatus(m.rho, p, th.T, en.e);
    if (st) atomicOr(&status[slot], st);
}

struct PatchState
{
    scalar *rho, *rhoUx, *rhoUy, *rhoUz, *rhoE, *e, *T, *psi, *Ux, *Uy, *Uz;
    const scalar *p, *TbEval;
};
// the cells' new rho and U, and each face's transform (rhoCentralPatchUpdate.cuh; Case::bUTransform and friends)
struct PatchTransforms
{
    const scalar *rhoC, *UxC, *UyC, *UzC;
    const int* kind;
    const tensor* R;
    const char *rhoConstraint, *TseedOwn;
};

// ioCount: the inletOutlet diagnostic, over the faces where U, p or T is inletOutlet -- [0] the face-steps whose phi was
// negative (inflow), [1] all of them
__global__ void patchKernel(int nB, int slot, GasCoeffs gas, int ctrl, const label* __restrict__ bCell,
                            const int* __restrict__ TfixesValue, const scalar* __restrict__ eCell, PatchState B,
                            PatchTransforms X, int* __restrict__ status, const scalar* __restrict__ phiB,
                            const char* __restrict__ ioFace, unsigned long long* __restrict__ ioCount)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    const label c = bCell[k];
    if (ioFace[k])
    {
        atomicAdd(&ioCount[1], 1ull);
        if (phiB[k] < 0) atomicAdd(&ioCount[0], 1ull);
    }
    if (X.kind[k])   // U.correctBoundaryConditions() on a transform face (:232)
    {
        const vector Ub = transformPatchU(X.kind[k], X.R[k], vector{X.UxC[c], X.UyC[c], X.UzC[c]}, ctrl);
        B.Ux[k] = Ub.x;
        B.Uy[k] = Ub.y;
        B.Uz[k] = Ub.z;
    }
    const PatchNew b = updatePatchFace(TfixesValue[k], patchRhoMid(X.rhoConstraint[k], X.rhoC[c], B.rho[k], ctrl),
                                       B.Ux[k], B.Uy[k], B.Uz[k], eCell[c], X.TseedOwn[k] ? B.T[k] : B.TbEval[k],
                                       B.p[k], gas);
    B.rhoUx[k] = b.rhoUx;
    B.rhoUy[k] = b.rhoUy;
    B.rhoUz[k] = b.rhoUz;
    B.e[k] = b.e;
    B.T[k] = b.T;
    B.psi[k] = b.psi;
    B.rhoE[k] = b.rhoE;
    if (b.status) atomicOr(&status[slot], b.status);
}

// the control inletOutletInverted: the switch fed -phi
__global__ void negateKernel(int n, const scalar* __restrict__ a, scalar* __restrict__ b)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k < n) b[k] = -a[k];
}

// rho.boundaryFieldRef() == psi.boundaryField()*p.boundaryField(), rhoCentralFoam.C:288
__global__ void rhoBKernel(int nB, const scalar* __restrict__ psiB, const scalar* __restrict__ pB,
                           scalar* __restrict__ rhoB)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB) return;
    rhoB[k] = psiB[k]*pB[k];
}

// centralCourantNo.H: CoNum = 0.5*gMax(sumAmaxSf/V)*dt, meanCoNum = 0.5*(gSum(sumAmaxSf)/gSum(V))*dt
__global__ void courantKernel(int nC, const scalar* __restrict__ sumA, const scalar* __restrict__ V,
                              double* __restrict__ out)
{
    __shared__ double smax[1024], ssum[1024], sv[1024];
    const int t = threadIdx.x;
    double mx = -1e300, sa = 0, svol = 0;
    for (int i = t; i < nC; i += blockDim.x)
    {
        const double r = sumA[i]/V[i];
        mx = (r > mx) ? r : mx;
        sa += sumA[i];
        svol += V[i];
    }
    smax[t] = mx;
    ssum[t] = sa;
    sv[t] = svol;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            smax[t] = (smax[t + s] > smax[t]) ? smax[t + s] : smax[t];
            ssum[t] += ssum[t + s];
            sv[t] += sv[t + s];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        out[0] = smax[0];
        out[1] = ssum[0];
        out[2] = sv[0];
    }
}

void upCmpt(DeviceBuffer<scalar>& d, const std::vector<vector>& h, int j)
{
    std::vector<scalar> t(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) t[i] = (j == 0) ? h[i].x : (j == 1) ? h[i].y : h[i].z;
    d.copyFrom(t);
}

void downCmpt(const DeviceBuffer<scalar>& d, std::vector<vector>& h, int j)
{
    const std::vector<scalar> t = d.host();
    h.resize(t.size());
    for (std::size_t i = 0; i < t.size(); ++i)
    {
        if (j == 0) h[i].x = t[i];
        else if (j == 1) h[i].y = t[i];
        else h[i].z = t[i];
    }
}


class DeviceArm final : public Arm
{
public:
    DeviceArm(const Case& cs, const State& s0, PatchFields& pf, int control) : cs_(cs), control_(control)
    {
        // the device memory the arm holds: what brae's device pool took from the driver to build it (every DeviceBuffer
        // goes through it). Not cudaMemGetInfo's free memory: on the GB10's unified memory that moves with every host
        // allocation: it read 0 and 303 MiB on wedge15Ma5 refined to 307,200 and 76,800 cells, which the pool puts at
        // 898.5 and 224.6 MiB.
        const std::size_t pool0 = brae::detail::devicePool().mallocBytes() - brae::detail::devicePool().freedBytes();
        const int nC = cs.nC, nB = cs.nB;
        dm_ = buildDeviceMesh(cs.m, cs.g, cs.patches);
        if (dm_.nBndFaces != cs.nB || dm_.nInternalFaces != cs.nIF)
            throw std::runtime_error("brae rhoCentralFoam: the device mesh's face layout is not the case's.");

        DeviceFields& F = F_;
        F.rho.copyFrom(s0.rho);
        upCmpt(F.rhoUx, s0.rhoU, 0);
        upCmpt(F.rhoUy, s0.rhoU, 1);
        upCmpt(F.rhoUz, s0.rhoU, 2);
        F.rhoE.copyFrom(s0.rhoE);
        F.e.copyFrom(s0.e);
        F.T.copyFrom(s0.T);
        F.p.copyFrom(s0.p);
        F.psi.copyFrom(s0.psi);
        upCmpt(F.Ux, s0.U, 0);
        upCmpt(F.Uy, s0.U, 1);
        upCmpt(F.Uz, s0.U, 2);
        F.rhoB.copyFrom(s0.rhoB);
        upCmpt(F.rhoUxB, s0.rhoUB, 0);
        upCmpt(F.rhoUyB, s0.rhoUB, 1);
        upCmpt(F.rhoUzB, s0.rhoUB, 2);
        F.rhoEB.copyFrom(s0.rhoEB);
        F.eB.copyFrom(s0.eB);
        F.TB.copyFrom(s0.TB);
        F.pB.copyFrom(s0.pB);
        F.psiB.copyFrom(s0.psiB);
        upCmpt(F.UxB, s0.UB, 0);
        upCmpt(F.UyB, s0.UB, 1);
        upCmpt(F.UzB, s0.UB, 2);
        X_.allocate(nC, cs.nIF, nB);

        // the patch fields, through brae's device boundary (the same flat layout: no coupled patch is accepted)
        dbU_ = buildDeviceVectorBoundary(pf.U, cs.patches, cs.g);
        dbP_ = buildDeviceBoundary(pf.p, cs.patches, cs.g);
        dbT_ = buildDeviceBoundary(pf.T, cs.patches, cs.g);
        if (dbP_.n != nB || dbT_.n != nB || dbU_.comp[0].n != nB)
            throw std::runtime_error("brae rhoCentralFoam: the device boundary's face layout is not the case's.");
        {
            std::vector<int> fx(nB);
            for (label k = 0; k < nB; ++k) fx[k] = pf.T.boundary[cs.bPatch[k]]->fixesValue() ? 1 : 0;
            TfixesValue_.copyFrom(fx);
        }
        bCell_.copyFrom(cs.bCell);
        uTransform_.copyFrom(cs.bUTransform);
        uR_.copyFrom(cs.bUR);
        rhoConstraint_.copyFrom(cs.bRhoConstraint);
        TseedOwn_.copyFrom(cs.bTseedOwn);
        ioFace_.copyFrom(cs.bInletOutlet);
        ioCount_.copyFrom(std::vector<unsigned long long>(2, 0ull));
        if (control_ == controlInletOutletInverted || control_ == controlLaggedPhi) phiCtrl_.resize(nB);
        Told_.resize(nC);
        TbEval_.resize(nB);
        coCell_.resize(nC);
        status_.copyFrom(std::vector<int>(kReportWindow, 0));
        courant_.copyFrom(std::vector<double>(3*kReportWindow, 0.0));
        // pinned, so flush() reads a window back with two async copies and one synchronisation: under adjustTimeStep
        // it runs every step
        cudaCheck(cudaMallocHost(&hostCourant_, 3*kReportWindow*sizeof(double)), "pinned courant");
        cudaCheck(cudaMallocHost(&hostStatus_, kReportWindow*sizeof(int)), "pinned status");
        outflow_.copyFrom(std::vector<double>(10, 0.0));   // the run's totals, then this step's sums
        cudaCheck(cudaDeviceSynchronize(), "rhoCentralFoam upload");
        bytes_ = brae::detail::devicePool().mallocBytes() - brae::detail::devicePool().freedBytes() - pool0;
    }

    void step(scalar deltaT) override
    {
        if (pending_ >= kReportWindow)
            throw std::logic_error("brae rhoCentralFoam: device step with a full report window; flush() first.");
        const Case& cs = cs_;
        DeviceFields& F = F_;
        const int nC = cs.nC, nB = cs.nB;

        centralFluxes(dm_, cs, control_, F, X_, outflow_.data(), deltaT);

        const FluxesIn Fx{X_.phi.data(), X_.phiUx.data(), X_.phiUy.data(), X_.phiUz.data(), X_.phiEp.data(),
                          X_.amaxSf.data(), X_.phiB.data(), X_.phiUxB.data(), X_.phiUyB.data(), X_.phiUzB.data(),
                          X_.phiEpB.data(), X_.amaxSfB.data()};
        const CellState S{F.rho.data(), F.rhoUx.data(), F.rhoUy.data(), F.rhoUz.data(), F.rhoE.data(), F.e.data(),
                          F.T.data(), Told_.data(), F.p.data(), F.psi.data(), F.Ux.data(), F.Uy.data(), F.Uz.data()};
        cellKernel<<<blocksFor(nC), kTPB>>>(nC, pending_, 1.0/deltaT, cs.sd.valid(0), cs.sd.valid(1),
                                            cs.sd.valid(2), cs.gas, dm_.ownerStart.data(), dm_.losort.data(),
                                            dm_.losortStart.data(), dm_.bndCellStart.data(), dm_.bndPerm.data(),
                                            dm_.bndIsEmpty.data(), dm_.V.data(), Fx, S, coCell_.data(),
                                            status_.data());
        cudaCheck(cudaGetLastError(), "rhoCentral cells");

        if (nB)
        {
            // :195's phi decides every inletOutlet face of U, T and p before its evaluation (inletOutletFvPatchField::
            // updateCoeffs); the controls feed the switch another flux
            if (cs.ioU || cs.ioP || cs.ioT)
            {
                const DeviceBuffer<scalar>* phiSwitch = &X_.phiB;
                const cudaStream_t q = cudaStreamPerThread;
                if (control_ == controlInletOutletInverted)
                {
                    negateKernel<<<blocksFor(nB), kTPB>>>(nB, X_.phiB.data(), phiCtrl_.data());
                    cudaCheck(cudaGetLastError(), "rhoCentral -phi");
                    phiSwitch = &phiCtrl_;
                }
                else if (control_ == controlLaggedPhi)
                {
                    if (!lagPrimed_)   // the first step has no step before: its own
                        cudaCheck(cudaMemcpyAsync(phiCtrl_.data(), X_.phiB.data(), nB*sizeof(scalar),
                                                  cudaMemcpyDeviceToDevice, q), "lagged phi");
                    lagPrimed_ = true;
                    phiSwitch = &phiCtrl_;
                }
                if (cs.ioU) deviceUpdateInletOutlet(dbU_, *phiSwitch);
                if (cs.ioT) deviceUpdateInletOutlet(dbT_, *phiSwitch);
                if (cs.ioP) deviceUpdateInletOutlet(dbP_, *phiSwitch);
                if (control_ == controlLaggedPhi)   // what the next step will switch on
                    cudaCheck(cudaMemcpyAsync(phiCtrl_.data(), X_.phiB.data(), nB*sizeof(scalar),
                                              cudaMemcpyDeviceToDevice, q), "lagged phi");
            }
            deviceBCValue(dbU_.comp[0], F.Ux, F.UxB);   // :232
            deviceBCValue(dbU_.comp[1], F.Uy, F.UyB);
            deviceBCValue(dbU_.comp[2], F.Uz, F.UzB);
            // :265, Tw.evaluate() on the cells' old T, before thermo.correct() (the control: on the new)
            deviceBCValue(dbT_, control_ == controlInletOutletNewT ? F.T : Told_, TbEval_);
            const PatchState B{F.rhoB.data(), F.rhoUxB.data(), F.rhoUyB.data(), F.rhoUzB.data(), F.rhoEB.data(),
                               F.eB.data(), F.TB.data(), F.psiB.data(), F.UxB.data(), F.UyB.data(), F.UzB.data(),
                               F.pB.data(), TbEval_.data()};
            const PatchTransforms X{F.rho.data(), F.Ux.data(), F.Uy.data(), F.Uz.data(), uTransform_.data(),
                                    uR_.data(), rhoConstraint_.data(), TseedOwn_.data()};
            patchKernel<<<blocksFor(nB), kTPB>>>(nB, pending_, cs.gas, control_, bCell_.data(), TfixesValue_.data(),
                                                 F.e.data(), B, X, status_.data(), X_.phiB.data(), ioFace_.data(),
                                                 ioCount_.data());
            cudaCheck(cudaGetLastError(), "rhoCentral patches");
            deviceBCValue(dbP_, F.p, F.pB);             // :287
            rhoBKernel<<<blocksFor(nB), kTPB>>>(nB, F.psiB.data(), F.pB.data(), F.rhoB.data());   // :288
            cudaCheck(cudaGetLastError(), "rhoCentral rho_b");
        }
        courantKernel<<<1, 1024>>>(nC, coCell_.data(), dm_.V.data(), courant_.data() + 3*pending_);
        cudaCheck(cudaGetLastError(), "rhoCentral courant");
        deltaT_[pending_] = deltaT;
        ++pending_;
    }

    std::vector<StepReport> flush() override
    {
        std::vector<StepReport> out(pending_);
        if (!pending_) return out;
        const double* co = hostCourant_;
        const int* st = hostStatus_;
        const cudaStream_t q = cudaStreamPerThread;   // the stream the kernels run on (--default-stream per-thread)
        cudaCheck(cudaMemcpyAsync(hostCourant_, courant_.data(), 3*pending_*sizeof(double), cudaMemcpyDeviceToHost, q),
                  "courant D2H");
        cudaCheck(cudaMemcpyAsync(hostStatus_, status_.data(), pending_*sizeof(int), cudaMemcpyDeviceToHost, q),
                  "status D2H");
        cudaCheck(cudaMemsetAsync(status_.data(), 0, pending_*sizeof(int), q), "status reset");
        cudaCheck(cudaStreamSynchronize(q), "reports");
        for (int i = 0; i < pending_; ++i)
        {
            if (cs_.nIF)
            {
                out[i].CoNum = 0.5*co[3*i]*deltaT_[i];
                out[i].meanCoNum = 0.5*(co[3*i + 1]/co[3*i + 2])*deltaT_[i];
            }
            out[i].status = st[i];
        }
        pending_ = 0;
        return out;
    }
    int pending() const override { return pending_; }
    int window() const override { return kReportWindow; }

    void download(State& s) override
    {
        const DeviceFields& F = F_;
        s.rho = F.rho.host();
        downCmpt(F.rhoUx, s.rhoU, 0);
        downCmpt(F.rhoUy, s.rhoU, 1);
        downCmpt(F.rhoUz, s.rhoU, 2);
        s.rhoE = F.rhoE.host();
        s.e = F.e.host();
        s.T = F.T.host();
        s.p = F.p.host();
        s.psi = F.psi.host();
        downCmpt(F.Ux, s.U, 0);
        downCmpt(F.Uy, s.U, 1);
        downCmpt(F.Uz, s.U, 2);
        s.rhoB = F.rhoB.host();
        downCmpt(F.rhoUxB, s.rhoUB, 0);
        downCmpt(F.rhoUyB, s.rhoUB, 1);
        downCmpt(F.rhoUzB, s.rhoUB, 2);
        s.rhoEB = F.rhoEB.host();
        s.eB = F.eB.host();
        s.TB = F.TB.host();
        s.pB = F.pB.host();
        s.psiB = F.psiB.host();
        downCmpt(F.UxB, s.UB, 0);
        downCmpt(F.UyB, s.UB, 1);
        downCmpt(F.UzB, s.UB, 2);
    }

    void boundaryOutflow(scalar& mass, vector& mom, scalar& energy) const override
    {
        const std::vector<double> r = outflow_.host();
        mass = r[0];
        mom = vector{r[1], r[2], r[3]};
        energy = r[4];
    }

    void inletOutletInflow(long long& inflow, long long& faceSteps) const override
    {
        const std::vector<unsigned long long> r = ioCount_.host();
        inflow = static_cast<long long>(r[0]);
        faceSteps = static_cast<long long>(r[1]);
    }

    std::size_t deviceBytes() const override { return bytes_; }

    ~DeviceArm() override
    {
        cudaFreeHost(hostCourant_);
        cudaFreeHost(hostStatus_);
    }
    DeviceArm(const DeviceArm&) = delete;
    DeviceArm& operator=(const DeviceArm&) = delete;

private:
    const Case& cs_;
    int control_;
    DeviceMesh dm_;
    DeviceFields F_;
    DeviceFluxes X_;
    DeviceVectorBoundary dbU_;
    DeviceBoundary dbP_, dbT_;
    DeviceBuffer<int> TfixesValue_;
    DeviceBuffer<label> bCell_;
    DeviceBuffer<int> uTransform_;
    DeviceBuffer<tensor> uR_;
    DeviceBuffer<char> rhoConstraint_, TseedOwn_, ioFace_;
    DeviceBuffer<unsigned long long> ioCount_;
    DeviceBuffer<scalar> phiCtrl_;   // the controls' switch flux: -phi, or the step before's phi
    bool lagPrimed_ = false;
    DeviceBuffer<scalar> Told_, TbEval_, coCell_;
    DeviceBuffer<int> status_;
    DeviceBuffer<double> courant_, outflow_;
    scalar deltaT_[kReportWindow] = {};
    double* hostCourant_ = nullptr;
    int* hostStatus_ = nullptr;   // each pending step's deltaT, which its Courant numbers are scaled by
    int pending_ = 0;
    std::size_t bytes_ = 0;
};

} // namespace

std::unique_ptr<Arm> makeDeviceArm(const Case& cs, const State& s0, PatchFields& pf, int control)
{
    return std::make_unique<DeviceArm>(cs, s0, pf, control);
}

} // namespace rhoCentral
} // namespace gpu
} // namespace brae
