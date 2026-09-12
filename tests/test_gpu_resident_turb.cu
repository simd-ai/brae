// GPU offload, the TURBULENT device-resident driver. The full SIMPLE + k-epsilon loop on the GPU: each
// iteration runs momentum (div - laplacian(nuEff), nuEff = nu + nut) + pressure + corrector, then the closed
// device kEpsilon::correct(). U/p/phi/k/eps/nut all stay on the GPU. Validate N steps vs the CPU turbulent
// loop (same simplified momentum: div - laplacian(nuEff), no under-relaxation of the divDevReff term).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "pcg.cuh"
#include "k_epsilon.cuh"
#include "near_wall_dist.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"
#include "device_divdevreff.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const scalar nu = 1e-3, tol = 1e-10, relaxU = 0.7, relaxP = 0.3, relaxK = 0.7, relaxEps = 0.7; const int N = 5;

    auto mkU = [&]() { GeometricField<vector> U; U.internal.resize(nC);
        for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
        for (const FvPatch& q : fvp) { if (q.type=="empty") U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
            else if (q.name=="inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q,true,vector{10,0,0},std::vector<vector>{}));
            else if (q.type=="wall") U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
            else U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q)); } U.evaluateBoundary(); return U; };
    auto mkP = [&]() { GeometricField<scalar> p; p.internal.assign(nC, 0.0);
        for (const FvPatch& q : fvp) { if (q.type=="empty") p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
            else if (q.name=="outlet") p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q,true,0.0,std::vector<scalar>{}));
            else p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); } p.evaluateBoundary(); return p; };
    auto mkKE = [&](scalar inlet, scalar a, scalar b, scalar f) { GeometricField<scalar> s; s.internal.resize(nC);
        for (label c = 0; c < nC; ++c) s.internal[c] = a + b*std::sin(f*c);
        for (const FvPatch& q : fvp) { if (q.type=="empty") s.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
            else if (q.name=="inlet") s.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q,true,inlet,std::vector<scalar>{}));
            else s.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); } s.evaluateBoundary(); return s; };
    auto mkNut = [&](const std::vector<scalar>& v) { GeometricField<scalar> s; s.internal = v;
        for (const FvPatch& q : fvp) { if (q.type=="empty") s.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q)); else s.boundary.push_back(std::make_unique<CalculatedPatchField<scalar>>(q,true,0.0,std::vector<scalar>{})); } s.evaluateBoundary(); return s; };

    // ---------------- CPU turbulent loop ----------------
    std::vector<scalar> Ucx, Ucy, pc, kc, ec;
    {
        GeometricField<vector> U = mkU(); GeometricField<scalar> p = mkP();
        GeometricField<scalar> k = mkKE(0.2, 0.15, 0.03, 0.01), eps = mkKE(2.0, 1.5, 0.5, 0.013);
        std::vector<scalar> nv(nC); for (label c=0;c<nC;++c) nv[c]=0.09*k.internal[c]*k.internal[c]/eps.internal[c];
        GeometricField<scalar> nut = mkNut(nv);
        SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
        for (int it = 0; it < N; ++it) {
            std::vector<scalar> nuEff(nC); for (label c=0;c<nC;++c) nuEff[c] = nu + nut.internal[c];
            const SurfaceScalarField nuEff_f = fvc::interpolate(nuEff, m, g, fvp);
            const std::vector<vector> gP = fvc::gaussGrad(p, m, g, fvp);
            FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, m, fvp);
            addEqual(UEqn, fvm::laplacian(nuEff_f, U, m, g, fvp), -1.0);
            // explicit divDevReff stress: source += V*fvc::div(nuEff*dev2(T(grad U))). nuEff at boundary faces =
            // the adjacent-cell value (the convention this test uses for the laplacian boundary too).
            { const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, fvp);
              const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, m, g, fvp);
              std::vector<tensor> sigC(nC); for (label c=0;c<nC;++c) sigC[c]=nuEff[c]*dev2(transpose(gradC[c]));
              std::vector<std::vector<tensor>> sigB(fvp.size());
              for (std::size_t pi=0;pi<fvp.size();++pi){ sigB[pi].resize(fvp[pi].size);
                  for (label i=0;i<fvp[pi].size;++i) sigB[pi][i]=nuEff[fvp[pi].faceCells[i]]*dev2(transpose(gradB[pi][i])); }
              const std::vector<vector> divSig = fvc::div(sigC, sigB, m, g, fvp);
              for (label c=0;c<nC;++c) UEqn.source[c]+=g.V()[c]*divSig[c]; }
            relaxMatrix(UEqn, U, m, fvp, relaxU);
            FvVectorMatrix Mp = UEqn; for (label c=0;c<nC;++c) Mp.source[c] += (-g.V()[c])*gP[c];
            solveVector(Mp, U, m, fvp, tol, 0.0, 2000);
            const std::vector<scalar> A = matrixA(UEqn, m, g, fvp); std::vector<scalar> rAU(nC); for (label c=0;c<nC;++c) rAU[c]=1.0/A[c];
            const std::vector<vector> H = matrixH(UEqn, U, m, g, fvp);
            GeometricField<vector> HbyA = mkU(); for (label c=0;c<nC;++c) HbyA.internal[c]=rAU[c]*H[c]; HbyA.evaluateBoundary();
            SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, g, fvp);
            const SurfaceScalarField rAUf = fvc::interpolate(rAU, m, g, fvp);
            FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, m, g, fvp);
            const std::vector<scalar> dphi = fvc::div(phiHbyA, m, g, fvp); for (label c=0;c<nC;++c) pEqn.source[c]+=g.V()[c]*dphi[c];
            const std::vector<scalar> pPrev = p.internal; pcg(pEqn, p.internal, m, fvp, tol, 0.0, 10000, 0);
            const SurfaceScalarField pflux = matrixFlux(pEqn, p, m, fvp);
            for (label f=0;f<nIf;++f) phi.internal[f]=phiHbyA.internal[f]-pflux.internal[f];
            for (std::size_t pi=0;pi<fvp.size();++pi) for (label i=0;i<fvp[pi].size;++i) phi.boundary[pi][i]=phiHbyA.boundary[pi][i]-pflux.boundary[pi][i];
            for (label c=0;c<nC;++c) p.internal[c]=pPrev[c]+relaxP*(p.internal[c]-pPrev[c]); p.evaluateBoundary();
            const std::vector<vector> gPn = fvc::gaussGrad(p, m, g, fvp); for (label c=0;c<nC;++c) U.internal[c]=HbyA.internal[c]-rAU[c]*gPn[c]; U.evaluateBoundary();
            kepsilon::correct(U, k, eps, nut, phi, nu, m, g, fvp, relaxEps, relaxK, tol, 0.0, 3000);
        }
        Ucx.resize(nC); Ucy.resize(nC); pc=p.internal; kc=k.internal; ec=eps.internal;
        for (label c=0;c<nC;++c){ Ucx[c]=U.internal[c].x; Ucy[c]=U.internal[c].y; }
    }

    // ---------------- DEVICE turbulent resident loop ----------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    GeometricField<vector> U0 = mkU(); GeometricField<scalar> p0 = mkP();
    GeometricField<scalar> k0 = mkKE(0.2,0.15,0.03,0.01), e0 = mkKE(2.0,1.5,0.5,0.013);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U0, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(p0, fvp, g), dbEps = buildDeviceBoundary(e0, fvp, g), dbK = buildDeviceBoundary(k0, fvp, g);
    const DeviceWallData wall = buildDeviceWallData(m, g, fvp, U0);
    // nut's boundary in this case: mkNut gives every non-empty patch a `calculated` field of ZERO, and
    // with no wall patches nothing rewrites it. DkEff(patchi) is therefore nu, on BOTH paths -- the host
    // reads it from the field, so the device is handed the same numbers rather than the cell value.
    DeviceBuffer<scalar> dNutB;
    {
        GeometricField<scalar> nutRef = mkNut(std::vector<scalar>(nC, 0.0));
        std::vector<scalar> flat;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            const std::vector<scalar>& b = nutRef.boundary[pi]->value();
            flat.insert(flat.end(), b.begin(), b.end());
        }
        dNutB.copyFrom(flat);
    }
    // all-extrapolated boundary: deviceBCValue gathers a cell field to boundary faces (nuEff_bnd = cell value),
    // the nuBnd input for the divDevReff boundary stress.
    DeviceBoundary dbExtrap;
    { std::vector<label> ty, fc; std::vector<scalar> ref, dc, ms;
      for (std::size_t pi=0;pi<fvp.size();++pi) for (label i=0;i<fvp[pi].size;++i){
          ty.push_back(0); fc.push_back(fvp[pi].faceCells[i]); ref.push_back(0.0);
          dc.push_back(fvp[pi].deltaCoeffs[i]); ms.push_back(g.magSf()[fvp[pi].start+i]); }
      dbExtrap.n=(int)ty.size(); dbExtrap.bcType.copyFrom(ty); dbExtrap.refValue.copyFrom(ref);
      dbExtrap.deltaCoeffs.copyFrom(dc); dbExtrap.magSf.copyFrom(ms); dbExtrap.faceCell.copyFrom(fc); }
    const SurfaceScalarField phi0 = fvc::flux(U0, m, g, fvp);
    DeviceBuffer<scalar> Uk[3], dp(p0.internal), phiInt(phi0.internal), dk(k0.internal), de(e0.internal);
    { std::vector<scalar> ux(nC),uy(nC),uz(nC),nv(nC); for (label c=0;c<nC;++c){ux[c]=U0.internal[c].x;uy[c]=U0.internal[c].y;uz[c]=U0.internal[c].z;nv[c]=0.09*k0.internal[c]*k0.internal[c]/e0.internal[c];}
      Uk[0].copyFrom(ux);Uk[1].copyFrom(uy);Uk[2].copyFrom(uz); }
    DeviceBuffer<scalar> dnut(([&]{std::vector<scalar> nv(nC); for(label c=0;c<nC;++c) nv[c]=0.09*k0.internal[c]*k0.internal[c]/e0.internal[c]; return nv;}()));
    DeviceBuffer<scalar> phiBnd(([&]{std::vector<scalar> o; for(auto&a:phi0.boundary)o.insert(o.end(),a.begin(),a.end());return o;}()));
    DeviceBuffer<scalar> zeroSrc(std::vector<scalar>(nC,0.0)), zeroBndU(std::vector<scalar>(dbU.n,0.0)), nuConst(std::vector<scalar>(nC,nu));
    auto sm = [](DeviceBuffer<scalar>& b){ return deviceSumMag(b)+1e-20; };

    for (int it = 0; it < N; ++it) {
        DeviceBuffer<scalar> nuEff; deviceCopy(nuEff, dnut); deviceAxpy(1.0, nuConst, nuEff);     // nuEff = nu + nut
        DeviceBuffer<scalar> nuEff_f; deviceInterpolate(dm, nuEff, nuEff_f);
        // explicit divDevReff stress source (uses the incoming U; coupled across all 3 components)
        DeviceBuffer<scalar> nuBnd; deviceBCValue(dbExtrap, nuEff, nuBnd);
        DeviceBuffer<scalar> ddrX, ddrY, ddrZ; deviceDivDevReff(dm, dbU, Uk[0], Uk[1], Uk[2], nuEff, nuBnd, ddrX, ddrY, ddrZ);
        DeviceBuffer<scalar>* ddr[3] = { &ddrX, &ddrY, &ddrZ };
        DeviceBuffer<scalar> pbv; deviceBCValue(dbP, dp, pbv);
        DeviceBuffer<scalar> gx, gy, gz; deviceGaussGrad(dm, dp, pbv, gx, gy, gz);
        DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
        deviceDivUpwindCoeffs(dm, phiInt, mDiag, mUp, mLo); deviceLaplacianCoeffs(dm, nuEff_f, lD, lU, lL);
        deviceAxpy(-1.0,lD,mDiag); deviceAxpy(-1.0,lU,mUp); deviceAxpy(-1.0,lL,mLo);
        DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };
        DeviceBuffer<scalar> r0IC,r0BC,r0lIC,r0lBC; deviceBCDivCoeffs(dbU.comp[0], phiBnd, r0IC, r0BC); deviceBCLaplacianCoeffs(dbU.comp[0], nuEff, r0lIC, r0lBC); deviceAxpy(-1.0,r0lIC,r0IC);
        DeviceBuffer<scalar> mDiagR, delta; deviceRelaxDiag(deviceLduView(dm,mDiag,mUp,mLo), dm, r0IC, relaxU, mDiagR, delta);
        const DeviceLduView Uview = deviceLduView(dm, mDiagR, mUp, mLo);
        DeviceBuffer<scalar> iC[3], bCb[3], relaxSrc[3];
        for (int kk=0;kk<3;++kk){ DeviceBuffer<scalar> dIC,dBC,lIC,lBC; deviceBCDivCoeffs(dbU.comp[kk],phiBnd,dIC,dBC); deviceBCLaplacianCoeffs(dbU.comp[kk],nuEff,lIC,lBC);
            deviceAxpy(-1.0,lIC,dIC); deviceAxpy(-1.0,lBC,dBC); iC[kk]=std::move(dIC); bCb[kk]=std::move(dBC);
            deviceHadamard(relaxSrc[kk], delta, Uk[kk]);
            deviceAxpy(1.0, *ddr[kk], relaxSrc[kk]);     // += explicit divDevReff stress (into solve source + H)
            DeviceBuffer<scalar> s; deviceHadamard(s, dm.V, *gg[kk]); deviceScale(s,-1.0); deviceAxpy(1.0,relaxSrc[kk],s);
            DeviceBuffer<scalar> diagC,b; deviceFold(dm, mDiagR, s, iC[kk], bCb[kk], diagC, b);
            deviceJacobiBiCGStab(deviceLduView(dm,diagC,mUp,mLo), b, Uk[kk], sm(b), tol, 0.0, 5000); }
        DeviceBuffer<scalar> diagA,dumb,rAU; deviceFold(dm,mDiagR,zeroSrc,iC[0],zeroBndU,diagA,dumb); deviceReciprocalV(dm,diagA,rAU);
        DeviceBuffer<scalar> HbyA[3];
        for (int kk=0;kk<3;++kk){ DeviceBuffer<scalar> Hk; deviceMatrixH(Uview, dm, Uk[kk], relaxSrc[kk], zeroBndU, bCb[kk], Hk); deviceHadamard(HbyA[kk], rAU, Hk); }
        DeviceBuffer<scalar> phiHi; deviceVectorFlux(dm, HbyA[0], HbyA[1], HbyA[2], phiHi);
        DeviceBuffer<scalar> hxb,hyb,hzb; deviceBCValue(dbU.comp[0],HbyA[0],hxb); deviceBCValue(dbU.comp[1],HbyA[1],hyb); deviceBCValue(dbU.comp[2],HbyA[2],hzb);
        DeviceBuffer<scalar> phiHb; deviceBoundaryFlux(dm,hxb,hyb,hzb,phiHb);
        DeviceBuffer<scalar> rAUf; deviceInterpolate(dm,rAU,rAUf);
        DeviceBuffer<scalar> pD,pU,pL; deviceLaplacianCoeffs(dm,rAUf,pD,pU,pL);
        DeviceBuffer<scalar> pIC,pBC; deviceBCLaplacianCoeffs(dbP, rAU, pIC, pBC);
        DeviceBuffer<scalar> divPhiH; deviceDiv(dm, phiHi, phiHb, divPhiH);
        DeviceBuffer<scalar> diagCp, bp; deviceFoldPressure(dm, pD, divPhiH, pIC, pBC, diagCp, bp);
        DeviceBuffer<scalar> pPrev; deviceCopy(pPrev, dp);
        deviceJacobiPCG(deviceLduView(dm,diagCp,pU,pL), bp, dp, sm(bp), tol, 0.0, 30000);
        const DeviceLduView pview = deviceLduView(dm,diagCp,pU,pL);
        DeviceBuffer<scalar> pfi; deviceMatrixFluxInternal(pview, dp, pfi); deviceCopy(phiInt, phiHi); deviceAxpy(-1.0, pfi, phiInt);
        DeviceBuffer<scalar> pfb; deviceMatrixFluxBoundary(dbP, pIC, pBC, dp, pfb); deviceCopy(phiBnd, phiHb); deviceAxpy(-1.0, pfb, phiBnd);
        deviceScale(dp, relaxP); deviceAxpy(1.0-relaxP, pPrev, dp);
        DeviceBuffer<scalar> pbv2; deviceBCValue(dbP, dp, pbv2);
        DeviceBuffer<scalar> gnx,gny,gnz; deviceGaussGrad(dm, dp, pbv2, gnx,gny,gnz);
        DeviceBuffer<scalar>* gn[3]={&gnx,&gny,&gnz}; for (int kk=0;kk<3;++kk){ DeviceBuffer<scalar> Un; deviceCorrector(HbyA[kk], rAU, *gn[kk], Un); Uk[kk]=std::move(Un); }
        // turbulence
        // DkEff(patchi)/DepsilonEff(patchi) come from nut's OWN boundary on the host path, so the device
        // is given the same thing -- rebuilt from the CURRENT k each step, exactly as the host's
        // correctNut rewrites the wall value at the end of every correct().
        deviceKEpsilonCorrect(dm, wall, dbEps, dbK, dbU, Uk[0], Uk[1], Uk[2], dk, de, dnut, phiInt, phiBnd,
                              nu, relaxEps, relaxK, tol,
                              /*bounded*/false, /*boundedEps*/false,
                              /*limitedK*/false, /*limitedEps*/false, 2.0, 2.0,
                              /*co*/{}, /*relTolKE*/0.0, /*keCheckEvery*/1,
                              /*linearUpwindK*/false, /*linearUpwindEps*/false, /*nonOrth*/false,
                              /*gsK*/false, /*gsEps*/false, /*ami*/nullptr, /*cyc*/nullptr,
                              /*nutWall*/0, /*atmZ0*/0.0, /*atmBoundNut*/true,
                              /*kDdt*/{}, /*eDdt*/{},
                              /*rho*/nullptr, /*muLam*/nullptr, /*rhoBnd*/nullptr, /*nuWallFace*/nullptr,
                              &dNutB);
    }

    const std::vector<scalar> uxg=Uk[0].host(), uyg=Uk[1].host(), pg=dp.host(), kg=dk.host(), eg=de.host();
    auto rel=[&](const std::vector<scalar>&a,const std::vector<scalar>&b){scalar n=0,d=0;for(label c=0;c<nC;++c){n=std::fmax(n,std::fabs(a[c]-b[c]));d=std::fmax(d,std::fabs(b[c]));}return n/d;};
    std::printf("GPU TURBULENT resident loop (%d steps, nCells=%d, U/p/phi/k/eps/nut all on GPU):\n", N, nC);
    std::printf("  U.x %.3e  U.y %.3e  p %.3e  k %.3e  eps %.3e\n", rel(uxg,Ucx), rel(uyg,Ucy), rel(pg,pc), rel(kg,kc), rel(eg,ec));
    const bool pass = rel(uxg,Ucx)<1e-4 && rel(uyg,Ucy)<1e-4 && rel(pg,pc)<1e-4 && rel(kg,kc)<1e-4 && rel(eg,ec)<1e-4;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
