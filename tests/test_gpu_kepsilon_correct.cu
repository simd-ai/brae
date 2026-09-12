// GPU offload (#3): the CLOSED device kEpsilon::correct() incl. the wall setValues constraint. Run one
// CPU correct() and one device correct() from the same state and compare the resulting k / eps / nut.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "k_epsilon.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

static scalar relMax(const std::vector<scalar>& a, const std::vector<scalar>& b) {
    scalar n = 0, d = 0; for (std::size_t i = 0; i < a.size(); ++i) { n = std::fmax(n, std::fabs(a[i]-b[i])); d = std::fmax(d, std::fabs(b[i])); }
    return d > 0 ? n / d : n;
}

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const scalar nu = 1e-3, relaxEps = 0.7, relaxK = 0.7, tol = 1e-9;

    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")      U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")  U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                        U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    auto mkKE = [&](scalar inletVal, scalar a, scalar b, scalar f) { GeometricField<scalar> s; s.internal.resize(nC);
        for (label c = 0; c < nC; ++c) s.internal[c] = a + b*std::sin(f*c);
        for (const FvPatch& q : fvp) { if (q.type=="empty") s.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
            else if (q.name=="inlet") s.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q,true,inletVal,std::vector<scalar>{}));
            else s.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); } s.evaluateBoundary(); return s; };
    GeometricField<scalar> k = mkKE(0.2, 0.15, 0.03, 0.01), eps = mkKE(2.0, 1.5, 0.5, 0.013);
    GeometricField<scalar> nut; nut.internal.resize(nC); for (label c = 0; c < nC; ++c) nut.internal[c] = 0.09*k.internal[c]*k.internal[c]/eps.internal[c];
    for (const FvPatch& q : fvp) { if (q.type=="empty") nut.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (q.type=="wall") nut.boundary.push_back(std::make_unique<CalculatedPatchField<scalar>>(q, true, 0.0, std::vector<scalar>{}));
        else nut.boundary.push_back(std::make_unique<CalculatedPatchField<scalar>>(q, true, 0.0, std::vector<scalar>{})); }
    nut.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    // device state (snapshot the inputs BEFORE the CPU correct() mutates k/eps/nut).
    const std::vector<scalar> kIn = k.internal, eIn = eps.internal, nIn = nut.internal;
    // nut's OWN boundary, snapshotted before correct() overwrites it. DkEff(patchi)/DepsilonEff(patchi)
    // are built from this on BOTH paths -- passing it on one side only is what made this test fail.
    std::vector<scalar> nutBIn;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const std::vector<scalar>& b = nut.boundary[pi]->value();
        nutBIn.insert(nutBIn.end(), b.begin(), b.end());
    }

    // CPU correct() (mutates k, eps, nut).
    kepsilon::correct(U, k, eps, nut, phi, nu, m, g, fvp, relaxEps, relaxK, tol, 0.0, 3000);
    const std::vector<scalar> kCpu = k.internal, eCpu = eps.internal, nCpu = nut.internal;

    // Device correct().
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    const DeviceBoundary dbEps = buildDeviceBoundary(eps, fvp, g), dbK = buildDeviceBoundary(k, fvp, g);
    const DeviceWallData wall = buildDeviceWallData(m, g, fvp, U);
    std::vector<scalar> ux(nC),uy(nC),uz(nC); for (label c=0;c<nC;++c){ux[c]=U.internal[c].x;uy[c]=U.internal[c].y;uz[c]=U.internal[c].z;}
    DeviceBuffer<scalar> dUx(ux),dUy(uy),dUz(uz), dk(kIn), de(eIn), dnut(nIn);
    DeviceBuffer<scalar> phiB(([&]{std::vector<scalar> o; for(auto&a:phi.boundary)o.insert(o.end(),a.begin(),a.end());return o;}()));
    const DeviceBuffer<scalar> dNutB(nutBIn);
    deviceKEpsilonCorrect(dm, wall, dbEps, dbK, dbU, dUx, dUy, dUz, dk, de, dnut,
                          DeviceBuffer<scalar>(phi.internal), phiB, nu, relaxEps, relaxK, tol,
                          /*bounded*/false, /*boundedEps*/false,
                          /*limitedK*/false, /*limitedEps*/false, 2.0, 2.0,
                          /*co*/{}, /*relTolKE*/0.0, /*keCheckEvery*/1,
                          /*linearUpwindK*/false, /*linearUpwindEps*/false, /*nonOrth*/false,
                          /*gsK*/false, /*gsEps*/false, /*ami*/nullptr, /*cyc*/nullptr,
                          /*nutWall*/0, /*atmZ0*/0.0, /*atmBoundNut*/true,
                          /*kDdt*/{}, /*eDdt*/{},
                          /*rho*/nullptr, /*muLam*/nullptr, /*rhoBnd*/nullptr, /*nuWallFace*/nullptr,
                          &dNutB);

    std::printf("GPU closed kEpsilon::correct (nCells=%d):\n", nC);
    std::printf("  eps device vs CPU : %.3e\n", relMax(de.host(), eCpu));
    std::printf("  k   device vs CPU : %.3e\n", relMax(dk.host(), kCpu));
    std::printf("  nut device vs CPU : %.3e\n", relMax(dnut.host(), nCpu));
    const bool pass = relMax(de.host(), eCpu) < 1e-5 && relMax(dk.host(), kCpu) < 1e-5 && relMax(dnut.host(), nCpu) < 1e-5;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
