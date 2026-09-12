// The cyclicAMI CUDA path against the _cpp REFERENCE, one stage at a time.
//
// Each check pairs one device entry point with its host twin at IDENTICAL inputs. That is the property
// the reference exists for: brae's device AMI is a fused path, and until now a disagreement anywhere in
// it produced one number for the whole interface. On pipeCyclic 97% of the momentum residual sits on
// interface cells with the interior exactly zero, and four passes of reading the device code did not
// explain it -- because reading is not measuring.
//
// The inputs are the case's own converged fields, so neither side is fed the other's answer, and the
// bounds are 1e-12: these are the same arithmetic in two places, not two discretisations.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "ami_interface.cuh"
#include "cyclic_interface.cuh"
#include "cyclicAMI_cpp.cuh"
#include "device_mesh.cuh"
#include "device_ami.cuh"
#include "fvc.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static int report(const char* name, const std::vector<scalar>& host, const std::vector<scalar>& dev,
                  scalar bound)
{
    if (host.size() != dev.size())
    {
        std::printf("  %-26s SIZE MISMATCH host %zu device %zu   FAIL\n", name, host.size(), dev.size());
        return 1;
    }
    scalar linf = 0;
    int worst = -1;
    for (std::size_t i = 0; i < host.size(); ++i)
    {
        const scalar r = std::fabs(dev[i] - host[i]) / std::fmax(std::fabs(host[i]), 1e-30);
        if (r > linf) { linf = r; worst = (int)i; }
    }
    // A bound of exactly zero means EXACT equality is required, so it has to compare <= rather than <.
    // Used where the two sides are the same kernel on the same data and any difference at all is a bug,
    // not a tolerance question -- e.g. an explicitly-supplied default reproducing the implicit one.
    const bool ok = (bound > 0.0) ? (linf < bound) : (linf <= 0.0);
    std::printf("  %-26s L_inf rel %.3e   bound %.1e   %s\n", name, linf, bound, ok ? "ok" : "FAIL");
    if (!ok && worst >= 0)
        std::printf("      worst entry %d:  _cpp %.12e   device %.12e\n", worst, host[worst], dev[worst]);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <time>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);
    if (amis.empty()) { std::printf("SKIP: no cyclicAMI interface in %s\n", caseDir.c_str()); return 77; }

    auto rd = [&](const std::string& f) {
        GeometricField<scalar> x =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + f), fvp, nC);
        x.evaluateBoundary();
        return x;
    };
    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    U.evaluateBoundary();
    GeometricField<scalar> p = rd("p"), nut = rd("nut");
    std::vector<scalar> nuEff(nC);
    for (label c = 0; c < nC; ++c) nuEff[c] = nut.internal[c] + 1e-6;

    // The gate must run on a NON-CONFORMING interface. A conformal AMI has one stencil entry per source
    // face with weight exactly 1, so the weighted sum is a copy and the multi-entry path -- the thing
    // that makes an AMI an AMI rather than a cyclic -- is never executed. The first version of this gate
    // shipped a conformal case and passed twelve stages without touching it.
    // ...unless this run is the CROSS-CHECK, which needs the opposite. The two requirements are in
    // tension on purpose: the stencil test needs a non-conforming interface, and an AMI only has a
    // cyclic twin to be checked against when it IS conforming. So the two run as separate invocations.
    if (argc < 4)
    {
        label multi = 0, maxLen = 0;
        for (const AMIInterface& a : amis)
            for (std::size_t i = 0; i < a.ownCell.size(); ++i)
            {
                const label len = a.srcOffset[i+1] - a.srcOffset[i];
                maxLen = std::max(maxLen, len);
                if (len > 1) ++multi;
            }
        std::printf("  stencil: %d source face(s) with more than one entry, longest %d\n",
                    (int)multi, (int)maxLen);
        if (!multi)
        {
            std::printf("  FAIL: every stencil is 1:1, so this case exercises no AMI interpolation at "
                        "all -- use a non-conforming interface\n");
            return 1;
        }
    }

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceAMI ami = buildDeviceAMI(amis);
    std::printf("ami_cpp_vs_device: %s/%s   %d interface(s), %d source faces\n",
                caseDir.c_str(), t.c_str(), (int)amis.size(), (int)ami.n);

    // The device packs every interface end to end; the reference works one at a time, so the host
    // results are concatenated in the same order.
    auto cat = [&](auto fn) {
        std::vector<scalar> out;
        for (const AMIInterface& a : amis) { const auto v = fn(a); out.insert(out.end(), v.begin(), v.end()); }
        return out;
    };

    // The state has to actually EXERCISE the stages. On a uniform axial initial field the interface
    // faces are azimuthal, so phi = U.Sf is ~0 and the momentum assembly's upwind split never fires --
    // the gate would pass on arithmetic it never ran. These ranges are printed so a degenerate input
    // cannot masquerade as agreement, and the flux check below refuses one outright.
    int rc = 0;

    // ---- STAGE 0: the AMI GEOMETRY itself, against the separately-gated cyclic geometry -----------
    // Every stage below is a function of (weights, deltaCoeffs, Sf). Those come from
    // buildAMIInterfaces and had no independent check: proving the device reproduces them says nothing
    // about whether they are RIGHT. On a CONFORMAL interface an AMI is a cyclic, so the same mesh
    // declared cyclic must give the same numbers face for face -- and the cyclic geometry is covered by
    // cyclic_geometry/cyclic_rotational. Pass a second case dir (same mesh, patches declared cyclic) to
    // run the cross-check; it is skipped otherwise, since a non-conforming AMI has no cyclic twin.
    if (argc >= 4)
    {
        PrimitiveMesh mc;
        mc.read(std::string(argv[3]) + "/constant/polyMesh");
        FvGeometry gc;
        gc.build(mc);
        const std::vector<FvPatch> fvpc = buildPatches(mc, gc);
        const std::vector<CyclicInterface> cycs = buildCyclicInterfaces(mc, gc, fvpc);
        std::vector<scalar> cDc, cW, aDc, aW;
        for (const CyclicInterface& c : cycs)
        {
            cDc.insert(cDc.end(), c.deltaCoeffs.begin(), c.deltaCoeffs.end());
            cW.insert(cW.end(), c.weights.begin(), c.weights.end());
        }
        for (const AMIInterface& a : amis)
        {
            aDc.insert(aDc.end(), a.deltaCoeffs.begin(), a.deltaCoeffs.end());
            aW.insert(aW.end(), a.weights.begin(), a.weights.end());
        }
        std::printf("  cross-check against the same mesh declared cyclic (%d interfaces):\n",
                    (int)cycs.size());
        rc |= report("  deltaCoeffs AMI vs cyclic", cDc, aDc, 1e-12);
        rc |= report("  weights     AMI vs cyclic", cW,  aW,  1e-12);
    }

    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    DeviceBuffer<scalar> dUx, dUy, dUz, dP, dNuEff, dV;
    dUx.copyFrom(ux); dUy.copyFrom(uy); dUz.copyFrom(uz);
    dP.copyFrom(p.internal); dNuEff.copyFrom(nuEff); dV.copyFrom(g.V());

    // ---- STAGE 1: scalar interpolate-to-source ---------------------------------------------------
    {
        DeviceBuffer<scalar> o;
        deviceAmiInterpolate(ami, dP, o);
        rc |= report("interpolate(p)", cat([&](const AMIInterface& a){
            return cpu::cyclicAMI::interpolate(a, p.internal); }), o.host(), 1e-12);
    }

    // ---- STAGE 2: rotated vector interpolate-to-source -------------------------------------------
    {
        DeviceBuffer<scalar> oX, oY, oZ;
        deviceAmiInterpolateVec(ami, dUx, dUy, dUz, oX, oY, oZ);
        for (int comp = 0; comp < 3; ++comp)
        {
            const std::vector<scalar> hv = cat([&](const AMIInterface& a){
                const std::vector<vector> v = cpu::cyclicAMI::interpolateVec(a, U.internal);
                std::vector<scalar> c(v.size());
                for (std::size_t i = 0; i < v.size(); ++i) c[i] = comp == 0 ? v[i].x : comp == 1 ? v[i].y : v[i].z;
                return c;
            });
            rc |= report(comp == 0 ? "interpolateVec(U).x" : comp == 1 ? "interpolateVec(U).y"
                                                                       : "interpolateVec(U).z",
                         hv, (comp == 0 ? oX : comp == 1 ? oY : oZ).host(), 1e-12);
        }
    }

    // ---- STAGE 3: the face value ------------------------------------------------------------------
    {
        DeviceBuffer<scalar> o;
        deviceAmiFaceValue(ami, dNuEff, o);
        rc |= report("faceValue(nuEff)", cat([&](const AMIInterface& a){
            return cpu::cyclicAMI::faceValue(a, nuEff); }), o.host(), 1e-12);
    }

    // ---- STAGE 4: the pressure (laplacian) interface assembly -------------------------------------
    {
        DeviceBuffer<scalar> diagD;
        diagD.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAssembleLaplacian(ami, dNuEff, diagD, true);
        std::vector<scalar> diagH(nC, 0.0), ifH;
        for (const AMIInterface& a : amis)
        {
            cpu::cyclicAMI::State st;
            cpu::cyclicAMI::assembleLaplacian(a, nuEff, st, diagH, true);
            ifH.insert(ifH.end(), st.ifCoeff.begin(), st.ifCoeff.end());
        }
        rc |= report("assembleLaplacian ifCoeff", ifH,   ami.ifCoeff.host(), 1e-12);
        rc |= report("assembleLaplacian diag",    diagH, diagD.host(),       1e-12);
    }

    // ---- STAGE 5: the matrix action ---------------------------------------------------------------
    {
        std::vector<scalar> ApsiH(nC, 0.0);
        {
            std::size_t off = 0;
            const std::vector<scalar> ifAll = ami.ifCoeff.host();
            for (const AMIInterface& a : amis)
            {
                cpu::cyclicAMI::State st;
                st.ifCoeff.assign(ifAll.begin() + off, ifAll.begin() + off + a.ownCell.size());
                off += a.ownCell.size();
                cpu::cyclicAMI::amul(a, st, p.internal, ApsiH);
            }
        }
        DeviceBuffer<scalar> ApsiD;
        ApsiD.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAmul(ami, dP, ApsiD);
        rc |= report("amul(p)", ApsiH, ApsiD.host(), 1e-12);
    }

    // ---- STAGE 6: the interface flux --------------------------------------------------------------
    {
        deviceAmiFlux(ami, dUx, dUy, dUz);
        rc |= report("flux(U)", cat([&](const AMIInterface& a){
            cpu::cyclicAMI::State st;
            cpu::cyclicAMI::flux(a, U.internal, st);
            return st.phi; }), ami.phi.host(), 1e-12);
    }

    // ---- STAGE 7: the MOMENTUM interface assembly -------------------------------------------------
    // The stage the open question points at: pipeCyclic's Uy momentum residual sits 97% on interface
    // cells. It runs after the flux above, because the upwind split needs the interface phi -- and both
    // sides are handed the SAME phi (the device's, which stage 6 just proved equals the reference's).
    const std::vector<scalar> phiIf = ami.phi.host();
    {
        DeviceBuffer<scalar> diagD;
        diagD.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAssembleMomentum(ami, dNuEff, diagD);
        std::vector<scalar> diagH(nC, 0.0), ifH;
        std::size_t off = 0;
        for (const AMIInterface& a : amis)
        {
            const std::vector<scalar> ph(phiIf.begin() + off, phiIf.begin() + off + a.ownCell.size());
            off += a.ownCell.size();
            cpu::cyclicAMI::State st;
            cpu::cyclicAMI::assembleMomentum(a, nuEff, ph, st, diagH);
            ifH.insert(ifH.end(), st.ifCoeff.begin(), st.ifCoeff.end());
        }
        rc |= report("assembleMomentum ifCoeff", ifH,   ami.ifCoeff.host(), 1e-12);
        rc |= report("assembleMomentum diag",    diagH, diagD.host(),       1e-12);

        // The scheme weight is now an INPUT rather than baked in. Upwind is the special case
        // w = pos0(phi), so passing it explicitly must reproduce the default byte for byte -- that is
        // what makes the new parameter provably inert on every case that does not set it, and it is the
        // only way to tell a correctly-plumbed default from one that is silently ignored.
        {
            std::vector<scalar> wUp(phiIf.size());
            for (std::size_t i = 0; i < phiIf.size(); ++i) wUp[i] = phiIf[i] > 0.0 ? 1.0 : 0.0;
            DeviceBuffer<scalar> wD;
            wD.copyFrom(wUp);
            DeviceBuffer<scalar> diagE;
            diagE.copyFrom(std::vector<scalar>(nC, 0.0));
            const std::vector<scalar> ifDefault = ami.ifCoeff.host();
            deviceAmiAssembleMomentum(ami, dNuEff, diagE, &wD);
            rc |= report("  explicit pos0(phi) == default", ifDefault, ami.ifCoeff.host(), 0.0);
            rc |= report("  ...and its diagonal",           diagD.host(), diagE.host(),     0.0);

            std::vector<scalar> diagG(nC, 0.0), ifG;
            std::size_t off2 = 0;
            for (const AMIInterface& a : amis)
            {
                const std::vector<scalar> ph(phiIf.begin() + off2, phiIf.begin() + off2 + a.ownCell.size());
                const std::vector<scalar> wa(wUp.begin() + off2, wUp.begin() + off2 + a.ownCell.size());
                off2 += a.ownCell.size();
                cpu::cyclicAMI::State st;
                cpu::cyclicAMI::assembleMomentum(a, nuEff, ph, st, diagG, &wa);
                ifG.insert(ifG.end(), st.ifCoeff.begin(), st.ifCoeff.end());
            }
            rc |= report("  reference, same weight",        ifG,   ami.ifCoeff.host(), 1e-12);
        }
    }

    // ---- STAGE 8: UEqn.H() ------------------------------------------------------------------------
    {
        DeviceBuffer<scalar> oX, oY, oZ;
        deviceAmiInterpolateVec(ami, dUx, dUy, dUz, oX, oY, oZ);
        const std::vector<scalar> UNx = oX.host();
        DeviceBuffer<scalar> Hd;
        Hd.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAddH(ami, oX, dV, Hd);
        std::vector<scalar> Hh(nC, 0.0);
        std::size_t off = 0;
        const std::vector<scalar> ifAll = ami.ifCoeff.host();
        for (const AMIInterface& a : amis)
        {
            cpu::cyclicAMI::State st;
            st.ifCoeff.assign(ifAll.begin() + off, ifAll.begin() + off + a.ownCell.size());
            const std::vector<scalar> un(UNx.begin() + off, UNx.begin() + off + a.ownCell.size());
            off += a.ownCell.size();
            cpu::cyclicAMI::addH(a, st, un, g.V(), Hh);
        }
        rc |= report("addH(U.x)", Hh, Hd.host(), 1e-12);
    }

    // ---- STAGE 9: the interface's share of div(phi) ------------------------------------------------
    {
        DeviceBuffer<scalar> divD;
        divD.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAddDiv(ami, dV, divD);
        std::vector<scalar> divH(nC, 0.0);
        std::size_t off = 0;
        for (const AMIInterface& a : amis)
        {
            cpu::cyclicAMI::State st;
            st.phi.assign(phiIf.begin() + off, phiIf.begin() + off + a.ownCell.size());
            off += a.ownCell.size();
            cpu::cyclicAMI::addDiv(a, st, g.V(), divH);
        }
        rc |= report("addDiv(phi)", divH, divD.host(), 1e-12);
    }

    // The cross-check invocation exists to compare GEOMETRY and is handed a conformal case at its
    // initial state, where the azimuthal interface legitimately carries no flux. Demanding a developed
    // field there would only force a second stored state for a check that does not read one.
    if (argc < 4)
    {
        scalar mx = 0;
        for (scalar v : phiIf) mx = std::fmax(mx, std::fabs(v));
        std::printf("  interface flux |phi| max %.4e  (the upwind split is only exercised when this is "
                    "nonzero)\n", mx);
        if (!(mx > 1e-12))
        {
            std::printf("  FAIL: the interface flux is zero on this state, so assembleMomentum's "
                        "convective split was never exercised -- use a developed field, not a uniform one\n");
            rc = 1;
        }
    }

    // ---- STAGE 10 and 11: the gradient contribution and the non-orthogonal correction ------------
    // Both sides are handed the SAME per-component gradient, computed once on the host, so these test
    // the two stages and not the gradient build (which has its own gates). The packing matters and is
    // easy to get backwards: the device holds gUx[l]/gUy[l]/gUz[l] = d(U_l)/d(x,y,z), i.e. row = the
    // velocity COMPONENT and column = the derivative direction -- the TRANSPOSE of what fvc::gaussGrad
    // returns, where gradU_ij = d(U_j)/d(x_i). The tensor handed to lapCorr below is in the device's
    // packing, built component by component so there is nothing to mistake.
    {
        std::vector<std::vector<vector>> gc(3);
        for (int l = 0; l < 3; ++l)
        {
            std::vector<scalar> comp(nC);
            for (label c = 0; c < nC; ++c)
                comp[c] = l == 0 ? U.internal[c].x : l == 1 ? U.internal[c].y : U.internal[c].z;
            std::vector<std::vector<scalar>> bnd(fvp.size());
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                const std::vector<vector>& bv = U.boundary[pi]->value();
                bnd[pi].resize(fvp[pi].size);
                for (label i = 0; i < fvp[pi].size; ++i)
                    bnd[pi][i] = l == 0 ? bv[i].x : l == 1 ? bv[i].y : bv[i].z;
            }
            gc[l] = fvc::gaussGrad(comp, bnd, m, g, fvp);
        }
        DeviceBuffer<scalar> gUx[3], gUy[3], gUz[3];
        for (int l = 0; l < 3; ++l)
        {
            std::vector<scalar> xs(nC), ys(nC), zs(nC);
            for (label c = 0; c < nC; ++c) { xs[c] = gc[l][c].x; ys[c] = gc[l][c].y; zs[c] = gc[l][c].z; }
            gUx[l].copyFrom(xs); gUy[l].copyFrom(ys); gUz[l].copyFrom(zs);
        }
        // gradU[c] in the DEVICE packing: (row, col) = (component, derivative direction).
        std::vector<tensor> gradU(nC);
        for (label c = 0; c < nC; ++c)
            gradU[c] = tensor{gc[0][c].x, gc[0][c].y, gc[0][c].z,
                              gc[1][c].x, gc[1][c].y, gc[1][c].z,
                              gc[2][c].x, gc[2][c].y, gc[2][c].z};

        // STAGE 10: the gradient's interface contribution, component 0.
        {
            DeviceBuffer<scalar> oX, oY, oZ, gxD, gyD, gzD;
            deviceAmiInterpolateVec(ami, dUx, dUy, dUz, oX, oY, oZ);
            gxD.copyFrom(std::vector<scalar>(nC, 0.0));
            gyD.copyFrom(std::vector<scalar>(nC, 0.0));
            gzD.copyFrom(std::vector<scalar>(nC, 0.0));
            deviceAmiAddGradRot(ami, dUx, oX, dV, gxD, gyD, gzD);
            const std::vector<scalar> UNx = oX.host();
            std::vector<vector> gh(nC, vector{0, 0, 0});
            std::size_t off = 0;
            std::vector<scalar> uown(nC);
            for (label c = 0; c < nC; ++c) uown[c] = U.internal[c].x;
            for (const AMIInterface& a : amis)
            {
                const std::vector<scalar> un(UNx.begin() + off, UNx.begin() + off + a.ownCell.size());
                off += a.ownCell.size();
                cpu::cyclicAMI::addGrad(a, uown, un, g.V(), gh);
            }
            std::vector<scalar> hx(nC);
            for (label c = 0; c < nC; ++c) hx[c] = gh[c].x;
            rc |= report("addGrad(U.x).x", hx, gxD.host(), 1e-12);
        }

        // STAGE 11: the non-orthogonal correction, per component.
        for (int comp = 0; comp < 3; ++comp)
        {
            DeviceBuffer<scalar> corrD;
            corrD.copyFrom(std::vector<scalar>(nC, 0.0));
            deviceAmiAddLapCorr(ami, comp, dNuEff, gUx, gUy, gUz, corrD);
            std::vector<scalar> corrH(nC, 0.0);
            for (const AMIInterface& a : amis)
                cpu::cyclicAMI::lapCorr(a, comp, nuEff, gradU, corrH);
            rc |= report(comp == 0 ? "lapCorr(U.x)" : comp == 1 ? "lapCorr(U.y)" : "lapCorr(U.z)",
                         corrH, corrD.host(), 1e-12);
        }

        // ---- STAGE 12: the DIV SCHEME's face weight at the interface -----------------------------
        // What the case's div(phi,U) actually asks for. Every stage above agreed while the interface was
        // assembled UPWIND on a case naming `bounded Gauss limitedLinearV 1` -- agreement between brae's
        // two implementations of the same wrong scheme -- so this stage is the one the earlier sixteen
        // could not have caught, and it needs a control that upwind itself would fail.
        {
            // The PACKED gradient the device kernel reads: g[q*nC + c], q = 3i + j = d(U_j)/d(x_i).
            // That is OpenFOAM's packing and the TRANSPOSE of the `gradU` tensor built above for
            // lapCorr, and both are filled from the same gc[l] so neither can drift from the other.
            std::vector<scalar> packed(9 * (std::size_t)nC);
            for (int di = 0; di < 3; ++di)
                for (int cj = 0; cj < 3; ++cj)
                    for (label c = 0; c < nC; ++c)
                        packed[(3*di + cj)*(std::size_t)nC + c] =
                            di == 0 ? gc[cj][c].x : di == 1 ? gc[cj][c].y : gc[cj][c].z;
            DeviceBuffer<scalar> gPack;
            gPack.copyFrom(packed);
            const scalar twoByk = 2.0 / 1.0;         // `limitedLinearV 1`

            DeviceBuffer<scalar> wD;
            deviceAmiLimitedVWeights(ami, dUx, dUy, dUz, gPack, nC, twoByk, wD);
            std::vector<scalar> wH;
            for (const AMIInterface& a : amis)
            {
                std::size_t off = wH.size();
                const std::vector<scalar> ph(phiIf.begin() + off, phiIf.begin() + off + a.ownCell.size());
                const std::vector<scalar> wa =
                    cpu::cyclicAMI::limitedLinearVWeights(a, ph, U.internal, gradU, 1.0);
                wH.insert(wH.end(), wa.begin(), wa.end());
            }
            const std::vector<scalar> wDh = wD.host();
            rc |= report("limitedLinearV weight", wH, wDh, 1e-12);

            // THE CONTROL. A weight that happened to equal pos0(phi) everywhere would pass the line
            // above while changing nothing -- which is exactly the state this stage exists to end. So
            // demand that it actually differs from upwind on a real share of the interface, and that
            // the difference reaches the assembled matrix.
            int nDiff = 0;
            scalar wMax = 0;
            for (std::size_t i = 0; i < wDh.size(); ++i)
            {
                const scalar up = phiIf[i] >= 0.0 ? 1.0 : 0.0;
                const scalar dd = std::fabs(wDh[i] - up);
                if (dd > 1e-10) ++nDiff;
                wMax = std::fmax(wMax, dd);
            }
            std::printf("  limitedLinearV differs from upwind on %d of %zu faces, max |dw| %.4e\n",
                        nDiff, wDh.size(), wMax);
            if (nDiff == 0)
            {
                std::printf("  FAIL: the limited weight equals pos0(phi) on EVERY interface face, so this "
                            "stage cannot tell the new scheme from the upwind one it replaces\n");
                rc = 1;
            }

            DeviceBuffer<scalar> diagL, diagU;
            diagL.copyFrom(std::vector<scalar>(nC, 0.0));
            diagU.copyFrom(std::vector<scalar>(nC, 0.0));
            deviceAmiAssembleMomentum(ami, dNuEff, diagL, &wD);
            const std::vector<scalar> ifLim = ami.ifCoeff.host();
            deviceAmiAssembleMomentum(ami, dNuEff, diagU);
            const std::vector<scalar> ifUpw = ami.ifCoeff.host();
            scalar mx = 0;
            for (std::size_t i = 0; i < ifLim.size(); ++i)
                mx = std::fmax(mx, std::fabs(ifLim[i] - ifUpw[i]));
            std::printf("  ...and moves the interface off-diagonal by up to %.4e\n", mx);
            if (!(mx > 0.0))
            {
                std::printf("  FAIL: the limited weight reached the assembly and changed nothing\n");
                rc = 1;
            }

            // The SCALAR form, on U.x standing in for a turbulence transport: same limiter, NVDTVD's r,
            // and the value is not rotated across the interface while its gradient is.
            {
                std::vector<scalar> fx(nC);
                for (label c = 0; c < nC; ++c) fx[c] = U.internal[c].x;
                DeviceBuffer<scalar> fD, sxD, syD, szD, wsD;
                std::vector<scalar> sx(nC), sy(nC), sz(nC);
                for (label c = 0; c < nC; ++c) { sx[c] = gc[0][c].x; sy[c] = gc[0][c].y; sz[c] = gc[0][c].z; }
                fD.copyFrom(fx); sxD.copyFrom(sx); syD.copyFrom(sy); szD.copyFrom(sz);
                deviceAmiLimitedWeights(ami, fD, sxD, syD, szD, twoByk, wsD);
                std::vector<scalar> wsH;
                for (const AMIInterface& a : amis)
                {
                    std::size_t off = wsH.size();
                    const std::vector<scalar> ph(phiIf.begin() + off, phiIf.begin() + off + a.ownCell.size());
                    const std::vector<scalar> wa =
                        cpu::cyclicAMI::limitedLinearWeights(a, ph, fx, gc[0], 1.0);
                    wsH.insert(wsH.end(), wa.begin(), wa.end());
                }
                rc |= report("limitedLinear weight (scalar)", wsH, wsD.host(), 1e-12);
            }
        }
    }

    std::printf("%s\n", rc == 0 ? "  ok:   every cyclicAMI CUDA stage matches the _cpp reference"
                                : "  FAIL: a cyclicAMI CUDA stage disagrees with the _cpp reference");
    return rc;
}
