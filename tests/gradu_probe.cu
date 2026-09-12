// Does the device gaussGrad agree with the host one, at an IDENTICAL velocity field?
//
// The SpalartAllmaras CUDA port converges elsewhere than its _cpp reference, and a cell-level term dump
// traced that to Omega/Stilda -- median per-cell 1.1e-03 but max 8.4. Omega is sqrt(2)*mag(skew(gradU)),
// so either the device gradient differs or the two paths had simply drifted apart in U beforehand. That
// dump could not separate the two: each path had run a full SIMPLE iteration first, so its U was its own.
//
// This runs NO iteration. It reads ONE velocity field, brings both paths to the state a running
// iteration sees (the freestream valueFraction rebuilt from the flow angle, then re-evaluated), and
// compares the boundary values, the gradient, and Omega. It also scores both against the `value` entry
// OpenFOAM itself wrote, which is the only arbiter of which one is right.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

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

    const FieldData<vector> rawU = readField<vector>(caseDir + "/" + t + "/U");
    GeometricField<vector> U = buildField<vector>(rawU, fvp, nC);
    U.evaluateBoundary();
    GeometricField<scalar> P =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    P.evaluateBoundary();

    // The boundary flux, for the device's valueFraction update.
    std::vector<scalar> phiFlat;
    {
        const FieldData<scalar> pf = readField<scalar>(caseDir + "/" + t + "/phi");
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            std::vector<scalar> b(fvp[pi].size, 0.0);
            for (const auto& q : pf.boundary)
            {
                if (q.name == fvp[pi].name && q.hasValue
                    && static_cast<label>(q.values.size()) == fvp[pi].size) b = q.values;
            }
            phiFlat.insert(phiFlat.end(), b.begin(), b.end());
        }
    }

    // ---- host: update the valueFraction, then re-evaluate, as an iteration does ----
    {
        std::vector<std::vector<vector>> Ub(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) Ub[pi] = U.boundary[pi]->value();
        updateMixedFreestream(U.boundary, Ub, fvp);
        updateMixedFreestream(P.boundary, Ub, fvp);
        U.evaluateBoundary();
        P.evaluateBoundary();
    }

    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c)
    {
        ux[c] = U.internal[c].x;
        uy[c] = U.internal[c].y;
        uz[c] = U.internal[c].z;
    }

    // ---- device: the same, from the same field ----
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    DeviceBoundary dbP = buildDeviceBoundary(P, fvp, g);
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dPhiBnd(phiFlat);
    deviceUpdateMixedFreestream(dbU, dbP, dPhiBnd, dUx, dUy, dUz, nullptr);

    std::printf("gradu_probe: %s/%s   %d cells\n", caseDir.c_str(), t.c_str(), static_cast<int>(nC));

    // ---- boundary values: host vs device, and each vs what OpenFOAM wrote ----
    {
        DeviceBuffer<scalar> bx, by, bz;
        deviceBCValue(dbU.comp[0], dUx, bx);
        deviceBCValue(dbU.comp[1], dUy, by);
        deviceBCValue(dbU.comp[2], dUz, bz);
        const std::vector<scalar> hbx = bx.host(), hby = by.host(), hbz = bz.host();
        std::size_t j = 0;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            if (isCoupledInterfaceType(fvp[pi].type)) continue;
            const PatchFieldData<vector>* pd = nullptr;
            for (const auto& q : rawU.boundary)
            {
                if (q.name == fvp[pi].name) pd = &q;
            }
            const std::vector<vector>& hv = U.boundary[pi]->value();
            scalar diff = 0, hErr = 0, dErr = 0;
            for (label i = 0; i < fvp[pi].size; ++i, ++j)
            {
                if (j >= hbx.size()) break;
                const vector dv{hbx[j], hby[j], hbz[j]};
                diff = std::fmax(diff, mag(hv[i] - dv));
                if (pd && pd->hasValue)
                {
                    const vector ofv = pd->valueUniform
                        ? pd->uniformValue
                        : (static_cast<std::size_t>(i) < pd->values.size() ? pd->values[i] : vector{});
                    hErr = std::fmax(hErr, mag(hv[i] - ofv));
                    dErr = std::fmax(dErr, mag(dv - ofv));
                }
            }
            std::printf("  U_b %-14s (%-7s) host-vs-device %.4e | vs OF: host %.4e  device %.4e\n",
                        fvp[pi].name.c_str(), fvp[pi].type.c_str(), diff, hErr, dErr);
        }
    }

    // ---- the gradient ----
    const std::vector<tensor> gh = fvc::gaussGrad(U, m, g, fvp);
    DeviceBuffer<scalar> dGradU;
    deviceGradU(dm, dbU, dUx, dUy, dUz, dGradU);
    const std::vector<scalar> gd = dGradU.host();

    const char* nm[9] = {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"};
    for (int k = 0; k < 9; ++k)
    {
        scalar num = 0, den = 0, mx = 0;
        for (label c = 0; c < nC; ++c)
        {
            const scalar hv = (&gh[c].xx)[k];
            const scalar dv = gd[static_cast<std::size_t>(k) * nC + c];
            num += (hv - dv) * (hv - dv);
            den += hv * hv;
            mx = std::fmax(mx, std::fabs(hv - dv));
        }
        std::printf("  grad%s  L2rel %.4e   max|diff| %.4e\n", nm[k],
                    den > 0 ? std::sqrt(num / den) : std::sqrt(num), mx);
    }

    scalar onum = 0, oden = 0, omaxRel = 0;
    label worst = -1;
    for (label c = 0; c < nC; ++c)
    {
        const tensor& T = gh[c];
        const scalar Oh = std::sqrt((T.xy - T.yx) * (T.xy - T.yx) + (T.xz - T.zx) * (T.xz - T.zx)
                                  + (T.yz - T.zy) * (T.yz - T.zy));
        const scalar ad = gd[1 * nC + c] - gd[3 * nC + c];
        const scalar bd = gd[2 * nC + c] - gd[6 * nC + c];
        const scalar dd = gd[5 * nC + c] - gd[7 * nC + c];
        const scalar Od = std::sqrt(ad * ad + bd * bd + dd * dd);
        onum += (Oh - Od) * (Oh - Od);
        oden += Oh * Oh;
        const scalar rel = (Oh > 0) ? std::fabs(Oh - Od) / Oh : 0.0;
        if (rel > omaxRel)
        {
            omaxRel = rel;
            worst = c;
        }
    }
    std::printf("  Omega   L2rel %.4e   worst per-cell rel %.4e (cell %d)\n",
                oden > 0 ? std::sqrt(onum / oden) : std::sqrt(onum), omaxRel, static_cast<int>(worst));

    // Is the huge RELATIVE difference just Omega being near zero? Bin the per-cell relative difference by
    // |Omega|: if the big ones all sit where Omega ~ 0, the operator is fine and the MODEL is what is
    // ill-conditioned there -- SA divides by Stilda, which is Omega plus a term that also vanishes.
    {
        const scalar edges[5] = {1e-3, 1e-1, 1e+1, 1e+3, 1e+30};
        for (int b = 0; b < 5; ++b)
        {
            const scalar lo = (b == 0) ? 0.0 : edges[b - 1];
            scalar mx = 0;
            label cnt = 0;
            for (label c = 0; c < nC; ++c)
            {
                const tensor& T = gh[c];
                const scalar Oh = std::sqrt((T.xy - T.yx) * (T.xy - T.yx) + (T.xz - T.zx) * (T.xz - T.zx)
                                          + (T.yz - T.zy) * (T.yz - T.zy));
                if (Oh < lo || Oh >= edges[b]) continue;
                const scalar ad = gd[1 * nC + c] - gd[3 * nC + c];
                const scalar bd = gd[2 * nC + c] - gd[6 * nC + c];
                const scalar dd = gd[5 * nC + c] - gd[7 * nC + c];
                const scalar Od = std::sqrt(ad * ad + bd * bd + dd * dd);
                mx = std::fmax(mx, (Oh > 0) ? std::fabs(Oh - Od) / Oh : 0.0);
                ++cnt;
            }
            if (cnt)
                std::printf("    |Omega| in [%8.1e, %8.1e)  %6d cells   worst rel %.3e\n",
                            lo, edges[b], static_cast<int>(cnt), mx);
        }
    }
    return 0;
}
