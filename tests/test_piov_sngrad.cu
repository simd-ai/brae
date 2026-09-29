// pressureInletOutletVelocity's snGrad() is directionMixed's, built from the valueFraction and the cell.
//
// OpenFOAM (directionMixedFvPatchField.C, snGrad):
//     (transform(vf, refValue) + transform(I - vf, pif + refGrad/deltaCoeffs) - pif)*deltaCoeffs
// with this condition's refValue = refGrad = 0 and vf = neg(phi)*(I - nn), Zero until the first
// updateCoeffs (pressureInletOutletVelocityFvPatchVectorField.C:47-49, :180). That is
//     -neg(phi)*(pif - n*(n & pif))*deltaCoeffs
// and the STORED VALUE IS NOT IN IT. brae's class returned (value - pif)*deltaCoeffs, which is the same
// number once the value has been refreshed from the cell and a different one before: at construction
// interFoam's RAS/waterChannel holds the file's (0 0 0) on its atmosphere over cells moving at (1 0 0),
// kOmegaSST's correctNut takes grad(U)'s boundary value from this snGrad, and the patch's nut came out
// 3.7e-05 against OpenFOAM's 3.33e-02 (tests/interfoam_waterchannel_vs_openfoam.sh has the run).
//
//   LEG 1  before any flux is known, and on an outflow or zero-flux face: snGrad is exactly zero
//          whatever value is stored
//   LEG 2  on an inflow face: minus the TANGENTIAL part of the cell velocity, times deltaCoeffs
//   LEG 3  THE CONTROL: the old form, (stored value - cell)*deltaCoeffs, is NOT zero on leg 1's state,
//          so a class that still returned it would fail leg 1
//   LEG 4  once the value has been refreshed (updateFromPatchVelocity), the two forms agree -- which is
//          why every gate that reads this patch after its first evaluation never saw the difference
//
// WITH A tangentialVelocity (interFoam claims the entry and calls setTangentialVelocity), on a TILTED
// normal n = (0.6 0.8 0) with tv = (3 5 1), against a line-by-line transcription of OpenFOAM:
//   LEG 5  refValue = tv - n*(n & tv) (pressureInletOutletVelocityFvPatchVectorField.C:130-136)
//   LEG 6  the inflow value is directionMixed's evaluate, (vf & refValue) + ((I - vf) & pif), with
//          vf = neg(phi)*(I - sqr(n)) (.C:180, directionMixedFvPatchField.C:157-175); outflow is the cell
//   LEG 7  snGrad is ((vf & refValue) + ((I - vf) & pif) - pif)*deltaCoeffs (directionMixedFvPatchField.C:139-153)
//   LEG 8  the normals moved after construction leave refValue where it was, and the next update's value
//          takes the new valueFraction with the old refValue: OpenFOAM never reprojects
//   LEG 9  the patch reports the entry (the device builder refuses on it) and refuses a topology change
//   LEG 10 the reader: uniform, nonuniform and $internalField read; the bare form flagged for the claim to
//          refuse; another type's `tangentialVelocity` not parsed
// Controls that must differ: the projection skipped (only in the last bit: vf = I - nn discards the
// normal part itself, so no case can see the projection beyond rounding), the refValue recomputed from
// the moved normals, and the refValue term dropped from snGrad. DTCHullMoving's atmosphere is axis-aligned and can witness
// none of the projection legs, so these are their only witness.
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace brae;

namespace {
int failures = 0;

void check(
    const char* what,
    bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok)
    {
        ++failures;
    }
}

bool same(
    const vector& a,
    const vector& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

scalar magOf(const vector& v)
{
    return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}
}   // namespace

int main()
{
    std::printf("== pressureInletOutletVelocity::snGrad ==\n");
    // three faces on a top patch, normal +z, 0.25 from their cells
    FvPatch p;
    p.name = "atmosphere";
    p.type = "patch";
    p.size = 3;
    p.faceCells = {0, 1, 2};
    p.deltaCoeffs = {4.0, 4.0, 4.0};
    p.nf = {vector{0, 0, 1}, vector{0, 0, 1}, vector{0, 0, 1}};
    p.magSf = {1.0, 1.0, 1.0};
    p.Cf = {vector{0, 0, 1}, vector{1, 0, 1}, vector{2, 0, 1}};

    const std::vector<vector> cells = {vector{1, 0, 0}, vector{1, 0.5, -0.2}, vector{2, 0, 0.3}};
    // the file's value, as waterChannel ships it
    PressureInletOutletVelocityPatchField<vector> f(p, true, vector{0, 0, 0}, {});

    // LEG 1: no flux yet
    {
        const std::vector<vector> sn = f.snGrad(cells);
        scalar worst = 0;
        for (const vector& v : sn)
        {
            worst = std::fmax(worst, magOf(v));
        }
        check("LEG 1  before the first updateCoeffs the valueFraction is Zero and snGrad is exactly 0",
              sn.size() == 3 && worst == scalar(0));
    }
    // outflow, zero flux, inflow
    f.updateFromFlux({0.7, 0.0, -0.4});
    const std::vector<vector> sn = f.snGrad(cells);
    check("LEG 1  ...and on an outflow face (phi > 0), whatever value is stored", magOf(sn[0]) == scalar(0));
    check("LEG 1  ...and at phi = 0: neg(0) is 0", magOf(sn[1]) == scalar(0));
    // LEG 2: inflow, cell (2 0 0.3), normal z: tangential part (2 0 0), snGrad -(2 0 0)*4
    check("LEG 2  on an inflow face snGrad is -(the cell's tangential velocity)*deltaCoeffs",
          std::fabs(sn[2].x + 8.0) < 1e-15 && std::fabs(sn[2].y) < 1e-15 && std::fabs(sn[2].z) < 1e-15);

    // LEG 3: the old form on the same state
    {
        const std::vector<vector>& val = f.value();
        const vector old0{(val[0].x - cells[0].x)*4.0, (val[0].y - cells[0].y)*4.0, (val[0].z - cells[0].z)*4.0};
        std::printf("  the old form on face 0: |(value - cell)*deltaCoeffs| = %.3f\n", (double)magOf(old0));
        check("LEG 3  CONTROL: (stored value - cell)*deltaCoeffs is NOT zero there, so leg 1 can fail",
              magOf(old0) > scalar(1));
    }
    // LEG 4: refreshed, the two forms agree
    {
        f.updateFromPatchVelocity({}, cells, {});
        const std::vector<vector>& val = f.value();
        const std::vector<vector> snNew = f.snGrad(cells);
        scalar worst = 0;
        for (std::size_t i = 0; i < 3; ++i)
        {
            const vector old{(val[i].x - cells[i].x)*4.0, (val[i].y - cells[i].y)*4.0, (val[i].z - cells[i].z)*4.0};
            worst = std::fmax(worst, magOf(vector{old.x - snNew[i].x, old.y - snNew[i].y, old.z - snNew[i].z}));
        }
        check("LEG 4  once the value is refreshed from the cell the two forms agree", worst < 1e-14);
    }

    // ---- tangentialVelocity ----------------------------------------------------------------
    {
        FvPatch q;
        q.name = "tilted";
        q.type = "patch";
        q.size = 2;
        q.faceCells = {0, 1};
        q.deltaCoeffs = {4.0, 2.5};
        q.nf = {vector{0.6, 0.8, 0}, vector{0.6, 0.8, 0}};
        q.magSf = {1.0, 1.0};
        q.Cf = {vector{0, 0, 0}, vector{1, 0, 0}};
        const vector tvIn{3, 5, 1};
        const std::vector<vector> qc = {vector{1.5, -0.25, 0.75}, vector{-2, 1, 0.5}};

        // the transcription, OpenFOAM's operations in OpenFOAM's order
        auto ofRef = [](const vector& n, const vector& t)
        {
            const scalar s = n.x*t.x + n.y*t.y + n.z*t.z;
            return vector{t.x - n.x*s, t.y - n.y*s, t.z - n.z*s};
        };
        auto ofEval = [](const vector& n, const vector& rv, const vector& pif)
        {
            const scalar one = 1;
            const scalar vxx = one*(one - n.x*n.x), vxy = one*(-(n.x*n.y)), vxz = one*(-(n.x*n.z));
            const scalar vyy = one*(one - n.y*n.y), vyz = one*(-(n.y*n.z)), vzz = one*(one - n.z*n.z);
            const vector a{vxx*rv.x + vxy*rv.y + vxz*rv.z, vxy*rv.x + vyy*rv.y + vyz*rv.z,
                           vxz*rv.x + vyz*rv.y + vzz*rv.z};
            const scalar ixx = one - vxx, ixy = -vxy, ixz = -vxz, iyy = one - vyy, iyz = -vyz, izz = one - vzz;
            const vector b{ixx*pif.x + ixy*pif.y + ixz*pif.z, ixy*pif.x + iyy*pif.y + iyz*pif.z,
                           ixz*pif.x + iyz*pif.y + izz*pif.z};
            return vector{a.x + b.x, a.y + b.y, a.z + b.z};
        };
        auto dist = [](const vector& a, const vector& b)
        {
            return std::sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z));
        };

        PressureInletOutletVelocityPatchField<vector> t(q, true, vector{0, 0, 0}, {});
        check("LEG 9  a patch without the entry reports none and maps through a topology change",
              t.tangentialVelocityPtr() == nullptr && t.autoMapComplete());
        t.setTangentialVelocity({tvIn, tvIn});
        const vector rv = ofRef(q.nf[0], tvIn);
        std::printf("  refValue (%.17g %.17g %.17g), n & refValue %.3e\n",
                    (double)rv.x, (double)rv.y, (double)rv.z,
                    (double)(q.nf[0].x*rv.x + q.nf[0].y*rv.y + q.nf[0].z*rv.z));
        check("LEG 5  refValue is tv - n*(n & tv), bit for bit",
              t.tangentialRefValue().size() == 2 && same(t.tangentialRefValue()[0], rv)
           && same(t.tangentialRefValue()[1], rv));
        check("LEG 9  ...and with it the patch reports the entry and refuses a topology change",
              t.tangentialVelocityPtr() != nullptr && !t.autoMapComplete());

        // face 0 inflow, face 1 outflow
        t.updateFromFlux({-0.3, 0.2});
        t.updateFromPatchVelocity({}, qc, {});
        const vector v0 = ofEval(q.nf[0], rv, qc[0]);
        check("LEG 6  the inflow value is (vf & refValue) + ((I - vf) & pif), bit for bit",
              same(t.value()[0], v0));
        check("LEG 6  ...and an outflow face is the cell's, the refValue unread", same(t.value()[1], qc[1]));
        const std::vector<vector> snT = t.snGrad(qc);
        const vector sn0{(v0.x - qc[0].x)*4.0, (v0.y - qc[0].y)*4.0, (v0.z - qc[0].z)*4.0};
        check("LEG 7  snGrad is (evaluate - pif)*deltaCoeffs on inflow, bit for bit", same(snT[0], sn0));
        check("LEG 7  ...and 0 on outflow", same(snT[1], vector{0, 0, 0}));

        // CONTROL: the projection skipped (tv itself as the refValue). vf = I - nn annihilates the normal
        // part of whatever it is handed, so the projection is INVISIBLE in the value except for rounding:
        // MEASURED 1.6e-16 here. It is witnessed only by LEG 6's bit-for-bit comparison, and this control
        // asserts exactly that -- that the skipped form is a different double, not a different number.
        {
            const vector noProj = ofEval(q.nf[0], tvIn, qc[0]);
            std::printf("  CONTROL: the projection skipped moves the inflow value by %.4e (rounding only)\n",
                        (double)dist(noProj, v0));
            check("CONTROL: the projection skipped is not LEG 6's double", !same(noProj, v0));
        }
        // CONTROL: the refValue term dropped from snGrad (the refValue-free form)
        {
            const vector& n = q.nf[0];
            const scalar nd = n.x*qc[0].x + n.y*qc[0].y + n.z*qc[0].z;
            const vector bare{-(qc[0].x - nd*n.x)*4.0, -(qc[0].y - nd*n.y)*4.0, -(qc[0].z - nd*n.z)*4.0};
            std::printf("  CONTROL: snGrad without the refValue term moves by %.4e\n", (double)dist(bare, sn0));
            check("CONTROL: the refValue is in snGrad", dist(bare, sn0) > 1e-3);
        }
        // LEG 8: the normals move; the refValue is the construction-time one. OpenFOAM projects only in
        // setTangentialVelocity (.C:130-136) and never in updateCoeffs (.C:172-187), whose valueFraction does
        // take the NEW normals -- so the value after an update is (vf(new n) & refValue(old n)) + ...
        q.nf = {vector{0, 0.6, 0.8}, vector{0, 0.6, 0.8}};
        t.updateFromFlux({-0.3, 0.2});
        t.updateFromPatchVelocity({}, qc, {});
        const vector keep = ofEval(q.nf[0], rv, qc[0]);
        const vector reproj = ofEval(q.nf[0], ofRef(q.nf[0], tvIn), qc[0]);
        check("LEG 8  after the normals move, the refValue is the construction-time one",
              same(t.tangentialRefValue()[0], rv));
        check("LEG 8  ...and the updated inflow value is vf(new n) & refValue(old n), bit for bit",
              same(t.value()[0], keep));
        std::printf("  CONTROL: a refValue re-projected on the moved normals moves the value by %.4e\n",
                    (double)dist(reproj, keep));
        check("CONTROL: re-projecting at the update is a different value", dist(reproj, keep) > 1e-3);
    }

    // LEG 10: the reader
    {
        namespace fs = std::filesystem;
        const fs::path dir = fs::temp_directory_path() / ("brae_piov_tv_" + std::to_string((long)::getpid()));
        fs::create_directories(dir);
        auto writeU = [&](const std::string& name, const std::string& entry)
        {
            std::ofstream o(dir / name);
            o << "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
                 "dimensions [0 1 -1 0 0 0 0];\n"
                 "internalField uniform (-1.668 0 0);\n"
                 "boundaryField\n{\n    atmosphere\n    {\n        type pressureInletOutletVelocity;\n"
                 "        tangentialVelocity " << entry << ";\n        value uniform (0 0 0);\n    }\n}\n";
            return (dir / name).string();
        };
        const FieldData<vector> u1 = readField<vector>(writeU("U1", "uniform (1 2 3)"));
        check("LEG 10 `uniform` reads",
              u1.boundary.size() == 1 && u1.boundary[0].hasTangentialVelocity && u1.boundary[0].tvUniform
           && same(u1.boundary[0].tvUniformValue, vector{1, 2, 3}));
        const FieldData<vector> u2 = readField<vector>(writeU("U2", "nonuniform List<vector> 2((1 0 0) (0 2 0))"));
        check("LEG 10 `nonuniform` reads",
              u2.boundary.size() == 1 && !u2.boundary[0].tvUniform && u2.boundary[0].tvValues.size() == 2
           && same(u2.boundary[0].tvValues[1], vector{0, 2, 0}));
        const FieldData<vector> u3 = readField<vector>(writeU("U3", "$internalField"));
        check("LEG 10 `$internalField` reads the internal field (DTCHullMoving's form)",
              u3.boundary.size() == 1 && u3.boundary[0].tvUniform
           && same(u3.boundary[0].tvUniformValue, vector{-1.668, 0, 0}));
        // the bare form: Field.C stops on it only on a patch with faces, which the reader cannot see, so it
        // is flagged here and refused at interFoam's claim (interfoam_refusals `piov_tangential_bare`)
        const FieldData<vector> u4 = readField<vector>(writeU("U4", "(1 2 3)"));
        check("LEG 10 the bare form is flagged, not parsed, for the claim to refuse",
              u4.boundary.size() == 1 && u4.boundary[0].tvBare && u4.boundary[0].tvValues.empty());
        // ...and another type naming the key (swirlInletVelocity: a Function1<scalar>) is not parsed as a
        // vectorField, so the factory's refusal of the type is what the case meets
        {
            std::ofstream o(dir / "U5");
            o << "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
                 "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
                 "boundaryField\n{\n    inlet\n    {\n        type swirlInletVelocity;\n"
                 "        tangentialVelocity constant 100;\n        value uniform (0 0 0);\n    }\n}\n";
        }
        const FieldData<vector> u5 = readField<vector>((dir / "U5").string());
        check("LEG 10 swirlInletVelocity's Function1 `tangentialVelocity` reads without a parse error",
              u5.boundary.size() == 1 && u5.boundary[0].tvUnparsed && u5.boundary[0].type == "swirlInletVelocity");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    std::printf("test_piov_sngrad: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
