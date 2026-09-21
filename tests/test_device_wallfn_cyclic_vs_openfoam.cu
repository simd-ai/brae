// The DEVICE's epsilon at a cell constrained by a WALL FUNCTION that also touches a CYCLIC PAIR,
// against REAL OpenFOAM's own epsilon.
//
// fvMatrix::setValuesFromList (fvMatrix.C) walks EVERY face of a constrained cell and zeroes
// internalCoeffs AND boundaryCoeffs on the patch that owns it -- a cyclic patch like any other. So a
// row that epsilonWallFunction pinned is cut from its periodic neighbour exactly as it is cut from its
// internal ones, and the pinned value is what the cell keeps.
//
// The device's setValues zeroed the internal faces and the NON-COUPLED patches only, because the
// device's boundary arrays carry no coupled face (device_mesh.cuh:41-44). The pair's off-diagonal
// survived into the pinned row and deviceAmul went on adding ifCoeff*psi[nbr] to it.
//
// THE ORACLE IS OpenFOAM'S OWN epsilon after one step, with every solve in both codes pinned at 1e-16
// so that neither arm is reading a Krylov stopping point. THE REFERENCE is brae's own host arm on the
// same step: the question this gate asks is whether the device is as close to OpenFOAM at these cells
// as the host is, and the host is gated against OpenFOAM in tests/interfoam_baffle_vs_openfoam.sh.
//
// THE CONTROL is the pair's OTHER cells -- the ones that touch the pair but no wall, and are therefore
// not constrained. They must agree in both arms whatever the state of this fix, which is what makes the
// assertion specific to the constrained rows rather than to the case running at all.
//
// MEASURED with the defect in place: epsilon at the two cells touching both the baffle pair and
// lowerWall read 2.0234 where OpenFOAM and the host read 1.9862 -- and the device gave those two cells
// two DIFFERENT values where one wall function gives one -- while every pair-only cell agreed to 1e-12.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cyclic_interface.cuh"
#include "foam_field_reader.cuh"
#include "inter_driver_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

using namespace brae;
using namespace brae::cpu::interFoam;

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    std::printf(ok ? "  ok:   %s\n" : "  FAIL: %s\n", what);
    if (!ok) ++failures;
}

// the worst |a - of| over a set of cells, and the scale it is measured against
std::pair<scalar, scalar> worstOver(
    const std::set<label>&     cells,
    const std::vector<scalar>& a,
    const std::vector<scalar>& of)
{
    scalar w = 0;
    scalar sc = 0;
    for (label c : cells)
    {
        const std::size_t i = static_cast<std::size_t>(c);
        if (i >= a.size() || i >= of.size()) continue;
        w = std::fmax(w, std::fabs(a[i] - of[i]));
        sc = std::fmax(sc, std::fabs(of[i]));
    }
    return std::pair<scalar, scalar>(w, sc);
}
}   // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("  SKIP: usage: %s <caseDir> <ofTimeDir>\n", argv[0]);
        return 77;
    }
    const std::string caseDir = argv[1];
    const std::string timeDir = argv[2];
    std::printf("== the device's wall-function epsilon on a cyclic pair against OpenFOAM's ==\n");

    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }
    if (nDev <= 0) { std::printf("  SKIP: no CUDA device\n"); return 77; }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    attachCyclicCoupling(fvp, m, g);
    const label nC = m.nCells();

    // the two cell sets this gate turns on: the pair's cells that a wall also constrains, and the
    // pair's cells that no wall touches
    std::set<label> pairCells;
    std::set<label> wallCells;
    for (const FvPatch& q : fvp)
    {
        if (q.coupled)
        {
            for (label i = 0; i < q.size; ++i) pairCells.insert(q.faceCells[i]);
        }
        else if (q.type == "wall")
        {
            for (label i = 0; i < q.size; ++i) wallCells.insert(q.faceCells[i]);
        }
    }
    std::set<label> pairAndWall;
    std::set<label> pairOnly;
    for (label c : pairCells)
    {
        if (wallCells.count(c)) pairAndWall.insert(c);
        else                    pairOnly.insert(c);
    }
    std::printf("  %d cells; %zu on the pair, of which %zu are ALSO wall-adjacent\n",
                (int)nC, pairCells.size(), pairAndWall.size());
    check("createBaffles left a coupled pair in the mesh", !pairCells.empty());
    check("...and the baffle stands on a wall, so a constrained cell touches the pair",
          !pairAndWall.empty());
    check("...and the pair has unconstrained cells too, which are this gate's control",
          !pairOnly.empty());
    if (pairAndWall.empty() || pairOnly.empty())
    {
        std::printf("test_device_wallfn_cyclic_vs_openfoam: %d failures\n", failures);
        return 1;
    }

    const FieldData<scalar> ofEps = readField<scalar>(timeDir + "/epsilon");
    const FieldData<vector> ofU = readField<vector>(timeDir + "/U");
    check("OpenFOAM's epsilon has one value per cell",
          ofEps.internalField.size() == static_cast<std::size_t>(nC));
    if (ofEps.internalField.size() != static_cast<std::size_t>(nC))
    {
        std::printf("test_device_wallfn_cyclic_vs_openfoam: %d failures\n", failures);
        return 1;
    }

    InterFields fh;
    InterFields fd;
    runInterFoam(caseDir, caseDir + "/0", m, g, fvp, 1, false, &fh);
    runInterFoamDevice(caseDir, caseDir + "/0", m, g, fvp, 1, false, &fd);
    check("both arms ran the closure", fh.turbulence.on && fd.turbulence.on);

    const std::vector<scalar>& of = ofEps.internalField;
    const auto hWall = worstOver(pairAndWall, fh.turbulence.epsilon.internal, of);
    const auto dWall = worstOver(pairAndWall, fd.turbulence.epsilon.internal, of);
    const auto hFree = worstOver(pairOnly, fh.turbulence.epsilon.internal, of);
    const auto dFree = worstOver(pairOnly, fd.turbulence.epsilon.internal, of);
    std::printf("  epsilon on the %zu PAIR+WALL cells vs OpenFOAM: host %.4e, device %.4e (of %.4e)\n",
                pairAndWall.size(), (double)hWall.first, (double)dWall.first, (double)hWall.second);
    std::printf("  CONTROL, the %zu pair-only cells:              host %.4e, device %.4e (of %.4e)\n",
                pairOnly.size(), (double)hFree.first, (double)dFree.first, (double)hFree.second);
    for (label c : pairAndWall)
    {
        const std::size_t i = static_cast<std::size_t>(c);
        std::printf("    cell %5d: OpenFOAM %.12e, host %.12e, device %.12e\n",
                    (int)c, (double)of[i], (double)fh.turbulence.epsilon.internal[i],
                    (double)fd.turbulence.epsilon.internal[i]);
    }

    // THE BOUND IS THE DEVICE'S OWN FLOOR ON THE SAME FIELD, not a number picked to fit and not the
    // host's distance -- the host reaches OpenFOAM EXACTLY at these cells (0.0e+00, measured), so a
    // bound written against it would be a bound of zero. The floor is what this arm reaches at the
    // pair's UNCONSTRAINED cells, relative to their own scale: MEASURED 1.1174e-11 of 1.0795e-01,
    // which is 1.0e-10. The constrained cells sit at 1.1e-16 relative, a millionth of that, and with
    // the pair left in the pinned row they read 3.7286e-02 -- 1.9e-02 relative, eight orders above the
    // floor and past any margin this bound could carry.
    const scalar dWallRel = dWall.first/std::fmax(dWall.second, scalar(1e-300));
    const scalar dFreeRel = dFree.first/std::fmax(dFree.second, scalar(1e-300));
    const scalar hWallRel = hWall.first/std::fmax(hWall.second, scalar(1e-300));
    std::printf("  relative: device PAIR+WALL %.4e, device pair-only (the floor) %.4e, host PAIR+WALL %.4e\n",
                (double)dWallRel, (double)dFreeRel, (double)hWallRel);
    check("the device's epsilon at a wall-constrained cell on the pair is at this arm's own floor",
          dWallRel <= scalar(10)*std::fmax(dFreeRel, scalar(1e-15)));
    // ...and that floor is genuinely a floor. Without this the check above would pass on a run whose
    // every pair cell was wrong together.
    check("the floor it is measured against is a floor, not a gap of its own", dFreeRel < scalar(1e-8));
    check("OpenFOAM's epsilon is not zero there, so the comparison means something",
          hWall.second > scalar(0));
    // ...and brae's own host arm, which tests/interfoam_baffle_vs_openfoam.sh gates against OpenFOAM,
    // is at that floor too -- so the oracle, the reference and the arm under test all agree here.
    check("the host arm is at the same floor at those cells",
          hWallRel <= scalar(10)*std::fmax(dFreeRel, scalar(1e-15)));

    // ...AND grad(U) ACROSS THE PAIR, on the same two pinned runs. fvc::grad sums every face of a cell
    // and a cyclic face is a face, but the device's gradient walked the internal faces and the
    // non-coupled patches only, so the momentum assembly's grad(U) was missing the pair's half -- and
    // with it linearUpwind's deferred correction, the non-orthogonal correction and dev2(T(grad U)).
    // It is invisible at the FIRST corrector, where this case starts from rest and grad(U) is zero, so
    // it needs a case with nOuterCorrectors > 1: the tutorial's own 3.
    //
    // THE TUTORIAL GATE CANNOT SEE THIS. tests/interfoam_baffle_vs_openfoam.sh runs 20 steps at the
    // case's own tolerances, where the distance to OpenFOAM is the trajectory's: its device numbers
    // read alpha 4.9021e-05 and U 1.1369e-02 both WITH and WITHOUT the fix, to every digit. That is
    // why this assertion lives here, on a pinned single step, and why those bounds were left alone.
    if (ofU.internalField.size() == static_cast<std::size_t>(nC))
    {
        auto worstU = [&](const std::set<label>& cells, const std::vector<vector>& a)
        {
            scalar w = 0;
            scalar sc = 0;
            for (label c : cells)
            {
                const std::size_t i = static_cast<std::size_t>(c);
                const vector e{a[i].x - ofU.internalField[i].x,
                               a[i].y - ofU.internalField[i].y,
                               a[i].z - ofU.internalField[i].z};
                w = std::fmax(w, mag(e));
                sc = std::fmax(sc, mag(ofU.internalField[i]));
            }
            return std::pair<scalar, scalar>(w, sc);
        };
        const auto hUw = worstU(pairAndWall, fh.U.internal);
        const auto dUw = worstU(pairAndWall, fd.U.internal);
        const auto hUf = worstU(pairOnly, fh.U.internal);
        const auto dUf = worstU(pairOnly, fd.U.internal);
        std::printf("  U on the pair's cells vs OpenFOAM: host %.4e / %.4e, device %.4e / %.4e "
                    "(wall-touching / not, |U| up to %.4e)\n",
                    (double)hUw.first, (double)hUf.first, (double)dUw.first, (double)dUf.first,
                    (double)std::fmax(hUw.second, hUf.second));
        // THE BOUND IS NOT THE HOST'S LEVEL, AND THIS SAYS SO. The host reaches OpenFOAM at 3.9e-17
        // here; the device does not, and pretending otherwise would be a bound that cannot hold. With
        // the pair summed into grad(U) the device reads 5.6e-10 at the wall-touching cells and 4.2e-10
        // at the rest, of a 9.3e-04 |U|; with the pair taken back out of the gradient, 3.4071e-09 and
        // 2.7336e-09 -- so 2e-09 separates the two and fails the moment the pair leaves it again.
        //
        // WHAT KEEPS THE DEVICE OFF THE HOST'S LEVEL is the OTHER half of the same term:
        // deviceCyclicAddLinUpwindCorr -- linearUpwind's deferred correction across a pair, which this
        // case asks for by name (`div(rhoPhi,U) Gauss linearUpwind grad(U)`) -- has no caller in the
        // tree. The residual sits ON the pair's cells, which is its signature. Tighten this bound when
        // that is wired; it is an open finding, not a tolerance chosen to fit.
        check("the device's U at the pair's cells carries the pair's gradient",
              dUw.first < scalar(2e-09) && dUf.first < scalar(2e-09));
        check("OpenFOAM's U is not zero at those cells, so the comparison means something",
              std::fmax(hUw.second, hUf.second) > scalar(0));
    }
    else
    {
        check("OpenFOAM wrote U for this step", false);
    }

    std::printf("test_device_wallfn_cyclic_vs_openfoam: %d failures\n", failures);
    return failures ? 1 : 0;
}
