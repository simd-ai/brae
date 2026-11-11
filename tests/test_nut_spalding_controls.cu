// nutUSpaldingWallFunction's own Newton controls, `tolerance` and `maxIter` (brief 2026-11-11 #18, the execution audit).
//
// OpenFOAM v2412 reads both per patch, defaulting to 0.01 and 10 (nutUSpaldingWallFunctionFvPatchScalarField.C:256-257), and runs
// calcUTau's Newton under them (:150-180); when the tolerance is not 0.01, calcNut KEEPS a face's present nut wherever one step
// from it moves uTau by less than the tolerance (:69-90). brae read neither entry and stopped every Newton at 0.01 and 10: the
// official occDrivAerStaticMesh sets `tolerance 1e-9` on every car wall. The oracle for the law is the law itself: uTau solves
// Spalding's equation, so at OpenFOAM's 1e-9 the equation's residual is at rounding, and at 0.01 it is not.
//
// LEG 1  the reader and the factory: a nut file with two Spalding walls, one with `tolerance 1e-9; maxIter 25;`, one without,
//        becomes two patch objects carrying (1e-9, 25) and OpenFOAM's (0.01, 10). CONTROL: the entries' values must reach the
//        patch, so a patch built from an entry with the defaults put back must not equal the first.
// LEG 2  the law: from a cold seed, tolerance 1e-9 leaves Spalding's equation at rounding of its terms; CONTROL: tolerance 0.01
//        (brae's old stop) leaves it measurably unsolved on the same face.
// LEG 3  the keep branch: a seed one step from which moves uTau by less than the tolerance (a custom 1e-3, not 0.01) is returned
//        bit for bit; CONTROL: the same seed under tolerance 0.01 (no keep branch, OpenFOAM's own rule) comes back moved.
#include "foam_field_reader.cuh"
#include "fv_patch_field.cuh"
#include "nut_wall_function.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace
{

int failures = 0;

void check(bool ok, const char* what, double a, double b)
{
    std::printf("%s %s (%.17g, %.17g)\n", ok ? "  ok  " : "  FAIL", what, a, b);
    if (!ok) ++failures;
}

// Spalding's equation at uTau, relative to its largest term: y+ = u+ + (1/E)(exp(kappa u+) - 1 - kappa u+ - (kappa u+)^2/2
// - (kappa u+)^3/6), u+ = magUp/uTau, y+ = y uTau/nu
double spaldingResidual(double magUp, double y, double nu, double kappa, double E, double uTau)
{
    const double up = magUp/uTau;
    const double yp = y*uTau/nu;
    const double k = kappa*up;
    const double rhs = up + (std::exp(k) - 1 - k - 0.5*k*k - k*k*k/6)/E;
    return std::fabs(yp - rhs)/std::fmax(std::fabs(yp), std::fabs(rhs));
}

double uTauOf(double nut, double nu, double magGradU)
{
    return std::sqrt((nut + nu)*magGradU);
}

} // namespace

int main()
{
    const char* tmp = std::getenv("TMPDIR");
    const std::string dir = std::string(tmp ? tmp : "/tmp") + "/brae_spalding_controls";
    std::system(("rm -rf " + dir + " && mkdir -p " + dir).c_str());

    // ---- LEG 1: the reader and the factory ----
    {
        std::ofstream o(dir + "/nut");
        o << "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n    object      nut;\n}\n"
          << "dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform 7.5e-05;\n\nboundaryField\n{\n"
          << "    body\n    {\n        type            nutUSpaldingWallFunction;\n        value           $internalField;\n"
          << "        tolerance       1e-9;\n        maxIter         25;\n    }\n"
          << "    wheel\n    {\n        type            nutUSpaldingWallFunction;\n        value           $internalField;\n    }\n"
          << "}\n";
        o.close();
        const FieldData<scalar> fd = readField<scalar>(dir + "/nut");
        const PatchFieldData<scalar>* body = nullptr;
        const PatchFieldData<scalar>* wheel = nullptr;
        for (const auto& b : fd.boundary)
        {
            if (b.name == "body") body = &b;
            if (b.name == "wheel") wheel = &b;
        }
        check(body && wheel, "LEG 1 both patches read", body ? 1 : 0, wheel ? 1 : 0);
        if (body && wheel)
        {
            check(body->nutTolerance == 1e-9 && body->nutMaxIter == 25, "LEG 1 body: the file's tolerance and maxIter",
                  body->nutTolerance, body->nutMaxIter);
            check(wheel->nutTolerance == 0.01 && wheel->nutMaxIter == 10, "LEG 1 wheel: OpenFOAM's defaults", wheel->nutTolerance,
                  wheel->nutMaxIter);
            FvPatch p;
            p.name = "body";
            p.type = "wall";
            p.size = 1;
            p.faceCells = {0};
            p.deltaCoeffs = {1.0};
            const auto pf = makePatchField<scalar>(p, *body);
            check(pf->isNutUSpalding() && pf->nutUTolerance() == 1e-9 && pf->nutUMaxIter() == 25,
                  "LEG 1 the patch object carries the file's controls", pf->nutUTolerance(), pf->nutUMaxIter());
            PatchFieldData<scalar> reset = *body;
            reset.nutTolerance = 0.01;
            reset.nutMaxIter = 10;
            const auto pc = makePatchField<scalar>(p, reset);
            check(pc->nutUTolerance() != pf->nutUTolerance(), "LEG 1 CONTROL: the defaults put back give another patch",
                  pc->nutUTolerance(), pf->nutUTolerance());
        }
    }

    // a wall face of the official case's kind: 38.9 m/s outside, a first cell at y = 2e-4 m, nu of air
    const double nu = 1.507e-5;
    const double kappa = 0.41;
    const double E = 9.8;
    const double magUp = 12.0;
    const double y = 2e-4;
    const double magGradU = magUp*(1.0/(2.0*y));

    // ---- LEG 2: the law, cold seed ----
    {
        const double nutTight = spaldingNutValue(magUp, magGradU, y, nu, kappa, E, 0.0, 1e-9, 10);
        const double nutLoose = spaldingNutValue(magUp, magGradU, y, nu, kappa, E, 0.0, 0.01, 10);
        const double rTight = spaldingResidual(magUp, y, nu, kappa, E, uTauOf(nutTight, nu, magGradU));
        const double rLoose = spaldingResidual(magUp, y, nu, kappa, E, uTauOf(nutLoose, nu, magGradU));
        check(rTight < 1e-12, "LEG 2 tolerance 1e-9: Spalding's equation at rounding (relative residual)", rTight, nutTight);
        check(rLoose > 1e-9, "LEG 2 CONTROL tolerance 0.01: the equation left unsolved", rLoose, nutLoose);
    }

    // ---- LEG 3: the keep branch ----
    {
        const double converged = spaldingNutValue(magUp, magGradU, y, nu, kappa, E, 0.0, 1e-12, 50);
        const double seed = converged*(1.0 + 1e-5);
        const double kept = spaldingNutValue(magUp, magGradU, y, nu, kappa, E, seed, 1e-3, 10);
        check(kept == seed, "LEG 3 tolerance 1e-3: one step moves uTau by less, so the seed is kept bit for bit", kept, seed);
        const double moved = spaldingNutValue(magUp, magGradU, y, nu, kappa, E, seed, 0.01, 10);
        check(moved != seed && std::fabs(moved - converged) < std::fabs(seed - converged),
              "LEG 3 CONTROL tolerance 0.01: no keep branch, the seed moves towards the root", moved, seed);
    }

    std::system(("rm -rf " + dir).c_str());
    std::printf("%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
