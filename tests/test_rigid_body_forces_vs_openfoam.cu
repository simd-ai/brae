// brae's fluid force and moment on a body's patches against OpenFOAM's own functionObjects::forces,
// on RAS/floatingObject -- unit 3 of the rigidBodyMotion port.
//
// What rigidBodyMeshMotion::solve does with them is build a spatialVector(momentEff, forceEff) and
// hand it to the body's dynamics, so these two numbers are the whole interface. The gate compares
// them, and the per-face `force` field behind them, against the oracle in tools/dumpBodyForces.
//
// usage: test_rigid_body_forces_vs_openfoam <caseDir> <ofTimeDir> <patch>
#include "foam_field_reader.cuh"
#include "forces.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "primitive_mesh.cuh"
#include "two_phase_mixture_cpp.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok)
{
    std::printf("  %s:   %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

std::vector<vector> readPointsFile(const std::string& path)
{
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    static const std::regex head(R"(\n\s*([0-9]+)\s*\n?\()");
    std::smatch mh;
    std::vector<vector> out;
    if (!std::regex_search(text, mh, head)) return out;
    const std::size_t n = static_cast<std::size_t>(std::atol(mh[1].str().c_str()));
    const char* p = text.c_str() + mh.position(0) + mh.length(0);
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        while (*p && *p != '(') ++p;
        if (!*p) break;
        ++p;
        char* end = nullptr;
        vector v;
        v.x = std::strtod(p, &end);
        p = end;
        v.y = std::strtod(p, &end);
        p = end;
        v.z = std::strtod(p, &end);
        p = end;
        out.push_back(v);
    }
    return out;
}

// the oracle's own line: `# Time  total_x ... pressure_x ... viscous_x ...`
bool readForceDat(const std::string& path, vector& total, vector& pressure, vector& viscous)
{
    std::ifstream in(path);
    std::string line;
    std::string last;
    while (std::getline(in, line))
    {
        if (!line.empty() && line[0] != '#') last = line;
    }
    if (last.empty()) return false;
    std::istringstream is(last);
    scalar t = 0;
    is >> t >> total.x >> total.y >> total.z >> pressure.x >> pressure.y >> pressure.z
       >> viscous.x >> viscous.y >> viscous.z;
    return static_cast<bool>(is);
}

scalar rel(const vector& a, const vector& b)
{
    return mag(a - b)/std::fmax(mag(b), scalar(1e-300));
}

}   // namespace


int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <ofTimeDir> <patch>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string ofDir = argv[2];
    const std::string pname = argv[3];

    std::printf("== brae body forces vs OpenFOAM: patch `%s` at %s ==\n", pname.c_str(), ofDir.c_str());

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    // a MOVING mesh writes only its points at a time directory; the topology stays in constant/
    const std::vector<vector> movedPoints = readPointsFile(ofDir + "/polyMesh/points");
    check("OpenFOAM wrote the moved points at this time",
          movedPoints.size() == static_cast<std::size_t>(m.nPoints()));
    if (movedPoints.size() == static_cast<std::size_t>(m.nPoints())) m.movePoints(movedPoints);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    std::vector<label> ids;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].name == pname) ids.push_back(static_cast<label>(pi));
    }
    check("the case has the body's patch", !ids.empty());
    if (ids.empty()) { std::printf("test_rigid_body_forces_vs_openfoam: %d failures\n", ++failures); return 1; }

    // U, WITH THE PATCH VALUES OPENFOAM HELD. forces reads the value the field carries, and a
    // movingWallVelocity patch computes its own from the mesh motion -- brae's class seeds it at zero
    // without one, and that zero is a wall that is not moving: it took grad(U)'s wall-normal row 13%
    // out and the viscous force with it.
    const FieldData<vector> uf = readField<vector>(ofDir + "/U");
    GeometricField<vector> U = buildField<vector>(uf, patches, m.nCells());
    std::size_t nSeeded = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (const auto& b : uf.boundary)
        {
            if (b.name != patches[pi].name || b.values.empty()) continue;
            U.boundary[pi]->setValue(b.values);
            ++nSeeded;
        }
    }
    check("U's patch values were taken from the file, as forces reads them", nSeeded > 0);

    auto boundaryOf = [&](const std::string& name)
    {
        GeometricField<scalar> f =
            buildField<scalar>(readField<scalar>(ofDir + "/" + name), patches, m.nCells());
        f.evaluateBoundary();
        std::vector<std::vector<scalar>> out(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi) out[pi] = f.boundary[pi]->value();
        return out;
    };
    // ...AS FIELDS, not as patch VALUE lists: alpha.water's entry on the body is `zeroGradient` with
    // nothing written under it, and reading its `value` gives zero -- air where there is water.
    const auto pB = boundaryOf("p");
    const auto aB = boundaryOf("alpha.water");
    const auto nutB = boundaryOf("nut");

    // the case's own phases
    cpu::twoPhase::PhaseProperties pp;
    {
        const FoamDict tp = readDict(caseDir + "/constant/transportProperties");
        const FoamDict* w = tp.subDict("water");
        const FoamDict* a = tp.subDict("air");
        check("constant/transportProperties names `water` and `air`", w && a);
        if (!w || !a) { std::printf("test_rigid_body_forces_vs_openfoam: %d failures\n", ++failures); return 1; }
        pp.rho1 = w->scalarOr("rho", 0);
        pp.nu1 = w->scalarOr("nu", 0);
        pp.rho2 = a->scalarOr("rho", 0);
        pp.nu2 = a->scalarOr("nu", 0);
        std::printf("  phases: water rho %g nu %g, air rho %g nu %g\n",
                    (double)pp.rho1, (double)pp.nu1, (double)pp.rho2, (double)pp.nu2);
        check("...with a water density that is the tutorial's, not a round 1000",
              pp.rho1 > scalar(900) && pp.rho1 < scalar(1100) && pp.rho1 != scalar(1000));
    }

    std::vector<std::vector<scalar>> rhoB(patches.size()), nuEffB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        std::vector<scalar> mu, nu;
        cpu::twoPhase::mixtureMu(aB[pi], pp, mu);
        cpu::twoPhase::mixtureNu(aB[pi], mu, pp, nu);
        rhoB[pi].resize(aB[pi].size());
        nuEffB[pi].resize(aB[pi].size());
        for (std::size_t i = 0; i < aB[pi].size(); ++i)
        {
            rhoB[pi][i] = aB[pi][i]*pp.rho1 + (scalar(1) - aB[pi][i])*pp.rho2;
            nuEffB[pi][i] = nu[i] + nutB[pi][i];
        }
    }

    const ForceResult R =
        bodyForces(U, pB, rhoB, nuEffB, m, g, patches, ids, vector{0, 0, 0}, scalar(0));

    vector ofTotal{0, 0, 0}, ofP{0, 0, 0}, ofV{0, 0, 0};
    const bool haveDat = readForceDat(caseDir + "/postProcessing/forces/0/force.dat", ofTotal, ofP, ofV);
    check("the oracle wrote its force.dat with the pressure and viscous halves", haveDat);
    if (!haveDat) { std::printf("test_rigid_body_forces_vs_openfoam: %d failures\n", ++failures); return 1; }

    std::printf("  force   brae (%.15g %.15g %.15g)\n          OF   (%.15g %.15g %.15g)   relative %.3e\n",
                (double)R.total().x, (double)R.total().y, (double)R.total().z,
                (double)ofTotal.x, (double)ofTotal.y, (double)ofTotal.z, (double)rel(R.total(), ofTotal));
    std::printf("  pressure relative %.3e   viscous relative %.3e\n",
                (double)rel(R.pressure, ofP), (double)rel(R.viscous, ofV));
    check("brae's force on the body is OpenFOAM's", rel(R.total(), ofTotal) < scalar(1e-12));
    check("...its pressure half", rel(R.pressure, ofP) < scalar(1e-12));
    check("...and its viscous half, which is the one that carries the phase",
          rel(R.viscous, ofV) < scalar(1e-11));

    vector ofMTotal{0, 0, 0}, ofMP{0, 0, 0}, ofMV{0, 0, 0};
    if (readForceDat(caseDir + "/postProcessing/forces/0/moment.dat", ofMTotal, ofMP, ofMV))
    {
        std::printf("  moment  brae (%.15g %.15g %.15g)\n          OF   (%.15g %.15g %.15g)   relative %.3e\n",
                    (double)R.moment().x, (double)R.moment().y, (double)R.moment().z,
                    (double)ofMTotal.x, (double)ofMTotal.y, (double)ofMTotal.z,
                    (double)rel(R.moment(), ofMTotal));
        check("brae's moment about the body's origin is OpenFOAM's",
              rel(R.moment(), ofMTotal) < scalar(1e-12));
    }
    else
    {
        check("the oracle wrote its moment.dat", false);
    }

    // PER FACE, against the `force` field forces writes with `writeFields`: the totals could agree by
    // cancellation and this says they do not.
    // WHAT IT HOLDS, and what it does not: the expression is re-formed here rather than taken out of
    // bodyForces, which returns sums only. So this arm checks the INPUTS face by face -- Sf, p, the
    // per-face rho and nuEff, and grad(U) at the wall -- and NOT the summation above it. Breaking
    // bodyForces' own arithmetic leaves this arm green and the three total arms red, which is how it
    // was measured (devTwoSymm dropped: viscous 8.378e-02, per-face still 1.8e-15).
    {
        const FieldData<vector> ff = readField<vector>(ofDir + "/force");
        const PatchFieldData<vector>* e = nullptr;
        for (const auto& b : ff.boundary)
        {
            if (b.name == pname) e = &b;
        }
        check("OpenFOAM wrote its per-face force on the body", e && !e->values.empty());
        if (e && !e->values.empty())
        {
            const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, patches);
            const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, m, g, patches);
            const std::size_t pi = static_cast<std::size_t>(ids[0]);
            const FvPatch& wp = patches[pi];
            scalar worst = 0, sc = 0;
            std::size_t nAbove = 0;
            for (label i = 0; i < wp.size; ++i)
            {
                const std::size_t f = static_cast<std::size_t>(i);
                const vector Sf = g.Sf()[wp.start + i];
                const tensor devReff = (-rhoB[pi][f]*nuEffB[pi][f])*devTwoSymm(gradB[pi][f]);
                const vector mine = pB[pi][f]*Sf + dot(Sf, devReff);
                const scalar d = mag(mine - e->values[f]);
                if (d > worst) worst = d;
                if (d > scalar(1e-10)) ++nAbove;
                sc = std::fmax(sc, mag(e->values[f]));
            }
            std::printf("  per-face force: worst %.4e of %.4e, %zu of %d faces above 1e-10\n",
                        (double)worst, (double)sc, nAbove, (int)wp.size);
            check("...and it is OpenFOAM's face by face", nAbove == 0 && worst < scalar(1e-12)*sc);
        }
    }

    // CONTROL 1: the KINEMATIC reading -- rhoInf = 1 in the viscous stress, which is what
    // wallForces above carries and what a single-phase case can get away with. Here the wall is wet.
    {
        std::vector<std::vector<scalar>> one(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            one[pi].assign(rhoB[pi].size(), scalar(1));
        }
        const ForceResult K = bodyForces(U, pB, one, nuEffB, m, g, patches, ids, vector{0,0,0}, scalar(0));
        std::printf("  CONTROL: rhoInf = 1 instead of the per-face density: viscous relative %.3e\n",
                    (double)rel(K.viscous, ofV));
        check("...the two-phase density is what the viscous stress is built on",
              rel(K.viscous, ofV) > scalar(0.9));
    }

    // CONTROL 2: the viscous half dropped altogether
    {
        // The body is BUOYANT, so the total force is 257 N of pressure in z and the viscous half is
        // milli-newtons: 3.6e-05 of the whole. It is 9.1e-04 of the x component, which is the
        // direction the body is free to translate in, so it is not noise -- and it is stated at its
        // measured size rather than given a bound that flatters it.
        const scalar dropAll = rel(R.pressure, ofTotal);
        const scalar dropX = std::fabs(R.pressure.x - ofTotal.x)/std::fmax(std::fabs(ofTotal.x), scalar(1e-300));
        std::printf("  CONTROL: the viscous half dropped: force relative %.3e (and %.3e of its x "
                    "component, the direction this body's Py joint is free in)\n",
                    (double)dropAll, (double)dropX);
        check("...the viscous force is not noise beside the pressure one",
              dropAll > scalar(1e-5) && dropX > scalar(1e-4));
    }

    std::printf("test_rigid_body_forces_vs_openfoam: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
