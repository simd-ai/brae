/*---------------------------------------------------------------------------*\
  liqref -- print OpenFOAM's OWN liquid thermo answers, as an oracle for brae.

  WHY THIS EXISTS. brae's h2o correlations are already gated against a table this tool's ancestor
  produced. What that table cannot gate is the he -> T INVERSION's PATH. OpenFOAM's
  species::thermo<>::T (thermoI.H:43-88) is a do-while that stops when the TEMPERATURE STEP falls below
  T0*tol_ with tol_ = 1e-4 (thermo.C:33) -- it does NOT iterate to convergence, so the temperature it
  returns DEPENDS ON THE INITIAL GUESS T0. A table of (Es, p) -> T recorded at one T0 therefore cannot
  tell a path-faithful port from a port that simply converges harder.

  So this prints TEs / THs for a grid of (energy, p, T0) triples, with T0 deliberately cold and hot
  relative to the answer, at 17 significant digits. It also prints the raw properties so the same table
  covers the correlations and the inversion together.

  Nothing here is brae's arithmetic: the mixture is constructed exactly as liquidThermo.H builds it for
  `properties liquid` + `energy sensibleInternalEnergy`, and every number comes from OpenFOAM calling
  itself.
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "thermophysicalPropertiesSelector.H"
#include "liquidProperties.H"
#include "sensibleInternalEnergy.H"
#include "sensibleEnthalpy.H"
#include "thermo.H"
#include "IOmanip.H"
#include <vector>

using namespace Foam;

typedef species::thermo
<
    thermophysicalPropertiesSelector<liquidProperties>,
    sensibleInternalEnergy
> eThermo;

typedef species::thermo
<
    thermophysicalPropertiesSelector<liquidProperties>,
    sensibleEnthalpy
> hThermo;

int main(int argc, char *argv[])
{
    argList::noBanner();
    argList::noParallel();
    argList::noFunctionObjects();
    argList::addNote("print OpenFOAM's own H2O liquid properties and he->T inversions");
    argList args(argc, argv, false, false, false);

    const eThermo eT(thermophysicalPropertiesSelector<liquidProperties>("H2O"));
    const hThermo hT(thermophysicalPropertiesSelector<liquidProperties>("H2O"));

    Info<< setprecision(17);

    // The pressures and temperatures the squareBendLiq tutorial actually visits, plus the ends of the
    // correlation range so a port cannot pass by being right only in the middle.
    const std::vector<scalar> ps{1.0e5, 2.0e5, 5.0e5, 1.0e6};
    const std::vector<scalar> Ts{280.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 640.0};
    // T0 relative to the answer: exact, warm, cold, hot, and very cold -- the spread that exposes a
    // stopping test measured against T0 rather than against convergence.
    const std::vector<scalar> dT0{0.0, 0.1, -50.0, +50.0, -150.0, +150.0};

    Info<< "# PROPS p T rho mu kappa Cp Cv Es Hs" << nl;
    for (const scalar p : ps)
    {
        for (const scalar T : Ts)
        {
            Info<< "PROPS " << p << ' ' << T
                << ' ' << eT.rho(p, T)
                << ' ' << eT.mu(p, T)
                << ' ' << eT.kappa(p, T)
                << ' ' << eT.Cp(p, T)
                << ' ' << eT.Cv(p, T)
                << ' ' << eT.Es(p, T)
                << ' ' << hT.Hs(p, T)
                << nl;
        }
    }

    // THE POINT OF THE FILE. For each (p, T) the energy is evaluated once, then inverted from several
    // starting guesses. A path-faithful port reproduces EVERY row; a port that iterates to convergence
    // reproduces only the rows whose T0 happens to be close enough that OF converged too.
    Info<< "# INV form p Ttrue T0 Trecovered" << nl;
    for (const scalar p : ps)
    {
        for (const scalar T : Ts)
        {
            const scalar es = eT.Es(p, T);
            const scalar hs = hT.Hs(p, T);
            for (const scalar d : dT0)
            {
                const scalar T0 = T + d;
                if (T0 <= 0) continue;
                Info<< "INV e " << p << ' ' << T << ' ' << T0 << ' ' << eT.TEs(es, p, T0) << nl;
                Info<< "INV h " << p << ' ' << T << ' ' << T0 << ' ' << hT.THs(hs, p, T0) << nl;
            }
        }
    }

    Info<< "# END" << nl;
    return 0;
}
