#pragma once
// Shared turbulence model + start-field setup for the steady (gpuSimpleFoam) and transient (gpuPimpleFoam) drivers.
// readTurbulenceModel: constant/turbulenceProperties RASModel -> ctl.turbulent/sst/sa/lm + coeffs. readTurbulenceFields:
// reads k/second/nut (or nuTilda/nut for SA, +ReThetat/gammaInt for LM), the nut wall function from the 0/nut BC,
// guards wall-function-on-non-wall patches, and applies the turbulent-inlet BCs. Bodies extracted verbatim from
// gpuSimpleFoam so both drivers stay identical. Call readTurbulenceModel BEFORE readTurbulenceFields (needs ctl.sst).
#include "solver_controls.cuh"
#include "patch_entry_lookup.cuh"
#include "foam_dict.cuh"
#include "brae_notice.cuh"
#include "foam_field_reader.cuh"
#include "geometric_field.cuh"
#include "fv_patch.cuh"
#include "kepsilon_coeffs.cuh"
#include "komega_sst_coeffs.cuh"
#include "spalart_coeffs.cuh"
#include "turbulent_inlet.cuh"
#include "frozen_bc_guard.cuh"
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace brae {

// constant/turbulenceProperties RASModel -> ctl turbulence flags + coeffs (ctl.turbulent must already be set).
// The nut wall function the case's 0/nut SELECTS, and the refusals that go with it. Free rather than a
// lambda inside readTurbulenceFields because simpleFoamV2 needs the same answer: it had no selector at
// all and ran the k-based nutk under every BC, so a `nutUBlendedWallFunction` case got a wall viscosity
// from the wrong formula -- measured on backwardFacingStep2D as a wall nut of 0 where the dispatching
// path gives up to 1.5e-01. One implementation, because a second one is how the two paths disagree
// about what the case asked for.
inline bool isNutWallFnType(const std::string& t)
{
    return t == "nutkWallFunction"      || t == "nutUSpaldingWallFunction"
        || t == "nutLowReWallFunction"  || t == "nutUBlendedWallFunction"
        || t == "nutUWallFunction"      || t == "atmNutkWallFunction";
}

inline void selectNutWall(
    const FieldData<scalar>&    nutFD,
    const std::vector<FvPatch>& fvp,
    bool                        sa,
    const std::string&          modelName,
    NutWall&                    nutWall,
    scalar&                     atmZ0,
    bool&                       atmBoundNut)
{

            std::string wallFnSeen;   // the first wall function seen; a second, different one refuses
            for (const auto& pb : nutFD.boundary)
            {
                // Only wall patches drive the choice -- resolved through the same machinery buildField
                // uses, so a regex key covering the walls counts as the walls. An entry resolving to NO
                // patch keeps participating (the old exact-name compare let it, and the refusal
                // fixtures stage conflicts through exactly such entries).
                const auto resolved = patchesResolvingTo(nutFD.boundary, pb, fvp);
                bool onWall = resolved.empty();
                for (const FvPatch* q : resolved)
                    if (q->type == "wall") { onWall = true; break; }
                if (!onWall) continue;
                if (pb.type == "nutUSpaldingWallFunction") { nutWall = NutWall::Spalding; }
                else if (pb.type == "nutUBlendedWallFunction") { nutWall = NutWall::Blended; }
                // nutUWallFunction: OF's default blender is STEPWISE (nutUWallFunctionFvPatchScalarField.C:259,
                // wallFunctionBlenders(dict, blenderType::STEPWISE, 4)). Any other blender is a different
                // formula, so it is refused rather than approximated by the stepwise one.
                else if (pb.type == "nutUWallFunction") { nutWall = NutWall::NutU; }
                else if (pb.type == "atmNutkWallFunction")   // atmospheric rough-wall nut (k-based path + roughness z0)
                {
                    atmZ0 = pb.ablZ0;               // roughness length (from `z0` / $z0 include)
                    atmBoundNut = pb.atmBoundNut;  // clamp nut>=0 option
                    printf("  nut wall function: atmNutkWallFunction (rough, z0=%g, boundNut=%s) on %s per the BC\n",
                           (double)atmZ0, atmBoundNut ? "true" : "false", modelName.c_str());
                }
                // nutLowReWallFunction: OF's calcNut() returns Zero UNCONDITIONALLY
                // (nutLowReWallFunctionFvPatchScalarField.C:38-42 is the entire function). This used to
                // warn on stderr and fall through to nutk, justified as identical on a resolved mesh --
                // and that justification does not hold: nutk's yPlus is the K-BASED
                // Cmu^0.25*y*sqrt(k)/nu, not u_tau*y/nu, so a mesh resolved in friction units can carry
                // k-based y+ above yPlusLam and take the log branch where OpenFOAM returns 0. It is now
                // selected rather than substituted; writing zero is exact and cheaper than the log law.
                else if (pb.type == "nutLowReWallFunction") { nutWall = NutWall::LowRe; }
                // ONE SELECTOR, SO ONE FUNCTION. OpenFOAM dispatches per patch --
                // nutWallFunctionFvPatchScalarField.C:181-184 is operator==(calcNut()) on each patch's own
                // object -- so every wall may carry a different one and OpenFOAM honours each. nutWall
                // is a single case-wide value, and the winner's kernel then rewrites EVERY wall face
                // (device_kepsilon.cu spaldingNutKernel/blendedNutKernel/nutUWallKernel all write
                // unconditionally where isWall). The per-face rescues are gated `type != wall`, so nothing
                // spares the losing patch.
                //
                // Two ways that went wrong silently, both now refused rather than resolved by accident:
                //   * LAST WINS. The loop assigns as it walks the boundary list, so the last matching
                //     patch decided for all of them.
                //   * nutk CANNOT WIN BACK. There is no `nutkWallFunction` branch here and no restoring
                //     else, so once any patch selected a non-nutk function every wall got it -- including
                //     the walls that explicitly asked for nutkWallFunction.
                //
                // Same shape as the z0 refusal this driver already carries for atmNutkWallFunction
                // (simpleFoamV2.cu:942-952): brae holds one value, so two different ones must be refused
                // rather than averaged into a case nobody described.
                if (!wallFnSeen.empty() && wallFnSeen != pb.type && isNutWallFnType(pb.type))
                    throw std::runtime_error(
                        "brae: 0/nut carries more than one nut wall function on wall patches ('"
                        + wallFnSeen + "' and '" + pb.type + "'). This driver applies ONE wall function to "
                        "every wall, so running would give a wall the function another patch asked for. "
                        "OpenFOAM dispatches per patch and honours both. Refusing rather than silently "
                        "picking whichever the boundary list happens to end on.");
                if (isNutWallFnType(pb.type)) wallFnSeen = pb.type;
            }
            if (!sa && nutWall != NutWall::Nutk)
            {
                // LowRe is NOT velocity-based, so it cannot ride the ternary below -- labelling it
                // `nutUBlendedWallFunction (velocity-based)` would misreport the one case this branch
                // was just taught to handle.
                if (nutWall == NutWall::LowRe)
                    printf("  nut wall function: nutLowReWallFunction (nut = 0 at the wall, honoured on "
                           "%s per the BC)\n", modelName.c_str());
                else
                    printf("  nut wall function: %s (velocity-based, honoured on %s per the BC)\n",
                           nutWall == NutWall::Spalding ? "nutUSpaldingWallFunction"
                           : nutWall == NutWall::NutU ? "nutUWallFunction" : "nutUBlendedWallFunction",
                           modelName.c_str());
            }
}

inline void readTurbulenceModel(const FoamDict& turbProps, DeviceSimpleControls& ctl)
{
        // `simulationType laminar` DOES NOT MEAN "no model". OF selects a laminarModel, and the default
        // (Stokes) is the only one that leaves the molecular viscosity alone. A
        //     laminar { model generalizedNewtonian; viscosityModel powerLaw; ... }
        // replaces nu with a strain-rate-dependent field, which is a different momentum equation.
        //
        // REFUSED RATHER THAN IGNORED. Measured on squareBendLiqNoNewtonian: OF's powerLaw is
        //     nu = max(nuMin, min(nuMax, nu0*pow(max(strainRate, SMALL), n-1)))
        // and with that case's nu0 = mu/rho ~ 8.9e-7, n = 0.4 and nuMin = 1e-3, nu sits AT nuMin over
        // essentially the whole field -- about 1120x the Newtonian value. brae previously read
        // `simulationType laminar`, never looked inside the sub-dictionary, ran with the molecular nu and
        // produced a confident non-converged answer (Ux 1.9e-2, p 3.4e-1 still oscillating at iteration
        // 500). The dict audit did flag `laminar/` as unread, which is what a notice is for; a viscosity
        // model is not a notice-level omission.
        if (!ctl.turbulent)
        {
            if (const FoamDict* lam = turbProps.subDict("laminar"))
            {
                const std::string lmodel = lam->wordOr("model", "Stokes");
                if (lmodel == "generalizedNewtonian")
                {
                    // OF reads the coefficients from powerLawCoeffs{} if present, else from the enclosing
                    // dictionary (dictionary::optionalSubDict), which is how this tutorial writes them.
                    const std::string vm = lam->wordOr("viscosityModel", "");
                    if (vm != "powerLaw")
                        throw std::runtime_error(
                            "brae: unsupported generalizedNewtonian viscosityModel '" + vm +
                            "' (only 'powerLaw' is implemented).");
                    const FoamDict* co = lam->subDict("powerLawCoeffs");
                    const FoamDict& src = co ? *co : *lam;
                    ctl.gnPowerLaw = true;
                    // ALL THREE are required: OF powerLaw.C:63-65 constructs n_, nuMin_ and nuMax_
                    // straight from the dict with no default, and fatals on a missing entry. The old
                    // guard tested only nuMax, so a case missing `n` silently got n = 1.0 -- which makes
                    // nu = nu0 identically, the NEWTONIAN answer, on a case that asked for shear
                    // thinning. squareBendLiqNoNewtonian records what that is worth: nu sits at nuMin
                    // over essentially the whole field, ~1120x the Newtonian value (see above).
                    if (!src.found("n") || !src.found("nuMin") || !src.found("nuMax"))
                        throw std::runtime_error(
                            "brae: generalizedNewtonian powerLaw needs all three of n, nuMin, nuMax "
                            "(OpenFOAM powerLaw.C constructs each with no default and fatals without "
                            "it); missing: " + std::string(!src.found("n") ? "n " : "")
                            + (!src.found("nuMin") ? "nuMin " : "")
                            + (!src.found("nuMax") ? "nuMax" : ""));
                    ctl.gnN     = src.scalarOr("n", 1.0);
                    ctl.gnNuMin = src.scalarOr("nuMin", 0.0);
                    ctl.gnNuMax = src.scalarOr("nuMax", 0.0);
                    if (ctl.gnNuMax <= 0.0)
                        throw std::runtime_error("brae: generalizedNewtonian powerLaw needs nuMin and nuMax.");
                    std::printf("  laminar generalizedNewtonian/powerLaw: n=%.4g nuMin=%.4g nuMax=%.4g"
                                "  (nu = clamp(nu0*strainRate^(n-1)))\n", ctl.gnN, ctl.gnNuMin, ctl.gnNuMax);
                }
                else if (lmodel == "Maxwell")
                {
                    // OF laminarModels::Maxwell. dimensionedScalar(name, dims, coeffDict_) THROWS when the
                    // entry is absent, and coeffDict_ is dictionary::optionalSubDict("MaxwellCoeffs") --
                    // so both spellings the tutorials use are valid: planarPoiseuille writes the
                    // sub-dictionary, planarContraction writes the keys inline.
                    const FoamDict* mc = lam->subDict("MaxwellCoeffs");
                    const FoamDict& src = mc ? *mc : *lam;
                    ctl.maxwell = true;
                    ctl.maxwellNuM    = src.scalarOr("nuM", -1.0);
                    ctl.maxwellLambda = src.scalarOr("lambda", -1.0);
                    if (!(ctl.maxwellNuM >= 0.0) || !(ctl.maxwellLambda > 0.0))
                        throw std::runtime_error(
                            "brae: laminar model Maxwell needs `nuM` (>= 0) and `lambda` (> 0), in "
                            "MaxwellCoeffs{} or directly in laminar{}. OpenFOAM also refuses without them "
                            "-- there is no default relaxation time.");
                    std::printf("  laminar Maxwell (viscoelastic): nuM=%.4g lambda=%.4g"
                                "  (nu0 = nu + nuM; stress relaxes over lambda)\n",
                                ctl.maxwellNuM, ctl.maxwellLambda);
                }
                else if (lmodel != "Stokes")
                    throw std::runtime_error(
                        "brae: unsupported laminar model '" + lmodel + "' in constant/turbulenceProperties. "
                        "brae has 'Stokes' (the OF default, molecular viscosity unchanged), "
                        "'generalizedNewtonian' and 'Maxwell'. A "
                        "generalizedNewtonian model replaces nu with a strain-rate-dependent field, so "
                        "running without it is a different momentum equation, not an approximation.");
            }
        }
        if (ctl.turbulent)
        {
            // simulationType LES: DES/LES models live under an LES{} sub-dict (OF convention). SA-DDES reuses the SA
            // transport (ctl.sa) plus the DES length-scale limiter (ctl.des); only cubeRootVol delta is supported (v1).
            if (const FoamDict* les = turbProps.subDict("LES"))
            {
                const std::string model = les->wordOr("LESModel", "");
                const bool saIddes  = (model == "SpalartAllmarasIDDES");
                const bool sstIddes = (model == "kOmegaSSTIDDES");
                const bool saDes  = (model == "SpalartAllmarasDDES" || model == "SpalartAllmarasDES" || saIddes);
                const bool sstDes = (model == "kOmegaSSTDDES" || model == "kOmegaSSTDES" || sstIddes);
                const bool smag   = (model == "Smagorinsky");
                const bool wale   = (model == "WALE");
                if (!saDes && !sstDes && !smag && !wale)
                    throw std::runtime_error("brae: unsupported LESModel '" + model
                        + "' (Smagorinsky, WALE, SpalartAllmarasDDES/DES/IDDES or kOmegaSSTDDES/DES/IDDES)");
                ctl.modelName = model;
                const std::string delta = les->wordOr("delta", "cubeRootVol");
                if (delta == "maxDeltaxyz")
                {
                    ctl.lesDeltaMax = true;
                    if (const FoamDict* mc = les->subDict("maxDeltaxyzCoeffs"))
                        ctl.lesDeltaCoeff = mc->scalarOr("deltaCoeff", ctl.lesDeltaCoeff);
                }
                else if (!saIddes && !sstIddes && delta != "cubeRootVol")   // IDDES computes its own (maxDeltaxyz-based) length scale internally
                    std::fprintf(stderr, "brae WARNING: LES delta '%s' not supported; using cubeRootVol (V^(1/3)).\n", delta.c_str());
                const FoamDict* dc = les->subDict(model + "Coeffs");
                if (wale)   // WALE: the other ALGEBRAIC sub-grid nut. Same slot as Smagorinsky -- no transport
                {           // scalar, no DES limiter -- only the velocity scale differs (see WaleCoeffs).
                    ctl.les = true;
                    ctl.wale = true;
                    if (dc) { ctl.waleCoeffs.Ck = dc->scalarOr("Ck", ctl.waleCoeffs.Ck);
                              ctl.waleCoeffs.Ce = dc->scalarOr("Ce", ctl.waleCoeffs.Ce);
                              ctl.waleCoeffs.Cw = dc->scalarOr("Cw", ctl.waleCoeffs.Cw); }
                    std::printf("  WALE (LES, delta=%s): Ck=%.4g Cw=%.4g\n",
                                ctl.lesDeltaMax ? "maxDeltaxyz" : "cubeRootVol",
                                ctl.waleCoeffs.Ck, ctl.waleCoeffs.Cw);
                    return;
                }
                if (smag)   // pure LES Smagorinsky: ALGEBRAIC sub-grid nut (no transport scalar, no DES length-scale limiter)
                {
                    ctl.les = true;
                    if (dc) { ctl.smagCoeffs.Ck = dc->scalarOr("Ck", ctl.smagCoeffs.Ck);
                              ctl.smagCoeffs.Ce = dc->scalarOr("Ce", ctl.smagCoeffs.Ce); }
                    std::printf("  Smagorinsky (LES, delta=%s): Ck=%.4g Ce=%.4g (equivalent Cs=%.4g)\n",
                                ctl.lesDeltaMax ? "maxDeltaxyz" : "cubeRootVol",
                                ctl.smagCoeffs.Ck, ctl.smagCoeffs.Ce, ctl.smagCoeffs.Cs());
                    return;
                }
                ctl.des = true;
                if (saDes)   // SA-DDES/IDDES: reuse the SA transport + the DES length-scale limiter
                {
                    ctl.sa = true;
                    ctl.iddes = saIddes;   // IDDES -> the improved (WMLES) length scale (maxDeltaxyz + blending); else plain DDES
                    if (dc) ctl.saCoeffs.CDES = dc->scalarOr("CDES", ctl.saCoeffs.CDES);
                    // OF v2412 SpalartAllmarasDDES carries a `shielding` selector (standard | ZDES2020).
                    // ZDES2020 (Deck & Renard 2020) multiplies the standard fd by a second shielding built
                    // from grad(nuTilda).n and grad(|curl U|).n, which moves the RANS/LES switch -- it is a
                    // different model, not a coefficient. brae runs the standard fd, so SAY so: an
                    // unimplemented input read off disk and silently dropped is the failure mode this
                    // project keeps its notices for. NACA4412 is the tutorial that asks for it.
                    // OF v2412 SpalartAllmarasDDES carries a `shielding` selector (standard | ZDES2020).
                    // ZDES2020 (Deck & Renard 2020) multiplies the standard fd by a second shielding built
                    // from grad(nuTilda).n and grad(|curl U|).n -- a different model, not a coefficient.
                    // IDDES has its own length scale and never calls fd, so the selector does not apply there.
                    if (dc)
                    {
                        const std::string sh = dc->wordOr("shielding", "standard");
                        ctl.saCoeffs.Cd1 = dc->scalarOr("Cd1", ctl.saCoeffs.Cd1);
                        ctl.saCoeffs.Cd2 = dc->scalarOr("Cd2", ctl.saCoeffs.Cd2);
                        if (sh == "ZDES2020")
                        {
                            if (saIddes)
                                throw std::runtime_error(
                                    "brae: `shielding ZDES2020` under SpalartAllmarasIDDES -- IDDES uses its own "
                                    "length scale and never evaluates the DDES shielding function, so the entry "
                                    "would have no effect. Remove it or select SpalartAllmarasDDES.");
                            ctl.saCoeffs.zdes     = true;
                            ctl.saCoeffs.Cd3      = dc->scalarOr("Cd3", ctl.saCoeffs.Cd3);
                            ctl.saCoeffs.Cd4      = dc->scalarOr("Cd4", ctl.saCoeffs.Cd4);
                            ctl.saCoeffs.betaZDES = dc->scalarOr("betaZDES", ctl.saCoeffs.betaZDES);
                            const std::string fp2 = dc->wordOr("usefP2", "false");
                            ctl.saCoeffs.usefP2   = (fp2 == "true" || fp2 == "yes" || fp2 == "on" || fp2 == "1");
                        }
                        else if (sh != "standard")
                            throw std::runtime_error(
                                "brae: SpalartAllmarasDDES `shielding " + sh + "` is not implemented "
                                "(brae has `standard` and `ZDES2020`). The shielding function decides where "
                                "the model leaves RANS for LES, so running another one is a different answer.");
                    }
                    if (saIddes && dc)   // IDDES blending-constant overrides (defaults = Shur/Spalart/Strelets/Travin 2008)
                    {
                        ctl.saCoeffs.Cdt1 = dc->scalarOr("Cdt1", ctl.saCoeffs.Cdt1);
                        ctl.saCoeffs.Cl   = dc->scalarOr("Cl",   ctl.saCoeffs.Cl);
                        ctl.saCoeffs.Ct   = dc->scalarOr("Ct",   ctl.saCoeffs.Ct);
                        ctl.saCoeffs.Cw   = dc->scalarOr("Cw",   ctl.saCoeffs.Cw);
                    }
                    if (saIddes)
                        std::printf("  %s (SA-IDDES, delta=IDDESDelta [maxDeltaxyz+hwn]): CDES=%.4g Cdt1=%.4g Cl=%.4g Ct=%.4g Cw=%.4g kappa=%.4g Cv1=%.3g\n",
                                    model.c_str(), ctl.saCoeffs.CDES, ctl.saCoeffs.Cdt1, ctl.saCoeffs.Cl, ctl.saCoeffs.Ct, ctl.saCoeffs.Cw, ctl.saCoeffs.kappa, ctl.saCoeffs.Cv1);
                    else
                        std::printf("  %s (SA-DES, delta=%s, shielding=%s): CDES=%.4g kappa=%.4g Cb1=%.4g Cw1=%.4g Cv1=%.3g\n",
                                    model.c_str(), ctl.lesDeltaMax ? "maxDeltaxyz" : "cubeRootVol",
                                    ctl.saCoeffs.zdes ? "ZDES2020" : "standard", ctl.saCoeffs.CDES, ctl.saCoeffs.kappa, ctl.saCoeffs.Cb1, ctl.saCoeffs.Cw1(), ctl.saCoeffs.Cv1);
                }
                else         // kOmegaSST-DDES/IDDES: reuse the kOmegaSST transport + the DES factor on the k destruction
                {
                    ctl.sst = true;
                    ctl.iddes = sstIddes;   // IDDES -> the improved (WMLES) length scale (maxDeltaxyz + blending); else plain DDES
                    readKOmegaSSTCoeffs(les, ctl.ksstCoeffs);   // honour an inline kOmegaSSTCoeffs under LES{} if present
                    if (dc) { ctl.ksstCoeffs.CDES1 = dc->scalarOr("CDES1", ctl.ksstCoeffs.CDES1);
                              ctl.ksstCoeffs.CDES2 = dc->scalarOr("CDES2", ctl.ksstCoeffs.CDES2); }
                    if (sstIddes && dc)   // IDDES blending-constant overrides (defaults = Gritskevich et al. 2012)
                    {
                        ctl.ksstCoeffs.Cdt1 = dc->scalarOr("Cdt1", ctl.ksstCoeffs.Cdt1);
                        ctl.ksstCoeffs.Cl   = dc->scalarOr("Cl",   ctl.ksstCoeffs.Cl);
                        ctl.ksstCoeffs.Ct   = dc->scalarOr("Ct",   ctl.ksstCoeffs.Ct);
                        ctl.ksstCoeffs.Cw   = dc->scalarOr("Cw",   ctl.ksstCoeffs.Cw);
                    }
                    if (sstIddes)
                        std::printf("  %s (kOmegaSST-IDDES, delta=IDDESDelta [maxDeltaxyz+hwn]): CDES1=%.4g CDES2=%.4g Cdt1=%.4g Cl=%.4g Ct=%.4g Cw=%.4g betaStar=%.4g\n",
                                    model.c_str(), ctl.ksstCoeffs.CDES1, ctl.ksstCoeffs.CDES2, ctl.ksstCoeffs.Cdt1, ctl.ksstCoeffs.Cl, ctl.ksstCoeffs.Ct, ctl.ksstCoeffs.Cw, ctl.ksstCoeffs.betaStar);
                    else
                        std::printf("  %s (kOmegaSST-DES, delta=%s): CDES1=%.4g CDES2=%.4g betaStar=%.4g a1=%.4g\n",
                                    model.c_str(), ctl.lesDeltaMax ? "maxDeltaxyz" : "cubeRootVol", ctl.ksstCoeffs.CDES1, ctl.ksstCoeffs.CDES2, ctl.ksstCoeffs.betaStar, ctl.ksstCoeffs.a1);
                }
                return;
            }
            const FoamDict* ras = turbProps.subDict("RAS");
            // OF Switch: yes/no/on/off/true/false/1/0. Default true when absent (RASModel.C:70).
            if (ras)
            {
                const std::string sw = ras->wordOr("turbulence", "true");
                ctl.turbulenceOn = !(sw == "off" || sw == "no" || sw == "false" || sw == "0");
                if (!ctl.turbulenceOn)
                    noticeApplied("turbulenceProperties RAS/turbulence",
                                  "'" + sw + "' -- the model is FROZEN: k/epsilon|omega/nut keep their initial "
                                  "values and momentum uses that frozen nut (OF RASModel::correct() returns "
                                  "immediately). This is not the same as simulationType laminar.");
            }
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            ctl.lm  = (model == "kOmegaSSTLM");                 // Langtry-Menter transition = kOmegaSST + gamma-ReThetat
            ctl.sst = (model == "kOmegaSST") || ctl.lm;
            ctl.sa  = (model == "SpalartAllmaras");
            const bool rke = (model == "realizableKE");
            const bool rng = (model == "RNGkEpsilon");
            if (model != "kEpsilon" && !rke && !rng && !ctl.sst && !ctl.sa)
                throw std::runtime_error("brae: unsupported RASModel '" + model + "' (kEpsilon, RNGkEpsilon, realizableKE, kOmegaSST, kOmegaSSTLM or SpalartAllmaras)");
            ctl.modelName = model;
            if (ctl.sa)
            {
                // Spalart-Allmaras: OF defaults (coeffs read from RAS.SpalartAllmarasCoeffs would override; not needed here).
                const SpalartAllmarasCoeffs& c = ctl.saCoeffs;
                std::printf("  SpalartAllmaras (OF defaults): sigmaNut=%.4g kappa=%.4g Cb1=%.4g Cb2=%.4g Cw1=%.4g Cw2=%.3g Cw3=%.3g Cv1=%.3g Cs=%.3g\n",
                            c.sigmaNut, c.kappa, c.Cb1, c.Cb2, c.Cw1(), c.Cw2, c.Cw3, c.Cv1, c.Cs);
            }
            else if (ctl.sst)
            {
                // kOmegaSST coefficients: OF turbulenceProperties RAS.kOmegaSSTCoeffs (absent keys keep OF defaults).
                readKOmegaSSTCoeffs(ras, ctl.ksstCoeffs);
                const KOmegaSSTCoeffs& c = ctl.ksstCoeffs;
                std::printf("  kOmegaSSTCoeffs: a1=%.4g betaStar=%.4g alphaK1=%.4g alphaOmega1=%.4g gamma1=%.4g beta1=%.4g\n",
                            c.a1, c.betaStar, c.alphaK1, c.alphaOmega1, c.gamma1, c.beta1);
            }
            else
            {
                // k-eps coefficients: RAS.kEpsilonCoeffs (kappa/E wall-function coeffs accepted if present).
                KEpsilonCoeffs& c = ctl.keCoeffs;
                if (rke)   // realizableKE: variable Cmu + strain production; OF defaults A0=4, C2=1.9, sigmak=1, sigmaEps=1.2
                {
                    c.realizable = true;
                    c.A0 = 4.0;
                    c.C2 = 1.9;
                    c.sigmaK = 1.0;
                    c.sigmaEps = 1.2;
                    if (const FoamDict* rkc = ras ? ras->subDict("realizableKECoeffs") : nullptr)
                    {
                        c.A0 = rkc->scalarOr("A0", c.A0);
                        c.C2 = rkc->scalarOr("C2", c.C2);
                        c.sigmaK = rkc->scalarOr("sigmak", c.sigmaK);
                        c.sigmaEps = rkc->scalarOr("sigmaEps", c.sigmaEps);
                        c.kappa = rkc->scalarOr("kappa", c.kappa);
                        c.E = rkc->scalarOr("E", c.E);
                    }
                    std::printf("  realizableKECoeffs: A0=%.4g C2=%.4g sigmak=%.4g sigmaEps=%.4g kappa=%.4g E=%.4g (variable Cmu)\n",
                                c.A0, c.C2, c.sigmaK, c.sigmaEps, c.kappa, c.E);
                }
                else if (rng)
                {
                    // RNGkEpsilon: EVERY coefficient differs from the standard model, not just the extra R
                    // term -- Cmu 0.0845 (not 0.09) feeds nut and the nut wall functions too, so reading
                    // the RNG model with kEpsilon's defaults would be wrong even where R happens to vanish.
                    c.rng = true;
                    c.Cmu = 0.0845;
                    c.C1 = 1.42;
                    c.C2 = 1.68;
                    c.C3 = -0.33;
                    c.sigmaK = 0.71942;
                    c.sigmaEps = 0.71942;
                    if (const FoamDict* rgc = ras ? ras->subDict("RNGkEpsilonCoeffs") : nullptr)
                    {
                        c.Cmu = rgc->scalarOr("Cmu", c.Cmu);
                        c.C1 = rgc->scalarOr("C1", c.C1);
                        c.C2 = rgc->scalarOr("C2", c.C2);
                        c.C3 = rgc->scalarOr("C3", c.C3);
                        c.sigmaK = rgc->scalarOr("sigmak", c.sigmaK);
                        c.sigmaEps = rgc->scalarOr("sigmaEps", c.sigmaEps);
                        c.eta0 = rgc->scalarOr("eta0", c.eta0);
                        c.beta = rgc->scalarOr("beta", c.beta);
                        c.kappa = rgc->scalarOr("kappa", c.kappa);
                        c.E = rgc->scalarOr("E", c.E);
                    }
                    std::printf("  RNGkEpsilonCoeffs: Cmu=%.4g C1=%.4g C2=%.4g C3=%.4g sigmak=%.5g sigmaEps=%.5g "
                                "eta0=%.4g beta=%.4g kappa=%.4g E=%.4g\n",
                                c.Cmu, c.C1, c.C2, c.C3, c.sigmaK, c.sigmaEps, c.eta0, c.beta, c.kappa, c.E);
                }
                else
                {
                    const FoamDict* kec = ras ? ras->subDict("kEpsilonCoeffs") : nullptr;
                    if (kec)
                    {
                        c.Cmu = kec->scalarOr("Cmu", c.Cmu);
                        c.C1 = kec->scalarOr("C1", c.C1);
                        c.C2 = kec->scalarOr("C2", c.C2);
                        c.C3 = kec->scalarOr("C3", c.C3);
                        c.sigmaK = kec->scalarOr("sigmak", c.sigmaK);
                        c.sigmaEps = kec->scalarOr("sigmaEps", c.sigmaEps);
                        c.kappa = kec->scalarOr("kappa", c.kappa);
                        c.E = kec->scalarOr("E", c.E);
                    }
                    std::printf("  kEpsilonCoeffs%s: Cmu=%.4g C1=%.4g C2=%.4g C3=%.4g sigmak=%.4g sigmaEps=%.4g kappa=%.4g E=%.4g\n",
                                kec ? " (from dict)" : " (OF defaults)", c.Cmu, c.C1, c.C2, c.C3, c.sigmaK, c.sigmaEps, c.kappa, c.E);
                }
            }
        }
}

// The turbulence start fields, read from fieldDir with the nut wall function + turbulent-inlet BCs applied.
struct TurbulenceFields { GeometricField<scalar> k, eps, nut, ReThetat, gammaInt;     TurbulentInletMasks turbInletMasks;
};

inline TurbulenceFields readTurbulenceFields(const std::string& fieldDir, const std::vector<FvPatch>& fvp, label nC,
                                             DeviceSimpleControls& ctl, const std::string& secondName,
                                             const GeometricField<vector>& U,
                                             // Non-null: the calling driver does NOT maintain per-step
                                             // boundaries on these fields, so refuse them by that name
                                             // (frozen_bc_guard.cuh). gpuPimpleFoam maintains fixedMean
                                             // on k/epsilon/omega/nuTilda/nut and passes null.
                                             const char* frozenGuardDriver = nullptr,
                                             bool frozenGuardCodedMaintained = false)
{
    auto guardFrozen = [&](const FieldData<scalar>& fd, const std::string& nm)
    {
        if (frozenGuardDriver)
            refuseFrozenPerStepBC(fd, nm, frozenGuardDriver, frozenGuardCodedMaintained);
    };
    TurbulentInletMasks masks;
        // Wall-function fidelity guard -- fail loud on a nut/epsilon/omega wall-function BC placed on a patch NOT typed
        // 'wall': brae gates the near-wall model on the geometric patch type, so the wall function would be SILENTLY
        // inert. The entry resolves to its patches through the SAME machinery buildField uses
        // (patchesResolvingTo: exact name, group, regex, last pattern wins), so a regex-keyed wall
        // function -- `"(upperWall|lowerWall)"` is how backwardFacingStep2D writes every one of its wall
        // BCs -- is checked like a concrete one. It used to be compared by exact name and silently
        // skipped (audit finding #16), which disarmed this guard on exactly the cases that use it most.
        // An entry that resolves to NO patch is dead text and stays skipped, as OpenFOAM ignores it.
        // NOTE: nutUSpalding/nutUBlended on a non-SA model are NO LONGER an
        // error -- brae now honours the velocity-based nut wall function on any RAS model (see setNutWall + the
        // NutWall dispatch in device_simple_foam.cuh), matching OpenFOAM.
        auto guardWallFn = [&](const FieldData<scalar>& fd, const std::string& field) {
            auto isWF = [](const std::string& t) {
                return t == "nutkWallFunction" || t == "nutUSpaldingWallFunction" || t == "nutLowReWallFunction"
                    || t == "nutUBlendedWallFunction" || t == "atmNutkWallFunction" || t == "epsilonWallFunction" || t == "omegaWallFunction";
            };
            for (const auto& pb : fd.boundary)
            {
                if (!isWF(pb.type)) continue;
                for (const FvPatch* q : patchesResolvingTo(fd.boundary, pb, fvp))
                    if (q->type != "wall")
                        throw std::runtime_error(field + " boundaryField key '" + pb.name + "' (" + pb.type
                            + ") resolves to patch '" + q->name + "', which is type '" + q->type + "' (not"
                            " 'wall'). brae applies the near-wall model only on 'wall' patches, so"
                            " it would be SILENTLY inert (no wall shear / near-wall constraint). Retype the patch as 'wall'"
                            " in constant/polyMesh/boundary.");
            }
        };
        // Pick the nut wall function from the 0/nut wall-patch BC TYPE (OpenFOAM does this per-BC, not by model):
        // nutUSpalding -> Spalding, nutUBlended -> Blended, else nutk. Warn once on nutLowRe (mapped to nutk: identical
        // only on a resolved y+<yPlusLam mesh). SA keeps its Spalding path regardless (ctl.sa short-circuits below).
        // The nut wall-function family, in one place so the refusal below and guardWallFn cannot drift.
        auto isNutWallFn = [](const std::string& t) { return isNutWallFnType(t); };
        auto setNutWall = [&](const FieldData<scalar>& fd) {
            selectNutWall(fd, fvp, ctl.sa, ctl.modelName, ctl.nutWall, ctl.atmZ0, ctl.atmBoundNut);
        };
        GeometricField<scalar> k, eps, nut, ReThetat, gammaInt;   // ReThetat/gammaInt: kOmegaSSTLM transition
        if (ctl.les)   // pure LES Smagorinsky: ONLY nut (algebraic sub-grid viscosity); no k/epsilon/omega/nuTilda transport.
        {
            const FieldData<scalar> nutFD = readField<scalar>(fieldDir + "/nut");
            guardFrozen(nutFD, "nut");
            guardWallFn(nutFD, "nut");
            // The algebraic-LES device path honours EXACTLY ONE nut wall function -- nutUSpalding
            // (device_simple_foam.cu, ctl_.nutWall == NutWall::Spalding on the ctl_.les arm); every
            // other selection falls to plain cell-value extrapolation there, while setNutWall printed
            // the case's function as honoured -- the audit's finding #14: an LES case with
            // nutkWallFunction ran with no wall model at all and the log said otherwise. The k-based
            // family is not portable here either way -- algebraic LES carries no k field, and OpenFOAM
            // feeds those functions the model's own sgs k() estimate. BEFORE setNutWall, so the refused
            // run never prints a wall function as honoured.
            for (const auto& pb : nutFD.boundary)
            {
                if (isNutWallFn(pb.type) && pb.type != "nutUSpaldingWallFunction")
                    throw std::runtime_error(
                        "brae: 0/nut patch '" + pb.name + "' asks for " + pb.type + " on LESModel "
                        + ctl.modelName + ". The algebraic-LES path honours only "
                        "nutUSpaldingWallFunction (velocity-based); any other wall function would run "
                        "as plain extrapolation under the case's name. Refusing rather than running "
                        "without the wall model the case asked for.");
            }
            setNutWall(nutFD);   // honour a velocity-based nut wall function (nutUSpaldingWallFunction) if the case uses one
            nut = buildField<scalar>(nutFD, fvp, nC);
            nut.evaluateBoundary();
        }
        else if (ctl.sa)   // Spalart-Allmaras (one-equation): nuTilda -> the "k" slot, nut. BCs (freestream + fixedValue-0 wall) need no inlet calc.
        {
            k   = buildField<scalar>(readField<scalar>(fieldDir + "/nuTilda"), fvp, nC);
            k.evaluateBoundary();
            const FieldData<scalar> nutFD = readField<scalar>(fieldDir + "/nut");
            guardFrozen(nutFD, "nut");
            guardWallFn(nutFD, "nut");
            // The SA device path used to hard-force Spalding (`ctl_.sa || ...` in device_simple_foam.cu)
            // whatever 0/nut asked for -- the audit's finding #15. bump2D:SpalartAllmaras ships
            // nutLowReWallFunction, whose calcNut() returns Zero UNCONDITIONALLY on every model
            // (nutLowReWallFunctionFvPatchScalarField.C:38-42), and got a Newton uTau instead. The BC
            // selects now: Spalding and LowRe are honoured; the k-based family refuses -- OpenFOAM
            // feeds it SpalartAllmarasBase::k(), the derived estimate
            // cbrt(fv1)*nuTilda*sqrt(2/Cmu)*|symm(grad U)| (SpalartAllmarasBase.C:394-405), which brae
            // does not carry; and a concrete wall patch whose nut names NO wall function refuses too,
            // because the device writes the selected function on every wall face and has no
            // evaluate-the-BC path to spare it. A case where no wall-typed patch names any nut wall
            // function keeps the Spalding arithmetic every existing SA gate was measured on.
            std::string saWallFn;
            for (const auto& pb : nutFD.boundary)
            {
                if (isNutWallFn(pb.type))
                {
                    if (pb.type != "nutUSpaldingWallFunction" && pb.type != "nutLowReWallFunction")
                        throw std::runtime_error(
                            "brae: 0/nut patch '" + pb.name + "' asks for " + pb.type + " on "
                            "SpalartAllmaras. The SA path honours nutUSpaldingWallFunction (Newton "
                            "uTau) and nutLowReWallFunction (zero); OpenFOAM computes the k-based "
                            "family from the model's derived k() estimate, which brae does not carry. "
                            "Refusing rather than running Spalding under the case's name.");
                    if (!saWallFn.empty() && saWallFn != pb.type)
                        throw std::runtime_error(
                            "brae: 0/nut carries both '" + saWallFn + "' and '" + pb.type + "' on "
                            "SpalartAllmaras. This driver applies ONE wall function to every wall; "
                            "OpenFOAM dispatches per patch and honours both. Refusing rather than "
                            "silently picking one.");
                    saWallFn = pb.type;
                }
            }
            // The plain-BC-on-a-wall check runs PATCH-DRIVEN, resolving each wall patch's entry the way
            // buildField does -- an entry keyed `"wal.*"` used to be invisible to the exact-name compare
            // here (audit finding #16), so a regex-keyed plain fixedValue on the walls still got the
            // Spalding hard-force. A wall patch with NO entry at all is left to buildField's own fatal.
            for (const auto& q : fvp)
            {
                if (q.type != "wall") continue;
                const auto* e = findPatchEntry(nutFD.boundary, q);
                if (e && !isNutWallFn(e->type))
                    throw std::runtime_error(
                        "brae: wall patch '" + q.name + "' resolves its nut BC to key '" + e->name +
                        "' of type '" + e->type + "' (no wall function) on SpalartAllmaras. The SA "
                        "path writes its wall function on every wall face and would overwrite this "
                        "BC; OpenFOAM evaluates it. Refusing rather than substituting Spalding.");
            }
            ctl.nutWall = (saWallFn == "nutLowReWallFunction") ? NutWall::LowRe : NutWall::Spalding;
            std::printf("  nut wall function: %s (honoured on %s per the BC)\n",
                        saWallFn.empty() ? "nutUSpaldingWallFunction (no wall BC named one; SA default)"
                                         : saWallFn.c_str(),
                        ctl.modelName.c_str());
            nut = buildField<scalar>(nutFD, fvp, nC);
            nut.evaluateBoundary();
        }
        else if (ctl.turbulent)
        {
            const FieldData<scalar> kFD = readField<scalar>(fieldDir + "/k");
            guardFrozen(kFD, "k");
            const FieldData<scalar> sFD = readField<scalar>(fieldDir + "/" + secondName);
            guardFrozen(sFD, secondName);
            k   = buildField<scalar>(kFD, fvp, nC);
            k.evaluateBoundary();
            eps = buildField<scalar>(sFD, fvp, nC);
            eps.evaluateBoundary();
            const FieldData<scalar> nutFD = readField<scalar>(fieldDir + "/nut");
            guardFrozen(nutFD, "nut");
            guardWallFn(nutFD, "nut");
            guardWallFn(sFD, secondName);
            setNutWall(nutFD);   // honour the BC-specified velocity-based nut wall function (nutUSpalding/nutUBlended)
            // WHICH PATCHES THE TURBULENCE WALL FUNCTION ACTUALLY APPLIES TO. OpenFOAM's
            // epsilonWallFunction/omegaWallFunction are BC objects on this field: only a patch whose BC
            // is one of them gets an epsilon0/G0 override. brae built its wall set from the patch TYPE,
            // so a `wall`-typed patch carrying a plain BC was overridden too.
            //
            // turbulentFlatPlate's `topWall` is exactly that -- typed `wall`, but U slip, k and epsilon
            // zeroGradient, nut calculated, i.e. a slip far-field. brae pinned epsilon in its 545
            // adjacent cells to the wall-function value 1.556e-03 where OpenFOAM transports it to
            // 3.055e-02, so nut came out 1028x high, the freestream k never decayed from its 1.08e-03
            // inlet value (OpenFOAM's decays to 1.55e-04) and the run diverged at iteration 395.
            // findPatchEntry, NOT a name comparison: a boundaryField key may be an exact name, a GROUP or
            // a REGEX, and OpenFOAM resolves it in that order with the last match winning.
            // backwardFacingStep2D writes its wall BC as "(upperWall|lowerWall)", so comparing names
            // matched nothing, left this mask all zeros, and removed the wall function from the whole
            // case -- U went to 1.400e-01 against OpenFOAM and omega to 9.905e-01 before the suite
            // caught it.
            ctl.turbWallPatch.assign(fvp.size(), 0);
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                const auto* pb = findPatchEntry(sFD.boundary, fvp[pi]);
                if (pb && (pb->type == "epsilonWallFunction" || pb->type == "omegaWallFunction"))
                    ctl.turbWallPatch[pi] = 1;
            }
            {
                std::size_t nWF = 0, nWall = 0;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    if (fvp[pi].type != "wall") continue;
                    ++nWall;
                    if (ctl.turbWallPatch[pi]) ++nWF;
                }
                if (nWF != nWall)
                    std::printf("  turbulence wall function on %zu of %zu wall patch(es) -- the rest carry a "
                                "plain %s BC and are NOT overridden, as in OpenFOAM\n", nWF, nWall, secondName.c_str());
            }
            // epsilonWallFunction `lowReCorrection`, off the epsilon BC that names it. It was read by
            // NOTHING before: the entry sits inside a boundaryField patch dictionary, which the dict audit
            // does not track per key, so a case asking for it got the high-Re log-law epsilon and no
            // warning. turbulentFlatPlate:kEpsilon at y+ ~ 1 diverged at iteration 10 on that.
            for (const auto& pb : sFD.boundary)
            {
                if (pb.type == "epsilonWallFunction" && pb.epsLowRe)
                {
                    ctl.keCoeffs.epsLowRe = true;
                    std::printf("  epsilonWallFunction: lowReCorrection ON (resolved faces take "
                                "eps = 2*k*nu/y^2 and contribute no wall production)\n");
                    break;
                }
            }
            nut = buildField<scalar>(nutFD, fvp, nC);
            nut.evaluateBoundary();
            if (ctl.lm)   // kOmegaSSTLM transition fields
            {
                ReThetat = buildField<scalar>(readField<scalar>(fieldDir + "/ReThetat"), fvp, nC);
                ReThetat.evaluateBoundary();
                gammaInt = buildField<scalar>(readField<scalar>(fieldDir + "/gammaInt"), fvp, nC);
                gammaInt.evaluateBoundary();
                std::printf("  kOmegaSSTLM (Langtry-Menter transition): + ReThetat + gammaInt transport\n");
            }
            // turbulent-inlet BCs are NOT evaluated here. OF's rule is that a boundary condition updates
            // when ITS OWN equation is assembled: turbulentIntensityKineticEnergyInlet and
            // turbulentMixingLengthDissipationRateInlet are k/epsilon BCs, so their updateCoeffs() first
            // fires inside kEqn/epsEqn (kEpsilon.C:252,273) -- called from turbulence->correct(), the LAST
            // phase of the outer iteration. Until then the patch simply holds the `value` entry read from
            // the file, exactly like any other fixedValue-derived BC at construction.
            //
            // Evaluating them here instead made the FIRST momentum solve see an inlet k recomputed from
            // the set-up velocity. On squareBend that is k = 1.5*(0.05*523.087)^2 = 1026 against the
            // file's `uniform 1`, so the inlet `calculated` nut (Cmu*k_b^2/eps_b) came out 0.0877 instead
            // of 0.09*1/200 = 4.5e-04 -- and the inlet boundary diffusivity 157x OF's, which propagated
            // through Upred, HbyA, phiHbyA, phid and p.
            //
            // A3 fixed these inlets being FROZEN at set-up and never refreshed. The refresh is right; doing
            // it a phase early is not. Only the per-face MASKS are built here -- they say WHICH faces
            // correctTurbulence() must recompute, which is set-up information, not a value.
            masks = buildTurbulentInletMasks(kFD, sFD, fvp);
        }
    return { std::move(k), std::move(eps), std::move(nut), std::move(ReThetat), std::move(gammaInt), std::move(masks) };
}

}  // namespace brae
