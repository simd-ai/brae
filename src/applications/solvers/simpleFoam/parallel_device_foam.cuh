#pragma once
// brae distributed DEVICE simpleFoam: the multi-GPU app path.
//
//   mpirun -np N brae -case <dir> -parallel
//
// One rank == one partition == one GPU. Mirrors the control flow of the host brae_simpleFoam distributed path
// (simpleFoam.cu) -- read the global case on every rank, decompose in core with SCOTCH (no separate
// decomposePar step), iterate, gather on write -- but the SIMPLE loop itself runs entirely on the GPU via
// ParallelDeviceSimple: U/p/phi stay device-resident across iterations and only cross PCIe at write time.
//
// Laminar OR RAS (kEpsilon, kOmegaSST): the SIMPLE loop runs on the GPU via ParallelDeviceSimple, with
// turbulence enabled per the case's turbulenceProperties. Unsupported models/schemes are refused, not approximated.
//
// The single-GPU `brae` path is untouched: this is reached only when -parallel is passed, so a plain
// `brae -case <dir>` never initialises MPI (which also keeps the -cases fork orchestrator clear of MPI).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
#include "field_distribute.cuh"
#include "parallel_device_simple.cuh"
#include "parallel_device_ami.cuh"   // DistributedAMI: mapDistribute addressing + gather (distributed cyclicAMI, WIP)
#include "mrf_read.cuh"   // readMRFProperties / readCellZones / mrfCorrectBoundaryVelocity
#include "mrf_zone.cuh"   // MRFZone / buildMRFZone
#include "fv_options.cuh" // readFvOptions / FvOptionsData (per-cell momentum sources + porosity)
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <unordered_set>

namespace brae {

namespace detail {

// Gather a partitioned internal field to the global cell ordering. Partitions own disjoint cells, so a
// scatter-into-zeros + sum reduce reconstructs the global field on every rank (decomposePar's inverse).
inline std::vector<scalar> gatherToGlobal(const Partition& P, const std::vector<scalar>& loc)
{
    std::vector<scalar> g(P.globalNCells, 0.0);
    for (label c = 0; c < P.nCells(); ++c)
        g[P.Lm.cellProcAddr[c]] = loc[c];
    Pstream::allReduce(g.data(), static_cast<int>(g.size()), ReduceOp::Sum);
    return g;
}

inline std::vector<vector> gatherToGlobal(const Partition& P, const std::vector<vector>& loc)
{
    std::vector<scalar> g(3 * static_cast<std::size_t>(P.globalNCells), 0.0);
    for (label c = 0; c < P.nCells(); ++c)
    {
        const label gc = P.Lm.cellProcAddr[c];
        g[3 * gc]     = loc[c].x;
        g[3 * gc + 1] = loc[c].y;
        g[3 * gc + 2] = loc[c].z;
    }
    Pstream::allReduce(g.data(), static_cast<int>(g.size()), ReduceOp::Sum);
    std::vector<vector> out(P.globalNCells);
    for (label c = 0; c < P.globalNCells; ++c)
        out[c] = vector{g[3 * c], g[3 * c + 1], g[3 * c + 2]};
    return out;
}


} // namespace detail

// The `brae ... -parallel` entry point. Returns a process exit code.
inline int runParallelDeviceFoam(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const bool master = Pstream::master();
    const int  nproc  = Pstream::nProcs();
    const int  rank   = Pstream::myProcNo();
    int rc = 0;
    try
    {
        std::string caseDir = ".";
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "-case" && i + 1 < argc) caseDir = argv[++i];
            else if (a[0] != '-')             caseDir = a;
        }

        if (!Pstream::nvshmemActive() && nproc > 1)
            throw std::runtime_error(
                "brae -parallel: NVSHMEM is not active, so there is no GPU-to-GPU transport. Rebuild with "
                "-DBRAE_WITH_NVSHMEM=ON and check `nvidia-smi topo -p2p r` reports OK (P2P is required).");

        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
        const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
        const FoamDict turbProps   = readDict(caseDir + "/constant/turbulenceProperties");

        const std::string startStr = controlDict.wordOr("startTime", "0");
        const std::string fieldDir = caseDir + "/" + startStr;
        const int   endTime   = controlDict.intOr("endTime", 1000);
        const int   precision = controlDict.intOr("writePrecision", 16);
        const scalar nu       = transport.scalarOr("nu", 1e-5);

        // Turbulence: laminar, or RAS kEpsilon (the loop-integrated + validated model). kOmegaSST correct() is
        // validated standalone (test_gpu_parallel_komegasst) but not yet wired into the distributed loop, so it
        // is refused here -- honestly, not silently solved as something else.
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        const bool turbulent = (simType == "RAS");
        KEpsilonCoeffs keCoeffs;
        KOmegaSSTCoeffs sstCoeffs;
        SpalartAllmarasCoeffs saCoeffs;
        bool sst = false, lm = false, sa = false;
        if (turbulent)
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            if (model != "kEpsilon" && model != "kOmegaSST" && model != "realizableKE"
                && model != "kOmegaSSTLM" && model != "SpalartAllmaras")
                throw std::runtime_error(
                    "brae -parallel: RASModel '" + model + "' is not implemented on the multi-GPU device path "
                    "(kEpsilon, realizableKE, kOmegaSST, kOmegaSSTLM and SpalartAllmaras are). Use `mpirun -np N "
                    "brae_simpleFoam -case <dir>` (host) or `brae -case <dir>` (single GPU) for other models.");
            sst = (model == "kOmegaSST" || model == "kOmegaSSTLM");   // LM reuses the SST scaffolding (omega field)
            lm  = (model == "kOmegaSSTLM");
            sa  = (model == "SpalartAllmaras");
            if (sa)   // OF defaults are in the struct; read the SpalartAllmarasCoeffs overrides if present
            {
                const FoamDict* sac = ras ? ras->subDict("SpalartAllmarasCoeffs") : nullptr;
                if (sac)
                {
                    saCoeffs.sigmaNut = sac->scalarOr("sigmaNut", saCoeffs.sigmaNut);
                    saCoeffs.Cb1 = sac->scalarOr("Cb1", saCoeffs.Cb1);
                    saCoeffs.Cb2 = sac->scalarOr("Cb2", saCoeffs.Cb2);
                    saCoeffs.Cw2 = sac->scalarOr("Cw2", saCoeffs.Cw2);
                    saCoeffs.Cw3 = sac->scalarOr("Cw3", saCoeffs.Cw3);
                    saCoeffs.Cv1 = sac->scalarOr("Cv1", saCoeffs.Cv1);
                    saCoeffs.kappa = sac->scalarOr("kappa", saCoeffs.kappa);
                }
            }
            if (model == "realizableKE")
            {
                // realizableKE (OF RAS/realizableKE): variable-Cmu strain model, shares the k-eps loop but with
                // rCmu/magS + a strain-based eps reaction. Defaults differ from standard k-eps: A0=4, C2=1.9, sigmaEps=1.2.
                keCoeffs.realizable = true;
                keCoeffs.C2 = 1.9;
                keCoeffs.sigmaEps = 1.2;
                const FoamDict* rke = ras ? ras->subDict("realizableKECoeffs") : nullptr;
                if (rke)
                {
                    keCoeffs.A0 = rke->scalarOr("A0", keCoeffs.A0);
                    keCoeffs.C2 = rke->scalarOr("C2", keCoeffs.C2);
                    keCoeffs.sigmaK = rke->scalarOr("sigmak", keCoeffs.sigmaK);
                    keCoeffs.sigmaEps = rke->scalarOr("sigmaEps", keCoeffs.sigmaEps);
                    keCoeffs.kappa = rke->scalarOr("kappa", keCoeffs.kappa);
                    keCoeffs.E = rke->scalarOr("E", keCoeffs.E);
                }
            }
            const FoamDict* kec = ras ? ras->subDict("kEpsilonCoeffs") : nullptr;
            if (kec)
            {
                keCoeffs.Cmu = kec->scalarOr("Cmu", keCoeffs.Cmu);
                keCoeffs.C1 = kec->scalarOr("C1", keCoeffs.C1);
                keCoeffs.C2 = kec->scalarOr("C2", keCoeffs.C2);
                keCoeffs.C3 = kec->scalarOr("C3", keCoeffs.C3);
                keCoeffs.sigmaK = kec->scalarOr("sigmak", keCoeffs.sigmaK);
                keCoeffs.sigmaEps = kec->scalarOr("sigmaEps", keCoeffs.sigmaEps);
                keCoeffs.kappa = kec->scalarOr("kappa", keCoeffs.kappa);
                keCoeffs.E = kec->scalarOr("E", keCoeffs.E);
            }
            const FoamDict* ssc = ras ? ras->subDict("kOmegaSSTCoeffs") : nullptr;
            if (ssc)
            {
                sstCoeffs.alphaK1 = ssc->scalarOr("alphaK1", sstCoeffs.alphaK1);
                sstCoeffs.alphaK2 = ssc->scalarOr("alphaK2", sstCoeffs.alphaK2);
                sstCoeffs.alphaOmega1 = ssc->scalarOr("alphaOmega1", sstCoeffs.alphaOmega1);
                sstCoeffs.alphaOmega2 = ssc->scalarOr("alphaOmega2", sstCoeffs.alphaOmega2);
                sstCoeffs.gamma1 = ssc->scalarOr("gamma1", sstCoeffs.gamma1);
                sstCoeffs.gamma2 = ssc->scalarOr("gamma2", sstCoeffs.gamma2);
                sstCoeffs.beta1 = ssc->scalarOr("beta1", sstCoeffs.beta1);
                sstCoeffs.beta2 = ssc->scalarOr("beta2", sstCoeffs.beta2);
                sstCoeffs.betaStar = ssc->scalarOr("betaStar", sstCoeffs.betaStar);
                sstCoeffs.a1 = ssc->scalarOr("a1", sstCoeffs.a1);
                sstCoeffs.b1 = ssc->scalarOr("b1", sstCoeffs.b1);
                sstCoeffs.c1 = ssc->scalarOr("c1", sstCoeffs.c1);
            }
        }
        else if (simType != "laminar")
            throw std::runtime_error(
                "brae -parallel: simulationType '" + simType + "' is not supported (laminar or RAS).");

        // fvSchemes div(phi,U). The distributed path implements Gauss upwind, bounded, and linearUpwind (the
        // matrix is upwind either way; linearUpwind adds the deferred gradient correction, cut faces included).
        // Anything else -- limitedLinear, LUST, linearUpwindV, plain Gauss linear -- is a DIFFERENT
        // discretisation, and silently substituting upwind produces a converged, plausible, WRONG answer (on
        // the cavity that was a ~6% field difference). Refuse those, exactly as RAS is refused above.
        bool boundedDiv = false, linUpwind = false, lust = false, linUpwindV = false, limitedU = false, limitedVsch = false, lapCorrected = false;
        scalar limitedTwoByk = 2.0;   // limitedLinear[V] coefficient 2/max(k_,SMALL); default k_=1 -> 2
        {
            std::string divLine, lapLine;
            std::istringstream fsch(readFileExpanded(caseDir + "/system/fvSchemes"));
            std::string ln;
            bool inLap = false;   // the laplacian SCHEME ("... corrected;") is on the `default`/`laplacian(...)` line
            while (std::getline(fsch, ln))  // INSIDE the laplacianSchemes{} block, NOT the block-header line itself.
            {
                if (ln.find("div(phi,U)") != std::string::npos)      divLine = ln;
                if (ln.find("laplacianSchemes") != std::string::npos) { inLap = true; continue; }
                if (inLap)
                {
                    if (ln.find('}') != std::string::npos) inLap = false;
                    else lapLine += ln + " ";   // accumulate the block's scheme lines (default + any per-field)
                }
            }
            if (!divLine.empty())
            {
                boundedDiv = divLine.find("bounded") != std::string::npos;
                linUpwind  = divLine.find("linearUpwind") != std::string::npos;
                lust       = divLine.find("LUST") != std::string::npos;   // 0.75*linear + 0.25*linearUpwind (OF LUST.H)
                // NB "linearUpwind" has a capital U, so it does NOT contain the substring "upwind" -- the two
                // must be tested separately or a linearUpwind case reads as "no upwind scheme at all".
                linUpwindV = divLine.find("linearUpwindV") != std::string::npos;
                // limitedLinear (vector) = OF NVDTVD + magSqr: reduce U to s=|U|^2, ONE limiter/face from grad(s).
                // limitedLinearV = NVDVTVDV + null: ONE limiter/face from the VECTOR U directly. Both build the
                // limited convection IMPLICITLY into the momentum matrix (a deferred source under-converges). NB
                // "limitedLinear" is a prefix of "limitedLinearV", so test V FIRST.
                limitedVsch        = divLine.find("limitedLinearV") != std::string::npos;
                const bool limited = !limitedVsch && (divLine.find("limitedLinear") != std::string::npos);
                const bool upwindFamily = (divLine.find("upwind") != std::string::npos) || linUpwind || lust || limited || limitedVsch;
                if (limited || limitedVsch)   // parse the coefficient k_ in "limitedLinear[V] k_" -> twoByk = 2/max(k_,SMALL)
                {
                    linUpwind = false; lust = false;   // limitedLinear[V] is its own family
                    limitedU = limited;                // magSqr path
                    size_t q = divLine.find("limitedLinear") + std::string("limitedLinear").size();
                    if (q < divLine.size() && divLine[q] == 'V') ++q;   // skip the trailing 'V' of limitedLinearV
                    std::istringstream cs(divLine.substr(q));
                    scalar kv = 1.0;
                    limitedTwoByk = (cs >> kv) ? (2.0 / std::max(kv, static_cast<scalar>(1e-30))) : 2.0;
                }
                if (!upwindFamily)
                    throw std::runtime_error(
                        "brae -parallel: div(phi,U) scheme '" + divLine + "' is not implemented on the "
                        "multi-GPU device path (it supports 'Gauss upwind', 'bounded Gauss upwind', "
                        "'[bounded] Gauss linearUpwind[V] grad(U)', '[bounded] Gauss LUST grad(U)' and "
                        "'[bounded] Gauss limitedLinear[V] k'). Substituting a scheme would converge to a "
                        "DIFFERENT answer than fvSchemes asks for, so this is refused rather than solved "
                        "wrongly. Use `brae -case <dir>` (single GPU) for the full scheme set.");
            }
            // OF laplacian "corrected"/"limited" = nonOrthDeltaCoeffs implicit + an explicit corrVec.grad
            // correction. Checked against the mesh below, once the Partition exists.
            lapCorrected = (lapLine.find("corrected") != std::string::npos)
                        || (lapLine.find("limited")   != std::string::npos);
        }

        const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
        const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
        const FoamDict* fld = rf ? rf->subDict("fields") : nullptr;
        const scalar relaxU = eqs ? eqs->scalarOr("U", 1.0) : 1.0;
        const scalar relaxP = fld ? fld->scalarOr("p", 1.0) : 1.0;
        const scalar relaxK   = eqs ? eqs->scalarOr("k", 1.0) : 1.0;
        const scalar relaxEps = eqs ? eqs->scalarOr("epsilon", 1.0) : 1.0;
        // SpalartAllmaras relaxes ONE field (nuTilda) -- read it explicitly (there is no "k"); default 0.7 (a safe SA
        // under-relaxation) rather than 1.0, else the unrelaxed nuTilda destabilises after ~200 iters.
        const scalar relaxNuTilda = eqs ? eqs->scalarOr("nuTilda", 0.7) : 0.7;

        const FoamDict* solvers = fvSolution.subDict("solvers");
        auto solverTol = [&](const std::string& f, scalar def)
        {
            const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
            return s ? s->scalarOr("tolerance", def) : def;
        };
        const scalar tolP = solverTol("p", 1e-6), tolU = solverTol("U", 1e-8);
        // relTol: SIMPLE only needs a loose per-step solve, so OF stops each equation at a RELATIVE reduction.
        // Same read as the single-GPU path (gpuSimpleFoam.cu solverRelTol); default 0 = solve to absolute tol.
        auto solverRelTol = [&](const std::string& f)
        {
            const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
            return s ? s->scalarOr("relTol", 0.0) : 0.0;
        };
        const scalar relTolP = solverRelTol("p"), relTolU = solverRelTol("U");
        // Turbulence transport relTol: OF solves k and (epsilon|omega) loosely per SIMPLE step. One knob covers
        // both here, so take the TIGHTER of the two (fmin) -- never solve a field looser than its case asks.
        const scalar relTolKE = sa ? solverRelTol("nuTilda")
                                   : std::fmin(solverRelTol("k"), solverRelTol(sst ? "omega" : "epsilon"));

        const FoamDict* simple = fvSolution.subDict("SIMPLE");
        // nNonOrthogonalCorrectors: OF re-solves the pEqn (n+1) times, recomputing the explicit non-orth
        // correction from the UPDATED p each pass. The distributed path does ONE pass (pass 0, the entry
        // grad(p)), so a case asking for more would silently get a different pressure. Refuse.
        if (simple && simple->intOr("nNonOrthogonalCorrectors", 0) > 0)
            throw std::runtime_error(
                "brae -parallel: nNonOrthogonalCorrectors > 0 is not implemented on the multi-GPU device path "
                "(it does a single pEqn pass). Set it to 0, or use `brae -case <dir>` (single GPU).");
        // SIMPLEC (SIMPLE.consistent): the rAtU consistency correction that lets a case run at high pressure
        // relaxation (pitzDaily etc. use relaxP≈0.9, only stable WITH this). Without it the distributed path would
        // run plain SIMPLE at that relaxation and diverge -- so it must honour the flag, not ignore it.
        const std::string consWord = simple ? simple->wordOr("consistent", "no") : "no";
        const bool consistent = (consWord == "yes" || consWord == "true" || consWord == "on" || consWord == "1");
        const FoamDict* resCtl = simple ? simple->subDict("residualControl") : nullptr;
        const bool   hasRC = (resCtl != nullptr);
        const scalar rcP = resCtl ? resCtl->scalarOr("p", -1) : -1;
        const scalar rcU = resCtl ? resCtl->scalarOr("U", -1) : -1;

        // Startup phase timing (BRAE_TIME_STARTUP=1): the ~17-20s fixed cost dominates the 2-GPU wall on a short
        // run, so break it down before optimising. cudaDeviceSynchronize before each lap captures GPU work; timed
        // on master (a rough per-phase breakdown, enough to find the dominant contributor).
        const bool timeStartup = std::getenv("BRAE_TIME_STARTUP") != nullptr;
        auto tclk = std::chrono::high_resolution_clock::now();
        auto lap = [&](const char* name)
        {
            if (!timeStartup) return;
            cudaDeviceSynchronize();
            auto now = std::chrono::high_resolution_clock::now();
            if (master) std::printf("[startup] %-14s %8.1f ms\n", name,
                                    std::chrono::duration<double, std::milli>(now - tclk).count());
            tclk = now;
        };

        PrimitiveMesh gm;
        gm.read(caseDir + "/constant/polyMesh");
        const label nC = gm.nCells();
        lap("meshRead");

        // cyclicAMI: the distributed weaving (buildDistributedAMI -> solver.setAMI) is VALIDATED -- np=1 reproduces the
        // single-GPU BIT-FOR-BIT (U to 1e-13, p to 1e-12 after removing the singular-pressure level; both converge in
        // the same 148 iterations on tc_wedge_ami, rotational rotor-stator), and np=2 (cross-rank NVSHMEM gather)
        // matches to the residualControl tolerance (~1e-5). So cyclicAMI is coupled on the multi-GPU path by default.
        // Plain conformal `cyclic` (no interpolation) is still not implemented; single-GPU `brae -case` handles it.
        bool hasAMI = false;
        for (const auto& pinfo : gm.patches())
        {
            if (pinfo.type == "cyclic")
                throw std::runtime_error(
                    "brae -parallel: patch '" + pinfo.name + "' is type 'cyclic', which is not coupled on the "
                    "multi-GPU device path (the interface would be left open and the answer wrong). Use "
                    "`brae -case <dir>` (single GPU) for plain cyclic cases.");
            if (pinfo.type == "cyclicAMI") hasAMI = true;
        }

        // decompose in core: SCOTCH on master, broadcast so every rank agrees on the same partitioning
        std::vector<label> cellToPart(nC, 0);
        if (master) cellToPart = scotchDecomposeCached(gm, nproc, caseDir + "/constant/polyMesh");   // ~1s -> cached
        Pstream::broadcast(cellToPart.data(), nC, 0);
        const Partition P(gm, cellToPart, rank);
        lap("decompose");

        // The non-orthogonal correction is NOT implemented at processor faces. On an ORTHOGONAL mesh it is
        // identically zero, so "corrected" is safe there and 15/113 cases say "orthogonal" outright. On a
        // non-orthogonal mesh, silently dropping it solves a DIFFERENT equation than fvSchemes asks for --
        // the same failure mode as substituting a div scheme, and 96/113 cases ask for "corrected". Refuse.
        // 0.1 deg: below that the correction is numerical noise; above it, it is physics we are not doing.
        // The non-orthogonal correction IS implemented on the distributed path: nonOrthDeltaCoeffs implicit at cut
        // faces (computeProcNonOrth), the explicit corrVec.grad correction for the momentum, and for the pEqn BOTH
        // the source and the face-flux correction (or continuity breaks). Gated by test_gpu_parallel_duct on a
        // sheared 26.57 deg mesh; every piece teeth-proven. APPLY it when the scheme is "corrected" AND the mesh is
        // actually non-orthogonal (> 0.1 deg -- below that it is numerical noise and the extra work is a no-op).
        bool useNonOrth = false;
        if (lapCorrected)
        {
            const scalar nonOrtho = maxNonOrthogonality(P);
            useNonOrth = (nonOrtho > 0.1);
            if (master)
                std::printf("  mesh max non-orthogonality %.4g deg -> non-orth 'corrected' term %s\n",
                            nonOrtho, useNonOrth ? "ON" : "negligible (off)");
        }

        const FieldData<vector> Ufd = readField<vector>(fieldDir + "/U");
        const FieldData<scalar> pfd = readField<scalar>(fieldDir + "/p");
        GeometricField<vector> U0 = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        U0.evaluateBoundary();
        GeometricField<scalar> p0 = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        p0.evaluateBoundary();
        lap("distribute");

        // MRF rotating cell zone (constant/MRFProperties + polyMesh/cellZones), if present. Read the GLOBAL zone
        // and map it to THIS rank's local cells (cellProcAddr is local->global); build a per-rank MRFZone. The
        // in-zone velocity-fixing walls take U = Omega x (Cf-origin), then setMRF (after the solver is built) makes
        // the resident flux relative and wires the Coriolis source. Done before the solver ctor so phi0 = flux(U0)
        // already includes the rotating walls, exactly as single-GPU gpuSimpleFoam.
        const MRFConfig mrfCfg = readMRFProperties(caseDir + "/constant");
        MRFZone mrfZone;
        if (mrfCfg.active)
        {
            const auto zones = readCellZones(caseDir + "/constant/polyMesh");
            const auto it = zones.find(mrfCfg.cellZone);
            if (it == zones.end())
                throw std::runtime_error("brae -parallel: MRF cellZone '" + mrfCfg.cellZone +
                                         "' not found in constant/polyMesh/cellZones");
            std::unordered_set<label> gset(it->second.begin(), it->second.end());
            std::vector<label> localZoneCells;
            for (label c = 0; c < P.Lm.mesh.nCells(); ++c)
                if (gset.count(P.Lm.cellProcAddr[c])) localZoneCells.push_back(c);
            mrfZone = buildMRFZone(P.Lm.mesh, localZoneCells, mrfCfg.axis, mrfCfg.omega, mrfCfg.origin);
            mrfCorrectBoundaryVelocity(U0, mrfZone, P.lg, P.lp, mrfCfg.nonRotatingPatches);   // in-zone walls -> Omega x r
            const std::size_t nLoc = localZoneCells.size();
            const std::size_t nGlob = Pstream::allReduce(static_cast<scalar>(nLoc), ReduceOp::Sum);
            if (master)
                std::printf("  MRF: cellZone=%s omega=%.4g axis=(%.2g %.2g %.2g) origin=(%.2g %.2g %.2g) zoneCells=%zu\n",
                            mrfCfg.cellZone.c_str(), mrfCfg.omega, mrfCfg.axis.x, mrfCfg.axis.y, mrfCfg.axis.z,
                            mrfCfg.origin.x, mrfCfg.origin.y, mrfCfg.origin.z, static_cast<std::size_t>(nGlob));
        }

        // fvOptions (system/ or constant/fvOptions), if present. Per-cell momentum sources distribute like MRF
        // (global cell arrays -> this rank's local via cellProcAddr): vectorSemiImplicitSource(U) (Su/Sp) and
        // explicitPorositySource (DarcyForchheimer). Global/reduction sources (meanVelocityForce, actuationDisk),
        // scalar sources and rotorDisk are NOT yet distributed -> refused rather than solved wrong.
        FvOptionsData localFvo;
        if (std::filesystem::exists(caseDir + "/system/fvOptions") ||
            std::filesystem::exists(caseDir + "/constant/fvOptions"))
        {
            const auto fvoZones = readCellZones(caseDir + "/constant/polyMesh");
            FvGeometry gg;
            gg.build(gm);
            const FvOptionsData fvo = readFvOptions(caseDir, fvoZones, gg.V(), nC, gg.C());
            if (!fvo.empty())
            {
                // Supported so far: vectorSemiImplicitSource(U) (per-cell Su/Sp, validated). Everything else --
                // explicitPorositySource, meanVelocityForce, actuationDisk, scalar sources, rotorDisk -- is refused
                // until it has a distributed validation case (no small/no-op test would prove correctness).
                if (fvo.porActive || fvo.mvfActive || fvo.adActive || fvo.limUActive || fvo.vdcActive
                    || !fvo.scaSu.empty() || !fvo.scaSp.empty() || fvo.rotor.active)
                    throw std::runtime_error(
                        "brae -parallel: this fvOptions type is not yet on the multi-GPU device path (supported: "
                        "vectorSemiImplicitSource on U). Use `brae -case <dir>` (single GPU) for the full set.");
                localFvo.count = fvo.count;
                const label nLoc = P.nCells();
                if (fvo.hasMomentum)
                {
                    localFvo.hasMomentum = true;
                    for (int c = 0; c < 3; ++c)
                    {
                        localFvo.momSu[c].resize(nLoc);
                        for (label lc = 0; lc < nLoc; ++lc)
                            localFvo.momSu[c][lc] = fvo.momSu[c][P.Lm.cellProcAddr[lc]];
                    }
                    if (!fvo.momSp.empty())
                    {
                        localFvo.momSp.resize(nLoc);
                        for (label lc = 0; lc < nLoc; ++lc)
                            localFvo.momSp[lc] = fvo.momSp[P.Lm.cellProcAddr[lc]];
                    }
                }
                if (master)
                    std::printf("  fvOptions: %d source(s) vectorSemiImplicitSource(U)\n", fvo.count);
            }
        }

        // turbulence start fields: kEps/SST read k + eps|omega + nut (+ ReThetat/gammaInt for LM); SpalartAllmaras
        // is one-equation -> nuTilda + nut only (no k/eps).
        FieldData<scalar> kfd, efd, nfd, rfd, gfd, ntfd;
        GeometricField<scalar> k0, eps0, nut0, ReThetat0, gammaInt0, nuTilda0;
        if (turbulent)
        {
            nfd = readField<scalar>(fieldDir + "/nut");
            nut0 = distributeField<scalar>(nfd, gm.patches(), P.Lm, P.lp, P.procW, rank); nut0.evaluateBoundary();
            if (sa)
            {
                ntfd = readField<scalar>(fieldDir + "/nuTilda");
                nuTilda0 = distributeField<scalar>(ntfd, gm.patches(), P.Lm, P.lp, P.procW, rank); nuTilda0.evaluateBoundary();
            }
            else
            {
                kfd = readField<scalar>(fieldDir + "/k");
                efd = readField<scalar>(fieldDir + "/" + (sst ? "omega" : "epsilon"));
                k0   = distributeField<scalar>(kfd, gm.patches(), P.Lm, P.lp, P.procW, rank); k0.evaluateBoundary();
                eps0 = distributeField<scalar>(efd, gm.patches(), P.Lm, P.lp, P.procW, rank); eps0.evaluateBoundary();
                if (lm)
                {
                    rfd = readField<scalar>(fieldDir + "/ReThetat");
                    gfd = readField<scalar>(fieldDir + "/gammaInt");
                    ReThetat0 = distributeField<scalar>(rfd, gm.patches(), P.Lm, P.lp, P.procW, rank); ReThetat0.evaluateBoundary();
                    gammaInt0 = distributeField<scalar>(gfd, gm.patches(), P.Lm, P.lp, P.procW, rank); gammaInt0.evaluateBoundary();
                }
            }
        }

        if (master)
        {
            std::printf("brae (device, distributed) | case=%s np=%d | %s%s | nu=%.3g | %ld cells\n",
                        caseDir.c_str(), nproc, simType.c_str(), turbulent ? (sst ? " (kOmegaSST)" : " (kEpsilon)") : "", nu, (long)nC);
            std::printf("  relax U=%.2g p=%.2g%s | tol p=%.1g U=%.1g | endTime=%d | residualControl=%s | div(phi,U)=%s\n",
                        relaxU, relaxP, turbulent ? (" k=" + std::to_string(relaxK)).c_str() : "",
                        tolP, tolU, endTime, hasRC ? "on" : "off",
                        (std::string(boundedDiv ? "bounded " : "") + (lust ? "Gauss LUST" : limitedVsch ? "Gauss limitedLinearV" : limitedU ? "Gauss limitedLinear" : linUpwind ? (linUpwindV ? "Gauss linearUpwindV" : "Gauss linearUpwind") : "Gauss upwind")).c_str());
        }

        std::vector<vector> Ug;
        std::vector<scalar> pg, kg, eg, ng;
        int nIter = 0;
        {   // scope: the solver's symmetric-heap buffers must be freed BEFORE Pstream::finalize
            ParallelDeviceSimple solver(P, U0, p0, nu, relaxU, relaxP, tolU, tolP, 2000, boundedDiv, linUpwind, lust,
                                        /*linUpwindV*/ linUpwindV, /*nonOrth*/ useNonOrth, /*consistent*/ consistent,
                                        /*limitedU*/ limitedU, /*limitedTwoByk*/ limitedTwoByk, /*limitedV*/ limitedVsch,
                                        /*amgCacheDir*/ caseDir + "/constant/polyMesh",
                                        /*relTolU*/ relTolU, /*relTolP*/ relTolP, /*relTolKE*/ relTolKE);
            if (turbulent && sa)  solver.enableTurbulenceSA(U0, nuTilda0, nut0, saCoeffs, relaxNuTilda);
            else if (turbulent && lm)  solver.enableTurbulenceSSTLM(U0, k0, eps0, nut0, ReThetat0, gammaInt0, sstCoeffs, relaxK, relaxEps);
            else if (turbulent && sst) solver.enableTurbulenceSST(U0, k0, eps0, nut0, sstCoeffs, relaxK, relaxEps);
            else if (turbulent)   solver.enableTurbulence(U0, k0, eps0, nut0, keCoeffs, relaxK, relaxEps);
            if (mrfCfg.active) solver.setMRF(mrfZone, mrfCfg.nonRotatingPatches);   // Coriolis source + relative flux
            if (!localFvo.empty()) solver.setFvOptions(localFvo);                   // per-cell momentum Su/Sp + porosity
            {   // no fixedValue-type p patch -> singular pressure (needs interface regularisation when hasAMI, level
                // pinned post-hoc below). Same scan as the post-solve pin at the bottom of this function.
                bool needRef = true;
                for (const PatchFieldData<scalar>& b : pfd.boundary)
                    if (b.type == "fixedValue" || b.type == "totalPressure" || b.type == "outletInlet"
                        || b.type == "fixedMean" || b.type == "prghPressure")
                        needRef = false;
                solver.setPressureSingular(needRef);
            }
            if (const char* sd = std::getenv("BRAE_STAGE_DUMP"))
                solver.setStageDump(sd, std::getenv("BRAE_STAGE_DUMP_ITER") ? std::atoi(std::getenv("BRAE_STAGE_DUMP_ITER")) : 1);
            if (hasAMI)   // cyclicAMI: build the GLOBAL AMI (same on every rank) + this rank's DistributedAMI
            {
                FvGeometry gg;
                gg.build(gm);
                const std::vector<FvPatch> gfvp = buildPatches(gm, gg);
                const std::vector<AMIInterface> gAMIs = buildAMIInterfaces(gm, gg, gfvp);
                solver.setAMI(gAMIs, cellToPart);
                if (master) std::printf("  cyclicAMI: %zu interface(s) on the distributed path\n", gAMIs.size());
            }
            lap("buildSolver");   // buildDeviceMesh + buildAMG + NVSHMEM halo/symmetric-heap alloc

            auto ok = [](scalar res, scalar ctl) { return ctl < 0 || res < ctl; };
            bool converged = false;
            int iter = 0;
            for (iter = 1; iter <= endTime && !converged; ++iter)
            {
                const ParStepResidual r = solver.step();
                if (iter == 1) lap("firstStep");   // first step pays graph capture + first-touch allocation
                if (master && (iter == 1 || iter % 20 == 0))
                    std::printf("  iter %4d:  Ux %.3e  p %.3e  | avg krylov/iter %ld\n",
                                iter, r.Ux, r.p, solver.krylovIters() / iter);
                // residualControl on U/p (the turbulence fields co-converge; step() returns only U/p residuals).
                converged = hasRC && ok(r.p, rcP) && ok(r.Ux, rcU);
            }
            nIter = converged ? iter - 1 : endTime;
            if (master)
                std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                                      : "SIMPLE reached endTime (%d iterations)\n", nIter);

            Ug = detail::gatherToGlobal(P, solver.U());   // D2H once, at write time
            pg = detail::gatherToGlobal(P, solver.p());
            if (turbulent)
            {
                kg = detail::gatherToGlobal(P, solver.kLocal());
                eg = detail::gatherToGlobal(P, solver.epsLocal());
                ng = detail::gatherToGlobal(P, solver.nutLocal());
            }
        }

        // pRefCell/pRefValue. A CLOSED domain (no fixedValue p patch) has an all-Neumann pressure equation,
        // which is singular: p is determined only up to an additive constant. OF pins it with
        // pEqn.setReference(); we instead solve the consistent singular system and pin the LEVEL here -- the
        // field is identical either way (verified on the 3D cavity: after removing the offset the two agree to
        // 2.8e-07), but without this the written p carries an arbitrary constant and will not match OF.
        {
            bool needRef = true;
            for (const PatchFieldData<scalar>& b : pfd.boundary)
                if (b.type == "fixedValue" || b.type == "totalPressure" || b.type == "outletInlet"
                    || b.type == "fixedMean" || b.type == "prghPressure")
                    needRef = false;
            if (needRef && !pg.empty())
            {
                const label  refCell = static_cast<label>(simple ? simple->intOr("pRefCell", 0) : 0);
                const scalar refVal  = simple ? simple->scalarOr("pRefValue", 0.0) : 0.0;
                if (refCell >= 0 && refCell < static_cast<label>(pg.size()))
                {
                    const scalar shift = refVal - pg[refCell];
                    for (scalar& v : pg) v += shift;
                    if (master)
                        std::printf("  closed domain: p pinned to pRefValue=%g at pRefCell=%ld (shift %.3e)\n",
                                    refVal, (long)refCell, shift);
                }
            }
        }

        if (master)
        {
            const std::string outDir = caseDir + "/" + std::to_string(nIter);
            std::filesystem::create_directories(outDir);
            const std::string src = fieldDir + "/";
            writeVolField(src + "U", outDir + "/U", Ug, gm.patches(), precision);
            writeVolField(src + "p", outDir + "/p", pg, gm.patches(), precision);
            if (turbulent)
            {
                writeVolField(src + "k",       outDir + "/k",       kg, gm.patches(), precision);
                writeVolField(src + (sst ? "omega" : "epsilon"), outDir + "/" + (sst ? "omega" : "epsilon"), eg, gm.patches(), precision);
                writeVolField(src + "nut",     outDir + "/nut",     ng, gm.patches(), precision);
            }
            std::printf("wrote %s\n", outDir.c_str());
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "brae -parallel (rank %d): %s\n", rank, e.what());
        rc = 1;
    }
    Pstream::finalize();
    return rc;
}

} // namespace brae
