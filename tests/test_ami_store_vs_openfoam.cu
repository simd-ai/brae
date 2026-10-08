// The legacy AMI builder's STORE TEST and row normalisation against REAL OpenFOAM's own AMI, on a mesh blockMesh
// built whose two sides are offset by a sliver (tests/ami_store_vs_openfoam.sh and ami_rows_vs_openfoam.sh stage
// it and say what each case is for). The oracle is a dump of cyclicAMIPolyPatch::AMI() written by a coded function
// object: for the owner's faces ("src") and the neighbour's ("tgt"), each face's mask (a cyclicACMI's; the
// sum of weights otherwise), its partners and the weights OpenFOAM applies.
//   test_ami_store_vs_openfoam <case> <dump> <owner patch> <neighbour patch> <acmi|ami>
// brae builds each direction on its own; a stencil entry holds the partner face's CELL, and every face of these
// fixtures has a cell of its own, so the partner face is found from the cell.
// THREE CHECKS:
//   1. the partner sets of every face, both directions, are OpenFOAM's;
//   2. where OpenFOAM leaves a face uncovered brae's mask is exactly 0, and elsewhere the weights are within
//      B_WEIGHT of OpenFOAM's and the mask within B_MASK;
//   3. CONTROLS, read at every build so this one process runs them, each of which has to miss check 1 or 2:
//      the 1e-14 threshold over the direction's own face (both fixtures); OpenFOAM's threshold over the
//      direction's own face in place of the owner's (the cyclicACMI fixture, whose two sides' faces differ by a
//      factor of two -- on the cyclicAMI one they are equal to 1e-6 and the choice cannot show); and rows left
//      area-normalised (the cyclicAMI fixture; a cyclicACMI's are re-normalised either way).
// B_WEIGHT is 1e-12 on the plain cyclicAMI: every weight of that fixture is exactly 1 on both sides (measured
// gap 0), and its sums of weights are OpenFOAM's to 4.1e-13 (bound 5e-12).
// B_MASK IS NOT ROUNDING and is not claimed to be. OpenFOAM's intersection is a sum over triangle pairs with a
// coplanar snap (faceAreaIntersect.C:69); brae's is a polygon clip. MEASURED 2026-10-06 on the cyclicACMI
// fixture (offset 0.25 - 2.25e-7): OpenFOAM's area of the pair that overlaps WHOLLY is short of the face by
// 4.5e-7 of it, brae's is the face; the widest mask gap is 4.5e-7 and the widest weight gap 3.4e-7. The
// bounds there are a decade above (5e-6), which still tells a kept sliver from a dropped one only through
// checks 1 and the exact zero -- said here so the number is not read as agreement.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ami_interface.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace brae;

namespace {

struct Side
{
    std::vector<scalar> mask;
    std::vector<scalar> sum;
    std::vector<std::map<label, scalar>> w;
};

// "owner <0|1>", then "src N" and N lines "i mask sum n  a0 w0  a1 w1 ...", then the same for "tgt"
bool readDump(
    const std::string& path,
    int& owner,
    Side& src,
    Side& tgt)
{
    std::ifstream in(path);
    if (!in) return false;
    std::string tag;
    in >> tag >> owner;
    for (Side* s : {&src, &tgt})
    {
        std::size_t n = 0;
        in >> tag >> n;
        s->mask.resize(n);
        s->sum.resize(n);
        s->w.resize(n);
        for (std::size_t k = 0; k < n; ++k)
        {
            std::size_t i = 0;
            std::size_t m = 0;
            double mask = 0;
            double sum = 0;
            in >> i >> mask >> sum >> m;
            s->mask[i] = mask;
            s->sum[i] = sum;
            for (std::size_t j = 0; j < m; ++j)
            {
                long a = 0;
                double w = 0;
                in >> a >> w;
                s->w[i][static_cast<label>(a)] = w;
            }
        }
    }
    return static_cast<bool>(in);
}

struct Compared
{
    long faces = 0;
    long entriesBrae = 0;
    long entriesOf = 0;
    long setsDiffer = 0;
    long uncoveredOf = 0;
    long uncoveredNotZero = 0;
    scalar worstWeight = 0;
    scalar worstMask = 0;
};

// one direction of brae's against one side of the dump
void compare(
    const AMIInterface& a,
    const FvPatch& nbr,
    const Side& of,
    bool acmi,
    Compared& c)
{
    std::map<label, label> faceOfCell;
    for (label j = 0; j < nbr.size; ++j)
    {
        faceOfCell[nbr.faceCells[static_cast<std::size_t>(j)]] = j;
    }
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
    {
        ++c.faces;
        std::map<label, scalar> mine;
        for (label k = a.srcOffset[i]; k < a.srcOffset[i + 1]; ++k)
        {
            mine[faceOfCell.at(a.nbrCell[static_cast<std::size_t>(k)])] = a.weight[static_cast<std::size_t>(k)];
        }
        c.entriesBrae += static_cast<long>(mine.size());
        c.entriesOf += static_cast<long>(of.w[i].size());
        std::set<label> sm;
        std::set<label> so;
        for (const auto& e : mine)
        {
            sm.insert(e.first);
        }
        for (const auto& e : of.w[i])
        {
            so.insert(e.first);
        }
        // a cyclicACMI's mask is the clamped coverage; a cyclicAMI's dump carries its sum of weights there
        const scalar braeMask = acmi ? a.mask[i] : a.coverage[i];
        if (of.w[i].empty())
        {
            ++c.uncoveredOf;
            if (braeMask != scalar(0) || !mine.empty())
            {
                ++c.uncoveredNotZero;
            }
        }
        if (sm != so)
        {
            ++c.setsDiffer;
            continue;
        }
        for (const auto& e : mine)
        {
            c.worstWeight = std::fmax(c.worstWeight, std::fabs(e.second - of.w[i].at(e.first)));
        }
        c.worstMask = std::fmax(c.worstMask, std::fabs(braeMask - of.mask[i]));
    }
}

Compared build(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::string& ownerName,
    const std::string& nbrName,
    const Side& src,
    const Side& tgt,
    bool acmi)
{
    Compared c;
    const std::vector<AMIInterface> ai = buildAMIInterfaces(m, g, fvp);
    for (const AMIInterface& a : ai)
    {
        const FvPatch& own = fvp[static_cast<std::size_t>(a.patch)];
        const FvPatch& nbr = fvp[static_cast<std::size_t>(a.nbrPatch)];
        if (own.name == ownerName && nbr.name == nbrName)
        {
            compare(a, nbr, src, acmi, c);
        }
        if (own.name == nbrName && nbr.name == ownerName)
        {
            compare(a, nbr, tgt, acmi, c);
        }
    }
    return c;
}

}   // namespace

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        std::printf("usage: %s <case> <dump> <owner patch> <neighbour patch> <acmi|ami>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string ownerName = argv[3];
    const std::string nbrName = argv[4];
    const bool acmi = std::string(argv[5]) == "acmi";
    const scalar B_WEIGHT = acmi ? scalar(5e-6) : scalar(1e-12);
    const scalar B_MASK = acmi ? scalar(5e-6) : scalar(5e-12);
    int owner = 0;
    Side src;
    Side tgt;
    if (!readDump(argv[2], owner, src, tgt) || owner != 1)
    {
        std::printf("FAIL: the dump %s is not OpenFOAM's AMI of the owner patch\n", argv[2]);
        return 1;
    }
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    int fails = 0;

    const Compared c = build(m, g, fvp, ownerName, nbrName, src, tgt, acmi);
    const bool sized = c.faces == static_cast<long>(src.w.size() + tgt.w.size());
    std::printf("  %-5s the partner sets of %ld faces (%ld owner, %ld neighbour) are OpenFOAM's: %ld differ; "
                "entries brae %ld, OpenFOAM %ld\n",
                (sized && c.setsDiffer == 0 && c.entriesOf > 0) ? "ok:" : "FAIL:", c.faces,
                static_cast<long>(src.w.size()), static_cast<long>(tgt.w.size()), c.setsDiffer, c.entriesBrae,
                c.entriesOf);
    fails += (sized && c.setsDiffer == 0 && c.entriesOf > 0) ? 0 : 1;
    const bool values = c.uncoveredNotZero == 0 && c.worstWeight <= B_WEIGHT && c.worstMask <= B_MASK;
    std::printf("  %-5s %ld faces OpenFOAM leaves uncovered, %ld of them not exactly 0 in brae; elsewhere the "
                "widest weight gap %.3e (bound %.0e), the widest %s gap %.3e (bound %.0e)\n",
                values ? "ok:" : "FAIL:", c.uncoveredOf, c.uncoveredNotZero, c.worstWeight, B_WEIGHT,
                acmi ? "mask" : "sum-of-weights", c.worstMask, B_MASK);
    fails += values ? 0 : 1;

    // the CONTROLS: each one alone, then cleared
    const char* controls[2] = {"BRAE_CONTROL_AMI_OVERLAP_1E14",
                               acmi ? "BRAE_CONTROL_AMI_OWN_FACE_AREA" : "BRAE_CONTROL_AMI_AREA_NORMALISED"};
    bool allMiss = true;
    for (const char* name : controls)
    {
        setenv(name, "1", 1);
        const Compared k = build(m, g, fvp, ownerName, nbrName, src, tgt, acmi);
        unsetenv(name);
        const bool misses = k.setsDiffer > 0 || k.uncoveredNotZero > 0 || k.worstWeight > B_WEIGHT;
        std::printf("        CONTROL %s: %ld partner sets differ, entries %ld (OpenFOAM %ld), %ld uncovered faces "
                    "not 0, widest weight gap %.3e -- %s\n", name, k.setsDiffer, k.entriesBrae, k.entriesOf,
                    k.uncoveredNotZero, k.worstWeight, misses ? "misses, as it must" : "DOES NOT MISS");
        allMiss = allMiss && misses;
    }
    std::printf("  %-5s CONTROLS  each old path misses OpenFOAM's partner sets, its zeros or its weights\n",
                allMiss ? "ok:" : "FAIL:");
    fails += allMiss ? 0 : 1;
    std::printf("%s (%d failed)\n", fails == 0 ? "test_ami_store_vs_openfoam PASS" : "test_ami_store_vs_openfoam FAIL",
                fails);
    return fails == 0 ? 0 : 1;
}
