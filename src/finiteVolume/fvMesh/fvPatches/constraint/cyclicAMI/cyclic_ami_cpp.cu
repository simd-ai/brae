#include "cyclic_ami_cpp.cuh"
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace brae {
namespace cpu {
namespace cyclicAMIFvPatch {

namespace {

const char* const WHO = "brae cyclicAMI: ";

// The patch's polyMesh/boundary entry, keyword by keyword. PatchInfo keeps the few a cyclic needs; an
// AMI keyword it does not know would otherwise be skipped silently by the mesh reader.
void checkBoundaryEntry(
    const std::string& polyMeshDir,
    const std::string& name)
{
    std::ifstream in(polyMeshDir + "/boundary");
    if (!in)
    {
        throw std::runtime_error(
            std::string(WHO) + "cannot read " + polyMeshDir + "/boundary to check the keywords of `" + name
            + "`.");
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    const std::regex block("(^|\\s)" + name + "\\s*\\{([^}]*)\\}");
    std::smatch mb;
    if (!std::regex_search(text, mb, block))
    {
        throw std::runtime_error(
            std::string(WHO) + "no entry for `" + name + "` in " + polyMeshDir + "/boundary.");
    }
    const std::string body = mb[2].str();
    static const std::regex entry("([A-Za-z_][A-Za-z0-9_]*)\\s+([^;]*);");
    for (std::sregex_iterator it(body.begin(), body.end(), entry), end; it != end; ++it)
    {
        const std::string key = (*it)[1].str();
        std::string value = (*it)[2].str();
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        {
            value.pop_back();
        }
        const auto refuse = [&](const std::string& why)
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + name + "` sets `" + key + " " + value + "`. " + why);
        };
        if (key == "type" || key == "inGroups" || key == "nFaces" || key == "startFace"
         || key == "neighbourPatch" || key == "matchTolerance")
        {
            continue;
        }
        // advancingFrontAMI's walk: whether an uncovered source face restarts the front. brae evaluates
        // every pair and refuses an uncovered face outright, so the value changes nothing it computes.
        if (key == "restartUncoveredSourceFace")
        {
            continue;
        }
        if (key == "transform")
        {
            if (value != "noOrdering")
            {
                refuse("Only an untransformed pair (`noOrdering`) is ported; a rotational, translational or "
                       "computed transform moves the neighbour's points and every value across the pair.");
            }
            continue;
        }
        if (key == "AMIMethod" || key == "method")
        {
            if (value != "faceAreaWeightAMI")
            {
                refuse("Only faceAreaWeightAMI is ported.");
            }
            continue;
        }
        if (key == "requireMatch")
        {
            if (value != "true" && value != "yes" && value != "on" && value != "1")
            {
                refuse("requireMatch false normalises the weights by the face area instead of by their sum "
                       "(AMIInterpolation::normaliseWeights); only the conformal form is ported.");
            }
            continue;
        }
        if (key == "lowWeightCorrection")
        {
            if (std::strtod(value.c_str(), nullptr) > 0)
            {
                refuse("A low-weight correction replaces the interpolate by the face's own cell where the "
                       "weights sum below it; it is not ported.");
            }
            continue;
        }
        refuse("brae's cyclicAMI does not read that keyword, and running without it could be a different "
               "interface.");
    }
}

ami::Patch amiPatch(
    const PrimitiveMesh& m,
    const FvPatch& p)
{
    ami::Patch a;
    a.points = &m.points();
    a.faces.resize(static_cast<std::size_t>(p.size));
    for (label i = 0; i < p.size; ++i)
    {
        const label f = p.start + i;
        std::vector<label>& face = a.faces[static_cast<std::size_t>(i)];
        face.resize(static_cast<std::size_t>(m.faceSize(f)));
        for (label k = 0; k < m.faceSize(f); ++k)
        {
            face[static_cast<std::size_t>(k)] = m.faceVert(f, k);
        }
    }
    return a;
}

// The coupled half of `p` against `q` through the AMI `address`/`weight` (p's side of it)
void couple(
    FvPatch& p,
    const FvPatch& q,
    label nbr,
    bool owner,
    const std::vector<std::vector<label>>& address,
    const std::vector<std::vector<scalar>>& weight,
    const FvGeometry& g)
{
    p.coupled = true;
    p.nbrPatch = nbr;
    p.owner = owner;
    // syncTools exchanges across processor and cyclicPolyPatch only; an AMI pair is neither
    p.ami = true;
    p.nbrFaceCells.clear();
    const std::size_t n = static_cast<std::size_t>(p.size);
    p.amiOffsets.assign(n + 1, 0);
    p.amiNbrFaces.clear();
    p.amiNbrCells.clear();
    p.amiWeights.clear();
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < address[i].size(); ++j)
        {
            const label facej = address[i][j];
            p.amiNbrFaces.push_back(facej);
            p.amiNbrCells.push_back(q.faceCells[static_cast<std::size_t>(facej)]);
            p.amiWeights.push_back(weight[i][j]);
        }
        p.amiOffsets[i + 1] = static_cast<label>(p.amiNbrFaces.size());
    }

    // the neighbour's coupledFvPatch::delta() and nf & it, on its own faces, to be interpolated
    std::vector<vector> nbrPatchD(static_cast<std::size_t>(q.size));
    std::vector<scalar> nbrDeltas(static_cast<std::size_t>(q.size));
    for (label j = 0; j < q.size; ++j)
    {
        const std::size_t k = static_cast<std::size_t>(j);
        nbrPatchD[k] = g.Cf()[q.start + j] - g.C()[q.faceCells[k]];
        nbrDeltas[k] = dot(q.nf[k], nbrPatchD[k]);
    }
    p.weights.resize(n);
    p.delta.resize(n);
    p.deltaCoeffs.resize(n);
    p.nonOrthDeltaCoeffs.resize(n);
    p.nonOrthCorrectionVectors.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const label facei = static_cast<label>(i);
        const vector patchD = g.Cf()[p.start + facei] - g.C()[p.faceCells[i]];
        // makeWeights, with its mag
        const scalar di = std::fabs(dot(p.nf[i], patchD));
        const scalar dni = std::fabs(patchNeighbourFaceValue(p, facei, nbrDeltas));
        p.weights[i] = dni/(di + dni);
        // delta(), parallel
        const vector d = patchD - patchNeighbourFaceValue(p, facei, nbrPatchD);
        p.delta[i] = d;
        p.deltaCoeffs[i] = scalar(1)/mag(d);
        p.nonOrthDeltaCoeffs[i] = scalar(1)/std::fmax(dot(p.nf[i], d), scalar(0.05)*mag(d));
        p.nonOrthCorrectionVectors[i] = p.nf[i] - d*p.nonOrthDeltaCoeffs[i];
    }
}

} // namespace

void Interfaces::update(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    std::vector<FvPatch>& patches)
{
    for (Pair& pr : pairs_)
    {
        FvPatch& src = patches[static_cast<std::size_t>(pr.src)];
        FvPatch& tgt = patches[static_cast<std::size_t>(pr.tgt)];
        pr.weights = ami::faceAreaWeight(amiPatch(m, src), amiPatch(m, tgt));
        couple(src, tgt, pr.tgt, true, pr.weights.srcAddress, pr.weights.srcWeights, g);
        couple(tgt, src, pr.src, false, pr.weights.tgtAddress, pr.weights.tgtWeights, g);
    }
}

Interfaces setup(
    const std::string& polyMeshDir,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    std::vector<FvPatch>& patches)
{
    Interfaces I;
    const std::vector<PatchInfo>& info = m.patches();
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (info[pi].type != "cyclicAMI") continue;
        checkBoundaryEntry(polyMeshDir, info[pi].name);
        if (!info[pi].periodicPatch.empty())
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + info[pi].name + "` names the periodicPatch `"
                + info[pi].periodicPatch + "`; a periodic AMI is not ported.");
        }
        label nbr = -1;
        for (std::size_t qi = 0; qi < patches.size(); ++qi)
        {
            if (info[qi].name == info[pi].neighbourPatch)
            {
                nbr = static_cast<label>(qi);
            }
        }
        if (nbr < 0 || info[static_cast<std::size_t>(nbr)].type != "cyclicAMI")
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + info[pi].name + "` names the neighbourPatch `"
                + info[pi].neighbourPatch + "`, which is not a cyclicAMI patch of the mesh.");
        }
        // cyclicAMIPolyPatch::owner(): index() < neighbPatchID()
        if (static_cast<label>(pi) < nbr)
        {
            Pair pr;
            pr.src = static_cast<label>(pi);
            pr.tgt = nbr;
            I.pairs_.push_back(pr);
        }
    }
    I.update(m, g, patches);
    return I;
}

} // namespace cyclicAMIFvPatch
} // namespace cpu
} // namespace brae
