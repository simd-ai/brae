#pragma once
// OpenFOAM's faceAreaWeightAMI -- the weights a cyclicAMI couples its two sides with -- the host
// reference, for a serial, untransformed pair.
//
// provenance:
//   openfoam: src/meshTools/AMIInterpolation/faceAreaIntersect/faceAreaIntersect.C (triSliceWithPlane,
//                 triangleIntersect, calc), faceAreaIntersectI.H (planeIntersection, getTriPoints)
//             src/meshTools/AMIInterpolation/AMIInterpolation/faceAreaWeightAMI/faceAreaWeightAMI.C
//                 (calculate, calcAddressing, processSourceFace, setNextFaces, calcInterArea, overlaps,
//                 restartUncoveredSourceFace)
//             src/meshTools/AMIInterpolation/AMIInterpolation/advancingFrontAMI/advancingFrontAMI.C
//                 (triangulatePatch, isCandidate, initialiseWalk, findTargetFace, appendNbrFaces;
//                 defaults triMode tmMesh, areaNormalisationMode project)
//             src/OpenFOAM/meshes/primitiveMesh/PrimitivePatch/PrimitivePatchAddressing.C (faceFaces)
//             src/OpenFOAM/meshes/meshShapes/face/faceIntersection.C (nearestPointClassify)
//             src/meshTools/AMIInterpolation/AMIInterpolation/AMIInterpolation.C (normaliseWeights)
//             src/OpenFOAM/meshes/meshShapes/face/face.C (split, mostConcaveAngle, calcEdgeVectors,
//                 areaNormal)
//   brae:     ami_interface.cuh is the device drivers' AMI, which clips on a projected plane; this is
//             OpenFOAM's triangle-by-triangle intersection
//   tests:    tests/test_face_area_weight_ami.cu against the weights OpenFOAM's own cyclicAMI holds on
//             RAS/mixerVesselAMI, step by step as the rotor turns
//
// THE WALK IS PART OF THE ANSWER. OpenFOAM finds each source face's partners with an advancing front:
// from a seed target face it tries the seed and its neighbours, and keeps going through the neighbours of
// every face whose intersection passes faceAreaIntersect::tol = 1e-6 of the source face's area. A target
// face reached no other way is never tried, so a face can come out partly covered -- and its weights are
// then normalised by their sum, which moves every one of them. An earlier version of this file evaluated
// every pair whose bounding boxes met: the same areas, but not the same pairs. MEASURED on
// RAS/mixerVesselAMI one step of 2e-4 in (1e-3 rad): OpenFOAM gives source face 2354 two partners
// covering 96.4% of it, the search five covering all of it -- the patch weights 1.7e-04 and deltaCoeffs
// 1.0e-02 off, pcorr 632 iterations where OpenFOAM takes 781, U 5.7e-05 after one step. So this is the
// walk, transcribed: its start (source 0 against target 0), the order PrimitivePatch lists face
// neighbours in, the 89-degree neighbour filter, the seeds the walk leaves for the next source face,
// the octree's nearest face (found here by brute force over the same distance) when it stalls, and the
// restart of faces left under 95% covered. What the front could add beyond that is refused rather than
// reproduced: requireMatch false, a transform, lowWeightCorrection, a distributed patch.
#include "cf_types.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace ami {

// A patch as primitivePatch sees it: its faces (point indices) and the points they index
struct Patch
{
    std::vector<std::vector<label>> faces;
    const std::vector<vector>* points = nullptr;
};

struct Weights
{
    std::vector<std::vector<label>> srcAddress;
    std::vector<std::vector<scalar>> srcWeights;
    std::vector<scalar> srcWeightsSum;
    std::vector<scalar> srcMagSf;
    std::vector<std::vector<label>> tgtAddress;
    std::vector<std::vector<scalar>> tgtWeights;
    std::vector<scalar> tgtWeightsSum;
    std::vector<scalar> tgtMagSf;
};

// face::triangles(points, triFaces): face::split in SPLITTRIANGLE mode, appended to `tris`
void faceTriangles(
    const std::vector<label>& f,
    const std::vector<vector>& points,
    std::vector<std::vector<label>>& tris);

// face::areaNormal(points)
vector faceAreaNormal(
    const std::vector<label>& f,
    const std::vector<vector>& points);

// faceAreaWeightAMI::calculate for a serial pair with requireMatch true and no transform:
// normaliseWeights(conformal = true)
Weights faceAreaWeight(
    const Patch& src,
    const Patch& tgt);

} // namespace ami
} // namespace cpu
} // namespace brae
