#include "wave_maker_point_patch_vector_field_cpp.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace brae {

namespace {

const char* const WHO = "brae waveMaker: ";

// Foam::constant::mathematical
constexpr scalar pi = M_PI;
constexpr scalar twoPi = 2*M_PI;

// SMALL in double precision
constexpr scalar small = 1.0e-15;

scalar requiredScalar(
    const FoamDict& d,
    const std::string& patchName,
    const std::string& key)
{
    const std::vector<std::string>* v = d.find(key);
    if (!v || v->empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName + "` has no `" + key + "`. OpenFOAM reads it with "
            "a mandatory lookup and stops without it.");
    }
    return d.scalarOr(key, scalar(0));
}

scalar pow3(scalar x)
{
    return x*x*x;
}

} // namespace

WaveMakerPointPatchVectorField::WaveMakerPointPatchVectorField(
    const FoamDict& dict,
    const std::string& patchName,
    const std::vector<vector>& localPoints,
    const vector& g,
    scalar startTime)
:
    patchName_(patchName),
    g_(g)
{
    const std::string motionType = dict.wordOr("motionType", "");
    if (motionType == "piston")
    {
        motionType_ = MotionType::piston;
    }
    else if (motionType == "flap")
    {
        motionType_ = MotionType::flap;
    }
    else if (motionType == "solitary")
    {
        motionType_ = MotionType::solitary;
    }
    else
    {
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName + "` names motionType `" + motionType + "`. "
            "OpenFOAM knows piston, flap and solitary, and stops on anything else.");
    }
    const std::vector<scalar> n = dict.scalarListOr("n", {});
    if (n.size() != 3)
    {
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName + "` has no vector `n`. OpenFOAM reads it with a "
            "mandatory lookup and stops without it.");
    }
    n_ = vector{n[0], n[1], n[2]};
    initialDepth_ = requiredScalar(dict, patchName, "initialDepth");
    wavePeriod_ = requiredScalar(dict, patchName, "wavePeriod");
    waveHeight_ = requiredScalar(dict, patchName, "waveHeight");
    // wavePhase: mandatory, and unused
    requiredScalar(dict, patchName, "wavePhase");
    waveAngle_ = dict.scalarOr("waveAngle", scalar(0));
    startTime_ = dict.scalarOr("startTime", startTime);
    rampTime_ = requiredScalar(dict, patchName, "rampTime");
    const std::string secondOrder = dict.wordOr("secondOrder", "false");
    secondOrder_ = (secondOrder == "true" || secondOrder == "yes" || secondOrder == "on" || secondOrder == "1");
    nPaddle_ = dict.intOr("nPaddle", 1);
    hadValue_ = dict.found("value");

    // Create the co-ordinate system
    if (mag(n_) < small)
    {
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName + "`: Patch normal direction vector is not set.");
    }
    n_ = n_/mag(n_);
    if (mag(g_) < small)
    {
        throw std::runtime_error(std::string(WHO) + "Gravity vector is not set.");
    }
    vector gHat = g_ - n_*dot(n_, g_);
    if (mag(gHat) < small)
    {
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName + "`: Patch normal and gravity directions must not "
            "be aligned.");
    }
    waveAngle_ *= pi/180;

    // initialiseGeometry: global patch extents
    vector bbMin = localPoints.front();
    vector bbMax = localPoints.front();
    for (const vector& q : localPoints)
    {
        bbMin = vector{std::fmin(bbMin.x, q.x), std::fmin(bbMin.y, q.y), std::fmin(bbMin.z, q.z)};
        bbMax = vector{std::fmax(bbMax.x, q.x), std::fmax(bbMax.y, q.y), std::fmax(bbMax.z, q.z)};
    }
    const scalar xMin = bbMin.x;
    const scalar xMax = bbMax.x;
    const scalar yMin = bbMin.y;
    const scalar yMax = bbMax.y;
    zMinGb_ = bbMin.z;

    // Global x, y positions of the paddle centres
    xPaddle_.assign(static_cast<std::size_t>(nPaddle_), 0);
    yPaddle_.assign(static_cast<std::size_t>(nPaddle_), 0);
    const scalar xMid = xMin + 0.5*(xMax - xMin);
    const scalar paddleDy = (yMax - yMin)/scalar(nPaddle_);
    for (label paddlei = 0; paddlei < nPaddle_; ++paddlei)
    {
        xPaddle_[static_cast<std::size_t>(paddlei)] = xMid;
        yPaddle_[static_cast<std::size_t>(paddlei)] = paddlei*paddleDy + yMin + 0.5*paddleDy;
    }

    // Local point-to-paddle addressing
    pointToPaddle_.assign(localPoints.size(), -1);
    for (std::size_t ppi = 0; ppi < localPoints.size(); ++ppi)
    {
        pointToPaddle_[ppi] = static_cast<label>(std::floor((localPoints[ppi].y - yMin)/(paddleDy + 0.01*paddleDy)));
    }

    waterDepthRef_.assign(static_cast<std::size_t>(nPaddle_), -1);
    if (!hadValue_)
    {
        throw std::runtime_error(
            std::string(WHO) + "patch `" + patchName + "` has no `value`. OpenFOAM then evaluates the "
            "paddle at construction, at the start time; brae reads the displacement from `value` only.");
    }
}

scalar WaveMakerPointPatchVectorField::waveLength(
    scalar h,
    scalar T) const
{
    const scalar L0 = mag(g_)*T*T/twoPi;
    scalar L = L0;
    for (label i = 1; i <= 100; ++i)
    {
        L = L0*std::tanh(twoPi*h/L);
    }
    return L;
}

scalar WaveMakerPointPatchVectorField::timeCoeff(scalar t) const
{
    return std::min(std::max(t/rampTime_, scalar(0)), scalar(1));
}

std::vector<vector> WaveMakerPointPatchVectorField::updateCoeffs(
    scalar time,
    const std::vector<vector>& localPoints)
{
    if (firstTime_)
    {
        // Set the reference water depth
        if (initialDepth_ != 0)
        {
            for (scalar& d : waterDepthRef_)
            {
                d = initialDepth_;
            }
        }
        else
        {
            throw std::runtime_error(
                std::string(WHO) + "patch `" + patchName_ + "`: initialDepth is not set.");
        }
        firstTime_ = false;
    }

    const scalar t = time - startTime_;

    const std::size_t nP = static_cast<std::size_t>(nPaddle_);
    std::vector<scalar> waveK(nP, -1);
    std::vector<scalar> waveKx(nP, -1);
    std::vector<scalar> waveKy(nP, -1);
    for (std::size_t paddlei = 0; paddlei < nP; ++paddlei)
    {
        const scalar waveLengthi = waveLength(waterDepthRef_[paddlei], wavePeriod_);
        waveK[paddlei] = twoPi/waveLengthi;
        waveKx[paddlei] = waveK[paddlei]*std::cos(waveAngle_);
        waveKy[paddlei] = waveK[paddlei]*std::sin(waveAngle_);
    }
    const scalar sigma = 2*pi/wavePeriod_;

    std::vector<vector> value(localPoints.size());
    switch (motionType_)
    {
        case MotionType::flap:
        {
            std::vector<scalar> motionX(localPoints.size(), -1);
            for (std::size_t pointi = 0; pointi < localPoints.size(); ++pointi)
            {
                const std::size_t paddlei = static_cast<std::size_t>(pointToPaddle_[pointi]);
                const scalar phaseTot = waveKx[paddlei]*xPaddle_[paddlei] + waveKy[paddlei]*yPaddle_[paddlei];
                const scalar depthRef = waterDepthRef_[paddlei];
                const scalar kh = waveK[paddlei]*depthRef;
                const scalar pz = localPoints[pointi].z;
                const scalar m1 =
                    (4*std::sinh(kh)/(std::sinh(2*kh) + 2*kh))
                  * (std::sinh(kh) + 1/kh*(1 - std::cosh(kh)));
                const scalar boardStroke = waveHeight_/m1;
                motionX[pointi] = 0.5*boardStroke*std::sin(phaseTot - sigma*t);
                if (secondOrder_)
                {
                    motionX[pointi] +=
                        waveHeight_*waveHeight_/(16*depthRef)
                      * (3*std::cosh(kh)/pow3(std::sinh(kh)) - 2/m1)
                      * std::sin(phaseTot - 2*sigma*t);
                }
                motionX[pointi] *= 1.0 + (pz - zMinGb_ - depthRef)/depthRef;
            }
            const vector tn = timeCoeff(t)*n_;
            for (std::size_t pointi = 0; pointi < localPoints.size(); ++pointi)
            {
                value[pointi] = tn*motionX[pointi];
            }
            break;
        }
        case MotionType::piston:
        {
            std::vector<scalar> motionX(localPoints.size(), -1);
            for (std::size_t pointi = 0; pointi < localPoints.size(); ++pointi)
            {
                const std::size_t paddlei = static_cast<std::size_t>(pointToPaddle_[pointi]);
                const scalar phaseTot = waveKx[paddlei]*xPaddle_[paddlei] + waveKy[paddlei]*yPaddle_[paddlei];
                const scalar depthRef = waterDepthRef_[paddlei];
                const scalar kh = waveK[paddlei]*depthRef;
                const scalar m1 = 2*(std::cosh(2*kh) - 1.0)/(std::sinh(2*kh) + 2*kh);
                const scalar boardStroke = waveHeight_/m1;
                motionX[pointi] = 0.5*boardStroke*std::sin(phaseTot - sigma*t);
                if (secondOrder_)
                {
                    motionX[pointi] +=
                      + waveHeight_*waveHeight_
                      / (32*depthRef)*(3*std::cosh(kh)/pow3(std::sinh(kh)) - 2.0/m1)
                      * std::sin(phaseTot - 2*sigma*t);
                }
            }
            const vector tn = timeCoeff(t)*n_;
            for (std::size_t pointi = 0; pointi < localPoints.size(); ++pointi)
            {
                value[pointi] = tn*motionX[pointi];
            }
            break;
        }
        case MotionType::solitary:
        {
            const scalar magG = mag(g_);
            for (std::size_t pointi = 0; pointi < localPoints.size(); ++pointi)
            {
                const std::size_t paddlei = static_cast<std::size_t>(pointToPaddle_[pointi]);
                const scalar depthRef = waterDepthRef_[paddlei];
                const scalar kappa = std::sqrt(0.75*waveHeight_/pow3(depthRef));
                const scalar celerity = std::sqrt(magG*(depthRef + waveHeight_));
                const scalar stroke = std::sqrt(16*waveHeight_*depthRef/3.0);
                const scalar hr = waveHeight_/depthRef;
                wavePeriod_ = 2.0/(kappa*celerity)*(3.8 + hr);
                const scalar tSolitary = -0.5*wavePeriod_ + t;
                // Newton-Raphson
                scalar theta1 = 0;
                scalar theta2 = 0;
                scalar er = 10000;
                const scalar error = 0.001;
                while (er > error)
                {
                    theta2 =
                        theta1
                      - (theta1 - kappa*celerity*tSolitary + hr*std::tanh(theta1))
                       /(1.0 + hr*(1.0/std::cosh(theta1))*(1.0/std::cosh(theta1)));
                    er = std::fabs(theta1 - theta2);
                    theta1 = theta2;
                }
                const scalar motionX = waveHeight_/(kappa*depthRef)*std::tanh(theta1) + 0.5*stroke;
                value[pointi] = n_*motionX;
            }
            break;
        }
    }
    return value;
}

} // namespace brae
