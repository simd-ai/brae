#include "solid_body_motion_function_cpp.cuh"
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace brae {

namespace {

const char* const WHO = "brae solidBodyMotionFunction: ";
// Foam::constant::mathematical
constexpr scalar pi = M_PI;
constexpr scalar twoPi = 2*M_PI;
constexpr scalar piByTwo = 0.5*M_PI;
// degToRad()
constexpr scalar degToRadFactor = M_PI/180.0;

// dictionary::readEntry: mandatory
scalar requiredScalar(
    const FoamDict& d,
    const std::string& key,
    const std::string& type)
{
    const std::vector<std::string>* v = d.find(key);
    if (!v || v->empty())
    {
        throw std::runtime_error(
            std::string(WHO) + type + " has no `" + key + "`. OpenFOAM reads it with a mandatory lookup "
            "and stops without it.");
    }
    return d.scalarOr(key, scalar(0));
}

vector requiredVector(
    const FoamDict& d,
    const std::string& key,
    const std::string& type)
{
    const std::vector<scalar> v = d.scalarListOr(key, {});
    if (v.size() != 3)
    {
        throw std::runtime_error(
            std::string(WHO) + type + " has no vector `" + key + "`. OpenFOAM reads it with a mandatory "
            "lookup and stops without it.");
    }
    return vector{v[0], v[1], v[2]};
}

// A coefficient OpenFOAM reads through Function1<Type>::New, which a case may write as a plain
// value (a Constant), `constant <value>`, a table, a sub-dictionary with a `type`... Only the first
// two are a constant, and only a constant is what these functions evaluate.
void requireConstantFunction1(
    const FoamDict& d,
    const std::string& key,
    const std::string& type)
{
    if (d.subDict(key))
    {
        throw std::runtime_error(
            std::string(WHO) + type + "'s `" + key + "` is a sub-dictionary -- a Function1 that is not a "
            "constant. Only a constant is ported; a time-varying one changes the motion.");
    }
    const std::vector<std::string>* v = d.find(key);
    if (!v || v->empty()) return;
    const std::string& first = v->front();
    const bool numeric = !first.empty()
        && (std::isdigit(static_cast<unsigned char>(first[0])) || first[0] == '-' || first[0] == '+'
            || first[0] == '.' || first[0] == '(');
    if (!numeric && first != "constant")
    {
        throw std::runtime_error(
            std::string(WHO) + type + "'s `" + key + "` is the Function1 `" + first + "`. Only a "
            "constant is ported; a time-varying one changes the motion.");
    }
}

class LinearMotion : public SolidBodyMotionFunction
{
    vector velocity_;
public:
    explicit LinearMotion(const FoamDict& c)
    :
        velocity_(requiredVector(c, "velocity", "linearMotion"))
    {}
    std::string type() const override
    {
        return "linearMotion";
    }
    Septernion transformation(scalar t) const override
    {
        // Translation of centre of gravity with constant velocity
        const vector displacement = velocity_*t;
        const Quaternion R(scalar(1));
        return Septernion(scalar(-1)*displacement)*R;
    }
};

class OscillatingLinearMotion : public SolidBodyMotionFunction
{
    vector amplitude_;
    scalar omega_;
    scalar phaseShift_ = 0;
    vector verticalShift_{0, 0, 0};
public:
    explicit OscillatingLinearMotion(const FoamDict& c)
    {
        for (const char* key : {"amplitude", "omega", "phaseShift", "verticalShift"})
        {
            requireConstantFunction1(c, key, "oscillatingLinearMotion");
        }
        amplitude_ = requiredVector(c, "amplitude", "oscillatingLinearMotion");
        omega_ = requiredScalar(c, "omega", "oscillatingLinearMotion");
        // NewIfPresent
        phaseShift_ = c.scalarOr("phaseShift", scalar(0));
        if (c.found("verticalShift"))
        {
            verticalShift_ = requiredVector(c, "verticalShift", "oscillatingLinearMotion");
        }
    }
    std::string type() const override
    {
        return "oscillatingLinearMotion";
    }
    Septernion transformation(scalar t) const override
    {
        const vector displacement = amplitude_*std::sin(omega_*(t + phaseShift_)) + verticalShift_;
        const Quaternion R(scalar(1));
        return Septernion(scalar(-1)*displacement)*R;
    }
};

class RotatingMotion : public SolidBodyMotionFunction
{
    vector origin_;
    vector axis_;
    scalar omega_;
public:
    explicit RotatingMotion(const FoamDict& c)
    {
        requireConstantFunction1(c, "omega", "rotatingMotion");
        origin_ = requiredVector(c, "origin", "rotatingMotion");
        axis_ = requiredVector(c, "axis", "rotatingMotion");
        omega_ = requiredScalar(c, "omega", "rotatingMotion");
    }
    std::string type() const override
    {
        return "rotatingMotion";
    }
    Septernion transformation(scalar t) const override
    {
        // omega_->integrate(0, t) of a Constant: (x2 - x1)*value
        const scalar angle = (t - scalar(0))*omega_;
        const Quaternion R(axis_, angle);
        return Septernion(scalar(-1)*origin_)*R*Septernion(origin_);
    }
};

class AxisRotationMotion : public SolidBodyMotionFunction
{
    vector origin_;
    vector radialVelocity_;
public:
    explicit AxisRotationMotion(const FoamDict& c)
    :
        origin_(requiredVector(c, "origin", "axisRotationMotion")),
        radialVelocity_(requiredVector(c, "radialVelocity", "axisRotationMotion"))
    {}
    std::string type() const override
    {
        return "axisRotationMotion";
    }
    Septernion transformation(scalar t) const override
    {
        // degToRad(deg) is deg*M_PI/180.0
        const vector omega{
            t*(radialVelocity_.x*M_PI/180.0),
            t*(radialVelocity_.y*M_PI/180.0),
            t*(radialVelocity_.z*M_PI/180.0)};
        const scalar magOmega = mag(omega);
        const Quaternion R(omega/magOmega, magOmega);
        return Septernion(scalar(-1)*origin_)*R*Septernion(origin_);
    }
};

class OscillatingRotatingMotion : public SolidBodyMotionFunction
{
    vector origin_;
    vector amplitude_;
    scalar omega_;
public:
    explicit OscillatingRotatingMotion(const FoamDict& c)
    :
        origin_(requiredVector(c, "origin", "oscillatingRotatingMotion")),
        amplitude_(requiredVector(c, "amplitude", "oscillatingRotatingMotion")),
        omega_(requiredScalar(c, "omega", "oscillatingRotatingMotion"))
    {}
    std::string type() const override
    {
        return "oscillatingRotatingMotion";
    }
    Septernion transformation(scalar t) const override
    {
        vector eulerAngles = amplitude_*std::sin(omega_*t);
        // Convert the rotational motion from deg to rad
        eulerAngles = eulerAngles*degToRadFactor;
        const Quaternion R(Quaternion::EulerOrder::XYZ, eulerAngles);
        return Septernion(scalar(-1)*origin_)*R*Septernion(origin_);
    }
};

// Ship Design Analysis: roll, sway and heave of a tank, the roll period drifting through resonance
class SDA : public SolidBodyMotionFunction
{
    vector CofG_;
    scalar lamda_;
    scalar rollAmax_;
    scalar rollAmin_;
    scalar heaveA_;
    scalar swayA_;
    scalar Q_;
    scalar Tp_;
    scalar Tpn_;
    scalar dTi_;
    scalar dTp_;
public:
    explicit SDA(const FoamDict& c)
    {
        CofG_ = requiredVector(c, "CofG", "SDA");
        lamda_ = requiredScalar(c, "lamda", "SDA");
        rollAmax_ = requiredScalar(c, "rollAmax", "SDA");
        rollAmin_ = requiredScalar(c, "rollAmin", "SDA");
        heaveA_ = requiredScalar(c, "heaveA", "SDA");
        swayA_ = requiredScalar(c, "swayA", "SDA");
        Q_ = requiredScalar(c, "Q", "SDA");
        Tp_ = requiredScalar(c, "Tp", "SDA");
        Tpn_ = requiredScalar(c, "Tpn", "SDA");
        dTi_ = requiredScalar(c, "dTi", "SDA");
        dTp_ = requiredScalar(c, "dTp", "SDA");
        // Rescale parameters according to the given scale parameter. SMALL
        if (lamda_ > 1 + scalar(1e-15))
        {
            heaveA_ /= lamda_;
            swayA_ /= lamda_;
            Tp_ /= std::sqrt(lamda_);
            Tpn_ /= std::sqrt(lamda_);
            dTi_ /= std::sqrt(lamda_);
            dTp_ /= std::sqrt(lamda_);
        }
    }
    std::string type() const override
    {
        return "SDA";
    }
    Septernion transformation(scalar time) const override
    {
        // Current roll period [sec]
        const scalar Tpi = Tp_ + dTp_*(time/dTi_);
        // Current Freq [/sec]
        const scalar wr = twoPi/Tpi;
        // Current Phase for roll [rad]
        const scalar r = dTp_/dTi_;
        const scalar u = Tp_ + r*time;
        const scalar phr = twoPi*((Tp_/u - 1) + std::log(std::fabs(u)) - std::log(Tp_))/r;
        // Current Phase for Sway [rad]
        const scalar phs = phr + pi;
        // Current Phase for Heave [rad]
        const scalar phh = phr + piByTwo;
        const scalar dT = Tpi - Tpn_;
        const scalar rollA = std::fmax(rollAmax_*std::exp(-(dT*dT)/(2*Q_)), rollAmin_);
        const vector T{
            0,
            swayA_*(std::sin(wr*time + phs) - std::sin(phs)),
            heaveA_*(std::sin(wr*time + phh) - std::sin(phh))};
        const Quaternion R(Quaternion::EulerOrder::XYZ, vector{rollA*std::sin(wr*time + phr), 0, 0});
        return Septernion(scalar(-1)*CofG_ - T)*R*Septernion(CofG_);
    }
};

class Tabulated6DoFMotion : public SolidBodyMotionFunction
{
    vector CofG_;
    bool spline_ = true;
    std::vector<scalar> times_;
    // translation, then rotation in degrees
    std::vector<vector> translation_;
    std::vector<vector> rotation_;

    // interpolateSplineXY on one component triple
    static vector splineXY(
        scalar x,
        const std::vector<scalar>& xOld,
        const std::vector<vector>& yOld)
    {
        const std::size_t n = xOld.size();
        if (n == 1 || x <= xOld[0]) return yOld[0];
        if (x >= xOld[n - 1]) return yOld[n - 1];
        if (n == 2)
        {
            return ((x - xOld[0])/(xOld[1] - xOld[0]))*(yOld[1] - yOld[0]) + yOld[0];
        }
        // find bounding knots
        std::size_t hi = 0;
        while (hi < n && xOld[hi] < x)
        {
            ++hi;
        }
        const std::size_t lo = hi - 1;
        const vector& y1 = yOld[lo];
        const vector& y2 = yOld[hi];
        const vector y0 = (lo == 0) ? (scalar(2)*y1 - y2) : yOld[lo - 1];
        const vector y3 = (hi + 1 == n) ? (scalar(2)*y2 - y1) : yOld[hi + 1];
        const scalar mu = (x - xOld[lo])/(xOld[hi] - xOld[lo]);
        return scalar(0.5)
           *(
                scalar(2)*y1
              + mu
               *(
                    scalar(-1)*y0 + y2
                  + mu*((scalar(2)*y0 - scalar(5)*y1 + scalar(4)*y2 - y3)
                  + mu*(scalar(-1)*y0 + scalar(3)*y1 - scalar(3)*y2 + y3))
                )
            );
    }

    // interpolateXY (interpolateXY.C:57-112), scan for scan: the table need not be sorted
    static vector linearXY(
        scalar x,
        const std::vector<scalar>& xOld,
        const std::vector<vector>& yOld)
    {
        const std::size_t n = xOld.size();
        std::size_t lo = 0;
        for (lo = 0; lo < n && xOld[lo] > x; ++lo)
        {}
        const std::size_t low = lo;
        if (low < n)
        {
            for (std::size_t i = low; i < n; ++i)
            {
                if (xOld[i] > xOld[lo] && xOld[i] <= x)
                {
                    lo = i;
                }
            }
        }
        std::size_t hi = 0;
        for (hi = 0; hi < n && xOld[hi] < x; ++hi)
        {}
        const std::size_t high = hi;
        if (high < n)
        {
            for (std::size_t i = high; i < n; ++i)
            {
                if (xOld[i] < xOld[hi] && xOld[i] >= x)
                {
                    hi = i;
                }
            }
        }
        if (lo < n && hi < n && lo != hi)
        {
            return yOld[lo] + ((x - xOld[lo])/(xOld[hi] - xOld[lo]))*(yOld[hi] - yOld[lo]);
        }
        if (lo == hi) return yOld[lo];
        if (lo == n) return yOld[hi];
        return yOld[lo];
    }

public:
    Tabulated6DoFMotion(
        const FoamDict& c,
        const std::string& caseDir)
    {
        const std::vector<std::string>* nameTokens = c.find("timeDataFileName");
        if (!nameTokens || nameTokens->empty())
        {
            throw std::runtime_error(std::string(WHO) + "tabulated6DoFMotion has no `timeDataFileName`.");
        }
        std::string file = nameTokens->front();
        if (file.size() >= 2 && file.front() == '"' && file.back() == '"')
        {
            file = file.substr(1, file.size() - 2);
        }
        const std::pair<std::string, std::string> tags[] = {
            {"<constant>", caseDir + "/constant"},
            {"<case>", caseDir},
            {"$FOAM_CASE", caseDir},
            {"${FOAM_CASE}", caseDir}};
        for (const auto& tag : tags)
        {
            const std::size_t p = file.find(tag.first);
            if (p != std::string::npos)
            {
                file.replace(p, tag.first.size(), tag.second);
            }
        }
        std::ifstream in(file);
        if (!in.good())
        {
            throw std::runtime_error(
                std::string(WHO) + "tabulated6DoFMotion: cannot open time data file " + file);
        }
        std::stringstream buffer;
        buffer << in.rdbuf();
        const std::string text = buffer.str();
        // List<Tuple2<scalar, Vector2D<vector>>>: a count, then (t ((tx ty tz) (rx ry rz))) per entry
        static const std::regex number(R"([-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
        std::vector<scalar> values;
        for (std::sregex_iterator it(text.begin(), text.end(), number), end; it != end; ++it)
        {
            values.push_back(std::stod(it->str()));
        }
        if (values.empty() || (values.size() - 1)%7 != 0
            || static_cast<std::size_t>(values[0]) != (values.size() - 1)/7)
        {
            throw std::runtime_error(
                std::string(WHO) + "tabulated6DoFMotion: " + file + " is not a list of "
                "(time ((translation) (rotation))) entries headed by its count.");
        }
        for (std::size_t k = 1; k + 6 < values.size(); k += 7)
        {
            times_.push_back(values[k]);
            translation_.push_back(vector{values[k + 1], values[k + 2], values[k + 3]});
            rotation_.push_back(vector{values[k + 4], values[k + 5], values[k + 6]});
        }
        CofG_ = requiredVector(c, "CofG", "tabulated6DoFMotion");
        const std::string scheme = c.wordOr("interpolationScheme", "spline");
        if (scheme != "spline" && scheme != "linear")
        {
            throw std::runtime_error(
                std::string(WHO) + "tabulated6DoFMotion: unrecognised interpolationScheme `" + scheme
                + "`; OpenFOAM has spline and linear.");
        }
        spline_ = (scheme == "spline");
    }
    std::string type() const override
    {
        return "tabulated6DoFMotion";
    }
    Septernion transformation(scalar t) const override
    {
        if (t < times_.front() || t > times_.back())
        {
            throw std::runtime_error(
                std::string(WHO) + "tabulated6DoFMotion: current time " + std::to_string(t)
                + " is outside the data table [" + std::to_string(times_.front()) + ", "
                + std::to_string(times_.back()) + "]. OpenFOAM stops on the same condition.");
        }
        const vector TRV0 = spline_ ? splineXY(t, times_, translation_) : linearXY(t, times_, translation_);
        vector TRV1 = spline_ ? splineXY(t, times_, rotation_) : linearXY(t, times_, rotation_);
        // Convert the rotational motion from deg to rad
        TRV1 = TRV1*degToRadFactor;
        const Quaternion R(Quaternion::EulerOrder::XYZ, TRV1);
        return Septernion(scalar(-1)*CofG_ + scalar(-1)*TRV0)*R*Septernion(CofG_);
    }
};

class MultiMotion : public SolidBodyMotionFunction
{
    std::vector<std::unique_ptr<SolidBodyMotionFunction>> SBMFs_;
public:
    MultiMotion(
        const FoamDict& c,
        const std::string& caseDir)
    {
        // every entry of the coefficients that IS a dictionary, in the order the file lists them
        for (const auto& entry : c.subs)
        {
            // an IOdictionary's header is consumed before its entries are
            if (entry.first == "FoamFile") continue;
            SBMFs_.push_back(SolidBodyMotionFunction::New(entry.second, caseDir));
        }
        if (SBMFs_.empty())
        {
            throw std::runtime_error(
                std::string(WHO) + "multiMotion lists no motion: none of its coefficients is a dictionary.");
        }
    }
    std::string type() const override
    {
        return "multiMotion";
    }
    Septernion transformation(scalar t) const override
    {
        Septernion TR = SBMFs_[0]->transformation(t);
        for (std::size_t i = 1; i < SBMFs_.size(); ++i)
        {
            TR *= SBMFs_[i]->transformation(t);
        }
        return TR;
    }
};

} // namespace

std::unique_ptr<SolidBodyMotionFunction> SolidBodyMotionFunction::New(
    const FoamDict& dict,
    const std::string& caseDir)
{
    const std::string motionType = dict.wordOr("solidBodyMotionFunction", "");
    if (motionType.empty())
    {
        throw std::runtime_error(
            std::string(WHO) + "the dictionary names no `solidBodyMotionFunction`. OpenFOAM reads it "
            "with a mandatory lookup.");
    }
    // solidBodyMotionFunction.C:65
    const FoamDict& c = *dict.optionalSubDict(motionType + "Coeffs");
    if (motionType == "linearMotion") return std::make_unique<LinearMotion>(c);
    if (motionType == "oscillatingLinearMotion") return std::make_unique<OscillatingLinearMotion>(c);
    if (motionType == "rotatingMotion") return std::make_unique<RotatingMotion>(c);
    if (motionType == "axisRotationMotion") return std::make_unique<AxisRotationMotion>(c);
    if (motionType == "oscillatingRotatingMotion") return std::make_unique<OscillatingRotatingMotion>(c);
    if (motionType == "SDA") return std::make_unique<SDA>(c);
    if (motionType == "tabulated6DoFMotion") return std::make_unique<Tabulated6DoFMotion>(c, caseDir);
    if (motionType == "multiMotion") return std::make_unique<MultiMotion>(c, caseDir);
    if (motionType == "drivenLinearMotion")
    {
        throw std::runtime_error(
            std::string(WHO) + "`drivenLinearMotion` is not ported: its displacement is read each step "
            "from another registered object's field (drivenLinearMotion.C), which brae has no registry "
            "to look up in.");
    }
    throw std::runtime_error(
        std::string(WHO) + "unknown solidBodyMotionFunction `" + motionType + "`. OpenFOAM v2412 has "
        "SDA, axisRotationMotion, drivenLinearMotion, linearMotion, multiMotion, "
        "oscillatingLinearMotion, oscillatingRotatingMotion, rotatingMotion and tabulated6DoFMotion.");
}

} // namespace brae
