#include "rotorDiskSource_cpp.cuh"

#include <cmath>

namespace brae {
namespace cpu {

namespace {

constexpr scalar PI    = 3.14159265358979323846;
constexpr scalar TWOPI = 6.28318530717958647692;

// OF bladeModel::interpolate -- linear in radius, clamped outside the table.
void bladeAt(
    scalar                     r,
    const std::vector<scalar>& R,
    const std::vector<scalar>& tw,
    const std::vector<scalar>& ch,
    scalar&                    twist,
    scalar&                    chord)
{
    const int m = static_cast<int>(R.size());
    if (m == 0)
    {
        twist = 0;
        chord = 0;
        return;
    }
    if (m == 1)
    {
        twist = tw[0];
        chord = ch[0];
        return;
    }
    int i2 = 0;
    while (i2 < m && R[i2] < r) ++i2;
    if (i2 == 0)
    {
        twist = tw[0];
        chord = ch[0];
    }
    else if (i2 == m)
    {
        twist = tw[m - 1];
        chord = ch[m - 1];
    }
    else
    {
        const int i1 = i2 - 1;
        const scalar w = (r - R[i1]) / (R[i2] - R[i1]);
        twist = tw[i1] + w * (tw[i2] - tw[i1]);
        chord = ch[i1] + w * (ch[i2] - ch[i1]);
    }
}

// OF profileModel lookup::Cdl -- linear in alpha, clamped.
scalar tableAt(scalar a, const std::vector<scalar>& A, const std::vector<scalar>& C)
{
    const int n = static_cast<int>(A.size());
    if (n == 0) return 0;
    if (n == 1) return C[0];
    int i2 = 0;
    while (i2 < n && A[i2] < a) ++i2;
    if (i2 == 0) return C[0];
    if (i2 == n) return C[n - 1];
    const int i1 = i2 - 1;
    const scalar w = (a - A[i1]) / (A[i2] - A[i1]);
    return C[i1] + w * (C[i2] - C[i1]);
}

} // namespace

RotorDisk buildRotorDisk(
    const RotorDiskParams& p,
    const PrimitiveMesh&   m,
    const FvGeometry&      g)
{
    RotorDisk d;
    if (!p.active || p.cells.empty()) return d;

    d.active = true;
    const scalar am = mag(p.axis);
    d.axis = p.axis / am;
    d.omega = p.omega;
    d.nBlades = static_cast<scalar>(p.nBlades);
    d.tipEffect = p.tipEffect;
    d.theta0 = p.theta0;
    d.localInflow = p.localInflow;
    d.inletVel = p.inletVel;
    d.cells = p.cells;
    d.pAlpha = p.pAlpha;
    d.pCd = p.pCd;
    d.pCl = p.pCl;

    const int n = static_cast<int>(p.cells.size());
    d.e2.resize(n);
    d.radius.resize(n);
    d.twist.resize(n);
    d.chord.resize(n);
    d.area.assign(n, 0.0);

    std::vector<label> cellAddr(m.nCells(), -1);
    for (int i = 0; i < n; ++i) cellAddr[p.cells[i]] = i;

    for (int i = 0; i < n; ++i)
    {
        const vector rel = g.C()[p.cells[i]] - p.origin;
        const scalar ax = dot(rel, d.axis);
        const vector radv = rel - ax * d.axis;      // the component perpendicular to the axis
        const scalar r = mag(radv);
        d.radius[i] = r;
        d.rMax = std::fmax(d.rMax, r);
        const vector e1 = (r > 1e-12) ? (radv / r) : vector{1, 0, 0};
        d.e2[i] = cross(d.axis, e1);                // azimuthal
        bladeAt(r, p.bladeR, p.bladeTwist, p.bladeChord, d.twist[i], d.chord[i]);
    }

    // OF setFaceArea: the disk's frontal area, gathered from the internal faces on the zone boundary
    // whose normal is aligned with the axis. A face INSIDE the zone has both cells in it and adds
    // nothing -- the disk is a surface, not a volume.
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    for (label f = 0; f < m.nInternalFaces(); ++f)
    {
        const label oa = cellAddr[own[f]], na = cellAddr[nei[f]];
        if ((oa == -1) == (na == -1)) continue;
        const scalar ms = mag(g.Sf()[f]);
        if (ms < 1e-300) continue;
        const scalar nfDotAx = dot(g.Sf()[f], d.axis) / ms;
        if (na == -1 && nfDotAx > 0.8)  d.area[oa] += ms;
        if (oa == -1 && -nfDotAx > 0.8) d.area[na] += ms;
    }
    return d;
}

void rotorForce(
    const RotorDisk&           rd,
    const std::vector<vector>& U,
    std::vector<vector>&       force)
{
    force.assign(U.size(), vector{0, 0, 0});
    if (!rd.active) return;

    for (std::size_t i = 0; i < rd.cells.size(); ++i)
    {
        const label c = rd.cells[i];
        const scalar r = rd.radius[i];
        if (!(r > 1e-12) || !(rd.area[i] > 1e-30)) continue;

        const vector Uin = rd.localInflow ? U[c] : rd.inletVel;
        // The cylindrical frame. The radial component is discarded (OF sets Uc.x = 0) and the azimuthal
        // one is taken BLADE-RELATIVE: the blade moves at r*omega, the fluid at e2.U.
        const scalar Uaz = rd.omega * r - dot(rd.e2[i], Uin);
        const scalar Uax = dot(rd.axis, Uin);
        const scalar magSqrUc = Uaz * Uaz + Uax * Uax;

        scalar alphaGeom = rd.theta0 + rd.twist[i];
        if (rd.omega < 0) alphaGeom = PI - alphaGeom;
        scalar alphaEff = alphaGeom - std::atan2(-Uax, Uaz);
        if (alphaEff >  PI) alphaEff -= TWOPI;
        if (alphaEff < -PI) alphaEff += TWOPI;

        const scalar Cd = tableAt(alphaEff, rd.pAlpha, rd.pCd);
        const scalar Cl = tableAt(alphaEff, rd.pAlpha, rd.pCl);
        // OF neg(x): 1 where x < 0. Above tipEffect*rMax the blade produces no lift.
        const scalar tip = (r / rd.rMax - rd.tipEffect < 0) ? 1.0 : 0.0;

        const scalar pDyn = 0.5 * magSqrUc;         // incompressible: rho = 1 (kinematic)
        const scalar f = pDyn * rd.chord[i] * rd.nBlades * rd.area[i] / r / TWOPI;

        // localForce = (0, -f*Cd, tip*f*Cl) in (e1, e2, e3), rotated back to Cartesian.
        force[c] = (-f * Cd) * rd.e2[i] + (tip * f * Cl) * rd.axis;
    }
}

void addSup(
    const RotorDisk&           rd,
    const std::vector<vector>& U,
    std::vector<vector>&       source)
{
    if (!rd.active) return;
    std::vector<vector> f;
    rotorForce(rd, U, f);
    // OF: `eqn -= force` with force per-volume, and operator-= is source += V*su -- so the extensive
    // source gains the RAW force. See the header on why the sign is easy to get backwards.
    for (std::size_t i = 0; i < rd.cells.size(); ++i)
    {
        const label c = rd.cells[i];
        source[c] = source[c] + f[c];
    }
}

} // namespace cpu
} // namespace brae
