// glibc's acos and cos for the device, bit for bit. See device_glibc_math.cuh for what this is for and how the right
// variant is chosen at run time.
//
// DERIVED FROM:
// GNU C Library (glibc) 2.43, sysdeps/ieee754/dbl-64: e_asin.c (__ieee754_acos), s_sin.c (__cos), and the tables asincos.tbl,
// root.tbl, powtwo.tbl, sincostab.c that they include -- the "IBM Accurate Mathematical Library", written by
// International Business Machines Corp.
// Copyright (C) 2001-2026 Free Software Foundation, Inc.
//
// This program is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option)
// any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
// License for more details. You should have received a copy of the GNU Lesser General Public License along with this
// program; if not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: LGPL-2.1-or-later
//
// brae itself is distributed under the GNU Affero General Public License v3 (see LICENSE at the top of the tree). The
// glibc code above is licensed "version 2.1 or (at your option) any later version", so, as LGPL-2.1 section 3 and the
// "or later" clause allow, it is used here under the terms of version 3 (LGPL-3.0, GPL-3.0), with which the AGPL-3.0
// is compatible. The original copyright and license notices above are kept as the license requires.
//
// Changes: __ieee754_acos and __cos (every path reachable for |cos argument| < 105414350; glibc's branred path is not
// ported) were rewritten as __device__ functions, with every floating-point operation made explicit
// (__dmul_rn/__dadd_rn/__fma_rn) so that the compiler cannot contract them differently from glibc's own build.
// FMA=true reproduces glibc's *_fma ifunc variants (gcc's contraction pattern, replicated by hand); FMA=false
// reproduces the sse2/avx variants. The NaN results are glibc's bits.
#pragma once
#include <stdint.h>
#include "glibc_math_tables.cuh"

namespace glibcport {

__device__ __forceinline__ double asn(int i) { return __longlong_as_double((long long)d_ASNCS[i]); }
__device__ __forceinline__ double sct(int i) { return __longlong_as_double((long long)d_SINCOS[i]); }
__device__ __forceinline__ int hi(double x) { return (int)(__double_as_longlong(x) >> 32); }
__device__ __forceinline__ int lo(double x) { return (int)(__double_as_longlong(x) & 0xffffffff); }

// a*b + c
template <bool F> __device__ __forceinline__ double ma(double a, double b, double c) {
  if constexpr (F) return __fma_rn(a, b, c); else return __dadd_rn(__dmul_rn(a, b), c);
}
// c - a*b
template <bool F> __device__ __forceinline__ double nma(double a, double b, double c) {
  if constexpr (F) return __fma_rn(-a, b, c); else return __dadd_rn(c, -__dmul_rn(a, b));
}
// a*b - c
template <bool F> __device__ __forceinline__ double ms(double a, double b, double c) {
  if constexpr (F) return __fma_rn(a, b, -c); else return __dadd_rn(__dmul_rn(a, b), -c);
}
#define MUL(a, b) __dmul_rn((a), (b))
#define ADD(a, b) __dadd_rn((a), (b))
#define SUB(a, b) __dsub_rn((a), (b))

static constexpr double f1 = 1.66666666666664110590506577996662E-01, f2 = 7.50000000026122686814431784722623E-02,
  f3 = 4.46428561421059750978517350006940E-02, f4 = 3.03821268582119319911193410625235E-02,
  f5 = 2.23551211026525610742786300334557E-02, f6 = 1.81382903404565056280372531963613E-02;
static constexpr double t24 = 16777216.0, t27 = 134217728.0;
static constexpr double rt0 = 9.99999999859990725855365213134618E-01, rt1 = 4.99999999495955425917856814202739E-01,
  rt2 = 3.75017500867345182581453026130850E-01, rt3 = 3.12523626554518656309172508769531E-01;
static constexpr double hp0 = 0x1.921FB54442D18p0, hp1 = 0x1.1A62633145C07p-54;

// ---------------------------------------------------------------- acos
template <bool F> __device__ double acos_glibc(double x) {
  int m = hi(x);
  int k = 0x7fffffff & m;
  if (k < 0x3c880000) return hp0;
  if (k < 0x3fc00000) {
    double x2 = MUL(x, x);
    double h = ma<F>(f6, x2, f5); h = ma<F>(h, x2, f4); h = ma<F>(h, x2, f3); h = ma<F>(h, x2, f2); h = ma<F>(h, x2, f1);
    double x2x = MUL(x2, x);
    double r = SUB(hp0, x);
    double c0 = ADD(SUB(SUB(hp0, r), x), hp1);
    double cor = nma<F>(h, x2x, c0);   // c0 - t, t = h*x2x
    return ADD(r, cor);
  }
  int n;
  double xx, p, t, y, res;
  if (k < 0x3fe00000) {
    if (k < 0x3fd00000) n = 11 * ((k & 0x000fffff) >> 15); else n = 11 * ((k & 0x000fffff) >> 14) + 352;
    xx = (m > 0) ? SUB(x, asn(n)) : SUB(-x, asn(n));
    double q = ma<F>(xx, asn(n + 6), asn(n + 5)); q = ma<F>(xx, q, asn(n + 4)); q = ma<F>(xx, q, asn(n + 3)); q = ma<F>(xx, q, asn(n + 2));
    p = ma<F>(MUL(xx, xx), q, asn(n + 7));
    t = ma<F>(asn(n + 1), xx, p);
    y = (m > 0) ? SUB(hp0, asn(n + 8)) : ADD(hp0, asn(n + 8));
    t = (m > 0) ? SUB(hp1, t) : ADD(hp1, t);
    return ADD(y, t);
  }
  if (k < 0x3fe80000) {
    n = 1056 + ((k & 0x000fe000) >> 11) * 3;
    xx = (m > 0) ? SUB(x, asn(n)) : SUB(-x, asn(n));
    double q = ma<F>(xx, asn(n + 7), asn(n + 6)); q = ma<F>(xx, q, asn(n + 5)); q = ma<F>(xx, q, asn(n + 4)); q = ma<F>(xx, q, asn(n + 3)); q = ma<F>(xx, q, asn(n + 2));
    p = ma<F>(MUL(xx, xx), q, asn(n + 8));
    t = ma<F>(asn(n + 1), xx, p);
    y = (m > 0) ? SUB(hp0, asn(n + 9)) : ADD(hp0, asn(n + 9));
    t = (m > 0) ? SUB(hp1, t) : ADD(hp1, t);
    return ADD(y, t);
  }
  if (k < 0x3fed8000) {
    n = 992 + ((k & 0x000fe000) >> 13) * 13;
    xx = (m > 0) ? SUB(x, asn(n)) : SUB(-x, asn(n));
    double q = ma<F>(xx, asn(n + 8), asn(n + 7)); q = ma<F>(xx, q, asn(n + 6)); q = ma<F>(xx, q, asn(n + 5)); q = ma<F>(xx, q, asn(n + 4)); q = ma<F>(xx, q, asn(n + 3)); q = ma<F>(xx, q, asn(n + 2));
    p = ma<F>(MUL(xx, xx), q, asn(n + 9));
    t = ma<F>(asn(n + 1), xx, p);
    y = (m > 0) ? SUB(hp0, asn(n + 10)) : ADD(hp0, asn(n + 10));
    t = (m > 0) ? SUB(hp1, t) : ADD(hp1, t);
    return ADD(y, t);
  }
  if (k < 0x3fee8000) {
    n = 884 + ((k & 0x000fe000) >> 13) * 14;
    xx = (m > 0) ? SUB(x, asn(n)) : SUB(-x, asn(n));
    double q = ma<F>(xx, asn(n + 9), asn(n + 8)); q = ma<F>(xx, q, asn(n + 7)); q = ma<F>(xx, q, asn(n + 6)); q = ma<F>(xx, q, asn(n + 5)); q = ma<F>(xx, q, asn(n + 4)); q = ma<F>(xx, q, asn(n + 3)); q = ma<F>(xx, q, asn(n + 2));
    p = ma<F>(MUL(xx, xx), q, asn(n + 10));
    t = ma<F>(asn(n + 1), xx, p);
    y = (m > 0) ? SUB(hp0, asn(n + 11)) : ADD(hp0, asn(n + 11));
    t = (m > 0) ? SUB(hp1, t) : ADD(hp1, t);
    return ADD(y, t);
  }
  if (k < 0x3fef0000) {
    n = 768 + ((k & 0x000fe000) >> 13) * 15;
    xx = (m > 0) ? SUB(x, asn(n)) : SUB(-x, asn(n));
    double q = ma<F>(xx, asn(n + 10), asn(n + 9)); q = ma<F>(xx, q, asn(n + 8)); q = ma<F>(xx, q, asn(n + 7)); q = ma<F>(xx, q, asn(n + 6)); q = ma<F>(xx, q, asn(n + 5)); q = ma<F>(xx, q, asn(n + 4)); q = ma<F>(xx, q, asn(n + 3)); q = ma<F>(xx, q, asn(n + 2));
    p = ma<F>(MUL(xx, xx), q, asn(n + 11));
    t = ma<F>(asn(n + 1), xx, p);
    y = (m > 0) ? SUB(hp0, asn(n + 12)) : ADD(hp0, asn(n + 12));
    t = (m > 0) ? SUB(hp1, t) : ADD(hp1, t);
    return ADD(y, t);
  }
  if (k < 0x3ff00000) {
    double z = MUL(0.5, (m > 0) ? SUB(1.0, x) : ADD(1.0, x));
    int kz = hi(z);
    double tt = MUL(__longlong_as_double((long long)d_INROOT[(kz & 0x001fffff) >> 14]), __longlong_as_double((long long)d_POWTWO[511 - (kz >> 21)]));
    double r = nma<F>(MUL(tt, tt), z, 1.0);
    double pol = ma<F>(r, rt3, rt2); pol = ma<F>(r, pol, rt1); pol = ma<F>(r, pol, rt0);
    tt = MUL(tt, pol);
    double c = MUL(tt, z);
    double q = nma<F>(MUL(0.5, tt), c, 1.5);
    // t = c*q ; only use is (t+y) below -> fused when F
    double yy = ADD(MUL(t27, c), c);   // exact t27*c, then +c (== fma(t27,c,c))
    y = SUB(yy, MUL(t27, c));
    double tsum = ma<F>(c, q, y);                       // t + y
    double cc = (F ? nma<F>(y, y, z) : SUB(z, MUL(y, y))) / tsum;
    double pp = ma<F>(f6, z, f5); pp = ma<F>(pp, z, f4); pp = ma<F>(pp, z, f3); pp = ma<F>(pp, z, f2); pp = ma<F>(pp, z, f1);
    pp = MUL(pp, z);
    double yc = ADD(y, cc);
    if (m < 0) {
      double pm = MUL(pp, yc);   // gcc hoists p*(y+cc) above the branch (two uses) -> not contracted
      double cor = SUB(SUB(hp1, cc), pm);
      double res1 = SUB(hp0, y);
      res = ADD(res1, cor);
      return ADD(res, res);
    } else {
      double pm = MUL(pp, yc);
      double cor = ADD(cc, pm);
      res = ADD(y, cor);
      return ADD(res, res);
    }
  }
  if (k == 0x3ff00000 && lo(x) == 0) return (m > 0) ? 0.0 : 2.0 * hp0;
  // glibc: (x-x)/(x-x). x86 SSE: quiet-NaN input propagates, everything else gives the default NaN 0xfff8...
  if (x != x) return __longlong_as_double(__double_as_longlong(x) | 0x0008000000000000LL);
  return __longlong_as_double(0x7ff8000000000000LL);   // observed in libm.so.6: acos(inf), acos(1.0000000000000002) -> +qNaN
}

// ---------------------------------------------------------------- cos / sin helpers
static constexpr double s1 = -0x1.5555555555555p-3, s2 = 0x1.1111111110ECEp-7, s3 = -0x1.A01A019DB08B8p-13,
  s4 = 0x1.71DE27B9A7ED9p-19, s5 = -0x1.ADDFFC2FCDF59p-26;
static constexpr double bigc = 0x1.8000000000000p45;
static constexpr double mp1 = 0x1.921FB58000000p0, mp2 = -0x1.DDE973C000000p-27, pp3 = -0x1.CB3B398000000p-55,
  pp4 = -0x1.d747f23e32ed7p-83, hpinv = 0x1.45F306DC9C883p-1, toint = 0x1.8000000000000p52;
static constexpr double sn3 = -1.66666666666664880952546298448555E-01, sn5 = 8.33333214285722277379541354343671E-03,
  cs2 = 4.99999999999999999999950396842453E-01, cs4 = -4.16666666666664434524222570944589E-02,
  cs6 = 1.38888874007937613028114285595617E-03;

template <bool F> __device__ __forceinline__ double do_cos(double x, double dx) {
  if (x < 0) dx = -dx;
  double ax = fabs(x);
  double u = ADD(bigc, ax);
  int k = lo(u) << 2;
  x = ADD(SUB(ax, SUB(u, bigc)), dx);
  double xx = MUL(x, x);
  double inner = ma<F>(xx, sn5, sn3);
  double s = ma<F>(MUL(x, xx), inner, x);
  double c = MUL(xx, ma<F>(xx, ma<F>(xx, cs6, cs4), cs2));
  double sn = sct(k), ssn = sct(k + 1), cs = sct(k + 2), ccs = sct(k + 3);
  double cor = nma<F>(sn, s, nma<F>(cs, c, nma<F>(s, ssn, ccs)));
  return ADD(cs, cor);
}

template <bool F> __device__ __forceinline__ double do_sin(double x, double dx) {
  double xold = x;
  if (fabs(x) < 0.126) {
    double xx = MUL(x, x);
    double q = ma<F>(ma<F>(ma<F>(s5, xx, s4), xx, s3), xx, s2);
    double P = ma<F>(q, xx, s1);
    double inner = ms<F>(P, x, MUL(0.5, dx));
    double t = ma<F>(inner, xx, dx);
    return ADD(x, t);
  }
  if (x <= 0) dx = -dx;
  double ax = fabs(x);
  double u = ADD(bigc, ax);
  int k = lo(u) << 2;
  x = SUB(ax, SUB(u, bigc));
  double xx = MUL(x, x);
  double s = ADD(x, ma<F>(MUL(x, xx), ma<F>(xx, sn5, sn3), dx));
  double c = ma<F>(x, dx, MUL(xx, ma<F>(xx, ma<F>(xx, cs6, cs4), cs2)));
  double sn = sct(k), ssn = sct(k + 1), cs = sct(k + 2), ccs = sct(k + 3);
  double cor = ma<F>(cs, s, nma<F>(sn, c, ma<F>(s, ccs, ssn)));
  return copysign(ADD(sn, cor), xold);
}

template <bool F> __device__ __forceinline__ double do_sincos(double a, double da, int n) {
  double r = (n & 1) ? do_cos<F>(a, da) : do_sin<F>(a, da);
  return (n & 2) ? -r : r;
}

template <bool F> __device__ double cos_glibc(double x) {
  int m = hi(x);
  int k = 0x7fffffff & m;
  if (k < 0x3e400000) return 1.0;
  if (k < 0x3feb6000) return do_cos<F>(x, 0.0);
  if (k < 0x400368fd) {
    double y = SUB(hp0, fabs(x));
    double a = ADD(y, hp1);
    double da = ADD(SUB(y, a), hp1);
    return do_sin<F>(a, da);
  }
  if (k < 0x419921FB) {
    double t = ma<F>(x, hpinv, toint);
    double xn = SUB(t, toint);
    int n = lo(t) & 3;
    double y = nma<F>(xn, mp2, nma<F>(xn, mp1, x));
    double t2 = nma<F>(xn, pp3, y);
    double db = nma<F>(xn, pp3, SUB(y, t2));
    double b = nma<F>(xn, pp4, t2);
    db = ADD(db, nma<F>(xn, pp4, SUB(t2, b)));
    return do_sincos<F>(b, db, n + 1);
  }
  if (k >= 0x7ff00000) return (x != x) ? __longlong_as_double(__double_as_longlong(x) | 0x0008000000000000LL) : __longlong_as_double(0xfff8000000000000LL);  // glibc x/x
  return __longlong_as_double(0xfff8000000000000LL);  // finite |x| >= 105414350 (branred path) NOT ported
}
}  // namespace glibcport
