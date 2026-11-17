#pragma once
// b2 = cos(acos(a12) - theta) on the device, bit for bit what the host's glibc gives.
//
// WHY. The contact angle's rotation needs cos(acos(a12) - theta) per wall face, and the host hooks take it from glibc's
// libm. CUDA's acos and cos are not glibc's: MEASURED on 1e9 samples, stock CUDA differs from glibc in the last bit on
// 13.5% of acos and 12.6% of cos arguments and on 16-26% of the composition's, and on the G4 tank that moved the wall
// gradient by an ulp on ~100 faces a pass, fed back by the next. So the loop stayed on the host: a copy of a12 down, a
// libm loop, a copy of b2 up, and the GPU idle meanwhile -- 8% of the G4 step. This file is the device version that
// does not differ from glibc, on any input tried.
//
// HOW. glibc_acos_cos.cuh is glibc's own algorithm and tables ported to device code (glibc 2.34 and later), and
// correctly_rounded_acos.cuh a correctly rounded acos for glibc <= 2.33, whose acos is correctly rounded. glibc has
// more than one build of both: on x86-64 it selects at run time between an FMA and a plain-SSE variant (they differ in
// the last bit on 0.02-0.09% of arguments), and its algorithm changed between generations. So there are four device
// variants, and which one is THIS host's libm is found by a self-test at start-up: a few million values through the
// host's libm and through each variant, and the first variant that agrees on every bit is the one used. If none does
// (another architecture, a libm that is neither), the answer is -1 and the caller keeps the host's loop.
#include "cf_types.cuh"
#include <vector>
#include <cuda_runtime.h>

namespace brae {

// -1: no device variant reproduces this host's libm; the host loop must be used. Otherwise 0..3 (see the .cu).
// `thetas`: the contact angles the case uses, tested besides a fixed list. The variant is chosen once per process; a
// later call with an angle not yet tested checks the chosen variant on it alone.
int glibcB2Variant(const std::vector<scalar>& thetas);
const char* glibcB2VariantName(int variant);

// b2[i] = kind[i] == 1 ? cos(acos(a12[i]) - theta[i]) : 0 for i < n, on the calling thread's default stream.
void glibcB2(int variant, int n, const label* kind, const scalar* theta, const scalar* a12, scalar* b2);

} // namespace brae
