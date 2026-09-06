#pragma once
// cf GPU offload (G0): BLAS-1 device kernels used by the Krylov solvers (the SIMPLE loop's hot path).
// Each is validated against the CPU baseline; the CPU result is the oracle.
#include "cf_types.cuh"
#include "device_buffer.cuh"

namespace brae {

void   deviceAxpy(scalar a, const DeviceBuffer<scalar>& x, DeviceBuffer<scalar>& y);  // y += a*x
void   deviceScale(DeviceBuffer<scalar>& x, scalar a);                                // x *= a
scalar deviceDot(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y);       // x . y  (reduction)
void   deviceCopy(DeviceBuffer<scalar>& dst, const DeviceBuffer<scalar>& src);        // dst = src (D2D)
void   deviceJacobi(DeviceBuffer<scalar>& z, const DeviceBuffer<scalar>& r, const scalar* diag);  // z = r/diag
scalar deviceSumMag(const DeviceBuffer<scalar>& x);                                   // sum |x|  (reduction)
// max over i of x[i]/y[i], skipping y[i] <= 0. The Courant number is a MAXIMUM, which cannot be built
// from the sum/dot reductions above; this keeps it a single device pass returning one scalar.
scalar deviceMaxRatio(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y);
void   deviceHadamard(DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b);  // out = a.*b
void   deviceDivide(DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b);     // out = a./b (0 where b==0)
// fused multi-term BLAS-1 (fewer launches + less memory traffic, bit-identical to the separate-kernel sequences):
void   deviceFusedBicgP(const DeviceBuffer<scalar>& rA, DeviceBuffer<scalar>& pA, const DeviceBuffer<scalar>& AyA,
                        const scalar* beta, const scalar* negOmega);                            // pA = rA + beta*(pA - omega*AyA)
void   deviceFusedSxpy(DeviceBuffer<scalar>& out, const DeviceBuffer<scalar>& src, const scalar* a, const DeviceBuffer<scalar>& x); // out = src + a*x
void   deviceFusedAxpy2(DeviceBuffer<scalar>& y, const scalar* a, const DeviceBuffer<scalar>& x1, const scalar* b, const DeviceBuffer<scalar>& x2); // y += a*x1 + b*x2
void   deviceFusedScaleAxpy(DeviceBuffer<scalar>& p, const scalar* b, const DeviceBuffer<scalar>& w);  // p = b*p + w

// --- device-resident scalar plumbing: keep Krylov coefficients (alpha/beta/rho/...) OFF the host so the loop
// never blocks on a D2H reduction read except the once-per-iter convergence norm. The *Into reductions write to a
// caller-owned device scalar; the *Dev axpy/scale read their coefficient from a device scalar; the scalar ops do
// the recurrence arithmetic on-device. Same kernels/arithmetic as the host-scalar path -> bit-identical results.
void   deviceDotInto(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y, scalar* dResult);  // *dResult = x.y
void   deviceSumMagInto(const DeviceBuffer<scalar>& x, scalar* dResult);                              // *dResult = sum|x|
// A device vector of ones of length n, built once per length and kept. normFactor (lduMatrixSolver.C's
// sumA = A*1) and SIMPLEC's row sum both need one; rebuilding it on the host and uploading it from
// pageable memory on every call was a stream sync each time -- three per outer iteration on the flat
// plate, 4.4 ms. A vector of ones is a vector of ones, so sharing it changes no number.
const DeviceBuffer<scalar>& deviceOnes(int n);
void   deviceAxpyDev(const scalar* dA, const DeviceBuffer<scalar>& x, DeviceBuffer<scalar>& y);       // y += (*dA)*x
void   deviceScaleDev(const scalar* dA, DeviceBuffer<scalar>& x);                                     // x *= (*dA)
void   deviceScalarDiv(const scalar* num, const scalar* den, scalar* out);                            // *out = *num / *den
void   deviceScalarDivNeg(const scalar* num, const scalar* den, scalar* out, scalar* outNeg);         // *out=q, *outNeg=-q
void   deviceScalarDivConst(const scalar* num, scalar denom, scalar* out);                            // *out = *num / denom (host const)
void   deviceScalarAdd2(const scalar* a, const scalar* b, scalar c, scalar* out);                     // *out = *a + *b + c
void   deviceScalarCopy(const scalar* src, scalar* dst);                                              // *dst = *src
scalar deviceReadScalar(const scalar* dSrc);
// The stage-1 partials buffer behind every reduction is grown on demand and the old one FREED. A graph that
// captured a reduction holds that pointer; compare this before replaying and recapture when it changed.
int deviceReductionScratchEpoch();                                                          // pinned D2H (1 sync)

} // namespace brae
