#!/usr/bin/env python3
"""pimple_clock_replay.py -- replay a pimpleFoam log's time-step decisions through OpenFOAM v2412's own rules.

    pimple_clock_replay.py <log> --maxCo C [--maxDeltaT D] --deltaT0 DT --start T --index I
                           [--writeControl adjustable|timeStep|...] [--writeInterval W] [--fused-write 0|1]

The log is OpenFOAM's or brae's at writePrecision 17, so each printed number is the double the run held: per step a
`Courant Number mean: .. max: Co` line (CourantNo.H) and a `deltaT = dt` line (setDeltaT.H). The rules, as pimpleFoam.C
applies them (v2412, pimpleFoam.C:98-99 and :121-136):

  before the loop  setInitialDeltaT.H: if timeIndex == 0 and Co > SMALL, Time::setDeltaT(min(maxCo*dt/Co, min(dt, maxDeltaT)))
  every step       setDeltaT.H: fact = maxCo/(Co + SMALL); dt = min(min(fact, 1 + 0.1*fact), 1.2)*dt, min maxDeltaT,
                   through Time::setDeltaT, which applies Time::adjustDeltaT (Time.C:102-145) under adjustableRunTime:
                   timeToNextWrite = max(0, (writeTimeIndex+1)*writeInterval - (value - startTime)),
                   n = max(1, round(timeToNextWrite/dt)), newDt = timeToNextWrite/n, kept within [0.2 dt, 2 dt]
  ++runTime        value += dt; writeTimeIndex = label(((value - startTime) + 0.5*dt)/writeInterval) when larger

`1 + 0.1*fact` is a fused multiply-add in the aarch64 OpenFOAM these gates use (time_controls.cuh). --fused-write picks
whether adjustDeltaT's `(writeTimeIndex+1)*writeInterval - x` is one too; the gate replays OpenFOAM's own log first, so
the transcription is held to OpenFOAM before it is used on brae's. Exact: every step's replayed deltaT must equal the
logged one to the bit. Prints one line per mismatch (at most 10) and a summary; exit 0 when every step matches.
"""
import argparse, math, re, sys
from fractions import Fraction

SMALL = 1.0e-15
GREAT = 1.0e15


def fma(a, b, c):
    return float(Fraction(a) * Fraction(b) + Fraction(c))   # one rounding, as a fused multiply-add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--maxCo', type=float, required=True)
    ap.add_argument('--maxDeltaT', type=float, default=GREAT)
    ap.add_argument('--deltaT0', type=float, required=True)
    ap.add_argument('--start', type=float, required=True)
    ap.add_argument('--index', type=int, required=True)
    ap.add_argument('--writeControl', default='adjustable')
    ap.add_argument('--writeInterval', type=float, default=0.0)
    ap.add_argument('--fused-write', type=int, default=1)
    a = ap.parse_args()

    lines = open(a.log, errors='replace').read().splitlines()
    co_re = re.compile(r'^Courant Number mean: (\S+) max: (\S+)$')
    dt_re = re.compile(r'^deltaT = (\S+)$')
    pre, steps, last = None, [], None
    for l in lines:
        m = co_re.match(l)
        if m:
            last = float(m.group(2))
            continue
        m = dt_re.match(l)
        if m and last is not None:
            if pre is None and not steps:
                pass
            steps.append((last, float(m.group(1))))
            last = None
    # the pre-loop Courant number: the first one printed (CourantNo.H before setInitialDeltaT.H)
    for l in lines:
        m = co_re.match(l)
        if m:
            pre = float(m.group(2))
            break
    if pre is None or not steps:
        print('replay: no Courant/deltaT lines in the log'); return 2

    adjustable = a.writeControl in ('adjustable', 'adjustableRunTime')

    def adjust(dt, value, wti):
        if not adjustable:
            return dt
        x = value - a.start
        t2w = fma(float(wti + 1), a.writeInterval, -x) if a.fused_write else float(wti + 1) * a.writeInterval - x
        t2w = max(0.0, t2w)
        n = t2w / dt
        if not n < 2147483647.0:
            return dt
        k = max(1, int(math.floor(n + 0.5)) if n >= 0 else 1)   # OpenFOAM's round(): half away from zero
        new = t2w / k
        return min(new, 2.0 * dt) if new >= dt else max(new, 0.2 * dt)

    dt, value, wti = a.deltaT0, a.start, 0
    if a.index == 0 and pre > SMALL:
        dt = adjust(min(a.maxCo * dt / pre, min(dt, a.maxDeltaT)), value, wti)
    bad = 0
    for i, (co, logged) in enumerate(steps, 1):
        fact = a.maxCo / (co + SMALL)
        damped = fma(0.1, fact, 1.0)
        dt = min(min(min(fact, damped), 1.2) * dt, a.maxDeltaT)
        dt = adjust(dt, value, wti)
        if dt != logged:
            bad += 1
            if bad <= 10:
                print(f'  step {i}: replayed deltaT {dt!r}, logged {logged!r} (rel {abs(dt - logged) / logged:.2e})')
            dt = logged   # carry on from what the run did, so one difference is not counted at every later step
        value = value + dt
        if adjustable or a.writeControl == 'runTime':
            wi = int(fma(0.5, dt, value - a.start) / a.writeInterval) if a.fused_write else int(((value - a.start) + 0.5 * dt) / a.writeInterval)
            if wi > wti:
                wti = wi
    print(f'replay: {len(steps)} steps, {len(steps) - bad} to the bit, {bad} not')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
