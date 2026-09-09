#pragma once
// IOstream::defaultPrecision() -- the precision every OpenFOAM `Info` line is written at.
//
// provenance:
//   openfoam: src/OpenFOAM/db/IOstreams/IOstreams/IOstream.H (precision_, default 6)
//             src/OpenFOAM/db/Time/TimeIO.C:375-383 (writePrecision -> defaultPrecision -> Sout)
//   brae:     src/OpenFOAM/db/IOstreams/io_precision.cu
//
// OpenFOAM keeps ONE global for this: Time::readDict sets IOstream::defaultPrecision() from the case's
// `writePrecision` and re-points Sout, Serr, Pout and Perr at it, and Info writes through Sout. So every
// diagnostic brae emits to be line-comparable with OpenFOAM's log -- Foam::bound's `bounding <field>`,
// fv::limitTemperature's `LimitedCells` pair, and whatever comes next -- shares this one number rather
// than each carrying a static of its own.

namespace brae {

// The case's `writePrecision`, applied once where controlDict is read. Values outside what printf can
// use are ignored: OpenFOAM applies the entry as given and a nonsensical one is the case's problem, but
// a precision printf cannot honour would corrupt every line rather than one number.
void setIOPrecision(int writePrecision);

// 6 until a case says otherwise -- IOstream's own default.
int ioPrecision();

} // namespace brae
