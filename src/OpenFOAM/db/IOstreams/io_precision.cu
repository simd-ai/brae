#include "io_precision.cuh"

namespace brae {

namespace {
int& precisionRef()
{
    static int p = 6;   // IOstream.H, precision_ = 6
    return p;
}
}


void setIOPrecision(int writePrecision)
{
    if (writePrecision > 0 && writePrecision <= 40) precisionRef() = writePrecision;
}


int ioPrecision()
{
    return precisionRef();
}

} // namespace brae
