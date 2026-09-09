#include "limit_temperature_report.cuh"
#include "io_precision.cuh"
#include <cmath>
#include <cstdio>

namespace brae {

void printLimitTemperature(
    const char* optionName,
    bool        isLower,
    label       limitedCells,
    label       totalCells,
    scalar      limitValue,
    scalar      unlimitedT)
{
    // limitTemperature.C:194-198 -- "Percent, max 2 decimal places", and zero when there are no cells
    // rather than a division by zero. Reproduced rather than approximated: the rounding is what makes
    // the printed number match OpenFOAM's on a partial line.
    const double pct = totalCells
                     ? 1e-2 * std::round(1e4 * double(limitedCells) / double(totalCells))
                     : 0.0;
    const int prec = ioPrecision();
    std::printf("limitTemperature=%s, Type=%s, LimitedCells=%d, CellsPercent=%.*g, %s=%.*g, %s=%.*g\n",
                optionName,
                isLower ? "Lower" : "Upper",
                static_cast<int>(limitedCells),
                prec, pct,
                isLower ? "Tmin" : "Tmax",
                prec, static_cast<double>(limitValue),
                isLower ? "UnlimitedTmin" : "UnlimitedTmax",
                prec, static_cast<double>(unlimitedT));
}

} // namespace brae
