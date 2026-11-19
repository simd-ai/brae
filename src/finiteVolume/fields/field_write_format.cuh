#pragma once
// The field writer's two options (foam_field_writer.cuh has what each does), apart so that write_control.cuh can see whether the
// writer takes `writeFormat binary` without the writer's whole header. Set once by a driver, before its first write.
namespace brae {

struct FieldWriteFormat
{
    bool binary = false;   // OpenFOAM's binary lists and header (writeFormat binary)
    bool fast = false;     // ASCII formatted in blocks, each file written in one call, templates kept between writes
};
inline FieldWriteFormat& fieldWriteFormat()
{
    static FieldWriteFormat f;
    return f;
}

} // namespace brae
