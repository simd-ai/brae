// Reading an OpenFOAM BINARY field, and the macro-capture bug that finding it exposed.
//
// OF's binary format is a hybrid: the usual ASCII dictionary, except that a list of a PRIMITIVE type is
// written as  `List<vector> \n N \n ( <N*3 raw doubles> )`.  Word lists (`inGroups 1(wall)`) stay ASCII
// even in a binary file. brae already had dedicated binary readers for the polyMesh; fields had none,
// and pimpleFoam/LES/NACA4412 ships a binary 0/U (its Allrun writes it), so brae stopped with
// "expected '(' got <raw bytes>".
//
// Rather than a second dedicated parser, the payloads are transcoded to ASCII at read time, so the
// existing field reader -- with its #includes, $macros, boundaryField handling and regex patch matching
// -- is reused unchanged, and ANY binary foam dictionary works, not just the one file type.
//
// LEG 3 IS THE INTERESTING ONE. Getting past the binary file exposed a second, older bug: the $macro
// capture stopped at '{', so
//     internalField   uniform #eval{ 3.0/1520000.0 };
// registered as the value "uniform #eval", and every `$internalField` in the boundaryField expanded to a
// #eval with no expression. The error surfaced at the USE site, several patches later, naming the
// directive rather than the definition -- which is why 0/nut failed while 0/k, written identically,
// happened not to. That is the same class of scanner bug as the polyMesh `scaleCoeffs` and field-reader
// sub-dictionary ones: a delimiter search that ignores nesting.
#include "foam_field_reader.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

// The COUNT in the header is the number of ELEMENTS, not of doubles -- writing the double count for a
// vector field made the payload three times shorter than declared, and the reader (correctly) left the
// blob alone rather than running off the end.
void writeBinaryField(const std::string& path, const char* cls, const std::vector<double>& vals, int nCmpt)
{
    std::ofstream o(path, std::ios::binary);
    o << "FoamFile\n{\n    version     2.0;\n    format      binary;\n    class       " << cls
      << ";\n    object      f;\n}\n"
      << "dimensions      [0 1 -1 0 0 0 0];\n\n"
      << "internalField   nonuniform List<" << (nCmpt == 3 ? "vector" : "scalar") << "> \n"
      << (vals.size()/nCmpt) << "\n(";
    o.write(reinterpret_cast<const char*>(vals.data()), (std::streamsize)(vals.size()*sizeof(double)));
    o << ")\n;\n\nboundaryField\n{\n    inlet\n    {\n        type            zeroGradient;\n    }\n}\n";
}
}   // namespace

int main()
{
    const char* tmp = std::getenv("TMPDIR");
    const std::string dir = std::string(tmp ? tmp : "/tmp") + "/brae_binfield";
    std::system(("rm -rf " + dir + " && mkdir -p " + dir).c_str());

    // ---- 1. a binary scalar field ----
    {
        std::vector<double> v(97);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.5 + 0.25*std::sin(0.7*double(i));
        writeBinaryField(dir + "/p", "volScalarField", v, 1);
        const FieldData<scalar> fd = readField<scalar>(dir + "/p");
        std::printf("  binary scalar: read %zu values (wrote %zu)\n", fd.internalField.size(), v.size());
        if (fd.internalField.size() != v.size())
        { std::printf("  FAIL wrong count -- the raw payload was not decoded\n"); ++failures; }
        else
        {
            double w = 0;
            for (std::size_t i = 0; i < v.size(); ++i) w = std::fmax(w, std::fabs(fd.internalField[i] - v[i]));
            std::printf("  binary scalar: max|read - written| = %.3e\n", w);
            if (w > 1e-12) { std::printf("  FAIL the decoded values are wrong\n"); ++failures; }
        }
    }

    // ---- 2. a binary VECTOR field (the component interleaving is where a decoder goes wrong) ----
    {
        std::vector<double> v(3*61);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = -1.0 + 0.013*double(i);
        writeBinaryField(dir + "/U", "volVectorField", v, 3);
        const FieldData<vector> fd = readField<vector>(dir + "/U");
        std::printf("  binary vector: read %zu values (wrote %zu)\n", fd.internalField.size(), v.size()/3);
        if (fd.internalField.size() != v.size()/3)
        { std::printf("  FAIL wrong count\n"); ++failures; }
        else
        {
            double w = 0;
            for (std::size_t c = 0; c < fd.internalField.size(); ++c)
            {
                w = std::fmax(w, std::fabs(fd.internalField[c].x - v[3*c+0]));
                w = std::fmax(w, std::fabs(fd.internalField[c].y - v[3*c+1]));
                w = std::fmax(w, std::fabs(fd.internalField[c].z - v[3*c+2]));
            }
            std::printf("  binary vector: max|read - written| = %.3e\n", w);
            if (w > 1e-12)
            { std::printf("  FAIL components are transposed or misaligned\n"); ++failures; }
        }
    }

    // ---- 3. $macro capture must include a #eval{...} body ----
    {
        std::ofstream o(dir + "/k");
        o << "FoamFile { version 2.0; format ascii; class volScalarField; object k; }\n"
          << "dimensions      [0 2 -2 0 0 0 0];\n"
          << "internalField   uniform #eval{ 3.0/1520000.0 };\n"
          << "boundaryField\n{\n"
          << "    inlet\n    {\n        type            fixedValue;\n        value           $internalField;\n    }\n"
          << "    outlet\n    {\n        type            zeroGradient;\n    }\n}\n";
        o.close();
        bool threw = false;
        std::string msg;
        FieldData<scalar> fd;
        try { fd = readField<scalar>(dir + "/k"); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        if (threw)
        {
            std::printf("  FAIL $internalField carrying a #eval body did not parse: %s\n", msg.c_str());
            std::printf("       The value capture must treat a '{' after a #directive as that directive's\n"
                        "       BODY, not as the start of a sub-dictionary.\n");
            ++failures;
        }
        else
        {
            const scalar want = 3.0/1520000.0;
            std::printf("  #eval macro: internalField = %.10e (want %.10e), %zu patches\n",
                        (double)fd.internalUniformValue, (double)want, fd.boundary.size());
            if (std::fabs(fd.internalUniformValue - want) > 1e-15)
            { std::printf("  FAIL the #eval itself did not evaluate\n"); ++failures; }
            if (fd.boundary.size() != 2)
            { std::printf("  FAIL the patches after it were swallowed (%zu of 2)\n", fd.boundary.size()); ++failures; }
            else
            {
                const auto& b = fd.boundary[0];
                const scalar got = b.valueUniform ? b.uniformValue : (b.values.empty() ? scalar(0) : b.values[0]);
                std::printf("  #eval macro: inlet value = %.10e\n", (double)got);
                if (std::fabs(got - want) > 1e-15)
                {
                    std::printf("  FAIL $internalField expanded to something other than the evaluated value --\n"
                                "       the capture truncated at the '{'\n");
                    ++failures;
                }
            }
        }
    }

    // ---- 4. an EMPTY list in a binary file: its count alone, no parentheses ----
    // OpenFOAM writes a binary list's bytes and delimiters only when it is not empty (UListIO.C:92-102), so a patch with no face
    // on a decomposePar part reads `value nonuniform List<scalar> \n0\n;`. brae asked for '(' and refused the file (Arrhenius
    // 5144475). The patch must read with no value, the patches around it with theirs; CONTROL BRAE_CONTROL_BINARY_EMPTY_LIST_ASIS
    // (the list copied through as before) must make the read throw
    {
        const std::vector<double> in = {1.5, -2.25, 3.125};
        const std::vector<double> last = {7.0, 8.5};
        const auto write = [&](const std::string& path)
        {
            std::ofstream o(path, std::ios::binary);
            o << "FoamFile\n{\n    version     2.0;\n    format      binary;\n    class       volScalarField;\n    object      nut;\n}\n"
              << "dimensions      [0 2 -1 0 0 0 0];\n\n"
              << "internalField   nonuniform List<scalar> \n" << in.size() << "\n(";
            o.write(reinterpret_cast<const char*>(in.data()), (std::streamsize)(in.size() * sizeof(double)));
            o << ")\n;\n\nboundaryField\n{\n"
              << "    inlet\n    {\n        type            calculated;\n        value           nonuniform List<scalar> \n0\n;\n    }\n"
              << "    wall\n    {\n        type            calculated;\n        value           nonuniform List<scalar> \n" << last.size() << "\n(";
            o.write(reinterpret_cast<const char*>(last.data()), (std::streamsize)(last.size() * sizeof(double)));
            o << ")\n;\n    }\n}\n";
        };
        write(dir + "/nut");
        try
        {
            const FieldData<scalar> fd = readField<scalar>(dir + "/nut");
            const bool shape = fd.internalField.size() == in.size() && fd.boundary.size() == 2 && !fd.boundary[0].valueUniform
                               && fd.boundary[0].values.empty() && fd.boundary[1].values.size() == last.size();
            const bool values = shape && fd.internalField[1] == in[1] && fd.boundary[1].values[0] == last[0] && fd.boundary[1].values[1] == last[1];
            std::printf("  binary empty list: %zu cells, %zu patches, the empty one %zu values, the next %zu\n", fd.internalField.size(),
                        fd.boundary.size(), fd.boundary.empty() ? 0 : fd.boundary[0].values.size(),
                        fd.boundary.size() < 2 ? 0 : fd.boundary[1].values.size());
            if (!values) { std::printf("  FAIL the empty patch or the patch after it read wrongly\n"); ++failures; }
        }
        catch (const std::exception& e)
        {
            std::printf("  FAIL a binary empty list refused: %s\n", e.what());
            ++failures;
        }
        setenv("BRAE_CONTROL_BINARY_EMPTY_LIST_ASIS", "1", 1);
        bool threw = false;
        try
        {
            readField<scalar>(dir + "/nut");
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        unsetenv("BRAE_CONTROL_BINARY_EMPTY_LIST_ASIS");
        std::printf("  CONTROL BRAE_CONTROL_BINARY_EMPTY_LIST_ASIS: the read %s\n", threw ? "throws (red, as it must)" : "PASSED");
        if (!threw) { std::printf("  FAIL the control did not break the read: the leg does not see the fix\n"); ++failures; }
    }

    std::printf("binary_field_read: %d failures\n", failures);
    return failures ? 1 : 0;
}
