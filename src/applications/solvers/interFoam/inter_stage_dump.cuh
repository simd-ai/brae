#pragma once
// The stage dump of the multi-GPU operator gate (BRAE_MGPU_STAGE_DUMP=<dir>, inter_parallel.cuh): the first step's
// explicit stages, one OpenFOAM-format ascii field per quantity, written per rank so that tools/mgpu_oracle's
// compare_fields.py reads them next to OpenFOAM's own dump of the same quantities (tools/dumpInterFoam stageDump.H).
//
//   <dir>/processor<rank>/<time>/<name>.dump      under -parallel
//   <dir>/<time>/<name>.dump                      serial (the np = 1 floor)
//
// volScalarField: internalField, then one boundaryField entry per patch (the face cell's value, `calculated`;
// processor patches carry no value). surfaceScalarField: internalField = the internal faces, one entry per patch
// with its faces' values (the processor patches' too: a cut face takes its value from there in the comparator),
// `oriented oriented` for fluxes. 17 significant digits: the text is exact.
#include "cf_types.cuh"
#include "fv_patch.cuh"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace interFoam {

class StageDumper
{
public:
    // `root` is the BRAE_MGPU_STAGE_DUMP value, `time` the start time directory's name
    StageDumper(const std::string& root, int rank, bool parallel, const std::string& time, const std::vector<FvPatch>& patches)
        : patches_(patches)
    {
        std::filesystem::path p(root);
        if (parallel) p /= "processor" + std::to_string(rank);
        p /= time;
        std::filesystem::create_directories(p);
        dir_ = p.string();
    }

    // cells: nCells values; the patch values are `bnd` (nBndFaces, patch order) where given, else the face cell's
    void vol(const std::string& name, const std::vector<scalar>& cells, const std::vector<scalar>* bnd = nullptr)
    {
        std::size_t at = 0;
        std::ofstream o = open(name, "volScalarField");
        o << "dimensions      [0 0 0 0 0 0 0];\n\n";
        list(o, "internalField   nonuniform List<scalar> ", cells.data(), cells.size());
        o << ";\n\nboundaryField\n{\n";
        for (const FvPatch& q : patches_)
        {
            o << "    " << q.name << "\n    {\n        type            " << q.type << ";\n";
            if (q.type != "processor" && q.size > 0)
            {
                std::vector<scalar> v(static_cast<std::size_t>(q.size));
                for (label i = 0; i < q.size; ++i)
                {
                    v[static_cast<std::size_t>(i)] = bnd ? (*bnd)[at + static_cast<std::size_t>(i)]
                                                         : cells[static_cast<std::size_t>(q.faceCells[static_cast<std::size_t>(i)])];
                }
                list(o, "        value           nonuniform List<scalar> ", v.data(), v.size());
                o << ";\n";
            }
            o << "    }\n";
            at += static_cast<std::size_t>(q.size);
        }
        o << "}\n";
    }

    // internal: nInternalFaces values; bnd: the nBndFaces values of every patch in patch order (processor faces last)
    // `oriented`: a flux-like field whose sign follows the face's orientation (OpenFOAM's `oriented oriented`); a
    // magnitude such as phic is not
    void surface(const std::string& name, const std::vector<scalar>& internal, const std::vector<scalar>& bnd,
                 bool oriented = true)
    {
        std::ofstream o = open(name, "surfaceScalarField");
        o << "dimensions      [0 0 0 0 0 0 0];\n" << (oriented ? "oriented        oriented;\n" : "") << "\n";
        list(o, "internalField   nonuniform List<scalar> ", internal.data(), internal.size());
        o << ";\n\nboundaryField\n{\n";
        std::size_t at = 0;
        for (const FvPatch& q : patches_)
        {
            const std::size_t n = static_cast<std::size_t>(q.size);
            if (at + n > bnd.size())
            {
                throw std::runtime_error("stage dump: the boundary array of `" + name + "` is shorter than the patches.");
            }
            o << "    " << q.name << "\n    {\n        type            " << q.type << ";\n";
            if (n > 0)
            {
                list(o, "        value           nonuniform List<scalar> ", bnd.data() + at, n);
                o << ";\n";
            }
            o << "    }\n";
            at += n;
        }
        o << "}\n";
    }

    const std::string& dir() const { return dir_; }

private:
    std::ofstream open(const std::string& name, const char* cls)
    {
        std::ofstream o(dir_ + "/" + name + ".dump");
        if (!o) throw std::runtime_error("stage dump: cannot write " + dir_ + "/" + name + ".dump");
        o << std::setprecision(17);
        o << "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       " << cls
          << ";\n    object      " << name << ".dump;\n}\n\n";
        return o;
    }
    static void list(std::ofstream& o, const char* head, const scalar* v, std::size_t n)
    {
        o << head << n << "\n(\n";
        for (std::size_t i = 0; i < n; ++i) o << v[i] << "\n";
        o << ")";
    }
    std::string dir_;
    const std::vector<FvPatch>& patches_;
};

}   // namespace interFoam
}   // namespace cpu
}   // namespace brae
