#pragma once
// Minimal OpenFOAM ASCII reader. Strips the banner / comments and the FoamFile header
// dict, then exposes the payload as a flat token stream with typed cursor reads. Adjacent
// punctuation is split ( '3(0 1 2)' -> '3' '(' '0' '1' '2' ')' ), so polyMesh face lists
// and (x y z) point tuples parse uniformly. Sufficient for polyMesh + simple field files.
#include "cf_types.cuh"
#include <string>
#include <vector>
#include <utility>
#include <cstddef>

namespace brae {

// Slurp a whole file, transparently handling gzip and resolving <base> -> <base>.gz when <base> is absent
// (OF writeCompression). maxBytes>0 stops early (header sniffing). Throws "cannot open" if neither exists.
std::vector<char> gzSlurp(const std::string& base, std::size_t maxBytes = 0);

// Format of a foam file ("ascii" or "binary") from its FoamFile header `format` entry.
std::string foamFormat(const std::string& path);

// Binary polyMesh list readers (used when foamFormat(path) == "binary"). The boundary file
// is always an ASCII dictionary, so it uses TokenStream regardless of format.
std::vector<vector> readBinaryPoints(const std::string& path);
std::vector<label>  readBinaryLabelList(const std::string& path);
void readBinaryCompactFaces(const std::string& path, std::vector<label>& offsets, std::vector<label>& verts);

// Binary constant/polyMesh/cellZones reader. cellZones is a MIXED file: the ZoneMesh structure
// (nZones, zone names, `{ type cellZone; cellLabels ... }`) is ASCII, but each zone's cellLabels
// list payload is a raw binary blob. The ASCII TokenStream mangles that blob, so binary meshes
// need this dedicated parser. Returns (zoneName -> cell labels) in file order.
std::vector<std::pair<std::string, std::vector<label>>> readBinaryCellZones(const std::string& path);

class TokenStream
{
public:
    // expandVars: also expand OpenFOAM in-file $variable macros (e.g. `Uinlet (0 1 0); ... $Uinlet`) after the
    // #include expansion. Off by default (mesh/polyMesh files have no macros and skip the cost); the field reader
    // turns it on so `0/` field files using $-macros parse like OF.
    explicit TokenStream(const std::string& path, bool expandVars = false);

    bool               eof()  const { return pos_ >= toks_.size(); }
    const std::string& peek() const;
    std::string        next();
    void               expect(const std::string& t);
    label              nextLabel();
    scalar             nextScalar();
    // A `#{ ... #}` verbatim block is one token, a placeholder word; this returns the block's raw text
    // (comments, quotes and `$` intact, delimiters excluded) for such a token, false for any other.
    bool               verbatim(const std::string& token, std::string& body) const;

private:
    std::vector<std::string> toks_;
    std::vector<std::string> verbatim_;   // the raw bodies, indexed by the placeholder's number
    std::size_t              pos_ = 0;
};

} // namespace brae
