#pragma once
// functionObjects::properties -- the function objects' state dictionary
// (src/OpenFOAM/db/functionObjects/functionObjectProperties/).
//
// ONE dictionary for the whole list: <time>/uniform/functionObjects/functionObjectProperties. The list reads
// it at the start time if it is there (READ_IF_PRESENT, functionObjectList.C:88-108) and writes it after
// every object has run at a write time, at 16 digits whatever writePrecision is (functionObjectList.C:
// 776-789). An object's RESULTS go under one section named by the SHA-1 of the word "results"
// (functionObjectProperties.C:33-34), then the object's name, then the value's type name, then the entry
// (setObjectResult, functionObjectPropertiesTemplates.C):
//
//     cdf7e925f5746741c316f5fbcf39ad0dfca90775
//     {
//         probes1
//         {
//             scalar { average(p) 1.188311576999574; ... }
//             label  { size(p) 1; }
//         }
//     }
//
// An entry set again is replaced where it stands (dictionary::add with merge), so the order is that of the
// first write. What a start file holds besides is carried along untouched, as OpenFOAM's dictionary would.
//
// THE FILE IS READ BY ITS OWN READER, not brae's dictionary reader: a keyword here is `average(p)`, one word
// to OpenFOAM's ISstream, which brae's tokenizer splits at the parenthesis. MEASURED on a continuation of
// laminar/damBreak, 2026-10-08: the carried `average(p_rgh)` came back as the key `average` with the value
// `( p_rgh ) 1.28...`, beside the continuation's own entry. A file OpenFOAM or brae wrote has its tokens
// apart -- a vector is `( x y z )` -- so white space is the separator here.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "foam_token_reader.cuh"
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace brae {

class FunctionObjectProperties
{
public:
    // the dictionary is written at this precision (functionObjectList.C:779-780)
    static constexpr int writePrecision = 16;

    static const char* resultsName()
    {
        return "cdf7e925f5746741c316f5fbcf39ad0dfca90775";
    }

    static std::string token(scalar v)
    {
        std::ostringstream o;
        o.precision(writePrecision);
        o << v;
        return o.str();
    }

    // a vector is three tokens between two more: primitiveEntry writes its tokens a space apart
    // (primitiveEntryIO.C:280-314), `( x y z )`
    static std::string token(const vector& v)
    {
        return "( " + token(v.x) + " " + token(v.y) + " " + token(v.z) + " )";
    }

    static std::string token(label v)
    {
        return std::to_string(v);
    }

    // stateFunctionObject::setResult -> properties::setObjectResult
    void setResult(
        const std::string& objectName,
        const std::string& typeName,
        const std::string& entryName,
        const std::string& value)
    {
        Node& entry = child(child(child(child(root_, resultsName()), objectName), typeName), entryName);
        entry.leaf = true;
        entry.value = value;
    }

    // The start time's file, if there is one (READ_IF_PRESENT). Numbers come back as numbers and are written
    // at this dictionary's own precision; every other token as it stands.
    void read(const std::string& path)
    {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec) && !std::filesystem::exists(path + ".gz", ec))
        {
            return;
        }
        const std::vector<char> bytes = gzSlurp(path);
        const std::vector<std::string> tokens = tokensOf(std::string(bytes.begin(), bytes.end()));
        root_.children.clear();
        std::size_t at = 0;
        take(tokens, at, root_, path);
        // the header is not an entry of the dictionary
        for (std::size_t i = 0; i < root_.children.size(); ++i)
        {
            if (root_.children[i].name == "FoamFile")
            {
                root_.children.erase(root_.children.begin() + static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
    }

    // The entries as dictionary::writeEntries emits them at the top level of a file (dictionaryIO.C:180-208):
    // an empty line BETWEEN two entries, none after the last. Empty for an empty dictionary.
    std::string body() const
    {
        std::string out;
        for (const Node& n : root_.children)
        {
            if (!out.empty())
            {
                out += "\n";
            }
            emit(n, 0, out);
        }
        return out;
    }

    bool empty() const
    {
        return root_.children.empty();
    }

private:
    struct Node
    {
        std::string name;
        bool leaf = false;
        std::string value;
        std::vector<Node> children;
    };

    static Node& child(
        Node& parent,
        const std::string& name)
    {
        for (Node& c : parent.children)
        {
            if (c.name == name)
            {
                return c;
            }
        }
        Node n;
        n.name = name;
        parent.children.push_back(n);
        return parent.children.back();
    }

    // the file's tokens: comments out, then white space apart, with `{`, `}` and a closing `;` on their own
    static std::vector<std::string> tokensOf(const std::string& text)
    {
        std::string clean;
        for (std::size_t i = 0; i < text.size(); ++i)
        {
            if (text.compare(i, 2, "//") == 0)
            {
                while (i < text.size() && text[i] != '\n')
                {
                    ++i;
                }
                clean += '\n';
            }
            else if (text.compare(i, 2, "/*") == 0)
            {
                const std::size_t end = text.find("*/", i + 2);
                i = end == std::string::npos ? text.size() : end + 1;
                clean += ' ';
            }
            else
            {
                clean += text[i];
            }
        }
        std::vector<std::string> tokens;
        std::istringstream in(clean);
        std::string t;
        while (in >> t)
        {
            if (t.size() > 1 && t.back() == ';')
            {
                tokens.push_back(t.substr(0, t.size() - 1));
                tokens.push_back(";");
            }
            else
            {
                tokens.push_back(t);
            }
        }
        return tokens;
    }

    // One token as OpenFOAM writes it back (ISstream.C:712-790, as the solver's writer re-emits a dictionary,
    // inter_writer_cpp.cu dictToken): a label as its value, any other number as a scalar at this
    // dictionary's precision, a word unchanged; a leading `+` does not start a number
    static std::string echoed(const std::string& t)
    {
        if (t.empty() || t[0] == '+')
        {
            return t;
        }
        char* end = nullptr;
        const bool digits = t.find_first_not_of("-0123456789") == std::string::npos && t != "-";
        if (digits)
        {
            const long long v = std::strtoll(t.c_str(), &end, 10);
            if (end && *end == '\0' && v >= -2147483648LL && v <= 2147483647LL)
            {
                return std::to_string(v);
            }
        }
        const double v = std::strtod(t.c_str(), &end);
        if (end && end != t.c_str() && *end == '\0')
        {
            return token(static_cast<scalar>(v));
        }
        return t;
    }

    static void take(
        const std::vector<std::string>& tokens,
        std::size_t& at,
        Node& into,
        const std::string& path)
    {
        while (at < tokens.size() && tokens[at] != "}")
        {
            const std::string name = tokens[at++];
            if (at < tokens.size() && tokens[at] == "{")
            {
                ++at;
                // by index: the recursion may add to this dictionary's own list
                child(into, name);
                std::size_t k = 0;
                while (into.children[k].name != name)
                {
                    ++k;
                }
                take(tokens, at, into.children[k], path);
                if (at >= tokens.size())
                {
                    throw std::runtime_error("brae: " + path + ": the dictionary `" + name + "` is not closed.");
                }
                ++at;
            }
            else
            {
                std::string value;
                while (at < tokens.size() && tokens[at] != ";")
                {
                    value += (value.empty() ? "" : " ") + echoed(tokens[at++]);
                }
                if (at >= tokens.size())
                {
                    throw std::runtime_error("brae: " + path + ": the entry `" + name + "` has no `;`.");
                }
                ++at;
                Node& entry = child(into, name);
                entry.leaf = true;
                entry.value = value;
            }
        }
    }

    // primitiveEntry and dictionaryEntry as they are written: Ostream::writeKeyword pads the keyword to
    // sixteen columns with at least one space (Ostream.C:67-94), four spaces a level
    static void emit(
        const Node& n,
        int level,
        std::string& out)
    {
        const std::string indent(static_cast<std::size_t>(4*level), ' ');
        if (n.leaf)
        {
            std::string key = n.name;
            key.append(key.size() < 15 ? 16 - key.size() : 1, ' ');
            out += indent + key + n.value + ";\n";
            return;
        }
        out += indent + n.name + "\n" + indent + "{\n";
        for (const Node& c : n.children)
        {
            emit(c, level + 1, out);
        }
        out += indent + "}\n";
    }

    Node root_;
};

}   // namespace brae
