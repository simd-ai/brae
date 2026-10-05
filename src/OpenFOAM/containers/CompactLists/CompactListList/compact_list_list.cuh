// CompactListList<label>: a list of lists held as ONE array of values and one of row offsets. OpenFOAM's own
// class of the name (src/OpenFOAM/containers/CompactLists/CompactListList/CompactListList.H: offsets_, values_,
// operator[] a SubList, unpack() the list of lists).
// OpenFOAM's primitiveMesh keeps its addressing as labelListList -- a List per row. brae's refinement kept ten
// of those as std::vector<std::vector<label>>: a heap block a row, a million of them on a 90,000-cell mesh,
// built again after every change of topology. MEASURED on damBreakWithObstacle: one build of the ten lists was
// about 100 ms, and where the allocator happened to put the blocks moved every later stage that walks them by a
// third either way. Held compact, a build is two arrays a list.
#pragma once

#include "cf_types.cuh"
#include <cstddef>
#include <vector>

namespace brae {

// one row, wherever it lives: what operator[] returns (OpenFOAM's SubList)
struct LabelRow
{
    const label* first = nullptr;
    std::size_t  n = 0;

    LabelRow() = default;
    LabelRow(
        const label* p,
        std::size_t  count)
    :
        first(p),
        n(count)
    {}
    // a std::vector<label> read as a row: the vector must outlive the row (an argument does)
    LabelRow(const std::vector<label>& v) : first(v.data()), n(v.size()) {}

    const label* begin() const { return first; }
    const label* end() const { return first + n; }
    std::size_t size() const { return n; }
    bool empty() const { return n == 0; }
    const label& operator[](std::size_t i) const { return first[i]; }
    const label& front() const { return first[0]; }
    const label& back() const { return first[n - 1]; }
};

inline bool operator==(
    const LabelRow& a,
    const LabelRow& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i]) return false;
    }
    return true;
}
inline bool operator!=(
    const LabelRow& a,
    const LabelRow& b)
{
    return !(a == b);
}

class CompactListList
{
public:
    // rows: size(); entries: totalSize()
    std::size_t size() const { return offsets_.empty() ? 0 : offsets_.size() - 1; }
    std::size_t totalSize() const { return values_.size(); }
    bool empty() const { return size() == 0; }

    LabelRow operator[](std::size_t i) const
    {
        const std::size_t b = static_cast<std::size_t>(offsets_[i]);
        return LabelRow{values_.data() + b, static_cast<std::size_t>(offsets_[i + 1]) - b};
    }
    std::size_t localSize(std::size_t i) const
    {
        return static_cast<std::size_t>(offsets_[i + 1] - offsets_[i]);
    }

    const std::vector<label>& offsets() const { return offsets_; }
    const std::vector<label>& values() const { return values_; }
    std::vector<label>& values() { return values_; }

    void clear()
    {
        offsets_.clear();
        values_.clear();
    }

    // ROWS APPENDED IN ORDER: start(), then each row's entries with append() and its end with endRow()
    void start(
        std::size_t nRowsHint,
        std::size_t nValuesHint)
    {
        offsets_.clear();
        values_.clear();
        offsets_.reserve(nRowsHint + 1);
        values_.reserve(nValuesHint);
        offsets_.push_back(label(0));
    }
    void append(label v) { values_.push_back(v); }
    void endRow() { offsets_.push_back(static_cast<label>(values_.size())); }
    // a whole row at once
    void appendRow(LabelRow r)
    {
        values_.insert(values_.end(), r.begin(), r.end());
        endRow();
    }
    // the row being appended, to sort or scan before endRow()
    label* openRowBegin() { return values_.data() + static_cast<std::size_t>(offsets_.back()); }
    label* openRowEnd() { return values_.data() + values_.size(); }
    void truncateOpenRow(label* newEnd) { values_.resize(static_cast<std::size_t>(newEnd - values_.data())); }

    // ROWS OF KNOWN SIZES, filled in any order: resize(listSizes) (CompactListList.C). Returns the fill
    // cursor of each row -- the caller writes values()[cursor[row]++].
    std::vector<label> setSizes(const std::vector<label>& sizes)
    {
        offsets_.assign(sizes.size() + 1, label(0));
        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            offsets_[i + 1] = offsets_[i] + sizes[i];
        }
        values_.assign(static_cast<std::size_t>(offsets_.back()), label(0));
        return std::vector<label>(offsets_.begin(), offsets_.end() - 1);
    }
    // ...and with the offsets given whole (a list shaped like another one)
    void setOffsets(std::vector<label> offsets)
    {
        offsets_ = std::move(offsets);
        values_.assign(offsets_.empty() ? 0 : static_cast<std::size_t>(offsets_.back()), label(0));
    }

    // unpack(): the list of lists
    std::vector<std::vector<label>> unpack() const
    {
        std::vector<std::vector<label>> out(size());
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            const LabelRow r = (*this)[i];
            out[i].assign(r.begin(), r.end());
        }
        return out;
    }

    bool operator==(const CompactListList& o) const { return offsets_ == o.offsets_ && values_ == o.values_; }
    bool operator!=(const CompactListList& o) const { return !(*this == o); }

    // row for row what a list of lists holds
    bool sameAs(const std::vector<std::vector<label>>& l) const
    {
        if (l.size() != size()) return false;
        for (std::size_t i = 0; i < l.size(); ++i)
        {
            const LabelRow r = (*this)[i];
            if (r.size() != l[i].size()) return false;
            for (std::size_t k = 0; k < r.size(); ++k)
            {
                if (r[k] != l[i][k]) return false;
            }
        }
        return true;
    }

private:
    std::vector<label> offsets_;
    std::vector<label> values_;
};

// A LIST OF LISTS BY REFERENCE, IN EITHER FORM: what a routine that only READS one takes. The refinement holds
// the compact form; the tests' fixtures and the solver's other users hold std::vector<std::vector<label>>
// (meshCells, pointCellsFromCells, ...) and call the same routines. The referenced list must outlive this.
class LabelListListRef
{
public:
    LabelListListRef() = default;
    LabelListListRef(const std::vector<std::vector<label>>& l) : nested_(&l) {}
    LabelListListRef(const CompactListList& l) : compact_(&l) {}
    // ...or a pair of arrays that already ARE the compact form: n + 1 offsets and the values (a mesh's own
    // faceOffsets and faceVerts)
    LabelListListRef(
        const std::vector<label>& offsets,
        const std::vector<label>& values)
    :
        offsets_(&offsets),
        values_(&values)
    {}

    bool null() const { return nested_ == nullptr && compact_ == nullptr && offsets_ == nullptr; }
    const void* address() const
    {
        return nested_ ? static_cast<const void*>(nested_)
             : compact_ ? static_cast<const void*>(compact_)
             : static_cast<const void*>(offsets_);
    }
    std::size_t size() const
    {
        if (nested_) return nested_->size();
        if (compact_) return compact_->size();
        return offsets_->empty() ? 0 : offsets_->size() - 1;
    }
    LabelRow operator[](std::size_t i) const
    {
        if (nested_)
        {
            const std::vector<label>& r = (*nested_)[i];
            return LabelRow{r.data(), r.size()};
        }
        if (compact_) return (*compact_)[i];
        const std::size_t b = static_cast<std::size_t>((*offsets_)[i]);
        return LabelRow{values_->data() + b, static_cast<std::size_t>((*offsets_)[i + 1]) - b};
    }
    // the two arrays of the compact forms -- n + 1 offsets, and the values -- or null for a list of lists: for
    // a reader that takes the whole list in one copy
    const std::vector<label>* flatOffsets() const
    {
        return compact_ ? &compact_->offsets() : offsets_;
    }
    const std::vector<label>* flatValues() const
    {
        return compact_ ? &compact_->values() : values_;
    }

private:
    const std::vector<std::vector<label>>* nested_ = nullptr;
    const CompactListList*                 compact_ = nullptr;
    const std::vector<label>*              offsets_ = nullptr;
    const std::vector<label>*              values_ = nullptr;
};

// ...and the nullable pointer to one that a mesh view holds (hexRef8's MeshView, removeFaces' RemoveFacesView):
// assigned `&list` of either form, tested against null, dereferenced for the list.
class LabelListListPtr
{
public:
    LabelListListPtr() = default;
    LabelListListPtr(std::nullptr_t) {}
    LabelListListPtr(const std::vector<std::vector<label>>* l)
    {
        if (l) ref_ = LabelListListRef(*l);
    }
    LabelListListPtr(const CompactListList* l)
    {
        if (l) ref_ = LabelListListRef(*l);
    }

    const LabelListListRef& operator*() const { return ref_; }
    const LabelListListRef* operator->() const { return &ref_; }
    operator const void*() const { return ref_.address(); }

private:
    LabelListListRef ref_;
};

} // namespace brae
