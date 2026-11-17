// The registry behind proc_hooks.cuh's procGraphRegister/procGraphDestroyAll: CUDA graphs captured on a
// communicator's graph channel must be destroyed before the communicator is. See the header for why (NCCL's
// ncclCommDestroy waits for the graph's work to be globally quiescent).
#include "proc_hooks.cuh"
#include <map>
#include <vector>

namespace brae {
namespace {
struct Entry { void (*destroy)(void*); void* arg; };
std::map<const ProcComm*, std::vector<Entry>>& registry()
{
    static auto* r = new std::map<const ProcComm*, std::vector<Entry>>();   // leaked: no dtor after CUDA teardown
    return *r;
}
} // namespace

void procGraphRegister(const ProcComm* c, void (*destroy)(void*), void* arg)
{
    for (Entry& e : registry()[c]) if (e.arg == arg) { e.destroy = destroy; return; }   // re-registration replaces
    registry()[c].push_back({destroy, arg});
}

void procGraphUnregister(const ProcComm* c, void* arg)
{
    auto it = registry().find(c);
    if (it == registry().end()) return;
    auto& v = it->second;
    for (auto jt = v.begin(); jt != v.end(); ++jt) if (jt->arg == arg) { v.erase(jt); break; }
    if (v.empty()) registry().erase(it);
}

void procGraphDestroyAll(const ProcComm* c)
{
    auto it = registry().find(c);
    if (it == registry().end()) return;
    std::vector<Entry> entries = std::move(it->second);
    registry().erase(it);
    for (Entry& e : entries) e.destroy(e.arg);
}

} // namespace brae
