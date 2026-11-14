// The libcurl transport. Deliberately the only file in the agent that knows curl exists, and deliberately free
// of protocol logic -- that all lives in api_client.cpp, above the HttpClient interface, where it is tested
// against a fake.
//
// The security posture from docs/10-security.md is set here, and none of it is configurable:
//
//   * TLS verification is always on. There is no flag, no environment variable, no debug switch to turn it off,
//     because that switch is precisely what an attacker talks a contributor into setting.
//   * Redirects are never followed. A 302 would hand this node's bearer token to whatever host the redirect
//     names, and curl will happily do that if asked.
//   * Bodies are capped and requests are time-bounded, so a hung or hostile server cannot exhaust the node.
//   * The Authorization header exists only inside this function; it is never logged.
#include "http_curl.h"

#include <curl/curl.h>

#include <cstring>
#include <mutex>

namespace brae::node {
namespace {

std::once_flag gInit;

std::size_t writeCallback(char* ptr, std::size_t size, std::size_t nmemb, void* userdata)
{
    auto* out = static_cast<std::string*>(userdata);
    const std::size_t n = size * nmemb;
    if (out->size() + n > kMaxResponseBytes)
    {
        // Returning short tells curl to abort the transfer. A server that streams forever is a server we stop
        // listening to, rather than one that decides how much memory this machine spends.
        const std::size_t room = kMaxResponseBytes > out->size() ? kMaxResponseBytes - out->size() : 0;
        out->append(ptr, room);
        return 0;
    }
    out->append(ptr, n);
    return n;
}

}  // namespace

CurlHttp::CurlHttp()
{
    std::call_once(gInit, [] { ::curl_global_init(CURL_GLOBAL_DEFAULT); });
}

HttpResponse CurlHttp::send(const HttpRequest& req)
{
    HttpResponse res;

    CURL* curl = ::curl_easy_init();
    if (!curl) { res.error = "curl_easy_init failed"; return res; }

    struct curl_slist* headers = nullptr;
    std::string authHeader;                       // lives until the request completes; never logged
    if (!req.bearer.empty())
    {
        authHeader = "Authorization: Bearer " + req.bearer;
        headers = ::curl_slist_append(headers, authHeader.c_str());
    }
    headers = ::curl_slist_append(headers, "Content-Type: application/json");
    headers = ::curl_slist_append(headers, "Accept: application/json");
    headers = ::curl_slist_append(headers, ("X-Brae-Node: " + nodeId_).c_str());
    headers = ::curl_slist_append(headers, "Expect:");     // no 100-continue round trip on small POSTs

    ::curl_easy_setopt(curl, CURLOPT_URL, req.url.c_str());
    ::curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    ::curl_easy_setopt(curl, CURLOPT_USERAGENT, userAgent_.c_str());
    ::curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    ::curl_easy_setopt(curl, CURLOPT_WRITEDATA, &res.body);
    ::curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);        // safe in a threaded daemon
    ::curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(req.timeoutSeconds));
    ::curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);

    // Not negotiable, and not exposed as options anywhere above this line.
    ::curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    ::curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    ::curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);  // a redirect must never carry the token elsewhere
#if LIBCURL_VERSION_NUM >= 0x075500
    ::curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
#else
    // libcurl < 7.85 (Ubuntu 22.04 ships 7.81) has only the bitmask form, deprecated from 7.85 on.
    ::curl_easy_setopt(curl, CURLOPT_PROTOCOLS, static_cast<long>(CURLPROTO_HTTP | CURLPROTO_HTTPS));
#endif

    if (req.method == "POST")
    {
        ::curl_easy_setopt(curl, CURLOPT_POST, 1L);
        ::curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req.body.c_str());
        ::curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(req.body.size()));
    }
    else if (req.method != "GET")
    {
        ::curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, req.method.c_str());
    }

    const CURLcode rc = ::curl_easy_perform(curl);
    if (rc == CURLE_OK)
    {
        long status = 0;
        ::curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
        res.status = status;
    }
    else
    {
        // status stays 0, which api_client reads as "the request never completed" -- distinct from any HTTP
        // status, because the two want different handling.
        res.error = ::curl_easy_strerror(rc);
    }

    ::curl_slist_free_all(headers);
    ::curl_easy_cleanup(curl);

    // The token must not survive in freed-then-reused memory any longer than necessary.
    if (!authHeader.empty()) std::memset(authHeader.data(), 0, authHeader.size());
    return res;
}

}  // namespace brae::node
