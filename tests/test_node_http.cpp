// The real libcurl transport, against a real server (tests/node_http_server.py).
//
// api_client's behaviour is covered against a fake transport; this covers the things only a real socket can
// show: that TLS verification is on and cannot be turned off, that a redirect is refused rather than followed
// with the bearer token attached, that an endless body is cut off, and that a slow server hits the timeout
// instead of holding the agent forever.
#include "api_client.h"
#include "http_curl.h"
#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace brae::node;

static int failures = 0;

static void check(bool cond, const std::string& what)
{
    if (cond) { std::printf("ok:   %s\n", what.c_str()); return; }
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

static void checkEq(const std::string& got, const std::string& want, const std::string& what)
{
    if (got == want) { std::printf("ok:   %s\n", what.c_str()); return; }
    std::printf("FAIL: %s\n        got:  %s\n        want: %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: %s <base-url>\n", argv[0]); return 2; }
    const std::string base = argv[1];

    CurlHttp http;
    http.setNodeId("brae-7f21");

    // ---- a plain POST carries what it should, and nothing it should not ------------------------------------
    {
        HttpRequest req;
        req.method = "POST";
        req.url = base + "/echo";
        req.body = R"({"hello":"world"})";
        req.bearer = "brae_nt_supersecret";
        const HttpResponse res = http.send(req);

        check(res.status == 200, "post: reaches the server");
        Json j;
        std::string err;
        check(Json::parse(res.body, j, err), "post: server saw valid JSON");
        checkEq(j["method"].asString(), "POST", "post: method");
        checkEq(j["body"].asString(), R"({"hello":"world"})", "post: body arrives intact");
        checkEq(j["authorization"].asString(), "Bearer brae_nt_supersecret", "post: bearer token is sent");
        checkEq(j["content_type"].asString(), "application/json", "post: content type");
        checkEq(j["node_header"].asString(), "brae-7f21", "post: X-Brae-Node identifies the node");
        check(j["user_agent"].asString().find("brae-agent/") == 0, "post: user agent names the agent");
        check(j["path"].asString().find("brae_nt_") == std::string::npos,
              "post: the token is not in the path, where it would land in proxy logs");
    }

    // ---- a redirect must NOT be followed ---------------------------------------------------------------------
    {
        // curl follows redirects when asked, and would re-send the Authorization header. The whole point of
        // CURLOPT_FOLLOWLOCATION being off is that a compromised or misconfigured server cannot use a 302 to
        // walk this node's token to another host.
        HttpRequest req;
        req.url = base + "/redirect";
        req.bearer = "brae_nt_supersecret";
        const HttpResponse res = http.send(req);
        check(res.status == 302, "redirect: reported as a 302, not silently followed");
        check(res.body.find("leaked") == std::string::npos, "redirect: the target was never fetched");
    }

    // ---- only http and https are spoken ---------------------------------------------------------------------
    {
        // libcurl < 7.85 takes the restriction as CURLOPT_PROTOCOLS, newer ones as CURLOPT_PROTOCOLS_STR; either
        // way a file:// URL must not read the local disk into a request's response.
        HttpRequest req;
        req.url = "file:///proc/self/exe";
        const HttpResponse res = http.send(req);
        check(res.transportFailed(), "file://: refused as a transport failure");
        check(res.body.empty(), "file://: nothing was read from disk");
    }

    // ---- an endless body is cut off ---------------------------------------------------------------------------
    {
        HttpRequest req;
        req.url = base + "/huge";
        req.timeoutSeconds = 20;
        const HttpResponse res = http.send(req);
        check(res.body.size() <= kMaxResponseBytes,
              "huge: the body is capped at kMaxResponseBytes, not read to the end");
        check(res.transportFailed(),
              "huge: aborting the transfer is reported as a transport failure, not a partial success");
    }

    // ---- a slow server hits the timeout ------------------------------------------------------------------------
    {
        HttpRequest req;
        req.url = base + "/slow";
        req.timeoutSeconds = 2;                    // the route sleeps 10
        const HttpResponse res = http.send(req);
        check(res.transportFailed(), "slow: a server that stalls does not hold the agent");
        check(!res.error.empty(), "slow: the timeout is reported with a reason");
    }

    // ---- error bodies come back for the protocol layer to interpret ----------------------------------------------
    {
        HttpRequest req;
        req.url = base + "/status/500";
        const HttpResponse res = http.send(req);
        check(res.status == 500, "500: status is surfaced");
        check(!res.ok(), "500: not ok");
        checkEq(parseSimpleResponse(res).error, "internal_error", "500: the envelope's code is read");
    }
    {
        HttpRequest req;
        req.url = base + "/proxy-html";
        const HttpResponse res = http.send(req);
        const SimpleResult r = parseSimpleResponse(res);
        check(!r.ok, "proxy html: not ok");
        checkEq(r.error, "http_502", "proxy html: a non-JSON error page degrades to the status code");
    }
    {
        HttpRequest req;
        req.url = base + "/status/401";
        const HttpResponse res = http.send(req);
        check(parseSimpleResponse(res).authRejected, "401: flagged so the loop can stop rather than hammer");
    }

    // ---- TLS verification is on, and there is no way to turn it off -----------------------------------------------
    {
        // A self-signed host must fail. If this ever passes, the agent will happily talk to anyone who can
        // intercept DNS -- so the test asserts the failure, not the success.
        HttpRequest req;
        req.url = "https://self-signed.badssl.com/";
        req.timeoutSeconds = 8;
        const HttpResponse res = http.send(req);
        if (res.transportFailed() && res.error.find("SSL") == std::string::npos
            && res.error.find("certificate") == std::string::npos)
        {
            std::printf("skip: no network for the TLS check (%s)\n", res.error.c_str());
        }
        else
        {
            check(res.transportFailed(), "tls: a self-signed certificate is refused");
            check(res.status != 200, "tls: and the body is never delivered");
        }
    }

    // ---- an unreachable host is a transport failure, not a status --------------------------------------------------
    {
        HttpRequest req;
        req.url = "http://127.0.0.1:1/nothing";
        req.timeoutSeconds = 5;
        const HttpResponse res = http.send(req);
        check(res.transportFailed(), "unreachable: reported as a transport failure");
        check(res.status == 0, "unreachable: status stays 0, distinct from any HTTP code");
    }

    if (failures == 0) std::printf("PASS: brae-agent transport\n");
    else std::printf("FAILED: %d check(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
