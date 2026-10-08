#!/usr/bin/env bash
# `brae job run` — the operator's side of submitting work.
#
# Exercised through the real binary against a local stub of the control plane, because the thing worth checking
# is the whole path: argv parsing, where the token comes from, what is sent, and what a failure looks like to
# someone at a terminal.
set -uo pipefail

BRAE="${1:?usage: brae_run.sh /path/to/brae}"
PY="${2:-python3}"

# `brae job` EXECS brae-agent, and brae-agent is built only when libcurl's headers are present
# (CMakeLists.txt gates the target on them; install.sh says so and treats its absence as non-fatal,
# because someone building brae to run simulations does not need it). Without it every assertion below
# fails on the same hand-over error rather than on anything this script is about. Measured on a GH200
# with no libcurl-dev, 2026-09-16: 9 of 10 checks red, all reading
#   "cannot start 'brae-agent', which brae needs for `brae job`".
# A missing OPTIONAL component is a skip, not a failure -- the same rule already applied to
# energy_bc_vs_openfoam (dumpEnergyBC) and the manifest gates (ofscan).
if [ ! -x "$(dirname "$BRAE")/brae-agent" ]; then
    echo "SKIP: brae-agent not built next to $BRAE (needs libcurl headers); \`brae job\` cannot run"
    exit 77
fi

fails=0
check() { if [ "$2" = "$3" ]; then echo "ok:   $1"; else echo "FAIL: $1"; echo "  expected: $3"; echo "  actual:   $2"; fails=$((fails+1)); fi; }
contains() { if printf '%s' "$2" | grep -qF -- "$3"; then echo "ok:   $1"; else echo "FAIL: $1"; echo "  wanted substring: $3"; echo "  in: $2"; fails=$((fails+1)); fi; }

work=$(mktemp -d); trap 'rm -rf "$work"; [ -n "${srv:-}" ] && kill "$srv" 2>/dev/null' EXIT

# A control plane that records what it was asked and answers like the real one.
cat > "$work/stub.py" <<'PYEOF'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG = sys.argv[2]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _record(self, method, body):
        with open(LOG, "a") as f:
            f.write(json.dumps({"method": method, "path": self.path,
                                "auth": self.headers.get("Authorization", ""), "body": body}) + "\n")

    def _send(self, code, payload):
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        self._record("POST", body)
        if body.get("sample") == "bad/sample":
            return self._send(400, {"error": {"code": "bad_sample", "message": "'bad/sample' is not a valid sample name"}})
        self._send(201, {"job_id": "job-abc123", "type": "brae-benchmark", "state": "queued",
                         "sample": body.get("sample"), "node_id": None,
                         "requested_min_vram_mb": 6000, "requested_gpu_model": body.get("gpu"),
                         "created_at": "2026-08-03T10:00:00Z", "result": None, "error": None})

    def do_GET(self):
        self._record("GET", None)
        self._send(200, {"job_id": "job-abc123", "type": "brae-benchmark", "state": "completed",
                         "sample": "pimplefoam/pitzDaily-1M", "node_id": "brae-0001",
                         "requested_min_vram_mb": 6000, "requested_gpu_model": "gb10",
                         "created_at": "2026-08-03T10:00:00Z",
                         "result": {"runtime_s": 41.5}, "error": None})

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PYEOF

port=$(( ( RANDOM % 2000 ) + 18000 ))
log="$work/requests.jsonl"; : > "$log"
"$PY" "$work/stub.py" "$port" "$log" & srv=$!
for _ in $(seq 1 50); do curl -sf -o /dev/null "http://127.0.0.1:$port/v1/jobs/x" 2>/dev/null && break; sleep 0.1; done
api="http://127.0.0.1:$port"

# ---- the token has to come from somewhere ------------------------------------------------------------------
out=$(env -u BRAE_ADMIN_TOKEN HOME="$work/empty" "$BRAE" job run some/sample --api "$api" --no-wait 2>&1)
check "no token anywhere is refused" "$?" "1"
contains "and it says where to put one" "$out" ".config/brae/token"

# ---- environment ------------------------------------------------------------------------------------------
out=$(BRAE_ADMIN_TOKEN=env-token "$BRAE" job run pimplefoam/pitzDaily-1M --api "$api" --no-wait 2>&1 | tail -1)
check "--no-wait prints just the job id" "$out" "job-abc123"
contains "the env token is sent" "$(tail -1 "$log")" '"auth": "Bearer env-token"'

# ---- the config file ----------------------------------------------------------------------------------------
mkdir -p "$work/home/.config/brae"; printf 'file-token\n' > "$work/home/.config/brae/token"
: > "$log"
env -u BRAE_ADMIN_TOKEN HOME="$work/home" "$BRAE" job run a/b --api "$api" --no-wait >/dev/null 2>&1
contains "~/.config/brae/token is used when the env is unset" "$(tail -1 "$log")" '"auth": "Bearer file-token"'
contains "a trailing newline in the file is stripped" "$(tail -1 "$log")" 'Bearer file-token"'

# ---- --token beats both -------------------------------------------------------------------------------------
: > "$log"
BRAE_ADMIN_TOKEN=env-token HOME="$work/home" "$BRAE" job run a/b --api "$api" --token flag-token --no-wait >/dev/null 2>&1
contains "--token wins over the environment" "$(tail -1 "$log")" '"auth": "Bearer flag-token"'

# ---- what actually gets sent ----------------------------------------------------------------------------------
: > "$log"
BRAE_ADMIN_TOKEN=t "$BRAE" job run pimplefoam/pitzDaily-1M --gpu gb10 --api "$api" --no-wait >/dev/null 2>&1
body=$(tail -1 "$log")
contains "the sample is sent" "$body" '"sample": "pimplefoam/pitzDaily-1M"'
contains "--gpu is sent" "$body" '"gpu": "gb10"'

: > "$log"
BRAE_ADMIN_TOKEN=t "$BRAE" job run pimplefoam/pitzDaily-1M --api "$api" --no-wait >/dev/null 2>&1
if printf '%s' "$(tail -1 "$log")" | grep -q '"gpu"'; then
  echo "FAIL: no --gpu should send no gpu field"; fails=$((fails+1))
else
  echo "ok:   no --gpu sends no gpu field"
fi

# ---- failures read like something a person can act on ---------------------------------------------------------
out=$(BRAE_ADMIN_TOKEN=t "$BRAE" job run bad/sample --api "$api" --no-wait 2>&1)
check "a refused sample exits non-zero" "$?" "1"
contains "and shows the message, not the code" "$out" "is not a valid sample name"

out=$(BRAE_ADMIN_TOKEN=t "$BRAE" job run a/b --api "http://127.0.0.1:1" --no-wait 2>&1)
check "an unreachable control plane exits non-zero" "$?" "1"
contains "and says it was the transport" "$out" "transport"

# ---- status ---------------------------------------------------------------------------------------------------
out=$(BRAE_ADMIN_TOKEN=t "$BRAE" job status job-abc123 --api "$api" 2>&1)
check "status of a completed job exits zero" "$?" "0"
contains "status shows the state" "$out" "completed"
contains "status shows which machine ran it" "$out" "brae-0001"

# ---- argv --------------------------------------------------------------------------------------------------
out=$(BRAE_ADMIN_TOKEN=t "$BRAE" job run --api "$api" 2>&1); check "no sample is a usage error" "$?" "2"
out=$(BRAE_ADMIN_TOKEN=t "$BRAE" job run a/b --nope --api "$api" 2>&1); check "an unknown option is refused" "$?" "2"

if [ "$fails" -eq 0 ]; then echo "PASS: brae job run"; else echo "FAILED: $fails check(s)"; fi
exit $((fails > 0))
