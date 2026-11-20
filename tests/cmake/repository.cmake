# Repository and command line. Audits that read the tree (manifests, default and cache-key audits, header ODR, gate
# registration), the `brae` command line and dispatch, the node agent, and tools/ctest_cached.py.
#
# Included by tests/CMakeLists.txt.

add_test(NAME manifest_rhosimplefoam_not_drifted
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/of_manifest.py rhoSimpleFoam
                 --check ${PROJECT_SOURCE_DIR}/manifest/rhoSimpleFoam.yaml)
# ofscan is a sibling checkout, not a build dependency (see tools/of_manifest.py's db()). A box that
# lacks it cannot ask the drift question and must SKIP, not fail -- the GH200 reported a red here for
# exactly that reason on 2026-09-15.
set_tests_properties(manifest_rhosimplefoam_not_drifted PROPERTIES SKIP_RETURN_CODE 77)

# The simpleFoam twin existed only as a tool flag, and its manifest drifted for weeks: sixteen whole
# entries and four long validation/note blocks were hand-added to the YAML while the tool's curated
# table (where they are supposed to live) fell behind. Reconciled 2026-08-31; this test is what keeps
# the two from diverging again.
add_test(NAME manifest_simplefoam_not_drifted
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/of_manifest.py simpleFoam
                 --check ${PROJECT_SOURCE_DIR}/manifest/simpleFoam.yaml)
set_tests_properties(manifest_simplefoam_not_drifted PROPERTIES SKIP_RETURN_CODE 77)

# ...and the interFoam twin, added with the manifest itself rather than weeks later. The curated half of
# that table is where each component's brae_reference/brae_target path lives, and those paths were WRONG
# for six components within a day of being written -- the files landed under different names. Nothing
# caught it because nothing asked.
add_test(NAME manifest_interfoam_not_drifted
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/of_manifest.py interFoam
                 --check ${PROJECT_SOURCE_DIR}/manifest/interFoam.yaml)
set_tests_properties(manifest_interfoam_not_drifted PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME manifest_rhocentralfoam_not_drifted
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/of_manifest.py rhoCentralFoam
                 --check ${PROJECT_SOURCE_DIR}/manifest/rhoCentralFoam.yaml)
set_tests_properties(manifest_rhocentralfoam_not_drifted PROPERTIES SKIP_RETURN_CODE 77)
# Defaults left over from porting, in the interFoam mirror: a controls struct built by hand at several
# sites with a field set at some and not at others runs the default where the case said otherwise --
# the device's start-up CorrectPhi ran grad(pcorr) as Gauss linear that way, and its closure would have
# run a moving mesh on the current volumes. Every justified omission is in tools/default_audit_allow.txt
# with its reason; an unlisted one, or a listed one that no longer exists, fails here.
# ...over the WHOLE of src now: the backlog the first run found (341 across 28 structs) is closed, so
# a new omission anywhere is a new finding rather than one of a crowd.
# ...and the other half of the same question: what each ported file SAYS holds it. Every provenance
# block's `tests:` line names paths; this fails when one of them does not exist, which is the rot that
# let braeInterFoam.cu keep saying `-device` refuses LES for units after it ran. `--gaps` (not run here)
# lists the files that declare provenance and name no test at all.
add_test(NAME mirror_coverage
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/mirror_coverage.py --check)

add_test(NAME default_audit_interfoam
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/default_audit.py
                 --allow ${PROJECT_SOURCE_DIR}/tools/default_audit_allow.txt
                 ${PROJECT_SOURCE_DIR}/src)

# DeviceBuffer::resize does not zero -- it hands the old block back to a pool and takes one out, and
# `resize(n)` with n unchanged returns at once, keeping the LAST call's values. A kernel that
# accumulates (`out[i] +=`, atomicAdd) into such a buffer adds to whatever was there: interFoam's
# vector limitedLinear built its magSqr limiter that way and read U 5.1e-01 of OpenFOAM's. The audit
# asks it of every accumulating kernel in the tree, in the same function and across one call.
add_test(NAME resize_audit
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/resize_audit.py
                 --allow ${PROJECT_SOURCE_DIR}/tools/resize_audit_allow.txt
                 ${PROJECT_SOURCE_DIR}/src)
# ...and the proof that it fails when the defect is there: it cuts the zeroing out of two real sites
# (one same-function, one across a call) and checks each is named, plus that a memset written in a
# COMMENT does not count -- which is how the audit's own first run reported a comment as a defect.
# A device cache keyed on a POINTER: the pool hands equal-sized blocks back at the same address, so
# `A.diag` or `A.owner` is not the identity of an addressing. On a moving mesh the GAMG hierarchy is
# rebuilt every step and its levels return at the same pointers with the same counts and a different
# pairing -- device U 3.0057e-02 on waveMakerMultiPaddlePiston, and a second run in one process
# differing from the first. Every such guard must also compare an identity that survives recycling.
add_test(NAME cache_key_audit
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/cache_key_audit.py
                 --allow ${PROJECT_SOURCE_DIR}/tools/cache_key_audit_allow.txt
                 ${PROJECT_SOURCE_DIR}/src)
add_test(NAME cache_key_audit_selftest
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/cache_key_audit_selftest.sh)

# A function DEFINED in a .cuh without inline/static/constexpr is a strong symbol in every object that
# includes the header, and the link fails the moment two of them are put together. `readSurfaceField`
# was that for six translation units -- the ordinary build survived only because the archive handed out
# one of them, while a localiser that pulled the other could not be linked at all.
add_test(NAME header_odr_audit
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/header_odr_audit.py
                 --allow ${PROJECT_SOURCE_DIR}/tools/header_odr_audit_allow.txt
                 ${PROJECT_SOURCE_DIR}/src)
# ...and the proof that it names them: it restores both real defects in scratch copies, and checks that
# a declaration, a braced default argument, a template and a class member are NOT reported.
add_test(NAME header_odr_audit_selftest
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/header_odr_audit_selftest.sh)

# ...and the default ledger's own keys: `Compressible` is defined in kEpsilonRef AND kOmegaSST with the
# same field names, so a line keyed on the struct name alone exempted the field in both types and a
# line stale in one was held alive by the other. The key carries the namespace now.
add_test(NAME default_audit_selftest
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/default_audit_selftest.sh)

add_test(NAME resize_audit_selftest
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/resize_audit_selftest.sh)

add_test(NAME mirror_dict_audit
         COMMAND ${PROJECT_SOURCE_DIR}/tests/mirror_dict_audit.sh)
set_tests_properties(mirror_dict_audit PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

# Single-GPU hardening regression lock: assert each fail-loud guard still fires on a mutated case
# (non-Newtonian, endTime<1, #calc, missing div(phi,U), field-size, unsupported fvOptions, wall-fn) -> a future
# edit that silently drops a guard fails here. See cudafoam/single-gpu-hardening.md.
add_test(NAME hardening_guards
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/hardening_guards.sh
                 $<TARGET_FILE:brae>
                 ${PROJECT_SOURCE_DIR}/validation/kEpsCase
                 ${PROJECT_BINARY_DIR}/hardening_guards_work)

# D5: every validation/*_vs_openfoam.sh must be registered, runnable and asserting. 27 were none of those.
add_executable(test_gate_registration ${PROJECT_SOURCE_DIR}/tests/test_gate_registration.cu)
target_include_directories(test_gate_registration PRIVATE ${BRAE_HEADER_DIRS})
add_test(NAME gate_registration COMMAND test_gate_registration ${PROJECT_SOURCE_DIR})

# The CLI's two reserved words (`benchmark`, `node`) route to their subcommands, and -- the half that protects
# every existing user -- nothing else does: a case directory called `run`, `status` or `list` is still a case.
add_test(NAME brae_subcommands
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/brae_subcommands.sh
                 $<TARGET_FILE:brae>
                 ${PROJECT_BINARY_DIR}/brae_subcommands_work)
set_tests_properties(brae_subcommands PROPERTIES TIMEOUT 300)
# The command line's commands -- --version, solvers, capabilities, check, run -- on real cases, every refusal of `brae check` a
# negative control that must exit 2 naming its setting, case directories named like the commands still running as cases, and
# every gate a capability names held to this list of tests
add_test(NAME brae_cli
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/brae_cli.sh
                 $<TARGET_FILE:brae> ${PROJECT_BINARY_DIR}/brae_cli_work ${PROJECT_SOURCE_DIR}
                 ${PROJECT_BINARY_DIR})
set_tests_properties(brae_cli PROPERTIES TIMEOUT 900)
# patchFieldTypeNames (what `brae capabilities` lists and `brae check` consults) is the boundary-condition factory's branches,
# with its own mutation control
add_test(NAME brae_patch_types_vs_factory
         COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/brae_patch_types_vs_factory.py ${PROJECT_SOURCE_DIR})

# `brae benchmark <sample>` end to end against a LOCAL fixture template repo: fetch, cache, manifest, and the
# security guard that refuses a sample carrying coded (NVRTC-compiled) boundary conditions. Solves the committed
# pitzDaily cut to 5 iterations, so the happy path is a real run.
add_test(NAME brae_benchmark
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/brae_benchmark.sh
                 $<TARGET_FILE:brae>
                 ${PROJECT_SOURCE_DIR}/validation/pitzDaily
                 ${PROJECT_BINARY_DIR}/brae_benchmark_work)
set_tests_properties(brae_benchmark PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 600)

# brae-agent unit tests (milestone 3): JSON, SHA-256, the identity file, sample-name validation, argv
# construction and backoff. Deliberately a CXX-ONLY target that links neither brae_core nor CUDA -- the agent
# must build and run on a machine whose solver build is broken, and so must its tests. No GPU, no network.
add_executable(test_node_agent ${PROJECT_SOURCE_DIR}/tests/test_node_agent.cpp
               ${PROJECT_SOURCE_DIR}/src/node/json.cpp ${PROJECT_SOURCE_DIR}/src/node/jobs.cpp ${PROJECT_SOURCE_DIR}/src/node/sha256.cpp ${PROJECT_SOURCE_DIR}/src/node/identity.cpp)
target_include_directories(test_node_agent PRIVATE ${PROJECT_SOURCE_DIR}/src/node)
set_target_properties(test_node_agent PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_agent COMMAND test_node_agent)
set_tests_properties(node_agent PROPERTIES TIMEOUT 60)

# The agent's protocol against a FAKE transport: every status code, a body that is not JSON, JSON of the wrong
# shape, an assignment for a job type this agent does not know. Those are the cases a real orchestrator will not
# produce on request, and they decide whether a node quietly stops working. No network.
add_executable(test_node_api ${PROJECT_SOURCE_DIR}/tests/test_node_api.cpp
               ${PROJECT_SOURCE_DIR}/src/node/api_client.cpp ${PROJECT_SOURCE_DIR}/src/node/json.cpp ${PROJECT_SOURCE_DIR}/src/node/jobs.cpp)
target_include_directories(test_node_api PRIVATE ${PROJECT_SOURCE_DIR}/src/node)
set_target_properties(test_node_api PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_api COMMAND test_node_api)
set_tests_properties(node_api PROPERTIES TIMEOUT 60)

# Job execution and GPU detection. The runner is driven against fake `brae` binaries the test writes itself --
# one that succeeds, one that fails, one that hangs, one that is missing -- because those four are what actually
# happen on a contributor's machine. The GPU probe covers the no-driver path and, where there is a driver, the
# real one. Needs dl for NVML; still no CUDA and no network.
add_executable(test_node_runner ${PROJECT_SOURCE_DIR}/tests/test_node_runner.cpp
               ${PROJECT_SOURCE_DIR}/src/node/runner.cpp ${PROJECT_SOURCE_DIR}/src/node/gpu_probe.cpp ${PROJECT_SOURCE_DIR}/src/node/jobs.cpp ${PROJECT_SOURCE_DIR}/src/node/sha256.cpp)
target_include_directories(test_node_runner PRIVATE ${PROJECT_SOURCE_DIR}/src/node)
target_link_libraries(test_node_runner PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(test_node_runner PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_runner COMMAND test_node_runner)
set_tests_properties(node_runner PROPERTIES TIMEOUT 120)

# The loop itself, driven at exact instants with a fake transport, fake executor and fake clock: a job that
# outlives several heartbeats, a result POST that fails repeatedly, a process restarted between finishing a job
# and reporting it, a 401 after months of working. No threads, no sleeps, no network.
add_executable(test_node_loop ${PROJECT_SOURCE_DIR}/tests/test_node_loop.cpp
               ${PROJECT_SOURCE_DIR}/src/node/agent.cpp ${PROJECT_SOURCE_DIR}/src/node/api_client.cpp ${PROJECT_SOURCE_DIR}/src/node/json.cpp
               ${PROJECT_SOURCE_DIR}/src/node/jobs.cpp ${PROJECT_SOURCE_DIR}/src/node/runner.cpp)
target_include_directories(test_node_loop PRIVATE ${PROJECT_SOURCE_DIR}/src/node)
target_link_libraries(test_node_loop PRIVATE Threads::Threads)
set_target_properties(test_node_loop PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_loop COMMAND test_node_loop)
set_tests_properties(node_loop PROPERTIES TIMEOUT 120)

# `brae node register | status | unregister`, against a fake API and a fake service manager. The criterion is
# atomicity: each step of registration is made to fail in turn -- the GPU probe, the API, the identity write,
# the service account, the ownership hand-over, the unit install, the start -- and each must leave no identity
# file, no service and no registry row behind.
add_executable(test_node_cli ${PROJECT_SOURCE_DIR}/tests/test_node_cli.cpp
               ${PROJECT_SOURCE_DIR}/src/node/cli.cpp ${PROJECT_SOURCE_DIR}/src/node/api_client.cpp ${PROJECT_SOURCE_DIR}/src/node/json.cpp ${PROJECT_SOURCE_DIR}/src/node/jobs.cpp
               ${PROJECT_SOURCE_DIR}/src/node/identity.cpp ${PROJECT_SOURCE_DIR}/src/node/gpu_probe.cpp ${PROJECT_SOURCE_DIR}/src/node/sha256.cpp
               # for isServiceReachablePath: the rule that decides whether the unit can start at all
               ${PROJECT_SOURCE_DIR}/src/node/service_systemd.cpp)
target_include_directories(test_node_cli PRIVATE ${PROJECT_SOURCE_DIR}/src/node)
target_link_libraries(test_node_cli PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(test_node_cli PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME node_cli COMMAND test_node_cli)

# `brae run` against a stub control plane: argv, where the admin token comes from, what is actually sent, and
# what a failure looks like to someone at a terminal.
add_test(NAME brae_run
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/brae_run.sh $<TARGET_FILE:brae> python3)
# brae-agent is optional (built only with libcurl headers), and `brae job` execs it -- so a box without
# it must SKIP, not fail. The GH200 reported 9 red checks for exactly this on 2026-09-16.
set_tests_properties(brae_run PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
set_tests_properties(node_cli PROPERTIES TIMEOUT 60)

# The real transport, against a deliberately awkward local server: a redirect that would leak the bearer token,
# a body that never ends, a server that stalls, a proxy's HTML error page. Guarded on libcurl so a machine
# without the headers still builds and tests everything else -- the agent's behaviour is covered above the
# transport (node_api), and only these properties need a real socket.
find_package(CURL QUIET)
if(CURL_FOUND)
    add_executable(test_node_http ${PROJECT_SOURCE_DIR}/tests/test_node_http.cpp
                   ${PROJECT_SOURCE_DIR}/src/node/http_curl.cpp ${PROJECT_SOURCE_DIR}/src/node/api_client.cpp ${PROJECT_SOURCE_DIR}/src/node/json.cpp ${PROJECT_SOURCE_DIR}/src/node/jobs.cpp)
    target_include_directories(test_node_http PRIVATE ${PROJECT_SOURCE_DIR}/src/node)
    target_link_libraries(test_node_http PRIVATE CURL::libcurl)
    set_target_properties(test_node_http PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME node_http
             COMMAND bash ${PROJECT_SOURCE_DIR}/tests/node_http_run.sh
                     $<TARGET_FILE:test_node_http>
                     ${PROJECT_SOURCE_DIR}/tests/node_http_server.py)
    set_tests_properties(node_http PROPERTIES SKIP_RETURN_CODE 125 TIMEOUT 180)
else()
    message(STATUS "brae: libcurl not found -- the agent transport test is skipped "
                   "(install libcurl4-openssl-dev)")
endif()

# tools/ctest_cached.py -- ctest with every test left out that has already passed on the same inputs -- held
# to that on a throwaway project of four tests: held while the inputs stand, run when one changes, a failure
# and a test that rewrites its own inputs never held. No GPU, no solver.
add_test(NAME ctest_cache_selftest
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/ctest_cache_selftest.sh
                 ${PROJECT_SOURCE_DIR}/tools/ctest_cached.py
                 ${PROJECT_BINARY_DIR}/ctest_cache_selftest_work)
set_tests_properties(ctest_cache_selftest PROPERTIES TIMEOUT 120)

# Solver selection itself: routes on controlDict `application`, refuses the solvers brae does not have, and
# catches an application/ddtSchemes contradiction. Dict-only fake cases, no mesh and no GPU.
add_test(NAME solver_dispatch
         COMMAND bash ${PROJECT_SOURCE_DIR}/tests/solver_dispatch.sh
                 $<TARGET_FILE:brae>
                 ${PROJECT_BINARY_DIR}/solver_dispatch_work)
set_tests_properties(solver_dispatch PROPERTIES TIMEOUT 120)
