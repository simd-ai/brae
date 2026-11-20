# ---------------------------------------------------------------------------------------------------------
# brae-agent: the node service, and the implementation behind `brae node ...`.
#
# A CXX-ONLY target. It links neither CUDA nor brae_core, deliberately: a contributor's machine must keep
# reporting even when the solver build on it is broken, and a solver bug must not be able to take the node's
# reporting down with it. libcurl and dlopen are its entire dependency budget.
# ---------------------------------------------------------------------------------------------------------
set(BRAE_AGENT_SOURCES
    agent_main.cpp agent.cpp cli.cpp service_systemd.cpp
    api_client.cpp http_curl.cpp json.cpp jobs.cpp
    identity.cpp gpu_probe.cpp runner.cpp sha256.cpp)

if(CURL_FOUND)
    add_executable(brae-agent ${BRAE_AGENT_SOURCES})
    target_include_directories(brae-agent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(brae-agent PRIVATE CURL::libcurl Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(brae-agent PROPERTIES LINKER_LANGUAGE CXX)
    # `brae node <verb>` execs brae-agent from beside its own binary, so the two must land together.
    add_dependencies(brae brae-agent)
else()
    message(STATUS "brae: libcurl not found -- brae-agent is not built "
                   "(install libcurl4-openssl-dev); the solver is unaffected")
endif()
