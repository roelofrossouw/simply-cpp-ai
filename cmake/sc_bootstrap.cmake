# Locates the shared simply-cpp build helpers - get_sc_version(), add_sc_object(),
# add_sc_test(), find_or_install_package() - and includes them, so a module does not
# have to keep its own fork of them.
#
# Include this from a module's top level CMakeLists.txt before calling any of them:
#     include(cmake/sc_bootstrap.cmake)
#
# The helpers are looked for in this order:
#   1. An installed sc-core package. The live copy, on a machine that has core installed.
#   2. SimplyCppFunctions.cmake next to this file. This is what a machine with no sc
#      installed and no network uses, so it belongs in the repository.
#   3. The sc git repository, through FetchContent. Only reached on a machine that has
#      neither of the above, and only for long enough to create 2.
#
# Core is the source of truth. Every configure that can reach it takes its copies of
#   cmake/SimplyCppFunctions.cmake   the build helpers
#   cmake/sc_bootstrap.cmake         this file
#   scripts/*.sh.in                  the script templates, whatever core has
#   tests/sc_test.h                  the test harness, where add_sc_test() looks
# and regenerates each scripts/<name>.sh from its template. Adding a template to core
# is therefore all it takes for every module to pick it up. So the first
# configure on a fresh machine fetches, every configure after that is offline, and a
# module follows core rather than keeping whatever it was given.
#
# -DSC_UPDATE_HELPERS=OFF pins a module to the copies it has; differences are then
# reported and left alone.
#
# This file tracks core too, so it is the only file a new module has to start with and
# it keeps itself current from there. One line puts it in place:
#
#     curl -O --create-dirs --output-dir cmake \
#         https://raw.githubusercontent.com/roelofrossouw/simply-cpp/main/cmake/sc_bootstrap.cmake
#
# or, on a machine with sc installed, copy it out of <prefix>/lib/cmake/sc/. After that
# `include(cmake/sc_bootstrap.cmake)` is the whole of a module's setup.
#
# Point SC_HELPERS_REPOSITORY and SC_HELPERS_TAG somewhere else to fetch from a fork
# or a pinned revision.

set(SC_HELPERS_REPOSITORY "https://github.com/roelofrossouw/simply-cpp.git" CACHE STRING "Where to fetch the simply-cpp build helpers from")
set(SC_HELPERS_TAG "main" CACHE STRING "Which revision of the simply-cpp build helpers to fetch")
# On by default: core is the source of truth, so a module tracks it rather than keeping
# whatever it happened to be given. Turn it off to pin a module to its current copies.
option(SC_UPDATE_HELPERS "Track core's copies of the helpers, sc_test.h and the deploy scripts" ON)
option(SC_DEPLOY_SCRIPTS "Create and maintain scripts/deploy.sh and scripts/run.sh" ON)
# A module that builds core from source has to turn this off, or find_package() imports
# sc::sc-core here and the fetched source cannot then define a target of that name.
# Set it as a normal variable before including this file.
option(SC_HELPERS_USE_PACKAGE "Take the helpers from an installed sc-core when there is one" ON)
# The directory the module is rsynced to on the server, and the one run.sh builds in.
# Defaults to the module name so two modules cannot land on top of each other.
if (NOT SC_DEPLOY_NAME)
    if (SC_MODULE)
        set(SC_DEPLOY_NAME "${SC_MODULE}")
    else ()
        set(SC_DEPLOY_NAME "${PROJECT_NAME}")
    endif ()
endif ()

set(sc_helpers_cached "${CMAKE_CURRENT_LIST_DIR}/SimplyCppFunctions.cmake")
set(sc_test_header_cached "${CMAKE_CURRENT_SOURCE_DIR}/tests/sc_test.h")
set(sc_bootstrap_cached "${CMAKE_CURRENT_LIST_FILE}") # this file, kept in step with the rest
set(sc_helpers_source "")     # where to copy the helpers from, empty when already cached
set(sc_test_header_source "")
set(sc_bootstrap_source "")
set(sc_scripts_source "")     # directory holding deploy.sh.in and run.sh.in
set(sc_helpers_origin "")

# Reads SC_HELPERS_VERSION out of a helpers file without including it. Absent means 0,
# which is any copy predating the version stamp.
function(sc_helpers_file_version file output)
    set(${output} 0 PARENT_SCOPE)
    if (NOT EXISTS "${file}")
        return()
    endif ()
    file(STRINGS "${file}" version_line REGEX "^set\\(SC_HELPERS_VERSION ")
    if (version_line)
        string(REGEX MATCH "[0-9]+" file_version "${version_line}")
        set(${output} "${file_version}" PARENT_SCOPE)
    endif ()
endfunction()

# 1. An installed sc-core package. Its config includes the helpers itself, so the
#    functions exist afterwards; sc-core_DIR is where the copyable file sits.
if (SC_HELPERS_USE_PACKAGE)
    find_package(sc-core QUIET)
endif ()
if (COMMAND get_sc_version)
    set(sc_helpers_origin "the sc-core package at ${sc-core_DIR}")
    if (EXISTS "${sc-core_DIR}/SimplyCppFunctions.cmake")
        set(sc_helpers_source "${sc-core_DIR}/SimplyCppFunctions.cmake")
    endif ()
    if (SC_TEST_INCLUDE_DIR AND EXISTS "${SC_TEST_INCLUDE_DIR}/sc_test.h")
        set(sc_test_header_source "${SC_TEST_INCLUDE_DIR}/sc_test.h")
    endif ()
    if (EXISTS "${sc-core_DIR}/sc_bootstrap.cmake")
        set(sc_bootstrap_source "${sc-core_DIR}/sc_bootstrap.cmake")
    endif ()
    file(GLOB sc_installed_templates "${sc-core_DIR}/*.sh.in")
    if (sc_installed_templates)
        set(sc_scripts_source "${sc-core_DIR}") # installed flat next to the helpers
    endif ()
endif ()

# An installed sc-core older than this module's own copy is ignored outright. It would
# otherwise win on both counts: its helpers are the ones actually loaded, and with
# tracking on they overwrite the newer copy on the way past. A module then builds
# against helpers missing whatever its CMakeLists.txt relies on, and the errors look
# nothing like the cause.
if (COMMAND get_sc_version AND EXISTS "${sc_helpers_cached}")
    sc_helpers_file_version("${sc_helpers_source}" sc_installed_helpers_version)
    sc_helpers_file_version("${sc_helpers_cached}" sc_local_helpers_version)
    if (sc_local_helpers_version GREATER sc_installed_helpers_version)
        message(WARNING "The sc-core installed at ${sc-core_DIR} carries helpers version"
                " ${sc_installed_helpers_version}, older than this module's"
                " ${sc_local_helpers_version} - using the local copy instead."
                " Rebuild and reinstall simply-cpp core to clear this.")
        include("${sc_helpers_cached}")
        set(sc_helpers_origin "${sc_helpers_cached}")
        # Nothing is taken from an installation this far behind.
        set(sc_helpers_source "")
        set(sc_test_header_source "")
        set(sc_bootstrap_source "")
        set(sc_scripts_source "")
    endif ()
endif ()

# 2. The copy in this directory.
if (NOT COMMAND get_sc_version AND EXISTS "${sc_helpers_cached}")
    include("${sc_helpers_cached}")
    set(sc_helpers_origin "${sc_helpers_cached}")
endif ()

# 3. The repository. Reaching here means there is nothing else to fall back on, so a
#    failure to fetch is a real error rather than something to paper over.
if (NOT COMMAND get_sc_version)
    message(STATUS "No sc package and no cached helpers, fetching from ${SC_HELPERS_REPOSITORY}")
    include(FetchContent)
    FetchContent_Declare(sc_helpers
            GIT_REPOSITORY ${SC_HELPERS_REPOSITORY}
            GIT_TAG ${SC_HELPERS_TAG}
            GIT_SHALLOW TRUE
            # Nothing here is built. Naming a subdirectory that does not exist tells
            # FetchContent to populate the source and stop, instead of configuring
            # the whole of core as a subproject.
            SOURCE_SUBDIR sc-helpers-are-not-built)
    FetchContent_MakeAvailable(sc_helpers)

    set(sc_helpers_source "${sc_helpers_SOURCE_DIR}/cmake/SimplyCppFunctions.cmake")
    set(sc_test_header_source "${sc_helpers_SOURCE_DIR}/tests/sc_test.h")
    set(sc_bootstrap_source "${sc_helpers_SOURCE_DIR}/cmake/sc_bootstrap.cmake")
    set(sc_scripts_source "${sc_helpers_SOURCE_DIR}/scripts")
    set(sc_helpers_origin "${SC_HELPERS_REPOSITORY}@${SC_HELPERS_TAG}")
    include("${sc_helpers_source}")
endif ()

if (NOT COMMAND get_sc_version)
    message(FATAL_ERROR "Found simply-cpp build helpers at ${sc_helpers_origin} but they do not"
            " define get_sc_version(). They predate it - update the copy with"
            " -DSC_UPDATE_HELPERS=ON, or reinstall simply-cpp core.")
endif ()

message(STATUS "simply-cpp build helpers from ${sc_helpers_origin}")

# Cache what was found, so the next configure - and a machine with neither sc nor a
# network - does not have to look for it again.
function(sc_cache_helper source destination what)
    if (NOT source OR NOT EXISTS "${source}")
        return()
    endif ()
    get_filename_component(destination_dir "${destination}" DIRECTORY)
    if (NOT IS_DIRECTORY "${destination_dir}")
        return()
    endif ()

    # Resolving the helpers out of the very directory being cached into, which is what
    # core itself would do, must not copy a file over itself.
    get_filename_component(source_path "${source}" REALPATH)
    get_filename_component(destination_path "${destination}" ABSOLUTE)
    if (source_path STREQUAL destination_path)
        return()
    endif ()

    if (EXISTS "${destination}")
        file(SHA256 "${source}" source_hash)
        file(SHA256 "${destination}" destination_hash)
        if (source_hash STREQUAL destination_hash)
            return() # already in step
        endif ()
        if (NOT SC_UPDATE_HELPERS)
            # Pinned: SC_UPDATE_HELPERS is off, so the local copy stands even though core
            # has moved on.
            message(STATUS "${what} differs from the copy in ${sc_helpers_origin}"
                    " - pinned, SC_UPDATE_HELPERS is off")
            set(sc_helpers_drifted TRUE PARENT_SCOPE)
            return()
        endif ()
    endif ()

    file(COPY_FILE "${source}" "${destination}" ONLY_IF_DIFFERENT RESULT copy_error)
    if (copy_error)
        message(WARNING "Could not cache ${what} to ${destination}: ${copy_error}")
    else ()
        message(STATUS "Cached ${what} to ${destination} - commit it so a build without"
                " sc installed and without a network still works")
    endif ()
endfunction()

set(sc_helpers_drifted FALSE)
sc_cache_helper("${sc_helpers_source}" "${sc_helpers_cached}" "SimplyCppFunctions.cmake")
sc_cache_helper("${sc_test_header_source}" "${sc_test_header_cached}" "sc_test.h")
sc_cache_helper("${sc_bootstrap_source}" "${sc_bootstrap_cached}" "sc_bootstrap.cmake")

# The deploy scripts are templates rather than straight copies: the server directory is
# the module name, so two modules deployed to the same box do not overwrite each other.
if (SC_DEPLOY_SCRIPTS)
    set(sc_module_scripts "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
    file(MAKE_DIRECTORY "${sc_module_scripts}")

    # Take whatever templates core has, rather than a fixed list, so adding one there
    # is all it takes for every module to get it.
    if (sc_scripts_source)
        file(GLOB sc_core_templates "${sc_scripts_source}/*.sh.in")
        foreach (template ${sc_core_templates})
            get_filename_component(template_name "${template}" NAME)
            sc_cache_helper("${template}" "${sc_module_scripts}/${template_name}" "${template_name}")
        endforeach ()
    endif ()

    # Generate from the templates this module now holds, so one with no sc installed and
    # no network still rebuilds its scripts from the copies it has.
    file(GLOB sc_module_templates "${sc_module_scripts}/*.sh.in")
    foreach (template ${sc_module_templates})
        get_filename_component(script "${template}" NAME_WLE) # deploy.sh.in -> deploy.sh
        # Configured into the build tree first, so the comparison is against what this
        # module's copy should say, not against the unsubstituted template.
        configure_file("${template}" "${CMAKE_CURRENT_BINARY_DIR}/sc-scripts/${script}" @ONLY)
        sc_cache_helper("${CMAKE_CURRENT_BINARY_DIR}/sc-scripts/${script}"
                "${sc_module_scripts}/${script}" "${script}")
        if (EXISTS "${sc_module_scripts}/${script}")
            file(CHMOD "${sc_module_scripts}/${script}"
                    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
        endif ()
    endforeach ()
endif ()

if (sc_helpers_drifted)
    message(STATUS "The refreshed helpers take effect on the next configure")
endif ()

