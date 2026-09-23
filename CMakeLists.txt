cmake_minimum_required(VERSION 3.22)
set(SC_MODULE ai)
project(simply-ai-${SC_MODULE} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options:
option(SC_AI_BUILD_TESTS "Build simply-cpp-ai tests" ${PROJECT_IS_TOP_LEVEL})

# The build helpers - get_sc_version(), add_sc_object(), add_sc_test(),
# find_or_install_package() - come from simply-cpp core so every module configures the
# same way instead of keeping its own copy. The bootstrap takes them from an installed
# sc package, the copy in cmake/, or the sc repository, whichever it finds first.
include(cmake/sc_bootstrap.cmake)

get_sc_version()
message(STATUS "SimplyCPP ${SC_MODULE} Version: ${SC_VERSION}")

################
# Dependencies #
################

#models
if (APPLE)
    execute_process(COMMAND brew --prefix simply-cpp-models OUTPUT_VARIABLE BREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if (BREW_PREFIX)
        message(STATUS "Found Homebrew package: ${BREW_PREFIX}")
    endif ()
    set(SIMPLY_CPP_MODEL_DIR "${BREW_PREFIX}/share/simply-cpp-models")
else ()
    set(SIMPLY_CPP_MODEL_DIR "/opt/simply-cpp/models")
endif ()

find_file(SIMPLY_CPP_MODEL NAMES yolo26n.onnx PATHS ${SIMPLY_CPP_MODEL_DIR})
if (NOT SIMPLY_CPP_MODEL)
    if (APPLE)
        execute_process(COMMAND brew -y install simply-cpp-models)
    else ()
        execute_process(COMMAND apt -y install simply-cpp-models)
    endif ()
    find_file(SIMPLY_CPP_MODEL NAMES yolo26n.onnx PATHS ${SIMPLY_CPP_MODEL_DIR})
    if (NOT SIMPLY_CPP_MODEL)
        message(FATAL_ERROR "Could not find onnx models in ${SIMPLY_CPP_MODEL_DIR}.")
    endif ()
endif ()
configure_file(include/config.h.in ${CMAKE_BINARY_DIR}/generated/config.h @ONLY)

find_or_install_package(sc-core simply-cpp-dev simply-cpp)
find_or_install_package(sc-image simply-cpp-image-dev simply-cpp-image)
find_or_install_package(nlohmann_json nlohmann-json3-dev nlohmann-json)
find_or_install_package(onnxruntime simply-cpp-onnxruntime onnxruntime)

###########
# Objects #
###########
set(SOURCE_OBJECTS)
set(SOURCE_LIBRARIES)
set(SOURCE_LINK_LIBRARIES)

add_sc_object(onnx
        LINK_LIBRARIES sc::sc-core sc::sc-image onnxruntime::onnxruntime ${CMAKE_DL_LIBS}
        PUBLIC_LINK_LIBRARIES sc::sc-core sc::sc-image onnxruntime::onnxruntime ${CMAKE_DL_LIBS})
target_include_directories(sc-obj-onnx PRIVATE ${CMAKE_BINARY_DIR}/generated)

add_sc_object(yolo
        LINK_LIBRARIES sc::sc-image nlohmann_json::nlohmann_json
        PUBLIC_LINK_LIBRARIES sc::sc-image nlohmann_json::nlohmann_json)

######################
# Exported Libraries #
######################
add_sc_libraries(DESCRIPTION "simply-cpp-ai")

################
# Installation #
################
install_sc_module()

#########
# Tests #
#########
if (SC_AI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests sc-ai-test)
endif ()

############
# Binaries #
############

add_executable(sc-ai-demo apps/sc-ai-demo.cpp)
target_link_libraries(sc-ai-demo PRIVATE sc::sc-ai-shared)
install(TARGETS sc-ai-demo RUNTIME DESTINATION bin COMPONENT runtime)

#############
# Packaging #
#############
package_sc_module(DEPENDS simply-cpp-dev simply-cpp-image-dev simply-cpp-onnxruntime simply-cpp-models)