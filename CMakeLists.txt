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
find_or_fetch_package(sc-core GIT_REPOSITORY https://github.com/roelofrossouw/simply-cpp.git)
find_or_fetch_package(sc-image GIT_REPOSITORY https://github.com/roelofrossouw/simply-cpp-image.git)
## ONNXRuntime ##
find_or_install_package(onnxruntime onnxruntime onnxruntime)

###########
# Objects #
###########
set(SOURCE_OBJECTS)
set(SOURCE_LIBRARIES)
set(SOURCE_LINK_LIBRARIES)

add_sc_object(onnx LINK_LIBRARIES sc::sc-core sc::sc-image onnxruntime::onnxruntime PUBLIC_LINK_LIBRARIES sc::sc-core sc::sc-image onnxruntime::onnxruntime)
add_sc_object(yolo)

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
