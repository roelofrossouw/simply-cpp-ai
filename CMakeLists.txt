cmake_minimum_required(VERSION 3.22)
set(SC_MODULE ai)
project(simply-cpp-${SC_MODULE} LANGUAGES CXX)
include(cmake/sc_bootstrap.cmake)

# Options:
option(SC_AI_BUILD_TESTS "Build simply-cpp-ai tests" ${PROJECT_IS_TOP_LEVEL})

get_sc_version()
message(STATUS "SimplyCPP ${SC_MODULE} Version: ${SC_VERSION}")

################
# Dependencies #
################

find_or_install_package(sc-core simply-cpp-dev simply-cpp)
find_or_install_package(sc-image simply-cpp-image-dev simply-cpp-image)
find_or_install_package(nlohmann_json nlohmann-json3-dev nlohmann-json)
find_or_install_package(onnxruntime simply-cpp-onnxruntime onnxruntime)
find_or_install_package(sc-models simply-cpp-models simply-cpp-models)
set(SIMPLY_CPP_MODEL_DIR "${SC_MODELS_DIR}")
configure_file(include/config.h.in ${CMAKE_BINARY_DIR}/generated/config.h @ONLY)

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

add_sc_object(facedetector
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
package_sc_module(
        DEPENDS simply-cpp-onnxruntime simply-cpp-models
        DEV_DEPENDS simply-cpp-dev simply-cpp-image-dev simply-cpp-onnxruntime simply-cpp-models
)