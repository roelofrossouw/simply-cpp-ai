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

# The demos use sc::console subheadings and SC_SHOW with commas (1.5.15).
find_or_install_package(sc-core simply-cpp-dev simply-cpp 1.5.21)
find_or_install_package(sc-image simply-cpp-image-dev simply-cpp-image 1.2.8)
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

add_sc_object(ocr
        LINK_LIBRARIES sc::sc-image nlohmann_json::nlohmann_json
        PUBLIC_LINK_LIBRARIES sc::sc-image nlohmann_json::nlohmann_json)
target_include_directories(sc-obj-ocr PRIVATE ${CMAKE_BINARY_DIR}/generated)

add_sc_object(facedetector
        LINK_LIBRARIES sc::sc-image nlohmann_json::nlohmann_json
        PUBLIC_LINK_LIBRARIES sc::sc-image nlohmann_json::nlohmann_json)

add_sc_object(face
        LINK_LIBRARIES sc::sc-image nlohmann_json::nlohmann_json
        PUBLIC_LINK_LIBRARIES sc::sc-image nlohmann_json::nlohmann_json)

add_sc_object(identity
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
# Examples #
############
add_sc_example(sc-ai-demo)
add_sc_example(sc-ai-ocr)
add_sc_example(sc-ai-identity)

############
# Binaries #
############
find_or_install_package(sc-db simply-cpp-db-dev simply-cpp-db 1.2.1)

#add_executable(load-faces apps/load-faces.cpp)
#target_link_libraries(load-faces PRIVATE sc::sc-ai sc::sc-db)
#install(TARGETS load-faces RUNTIME DESTINATION bin COMPONENT runtime)

add_executable(process-id apps/process-id.cpp)
target_link_libraries(process-id PRIVATE sc::sc-ai sc::sc-db)
install(TARGETS process-id RUNTIME DESTINATION bin COMPONENT runtime)

add_executable(run-ocr apps/run-ocr.cpp)
target_link_libraries(run-ocr PRIVATE sc::sc-ai)

#############
# Packaging #
#############
package_sc_module(
        DEPENDS simply-cpp-onnxruntime simply-cpp-models
        DEV_DEPENDS "simply-cpp-dev (>= 1.5.21~)" "simply-cpp-image-dev (>= 1.2.8~)" simply-cpp-onnxruntime simply-cpp-models
)