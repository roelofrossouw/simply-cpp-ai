## TESTS ##
# Resources for Tests #
set(RESOURCE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resource")
set(RESOURCE_DESTINATION_DIR "${CMAKE_CURRENT_BINARY_DIR}/resource")
add_custom_target(copy_sc_ai_resources ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${RESOURCE_SOURCE_DIR} ${RESOURCE_DESTINATION_DIR})


# What every test in this directory links against.
set(SC_TEST_LINK_LIBRARIES sc-ai)

add_sc_test(yolo TIMEOUT 120)
add_dependencies(test-yolo copy_sc_ai_resources)
add_sc_test(face TIMEOUT 120)
add_dependencies(test-face copy_sc_ai_resources)
