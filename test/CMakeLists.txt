cmake_minimum_required (VERSION 3.12)

file(GLOB_RECURSE SOURCE RELATIVE "${CMAKE_SOURCE_DIR}/test" "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE TEST_SOURCE RELATIVE "${CMAKE_SOURCE_DIR}/test" "*.cpp" "*.h")
list(REMOVE_ITEM SOURCE "../src/main.cpp")

function(print_list TITLE LIST)
    message(STATUS "##### ${TITLE} #####")
    foreach(ITEM ${LIST})
        message(STATUS "${ITEM}")
    endforeach()
    message("")
endfunction()

# print_list("GLOB RELATIVE ${CMAKE_SOURCE_DIR}/src" "${SOURCE}")
# print_list("GLOB RELATIVE ${CMAKE_SOURCE_DIR}/test" "${TEST_SOURCE}")

add_executable(UnitTest ${SOURCE} ${TEST_SOURCE})
target_link_libraries(UnitTest gtest gtest_main pthread)

include(GoogleTest)
gtest_add_tests(TARGET UnitTest)