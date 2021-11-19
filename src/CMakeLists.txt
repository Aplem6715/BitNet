cmake_minimum_required(VERSION 3.0.0)

file(GLOB_RECURSE SOURCE RELATIVE "${CMAKE_SOURCE_DIR}/src" "*.cpp" "*.h")

function(print_list TITLE LIST)
    message(STATUS "##### ${TITLE} #####")
    foreach(ITEM ${LIST})
        message(STATUS "${ITEM}")
    endforeach()
    message("")
endfunction()
# print_list("GLOB RELATIVE ${CMAKE_SOURCE_DIR}/src" "${SOURCE}")

add_executable(BitNet ${SOURCE})
