# gtest_discover_tests is stable in cmake-3.10.3
cmake_minimum_required(VERSION 3.10.3)
project(icecake)

# Adds options to the compilation of source files
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH}) # To allow CMake to locate our Find*.cmake files

add_compile_options(-Wall -Wno-unused-variable -Wpointer-arith -Werror)

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.1)
    # https://github.com/facebook/rocksdb/pull/2699
    # https://stackoverflow.com/questions/25525047/cmake-generator-expression-differentiate-c-c-code
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-faligned-new>)
endif ()

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

option(DEBUG_MODE "build in debug mode" OFF)
option(BUILD_TEST "build tests" ON)

if (DEBUG_MODE)
    add_compile_options(-g)
else ()
    add_compile_options(-O2)
endif ()

add_subdirectory(src)

if (BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()