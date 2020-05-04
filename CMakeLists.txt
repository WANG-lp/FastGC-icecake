# gtest_discover_tests is stable in cmake-3.10.3
cmake_minimum_required(VERSION 3.10.3)
project(icecake LANGUAGES C CXX CUDA)

# Adds options to the compilation of source files
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH}) # To allow CMake to locate our Find*.cmake files

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.1)
    # https://github.com/facebook/rocksdb/pull/2699
    # https://stackoverflow.com/questions/25525047/cmake-generator-expression-differentiate-c-c-code
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-faligned-new>)
endif ()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(DEBUG_MODE "build in debug mode" OFF)
option(BUILD_TEST "build tests" ON)
option(BUILD_DALI_PLUGIN "build dali plugin" ON)

if (DEBUG_MODE)
    add_compile_options(-g)
else ()
    add_compile_options(-O2)
endif ()

include_directories(3rd-deps/spdlog/include)
include_directories(3rd-deps/parallel-hashmap)
include_directories(3rd-deps/DALI)
include_directories(3rd-deps/pybind11/include)
include_directories(3rd-deps/cereal/include)
add_subdirectory(3rd-deps/pybind11)
add_subdirectory(src)

if (BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()