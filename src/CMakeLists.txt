include(TestBigEndian)
TEST_BIG_ENDIAN(IS_BIG_ENDIAN)
if (IS_BIG_ENDIAN)
    message(STATUS "System is BIG_ENDIAN")
    add_definitions(-D_BIG_ENDIAN)
else ()
    message(STATUS "System is LITTLE_ENDIAN")
endif ()
find_package(Threads REQUIRED)
find_package(CUDA 10.0 REQUIRED)
find_package (Python COMPONENTS Interpreter Development)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Python_INCLUDE_DIRS})
set(CPP_SRC
        ../include/icecake.hpp
        ../include/icecake_base.hpp
        utils/int2bytes.h
        utils/int2bytes.cpp
        utils/chromiumbase64.cpp
        icecake.cpp
        icecake-python.cpp
        )
set(CUDA_SRC
        cuda/cuda.cuh
        cuda/sub.cu
        )
add_library(icecake_c STATIC ${CPP_SRC} ${CUDA_SRC})
target_link_libraries(icecake_c PRIVATE Threads::Threads ${CUDA_LIBRARIES})
set_property(TARGET icecake_c PROPERTY POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(pyicecake SHARED icecake-python.cpp ${CPP_SRC})
target_link_libraries(pyicecake PRIVATE Threads::Threads ${CUDA_LIBRARIES})

if (BUILD_DALI_PLUGIN)
    add_subdirectory(dali-ops)
endif()