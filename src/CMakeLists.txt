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
include_directories(${CUDA_INCLUDE_DIRS})
set(CPP_SRC
        ../include/icecake.hpp
        ../include/icecake_base.hpp
        utils/int2bytes.h
        utils/int2bytes.cpp
        icecake.cpp

        )
set(CUDA_SRC
        cuda/cuda.cuh
        cuda/sub.cu
        )

add_library(icecake_c SHARED ${CPP_SRC} ${CUDA_SRC})
target_link_libraries(icecake_c PRIVATE Threads::Threads ${CUDA_LIBRARIES})

pybind11_add_module(pyicecake icecake-python.cpp ${CPP_SRC})
target_link_libraries(pyicecake PRIVATE Threads::Threads ${CUDA_LIBRARIES})

add_subdirectory(dali-ops)