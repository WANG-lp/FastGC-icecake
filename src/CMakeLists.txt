include(TestBigEndian)
TEST_BIG_ENDIAN(IS_BIG_ENDIAN)
if (IS_BIG_ENDIAN)
    message(STATUS "System is BIG_ENDIAN")
    add_definitions(-D_BIG_ENDIAN)
else ()
    message(STATUS "System is LITTLE_ENDIAN")
endif ()
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA 10.0 REQUIRED)
find_package (Python COMPONENTS Interpreter Development)
#find_package( OpenCV REQUIRED NO_DEFAULT_PATH PATHS "/home/lwangay/root/opencv-3.4.10/usr/local/share/OpenCV")
find_package( OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
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
target_link_libraries(icecake_c PRIVATE Threads::Threads ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX ${OpenCV_LIBS})
set_property(TARGET icecake_c PROPERTY POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(pyicecake SHARED icecake-python.cpp ${CPP_SRC})
target_link_libraries(pyicecake PRIVATE Threads::Threads ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX ${OpenCV_LIBS})

if (BUILD_DALI_PLUGIN)
    add_subdirectory(dali-ops)
endif()

add_executable(jpeg_dec jpeg_decoder.cpp)
target_link_libraries(jpeg_dec PRIVATE Threads::Threads ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

