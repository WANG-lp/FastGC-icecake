find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA 10.2 REQUIRED)
find_package (Python COMPONENTS Interpreter Development)
#find_package( OpenCV REQUIRED NO_DEFAULT_PATH PATHS "/home/lwangay/root/opencv-3.4.10/usr/local/share/OpenCV")
find_package( OpenCV REQUIRED)

find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

find_package(THRIFT REQUIRED)
include_directories(${THRIFT_INCLUDE_DIR})

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
        gen-cpp/JPEGCache_constants.cpp
        gen-cpp/JPEGCache_types.cpp
        gen-cpp/JPEGCache.cpp
        )
set(CUDA_SRC
        cuda/cuda.cuh
        cuda/sub.cu
        )
#add_library(icecake_c STATIC ${CPP_SRC} ${CUDA_SRC})
#target_link_libraries(icecake_c PRIVATE Threads::Threads ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX ${OpenCV_LIBS})
#set_property(TARGET icecake_c PROPERTY POSITION_INDEPENDENT_CODE ON)

#pybind11_add_module(pyicecake SHARED icecake-python.cpp ${CPP_SRC})
#target_link_libraries(pyicecake PRIVATE Threads::Threads ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX ${OpenCV_LIBS})


if (BUILD_DALI_PLUGIN)
    add_subdirectory(dali-ops)
endif()

set(DECODER_CPP_SRC
        jpeg_decoder.cpp jpeg_decoder_export.cpp JCache.cpp
        ../include/JCache.hpp
        gen-cpp/JPEGCache_constants.cpp
        gen-cpp/JPEGCache_types.cpp
        gen-cpp/JPEGCache.cpp
        )

add_library(jpegdec ${DECODER_CPP_SRC})
target_link_libraries(jpegdec PRIVATE Threads::Threads ${THRIFT_LIBRARIES} ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
set_property(TARGET jpegdec PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(jcache_server main.cpp)
target_link_libraries(jcache_server PRIVATE Threads::Threads OpenMP::OpenMP_CXX jpegdec)


# GPUJPEG library
include_directories(${CMAKE_SOURCE_DIR}/GPUJPEG)

file(GLOB H_FILES ${CMAKE_SOURCE_DIR}/GPUJPEG/libgpujpeg/*.h)
file(GLOB C_FILES ${CMAKE_SOURCE_DIR}/GPUJPEG/src/*.c ${CMAKE_SOURCE_DIR}/GPUJPEG/src/*/*.cpp ${CMAKE_SOURCE_DIR}/GPUJPEG/src/*.cu)

cuda_add_library(gpujpeg ${H_FILES} ${C_FILES})
set_target_properties(gpujpeg PROPERTIES SOVERSION 0)
SET_TARGET_PROPERTIES(
  gpujpeg
PROPERTIES
  PUBLIC_HEADER "${H_FILES}"
)

# GPUJPEG commmand-line application
file(GLOB C_FILES ${CMAKE_SOURCE_DIR}/GPUJPEG/src/main.c)
cuda_add_executable(gpujpeg-bin ${C_FILES})
target_link_libraries(gpujpeg-bin gpujpeg OpenMP::OpenMP_CXX jpegdec)
set_target_properties(gpujpeg-bin PROPERTIES OUTPUT_NAME "gpujpeg")

cuda_add_executable(gpujpeg_test ${CMAKE_SOURCE_DIR}/GPUJPEG/src/decode_test.cpp)
target_link_libraries(gpujpeg_test gpujpeg OpenMP::OpenMP_CXX jpegdec)

pybind11_add_module(pjcache SHARED JCache-python.cpp)
target_link_libraries(pjcache PRIVATE Threads::Threads ${CUDA_LIBRARIES} gpujpeg OpenMP::OpenMP_CXX jpegdec)

