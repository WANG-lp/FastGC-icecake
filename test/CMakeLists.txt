find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
find_package(OpenMP REQUIRED)
add_library(test_main OBJECT tests.cpp common.cpp)

add_executable(testlib testlib.cpp $<TARGET_OBJECTS:test_main>)
target_link_libraries(testlib GTest::GTest GTest::Main icecake_c)

gtest_discover_tests(testlib)


add_executable(valgrind_test valgrind_test.cpp common.cpp)
target_link_libraries(valgrind_test icecake_c)

find_package(FUSE)
add_executable(rofs rofs.cpp)
target_compile_options(rofs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64>)
target_link_libraries(rofs ${FUSE_LIBRARIES})

add_executable(opencv_test opencv_test.cpp)
target_link_libraries(opencv_test icecake_c OpenMP::OpenMP_CXX)


add_executable(test_jpegdec test_jpeg_decoder.cpp)
target_link_libraries(test_jpegdec jpegdec OpenMP::OpenMP_CXX)
