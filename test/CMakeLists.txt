find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_library(test_main OBJECT tests.cpp common.cpp)

add_executable(testlib testlib.cpp $<TARGET_OBJECTS:test_main>)
target_link_libraries(testlib GTest::GTest GTest::Main icecake)

gtest_discover_tests(testlib)


add_executable(valgrind_test valgrind_test.cpp common.cpp)
target_link_libraries(valgrind_test icecake)