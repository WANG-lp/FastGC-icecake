include(TestBigEndian)
TEST_BIG_ENDIAN(IS_BIG_ENDIAN)
if (IS_BIG_ENDIAN)
    message(STATUS "System is BIG_ENDIAN")
    add_definitions(-D_BIG_ENDIAN)
else ()
    message(STATUS "System is LITTLE_ENDIAN")
endif ()
find_package(Threads REQUIRED)

set(CPP_SRC
        icecake.cpp
        )


add_library(icecake SHARED ${CPP_SRC})
target_link_libraries(icecake Threads::Threads)


cmake_policy(SET CMP0078 NEW)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14.0")
    cmake_policy(SET CMP0086 OLD)
endif ()
FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
FIND_PACKAGE(Python3 COMPONENTS Interpreter Development)
SET_PROPERTY(SOURCE python/icecake.i PROPERTY CPLUSPLUS ON)
SET_PROPERTY(SOURCE python/icecake.i PROPERTY SWIG_FLAGS "-includeall" "-O" "-builtin")
SET_PROPERTY(SOURCE python/icecake.i PROPERTY SWIG_FLAGS "-py3")
swig_add_library(pyicecake TYPE SHARED LANGUAGE python SOURCES python/icecake.i ${CPP_SRC})
target_include_directories(pyicecake PRIVATE ${Python3_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(pyicecake ${Python3_LIBRARIES} Threads::Threads)
