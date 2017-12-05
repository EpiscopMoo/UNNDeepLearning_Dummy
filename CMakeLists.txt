cmake_minimum_required(VERSION 3.8)
project(neuro_0_0)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp NNetwork.cpp NNetwork.h typedefs.h Utils.h)
add_executable(neuro_0_0 ${SOURCE_FILES})

project(test)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES test/vec_test.cpp NNetwork.cpp NNetwork.h typedefs.h Utils.h)

find_package (Threads)
add_executable(neuro_0_0_test ${SOURCE_FILES})
target_link_libraries(neuro_0_0_test gtest)
target_link_libraries (neuro_0_0_test ${CMAKE_THREAD_LIBS_INIT})