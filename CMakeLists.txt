cmake_minimum_required (VERSION 3.8)
project ("CNN")

message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fopenmp -mavx")
set(CMAKE_BUILD_TYPE "Release")
add_executable (CNN ${source_list})
