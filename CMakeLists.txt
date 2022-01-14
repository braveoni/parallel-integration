cmake_minimum_required(VERSION 3.20)
project(parallel)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

add_executable(parallel main.cpp)
target_link_libraries(parallel PUBLIC OpenMP::OpenMP_CXX pthread)