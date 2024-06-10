cmake_minimum_required (VERSION 3.7.2)
project(CG-for-fpga CXX)

find_package(MPI REQUIRED)

add_subdirectory(googletest)
add_subdirectory(challenge)
add_subdirectory(test)

