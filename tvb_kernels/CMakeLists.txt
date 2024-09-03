cmake_minimum_required(VERSION 3.20)
project(libtvbk)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -msse4.2 -fopenmp-simd)
add_library(tvbk SHARED src/nodes.c)
