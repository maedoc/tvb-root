cmake_minimum_required(VERSION 3.30)
project(libtvbk)

add_compile_options(-Wall -Wextra -Wpedantic -msse4.2 -fopenmp-simd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(tvbk SHARED src/nodes.c)
