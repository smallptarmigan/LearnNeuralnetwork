cmake_minimum_required(VERSION 3.17)
project(task)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>")

set(CMAKE_CXX_STANDARD 17)

add_executable(task main.cpp neuralnet.cpp neuralnet.h model.cpp dataset.cpp)