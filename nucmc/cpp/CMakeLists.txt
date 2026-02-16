cmake_minimum_required(VERSION 3.14)
project(nucmc)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

file(GLOB CPP_SOURCES "*.cpp")   # all .cpp files in cpp/
pybind11_add_module(_core ${CPP_SOURCES})
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
