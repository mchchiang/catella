cmake_minimum_required(VERSION 3.15)
project(nucmc_cpp LANGUAGES CXX)

# Pick up conda-installed packages (HDF5, pybind11)
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

file(GLOB CPP_SOURCES "src/nucmc/cpp/*.cpp")

python_add_library(nucmc_cpp MODULE ${CPP_SOURCES} WITH_SOABI)
target_link_libraries(nucmc_cpp PRIVATE pybind11::headers HDF5::HDF5)
target_include_directories(nucmc_cpp PRIVATE src/nucmc/cpp)
target_compile_features(nucmc_cpp PRIVATE cxx_std_17)
target_compile_options(nucmc_cpp PRIVATE -O3)

# Install to wheel root alongside the nucmc package
install(TARGETS nucmc_cpp DESTINATION .)
