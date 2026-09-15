cmake_minimum_required(VERSION 3.20)
project(catella_cpp LANGUAGES CXX)

# Pick up conda-installed packages (HDF5, pybind11)
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# Ensure the installed extension can find libhdf5 at import time,
# rather than relying implicitly on the compiler wrapper's LDFLAGS.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "src/catella/cpp/*.cpp")

python_add_library(catella_cpp MODULE ${CPP_SOURCES} WITH_SOABI)
target_link_libraries(catella_cpp PRIVATE pybind11::headers HDF5::HDF5)
target_include_directories(catella_cpp PRIVATE src/catella/cpp)
target_compile_features(catella_cpp PRIVATE cxx_std_17)
target_compile_options(catella_cpp PRIVATE -O3)

# Install to wheel root alongside the catella package
install(TARGETS catella_cpp DESTINATION .)
