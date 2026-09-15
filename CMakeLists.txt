cmake_minimum_required(VERSION 3.20)
project(catella_cpp LANGUAGES CXX)

# Pick up conda-installed packages (HDF5, pybind11). On Windows, conda
# installs headers, libs and CMake configs under Library/ rather than
# directly under the env root.
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    if(WIN32)
        list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
    endif()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# CMake's bundled FindHDF5.cmake module relies on Unix naming
# conventions (find_library auto-adds a "lib" prefix), which does not
# match conda-forge's Windows HDF5 package (still "lib"-prefixed).
# Search explicitly on Windows instead of relying on that module.
if(WIN32)
    find_path(HDF5_INCLUDE_DIRS
        NAMES hdf5.h
        HINTS "$ENV{CONDA_PREFIX}/Library/include")
    find_library(HDF5_LIBRARIES
        NAMES hdf5 libhdf5
        HINTS "$ENV{CONDA_PREFIX}/Library/lib")
    if(NOT HDF5_INCLUDE_DIRS OR NOT HDF5_LIBRARIES)
        message(FATAL_ERROR "Could not find HDF5 under "
            "$ENV{CONDA_PREFIX}/Library")
    endif()
    add_library(HDF5::HDF5 UNKNOWN IMPORTED)
    set_target_properties(HDF5::HDF5 PROPERTIES
        IMPORTED_LOCATION "${HDF5_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}")
else()
    find_package(HDF5 REQUIRED COMPONENTS C)
endif()

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

# Install to wheel root alongside the catella package
install(TARGETS catella_cpp DESTINATION .)
