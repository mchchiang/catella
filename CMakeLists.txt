cmake_minimum_required(VERSION 3.20)
project(catella_cpp LANGUAGES CXX)

# Pick up conda-installed packages (pybind11) when building inside a
# conda env. On Windows, conda installs CMake configs under Library/
# rather than directly under the env root.
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    if(WIN32)
        list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
    endif()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB CPP_SOURCES CONFIGURE_DEPENDS "src/catella/cpp/*.cpp")

python_add_library(catella_cpp MODULE ${CPP_SOURCES} WITH_SOABI)
target_link_libraries(catella_cpp PRIVATE pybind11::headers)
target_include_directories(catella_cpp PRIVATE src/catella/cpp)
target_compile_features(catella_cpp PRIVATE cxx_std_17)

# Install to wheel root alongside the catella package
install(TARGETS catella_cpp DESTINATION .)
