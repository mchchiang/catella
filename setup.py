from setuptools import setup, Extension
import pybind11
import glob

cpp_files = glob.glob("nucmc/sim/cpp/*.cpp")

ext_modules = [
    Extension(
        "nucmc.sim._core",
        sources=cpp_files,
        include_dirs=[pybind11.get_include(), "nucmc/sim/cpp"],
        language="c++",
        )
    ]

setup(
    name="nucmc",
    version="0.1.0",
    packages=["nucmc","nucmc.sim","nucmc.prep","nucmc.post"],
    ext_modules=ext_modules,
)
        
