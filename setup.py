import os
from setuptools import setup, Extension
import pybind11
import glob

cpp_files = glob.glob("nucmc/cpp/*.cpp")

# Handle Conda environment for HDF5 and Headers
conda_prefix = os.environ.get("CONDA_PREFIX")
if not conda_prefix:
    # Fall back to common system paths if not in Conda
    include_dirs = [pybind11.get_include(), "nucmc/cpp"]
    library_dirs = []
else:
    include_dirs = [
        pybind11.get_include(),
        "nucmc/cpp",
        os.path.join(conda_prefix, "include")
    ]
    library_dirs = [os.path.join(conda_prefix, "lib")]
    
ext_modules = [
    Extension(
        "nucmc_cpp",
        sources=cpp_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["hdf5"],
        language="c++",
        extra_compile_args=["-std=c++17","-O3"],
    )
]

setup(
    ext_modules=ext_modules,
)
        
