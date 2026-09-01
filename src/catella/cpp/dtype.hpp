// dtype.hpp

#ifndef DTYPE_HPP
#define DTYPE_HPP

#include <hdf5.h>
#include <cstdint>

using uint = uint32_t;
using lint = int64_t;
using ulint = uint64_t;

template <typename T>
hid_t h5_dtype();

template <>
inline hid_t h5_dtype<int>() {
    return H5T_NATIVE_INT;
}

template <>
inline hid_t h5_dtype<double>() {
    return H5T_NATIVE_DOUBLE;
}

template <>
inline hid_t h5_dtype<float>() {
    return H5T_NATIVE_FLOAT;
}

template <>
inline hid_t h5_dtype<uint>() {
  return H5T_NATIVE_UINT32;
}

template <>
inline hid_t h5_dtype<lint>() {
  return H5T_NATIVE_INT64;
}

template <>
inline hid_t h5_dtype<ulint>() {
  return H5T_NATIVE_UINT64;
}

#endif
