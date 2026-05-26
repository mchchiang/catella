// dump_h5.cpp

#include <stdexcept>
#include <string>
#include <vector>
#include <hdf5.h>
#include "dtype.hpp"
#include "model.hpp"
#include "dump_h5.hpp"

#define MAX_CHUNK_SIZE 1024

using std::string;
using std::vector;

// Helper to create the compression property list
hid_t createCompressionPlist(hsize_t rank, const hsize_t* dims,
			     int level = 4) {
  hid_t plist = H5Pcreate(H5P_DATASET_CREATE);

  // Compression requires chunking
  // For 1D vectors, chunk in blocks of 1024 (or the total size if smaller)
  hsize_t chunk_dims[1];
  chunk_dims[0] = (dims[0] < MAX_CHUNK_SIZE) ? dims[0] : MAX_CHUNK_SIZE;

  // Compression is not possible/useful for dims[0] = 0
  if (dims[0] > 0) {
    H5Pset_chunk(plist, rank, chunk_dims);
    H5Pset_deflate(plist, level);
  }
  return plist;
}

// Helper function for writing a vector to an h5 file
template <typename T>
void writeVector(const string& name, const vector<T>& vec, hid_t group) {
  hsize_t dims[1] = {vec.size()};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dtype = h5_dtype<T>();

  // Create property list with compression
  hid_t plist = createCompressionPlist(1, dims);
  hid_t dset = H5Dcreate(group, name.c_str(), dtype, space,
			 H5P_DEFAULT, plist, H5P_DEFAULT);
  H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
  H5Pclose(plist);
  H5Dclose(dset);
  H5Sclose(space);
}

// For writing a vector of vectors
template <typename T>
void writeVector(const string& name, const vector<vector<T> >& vec,
		 hid_t group) {
  vector<T> flat;
  vector<hsize_t> offset;
  offset.reserve(vec.size()+1);
  offset.push_back(0);
  for (const auto& v : vec) {
    flat.insert(flat.end(), v.begin(), v.end());
    offset.push_back(flat.size());
  }
  writeVector(name+"_flat", flat, group);
  writeVector(name+"_offset", offset, group);
}

// For writing strings
void writeVector(const string& name, const vector<string>& vec, hid_t group) {
  hsize_t dims[1] = {vec.size()};
  hid_t space = H5Screate_simple(1, dims, nullptr);
  hid_t dtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(dtype, H5T_VARIABLE);
  vector<const char*> cstrs;
  for (const auto& s: vec) {
    cstrs.push_back(s.c_str());
  }
  // Create property list with compression
  hid_t plist = createCompressionPlist(1, dims);
  hid_t dset = H5Dcreate(group, name.c_str(), dtype, space,
			 H5P_DEFAULT, plist, H5P_DEFAULT);
  H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, cstrs.data());
  H5Pclose(plist);
  H5Dclose(dset);
  H5Tclose(dtype);
  H5Sclose(space);  
}

template <typename T>
void writeAttribute(const string& name, const T& value, hid_t group) {
  hid_t dtype = h5_dtype<T>();
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate(group, name.c_str(), dtype, space,
			 H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, dtype, &value);
  H5Aclose(attr);
  H5Sclose(space);
}

void writeAttribute(const string& name, const string& value, hid_t group) {
  hid_t dtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(dtype, H5T_VARIABLE);
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate(group, name.c_str(), dtype, space,
			 H5P_DEFAULT, H5P_DEFAULT);
  const char* cstr = value.c_str();
  H5Awrite(attr, dtype, &cstr);
  H5Aclose(attr);
  H5Tclose(dtype);
  H5Sclose(space);
}

DumpH5::DumpH5(lint freq, const string& f, OutputType otype) :
  Tracker(freq), file(f), outType(otype) {}

DumpH5::~DumpH5() {}

void DumpH5::initialize(lint time, const NucPosModel& model) {
  h5file = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (h5file < 0) {
    throw std::runtime_error("Failed to create the HDF5 file " + file);
  }
  h5data = H5Gcreate(h5file, "/data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  h5params = H5Gcreate(h5file, "/params", H5P_DEFAULT, H5P_DEFAULT,
		       H5P_DEFAULT);

  // Write parameters
  const auto& params = model.getParams();
  writeAttribute("nucbp", params.nucbp, h5params);
  writeAttribute("nbp", params.nbp, h5params);
  writeAttribute("llink", params.llink, h5params);
  writeAttribute("mu", params.mu, h5params);
  writeAttribute("seed", params.seed, h5params);
}

void DumpH5::update(lint time, const NucPosModel& model) {
  times.push_back(time);
  if (static_cast<int>(outType) & static_cast<int>(OutputType::Energy)) {
    energy.push_back(model.getEnergy());
  }
  if (static_cast<int>(outType) & static_cast<int>(OutputType::Temp)) {
    temp.push_back(model.getTemp());
  }  
  if (static_cast<int>(outType) & static_cast<int>(OutputType::Position)) {
    nucpos.push_back(model.getNucPos());
  }
}

void DumpH5::finalize(lint time, const NucPosModel& model) {  
  writeVector("time", times, h5data);
  if (static_cast<int>(outType) & static_cast<int>(OutputType::Energy)) {
    writeVector("energy", energy, h5data);
  }
  if (static_cast<int>(outType) & static_cast<int>(OutputType::Temp)) {
    writeVector("temp", temp, h5data);
  }  
  if (static_cast<int>(outType) & static_cast<int>(OutputType::Position)) {  
    writeVector("position", nucpos, h5data);
  }
  H5Gclose(h5data);
  H5Gclose(h5params);
  H5Fclose(h5file);
}
