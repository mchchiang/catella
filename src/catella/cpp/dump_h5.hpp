// dump_h5.hpp

#ifndef DUMP_H5_HPP
#define DUMP_H5_HPP

#include <string>
#include <vector>
#include <hdf5.h>
#include "dtype.hpp"
#include "pytrack_obs.hpp"
#include "tracker.hpp"

class DumpH5 : public Tracker {

public:

  using OutputType = PyTrackObs::OutputType;

  DumpH5(lint freq, const std::string& file, OutputType opt);
  ~DumpH5();

  void initialize(lint time, const NucPosMCMC& model) override final;
  void update(lint time, const NucPosMCMC& model) override final;
  void finalize(lint time, const NucPosMCMC& model) override final;
  
private:

  std::string file;
  OutputType outType;  
  
  // HDF5 handles
  hid_t h5file;
  hid_t h5params;
  hid_t h5data;
  std::vector<lint> times;
  std::vector<double> energy;
  std::vector<double> temp;
  std::vector<std::vector<int> > nucpos;
  
};


#endif
