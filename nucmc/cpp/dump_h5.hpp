// dump_h5.hpp

#ifndef DUMP_H5_HPP
#define DUMP_H5_HPP

#include <string>
#include <vector>
#include <hdf5.h>
#include "dtype.hpp"
#include "tracker.hpp"

class DumpH5 : public Tracker {

public:

  enum class OutputType : uint {
    Energy = 1 << 0,
    Position = 1 << 1,
    Temp = 1 << 2,
    All = Energy | Position | Temp
  };
  
  DumpH5(lint freq, const std::string& file, OutputType opt);
  ~DumpH5();
  
  void initialize(lint time, const NucPosModel& model) override final;
  void update(lint time, const NucPosModel& model) override final;
  void finalize(lint time, const NucPosModel& model) override final;
  
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
