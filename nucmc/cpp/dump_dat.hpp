// dump_dat.hpp

#ifndef DUMP_DAT_HPP
#define DUMP_DAT_HPP

#include <fstream>
#include <string>
#include "dtype.hpp"
#include "model.hpp"
#include "tracker.hpp"

class NucPosModel;

class DumpDat : public Tracker {

private:
  std::string file;
  bool append;  
  std::ofstream fileStream;

public:
  DumpDat(lint printFreq, const std::string& file, bool append);
  virtual ~DumpDat() = default;
  virtual void dump(lint time, const NucPosModel& model,
		    std::ofstream& writer) = 0;
  void initialize(lint time, const NucPosModel& model) override final;
  void update(lint time, const NucPosModel& model) override final;
  void finalize(lint time, const NucPosModel& model) override final;
};

#endif
