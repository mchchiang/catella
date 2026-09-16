// dump_dat.hpp

#ifndef DUMP_DAT_HPP
#define DUMP_DAT_HPP

#include <fstream>
#include <string>
#include "dtype.hpp"
#include "mcmc.hpp"
#include "tracker.hpp"

class NucPosMCMC;

class DumpDat : public Tracker {

private:
  std::string file;
  bool append;  
  std::ofstream fileStream;

public:
  DumpDat(lint printFreq, const std::string& file, bool append);
  virtual ~DumpDat() = default;
  virtual void dump(lint time, const NucPosMCMC& model,
		    std::ofstream& writer) = 0;
  void initialize(lint time, const NucPosMCMC& model) override final;
  void update(lint time, const NucPosMCMC& model) override final;
  void finalize(lint time, const NucPosMCMC& model) override final;
};

#endif
