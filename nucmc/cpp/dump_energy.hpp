// dump_energy.hpp
// Dump the energy of the system

#ifndef DUMP_ENERGY_HPP
#define DUMP_ENERGY_HPP

#include <string>
#include <fstream>
#include "dtype.hpp"
#include "model.hpp"
#include "dump_dat.hpp"

class DumpEnergy : public DumpDat {
public:
  DumpEnergy(lint printFreq, const std::string& file);
  void dump(lint time, const NucPosModel& model,
	    std::ofstream& writer) override final;
};

#endif
