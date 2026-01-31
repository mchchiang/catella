// dump_energy.hpp
// Dump the energy of the system

#ifndef DUMP_ENERGY_HPP
#define DUMP_ENERGY_HPP

#include <string>
#include "dump.hpp"

class DumpEnergy : public Dump {
public:
  DumpEnergy(int printFreq, std::string file);
  void update(int time, const NucPosModel& model,
	      std::ofstream& writer) override;
};

#endif
