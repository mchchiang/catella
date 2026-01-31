// dump_pos.hpp
// Dump the nucleosome positions

#ifndef DUMP_POS_HPP
#define DUMP_POS_HPP

#include <string>
#include "dump.hpp"

class DumpPos : public Dump {
public:
  DumpPos(int printFreq, std::string file);
  void update(int time, const NucPosModel& model,
	      std::ofstream& writer) override;
};

#endif
