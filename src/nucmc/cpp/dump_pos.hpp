// dump_pos.hpp
// Dump the nucleosome positions

#ifndef DUMP_POS_HPP
#define DUMP_POS_HPP

#include <string>
#include <fstream>
#include "dtype.hpp"
#include "model.hpp"
#include "dump_dat.hpp"

class DumpPos : public DumpDat {
public:
  DumpPos(lint printFreq, const std::string& file);
  void dump(lint time, const NucPosModel& model,
	    std::ofstream& writer) override final;
};

#endif
