// dump_pos.cpp

#include <fstream>
#include <string>
#include "dtype.hpp"
#include "model.hpp"
#include "dump_dat.hpp"
#include "dump_pos.hpp"

using std::string;
using std::ofstream;

DumpPos::DumpPos(lint freq, const string& outFile) :
  DumpDat(freq, outFile, true) {}

void DumpPos::dump(lint time, const NucPosModel& model, ofstream& writer) {
  auto nucpos = model.getNucPos();
  auto nucbp = model.getParams().nucbp;
  writer << "Nucleosomes: " << nucpos.size() << "\n";
  writer << "Timesteps: " << time << "\n";
  for (auto& pos : nucpos) {
    writer << pos << " " << pos+nucbp << "\n";
  }
}
