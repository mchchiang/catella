// dump_pos.cpp

#include <fstream>
#include <string>
#include "dump_pos.hpp"

using std::string;
using std::ofstream;

DumpPos::DumpPos(int freq, string outFile) : Dump(freq, true, outFile) {}

void DumpPos::update(int time, const NucPosModel& model, ofstream& writer) {
  auto nucpos = model.getNucPos();
  auto nucbp = model.getNucbp();
  writer << "Nucleosomes: " << nucpos.size() << "\n";
  writer << "Timesteps: " << time << "\n";
  for (auto& pos : nucpos) {
    writer << pos << " " << pos+nucbp << "\n";
  }
}
