// dump_energy.cpp

#include <fstream>
#include <string>
#include "dump_energy.hpp"

using std::string;
using std::ofstream;

DumpEnergy::DumpEnergy(int freq, string outFile) : Dump(freq, true, outFile) {}

void DumpEnergy::update(int time, const NucPosModel& model, ofstream& writer) {
  writer << time << " " << model.getEnergy() << "\n";
}
