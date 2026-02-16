// dump_energy.cpp

#include <fstream>
#include <string>
#include "dtype.hpp"
#include "model.hpp"
#include "dump_dat.hpp"
#include "dump_energy.hpp"

using std::string;
using std::ofstream;

DumpEnergy::DumpEnergy(lint freq, const string& outFile) :
  DumpDat(freq, outFile, true) {}

void DumpEnergy::dump(lint time, const NucPosModel& model, ofstream& writer) {
  writer << time << " " << model.getEnergy() << "\n";
}
