// dump_factory.cpp

#include <string>
#include <memory>
#include "dump.hpp"
#include "dump_factory.hpp"
#include "dump_pos.hpp"
#include "dump_energy.hpp"

using std::string;
using std::shared_ptr;

shared_ptr<Dump> DumpFactory::createPosDump(int printFreq, string outFile) {
  return std::make_shared<DumpPos>(printFreq, outFile);
}

shared_ptr<Dump> DumpFactory::createEnergyDump(int printFreq, string outFile) {
  return std::make_shared<DumpEnergy>(printFreq, outFile);
}
