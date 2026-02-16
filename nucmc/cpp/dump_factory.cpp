// dump_factory.cpp

#include <string>
#include <memory>
#include "dtype.hpp"
#include "dump_dat.hpp"
#include "dump_pos.hpp"
#include "dump_energy.hpp"
#include "dump_factory.hpp"

using std::string;
using std::shared_ptr;

shared_ptr<DumpDat> DumpFactory::createPosDump(lint printFreq,
					       const string& outFile) {
  return std::make_shared<DumpPos>(printFreq, outFile);
}

shared_ptr<DumpDat> DumpFactory::createEnergyDump(lint printFreq,
						  const string& outFile) {
  return std::make_shared<DumpEnergy>(printFreq, outFile);
}
