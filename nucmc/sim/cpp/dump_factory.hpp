// dump_factory.hpp

#ifndef DUMP_FACTORY_HPP
#define DUMP_FACTORY_HPP

#include <string>
#include <memory>
#include "dump.hpp"
#include "dump_pos.hpp"
#include "dump_energy.hpp"

class DumpFactory {
public:
  std::shared_ptr<Dump> createPosDump(int printFreq, std::string outFile);  
  std::shared_ptr<Dump> createEnergyDump(int printFreq, std::string outFile);  
};

#endif
