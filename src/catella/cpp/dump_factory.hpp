// dump_factory.hpp

#ifndef DUMP_FACTORY_HPP
#define DUMP_FACTORY_HPP

#include <string>
#include <memory>
#include "dtype.hpp"
#include "dump_dat.hpp"
#include "dump_pos.hpp"
#include "dump_energy.hpp"

class DumpFactory {
public:
  std::shared_ptr<DumpDat> createPosDump(lint printFreq,
					 const std::string& outFile);  
  std::shared_ptr<DumpDat> createEnergyDump(lint printFreq,
					    const std::string& outFile);
};

#endif
