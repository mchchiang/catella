// pytrack_factory.hpp

#ifndef PYTRACK_FACTORY_HPP
#define PYTRACK_FACTORY_HPP

#include <string>
#include <memory>
#include "pytrack.hpp"
#include "pytrack_pos.hpp"
#include "pytrack_energy.hpp"

class PyTrackFactory {
public:
  std::shared_ptr<PyTrackBase> createPosTrack(int printFreq);  
  std::shared_ptr<PyTrackBase> createEnergyTrack(int printFreq);
};

#endif
