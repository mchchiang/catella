// pytrack_factory.cpp

#include <string>
#include <memory>
#include "pytrack.hpp"
#include "pytrack_factory.hpp"
#include "pytrack_pos.hpp"
#include "pytrack_energy.hpp"

using std::shared_ptr;

shared_ptr<PyTrackBase> PyTrackFactory::createPosTrack(int printFreq) {
  return std::make_shared<PyTrackPos>(printFreq);
}

shared_ptr<PyTrackBase> PyTrackFactory::createEnergyTrack(int printFreq) {
  return std::make_shared<PyTrackEnergy>(printFreq);
}
