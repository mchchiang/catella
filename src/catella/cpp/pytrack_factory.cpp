// pytrack_factory.cpp

#include <memory>
#include "dtype.hpp"
#include "pytrack.hpp"
#include "pytrack_factory.hpp"
#include "pytrack_pos.hpp"
#include "pytrack_energy.hpp"

using std::shared_ptr;

shared_ptr<PyTrackBase> PyTrackFactory::createPosTrack(lint printFreq) {
  return std::make_shared<PyTrackPos>(printFreq);
}

shared_ptr<PyTrackBase> PyTrackFactory::createEnergyTrack(lint printFreq) {
  return std::make_shared<PyTrackEnergy>(printFreq);
}
