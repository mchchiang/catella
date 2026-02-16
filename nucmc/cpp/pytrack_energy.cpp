// pytrack_energy.cpp

#include "dtype.hpp"
#include "model.hpp"
#include "pytrack_energy.hpp"

PyTrackEnergy::PyTrackEnergy(lint freq) : PyTrack(freq) {}

void PyTrackEnergy::track(lint time, const NucPosModel& model) {
  data.push_back(model.getEnergy());
}
