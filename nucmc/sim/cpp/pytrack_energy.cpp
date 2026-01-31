// pytrack_energy.cpp

#include "pytrack_energy.hpp"

PyTrackEnergy::PyTrackEnergy(int freq) : PyTrack(freq) {}

void PyTrackEnergy::track(int time, const NucPosModel& model) {
  data.push_back(model.getEnergy());
}
