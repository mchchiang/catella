// pytrack_energy.cpp

#include "dtype.hpp"
#include "mcmc.hpp"
#include "pytrack_energy.hpp"

PyTrackEnergy::PyTrackEnergy(lint freq) : PyTrack(freq) {}

void PyTrackEnergy::track(lint time, const NucPosMCMC& model) {
  data.push_back(model.getEnergy());
}
