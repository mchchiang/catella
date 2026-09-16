// pytrack_pos.cpp

#include "dtype.hpp"
#include "mcmc.hpp"
#include "pytrack_pos.hpp"

PyTrackPos::PyTrackPos(lint freq) : PyTrack(freq) {}

void PyTrackPos::track(lint time, const NucPosMCMC& model) {
  data.push_back(model.getNucPos());
}
