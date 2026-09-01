// pytrack_pos.cpp

#include "dtype.hpp"
#include "model.hpp"
#include "pytrack_pos.hpp"

PyTrackPos::PyTrackPos(lint freq) : PyTrack(freq) {}

void PyTrackPos::track(lint time, const NucPosModel& model) {
  data.push_back(model.getNucPos());
}
