// pytrack_pos.cpp

#include "pytrack_pos.hpp"

PyTrackPos::PyTrackPos(int freq) : PyTrack(freq) {}

void PyTrackPos::track(int time, const NucPosModel& model) {
  data.push_back(model.getNucPos());
}
