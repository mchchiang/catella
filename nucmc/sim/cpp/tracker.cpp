// tracker.cpp

#include "tracker.hpp"

Tracker::Tracker(int freq) : printFreq(freq) {}

void Tracker::track(int time, const NucPosModel& model) {
  if (time % printFreq == 0) {
    update(time, model);
  }
}
