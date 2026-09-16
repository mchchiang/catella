// tracker.cpp

#include <optional>
#include "dtype.hpp"
#include "tracker.hpp"
#include "mcmc.hpp"

Tracker::Tracker(lint freq) : printFreq(freq) {}

void Tracker::track(lint time, const NucPosMCMC& model) {
  if (time % printFreq == 0 && (!prevTime || time != *prevTime)) {
    update(time, model);
    prevTime = time;
  }
}

void Tracker::reset() {
  prevTime.reset();
}
