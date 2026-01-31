// tracker.hpp

#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <string>
#include "model.hpp"

class NucPosModel;

class Tracker {

protected:
  int printFreq;

public:
  Tracker(int printFreq);
  virtual ~Tracker() {};
  virtual void update(int time, const NucPosModel& model) = 0;
  void track(int time, const NucPosModel& model);
};

#endif
