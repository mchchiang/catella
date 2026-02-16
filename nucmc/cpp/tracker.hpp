// tracker.hpp

#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <string>
#include "dtype.hpp"

class NucPosModel;

class Tracker {

protected:
  lint printFreq;

public:
  Tracker(lint printFreq);
  virtual ~Tracker() = default;
  virtual void initialize(lint time, const NucPosModel& model) = 0;
  virtual void update(lint time, const NucPosModel& model) = 0;
  virtual void finalize(lint time, const NucPosModel& model) = 0;
  void track(lint time, const NucPosModel& model);
};

#endif
