// tracker.hpp

#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <optional>
#include <string>
#include "dtype.hpp"

class NucPosModel;

class Tracker {

protected:
  lint printFreq;
  std::optional<lint> prevTime;

public:
  Tracker(lint printFreq);
  virtual ~Tracker() = default;
  virtual void initialize(lint time, const NucPosModel& model) = 0;
  virtual void update(lint time, const NucPosModel& model) = 0;
  virtual void finalize(lint time, const NucPosModel& model) = 0;
  void track(lint time, const NucPosModel& model);
  void reset();
};

#endif
