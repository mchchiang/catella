// tracker.hpp

#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <optional>
#include <string>
#include "dtype.hpp"

class NucPosMCMC;

class Tracker {

protected:
  lint printFreq;
  std::optional<lint> prevTime;

public:
  Tracker(lint printFreq);
  virtual ~Tracker() = default;
  virtual void initialize(lint time, const NucPosMCMC& model) = 0;
  virtual void update(lint time, const NucPosMCMC& model) = 0;
  virtual void finalize(lint time, const NucPosMCMC& model) = 0;
  void track(lint time, const NucPosMCMC& model);
  void reset();
};

#endif
