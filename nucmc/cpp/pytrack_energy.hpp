// pytrack_energy.hpp
// Track the system energy

#ifndef PYTRACK_ENERGY_HPP
#define PYTRACK_ENERGY_HPP

#include <vector>
#include "dtype.hpp"
#include "pytrack.hpp"

class PyTrackEnergy : public PyTrack<PyTrackEnergy,double> {
public:
  PyTrackEnergy(lint printFreq);
  void track(lint time, const NucPosModel& model);
};

#endif
