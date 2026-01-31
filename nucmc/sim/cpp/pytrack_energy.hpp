// pytrack_energy.hpp
// Track the system energy

#ifndef PYTRACK_ENERGY_HPP
#define PYTRACK_ENERGY_HPP

#include <vector>
#include "pytrack.hpp"

class PyTrackEnergy : public PyTrack<PyTrackEnergy,double> {
public:
  PyTrackEnergy(int printFreq);
  void track(int time, const NucPosModel& model);
};

#endif
