// pytrack_pos.hpp
// Track the nucleosome positions

#ifndef PYTRACK_POS_HPP
#define PYTRACK_POS_HPP

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pytrack.hpp"

namespace py = pybind11;

class PyTrackPos : public PyTrack<PyTrackPos,std::vector<int> > {
public:
  PyTrackPos(int printFreq);
  void track(int time, const NucPosModel& model);
};

#endif
