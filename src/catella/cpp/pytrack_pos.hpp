// pytrack_pos.hpp
// Track the nucleosome positions

#ifndef PYTRACK_POS_HPP
#define PYTRACK_POS_HPP

#include <vector>
#include "dtype.hpp"
#include "pytrack.hpp"

class PyTrackPos : public PyTrack<PyTrackPos,std::vector<int> > {
public:
  PyTrackPos(lint printFreq);
  void track(lint time, const NucPosModel& model);
};

#endif
