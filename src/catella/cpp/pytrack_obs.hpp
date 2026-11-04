// pytrack_obs.hpp
// Track the selected observables (time, energy, temperature, positions)

#ifndef PYTRACK_OBS_HPP
#define PYTRACK_OBS_HPP

#include <vector>
#include "dtype.hpp"
#include "pytrack.hpp"

class PyTrackObs : public PyTrackBase {

public:

  enum class OutputType : uint {
    Energy = 1 << 0,
    Position = 1 << 1,
    Temp = 1 << 2,
    All = Energy | Position | Temp
  };

  PyTrackObs(lint printFreq, OutputType otype);

  void initialize(lint time, const NucPosMCMC& model) override final;
  void update(lint time, const NucPosMCMC& model) override final;
  void finalize(lint time, const NucPosMCMC& model) override final;
  void reset() override;
  py::object values() override;

private:

  OutputType outType;
  std::vector<lint> times;
  std::vector<double> energy;
  std::vector<double> temp;
  // Positions of all frames concatenated; frame i is
  // posFlat[posOffset[i]:posOffset[i+1]]
  std::vector<int> posFlat;
  std::vector<ulint> posOffset;

  bool has(OutputType otype) const;

};

#endif
